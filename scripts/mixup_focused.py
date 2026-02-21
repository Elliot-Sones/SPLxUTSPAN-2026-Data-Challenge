"""
Focused Mixup Augmentation: Best config only (alpha=1.0, 2x, full weight)
Updated to use bandwidth_quantile=0.3 (matching current best pipeline)
Generates standalone submission + blends with Sub 2169.
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# PIPELINE PARAMS - matching current best
BANDWIDTH_QUANTILE = 0.3
RIDGE_ALPHA = 10.0

# MIXUP PARAMS - best config from search
MIXUP_ALPHA = 1.0
MIXUP_MULTIPLIER = 2
SYN_WEIGHT_DISCOUNT = 1.0


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split('_')[1]) for fp in existing
                    if fp.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        X_raw = np.zeros((n, n_kp_cols * 240), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


def generate_mixup_samples(X_3d, X_raw, y, pids, alpha=1.0, n_multiplier=2, seed=42):
    rng = np.random.RandomState(seed)
    unique_pids = sorted(np.unique(pids))

    syn_X_3d_list, syn_X_raw_list, syn_y_list = [], [], []
    syn_pids_list, syn_parents_list = [], []

    for pid in unique_pids:
        mask = pids == pid
        indices = np.where(mask)[0]
        n_player = len(indices)
        n_synthetic = n_player * n_multiplier

        for _ in range(n_synthetic):
            i, j = rng.choice(n_player, size=2, replace=False)
            idx_a, idx_b = indices[i], indices[j]
            lam = rng.beta(alpha, alpha)

            syn_X_3d_list.append(lam * X_3d[idx_a] + (1 - lam) * X_3d[idx_b])
            syn_X_raw_list.append(lam * X_raw[idx_a] + (1 - lam) * X_raw[idx_b])
            syn_y_list.append(lam * y[idx_a] + (1 - lam) * y[idx_b])
            syn_pids_list.append(pid)
            syn_parents_list.append((idx_a, idx_b))

    return (np.array(syn_X_3d_list, dtype=np.float32),
            np.array(syn_X_raw_list, dtype=np.float32),
            np.array(syn_y_list, dtype=np.float32),
            np.array(syn_pids_list),
            syn_parents_list)


def compute_hoop_transform(ts_3d, kp_index):
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0

    forward = HOOP_POS[:2] - player_pos[:2]
    fn = np.linalg.norm(forward)
    if fn > 1e-6:
        forward /= fn
    else:
        forward = np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])

    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]

    centered = ts_3d - player_pos.reshape(1, 1, 3)
    return np.einsum('ij,fkj->fki', R, centered)


def detect_release_frame(ts_3d, kp_index):
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 120
    wrist_traj = ts_3d[:, rw_idx, :].copy()
    for ax in range(3):
        vals = wrist_traj[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 120
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, ax] = vals

    wrist_z_smooth = safe_savgol(wrist_traj[:, 2], 11, 3)
    wrist_peak = 80 + np.argmax(wrist_z_smooth[80:200])

    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = [ts_3d[:, kp_index[k], :] for k in ft_keys if k in kp_index]
    if ft_trajs:
        ft_center = np.nanmean(ft_trajs, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    vel = np.zeros_like(ball * FEET_TO_METERS)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_METERS, 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)

    s, e = max(80, wrist_peak - 40), min(wrist_peak + 5, 200)
    return int(np.clip(s + np.argmax(speed[s:e]), 80, 200))


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    f = int(np.clip(frame, 0, 239))
    feats = []

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])

    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 9)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            feats.append(np.nanmean(series))
            feats.append(np.nanstd(series))
            feats.append(np.nanmax(series) - np.nanmin(series))

    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')
    if all(i is not None for i in [rw, re, rs]):
        feats.append(ts_hr[f, rw, 0] - ts_hr[f, rs, 0])
        feats.append(ts_hr[f, rw, 1] - ts_hr[f, rs, 1])
        feats.append(ts_hr[f, rw, 2] - ts_hr[f, rs, 2])
        ua = ts_3d[f, re] - ts_3d[f, rs]
        fa = ts_3d[f, rw] - ts_3d[f, re]
        ua_n, fa_n = np.linalg.norm(ua), np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            feats.append(np.degrees(np.arccos(np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1))))
        else:
            feats.append(90.0)
        for coord in range(3):
            vel = np.gradient(ts_hr[:, rw, coord], DT)
            feats.append(vel[f])
    else:
        feats.extend([0.0] * 7)

    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    ls = kp_index.get('left_shoulder')
    if rh is not None and lh is not None:
        feats.append(ts_hr[f, rh, 1] - ts_hr[f, lh, 1])
        feats.append(ts_hr[f, rh, 0] - ts_hr[f, lh, 0])
    else:
        feats.extend([0.0, 0.0])
    if rs is not None and ls is not None:
        feats.append(ts_hr[f, rs, 1] - ts_hr[f, ls, 1])
    else:
        feats.append(0.0)

    lw = kp_index.get('left_wrist')
    if lw is not None and rw is not None:
        feats.append(ts_hr[f, lw, 1] - ts_hr[f, rw, 1])
    else:
        feats.append(0.0)

    feats.append(release_frame)

    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats.append(np.nanmean(series[140:180]))
            feats.append(np.nanmax(vel[140:180]))
    else:
        feats.extend([0.0] * 6)

    return np.array(feats, dtype=np.float32)


def extract_all_features(X_3d, pids, kp_index, kp_names, target):
    frame = TARGET_FRAMES[target]
    n = len(pids)
    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = X_3d[i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


def augment_with_pls_mixed(X_orig, y_raw_orig, pids_orig, X_syn, pids_syn,
                           X_test, pids_test,
                           X_raw_orig, X_raw_syn, X_raw_test):
    unique_pids = sorted(np.unique(pids_orig))
    max_nc = 15
    pls_orig = np.zeros((len(pids_orig), max_nc), dtype=np.float32)
    pls_syn = np.zeros((len(pids_syn), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        orig_mask = pids_orig == pid
        syn_mask = pids_syn == pid
        te_mask = pids_test == pid
        n_p = orig_mask.sum()

        scaler = StandardScaler()
        raw_orig = scaler.fit_transform(X_raw_orig[orig_mask])
        raw_syn = scaler.transform(X_raw_syn[syn_mask]) if syn_mask.sum() > 0 else np.zeros((0, raw_orig.shape[1]))
        raw_te = scaler.transform(X_raw_test[te_mask])

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15]:
            if c > nc:
                break
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(raw_orig):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_orig[tr_idx], y_raw_orig[orig_mask][tr_idx])
                pred = pls.predict(raw_orig[val_idx]).flatten()
                mses.append(np.mean((pred - y_raw_orig[orig_mask][val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_orig, y_raw_orig[orig_mask])
        pls_orig[orig_mask, :best_nc] = pls.transform(raw_orig)
        if syn_mask.sum() > 0:
            pls_syn[syn_mask, :best_nc] = pls.transform(raw_syn)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)

    return (np.hstack([X_orig, pls_orig]),
            np.hstack([X_syn, pls_syn]),
            np.hstack([X_test, pls_test]))


def locally_weighted_prediction_augmented(
        X_orig, y_orig, pids_orig,
        X_syn, y_syn, pids_syn, syn_parents,
        X_test, pids_test,
        bandwidth_quantile=0.3, alpha=10.0, syn_weight_discount=1.0):
    unique_pids = sorted(np.unique(pids_orig))
    n_orig = len(X_orig)
    n_syn = len(X_syn)
    n_test = len(X_test)

    X_all = np.vstack([X_orig, X_syn])
    y_all = np.concatenate([y_orig, y_syn])
    pids_all = np.concatenate([pids_orig, pids_syn])
    is_synthetic = np.concatenate([np.zeros(n_orig, dtype=bool),
                                   np.ones(n_syn, dtype=bool)])

    parent_of = {}
    for s_i, (pa, pb) in enumerate(syn_parents):
        parent_of[n_orig + s_i] = {pa, pb}

    oof_preds = np.zeros(n_orig)
    test_preds = np.zeros(n_test)

    for pid in unique_pids:
        all_mask = pids_all == pid
        orig_mask = (pids_orig == pid)
        te_mask = (pids_test == pid)

        X_all_p = X_all[all_mask]
        y_all_p = y_all[all_mask]
        is_syn_p = is_synthetic[all_mask]
        all_indices = np.where(all_mask)[0]
        orig_indices = np.where(orig_mask)[0]
        te_indices = np.where(te_mask)[0]

        n_all_p = len(X_all_p)

        orig_in_player = ~is_syn_p
        scaler = StandardScaler()
        scaler.fit(X_all_p[orig_in_player])
        X_all_p_s = scaler.transform(X_all_p)
        X_te_p_s = scaler.transform(X_test[te_mask]) if te_mask.sum() > 0 else np.zeros((0, X_all_p.shape[1]))

        D_all = cdist(X_all_p_s, X_all_p_s, metric='euclidean')
        D_te = cdist(X_te_p_s, X_all_p_s, metric='euclidean') if te_mask.sum() > 0 else np.zeros((0, n_all_p))

        orig_local_idx = np.where(orig_in_player)[0]
        D_orig_orig = D_all[np.ix_(orig_local_idx, orig_local_idx)]
        n_orig_p = len(orig_local_idx)
        all_dists = D_orig_orig[np.triu_indices(n_orig_p, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        for i_local, orig_idx in enumerate(orig_local_idx):
            orig_global = all_indices[orig_idx]
            exclude_local = {orig_idx}

            for j_local in range(n_all_p):
                j_global = all_indices[j_local]
                if j_global in parent_of:
                    if orig_global in parent_of[j_global]:
                        exclude_local.add(j_local)

            dists = D_all[orig_idx, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            for ex in exclude_local:
                weights[ex] = 0

            if syn_weight_discount < 1.0:
                syn_local_mask = is_syn_p.copy()
                weights[syn_local_mask] *= syn_weight_discount

            if weights.sum() < 1e-10:
                oof_preds[orig_indices[i_local]] = np.mean(y_all_p[orig_in_player])
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_all_p_s, y_all_p, sample_weight=weights)
            oof_preds[orig_indices[i_local]] = ridge.predict(X_all_p_s[orig_idx:orig_idx+1])[0]

        for j in range(te_mask.sum()):
            dists = D_te[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            if syn_weight_discount < 1.0:
                weights[is_syn_p] *= syn_weight_discount

            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_all_p[orig_in_player])
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_all_p_s, y_all_p, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_p_s[j:j+1])[0]

    return oof_preds, test_preds


def locally_weighted_prediction_baseline(X_train, y_train, X_test, pids_train, pids_test,
                                         bandwidth_quantile=0.3, alpha=10.0):
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0

            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test,
                     X_raw_train, X_raw_test):
    unique_pids = sorted(np.unique(pids_train))
    max_nc = 15
    pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        n_p = tr_mask.sum()

        scaler = StandardScaler()
        raw_tr = scaler.fit_transform(X_raw_train[tr_mask])
        raw_te = scaler.transform(X_raw_test[te_mask])

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15]:
            if c > nc:
                break
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(raw_tr):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_tr[tr_idx], y_raw_train[tr_mask][tr_idx])
                pred = pls.predict(raw_tr[val_idx]).flatten()
                mses.append(np.mean((pred - y_raw_train[tr_mask][val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_tr, y_raw_train[tr_mask])
        pls_train[tr_mask, :best_nc] = pls.transform(raw_tr)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)

    return np.hstack([X_train, pls_train]), np.hstack([X_test, pls_test])


def main():
    t0 = time.time()
    print("=" * 70)
    print("FOCUSED MIXUP AUGMENTATION")
    print(f"  Config: alpha={MIXUP_ALPHA}, {MIXUP_MULTIPLIER}x, discount={SYN_WEIGHT_DISCOUNT}")
    print(f"  Pipeline: bw_q={BANDWIDTH_QUANTILE}, Ridge alpha={RIDGE_ALPHA}")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    kp_names = train_data['kp_names']

    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    scalers = {}
    y_scaled = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # ===================== BASELINE =====================
    print("\n--- BASELINE (no augmentation, bw_q=0.3) ---")
    baseline_results = {}
    for target in TARGETS:
        print(f"  {target}: extracting features (frame {TARGET_FRAMES[target]})...")
        X_train_hc, _ = extract_all_features(
            train_data['X_3d'], pids_train, kp_index, kp_names, target)
        X_test_hc, _ = extract_all_features(
            test_data['X_3d'], pids_test, kp_index, kp_names, target)

        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        y_target = y_scaled[target]
        oof_base, test_base = locally_weighted_prediction_baseline(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=BANDWIDTH_QUANTILE, alpha=RIDGE_ALPHA)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"  {target} baseline LOO MSE: {mse_base:.6f}")

        baseline_results[target] = {
            'oof': oof_base, 'test': test_base, 'mse': mse_base,
            'X_train_hc': X_train_hc, 'X_test_hc': X_test_hc,
        }

    base_mean = np.mean([baseline_results[t]['mse'] for t in TARGETS])
    print(f"\n  BASELINE MEAN LOO MSE: {base_mean:.6f}")

    # ===================== MIXUP =====================
    print(f"\n--- MIXUP: alpha={MIXUP_ALPHA}, {MIXUP_MULTIPLIER}x, discount={SYN_WEIGHT_DISCOUNT} ---")

    # Use seed=44 to match original best config (config index 2 -> seed=42+2=44)
    syn_X_3d, syn_X_raw, syn_y, syn_pids, syn_parents = generate_mixup_samples(
        train_data['X_3d'], train_data['X_raw'], y_train, pids_train,
        alpha=MIXUP_ALPHA, n_multiplier=MIXUP_MULTIPLIER, seed=44)

    n_syn = len(syn_pids)
    print(f"  Generated {n_syn} synthetic samples ({n_syn/345:.1f}x original)")

    mixup_results = {}
    for target in TARGETS:
        print(f"\n  TARGET: {target}")
        print(f"    Extracting features from {n_syn} synthetic shots...")
        X_syn_hc, _ = extract_all_features(
            syn_X_3d, syn_pids, kp_index, kp_names, target)

        y_raw = y_train[:, target_idx[target]]

        X_orig_aug, X_syn_aug, X_test_aug = augment_with_pls_mixed(
            baseline_results[target]['X_train_hc'], y_raw, pids_train,
            X_syn_hc, syn_pids,
            baseline_results[target]['X_test_hc'], pids_test,
            train_data['X_raw'], syn_X_raw, test_data['X_raw'])

        y_target = y_scaled[target]
        y_syn_scaled = scalers[target].transform(
            syn_y[:, target_idx[target]].reshape(-1, 1)).ravel()

        print(f"    Running augmented locally weighted Ridge (bw_q={BANDWIDTH_QUANTILE})...")
        oof_aug, test_aug = locally_weighted_prediction_augmented(
            X_orig_aug, y_target, pids_train,
            X_syn_aug, y_syn_scaled, syn_pids, syn_parents,
            X_test_aug, pids_test,
            bandwidth_quantile=BANDWIDTH_QUANTILE, alpha=RIDGE_ALPHA,
            syn_weight_discount=SYN_WEIGHT_DISCOUNT)

        mse_aug = np.mean((oof_aug - y_target) ** 2)
        mse_base = baseline_results[target]['mse']
        delta_pct = (mse_aug - mse_base) / mse_base * 100

        print(f"    {target}: LOO MSE={mse_aug:.6f} (baseline={mse_base:.6f}, delta={delta_pct:+.2f}%)")

        mixup_results[target] = {
            'oof': oof_aug, 'test': test_aug,
            'mse': mse_aug, 'delta_pct': delta_pct,
        }

    mixup_mean = np.mean([mixup_results[t]['mse'] for t in TARGETS])
    mixup_delta = np.mean([mixup_results[t]['delta_pct'] for t in TARGETS])
    print(f"\n  MIXUP MEAN LOO MSE: {mixup_mean:.6f} (delta={mixup_delta:+.2f}%)")

    # ===================== GENERATE SUBMISSIONS =====================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Load Sub 2169 for blending
    sub_2169 = pd.read_csv(SUBMISSION_DIR / "submission_2169.csv")

    # Standalone mixup submission
    sub_num = get_next_submission_number()
    sub_standalone = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': mixup_results['angle']['test'],
        'scaled_depth': mixup_results['depth']['test'],
        'scaled_left_right': mixup_results['left_right']['test'],
    })
    sub_standalone.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    standalone_num = sub_num
    print(f"  Sub {sub_num}: MIXUP standalone (alpha={MIXUP_ALPHA}, {MIXUP_MULTIPLIER}x, bw_q={BANDWIDTH_QUANTILE})")

    # Blends with Sub 2169 at 10%, 20%, 30%
    blend_nums = {}
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_2169.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_2169[col] + w * mixup_results[target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        blend_nums[w] = sub_num

        # Correlation between mixup and Sub 2169
        r_vals = []
        for target in TARGETS:
            col = f'scaled_{target}'
            r = np.corrcoef(sub_2169[col].values, mixup_results[target]['test'])[0, 1]
            r_vals.append(r)
        mean_r = np.mean(r_vals)
        print(f"  Sub {sub_num}: {w*100:.0f}% mixup + {(1-w)*100:.0f}% Sub 2169 (mean r with 2169: {mean_r:.4f})")

    # Baseline (no mixup) blended with Sub 2169 for comparison
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_2169.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_2169[col] + w * baseline_results[target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {w*100:.0f}% baseline (no mixup) + {(1-w)*100:.0f}% Sub 2169")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Pipeline: bw_q={BANDWIDTH_QUANTILE}, Ridge alpha={RIDGE_ALPHA}")
    print(f"  Mixup: alpha={MIXUP_ALPHA}, {MIXUP_MULTIPLIER}x, discount={SYN_WEIGHT_DISCOUNT}, seed=44")
    print(f"  Baseline mean LOO MSE: {base_mean:.6f}")
    print(f"  Mixup mean LOO MSE:    {mixup_mean:.6f} ({mixup_delta:+.2f}%)")
    print()
    print("  Per-target:")
    for t in TARGETS:
        b = baseline_results[t]['mse']
        m = mixup_results[t]['mse']
        d = mixup_results[t]['delta_pct']
        print(f"    {t}: baseline={b:.6f}, mixup={m:.6f} ({d:+.2f}%)")
    print()
    print(f"  Standalone: Sub {standalone_num}")
    for w, sn in blend_nums.items():
        print(f"  {w*100:.0f}% mixup + {(1-w)*100:.0f}% Sub 2169: Sub {sn}")
    print(f"\n  Total time: {elapsed:.1f}s")

    return baseline_results, mixup_results


if __name__ == "__main__":
    baseline_results, mixup_results = main()
