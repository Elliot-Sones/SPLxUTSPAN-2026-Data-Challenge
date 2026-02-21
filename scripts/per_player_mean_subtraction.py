"""
Per-Player Mean Subtraction Pipeline

Instead of predicting raw target values, subtract per-player target means
from training targets, train models on RESIDUALS, then add means back.
This reduces the model's burden of learning player-level baselines.

Phases:
1. Measure per-player target statistics
2. Naive mean subtraction
3. Shrunk mean subtraction (James-Stein style)
4. LOPO-honest mean computation (leakage-free LOO)
5. Per-player standardization (subtract mean + divide by std)
6. Diversity analysis vs Sub 2063
7. Generate submissions blended with Sub 2063
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


# ==============================================================
# DATA LOADING (copied from per_example_pipeline.py)
# ==============================================================

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


# ==============================================================
# FEATURE EXTRACTION (copied from per_example_pipeline.py)
# ==============================================================

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


def extract_all_features(data, target):
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test, X_raw_train, X_raw_test):
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


# ==============================================================
# CORE: LOCALLY WEIGHTED PREDICTION WITH MEAN SUBTRACTION
# ==============================================================

def locally_weighted_prediction_baseline(X_train, y_train, pids_train, pids_test,
                                         X_test, bandwidth_quantile=0.3, alpha=10.0):
    """Baseline: standard locally weighted ridge (no mean subtraction).
    Returns LOO predictions for train and predictions for test."""
    unique_pids = sorted(np.unique(pids_train))
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

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
        sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # LOO
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

        # Test
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


def locally_weighted_prediction_mean_sub(X_train, y_train, pids_train, pids_test,
                                          X_test, bandwidth_quantile=0.3, alpha=10.0,
                                          shrinkage=1.0, standardize=False,
                                          lopo_honest=False):
    """Locally weighted ridge with per-player mean subtraction.

    Args:
        shrinkage: 1.0 = full player mean, 0.0 = global mean only.
            mean_used = shrinkage * player_mean + (1 - shrinkage) * global_mean
        standardize: if True, also divide by per-player std (and reverse on prediction)
        lopo_honest: if True, compute per-player means in LOO leaving out the current sample
    """
    unique_pids = sorted(np.unique(pids_train))
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    global_mean = np.mean(y_train)
    global_std = np.std(y_train)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        # Per-player statistics (full training set for this player)
        player_mean = np.mean(y_tr)
        player_std = np.std(y_tr)
        if player_std < 1e-8:
            player_std = global_std  # fallback

        # Shrunk mean for this player
        shrunk_mean = shrinkage * player_mean + (1 - shrinkage) * global_mean

        # Transform targets: subtract mean, optionally divide by std
        if standardize:
            y_tr_transformed = (y_tr - shrunk_mean) / player_std
        else:
            y_tr_transformed = y_tr - shrunk_mean

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # LOO
        for i in range(n_tr):
            if lopo_honest:
                # Recompute mean excluding sample i
                y_tr_excl = np.delete(y_tr, i)
                honest_player_mean = np.mean(y_tr_excl)
                honest_shrunk_mean = shrinkage * honest_player_mean + (1 - shrinkage) * global_mean
                if standardize:
                    honest_player_std = np.std(y_tr_excl)
                    if honest_player_std < 1e-8:
                        honest_player_std = global_std
                    # Retransform ALL targets using the honest mean
                    y_tr_honest = (y_tr - honest_shrunk_mean) / honest_player_std
                else:
                    y_tr_honest = y_tr - honest_shrunk_mean
                    honest_player_std = player_std
                y_local = y_tr_honest
                local_mean = honest_shrunk_mean
                local_std = honest_player_std
            else:
                y_local = y_tr_transformed
                local_mean = shrunk_mean
                local_std = player_std

            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_local, sample_weight=weights)
            pred_residual = ridge.predict(X_tr_s[i:i+1])[0]

            # Reverse transform
            if standardize:
                oof_preds[tr_indices[i]] = pred_residual * local_std + local_mean
            else:
                oof_preds[tr_indices[i]] = pred_residual + local_mean

        # Test predictions: use full player mean (no leakage concern for test)
        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr_transformed, sample_weight=weights)
            pred_residual = ridge.predict(X_te_s[j:j+1])[0]

            if standardize:
                test_preds[te_indices[j]] = pred_residual * player_std + shrunk_mean
            else:
                test_preds[te_indices[j]] = pred_residual + shrunk_mean

    return oof_preds, test_preds


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("PER-PLAYER MEAN SUBTRACTION PIPELINE")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    # Load scalers
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # Load Sub 2063 for diversity analysis
    sub_2063 = pd.read_csv(SUBMISSION_DIR / "submission_2063.csv")

    # ==============================================================
    # PHASE 1: Per-Player Target Statistics
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: PER-PLAYER TARGET STATISTICS")
    print(f"{'=' * 70}")

    unique_pids = sorted(np.unique(pids_train))

    for target in TARGETS:
        y_t = y_scaled[target]
        global_mean = np.mean(y_t)
        global_std = np.std(y_t)
        print(f"\n  {target.upper()} (scaled space):")
        print(f"    Global mean={global_mean:.6f}, std={global_std:.6f}")
        for pid in unique_pids:
            mask = pids_train == pid
            pm = np.mean(y_t[mask])
            ps = np.std(y_t[mask])
            n = mask.sum()
            print(f"    Player {pid}: n={n}, mean={pm:.6f}, std={ps:.6f}, "
                  f"mean_delta={pm - global_mean:+.6f}")

    # ==============================================================
    # PER-TARGET PIPELINE
    # ==============================================================

    # Store all results for each phase x target
    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)
        print(f"  Features: {X_train_hc.shape[1]}")

        # Augment with PLS
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Augmented features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        target_results = {}

        # --- PHASE 2: Baseline (no mean subtraction) ---
        print("\n  PHASE 2: Baseline (no mean subtraction)...")
        oof_base, test_base = locally_weighted_prediction_baseline(
            X_train_aug, y_target, pids_train, pids_test, X_test_aug,
            bandwidth_quantile=0.3, alpha=10.0)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"    Baseline LOO MSE: {mse_base:.6f}")
        target_results['baseline'] = {
            'oof': oof_base, 'test': test_base, 'mse': mse_base}

        # --- PHASE 2: Naive mean subtraction ---
        print("\n  PHASE 2: Naive mean subtraction (shrinkage=1.0)...")
        oof_naive, test_naive = locally_weighted_prediction_mean_sub(
            X_train_aug, y_target, pids_train, pids_test, X_test_aug,
            bandwidth_quantile=0.3, alpha=10.0, shrinkage=1.0)
        mse_naive = np.mean((oof_naive - y_target) ** 2)
        delta_naive = (mse_naive - mse_base) / mse_base * 100
        print(f"    Naive mean-sub LOO MSE: {mse_naive:.6f} (delta: {delta_naive:+.2f}%)")
        target_results['naive_mean_sub'] = {
            'oof': oof_naive, 'test': test_naive, 'mse': mse_naive}

        # --- PHASE 3: Shrunk mean subtraction ---
        print("\n  PHASE 3: Shrunk mean subtraction...")
        best_shrink_mse = mse_naive
        best_shrink_w = 1.0
        shrink_results = {}
        for w in [0.5, 0.7, 0.9, 1.0]:
            oof_s, test_s = locally_weighted_prediction_mean_sub(
                X_train_aug, y_target, pids_train, pids_test, X_test_aug,
                bandwidth_quantile=0.3, alpha=10.0, shrinkage=w)
            mse_s = np.mean((oof_s - y_target) ** 2)
            delta_s = (mse_s - mse_base) / mse_base * 100
            print(f"    shrinkage={w:.1f}: LOO MSE={mse_s:.6f} (delta vs baseline: {delta_s:+.2f}%)")
            shrink_results[w] = {'oof': oof_s, 'test': test_s, 'mse': mse_s}
            if mse_s < best_shrink_mse:
                best_shrink_mse = mse_s
                best_shrink_w = w
        print(f"    Best shrinkage: w={best_shrink_w:.1f} (MSE={best_shrink_mse:.6f})")
        target_results['shrunk_mean_sub'] = shrink_results[best_shrink_w]
        target_results['shrunk_mean_sub']['best_w'] = best_shrink_w

        # --- PHASE 4: LOPO-honest mean computation ---
        print("\n  PHASE 4: LOPO-honest mean (leakage-free LOO)...")
        oof_lopo, test_lopo = locally_weighted_prediction_mean_sub(
            X_train_aug, y_target, pids_train, pids_test, X_test_aug,
            bandwidth_quantile=0.3, alpha=10.0, shrinkage=best_shrink_w,
            lopo_honest=True)
        mse_lopo = np.mean((oof_lopo - y_target) ** 2)
        delta_lopo = (mse_lopo - mse_base) / mse_base * 100
        leakage = (mse_lopo - best_shrink_mse) / best_shrink_mse * 100
        print(f"    LOPO-honest LOO MSE: {mse_lopo:.6f} (delta vs baseline: {delta_lopo:+.2f}%)")
        print(f"    Leakage estimate (LOPO vs non-LOPO): {leakage:+.2f}%")
        target_results['lopo_honest'] = {
            'oof': oof_lopo, 'test': test_lopo, 'mse': mse_lopo}

        # --- PHASE 5: Per-player standardization ---
        print("\n  PHASE 5: Per-player standardization (mean + std)...")
        best_std_mse = float('inf')
        best_std_w = 1.0
        for w in [0.5, 0.7, 0.9, 1.0]:
            oof_std, test_std = locally_weighted_prediction_mean_sub(
                X_train_aug, y_target, pids_train, pids_test, X_test_aug,
                bandwidth_quantile=0.3, alpha=10.0, shrinkage=w,
                standardize=True)
            mse_std = np.mean((oof_std - y_target) ** 2)
            delta_std = (mse_std - mse_base) / mse_base * 100
            print(f"    standardize + shrinkage={w:.1f}: LOO MSE={mse_std:.6f} (delta: {delta_std:+.2f}%)")
            if mse_std < best_std_mse:
                best_std_mse = mse_std
                best_std_w = w
        # Re-run best
        oof_std_best, test_std_best = locally_weighted_prediction_mean_sub(
            X_train_aug, y_target, pids_train, pids_test, X_test_aug,
            bandwidth_quantile=0.3, alpha=10.0, shrinkage=best_std_w,
            standardize=True)
        print(f"    Best standardize: w={best_std_w:.1f} (MSE={best_std_mse:.6f})")
        target_results['standardized'] = {
            'oof': oof_std_best, 'test': test_std_best, 'mse': best_std_mse,
            'best_w': best_std_w}

        # Also try LOPO-honest + standardize at best w
        print("\n  PHASE 5b: LOPO-honest + standardization...")
        oof_lopo_std, test_lopo_std = locally_weighted_prediction_mean_sub(
            X_train_aug, y_target, pids_train, pids_test, X_test_aug,
            bandwidth_quantile=0.3, alpha=10.0, shrinkage=best_std_w,
            standardize=True, lopo_honest=True)
        mse_lopo_std = np.mean((oof_lopo_std - y_target) ** 2)
        delta_lopo_std = (mse_lopo_std - mse_base) / mse_base * 100
        print(f"    LOPO-honest + standardize: LOO MSE={mse_lopo_std:.6f} (delta: {delta_lopo_std:+.2f}%)")
        target_results['lopo_standardized'] = {
            'oof': oof_lopo_std, 'test': test_lopo_std, 'mse': mse_lopo_std}

        # --- SUMMARY for this target ---
        print(f"\n  {target.upper()} SUMMARY:")
        for name, res in target_results.items():
            if name == 'shrunk_mean_sub':
                extra = f" (w={res.get('best_w', '?')})"
            elif name == 'standardized':
                extra = f" (w={res.get('best_w', '?')})"
            else:
                extra = ""
            delta = (res['mse'] - mse_base) / mse_base * 100
            print(f"    {name:25s}: LOO MSE={res['mse']:.6f} (delta: {delta:+.2f}%){extra}")

        all_results[target] = target_results

    # ==============================================================
    # PHASE 6: DIVERSITY ANALYSIS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 6: DIVERSITY ANALYSIS vs Sub 2063")
    print(f"{'=' * 70}")

    # Determine best variant per target (lowest LOO MSE)
    best_variants = {}
    for target in TARGETS:
        best_name = min(all_results[target],
                        key=lambda k: all_results[target][k]['mse'])
        best_variants[target] = best_name
        print(f"  {target}: best variant = {best_name} "
              f"(LOO MSE={all_results[target][best_name]['mse']:.6f})")

    print(f"\n  Correlation with Sub 2063 (per variant):")
    for target in TARGETS:
        col = f'scaled_{target}'
        sub_vals = sub_2063[col].values
        for name, res in all_results[target].items():
            r = np.corrcoef(sub_vals, res['test'])[0, 1]
            print(f"    {target}/{name:25s}: r={r:.4f}")

    # ==============================================================
    # OVERALL LOO RESULTS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("OVERALL LOO RESULTS")
    print(f"{'=' * 70}")

    # Table: variant x target
    variant_names = ['baseline', 'naive_mean_sub', 'shrunk_mean_sub',
                     'lopo_honest', 'standardized', 'lopo_standardized']
    print(f"\n  {'Variant':25s} {'angle':>10s} {'depth':>10s} {'left_right':>10s} {'mean':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for vname in variant_names:
        mses = []
        for target in TARGETS:
            if vname in all_results[target]:
                mses.append(all_results[target][vname]['mse'])
            else:
                mses.append(float('nan'))
        mean_mse = np.nanmean(mses)
        print(f"  {vname:25s} {mses[0]:10.6f} {mses[1]:10.6f} {mses[2]:10.6f} {mean_mse:10.6f}")

    # ==============================================================
    # PHASE 7: GENERATE SUBMISSIONS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 7: GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # For each interesting variant, generate blends with Sub 2063
    variants_to_submit = [
        ('naive_mean_sub', 'Naive mean subtraction'),
        ('shrunk_mean_sub', 'Shrunk mean subtraction'),
        ('lopo_honest', 'LOPO-honest mean subtraction'),
        ('standardized', 'Per-player standardization'),
        ('lopo_standardized', 'LOPO-honest + standardization'),
    ]

    for vname, desc in variants_to_submit:
        # Build full test predictions (best variant per target might differ)
        test_angle = all_results['angle'][vname]['test']
        test_depth = all_results['depth'][vname]['test']
        test_lr = all_results['left_right'][vname]['test']

        # Diversity with Sub 2063
        r_angle = np.corrcoef(sub_2063['scaled_angle'].values, test_angle)[0, 1]
        r_depth = np.corrcoef(sub_2063['scaled_depth'].values, test_depth)[0, 1]
        r_lr = np.corrcoef(sub_2063['scaled_left_right'].values, test_lr)[0, 1]

        print(f"\n  Variant: {desc} ({vname})")
        print(f"    Diversity with Sub 2063: angle r={r_angle:.4f}, depth r={r_depth:.4f}, LR r={r_lr:.4f}")

        # Blend weights
        for w in [0.10, 0.20, 0.30]:
            sub_num = get_next_submission_number()
            blended = sub_2063.copy()
            blended['scaled_angle'] = (1-w) * sub_2063['scaled_angle'].values + w * test_angle
            blended['scaled_depth'] = (1-w) * sub_2063['scaled_depth'].values + w * test_depth
            blended['scaled_left_right'] = (1-w) * sub_2063['scaled_left_right'].values + w * test_lr

            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            a_std = blended['scaled_angle'].std()
            d_mean = blended['scaled_depth'].mean()
            print(f"    Sub {sub_num}: w={w:.2f} blend with Sub 2063 ({desc})")
            print(f"      angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # Also: standalone submission from the overall best config per target
    print(f"\n  Best-per-target standalone:")
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({'id': test_data['ids']})
    for target in TARGETS:
        best_name = best_variants[target]
        col = f'scaled_{target}'
        sub[col] = all_results[target][best_name]['test']
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: STANDALONE best-per-target")
    for target in TARGETS:
        print(f"    {target}: variant={best_variants[target]}, "
              f"LOO MSE={all_results[target][best_variants[target]]['mse']:.6f}")

    # Blend standalone with Sub 2063
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_2063.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            best_name = best_variants[target]
            blended[col] = (1-w) * sub_2063[col].values + w * all_results[target][best_name]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: w={w:.2f} blend best-per-target with Sub 2063")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
