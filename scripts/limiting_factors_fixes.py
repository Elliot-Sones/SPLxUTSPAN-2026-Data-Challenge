"""
Five Limiting Factor Fixes for Sub 1350

Addresses identified root causes of the CV-LB gap:
1. Hyperparameter averaging: average across configs instead of picking best
2. Feature selection: LASSO stability selection to trim 213 -> ~80 features
3. Target transformation: quantile transform to linearize relationships
4. Player 5 wider kernel: wider bandwidth for the smallest player group
5. Submission ensembling: average top LB-validated submissions

Each fix is tested independently vs the Sub 1350 baseline.
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
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.model_selection import KFold
from scipy.stats import rankdata

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
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


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


# ================================================================
# BASELINE: Sub 1350 method (locally weighted Ridge)
# ================================================================

def baseline_locally_weighted(X_train, y_train, X_test, pids_train, pids_test,
                               bandwidth_quantile=0.5, alpha=10.0):
    """Original Sub 1350 method."""
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
        sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

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


# ================================================================
# FIX 1: Hyperparameter Averaging
# ================================================================

def fix1_hyperparameter_averaging(X_train, y_train, X_test, pids_train, pids_test):
    """Average predictions across multiple (bandwidth, alpha) configs
    instead of picking the single 'best'. Reduces selection bias."""
    configs = [
        (0.3, 5.0), (0.3, 10.0), (0.3, 20.0),
        (0.4, 5.0), (0.4, 10.0), (0.4, 20.0),
        (0.5, 5.0), (0.5, 10.0), (0.5, 20.0),
    ]

    all_oof = []
    all_test = []
    for bw, alpha in configs:
        oof, test = baseline_locally_weighted(
            X_train, y_train, X_test, pids_train, pids_test,
            bandwidth_quantile=bw, alpha=alpha)
        all_oof.append(oof)
        all_test.append(test)

    # Simple average
    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)

    # Also try trimmed average (drop best and worst per sample)
    all_oof_arr = np.array(all_oof)  # (9, n_train)
    all_test_arr = np.array(all_test)  # (9, n_test)

    # Trimmed mean: for each sample, drop highest and lowest prediction
    trimmed_oof = np.zeros(len(X_train))
    trimmed_test = np.zeros(len(X_test))
    for i in range(len(X_train)):
        vals = all_oof_arr[:, i]
        trimmed_oof[i] = np.mean(np.sort(vals)[1:-1])  # drop min and max
    for i in range(len(X_test)):
        vals = all_test_arr[:, i]
        trimmed_test[i] = np.mean(np.sort(vals)[1:-1])

    return {
        'avg': (avg_oof, avg_test),
        'trimmed': (trimmed_oof, trimmed_test),
        'individual_oof': all_oof,
        'configs': configs,
    }


# ================================================================
# FIX 2: Feature Selection via Stability Selection
# ================================================================

def fix2_feature_selection(X_train, y_train, X_test, pids_train, pids_test,
                           max_features=80):
    """Use LASSO stability selection to identify most stable features,
    then run locally weighted Ridge on reduced set."""
    unique_pids = sorted(np.unique(pids_train))

    # Per-player stability selection
    selected_features = np.zeros(X_train.shape[1])

    for pid in unique_pids:
        tr_mask = pids_train == pid
        X_p = X_train[tr_mask]
        y_p = y_train[tr_mask]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        n_bootstrap = 30
        for b in range(n_bootstrap):
            rng = np.random.RandomState(b)
            idx = rng.choice(len(X_s), len(X_s), replace=True)
            lasso = Lasso(alpha=0.01, max_iter=5000)
            lasso.fit(X_s[idx], y_p[idx])
            selected_features += (np.abs(lasso.coef_) > 1e-6).astype(float)

    # Normalize by total bootstraps
    stability_scores = selected_features / (n_bootstrap * len(unique_pids))

    # Select top features by stability
    n_select = min(max_features, (stability_scores > 0.3).sum())
    n_select = max(n_select, 30)  # at least 30 features
    top_idx = np.argsort(-stability_scores)[:n_select]

    print(f"    Selected {n_select} features (stability > {stability_scores[top_idx[-1]]:.3f})")

    # Run baseline with reduced features
    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    oof, test = baseline_locally_weighted(
        X_train_sel, y_train, X_test_sel, pids_train, pids_test,
        bandwidth_quantile=0.5, alpha=10.0)

    return oof, test, top_idx, stability_scores


# ================================================================
# FIX 3: Target Transformation
# ================================================================

def fix3_target_transform(X_train, y_train, X_test, pids_train, pids_test):
    """Apply rank/quantile transform to targets before fitting,
    then inverse transform predictions."""

    # Rank transform: map targets to uniform [0,1]
    n = len(y_train)
    ranks = rankdata(y_train) / (n + 1)  # avoid 0 and 1

    # Fit on rank-transformed targets
    oof_rank, test_rank = baseline_locally_weighted(
        X_train, ranks, X_test, pids_train, pids_test,
        bandwidth_quantile=0.5, alpha=10.0)

    # Inverse transform: map rank predictions back to target space
    # Use interpolation: for each predicted rank, find corresponding target value
    sorted_idx = np.argsort(y_train)
    sorted_y = y_train[sorted_idx]
    sorted_ranks = ranks[sorted_idx]

    oof_inv = np.interp(oof_rank, sorted_ranks, sorted_y)
    test_inv = np.interp(test_rank, sorted_ranks, sorted_y)

    # Also try: quantile Gaussian transform
    qt = QuantileTransformer(n_quantiles=min(100, n), output_distribution='normal',
                              random_state=42)
    y_qt = qt.fit_transform(y_train.reshape(-1, 1)).ravel()

    oof_qt, test_qt = baseline_locally_weighted(
        X_train, y_qt, X_test, pids_train, pids_test,
        bandwidth_quantile=0.5, alpha=10.0)

    oof_qt_inv = qt.inverse_transform(oof_qt.reshape(-1, 1)).ravel()
    test_qt_inv = qt.inverse_transform(test_qt.reshape(-1, 1)).ravel()

    return {
        'rank': (oof_inv, test_inv),
        'quantile_gaussian': (oof_qt_inv, test_qt_inv),
    }


# ================================================================
# FIX 4: Player 5 Wider Kernel
# ================================================================

def fix4_player5_handling(X_train, y_train, X_test, pids_train, pids_test):
    """Use wider bandwidth for Player 5 (smallest group, ~74 samples).
    Also try cross-player pooling for Player 5."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    # Count per player
    for pid in unique_pids:
        n = (pids_train == pid).sum()
        print(f"    Player {pid}: {n} train shots")

    # Method A: wider bandwidth for small players
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

        # Adaptive bandwidth per player: wider for smaller groups
        if n_tr < 80:
            bw = 0.7  # wider for small groups (Player 5)
            alpha = 20.0  # more regularization
        else:
            bw = 0.5
            alpha = 10.0

        sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

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

    oof_A = oof_preds.copy()
    test_A = test_preds.copy()

    # Method B: For Player 5, pool with all players but add player indicator
    test_preds_B = np.zeros(len(X_test))
    oof_preds_B = np.zeros(len(X_train))

    # For non-P5 players, use standard method
    p5_id = None
    for pid in unique_pids:
        n = (pids_train == pid).sum()
        if n < 80:
            p5_id = pid
            break

    if p5_id is None:
        return {'wider_bw': (oof_A, test_A), 'cross_player': (oof_A, test_A)}

    for pid in unique_pids:
        if pid == p5_id:
            continue
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
        sigma = np.quantile(all_dists, 0.5) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds_B[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds_B[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                test_preds_B[te_indices[j]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds_B[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    # For Player 5: use ALL training data with player-ID feature
    tr_mask_p5 = pids_train == p5_id
    te_mask_p5 = pids_test == p5_id
    tr_indices_p5 = np.where(tr_mask_p5)[0]
    te_indices_p5 = np.where(te_mask_p5)[0]

    # Add player indicator as extra feature
    pid_feat_train = (pids_train == p5_id).astype(float).reshape(-1, 1)
    pid_feat_test = (pids_test == p5_id).astype(float).reshape(-1, 1)
    X_train_ext = np.hstack([X_train, pid_feat_train])
    X_test_ext = np.hstack([X_test, pid_feat_test])

    scaler = StandardScaler()
    X_tr_s_all = scaler.fit_transform(X_train_ext)
    X_te_s_all = scaler.transform(X_test_ext)

    # LOO for Player 5 using all data
    D_all = cdist(X_tr_s_all, X_tr_s_all, metric='euclidean')
    all_dists = D_all[np.triu_indices(len(X_train), k=1)]
    sigma_all = np.quantile(all_dists, 0.5)
    sigma_all = max(sigma_all, 1e-6)

    for i in tr_indices_p5:
        dists = D_all[i, :]
        weights = np.exp(-dists ** 2 / (2 * sigma_all ** 2))
        weights[i] = 0
        if weights.sum() < 1e-10:
            oof_preds_B[i] = np.mean(y_train)
            continue
        ridge = Ridge(alpha=20.0)
        ridge.fit(X_tr_s_all, y_train, sample_weight=weights)
        oof_preds_B[i] = ridge.predict(X_tr_s_all[i:i+1])[0]

    D_te_all = cdist(X_te_s_all[te_mask_p5], X_tr_s_all, metric='euclidean')
    for j_local, j in enumerate(te_indices_p5):
        dists = D_te_all[j_local, :]
        weights = np.exp(-dists ** 2 / (2 * sigma_all ** 2))
        if weights.sum() < 1e-10:
            test_preds_B[j] = np.mean(y_train)
            continue
        ridge = Ridge(alpha=20.0)
        ridge.fit(X_tr_s_all, y_train, sample_weight=weights)
        test_preds_B[j] = ridge.predict(X_te_s_all[j:j+1])[0]

    return {'wider_bw': (oof_A, test_A), 'cross_player': (oof_preds_B, test_preds_B)}


# ================================================================
# FIX 5: Submission Ensembling
# ================================================================

def fix5_submission_ensemble():
    """Average top LB-validated submissions."""
    sub_ids = [1350, 1354, 1421, 1430]
    subs = []
    for sid in sub_ids:
        df = pd.read_csv(SUBMISSION_DIR / f"submission_{sid}.csv")
        subs.append(df)

    # Simple average
    avg = subs[0].copy()
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        avg[col] = np.mean([s[col].values for s in subs], axis=0)

    # Weighted average (weight by 1/LB score)
    lb_scores = [0.006776, 0.006782, 0.006789, 0.006782]
    weights = 1.0 / np.array(lb_scores)
    weights /= weights.sum()

    wavg = subs[0].copy()
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        wavg[col] = np.average([s[col].values for s in subs], axis=0, weights=weights)

    # Median
    med = subs[0].copy()
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        med[col] = np.median([s[col].values for s in subs], axis=0)

    return {'avg': avg, 'weighted_avg': wavg, 'median': med}


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("LIMITING FACTOR FIXES")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # ============================================================
    # FIX 5: Submission ensemble (no model training needed)
    # ============================================================
    print("\n" + "=" * 70)
    print("FIX 5: SUBMISSION ENSEMBLING")
    print("=" * 70)

    ensembles = fix5_submission_ensemble()
    for name, df in ensembles.items():
        sub_num = get_next_submission_number()
        df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {name} ensemble of [1350, 1354, 1421, 1430]")

    # ============================================================
    # PER-TARGET FIXES
    # ============================================================

    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)

        # Augment with PLS
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # --- BASELINE ---
        print("  [BASELINE] Sub 1350 method (bw=0.5, alpha=10)...")
        oof_base, test_base = baseline_locally_weighted(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"    LOO MSE: {mse_base:.6f}")

        results = {'baseline': {'oof': oof_base, 'test': test_base, 'mse': mse_base}}

        # --- FIX 1: Hyperparameter Averaging ---
        print("  [FIX 1] Hyperparameter averaging (9 configs)...")
        fix1 = fix1_hyperparameter_averaging(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)

        for variant in ['avg', 'trimmed']:
            oof, test = fix1[variant]
            mse = np.mean((oof - y_target) ** 2)
            delta = (mse - mse_base) / mse_base * 100
            print(f"    {variant}: LOO MSE={mse:.6f} ({delta:+.1f}%)")
            results[f'fix1_{variant}'] = {'oof': oof, 'test': test, 'mse': mse}

        # Also log individual config results
        for i, (bw, alpha) in enumerate(fix1['configs']):
            mse_i = np.mean((fix1['individual_oof'][i] - y_target) ** 2)
            print(f"      bw={bw}, alpha={alpha}: LOO MSE={mse_i:.6f}")

        # --- FIX 2: Feature Selection ---
        print("  [FIX 2] Stability selection (LASSO)...")
        oof_fs, test_fs, selected_idx, scores = fix2_feature_selection(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)
        mse_fs = np.mean((oof_fs - y_target) ** 2)
        delta = (mse_fs - mse_base) / mse_base * 100
        print(f"    LOO MSE: {mse_fs:.6f} ({delta:+.1f}%)")
        results['fix2'] = {'oof': oof_fs, 'test': test_fs, 'mse': mse_fs}

        # --- FIX 3: Target Transform ---
        print("  [FIX 3] Target transformation...")
        fix3 = fix3_target_transform(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)

        for variant in ['rank', 'quantile_gaussian']:
            oof, test = fix3[variant]
            mse = np.mean((oof - y_target) ** 2)
            delta = (mse - mse_base) / mse_base * 100
            print(f"    {variant}: LOO MSE={mse:.6f} ({delta:+.1f}%)")
            results[f'fix3_{variant}'] = {'oof': oof, 'test': test, 'mse': mse}

        # --- FIX 4: Player 5 Handling ---
        print("  [FIX 4] Player 5 wider kernel / cross-player...")
        fix4 = fix4_player5_handling(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)

        for variant in ['wider_bw', 'cross_player']:
            oof, test = fix4[variant]
            mse = np.mean((oof - y_target) ** 2)
            delta = (mse - mse_base) / mse_base * 100
            print(f"    {variant}: LOO MSE={mse:.6f} ({delta:+.1f}%)")

            # Also per-player breakdown
            for pid in sorted(np.unique(pids_train)):
                mask = pids_train == pid
                mse_p = np.mean((oof[mask] - y_target[mask]) ** 2)
                mse_b = np.mean((oof_base[mask] - y_target[mask]) ** 2)
                n_p = mask.sum()
                d = (mse_p - mse_b) / mse_b * 100 if mse_b > 0 else 0
                print(f"      P{pid} (n={n_p}): MSE={mse_p:.6f} vs base={mse_b:.6f} ({d:+.1f}%)")

            results[f'fix4_{variant}'] = {'oof': oof, 'test': test, 'mse': mse}

        # --- SUMMARY ---
        print(f"\n  {target.upper()} SUMMARY:")
        for name, r in sorted(results.items(), key=lambda x: x[1]['mse']):
            delta = (r['mse'] - mse_base) / mse_base * 100
            marker = " <-- BEST" if r['mse'] == min(v['mse'] for v in results.values()) else ""
            print(f"    {name:30s}: LOO MSE={r['mse']:.6f} ({delta:+.1f}%){marker}")

        all_results[target] = results

    # ============================================================
    # COMBINED RESULTS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("COMBINED RESULTS (mean across targets)")
    print(f"{'=' * 70}")

    # Get all method names that appear in all targets
    all_methods = set(all_results['angle'].keys())
    for target in TARGETS:
        all_methods &= set(all_results[target].keys())

    method_scores = {}
    for method in sorted(all_methods):
        mses = [all_results[t][method]['mse'] for t in TARGETS]
        mean_mse = np.mean(mses)
        method_scores[method] = mean_mse

    base_mean = method_scores['baseline']
    for method, score in sorted(method_scores.items(), key=lambda x: x[1]):
        delta = (score - base_mean) / base_mean * 100
        print(f"  {method:30s}: {score:.6f} ({delta:+.1f}%)")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # For each fix method, generate standalone + blended submissions
    best_methods = []
    for method in sorted(all_methods):
        if method == 'baseline':
            continue
        best_methods.append(method)

    for method in best_methods:
        # Standalone
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': all_results['angle'][method]['test'],
            'scaled_depth': all_results['depth'][method]['test'],
            'scaled_left_right': all_results['left_right'][method]['test'],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        mean_mse = method_scores[method]
        delta = (mean_mse - base_mean) / base_mean * 100
        print(f"  Sub {sub_num}: {method} standalone (LOO mean={mean_mse:.6f}, {delta:+.1f}%)")

        # Correlation with Sub 1350
        sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            r = np.corrcoef(sub_1350[col].values, sub[col].values)[0, 1]
            print(f"    {col}: r={r:.4f} with Sub 1350")

        # Blended with Sub 784 at standard weights
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = sub_784['scaled_angle']  # aw=0
        blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * all_results['depth'][method]['test']
        blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * all_results['left_right'][method]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {method} blend (aw=0, dw=0.30, lw=0.50)")

    # Best combined: pick best method per target
    print(f"\n  Best per-target combination:")
    best_per_target = {}
    for target in TARGETS:
        best_method = min(all_results[target].keys(),
                         key=lambda m: all_results[target][m]['mse'])
        best_per_target[target] = best_method
        print(f"    {target}: {best_method} (MSE={all_results[target][best_method]['mse']:.6f})")

    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': all_results['angle'][best_per_target['angle']]['test'],
        'scaled_depth': all_results['depth'][best_per_target['depth']]['test'],
        'scaled_left_right': all_results['left_right'][best_per_target['left_right']]['test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: best-per-target combination")

    # Blend best-per-target with Sub 784
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    blended['scaled_angle'] = sub_784['scaled_angle']
    blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * all_results['depth'][best_per_target['depth']]['test']
    blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * all_results['left_right'][best_per_target['left_right']]['test']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: best-per-target blend (aw=0, dw=0.30, lw=0.50)")

    # Blend best-per-target with Sub 1350 (10%)
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
    for pct in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for col, tgt in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            method = best_per_target[tgt]
            blended[col] = (1-pct) * sub_1350[col] + pct * all_results[tgt][method]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(pct*100)}% best-per-target + {int((1-pct)*100)}% Sub 1350")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
