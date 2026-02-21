"""
Novel Overfitting Attack - 3 Fundamentally Different Approaches

Current gap: LOO 0.003743 -> LB 0.006619 (1.77x overfit ratio).
The entire gap is overfitting. These 3 approaches attack it structurally
in ways never tried before in this project.

1. RANDOM PROJECTION ENSEMBLE: Project 213 features to k random dims,
   run per-player Ridge with LOO via hat matrix, average 200+ projections.
   Reduces effective feature dimension while preserving distance (JL lemma).

2. CROSS-FITTED BLEND WEIGHT CALIBRATION: Split training 50/50, get
   debiased OOF predictions, use those to select honest blend weights
   and hyperparameters.

3. TEST-TIME AUGMENTATION (TTA): Add calibrated noise to test features,
   re-predict using precomputed training models, average.
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
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
TARGETS = ["angle", "depth", "left_right"]
TARGET_IDX = {"angle": 0, "depth": 1, "left_right": 2}


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


# ==============================================================
# FEATURE EXTRACTION (from per_example_pipeline.py)
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
    FEET_TO_METERS = 0.3048
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
    lw_idx = kp_index.get('left_wrist')
    if lw_idx is not None and rw is not None:
        feats.append(ts_hr[f, lw_idx, 1] - ts_hr[f, rw, 1])
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
    all_feats, release_frames = [], []
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


# ==============================================================
# FAST per-example locally weighted Ridge
# ==============================================================

def _weighted_ridge_predict(X_tr_s, y_tr, x_query, weights, alpha):
    """Weighted Ridge with proper intercept handling via centering."""
    w_sum = weights.sum()
    X_mean = (weights[:, None] * X_tr_s).sum(axis=0) / w_sum
    y_mean = (weights * y_tr).sum() / w_sum
    X_c = X_tr_s - X_mean
    y_c = y_tr - y_mean
    W_sqrt = np.sqrt(weights)
    Xw = X_c * W_sqrt[:, None]
    yw = y_c * W_sqrt
    d = X_tr_s.shape[1]
    XtX = Xw.T @ Xw + alpha * np.eye(d)
    Xty = Xw.T @ yw
    beta = np.linalg.solve(XtX, Xty)
    return (x_query - X_mean) @ beta + y_mean


def fast_lw_ridge(X_train, y_train, X_test, pids_train, pids_test,
                  bw=0.5, alpha=10.0):
    """Per-example locally weighted Ridge with proper intercept."""
    unique_pids = sorted(np.unique(pids_train))
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask] if np.any(te_mask) else np.zeros((0, X_train.shape[1]))
        n_tr = len(X_tr)
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr = cdist(X_tr_s, X_tr_s)
        dists_upper = D_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(dists_upper, bw) if len(dists_upper) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # LOO predictions
        for i in range(n_tr):
            w = np.exp(-D_tr[i] ** 2 / (2 * sigma ** 2))
            w[i] = 0
            if w.sum() < 1e-10:
                oof[tr_idx[i]] = np.mean(y_tr)
                continue
            oof[tr_idx[i]] = _weighted_ridge_predict(
                X_tr_s, y_tr, X_tr_s[i], w, alpha)

        # Test predictions
        if len(X_te) > 0:
            D_te = cdist(X_te_s, X_tr_s)
            for j in range(len(X_te)):
                w = np.exp(-D_te[j] ** 2 / (2 * sigma ** 2))
                if w.sum() < 1e-10:
                    test_preds[te_idx[j]] = np.mean(y_tr)
                    continue
                test_preds[te_idx[j]] = _weighted_ridge_predict(
                    X_tr_s, y_tr, X_te_s[j], w, alpha)

    return oof, test_preds


def fast_lw_ridge_test_only(X_train, y_train, X_test_noisy, pids_train, pids_test,
                            precomputed, bw=0.5, alpha=10.0):
    """Fast test-only prediction reusing precomputed training data."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test_noisy))

    for pid in unique_pids:
        te_mask = pids_test == pid
        if not np.any(te_mask):
            continue
        te_idx = np.where(te_mask)[0]

        pc = precomputed[pid]
        X_tr_s = pc['X_tr_s']
        y_tr = pc['y_tr']
        sigma = pc['sigma']
        scaler = pc['scaler']

        X_te_s = scaler.transform(X_test_noisy[te_mask])
        D_te = cdist(X_te_s, X_tr_s)

        for j in range(len(X_te_s)):
            w = np.exp(-D_te[j] ** 2 / (2 * sigma ** 2))
            if w.sum() < 1e-10:
                test_preds[te_idx[j]] = np.mean(y_tr)
                continue
            test_preds[te_idx[j]] = _weighted_ridge_predict(
                X_tr_s, y_tr, X_te_s[j], w, alpha)

    return test_preds


def precompute_training(X_train, y_train, pids_train, bw=0.5):
    """Precompute training-side data for fast TTA."""
    unique_pids = sorted(np.unique(pids_train))
    precomputed = {}
    for pid in unique_pids:
        tr_mask = pids_train == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        n_tr = len(X_tr)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        D_tr = cdist(X_tr_s, X_tr_s)
        dists_upper = D_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(dists_upper, bw) if len(dists_upper) > 0 else 1.0
        sigma = max(sigma, 1e-6)
        precomputed[pid] = {
            'X_tr_s': X_tr_s, 'y_tr': y_tr, 'sigma': sigma, 'scaler': scaler
        }
    return precomputed


# ==============================================================
# APPROACH 1: RANDOM PROJECTION ENSEMBLE
# ==============================================================

def random_projection_ensemble(X_train, y_train, X_test, pids_train, pids_test,
                                n_projections=200, proj_dim=20,
                                bw=0.5, alpha=10.0, seed=42):
    """
    Project 213 features to proj_dim random linear combinations, run per-example
    Ridge on the projected features, average over many projections.

    Key: each projection uses only proj_dim features for Ridge (low overfit),
    but averaging over projections recovers full 213-dim information.
    """
    rng = np.random.RandomState(seed)
    n_features = X_train.shape[1]
    unique_pids = sorted(np.unique(pids_train))

    all_oof = np.zeros((n_projections, len(X_train)))
    all_test = np.zeros((n_projections, len(X_test)))

    for p_idx in range(n_projections):
        # Gaussian random projection (preserves distances by JL lemma)
        R = rng.randn(n_features, proj_dim).astype(np.float32) / np.sqrt(proj_dim)
        X_tr_proj = X_train @ R
        X_te_proj = X_test @ R

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            X_tr = X_tr_proj[tr_mask]
            y_tr = y_train[tr_mask]
            n_tr = len(X_tr)
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)

            D_tr = cdist(X_tr_s, X_tr_s)
            dists_upper = D_tr[np.triu_indices(n_tr, k=1)]
            sigma = np.quantile(dists_upper, bw) if len(dists_upper) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            # LOO
            for i in range(n_tr):
                w = np.exp(-D_tr[i] ** 2 / (2 * sigma ** 2))
                w[i] = 0
                if w.sum() < 1e-10:
                    all_oof[p_idx, tr_idx[i]] = np.mean(y_tr)
                    continue
                all_oof[p_idx, tr_idx[i]] = _weighted_ridge_predict(
                    X_tr_s, y_tr, X_tr_s[i], w, alpha)

            # Test
            if np.any(te_mask):
                X_te = X_te_proj[te_mask]
                X_te_s = scaler.transform(X_te)
                D_te = cdist(X_te_s, X_tr_s)
                for j in range(len(X_te)):
                    w = np.exp(-D_te[j] ** 2 / (2 * sigma ** 2))
                    if w.sum() < 1e-10:
                        all_test[p_idx, te_idx[j]] = np.mean(y_tr)
                        continue
                    all_test[p_idx, te_idx[j]] = _weighted_ridge_predict(
                        X_tr_s, y_tr, X_te_s[j], w, alpha)

        if (p_idx + 1) % 50 == 0:
            avg_oof = np.mean(all_oof[:p_idx+1], axis=0)
            mse = np.mean((avg_oof - y_train) ** 2)
            print(f"      {p_idx+1}/{n_projections}: running LOO MSE={mse:.6f}")

    final_oof = np.mean(all_oof, axis=0)
    final_test = np.mean(all_test, axis=0)
    return final_oof, final_test


# ==============================================================
# APPROACH 2: CROSS-FITTED CALIBRATION
# ==============================================================

def cross_fitted_calibration(X_train, y_train, X_test, pids_train, pids_test,
                             target_name):
    """
    Split training 50/50 by player (stratified), train on each half,
    predict the other. This gives honest OOF predictions.
    Compare these to LOO to measure true overfit ratio.
    Select best config by cross-fitted MSE.
    """
    unique_pids = sorted(np.unique(pids_train))

    # Stratified 2-fold split (within each player)
    folds = [[], []]
    for pid in unique_pids:
        pid_idx = np.where(pids_train == pid)[0]
        rng = np.random.RandomState(42)
        rng.shuffle(pid_idx)
        half = len(pid_idx) // 2
        folds[0].extend(pid_idx[:half].tolist())
        folds[1].extend(pid_idx[half:].tolist())
    folds[0] = np.array(sorted(folds[0]))
    folds[1] = np.array(sorted(folds[1]))

    configs = [
        (0.50, 10.0),
        (0.50, 50.0),
        (0.50, 100.0),
        (0.50, 200.0),
        (0.50, 500.0),
        (0.70, 50.0),
        (0.70, 100.0),
        (0.70, 200.0),
        (0.80, 50.0),
        (0.80, 100.0),
        (0.80, 200.0),
        (0.80, 500.0),
        (0.90, 200.0),
        (0.90, 500.0),
        (0.95, 500.0),
        (0.95, 1000.0),
    ]

    cf_results = {}
    print(f"    Cross-fitting {len(configs)} configs (2-fold within player)...")

    for bw, alp in configs:
        cf_oof = np.zeros(len(X_train))

        for fold_id in range(2):
            val_idx = folds[fold_id]
            tr_idx = folds[1 - fold_id]

            X_tr_fold = X_train[tr_idx]
            y_tr_fold = y_train[tr_idx]
            pids_tr_fold = pids_train[tr_idx]
            X_val_fold = X_train[val_idx]
            pids_val_fold = pids_train[val_idx]

            for pid in unique_pids:
                tr_pid = pids_tr_fold == pid
                val_pid = pids_val_fold == pid
                if not np.any(val_pid):
                    continue

                X_tr_p = X_tr_fold[tr_pid]
                y_tr_p = y_tr_fold[tr_pid]
                X_val_p = X_val_fold[val_pid]
                n_tr_p = len(X_tr_p)

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_p)
                X_val_s = scaler.transform(X_val_p)

                D_tr = cdist(X_tr_s, X_tr_s)
                dists_upper = D_tr[np.triu_indices(n_tr_p, k=1)]
                sigma = np.quantile(dists_upper, bw) if len(dists_upper) > 0 else 1.0
                sigma = max(sigma, 1e-6)

                D_val_tr = cdist(X_val_s, X_tr_s)
                val_global = val_idx[val_pid]

                for j in range(len(X_val_p)):
                    w = np.exp(-D_val_tr[j] ** 2 / (2 * sigma ** 2))
                    if w.sum() < 1e-10:
                        cf_oof[val_global[j]] = np.mean(y_tr_p)
                        continue
                    cf_oof[val_global[j]] = _weighted_ridge_predict(
                        X_tr_s, y_tr_p, X_val_s[j], w, alp)

        cf_mse = np.mean((cf_oof - y_train) ** 2)
        label = f"bw{bw:.2f}_a{int(alp)}"
        cf_results[label] = {'oof': cf_oof, 'mse': cf_mse, 'bw': bw, 'alpha': alp}

    # Also get LOO for same configs
    print("    Computing LOO for comparison...")
    loo_results = {}
    for bw, alp in configs:
        oof, _ = fast_lw_ridge(X_train, y_train, X_test, pids_train, pids_test,
                               bw=bw, alpha=alp)
        loo_mse = np.mean((oof - y_train) ** 2)
        label = f"bw{bw:.2f}_a{int(alp)}"
        loo_results[label] = {'mse': loo_mse}

    # Print comparison
    print(f"\n    {'Config':<20s} {'LOO MSE':>10s} {'CF MSE':>10s} {'Ratio':>8s}")
    print(f"    {'-'*48}")
    for label in sorted(cf_results.keys()):
        loo_m = loo_results[label]['mse']
        cf_m = cf_results[label]['mse']
        ratio = cf_m / loo_m if loo_m > 0 else 0
        best_marker = ""
        print(f"    {label:<20s} {loo_m:>10.6f} {cf_m:>10.6f} {ratio:>7.2f}x {best_marker}")

    # Best by CF
    best_cf_label = min(cf_results, key=lambda x: cf_results[x]['mse'])
    best_cf = cf_results[best_cf_label]
    # Best by LOO
    best_loo_label = min(loo_results, key=lambda x: loo_results[x]['mse'])
    print(f"\n    Best by CF:  {best_cf_label} (CF-MSE={best_cf['mse']:.6f})")
    print(f"    Best by LOO: {best_loo_label} (LOO-MSE={loo_results[best_loo_label]['mse']:.6f})")

    # Generate test predictions for best CF config
    _, test_best_cf = fast_lw_ridge(
        X_train, y_train, X_test, pids_train, pids_test,
        bw=best_cf['bw'], alpha=best_cf['alpha'])

    # Average top-3 CF configs
    sorted_cf = sorted(cf_results.items(), key=lambda x: x[1]['mse'])
    top3 = [c[0] for c in sorted_cf[:3]]
    top3_test = []
    for label in top3:
        cfg = cf_results[label]
        _, tp = fast_lw_ridge(X_train, y_train, X_test, pids_train, pids_test,
                              bw=cfg['bw'], alpha=cfg['alpha'])
        top3_test.append(tp)
    test_top3_avg = np.mean(top3_test, axis=0)

    # Average top-5 CF configs
    top5 = [c[0] for c in sorted_cf[:5]]
    top5_test = []
    for label in top5:
        cfg = cf_results[label]
        _, tp = fast_lw_ridge(X_train, y_train, X_test, pids_train, pids_test,
                              bw=cfg['bw'], alpha=cfg['alpha'])
        top5_test.append(tp)
    test_top5_avg = np.mean(top5_test, axis=0)

    print(f"    Top-3 CF configs: {top3}")
    print(f"    Top-5 CF configs: {top5}")

    return {
        'test_best_cf': test_best_cf,
        'test_top3_avg': test_top3_avg,
        'test_top5_avg': test_top5_avg,
        'best_cf_label': best_cf_label,
        'best_cf_mse': best_cf['mse'],
        'best_cf_bw': best_cf['bw'],
        'best_cf_alpha': best_cf['alpha'],
        'top3': top3,
        'top5': top5,
        'cf_results': cf_results,
        'loo_results': loo_results,
    }


# ==============================================================
# APPROACH 3: TEST-TIME AUGMENTATION
# ==============================================================

def test_time_augmentation(X_train, y_train, X_test, pids_train, pids_test,
                           n_aug=50, noise_scales=[0.01, 0.02, 0.05, 0.10],
                           bw=0.5, alpha=10.0, seed=42):
    """
    Add calibrated Gaussian noise to test features, predict, average.
    Training side is fixed - only test features are perturbed.
    """
    rng = np.random.RandomState(seed)
    feature_stds = np.std(X_train, axis=0)
    feature_stds = np.maximum(feature_stds, 1e-8)

    # Precompute training side
    pc = precompute_training(X_train, y_train, pids_train, bw=bw)

    # Baseline prediction
    test_base = fast_lw_ridge_test_only(
        X_train, y_train, X_test, pids_train, pids_test, pc, bw=bw, alpha=alpha)

    results = {}

    for ns in noise_scales:
        all_test = [test_base.copy()]

        for aug_i in range(n_aug):
            noise = rng.randn(*X_test.shape).astype(np.float32) * feature_stds * ns
            X_test_noisy = X_test + noise
            test_aug = fast_lw_ridge_test_only(
                X_train, y_train, X_test_noisy, pids_train, pids_test,
                pc, bw=bw, alpha=alpha)
            all_test.append(test_aug)

        avg_test = np.mean(all_test, axis=0)
        r_base = np.corrcoef(avg_test, test_base)[0, 1]
        max_delta = np.max(np.abs(avg_test - test_base))
        print(f"    ns={ns:.2f}: r(base)={r_base:.6f}, max_delta={max_delta:.6f}")

        results[ns] = {'test': avg_test, 'r_base': r_base}

    return results, test_base


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("NOVEL OVERFITTING ATTACK")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scalers, y_scaled = {}, {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled[target] = scalers[target].transform(
            y_train[:, TARGET_IDX[target]].reshape(-1, 1)).ravel()

    sub_1828 = pd.read_csv(SUBMISSION_DIR / "submission_1828.csv")

    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        print("  Extracting features...")
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)
        y_raw = y_train[:, TARGET_IDX[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train, X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        n_feat = X_train_aug.shape[1]
        print(f"  Features: {n_feat}")

        y_target = y_scaled[target]
        sub_1828_vals = sub_1828[f'scaled_{target}'].values

        # Baseline
        oof_base, test_base = fast_lw_ridge(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)
        base_mse = np.mean((oof_base - y_target) ** 2)
        base_r = np.corrcoef(test_base, sub_1828_vals)[0, 1]
        print(f"  Baseline: LOO MSE={base_mse:.6f}, r(1828)={base_r:.4f}")

        target_results = {
            'baseline_mse': base_mse, 'baseline_r': base_r,
            'test_base': test_base
        }

        # ----- APPROACH 1: Random Projection Ensemble -----
        print(f"\n  [1] RANDOM PROJECTION ENSEMBLE")
        rp_results = {}
        for proj_dim, n_proj, alp in [
            (20, 200, 10.0),
            (20, 200, 50.0),
            (15, 200, 10.0),
            (30, 200, 10.0),
        ]:
            label = f"d{proj_dim}_n{n_proj}_a{int(alp)}"
            print(f"    Config: proj_dim={proj_dim}, n_proj={n_proj}, alpha={alp}")
            oof_rp, test_rp = random_projection_ensemble(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                n_projections=n_proj, proj_dim=proj_dim, bw=0.5, alpha=alp)
            rp_mse = np.mean((oof_rp - y_target) ** 2)
            rp_r = np.corrcoef(test_rp, sub_1828_vals)[0, 1]
            print(f"    -> LOO MSE={rp_mse:.6f}, r(1828)={rp_r:.4f}")
            rp_results[label] = {'oof': oof_rp, 'test': test_rp, 'mse': rp_mse, 'r': rp_r}
        target_results['rp'] = rp_results

        # ----- APPROACH 2: Cross-Fitted Calibration -----
        print(f"\n  [2] CROSS-FITTED CALIBRATION")
        cf_result = cross_fitted_calibration(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test, target)
        cf_r_best = np.corrcoef(cf_result['test_best_cf'], sub_1828_vals)[0, 1]
        cf_r_top3 = np.corrcoef(cf_result['test_top3_avg'], sub_1828_vals)[0, 1]
        cf_r_top5 = np.corrcoef(cf_result['test_top5_avg'], sub_1828_vals)[0, 1]
        print(f"    Best CF test: r(1828)={cf_r_best:.4f}")
        print(f"    Top-3 avg:    r(1828)={cf_r_top3:.4f}")
        print(f"    Top-5 avg:    r(1828)={cf_r_top5:.4f}")
        target_results['cf'] = cf_result
        target_results['cf_r'] = {
            'best': cf_r_best, 'top3': cf_r_top3, 'top5': cf_r_top5
        }

        # ----- APPROACH 3: TTA -----
        print(f"\n  [3] TEST-TIME AUGMENTATION")
        tta_results, tta_base = test_time_augmentation(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            n_aug=50, noise_scales=[0.01, 0.02, 0.05, 0.10])
        for ns, res in tta_results.items():
            r = np.corrcoef(res['test'], sub_1828_vals)[0, 1]
            res['r_1828'] = r
            print(f"    ns={ns}: r(1828)={r:.4f}")
        target_results['tta'] = tta_results
        target_results['tta_base'] = tta_base

        all_results[target] = target_results

    # ==============================================================
    # GENERATE SUBMISSIONS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    def make_submission(preds_dict, desc, weights=None):
        """Create submission blending with Sub 1828 at various weights."""
        if weights is None:
            weights = [0.20, 0.50, 1.00]
        subs = []
        for w in weights:
            sub_num = get_next_submission_number()
            blended = sub_1828.copy()
            for target in TARGETS:
                col = f"scaled_{target}"
                blended[col] = (1 - w) * sub_1828[col] + w * preds_dict[target]
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            avg_r = np.mean([np.corrcoef(preds_dict[t], sub_1828[f'scaled_{t}'].values)[0, 1]
                            for t in TARGETS])
            print(f"  Sub {sub_num}: {desc} w={w:.2f} (avg_r={avg_r:.4f})")
            subs.append(sub_num)
        return subs

    # 1. Random Projection configs
    print("\n  --- Random Projection Ensemble ---")
    for rp_label in ["d20_n200_a10", "d20_n200_a50", "d15_n200_a10", "d30_n200_a10"]:
        preds = {t: all_results[t]['rp'][rp_label]['test'] for t in TARGETS}
        make_submission(preds, f"RP {rp_label}")

    # 2. Cross-fitted configs
    print("\n  --- Cross-Fitted Calibration ---")
    preds_cf_best = {t: all_results[t]['cf']['test_best_cf'] for t in TARGETS}
    make_submission(preds_cf_best, "CF-best")

    preds_cf_top3 = {t: all_results[t]['cf']['test_top3_avg'] for t in TARGETS}
    make_submission(preds_cf_top3, "CF-top3avg")

    preds_cf_top5 = {t: all_results[t]['cf']['test_top5_avg'] for t in TARGETS}
    make_submission(preds_cf_top5, "CF-top5avg")

    # 3. TTA
    print("\n  --- Test-Time Augmentation ---")
    for ns in [0.01, 0.02, 0.05]:
        preds_tta = {t: all_results[t]['tta'][ns]['test'] for t in TARGETS}
        make_submission(preds_tta, f"TTA ns={ns}", weights=[0.50, 1.00])

    # 4. Best RP per target (pick lowest LOO MSE)
    print("\n  --- Per-Target Best RP ---")
    preds_best_rp = {}
    for target in TARGETS:
        rp = all_results[target]['rp']
        best = min(rp, key=lambda x: rp[x]['mse'])
        preds_best_rp[target] = rp[best]['test']
        print(f"    {target}: {best} (LOO={rp[best]['mse']:.6f}, r={rp[best]['r']:.4f})")
    make_submission(preds_best_rp, "per-target best RP")

    # 5. Mega-blend: average RP + CF + TTA per target
    print("\n  --- Mega-Blend ---")
    for target in TARGETS:
        rp = all_results[target]['rp']
        best_rp_label = min(rp, key=lambda x: rp[x]['mse'])
        components = [
            rp[best_rp_label]['test'],
            all_results[target]['cf']['test_top3_avg'],
        ]
        if 0.02 in all_results[target]['tta']:
            components.append(all_results[target]['tta'][0.02]['test'])
        preds_best_rp[target] = np.mean(components, axis=0)
    make_submission(preds_best_rp, "mega-blend (RP+CF+TTA)", weights=[0.20, 0.50])

    # 6. Per-target most diverse (lowest r with Sub 1828)
    print("\n  --- Per-Target Most Diverse ---")
    preds_diverse = {}
    for target in TARGETS:
        rp = all_results[target]['rp']
        most_diverse = min(rp, key=lambda x: rp[x]['r'])
        preds_diverse[target] = rp[most_diverse]['test']
        print(f"    {target}: {most_diverse} (r={rp[most_diverse]['r']:.4f})")
    make_submission(preds_diverse, "per-target most diverse", weights=[0.10, 0.20, 0.30])

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
