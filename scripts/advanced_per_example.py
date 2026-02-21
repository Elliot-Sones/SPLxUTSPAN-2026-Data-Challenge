"""
Advanced Per-Example Regression Variants

Tests 4 genuinely different per-example regression approaches:
1. Feature-weighted distance (weight by target-specific importance)
2. PCA-subspace distance (PCA for neighbors, full features for prediction)
3. Multi-kernel blending (average predictions from 3 bandwidth values)
4. Huber-weighted per-example regression (robust to outlier shots)

Each approach changes the notion of "similar shots" or prediction method,
producing genuinely different predictions for diversity.
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
from sklearn.decomposition import PCA
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
    lw_kp = kp_index.get('left_wrist')
    if lw_kp is not None and rw is not None:
        feats.append(ts_hr[f, lw_kp, 1] - ts_hr[f, rw, 1])
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
# BASELINE: Standard locally weighted regression (from Sub 1350)
# ==============================================================

def baseline_lw(X_train, y_train, X_test, pids_train, pids_test,
                bandwidth_quantile=0.5, alpha=10.0):
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


# ==============================================================
# VARIANT 1: Feature-weighted distance
# ==============================================================

def feature_weighted_lw(X_train, y_train, X_test, pids_train, pids_test,
                        bandwidth_quantile=0.5, alpha=10.0):
    """
    Use target-specific feature importance to weight distances.
    Features with higher |coefficient| in Ridge get higher weight in distance.
    """
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

        # Learn feature weights from Ridge coefficients
        ridge_global = Ridge(alpha=alpha)
        ridge_global.fit(X_tr_s, y_tr)
        feat_weights = np.abs(ridge_global.coef_)
        feat_weights = feat_weights / (feat_weights.sum() + 1e-10)
        feat_weights = np.sqrt(feat_weights)  # sqrt to moderate the weighting

        # Apply feature weights to compute weighted distances
        X_tr_w = X_tr_s * feat_weights[np.newaxis, :]
        X_te_w = X_te_s * feat_weights[np.newaxis, :] if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr_tr = cdist(X_tr_w, X_tr_w, metric='euclidean')
        D_te_tr = cdist(X_te_w, X_tr_w, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

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


# ==============================================================
# VARIANT 2: PCA-subspace distance
# ==============================================================

def pca_subspace_lw(X_train, y_train, X_test, pids_train, pids_test,
                    bandwidth_quantile=0.5, alpha=10.0, n_components=25):
    """
    Use PCA for distance computation (denoised), full features for prediction.
    Separates 'who are your neighbors' from 'what's the prediction'.
    """
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

        # PCA for distance computation
        nc = min(n_components, n_tr - 1, X_tr.shape[1])
        pca = PCA(n_components=nc)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_te_pca = pca.transform(X_te_s) if len(X_te) > 0 else np.zeros((0, nc))

        D_tr_tr = cdist(X_tr_pca, X_tr_pca, metric='euclidean')
        D_te_tr = cdist(X_te_pca, X_tr_pca, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # Regression on FULL features (not PCA)
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


# ==============================================================
# VARIANT 3: Multi-kernel blending
# ==============================================================

def multi_kernel_lw(X_train, y_train, X_test, pids_train, pids_test,
                    bandwidths=(0.3, 0.5, 0.7), alpha=10.0):
    """
    Blend predictions from multiple bandwidth values.
    Each bandwidth captures different neighborhood sizes.
    """
    all_oof = []
    all_test = []
    for bw in bandwidths:
        oof, test = baseline_lw(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=bw, alpha=alpha)
        all_oof.append(oof)
        all_test.append(test)

    # Simple average across bandwidths
    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    return oof_avg, test_avg


# ==============================================================
# VARIANT 4: Weighted LOO (Huber-style outlier downweighting)
# ==============================================================

def huber_weighted_lw(X_train, y_train, X_test, pids_train, pids_test,
                      bandwidth_quantile=0.5, alpha=10.0, n_iter=3):
    """
    Iteratively reweight training samples based on LOO residuals.
    Shots that are hard to predict get downweighted (Huber-like).
    """
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

        # Iterative reweighting
        sample_quality = np.ones(n_tr)  # Start with uniform quality

        for iteration in range(n_iter):
            # LOO pass to compute residuals
            loo_residuals = np.zeros(n_tr)
            for i in range(n_tr):
                dists = D_tr_tr[i, :]
                weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
                weights[i] = 0
                weights *= sample_quality  # Apply sample quality weights
                if weights.sum() < 1e-10:
                    loo_residuals[i] = 0
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=weights)
                pred = ridge.predict(X_tr_s[i:i+1])[0]
                loo_residuals[i] = abs(pred - y_tr[i])

            # Update sample quality: downweight high-residual shots
            mad = np.median(loo_residuals)
            if mad > 1e-10:
                # Huber-style: shots with residual > 2*MAD get downweighted
                normalized = loo_residuals / (1.4826 * mad)  # Robust scale
                sample_quality = np.where(normalized < 2.0, 1.0, 2.0 / normalized)
            else:
                sample_quality = np.ones(n_tr)

        # Final predictions with quality weights
        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            weights *= sample_quality
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights *= sample_quality
            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ADVANCED PER-EXAMPLE REGRESSION VARIANTS")
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

    # Load Sub 1350 and Sub 1640 for comparison
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
    sub_1640 = pd.read_csv(SUBMISSION_DIR / "submission_1640.csv")

    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)

        # Augment with PLS
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # Baseline
        print("\n  [0] Baseline (bw=0.5, alpha=10)...")
        oof_base, test_base = baseline_lw(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"      LOO MSE: {mse_base:.6f}")

        # Variant 1: Feature-weighted distance
        print("\n  [1] Feature-weighted distance...")
        oof_fw, test_fw = feature_weighted_lw(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)
        mse_fw = np.mean((oof_fw - y_target) ** 2)
        r_fw = np.corrcoef(test_fw, sub_1350[f'scaled_{target}'].values)[0, 1]
        print(f"      LOO MSE: {mse_fw:.6f} ({(mse_fw - mse_base) / mse_base * 100:+.1f}% vs baseline)")
        print(f"      Test r with Sub 1350: {r_fw:.4f}")

        # Variant 2: PCA-subspace distance
        print("\n  [2] PCA-subspace distance...")
        best_nc, best_mse_pca = 25, float('inf')
        for nc in [10, 15, 20, 25, 30]:
            oof_tmp, _ = pca_subspace_lw(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                n_components=nc)
            mse_tmp = np.mean((oof_tmp - y_target) ** 2)
            print(f"      nc={nc}: LOO MSE={mse_tmp:.6f} ({(mse_tmp - mse_base) / mse_base * 100:+.1f}%)")
            if mse_tmp < best_mse_pca:
                best_mse_pca = mse_tmp
                best_nc = nc

        oof_pca, test_pca = pca_subspace_lw(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            n_components=best_nc)
        r_pca = np.corrcoef(test_pca, sub_1350[f'scaled_{target}'].values)[0, 1]
        print(f"      Best: nc={best_nc}, LOO MSE={best_mse_pca:.6f}")
        print(f"      Test r with Sub 1350: {r_pca:.4f}")

        # Variant 3: Multi-kernel blending
        print("\n  [3] Multi-kernel blending...")
        oof_mk, test_mk = multi_kernel_lw(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidths=(0.3, 0.5, 0.7))
        mse_mk = np.mean((oof_mk - y_target) ** 2)
        r_mk = np.corrcoef(test_mk, sub_1350[f'scaled_{target}'].values)[0, 1]
        print(f"      LOO MSE: {mse_mk:.6f} ({(mse_mk - mse_base) / mse_base * 100:+.1f}% vs baseline)")
        print(f"      Test r with Sub 1350: {r_mk:.4f}")

        # Variant 4: Huber-weighted
        print("\n  [4] Huber-weighted (outlier downweighting)...")
        oof_hub, test_hub = huber_weighted_lw(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            n_iter=3)
        mse_hub = np.mean((oof_hub - y_target) ** 2)
        r_hub = np.corrcoef(test_hub, sub_1350[f'scaled_{target}'].values)[0, 1]
        print(f"      LOO MSE: {mse_hub:.6f} ({(mse_hub - mse_base) / mse_base * 100:+.1f}% vs baseline)")
        print(f"      Test r with Sub 1350: {r_hub:.4f}")

        # Summary
        variants = {
            'baseline': (mse_base, oof_base, test_base),
            'feat_weighted': (mse_fw, oof_fw, test_fw),
            'pca_subspace': (best_mse_pca, oof_pca, test_pca),
            'multi_kernel': (mse_mk, oof_mk, test_mk),
            'huber_weighted': (mse_hub, oof_hub, test_hub),
        }

        print(f"\n  {target.upper()} SUMMARY:")
        for name, (mse, _, _) in sorted(variants.items(), key=lambda x: x[1][0]):
            delta = (mse - mse_base) / mse_base * 100
            print(f"    {name}: {mse:.6f} ({delta:+.1f}%)")

        results[target] = variants

    # ==============================================================
    # GENERATE SUBMISSIONS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Find best variant per target
    best_per_target = {}
    for target in TARGETS:
        best_name = min(results[target], key=lambda x: results[target][x][0])
        best_per_target[target] = best_name
        print(f"  {target}: {best_name} (LOO {results[target][best_name][0]:.6f})")

    # Sub: Best-per-target standalone
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle'][best_per_target['angle']][2],
        'scaled_depth': results['depth'][best_per_target['depth']][2],
        'scaled_left_right': results['left_right'][best_per_target['left_right']][2],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: STANDALONE best-per-target")

    # Blends with Sub 1350 and Sub 1640
    for base_name, base_sub in [("Sub 1350", sub_1350), ("Sub 1640", sub_1640)]:
        for blend_w in [0.05, 0.10, 0.15]:
            sub_num = get_next_submission_number()
            blended = base_sub.copy()
            for target in TARGETS:
                col = f'scaled_{target}'
                best_test = results[target][best_per_target[target]][2]
                blended[col] = (1 - blend_w) * base_sub[col] + blend_w * best_test
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            print(f"  Sub {sub_num}: {blend_w*100:.0f}% advanced + {(1-blend_w)*100:.0f}% {base_name}")

    # Per-variant standalone
    for variant_name in ['feat_weighted', 'pca_subspace', 'multi_kernel', 'huber_weighted']:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': results['angle'][variant_name][2],
            'scaled_depth': results['depth'][variant_name][2],
            'scaled_left_right': results['left_right'][variant_name][2],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

        # Diversity analysis
        r_angle = np.corrcoef(sub['scaled_angle'], sub_1350['scaled_angle'])[0, 1]
        r_depth = np.corrcoef(sub['scaled_depth'], sub_1350['scaled_depth'])[0, 1]
        r_lr = np.corrcoef(sub['scaled_left_right'], sub_1350['scaled_left_right'])[0, 1]
        print(f"  Sub {sub_num}: {variant_name} standalone")
        print(f"    r with Sub 1350: angle={r_angle:.4f}, depth={r_depth:.4f}, lr={r_lr:.4f}")

    # Per-variant 10% blend with Sub 1640
    for variant_name in ['feat_weighted', 'pca_subspace', 'multi_kernel', 'huber_weighted']:
        sub_num = get_next_submission_number()
        blended = sub_1640.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = 0.90 * sub_1640[col] + 0.10 * results[target][variant_name][2]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: 10% {variant_name} + 90% Sub 1640")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
