"""
Target Denoising Pipeline

Hypothesis: training targets contain measurement noise from camera-based tracking.
Denoising the targets before training could reduce overfitting to noise and improve
generalization on test data.

Methods:
A) KNN smoothing: replace each target with weighted average of K nearest neighbors
B) Ridge smoothing: LOO Ridge prediction blended with original label
C) Per-player mean shrinkage: pull outlier targets toward player-specific mean

Evaluation: always on ORIGINAL (raw) targets for honest comparison.
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
from sklearn.neighbors import NearestNeighbors

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


# ==============================================================
# DATA LOADING (same as per_example_pipeline.py)
# ==============================================================

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


# ==============================================================
# PLS AUGMENTATION (same as per_example_pipeline.py)
# ==============================================================

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
# TARGET DENOISING METHODS
# ==============================================================

def denoise_knn(X_train, y_train, pids_train, k=10, blend_alpha=0.3):
    """
    KNN smoothing: for each training sample, replace its target with a weighted
    average of itself (weight=1-blend_alpha) and its K nearest neighbors' targets
    (weight=blend_alpha).

    blend_alpha=0 means no denoising, blend_alpha=1 means full KNN replacement.
    """
    y_denoised = y_train.copy()
    unique_pids = sorted(np.unique(pids_train))

    for pid in unique_pids:
        mask = pids_train == pid
        X_p = X_train[mask]
        y_p = y_train[mask]
        indices = np.where(mask)[0]
        n_p = len(X_p)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        k_actual = min(k, n_p - 1)
        nn = NearestNeighbors(n_neighbors=k_actual + 1, metric='euclidean')
        nn.fit(X_s)

        for i in range(n_p):
            dists, neighbor_idx = nn.kneighbors(X_s[i:i+1])
            # Exclude self
            self_pos = np.where(neighbor_idx[0] == i)[0]
            neigh_idx = np.delete(neighbor_idx[0], self_pos)[:k_actual]
            neigh_dists = np.delete(dists[0], self_pos)[:k_actual]

            # Distance-weighted average of neighbors
            weights = 1.0 / (neigh_dists + 1e-6)
            weights /= weights.sum()
            knn_target = np.dot(weights, y_p[neigh_idx])

            # Blend original with KNN prediction
            y_denoised[indices[i]] = (1 - blend_alpha) * y_p[i] + blend_alpha * knn_target

    return y_denoised


def denoise_ridge_loo(X_train, y_train, pids_train, blend_alpha=0.2, ridge_alpha=10.0):
    """
    Ridge LOO smoothing: for each training sample, compute its LOO Ridge prediction
    using locally weighted regression, then blend with original.

    blend_alpha=0 means no denoising, blend_alpha=1 means full Ridge replacement.
    """
    y_denoised = y_train.copy()
    unique_pids = sorted(np.unique(pids_train))

    for pid in unique_pids:
        mask = pids_train == pid
        X_p = X_train[mask]
        y_p = y_train[mask]
        indices = np.where(mask)[0]
        n_p = len(X_p)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        # Pairwise distances for Gaussian kernel weighting
        D = cdist(X_s, X_s, metric='euclidean')
        all_dists = D[np.triu_indices(n_p, k=1)]
        sigma = np.quantile(all_dists, 0.5) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        for i in range(n_p):
            dists = D[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0  # Leave out self

            if weights.sum() < 1e-10:
                continue

            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(X_s, y_p, sample_weight=weights)
            ridge_pred = ridge.predict(X_s[i:i+1])[0]

            y_denoised[indices[i]] = (1 - blend_alpha) * y_p[i] + blend_alpha * ridge_pred

    return y_denoised


def denoise_mean_shrinkage(y_train, pids_train, shrinkage=0.1):
    """
    Per-player mean shrinkage: pull each target toward the player's mean.
    This specifically targets outliers while leaving average shots unchanged.

    shrinkage=0 means no change, shrinkage=1 means replace with player mean.
    """
    y_denoised = y_train.copy()
    unique_pids = sorted(np.unique(pids_train))

    for pid in unique_pids:
        mask = pids_train == pid
        y_p = y_train[mask]
        indices = np.where(mask)[0]

        player_mean = np.mean(y_p)
        y_denoised[indices] = (1 - shrinkage) * y_p + shrinkage * player_mean

    return y_denoised


def denoise_adaptive_knn(X_train, y_train, pids_train, k=10, sigma_threshold=1.5):
    """
    Adaptive KNN denoising: only denoise samples whose target deviates significantly
    from their local neighborhood. If a sample's target is within sigma_threshold
    standard deviations of its neighbors, leave it alone.
    """
    y_denoised = y_train.copy()
    unique_pids = sorted(np.unique(pids_train))
    n_modified = 0

    for pid in unique_pids:
        mask = pids_train == pid
        X_p = X_train[mask]
        y_p = y_train[mask]
        indices = np.where(mask)[0]
        n_p = len(X_p)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        k_actual = min(k, n_p - 1)
        nn = NearestNeighbors(n_neighbors=k_actual + 1, metric='euclidean')
        nn.fit(X_s)

        for i in range(n_p):
            dists, neighbor_idx = nn.kneighbors(X_s[i:i+1])
            self_pos = np.where(neighbor_idx[0] == i)[0]
            neigh_idx = np.delete(neighbor_idx[0], self_pos)[:k_actual]
            neigh_dists = np.delete(dists[0], self_pos)[:k_actual]

            neighbor_targets = y_p[neigh_idx]
            neigh_mean = np.mean(neighbor_targets)
            neigh_std = np.std(neighbor_targets)

            if neigh_std < 1e-8:
                continue

            z_score = abs(y_p[i] - neigh_mean) / neigh_std

            if z_score > sigma_threshold:
                # This sample is an outlier - pull toward neighborhood
                weights = 1.0 / (neigh_dists + 1e-6)
                weights /= weights.sum()
                knn_target = np.dot(weights, neighbor_targets)
                # Stronger correction for more extreme outliers
                correction_strength = min(0.5, (z_score - sigma_threshold) * 0.2)
                y_denoised[indices[i]] = (1 - correction_strength) * y_p[i] + correction_strength * knn_target
                n_modified += 1

    return y_denoised, n_modified


# ==============================================================
# LOCALLY WEIGHTED PREDICTION (from per_example_pipeline.py)
# ==============================================================

def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.5, ridge_alpha=10.0):
    """Locally weighted Ridge regression - the core of Sub 1350."""
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

        # OOF: leave-one-out locally weighted
        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0

            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test: locally weighted for each test example
        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=ridge_alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("TARGET DENOISING PIPELINE")
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

    # Load Sub 784 for blending
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # ==============================================================
    # EXTRACT FEATURES (per target)
    # ==============================================================

    features = {}
    for target in TARGETS:
        print(f"\nExtracting features for {target} (frame {TARGET_FRAMES[target]})...")
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)

        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Features: {X_train_aug.shape[1]}")
        features[target] = (X_train_aug, X_test_aug)

    # ==============================================================
    # BASELINE: no denoising (same as Sub 1350)
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("BASELINE (no denoising - should match Sub 1350)")
    print(f"{'=' * 70}")

    baseline_results = {}
    for target in TARGETS:
        X_train_aug, X_test_aug = features[target]
        y_target = y_scaled[target]
        oof, test_pred = locally_weighted_prediction(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.5, ridge_alpha=10.0)
        mse = np.mean((oof - y_target) ** 2)
        baseline_results[target] = {
            'oof': oof, 'test': test_pred, 'mse': mse
        }
        print(f"  {target}: LOO MSE = {mse:.6f}")
    baseline_mean = np.mean([baseline_results[t]['mse'] for t in TARGETS])
    print(f"  MEAN: {baseline_mean:.6f}")

    # ==============================================================
    # DENOISING EXPERIMENTS
    # ==============================================================

    all_results = {}

    # --- Method A: KNN smoothing ---
    print(f"\n{'=' * 70}")
    print("METHOD A: KNN SMOOTHING")
    print(f"{'=' * 70}")

    for k in [5, 10, 15]:
        for blend_alpha in [0.1, 0.2, 0.3, 0.5]:
            config_name = f"knn_k{k}_a{blend_alpha:.1f}"
            print(f"\n  Config: {config_name}")

            target_mses = {}
            target_tests = {}
            for target in TARGETS:
                X_train_aug, X_test_aug = features[target]
                y_original = y_scaled[target]

                # Denoise targets
                y_denoised = denoise_knn(X_train_aug, y_original, pids_train,
                                         k=k, blend_alpha=blend_alpha)
                delta = np.mean(np.abs(y_denoised - y_original))

                # Train on denoised, LOO eval on ORIGINAL
                oof, test_pred = locally_weighted_prediction(
                    X_train_aug, y_denoised, X_test_aug, pids_train, pids_test,
                    bandwidth_quantile=0.5, ridge_alpha=10.0)

                # Evaluate on ORIGINAL targets
                mse_original = np.mean((oof - y_original) ** 2)
                # Also MSE on denoised (for reference only)
                mse_denoised = np.mean((oof - y_denoised) ** 2)

                target_mses[target] = mse_original
                target_tests[target] = test_pred
                pct_change = (mse_original - baseline_results[target]['mse']) / baseline_results[target]['mse'] * 100
                print(f"    {target}: MSE(orig)={mse_original:.6f} (delta={pct_change:+.1f}%), "
                      f"MSE(denoised)={mse_denoised:.6f}, mean_shift={delta:.6f}")

            mean_mse = np.mean(list(target_mses.values()))
            pct_mean = (mean_mse - baseline_mean) / baseline_mean * 100
            print(f"    MEAN: {mean_mse:.6f} ({pct_mean:+.1f}%)")

            all_results[config_name] = {
                'mses': target_mses, 'tests': target_tests, 'mean': mean_mse
            }

    # --- Method B: Ridge LOO smoothing ---
    print(f"\n{'=' * 70}")
    print("METHOD B: RIDGE LOO SMOOTHING")
    print(f"{'=' * 70}")

    for blend_alpha in [0.1, 0.2, 0.3, 0.5]:
        for ridge_alpha in [10.0, 50.0]:
            config_name = f"ridge_a{blend_alpha:.1f}_ra{ridge_alpha:.0f}"
            print(f"\n  Config: {config_name}")

            target_mses = {}
            target_tests = {}
            for target in TARGETS:
                X_train_aug, X_test_aug = features[target]
                y_original = y_scaled[target]

                y_denoised = denoise_ridge_loo(X_train_aug, y_original, pids_train,
                                               blend_alpha=blend_alpha, ridge_alpha=ridge_alpha)
                delta = np.mean(np.abs(y_denoised - y_original))

                oof, test_pred = locally_weighted_prediction(
                    X_train_aug, y_denoised, X_test_aug, pids_train, pids_test,
                    bandwidth_quantile=0.5, ridge_alpha=10.0)

                mse_original = np.mean((oof - y_original) ** 2)
                target_mses[target] = mse_original
                target_tests[target] = test_pred
                pct_change = (mse_original - baseline_results[target]['mse']) / baseline_results[target]['mse'] * 100
                print(f"    {target}: MSE(orig)={mse_original:.6f} ({pct_change:+.1f}%), shift={delta:.6f}")

            mean_mse = np.mean(list(target_mses.values()))
            pct_mean = (mean_mse - baseline_mean) / baseline_mean * 100
            print(f"    MEAN: {mean_mse:.6f} ({pct_mean:+.1f}%)")

            all_results[config_name] = {
                'mses': target_mses, 'tests': target_tests, 'mean': mean_mse
            }

    # --- Method C: Mean shrinkage ---
    print(f"\n{'=' * 70}")
    print("METHOD C: PER-PLAYER MEAN SHRINKAGE")
    print(f"{'=' * 70}")

    for shrinkage in [0.05, 0.10, 0.15, 0.20, 0.30]:
        config_name = f"shrink_{shrinkage:.2f}"
        print(f"\n  Config: {config_name}")

        target_mses = {}
        target_tests = {}
        for target in TARGETS:
            X_train_aug, X_test_aug = features[target]
            y_original = y_scaled[target]

            y_denoised = denoise_mean_shrinkage(y_original, pids_train, shrinkage=shrinkage)
            delta = np.mean(np.abs(y_denoised - y_original))

            oof, test_pred = locally_weighted_prediction(
                X_train_aug, y_denoised, X_test_aug, pids_train, pids_test,
                bandwidth_quantile=0.5, ridge_alpha=10.0)

            mse_original = np.mean((oof - y_original) ** 2)
            target_mses[target] = mse_original
            target_tests[target] = test_pred
            pct_change = (mse_original - baseline_results[target]['mse']) / baseline_results[target]['mse'] * 100
            print(f"    {target}: MSE(orig)={mse_original:.6f} ({pct_change:+.1f}%), shift={delta:.6f}")

        mean_mse = np.mean(list(target_mses.values()))
        pct_mean = (mean_mse - baseline_mean) / baseline_mean * 100
        print(f"    MEAN: {mean_mse:.6f} ({pct_mean:+.1f}%)")

        all_results[config_name] = {
            'mses': target_mses, 'tests': target_tests, 'mean': mean_mse
        }

    # --- Method D: Adaptive KNN (only denoise outliers) ---
    print(f"\n{'=' * 70}")
    print("METHOD D: ADAPTIVE KNN (outlier-only denoising)")
    print(f"{'=' * 70}")

    for k in [8, 12]:
        for sigma_thresh in [1.0, 1.5, 2.0]:
            config_name = f"adaptive_k{k}_s{sigma_thresh:.1f}"
            print(f"\n  Config: {config_name}")

            target_mses = {}
            target_tests = {}
            total_modified = 0
            for target in TARGETS:
                X_train_aug, X_test_aug = features[target]
                y_original = y_scaled[target]

                y_denoised, n_mod = denoise_adaptive_knn(
                    X_train_aug, y_original, pids_train,
                    k=k, sigma_threshold=sigma_thresh)
                total_modified += n_mod
                delta = np.mean(np.abs(y_denoised - y_original))

                oof, test_pred = locally_weighted_prediction(
                    X_train_aug, y_denoised, X_test_aug, pids_train, pids_test,
                    bandwidth_quantile=0.5, ridge_alpha=10.0)

                mse_original = np.mean((oof - y_original) ** 2)
                target_mses[target] = mse_original
                target_tests[target] = test_pred
                pct_change = (mse_original - baseline_results[target]['mse']) / baseline_results[target]['mse'] * 100
                print(f"    {target}: MSE(orig)={mse_original:.6f} ({pct_change:+.1f}%), "
                      f"modified={n_mod}, shift={delta:.6f}")

            mean_mse = np.mean(list(target_mses.values()))
            pct_mean = (mean_mse - baseline_mean) / baseline_mean * 100
            print(f"    MEAN: {mean_mse:.6f} ({pct_mean:+.1f}%), total modified: {total_modified}")

            all_results[config_name] = {
                'mses': target_mses, 'tests': target_tests, 'mean': mean_mse
            }

    # ==============================================================
    # FIND BEST CONFIGS AND GENERATE SUBMISSIONS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("ALL RESULTS RANKED")
    print(f"{'=' * 70}")

    ranked = sorted(all_results.items(), key=lambda x: x[1]['mean'])
    for i, (name, r) in enumerate(ranked):
        pct = (r['mean'] - baseline_mean) / baseline_mean * 100
        print(f"  {i+1}. {name}: mean={r['mean']:.6f} ({pct:+.1f}%) "
              f"[a={r['mses']['angle']:.6f}, d={r['mses']['depth']:.6f}, lr={r['mses']['left_right']:.6f}]")

    print(f"\n  Baseline: mean={baseline_mean:.6f}")

    # Generate submissions for top 3 configs
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Best config standalone
    top_configs = ranked[:3]

    for config_name, config_data in top_configs:
        # Standalone submission
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': config_data['tests']['angle'],
            'scaled_depth': config_data['tests']['depth'],
            'scaled_left_right': config_data['tests']['left_right'],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"\n  Sub {sub_num}: STANDALONE {config_name}")
        print(f"    mean LOO MSE(orig): {config_data['mean']:.6f}")

        # Correlation with Sub 784
        for t in TARGETS:
            col = f'scaled_{t}'
            r = np.corrcoef(sub_784[col].values, config_data['tests'][t])[0, 1]
            print(f"    {t} vs Sub 784: r={r:.4f}")

        # Blended with Sub 784 at dw=0.30, lw=0.50
        sub_num2 = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = sub_784['scaled_angle']  # Don't touch angle
        blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * config_data['tests']['depth']
        blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * config_data['tests']['left_right']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num2}.csv", index=False)
        print(f"  Sub {sub_num2}: BLENDED {config_name} (aw=0, dw=0.30, lw=0.50)")

    # Also: generate a "best per target" submission
    # For each target, pick the config that gives best MSE on that target
    print(f"\n  Best per-target config:")
    best_per_target = {}
    for target in TARGETS:
        best_config = min(all_results.items(), key=lambda x: x[1]['mses'][target])
        best_per_target[target] = best_config
        pct = (best_config[1]['mses'][target] - baseline_results[target]['mse']) / baseline_results[target]['mse'] * 100
        print(f"    {target}: {best_config[0]} MSE={best_config[1]['mses'][target]:.6f} ({pct:+.1f}%)")

    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    blended['scaled_angle'] = sub_784['scaled_angle']
    blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * best_per_target['depth'][1]['tests']['depth']
    blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * best_per_target['left_right'][1]['tests']['left_right']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: BEST-PER-TARGET blended (aw=0, dw=0.30, lw=0.50)")

    # Diversity: compare with Sub 1350
    print(f"\n{'=' * 70}")
    print("DIVERSITY vs Sub 1350")
    print(f"{'=' * 70}")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
    for config_name, config_data in top_configs[:3]:
        print(f"\n  {config_name}:")
        for t in TARGETS:
            col = f'scaled_{t}'
            r = np.corrcoef(sub_1350[col].values, config_data['tests'][t])[0, 1]
            print(f"    {t} vs Sub 1350: r={r:.4f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
