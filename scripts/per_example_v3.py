"""
Per-Example Pipeline V3 - Trajectory-Based Distance for Kernel Weighting

KEY IDEA: V1/V2 compute Gaussian kernel weights from distances in the
213-feature space (extracted at a SINGLE frame). V3 instead computes
distances using the FULL 240-frame trajectory of key joints, then still
uses the same compact 213-feature Ridge for prediction.

Why this might help:
- Full trajectory captures the entire shooting motion, not just a snapshot
- Two shots with identical frame-153 poses but different wind-ups should be
  weighted differently
- Trajectory distance is a richer measure of biomechanical similarity

Distance metrics explored:
1. PCA trajectory distance: PCA on flattened trajectory, Euclidean in PCA space
2. Correlation distance: 1 - correlation of key joint trajectories
3. Key-joint trajectory distance: Euclidean on smoothed trajectories of
   shooting-relevant joints
4. Per-target joint selection: Different joints matter for angle vs depth vs LR

We keep the prediction model identical to V1 (Ridge alpha=10 with Gaussian
kernel sample weighting, 213 features + 15 PLS) - only the DISTANCE
computation changes.
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
from scipy.spatial.distance import cdist, squareform, pdist
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

# Joints most relevant to each target's outcome
TARGET_JOINTS = {
    "angle": ['right_wrist', 'right_elbow', 'right_shoulder', 'neck', 'nose',
              'right_hip', 'right_knee'],
    "depth": ['right_wrist', 'right_elbow', 'right_shoulder', 'mid_hip',
              'right_hip', 'left_hip', 'right_knee', 'left_knee'],
    "left_right": ['right_wrist', 'left_wrist', 'right_shoulder', 'left_shoulder',
                   'right_hip', 'left_hip', 'right_elbow', 'neck'],
}

# Frame windows that matter for each target
TARGET_WINDOWS = {
    "angle": (100, 180),     # Release mechanics
    "depth": (90, 175),      # Includes push-off
    "left_right": (110, 190), # Includes follow-through
}


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
# DATA LOADING
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
# FEATURE EXTRACTION (identical to V1)
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
    """Extract compact feature set at a specific frame (identical to V1)."""
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
# TRAJECTORY DISTANCE COMPUTATION (NEW IN V3)
# ==============================================================

def extract_trajectory_features(data, target, kp_index):
    """Extract smoothed trajectory representation for distance computation.

    For each shot, extract smoothed hoop-relative trajectories of target-relevant
    joints within the target-relevant frame window. Return a flattened vector
    per shot that captures the full motion pattern.
    """
    joints = TARGET_JOINTS[target]
    f_start, f_end = TARGET_WINDOWS[target]
    n_frames = f_end - f_start
    n = len(data['pids'])

    # Each joint contributes 3 coords x n_frames values
    traj_dim = len(joints) * 3 * n_frames
    traj_feats = np.zeros((n, traj_dim), dtype=np.float32)

    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        idx = 0
        for jname in joints:
            j_idx = kp_index.get(jname)
            if j_idx is None:
                idx += 3 * n_frames
                continue
            for coord in range(3):
                series = ts_hr[f_start:f_end, j_idx, coord].astype(np.float64)
                # Smooth to reduce noise
                bad = np.isnan(series) | np.isinf(series)
                if np.any(bad):
                    good = ~bad
                    if np.any(good):
                        series[bad] = np.interp(
                            np.where(bad)[0], np.where(good)[0], series[good])
                    else:
                        series[:] = 0.0
                if len(series) >= 7:
                    series = savgol_filter(series, 7, 3)
                traj_feats[i, idx:idx + n_frames] = series
                idx += n_frames

    traj_feats = np.nan_to_num(traj_feats, nan=0.0, posinf=0.0, neginf=0.0)
    return traj_feats


def compute_pca_trajectory_distance(traj_train, traj_test, pids_train, pids_test,
                                      n_components=20):
    """PCA on trajectory features, then Euclidean distance in PCA space.

    Returns per-player distance matrices (train-train, test-train).
    """
    unique_pids = sorted(np.unique(pids_train))
    D_tr_tr_all = {}
    D_te_tr_all = {}

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        traj_p = traj_train[tr_mask]
        n_p = traj_p.shape[0]

        scaler = StandardScaler()
        traj_p_s = scaler.fit_transform(traj_p)
        nc = min(n_components, n_p - 1, traj_p_s.shape[1])
        pca = PCA(n_components=nc)
        Z_tr = pca.fit_transform(traj_p_s)

        D_tr_tr_all[pid] = cdist(Z_tr, Z_tr, metric='euclidean')

        if np.any(te_mask):
            traj_te = traj_test[te_mask]
            traj_te_s = scaler.transform(traj_te)
            Z_te = pca.transform(traj_te_s)
            D_te_tr_all[pid] = cdist(Z_te, Z_tr, metric='euclidean')
        else:
            D_te_tr_all[pid] = np.zeros((0, n_p))

    return D_tr_tr_all, D_te_tr_all


def compute_correlation_trajectory_distance(traj_train, traj_test,
                                              pids_train, pids_test):
    """Correlation-based trajectory distance (1 - abs(correlation)).

    Captures shape similarity regardless of scale.
    """
    unique_pids = sorted(np.unique(pids_train))
    D_tr_tr_all = {}
    D_te_tr_all = {}

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        traj_p = traj_train[tr_mask]
        n_p = traj_p.shape[0]

        # Standardize per-feature
        scaler = StandardScaler()
        traj_p_s = scaler.fit_transform(traj_p)

        D_tr_tr_all[pid] = cdist(traj_p_s, traj_p_s, metric='correlation')
        # Handle NaN from constant features
        D_tr_tr_all[pid] = np.nan_to_num(D_tr_tr_all[pid], nan=1.0)

        if np.any(te_mask):
            traj_te_s = scaler.transform(traj_test[te_mask])
            D_te_tr_all[pid] = cdist(traj_te_s, traj_p_s, metric='correlation')
            D_te_tr_all[pid] = np.nan_to_num(D_te_tr_all[pid], nan=1.0)
        else:
            D_te_tr_all[pid] = np.zeros((0, n_p))

    return D_tr_tr_all, D_te_tr_all


def compute_euclidean_trajectory_distance(traj_train, traj_test,
                                            pids_train, pids_test):
    """Standard Euclidean distance on standardized trajectory features."""
    unique_pids = sorted(np.unique(pids_train))
    D_tr_tr_all = {}
    D_te_tr_all = {}

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        traj_p = traj_train[tr_mask]
        n_p = traj_p.shape[0]

        scaler = StandardScaler()
        traj_p_s = scaler.fit_transform(traj_p)

        D_tr_tr_all[pid] = cdist(traj_p_s, traj_p_s, metric='euclidean')

        if np.any(te_mask):
            traj_te_s = scaler.transform(traj_test[te_mask])
            D_te_tr_all[pid] = cdist(traj_te_s, traj_p_s, metric='euclidean')
        else:
            D_te_tr_all[pid] = np.zeros((0, n_p))

    return D_tr_tr_all, D_te_tr_all


def compute_feature_distance(X_train, X_test, pids_train, pids_test):
    """V1-style Euclidean distance in the 213-feature space (baseline)."""
    unique_pids = sorted(np.unique(pids_train))
    D_tr_tr_all = {}
    D_te_tr_all = {}

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_p = X_train[tr_mask]
        n_p = X_p.shape[0]

        scaler = StandardScaler()
        X_p_s = scaler.fit_transform(X_p)

        D_tr_tr_all[pid] = cdist(X_p_s, X_p_s, metric='euclidean')

        if np.any(te_mask):
            X_te_s = scaler.transform(X_test[te_mask])
            D_te_tr_all[pid] = cdist(X_te_s, X_p_s, metric='euclidean')
        else:
            D_te_tr_all[pid] = np.zeros((0, n_p))

    return D_tr_tr_all, D_te_tr_all


def compute_blended_distance(D1_tr, D1_te, D2_tr, D2_te, weight1=0.5):
    """Blend two distance matrices. Normalize each to [0,1] first."""
    D_tr_bl = {}
    D_te_bl = {}
    for pid in D1_tr:
        # Normalize each distance matrix by its max for fair blending
        d1_max = D1_tr[pid].max() if D1_tr[pid].size > 0 else 1.0
        d2_max = D2_tr[pid].max() if D2_tr[pid].size > 0 else 1.0
        d1_max = max(d1_max, 1e-10)
        d2_max = max(d2_max, 1e-10)

        D_tr_bl[pid] = weight1 * (D1_tr[pid] / d1_max) + (1 - weight1) * (D2_tr[pid] / d2_max)

        if D1_te[pid].shape[0] > 0:
            D_te_bl[pid] = weight1 * (D1_te[pid] / d1_max) + (1 - weight1) * (D2_te[pid] / d2_max)
        else:
            D_te_bl[pid] = D1_te[pid]

    return D_tr_bl, D_te_bl


# ==============================================================
# LOCALLY WEIGHTED REGRESSION WITH CUSTOM DISTANCE
# ==============================================================

def locally_weighted_with_distance(X_train, y_train, X_test,
                                    pids_train, pids_test,
                                    D_tr_tr_dict, D_te_tr_dict,
                                    bandwidth_quantile=0.5, alpha=10.0):
    """Per-example locally weighted Ridge using pre-computed distance matrices.

    The DISTANCE comes from trajectory-based metrics.
    The FEATURES for Ridge are the compact 213 + PLS set.
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

        # Standardize features for Ridge (NOT for distance)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        # Get pre-computed distance matrices
        D_tr_tr = D_tr_tr_dict[pid]
        D_te_tr = D_te_tr_dict[pid]

        # Adaptive bandwidth from pairwise distances
        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        # LOO for training data
        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0  # Leave out self

            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test predictions
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
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("PER-EXAMPLE PIPELINE V3 - TRAJECTORY DISTANCE")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']

    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # ============================================================
    # PRECOMPUTE TRAJECTORY FEATURES FOR ALL TARGETS
    # ============================================================
    print("\nPrecomputing trajectory features...")
    traj_feats = {}
    traj_feats_test = {}
    for target in TARGETS:
        print(f"  {target}: joints={TARGET_JOINTS[target]}, "
              f"window={TARGET_WINDOWS[target]}")
        traj_feats[target] = extract_trajectory_features(
            train_data, target, kp_index)
        traj_feats_test[target] = extract_trajectory_features(
            test_data, target, kp_index)
        print(f"    shape: {traj_feats[target].shape}")

    # ============================================================
    # PER-TARGET PIPELINE
    # ============================================================

    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features (same as V1)
        print("  Extracting prediction features...")
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)
        print(f"  HC features: {X_train_hc.shape[1]}")

        # Augment with PLS
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Augmented features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # Compute distance matrices using different methods
        print("  Computing distance matrices...")
        traj_tr = traj_feats[target]
        traj_te = traj_feats_test[target]

        # 1. PCA trajectory distance
        D_pca_tr, D_pca_te = compute_pca_trajectory_distance(
            traj_tr, traj_te, pids_train, pids_test, n_components=20)

        # 2. Correlation trajectory distance
        D_corr_tr, D_corr_te = compute_correlation_trajectory_distance(
            traj_tr, traj_te, pids_train, pids_test)

        # 3. Euclidean trajectory distance
        D_euc_tr, D_euc_te = compute_euclidean_trajectory_distance(
            traj_tr, traj_te, pids_train, pids_test)

        # 4. V1-style feature distance (baseline)
        D_feat_tr, D_feat_te = compute_feature_distance(
            X_train_aug, X_test_aug, pids_train, pids_test)

        # 5. Blended: trajectory + feature distance
        D_blend50_tr, D_blend50_te = compute_blended_distance(
            D_pca_tr, D_pca_te, D_feat_tr, D_feat_te, weight1=0.5)
        D_blend30_tr, D_blend30_te = compute_blended_distance(
            D_pca_tr, D_pca_te, D_feat_tr, D_feat_te, weight1=0.3)
        D_blend70_tr, D_blend70_te = compute_blended_distance(
            D_pca_tr, D_pca_te, D_feat_tr, D_feat_te, weight1=0.7)

        # Test all distance metrics with bandwidth search
        distance_configs = [
            ("feat_v1", D_feat_tr, D_feat_te),
            ("pca_traj", D_pca_tr, D_pca_te),
            ("corr_traj", D_corr_tr, D_corr_te),
            ("euc_traj", D_euc_tr, D_euc_te),
            ("blend_50_50", D_blend50_tr, D_blend50_te),
            ("blend_30_70", D_blend30_tr, D_blend30_te),
            ("blend_70_30", D_blend70_tr, D_blend70_te),
        ]

        print("\n  Distance metric evaluation (LOO CV):")
        approach_results = {}

        for dist_name, D_tr, D_te in distance_configs:
            best_bw = 0.5
            best_mse = float('inf')
            best_oof = None
            best_test = None

            for bw in [0.3, 0.4, 0.5, 0.6]:
                oof, test_pred = locally_weighted_with_distance(
                    X_train_aug, y_target, X_test_aug,
                    pids_train, pids_test,
                    D_tr, D_te,
                    bandwidth_quantile=bw, alpha=10.0)
                mse = np.mean((oof - y_target) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_bw = bw
                    best_oof = oof.copy()
                    best_test = test_pred.copy()

            approach_results[dist_name] = {
                'mse': best_mse, 'bw': best_bw,
                'oof': best_oof, 'test': best_test,
            }
            print(f"    {dist_name:20s}: MSE={best_mse:.6f} (bw={best_bw})")

        # Also try with alpha=5 for the best trajectory approaches
        for dist_name, D_tr, D_te in distance_configs:
            if dist_name == "feat_v1":
                continue  # Already well-tuned in V1
            for alpha in [5.0, 20.0]:
                bw = approach_results[dist_name]['bw']
                oof, test_pred = locally_weighted_with_distance(
                    X_train_aug, y_target, X_test_aug,
                    pids_train, pids_test,
                    D_tr, D_te,
                    bandwidth_quantile=bw, alpha=alpha)
                mse = np.mean((oof - y_target) ** 2)
                key = f"{dist_name}_a{int(alpha)}"
                approach_results[key] = {
                    'mse': mse, 'bw': bw,
                    'oof': oof, 'test': test_pred,
                }
                if mse < approach_results[dist_name]['mse']:
                    print(f"    {key:20s}: MSE={mse:.6f} (BETTER than a10)")

        # Find best overall
        best_name = min(approach_results, key=lambda x: approach_results[x]['mse'])
        best_r = approach_results[best_name]
        print(f"\n  BEST for {target}: {best_name} "
              f"(MSE={best_r['mse']:.6f}, bw={best_r['bw']})")

        # Also keep feat_v1 as reference
        v1_r = approach_results['feat_v1']
        print(f"  V1 baseline:        feat_v1 "
              f"(MSE={v1_r['mse']:.6f}, bw={v1_r['bw']})")
        if best_r['mse'] < v1_r['mse']:
            pct = (v1_r['mse'] - best_r['mse']) / v1_r['mse'] * 100
            print(f"  IMPROVEMENT: {pct:.2f}% over V1 feature distance")
        else:
            pct = (best_r['mse'] - v1_r['mse']) / v1_r['mse'] * 100
            print(f"  WORSE: {pct:.2f}% vs V1 feature distance")

        results[target] = {
            'approach_results': approach_results,
            'best_name': best_name,
            'best_test': best_r['test'],
            'best_oof': best_r['oof'],
            'best_mse': best_r['mse'],
            'v1_test': v1_r['test'],
            'v1_mse': v1_r['mse'],
        }

    # ============================================================
    # OVERALL CV RESULTS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("OVERALL CV RESULTS")
    print(f"{'=' * 70}")

    total_mse_best = 0
    total_mse_v1 = 0
    for target in TARGETS:
        mse_best = results[target]['best_mse']
        mse_v1 = results[target]['v1_mse']
        total_mse_best += mse_best
        total_mse_v1 += mse_v1
        bname = results[target]['best_name']
        print(f"  {target:12s}: BEST={mse_best:.6f} ({bname}), V1={mse_v1:.6f}")

    mean_best = total_mse_best / 3
    mean_v1 = total_mse_v1 / 3
    print(f"  {'MEAN':12s}: BEST={mean_best:.6f}, V1={mean_v1:.6f}")
    if mean_best < mean_v1:
        pct = (mean_v1 - mean_best) / mean_v1 * 100
        print(f"  Overall improvement: {pct:.2f}%")
    else:
        pct = (mean_best - mean_v1) / mean_v1 * 100
        print(f"  Overall worse by: {pct:.2f}%")

    # ============================================================
    # DIVERSITY ANALYSIS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("DIVERSITY ANALYSIS")
    print(f"{'=' * 70}")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    for target in TARGETS:
        col = f"scaled_{target}"
        best_t = results[target]['best_test']
        v1_t = results[target]['v1_test']
        r_784 = np.corrcoef(sub_784[col].values, best_t)[0, 1]
        r_1350 = np.corrcoef(sub_1350[col].values, best_t)[0, 1]
        r_v1 = np.corrcoef(v1_t, best_t)[0, 1]
        print(f"  {target}: vs Sub784 r={r_784:.4f}, vs Sub1350 r={r_1350:.4f}, "
              f"vs V1_feat r={r_v1:.4f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Standalone best trajectory approach per target
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle']['best_test'],
        'scaled_depth': results['depth']['best_test'],
        'scaled_left_right': results['left_right']['best_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: STANDALONE (best trajectory per target)")

    # Blend with Sub 784 at tested-optimal weights
    blend_configs = [
        (0.00, 0.30, 0.50, "traj best, Sub784 opt weights"),
        (0.00, 0.25, 0.40, "traj best, conservative"),
        (0.00, 0.35, 0.55, "traj best, aggressive"),
        (0.00, 0.20, 0.30, "traj best, more conservative"),
    ]

    for aw, dw, lw, desc in blend_configs:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1 - aw) * sub_784['scaled_angle'] + aw * results['angle']['best_test']
        blended['scaled_depth'] = (1 - dw) * sub_784['scaled_depth'] + dw * results['depth']['best_test']
        blended['scaled_left_right'] = (1 - lw) * sub_784['scaled_left_right'] + lw * results['left_right']['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # Also: V3-V1 ensemble blend
    # If trajectory distance differs from feature distance, the average might help
    print("\n  V3+V1 ensemble blends:")
    for v3_w in [0.3, 0.5, 0.7]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            v3_pred = results[target]['best_test']
            v1_pred = results[target]['v1_test']
            combined = v3_w * v3_pred + (1 - v3_w) * v1_pred
            if target == 'angle':
                w = 0.0
            elif target == 'depth':
                w = 0.30
            else:
                w = 0.50
            blended[col] = (1 - w) * sub_784[col] + w * combined
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: V3+V1 (v3_w={v3_w:.1f}, dw=0.30 lw=0.50)")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # V3 best blended with Sub 1350 directly
    print("\n  V3 blended with Sub 1350:")
    for v3_w in [0.2, 0.3, 0.5]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            v3_pred = results[target]['best_test']
            blended[col] = (1 - v3_w) * sub_1350[col] + v3_w * v3_pred
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: {1-v3_w:.0%} Sub1350 + {v3_w:.0%} V3")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
