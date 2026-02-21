"""
Five Approaches to Close the Gap to 0.0059

Current best: LB 0.006619 (Sub 1828). Target: 0.0059. Gap: 10.9%.

Per-target MSE decomposition (Sub 1828):
  Angle:  ~0.006217 (overfit 2.48x, LOO 0.002511)
  Depth:  ~0.006820 (overfit 1.51x)
  LR:     ~0.006820 (overfit 1.62x)

Approaches (all target overfitting reduction):
1. Test-Time Augmentation (TTA) - perturb test keypoints, average predictions
2. Per-Player Nested CV Regularization - LOPO for honest HP selection
3. Transductive PCA Distance - use train+test for better kernel distance
4. Bayesian Ridge with Global Prior - shrink per-example toward global model
5. Joint Angle Features - body-proportion invariant supplementary features
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
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Measured keypoint noise in feet (converted from mm)
# ankle ~1.4mm = 0.0046ft, wrist ~6.6mm = 0.0217ft, median ~4mm = 0.013ft
NOISE_SIGMA_FEET = 0.013  # median noise level


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
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result
    return process(train_df, True), process(test_df, False)


# ==============================================================
# FEATURE EXTRACTION
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


def _angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame, include_joint_angles=False):
    """Extract feature set at a specific frame.

    If include_joint_angles=True, appends 10 joint angle features.
    """
    f = int(np.clip(frame, 0, 239))
    feats = []

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    # Hoop-relative positions + velocities at target frame
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])

    # Hoop-relative summary stats
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

    # Arm mechanics
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

    # Body alignment
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

    # ============================================================
    # APPROACH 5: JOINT ANGLE FEATURES (body-proportion invariant)
    # ============================================================
    if include_joint_angles:
        rw_i = kp_index.get('right_wrist')
        re_i = kp_index.get('right_elbow')
        rs_i = kp_index.get('right_shoulder')
        rh_i = kp_index.get('right_hip')
        lh_i = kp_index.get('left_hip')
        rk_i = kp_index.get('right_knee')
        lk_i = kp_index.get('left_knee')
        ra_i = kp_index.get('right_ankle')
        la_i = kp_index.get('left_ankle')
        neck_i = kp_index.get('neck')
        mh_i = kp_index.get('mid_hip')
        ls_i = kp_index.get('left_shoulder')

        # 1. Shoulder elevation angle: angle(hip->shoulder, shoulder->elbow)
        if all(x is not None for x in [rs_i, rh_i, re_i]):
            v1 = ts_3d[f, rs_i] - ts_3d[f, rh_i]  # torso direction
            v2 = ts_3d[f, re_i] - ts_3d[f, rs_i]   # upper arm direction
            feats.append(_angle_between(v1, v2))
        else:
            feats.append(90.0)

        # 2. Trunk forward lean: angle of (neck - mid_hip) with vertical
        if neck_i is not None and mh_i is not None:
            trunk = ts_hr[f, neck_i] - ts_hr[f, mh_i]
            vertical = np.array([0, 0, 1], dtype=np.float32)
            feats.append(_angle_between(trunk, vertical))
            # 3. Trunk lateral lean: atan2(lateral, vertical) component
            feats.append(np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8)))
        else:
            feats.append(0.0)
            feats.append(0.0)

        # 4. Right knee flexion: angle at knee between (knee->hip) and (knee->ankle)
        if all(x is not None for x in [rk_i, rh_i, ra_i]):
            v1 = ts_3d[f, rh_i] - ts_3d[f, rk_i]
            v2 = ts_3d[f, ra_i] - ts_3d[f, rk_i]
            feats.append(_angle_between(v1, v2))
        else:
            feats.append(90.0)

        # 5. Left knee flexion
        if all(x is not None for x in [lk_i, lh_i, la_i]):
            v1 = ts_3d[f, lh_i] - ts_3d[f, lk_i]
            v2 = ts_3d[f, la_i] - ts_3d[f, lk_i]
            feats.append(_angle_between(v1, v2))
        else:
            feats.append(90.0)

        # 6. Wrist deviation: angle of forearm with vertical
        if re_i is not None and rw_i is not None:
            forearm = ts_3d[f, rw_i] - ts_3d[f, re_i]
            vertical = np.array([0, 0, 1], dtype=np.float32)
            feats.append(_angle_between(forearm, vertical))
        else:
            feats.append(90.0)

        # 7. Arm line angle: shoulder-to-wrist direction vs hoop direction (in HR coords)
        if rs_i is not None and rw_i is not None:
            arm_line = ts_hr[f, rw_i] - ts_hr[f, rs_i]
            hoop_dir = np.array([1, 0, 0.5], dtype=np.float32)  # forward + up
            feats.append(_angle_between(arm_line, hoop_dir))
        else:
            feats.append(90.0)

        # 8. Shoulder rotation: angle between shoulder line and forward direction
        if rs_i is not None and ls_i is not None:
            shoulder_line = ts_hr[f, rs_i] - ts_hr[f, ls_i]
            lateral = np.array([0, 1, 0], dtype=np.float32)
            feats.append(_angle_between(shoulder_line, lateral))
        else:
            feats.append(90.0)

        # 9. Hip-shoulder twist: angle between hip line and shoulder line projected to XY
        if all(x is not None for x in [rh_i, lh_i, rs_i, ls_i]):
            hip_line = ts_hr[f, rh_i, :2] - ts_hr[f, lh_i, :2]
            shoulder_line = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
            hn = np.linalg.norm(hip_line)
            sn = np.linalg.norm(shoulder_line)
            if hn > 1e-6 and sn > 1e-6:
                cos_a = np.clip(np.dot(hip_line, shoulder_line) / (hn * sn), -1, 1)
                feats.append(np.degrees(np.arccos(cos_a)))
            else:
                feats.append(0.0)
        else:
            feats.append(0.0)

        # 10. Elbow height relative to shoulder (normalized)
        if re_i is not None and rs_i is not None:
            feats.append(ts_hr[f, re_i, 2] - ts_hr[f, rs_i, 2])
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def extract_all_features(data, target, include_joint_angles=False):
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
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame,
                                 include_joint_angles=include_joint_angles)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


def extract_test_features_with_noise(data, target, sigma, rng,
                                      include_joint_angles=False):
    """Extract test features after adding Gaussian noise to keypoints."""
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []

    for i in range(n):
        ts_3d = data['X_3d'][i].copy()
        # Add noise to all keypoints (shape: 240, n_kp, 3)
        noise = rng.normal(0, sigma, ts_3d.shape).astype(np.float32)
        # Zero out noise for NaN positions
        nan_mask = np.isnan(ts_3d)
        noise[nan_mask] = 0
        ts_3d = ts_3d + noise

        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame,
                                 include_joint_angles=include_joint_angles)
        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test,
                     X_raw_train, X_raw_test):
    """Add PLS components to feature matrices. Returns (X_train_aug, X_test_aug, pls_models)."""
    unique_pids = sorted(np.unique(pids_train))
    max_nc = 15
    pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)
    pls_models = {}  # Cache for TTA reuse

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
        pls_models[pid] = (scaler, pls, best_nc)

    return (np.hstack([X_train, pls_train]),
            np.hstack([X_test, pls_test]),
            pls_models)


def augment_test_with_cached_pls(X_test_hc, pids_test, X_raw_test, pls_models):
    """Apply cached PLS transforms to new test features (for TTA)."""
    unique_pids = sorted(np.unique(pids_test))
    max_nc = 15
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        te_mask = pids_test == pid
        if pid not in pls_models:
            continue
        raw_scaler, pls, nc = pls_models[pid]
        raw_te = raw_scaler.transform(X_raw_test[te_mask])
        pls_test[te_mask, :nc] = pls.transform(raw_te)

    return np.hstack([X_test_hc, pls_test])


# ==============================================================
# APPROACH 1: BASELINE PER-EXAMPLE LOCALLY WEIGHTED RIDGE
# ==============================================================

def locally_weighted_ridge(X_tr_s, y_tr, X_query, dists_to_train, sigma, alpha=10.0,
                           global_prior_coef=None):
    """Single-query locally weighted Ridge prediction.

    If global_prior_coef is provided, implements Bayesian Ridge that shrinks
    toward the global model instead of toward zero (Approach 4).
    Standard Ridge: (X'WX + alpha*I) b = X'Wy
    Bayesian Ridge: (X'WX + alpha*I) b = X'Wy + alpha * b_prior
    """
    weights = np.exp(-dists_to_train ** 2 / (2 * sigma ** 2))

    if weights.sum() < 1e-10:
        return np.mean(y_tr)

    if global_prior_coef is not None:
        # Bayesian Ridge: shrink toward global model instead of zero
        W = np.diag(weights)
        XtWX = X_tr_s.T @ W @ X_tr_s
        XtWy = X_tr_s.T @ (W @ y_tr)
        p = X_tr_s.shape[1]
        A = XtWX + alpha * np.eye(p)
        rhs = XtWy + alpha * global_prior_coef
        try:
            b = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return float(X_query @ b)
    else:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_s, y_tr, sample_weight=weights)
        return ridge.predict(X_query.reshape(1, -1))[0]


def run_per_example(X_train, y_train, X_test, pids_train, pids_test,
                    bandwidth_quantile=0.3, alpha=10.0,
                    use_transductive_pca=False, pca_dims=30,
                    use_bayesian_prior=False, prior_alpha=100.0):
    """Core per-example prediction with optional enhancements.

    Approach 3: use_transductive_pca=True uses PCA on combined train+test for distances
    Approach 4: use_bayesian_prior=True shrinks toward a global Ridge per player
                (prior computed in same standardized space to avoid scale mismatch)
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

        # Standardize within player
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        # APPROACH 4: Compute global prior in SAME standardized space
        prior_coef = None
        if use_bayesian_prior:
            global_ridge = Ridge(alpha=prior_alpha)
            global_ridge.fit(X_tr_s, y_tr)
            prior_coef = global_ridge.coef_.copy()

        # APPROACH 3: Transductive PCA distance
        if use_transductive_pca and len(X_te) > 0:
            X_combined = np.vstack([X_tr_s, X_te_s])
            n_components = min(pca_dims, X_combined.shape[1], X_combined.shape[0] - 1)
            pca = PCA(n_components=n_components)
            X_combined_pca = pca.fit_transform(X_combined)
            X_tr_pca = X_combined_pca[:n_tr]
            X_te_pca = X_combined_pca[n_tr:]

            D_tr_tr = cdist(X_tr_pca, X_tr_pca, metric='euclidean')
            D_te_tr = cdist(X_te_pca, X_tr_pca, metric='euclidean')

            all_dists_combined = cdist(X_combined_pca, X_combined_pca, metric='euclidean')
            upper_tri = all_dists_combined[np.triu_indices(len(X_combined_pca), k=1)]
            sigma = np.quantile(upper_tri, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
            D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))
            all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
            sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
            sigma = max(sigma, 1e-6)

        # OOF: leave-one-out
        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            oof_preds[tr_indices[i]] = locally_weighted_ridge(
                X_tr_s, y_tr, X_tr_s[i], dists, sigma, alpha, prior_coef)

        # Test predictions
        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            test_preds[te_indices[j]] = locally_weighted_ridge(
                X_tr_s, y_tr, X_te_s[j], dists, sigma, alpha, prior_coef)

    return oof_preds, test_preds


# ==============================================================
# APPROACH 2: NESTED CV FOR PER-PLAYER REGULARIZATION
# ==============================================================

def nested_cv_select_hyperparams(X_train, y_train, pids_train):
    """Use LOPO to honestly select per-player alpha and bandwidth.

    For each held-out player, find the best (alpha, bandwidth) on the
    remaining 4 players, then use those settings for the held-out player.
    """
    unique_pids = sorted(np.unique(pids_train))
    alphas = [5.0, 10.0, 50.0, 100.0]
    bandwidths = [0.2, 0.3, 0.5, 0.7]

    best_params = {}

    for held_out_pid in unique_pids:
        # Inner CV: evaluate hyperparams on the other 4 players
        inner_pids = [p for p in unique_pids if p != held_out_pid]
        best_score = float('inf')
        best_alpha = 10.0
        best_bw = 0.3

        for alpha in alphas:
            for bw in bandwidths:
                # LOO on inner players
                total_mse = 0
                total_n = 0
                for val_pid in inner_pids:
                    val_mask = pids_train == val_pid
                    train_inner_mask = np.isin(pids_train, [p for p in inner_pids if p != val_pid])

                    X_inner_tr = X_train[val_mask]  # This player's data
                    y_inner_tr = y_train[val_mask]
                    n_p = len(X_inner_tr)

                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X_inner_tr)
                    D = cdist(X_s, X_s, metric='euclidean')
                    all_d = D[np.triu_indices(n_p, k=1)]
                    sigma = np.quantile(all_d, bw) if len(all_d) > 0 else 1.0
                    sigma = max(sigma, 1e-6)

                    for i in range(n_p):
                        dists = D[i, :]
                        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
                        weights[i] = 0
                        if weights.sum() < 1e-10:
                            pred = np.mean(y_inner_tr)
                        else:
                            ridge = Ridge(alpha=alpha)
                            ridge.fit(X_s, y_inner_tr, sample_weight=weights)
                            pred = ridge.predict(X_s[i:i+1])[0]
                        total_mse += (pred - y_inner_tr[i]) ** 2
                        total_n += 1

                avg_mse = total_mse / total_n if total_n > 0 else float('inf')
                if avg_mse < best_score:
                    best_score = avg_mse
                    best_alpha = alpha
                    best_bw = bw

        best_params[held_out_pid] = (best_alpha, best_bw)
        print(f"      Player {held_out_pid}: alpha={best_alpha}, bw={best_bw} (inner MSE={best_score:.6f})")

    return best_params


def run_nested_cv(X_train, y_train, X_test, pids_train, pids_test, best_params):
    """Per-example prediction using per-player hyperparameters from nested CV."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        alpha, bw = best_params[pid]
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

    return oof_preds, test_preds


# ==============================================================
# APPROACH 1: TEST-TIME AUGMENTATION
# ==============================================================

def run_tta(train_data, test_data, target, X_train_aug, y_target,
            pids_train, pids_test, pls_models,
            n_tta=30, sigma=NOISE_SIGMA_FEET,
            bandwidth_quantile=0.3, alpha=10.0,
            include_joint_angles=False,
            use_transductive_pca=False, pca_dims=30,
            use_bayesian_prior=False, prior_alpha=100.0):
    """Run prediction with TTA: perturb test keypoints, average predictions."""
    unique_pids = sorted(np.unique(pids_train))
    n_test = len(pids_test)
    all_test_preds = np.zeros((n_tta, n_test))

    # Pre-compute training-side data (fixed across TTA iterations)
    player_data = {}
    for pid in unique_pids:
        tr_mask = pids_train == pid
        X_tr = X_train_aug[tr_mask]
        y_tr = y_target[tr_mask]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        n_tr = len(X_tr)
        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        sig = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
        sig = max(sig, 1e-6)

        # Bayesian prior: compute in same standardized space
        prior_coef = None
        if use_bayesian_prior:
            global_ridge = Ridge(alpha=prior_alpha)
            global_ridge.fit(X_tr_s, y_tr)
            prior_coef = global_ridge.coef_.copy()

        player_data[pid] = {
            'tr_mask': tr_mask,
            'X_tr_s': X_tr_s,
            'y_tr': y_tr,
            'scaler': scaler,
            'sigma': sig,
            'prior_coef': prior_coef,
        }

    # Also get the clean (no-noise) prediction as iteration 0
    for it in range(n_tta):
        rng = np.random.RandomState(42 + it)

        if it == 0:
            # Clean prediction (no noise)
            X_test_hc = extract_all_features(test_data, target,
                                              include_joint_angles=include_joint_angles)[0]
        else:
            # Perturbed prediction
            X_test_hc = extract_test_features_with_noise(
                test_data, target, sigma, rng,
                include_joint_angles=include_joint_angles)

        # Augment with cached PLS
        X_test_aug_tta = augment_test_with_cached_pls(
            X_test_hc, pids_test, test_data['X_raw'], pls_models)

        # Per-player prediction
        for pid in unique_pids:
            te_mask = pids_test == pid
            if not np.any(te_mask):
                continue
            te_indices = np.where(te_mask)[0]
            pd_ = player_data[pid]
            X_te_s = pd_['scaler'].transform(X_test_aug_tta[te_mask])
            D_te_tr = cdist(X_te_s, pd_['X_tr_s'], metric='euclidean')

            for j in range(len(X_te_s)):
                dists = D_te_tr[j, :]
                pred = locally_weighted_ridge(
                    pd_['X_tr_s'], pd_['y_tr'], X_te_s[j], dists,
                    pd_['sigma'], alpha, pd_['prior_coef'])
                all_test_preds[it, te_indices[j]] = pred

    # Average across all TTA iterations
    test_preds = np.mean(all_test_preds, axis=0)
    return test_preds, all_test_preds


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("FIVE APPROACHES TO CLOSE THE GAP TO 0.0059")
    print("=" * 70)
    print(f"  Current best: LB 0.006619 (Sub 1828)")
    print(f"  Target: 0.0059")
    print(f"  Gap: {(0.006619 - 0.0059)/0.006619*100:.1f}%")

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

    sub_1828 = pd.read_csv(SUBMISSION_DIR / "submission_1828.csv")
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # ==============================================================
    # PER-TARGET PIPELINE
    # ==============================================================

    all_results = {}

    for target in TARGETS:
        t_start = time.time()
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]

        # ---- EXTRACT FEATURES (standard + joint angles) ----
        print("  Extracting standard features...")
        X_train_hc, rf_train = extract_all_features(train_data, target,
                                                      include_joint_angles=False)
        X_test_hc, rf_test = extract_all_features(test_data, target,
                                                    include_joint_angles=False)
        print(f"  HC features: {X_train_hc.shape[1]}")

        print("  Extracting features with joint angles (Approach 5)...")
        X_train_hc_ja, _ = extract_all_features(train_data, target,
                                                  include_joint_angles=True)
        X_test_hc_ja, _ = extract_all_features(test_data, target,
                                                 include_joint_angles=True)
        print(f"  HC+JA features: {X_train_hc_ja.shape[1]} (+{X_train_hc_ja.shape[1] - X_train_hc.shape[1]} joint angles)")

        # ---- PLS AUGMENTATION ----
        print("  Adding PLS components...")
        X_train_aug, X_test_aug, pls_models = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        X_train_aug_ja, X_test_aug_ja, pls_models_ja = augment_with_pls(
            X_train_hc_ja, y_raw, pids_train,
            X_test_hc_ja, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        print(f"  Augmented: {X_train_aug.shape[1]} (standard), {X_train_aug_ja.shape[1]} (with JA)")

        # ============================================================
        # BASELINE: Standard per-example (Sub 1350 settings)
        # ============================================================
        print("\n  [BASELINE] Standard per-example (bw=0.3, alpha=10)...")
        oof_base, test_base = run_per_example(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.3, alpha=10.0)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"    LOO MSE: {mse_base:.6f}")

        # ============================================================
        # APPROACH 1: TTA (Test-Time Augmentation)
        # ============================================================
        print(f"\n  [1] TTA (n=30, sigma={NOISE_SIGMA_FEET})...")
        test_tta, all_tta = run_tta(
            train_data, test_data, target, X_train_aug, y_target,
            pids_train, pids_test, pls_models,
            n_tta=30, sigma=NOISE_SIGMA_FEET,
            bandwidth_quantile=0.3, alpha=10.0)

        # Also test different sigma values
        tta_results = {'tta_0.013': test_tta}
        for sigma_test in [0.005, 0.008, 0.020]:
            key = f'tta_{sigma_test:.3f}'
            print(f"    TTA sigma={sigma_test}...")
            t_s, _ = run_tta(
                train_data, test_data, target, X_train_aug, y_target,
                pids_train, pids_test, pls_models,
                n_tta=20, sigma=sigma_test,
                bandwidth_quantile=0.3, alpha=10.0)
            tta_results[key] = t_s

        # TTA variance analysis
        tta_std = np.std(all_tta, axis=0)
        print(f"    TTA prediction std: mean={np.mean(tta_std):.6f}, max={np.max(tta_std):.6f}")
        print(f"    TTA vs clean: max_diff={np.max(np.abs(test_tta - all_tta[0])):.6f}")

        # ============================================================
        # APPROACH 2: Nested CV for per-player regularization
        # ============================================================
        print(f"\n  [2] Nested CV per-player regularization...")
        best_params = nested_cv_select_hyperparams(X_train_aug, y_target, pids_train)
        oof_nested, test_nested = run_nested_cv(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test, best_params)
        mse_nested = np.mean((oof_nested - y_target) ** 2)
        print(f"    LOO MSE: {mse_nested:.6f} (delta vs baseline: {(mse_nested - mse_base)/mse_base*100:+.2f}%)")

        # ============================================================
        # APPROACH 3: Transductive PCA distance
        # ============================================================
        print(f"\n  [3] Transductive PCA distance...")
        for pca_dims in [20, 30, 50]:
            oof_tpca, test_tpca = run_per_example(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                bandwidth_quantile=0.3, alpha=10.0,
                use_transductive_pca=True, pca_dims=pca_dims)
            mse_tpca = np.mean((oof_tpca - y_target) ** 2)
            print(f"    PCA dims={pca_dims}: LOO MSE={mse_tpca:.6f} (delta: {(mse_tpca - mse_base)/mse_base*100:+.2f}%)")
            if pca_dims == 30:
                test_tpca_best = test_tpca
                oof_tpca_best = oof_tpca

        # ============================================================
        # APPROACH 4: Bayesian Ridge with global prior
        # Prior computed in same standardized space as per-example models
        # Standard Ridge: shrink toward zero. Bayesian: shrink toward global model.
        # ============================================================
        print(f"\n  [4] Bayesian Ridge with global prior...")
        best_bayes_mse = float('inf')
        for prior_alpha in [50.0, 100.0, 200.0, 500.0]:
            oof_bayes, test_bayes = run_per_example(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                bandwidth_quantile=0.3, alpha=10.0,
                use_bayesian_prior=True, prior_alpha=prior_alpha)
            mse_bayes = np.mean((oof_bayes - y_target) ** 2)
            print(f"    Prior alpha={prior_alpha}: LOO MSE={mse_bayes:.6f} (delta: {(mse_bayes - mse_base)/mse_base*100:+.2f}%)")
            if mse_bayes < best_bayes_mse:
                best_bayes_mse = mse_bayes
                test_bayes_best = test_bayes
                oof_bayes_best = oof_bayes
                best_prior_alpha = prior_alpha
        print(f"    Best prior_alpha={best_prior_alpha}")

        # ============================================================
        # APPROACH 5: Joint angle features
        # ============================================================
        print(f"\n  [5] Joint angle features...")
        oof_ja, test_ja = run_per_example(
            X_train_aug_ja, y_target, X_test_aug_ja, pids_train, pids_test,
            bandwidth_quantile=0.3, alpha=10.0)
        mse_ja = np.mean((oof_ja - y_target) ** 2)
        print(f"    LOO MSE: {mse_ja:.6f} (delta: {(mse_ja - mse_base)/mse_base*100:+.2f}%)")

        # ============================================================
        # COMBINATIONS
        # ============================================================
        # TTA + Bayesian: use TTA with Bayesian prior inside
        print(f"\n  [COMBO] TTA + Bayesian Ridge (prior_alpha={best_prior_alpha})...")
        test_tta_bayes, _ = run_tta(
            train_data, test_data, target, X_train_aug, y_target,
            pids_train, pids_test, pls_models,
            n_tta=30, sigma=NOISE_SIGMA_FEET,
            bandwidth_quantile=0.3, alpha=10.0,
            use_bayesian_prior=True, prior_alpha=best_prior_alpha)

        print(f"  [COMBO] TTA + Joint Angles...")
        test_tta_ja, _ = run_tta(
            train_data, test_data, target, X_train_aug_ja, y_target,
            pids_train, pids_test, pls_models_ja,
            n_tta=30, sigma=NOISE_SIGMA_FEET,
            bandwidth_quantile=0.3, alpha=10.0,
            include_joint_angles=True)

        print(f"  [COMBO] TTA + Bayesian + Joint Angles...")
        test_tta_bayes_ja, _ = run_tta(
            train_data, test_data, target, X_train_aug_ja, y_target,
            pids_train, pids_test, pls_models_ja,
            n_tta=30, sigma=NOISE_SIGMA_FEET,
            bandwidth_quantile=0.3, alpha=10.0,
            include_joint_angles=True,
            use_bayesian_prior=True, prior_alpha=best_prior_alpha)

        # ============================================================
        # DIVERSITY ANALYSIS
        # ============================================================
        print(f"\n  Diversity vs Sub 1828 ({target}):")
        col = f'scaled_{target}'
        ref = sub_1828[col].values
        approaches = {
            'baseline': test_base,
            'tta_0.013': test_tta,
            'nested_cv': test_nested,
            'transductive_pca': test_tpca_best,
            'bayesian_ridge': test_bayes_best,
            'joint_angles': test_ja,
            'tta+bayes': test_tta_bayes,
            'tta+ja': test_tta_ja,
            'tta+bayes+ja': test_tta_bayes_ja,
        }
        # Add TTA sigma variants
        for key, t_preds in tta_results.items():
            if key != 'tta_0.013':
                approaches[key] = t_preds

        for name, preds in approaches.items():
            r = np.corrcoef(ref, preds)[0, 1]
            diff = np.mean(np.abs(ref - preds))
            print(f"    {name:25s}: r={r:.4f}, mean_abs_diff={diff:.6f}")

        all_results[target] = approaches

        t_elapsed = time.time() - t_start
        print(f"\n  {target} completed in {t_elapsed:.0f}s")

    # ==============================================================
    # GENERATE SUBMISSIONS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Define submission configs
    submission_configs = [
        # (name, angle_source, depth_source, lr_source)
        ("baseline", "baseline", "baseline", "baseline"),
        ("tta_only", "tta_0.013", "tta_0.013", "tta_0.013"),
        ("tta_s005", "tta_0.005", "tta_0.005", "tta_0.005"),
        ("tta_s008", "tta_0.008", "tta_0.008", "tta_0.008"),
        ("tta_s020", "tta_0.020", "tta_0.020", "tta_0.020"),  # key becomes tta_0.020
        ("nested_cv", "nested_cv", "nested_cv", "nested_cv"),
        ("tpca", "transductive_pca", "transductive_pca", "transductive_pca"),
        ("bayesian", "bayesian_ridge", "bayesian_ridge", "bayesian_ridge"),
        ("joint_angles", "joint_angles", "joint_angles", "joint_angles"),
        ("tta_bayes", "tta+bayes", "tta+bayes", "tta+bayes"),
        ("tta_ja", "tta+ja", "tta+ja", "tta+ja"),
        ("tta_bayes_ja", "tta+bayes+ja", "tta+bayes+ja", "tta+bayes+ja"),
        # Target-specific: best for each target
        ("target_specific", "tta+bayes+ja", "tta+bayes", "tta+ja"),
    ]

    # For each config, blend with Sub 1828 at various weights
    sub_1640 = pd.read_csv(SUBMISSION_DIR / "submission_1640.csv")

    for config_name, a_src, d_src, lr_src in submission_configs:
        angle_preds = all_results["angle"][a_src]
        depth_preds = all_results["depth"][d_src]
        lr_preds = all_results["left_right"][lr_src]

        # Blend weights to try:
        # The current Sub 1828 uses aw=0.50 safe_avg + Sub 1640
        # Our new angle predictions should replace Sub 1828's angle directly
        # For depth/LR, Sub 1640 = 10% LASSO + 90% (30% per-example + 70% Sub 784)
        for aw, dw, lw in [(0.50, 0.30, 0.50), (0.30, 0.20, 0.30), (1.00, 0.50, 0.70)]:
            sub_num = get_next_submission_number()
            blended = sub_784.copy()
            blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*angle_preds
            blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*depth_preds
            blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*lr_preds

            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

            if aw == 0.50 and dw == 0.30 and lw == 0.50:
                a_std = blended['scaled_angle'].std()
                d_mean = blended['scaled_depth'].mean()
                print(f"  Sub {sub_num}: {config_name} (aw={aw} dw={dw} lw={lw}) "
                      f"angle_std={a_std:.4f} depth_mean={d_mean:.4f}")

    # Also blend with Sub 1828 (keep its angle, replace depth/LR)
    print(f"\n  Blending with Sub 1828 (keep angle, try new depth/LR):")
    for config_name, _, d_src, lr_src in submission_configs[:8]:
        depth_preds = all_results["depth"][d_src]
        lr_preds = all_results["left_right"][lr_src]

        for dw, lw in [(0.15, 0.25), (0.30, 0.50)]:
            sub_num = get_next_submission_number()
            blended = sub_1828.copy()
            blended['scaled_depth'] = (1-dw)*sub_1828['scaled_depth'] + dw*depth_preds
            blended['scaled_left_right'] = (1-lw)*sub_1828['scaled_left_right'] + lw*lr_preds

            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            if dw == 0.15:
                r_depth = np.corrcoef(sub_1828['scaled_depth'].values, depth_preds)[0, 1]
                r_lr = np.corrcoef(sub_1828['scaled_left_right'].values, lr_preds)[0, 1]
                print(f"  Sub {sub_num}: 1828+{config_name} dw={dw} lw={lw} "
                      f"(r_depth={r_depth:.4f}, r_lr={r_lr:.4f})")

    # Full replacement: new angle + new depth/LR
    print(f"\n  Full replacement submissions:")
    for config_name, a_src, d_src, lr_src in [
        ("tta_bayes_ja_full", "tta+bayes+ja", "tta+bayes+ja", "tta+bayes+ja"),
        ("tta_full", "tta_0.013", "tta_0.013", "tta_0.013"),
    ]:
        angle_preds = all_results["angle"][a_src]
        depth_preds = all_results["depth"][d_src]
        lr_preds = all_results["left_right"][lr_src]

        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': angle_preds,
            'scaled_depth': depth_preds,
            'scaled_left_right': lr_preds,
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {config_name} (standalone)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
