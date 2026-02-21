"""
Overfitting Reduction + Diverse Ensemble Pipeline

Two-pronged attack:
1. REDUCE OVERFITTING: Apply safe_average (multi-config averaging) to ALL 3 targets,
   not just angle. The angle fix (2.48x -> lower) proved this works.
2. DIVERSE ENSEMBLE: Generate predictions from fundamentally different feature spaces
   (different frames, joint angles, PLS-only, temporal stats) to achieve r<0.70
   with Sub 1828 for ensemble gain.

Math: If we reduce overfit ratio from 1.77x to 1.3x -> MSE 0.004866 (beats 0.005).
Alternatively: two models with r=0.50 -> ensemble MSE = 0.75*single_MSE.
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048
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


# ================================================================
# FEATURE EXTRACTION - Multiple Spaces
# ================================================================

KEY_JOINTS = ['right_wrist', 'right_elbow', 'right_shoulder',
              'left_wrist', 'left_shoulder',
              'right_hip', 'left_hip', 'mid_hip',
              'right_knee', 'left_knee', 'neck', 'nose']


def extract_hc_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Standard HC features at a given frame (same as per_example_pipeline)."""
    f = int(np.clip(frame, 0, 239))
    feats = []
    for jname in KEY_JOINTS:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])
    for jname in KEY_JOINTS:
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


def extract_joint_angle_features(ts_3d, ts_hr, kp_index, frame):
    """Joint angles - fundamentally different from raw coordinates.
    Invariant to player position and partially to body size."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    def angle_between(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 90.0
        return np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))

    def get_pos(name):
        idx = kp_index.get(name)
        if idx is None:
            return np.zeros(3)
        return ts_hr[f, idx, :]

    # Right arm angles
    rs = get_pos('right_shoulder')
    re = get_pos('right_elbow')
    rw = get_pos('right_wrist')
    feats.append(angle_between(rs - re, rw - re))  # elbow angle
    feats.append(angle_between(re - rs, get_pos('right_hip') - rs))  # shoulder angle
    feats.append(angle_between(re - rs, np.array([0, 0, 1])))  # shoulder elevation

    # Left arm angles
    ls = get_pos('left_shoulder')
    le = get_pos('left_elbow') if 'left_elbow' in kp_index else np.zeros(3)
    lw = get_pos('left_wrist')
    feats.append(angle_between(ls - le, lw - le))  # left elbow
    feats.append(angle_between(le - ls, get_pos('left_hip') - ls))  # left shoulder

    # Leg angles
    rh = get_pos('right_hip')
    rk = get_pos('right_knee')
    ra = get_pos('right_ankle') if 'right_ankle' in kp_index else np.zeros(3)
    feats.append(angle_between(rh - rk, ra - rk))  # right knee
    lh = get_pos('left_hip')
    lk = get_pos('left_knee')
    la = get_pos('left_ankle') if 'left_ankle' in kp_index else np.zeros(3)
    feats.append(angle_between(lh - lk, la - lk))  # left knee

    # Trunk angle (hip-shoulder vs vertical)
    mh = get_pos('mid_hip')
    neck = get_pos('neck')
    trunk = neck - mh
    feats.append(angle_between(trunk, np.array([0, 0, 1])))  # trunk lean from vertical
    feats.append(np.degrees(np.arctan2(trunk[1], trunk[0])))  # trunk direction

    # Arm plane angle (wrist-elbow-shoulder plane vs vertical)
    if np.linalg.norm(rw - re) > 1e-6 and np.linalg.norm(rs - re) > 1e-6:
        plane_normal = np.cross(rw - re, rs - re)
        pn = np.linalg.norm(plane_normal)
        if pn > 1e-6:
            feats.append(angle_between(plane_normal / pn, np.array([0, 0, 1])))
        else:
            feats.append(90.0)
    else:
        feats.append(90.0)

    # Wrist-to-shoulder direction angles
    ws_vec = rw - rs
    ws_n = np.linalg.norm(ws_vec)
    if ws_n > 1e-6:
        ws_unit = ws_vec / ws_n
        feats.append(np.degrees(np.arctan2(ws_unit[1], ws_unit[0])))  # horizontal angle
        feats.append(np.degrees(np.arcsin(np.clip(ws_unit[2], -1, 1))))  # elevation
        feats.append(ws_n)  # arm reach
    else:
        feats.extend([0.0, 0.0, 0.0])

    # Body symmetry angles
    rs_pos = get_pos('right_shoulder')
    ls_pos = get_pos('left_shoulder')
    shoulder_vec = rs_pos - ls_pos
    feats.append(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
    hip_vec = rh - lh
    feats.append(np.degrees(np.arctan2(hip_vec[1], hip_vec[0])))
    feats.append(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])) -
                 np.degrees(np.arctan2(hip_vec[1], hip_vec[0])))  # shoulder-hip rotation offset

    # Angular velocities at target frame (derivative of angles over time)
    # Compute elbow angle over time window
    for dt_frames in [-3, -1, 1, 3]:
        f2 = int(np.clip(f + dt_frames, 0, 239))
        re2 = ts_hr[f2, kp_index.get('right_elbow', 0), :]
        rs2 = ts_hr[f2, kp_index.get('right_shoulder', 0), :]
        rw2 = ts_hr[f2, kp_index.get('right_wrist', 0), :]
        feats.append(angle_between(rs2 - re2, rw2 - re2))

    return np.array(feats, dtype=np.float32)


def extract_temporal_stats(ts_hr, kp_index, frame_ranges=None):
    """Temporal summary statistics across frame ranges.
    Different from point-in-time features - captures dynamics."""
    if frame_ranges is None:
        frame_ranges = [(80, 130), (130, 160), (160, 200), (80, 200)]
    feats = []
    for jname in ['right_wrist', 'right_elbow', 'right_shoulder',
                   'right_hip', 'right_knee', 'mid_hip', 'neck']:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * (len(frame_ranges) * 9))
            continue
        for f_start, f_end in frame_ranges:
            segment = ts_hr[f_start:f_end, idx, :]
            for coord in range(3):
                s = segment[:, coord]
                feats.append(np.nanmean(s))
                feats.append(np.nanstd(s))
                feats.append(np.nanmax(s) - np.nanmin(s))
    return np.array(feats, dtype=np.float32)


def extract_velocity_features(ts_hr, kp_index, frames):
    """Velocity and acceleration at multiple frames."""
    feats = []
    for jname in ['right_wrist', 'right_elbow', 'right_shoulder', 'mid_hip']:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * len(frames) * 6)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            vel = np.gradient(series, DT)
            acc = np.gradient(vel, DT)
            for f in frames:
                f = int(np.clip(f, 1, 238))
                feats.append(vel[f])
                feats.append(acc[f])
    return np.array(feats, dtype=np.float32)


# ================================================================
# FEATURE SPACE EXTRACTION
# ================================================================

def extract_feature_space(data, space_name, target_frame=153):
    """Extract features for a given feature space."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    release_frames = []

    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)

        if space_name == 'hc_standard':
            feats = extract_hc_features(ts_3d, ts_hr, kp_index, rf, target_frame)
        elif space_name == 'hc_early':
            feats = extract_hc_features(ts_3d, ts_hr, kp_index, rf, 110)
        elif space_name == 'hc_late':
            feats = extract_hc_features(ts_3d, ts_hr, kp_index, rf, 210)
        elif space_name == 'hc_release':
            feats = extract_hc_features(ts_3d, ts_hr, kp_index, rf, rf)
        elif space_name == 'joint_angles':
            feats = extract_joint_angle_features(ts_3d, ts_hr, kp_index, target_frame)
        elif space_name == 'joint_angles_early':
            feats = extract_joint_angle_features(ts_3d, ts_hr, kp_index, 110)
        elif space_name == 'temporal_stats':
            feats = extract_temporal_stats(ts_hr, kp_index)
        elif space_name == 'velocity_multi':
            feats = extract_velocity_features(ts_hr, kp_index,
                                               [100, 120, 140, 150, 160, 170, 180])
        elif space_name == 'combined_diverse':
            # Joint angles at multiple frames + temporal stats
            f1 = extract_joint_angle_features(ts_3d, ts_hr, kp_index, target_frame)
            f2 = extract_joint_angle_features(ts_3d, ts_hr, kp_index, 110)
            f3 = extract_temporal_stats(ts_hr, kp_index, [(100, 140), (140, 180)])
            feats = np.concatenate([f1, f2, f3])
        else:
            feats = extract_hc_features(ts_3d, ts_hr, kp_index, rf, target_frame)

        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


# ================================================================
# PLS AUGMENTATION
# ================================================================

def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test,
                     X_raw_train, X_raw_test, max_nc=15):
    unique_pids = sorted(np.unique(pids_train))
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
# PER-EXAMPLE LOCALLY WEIGHTED RIDGE
# ================================================================

def per_example_lw(X_train, y_train, X_test, pids_train, pids_test,
                   bandwidth_quantile=0.5, alpha=10.0, feature_select=None):
    """Per-example locally weighted Ridge. Returns (oof_preds, test_preds)."""
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

        if feature_select is not None and feature_select < X_tr_s.shape[1]:
            lasso = Lasso(alpha=0.001, max_iter=5000)
            lasso.fit(X_tr_s, y_tr)
            importances = np.abs(lasso.coef_)
            top_k = np.argsort(importances)[-feature_select:]
            X_tr_s = X_tr_s[:, top_k]
            X_te_s = X_te_s[:, top_k]

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


def global_ridge(X_train, y_train, X_test, pids_train, pids_test, alpha=100.0):
    """Simple per-player global Ridge."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        for i in range(len(X_tr)):
            mask = np.ones(len(X_tr), dtype=bool)
            mask[i] = False
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s[mask], y_tr[mask])
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        if np.any(te_mask):
            X_te_s = scaler.transform(X_test[te_mask])
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr)
            test_preds[te_indices] = ridge.predict(X_te_s)

    return oof_preds, test_preds


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("OVERFITTING REDUCTION + DIVERSE ENSEMBLE PIPELINE")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scalers = {}
    y_scaled = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled[target] = scalers[target].transform(
            y_train[:, TARGET_IDX[target]].reshape(-1, 1)).ravel()

    sub_1828 = pd.read_csv(SUBMISSION_DIR / "submission_1828.csv")

    # ============================================================
    # PRONG 1: SAFE AVERAGE FOR ALL 3 TARGETS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PRONG 1: SAFE AVERAGE (reduce overfitting for all targets)")
    print(f"{'=' * 70}")

    # Safe average configs - heavily regularized, averaged for variance reduction
    safe_configs = [
        # (name, bandwidth, alpha, feature_select)
        ("bw050_a100", 0.50, 100.0, None),
        ("bw070_a50",  0.70, 50.0,  None),
        ("bw080_a50",  0.80, 50.0,  None),
        ("bw080_a200", 0.80, 200.0, None),
        ("lasso30_bw070_a50", 0.70, 50.0, 30),
        ("global_a100", None, 100.0, None),  # None bandwidth = global Ridge
    ]

    safe_avg = {}  # target -> {'oof': ..., 'test': ...}

    for target in TARGETS:
        print(f"\n  --- {target.upper()} (frame {TARGET_FRAMES[target]}) ---")
        frame = TARGET_FRAMES[target]

        # Extract features
        X_tr_hc, _ = extract_feature_space(train_data, 'hc_standard', frame)
        X_te_hc, _ = extract_feature_space(test_data, 'hc_standard', frame)

        # Augment with PLS
        y_raw = y_train[:, TARGET_IDX[target]]
        X_tr_aug, X_te_aug = augment_with_pls(
            X_tr_hc, y_raw, pids_train, X_te_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        y_t = y_scaled[target]
        config_results = []

        for name, bw, alpha, fs in safe_configs:
            if bw is None:
                oof, test = global_ridge(X_tr_aug, y_t, X_te_aug, pids_train, pids_test, alpha)
            else:
                oof, test = per_example_lw(X_tr_aug, y_t, X_te_aug, pids_train, pids_test,
                                           bandwidth_quantile=bw, alpha=alpha, feature_select=fs)
            mse = np.mean((oof - y_t) ** 2)
            col = f'scaled_{target}'
            r_1828 = np.corrcoef(test, sub_1828[col].values)[0, 1]
            config_results.append({'name': name, 'oof': oof, 'test': test, 'mse': mse, 'r': r_1828})
            print(f"    {name:25s}: LOO={mse:.6f}, r(1828)={r_1828:.4f}")

        # Average all configs
        avg_oof = np.mean([c['oof'] for c in config_results], axis=0)
        avg_test = np.mean([c['test'] for c in config_results], axis=0)
        avg_mse = np.mean((avg_oof - y_t) ** 2)
        col = f'scaled_{target}'
        avg_r = np.corrcoef(avg_test, sub_1828[col].values)[0, 1]
        print(f"    {'SAFE_AVERAGE':25s}: LOO={avg_mse:.6f}, r(1828)={avg_r:.4f}")

        safe_avg[target] = {'oof': avg_oof, 'test': avg_test, 'mse': avg_mse, 'r': avg_r}

    # ============================================================
    # PRONG 2: DIVERSE FEATURE SPACES
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PRONG 2: DIVERSE FEATURE SPACES")
    print(f"{'=' * 70}")

    diverse_spaces = ['hc_early', 'hc_late', 'hc_release', 'joint_angles',
                      'joint_angles_early', 'temporal_stats', 'velocity_multi',
                      'combined_diverse']

    diverse_results = {}  # space_name -> target -> {'oof', 'test', 'mse', 'r'}

    for space in diverse_spaces:
        print(f"\n  --- Feature space: {space} ---")
        diverse_results[space] = {}

        for target in TARGETS:
            frame = TARGET_FRAMES[target]
            X_tr, _ = extract_feature_space(train_data, space, frame)
            X_te, _ = extract_feature_space(test_data, space, frame)

            # Don't add PLS for diverse spaces - keep them truly different
            y_t = y_scaled[target]

            # Standardize
            scaler_f = StandardScaler()
            X_tr_s = scaler_f.fit_transform(X_tr)
            X_te_s = scaler_f.transform(X_te)

            # Use moderate regularization (not trying to overfit-reduce here,
            # trying to get DIFFERENT predictions)
            oof, test = per_example_lw(X_tr_s, y_t, X_te_s, pids_train, pids_test,
                                       bandwidth_quantile=0.5, alpha=50.0)
            mse = np.mean((oof - y_t) ** 2)
            col = f'scaled_{target}'
            r_1828 = np.corrcoef(test, sub_1828[col].values)[0, 1]

            diverse_results[space][target] = {'oof': oof, 'test': test, 'mse': mse, 'r': r_1828}
            print(f"    {target:12s}: LOO={mse:.6f}, r(1828)={r_1828:.4f}, n_feat={X_tr.shape[1]}")

    # Also: PLS-only predictions (just the 15 PLS components, no HC)
    print(f"\n  --- Feature space: pls_only ---")
    diverse_results['pls_only'] = {}
    for target in TARGETS:
        y_raw = y_train[:, TARGET_IDX[target]]
        y_t = y_scaled[target]

        # Extract PLS components alone
        unique_pids = sorted(np.unique(pids_train))
        max_nc = 15
        pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
        pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)
        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            scaler_r = StandardScaler()
            raw_tr = scaler_r.fit_transform(train_data['X_raw'][tr_mask])
            raw_te = scaler_r.transform(test_data['X_raw'][te_mask])
            pls = PLSRegression(n_components=min(15, tr_mask.sum() - 5))
            pls.fit(raw_tr, y_raw[tr_mask])
            pls_train[tr_mask, :] = pls.transform(raw_tr)
            pls_test[te_mask, :] = pls.transform(raw_te)

        oof, test = per_example_lw(pls_train, y_t, pls_test, pids_train, pids_test,
                                    bandwidth_quantile=0.5, alpha=50.0)
        mse = np.mean((oof - y_t) ** 2)
        col = f'scaled_{target}'
        r_1828 = np.corrcoef(test, sub_1828[col].values)[0, 1]
        diverse_results['pls_only'][target] = {'oof': oof, 'test': test, 'mse': mse, 'r': r_1828}
        print(f"    {target:12s}: LOO={mse:.6f}, r(1828)={r_1828:.4f}, n_feat=15")

    # ============================================================
    # DIVERSITY ANALYSIS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("DIVERSITY ANALYSIS (correlation with Sub 1828)")
    print(f"{'=' * 70}")

    all_sources = ['safe_avg'] + list(diverse_results.keys())
    for target in TARGETS:
        print(f"\n  {target}:")
        for source in all_sources:
            if source == 'safe_avg':
                r = safe_avg[target]['r']
                mse = safe_avg[target]['mse']
            else:
                r = diverse_results[source][target]['r']
                mse = diverse_results[source][target]['mse']
            diversity_label = "LOW" if r > 0.90 else ("MED" if r > 0.70 else "HIGH")
            print(f"    {source:25s}: r={r:.4f} ({diversity_label}), LOO={mse:.6f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # --- Submission type 1: Safe average for all 3 targets ---
    # Apply same approach as angle fix but to all targets
    for w in [0.20, 0.30, 0.50]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        # For angle: keep Sub 1828's angle (already fixed)
        # For depth and LR: blend safe_average
        blended['scaled_depth'] = (1 - w) * sub_1828['scaled_depth'] + w * safe_avg['depth']['test']
        blended['scaled_left_right'] = (1 - w) * sub_1828['scaled_left_right'] + w * safe_avg['left_right']['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: safe_avg depth+LR w={w:.2f} (angle from Sub 1828)")

    # --- Submission type 2: Safe average ALL targets including angle ---
    for w in [0.20, 0.30, 0.50]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_1828[col] + w * safe_avg[target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: safe_avg ALL targets w={w:.2f}")

    # --- Submission type 3: Most diverse feature space blends ---
    # Find most diverse source per target
    for target in TARGETS:
        best_diverse = None
        best_r = 1.0
        for source in diverse_results:
            r = diverse_results[source][target]['r']
            mse = diverse_results[source][target]['mse']
            # Only consider if LOO MSE is not terrible (< 3x safe_avg)
            if mse < safe_avg[target]['mse'] * 3 and r < best_r:
                best_r = r
                best_diverse = source
        if best_diverse:
            print(f"  Most diverse for {target}: {best_diverse} (r={best_r:.4f})")

    # Per-target diverse blend at small weights
    for w_diverse in [0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            # Find most diverse source with reasonable accuracy
            best_source = None
            best_r = 1.0
            for source in diverse_results:
                r = diverse_results[source][target]['r']
                mse = diverse_results[source][target]['mse']
                if mse < safe_avg[target]['mse'] * 3 and r < best_r:
                    best_r = r
                    best_source = source
            if best_source:
                div_test = diverse_results[best_source][target]['test']
                blended[col] = (1 - w_diverse) * sub_1828[col] + w_diverse * div_test
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: per-target most-diverse w={w_diverse:.2f}")

    # --- Submission type 4: Combined safe_avg + diverse ---
    # First apply safe_avg, then add diversity on top
    for w_safe, w_div in [(0.30, 0.05), (0.30, 0.10), (0.50, 0.10)]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            # Apply safe_avg first
            intermediate = (1 - w_safe) * sub_1828[col] + w_safe * safe_avg[target]['test']
            # Then add diversity
            best_source = None
            best_r = 1.0
            for source in diverse_results:
                r = diverse_results[source][target]['r']
                mse = diverse_results[source][target]['mse']
                if mse < safe_avg[target]['mse'] * 3 and r < best_r:
                    best_r = r
                    best_source = source
            if best_source:
                div_test = diverse_results[best_source][target]['test']
                blended[col] = (1 - w_div) * intermediate + w_div * div_test
            else:
                blended[col] = intermediate
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: safe_avg w={w_safe:.2f} + diverse w={w_div:.2f}")

    # --- Submission type 5: Multi-diverse average ---
    # Average top-3 most diverse sources per target
    sub_num = get_next_submission_number()
    blended = sub_1828.copy()
    for target in TARGETS:
        col = f'scaled_{target}'
        # Rank by diversity (lowest r)
        ranked = sorted(diverse_results.items(),
                       key=lambda x: x[1][target]['r'])
        # Take top-3 most diverse with reasonable accuracy
        selected = []
        for source, res in ranked:
            if res[target]['mse'] < safe_avg[target]['mse'] * 3:
                selected.append(res[target]['test'])
                if len(selected) >= 3:
                    break
        if selected:
            avg_diverse = np.mean(selected, axis=0)
            blended[col] = 0.90 * sub_1828[col] + 0.10 * avg_diverse
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: top-3 diverse avg w=0.10 per target")

    # --- Submission type 6: Per-target weight optimization ---
    # Use different weights per target based on diversity
    for config_name, aw, dw, lw in [
        ("balanced", 0.30, 0.30, 0.30),
        ("depth_focused", 0.20, 0.50, 0.20),
        ("lr_focused", 0.20, 0.20, 0.50),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        blended['scaled_angle'] = (1 - aw) * sub_1828['scaled_angle'] + aw * safe_avg['angle']['test']
        blended['scaled_depth'] = (1 - dw) * sub_1828['scaled_depth'] + dw * safe_avg['depth']['test']
        blended['scaled_left_right'] = (1 - lw) * sub_1828['scaled_left_right'] + lw * safe_avg['left_right']['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {config_name} (aw={aw}, dw={dw}, lw={lw}) safe_avg")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
