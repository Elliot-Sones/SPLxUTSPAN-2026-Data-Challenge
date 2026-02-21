"""
Ultra-Minimal Feature Pipeline

Hypothesis: 200+ features with 345 samples causes massive overfitting (1.77x ratio).
With 3-5 hand-picked biomechanical features, we sacrifice some signal but
dramatically reduce overfitting. If the LOO/test ratio drops from 1.77x to 1.1x,
even worse LOO MSE can produce better test MSE.

Example: Current angle LOO=0.002511, test=0.006454 (2.48x).
If 5-feature angle LOO=0.005000 with 1.1x ratio: test=0.005500 (better!).

Feature selection is based on biomechanical reasoning, NOT data-driven,
to avoid selection overfitting.
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

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


def butter_lowpass(data, cutoff=8.0, fs=60.0, order=4):
    """Butterworth lowpass filter for denoising keypoint trajectories."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    x = np.asarray(data, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    if len(x) < 15:
        return x
    return filtfilt(b, a, x)


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
        ids, pids, targets = [], [], []
        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
        result = {'X_3d': X_3d, 'pids': np.array(pids),
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


def get_smooth_pos(ts_hr, kp_index, joint_name, frame):
    """Get Butterworth-smoothed position at a specific frame."""
    idx = kp_index.get(joint_name)
    if idx is None:
        return np.zeros(3)
    pos = np.zeros(3)
    for c in range(3):
        series = ts_hr[:, idx, c]
        smoothed = butter_lowpass(series)
        pos[c] = smoothed[frame]
    return pos


def get_smooth_vel(ts_hr, kp_index, joint_name, frame):
    """Get velocity (from Butterworth-smoothed position) at a specific frame."""
    idx = kp_index.get(joint_name)
    if idx is None:
        return np.zeros(3)
    vel = np.zeros(3)
    for c in range(3):
        series = ts_hr[:, idx, c]
        smoothed = butter_lowpass(series)
        vel[c] = np.gradient(smoothed, DT)[frame]
    return vel


def angle_between(a, b):
    """Angle in degrees between vectors a and b."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 90.0
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1)))


# ================================================================
# ULTRA-MINIMAL FEATURE EXTRACTION
# ================================================================

def extract_minimal_features(ts_3d, ts_hr, kp_index, release_frame):
    """Extract ALL candidate minimal features for all targets.
    Returns dict of {feature_name: value}."""
    feats = {}

    # Key frames
    f_angle = 153
    f_depth = 150
    f_lr = 170

    # ---- POSITIONS (Butterworth smoothed, hoop-relative) ----
    rw_a = get_smooth_pos(ts_hr, kp_index, 'right_wrist', f_angle)
    re_a = get_smooth_pos(ts_hr, kp_index, 'right_elbow', f_angle)
    rs_a = get_smooth_pos(ts_hr, kp_index, 'right_shoulder', f_angle)
    neck_a = get_smooth_pos(ts_hr, kp_index, 'neck', f_angle)
    mh_a = get_smooth_pos(ts_hr, kp_index, 'mid_hip', f_angle)
    rh_a = get_smooth_pos(ts_hr, kp_index, 'right_hip', f_angle)
    lh_a = get_smooth_pos(ts_hr, kp_index, 'left_hip', f_angle)
    ls_a = get_smooth_pos(ts_hr, kp_index, 'left_shoulder', f_angle)
    rk_a = get_smooth_pos(ts_hr, kp_index, 'right_knee', f_angle)

    rw_d = get_smooth_pos(ts_hr, kp_index, 'right_wrist', f_depth)
    re_d = get_smooth_pos(ts_hr, kp_index, 'right_elbow', f_depth)
    rs_d = get_smooth_pos(ts_hr, kp_index, 'right_shoulder', f_depth)
    mh_d = get_smooth_pos(ts_hr, kp_index, 'mid_hip', f_depth)
    rk_d = get_smooth_pos(ts_hr, kp_index, 'right_knee', f_depth)

    rw_lr = get_smooth_pos(ts_hr, kp_index, 'right_wrist', f_lr)
    re_lr = get_smooth_pos(ts_hr, kp_index, 'right_elbow', f_lr)
    rs_lr = get_smooth_pos(ts_hr, kp_index, 'right_shoulder', f_lr)
    ls_lr = get_smooth_pos(ts_hr, kp_index, 'left_shoulder', f_lr)
    rh_lr = get_smooth_pos(ts_hr, kp_index, 'right_hip', f_lr)
    lh_lr = get_smooth_pos(ts_hr, kp_index, 'left_hip', f_lr)

    # ---- VELOCITIES ----
    rw_vel_a = get_smooth_vel(ts_hr, kp_index, 'right_wrist', f_angle)
    rw_vel_d = get_smooth_vel(ts_hr, kp_index, 'right_wrist', f_depth)
    rw_vel_lr = get_smooth_vel(ts_hr, kp_index, 'right_wrist', f_lr)
    mh_vel_a = get_smooth_vel(ts_hr, kp_index, 'mid_hip', f_angle)

    # ============================================================
    # ANGLE FEATURES (biomechanical reasoning)
    # ============================================================

    # 1. Trunk forward lean: angle of (neck-hip) from vertical
    trunk_vec = neck_a - mh_a
    feats['trunk_lean'] = angle_between(trunk_vec, np.array([0, 0, 1]))

    # 2. Trunk lean direction (forward component specifically)
    trunk_n = np.linalg.norm(trunk_vec[:2])
    feats['trunk_forward_lean'] = np.degrees(np.arctan2(trunk_vec[0], trunk_vec[2])) if trunk_n > 1e-6 else 0.0

    # 3. Arm elevation: angle of (wrist-shoulder) from horizontal
    arm_vec = rw_a - rs_a
    arm_n = np.linalg.norm(arm_vec)
    feats['arm_elevation'] = np.degrees(np.arcsin(np.clip(arm_vec[2] / max(arm_n, 1e-6), -1, 1)))

    # 4. Elbow angle at frame 153
    ua = re_a - rs_a
    fa = rw_a - re_a
    feats['elbow_angle_153'] = angle_between(-ua, fa)

    # 5. Wrist height relative to shoulder
    feats['wrist_z_over_shoulder'] = rw_a[2] - rs_a[2]

    # 6. Wrist forward position relative to shoulder
    feats['wrist_forward_over_shoulder'] = rw_a[0] - rs_a[0]

    # 7. Release height over hoop
    feats['release_z_over_hoop'] = rw_a[2] - HOOP_POS[2]

    # 8. CoM vertical velocity at frame 153
    # CoM approximated as mid_hip
    feats['com_vel_z_153'] = mh_vel_a[2]

    # 9. Wrist vertical velocity at frame 153
    feats['wrist_vel_z_153'] = rw_vel_a[2]

    # 10. Wrist speed at frame 153
    feats['wrist_speed_153'] = np.linalg.norm(rw_vel_a)

    # ============================================================
    # DEPTH FEATURES (biomechanical reasoning)
    # ============================================================

    # 1. Wrist speed at frame 150 (power/force indicator)
    feats['wrist_speed_150'] = np.linalg.norm(rw_vel_d)

    # 2. Wrist vertical velocity at frame 150
    feats['wrist_vel_z_150'] = rw_vel_d[2]

    # 3. Wrist forward velocity at frame 150
    feats['wrist_vel_forward_150'] = rw_vel_d[0]

    # 4. Release frame timing
    feats['release_frame'] = float(release_frame)

    # 5. Elbow angular velocity at frame 150
    # Compute elbow angle at frames 148 and 152, take derivative
    for label, f1, f2 in [('elbow_angvel_150', 148, 152)]:
        re1 = get_smooth_pos(ts_hr, kp_index, 'right_elbow', f1)
        rs1 = get_smooth_pos(ts_hr, kp_index, 'right_shoulder', f1)
        rw1 = get_smooth_pos(ts_hr, kp_index, 'right_wrist', f1)
        re2 = get_smooth_pos(ts_hr, kp_index, 'right_elbow', f2)
        rs2 = get_smooth_pos(ts_hr, kp_index, 'right_shoulder', f2)
        rw2 = get_smooth_pos(ts_hr, kp_index, 'right_wrist', f2)
        a1 = angle_between(rs1 - re1, rw1 - re1)
        a2 = angle_between(rs2 - re2, rw2 - re2)
        feats[label] = (a2 - a1) / ((f2 - f1) * DT)

    # 6. Knee angle at frame 150 (leg drive)
    ra_d = get_smooth_pos(ts_hr, kp_index, 'right_ankle', f_depth)
    feats['knee_angle_150'] = angle_between(rh_a - rk_d, ra_d - rk_d)

    # 7. CoM vertical velocity at frame 150
    mh_vel_d = get_smooth_vel(ts_hr, kp_index, 'mid_hip', f_depth)
    feats['com_vel_z_150'] = mh_vel_d[2]

    # 8. Wrist height at frame 150
    feats['wrist_z_150'] = rw_d[2]

    # ============================================================
    # LEFT_RIGHT FEATURES (biomechanical reasoning)
    # ============================================================

    # 1. Wrist lateral position at frame 170 (hoop-relative)
    feats['wrist_lateral_170'] = rw_lr[1]

    # 2. Wrist lateral offset from shoulder
    feats['wrist_lat_offset_170'] = rw_lr[1] - rs_lr[1]

    # 3. Shoulder rotation (right-left shoulder line angle)
    shoulder_vec = rs_lr - ls_lr
    feats['shoulder_rotation_170'] = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))

    # 4. Hip rotation
    hip_vec = rh_lr - lh_lr
    feats['hip_rotation_170'] = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))

    # 5. Shoulder-hip rotation difference
    feats['rotation_offset_170'] = feats['shoulder_rotation_170'] - feats['hip_rotation_170']

    # 6. Wrist lateral velocity at frame 170
    feats['wrist_vel_lateral_170'] = rw_vel_lr[1]

    # 7. Elbow lateral position
    feats['elbow_lateral_170'] = re_lr[1]

    # 8. Arm plane lateral tilt
    arm_vec_lr = rw_lr - rs_lr
    arm_n_lr = np.linalg.norm(arm_vec_lr)
    if arm_n_lr > 1e-6:
        feats['arm_lateral_tilt_170'] = np.degrees(np.arcsin(np.clip(arm_vec_lr[1] / arm_n_lr, -1, 1)))
    else:
        feats['arm_lateral_tilt_170'] = 0.0

    # 9. Wrist forward position at frame 170
    feats['wrist_forward_170'] = rw_lr[0]

    return feats


# ================================================================
# MODELS
# ================================================================

def per_player_ridge_loo(X, y, pids, alpha=10.0):
    """Per-player Ridge with LOO cross-validation."""
    unique_pids = sorted(np.unique(pids))
    oof = np.zeros(len(y))
    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y[mask]
        indices = np.where(mask)[0]
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)
        for i in range(len(X_p)):
            m = np.ones(len(X_p), dtype=bool)
            m[i] = False
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_s[m], y_p[m])
            oof[indices[i]] = ridge.predict(X_s[i:i+1])[0]
    return oof


def per_player_ridge_predict(X_train, y_train, X_test, pids_train, pids_test, alpha=10.0):
    """Per-player Ridge prediction."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        if not np.any(te_mask):
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train[tr_mask])
        X_te_s = scaler.transform(X_test[te_mask])
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_s, y_train[tr_mask])
        test_preds[te_mask] = ridge.predict(X_te_s)
    return test_preds


def per_example_lw(X_train, y_train, X_test, pids_train, pids_test,
                   bandwidth_quantile=0.5, alpha=10.0):
    """Per-example locally weighted Ridge."""
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
# MAIN
# ================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ULTRA-MINIMAL FEATURE PIPELINE")
    print("=" * 70)
    print("  Hypothesis: 3-5 features -> less overfit -> better test MSE")
    print("  Current: 213 features, LOO/test ratio = 1.77x")
    print("  Target: 3-5 features, LOO/test ratio ~ 1.1-1.3x")

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']

    scalers = {}
    y_scaled = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled[target] = scalers[target].transform(
            y_train[:, TARGET_IDX[target]].reshape(-1, 1)).ravel()

    sub_1828 = pd.read_csv(SUBMISSION_DIR / "submission_1828.csv")

    # ============================================================
    # EXTRACT ALL MINIMAL FEATURES
    # ============================================================
    print("\nExtracting minimal features...")
    all_feats_train = []
    all_feats_test = []

    for i in range(len(pids_train)):
        ts_3d = train_data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_minimal_features(ts_3d, ts_hr, kp_index, rf)
        all_feats_train.append(feats)

    for i in range(len(pids_test)):
        ts_3d = test_data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_minimal_features(ts_3d, ts_hr, kp_index, rf)
        all_feats_test.append(feats)

    feat_names = list(all_feats_train[0].keys())
    X_all_train = np.array([[f[n] for n in feat_names] for f in all_feats_train], dtype=np.float32)
    X_all_test = np.array([[f[n] for n in feat_names] for f in all_feats_test], dtype=np.float32)
    X_all_train = np.nan_to_num(X_all_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_all_test = np.nan_to_num(X_all_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Total candidate features: {len(feat_names)}")

    # ============================================================
    # FEATURE CORRELATION ANALYSIS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("FEATURE CORRELATIONS WITH TARGETS")
    print(f"{'=' * 70}")

    # Define target-specific feature sets (biomechanically motivated)
    target_features = {
        'angle': {
            'biomech_3': ['trunk_forward_lean', 'arm_elevation', 'elbow_angle_153'],
            'biomech_5': ['trunk_forward_lean', 'arm_elevation', 'elbow_angle_153',
                          'wrist_z_over_shoulder', 'com_vel_z_153'],
            'biomech_7': ['trunk_forward_lean', 'arm_elevation', 'elbow_angle_153',
                          'wrist_z_over_shoulder', 'com_vel_z_153',
                          'wrist_forward_over_shoulder', 'release_z_over_hoop'],
        },
        'depth': {
            'biomech_3': ['wrist_speed_150', 'release_frame', 'com_vel_z_150'],
            'biomech_5': ['wrist_speed_150', 'release_frame', 'com_vel_z_150',
                          'wrist_vel_z_150', 'elbow_angvel_150'],
            'biomech_7': ['wrist_speed_150', 'release_frame', 'com_vel_z_150',
                          'wrist_vel_z_150', 'elbow_angvel_150',
                          'knee_angle_150', 'wrist_z_150'],
        },
        'left_right': {
            'biomech_3': ['wrist_lateral_170', 'shoulder_rotation_170', 'arm_lateral_tilt_170'],
            'biomech_5': ['wrist_lateral_170', 'shoulder_rotation_170', 'arm_lateral_tilt_170',
                          'wrist_vel_lateral_170', 'rotation_offset_170'],
            'biomech_7': ['wrist_lateral_170', 'shoulder_rotation_170', 'arm_lateral_tilt_170',
                          'wrist_vel_lateral_170', 'rotation_offset_170',
                          'hip_rotation_170', 'elbow_lateral_170'],
        },
    }

    for target in TARGETS:
        print(f"\n  {target.upper()}:")
        y_t = y_scaled[target]
        correlations = []
        for fi, fname in enumerate(feat_names):
            r = np.corrcoef(X_all_train[:, fi], y_t)[0, 1]
            correlations.append((fname, r))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for fname, r in correlations[:12]:
            marker = " <--" if fname in target_features[target]['biomech_5'] else ""
            print(f"    {fname:35s}: r={r:+.4f}{marker}")

    # ============================================================
    # TRAIN AND EVALUATE MINIMAL MODELS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("TRAINING MINIMAL MODELS")
    print(f"{'=' * 70}")

    results = {}  # target -> feature_set_name -> {mse, oof, test, r_1828, n_features}

    for target in TARGETS:
        print(f"\n  --- {target.upper()} ---")
        y_t = y_scaled[target]
        col = f'scaled_{target}'
        results[target] = {}

        for fs_name, fs_features in target_features[target].items():
            # Get feature indices
            f_indices = [feat_names.index(f) for f in fs_features if f in feat_names]
            X_tr = X_all_train[:, f_indices]
            X_te = X_all_test[:, f_indices]
            n_feats = len(f_indices)

            print(f"\n    {fs_name} ({n_feats} features):")

            # Model 1: Global Ridge (various alphas)
            best_alpha = 1.0
            best_mse = float('inf')
            for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
                oof = per_player_ridge_loo(X_tr, y_t, pids_train, alpha=alpha)
                mse = np.mean((oof - y_t) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha

            oof_ridge = per_player_ridge_loo(X_tr, y_t, pids_train, alpha=best_alpha)
            test_ridge = per_player_ridge_predict(X_tr, y_t, X_te, pids_train, pids_test, alpha=best_alpha)
            mse_ridge = np.mean((oof_ridge - y_t) ** 2)
            r_ridge = np.corrcoef(test_ridge, sub_1828[col].values)[0, 1]
            print(f"      Ridge(a={best_alpha:5.1f}):   LOO={mse_ridge:.6f}, r(1828)={r_ridge:.4f}")

            results[target][f'{fs_name}_ridge'] = {
                'mse': mse_ridge, 'oof': oof_ridge, 'test': test_ridge,
                'r': r_ridge, 'n_features': n_feats, 'alpha': best_alpha
            }

            # Model 2: Per-example LW Ridge (various configs)
            for bw, alpha in [(0.5, 5.0), (0.5, 10.0), (0.7, 10.0), (0.7, 50.0)]:
                oof_lw, test_lw = per_example_lw(X_tr, y_t, X_te, pids_train, pids_test,
                                                  bandwidth_quantile=bw, alpha=alpha)
                mse_lw = np.mean((oof_lw - y_t) ** 2)
                r_lw = np.corrcoef(test_lw, sub_1828[col].values)[0, 1]
                label = f'{fs_name}_lw_bw{int(bw*100)}_a{int(alpha)}'
                results[target][label] = {
                    'mse': mse_lw, 'oof': oof_lw, 'test': test_lw,
                    'r': r_lw, 'n_features': n_feats
                }
                print(f"      LW(bw={bw},a={alpha:4.0f}): LOO={mse_lw:.6f}, r(1828)={r_lw:.4f}")

        # Also try data-driven top-5 features (for comparison)
        y_t = y_scaled[target]
        correlations = [(fi, abs(np.corrcoef(X_all_train[:, fi], y_t)[0, 1]))
                       for fi in range(len(feat_names))]
        correlations.sort(key=lambda x: x[1], reverse=True)
        top5_indices = [c[0] for c in correlations[:5]]
        top5_names = [feat_names[i] for i in top5_indices]
        print(f"\n    data_driven_5: {top5_names}")

        X_tr = X_all_train[:, top5_indices]
        X_te = X_all_test[:, top5_indices]

        for alpha in [1.0, 10.0]:
            oof = per_player_ridge_loo(X_tr, y_t, pids_train, alpha=alpha)
            test_pred = per_player_ridge_predict(X_tr, y_t, X_te, pids_train, pids_test, alpha=alpha)
            mse = np.mean((oof - y_t) ** 2)
            r = np.corrcoef(test_pred, sub_1828[col].values)[0, 1]
            label = f'data_driven_5_ridge_a{int(alpha)}'
            results[target][label] = {
                'mse': mse, 'oof': oof, 'test': test_pred,
                'r': r, 'n_features': 5
            }
            print(f"      Ridge(a={alpha:5.1f}):   LOO={mse:.6f}, r(1828)={r:.4f}")

    # ============================================================
    # COMPARISON WITH FULL-FEATURE BASELINE
    # ============================================================
    print(f"\n{'=' * 70}")
    print("COMPARISON: MINIMAL vs FULL FEATURES")
    print(f"{'=' * 70}")

    # Full-feature LOO baselines (from angle_fix.py and per_example_pipeline.py)
    full_loo = {'angle': 0.002511, 'depth': 0.004473, 'left_right': 0.004163}
    full_test_est = {'angle': 0.006217, 'depth': 0.006820, 'left_right': 0.006820}

    for target in TARGETS:
        print(f"\n  {target.upper()}:")
        print(f"    Full features (213): LOO={full_loo[target]:.6f}, est test={full_test_est[target]:.6f}, ratio={full_test_est[target]/full_loo[target]:.2f}x")

        # Find best minimal model
        best_name = min(results[target], key=lambda x: results[target][x]['mse'])
        best = results[target][best_name]
        print(f"    Best minimal ({best['n_features']}f): LOO={best['mse']:.6f}, r(1828)={best['r']:.4f}")
        print(f"      Config: {best_name}")

        # Predict what test MSE would be at different overfit ratios
        for ratio in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
            pred_test = best['mse'] * ratio
            label = "BETTER!" if pred_test < full_test_est[target] else ""
            print(f"      If ratio={ratio:.1f}x: test MSE={pred_test:.6f} {label}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Strategy: for each target, pick best minimal model and blend with Sub 1828
    for target in TARGETS:
        # Sort by LOO MSE
        sorted_configs = sorted(results[target].items(), key=lambda x: x[1]['mse'])
        best_name, best = sorted_configs[0]
        second_name, second = sorted_configs[1]

        print(f"\n  {target}: best={best_name} (LOO={best['mse']:.6f})")
        print(f"           2nd={second_name} (LOO={second['mse']:.6f})")

    # Per-target best minimal predictions
    best_per_target = {}
    for target in TARGETS:
        sorted_configs = sorted(results[target].items(), key=lambda x: x[1]['mse'])
        best_per_target[target] = sorted_configs[0][1]

    # Also: average of top-3 configs per target (variance reduction)
    avg_per_target = {}
    for target in TARGETS:
        sorted_configs = sorted(results[target].items(), key=lambda x: x[1]['mse'])
        top3 = [c[1] for c in sorted_configs[:3]]
        avg_per_target[target] = {
            'test': np.mean([c['test'] for c in top3], axis=0),
            'oof': np.mean([c['oof'] for c in top3], axis=0),
            'mse': np.mean((np.mean([c['oof'] for c in top3], axis=0) - y_scaled[target]) ** 2),
        }

    # Blend weights to test
    for w in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_1828[col] + w * best_per_target[target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: best_minimal w={w:.2f} all targets")

    # Same with averaged top-3
    for w in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_1828[col] + w * avg_per_target[target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: avg_top3_minimal w={w:.2f} all targets")

    # Per-target independent blending (different weight per target)
    for target in TARGETS:
        for w in [0.30, 0.50, 1.00]:
            sub_num = get_next_submission_number()
            blended = sub_1828.copy()
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_1828[col] + w * best_per_target[target]['test']
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            print(f"  Sub {sub_num}: {target} only w={w:.2f} minimal")

    # Standalone minimal (no blending)
    sub_num = get_next_submission_number()
    standalone = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': best_per_target['angle']['test'],
        'scaled_depth': best_per_target['depth']['test'],
        'scaled_left_right': best_per_target['left_right']['test'],
    })
    standalone.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: STANDALONE minimal (no blending)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
