"""
Inverse Dynamics V2: Surgical Feature Integration

Key insight from V1: Adding all 58 dynamics features to the 213 HC+PLS features
HURTS performance (+11-24% worse LOO MSE) because:
1. Dynamics features are highly correlated with HC features (r=0.93-0.96)
2. Extra dimensions add noise to the distance computation in locally weighted regression

Strategy for V2:
1. Select ONLY the few dynamics features with lowest redundancy vs HC
2. Use dynamics features to build a SEPARATE prediction model
3. Blend at the PREDICTION level (not feature level)
4. Focus on depth (biggest bottleneck) and targets where diversity is highest
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
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FPS = 60.0
FEET_TO_METERS = 0.3048

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

REFERENCE_BODY_MASS = 85.0
SEGMENT_MASS_RATIOS = {
    'upper_arm': 0.028, 'forearm': 0.016, 'hand': 0.006,
    'thigh': 0.100, 'shank': 0.047, 'trunk': 0.497,
}
SEGMENT_ROG_RATIOS = {
    'upper_arm': 0.322, 'forearm': 0.303, 'hand': 0.297,
    'thigh': 0.323, 'shank': 0.302,
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


def smooth_trajectory(traj, window=11, polyorder=3):
    smoothed = np.zeros_like(traj, dtype=np.float64)
    for ax in range(3):
        smoothed[:, ax] = safe_savgol(traj[:, ax], window, polyorder)
    return smoothed


def compute_joint_angle_series(p1_traj, p2_traj, p3_traj):
    v1 = p1_traj - p2_traj
    v2 = p3_traj - p2_traj
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos_angle = np.clip(dot / (n1 * n2 + 1e-9), -1, 1)
    return np.arccos(cos_angle)


def compute_angular_velocity_3d(parent_traj, child_traj, window=9, polyorder=3):
    parent_smooth = smooth_trajectory(parent_traj, window, polyorder)
    child_smooth = smooth_trajectory(child_traj, window, polyorder)
    parent_vel = np.zeros_like(parent_smooth)
    child_vel = np.zeros_like(child_smooth)
    for ax in range(3):
        parent_vel[:, ax] = safe_savgol(parent_smooth[:, ax], window, polyorder, deriv=1, delta=DT)
        child_vel[:, ax] = safe_savgol(child_smooth[:, ax], window, polyorder, deriv=1, delta=DT)
    r = child_smooth - parent_smooth
    v_rel = child_vel - parent_vel
    r_norm_sq = np.sum(r**2, axis=1, keepdims=True) + 1e-9
    omega = np.cross(r, v_rel) / r_norm_sq
    return omega


def compute_segment_length(p1_traj, p2_traj):
    diffs = smooth_trajectory(p2_traj) - smooth_trajectory(p1_traj)
    lengths = np.linalg.norm(diffs, axis=1)
    return np.median(lengths)


def extract_inverse_dynamics_features(ts_3d, kp_index, target_frame):
    """Extract inverse dynamics features."""
    feats = {}

    def get_traj(name):
        idx = kp_index.get(name)
        if idx is None:
            return None
        traj = ts_3d[:, idx, :].copy() * FEET_TO_METERS
        for ax in range(3):
            vals = traj[:, ax]
            bad = np.isnan(vals) | np.isinf(vals)
            if np.all(bad):
                return None
            if np.any(bad):
                good = ~bad
                vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
            traj[:, ax] = vals
        return traj

    shoulder = get_traj('right_shoulder')
    elbow = get_traj('right_elbow')
    wrist = get_traj('right_wrist')
    hip_r = get_traj('right_hip')
    knee = get_traj('right_knee')
    ankle = get_traj('right_ankle')
    mid_hip = get_traj('mid_hip')
    fingertip = get_traj('right_third_finger_distal')

    if shoulder is None or elbow is None or wrist is None:
        return None

    # Estimate body mass (use reference since we can't measure precisely)
    body_mass = REFERENCE_BODY_MASS

    # Segment properties
    L_upper_arm = compute_segment_length(shoulder, elbow)
    L_forearm = compute_segment_length(elbow, wrist)
    m_upper_arm = SEGMENT_MASS_RATIOS['upper_arm'] * body_mass
    m_forearm = SEGMENT_MASS_RATIOS['forearm'] * body_mass
    m_hand = SEGMENT_MASS_RATIOS['hand'] * body_mass
    I_forearm = m_forearm * (SEGMENT_ROG_RATIOS['forearm'] * L_forearm)**2
    I_upper_arm = m_upper_arm * (SEGMENT_ROG_RATIOS['upper_arm'] * L_upper_arm)**2

    # Joint angles
    elbow_angles = compute_joint_angle_series(
        smooth_trajectory(shoulder), smooth_trajectory(elbow), smooth_trajectory(wrist))
    elbow_omega = safe_savgol(elbow_angles, 11, 3, deriv=1, delta=DT)
    elbow_alpha = safe_savgol(elbow_angles, 11, 3, deriv=2, delta=DT)

    shoulder_angles = None
    shoulder_omega = None
    shoulder_alpha = None
    if hip_r is not None:
        shoulder_angles = compute_joint_angle_series(
            smooth_trajectory(hip_r), smooth_trajectory(shoulder), smooth_trajectory(elbow))
        shoulder_omega = safe_savgol(shoulder_angles, 11, 3, deriv=1, delta=DT)
        shoulder_alpha = safe_savgol(shoulder_angles, 11, 3, deriv=2, delta=DT)

    knee_angles = None
    knee_omega = None
    knee_alpha = None
    if hip_r is not None and knee is not None and ankle is not None:
        knee_angles = compute_joint_angle_series(
            smooth_trajectory(hip_r), smooth_trajectory(knee), smooth_trajectory(ankle))
        knee_omega = safe_savgol(knee_angles, 11, 3, deriv=1, delta=DT)
        knee_alpha = safe_savgol(knee_angles, 11, 3, deriv=2, delta=DT)

    hip_omega = None
    hip_alpha = None
    if mid_hip is not None and hip_r is not None and knee is not None:
        hip_angles = compute_joint_angle_series(
            smooth_trajectory(mid_hip), smooth_trajectory(hip_r), smooth_trajectory(knee))
        hip_omega = safe_savgol(hip_angles, 11, 3, deriv=1, delta=DT)
        hip_alpha = safe_savgol(hip_angles, 11, 3, deriv=2, delta=DT)

    # Torques
    tau_elbow = I_forearm * elbow_alpha
    I_shoulder_eff = I_upper_arm + m_forearm * L_upper_arm**2 + m_hand * L_upper_arm**2
    tau_shoulder = I_shoulder_eff * shoulder_alpha if shoulder_alpha is not None else np.zeros(240)

    # Power
    P_elbow = tau_elbow * elbow_omega
    P_shoulder = tau_shoulder * (shoulder_omega if shoulder_omega is not None else np.zeros(240))

    # 3D angular velocities for KE
    omega_shoulder_3d = compute_angular_velocity_3d(shoulder, elbow)
    omega_elbow_3d = compute_angular_velocity_3d(elbow, wrist)

    # Segment KE
    def segment_ke(p1, p2, mass, I_seg, omega_3d):
        mid = 0.5 * (smooth_trajectory(p1) + smooth_trajectory(p2))
        vel = np.zeros_like(mid)
        for ax in range(3):
            vel[:, ax] = safe_savgol(mid[:, ax], 11, 3, deriv=1, delta=DT)
        v_sq = np.sum(vel**2, axis=1)
        omega_sq = np.sum(omega_3d**2, axis=1)
        return 0.5 * mass * v_sq + 0.5 * I_seg * omega_sq

    KE_upper_arm = segment_ke(shoulder, elbow, m_upper_arm, I_upper_arm, omega_shoulder_3d)
    KE_forearm = segment_ke(elbow, wrist, m_forearm, I_forearm, omega_elbow_3d)
    KE_arm = KE_upper_arm + KE_forearm

    # Lower body KE
    KE_lower = np.zeros(240)
    if hip_r is not None and knee is not None:
        m_thigh = SEGMENT_MASS_RATIOS['thigh'] * body_mass
        L_thigh = compute_segment_length(hip_r, knee)
        I_thigh = m_thigh * (SEGMENT_ROG_RATIOS['thigh'] * L_thigh)**2
        omega_hip_3d = compute_angular_velocity_3d(hip_r, knee)
        KE_lower += segment_ke(hip_r, knee, m_thigh, I_thigh, omega_hip_3d)
    if knee is not None and ankle is not None:
        m_shank = SEGMENT_MASS_RATIOS['shank'] * body_mass
        L_shank = compute_segment_length(knee, ankle)
        I_shank = m_shank * (SEGMENT_ROG_RATIOS['shank'] * L_shank)**2
        omega_knee_3d = compute_angular_velocity_3d(knee, ankle)
        KE_lower += segment_ke(knee, ankle, m_shank, I_shank, omega_knee_3d)

    f = int(np.clip(target_frame, 5, 234))
    propulsion_start = max(80, f - 60)
    propulsion_end = min(f + 10, 235)

    # Cumulative energy
    E_elbow_cum = np.cumsum(np.abs(P_elbow)) * DT
    E_shoulder_cum = np.cumsum(np.abs(P_shoulder)) * DT

    # Peak power timing
    P_knee = np.zeros(240)
    P_hip = np.zeros(240)
    if knee_alpha is not None and knee_omega is not None:
        m_shank_tmp = SEGMENT_MASS_RATIOS['shank'] * body_mass
        L_shank_tmp = compute_segment_length(knee, ankle) if ankle is not None else 0.42
        I_shank_tmp = m_shank_tmp * (SEGMENT_ROG_RATIOS['shank'] * L_shank_tmp)**2
        P_knee = I_shank_tmp * knee_alpha * knee_omega
    if hip_alpha is not None and hip_omega is not None:
        m_thigh_tmp = SEGMENT_MASS_RATIOS['thigh'] * body_mass
        L_thigh_tmp = compute_segment_length(hip_r, knee) if (hip_r is not None and knee is not None) else 0.45
        I_thigh_tmp = m_thigh_tmp * (SEGMENT_ROG_RATIOS['thigh'] * L_thigh_tmp)**2
        P_hip = I_thigh_tmp * hip_alpha * hip_omega

    pk_knee = propulsion_start + np.argmax(P_knee[propulsion_start:propulsion_end])
    pk_hip = propulsion_start + np.argmax(P_hip[propulsion_start:propulsion_end])
    pk_shoulder = propulsion_start + np.argmax(P_shoulder[propulsion_start:propulsion_end])
    pk_elbow = propulsion_start + np.argmax(P_elbow[propulsion_start:propulsion_end])

    # ===================================================================
    # FEATURES - carefully chosen set
    # ===================================================================

    # --- Torque features ---
    feats['tau_elbow_at_f'] = tau_elbow[f]
    feats['tau_elbow_peak'] = np.max(np.abs(tau_elbow[propulsion_start:propulsion_end]))
    feats['tau_shoulder_at_f'] = tau_shoulder[f]
    feats['tau_shoulder_peak'] = np.max(np.abs(tau_shoulder[propulsion_start:propulsion_end]))

    # --- Power features ---
    feats['P_elbow_at_f'] = P_elbow[f]
    feats['P_elbow_peak'] = np.max(P_elbow[propulsion_start:propulsion_end])
    feats['P_shoulder_at_f'] = P_shoulder[f]
    feats['P_shoulder_peak'] = np.max(P_shoulder[propulsion_start:propulsion_end])
    feats['P_arm_peak'] = np.max((P_elbow + P_shoulder)[propulsion_start:propulsion_end])

    # --- KE features ---
    feats['KE_arm_at_f'] = KE_arm[f]
    feats['KE_arm_peak'] = np.max(KE_arm[propulsion_start:propulsion_end])
    feats['KE_forearm_at_f'] = KE_forearm[f]
    feats['KE_upper_arm_at_f'] = KE_upper_arm[f]
    feats['KE_lower_peak'] = np.max(KE_lower[propulsion_start:propulsion_end])
    feats['KE_transfer_ratio'] = KE_arm[f] / (np.max(KE_lower[propulsion_start:propulsion_end]) + 1e-9)
    feats['KE_distal_proximal'] = KE_forearm[f] / (KE_upper_arm[f] + 1e-9)

    # Rate of arm KE change
    dKE = safe_savgol(KE_arm, 11, 3, deriv=1, delta=DT)
    feats['dKE_arm_at_f'] = dKE[f]
    feats['dKE_arm_peak'] = np.max(dKE[propulsion_start:propulsion_end])

    # --- Cumulative energy ---
    feats['E_elbow_cum'] = E_elbow_cum[f]
    feats['E_shoulder_cum'] = E_shoulder_cum[f]
    E_total = E_elbow_cum[f] + E_shoulder_cum[f] + 1e-9
    feats['E_elbow_frac'] = E_elbow_cum[f] / E_total

    # --- Timing features ---
    feats['pk_shoulder_timing'] = (pk_shoulder - f) / FPS
    feats['pk_elbow_timing'] = (pk_elbow - f) / FPS
    feats['delay_shoulder_to_elbow'] = (pk_elbow - pk_shoulder) / FPS
    feats['delay_knee_to_elbow'] = (pk_elbow - pk_knee) / FPS
    feats['chain_spread'] = (max(pk_knee, pk_hip, pk_shoulder, pk_elbow) -
                             min(pk_knee, pk_hip, pk_shoulder, pk_elbow)) / FPS

    # --- Angular dynamics at target frame ---
    feats['omega_elbow_at_f'] = elbow_omega[f]
    feats['alpha_elbow_at_f'] = elbow_alpha[f]
    if shoulder_omega is not None:
        feats['omega_shoulder_at_f'] = shoulder_omega[f]
    else:
        feats['omega_shoulder_at_f'] = 0.0

    # --- Impulse ---
    feats['impulse_elbow'] = np.sum(np.abs(tau_elbow[propulsion_start:propulsion_end])) * DT
    feats['impulse_shoulder'] = np.sum(np.abs(tau_shoulder[propulsion_start:propulsion_end])) * DT

    return feats


def extract_hc_features(ts_3d, ts_hr, kp_index, release_frame, frame):
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

    all_hc = []
    all_dyn = []
    rfs = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        rfs.append(rf)
        hc = extract_hc_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_hc.append(hc)
        dyn = extract_inverse_dynamics_features(ts_3d, kp_index, frame)
        all_dyn.append(dyn if dyn is not None else {})

    X_hc = np.nan_to_num(np.array(all_hc, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    all_keys = sorted(set().union(*[d.keys() for d in all_dyn if d]))
    X_dyn = np.zeros((n, len(all_keys)), dtype=np.float32)
    for i, d in enumerate(all_dyn):
        for j, k in enumerate(all_keys):
            X_dyn[i, j] = d.get(k, 0.0)
    X_dyn = np.nan_to_num(X_dyn, nan=0.0, posinf=0.0, neginf=0.0)

    return X_hc, X_dyn, np.array(rfs), all_keys


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


def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
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


def select_nonredundant_features(X_hc, X_dyn, dyn_keys, y, max_features=10):
    """Select dynamics features that are least redundant with HC features."""
    # For each dynamics feature, compute:
    # 1. Correlation with target
    # 2. Maximum correlation with any HC feature (redundancy)
    # Score = |corr_target| / (max_corr_hc + 0.1)

    scores = []
    for j, key in enumerate(dyn_keys):
        r_target = abs(np.corrcoef(X_dyn[:, j], y)[0, 1])
        if np.isnan(r_target):
            r_target = 0.0

        # Max correlation with HC features
        max_r_hc = 0.0
        for k in range(X_hc.shape[1]):
            r = abs(np.corrcoef(X_dyn[:, j], X_hc[:, k])[0, 1])
            if not np.isnan(r) and r > max_r_hc:
                max_r_hc = r

        novelty_score = r_target / (max_r_hc + 0.1)
        scores.append((j, key, r_target, max_r_hc, novelty_score))

    scores.sort(key=lambda x: x[4], reverse=True)
    selected_idx = [s[0] for s in scores[:max_features]]
    return selected_idx, scores[:max_features]


def main():
    t0 = time.time()
    print("=" * 70)
    print("INVERSE DYNAMICS V2: SURGICAL FEATURE INTEGRATION")
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

    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc, X_train_dyn, rf_train, dyn_keys = extract_all_features(train_data, target)
        X_test_hc, X_test_dyn, rf_test, _ = extract_all_features(test_data, target)

        y_raw = y_train[:, target_idx[target]]
        y_target = y_scaled[target]

        # PLS augmentation on HC
        print("  Adding PLS...")
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        print(f"  HC+PLS features: {X_train_aug.shape[1]}")
        print(f"  Dynamics features: {X_train_dyn.shape[1]}")

        # --- Strategy A: Baseline HC+PLS ---
        print("\n  [A] HC+PLS baseline...")
        oof_a, test_a = locally_weighted_prediction(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_a = np.mean((oof_a - y_target) ** 2)
        print(f"      LOO MSE: {mse_a:.6f}")

        # --- Strategy B: Dynamics-only prediction ---
        print("  [B] Dynamics-only prediction...")
        oof_b, test_b = locally_weighted_prediction(
            X_train_dyn, y_target, X_test_dyn, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_b = np.mean((oof_b - y_target) ** 2)
        print(f"      LOO MSE: {mse_b:.6f}")

        # --- Strategy C: Select non-redundant dynamics features + HC+PLS ---
        print("  [C] Non-redundant dynamics + HC+PLS...")
        selected_idx, top_scores = select_nonredundant_features(
            X_train_aug, X_train_dyn, dyn_keys, y_raw, max_features=8)
        print(f"      Selected {len(selected_idx)} features:")
        for idx, key, r_target, r_hc, score in top_scores:
            print(f"        {key:35s}: r_target={r_target:.4f}, max_r_hc={r_hc:.4f}, novelty={score:.4f}")

        X_train_sel = np.hstack([X_train_aug, X_train_dyn[:, selected_idx]])
        X_test_sel = np.hstack([X_test_aug, X_test_dyn[:, selected_idx]])

        oof_c, test_c = locally_weighted_prediction(
            X_train_sel, y_target, X_test_sel, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_c = np.mean((oof_c - y_target) ** 2)
        print(f"      LOO MSE: {mse_c:.6f}")

        # Try with fewer features (5)
        selected_idx_5, _ = select_nonredundant_features(
            X_train_aug, X_train_dyn, dyn_keys, y_raw, max_features=5)
        X_train_sel5 = np.hstack([X_train_aug, X_train_dyn[:, selected_idx_5]])
        X_test_sel5 = np.hstack([X_test_aug, X_test_dyn[:, selected_idx_5]])
        oof_c5, test_c5 = locally_weighted_prediction(
            X_train_sel5, y_target, X_test_sel5, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_c5 = np.mean((oof_c5 - y_target) ** 2)
        print(f"      top-5 LOO MSE: {mse_c5:.6f}")

        # Try with 3
        selected_idx_3, _ = select_nonredundant_features(
            X_train_aug, X_train_dyn, dyn_keys, y_raw, max_features=3)
        X_train_sel3 = np.hstack([X_train_aug, X_train_dyn[:, selected_idx_3]])
        X_test_sel3 = np.hstack([X_test_aug, X_test_dyn[:, selected_idx_3]])
        oof_c3, test_c3 = locally_weighted_prediction(
            X_train_sel3, y_target, X_test_sel3, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_c3 = np.mean((oof_c3 - y_target) ** 2)
        print(f"      top-3 LOO MSE: {mse_c3:.6f}")

        # --- Strategy D: Prediction-level blend of A and B ---
        print("  [D] Prediction-level blending (HC + Dynamics)...")
        best_blend = 0.0
        best_blend_mse = mse_a
        best_blend_oof = oof_a
        best_blend_test = test_a
        for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            oof_d = (1-w) * oof_a + w * oof_b
            mse_d = np.mean((oof_d - y_target) ** 2)
            print(f"      w={w:.2f}: LOO MSE={mse_d:.6f}")
            if mse_d < best_blend_mse:
                best_blend_mse = mse_d
                best_blend = w
                best_blend_oof = oof_d
                best_blend_test = (1-w) * test_a + w * test_b

        print(f"\n  {target} SUMMARY:")
        print(f"    [A] HC+PLS baseline:     LOO MSE={mse_a:.6f}")
        print(f"    [B] Dynamics only:        LOO MSE={mse_b:.6f}")
        print(f"    [C] Top-8 non-redundant:  LOO MSE={mse_c:.6f}  ({(mse_c-mse_a)/mse_a*100:+.2f}%)")
        print(f"    [C] Top-5 non-redundant:  LOO MSE={mse_c5:.6f}  ({(mse_c5-mse_a)/mse_a*100:+.2f}%)")
        print(f"    [C] Top-3 non-redundant:  LOO MSE={mse_c3:.6f}  ({(mse_c3-mse_a)/mse_a*100:+.2f}%)")
        print(f"    [D] Best blend (w={best_blend:.2f}):  LOO MSE={best_blend_mse:.6f}  ({(best_blend_mse-mse_a)/mse_a*100:+.2f}%)")

        # Diversity
        r_div_b = np.corrcoef(test_a, test_b)[0, 1]
        print(f"    Diversity: dynamics vs HC: r={r_div_b:.4f}")

        # Determine best approach for this target
        candidates = {
            'hc_baseline': (mse_a, oof_a, test_a),
            'top_8_sel': (mse_c, oof_c, test_c),
            'top_5_sel': (mse_c5, oof_c5, test_c5),
            'top_3_sel': (mse_c3, oof_c3, test_c3),
            'pred_blend': (best_blend_mse, best_blend_oof, best_blend_test),
        }
        best_name = min(candidates, key=lambda x: candidates[x][0])
        print(f"    BEST: {best_name}")

        results[target] = {
            'mse_a': mse_a,
            'mse_b': mse_b,
            'test_a': test_a,
            'test_b': test_b,
            'best_name': best_name,
            'best_mse': candidates[best_name][0],
            'best_test': candidates[best_name][2],
            'blend_w': best_blend,
            'r_diversity': r_div_b,
        }

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    total_a = sum(results[t]['mse_a'] for t in TARGETS) / 3
    total_best = sum(results[t]['best_mse'] for t in TARGETS) / 3
    for target in TARGETS:
        r = results[target]
        print(f"  {target:12s}: baseline={r['mse_a']:.6f}  best={r['best_mse']:.6f} ({r['best_name']})  diversity={r['r_diversity']:.4f}")
    print(f"  MEAN:         baseline={total_a:.6f}  best={total_best:.6f}")
    print(f"  Change: {(total_best - total_a) / total_a * 100:+.2f}%")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    print("\n  Correlation with Sub 784 and Sub 1350:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r_784 = np.corrcoef(sub_784[col].values, results[target]['best_test'])[0, 1]
        r_1350 = np.corrcoef(sub_1350[col].values, results[target]['best_test'])[0, 1]
        print(f"    {target}: r_784={r_784:.4f}, r_1350={r_1350:.4f}")

    # Best predictions per target - blend with Sub 784
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "standard"),
        (0.00, 0.20, 0.30, "conservative"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*results['angle']['best_test']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*results['depth']['best_test']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*results['left_right']['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")

    # Dynamics predictions blend with Sub 1350
    for w in [0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_1350[col] + w*results[target]['test_b']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {w*100:.0f}% dynamics-only blend with Sub 1350")

    # Dynamics-only standalone blend with Sub 784
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    blended['scaled_depth'] = 0.80*sub_784['scaled_depth'] + 0.20*results['depth']['test_b']
    blended['scaled_left_right'] = 0.70*sub_784['scaled_left_right'] + 0.30*results['left_right']['test_b']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: Dynamics-only depth=0.20 lr=0.30 with Sub 784")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
