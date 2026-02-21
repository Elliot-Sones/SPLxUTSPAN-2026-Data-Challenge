"""
Inverse Dynamics Feature Extraction

Computes joint torques, power, kinetic energy, and energy transfer features
from keypoint motion data. These features capture the FORCES and ENERGIES
that produce ball velocity, not just positions.

Key features:
1. Simplified joint torques: tau = I * alpha (moment of inertia x angular acceleration)
2. Joint power: P = tau * omega (torque x angular velocity)
3. Segment kinetic energy: KE = 0.5 * m * v^2 + 0.5 * I * omega^2
4. Kinetic chain energy transfer ratios
5. Peak timing and sequencing through the chain

Uses standard anthropometric mass ratios and estimates segment properties
from keypoint distances.
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
FPS = 60.0
FEET_TO_METERS = 0.3048

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Standard anthropometric segment mass ratios (as fraction of body mass)
# From Winter (2009) "Biomechanics and Motor Control of Human Movement"
# Using a reference body mass of 85 kg (typical male basketball player)
REFERENCE_BODY_MASS = 85.0  # kg

SEGMENT_MASS_RATIOS = {
    'upper_arm': 0.028,    # 2.8% of body mass
    'forearm': 0.016,      # 1.6% of body mass
    'hand': 0.006,         # 0.6% of body mass
    'thigh': 0.100,        # 10.0% of body mass
    'shank': 0.047,        # 4.7% of body mass
    'trunk': 0.497,        # 49.7% of body mass (head+trunk)
}

# Segment radius of gyration ratios (as fraction of segment length)
# For computing moment of inertia: I = m * (rg * L)^2
SEGMENT_ROG_RATIOS = {
    'upper_arm': 0.322,
    'forearm': 0.303,
    'hand': 0.297,
    'thigh': 0.323,
    'shank': 0.302,
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
# HOOP TRANSFORM (same as per_example_pipeline.py)
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


# ==============================================================
# INVERSE DYNAMICS FEATURE EXTRACTION
# ==============================================================

def smooth_trajectory(traj, window=11, polyorder=3):
    """Smooth a 3D trajectory using Savitzky-Golay filter."""
    smoothed = np.zeros_like(traj, dtype=np.float64)
    for ax in range(3):
        smoothed[:, ax] = safe_savgol(traj[:, ax], window, polyorder)
    return smoothed


def compute_joint_angle_series(p1_traj, p2_traj, p3_traj):
    """Compute angle at p2 over all frames. Returns angles in radians."""
    v1 = p1_traj - p2_traj  # (240, 3)
    v2 = p3_traj - p2_traj  # (240, 3)

    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)

    cos_angle = np.clip(dot / (n1 * n2 + 1e-9), -1, 1)
    return np.arccos(cos_angle)


def compute_angular_velocity_3d(parent_traj, child_traj, window=9, polyorder=3):
    """
    Compute angular velocity vector from parent-child joint trajectories.
    v_child = v_parent + omega x r
    omega = (r x (v_child - v_parent)) / |r|^2
    Returns (240, 3) angular velocity vectors.
    """
    # Smooth first to reduce noise
    parent_smooth = smooth_trajectory(parent_traj, window, polyorder)
    child_smooth = smooth_trajectory(child_traj, window, polyorder)

    # Compute velocities using Savgol derivative
    parent_vel = np.zeros_like(parent_smooth)
    child_vel = np.zeros_like(child_smooth)
    for ax in range(3):
        parent_vel[:, ax] = safe_savgol(parent_smooth[:, ax], window, polyorder, deriv=1, delta=DT)
        child_vel[:, ax] = safe_savgol(child_smooth[:, ax], window, polyorder, deriv=1, delta=DT)

    r = child_smooth - parent_smooth  # (240, 3)
    v_rel = child_vel - parent_vel  # (240, 3)
    r_norm_sq = np.sum(r**2, axis=1, keepdims=True) + 1e-9

    omega = np.cross(r, v_rel) / r_norm_sq  # (240, 3)
    return omega


def compute_segment_length(p1_traj, p2_traj):
    """Compute segment length from smoothed trajectories."""
    diffs = smooth_trajectory(p2_traj) - smooth_trajectory(p1_traj)
    lengths = np.linalg.norm(diffs, axis=1)
    # Use median length as the "true" segment length
    return np.median(lengths)


def estimate_body_mass_from_skeleton(ts_3d, kp_index):
    """
    Estimate body mass from skeleton dimensions.
    Uses the relationship between stature and mass (BMI-based approximation).
    Returns estimated mass in kg.
    """
    # Estimate height from head to ankle distance at frame 120
    head_idx = kp_index.get('nose')
    ankle_idx = kp_index.get('right_ankle')
    if head_idx is not None and ankle_idx is not None:
        head_pos = ts_3d[120, head_idx, :]
        ankle_pos = ts_3d[120, ankle_idx, :]
        if not np.any(np.isnan(head_pos)) and not np.any(np.isnan(ankle_pos)):
            height_feet = abs(head_pos[2] - ankle_pos[2]) * 1.05  # Scale up for head top
            height_m = height_feet * FEET_TO_METERS
            if 1.5 < height_m < 2.3:
                # Use BMI of ~24 (athletic male)
                return 24.0 * height_m**2
    return REFERENCE_BODY_MASS


def extract_inverse_dynamics_features(ts_3d, kp_index, target_frame):
    """
    Extract inverse dynamics features from a single shot.

    Features computed:
    1. Joint torques (simplified: tau = I * alpha)
    2. Joint power (P = tau * omega_magnitude)
    3. Segment kinetic energy
    4. Energy transfer ratios
    5. Peak timing and sequencing
    6. Cumulative energy at target frame
    """
    feats = {}

    # Get joint trajectories - convert to meters for physics
    def get_traj(name):
        idx = kp_index.get(name)
        if idx is None:
            return None
        traj = ts_3d[:, idx, :].copy() * FEET_TO_METERS
        # Interpolate NaN
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

    # Essential joints for the kinetic chain
    shoulder = get_traj('right_shoulder')
    elbow = get_traj('right_elbow')
    wrist = get_traj('right_wrist')
    hip_r = get_traj('right_hip')
    knee = get_traj('right_knee')
    ankle = get_traj('right_ankle')
    mid_hip = get_traj('mid_hip')
    neck = get_traj('neck')
    fingertip = get_traj('right_third_finger_distal')

    if shoulder is None or elbow is None or wrist is None:
        return None

    # Estimate body mass
    body_mass = estimate_body_mass_from_skeleton(ts_3d, kp_index)
    feats['body_mass_est'] = body_mass

    # ===================================================================
    # 1. SEGMENT PROPERTIES
    # ===================================================================
    # Segment lengths (meters)
    L_upper_arm = compute_segment_length(shoulder, elbow)
    L_forearm = compute_segment_length(elbow, wrist)
    L_hand = compute_segment_length(wrist, fingertip) if fingertip is not None else 0.19 * FEET_TO_METERS

    # Segment masses (kg)
    m_upper_arm = SEGMENT_MASS_RATIOS['upper_arm'] * body_mass
    m_forearm = SEGMENT_MASS_RATIOS['forearm'] * body_mass
    m_hand = SEGMENT_MASS_RATIOS['hand'] * body_mass

    # Moments of inertia (kg*m^2) - rod model: I = m * (rg * L)^2
    I_upper_arm = m_upper_arm * (SEGMENT_ROG_RATIOS['upper_arm'] * L_upper_arm)**2
    I_forearm = m_forearm * (SEGMENT_ROG_RATIOS['forearm'] * L_forearm)**2
    I_hand = m_hand * (SEGMENT_ROG_RATIOS.get('hand', 0.297) * L_hand)**2

    # Lower body segments
    if hip_r is not None and knee is not None:
        L_thigh = compute_segment_length(hip_r, knee)
        m_thigh = SEGMENT_MASS_RATIOS['thigh'] * body_mass
        I_thigh = m_thigh * (SEGMENT_ROG_RATIOS['thigh'] * L_thigh)**2
    else:
        L_thigh = 0.45
        m_thigh = SEGMENT_MASS_RATIOS['thigh'] * body_mass
        I_thigh = m_thigh * (0.323 * L_thigh)**2

    if knee is not None and ankle is not None:
        L_shank = compute_segment_length(knee, ankle)
        m_shank = SEGMENT_MASS_RATIOS['shank'] * body_mass
        I_shank = m_shank * (SEGMENT_ROG_RATIOS['shank'] * L_shank)**2
    else:
        L_shank = 0.42
        m_shank = SEGMENT_MASS_RATIOS['shank'] * body_mass
        I_shank = m_shank * (0.302 * L_shank)**2

    # ===================================================================
    # 2. ANGULAR VELOCITIES & ACCELERATIONS
    # ===================================================================
    # Compute joint angle time series (scalar angles)
    elbow_angles = compute_joint_angle_series(
        smooth_trajectory(shoulder), smooth_trajectory(elbow), smooth_trajectory(wrist))

    shoulder_angles = None
    if hip_r is not None:
        shoulder_angles = compute_joint_angle_series(
            smooth_trajectory(hip_r), smooth_trajectory(shoulder), smooth_trajectory(elbow))

    knee_angles = None
    if hip_r is not None and knee is not None and ankle is not None:
        knee_angles = compute_joint_angle_series(
            smooth_trajectory(hip_r), smooth_trajectory(knee), smooth_trajectory(ankle))

    hip_angles = None
    if mid_hip is not None and hip_r is not None and knee is not None:
        hip_angles = compute_joint_angle_series(
            smooth_trajectory(mid_hip), smooth_trajectory(hip_r), smooth_trajectory(knee))

    # Scalar angular velocities (d(angle)/dt) using Savgol derivative
    elbow_omega = safe_savgol(elbow_angles, 11, 3, deriv=1, delta=DT)
    elbow_alpha = safe_savgol(elbow_angles, 11, 3, deriv=2, delta=DT)

    shoulder_omega = None
    shoulder_alpha = None
    if shoulder_angles is not None:
        shoulder_omega = safe_savgol(shoulder_angles, 11, 3, deriv=1, delta=DT)
        shoulder_alpha = safe_savgol(shoulder_angles, 11, 3, deriv=2, delta=DT)

    knee_omega = None
    knee_alpha = None
    if knee_angles is not None:
        knee_omega = safe_savgol(knee_angles, 11, 3, deriv=1, delta=DT)
        knee_alpha = safe_savgol(knee_angles, 11, 3, deriv=2, delta=DT)

    hip_omega = None
    hip_alpha = None
    if hip_angles is not None:
        hip_omega = safe_savgol(hip_angles, 11, 3, deriv=1, delta=DT)
        hip_alpha = safe_savgol(hip_angles, 11, 3, deriv=2, delta=DT)

    # 3D angular velocities for energy computation
    omega_shoulder_3d = compute_angular_velocity_3d(shoulder, elbow)
    omega_elbow_3d = compute_angular_velocity_3d(elbow, wrist)
    omega_wrist_3d = None
    if fingertip is not None:
        omega_wrist_3d = compute_angular_velocity_3d(wrist, fingertip)

    # ===================================================================
    # 3. JOINT TORQUES (simplified: tau = I * alpha)
    # ===================================================================
    # Elbow torque
    tau_elbow = I_forearm * elbow_alpha  # (240,)

    # Shoulder torque (includes distal segment inertia coupling)
    # Simplified: tau_shoulder = I_upper_arm * alpha_shoulder + m_forearm * L_upper_arm * a_cm_forearm (projected)
    # Further simplified: tau_shoulder approx = (I_upper_arm + m_forearm * L_upper_arm^2) * alpha_shoulder
    if shoulder_alpha is not None:
        I_shoulder_eff = I_upper_arm + m_forearm * L_upper_arm**2 + m_hand * L_upper_arm**2
        tau_shoulder = I_shoulder_eff * shoulder_alpha
    else:
        tau_shoulder = np.zeros(240)

    # Knee torque
    if knee_alpha is not None:
        I_knee_eff = I_shank + 0.3 * m_shank * L_shank**2  # simplified
        tau_knee = I_knee_eff * knee_alpha
    else:
        tau_knee = np.zeros(240)

    # Hip torque
    if hip_alpha is not None:
        I_hip_eff = I_thigh + m_shank * L_thigh**2
        tau_hip = I_hip_eff * hip_alpha
    else:
        tau_hip = np.zeros(240)

    # ===================================================================
    # 4. JOINT POWER (P = tau * omega)
    # ===================================================================
    P_elbow = tau_elbow * elbow_omega
    P_shoulder = tau_shoulder * (shoulder_omega if shoulder_omega is not None else np.zeros(240))
    P_knee = tau_knee * (knee_omega if knee_omega is not None else np.zeros(240))
    P_hip = tau_hip * (hip_omega if hip_omega is not None else np.zeros(240))

    # ===================================================================
    # 5. SEGMENT KINETIC ENERGY
    # ===================================================================
    # Translational KE: 0.5 * m * v^2
    # Rotational KE: 0.5 * I * omega^2

    def compute_segment_ke(p1_traj, p2_traj, mass, I_seg, omega_3d=None):
        """Compute total kinetic energy of a segment."""
        # Segment CoM velocity (approximate as midpoint velocity)
        midpoint = 0.5 * (smooth_trajectory(p1_traj) + smooth_trajectory(p2_traj))
        vel = np.zeros_like(midpoint)
        for ax in range(3):
            vel[:, ax] = safe_savgol(midpoint[:, ax], 11, 3, deriv=1, delta=DT)
        v_sq = np.sum(vel**2, axis=1)
        KE_trans = 0.5 * mass * v_sq

        # Rotational KE
        if omega_3d is not None:
            omega_mag_sq = np.sum(omega_3d**2, axis=1)
            KE_rot = 0.5 * I_seg * omega_mag_sq
        else:
            KE_rot = np.zeros(240)

        return KE_trans + KE_rot, KE_trans, KE_rot

    KE_upper_arm, KE_ua_trans, KE_ua_rot = compute_segment_ke(
        shoulder, elbow, m_upper_arm, I_upper_arm, omega_shoulder_3d)
    KE_forearm, KE_fa_trans, KE_fa_rot = compute_segment_ke(
        elbow, wrist, m_forearm, I_forearm, omega_elbow_3d)

    KE_hand = np.zeros(240)
    if fingertip is not None and omega_wrist_3d is not None:
        KE_hand, _, _ = compute_segment_ke(
            wrist, fingertip, m_hand, I_hand, omega_wrist_3d)

    KE_arm_total = KE_upper_arm + KE_forearm + KE_hand

    # Lower body KE
    KE_thigh = np.zeros(240)
    KE_shank = np.zeros(240)
    if hip_r is not None and knee is not None:
        omega_hip_3d = compute_angular_velocity_3d(hip_r, knee)
        KE_thigh, _, _ = compute_segment_ke(hip_r, knee, m_thigh, I_thigh, omega_hip_3d)
    if knee is not None and ankle is not None:
        omega_knee_3d = compute_angular_velocity_3d(knee, ankle)
        KE_shank, _, _ = compute_segment_ke(knee, ankle, m_shank, I_shank, omega_knee_3d)

    KE_lower = KE_thigh + KE_shank

    # ===================================================================
    # 6. CUMULATIVE ENERGY (integral of power)
    # ===================================================================
    E_elbow_cum = np.cumsum(np.abs(P_elbow)) * DT
    E_shoulder_cum = np.cumsum(np.abs(P_shoulder)) * DT
    E_knee_cum = np.cumsum(np.abs(P_knee)) * DT
    E_hip_cum = np.cumsum(np.abs(P_hip)) * DT

    # ===================================================================
    # 7. EXTRACT FEATURES AT TARGET FRAME AND KEY MOMENTS
    # ===================================================================
    f = int(np.clip(target_frame, 5, 234))

    # Window around target frame for peak detection
    propulsion_start = max(80, f - 60)
    propulsion_end = min(f + 10, 235)

    # --- Torque features ---
    feats['tau_elbow_at_f'] = tau_elbow[f]
    feats['tau_elbow_peak'] = np.max(np.abs(tau_elbow[propulsion_start:propulsion_end]))
    feats['tau_elbow_mean_propulsion'] = np.mean(np.abs(tau_elbow[propulsion_start:propulsion_end]))

    feats['tau_shoulder_at_f'] = tau_shoulder[f]
    feats['tau_shoulder_peak'] = np.max(np.abs(tau_shoulder[propulsion_start:propulsion_end]))
    feats['tau_shoulder_mean_propulsion'] = np.mean(np.abs(tau_shoulder[propulsion_start:propulsion_end]))

    feats['tau_knee_peak'] = np.max(np.abs(tau_knee[propulsion_start:propulsion_end]))
    feats['tau_hip_peak'] = np.max(np.abs(tau_hip[propulsion_start:propulsion_end]))

    # Torque ratios
    tau_total = np.abs(tau_elbow[f]) + np.abs(tau_shoulder[f]) + 1e-9
    feats['tau_elbow_ratio'] = np.abs(tau_elbow[f]) / tau_total
    feats['tau_shoulder_ratio'] = np.abs(tau_shoulder[f]) / tau_total

    # --- Power features ---
    feats['P_elbow_at_f'] = P_elbow[f]
    feats['P_elbow_peak'] = np.max(P_elbow[propulsion_start:propulsion_end])
    feats['P_elbow_mean_propulsion'] = np.mean(P_elbow[propulsion_start:propulsion_end])

    feats['P_shoulder_at_f'] = P_shoulder[f]
    feats['P_shoulder_peak'] = np.max(P_shoulder[propulsion_start:propulsion_end])
    feats['P_shoulder_mean_propulsion'] = np.mean(P_shoulder[propulsion_start:propulsion_end])

    feats['P_knee_peak'] = np.max(P_knee[propulsion_start:propulsion_end])
    feats['P_hip_peak'] = np.max(P_hip[propulsion_start:propulsion_end])

    # Total arm power
    P_arm_total = P_elbow + P_shoulder
    feats['P_arm_total_at_f'] = P_arm_total[f]
    feats['P_arm_total_peak'] = np.max(P_arm_total[propulsion_start:propulsion_end])

    # Power ratios (how much each joint contributes to total)
    P_total = np.abs(P_elbow[f]) + np.abs(P_shoulder[f]) + 1e-9
    feats['P_elbow_ratio'] = np.abs(P_elbow[f]) / P_total
    feats['P_shoulder_ratio'] = np.abs(P_shoulder[f]) / P_total

    # --- Kinetic energy features ---
    feats['KE_arm_at_f'] = KE_arm_total[f]
    feats['KE_arm_peak'] = np.max(KE_arm_total[propulsion_start:propulsion_end])
    feats['KE_upper_arm_at_f'] = KE_upper_arm[f]
    feats['KE_forearm_at_f'] = KE_forearm[f]
    feats['KE_hand_at_f'] = KE_hand[f]

    feats['KE_lower_at_f'] = KE_lower[f]
    feats['KE_lower_peak'] = np.max(KE_lower[propulsion_start:propulsion_end])

    # Energy transfer ratio: arm KE / lower body KE
    KE_lower_peak = np.max(KE_lower[propulsion_start:propulsion_end])
    feats['KE_transfer_ratio'] = KE_arm_total[f] / (KE_lower_peak + 1e-9)

    # Distal-to-proximal energy ratio (higher = better kinetic chain)
    feats['KE_distal_proximal_ratio'] = (KE_forearm[f] + KE_hand[f]) / (KE_upper_arm[f] + 1e-9)

    # --- Cumulative energy features ---
    feats['E_elbow_cumulative'] = E_elbow_cum[f]
    feats['E_shoulder_cumulative'] = E_shoulder_cum[f]
    feats['E_knee_cumulative'] = E_knee_cum[f]
    feats['E_hip_cumulative'] = E_hip_cum[f]

    E_total_cum = E_elbow_cum[f] + E_shoulder_cum[f] + E_knee_cum[f] + E_hip_cum[f] + 1e-9
    feats['E_elbow_fraction'] = E_elbow_cum[f] / E_total_cum
    feats['E_shoulder_fraction'] = E_shoulder_cum[f] / E_total_cum
    feats['E_lower_fraction'] = (E_knee_cum[f] + E_hip_cum[f]) / E_total_cum

    # --- Peak timing features (kinetic chain sequencing) ---
    pk_knee = propulsion_start + np.argmax(P_knee[propulsion_start:propulsion_end])
    pk_hip = propulsion_start + np.argmax(P_hip[propulsion_start:propulsion_end])
    pk_shoulder = propulsion_start + np.argmax(P_shoulder[propulsion_start:propulsion_end])
    pk_elbow = propulsion_start + np.argmax(P_elbow[propulsion_start:propulsion_end])

    # Peak power timing (in frames relative to target frame)
    feats['pk_knee_timing'] = (pk_knee - f) / FPS
    feats['pk_hip_timing'] = (pk_hip - f) / FPS
    feats['pk_shoulder_timing'] = (pk_shoulder - f) / FPS
    feats['pk_elbow_timing'] = (pk_elbow - f) / FPS

    # Inter-joint timing delays
    feats['delay_knee_to_hip'] = (pk_hip - pk_knee) / FPS
    feats['delay_hip_to_shoulder'] = (pk_shoulder - pk_hip) / FPS
    feats['delay_shoulder_to_elbow'] = (pk_elbow - pk_shoulder) / FPS
    feats['delay_knee_to_elbow'] = (pk_elbow - pk_knee) / FPS

    # Is the kinetic chain in correct proximal-to-distal order?
    chain_order = [pk_knee, pk_hip, pk_shoulder, pk_elbow]
    feats['chain_sequential'] = 1.0 if chain_order == sorted(chain_order) else 0.0

    # Chain timing spread (tighter = more coordinated)
    feats['chain_timing_spread'] = (max(chain_order) - min(chain_order)) / FPS

    # --- Angular acceleration at target frame ---
    feats['alpha_elbow_at_f'] = elbow_alpha[f]
    if shoulder_alpha is not None:
        feats['alpha_shoulder_at_f'] = shoulder_alpha[f]
    else:
        feats['alpha_shoulder_at_f'] = 0.0

    # Angular velocity at target frame
    feats['omega_elbow_at_f'] = elbow_omega[f]
    if shoulder_omega is not None:
        feats['omega_shoulder_at_f'] = shoulder_omega[f]
    else:
        feats['omega_shoulder_at_f'] = 0.0

    # --- Torque impulse (integral of torque over propulsion phase) ---
    feats['impulse_elbow'] = np.sum(np.abs(tau_elbow[propulsion_start:propulsion_end])) * DT
    feats['impulse_shoulder'] = np.sum(np.abs(tau_shoulder[propulsion_start:propulsion_end])) * DT
    feats['impulse_total'] = feats['impulse_elbow'] + feats['impulse_shoulder']

    # --- Rate of energy change at target frame ---
    dKE_arm = safe_savgol(KE_arm_total, 11, 3, deriv=1, delta=DT)
    feats['dKE_arm_at_f'] = dKE_arm[f]
    feats['dKE_arm_peak'] = np.max(dKE_arm[propulsion_start:propulsion_end])

    return feats


# ==============================================================
# EXISTING HC FEATURES (from per_example_pipeline)
# ==============================================================

def extract_hc_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract compact HC feature set at a specific frame (from per_example_pipeline)."""
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


# ==============================================================
# FEATURE EXTRACTION PIPELINE
# ==============================================================

def extract_all_features(data, target, include_dynamics=True):
    """Extract HC + inverse dynamics features for all shots."""
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']

    all_hc_feats = []
    all_dyn_feats = []
    release_frames = []

    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)

        # HC features
        hc = extract_hc_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_hc_feats.append(hc)

        # Inverse dynamics features
        if include_dynamics:
            dyn = extract_inverse_dynamics_features(ts_3d, kp_index, frame)
            if dyn is not None:
                all_dyn_feats.append(dyn)
            else:
                # Fallback: zeros
                all_dyn_feats.append({})

    X_hc = np.array(all_hc_feats, dtype=np.float32)
    X_hc = np.nan_to_num(X_hc, nan=0.0, posinf=0.0, neginf=0.0)

    if include_dynamics and all_dyn_feats:
        # Convert dict features to array
        all_keys = sorted(set().union(*[d.keys() for d in all_dyn_feats if d]))
        X_dyn = np.zeros((n, len(all_keys)), dtype=np.float32)
        for i, d in enumerate(all_dyn_feats):
            for j, k in enumerate(all_keys):
                X_dyn[i, j] = d.get(k, 0.0)
        X_dyn = np.nan_to_num(X_dyn, nan=0.0, posinf=0.0, neginf=0.0)
        return X_hc, X_dyn, np.array(release_frames), all_keys
    else:
        return X_hc, None, np.array(release_frames), []


# ==============================================================
# PLS AUGMENTATION
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
# LOCALLY WEIGHTED REGRESSION
# ==============================================================

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
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

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
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("INVERSE DYNAMICS FEATURE EXTRACTION + PER-EXAMPLE PIPELINE")
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

    # ============================================================
    # FEATURE EXTRACTION
    # ============================================================
    results = {}
    dyn_feature_names = None

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        print("  Extracting HC + inverse dynamics features...")
        X_train_hc, X_train_dyn, rf_train, dyn_keys = extract_all_features(train_data, target, include_dynamics=True)
        X_test_hc, X_test_dyn, rf_test, _ = extract_all_features(test_data, target, include_dynamics=True)

        if dyn_feature_names is None:
            dyn_feature_names = dyn_keys
            print(f"  Inverse dynamics features: {len(dyn_keys)}")
            for k in dyn_keys[:10]:
                print(f"    {k}")
            if len(dyn_keys) > 10:
                print(f"    ... and {len(dyn_keys)-10} more")

        print(f"  HC features: {X_train_hc.shape[1]}")
        print(f"  Dynamics features: {X_train_dyn.shape[1]}")

        # Correlation analysis of dynamics features with target
        y_raw = y_train[:, target_idx[target]]
        y_target = y_scaled[target]

        print(f"\n  Top dynamics feature correlations with {target}:")
        corrs = []
        for j, k in enumerate(dyn_keys):
            r = np.corrcoef(X_train_dyn[:, j], y_raw)[0, 1]
            if not np.isnan(r):
                corrs.append((k, r))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for k, r in corrs[:10]:
            print(f"    {k:40s}: r={r:+.4f}")

        # PLS augmentation
        print("\n  Adding PLS components...")
        X_train_aug_hc, X_test_aug_hc = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        # Combined: HC + PLS + dynamics
        X_train_combined = np.hstack([X_train_aug_hc, X_train_dyn])
        X_test_combined = np.hstack([X_test_aug_hc, X_test_dyn])
        print(f"  Combined features: {X_train_combined.shape[1]}")

        # --- Evaluate: HC+PLS only (baseline, matches Sub 1350) ---
        print("\n  [A] HC+PLS only (baseline)...")
        oof_a, test_a = locally_weighted_prediction(
            X_train_aug_hc, y_target, X_test_aug_hc, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_a = np.mean((oof_a - y_target) ** 2)
        print(f"      LOO MSE: {mse_a:.6f}")

        # --- Evaluate: Dynamics only ---
        print("  [B] Dynamics only...")
        oof_b, test_b = locally_weighted_prediction(
            X_train_dyn, y_target, X_test_dyn, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_b = np.mean((oof_b - y_target) ** 2)
        print(f"      LOO MSE: {mse_b:.6f}")

        # --- Evaluate: HC+PLS+Dynamics combined ---
        print("  [C] HC+PLS+Dynamics combined...")
        oof_c, test_c = locally_weighted_prediction(
            X_train_combined, y_target, X_test_combined, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_c = np.mean((oof_c - y_target) ** 2)
        print(f"      LOO MSE: {mse_c:.6f}")

        # --- Evaluate: bandwidth search on combined ---
        best_bw = 0.5
        best_mse = mse_c
        best_oof = oof_c
        best_test = test_c
        for bw in [0.3, 0.4, 0.6, 0.7]:
            oof_tmp, test_tmp = locally_weighted_prediction(
                X_train_combined, y_target, X_test_combined, pids_train, pids_test,
                bandwidth_quantile=bw, alpha=10.0)
            mse_tmp = np.mean((oof_tmp - y_target) ** 2)
            print(f"      bw={bw:.1f}: LOO MSE={mse_tmp:.6f}")
            if mse_tmp < best_mse:
                best_mse = mse_tmp
                best_bw = bw
                best_oof = oof_tmp
                best_test = test_tmp

        print(f"\n  {target} SUMMARY:")
        print(f"    [A] HC+PLS only:        LOO MSE={mse_a:.6f}")
        print(f"    [B] Dynamics only:       LOO MSE={mse_b:.6f}")
        print(f"    [C] Combined (best bw={best_bw:.1f}): LOO MSE={best_mse:.6f}")

        delta = (best_mse - mse_a) / mse_a * 100
        print(f"    Combined vs baseline: {delta:+.2f}%")

        # Diversity with HC-only
        r_diversity = np.corrcoef(test_a, best_test)[0, 1]
        print(f"    Diversity (r with HC-only): {r_diversity:.4f}")

        results[target] = {
            'mse_hc': mse_a,
            'mse_dyn': mse_b,
            'mse_combined': best_mse,
            'best_bw': best_bw,
            'oof_hc': oof_a,
            'oof_combined': best_oof,
            'test_hc': test_a,
            'test_dyn': test_b,
            'test_combined': best_test,
        }

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")

    total_hc = 0
    total_dyn = 0
    total_combined = 0
    for target in TARGETS:
        r = results[target]
        total_hc += r['mse_hc']
        total_dyn += r['mse_dyn']
        total_combined += r['mse_combined']
        delta = (r['mse_combined'] - r['mse_hc']) / r['mse_hc'] * 100
        print(f"  {target:12s}: HC={r['mse_hc']:.6f}  Dyn={r['mse_dyn']:.6f}  Comb={r['mse_combined']:.6f}  ({delta:+.2f}%)")

    print(f"  {'MEAN':12s}: HC={total_hc/3:.6f}  Dyn={total_dyn/3:.6f}  Comb={total_combined/3:.6f}")

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
        r_784 = np.corrcoef(sub_784[col].values, results[target]['test_combined'])[0, 1]
        r_1350 = np.corrcoef(sub_1350[col].values, results[target]['test_combined'])[0, 1]
        print(f"    {target}: r_784={r_784:.4f}, r_1350={r_1350:.4f}")

    # --- Submission 1: Standalone combined ---
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle']['test_combined'],
        'scaled_depth': results['depth']['test_combined'],
        'scaled_left_right': results['left_right']['test_combined'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: STANDALONE combined (HC+PLS+Dynamics)")

    # --- Blends with Sub 784 ---
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "standard"),
        (0.00, 0.20, 0.30, "conservative"),
        (0.00, 0.40, 0.60, "aggressive"),
        (0.10, 0.30, 0.50, "with angle"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*results['angle']['test_combined']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*results['depth']['test_combined']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*results['left_right']['test_combined']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # --- Blends with Sub 1350 ---
    for w, desc in [(0.10, "10% dynamics"), (0.20, "20% dynamics"), (0.30, "30% dynamics")]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_1350[col] + w*results[target]['test_combined']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: {w*100:.0f}% dynamics blend with Sub 1350 ({desc})")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # --- Dynamics-only blend with Sub 784 ---
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    blended['scaled_depth'] = 0.70*sub_784['scaled_depth'] + 0.30*results['depth']['test_dyn']
    blended['scaled_left_right'] = 0.50*sub_784['scaled_left_right'] + 0.50*results['left_right']['test_dyn']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: Dynamics-only blend with Sub 784 (dw=0.30, lw=0.50)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
