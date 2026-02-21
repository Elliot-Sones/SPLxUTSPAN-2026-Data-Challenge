"""
Biomechanical Feature Extraction + Enhanced Blend with Sub 784

Extracts ~35 new biomechanical features from joint angular velocities, timing
delays, trunk lean, CoM velocity, coordination variability, etc. These features
capture fundamentally different signal from position-based hoop-relative features.

Integrates with existing hoop-relative feature pipeline and blends with Sub 784.
"""

import json
import sys
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
FRAME_RATE = 60
DT = 1.0 / FRAME_RATE

# Target-specific extraction frames (from physics frame analysis)
TARGET_FRAMES = {'angle': 153, 'depth': 150, 'left_right': 170}
# Propulsion phase
PROPULSION_START = 120
PROPULSION_END = 170


def get_next_submission_number():
    """Atomically get the next submission number using a lock file."""
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)

    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
            placeholder.touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_raw_data():
    """Load train and test, returning raw timeseries + metadata."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp = len(keypoint_cols)

    keypoint_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoint_names.append(col[:-2])

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Keypoints: {len(keypoint_names)}")

    def process(df, is_train=True):
        n = len(df)
        X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)
        X_3d = np.zeros((n, 240, len(keypoint_names), 3), dtype=np.float32)

        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr
                kp_idx = col_i // 3
                coord_idx = col_i % 3
                X_3d[idx, :, kp_idx, coord_idx] = arr

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            'X_raw': X_raw,
            'X_3d': X_3d,
            'pids': np.array(pids),
            'ids': np.array(ids),
            'keypoint_names': keypoint_names,
        }
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    train = process(train_df, True)
    test = process(test_df, False)
    return train, test


# ============================================================
# HOOP-RELATIVE FEATURES (from target_specific_blend.py)
# ============================================================

def compute_hoop_relative_transform(player_pos):
    hoop_2d = HOOP_POS[:2]
    player_2d = player_pos[:2]
    forward = hoop_2d - player_2d
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, -1.0])
    else:
        forward = forward / forward_norm
    lateral = np.array([-forward[1], forward[0]])
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]
    R[0, 1] = forward[1]
    R[1, 0] = lateral[0]
    R[1, 1] = lateral[1]
    return R, player_pos


def extract_hoop_relative_features(ts_3d, keypoint_names, pid):
    """Extract hoop-relative + original coordinate features (same as target_specific_blend.py)."""
    feats = {}
    feats['participant_id'] = pid
    kp_index = {name: i for i, name in enumerate(keypoint_names)}

    mh_idx = kp_index.get('mid_hip')
    if mh_idx is not None:
        player_pos = np.nanmean(ts_3d[:10, mh_idx, :], axis=0)
    else:
        player_pos = np.nanmean(ts_3d[:10, :, :].mean(axis=1), axis=0)

    R, origin = compute_hoop_relative_transform(player_pos)
    centered = ts_3d - origin.reshape(1, 1, 3)
    ts_hoop = np.einsum('ij,fkj->fki', R, centered)

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist',
                  'left_shoulder', 'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'neck', 'nose']

    for joint in key_joints:
        if joint not in kp_index:
            continue
        idx = kp_index[joint]
        for coord, cname in enumerate(['forward', 'lateral', 'vertical']):
            s = ts_hoop[:, idx, coord]
            prefix = f"hr_{joint}_{cname}"
            feats[f"{prefix}_mean"] = np.nanmean(s)
            feats[f"{prefix}_std"] = np.nanstd(s)
            feats[f"{prefix}_min"] = np.nanmin(s)
            feats[f"{prefix}_max"] = np.nanmax(s)
            feats[f"{prefix}_range"] = np.nanmax(s) - np.nanmin(s)
            feats[f"{prefix}_release_mean"] = np.nanmean(s[140:180])
            vel = np.gradient(s, DT)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)
            feats[f"{prefix}_vel_at_release"] = vel[153] if len(vel) > 153 else 0.0

        for c, cname in enumerate(['x', 'y', 'z']):
            s = ts_3d[:, idx, c]
            prefix = f"{joint}_{cname}"
            feats[f"{prefix}_mean"] = np.nanmean(s)
            feats[f"{prefix}_std"] = np.nanstd(s)
            feats[f"{prefix}_min"] = np.nanmin(s)
            feats[f"{prefix}_max"] = np.nanmax(s)
            feats[f"{prefix}_range"] = np.nanmax(s) - np.nanmin(s)
            vel = np.gradient(s, DT)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)
            feats[f"f153_{prefix}"] = s[153]

    # Body alignment
    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    rs, ls = kp_index.get('right_shoulder'), kp_index.get('left_shoulder')
    rw, lw = kp_index.get('right_wrist'), kp_index.get('left_wrist')

    if rh is not None and lh is not None:
        hip_lat = ts_hoop[:, rh, 1] - ts_hoop[:, lh, 1]
        feats['hr_hip_alignment_mean'] = np.nanmean(hip_lat)
        feats['hr_hip_alignment_release'] = hip_lat[153]
    if rs is not None and ls is not None:
        shoulder_lat = ts_hoop[:, rs, 1] - ts_hoop[:, ls, 1]
        feats['hr_shoulder_alignment_mean'] = np.nanmean(shoulder_lat)
        feats['hr_shoulder_alignment_release'] = shoulder_lat[153]
    if rw is not None and lw is not None:
        guide_lat = ts_hoop[:, lw, 1] - ts_hoop[:, rw, 1]
        feats['hr_guide_hand_lateral_release'] = guide_lat[153]
    if rw is not None and rs is not None:
        arm_lat = ts_hoop[:, rw, 1] - ts_hoop[:, rs, 1]
        feats['hr_arm_lateral_dev_release'] = arm_lat[153]

    # Joint angles
    for j1, j2, j3, name in [
        ('right_shoulder', 'right_elbow', 'right_wrist', 'elbow'),
    ]:
        if all(j in kp_index for j in [j1, j2, j3]):
            p1, p2, p3 = ts_3d[:, kp_index[j1]], ts_3d[:, kp_index[j2]], ts_3d[:, kp_index[j3]]
            v1, v2 = p1 - p2, p3 - p2
            dot = np.sum(v1 * v2, axis=1)
            n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
            denom = n1 * n2; denom[denom == 0] = 1e-10
            angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            feats[f"{name}_angle_release"] = angle[153]
            feats[f"{name}_angle_range"] = np.nanmax(angle) - np.nanmin(angle)

    # Phase velocity
    for pname, (s, e) in [('load', (60, 120)), ('propel', (120, 170))]:
        for joint in ['right_wrist', 'right_elbow']:
            if joint not in kp_index: continue
            idx2 = kp_index[joint]
            for c in range(3):
                vel = np.gradient(ts_3d[s:e, idx2, c], DT)
                feats[f"phase_{pname}_{joint}_{'xyz'[c]}_vel_max"] = np.nanmax(vel)

    return feats


# ============================================================
# BIOMECHANICAL FEATURES (NEW)
# ============================================================

def _safe_joint_angle_timeseries(p1, p2, p3):
    """Compute joint angle timeseries (degrees) for three joint positions.

    p1, p2, p3: arrays of shape (N, 3) - positions of proximal, vertex, distal joints.
    Returns angle at vertex joint for each frame.
    """
    v1 = p1 - p2  # proximal to vertex
    v2 = p3 - p2  # distal to vertex
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    denom = np.where(denom < 1e-10, 1e-10, denom)
    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg


def _savgol_smooth(signal, window=7, polyorder=2):
    """Smooth a signal with Savitzky-Golay filter, handling NaNs."""
    valid = ~np.isnan(signal)
    if valid.sum() < window:
        return signal.copy()
    out = signal.copy()
    out[valid] = savgol_filter(signal[valid], min(window, valid.sum()),
                                min(polyorder, min(window, valid.sum()) - 1))
    return out


def _savgol_derivative(signal, window=7, polyorder=2, deriv=1, delta=DT):
    """Compute derivative via Savitzky-Golay filter."""
    valid = ~np.isnan(signal)
    n_valid = valid.sum()
    if n_valid < window:
        return np.zeros_like(signal)
    w = min(window, n_valid)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return np.zeros_like(signal)
    po = min(polyorder, w - 1)
    out = np.zeros_like(signal)
    out[valid] = savgol_filter(signal[valid], w, po, deriv=deriv, delta=delta)
    return out


def _fingertip_center(ts_3d, kp_index):
    """Average of 3 finger distal positions (index, middle, ring) to reduce noise.
    Falls back to wrist if finger keypoints not available."""
    finger_tips = ['right_second_finger_distal', 'right_third_finger_distal',
                   'right_fourth_finger_distal']
    positions = []
    for ft in finger_tips:
        if ft in kp_index:
            positions.append(ts_3d[:, kp_index[ft], :])
    if len(positions) >= 2:
        return np.nanmean(np.stack(positions), axis=0)  # (240, 3)
    # Fallback: use wrist
    if 'right_wrist' in kp_index:
        return ts_3d[:, kp_index['right_wrist'], :]
    return None


def extract_biomech_features(ts_3d, keypoint_names, pid):
    """Extract ~35 biomechanical features from 3D timeseries.

    ts_3d: shape (240, n_keypoints, 3)
    keypoint_names: list of keypoint names
    pid: player id

    Returns dict of feature_name -> value
    """
    feats = {}
    kp_index = {name: i for i, name in enumerate(keypoint_names)}

    # Required joints
    rs = kp_index.get('right_shoulder')
    re = kp_index.get('right_elbow')
    rw = kp_index.get('right_wrist')
    rh = kp_index.get('right_hip')
    lh = kp_index.get('left_hip')
    ls = kp_index.get('left_shoulder')
    rk = kp_index.get('right_knee')
    ra = kp_index.get('right_ankle')
    mh = kp_index.get('mid_hip')
    nk = kp_index.get('neck')
    ns = kp_index.get('nose')

    ps = PROPULSION_START
    pe = PROPULSION_END

    # ---- A. ANGULAR VELOCITIES AT RELEASE FRAME ----

    # A1. Elbow extension angular velocity (shoulder-elbow-wrist)
    if all(v is not None for v in [rs, re, rw]):
        elbow_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        elbow_angle_smooth = _savgol_smooth(elbow_angle)
        elbow_angvel = _savgol_derivative(elbow_angle_smooth, window=7, polyorder=2, deriv=1)

        feats['bm_elbow_angvel_at_153'] = elbow_angvel[153]
        feats['bm_elbow_angvel_at_150'] = elbow_angvel[150]
        feats['bm_elbow_angvel_at_170'] = elbow_angvel[170] if len(elbow_angvel) > 170 else 0.0
        feats['bm_elbow_angvel_peak_propulsion'] = np.nanmax(np.abs(elbow_angvel[ps:pe]))
        feats['bm_elbow_angle_at_153'] = elbow_angle_smooth[153]
        feats['bm_elbow_angle_at_150'] = elbow_angle_smooth[150]

    # A2. Shoulder flexion angular velocity
    # Shoulder angle: angle in vertical plane between trunk and upper arm
    if all(v is not None for v in [rs, re, rh]):
        shoulder_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rh, :], ts_3d[:, rs, :], ts_3d[:, re, :])
        shoulder_angle_smooth = _savgol_smooth(shoulder_angle)
        shoulder_angvel = _savgol_derivative(shoulder_angle_smooth, window=7, polyorder=2, deriv=1)

        feats['bm_shoulder_angvel_at_153'] = shoulder_angvel[153]
        feats['bm_shoulder_angvel_at_150'] = shoulder_angvel[150]
        feats['bm_shoulder_angvel_peak_propulsion'] = np.nanmax(np.abs(shoulder_angvel[ps:pe]))
        feats['bm_shoulder_angle_at_153'] = shoulder_angle_smooth[153]

    # A3. Wrist snap angular velocity (elbow-wrist-fingertip_center)
    fingertip = _fingertip_center(ts_3d, kp_index)
    if all(v is not None for v in [re, rw]) and fingertip is not None:
        wrist_angle = _safe_joint_angle_timeseries(
            ts_3d[:, re, :], ts_3d[:, rw, :], fingertip)
        wrist_angle_smooth = _savgol_smooth(wrist_angle)
        wrist_angvel = _savgol_derivative(wrist_angle_smooth, window=7, polyorder=2, deriv=1)

        feats['bm_wrist_angvel_at_153'] = wrist_angvel[153]
        feats['bm_wrist_angvel_at_170'] = wrist_angvel[170] if len(wrist_angvel) > 170 else 0.0
        feats['bm_wrist_angvel_peak_propulsion'] = np.nanmax(np.abs(wrist_angvel[ps:pe]))

    # ---- B. PROXIMAL-TO-DISTAL TIMING ----

    # Compute peak angular velocity frame for each joint in propulsion phase
    peak_frames = {}

    if all(v is not None for v in [rs, re, rw]):
        elbow_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        elbow_angvel_prop = _savgol_derivative(_savgol_smooth(elbow_angle), window=7, polyorder=2, deriv=1)
        peak_frames['elbow'] = ps + np.argmax(np.abs(elbow_angvel_prop[ps:pe]))

    if all(v is not None for v in [rs, re, rh]):
        shoulder_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rh, :], ts_3d[:, rs, :], ts_3d[:, re, :])
        shoulder_angvel_prop = _savgol_derivative(_savgol_smooth(shoulder_angle), window=7, polyorder=2, deriv=1)
        peak_frames['shoulder'] = ps + np.argmax(np.abs(shoulder_angvel_prop[ps:pe]))

    if all(v is not None for v in [re, rw]) and fingertip is not None:
        wrist_angle = _safe_joint_angle_timeseries(
            ts_3d[:, re, :], ts_3d[:, rw, :], fingertip)
        wrist_angvel_prop = _savgol_derivative(_savgol_smooth(wrist_angle), window=7, polyorder=2, deriv=1)
        peak_frames['wrist'] = ps + np.argmax(np.abs(wrist_angvel_prop[ps:pe]))

    # Fingertip peak linear velocity
    if fingertip is not None:
        ft_vel = np.linalg.norm(np.gradient(fingertip[ps:pe], DT, axis=0), axis=1)
        peak_frames['fingertip'] = ps + np.argmax(ft_vel)

    # Timing delays (in seconds)
    if 'shoulder' in peak_frames and 'elbow' in peak_frames:
        feats['bm_timing_shoulder_to_elbow'] = (peak_frames['elbow'] - peak_frames['shoulder']) * DT
    if 'elbow' in peak_frames and 'wrist' in peak_frames:
        feats['bm_timing_elbow_to_wrist'] = (peak_frames['wrist'] - peak_frames['elbow']) * DT
    if 'wrist' in peak_frames and 'fingertip' in peak_frames:
        feats['bm_timing_wrist_to_fingertip'] = (peak_frames['fingertip'] - peak_frames['wrist']) * DT
    if 'shoulder' in peak_frames and 'fingertip' in peak_frames:
        feats['bm_timing_total_chain'] = (peak_frames['fingertip'] - peak_frames['shoulder']) * DT

    # ---- C. TRUNK LEAN ----

    if mh is not None and nk is not None:
        trunk_vec = ts_3d[:, nk, :] - ts_3d[:, mh, :]  # neck - mid_hip

        # Forward lean: angle between trunk and vertical in sagittal plane
        # Use hoop-relative coordinates for forward/lateral decomposition
        player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0)
        R, origin = compute_hoop_relative_transform(player_pos)
        trunk_centered = trunk_vec.copy()  # trunk vector doesn't need translation
        trunk_hoop = np.einsum('ij,fj->fi', R, trunk_centered)  # (240, 3): forward, lateral, vertical

        # Forward lean angle: arctan2(forward_component, vertical_component)
        forward_lean = np.degrees(np.arctan2(trunk_hoop[:, 0], trunk_hoop[:, 2]))
        lateral_lean = np.degrees(np.arctan2(trunk_hoop[:, 1], trunk_hoop[:, 2]))

        forward_lean_smooth = _savgol_smooth(forward_lean)
        lateral_lean_smooth = _savgol_smooth(lateral_lean)

        feats['bm_trunk_forward_lean_153'] = forward_lean_smooth[153]
        feats['bm_trunk_lateral_lean_153'] = lateral_lean_smooth[153]
        feats['bm_trunk_forward_lean_rate_153'] = _savgol_derivative(forward_lean_smooth)[153]

    # ---- D. CENTER OF MASS VELOCITY ----

    if mh is not None and nk is not None and rs is not None:
        # CoM approximation: weighted average
        com = (0.5 * ts_3d[:, mh, :] +
               0.3 * ts_3d[:, nk, :] +
               0.2 * ts_3d[:, rs, :])
        com_smooth = np.stack([_savgol_smooth(com[:, c]) for c in range(3)], axis=1)
        com_vel = np.gradient(com_smooth, DT, axis=0)

        com_speed = np.linalg.norm(com_vel, axis=1)
        feats['bm_com_speed_at_153'] = com_speed[153]
        feats['bm_com_vel_z_at_153'] = com_vel[153, 2]  # vertical velocity

        # Forward velocity in hoop direction
        player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0)
        R, _ = compute_hoop_relative_transform(player_pos)
        com_vel_hoop = np.einsum('ij,fj->fi', R, com_vel)
        feats['bm_com_vel_forward_at_153'] = com_vel_hoop[153, 0]

        # Rising or falling at release
        feats['bm_com_rising_at_153'] = float(com_vel[153, 2] > 0)

    # ---- E. COORDINATION VARIABILITY ----

    if all(v is not None for v in [rs, re, rw]):
        elbow_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        elbow_angvel_full = _savgol_derivative(_savgol_smooth(elbow_angle), window=7, polyorder=2, deriv=1)

        # Std of angular velocity during propulsion
        feats['bm_elbow_angvel_std_propulsion'] = np.nanstd(elbow_angvel_full[ps:pe])

    if all(v is not None for v in [rs, re, rh]):
        shoulder_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rh, :], ts_3d[:, rs, :], ts_3d[:, re, :])
        shoulder_angvel_full = _savgol_derivative(_savgol_smooth(shoulder_angle), window=7, polyorder=2, deriv=1)

        # Coefficient of variation of angular velocity during propulsion
        mean_av = np.nanmean(np.abs(shoulder_angvel_full[ps:pe]))
        std_av = np.nanstd(shoulder_angvel_full[ps:pe])
        feats['bm_shoulder_angvel_cv_propulsion'] = std_av / max(mean_av, 1e-6)

    # Wrist jerk (derivative of acceleration) - lower = smoother release
    if rw is not None:
        wrist_pos = ts_3d[:, rw, :]
        wrist_smooth = np.stack([_savgol_smooth(wrist_pos[:, c]) for c in range(3)], axis=1)
        # Third derivative of position = jerk
        wrist_jerk = np.zeros_like(wrist_smooth)
        for c in range(3):
            wrist_jerk[:, c] = _savgol_derivative(wrist_smooth[:, c], window=9, polyorder=3, deriv=3)
        jerk_mag = np.linalg.norm(wrist_jerk, axis=1)
        feats['bm_wrist_jerk_at_153'] = jerk_mag[153]
        feats['bm_wrist_jerk_mean_propulsion'] = np.nanmean(jerk_mag[ps:pe])

    # ---- F. KNEE ANGULAR VELOCITY ----

    if all(v is not None for v in [rh, rk, ra]):
        knee_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rh, :], ts_3d[:, rk, :], ts_3d[:, ra, :])
        knee_angle_smooth = _savgol_smooth(knee_angle)
        knee_angvel = _savgol_derivative(knee_angle_smooth, window=7, polyorder=2, deriv=1)

        feats['bm_knee_angvel_at_153'] = knee_angvel[153]
        feats['bm_knee_angvel_peak_propulsion'] = np.nanmax(np.abs(knee_angvel[ps:pe]))

        # Knee-to-elbow timing delay
        knee_peak_frame = ps + np.argmax(np.abs(knee_angvel[ps:pe]))
        if 'elbow' in peak_frames:
            feats['bm_timing_knee_to_elbow'] = (peak_frames['elbow'] - knee_peak_frame) * DT

    # ---- G. RELEASE HEIGHT RATIO ----

    if rw is not None and ns is not None:
        # Player height: max nose_z across all frames
        nose_z_max = np.nanmax(ts_3d[:, ns, 2])
        wrist_z_153 = ts_3d[153, rw, 2]

        if nose_z_max > 0:
            feats['bm_release_height_ratio'] = wrist_z_153 / nose_z_max
        feats['bm_release_z_over_hoop'] = wrist_z_153 / HOOP_POS[2]

    # ---- H. BODY ALIGNMENT (ENHANCED) ----

    if rs is not None and rw is not None and re is not None:
        # Shooting arm plane normal vector alignment with hoop direction
        # Arm plane defined by shoulder, elbow, wrist
        v1 = ts_3d[153, re, :] - ts_3d[153, rs, :]  # shoulder to elbow
        v2 = ts_3d[153, rw, :] - ts_3d[153, rs, :]  # shoulder to wrist
        arm_normal = np.cross(v1, v2)
        arm_normal_norm = np.linalg.norm(arm_normal)
        if arm_normal_norm > 1e-6:
            arm_normal = arm_normal / arm_normal_norm
            # Hoop direction
            player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0) if mh is not None else np.nanmean(ts_3d[:10, rs, :], axis=0)
            hoop_dir = HOOP_POS - player_pos
            hoop_dir_norm = np.linalg.norm(hoop_dir)
            if hoop_dir_norm > 1e-6:
                hoop_dir = hoop_dir / hoop_dir_norm
                # Dot product of arm normal with hoop direction
                feats['bm_arm_plane_hoop_alignment'] = np.abs(np.dot(arm_normal, hoop_dir))

    if rs is not None and ls is not None and mh is not None:
        # Shoulder rotation relative to hoop line
        shoulder_vec = ts_3d[153, rs, :2] - ts_3d[153, ls, :2]  # XY plane
        player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0)
        hoop_dir_2d = (HOOP_POS[:2] - player_pos[:2])
        hoop_dir_2d_norm = np.linalg.norm(hoop_dir_2d)
        if hoop_dir_2d_norm > 1e-6:
            hoop_dir_2d = hoop_dir_2d / hoop_dir_2d_norm
            shoulder_vec_norm = np.linalg.norm(shoulder_vec)
            if shoulder_vec_norm > 1e-6:
                shoulder_vec = shoulder_vec / shoulder_vec_norm
                cos_ang = np.clip(np.dot(shoulder_vec, hoop_dir_2d), -1, 1)
                feats['bm_shoulder_rotation_vs_hoop'] = np.degrees(np.arccos(cos_ang))

    if rh is not None and lh is not None and mh is not None:
        # Hip rotation relative to hoop line
        hip_vec = ts_3d[153, rh, :2] - ts_3d[153, lh, :2]
        player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0)
        hoop_dir_2d = (HOOP_POS[:2] - player_pos[:2])
        hoop_dir_2d_norm = np.linalg.norm(hoop_dir_2d)
        if hoop_dir_2d_norm > 1e-6:
            hoop_dir_2d = hoop_dir_2d / hoop_dir_2d_norm
            hip_vec_norm = np.linalg.norm(hip_vec)
            if hip_vec_norm > 1e-6:
                hip_vec = hip_vec / hip_vec_norm
                cos_ang = np.clip(np.dot(hip_vec, hoop_dir_2d), -1, 1)
                feats['bm_hip_rotation_vs_hoop'] = np.degrees(np.arccos(cos_ang))

    # Arm tilt: deviation of arm plane from sagittal plane
    if rs is not None and re is not None and rw is not None:
        arm_vec = ts_3d[153, rw, :] - ts_3d[153, rs, :]
        arm_vec_2d = arm_vec[:2]  # XY projection
        arm_vec_z = arm_vec[2]
        arm_vec_2d_norm = np.linalg.norm(arm_vec_2d)
        if arm_vec_2d_norm > 1e-6 and mh is not None:
            # Angle of arm projection relative to hoop direction
            player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0)
            hoop_dir_2d = (HOOP_POS[:2] - player_pos[:2])
            hoop_dir_2d_norm = np.linalg.norm(hoop_dir_2d)
            if hoop_dir_2d_norm > 1e-6:
                hoop_dir_2d = hoop_dir_2d / hoop_dir_2d_norm
                arm_dir_2d = arm_vec_2d / arm_vec_2d_norm
                # Lateral deviation angle
                cross_val = arm_dir_2d[0] * hoop_dir_2d[1] - arm_dir_2d[1] * hoop_dir_2d[0]
                feats['bm_arm_lateral_tilt'] = np.degrees(np.arcsin(np.clip(cross_val, -1, 1)))

    # ---- I. TEMPORAL SHAPE FEATURES ----

    # Time-to-peak within propulsion phase
    if 'elbow' in peak_frames:
        feats['bm_elbow_time_to_peak'] = (peak_frames['elbow'] - ps) * DT
    if 'shoulder' in peak_frames:
        feats['bm_shoulder_time_to_peak'] = (peak_frames['shoulder'] - ps) * DT

    # Wrist snap duration: from wrist angular accel start to peak angular velocity
    if all(v is not None for v in [re, rw]) and fingertip is not None:
        wrist_angle = _safe_joint_angle_timeseries(
            ts_3d[:, re, :], ts_3d[:, rw, :], fingertip)
        wrist_angvel_prop = _savgol_derivative(_savgol_smooth(wrist_angle), window=7, polyorder=2, deriv=1)
        wrist_peak_frame = ps + np.argmax(np.abs(wrist_angvel_prop[ps:pe]))

        # Angular acceleration
        wrist_angacc = _savgol_derivative(_savgol_smooth(wrist_angle), window=7, polyorder=2, deriv=2)
        # Find start of snap: first frame in propulsion where angular acceleration exceeds 10% of peak
        acc_thresh = 0.1 * np.nanmax(np.abs(wrist_angacc[ps:pe]))
        acc_above = np.abs(wrist_angacc[ps:wrist_peak_frame]) > acc_thresh
        if np.any(acc_above):
            snap_start = ps + np.argmax(acc_above)
            feats['bm_wrist_snap_duration'] = (wrist_peak_frame - snap_start) * DT

    # Propulsion phase duration: from knee extension start to release
    if all(v is not None for v in [rh, rk, ra]):
        knee_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rh, :], ts_3d[:, rk, :], ts_3d[:, ra, :])
        knee_angvel_full = _savgol_derivative(_savgol_smooth(knee_angle), window=7, polyorder=2, deriv=1)
        # Find when knee starts extending (angular velocity crosses threshold)
        knee_thresh = 0.1 * np.nanmax(np.abs(knee_angvel_full[ps:pe]))
        knee_above = np.abs(knee_angvel_full[90:pe]) > knee_thresh
        if np.any(knee_above):
            knee_start = 90 + np.argmax(knee_above)
            feats['bm_propulsion_duration'] = (153 - knee_start) * DT  # to frame 153

    # Angular velocity profile skewness (elbow)
    if all(v is not None for v in [rs, re, rw]):
        elbow_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        elbow_angvel_prop = _savgol_derivative(_savgol_smooth(elbow_angle), window=7, polyorder=2, deriv=1)
        av_prop = elbow_angvel_prop[ps:pe]
        mean_av = np.nanmean(av_prop)
        std_av = np.nanstd(av_prop)
        if std_av > 1e-6:
            feats['bm_elbow_angvel_skewness'] = np.nanmean(((av_prop - mean_av) / std_av) ** 3)

        # Peak-to-release deceleration rate
        peak_idx = ps + np.argmax(np.abs(av_prop))
        if peak_idx < 153:
            peak_val = np.abs(elbow_angvel_prop[peak_idx])
            release_val = np.abs(elbow_angvel_prop[153])
            dt_frames = 153 - peak_idx
            if dt_frames > 0 and peak_val > 1e-6:
                feats['bm_elbow_decel_rate'] = (peak_val - release_val) / (dt_frames * DT)

    return feats


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(X_feat, y_target, pids, target_name, feat_names, use_feature_selection=True, top_k=80):
    """Train per-player per-target model with optional feature selection."""
    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros(len(y_target))
    models = {}
    scalers = {}
    selected_features = {}

    print(f"\n--- {target_name.upper()} MODEL ({len(feat_names)} features) ---")

    for pid in unique_pids:
        mask = pids == pid
        X_p = X_feat[mask]
        y_p = y_target[mask]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)
        scalers[pid] = scaler

        # Feature selection via mutual information
        if use_feature_selection and X_scaled.shape[1] > top_k:
            mi_scores = mutual_info_regression(X_scaled, y_p, random_state=42, n_neighbors=5)
            top_indices = np.argsort(mi_scores)[::-1][:top_k]
            selected_features[pid] = top_indices
            X_sel = X_scaled[:, top_indices]
        else:
            selected_features[pid] = np.arange(X_scaled.shape[1])
            X_sel = X_scaled

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sel)):
            X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
            y_tr = y_p[tr_idx]
            preds = []
            for cls, params in [
                (lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
                (xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
                (CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)),
                (Ridge, dict(alpha=1.0)),
            ]:
                m = cls(**params)
                m.fit(X_tr, y_tr)
                preds.append(m.predict(X_val))
            fold_preds[val_idx] = 0.3*preds[0] + 0.3*preds[1] + 0.3*preds[2] + 0.1*preds[3]

        oof_preds[global_idx] = fold_preds
        mse = np.mean((fold_preds - y_p)**2)
        print(f"  Player {pid}: CV MSE = {mse:.6f} (n={n}, features={X_sel.shape[1]})")

        # Train final models on all data
        for name, cls, params in [
            ('lgb', lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
            ('cat', CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False)),
            ('ridge', Ridge, dict(alpha=1.0)),
        ]:
            m = cls(**params)
            m.fit(X_sel, y_p)
            models[(pid, name)] = m

    overall_mse = np.mean((oof_preds - y_target)**2)
    print(f"  Overall {target_name} CV MSE: {overall_mse:.6f}")
    return oof_preds, models, scalers, selected_features


def predict_model(X_feat, pids, models, scalers, selected_features):
    """Predict using trained per-player models."""
    preds = np.zeros(len(X_feat))
    for i, (x, pid) in enumerate(zip(X_feat, pids)):
        x_scaled = np.nan_to_num(scalers[pid].transform(x.reshape(1, -1)), nan=0.0)
        x_sel = x_scaled[:, selected_features[pid]]
        p = [models[(pid, n)].predict(x_sel)[0] for n in ['lgb', 'xgb', 'cat', 'ridge']]
        preds[i] = 0.3*p[0] + 0.3*p[1] + 0.3*p[2] + 0.1*p[3]
    return preds


# ============================================================
# PLS DEPTH MODEL (from target_specific_blend.py)
# ============================================================

def train_pls_depth(train_data):
    """Train PLS models specifically for depth prediction."""
    X_raw = train_data['X_raw']
    y = train_data['y']
    pids = train_data['pids']

    unique_pids = sorted(np.unique(pids))
    depth_idx = 1
    oof_depth = np.zeros(len(y))
    models = {}
    scalers = {}

    print("\n--- PLS DEPTH MODEL ---")

    for pid in unique_pids:
        mask = pids == pid
        X_p = X_raw[mask]
        y_depth = y[mask, depth_idx]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        scalers[pid] = scaler

        candidates = [3, 5, 8, 10, 15, 20, 25, 30]
        max_comp = min(30, n - n // 5 - 1)
        candidates = [c for c in candidates if c <= max_comp]

        best_n, best_mse = 5, float('inf')
        for nc in candidates:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(X_scaled):
                pls = PLSRegression(n_components=nc)
                pls.fit(X_scaled[tr_idx], y_depth[tr_idx])
                pls_pred = pls.predict(X_scaled[val_idx]).flatten()
                X_tr_pls = pls.transform(X_scaled[tr_idx])
                X_val_pls = pls.transform(X_scaled[val_idx])
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_pls, y_depth[tr_idx])
                ridge_pred = ridge.predict(X_val_pls)
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=50, num_leaves=8, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(X_tr_pls, y_depth[tr_idx])
                lgb_pred = lgb_m.predict(X_val_pls)
                pred = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred
                mses.append(np.mean((pred - y_depth[val_idx]) ** 2))
            avg = np.mean(mses)
            if avg < best_mse:
                best_mse = avg
                best_n = nc

        print(f"  Player {pid}: best PLS components = {best_n}, CV MSE = {best_mse:.4f}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            pls = PLSRegression(n_components=best_n)
            pls.fit(X_scaled[tr_idx], y_depth[tr_idx])
            pls_pred = pls.predict(X_scaled[val_idx]).flatten()
            X_tr_pls = pls.transform(X_scaled[tr_idx])
            X_val_pls = pls.transform(X_scaled[val_idx])
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_tr_pls, y_depth[tr_idx])
            ridge_pred = ridge.predict(X_val_pls)
            lgb_m = lgb.LGBMRegressor(
                n_estimators=50, num_leaves=8, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_tr_pls, y_depth[tr_idx])
            lgb_pred = lgb_m.predict(X_val_pls)
            fold_preds[val_idx] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred

        oof_depth[global_idx] = fold_preds

        pls_final = PLSRegression(n_components=best_n)
        pls_final.fit(X_scaled, y_depth)
        X_pls_all = pls_final.transform(X_scaled)
        ridge_final = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge_final.fit(X_pls_all, y_depth)
        lgb_final = lgb.LGBMRegressor(
            n_estimators=50, num_leaves=8, learning_rate=0.05,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
        lgb_final.fit(X_pls_all, y_depth)
        models[pid] = {'pls': pls_final, 'ridge': ridge_final, 'lgb': lgb_final}

    mse = np.mean((oof_depth - y[:, depth_idx]) ** 2)
    print(f"  Overall depth CV MSE: {mse:.6f}")
    return oof_depth, models, scalers


def predict_pls_depth(test_data, models, scalers):
    X_raw = test_data['X_raw']
    pids = test_data['pids']
    preds = np.zeros(len(X_raw))
    for i, (x, pid) in enumerate(zip(X_raw, pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        pls = models[pid]['pls']
        pls_pred = pls.predict(x_scaled).flatten()[0]
        x_pls = pls.transform(x_scaled)
        ridge_pred = models[pid]['ridge'].predict(x_pls)[0]
        lgb_pred = models[pid]['lgb'].predict(x_pls)[0]
        preds[i] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred
    return preds


# ============================================================
# SCALE + BLEND
# ============================================================

def scale_predictions(raw_preds, target):
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


def main():
    t0 = time.time()
    print("=" * 70)
    print("BIOMECHANICAL ENHANCED BLEND EXPERIMENT")
    print("=" * 70)

    train_data, test_data = load_raw_data()
    X_3d_train = train_data['X_3d']
    X_3d_test = test_data['X_3d']
    kp_names = train_data['keypoint_names']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    y = train_data['y']

    # ---- Step 1: Extract biomechanical features ----
    print("\n" + "=" * 70)
    print("STEP 1: Extract biomechanical features")
    print("=" * 70)

    # Sanity check on 5 shots first
    print("\n  Sanity check on first 5 shots:")
    for i in range(5):
        bm_feats = extract_biomech_features(X_3d_train[i], kp_names, pids_train[i])
        bm_keys = sorted(bm_feats.keys())
        if i == 0:
            print(f"    {len(bm_keys)} biomech features extracted")
            for k in bm_keys[:5]:
                print(f"      {k}: {bm_feats[k]:.4f}")
        # Check angular velocity ranges
        elbow_av = bm_feats.get('bm_elbow_angvel_peak_propulsion', 0)
        shoulder_av = bm_feats.get('bm_shoulder_angvel_peak_propulsion', 0)
        trunk_lean = bm_feats.get('bm_trunk_forward_lean_153', 0)
        print(f"    Shot {i}: elbow_av={elbow_av:.1f} deg/s, shoulder_av={shoulder_av:.1f} deg/s, "
              f"trunk_lean={trunk_lean:.1f} deg")

    # Extract all biomech features
    print("\n  Extracting biomech features for all training shots...")
    train_bm_feats = []
    for i in range(len(X_3d_train)):
        bm_feats = extract_biomech_features(X_3d_train[i], kp_names, pids_train[i])
        train_bm_feats.append(bm_feats)
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(X_3d_train)}")

    print("  Extracting biomech features for all test shots...")
    test_bm_feats = []
    for i in range(len(X_3d_test)):
        bm_feats = extract_biomech_features(X_3d_test[i], kp_names, pids_test[i])
        test_bm_feats.append(bm_feats)

    # Get biomech feature names
    bm_feat_names = sorted([k for k in train_bm_feats[0].keys()])
    print(f"\n  Total biomech features: {len(bm_feat_names)}")

    # ---- Step 2: Extract hoop-relative features ----
    print("\n" + "=" * 70)
    print("STEP 2: Extract hoop-relative features")
    print("=" * 70)

    print("  Extracting HR features for training shots...")
    train_hr_feats = []
    for i in range(len(X_3d_train)):
        hr_feats = extract_hoop_relative_features(X_3d_train[i], kp_names, pids_train[i])
        train_hr_feats.append(hr_feats)
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(X_3d_train)}")

    print("  Extracting HR features for test shots...")
    test_hr_feats = []
    for i in range(len(X_3d_test)):
        hr_feats = extract_hoop_relative_features(X_3d_test[i], kp_names, pids_test[i])
        test_hr_feats.append(hr_feats)

    hr_feat_names = sorted(train_hr_feats[0].keys())
    print(f"  Total HR features: {len(hr_feat_names)}")

    # ---- Step 3: Combine features ----
    # HR features only (baseline)
    X_hr_train = np.array([[f.get(name, 0.0) for name in hr_feat_names] for f in train_hr_feats], dtype=np.float32)
    X_hr_test = np.array([[f.get(name, 0.0) for name in hr_feat_names] for f in test_hr_feats], dtype=np.float32)
    X_hr_train = np.nan_to_num(X_hr_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_hr_test = np.nan_to_num(X_hr_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Biomech features only
    X_bm_train = np.array([[f.get(name, 0.0) for name in bm_feat_names] for f in train_bm_feats], dtype=np.float32)
    X_bm_test = np.array([[f.get(name, 0.0) for name in bm_feat_names] for f in test_bm_feats], dtype=np.float32)
    X_bm_train = np.nan_to_num(X_bm_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_bm_test = np.nan_to_num(X_bm_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Combined
    combined_feat_names = hr_feat_names + bm_feat_names
    X_combined_train = np.hstack([X_hr_train, X_bm_train])
    X_combined_test = np.hstack([X_hr_test, X_bm_test])

    print(f"\n  Feature counts: HR={X_hr_train.shape[1]}, BM={X_bm_train.shape[1]}, "
          f"Combined={X_combined_train.shape[1]}")

    # ---- Step 4: Check biomech feature correlations with targets ----
    print("\n" + "=" * 70)
    print("STEP 3: Biomech feature correlations with targets")
    print("=" * 70)

    for t_idx, t_name in enumerate(TARGETS):
        print(f"\n  {t_name}:")
        corrs = []
        for j, fn in enumerate(bm_feat_names):
            r = np.corrcoef(X_bm_train[:, j], y[:, t_idx])[0, 1]
            if not np.isnan(r):
                corrs.append((fn, r))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for fn, r in corrs[:8]:
            print(f"    {fn:45s}  r={r:+.4f}")
        n_significant = sum(1 for _, r in corrs if abs(r) > 0.15)
        print(f"    Features with |r| > 0.15: {n_significant}/{len(corrs)}")

    # ---- Step 5: CV evaluation - baseline vs combined ----
    print("\n" + "=" * 70)
    print("STEP 4: CV evaluation - Baseline (HR) vs Combined (HR+BM)")
    print("=" * 70)

    results = {}
    for t_idx, t_name in enumerate(TARGETS):
        print(f"\n{'='*40}")
        print(f"  TARGET: {t_name}")
        print(f"{'='*40}")

        # Baseline: HR only
        print(f"\n  [Baseline: HR features only]")
        oof_hr, _, _, _ = train_model(
            X_hr_train, y[:, t_idx], pids_train, f"{t_name}_hr_baseline", hr_feat_names,
            use_feature_selection=True, top_k=80)
        mse_hr = np.mean((oof_hr - y[:, t_idx])**2)

        # Combined: HR + BM
        print(f"\n  [Combined: HR + BM features]")
        oof_comb, models_comb, scalers_comb, sel_feats_comb = train_model(
            X_combined_train, y[:, t_idx], pids_train, f"{t_name}_combined", combined_feat_names,
            use_feature_selection=True, top_k=80)
        mse_comb = np.mean((oof_comb - y[:, t_idx])**2)

        improvement = (mse_hr - mse_comb) / mse_hr * 100
        print(f"\n  {t_name} comparison:")
        print(f"    HR baseline:  MSE = {mse_hr:.6f}")
        print(f"    HR + BM:      MSE = {mse_comb:.6f}")
        print(f"    Improvement:  {improvement:+.2f}%")

        results[t_name] = {
            'mse_hr': mse_hr,
            'mse_comb': mse_comb,
            'improvement': improvement,
            'oof_hr': oof_hr,
            'oof_comb': oof_comb,
            'models': models_comb,
            'scalers': scalers_comb,
            'sel_feats': sel_feats_comb,
        }

    # ---- Step 5b: Also check biomech-only features ----
    print("\n" + "=" * 70)
    print("STEP 4b: BM-only features (for diversity check)")
    print("=" * 70)
    for t_idx, t_name in enumerate(TARGETS):
        oof_bm, _, _, _ = train_model(
            X_bm_train, y[:, t_idx], pids_train, f"{t_name}_bm_only", bm_feat_names,
            use_feature_selection=False)
        mse_bm = np.mean((oof_bm - y[:, t_idx])**2)
        r_with_hr = np.corrcoef(oof_bm, results[t_name]['oof_hr'])[0, 1]
        print(f"  {t_name} BM-only MSE: {mse_bm:.6f}, correlation with HR: {r_with_hr:.4f}")

    # ---- Step 6: Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_mse_hr = np.mean([results[t]['mse_hr'] for t in TARGETS])
    mean_mse_comb = np.mean([results[t]['mse_comb'] for t in TARGETS])
    print(f"\n  Mean CV MSE (HR baseline): {mean_mse_hr:.6f}")
    print(f"  Mean CV MSE (HR + BM):     {mean_mse_comb:.6f}")
    print(f"  Mean improvement:          {(mean_mse_hr - mean_mse_comb) / mean_mse_hr * 100:+.2f}%")

    # Determine which targets improved
    improved_targets = [t for t in TARGETS if results[t]['improvement'] > 0.5]
    print(f"\n  Targets improved (>0.5%): {improved_targets if improved_targets else 'NONE'}")

    # ---- Step 7: Generate test predictions and blend with Sub 784 ----
    # Always generate submissions for the best configuration (combined or target-specific best)
    print("\n" + "=" * 70)
    print("STEP 5: Generate test predictions and blend with Sub 784")
    print("=" * 70)

    # For each target, use whichever is better: combined or HR-only
    # Also train PLS depth as an additional signal
    oof_pls_depth, pls_models, pls_scalers = train_pls_depth(train_data)
    pls_depth_test = predict_pls_depth(test_data, pls_models, pls_scalers)

    # Generate test predictions for combined model
    test_preds = {}
    for t_idx, t_name in enumerate(TARGETS):
        test_pred = predict_model(
            X_combined_test, pids_test,
            results[t_name]['models'], results[t_name]['scalers'], results[t_name]['sel_feats'])
        test_preds[t_name] = test_pred
        print(f"  {t_name} test predictions: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")

    # Scale predictions
    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub771 = pd.read_csv(SUBMISSION_DIR / "submission_771.csv")

    angle_scaled = scale_predictions(test_preds['angle'], 'angle')
    depth_combined_scaled = scale_predictions(test_preds['depth'], 'depth')
    lr_scaled = scale_predictions(test_preds['left_right'], 'left_right')
    pls_depth_scaled = scale_predictions(pls_depth_test, 'depth')

    our = pd.DataFrame({
        'id': test_data['ids'],
        'new_angle': angle_scaled,
        'new_depth_combined': depth_combined_scaled,
        'new_depth_pls': pls_depth_scaled,
        'new_lr': lr_scaled,
    })
    merged = sub784.merge(our, on='id')

    print("\n  Correlations with Sub 784:")
    for col, nc in [('scaled_angle', 'new_angle'), ('scaled_depth', 'new_depth_combined'),
                     ('scaled_left_right', 'new_lr')]:
        r = np.corrcoef(merged[col], merged[nc])[0, 1]
        print(f"    {col} vs {nc}: r={r:.4f}")

    # Grid search over blend weights with Sub 784
    print("\n  Grid search over blend weights with Sub 784:")
    results_blend = []
    for aw in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for dw in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for lw in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                ba = (1-aw)*merged['scaled_angle'] + aw*merged['new_angle']
                # For depth, use PLS (proven) blended with combined
                bd_pls = (1-dw)*merged['scaled_depth'] + dw*merged['new_depth_pls']
                bd_comb = (1-dw)*merged['scaled_depth'] + dw*merged['new_depth_combined']
                bl = (1-lw)*merged['scaled_left_right'] + lw*merged['new_lr']

                for depth_source, bd in [('pls', bd_pls), ('combined', bd_comb)]:
                    results_blend.append({
                        'aw': aw, 'dw': dw, 'lw': lw,
                        'depth_source': depth_source,
                        'angle_std': ba.std(), 'depth_mean': bd.mean(),
                        'blend_angle': ba.values, 'blend_depth': bd.values, 'blend_lr': bl.values,
                    })

    # Compute diversity from Sub 784
    for r in results_blend:
        da = np.mean((r['blend_angle'] - merged['scaled_angle'].values)**2)
        dd = np.mean((r['blend_depth'] - merged['scaled_depth'].values)**2)
        dl = np.mean((r['blend_lr'] - merged['scaled_left_right'].values)**2)
        r['diversity'] = da + dd + dl

    # Filter by profile constraints
    valid = [r for r in results_blend if r['angle_std'] < 0.15 and 0.49 < r['depth_mean'] < 0.52]
    valid.sort(key=lambda x: -x['diversity'])

    print(f"\n  Total configs: {len(results_blend)}, Valid: {len(valid)}")
    print(f"  {'aw':>4} {'dw':>4} {'lw':>4} {'depth':>8} | {'angle_std':>10} {'depth_mean':>10} | diversity")
    print(f"  " + "-" * 72)

    for r in valid[:20]:
        print(f"  {r['aw']:>4.2f} {r['dw']:>4.2f} {r['lw']:>4.2f} {r['depth_source']:>8} | "
              f"{r['angle_std']:>10.6f} {r['depth_mean']:>10.6f} | {r['diversity']:.8f}")

    # Save top 5 submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    saved_subs = []
    for config in valid[:5]:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': merged['id'],
            'scaled_angle': config['blend_angle'],
            'scaled_depth': config['blend_depth'],
            'scaled_left_right': config['blend_lr'],
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(filepath, index=False)
        saved_subs.append(sub_num)
        print(f"  Sub {sub_num}: aw={config['aw']:.2f} dw={config['dw']:.2f} lw={config['lw']:.2f} "
              f"depth={config['depth_source']} "
              f"angle_std={config['angle_std']:.6f} depth_mean={config['depth_mean']:.6f} "
              f"div={config['diversity']:.8f}")

    # Also save a "best per target" submission: use HR+BM for improved targets, Sub 784 for others
    if improved_targets:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({'id': merged['id']})
        sub['scaled_angle'] = merged['scaled_angle']  # Sub 784 angle by default
        sub['scaled_depth'] = merged['scaled_depth']   # Sub 784 depth by default
        sub['scaled_left_right'] = merged['scaled_left_right']  # Sub 784 LR by default

        for t_name in improved_targets:
            best_weight = 0.15  # conservative blend weight
            if t_name == 'angle':
                sub['scaled_angle'] = (1 - best_weight) * merged['scaled_angle'] + best_weight * merged['new_angle']
            elif t_name == 'depth':
                sub['scaled_depth'] = (1 - best_weight) * merged['scaled_depth'] + best_weight * merged['new_depth_combined']
            elif t_name == 'left_right':
                sub['scaled_left_right'] = (1 - best_weight) * merged['scaled_left_right'] + best_weight * merged['new_lr']

        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(filepath, index=False)
        saved_subs.append(sub_num)
        print(f"  Sub {sub_num}: BEST-PER-TARGET (improved: {improved_targets}, w=0.15)")

    print(f"\n  Saved submissions: {saved_subs}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
