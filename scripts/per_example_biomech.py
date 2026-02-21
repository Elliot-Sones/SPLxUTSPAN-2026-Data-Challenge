"""
Per-Example Pipeline V3: Biomechanical Features

Adds 43 biomechanical features (angular velocities, timing delays, trunk lean,
CoM velocity, coordination variability) to the per-example locally weighted
regression pipeline.

Total features: 198 HC + 43 BM + 15 PLS = ~256 per target
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
FRAME_RATE = 60
DT = 1.0 / FRAME_RATE
FEET_TO_METERS = 0.3048

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
PROPULSION_START = 120
PROPULSION_END = 170


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
# HOOP-RELATIVE COORDINATE TRANSFORM
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
    return R, np.einsum('ij,fkj->fki', R, centered)


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
# HOOP-RELATIVE FEATURES (from per_example_pipeline.py)
# ==============================================================

def extract_hc_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract compact hoop-relative + coordinate feature set at a specific frame."""
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

    # Guide hand
    lw = kp_index.get('left_wrist')
    if lw is not None and rw is not None:
        feats.append(ts_hr[f, lw, 1] - ts_hr[f, rw, 1])
    else:
        feats.append(0.0)

    # Release frame timing
    feats.append(release_frame)

    # Release window dynamics
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
# BIOMECHANICAL FEATURES (from biomech_enhanced_blend.py)
# ==============================================================

def _safe_joint_angle_timeseries(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    denom = np.where(denom < 1e-10, 1e-10, denom)
    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _savgol_smooth(signal, window=7, polyorder=2):
    valid = ~np.isnan(signal)
    if valid.sum() < window:
        return signal.copy()
    out = signal.copy()
    out[valid] = savgol_filter(signal[valid], min(window, valid.sum()),
                                min(polyorder, min(window, valid.sum()) - 1))
    return out


def _savgol_derivative(signal, window=7, polyorder=2, deriv=1, delta=DT):
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
    finger_tips = ['right_second_finger_distal', 'right_third_finger_distal',
                   'right_fourth_finger_distal']
    positions = []
    for ft in finger_tips:
        if ft in kp_index:
            positions.append(ts_3d[:, kp_index[ft], :])
    if len(positions) >= 2:
        return np.nanmean(np.stack(positions), axis=0)
    if 'right_wrist' in kp_index:
        return ts_3d[:, kp_index['right_wrist'], :]
    return None


def _compute_hoop_relative_transform(player_pos):
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
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    return R


def extract_biomech_features(ts_3d, kp_index):
    """Extract ~43 biomechanical features. Returns a 1D array of features."""
    # We'll collect features in a fixed order to ensure consistency
    feature_values = []

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

    # A. ANGULAR VELOCITIES (13 features)
    # A1. Elbow extension
    if all(v is not None for v in [rs, re, rw]):
        elbow_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        elbow_angle_smooth = _savgol_smooth(elbow_angle)
        elbow_angvel = _savgol_derivative(elbow_angle_smooth)
        feature_values.extend([
            elbow_angvel[153],   # at release (angle target frame)
            elbow_angvel[150],   # at depth target frame
            elbow_angvel[170] if len(elbow_angvel) > 170 else 0.0,  # at LR target frame
            np.nanmax(np.abs(elbow_angvel[ps:pe])),  # peak during propulsion
            elbow_angle_smooth[153],  # angle at release
            elbow_angle_smooth[150],  # angle at depth frame
        ])
    else:
        feature_values.extend([0.0] * 6)

    # A2. Shoulder flexion
    if all(v is not None for v in [rs, re, rh]):
        shoulder_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rh, :], ts_3d[:, rs, :], ts_3d[:, re, :])
        shoulder_angle_smooth = _savgol_smooth(shoulder_angle)
        shoulder_angvel = _savgol_derivative(shoulder_angle_smooth)
        feature_values.extend([
            shoulder_angvel[153],
            shoulder_angvel[150],
            np.nanmax(np.abs(shoulder_angvel[ps:pe])),
            shoulder_angle_smooth[153],
        ])
    else:
        feature_values.extend([0.0] * 4)

    # A3. Wrist snap
    fingertip = _fingertip_center(ts_3d, kp_index)
    if all(v is not None for v in [re, rw]) and fingertip is not None:
        wrist_angle = _safe_joint_angle_timeseries(
            ts_3d[:, re, :], ts_3d[:, rw, :], fingertip)
        wrist_angle_smooth = _savgol_smooth(wrist_angle)
        wrist_angvel = _savgol_derivative(wrist_angle_smooth)
        feature_values.extend([
            wrist_angvel[153],
            wrist_angvel[170] if len(wrist_angvel) > 170 else 0.0,
            np.nanmax(np.abs(wrist_angvel[ps:pe])),
        ])
    else:
        feature_values.extend([0.0] * 3)

    # B. PROXIMAL-TO-DISTAL TIMING (4 features)
    peak_frames = {}
    if all(v is not None for v in [rs, re, rw]):
        ea = _safe_joint_angle_timeseries(ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        eav = _savgol_derivative(_savgol_smooth(ea))
        peak_frames['elbow'] = ps + np.argmax(np.abs(eav[ps:pe]))
    if all(v is not None for v in [rs, re, rh]):
        sa = _safe_joint_angle_timeseries(ts_3d[:, rh, :], ts_3d[:, rs, :], ts_3d[:, re, :])
        sav = _savgol_derivative(_savgol_smooth(sa))
        peak_frames['shoulder'] = ps + np.argmax(np.abs(sav[ps:pe]))
    if all(v is not None for v in [re, rw]) and fingertip is not None:
        wa = _safe_joint_angle_timeseries(ts_3d[:, re, :], ts_3d[:, rw, :], fingertip)
        wav = _savgol_derivative(_savgol_smooth(wa))
        peak_frames['wrist'] = ps + np.argmax(np.abs(wav[ps:pe]))
    if fingertip is not None:
        ft_vel = np.linalg.norm(np.gradient(fingertip[ps:pe], DT, axis=0), axis=1)
        peak_frames['fingertip'] = ps + np.argmax(ft_vel)

    feature_values.append(
        (peak_frames['elbow'] - peak_frames['shoulder']) * DT
        if 'shoulder' in peak_frames and 'elbow' in peak_frames else 0.0)
    feature_values.append(
        (peak_frames['wrist'] - peak_frames['elbow']) * DT
        if 'elbow' in peak_frames and 'wrist' in peak_frames else 0.0)
    feature_values.append(
        (peak_frames['fingertip'] - peak_frames['wrist']) * DT
        if 'wrist' in peak_frames and 'fingertip' in peak_frames else 0.0)
    feature_values.append(
        (peak_frames['fingertip'] - peak_frames['shoulder']) * DT
        if 'shoulder' in peak_frames and 'fingertip' in peak_frames else 0.0)

    # C. TRUNK LEAN (3 features)
    if mh is not None and nk is not None:
        trunk_vec = ts_3d[:, nk, :] - ts_3d[:, mh, :]
        player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0)
        R = _compute_hoop_relative_transform(player_pos)
        trunk_hoop = np.einsum('ij,fj->fi', R, trunk_vec)
        forward_lean = np.degrees(np.arctan2(trunk_hoop[:, 0], trunk_hoop[:, 2]))
        lateral_lean = np.degrees(np.arctan2(trunk_hoop[:, 1], trunk_hoop[:, 2]))
        fl_smooth = _savgol_smooth(forward_lean)
        ll_smooth = _savgol_smooth(lateral_lean)
        feature_values.extend([
            fl_smooth[153],
            ll_smooth[153],
            _savgol_derivative(fl_smooth)[153],
        ])
    else:
        feature_values.extend([0.0] * 3)

    # D. CENTER OF MASS VELOCITY (4 features)
    if mh is not None and nk is not None and rs is not None:
        com = 0.5 * ts_3d[:, mh, :] + 0.3 * ts_3d[:, nk, :] + 0.2 * ts_3d[:, rs, :]
        com_smooth = np.stack([_savgol_smooth(com[:, c]) for c in range(3)], axis=1)
        com_vel = np.gradient(com_smooth, DT, axis=0)
        com_speed = np.linalg.norm(com_vel, axis=1)

        player_pos = np.nanmean(ts_3d[:10, mh, :], axis=0)
        R = _compute_hoop_relative_transform(player_pos)
        com_vel_hoop = np.einsum('ij,fj->fi', R, com_vel)

        feature_values.extend([
            com_speed[153],
            com_vel[153, 2],         # vertical velocity
            com_vel_hoop[153, 0],    # forward velocity
            float(com_vel[153, 2] > 0),  # rising flag
        ])
    else:
        feature_values.extend([0.0] * 4)

    # E. COORDINATION VARIABILITY (4 features)
    if all(v is not None for v in [rs, re, rw]):
        ea = _safe_joint_angle_timeseries(ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        eav = _savgol_derivative(_savgol_smooth(ea))
        feature_values.append(np.nanstd(eav[ps:pe]))
    else:
        feature_values.append(0.0)

    if all(v is not None for v in [rs, re, rh]):
        sa = _safe_joint_angle_timeseries(ts_3d[:, rh, :], ts_3d[:, rs, :], ts_3d[:, re, :])
        sav = _savgol_derivative(_savgol_smooth(sa))
        mean_av = np.nanmean(np.abs(sav[ps:pe]))
        std_av = np.nanstd(sav[ps:pe])
        feature_values.append(std_av / max(mean_av, 1e-6))
    else:
        feature_values.append(0.0)

    if rw is not None:
        wrist_pos = ts_3d[:, rw, :]
        wrist_smooth = np.stack([_savgol_smooth(wrist_pos[:, c]) for c in range(3)], axis=1)
        wrist_jerk = np.zeros_like(wrist_smooth)
        for c in range(3):
            wrist_jerk[:, c] = _savgol_derivative(wrist_smooth[:, c], window=9, polyorder=3, deriv=3)
        jerk_mag = np.linalg.norm(wrist_jerk, axis=1)
        feature_values.extend([jerk_mag[153], np.nanmean(jerk_mag[ps:pe])])
    else:
        feature_values.extend([0.0, 0.0])

    # F. KNEE ANGULAR VELOCITY (3 features)
    if all(v is not None for v in [rh, rk, ra]):
        knee_angle = _safe_joint_angle_timeseries(
            ts_3d[:, rh, :], ts_3d[:, rk, :], ts_3d[:, ra, :])
        knee_angvel = _savgol_derivative(_savgol_smooth(knee_angle))
        knee_peak_frame = ps + np.argmax(np.abs(knee_angvel[ps:pe]))
        feature_values.extend([
            knee_angvel[153],
            np.nanmax(np.abs(knee_angvel[ps:pe])),
            (peak_frames['elbow'] - knee_peak_frame) * DT if 'elbow' in peak_frames else 0.0,
        ])
    else:
        feature_values.extend([0.0] * 3)

    # G. RELEASE HEIGHT RATIO (2 features)
    if rw is not None and ns is not None:
        nose_z_max = np.nanmax(ts_3d[:, ns, 2])
        wrist_z_153 = ts_3d[153, rw, 2]
        feature_values.append(wrist_z_153 / max(nose_z_max, 1e-6))
        feature_values.append(wrist_z_153 / HOOP_POS[2])
    else:
        feature_values.extend([0.0] * 2)

    # H. BODY ALIGNMENT (4 features)
    if rs is not None and rw is not None and re is not None:
        v1 = ts_3d[153, re, :] - ts_3d[153, rs, :]
        v2 = ts_3d[153, rw, :] - ts_3d[153, rs, :]
        arm_normal = np.cross(v1, v2)
        arm_normal_norm = np.linalg.norm(arm_normal)
        if arm_normal_norm > 1e-6:
            arm_normal = arm_normal / arm_normal_norm
            pp = np.nanmean(ts_3d[:10, mh, :], axis=0) if mh is not None else np.nanmean(ts_3d[:10, rs, :], axis=0)
            hoop_dir = HOOP_POS - pp
            hdn = np.linalg.norm(hoop_dir)
            if hdn > 1e-6:
                hoop_dir = hoop_dir / hdn
                feature_values.append(np.abs(np.dot(arm_normal, hoop_dir)))
            else:
                feature_values.append(0.0)
        else:
            feature_values.append(0.0)
    else:
        feature_values.append(0.0)

    if rs is not None and ls is not None and mh is not None:
        shoulder_vec = ts_3d[153, rs, :2] - ts_3d[153, ls, :2]
        pp = np.nanmean(ts_3d[:10, mh, :], axis=0)
        hd2d = HOOP_POS[:2] - pp[:2]
        hdn = np.linalg.norm(hd2d)
        svn = np.linalg.norm(shoulder_vec)
        if hdn > 1e-6 and svn > 1e-6:
            feature_values.append(np.degrees(np.arccos(np.clip(np.dot(shoulder_vec / svn, hd2d / hdn), -1, 1))))
        else:
            feature_values.append(0.0)
    else:
        feature_values.append(0.0)

    if rh is not None and lh is not None and mh is not None:
        hip_vec = ts_3d[153, rh, :2] - ts_3d[153, lh, :2]
        pp = np.nanmean(ts_3d[:10, mh, :], axis=0)
        hd2d = HOOP_POS[:2] - pp[:2]
        hdn = np.linalg.norm(hd2d)
        hvn = np.linalg.norm(hip_vec)
        if hdn > 1e-6 and hvn > 1e-6:
            feature_values.append(np.degrees(np.arccos(np.clip(np.dot(hip_vec / hvn, hd2d / hdn), -1, 1))))
        else:
            feature_values.append(0.0)
    else:
        feature_values.append(0.0)

    if rs is not None and re is not None and rw is not None and mh is not None:
        arm_vec = ts_3d[153, rw, :] - ts_3d[153, rs, :]
        av2d = arm_vec[:2]
        avn = np.linalg.norm(av2d)
        pp = np.nanmean(ts_3d[:10, mh, :], axis=0)
        hd2d = HOOP_POS[:2] - pp[:2]
        hdn = np.linalg.norm(hd2d)
        if avn > 1e-6 and hdn > 1e-6:
            ad = av2d / avn
            hd = hd2d / hdn
            cross_val = ad[0] * hd[1] - ad[1] * hd[0]
            feature_values.append(np.degrees(np.arcsin(np.clip(cross_val, -1, 1))))
        else:
            feature_values.append(0.0)
    else:
        feature_values.append(0.0)

    # I. TEMPORAL SHAPE (6 features)
    feature_values.append(
        (peak_frames['elbow'] - ps) * DT if 'elbow' in peak_frames else 0.0)
    feature_values.append(
        (peak_frames['shoulder'] - ps) * DT if 'shoulder' in peak_frames else 0.0)

    # Wrist snap duration
    if all(v is not None for v in [re, rw]) and fingertip is not None:
        wa = _safe_joint_angle_timeseries(ts_3d[:, re, :], ts_3d[:, rw, :], fingertip)
        wav = _savgol_derivative(_savgol_smooth(wa))
        wpf = ps + np.argmax(np.abs(wav[ps:pe]))
        waa = _savgol_derivative(_savgol_smooth(wa), deriv=2)
        acc_thresh = 0.1 * np.nanmax(np.abs(waa[ps:pe]))
        acc_above = np.abs(waa[ps:wpf]) > acc_thresh
        if np.any(acc_above):
            snap_start = ps + np.argmax(acc_above)
            feature_values.append((wpf - snap_start) * DT)
        else:
            feature_values.append(0.0)
    else:
        feature_values.append(0.0)

    # Propulsion duration
    if all(v is not None for v in [rh, rk, ra]):
        ka = _safe_joint_angle_timeseries(ts_3d[:, rh, :], ts_3d[:, rk, :], ts_3d[:, ra, :])
        kav = _savgol_derivative(_savgol_smooth(ka))
        kt = 0.1 * np.nanmax(np.abs(kav[ps:pe]))
        ka_above = np.abs(kav[90:pe]) > kt
        if np.any(ka_above):
            ks = 90 + np.argmax(ka_above)
            feature_values.append((153 - ks) * DT)
        else:
            feature_values.append(0.0)
    else:
        feature_values.append(0.0)

    # Angular velocity skewness + decel rate
    if all(v is not None for v in [rs, re, rw]):
        ea = _safe_joint_angle_timeseries(ts_3d[:, rs, :], ts_3d[:, re, :], ts_3d[:, rw, :])
        eav = _savgol_derivative(_savgol_smooth(ea))
        av_prop = eav[ps:pe]
        mean_av = np.nanmean(av_prop)
        std_av = np.nanstd(av_prop)
        if std_av > 1e-6:
            feature_values.append(np.nanmean(((av_prop - mean_av) / std_av) ** 3))
        else:
            feature_values.append(0.0)

        peak_idx = ps + np.argmax(np.abs(av_prop))
        if peak_idx < 153:
            pv = np.abs(eav[peak_idx])
            rv = np.abs(eav[153])
            dtf = 153 - peak_idx
            if dtf > 0 and pv > 1e-6:
                feature_values.append((pv - rv) / (dtf * DT))
            else:
                feature_values.append(0.0)
        else:
            feature_values.append(0.0)
    else:
        feature_values.extend([0.0] * 2)

    return np.array(feature_values, dtype=np.float32)


# ==============================================================
# FEATURE EXTRACTION
# ==============================================================

def extract_all_features(data, target):
    """Extract HC + biomech features for all shots."""
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']

    all_hc = []
    all_bm = []
    release_frames = []

    for i in range(n):
        ts_3d = data['X_3d'][i]
        _, ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)

        hc_feats = extract_hc_features(ts_3d, ts_hr, kp_index, rf, frame)
        bm_feats = extract_biomech_features(ts_3d, kp_index)

        all_hc.append(hc_feats)
        all_bm.append(bm_feats)

    X_hc = np.array(all_hc, dtype=np.float32)
    X_bm = np.array(all_bm, dtype=np.float32)

    X_hc = np.nan_to_num(X_hc, nan=0.0, posinf=0.0, neginf=0.0)
    X_bm = np.nan_to_num(X_bm, nan=0.0, posinf=0.0, neginf=0.0)

    # Combine HC + BM
    X = np.hstack([X_hc, X_bm])
    return X, np.array(release_frames), X_hc.shape[1], X_bm.shape[1]


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
    """For each test example, train a weighted ridge model where nearby
    training samples get higher weight."""
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

        # LOO
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
    print("PER-EXAMPLE PIPELINE V3 - BIOMECHANICAL FEATURES")
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

    # Store results for each target
    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting HC + biomech features...")
        X_train_feats, rf_train, n_hc, n_bm = extract_all_features(train_data, target)
        X_test_feats, rf_test, _, _ = extract_all_features(test_data, target)
        print(f"  HC features: {n_hc}, BM features: {n_bm}, Total: {X_train_feats.shape[1]}")

        # Add PLS
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_feats, y_raw, pids_train,
            X_test_feats, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Total features with PLS: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # Also extract HC-only for comparison
        X_train_hc_only = X_train_feats[:, :n_hc]
        X_test_hc_only = X_test_feats[:, :n_hc]
        # Add PLS to HC-only
        X_train_hc_pls, X_test_hc_pls = augment_with_pls(
            X_train_hc_only, y_raw, pids_train,
            X_test_hc_only, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        # Config search: compare HC-only vs HC+BM across bandwidth/alpha configs
        print("\n  Config search (HC-only vs HC+BM):")
        configs = []
        for bw in [0.4, 0.5, 0.6]:
            for alpha in [5.0, 10.0]:
                # HC-only (V1 baseline)
                oof_hc, test_hc = locally_weighted_prediction(
                    X_train_hc_pls, y_target, X_test_hc_pls, pids_train, pids_test,
                    bandwidth_quantile=bw, alpha=alpha)
                mse_hc = np.mean((oof_hc - y_target) ** 2)

                # HC+BM (V3)
                oof_bm, test_bm = locally_weighted_prediction(
                    X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                    bandwidth_quantile=bw, alpha=alpha)
                mse_bm = np.mean((oof_bm - y_target) ** 2)

                delta = (mse_hc - mse_bm) / mse_hc * 100
                print(f"    bw={bw:.1f} a={alpha:.0f}: HC={mse_hc:.6f} HC+BM={mse_bm:.6f} delta={delta:+.1f}%")
                configs.append({
                    'bw': bw, 'alpha': alpha,
                    'mse_hc': mse_hc, 'mse_bm': mse_bm,
                    'oof_hc': oof_hc, 'test_hc': test_hc,
                    'oof_bm': oof_bm, 'test_bm': test_bm,
                })

        # Find best HC-only and best HC+BM configs
        best_hc = min(configs, key=lambda c: c['mse_hc'])
        best_bm = min(configs, key=lambda c: c['mse_bm'])

        print(f"\n  BEST HC-only: bw={best_hc['bw']:.1f} a={best_hc['alpha']:.0f} MSE={best_hc['mse_hc']:.6f}")
        print(f"  BEST HC+BM:  bw={best_bm['bw']:.1f} a={best_bm['alpha']:.0f} MSE={best_bm['mse_bm']:.6f}")

        # Correlation between HC and HC+BM predictions
        r_oof = np.corrcoef(best_hc['oof_hc'], best_bm['oof_bm'])[0, 1]
        r_test = np.corrcoef(best_hc['test_hc'], best_bm['test_bm'])[0, 1]
        print(f"  HC vs HC+BM correlation: OOF r={r_oof:.4f}, Test r={r_test:.4f}")

        all_results[target] = {
            'configs': configs,
            'best_hc': best_hc,
            'best_bm': best_bm,
        }

    # ==============================================================
    # GENERATE SUBMISSIONS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Get V1 standalone predictions (from sub 1349)
    sub_1349 = pd.read_csv(SUBMISSION_DIR / "submission_1349.csv")

    # For each target, use the best HC+BM test predictions
    v3_preds = {}
    for target in TARGETS:
        v3_preds[target] = all_results[target]['best_bm']['test_bm']

    # Correlation with Sub 784
    print("\n  V3 (HC+BM) correlation with Sub 784:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_784[col].values, v3_preds[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # V3 standalone
    sub_num = get_next_submission_number()
    standalone = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': v3_preds['angle'],
        'scaled_depth': v3_preds['depth'],
        'scaled_left_right': v3_preds['left_right'],
    })
    standalone.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: V3 STANDALONE (HC+BM best per target)")

    # V1 correlation with V3
    print("\n  V3 vs V1 (Sub 1349) correlation:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_1349[col].values, v3_preds[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # Blend V3 with Sub 784
    blend_configs = [
        (0.00, 0.30, 0.50, "optimal weights"),
        (0.00, 0.25, 0.40, "conservative"),
        (0.00, 0.35, 0.55, "aggressive"),
        (0.10, 0.30, 0.50, "with angle"),
    ]

    for aw, dw, lw, desc in blend_configs:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*v3_preds['angle']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*v3_preds['depth']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*v3_preds['left_right']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # V1+V3 ensemble: average V1 and V3 standalone predictions, then blend
    print("\n  V1+V3 ensembles:")
    for v1_weight in [0.3, 0.5, 0.7]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            v1_pred = sub_1349[col].values
            v3_pred = v3_preds[target]
            ensemble = v1_weight * v1_pred + (1 - v1_weight) * v3_pred

            if target == 'angle':
                w = 0.0
            elif target == 'depth':
                w = 0.30
            else:
                w = 0.50
            blended[col] = (1 - w) * sub_784[col].values + w * ensemble

        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: V1+V3 ensemble (v1_w={v1_weight:.1f}, dw=0.30, lw=0.50)")

    # Best-per-source: use HC+BM for targets where it improves, HC-only for others
    print("\n  Best-per-source submission:")
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    for target in TARGETS:
        col = f'scaled_{target}'
        best_hc_mse = all_results[target]['best_hc']['mse_hc']
        best_bm_mse = all_results[target]['best_bm']['mse_bm']
        if best_bm_mse < best_hc_mse:
            pred = all_results[target]['best_bm']['test_bm']
            source = "HC+BM"
        else:
            pred = all_results[target]['best_hc']['test_hc']
            source = "HC-only"

        if target == 'angle':
            w = 0.0
        elif target == 'depth':
            w = 0.30
        else:
            w = 0.50
        blended[col] = (1 - w) * sub_784[col].values + w * pred
        print(f"    {target}: using {source} (MSE hc={best_hc_mse:.6f} bm={best_bm_mse:.6f})")

    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    a_std = blended['scaled_angle'].std()
    d_mean = blended['scaled_depth'].mean()
    print(f"  Sub {sub_num}: BEST-PER-SOURCE (dw=0.30, lw=0.50)")
    print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
