"""
Pseudo-Labeling Pipeline for SPLxUTSPAN 2026

Uses current best test predictions (Sub 2716) as pseudo-labels for 113 test shots,
creating an augmented training set of 458 samples (345 real + 113 pseudo).

Key design decisions:
- PLS is fit on REAL training data only (no pseudo-labels in PLS fit) to avoid circular leakage
- When predicting for a test example, that example is excluded from pseudo-labels
- Confidence-weighted variants downweight pseudo-labeled samples in the kernel
- Per-player models with Gaussian kernel, per-player optimal frames, per-target bandwidth

Pipeline matches extended_physics_features.py (the best-scoring pipeline):
- 208 base features (198 hoop-relative + 10 joint angles)
- 15 PLS components per target from raw timeseries
- KC-PLS (3 components from 40 kinetic chain features)
- Right-hand PLS (3 components from 53 fingertip features)
- Per-player bandwidth, P1 LR narrow override
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

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

TARGET_SCALERS = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}

# Per-player optimal frames from optimization
PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}

# Per-target optimal bandwidth
BEST_BW = {"angle": 0.80, "depth": 0.55, "left_right": 0.30}
PLAYER_OVERRIDES = {(1, "left_right"): {"bw": 0.15}}

# Anchor submission for blending
ANCHOR_SUB = 2716
# Pseudo-label source (same as anchor in this case)
PSEUDO_LABEL_SUB = 2716

# Confidence weights to test for pseudo-labeled examples
CONFIDENCE_WEIGHTS = [1.0, 0.5, 0.3, 0.1]


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
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


def safe_savgol(x, window=7, polyorder=2, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def smooth_joint(ts_3d, kp_idx, window=7):
    data = ts_3d[:, kp_idx, :].copy()
    for ax in range(3):
        vals = data[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            data[:, ax] = 0.0
        elif np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
            data[:, ax] = vals
        data[:, ax] = safe_savgol(data[:, ax], window, 2)
    return data


# --------------------------------------------------------------------------
# Feature extraction (matching extended_physics_features.py exactly)
# --------------------------------------------------------------------------
RIGHT_TIPS = [
    'right_first_finger_distal', 'right_second_finger_distal',
    'right_third_finger_distal', 'right_fourth_finger_distal',
    'right_fifth_finger_distal',
]


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


def extract_base_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """208 base features: 198 hoop-relative + 10 joint angles."""
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

    # Joint angles (10 features)
    rw_i = kp_index.get('right_wrist')
    re_i = kp_index.get('right_elbow')
    rs_i = kp_index.get('right_shoulder')
    rh_i, lh_i = kp_index.get('right_hip'), kp_index.get('left_hip')
    rk_i, lk_i = kp_index.get('right_knee'), kp_index.get('left_knee')
    ra_i, la_i = kp_index.get('right_ankle'), kp_index.get('left_ankle')
    neck_i = kp_index.get('neck')
    mh_i = kp_index.get('mid_hip')
    ls_i = kp_index.get('left_shoulder')
    if all(x is not None for x in [rs_i, rh_i, re_i]):
        feats.append(_angle_between(ts_3d[f, rs_i] - ts_3d[f, rh_i], ts_3d[f, re_i] - ts_3d[f, rs_i]))
    else:
        feats.append(90.0)
    if neck_i is not None and mh_i is not None:
        trunk = ts_hr[f, neck_i] - ts_hr[f, mh_i]
        feats.append(_angle_between(trunk, np.array([0, 0, 1], dtype=np.float32)))
        feats.append(np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8)))
    else:
        feats.extend([0.0, 0.0])
    if all(x is not None for x in [rk_i, rh_i, ra_i]):
        feats.append(_angle_between(ts_3d[f, rh_i] - ts_3d[f, rk_i], ts_3d[f, ra_i] - ts_3d[f, rk_i]))
    else:
        feats.append(90.0)
    if all(x is not None for x in [lk_i, lh_i, la_i]):
        feats.append(_angle_between(ts_3d[f, lh_i] - ts_3d[f, lk_i], ts_3d[f, la_i] - ts_3d[f, lk_i]))
    else:
        feats.append(90.0)
    if re_i is not None and rw_i is not None:
        feats.append(_angle_between(ts_3d[f, rw_i] - ts_3d[f, re_i], np.array([0, 0, 1], dtype=np.float32)))
    else:
        feats.append(90.0)
    if rs_i is not None and rw_i is not None:
        feats.append(_angle_between(ts_hr[f, rw_i] - ts_hr[f, rs_i], np.array([1, 0, 0.5], dtype=np.float32)))
    else:
        feats.append(90.0)
    if rs_i is not None and ls_i is not None:
        feats.append(_angle_between(ts_hr[f, rs_i] - ts_hr[f, ls_i], np.array([0, 1, 0], dtype=np.float32)))
    else:
        feats.append(90.0)
    if all(x is not None for x in [rh_i, lh_i, rs_i, ls_i]):
        hip_line = ts_hr[f, rh_i, :2] - ts_hr[f, lh_i, :2]
        shoulder_xy = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
        hn, sn = np.linalg.norm(hip_line), np.linalg.norm(shoulder_xy)
        if hn > 1e-6 and sn > 1e-6:
            feats.append(np.degrees(np.arccos(np.clip(np.dot(hip_line, shoulder_xy) / (hn * sn), -1, 1))))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)
    if re_i is not None and rs_i is not None:
        feats.append(ts_hr[f, re_i, 2] - ts_hr[f, rs_i, 2])
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def extract_kinetic_chain(ts_3d, kp_index):
    """Extract 40 kinetic chain features."""
    feats = []
    chain_joints = ['right_ankle', 'right_knee', 'right_hip',
                    'right_shoulder', 'right_elbow', 'right_wrist']
    joint_data = {}
    for jname in chain_joints:
        idx = kp_index.get(jname)
        if idx is None:
            return np.zeros(40, dtype=np.float32)
        data = ts_3d[:, idx, :].copy()
        for ax in range(3):
            vals = data[:, ax]
            bad = np.isnan(vals) | np.isinf(vals)
            if np.all(bad):
                data[:, ax] = 0.0
            elif np.any(bad):
                good = ~bad
                vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
                data[:, ax] = vals
        for ax in range(3):
            data[:, ax] = safe_savgol(data[:, ax], 7, 2)
        joint_data[jname] = data

    velocities = {}
    vel_mag = {}
    vert_vel = {}
    for jname, data in joint_data.items():
        vel = np.gradient(data, DT, axis=0)
        velocities[jname] = vel
        vel_mag[jname] = np.linalg.norm(vel, axis=1)
        vert_vel[jname] = vel[:, 2]

    wrist_speed = vel_mag['right_wrist']
    release_frame = 120 + np.argmax(wrist_speed[120:])
    release_frame = int(np.clip(release_frame, 120, 230))
    feats.append(release_frame)

    prop_start = max(0, release_frame - 60)
    prop_end = min(240, release_frame + 5)

    chain_order = ['right_knee', 'right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
    peak_times = {}
    peak_mags = {}

    for jname in chain_order:
        window = vel_mag[jname][prop_start:prop_end]
        if len(window) == 0:
            peak_times[jname] = release_frame
            peak_mags[jname] = 0.0
            feats.extend([0.0, 0.0, 0.0])
            continue

        if jname == 'right_knee':
            hip_pos = joint_data['right_hip']
            knee_pos = joint_data['right_knee']
            ankle_pos = joint_data['right_ankle']
            v1 = hip_pos - knee_pos
            v2 = ankle_pos - knee_pos
            dot = np.sum(v1 * v2, axis=1)
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            denom = n1 * n2
            denom[denom < 1e-10] = 1e-10
            knee_angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            ext_rate = np.gradient(knee_angle, DT)
            window_ext = ext_rate[prop_start:prop_end]
            idx_max = np.argmax(window_ext)
            peak_val = window_ext[idx_max]
            peak_frame = prop_start + idx_max
        elif jname == 'right_hip':
            window_vert = vert_vel['right_hip'][prop_start:prop_end]
            idx_max = np.argmax(window_vert)
            peak_val = window_vert[idx_max]
            peak_frame = prop_start + idx_max
        else:
            idx_max = np.argmax(window)
            peak_val = window[idx_max]
            peak_frame = prop_start + idx_max

        peak_times[jname] = peak_frame
        peak_mags[jname] = peak_val
        feats.append(float(peak_frame))
        feats.append(float(peak_val))
        feats.append(float((peak_frame - release_frame) * DT))

    pairs = [('right_knee', 'right_hip'), ('right_hip', 'right_shoulder'),
             ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist')]
    for prev_j, curr_j in pairs:
        delta = peak_times[curr_j] - peak_times[prev_j]
        feats.append(float(delta))
        feats.append(float(delta * DT))
        if abs(peak_mags[prev_j]) > 1e-6:
            feats.append(float(peak_mags[curr_j] / peak_mags[prev_j]))
        else:
            feats.append(1.0)

    for jname in chain_order:
        energy = np.sum(vel_mag[jname][prop_start:release_frame] ** 2)
        feats.append(float(energy))

    v_rel = velocities['right_wrist'][release_frame]
    feats.append(float(v_rel[0]))
    feats.append(float(v_rel[1]))
    feats.append(float(v_rel[2]))
    feats.append(float(np.linalg.norm(v_rel)))
    v_horiz = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
    feats.append(float(np.degrees(np.arctan2(v_rel[2], max(v_horiz, 1e-8)))))
    feats.append(float(np.degrees(np.arctan2(v_rel[0], max(abs(v_rel[1]), 1e-8)))))

    acc = np.gradient(velocities['right_wrist'], DT, axis=0)
    jerk = np.gradient(acc, DT, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    feats.append(float(jerk_mag[release_frame]))

    return np.array(feats[:40], dtype=np.float32)


def extract_right_hand_features(ts_3d, kp_index, release_frame):
    """Extract shooting hand features - 53 features."""
    feats = []
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return np.zeros(53, dtype=np.float32)

    wrist = smooth_joint(ts_3d, rw_idx)
    rf = int(np.clip(release_frame, 5, 234))
    pre_release = max(0, rf - 10)
    at_release = rf
    post_release = min(239, rf + 5)

    tip_positions = []
    tip_velocities = []
    for tip_name in RIGHT_TIPS:
        idx = kp_index.get(tip_name)
        if idx is None:
            tip_positions.append(np.zeros(3))
            tip_velocities.append(np.zeros(3))
            continue
        tip = smooth_joint(ts_3d, idx)
        rel_pos = tip[at_release] - wrist[at_release]
        tip_vel = np.gradient(tip, DT, axis=0)[at_release]
        tip_positions.append(rel_pos)
        tip_velocities.append(tip_vel)

    for pos in tip_positions:
        feats.extend(pos.tolist())
    for vel in tip_velocities:
        feats.extend(vel.tolist())

    if len(tip_positions) >= 2:
        positions_arr = np.array(tip_positions)
        dists = cdist(positions_arr, positions_arr)
        feats.append(np.max(dists))
        feats.append(np.mean(dists[np.triu_indices(5, k=1)]))
    else:
        feats.extend([0.0, 0.0])

    pre_tips = []
    for tip_name in RIGHT_TIPS:
        idx = kp_index.get(tip_name)
        if idx is None:
            pre_tips.append(np.zeros(3))
            continue
        tip = smooth_joint(ts_3d, idx)
        pre_tips.append(tip[pre_release] - wrist[pre_release])
    pre_arr = np.array(pre_tips)
    pre_dists = cdist(pre_arr, pre_arr)
    feats.append(np.max(dists) - np.max(pre_dists))
    feats.append(np.mean(dists[np.triu_indices(5, k=1)]) - np.mean(pre_dists[np.triu_indices(5, k=1)]))

    tip_arr = np.array(tip_positions)
    centroid_pos = np.mean(tip_arr, axis=0)
    feats.extend(centroid_pos.tolist())

    all_tip_trajs = []
    for tip_name in RIGHT_TIPS:
        idx = kp_index.get(tip_name)
        if idx is None:
            continue
        all_tip_trajs.append(smooth_joint(ts_3d, idx))
    if all_tip_trajs:
        centroid_traj = np.mean(all_tip_trajs, axis=0)
        centroid_vel = np.gradient(centroid_traj, DT, axis=0)
        feats.extend(centroid_vel[at_release].tolist())
        feats.append(np.linalg.norm(centroid_vel[at_release]))
        cv = centroid_vel[at_release]
        v_h = np.sqrt(cv[0]**2 + cv[1]**2)
        feats.append(np.degrees(np.arctan2(cv[2], max(v_h, 1e-8))))
        feats.append(np.degrees(np.arctan2(cv[0], max(abs(cv[1]), 1e-8))))
    else:
        feats.extend([0.0] * 6)

    re_idx = kp_index.get('right_elbow')
    if re_idx is not None and all_tip_trajs:
        elbow = smooth_joint(ts_3d, re_idx)
        forearm = wrist[at_release] - elbow[at_release]
        hand_dir = centroid_pos
        feats.append(_angle_between(forearm, hand_dir))
        for dt_off in [-2, -1, 0, 1, 2]:
            f_idx = int(np.clip(rf + dt_off, 5, 234))
            fore_f = wrist[f_idx] - elbow[f_idx]
            cent_f = np.mean([smooth_joint(ts_3d, kp_index[tn])[f_idx] for tn in RIGHT_TIPS
                              if tn in kp_index], axis=0) - wrist[f_idx]
            feats.append(_angle_between(fore_f, cent_f))
    else:
        feats.extend([90.0] * 6)

    for finger_base, finger_tip_name in [
        ('right_second_finger_mcp', 'right_second_finger_distal'),
        ('right_third_finger_mcp', 'right_third_finger_distal'),
        ('right_fourth_finger_mcp', 'right_fourth_finger_distal'),
        ('right_fifth_finger_mcp', 'right_fifth_finger_distal'),
    ]:
        b_idx = kp_index.get(finger_base)
        t_idx = kp_index.get(finger_tip_name)
        if b_idx is not None and t_idx is not None:
            base = smooth_joint(ts_3d, b_idx)
            tip = smooth_joint(ts_3d, t_idx)
            finger_vec = tip[at_release] - base[at_release]
            hand_vec = base[at_release] - wrist[at_release]
            feats.append(_angle_between(finger_vec, hand_vec))
        else:
            feats.append(90.0)

    result = np.array(feats, dtype=np.float32)
    # Pad or truncate to exactly 53
    if len(result) < 53:
        result = np.pad(result, (0, 53 - len(result)), constant_values=0.0)
    return result[:53]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
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
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index,
                  'n_kp_cols': n_kp_cols}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result
    return process(train_df, True), process(test_df, False)


def load_pseudo_labels():
    """Load pseudo-labels from anchor submission."""
    sub_path = SUBMISSION_DIR / f"submission_{PSEUDO_LABEL_SUB}.csv"
    pseudo_df = pd.read_csv(sub_path)
    print(f"Loaded pseudo-labels from Sub {PSEUDO_LABEL_SUB}: {len(pseudo_df)} test rows")
    return pseudo_df


# --------------------------------------------------------------------------
# PLS helpers
# --------------------------------------------------------------------------
def pls_augment_real_only(X_raw_real, X_raw_pseudo, X_raw_query, y_real, n_max=15):
    """
    Fit PLS on REAL training data only, then transform real, pseudo, and query.
    This avoids circular leakage from pseudo-labels into PLS.
    """
    n_real = len(X_raw_real)
    pls_scaler = StandardScaler()
    raw_real_s = pls_scaler.fit_transform(X_raw_real)
    raw_pseudo_s = pls_scaler.transform(X_raw_pseudo) if len(X_raw_pseudo) > 0 else np.zeros((0, raw_real_s.shape[1]))
    raw_query_s = pls_scaler.transform(X_raw_query) if len(X_raw_query) > 0 else np.zeros((0, raw_real_s.shape[1]))

    nc = min(n_max, n_real - n_real // 5 - 1)
    nc = max(3, nc)
    pls = PLSRegression(n_components=nc)
    pls.fit(raw_real_s, y_real)

    pls_real = pls.transform(raw_real_s)
    pls_pseudo = pls.transform(raw_pseudo_s) if len(raw_pseudo_s) > 0 else np.zeros((0, nc))
    pls_query = pls.transform(raw_query_s) if len(raw_query_s) > 0 else np.zeros((0, nc))

    # Pad to n_max columns
    def pad(arr, target_cols):
        if arr.shape[1] < target_cols:
            return np.hstack([arr, np.zeros((arr.shape[0], target_cols - arr.shape[1]))])
        return arr

    return pad(pls_real, n_max), pad(pls_pseudo, n_max), pad(pls_query, n_max)


def physics_pls_augment_real_only(feats_real, feats_pseudo, feats_query, y_real, n_components=3):
    """Compress physics features via PLS, fit on REAL data only."""
    if feats_real.shape[1] == 0:
        return (np.zeros((len(feats_real), 0)),
                np.zeros((len(feats_pseudo), 0)),
                np.zeros((len(feats_query), 0)))
    sc = StandardScaler()
    X_real = sc.fit_transform(feats_real)
    X_pseudo = sc.transform(feats_pseudo) if len(feats_pseudo) > 0 else np.zeros((0, X_real.shape[1]))
    X_query = sc.transform(feats_query) if len(feats_query) > 0 else np.zeros((0, X_real.shape[1]))

    nc = min(n_components, X_real.shape[1], len(X_real) - 2)
    nc = max(1, nc)
    pls = PLSRegression(n_components=nc)
    pls.fit(X_real, y_real)
    return (pls.transform(X_real),
            pls.transform(X_pseudo) if len(X_pseudo) > 0 else np.zeros((0, nc)),
            pls.transform(X_query) if len(X_query) > 0 else np.zeros((0, nc)))


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_pseudo_label_pipeline(train_data, test_data, pseudo_labels_df,
                               confidence_weight=1.0):
    """
    Run the full pseudo-labeling pipeline.

    For each test example:
    1. Build augmented training set = real train + pseudo-labeled test (excluding self)
    2. PLS fit on REAL data only
    3. Locally weighted Ridge with Gaussian kernel
    4. Pseudo-labeled examples get kernel weights multiplied by confidence_weight

    Returns: oof_preds (345,3), test_preds (113,3), per-target LOO MSE
    """
    y_train = train_data['y']
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    unique_pids = sorted(np.unique(pids_train))

    target_idx = {"angle": 0, "depth": 1, "left_right": 2}

    # Parse pseudo-labels: map test id -> scaled predictions
    pseudo_map = {}
    for _, row in pseudo_labels_df.iterrows():
        pseudo_map[row['id']] = {
            'angle': row['scaled_angle'],
            'depth': row['scaled_depth'],
            'left_right': row['scaled_left_right'],
        }

    all_oof = {}
    all_test = {}
    all_loo = {}

    for target in TARGETS:
        t0 = time.time()
        tidx = target_idx[target]
        scaler = TARGET_SCALERS[target]
        y_raw = y_train[:, tidx]
        y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()

        bw_default = BEST_BW[target]

        oof_preds = np.zeros(len(pids_train))
        test_preds = np.zeros(len(pids_test))

        for pid in unique_pids:
            override = PLAYER_OVERRIDES.get((pid, target), {})
            pid_bw = override.get('bw', bw_default)

            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            tr_indices = np.where(tr_mask)[0]
            te_indices = np.where(te_mask)[0]
            n_tr = len(tr_indices)
            n_te = len(te_indices)

            center_frame = PLAYER_FRAMES[target][pid]

            # ---------------------------------------------------------------
            # Extract features for REAL training samples (this player)
            # ---------------------------------------------------------------
            base_tr = []
            kc_tr = []
            rh_tr = []
            for i in tr_indices:
                ts_3d = train_data['X_3d'][i]
                ts_hr = compute_hoop_transform(ts_3d, kp_index)
                rf = detect_release_frame(ts_3d, kp_index)
                base_tr.append(extract_base_features(ts_3d, ts_hr, kp_index, rf, center_frame))
                kc_tr.append(extract_kinetic_chain(ts_3d, kp_index))
                rh_tr.append(extract_right_hand_features(ts_3d, kp_index, rf))

            X_base_tr = np.nan_to_num(np.array(base_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            kc_tr = np.nan_to_num(np.array(kc_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            rh_tr = np.nan_to_num(np.array(rh_tr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            # ---------------------------------------------------------------
            # Extract features for PSEUDO-LABELED test samples (this player)
            # ---------------------------------------------------------------
            base_pseudo = []
            kc_pseudo = []
            rh_pseudo = []
            pseudo_y_scaled = []
            pseudo_indices_in_test = []

            for i in te_indices:
                ts_3d = test_data['X_3d'][i]
                ts_hr = compute_hoop_transform(ts_3d, kp_index)
                rf = detect_release_frame(ts_3d, kp_index)
                base_pseudo.append(extract_base_features(ts_3d, ts_hr, kp_index, rf, center_frame))
                kc_pseudo.append(extract_kinetic_chain(ts_3d, kp_index))
                rh_pseudo.append(extract_right_hand_features(ts_3d, kp_index, rf))

                test_id = test_data['ids'][i]
                pseudo_y_scaled.append(pseudo_map[test_id][target])
                pseudo_indices_in_test.append(i)

            n_pseudo = len(base_pseudo)
            if n_pseudo > 0:
                X_base_pseudo = np.nan_to_num(np.array(base_pseudo, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                kc_pseudo = np.nan_to_num(np.array(kc_pseudo, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                rh_pseudo = np.nan_to_num(np.array(rh_pseudo, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                pseudo_y_arr = np.array(pseudo_y_scaled, dtype=np.float64)
            else:
                X_base_pseudo = np.zeros((0, X_base_tr.shape[1]))
                kc_pseudo = np.zeros((0, kc_tr.shape[1]))
                rh_pseudo = np.zeros((0, rh_tr.shape[1]))
                pseudo_y_arr = np.zeros(0)

            # ---------------------------------------------------------------
            # PLS: fit on REAL training data only, transform real+pseudo
            # ---------------------------------------------------------------
            y_real_raw = y_raw[tr_mask]

            # Standard PLS from raw timeseries
            pls_tr, pls_pseudo_out, _ = pls_augment_real_only(
                train_data['X_raw'][tr_mask],
                test_data['X_raw'][te_mask] if n_pseudo > 0 else np.zeros((0, train_data['X_raw'].shape[1])),
                np.zeros((0, train_data['X_raw'].shape[1])),  # no separate query set yet
                y_real_raw, n_max=15)

            # KC PLS
            kc_pls_tr, kc_pls_pseudo, _ = physics_pls_augment_real_only(
                kc_tr, kc_pseudo, np.zeros((0, kc_tr.shape[1])),
                y_scaled[tr_mask], n_components=3)

            # Right-hand PLS
            rh_pls_tr, rh_pls_pseudo, _ = physics_pls_augment_real_only(
                rh_tr, rh_pseudo, np.zeros((0, rh_tr.shape[1])),
                y_scaled[tr_mask], n_components=3)

            # ---------------------------------------------------------------
            # Assemble augmented feature matrices
            # ---------------------------------------------------------------
            X_tr_aug = np.hstack([X_base_tr, pls_tr, kc_pls_tr, rh_pls_tr])
            if n_pseudo > 0:
                X_pseudo_aug = np.hstack([X_base_pseudo, pls_pseudo_out, kc_pls_pseudo, rh_pls_pseudo])
            else:
                X_pseudo_aug = np.zeros((0, X_tr_aug.shape[1]))

            # Combined: real (n_tr) + pseudo (n_pseudo)
            X_combined = np.vstack([X_tr_aug, X_pseudo_aug])
            y_combined = np.concatenate([y_scaled[tr_mask], pseudo_y_arr])
            n_combined = len(X_combined)

            # Track which rows are pseudo
            is_pseudo = np.zeros(n_combined, dtype=bool)
            is_pseudo[n_tr:] = True

            # Standardize
            sc = StandardScaler()
            X_combined_s = sc.fit_transform(X_combined)
            X_real_s = X_combined_s[:n_tr]
            X_pseudo_s = X_combined_s[n_tr:]

            # Distance matrix (full combined)
            D_full = cdist(X_combined_s, X_combined_s, 'euclidean')
            # Sigma from REAL-REAL distances only (avoid pseudo inflation)
            D_real = D_full[:n_tr, :n_tr]
            real_dists = D_real[np.triu_indices(n_tr, k=1)]
            sigma = np.quantile(real_dists, pid_bw) if len(real_dists) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            # ---------------------------------------------------------------
            # LOO on REAL training data (with pseudo as extra neighbors)
            # ---------------------------------------------------------------
            for li in range(n_tr):
                weights = np.exp(-D_full[li, :] ** 2 / (2 * sigma ** 2))
                weights[li] = 0  # exclude self
                # Downweight pseudo-labeled samples
                weights[n_tr:] *= confidence_weight
                if weights.sum() < 1e-10:
                    oof_preds[tr_indices[li]] = np.mean(y_scaled[tr_mask])
                    continue
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_combined_s, y_combined, sample_weight=weights)
                oof_preds[tr_indices[li]] = ridge.predict(X_combined_s[li:li+1])[0]

            # ---------------------------------------------------------------
            # Test predictions (each pseudo example excluded when predicting itself)
            # ---------------------------------------------------------------
            if n_pseudo > 0:
                # Distance from pseudo rows to all combined rows
                for j_local in range(n_pseudo):
                    j_global = n_tr + j_local  # index in combined array
                    weights = np.exp(-D_full[j_global, :] ** 2 / (2 * sigma ** 2))
                    weights[j_global] = 0  # exclude self (critical!)
                    # Downweight OTHER pseudo-labeled samples (not self, already 0)
                    pseudo_mask_others = is_pseudo.copy()
                    pseudo_mask_others[j_global] = False
                    weights[pseudo_mask_others] *= confidence_weight
                    if weights.sum() < 1e-10:
                        test_preds[te_indices[j_local]] = np.mean(y_scaled[tr_mask])
                        continue
                    ridge = Ridge(alpha=10.0)
                    ridge.fit(X_combined_s, y_combined, sample_weight=weights)
                    test_preds[te_indices[j_local]] = ridge.predict(X_combined_s[j_global:j_global+1])[0]

        # Clip predictions
        oof_preds = np.clip(oof_preds, 0.0, 1.0)
        test_preds = np.clip(test_preds, 0.0, 1.0)

        loo = np.mean((oof_preds - y_scaled) ** 2)
        all_oof[target] = oof_preds.copy()
        all_test[target] = test_preds.copy()
        all_loo[target] = loo

        elapsed = time.time() - t0
        print(f"  {target}: LOO = {loo:.6f} [{elapsed:.1f}s]")
        for pid in unique_pids:
            mask = pids_train == pid
            pmse = np.mean((oof_preds[mask] - y_scaled[mask]) ** 2)
            extra = ""
            if (pid, target) in PLAYER_OVERRIDES:
                extra = f" [override: {PLAYER_OVERRIDES[(pid, target)]}]"
            print(f"    P{pid}: {pmse:.6f}{extra}")

    mean_loo = np.mean([all_loo[t] for t in TARGETS])
    return all_oof, all_test, all_loo, mean_loo


def main():
    t0 = time.time()
    print("=" * 70)
    print("PSEUDO-LABELING PIPELINE")
    print(f"Pseudo-labels from Sub {PSEUDO_LABEL_SUB}, anchor Sub {ANCHOR_SUB}")
    print("=" * 70)

    train_data, test_data = load_data()
    pseudo_labels_df = load_pseudo_labels()

    # Load anchor for blending
    anchor_df = pd.read_csv(SUBMISSION_DIR / f"submission_{ANCHOR_SUB}.csv")
    anchor_vals = anchor_df[["scaled_angle", "scaled_depth", "scaled_left_right"]].values

    results = {}

    # =================================================================
    # Run each confidence weight
    # =================================================================
    for cw in CONFIDENCE_WEIGHTS:
        print(f"\n{'='*70}")
        print(f"CONFIDENCE WEIGHT = {cw}")
        print(f"{'='*70}")

        all_oof, all_test, all_loo, mean_loo = run_pseudo_label_pipeline(
            train_data, test_data, pseudo_labels_df, confidence_weight=cw)

        results[cw] = {
            'oof': all_oof,
            'test': all_test,
            'loo': all_loo,
            'mean_loo': mean_loo,
        }

    # =================================================================
    # Summary comparison
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All confidence weights")
    print("=" * 70)
    print(f"{'Weight':>8} | {'Angle LOO':>12} | {'Depth LOO':>12} | {'LR LOO':>12} | {'Mean LOO':>12}")
    print("-" * 70)
    for cw in CONFIDENCE_WEIGHTS:
        r = results[cw]
        print(f"{cw:>8.2f} | {r['loo']['angle']:>12.6f} | {r['loo']['depth']:>12.6f} | "
              f"{r['loo']['left_right']:>12.6f} | {r['mean_loo']:>12.6f}")

    # Find best confidence weight
    best_cw = min(CONFIDENCE_WEIGHTS, key=lambda cw: results[cw]['mean_loo'])
    print(f"\nBest confidence weight: {best_cw} (LOO {results[best_cw]['mean_loo']:.6f})")

    # =================================================================
    # Diversity analysis with anchor
    # =================================================================
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS vs Sub " + str(ANCHOR_SUB))
    print("=" * 70)

    for cw in CONFIDENCE_WEIGHTS:
        test_vals = np.column_stack([results[cw]['test'][t] for t in TARGETS])
        corrs = []
        for col in range(3):
            r = np.corrcoef(test_vals[:, col], anchor_vals[:, col])[0, 1]
            corrs.append(r)
        print(f"  cw={cw:.2f}: corr angle={corrs[0]:.4f}, depth={corrs[1]:.4f}, "
              f"LR={corrs[2]:.4f}, mean={np.mean(corrs):.4f}")

    # =================================================================
    # Generate submissions
    # =================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    submissions = []

    # 1. Standalone pseudo-label (cw=1.0)
    cw_1 = 1.0
    test_1 = results[cw_1]['test']
    sn = get_next_submission_number()
    sub_df = pd.DataFrame({
        "id": test_data['ids'],
        "scaled_angle": test_1['angle'],
        "scaled_depth": test_1['depth'],
        "scaled_left_right": test_1['left_right'],
    })
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
    desc = f"Sub {sn}: pseudo-label standalone cw=1.0 (LOO {results[cw_1]['mean_loo']:.6f})"
    print(desc)
    submissions.append((sn, desc))

    # 2. 10% pseudo (cw=1.0) + 90% anchor
    test_vals_1 = np.column_stack([test_1['angle'], test_1['depth'], test_1['left_right']])
    for w in [0.10, 0.05]:
        sn = get_next_submission_number()
        blended = w * test_vals_1 + (1 - w) * anchor_vals
        bl_df = pd.DataFrame({
            "id": test_data['ids'],
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2],
        })
        bl_df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        desc = f"Sub {sn}: {int(w*100)}% pseudo cw=1.0 + {int((1-w)*100)}% Sub {ANCHOR_SUB}"
        print(desc)
        submissions.append((sn, desc))

    # 3. Best confidence-weighted standalone
    best_test = results[best_cw]['test']
    sn = get_next_submission_number()
    sub_df = pd.DataFrame({
        "id": test_data['ids'],
        "scaled_angle": best_test['angle'],
        "scaled_depth": best_test['depth'],
        "scaled_left_right": best_test['left_right'],
    })
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
    desc = f"Sub {sn}: pseudo-label standalone cw={best_cw} (LOO {results[best_cw]['mean_loo']:.6f})"
    print(desc)
    submissions.append((sn, desc))

    # 4. 10% best cw + 90% anchor
    best_test_vals = np.column_stack([best_test['angle'], best_test['depth'], best_test['left_right']])
    for w in [0.10, 0.05]:
        sn = get_next_submission_number()
        blended = w * best_test_vals + (1 - w) * anchor_vals
        bl_df = pd.DataFrame({
            "id": test_data['ids'],
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2],
        })
        bl_df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        desc = f"Sub {sn}: {int(w*100)}% pseudo cw={best_cw} + {int((1-w)*100)}% Sub {ANCHOR_SUB}"
        print(desc)
        submissions.append((sn, desc))

    # If best_cw != 1.0 and best_cw != 0.5, also do cw=0.5 blends
    if best_cw != 0.5 and 0.5 in results:
        mid_test = results[0.5]['test']
        mid_test_vals = np.column_stack([mid_test['angle'], mid_test['depth'], mid_test['left_right']])
        sn = get_next_submission_number()
        blended = 0.10 * mid_test_vals + 0.90 * anchor_vals
        bl_df = pd.DataFrame({
            "id": test_data['ids'],
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2],
        })
        bl_df.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        desc = f"Sub {sn}: 10% pseudo cw=0.5 + 90% Sub {ANCHOR_SUB}"
        print(desc)
        submissions.append((sn, desc))

    total = time.time() - t0
    print(f"\nTotal runtime: {total:.1f}s")

    # =================================================================
    # Report
    # =================================================================
    print("\n" + "=" * 70)
    print("FULL REPORT")
    print("=" * 70)
    print(f"\nPipeline: Pseudo-labeling with Sub {PSEUDO_LABEL_SUB} as pseudo-labels")
    print(f"Real training: 345, Pseudo-labeled: 113, Total: 458")
    print(f"PLS fit on REAL data only (no circular leakage)")
    print(f"Each test example excluded from pseudo-labels when predicting itself")
    print(f"Base features: 208 (198 hoop-relative + 10 joint angles)")
    print(f"PLS: 15 components per target from raw timeseries")
    print(f"KC-PLS: 3 components from 40 kinetic chain features")
    print(f"RH-PLS: 3 components from 53 right-hand features")
    print(f"Per-player models with Gaussian kernel")
    print(f"Per-player frames: {PLAYER_FRAMES}")
    print(f"Per-target BW: {BEST_BW}")
    print(f"Ridge alpha: 10.0")
    print(f"\nConfidence weight comparison:")
    for cw in CONFIDENCE_WEIGHTS:
        r = results[cw]
        print(f"  cw={cw:.1f}: angle={r['loo']['angle']:.6f}, depth={r['loo']['depth']:.6f}, "
              f"LR={r['loo']['left_right']:.6f}, mean={r['mean_loo']:.6f}")
    print(f"\nBest: cw={best_cw}")
    print(f"\nSubmissions generated:")
    for sn, desc in submissions:
        print(f"  {desc}")
    print(f"\nTotal runtime: {total:.1f}s")


if __name__ == "__main__":
    main()
