"""
Extended Physics Features Pipeline

Expands the kinetic chain approach that proved successful (KC-PLS, Sub 2475 LB 0.006485)
to include previously untapped keypoint data:

1. Right hand/finger chain (20 joints) - ball release control
2. Left hand/guide hand (20 joints) - lateral aim, separation timing
3. Torso/spine chain - energy generation and transfer
4. Two-hand interaction features - guide hand separation, relative positioning

All sub-chains compressed via PLS before adding to the Ridge pipeline,
following the proven KC-PLS approach.

Current KC uses only 5 right-side gross joints (ankle->wrist).
We have 46 hand joints + torso/spine joints that are untapped.
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
FEET_TO_METERS = 0.3048

TARGET_SCALERS = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}

PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}

BEST_BW = {"angle": 0.80, "depth": 0.55, "left_right": 0.30}
PLAYER_OVERRIDES = {(1, "left_right"): {"bw": 0.15}}


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
    """Get smoothed (240,3) trajectory for a keypoint index."""
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


# ---------------------------------------------------------------------------
# EXTENDED PHYSICS FEATURE EXTRACTION
# ---------------------------------------------------------------------------

# Right hand finger joints (shooting hand)
RIGHT_FINGER_JOINTS = [
    'right_first_finger_cmc', 'right_first_finger_mcp',
    'right_first_finger_ip', 'right_first_finger_distal',
    'right_second_finger_mcp', 'right_second_finger_pip',
    'right_second_finger_dip', 'right_second_finger_distal',
    'right_third_finger_mcp', 'right_third_finger_pip',
    'right_third_finger_dip', 'right_third_finger_distal',
    'right_fourth_finger_mcp', 'right_fourth_finger_pip',
    'right_fourth_finger_dip', 'right_fourth_finger_distal',
    'right_fifth_finger_mcp', 'right_fifth_finger_pip',
    'right_fifth_finger_dip', 'right_fifth_finger_distal',
]

# Left hand finger joints (guide hand)
LEFT_FINGER_JOINTS = [
    'left_first_finger_cmc', 'left_first_finger_mcp',
    'left_first_finger_ip', 'left_first_finger_distal',
    'left_second_finger_mcp', 'left_second_finger_pip',
    'left_second_finger_dip', 'left_second_finger_distal',
    'left_third_finger_mcp', 'left_third_finger_pip',
    'left_third_finger_dip', 'left_third_finger_distal',
    'left_fourth_finger_mcp', 'left_fourth_finger_pip',
    'left_fourth_finger_dip', 'left_fourth_finger_distal',
    'left_fifth_finger_mcp', 'left_fifth_finger_pip',
    'left_fifth_finger_dip', 'left_fifth_finger_distal',
]

# Distal (fingertip) joints
RIGHT_TIPS = [
    'right_first_finger_distal', 'right_second_finger_distal',
    'right_third_finger_distal', 'right_fourth_finger_distal',
    'right_fifth_finger_distal',
]
LEFT_TIPS = [
    'left_first_finger_distal', 'left_second_finger_distal',
    'left_third_finger_distal', 'left_fourth_finger_distal',
    'left_fifth_finger_distal',
]


def detect_release_frame(ts_3d, kp_index):
    """Detect ball release frame from wrist kinematics."""
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 150
    wrist = smooth_joint(ts_3d, rw_idx, 11)
    vel = np.gradient(wrist * FEET_TO_METERS, DT, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    # Peak speed in frames 80-200
    search = speed[80:200]
    return int(80 + np.argmax(search))


def extract_right_hand_features(ts_3d, kp_index, release_frame):
    """
    Extract shooting hand features: finger positions, velocities, spread,
    extension angles at and around the release frame.
    """
    feats = []
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return np.zeros(80, dtype=np.float32)

    wrist = smooth_joint(ts_3d, rw_idx)
    rf = int(np.clip(release_frame, 5, 234))

    # Windows around release
    pre_release = max(0, rf - 10)  # 10 frames before
    at_release = rf
    post_release = min(239, rf + 5)

    # 1. Fingertip positions/velocities at release (relative to wrist)
    tip_positions = []
    tip_velocities = []
    for tip_name in RIGHT_TIPS:
        idx = kp_index.get(tip_name)
        if idx is None:
            tip_positions.append(np.zeros(3))
            tip_velocities.append(np.zeros(3))
            continue
        tip = smooth_joint(ts_3d, idx)
        rel_pos = tip[at_release] - wrist[at_release]  # relative to wrist
        tip_vel = np.gradient(tip, DT, axis=0)[at_release]
        tip_positions.append(rel_pos)
        tip_velocities.append(tip_vel)

    # Flatten tip data: 5 tips x 3 coords x 2 (pos + vel) = 30 features
    for pos in tip_positions:
        feats.extend(pos.tolist())
    for vel in tip_velocities:
        feats.extend(vel.tolist())

    # 2. Finger spread at release: max distance between any two fingertips
    if len(tip_positions) >= 2:
        positions_arr = np.array(tip_positions)
        dists = cdist(positions_arr, positions_arr)
        feats.append(np.max(dists))  # max spread
        feats.append(np.mean(dists[np.triu_indices(5, k=1)]))  # mean spread
    else:
        feats.extend([0.0, 0.0])

    # 3. Finger spread CHANGE (pre-release to release) - opening/closing of hand
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
    feats.append(np.max(dists) - np.max(pre_dists))  # spread change
    feats.append(np.mean(dists[np.triu_indices(5, k=1)]) - np.mean(pre_dists[np.triu_indices(5, k=1)]))

    # 4. Centroid of fingertips: velocity and position
    tip_arr = np.array(tip_positions)
    centroid_pos = np.mean(tip_arr, axis=0)  # relative to wrist
    feats.extend(centroid_pos.tolist())  # 3 features

    # Centroid velocity
    all_tip_trajs = []
    for tip_name in RIGHT_TIPS:
        idx = kp_index.get(tip_name)
        if idx is None:
            continue
        all_tip_trajs.append(smooth_joint(ts_3d, idx))
    if all_tip_trajs:
        centroid_traj = np.mean(all_tip_trajs, axis=0)
        centroid_vel = np.gradient(centroid_traj, DT, axis=0)
        feats.extend(centroid_vel[at_release].tolist())  # 3 features
        feats.append(np.linalg.norm(centroid_vel[at_release]))  # speed
        # Release direction (elevation + azimuth)
        cv = centroid_vel[at_release]
        v_h = np.sqrt(cv[0]**2 + cv[1]**2)
        feats.append(np.degrees(np.arctan2(cv[2], max(v_h, 1e-8))))  # elevation
        feats.append(np.degrees(np.arctan2(cv[0], max(abs(cv[1]), 1e-8))))  # azimuth
    else:
        feats.extend([0.0] * 6)

    # 5. Wrist flexion: angle between forearm and hand (elbow->wrist->centroid)
    re_idx = kp_index.get('right_elbow')
    if re_idx is not None and all_tip_trajs:
        elbow = smooth_joint(ts_3d, re_idx)
        forearm = wrist[at_release] - elbow[at_release]
        hand_dir = centroid_pos  # wrist-relative centroid
        feats.append(_angle_between(forearm, hand_dir))
        # Wrist flexion angular velocity
        for dt_off in [-2, -1, 0, 1, 2]:
            f = int(np.clip(rf + dt_off, 5, 234))
            fore_f = wrist[f] - elbow[f]
            cent_f = np.mean([smooth_joint(ts_3d, kp_index[tn])[f] for tn in RIGHT_TIPS
                              if tn in kp_index], axis=0) - wrist[f]
            feats.append(_angle_between(fore_f, cent_f))
    else:
        feats.extend([90.0] * 6)

    # 6. Per-finger MCP joint angles at release (finger curl)
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

    # Total: 30 + 2 + 2 + 3 + 6 + 6 + 4 = 53 features (variable, padded below)
    return np.array(feats, dtype=np.float32)


def extract_left_hand_features(ts_3d, kp_index, release_frame):
    """
    Extract guide hand features: separation timing, relative position,
    guide hand role in lateral aim control.
    """
    feats = []
    lw_idx = kp_index.get('left_wrist')
    rw_idx = kp_index.get('right_wrist')
    if lw_idx is None or rw_idx is None:
        return np.zeros(30, dtype=np.float32)

    l_wrist = smooth_joint(ts_3d, lw_idx)
    r_wrist = smooth_joint(ts_3d, rw_idx)
    rf = int(np.clip(release_frame, 5, 234))

    # 1. Hand separation distance over time
    sep_dist = np.linalg.norm(l_wrist - r_wrist, axis=1)

    # Separation at key points
    feats.append(sep_dist[rf])  # at release
    feats.append(sep_dist[max(0, rf - 20)])  # 20 frames before
    feats.append(sep_dist[max(0, rf - 40)])  # 40 frames before
    feats.append(sep_dist[min(239, rf + 10)])  # after release

    # 2. Separation velocity (how fast hands move apart)
    sep_vel = np.gradient(sep_dist, DT)
    feats.append(sep_vel[rf])
    feats.append(np.max(sep_vel[max(0, rf-30):rf+5]))  # peak separation speed

    # 3. Guide hand separation frame: when left hand starts pulling away
    # Find frame where separation distance starts increasing fastest
    sep_accel = np.gradient(sep_vel, DT)
    search_start = max(0, rf - 60)
    search_end = rf
    if search_end > search_start:
        sep_frame = search_start + np.argmax(sep_accel[search_start:search_end])
        feats.append(float(sep_frame))
        feats.append(float((sep_frame - rf) * DT))  # time before release
    else:
        feats.extend([rf, 0.0])

    # 4. Left hand position relative to right at release (in 3D)
    rel_pos = l_wrist[rf] - r_wrist[rf]
    feats.extend(rel_pos.tolist())  # 3 features: x (lateral), y (forward), z (vertical)

    # 5. Left hand velocity at release
    l_vel = np.gradient(l_wrist, DT, axis=0)
    feats.extend(l_vel[rf].tolist())  # 3 features
    feats.append(np.linalg.norm(l_vel[rf]))  # speed

    # 6. Left hand lateral position relative to body center (aim alignment)
    mh_idx = kp_index.get('mid_hip')
    if mh_idx is not None:
        mid_hip = smooth_joint(ts_3d, mh_idx)
        l_lateral = l_wrist[rf, 0] - mid_hip[rf, 0]  # x offset
        feats.append(l_lateral)
    else:
        feats.append(0.0)

    # 7. Left fingertip centroid relative to right fingertip centroid
    l_tips_pos = []
    r_tips_pos = []
    for tip in LEFT_TIPS:
        idx = kp_index.get(tip)
        if idx is not None:
            l_tips_pos.append(smooth_joint(ts_3d, idx)[rf])
    for tip in RIGHT_TIPS:
        idx = kp_index.get(tip)
        if idx is not None:
            r_tips_pos.append(smooth_joint(ts_3d, idx)[rf])

    if l_tips_pos and r_tips_pos:
        l_centroid = np.mean(l_tips_pos, axis=0)
        r_centroid = np.mean(r_tips_pos, axis=0)
        rel = l_centroid - r_centroid
        feats.extend(rel.tolist())  # 3 features
        feats.append(np.linalg.norm(rel))  # distance
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # 8. Guide hand curl (are fingers spread or closed at release)
    l_tip_positions = []
    for tip in LEFT_TIPS:
        idx = kp_index.get(tip)
        if idx is not None:
            l_tip_positions.append(smooth_joint(ts_3d, idx)[rf] - l_wrist[rf])
    if len(l_tip_positions) >= 2:
        l_arr = np.array(l_tip_positions)
        l_dists = cdist(l_arr, l_arr)
        feats.append(np.max(l_dists))  # spread
    else:
        feats.append(0.0)

    # Total: 4 + 2 + 2 + 3 + 4 + 1 + 4 + 1 = 21 features (variable, padded below)
    return np.array(feats, dtype=np.float32)


def extract_torso_features(ts_3d, kp_index, release_frame):
    """
    Extract torso/spine features: rotation, lean, energy transfer timing.
    """
    feats = []
    rf = int(np.clip(release_frame, 5, 234))

    # Required joints
    joint_names = ['neck', 'mid_hip', 'right_shoulder', 'left_shoulder',
                   'right_hip', 'left_hip']
    joints = {}
    for jname in joint_names:
        idx = kp_index.get(jname)
        if idx is None:
            return np.zeros(25, dtype=np.float32)
        joints[jname] = smooth_joint(ts_3d, idx)

    # 1. Shoulder rotation angle (in horizontal plane) at multiple frames
    for f in [max(0, rf-20), max(0, rf-10), rf, min(239, rf+5)]:
        shoulder_line = joints['right_shoulder'][f, :2] - joints['left_shoulder'][f, :2]
        hip_line = joints['right_hip'][f, :2] - joints['left_hip'][f, :2]
        # Angle between shoulder and hip lines (torso twist)
        sn, hn = np.linalg.norm(shoulder_line), np.linalg.norm(hip_line)
        if sn > 1e-6 and hn > 1e-6:
            cos_a = np.clip(np.dot(shoulder_line, hip_line) / (sn * hn), -1, 1)
            feats.append(np.degrees(np.arccos(cos_a)))
        else:
            feats.append(0.0)

    # 2. Trunk lean angle (neck-hip vector vs vertical) at release
    trunk = joints['neck'][rf] - joints['mid_hip'][rf]
    vertical = np.array([0, 0, 1], dtype=np.float64)
    feats.append(_angle_between(trunk, vertical))

    # Trunk lean in forward direction
    feats.append(np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8)))
    # Trunk lean in lateral direction
    feats.append(np.degrees(np.arctan2(trunk[0], trunk[2] + 1e-8)))

    # 3. Shoulder height relative to hip (jump height proxy)
    shoulder_mid_z = (joints['right_shoulder'][rf, 2] + joints['left_shoulder'][rf, 2]) / 2
    hip_mid_z = (joints['right_hip'][rf, 2] + joints['left_hip'][rf, 2]) / 2
    feats.append(shoulder_mid_z - hip_mid_z)

    # Same at pre-release and post-release
    for f in [max(0, rf-20), min(239, rf+10)]:
        sz = (joints['right_shoulder'][f, 2] + joints['left_shoulder'][f, 2]) / 2
        hz = (joints['right_hip'][f, 2] + joints['left_hip'][f, 2]) / 2
        feats.append(sz - hz)

    # 4. Shoulder rotation angular velocity
    angles = []
    for f in range(max(0, rf-30), min(240, rf+10)):
        sl = joints['right_shoulder'][f, :2] - joints['left_shoulder'][f, :2]
        hl = joints['right_hip'][f, :2] - joints['left_hip'][f, :2]
        sn, hn = np.linalg.norm(sl), np.linalg.norm(hl)
        if sn > 1e-6 and hn > 1e-6:
            angles.append(np.degrees(np.arccos(np.clip(np.dot(sl, hl) / (sn * hn), -1, 1))))
        else:
            angles.append(angles[-1] if angles else 0.0)
    if len(angles) > 3:
        ang_vel = np.gradient(angles, DT)
        feats.append(np.max(np.abs(ang_vel)))  # peak rotation speed
        feats.append(ang_vel[min(len(ang_vel)-1, rf - max(0, rf-30))])  # rotation speed at release
    else:
        feats.extend([0.0, 0.0])

    # 5. Hip rise velocity (jump indicator)
    hip_z = joints['mid_hip'][:, 2]
    hip_vel_z = np.gradient(hip_z, DT)
    feats.append(hip_vel_z[rf])  # hip vertical velocity at release
    feats.append(np.max(hip_vel_z[max(0, rf-40):rf+5]))  # peak hip rise

    # 6. Body alignment: shoulder center directly above hip center
    shoulder_center = (joints['right_shoulder'][rf, :2] + joints['left_shoulder'][rf, :2]) / 2
    hip_center = (joints['right_hip'][rf, :2] + joints['left_hip'][rf, :2]) / 2
    offset = shoulder_center - hip_center
    feats.append(np.linalg.norm(offset))  # total offset
    feats.append(offset[0])  # lateral offset
    feats.append(offset[1])  # forward offset

    # Total: 4 + 3 + 3 + 2 + 2 + 3 = 17 features (variable)
    return np.array(feats, dtype=np.float32)


def extract_original_kc_features(ts_3d, kp_index, release_frame):
    """Original 40 kinetic chain features (for baseline comparison)."""
    feats = []
    chain_joints = ['right_ankle', 'right_knee', 'right_hip',
                    'right_shoulder', 'right_elbow', 'right_wrist']
    joint_data = {}
    for jname in chain_joints:
        idx = kp_index.get(jname)
        if idx is None:
            return np.zeros(40, dtype=np.float32)
        joint_data[jname] = smooth_joint(ts_3d, idx)

    vel_mag = {}
    vert_vel = {}
    velocities = {}
    for jname, data in joint_data.items():
        vel = np.gradient(data, DT, axis=0)
        velocities[jname] = vel
        vel_mag[jname] = np.linalg.norm(vel, axis=1)
        vert_vel[jname] = vel[:, 2]

    rf = int(np.clip(release_frame, 5, 234))
    feats.append(rf)
    prop_start = max(0, rf - 60)
    prop_end = min(240, rf + 5)

    chain_order = ['right_knee', 'right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
    peak_times = {}
    peak_mags = {}

    for jname in chain_order:
        window = vel_mag[jname][prop_start:prop_end]
        if len(window) == 0:
            peak_times[jname] = rf
            peak_mags[jname] = 0.0
            feats.extend([0.0, 0.0, 0.0])
            continue
        if jname == 'right_knee':
            v1 = joint_data['right_hip'] - joint_data['right_knee']
            v2 = joint_data['right_ankle'] - joint_data['right_knee']
            dot = np.sum(v1 * v2, axis=1)
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            denom = n1 * n2; denom[denom < 1e-10] = 1e-10
            ka = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            ext_rate = np.gradient(ka, DT)
            w = ext_rate[prop_start:prop_end]
            im = np.argmax(w); pv = w[im]; pf = prop_start + im
        elif jname == 'right_hip':
            w = vert_vel['right_hip'][prop_start:prop_end]
            im = np.argmax(w); pv = w[im]; pf = prop_start + im
        else:
            im = np.argmax(window); pv = window[im]; pf = prop_start + im
        peak_times[jname] = pf
        peak_mags[jname] = pv
        feats.extend([float(pf), float(pv), float((pf - rf) * DT)])

    pairs = [('right_knee', 'right_hip'), ('right_hip', 'right_shoulder'),
             ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist')]
    for prev_j, curr_j in pairs:
        delta = peak_times[curr_j] - peak_times[prev_j]
        feats.append(float(delta))
        feats.append(float(delta * DT))
        feats.append(float(peak_mags[curr_j] / max(abs(peak_mags[prev_j]), 1e-6)))

    for jname in chain_order:
        feats.append(float(np.sum(vel_mag[jname][prop_start:rf] ** 2)))

    v_rel = velocities['right_wrist'][rf]
    feats.extend([float(v_rel[0]), float(v_rel[1]), float(v_rel[2]),
                  float(np.linalg.norm(v_rel))])
    v_h = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
    feats.append(float(np.degrees(np.arctan2(v_rel[2], max(v_h, 1e-8)))))
    feats.append(float(np.degrees(np.arctan2(v_rel[0], max(abs(v_rel[1]), 1e-8)))))

    acc = np.gradient(velocities['right_wrist'], DT, axis=0)
    jerk = np.gradient(acc, DT, axis=0)
    jm = np.linalg.norm(jerk, axis=1)
    feats.append(float(jm[rf]))

    return np.array(feats[:40], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading and pipeline
# ---------------------------------------------------------------------------
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


def extract_base_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """198 hoop-relative + 10 joint angle features = 208 base features."""
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
    rw_i, re_i, rs_i = kp_index.get('right_wrist'), kp_index.get('right_elbow'), kp_index.get('right_shoulder')
    rh_i, lh_i = kp_index.get('right_hip'), kp_index.get('left_hip')
    rk_i, lk_i = kp_index.get('right_knee'), kp_index.get('left_knee')
    ra_i, la_i = kp_index.get('right_ankle'), kp_index.get('left_ankle')
    neck_i, mh_i, ls_i = kp_index.get('neck'), kp_index.get('mid_hip'), kp_index.get('left_shoulder')
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


def pls_augment(X_raw_train, X_raw_test, y_target, n_max=15):
    n_p = len(X_raw_train)
    pls_scaler = StandardScaler()
    raw_tr = pls_scaler.fit_transform(X_raw_train)
    raw_te = pls_scaler.transform(X_raw_test) if len(X_raw_test) > 0 else np.zeros((0, raw_tr.shape[1]))
    nc = min(n_max, n_p - n_p // 5 - 1)
    nc = max(3, nc)
    best_nc = 3
    pls_train = np.zeros((n_p, n_max), dtype=np.float32)
    pls_test = np.zeros((len(X_raw_test), n_max), dtype=np.float32)
    pls = PLSRegression(n_components=best_nc)
    pls.fit(raw_tr, y_target)
    pls_train[:, :best_nc] = pls.transform(raw_tr)
    if len(raw_te) > 0:
        pls_test[:, :best_nc] = pls.transform(raw_te)
    return pls_train, pls_test


def physics_pls_augment(physics_feats_train, physics_feats_test, y_target, n_components=3):
    """Compress physics features via PLS, exactly as proven with KC-PLS."""
    if physics_feats_train.shape[1] == 0:
        return np.zeros((len(physics_feats_train), 0)), np.zeros((len(physics_feats_test), 0))
    sc = StandardScaler()
    X_tr = sc.fit_transform(physics_feats_train)
    X_te = sc.transform(physics_feats_test) if len(physics_feats_test) > 0 else np.zeros((0, X_tr.shape[1]))

    nc = min(n_components, X_tr.shape[1], len(X_tr) - 2)
    nc = max(1, nc)
    pls = PLSRegression(n_components=nc)
    pls.fit(X_tr, y_target)
    return pls.transform(X_tr), pls.transform(X_te) if len(X_te) > 0 else np.zeros((0, nc))


def run_target(train_data, test_data, y_train, scaler, target, target_idx,
               physics_config="all"):
    """Run pipeline for one target with extended physics features."""
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    unique_pids = sorted(np.unique(pids_train))

    tidx = target_idx[target]
    y_raw = y_train[:, tidx]
    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()

    bw = BEST_BW[target]
    oof_preds = np.zeros(len(pids_train))
    test_preds = np.zeros(len(pids_test))

    t0 = time.time()
    for pid in unique_pids:
        # Check for player overrides
        override = PLAYER_OVERRIDES.get((pid, target), {})
        pid_bw = override.get('bw', bw)

        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]
        n_tr = len(tr_indices)
        n_te = len(te_indices)

        center_frame = PLAYER_FRAMES[target][pid]

        # Standard PLS from raw timeseries
        pls_train, pls_test = pls_augment(
            train_data['X_raw'][tr_mask],
            test_data['X_raw'][te_mask] if n_te > 0 else np.zeros((0, train_data['X_raw'].shape[1])),
            y_raw[tr_mask], n_max=15)

        # Extract base features (208)
        base_tr, base_te = [], []
        for i in tr_indices:
            ts_3d = train_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            base_tr.append(extract_base_features(ts_3d, ts_hr, kp_index, rf, center_frame))
        for i in te_indices:
            ts_3d = test_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            base_te.append(extract_base_features(ts_3d, ts_hr, kp_index, rf, center_frame))
        X_base_tr = np.array(base_tr, dtype=np.float32)
        X_base_te = np.array(base_te, dtype=np.float32) if n_te > 0 else np.zeros((0, X_base_tr.shape[1]))

        # Extract physics features
        kc_tr, rh_tr, lh_tr, torso_tr = [], [], [], []
        kc_te, rh_te, lh_te, torso_te = [], [], [], []

        for i in tr_indices:
            ts_3d = train_data['X_3d'][i]
            rf = detect_release_frame(ts_3d, kp_index)
            kc_tr.append(extract_original_kc_features(ts_3d, kp_index, rf))
            rh_tr.append(extract_right_hand_features(ts_3d, kp_index, rf))
            lh_tr.append(extract_left_hand_features(ts_3d, kp_index, rf))
            torso_tr.append(extract_torso_features(ts_3d, kp_index, rf))

        for i in te_indices:
            ts_3d = test_data['X_3d'][i]
            rf = detect_release_frame(ts_3d, kp_index)
            kc_te.append(extract_original_kc_features(ts_3d, kp_index, rf))
            rh_te.append(extract_right_hand_features(ts_3d, kp_index, rf))
            lh_te.append(extract_left_hand_features(ts_3d, kp_index, rf))
            torso_te.append(extract_torso_features(ts_3d, kp_index, rf))

        kc_tr = np.array(kc_tr); kc_te = np.array(kc_te) if n_te > 0 else np.zeros((0, kc_tr.shape[1]))
        rh_tr = np.array(rh_tr); rh_te = np.array(rh_te) if n_te > 0 else np.zeros((0, rh_tr.shape[1]))
        lh_tr = np.array(lh_tr); lh_te = np.array(lh_te) if n_te > 0 else np.zeros((0, lh_tr.shape[1]))
        torso_tr = np.array(torso_tr); torso_te = np.array(torso_te) if n_te > 0 else np.zeros((0, torso_tr.shape[1]))

        # Clean NaN/inf
        for arr in [kc_tr, kc_te, rh_tr, rh_te, lh_tr, lh_te, torso_tr, torso_te]:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # PLS compress each sub-chain separately
        y_p = y_scaled[tr_mask]

        kc_pls_tr, kc_pls_te = physics_pls_augment(kc_tr, kc_te, y_p, n_components=3)

        include_rh = physics_config in ("all", "right_hand")
        include_lh = physics_config in ("all", "left_hand")
        include_to = physics_config in ("all", "torso")

        if include_rh:
            rh_pls_tr, rh_pls_te = physics_pls_augment(rh_tr, rh_te, y_p, n_components=3)
        else:
            rh_pls_tr = np.zeros((n_tr, 0)); rh_pls_te = np.zeros((n_te, 0))

        if include_lh:
            lh_pls_tr, lh_pls_te = physics_pls_augment(lh_tr, lh_te, y_p, n_components=3)
        else:
            lh_pls_tr = np.zeros((n_tr, 0)); lh_pls_te = np.zeros((n_te, 0))

        if include_to:
            to_pls_tr, to_pls_te = physics_pls_augment(torso_tr, torso_te, y_p, n_components=3)
        else:
            to_pls_tr = np.zeros((n_tr, 0)); to_pls_te = np.zeros((n_te, 0))

        # Assemble: base + standard PLS + KC-PLS + right_hand-PLS + left_hand-PLS + torso-PLS
        X_tr_aug = np.hstack([X_base_tr, pls_train, kc_pls_tr, rh_pls_tr, lh_pls_tr, to_pls_tr])
        X_te_aug = np.hstack([X_base_te, pls_test, kc_pls_te, rh_pls_te, lh_pls_te, to_pls_te]) if n_te > 0 \
            else np.zeros((0, X_tr_aug.shape[1]))

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_te_s = sc.transform(X_te_aug) if n_te > 0 else np.zeros((0, X_tr_aug.shape[1]))

        # Distance matrix and kernel
        D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        all_dists = D_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(all_dists, pid_bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # LOO predictions
        oof = np.zeros(n_tr)
        for i in range(n_tr):
            weights = np.exp(-D_tr[i, :] ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof[i] = np.mean(y_scaled[tr_mask])
                continue
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_scaled[tr_mask], sample_weight=weights)
            oof[i] = ridge.predict(X_tr_s[i:i+1])[0]

        oof_preds[tr_indices] = oof

        # Test predictions
        if n_te > 0:
            D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
            tpred = np.zeros(n_te)
            for j in range(n_te):
                weights = np.exp(-D_te[j, :] ** 2 / (2 * sigma ** 2))
                if weights.sum() < 1e-10:
                    tpred[j] = np.mean(y_scaled[tr_mask])
                    continue
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr_s, y_scaled[tr_mask], sample_weight=weights)
                tpred[j] = ridge.predict(X_te_s[j:j+1])[0]
            test_preds[te_indices] = tpred

    mse = np.mean((oof_preds - y_scaled) ** 2)
    print(f"\n{target}: LOO = {mse:.6f} [{time.time()-t0:.1f}s]")
    for pid in unique_pids:
        mask = pids_train == pid
        pmse = np.mean((oof_preds[mask] - y_scaled[mask]) ** 2)
        extra = ""
        if (pid, target) in PLAYER_OVERRIDES:
            extra = f" [override: {PLAYER_OVERRIDES[(pid, target)]}]"
        print(f"  P{pid}: {pmse:.6f}{extra}")

    return oof_preds, test_preds, mse


def main():
    t0 = time.time()
    train_data, test_data = load_data()
    y_train = train_data['y']
    target_idx = {t: i for i, t in enumerate(TARGETS)}

    # ====================================================================
    # Test 1: Baseline (KC-PLS only, no new features)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Baseline (KC-PLS only)")
    print("=" * 70)

    baseline_mses = {}
    for target in TARGETS:
        scaler = TARGET_SCALERS[target]
        _, _, mse = run_target(train_data, test_data, y_train, scaler, target, target_idx,
                               physics_config="kc_only")
        baseline_mses[target] = mse

    baseline_mean = np.mean(list(baseline_mses.values()))
    print(f"\nBaseline mean LOO: {baseline_mean:.6f}")

    # ====================================================================
    # Test 2: All extended physics features
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: All extended physics features (KC + RH + LH + Torso)")
    print("=" * 70)

    all_mses = {}
    all_oof = {}
    all_test = {}
    for target in TARGETS:
        scaler = TARGET_SCALERS[target]
        oof, test, mse = run_target(train_data, test_data, y_train, scaler, target, target_idx,
                                     physics_config="all")
        all_mses[target] = mse
        all_oof[target] = oof
        all_test[target] = test

    all_mean = np.mean(list(all_mses.values()))
    print(f"\nAll physics mean LOO: {all_mean:.6f}")
    print(f"vs baseline: {(all_mean - baseline_mean) / baseline_mean * 100:+.2f}%")

    # ====================================================================
    # Test 3: Ablation - each sub-chain individually
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Ablation (each sub-chain added individually to KC)")
    print("=" * 70)

    for config_name in ["right_hand", "left_hand", "torso"]:
        mses = {}
        for target in TARGETS:
            scaler = TARGET_SCALERS[target]
            _, _, mse = run_target(train_data, test_data, y_train, scaler, target, target_idx,
                                    physics_config=config_name)
            mses[target] = mse
        m = np.mean(list(mses.values()))
        pct = (m - baseline_mean) / baseline_mean * 100
        print(f"  {config_name}: mean={m:.6f} ({pct:+.2f}%) "
              f"[a={mses['angle']:.6f} d={mses['depth']:.6f} lr={mses['left_right']:.6f}]")

    # ====================================================================
    # Generate submissions
    # ====================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Standalone submission
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        "id": test_data['ids'],
        "scaled_angle": all_test['angle'],
        "scaled_depth": all_test['depth'],
        "scaled_left_right": all_test['left_right'],
    })
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Sub {sub_num}: extended physics standalone (LOO {all_mean:.6f})")

    # Blends with Sub 2475
    anchor = pd.read_csv(SUBMISSION_DIR / "submission_2475.csv")
    anchor_vals = anchor[["scaled_angle", "scaled_depth", "scaled_left_right"]].values
    test_vals = np.column_stack([all_test['angle'], all_test['depth'], all_test['left_right']])

    for w in [0.10, 0.20, 0.30, 0.50]:
        blended = w * test_vals + (1 - w) * anchor_vals
        sub_num = get_next_submission_number()
        blend_df = pd.DataFrame({
            "id": test_data['ids'],
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2],
        })
        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blend_df.to_csv(blend_path, index=False)
        print(f"Sub {sub_num}: {int(w*100)}% extended physics + {int((1-w)*100)}% Sub 2475")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
