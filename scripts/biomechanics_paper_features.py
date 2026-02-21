"""
Biomechanics Paper Features Pipeline

Implements novel features from three research papers:
1. Cabarkapa et al. (2023) - COM velocity, knee angular velocity profile, release height
2. Li et al. (2025) - Coupled Angular Variability (CAV), coupling angle shift
3. Chen et al. (2026) - Angular impulse proxies, VV/HV ratio, RTD proxies

These are genuinely NEW features not already captured by our pipeline:
- COM velocity (all 69 keypoints, not just hip)
- Knee/elbow/wrist ANGULAR velocity time series (not just position at one frame)
- Angular impulse proxies (integral of angular velocity over release window)
- VV/HV decomposition ratio
- Knee peak power proxy (angular velocity * angular acceleration)
- Per-player CAV (cross-shot coordination variability)
- Coupling angle at release and shift (proximal->distal transition)

Pipeline: extract -> PLS compress per player -> Ridge LOO -> blend with Sub2716/Sub3294
"""

import json
import time
import fcntl
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
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


def detect_release_frame(ts_3d, kp_index):
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 150
    wrist = smooth_joint(ts_3d, rw_idx, 11)
    vel = np.gradient(wrist * FEET_TO_METERS, DT, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    search = speed[80:200]
    return int(80 + np.argmax(search))


def compute_joint_angle_timeseries(ts_3d, kp_index, joint_name, proximal_name, distal_name):
    """Compute angle at a joint across all 240 frames.

    angle at joint = angle between (proximal - joint) and (distal - joint)
    Returns array of shape (240,) in degrees.
    """
    j_idx = kp_index.get(joint_name)
    p_idx = kp_index.get(proximal_name)
    d_idx = kp_index.get(distal_name)
    if any(x is None for x in [j_idx, p_idx, d_idx]):
        return np.full(240, 90.0)

    j_pos = smooth_joint(ts_3d, j_idx)
    p_pos = smooth_joint(ts_3d, p_idx)
    d_pos = smooth_joint(ts_3d, d_idx)

    v1 = p_pos - j_pos  # proximal direction
    v2 = d_pos - j_pos  # distal direction

    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    n1 = np.maximum(n1, 1e-6)
    n2 = np.maximum(n2, 1e-6)

    cos_angle = np.sum(v1 * v2, axis=1) / (n1.ravel() * n2.ravel())
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))


# ---------------------------------------------------------------------------
# PAPER 1: Cabarkapa - Kinematic features
# ---------------------------------------------------------------------------

def extract_cabarkapa_features(ts_3d, kp_index, release_frame):
    """
    Novel features from Cabarkapa et al. (2023):
    - COM velocity (peak and at release) using ALL keypoints
    - Knee peak/mean angular velocity
    - Elbow peak/mean angular velocity
    - Release height normalized by standing height
    - Stance width during preparatory phase
    - Joint ROM (PP to RP) for knee, hip, ankle, elbow
    """
    feats = []
    rf = int(np.clip(release_frame, 10, 234))

    # Preparatory phase: approximate as frame rf-40 (when downward motion starts)
    pp_frame = max(10, rf - 40)

    # 1. COM velocity using all available keypoints
    n_kp = ts_3d.shape[1]
    # Average position of all keypoints = rough COM
    com_pos = np.nanmean(ts_3d, axis=1)  # (240, 3)
    # Handle NaN
    for ax in range(3):
        bad = np.isnan(com_pos[:, ax]) | np.isinf(com_pos[:, ax])
        if np.any(bad) and not np.all(bad):
            good = ~bad
            com_pos[bad, ax] = np.interp(np.where(bad)[0], np.where(good)[0], com_pos[good, ax])
        elif np.all(bad):
            com_pos[:, ax] = 0.0
        com_pos[:, ax] = safe_savgol(com_pos[:, ax], 11, 2)

    com_vel = np.gradient(com_pos * FEET_TO_METERS, DT, axis=0)
    com_speed = np.linalg.norm(com_vel, axis=1)

    # COM velocity at release
    feats.append(com_speed[rf])
    # COM peak velocity in PP-to-RP window
    window = com_speed[pp_frame:min(rf + 5, 240)]
    feats.append(np.max(window) if len(window) > 0 else 0.0)
    # COM mean velocity PP to RP
    feats.append(np.mean(window) if len(window) > 0 else 0.0)
    # COM vertical velocity at release
    feats.append(com_vel[rf, 2])
    # COM horizontal speed at release
    feats.append(np.sqrt(com_vel[rf, 0]**2 + com_vel[rf, 1]**2))

    # 2. COM height at release (normalized by standing height)
    # Standing height proxy: max z range of nose-to-ankle at a calm frame (frame 30)
    nose_idx = kp_index.get('nose')
    ra_idx = kp_index.get('right_ankle')
    la_idx = kp_index.get('left_ankle')
    if nose_idx is not None and (ra_idx is not None or la_idx is not None):
        nose_z = smooth_joint(ts_3d, nose_idx)[30, 2]
        if ra_idx is not None and la_idx is not None:
            ankle_z = min(smooth_joint(ts_3d, ra_idx)[30, 2],
                          smooth_joint(ts_3d, la_idx)[30, 2])
        elif ra_idx is not None:
            ankle_z = smooth_joint(ts_3d, ra_idx)[30, 2]
        else:
            ankle_z = smooth_joint(ts_3d, la_idx)[30, 2]
        standing_height = max(abs(nose_z - ankle_z), 0.5)
    else:
        standing_height = 5.5  # ~5.5 feet average

    feats.append(com_pos[rf, 2] / standing_height)  # COM height normalized
    feats.append(com_pos[pp_frame, 2] / standing_height)  # COM height at PP

    # 3. Release height normalized
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is not None:
        wrist = smooth_joint(ts_3d, rw_idx)
        feats.append(wrist[rf, 2] / standing_height)  # release height normalized
    else:
        feats.append(1.17)  # reference value

    # 4. Knee angular velocity time series
    knee_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_knee', 'right_hip', 'right_ankle')
    knee_ang_vel = np.gradient(knee_angles, DT)  # deg/s

    # Knee peak angular velocity PP to RP
    kav_window = knee_ang_vel[pp_frame:min(rf + 5, 240)]
    feats.append(np.max(np.abs(kav_window)) if len(kav_window) > 0 else 0.0)
    # Knee mean angular velocity PP to RP
    feats.append(np.mean(np.abs(kav_window)) if len(kav_window) > 0 else 0.0)
    # Knee angle ROM (PP to RP)
    feats.append(knee_angles[rf] - knee_angles[pp_frame])
    # Knee angle at release
    feats.append(knee_angles[rf])
    # Knee angle at PP
    feats.append(knee_angles[pp_frame])

    # 5. Elbow angular velocity time series
    elbow_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_elbow', 'right_shoulder', 'right_wrist')
    elbow_ang_vel = np.gradient(elbow_angles, DT)

    eav_window = elbow_ang_vel[pp_frame:min(rf + 5, 240)]
    feats.append(np.max(np.abs(eav_window)) if len(eav_window) > 0 else 0.0)
    feats.append(np.mean(np.abs(eav_window)) if len(eav_window) > 0 else 0.0)
    feats.append(elbow_angles[rf] - elbow_angles[pp_frame])
    feats.append(elbow_angles[rf])

    # 6. Hip angular velocity
    hip_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_hip', 'mid_hip', 'right_knee')
    hip_ang_vel = np.gradient(hip_angles, DT)
    hav_window = hip_ang_vel[pp_frame:min(rf + 5, 240)]
    feats.append(np.max(np.abs(hav_window)) if len(hav_window) > 0 else 0.0)
    feats.append(np.mean(np.abs(hav_window)) if len(hav_window) > 0 else 0.0)
    feats.append(hip_angles[rf] - hip_angles[pp_frame])

    # 7. Ankle angular velocity
    ankle_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_ankle', 'right_knee', 'right_foot_index')
    ankle_ang_vel = np.gradient(ankle_angles, DT)
    aav_window = ankle_ang_vel[pp_frame:min(rf + 5, 240)]
    feats.append(np.max(np.abs(aav_window)) if len(aav_window) > 0 else 0.0)
    feats.append(ankle_angles[rf] - ankle_angles[pp_frame])

    # 8. Stance width (distance between feet) during PP
    ra_idx = kp_index.get('right_ankle')
    la_idx = kp_index.get('left_ankle')
    if ra_idx is not None and la_idx is not None:
        ra = smooth_joint(ts_3d, ra_idx)
        la = smooth_joint(ts_3d, la_idx)
        feats.append(np.linalg.norm(ra[pp_frame] - la[pp_frame]))
        # Stance alignment (forward-backward foot offset)
        feats.append(ra[pp_frame, 1] - la[pp_frame, 1])
    else:
        feats.extend([0.0, 0.0])

    # 9. Elbow height at PP (normalized)
    re_idx = kp_index.get('right_elbow')
    if re_idx is not None:
        elbow = smooth_joint(ts_3d, re_idx)
        feats.append(elbow[pp_frame, 2] / standing_height)
    else:
        feats.append(0.61)

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# PAPER 3: Chen - Joint kinetics proxies
# ---------------------------------------------------------------------------

def extract_chen_kinetics_features(ts_3d, kp_index, release_frame):
    """
    Kinematic proxies for Chen et al. (2026) kinetic variables:
    - Angular Impulse proxy (integral of |angular velocity|)
    - Peak power proxy (angular velocity * angular acceleration)
    - RTD proxy (peak angular acceleration)
    - VV/HV decomposition of release velocity
    """
    feats = []
    rf = int(np.clip(release_frame, 10, 234))

    # Release window: last ~30% of shot before release
    # Approximation: flexion-to-extension switch ~ rf-30
    release_start = max(10, rf - 30)

    # 1. Wrist angular impulse proxy
    # Wrist flexion angle = angle between forearm and hand
    re_idx = kp_index.get('right_elbow')
    rw_idx = kp_index.get('right_wrist')
    # Use second finger MCP as hand direction
    hand_idx = kp_index.get('right_second_finger_mcp')
    if hand_idx is None:
        hand_idx = kp_index.get('right_first_finger_mcp')

    if all(x is not None for x in [re_idx, rw_idx]) and hand_idx is not None:
        elbow = smooth_joint(ts_3d, re_idx)
        wrist = smooth_joint(ts_3d, rw_idx)
        hand = smooth_joint(ts_3d, hand_idx)

        # Wrist flexion angle time series
        forearm = wrist - elbow  # (240, 3)
        hand_dir = hand - wrist
        wrist_angles = np.zeros(240)
        for f in range(240):
            wrist_angles[f] = _angle_between(forearm[f], hand_dir[f])
        wrist_angles = safe_savgol(wrist_angles, 7, 2)
        wrist_ang_vel = np.gradient(wrist_angles, DT)

        # Angular impulse proxy = integral of |angular velocity| over release window
        window = np.abs(wrist_ang_vel[release_start:rf + 1])
        feats.append(np.trapz(window, dx=DT))  # wrist AI proxy
        feats.append(np.max(window))  # wrist peak angular velocity
        feats.append(np.mean(window))  # wrist mean angular velocity
    else:
        feats.extend([0.0, 0.0, 0.0])

    # 2. Elbow angular impulse proxy
    elbow_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_elbow', 'right_shoulder', 'right_wrist')
    elbow_ang_vel = np.gradient(elbow_angles, DT)

    window = np.abs(elbow_ang_vel[release_start:rf + 1])
    feats.append(np.trapz(window, dx=DT))  # elbow AI proxy
    feats.append(np.max(window))  # elbow peak angular velocity

    # Elbow angular acceleration (for RTD proxy)
    elbow_ang_acc = np.gradient(elbow_ang_vel, DT)
    acc_window = np.abs(elbow_ang_acc[release_start:rf + 1])
    feats.append(np.max(acc_window))  # elbow RTD proxy (peak acceleration)

    # 3. Shoulder angular impulse and RTD
    shoulder_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_shoulder', 'neck', 'right_elbow')
    shoulder_ang_vel = np.gradient(shoulder_angles, DT)

    window = np.abs(shoulder_ang_vel[release_start:rf + 1])
    feats.append(np.trapz(window, dx=DT))  # shoulder AI proxy

    shoulder_ang_acc = np.gradient(shoulder_ang_vel, DT)
    acc_window = np.abs(shoulder_ang_acc[release_start:rf + 1])
    feats.append(np.max(acc_window))  # shoulder RTD proxy

    # 4. Knee peak power proxy
    knee_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_knee', 'right_hip', 'right_ankle')
    knee_ang_vel = np.gradient(knee_angles, DT)
    knee_ang_acc = np.gradient(knee_ang_vel, DT)

    # Power proxy ~ angular_velocity * angular_acceleration (simplified)
    power_proxy = np.abs(knee_ang_vel * knee_ang_acc)
    pp_window = power_proxy[max(10, rf - 50):rf + 1]
    feats.append(np.max(pp_window) if len(pp_window) > 0 else 0.0)  # knee peak power proxy
    # Knee RTD proxy
    kacc_window = np.abs(knee_ang_acc[max(10, rf - 50):rf + 1])
    feats.append(np.max(kacc_window) if len(kacc_window) > 0 else 0.0)

    # 5. VV/HV decomposition of release velocity
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is not None:
        wrist = smooth_joint(ts_3d, rw_idx)
        wrist_vel = np.gradient(wrist * FEET_TO_METERS, DT, axis=0)
        vv = wrist_vel[rf, 2]  # vertical velocity
        hv = np.sqrt(wrist_vel[rf, 0]**2 + wrist_vel[rf, 1]**2)  # horizontal velocity

        feats.append(vv)
        feats.append(hv)
        feats.append(vv / max(hv, 1e-6))  # VV/HV ratio
        feats.append(np.linalg.norm(wrist_vel[rf]))  # total release speed
    else:
        feats.extend([0.0, 0.0, 1.0, 0.0])

    # 6. Wrist peak power proxy
    if all(x is not None for x in [re_idx, rw_idx]) and hand_idx is not None:
        wrist_ang_acc = np.gradient(wrist_ang_vel, DT)
        wrist_power = np.abs(wrist_ang_vel * wrist_ang_acc)
        wp_window = wrist_power[release_start:rf + 1]
        feats.append(np.max(wp_window) if len(wp_window) > 0 else 0.0)
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# PAPER 2: Li - Coupled Angular Variability
# ---------------------------------------------------------------------------

def compute_coupling_angle(theta_P, theta_D):
    """Compute coupling angle between proximal and distal joint angles.

    Following Li et al. (2025) vector coding method.
    theta_P, theta_D: 1D arrays of joint angles (same length).
    Returns array of coupling angles in [0, 360] degrees.
    """
    n = len(theta_P) - 1
    gamma = np.zeros(n)

    for i in range(n):
        dP = theta_P[i + 1] - theta_P[i]
        dD = theta_D[i + 1] - theta_D[i]

        if abs(dP) < 1e-10 and abs(dD) < 1e-10:
            gamma[i] = 0.0  # undefined, use 0
        elif abs(dP) < 1e-10:
            gamma[i] = 90.0 if dD > 0 else 270.0
        elif abs(dD) < 1e-10:
            gamma[i] = 0.0 if dP > 0 else 180.0
        else:
            g = np.degrees(np.arctan2(dD, dP))
            if g < 0:
                g += 360.0
            gamma[i] = g

    return gamma


def compute_cav(coupling_angles_list):
    """Compute Coupled Angular Variability across multiple trials.

    coupling_angles_list: list of arrays, each of shape (n_timepoints,)
        from different shots of the same player.
    Returns: mean CAV across normalized time, and CAV at specific phases.
    """
    if len(coupling_angles_list) < 3:
        return np.zeros(5)

    # Normalize all to same length (100 points)
    norm_length = 100
    normalized = []
    for ca in coupling_angles_list:
        if len(ca) < 5:
            continue
        x_old = np.linspace(0, 1, len(ca))
        x_new = np.linspace(0, 1, norm_length)
        # Interpolate using circular-aware method
        cos_interp = np.interp(x_new, x_old, np.cos(np.radians(ca)))
        sin_interp = np.interp(x_new, x_old, np.sin(np.radians(ca)))
        gamma_interp = np.degrees(np.arctan2(sin_interp, cos_interp))
        gamma_interp[gamma_interp < 0] += 360.0
        normalized.append(gamma_interp)

    if len(normalized) < 3:
        return np.zeros(5)

    normalized = np.array(normalized)  # (n_trials, 100)

    # CAV at each normalized time point
    cav_per_time = np.zeros(norm_length)
    for t in range(norm_length):
        angles_rad = np.radians(normalized[:, t])
        x_bar = np.mean(np.cos(angles_rad))
        y_bar = np.mean(np.sin(angles_rad))
        r_bar = np.sqrt(x_bar**2 + y_bar**2)
        r_bar = min(r_bar, 1.0)
        cav_per_time[t] = np.degrees(np.sqrt(2 * (1 - r_bar)))

    feats = [
        np.mean(cav_per_time),        # overall CAV
        np.mean(cav_per_time[:33]),    # CAV in first third (preparatory)
        np.mean(cav_per_time[33:67]),  # CAV in middle third (loading)
        np.mean(cav_per_time[67:]),    # CAV in last third (release)
        np.std(cav_per_time),          # CAV variability across phases
    ]
    return np.array(feats, dtype=np.float32)


def extract_coupling_features(ts_3d, kp_index, release_frame):
    """Per-shot coupling angle features (not CAV, which is cross-shot).

    Captures the coupling pattern WITHIN a single shot:
    - Mean coupling angle at release phase
    - Coupling angle shift from early to late (proximal->distal transition)
    - Dominant coordination pattern category
    """
    feats = []
    rf = int(np.clip(release_frame, 20, 234))

    # Compute shoulder and elbow angle time series
    shoulder_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_shoulder', 'neck', 'right_elbow')
    elbow_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_elbow', 'right_shoulder', 'right_wrist')

    # Shooting phase: approximate as pp_frame to release
    pp_frame = max(10, rf - 40)
    shot_shoulder = shoulder_angles[pp_frame:rf + 1]
    shot_elbow = elbow_angles[pp_frame:rf + 1]

    if len(shot_shoulder) < 5:
        return np.zeros(10, dtype=np.float32)

    # Coupling angles for this shot
    gamma = compute_coupling_angle(shot_shoulder, shot_elbow)

    if len(gamma) < 4:
        return np.zeros(10, dtype=np.float32)

    # 1. Mean coupling angle at release (last 20%)
    n = len(gamma)
    release_phase = gamma[int(0.8 * n):]
    cos_mean = np.mean(np.cos(np.radians(release_phase)))
    sin_mean = np.mean(np.sin(np.radians(release_phase)))
    gamma_release = np.degrees(np.arctan2(sin_mean, cos_mean))
    if gamma_release < 0:
        gamma_release += 360
    feats.append(gamma_release)

    # 2. Mean coupling angle in early phase (first 30%)
    early_phase = gamma[:int(0.3 * n)]
    if len(early_phase) > 0:
        cos_e = np.mean(np.cos(np.radians(early_phase)))
        sin_e = np.mean(np.sin(np.radians(early_phase)))
        gamma_early = np.degrees(np.arctan2(sin_e, cos_e))
        if gamma_early < 0:
            gamma_early += 360
        feats.append(gamma_early)
    else:
        feats.append(0.0)

    # 3. Coupling angle shift (early->release transition)
    feats.append(gamma_release - feats[-1])  # shift magnitude

    # 4. Circular standard deviation of coupling angles
    cos_all = np.mean(np.cos(np.radians(gamma)))
    sin_all = np.mean(np.sin(np.radians(gamma)))
    r_bar = np.sqrt(cos_all**2 + sin_all**2)
    circ_std = np.degrees(np.sqrt(2 * (1 - min(r_bar, 1.0))))
    feats.append(circ_std)

    # 5. Dominant coordination pattern (8 categories -> one-hot not practical,
    #    use the mean coupling angle as a continuous feature)
    gamma_mean = np.degrees(np.arctan2(sin_all, cos_all))
    if gamma_mean < 0:
        gamma_mean += 360
    feats.append(gamma_mean)

    # 6. Shoulder-elbow velocity ratio at release
    shoulder_vel = np.gradient(shoulder_angles, DT)
    elbow_vel = np.gradient(elbow_angles, DT)
    feats.append(np.abs(shoulder_vel[rf]) / max(np.abs(elbow_vel[rf]), 1e-6))

    # 7. Elbow-wrist coupling (secondary kinetic chain pair)
    wrist_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_wrist', 'right_elbow', 'right_second_finger_mcp')
    shot_wrist = wrist_angles[pp_frame:rf + 1]
    if len(shot_wrist) >= 5:
        gamma_ew = compute_coupling_angle(shot_elbow[:-1] if len(shot_elbow) > len(shot_wrist) else shot_elbow,
                                           shot_wrist[:len(shot_elbow) - 1] if len(shot_elbow) > len(shot_wrist) else shot_wrist)
        if len(gamma_ew) > 2:
            cos_ew = np.mean(np.cos(np.radians(gamma_ew)))
            sin_ew = np.mean(np.sin(np.radians(gamma_ew)))
            feats.append(np.degrees(np.arctan2(sin_ew, cos_ew)) % 360)
            r_ew = np.sqrt(cos_ew**2 + sin_ew**2)
            feats.append(np.degrees(np.sqrt(2 * (1 - min(r_ew, 1.0)))))
        else:
            feats.extend([0.0, 0.0])
    else:
        feats.extend([0.0, 0.0])

    # 8. Hip-knee coupling
    hip_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_hip', 'mid_hip', 'right_knee')
    knee_angles = compute_joint_angle_timeseries(
        ts_3d, kp_index, 'right_knee', 'right_hip', 'right_ankle')
    shot_hip = hip_angles[pp_frame:rf + 1]
    shot_knee = knee_angles[pp_frame:rf + 1]
    min_len = min(len(shot_hip), len(shot_knee))
    if min_len >= 5:
        gamma_hk = compute_coupling_angle(shot_hip[:min_len], shot_knee[:min_len])
        if len(gamma_hk) > 2:
            cos_hk = np.mean(np.cos(np.radians(gamma_hk)))
            sin_hk = np.mean(np.sin(np.radians(gamma_hk)))
            feats.append(np.degrees(np.arctan2(sin_hk, cos_hk)) % 360)
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------

def extract_all_biomech_features(ts_3d, kp_index, release_frame):
    """Extract all novel biomechanical features from the three papers."""
    cab = extract_cabarkapa_features(ts_3d, kp_index, release_frame)
    chen = extract_chen_kinetics_features(ts_3d, kp_index, release_frame)
    coupling = extract_coupling_features(ts_3d, kp_index, release_frame)
    return np.concatenate([cab, chen, coupling])


# ---------------------------------------------------------------------------
# Data loading (same as extended_physics_features.py)
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
    pls = PLSRegression(n_components=best_nc)
    pls.fit(raw_tr, y_target)
    pls_train = np.zeros((n_p, n_max), dtype=np.float32)
    pls_test = np.zeros((len(X_raw_test), n_max), dtype=np.float32)
    pls_train[:, :best_nc] = pls.transform(raw_tr)
    if len(raw_te) > 0:
        pls_test[:, :best_nc] = pls.transform(raw_te)
    return pls_train, pls_test


def physics_pls_augment(physics_feats_train, physics_feats_test, y_target, n_components=3):
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_target(train_data, test_data, y_train, scaler, target, target_idx,
               cav_features_train=None, cav_features_test=None):
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

    for pid in unique_pids:
        override = PLAYER_OVERRIDES.get((pid, target), {})
        pid_bw = override.get('bw', bw)

        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]
        n_tr = len(tr_indices)
        n_te = len(te_indices)

        center_frame = PLAYER_FRAMES[target][pid]

        # Standard PLS
        pls_train, pls_test = pls_augment(
            train_data['X_raw'][tr_mask],
            test_data['X_raw'][te_mask] if n_te > 0 else np.zeros((0, train_data['X_raw'].shape[1])),
            y_raw[tr_mask], n_max=15)

        # Base features
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

        # Biomechanics features (Papers 1+2+3)
        bio_tr, bio_te = [], []
        for i in tr_indices:
            ts_3d = train_data['X_3d'][i]
            rf = detect_release_frame(ts_3d, kp_index)
            bio_tr.append(extract_all_biomech_features(ts_3d, kp_index, rf))
        for i in te_indices:
            ts_3d = test_data['X_3d'][i]
            rf = detect_release_frame(ts_3d, kp_index)
            bio_te.append(extract_all_biomech_features(ts_3d, kp_index, rf))

        bio_tr = np.array(bio_tr)
        bio_te = np.array(bio_te) if n_te > 0 else np.zeros((0, bio_tr.shape[1]))

        # Clean NaN/inf
        for arr in [bio_tr, bio_te]:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # PLS compress biomechanics features
        y_p = y_scaled[tr_mask]
        bio_pls_tr, bio_pls_te = physics_pls_augment(bio_tr, bio_te, y_p, n_components=5)

        # Add CAV features (player-level, same for all shots of same player)
        if cav_features_train is not None:
            cav_tr = cav_features_train[tr_mask]
            cav_te = cav_features_test[te_mask] if n_te > 0 else np.zeros((0, cav_tr.shape[1]))
        else:
            cav_tr = np.zeros((n_tr, 0))
            cav_te = np.zeros((n_te, 0))

        # Assemble
        X_tr_aug = np.hstack([X_base_tr, pls_train, bio_pls_tr, cav_tr])
        X_te_aug = np.hstack([X_base_te, pls_test, bio_pls_te, cav_te]) if n_te > 0 \
            else np.zeros((0, X_tr_aug.shape[1]))

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_te_s = sc.transform(X_te_aug) if n_te > 0 else np.zeros((0, X_tr_aug.shape[1]))

        # Distance and kernel
        D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        all_dists = D_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(all_dists, pid_bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # LOO predictions
        for j, idx_j in enumerate(tr_indices):
            w = np.exp(-0.5 * (D_tr[j] / sigma) ** 2)
            w[j] = 0
            w /= w.sum() + 1e-12
            model = Ridge(alpha=10.0, fit_intercept=True)
            model.fit(X_tr_s, y_scaled[tr_mask], sample_weight=w)
            oof_preds[idx_j] = model.predict(X_tr_s[j:j+1])[0]

        # Test predictions
        if n_te > 0:
            D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
            for j, idx_j in enumerate(te_indices):
                w = np.exp(-0.5 * (D_te[j] / sigma) ** 2)
                w /= w.sum() + 1e-12
                model = Ridge(alpha=10.0, fit_intercept=True)
                model.fit(X_tr_s, y_scaled[tr_mask], sample_weight=w)
                test_preds[idx_j] = model.predict(X_te_s[j:j+1])[0]

    # Inverse transform
    oof_raw = scaler.inverse_transform(oof_preds.reshape(-1, 1)).ravel()
    test_raw = scaler.inverse_transform(test_preds.reshape(-1, 1)).ravel()

    # Re-scale to [0,1]
    oof_scaled = scaler.transform(oof_raw.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_raw.reshape(-1, 1)).ravel()

    # Compute LOO MSE
    y_true_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()
    mse = np.mean((oof_scaled - y_true_scaled) ** 2)

    return oof_scaled, test_scaled, mse


def compute_player_cav(train_data, test_data):
    """Compute per-player CAV features (cross-shot variability).

    This is a player-level feature: all shots of the same player get the same CAV.
    We compute shoulder-elbow CAV using ALL training shots per player.
    """
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    unique_pids = sorted(np.unique(pids_train))

    cav_dim = 5  # from compute_cav
    cav_train = np.zeros((len(pids_train), cav_dim), dtype=np.float32)
    cav_test = np.zeros((len(pids_test), cav_dim), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        tr_indices = np.where(tr_mask)[0]

        # Collect coupling angles from all shots of this player
        coupling_angles_list = []
        for i in tr_indices:
            ts_3d = train_data['X_3d'][i]
            rf = detect_release_frame(ts_3d, kp_index)
            pp_frame = max(10, rf - 40)

            shoulder_angles = compute_joint_angle_timeseries(
                ts_3d, kp_index, 'right_shoulder', 'neck', 'right_elbow')
            elbow_angles = compute_joint_angle_timeseries(
                ts_3d, kp_index, 'right_elbow', 'right_shoulder', 'right_wrist')

            shot_s = shoulder_angles[pp_frame:rf + 1]
            shot_e = elbow_angles[pp_frame:rf + 1]
            if len(shot_s) >= 5:
                gamma = compute_coupling_angle(shot_s, shot_e)
                if len(gamma) >= 3:
                    coupling_angles_list.append(gamma)

        # Compute CAV across all shots
        cav_values = compute_cav(coupling_angles_list)

        # Assign same CAV to all shots of this player
        cav_train[tr_mask] = cav_values

        te_mask = pids_test == pid
        if np.any(te_mask):
            cav_test[te_mask] = cav_values

    return cav_train, cav_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Quick test with one target")
    parser.add_argument("--base-subs", type=int, nargs="+", default=[2716, 3294, 3336],
                        help="Base submissions to blend with")
    parser.add_argument("--blend-weights", type=float, nargs="+",
                        default=[0.03, 0.05, 0.07, 0.10],
                        help="Blend weights to try")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("BIOMECHANICS PAPER FEATURES PIPELINE")
    print("=" * 70)
    print(f"Papers: Cabarkapa (COM, knee vel), Li (CAV), Chen (AI proxy, VV/HV)")
    print(f"Pilot: {args.pilot}")

    train_data, test_data = load_data()
    y_train = train_data['y']
    print(f"  Train: {len(y_train)}, Test: {len(test_data['ids'])}")

    # Compute player-level CAV features
    print("\nComputing player-level CAV features...")
    cav_train, cav_test = compute_player_cav(train_data, test_data)
    print(f"  CAV shape: {cav_train.shape}")
    for pid in sorted(np.unique(train_data['pids'])):
        mask = train_data['pids'] == pid
        print(f"  P{pid} CAV: {cav_train[mask][0]}")

    # Run per-target
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    targets_to_run = ["angle"] if args.pilot else TARGETS

    results = {}
    oof_all = {}
    test_all = {}

    for target in targets_to_run:
        print(f"\n--- {target.upper()} ---")
        scaler = TARGET_SCALERS[target]

        oof, test_pred, mse = run_target(
            train_data, test_data, y_train, scaler, target, target_idx,
            cav_features_train=cav_train, cav_features_test=cav_test)

        results[target] = mse
        oof_all[target] = oof
        test_all[target] = test_pred
        print(f"  LOO MSE: {mse:.6f}")

    if not args.pilot:
        mean_mse = np.mean(list(results.values()))
        print(f"\n  MEAN LOO MSE: {mean_mse:.6f}")

        # Compute diversity with base submissions
        print("\n--- DIVERSITY ANALYSIS ---")
        for base_num in args.base_subs:
            try:
                base_df = pd.read_csv(SUBMISSION_DIR / f"submission_{base_num}.csv")
                print(f"\n  vs Sub {base_num}:")
                for t in TARGETS:
                    col = f"scaled_{t}"
                    r = np.corrcoef(test_all[t], base_df[col].values)[0, 1]
                    print(f"    {t}: r={r:.4f}")
            except FileNotFoundError:
                print(f"  Sub {base_num} not found, skipping")

        # Generate submissions
        print("\n--- GENERATING SUBMISSIONS ---")
        submissions = []

        # Standalone
        sub_num = get_next_submission_number()
        result_df = pd.DataFrame({"id": test_data['ids']})
        for t in TARGETS:
            result_df[f"scaled_{t}"] = np.clip(test_all[t], 0.0, 1.0)
        result_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: Standalone biomech features (CV={mean_mse:.6f})")
        submissions.append({"num": sub_num, "desc": f"Standalone biomech (CV={mean_mse:.6f})"})

        # Blends with base submissions
        for base_num in args.base_subs:
            try:
                base_df = pd.read_csv(SUBMISSION_DIR / f"submission_{base_num}.csv")
            except FileNotFoundError:
                continue

            for w in args.blend_weights:
                sub_num = get_next_submission_number()
                blend_df = base_df[["id"]].copy()
                for t in TARGETS:
                    col = f"scaled_{t}"
                    blend_df[col] = np.clip(
                        w * test_all[t] + (1 - w) * base_df[col].values,
                        0.0, 1.0)
                blend_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
                desc = f"{int(w*100)}% biomech + {int((1-w)*100)}% Sub{base_num}"
                print(f"  Sub {sub_num}: {desc}")
                submissions.append({"num": sub_num, "desc": desc})

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Save metadata
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": ts,
        "pilot": args.pilot,
        "results": {k: float(v) for k, v in results.items()},
        "mean_cv": float(np.mean(list(results.values()))) if not args.pilot else None,
        "submissions": submissions if not args.pilot else [],
        "total_time_s": elapsed,
        "features": {
            "cabarkapa": "COM velocity, knee/elbow/hip/ankle angular velocity, release height, stance",
            "chen": "angular impulse proxies, VV/HV ratio, peak power proxies, RTD proxies",
            "li": "coupling angles, CAV (player-level), coordination shift",
        },
    }
    meta_path = OUTPUT_DIR / f"biomech_paper_features_{ts}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {meta_path}")


if __name__ == "__main__":
    main()
