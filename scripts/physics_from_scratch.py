"""
Physics Engine From Scratch - Layer-by-Layer Rebuild (v2).

Fixes from v1:
  - Compute ball velocity via Savgol derivative of ball POSITION trajectory
    (NOT Jacobian angular velocity decomposition - that amplifies noise)
  - Release frame = pre-release peak (max ball speed, ~10 frames before wrist peak)
  - Bone-length smoothing before any velocity computation
  - Arm geometry angles computed robustly (no division by noisy quantities)
  - Angular velocities computed from ANGLE timeseries, not position ratios

Validated findings from diagnostics:
  - Ball velocity from position Savgol: r=0.80 with true speed (pre-release peak)
  - Jacobian omega x r approach: r=-0.24 (noise amplification - REJECTED)
  - Finger data: 50-100% noise CV (EXCLUDED from all computations)
  - 91% of ball velocity comes from rotation, but measuring it directly
    from ball position changes is more robust than computing omega
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "physics_engine"))

from physics_engine.core.data_loader import (
    DataLoader, Shot, KEYPOINT_INDEX, HOOP_POSITION,
    FEET_TO_METERS, FRAMES_PER_SHOT, FRAME_RATE,
)

# Constants
G = 9.81
HOOP_FT = np.array([5.25, -25.0, 10.0])
HOOP_M = HOOP_FT * FEET_TO_METERS
DT = 1.0 / FRAME_RATE

# Fingertip keypoints for ball position estimation (average reduces noise)
RIGHT_FINGERTIP_KEYS = [
    'right_second_finger_distal',
    'right_third_finger_distal',
    'right_fourth_finger_distal',
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_trajectory(shot, joint_name, start=0, end=240):
    """Get trajectory with NaN/inf interpolation."""
    traj = shot.get_trajectory(joint_name, start, end).copy()
    for axis in range(3):
        vals = traj[:, axis]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            traj[:, axis] = 0.0
        elif np.any(bad):
            good = ~bad
            traj[bad, axis] = np.interp(
                np.where(bad)[0], np.where(good)[0], vals[good]
            )
    return traj


def savgol_smooth(traj, window=11, poly=3):
    """Apply Savgol filter to trajectory (N, 3)."""
    result = np.zeros_like(traj)
    n = traj.shape[0]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return traj.copy()
    for axis in range(3):
        result[:, axis] = savgol_filter(traj[:, axis], w, poly)
    return result


def savgol_velocity(traj_m, window=7, poly=2):
    """Compute velocity via Savgol derivative. Input in meters, output in m/s."""
    n = traj_m.shape[0]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        vel = np.zeros_like(traj_m)
        if n >= 2:
            vel[1:] = (traj_m[1:] - traj_m[:-1]) / DT
            vel[0] = vel[1]
        return vel
    result = np.zeros_like(traj_m)
    for axis in range(3):
        result[:, axis] = savgol_filter(traj_m[:, axis], w, poly, deriv=1, delta=DT)
    return result


def compute_angle_between(v1, v2):
    """Angle between two vectors in degrees."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_a))


def compute_true_velocity(release_pos_ft, angle_deg, depth_in, lr_in):
    """Compute exact ball release velocity from targets."""
    x_land = HOOP_FT[0] + lr_in / 12.0
    y_land = HOOP_FT[1] + depth_in / 12.0
    dx = (x_land - release_pos_ft[0]) * FEET_TO_METERS
    dy = (y_land - release_pos_ft[1]) * FEET_TO_METERS
    dz = (HOOP_FT[2] - release_pos_ft[2]) * FEET_TO_METERS
    D = np.sqrt(dx**2 + dy**2)
    angle_rad = np.radians(angle_deg)
    t2 = 2.0 * (dz + D * np.tan(angle_rad)) / G
    if t2 <= 0:
        return None
    t = np.sqrt(t2)
    return {
        'vx': dx / t, 'vy': dy / t,
        'vz': (dz + 0.5 * G * t**2) / t,
        'speed': np.sqrt((dx/t)**2 + (dy/t)**2 + ((dz + 0.5*G*t**2)/t)**2),
    }


# ============================================================================
# LAYER 1: Bone-Length Smoothing
# ============================================================================

def smooth_arm_chain(shot, start=60, end=210):
    """
    Smooth arm chain: Savgol + bone length constraints.
    Returns smoothed trajectories in FEET.
    """
    n = end - start

    # Raw trajectories
    shoulder_raw = safe_trajectory(shot, 'right_shoulder', start, end)
    elbow_raw = safe_trajectory(shot, 'right_elbow', start, end)
    wrist_raw = safe_trajectory(shot, 'right_wrist', start, end)

    # Savgol smooth (window=11 = 183ms, good balance of noise reduction vs signal)
    shoulder = savgol_smooth(shoulder_raw, window=11, poly=3)
    elbow = savgol_smooth(elbow_raw, window=11, poly=3)
    wrist = savgol_smooth(wrist_raw, window=11, poly=3)

    # Enforce constant bone lengths
    upper_arm_lens = np.linalg.norm(elbow - shoulder, axis=1)
    forearm_lens = np.linalg.norm(wrist - elbow, axis=1)
    target_ua = np.median(upper_arm_lens)
    target_fa = np.median(forearm_lens)

    for f in range(n):
        # Project elbow onto sphere of radius target_ua centered at shoulder
        se = elbow[f] - shoulder[f]
        se_len = np.linalg.norm(se)
        if se_len > 1e-6:
            elbow[f] = shoulder[f] + (se / se_len) * target_ua

        # Project wrist onto sphere of radius target_fa centered at elbow
        ew = wrist[f] - elbow[f]
        ew_len = np.linalg.norm(ew)
        if ew_len > 1e-6:
            wrist[f] = elbow[f] + (ew / ew_len) * target_fa

    return {
        'shoulder': shoulder, 'elbow': elbow, 'wrist': wrist,
        'start': start, 'end': end,
        'ua_len': target_ua, 'fa_len': target_fa,
    }


# ============================================================================
# LAYER 2: Ball Position + Release Frame Detection
# ============================================================================

def estimate_ball_trajectory(shot, arm):
    """
    Estimate ball center trajectory using fingertip center + wrist.

    Uses: ball = wrist + 0.6 * (fingertip_center - wrist)
    The fingertip_center is the AVERAGE of 3 fingertip distal positions,
    which reduces noise by sqrt(3) compared to individual fingers.
    Then smooth aggressively (window=15) to further reduce noise.

    Returns trajectory in feet, shape (N, 3).
    """
    start = arm['start']
    end = arm['end']
    wrist = arm['wrist']
    n = wrist.shape[0]

    # Get fingertip trajectories and average them
    fingertip_trajs = []
    for key in RIGHT_FINGERTIP_KEYS:
        traj = safe_trajectory(shot, key, start, end)
        fingertip_trajs.append(traj)

    # Average of 3 fingertips (noise reduces by factor sqrt(3))
    fingertip_center = np.mean(fingertip_trajs, axis=0)

    # Smooth fingertip center aggressively (window=15 = 250ms)
    fingertip_smooth = savgol_smooth(fingertip_center, window=15, poly=3)

    # Ball position: wrist + 0.6 * (fingertip_center - wrist)
    # This places the ball center between wrist and fingertips
    ball = wrist + 0.6 * (fingertip_smooth - wrist)

    # Final smooth of ball trajectory
    ball = savgol_smooth(ball, window=11, poly=3)

    return ball


def find_release_frame(arm, ball_traj_ft):
    """
    Find release frame using two methods:
    1. Wrist peak height (anchor point)
    2. Peak ball speed BEFORE wrist peak (the actual release, ~10 frames earlier)

    Diagnostic showed: at wrist peak, hand has decelerated to 6% of true speed.
    The pre-release peak (max ball speed before wrist peak) gives r=0.80 with true speed.
    """
    # Convert ball trajectory to meters and compute velocity
    ball_m = ball_traj_ft * FEET_TO_METERS
    ball_vel = savgol_velocity(ball_m, window=9, poly=3)
    ball_speed = np.linalg.norm(ball_vel, axis=1)

    n = ball_traj_ft.shape[0]
    offset = arm['start']

    # Find wrist peak height (anchor)
    wrist_z = arm['wrist'][:, 2]
    search_start = 15
    search_end = n - 5
    wrist_peak_local = search_start + np.argmax(wrist_z[search_start:search_end])
    wrist_peak_global = offset + wrist_peak_local

    # Find peak ball speed BEFORE wrist peak (the actual release)
    # Search from frame 15 to wrist_peak + a small buffer
    release_search_end = min(wrist_peak_local + 5, search_end)
    release_search_start = max(search_start, wrist_peak_local - 40)

    search_speeds = ball_speed[release_search_start:release_search_end]
    if len(search_speeds) > 0:
        peak_local = release_search_start + np.argmax(search_speeds)
    else:
        # Fallback: wrist peak - 10
        peak_local = max(search_start, wrist_peak_local - 10)

    release_global = offset + peak_local

    return {
        'release_frame': release_global,
        'release_local': peak_local,
        'wrist_peak_frame': wrist_peak_global,
        'wrist_peak_local': wrist_peak_local,
        'ball_vel': ball_vel,
        'ball_speed': ball_speed,
    }


# ============================================================================
# LAYER 3: Feature Extraction at Release Frame
# ============================================================================

def extract_arm_angles_timeseries(arm):
    """
    Compute arm joint angles over time.
    Returns angle timeseries (smoother than angular velocity).
    """
    shoulder = arm['shoulder']
    elbow = arm['elbow']
    wrist = arm['wrist']
    n = shoulder.shape[0]

    elbow_angles = np.zeros(n)
    forearm_elevations = np.zeros(n)
    upper_arm_elevations = np.zeros(n)

    for f in range(n):
        # Elbow angle: angle between upper arm and forearm
        ua = elbow[f] - shoulder[f]
        fa = wrist[f] - elbow[f]
        elbow_angles[f] = compute_angle_between(-ua, fa)

        # Forearm elevation (angle above horizontal)
        fa_len = np.linalg.norm(fa)
        if fa_len > 1e-6:
            forearm_elevations[f] = np.degrees(np.arcsin(np.clip(fa[2] / fa_len, -1, 1)))

        # Upper arm elevation
        ua_len = np.linalg.norm(ua)
        if ua_len > 1e-6:
            upper_arm_elevations[f] = np.degrees(np.arcsin(np.clip(ua[2] / ua_len, -1, 1)))

    # Smooth the angle timeseries
    w = 11
    elbow_angles_smooth = savgol_filter(elbow_angles, w, 3)
    forearm_elev_smooth = savgol_filter(forearm_elevations, w, 3)
    upper_arm_elev_smooth = savgol_filter(upper_arm_elevations, w, 3)

    # Angular velocities from angle derivatives (much more robust than omega from positions)
    elbow_angular_vel = savgol_filter(elbow_angles, w, 3, deriv=1, delta=DT)
    forearm_angular_vel = savgol_filter(forearm_elevations, w, 3, deriv=1, delta=DT)
    upper_arm_angular_vel = savgol_filter(upper_arm_elevations, w, 3, deriv=1, delta=DT)

    return {
        'elbow_angle': elbow_angles_smooth,
        'forearm_elevation': forearm_elev_smooth,
        'upper_arm_elevation': upper_arm_elev_smooth,
        'elbow_angular_vel': elbow_angular_vel,
        'forearm_angular_vel': forearm_angular_vel,
        'upper_arm_angular_vel': upper_arm_angular_vel,
    }


def extract_features(shot, arm, ball_traj_ft, release_info, angle_ts):
    """
    Extract physics features at the release frame.
    All features derived from PHYSICS of the shooting motion.
    """
    f = release_info['release_local']
    n = arm['shoulder'].shape[0]
    feats = {}

    # --- Release timing ---
    feats['release_frame'] = release_info['release_frame']
    feats['wrist_peak_frame'] = release_info['wrist_peak_frame']
    feats['release_to_wrist_peak'] = release_info['wrist_peak_frame'] - release_info['release_frame']

    # --- Ball velocity at release (from Savgol derivative - the ROBUST method) ---
    ball_vel = release_info['ball_vel']  # in m/s
    ball_speed = release_info['ball_speed']
    v = ball_vel[f]
    feats['ball_vx'] = v[0]
    feats['ball_vy'] = v[1]
    feats['ball_vz'] = v[2]
    feats['ball_speed'] = ball_speed[f]

    # --- Peak ball speed ---
    feats['peak_ball_speed'] = np.max(ball_speed)
    if feats['peak_ball_speed'] > 1e-6:
        feats['release_vs_peak_speed'] = ball_speed[f] / feats['peak_ball_speed']
    else:
        feats['release_vs_peak_speed'] = 0.0

    # --- Ball velocity direction ---
    # Project onto hoop direction (horizontal)
    ball_pos_m = ball_traj_ft[f] * FEET_TO_METERS
    to_hoop = HOOP_M - ball_pos_m
    to_hoop_horiz = to_hoop.copy()
    to_hoop_horiz[2] = 0
    th_norm = np.linalg.norm(to_hoop_horiz)
    if th_norm > 1e-6:
        to_hoop_unit = to_hoop_horiz / th_norm
    else:
        to_hoop_unit = np.array([1, 0, 0])

    # Lateral direction (perpendicular to hoop in horizontal plane)
    lateral_unit = np.array([-to_hoop_unit[1], to_hoop_unit[0], 0])

    feats['v_toward_hoop'] = np.dot(v, to_hoop_unit)
    feats['v_lateral'] = np.dot(v, lateral_unit)
    feats['v_vertical'] = v[2]

    # Hoop alignment
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-6:
        feats['hoop_alignment'] = np.dot(v / v_norm, to_hoop / np.linalg.norm(to_hoop))
    else:
        feats['hoop_alignment'] = 0.0

    # --- Wrist velocity ---
    wrist_m = arm['wrist'] * FEET_TO_METERS
    wrist_vel = savgol_velocity(wrist_m, window=9, poly=3)
    wv = wrist_vel[f]
    feats['wrist_vx'] = wv[0]
    feats['wrist_vy'] = wv[1]
    feats['wrist_vz'] = wv[2]
    feats['wrist_speed'] = np.linalg.norm(wv)

    # Wrist velocity projected onto hoop/lateral
    feats['wrist_v_toward_hoop'] = np.dot(wv, to_hoop_unit)
    feats['wrist_v_lateral'] = np.dot(wv, lateral_unit)

    # --- Shoulder velocity (body motion) ---
    shoulder_m = arm['shoulder'] * FEET_TO_METERS
    shoulder_vel = savgol_velocity(shoulder_m, window=9, poly=3)
    sv = shoulder_vel[f]
    feats['shoulder_speed'] = np.linalg.norm(sv)
    feats['shoulder_v_toward_hoop'] = np.dot(sv, to_hoop_unit)

    # --- Velocity decomposition: how much from body vs arm ---
    if feats['wrist_speed'] > 1e-6:
        feats['body_fraction'] = feats['shoulder_speed'] / feats['wrist_speed']
    else:
        feats['body_fraction'] = 0.0

    # Ball velocity beyond wrist velocity (rotational contribution)
    v_beyond_wrist = v - wv
    feats['rotational_contribution_mag'] = np.linalg.norm(v_beyond_wrist)
    feats['rotational_toward_hoop'] = np.dot(v_beyond_wrist, to_hoop_unit)
    feats['rotational_lateral'] = np.dot(v_beyond_wrist, lateral_unit)

    # --- Arm geometry at release ---
    feats['elbow_angle'] = angle_ts['elbow_angle'][f]
    feats['forearm_elevation'] = angle_ts['forearm_elevation'][f]
    feats['upper_arm_elevation'] = angle_ts['upper_arm_elevation'][f]

    # --- Angular velocities (from ANGLE timeseries - robust method) ---
    feats['elbow_angular_vel'] = angle_ts['elbow_angular_vel'][f]
    feats['forearm_angular_vel'] = angle_ts['forearm_angular_vel'][f]
    feats['upper_arm_angular_vel'] = angle_ts['upper_arm_angular_vel'][f]

    # Peak angular velocities
    feats['elbow_angular_vel_peak'] = np.max(np.abs(angle_ts['elbow_angular_vel']))
    feats['forearm_angular_vel_peak'] = np.max(np.abs(angle_ts['forearm_angular_vel']))

    # --- Release position ---
    ball_ft = ball_traj_ft[f]
    feats['release_x_ft'] = ball_ft[0]
    feats['release_y_ft'] = ball_ft[1]
    feats['release_z_ft'] = ball_ft[2]
    feats['release_dist_to_hoop'] = np.linalg.norm(ball_ft - HOOP_FT)

    # Arm extension
    shoulder_ft = arm['shoulder'][f]
    feats['arm_extension'] = np.linalg.norm(ball_ft - shoulder_ft)
    total_arm = arm['ua_len'] + arm['fa_len']
    if total_arm > 1e-6:
        feats['arm_extension_ratio'] = feats['arm_extension'] / total_arm
    else:
        feats['arm_extension_ratio'] = 0.0

    # --- Temporal dynamics (ball speed evolution) ---
    # Speed in window around release
    w_start = max(0, f - 5)
    w_end = min(n, f + 6)
    speed_window = ball_speed[w_start:w_end]

    feats['ball_speed_window_mean'] = np.mean(speed_window)
    feats['ball_speed_window_std'] = np.std(speed_window)

    if len(speed_window) >= 3:
        t_axis = np.arange(len(speed_window)) * DT
        feats['ball_speed_slope'] = np.polyfit(t_axis, speed_window, 1)[0]
    else:
        feats['ball_speed_slope'] = 0.0

    # Elbow angle evolution around release
    ea_window = angle_ts['elbow_angle'][w_start:w_end]
    feats['elbow_angle_window_mean'] = np.mean(ea_window)
    if len(ea_window) >= 3:
        t_axis = np.arange(len(ea_window)) * DT
        feats['elbow_angle_slope'] = np.polyfit(t_axis, ea_window, 1)[0]
    else:
        feats['elbow_angle_slope'] = 0.0

    # --- Ball acceleration at release ---
    if f >= 1 and f < n - 1:
        ball_accel = (ball_vel[f+1] - ball_vel[f-1]) / (2 * DT)
        feats['ball_accel_mag'] = np.linalg.norm(ball_accel)
        feats['ball_accel_vertical'] = ball_accel[2]
    else:
        feats['ball_accel_mag'] = 0.0
        feats['ball_accel_vertical'] = 0.0

    # --- Wrist height at release ---
    feats['wrist_height_at_release'] = arm['wrist'][f, 2]

    # --- Launch angle estimate (vertical angle of ball velocity) ---
    horiz_speed = np.sqrt(v[0]**2 + v[1]**2)
    if horiz_speed > 1e-6:
        feats['launch_angle_est'] = np.degrees(np.arctan2(v[2], horiz_speed))
    else:
        feats['launch_angle_est'] = 90.0 if v[2] > 0 else -90.0

    return feats


# ============================================================================
# PIPELINE: Process All Shots
# ============================================================================

def process_all_shots(loader, train=True, verbose=True):
    """Process all shots through the physics pipeline."""
    df = loader.load_train() if train else loader.load_test()
    n_shots = len(df)

    all_features = []
    all_player_ids = []
    all_shot_ids = []
    targets = {'angle': [], 'depth': [], 'left_right': []}
    validation = []

    for i in range(n_shots):
        shot = loader.get_shot(i, train=train)

        # Layer 1: Smooth arm chain
        arm = smooth_arm_chain(shot, start=60, end=210)

        # Layer 2: Ball position + release frame
        ball_traj = estimate_ball_trajectory(shot, arm)
        rel = find_release_frame(arm, ball_traj)

        # Layer 3: Arm angle timeseries
        angle_ts = extract_arm_angles_timeseries(arm)

        # Layer 4: Features
        feats = extract_features(shot, arm, ball_traj, rel, angle_ts)
        all_features.append(feats)
        all_player_ids.append(shot.player_id)
        all_shot_ids.append(shot.shot_id)

        if train and shot.targets:
            targets['angle'].append(shot.targets['angle'])
            targets['depth'].append(shot.targets['depth'])
            targets['left_right'].append(shot.targets['left_right'])

            # Validate: compare ball velocity with true velocity
            ball_pos_ft = ball_traj[rel['release_local']]
            true_v = compute_true_velocity(
                ball_pos_ft, shot.targets['angle'],
                shot.targets['depth'], shot.targets['left_right']
            )
            if true_v is not None:
                v_est = rel['ball_vel'][rel['release_local']]
                validation.append({
                    'player_id': shot.player_id,
                    'est_speed': np.linalg.norm(v_est),
                    'true_speed': true_v['speed'],
                    'est_vx': v_est[0], 'true_vx': true_v['vx'],
                    'est_vy': v_est[1], 'true_vy': true_v['vy'],
                    'est_vz': v_est[2], 'true_vz': true_v['vz'],
                    'release_frame': rel['release_frame'],
                    'wrist_peak': rel['wrist_peak_frame'],
                })

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_shots} shots")

    feat_df = pd.DataFrame(all_features)
    feat_df['player_id'] = all_player_ids
    feat_df['shot_id'] = all_shot_ids

    if train:
        feat_df['angle'] = targets['angle']
        feat_df['depth'] = targets['depth']
        feat_df['left_right'] = targets['left_right']

    return feat_df, validation


def validate_pipeline(validation):
    """Print validation results for each layer."""
    if not validation:
        return

    vd = pd.DataFrame(validation)

    print("\n" + "="*70)
    print("LAYER VALIDATION")
    print("="*70)

    # Release frame
    print("\n--- Layer 2: Release Frame ---")
    print(f"  Release frame: mean={vd['release_frame'].mean():.1f}, "
          f"std={vd['release_frame'].std():.1f}")
    print(f"  Wrist peak:    mean={vd['wrist_peak'].mean():.1f}, "
          f"std={vd['wrist_peak'].std():.1f}")
    diff = vd['wrist_peak'] - vd['release_frame']
    print(f"  Difference (wrist_peak - release): mean={diff.mean():.1f}, "
          f"std={diff.std():.1f}")

    # Velocity accuracy
    print("\n--- Layer 3: Ball Velocity (Savgol derivative) ---")
    print(f"  Estimated speed: mean={vd['est_speed'].mean():.3f}, "
          f"std={vd['est_speed'].std():.3f} m/s")
    print(f"  True speed:      mean={vd['true_speed'].mean():.3f}, "
          f"std={vd['true_speed'].std():.3f} m/s")
    ratio = vd['est_speed'].mean() / vd['true_speed'].mean()
    print(f"  Speed ratio: {ratio:.1%} of true")

    print("\n  Component correlations:")
    for comp in ['vx', 'vy', 'vz', 'speed']:
        est_col = f'est_{comp}'
        true_col = f'true_{comp}'
        r = np.corrcoef(vd[est_col], vd[true_col])[0, 1]
        rmse = np.sqrt(np.mean((vd[est_col] - vd[true_col])**2))
        print(f"    {comp:6s}: r={r:+.4f}, RMSE={rmse:.4f} m/s")

    print("\n  Per-player:")
    for pid in sorted(vd['player_id'].unique()):
        p = vd[vd['player_id'] == pid]
        ratio = p['est_speed'].mean() / p['true_speed'].mean()
        r = np.corrcoef(p['est_speed'], p['true_speed'])[0, 1]
        print(f"    Player {pid}: {ratio:.1%} of true, r={r:+.3f} (n={len(p)})")


# ============================================================================
# CV EVALUATION
# ============================================================================

def evaluate_cv(feat_df):
    """
    Evaluate features using WITHIN-PLAYER KFold CV.

    GroupKFold with 5 players/5 folds holds out entire players, making
    per-player models impossible (no training data for held-out player).
    Instead, split each player's shots into train/val independently.
    """
    from sklearn.model_selection import KFold

    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
        print("  (LightGBM not available, using Ridge only)")

    try:
        import xgboost as xgb
    except ImportError:
        xgb = None

    exclude = {'player_id', 'shot_id', 'angle', 'depth', 'left_right'}
    feat_cols = [c for c in feat_df.columns if c not in exclude]

    X = feat_df[feat_cols].values
    players = feat_df['player_id'].values
    unique_players = sorted(feat_df['player_id'].unique())

    results = {}
    for target in ['angle', 'depth', 'left_right']:
        y_raw = feat_df[target].values

        scaler_path = PROJECT_ROOT / f"data/scaler_{target}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            y = scaler.transform(y_raw.reshape(-1, 1)).ravel()
        else:
            y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)

        oof = np.full(len(y), np.nan)

        # Within-player 5-fold CV
        for pid in unique_players:
            mask = players == pid
            X_p = X[mask]
            y_p = y[mask]
            indices = np.where(mask)[0]

            if len(X_p) < 10:
                oof[mask] = np.mean(y_p)
                continue

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for tr_idx, val_idx in kf.split(X_p):
                X_tr, X_val = X_p[tr_idx], X_p[val_idx]
                y_tr = y_p[tr_idx]

                # Ridge
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr, y_tr)
                pred_r = ridge.predict(X_val)

                pred_final = pred_r

                if lgb is not None and len(X_tr) >= 10:
                    lgb_m = lgb.LGBMRegressor(
                        n_estimators=100, num_leaves=8,
                        min_child_samples=max(5, len(X_tr) // 10),
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        verbose=-1, n_jobs=1,
                    )
                    lgb_m.fit(X_tr, y_tr)
                    pred_l = lgb_m.predict(X_val)
                    pred_final = 0.4 * pred_r + 0.6 * pred_l

                if xgb is not None and len(X_tr) >= 10:
                    xgb_m = xgb.XGBRegressor(
                        n_estimators=100, max_depth=3,
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        verbosity=0, n_jobs=1,
                    )
                    xgb_m.fit(X_tr, y_tr)
                    pred_x = xgb_m.predict(X_val)
                    # 3-way blend: Ridge 30%, LGB 35%, XGB 35%
                    pred_final = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x

                global_idx = indices[val_idx]
                oof[global_idx] = pred_final

        # Handle any remaining NaN
        nan_mask = np.isnan(oof)
        if np.any(nan_mask):
            oof[nan_mask] = np.mean(y)

        mse = mean_squared_error(y, oof)
        r = np.corrcoef(y, oof)[0, 1] if np.std(oof) > 1e-9 else 0.0
        results[target] = {'mse': mse, 'r': r, 'oof': oof}
        print(f"  {target:12s}: CV MSE = {mse:.6f}, r = {r:+.4f}")

    mean_mse = np.mean([results[t]['mse'] for t in results])
    print(f"\n  MEAN MSE = {mean_mse:.6f}")
    print(f"  (Current best LB: 0.007224)")

    return results


# ============================================================================
# BLEND WITH SUB 784
# ============================================================================

def generate_test_predictions(train_df, test_df):
    """Train on full training data, predict test."""
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

    exclude = {'player_id', 'shot_id', 'angle', 'depth', 'left_right'}
    feat_cols = [c for c in train_df.columns if c not in exclude]

    X_train = train_df[feat_cols].values
    X_test = test_df[feat_cols].values
    players_tr = train_df['player_id'].values
    players_te = test_df['player_id'].values

    test_preds = {}
    for target in ['angle', 'depth', 'left_right']:
        scaler_path = PROJECT_ROOT / f"data/scaler_{target}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            y_train = scaler.transform(train_df[target].values.reshape(-1, 1)).ravel()
        else:
            y_raw = train_df[target].values
            y_train = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)

        preds = np.zeros(len(X_test))
        for pid in sorted(train_df['player_id'].unique()):
            tr_m = players_tr == pid
            te_m = players_te == pid
            if not np.any(tr_m) or not np.any(te_m):
                continue

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_train[tr_m], y_train[tr_m])
            pred_r = ridge.predict(X_test[te_m])

            if lgb is not None and np.sum(tr_m) >= 10:
                model = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=8,
                    min_child_samples=max(5, int(np.sum(tr_m)) // 10),
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                    verbose=-1, n_jobs=1,
                )
                model.fit(X_train[tr_m], y_train[tr_m])
                pred_l = model.predict(X_test[te_m])
                preds[te_m] = 0.5 * pred_r + 0.5 * pred_l
            else:
                preds[te_m] = pred_r

        test_preds[target] = np.clip(preds, 0, 1)

    return test_preds


def blend_and_save(test_preds):
    """Blend with Sub 784 and save submissions."""
    sub_784 = pd.read_csv(PROJECT_ROOT / "submission/submission_784.csv")

    sub_dir = PROJECT_ROOT / "submission"
    existing = list(sub_dir.glob("submission_*.csv"))
    max_num = max([int(p.stem.split('_')[1]) for p in existing]) if existing else 0

    print("\n  Blend weights with Sub 784:")
    saved = []
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sub = sub_784.copy()
        for target in ['angle', 'depth', 'left_right']:
            col = f'scaled_{target}'
            sub[col] = (1 - w) * sub_784[col] + w * test_preds[target]

        a_std = sub['scaled_angle'].std()
        d_mean = sub['scaled_depth'].mean()
        ok = a_std < 0.14 and 0.50 <= d_mean <= 0.51

        print(f"    w={w:.2f}: angle_std={a_std:.4f}, depth_mean={d_mean:.4f}, "
              f"profile={'OK' if ok else 'FAIL'}")

        if ok:
            max_num += 1
            path = sub_dir / f"submission_{max_num}.csv"
            sub.to_csv(path, index=False)
            saved.append((w, path.name))
            print(f"      -> Saved as {path.name}")

    # Always save a small blend (even if profile fails)
    if not saved:
        w = 0.05
        sub = sub_784.copy()
        for target in ['angle', 'depth', 'left_right']:
            col = f'scaled_{target}'
            sub[col] = (1 - w) * sub_784[col] + w * test_preds[target]
        max_num += 1
        path = sub_dir / f"submission_{max_num}.csv"
        sub.to_csv(path, index=False)
        saved.append((w, path.name))
        print(f"\n  Saved 5% blend as {path.name} (profile may fail)")

    return saved


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("PHYSICS FROM SCRATCH - v2 (Savgol Ball Velocity)")
    print("="*70)

    loader = DataLoader()

    # Process training
    print("\nProcessing training shots...")
    train_df, validation = process_all_shots(loader, train=True)
    print(f"Extracted {len([c for c in train_df.columns if c not in {'player_id','shot_id','angle','depth','left_right'}])} features for {len(train_df)} shots")

    # Validate each layer
    validate_pipeline(validation)

    # Feature correlations
    exclude = {'player_id', 'shot_id', 'angle', 'depth', 'left_right'}
    feat_cols = [c for c in train_df.columns if c not in exclude]

    print("\n" + "="*70)
    print("TOP FEATURE CORRELATIONS")
    print("="*70)

    for target in ['angle', 'depth', 'left_right']:
        scaler_path = PROJECT_ROOT / f"data/scaler_{target}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            y = scaler.transform(train_df[target].values.reshape(-1, 1)).ravel()
        else:
            yr = train_df[target].values
            y = (yr - yr.min()) / (yr.max() - yr.min() + 1e-9)

        corrs = {}
        for col in feat_cols:
            vals = train_df[col].values
            if np.std(vals) > 1e-9 and not np.any(np.isnan(vals)):
                corrs[col] = np.corrcoef(vals, y)[0, 1]

        sorted_c = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  {target} (top 10):")
        for col, r in sorted_c[:10]:
            print(f"    {col:40s}  r={r:+.4f}")

    # CV evaluation
    print("\n" + "="*70)
    print("CROSS-VALIDATION (Per-Player Ridge + LGB)")
    print("="*70)
    results = evaluate_cv(train_df)

    # Process test
    print("\nProcessing test shots...")
    test_df, _ = process_all_shots(loader, train=False)

    # Save features
    out = PROJECT_ROOT / "output"
    out.mkdir(exist_ok=True)
    train_df.to_csv(out / "physics_scratch_v2_train.csv", index=False)
    test_df.to_csv(out / "physics_scratch_v2_test.csv", index=False)
    print(f"Saved features to {out}/physics_scratch_v2_*.csv")

    # Generate test predictions and blend
    print("\n" + "="*70)
    print("BLEND WITH SUB 784")
    print("="*70)
    test_preds = generate_test_predictions(train_df, test_df)
    saved = blend_and_save(test_preds)

    return train_df, test_df, results, saved


if __name__ == "__main__":
    main()
