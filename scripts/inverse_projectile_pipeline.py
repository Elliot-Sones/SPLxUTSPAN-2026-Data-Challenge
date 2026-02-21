"""
Inverse Projectile Pipeline

Key insight: For each training shot, we KNOW the targets. Combined with the
release position (from hand keypoints), we can compute the EXACT ball release
velocity via inverse projectile physics. Then we learn the hand-to-ball
transfer function and apply it to test shots.

Pipeline:
1. Inverse projectile: compute true (vx, vy, vz) for all 345 training shots
2. Extract hand geometry features at release (palm normal, shooting axis, etc.)
3. Train per-player models: hand_features -> velocity_components
4. Forward simulate test shots to get target predictions
5. Blend with Sub 784
"""

import sys
import json
import fcntl
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cross_decomposition import PLSRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Add physics engine to path
sys.path.insert(0, str(PROJECT_DIR))
from physics_engine.core.data_loader import (
    DataLoader, Shot, KEYPOINT_INDEX, HOOP_POSITION, FEET_TO_METERS
)

G = 9.81  # m/s^2
HOOP_X_FT = 5.25
HOOP_Y_FT = -25.0
HOOP_Z_FT = 10.0


# ============================================================
# STEP 1: INVERSE PROJECTILE SOLVER
# ============================================================

def compute_true_velocity(release_pos_ft, angle_deg, depth_in, lr_in):
    """
    Compute exact release velocity from known targets and release position.

    Args:
        release_pos_ft: [x, y, z] ball position at release in feet
        angle_deg: entry angle at hoop plane (degrees, positive = descending)
        depth_in: depth from front of hoop along axis (inches)
        lr_in: lateral displacement from hoop center (inches)

    Returns:
        dict with vx, vy, vz (m/s), speed (m/s), flight_time (s), is_valid (bool)
    """
    # Landing position at hoop plane (z=10ft)
    x_land_ft = HOOP_X_FT + lr_in / 12.0
    y_land_ft = HOOP_Y_FT + depth_in / 12.0
    z_land_ft = HOOP_Z_FT

    # Convert displacements to meters
    dx = (x_land_ft - release_pos_ft[0]) * FEET_TO_METERS
    dy = (y_land_ft - release_pos_ft[1]) * FEET_TO_METERS
    dz = (z_land_ft - release_pos_ft[2]) * FEET_TO_METERS

    D_horiz = np.sqrt(dx**2 + dy**2)
    angle_rad = np.radians(angle_deg)

    # Solve for flight time:
    # dz = -D_horiz * tan(angle) + 0.5 * g * t^2
    # t^2 = 2 * (dz + D_horiz * tan(angle)) / g
    t_squared = 2.0 * (dz + D_horiz * np.tan(angle_rad)) / G

    if t_squared <= 0:
        return {'vx': 0, 'vy': 0, 'vz': 0, 'speed': 0, 'flight_time': 0,
                'is_valid': False, 'launch_angle_deg': 0}

    t = np.sqrt(t_squared)

    vx = dx / t
    vy = dy / t
    vz = (dz + 0.5 * G * t**2) / t

    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    launch_angle = np.degrees(np.arctan2(vz, np.sqrt(vx**2 + vy**2)))

    return {
        'vx': vx, 'vy': vy, 'vz': vz,
        'speed': speed, 'flight_time': t,
        'launch_angle_deg': launch_angle,
        'is_valid': True,
    }


def forward_simulate(release_pos_ft, vel_ms):
    """
    Forward simulate projectile from release to hoop plane (z=10ft).

    Returns:
        dict with predicted angle, depth, left_right (raw units), or None if miss
    """
    pos_m = np.array(release_pos_ft) * FEET_TO_METERS
    target_z_m = HOOP_Z_FT * FEET_TO_METERS

    dz_m = target_z_m - pos_m[2]
    vz = vel_ms[2]

    # Solve quadratic: 0.5*g*t^2 - vz*t + dz = 0
    a = 0.5 * G
    b = -vz
    c = dz_m

    disc = b**2 - 4*a*c
    if disc < 0:
        return None

    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)

    times = [t for t in [t1, t2] if t > 0]
    if not times:
        return None
    t = max(times)  # descending arc

    if t > 5.0:
        return None

    # Landing position (meters)
    landing_x_m = pos_m[0] + vel_ms[0] * t
    landing_y_m = pos_m[1] + vel_ms[1] * t

    # Landing velocity
    landing_vz = vz - G * t
    horiz_speed = np.sqrt(vel_ms[0]**2 + vel_ms[1]**2)

    # Convert to feet
    landing_x_ft = landing_x_m / FEET_TO_METERS
    landing_y_ft = landing_y_m / FEET_TO_METERS

    # Compute targets
    entry_angle = np.degrees(np.arctan2(-landing_vz, horiz_speed))
    depth_in = (landing_y_ft - HOOP_Y_FT) * 12.0
    lr_in = (landing_x_ft - HOOP_X_FT) * 12.0

    return {
        'angle': entry_angle,
        'depth': depth_in,
        'left_right': lr_in,
        'flight_time': t,
    }


# ============================================================
# STEP 2: RELEASE POSITION ESTIMATION
# ============================================================

def estimate_release_frame(shot):
    """Find release frame as wrist peak height."""
    wrist_z = []
    for f in range(shot.num_frames):
        w = shot.get_position('right_wrist', f)
        wrist_z.append(w[2] if w is not None else 0)
    return int(np.argmax(wrist_z))


def estimate_release_position(shot, release_frame):
    """Estimate ball position at release (in feet)."""
    wrist = shot.get_position('right_wrist', release_frame)
    fingertips = []
    for name in ['right_second_finger_distal', 'right_third_finger_distal',
                  'right_fourth_finger_distal']:
        pos = shot.get_position(name, release_frame)
        if pos is not None:
            fingertips.append(pos)

    if wrist is None or len(fingertips) == 0:
        return None

    ft_center = np.mean(fingertips, axis=0)
    ball_pos = wrist + 0.6 * (ft_center - wrist)
    return ball_pos  # in feet


# ============================================================
# STEP 3: HAND GEOMETRY FEATURES
# ============================================================

def safe_angle_between(v1, v2):
    """Compute angle in degrees between two vectors, safely."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_a))


def extract_hand_geometry_features(shot, release_frame):
    """
    Extract ~32 hand geometry features at the release frame.
    These are the inputs for predicting ball velocity.
    """
    feats = {}
    rf = release_frame

    # Get key positions
    wrist = shot.get_position('right_wrist', rf)
    elbow = shot.get_position('right_elbow', rf)
    shoulder = shot.get_position('right_shoulder', rf)

    # Fingertip distal positions
    ft_distal = {}
    for fname in ['second', 'third', 'fourth', 'fifth']:
        pos = shot.get_position(f'right_{fname}_finger_distal', rf)
        if pos is not None:
            ft_distal[fname] = pos

    # Finger MCP positions
    ft_mcp = {}
    for fname in ['second', 'third', 'fourth']:
        pos = shot.get_position(f'right_{fname}_finger_MCP', rf)
        if pos is not None:
            ft_mcp[fname] = pos

    # Finger PIP positions
    ft_pip = {}
    for fname in ['second', 'third', 'fourth', 'fifth']:
        pos = shot.get_position(f'right_{fname}_finger_PIP', rf)
        if pos is not None:
            ft_pip[fname] = pos

    # Finger DIP positions
    ft_dip = {}
    for fname in ['second', 'third', 'fourth', 'fifth']:
        pos = shot.get_position(f'right_{fname}_finger_DIP', rf)
        if pos is not None:
            ft_dip[fname] = pos

    if wrist is None or elbow is None or shoulder is None or len(ft_distal) < 2:
        return None

    ft_center = np.mean(list(ft_distal.values()), axis=0)
    ball_pos = wrist + 0.6 * (ft_center - wrist)
    hoop_dir = HOOP_POSITION - ball_pos
    hoop_dir_norm = hoop_dir / (np.linalg.norm(hoop_dir) + 1e-8)

    # --- A. Palm Normal Vector (5 features) ---
    if all(k in ft_mcp for k in ['second', 'third', 'fourth']):
        v1 = ft_mcp['third'] - ft_mcp['second']
        v2 = ft_mcp['fourth'] - ft_mcp['second']
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-8:
            normal = normal / norm_len
            # Orient away from palm (toward ball side)
            if np.dot(normal, ft_center - wrist) < 0:
                normal = -normal
            feats['palm_normal_x'] = normal[0]
            feats['palm_normal_y'] = normal[1]
            feats['palm_normal_z'] = normal[2]
            feats['palm_normal_elevation'] = np.degrees(
                np.arctan2(normal[2], np.sqrt(normal[0]**2 + normal[1]**2)))
            feats['palm_normal_hoop_align'] = np.dot(normal, hoop_dir_norm)
        else:
            feats['palm_normal_x'] = 0
            feats['palm_normal_y'] = 0
            feats['palm_normal_z'] = 0
            feats['palm_normal_elevation'] = 0
            feats['palm_normal_hoop_align'] = 0
    else:
        feats['palm_normal_x'] = 0
        feats['palm_normal_y'] = 0
        feats['palm_normal_z'] = 0
        feats['palm_normal_elevation'] = 0
        feats['palm_normal_hoop_align'] = 0

    # --- B. Shooting Axis (3 features) ---
    axis = ft_center - shoulder
    axis_len = np.linalg.norm(axis)
    if axis_len > 1e-8:
        axis_unit = axis / axis_len
        feats['shooting_axis_elevation'] = np.degrees(
            np.arctan2(axis_unit[2], np.sqrt(axis_unit[0]**2 + axis_unit[1]**2)))
        feats['shooting_axis_azimuth'] = np.degrees(np.arctan2(axis_unit[1], axis_unit[0]))
        feats['shooting_axis_hoop_align'] = np.dot(axis_unit, hoop_dir_norm)
    else:
        feats['shooting_axis_elevation'] = 0
        feats['shooting_axis_azimuth'] = 0
        feats['shooting_axis_hoop_align'] = 0

    # --- C. Wrist Snap Angle (2 features) ---
    forearm = wrist - elbow
    hand = ft_center - wrist
    feats['wrist_snap_angle'] = safe_angle_between(forearm, hand)

    # Wrist snap rate (change from previous frame)
    if rf > 0:
        wrist_prev = shot.get_position('right_wrist', rf - 1)
        elbow_prev = shot.get_position('right_elbow', rf - 1)
        ft_prev = []
        for fname in ['second', 'third', 'fourth']:
            p = shot.get_position(f'right_{fname}_finger_distal', rf - 1)
            if p is not None:
                ft_prev.append(p)
        if wrist_prev is not None and elbow_prev is not None and len(ft_prev) > 0:
            ft_center_prev = np.mean(ft_prev, axis=0)
            forearm_prev = wrist_prev - elbow_prev
            hand_prev = ft_center_prev - wrist_prev
            prev_angle = safe_angle_between(forearm_prev, hand_prev)
            feats['wrist_snap_rate'] = (feats['wrist_snap_angle'] - prev_angle) * 60.0  # deg/s
        else:
            feats['wrist_snap_rate'] = 0
    else:
        feats['wrist_snap_rate'] = 0

    # --- D. Finger Extension (4 features) ---
    extensions = []
    for fname in ['second', 'third', 'fourth', 'fifth']:
        if fname in ft_mcp and fname in ft_pip and fname in ft_dip:
            # Angle at PIP joint
            v1 = ft_mcp[fname] - ft_pip[fname]
            v2 = ft_dip[fname] - ft_pip[fname]
            angle = safe_angle_between(v1, v2)
            extensions.append(angle)

    if extensions:
        feats['finger_ext_mean'] = np.mean(extensions)
        feats['finger_ext_std'] = np.std(extensions) if len(extensions) > 1 else 0
    else:
        feats['finger_ext_mean'] = 0
        feats['finger_ext_std'] = 0

    # Finger spread (distance between outer fingertips)
    if 'second' in ft_distal and 'fifth' in ft_distal:
        feats['finger_spread'] = np.linalg.norm(ft_distal['second'] - ft_distal['fifth'])
    elif 'second' in ft_distal and 'fourth' in ft_distal:
        feats['finger_spread'] = np.linalg.norm(ft_distal['second'] - ft_distal['fourth'])
    else:
        feats['finger_spread'] = 0

    # Finger spread rate
    if rf > 0:
        ft_d_prev = {}
        for fname in ['second', 'fifth']:
            p = shot.get_position(f'right_{fname}_finger_distal', rf - 1)
            if p is not None:
                ft_d_prev[fname] = p
        if 'second' in ft_d_prev and 'fifth' in ft_d_prev:
            prev_spread = np.linalg.norm(ft_d_prev['second'] - ft_d_prev['fifth'])
            feats['finger_spread_rate'] = (feats['finger_spread'] - prev_spread) * 60.0
        else:
            feats['finger_spread_rate'] = 0
    else:
        feats['finger_spread_rate'] = 0

    # --- E. Kinematic Velocity at Release (6 features) ---
    dt = 1.0 / 60.0
    if rf >= 2:
        w_prev2 = shot.get_position('right_wrist', rf - 2)
        w_curr = wrist
        if w_prev2 is not None:
            wrist_vel = (w_curr - w_prev2) / (2 * dt)  # central diff in ft/s
            wrist_vel_ms = wrist_vel * FEET_TO_METERS  # convert to m/s
            feats['wrist_vel_x'] = wrist_vel_ms[0]
            feats['wrist_vel_y'] = wrist_vel_ms[1]
            feats['wrist_vel_z'] = wrist_vel_ms[2]
            feats['wrist_speed'] = np.linalg.norm(wrist_vel_ms)
        else:
            feats['wrist_vel_x'] = 0
            feats['wrist_vel_y'] = 0
            feats['wrist_vel_z'] = 0
            feats['wrist_speed'] = 0
    else:
        feats['wrist_vel_x'] = 0
        feats['wrist_vel_y'] = 0
        feats['wrist_vel_z'] = 0
        feats['wrist_speed'] = 0

    # Fingertip velocity differential
    if rf >= 2:
        ft_prev2 = []
        for fname in ['second', 'third', 'fourth']:
            p = shot.get_position(f'right_{fname}_finger_distal', rf - 2)
            if p is not None:
                ft_prev2.append(p)
        if len(ft_prev2) > 0:
            ft_center_prev2 = np.mean(ft_prev2, axis=0)
            ft_vel = (ft_center - ft_center_prev2) / (2 * dt) * FEET_TO_METERS
            feats['fingertip_speed'] = np.linalg.norm(ft_vel)
            feats['finger_wrist_speed_diff'] = feats['fingertip_speed'] - feats['wrist_speed']
        else:
            feats['fingertip_speed'] = 0
            feats['finger_wrist_speed_diff'] = 0
    else:
        feats['fingertip_speed'] = 0
        feats['finger_wrist_speed_diff'] = 0

    # --- F. Arm Geometry (4 features) ---
    upper_arm = elbow - shoulder
    forearm_vec = wrist - elbow

    feats['elbow_angle'] = safe_angle_between(upper_arm, forearm_vec)
    feats['forearm_elevation'] = np.degrees(
        np.arctan2(forearm_vec[2], np.sqrt(forearm_vec[0]**2 + forearm_vec[1]**2)))
    feats['upper_arm_elevation'] = np.degrees(
        np.arctan2(upper_arm[2], np.sqrt(upper_arm[0]**2 + upper_arm[1]**2)))
    feats['arm_extension'] = np.linalg.norm(ft_center - shoulder)

    # --- G. Release Position (4 features) ---
    feats['release_x'] = ball_pos[0]
    feats['release_y'] = ball_pos[1]
    feats['release_z'] = ball_pos[2]
    feats['release_dist_to_hoop'] = np.linalg.norm(ball_pos - HOOP_POSITION)

    # --- H. Temporal Features (4 features) ---
    # Palm normal angular velocity over last 5 frames
    if all(k in ft_mcp for k in ['second', 'third', 'fourth']) and rf >= 5:
        normals = []
        for f in range(rf - 5, rf + 1):
            mcps = {}
            for fname in ['second', 'third', 'fourth']:
                p = shot.get_position(f'right_{fname}_finger_MCP', f)
                if p is not None:
                    mcps[fname] = p
            if len(mcps) == 3:
                v1 = mcps['third'] - mcps['second']
                v2 = mcps['fourth'] - mcps['second']
                n = np.cross(v1, v2)
                nl = np.linalg.norm(n)
                if nl > 1e-8:
                    normals.append(n / nl)
        if len(normals) >= 2:
            angular_changes = [safe_angle_between(normals[i], normals[i+1])
                               for i in range(len(normals) - 1)]
            feats['palm_angular_vel'] = np.mean(angular_changes) * 60.0  # deg/s
        else:
            feats['palm_angular_vel'] = 0
    else:
        feats['palm_angular_vel'] = 0

    # Wrist velocity trend (is it accelerating or decelerating?)
    if rf >= 4:
        speeds = []
        for f in range(rf - 3, rf + 1):
            w1 = shot.get_position('right_wrist', f)
            w0 = shot.get_position('right_wrist', f - 1)
            if w1 is not None and w0 is not None:
                speeds.append(np.linalg.norm(w1 - w0) * 60.0 * FEET_TO_METERS)
        if len(speeds) >= 2:
            feats['wrist_vel_trend'] = speeds[-1] - speeds[0]
        else:
            feats['wrist_vel_trend'] = 0
    else:
        feats['wrist_vel_trend'] = 0

    # Elbow extension rate
    if rf >= 2:
        e_prev = shot.get_position('right_elbow', rf - 2)
        s_prev = shot.get_position('right_shoulder', rf - 2)
        w_prev = shot.get_position('right_wrist', rf - 2)
        if e_prev is not None and s_prev is not None and w_prev is not None:
            ua_prev = e_prev - s_prev
            fa_prev = w_prev - e_prev
            prev_elbow = safe_angle_between(ua_prev, fa_prev)
            feats['elbow_extension_rate'] = (feats['elbow_angle'] - prev_elbow) * 30.0  # /2frames
        else:
            feats['elbow_extension_rate'] = 0
    else:
        feats['elbow_extension_rate'] = 0

    # Guide hand lateral position (left wrist relative to right)
    left_wrist = shot.get_position('left_wrist', rf)
    if left_wrist is not None:
        feats['guide_hand_lateral'] = left_wrist[1] - wrist[1]  # Y difference
        feats['guide_hand_forward'] = left_wrist[0] - wrist[0]  # X difference
    else:
        feats['guide_hand_lateral'] = 0
        feats['guide_hand_forward'] = 0

    return feats


# ============================================================
# STEP 4: VELOCITY PREDICTION MODEL
# ============================================================

def train_velocity_models(X, y_vel, pids, feature_names):
    """
    Train per-player models predicting velocity components.

    Returns OOF predictions and trained models.
    """
    unique_pids = sorted(np.unique(pids))
    n = len(X)
    oof_vel = np.zeros((n, 3))
    models = {}

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y_vel[mask]
        n_p = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros((n_p, 3))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]

            for vi in range(3):
                y_tr = y_p[tr_idx, vi]

                # Ridge
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr, y_tr)
                ridge_pred = ridge.predict(X_val)

                # LightGBM
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=50, num_leaves=6, learning_rate=0.05,
                    min_data_in_leaf=8, reg_alpha=1.0, reg_lambda=1.0,
                    random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(X_tr, y_tr)
                lgb_pred = lgb_m.predict(X_val)

                # PLS (if enough samples)
                max_comp = min(8, n_p - len(val_idx) - 1)
                if max_comp >= 2:
                    pls = PLSRegression(n_components=min(5, max_comp))
                    pls.fit(X_tr, y_tr)
                    pls_pred = pls.predict(X_val).flatten()
                    fold_preds[val_idx, vi] = 0.4 * ridge_pred + 0.3 * lgb_pred + 0.3 * pls_pred
                else:
                    fold_preds[val_idx, vi] = 0.5 * ridge_pred + 0.5 * lgb_pred

        oof_vel[global_idx] = fold_preds

        # Train final models
        X_all_scaled = scaler.fit_transform(X_p)
        X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0)

        player_models = {'scaler': scaler}
        for vi, vname in enumerate(['vx', 'vy', 'vz']):
            y_all = y_p[:, vi]

            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_all_scaled, y_all)

            lgb_m = lgb.LGBMRegressor(
                n_estimators=50, num_leaves=6, learning_rate=0.05,
                min_data_in_leaf=8, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_all_scaled, y_all)

            max_comp = min(8, n_p - 1)
            if max_comp >= 2:
                pls = PLSRegression(n_components=min(5, max_comp))
                pls.fit(X_all_scaled, y_all)
                player_models[f'{vname}_pls'] = pls

            player_models[f'{vname}_ridge'] = ridge
            player_models[f'{vname}_lgb'] = lgb_m

        models[pid] = player_models

    return oof_vel, models


def predict_velocity(X, pids, models):
    """Predict velocity for new shots."""
    preds = np.zeros((len(X), 3))
    for i in range(len(X)):
        pid = pids[i]
        m = models[pid]
        x_scaled = m['scaler'].transform(X[i:i+1])
        x_scaled = np.nan_to_num(x_scaled, nan=0.0)

        for vi, vname in enumerate(['vx', 'vy', 'vz']):
            ridge_pred = m[f'{vname}_ridge'].predict(x_scaled)[0]
            lgb_pred = m[f'{vname}_lgb'].predict(x_scaled)[0]
            if f'{vname}_pls' in m:
                pls_pred = m[f'{vname}_pls'].predict(x_scaled).flatten()[0]
                preds[i, vi] = 0.4 * ridge_pred + 0.3 * lgb_pred + 0.3 * pls_pred
            else:
                preds[i, vi] = 0.5 * ridge_pred + 0.5 * lgb_pred

    return preds


# ============================================================
# STEP 5: SCALING AND BLENDING
# ============================================================

def scale_predictions(raw_preds, target):
    """Scale raw predictions to [0,1] using competition scalers."""
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


def get_next_submission_number():
    """Atomically get next submission number."""
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


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("INVERSE PROJECTILE PIPELINE")
    print("=" * 70)

    loader = DataLoader()
    df_train = loader.load_train()
    df_test = loader.load_test()
    n_train = len(df_train)
    n_test = len(df_test)

    print(f"Train: {n_train} shots, Test: {n_test} shots")

    # ============================================================
    # Phase 1: Compute true velocities for training shots
    # ============================================================
    print("\n--- PHASE 1: Inverse Projectile Solver ---")

    release_positions = []
    release_frames = []
    true_velocities = []
    player_ids = []
    shot_ids = []
    raw_targets = []
    valid_mask = []

    for i in range(n_train):
        shot = loader.get_shot(i, train=True)
        rf = estimate_release_frame(shot)
        pos = estimate_release_position(shot, rf)

        if pos is None:
            release_positions.append(np.zeros(3))
            release_frames.append(rf)
            true_velocities.append(np.zeros(3))
            player_ids.append(shot.player_id)
            shot_ids.append(shot.id)
            raw_targets.append([shot.targets['angle'], shot.targets['depth'],
                                shot.targets['left_right']])
            valid_mask.append(False)
            continue

        vel = compute_true_velocity(
            pos, shot.targets['angle'], shot.targets['depth'], shot.targets['left_right'])

        release_positions.append(pos)
        release_frames.append(rf)
        true_velocities.append(np.array([vel['vx'], vel['vy'], vel['vz']]))
        player_ids.append(shot.player_id)
        shot_ids.append(shot.id)
        raw_targets.append([shot.targets['angle'], shot.targets['depth'],
                            shot.targets['left_right']])
        valid_mask.append(vel['is_valid'])

        if i == 0:
            print(f"  Shot 0: pos={pos}, vel=({vel['vx']:.2f}, {vel['vy']:.2f}, {vel['vz']:.2f}) m/s"
                  f" speed={vel['speed']:.2f} m/s, t={vel['flight_time']:.3f}s"
                  f" launch={vel['launch_angle_deg']:.1f}deg")

    release_positions = np.array(release_positions)
    true_velocities = np.array(true_velocities)
    player_ids = np.array(player_ids)
    raw_targets = np.array(raw_targets)
    valid_mask = np.array(valid_mask)

    n_valid = valid_mask.sum()
    print(f"  Valid inverse solutions: {n_valid}/{n_train}")

    # Velocity statistics
    speeds = np.linalg.norm(true_velocities[valid_mask], axis=1)
    print(f"  Speed: mean={speeds.mean():.2f}, std={speeds.std():.2f}, "
          f"range=[{speeds.min():.2f}, {speeds.max():.2f}] m/s")
    launch_angles = np.degrees(np.arctan2(
        true_velocities[valid_mask, 2],
        np.sqrt(true_velocities[valid_mask, 0]**2 + true_velocities[valid_mask, 1]**2)))
    print(f"  Launch angle: mean={launch_angles.mean():.1f}, std={launch_angles.std():.1f} deg")

    # Round-trip validation
    print("\n  Round-trip validation (first 10 shots):")
    max_error = 0
    for i in range(min(10, n_train)):
        if not valid_mask[i]:
            continue
        sim = forward_simulate(release_positions[i], true_velocities[i])
        if sim is not None:
            err_a = abs(sim['angle'] - raw_targets[i, 0])
            err_d = abs(sim['depth'] - raw_targets[i, 1])
            err_lr = abs(sim['left_right'] - raw_targets[i, 2])
            max_error = max(max_error, err_a, err_d, err_lr)
            if i < 5:
                print(f"    Shot {i}: angle err={err_a:.6f} depth err={err_d:.6f} "
                      f"lr err={err_lr:.6f}")

    # Full round-trip
    errors = {'angle': [], 'depth': [], 'lr': []}
    for i in range(n_train):
        if not valid_mask[i]:
            continue
        sim = forward_simulate(release_positions[i], true_velocities[i])
        if sim is not None:
            errors['angle'].append(abs(sim['angle'] - raw_targets[i, 0]))
            errors['depth'].append(abs(sim['depth'] - raw_targets[i, 1]))
            errors['lr'].append(abs(sim['left_right'] - raw_targets[i, 2]))
    print(f"  Max round-trip error: angle={max(errors['angle']):.8f}, "
          f"depth={max(errors['depth']):.8f}, lr={max(errors['lr']):.8f}")

    # ============================================================
    # Phase 2: Extract hand geometry features
    # ============================================================
    print("\n--- PHASE 2: Hand Geometry Features ---")

    train_features = []
    for i in range(n_train):
        shot = loader.get_shot(i, train=True)
        feats = extract_hand_geometry_features(shot, release_frames[i])
        train_features.append(feats)
        if i == 0 and feats is not None:
            print(f"  Features per shot: {len(feats)}")
            print(f"  Feature names: {sorted(feats.keys())[:10]}...")

    # Handle None features
    feature_names = None
    for f in train_features:
        if f is not None:
            feature_names = sorted(f.keys())
            break

    if feature_names is None:
        print("ERROR: No valid features extracted!")
        return

    print(f"  Total features: {len(feature_names)}")

    X_train = np.zeros((n_train, len(feature_names)))
    feat_valid = np.ones(n_train, dtype=bool)
    for i in range(n_train):
        if train_features[i] is None:
            feat_valid[i] = False
            continue
        for j, fn in enumerate(feature_names):
            X_train[i, j] = train_features[i].get(fn, 0.0)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Combined validity
    combined_valid = valid_mask & feat_valid
    n_combined = combined_valid.sum()
    print(f"  Shots with valid velocity + features: {n_combined}/{n_train}")

    # Test features
    test_release_positions = []
    test_release_frames = []
    test_player_ids = []
    test_shot_ids = []
    test_features = []

    for i in range(n_test):
        shot = loader.get_shot(i, train=False)
        rf = estimate_release_frame(shot)
        pos = estimate_release_position(shot, rf)

        test_release_frames.append(rf)
        test_release_positions.append(pos if pos is not None else np.zeros(3))
        test_player_ids.append(shot.player_id)
        test_shot_ids.append(shot.id)
        feats = extract_hand_geometry_features(shot, rf)
        test_features.append(feats)

    X_test = np.zeros((n_test, len(feature_names)))
    for i in range(n_test):
        if test_features[i] is not None:
            for j, fn in enumerate(feature_names):
                X_test[i, j] = test_features[i].get(fn, 0.0)

    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    test_release_positions = np.array(test_release_positions)
    test_player_ids = np.array(test_player_ids)

    # ============================================================
    # Phase 3: Feature-velocity correlations
    # ============================================================
    print("\n--- PHASE 3: Feature-Velocity Correlations ---")

    for vi, vname in enumerate(['vx', 'vy', 'vz']):
        print(f"\n  {vname}:")
        corrs = []
        y_v = true_velocities[combined_valid, vi]
        for fi, fn in enumerate(feature_names):
            x_f = X_train[combined_valid, fi]
            if np.std(x_f) > 1e-8 and np.std(y_v) > 1e-8:
                r, p = pearsonr(x_f, y_v)
                corrs.append((fn, r, p))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for fn, r, p in corrs[:8]:
            sig = "*" if p < 0.05 else " "
            print(f"    {fn:>35}: r={r:+.4f} p={p:.4f} {sig}")

    # ============================================================
    # Phase 4: Train velocity prediction model
    # ============================================================
    print("\n--- PHASE 4: Velocity Prediction Model ---")

    # Use only valid shots for training
    X_valid = X_train[combined_valid]
    y_vel_valid = true_velocities[combined_valid]
    pids_valid = player_ids[combined_valid]

    oof_vel, vel_models = train_velocity_models(
        X_valid, y_vel_valid, pids_valid, feature_names)

    # Evaluate velocity prediction
    print("\n  Velocity prediction (OOF):")
    for vi, vname in enumerate(['vx', 'vy', 'vz']):
        actual = y_vel_valid[:, vi]
        pred = oof_vel[:, vi]
        r, _ = pearsonr(actual, pred)
        rmse = np.sqrt(np.mean((actual - pred)**2))
        print(f"    {vname}: R={r:.4f}, RMSE={rmse:.4f} m/s")

    speed_actual = np.linalg.norm(y_vel_valid, axis=1)
    speed_pred = np.linalg.norm(oof_vel, axis=1)
    r_speed, _ = pearsonr(speed_actual, speed_pred)
    print(f"    speed: R={r_speed:.4f}")

    # ============================================================
    # Phase 5: Forward simulate OOF predictions -> target predictions
    # ============================================================
    print("\n--- PHASE 5: Forward Simulation (OOF) ---")

    # For valid training shots, forward simulate from predicted velocity
    oof_targets = np.zeros((combined_valid.sum(), 3))  # angle, depth, lr
    sim_failures = 0

    valid_indices = np.where(combined_valid)[0]
    for ii, orig_idx in enumerate(valid_indices):
        sim = forward_simulate(release_positions[orig_idx], oof_vel[ii])
        if sim is not None:
            oof_targets[ii, 0] = sim['angle']
            oof_targets[ii, 1] = sim['depth']
            oof_targets[ii, 2] = sim['left_right']
        else:
            sim_failures += 1
            # Fallback to mean targets
            oof_targets[ii, 0] = raw_targets[combined_valid, 0].mean()
            oof_targets[ii, 1] = raw_targets[combined_valid, 1].mean()
            oof_targets[ii, 2] = raw_targets[combined_valid, 2].mean()

    print(f"  Simulation failures: {sim_failures}/{combined_valid.sum()}")

    # Evaluate target prediction
    actual_targets = raw_targets[combined_valid]
    print("\n  Target prediction accuracy (raw units):")
    for ti, tname in enumerate(['angle', 'depth', 'left_right']):
        actual = actual_targets[:, ti]
        pred = oof_targets[:, ti]
        r, _ = pearsonr(actual, pred)
        rmse = np.sqrt(np.mean((actual - pred)**2))
        mse = np.mean((actual - pred)**2)
        mean_baseline_mse = np.var(actual)
        print(f"    {tname:>12}: R={r:.4f}, RMSE={rmse:.4f}, MSE={mse:.4f}, "
              f"mean_baseline_MSE={mean_baseline_mse:.4f}")

    # Compute scaled MSE (competition metric)
    print("\n  Scaled MSE (competition metric):")
    total_scaled_mse = 0
    for ti, tname in enumerate(['angle', 'depth', 'left_right']):
        actual_raw = actual_targets[:, ti]
        pred_raw = oof_targets[:, ti]

        actual_scaled = scale_predictions(actual_raw, tname)
        pred_scaled = scale_predictions(pred_raw, tname)

        mse = np.mean((actual_scaled - pred_scaled)**2)
        total_scaled_mse += mse
        print(f"    {tname:>12}: scaled MSE = {mse:.6f}")

    mean_scaled_mse = total_scaled_mse / 3
    print(f"    {'MEAN':>12}: {mean_scaled_mse:.6f}")
    print(f"    {'LB best':>12}: 0.007224 (Sub 784)")
    print(f"    {'Pred mean':>12}: ~0.0088")

    # ============================================================
    # Phase 6: Generate submission
    # ============================================================
    print("\n--- PHASE 6: Test Predictions + Submission ---")

    # Predict velocity for test shots
    test_vel_preds = predict_velocity(X_test, test_player_ids, vel_models)
    print(f"  Test velocity predictions: shape={test_vel_preds.shape}")
    test_speeds = np.linalg.norm(test_vel_preds, axis=1)
    print(f"  Test speeds: mean={test_speeds.mean():.2f}, range=[{test_speeds.min():.2f}, "
          f"{test_speeds.max():.2f}] m/s")

    # Forward simulate test shots
    test_targets = np.zeros((n_test, 3))
    test_sim_fails = 0
    for i in range(n_test):
        sim = forward_simulate(test_release_positions[i], test_vel_preds[i])
        if sim is not None:
            test_targets[i, 0] = sim['angle']
            test_targets[i, 1] = sim['depth']
            test_targets[i, 2] = sim['left_right']
        else:
            test_sim_fails += 1
            # Fallback
            test_targets[i, 0] = raw_targets[:, 0].mean()
            test_targets[i, 1] = raw_targets[:, 1].mean()
            test_targets[i, 2] = raw_targets[:, 2].mean()

    print(f"  Test simulation failures: {test_sim_fails}/{n_test}")

    # Scale to competition format
    test_angle_scaled = scale_predictions(test_targets[:, 0], 'angle')
    test_depth_scaled = scale_predictions(test_targets[:, 1], 'depth')
    test_lr_scaled = scale_predictions(test_targets[:, 2], 'left_right')

    print(f"  Scaled ranges:")
    print(f"    angle: [{test_angle_scaled.min():.4f}, {test_angle_scaled.max():.4f}], "
          f"std={test_angle_scaled.std():.4f}")
    print(f"    depth: [{test_depth_scaled.min():.4f}, {test_depth_scaled.max():.4f}], "
          f"mean={test_depth_scaled.mean():.4f}")
    print(f"    lr: [{test_lr_scaled.min():.4f}, {test_lr_scaled.max():.4f}]")

    # Save standalone submission
    sub_num = get_next_submission_number()
    test_ids = [loader.get_shot(i, train=False).id for i in range(n_test)]
    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': test_angle_scaled,
        'scaled_depth': test_depth_scaled,
        'scaled_left_right': test_lr_scaled,
    })
    filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(filepath, index=False)
    print(f"\n  Standalone submission saved: {filepath}")

    # Blend with Sub 784
    sub784_path = SUBMISSION_DIR / "submission_784.csv"
    if sub784_path.exists():
        print("\n  Blending with Sub 784...")
        sub784 = pd.read_csv(sub784_path)

        our = pd.DataFrame({
            'id': test_ids,
            'new_angle': test_angle_scaled,
            'new_depth': test_depth_scaled,
            'new_lr': test_lr_scaled,
        })
        merged = sub784.merge(our, on='id')

        for col, nc in [('scaled_angle', 'new_angle'), ('scaled_depth', 'new_depth'),
                         ('scaled_left_right', 'new_lr')]:
            corr = np.corrcoef(merged[col], merged[nc])[0, 1]
            print(f"    {col} vs physics pred: r={corr:.4f}")

        # Grid search blend weights
        best_configs = []
        for aw in [0.0, 0.05, 0.10, 0.15]:
            for dw in [0.0, 0.05, 0.10, 0.15, 0.20]:
                for lw in [0.0, 0.05, 0.10, 0.15, 0.20]:
                    ba = (1-aw)*merged['scaled_angle'] + aw*merged['new_angle']
                    bd = (1-dw)*merged['scaled_depth'] + dw*merged['new_depth']
                    bl = (1-lw)*merged['scaled_left_right'] + lw*merged['new_lr']

                    a_std = ba.std()
                    d_mean = bd.mean()

                    if a_std < 0.15 and 0.49 < d_mean < 0.52:
                        da = np.mean((ba - merged['scaled_angle'])**2)
                        dd = np.mean((bd - merged['scaled_depth'])**2)
                        dl = np.mean((bl - merged['scaled_left_right'])**2)
                        div = da + dd + dl

                        best_configs.append({
                            'aw': aw, 'dw': dw, 'lw': lw,
                            'angle_std': a_std, 'depth_mean': d_mean,
                            'diversity': div,
                            'ba': ba.values, 'bd': bd.values, 'bl': bl.values,
                        })

        best_configs.sort(key=lambda x: -x['diversity'])

        print(f"\n  Valid configs: {len(best_configs)}")
        print(f"  {'aw':>4} {'dw':>4} {'lw':>4} | {'angle_std':>10} {'depth_mean':>10} | diversity")
        for c in best_configs[:10]:
            print(f"  {c['aw']:>4.2f} {c['dw']:>4.2f} {c['lw']:>4.2f} | "
                  f"{c['angle_std']:>10.6f} {c['depth_mean']:>10.6f} | {c['diversity']:.8f}")

        # Save top blended submissions
        for config in best_configs[:3]:
            sub_num = get_next_submission_number()
            sub = pd.DataFrame({
                'id': merged['id'],
                'scaled_angle': config['ba'],
                'scaled_depth': config['bd'],
                'scaled_left_right': config['bl'],
            })
            filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            sub.to_csv(filepath, index=False)
            print(f"  Sub {sub_num}: aw={config['aw']:.2f} dw={config['dw']:.2f} "
                  f"lw={config['lw']:.2f} div={config['diversity']:.8f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
