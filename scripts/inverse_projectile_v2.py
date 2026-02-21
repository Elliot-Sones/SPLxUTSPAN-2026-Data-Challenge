"""
Inverse Projectile Pipeline v2

Improvements over v1:
1. Direct target prediction from hand features + OOF velocity features
   (bypasses error-amplifying forward simulation)
2. Velocity prediction with clipping to physical range
3. Feature augmentation: add physics features to existing hoop-relative pipeline
4. Multiple prediction paths blended together
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
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

sys.path.insert(0, str(PROJECT_DIR))
from physics_engine.core.data_loader import (
    DataLoader, HOOP_POSITION, FEET_TO_METERS
)

G = 9.81
HOOP_X_FT = 5.25
HOOP_Y_FT = -25.0
HOOP_Z_FT = 10.0

# Physical bounds for free throw velocity
SPEED_MIN = 5.5  # m/s
SPEED_MAX = 8.5  # m/s


# ============================================================
# INVERSE PROJECTILE (from v1)
# ============================================================

def compute_true_velocity(release_pos_ft, angle_deg, depth_in, lr_in):
    x_land_ft = HOOP_X_FT + lr_in / 12.0
    y_land_ft = HOOP_Y_FT + depth_in / 12.0
    dx = (x_land_ft - release_pos_ft[0]) * FEET_TO_METERS
    dy = (y_land_ft - release_pos_ft[1]) * FEET_TO_METERS
    dz = (HOOP_Z_FT - release_pos_ft[2]) * FEET_TO_METERS
    D = np.sqrt(dx**2 + dy**2)
    angle_rad = np.radians(angle_deg)
    t_sq = 2.0 * (dz + D * np.tan(angle_rad)) / G
    if t_sq <= 0:
        return None
    t = np.sqrt(t_sq)
    vx = dx / t
    vy = dy / t
    vz = (dz + 0.5 * G * t**2) / t
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    return {'vx': vx, 'vy': vy, 'vz': vz, 'speed': speed, 'flight_time': t}


def forward_simulate(release_pos_ft, vel_ms):
    pos_m = np.array(release_pos_ft) * FEET_TO_METERS
    dz_m = HOOP_Z_FT * FEET_TO_METERS - pos_m[2]
    a = 0.5 * G
    b = -vel_ms[2]
    c = dz_m
    disc = b**2 - 4*a*c
    if disc < 0:
        return None
    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)
    times = [t for t in [t1, t2] if t > 0]
    if not times:
        return None
    t = max(times)
    if t > 5.0:
        return None
    landing_x_m = pos_m[0] + vel_ms[0] * t
    landing_y_m = pos_m[1] + vel_ms[1] * t
    landing_vz = vel_ms[2] - G * t
    horiz_speed = np.sqrt(vel_ms[0]**2 + vel_ms[1]**2)
    entry_angle = np.degrees(np.arctan2(-landing_vz, horiz_speed))
    depth_in = (landing_y_m / FEET_TO_METERS - HOOP_Y_FT) * 12.0
    lr_in = (landing_x_m / FEET_TO_METERS - HOOP_X_FT) * 12.0
    return {'angle': entry_angle, 'depth': depth_in, 'left_right': lr_in, 'flight_time': t}


# ============================================================
# RELEASE DETECTION + HAND FEATURES (from v1)
# ============================================================

def estimate_release_frame(shot):
    wrist_z = [shot.get_position('right_wrist', f)[2]
               if shot.get_position('right_wrist', f) is not None else 0
               for f in range(shot.num_frames)]
    return int(np.argmax(wrist_z))


def estimate_release_position(shot, rf):
    wrist = shot.get_position('right_wrist', rf)
    fts = []
    for name in ['right_second_finger_distal', 'right_third_finger_distal',
                  'right_fourth_finger_distal']:
        p = shot.get_position(name, rf)
        if p is not None:
            fts.append(p)
    if wrist is None or len(fts) == 0:
        return None
    ft_center = np.mean(fts, axis=0)
    return wrist + 0.6 * (ft_center - wrist)


def safe_angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def extract_hand_geometry_features(shot, rf):
    """Extract hand geometry features (same as v1)."""
    feats = {}
    wrist = shot.get_position('right_wrist', rf)
    elbow = shot.get_position('right_elbow', rf)
    shoulder = shot.get_position('right_shoulder', rf)

    ft_distal = {}
    for fname in ['second', 'third', 'fourth', 'fifth']:
        pos = shot.get_position(f'right_{fname}_finger_distal', rf)
        if pos is not None:
            ft_distal[fname] = pos

    ft_mcp = {}
    for fname in ['second', 'third', 'fourth']:
        pos = shot.get_position(f'right_{fname}_finger_MCP', rf)
        if pos is not None:
            ft_mcp[fname] = pos

    ft_pip = {}
    for fname in ['second', 'third', 'fourth', 'fifth']:
        pos = shot.get_position(f'right_{fname}_finger_PIP', rf)
        if pos is not None:
            ft_pip[fname] = pos

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

    # Palm normal
    if all(k in ft_mcp for k in ['second', 'third', 'fourth']):
        v1 = ft_mcp['third'] - ft_mcp['second']
        v2 = ft_mcp['fourth'] - ft_mcp['second']
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-8:
            normal = normal / norm_len
            if np.dot(normal, ft_center - wrist) < 0:
                normal = -normal
            feats['palm_normal_x'] = normal[0]
            feats['palm_normal_y'] = normal[1]
            feats['palm_normal_z'] = normal[2]
            feats['palm_normal_elevation'] = np.degrees(
                np.arctan2(normal[2], np.sqrt(normal[0]**2 + normal[1]**2)))
            feats['palm_normal_hoop_align'] = np.dot(normal, hoop_dir_norm)
        else:
            for k in ['palm_normal_x', 'palm_normal_y', 'palm_normal_z',
                       'palm_normal_elevation', 'palm_normal_hoop_align']:
                feats[k] = 0
    else:
        for k in ['palm_normal_x', 'palm_normal_y', 'palm_normal_z',
                   'palm_normal_elevation', 'palm_normal_hoop_align']:
            feats[k] = 0

    # Shooting axis
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

    # Wrist snap
    forearm = wrist - elbow
    hand = ft_center - wrist
    feats['wrist_snap_angle'] = safe_angle_between(forearm, hand)

    if rf > 0:
        wp = shot.get_position('right_wrist', rf - 1)
        ep = shot.get_position('right_elbow', rf - 1)
        ftp = [shot.get_position(f'right_{f}_finger_distal', rf - 1)
               for f in ['second', 'third', 'fourth']]
        ftp = [p for p in ftp if p is not None]
        if wp is not None and ep is not None and len(ftp) > 0:
            fc_prev = np.mean(ftp, axis=0)
            prev_angle = safe_angle_between(wp - ep, fc_prev - wp)
            feats['wrist_snap_rate'] = (feats['wrist_snap_angle'] - prev_angle) * 60.0
        else:
            feats['wrist_snap_rate'] = 0
    else:
        feats['wrist_snap_rate'] = 0

    # Finger extension
    extensions = []
    for fname in ['second', 'third', 'fourth', 'fifth']:
        if fname in ft_mcp and fname in ft_pip and fname in ft_dip:
            v1 = ft_mcp[fname] - ft_pip[fname]
            v2 = ft_dip[fname] - ft_pip[fname]
            extensions.append(safe_angle_between(v1, v2))
    feats['finger_ext_mean'] = np.mean(extensions) if extensions else 0
    feats['finger_ext_std'] = np.std(extensions) if len(extensions) > 1 else 0

    if 'second' in ft_distal and 'fifth' in ft_distal:
        feats['finger_spread'] = np.linalg.norm(ft_distal['second'] - ft_distal['fifth'])
    elif 'second' in ft_distal and 'fourth' in ft_distal:
        feats['finger_spread'] = np.linalg.norm(ft_distal['second'] - ft_distal['fourth'])
    else:
        feats['finger_spread'] = 0

    if rf > 0:
        fd_prev = {}
        for fname in ['second', 'fifth']:
            p = shot.get_position(f'right_{fname}_finger_distal', rf - 1)
            if p is not None:
                fd_prev[fname] = p
        if 'second' in fd_prev and 'fifth' in fd_prev:
            ps = np.linalg.norm(fd_prev['second'] - fd_prev['fifth'])
            feats['finger_spread_rate'] = (feats['finger_spread'] - ps) * 60.0
        else:
            feats['finger_spread_rate'] = 0
    else:
        feats['finger_spread_rate'] = 0

    # Wrist velocity
    dt = 1.0 / 60.0
    if rf >= 2:
        wp2 = shot.get_position('right_wrist', rf - 2)
        if wp2 is not None:
            wv = (wrist - wp2) / (2 * dt) * FEET_TO_METERS
            feats['wrist_vel_x'] = wv[0]
            feats['wrist_vel_y'] = wv[1]
            feats['wrist_vel_z'] = wv[2]
            feats['wrist_speed'] = np.linalg.norm(wv)
        else:
            feats['wrist_vel_x'] = feats['wrist_vel_y'] = feats['wrist_vel_z'] = 0
            feats['wrist_speed'] = 0
    else:
        feats['wrist_vel_x'] = feats['wrist_vel_y'] = feats['wrist_vel_z'] = 0
        feats['wrist_speed'] = 0

    if rf >= 2:
        ftp2 = [shot.get_position(f'right_{f}_finger_distal', rf - 2)
                for f in ['second', 'third', 'fourth']]
        ftp2 = [p for p in ftp2 if p is not None]
        if len(ftp2) > 0:
            fc2 = np.mean(ftp2, axis=0)
            fv = (ft_center - fc2) / (2 * dt) * FEET_TO_METERS
            feats['fingertip_speed'] = np.linalg.norm(fv)
            feats['finger_wrist_speed_diff'] = feats['fingertip_speed'] - feats['wrist_speed']
        else:
            feats['fingertip_speed'] = feats['finger_wrist_speed_diff'] = 0
    else:
        feats['fingertip_speed'] = feats['finger_wrist_speed_diff'] = 0

    # Arm geometry
    upper_arm = elbow - shoulder
    forearm_vec = wrist - elbow
    feats['elbow_angle'] = safe_angle_between(upper_arm, forearm_vec)
    feats['forearm_elevation'] = np.degrees(
        np.arctan2(forearm_vec[2], np.sqrt(forearm_vec[0]**2 + forearm_vec[1]**2)))
    feats['upper_arm_elevation'] = np.degrees(
        np.arctan2(upper_arm[2], np.sqrt(upper_arm[0]**2 + upper_arm[1]**2)))
    feats['arm_extension'] = np.linalg.norm(ft_center - shoulder)

    # Release position
    feats['release_x'] = ball_pos[0]
    feats['release_y'] = ball_pos[1]
    feats['release_z'] = ball_pos[2]
    feats['release_dist_to_hoop'] = np.linalg.norm(ball_pos - HOOP_POSITION)

    # Temporal features
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
            ac = [safe_angle_between(normals[i], normals[i+1]) for i in range(len(normals)-1)]
            feats['palm_angular_vel'] = np.mean(ac) * 60.0
        else:
            feats['palm_angular_vel'] = 0
    else:
        feats['palm_angular_vel'] = 0

    if rf >= 4:
        speeds = []
        for f in range(rf - 3, rf + 1):
            w1 = shot.get_position('right_wrist', f)
            w0 = shot.get_position('right_wrist', f - 1)
            if w1 is not None and w0 is not None:
                speeds.append(np.linalg.norm(w1 - w0) * 60.0 * FEET_TO_METERS)
        feats['wrist_vel_trend'] = speeds[-1] - speeds[0] if len(speeds) >= 2 else 0
    else:
        feats['wrist_vel_trend'] = 0

    if rf >= 2:
        ep2 = shot.get_position('right_elbow', rf - 2)
        sp2 = shot.get_position('right_shoulder', rf - 2)
        wp2 = shot.get_position('right_wrist', rf - 2)
        if ep2 is not None and sp2 is not None and wp2 is not None:
            prev_elbow = safe_angle_between(ep2 - sp2, wp2 - ep2)
            feats['elbow_extension_rate'] = (feats['elbow_angle'] - prev_elbow) * 30.0
        else:
            feats['elbow_extension_rate'] = 0
    else:
        feats['elbow_extension_rate'] = 0

    lw = shot.get_position('left_wrist', rf)
    if lw is not None:
        feats['guide_hand_lateral'] = lw[1] - wrist[1]
        feats['guide_hand_forward'] = lw[0] - wrist[0]
    else:
        feats['guide_hand_lateral'] = feats['guide_hand_forward'] = 0

    return feats


# ============================================================
# APPROACH 1: Direct target prediction from hand features
# (bypasses forward simulation - avoids error amplification)
# ============================================================

def train_direct_target_models(X, y_targets, pids, vel_features=None):
    """
    Train per-player per-target models that predict scaled targets
    directly from hand features (+ optionally OOF velocity features).

    This avoids the error amplification of forward simulation.
    """
    unique_pids = sorted(np.unique(pids))
    n = len(X)
    n_targets = y_targets.shape[1]
    oof_preds = np.zeros((n, n_targets))
    models = {}

    # Optionally augment features with velocity predictions
    if vel_features is not None:
        X_aug = np.hstack([X, vel_features])
    else:
        X_aug = X

    for pid in unique_pids:
        mask = pids == pid
        X_p = X_aug[mask]
        n_p = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)

        player_models = {'scaler': scaler}

        for ti in range(n_targets):
            y_t = y_targets[mask, ti]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n_p)

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
                y_tr = y_t[tr_idx]

                preds_list = []

                # Ridge
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr, y_tr)
                preds_list.append(ridge.predict(X_val))

                # LightGBM
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=8, learning_rate=0.05,
                    min_data_in_leaf=8, reg_alpha=0.5, reg_lambda=0.5,
                    random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(X_tr, y_tr)
                preds_list.append(lgb_m.predict(X_val))

                # XGBoost
                xgb_m = xgb.XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42,
                    verbosity=0, n_jobs=-1)
                xgb_m.fit(X_tr, y_tr)
                preds_list.append(xgb_m.predict(X_val))

                # CatBoost
                cat_m = CatBoostRegressor(
                    iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)
                cat_m.fit(X_tr, y_tr)
                preds_list.append(cat_m.predict(X_val))

                fold_preds[val_idx] = (0.3 * preds_list[0] + 0.3 * preds_list[1] +
                                       0.25 * preds_list[2] + 0.15 * preds_list[3])

            oof_preds[global_idx, ti] = fold_preds

            # Train final models on all data
            for name, cls, params in [
                ('ridge', RidgeCV, dict(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
                ('lgb', lgb.LGBMRegressor, dict(
                    n_estimators=100, num_leaves=8, learning_rate=0.05,
                    min_data_in_leaf=8, reg_alpha=0.5, reg_lambda=0.5,
                    random_state=42, verbose=-1, n_jobs=-1)),
                ('xgb', xgb.XGBRegressor, dict(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42,
                    verbosity=0, n_jobs=-1)),
                ('cat', CatBoostRegressor, dict(
                    iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)),
            ]:
                m = cls(**params)
                m.fit(X_scaled, y_t)
                player_models[f't{ti}_{name}'] = m

        models[pid] = player_models

    return oof_preds, models


def predict_direct_targets(X, pids, models, vel_features=None):
    """Predict targets directly for new shots."""
    if vel_features is not None:
        X_aug = np.hstack([X, vel_features])
    else:
        X_aug = X

    n = len(X)
    preds = np.zeros((n, 3))
    for i in range(n):
        pid = pids[i]
        m = models[pid]
        x_scaled = np.nan_to_num(m['scaler'].transform(X_aug[i:i+1]), nan=0.0)
        for ti in range(3):
            p = []
            for name in ['ridge', 'lgb', 'xgb', 'cat']:
                p.append(m[f't{ti}_{name}'].predict(x_scaled)[0])
            preds[i, ti] = 0.3*p[0] + 0.3*p[1] + 0.25*p[2] + 0.15*p[3]
    return preds


# ============================================================
# VELOCITY PREDICTION (with clipping)
# ============================================================

def train_velocity_models(X, y_vel, pids):
    """Train per-player velocity prediction models."""
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
        X_scaled = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros((n_p, 3))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            for vi in range(3):
                y_tr = y_p[tr_idx, vi]
                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr, y_tr)
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=50, num_leaves=6, learning_rate=0.05,
                    min_data_in_leaf=8, reg_alpha=1.0, reg_lambda=1.0,
                    random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(X_tr, y_tr)
                max_comp = min(8, len(tr_idx) - 1)
                if max_comp >= 2:
                    pls = PLSRegression(n_components=min(5, max_comp))
                    pls.fit(X_tr, y_tr)
                    fold_preds[val_idx, vi] = (0.4 * ridge.predict(X_val) +
                                               0.3 * lgb_m.predict(X_val) +
                                               0.3 * pls.predict(X_val).flatten())
                else:
                    fold_preds[val_idx, vi] = (0.5 * ridge.predict(X_val) +
                                               0.5 * lgb_m.predict(X_val))

        oof_vel[global_idx] = fold_preds

        # Final models
        X_all = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)
        pm = {'scaler': scaler}
        for vi, vn in enumerate(['vx', 'vy', 'vz']):
            y_all = y_p[:, vi]
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_all, y_all)
            lgb_m = lgb.LGBMRegressor(
                n_estimators=50, num_leaves=6, learning_rate=0.05,
                min_data_in_leaf=8, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_all, y_all)
            pm[f'{vn}_ridge'] = ridge
            pm[f'{vn}_lgb'] = lgb_m
            max_comp = min(8, n_p - 1)
            if max_comp >= 2:
                pls = PLSRegression(n_components=min(5, max_comp))
                pls.fit(X_all, y_all)
                pm[f'{vn}_pls'] = pls
        models[pid] = pm

    return oof_vel, models


def predict_velocity(X, pids, models):
    preds = np.zeros((len(X), 3))
    for i in range(len(X)):
        pid = pids[i]
        m = models[pid]
        x_s = np.nan_to_num(m['scaler'].transform(X[i:i+1]), nan=0.0)
        for vi, vn in enumerate(['vx', 'vy', 'vz']):
            rp = m[f'{vn}_ridge'].predict(x_s)[0]
            lp = m[f'{vn}_lgb'].predict(x_s)[0]
            if f'{vn}_pls' in m:
                pp = m[f'{vn}_pls'].predict(x_s).flatten()[0]
                preds[i, vi] = 0.4 * rp + 0.3 * lp + 0.3 * pp
            else:
                preds[i, vi] = 0.5 * rp + 0.5 * lp
    return preds


def clip_velocity(vel_array):
    """Clip velocity predictions to physical range."""
    clipped = vel_array.copy()
    speeds = np.linalg.norm(clipped, axis=1)
    for i in range(len(clipped)):
        if speeds[i] < SPEED_MIN:
            clipped[i] = clipped[i] * (SPEED_MIN / (speeds[i] + 1e-8))
        elif speeds[i] > SPEED_MAX:
            clipped[i] = clipped[i] * (SPEED_MAX / (speeds[i] + 1e-8))
    return clipped


# ============================================================
# UTILS
# ============================================================

def scale_predictions(raw_preds, target):
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


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


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("INVERSE PROJECTILE PIPELINE v2")
    print("=" * 70)

    loader = DataLoader()
    df_train = loader.load_train()
    df_test = loader.load_test()
    n_train = len(df_train)
    n_test = len(df_test)

    # ============================================================
    # Phase 1: Compute true velocities + extract features
    # ============================================================
    print("\n--- PHASE 1: Data Preparation ---")

    release_positions = np.zeros((n_train, 3))
    release_frames_arr = np.zeros(n_train, dtype=int)
    true_velocities = np.zeros((n_train, 3))
    player_ids = np.zeros(n_train, dtype=int)
    raw_targets = np.zeros((n_train, 3))
    valid_mask = np.zeros(n_train, dtype=bool)
    train_ids = []

    for i in range(n_train):
        shot = loader.get_shot(i, train=True)
        rf = estimate_release_frame(shot)
        pos = estimate_release_position(shot, rf)
        release_frames_arr[i] = rf
        player_ids[i] = shot.player_id
        train_ids.append(shot.id)
        raw_targets[i] = [shot.targets['angle'], shot.targets['depth'],
                           shot.targets['left_right']]

        if pos is not None:
            release_positions[i] = pos
            vel = compute_true_velocity(pos, *raw_targets[i])
            if vel is not None:
                true_velocities[i] = [vel['vx'], vel['vy'], vel['vz']]
                valid_mask[i] = True

    n_valid = valid_mask.sum()
    print(f"  Valid inverse solutions: {n_valid}/{n_train}")
    speeds = np.linalg.norm(true_velocities[valid_mask], axis=1)
    print(f"  Speed: mean={speeds.mean():.2f}, std={speeds.std():.2f}, "
          f"range=[{speeds.min():.2f}, {speeds.max():.2f}] m/s")

    # Extract hand features
    print("  Extracting hand geometry features (train)...")
    train_feats = []
    for i in range(n_train):
        shot = loader.get_shot(i, train=True)
        f = extract_hand_geometry_features(shot, release_frames_arr[i])
        train_feats.append(f)

    feature_names = sorted(next(f for f in train_feats if f is not None).keys())
    print(f"  Features: {len(feature_names)}")

    X_train = np.zeros((n_train, len(feature_names)))
    feat_valid = np.ones(n_train, dtype=bool)
    for i in range(n_train):
        if train_feats[i] is None:
            feat_valid[i] = False
        else:
            for j, fn in enumerate(feature_names):
                X_train[i, j] = train_feats[i].get(fn, 0.0)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    combined_valid = valid_mask & feat_valid
    print(f"  Combined valid: {combined_valid.sum()}/{n_train}")

    # Test features
    print("  Extracting hand geometry features (test)...")
    test_release_positions = np.zeros((n_test, 3))
    test_release_frames = np.zeros(n_test, dtype=int)
    test_player_ids = np.zeros(n_test, dtype=int)
    test_ids = []

    X_test = np.zeros((n_test, len(feature_names)))
    for i in range(n_test):
        shot = loader.get_shot(i, train=False)
        rf = estimate_release_frame(shot)
        pos = estimate_release_position(shot, rf)
        test_release_frames[i] = rf
        test_release_positions[i] = pos if pos is not None else np.zeros(3)
        test_player_ids[i] = shot.player_id
        test_ids.append(shot.id)

        f = extract_hand_geometry_features(shot, rf)
        if f is not None:
            for j, fn in enumerate(feature_names):
                X_test[i, j] = f.get(fn, 0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ============================================================
    # Phase 2: Train velocity models + get OOF velocity predictions
    # ============================================================
    print("\n--- PHASE 2: Velocity Prediction ---")

    X_v = X_train[combined_valid]
    y_v = true_velocities[combined_valid]
    pids_v = player_ids[combined_valid]

    oof_vel, vel_models = train_velocity_models(X_v, y_v, pids_v)

    for vi, vn in enumerate(['vx', 'vy', 'vz']):
        r, _ = pearsonr(y_v[:, vi], oof_vel[:, vi])
        rmse = np.sqrt(np.mean((y_v[:, vi] - oof_vel[:, vi])**2))
        print(f"  {vn}: R={r:.4f}, RMSE={rmse:.4f} m/s")

    # Clip OOF velocities
    oof_vel_clipped = clip_velocity(oof_vel)

    # Compute scaled targets for training (using competition scalers)
    scaled_targets = np.zeros((n_train, 3))
    for ti, tname in enumerate(['angle', 'depth', 'left_right']):
        scaled_targets[:, ti] = scale_predictions(raw_targets[:, ti], tname)

    # ============================================================
    # Phase 3: APPROACH A - Direct target prediction from hand features
    # ============================================================
    print("\n--- PHASE 3A: Direct Target Prediction (hand features only) ---")

    X_direct = X_train[combined_valid]
    y_direct = scaled_targets[combined_valid]
    pids_direct = player_ids[combined_valid]

    oof_direct, direct_models = train_direct_target_models(
        X_direct, y_direct, pids_direct)

    total_mse_a = 0
    for ti, tname in enumerate(['angle', 'depth', 'left_right']):
        mse = np.mean((oof_direct[:, ti] - y_direct[:, ti])**2)
        r, _ = pearsonr(oof_direct[:, ti], y_direct[:, ti])
        total_mse_a += mse
        print(f"  {tname:>12}: MSE={mse:.6f}, R={r:.4f}")
    print(f"  {'MEAN':>12}: {total_mse_a/3:.6f}")

    # ============================================================
    # Phase 3B: Direct prediction with velocity-augmented features
    # ============================================================
    print("\n--- PHASE 3B: Direct Target Prediction (hand + OOF velocity) ---")

    oof_direct_aug, direct_aug_models = train_direct_target_models(
        X_direct, y_direct, pids_direct, vel_features=oof_vel_clipped)

    total_mse_b = 0
    for ti, tname in enumerate(['angle', 'depth', 'left_right']):
        mse = np.mean((oof_direct_aug[:, ti] - y_direct[:, ti])**2)
        r, _ = pearsonr(oof_direct_aug[:, ti], y_direct[:, ti])
        total_mse_b += mse
        print(f"  {tname:>12}: MSE={mse:.6f}, R={r:.4f}")
    print(f"  {'MEAN':>12}: {total_mse_b/3:.6f}")

    # ============================================================
    # Phase 3C: Forward simulation with clipped velocity
    # ============================================================
    print("\n--- PHASE 3C: Forward Simulation (clipped velocity) ---")

    valid_idx = np.where(combined_valid)[0]
    oof_sim = np.zeros((combined_valid.sum(), 3))
    sim_fails = 0
    for ii, oi in enumerate(valid_idx):
        sim = forward_simulate(release_positions[oi], oof_vel_clipped[ii])
        if sim is not None:
            for ti, tname in enumerate(['angle', 'depth', 'left_right']):
                oof_sim[ii, ti] = scale_predictions(
                    np.array([sim[tname]]), tname)[0]
        else:
            sim_fails += 1
            oof_sim[ii] = y_direct[ii]  # fallback to actual

    print(f"  Sim failures: {sim_fails}/{combined_valid.sum()}")
    total_mse_c = 0
    for ti, tname in enumerate(['angle', 'depth', 'left_right']):
        mse = np.mean((oof_sim[:, ti] - y_direct[:, ti])**2)
        r, _ = pearsonr(oof_sim[:, ti], y_direct[:, ti])
        total_mse_c += mse
        print(f"  {tname:>12}: MSE={mse:.6f}, R={r:.4f}")
    print(f"  {'MEAN':>12}: {total_mse_c/3:.6f}")

    # ============================================================
    # Phase 4: Select best approach and generate test predictions
    # ============================================================
    print("\n--- PHASE 4: Test Predictions ---")

    approaches = {
        'direct': total_mse_a / 3,
        'direct_aug': total_mse_b / 3,
        'forward_sim': total_mse_c / 3,
    }
    best_approach = min(approaches, key=approaches.get)
    print(f"  Best approach: {best_approach} (MSE={approaches[best_approach]:.6f})")
    print(f"  LB best: 0.007224 (Sub 784)")

    # Generate test predictions for all approaches
    test_vel = predict_velocity(X_test, test_player_ids, vel_models)
    test_vel_clipped = clip_velocity(test_vel)

    print(f"  Test vel speed range: [{np.linalg.norm(test_vel, axis=1).min():.2f}, "
          f"{np.linalg.norm(test_vel, axis=1).max():.2f}] -> clipped: "
          f"[{np.linalg.norm(test_vel_clipped, axis=1).min():.2f}, "
          f"{np.linalg.norm(test_vel_clipped, axis=1).max():.2f}]")

    # Direct prediction
    test_direct = predict_direct_targets(X_test, test_player_ids, direct_models)
    test_direct_aug = predict_direct_targets(
        X_test, test_player_ids, direct_aug_models, vel_features=test_vel_clipped)

    # Forward sim prediction
    test_sim = np.zeros((n_test, 3))
    test_sim_fails = 0
    for i in range(n_test):
        sim = forward_simulate(test_release_positions[i], test_vel_clipped[i])
        if sim is not None:
            for ti, tname in enumerate(['angle', 'depth', 'left_right']):
                test_sim[i, ti] = scale_predictions(np.array([sim[tname]]), tname)[0]
        else:
            test_sim_fails += 1
            test_sim[i] = test_direct[i]  # fallback
    print(f"  Test sim failures (clipped): {test_sim_fails}/{n_test}")

    # ============================================================
    # Phase 5: Blend with Sub 784
    # ============================================================
    print("\n--- PHASE 5: Blending with Sub 784 ---")

    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Try all approaches
    for approach_name, test_preds in [
        ('direct', test_direct),
        ('direct_aug', test_direct_aug),
        ('forward_sim', test_sim),
    ]:
        print(f"\n  === {approach_name.upper()} ===")

        our = pd.DataFrame({
            'id': test_ids,
            'new_angle': test_preds[:, 0],
            'new_depth': test_preds[:, 1],
            'new_lr': test_preds[:, 2],
        })
        merged = sub784.merge(our, on='id')

        for col, nc in [('scaled_angle', 'new_angle'), ('scaled_depth', 'new_depth'),
                         ('scaled_left_right', 'new_lr')]:
            corr = np.corrcoef(merged[col], merged[nc])[0, 1]
            print(f"    {col} correlation with Sub 784: r={corr:.4f}")

        best_configs = []
        for aw in np.arange(0, 0.21, 0.05):
            for dw in np.arange(0, 0.31, 0.05):
                for lw in np.arange(0, 0.31, 0.05):
                    ba = (1-aw)*merged['scaled_angle'] + aw*merged['new_angle']
                    bd = (1-dw)*merged['scaled_depth'] + dw*merged['new_depth']
                    bl = (1-lw)*merged['scaled_left_right'] + lw*merged['new_lr']
                    a_std = ba.std()
                    d_mean = bd.mean()
                    if a_std < 0.15 and 0.49 < d_mean < 0.52:
                        da = np.mean((ba - merged['scaled_angle'])**2)
                        dd = np.mean((bd - merged['scaled_depth'])**2)
                        dl = np.mean((bl - merged['scaled_left_right'])**2)
                        best_configs.append({
                            'aw': aw, 'dw': dw, 'lw': lw,
                            'angle_std': a_std, 'depth_mean': d_mean,
                            'diversity': da + dd + dl,
                            'ba': ba.values, 'bd': bd.values, 'bl': bl.values,
                        })

        best_configs.sort(key=lambda x: -x['diversity'])
        print(f"    Valid blend configs: {len(best_configs)}")

        if best_configs:
            print(f"    {'aw':>4} {'dw':>4} {'lw':>4} | {'a_std':>8} {'d_mean':>8} | diversity")
            for c in best_configs[:5]:
                print(f"    {c['aw']:>4.2f} {c['dw']:>4.2f} {c['lw']:>4.2f} | "
                      f"{c['angle_std']:>8.4f} {c['depth_mean']:>8.4f} | {c['diversity']:.8f}")

            # Save top blended submission for this approach
            for c in best_configs[:2]:
                sub_num = get_next_submission_number()
                sub = pd.DataFrame({
                    'id': merged['id'],
                    'scaled_angle': c['ba'],
                    'scaled_depth': c['bd'],
                    'scaled_left_right': c['bl'],
                })
                filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
                sub.to_csv(filepath, index=False)
                print(f"    Sub {sub_num} ({approach_name}): "
                      f"aw={c['aw']:.2f} dw={c['dw']:.2f} lw={c['lw']:.2f}")

    # Save standalone best approach
    approach_preds = {
        'direct': test_direct,
        'direct_aug': test_direct_aug,
        'forward_sim': test_sim,
    }
    best_preds = approach_preds[best_approach]
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': best_preds[:, 0],
        'scaled_depth': best_preds[:, 1],
        'scaled_left_right': best_preds[:, 2],
    })
    filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(filepath, index=False)
    print(f"\n  Standalone ({best_approach}): {filepath}")
    print(f"    angle_std={sub['scaled_angle'].std():.4f}, "
          f"depth_mean={sub['scaled_depth'].mean():.4f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
