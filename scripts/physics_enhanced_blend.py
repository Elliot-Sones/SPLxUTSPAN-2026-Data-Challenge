"""
Physics-Enhanced Target-Specific Blend.

Builds on target_specific_blend.py (which created Sub 784) with physics insights:
1. Target-specific feature extraction frames:
   - angle: frame 153 (optimal from frame analysis)
   - depth: frame 150 + release_frame as feature
   - left_right: frame 165 (16.5% improvement over 153)
2. Physics-detected release frame as a feature (r=0.45-0.75 per-player for depth)
3. PLS depth model with release_frame augmentation

Key finding: the follow-through phase (frames 140-180) is MORE informative than
the actual release frame (~120). Different targets are best predicted from different
follow-through phases. The release frame TIMING itself is a strong depth predictor.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
FEET_TO_METERS = 0.3048
DT = 1.0 / 60.0

# Target-specific optimal extraction frames (from frame analysis)
TARGET_FRAMES = {
    'angle': 153,
    'depth': 150,
    'left_right': 165,
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


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def detect_release_frame(ts_3d, kp_index):
    """Detect per-shot release frame via peak ball speed before wrist peak."""
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 120.0

    wrist_traj = ts_3d[:, rw_idx, :].copy()
    for ax in range(3):
        vals = wrist_traj[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 120.0
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, ax] = vals

    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = []
    for key in ft_keys:
        idx = kp_index.get(key)
        if idx is not None:
            traj = ts_3d[:, idx, :].copy()
            ft_trajs.append(traj)

    if len(ft_trajs) > 0:
        ft_center = np.nanmean(ft_trajs, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()

    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    ball_m = ball * FEET_TO_METERS
    vel = np.zeros_like(ball_m)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball_m[:, ax], 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)

    wrist_z_smooth = safe_savgol(wrist_traj[:, 2], 11, 3)
    search_start = 80
    search_end = 200
    wrist_peak = search_start + np.argmax(wrist_z_smooth[search_start:search_end])

    release_end = min(wrist_peak + 5, search_end)
    release_start = max(search_start, wrist_peak - 40)
    search_speeds = speed[release_start:release_end]

    if len(search_speeds) > 0:
        release_frame = release_start + np.argmax(search_speeds)
    else:
        release_frame = max(search_start, wrist_peak - 10)

    return float(np.clip(release_frame, 80, 200))


def load_raw_data():
    """Load train and test, compute release frames."""
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
    kp_index = {name: i for i, name in enumerate(keypoint_names)}

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Keypoints: {len(keypoint_names)}")

    def process(df, is_train=True):
        n = len(df)
        X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)
        X_3d = np.zeros((n, 240, len(keypoint_names), 3), dtype=np.float32)
        release_frames = np.zeros(n, dtype=np.float32)

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

            # Physics-detected release frame
            release_frames[idx] = detect_release_frame(X_3d[idx], kp_index)

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
            'release_frames': release_frames,
        }
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    train = process(train_df, True)
    test = process(test_df, False)

    print(f"  Release frame: train mean={train['release_frames'].mean():.1f}, "
          f"test mean={test['release_frames'].mean():.1f}")

    return train, test


# ============================================================
# PLS DEPTH MODEL (with release_frame augmentation)
# ============================================================

def train_pls_depth(train_data):
    """PLS depth model with release_frame as an extra feature."""
    X_raw = train_data['X_raw']
    y = train_data['y']
    pids = train_data['pids']
    release_frames = train_data['release_frames']

    unique_pids = sorted(np.unique(pids))
    depth_idx = 1
    oof_depth = np.zeros(len(y))
    models = {}
    scalers = {}

    print("\n--- PLS DEPTH MODEL (with release_frame) ---")

    for pid in unique_pids:
        mask = pids == pid
        X_p = X_raw[mask]
        y_depth = y[mask, depth_idx]
        rf_p = release_frames[mask].reshape(-1, 1)
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        scalers[pid] = scaler

        # Append release_frame (standardized per-player)
        rf_mean = rf_p.mean()
        rf_std = rf_p.std() + 1e-6
        rf_scaled = (rf_p - rf_mean) / rf_std

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

                X_tr_pls = np.hstack([pls.transform(X_scaled[tr_idx]), rf_scaled[tr_idx]])
                X_val_pls = np.hstack([pls.transform(X_scaled[val_idx]), rf_scaled[val_idx]])

                pls_pred = pls.predict(X_scaled[val_idx]).flatten()

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

        print(f"  Player {pid}: best PLS comp={best_n}, CV MSE={best_mse:.4f}")

        # OOF predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            pls = PLSRegression(n_components=best_n)
            pls.fit(X_scaled[tr_idx], y_depth[tr_idx])
            pls_pred = pls.predict(X_scaled[val_idx]).flatten()

            X_tr_pls = np.hstack([pls.transform(X_scaled[tr_idx]), rf_scaled[tr_idx]])
            X_val_pls = np.hstack([pls.transform(X_scaled[val_idx]), rf_scaled[val_idx]])

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

        # Final models
        pls_final = PLSRegression(n_components=best_n)
        pls_final.fit(X_scaled, y_depth)
        X_pls_all = np.hstack([pls_final.transform(X_scaled), rf_scaled])
        ridge_final = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge_final.fit(X_pls_all, y_depth)
        lgb_final = lgb.LGBMRegressor(
            n_estimators=50, num_leaves=8, learning_rate=0.05,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
        lgb_final.fit(X_pls_all, y_depth)
        models[pid] = {
            'pls': pls_final, 'ridge': ridge_final, 'lgb': lgb_final,
            'rf_mean': rf_mean, 'rf_std': rf_std
        }

    mse = np.mean((oof_depth - y[:, depth_idx]) ** 2)
    print(f"  Overall depth CV MSE: {mse:.6f}")
    return oof_depth, models, scalers


def predict_pls_depth(test_data, models, scalers):
    X_raw = test_data['X_raw']
    pids = test_data['pids']
    release_frames = test_data['release_frames']
    preds = np.zeros(len(X_raw))

    for i, (x, pid, rf) in enumerate(zip(X_raw, pids, release_frames)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        model = models[pid]
        rf_scaled = np.array([[(rf - model['rf_mean']) / model['rf_std']]])

        pls = model['pls']
        pls_pred = pls.predict(x_scaled).flatten()[0]
        x_pls = np.hstack([pls.transform(x_scaled), rf_scaled])
        ridge_pred = model['ridge'].predict(x_pls)[0]
        lgb_pred = model['lgb'].predict(x_pls)[0]
        preds[i] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred

    return preds


# ============================================================
# HOOP-RELATIVE MODEL (target-specific frames + release_frame)
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
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    return R, player_pos


def extract_hoop_relative_features(ts_3d, keypoint_names, pid, release_frame,
                                    extraction_frame=153):
    """Extract hoop-relative features at a specified frame."""
    feats = {}
    feats['participant_id'] = pid
    feats['release_frame'] = release_frame
    kp_index = {name: i for i, name in enumerate(keypoint_names)}

    ef = extraction_frame  # target-specific frame

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
            feats[f"{prefix}_release_mean"] = np.nanmean(s[max(0,ef-20):min(240,ef+20)])
            vel = np.gradient(s, DT)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)
            feats[f"{prefix}_vel_at_release"] = vel[ef] if ef < len(vel) else 0.0

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
            feats[f"f{ef}_{prefix}"] = s[ef]

    # Body alignment at extraction frame
    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    rs, ls = kp_index.get('right_shoulder'), kp_index.get('left_shoulder')
    rw, lw = kp_index.get('right_wrist'), kp_index.get('left_wrist')

    if rh is not None and lh is not None:
        hip_lat = ts_hoop[:, rh, 1] - ts_hoop[:, lh, 1]
        feats['hr_hip_alignment_mean'] = np.nanmean(hip_lat)
        feats['hr_hip_alignment_release'] = hip_lat[ef]
    if rs is not None and ls is not None:
        shoulder_lat = ts_hoop[:, rs, 1] - ts_hoop[:, ls, 1]
        feats['hr_shoulder_alignment_mean'] = np.nanmean(shoulder_lat)
        feats['hr_shoulder_alignment_release'] = shoulder_lat[ef]
    if rw is not None and lw is not None:
        guide_lat = ts_hoop[:, lw, 1] - ts_hoop[:, rw, 1]
        feats['hr_guide_hand_lateral_release'] = guide_lat[ef]
    if rw is not None and rs is not None:
        arm_lat = ts_hoop[:, rw, 1] - ts_hoop[:, rs, 1]
        feats['hr_arm_lateral_dev_release'] = arm_lat[ef]

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
            feats[f"{name}_angle_release"] = angle[ef]
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


def train_hoop_relative_model(train_data, target_idx):
    """Train hoop-relative model for a specific target with optimal frame."""
    X_3d = train_data['X_3d']
    y = train_data['y']
    pids = train_data['pids']
    kp_names = train_data['keypoint_names']
    release_frames = train_data['release_frames']
    target_name = TARGETS[target_idx]
    unique_pids = sorted(np.unique(pids))
    ef = TARGET_FRAMES[target_name]

    print(f"\n--- HOOP-RELATIVE {target_name.upper()} MODEL (frame={ef}) ---")

    all_feats = [extract_hoop_relative_features(
        X_3d[i], kp_names, pids[i], release_frames[i], extraction_frame=ef)
        for i in range(len(X_3d))]
    feat_names = sorted(all_feats[0].keys())
    X_feat = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats], dtype=np.float32)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Features: {X_feat.shape[1]}")

    oof_preds = np.zeros(len(y))
    models = {}
    scalers = {}

    for pid in unique_pids:
        mask = pids == pid
        X_p, y_t = X_feat[mask], y[mask, target_idx]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)
        scalers[pid] = scaler

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr = y_t[tr_idx]
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
                m = cls(**params); m.fit(X_tr, y_tr)
                preds.append(m.predict(X_val))
            fold_preds[val_idx] = 0.3*preds[0] + 0.3*preds[1] + 0.3*preds[2] + 0.1*preds[3]

        oof_preds[global_idx] = fold_preds
        print(f"  Player {pid}: CV MSE = {np.mean((fold_preds - y_t)**2):.4f}")

        for name, cls, params in [
            ('lgb', lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
            ('cat', CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False)),
            ('ridge', Ridge, dict(alpha=1.0)),
        ]:
            m = cls(**params); m.fit(X_scaled, y_t)
            models[(pid, name)] = m

    mse = np.mean((oof_preds - y[:, target_idx])**2)
    print(f"  Overall {target_name} CV MSE: {mse:.6f}")
    return oof_preds, models, scalers, feat_names


def predict_hoop_relative(test_data, models, scalers, feat_names, target_name):
    X_3d = test_data['X_3d']
    pids = test_data['pids']
    kp_names = test_data['keypoint_names']
    release_frames = test_data['release_frames']
    ef = TARGET_FRAMES[target_name]

    all_feats = [extract_hoop_relative_features(
        X_3d[i], kp_names, pids[i], release_frames[i], extraction_frame=ef)
        for i in range(len(X_3d))]
    X_feat = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats], dtype=np.float32)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    preds = np.zeros(len(X_feat))
    for i, (x, pid) in enumerate(zip(X_feat, pids)):
        x_scaled = np.nan_to_num(scalers[pid].transform(x.reshape(1, -1)), nan=0.0)
        p = [models[(pid, n)].predict(x_scaled)[0] for n in ['lgb', 'xgb', 'cat', 'ridge']]
        preds[i] = 0.3*p[0] + 0.3*p[1] + 0.3*p[2] + 0.1*p[3]
    return preds


# ============================================================
# BLENDING
# ============================================================

def scale_predictions(raw_preds, target):
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


def blend_and_save(test_ids, test_preds_raw, oof_preds, y_train):
    """Blend with Sub 771 and Sub 784, save best submissions."""
    sub771 = pd.read_csv(SUBMISSION_DIR / "submission_771.csv")
    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Scale our predictions
    scaled_preds = {}
    for target in TARGETS:
        scaled_preds[target] = scale_predictions(test_preds_raw[target], target)

    our = pd.DataFrame({'id': test_ids})
    for target in TARGETS:
        our[f'new_{target}'] = scaled_preds[target]

    merged_771 = sub771.merge(our, on='id')
    merged_784 = sub784.merge(our, on='id')

    print("\n" + "=" * 70)
    print("BLENDING ANALYSIS")
    print("=" * 70)

    print("\n  Correlation with Sub 771:")
    for target in TARGETS:
        r = np.corrcoef(merged_771[f'scaled_{target}'], merged_771[f'new_{target}'])[0, 1]
        print(f"    {target}: r={r:.4f}")

    print("\n  Correlation with Sub 784:")
    for target in TARGETS:
        r = np.corrcoef(merged_784[f'scaled_{target}'], merged_784[f'new_{target}'])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # Blend with Sub 771 (like original)
    print("\n  Grid search - blend with Sub 771:")
    results_771 = []
    for aw in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for dw in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
            for lw in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
                ba = (1-aw)*merged_771['scaled_angle'] + aw*merged_771['new_angle']
                bd = (1-dw)*merged_771['scaled_depth'] + dw*merged_771['new_depth']
                bl = (1-lw)*merged_771['scaled_left_right'] + lw*merged_771['new_left_right']
                a_std = ba.std()
                d_mean = bd.mean()
                if a_std < 0.15 and 0.49 < d_mean < 0.52:
                    da = np.mean((ba.values - merged_771['scaled_angle'].values)**2)
                    dd = np.mean((bd.values - merged_771['scaled_depth'].values)**2)
                    dl = np.mean((bl.values - merged_771['scaled_left_right'].values)**2)
                    results_771.append({
                        'aw': aw, 'dw': dw, 'lw': lw,
                        'angle_std': a_std, 'depth_mean': d_mean,
                        'ba': ba.values, 'bd': bd.values, 'bl': bl.values,
                        'diversity': da + dd + dl, 'base': '771'
                    })

    results_771.sort(key=lambda x: -x['diversity'])
    print(f"  Valid configs (771): {len(results_771)}")
    for r in results_771[:5]:
        print(f"    aw={r['aw']:.2f} dw={r['dw']:.2f} lw={r['lw']:.2f} "
              f"a_std={r['angle_std']:.4f} d_mean={r['depth_mean']:.4f} div={r['diversity']:.6f}")

    # Blend with Sub 784
    print("\n  Grid search - blend with Sub 784:")
    results_784 = []
    for aw in [0.0, 0.05, 0.10]:
        for dw in [0.0, 0.05, 0.10, 0.15, 0.20]:
            for lw in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
                ba = (1-aw)*merged_784['scaled_angle'] + aw*merged_784['new_angle']
                bd = (1-dw)*merged_784['scaled_depth'] + dw*merged_784['new_depth']
                bl = (1-lw)*merged_784['scaled_left_right'] + lw*merged_784['new_left_right']
                a_std = ba.std()
                d_mean = bd.mean()
                if a_std < 0.15 and 0.49 < d_mean < 0.52:
                    da = np.mean((ba.values - merged_784['scaled_angle'].values)**2)
                    dd = np.mean((bd.values - merged_784['scaled_depth'].values)**2)
                    dl = np.mean((bl.values - merged_784['scaled_left_right'].values)**2)
                    results_784.append({
                        'aw': aw, 'dw': dw, 'lw': lw,
                        'angle_std': a_std, 'depth_mean': d_mean,
                        'ba': ba.values, 'bd': bd.values, 'bl': bl.values,
                        'diversity': da + dd + dl, 'base': '784'
                    })

    results_784.sort(key=lambda x: -x['diversity'])
    print(f"  Valid configs (784): {len(results_784)}")
    for r in results_784[:5]:
        print(f"    aw={r['aw']:.2f} dw={r['dw']:.2f} lw={r['lw']:.2f} "
              f"a_std={r['angle_std']:.4f} d_mean={r['depth_mean']:.4f} div={r['diversity']:.6f}")

    # Save top submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    # Top 5 from Sub 771 blend
    for r in results_771[:5]:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': merged_771['id'],
            'scaled_angle': r['ba'],
            'scaled_depth': r['bd'],
            'scaled_left_right': r['bl'],
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(filepath, index=False)
        print(f"  Sub {sub_num} (771 blend): aw={r['aw']:.2f} dw={r['dw']:.2f} lw={r['lw']:.2f} "
              f"a_std={r['angle_std']:.4f} d_mean={r['depth_mean']:.4f}")

    # Top 5 from Sub 784 blend
    for r in results_784[:5]:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': merged_784['id'],
            'scaled_angle': r['ba'],
            'scaled_depth': r['bd'],
            'scaled_left_right': r['bl'],
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(filepath, index=False)
        print(f"  Sub {sub_num} (784 blend): aw={r['aw']:.2f} dw={r['dw']:.2f} lw={r['lw']:.2f} "
              f"a_std={r['angle_std']:.4f} d_mean={r['depth_mean']:.4f}")

    # Also save our standalone prediction
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds['angle'],
        'scaled_depth': scaled_preds['depth'],
        'scaled_left_right': scaled_preds['left_right'],
    })
    filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(filepath, index=False)
    print(f"  Sub {sub_num} (standalone): a_std={sub['scaled_angle'].std():.4f} "
          f"d_mean={sub['scaled_depth'].mean():.4f}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("PHYSICS-ENHANCED TARGET-SPECIFIC BLEND")
    print("=" * 70)

    train_data, test_data = load_raw_data()

    # Train models
    # 1. PLS depth with release_frame
    oof_depth, pls_models, pls_scalers = train_pls_depth(train_data)

    # 2. HR left_right at frame 165
    oof_lr, hr_lr_models, hr_lr_scalers, hr_lr_feats = \
        train_hoop_relative_model(train_data, target_idx=2)

    # 3. HR angle at frame 153
    oof_angle, hr_angle_models, hr_angle_scalers, hr_angle_feats = \
        train_hoop_relative_model(train_data, target_idx=0)

    # Summary
    y = train_data['y']
    print("\n" + "=" * 70)
    print("CV SUMMARY")
    print("=" * 70)
    for name, oof, idx in [('angle', oof_angle, 0), ('depth', oof_depth, 1), ('left_right', oof_lr, 2)]:
        mse = np.mean((oof - y[:, idx])**2)
        print(f"  {name:12s}: MSE = {mse:.6f}")
    mean_mse = np.mean([np.mean((oof_angle - y[:, 0])**2),
                         np.mean((oof_depth - y[:, 1])**2),
                         np.mean((oof_lr - y[:, 2])**2)])
    print(f"  MEAN MSE   = {mean_mse:.6f}")

    # Test predictions
    print("\nGenerating test predictions...")
    test_preds = {
        'depth': predict_pls_depth(test_data, pls_models, pls_scalers),
        'left_right': predict_hoop_relative(test_data, hr_lr_models, hr_lr_scalers,
                                             hr_lr_feats, 'left_right'),
        'angle': predict_hoop_relative(test_data, hr_angle_models, hr_angle_scalers,
                                        hr_angle_feats, 'angle'),
    }

    # Blend and save
    blend_and_save(test_data['ids'], test_preds,
                   {'angle': oof_angle, 'depth': oof_depth, 'left_right': oof_lr},
                   y)

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
