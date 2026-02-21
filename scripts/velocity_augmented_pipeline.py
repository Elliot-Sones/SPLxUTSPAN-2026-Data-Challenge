"""
Velocity-Augmented Pipeline

Key insight: Use PLS on the full 49K raw timeseries to predict velocity
components (vx, vy, vz) derived from inverse projectile. Then add these
as extra features to the existing hoop-relative pipeline.

This avoids error amplification from forward simulation while adding
orthogonal physics-informed signal to the proven tree ensemble.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

sys.path.insert(0, str(PROJECT_DIR))
from physics_engine.core.data_loader import DataLoader, HOOP_POSITION, FEET_TO_METERS

G = 9.81
HOOP_X_FT, HOOP_Y_FT, HOOP_Z_FT = 5.25, -25.0, 10.0
TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def compute_true_velocity(release_pos_ft, angle_deg, depth_in, lr_in):
    """Compute exact release velocity from targets + release position."""
    dx = (HOOP_X_FT + lr_in/12.0 - release_pos_ft[0]) * FEET_TO_METERS
    dy = (HOOP_Y_FT + depth_in/12.0 - release_pos_ft[1]) * FEET_TO_METERS
    dz = (HOOP_Z_FT - release_pos_ft[2]) * FEET_TO_METERS
    D = np.sqrt(dx**2 + dy**2)
    angle_rad = np.radians(angle_deg)
    t_sq = 2.0 * (dz + D * np.tan(angle_rad)) / G
    if t_sq <= 0:
        return None
    t = np.sqrt(t_sq)
    return np.array([dx/t, dy/t, (dz + 0.5*G*t**2)/t])


def estimate_release_position(timeseries_3d, keypoint_names):
    """Estimate ball position at release frame from 3D timeseries."""
    kp_idx = {name: i for i, name in enumerate(keypoint_names)}
    rw_idx = kp_idx.get('right_wrist')
    if rw_idx is None:
        return None, 0

    # Find release frame from wrist peak height
    wrist_z = timeseries_3d[:, rw_idx, 2]
    wrist_z = np.nan_to_num(wrist_z, nan=0.0)
    rf = int(np.argmax(wrist_z))

    # Ball position = wrist + 0.6 * (fingertip_center - wrist)
    ft_names = []
    for prefix in ['right_second_finger_distal', 'right_third_finger_distal',
                    'right_fourth_finger_distal']:
        # Find the keypoint name (might have different casing in parsed list)
        for name in keypoint_names:
            if name.lower() == prefix.lower() or name == prefix:
                ft_names.append(name)
                break

    if not ft_names:
        # Try with the original names from KEYPOINT_ORDER
        for fn in ['second', 'third', 'fourth']:
            key = f'right_{fn}_finger_distal'
            if key in kp_idx:
                ft_names.append(key)

    wrist = timeseries_3d[rf, rw_idx]
    if np.any(np.isnan(wrist)):
        return None, rf

    ft_positions = []
    for name in ft_names:
        idx = kp_idx.get(name)
        if idx is not None:
            pos = timeseries_3d[rf, idx]
            if not np.any(np.isnan(pos)):
                ft_positions.append(pos)

    if len(ft_positions) == 0:
        return wrist, rf

    ft_center = np.mean(ft_positions, axis=0)
    ball_pos = wrist + 0.6 * (ft_center - wrist)
    return ball_pos, rf


def compute_hoop_relative_transform(player_pos):
    """Compute rotation matrix from court to hoop-relative frame."""
    hoop_2d = HOOP_POSITION[:2]
    player_2d = player_pos[:2]
    forward = hoop_2d - player_2d
    fn = np.linalg.norm(forward)
    if fn < 1e-6:
        forward = np.array([0.0, -1.0])
    else:
        forward = forward / fn
    lateral = np.array([-forward[1], forward[0]])
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    return R, player_pos


def extract_hoop_features(ts_3d, keypoint_names, pid, vel_feats=None):
    """Extract hoop-relative features + optional velocity features."""
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
            vel = np.gradient(s, 1.0/60.0)
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
            vel = np.gradient(s, 1.0/60.0)
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
    for j1, j2, j3, name in [('right_shoulder', 'right_elbow', 'right_wrist', 'elbow')]:
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
            if joint not in kp_index:
                continue
            idx2 = kp_index[joint]
            for c in range(3):
                vel = np.gradient(ts_3d[s:e, idx2, c], 1.0/60.0)
                feats[f"phase_{pname}_{joint}_{'xyz'[c]}_vel_max"] = np.nanmax(vel)

    # Add velocity features if provided
    if vel_feats is not None:
        for k, v in vel_feats.items():
            feats[k] = v

    return feats


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


def main():
    t0 = time.time()
    print("=" * 70)
    print("VELOCITY-AUGMENTED PIPELINE")
    print("=" * 70)

    # ============================================================
    # Load data
    # ============================================================
    print("\nLoading data...")
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

    def process_data(df, is_train=True):
        n = len(df)
        X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)
        X_3d = np.zeros((n, 240, len(keypoint_names), 3), dtype=np.float32)
        ids, pids, targets_list = [], [], []

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
                targets_list.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        result = {'X_raw': X_raw, 'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'keypoint_names': keypoint_names}
        if is_train:
            result['y'] = np.array(targets_list, dtype=np.float32)
        return result

    train = process_data(train_df, True)
    test = process_data(test_df, False)

    # ============================================================
    # Compute true velocities for training shots
    # ============================================================
    print("\n--- PHASE 1: Compute True Velocities ---")

    n_train = len(train_df)
    true_vel = np.zeros((n_train, 3), dtype=np.float32)
    vel_valid = np.zeros(n_train, dtype=bool)

    for i in range(n_train):
        pos, rf = estimate_release_position(train['X_3d'][i], keypoint_names)
        if pos is not None:
            v = compute_true_velocity(pos, train['y'][i, 0], train['y'][i, 1], train['y'][i, 2])
            if v is not None:
                true_vel[i] = v
                vel_valid[i] = True

    n_valid = vel_valid.sum()
    speeds = np.linalg.norm(true_vel[vel_valid], axis=1)
    print(f"  Valid: {n_valid}/{n_train}")
    print(f"  Speed: mean={speeds.mean():.2f}, std={speeds.std():.2f} m/s")

    # For invalid shots, use player-specific mean velocity
    for pid in sorted(np.unique(train['pids'])):
        mask_pid = train['pids'] == pid
        mask_valid = vel_valid & mask_pid
        if mask_valid.sum() > 0:
            mean_vel = true_vel[mask_valid].mean(axis=0)
            mask_invalid = ~vel_valid & mask_pid
            true_vel[mask_invalid] = mean_vel
            vel_valid[mask_invalid] = True

    print(f"  After imputation: {vel_valid.sum()}/{n_train} valid")

    # ============================================================
    # PLS velocity prediction from raw timeseries
    # ============================================================
    print("\n--- PHASE 2: PLS Velocity Prediction ---")

    unique_pids = sorted(np.unique(train['pids']))
    oof_vel = np.zeros((n_train, 3), dtype=np.float32)
    vel_models = {}

    for vi, vname in enumerate(['vx', 'vy', 'vz']):
        print(f"\n  Predicting {vname}...")

        for pid in unique_pids:
            mask = train['pids'] == pid
            X_p = train['X_raw'][mask]
            y_p = true_vel[mask, vi]
            n_p = len(X_p)
            global_idx = np.where(mask)[0]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_p)

            # Find optimal PLS components
            candidates = [3, 5, 8, 10, 15, 20]
            max_comp = min(20, n_p - n_p // 5 - 1)
            candidates = [c for c in candidates if c <= max_comp]

            best_n, best_mse = 5, float('inf')
            for nc in candidates:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                mses = []
                for tr_idx, val_idx in kf.split(X_scaled):
                    pls = PLSRegression(n_components=nc)
                    pls.fit(X_scaled[tr_idx], y_p[tr_idx])
                    pred = pls.predict(X_scaled[val_idx]).flatten()
                    mses.append(np.mean((pred - y_p[val_idx])**2))
                avg = np.mean(mses)
                if avg < best_mse:
                    best_mse = avg
                    best_n = nc

            # Get OOF predictions with best n_components
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n_p)
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
                # PLS
                pls = PLSRegression(n_components=best_n)
                pls.fit(X_scaled[tr_idx], y_p[tr_idx])
                pls_pred = pls.predict(X_scaled[val_idx]).flatten()

                # Ridge on PLS components
                X_tr_pls = pls.transform(X_scaled[tr_idx])
                X_val_pls = pls.transform(X_scaled[val_idx])
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_pls, y_p[tr_idx])
                ridge_pred = ridge.predict(X_val_pls)

                # LGB on PLS components
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=50, num_leaves=8, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(X_tr_pls, y_p[tr_idx])
                lgb_pred = lgb_m.predict(X_val_pls)

                fold_preds[val_idx] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred

            oof_vel[global_idx, vi] = fold_preds

            # Final models for test prediction
            X_all_scaled = scaler.fit_transform(X_p)
            pls_final = PLSRegression(n_components=best_n)
            pls_final.fit(X_all_scaled, y_p)
            X_pls_all = pls_final.transform(X_all_scaled)
            ridge_final = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            ridge_final.fit(X_pls_all, y_p)
            lgb_final = lgb.LGBMRegressor(
                n_estimators=50, num_leaves=8, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
            lgb_final.fit(X_pls_all, y_p)

            vel_models[(pid, vname)] = {
                'scaler': scaler, 'pls': pls_final,
                'ridge': ridge_final, 'lgb': lgb_final, 'best_n': best_n
            }

            r, _ = pearsonr(y_p, fold_preds)
            print(f"    Player {pid}: n_comp={best_n}, R={r:.4f}")

        # Overall correlation
        r, _ = pearsonr(true_vel[:, vi], oof_vel[:, vi])
        rmse = np.sqrt(np.mean((true_vel[:, vi] - oof_vel[:, vi])**2))
        print(f"  {vname} overall: R={r:.4f}, RMSE={rmse:.4f}")

    # Test velocity predictions
    print("\n  Predicting test velocities...")
    test_vel = np.zeros((len(test_df), 3), dtype=np.float32)
    for vi, vname in enumerate(['vx', 'vy', 'vz']):
        for i in range(len(test_df)):
            pid = test['pids'][i]
            m = vel_models[(pid, vname)]
            x_s = m['scaler'].transform(test['X_raw'][i:i+1])
            pls_pred = m['pls'].predict(x_s).flatten()[0]
            x_pls = m['pls'].transform(x_s)
            ridge_pred = m['ridge'].predict(x_pls)[0]
            lgb_pred = m['lgb'].predict(x_pls)[0]
            test_vel[i, vi] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred

    test_speeds = np.linalg.norm(test_vel, axis=1)
    print(f"  Test speed: mean={test_speeds.mean():.2f}, "
          f"range=[{test_speeds.min():.2f}, {test_speeds.max():.2f}]")

    # ============================================================
    # Phase 3: Train target models WITH velocity features
    # ============================================================
    print("\n--- PHASE 3: Velocity-Augmented Target Models ---")

    # Extract hoop-relative features for all shots
    print("  Extracting hoop-relative features (train)...")
    train_feats_base = []
    for i in range(n_train):
        f = extract_hoop_features(train['X_3d'][i], keypoint_names, train['pids'][i])
        train_feats_base.append(f)

    feat_names_base = sorted([k for k in train_feats_base[0].keys() if k != 'participant_id'])
    print(f"  Base features: {len(feat_names_base)}")

    # Create feature matrices: base only and base + velocity
    X_base = np.array([[f.get(n, 0.0) for n in feat_names_base] for f in train_feats_base],
                       dtype=np.float32)
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)

    vel_feat_names = ['oof_vx', 'oof_vy', 'oof_vz', 'oof_speed',
                       'oof_launch_angle', 'oof_horiz_speed']
    X_vel = np.zeros((n_train, len(vel_feat_names)), dtype=np.float32)
    X_vel[:, 0] = oof_vel[:, 0]  # vx
    X_vel[:, 1] = oof_vel[:, 1]  # vy
    X_vel[:, 2] = oof_vel[:, 2]  # vz
    X_vel[:, 3] = np.linalg.norm(oof_vel, axis=1)  # speed
    X_vel[:, 4] = np.degrees(np.arctan2(oof_vel[:, 2],
                   np.sqrt(oof_vel[:, 0]**2 + oof_vel[:, 1]**2) + 1e-8))  # launch angle
    X_vel[:, 5] = np.sqrt(oof_vel[:, 0]**2 + oof_vel[:, 1]**2)  # horiz speed

    X_aug = np.hstack([X_base, X_vel])
    aug_feat_names = feat_names_base + vel_feat_names
    print(f"  Augmented features: {len(aug_feat_names)}")

    # Scale targets
    y_raw = train['y']
    y_scaled = np.zeros_like(y_raw)
    for ti, tname in enumerate(TARGETS):
        y_scaled[:, ti] = scale_predictions(y_raw[:, ti], tname)

    # Train per-player per-target models (both base and augmented)
    for config_name, X_all in [('base', X_base), ('augmented', X_aug)]:
        print(f"\n  === {config_name.upper()} ===")

        oof_preds = np.zeros((n_train, 3))

        for target_idx, tname in enumerate(TARGETS):
            target_mse_list = []

            for pid in unique_pids:
                mask = train['pids'] == pid
                X_p = X_all[mask]
                y_t = y_scaled[mask, target_idx]
                n_p = len(X_p)
                global_idx = np.where(mask)[0]

                scaler = StandardScaler()
                X_scaled = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                fold_preds = np.zeros(n_p)

                for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
                    X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
                    y_tr = y_t[tr_idx]
                    preds = []

                    for cls, params in [
                        (lgb.LGBMRegressor, dict(
                            n_estimators=100, num_leaves=10, learning_rate=0.05,
                            reg_alpha=0.5, reg_lambda=0.5, random_state=42,
                            verbose=-1, n_jobs=-1)),
                        (xgb.XGBRegressor, dict(
                            n_estimators=100, max_depth=4, learning_rate=0.05,
                            reg_alpha=0.5, reg_lambda=0.5, random_state=42,
                            verbosity=0, n_jobs=-1)),
                        (CatBoostRegressor, dict(
                            iterations=100, depth=4, learning_rate=0.05,
                            l2_leaf_reg=3.0, random_state=42, verbose=False)),
                        (Ridge, dict(alpha=1.0)),
                    ]:
                        m = cls(**params)
                        m.fit(X_tr, y_tr)
                        preds.append(m.predict(X_val))

                    fold_preds[val_idx] = (0.3 * preds[0] + 0.3 * preds[1] +
                                           0.3 * preds[2] + 0.1 * preds[3])

                oof_preds[global_idx, target_idx] = fold_preds
                mse = np.mean((fold_preds - y_t)**2)
                target_mse_list.append(mse)

            overall_mse = np.mean((oof_preds[:, target_idx] - y_scaled[:, target_idx])**2)
            r, _ = pearsonr(oof_preds[:, target_idx], y_scaled[:, target_idx])
            print(f"    {tname:>12}: MSE={overall_mse:.6f}, R={r:.4f}")

        mean_mse = np.mean([(np.mean((oof_preds[:, ti] - y_scaled[:, ti])**2))
                            for ti in range(3)])
        print(f"    {'MEAN':>12}: {mean_mse:.6f}")

        # Store augmented results for blending
        if config_name == 'augmented':
            oof_aug = oof_preds.copy()

    # ============================================================
    # Phase 4: Generate test predictions with augmented features
    # ============================================================
    print("\n--- PHASE 4: Test Predictions ---")

    print("  Extracting hoop-relative features (test)...")
    test_feats = []
    for i in range(len(test_df)):
        f = extract_hoop_features(test['X_3d'][i], keypoint_names, test['pids'][i])
        test_feats.append(f)

    X_test_base = np.array([[f.get(n, 0.0) for n in feat_names_base] for f in test_feats],
                           dtype=np.float32)
    X_test_base = np.nan_to_num(X_test_base, nan=0.0, posinf=0.0, neginf=0.0)

    X_test_vel = np.zeros((len(test_df), len(vel_feat_names)), dtype=np.float32)
    X_test_vel[:, 0] = test_vel[:, 0]
    X_test_vel[:, 1] = test_vel[:, 1]
    X_test_vel[:, 2] = test_vel[:, 2]
    X_test_vel[:, 3] = np.linalg.norm(test_vel, axis=1)
    X_test_vel[:, 4] = np.degrees(np.arctan2(test_vel[:, 2],
                        np.sqrt(test_vel[:, 0]**2 + test_vel[:, 1]**2) + 1e-8))
    X_test_vel[:, 5] = np.sqrt(test_vel[:, 0]**2 + test_vel[:, 1]**2)

    X_test_aug = np.hstack([X_test_base, X_test_vel])

    # Train final models on all training data
    print("  Training final models...")
    test_preds = np.zeros((len(test_df), 3))

    for target_idx, tname in enumerate(TARGETS):
        for pid in unique_pids:
            mask_train = train['pids'] == pid
            mask_test = test['pids'] == pid

            X_tr = X_aug[mask_train]
            y_tr = y_scaled[mask_train, target_idx]

            scaler = StandardScaler()
            X_tr_scaled = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)

            X_te = X_test_aug[mask_test]
            X_te_scaled = np.nan_to_num(scaler.transform(X_te), nan=0.0)

            preds_list = []
            for cls, params in [
                (lgb.LGBMRegressor, dict(
                    n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42,
                    verbose=-1, n_jobs=-1)),
                (xgb.XGBRegressor, dict(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42,
                    verbosity=0, n_jobs=-1)),
                (CatBoostRegressor, dict(
                    iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)),
                (Ridge, dict(alpha=1.0)),
            ]:
                m = cls(**params)
                m.fit(X_tr_scaled, y_tr)
                preds_list.append(m.predict(X_te_scaled))

            test_idx = np.where(mask_test)[0]
            test_preds[test_idx, target_idx] = (0.3 * preds_list[0] + 0.3 * preds_list[1] +
                                                 0.3 * preds_list[2] + 0.1 * preds_list[3])

    print(f"  Test pred ranges:")
    for ti, tname in enumerate(TARGETS):
        p = test_preds[:, ti]
        print(f"    {tname}: [{p.min():.4f}, {p.max():.4f}], "
              f"mean={p.mean():.4f}, std={p.std():.4f}")

    # ============================================================
    # Phase 5: Blend with Sub 784
    # ============================================================
    print("\n--- PHASE 5: Blending with Sub 784 ---")

    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    our = pd.DataFrame({
        'id': test['ids'],
        'new_angle': test_preds[:, 0],
        'new_depth': test_preds[:, 1],
        'new_lr': test_preds[:, 2],
    })
    merged = sub784.merge(our, on='id')

    for col, nc in [('scaled_angle', 'new_angle'), ('scaled_depth', 'new_depth'),
                     ('scaled_left_right', 'new_lr')]:
        corr = np.corrcoef(merged[col], merged[nc])[0, 1]
        print(f"  {col} correlation: r={corr:.4f}")

    # Grid search
    best_configs = []
    for aw in np.arange(0, 0.26, 0.05):
        for dw in np.arange(0, 0.51, 0.05):
            for lw in np.arange(0, 0.51, 0.05):
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
    print(f"\n  Valid configs: {len(best_configs)}")
    print(f"  {'aw':>4} {'dw':>4} {'lw':>4} | {'a_std':>8} {'d_mean':>8} | diversity")
    for c in best_configs[:10]:
        print(f"  {c['aw']:>4.2f} {c['dw']:>4.2f} {c['lw']:>4.2f} | "
              f"{c['angle_std']:>8.4f} {c['depth_mean']:>8.4f} | {c['diversity']:.8f}")

    # Save submissions
    print("\n  Saving submissions...")
    for c in best_configs[:5]:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': merged['id'],
            'scaled_angle': c['ba'],
            'scaled_depth': c['bd'],
            'scaled_left_right': c['bl'],
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(filepath, index=False)
        print(f"  Sub {sub_num}: aw={c['aw']:.2f} dw={c['dw']:.2f} lw={c['lw']:.2f} "
              f"a_std={c['angle_std']:.4f} d_mean={c['depth_mean']:.4f}")

    # Also save standalone
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test['ids'],
        'scaled_angle': test_preds[:, 0],
        'scaled_depth': test_preds[:, 1],
        'scaled_left_right': test_preds[:, 2],
    })
    filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(filepath, index=False)
    print(f"\n  Standalone: {filepath}")
    print(f"    angle_std={sub['scaled_angle'].std():.4f}, "
          f"depth_mean={sub['scaled_depth'].mean():.4f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
