"""
Experiment: Hoop-Relative Coordinate Transformation

Hypothesis: Transforming all keypoints from court coordinates into a player-to-hoop
reference frame should improve depth and left_right predictions by aligning the
coordinate axes with the prediction targets.

Current coordinates: Origin at top-left of court, hoop at [5.25, -25, 10] feet.
New coordinates: Forward axis = player center -> hoop, lateral axis = perpendicular,
vertical axis = up.

This should decouple depth signal (along forward axis) from left_right signal
(along lateral axis), improving both predictions.
"""

import json
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]

# Hoop position in court coordinates (feet)
HOOP_POS = np.array([5.25, -25.0, 10.0])


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_keypoint_cols(df):
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    return [c for c in df.columns if c not in meta_cols]


def get_keypoint_names(keypoint_cols):
    """Get unique keypoint names from column list."""
    names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            names.append(col[:-2])
    return names


def parse_shot_timeseries(row, keypoint_cols):
    """Parse a single shot's timeseries data. Returns (240, n_keypoints, 3) array."""
    n_keypoints = len(keypoint_cols) // 3
    timeseries = np.zeros((240, n_keypoints, 3), dtype=np.float32)
    for i, col in enumerate(keypoint_cols):
        kp_idx = i // 3
        coord_idx = i % 3
        timeseries[:, kp_idx, coord_idx] = parse_array_string(row[col])
    return timeseries


def compute_hoop_relative_transform(player_pos):
    """
    Compute rotation matrix to transform from court coords to hoop-relative coords.

    New axes:
    - forward (x'): player -> hoop direction (horizontal plane)
    - lateral (y'): perpendicular to forward in horizontal plane (right-positive)
    - vertical (z'): up (same as original z)

    Returns: 3x3 rotation matrix, translation vector
    """
    # Forward direction in horizontal plane (x-y)
    hoop_2d = HOOP_POS[:2]
    player_2d = player_pos[:2]
    forward = hoop_2d - player_2d
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, -1.0])  # default: toward negative y
    else:
        forward = forward / forward_norm

    # Lateral direction (perpendicular in horizontal plane, right-hand rule)
    lateral = np.array([-forward[1], forward[0]])

    # Build 3x3 rotation matrix (z stays the same)
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]
    R[0, 1] = forward[1]
    R[1, 0] = lateral[0]
    R[1, 1] = lateral[1]
    # R[2, 2] = 1.0 already

    return R, player_pos


def transform_to_hoop_relative(timeseries_3d, R, origin):
    """
    Transform (240, n_keypoints, 3) timeseries to hoop-relative coordinates.
    """
    n_frames, n_kp, _ = timeseries_3d.shape
    # Center on origin (player position at first frame)
    centered = timeseries_3d - origin.reshape(1, 1, 3)
    # Rotate: apply R to each point
    # R is (3,3), points are (240, n_kp, 3) -> result is (240, n_kp, 3)
    transformed = np.einsum('ij,fkj->fki', R, centered)
    return transformed


def smooth_ts(data, window=7):
    """Apply Savitzky-Golay smoothing."""
    if window < 3 or np.any(np.isnan(data)):
        return data
    if window % 2 == 0:
        window += 1
    try:
        return savgol_filter(data, window, min(2, window - 1), axis=0)
    except Exception:
        return data


def extract_features_from_transformed(ts_orig, ts_hoop, keypoint_names, participant_id):
    """
    Extract features from both original and hoop-relative coordinate systems.

    ts_orig: (240, n_keypoints, 3) in original court coords
    ts_hoop: (240, n_keypoints, 3) in hoop-relative coords
    """
    features = {}
    features['participant_id'] = participant_id

    # Key body parts for basketball shooting
    key_joints = {
        'right_wrist': None, 'right_elbow': None, 'right_shoulder': None,
        'left_wrist': None, 'left_shoulder': None,
        'right_hip': None, 'left_hip': None, 'mid_hip': None,
        'right_knee': None, 'left_knee': None,
        'right_ankle': None, 'left_ankle': None,
        'neck': None, 'nose': None,
    }

    kp_index = {name: i for i, name in enumerate(keypoint_names)}

    # Map available keypoints
    for joint in list(key_joints.keys()):
        if joint in kp_index:
            key_joints[joint] = kp_index[joint]

    # --- HOOP-RELATIVE FEATURES ---
    # These are the novel features: positions/velocities in the player-to-hoop frame
    for joint_name, idx in key_joints.items():
        if idx is None:
            continue

        # Hoop-relative position stats
        for coord, coord_name in enumerate(['forward', 'lateral', 'vertical']):
            series = ts_hoop[:, idx, coord]
            prefix = f"hr_{joint_name}_{coord_name}"

            features[f"{prefix}_mean"] = np.nanmean(series)
            features[f"{prefix}_std"] = np.nanstd(series)
            features[f"{prefix}_min"] = np.nanmin(series)
            features[f"{prefix}_max"] = np.nanmax(series)
            features[f"{prefix}_range"] = np.nanmax(series) - np.nanmin(series)

            # Release window (frames 140-180, roughly 2.3-3s into shot)
            release_window = series[140:180]
            features[f"{prefix}_release_mean"] = np.nanmean(release_window)
            features[f"{prefix}_release_std"] = np.nanstd(release_window)

            # Velocity in hoop-relative frame
            vel = np.gradient(series, 1.0/60.0)
            features[f"{prefix}_vel_mean"] = np.nanmean(vel)
            features[f"{prefix}_vel_max"] = np.nanmax(vel)
            features[f"{prefix}_vel_at_release"] = vel[153] if len(vel) > 153 else 0.0

            # Late phase (follow-through)
            features[f"{prefix}_late_mean"] = np.nanmean(series[180:240])

    # --- SHOOTING ARM MECHANICS IN HOOP FRAME ---
    rw_idx = key_joints.get('right_wrist')
    re_idx = key_joints.get('right_elbow')
    rs_idx = key_joints.get('right_shoulder')

    if all(idx is not None for idx in [rw_idx, re_idx, rs_idx]):
        # Arm extension in forward direction (depth-relevant)
        wrist_fwd = ts_hoop[:, rw_idx, 0]
        elbow_fwd = ts_hoop[:, re_idx, 0]
        shoulder_fwd = ts_hoop[:, rs_idx, 0]

        arm_extension_fwd = wrist_fwd - shoulder_fwd
        features['hr_arm_extension_fwd_at_release'] = arm_extension_fwd[153]
        features['hr_arm_extension_fwd_max'] = np.nanmax(arm_extension_fwd[140:200])
        features['hr_arm_extension_fwd_vel'] = np.gradient(arm_extension_fwd, 1/60.0)[153]

        # Arm deviation in lateral direction (left_right-relevant)
        wrist_lat = ts_hoop[:, rw_idx, 1]
        shoulder_lat = ts_hoop[:, rs_idx, 1]

        arm_lateral_dev = wrist_lat - shoulder_lat
        features['hr_arm_lateral_dev_at_release'] = arm_lateral_dev[153]
        features['hr_arm_lateral_dev_max'] = np.nanmax(np.abs(arm_lateral_dev[140:200]))
        features['hr_arm_lateral_dev_vel'] = np.gradient(arm_lateral_dev, 1/60.0)[153]

        # Release height (vertical, angle-relevant)
        wrist_vert = ts_hoop[:, rw_idx, 2]
        features['hr_release_height'] = wrist_vert[153]
        features['hr_release_height_vel'] = np.gradient(wrist_vert, 1/60.0)[153]

    # --- BODY ALIGNMENT FEATURES ---
    lh_idx = key_joints.get('left_hip')
    rh_idx = key_joints.get('right_hip')
    ls_idx = key_joints.get('left_shoulder')

    if all(idx is not None for idx in [lh_idx, rh_idx]):
        # Hip alignment relative to hoop direction
        hip_lateral_diff = ts_hoop[:, rh_idx, 1] - ts_hoop[:, lh_idx, 1]
        features['hr_hip_alignment_mean'] = np.nanmean(hip_lateral_diff)
        features['hr_hip_alignment_at_release'] = hip_lateral_diff[153]

        hip_fwd_diff = ts_hoop[:, rh_idx, 0] - ts_hoop[:, lh_idx, 0]
        features['hr_hip_rotation_mean'] = np.nanmean(hip_fwd_diff)
        features['hr_hip_rotation_at_release'] = hip_fwd_diff[153]

    if all(idx is not None for idx in [ls_idx, rs_idx]):
        # Shoulder alignment
        shoulder_lateral_diff = ts_hoop[:, rs_idx, 1] - ts_hoop[:, ls_idx, 1]
        features['hr_shoulder_alignment_mean'] = np.nanmean(shoulder_lateral_diff)
        features['hr_shoulder_alignment_at_release'] = shoulder_lateral_diff[153]

        shoulder_fwd_diff = ts_hoop[:, rs_idx, 0] - ts_hoop[:, ls_idx, 0]
        features['hr_shoulder_rotation_mean'] = np.nanmean(shoulder_fwd_diff)
        features['hr_shoulder_rotation_at_release'] = shoulder_fwd_diff[153]

    # --- GUIDE HAND IN HOOP FRAME ---
    lw_idx = key_joints.get('left_wrist')
    if lw_idx is not None and rw_idx is not None:
        # Guide hand lateral position relative to shooting hand
        guide_lateral = ts_hoop[:, lw_idx, 1] - ts_hoop[:, rw_idx, 1]
        features['hr_guide_hand_lateral_at_release'] = guide_lateral[153]
        features['hr_guide_hand_lateral_mean'] = np.nanmean(guide_lateral[140:180])

        # Guide hand forward offset
        guide_fwd = ts_hoop[:, lw_idx, 0] - ts_hoop[:, rw_idx, 0]
        features['hr_guide_hand_fwd_at_release'] = guide_fwd[153]

    # --- ORIGINAL COORDINATE FEATURES (proven to work) ---
    # Keep the proven features from the existing pipeline
    for joint_name, idx in key_joints.items():
        if idx is None:
            continue
        for coord, coord_name in enumerate(['x', 'y', 'z']):
            series = ts_orig[:, idx, coord]
            prefix = f"{joint_name}_{coord_name}"

            features[f"{prefix}_mean"] = np.nanmean(series)
            features[f"{prefix}_std"] = np.nanstd(series)
            features[f"{prefix}_min"] = np.nanmin(series)
            features[f"{prefix}_max"] = np.nanmax(series)
            features[f"{prefix}_range"] = np.nanmax(series) - np.nanmin(series)
            features[f"{prefix}_q25"] = np.nanpercentile(series, 25)
            features[f"{prefix}_q75"] = np.nanpercentile(series, 75)

            # Velocity
            vel = np.gradient(series, 1.0/60.0)
            features[f"{prefix}_vel_mean"] = np.nanmean(vel)
            features[f"{prefix}_vel_max"] = np.nanmax(vel)
            features[f"{prefix}_vel_min"] = np.nanmin(vel)

            # Key frames
            features[f"f153_{prefix}"] = series[153]
            features[f"f102_{prefix}"] = series[102]
            features[f"f237_{prefix}"] = min(len(series)-1, 237)

    # --- PHASE-BASED FEATURES ---
    # Phases: stance (0-60), load (60-120), propulsion (120-170), follow-through (170-240)
    phases = {'stance': (0, 60), 'load': (60, 120), 'propel': (120, 170), 'follow': (170, 240)}
    for phase_name, (start, end) in phases.items():
        for joint_name in ['right_wrist', 'right_elbow', 'right_shoulder']:
            idx = key_joints.get(joint_name)
            if idx is None:
                continue
            for coord in range(3):
                series = ts_orig[:, idx, coord]
                phase_series = series[start:end]
                vel = np.gradient(phase_series, 1.0/60.0)
                prefix = f"phase_{phase_name}_{joint_name}_{'xyz'[coord]}"
                features[f"{prefix}_mean"] = np.nanmean(phase_series)
                features[f"{prefix}_vel_max"] = np.nanmax(vel)
                features[f"{prefix}_vel_mean"] = np.nanmean(vel)

    # --- JOINT ANGLES ---
    if all(key_joints.get(j) is not None for j in ['right_shoulder', 'right_elbow', 'right_wrist']):
        rs = ts_orig[:, key_joints['right_shoulder']]
        re = ts_orig[:, key_joints['right_elbow']]
        rw = ts_orig[:, key_joints['right_wrist']]
        v1 = rs - re
        v2 = rw - re
        dot = np.sum(v1 * v2, axis=1)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = n1 * n2
        denom[denom == 0] = 1e-10
        elbow_angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
        features['elbow_angle_at_release'] = elbow_angle[153]
        features['elbow_angle_max'] = np.nanmax(elbow_angle)
        features['elbow_angle_range'] = np.nanmax(elbow_angle) - np.nanmin(elbow_angle)
        features['elbow_angle_vel_at_release'] = np.gradient(elbow_angle, 1/60.0)[153]

    return features


def load_and_transform(csv_path, is_train=True):
    """Load data and apply hoop-relative coordinate transformation."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    keypoint_cols = get_keypoint_cols(df)
    keypoint_names = get_keypoint_names(keypoint_cols)

    print(f"  {len(df)} shots, {len(keypoint_names)} keypoints, {len(keypoint_cols)} columns")

    all_features = []
    all_targets = []
    all_pids = []
    all_ids = []

    for idx, row in df.iterrows():
        # Parse timeseries as (240, n_kp, 3)
        ts_3d = parse_shot_timeseries(row, keypoint_cols)

        # Get player position (mid_hip at first few frames)
        mh_idx = keypoint_names.index('mid_hip') if 'mid_hip' in keypoint_names else None
        if mh_idx is not None:
            player_pos = np.nanmean(ts_3d[:10, mh_idx, :], axis=0)
        else:
            player_pos = np.nanmean(ts_3d[:10, :, :].mean(axis=1), axis=0)

        # Compute transformation
        R, origin = compute_hoop_relative_transform(player_pos)
        ts_hoop = transform_to_hoop_relative(ts_3d, R, origin)

        # Extract features from both coordinate systems
        feats = extract_features_from_transformed(
            ts_3d, ts_hoop, keypoint_names, row['participant_id']
        )
        all_features.append(feats)
        all_ids.append(row['id'])
        all_pids.append(row['participant_id'])

        if is_train:
            all_targets.append([row['angle'], row['depth'], row['left_right']])

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    # Convert to arrays
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)

    result = {
        'X': X,
        'pids': np.array(all_pids),
        'ids': np.array(all_ids),
        'feature_names': feature_names,
    }
    if is_train:
        result['y'] = np.array(all_targets, dtype=np.float32)

    print(f"  Feature matrix: {X.shape}")
    return result


def train_and_evaluate(train_data):
    """Train per-player per-target models with CV."""
    X = train_data['X']
    y = train_data['y']
    pids = train_data['pids']
    feature_names = train_data['feature_names']

    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros_like(y)
    all_models = {}
    all_scalers = {}

    print("\n" + "=" * 70)
    print("TRAINING PER-PLAYER PER-TARGET MODELS")
    print("=" * 70)

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y[mask]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        print(f"\nPlayer {pid} ({n} samples)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        all_scalers[pid] = scaler

        for t_idx, target in enumerate(TARGETS):
            y_t = y_p[:, t_idx]
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n)

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
                y_tr, y_val = y_t[tr_idx], y_t[val_idx]

                # LightGBM
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1
                )
                lgb_m.fit(X_tr, y_tr)

                # XGBoost
                xgb_m = xgb.XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1
                )
                xgb_m.fit(X_tr, y_tr)

                # CatBoost
                cat_m = CatBoostRegressor(
                    iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False
                )
                cat_m.fit(X_tr, y_tr)

                # Ridge
                ridge_m = Ridge(alpha=1.0, random_state=42)
                ridge_m.fit(X_tr, y_tr)

                # Ensemble
                pred = (0.3 * lgb_m.predict(X_val) + 0.3 * xgb_m.predict(X_val) +
                        0.3 * cat_m.predict(X_val) + 0.1 * ridge_m.predict(X_val))
                fold_preds[val_idx] = pred

            oof_preds[global_idx, t_idx] = fold_preds
            mse = np.mean((fold_preds - y_t) ** 2)
            print(f"  {target}: CV MSE = {mse:.6f}")

            # Train final models on all player data
            for name, cls, params in [
                ('lgb', lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
                ('xgb', xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
                ('cat', CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)),
                ('ridge', Ridge, dict(alpha=1.0, random_state=42)),
            ]:
                m = cls(**params)
                m.fit(X_scaled, y_t)
                all_models[(pid, target, name)] = m

    # Overall CV
    print("\n" + "=" * 70)
    print("OVERALL CV RESULTS")
    print("=" * 70)

    scalers_target = {}
    for target in TARGETS:
        scalers_target[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    total_scaled_mse = 0
    for t_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((oof_preds[:, t_idx] - y[:, t_idx]) ** 2)
        scale_range = scalers_target[target].data_range_[0]
        scaled_mse = raw_mse / (scale_range ** 2)
        total_scaled_mse += scaled_mse
        print(f"  {target}: raw MSE = {raw_mse:.6f}, scaled MSE = {scaled_mse:.8f}")

    avg_scaled_mse = total_scaled_mse / 3
    print(f"\n  AVERAGE SCALED MSE (CV): {avg_scaled_mse:.8f}")

    return {
        'models': all_models,
        'scalers': all_scalers,
        'oof_preds': oof_preds,
        'cv_score': avg_scaled_mse,
    }


def predict_test(test_data, trained):
    """Generate test predictions."""
    X = test_data['X']
    pids = test_data['pids']
    models = trained['models']
    scalers = trained['scalers']

    predictions = np.zeros((len(X), 3))

    for i, (x, pid) in enumerate(zip(X, pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        for t_idx, target in enumerate(TARGETS):
            lgb_pred = models[(pid, target, 'lgb')].predict(x_scaled)[0]
            xgb_pred = models[(pid, target, 'xgb')].predict(x_scaled)[0]
            cat_pred = models[(pid, target, 'cat')].predict(x_scaled)[0]
            ridge_pred = models[(pid, target, 'ridge')].predict(x_scaled)[0]
            predictions[i, t_idx] = 0.3*lgb_pred + 0.3*xgb_pred + 0.3*cat_pred + 0.1*ridge_pred

    return predictions


def create_submission(test_ids, predictions, submission_num, cv_score):
    """Create and save submission file."""
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled[:, 0],
        'scaled_depth': scaled[:, 1],
        'scaled_left_right': scaled[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{submission_num}.csv"
    sub.to_csv(filepath, index=False)

    print(f"\nSubmission saved: {filepath}")
    print(f"CV Score: {cv_score:.8f}")
    print(f"\nProfile check:")
    print(f"  angle_std:  {sub['scaled_angle'].std():.6f} (need < 0.14)")
    print(f"  depth_mean: {sub['scaled_depth'].mean():.6f} (need 0.50-0.51)")

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={sub[col].mean():.4f} std={sub[col].std():.4f}")

    return filepath


def main():
    t0 = time.time()

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    print("=" * 70)
    print(f"HOOP-RELATIVE COORDINATE TRANSFORM EXPERIMENT (Sub {next_num})")
    print("=" * 70)

    # Load and transform data
    train_data = load_and_transform(DATA_DIR / "train.csv", is_train=True)
    test_data = load_and_transform(DATA_DIR / "test.csv", is_train=False)

    # Train and evaluate
    trained = train_and_evaluate(train_data)

    # Generate test predictions
    predictions = predict_test(test_data, trained)

    # Create submission
    filepath = create_submission(test_data['ids'], predictions, next_num, trained['cv_score'])

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    return filepath


if __name__ == "__main__":
    main()
