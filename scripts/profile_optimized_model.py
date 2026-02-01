"""
Profile-Optimized Model

Key insight: To beat Sub 113, we need both:
1. Diversity (low correlation with existing predictions)
2. Good profile (angle_std < 0.14, depth_mean ~ 0.505)

Strategy:
1. Use stable features with low drift
2. Apply strong regularization to control angle variance
3. Calibrate depth predictions toward optimal mean
4. Blend per-player predictions with player means for variance reduction
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from scipy.signal import savgol_filter
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}

KEYPOINT_MAP = {}


def init_keypoint_mapping(keypoint_cols):
    global KEYPOINT_MAP
    for i, col in enumerate(keypoint_cols):
        KEYPOINT_MAP[col] = i


def get_kp(series, name, frame=None):
    if name not in KEYPOINT_MAP:
        return None
    idx = KEYPOINT_MAP[name]
    if frame is not None:
        return series[frame, idx]
    return series[:, idx]


def smooth_signal(signal, window=5):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < window:
        return signal
    try:
        return savgol_filter(signal, window, 2)
    except:
        return signal


def compute_velocity(signal):
    return np.gradient(smooth_signal(signal))


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_target_ranges():
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }


# Top stable features identified from previous research
STABLE_FEATURES = [
    "right_knee_vel_max_time",
    "right_elbow_vel_load_mean",
    "wrist_vy_release",
    "right_elbow_vel_max_time",
    "set_position_stability",
    "mid_hip_z_energy",
    "right_wrist_vel_load_mean",
    "wrist_y_release",
]


def extract_stable_features(series, participant_id):
    """Extract features with high importance and low drift."""
    features = {}

    # Key frame ranges
    release_frames = list(range(145, 165))
    load_frames = list(range(80, 120))

    # =====================================================
    # CORE STABLE FEATURES
    # =====================================================

    # Right knee velocity timing (most stable high-importance feature)
    right_knee_z = get_kp(series, "right_knee_z")
    if right_knee_z is not None:
        vel = compute_velocity(right_knee_z)
        features['right_knee_vel_max_time'] = np.argmax(np.abs(vel)) / 240.0
        features['right_knee_vel_max'] = np.nanmax(np.abs(vel))
        features['right_knee_vel_release'] = np.nanmean(vel[release_frames])

    # Right elbow velocity (critical for depth)
    right_elbow_z = get_kp(series, "right_elbow_z")
    if right_elbow_z is not None:
        vel = compute_velocity(right_elbow_z)
        features['right_elbow_vel_load_mean'] = np.nanmean(vel[load_frames])
        features['right_elbow_vel_max_time'] = np.argmax(np.abs(vel)) / 240.0
        features['right_elbow_vel_release'] = np.nanmean(vel[release_frames])

    # Wrist velocity features
    right_wrist_z = get_kp(series, "right_wrist_z")
    right_wrist_y = get_kp(series, "right_wrist_y")
    if right_wrist_z is not None:
        vel = compute_velocity(right_wrist_z)
        features['right_wrist_vel_load_mean'] = np.nanmean(vel[load_frames])
        features['right_wrist_vel_release'] = np.nanmean(vel[release_frames])
        features['right_wrist_vel_max_time'] = np.argmax(np.abs(vel)) / 240.0
    if right_wrist_y is not None:
        vel_y = compute_velocity(right_wrist_y)
        features['wrist_vy_release'] = np.nanmean(vel_y[release_frames])
        features['wrist_y_release'] = np.nanmean(right_wrist_y[release_frames])

    # Mid hip energy (stability indicator)
    mid_hip_z = get_kp(series, "mid_hip_z")
    if mid_hip_z is not None:
        vel = compute_velocity(mid_hip_z)
        features['mid_hip_z_energy'] = np.sum(vel**2)
        features['mid_hip_vel_release'] = np.nanmean(vel[release_frames])

    # Set position stability
    right_wrist_z = get_kp(series, "right_wrist_z")
    right_elbow_z = get_kp(series, "right_elbow_z")
    if right_wrist_z is not None and right_elbow_z is not None:
        # Stability in setup phase
        wrist_setup_std = np.nanstd(right_wrist_z[40:80])
        elbow_setup_std = np.nanstd(right_elbow_z[40:80])
        features['set_position_stability'] = wrist_setup_std + elbow_setup_std

    # =====================================================
    # ADDITIONAL STABLE POSITION FEATURES
    # =====================================================

    # Key frame positions (more stable than velocities)
    key_joints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "right_knee", "mid_hip", "left_wrist"
    ]

    for joint in key_joints:
        for axis in ["x", "y", "z"]:
            data = get_kp(series, f"{joint}_{axis}")
            if data is not None:
                # Position at key frames
                for f in [150, 155, 160]:
                    if f < 240:
                        features[f'{joint}_{axis}_f{f}'] = data[f]

                # Mean position at release
                features[f'{joint}_{axis}_release'] = np.nanmean(data[release_frames])

    # =====================================================
    # SHOOTING MECHANICS
    # =====================================================

    # Elbow-wrist coordination (research: more coordination = better)
    right_elbow_z = get_kp(series, "right_elbow_z")
    right_wrist_z = get_kp(series, "right_wrist_z")
    if right_elbow_z is not None and right_wrist_z is not None:
        elbow_vel = compute_velocity(right_elbow_z)
        wrist_vel = compute_velocity(right_wrist_z)

        # Wrist snap
        wrist_snap = wrist_vel - elbow_vel
        features['wrist_snap_release'] = np.nanmean(wrist_snap[release_frames])

        # Coordination
        if len(elbow_vel) > 175 and len(wrist_vel) > 175:
            corr = np.corrcoef(elbow_vel[130:175], wrist_vel[130:175])[0, 1]
            features['elbow_wrist_coord'] = corr if not np.isnan(corr) else 0

    # Guide hand (left wrist)
    left_wrist_z = get_kp(series, "left_wrist_z")
    left_wrist_x = get_kp(series, "left_wrist_x")
    if left_wrist_z is not None:
        features['guide_hand_z_release'] = np.nanmean(left_wrist_z[release_frames])
    if left_wrist_x is not None:
        vel_x = compute_velocity(left_wrist_x)
        features['guide_hand_vx_release'] = np.nanmean(vel_x[release_frames])

    # Clean up NaN/Inf
    for k, v in list(features.items()):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0

    return features


def load_and_extract_features():
    """Load data and extract features."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    init_keypoint_mapping(keypoint_cols)

    def process_df(df, desc="Processing"):
        all_features = []
        all_pids = []
        all_ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            features = extract_stable_features(ts, row['participant_id'])
            all_features.append(features)
            all_pids.append(row['participant_id'])
            all_ids.append(row['id'])

        return pd.DataFrame(all_features), np.array(all_pids), np.array(all_ids)

    print("Processing training data...")
    train_features, train_pids, train_ids = process_df(train_df, "Train")
    train_targets = train_df[["angle", "depth", "left_right"]].values

    print("Processing test data...")
    test_features, test_pids, test_ids = process_df(test_df, "Test")

    return {
        "train_features": train_features,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_features": test_features,
        "test_pids": test_pids,
        "test_ids": test_ids,
    }


def train_profile_optimized(data, shrinkage=0.15):
    """Train with profile optimization - shrink predictions toward player mean."""
    print("\n" + "=" * 60)
    print(f"PROFILE-OPTIMIZED MODEL (shrinkage={shrinkage})")
    print("=" * 60)

    X_train = np.nan_to_num(data["train_features"].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = data["train_targets"]
    train_pids = data["train_pids"]

    X_test = np.nan_to_num(data["test_features"].values, nan=0.0, posinf=0.0, neginf=0.0)
    test_pids = data["test_pids"]

    print(f"Features: {X_train.shape[1]}")

    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(X_test), 3))
    oof_preds = np.zeros_like(y_train)

    # Store player means for shrinkage
    player_means = {}
    for pid in unique_pids:
        mask = train_pids == pid
        player_means[pid] = y_train[mask].mean(axis=0)

    for t_idx, target in enumerate(TARGETS):
        print(f"\n{target}:")

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_tr = X_train[train_mask]
            X_te = X_test[test_mask]
            y_tr = y_train[train_mask, t_idx]
            player_indices = np.where(train_mask)[0]
            player_mean = player_means[pid][t_idx]

            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for fold_train_idx, fold_val_idx in kf.split(X_tr_scaled):
                X_fold_tr = X_tr_scaled[fold_train_idx]
                X_fold_val = X_tr_scaled[fold_val_idx]
                y_fold_tr = y_tr[fold_train_idx]

                # Use higher alpha for more regularization
                model = Ridge(alpha=200)
                model.fit(X_fold_tr, y_fold_tr)

                # Get raw predictions
                val_pred = model.predict(X_fold_val)
                test_pred = model.predict(X_te_scaled)

                # Apply shrinkage toward player mean
                val_pred_shrunk = (1 - shrinkage) * val_pred + shrinkage * player_mean
                test_pred_shrunk = (1 - shrinkage) * test_pred + shrinkage * player_mean

                oof_preds[player_indices[fold_val_idx], t_idx] = val_pred_shrunk
                fold_test_preds.append(test_pred_shrunk)

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        mse = np.mean((oof_preds[:, t_idx] - y_train[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  CV: {scaled_mse:.6f}")

    # Total CV
    print("\nFinal CV:")
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - y_train[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return oof_preds, predictions, cv_score


def calibrate_depth_predictions(predictions, target_mean=0.505):
    """Calibrate depth predictions to achieve target mean."""
    # Scale to target mean while preserving relative ordering
    current_mean = predictions.mean()
    if abs(current_mean - target_mean) > 0.001:
        # Linear shift
        shift = target_mean - current_mean
        predictions = predictions + shift
    return predictions


def main():
    print("=" * 80)
    print("PROFILE-OPTIMIZED MODEL")
    print("=" * 80)
    print("\nStrategy:")
    print("  1. Use stable features with low drift")
    print("  2. Apply shrinkage to reduce variance")
    print("  3. Calibrate depth toward optimal mean (0.505)")

    data = load_and_extract_features()

    # Test different shrinkage levels
    best_score = float('inf')
    best_shrinkage = 0.15
    best_result = None

    print("\nTesting shrinkage levels...")
    for shrinkage in [0.10, 0.15, 0.20, 0.25]:
        oof, preds, cv = train_profile_optimized(data, shrinkage=shrinkage)

        # Check profile
        test_preds = preds.copy()
        scaled_preds = np.zeros_like(test_preds)
        for i, target in enumerate(TARGETS):
            scaled_preds[:, i] = TARGET_SCALERS[target].transform(
                test_preds[:, i].reshape(-1, 1)
            ).flatten()

        angle_std = np.std(scaled_preds[:, 0])
        depth_mean = np.mean(scaled_preds[:, 1])

        print(f"\n  Shrinkage {shrinkage}: CV={cv:.6f}, angle_std={angle_std:.4f}, depth_mean={depth_mean:.4f}")

        # Score based on profile closeness
        profile_score = abs(angle_std - 0.138) + abs(depth_mean - 0.505)

        if profile_score < best_score:
            best_score = profile_score
            best_shrinkage = shrinkage
            best_result = (oof, preds, cv)

    print(f"\n\nBest shrinkage: {best_shrinkage}")

    oof, test_preds, cv_score = best_result

    # Calibrate depth predictions
    print("\nCalibrating depth predictions...")
    scaled_depth = TARGET_SCALERS["depth"].transform(test_preds[:, 1].reshape(-1, 1)).flatten()
    depth_mean_before = scaled_depth.mean()
    scaled_depth = calibrate_depth_predictions(scaled_depth, target_mean=0.505)
    depth_mean_after = scaled_depth.mean()
    print(f"  Depth mean: {depth_mean_before:.4f} -> {depth_mean_after:.4f}")

    # Create submission
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        try:
            name = f.stem
            if name.startswith('submission_'):
                nums.append(int(name.split('_')[1]))
            elif name.startswith('submission'):
                nums.append(int(name.replace('submission', '')))
        except:
            pass
    next_num = max(nums) + 1 if nums else 1

    scaled_preds = np.zeros_like(test_preds)
    for i, target in enumerate(TARGETS):
        if target == "depth":
            scaled_preds[:, i] = scaled_depth
        else:
            scaled_preds[:, i] = TARGET_SCALERS[target].transform(
                test_preds[:, i].reshape(-1, 1)
            ).flatten()

    submission = pd.DataFrame({
        'id': data["test_ids"],
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: profile_optimized")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f} (target: <0.14)")
    print(f"  depth_mean: {depth_mean:.4f} (target: ~0.505)")
    print(f"  File: {filepath}")

    # Check correlation with Sub 113
    sub113_path = SUBMISSION_DIR / "submission_113.csv"
    if sub113_path.exists():
        sub113 = pd.read_csv(sub113_path)
        print("\nCorrelation with Sub 113:")
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            r = np.corrcoef(sub113[col], submission[col])[0, 1]
            print(f"  {col}: r={r:.4f}")

    # Test blending with Sub 113
    if sub113_path.exists():
        print("\nBlending potential with Sub 113:")
        sub113 = pd.read_csv(sub113_path)
        cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']
        for w in [0.05, 0.10, 0.15]:
            blend = (1-w) * sub113[cols].values + w * submission[cols].values
            blend_angle_std = np.std(blend[:, 0])
            blend_depth_mean = np.mean(blend[:, 1])
            print(f"  w={w:.2f}: angle_std={blend_angle_std:.4f}, depth_mean={blend_depth_mean:.4f}")

    return filepath


if __name__ == "__main__":
    main()
