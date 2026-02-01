"""
Stable Feature Model - Uses only high-importance, low-drift features.

Goal: Create a model with uncorrelated errors compared to Sub 9/10.
Then blend with Sub 25 for potential improvement.

Top stability-adjusted features (from drift report):
- right_knee_vel_max_time: 0.0007
- right_elbow_vel_load_mean: 0.0003
- wrist_vy_release: 0.0002
- right_elbow_vel_max_time: 0.0002
- set_position_stability: 0.00012
- mid_hip_z_energy: 0.00009
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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


def load_raw_data():
    """Load raw keypoint data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def extract_series(df):
        all_series = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])
            all_series.append(ts)
        return np.array(all_series)

    print("Loading training data...")
    train_series = extract_series(train_df)
    train_targets = train_df[["angle", "depth", "left_right"]].values
    train_pids = train_df["participant_id"].values
    train_ids = train_df["id"].values

    print("Loading test data...")
    test_series = extract_series(test_df)
    test_pids = test_df["participant_id"].values
    test_ids = test_df["id"].values

    return {
        "train_series": train_series,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_series": test_series,
        "test_pids": test_pids,
        "test_ids": test_ids,
        "keypoint_cols": keypoint_cols,
    }


def extract_stable_features(series, keypoint_cols):
    """
    Extract only the most stable features (high importance, low drift).
    Focus on velocity and timing features that generalize well.
    """
    n_samples = len(series)
    features_list = []
    feature_names = []

    # Key frame windows based on research
    release_frames = range(140, 170)  # Release phase
    load_frames = range(80, 120)  # Load phase
    prep_frames = range(20, 60)  # Preparation phase

    # Find keypoint indices
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    def get_idx(col):
        return col_to_idx.get(col, None)

    # Right knee velocity timing (most stable feature)
    right_knee_z_idx = get_idx("right_knee_z")
    if right_knee_z_idx is not None:
        knee_z = series[:, :, right_knee_z_idx]
        knee_vel = np.gradient(knee_z, axis=1)
        # Velocity max time (which frame has max velocity)
        vel_max_time = np.argmax(np.abs(knee_vel), axis=1) / 240.0
        features_list.append(vel_max_time.reshape(-1, 1))
        feature_names.append("right_knee_vel_max_time")
        # Mean velocity in release phase
        vel_release = np.nanmean(knee_vel[:, list(release_frames)], axis=1)
        features_list.append(vel_release.reshape(-1, 1))
        feature_names.append("right_knee_vel_release_mean")

    # Right elbow velocity features
    right_elbow_z_idx = get_idx("right_elbow_z")
    if right_elbow_z_idx is not None:
        elbow_z = series[:, :, right_elbow_z_idx]
        elbow_vel = np.gradient(elbow_z, axis=1)
        # Velocity max time
        vel_max_time = np.argmax(np.abs(elbow_vel), axis=1) / 240.0
        features_list.append(vel_max_time.reshape(-1, 1))
        feature_names.append("right_elbow_vel_max_time")
        # Load phase velocity
        vel_load = np.nanmean(elbow_vel[:, list(load_frames)], axis=1)
        features_list.append(vel_load.reshape(-1, 1))
        feature_names.append("right_elbow_vel_load_mean")

    # Wrist y velocity at release
    right_wrist_y_idx = get_idx("right_wrist_y")
    if right_wrist_y_idx is not None:
        wrist_y = series[:, :, right_wrist_y_idx]
        wrist_vy = np.gradient(wrist_y, axis=1)
        # Velocity at release (mean over release window)
        vy_release = np.nanmean(wrist_vy[:, list(release_frames)], axis=1)
        features_list.append(vy_release.reshape(-1, 1))
        feature_names.append("wrist_vy_release")
        # Y position at release
        y_release = np.nanmean(wrist_y[:, list(release_frames)], axis=1)
        features_list.append(y_release.reshape(-1, 1))
        feature_names.append("wrist_y_release")

    # Mid hip z energy (stability)
    mid_hip_z_idx = get_idx("mid_hip_z")
    if mid_hip_z_idx is not None:
        hip_z = series[:, :, mid_hip_z_idx]
        hip_vel = np.gradient(hip_z, axis=1)
        # Energy (sum of squared velocity)
        hip_energy = np.sum(hip_vel**2, axis=1)
        features_list.append(hip_energy.reshape(-1, 1))
        feature_names.append("mid_hip_z_energy")

    # Set position stability (variance of position in prep phase)
    right_wrist_z_idx = get_idx("right_wrist_z")
    if right_wrist_z_idx is not None:
        wrist_z = series[:, :, right_wrist_z_idx]
        prep_var = np.nanvar(wrist_z[:, list(prep_frames)], axis=1)
        features_list.append(prep_var.reshape(-1, 1))
        feature_names.append("set_position_stability")

    # Guide hand (left wrist) features
    left_wrist_z_idx = get_idx("left_wrist_z")
    if left_wrist_z_idx is not None:
        left_wrist_z = series[:, :, left_wrist_z_idx]
        # Mean in release phase
        lw_release = np.nanmean(left_wrist_z[:, list(release_frames)], axis=1)
        features_list.append(lw_release.reshape(-1, 1))
        feature_names.append("left_wrist_z_release")
        # Std (measure of consistency)
        lw_std = np.nanstd(left_wrist_z[:, list(release_frames)], axis=1)
        features_list.append(lw_std.reshape(-1, 1))
        feature_names.append("left_wrist_z_std")

    left_wrist_x_idx = get_idx("left_wrist_x")
    if left_wrist_x_idx is not None:
        left_wrist_x = series[:, :, left_wrist_x_idx]
        vx = np.gradient(left_wrist_x, axis=1)
        vx_release = np.nanmean(vx[:, list(release_frames)], axis=1)
        features_list.append(vx_release.reshape(-1, 1))
        feature_names.append("guide_hand_vx")

    # Shoulder features
    right_shoulder_z_idx = get_idx("right_shoulder_z")
    if right_shoulder_z_idx is not None:
        shoulder_z = series[:, :, right_shoulder_z_idx]
        # Q75 in release phase
        q75 = np.nanpercentile(shoulder_z[:, list(release_frames)], 75, axis=1)
        features_list.append(q75.reshape(-1, 1))
        feature_names.append("right_shoulder_z_q75")

    # Hip lateral range
    left_hip_x_idx = get_idx("left_hip_x")
    right_hip_x_idx = get_idx("right_hip_x")
    if left_hip_x_idx is not None and right_hip_x_idx is not None:
        left_hip_x = series[:, :, left_hip_x_idx]
        right_hip_x = series[:, :, right_hip_x_idx]
        lateral = left_hip_x - right_hip_x
        lateral_range = np.nanmax(lateral, axis=1) - np.nanmin(lateral, axis=1)
        features_list.append(lateral_range.reshape(-1, 1))
        feature_names.append("hip_lateral_range")

    # Jerk at release (second derivative of velocity)
    if right_wrist_z_idx is not None:
        wrist_z = series[:, :, right_wrist_z_idx]
        vel = np.gradient(wrist_z, axis=1)
        acc = np.gradient(vel, axis=1)
        jerk = np.gradient(acc, axis=1)
        jerk_release = np.nanmean(np.abs(jerk[:, list(release_frames)]), axis=1)
        features_list.append(jerk_release.reshape(-1, 1))
        feature_names.append("jerk_at_release")

    # Additional frame-based features for diversity
    # Use specific frames that showed high R2 in research
    for frame in [102, 153, 237]:
        if frame < 240:
            # Right wrist position at key frame
            if right_wrist_z_idx is not None:
                features_list.append(series[:, frame, right_wrist_z_idx].reshape(-1, 1))
                feature_names.append(f"right_wrist_z_f{frame}")
            if right_wrist_y_idx is not None:
                features_list.append(series[:, frame, right_wrist_y_idx].reshape(-1, 1))
                feature_names.append(f"right_wrist_y_f{frame}")

    # Combine all features
    if features_list:
        X = np.hstack(features_list)
        X = np.nan_to_num(X, nan=0.0)
        print(f"Extracted {X.shape[1]} stable features")
        return X, feature_names
    else:
        return None, []


def train_stable_model(data):
    """Train per-player Ridge model using stable features."""
    print("\n" + "=" * 60)
    print("STABLE FEATURE MODEL")
    print("=" * 60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]
    keypoint_cols = data["keypoint_cols"]

    ranges = get_target_ranges()

    # Extract stable features
    print("\nExtracting stable features for training data...")
    X_train, feature_names = extract_stable_features(train_series, keypoint_cols)
    print("\nExtracting stable features for test data...")
    X_test, _ = extract_stable_features(test_series, keypoint_cols)

    print(f"\nFeatures: {feature_names}")

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for t_idx, target in enumerate(TARGETS):
        print(f"\n{target}:")

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_tr = X_train[train_mask]
            y_tr = train_targets[train_mask, t_idx]
            X_te = X_test[test_mask]

            player_indices = np.where(train_mask)[0]

            # Scale features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            # 5-fold CV within player
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_tr_scaled):
                X_fold_tr = X_tr_scaled[train_idx]
                X_fold_val = X_tr_scaled[val_idx]
                y_fold_tr = y_tr[train_idx]

                # Use high regularization for stability
                model = Ridge(alpha=100)
                model.fit(X_fold_tr, y_fold_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_fold_val)
                fold_test_preds.append(model.predict(X_te_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        # Report CV for this target
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  CV: {scaled_mse:.6f}")

    # Total CV
    print("\nFinal CV:")
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return oof_preds, predictions, cv_score


def create_submission(test_ids, predictions, cv_score, approach_name):
    """Create submission file."""
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        name = f.stem
        if name.startswith("submission_"):
            try:
                nums.append(int(name.split('_')[1]))
            except:
                pass
        elif name.startswith("submission"):
            try:
                nums.append(int(name[10:]))
            except:
                pass

    next_num = max(nums) + 1 if nums else 1

    scaled_preds = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: {approach_name}")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")
    print(f"  depth_mean: {depth_mean:.4f}")
    print(f"  File: {filepath}")

    return filepath, next_num


def main():
    print("=" * 80)
    print("STABLE FEATURE MODEL")
    print("=" * 80)
    print("\nApproach: Use only high-importance, low-drift features")
    print("Goal: Create diverse predictions for blending with Sub 25")

    data = load_raw_data()

    # Train model
    oof, test_preds, cv_score = train_stable_model(data)

    # Create submission
    filepath, sub_num = create_submission(
        data["test_ids"], test_preds, cv_score, "stable_features"
    )

    # Analyze correlation with existing submissions
    print("\n" + "=" * 60)
    print("CORRELATION WITH EXISTING SUBMISSIONS")
    print("=" * 60)

    sub25 = pd.read_csv(SUBMISSION_DIR / "submission_25.csv")
    sub9 = pd.read_csv(SUBMISSION_DIR / "submission_9.csv")
    sub10 = pd.read_csv(SUBMISSION_DIR / "submission_10.csv")

    new_sub = pd.read_csv(filepath)

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        r25 = np.corrcoef(sub25[col], new_sub[col])[0, 1]
        r9 = np.corrcoef(sub9[col], new_sub[col])[0, 1]
        r10 = np.corrcoef(sub10[col], new_sub[col])[0, 1]
        print(f"{col}:")
        print(f"  vs Sub 25: r={r25:.4f}")
        print(f"  vs Sub 9:  r={r9:.4f}")
        print(f"  vs Sub 10: r={r10:.4f}")

    # Test blending with Sub 25
    print("\n" + "=" * 60)
    print("BLEND WITH SUB 25")
    print("=" * 60)

    for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        blend = (1-w) * sub25[['scaled_angle', 'scaled_depth', 'scaled_left_right']] + \
                w * new_sub[['scaled_angle', 'scaled_depth', 'scaled_left_right']]
        print(f"  w={w:.2f}: angle_std={blend.scaled_angle.std():.4f}, "
              f"depth_mean={blend.scaled_depth.mean():.4f}")

    return filepath, sub_num


if __name__ == "__main__":
    main()
