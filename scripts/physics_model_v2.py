"""
Physics-based prediction model using correlated features.

Key findings:
- angle: STRONGLY correlated with vz (r=0.78)
- depth: weakly correlated with peak_height (r=0.18)
- left_right: weakly correlated with max_vy (r=0.18)

We'll build a model using the best physics features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

HOOP_POSITION = np.array([5.25, -25.0, 10.0])
FPS = 60
SUBMISSION_DIR = PROJECT_DIR / "submission"


def get_keypoint_map(keypoint_cols):
    keypoint_map = {}
    for i, col in enumerate(keypoint_cols):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in keypoint_map:
                keypoint_map[name] = {}
            keypoint_map[name][axis] = i
    return keypoint_map


def get_position(timeseries, keypoint_map, name, frame):
    if name not in keypoint_map:
        return None
    km = keypoint_map[name]
    if 'x' not in km or 'y' not in km or 'z' not in km:
        return None
    pos = np.array([
        timeseries[frame, km['x']],
        timeseries[frame, km['y']],
        timeseries[frame, km['z']]
    ])
    if np.any(np.isnan(pos)):
        return None
    return pos


def extract_physics_features(timeseries, keypoint_map):
    """
    Extract physics features that correlate with targets.
    Based on correlation analysis findings.
    """
    features = {}

    # Track wrist and fingers
    wrist_pos = []
    wrist_frames = []
    for frame in range(60, 200):
        pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if pos is not None:
            wrist_pos.append(pos)
            wrist_frames.append(frame)

    if len(wrist_pos) < 20:
        return None

    wrist_pos = np.array(wrist_pos)
    wrist_frames = np.array(wrist_frames)

    # Finger positions
    finger_pos = []
    finger_frames = []
    for frame in range(60, 200):
        pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
        if pos is not None:
            finger_pos.append(pos)
            finger_frames.append(frame)

    # Compute wrist velocities
    window = 7
    vx = savgol_filter(wrist_pos[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(wrist_pos[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(wrist_pos[:, 2], window, 3, deriv=1) * FPS
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Accelerations
    ax = savgol_filter(wrist_pos[:, 0], window, 3, deriv=2) * FPS * FPS
    az = savgol_filter(wrist_pos[:, 2], window, 3, deriv=2) * FPS * FPS

    # Key indices
    peak_vz_idx = np.argmax(vz)
    peak_speed_idx = np.argmax(speed)

    # === ANGLE PREDICTORS (strong correlations) ===
    features['vz_at_peak_vz'] = vz[peak_vz_idx]
    features['max_vz'] = np.max(vz)
    features['speed_at_peak_vz'] = speed[peak_vz_idx]
    features['peak_height'] = np.max(wrist_pos[:, 2])
    features['release_height'] = wrist_pos[peak_vz_idx, 2]
    features['max_speed'] = np.max(speed)
    features['max_az'] = np.max(az)

    # === DEPTH PREDICTORS (weak but significant) ===
    features['vx_at_peak_speed'] = vx[peak_speed_idx]
    features['max_vy_abs'] = np.max(np.abs(vy))
    features['forearm_angle'] = 0.0  # Will compute if possible

    # Release angle ratio
    horiz_vel = np.sqrt(vx[peak_vz_idx]**2 + vy[peak_vz_idx]**2)
    if horiz_vel > 0.1:
        features['release_angle_ratio'] = vz[peak_vz_idx] / horiz_vel
    else:
        features['release_angle_ratio'] = 10.0

    # === LEFT_RIGHT PREDICTORS ===
    features['max_vy'] = np.max(vy)
    features['min_vy'] = np.min(vy)
    features['vy_at_peak_vz'] = vy[peak_vz_idx]

    # Finger features
    if len(finger_pos) > 20:
        finger_pos = np.array(finger_pos)
        fvz = savgol_filter(finger_pos[:, 2], window, 3, deriv=1) * FPS
        features['finger_max_vz'] = np.max(fvz)
        features['finger_release_y'] = finger_pos[np.argmax(fvz), 1]
    else:
        features['finger_max_vz'] = features['max_vz']
        features['finger_release_y'] = wrist_pos[peak_vz_idx, 1]

    # Position features
    features['release_y_offset'] = wrist_pos[peak_vz_idx, 1] - HOOP_POSITION[1]
    features['release_x_dist'] = wrist_pos[peak_vz_idx, 0] - HOOP_POSITION[0]

    # Timing
    features['peak_vz_frame'] = wrist_frames[peak_vz_idx]

    return features


def get_next_submission_number():
    """Get next submission number."""
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            pass
    return max(numbers) + 1 if numbers else 1


def main():
    print("=" * 80)
    print("PHYSICS-BASED PREDICTION MODEL")
    print("=" * 80)
    print("\nUsing features with strong correlations to targets")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Extract features for training data
    print("\nExtracting features from training data...")
    train_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_physics_features(timeseries, keypoint_map)
        if features is None:
            continue

        features['shot_id'] = metadata['id']
        features['player_id'] = metadata['participant_id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']

        train_data.append(features)

    train_df = pd.DataFrame(train_data)
    print(f"  Training samples: {len(train_df)}")

    # Define feature columns for each target
    angle_features = [
        'vz_at_peak_vz', 'max_vz', 'speed_at_peak_vz', 'finger_max_vz',
        'peak_height', 'release_height', 'max_speed', 'max_az',
        'release_angle_ratio', 'release_y_offset'
    ]

    depth_features = [
        'peak_height', 'vx_at_peak_speed', 'max_vy_abs', 'release_angle_ratio',
        'release_height', 'finger_release_y', 'release_y_offset'
    ]

    lr_features = [
        'max_vy', 'min_vy', 'vy_at_peak_vz', 'finger_release_y',
        'release_angle_ratio', 'release_y_offset'
    ]

    # Cross-validation
    print("\nCross-validation (GroupKFold by player)...")
    gkf = GroupKFold(n_splits=5)

    targets = ['angle', 'depth', 'left_right']
    feature_sets = [angle_features, depth_features, lr_features]

    cv_results = {t: [] for t in targets}
    models = {}
    scalers = {}

    for target, feat_cols in zip(targets, feature_sets):
        X = train_df[feat_cols].values
        y = train_df[target].values
        groups = train_df['player_id'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for train_idx, val_idx in gkf.split(X_scaled, y, groups):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            pred = model.predict(X_val)
            mse = np.mean((pred - y_val) ** 2)
            cv_results[target].append(mse)

        # Train final model on all data
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        models[target] = model
        scalers[target] = scaler

        print(f"  {target}: CV MSE = {np.mean(cv_results[target]):.6f} +/- {np.std(cv_results[target]):.6f}")

    # Total CV MSE
    total_mse = sum(np.mean(cv_results[t]) for t in targets) / 3
    print(f"\n  TOTAL CV MSE: {total_mse:.6f}")

    # Extract features for test data
    print("\nExtracting features from test data...")
    test_data = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_physics_features(timeseries, keypoint_map)
        if features is None:
            # Use mean prediction if feature extraction fails
            features = {col: train_df[col].mean() for col in angle_features + depth_features + lr_features}

        features['shot_id'] = metadata['id']
        test_data.append(features)

    test_df = pd.DataFrame(test_data)
    print(f"  Test samples: {len(test_df)}")

    # Make predictions
    print("\nMaking predictions...")
    predictions = {'id': test_df['shot_id'].values}

    for target, feat_cols in zip(targets, feature_sets):
        X_test = test_df[feat_cols].values
        X_test_scaled = scalers[target].transform(X_test)
        pred = models[target].predict(X_test_scaled)
        predictions[target] = pred

    # Create submission
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        'id': predictions['id'],
        'angle': predictions['angle'],
        'depth': predictions['depth'],
        'left_right': predictions['left_right']
    })

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"\nSubmission saved to: {sub_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUBMISSION DETAILS")
    print("=" * 80)
    print(f"Submission: {sub_num}")
    print(f"Model: Ridge regression on physics features")
    print(f"CV MSE: {total_mse:.6f}")
    print(f"\nKey features used:")
    print(f"  angle: vz_at_peak_vz (r=0.78), speed_at_peak_vz (r=0.75)")
    print(f"  depth: peak_height (r=0.18), vx_at_peak_speed (r=-0.15)")
    print(f"  left_right: max_vy (r=0.18)")


if __name__ == "__main__":
    main()
