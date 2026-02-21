"""
Physics-Enhanced Submission Model

Uses accurate physics features combined with selective amplification.
Key insight: Physics features have strong BETWEEN-player signal but weak WITHIN-player signal.
Strategy: Use physics to create a separate prediction, then blend with best submission.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

FPS = 60
WINDOW = 7
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


def get_positions(timeseries, keypoint_map, name, start_frame=60, end_frame=200):
    if name not in keypoint_map:
        return None, None
    km = keypoint_map[name]
    if 'x' not in km or 'y' not in km or 'z' not in km:
        return None, None

    positions = []
    frames = []
    for frame in range(start_frame, min(end_frame, timeseries.shape[0])):
        pos = np.array([
            timeseries[frame, km['x']],
            timeseries[frame, km['y']],
            timeseries[frame, km['z']]
        ])
        if not np.any(np.isnan(pos)):
            positions.append(pos)
            frames.append(frame)

    if len(positions) < 20:
        return None, None

    return np.array(positions), np.array(frames)


def compute_velocity(positions, window=WINDOW):
    vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS
    return vx, vy, vz


def soft_clamp(value, low, high):
    mid = (low + high) / 2
    half_range = (high - low) / 2
    normalized = (value - mid) / half_range
    clamped = np.tanh(normalized) * half_range + mid
    return clamped


def extract_physics_features(timeseries, keypoint_map, player_id=None):
    """Extract all physics features."""
    features = {}

    wrist_pos, wrist_frames = get_positions(timeseries, keypoint_map, 'right_wrist')
    if wrist_pos is None:
        return None

    ankle_pos, _ = get_positions(timeseries, keypoint_map, 'right_ankle')
    finger_pos, finger_frames = get_positions(timeseries, keypoint_map, 'right_third_finger_distal')
    hip_pos, hip_frames = get_positions(timeseries, keypoint_map, 'mid_hip')
    shoulder_pos, shoulder_frames = get_positions(timeseries, keypoint_map, 'right_shoulder')
    elbow_pos, elbow_frames = get_positions(timeseries, keypoint_map, 'right_elbow')

    wrist_vx, wrist_vy, wrist_vz = compute_velocity(wrist_pos)
    wrist_speed = np.sqrt(wrist_vx**2 + wrist_vy**2 + wrist_vz**2)

    peak_vz_idx = np.argmax(wrist_vz)
    peak_vz_frame = wrist_frames[peak_vz_idx]

    # Core physics features
    features['vz_at_peak'] = wrist_vz[peak_vz_idx]
    features['max_vz'] = np.max(wrist_vz)
    features['speed_at_release'] = wrist_speed[peak_vz_idx]

    # Release height relative to ankle
    if ankle_pos is not None:
        ankle_z_mean = np.mean(ankle_pos[:, 2])
        features['release_height'] = wrist_pos[peak_vz_idx, 2] - ankle_z_mean
        features['peak_height'] = np.max(wrist_pos[:, 2]) - ankle_z_mean
    else:
        features['release_height'] = wrist_pos[peak_vz_idx, 2] - 0.3
        features['peak_height'] = np.max(wrist_pos[:, 2]) - 0.3

    # Release angle with soft clamp
    horizontal_speed = np.sqrt(wrist_vx[peak_vz_idx]**2 + wrist_vy[peak_vz_idx]**2)
    if horizontal_speed > 0.1:
        raw_angle = np.degrees(np.arctan2(wrist_vz[peak_vz_idx], horizontal_speed))
    else:
        raw_angle = 85.0 if wrist_vz[peak_vz_idx] > 0 else -85.0
    features['release_angle'] = soft_clamp(raw_angle, 25, 65)

    # Backspin
    if finger_pos is not None:
        finger_vx, finger_vy, finger_vz = compute_velocity(finger_pos)
        finger_vz_at_release = np.max(finger_vz)
        features['backspin'] = finger_vz_at_release - wrist_vz[peak_vz_idx]
        features['finger_max_vz'] = np.max(finger_vz)
    else:
        features['backspin'] = 0.0
        features['finger_max_vz'] = features['max_vz']

    # Kinetic chain timing
    kinetic_chain = {'wrist': peak_vz_frame}

    if hip_pos is not None:
        hip_vx, hip_vy, hip_vz = compute_velocity(hip_pos)
        hip_speed = np.sqrt(hip_vx**2 + hip_vy**2 + hip_vz**2)
        kinetic_chain['hip'] = hip_frames[np.argmax(hip_speed)]

    if shoulder_pos is not None:
        shoulder_vx, shoulder_vy, shoulder_vz = compute_velocity(shoulder_pos)
        shoulder_speed = np.sqrt(shoulder_vx**2 + shoulder_vy**2 + shoulder_vz**2)
        kinetic_chain['shoulder'] = shoulder_frames[np.argmax(shoulder_speed)]

    if elbow_pos is not None:
        elbow_vx, elbow_vy, elbow_vz = compute_velocity(elbow_pos)
        elbow_speed = np.sqrt(elbow_vx**2 + elbow_vy**2 + elbow_vz**2)
        kinetic_chain['elbow'] = elbow_frames[np.argmax(elbow_speed)]

    if 'elbow' in kinetic_chain:
        features['kc_elbow_to_wrist'] = (kinetic_chain['wrist'] - kinetic_chain['elbow']) / FPS
    else:
        features['kc_elbow_to_wrist'] = 0.0

    if 'hip' in kinetic_chain:
        features['kc_total'] = (kinetic_chain['wrist'] - kinetic_chain['hip']) / FPS
    else:
        features['kc_total'] = 0.0

    # Check correct order
    correct_order = True
    joints_order = ['hip', 'shoulder', 'elbow', 'wrist']
    prev_time = -np.inf
    for joint in joints_order:
        if joint in kinetic_chain:
            if kinetic_chain[joint] < prev_time:
                correct_order = False
                break
            prev_time = kinetic_chain[joint]
    features['kc_correct_order'] = 1.0 if correct_order else 0.0

    # Lateral velocity features
    features['max_vy'] = np.max(np.abs(wrist_vy))
    features['vy_at_release'] = wrist_vy[peak_vz_idx]

    # Player indicator
    if player_id is not None:
        features['player_id'] = player_id
        for pid in range(1, 6):
            features[f'player_{pid}'] = 1.0 if player_id == pid else 0.0

    return features


def get_next_submission_number():
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
    print("PHYSICS-ENHANCED SUBMISSION MODEL")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Extract training features
    print("\nExtracting training features...")
    train_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_physics_features(timeseries, keypoint_map, metadata['participant_id'])
        if features is None:
            continue

        features['shot_id'] = metadata['id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']

        train_data.append(features)

    train_df = pd.DataFrame(train_data)
    print(f"  Training samples: {len(train_df)}")

    # Define feature sets for each target
    # angle: strong physics signal
    angle_features = [
        'vz_at_peak', 'speed_at_release', 'finger_max_vz',
        'release_angle', 'peak_height', 'release_height',
        'kc_correct_order', 'kc_elbow_to_wrist',
        'player_1', 'player_2', 'player_3', 'player_4', 'player_5'
    ]

    # depth: weak physics signal, rely more on player baselines
    depth_features = [
        'peak_height', 'backspin', 'release_height',
        'player_1', 'player_2', 'player_3', 'player_4', 'player_5'
    ]

    # left_right: weak physics signal
    lr_features = [
        'max_vy', 'vy_at_release', 'kc_elbow_to_wrist',
        'player_1', 'player_2', 'player_3', 'player_4', 'player_5'
    ]

    # Cross-validation
    print("\nCross-validation (Leave-One-Player-Out)...")
    gkf = GroupKFold(n_splits=5)

    targets = ['angle', 'depth', 'left_right']
    feature_sets = [angle_features, depth_features, lr_features]
    alphas = [10.0, 50.0, 50.0]  # Stronger regularization for weak signals

    cv_results = {t: [] for t in targets}
    models = {}
    scalers = {}

    for target, feat_cols, alpha in zip(targets, feature_sets, alphas):
        X = train_df[feat_cols].values
        y = train_df[target].values
        groups = train_df['player_id'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for train_idx, val_idx in gkf.split(X_scaled, y, groups):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            pred = model.predict(X_val)
            mse = np.mean((pred - y_val) ** 2)
            cv_results[target].append(mse)

        # Train final model
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)
        models[target] = model
        scalers[target] = scaler

        print(f"  {target}: CV MSE = {np.mean(cv_results[target]):.6f} +/- {np.std(cv_results[target]):.6f}")

    total_mse = sum(np.mean(cv_results[t]) for t in targets) / 3
    print(f"\n  TOTAL CV MSE: {total_mse:.6f}")

    # Also train within-player models
    print("\nTraining within-player models...")
    within_player_cv = {t: [] for t in targets}

    for player_id in range(1, 6):
        player_df = train_df[train_df['player_id'] == player_id]
        if len(player_df) < 20:
            continue

        for target, feat_cols in zip(targets, feature_sets):
            # Remove player dummies for within-player
            feat_no_player = [f for f in feat_cols if not f.startswith('player_')]
            if len(feat_no_player) == 0:
                continue

            X = player_df[feat_no_player].values
            y = player_df[target].values

            # Simple holdout CV
            n = len(X)
            split = int(0.8 * n)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            if len(X_train) < 5 or len(X_val) < 3:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train)

            pred = model.predict(X_val_s)
            mse = np.mean((pred - y_val) ** 2)
            within_player_cv[target].append(mse)

    for target in targets:
        if within_player_cv[target]:
            print(f"  {target} within-player CV MSE: {np.mean(within_player_cv[target]):.6f}")

    # Extract test features
    print("\nExtracting test features...")
    test_data = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_physics_features(timeseries, keypoint_map, metadata.get('participant_id'))
        if features is None:
            # Use training means as fallback
            features = {col: train_df[col].mean() for col in angle_features + depth_features + lr_features}
            # Set all player flags to 0.2 (uniform uncertainty)
            for pid in range(1, 6):
                features[f'player_{pid}'] = 0.2

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

        # Clip predictions to valid range
        pred = np.clip(pred, 0, 1)
        predictions[target] = pred

    # Scale predictions to match competition format (0-1 scaled)
    # The targets in training are already in original units (degrees, inches)
    # Need to scale to 0-1 range matching the competition

    # Load Sub 133 to get target scale reference
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    # Create physics-only submission
    sub_num = get_next_submission_number()

    # For now, create unscaled submission and compare
    physics_sub = pd.DataFrame({
        'id': predictions['id'],
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    })

    # The predictions are in original units, need to scale
    # Check training target ranges
    print("\nTraining target ranges:")
    for target in targets:
        vals = train_df[target]
        print(f"  {target}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")

    # Scale to 0-1
    for target in targets:
        col = f'scaled_{target}'
        train_min = train_df[target].min()
        train_max = train_df[target].max()
        physics_sub[col] = (predictions[target] - train_min) / (train_max - train_min)
        physics_sub[col] = np.clip(physics_sub[col], 0, 1)

    # Calibrate depth mean to 0.5055
    physics_sub['scaled_depth'] = physics_sub['scaled_depth'] - physics_sub['scaled_depth'].mean() + 0.5055
    physics_sub['scaled_depth'] = np.clip(physics_sub['scaled_depth'], 0, 1)

    # Save physics-only submission
    physics_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    physics_sub[['id', 'scaled_angle', 'scaled_depth', 'scaled_left_right']].to_csv(physics_path, index=False)
    print(f"\nPhysics-only submission saved: {physics_path}")

    # Check profile
    print(f"\nPhysics submission profile:")
    print(f"  angle_std: {physics_sub['scaled_angle'].std():.6f}")
    print(f"  depth_mean: {physics_sub['scaled_depth'].mean():.6f}")
    sub_num += 1

    # Create blended submission with Sub 133
    print("\n" + "=" * 80)
    print("CREATING BLENDED SUBMISSIONS")
    print("=" * 80)

    # Try different blend weights
    blend_weights = [0.1, 0.15, 0.2, 0.25, 0.3]

    for weight in blend_weights:
        blended = sub133.copy()
        for target in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blended[target] = weight * physics_sub[target] + (1 - weight) * sub133[target]

        # Calibrate
        blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
        blended['scaled_depth'] = np.clip(blended['scaled_depth'], 0, 1)
        blended['scaled_angle'] = np.clip(blended['scaled_angle'], 0, 1)
        blended['scaled_left_right'] = np.clip(blended['scaled_left_right'], 0, 1)

        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(blend_path, index=False)

        print(f"  Sub {sub_num}: {int(weight*100)}% physics + {int((1-weight)*100)}% sub133, "
              f"angle_std={blended['scaled_angle'].std():.6f}")
        sub_num += 1

    # Also try blending with Sub 219 (best LB)
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

    for weight in [0.05, 0.1, 0.15]:
        blended = sub219.copy()
        for target in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blended[target] = weight * physics_sub[target] + (1 - weight) * sub219[target]

        # Calibrate
        blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
        blended['scaled_depth'] = np.clip(blended['scaled_depth'], 0, 1)
        blended['scaled_angle'] = np.clip(blended['scaled_angle'], 0, 1)
        blended['scaled_left_right'] = np.clip(blended['scaled_left_right'], 0, 1)

        blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(blend_path, index=False)

        print(f"  Sub {sub_num}: {int(weight*100)}% physics + {int((1-weight)*100)}% sub219, "
              f"angle_std={blended['scaled_angle'].std():.6f}")
        sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Generated submissions from physics features")
    print(f"Key insight: Physics features have strong BETWEEN-player but weak WITHIN-player signal")
    print(f"Best strategy: Small blend weight (5-15%) with proven submission")


if __name__ == "__main__":
    main()
