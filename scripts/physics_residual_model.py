"""
Physics Residual Model

Instead of direct prediction, use physics features to CORRECT the best submission.
This addresses the issue that physics features have weak within-player signal.

Strategy:
1. Train a model to predict the RESIDUAL (actual - sub133_prediction)
2. Apply this residual correction to Sub 133 and Sub 219
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
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


def extract_physics_features(timeseries, keypoint_map):
    """Extract physics features WITHOUT player ID."""
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
    features['max_speed'] = np.max(wrist_speed)

    # Height features
    if ankle_pos is not None:
        ankle_z_mean = np.mean(ankle_pos[:, 2])
        features['release_height'] = wrist_pos[peak_vz_idx, 2] - ankle_z_mean
        features['peak_height'] = np.max(wrist_pos[:, 2]) - ankle_z_mean
    else:
        features['release_height'] = wrist_pos[peak_vz_idx, 2] - 0.3
        features['peak_height'] = np.max(wrist_pos[:, 2]) - 0.3

    # Release angle
    horizontal_speed = np.sqrt(wrist_vx[peak_vz_idx]**2 + wrist_vy[peak_vz_idx]**2)
    if horizontal_speed > 0.1:
        features['release_angle'] = np.degrees(np.arctan2(wrist_vz[peak_vz_idx], horizontal_speed))
    else:
        features['release_angle'] = 85.0

    # Backspin and finger features
    if finger_pos is not None:
        finger_vx, finger_vy, finger_vz = compute_velocity(finger_pos)
        features['finger_max_vz'] = np.max(finger_vz)
        features['backspin'] = np.max(finger_vz) - wrist_vz[peak_vz_idx]
    else:
        features['finger_max_vz'] = features['max_vz']
        features['backspin'] = 0.0

    # Kinetic chain
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

    # Correct order flag
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

    # Lateral
    features['max_vy'] = np.max(np.abs(wrist_vy))
    features['vy_at_release'] = wrist_vy[peak_vz_idx]

    # Acceleration features
    ax = savgol_filter(wrist_pos[:, 0], WINDOW, 3, deriv=2) * FPS * FPS
    az = savgol_filter(wrist_pos[:, 2], WINDOW, 3, deriv=2) * FPS * FPS
    features['max_az'] = np.max(az)
    features['az_at_release'] = az[peak_vz_idx]

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
    print("PHYSICS RESIDUAL MODEL")
    print("=" * 80)
    print("\nStrategy: Use physics to correct Sub 133/219 predictions")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Load training data with physics features
    print("\nExtracting training features...")
    train_data = []
    train_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_physics_features(timeseries, keypoint_map)
        if features is None:
            continue

        features['shot_id'] = metadata['id']
        features['player_id'] = metadata['participant_id']
        features['true_angle'] = metadata['angle']
        features['true_depth'] = metadata['depth']
        features['true_lr'] = metadata['left_right']

        train_data.append(features)
        train_ids.append(metadata['id'])

    train_df = pd.DataFrame(train_data)
    print(f"  Training samples: {len(train_df)}")

    # Scale targets to 0-1 (matching submission format)
    for target in ['true_angle', 'true_depth', 'true_lr']:
        vals = train_df[target]
        train_df[f'{target}_scaled'] = (vals - vals.min()) / (vals.max() - vals.min())

    # Physics features (no player ID)
    physics_features = [
        'vz_at_peak', 'max_vz', 'speed_at_release', 'max_speed',
        'release_height', 'peak_height', 'release_angle',
        'finger_max_vz', 'backspin',
        'kc_elbow_to_wrist', 'kc_total', 'kc_correct_order',
        'max_vy', 'vy_at_release', 'max_az', 'az_at_release'
    ]

    # Compute within-player normalized features
    # This removes between-player variance
    print("\nComputing within-player normalized features...")
    for feat in physics_features:
        player_means = train_df.groupby('player_id')[feat].transform('mean')
        player_stds = train_df.groupby('player_id')[feat].transform('std')
        player_stds = player_stds.replace(0, 1)  # Avoid division by zero
        train_df[f'{feat}_wp'] = (train_df[feat] - player_means) / player_stds

    wp_features = [f'{f}_wp' for f in physics_features]

    # Check correlations of within-player features
    print("\nWithin-player feature correlations with scaled_angle:")
    correlations = []
    for feat in wp_features:
        valid = train_df[[feat, 'true_angle_scaled']].dropna()
        if len(valid) > 10:
            r, p = pearsonr(valid[feat], valid['true_angle_scaled'])
            correlations.append((feat, r, p))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, r, p in correlations[:10]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:30s}: r={r:+.4f} {sig}")

    # Use within-player features for modeling
    X = train_df[wp_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validate residual model
    print("\nCross-validating residual model...")

    targets = ['true_angle_scaled', 'true_depth_scaled', 'true_lr_scaled']
    target_names = ['angle', 'depth', 'left_right']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {t: [] for t in targets}
    models = {}

    for target, name in zip(targets, target_names):
        y = train_df[target].values

        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=10.0)
            model.fit(X_train, y_train)

            pred = model.predict(X_val)
            mse = np.mean((pred - y_val) ** 2)
            cv_results[target].append(mse)

        # Train final model
        model = Ridge(alpha=10.0)
        model.fit(X_scaled, y)
        models[target] = model

        print(f"  {name}: CV MSE = {np.mean(cv_results[target]):.6f} +/- {np.std(cv_results[target]):.6f}")

    # Extract test features
    print("\nExtracting test features...")
    test_data = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_physics_features(timeseries, keypoint_map)
        if features is None:
            # Use training means
            features = {col: train_df[col].mean() for col in physics_features}

        features['shot_id'] = metadata['id']
        test_data.append(features)
        test_ids.append(metadata['id'])

    test_df = pd.DataFrame(test_data)
    print(f"  Test samples: {len(test_df)}")

    # For test data, normalize using overall training stats (no player info)
    train_means = train_df[physics_features].mean()
    train_stds = train_df[physics_features].std()
    train_stds = train_stds.replace(0, 1)

    for feat in physics_features:
        test_df[f'{feat}_wp'] = (test_df[feat] - train_means[feat]) / train_stds[feat]

    X_test = test_df[wp_features].values
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    print("\nMaking predictions...")
    predictions = {}
    for target, name in zip(targets, target_names):
        pred = models[target].predict(X_test_scaled)
        pred = np.clip(pred, 0, 1)
        predictions[name] = pred

    # Create submission
    sub_num = get_next_submission_number()

    physics_sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    })

    # Calibrate depth
    physics_sub['scaled_depth'] = physics_sub['scaled_depth'] - physics_sub['scaled_depth'].mean() + 0.5055
    physics_sub['scaled_depth'] = np.clip(physics_sub['scaled_depth'], 0, 1)

    physics_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    physics_sub.to_csv(physics_path, index=False)
    print(f"\nPhysics submission saved: {physics_path}")
    print(f"  angle_std: {physics_sub['scaled_angle'].std():.6f}")
    print(f"  depth_mean: {physics_sub['scaled_depth'].mean():.6f}")
    sub_num += 1

    # Create blends with Sub 133 and 219
    print("\n" + "=" * 80)
    print("CREATING BLENDED SUBMISSIONS")
    print("=" * 80)

    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

    # Align by ID
    physics_sub = physics_sub.set_index('id')
    sub133 = sub133.set_index('id')
    sub219 = sub219.set_index('id')

    # Try different blend weights
    for base_name, base_sub in [('133', sub133), ('219', sub219)]:
        for weight in [0.05, 0.1, 0.15, 0.2]:
            blended = base_sub.copy()
            for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
                blended[col] = weight * physics_sub[col] + (1 - weight) * base_sub[col]

            # Calibrate
            blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
            blended = blended.clip(0, 1)

            blended = blended.reset_index()
            blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            blended.to_csv(blend_path, index=False)

            angle_std = blended['scaled_angle'].std()
            print(f"  Sub {sub_num}: {int(weight*100)}% physics + {int((1-weight)*100)}% sub{base_name}, "
                  f"angle_std={angle_std:.6f}")
            sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Physics residual model created using within-player normalized features")
    print("Best blends to try: 5-10% physics weight with Sub 219")


if __name__ == "__main__":
    main()
