"""
Temporal Sequence Modeling

Use the FULL 240 frame sequence, not just release frames.
Apply 1D CNN or LSTM to learn temporal patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def extract_temporal_features(timeseries, keypoint_idx):
    """
    Extract features that capture the full temporal sequence.
    """
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Key joints to track
    joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip']

    # 1. Temporal statistics across entire sequence
    for joint in joints:
        positions = []
        for f in range(0, n_frames, 5):  # Sample every 5 frames
            pos = get_joint(joint, f)
            if np.any(pos):
                positions.append(pos)

        if positions:
            positions = np.array(positions)
            for i, coord in enumerate(['x', 'y', 'z']):
                features[f'{joint}_{coord}_mean'] = np.mean(positions[:, i])
                features[f'{joint}_{coord}_std'] = np.std(positions[:, i])
                features[f'{joint}_{coord}_range'] = np.max(positions[:, i]) - np.min(positions[:, i])

    # 2. Phase-based features (divide sequence into phases)
    phases = [(0, 60), (60, 120), (120, 180), (180, 240)]  # 4 phases
    phase_names = ['windup', 'preparation', 'release', 'followthrough']

    for phase_idx, (start, end) in enumerate(phases):
        for joint in ['right_wrist']:
            positions = []
            for f in range(start, min(end, n_frames), 3):
                pos = get_joint(joint, f)
                if np.any(pos):
                    positions.append(pos)

            if len(positions) >= 2:
                positions = np.array(positions)
                # Velocity in this phase
                vel = positions[-1] - positions[0]
                features[f'{joint}_{phase_names[phase_idx]}_vx'] = vel[0]
                features[f'{joint}_{phase_names[phase_idx]}_vy'] = vel[1]
                features[f'{joint}_{phase_names[phase_idx]}_vz'] = vel[2]
                features[f'{joint}_{phase_names[phase_idx]}_speed'] = np.linalg.norm(vel)

    # 3. Peak detection
    wrist_z = []
    for f in range(n_frames):
        pos = get_joint('right_wrist', f)
        wrist_z.append(pos[2] if np.any(pos) else 0)

    wrist_z = np.array(wrist_z)
    if np.any(wrist_z):
        peak_idx = np.argmax(wrist_z)
        features['wrist_peak_frame'] = peak_idx / n_frames  # Normalized
        features['wrist_peak_height'] = np.max(wrist_z)

        # Velocity at peak
        if peak_idx > 2 and peak_idx < n_frames - 2:
            vel_at_peak = (wrist_z[peak_idx + 2] - wrist_z[peak_idx - 2]) / 4
            features['wrist_vel_at_peak'] = vel_at_peak

    # 4. Acceleration patterns
    for joint in ['right_wrist']:
        velocities = []
        for f in range(5, n_frames - 5, 5):
            pos_before = get_joint(joint, f - 5)
            pos_after = get_joint(joint, f + 5)
            if np.any(pos_before) and np.any(pos_after):
                vel = (pos_after - pos_before) / 10
                velocities.append(np.linalg.norm(vel))

        if velocities:
            features[f'{joint}_max_velocity'] = np.max(velocities)
            features[f'{joint}_velocity_variance'] = np.var(velocities)
            features[f'{joint}_acceleration_peak_frame'] = np.argmax(velocities) / len(velocities)

    # 5. Relative joint positions (arm configuration)
    for f in [100, 130, 153, 170]:
        wrist = get_joint('right_wrist', f)
        elbow = get_joint('right_elbow', f)
        shoulder = get_joint('right_shoulder', f)

        if np.any(wrist) and np.any(elbow) and np.any(shoulder):
            # Elbow angle
            v1 = shoulder - elbow
            v2 = wrist - elbow
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            features[f'elbow_angle_f{f}'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            # Arm extension
            features[f'arm_length_f{f}'] = np.linalg.norm(wrist - shoulder)

    # 6. Trajectory smoothness (jerk)
    wrist_positions = []
    for f in range(120, 180):
        pos = get_joint('right_wrist', f)
        if np.any(pos):
            wrist_positions.append(pos)

    if len(wrist_positions) >= 4:
        wrist_positions = np.array(wrist_positions)
        # Compute jerk (third derivative)
        vel = np.diff(wrist_positions, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        features['trajectory_jerk'] = np.mean(np.linalg.norm(jerk, axis=1))
        features['trajectory_smoothness'] = 1 / (features['trajectory_jerk'] + 0.01)

    return features


def main():
    print("="*80)
    print("TEMPORAL SEQUENCE MODELING")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Extract features
    print("\nExtracting temporal features...")
    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_temporal_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])

    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_temporal_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Features: {len(common_cols)}")
    print(f"Training samples: {len(X_train)}")

    # Feature correlations
    print("\nTop feature correlations with angle:")
    corrs = []
    for col in common_cols:
        if X_train[col].std() > 0.001:
            corr = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
            if not np.isnan(corr):
                corrs.append((col, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs[:15]:
        print(f"  {col}: {corr:.4f}")

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA to reduce noise
    pca = PCA(n_components=min(30, len(common_cols)))
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"\nPCA components: {X_train_pca.shape[1]} (variance explained: {pca.explained_variance_ratio_.sum():.2%})")

    # Train model
    print("\nTraining model...")
    predictions = np.zeros((len(X_test), 3))

    from sklearn.model_selection import GroupKFold

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        # Find best alpha
        best_alpha = 50
        best_score = float('inf')

        for alpha in [10, 50, 100, 200]:
            gkf = GroupKFold(n_splits=5)
            scores = []
            for train_idx, val_idx in gkf.split(X_train_pca, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_pca[train_idx], y[train_idx])
                pred = model.predict(X_train_pca[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                scores.append(mse)
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        print(f"  {target}: best_alpha={best_alpha}, CV MSE={best_score:.4f}")

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_pca, y)
        predictions[:, i] = model.predict(X_test_pca)

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    print(f"\nangle_std: {np.std(predictions[:, 0]):.6f}")

    # Compare with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions[:, 0],
        'scaled_depth': predictions[:, 1],
        'scaled_left_right': predictions[:, 2]
    })
    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Blend with Sub 219
    for w in [0.1, 0.2, 0.3]:
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"Saved: {blend_file} ({w:.0%} temporal + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")


if __name__ == "__main__":
    main()
