"""
Data Augmentation for Training Enhancement

Generate synthetic training samples to expand our dataset:
1. Mirror shots (simulate left-handed shooters)
2. Add controlled noise to keypoints
3. Time-shift sequences
4. Interpolate between similar shots
5. Scale variations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
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


def mirror_shot(timeseries, keypoint_idx):
    """
    Mirror a shot to simulate left-handed version.
    Swap left/right keypoints and flip Y coordinate.
    """
    mirrored = timeseries.copy()
    n_keypoints = len(keypoint_idx)

    # Pairs of left/right keypoints to swap
    swap_pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_elbow', 'right_elbow'),
        ('left_wrist', 'right_wrist'),
        ('left_hip', 'right_hip'),
        ('left_knee', 'right_knee'),
        ('left_ankle', 'right_ankle'),
        ('left_eye', 'right_eye'),
        ('left_ear', 'right_ear'),
    ]

    for left_name, right_name in swap_pairs:
        if left_name in keypoint_idx and right_name in keypoint_idx:
            left_idx = keypoint_idx[left_name]
            right_idx = keypoint_idx[right_name]

            # Swap the keypoints
            left_data = mirrored[:, left_idx*3:(left_idx+1)*3].copy()
            right_data = mirrored[:, right_idx*3:(right_idx+1)*3].copy()

            mirrored[:, left_idx*3:(left_idx+1)*3] = right_data
            mirrored[:, right_idx*3:(right_idx+1)*3] = left_data

    # Flip Y coordinate for all keypoints
    for name, idx in keypoint_idx.items():
        mirrored[:, idx*3 + 1] *= -1  # Flip Y

    return mirrored


def add_noise(timeseries, noise_std=0.01):
    """Add Gaussian noise to keypoint positions."""
    noise = np.random.normal(0, noise_std, timeseries.shape)
    return timeseries + noise


def time_shift(timeseries, shift_frames=2):
    """Shift the sequence in time."""
    if shift_frames > 0:
        shifted = np.zeros_like(timeseries)
        shifted[shift_frames:] = timeseries[:-shift_frames]
        shifted[:shift_frames] = timeseries[0]
    else:
        shift_frames = abs(shift_frames)
        shifted = np.zeros_like(timeseries)
        shifted[:-shift_frames] = timeseries[shift_frames:]
        shifted[-shift_frames:] = timeseries[-1]
    return shifted


def scale_shot(timeseries, scale_factor=1.05):
    """Scale keypoint positions (simulate different body sizes)."""
    # Scale relative to center of mass
    center = timeseries.mean(axis=1, keepdims=True)
    scaled = center + (timeseries - center) * scale_factor
    return scaled


def interpolate_shots(ts1, ts2, alpha=0.5):
    """Interpolate between two shots."""
    return alpha * ts1 + (1 - alpha) * ts2


def extract_features(timeseries, keypoint_idx):
    """Extract features from timeseries."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Release frame features
    for f in [150, 153, 156]:
        for joint in ['right_wrist', 'right_elbow', 'right_shoulder']:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    # Velocity
    wrist_150 = get_joint("right_wrist", 150)
    wrist_153 = get_joint("right_wrist", 153)
    if np.any(wrist_150) and np.any(wrist_153):
        vel = wrist_153 - wrist_150
        features['wrist_vx'] = vel[0]
        features['wrist_vy'] = vel[1]
        features['wrist_vz'] = vel[2]
        features['wrist_speed'] = np.linalg.norm(vel)

    return features


def main():
    print("="*80)
    print("DATA AUGMENTATION FOR TRAINING ENHANCEMENT")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Load original training data
    print("\nLoading original training data...")
    original_data = []

    for metadata, timeseries in iterate_shots(train=True):
        targets = {
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        }
        original_data.append({
            'timeseries': timeseries,
            'targets': targets,
            'player': metadata['participant_id']
        })

    print(f"Original samples: {len(original_data)}")

    # Generate augmented data
    print("\nGenerating augmented data...")
    augmented_data = []

    for i, sample in enumerate(original_data):
        ts = sample['timeseries']
        targets = sample['targets']
        player = sample['player']

        # 1. Mirror (flip left_right target)
        mirrored_ts = mirror_shot(ts, keypoint_idx)
        mirrored_targets = {
            'angle': targets['angle'],  # angle stays same
            'depth': targets['depth'],  # depth stays same
            'left_right': 1 - targets['left_right']  # flip left_right
        }
        augmented_data.append({
            'timeseries': mirrored_ts,
            'targets': mirrored_targets,
            'player': player,
            'aug_type': 'mirror'
        })

        # 2. Add noise (multiple versions)
        for noise_std in [0.005, 0.01]:
            noisy_ts = add_noise(ts, noise_std)
            augmented_data.append({
                'timeseries': noisy_ts,
                'targets': targets.copy(),
                'player': player,
                'aug_type': f'noise_{noise_std}'
            })

        # 3. Time shift
        for shift in [-2, -1, 1, 2]:
            shifted_ts = time_shift(ts, shift)
            augmented_data.append({
                'timeseries': shifted_ts,
                'targets': targets.copy(),
                'player': player,
                'aug_type': f'shift_{shift}'
            })

        # 4. Scale
        for scale in [0.95, 1.05]:
            scaled_ts = scale_shot(ts, scale)
            augmented_data.append({
                'timeseries': scaled_ts,
                'targets': targets.copy(),
                'player': player,
                'aug_type': f'scale_{scale}'
            })

    print(f"Augmented samples: {len(augmented_data)}")
    print(f"Total samples: {len(original_data) + len(augmented_data)}")

    # 5. Interpolate between similar shots (same player, similar targets)
    print("\nGenerating interpolated samples...")
    interpolated_data = []

    for player_id in set(d['player'] for d in original_data):
        player_samples = [d for d in original_data if d['player'] == player_id]

        for i in range(len(player_samples)):
            for j in range(i+1, min(i+3, len(player_samples))):
                for alpha in [0.25, 0.5, 0.75]:
                    ts_interp = interpolate_shots(
                        player_samples[i]['timeseries'],
                        player_samples[j]['timeseries'],
                        alpha
                    )
                    targets_interp = {
                        'angle': alpha * player_samples[i]['targets']['angle'] + (1-alpha) * player_samples[j]['targets']['angle'],
                        'depth': alpha * player_samples[i]['targets']['depth'] + (1-alpha) * player_samples[j]['targets']['depth'],
                        'left_right': alpha * player_samples[i]['targets']['left_right'] + (1-alpha) * player_samples[j]['targets']['left_right'],
                    }
                    interpolated_data.append({
                        'timeseries': ts_interp,
                        'targets': targets_interp,
                        'player': player_id,
                        'aug_type': 'interpolate'
                    })

    print(f"Interpolated samples: {len(interpolated_data)}")

    # Combine all data
    all_augmented = original_data + augmented_data + interpolated_data
    print(f"\nTotal training samples: {len(all_augmented)}")

    # Extract features
    print("\nExtracting features...")
    train_features = []
    train_targets = []
    train_players = []
    train_weights = []  # Weight original samples higher

    for sample in all_augmented:
        features = extract_features(sample['timeseries'], keypoint_idx)
        train_features.append(features)
        train_targets.append([
            sample['targets']['angle'],
            sample['targets']['depth'],
            sample['targets']['left_right']
        ])
        train_players.append(sample['player'])

        # Original samples get weight 1, augmented get 0.5
        if 'aug_type' not in sample:
            train_weights.append(1.0)
        else:
            train_weights.append(0.5)

    # Also extract original-only for comparison
    original_features = []
    original_targets = []
    original_players = []

    for sample in original_data:
        features = extract_features(sample['timeseries'], keypoint_idx)
        original_features.append(features)
        original_targets.append([
            sample['targets']['angle'],
            sample['targets']['depth'],
            sample['targets']['left_right']
        ])
        original_players.append(sample['player'])

    # Load test data
    print("Loading test data...")
    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    # Convert to DataFrames
    X_train_aug = pd.DataFrame(train_features).fillna(0)
    X_train_orig = pd.DataFrame(original_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train_aug.columns) & set(X_test.columns))
    X_train_aug = X_train_aug[common_cols]
    X_train_orig = X_train_orig[common_cols]
    X_test = X_test[common_cols]

    y_train_aug = np.array(train_targets)
    y_train_orig = np.array(original_targets)
    players_aug = np.array(train_players)
    players_orig = np.array(original_players)
    weights = np.array(train_weights)

    print(f"\nFeatures: {len(common_cols)}")
    print(f"Augmented training shape: {X_train_aug.shape}")
    print(f"Original training shape: {X_train_orig.shape}")

    # Scale
    scaler = StandardScaler()
    X_train_aug_scaled = scaler.fit_transform(X_train_aug)
    X_train_orig_scaled = scaler.transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test)

    # Train and compare
    print("\n" + "="*60)
    print("COMPARING ORIGINAL vs AUGMENTED")
    print("="*60)

    # Original only
    print("\n--- Original Data Only ---")
    cv_scores_orig = []
    for i in range(3):
        gkf = GroupKFold(n_splits=5)
        scores = []
        for train_idx, val_idx in gkf.split(X_train_orig_scaled, y_train_orig[:, i], players_orig):
            model = Ridge(alpha=50)
            model.fit(X_train_orig_scaled[train_idx], y_train_orig[train_idx, i])
            pred = model.predict(X_train_orig_scaled[val_idx])
            mse = np.mean((pred - y_train_orig[val_idx, i])**2)
            scores.append(mse)
        cv_scores_orig.append(np.mean(scores))
    print(f"CV MSE: angle={cv_scores_orig[0]:.4f}, depth={cv_scores_orig[1]:.4f}, lr={cv_scores_orig[2]:.4f}")
    print(f"Mean CV: {np.mean(cv_scores_orig):.6f}")

    # Augmented (weighted)
    print("\n--- Augmented Data (Weighted) ---")
    predictions_aug = np.zeros((len(X_test), 3))

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        model = Ridge(alpha=50)
        model.fit(X_train_aug_scaled, y_train_aug[:, i], sample_weight=weights)
        predictions_aug[:, i] = model.predict(X_test_scaled)

    predictions_aug[:, 1] = predictions_aug[:, 1] - np.mean(predictions_aug[:, 1]) + 0.5055
    predictions_aug = np.clip(predictions_aug, 0, 1)

    print(f"angle_std: {np.std(predictions_aug[:, 0]):.6f}")

    # Original training only
    print("\n--- Original Training Only ---")
    predictions_orig = np.zeros((len(X_test), 3))

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        model = Ridge(alpha=50)
        model.fit(X_train_orig_scaled, y_train_orig[:, i])
        predictions_orig[:, i] = model.predict(X_test_scaled)

    predictions_orig[:, 1] = predictions_orig[:, 1] - np.mean(predictions_orig[:, 1]) + 0.5055
    predictions_orig = np.clip(predictions_orig, 0, 1)

    print(f"angle_std: {np.std(predictions_orig[:, 0]):.6f}")

    # Compare
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr_aug = np.corrcoef(predictions_aug[:, 0], sub219['scaled_angle'].values)[0, 1]
    corr_orig = np.corrcoef(predictions_orig[:, 0], sub219['scaled_angle'].values)[0, 1]

    print(f"\nCorrelation with Sub 219:")
    print(f"  Augmented: {corr_aug:.4f}")
    print(f"  Original: {corr_orig:.4f}")

    # Save submissions
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    for name, preds in [('augmented', predictions_aug), ('original_baseline', predictions_orig)]:
        submission = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': preds[:, 0],
            'scaled_depth': preds[:, 1],
            'scaled_left_right': preds[:, 2]
        })
        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file} ({name})")
        next_num += 1

    # Blend augmented with Sub 219
    print("\n" + "="*60)
    print("BLENDING AUGMENTED WITH SUB 219")
    print("="*60)

    for w in [0.1, 0.2, 0.3]:
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions_aug[:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions_aug[:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions_aug[:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({w:.0%} aug + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")
        next_num += 1


if __name__ == "__main__":
    main()
