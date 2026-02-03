"""
Test-Time Augmentation (TTA)

Apply augmentations to test samples and average predictions.
This is a data enhancement technique that reduces variance without
needing external data.
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


def extract_features(timeseries, keypoint_idx, release_frame=153):
    """Extract features from a single shot."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Key frames
    frames = [100, 130, 153, 170]
    joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip',
              'left_wrist', 'left_elbow', 'left_shoulder']

    for f in frames:
        for joint in joints:
            pos = get_joint(joint, f)
            for i, coord in enumerate(['x', 'y', 'z']):
                features[f'{joint}_{coord}_f{f}'] = pos[i]

    # Arm configuration at key frames
    for f in frames:
        wrist = get_joint('right_wrist', f)
        elbow = get_joint('right_elbow', f)
        shoulder = get_joint('right_shoulder', f)

        if np.any(wrist) and np.any(elbow) and np.any(shoulder):
            # Elbow angle
            v1 = shoulder - elbow
            v2 = wrist - elbow
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            features[f'elbow_angle_f{f}'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

            # Arm length
            features[f'arm_length_f{f}'] = np.linalg.norm(wrist - shoulder)

    # Velocity features
    for joint in ['right_wrist', 'right_elbow']:
        for f in [130, 153]:
            if f - 10 >= 0:
                pos_before = get_joint(joint, f - 10)
                pos_after = get_joint(joint, f)
                if np.any(pos_before) and np.any(pos_after):
                    vel = pos_after - pos_before
                    features[f'{joint}_vel_f{f}'] = np.linalg.norm(vel)
                    for i, coord in enumerate(['x', 'y', 'z']):
                        features[f'{joint}_vel_{coord}_f{f}'] = vel[i]

    return features


def augment_timeseries(timeseries, aug_type):
    """Apply augmentation to timeseries data."""
    augmented = timeseries.copy()

    if aug_type == 'noise':
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.01, augmented.shape)
        augmented = augmented + noise

    elif aug_type == 'time_shift_forward':
        # Shift forward by 2 frames
        augmented = np.roll(augmented, 2, axis=0)
        augmented[:2] = augmented[2]

    elif aug_type == 'time_shift_backward':
        # Shift backward by 2 frames
        augmented = np.roll(augmented, -2, axis=0)
        augmented[-2:] = augmented[-3]

    elif aug_type == 'scale_up':
        # Scale positions slightly (1.02x)
        augmented = augmented * 1.02

    elif aug_type == 'scale_down':
        # Scale positions slightly (0.98x)
        augmented = augmented * 0.98

    elif aug_type == 'jitter':
        # Random jitter per frame
        for f in range(len(augmented)):
            augmented[f] += np.random.normal(0, 0.005, augmented[f].shape)

    return augmented


def main():
    print("="*80)
    print("TEST-TIME AUGMENTATION")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Augmentation types for training
    train_augs = ['original', 'noise', 'time_shift_forward', 'time_shift_backward']

    # Augmentation types for test (TTA)
    test_augs = ['original', 'noise', 'time_shift_forward', 'time_shift_backward',
                 'scale_up', 'scale_down']

    print(f"\nTraining augmentations: {len(train_augs)}")
    print(f"Test augmentations (TTA): {len(test_augs)}")

    # Extract training features with augmentation
    print("\nExtracting augmented training features...")
    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        for aug_type in train_augs:
            if aug_type == 'original':
                ts = timeseries
            else:
                ts = augment_timeseries(timeseries, aug_type)

            features = extract_features(ts, keypoint_idx)
            train_features.append(features)
            train_targets.append({
                'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
                'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
                'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
            })
            train_players.append(metadata['participant_id'])

    print(f"Augmented training samples: {len(train_features)}")

    # Extract test features with TTA
    print("\nExtracting TTA test features...")
    test_features_tta = {aug: [] for aug in test_augs}
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        test_ids.append(metadata['id'])

        for aug_type in test_augs:
            if aug_type == 'original':
                ts = timeseries
            else:
                ts = augment_timeseries(timeseries, aug_type)

            features = extract_features(ts, keypoint_idx)
            test_features_tta[aug_type].append(features)

    # Create DataFrames
    X_train = pd.DataFrame(train_features).fillna(0)
    X_test_tta = {aug: pd.DataFrame(feats).fillna(0) for aug, feats in test_features_tta.items()}

    common_cols = list(set(X_train.columns) & set(X_test_tta['original'].columns))
    X_train = X_train[common_cols]
    for aug in test_augs:
        X_test_tta[aug] = X_test_tta[aug][common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Features: {len(common_cols)}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_tta_scaled = {aug: scaler.transform(X_test_tta[aug]) for aug in test_augs}

    # Train and predict
    print("\nTraining models...")
    predictions_tta = {aug: np.zeros((len(test_ids), 3)) for aug in test_augs}

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        # Find best alpha using original data CV
        best_alpha = 100
        best_score = float('inf')

        for alpha in [10, 50, 100, 200, 500]:
            gkf = GroupKFold(n_splits=5)
            scores = []
            for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled[train_idx], y[train_idx])
                pred = model.predict(X_train_scaled[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                scores.append(mse)
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        print(f"  {target}: best_alpha={best_alpha}, CV MSE={best_score:.4f}")

        # Train on all augmented data
        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y)

        # Predict for each augmentation
        for aug in test_augs:
            predictions_tta[aug][:, i] = model.predict(X_test_tta_scaled[aug])

    # Average TTA predictions
    print("\nAveraging TTA predictions...")
    predictions_avg = np.zeros((len(test_ids), 3))
    for aug in test_augs:
        predictions_avg += predictions_tta[aug]
    predictions_avg /= len(test_augs)

    # Also try weighted average (less weight for more extreme augmentations)
    weights = {'original': 2.0, 'noise': 1.0, 'time_shift_forward': 0.8,
               'time_shift_backward': 0.8, 'scale_up': 0.6, 'scale_down': 0.6}
    total_weight = sum(weights.values())
    predictions_weighted = np.zeros((len(test_ids), 3))
    for aug in test_augs:
        predictions_weighted += predictions_tta[aug] * weights[aug]
    predictions_weighted /= total_weight

    # Calibrate depth
    predictions_avg[:, 1] = predictions_avg[:, 1] - np.mean(predictions_avg[:, 1]) + 0.5055
    predictions_weighted[:, 1] = predictions_weighted[:, 1] - np.mean(predictions_weighted[:, 1]) + 0.5055

    predictions_avg = np.clip(predictions_avg, 0, 1)
    predictions_weighted = np.clip(predictions_weighted, 0, 1)

    print(f"\nTTA Average - angle_std: {np.std(predictions_avg[:, 0]):.6f}")
    print(f"TTA Weighted - angle_std: {np.std(predictions_weighted[:, 0]):.6f}")

    # Compare with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr_avg = np.corrcoef(predictions_avg[:, 0], sub219['scaled_angle'].values)[0, 1]
    corr_weighted = np.corrcoef(predictions_weighted[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"\nCorrelation with Sub 219:")
    print(f"  TTA Average: {corr_avg:.4f}")
    print(f"  TTA Weighted: {corr_weighted:.4f}")

    # Save submissions
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    # TTA Average
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions_avg[:, 0],
        'scaled_depth': predictions_avg[:, 1],
        'scaled_left_right': predictions_avg[:, 2]
    })
    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved TTA Average: {output_file}")

    # TTA Weighted
    next_num += 1
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions_weighted[:, 0],
        'scaled_depth': predictions_weighted[:, 1],
        'scaled_left_right': predictions_weighted[:, 2]
    })
    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved TTA Weighted: {output_file}")

    # Blend with Sub 219
    for w in [0.2, 0.3, 0.4]:
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions_avg[:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions_avg[:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions_avg[:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"Saved: {blend_file} ({w:.0%} TTA + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")


if __name__ == "__main__":
    main()
