"""
Combined Data Enhancement

Combine all three data enhancement approaches:
1. Test-Time Augmentation
2. Skeleton Graph Features
3. Self-Supervised Embeddings

This is a comprehensive data enhancement strategy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


# Skeleton graph edges
SKELETON_EDGES = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
]


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def get_joint(timeseries, keypoint_idx, name, frame):
    """Get joint position at a frame."""
    if name not in keypoint_idx or frame >= len(timeseries):
        return None
    idx = keypoint_idx[name]
    pos = timeseries[frame, idx*3:(idx+1)*3]
    if np.all(pos == 0):
        return None
    return pos


def extract_graph_features(timeseries, keypoint_idx, frame):
    """Extract graph-based features at a frame."""
    features = {}

    # Edge lengths
    for j1, j2 in SKELETON_EDGES:
        p1 = get_joint(timeseries, keypoint_idx, j1, frame)
        p2 = get_joint(timeseries, keypoint_idx, j2, frame)
        if p1 is not None and p2 is not None:
            features[f'edge_{j1}_{j2}_f{frame}'] = np.linalg.norm(p2 - p1)

    # Body twist
    ls = get_joint(timeseries, keypoint_idx, 'left_shoulder', frame)
    rs = get_joint(timeseries, keypoint_idx, 'right_shoulder', frame)
    lh = get_joint(timeseries, keypoint_idx, 'left_hip', frame)
    rh = get_joint(timeseries, keypoint_idx, 'right_hip', frame)

    if all(p is not None for p in [ls, rs, lh, rh]):
        shoulder_vec = rs - ls
        hip_vec = rh - lh
        cos_ang = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8)
        features[f'body_twist_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
        features[f'shoulder_tilt_z_f{frame}'] = rs[2] - ls[2]

    # Arm angles
    wrist = get_joint(timeseries, keypoint_idx, 'right_wrist', frame)
    elbow = get_joint(timeseries, keypoint_idx, 'right_elbow', frame)
    shoulder = get_joint(timeseries, keypoint_idx, 'right_shoulder', frame)

    if all(p is not None for p in [wrist, elbow, shoulder]):
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features[f'elbow_angle_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
        features[f'arm_length_f{frame}'] = np.linalg.norm(wrist - shoulder)

    return features


def extract_position_features(timeseries, keypoint_idx, frame):
    """Extract raw position features at a frame."""
    features = {}
    joints = ['right_wrist', 'right_elbow', 'right_shoulder']

    for joint in joints:
        pos = get_joint(timeseries, keypoint_idx, joint, frame)
        if pos is not None:
            for i, coord in enumerate(['x', 'y', 'z']):
                features[f'{joint}_{coord}_f{frame}'] = pos[i]

    return features


def extract_velocity_features(timeseries, keypoint_idx, frame, delta=5):
    """Extract velocity features around a frame."""
    features = {}
    joints = ['right_wrist', 'right_elbow']

    for joint in joints:
        pos_before = get_joint(timeseries, keypoint_idx, joint, frame - delta)
        pos_after = get_joint(timeseries, keypoint_idx, joint, frame + delta)

        if pos_before is not None and pos_after is not None:
            vel = (pos_after - pos_before) / (2 * delta)
            features[f'{joint}_vel_f{frame}'] = np.linalg.norm(vel)
            for i, coord in enumerate(['x', 'y', 'z']):
                features[f'{joint}_vel_{coord}_f{frame}'] = vel[i]

    return features


def extract_sequence_representation(timeseries, keypoint_idx):
    """Extract a compact sequence representation for self-supervised learning."""
    n_frames = len(timeseries)
    sample_frames = np.linspace(0, n_frames-1, 24).astype(int)
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_elbow', 'left_shoulder',
                  'right_hip', 'left_hip']

    representation = []
    for f in sample_frames:
        for joint in key_joints:
            if joint in keypoint_idx:
                idx = keypoint_idx[joint]
                pos = timeseries[f, idx*3:(idx+1)*3]
                representation.extend(pos)

    return np.array(representation)


def extract_all_features(timeseries, keypoint_idx):
    """Extract combined features from a single shot."""
    features = {}

    # Key frames
    frames = [80, 100, 120, 140, 153, 160, 170]

    for frame in frames:
        # Graph features (strongest correlations)
        features.update(extract_graph_features(timeseries, keypoint_idx, frame))

        # Position features
        features.update(extract_position_features(timeseries, keypoint_idx, frame))

        # Velocity features
        if frame >= 10 and frame < len(timeseries) - 10:
            features.update(extract_velocity_features(timeseries, keypoint_idx, frame))

    # Temporal changes
    for f1, f2 in [(100, 153), (120, 160), (140, 170)]:
        key = f'elbow_angle_f{f1}'
        key2 = f'elbow_angle_f{f2}'
        if key in features and key2 in features:
            features[f'elbow_angle_change_{f1}_{f2}'] = features[key2] - features[key]

    return features


def augment_timeseries(timeseries, aug_type):
    """Apply augmentation to timeseries."""
    augmented = timeseries.copy()

    if aug_type == 'noise':
        augmented = augmented + np.random.normal(0, 0.01, augmented.shape)
    elif aug_type == 'time_shift_forward':
        augmented = np.roll(augmented, 2, axis=0)
        augmented[:2] = augmented[2]
    elif aug_type == 'time_shift_backward':
        augmented = np.roll(augmented, -2, axis=0)
        augmented[-2:] = augmented[-3]
    elif aug_type == 'scale':
        augmented = augmented * np.random.uniform(0.98, 1.02)

    return augmented


def main():
    print("="*80)
    print("COMBINED DATA ENHANCEMENT")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Collect all data
    print("\nCollecting all data...")
    all_timeseries = []
    all_metadata = []
    is_train = []

    for metadata, timeseries in iterate_shots(train=True):
        all_timeseries.append(timeseries)
        all_metadata.append(metadata)
        is_train.append(True)

    for metadata, timeseries in iterate_shots(train=False):
        all_timeseries.append(timeseries)
        all_metadata.append(metadata)
        is_train.append(False)

    is_train = np.array(is_train)
    print(f"Total samples: {len(all_timeseries)} (train: {is_train.sum()}, test: {(~is_train).sum()})")

    # Self-supervised: learn motion embeddings from ALL data
    print("\nLearning self-supervised embeddings...")
    all_seq_repr = [extract_sequence_representation(ts, keypoint_idx) for ts in all_timeseries]
    all_seq_repr = np.array(all_seq_repr)
    all_seq_repr = np.nan_to_num(all_seq_repr, nan=0.0)

    seq_scaler = StandardScaler()
    all_seq_scaled = seq_scaler.fit_transform(all_seq_repr)

    n_components = min(30, len(all_seq_scaled) - 1)
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(all_seq_scaled)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Extract features with augmentation for training
    print("\nExtracting features...")
    augmentations = ['original', 'noise', 'time_shift_forward', 'time_shift_backward']

    train_features = []
    train_targets = []
    train_players = []

    for i, (metadata, ts) in enumerate(zip(all_metadata, all_timeseries)):
        if not is_train[i]:
            continue

        for aug_type in augmentations:
            if aug_type == 'original':
                ts_aug = ts
            else:
                ts_aug = augment_timeseries(ts, aug_type)

            features = extract_all_features(ts_aug, keypoint_idx)

            # Add self-supervised embeddings (from original, not augmented)
            for j in range(min(15, n_components)):
                features[f'emb_{j}'] = embeddings[i, j]

            train_features.append(features)
            train_targets.append({
                'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
                'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
                'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
            })
            train_players.append(metadata['participant_id'])

    print(f"Augmented training samples: {len(train_features)}")

    # Test features with TTA
    test_augs = ['original', 'noise', 'time_shift_forward', 'time_shift_backward']
    test_features_tta = {aug: [] for aug in test_augs}
    test_ids = []
    test_indices = []

    for i, (metadata, ts) in enumerate(zip(all_metadata, all_timeseries)):
        if is_train[i]:
            continue

        test_ids.append(metadata['id'])
        test_indices.append(i)

        for aug_type in test_augs:
            if aug_type == 'original':
                ts_aug = ts
            else:
                ts_aug = augment_timeseries(ts, aug_type)

            features = extract_all_features(ts_aug, keypoint_idx)

            # Add embeddings
            for j in range(min(15, n_components)):
                features[f'emb_{j}'] = embeddings[i, j]

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

    print(f"Combined features: {len(common_cols)}")

    # Top correlations
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

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_tta_scaled = {aug: scaler.transform(X_test_tta[aug]) for aug in test_augs}

    # Train
    print("\nTraining models...")
    predictions_tta = {aug: np.zeros((len(test_ids), 3)) for aug in test_augs}

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        best_alpha = 100
        best_score = float('inf')

        for alpha in [50, 100, 200, 500]:
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

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y)

        for aug in test_augs:
            predictions_tta[aug][:, i] = model.predict(X_test_tta_scaled[aug])

    # Average TTA predictions
    predictions = np.zeros((len(test_ids), 3))
    for aug in test_augs:
        predictions += predictions_tta[aug]
    predictions /= len(test_augs)

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
    print(f"\nSaved Combined: {output_file}")

    # Blends with Sub 219
    for w in [0.15, 0.25, 0.35, 0.45]:
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
        print(f"Saved: {blend_file} ({w:.0%} combined + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")


if __name__ == "__main__":
    main()
