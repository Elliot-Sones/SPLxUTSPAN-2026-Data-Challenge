"""
Self-Supervised Motion Embeddings

Learn motion embeddings from ALL shots (train + test) using
self-supervised learning, then use these embeddings for prediction.

This is genuine data enhancement - we use test data structure
(without labels) to learn better representations.
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


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def extract_sequence_representation(timeseries, keypoint_idx):
    """Extract a compact representation of the full sequence."""
    n_frames = len(timeseries)
    n_keypoints = len(keypoint_idx)

    # Sample frames uniformly
    sample_frames = np.linspace(0, n_frames-1, 30).astype(int)

    # Key joints to track
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_elbow', 'left_shoulder',
                  'right_hip', 'left_hip', 'right_knee', 'left_knee']

    representation = []
    for f in sample_frames:
        for joint in key_joints:
            if joint in keypoint_idx:
                idx = keypoint_idx[joint]
                pos = timeseries[f, idx*3:(idx+1)*3]
                representation.extend(pos)

    return np.array(representation)


def learn_motion_basis(all_sequences):
    """Learn a motion basis from all sequences using PCA."""
    # Stack all sequences
    X = np.array(all_sequences)

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA to find motion basis
    n_components = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(X_scaled)

    print(f"  PCA components: {n_components}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    return embeddings, pca, scaler


def contrastive_features(embeddings, is_train):
    """Create features based on similarity to other shots."""
    n_samples = len(embeddings)
    train_embeddings = embeddings[is_train]
    test_embeddings = embeddings[~is_train]

    features = []
    for i, emb in enumerate(embeddings):
        feat = {}

        # Distance to all training samples
        distances = np.linalg.norm(train_embeddings - emb, axis=1)

        # Statistics of distances
        feat['mean_dist_to_train'] = np.mean(distances)
        feat['min_dist_to_train'] = np.min(distances)
        feat['max_dist_to_train'] = np.max(distances)
        feat['std_dist_to_train'] = np.std(distances)

        # K-nearest training samples
        k = 5
        nearest_k = np.argsort(distances)[:k]
        feat['mean_nearest_k_dist'] = np.mean(distances[nearest_k])

        # Embedding coordinates (top components)
        for j in range(min(20, embeddings.shape[1])):
            feat[f'emb_{j}'] = emb[j]

        features.append(feat)

    return pd.DataFrame(features)


def extract_temporal_features(timeseries, keypoint_idx):
    """Extract hand-crafted temporal features."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Key frames and joints
    frames = [100, 130, 153, 170]
    joints = ['right_wrist', 'right_elbow', 'right_shoulder']

    for f in frames:
        for joint in joints:
            pos = get_joint(joint, f)
            for i, coord in enumerate(['x', 'y', 'z']):
                features[f'{joint}_{coord}_f{f}'] = pos[i]

        # Arm configuration
        wrist = get_joint('right_wrist', f)
        elbow = get_joint('right_elbow', f)
        shoulder = get_joint('right_shoulder', f)

        if np.any(wrist) and np.any(elbow) and np.any(shoulder):
            v1 = shoulder - elbow
            v2 = wrist - elbow
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            features[f'elbow_angle_f{f}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            features[f'arm_length_f{f}'] = np.linalg.norm(wrist - shoulder)

    # Velocity at release
    for joint in ['right_wrist']:
        pos_before = get_joint(joint, 148)
        pos_after = get_joint(joint, 158)
        if np.any(pos_before) and np.any(pos_after):
            vel = (pos_after - pos_before) / 10
            features[f'{joint}_vel_release'] = np.linalg.norm(vel)
            for i, coord in enumerate(['x', 'y', 'z']):
                features[f'{joint}_vel_{coord}_release'] = vel[i]

    return features


def main():
    print("="*80)
    print("SELF-SUPERVISED MOTION EMBEDDINGS")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Collect all sequences (train + test) for self-supervised learning
    print("\nCollecting all sequences...")
    all_sequences = []
    all_metadata = []
    is_train = []

    for metadata, timeseries in iterate_shots(train=True):
        seq_repr = extract_sequence_representation(timeseries, keypoint_idx)
        all_sequences.append(seq_repr)
        all_metadata.append(metadata)
        is_train.append(True)

    for metadata, timeseries in iterate_shots(train=False):
        seq_repr = extract_sequence_representation(timeseries, keypoint_idx)
        all_sequences.append(seq_repr)
        all_metadata.append(metadata)
        is_train.append(False)

    is_train = np.array(is_train)
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training: {is_train.sum()}, Test: {(~is_train).sum()}")

    # Learn motion basis from ALL data (self-supervised)
    print("\nLearning motion basis from ALL data...")
    embeddings, pca, seq_scaler = learn_motion_basis(all_sequences)

    # Create contrastive features
    print("\nCreating contrastive features...")
    contrastive_df = contrastive_features(embeddings, is_train)

    # Extract hand-crafted features too
    print("\nExtracting temporal features...")
    temporal_features = []
    for metadata, timeseries in iterate_shots(train=True):
        temporal_features.append(extract_temporal_features(timeseries, keypoint_idx))
    for metadata, timeseries in iterate_shots(train=False):
        temporal_features.append(extract_temporal_features(timeseries, keypoint_idx))

    temporal_df = pd.DataFrame(temporal_features).fillna(0)

    # Combine features
    X_all = pd.concat([contrastive_df.reset_index(drop=True),
                       temporal_df.reset_index(drop=True)], axis=1)

    # Split back to train/test
    X_train = X_all[is_train].reset_index(drop=True)
    X_test = X_all[~is_train].reset_index(drop=True)

    # Get targets
    train_targets = []
    train_players = []
    for i, metadata in enumerate([m for m, is_t in zip(all_metadata, is_train) if is_t]):
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])

    test_ids = [m['id'] for m, is_t in zip(all_metadata, is_train) if not is_t]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"\nCombined features: {X_train.shape[1]}")

    # Feature correlations with angle
    print("\nTop feature correlations with angle:")
    corrs = []
    for col in X_train.columns:
        if X_train[col].std() > 0.001:
            corr = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
            if not np.isnan(corr):
                corrs.append((col, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs[:15]:
        print(f"  {col}: {corr:.4f}")

    # Scale and train
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    print("\nTraining models...")
    predictions = np.zeros((len(X_test), 3))

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

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

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

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
    for w in [0.2, 0.3, 0.4]:
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
        print(f"Saved: {blend_file} ({w:.0%} self-sup + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")


if __name__ == "__main__":
    main()
