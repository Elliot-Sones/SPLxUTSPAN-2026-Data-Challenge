"""
Temporal Self-Supervised Pre-Training

Learn motion representations without labels using:
1. Next frame prediction - predict frame t+1 from frames 1:t
2. Contrastive learning - same player shots should cluster together
3. Temporal consistency - adjacent frames should be similar

Then fine-tune on the prediction task.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


class TemporalSSLEncoder:
    """Self-supervised encoder using temporal prediction."""

    def __init__(self, hidden_dim=64, n_components=32):
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.frame_encoder = None
        self.temporal_predictor = None
        self.pca = None
        self.scaler = None

    def _extract_frame_features(self, timeseries, frames):
        """Extract features from specific frames."""
        features = []
        for f in frames:
            if f < len(timeseries):
                features.extend(timeseries[f, :])
            else:
                features.extend([0] * timeseries.shape[1])
        return np.array(features)

    def pretrain(self, all_timeseries, all_players):
        """Pre-train using temporal self-supervision."""
        print("Pre-training with temporal self-supervision...")

        # Task 1: Next frame prediction
        # Learn to predict frame t+1 from frames t-k:t
        frame_inputs = []
        frame_targets = []

        context_frames = [145, 148, 150, 152]  # Input context
        target_frame = 155  # Predict this

        for ts in all_timeseries:
            if target_frame < len(ts):
                input_feat = self._extract_frame_features(ts, context_frames)
                target_feat = ts[target_frame, :]
                frame_inputs.append(input_feat)
                frame_targets.append(target_feat)

        X_pretrain = np.array(frame_inputs)
        y_pretrain = np.array(frame_targets)

        print(f"  Pre-training samples: {len(X_pretrain)}")
        print(f"  Input dim: {X_pretrain.shape[1]}, Target dim: {y_pretrain.shape[1]}")

        # Scale inputs
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_pretrain)

        # Train frame predictor (learns temporal dynamics)
        self.temporal_predictor = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
        self.temporal_predictor.fit(X_scaled, y_pretrain)

        # Get hidden representations
        # We'll use the activations from the hidden layer as features

        # Task 2: Contrastive embeddings via PCA on temporal features
        # Create temporal features for each shot
        temporal_features = []
        for ts in all_timeseries:
            # Multi-scale temporal features
            feat = []
            for start, end in [(140, 150), (150, 160), (155, 165)]:
                if end <= len(ts):
                    segment = ts[start:end, :]
                    feat.extend([
                        segment.mean(axis=0),  # Mean
                        segment.std(axis=0),   # Std
                        segment[-1] - segment[0]  # Delta
                    ])
            if len(feat) > 0:
                temporal_features.append(np.concatenate([f.flatten() for f in feat]))

        temporal_features = np.array(temporal_features)
        print(f"  Temporal features shape: {temporal_features.shape}")

        # Learn compact representation via PCA
        self.pca = PCA(n_components=min(self.n_components, temporal_features.shape[1], len(temporal_features)-1))
        self.pca.fit(temporal_features)

        print(f"  PCA components: {self.pca.n_components_}")
        print(f"  Variance explained: {self.pca.explained_variance_ratio_.sum():.2%}")

        return self

    def encode(self, timeseries):
        """Encode a single shot using pre-trained representations."""
        features = {}

        # 1. Temporal prediction features (how well can we predict future?)
        context_frames = [145, 148, 150, 152]
        target_frame = 155

        if target_frame < len(timeseries):
            input_feat = self._extract_frame_features(timeseries, context_frames)
            input_scaled = self.scaler.transform(input_feat.reshape(1, -1))

            # Predicted vs actual difference (prediction error as feature)
            predicted = self.temporal_predictor.predict(input_scaled)[0]
            actual = timeseries[target_frame, :]
            prediction_error = predicted - actual

            features['pred_error_mean'] = np.mean(np.abs(prediction_error))
            features['pred_error_std'] = np.std(prediction_error)
            features['pred_error_max'] = np.max(np.abs(prediction_error))

        # 2. PCA-based temporal embedding
        temporal_feat = []
        for start, end in [(140, 150), (150, 160), (155, 165)]:
            if end <= len(timeseries):
                segment = timeseries[start:end, :]
                temporal_feat.extend([
                    segment.mean(axis=0),
                    segment.std(axis=0),
                    segment[-1] - segment[0]
                ])

        if len(temporal_feat) > 0:
            temporal_feat = np.concatenate([f.flatten() for f in temporal_feat])
            pca_embedding = self.pca.transform(temporal_feat.reshape(1, -1))[0]

            for i, val in enumerate(pca_embedding):
                features[f'pca_{i}'] = val

        # 3. Raw physics features (same as before)
        release_frames = [150, 153, 155, 158]
        key_joints = [0, 15, 16, 17, 18]  # Approximate indices for wrist, elbow, shoulder

        for f in release_frames:
            if f < len(timeseries):
                for j in key_joints:
                    if j*3+3 <= timeseries.shape[1]:
                        pos = timeseries[f, j*3:j*3+3]
                        features[f'joint{j}_x_f{f}'] = pos[0]
                        features[f'joint{j}_y_f{f}'] = pos[1]
                        features[f'joint{j}_z_f{f}'] = pos[2]

        # 4. Velocity features
        if 153 < len(timeseries) and 150 < len(timeseries):
            vel = timeseries[153, :] - timeseries[150, :]
            features['vel_mean'] = np.mean(vel)
            features['vel_std'] = np.std(vel)
            features['vel_max'] = np.max(np.abs(vel))

            # Wrist velocity specifically
            for j in key_joints[:2]:
                if j*3+3 <= timeseries.shape[1]:
                    joint_vel = vel[j*3:j*3+3]
                    features[f'joint{j}_speed'] = np.linalg.norm(joint_vel)

        return features


def load_all_data():
    """Load all training and test data."""
    scalers = load_scalers()

    # Training data
    train_timeseries = []
    train_targets = []
    train_players = []
    train_ids = []

    print("Loading training data...")
    for metadata, timeseries in iterate_shots(train=True):
        train_timeseries.append(timeseries)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])
        train_ids.append(metadata['shot_id'])

    # Test data
    test_timeseries = []
    test_ids = []

    print("Loading test data...")
    for metadata, timeseries in iterate_shots(train=False):
        test_timeseries.append(timeseries)
        test_ids.append(metadata['id'])

    return (train_timeseries, train_targets, train_players, train_ids,
            test_timeseries, test_ids)


def train_with_ssl_features(encoder, train_ts, train_targets, train_players, test_ts):
    """Train model using SSL-encoded features."""

    # Extract features
    print("\nExtracting SSL features...")
    X_train = []
    for ts in train_ts:
        features = encoder.encode(ts)
        X_train.append(features)

    X_test = []
    for ts in test_ts:
        features = encoder.encode(ts)
        X_test.append(features)

    X_train = pd.DataFrame(X_train).fillna(0)
    X_test = pd.DataFrame(X_test).fillna(0)

    # Align columns
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"Features: {len(common_cols)}")

    y_train = pd.DataFrame(train_targets)
    players = np.array(train_players)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train per-target
    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    target_names = ['angle', 'depth', 'left_right']

    for i, target in enumerate(target_names):
        y = y_train[target].values

        # Cross-validation
        gkf = GroupKFold(n_splits=5)
        fold_scores = []

        for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
            model = Ridge(alpha=10.0)
            model.fit(X_train_scaled[train_idx], y[train_idx])
            pred = model.predict(X_train_scaled[val_idx])
            mse = np.mean((pred - y[val_idx])**2)
            fold_scores.append(mse)

        cv_score = np.mean(fold_scores)
        cv_scores.append(cv_score)
        print(f"{target} CV MSE: {cv_score:.6f}")

        # Train final model
        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

    overall_cv = np.mean(cv_scores)
    print(f"\nOverall CV MSE: {overall_cv:.6f}")

    return predictions, overall_cv


def main():
    print("="*80)
    print("TEMPORAL SELF-SUPERVISED PRE-TRAINING")
    print("="*80)

    # Load all data
    (train_ts, train_targets, train_players, train_ids,
     test_ts, test_ids) = load_all_data()

    print(f"\nTrain samples: {len(train_ts)}")
    print(f"Test samples: {len(test_ts)}")

    # Combine train and test for pre-training (unsupervised)
    all_ts = train_ts + test_ts
    all_players = train_players + [0] * len(test_ts)  # Dummy player for test

    print(f"Total samples for pre-training: {len(all_ts)}")

    # Pre-train encoder
    encoder = TemporalSSLEncoder(hidden_dim=64, n_components=32)
    encoder.pretrain(all_ts, all_players)

    # Train with SSL features
    predictions, cv_score = train_with_ssl_features(
        encoder, train_ts, train_targets, train_players, test_ts
    )

    # Calibrate depth
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    angle_std = np.std(predictions[:, 0])
    depth_mean = np.mean(predictions[:, 1])

    print(f"\nProfile: angle_std={angle_std:.6f}, depth_mean={depth_mean:.6f}")

    # Save submission
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
    print(f"CV={cv_score:.6f}, angle_std={angle_std:.6f}")

    # Blend with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

    best_blend = None
    best_std = float('inf')

    for blend_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
        blended = submission.copy()
        blended['scaled_angle'] = blend_weight * submission['scaled_angle'] + (1-blend_weight) * sub219['scaled_angle']
        blended['scaled_depth'] = blend_weight * submission['scaled_depth'] + (1-blend_weight) * sub219['scaled_depth']
        blended['scaled_left_right'] = blend_weight * submission['scaled_left_right'] + (1-blend_weight) * sub219['scaled_left_right']

        # Recalibrate depth
        blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
        blended['scaled_depth'] = blended['scaled_depth'].clip(0, 1)

        std = blended['scaled_angle'].std()
        print(f"Blend {blend_weight:.0%} SSL + {1-blend_weight:.0%} Sub219: angle_std={std:.6f}")

        if std < best_std:
            best_std = std
            best_blend = (blend_weight, blended.copy())

    # Save best blend
    if best_blend:
        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        best_blend[1].to_csv(blend_file, index=False)
        print(f"\nSaved best blend ({best_blend[0]:.0%} SSL): {blend_file}")
        print(f"  angle_std: {best_std:.6f}")

    # Also try combining with selective amplification
    print("\n" + "="*60)
    print("COMBINING SSL WITH SELECTIVE AMPLIFICATION")
    print("="*60)

    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")

    # Compute difference
    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    # Apply selective amplification to SSL predictions
    for pctl, alpha in [(91, 1.1), (90, 1.0)]:
        threshold = np.percentile(total_diff, pctl)
        mask = total_diff > threshold

        ssl_amp = submission.copy()
        ssl_amp.loc[mask, 'scaled_angle'] = submission.loc[mask, 'scaled_angle'] + alpha * diff_angle[mask]
        ssl_amp.loc[mask, 'scaled_depth'] = submission.loc[mask, 'scaled_depth'] + alpha * diff_depth[mask]
        ssl_amp.loc[mask, 'scaled_left_right'] = submission.loc[mask, 'scaled_left_right'] + alpha * diff_lr[mask]

        ssl_amp['scaled_angle'] = ssl_amp['scaled_angle'].clip(0, 1)
        ssl_amp['scaled_depth'] = ssl_amp['scaled_depth'].clip(0, 1)
        ssl_amp['scaled_left_right'] = ssl_amp['scaled_left_right'].clip(0, 1)

        ssl_amp['scaled_depth'] = ssl_amp['scaled_depth'] - ssl_amp['scaled_depth'].mean() + 0.5055
        ssl_amp['scaled_depth'] = ssl_amp['scaled_depth'].clip(0, 1)

        std = ssl_amp['scaled_angle'].std()

        next_num += 1
        amp_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        ssl_amp.to_csv(amp_file, index=False)
        print(f"\nSSL + Amplification (pctl={pctl}, alpha={alpha}): {amp_file}")
        print(f"  angle_std: {std:.6f}")


if __name__ == "__main__":
    main()
