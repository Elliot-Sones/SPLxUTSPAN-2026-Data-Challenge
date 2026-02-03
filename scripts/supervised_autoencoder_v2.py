"""
Supervised Autoencoder V2 - Massive Feature Space + Pseudo-Labels

Key ideas:
1. Use FULL feature space (raw frames + engineered features)
2. Fabricate data to increase quantity
3. Pseudo-label ALL data using Sub 219
4. Train supervised autoencoder PER PLAYER, PER TARGET
5. Compression learns what matters for each specific case

This addresses the data quantity problem by:
- Using 10,000+ samples for training (not just 345)
- Making compression supervised (knows about targets)
- Specializing per player and per target
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def extract_massive_features(timeseries):
    """
    Extract MASSIVE feature set from raw timeseries.

    Raw data: 240 frames × 51 features (17 keypoints × 3 coords) = 12,240 values
    Plus engineered features across key frames.
    """
    features = []
    n_frames = len(timeseries)
    n_keypoints = timeseries.shape[1] // 3 if len(timeseries.shape) > 1 else 17

    # === RAW FRAME DATA (sampled to reduce dimensionality) ===
    # Sample 24 frames across the motion
    sample_frames = np.linspace(0, min(n_frames-1, 239), 24).astype(int)
    for f in sample_frames:
        if f < n_frames:
            features.extend(timeseries[f].flatten())
        else:
            features.extend([0] * timeseries.shape[1])

    # === KEY FRAME DETAILED FEATURES ===
    key_frames = [80, 100, 120, 140, 150, 153, 156, 160, 170, 180]

    for f in key_frames:
        if f >= n_frames:
            continue
        frame_data = timeseries[f]

        # All positions
        features.extend(frame_data.flatten())

        # Velocities (if we have adjacent frames)
        if f > 0 and f < n_frames - 1:
            vel = timeseries[min(f+2, n_frames-1)] - timeseries[max(f-2, 0)]
            features.extend(vel.flatten())

        # Accelerations
        if f > 1 and f < n_frames - 2:
            acc = timeseries[min(f+2, n_frames-1)] - 2*timeseries[f] + timeseries[max(f-2, 0)]
            features.extend(acc.flatten())

    # === TEMPORAL STATISTICS ===
    # Mean, std, min, max across all frames
    features.extend(np.mean(timeseries, axis=0).flatten())
    features.extend(np.std(timeseries, axis=0).flatten())
    features.extend(np.min(timeseries, axis=0).flatten())
    features.extend(np.max(timeseries, axis=0).flatten())

    # === RELEASE WINDOW FOCUS (frames 145-165) ===
    release_start = min(145, n_frames-1)
    release_end = min(165, n_frames)
    if release_end > release_start:
        release_data = timeseries[release_start:release_end]
        features.extend(np.mean(release_data, axis=0).flatten())
        features.extend(np.std(release_data, axis=0).flatten())
        # Trajectory through release
        if len(release_data) > 1:
            release_vel = release_data[-1] - release_data[0]
            features.extend(release_vel.flatten())

    features = np.array(features, dtype=np.float32)
    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def fabricate_motion(timeseries, method='noise'):
    """Fabricate synthetic motion data."""
    if method == 'noise':
        return timeseries + np.random.normal(0, 0.02, timeseries.shape)
    elif method == 'scale':
        return timeseries * np.random.uniform(0.95, 1.05)
    elif method == 'time_warp':
        n = len(timeseries)
        warp = 1 + np.random.uniform(-0.1, 0.1)
        indices = np.clip((np.arange(n) * warp).astype(int), 0, n-1)
        return timeseries[indices]
    elif method == 'jitter':
        jitter = np.random.normal(0, 0.01, timeseries.shape)
        for i in range(1, len(jitter)):
            jitter[i] = 0.5 * jitter[i] + 0.5 * jitter[i-1]
        return timeseries + jitter
    return timeseries


def interpolate_motion(ts1, ts2, alpha):
    """Interpolate between two motions."""
    min_len = min(len(ts1), len(ts2))
    return ts1[:min_len] * (1-alpha) + ts2[:min_len] * alpha


class SupervisedAutoencoder(nn.Module):
    """
    Supervised Autoencoder - trains reconstruction + prediction TOGETHER.

    The encoder learns compression that's good for BOTH:
    1. Reconstructing the input (unsupervised signal)
    2. Predicting the target (supervised signal)
    """
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()

        # Encoder: compress massive input to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, bottleneck_dim),
        )

        # Decoder: reconstruct input from bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

        # Prediction head: predict target from bottleneck
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        prediction = self.predictor(z)
        return reconstruction, prediction, z


def train_supervised_autoencoder(X, y, epochs=100, batch_size=64, bottleneck_dim=64,
                                  recon_weight=0.3, pred_weight=1.0):
    """
    Train supervised autoencoder with BOTH losses simultaneously.

    Total Loss = recon_weight * reconstruction_loss + pred_weight * prediction_loss
    """
    # Handle NaN/Inf in input
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0)

    input_dim = X.shape[1]

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y.reshape(-1, 1)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SupervisedAutoencoder(input_dim, bottleneck_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    recon_criterion = nn.MSELoss()
    pred_criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = model.state_dict().copy()  # Initialize with starting state
    patience = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_pred = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            recon, pred, _ = model(batch_x)

            recon_loss = recon_criterion(recon, batch_x)
            pred_loss = pred_criterion(pred, batch_y)

            # Combined loss - both signals train the encoder
            loss = recon_weight * recon_loss + pred_weight * pred_loss

            # Skip if NaN
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_pred += pred_loss.item()

        if len(loader) == 0:
            break

        avg_loss = total_loss / len(loader)

        if not np.isnan(avg_loss) and avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1

        scheduler.step(avg_loss if not np.isnan(avg_loss) else 1.0)

        if patience > 20:
            break

        if epoch % 25 == 0:
            print(f"    Epoch {epoch}: total={avg_loss:.4f}, recon={total_recon/len(loader):.4f}, pred={total_pred/len(loader):.4f}")

    model.load_state_dict(best_state)
    return model


def predict_with_model(model, X):
    """Get predictions from trained model."""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X).to(DEVICE)
        _, pred, _ = model(x)
    return pred.cpu().numpy().flatten()


def main():
    print("="*70)
    print("SUPERVISED AUTOENCODER V2")
    print("Massive Features + Pseudo-Labels + Per-Player Per-Target")
    print("="*70)

    scalers = load_scalers()

    # Load all data
    print("\nLoading data...")
    all_data = []

    for metadata, timeseries in iterate_shots(train=True):
        all_data.append({
            'timeseries': timeseries,
            'metadata': metadata,
            'is_train': True,
            'player': metadata['participant_id'],
            'targets': {
                'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
                'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
                'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
            }
        })

    for metadata, timeseries in iterate_shots(train=False):
        all_data.append({
            'timeseries': timeseries,
            'metadata': metadata,
            'is_train': False,
            'player': metadata['participant_id'],
            'id': metadata['id'],
            'targets': None
        })

    n_train = sum(1 for d in all_data if d['is_train'])
    n_test = sum(1 for d in all_data if not d['is_train'])
    print(f"Real data: {len(all_data)} (train={n_train}, test={n_test})")

    # Load Sub 219 for pseudo-labels
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    test_ids = [d['id'] for d in all_data if not d['is_train']]

    # Assign pseudo-labels to test data
    for d in all_data:
        if not d['is_train']:
            idx = test_ids.index(d['id'])
            d['targets'] = {
                'angle': sub219['scaled_angle'].iloc[idx],
                'depth': sub219['scaled_depth'].iloc[idx],
                'left_right': sub219['scaled_left_right'].iloc[idx]
            }

    # Fabricate data with pseudo-labels
    print("\nFabricating data with pseudo-labels...")
    n_fabricate = 5000
    fabricated = []

    methods = ['noise', 'scale', 'time_warp', 'jitter']

    for i in range(n_fabricate):
        # Pick random source
        src = all_data[np.random.randint(len(all_data))]
        method = np.random.choice(methods)

        new_ts = fabricate_motion(src['timeseries'], method)

        # Interpolate sometimes
        if np.random.random() < 0.3:
            src2 = all_data[np.random.randint(len(all_data))]
            alpha = np.random.uniform(0.2, 0.8)
            try:
                new_ts = interpolate_motion(new_ts, src2['timeseries'], alpha)
                # Interpolate targets too
                new_targets = {
                    k: src['targets'][k] * (1-alpha) + src2['targets'][k] * alpha
                    for k in ['angle', 'depth', 'left_right']
                }
            except:
                new_targets = src['targets'].copy()
        else:
            # Small perturbation to targets (pseudo-label noise)
            new_targets = {
                k: src['targets'][k] + np.random.normal(0, 0.01)
                for k in ['angle', 'depth', 'left_right']
            }

        fabricated.append({
            'timeseries': new_ts,
            'player': src['player'],
            'targets': new_targets,
            'is_fabricated': True
        })

    print(f"Fabricated: {len(fabricated)}")

    # Combine all data
    all_with_fab = all_data + fabricated
    print(f"Total samples: {len(all_with_fab)}")

    # Extract massive features
    print("\nExtracting massive features...")
    for d in all_with_fab:
        d['features'] = extract_massive_features(d['timeseries'])

    feature_dim = len(all_with_fab[0]['features'])
    print(f"Feature dimension: {feature_dim}")

    # Get unique players
    players = sorted(set(d['player'] for d in all_with_fab))
    print(f"Players: {players}")

    # Train per-player, per-target models
    print("\n" + "="*70)
    print("TRAINING PER-PLAYER, PER-TARGET MODELS")
    print("="*70)

    targets = ['angle', 'depth', 'left_right']
    models = {}
    scalers_dict = {}

    for player in players:
        models[player] = {}
        scalers_dict[player] = {}

        # Get all data for this player
        player_data = [d for d in all_with_fab if d['player'] == player]

        if len(player_data) < 50:
            print(f"\nPlayer {player}: Only {len(player_data)} samples, skipping...")
            continue

        print(f"\nPlayer {player}: {len(player_data)} samples")

        # Prepare features
        X_player = np.array([d['features'] for d in player_data])

        # Scale features
        scaler = StandardScaler()
        X_player_scaled = scaler.fit_transform(X_player)
        scalers_dict[player]['features'] = scaler

        for target in targets:
            print(f"  Target: {target}")

            y_player = np.array([d['targets'][target] for d in player_data])

            # Train supervised autoencoder
            model = train_supervised_autoencoder(
                X_player_scaled, y_player,
                epochs=100,
                batch_size=min(64, len(player_data)//4),
                bottleneck_dim=32,
                recon_weight=0.2,
                pred_weight=1.0
            )

            models[player][target] = model

    # Generate predictions for test data
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)

    test_data = [d for d in all_data if not d['is_train']]
    predictions = {t: [] for t in targets}

    for d in test_data:
        player = d['player']
        features = d['features'].reshape(1, -1)

        if player in scalers_dict and 'features' in scalers_dict[player]:
            features_scaled = scalers_dict[player]['features'].transform(features)

            for target in targets:
                if player in models and target in models[player]:
                    pred = predict_with_model(models[player][target], features_scaled)
                    predictions[target].append(pred[0])
                else:
                    # Fallback to Sub 219
                    idx = test_ids.index(d['id'])
                    predictions[target].append(sub219[f'scaled_{target}'].iloc[idx])
        else:
            # Fallback to Sub 219
            idx = test_ids.index(d['id'])
            for target in targets:
                predictions[target].append(sub219[f'scaled_{target}'].iloc[idx])

    # Create submission
    predictions_arr = np.column_stack([predictions[t] for t in targets])

    # Calibrate depth
    predictions_arr[:, 1] = predictions_arr[:, 1] - np.mean(predictions_arr[:, 1]) + 0.5055
    predictions_arr = np.clip(predictions_arr, 0, 1)

    angle_std = np.std(predictions_arr[:, 0])
    corr = np.corrcoef(predictions_arr[:, 0], sub219['scaled_angle'].values)[0, 1]

    print(f"\nResults:")
    print(f"  angle_std: {angle_std:.6f} (target: 0.137162)")
    print(f"  Correlation with Sub 219: {corr:.4f}")

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions_arr[:, 0],
        'scaled_depth': predictions_arr[:, 1],
        'scaled_left_right': predictions_arr[:, 2]
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"  Description: Supervised AE V2 - Per-player per-target, {feature_dim} features, {n_fabricate} fabricated")

    # Blends with Sub 219
    for w in [0.1, 0.2, 0.3, 0.5]:
        next_num += 1
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions_arr[:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions_arr[:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions_arr[:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend.to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
        print(f"Saved: submission_{next_num}.csv ({w:.0%} SAE + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Feature dimension: {feature_dim}")
    print(f"Total training samples: {len(all_with_fab)}")
    print(f"Per-player per-target models trained")
    print(f"Compression learns what matters for EACH player's EACH target")


if __name__ == "__main__":
    main()
