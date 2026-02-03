"""
Supervised Autoencoder V3 - Optimized for Speed

Same concept but practical:
1. Reduced feature space (~2000 features)
2. Smaller network
3. Fewer fabricated samples
4. Per-target models (not per-player to save time)
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


def extract_features(timeseries):
    """Extract meaningful features - ~2000 total."""
    features = []
    n_frames = len(timeseries)

    # Sample 12 key frames
    key_frames = [50, 80, 100, 120, 140, 148, 150, 153, 156, 160, 170, 180]

    for f in key_frames:
        if f >= n_frames:
            features.extend([0] * timeseries.shape[1])
            continue

        # Raw positions
        features.extend(timeseries[f].flatten())

        # Velocities
        if f > 2 and f < n_frames - 2:
            vel = timeseries[min(f+2, n_frames-1)] - timeseries[max(f-2, 0)]
            features.extend(vel.flatten())

    # Global stats
    features.extend(np.mean(timeseries, axis=0).flatten())
    features.extend(np.std(timeseries, axis=0).flatten())

    # Release window (145-165)
    rs, re = min(145, n_frames-1), min(165, n_frames)
    if re > rs:
        release = timeseries[rs:re]
        features.extend(np.mean(release, axis=0).flatten())
        if len(release) > 1:
            features.extend((release[-1] - release[0]).flatten())

    return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0)


def fabricate(ts, method):
    if method == 'noise':
        return ts + np.random.normal(0, 0.02, ts.shape)
    elif method == 'scale':
        return ts * np.random.uniform(0.95, 1.05)
    elif method == 'warp':
        n = len(ts)
        idx = np.clip((np.arange(n) * np.random.uniform(0.9, 1.1)).astype(int), 0, n-1)
        return ts[idx]
    return ts


class SupervisedAE(nn.Module):
    def __init__(self, input_dim, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), self.predictor(z), z


def train_model(X, y, epochs=80, batch_size=64, bottleneck=32):
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.5)

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y.reshape(-1, 1)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SupervisedAE(X.shape[1], bottleneck).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss = float('inf')
    best_state = model.state_dict().copy()

    for epoch in range(epochs):
        model.train()
        total = 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            recon, pred, _ = model(bx)

            loss = 0.2 * nn.MSELoss()(recon, bx) + nn.MSELoss()(pred, by)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / max(len(loader), 1)
        if not np.isnan(avg) and avg < best_loss:
            best_loss = avg
            best_state = model.state_dict().copy()

        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: loss={avg:.4f}")

    model.load_state_dict(best_state)
    return model


def main():
    print("="*60)
    print("SUPERVISED AUTOENCODER V3 - Optimized")
    print("="*60)

    scalers = load_scalers()

    # Load data
    print("\nLoading data...")
    data = []

    for meta, ts in iterate_shots(train=True):
        data.append({
            'ts': ts, 'train': True, 'player': meta['participant_id'],
            'targets': {
                'angle': scalers['angle'].transform([[meta['angle']]])[0, 0],
                'depth': scalers['depth'].transform([[meta['depth']]])[0, 0],
                'left_right': scalers['left_right'].transform([[meta['left_right']]])[0, 0]
            }
        })

    for meta, ts in iterate_shots(train=False):
        data.append({
            'ts': ts, 'train': False, 'player': meta['participant_id'],
            'id': meta['id'], 'targets': None
        })

    print(f"Loaded: {len(data)} samples")

    # Load Sub 219 for pseudo-labels
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    test_ids = [d['id'] for d in data if not d['train']]

    for d in data:
        if not d['train']:
            idx = test_ids.index(d['id'])
            d['targets'] = {
                'angle': sub219['scaled_angle'].iloc[idx],
                'depth': sub219['scaled_depth'].iloc[idx],
                'left_right': sub219['scaled_left_right'].iloc[idx]
            }

    # Fabricate data
    print("\nFabricating data...")
    n_fab = 3000
    methods = ['noise', 'scale', 'warp']

    for _ in range(n_fab):
        src = data[np.random.randint(len(data))]
        new_ts = fabricate(src['ts'], np.random.choice(methods))

        # Sometimes interpolate
        if np.random.random() < 0.3:
            src2 = data[np.random.randint(len(data))]
            alpha = np.random.uniform(0.3, 0.7)
            min_len = min(len(new_ts), len(src2['ts']))
            new_ts = new_ts[:min_len] * (1-alpha) + src2['ts'][:min_len] * alpha
            new_targets = {k: src['targets'][k]*(1-alpha) + src2['targets'][k]*alpha
                          for k in ['angle', 'depth', 'left_right']}
        else:
            new_targets = src['targets'].copy()

        data.append({
            'ts': new_ts, 'train': True, 'player': src['player'],
            'targets': new_targets, 'fabricated': True
        })

    print(f"Total: {len(data)} samples")

    # Extract features
    print("\nExtracting features...")
    for d in data:
        d['feat'] = extract_features(d['ts'])

    feat_dim = len(data[0]['feat'])
    print(f"Feature dim: {feat_dim}")

    # Prepare data
    X_all = np.array([d['feat'] for d in data])
    test_mask = np.array([not d['train'] or d.get('fabricated', False) is False and not d['train']
                          for d in data])

    # Actually, let me redo this logic
    train_data = [d for d in data if d['train'] or d.get('fabricated', False)]
    test_data = [d for d in data if not d['train'] and not d.get('fabricated', False)]

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    X_train = np.array([d['feat'] for d in train_data])
    X_test = np.array([d['feat'] for d in test_data])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train per-target models
    targets = ['angle', 'depth', 'left_right']
    predictions = {}

    for target in targets:
        print(f"\nTraining {target} model...")
        y_train = np.array([d['targets'][target] for d in train_data])

        model = train_model(X_train_scaled, y_train, epochs=80, bottleneck=32)

        # Predict
        model.eval()
        with torch.no_grad():
            _, pred, _ = model(torch.FloatTensor(X_test_scaled).to(DEVICE))
            predictions[target] = pred.cpu().numpy().flatten()

    # Calibrate
    predictions['depth'] = predictions['depth'] - np.mean(predictions['depth']) + 0.5055
    for t in targets:
        predictions[t] = np.clip(predictions[t], 0, 1)

    angle_std = np.std(predictions['angle'])
    corr = np.corrcoef(predictions['angle'], sub219['scaled_angle'].values)[0, 1]

    print(f"\nResults:")
    print(f"  angle_std: {angle_std:.6f}")
    print(f"  Correlation with Sub 219: {corr:.4f}")

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    test_ids = [d['id'] for d in test_data]

    pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    }).to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
    print(f"\nSaved: submission_{next_num}.csv")

    # Blends
    for w in [0.1, 0.2, 0.3]:
        next_num += 1
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions['angle'] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions['depth'] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions['left_right'] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend.to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
        print(f"Saved: submission_{next_num}.csv ({w:.0%} + {1-w:.0%} Sub219)")


if __name__ == "__main__":
    main()
