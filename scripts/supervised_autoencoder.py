"""
Supervised Autoencoder with Data Fabrication

Based on Jane Street 1st Place Solution:
https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp

Key insight: Autoencoder reconstruction doesn't need labels.
We can fabricate unlimited data to train a robust encoder,
then use only labeled data for the prediction head.

Stage 1: Train autoencoder on MASSIVE fabricated dataset (no labels needed)
Stage 2: Freeze encoder, train prediction head on 345 labeled samples
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


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def get_joint(timeseries, keypoint_idx, name, frame):
    if name not in keypoint_idx or frame >= len(timeseries):
        return None
    idx = keypoint_idx[name]
    pos = timeseries[frame, idx*3:(idx+1)*3]
    if np.all(pos == 0):
        return None
    return pos


def extract_features(timeseries, keypoint_idx):
    features = {}
    n_frames = len(timeseries)
    key_frames = [50, 80, 100, 120, 140, 150, 153, 156, 160, 170]
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_elbow', 'left_shoulder',
                  'right_hip', 'left_hip']

    for frame in key_frames:
        if frame >= n_frames:
            continue
        for joint in key_joints:
            pos = get_joint(timeseries, keypoint_idx, joint, frame)
            if pos is not None:
                features[f'{joint}_x_f{frame}'] = pos[0]
                features[f'{joint}_y_f{frame}'] = pos[1]
                features[f'{joint}_z_f{frame}'] = pos[2]

        ls = get_joint(timeseries, keypoint_idx, 'left_shoulder', frame)
        rs = get_joint(timeseries, keypoint_idx, 'right_shoulder', frame)
        lh = get_joint(timeseries, keypoint_idx, 'left_hip', frame)
        rh = get_joint(timeseries, keypoint_idx, 'right_hip', frame)

        if all(p is not None for p in [ls, rs, lh, rh]):
            shoulder_vec = rs - ls
            hip_vec = rh - lh
            cos_ang = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8)
            features[f'body_twist_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            features[f'shoulder_width_f{frame}'] = np.linalg.norm(shoulder_vec)

        wrist = get_joint(timeseries, keypoint_idx, 'right_wrist', frame)
        elbow = get_joint(timeseries, keypoint_idx, 'right_elbow', frame)
        shoulder = get_joint(timeseries, keypoint_idx, 'right_shoulder', frame)

        if all(p is not None for p in [wrist, elbow, shoulder]):
            v1 = shoulder - elbow
            v2 = wrist - elbow
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            features[f'elbow_angle_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

    for f_start, f_end in [(148, 153), (150, 155), (153, 158)]:
        if f_end >= n_frames:
            continue
        for joint in ['right_wrist', 'right_elbow']:
            pos_s = get_joint(timeseries, keypoint_idx, joint, f_start)
            pos_e = get_joint(timeseries, keypoint_idx, joint, f_end)
            if pos_s is not None and pos_e is not None:
                vel = pos_e - pos_s
                features[f'{joint}_speed_{f_start}_{f_end}'] = np.linalg.norm(vel)
    return features


def fabricate_noise(timeseries, noise_std=0.02):
    return timeseries + np.random.normal(0, noise_std, timeseries.shape)


def fabricate_interpolation(ts1, ts2, alpha):
    return ts1 * (1 - alpha) + ts2 * alpha


def fabricate_time_warp(timeseries, warp_factor=0.1):
    n_frames = len(timeseries)
    warp = 1 + np.random.uniform(-warp_factor, warp_factor)
    new_indices = np.clip(np.arange(n_frames) * warp, 0, n_frames - 1).astype(int)
    return timeseries[new_indices]


def fabricate_jitter(timeseries, jitter_std=0.01):
    jitter = np.random.normal(0, jitter_std, timeseries.shape)
    for i in range(1, len(jitter)):
        jitter[i] = 0.5 * jitter[i] + 0.5 * jitter[i-1]
    return timeseries + jitter


def fabricate_scale(timeseries, scale_range=(0.95, 1.05)):
    scale = np.random.uniform(*scale_range)
    return timeseries * scale


def fabricate_data(all_timeseries, n_fabricated=5000):
    print(f"\nFabricating {n_fabricated} synthetic samples...")
    fabricated = []
    n_real = len(all_timeseries)
    methods = ['noise', 'interpolation', 'time_warp', 'jitter', 'scale', 'combo']
    method_counts = {m: 0 for m in methods}

    for i in range(n_fabricated):
        method = np.random.choice(methods)
        method_counts[method] += 1
        idx1 = np.random.randint(n_real)
        ts1 = all_timeseries[idx1]

        if method == 'noise':
            new_ts = fabricate_noise(ts1, noise_std=np.random.uniform(0.01, 0.03))
        elif method == 'interpolation':
            idx2 = np.random.randint(n_real)
            ts2 = all_timeseries[idx2]
            alpha = np.random.uniform(0.2, 0.8)
            new_ts = fabricate_interpolation(ts1, ts2, alpha)
        elif method == 'time_warp':
            new_ts = fabricate_time_warp(ts1, warp_factor=np.random.uniform(0.05, 0.15))
        elif method == 'jitter':
            new_ts = fabricate_jitter(ts1, jitter_std=np.random.uniform(0.005, 0.02))
        elif method == 'scale':
            new_ts = fabricate_scale(ts1, scale_range=(0.95, 1.05))
        elif method == 'combo':
            new_ts = fabricate_noise(ts1, noise_std=0.01)
            new_ts = fabricate_jitter(new_ts, jitter_std=0.005)
            if np.random.random() > 0.5:
                new_ts = fabricate_scale(new_ts)
        fabricated.append(new_ts)

    print("  Methods used:", method_counts)
    return fabricated


class GaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            GaussianNoise(std=0.05),
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z


class PredictionHead(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim=3):
        super().__init__()
        concat_dim = bottleneck_dim + input_dim
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(concat_dim),
            nn.Dropout(0.13),
            nn.Linear(concat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim),
        )

    def forward(self, encoded, raw):
        x = torch.cat([encoded, raw], dim=1)
        return self.mlp(x)


def train_autoencoder(X_all, epochs=200, batch_size=64, bottleneck_dim=32):
    print(f"\nSTAGE 1: Training Autoencoder on {len(X_all)} samples")
    input_dim = X_all.shape[1]
    dataset = TensorDataset(torch.FloatTensor(X_all))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim, bottleneck_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(DEVICE)
            reconstruction, _ = model(x)
            loss = criterion(reconstruction, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 30:
            print(f"  Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss = {avg_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"  Best loss: {best_loss:.6f}")
    return model


def train_prediction_head(autoencoder, X_train, y_train, epochs=100, batch_size=32):
    print(f"\nSTAGE 2: Training Prediction Head on {len(X_train)} labeled samples")
    input_dim = X_train.shape[1]
    bottleneck_dim = autoencoder.encoder[-1].out_features

    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.eval()

    pred_head = PredictionHead(input_dim, bottleneck_dim, output_dim=3).to(DEVICE)
    optimizer = optim.Adam(pred_head.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        pred_head.train()
        total_loss = 0
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            with torch.no_grad():
                _, encoded = autoencoder(x)
            predictions = pred_head(encoded, x)
            loss = criterion(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = pred_head.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 20:
            print(f"  Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss = {avg_loss:.6f}")

    pred_head.load_state_dict(best_state)
    print(f"  Best loss: {best_loss:.6f}")
    return pred_head


def predict(autoencoder, pred_head, X_test):
    autoencoder.eval()
    pred_head.eval()
    with torch.no_grad():
        x = torch.FloatTensor(X_test).to(DEVICE)
        _, encoded = autoencoder(x)
        predictions = pred_head(encoded, x)
    return predictions.cpu().numpy()


def main():
    print("="*70)
    print("SUPERVISED AUTOENCODER WITH DATA FABRICATION")
    print("Based on Jane Street 1st Place Solution")
    print("="*70)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    print("\nLoading data...")
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
    print(f"Real: {len(all_timeseries)} (train={is_train.sum()}, test={(~is_train).sum()})")

    n_fabricated = 10000
    fabricated = fabricate_data(all_timeseries, n_fabricated=n_fabricated)
    all_for_ae = all_timeseries + fabricated
    print(f"Total for autoencoder: {len(all_for_ae)}")

    print("\nExtracting features...")
    features_ae = [extract_features(ts, keypoint_idx) for ts in all_for_ae]
    X_ae = pd.DataFrame(features_ae).fillna(0)
    feature_cols = list(X_ae.columns)
    print(f"Features: {len(feature_cols)}")

    features_real = [extract_features(ts, keypoint_idx) for ts in all_timeseries]
    X_real = pd.DataFrame(features_real).fillna(0)[feature_cols]

    scaler = StandardScaler()
    X_ae_scaled = scaler.fit_transform(X_ae)
    X_real_scaled = scaler.transform(X_real)

    X_train = X_real_scaled[is_train]
    X_test = X_real_scaled[~is_train]

    train_meta = [m for i, m in enumerate(all_metadata) if is_train[i]]
    y_train = np.array([
        [scalers['angle'].transform([[m['angle']]])[0, 0],
         scalers['depth'].transform([[m['depth']]])[0, 0],
         scalers['left_right'].transform([[m['left_right']]])[0, 0]]
        for m in train_meta
    ])
    test_ids = [m['id'] for i, m in enumerate(all_metadata) if not is_train[i]]

    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

    configs = [
        {'bottleneck_dim': 16, 'ae_epochs': 150, 'pred_epochs': 100},
        {'bottleneck_dim': 32, 'ae_epochs': 150, 'pred_epochs': 100},
        {'bottleneck_dim': 64, 'ae_epochs': 150, 'pred_epochs': 100},
    ]

    results = []
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Config: bottleneck={config['bottleneck_dim']}")

        ae = train_autoencoder(X_ae_scaled, config['ae_epochs'], 64, config['bottleneck_dim'])
        head = train_prediction_head(ae, X_train, y_train, config['pred_epochs'])
        preds = predict(ae, head, X_test)

        preds[:, 1] = preds[:, 1] - np.mean(preds[:, 1]) + 0.5055
        preds = np.clip(preds, 0, 1)

        angle_std = np.std(preds[:, 0])
        corr = np.corrcoef(preds[:, 0], sub219['scaled_angle'].values)[0, 1]
        print(f"  angle_std: {angle_std:.6f}, corr: {corr:.4f}")

        results.append({'config': config, 'preds': preds, 'angle_std': angle_std, 'corr': corr})

    best = min(results, key=lambda r: abs(r['angle_std'] - 0.137162))
    print(f"\nBest: bottleneck={best['config']['bottleneck_dim']}, angle_std={best['angle_std']:.6f}")

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    pd.DataFrame({
        'id': test_ids,
        'scaled_angle': best['preds'][:, 0],
        'scaled_depth': best['preds'][:, 1],
        'scaled_left_right': best['preds'][:, 2]
    }).to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
    print(f"Saved: submission_{next_num}.csv")

    for w in [0.1, 0.15, 0.2, 0.3]:
        next_num += 1
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * best['preds'][:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * best['preds'][:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * best['preds'][:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend.to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
        print(f"Saved: submission_{next_num}.csv ({w:.0%} SAE + {1-w:.0%} Sub219)")


if __name__ == "__main__":
    main()
