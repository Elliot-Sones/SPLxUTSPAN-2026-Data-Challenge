"""
Temporal Autoencoder Transfer Learning

Full implementation now that 240 temporal frames are confirmed.

Strategy:
1. Pre-train temporal autoencoder on SPL (125) + competition (345) = 470 shots
2. Learn motion primitives (acceleration, release patterns) without outcome labels
3. Extract 64-dim latent embeddings
4. Fine-tune ensemble models with embeddings + static features
5. Generate multiple submissions with different architectures
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data")
EXTERNAL_DIR = Path("external_data/SPL-Open-Data/basketball/freethrow")
SUBMISSION_DIR = Path("submission")
OUTPUT_DIR = Path("output/temporal_autoencoder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
OPTIMAL_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


def parse_temporal_array(s):
    """Parse string-serialized array"""
    s_clean = s.replace('nan', 'null')
    arr = json.loads(s_clean)
    arr = [np.nan if x is None else x for x in arr]
    return np.array(arr, dtype=np.float32)


def load_spl_sequences():
    """Load SPL temporal sequences (key joints only for efficiency)"""
    print("Loading SPL sequences...")
    data_dir = EXTERNAL_DIR / "data/P0001"

    sequences = []
    for trial_file in sorted(data_dir.glob("BB_FT_P0001_T*.json"))[:125]:
        with open(trial_file) as f:
            trial = json.load(f)

        frames_data = []
        for frame_data in trial["tracking"]:
            player = frame_data["data"]["player"]
            # Key joints: wrists, elbows, shoulders (18 features)
            frame_vec = []
            for kp in ["R_WRIST", "L_WRIST", "R_ELBOW", "L_ELBOW", "R_SHOULDER", "L_SHOULDER"]:
                if kp in player:
                    frame_vec.extend(player[kp])
                else:
                    frame_vec.extend([0.0, 0.0, 0.0])
            frames_data.append(frame_vec)

        frames_data = np.array(frames_data, dtype=np.float32)
        # Pad/trim to 240
        if len(frames_data) < 240:
            pad = np.zeros((240 - len(frames_data), 18), dtype=np.float32)
            frames_data = np.vstack([frames_data, pad])
        else:
            frames_data = frames_data[:240]

        sequences.append(frames_data)

    print(f"Loaded {len(sequences)} SPL sequences (shape: {sequences[0].shape})")
    return sequences


def load_competition_sequences(df):
    """Load competition temporal sequences (key joints only)"""
    print(f"Loading {len(df)} competition sequences...")

    sequences = []
    for idx, row in df.iterrows():
        # Key joints: wrists, elbows, shoulders
        wrist_r = np.column_stack([
            parse_temporal_array(row["right_wrist_x"]),
            parse_temporal_array(row["right_wrist_y"]),
            parse_temporal_array(row["right_wrist_z"])
        ])
        wrist_l = np.column_stack([
            parse_temporal_array(row["left_wrist_x"]),
            parse_temporal_array(row["left_wrist_y"]),
            parse_temporal_array(row["left_wrist_z"])
        ])
        elbow_r = np.column_stack([
            parse_temporal_array(row["right_elbow_x"]),
            parse_temporal_array(row["right_elbow_y"]),
            parse_temporal_array(row["right_elbow_z"])
        ])
        elbow_l = np.column_stack([
            parse_temporal_array(row["left_elbow_x"]),
            parse_temporal_array(row["left_elbow_y"]),
            parse_temporal_array(row["left_elbow_z"])
        ])
        shoulder_r = np.column_stack([
            parse_temporal_array(row["right_shoulder_x"]),
            parse_temporal_array(row["right_shoulder_y"]),
            parse_temporal_array(row["right_shoulder_z"])
        ])
        shoulder_l = np.column_stack([
            parse_temporal_array(row["left_shoulder_x"]),
            parse_temporal_array(row["left_shoulder_y"]),
            parse_temporal_array(row["left_shoulder_z"])
        ])

        # Concatenate: (240, 18)
        sequence = np.hstack([wrist_r, wrist_l, elbow_r, elbow_l, shoulder_r, shoulder_l])
        sequences.append(sequence)

    print(f"Loaded {len(sequences)} competition sequences (shape: {sequences[0].shape})")
    return sequences


class TemporalDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [self.normalize_sequence(seq) for seq in sequences]

    def normalize_sequence(self, seq):
        # Replace NaN with 0 and normalize per-feature
        seq = np.nan_to_num(seq, nan=0.0)
        # Z-score normalization per feature
        mean = seq.mean(axis=0, keepdims=True)
        std = seq.std(axis=0, keepdims=True) + 1e-8
        seq_norm = (seq - mean) / std
        return seq_norm.astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.FloatTensor(seq)


class TemporalAutoencoder(nn.Module):
    """1D CNN autoencoder for temporal sequences"""
    def __init__(self, input_dim=18, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: (240, 18) → (30, 64) → (64,)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_encode = nn.Linear(256, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 30)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.squeeze(2)
        z = self.fc_encode(x)
        return z

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 256, 30)
        x = self.decoder(x)
        x = x.transpose(1, 2)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def pretrain_autoencoder(spl_sequences, comp_sequences, epochs=30, batch_size=32):
    """Pre-train autoencoder on combined data"""
    print(f"\nPre-training autoencoder on {len(spl_sequences)} SPL + {len(comp_sequences)} competition sequences...")

    all_sequences = spl_sequences + comp_sequences
    dataset = TemporalDataset(all_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemporalAutoencoder(input_dim=18, latent_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x_recon, z = model(batch)

            # Match sequence length
            if x_recon.size(1) != batch.size(1):
                x_recon = F.interpolate(
                    x_recon.transpose(1, 2),
                    size=batch.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            loss = criterion(x_recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    print("Pre-training complete!")
    return model


def extract_embeddings(model, sequences):
    """Extract latent embeddings from trained autoencoder"""
    model.eval()
    embeddings = []

    with torch.no_grad():
        for seq in sequences:
            # Normalize
            seq = np.nan_to_num(seq, nan=0.0)
            mean = seq.mean(axis=0, keepdims=True)
            std = seq.std(axis=0, keepdims=True) + 1e-8
            seq_norm = (seq - mean) / std

            seq_tensor = torch.FloatTensor(seq_norm).unsqueeze(0)
            z = model.encode(seq_tensor)
            embeddings.append(z.squeeze(0).numpy())

    return np.array(embeddings)


def extract_static_features(df, target_frame):
    """Extract static keypoints at optimal frame"""
    features = []
    for idx, row in df.iterrows():
        # Key joints at target frame
        frame_features = []
        for joint in ["right_wrist", "left_wrist", "right_elbow", "left_elbow", "right_shoulder", "left_shoulder"]:
            for coord in ["x", "y", "z"]:
                col = f"{joint}_{coord}"
                temporal = parse_temporal_array(row[col])
                frame_features.append(temporal[target_frame])
        features.append(frame_features)
    return np.array(features)


def train_with_embeddings(train_df, test_df, train_embeddings, test_embeddings, model_type="lgb"):
    """Train supervised models with temporal embeddings + static features"""
    print(f"\nTraining with temporal embeddings (model={model_type})...")

    results = {}

    for target_name in ["angle", "depth", "left_right"]:
        print(f"\n--- Target: {target_name} ---")
        target_frame = OPTIMAL_FRAMES[target_name]

        # Combine embeddings + static features
        static_train = extract_static_features(train_df, target_frame)
        static_test = extract_static_features(test_df, target_frame)

        X_train = np.hstack([train_embeddings, static_train])
        X_test = np.hstack([test_embeddings, static_test])
        y_train = train_df[target_name].values

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        if model_type == "lgb":
            model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        elif model_type == "xgb":
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        elif model_type == "cat":
            model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=5,
                random_state=42,
                verbose=0
            )
        else:  # ridge
            model = Ridge(alpha=10.0)

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
        model.fit(X_train_scaled, y_train)
        test_pred = model.predict(X_test_scaled)

        results[target_name] = {
            "predictions": test_pred,
            "cv_mse": -cv_scores.mean(),
        }

        print(f"CV MSE: {-cv_scores.mean():.6f}")

    return results


def scale_targets(values, target_name):
    """Scale targets to [0, 1]"""
    ranges = {
        "angle": (23.78, 56.47),
        "depth": (-10.17, 24.97),
        "left_right": (-12.98, 10.15)
    }
    min_val, max_val = ranges[target_name]
    return np.clip((values - min_val) / (max_val - min_val), 0, 1)


def generate_submissions(results_list, test_df):
    """Generate multiple submissions"""
    print("\n" + "="*80)
    print("Generating submissions")
    print("="*80)

    existing_subs = list(SUBMISSION_DIR.glob("submission_*.csv"))
    max_num = max([int(s.stem.split("_")[1]) for s in existing_subs])

    submission_info = []

    for model_name, results in results_list:
        max_num += 1

        angle_scaled = scale_targets(results["angle"]["predictions"], "angle")
        depth_scaled = scale_targets(results["depth"]["predictions"], "depth")
        lr_scaled = scale_targets(results["left_right"]["predictions"], "left_right")

        submission_rows = []
        for idx, row in test_df.iterrows():
            submission_rows.append({
                "id": row["shot_id"],
                "scaled_angle": angle_scaled[idx],
                "scaled_depth": depth_scaled[idx],
                "scaled_left_right": lr_scaled[idx]
            })

        sub_df = pd.DataFrame(submission_rows)
        sub_path = SUBMISSION_DIR / f"submission_{max_num}.csv"
        sub_df.to_csv(sub_path, index=False)

        submission_info.append({
            "submission": max_num,
            "model": model_name,
            "cv_scores": {t: results[t]["cv_mse"] for t in ["angle", "depth", "left_right"]}
        })

        print(f"Generated Sub {max_num} ({model_name}): angle={results['angle']['cv_mse']:.6f}, depth={results['depth']['cv_mse']:.6f}, LR={results['left_right']['cv_mse']:.6f}")

    return submission_info


def main():
    print("="*80)
    print("Temporal Autoencoder Transfer Learning")
    print("="*80)

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Load temporal sequences
    spl_sequences = load_spl_sequences()
    comp_train_sequences = load_competition_sequences(train_df)
    comp_test_sequences = load_competition_sequences(test_df)

    # Pre-train autoencoder
    model = pretrain_autoencoder(spl_sequences, comp_train_sequences, epochs=30, batch_size=32)

    # Save model
    torch.save(model.state_dict(), OUTPUT_DIR / "autoencoder.pt")
    print(f"\nSaved model to {OUTPUT_DIR / 'autoencoder.pt'}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    train_embeddings = extract_embeddings(model, comp_train_sequences)
    test_embeddings = extract_embeddings(model, comp_test_sequences)

    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # Train with different models
    results_list = []

    for model_type in ["lgb", "xgb", "ridge"]:
        results = train_with_embeddings(train_df, test_df, train_embeddings, test_embeddings, model_type=model_type)
        results_list.append((model_type, results))

    # Generate submissions
    submission_info = generate_submissions(results_list, test_df)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Generated {len(submission_info)} submissions with temporal autoencoder embeddings")
    print("\nSubmissions:")
    for info in submission_info:
        mean_mse = np.mean([info['cv_scores'][t] for t in ['angle', 'depth', 'left_right']])
        print(f"  Sub {info['submission']} ({info['model']}): mean CV MSE = {mean_mse:.6f}")


if __name__ == "__main__":
    main()
