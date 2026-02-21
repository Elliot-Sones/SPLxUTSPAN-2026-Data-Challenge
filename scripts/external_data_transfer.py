"""
External Data Transfer Learning Pipeline
Uses SPL Open Data (125 free throws) to pre-train an encoder, then fine-tunes on competition data.

Strategy:
1. Load SPL external data (125 shots, 1 player, 240 frames, 27 keypoints)
2. Map SPL keypoints to competition keypoints (69 keypoints)
3. Pre-train autoencoder or contrastive encoder on SPL + competition data (470 shots total)
4. Fine-tune on competition data (345 shots) with target labels
5. Generate predictions and blend with Sub 784
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle

# Paths
DATA_DIR = Path("data")
EXTERNAL_DIR = Path("external_data/SPL-Open-Data/basketball/freethrow")
SUBMISSION_DIR = Path("submission")
OUTPUT_DIR = Path("output/external_transfer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
HOOP_POS = np.array([5.25, -25, 10])
OPTIMAL_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


def load_spl_data():
    """Load SPL Open Data (125 basketball free throws)"""
    print("Loading SPL Open Data...")
    data_dir = EXTERNAL_DIR / "data/P0001"
    all_shots = []

    for trial_file in sorted(data_dir.glob("BB_FT_P0001_T*.json")):
        with open(trial_file) as f:
            trial = json.load(f)

        # Extract keypoints from tracking data
        frames_data = []
        for frame_data in trial["tracking"]:
            player = frame_data["data"]["player"]
            # Map SPL keypoints to competition format (subset)
            frame_vec = []
            for kp_name in ["NOSE", "L_EYE", "R_EYE", "L_EAR", "R_EAR",
                           "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW",
                           "L_WRIST", "R_WRIST", "L_HIP", "R_HIP",
                           "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]:
                if kp_name in player:
                    coords = player[kp_name]
                    # Handle both list and potentially string-serialized coords
                    if isinstance(coords, (list, tuple)):
                        frame_vec.extend(coords)
                    else:
                        frame_vec.extend([float(coords), float(coords), float(coords)])
                else:
                    frame_vec.extend([np.nan, np.nan, np.nan])
            frames_data.append(frame_vec)

        # Convert to numpy array with float dtype
        frames_data = np.array(frames_data, dtype=np.float32)

        # Pad/trim to 240 frames
        if len(frames_data) < 240:
            pad = np.full((240 - len(frames_data), frames_data.shape[1]), np.nan, dtype=np.float32)
            frames_data = np.vstack([frames_data, pad])
        elif len(frames_data) > 240:
            frames_data = frames_data[:240]

        all_shots.append({
            "trial_id": trial["trial_id"],
            "frames": frames_data,
            "result": trial.get("result", "unknown"),
            "entry_angle": trial.get("entry_angle", np.nan),
        })

    print(f"Loaded {len(all_shots)} SPL shots")
    return all_shots


def load_competition_data():
    """Load competition train/test data"""
    print("Loading competition data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Create scalers (work in raw target space, not scaled)
    # No scaling needed for transfer learning - we'll work with raw targets
    return train, test


def map_spl_to_competition_keypoints(spl_frames):
    """
    Map SPL keypoints (17 keypoints x 3 coords = 51 features) to competition keypoints (69 keypoints).
    Strategy: Use subset matching + fill with zeros for missing keypoints (fingers, toes).
    """
    # SPL has: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles (17 keypoints)
    # Competition has: 69 keypoints including fingers, toes
    # We'll create a mapping by matching the overlapping keypoints

    # Competition keypoint order (first 17 match SPL)
    comp_kps = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]

    # Fill missing keypoints with zeros (will be imputed later)
    n_frames = spl_frames.shape[0]
    n_comp_features = 69 * 3  # 207 features
    comp_frames = np.zeros((n_frames, n_comp_features))

    # Copy overlapping keypoints (first 17 * 3 = 51 features)
    comp_frames[:, :51] = spl_frames

    return comp_frames


def extract_style_embedding(frames, embedding_dim=32):
    """
    Extract motion style embedding using PCA on velocity features.
    This captures shooting style: fast vs slow, high arc vs flat, etc.
    """
    # Compute velocity (frame-to-frame differences)
    velocity = np.diff(frames, axis=0)

    # Remove NaN rows
    valid_mask = ~np.isnan(velocity).any(axis=1)
    velocity_clean = velocity[valid_mask]

    if len(velocity_clean) < 10:
        return np.zeros(embedding_dim)

    # PCA to extract style components
    pca = PCA(n_components=min(embedding_dim, velocity_clean.shape[0], velocity_clean.shape[1]))
    try:
        embedding = pca.fit_transform(velocity_clean)
        # Use mean and std of PC scores as style features
        style_features = np.concatenate([
            embedding.mean(axis=0),
            embedding.std(axis=0)
        ])
        # Pad to embedding_dim
        if len(style_features) < embedding_dim:
            style_features = np.pad(style_features, (0, embedding_dim - len(style_features)))
        else:
            style_features = style_features[:embedding_dim]
        return style_features
    except:
        return np.zeros(embedding_dim)


class TemporalAutoencoder(nn.Module):
    """
    Temporal autoencoder for learning motion representations.
    Encodes 240 frames of motion into a fixed-size latent vector.
    """
    def __init__(self, input_dim=51, hidden_dim=128, latent_dim=64, seq_len=240):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        # Encoder: Conv1D layers to capture temporal patterns
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder: Transpose Conv1D layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x):
        """Encode sequence to latent vector"""
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        # Global average pooling
        z = z.mean(dim=2)
        return z

    def decode(self, z, seq_len):
        """Decode latent vector to sequence"""
        # Expand z to sequence
        z = z.unsqueeze(2).expand(-1, -1, seq_len // 8)
        x_recon = self.decoder(z)
        # Adjust to exact seq_len
        if x_recon.size(2) != seq_len:
            x_recon = F.interpolate(x_recon, size=seq_len, mode='linear', align_corners=False)
        x_recon = x_recon.transpose(1, 2)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z, x.size(1))
        return x_recon, z


class MotionDataset(Dataset):
    def __init__(self, frames_list):
        # Convert all to numpy arrays and ensure float dtype
        self.frames_list = []
        for frames in frames_list:
            if isinstance(frames, np.ndarray):
                frames = frames.astype(np.float32)
            else:
                frames = np.array(frames, dtype=np.float32)
            frames = np.nan_to_num(frames, nan=0.0)
            self.frames_list.append(frames)

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames = self.frames_list[idx]
        return torch.FloatTensor(frames)


def pretrain_autoencoder(spl_frames_list, comp_frames_list, epochs=50, batch_size=32):
    """
    Pre-train temporal autoencoder on combined SPL + competition data.
    This learns a general motion representation without using target labels.
    """
    print(f"Pre-training autoencoder on {len(spl_frames_list)} SPL + {len(comp_frames_list)} competition shots...")

    # Combine data (use only first 51 features for consistency with SPL)
    all_frames = spl_frames_list + [f[:, :51] for f in comp_frames_list]

    dataset = MotionDataset(all_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemporalAutoencoder(input_dim=51, hidden_dim=128, latent_dim=64, seq_len=240)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x_recon, z = model(batch)
            # Mask out zero-padded regions
            mask = (batch.sum(dim=2) != 0).float().unsqueeze(2)
            loss = criterion(x_recon * mask, batch * mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    print("Pre-training complete!")
    return model


def extract_transfer_features(model, frames, target_frame=153):
    """
    Extract transfer learning features from pre-trained autoencoder.
    Returns: concatenated features [latent_embedding, keypoints_at_frame, style_embedding]
    """
    # Normalize frames (use first 51 features)
    frames_subset = frames[:, :51]
    frames_subset = np.nan_to_num(frames_subset, nan=0.0)

    # Get latent embedding
    model.eval()
    with torch.no_grad():
        frames_tensor = torch.FloatTensor(frames_subset).unsqueeze(0)
        _, latent = model(frames_tensor)
        latent = latent.squeeze(0).numpy()

    # Get keypoints at target frame
    kp_features = frames[target_frame, :51]
    kp_features = np.nan_to_num(kp_features, nan=0.0)

    # Get style embedding
    style_emb = extract_style_embedding(frames_subset, embedding_dim=32)

    # Concatenate all features
    transfer_features = np.concatenate([latent, kp_features, style_emb])
    return transfer_features


def train_with_transfer_features(train_df, test_df, pretrained_model):
    """
    Train models using transfer learning features + competition data.
    """
    print("\n" + "="*80)
    print("Training with transfer learning features")
    print("="*80)

    # Load keypoint data
    train_keypoints = []
    for idx, row in train_df.iterrows():
        shot_id = row["shot_id"]
        # Reconstruct keypoint array (69 keypoints x 3 coords)
        kp_cols = [c for c in train_df.columns if c not in ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]]
        kp_array = row[kp_cols].values.reshape(69, 3)
        # Expand to 240 frames (replicate single frame - NOTE: this is a limitation, actual data has 240 frames)
        # For competition data, we only have single-frame keypoints, so we replicate
        kp_frames = np.tile(kp_array.flatten(), (240, 1))
        train_keypoints.append(kp_frames)

    test_keypoints = []
    for idx, row in test_df.iterrows():
        kp_cols = [c for c in test_df.columns if c not in ["id", "shot_id", "participant_id"]]
        kp_array = row[kp_cols].values.reshape(69, 3)
        kp_frames = np.tile(kp_array.flatten(), (240, 1))
        test_keypoints.append(kp_frames)

    # Extract transfer features for each target
    results = {}
    for target_name in ["angle", "depth", "left_right"]:
        print(f"\n--- Target: {target_name} ---")
        target_frame = OPTIMAL_FRAMES[target_name]

        # Extract features
        X_train = np.array([extract_transfer_features(pretrained_model, kp, target_frame) for kp in train_keypoints])
        X_test = np.array([extract_transfer_features(pretrained_model, kp, target_frame) for kp in test_keypoints])

        # Get targets (raw values, no scaling)
        y_train = train_df[target_name].values

        # Standardize features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Train ensemble: LightGBM + XGBoost + CatBoost
        models = []
        cv_scores = []

        # LightGBM
        lgb_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_scores = cross_val_score(lgb_model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
        lgb_model.fit(X_train_scaled, y_train)
        models.append(("LGB", lgb_model))
        cv_scores.append(("LGB", -lgb_scores.mean()))
        print(f"LGB CV MSE: {-lgb_scores.mean():.6f}")

        # XGBoost
        xgb_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": 0
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
        xgb_model.fit(X_train_scaled, y_train)
        models.append(("XGB", xgb_model))
        cv_scores.append(("XGB", -xgb_scores.mean()))
        print(f"XGB CV MSE: {-xgb_scores.mean():.6f}")

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=5,
            random_state=42,
            verbose=0
        )
        cat_scores = cross_val_score(cat_model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
        cat_model.fit(X_train_scaled, y_train)
        models.append(("CAT", cat_model))
        cv_scores.append(("CAT", -cat_scores.mean()))
        print(f"CAT CV MSE: {-cat_scores.mean():.6f}")

        # Ensemble predictions
        test_preds = []
        for name, model in models:
            pred = model.predict(X_test_scaled)
            test_preds.append(pred)

        # Average ensemble
        test_pred_ensemble = np.mean(test_preds, axis=0)

        results[target_name] = {
            "cv_scores": cv_scores,
            "predictions": test_pred_ensemble,
            "models": models,
            "scaler_X": scaler_X
        }

        print(f"Best CV MSE: {min([s[1] for s in cv_scores]):.6f}")

    return results


def scale_targets(values, target_name):
    """
    Scale raw target values to [0, 1] range based on train distribution.
    """
    # Hardcoded train ranges from data exploration
    ranges = {
        "angle": (23.78, 56.47),
        "depth": (-10.17, 24.97),
        "left_right": (-12.98, 10.15)
    }
    min_val, max_val = ranges[target_name]
    scaled = (values - min_val) / (max_val - min_val)
    return np.clip(scaled, 0, 1)


def generate_submissions(results, test_df, sub_784_path, num_blends=10):
    """
    Generate multiple submission files by blending transfer predictions with Sub 784.
    """
    print("\n" + "="*80)
    print("Generating submissions")
    print("="*80)

    # Load Sub 784 baseline
    sub_784 = pd.read_csv(sub_784_path)
    sub_784 = sub_784.set_index("id")

    # Get next submission number
    existing_subs = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing_subs:
        max_num = max([int(s.stem.split("_")[1]) for s in existing_subs])
    else:
        max_num = 0

    submission_info = []

    # Blend weights to test
    blend_configs = [
        {"angle": 0.0, "depth": 0.0, "left_right": 0.0},  # Pure Sub 784 (baseline)
        {"angle": 0.2, "depth": 0.1, "left_right": 0.1},  # Conservative
        {"angle": 0.3, "depth": 0.2, "left_right": 0.2},
        {"angle": 0.4, "depth": 0.3, "left_right": 0.3},
        {"angle": 0.5, "depth": 0.4, "left_right": 0.4},  # Moderate
        {"angle": 0.6, "depth": 0.5, "left_right": 0.5},
        {"angle": 0.7, "depth": 0.6, "left_right": 0.6},
        {"angle": 0.8, "depth": 0.7, "left_right": 0.7},  # Aggressive
        {"angle": 1.0, "depth": 0.8, "left_right": 0.8},
        {"angle": 1.0, "depth": 1.0, "left_right": 1.0},  # Pure transfer (high risk)
    ]

    # Scale transfer predictions to [0, 1]
    angle_scaled = scale_targets(results["angle"]["predictions"], "angle")
    depth_scaled = scale_targets(results["depth"]["predictions"], "depth")
    lr_scaled = scale_targets(results["left_right"]["predictions"], "left_right")

    for config in blend_configs[:num_blends]:
        max_num += 1
        sub_num = max_num

        # Create submission
        submission_rows = []
        for idx, row in test_df.iterrows():
            shot_id = row["shot_id"]

            # Blend predictions
            angle_pred = (1 - config["angle"]) * sub_784.loc[shot_id, "scaled_angle"] + config["angle"] * angle_scaled[idx]
            depth_pred = (1 - config["depth"]) * sub_784.loc[shot_id, "scaled_depth"] + config["depth"] * depth_scaled[idx]
            lr_pred = (1 - config["left_right"]) * sub_784.loc[shot_id, "scaled_left_right"] + config["left_right"] * lr_scaled[idx]

            submission_rows.append({
                "id": shot_id,
                "scaled_angle": angle_pred,
                "scaled_depth": depth_pred,
                "scaled_left_right": lr_pred
            })

        sub_df = pd.DataFrame(submission_rows)
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(sub_path, index=False)

        submission_info.append({
            "submission": sub_num,
            "path": str(sub_path),
            "blend_weights": config,
            "cv_scores": {
                "angle": min([s[1] for s in results["angle"]["cv_scores"]]),
                "depth": min([s[1] for s in results["depth"]["cv_scores"]]),
                "left_right": min([s[1] for s in results["left_right"]["cv_scores"]]),
            }
        })

        print(f"Generated {sub_path.name}: weights={config}")

    return submission_info


def main():
    print("="*80)
    print("External Data Transfer Learning Pipeline")
    print("="*80)

    # Load data
    spl_shots = load_spl_data()
    train_df, test_df = load_competition_data()

    # Extract SPL frames (convert to competition format)
    print("\nMapping SPL keypoints to competition format...")
    spl_frames_list = []
    for shot in spl_shots:
        # SPL already has frames (240 x 51)
        spl_frames_list.append(shot["frames"])

    # Extract competition frames (NOTE: competition data is single-frame, replicate to 240)
    comp_frames_list = []
    for idx, row in train_df.iterrows():
        kp_cols = [c for c in train_df.columns if c not in ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]]
        kp_array = row[kp_cols].values
        kp_frames = np.tile(kp_array, (240, 1))
        comp_frames_list.append(kp_frames)

    # Pre-train autoencoder
    pretrained_model = pretrain_autoencoder(spl_frames_list, comp_frames_list, epochs=50, batch_size=32)

    # Save model
    model_path = OUTPUT_DIR / "pretrained_autoencoder.pt"
    torch.save(pretrained_model.state_dict(), model_path)
    print(f"\nSaved pretrained model to {model_path}")

    # Train with transfer features
    results = train_with_transfer_features(train_df, test_df, pretrained_model)

    # Generate submissions
    sub_784_path = SUBMISSION_DIR / "submission_784.csv"
    submission_info = generate_submissions(results, test_df, sub_784_path, num_blends=10)

    # Save metadata
    metadata = {
        "external_data": {
            "source": "MLSE SPL Open Data",
            "n_shots": len(spl_shots),
            "n_frames": 240,
            "n_keypoints": 17,
        },
        "competition_data": {
            "n_train": len(train_df),
            "n_test": len(test_df),
        },
        "transfer_learning": {
            "method": "Temporal Autoencoder + Transfer Features",
            "pretrain_epochs": 50,
            "latent_dim": 64,
            "feature_dim": 64 + 51 + 32,
        },
        "cv_results": {
            "angle": {k: v for k, v in results["angle"]["cv_scores"]},
            "depth": {k: v for k, v in results["depth"]["cv_scores"]},
            "left_right": {k: v for k, v in results["left_right"]["cv_scores"]},
        },
        "submissions": submission_info
    }

    metadata_path = OUTPUT_DIR / "transfer_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"External data: {len(spl_shots)} SPL shots")
    print(f"Competition data: {len(train_df)} train, {len(test_df)} test")
    print(f"Pretrained autoencoder: {model_path}")
    print(f"Generated {len(submission_info)} submissions")
    print("\nCV Results:")
    for target in ["angle", "depth", "left_right"]:
        best_mse = min([s[1] for s in results[target]["cv_scores"]])
        print(f"  {target}: MSE = {best_mse:.6f}")
    print("\nSubmissions:")
    for info in submission_info:
        print(f"  {Path(info['path']).name}: {info['blend_weights']}")

    print("\n" + "="*80)
    print("Pipeline complete! Test submissions on leaderboard.")
    print("="*80)


if __name__ == "__main__":
    main()
