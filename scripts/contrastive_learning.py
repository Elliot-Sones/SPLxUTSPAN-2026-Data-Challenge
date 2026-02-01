"""
Contrastive Learning for Player-Invariant Representations (Plan 2.4)

Pre-train encoder to embed similar-outcome shots nearby regardless of player.
Then use learned representations for prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, NUM_FRAMES
from data_loader import get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")


def build_feature_matrix(train: bool = True):
    """Build feature matrix."""
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    meta_df = load_metadata(train)
    n_shots = len(meta_df)

    all_features = []
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    feature_df = pd.DataFrame(all_features)
    feature_df = feature_df.fillna(feature_df.median())
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


class ContrastivePairsDataset(Dataset):
    """Dataset that generates positive/negative pairs for contrastive learning."""

    def __init__(self, X, y, margin=0.1):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.margin = margin  # Within this margin, shots are "similar"

        # Pre-compute pairwise target distances
        self.y_normalized = (self.y - self.y.mean(0)) / (self.y.std(0) + 1e-6)

    def __len__(self):
        return len(self.X) * 10  # Generate many pairs per epoch

    def __getitem__(self, idx):
        # Sample anchor
        anchor_idx = idx % len(self.X)
        anchor = self.X[anchor_idx]
        anchor_y = self.y_normalized[anchor_idx]

        # Sample positive (similar outcome)
        distances = torch.norm(self.y_normalized - anchor_y, dim=1)
        positive_mask = distances < self.margin
        positive_mask[anchor_idx] = False

        if positive_mask.sum() > 0:
            positive_indices = torch.where(positive_mask)[0]
            pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
        else:
            # If no positive, use closest
            distances[anchor_idx] = float("inf")
            pos_idx = torch.argmin(distances)

        positive = self.X[pos_idx]

        # Sample negative (different outcome)
        negative_mask = distances > 2 * self.margin
        if negative_mask.sum() > 0:
            negative_indices = torch.where(negative_mask)[0]
            neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
        else:
            # If no negative, use farthest
            neg_idx = torch.argmax(distances)

        negative = self.X[neg_idx]

        return anchor, positive.squeeze(), negative.squeeze()


class ContrastiveEncoder(nn.Module):
    """Encoder that learns player-invariant representations."""

    def __init__(self, input_dim, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)


def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet loss: push positive closer, negative farther."""
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0)
    return loss.mean()


def train_contrastive(X, y, embedding_dim=32, epochs=50, batch_size=32, lr=0.001):
    """Train contrastive encoder."""
    if not TORCH_AVAILABLE:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset
    dataset = ContrastivePairsDataset(X, y, margin=0.2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = ContrastiveEncoder(X.shape[1], embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Get embeddings
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            # Compute loss
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")

    return model


def get_embeddings(model, X, device="cpu"):
    """Get embeddings from trained encoder."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        embeddings = model(X_tensor).cpu().numpy()
    return embeddings


def evaluate_transfer(X_train, y_train, pids_train, embedding_dim=32, epochs=50):
    """Evaluate if contrastive representations transfer across players."""
    scalers_dict = load_scalers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unique_pids = np.unique(pids_train)
    results = []

    for test_pid in unique_pids:
        print(f"\n  Testing transfer to player {test_pid}...")

        # Split data
        train_mask = pids_train != test_pid
        test_mask = pids_train == test_pid

        X_train_4 = X_train[train_mask]
        y_train_4 = y_train[train_mask]
        X_test_1 = X_train[test_mask]
        y_test_1 = y_train[test_mask]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_4)
        X_test_scaled = scaler.transform(X_test_1)

        # Train contrastive encoder on 4 players
        model = train_contrastive(X_train_scaled, y_train_4, embedding_dim=embedding_dim, epochs=epochs)

        if model is None:
            continue

        # Get embeddings
        train_emb = get_embeddings(model, X_train_scaled, device)
        test_emb = get_embeddings(model, X_test_scaled, device)

        # Train predictor on embeddings
        preds = np.zeros_like(y_test_1)
        for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
            ridge = Ridge(alpha=100)
            ridge.fit(train_emb, y_train_4[:, target_idx])
            preds[:, target_idx] = ridge.predict(test_emb)

        # Compare with raw features baseline
        preds_baseline = np.zeros_like(y_test_1)
        for target_idx in range(3):
            ridge = Ridge(alpha=100)
            ridge.fit(X_train_scaled, y_train_4[:, target_idx])
            preds_baseline[:, target_idx] = ridge.predict(X_test_scaled)

        # Calculate MSE
        mse_contrastive = {}
        mse_baseline = {}
        for i, target in enumerate(["angle", "depth", "left_right"]):
            scaler_y = scalers_dict[target]
            y_scaled = scaler_y.transform(y_test_1[:, i].reshape(-1, 1)).ravel()

            pred_contr_scaled = scaler_y.transform(preds[:, i].reshape(-1, 1)).ravel()
            pred_base_scaled = scaler_y.transform(preds_baseline[:, i].reshape(-1, 1)).ravel()

            mse_contrastive[target] = np.mean((y_scaled - pred_contr_scaled) ** 2)
            mse_baseline[target] = np.mean((y_scaled - pred_base_scaled) ** 2)

        total_contrastive = np.mean(list(mse_contrastive.values()))
        total_baseline = np.mean(list(mse_baseline.values()))

        improvement = (total_baseline - total_contrastive) / total_baseline * 100

        results.append({
            "test_player": test_pid,
            "mse_contrastive": total_contrastive,
            "mse_baseline": total_baseline,
            "improvement_pct": improvement,
            **{f"contr_{k}": v for k, v in mse_contrastive.items()},
            **{f"base_{k}": v for k, v in mse_baseline.items()},
        })

        print(f"    Contrastive: {total_contrastive:.6f}, Baseline: {total_baseline:.6f}")
        print(f"    Improvement: {improvement:.2f}%")

    return results


def main():
    print("=" * 60)
    print("CONTRASTIVE LEARNING (Plan 2.4)")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch required. Install with: uv pip install torch")
        return

    print("\nBuilding feature matrix...")
    X, y, pids, feature_names = build_feature_matrix(train=True)
    print(f"X shape: {X.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different embedding dimensions
    all_results = []

    for embedding_dim in [16, 32, 64]:
        print(f"\n=== Testing embedding_dim={embedding_dim} ===")

        results = evaluate_transfer(X_scaled, y, pids, embedding_dim=embedding_dim, epochs=50)
        for r in results:
            r["embedding_dim"] = embedding_dim
        all_results.extend(results)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "contrastive_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'contrastive_results.csv'}")

    # Summary
    print("\n=== SUMMARY ===")
    for emb_dim in [16, 32, 64]:
        subset = results_df[results_df["embedding_dim"] == emb_dim]
        avg_improvement = subset["improvement_pct"].mean()
        avg_contrastive = subset["mse_contrastive"].mean()
        avg_baseline = subset["mse_baseline"].mean()
        print(f"embedding_dim={emb_dim}: improvement={avg_improvement:.2f}%, contrastive={avg_contrastive:.6f}, baseline={avg_baseline:.6f}")


if __name__ == "__main__":
    main()
