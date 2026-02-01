"""
LSTM Sequence Modeling (Plan 2.3)

Single-layer LSTM with heavy dropout for temporal pattern capture.
Test whether recurrent models capture temporal dependencies better than frame-specific features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, NUM_FRAMES

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: uv pip install torch")


def build_raw_data(train: bool = True):
    """Load raw timeseries data."""
    meta_df = load_metadata(train)
    n_shots = len(meta_df)

    # Pre-allocate arrays
    X = np.zeros((n_shots, NUM_FRAMES, 207), dtype=np.float32)  # 207 features per frame
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        X[i] = timeseries
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids


def preprocess_sequences(X, downsample=4):
    """Preprocess sequences: handle NaN, normalize, downsample."""
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    # Downsample to reduce sequence length (240 -> 60 if downsample=4)
    if downsample > 1:
        n_frames = X.shape[1] // downsample
        X = X[:, ::downsample, :][:, :n_frames, :]

    return X


class LSTMModel(nn.Module):
    """Simple LSTM for regression."""

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.5, output_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        out = self.dropout(last_hidden)
        out = self.fc(out)  # (batch, output_dim)
        return out


class GRUModel(nn.Module):
    """GRU alternative to LSTM."""

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.5, output_dim=3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10, device="cpu"):
    """Train with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


def evaluate_lopo_cv(X, y, pids, model_class=LSTMModel, hidden_dim=32, dropout=0.5, epochs=100, patience=10):
    """LOPO CV with LSTM/GRU."""
    if not TORCH_AVAILABLE:
        return float("inf"), {}, None

    scalers_dict = load_scalers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        print(f"  Training for participant {pid}...")

        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Normalize features per training set
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)

        scaler = StandardScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)

        X_train = X_train_flat.reshape(n_samples, seq_len, n_features)
        X_test = X_test_flat.reshape(len(X_test), seq_len, n_features)

        # Scale targets
        y_train_scaled = np.zeros_like(y_train)
        for i, target in enumerate(["angle", "depth", "left_right"]):
            y_train_scaled[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train_scaled)
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # Use last 20% of training for validation
        val_size = max(1, int(len(X_train) * 0.2))
        val_dataset = TensorDataset(
            torch.FloatTensor(X_train[-val_size:]),
            torch.FloatTensor(y_train_scaled[-val_size:])
        )
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Train model
        model = model_class(
            input_dim=n_features, hidden_dim=hidden_dim,
            dropout=dropout, output_dim=3
        )

        model, _ = train_model(model, train_loader, val_loader, epochs=epochs, patience=patience, device=device)

        # Predict
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            pred_scaled = model(X_test_tensor).cpu().numpy()

        # Inverse scale predictions
        for i, target in enumerate(["angle", "depth", "left_right"]):
            all_preds[test_mask, i] = scalers_dict[target].inverse_transform(
                pred_scaled[:, i].reshape(-1, 1)
            ).ravel()

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target, all_preds


def evaluate_within_player_cv(X, y, pids, model_class=LSTMModel, hidden_dim=32, dropout=0.5, epochs=100, patience=10, n_folds=5):
    """Within-player CV with LSTM/GRU."""
    if not TORCH_AVAILABLE:
        return float("inf"), {}

    scalers_dict = load_scalers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask]

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_pid):
            X_train, X_val = X_pid[train_idx], X_pid[val_idx]
            y_train = y_pid[train_idx]

            # Normalize
            n_samples, seq_len, n_features = X_train.shape
            X_train_flat = X_train.reshape(-1, n_features)
            X_val_flat = X_val.reshape(-1, n_features)

            scaler = StandardScaler()
            X_train_flat = scaler.fit_transform(X_train_flat)
            X_val_flat = scaler.transform(X_val_flat)

            X_train = X_train_flat.reshape(n_samples, seq_len, n_features)
            X_val = X_val_flat.reshape(len(X_val), seq_len, n_features)

            # Scale targets
            y_train_scaled = np.zeros_like(y_train)
            for i, target in enumerate(["angle", "depth", "left_right"]):
                y_train_scaled[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train_scaled)
            )
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

            inner_val_size = max(1, int(len(X_train) * 0.2))
            inner_val_dataset = TensorDataset(
                torch.FloatTensor(X_train[-inner_val_size:]),
                torch.FloatTensor(y_train_scaled[-inner_val_size:])
            )
            inner_val_loader = DataLoader(inner_val_dataset, batch_size=8, shuffle=False)

            # Train model
            model = model_class(
                input_dim=n_features, hidden_dim=hidden_dim,
                dropout=dropout, output_dim=3
            )

            model, _ = train_model(model, train_loader, inner_val_loader, epochs=epochs, patience=patience, device=device)

            # Predict
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                pred_scaled = model(X_val_tensor).cpu().numpy()

            # Map back to global indices
            global_idx = np.where(mask)[0][val_idx]
            for i, target in enumerate(["angle", "depth", "left_right"]):
                all_preds[global_idx, i] = scalers_dict[target].inverse_transform(
                    pred_scaled[:, i].reshape(-1, 1)
                ).ravel()

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def main():
    print("=" * 60)
    print("LSTM/GRU SEQUENCE MODELING (Plan 2.3)")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch required. Install with: uv pip install torch")
        return

    print("\nLoading raw timeseries data...")
    X, y, pids = build_raw_data(train=True)
    print(f"X shape: {X.shape}")  # (n_shots, 240, 207)

    # Preprocess
    print("\nPreprocessing sequences...")
    X = preprocess_sequences(X, downsample=4)  # Downsample to 60 frames
    print(f"X shape after preprocessing: {X.shape}")

    results = []

    # Test configurations
    configs = [
        {"model": "LSTM", "hidden_dim": 16, "dropout": 0.3},
        {"model": "LSTM", "hidden_dim": 32, "dropout": 0.5},
        {"model": "LSTM", "hidden_dim": 64, "dropout": 0.5},
        {"model": "GRU", "hidden_dim": 16, "dropout": 0.3},
        {"model": "GRU", "hidden_dim": 32, "dropout": 0.5},
        {"model": "GRU", "hidden_dim": 64, "dropout": 0.5},
    ]

    for config in configs:
        model_name = config["model"]
        hidden_dim = config["hidden_dim"]
        dropout = config["dropout"]

        print(f"\n--- {model_name} hidden={hidden_dim}, dropout={dropout} ---")

        model_class = LSTMModel if model_name == "LSTM" else GRUModel

        lopo_mse, lopo_per_target, _ = evaluate_lopo_cv(
            X, y, pids, model_class=model_class,
            hidden_dim=hidden_dim, dropout=dropout,
            epochs=100, patience=15
        )

        within_mse, within_per_target = evaluate_within_player_cv(
            X, y, pids, model_class=model_class,
            hidden_dim=hidden_dim, dropout=dropout,
            epochs=100, patience=15
        )

        results.append({
            "model": model_name,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "lopo_mse": lopo_mse,
            "within_mse": within_mse,
            "lopo_angle": lopo_per_target.get("angle", np.nan),
            "lopo_depth": lopo_per_target.get("depth", np.nan),
            "lopo_lr": lopo_per_target.get("left_right", np.nan),
        })

        print(f"  LOPO MSE: {lopo_mse:.6f}")
        print(f"  Within-player MSE: {within_mse:.6f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "lstm_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'lstm_results.csv'}")

    # Find best configuration
    best_idx = results_df["lopo_mse"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n=== BEST LSTM/GRU CONFIGURATION ===")
    print(f"Model: {best['model']}")
    print(f"Hidden dim: {best['hidden_dim']}")
    print(f"Dropout: {best['dropout']}")
    print(f"LOPO MSE: {best['lopo_mse']:.6f}")


if __name__ == "__main__":
    main()
