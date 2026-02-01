"""
1D Convolutional Neural Network on Raw Time Series

This is fundamentally different from feature-based approaches:
- Learns features directly from raw sequences
- Can capture temporal patterns that hand-crafted features miss
- Uses per-player models to avoid distribution shift
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_target_ranges():
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }


def load_data():
    """Load data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def extract_series(df):
        all_series = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])
            all_series.append(ts)
        return np.array(all_series)

    print("Loading training data...")
    train_series = extract_series(train_df)
    train_targets = train_df[["angle", "depth", "left_right"]].values
    train_pids = train_df["participant_id"].values
    train_ids = train_df["id"].values

    print("Loading test data...")
    test_series = extract_series(test_df)
    test_pids = test_df["participant_id"].values
    test_ids = test_df["id"].values

    return {
        "train_series": train_series,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_series": test_series,
        "test_pids": test_pids,
        "test_ids": test_ids,
        "keypoint_cols": keypoint_cols,
    }


def build_cnn_model(input_shape, output_dim=1):
    """Build a simple 1D CNN model using PyTorch."""
    import torch
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(self, n_channels, seq_len):
            super().__init__()

            # Conv layers
            self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2)

            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2)

            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.AdaptiveAvgPool1d(1)

            # Dense layers
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, output_dim)

            self.relu = nn.ReLU()

        def forward(self, x):
            # x shape: (batch, seq_len, n_channels)
            x = x.permute(0, 2, 1)  # (batch, n_channels, seq_len)

            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)

            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)

            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)

            x = x.squeeze(-1)

            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

    return CNN1D(input_shape[1], input_shape[0])


def train_cnn(X_train, y_train, X_val=None, y_val=None, epochs=100, lr=0.001, patience=15):
    """Train CNN model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train.reshape(-1, 1))

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=min(32, len(X_train)), shuffle=True)

    # Build model
    model = build_cnn_model(X_train.shape[1:])
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val).to(device)
                y_val_t = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break
        else:
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_cnn(model, X):
    """Make predictions with CNN model."""
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        outputs = model(X_t)
        return outputs.cpu().numpy().flatten()


def approach_1dcnn(data):
    """
    Approach: 1D CNN per player per target

    Learn temporal features directly from raw sequences.
    """
    print("\n" + "="*60)
    print("APPROACH: 1D CNN")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Use release window (frames 100-200)
    frames = list(range(100, 200))
    n_dims = 30  # Use first 30 keypoint columns

    def prepare_data(series, frames, n_dims):
        data = series[:, frames, :n_dims].copy()
        data = np.nan_to_num(data, nan=0.0)
        # Normalize per sample
        for i in range(len(data)):
            for j in range(n_dims):
                std = np.std(data[i, :, j])
                if std > 0:
                    data[i, :, j] = (data[i, :, j] - np.mean(data[i, :, j])) / std
        return data

    train_data = prepare_data(train_series, frames, n_dims)
    test_data = prepare_data(test_series, frames, n_dims)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        print(f"  Player {pid}...")
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_data[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_data[test_mask]

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for t_idx, target in enumerate(TARGETS):
            print(f"    Target: {target}")

            fold_test_preds = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr = y_train[train_idx, t_idx]
                y_val = y_train[val_idx, t_idx]

                model = train_cnn(X_tr, y_tr, X_val, y_val, epochs=50, lr=0.001, patience=10)
                oof_preds[player_indices[val_idx], t_idx] = predict_cnn(model, X_val)
                fold_test_preds.append(predict_cnn(model, X_test))

            # Average predictions from all folds
            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "1dcnn"


def approach_simple_mlp(data):
    """
    Approach: Simple MLP on flattened release window

    A simpler neural network that might generalize better.
    """
    print("\n" + "="*60)
    print("APPROACH: Simple MLP")
    print("="*60)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Use release window
    frames = list(range(140, 170))
    n_dims = 20

    def prepare_data(series, frames, n_dims):
        data = series[:, frames, :n_dims].reshape(len(series), -1)
        data = np.nan_to_num(data, nan=0.0)
        return data

    train_flat = prepare_data(train_series, frames, n_dims)
    test_flat = prepare_data(test_series, frames, n_dims)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for pid in unique_pids:
        print(f"  Player {pid}...")
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for t_idx, target in enumerate(TARGETS):
            fold_test_preds = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
                X_tr = torch.FloatTensor(X_train_scaled[train_idx]).to(device)
                X_val = torch.FloatTensor(X_train_scaled[val_idx]).to(device)
                y_tr = torch.FloatTensor(y_train[train_idx, t_idx:t_idx+1]).to(device)
                y_val = torch.FloatTensor(y_train[val_idx, t_idx:t_idx+1]).to(device)

                model = SimpleMLP(X_train_scaled.shape[1]).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
                criterion = nn.MSELoss()

                best_val_loss = float('inf')
                best_state = None
                no_improve = 0

                for epoch in range(200):
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_tr)
                    loss = criterion(outputs, y_tr)
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_loss = criterion(val_outputs, y_val).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = model.state_dict().copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= 20:
                        break

                model.load_state_dict(best_state)
                model.eval()

                with torch.no_grad():
                    oof_preds[player_indices[val_idx], t_idx] = model(X_val).cpu().numpy().flatten()
                    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
                    fold_test_preds.append(model(X_test_t).cpu().numpy().flatten())

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "simple_mlp"


def create_submission(test_ids, predictions, cv_score, approach_name):
    """Create submission CSV."""
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        name = f.stem
        if name.startswith("submission_"):
            try:
                nums.append(int(name.split('_')[1]))
            except:
                pass
        elif name.startswith("submission"):
            try:
                nums.append(int(name[10:]))
            except:
                pass

    next_num = max(nums) + 1 if nums else 1

    # Scale predictions
    scaled_preds = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    print(f"\nSubmission {next_num} created: {filepath}")
    print(f"  Approach: {approach_name}")
    print(f"  CV Score: {cv_score:.6f}")

    return filepath, next_num


def main():
    print("="*80)
    print("NEURAL NETWORK APPROACHES")
    print("="*80)

    data = load_data()

    results = []
    all_predictions = {}

    # Test MLP first (faster)
    try:
        preds, cv, name = approach_simple_mlp(data)
        results.append({"approach": name, "cv_score": cv})
        all_predictions[name] = preds
    except Exception as e:
        print(f"Error in MLP: {e}")
        import traceback
        traceback.print_exc()

    # Test 1D CNN
    try:
        preds, cv, name = approach_1dcnn(data)
        results.append({"approach": name, "cv_score": cv})
        all_predictions[name] = preds
    except Exception as e:
        print(f"Error in CNN: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF NEURAL NETWORK APPROACHES")
    print("="*80)

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("cv_score")
        print(results_df.to_string(index=False))

        # Save results
        results_df.to_csv(OUTPUT_DIR / "neural_approaches_results.csv", index=False)

        # Create submission with best approach
        best = results_df.iloc[0]
        best_name = best["approach"]
        best_preds = all_predictions[best_name]

        print(f"\nCreating submission with best approach: {best_name}")
        filepath, sub_num = create_submission(
            data["test_ids"],
            best_preds,
            best["cv_score"],
            best_name
        )

    return results_df if results else None


if __name__ == "__main__":
    main()
