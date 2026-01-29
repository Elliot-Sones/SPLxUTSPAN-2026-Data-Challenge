"""
Neural Network Model: Try a completely different architecture.

The tree-based models (LightGBM, XGBoost, CatBoost) might be capturing similar patterns.
A neural network with dropout and regularization might find different signal.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

# Use CPU for small dataset
device = torch.device("cpu")


class SimpleNet(nn.Module):
    """Simple feedforward neural network with regularization."""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)


class WideNet(nn.Module):
    """Wide network with heavy regularization."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    """Load train and test data."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    def parse_array_string(s):
        if pd.isna(s):
            return np.full(240, np.nan, dtype=np.float32)
        s = s.replace("nan", "null")
        return np.array(json.loads(s), dtype=np.float32)

    def extract_features(df, is_train=True):
        all_features = []
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
            combined = {**hybrid_feats, **advanced_feats}
            all_features.append(combined)

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        return all_features, ids, pids, targets if is_train else None

    print("Extracting training features...")
    train_feats, train_ids, train_pids, train_targets = extract_features(train_df, True)

    print("Extracting test features...")
    test_feats, test_ids, test_pids, _ = extract_features(test_df, False)

    feature_names = sorted(train_feats[0].keys())
    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)

    # Clean data
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    return X_train, y_train, np.array(train_pids), X_test, np.array(test_ids), np.array(test_pids)


def train_nn_model(X_train, y_train, X_val, y_val, input_dim, target_idx,
                   epochs=100, lr=0.01, weight_decay=0.01, hidden_dim=64, dropout=0.3,
                   model_type='simple'):
    """Train a neural network model."""
    if model_type == 'simple':
        model = SimpleNet(input_dim, hidden_dim, dropout).to(device)
    else:
        model = WideNet(input_dim, hidden_dim, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train[:, target_idx:target_idx+1]).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val[:, target_idx:target_idx+1]).to(device)

    # Create dataset and loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def train_per_player(X_train, y_train, pids_train, X_test, pids_test, model_config):
    """Train models per player per target."""
    unique_pids = sorted(np.unique(pids_train))
    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for pid in unique_pids:
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_p_train = X_train[train_mask]
        y_p_train = y_train[train_mask]
        X_p_test = X_test[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        input_dim = X_p_train_scaled.shape[1]

        for target_idx, target in enumerate(TARGETS):
            # 5-fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = []

            for train_idx, val_idx in kf.split(X_p_train_scaled):
                X_tr = X_p_train_scaled[train_idx]
                y_tr = y_p_train[train_idx]
                X_val = X_p_train_scaled[val_idx]
                y_val = y_p_train[val_idx]

                model, val_loss = train_nn_model(
                    X_tr, y_tr, X_val, y_val,
                    input_dim, target_idx,
                    **model_config
                )
                cv_scores.append(val_loss)

            # Train on all data for final prediction
            # Use first 80% as train, last 20% as pseudo-val
            n = len(X_p_train_scaled)
            split_idx = int(n * 0.8)
            model, _ = train_nn_model(
                X_p_train_scaled[:split_idx], y_p_train[:split_idx],
                X_p_train_scaled[split_idx:], y_p_train[split_idx:],
                input_dim, target_idx,
                **model_config
            )

            # Predict
            model.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X_p_test_scaled).to(device)
                pred = model(X_test_t).cpu().numpy().flatten()

            test_indices = np.where(test_mask)[0]
            predictions[test_indices, target_idx] = pred

    return predictions, np.mean(cv_scores)


def main():
    print("="*70)
    print("NEURAL NETWORK MODEL")
    print("="*70)

    # Load data
    X_train, y_train, pids_train, X_test, test_ids, pids_test = load_data()

    print(f"\nTrain: {X_train.shape}")
    print(f"Test: {X_test.shape}")

    # Try different configurations
    configs = {
        "simple_shallow": {
            "epochs": 100,
            "lr": 0.01,
            "weight_decay": 0.01,
            "hidden_dim": 32,
            "dropout": 0.4,
            "model_type": "simple"
        },
        "wide_heavy_reg": {
            "epochs": 100,
            "lr": 0.01,
            "weight_decay": 0.1,
            "hidden_dim": 64,
            "dropout": 0.5,
            "model_type": "wide"
        },
        "simple_medium": {
            "epochs": 100,
            "lr": 0.005,
            "weight_decay": 0.05,
            "hidden_dim": 64,
            "dropout": 0.3,
            "model_type": "simple"
        },
    }

    results = {}

    for config_name, config in configs.items():
        print(f"\n--- {config_name} ---")
        predictions, cv_score = train_per_player(
            X_train, y_train, pids_train,
            X_test, pids_test,
            config
        )
        results[config_name] = {
            "predictions": predictions,
            "cv_score": cv_score
        }
        print(f"  CV Score: {cv_score:.6f}")

    # Scale predictions and create submissions
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    best_depth_max = 0
    best_config = None

    for config_name, result in results.items():
        preds = result["predictions"]
        scaled_preds = np.zeros_like(preds)

        for i, target in enumerate(TARGETS):
            scaled_preds[:, i] = target_scalers[target].transform(
                preds[:, i].reshape(-1, 1)
            ).flatten()

        # Clip to valid range
        scaled_preds = np.clip(scaled_preds, 0, 1)

        depth_max = scaled_preds[:, 1].max()
        print(f"\n{config_name}: depth_max = {depth_max:.4f}")

        if depth_max > best_depth_max:
            best_depth_max = depth_max
            best_config = config_name
            best_preds = scaled_preds

    # Save best neural network submission
    print(f"\n--- Saving best config: {best_config} (depth_max={best_depth_max:.4f}) ---")

    nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
            if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    df = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': best_preds[:, 0],
        'scaled_depth': best_preds[:, 1],
        'scaled_left_right': best_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    df.to_csv(filepath, index=False)

    print(f"\nSubmission {next_num}: Neural Network ({best_config})")
    print(f"File: {filepath}")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
              f"min={df[col].min():.4f}, max={df[col].max():.4f}")

    # Also create ensemble blend with existing best
    print("\n--- Creating NN + sub25 blend ---")

    sub25 = pd.read_csv(SUBMISSION_DIR / "submission_25.csv")

    for w_nn in [0.3, 0.5]:
        w_25 = 1 - w_nn
        blended = pd.DataFrame({'id': test_ids})
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blended[col] = w_nn * df[col] + w_25 * sub25[col]

        next_num += 1
        filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blended.to_csv(filepath, index=False)

        print(f"\nSubmission {next_num}: {w_nn*100:.0f}% NN + {w_25*100:.0f}% sub25")
        print(f"  depth_max = {blended['scaled_depth'].max():.4f}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
