"""
Quick test: MC Dropout for uncertainty quantification
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
TARGETS = ["angle", "depth", "left_right"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    return np.array(json.loads(s.replace("nan", "null")), dtype=np.float32)

def load_raw_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = ["id", "shot_id", "participant_id"] + TARGETS
    keypoint_cols = [c for c in df.columns if c not in meta_cols]
    n = len(df)
    X = np.zeros((n, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X[idx, :, i] = parse_array_string(row[col])
    y = df[TARGETS].values.astype(np.float32)
    pids = df["participant_id"].values
    return X, y, pids

class MCDropoutMLP(nn.Module):
    def __init__(self, in_dim, hidden=[128, 64], out_dim=3, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        self.train()  # Keep dropout active
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x))
        preds = torch.stack(preds)
        return preds.mean(0), preds.std(0)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, wd=1e-3, patience=20):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    best_loss, best_state, wait = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                val_loss += crit(model(X), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, model.state_dict().copy(), 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss

if __name__ == "__main__":
    print("Loading data...")
    X_raw, y, pids = load_raw_data()

    # Simple features: mean, std, range per keypoint
    X = np.concatenate([
        X_raw.mean(axis=1),
        X_raw.std(axis=1),
        X_raw.max(axis=1) - X_raw.min(axis=1)
    ], axis=1)
    X = np.nan_to_num(X, nan=0.0)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"Shape: {X.shape}")

    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)
    all_stds = np.zeros_like(y)

    for fold, (tr, va) in enumerate(gkf.split(X, y, pids)):
        print(f"\nFold {fold+1}/5")

        # Re-fit scaler on train only
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X[tr])
        X_va = scaler_fold.transform(X[va])

        X_tr_t = torch.FloatTensor(X_tr)
        y_tr_t = torch.FloatTensor(y[tr])
        X_va_t = torch.FloatTensor(X_va)

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va_t, torch.FloatTensor(y[va])), batch_size=32)

        model = MCDropoutMLP(in_dim=X.shape[1], hidden=[128, 64], out_dim=3, dropout=0.3)
        model, _ = train_model(model, train_loader, val_loader, epochs=200, patience=20)

        mean_pred, std_pred = model.predict_with_uncertainty(X_va_t.to(DEVICE), n_samples=50)
        all_preds[va] = mean_pred.cpu().numpy()
        all_stds[va] = std_pred.cpu().numpy()

        mse = np.mean((y[va] - mean_pred.cpu().numpy())**2)
        print(f"  Fold MSE: {mse:.4f}, Mean Uncertainty: {std_pred.mean():.4f}")

    print("\n=== MC Dropout Results ===")
    for i, t in enumerate(TARGETS):
        mse = np.mean((y[:, i] - all_preds[:, i])**2)
        print(f"  {t}: MSE = {mse:.6f}, Uncertainty = {all_stds[:, i].mean():.6f}")
    total = np.mean((y - all_preds)**2)
    print(f"  TOTAL: MSE = {total:.6f}")

    # Analyze uncertainty-error correlation
    print("\n=== Uncertainty-Error Correlation ===")
    errors = np.abs(y - all_preds)
    for i, t in enumerate(TARGETS):
        corr = np.corrcoef(errors[:, i], all_stds[:, i])[0, 1]
        print(f"  {t}: r = {corr:.4f}")
