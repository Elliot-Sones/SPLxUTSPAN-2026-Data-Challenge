"""
Quick test: TCN (Temporal Convolutional Network)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
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

class TCN(nn.Module):
    def __init__(self, in_ch=207, out_ch=3, channels=[32, 32], ks=3, dropout=0.5):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            d = 2 ** i
            in_c = in_ch if i == 0 else channels[i-1]
            layers.append(nn.Conv1d(in_c, ch, ks, padding=d*(ks-1)//2, dilation=d))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, wd=1e-2, patience=10):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    # Normalize per sample
    X = np.zeros_like(X_raw)
    for i in range(len(X_raw)):
        m, s = np.nanmean(X_raw[i]), np.nanstd(X_raw[i]) + 1e-8
        X[i] = (X_raw[i] - m) / s
    X = np.nan_to_num(X, nan=0.0)
    print(f"Shape: {X.shape}")

    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)

    for fold, (tr, va) in enumerate(gkf.split(X, y, pids)):
        print(f"\nFold {fold+1}/5")
        X_tr, y_tr = torch.FloatTensor(X[tr]), torch.FloatTensor(y[tr])
        X_va, y_va = torch.FloatTensor(X[va]), torch.FloatTensor(y[va])

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=16)

        model = TCN(in_ch=207, out_ch=3, channels=[32, 32], ks=3, dropout=0.5)
        model, val_loss = train_model(model, train_loader, val_loader, epochs=100, patience=15)

        model.eval()
        with torch.no_grad():
            preds = model(X_va.to(DEVICE)).cpu().numpy()
        all_preds[va] = preds
        mse = np.mean((y[va] - preds)**2)
        print(f"  Fold MSE: {mse:.4f}")

    print("\n=== TCN Results ===")
    for i, t in enumerate(TARGETS):
        mse = np.mean((y[:, i] - all_preds[:, i])**2)
        print(f"  {t}: MSE = {mse:.6f}")
    total = np.mean((y - all_preds)**2)
    print(f"  TOTAL: MSE = {total:.6f}")
