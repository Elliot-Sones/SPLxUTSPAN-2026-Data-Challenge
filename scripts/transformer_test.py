"""
Quick test: Simple Transformer with heavy regularization
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

class SimpleTransformer(nn.Module):
    def __init__(self, in_dim=207, d_model=32, nhead=4, layers=2, out_dim=3, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 60, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*2, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model//2, out_dim)
        )

    def forward(self, x):
        x = self.proj(x)
        x = x + self.pos[:, :x.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.0005, wd=1e-2, patience=10):
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

    # Normalize and subsample (every 4th frame -> 60 frames)
    X = np.zeros_like(X_raw)
    for i in range(len(X_raw)):
        m, s = np.nanmean(X_raw[i]), np.nanstd(X_raw[i]) + 1e-8
        X[i] = (X_raw[i] - m) / s
    X = np.nan_to_num(X, nan=0.0)
    X = X[:, ::4, :]  # Subsample to 60 frames
    print(f"Shape: {X.shape}")

    gkf = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y)

    for fold, (tr, va) in enumerate(gkf.split(X, y, pids)):
        print(f"\nFold {fold+1}/5")
        X_tr, y_tr = torch.FloatTensor(X[tr]), torch.FloatTensor(y[tr])
        X_va, y_va = torch.FloatTensor(X[va]), torch.FloatTensor(y[va])

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=16)

        model = SimpleTransformer(in_dim=207, d_model=32, nhead=4, layers=2, out_dim=3, dropout=0.5)
        model, val_loss = train_model(model, train_loader, val_loader, epochs=100, patience=15)

        model.eval()
        with torch.no_grad():
            preds = model(X_va.to(DEVICE)).cpu().numpy()
        all_preds[va] = preds
        mse = np.mean((y[va] - preds)**2)
        print(f"  Fold MSE: {mse:.4f}")

    print("\n=== Transformer Results ===")
    for i, t in enumerate(TARGETS):
        mse = np.mean((y[:, i] - all_preds[:, i])**2)
        print(f"  {t}: MSE = {mse:.6f}")
    total = np.mean((y - all_preds)**2)
    print(f"  TOTAL: MSE = {total:.6f}")
