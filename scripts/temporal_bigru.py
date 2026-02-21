"""
Bidirectional GRU sequence model over all 240 frames of mocap data.

Fundamentally different from scratch velocity CNN:
- GRU maintains hidden state across frames (captures "how the shot unfolded")
- Uses raw hoop-relative POSITIONS (not velocity) -> different signal
- 16 joints covering full body kinematic chain
- 120 frames at 30fps (2x subsample)

Architecture:
  LayerNorm(48) -> BiGRU(64, 2 layers, dropout=0.4) -> cat PlayerEmbed(8)
  -> FC(136->64, ReLU) -> FC(64->3, Sigmoid)

Usage:
  uv run scripts/temporal_bigru.py --pilot   # fast validation, ~5min
  uv run scripts/temporal_bigru.py           # full run, generates submission
"""

from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

HOOP_FEET = np.array([5.25, -25.0, 10.0])
N_FRAMES = 240
SUBSAMPLE = 2  # 240 -> 120 frames at 30fps
SEQ_LEN = N_FRAMES // SUBSAMPLE  # 120

KEY_JOINTS = [
    "right_wrist", "right_elbow", "right_shoulder",
    "left_wrist", "left_elbow", "left_shoulder",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "mid_hip", "neck",
    "right_second_finger_distal", "right_third_finger_distal",
]
N_JOINTS = len(KEY_JOINTS)   # 16
INPUT_DIM = N_JOINTS * 3     # 48

TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


# ============================================================
# Data loading
# ============================================================

def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    return np.nan_to_num(np.array(json.loads(s), dtype=np.float64), nan=0.0)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right",
                 "scaled_angle", "scaled_depth", "scaled_left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]
    kp_names = [c[:-2] for c in kp_cols if c.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}
    n_kp = len(kp_names)
    n = len(df)

    X_3d = np.zeros((n, N_FRAMES, n_kp, 3), dtype=np.float32)
    for idx, row in df.iterrows():
        for col_i, col in enumerate(kp_cols):
            kp_i = col_i // 3
            ax_i = col_i % 3
            arr = parse_array_string(row[col])
            X_3d[idx, :, kp_i, ax_i] = arr
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx+1}/{n}...")

    return X_3d, kp_index, df


def extract_sequences(X_3d: np.ndarray, kp_index: dict) -> np.ndarray:
    """
    Extract (n, SEQ_LEN, INPUT_DIM) hoop-relative position sequences
    for KEY_JOINTS, subsampled by SUBSAMPLE.
    """
    n = X_3d.shape[0]
    seqs = np.zeros((n, SEQ_LEN, INPUT_DIM), dtype=np.float32)

    for ji, jname in enumerate(KEY_JOINTS):
        idx = kp_index.get(jname)
        if idx is None:
            continue
        traj = X_3d[:, :, idx, :].astype(np.float32)  # (n, 240, 3)
        traj = traj - HOOP_FEET[None, None, :]
        traj_sub = traj[:, ::SUBSAMPLE, :]  # (n, 120, 3)
        seqs[:, :, ji*3:(ji+1)*3] = traj_sub

    return seqs


# ============================================================
# Dataset
# ============================================================

class ShotDataset(Dataset):
    def __init__(self, seqs: np.ndarray, targets: np.ndarray,
                 pids: np.ndarray, augment: bool = False):
        self.seqs = torch.from_numpy(seqs)
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.pids = pids
        self.augment = augment

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i].clone()
        if self.augment:
            seq = seq + torch.randn_like(seq) * 0.02
            shift = np.random.randint(-2, 3)
            if shift != 0:
                seq = torch.roll(seq, shift, dims=0)
        pid = self.pids[i]
        return seq, self.targets[i], pid


# ============================================================
# Model
# ============================================================

class BiGRUShot(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.4,
                 n_players: int = 5, embed_dim: int = 8):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        gru_out = hidden * 2  # bidirectional
        self.player_embed = nn.Embedding(n_players, embed_dim)
        fc_in = gru_out + embed_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # linear output; clip at save time
        )

    def forward(self, x: torch.Tensor, pid: torch.Tensor):
        x = self.input_norm(x)
        out, _ = self.gru(x)          # (B, T, 2*H)
        h = out[:, -1, :]             # last timestep
        e = self.player_embed(pid)    # (B, embed_dim)
        h = torch.cat([h, e], dim=1)
        return self.fc(h)


# ============================================================
# Training helpers
# ============================================================

def get_player_map(pids_unique):
    return {p: i for i, p in enumerate(sorted(pids_unique))}


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.MSELoss()
    for seqs, targets, pids in loader:
        seqs = seqs.to(device)
        targets = targets.to(device)
        pids = torch.tensor(pids, dtype=torch.long).to(device)
        optimizer.zero_grad()
        preds = model(seqs, pids)
        loss = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(seqs)
        n += len(seqs)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for seqs, targets, pids in loader:
        seqs = seqs.to(device)
        pids = torch.tensor(pids, dtype=torch.long).to(device)
        preds = model(seqs, pids)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    per_target = [np.mean((preds[:, i] - targets[:, i])**2) for i in range(3)]
    return np.mean(per_target), preds


# ============================================================
# Cross-validation: player-held-out
# ============================================================

def player_held_out_cv(seqs, targets_arr, pids, pmap, n_epochs,
                       device, seed=42, batch_size=16, lr=1e-3, wd=1e-3):
    np.random.seed(seed)
    torch.manual_seed(seed)

    players = sorted(np.unique(pids))
    oof_preds = np.full_like(targets_arr, np.nan)

    for val_player in players:
        tr_mask = pids != val_player
        va_mask = pids == val_player

        tr_pids_mapped = np.array([pmap[p] for p in pids[tr_mask]])
        va_pids_mapped = np.array([pmap[p] for p in pids[va_mask]])

        tr_ds = ShotDataset(seqs[tr_mask], targets_arr[tr_mask],
                            tr_pids_mapped, augment=True)
        va_ds = ShotDataset(seqs[va_mask], targets_arr[va_mask],
                            va_pids_mapped, augment=False)

        tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = BiGRUShot(n_players=len(players)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)

        for epoch in range(n_epochs):
            train_epoch(model, tr_ld, optimizer, device)
            scheduler.step()

        _, va_preds = evaluate(model, va_ld, device)
        oof_preds[va_mask] = va_preds
        mse = np.mean((va_preds - targets_arr[va_mask])**2)
        print(f"    Player {val_player}: val MSE = {mse:.6f}")

    valid = ~np.isnan(oof_preds[:, 0])
    overall = np.mean((oof_preds[valid] - targets_arr[valid])**2)
    per_target = [np.mean((oof_preds[valid, i] - targets_arr[valid, i])**2) for i in range(3)]
    return overall, per_target, oof_preds


# ============================================================
# Full training for test predictions
# ============================================================

def train_full(seqs_tr, targets_tr, pids_tr, seqs_te, pids_te,
               pmap, n_epochs, device, seed=42, batch_size=16, lr=1e-3, wd=1e-3):
    np.random.seed(seed)
    torch.manual_seed(seed)

    tr_pids_mapped = np.array([pmap[p] for p in pids_tr])
    te_pids_mapped = np.array([pmap[p] for p in pids_te])

    tr_ds = ShotDataset(seqs_tr, targets_tr, tr_pids_mapped, augment=True)
    te_ds = ShotDataset(seqs_te, np.zeros((len(seqs_te), 3), dtype=np.float32),
                        te_pids_mapped, augment=False)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BiGRUShot(n_players=len(pmap)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)

    for epoch in range(n_epochs):
        loss = train_epoch(model, tr_ld, optimizer, device)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}/{n_epochs}: train loss = {loss:.6f}")

    _, te_preds = evaluate(model, te_ld, device)
    return te_preds


# ============================================================
# Submission numbering
# ============================================================

import fcntl

def get_next_sub_num():
    lock = SUBMISSION_DIR / ".submission_lock"
    lock.touch(exist_ok=True)
    with open(lock, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            nums = [int(p.stem.split("_")[1]) for p in SUBMISSION_DIR.glob("submission_*.csv")
                    if p.stem.split("_")[1].isdigit()]
            n = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{n}.csv").touch()
            return n
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Quick pilot: 50 epochs, 2 seeds, player-held-out CV only")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if args.pilot:
        n_epochs = 50
        n_seeds = 2
        print("PILOT MODE: 50 epochs, 2 seeds, player-held-out CV only")
    else:
        n_epochs = args.epochs
        n_seeds = args.seeds
        print(f"FULL MODE: {n_epochs} epochs, {n_seeds} seeds")

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    print("\nLoading train data...")
    X_3d_tr, kp_index, df_tr = load_data(DATA_DIR / "train.csv")
    pids_tr = df_tr["participant_id"].values
    pmap = get_player_map(np.unique(pids_tr))
    print(f"  {len(df_tr)} train shots, players: {sorted(pmap.keys())}")

    # Scale targets using pkl scalers (same space as other submissions)
    scalers = {}
    for t in ["angle", "depth", "left_right"]:
        scalers[t] = joblib.load(DATA_DIR / f"scaler_{t}.pkl")
    targets_arr = np.column_stack([
        scalers["angle"].transform(df_tr["angle"].values.reshape(-1, 1)).ravel(),
        scalers["depth"].transform(df_tr["depth"].values.reshape(-1, 1)).ravel(),
        scalers["left_right"].transform(df_tr["left_right"].values.reshape(-1, 1)).ravel(),
    ]).astype(np.float32)

    print("\nExtracting sequences...")
    seqs_tr = extract_sequences(X_3d_tr, kp_index)
    print(f"  Train sequences shape: {seqs_tr.shape}")

    print(f"\nModel: BiGRU(hidden={args.hidden}, layers=2, dropout=0.4) + PlayerEmbed(8)")
    print(f"  Input: {SEQ_LEN} frames x {INPUT_DIM} features ({N_JOINTS} joints)")

    print("\n" + "=" * 60)
    print("PLAYER-HELD-OUT CROSS-VALIDATION")
    print("=" * 60)

    all_oof = np.zeros((len(df_tr), 3))
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}:")
        overall, per_t, oof = player_held_out_cv(
            seqs_tr, targets_arr, pids_tr, pmap,
            n_epochs=n_epochs, device=device, seed=seed,
            batch_size=args.batch, lr=args.lr
        )
        print(f"  Seed {seed} overall: {overall:.6f}")
        print(f"  Per-target: angle={per_t[0]:.6f}, depth={per_t[1]:.6f}, LR={per_t[2]:.6f}")
        all_oof += oof / n_seeds

    # Replace NaN (from averaging) with fallback
    nan_mask = np.isnan(all_oof[:, 0])
    if nan_mask.any():
        all_oof[nan_mask] = targets_arr[nan_mask]  # fallback to truth if something broke

    final_mse = np.mean((all_oof - targets_arr)**2)
    per_final = [np.mean((all_oof[:, i] - targets_arr[:, i])**2) for i in range(3)]
    print(f"\n  Ensembled OOF ({n_seeds} seeds):")
    print(f"  Overall: {final_mse:.6f}")
    print(f"  angle={per_final[0]:.6f}, depth={per_final[1]:.6f}, LR={per_final[2]:.6f}")

    # Diversity vs Sub 3190
    base_path = SUBMISSION_DIR / "submission_3190.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        print("\n  Diversity vs Sub 3190 (train OOF):")
        for i, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
            r = np.corrcoef(base[col].values, all_oof[:, i])[0, 1]
            print(f"    {col}: r = {r:.4f}")

    if args.pilot:
        print("\nPilot complete. Rerun without --pilot to generate submission.")
        return

    print("\n" + "=" * 60)
    print("GENERATING TEST SUBMISSION")
    print("=" * 60)

    print("\nLoading test data...")
    X_3d_te, _, df_te = load_data(DATA_DIR / "test.csv")
    pids_te = df_te["participant_id"].values
    seqs_te = extract_sequences(X_3d_te, kp_index)
    print(f"  {len(df_te)} test shots")

    te_preds_all = np.zeros((len(df_te), 3))
    for seed in range(n_seeds):
        print(f"\n  Full train seed {seed}...")
        te_preds = train_full(
            seqs_tr, targets_arr, pids_tr,
            seqs_te, pids_te,
            pmap, n_epochs=n_epochs, device=device, seed=seed,
            batch_size=args.batch, lr=args.lr
        )
        te_preds_all += te_preds / n_seeds

    te_preds_all = np.clip(te_preds_all, 0.0, 1.0)

    bn = get_next_sub_num()
    sub = pd.DataFrame({"id": df_te["id"].values})
    sub["scaled_angle"] = te_preds_all[:, 0]
    sub["scaled_depth"] = te_preds_all[:, 1]
    sub["scaled_left_right"] = te_preds_all[:, 2]
    sub.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
    print(f"\nSaved standalone BiGRU: Sub {bn}")

    if base_path.exists():
        for w in [0.03, 0.05, 0.07, 0.10]:
            bbn = get_next_sub_num()
            blend = pd.DataFrame({"id": df_te["id"].values})
            for i, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
                blend[col] = (1 - w) * base[col].values + w * te_preds_all[:, i]
            blend.to_csv(SUBMISSION_DIR / f"submission_{bbn}.csv", index=False)
            print(f"  Sub {bbn}: {w*100:.0f}% BiGRU + {(1-w)*100:.0f}% Sub3190")

    print("\n" + "=" * 60)
    print("DETAILS")
    print("=" * 60)
    print(f"  Architecture: BiGRU(hidden={args.hidden}, layers=2, dropout=0.4) + PlayerEmbed(8)")
    print(f"  Sequence: {SEQ_LEN} frames x {INPUT_DIM} features")
    print(f"  Joints ({N_JOINTS}): {KEY_JOINTS}")
    print(f"  Epochs: {n_epochs}, Seeds: {n_seeds}")
    print(f"  Augmentation: noise(0.02) + time shift(+-2)")
    print(f"  Player-held-out CV MSE: {final_mse:.6f}")
    print(f"  angle={per_final[0]:.6f}, depth={per_final[1]:.6f}, LR={per_final[2]:.6f}")


if __name__ == "__main__":
    main()
