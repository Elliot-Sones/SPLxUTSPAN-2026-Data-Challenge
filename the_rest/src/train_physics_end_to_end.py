#!/usr/bin/env python3
"""
Train/evaluate the end-to-end differentiable physics model.

This uses a small NN + differentiable projectile layer to predict the
scaled targets directly (as used by evaluation).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from src.data_loader import load_all_as_arrays, get_keypoint_columns, load_scalers, scale_targets
from src.physics_features import init_keypoint_mapping
from src.models.physics_end_to_end import PhysicsEndToEndModel


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    pid: torch.Tensor
    y_scaled: torch.Tensor
    y_raw: torch.Tensor


class ShotDataset(Dataset):
    def __init__(self, X: np.ndarray, pid: np.ndarray, y_scaled: np.ndarray, y_raw: np.ndarray):
        self.X = X.astype(np.float32)
        self.pid = pid.astype(np.int64)
        self.y_scaled = y_scaled.astype(np.float32)
        self.y_raw = y_raw.astype(np.float32)

        # Standardize input with train-set stats computed on this dataset split.
        Xf = self.X.reshape(self.X.shape[0], -1).astype(np.float64)
        mu = np.nanmean(Xf, axis=0)
        sd = np.nanstd(Xf, axis=0)
        sd[sd < 1e-12] = 1.0
        bad = ~np.isfinite(Xf)
        if bad.any():
            Xf[bad] = np.take(mu, np.where(bad)[1])
        Xf = (Xf - mu) / sd
        self.X = Xf.reshape(self.X.shape).astype(np.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int) -> Batch:
        # Use a plain dict so PyTorch's default collate works.
        return {
            "x": torch.from_numpy(self.X[i]),
            "pid": torch.tensor(self.pid[i], dtype=torch.int64),
            "y_scaled": torch.from_numpy(self.y_scaled[i]),
            "y_raw": torch.from_numpy(self.y_raw[i]),
        }


def build_keypoint_index(keypoint_cols: List[str]) -> Dict[str, int]:
    names = []
    for c in keypoint_cols:
        if c.endswith("_x"):
            names.append(c[:-2])
    return {n: i for i, n in enumerate(names)}


@torch.no_grad()
def eval_model(
    model: PhysicsEndToEndModel,
    loader: DataLoader,
    keypoint_index: Dict[str, int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true_s = []
    y_pred_s = []
    y_true_a = []
    y_pred_a = []
    for b in loader:
        x = b["x"].to(device)
        pid = b["pid"].to(device)
        out = model(x, pid, keypoint_index=keypoint_index)
        y_true_s.append(b["y_scaled"].numpy())
        y_pred_s.append(out["scaled"].cpu().numpy())
        y_true_a.append(b["y_raw"][:, 0].numpy())
        y_pred_a.append(out["raw"][:, 0].cpu().numpy())
    yt = np.concatenate(y_true_s, axis=0)
    yp = np.concatenate(y_pred_s, axis=0)
    mse_scaled = float(np.mean((yt - yp) ** 2))
    a_true = np.concatenate(y_true_a)
    a_pred = np.concatenate(y_pred_a)
    mae_angle = float(np.mean(np.abs(a_true - a_pred)))
    rmse_angle = float(np.sqrt(np.mean((a_true - a_pred) ** 2)))
    pct_angle_0p1 = float(np.mean(np.abs(a_true - a_pred) <= 0.1) * 100.0)
    return {
        "mse_scaled": mse_scaled,
        "mae_angle": mae_angle,
        "rmse_angle": rmse_angle,
        "pct_angle_le_0p1": pct_angle_0p1,
    }


def train_one_fold(
    X: np.ndarray,
    pid: np.ndarray,
    y_scaled: np.ndarray,
    y_raw: np.ndarray,
    keypoint_index: Dict[str, int],
    *,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_tr = ShotDataset(X[tr_idx], pid[tr_idx], y_scaled[tr_idx], y_raw[tr_idx])
    ds_va = ShotDataset(X[va_idx], pid[va_idx], y_scaled[va_idx], y_raw[va_idx])
    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    model = PhysicsEndToEndModel(num_keypoints=69, num_coords=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best = None
    best_state = None
    patience = 20
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        for b in tr_loader:
            x = b["x"].to(device)
            pid_t = b["pid"].to(device)
            y_t = b["y_scaled"].to(device)
            out = model(x, pid_t, keypoint_index=keypoint_index)
            loss = torch.mean((out["scaled"] - y_t) ** 2)
            # Encourage peaked attention (optional, small).
            w = model(x, pid_t, keypoint_index=keypoint_index, return_debug=True)["release_weights"]
            ent = -(w * (w.clamp_min(1e-9)).log()).sum(dim=1).mean()
            loss = loss + 1e-4 * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        metrics = eval_model(model, va_loader, keypoint_index, device)
        score = metrics["mse_scaled"]
        if best is None or score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return eval_model(model, va_loader, keypoint_index, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y_raw, meta = load_all_as_arrays(train=True)
    pid = meta["participant_id"].to_numpy(dtype=np.int64)

    scalers = load_scalers()
    y_scaled = scale_targets(y_raw.astype(np.float64), scalers).astype(np.float32)

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    keypoint_index = build_keypoint_index(keypoint_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stratify by participant because test includes same participants.
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(pid)), pid), start=1):
        m = train_one_fold(
            X,
            pid,
            y_scaled,
            y_raw.astype(np.float32),
            keypoint_index,
            tr_idx=tr_idx,
            va_idx=va_idx,
            device=device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed + fold),
        )
        fold_metrics.append(m)
        print(f"fold {fold}: {m}")

    def avg(k: str) -> float:
        return float(np.mean([m[k] for m in fold_metrics]))

    print("\nCV mean:")
    print(
        {
            "mse_scaled": avg("mse_scaled"),
            "mae_angle": avg("mae_angle"),
            "rmse_angle": avg("rmse_angle"),
            "pct_angle_le_0p1": avg("pct_angle_le_0p1"),
        }
    )


if __name__ == "__main__":
    main()
