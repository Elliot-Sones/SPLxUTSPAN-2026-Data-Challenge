#!/usr/bin/env python
"""Self-supervised pretraining with masked reconstruction on motion sequences.

CPU-friendly baseline: downsample frames, flatten, masked MLP autoencoder.
Then evaluate encoder with Ridge regression using GroupKFold.
"""

import argparse
import ast
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_list(s: str) -> np.ndarray:
    if isinstance(s, list):
        return np.array(s, dtype=np.float32)
    if not isinstance(s, str):
        return np.array(s, dtype=np.float32)
    # Replace NaN tokens so literal_eval succeeds
    safe = s.replace("NaN", "None").replace("nan", "None")
    lst = ast.literal_eval(safe)
    return np.array([np.nan if v is None else v for v in lst], dtype=np.float32)


def parse_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Return array of shape (n_samples, n_frames, n_features)."""
    arrays = []
    for col in feature_cols:
        # Each cell is a list-like string of length n_frames
        col_vals = df[col].apply(_parse_list).values
        arrays.append(np.stack(col_vals))
    # arrays: list of (n_samples, n_frames)
    x = np.stack(arrays, axis=2)
    return x


class MaskedAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_mae(
    x_all: np.ndarray,
    mask_ratio: float,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> Tuple[MaskedAutoencoder, List[float]]:
    set_seed(seed)
    model = MaskedAutoencoder(x_all.shape[1], hidden_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    x_tensor = torch.tensor(x_all, dtype=torch.float32)
    n = x_tensor.shape[0]

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = x_tensor[idx].to(device)
            # mask
            mask = torch.rand_like(batch) < mask_ratio
            masked = batch.clone()
            masked[mask] = 0.0

            recon, _ = model(masked)
            loss = torch.mean((recon[mask] - batch[mask]) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * batch.shape[0]

        epoch_loss /= n
        losses.append(epoch_loss)
        print(f"epoch {epoch} loss {epoch_loss:.6f}")

    return model, losses


def evaluate_encoder(
    model: MaskedAutoencoder,
    x_train: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Tuple[float, float, float, float]:
    set_seed(seed)
    model.eval()
    with torch.no_grad():
        z = model.encoder(torch.tensor(x_train, dtype=torch.float32)).cpu().numpy()

    scaler = StandardScaler()
    z = scaler.fit_transform(z)

    gkf = GroupKFold(n_splits=5)
    mse = np.zeros(3, dtype=np.float64)
    for train_idx, val_idx in gkf.split(z, y, groups):
        z_train, z_val = z[train_idx], z[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        for t in range(3):
            model_ridge = Ridge(alpha=1.0, random_state=seed)
            model_ridge.fit(z_train, y_train[:, t])
            pred = model_ridge.predict(z_val)
            mse[t] += mean_squared_error(y_val[:, t], pred)

    mse /= gkf.get_n_splits()
    total = mse.mean()
    return float(mse[0]), float(mse[1]), float(mse[2]), float(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="Use only N samples (0 = full)")
    ap.add_argument("--output-dir", default="output/ssl_pretrain")
    args = ap.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    if args.subset and args.subset > 0:
        train_df = train_df.sample(n=min(args.subset, len(train_df)), random_state=args.seed)
        test_df = test_df.sample(n=min(args.subset, len(test_df)), random_state=args.seed)

    x_train = parse_feature_matrix(train_df, feature_cols)
    x_test = parse_feature_matrix(test_df, feature_cols)

    # downsample frames
    x_train = x_train[:, :: args.frame_stride, :]
    x_test = x_test[:, :: args.frame_stride, :]

    # flatten
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # normalize using train+test for self-supervised pretraining
    x_all = np.vstack([x_train, x_test])
    if np.isnan(x_all).any():
        col_mean = np.nanmean(x_all, axis=0)
        nan_idx = np.where(np.isnan(x_all))
        x_all[nan_idx] = col_mean[nan_idx[1]]
    mean = x_all.mean(axis=0)
    std = x_all.std(axis=0) + 1e-6
    x_all = (x_all - mean) / std
    x_train = x_all[: len(x_train)]

    device = "cpu"
    model, losses = train_mae(
        x_all,
        mask_ratio=args.mask_ratio,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )

    # save checkpoint
    ckpt_path = out_dir / "mae_checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "feature_cols": feature_cols,
            "frame_stride": args.frame_stride,
            "mask_ratio": args.mask_ratio,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        ckpt_path,
    )

    y = train_df[["angle", "depth", "left_right"]].values.astype(np.float32)
    groups = train_df["participant_id"].values

    angle_mse, depth_mse, lr_mse, total = evaluate_encoder(model, x_train, y, groups, args.seed)

    print("cv_mse_angle", f"{angle_mse:.6f}")
    print("cv_mse_depth", f"{depth_mse:.6f}")
    print("cv_mse_left_right", f"{lr_mse:.6f}")
    print("cv_mse_total", f"{total:.6f}")

    metrics_path = out_dir / "metrics.txt"
    with metrics_path.open("w") as f:
        f.write(f"cv_mse_angle {angle_mse:.6f}\n")
        f.write(f"cv_mse_depth {depth_mse:.6f}\n")
        f.write(f"cv_mse_left_right {lr_mse:.6f}\n")
        f.write(f"cv_mse_total {total:.6f}\n")
        f.write("losses " + ",".join(f"{x:.6f}" for x in losses) + "\n")


if __name__ == "__main__":
    main()
