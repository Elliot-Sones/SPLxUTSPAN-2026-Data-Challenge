#!/usr/bin/env python
"""External video pretraining -> challenge fine-tuning transfer experiment.

Pipeline:
1. Build external video sequences from Basketball_51 dataset (ft0/ft1 labels).
2. Pretrain a temporal encoder on external make/miss classification.
3. Fine-tune same encoder on challenge regression targets (scaled angle/depth/left_right).
4. Compare against identical model without external pretraining using GroupKFold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import iterate_shots, load_scalers, scale_targets  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def float_or_none(x: float) -> float | None:
    if np.isnan(x):
        return None
    return float(x)


def resize_to_grid(frame_gray: np.ndarray, frame_size: int) -> np.ndarray:
    x = torch.tensor(frame_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(frame_size, frame_size), mode="bilinear", align_corners=False)
    x = x.squeeze(0).squeeze(0)
    if x.max() > 0:
        x = x / 255.0
    return x.numpy().astype(np.float32)


def sample_frames_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames <= 1:
        return np.zeros(num_frames, dtype=np.int64)
    return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)


def video_to_sequence(video_path: Path, num_frames: int, frame_size: int) -> np.ndarray:
    reader = iio.get_reader(str(video_path), "ffmpeg")
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()

    if not frames:
        return np.zeros((num_frames, frame_size * frame_size), dtype=np.float32)

    idx = sample_frames_indices(len(frames), num_frames)
    seq = np.zeros((num_frames, frame_size * frame_size), dtype=np.float32)
    for i, frame_idx in enumerate(idx):
        frame = frames[int(frame_idx)]
        gray = frame.astype(np.float32).mean(axis=2)
        grid = resize_to_grid(gray, frame_size)
        seq[i] = grid.reshape(-1)
    return seq


def keypoint_timeseries_to_sequence(
    timeseries: np.ndarray,
    num_frames: int,
    frame_size: int,
) -> np.ndarray:
    xs = timeseries[:, 0::3]
    ys = timeseries[:, 1::3]
    valid_global = ~(np.isnan(xs) | np.isnan(ys))
    if not valid_global.any():
        return np.zeros((num_frames, frame_size * frame_size), dtype=np.float32)

    x_all = xs[valid_global]
    y_all = ys[valid_global]
    x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    x_scale = max(x_max - x_min, 1e-6)
    y_scale = max(y_max - y_min, 1e-6)

    seq = np.zeros((num_frames, frame_size * frame_size), dtype=np.float32)
    idx = sample_frames_indices(timeseries.shape[0], num_frames)
    for i, t in enumerate(idx):
        x = xs[t]
        y = ys[t]
        valid = ~(np.isnan(x) | np.isnan(y))
        hist = np.zeros((frame_size, frame_size), dtype=np.float32)
        if valid.any():
            x_norm = np.clip((x[valid] - x_min) / x_scale, 0.0, 1.0)
            y_norm = np.clip((y[valid] - y_min) / y_scale, 0.0, 1.0)
            x_bin = np.minimum((x_norm * frame_size).astype(np.int64), frame_size - 1)
            y_bin = np.minimum((y_norm * frame_size).astype(np.int64), frame_size - 1)
            for yy, xx in zip(y_bin, x_bin):
                hist[yy, xx] += 1.0
            hist /= max(float(valid.sum()), 1.0)
        seq[i] = hist.reshape(-1)
    return seq


def maybe_add_diffs(x: np.ndarray, use_diff: bool) -> np.ndarray:
    if not use_diff:
        return x
    diffs = np.diff(x, axis=1, prepend=x[:, :1, :])
    return np.concatenate([x, diffs], axis=2)


def load_external_dataset(
    external_root: Path,
    max_external: int,
    num_frames: int,
    frame_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ft0 = sorted((external_root / "ft0").glob("*.mp4"))
    ft1 = sorted((external_root / "ft1").glob("*.mp4"))
    if not ft0 or not ft1:
        raise FileNotFoundError(
            f"Expected mp4 files in both {external_root / 'ft0'} and {external_root / 'ft1'}"
        )

    rng = np.random.default_rng(seed)
    max_per_class = max_external // 2
    n0 = min(len(ft0), max_per_class)
    n1 = min(len(ft1), max_per_class)

    sel0 = rng.choice(ft0, size=n0, replace=False).tolist()
    sel1 = rng.choice(ft1, size=n1, replace=False).tolist()
    paths = sel0 + sel1
    labels = np.array([0.0] * len(sel0) + [1.0] * len(sel1), dtype=np.float32)

    order = rng.permutation(len(paths))
    paths = [paths[i] for i in order]
    labels = labels[order]

    seqs = []
    used_paths = []
    used_labels = []
    for p, lbl in zip(paths, labels):
        try:
            seq = video_to_sequence(p, num_frames=num_frames, frame_size=frame_size)
            seqs.append(seq)
            used_paths.append(str(p))
            used_labels.append(float(lbl))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[warn] failed to decode {p}: {exc}")

    x = np.stack(seqs) if seqs else np.zeros((0, num_frames, frame_size * frame_size), dtype=np.float32)
    y = np.array(used_labels, dtype=np.float32)
    return x, y, used_paths


def load_challenge_dataset(num_frames: int, frame_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scalers = load_scalers()
    seqs = []
    targets = []
    groups = []

    for meta, ts in iterate_shots(train=True):
        seqs.append(keypoint_timeseries_to_sequence(ts, num_frames=num_frames, frame_size=frame_size))
        targets.append([meta["angle"], meta["depth"], meta["left_right"]])
        groups.append(meta["participant_id"])

    x = np.stack(seqs).astype(np.float32)
    y_raw = np.array(targets, dtype=np.float32)
    y_scaled = scale_targets(y_raw, scalers).astype(np.float32)
    groups_arr = np.array(groups)
    return x, y_scaled, groups_arr


def load_challenge_test_sequences(num_frames: int, frame_size: int) -> np.ndarray:
    seqs = []
    for _, ts in iterate_shots(train=False):
        seqs.append(keypoint_timeseries_to_sequence(ts, num_frames=num_frames, frame_size=frame_size))
    return np.stack(seqs).astype(np.float32)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.relu(self.proj(x))
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.dropout(pooled)


class ExternalClassifier(nn.Module):
    def __init__(self, encoder: TemporalEncoder, embed_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z).squeeze(-1)


class ChallengeRegressor(nn.Module):
    def __init__(self, encoder: TemporalEncoder, embed_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


class SequenceMAE(nn.Module):
    def __init__(self, encoder: TemporalEncoder, embed_dim: int, num_frames: int, input_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_frames * input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon.reshape(x.shape[0], self.num_frames, self.input_dim)


def standardize_by_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_val - mean) / std


def pretrain_external(
    x_external: np.ndarray,
    y_external: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
) -> Dict[str, object]:
    set_seed(seed)
    x_tensor = torch.tensor(x_external, dtype=torch.float32)
    y_tensor = torch.tensor(y_external, dtype=torch.float32)

    encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model = ExternalClassifier(encoder, embed_dim=hidden_dim * 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos = float(y_external.sum())
    neg = float(len(y_external) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)
    losses: List[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            total_n += int(xb.size(0))
        epoch_loss = total_loss / max(total_n, 1)
        losses.append(epoch_loss)
        print(f"pretrain_epoch={epoch} bce_loss={epoch_loss:.12f}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_tensor.to(device))).cpu().numpy()
    pred_labels = (probs >= 0.5).astype(np.float32)
    train_acc = float((pred_labels == y_external).mean())

    return {
        "encoder_state_dict": encoder.state_dict(),
        "losses": losses,
        "train_acc": train_acc,
        "pos_weight": float(pos_weight.item()),
        "pretrain_mode": "bce",
    }


def pretrain_external_mae(
    x_unlabeled: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    mask_ratio: float,
) -> Dict[str, object]:
    set_seed(seed)
    x_tensor = torch.tensor(x_unlabeled, dtype=torch.float32)
    num_frames = x_unlabeled.shape[1]

    encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model = SequenceMAE(
        encoder=encoder,
        embed_dim=hidden_dim * 2,
        num_frames=num_frames,
        input_dim=input_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
    losses: List[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for (xb,) in loader:
            xb = xb.to(device)
            mask = torch.rand_like(xb) < mask_ratio
            xb_masked = xb.clone()
            xb_masked[mask] = 0.0
            recon = model(xb_masked)
            loss = ((recon[mask] - xb[mask]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            total_n += int(xb.size(0))
        epoch_loss = total_loss / max(total_n, 1)
        losses.append(epoch_loss)
        print(f"pretrain_epoch={epoch} mae_loss={epoch_loss:.12f}")

    return {
        "encoder_state_dict": encoder.state_dict(),
        "losses": losses,
        "train_acc": float("nan"),
        "pos_weight": float("nan"),
        "pretrain_mode": "mae",
    }


def run_groupkfold_regression(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    pretrained_encoder_state: Dict[str, torch.Tensor] | None,
) -> Dict[str, object]:
    gkf = GroupKFold(n_splits=5)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(x, y, groups), start=1):
        set_seed(seed + fold_idx)
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        x_train_std, x_val_std = standardize_by_train(x_train, x_val)
        x_train_t = torch.tensor(x_train_std, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        x_val_t = torch.tensor(x_val_std, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)

        encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        if pretrained_encoder_state is not None:
            encoder.load_state_dict(pretrained_encoder_state)
        model = ChallengeRegressor(encoder=encoder, embed_dim=hidden_dim * 2).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t.to(device)).cpu().numpy()

        mse_targets = ((pred_val - y_val) ** 2).mean(axis=0)
        mse_total = float(mse_targets.mean())
        fold_metrics.append(
            {
                "fold": fold_idx,
                "mse_angle": float(mse_targets[0]),
                "mse_depth": float(mse_targets[1]),
                "mse_left_right": float(mse_targets[2]),
                "mse_total": mse_total,
            }
        )
        print(
            "fold="
            f"{fold_idx} mse_angle={mse_targets[0]:.12f} mse_depth={mse_targets[1]:.12f} "
            f"mse_left_right={mse_targets[2]:.12f} mse_total={mse_total:.12f}"
        )

    avg = {
        "mse_angle": float(np.mean([m["mse_angle"] for m in fold_metrics])),
        "mse_depth": float(np.mean([m["mse_depth"] for m in fold_metrics])),
        "mse_left_right": float(np.mean([m["mse_left_right"] for m in fold_metrics])),
        "mse_total": float(np.mean([m["mse_total"] for m in fold_metrics])),
    }

    return {
        "folds": fold_metrics,
        "avg": avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Video SSL transfer experiment")
    parser.add_argument("--external-root", type=str, default="Basketball_51 dataset")
    parser.add_argument("--max-external", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--frame-size", type=int, default=8)
    parser.add_argument("--use-diff", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-mode", type=str, default="bce", choices=["bce", "mae"])
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.3)
    parser.add_argument("--include-challenge-train-unlabeled", action="store_true")
    parser.add_argument("--include-challenge-test-unlabeled", action="store_true")
    parser.add_argument("--output-dir", type=str, default="output/video_ssl_transfer")
    args = parser.parse_args()

    out_dir = PROJECT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    external_root = PROJECT_DIR / args.external_root
    x_external, y_external, used_paths = load_external_dataset(
        external_root=external_root,
        max_external=args.max_external,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        seed=args.seed,
    )
    x_external = maybe_add_diffs(x_external, args.use_diff)
    print(f"external_samples={len(x_external)} external_dim={x_external.shape[2]}")
    print(f"external_label_mean={float(y_external.mean()):.12f}")

    x_challenge, y_challenge, groups = load_challenge_dataset(
        num_frames=args.num_frames,
        frame_size=args.frame_size,
    )
    x_challenge = maybe_add_diffs(x_challenge, args.use_diff)
    print(f"challenge_samples={len(x_challenge)} challenge_dim={x_challenge.shape[2]}")

    input_dim = x_challenge.shape[2]

    extra_unlabeled_blocks = []
    if args.pretrain_mode == "mae":
        if args.include_challenge_train_unlabeled:
            extra_unlabeled_blocks.append(x_challenge.copy())
        if args.include_challenge_test_unlabeled:
            x_challenge_test = load_challenge_test_sequences(
                num_frames=args.num_frames,
                frame_size=args.frame_size,
            )
            x_challenge_test = maybe_add_diffs(x_challenge_test, args.use_diff)
            extra_unlabeled_blocks.append(x_challenge_test)

    if extra_unlabeled_blocks:
        x_pretrain_raw = np.concatenate([x_external] + extra_unlabeled_blocks, axis=0)
    else:
        x_pretrain_raw = x_external

    # Standardize pretraining corpus with its own stats.
    pre_mean = x_pretrain_raw.mean(axis=(0, 1), keepdims=True)
    pre_std = x_pretrain_raw.std(axis=(0, 1), keepdims=True) + 1e-6
    x_pretrain_std = (x_pretrain_raw - pre_mean) / pre_std

    if args.pretrain_mode == "bce":
        print("=== pretraining on external video (bce) ===")
        x_external_std = x_pretrain_std[: len(x_external)]
        pretrain = pretrain_external(
            x_external=x_external_std,
            y_external=y_external,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            lr=args.lr_pretrain,
            seed=args.seed,
            device=device,
        )
        print(f"external_train_acc={pretrain['train_acc']:.12f}")
    else:
        print("=== pretraining on external video (mae) ===")
        print(f"pretrain_corpus_samples={len(x_pretrain_std)}")
        pretrain = pretrain_external_mae(
            x_unlabeled=x_pretrain_std,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            lr=args.lr_pretrain,
            seed=args.seed,
            device=device,
            mask_ratio=args.pretrain_mask_ratio,
        )

    print("=== challenge regression: pretrained encoder ===")
    pretrained_result = run_groupkfold_regression(
        x=x_challenge,
        y=y_challenge,
        groups=groups,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.finetune_epochs,
        lr=args.lr_finetune,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
    )
    print(
        "pretrained_avg "
        f"mse_angle={pretrained_result['avg']['mse_angle']:.12f} "
        f"mse_depth={pretrained_result['avg']['mse_depth']:.12f} "
        f"mse_left_right={pretrained_result['avg']['mse_left_right']:.12f} "
        f"mse_total={pretrained_result['avg']['mse_total']:.12f}"
    )

    print("=== challenge regression: no pretraining baseline ===")
    baseline_result = run_groupkfold_regression(
        x=x_challenge,
        y=y_challenge,
        groups=groups,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.finetune_epochs,
        lr=args.lr_finetune,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=None,
    )
    print(
        "baseline_avg "
        f"mse_angle={baseline_result['avg']['mse_angle']:.12f} "
        f"mse_depth={baseline_result['avg']['mse_depth']:.12f} "
        f"mse_left_right={baseline_result['avg']['mse_left_right']:.12f} "
        f"mse_total={baseline_result['avg']['mse_total']:.12f}"
    )

    delta = baseline_result["avg"]["mse_total"] - pretrained_result["avg"]["mse_total"]
    rel = delta / max(baseline_result["avg"]["mse_total"], 1e-12)
    print(f"transfer_delta_total={delta:.12f}")
    print(f"transfer_relative_improvement={rel:.12%}")

    metrics = {
        "config": vars(args),
        "device": device,
        "external_samples_used": int(len(x_external)),
        "external_paths_used": used_paths,
        "external_positive_rate": float(y_external.mean()),
        "pretrain": {
            "losses": [float(x) for x in pretrain["losses"]],
            "train_acc": float_or_none(float(pretrain["train_acc"])),
            "pos_weight": float_or_none(float(pretrain["pos_weight"])),
            "mode": pretrain["pretrain_mode"],
        },
        "challenge_pretrained": pretrained_result,
        "challenge_baseline": baseline_result,
        "transfer_delta_total": float(delta),
        "transfer_relative_improvement": float(rel),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
