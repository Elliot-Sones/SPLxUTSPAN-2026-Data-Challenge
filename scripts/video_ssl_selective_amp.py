#!/usr/bin/env python
"""Inject video-pretrained embeddings into selective amplification workflow.

Steps:
1. Pretrain temporal encoder on external videos (ft0/ft1).
2. Fine-tune regressor on challenge train set (scaled targets).
3. Predict challenge test set with pretrained model.
4. Apply target-specific, per-player disagreement masks against Sub 771.
5. Save candidate submissions and metadata.
"""

from __future__ import annotations

import json
import fcntl
from pathlib import Path
from typing import Dict, List

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from video_ssl_transfer import (
    set_seed,
    maybe_add_diffs,
    load_external_dataset,
    load_challenge_dataset,
    keypoint_timeseries_to_sequence,
    pretrain_external,
    pretrain_external_mae,
    TemporalEncoder,
    ChallengeRegressor,
    run_groupkfold_regression,
)

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import iterate_shots  # noqa: E402

SUBMISSION_DIR = PROJECT_DIR / "submission"


TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def float_or_none(x: float) -> float | None:
    if np.isnan(x):
        return None
    return float(x)


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)

    with lock_path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
            placeholder.touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_challenge_test_dataset(num_frames: int, frame_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs = []
    test_ids = []
    test_pids = []
    for meta, ts in iterate_shots(train=False):
        seqs.append(keypoint_timeseries_to_sequence(ts, num_frames=num_frames, frame_size=frame_size))
        test_ids.append(meta["id"])
        test_pids.append(meta["participant_id"])
    x = np.stack(seqs).astype(np.float32)
    return x, np.array(test_ids), np.array(test_pids)


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def train_full_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    pretrained_encoder_state: Dict[str, torch.Tensor] | None,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    x_train_std, x_test_std = standardize_train_test(x_train, x_test)

    x_train_t = torch.tensor(x_train_std, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_std, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if pretrained_encoder_state is not None:
        encoder.load_state_dict(pretrained_encoder_state)
    model = ChallengeRegressor(encoder=encoder, embed_dim=hidden_dim * 2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
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
        train_pred = model(x_train_t.to(device)).cpu().numpy()
        test_pred = model(x_test_t.to(device)).cpu().numpy()
    return train_pred, test_pred


def apply_target_specific_player_masks(
    base_df: pd.DataFrame,
    alt_df: pd.DataFrame,
    test_pids: np.ndarray,
    config: Dict[str, float],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    result = base_df.copy()
    mask_stats: Dict[str, object] = {}
    selected_masks: Dict[str, np.ndarray] = {}

    for col in TARGET_COLS:
        pctl = float(config[f"{col}_pctl"])
        alpha = float(config[f"{col}_alpha"])
        base = base_df[col].values
        alt = alt_df[col].values
        diff = alt - base

        if alpha <= 0.0:
            mask_stats[col] = {"selected": 0, "fraction": 0.0, "alpha": alpha, "pctl": pctl}
            continue

        selected = np.zeros(len(base), dtype=bool)
        for pid in sorted(np.unique(test_pids)):
            idx = np.where(test_pids == pid)[0]
            if len(idx) == 0:
                continue
            abs_diff = np.abs(diff[idx])
            threshold = np.percentile(abs_diff, pctl)
            local_mask = abs_diff >= threshold
            selected[idx[local_mask]] = True

        new_vals = base.copy()
        new_vals[selected] = base[selected] + alpha * diff[selected]
        new_vals = np.clip(new_vals, 0.0, 1.0)
        result[col] = new_vals
        selected_masks[col] = selected

        mask_stats[col] = {
            "selected": int(selected.sum()),
            "fraction": float(selected.mean()),
            "alpha": alpha,
            "pctl": pctl,
        }

    # Keep depth mean close to anchor, but only adjust already-selected depth rows.
    depth_selected = selected_masks.get("scaled_depth", np.zeros(len(result), dtype=bool))
    if depth_selected.any():
        depth_vals = result["scaled_depth"].values.copy()
        base_depth_mean = float(base_df["scaled_depth"].mean())
        current_depth_mean = float(depth_vals.mean())
        correction = base_depth_mean - current_depth_mean
        depth_vals[depth_selected] = np.clip(depth_vals[depth_selected] + correction, 0.0, 1.0)
        result["scaled_depth"] = depth_vals

    # Final lightweight mean lock via selected rows only.
    target_depth_mean = float(base_df["scaled_depth"].mean())
    if depth_selected.any():
        depth = result["scaled_depth"].values.copy()
        residual = target_depth_mean - float(depth.mean())
        depth[depth_selected] = np.clip(depth[depth_selected] + residual, 0.0, 1.0)
        result["scaled_depth"] = depth

    profile = {
        "angle_std": float(result["scaled_angle"].std()),
        "depth_mean": float(result["scaled_depth"].mean()),
        "left_right_std": float(result["scaled_left_right"].std()),
    }
    return result, {"mask_stats": mask_stats, "profile": profile}


def correlation_report(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for col in TARGET_COLS:
        out[col] = float(np.corrcoef(a[col].values, b[col].values)[0, 1])
    return out


def build_default_candidate_configs() -> List[Dict[str, float]]:
    # Target-specific, per-player mask percentiles and move strengths.
    return [
        {
            "name": "ssl_micro_depth_p98_a004",
            "scaled_angle_pctl": 99,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 98,
            "scaled_depth_alpha": 0.04,
            "scaled_left_right_pctl": 99,
            "scaled_left_right_alpha": 0.00,
        },
        {
            "name": "ssl_micro_depth_p96_a005",
            "scaled_angle_pctl": 99,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 96,
            "scaled_depth_alpha": 0.05,
            "scaled_left_right_pctl": 99,
            "scaled_left_right_alpha": 0.00,
        },
        {
            "name": "ssl_micro_depth_p95_a008",
            "scaled_angle_pctl": 99,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 95,
            "scaled_depth_alpha": 0.08,
            "scaled_left_right_pctl": 99,
            "scaled_left_right_alpha": 0.00,
        },
        {
            "name": "ssl_ultra_safe_depth",
            "scaled_angle_pctl": 98,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 95,
            "scaled_depth_alpha": 0.06,
            "scaled_left_right_pctl": 98,
            "scaled_left_right_alpha": 0.00,
        },
        {
            "name": "ssl_safe_depth",
            "scaled_angle_pctl": 95,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 90,
            "scaled_depth_alpha": 0.10,
            "scaled_left_right_pctl": 95,
            "scaled_left_right_alpha": 0.00,
        },
        {
            "name": "ssl_safe_depth_lr",
            "scaled_angle_pctl": 95,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 90,
            "scaled_depth_alpha": 0.10,
            "scaled_left_right_pctl": 90,
            "scaled_left_right_alpha": 0.10,
        },
        {
            "name": "ssl_medium_depth_lr",
            "scaled_angle_pctl": 95,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 85,
            "scaled_depth_alpha": 0.18,
            "scaled_left_right_pctl": 85,
            "scaled_left_right_alpha": 0.18,
        },
        {
            "name": "ssl_lr_aggressive",
            "scaled_angle_pctl": 95,
            "scaled_angle_alpha": 0.00,
            "scaled_depth_pctl": 88,
            "scaled_depth_alpha": 0.15,
            "scaled_left_right_pctl": 80,
            "scaled_left_right_alpha": 0.25,
        },
        {
            "name": "ssl_all_targets_low",
            "scaled_angle_pctl": 92,
            "scaled_angle_alpha": 0.05,
            "scaled_depth_pctl": 88,
            "scaled_depth_alpha": 0.15,
            "scaled_left_right_pctl": 85,
            "scaled_left_right_alpha": 0.20,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Video SSL selective amplification submission generator")
    parser.add_argument("--external-root", type=str, default="Basketball_51 dataset")
    parser.add_argument("--max-external", type=int, default=1200)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--frame-size", type=int, default=8)
    parser.add_argument("--use-diff", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-mode", type=str, default="bce", choices=["bce", "mae"])
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.3)
    parser.add_argument("--include-challenge-train-unlabeled", action="store_true")
    parser.add_argument("--include-challenge-test-unlabeled", action="store_true")
    parser.add_argument("--base-submission", type=int, default=771)
    parser.add_argument("--save-direct", action="store_true")
    parser.add_argument("--output-dir", type=str, default="output/video_ssl_selective_amp")
    args = parser.parse_args()

    out_dir = PROJECT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # Load external video dataset and pretrain encoder.
    x_external, y_external, used_paths = load_external_dataset(
        external_root=PROJECT_DIR / args.external_root,
        max_external=args.max_external,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        seed=args.seed,
    )
    x_external = maybe_add_diffs(x_external, args.use_diff)

    print(f"external_samples={len(x_external)} external_dim={x_external.shape[2]}")
    print(f"external_positive_rate={float(y_external.mean()):.12f}")

    # Load challenge train/test in matching sequence format.
    x_train, y_train, groups = load_challenge_dataset(num_frames=args.num_frames, frame_size=args.frame_size)
    x_test, test_ids, test_pids = load_challenge_test_dataset(num_frames=args.num_frames, frame_size=args.frame_size)
    x_train = maybe_add_diffs(x_train, args.use_diff)
    x_test = maybe_add_diffs(x_test, args.use_diff)

    print(f"challenge_train_samples={len(x_train)} challenge_test_samples={len(x_test)}")
    print(f"challenge_dim={x_train.shape[2]}")

    extra_unlabeled_blocks = []
    if args.pretrain_mode == "mae":
        if args.include_challenge_train_unlabeled:
            extra_unlabeled_blocks.append(x_train.copy())
        if args.include_challenge_test_unlabeled:
            extra_unlabeled_blocks.append(x_test.copy())

    if extra_unlabeled_blocks:
        x_pretrain_raw = np.concatenate([x_external] + extra_unlabeled_blocks, axis=0)
    else:
        x_pretrain_raw = x_external

    pre_mean = x_pretrain_raw.mean(axis=(0, 1), keepdims=True)
    pre_std = x_pretrain_raw.std(axis=(0, 1), keepdims=True) + 1e-6
    x_pretrain_std = (x_pretrain_raw - pre_mean) / pre_std

    if args.pretrain_mode == "bce":
        x_external_std = x_pretrain_std[: len(x_external)]
        pretrain = pretrain_external(
            x_external=x_external_std,
            y_external=y_external,
            input_dim=x_external.shape[2],
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
        print(f"pretrain_corpus_samples={len(x_pretrain_std)}")
        pretrain = pretrain_external_mae(
            x_unlabeled=x_pretrain_std,
            input_dim=x_external.shape[2],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            lr=args.lr_pretrain,
            seed=args.seed,
            device=device,
            mask_ratio=args.pretrain_mask_ratio,
        )
        print(f"external_train_acc=nan pretrain_mode=mae")

    # Measure CV of pretrained model.
    cv_pretrained = run_groupkfold_regression(
        x=x_train,
        y=y_train,
        groups=groups,
        input_dim=x_train.shape[2],
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
        "cv_pretrained "
        f"mse_angle={cv_pretrained['avg']['mse_angle']:.12f} "
        f"mse_depth={cv_pretrained['avg']['mse_depth']:.12f} "
        f"mse_left_right={cv_pretrained['avg']['mse_left_right']:.12f} "
        f"mse_total={cv_pretrained['avg']['mse_total']:.12f}"
    )

    # Train full model and predict test.
    _, test_pred = train_full_and_predict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.finetune_epochs,
        lr=args.lr_finetune,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
    )

    ssl_df = pd.DataFrame(
        {
            "id": test_ids,
            "scaled_angle": np.clip(test_pred[:, 0], 0.0, 1.0),
            "scaled_depth": np.clip(test_pred[:, 1], 0.0, 1.0),
            "scaled_left_right": np.clip(test_pred[:, 2], 0.0, 1.0),
        }
    )

    direct_record = None
    if args.save_direct:
        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        ssl_df.to_csv(sub_path, index=False)
        direct_record = {
            "submission_num": sub_num,
            "submission_file": str(sub_path),
            "config_name": "ssl_direct",
            "profile": {
                "angle_std": float(ssl_df["scaled_angle"].std()),
                "depth_mean": float(ssl_df["scaled_depth"].mean()),
                "left_right_std": float(ssl_df["scaled_left_right"].std()),
            },
        }
        print(
            f"saved_submission={sub_num} name=ssl_direct "
            f"angle_std={direct_record['profile']['angle_std']:.12f} "
            f"depth_mean={direct_record['profile']['depth_mean']:.12f}"
        )

    base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Base submission not found: {base_path}")
    base_df = pd.read_csv(base_path)

    # Align by id.
    base_df = base_df.sort_values("id").reset_index(drop=True)
    ssl_df = ssl_df.sort_values("id").reset_index(drop=True)
    if not np.array_equal(base_df["id"].values, ssl_df["id"].values):
        raise ValueError("ID alignment mismatch between base submission and SSL predictions")

    corr = correlation_report(base_df, ssl_df)
    print(
        "corr_with_base "
        f"angle={corr['scaled_angle']:.12f} "
        f"depth={corr['scaled_depth']:.12f} "
        f"left_right={corr['scaled_left_right']:.12f}"
    )

    # Generate candidate submissions.
    candidate_configs = build_default_candidate_configs()
    candidate_records = []
    for cfg in candidate_configs:
        cand_df, stats = apply_target_specific_player_masks(
            base_df=base_df,
            alt_df=ssl_df,
            test_pids=test_pids,
            config=cfg,
        )
        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        cand_df.to_csv(sub_path, index=False)

        record = {
            "submission_num": sub_num,
            "submission_file": str(sub_path),
            "config_name": cfg["name"],
            "config": cfg,
            "profile": stats["profile"],
            "mask_stats": stats["mask_stats"],
            "corr_with_base": correlation_report(base_df, cand_df),
            "corr_with_ssl": correlation_report(ssl_df, cand_df),
        }
        candidate_records.append(record)
        print(
            f"saved_submission={sub_num} name={cfg['name']} "
            f"angle_std={stats['profile']['angle_std']:.12f} "
            f"depth_mean={stats['profile']['depth_mean']:.12f}"
        )

    # Save metadata.
    run_meta = {
        "config": vars(args),
        "device": device,
        "external_samples_used": int(len(x_external)),
        "external_positive_rate": float(y_external.mean()),
        "external_paths_used": used_paths,
        "pretrain": {
            "train_acc": float_or_none(float(pretrain["train_acc"])),
            "losses": [float(x) for x in pretrain["losses"]],
            "pos_weight": float_or_none(float(pretrain["pos_weight"])),
            "mode": pretrain["pretrain_mode"],
        },
        "cv_pretrained": cv_pretrained,
        "base_submission": int(args.base_submission),
        "base_profile": {
            "angle_std": float(base_df["scaled_angle"].std()),
            "depth_mean": float(base_df["scaled_depth"].mean()),
            "left_right_std": float(base_df["scaled_left_right"].std()),
        },
        "ssl_profile": {
            "angle_std": float(ssl_df["scaled_angle"].std()),
            "depth_mean": float(ssl_df["scaled_depth"].mean()),
            "left_right_std": float(ssl_df["scaled_left_right"].std()),
        },
        "ssl_vs_base_corr": corr,
        "direct_submission": direct_record,
        "candidates": candidate_records,
    }
    metrics_path = out_dir / "run_metrics.json"
    metrics_path.write_text(json.dumps(run_meta, indent=2))
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
