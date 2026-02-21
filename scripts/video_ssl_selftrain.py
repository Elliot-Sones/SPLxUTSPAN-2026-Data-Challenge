#!/usr/bin/env python
"""Video SSL pretraining + teacher-student self-training for submission generation.

Pipeline:
1. Pretrain temporal encoder with MAE on external videos plus optional challenge unlabeled data.
2. Train teacher ensemble on challenge train labels.
3. Build uncertainty-weighted pseudo labels on challenge test.
4. Train student on train labels + pseudo-labeled test.
5. Write direct model submission and optional base-blend variants.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from video_ssl_transfer import (
    ChallengeRegressor,
    TemporalEncoder,
    keypoint_timeseries_to_sequence,
    load_challenge_dataset,
    load_external_dataset,
    maybe_add_diffs,
    pretrain_external_mae,
    run_groupkfold_regression,
    set_seed,
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


def load_challenge_test_dataset(num_frames: int, frame_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs = []
    test_ids = []
    test_pids = []
    for meta, ts in iterate_shots(train=False):
        seqs.append(keypoint_timeseries_to_sequence(ts, num_frames=num_frames, frame_size=frame_size))
        test_ids.append(meta["id"])
        test_pids.append(meta["participant_id"])
    x = np.stack(seqs).astype(np.float32)
    return x, np.array(test_ids), np.array(test_pids)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std
    return x_train_std, x_test_std, mean, std


def train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray | None,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    pretrained_encoder_state: Dict[str, torch.Tensor] | None,
) -> Tuple[ChallengeRegressor, np.ndarray, np.ndarray]:
    set_seed(seed)

    if sample_weights is None:
        sample_weights = np.ones(len(x_train), dtype=np.float32)

    x_train_std, _, mean, std = standardize_train_test(x_train, x_train)

    x_train_t = torch.tensor(x_train_std, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(sample_weights.astype(np.float32), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t, w_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if pretrained_encoder_state is not None:
        encoder.load_state_dict(pretrained_encoder_state)
    model = ChallengeRegressor(encoder=encoder, embed_dim=hidden_dim * 2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            pred = model(xb)
            per_sample = ((pred - yb) ** 2).mean(dim=1)
            loss = (per_sample * wb).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model, mean, std


def predict_regressor(
    model: ChallengeRegressor,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> np.ndarray:
    x_std = (x - mean) / std
    x_t = torch.tensor(x_std, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(x_t.to(device)).cpu().numpy()
    return pred


def ensemble_teacher_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    pretrained_encoder_state: Dict[str, torch.Tensor] | None,
    teacher_seeds: int,
    sample_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    for i in range(teacher_seeds):
        local_seed = seed + 1000 * (i + 1)
        model, mean, std = train_regressor(
            x_train=x_train,
            y_train=y_train,
            sample_weights=sample_weights,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            seed=local_seed,
            device=device,
            pretrained_encoder_state=pretrained_encoder_state,
        )
        pred = predict_regressor(model=model, x=x_eval, mean=mean, std=std, device=device)
        preds.append(pred)

    pred_arr = np.stack(preds, axis=0)
    pred_mean = pred_arr.mean(axis=0)
    pred_std = pred_arr.std(axis=0)
    return pred_mean, pred_std


def std_to_confidence_weights(
    pred_std: np.ndarray,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    unc = pred_std.mean(axis=1)
    umin = float(unc.min())
    umax = float(unc.max())
    if umax - umin < 1e-12:
        conf = np.ones_like(unc)
    else:
        conf = 1.0 - (unc - umin) / (umax - umin)
    return min_weight + conf * (max_weight - min_weight)


def evaluate_groupkfold_teacher_student(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    teacher_epochs: int,
    student_epochs: int,
    lr: float,
    seed: int,
    device: str,
    pretrained_encoder_state: Dict[str, torch.Tensor] | None,
    teacher_seeds: int,
    pseudo_weight_min: float,
    pseudo_weight_max: float,
    cv_folds: int,
) -> Dict[str, object]:
    gkf = GroupKFold(n_splits=cv_folds)
    teacher_mses = []
    student_mses = []

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        x_tr = x[tr_idx]
        y_tr = y[tr_idx]
        x_va = x[va_idx]
        y_va = y[va_idx]

        teacher_mean, teacher_std = ensemble_teacher_predictions(
            x_train=x_tr,
            y_train=y_tr,
            x_eval=x_va,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            epochs=teacher_epochs,
            lr=lr,
            seed=seed + 10000 * fold_idx,
            device=device,
            pretrained_encoder_state=pretrained_encoder_state,
            teacher_seeds=teacher_seeds,
            sample_weights=None,
        )
        teacher_mse = float(((teacher_mean - y_va) ** 2).mean())
        teacher_mses.append(teacher_mse)

        pseudo_w = std_to_confidence_weights(
            pred_std=teacher_std,
            min_weight=pseudo_weight_min,
            max_weight=pseudo_weight_max,
        )

        x_student = np.concatenate([x_tr, x_va], axis=0)
        y_student = np.concatenate([y_tr, teacher_mean], axis=0)
        w_student = np.concatenate([np.ones(len(x_tr), dtype=np.float32), pseudo_w.astype(np.float32)], axis=0)

        student_model, mean, std = train_regressor(
            x_train=x_student,
            y_train=y_student,
            sample_weights=w_student,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            epochs=student_epochs,
            lr=lr,
            seed=seed + 20000 * fold_idx,
            device=device,
            pretrained_encoder_state=pretrained_encoder_state,
        )
        student_pred = predict_regressor(model=student_model, x=x_va, mean=mean, std=std, device=device)
        student_mse = float(((student_pred - y_va) ** 2).mean())
        student_mses.append(student_mse)

        print(
            f"fold={fold_idx} "
            f"teacher_mse={teacher_mse:.12f} "
            f"student_mse={student_mse:.12f} "
            f"delta={teacher_mse - student_mse:.12f}"
        )

    teacher_avg = float(np.mean(teacher_mses))
    student_avg = float(np.mean(student_mses))
    return {
        "teacher_fold_mse": teacher_mses,
        "student_fold_mse": student_mses,
        "teacher_avg_mse": teacher_avg,
        "student_avg_mse": student_avg,
        "student_minus_teacher": float(student_avg - teacher_avg),
    }


def blend_with_base(
    ids: np.ndarray,
    direct_pred: np.ndarray,
    base_submission_num: int,
    blend_weights: List[float],
) -> List[Dict[str, object]]:
    base_path = SUBMISSION_DIR / f"submission_{base_submission_num}.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Base submission not found: {base_path}")

    base_df = pd.read_csv(base_path).sort_values("id").reset_index(drop=True)
    direct_df = pd.DataFrame(
        {
            "id": ids,
            "scaled_angle": np.clip(direct_pred[:, 0], 0.0, 1.0),
            "scaled_depth": np.clip(direct_pred[:, 1], 0.0, 1.0),
            "scaled_left_right": np.clip(direct_pred[:, 2], 0.0, 1.0),
        }
    ).sort_values("id").reset_index(drop=True)

    if not np.array_equal(base_df["id"].values, direct_df["id"].values):
        raise ValueError("ID alignment mismatch between base submission and direct student predictions")

    records = []
    for w in blend_weights:
        out = base_df.copy()
        for c in TARGET_COLS:
            out[c] = np.clip((1.0 - w) * base_df[c].values + w * direct_df[c].values, 0.0, 1.0)
        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        out.to_csv(sub_path, index=False)
        records.append(
            {
                "submission_num": int(sub_num),
                "submission_file": str(sub_path),
                "blend_weight_direct": float(w),
                "angle_std": float(out["scaled_angle"].std()),
                "depth_mean": float(out["scaled_depth"].mean()),
                "left_right_std": float(out["scaled_left_right"].std()),
            }
        )
        print(
            f"saved_blend_submission={sub_num} "
            f"w_direct={w:.6f} "
            f"angle_std={records[-1]['angle_std']:.12f} "
            f"depth_mean={records[-1]['depth_mean']:.12f}"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Video SSL teacher-student self-training submission generator")
    parser.add_argument("--external-root", type=str, default="Basketball_51 dataset")
    parser.add_argument("--max-external", type=int, default=1132)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--frame-size", type=int, default=8)
    parser.add_argument("--use-diff", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--teacher-epochs", type=int, default=6)
    parser.add_argument("--student-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-supervised", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-seeds", type=int, default=3)
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.3)
    parser.add_argument("--include-challenge-train-unlabeled", action="store_true")
    parser.add_argument("--include-challenge-test-unlabeled", action="store_true")
    parser.add_argument("--pseudo-weight-min", type=float, default=0.15)
    parser.add_argument("--pseudo-weight-max", type=float, default=0.65)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--base-submission", type=int, default=771)
    parser.add_argument("--blend-weights", type=float, nargs="*", default=[0.15, 0.25, 0.35, 0.50, 0.70, 1.0])
    parser.add_argument("--output-dir", type=str, default="output/video_ssl_selftrain")
    args = parser.parse_args()

    out_dir = PROJECT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    x_external, y_external, used_paths = load_external_dataset(
        external_root=PROJECT_DIR / args.external_root,
        max_external=args.max_external,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        seed=args.seed,
    )
    x_external = maybe_add_diffs(x_external, args.use_diff)

    x_train, y_train, groups = load_challenge_dataset(
        num_frames=args.num_frames,
        frame_size=args.frame_size,
    )
    x_train = maybe_add_diffs(x_train, args.use_diff)
    x_test, test_ids, test_pids = load_challenge_test_dataset(
        num_frames=args.num_frames,
        frame_size=args.frame_size,
    )
    x_test = maybe_add_diffs(x_test, args.use_diff)

    print(f"external_samples={len(x_external)} external_dim={x_external.shape[2]}")
    print(f"external_positive_rate={float(y_external.mean()):.12f}")
    print(f"challenge_train_samples={len(x_train)} challenge_test_samples={len(x_test)}")
    print(f"challenge_dim={x_train.shape[2]}")

    extra_unlabeled_blocks = []
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

    print(f"pretrain_corpus_samples={len(x_pretrain_std)}")
    pretrain = pretrain_external_mae(
        x_unlabeled=x_pretrain_std,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.pretrain_epochs,
        lr=args.lr_pretrain,
        seed=args.seed,
        device=device,
        mask_ratio=args.pretrain_mask_ratio,
    )
    print(f"pretrain_last_loss={pretrain['losses'][-1]:.12f}")

    teacher_cv = run_groupkfold_regression(
        x=x_train,
        y=y_train,
        groups=groups,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.teacher_epochs,
        lr=args.lr_supervised,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
    )
    print(
        "cv_teacher "
        f"mse_angle={teacher_cv['avg']['mse_angle']:.12f} "
        f"mse_depth={teacher_cv['avg']['mse_depth']:.12f} "
        f"mse_left_right={teacher_cv['avg']['mse_left_right']:.12f} "
        f"mse_total={teacher_cv['avg']['mse_total']:.12f}"
    )

    ts_cv = evaluate_groupkfold_teacher_student(
        x=x_train,
        y=y_train,
        groups=groups,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        lr=args.lr_supervised,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
        teacher_seeds=args.teacher_seeds,
        pseudo_weight_min=args.pseudo_weight_min,
        pseudo_weight_max=args.pseudo_weight_max,
        cv_folds=args.cv_folds,
    )
    print(
        "cv_teacher_student "
        f"teacher_avg_mse={ts_cv['teacher_avg_mse']:.12f} "
        f"student_avg_mse={ts_cv['student_avg_mse']:.12f} "
        f"student_minus_teacher={ts_cv['student_minus_teacher']:.12f}"
    )

    teacher_test_mean, teacher_test_std = ensemble_teacher_predictions(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_test,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.teacher_epochs,
        lr=args.lr_supervised,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
        teacher_seeds=args.teacher_seeds,
        sample_weights=None,
    )
    pseudo_weights_test = std_to_confidence_weights(
        pred_std=teacher_test_std,
        min_weight=args.pseudo_weight_min,
        max_weight=args.pseudo_weight_max,
    )
    print(
        "pseudo_weight_stats "
        f"min={float(pseudo_weights_test.min()):.12f} "
        f"mean={float(pseudo_weights_test.mean()):.12f} "
        f"max={float(pseudo_weights_test.max()):.12f}"
    )

    x_student = np.concatenate([x_train, x_test], axis=0)
    y_student = np.concatenate([y_train, teacher_test_mean], axis=0)
    w_student = np.concatenate([np.ones(len(x_train), dtype=np.float32), pseudo_weights_test.astype(np.float32)], axis=0)

    student_model, student_mean, student_std = train_regressor(
        x_train=x_student,
        y_train=y_student,
        sample_weights=w_student,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.student_epochs,
        lr=args.lr_supervised,
        seed=args.seed + 777,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
    )

    student_test_pred = predict_regressor(
        model=student_model,
        x=x_test,
        mean=student_mean,
        std=student_std,
        device=device,
    )
    student_test_pred = np.clip(student_test_pred, 0.0, 1.0)

    direct_df = pd.DataFrame(
        {
            "id": test_ids,
            "scaled_angle": student_test_pred[:, 0],
            "scaled_depth": student_test_pred[:, 1],
            "scaled_left_right": student_test_pred[:, 2],
        }
    )
    sub_num_direct = get_next_submission_number()
    sub_path_direct = SUBMISSION_DIR / f"submission_{sub_num_direct}.csv"
    direct_df.to_csv(sub_path_direct, index=False)
    print(
        f"saved_submission={sub_num_direct} name=student_direct "
        f"angle_std={float(direct_df['scaled_angle'].std()):.12f} "
        f"depth_mean={float(direct_df['scaled_depth'].mean()):.12f}"
    )

    blend_records = blend_with_base(
        ids=test_ids,
        direct_pred=student_test_pred,
        base_submission_num=args.base_submission,
        blend_weights=args.blend_weights,
    )

    run_meta = {
        "config": vars(args),
        "device": device,
        "external_samples_used": int(len(x_external)),
        "external_positive_rate": float(y_external.mean()),
        "external_paths_used": used_paths,
        "pretrain": {
            "mode": pretrain["pretrain_mode"],
            "losses": [float(x) for x in pretrain["losses"]],
            "train_acc": float_or_none(float(pretrain["train_acc"])),
            "pos_weight": float_or_none(float(pretrain["pos_weight"])),
        },
        "cv_teacher": teacher_cv,
        "cv_teacher_student": ts_cv,
        "teacher_test_uncertainty": {
            "mean_std_angle": float(teacher_test_std[:, 0].mean()),
            "mean_std_depth": float(teacher_test_std[:, 1].mean()),
            "mean_std_left_right": float(teacher_test_std[:, 2].mean()),
            "pseudo_weight_min": float(pseudo_weights_test.min()),
            "pseudo_weight_mean": float(pseudo_weights_test.mean()),
            "pseudo_weight_max": float(pseudo_weights_test.max()),
        },
        "student_direct_submission": {
            "submission_num": int(sub_num_direct),
            "submission_file": str(sub_path_direct),
            "angle_std": float(direct_df["scaled_angle"].std()),
            "depth_mean": float(direct_df["scaled_depth"].mean()),
            "left_right_std": float(direct_df["scaled_left_right"].std()),
        },
        "blend_submissions": blend_records,
    }
    metrics_path = out_dir / "run_metrics.json"
    metrics_path.write_text(json.dumps(run_meta, indent=2))
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
