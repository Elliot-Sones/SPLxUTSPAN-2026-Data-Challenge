"""
Probabilistic gate ensemble search.

This script does three things:
1. Generates blend candidates around strong known submissions.
2. Fits an uncertainty-aware leaderboard surrogate via bootstrap ridge.
3. Writes top candidates as numbered submission files and full run artifacts.

Design constraints:
- No guessed leaderboard labels are used.
- Manual labels are only added where exact values are documented in repo research.
- Candidate ranking is uncertainty-aware (mean + tail risk), not point estimate only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]

# Exact modern anchors documented in Research/FIVE_APPROACHES_RESULTS.md.
MANUAL_EXACT_LB = {
    2020: 0.006619,
    2063: 0.006603,
    2151: 0.006605,
    2152: 0.006604,
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    template: str
    base_sub: int
    weights: Dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probabilistic gate ensemble search")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor for search breadth")
    parser.add_argument("--seed", type=int, default=20260214, help="Random seed")
    parser.add_argument("--top-k", type=int, default=5, help="Number of submissions to export")
    parser.add_argument(
        "--best-lb",
        type=float,
        default=0.006603,
        help="Current best known LB to beat",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=Path("submission"),
        help="Submission directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory",
    )
    parser.add_argument(
        "--anchors-csv",
        type=Path,
        default=Path("output/lb_table_parse_summary_2026-02-10.csv"),
        help="High-confidence LB table parse summary",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional run tag. Default: UTC timestamp",
    )
    parser.add_argument(
        "--base-subs",
        type=str,
        default="2063,2151,2152",
        help="Comma-separated base submission ids",
    )
    parser.add_argument(
        "--injector-subs",
        type=str,
        default="1506,1546,1570,1589,2020,1828",
        help="Comma-separated injector submission ids",
    )
    parser.add_argument(
        "--reference-subs",
        type=str,
        default="784,1109,1350,1640,1828,2020,2063,2151,2152,1506",
        help="Comma-separated reference submission ids for feature construction",
    )
    parser.add_argument(
        "--extra-anchors",
        type=str,
        default="",
        help="Optional extra exact LB anchors in format sub=lb,sub=lb",
    )
    return parser.parse_args()


def parse_sub_list(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Parsed empty submission list")
    return values


def parse_extra_anchors(text: str) -> Dict[int, float]:
    if not text.strip():
        return {}
    result: Dict[int, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid anchor token: {token}")
        sub_text, lb_text = token.split("=", 1)
        sub = int(sub_text.strip())
        lb = float(lb_text.strip())
        result[sub] = lb
    return result


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-12 or b_std < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def get_next_submission_number(submission_dir: Path) -> int:
    lock_path = submission_dir / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            existing = list(submission_dir.glob("submission_*.csv"))
            nums = [
                int(path.stem.split("_")[1])
                for path in existing
                if path.stem.split("_")[1].isdigit()
            ]
            next_num = max(nums) + 1 if nums else 1
            # Reserve the number immediately.
            (submission_dir / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_submission_matrix(
    sub_num: int,
    submission_dir: Path,
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
    id_reference: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if sub_num in cache:
        return cache[sub_num]
    path = submission_dir / f"submission_{sub_num}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing submission file: {path}")
    df = pd.read_csv(path)
    ids = df["id"].to_numpy(copy=True)
    preds = df[TARGET_COLS].to_numpy(dtype=float, copy=True)
    if id_reference is not None and not np.array_equal(ids, id_reference):
        raise ValueError(f"id column mismatch in submission_{sub_num}.csv")
    cache[sub_num] = (ids, preds)
    return ids, preds


def build_anchor_lb_map(
    anchors_csv: Path,
    submission_dir: Path,
    extra_anchors: Dict[int, float],
) -> Dict[int, float]:
    if not anchors_csv.exists():
        raise FileNotFoundError(f"Missing anchors csv: {anchors_csv}")
    summary = pd.read_csv(anchors_csv)
    required_cols = {"sub", "lb_mode"}
    missing = required_cols - set(summary.columns)
    if missing:
        raise ValueError(f"Anchors csv missing columns: {sorted(missing)}")

    lb_map: Dict[int, float] = {}
    for row in summary.itertuples(index=False):
        sub = int(row.sub)
        path = submission_dir / f"submission_{sub}.csv"
        if path.exists():
            lb_map[sub] = float(row.lb_mode)

    # Add exact modern anchors when files are present.
    for sub, lb in MANUAL_EXACT_LB.items():
        path = submission_dir / f"submission_{sub}.csv"
        if path.exists():
            lb_map[sub] = lb
    # Add optional user-provided anchors when files are present.
    for sub, lb in extra_anchors.items():
        path = submission_dir / f"submission_{sub}.csv"
        if path.exists():
            lb_map[sub] = lb
    return lb_map


def build_feature_names(reference_subs: List[int]) -> List[str]:
    names: List[str] = []
    for target in TARGET_COLS:
        names.append(f"{target}_mean")
        names.append(f"{target}_std")
    for ref in reference_subs:
        for target in TARGET_COLS:
            names.append(f"{target}_corr_ref{ref}")
            names.append(f"{target}_mse_ref{ref}")
            names.append(f"{target}_mae_ref{ref}")
    return names


def build_feature_vector(
    preds: np.ndarray,
    reference_preds: Dict[int, np.ndarray],
    reference_subs: List[int],
) -> np.ndarray:
    feats: List[float] = []
    for t_idx in range(preds.shape[1]):
        target = preds[:, t_idx]
        feats.append(float(np.mean(target)))
        feats.append(float(np.std(target)))
    for ref in reference_subs:
        ref_arr = reference_preds[ref]
        for t_idx in range(preds.shape[1]):
            target = preds[:, t_idx]
            ref_target = ref_arr[:, t_idx]
            diff = target - ref_target
            feats.append(corr_safe(target, ref_target))
            feats.append(float(np.mean(diff * diff)))
            feats.append(float(np.mean(np.abs(diff))))
    return np.asarray(feats, dtype=float)


def standardize_fit_transform(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    x_norm = (x - mean) / std
    return x_norm, mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_norm, mean, std = standardize_fit_transform(x_train)
    x_pred_norm = standardize_apply(x_pred, mean, std)

    y_mean = float(np.mean(y_train))
    y_center = y_train - y_mean

    xtx = x_norm.T @ x_norm
    xty = x_norm.T @ y_center
    mat = xtx + alpha * np.eye(xtx.shape[0], dtype=float)

    try:
        coef = np.linalg.solve(mat, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(mat) @ xty
    return x_pred_norm @ coef + y_mean


def loocv_alpha_search(x: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> Tuple[float, float, float]:
    best_alpha = float(alphas[0])
    best_mae = float("inf")
    best_rmse = float("inf")

    for alpha in alphas:
        preds = np.zeros_like(y)
        for i in range(len(y)):
            mask = np.ones(len(y), dtype=bool)
            mask[i] = False
            pred_i = ridge_predict(x[mask], y[mask], x[i : i + 1], float(alpha))
            preds[i] = pred_i[0]
        mae = float(np.mean(np.abs(preds - y)))
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        if mae < best_mae:
            best_alpha = float(alpha)
            best_mae = mae
            best_rmse = rmse
    return best_alpha, best_mae, best_rmse


def generate_candidate_specs(
    base_subs: List[int],
    injector_subs: List[int],
    scale: int,
    seed: int,
) -> List[CandidateSpec]:
    rng = np.random.default_rng(seed)
    specs: List[CandidateSpec] = []
    counter = 0

    # Include unmodified bases for diagnostics.
    for base in base_subs:
        specs.append(
            CandidateSpec(
                candidate_id=f"cand_{counter:05d}",
                template="base_only",
                base_sub=base,
                weights={base: 1.0},
            )
        )
        counter += 1

    single_steps = 12 * scale
    single_weights = np.linspace(0.01, 0.25, single_steps)
    for base in base_subs:
        for inj in injector_subs:
            if inj == base:
                continue
            for w in single_weights:
                weights = {base: float(1.0 - w), inj: float(w)}
                specs.append(
                    CandidateSpec(
                        candidate_id=f"cand_{counter:05d}",
                        template="single_inject",
                        base_sub=base,
                        weights=weights,
                    )
                )
                counter += 1

    pair_draws = 20 * scale
    for base in base_subs:
        pool = [s for s in injector_subs if s != base]
        for _ in range(pair_draws):
            i, j = rng.choice(pool, size=2, replace=False)
            total_w = float(rng.uniform(0.02, 0.28))
            frac = float(rng.beta(2.0, 2.0))
            wi = total_w * frac
            wj = total_w * (1.0 - frac)
            weights = {
                base: float(1.0 - total_w),
                int(i): float(wi),
                int(j): float(wj),
            }
            specs.append(
                CandidateSpec(
                    candidate_id=f"cand_{counter:05d}",
                    template="pair_inject",
                    base_sub=base,
                    weights=weights,
                )
            )
            counter += 1

    # Fine search around the proven high-performing region between 2063 and 2020.
    local_steps = 8 * scale
    local_weights = np.linspace(0.15, 0.55, local_steps)
    for w in local_weights:
        specs.append(
            CandidateSpec(
                candidate_id=f"cand_{counter:05d}",
                template="local_2063_2020",
                base_sub=2063,
                weights={2063: float(1.0 - w), 2020: float(w)},
            )
        )
        counter += 1
        specs.append(
            CandidateSpec(
                candidate_id=f"cand_{counter:05d}",
                template="local_2063_1828",
                base_sub=2063,
                weights={2063: float(1.0 - w), 1828: float(w)},
            )
        )
        counter += 1

    return specs


def blend_predictions(
    weights: Dict[int, float],
    pred_map: Dict[int, np.ndarray],
) -> np.ndarray:
    output = None
    for sub, weight in weights.items():
        arr = pred_map[sub]
        if output is None:
            output = weight * arr
        else:
            output += weight * arr
    if output is None:
        raise ValueError("Empty weight dictionary")
    return np.clip(output, 0.0, 1.0)


def mean_target_corr(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for t_idx in range(a.shape[1]):
        vals.append(corr_safe(a[:, t_idx], b[:, t_idx]))
    return float(np.mean(vals))


def main() -> None:
    t0 = time.time()
    args = parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    run_tag = args.run_tag.strip() or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    submission_dir = args.submission_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Candidate search sets.
    base_subs = parse_sub_list(args.base_subs)
    injector_subs = parse_sub_list(args.injector_subs)
    reference_subs = parse_sub_list(args.reference_subs)
    extra_anchors = parse_extra_anchors(args.extra_anchors)

    cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # Load references and validate id alignment.
    ids_ref = None
    pred_map: Dict[int, np.ndarray] = {}
    required_subs = sorted(set(base_subs + injector_subs + reference_subs))
    for sub in required_subs:
        ids, preds = load_submission_matrix(sub, submission_dir, cache, ids_ref)
        if ids_ref is None:
            ids_ref = ids
        pred_map[sub] = preds

    assert ids_ref is not None
    n_test = len(ids_ref)

    # Build anchor labels.
    anchor_lb = build_anchor_lb_map(args.anchors_csv, submission_dir, extra_anchors)
    anchor_subs = sorted(anchor_lb.keys())
    if len(anchor_subs) < 10:
        raise RuntimeError(f"Too few anchor submissions: {len(anchor_subs)}")

    # Load anchor predictions and expand map.
    for sub in anchor_subs:
        ids, preds = load_submission_matrix(sub, submission_dir, cache, ids_ref)
        pred_map[sub] = preds

    # Build design matrices for anchors.
    feature_names = build_feature_names(reference_subs)
    x_anchor = []
    y_anchor = []
    for sub in anchor_subs:
        feats = build_feature_vector(pred_map[sub], pred_map, reference_subs)
        x_anchor.append(feats)
        y_anchor.append(anchor_lb[sub])
    x_anchor_arr = np.vstack(x_anchor)
    y_anchor_arr = np.asarray(y_anchor, dtype=float)

    # Surrogate calibration.
    alpha_grid = np.logspace(-4, 1, 12)
    best_alpha, loocv_mae, loocv_rmse = loocv_alpha_search(x_anchor_arr, y_anchor_arr, alpha_grid)

    # Candidate generation.
    specs = generate_candidate_specs(base_subs, injector_subs, args.scale, args.seed)
    candidate_rows: List[Dict[str, object]] = []
    candidate_preds: List[np.ndarray] = []
    candidate_feats: List[np.ndarray] = []

    for spec in specs:
        preds = blend_predictions(spec.weights, pred_map)
        feats = build_feature_vector(preds, pred_map, reference_subs)
        corr2063 = [corr_safe(preds[:, i], pred_map[2063][:, i]) for i in range(3)]
        mse2063 = [float(np.mean((preds[:, i] - pred_map[2063][:, i]) ** 2)) for i in range(3)]
        candidate_index = len(candidate_rows)
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": spec.candidate_id,
                "template": spec.template,
                "base_sub": spec.base_sub,
                "weights_json": json.dumps(spec.weights, sort_keys=True),
                "angle_std": float(np.std(preds[:, 0])),
                "depth_mean": float(np.mean(preds[:, 1])),
                "left_right_mean": float(np.mean(preds[:, 2])),
                "corr2063_angle": corr2063[0],
                "corr2063_depth": corr2063[1],
                "corr2063_left_right": corr2063[2],
                "mse2063_angle": mse2063[0],
                "mse2063_depth": mse2063[1],
                "mse2063_left_right": mse2063[2],
            }
        )
        candidate_preds.append(preds)
        candidate_feats.append(feats)

    x_cand_arr = np.vstack(candidate_feats)
    n_candidates = len(candidate_rows)

    # Bootstrap uncertainty.
    n_bootstrap = 80 * args.scale
    rng = np.random.default_rng(args.seed + 1)
    pred_boot = np.zeros((n_bootstrap, n_candidates), dtype=float)
    anchor_indices = np.arange(len(anchor_subs))

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, len(anchor_subs), size=len(anchor_subs))
        oob_mask = np.ones(len(anchor_subs), dtype=bool)
        oob_mask[np.unique(sample_idx)] = False

        alpha = float(best_alpha * math.exp(rng.normal(0.0, 0.35)))
        alpha = float(np.clip(alpha, 1e-5, 1e2))

        x_train = x_anchor_arr[sample_idx]
        y_train = y_anchor_arr[sample_idx]
        pred = ridge_predict(x_train, y_train, x_cand_arr, alpha)

        # Simple OOB bias correction.
        if int(np.sum(oob_mask)) >= 5:
            oob_idx = anchor_indices[oob_mask]
            oob_pred = ridge_predict(x_train, y_train, x_anchor_arr[oob_idx], alpha)
            bias = float(np.mean(y_anchor_arr[oob_idx] - oob_pred))
            pred = pred + bias

        pred_boot[b] = pred

    # Aggregate candidate statistics.
    rows = []
    for idx, base_row in enumerate(candidate_rows):
        sample = pred_boot[:, idx]
        q10, q50, q90 = np.quantile(sample, [0.10, 0.50, 0.90])
        mean = float(np.mean(sample))
        std = float(np.std(sample, ddof=0))
        p_better = float(np.mean(sample < args.best_lb))
        # Risk-aware gate score.
        gate_score = float(q50 + 0.5 * (q90 - q50))

        row = dict(base_row)
        row.update(
            {
                "pred_mean": mean,
                "pred_std": std,
                "pred_q10": float(q10),
                "pred_q50": float(q50),
                "pred_q90": float(q90),
                "p_better_best": p_better,
                "gate_score": gate_score,
            }
        )
        rows.append(row)

    candidates_df = pd.DataFrame(rows)
    candidates_df.sort_values(
        by=["gate_score", "pred_mean", "pred_std"],
        ascending=[True, True, True],
        inplace=True,
    )
    candidates_df.reset_index(drop=True, inplace=True)

    # Greedy diversity filter for final exported submissions.
    selected_idx: List[int] = []
    selected_pred_arrays: List[np.ndarray] = []
    max_corr_between_selected = 0.99990

    for row_idx in candidates_df.index:
        row = candidates_df.loc[row_idx]
        if row["template"] == "base_only":
            continue
        cand_idx = int(row["candidate_index"])
        arr = candidate_preds[cand_idx]

        keep = True
        for prev_arr in selected_pred_arrays:
            corr = mean_target_corr(arr, prev_arr)
            if corr > max_corr_between_selected:
                keep = False
                break
        if keep:
            selected_idx.append(cand_idx)
            selected_pred_arrays.append(arr)
        if len(selected_idx) >= args.top_k:
            break

    if len(selected_idx) < args.top_k:
        for row_idx in candidates_df.index:
            cand_idx = int(candidates_df.loc[row_idx, "candidate_index"])
            if cand_idx in selected_idx:
                continue
            if candidates_df.loc[row_idx, "template"] == "base_only":
                continue
            selected_idx.append(cand_idx)
            selected_pred_arrays.append(candidate_preds[cand_idx])
            if len(selected_idx) >= args.top_k:
                break

    selected_df = candidates_df[
        candidates_df["candidate_index"].isin(selected_idx)
    ].copy()
    selected_df.sort_values(by=["gate_score", "pred_mean", "pred_std"], inplace=True)
    selected_df.reset_index(drop=True, inplace=True)

    # Write submissions.
    exported: List[Dict[str, object]] = []
    for i, row in selected_df.iterrows():
        cand_global_idx = int(row["candidate_index"])
        pred_arr = candidate_preds[cand_global_idx]
        sub_num = get_next_submission_number(submission_dir)
        out_path = submission_dir / f"submission_{sub_num}.csv"
        sub_df = pd.DataFrame(
            {
                "id": ids_ref,
                "scaled_angle": pred_arr[:, 0],
                "scaled_depth": pred_arr[:, 1],
                "scaled_left_right": pred_arr[:, 2],
            }
        )
        sub_df.to_csv(out_path, index=False, float_format="%.15f")
        exported.append(
            {
                "submission_num": sub_num,
                "submission_file": str(out_path),
                "candidate_id": row["candidate_id"],
                "template": row["template"],
                "weights_json": row["weights_json"],
                "pred_mean": float(row["pred_mean"]),
                "pred_std": float(row["pred_std"]),
                "pred_q10": float(row["pred_q10"]),
                "pred_q50": float(row["pred_q50"]),
                "pred_q90": float(row["pred_q90"]),
                "p_better_best": float(row["p_better_best"]),
                "gate_score": float(row["gate_score"]),
                "corr2063_angle": float(row["corr2063_angle"]),
                "corr2063_depth": float(row["corr2063_depth"]),
                "corr2063_left_right": float(row["corr2063_left_right"]),
                "angle_std": float(row["angle_std"]),
                "depth_mean": float(row["depth_mean"]),
                "left_right_mean": float(row["left_right_mean"]),
            }
        )

    # Persist artifacts.
    cand_path = output_dir / f"probabilistic_gate_ensemble_candidates_{run_tag}.csv"
    sel_path = output_dir / f"probabilistic_gate_ensemble_selected_{run_tag}.csv"
    run_json_path = output_dir / f"probabilistic_gate_ensemble_run_{run_tag}.json"
    details_md_path = output_dir / f"probabilistic_gate_ensemble_submission_details_{run_tag}.md"

    candidates_df.to_csv(cand_path, index=False, float_format="%.15f")
    selected_df.to_csv(sel_path, index=False, float_format="%.15f")

    elapsed = time.time() - t0
    run_payload = {
        "run_tag": run_tag,
        "command": " ".join(sys.argv),
        "scale": args.scale,
        "seed": args.seed,
        "top_k": args.top_k,
        "best_lb": args.best_lb,
        "n_test_rows": n_test,
        "n_anchors": len(anchor_subs),
        "anchor_subs": anchor_subs,
        "manual_exact_lb_used": MANUAL_EXACT_LB,
        "extra_anchors_used": extra_anchors,
        "base_subs": base_subs,
        "injector_subs": injector_subs,
        "reference_subs": reference_subs,
        "n_candidates": n_candidates,
        "n_bootstrap": n_bootstrap,
        "surrogate_best_alpha_loocv": best_alpha,
        "surrogate_loocv_mae": loocv_mae,
        "surrogate_loocv_rmse": loocv_rmse,
        "candidate_csv": str(cand_path),
        "selected_csv": str(sel_path),
        "exported_submissions": exported,
        "elapsed_seconds": elapsed,
    }
    with open(run_json_path, "w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2)

    lines: List[str] = []
    lines.append("# Probabilistic Gate Ensemble Submission Details")
    lines.append("")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- command: `{run_payload['command']}`")
    lines.append(f"- scale: `{args.scale}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- top_k: `{args.top_k}`")
    lines.append(f"- best_lb_threshold: `{args.best_lb:.15f}`")
    lines.append(f"- anchor_count: `{len(anchor_subs)}`")
    lines.append(f"- anchor_subs: `{anchor_subs}`")
    lines.append(f"- n_candidates: `{n_candidates}`")
    lines.append(f"- n_bootstrap: `{n_bootstrap}`")
    lines.append(f"- surrogate_best_alpha_loocv: `{best_alpha:.15f}`")
    lines.append(f"- surrogate_loocv_mae: `{loocv_mae:.15f}`")
    lines.append(f"- surrogate_loocv_rmse: `{loocv_rmse:.15f}`")
    lines.append("")
    lines.append("## Exported submissions")
    lines.append("")
    for item in exported:
        lines.append(f"- file: `{item['submission_file']}`")
        lines.append(f"  - submission_num: `{item['submission_num']}`")
        lines.append(f"  - candidate_id: `{item['candidate_id']}`")
        lines.append(f"  - template: `{item['template']}`")
        lines.append(f"  - weights_json: `{item['weights_json']}`")
        lines.append(f"  - pred_mean: `{item['pred_mean']:.15f}`")
        lines.append(f"  - pred_std: `{item['pred_std']:.15f}`")
        lines.append(f"  - pred_q10: `{item['pred_q10']:.15f}`")
        lines.append(f"  - pred_q50: `{item['pred_q50']:.15f}`")
        lines.append(f"  - pred_q90: `{item['pred_q90']:.15f}`")
        lines.append(f"  - p_better_best: `{item['p_better_best']:.15f}`")
        lines.append(f"  - gate_score: `{item['gate_score']:.15f}`")
        lines.append(f"  - corr2063_angle: `{item['corr2063_angle']:.15f}`")
        lines.append(f"  - corr2063_depth: `{item['corr2063_depth']:.15f}`")
        lines.append(f"  - corr2063_left_right: `{item['corr2063_left_right']:.15f}`")
        lines.append(f"  - angle_std: `{item['angle_std']:.15f}`")
        lines.append(f"  - depth_mean: `{item['depth_mean']:.15f}`")
        lines.append(f"  - left_right_mean: `{item['left_right_mean']:.15f}`")
    lines.append("")
    lines.append("## Artifact paths")
    lines.append("")
    lines.append(f"- candidates_csv: `{cand_path}`")
    lines.append(f"- selected_csv: `{sel_path}`")
    lines.append(f"- run_json: `{run_json_path}`")
    lines.append(f"- details_md: `{details_md_path}`")
    lines.append("")
    lines.append(f"- elapsed_seconds: `{elapsed:.6f}`")
    lines.append("")

    with open(details_md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("=" * 80)
    print("PROBABILISTIC GATE ENSEMBLE COMPLETE")
    print("=" * 80)
    print(f"run_tag={run_tag}")
    print(f"scale={args.scale}")
    print(f"seed={args.seed}")
    print(f"anchors={len(anchor_subs)}")
    print(f"candidates={n_candidates}")
    print(f"bootstrap={n_bootstrap}")
    print(f"surrogate_best_alpha_loocv={best_alpha:.15f}")
    print(f"surrogate_loocv_mae={loocv_mae:.15f}")
    print(f"surrogate_loocv_rmse={loocv_rmse:.15f}")
    print(f"candidates_csv={cand_path}")
    print(f"selected_csv={sel_path}")
    print(f"run_json={run_json_path}")
    print(f"details_md={details_md_path}")
    for item in exported:
        print(
            f"submission_{item['submission_num']}.csv "
            f"pred_mean={item['pred_mean']:.15f} "
            f"pred_std={item['pred_std']:.15f} "
            f"p_better_best={item['p_better_best']:.15f}"
        )
    print(f"elapsed_seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
