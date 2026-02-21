"""
Sub 2503 Per-Row Probabilistic Confidence and Gate Analysis

Goal:
- Treat submission_2503 as base predictions.
- Estimate per-row confidence using ensemble disagreement + feature-space OOD.
- "Try harder" on uncertain rows using a stronger local-weighted model.
- Produce diagnostics: most confident rows and most likely-wrong rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import per_example_pipeline as pep


TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-row probabilistic gate for Sub 2503")
    parser.add_argument("--base-sub", type=int, default=2503)
    parser.add_argument(
        "--expert-subs",
        type=str,
        default="2475,2493,2497,2502,2504,2505,2506,2507",
        help="Comma-separated expert submission ids (excluding base)",
    )
    parser.add_argument("--ood-k", type=int, default=5)
    parser.add_argument("--hard-bw", type=float, default=0.50)
    parser.add_argument("--uncertain-quantile", type=float, default=0.20)
    parser.add_argument("--max-hard-weight", type=float, default=0.35)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument("--run-tag", type=str, default="")
    return parser.parse_args()


def parse_subs(text: str) -> List[int]:
    vals: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("expert-subs parsed to empty list")
    return vals


def load_submission(sub_num: int) -> pd.DataFrame:
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing submission file: {path}")
    df = pd.read_csv(path)
    need = {"id", *TARGET_COLS}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"submission_{sub_num}.csv missing columns: {sorted(missing)}")
    return df[["id", *TARGET_COLS]].copy()


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def percentile_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    return ranks / (len(x) - 1)


def empirical_percentile(train_vals_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(train_vals_sorted, x, side="right")
    return pos / max(len(train_vals_sorted), 1)


def compute_ood_scores(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    pids_train: np.ndarray,
    pids_test: np.ndarray,
    k: int,
) -> Dict[str, np.ndarray]:
    n_test = len(pids_test)
    out: Dict[str, np.ndarray] = {}

    for target in TARGETS:
        X_train, _ = pep.extract_all_features(train_data, target)
        X_test, _ = pep.extract_all_features(test_data, target)
        ood = np.zeros(n_test, dtype=float)

        for pid in sorted(np.unique(pids_train)):
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            if not np.any(te_mask):
                continue
            X_tr = X_train[tr_mask]
            X_te = X_test[te_mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            k_eff = max(1, min(k, len(X_tr_s)))
            nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
            nn.fit(X_tr_s)

            d_te, _ = nn.kneighbors(X_te_s)
            d_te_mean = d_te.mean(axis=1)

            # Reference distribution: train-to-train local distances.
            k_ref = max(2, min(k_eff + 1, len(X_tr_s)))
            nn_ref = NearestNeighbors(n_neighbors=k_ref, metric="euclidean")
            nn_ref.fit(X_tr_s)
            d_tr, _ = nn_ref.kneighbors(X_tr_s)
            if k_ref > 1:
                d_tr_ref = d_tr[:, 1:].mean(axis=1)
            else:
                d_tr_ref = d_tr[:, 0]
            d_tr_ref = np.sort(d_tr_ref)
            pct = empirical_percentile(d_tr_ref, d_te_mean)
            ood[te_mask] = pct

        out[target] = ood
    return out


def compute_hard_model_predictions(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    pids_train: np.ndarray,
    pids_test: np.ndarray,
    hard_bw: float,
) -> Dict[str, np.ndarray]:
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    out = {"test": {}, "oof_mse": {}}

    for target in TARGETS:
        X_train_hc, _ = pep.extract_all_features(train_data, target)
        X_test_hc, _ = pep.extract_all_features(test_data, target)

        y_raw = train_data["y"][:, target_idx[target]]
        X_train_aug, X_test_aug = pep.augment_with_pls(
            X_train_hc,
            y_raw,
            pids_train,
            X_test_hc,
            pids_test,
            train_data["X_raw"],
            test_data["X_raw"],
        )

        scaler = joblib.load(pep.DATA_DIR / f"scaler_{target}.pkl")
        y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()

        oof, test_pred = pep.locally_weighted_prediction(
            X_train_aug,
            y_scaled,
            X_test_aug,
            pids_train,
            pids_test,
            bandwidth_quantile=hard_bw,
        )
        mse = float(np.mean((oof - y_scaled) ** 2))
        out["test"][target] = test_pred
        out["oof_mse"][target] = mse

    return out


def build_reason_flags(row: pd.Series, thresholds: Dict[str, float]) -> str:
    reasons: List[str] = []
    if row["disagreement_std_mean"] >= thresholds["std_p90"]:
        reasons.append("high_model_disagreement")
    if row["ood_mean"] >= thresholds["ood_p90"]:
        reasons.append("ood_motion_pattern")
    if row["extremeness_mean"] >= thresholds["ext_p90"]:
        reasons.append("extreme_prediction_profile")
    if row["hard_delta_mean"] >= thresholds["hard_p90"]:
        reasons.append("hard_model_disagrees_with_base")
    if not reasons:
        reasons.append("moderate_combined_risk")
    return ";".join(reasons)


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag.strip() or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    expert_subs = parse_subs(args.expert_subs)
    if args.base_sub in expert_subs:
        expert_subs = [s for s in expert_subs if s != args.base_sub]

    print("=" * 80)
    print("SUB 2503 PROBABILISTIC ROW GATE")
    print("=" * 80)
    print(f"base_sub={args.base_sub}")
    print(f"expert_subs={expert_subs}")
    print(f"ood_k={args.ood_k} hard_bw={args.hard_bw}")
    print(
        f"uncertain_quantile={args.uncertain_quantile} "
        f"max_hard_weight={args.max_hard_weight}"
    )

    base_df = load_submission(args.base_sub)
    preds: Dict[int, pd.DataFrame] = {args.base_sub: base_df}
    for sub in expert_subs:
        df = load_submission(sub)
        if not np.array_equal(df["id"].values, base_df["id"].values):
            raise ValueError(f"id mismatch between base {args.base_sub} and expert {sub}")
        preds[sub] = df

    all_subs = [args.base_sub, *expert_subs]
    n_rows = len(base_df)
    report = pd.DataFrame({"id": base_df["id"].values})

    # 1) Submission disagreement features
    for t_name, col in zip(TARGETS, TARGET_COLS):
        stack = np.stack([preds[s][col].values for s in all_subs], axis=1)
        median = np.median(stack, axis=1)
        iqr = np.quantile(stack, 0.75, axis=1) - np.quantile(stack, 0.25, axis=1)
        std = np.std(stack, axis=1)
        mad_base = np.abs(base_df[col].values - median)

        report[f"{t_name}_std"] = std
        report[f"{t_name}_iqr"] = iqr
        report[f"{t_name}_base_dev_from_median"] = mad_base

    report["disagreement_std_mean"] = report[[f"{t}_std" for t in TARGETS]].mean(axis=1)
    report["disagreement_iqr_mean"] = report[[f"{t}_iqr" for t in TARGETS]].mean(axis=1)
    report["base_dev_mean"] = report[
        [f"{t}_base_dev_from_median" for t in TARGETS]
    ].mean(axis=1)

    # 2) Feature-space OOD on per-example handcrafted features
    train_data, test_data = pep.load_data()
    pids_train = train_data["pids"]
    pids_test = test_data["pids"]
    report["participant_id"] = pids_test

    ood = compute_ood_scores(
        train_data=train_data,
        test_data=test_data,
        pids_train=pids_train,
        pids_test=pids_test,
        k=args.ood_k,
    )
    for t in TARGETS:
        report[f"{t}_ood_pct"] = ood[t]
    report["ood_mean"] = report[[f"{t}_ood_pct" for t in TARGETS]].mean(axis=1)

    # 3) Prediction extremeness relative to train target distribution
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for t_name, col in zip(TARGETS, TARGET_COLS):
        scaler = joblib.load(pep.DATA_DIR / f"scaler_{t_name}.pkl")
        y_scaled_train = scaler.transform(
            train_data["y"][:, target_idx[t_name]].reshape(-1, 1)
        ).ravel()
        y_sorted = np.sort(y_scaled_train)
        pct = empirical_percentile(y_sorted, base_df[col].values)
        report[f"{t_name}_pred_pct"] = pct
        report[f"{t_name}_extremeness"] = np.abs(pct - 0.5) * 2.0
    report["extremeness_mean"] = report[[f"{t}_extremeness" for t in TARGETS]].mean(axis=1)

    # 4) "Try harder" model predictions for uncertain rows
    hard = compute_hard_model_predictions(
        train_data=train_data,
        test_data=test_data,
        pids_train=pids_train,
        pids_test=pids_test,
        hard_bw=args.hard_bw,
    )
    for t_name, col in zip(TARGETS, TARGET_COLS):
        report[f"hard_{col}"] = hard["test"][t_name]
        report[f"hard_abs_delta_{t_name}"] = np.abs(
            report[f"hard_{col}"].values - base_df[col].values
        )
    report["hard_delta_mean"] = report[[f"hard_abs_delta_{t}" for t in TARGETS]].mean(axis=1)

    # 5) Risk and confidence score
    risk = (
        0.35 * zscore(report["disagreement_std_mean"].values)
        + 0.20 * zscore(report["disagreement_iqr_mean"].values)
        + 0.15 * zscore(report["base_dev_mean"].values)
        + 0.15 * zscore(report["ood_mean"].values)
        + 0.05 * zscore(report["extremeness_mean"].values)
        + 0.10 * zscore(report["hard_delta_mean"].values)
    )
    report["risk_raw"] = risk
    report["risk_pct"] = percentile_rank(risk)
    report["confidence"] = 1.0 - report["risk_pct"]

    # 6) Gate: keep base where confident, blend in hard model where uncertain
    conf_threshold = float(np.quantile(report["confidence"], args.uncertain_quantile))
    report["is_uncertain"] = report["confidence"] <= conf_threshold
    hard_w = np.clip((conf_threshold - report["confidence"]) / max(conf_threshold, 1e-12), 0.0, 1.0)
    hard_w = hard_w * args.max_hard_weight
    hard_w = np.where(report["is_uncertain"], hard_w, 0.0)
    report["hard_weight"] = hard_w

    gated = base_df.copy()
    for t_name, col in zip(TARGETS, TARGET_COLS):
        gated[col] = (
            (1.0 - report["hard_weight"].values) * base_df[col].values
            + report["hard_weight"].values * report[f"hard_{col}"].values
        )
        report[f"gated_abs_delta_{t_name}"] = np.abs(gated[col].values - base_df[col].values)
    report["gated_delta_mean"] = report[[f"gated_abs_delta_{t}" for t in TARGETS]].mean(axis=1)

    # 7) Reasons for risky rows
    thresholds = {
        "std_p90": float(np.quantile(report["disagreement_std_mean"], 0.90)),
        "ood_p90": float(np.quantile(report["ood_mean"], 0.90)),
        "ext_p90": float(np.quantile(report["extremeness_mean"], 0.90)),
        "hard_p90": float(np.quantile(report["hard_delta_mean"], 0.90)),
    }
    report["risk_reasons"] = report.apply(lambda r: build_reason_flags(r, thresholds), axis=1)

    # 8) Outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"sub2503_row_confidence_{run_tag}.csv"
    confident_path = OUTPUT_DIR / f"sub2503_most_confident_{run_tag}.csv"
    risky_path = OUTPUT_DIR / f"sub2503_most_risky_{run_tag}.csv"
    gated_path = OUTPUT_DIR / f"sub2503_gated_candidate_{run_tag}.csv"
    run_path = OUTPUT_DIR / f"sub2503_probabilistic_gate_run_{run_tag}.json"

    cols_view = [
        "id",
        "participant_id",
        "confidence",
        "risk_pct",
        "disagreement_std_mean",
        "ood_mean",
        "extremeness_mean",
        "hard_delta_mean",
        "hard_weight",
        "gated_delta_mean",
        "risk_reasons",
    ] + TARGET_COLS
    report = report.merge(base_df, on="id", how="left")
    report.sort_values("confidence", ascending=False, inplace=True)
    report.to_csv(all_path, index=False)

    report.head(args.top_n)[cols_view].to_csv(confident_path, index=False)
    report.sort_values("confidence", ascending=True).head(args.top_n)[cols_view].to_csv(
        risky_path, index=False
    )
    gated.to_csv(gated_path, index=False)

    summary = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_sub": args.base_sub,
        "expert_subs": expert_subs,
        "n_rows": int(n_rows),
        "ood_k": int(args.ood_k),
        "hard_bw": float(args.hard_bw),
        "uncertain_quantile": float(args.uncertain_quantile),
        "max_hard_weight": float(args.max_hard_weight),
        "confidence_threshold": conf_threshold,
        "n_uncertain_rows": int(report["is_uncertain"].sum()),
        "mean_confidence": float(report["confidence"].mean()),
        "min_confidence": float(report["confidence"].min()),
        "max_confidence": float(report["confidence"].max()),
        "hard_model_oof_mse": hard["oof_mse"],
        "mean_hard_weight_uncertain": float(
            report.loc[report["is_uncertain"], "hard_weight"].mean()
            if report["is_uncertain"].any()
            else 0.0
        ),
        "mean_gated_delta_per_target": {
            t: float(report[f"gated_abs_delta_{t}"].mean()) for t in TARGETS
        },
        "output_files": {
            "all_rows": str(all_path),
            "most_confident": str(confident_path),
            "most_risky": str(risky_path),
            "gated_candidate": str(gated_path),
        },
    }
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"n_rows={n_rows}")
    print(f"confidence_threshold={conf_threshold:.12f}")
    print(f"n_uncertain_rows={summary['n_uncertain_rows']}")
    print(f"mean_confidence={summary['mean_confidence']:.12f}")
    print("hard_model_oof_mse:")
    for t in TARGETS:
        print(f"  {t}={hard['oof_mse'][t]:.12f}")
    print("mean_gated_delta_per_target:")
    for t in TARGETS:
        print(f"  {t}={summary['mean_gated_delta_per_target'][t]:.12f}")
    print(f"saved_all_rows={all_path}")
    print(f"saved_confident={confident_path}")
    print(f"saved_risky={risky_path}")
    print(f"saved_gated={gated_path}")
    print(f"saved_run_json={run_path}")


if __name__ == "__main__":
    main()

