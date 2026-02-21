#!/usr/bin/env python3
"""
Stability-first LOPO ensemble for better CV-LB alignment.

Pipeline:
1. LOPO-stable feature selection per target
2. Two-model blend: Ridge(stable) + RF(all)
3. Row-level risk gate (plausibility + RF uncertainty)
4. Inner-fold linear calibration per outer fold
5. Report mean and variance across LOPO folds
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = PROJECT_DIR / "output" / "physics_scratch_train.csv"
DEFAULT_TEST = PROJECT_DIR / "output" / "physics_scratch_test.csv"
DEFAULT_OUT_JSON = PROJECT_DIR / "output" / "cv_lb_proxy_ensemble_results_20260218.json"
DEFAULT_OUT_OOF = PROJECT_DIR / "output" / "cv_lb_proxy_ensemble_oof_20260218.csv"
DEFAULT_OUT_TEST = PROJECT_DIR / "output" / "cv_lb_proxy_ensemble_test_preds_20260218.csv"

TARGETS = ["angle", "depth", "left_right"]
ID_COLS = {"id", "shot_id"}

# Physics-oriented columns used for conservative gating.
PlausibilityCandidates = [
    "release_dist_to_hoop",
    "release_vs_peak_speed",
    "v_toward_hoop",
    "v_lateral",
    "v_vertical",
    "hoop_alignment",
    "elbow_angle_deg",
    "forearm_elevation_deg",
    "upper_arm_elevation_deg",
    "wrist_speed",
    "release_z_m",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def robust_scale(train_vals: np.ndarray, vals: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    med = np.nanmedian(train_vals)
    q1 = np.nanpercentile(train_vals, 25)
    q3 = np.nanpercentile(train_vals, 75)
    sigma = (q3 - q1) / 1.349
    if not np.isfinite(sigma) or sigma < eps:
        sigma = float(np.nanstd(train_vals) + eps)
    return (vals - med) / sigma


def numeric_feature_cols(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    common = [c for c in train_df.columns if c in test_df.columns]
    cols = []
    for c in common:
        if c in ID_COLS or c in TARGETS or c == "player_id":
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(test_df[c]):
            cols.append(c)
    return cols


def rf_tree_std(rf: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    tree_preds = np.stack([est.predict(X) for est in rf.estimators_], axis=1)
    return np.std(tree_preds, axis=1)


def plausibility_score(
    X_train_all: np.ndarray,
    X_eval_all: np.ndarray,
    plaus_idx: np.ndarray,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    if plaus_idx.size == 0:
        return np.zeros(X_train_all.shape[0], dtype=np.float64), np.zeros(X_eval_all.shape[0], dtype=np.float64)

    tr = X_train_all[:, plaus_idx]
    ev = X_eval_all[:, plaus_idx]

    med = np.nanmedian(tr, axis=0)
    q1 = np.nanpercentile(tr, 25, axis=0)
    q3 = np.nanpercentile(tr, 75, axis=0)
    sigma = (q3 - q1) / 1.349
    sigma = np.where(np.isfinite(sigma) & (sigma > eps), sigma, np.nanstd(tr, axis=0) + eps)

    z_tr = np.abs((tr - med) / sigma)
    z_ev = np.abs((ev - med) / sigma)

    score_tr = np.nanmedian(z_tr, axis=1)
    score_ev = np.nanmedian(z_ev, axis=1)

    score_tr = np.nan_to_num(score_tr, nan=0.0, posinf=10.0, neginf=0.0)
    score_ev = np.nan_to_num(score_ev, nan=0.0, posinf=10.0, neginf=0.0)
    return score_tr, score_ev


def select_stable_features_outer(
    X_outer: np.ndarray,
    y_outer: np.ndarray,
    groups_outer: np.ndarray,
    feature_names: list[str],
    rf_n_estimators: int,
    seed: int,
    top_k: int = 20,
    min_freq: int = 3,
) -> list[str]:
    unique_groups = np.unique(groups_outer)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    freq: dict[str, int] = {}
    for tr_idx, _ in gkf.split(X_outer, y_outer, groups_outer):
        model = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=8,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_outer[tr_idx], y_outer[tr_idx])
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:top_k]
        for i in top_idx:
            f = feature_names[i]
            freq[f] = freq.get(f, 0) + 1

    selected = [f for f, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_freq]
    if len(selected) == 0:
        selected = [f for f, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
    return selected


def fit_predict_blend(
    X_train_all: np.ndarray,
    y_train: np.ndarray,
    X_eval_all: np.ndarray,
    stable_idx: np.ndarray,
    plaus_idx: np.ndarray,
    rf_n_estimators: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0, random_state=seed)),
    ])
    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )

    X_train_stable = X_train_all[:, stable_idx]
    X_eval_stable = X_eval_all[:, stable_idx]

    ridge.fit(X_train_stable, y_train)
    rf.fit(X_train_all, y_train)

    pred_ridge_eval = ridge.predict(X_eval_stable)
    pred_rf_eval = rf.predict(X_eval_all)

    pred_ridge_train = ridge.predict(X_train_stable)
    pred_rf_train = rf.predict(X_train_all)

    unc_train = rf_tree_std(rf, X_train_all)
    unc_eval = rf_tree_std(rf, X_eval_all)

    plaus_train, plaus_eval = plausibility_score(X_train_all, X_eval_all, plaus_idx)

    unc_train_z = robust_scale(unc_train, unc_train)
    unc_eval_z = robust_scale(unc_train, unc_eval)
    plaus_train_z = robust_scale(plaus_train, plaus_train)
    plaus_eval_z = robust_scale(plaus_train, plaus_eval)

    risk_train = 0.4 * unc_train_z + 0.6 * plaus_train_z
    risk_eval = 0.4 * unc_eval_z + 0.6 * plaus_eval_z

    risk_eval_z = robust_scale(risk_train, risk_eval)
    # Conservative gate: high risk -> more Ridge, low risk -> more RF.
    w_rf = 0.15 + 0.70 * sigmoid(-risk_eval_z)

    raw_eval = w_rf * pred_rf_eval + (1.0 - w_rf) * pred_ridge_eval

    # Train blend diagnostic (used by callers if needed).
    risk_train_z = robust_scale(risk_train, risk_train)
    w_rf_train = 0.15 + 0.70 * sigmoid(-risk_train_z)
    raw_train = w_rf_train * pred_rf_train + (1.0 - w_rf_train) * pred_ridge_train

    return raw_eval, raw_train


def fit_linear_calibrator(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(np.float64)
    y = y.astype(np.float64)
    if np.std(pred) < 1e-10:
        return 1.0, 0.0
    A = np.column_stack([pred, np.ones_like(pred)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    return a, b


@dataclass
class StrategyResult:
    name: str
    oof: np.ndarray
    per_fold_rows: list[dict[str, Any]]
    target_ranges: dict[str, float]

    def summary(self) -> dict[str, Any]:
        rows = self.per_fold_rows
        fold_avg = np.array([r["avg_scaled_mse"] for r in rows], dtype=np.float64)

        target_stats = {}
        for t in TARGETS:
            vals = np.array([r["targets"][t]["scaled_mse"] for r in rows], dtype=np.float64)
            target_stats[t] = {
                "mean_scaled_mse": float(np.mean(vals)),
                "std_scaled_mse": float(np.std(vals)),
                "min_scaled_mse": float(np.min(vals)),
                "max_scaled_mse": float(np.max(vals)),
            }

        return {
            "name": self.name,
            "avg_scaled_mse_mean": float(np.mean(fold_avg)),
            "avg_scaled_mse_std": float(np.std(fold_avg)),
            "avg_scaled_mse_min": float(np.min(fold_avg)),
            "avg_scaled_mse_max": float(np.max(fold_avg)),
            "target_stats": target_stats,
            "folds": rows,
        }


def evaluate_strategies(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    rf_n_estimators: int,
    seed: int,
) -> tuple[dict[str, StrategyResult], dict[str, list[str]]]:
    X_all = train_df[feature_names].to_numpy(dtype=np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    groups = train_df["player_id"].to_numpy(dtype=np.int64)

    target_ranges = {t: float(train_df[t].max() - train_df[t].min()) for t in TARGETS}

    plaus_features = [c for c in PlausibilityCandidates if c in feature_names]
    plaus_idx = np.array([feature_names.index(c) for c in plaus_features], dtype=np.int64)

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    outer_splits = []
    for tr_idx, va_idx in gkf.split(X_all, groups=groups):
        held_out = int(np.unique(groups[va_idx])[0])
        outer_splits.append((tr_idx, va_idx, held_out))

    oof_ridge_stable = np.zeros((len(train_df), len(TARGETS)), dtype=np.float64)
    oof_rf_all = np.zeros((len(train_df), len(TARGETS)), dtype=np.float64)
    oof_hybrid = np.zeros((len(train_df), len(TARGETS)), dtype=np.float64)

    rows_ridge_stable: list[dict[str, Any]] = []
    rows_rf_all: list[dict[str, Any]] = []
    rows_hybrid: list[dict[str, Any]] = []

    stable_features_full: dict[str, list[str]] = {}

    for target_idx, target in enumerate(TARGETS):
        y_all = train_df[target].to_numpy(dtype=np.float64)

        for fold_id, (tr_idx, va_idx, held_out) in enumerate(outer_splits):
            X_tr = X_all[tr_idx]
            y_tr = y_all[tr_idx]
            X_va = X_all[va_idx]
            y_va = y_all[va_idx]
            groups_tr = groups[tr_idx]

            stable_feats = select_stable_features_outer(
                X_outer=X_tr,
                y_outer=y_tr,
                groups_outer=groups_tr,
                feature_names=feature_names,
                rf_n_estimators=rf_n_estimators,
                seed=seed + 1000 * target_idx + fold_id,
                top_k=20,
                min_freq=3,
            )
            stable_idx = np.array([feature_names.index(c) for c in stable_feats], dtype=np.int64)

            # Baseline A: Ridge on stable features.
            ridge = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0, random_state=seed)),
            ])
            ridge.fit(X_tr[:, stable_idx], y_tr)
            pred_ridge = ridge.predict(X_va[:, stable_idx])
            oof_ridge_stable[va_idx, target_idx] = pred_ridge

            # Baseline B: RF on all features.
            rf = RandomForestRegressor(
                n_estimators=rf_n_estimators,
                max_depth=8,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            pred_rf = rf.predict(X_va)
            oof_rf_all[va_idx, target_idx] = pred_rf

            # Proposed: hybrid gate + inner calibration.
            inner_groups = groups_tr
            inner_gkf = GroupKFold(n_splits=len(np.unique(inner_groups)))
            inner_oof_raw = np.zeros(len(tr_idx), dtype=np.float64)

            for inner_fold, (itr, iva) in enumerate(inner_gkf.split(X_tr, y_tr, groups=inner_groups)):
                raw_iva, _ = fit_predict_blend(
                    X_train_all=X_tr[itr],
                    y_train=y_tr[itr],
                    X_eval_all=X_tr[iva],
                    stable_idx=stable_idx,
                    plaus_idx=plaus_idx,
                    rf_n_estimators=rf_n_estimators,
                    seed=seed + 100 * target_idx + 10 * fold_id + inner_fold,
                )
                inner_oof_raw[iva] = raw_iva

            a, b = fit_linear_calibrator(inner_oof_raw, y_tr)
            raw_va, _ = fit_predict_blend(
                X_train_all=X_tr,
                y_train=y_tr,
                X_eval_all=X_va,
                stable_idx=stable_idx,
                plaus_idx=plaus_idx,
                rf_n_estimators=rf_n_estimators,
                seed=seed + 100 * target_idx + fold_id,
            )
            pred_hybrid = a * raw_va + b
            oof_hybrid[va_idx, target_idx] = pred_hybrid

            # Fold row bookkeeping done once per strategy below.
            if target_idx == 0:
                rows_ridge_stable.append({"held_out_player": held_out, "targets": {}})
                rows_rf_all.append({"held_out_player": held_out, "targets": {}})
                rows_hybrid.append({"held_out_player": held_out, "targets": {}})

            scale = target_ranges[target] ** 2 if target_ranges[target] > 0 else 1.0
            mse_ridge = float(mean_squared_error(y_va, pred_ridge))
            mse_rf = float(mean_squared_error(y_va, pred_rf))
            mse_hybrid = float(mean_squared_error(y_va, pred_hybrid))

            rows_ridge_stable[fold_id]["targets"][target] = {
                "mse": mse_ridge,
                "scaled_mse": mse_ridge / scale,
                "n_features": int(len(stable_idx)),
                "stable_features": stable_feats,
            }
            rows_rf_all[fold_id]["targets"][target] = {
                "mse": mse_rf,
                "scaled_mse": mse_rf / scale,
                "n_features": int(len(feature_names)),
            }
            rows_hybrid[fold_id]["targets"][target] = {
                "mse": mse_hybrid,
                "scaled_mse": mse_hybrid / scale,
                "n_features_stable": int(len(stable_idx)),
                "calibration_a": float(a),
                "calibration_b": float(b),
            }

        # Full-data stable features for final test prediction.
        stable_full = select_stable_features_outer(
            X_outer=X_all,
            y_outer=y_all,
            groups_outer=groups,
            feature_names=feature_names,
            rf_n_estimators=rf_n_estimators,
            seed=seed + 5000 + target_idx,
            top_k=20,
            min_freq=4,
        )
        stable_features_full[target] = stable_full

    for rows in [rows_ridge_stable, rows_rf_all, rows_hybrid]:
        for r in rows:
            r["avg_scaled_mse"] = float(np.mean([r["targets"][t]["scaled_mse"] for t in TARGETS]))

    results = {
        "ridge_stable": StrategyResult("ridge_stable", oof_ridge_stable, rows_ridge_stable, target_ranges),
        "rf_all": StrategyResult("rf_all", oof_rf_all, rows_rf_all, target_ranges),
        "hybrid_calibrated_gate": StrategyResult("hybrid_calibrated_gate", oof_hybrid, rows_hybrid, target_ranges),
    }

    return results, stable_features_full


def fit_full_and_predict_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    stable_features_full: dict[str, list[str]],
    oof_hybrid: np.ndarray,
    rf_n_estimators: int,
    seed: int,
) -> np.ndarray:
    X_train = train_df[feature_names].to_numpy(dtype=np.float64)
    X_test = test_df[feature_names].to_numpy(dtype=np.float64)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    plaus_features = [c for c in PlausibilityCandidates if c in feature_names]
    plaus_idx = np.array([feature_names.index(c) for c in plaus_features], dtype=np.int64)

    preds_test = np.zeros((len(test_df), len(TARGETS)), dtype=np.float64)

    for target_idx, target in enumerate(TARGETS):
        y_train = train_df[target].to_numpy(dtype=np.float64)
        stable_feats = stable_features_full[target]
        stable_idx = np.array([feature_names.index(c) for c in stable_feats], dtype=np.int64)

        # Calibrator from OOF hybrid predictions.
        a, b = fit_linear_calibrator(oof_hybrid[:, target_idx], y_train)

        raw_test, _ = fit_predict_blend(
            X_train_all=X_train,
            y_train=y_train,
            X_eval_all=X_test,
            stable_idx=stable_idx,
            plaus_idx=plaus_idx,
            rf_n_estimators=rf_n_estimators,
            seed=seed + 8000 + target_idx,
        )
        preds_test[:, target_idx] = a * raw_test + b

    return preds_test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--test", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-oof", type=Path, default=DEFAULT_OUT_OOF)
    ap.add_argument("--out-test", type=Path, default=DEFAULT_OUT_TEST)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    features = numeric_feature_cols(train_df, test_df)
    rf_n_estimators = max(50, int(round(300 * args.scale)))

    results, stable_features_full = evaluate_strategies(
        train_df=train_df,
        test_df=test_df,
        feature_names=features,
        rf_n_estimators=rf_n_estimators,
        seed=args.seed,
    )

    # Build summary JSON.
    summary = {
        "inputs": {
            "train": str(args.train),
            "test": str(args.test),
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "feature_count": len(features),
            "rf_n_estimators": rf_n_estimators,
            "seed": args.seed,
            "scale": args.scale,
        },
        "stable_features_full": stable_features_full,
        "strategies": {
            name: strategy.summary() for name, strategy in results.items()
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # OOF dump.
    oof_df = train_df[["shot_id", "player_id"] + TARGETS].copy()
    for name, strategy in results.items():
        for i, t in enumerate(TARGETS):
            oof_df[f"pred_{name}_{t}"] = strategy.oof[:, i]
    oof_df.to_csv(args.out_oof, index=False)

    # Final test predictions from best strategy.
    hybrid = results["hybrid_calibrated_gate"]
    preds_test = fit_full_and_predict_test(
        train_df=train_df,
        test_df=test_df,
        feature_names=features,
        stable_features_full=stable_features_full,
        oof_hybrid=hybrid.oof,
        rf_n_estimators=rf_n_estimators,
        seed=args.seed,
    )
    test_pred_df = test_df[["shot_id", "player_id"]].copy()
    for i, t in enumerate(TARGETS):
        test_pred_df[f"pred_{t}"] = preds_test[:, i]
    test_pred_df.to_csv(args.out_test, index=False)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_oof}")
    print(f"Wrote: {args.out_test}")

    for name in ["ridge_stable", "rf_all", "hybrid_calibrated_gate"]:
        s = summary["strategies"][name]
        print(
            f"{name}: avg_scaled_mse_mean={s['avg_scaled_mse_mean']:.9f} "
            f"avg_scaled_mse_std={s['avg_scaled_mse_std']:.9f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
