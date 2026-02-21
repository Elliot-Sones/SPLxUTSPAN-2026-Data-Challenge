#!/usr/bin/env python3
"""
CV-LB proxy alignment diagnostics on paired train/test feature tables.

Implements:
1. Adversarial validation (distribution gap)
2. LOPO variance analysis (mean vs stability)
3. Feature-importance stability (Spearman rank corr + top-20 intersections)
4. Target distribution alignment (null-target probing)
5. Physics plausibility outlier scan
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = PROJECT_DIR / "output" / "physics_scratch_train.csv"
DEFAULT_TEST = PROJECT_DIR / "output" / "physics_scratch_test.csv"
DEFAULT_RIGOROUS = PROJECT_DIR / "physics_engine" / "output" / "rigorous_features_all.csv"
DEFAULT_OUT = PROJECT_DIR / "output" / "cv_lb_proxy_research_20260217.json"

TARGETS = ["angle", "depth", "left_right"]
ID_COLS = {"id", "shot_id"}


@dataclass(frozen=True)
class AdvResult:
    label: str
    auc_mean: float
    auc_std: float
    top_shift_features: list[dict[str, float | str]]


def _numeric_feature_cols(df_train: pd.DataFrame, df_test: pd.DataFrame) -> list[str]:
    common = [c for c in df_train.columns if c in df_test.columns]
    cols = []
    for c in common:
        if c in ID_COLS or c in TARGETS:
            continue
        if pd.api.types.is_numeric_dtype(df_train[c]) and pd.api.types.is_numeric_dtype(df_test[c]):
            cols.append(c)
    return cols


def _cv_auc_scores(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X[tr_idx], y[tr_idx])
        p = model.predict_proba(X[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], p))
    return np.array(aucs, dtype=np.float64)


def adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    include_player_id: bool,
) -> AdvResult:
    cols = list(feature_cols)
    label = "exclude_player_id"
    if include_player_id:
        if "player_id" in train_df.columns and "player_id" in test_df.columns and "player_id" not in cols:
            cols.append("player_id")
        label = "include_player_id"
    else:
        cols = [c for c in cols if c != "player_id"]

    X_train = train_df[cols].to_numpy(dtype=np.float64)
    X_test = test_df[cols].to_numpy(dtype=np.float64)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    aucs = _cv_auc_scores(X_all, y_all)

    full = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    full.fit(X_all, y_all)
    importances = full.feature_importances_
    order = np.argsort(importances)[::-1]
    top = [
        {"feature": cols[i], "importance": float(importances[i])}
        for i in order[:20]
    ]

    return AdvResult(
        label=label,
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        top_shift_features=top,
    )


def adversarial_validation_per_player(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    if "player_id" not in train_df.columns or "player_id" not in test_df.columns:
        return {"error": "player_id missing"}

    cols = [c for c in feature_cols if c != "player_id"]
    rows = []
    for pid in sorted(train_df["player_id"].unique()):
        tr = train_df[train_df["player_id"] == pid]
        te = test_df[test_df["player_id"] == pid]
        if len(tr) < 10 or len(te) < 10:
            continue

        X = np.vstack([
            tr[cols].to_numpy(dtype=np.float64),
            te[cols].to_numpy(dtype=np.float64),
        ])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(te))])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        aucs = _cv_auc_scores(X, y)
        rows.append(
            {
                "player_id": int(pid),
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
            }
        )

    return {
        "rows": rows,
        "auc_mean_across_players": float(np.mean([r["auc_mean"] for r in rows])) if rows else None,
    }


def _lopo_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, int]]:
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    splits = []
    for tr_idx, va_idx in gkf.split(np.zeros(len(groups)), groups=groups):
        held_out = int(np.unique(groups[va_idx])[0])
        splits.append((tr_idx, va_idx, held_out))
    return splits


def lopo_feature_importance_stability(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    cols = [c for c in feature_cols if c != "player_id"]
    X = train_df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    groups = train_df["player_id"].to_numpy(dtype=np.int64)

    splits = _lopo_splits(groups)

    out: dict[str, Any] = {"targets": {}, "safe_features_intersection_all_targets": []}
    global_intersection: set[str] | None = None

    for target in TARGETS:
        y = train_df[target].to_numpy(dtype=np.float64)
        fold_importances = []
        fold_top20: list[list[str]] = []
        per_fold_rows = []

        for tr_idx, va_idx, held_out in splits:
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X[tr_idx], y[tr_idx])
            imp = model.feature_importances_
            fold_importances.append(imp)

            order = np.argsort(imp)[::-1]
            top20 = [cols[i] for i in order[:20]]
            fold_top20.append(top20)
            per_fold_rows.append(
                {
                    "held_out_player": held_out,
                    "top20": top20,
                }
            )

        fold_importances_arr = np.array(fold_importances)

        # Pairwise Spearman correlation on importances
        pairwise = []
        n_folds = fold_importances_arr.shape[0]
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                rho, _ = spearmanr(fold_importances_arr[i], fold_importances_arr[j])
                pairwise.append(float(rho))

        top20_intersection = sorted(set.intersection(*[set(x) for x in fold_top20])) if fold_top20 else []

        # Frequency in fold top20
        freq = {}
        for lst in fold_top20:
            for f in lst:
                freq[f] = freq.get(f, 0) + 1
        freq_sorted = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

        out["targets"][target] = {
            "spearman_pairwise": pairwise,
            "spearman_mean": float(np.mean(pairwise)) if pairwise else None,
            "spearman_std": float(np.std(pairwise)) if pairwise else None,
            "spearman_min": float(np.min(pairwise)) if pairwise else None,
            "spearman_max": float(np.max(pairwise)) if pairwise else None,
            "top20_intersection_all_5_folds": top20_intersection,
            "top20_frequency": [{"feature": k, "count": int(v)} for k, v in freq_sorted],
            "per_fold_top20": per_fold_rows,
        }

        if global_intersection is None:
            global_intersection = set(top20_intersection)
        else:
            global_intersection &= set(top20_intersection)

    out["safe_features_intersection_all_targets"] = sorted(global_intersection) if global_intersection is not None else []
    return out


def _evaluate_lopo_model(
    train_df: pd.DataFrame,
    feature_map: dict[str, list[str]],
    model_name: str,
    random_state: int = 42,
) -> dict[str, Any]:
    groups = train_df["player_id"].to_numpy(dtype=np.int64)
    splits = _lopo_splits(groups)

    target_ranges = {
        t: float(train_df[t].max() - train_df[t].min())
        for t in TARGETS
    }

    fold_rows = []

    for tr_idx, va_idx, held_out in splits:
        fold_target = {}
        fold_scaled = []
        for target in TARGETS:
            cols = feature_map[target]
            X = train_df[cols].to_numpy(dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = train_df[target].to_numpy(dtype=np.float64)

            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            if model_name == "ridge":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=10.0, random_state=random_state)),
                ])
            elif model_name == "rf":
                model = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                raise ValueError(f"Unknown model_name={model_name}")

            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            mse = float(mean_squared_error(y_va, pred))
            scale = (target_ranges[target] ** 2) if target_ranges[target] > 0 else 1.0
            scaled_mse = mse / scale

            fold_target[target] = {
                "mse": mse,
                "scaled_mse": scaled_mse,
                "n_features": len(cols),
            }
            fold_scaled.append(scaled_mse)

        fold_rows.append(
            {
                "held_out_player": held_out,
                "targets": fold_target,
                "avg_scaled_mse": float(np.mean(fold_scaled)),
            }
        )

    avg_scaled = np.array([r["avg_scaled_mse"] for r in fold_rows], dtype=np.float64)
    return {
        "folds": fold_rows,
        "avg_scaled_mse_mean": float(np.mean(avg_scaled)),
        "avg_scaled_mse_std": float(np.std(avg_scaled)),
        "avg_scaled_mse_min": float(np.min(avg_scaled)),
        "avg_scaled_mse_max": float(np.max(avg_scaled)),
    }


def target_distribution_alignment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_map: dict[str, list[str]],
    random_state: int = 42,
) -> dict[str, Any]:
    out = {"ridge": {}, "rf": {}}

    for model_key in ["ridge", "rf"]:
        for target in TARGETS:
            cols = feature_map[target]
            X_train = train_df[cols].to_numpy(dtype=np.float64)
            X_test = test_df[cols].to_numpy(dtype=np.float64)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = train_df[target].to_numpy(dtype=np.float64)

            if model_key == "ridge":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=10.0, random_state=random_state)),
                ])
            else:
                model = RandomForestRegressor(
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                )

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            tr_mean = float(np.mean(y_train))
            tr_std = float(np.std(y_train, ddof=1))
            pr_mean = float(np.mean(pred))
            pr_std = float(np.std(pred, ddof=1))
            delta_mean = pr_mean - tr_mean
            z_mean_shift = abs(delta_mean) / (tr_std + 1e-12)
            std_ratio = pr_std / (tr_std + 1e-12)

            out[model_key][target] = {
                "train_mean": tr_mean,
                "train_std": tr_std,
                "pred_test_mean": pr_mean,
                "pred_test_std": pr_std,
                "delta_mean": float(delta_mean),
                "z_mean_shift": float(z_mean_shift),
                "std_ratio": float(std_ratio),
            }

    return out


def physics_outlier_scan(rigorous_path: Path, top_k: int = 15) -> dict[str, Any]:
    if not rigorous_path.exists():
        return {"error": f"missing file: {rigorous_path}"}

    df = pd.read_csv(rigorous_path)
    required = {"is_train", "physics_valid", "id", "shot_id", "player_id"}
    if not required.issubset(set(df.columns)):
        return {"error": f"missing required columns in {rigorous_path}"}

    core_features = [
        "release_dist_to_hoop_xy",
        "release_speed_fts",
        "launch_angle_deg",
        "backspin_hz",
        "forearm_length_ft",
        "upper_arm_length_ft",
        "hand_length_ft",
        "pred_entry_angle_deg",
        "pred_flight_time_s",
        "pred_left_right_inches",
        "vel_ratio_horizontal",
        "vel_ratio_vertical",
        "vel_ratio_lateral",
    ]
    core_features = [c for c in core_features if c in df.columns]

    train = df[df["is_train"] == 1].copy()
    test = df[df["is_train"] == 0].copy()

    valid_signal = df["physics_valid"].dropna().nunique() > 1
    if valid_signal:
        train_base = train[train["physics_valid"] == 1].copy()
        test_base = test[test["physics_valid"] == 1].copy()
    else:
        train_base = train.copy()
        test_base = test.copy()

    # Prefer rows without explicit extraction failures.
    if "extraction_error" in train_base.columns:
        train_base = train_base[train_base["extraction_error"].fillna(0) != 1].copy()
    if "extraction_error" in test_base.columns:
        test_base = test_base[test_base["extraction_error"].fillna(0) != 1].copy()

    med = train_base[core_features].median(numeric_only=True)
    iqr = train_base[core_features].quantile(0.75, numeric_only=True) - train_base[core_features].quantile(0.25, numeric_only=True)
    sigma = (iqr / 1.349).replace(0.0, np.nan)

    z = (test_base[core_features] - med) / sigma
    z = z.replace([np.inf, -np.inf], np.nan)
    abs_z = z.abs()

    # Robust plausibility score: median absolute robust z across core features.
    plausibility_score = abs_z.median(axis=1, skipna=True)
    extreme_count = int((plausibility_score > 3.0).sum())

    outlier_rows = test_base.loc[:, ["id", "shot_id", "player_id"]].copy()
    outlier_rows["plausibility_score"] = plausibility_score.values
    outlier_rows = outlier_rows.sort_values("plausibility_score", ascending=False).head(top_k)

    # Which features drive outliers most often.
    top_feature_counts: dict[str, int] = {}
    for idx in outlier_rows.index:
        row = abs_z.loc[idx].dropna().sort_values(ascending=False)
        for feat in row.head(3).index:
            top_feature_counts[feat] = top_feature_counts.get(feat, 0) + 1
    top_feature_counts_sorted = sorted(top_feature_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    return {
        "core_feature_count": len(core_features),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "physics_valid_flag_informative": bool(valid_signal),
        "train_physics_valid_rate": float(train["physics_valid"].mean()),
        "test_physics_valid_rate": float(test["physics_valid"].mean()),
        "test_valid_n": int(len(test_base)),
        "test_extreme_outlier_count_score_gt_3": extreme_count,
        "test_extreme_outlier_rate_score_gt_3": float(extreme_count / max(len(test_base), 1)),
        "top_outliers": outlier_rows.to_dict(orient="records"),
        "top_driver_feature_counts": [
            {"feature": f, "count": int(c)}
            for f, c in top_feature_counts_sorted
        ],
    }


def build_feature_map(
    base_cols: list[str],
    stability: dict[str, Any],
    mode: str,
) -> dict[str, list[str]]:
    cols_no_player = [c for c in base_cols if c != "player_id"]

    if mode == "all":
        return {t: cols_no_player[:] for t in TARGETS}

    if mode == "stable_intersection":
        feature_map: dict[str, list[str]] = {}
        for t in TARGETS:
            inter = stability["targets"][t]["top20_intersection_all_5_folds"]
            if len(inter) == 0:
                # Fallback for empty intersection
                inter = [x["feature"] for x in stability["targets"][t]["top20_frequency"][:20]]
            feature_map[t] = inter
        return feature_map

    raise ValueError(f"Unknown mode={mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--test", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--rigorous", type=Path, default=DEFAULT_RIGOROUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    feature_cols = _numeric_feature_cols(train_df, test_df)

    adv_excl = adversarial_validation(train_df, test_df, feature_cols, include_player_id=False)
    adv_incl = adversarial_validation(train_df, test_df, feature_cols, include_player_id=True)
    adv_per_player = adversarial_validation_per_player(train_df, test_df, feature_cols)

    stability = lopo_feature_importance_stability(train_df, feature_cols)

    fmap_all = build_feature_map(feature_cols, stability, mode="all")
    fmap_stable = build_feature_map(feature_cols, stability, mode="stable_intersection")

    lopo_ridge_all = _evaluate_lopo_model(train_df, fmap_all, model_name="ridge")
    lopo_rf_all = _evaluate_lopo_model(train_df, fmap_all, model_name="rf")
    lopo_ridge_stable = _evaluate_lopo_model(train_df, fmap_stable, model_name="ridge")

    dist_align_all = target_distribution_alignment(train_df, test_df, fmap_all)
    dist_align_stable = target_distribution_alignment(train_df, test_df, fmap_stable)

    physics_outliers = physics_outlier_scan(args.rigorous)

    result = {
        "inputs": {
            "train": str(args.train),
            "test": str(args.test),
            "rigorous": str(args.rigorous),
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "feature_count_common_numeric": len(feature_cols),
        },
        "technique_1_adversarial_validation": {
            "exclude_player_id": {
                "auc_mean": adv_excl.auc_mean,
                "auc_std": adv_excl.auc_std,
                "top_shift_features": adv_excl.top_shift_features,
            },
            "include_player_id": {
                "auc_mean": adv_incl.auc_mean,
                "auc_std": adv_incl.auc_std,
                "top_shift_features": adv_incl.top_shift_features,
            },
            "per_player": adv_per_player,
        },
        "technique_2_lopo_variance": {
            "ridge_all_features": lopo_ridge_all,
            "rf_all_features": lopo_rf_all,
            "ridge_stable_intersection_features": lopo_ridge_stable,
        },
        "technique_3_feature_importance_stability": stability,
        "technique_4_target_distribution_alignment": {
            "all_features": dist_align_all,
            "stable_intersection_features": dist_align_stable,
        },
        "technique_5_physics_outlier_filtering": physics_outliers,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {args.out}")
    print(f"Adversarial AUC (exclude player_id): {adv_excl.auc_mean:.6f} +/- {adv_excl.auc_std:.6f}")
    print(f"Adversarial AUC (include player_id): {adv_incl.auc_mean:.6f} +/- {adv_incl.auc_std:.6f}")
    print(f"LOPO Ridge all - avg scaled MSE mean/std: {lopo_ridge_all['avg_scaled_mse_mean']:.6f} +/- {lopo_ridge_all['avg_scaled_mse_std']:.6f}")
    print(f"LOPO RF all - avg scaled MSE mean/std: {lopo_rf_all['avg_scaled_mse_mean']:.6f} +/- {lopo_rf_all['avg_scaled_mse_std']:.6f}")
    print(f"LOPO Ridge stable - avg scaled MSE mean/std: {lopo_ridge_stable['avg_scaled_mse_mean']:.6f} +/- {lopo_ridge_stable['avg_scaled_mse_std']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
