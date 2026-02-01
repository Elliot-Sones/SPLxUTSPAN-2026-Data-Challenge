import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Ensure repository root is on sys.path so `import src.*` works under `uv run`.
import sys
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@dataclass(frozen=True)
class DriftRow:
    feature: str
    train_mean: float
    test_mean: float
    abs_mean_diff: float
    train_std: float
    test_std: float
    std_ratio: float
    mean_shift_z: float
    wasserstein: float
    train_n: int
    test_n: int
    drift_score: float


def parse_array_string(s: str) -> np.ndarray:
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_f4_train_cache(cache_path: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    import pickle

    with cache_path.open("rb") as f:
        data = pickle.load(f)

    X = data["X"]
    meta = data["meta"]
    feature_names = list(data["feature_names"])

    if not isinstance(X, np.ndarray) or not isinstance(meta, pd.DataFrame):
        raise TypeError("Unexpected cache structure")
    if "participant_id" not in meta.columns:
        raise ValueError("Cache meta missing participant_id")

    return X.astype(np.float64, copy=False), meta, feature_names


def compute_f4_test_features(feature_names: list[str], smooth: bool) -> tuple[np.ndarray, pd.DataFrame]:
    from src.hybrid_features import extract_hybrid_features, init_keypoint_mapping  # type: ignore

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in test_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)

    X_test = np.zeros((len(test_df), len(feature_names)), dtype=np.float64)
    meta = test_df[meta_cols].copy()

    for idx, row in test_df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        feats = extract_hybrid_features(timeseries, participant_id=int(row["participant_id"]), smooth=smooth)

        # Ensure participant features exist if present in feature_names.
        if "participant_id" in feature_names:
            feats["participant_id"] = float(row["participant_id"])
        for pid in range(1, 6):
            name = f"participant_{pid}"
            if name in feature_names:
                feats[name] = 1.0 if int(row["participant_id"]) == pid else 0.0

        for j, name in enumerate(feature_names):
            X_test[idx, j] = float(feats.get(name, 0.0))

        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)} test shots")

    return X_test, meta


def _safe_mean_std(x: np.ndarray) -> tuple[float, float, int]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(x)), float(np.std(x)), int(x.size)


def drift_report(
    X_train: np.ndarray,
    meta_train: pd.DataFrame,
    X_test: np.ndarray,
    meta_test: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    if "participant_id" not in meta_train.columns or "participant_id" not in meta_test.columns:
        raise ValueError("meta missing participant_id")

    train_pid = meta_train["participant_id"].to_numpy()
    test_pid = meta_test["participant_id"].to_numpy()

    rows: list[DriftRow] = []

    # Ignore pure-id features for drift ranking.
    ignore = set()
    ignore.add("participant_id")
    ignore.update({f"participant_{pid}" for pid in range(1, 6)})

    eps = 1e-12

    for j, feat in enumerate(feature_names):
        if feat in ignore:
            continue

        # Aggregate per participant to reduce composition effects.
        per_pid_stats = []
        for pid in sorted(np.unique(train_pid)):
            tr_mask = train_pid == pid
            te_mask = test_pid == pid
            tr = X_train[tr_mask, j]
            te = X_test[te_mask, j]

            tr_mean, tr_std, tr_n = _safe_mean_std(tr)
            te_mean, te_std, te_n = _safe_mean_std(te)
            if tr_n == 0 or te_n == 0:
                continue

            abs_mean_diff = abs(te_mean - tr_mean)
            std_ratio = (te_std + eps) / (tr_std + eps)
            mean_shift_z = abs_mean_diff / (tr_std + eps)
            tr_clean = tr[~np.isnan(tr)]
            te_clean = te[~np.isnan(te)]
            # Use standardized Wasserstein to avoid scale artifacts (energy features would dominate otherwise).
            tr_z = (tr_clean - tr_mean) / (tr_std + eps)
            te_z = (te_clean - tr_mean) / (tr_std + eps)
            w = wasserstein_distance(tr_z, te_z)

            per_pid_stats.append(
                (pid, tr_mean, te_mean, abs_mean_diff, tr_std, te_std, std_ratio, mean_shift_z, w, tr_n, te_n)
            )

        if not per_pid_stats:
            continue

        # Weighted by train sample counts per pid.
        weights = np.array([s[9] for s in per_pid_stats], dtype=np.float64)
        weights = weights / np.sum(weights)

        train_mean = float(np.sum(weights * np.array([s[1] for s in per_pid_stats])))
        test_mean = float(np.sum(weights * np.array([s[2] for s in per_pid_stats])))
        abs_mean_diff = float(np.sum(weights * np.array([s[3] for s in per_pid_stats])))
        train_std = float(np.sum(weights * np.array([s[4] for s in per_pid_stats])))
        test_std = float(np.sum(weights * np.array([s[5] for s in per_pid_stats])))
        std_ratio = float(np.sum(weights * np.array([s[6] for s in per_pid_stats])))
        mean_shift_z = float(np.sum(weights * np.array([s[7] for s in per_pid_stats])))
        w = float(np.sum(weights * np.array([s[8] for s in per_pid_stats])))
        train_n = int(np.sum(np.array([s[9] for s in per_pid_stats])))
        test_n = int(np.sum(np.array([s[10] for s in per_pid_stats])))

        # Single scalar drift score: primarily mean shift, secondarily shape shift.
        drift_score = float(mean_shift_z + 0.1 * w)

        rows.append(
            DriftRow(
                feature=feat,
                train_mean=train_mean,
                test_mean=test_mean,
                abs_mean_diff=abs_mean_diff,
                train_std=train_std,
                test_std=test_std,
                std_ratio=std_ratio,
                mean_shift_z=mean_shift_z,
                wasserstein=w,
                train_n=train_n,
                test_n=test_n,
                drift_score=drift_score,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("drift_score", ascending=False).reset_index(drop=True)
    return df


def load_importance_summary(path: Path) -> pd.DataFrame:
    """
    Loads `output/feature_signal_perm_importance.csv` and returns per-feature aggregates:
    - delta_mse_mean_all = mean(delta_mse_mean) over participant_id and target
    """
    df = pd.read_csv(path)
    if not {"feature", "delta_mse_mean"}.issubset(set(df.columns)):
        raise ValueError("Unexpected importance file format")
    agg = df.groupby("feature", as_index=False).agg(delta_mse_mean_all=("delta_mse_mean", "mean"))
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-cache",
        type=Path,
        default=PROJECT_DIR / "the_rest" / "output" / "feature_cache" / "features_F4_smooth.pkl",
    )
    ap.add_argument("--smooth", action="store_true", help="Use smooth=True for test feature extraction (F4_smooth)")
    ap.add_argument("--importance", type=Path, default=PROJECT_DIR / "output" / "feature_signal_perm_importance.csv")
    ap.add_argument("--out", type=Path, default=PROJECT_DIR / "output" / "feature_drift_f4.csv")
    args = ap.parse_args()

    print(f"Loading train cache: {args.train_cache}")
    X_train, meta_train, feature_names = load_f4_train_cache(args.train_cache)
    print(f"Train X: {X_train.shape}, features: {len(feature_names)}")

    print("Computing F4 test features from data/test.csv...")
    X_test, meta_test = compute_f4_test_features(feature_names=feature_names, smooth=args.smooth)
    print(f"Test X: {X_test.shape}")

    print("Computing drift report...")
    drift_df = drift_report(
        X_train=X_train,
        meta_train=meta_train,
        X_test=X_test,
        meta_test=meta_test,
        feature_names=feature_names,
    )

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_csv(out, index=False)
    print(f"Wrote: {out}")
    print(f"Rows: {len(drift_df)}")

    if args.importance.exists():
        imp = load_importance_summary(args.importance)
        merged = drift_df.merge(imp, on="feature", how="left").fillna({"delta_mse_mean_all": 0.0})
        # Stability-adjusted score: prefer high importance and low drift.
        merged["stability_adjusted"] = merged["delta_mse_mean_all"] / (1.0 + merged["drift_score"])
        merged_out = out.with_name(re.sub(r"\.csv$", "_with_importance.csv", out.name))
        merged.to_csv(merged_out, index=False)
        print(f"Wrote: {merged_out}")

        topk = 15
        print(f"\nTop {topk} drift_score features (highest drift first):")
        for _, r in drift_df.head(topk).iterrows():
            print(
                f"  {r['feature']}: drift_score={r['drift_score']:.8f} mean_shift_z={r['mean_shift_z']:.8f} "
                f"wasserstein={r['wasserstein']:.8f}"
            )

        print(f"\nTop {topk} stability_adjusted features (highest first):")
        best = merged.sort_values("stability_adjusted", ascending=False).head(topk)
        for _, r in best.iterrows():
            print(
                f"  {r['feature']}: stability_adjusted={r['stability_adjusted']:.10f} "
                f"importance={r['delta_mse_mean_all']:.10f} drift_score={r['drift_score']:.10f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
