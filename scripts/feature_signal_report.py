import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import warnings


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

TARGETS = ["angle", "depth", "left_right"]


@dataclass(frozen=True)
class PermResult:
    target: str
    participant_id: int
    feature: str
    delta_mse_mean: float
    delta_mse_std: float
    baseline_mse_mean: float


def _load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    import pickle

    with path.open("rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    meta = data["meta"]
    feature_names = list(data["feature_names"])

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Expected X/y numpy arrays in cache")
    if not isinstance(meta, pd.DataFrame):
        raise TypeError("Expected meta pandas DataFrame in cache")
    if X.shape[0] != y.shape[0] or X.shape[0] != len(meta):
        raise ValueError("X/y/meta row count mismatch")
    if X.shape[1] != len(feature_names):
        raise ValueError("X columns != feature_names length")

    return X, y, meta, feature_names


def _load_target_scalers() -> dict[str, object]:
    import joblib
    import warnings

    scalers = {}
    for t in TARGETS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scalers[t] = joblib.load(DATA_DIR / f"scaler_{t}.pkl")
    return scalers


def _scale_targets(y_raw: np.ndarray, scalers: dict[str, object]) -> np.ndarray:
    y_scaled = np.zeros_like(y_raw, dtype=np.float64)
    for i, t in enumerate(TARGETS):
        y_scaled[:, i] = scalers[t].transform(y_raw[:, i].reshape(-1, 1)).ravel()
    return y_scaled


def _fit_lgb_and_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    lgb_params: dict,
    rng_seed: int,
) -> object:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**lgb_params, random_state=rng_seed, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr)
    return model


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def run_permutation_importance(
    X: np.ndarray,
    y_scaled: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
    lgb_params_by_target: dict[str, dict],
    n_splits: int,
    n_repeats: int,
    max_features: int | None,
    rng_seed: int,
) -> pd.DataFrame:
    from sklearn.model_selection import KFold

    if "participant_id" not in meta.columns:
        raise ValueError("meta missing participant_id")

    feature_indices = np.arange(X.shape[1])
    if max_features is not None:
        feature_indices = feature_indices[:max_features]

    results: list[PermResult] = []

    for pid in sorted(meta["participant_id"].unique()):
        pid_mask = meta["participant_id"].to_numpy() == pid
        X_pid = X[pid_mask]
        y_pid = y_scaled[pid_mask]

        if X_pid.shape[0] < n_splits:
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)

        for target_idx, target in enumerate(TARGETS):
            deltas_by_feature: dict[int, list[float]] = {int(j): [] for j in feature_indices}
            baseline_mses: list[float] = []

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_pid)):
                X_tr, X_val = X_pid[tr_idx], X_pid[val_idx]
                y_tr, y_val = y_pid[tr_idx, target_idx], y_pid[val_idx, target_idx]

                model = _fit_lgb_and_predict(
                    X_tr=X_tr,
                    y_tr=y_tr,
                    lgb_params=lgb_params_by_target[target],
                    rng_seed=rng_seed + fold + 1000 * pid + 10000 * target_idx,
                )
                y_pred = model.predict(X_val)
                base = _mse(y_val, y_pred)
                baseline_mses.append(base)

                for j in feature_indices:
                    rep_deltas: list[float] = []
                    for rep in range(n_repeats):
                        rng = np.random.default_rng(rng_seed + rep + 100000 * (pid + 7 * target_idx) + 13 * fold)
                        X_perm = X_val.copy()
                        perm = rng.permutation(X_perm.shape[0])
                        X_perm[:, j] = X_perm[perm, j]
                        y_perm_pred = model.predict(X_perm)
                        rep_deltas.append(_mse(y_val, y_perm_pred) - base)
                    deltas_by_feature[int(j)].append(float(np.mean(rep_deltas)))

            baseline_mse_mean = float(np.mean(baseline_mses))
            for j in feature_indices:
                deltas = np.array(deltas_by_feature[int(j)], dtype=np.float64)
                results.append(
                    PermResult(
                        target=target,
                        participant_id=int(pid),
                        feature=feature_names[int(j)],
                        delta_mse_mean=float(np.mean(deltas)),
                        delta_mse_std=float(np.std(deltas)),
                        baseline_mse_mean=baseline_mse_mean,
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in results])
    return df


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-cache",
        type=Path,
        default=PROJECT_DIR / "the_rest" / "output" / "feature_cache" / "features_F4_smooth.pkl",
    )
    ap.add_argument("--out", type=Path, default=PROJECT_DIR / "output" / "feature_signal_perm_importance.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y_raw, meta, feature_names = _load_feature_cache(args.feature_cache)
    scalers = _load_target_scalers()
    y_scaled = _scale_targets(y_raw, scalers)

    # Conservative, fixed LightGBM params. Intentionally not reusing per-submission tuned params.
    # Goal: stable importance estimates, not absolute best score.
    lgb_params_by_target = {
        "angle": {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0,
            "min_child_samples": 10,
        },
        "depth": {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0,
            "min_child_samples": 10,
        },
        "left_right": {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0,
            "min_child_samples": 10,
        },
    }

    df = run_permutation_importance(
        X=X,
        y_scaled=y_scaled,
        meta=meta,
        feature_names=feature_names,
        lgb_params_by_target=lgb_params_by_target,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        max_features=args.max_features,
        rng_seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}")
    print(f"Rows: {len(df)} (participant x target x feature)")

    # Aggregate across participants
    agg = (
        df.groupby(["target", "feature"], as_index=False)
        .agg(
            delta_mse_mean=("delta_mse_mean", "mean"),
            delta_mse_std=("delta_mse_mean", "std"),
            baseline_mse_mean=("baseline_mse_mean", "mean"),
        )
        .sort_values(["target", "delta_mse_mean"], ascending=[True, False])
    )

    topk = 15
    print(f"\nTop {topk} features by permutation delta MSE (averaged across participants):")
    for target in TARGETS:
        sub = agg[agg["target"] == target].head(topk)
        print(f"\n{target}:")
        for _, r in sub.iterrows():
            print(
                f"  {r['feature']}: delta_mse_mean={r['delta_mse_mean']:.8f} "
                f"delta_mse_std={r['delta_mse_std']:.8f} baseline_mse_mean={r['baseline_mse_mean']:.8f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
