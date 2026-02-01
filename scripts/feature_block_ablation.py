import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import warnings


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

TARGETS = ["angle", "depth", "left_right"]


@dataclass(frozen=True)
class AblationRow:
    block: str
    n_features: int
    angle_mse_mean: float
    depth_mse_mean: float
    left_right_mse_mean: float
    total_scaled_mse_mean: float


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

    return X.astype(np.float64, copy=False), y.astype(np.float64, copy=False), meta, feature_names


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


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _fit_predict_lgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    params: dict,
    seed: int,
) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params, random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr)
    return model.predict(X_val)


def _evaluate_blocks(
    X: np.ndarray,
    y_scaled: np.ndarray,
    meta: pd.DataFrame,
    feature_names: list[str],
    blocks: dict[str, list[int]],
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    from sklearn.model_selection import KFold

    if "participant_id" not in meta.columns:
        raise ValueError("meta missing participant_id")

    # Conservative fixed params: comparison is about blocks, not tuning.
    lgb_params = {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "min_child_samples": 10,
    }

    rows: list[AblationRow] = []

    for block_name, remove_cols in blocks.items():
        keep = np.ones(X.shape[1], dtype=bool)
        keep[np.array(remove_cols, dtype=int)] = False
        X_keep = X[:, keep]

        per_pid_mse: dict[str, list[float]] = {t: [] for t in TARGETS}

        for pid in sorted(meta["participant_id"].unique()):
            pid_mask = meta["participant_id"].to_numpy() == pid
            X_pid = X_keep[pid_mask]
            y_pid = y_scaled[pid_mask]

            if X_pid.shape[0] < n_splits:
                continue

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_pid)):
                X_tr, X_val = X_pid[tr_idx], X_pid[val_idx]
                for target_idx, target in enumerate(TARGETS):
                    y_tr = y_pid[tr_idx, target_idx]
                    y_val = y_pid[val_idx, target_idx]
                    pred = _fit_predict_lgb(X_tr, y_tr, X_val, lgb_params, seed=seed + 1000 * fold + 17 * pid)
                    per_pid_mse[target].append(_mse(y_val, pred))

        angle = float(np.mean(per_pid_mse["angle"]))
        depth = float(np.mean(per_pid_mse["depth"]))
        lr = float(np.mean(per_pid_mse["left_right"]))
        total = float(np.mean([angle, depth, lr]))

        rows.append(
            AblationRow(
                block=block_name,
                n_features=int(X_keep.shape[1]),
                angle_mse_mean=angle,
                depth_mse_mean=depth,
                left_right_mse_mean=lr,
                total_scaled_mse_mean=total,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("total_scaled_mse_mean").reset_index(drop=True)
    return df


def _build_blocks(feature_names: list[str]) -> dict[str, list[int]]:
    blocks: dict[str, list[int]] = {}

    pid_re = re.compile(r"^participant_(\d+)$")
    z_re = re.compile(r"_z_(mean|std|min|max|range|q25|q75|energy)$")
    vel_re = re.compile(r"_vel_")
    angle_re = re.compile(r"_angle_(mean|std|min|max|range)$")

    pid_cols: list[int] = []
    z_cols: list[int] = []
    vel_cols: list[int] = []
    joint_angle_cols: list[int] = []
    other_cols: list[int] = []

    for i, name in enumerate(feature_names):
        if name == "participant_id" or pid_re.match(name):
            pid_cols.append(i)
        elif z_re.search(name):
            z_cols.append(i)
        elif angle_re.search(name):
            joint_angle_cols.append(i)
        elif vel_re.search(name):
            vel_cols.append(i)
        else:
            other_cols.append(i)

    # With the cached F4 feature set, "other" corresponds to the physics-like features
    # (everything not pid/z/vel). We keep the naming "physics" because that's the intent
    # of that block in hybrid_features.py.
    physics_cols = other_cols
    other_cols = []

    blocks["baseline_all_features"] = []
    blocks["remove_pid_features"] = pid_cols
    blocks["remove_z_stats"] = z_cols
    blocks["remove_velocity_stats"] = vel_cols
    blocks["remove_joint_angle_stats"] = joint_angle_cols
    blocks["remove_physics_features"] = physics_cols
    blocks["remove_uncategorized_other"] = other_cols

    return blocks


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
    ap.add_argument("--out", type=Path, default=PROJECT_DIR / "output" / "feature_block_ablation.csv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y_raw, meta, feature_names = _load_feature_cache(args.feature_cache)
    scalers = _load_target_scalers()
    y_scaled = _scale_targets(y_raw, scalers)

    blocks = _build_blocks(feature_names)
    df = _evaluate_blocks(
        X=X,
        y_scaled=y_scaled,
        meta=meta,
        feature_names=feature_names,
        blocks=blocks,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
