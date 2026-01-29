#!/usr/bin/env python3
"""
Evaluate "release velocity factorization" strategies using the cached training features.

This script is intentionally self-contained: it runs even if `data/train.csv` is not present,
by relying on `output/features_train.pkl` (pre-extracted features + raw targets).

Primary questions it answers:
1) How much leaderboard-style (scaled) performance is achievable using only release-velocity
   features (proxying ball release velocity as wrist release velocity)?
2) Do "factorized" components (relative velocities + upstream mechanics) close the gap to
   full-feature models?
3) How well can upstream factors reconstruct wrist release velocity (pseudo-label test)?
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EvalResult:
    name: str
    mse_scaled_mean: float
    mse_scaled_per_target: Tuple[float, float, float]


def _load_scalers(data_dir: Path) -> Dict[str, object]:
    import warnings
    import joblib

    scalers = {}
    for target in ("angle", "depth", "left_right"):
        p = data_dir / f"scaler_{target}.pkl"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scalers[target] = joblib.load(p)
    return scalers


def _scale_targets(y_raw: np.ndarray, scalers: Dict[str, object]) -> np.ndarray:
    y_scaled = np.zeros_like(y_raw, dtype=np.float32)
    for i, target in enumerate(("angle", "depth", "left_right")):
        y_scaled[:, i] = scalers[target].transform(y_raw[:, i].reshape(-1, 1)).ravel()
    return y_scaled


def _idx(feature_names: List[str], names: Iterable[str]) -> List[int]:
    name_to_i = {n: i for i, n in enumerate(feature_names)}
    out = []
    missing = []
    for n in names:
        if n in name_to_i:
            out.append(name_to_i[n])
        else:
            missing.append(n)
    if missing:
        raise KeyError(f"Missing required features: {missing}")
    return out


def _safe_get(feature_names: List[str], name: str) -> Optional[int]:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def _stack_derived_features(
    X: np.ndarray, feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Add derived "factorization" features from existing release-velocity columns.
    """
    # Required base components
    req = [
        "right_wrist_release_vel_x",
        "right_wrist_release_vel_y",
        "right_wrist_release_vel_z",
        "right_elbow_release_vel_x",
        "right_elbow_release_vel_y",
        "right_elbow_release_vel_z",
        "right_shoulder_release_vel_x",
        "right_shoulder_release_vel_y",
        "right_shoulder_release_vel_z",
        "right_hip_release_vel_x",
        "right_hip_release_vel_y",
        "right_hip_release_vel_z",
    ]
    base_idx = {n: _safe_get(feature_names, n) for n in req}
    if any(v is None for v in base_idx.values()):
        # If these aren't present, don't derive anything.
        return X, feature_names

    def col(n: str) -> np.ndarray:
        return X[:, int(base_idx[n])]

    # Relative velocities (simple chain decomposition proxies)
    forearm_vx = col("right_wrist_release_vel_x") - col("right_elbow_release_vel_x")
    forearm_vy = col("right_wrist_release_vel_y") - col("right_elbow_release_vel_y")
    forearm_vz = col("right_wrist_release_vel_z") - col("right_elbow_release_vel_z")

    upperarm_vx = col("right_elbow_release_vel_x") - col("right_shoulder_release_vel_x")
    upperarm_vy = col("right_elbow_release_vel_y") - col("right_shoulder_release_vel_y")
    upperarm_vz = col("right_elbow_release_vel_z") - col("right_shoulder_release_vel_z")

    torso_vx = col("right_shoulder_release_vel_x") - col("right_hip_release_vel_x")
    torso_vy = col("right_shoulder_release_vel_y") - col("right_hip_release_vel_y")
    torso_vz = col("right_shoulder_release_vel_z") - col("right_hip_release_vel_z")

    def norm3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.sqrt(a * a + b * b + c * c)

    forearm_speed = norm3(forearm_vx, forearm_vy, forearm_vz)
    upperarm_speed = norm3(upperarm_vx, upperarm_vy, upperarm_vz)
    torso_speed = norm3(torso_vx, torso_vy, torso_vz)
    hip_speed = norm3(col("right_hip_release_vel_x"), col("right_hip_release_vel_y"), col("right_hip_release_vel_z"))

    derived = np.column_stack(
        [
            forearm_vx,
            forearm_vy,
            forearm_vz,
            upperarm_vx,
            upperarm_vy,
            upperarm_vz,
            torso_vx,
            torso_vy,
            torso_vz,
            forearm_speed,
            upperarm_speed,
            torso_speed,
            hip_speed,
        ]
    ).astype(np.float32)

    derived_names = [
        "derived_forearm_rel_vel_x",
        "derived_forearm_rel_vel_y",
        "derived_forearm_rel_vel_z",
        "derived_upperarm_rel_vel_x",
        "derived_upperarm_rel_vel_y",
        "derived_upperarm_rel_vel_z",
        "derived_torso_rel_vel_x",
        "derived_torso_rel_vel_y",
        "derived_torso_rel_vel_z",
        "derived_forearm_rel_speed",
        "derived_upperarm_rel_speed",
        "derived_torso_rel_speed",
        "derived_hip_speed",
    ]

    X2 = np.concatenate([X, derived], axis=1)
    fn2 = feature_names + derived_names
    return X2, fn2


def _make_model(model: str, seed: int) -> Pipeline:
    if model == "ridge":
        from sklearn.linear_model import Ridge

        est = Ridge(alpha=3.0)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", est),
            ]
        )

    if model == "lgbm":
        import lightgbm as lgb

        base = lgb.LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=10,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", MultiOutputRegressor(base)),
            ]
        )

    if model == "xgb":
        import xgboost as xgb

        base = xgb.XGBRegressor(
            n_estimators=900,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", MultiOutputRegressor(base)),
            ]
        )

    raise ValueError(f"Unknown model '{model}'. Choose from ridge|lgbm|xgb.")


def _cv_eval(
    *,
    X: np.ndarray,
    y_scaled: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    seed: int,
    n_splits: int,
) -> Tuple[float, Tuple[float, float, float]]:
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(y_scaled, dtype=np.float32)
    for tr, va in gkf.split(X, y_scaled, groups=groups):
        m = _make_model(model_name, seed)
        m.fit(X[tr], y_scaled[tr])
        preds[va] = m.predict(X[va]).astype(np.float32)
    mse_all = float(mean_squared_error(y_scaled, preds))
    per = tuple(float(mean_squared_error(y_scaled[:, i], preds[:, i])) for i in range(3))
    return mse_all, per


def _print_results(title: str, results: List[EvalResult]) -> None:
    print(f"\n== {title} ==")
    print("name\tmse_scaled\tmse_angle\tmse_depth\tmse_left_right")
    for r in results:
        a, d, lr = r.mse_scaled_per_target
        print(f"{r.name}\t{r.mse_scaled_mean:.6f}\t{a:.6f}\t{d:.6f}\t{lr:.6f}")


def _pseudo_label_reconstruction(
    *,
    X: np.ndarray,
    feature_names: List[str],
    groups: np.ndarray,
    seed: int,
    n_splits: int,
) -> None:
    # Predict wrist release velocity from "upstream" factors (exclude wrist release velocities).
    upstream = [
        # upstream release velocities
        "right_elbow_release_vel_x",
        "right_elbow_release_vel_y",
        "right_elbow_release_vel_z",
        "right_shoulder_release_vel_x",
        "right_shoulder_release_vel_y",
        "right_shoulder_release_vel_z",
        "right_hip_release_vel_x",
        "right_hip_release_vel_y",
        "right_hip_release_vel_z",
        "neck_release_vel_x",
        "neck_release_vel_y",
        "neck_release_vel_z",
        # kinematic / alignment helpers
        "arm_extension_at_release",
        "wrist_snap_angle_at_release",
        "elbow_lateral_deviation",
        "trunk_lean_at_release",
        "knee_extension_rate_max",
        "knee_extension_rate_mean",
        "wrist_lateral_from_shoulder",
    ]
    upstream = [n for n in upstream if n in feature_names]
    if len(upstream) < 8:
        print("\n[pseudo-label] Not enough upstream features found; skipping.")
        return

    target = [
        "right_wrist_release_vel_x",
        "right_wrist_release_vel_y",
        "right_wrist_release_vel_z",
    ]
    if any(n not in feature_names for n in target):
        print("\n[pseudo-label] Wrist release velocity columns not present; skipping.")
        return

    Xi = X[:, _idx(feature_names, upstream)]
    yi = X[:, _idx(feature_names, target)].astype(np.float32)

    gkf = GroupKFold(n_splits=n_splits)
    preds = np.zeros_like(yi, dtype=np.float32)
    for tr, va in gkf.split(Xi, yi, groups=groups):
        model = _make_model("lgbm", seed)
        model.fit(Xi[tr], yi[tr])
        preds[va] = model.predict(Xi[va]).astype(np.float32)

    rmse = np.sqrt(np.mean((yi - preds) ** 2, axis=0))
    r2 = [float(r2_score(yi[:, i], preds[:, i])) for i in range(3)]
    print("\n== Pseudo-label: reconstruct wrist release velocity ==")
    print(f"features_used={len(upstream)} targets=3 model=lgbm")
    print(f"rmse(vx,vy,vz)={rmse.tolist()}")
    print(f"r2(vx,vy,vz)={r2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-pkl", type=Path, default=Path("output/features_train.pkl"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--model", choices=["ridge", "lgbm", "xgb"], default="lgbm")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    obj = pickle.load(open(args.features_pkl, "rb"))
    X = obj["X"].astype(np.float32)
    y_raw = obj["y"].astype(np.float32)
    feature_names = list(obj["feature_names"])
    meta = obj["meta"]
    groups = meta["participant_id"].to_numpy(dtype=np.int64)
    n_splits = int(len(np.unique(groups)))

    scalers = _load_scalers(args.data_dir)
    y_scaled = _scale_targets(y_raw, scalers)

    Xd, fnd = _stack_derived_features(X, feature_names)

    # Feature sets (indices into current feature space)
    wrist_release = [
        "right_wrist_release_vel_x",
        "right_wrist_release_vel_y",
        "right_wrist_release_vel_z",
        "right_wrist_release_vel_magnitude",
        "release_elevation_angle",
        "right_wrist_release_x",
        "right_wrist_release_y",
        "right_wrist_release_z",
        "wrist_snap_angle_at_release",
        "arm_extension_at_release",
        "elbow_lateral_deviation",
        "trunk_lean_at_release",
        "wrist_lateral_from_shoulder",
    ]
    wrist_release = [n for n in wrist_release if n in fnd]

    chain_release = [
        # velocities
        "neck_release_vel_x",
        "neck_release_vel_y",
        "neck_release_vel_z",
        "right_hip_release_vel_x",
        "right_hip_release_vel_y",
        "right_hip_release_vel_z",
        "right_shoulder_release_vel_x",
        "right_shoulder_release_vel_y",
        "right_shoulder_release_vel_z",
        "right_elbow_release_vel_x",
        "right_elbow_release_vel_y",
        "right_elbow_release_vel_z",
        "right_wrist_release_vel_x",
        "right_wrist_release_vel_y",
        "right_wrist_release_vel_z",
        # alignment
        "arm_extension_at_release",
        "wrist_snap_angle_at_release",
        "elbow_lateral_deviation",
        "trunk_lean_at_release",
        "knee_extension_rate_max",
        "knee_extension_rate_mean",
        "wrist_lateral_from_shoulder",
        # derived decomposition terms
        "derived_forearm_rel_vel_x",
        "derived_forearm_rel_vel_y",
        "derived_forearm_rel_vel_z",
        "derived_upperarm_rel_vel_x",
        "derived_upperarm_rel_vel_y",
        "derived_upperarm_rel_vel_z",
        "derived_torso_rel_vel_x",
        "derived_torso_rel_vel_y",
        "derived_torso_rel_vel_z",
        "derived_forearm_rel_speed",
        "derived_upperarm_rel_speed",
        "derived_torso_rel_speed",
        "derived_hip_speed",
    ]
    chain_release = [n for n in chain_release if n in fnd]

    feature_sets: List[Tuple[str, np.ndarray]] = [
        ("wrist_release_only", Xd[:, _idx(fnd, wrist_release)]),
        ("chain_plus_factors", Xd[:, _idx(fnd, chain_release)]),
        ("full_features", Xd),
    ]

    results: List[EvalResult] = []
    for name, Xi in feature_sets:
        mse_all, per = _cv_eval(
            X=Xi,
            y_scaled=y_scaled,
            groups=groups,
            model_name=args.model,
            seed=int(args.seed),
            n_splits=n_splits,
        )
        results.append(EvalResult(name=f"{args.model}:{name}", mse_scaled_mean=mse_all, mse_scaled_per_target=per))

    _print_results("CV (GroupKFold by participant) on scaled targets", results)
    _pseudo_label_reconstruction(X=Xd, feature_names=fnd, groups=groups, seed=int(args.seed), n_splits=n_splits)


if __name__ == "__main__":
    main()

