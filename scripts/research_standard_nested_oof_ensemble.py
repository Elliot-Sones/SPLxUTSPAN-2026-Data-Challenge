"""
Research-standard nested validation and probabilistic OOF ensemble pipeline.

Implements:
1. Repeated participant-stratified outer validation.
2. Nested participant-stratified inner tuning per model/target.
3. Probabilistic OOF ensembling with uncertainty penalty.
4. Information-gain-oriented submission batch selection.

This script avoids LB-profile heuristics and uses only training/OOF evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import load_scalers, scale_targets
from src.hybrid_features import (
    extract_hybrid_features,
    init_keypoint_mapping as init_hybrid_mapping,
)
from src.advanced_features import (
    extract_advanced_features,
    init_keypoint_mapping as init_advanced_mapping,
)


TARGETS = ["angle", "depth", "left_right"]
TARGETS_SCALED = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAME_MAP = {"angle": 153, "depth": 150, "left_right": 170}
HOOP_POS = np.array([5.25, -25.0, 10.0], dtype=np.float32)
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_grid: Tuple[Dict[str, object], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-standard nested OOF probabilistic ensemble"
    )
    parser.add_argument("--scale", type=int, default=1, help="Scale factor")
    parser.add_argument("--seed", type=int, default=20260214, help="Random seed")
    parser.add_argument(
        "--best-lb",
        type=float,
        default=0.006596,
        help="Current best LB for reference",
    )
    parser.add_argument(
        "--top-k-submissions",
        type=int,
        default=5,
        help="Number of submissions to create",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional run tag, default UTC timestamp",
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
        "--research-dir",
        type=Path,
        default=Path("Research"),
        help="Research notes directory",
    )
    parser.add_argument(
        "--feature-bank",
        type=str,
        choices=[
            "hybrid_advanced",
            "frame_triplet_compact",
            "hybrid_advanced_plus_frame_triplet",
        ],
        default="hybrid_advanced",
        help="Feature bank to use",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names, or 'all'",
    )
    return parser.parse_args()


def get_model_specs(scale: int) -> List[ModelSpec]:
    if scale <= 1:
        # Sub-minute pilot validation mode.
        return [
            ModelSpec(name="ridge", param_grid=tuple({"alpha": a} for a in [0.3, 3.0])),
            ModelSpec(
                name="elasticnet",
                param_grid=tuple(
                    {"alpha": alpha, "l1_ratio": l1_ratio}
                    for alpha in [0.003]
                    for l1_ratio in [0.2, 0.8]
                ),
            ),
            ModelSpec(
                name="knn",
                param_grid=tuple(
                    {"n_neighbors": n_neighbors, "weights": weights}
                    for n_neighbors in [9]
                    for weights in ["uniform", "distance"]
                ),
            ),
            ModelSpec(
                name="random_forest",
                param_grid=tuple(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }
                    for n_estimators in [120]
                    for max_depth in [8]
                    for min_samples_leaf in [1, 3]
                ),
            ),
            ModelSpec(
                name="extra_trees",
                param_grid=tuple(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }
                    for n_estimators in [160]
                    for max_depth in [10]
                    for min_samples_leaf in [1, 3]
                ),
            ),
        ]

    # Full search mode.
    return [
        ModelSpec(
            name="ridge",
            param_grid=tuple({"alpha": alpha} for alpha in [0.1, 0.3, 1.0, 3.0, 10.0]),
        ),
        ModelSpec(
            name="elasticnet",
            param_grid=tuple(
                {"alpha": alpha, "l1_ratio": l1_ratio}
                for alpha in [0.001, 0.003, 0.01]
                for l1_ratio in [0.2, 0.5, 0.8]
            ),
        ),
        ModelSpec(
            name="knn",
            param_grid=tuple(
                {"n_neighbors": n_neighbors, "weights": weights}
                for n_neighbors in [5, 9, 13]
                for weights in ["uniform", "distance"]
            ),
        ),
        ModelSpec(
            name="random_forest",
            param_grid=tuple(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                }
                for n_estimators in [160, 260]
                for max_depth in [8, None]
                for min_samples_leaf in [1, 3]
            ),
        ),
        ModelSpec(
            name="extra_trees",
            param_grid=tuple(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                }
                for n_estimators in [220, 340]
                for max_depth in [10, None]
                for min_samples_leaf in [1, 2]
            ),
        ),
    ]


def filter_model_specs(model_specs: List[ModelSpec], model_arg: str) -> List[ModelSpec]:
    if model_arg.strip().lower() == "all":
        return model_specs
    requested: Set[str] = {x.strip() for x in model_arg.split(",") if x.strip()}
    if not requested:
        raise ValueError("--models produced an empty model list")
    filtered = [spec for spec in model_specs if spec.name in requested]
    missing = sorted(requested.difference({spec.name for spec in filtered}))
    if missing:
        raise ValueError(f"Unknown model names in --models: {missing}")
    if not filtered:
        raise ValueError("No models selected after applying --models")
    return filtered


def participant_stratified_holdout(
    groups: np.ndarray,
    holdout_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx_parts = []
    val_idx_parts = []
    for pid in sorted(np.unique(groups)):
        idx = np.where(groups == pid)[0]
        perm = rng.permutation(idx)
        n_val = int(round(len(idx) * holdout_frac))
        n_val = max(1, min(len(idx) - 1, n_val))
        val_idx_parts.append(perm[:n_val])
        train_idx_parts.append(perm[n_val:])
    train_idx = np.sort(np.concatenate(train_idx_parts))
    val_idx = np.sort(np.concatenate(val_idx_parts))
    return train_idx, val_idx


def participant_stratified_kfold(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pid_chunks: Dict[int, List[np.ndarray]] = {}
    for pid in sorted(np.unique(groups)):
        idx = np.where(groups == pid)[0]
        shuffled = rng.permutation(idx)
        pid_chunks[int(pid)] = [np.asarray(x, dtype=int) for x in np.array_split(shuffled, n_splits)]

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        val_parts = []
        train_parts = []
        for pid in sorted(pid_chunks):
            chunks = pid_chunks[pid]
            val_parts.append(chunks[fold])
            train_parts.extend(chunks[:fold] + chunks[fold + 1 :])
        val_idx = np.sort(np.concatenate(val_parts))
        train_idx = np.sort(np.concatenate(train_parts))
        splits.append((train_idx, val_idx))
    return splits


def parse_array_string(value: str) -> np.ndarray:
    if pd.isna(value):
        return np.full(240, np.nan, dtype=np.float32)
    text = str(value).replace("nan", "null")
    return np.array(json.loads(text), dtype=np.float32)


def safe_savgol_series(
    values: np.ndarray,
    window: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x, dtype=np.float64)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, deriv=deriv, delta=delta)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def compute_hoop_transform(ts_3d: np.ndarray, kp_index: Dict[str, int]) -> np.ndarray:
    mid_hip_idx = kp_index.get("mid_hip", 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0.0

    forward = HOOP_POS[:2] - player_pos[:2]
    fn = float(np.linalg.norm(forward))
    if fn > 1e-6:
        forward = forward / fn
    else:
        forward = np.array([0.0, -1.0], dtype=np.float32)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float32)

    rot = np.eye(3, dtype=np.float32)
    rot[0, 0] = forward[0]
    rot[0, 1] = forward[1]
    rot[1, 0] = lateral[0]
    rot[1, 1] = lateral[1]

    centered = ts_3d - player_pos.reshape(1, 1, 3)
    return np.einsum("ij,fkj->fki", rot, centered)


def detect_release_frame(ts_3d: np.ndarray, kp_index: Dict[str, int]) -> int:
    rw_idx = kp_index.get("right_wrist")
    if rw_idx is None:
        return 120
    wrist_traj = ts_3d[:, rw_idx, :].copy()
    for axis in range(3):
        vals = wrist_traj[:, axis]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 120
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, axis] = vals

    wrist_z_smooth = safe_savgol_series(wrist_traj[:, 2], 11, 3)
    wrist_peak = 80 + int(np.argmax(wrist_z_smooth[80:200]))

    ft_keys = [
        "right_second_finger_distal",
        "right_third_finger_distal",
        "right_fourth_finger_distal",
    ]
    ft_trajs = [ts_3d[:, kp_index[k], :] for k in ft_keys if k in kp_index]
    if ft_trajs:
        ft_center = np.nanmean(ft_trajs, axis=0)
        for axis in range(3):
            ft_center[:, axis] = safe_savgol_series(ft_center[:, axis], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()
    for axis in range(3):
        ball[:, axis] = safe_savgol_series(ball[:, axis], 11, 3)

    vel = np.zeros_like(ball * FEET_TO_METERS)
    for axis in range(3):
        vel[:, axis] = safe_savgol_series(
            ball[:, axis] * FEET_TO_METERS,
            9,
            3,
            deriv=1,
            delta=DT,
        )
    speed = np.linalg.norm(vel, axis=1)
    start = max(80, wrist_peak - 40)
    end = min(wrist_peak + 5, 200)
    return int(np.clip(start + int(np.argmax(speed[start:end])), 80, 200))


def extract_compact_features_at_frame(
    ts_3d: np.ndarray,
    ts_hr: np.ndarray,
    kp_index: Dict[str, int],
    release_frame: int,
    frame: int,
) -> np.ndarray:
    f = int(np.clip(frame, 0, 239))
    feats: List[float] = []
    key_joints = [
        "right_wrist",
        "right_elbow",
        "right_shoulder",
        "left_wrist",
        "left_shoulder",
        "right_hip",
        "left_hip",
        "mid_hip",
        "right_knee",
        "left_knee",
        "neck",
        "nose",
    ]

    # Hoop-relative positions + velocities at target frame.
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            vel = np.gradient(series, DT)
            feats.append(float(ts_hr[f, idx, coord]))
            feats.append(float(vel[f]))

    # Hoop-relative summary stats.
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 9)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            feats.append(float(np.nanmean(series)))
            feats.append(float(np.nanstd(series)))
            feats.append(float(np.nanmax(series) - np.nanmin(series)))

    # Arm mechanics.
    rw = kp_index.get("right_wrist")
    re = kp_index.get("right_elbow")
    rs = kp_index.get("right_shoulder")
    if all(v is not None for v in [rw, re, rs]):
        feats.append(float(ts_hr[f, rw, 0] - ts_hr[f, rs, 0]))
        feats.append(float(ts_hr[f, rw, 1] - ts_hr[f, rs, 1]))
        feats.append(float(ts_hr[f, rw, 2] - ts_hr[f, rs, 2]))
        ua = ts_3d[f, re] - ts_3d[f, rs]
        fa = ts_3d[f, rw] - ts_3d[f, re]
        feats.append(_angle_between(-ua, fa))
        for coord in range(3):
            vel = np.gradient(ts_hr[:, rw, coord], DT)
            feats.append(float(vel[f]))
    else:
        feats.extend([0.0] * 7)

    # Body alignment + guide hand.
    rh = kp_index.get("right_hip")
    lh = kp_index.get("left_hip")
    ls = kp_index.get("left_shoulder")
    if rh is not None and lh is not None:
        feats.append(float(ts_hr[f, rh, 1] - ts_hr[f, lh, 1]))
        feats.append(float(ts_hr[f, rh, 0] - ts_hr[f, lh, 0]))
    else:
        feats.extend([0.0, 0.0])
    if rs is not None and ls is not None:
        feats.append(float(ts_hr[f, rs, 1] - ts_hr[f, ls, 1]))
    else:
        feats.append(0.0)
    lw = kp_index.get("left_wrist")
    if lw is not None and rw is not None:
        feats.append(float(ts_hr[f, lw, 1] - ts_hr[f, rw, 1]))
    else:
        feats.append(0.0)

    # Timing and release-window dynamics.
    feats.append(float(release_frame))
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats.append(float(np.nanmean(series[140:180])))
            feats.append(float(np.nanmax(vel[140:180])))
    else:
        feats.extend([0.0] * 6)

    return np.asarray(feats, dtype=np.float32)


def extract_feature_matrices_frame_triplet_compact(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return (
            cached["x_train"].astype(np.float32),
            cached["x_test"].astype(np.float32),
            [str(x) for x in cached["feature_names"].tolist()],
            cached["train_ids"],
            cached["test_ids"],
            cached["groups"],
        )

    meta_train = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    meta_test = {"id", "shot_id", "participant_id"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_train]
    keypoint_cols_test = [c for c in test_df.columns if c not in meta_test]
    if keypoint_cols != keypoint_cols_test:
        raise ValueError("Train/test keypoint columns mismatch")
    if len(keypoint_cols) % 3 != 0:
        raise ValueError("Expected keypoint columns grouped by x/y/z")

    kp_names = [col[:-2] for col in keypoint_cols if col.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}
    n_kp = len(kp_names)
    target_frames = [TARGET_FRAME_MAP[t] for t in TARGETS]

    frame_feature_names: List[str] = []

    def extract(df: pd.DataFrame, is_train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_feats: List[np.ndarray] = []
        ids: List[str] = []
        pids: List[int] = []
        for idx, row in df.iterrows():
            ts_3d = np.zeros((240, n_kp, 3), dtype=np.float32)
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                kp_i = col_i // 3
                coord_i = col_i % 3
                ts_3d[:, kp_i, coord_i] = arr

            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            release_frame = detect_release_frame(ts_3d, kp_index)
            feats = []
            for frame in target_frames:
                feats.extend(extract_compact_features_at_frame(ts_3d, ts_hr, kp_index, release_frame, frame))
            feats.append(float(row["participant_id"]))
            feat_arr = np.asarray(feats, dtype=np.float32)
            all_feats.append(feat_arr)
            ids.append(str(row["id"]))
            pids.append(int(row["participant_id"]))

            if (idx + 1) % 25 == 0:
                kind = "train" if is_train else "test"
                print(f"feature_extract_frame_triplet_{kind}: {idx + 1}/{len(df)}", flush=True)

        return (
            np.vstack(all_feats).astype(np.float32),
            np.asarray(ids),
            np.asarray(pids, dtype=int),
        )

    x_train, train_ids, groups = extract(train_df, is_train=True)
    x_test, test_ids, _ = extract(test_df, is_train=False)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    per_frame_size = (x_train.shape[1] - 1) // len(target_frames)
    for tname in TARGETS:
        for i in range(per_frame_size):
            frame_feature_names.append(f"{tname}_compact_f{i}")
    frame_feature_names.append("participant_id_feature")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x_train=x_train,
        x_test=x_test,
        feature_names=np.asarray(frame_feature_names, dtype=object),
        train_ids=train_ids,
        test_ids=test_ids,
        groups=groups,
    )
    return x_train, x_test, frame_feature_names, train_ids, test_ids, groups


def extract_feature_matrices_hybrid_advanced(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return (
            cached["x_train"].astype(np.float32),
            cached["x_test"].astype(np.float32),
            [str(x) for x in cached["feature_names"].tolist()],
            cached["train_ids"],
            cached["test_ids"],
            cached["groups"],
        )

    meta_train = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    meta_test = {"id", "shot_id", "participant_id"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_train]
    keypoint_cols_test = [c for c in test_df.columns if c not in meta_test]
    if keypoint_cols != keypoint_cols_test:
        raise ValueError("Train/test keypoint columns mismatch")

    init_hybrid_mapping(keypoint_cols)
    init_advanced_mapping(keypoint_cols)

    def extract(df: pd.DataFrame, is_train: bool) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
        feature_dicts: List[Dict[str, float]] = []
        ids: List[str] = []
        pids: List[int] = []
        for idx, row in df.iterrows():
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for c_idx, col in enumerate(keypoint_cols):
                ts[:, c_idx] = parse_array_string(row[col])
            hybrid = extract_hybrid_features(ts, int(row["participant_id"]), smooth=False)
            advanced = extract_advanced_features(ts, int(row["participant_id"]))
            feats = {**hybrid, **advanced}
            feature_dicts.append(feats)
            ids.append(row["id"])
            pids.append(int(row["participant_id"]))
            if (idx + 1) % 25 == 0:
                kind = "train" if is_train else "test"
                print(f"feature_extract_{kind}: {idx + 1}/{len(df)}", flush=True)
        return feature_dicts, np.array(ids), np.array(pids, dtype=int)

    train_dicts, train_ids, groups = extract(train_df, is_train=True)
    test_dicts, test_ids, _ = extract(test_df, is_train=False)

    feature_names = sorted(set().union(*(d.keys() for d in train_dicts + test_dicts)))
    x_train = np.array(
        [[float(d.get(name, 0.0)) for name in feature_names] for d in train_dicts],
        dtype=np.float32,
    )
    x_test = np.array(
        [[float(d.get(name, 0.0)) for name in feature_names] for d in test_dicts],
        dtype=np.float32,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x_train=x_train,
        x_test=x_test,
        feature_names=np.array(feature_names, dtype=object),
        train_ids=train_ids,
        test_ids=test_ids,
        groups=groups,
    )
    return x_train, x_test, feature_names, train_ids, test_ids, groups


def extract_feature_matrices_by_bank(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    feature_bank: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray]:
    hybrid_cache_path = output_dir / "research_standard_hybrid_advanced_feature_cache.npz"
    frame_cache_path = output_dir / "research_standard_frame_triplet_compact_feature_cache.npz"

    if feature_bank == "hybrid_advanced":
        return extract_feature_matrices_hybrid_advanced(
            train_df=train_df,
            test_df=test_df,
            cache_path=hybrid_cache_path,
        )
    if feature_bank == "frame_triplet_compact":
        return extract_feature_matrices_frame_triplet_compact(
            train_df=train_df,
            test_df=test_df,
            cache_path=frame_cache_path,
        )
    if feature_bank == "hybrid_advanced_plus_frame_triplet":
        (
            x_train_h,
            x_test_h,
            names_h,
            train_ids_h,
            test_ids_h,
            groups_h,
        ) = extract_feature_matrices_hybrid_advanced(
            train_df=train_df,
            test_df=test_df,
            cache_path=hybrid_cache_path,
        )
        (
            x_train_f,
            x_test_f,
            names_f,
            train_ids_f,
            test_ids_f,
            groups_f,
        ) = extract_feature_matrices_frame_triplet_compact(
            train_df=train_df,
            test_df=test_df,
            cache_path=frame_cache_path,
        )
        if not np.array_equal(train_ids_h, train_ids_f):
            raise ValueError("Train ID mismatch between feature banks")
        if not np.array_equal(test_ids_h, test_ids_f):
            raise ValueError("Test ID mismatch between feature banks")
        if not np.array_equal(groups_h, groups_f):
            raise ValueError("Participant group mismatch between feature banks")

        x_train = np.hstack([x_train_h, x_train_f]).astype(np.float32)
        x_test = np.hstack([x_test_h, x_test_f]).astype(np.float32)
        names = [f"hyb::{n}" for n in names_h] + [f"frm::{n}" for n in names_f]
        return x_train, x_test, names, train_ids_h, test_ids_h, groups_h

    raise ValueError(f"Unknown --feature-bank: {feature_bank}")


def build_model_pipeline(
    spec: ModelSpec,
    params: Dict[str, object],
    seed: int,
) -> Pipeline:
    steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.name in {"ridge", "elasticnet", "knn"}:
        steps.append(("scaler", StandardScaler()))

    if spec.name == "ridge":
        reg = Ridge(alpha=float(params["alpha"]), random_state=seed)
    elif spec.name == "elasticnet":
        reg = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=10000,
            random_state=seed,
        )
    elif spec.name == "knn":
        reg = KNeighborsRegressor(
            n_neighbors=int(params["n_neighbors"]),
            weights=str(params["weights"]),
        )
    elif spec.name == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
    elif spec.name == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model spec: {spec.name}")

    steps.append(("regressor", reg))
    return Pipeline(steps)


def tune_model_params(
    spec: ModelSpec,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    inner_folds: int,
    seed: int,
) -> Tuple[Dict[str, object], float, float]:
    splits = participant_stratified_kfold(groups, inner_folds, seed)
    best_params = dict(spec.param_grid[0])
    best_mean = float("inf")
    best_std = float("inf")

    for grid_idx, params in enumerate(spec.param_grid):
        fold_mses = []
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            model = build_model_pipeline(
                spec,
                dict(params),
                seed=seed + 100 * grid_idx + fold_id,
            )
            model.fit(x[tr_idx], y[tr_idx])
            pred = model.predict(x[va_idx])
            fold_mses.append(float(mean_squared_error(y[va_idx], pred)))
        mean_mse = float(np.mean(fold_mses))
        std_mse = float(np.std(fold_mses))
        if mean_mse < best_mean - 1e-15 or (
            abs(mean_mse - best_mean) <= 1e-15 and std_mse < best_std
        ):
            best_mean = mean_mse
            best_std = std_mse
            best_params = dict(params)

    return best_params, best_mean, best_std


def bootstrap_metric_distribution(
    errors: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(errors)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    return errors[idx].mean(axis=1)


def generate_weight_candidates(
    n_models: int,
    rng: np.random.Generator,
    n_dirichlet: int,
    pair_step: float,
) -> np.ndarray:
    candidates = []
    # One-hot.
    for i in range(n_models):
        w = np.zeros(n_models, dtype=float)
        w[i] = 1.0
        candidates.append(w)
    # Pairwise simplex line.
    steps = np.arange(pair_step, 1.0, pair_step)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            for wi in steps:
                w = np.zeros(n_models, dtype=float)
                w[i] = float(wi)
                w[j] = float(1.0 - wi)
                candidates.append(w)
    # Dirichlet random draws.
    alpha = np.ones(n_models, dtype=float)
    dir_draws = rng.dirichlet(alpha, size=n_dirichlet)
    candidates.extend(dir_draws)
    return np.asarray(candidates, dtype=float)


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
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
            (submission_dir / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> None:
    t0 = time.time()
    args = parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.top_k_submissions < 1:
        raise ValueError("--top-k-submissions must be >= 1")

    run_tag = args.run_tag.strip() or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    submission_dir = args.submission_dir
    research_dir = args.research_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Scale-driven compute budget.
    if args.scale <= 1:
        n_outer_repeats = 1
        inner_folds = 2
        n_bootstrap = 40
        n_dirichlet_target = 80
    else:
        n_outer_repeats = 3 * args.scale
        inner_folds = 3
        n_bootstrap = 60 * args.scale
        n_dirichlet_target = 120 * args.scale
    outer_holdout_frac = 0.20
    pair_step = 0.10
    uncertainty_lambda = 0.40
    info_diversity_weight = 0.20
    info_uncertainty_weight = 0.10

    # Load data.
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    (
        x_train,
        x_test,
        feature_names,
        train_ids,
        test_ids,
        groups,
    ) = extract_feature_matrices_by_bank(
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir,
        feature_bank=args.feature_bank,
    )

    y_raw = train_df[TARGETS].to_numpy(dtype=float)
    scalers = load_scalers()
    y_scaled = scale_targets(y_raw, scalers)

    model_specs = filter_model_specs(get_model_specs(scale=args.scale), args.models)
    model_names = [spec.name for spec in model_specs]
    n_models = len(model_specs)
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    # Storage for nested results.
    oof_sum = np.zeros((n_models, n_train, len(TARGETS)), dtype=float)
    oof_count = np.zeros((n_models, n_train, len(TARGETS)), dtype=int)
    test_pred_repeats = np.zeros(
        (n_models, n_outer_repeats, n_test, len(TARGETS)), dtype=float
    )
    outer_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []

    # Nested validation.
    for repeat_idx in range(n_outer_repeats):
        print(f"nested_repeat: {repeat_idx + 1}/{n_outer_repeats}", flush=True)
        train_idx, val_idx = participant_stratified_holdout(
            groups=groups,
            holdout_frac=outer_holdout_frac,
            seed=args.seed + 1000 + repeat_idx,
        )
        g_train = groups[train_idx]
        for target_idx, target in enumerate(TARGETS):
            y_target = y_scaled[:, target_idx]
            y_target_train = y_target[train_idx]
            print(
                f"nested_target: repeat={repeat_idx + 1}/{n_outer_repeats} "
                f"target={target} ({target_idx + 1}/{len(TARGETS)})",
                flush=True,
            )

            for model_idx, spec in enumerate(model_specs):
                print(
                    f"nested_model: repeat={repeat_idx + 1}/{n_outer_repeats} "
                    f"target={target} model={spec.name} ({model_idx + 1}/{n_models})",
                    flush=True,
                )
                best_params, inner_mean, inner_std = tune_model_params(
                    spec=spec,
                    x=x_train[train_idx],
                    y=y_target_train,
                    groups=g_train,
                    inner_folds=inner_folds,
                    seed=args.seed + 100000 + repeat_idx * 97 + target_idx * 13 + model_idx,
                )

                model = build_model_pipeline(
                    spec=spec,
                    params=best_params,
                    seed=args.seed + 200000 + repeat_idx * 97 + target_idx * 13 + model_idx,
                )
                model.fit(x_train[train_idx], y_target[train_idx])

                val_pred = np.asarray(model.predict(x_train[val_idx]), dtype=float)
                val_mse = float(mean_squared_error(y_target[val_idx], val_pred))
                oof_sum[model_idx, val_idx, target_idx] += val_pred
                oof_count[model_idx, val_idx, target_idx] += 1

                test_pred = np.asarray(model.predict(x_test), dtype=float)
                test_pred_repeats[model_idx, repeat_idx, :, target_idx] = test_pred

                outer_rows.append(
                    {
                        "repeat": repeat_idx,
                        "target": target,
                        "model": spec.name,
                        "outer_val_mse": val_mse,
                        "inner_best_mean_mse": inner_mean,
                        "inner_best_std_mse": inner_std,
                        "n_train_outer": int(len(train_idx)),
                        "n_val_outer": int(len(val_idx)),
                    }
                )
                tuning_rows.append(
                    {
                        "repeat": repeat_idx,
                        "target": target,
                        "model": spec.name,
                        "best_params_json": json.dumps(best_params, sort_keys=True),
                        "inner_best_mean_mse": inner_mean,
                        "inner_best_std_mse": inner_std,
                    }
                )

    # Aggregate OOF predictions.
    oof_preds = np.full((n_models, n_train, len(TARGETS)), np.nan, dtype=float)
    for m in range(n_models):
        for t in range(len(TARGETS)):
            mask = oof_count[m, :, t] > 0
            if np.any(mask):
                oof_preds[m, mask, t] = oof_sum[m, mask, t] / oof_count[m, mask, t]
                fill_value = float(np.nanmean(oof_preds[m, mask, t]))
            else:
                fill_value = float(np.mean(y_scaled[:, t]))
            oof_preds[m, ~mask, t] = fill_value

    # Mean test predictions across repeats.
    test_preds_mean = np.mean(test_pred_repeats, axis=1)

    # Per-target probabilistic weight search.
    target_weight_rows: List[Dict[str, object]] = []
    target_top_weight_indices: Dict[int, np.ndarray] = {}
    target_weight_candidates: Dict[int, np.ndarray] = {}
    target_oof_matrix: Dict[int, np.ndarray] = {}

    for target_idx, target in enumerate(TARGETS):
        mat = oof_preds[:, :, target_idx].T  # (n_train, n_models)
        y_t = y_scaled[:, target_idx]
        target_oof_matrix[target_idx] = mat

        # Baseline single-model distribution.
        single_mses = []
        single_boot = []
        for model_idx in range(n_models):
            err = (mat[:, model_idx] - y_t) ** 2
            boot = bootstrap_metric_distribution(
                errors=err,
                n_bootstrap=n_bootstrap,
                rng=np.random.default_rng(args.seed + 300000 + target_idx * 100 + model_idx),
            )
            single_mses.append(float(np.mean(err)))
            single_boot.append(boot)
        best_single_idx = int(np.argmin(single_mses))
        baseline_boot = single_boot[best_single_idx]

        weights = generate_weight_candidates(
            n_models=n_models,
            rng=np.random.default_rng(args.seed + 400000 + target_idx),
            n_dirichlet=n_dirichlet_target,
            pair_step=pair_step,
        )
        target_weight_candidates[target_idx] = weights

        for w_idx, w in enumerate(weights):
            pred = mat @ w
            err = (pred - y_t) ** 2
            boot = bootstrap_metric_distribution(
                errors=err,
                n_bootstrap=n_bootstrap,
                rng=np.random.default_rng(args.seed + 500000 + target_idx * 200000 + w_idx),
            )
            mean_b = float(np.mean(boot))
            std_b = float(np.std(boot))
            q10, q50, q90 = np.quantile(boot, [0.10, 0.50, 0.90])
            p_better_single = float(np.mean(boot < baseline_boot))
            score = mean_b + uncertainty_lambda * std_b
            target_weight_rows.append(
                {
                    "target": target,
                    "weight_index": w_idx,
                    "weights_json": json.dumps({model_names[i]: float(w[i]) for i in range(n_models)}),
                    "mean_boot_mse": mean_b,
                    "std_boot_mse": std_b,
                    "q10": float(q10),
                    "q50": float(q50),
                    "q90": float(q90),
                    "p_better_single": p_better_single,
                    "score": score,
                    "best_single_model": model_names[best_single_idx],
                    "best_single_mse": float(single_mses[best_single_idx]),
                }
            )

        target_df = pd.DataFrame([r for r in target_weight_rows if r["target"] == target]).sort_values(
            by=["score", "mean_boot_mse", "std_boot_mse"], ascending=[True, True, True]
        )
        target_top_weight_indices[target_idx] = target_df["weight_index"].head(4).to_numpy(dtype=int)

    target_weights_df = pd.DataFrame(target_weight_rows)

    # Ensemble candidate combinations from top target weights.
    combo_rows: List[Dict[str, object]] = []
    combo_test_preds: List[np.ndarray] = []
    combo_boot_std: List[float] = []

    weight_index_product = itertools.product(
        target_top_weight_indices[0],
        target_top_weight_indices[1],
        target_top_weight_indices[2],
    )
    for combo_id, (w0, w1, w2) in enumerate(weight_index_product):
        w_angle = target_weight_candidates[0][int(w0)]
        w_depth = target_weight_candidates[1][int(w1)]
        w_lr = target_weight_candidates[2][int(w2)]

        pred_train = np.zeros((n_train, 3), dtype=float)
        pred_train[:, 0] = target_oof_matrix[0] @ w_angle
        pred_train[:, 1] = target_oof_matrix[1] @ w_depth
        pred_train[:, 2] = target_oof_matrix[2] @ w_lr

        err = np.mean((pred_train - y_scaled) ** 2, axis=1)
        boot = bootstrap_metric_distribution(
            errors=err,
            n_bootstrap=n_bootstrap,
            rng=np.random.default_rng(args.seed + 800000 + combo_id),
        )
        mean_b = float(np.mean(boot))
        std_b = float(np.std(boot))
        q10, q50, q90 = np.quantile(boot, [0.10, 0.50, 0.90])
        score = mean_b + uncertainty_lambda * std_b

        # Test predictions.
        pred_test = np.zeros((n_test, 3), dtype=float)
        pred_test[:, 0] = test_preds_mean[:, :, 0].T @ w_angle
        pred_test[:, 1] = test_preds_mean[:, :, 1].T @ w_depth
        pred_test[:, 2] = test_preds_mean[:, :, 2].T @ w_lr
        pred_test = np.clip(pred_test, 0.0, 1.0)

        combo_rows.append(
            {
                "combo_id": combo_id,
                "w_angle_index": int(w0),
                "w_depth_index": int(w1),
                "w_lr_index": int(w2),
                "w_angle_json": json.dumps({model_names[i]: float(w_angle[i]) for i in range(n_models)}),
                "w_depth_json": json.dumps({model_names[i]: float(w_depth[i]) for i in range(n_models)}),
                "w_lr_json": json.dumps({model_names[i]: float(w_lr[i]) for i in range(n_models)}),
                "mean_boot_mse": mean_b,
                "std_boot_mse": std_b,
                "q10": float(q10),
                "q50": float(q50),
                "q90": float(q90),
                "score": score,
            }
        )
        combo_boot_std.append(std_b)
        combo_test_preds.append(pred_test)

    combo_df = pd.DataFrame(combo_rows).sort_values(
        by=["score", "mean_boot_mse", "std_boot_mse"], ascending=[True, True, True]
    )
    combo_df.reset_index(drop=True, inplace=True)

    # Information-gain-oriented selection:
    # first best score, then trade off score, diversity, and uncertainty.
    selected_combo_ids: List[int] = []
    selected_indices: List[int] = []
    for rank_pos, row in combo_df.iterrows():
        combo_id = int(row["combo_id"])
        if len(selected_combo_ids) == 0:
            selected_combo_ids.append(combo_id)
            selected_indices.append(rank_pos)
            if len(selected_combo_ids) >= args.top_k_submissions:
                break
            continue

        pred = combo_test_preds[combo_id]
        min_dist = 1.0
        for prev_id in selected_combo_ids:
            prev_pred = combo_test_preds[prev_id]
            corr_vals = [corr_safe(pred[:, i], prev_pred[:, i]) for i in range(3)]
            dist = 1.0 - float(np.mean(corr_vals))
            min_dist = min(min_dist, dist)

        utility = (
            -float(row["score"])
            + info_diversity_weight * float(min_dist)
            + info_uncertainty_weight * float(row["std_boot_mse"])
        )

        row_for_pick = dict(row)
        row_for_pick["utility"] = utility
        # Temporarily store utility for candidates considered.
        combo_df.loc[rank_pos, "utility"] = utility

    # Greedy pick based on utility among non-selected, recomputing each round.
    while len(selected_combo_ids) < args.top_k_submissions:
        best_rank = None
        best_utility = -float("inf")
        for rank_pos, row in combo_df.iterrows():
            combo_id = int(row["combo_id"])
            if combo_id in selected_combo_ids:
                continue
            pred = combo_test_preds[combo_id]
            min_dist = 1.0
            for prev_id in selected_combo_ids:
                prev_pred = combo_test_preds[prev_id]
                corr_vals = [corr_safe(pred[:, i], prev_pred[:, i]) for i in range(3)]
                dist = 1.0 - float(np.mean(corr_vals))
                min_dist = min(min_dist, dist)
            utility = (
                -float(row["score"])
                + info_diversity_weight * float(min_dist)
                + info_uncertainty_weight * float(row["std_boot_mse"])
            )
            if utility > best_utility:
                best_utility = utility
                best_rank = rank_pos
        if best_rank is None:
            break
        combo_id = int(combo_df.loc[best_rank, "combo_id"])
        selected_combo_ids.append(combo_id)
        selected_indices.append(best_rank)

    selected_df = combo_df.loc[selected_indices].copy()
    selected_df.sort_values(by=["score", "mean_boot_mse", "std_boot_mse"], inplace=True)
    selected_df.reset_index(drop=True, inplace=True)

    # Create submissions.
    submission_rows: List[Dict[str, object]] = []
    for _, row in selected_df.iterrows():
        combo_id = int(row["combo_id"])
        pred_test = combo_test_preds[combo_id]
        sub_num = get_next_submission_number(submission_dir)
        sub_path = submission_dir / f"submission_{sub_num}.csv"
        sub_df = pd.DataFrame(
            {
                "id": test_ids,
                "scaled_angle": pred_test[:, 0],
                "scaled_depth": pred_test[:, 1],
                "scaled_left_right": pred_test[:, 2],
            }
        )
        sub_df.to_csv(sub_path, index=False, float_format="%.15f")
        submission_rows.append(
            {
                "submission_num": sub_num,
                "submission_file": str(sub_path),
                "combo_id": combo_id,
                "score": float(row["score"]),
                "mean_boot_mse": float(row["mean_boot_mse"]),
                "std_boot_mse": float(row["std_boot_mse"]),
                "q10": float(row["q10"]),
                "q50": float(row["q50"]),
                "q90": float(row["q90"]),
                "w_angle_json": row["w_angle_json"],
                "w_depth_json": row["w_depth_json"],
                "w_lr_json": row["w_lr_json"],
            }
        )

    # Save artifacts.
    outer_path = output_dir / f"research_standard_outer_metrics_{run_tag}.csv"
    tuning_path = output_dir / f"research_standard_tuning_{run_tag}.csv"
    target_weight_path = output_dir / f"research_standard_target_weight_search_{run_tag}.csv"
    combo_path = output_dir / f"research_standard_ensemble_candidates_{run_tag}.csv"
    selected_path = output_dir / f"research_standard_selected_{run_tag}.csv"
    run_json_path = output_dir / f"research_standard_run_{run_tag}.json"
    details_path = output_dir / f"research_standard_submission_details_{run_tag}.md"
    research_summary_path = research_dir / f"RESEARCH_STANDARD_PLAN_RESULTS_{run_tag}.md"

    pd.DataFrame(outer_rows).to_csv(outer_path, index=False, float_format="%.15f")
    pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False, float_format="%.15f")
    target_weights_df.to_csv(target_weight_path, index=False, float_format="%.15f")
    combo_df.to_csv(combo_path, index=False, float_format="%.15f")
    selected_df.to_csv(selected_path, index=False, float_format="%.15f")

    elapsed = time.time() - t0

    # Summaries.
    outer_df = pd.DataFrame(outer_rows)
    model_summary = (
        outer_df.groupby(["model", "target"], as_index=False)
        .agg(
            mean_mse=("outer_val_mse", "mean"),
            std_mse=("outer_val_mse", "std"),
            min_mse=("outer_val_mse", "min"),
            max_mse=("outer_val_mse", "max"),
        )
    )

    run_payload = {
        "run_tag": run_tag,
        "command": " ".join(__import__("sys").argv),
        "seed": args.seed,
        "scale": args.scale,
        "feature_bank": args.feature_bank,
        "models_arg": args.models,
        "best_lb": args.best_lb,
        "n_outer_repeats": n_outer_repeats,
        "outer_holdout_frac": outer_holdout_frac,
        "inner_folds": inner_folds,
        "n_bootstrap": n_bootstrap,
        "n_dirichlet_target": n_dirichlet_target,
        "pair_step": pair_step,
        "uncertainty_lambda": uncertainty_lambda,
        "model_names": model_names,
        "n_train": n_train,
        "n_test": n_test,
        "feature_count": len(feature_names),
        "outer_metrics_csv": str(outer_path),
        "tuning_csv": str(tuning_path),
        "target_weight_csv": str(target_weight_path),
        "ensemble_candidates_csv": str(combo_path),
        "selected_csv": str(selected_path),
        "submission_details_md": str(details_path),
        "research_summary_md": str(research_summary_path),
        "selected_submissions": submission_rows,
        "model_summary": model_summary.to_dict(orient="records"),
        "elapsed_seconds": elapsed,
    }
    with open(run_json_path, "w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2)

    lines: List[str] = []
    lines.append("# Research-Standard Nested OOF Ensemble Run")
    lines.append("")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- command: `{run_payload['command']}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- scale: `{args.scale}`")
    lines.append(f"- feature_bank: `{args.feature_bank}`")
    lines.append(f"- models_arg: `{args.models}`")
    lines.append(f"- best_lb_reference: `{args.best_lb:.15f}`")
    lines.append(f"- n_outer_repeats: `{n_outer_repeats}`")
    lines.append(f"- outer_holdout_frac: `{outer_holdout_frac}`")
    lines.append(f"- inner_folds: `{inner_folds}`")
    lines.append(f"- n_bootstrap: `{n_bootstrap}`")
    lines.append(f"- n_dirichlet_target: `{n_dirichlet_target}`")
    lines.append(f"- uncertainty_lambda: `{uncertainty_lambda}`")
    lines.append(f"- n_models: `{n_models}`")
    lines.append(f"- n_train: `{n_train}`")
    lines.append(f"- n_test: `{n_test}`")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    for rec in run_payload["model_summary"]:
        lines.append(
            f"- `{rec['model']}` `{rec['target']}`: "
            f"mean_mse=`{rec['mean_mse']:.15f}`, std_mse=`{rec['std_mse']:.15f}`"
        )
    lines.append("")
    lines.append("## Selected Submissions")
    lines.append("")
    for item in submission_rows:
        lines.append(f"- file: `{item['submission_file']}`")
        lines.append(f"  - submission_num: `{item['submission_num']}`")
        lines.append(f"  - combo_id: `{item['combo_id']}`")
        lines.append(f"  - score: `{item['score']:.15f}`")
        lines.append(f"  - mean_boot_mse: `{item['mean_boot_mse']:.15f}`")
        lines.append(f"  - std_boot_mse: `{item['std_boot_mse']:.15f}`")
        lines.append(f"  - q10/q50/q90: `{item['q10']:.15f}` / `{item['q50']:.15f}` / `{item['q90']:.15f}`")
        lines.append(f"  - w_angle: `{item['w_angle_json']}`")
        lines.append(f"  - w_depth: `{item['w_depth_json']}`")
        lines.append(f"  - w_lr: `{item['w_lr_json']}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- outer_metrics_csv: `{outer_path}`")
    lines.append(f"- tuning_csv: `{tuning_path}`")
    lines.append(f"- target_weight_csv: `{target_weight_path}`")
    lines.append(f"- ensemble_candidates_csv: `{combo_path}`")
    lines.append(f"- selected_csv: `{selected_path}`")
    lines.append(f"- run_json: `{run_json_path}`")
    lines.append(f"- details_md: `{details_path}`")
    lines.append(f"- elapsed_seconds: `{elapsed:.6f}`")
    lines.append("")
    with open(details_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    with open(research_summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print("=" * 80)
    print("RESEARCH-STANDARD NESTED OOF ENSEMBLE COMPLETE")
    print("=" * 80)
    print(f"run_tag={run_tag}")
    print(f"scale={args.scale}")
    print(f"feature_bank={args.feature_bank}")
    print(f"models_arg={args.models}")
    print(f"seed={args.seed}")
    print(f"n_outer_repeats={n_outer_repeats}")
    print(f"inner_folds={inner_folds}")
    print(f"n_bootstrap={n_bootstrap}")
    print(f"n_models={n_models}")
    print(f"n_train={n_train}")
    print(f"n_test={n_test}")
    print(f"outer_metrics_csv={outer_path}")
    print(f"tuning_csv={tuning_path}")
    print(f"target_weight_csv={target_weight_path}")
    print(f"ensemble_candidates_csv={combo_path}")
    print(f"selected_csv={selected_path}")
    print(f"run_json={run_json_path}")
    print(f"details_md={details_path}")
    for item in submission_rows:
        print(
            f"submission_{item['submission_num']}.csv "
            f"score={item['score']:.15f} "
            f"mean_boot_mse={item['mean_boot_mse']:.15f} "
            f"std_boot_mse={item['std_boot_mse']:.15f}"
        )
    print(f"elapsed_seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
