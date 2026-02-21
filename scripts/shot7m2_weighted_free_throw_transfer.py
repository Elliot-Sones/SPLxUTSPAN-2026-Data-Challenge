"""
Weighted SHOT7M2 Transfer Pipeline

Purpose:
1. Align coordinate systems between competition data and SHOT7M2.
2. Weight SHOT7M2 shooting frames by free-throw-likeness (low move/dribble/sprint).
3. Evaluate with honest per-player LOO.
4. Generate submission candidates and exact run artifacts.
"""

import argparse
import fcntl
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
SHOT7M2_DIR = PROJECT_DIR / "external_data" / "shot7m2_sample"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP = np.array([5.25, -25.0, 10.0], dtype=float)

JOINT_MAP = {
    "mid_hip": 0,
    "left_hip": 1,
    "left_knee": 2,
    "left_ankle": 3,
    "right_hip": 6,
    "right_knee": 7,
    "right_ankle": 8,
    "left_shoulder": 16,
    "left_elbow": 17,
    "left_wrist": 18,
    "right_shoulder": 23,
    "right_elbow": 24,
    "right_wrist": 25,
    "neck": 19,
}
MAPPED_JOINTS = list(JOINT_MAP.keys())
SHOT7M2_INDICES = [JOINT_MAP[j] for j in MAPPED_JOINTS]

JOINT_ANGLE_DEFS = [
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("neck", "left_shoulder", "left_elbow"),
    ("neck", "right_shoulder", "right_elbow"),
    ("left_shoulder", "mid_hip", "right_shoulder"),
    ("left_hip", "mid_hip", "right_hip"),
    ("mid_hip", "neck", "right_shoulder"),
    ("mid_hip", "neck", "left_shoulder"),
]


@dataclass
class WeightedPCA:
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components.T

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return z @ self.components + self.mean


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_triplet_int(s: str) -> tuple[int, int, int]:
    vals = parse_int_list(s)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 ints, got: {s}")
    return vals[0], vals[1], vals[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weighted SHOT7M2 transfer pipeline")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--frames-per-scale", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260215)
    p.add_argument("--shoot-threshold", type=float, default=0.3)
    p.add_argument("--context-frames", type=int, default=30)
    p.add_argument("--axis-perm", type=str, default="0,2,1")
    p.add_argument("--axis-signs", type=str, default="1,1,1")
    p.add_argument("--n-pca-components", type=int, default=20)
    p.add_argument("--n-pca-features", type=int, default=10)
    p.add_argument("--n-pls-hoop", type=int, default=15)
    p.add_argument("--n-pls-shot-grid", type=str, default="1,3,5")
    p.add_argument("--bw-quantile", type=float, default=0.45)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--weight-dribble", type=float, default=1.0)
    p.add_argument("--weight-move", type=float, default=1.0)
    p.add_argument("--weight-sprint", type=float, default=1.0)
    p.add_argument("--weight-temperature", type=float, default=0.3)
    p.add_argument("--min-frame-weight", type=float, default=0.05)
    p.add_argument("--base-submission", type=int, default=2503)
    p.add_argument("--blend-weights", type=str, default="0.05,0.10,0.15,0.20,0.30")
    p.add_argument("--skip-submissions", action="store_true")
    return p.parse_args()


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                stem = fp.stem.split("_")
                if len(stem) == 2 and stem[1].isdigit():
                    nums.append(int(stem[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_json_safe(s: str, frame: int) -> float:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    val = arr[frame]
    if val is None:
        for offset in range(1, 10):
            if frame - offset >= 0 and arr[frame - offset] is not None:
                return float(arr[frame - offset])
            if frame + offset < len(arr) and arr[frame + offset] is not None:
                return float(arr[frame + offset])
        return 0.0
    return float(val)


def apply_axis_transform(
    positions: np.ndarray,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> np.ndarray:
    transformed = positions[:, :, perm]
    return transformed * np.asarray(signs, dtype=float)[np.newaxis, np.newaxis, :]


def normalize_pose(positions: np.ndarray) -> np.ndarray:
    pelvis_idx = MAPPED_JOINTS.index("mid_hip")
    neck_idx = MAPPED_JOINTS.index("neck")
    centered = positions - positions[:, pelvis_idx : pelvis_idx + 1, :]
    torso_len = np.linalg.norm(
        centered[:, neck_idx] - centered[:, pelvis_idx], axis=1, keepdims=True
    )
    torso_len = np.maximum(torso_len, 1e-6)
    return centered / torso_len[:, :, np.newaxis]


def compute_joint_angles(positions: np.ndarray) -> np.ndarray:
    angles = np.zeros((len(positions), len(JOINT_ANGLE_DEFS)))
    for i, (parent, joint, child) in enumerate(JOINT_ANGLE_DEFS):
        p_idx = MAPPED_JOINTS.index(parent)
        j_idx = MAPPED_JOINTS.index(joint)
        c_idx = MAPPED_JOINTS.index(child)
        v1 = positions[:, p_idx] - positions[:, j_idx]
        v2 = positions[:, c_idx] - positions[:, j_idx]
        v1 = v1 / np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-8)
        v2 = v2 / np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-8)
        cos_a = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
        angles[:, i] = np.arccos(cos_a)
    return angles


def fit_weighted_pca(x: np.ndarray, weights: np.ndarray, n_components: int) -> WeightedPCA:
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 1e-12)
    w = w / w.sum()

    mean = np.average(x, axis=0, weights=w)
    xc = x - mean
    xw = xc * np.sqrt(w)[:, np.newaxis]

    _, s, vt = np.linalg.svd(xw, full_matrices=False)
    max_components = min(n_components, vt.shape[0], x.shape[1])
    components = vt[:max_components]
    ev = (s ** 2) / np.maximum(np.sum(s ** 2), 1e-12)
    explained = ev[:max_components]
    return WeightedPCA(
        components=components,
        mean=mean,
        explained_variance_ratio=explained,
    )


def load_shot7m2_weighted(args: argparse.Namespace) -> dict:
    print("Loading SHOT7M2 with free-throw-like weighting...")
    poses = np.load(SHOT7M2_DIR / "train" / "train_dictionary_poses.npy", allow_pickle=True).item()
    actions = np.load(SHOT7M2_DIR / "train" / "train_dictionary_actions.npy", allow_pickle=True).item()

    vocab = actions["vocabulary"]
    labels = actions["label_array"]
    fnm = actions["frame_number_map"]

    shoot_idx = vocab.index("action_Shoot")
    dribble_idx = vocab.index("action_Dribble")
    move_idx = vocab.index("action_Move")
    sprint_idx = vocab.index("action_Sprint")

    all_poses = []
    all_weights = []
    all_conf = []
    all_mobility = []

    seq_keys = list(poses["sequences"]["keypoints"].keys())
    for key in seq_keys:
        start, end = fnm[key]
        ep_shoot = labels[shoot_idx, start:end]
        shoot_frames = np.where(ep_shoot > args.shoot_threshold)[0]
        if len(shoot_frames) == 0:
            continue

        ep_dribble = labels[dribble_idx, start:end]
        ep_move = labels[move_idx, start:end]
        ep_sprint = labels[sprint_idx, start:end]
        ep_poses = poses["sequences"]["keypoints"][key]

        for f in shoot_frames:
            ctx0 = max(0, f - args.context_frames)
            ctx1 = min(len(ep_shoot), f + args.context_frames + 1)

            d_mean = float(np.clip(np.mean(ep_dribble[ctx0:ctx1]), 0.0, 1.0))
            m_mean = float(np.clip(np.mean(ep_move[ctx0:ctx1]), 0.0, 1.0))
            s_mean = float(np.clip(np.mean(ep_sprint[ctx0:ctx1]), 0.0, 1.0))

            mobility = (
                args.weight_dribble * d_mean
                + args.weight_move * m_mean
                + args.weight_sprint * s_mean
            )
            ft_like = np.exp(-mobility / max(args.weight_temperature, 1e-8))
            conf = max(float(ep_shoot[f]), 0.0)
            weight = conf * (args.min_frame_weight + (1.0 - args.min_frame_weight) * ft_like)

            all_poses.append(ep_poses[f, 0, SHOT7M2_INDICES, :])
            all_weights.append(weight)
            all_conf.append(conf)
            all_mobility.append(mobility)

    if not all_poses:
        raise RuntimeError("No SHOT7M2 shooting frames found after filtering.")

    shoot_poses = np.asarray(all_poses, dtype=float)
    shoot_weights = np.asarray(all_weights, dtype=float)
    shoot_conf = np.asarray(all_conf, dtype=float)
    shoot_mobility = np.asarray(all_mobility, dtype=float)

    total_frames = len(shoot_poses)
    max_frames = min(total_frames, args.scale * args.frames_per_scale)
    order = np.argsort(-shoot_weights)
    keep_idx = order[:max_frames]

    keep_poses = shoot_poses[keep_idx]
    keep_weights = shoot_weights[keep_idx]
    keep_conf = shoot_conf[keep_idx]
    keep_mobility = shoot_mobility[keep_idx]

    keep_norm = normalize_pose(keep_poses)
    keep_angles = compute_joint_angles(keep_norm)
    keep_flat = keep_norm.reshape(len(keep_norm), -1)

    pca = fit_weighted_pca(keep_flat, keep_weights, args.n_pca_components)
    angle_means = np.average(keep_angles, axis=0, weights=keep_weights)
    angle_vars = np.average((keep_angles - angle_means) ** 2, axis=0, weights=keep_weights)
    angle_stds = np.sqrt(np.maximum(angle_vars, 1e-10))
    centroid = np.average(keep_flat, axis=0, weights=keep_weights)

    shot_stats = {
        "total_shooting_frames": int(total_frames),
        "kept_frames": int(max_frames),
        "kept_ratio": float(max_frames / total_frames),
        "weight_mean": float(np.mean(shoot_weights)),
        "weight_median": float(np.median(shoot_weights)),
        "weight_max": float(np.max(shoot_weights)),
        "kept_weight_mean": float(np.mean(keep_weights)),
        "kept_conf_mean": float(np.mean(keep_conf)),
        "kept_mobility_mean": float(np.mean(keep_mobility)),
        "pca_explained_top10": float(np.sum(pca.explained_variance_ratio[:10])),
    }

    print(f"  total shoot frames: {shot_stats['total_shooting_frames']}")
    print(f"  kept frames (scale={args.scale}): {shot_stats['kept_frames']}")
    print(f"  kept ratio: {shot_stats['kept_ratio']:.6f}")
    print(f"  kept conf mean: {shot_stats['kept_conf_mean']:.6f}")
    print(f"  kept mobility mean: {shot_stats['kept_mobility_mean']:.6f}")
    print(f"  weighted PCA explained (top10): {shot_stats['pca_explained_top10']:.6f}")

    return {
        "pca": pca,
        "angle_means": angle_means,
        "angle_stds": angle_stds,
        "centroid": centroid,
        "stats": shot_stats,
    }


def extract_shot7m2_features(
    positions_14: np.ndarray,
    shot_data: dict,
    n_pca_features: int,
) -> np.ndarray:
    n = len(positions_14)
    pca: WeightedPCA = shot_data["pca"]
    angle_means = shot_data["angle_means"]
    angle_stds = shot_data["angle_stds"]
    centroid = shot_data["centroid"]

    norm = normalize_pose(positions_14)
    flat = norm.reshape(n, -1)

    transformed = pca.transform(flat)
    k = min(n_pca_features, transformed.shape[1])
    pca_feats = transformed[:, :k]
    if k < n_pca_features:
        pad = np.zeros((n, n_pca_features - k))
        pca_feats = np.hstack([pca_feats, pad])

    reconstructed = pca.inverse_transform(transformed)
    recon_error = np.mean((flat - reconstructed) ** 2, axis=1, keepdims=True)

    angles = compute_joint_angles(norm)
    z_scores = (angles - angle_means[np.newaxis, :]) / np.maximum(
        angle_stds[np.newaxis, :], 1e-8
    )

    dist_centroid = np.linalg.norm(flat - centroid[np.newaxis, :], axis=1, keepdims=True)

    neck_idx = MAPPED_JOINTS.index("neck")
    rw_idx = MAPPED_JOINTS.index("right_wrist")
    lw_idx = MAPPED_JOINTS.index("left_wrist")
    rs_idx = MAPPED_JOINTS.index("right_shoulder")
    ls_idx = MAPPED_JOINTS.index("left_shoulder")
    re_idx = MAPPED_JOINTS.index("right_elbow")
    le_idx = MAPPED_JOINTS.index("left_elbow")

    r_wrist_h = (norm[:, rw_idx, 1] - norm[:, neck_idx, 1]).reshape(-1, 1)
    l_wrist_h = (norm[:, lw_idx, 1] - norm[:, neck_idx, 1]).reshape(-1, 1)
    r_ext = np.linalg.norm(norm[:, rw_idx] - norm[:, rs_idx], axis=1, keepdims=True)
    l_ext = np.linalg.norm(norm[:, lw_idx] - norm[:, ls_idx], axis=1, keepdims=True)
    r_upper = np.linalg.norm(norm[:, rs_idx] - norm[:, re_idx], axis=1, keepdims=True)
    l_upper = np.linalg.norm(norm[:, ls_idx] - norm[:, le_idx], axis=1, keepdims=True)
    arm_asym = r_upper - l_upper

    return np.hstack(
        [
            pca_feats,
            recon_error,
            z_scores,
            angles,
            dist_centroid,
            r_wrist_h,
            l_wrist_h,
            r_ext,
            l_ext,
            arm_asym,
        ]
    )


def extract_14_joints(
    df: pd.DataFrame,
    frame: int,
    axis_perm: tuple[int, int, int],
    axis_signs: tuple[int, int, int],
) -> np.ndarray:
    n = len(df)
    positions = np.zeros((n, 14, 3), dtype=float)
    for j_idx, name in enumerate(MAPPED_JOINTS):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{name}_{coord}"
            positions[:, j_idx, c_idx] = df[col].apply(lambda x: parse_json_safe(x, frame)).values
    return apply_axis_transform(positions, axis_perm, axis_signs)


def build_all_features(
    df: pd.DataFrame,
    frame: int,
    shot_data: dict,
    axis_perm: tuple[int, int, int],
    axis_signs: tuple[int, int, int],
    n_pca_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    kp_names = sorted(
        set(c.rsplit("_", 1)[0] for c in df.columns if c.endswith(("_x", "_y", "_z")))
    )

    positions = np.zeros((n, len(kp_names), 3), dtype=float)
    for k_idx, kp in enumerate(kp_names):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{kp}_{coord}"
            positions[:, k_idx, c_idx] = df[col].apply(lambda x: parse_json_safe(x, frame)).values
    hoop_rel = positions - HOOP[np.newaxis, np.newaxis, :]
    hoop_features = hoop_rel.reshape(n, -1)

    positions_14 = extract_14_joints(df, frame, axis_perm, axis_signs)
    shot7m2_features = extract_shot7m2_features(positions_14, shot_data, n_pca_features)
    return hoop_features, shot7m2_features


def run_per_player_loo(
    hoop_features: np.ndarray,
    shot7m2_features: np.ndarray | None,
    targets: np.ndarray,
    player_ids: np.ndarray,
    use_shot7m2: bool,
    n_pls_hoop: int,
    n_pls_shot7m2: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    n = len(targets)
    predictions = np.zeros(n, dtype=float)
    players = sorted(np.unique(player_ids))

    for pid in players:
        mask = player_ids == pid
        idx = np.where(mask)[0]
        n_player = len(idx)

        x_hoop = hoop_features[mask]
        y = targets[mask]
        x_s7 = shot7m2_features[mask] if use_shot7m2 and shot7m2_features is not None else None

        for i_local in range(n_player):
            i_global = idx[i_local]
            loo_mask = np.ones(n_player, dtype=bool)
            loo_mask[i_local] = False

            xh_tr = x_hoop[loo_mask]
            xh_te = x_hoop[~loo_mask]
            y_tr = y[loo_mask]

            scaler_h = StandardScaler()
            xh_tr_s = scaler_h.fit_transform(xh_tr)
            xh_te_s = scaler_h.transform(xh_te)

            parts_tr = [xh_tr_s]
            parts_te = [xh_te_s]

            n_pls_h = min(n_pls_hoop, xh_tr_s.shape[1], xh_tr_s.shape[0] - 1)
            if n_pls_h > 0:
                pls_h = PLSRegression(n_components=n_pls_h)
                pls_h.fit(xh_tr_s, y_tr)
                parts_tr.append(pls_h.transform(xh_tr_s))
                parts_te.append(pls_h.transform(xh_te_s))

            if use_shot7m2 and x_s7 is not None:
                xs_tr = x_s7[loo_mask]
                xs_te = x_s7[~loo_mask]
                scaler_s = StandardScaler()
                xs_tr_s = scaler_s.fit_transform(xs_tr)
                xs_te_s = scaler_s.transform(xs_te)

                n_pls_s = min(n_pls_shot7m2, xs_tr_s.shape[1], xs_tr_s.shape[0] - 1)
                if n_pls_s > 0:
                    pls_s = PLSRegression(n_components=n_pls_s)
                    pls_s.fit(xs_tr_s, y_tr)
                    parts_tr.append(pls_s.transform(xs_tr_s))
                    parts_te.append(pls_s.transform(xs_te_s))

            x_tr_full = np.hstack(parts_tr)
            x_te_full = np.hstack(parts_te)

            dists = np.linalg.norm(x_tr_full - x_te_full, axis=1)
            bw = np.quantile(dists, bw_quantile)
            weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)

            w = np.diag(np.sqrt(weights))
            ridge = Ridge(alpha=alpha)
            ridge.fit(w @ x_tr_full, w @ y_tr)
            predictions[i_global] = ridge.predict(x_te_full)[0]

    return predictions


def predict_test_target(
    xh_train: np.ndarray,
    xs_train: np.ndarray,
    y_train: np.ndarray,
    train_pids: np.ndarray,
    xh_test: np.ndarray,
    xs_test: np.ndarray,
    test_pids: np.ndarray,
    n_pls_hoop: int,
    n_pls_shot: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    preds = np.zeros(len(xh_test), dtype=float)
    players = sorted(np.unique(train_pids))

    for pid in players:
        tr_mask = train_pids == pid
        te_mask = test_pids == pid
        te_idx = np.where(te_mask)[0]
        if len(te_idx) == 0:
            continue

        xh = xh_train[tr_mask]
        xs = xs_train[tr_mask]
        y = y_train[tr_mask]
        xh_te = xh_test[te_mask]
        xs_te = xs_test[te_mask]

        sc_h = StandardScaler()
        xh_s = sc_h.fit_transform(xh)
        xh_te_s = sc_h.transform(xh_te)

        sc_s = StandardScaler()
        xs_s = sc_s.fit_transform(xs)
        xs_te_s = sc_s.transform(xs_te)

        parts_train = [xh_s]
        parts_test = [xh_te_s]

        nph = min(n_pls_hoop, xh_s.shape[1], xh_s.shape[0] - 1)
        if nph > 0:
            pls_h = PLSRegression(n_components=nph)
            pls_h.fit(xh_s, y)
            parts_train.append(pls_h.transform(xh_s))
            parts_test.append(pls_h.transform(xh_te_s))

        nps = min(n_pls_shot, xs_s.shape[1], xs_s.shape[0] - 1)
        if nps > 0:
            pls_s = PLSRegression(n_components=nps)
            pls_s.fit(xs_s, y)
            parts_train.append(pls_s.transform(xs_s))
            parts_test.append(pls_s.transform(xs_te_s))

        x_full = np.hstack(parts_train)
        x_full_te = np.hstack(parts_test)

        for i in range(len(x_full_te)):
            dists = np.linalg.norm(x_full - x_full_te[i], axis=1)
            bw = np.quantile(dists, bw_quantile)
            weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)
            w = np.diag(np.sqrt(weights))
            ridge = Ridge(alpha=alpha)
            ridge.fit(w @ x_full, w @ y)
            preds[te_idx[i]] = ridge.predict(x_full_te[i : i + 1])[0]

    return preds


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    axis_perm = parse_triplet_int(args.axis_perm)
    axis_signs = parse_triplet_int(args.axis_signs)
    n_pls_shot_grid = parse_int_list(args.n_pls_shot_grid)
    blend_weights = parse_float_list(args.blend_weights)

    print("=" * 80)
    print("SHOT7M2 WEIGHTED FREE-THROW-LIKE TRANSFER")
    print("=" * 80)
    print(f"axis_perm={axis_perm}, axis_signs={axis_signs}")
    print(f"scale={args.scale}, frames_per_scale={args.frames_per_scale}")
    print(f"n_pls_shot_grid={n_pls_shot_grid}")

    shot_data = load_shot7m2_weighted(args)

    print("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    player_ids = train_df["participant_id"].values
    test_pids = test_df["participant_id"].values

    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    scaled_targets = {
        t: scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel() for t in TARGETS
    }

    train_hoop = {}
    train_s7 = {}
    test_hoop = {}
    test_s7 = {}

    print("\nExtracting features...")
    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        print(f"  {target}: frame={frame}")
        h_tr, s_tr = build_all_features(
            train_df,
            frame,
            shot_data,
            axis_perm,
            axis_signs,
            args.n_pca_features,
        )
        h_te, s_te = build_all_features(
            test_df,
            frame,
            shot_data,
            axis_perm,
            axis_signs,
            args.n_pca_features,
        )
        train_hoop[target] = h_tr
        train_s7[target] = s_tr
        test_hoop[target] = h_te
        test_s7[target] = s_te
        print(f"    hoop={h_tr.shape} shot7m2={s_tr.shape}")

    print("\n" + "=" * 80)
    print("HONEST LOO EVALUATION")
    print("=" * 80)

    configs = [("baseline_hoop_only", False, 0)]
    for n_pls_s in n_pls_shot_grid:
        configs.append((f"hoop_plus_weighted_s7_{n_pls_s}pls", True, n_pls_s))

    all_results: dict[str, dict] = {}
    for name, use_s7, n_pls_s in configs:
        t0 = time.time()
        target_mse = {}
        print(f"\nConfig: {name}")
        for target in TARGETS:
            preds = run_per_player_loo(
                train_hoop[target],
                train_s7[target] if use_s7 else None,
                scaled_targets[target],
                player_ids,
                use_shot7m2=use_s7,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_shot7m2=n_pls_s,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            mse = float(np.mean((preds - scaled_targets[target]) ** 2))
            target_mse[target] = mse
            print(f"  {target}: {mse:.12f}")
        target_mse["mean"] = float(np.mean([target_mse[t] for t in TARGETS]))
        target_mse["time_s"] = time.time() - t0
        all_results[name] = target_mse
        print(f"  mean: {target_mse['mean']:.12f} [{target_mse['time_s']:.3f}s]")

    baseline = all_results["baseline_hoop_only"]["mean"]
    print("\nComparison vs baseline:")
    for name in all_results:
        delta_pct = (all_results[name]["mean"] - baseline) / baseline * 100.0
        all_results[name]["delta_pct"] = float(delta_pct)
        print(f"  {name}: mean={all_results[name]['mean']:.12f}, delta_pct={delta_pct:+.12f}")

    best_name = min(
        [k for k in all_results.keys() if k != "baseline_hoop_only"],
        key=lambda k: all_results[k]["mean"],
    )
    best_n_pls = int(best_name.split("_")[-1].replace("pls", ""))
    print(f"\nBest config: {best_name} (n_pls_shot={best_n_pls})")

    submissions_created = []
    if not args.skip_submissions:
        print("\n" + "=" * 80)
        print("GENERATING SUBMISSIONS")
        print("=" * 80)

        base_sub_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
        if not base_sub_path.exists():
            raise FileNotFoundError(f"Base submission not found: {base_sub_path}")
        base_sub = pd.read_csv(base_sub_path)
        standalone = base_sub.copy()

        for target, col in zip(TARGETS, TARGET_COLS):
            preds = predict_test_target(
                xh_train=train_hoop[target],
                xs_train=train_s7[target],
                y_train=scaled_targets[target],
                train_pids=player_ids,
                xh_test=test_hoop[target],
                xs_test=test_s7[target],
                test_pids=test_pids,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_shot=best_n_pls,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            standalone[col] = np.clip(preds, 0.0, 1.0)

        sub_num = get_next_submission_number()
        standalone_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        standalone.to_csv(standalone_path, index=False)
        submissions_created.append(
            {
                "submission_number": sub_num,
                "path": str(standalone_path),
                "type": "standalone_weighted_shot7m2",
                "blend_weight_shot7m2": 1.0,
            }
        )
        print(f"  standalone: submission_{sub_num}.csv")

        for bw in blend_weights:
            blended = base_sub.copy()
            for col in TARGET_COLS:
                blended[col] = np.clip(
                    bw * standalone[col].values + (1.0 - bw) * base_sub[col].values,
                    0.0,
                    1.0,
                )
            sub_num = get_next_submission_number()
            blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            blended.to_csv(blend_path, index=False)
            submissions_created.append(
                {
                    "submission_number": sub_num,
                    "path": str(blend_path),
                    "type": "blend_with_base",
                    "blend_weight_shot7m2": float(bw),
                    "blend_weight_base": float(1.0 - bw),
                    "base_submission": int(args.base_submission),
                }
            )
            print(
                f"  blend: submission_{sub_num}.csv "
                f"(shot7m2={bw:.6f}, base={1.0-bw:.6f})"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json = OUTPUT_DIR / f"shot7m2_weighted_ft_transfer_run_{ts}.json"
    run_md = OUTPUT_DIR / f"shot7m2_weighted_ft_transfer_details_{ts}.md"

    payload = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "config": {
            "scale": args.scale,
            "frames_per_scale": args.frames_per_scale,
            "seed": args.seed,
            "shoot_threshold": args.shoot_threshold,
            "context_frames": args.context_frames,
            "axis_perm": axis_perm,
            "axis_signs": axis_signs,
            "n_pca_components": args.n_pca_components,
            "n_pca_features": args.n_pca_features,
            "n_pls_hoop": args.n_pls_hoop,
            "n_pls_shot_grid": n_pls_shot_grid,
            "bw_quantile": args.bw_quantile,
            "alpha": args.alpha,
            "weight_dribble": args.weight_dribble,
            "weight_move": args.weight_move,
            "weight_sprint": args.weight_sprint,
            "weight_temperature": args.weight_temperature,
            "min_frame_weight": args.min_frame_weight,
            "base_submission": args.base_submission,
            "blend_weights": blend_weights,
            "skip_submissions": args.skip_submissions,
        },
        "shot7m2_stats": shot_data["stats"],
        "results": all_results,
        "best_config_name": best_name,
        "best_n_pls_shot": best_n_pls,
        "submissions_created": submissions_created,
    }
    run_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# SHOT7M2 Weighted Free-Throw-Like Transfer Run",
        "",
        f"- Timestamp: `{ts}`",
        f"- Command: `{payload['command']}`",
        "",
        "## SHOT7M2 Stats",
    ]
    for k, v in shot_data["stats"].items():
        lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "## LOO Results",
    ]
    for cfg, vals in all_results.items():
        lines.append(
            "- "
            + cfg
            + f": angle={vals['angle']}, depth={vals['depth']}, "
            + f"left_right={vals['left_right']}, mean={vals['mean']}, "
            + f"delta_pct={vals['delta_pct']}"
        )
    lines += [
        "",
        f"## Best Config",
        f"- name: `{best_name}`",
        f"- n_pls_shot: `{best_n_pls}`",
        "",
        "## Submissions",
    ]
    if submissions_created:
        for row in submissions_created:
            lines.append(f"- {row}")
    else:
        lines.append("- None (`--skip-submissions` used)")
    run_md.write_text("\n".join(lines) + "\n")

    print("\nArtifacts:")
    print(f"  {run_json}")
    print(f"  {run_md}")
    print("\nDone.")


if __name__ == "__main__":
    main()
