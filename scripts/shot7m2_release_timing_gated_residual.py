"""
SHOT7M2 Release Timing Transfer + Gated Residual

Goal:
- Use SHOT7M2 to learn a release-pose prototype.
- Use that prototype to infer per-shot release timing in competition data.
- Build conservative release-timing features.
- Apply no-harm gated residual correction on top of hoop-only baseline.

Validation:
- Honest per-player LOO.
- Pilot first (scale=1), then full (scale=8) with only --scale changed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SHOT7M2_DIR = PROJECT_DIR / "external_data" / "shot7m2_sample"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP = np.array([5.25, -25.0, 10.0], dtype=float)

FEET_TO_METERS = 0.3048
DT = 1.0 / 60.0

# 14-joint mapping for SHOT7M2 transfer
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

RELEASE_FINGERS = [
    "right_second_finger_distal",
    "right_third_finger_distal",
    "right_fourth_finger_distal",
]


@dataclass
class ReleasePrototype:
    centroid: np.ndarray
    scale: np.ndarray
    pose_count: int


def parse_triplet_int(s: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 ints, got {s}")
    return vals[0], vals[1], vals[2]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHOT7M2 release timing gated residual")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260215)
    p.add_argument("--axis-perm", type=str, default="0,2,1")
    p.add_argument("--axis-signs", type=str, default="1,1,1")
    p.add_argument("--shot-threshold", type=float, default=0.3)
    p.add_argument("--context-frames", type=int, default=30)
    p.add_argument("--shot-segments-per-scale", type=int, default=400)
    p.add_argument("--search-start", type=int, default=80)
    p.add_argument("--search-end", type=int, default=200)
    p.add_argument("--w-phys-release", type=float, default=0.7)
    p.add_argument("--w-shot-release", type=float, default=0.3)
    p.add_argument("--n-pls-hoop", type=int, default=15)
    p.add_argument("--bw-quantile", type=float, default=0.45)
    p.add_argument("--alpha-base", type=float, default=10.0)
    p.add_argument("--alpha-resid", type=float, default=20.0)
    p.add_argument("--gate-quantile", type=float, default=0.45)
    p.add_argument("--lambda-grid", type=str, default="0.0,0.02,0.05,0.08,0.10")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_array_string(s: str) -> np.ndarray:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    out = np.zeros(240, dtype=float)
    for i in range(min(len(arr), 240)):
        v = arr[i]
        out[i] = 0.0 if v is None else float(v)
    return out


def parse_json_safe(s: str, frame: int) -> float:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    v = arr[frame]
    if v is None:
        for o in range(1, 10):
            if frame - o >= 0 and arr[frame - o] is not None:
                return float(arr[frame - o])
            if frame + o < len(arr) and arr[frame + o] is not None:
                return float(arr[frame + o])
        return 0.0
    return float(v)


def apply_axis_transform(
    positions: np.ndarray,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> np.ndarray:
    x = positions[:, perm]
    return x * np.asarray(signs, dtype=float)[np.newaxis, :]


def normalize_pose_14(pose_14: np.ndarray) -> np.ndarray:
    """
    pose_14: (14,3)
    """
    pelvis_idx = MAPPED_JOINTS.index("mid_hip")
    neck_idx = MAPPED_JOINTS.index("neck")
    centered = pose_14 - pose_14[pelvis_idx : pelvis_idx + 1]
    torso = np.linalg.norm(centered[neck_idx] - centered[pelvis_idx])
    torso = max(torso, 1e-6)
    return centered / torso


def safe_savgol(x: np.ndarray, window: int, polyorder: int, **kwargs) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def detect_release_frame_physics(
    wrist: np.ndarray,
    fingers: list[np.ndarray],
    search_start: int,
    search_end: int,
) -> int:
    """
    wrist: (240,3), fingers: list[(240,3)]
    """
    wrist_traj = wrist.copy()
    for ax in range(3):
        vals = wrist_traj[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 153
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, ax] = vals

    wrist_z = safe_savgol(wrist_traj[:, 2], 11, 3)
    wrist_peak = search_start + np.argmax(wrist_z[search_start:search_end])

    if fingers:
        ft_center = np.nanmean(fingers, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()

    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    vel = np.zeros_like(ball * FEET_TO_METERS)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_METERS, 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)

    s = max(search_start, wrist_peak - 40)
    e = min(wrist_peak + 5, search_end)
    if e <= s:
        return int(np.clip(wrist_peak, search_start, search_end))
    return int(np.clip(s + np.argmax(speed[s:e]), search_start, search_end))


def build_shot7m2_release_prototype(
    args: argparse.Namespace,
) -> tuple[ReleasePrototype, dict]:
    poses = np.load(SHOT7M2_DIR / "train" / "train_dictionary_poses.npy", allow_pickle=True).item()
    actions = np.load(SHOT7M2_DIR / "train" / "train_dictionary_actions.npy", allow_pickle=True).item()

    labels = actions["label_array"]
    vocab = actions["vocabulary"]
    fnm = actions["frame_number_map"]
    shoot_idx = vocab.index("action_Shoot")
    dribble_idx = vocab.index("action_Dribble")
    move_idx = vocab.index("action_Move")
    sprint_idx = vocab.index("action_Sprint")

    release_poses = []
    segment_weights = []
    segment_lengths = []

    for key, seq in poses["sequences"]["keypoints"].items():
        start, end = fnm[key]
        shoot = labels[shoot_idx, start:end]
        drib = labels[dribble_idx, start:end]
        move = labels[move_idx, start:end]
        sprint = labels[sprint_idx, start:end]
        ep = seq[:, 0, SHOT7M2_INDICES, :]  # (1800,14,3)

        mask = shoot > args.shot_threshold
        if not np.any(mask):
            continue
        transitions = np.diff(mask.astype(int))
        seg_starts = np.where(transitions == 1)[0] + 1
        seg_ends = np.where(transitions == -1)[0] + 1
        if mask[0]:
            seg_starts = np.r_[0, seg_starts]
        if mask[-1]:
            seg_ends = np.r_[seg_ends, len(mask)]

        rw_idx = MAPPED_JOINTS.index("right_wrist")
        wrist = ep[:, rw_idx, :]
        vel = np.zeros_like(wrist)
        for ax in range(3):
            vel[:, ax] = safe_savgol(wrist[:, ax], 9, 3, deriv=1, delta=DT)
        speed = np.linalg.norm(vel, axis=1)

        for s, e in zip(seg_starts, seg_ends):
            if e - s < 3:
                continue
            rel = s + int(np.argmax(speed[s:e]))
            pose = normalize_pose_14(ep[rel])

            c0 = max(0, rel - args.context_frames)
            c1 = min(len(shoot), rel + args.context_frames + 1)
            d_mean = float(np.clip(np.mean(drib[c0:c1]), 0.0, 1.0))
            m_mean = float(np.clip(np.mean(move[c0:c1]), 0.0, 1.0))
            s_mean = float(np.clip(np.mean(sprint[c0:c1]), 0.0, 1.0))
            mobility = d_mean + m_mean + s_mean
            ft_like = math.exp(-mobility / 0.3)
            conf = max(float(np.max(shoot[s:e])), 0.0)
            weight = conf * (0.05 + 0.95 * ft_like)

            release_poses.append(pose.reshape(-1))
            segment_weights.append(weight)
            segment_lengths.append(int(e - s))

    release_poses = np.asarray(release_poses, dtype=float)
    segment_weights = np.asarray(segment_weights, dtype=float)

    keep_n = min(len(release_poses), args.scale * args.shot_segments_per_scale)
    keep_idx = np.argsort(-segment_weights)[:keep_n]
    kept = release_poses[keep_idx]

    centroid = kept.mean(axis=0)
    scale = kept.std(axis=0)
    scale = np.maximum(scale, 1e-6)

    prototype = ReleasePrototype(centroid=centroid, scale=scale, pose_count=len(kept))
    stats = {
        "total_release_poses": int(len(release_poses)),
        "kept_release_poses": int(len(kept)),
        "kept_ratio": float(len(kept) / max(len(release_poses), 1)),
        "kept_weight_mean": float(segment_weights[keep_idx].mean()) if len(kept) else 0.0,
        "segment_len_mean": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
        "segment_len_median": float(np.median(segment_lengths)) if segment_lengths else 0.0,
    }
    return prototype, stats


def infer_release_frame_from_prototype(
    seq_14: np.ndarray,
    proto: ReleasePrototype,
    search_start: int,
    search_end: int,
) -> tuple[int, float]:
    best_f = search_start
    best_d = 1e18
    for f in range(search_start, search_end + 1):
        pose = normalize_pose_14(seq_14[f]).reshape(-1)
        d = np.linalg.norm((pose - proto.centroid) / proto.scale)
        if d < best_d:
            best_d = d
            best_f = f
    return int(best_f), float(best_d)


def build_release_features_for_df(
    df: pd.DataFrame,
    proto: ReleasePrototype,
    axis_perm: tuple[int, int, int],
    axis_signs: tuple[int, int, int],
    search_start: int,
    search_end: int,
    w_phys: float,
    w_shot: float,
) -> dict[str, np.ndarray]:
    n = len(df)
    out = {
        "rf_phys": np.zeros(n, dtype=float),
        "rf_shot": np.zeros(n, dtype=float),
        "rf_blend": np.zeros(n, dtype=float),
        "dist_shot": np.zeros(n, dtype=float),
        "rw_h_blend": np.zeros(n, dtype=float),
        "lw_h_blend": np.zeros(n, dtype=float),
        "r_ext_blend": np.zeros(n, dtype=float),
        "l_ext_blend": np.zeros(n, dtype=float),
        "arm_asym_blend": np.zeros(n, dtype=float),
        "rw_speed_blend": np.zeros(n, dtype=float),
        "lw_speed_blend": np.zeros(n, dtype=float),
    }

    for i, row in df.iterrows():
        # Build 14-joint sequence
        seq_14 = np.zeros((240, 14, 3), dtype=float)
        for j_idx, name in enumerate(MAPPED_JOINTS):
            x = parse_array_string(row[f"{name}_x"])
            y = parse_array_string(row[f"{name}_y"])
            z = parse_array_string(row[f"{name}_z"])
            p = np.stack([x, y, z], axis=1)  # (240,3)
            p = apply_axis_transform(p, axis_perm, axis_signs)
            seq_14[:, j_idx, :] = p

        # Physics release from wrist/fingers in native coordinates
        wrist = np.stack(
            [
                parse_array_string(row["right_wrist_x"]),
                parse_array_string(row["right_wrist_y"]),
                parse_array_string(row["right_wrist_z"]),
            ],
            axis=1,
        )
        fingers = []
        for name in RELEASE_FINGERS:
            x_col = f"{name}_x"
            y_col = f"{name}_y"
            z_col = f"{name}_z"
            if x_col in row.index and y_col in row.index and z_col in row.index:
                fingers.append(
                    np.stack(
                        [
                            parse_array_string(row[x_col]),
                            parse_array_string(row[y_col]),
                            parse_array_string(row[z_col]),
                        ],
                        axis=1,
                    )
                )
        rf_phys = detect_release_frame_physics(wrist, fingers, search_start, search_end)
        rf_shot, dist_shot = infer_release_frame_from_prototype(seq_14, proto, search_start, search_end)
        rf_blend = int(np.clip(round(w_phys * rf_phys + w_shot * rf_shot), search_start, search_end))

        # Normalized kinematics at blended frame
        norm_seq = np.zeros_like(seq_14)
        for f in range(240):
            norm_seq[f] = normalize_pose_14(seq_14[f])
        rw_idx = MAPPED_JOINTS.index("right_wrist")
        lw_idx = MAPPED_JOINTS.index("left_wrist")
        neck_idx = MAPPED_JOINTS.index("neck")
        rs_idx = MAPPED_JOINTS.index("right_shoulder")
        ls_idx = MAPPED_JOINTS.index("left_shoulder")
        re_idx = MAPPED_JOINTS.index("right_elbow")
        le_idx = MAPPED_JOINTS.index("left_elbow")

        out["rf_phys"][i] = rf_phys
        out["rf_shot"][i] = rf_shot
        out["rf_blend"][i] = rf_blend
        out["dist_shot"][i] = dist_shot
        out["rw_h_blend"][i] = norm_seq[rf_blend, rw_idx, 1] - norm_seq[rf_blend, neck_idx, 1]
        out["lw_h_blend"][i] = norm_seq[rf_blend, lw_idx, 1] - norm_seq[rf_blend, neck_idx, 1]
        out["r_ext_blend"][i] = np.linalg.norm(norm_seq[rf_blend, rw_idx] - norm_seq[rf_blend, rs_idx])
        out["l_ext_blend"][i] = np.linalg.norm(norm_seq[rf_blend, lw_idx] - norm_seq[rf_blend, ls_idx])
        r_upper = np.linalg.norm(norm_seq[rf_blend, rs_idx] - norm_seq[rf_blend, re_idx])
        l_upper = np.linalg.norm(norm_seq[rf_blend, ls_idx] - norm_seq[rf_blend, le_idx])
        out["arm_asym_blend"][i] = r_upper - l_upper

        # speed at blend frame from normalized trajectory
        rw_vel = np.gradient(norm_seq[:, rw_idx, :], DT, axis=0)
        lw_vel = np.gradient(norm_seq[:, lw_idx, :], DT, axis=0)
        out["rw_speed_blend"][i] = np.linalg.norm(rw_vel[rf_blend])
        out["lw_speed_blend"][i] = np.linalg.norm(lw_vel[rf_blend])

    return out


def build_release_matrix(
    rel_feats: dict[str, np.ndarray],
    target_frame: int,
) -> np.ndarray:
    rf_phys = rel_feats["rf_phys"]
    rf_shot = rel_feats["rf_shot"]
    rf_blend = rel_feats["rf_blend"]

    mats = [
        rf_phys / 240.0,
        rf_shot / 240.0,
        rf_blend / 240.0,
        (rf_phys - rf_shot) / 240.0,
        (target_frame - rf_phys) / 240.0,
        (target_frame - rf_shot) / 240.0,
        (target_frame - rf_blend) / 240.0,
        rel_feats["dist_shot"],
        rel_feats["rw_h_blend"],
        rel_feats["lw_h_blend"],
        rel_feats["r_ext_blend"],
        rel_feats["l_ext_blend"],
        rel_feats["arm_asym_blend"],
        rel_feats["rw_speed_blend"],
        rel_feats["lw_speed_blend"],
    ]
    return np.vstack(mats).T.astype(float)


def build_hoop_features(df: pd.DataFrame, frame: int) -> np.ndarray:
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
    return hoop_rel.reshape(n, -1)


def compute_base_correction_components(
    x_hoop: np.ndarray,
    x_rel: np.ndarray,
    y: np.ndarray,
    pids: np.ndarray,
    n_pls_hoop: int,
    bw_quantile: float,
    alpha_base: float,
    alpha_resid: float,
    gate_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    base = np.zeros(n, dtype=float)
    corr = np.zeros(n, dtype=float)
    gate = np.zeros(n, dtype=float)

    for pid in sorted(np.unique(pids)):
        m = pids == pid
        idx = np.where(m)[0]
        xh = x_hoop[m]
        xr = x_rel[m]
        yp = y[m]
        n_p = len(idx)

        for i_local, i_global in enumerate(idx):
            loo = np.ones(n_p, dtype=bool)
            loo[i_local] = False
            xh_tr, xh_te = xh[loo], xh[~loo]
            xr_tr, xr_te = xr[loo], xr[~loo]
            y_tr = yp[loo]

            # Base model (hoop-only)
            sh = StandardScaler()
            xh_tr_s = sh.fit_transform(xh_tr)
            xh_te_s = sh.transform(xh_te)
            nph = min(n_pls_hoop, xh_tr_s.shape[1], xh_tr_s.shape[0] - 1)
            if nph > 0:
                plsh = PLSRegression(n_components=nph)
                plsh.fit(xh_tr_s, y_tr)
                z_tr = plsh.transform(xh_tr_s)
                z_te = plsh.transform(xh_te_s)
                xb_tr = np.hstack([xh_tr_s, z_tr])
                xb_te = np.hstack([xh_te_s, z_te])
            else:
                xb_tr, xb_te = xh_tr_s, xh_te_s

            d = np.linalg.norm(xb_tr - xb_te, axis=1)
            bw = np.quantile(d, bw_quantile)
            w = np.exp(-0.5 * (d / max(bw, 1e-8)) ** 2)
            wm = np.diag(np.sqrt(w))
            ridge_base = Ridge(alpha=alpha_base)
            ridge_base.fit(wm @ xb_tr, wm @ y_tr)
            base_pred_te = ridge_base.predict(xb_te)[0]
            base_pred_tr = ridge_base.predict(xb_tr)

            # Residual model
            resid_tr = y_tr - base_pred_tr
            sr = StandardScaler()
            xr_tr_s = sr.fit_transform(xr_tr)
            xr_te_s = sr.transform(xr_te)
            ridge_resid = Ridge(alpha=alpha_resid)
            ridge_resid.fit(xr_tr_s, resid_tr)
            corr_pred_te = ridge_resid.predict(xr_te_s)[0]

            # Gate by proximity in release-feature space
            dr = np.linalg.norm(xr_tr_s - xr_te_s, axis=1)
            bw_r = np.quantile(dr, gate_quantile)
            dmin = float(np.min(dr))
            g = math.exp(-0.5 * (dmin / max(bw_r, 1e-8)) ** 2)

            base[i_global] = base_pred_te
            corr[i_global] = corr_pred_te
            gate[i_global] = g

    return base, corr, gate


def evaluate_lambdas(
    base: np.ndarray,
    corr: np.ndarray,
    gate: np.ndarray,
    y: np.ndarray,
    lambdas: list[float],
) -> dict[str, float]:
    out = {}
    for lam in lambdas:
        pred = base + lam * gate * corr
        out[f"{lam:.12f}"] = float(np.mean((pred - y) ** 2))
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    axis_perm = parse_triplet_int(args.axis_perm)
    axis_signs = parse_triplet_int(args.axis_signs)
    lambdas = parse_float_list(args.lambda_grid)

    print("=" * 80)
    print("SHOT7M2 RELEASE TIMING GATED RESIDUAL")
    print("=" * 80)
    print(f"scale={args.scale}, seed={args.seed}")
    print(f"axis_perm={axis_perm}, axis_signs={axis_signs}")
    print(f"lambdas={lambdas}")

    t0 = time.time()
    proto, shot_stats = build_shot7m2_release_prototype(args)
    print(f"SHOT7M2 prototype poses kept: {proto.pose_count}")
    print(f"SHOT7M2 kept_ratio: {shot_stats['kept_ratio']:.12f}")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_pids = train_df["participant_id"].values

    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_scaled = {
        t: scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel() for t in TARGETS
    }

    print("Building release timing features from prototype + physics...")
    rel_train = build_release_features_for_df(
        train_df,
        proto,
        axis_perm,
        axis_signs,
        args.search_start,
        args.search_end,
        args.w_phys_release,
        args.w_shot_release,
    )
    rel_test = build_release_features_for_df(
        test_df,
        proto,
        axis_perm,
        axis_signs,
        args.search_start,
        args.search_end,
        args.w_phys_release,
        args.w_shot_release,
    )

    results = {}
    best_lam_per_target = {}
    best_mse_per_target = {}

    for t in TARGETS:
        frame = TARGET_FRAMES[t]
        print(f"\nTarget {t} (frame {frame})")
        xh = build_hoop_features(train_df, frame)
        xr = build_release_matrix(rel_train, frame)

        base, corr, gate = compute_base_correction_components(
            x_hoop=xh,
            x_rel=xr,
            y=y_scaled[t],
            pids=train_pids,
            n_pls_hoop=args.n_pls_hoop,
            bw_quantile=args.bw_quantile,
            alpha_base=args.alpha_base,
            alpha_resid=args.alpha_resid,
            gate_quantile=args.gate_quantile,
        )

        lambda_mse = evaluate_lambdas(base, corr, gate, y_scaled[t], lambdas)
        baseline_key = f"{0.0:.12f}"
        best_key = min(lambda_mse, key=lambda k: lambda_mse[k])
        best_lam = float(best_key)
        best_mse = float(lambda_mse[best_key])
        base_mse = float(lambda_mse[baseline_key])
        delta_pct = (best_mse - base_mse) / base_mse * 100.0
        print(f"  baseline_mse={base_mse:.12f}")
        print(f"  best_lambda={best_lam:.12f}")
        print(f"  best_mse={best_mse:.12f}")
        print(f"  delta_pct={delta_pct:+.12f}")

        best_lam_per_target[t] = best_lam
        best_mse_per_target[t] = best_mse
        results[t] = {
            "baseline_mse": base_mse,
            "lambda_mse": lambda_mse,
            "best_lambda": best_lam,
            "best_mse": best_mse,
            "best_delta_pct": float(delta_pct),
            "gate_mean": float(np.mean(gate)),
            "gate_median": float(np.median(gate)),
            "corr_mean_abs": float(np.mean(np.abs(corr))),
        }

    baseline_mean = float(
        np.mean([results[t]["baseline_mse"] for t in TARGETS])
    )
    best_mean = float(np.mean([best_mse_per_target[t] for t in TARGETS]))
    mean_delta_pct = (best_mean - baseline_mean) / baseline_mean * 100.0

    print("\nSummary")
    print(f"  baseline_mean={baseline_mean:.12f}")
    print(f"  best_mean={best_mean:.12f}")
    print(f"  mean_delta_pct={mean_delta_pct:+.12f}")
    print(f"  best_lambda_per_target={best_lam_per_target}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json = OUTPUT_DIR / f"shot7m2_release_timing_gated_residual_run_{ts}.json"
    run_md = OUTPUT_DIR / f"shot7m2_release_timing_gated_residual_details_{ts}.md"

    payload = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "config": {
            "scale": args.scale,
            "seed": args.seed,
            "axis_perm": axis_perm,
            "axis_signs": axis_signs,
            "shot_threshold": args.shot_threshold,
            "context_frames": args.context_frames,
            "shot_segments_per_scale": args.shot_segments_per_scale,
            "search_start": args.search_start,
            "search_end": args.search_end,
            "w_phys_release": args.w_phys_release,
            "w_shot_release": args.w_shot_release,
            "n_pls_hoop": args.n_pls_hoop,
            "bw_quantile": args.bw_quantile,
            "alpha_base": args.alpha_base,
            "alpha_resid": args.alpha_resid,
            "gate_quantile": args.gate_quantile,
            "lambda_grid": lambdas,
        },
        "shot7m2_stats": shot_stats,
        "prototype_pose_count": proto.pose_count,
        "per_target_results": results,
        "summary": {
            "baseline_mean_mse": baseline_mean,
            "best_mean_mse": best_mean,
            "mean_delta_pct": mean_delta_pct,
            "best_lambda_per_target": best_lam_per_target,
            "total_runtime_s": float(time.time() - t0),
        },
    }
    run_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# SHOT7M2 Release Timing Gated Residual",
        "",
        f"- Timestamp: `{ts}`",
        f"- Command: `{payload['command']}`",
        "",
        "## SHOT7M2 Prototype Stats",
    ]
    for k, v in shot_stats.items():
        lines.append(f"- {k}: `{v}`")
    lines += [
        f"- prototype_pose_count: `{proto.pose_count}`",
        "",
        "## Per-target Results",
    ]
    for t in TARGETS:
        r = results[t]
        lines.append(f"- {t}: baseline_mse={r['baseline_mse']}, best_lambda={r['best_lambda']}, best_mse={r['best_mse']}, best_delta_pct={r['best_delta_pct']}")
        lines.append(f"  lambda_mse={r['lambda_mse']}")
        lines.append(f"  gate_mean={r['gate_mean']}, gate_median={r['gate_median']}, corr_mean_abs={r['corr_mean_abs']}")
    lines += [
        "",
        "## Summary",
        f"- baseline_mean_mse: `{baseline_mean}`",
        f"- best_mean_mse: `{best_mean}`",
        f"- mean_delta_pct: `{mean_delta_pct}`",
        f"- best_lambda_per_target: `{best_lam_per_target}`",
    ]
    run_md.write_text("\n".join(lines) + "\n")

    print("\nArtifacts:")
    print(f"  {run_json}")
    print(f"  {run_md}")
    print("Done.")


if __name__ == "__main__":
    main()
