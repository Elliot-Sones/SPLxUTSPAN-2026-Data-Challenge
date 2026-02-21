"""
P5-targeted depth ablations with conservative anchor blending.

Ablations:
1) leakage_only
2) physics_residual
3) physics_residual_ghost_dtw

Design goals:
- Target the depth bottleneck (especially Player 5) while keeping angle/LR untouched.
- Keep final leaderboard risk low via small depth-only injections into Sub 2716.
- Produce exact reproducibility artifacts (JSON + Markdown details).

Usage:
  uv run python scripts/p5_physics_leakage_ghost_dtw_ablation.py --scale 1
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

DEPTH_COL = "scaled_depth"
ANCHOR_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGETS_RAW = ["angle", "depth", "left_right"]

REQUIRED_JOINTS = [
    "right_wrist",
    "right_elbow",
    "right_hip",
    "right_knee",
    "left_hip",
    "left_knee",
    "mid_hip",
]


@dataclass
class ShotFeatures:
    leak: np.ndarray
    phys: np.ndarray
    ghost_seq: np.ndarray  # shape: (n_windows, 4) for chain speeds
    pid: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P5 physics/leakage/ghost-DTW depth ablations")
    p.add_argument("--scale", type=int, default=1, help="Scaling knob. Use 1 for pilot; increase for heavier models.")
    p.add_argument("--anchor-sub", type=int, default=2716, help="Anchor submission number.")
    p.add_argument(
        "--weights",
        type=str,
        default="0.01,0.02,0.03",
        help="Comma-separated depth blend weights into anchor.",
    )
    p.add_argument("--seed", type=int, default=20260217, help="Random seed.")
    p.add_argument("--run-tag", type=str, default="", help="Optional run tag.")
    p.add_argument("--max-iter-base", type=int, default=120, help="Base boosting iterations multiplied by scale.")
    return p.parse_args()


def parse_weight_list(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No blend weights parsed")
    for w in values:
        if not (0.0 < w < 1.0):
            raise ValueError(f"Blend weight must be in (0,1), got {w}")
    return values


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for path in existing:
                parts = path.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_array_string(value: str) -> np.ndarray:
    if pd.isna(value):
        return np.full(240, np.nan, dtype=np.float32)
    cleaned = str(value).replace("nan", "null")
    return np.asarray(json.loads(cleaned), dtype=np.float32)


def load_joint_timeseries(row: pd.Series, joints: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for joint in joints:
        xyz = []
        for axis in ("x", "y", "z"):
            col = f"{joint}_{axis}"
            xyz.append(parse_array_string(row[col]))
        out[joint] = np.stack(xyz, axis=1)  # (240, 3)
    return out


def safe_gradient(values: np.ndarray, fps: float = 60.0) -> np.ndarray:
    return np.gradient(values, axis=0) * fps


def compute_chain_speed_sequence(
    joint_ts: dict[str, np.ndarray],
    frame_start: int,
    frame_end: int,
    n_windows: int,
) -> np.ndarray:
    chain = ["right_knee", "right_hip", "right_elbow", "right_wrist"]
    seq = []
    for joint in chain:
        vel = safe_gradient(joint_ts[joint][frame_start:frame_end, :])
        speed = np.sqrt(np.sum(vel * vel, axis=1))
        speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
        speed = speed - float(np.mean(speed))
        std = float(np.std(speed))
        if std > 1e-8:
            speed = speed / std
        seq.append(speed)
    # (T, 4)
    seq_t = np.stack(seq, axis=1)

    t = seq_t.shape[0]
    ws = max(1, t // n_windows)
    windows = []
    for w in range(n_windows):
        a = w * ws
        b = t if w == n_windows - 1 else min(t, (w + 1) * ws)
        if b <= a:
            windows.append(np.zeros(4, dtype=np.float32))
        else:
            windows.append(seq_t[a:b].mean(axis=0))
    return np.asarray(windows, dtype=np.float32)


def extract_features_for_row(
    row: pd.Series,
    frame_start: int,
    frame_end: int,
    n_windows: int,
) -> ShotFeatures:
    joint_ts = load_joint_timeseries(row, REQUIRED_JOINTS)

    wrist = joint_ts["right_wrist"]
    elbow = joint_ts["right_elbow"]
    hip = joint_ts["right_hip"]
    knee = joint_ts["right_knee"]
    mid_hip = joint_ts["mid_hip"]

    wrist_vel = safe_gradient(wrist)
    elbow_vel = safe_gradient(elbow)
    hip_vel = safe_gradient(hip)
    knee_vel = safe_gradient(knee)

    fw_a = frame_start
    fw_b = frame_end

    wrist_v = wrist_vel[fw_a:fw_b]
    elbow_v = elbow_vel[fw_a:fw_b]
    hip_v = hip_vel[fw_a:fw_b]
    knee_v = knee_vel[fw_a:fw_b]

    wrist_vx = wrist_v[:, 0]
    wrist_vy = wrist_v[:, 1]
    wrist_vz = wrist_v[:, 2]

    # Release proxy: frame of max upward wrist velocity.
    rel_local = int(np.argmax(wrist_vz))
    rel_frame = fw_a + rel_local

    def _energy(block: np.ndarray) -> float:
        speed_sq = np.sum(block * block, axis=1)
        return float(np.sum(speed_sq))

    prop_a = fw_a
    prop_b = max(prop_a + 1, rel_frame + 1)

    knee_e = _energy(knee_vel[prop_a:prop_b])
    hip_e = _energy(hip_vel[prop_a:prop_b])
    elbow_e = _energy(elbow_vel[prop_a:prop_b])
    wrist_e = _energy(wrist_vel[prop_a:prop_b])
    total_e = knee_e + hip_e + elbow_e + wrist_e + 1e-9

    rel_v = wrist_vel[rel_frame]
    rel_speed = float(np.linalg.norm(rel_v))
    rel_horiz = float(np.sqrt(rel_v[0] * rel_v[0] + rel_v[1] * rel_v[1]))
    rel_elev = float(np.degrees(np.arctan2(rel_v[2], max(rel_horiz, 1e-8))))
    rel_azi = float(np.degrees(np.arctan2(rel_v[0], max(abs(rel_v[1]), 1e-8))))

    lateral_prop = float(np.sum(wrist_vx * wrist_vx) / (np.sum(wrist_vx * wrist_vx + wrist_vy * wrist_vy + wrist_vz * wrist_vz) + 1e-9))
    elbow_lateral_prop = float(
        np.sum(elbow_v[:, 0] * elbow_v[:, 0])
        / (np.sum(elbow_v[:, 0] * elbow_v[:, 0] + elbow_v[:, 1] * elbow_v[:, 1] + elbow_v[:, 2] * elbow_v[:, 2]) + 1e-9)
    )

    knee_speed = np.sqrt(np.sum(knee_v * knee_v, axis=1))
    hip_speed = np.sqrt(np.sum(hip_v * hip_v, axis=1))
    elbow_speed = np.sqrt(np.sum(elbow_v * elbow_v, axis=1))
    wrist_speed = np.sqrt(np.sum(wrist_v * wrist_v, axis=1))

    peak_knee = fw_a + int(np.argmax(knee_speed))
    peak_hip = fw_a + int(np.argmax(hip_speed))
    peak_elbow = fw_a + int(np.argmax(elbow_speed))
    peak_wrist = fw_a + int(np.argmax(wrist_speed))

    chain_order = float(
        (peak_knee <= peak_hip)
        + (peak_hip <= peak_elbow)
        + (peak_elbow <= peak_wrist)
    ) / 3.0

    rel_pos = wrist[rel_frame] - mid_hip[rel_frame]
    is_p5 = 1.0 if int(row["participant_id"]) == 5 else 0.0

    leak_features = np.array(
        [
            rel_v[0],
            rel_v[1],
            rel_v[2],
            rel_speed,
            rel_horiz,
            rel_elev,
            rel_azi,
            lateral_prop,
            elbow_lateral_prop,
            abs(rel_v[0]) / (abs(rel_v[2]) + 1e-8),
            knee_e / (wrist_e + 1e-9),
            hip_e / (wrist_e + 1e-9),
            elbow_e / (wrist_e + 1e-9),
            knee_e / total_e,
            hip_e / total_e,
            elbow_e / total_e,
            wrist_e / total_e,
            float(peak_hip - peak_knee),
            float(peak_elbow - peak_hip),
            float(peak_wrist - peak_elbow),
            chain_order,
            rel_pos[0],
            rel_pos[1],
            rel_pos[2],
            is_p5,
            rel_azi * is_p5,
            lateral_prop * is_p5,
            elbow_lateral_prop * is_p5,
            (abs(rel_v[0]) / (abs(rel_v[2]) + 1e-8)) * is_p5,
        ],
        dtype=np.float32,
    )

    physics_features = np.array(
        [
            rel_speed,
            rel_speed * rel_speed,
            rel_elev,
            rel_azi,
            rel_horiz,
            rel_v[2],
            rel_pos[2],
            rel_pos[1],
            lateral_prop,
            chain_order,
        ],
        dtype=np.float32,
    )

    ghost_seq = compute_chain_speed_sequence(
        joint_ts=joint_ts,
        frame_start=frame_start,
        frame_end=frame_end,
        n_windows=n_windows,
    )

    return ShotFeatures(
        leak=leak_features,
        phys=physics_features,
        ghost_seq=ghost_seq,
        pid=int(row["participant_id"]),
    )


def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    # seq: (T, D)
    t1, d1 = seq_a.shape
    t2, d2 = seq_b.shape
    if d1 != d2:
        raise ValueError("DTW dimensionality mismatch")

    inf = 1e18
    dp = np.full((t1 + 1, t2 + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, t1 + 1):
        ai = seq_a[i - 1]
        for j in range(1, t2 + 1):
            bj = seq_b[j - 1]
            cost = float(np.linalg.norm(ai - bj))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[t1, t2] / max(t1, t2))


def build_ghost_templates(
    ghost_train: np.ndarray,
    y_depth_train: np.ndarray,
    pids_train: np.ndarray,
) -> dict[tuple[int, str], np.ndarray]:
    """
    Build per-player and global templates over depth tertiles.
    Returns keys: (pid, bin_name), with pid in players and -1 for global.
    """
    templates: dict[tuple[int, str], np.ndarray] = {}
    bin_names = ["low", "mid", "high"]

    def _bins(y: np.ndarray) -> list[np.ndarray]:
        q1 = float(np.quantile(y, 1.0 / 3.0))
        q2 = float(np.quantile(y, 2.0 / 3.0))
        low = y <= q1
        mid = (y > q1) & (y <= q2)
        high = y > q2
        return [low, mid, high]

    # Global templates
    g_masks = _bins(y_depth_train)
    for name, mask in zip(bin_names, g_masks):
        if np.any(mask):
            templates[(-1, name)] = ghost_train[mask].mean(axis=0)

    # Per-player templates
    for pid in sorted(np.unique(pids_train).tolist()):
        pid_mask = pids_train == pid
        y_pid = y_depth_train[pid_mask]
        g_pid = ghost_train[pid_mask]
        if len(y_pid) < 3:
            continue
        pmasks = _bins(y_pid)
        for name, mask in zip(bin_names, pmasks):
            if np.any(mask):
                templates[(int(pid), name)] = g_pid[mask].mean(axis=0)

    return templates


def ghost_distance_features(
    ghost_seq: np.ndarray,
    pid: int,
    templates: dict[tuple[int, str], np.ndarray],
) -> np.ndarray:
    feats = []
    for key in [(pid, "low"), (pid, "mid"), (pid, "high"), (-1, "low"), (-1, "mid"), (-1, "high")]:
        if key in templates:
            feats.append(dtw_distance(ghost_seq, templates[key]))
        else:
            feats.append(0.0)
    # Derived contrasts
    feats.append(feats[2] - feats[0])  # player high-low
    feats.append(feats[5] - feats[3])  # global high-low
    return np.asarray(feats, dtype=np.float32)


def make_hgbr(scale: int, max_iter_base: int, seed: int) -> HistGradientBoostingRegressor:
    max_iter = int(max_iter_base * scale)
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_depth=3,
        min_samples_leaf=max(8, 20 // max(scale, 1)),
        l2_regularization=0.1,
        random_state=seed,
    )


def clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def per_player_oof_eval(
    method: str,
    leak: np.ndarray,
    phys: np.ndarray,
    ghost: np.ndarray,
    pids: np.ndarray,
    y_depth: np.ndarray,
    scale: int,
    max_iter_base: int,
    seed: int,
) -> np.ndarray:
    oof = np.zeros_like(y_depth)

    unique_pids = sorted(np.unique(pids).tolist())
    for pid in unique_pids:
        idx = np.where(pids == pid)[0]
        n = len(idx)
        if n < 5:
            # tiny fallback
            oof[idx] = float(np.mean(y_depth[idx]))
            continue

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_local, va_local in kf.split(np.arange(n)):
            tr_idx = idx[tr_local]
            va_idx = idx[va_local]

            x_leak_tr = leak[tr_idx]
            x_leak_va = leak[va_idx]
            x_phys_tr = phys[tr_idx]
            x_phys_va = phys[va_idx]
            y_tr = y_depth[tr_idx]

            if method == "leakage_only":
                model = make_hgbr(scale, max_iter_base, seed)
                model.fit(x_leak_tr, y_tr)
                pred = model.predict(x_leak_va)

            elif method == "physics_residual":
                phys_model = Ridge(alpha=1.0)
                phys_model.fit(x_phys_tr, y_tr)
                phys_tr = phys_model.predict(x_phys_tr)
                phys_va = phys_model.predict(x_phys_va)
                res_tr = y_tr - phys_tr

                x_res_tr = np.hstack([x_leak_tr, phys_tr.reshape(-1, 1)])
                x_res_va = np.hstack([x_leak_va, phys_va.reshape(-1, 1)])

                res_model = make_hgbr(scale, max_iter_base, seed)
                res_model.fit(x_res_tr, res_tr)
                pred = phys_va + res_model.predict(x_res_va)

            elif method == "physics_residual_ghost_dtw":
                templates = build_ghost_templates(
                    ghost_train=ghost[tr_idx],
                    y_depth_train=y_tr,
                    pids_train=pids[tr_idx],
                )
                g_tr = np.vstack([ghost_distance_features(ghost[i], int(pids[i]), templates) for i in tr_idx])
                g_va = np.vstack([ghost_distance_features(ghost[i], int(pids[i]), templates) for i in va_idx])

                phys_model = Ridge(alpha=1.0)
                phys_model.fit(x_phys_tr, y_tr)
                phys_tr = phys_model.predict(x_phys_tr)
                phys_va = phys_model.predict(x_phys_va)
                res_tr = y_tr - phys_tr

                x_res_tr = np.hstack([x_leak_tr, phys_tr.reshape(-1, 1), g_tr])
                x_res_va = np.hstack([x_leak_va, phys_va.reshape(-1, 1), g_va])

                res_model = make_hgbr(scale, max_iter_base, seed)
                res_model.fit(x_res_tr, res_tr)
                pred = phys_va + res_model.predict(x_res_va)

            else:
                raise ValueError(f"Unknown method: {method}")

            oof[va_idx] = clip01(pred)

    return oof


def per_player_fit_predict_test(
    method: str,
    leak_train: np.ndarray,
    phys_train: np.ndarray,
    ghost_train: np.ndarray,
    pids_train: np.ndarray,
    y_depth_train: np.ndarray,
    leak_test: np.ndarray,
    phys_test: np.ndarray,
    ghost_test: np.ndarray,
    pids_test: np.ndarray,
    scale: int,
    max_iter_base: int,
    seed: int,
) -> np.ndarray:
    preds = np.zeros(len(pids_test), dtype=np.float64)

    unique_pids = sorted(np.unique(pids_train).tolist())
    for pid in unique_pids:
        tr_idx = np.where(pids_train == pid)[0]
        te_idx = np.where(pids_test == pid)[0]
        if len(te_idx) == 0:
            continue

        x_leak_tr = leak_train[tr_idx]
        x_leak_te = leak_test[te_idx]
        x_phys_tr = phys_train[tr_idx]
        x_phys_te = phys_test[te_idx]
        y_tr = y_depth_train[tr_idx]

        if method == "leakage_only":
            model = make_hgbr(scale, max_iter_base, seed)
            model.fit(x_leak_tr, y_tr)
            pred = model.predict(x_leak_te)

        elif method == "physics_residual":
            phys_model = Ridge(alpha=1.0)
            phys_model.fit(x_phys_tr, y_tr)
            phys_tr = phys_model.predict(x_phys_tr)
            phys_te = phys_model.predict(x_phys_te)
            res_tr = y_tr - phys_tr

            x_res_tr = np.hstack([x_leak_tr, phys_tr.reshape(-1, 1)])
            x_res_te = np.hstack([x_leak_te, phys_te.reshape(-1, 1)])

            res_model = make_hgbr(scale, max_iter_base, seed)
            res_model.fit(x_res_tr, res_tr)
            pred = phys_te + res_model.predict(x_res_te)

        elif method == "physics_residual_ghost_dtw":
            templates = build_ghost_templates(
                ghost_train=ghost_train[tr_idx],
                y_depth_train=y_tr,
                pids_train=pids_train[tr_idx],
            )
            g_tr = np.vstack([ghost_distance_features(ghost_train[i], int(pids_train[i]), templates) for i in tr_idx])
            g_te = np.vstack([ghost_distance_features(ghost_test[i], int(pids_test[i]), templates) for i in te_idx])

            phys_model = Ridge(alpha=1.0)
            phys_model.fit(x_phys_tr, y_tr)
            phys_tr = phys_model.predict(x_phys_tr)
            phys_te = phys_model.predict(x_phys_te)
            res_tr = y_tr - phys_tr

            x_res_tr = np.hstack([x_leak_tr, phys_tr.reshape(-1, 1), g_tr])
            x_res_te = np.hstack([x_leak_te, phys_te.reshape(-1, 1), g_te])

            res_model = make_hgbr(scale, max_iter_base, seed)
            res_model.fit(x_res_tr, res_tr)
            pred = phys_te + res_model.predict(x_res_te)

        else:
            raise ValueError(f"Unknown method: {method}")

        preds[te_idx] = clip01(pred)

    return preds


def load_feature_matrices(scale: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    depth_scaler = joblib.load(DATA_DIR / "scaler_depth.pkl")
    y_depth_scaled = depth_scaler.transform(train_df[["depth"]]).ravel().astype(np.float64)

    frame_start = 90
    frame_end = 190
    n_windows = 6 * max(scale, 1)

    train_feats: list[ShotFeatures] = []
    test_feats: list[ShotFeatures] = []

    for row in train_df.itertuples(index=False):
        train_feats.append(
            extract_features_for_row(
                row=pd.Series(row._asdict()),
                frame_start=frame_start,
                frame_end=frame_end,
                n_windows=n_windows,
            )
        )

    for row in test_df.itertuples(index=False):
        test_feats.append(
            extract_features_for_row(
                row=pd.Series(row._asdict()),
                frame_start=frame_start,
                frame_end=frame_end,
                n_windows=n_windows,
            )
        )

    leak_train = np.vstack([f.leak for f in train_feats]).astype(np.float32)
    phys_train = np.vstack([f.phys for f in train_feats]).astype(np.float32)
    ghost_train = np.stack([f.ghost_seq for f in train_feats]).astype(np.float32)
    pids_train = np.asarray([f.pid for f in train_feats], dtype=np.int32)

    leak_test = np.vstack([f.leak for f in test_feats]).astype(np.float32)
    phys_test = np.vstack([f.phys for f in test_feats]).astype(np.float32)
    ghost_test = np.stack([f.ghost_seq for f in test_feats]).astype(np.float32)
    pids_test = np.asarray([f.pid for f in test_feats], dtype=np.int32)

    return (
        leak_train,
        phys_train,
        ghost_train,
        pids_train,
        y_depth_scaled,
        leak_test,
        phys_test,
        ghost_test,
        pids_test,
    )


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def per_player_mse(y_true: np.ndarray, y_pred: np.ndarray, pids: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for pid in sorted(np.unique(pids).tolist()):
        mask = pids == pid
        out[f"P{pid}"] = mse(y_true[mask], y_pred[mask])
    return out


def main() -> None:
    args = parse_args()
    t0 = time.time()

    np.random.seed(args.seed)

    if args.scale < 1:
        raise ValueError("scale must be >= 1")

    weights = parse_weight_list(args.weights)

    run_tag = args.run_tag.strip()
    if not run_tag:
        run_tag = f"p5_phy_leak_ghost_scale{args.scale}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    anchor_path = SUBMISSION_DIR / f"submission_{args.anchor_sub}.csv"
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor file missing: {anchor_path}")

    anchor_df = pd.read_csv(anchor_path)
    if not all(col in anchor_df.columns for col in ANCHOR_COLS):
        raise ValueError(f"Anchor missing columns: {ANCHOR_COLS}")

    print("=" * 88)
    print("P5 Physics-Leakage-GhostDTW Depth Ablations")
    print("=" * 88)
    print(f"scale={args.scale}")
    print(f"seed={args.seed}")
    print(f"anchor=submission_{args.anchor_sub}.csv")
    print(f"weights={weights}")

    print("\n[1/5] Building feature matrices...")
    (
        leak_train,
        phys_train,
        ghost_train,
        pids_train,
        y_depth,
        leak_test,
        phys_test,
        ghost_test,
        pids_test,
    ) = load_feature_matrices(scale=args.scale)

    print(f"train leak shape={leak_train.shape}, phys shape={phys_train.shape}, ghost shape={ghost_train.shape}")
    print(f"test  leak shape={leak_test.shape},  phys shape={phys_test.shape},  ghost shape={ghost_test.shape}")

    methods = [
        "leakage_only",
        "physics_residual",
        "physics_residual_ghost_dtw",
    ]

    print("\n[2/5] Running per-player OOF evaluation...")
    oof_map: dict[str, np.ndarray] = {}
    oof_metrics: dict[str, dict[str, float | dict[str, float]]] = {}

    for method in methods:
        oof = per_player_oof_eval(
            method=method,
            leak=leak_train,
            phys=phys_train,
            ghost=ghost_train,
            pids=pids_train,
            y_depth=y_depth,
            scale=args.scale,
            max_iter_base=args.max_iter_base,
            seed=args.seed,
        )
        oof_map[method] = oof
        oof_metrics[method] = {
            "oof_mse_depth": mse(y_depth, oof),
            "oof_per_player_mse_depth": per_player_mse(y_depth, oof, pids_train),
        }
        print(f"  {method:32s} oof_mse_depth={oof_metrics[method]['oof_mse_depth']:.15f}")

    print("\n[3/5] Fitting full per-player models and predicting test depth...")
    test_pred_map: dict[str, np.ndarray] = {}
    test_stats: dict[str, dict[str, float]] = {}

    anchor_depth = anchor_df[DEPTH_COL].to_numpy(dtype=np.float64)

    for method in methods:
        pred = per_player_fit_predict_test(
            method=method,
            leak_train=leak_train,
            phys_train=phys_train,
            ghost_train=ghost_train,
            pids_train=pids_train,
            y_depth_train=y_depth,
            leak_test=leak_test,
            phys_test=phys_test,
            ghost_test=ghost_test,
            pids_test=pids_test,
            scale=args.scale,
            max_iter_base=args.max_iter_base,
            seed=args.seed,
        )
        pred = clip01(pred)
        test_pred_map[method] = pred

        corr = float(np.corrcoef(pred, anchor_depth)[0, 1])
        rmsd = float(np.sqrt(np.mean((pred - anchor_depth) ** 2)))
        delta_mean = float(np.mean(pred - anchor_depth))
        delta_abs_mean = float(np.mean(np.abs(pred - anchor_depth)))

        test_stats[method] = {
            "corr_vs_anchor_depth": corr,
            "rmsd_vs_anchor_depth": rmsd,
            "mean_delta_vs_anchor_depth": delta_mean,
            "mean_abs_delta_vs_anchor_depth": delta_abs_mean,
        }

        print(
            f"  {method:32s} "
            f"corr={corr:.15f} rmsd={rmsd:.15f} mean_abs_delta={delta_abs_mean:.15f}"
        )

    print("\n[4/5] Writing depth-only blend submissions...")
    submission_records: list[dict[str, object]] = []

    for method in methods:
        pred_depth = test_pred_map[method]
        for w in weights:
            out_df = anchor_df.copy()
            out_df[DEPTH_COL] = clip01((1.0 - w) * anchor_depth + w * pred_depth)
            sub_num = get_next_submission_number()
            out_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            out_df.to_csv(out_path, index=False, float_format="%.15f")

            rmsd_vs_anchor = float(np.sqrt(np.mean((out_df[DEPTH_COL].to_numpy() - anchor_depth) ** 2)))
            changed_rows = int(np.sum(np.abs(out_df[DEPTH_COL].to_numpy() - anchor_depth) > 1e-12))

            rec = {
                "method": method,
                "blend_weight_depth": float(w),
                "submission_num": int(sub_num),
                "submission_file": str(out_path),
                "changed_rows_depth": int(changed_rows),
                "rmsd_depth_vs_anchor": rmsd_vs_anchor,
            }
            submission_records.append(rec)

            print(
                f"  Sub {sub_num}: method={method}, depth_weight={w:.15f}, "
                f"rmsd_depth_vs_anchor={rmsd_vs_anchor:.15f}, changed_rows={changed_rows}"
            )

    elapsed = time.time() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_json_path = OUTPUT_DIR / f"p5_physics_leakage_ghost_dtw_run_{run_tag}.json"
    run_md_path = OUTPUT_DIR / f"p5_physics_leakage_ghost_dtw_details_{run_tag}.md"

    payload = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "exact_command": " ".join(sys.argv),
        "scale": int(args.scale),
        "seed": int(args.seed),
        "max_iter_base": int(args.max_iter_base),
        "anchor_submission": int(args.anchor_sub),
        "blend_weights_depth": [float(w) for w in weights],
        "data": {
            "train_csv": "data/train.csv",
            "test_csv": "data/test.csv",
            "scaler_depth": "data/scaler_depth.pkl",
            "train_rows": int(len(y_depth)),
            "test_rows": int(len(pids_test)),
        },
        "feature_shapes": {
            "leak_train": list(leak_train.shape),
            "phys_train": list(phys_train.shape),
            "ghost_train": list(ghost_train.shape),
            "leak_test": list(leak_test.shape),
            "phys_test": list(phys_test.shape),
            "ghost_test": list(ghost_test.shape),
        },
        "oof_metrics": oof_metrics,
        "test_stats": test_stats,
        "submissions": submission_records,
        "elapsed_seconds": float(elapsed),
    }

    run_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# P5 Physics-Leakage-GhostDTW Ablation Run",
        "",
        f"- run_tag: `{run_tag}`",
        f"- exact_command: `{payload['exact_command']}`",
        f"- elapsed_seconds: `{elapsed:.15f}`",
        f"- anchor: `submission/submission_{args.anchor_sub}.csv`",
        f"- scale: `{args.scale}`",
        f"- seed: `{args.seed}`",
        f"- max_iter_base: `{args.max_iter_base}`",
        "",
        "## Data",
        "",
        f"- train_csv: `data/train.csv` (rows `{payload['data']['train_rows']}`)",
        f"- test_csv: `data/test.csv` (rows `{payload['data']['test_rows']}`)",
        f"- scaler_depth: `data/scaler_depth.pkl`",
        "",
        "## OOF Metrics (Depth, scaled)",
        "",
    ]

    for method in methods:
        metrics = oof_metrics[method]
        lines.append(f"### {method}")
        lines.append(f"- oof_mse_depth: `{metrics['oof_mse_depth']:.15f}`")
        per_player = metrics["oof_per_player_mse_depth"]
        for key in sorted(per_player.keys()):
            lines.append(f"- {key}: `{per_player[key]:.15f}`")
        lines.append("")

    lines.append("## Test Diversity vs Anchor Depth")
    lines.append("")
    for method in methods:
        st = test_stats[method]
        lines.append(f"### {method}")
        lines.append(f"- corr_vs_anchor_depth: `{st['corr_vs_anchor_depth']:.15f}`")
        lines.append(f"- rmsd_vs_anchor_depth: `{st['rmsd_vs_anchor_depth']:.15f}`")
        lines.append(f"- mean_delta_vs_anchor_depth: `{st['mean_delta_vs_anchor_depth']:.15f}`")
        lines.append(f"- mean_abs_delta_vs_anchor_depth: `{st['mean_abs_delta_vs_anchor_depth']:.15f}`")
        lines.append("")

    lines.append("## Generated Submission Files")
    lines.append("")
    for rec in submission_records:
        lines.append(
            f"- `submission/submission_{rec['submission_num']}.csv`: "
            f"method=`{rec['method']}`, depth_weight=`{rec['blend_weight_depth']:.15f}`, "
            f"changed_rows_depth=`{rec['changed_rows_depth']}`, rmsd_depth_vs_anchor=`{rec['rmsd_depth_vs_anchor']:.15f}`"
        )

    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "- Re-run command exactly to reproduce this run (submission numbers may differ due to atomic numbering)."
    )
    lines.append(f"- JSON artifact: `{run_json_path}`")

    run_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n[5/5] Artifacts written")
    print(f"  JSON: {run_json_path}")
    print(f"  MD:   {run_md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
