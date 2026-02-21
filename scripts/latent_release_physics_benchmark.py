"""
Latent Release Physics Benchmark

Compares two variants on the same per-example locally weighted model:
1) baseline: existing handcrafted compact features + PLS
2) latent: baseline + latent release-state physics features

Design:
- Uses existing per_example_pipeline functions for data loading/modeling.
- No submission files are written (CV benchmark only).
- Supports scale-controlled runs where only --scale changes between pilot/full.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import per_example_pipeline as pep


PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = pep.TARGETS
DT = pep.DT
FEET_TO_METERS = pep.FEET_TO_METERS
G = 9.81


@dataclass
class VariantResult:
    name: str
    per_target_mse: Dict[str, float]
    per_target_bw: Dict[str, float]
    mean_mse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent release physics benchmark")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor 1..8")
    parser.add_argument("--seed", type=int, default=20260215, help="Random seed")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional run tag. Default: UTC timestamp",
    )
    return parser.parse_args()


def build_scaled_train_subset(
    train_data: Dict[str, np.ndarray],
    scale: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Scale-controlled train subset:
    - scale >= 8: full training set
    - smaller scale: deterministic per-player subset
    """
    if scale >= 8:
        return train_data

    rng = np.random.default_rng(seed)
    pids = train_data["pids"]
    keep_mask = np.zeros(len(pids), dtype=bool)
    unique_pids = sorted(np.unique(pids))

    frac = max(0.15, min(1.0, scale / 8.0))
    for pid in unique_pids:
        idx = np.where(pids == pid)[0]
        n_keep = max(10, int(math.ceil(len(idx) * frac)))
        chosen = rng.choice(idx, size=n_keep, replace=False)
        keep_mask[chosen] = True

    subset = {}
    for key, value in train_data.items():
        if isinstance(value, np.ndarray) and len(value) == len(keep_mask):
            subset[key] = value[keep_mask]
        else:
            subset[key] = value
    return subset


def smooth_xyz(arr_xyz: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    out = np.array(arr_xyz, dtype=np.float64, copy=True)
    for ax in range(3):
        out[:, ax] = pep.safe_savgol(out[:, ax], window, poly)
    return out


def pseudo_ball_traj_hr(
    ts_hr: np.ndarray,
    kp_index: Dict[str, int],
) -> np.ndarray:
    rw_idx = kp_index.get("right_wrist")
    if rw_idx is None:
        return np.zeros((240, 3), dtype=np.float64)

    wrist = smooth_xyz(ts_hr[:, rw_idx, :], window=11, poly=3)
    ft_keys = [
        "right_second_finger_distal",
        "right_third_finger_distal",
        "right_fourth_finger_distal",
    ]
    ft_trajs = []
    for name in ft_keys:
        idx = kp_index.get(name)
        if idx is not None:
            ft_trajs.append(smooth_xyz(ts_hr[:, idx, :], window=15, poly=3))

    if ft_trajs:
        ft_center = np.mean(np.stack(ft_trajs, axis=0), axis=0)
        return wrist + 0.6 * (ft_center - wrist)
    return wrist


def fit_axis_with_accel_prior(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    accel_prior: float,
    accel_lambda: float,
) -> np.ndarray:
    """
    Fits y(t) = b0 + b1*t + 0.5*b2*t^2 with weighted least squares and
    soft acceleration prior (b2 ~ accel_prior).
    """
    A = np.column_stack([np.ones_like(t), t, 0.5 * t * t])
    sw = np.sqrt(np.clip(w, 1e-12, None))
    Aw = A * sw[:, None]
    yw = y * sw

    ata = Aw.T @ Aw
    aty = Aw.T @ yw

    # Soft prior: accel term close to accel_prior.
    ata[2, 2] += accel_lambda
    aty[2] += accel_lambda * accel_prior

    try:
        beta = np.linalg.solve(ata, aty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(ata) @ aty
    return beta


def release_context(
    ts_3d: np.ndarray,
    kp_index: Dict[str, int],
) -> Tuple[float, np.ndarray]:
    mid_hip_idx = kp_index.get("mid_hip", 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0.0
    hoop_xy = pep.HOOP_POS[:2]
    hoop_x = float(np.linalg.norm(hoop_xy - player_pos[:2]))
    return hoop_x, player_pos


def extract_latent_release_features(
    ts_3d: np.ndarray,
    ts_hr: np.ndarray,
    kp_index: Dict[str, int],
    release_frame: int,
) -> np.ndarray:
    """
    Latent release-state estimator:
    - estimate local release state (r0, v0, a0) from pseudo-ball trajectory
    - add soft physics priors on acceleration
    - return compact descriptors + reliability metrics
    """
    rf = int(np.clip(release_frame, 6, 235))
    idx = np.arange(rf - 6, rf + 5, dtype=int)
    t = (idx - rf) * DT

    # Heavier weight near and after release.
    w = np.exp(-((t / 0.06) ** 2))
    w[t >= 0] *= 1.35

    ball = pseudo_ball_traj_hr(ts_hr, kp_index)[idx] * FEET_TO_METERS

    bx = fit_axis_with_accel_prior(
        t=t,
        y=ball[:, 0],
        w=w,
        accel_prior=0.0,
        accel_lambda=1.5,
    )
    by = fit_axis_with_accel_prior(
        t=t,
        y=ball[:, 1],
        w=w,
        accel_prior=0.0,
        accel_lambda=1.5,
    )
    bz = fit_axis_with_accel_prior(
        t=t,
        y=ball[:, 2],
        w=w,
        accel_prior=-G,
        accel_lambda=2.0,
    )

    r0 = np.array([bx[0], by[0], bz[0]])
    v0 = np.array([bx[1], by[1], bz[1]])
    a0 = np.array([bx[2], by[2], bz[2]])

    pred = np.column_stack(
        [
            bx[0] + bx[1] * t + 0.5 * bx[2] * t * t,
            by[0] + by[1] * t + 0.5 * by[2] * t * t,
            bz[0] + bz[1] * t + 0.5 * bz[2] * t * t,
        ]
    )
    residual = ball - pred
    rmse_total = float(np.sqrt(np.mean(residual * residual)))
    pre_mask = t < 0
    post_mask = t >= 0
    rmse_pre = float(np.sqrt(np.mean(residual[pre_mask] * residual[pre_mask])))
    rmse_post = float(np.sqrt(np.mean(residual[post_mask] * residual[post_mask])))

    speed = float(np.linalg.norm(v0))
    horiz_speed = float(np.linalg.norm(v0[:2]))
    elev_deg = float(np.degrees(np.arctan2(v0[2], max(horiz_speed, 1e-9))))
    azim_deg = float(np.degrees(np.arctan2(v0[1], max(v0[0], 1e-9))))
    gravity_resid = float(a0[2] + G)
    lateral_acc = float(np.linalg.norm(a0[:2]))

    hoop_x, _ = release_context(ts_3d, kp_index)
    vx = float(v0[0])
    if vx > 0.1:
        t_hoop = (hoop_x * FEET_TO_METERS - r0[0]) / vx
    else:
        t_hoop = 0.0
    if t_hoop > 0:
        pred_z_hoop = float(r0[2] + v0[2] * t_hoop - 0.5 * G * t_hoop * t_hoop)
        pred_y_hoop = float(r0[1] + v0[1] * t_hoop)
        vz_hoop = float(v0[2] - G * t_hoop)
        entry_deg = float(np.degrees(np.arctan2(vz_hoop, max(horiz_speed, 1e-9))))
    else:
        pred_z_hoop = 0.0
        pred_y_hoop = 0.0
        entry_deg = 0.0

    reliability = float(
        np.exp(-rmse_total / 0.10)
        * np.exp(-abs(gravity_resid) / 10.0)
        * np.exp(-lateral_acc / 10.0)
    )
    reliability = float(np.clip(reliability, 0.0, 1.0))

    feats = np.array(
        [
            rf / 240.0,
            r0[0], r0[1], r0[2],
            v0[0], v0[1], v0[2],
            speed, elev_deg, azim_deg,
            a0[0], a0[1], a0[2],
            rmse_total, rmse_pre, rmse_post,
            gravity_resid, lateral_acc,
            t_hoop, pred_z_hoop, pred_y_hoop, entry_deg,
            reliability,
            v0[0] * reliability,
            v0[1] * reliability,
            v0[2] * reliability,
            entry_deg * reliability,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_latent_matrix(
    data: Dict[str, np.ndarray],
    release_frames: np.ndarray,
) -> np.ndarray:
    kp_index = data["kp_index"]
    n = len(release_frames)
    out = np.zeros((n, 27), dtype=np.float32)
    for i in range(n):
        ts_3d = data["X_3d"][i]
        ts_hr = pep.compute_hoop_transform(ts_3d, kp_index)
        out[i] = extract_latent_release_features(
            ts_3d=ts_3d,
            ts_hr=ts_hr,
            kp_index=kp_index,
            release_frame=int(release_frames[i]),
        )
    return out


def bandwidth_grid(scale: int) -> List[float]:
    # scale=1: fastest pilot
    # scale=8: full search used in this benchmark
    if scale <= 1:
        return [0.30]
    if scale <= 2:
        return [0.25, 0.30, 0.35]
    if scale <= 4:
        return [0.20, 0.25, 0.30, 0.35, 0.40]
    return [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def evaluate_variant(
    *,
    variant_name: str,
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    y_train: np.ndarray,
    pids_train: np.ndarray,
    pids_test: np.ndarray,
    scalers: Dict[str, object],
    scale: int,
) -> VariantResult:
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    per_target_mse: Dict[str, float] = {}
    per_target_bw: Dict[str, float] = {}

    bws = bandwidth_grid(scale)
    print(f"\n[{variant_name}] bandwidth grid: {bws}")

    for target in TARGETS:
        print(f"\n[{variant_name}] target={target}")

        X_train_hc, rf_train = pep.extract_all_features(train_data, target)
        X_test_hc, rf_test = pep.extract_all_features(test_data, target)

        if variant_name == "latent":
            X_train_latent = extract_latent_matrix(train_data, rf_train)
            X_test_latent = extract_latent_matrix(test_data, rf_test)
            X_train_hc = np.hstack([X_train_hc, X_train_latent])
            X_test_hc = np.hstack([X_test_hc, X_test_latent])
            print(
                f"[{variant_name}] feature dims: base+latent={X_train_hc.shape[1]}"
            )
        else:
            print(f"[{variant_name}] feature dims: base={X_train_hc.shape[1]}")

        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = pep.augment_with_pls(
            X_train_hc,
            y_raw,
            pids_train,
            X_test_hc,
            pids_test,
            train_data["X_raw"],
            test_data["X_raw"],
        )
        print(f"[{variant_name}] augmented dims={X_train_aug.shape[1]}")

        y_scaled = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        best_bw = bws[0]
        oof, _ = pep.locally_weighted_prediction(
            X_train_aug,
            y_scaled,
            X_test_aug,
            pids_train,
            pids_test,
            bandwidth_quantile=best_bw,
        )
        best_mse = float(np.mean((oof - y_scaled) ** 2))
        print(
            f"[{variant_name}] target={target} bw={best_bw:.2f} mse={best_mse:.12f}"
        )

        for bw in bws[1:]:
            oof_bw, _ = pep.locally_weighted_prediction(
                X_train_aug,
                y_scaled,
                X_test_aug,
                pids_train,
                pids_test,
                bandwidth_quantile=bw,
            )
            mse_bw = float(np.mean((oof_bw - y_scaled) ** 2))
            print(
                f"[{variant_name}] target={target} bw={bw:.2f} mse={mse_bw:.12f}"
            )
            if mse_bw < best_mse:
                best_mse = mse_bw
                best_bw = bw

        per_target_mse[target] = best_mse
        per_target_bw[target] = float(best_bw)
        print(
            f"[{variant_name}] target={target} best_bw={best_bw:.2f} "
            f"best_mse={best_mse:.12f}"
        )

    mean_mse = float(np.mean([per_target_mse[t] for t in TARGETS]))
    return VariantResult(
        name=variant_name,
        per_target_mse=per_target_mse,
        per_target_bw=per_target_bw,
        mean_mse=mean_mse,
    )


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag.strip() or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 80)
    print("LATENT RELEASE PHYSICS BENCHMARK")
    print("=" * 80)
    print(f"scale={args.scale} seed={args.seed} run_tag={run_tag}")

    train_full, test_data = pep.load_data()
    train_data = build_scaled_train_subset(train_full, args.scale, args.seed)
    y_train = train_data["y"]
    pids_train = train_data["pids"]
    pids_test = test_data["pids"]

    print(f"train shots used: {len(y_train)} / {len(train_full['y'])}")
    print(f"test shots used: {len(test_data['ids'])}")

    scalers = {t: pep.joblib.load(pep.DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}

    baseline = evaluate_variant(
        variant_name="baseline",
        train_data=train_data,
        test_data=test_data,
        y_train=y_train,
        pids_train=pids_train,
        pids_test=pids_test,
        scalers=scalers,
        scale=args.scale,
    )

    latent = evaluate_variant(
        variant_name="latent",
        train_data=train_data,
        test_data=test_data,
        y_train=y_train,
        pids_train=pids_train,
        pids_test=pids_test,
        scalers=scalers,
        scale=args.scale,
    )

    result = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scale": int(args.scale),
        "seed": int(args.seed),
        "n_train_used": int(len(y_train)),
        "n_train_full": int(len(train_full["y"])),
        "n_test": int(len(test_data["ids"])),
        "bandwidth_grid": bandwidth_grid(args.scale),
        "baseline": {
            "per_target_mse": baseline.per_target_mse,
            "per_target_bw": baseline.per_target_bw,
            "mean_mse": baseline.mean_mse,
        },
        "latent": {
            "per_target_mse": latent.per_target_mse,
            "per_target_bw": latent.per_target_bw,
            "mean_mse": latent.mean_mse,
        },
        "delta_mean_pct": (latent.mean_mse - baseline.mean_mse) / baseline.mean_mse * 100.0,
        "delta_targets_pct": {
            t: (latent.per_target_mse[t] - baseline.per_target_mse[t])
            / baseline.per_target_mse[t]
            * 100.0
            for t in TARGETS
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / f"latent_release_physics_benchmark_{run_tag}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULT SUMMARY")
    print("=" * 80)
    print(f"baseline mean mse: {baseline.mean_mse:.12f}")
    print(f"latent   mean mse: {latent.mean_mse:.12f}")
    print(f"delta mean (%):    {result['delta_mean_pct']:+.6f}")
    for t in TARGETS:
        print(
            f"{t}: base={baseline.per_target_mse[t]:.12f} "
            f"latent={latent.per_target_mse[t]:.12f} "
            f"delta={result['delta_targets_pct'][t]:+.6f}%"
        )
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()

