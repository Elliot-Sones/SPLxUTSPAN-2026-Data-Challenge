#!/usr/bin/env python3
"""
Reverse-engineer a "ball release velocity" on train, then learn a mapping from
measured wrist/finger velocities -> ball release velocity, within participant.

What this does (high level):
1) Choose a release frame from kinematics (no labels).
2) At that frame, compute wrist velocity and fingertip velocity (ft/frame).
3) Use TRAIN labels (angle, depth, left_right) + hoop geometry to back-solve a
   projectile-consistent release velocity v_ball_gt (ft/frame) for that shot.
   - This uses the label, so it's only possible on train. It's used as a teacher
     signal to learn a mapping.
4) Fit a simple linear model per participant:
     v_ball ~= A * [v_wrist, v_tips, v_tips - v_wrist, speeds...] + b
5) Evaluate angle error when using predicted v_ball (LOOCV within participant).

This is a "physics teacher -> kinematics student" approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from src.data_loader import get_keypoint_columns, load_all_as_arrays
from src.physics_features import init_keypoint_mapping, get_keypoint_data, compute_joint_angle


ReleaseMethod = Literal["wrist_max_speed", "tips_max_vz"]

HOOP_POS_FT = np.array([5.25, -25.0, 10.0], dtype=np.float64)
RIM_Z_FT = 10.0


@dataclass(frozen=True)
class EvalResult:
    release_method: str
    g_frame: float
    n: int
    mae: float
    rmse: float
    maxe: float
    pct_le_0p1: float


def _vel_per_frame(p: np.ndarray) -> np.ndarray:
    """Finite difference velocity in ft/frame."""
    return np.gradient(p.astype(np.float64), axis=0)


def _release_frame(ts: np.ndarray, method: ReleaseMethod, start: int = 80, end: int = 235) -> int:
    start = int(max(1, min(start, 238)))
    end = int(max(start + 1, min(end, 239)))

    if method == "wrist_max_speed":
        w = get_keypoint_data(ts, "right_wrist")
        v = _vel_per_frame(w)
        s = np.linalg.norm(v, axis=1)
        return int(start + np.nanargmax(s[start:end]))

    if method == "tips_max_vz":
        idx_tip = get_keypoint_data(ts, "right_second_finger_distal")
        mid_tip = get_keypoint_data(ts, "right_third_finger_distal")
        ring_tip = get_keypoint_data(ts, "right_fourth_finger_distal")
        tips = (idx_tip + mid_tip + ring_tip) / 3.0
        v = _vel_per_frame(tips)
        vz = v[:, 2]
        return int(start + np.nanargmax(vz[start:end]))

    raise ValueError(f"unknown release method: {method}")


def _rim_point_from_outcomes(depth_in: float, left_right_in: float) -> np.ndarray:
    """Convert outcome offsets (inches) to a point at the rim plane in feet."""
    return np.array(
        [
            HOOP_POS_FT[0] + float(left_right_in) / 12.0,
            HOOP_POS_FT[1] + float(depth_in) / 12.0,
            RIM_Z_FT,
        ],
        dtype=np.float64,
    )


def _angle_at_rim_from_v0(p0: np.ndarray, v0: np.ndarray, g_frame: float) -> Optional[float]:
    """Compute entry angle (deg, positive) when z(t)=10 ft, using time in frames."""
    z0 = float(p0[2])
    vz0 = float(v0[2])
    a = -0.5 * float(g_frame)
    b = float(vz0)
    c = float(z0 - RIM_Z_FT)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sd = float(np.sqrt(disc))
    t1 = (-b + sd) / (2.0 * a)
    t2 = (-b - sd) / (2.0 * a)
    ts = [t for t in (t1, t2) if np.isfinite(t) and t > 1e-6]
    if not ts:
        return None
    t = float(max(ts))
    vz = vz0 - float(g_frame) * t
    vh = float(np.sqrt(float(v0[0]) ** 2 + float(v0[1]) ** 2))
    if vh < 1e-9:
        return None
    return float(np.degrees(np.arctan2(abs(vz), vh)))


def solve_v0_from_outcomes(
    p0: np.ndarray,
    rim: np.ndarray,
    angle_target_deg: float,
    g_frame: float,
    *,
    t_min: float = 5.0,
    t_max: float = 200.0,
    n_grid: int = 400,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Back-solve a release velocity v0 (ft/frame) that yields:
      - position hits rim point at some time t (frames)
      - entry angle at rim plane equals angle_target_deg

    Unknown is t. For any candidate t:
      vx = (x_rim - x0)/t
      vy = (y_rim - y0)/t
      v0z chosen so z(t)=RIM_Z_FT holds.
      Then compute entry angle at rim plane and match.

    Returns (v0, t_frames) or None if numeric failure.
    """
    x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2])
    xr, yr = float(rim[0]), float(rim[1])
    dz = float(RIM_Z_FT - z0)

    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(z0)):
        return None
    if not np.isfinite(angle_target_deg):
        return None

    angle_target = float(abs(angle_target_deg))
    g = float(g_frame)

    # Evaluate f(t) = angle_pred(t) - angle_target over a grid and pick root/min.
    t_grid = np.linspace(float(t_min), float(t_max), int(n_grid), dtype=np.float64)
    f = np.full_like(t_grid, np.nan, dtype=np.float64)
    vx = (xr - x0) / t_grid
    vy = (yr - y0) / t_grid

    # v0z ensuring z(t)=RIM_Z_FT
    v0z = (dz + 0.5 * g * t_grid * t_grid) / t_grid
    vz_rim = v0z - g * t_grid
    vh = np.sqrt(vx * vx + vy * vy)
    ang = np.degrees(np.arctan2(np.abs(vz_rim), vh))
    f = ang - angle_target

    ok = np.isfinite(f)
    if ok.sum() < 10:
        return None

    f_ok = f[ok]
    t_ok = t_grid[ok]

    # Find a sign change for bisection.
    sign = np.sign(f_ok)
    sign[sign == 0] = 1.0
    changes = np.where(sign[:-1] * sign[1:] < 0)[0]
    if changes.size > 0:
        i = int(changes[0])
        lo, hi = float(t_ok[i]), float(t_ok[i + 1])
        flo, fhi = float(f_ok[i]), float(f_ok[i + 1])
        # Bisection
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            vxm = (xr - x0) / mid
            vym = (yr - y0) / mid
            v0zm = (dz + 0.5 * g * mid * mid) / mid
            vzm = v0zm - g * mid
            vhm = float(np.sqrt(vxm * vxm + vym * vym))
            am = float(np.degrees(np.arctan2(abs(vzm), vhm))) - angle_target
            if not np.isfinite(am):
                break
            if abs(am) < 1e-8:
                lo = hi = mid
                break
            if flo * am <= 0:
                hi = mid
                fhi = am
            else:
                lo = mid
                flo = am
        t_star = 0.5 * (lo + hi)
    else:
        # No sign change found; pick t that minimizes |f(t)|.
        t_star = float(t_ok[int(np.argmin(np.abs(f_ok)))])

    vx0 = (xr - x0) / t_star
    vy0 = (yr - y0) / t_star
    vz0 = (dz + 0.5 * g * t_star * t_star) / t_star
    v0 = np.array([vx0, vy0, vz0], dtype=np.float64)
    return v0, float(t_star)


def build_features_at_release(ts: np.ndarray, r: int) -> np.ndarray:
    """
    Feature vector focused on what you asked:
      - wrist velocity
      - fingertip velocity (avg of 2nd/3rd/4th distal)
      - relative (tip - wrist)
      - speeds
      - a couple of release angles (elbow + wrist) for stability
    """
    w = get_keypoint_data(ts, "right_wrist").astype(np.float64)
    e = get_keypoint_data(ts, "right_elbow").astype(np.float64)
    s = get_keypoint_data(ts, "right_shoulder").astype(np.float64)
    idx_tip = get_keypoint_data(ts, "right_second_finger_distal").astype(np.float64)
    mid_tip = get_keypoint_data(ts, "right_third_finger_distal").astype(np.float64)
    ring_tip = get_keypoint_data(ts, "right_fourth_finger_distal").astype(np.float64)
    tips = (idx_tip + mid_tip + ring_tip) / 3.0

    vw = _vel_per_frame(w)[r]
    vt = _vel_per_frame(tips)[r]
    vrel = vt - vw

    sw = float(np.linalg.norm(vw))
    st = float(np.linalg.norm(vt))
    srel = float(np.linalg.norm(vrel))

    # Joint angles at release.
    elbow_ang = float(compute_joint_angle(s, e, w)[r])
    # Wrist angle proxy (elbow-wrist-index tip)
    wrist_ang = float(compute_joint_angle(e, w, idx_tip)[r])

    feats = np.concatenate(
        [
            vw,
            vt,
            vrel,
            np.array([sw, st, srel, elbow_ang, wrist_ang], dtype=np.float64),
        ],
        axis=0,
    )
    return feats.astype(np.float64)


def ridge_fit_predict(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Multi-output ridge (closed form via sklearn)."""
    from sklearn.linear_model import Ridge

    # Impute NaNs with train means.
    mu = np.nanmean(Xtr, axis=0)
    Xtr2 = Xtr.copy()
    Xte2 = Xte.copy()
    bad = ~np.isfinite(Xtr2)
    if bad.any():
        Xtr2[bad] = np.take(mu, np.where(bad)[1])
    bad = ~np.isfinite(Xte2)
    if bad.any():
        Xte2[bad] = np.take(mu, np.where(bad)[1])

    model = Ridge(alpha=float(alpha), fit_intercept=True, random_state=0)
    model.fit(Xtr2, Ytr)
    return model.predict(Xte2)


def eval_within_participant_loocv(
    X: np.ndarray,
    y_angle: np.ndarray,
    y_depth: np.ndarray,
    y_lr: np.ndarray,
    participants: np.ndarray,
    *,
    release_method: ReleaseMethod,
    g_frame: float,
) -> EvalResult:
    preds = []
    trues = []
    n = 0
    for pid in np.unique(participants):
        idxs = np.where(participants == pid)[0]
        if idxs.size < 3:
            continue
        # Precompute per shot teacher v0_gt and kinematic features.
        feats = []
        v_gt = []
        p0s = []
        ok_mask = []
        for i in idxs:
            ts = X[i]
            r = _release_frame(ts, release_method)
            p0 = get_keypoint_data(ts, "right_wrist")[r].astype(np.float64)
            rim = _rim_point_from_outcomes(float(y_depth[i]), float(y_lr[i]))
            sol = solve_v0_from_outcomes(p0, rim, float(y_angle[i]), float(g_frame))
            if sol is None:
                ok_mask.append(False)
                feats.append(np.full((14,), np.nan))
                v_gt.append(np.full((3,), np.nan))
                p0s.append(p0)
                continue
            v0, _t = sol
            feats.append(build_features_at_release(ts, r))
            v_gt.append(v0)
            p0s.append(p0)
            ok_mask.append(True)

        feats = np.stack(feats, axis=0).astype(np.float64)
        v_gt = np.stack(v_gt, axis=0).astype(np.float64)
        p0s = np.stack(p0s, axis=0).astype(np.float64)
        ok_mask = np.array(ok_mask, dtype=bool)

        # LOOCV
        for k in range(idxs.size):
            if not ok_mask[k]:
                continue
            train_mask = ok_mask.copy()
            train_mask[k] = False
            if train_mask.sum() < 2:
                continue
            pred_v0 = ridge_fit_predict(feats[train_mask], v_gt[train_mask], feats[[k]], alpha=1.0)[0]
            ang = _angle_at_rim_from_v0(p0s[k], pred_v0, float(g_frame))
            if ang is None or not np.isfinite(ang):
                continue
            preds.append(float(ang))
            trues.append(float(y_angle[idxs[k]]))
            n += 1

    if n == 0:
        return EvalResult(str(release_method), float(g_frame), 0, float("nan"), float("nan"), float("nan"), 0.0)

    preds = np.array(preds, dtype=np.float64)
    trues = np.array(trues, dtype=np.float64)
    err = preds - trues
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    maxe = float(np.max(np.abs(err)))
    pct = float(np.mean(np.abs(err) <= 0.1) * 100.0)
    return EvalResult(str(release_method), float(g_frame), int(n), mae, rmse, maxe, pct)


def main() -> None:
    X, y, meta = load_all_as_arrays(train=True)
    y_angle = y[:, 0].astype(np.float64)
    y_depth = y[:, 1].astype(np.float64)
    y_lr = y[:, 2].astype(np.float64)
    participants = meta["participant_id"].to_numpy()

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    release_methods: List[ReleaseMethod] = ["wrist_max_speed", "tips_max_vz"]
    # g in ft/frame^2; since time is normalized, sweep a broad range.
    g_grid = np.concatenate(
        [
            np.linspace(0.00005, 0.0010, 20),
            np.linspace(0.0012, 0.0100, 20),
            np.linspace(0.012, 0.030, 10),
        ]
    )

    results: List[EvalResult] = []
    best: Optional[EvalResult] = None
    for rm in release_methods:
        for g in g_grid:
            r = eval_within_participant_loocv(
                X,
                y_angle,
                y_depth,
                y_lr,
                participants,
                release_method=rm,
                g_frame=float(g),
            )
            if r.n < 250:
                continue
            results.append(r)
            if best is None or r.mae < best.mae:
                best = r

    results.sort(key=lambda r: (np.nan_to_num(r.mae, nan=1e9), np.nan_to_num(r.rmse, nan=1e9)))
    print("Reverse-engineer ball release velocity from outcomes, learn v_ball from wrist+finger velocities")
    print("Evaluation: LOOCV within participant; metric is angle error at rim plane")
    print("method\tg_frame\tn\tmae\trmse\tmaxe\t%<=0.1")
    for r in results[:15]:
        print(f"{r.release_method}\t{r.g_frame:.6f}\t{r.n}\t{r.mae:.3f}\t{r.rmse:.3f}\t{r.maxe:.3f}\t{r.pct_le_0p1:.2f}")
    if best is not None:
        print(f"\nBEST: {best.release_method} g_frame={best.g_frame:.6f} n={best.n} MAE={best.mae:.3f} RMSE={best.rmse:.3f} max={best.maxe:.3f} %<=0.1={best.pct_le_0p1:.2f}")


if __name__ == "__main__":
    main()

