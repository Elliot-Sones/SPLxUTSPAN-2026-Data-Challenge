"""
Reverse-engineering experiments for the `angle` target.

Goal (per user request): try many plausible ways to reconstruct `angle` from
the kinematic time-series, measure how far off each option is on train.csv.

NOTE:
- The dataset does NOT include an explicit ball trajectory. Physics-based
  reconstruction therefore relies on assumptions (release frame, ball proxy
  point, scaling, gravity), and we expect non-zero error unless the label is
  deterministically encoded in the kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.data_loader import get_keypoint_columns, load_all_as_arrays
from src.physics_features import NUM_FRAMES, init_keypoint_mapping, get_keypoint_data
from src.fingertip_ballistics import (
    G_FT_S2,
    LEFT_HAND_DEFAULT_POINTS,
    RIGHT_HAND_DEFAULT_POINTS,
    detect_release_from_sphere_fit,
    detect_release_from_contact_constraints,
    entry_angle_at_rim_from_release,
    estimate_ball_center_series,
)


@dataclass(frozen=True)
class FitResult:
    name: str
    n: int
    mae: float
    rmse: float
    maxe: float
    pct_within_0p01: float
    meta: Dict[str, object]


def _impute_nan_with_col_mean(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    col_mean = np.nanmean(X, axis=0)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(col_mean, np.where(bad)[1])
    return X


def _metrics(pred: np.ndarray, y: np.ndarray, name: str, meta: Optional[Dict[str, object]] = None) -> FitResult:
    meta = {} if meta is None else dict(meta)
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() == 0:
        return FitResult(
            name=name,
            n=0,
            mae=float("nan"),
            rmse=float("nan"),
            maxe=float("nan"),
            pct_within_0p01=0.0,
            meta=meta,
        )
    err = pred[mask] - y[mask]
    return FitResult(
        name=name,
        n=int(mask.sum()),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err * err))),
        maxe=float(np.max(np.abs(err))),
        pct_within_0p01=float(np.mean(np.abs(err) <= 0.01) * 100.0),
        meta=meta,
    )


def _fit_affine(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    """Least-squares fit y ~= a*x + b (ignoring NaNs)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 50:
        return None
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    var = float(np.var(xx))
    if var < 1e-12:
        return None
    a = float(np.cov(xx, yy, bias=True)[0, 1] / var)
    b = float(yy.mean() - a * xx.mean())
    return a, b


def best_single_scalar_feature_affine(X: np.ndarray, y: np.ndarray, keypoint_cols: List[str]) -> FitResult:
    """
    Search over every scalar (frame, coordinate-column) and fit an affine mapping.

    This is a strong "label leak" check: if `angle` is directly encoded in some
    single time-series coordinate, this will find it.
    """
    N, T, C = X.shape
    Xf = X.reshape(N, T * C).astype(np.float64)

    # Simplify masking: drop rows with any NaNs across the full flattened input.
    rows_ok = np.isfinite(Xf).all(axis=1) & np.isfinite(y)
    Xf = Xf[rows_ok]
    yy = y[rows_ok].astype(np.float64)
    n = int(len(yy))

    # Closed-form best affine per feature using sums (fast).
    sum_x = np.sum(Xf, axis=0)
    sum_x2 = np.sum(Xf * Xf, axis=0)
    sum_y = float(np.sum(yy))
    sum_y2 = float(np.sum(yy * yy))
    sum_xy = np.dot(yy, Xf)  # (F,)

    mean_x = sum_x / n
    mean_y = sum_y / n
    var_x = sum_x2 / n - mean_x * mean_x
    cov_xy = sum_xy / n - mean_x * mean_y

    valid = var_x > 1e-12
    a = np.zeros_like(var_x)
    b = np.full_like(var_x, mean_y)
    a[valid] = cov_xy[valid] / var_x[valid]
    b[valid] = mean_y - a[valid] * mean_x[valid]

    # SSE via expansion (no need to allocate N x F predictions).
    sse = (
        sum_y2
        - 2.0 * a * sum_xy
        - 2.0 * b * sum_y
        + (a * a) * sum_x2
        + 2.0 * a * b * sum_x
        + n * (b * b)
    )
    rmse = np.sqrt(np.maximum(sse / n, 0.0))
    best_idx = int(np.nanargmin(rmse))

    frame = best_idx // C
    coord_col = keypoint_cols[best_idx % C]
    a_best = float(a[best_idx])
    b_best = float(b[best_idx])

    # Recompute predictions for metrics on the original N rows (drop rows with NaNs as above).
    pred = a_best * X.reshape(N, T * C)[:, best_idx] + b_best
    pred[~rows_ok] = np.nan
    return _metrics(
        pred=pred,
        y=y,
        name="best_single_scalar_affine",
        meta={"n_used_for_search": n, "frame": int(frame), "coord_col": coord_col, "a": a_best, "b": b_best},
    )


def best_arm_vector_pitch_affine(X: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Search over (point_a, point_b, frame, abs/not) for pitch angle, then affine-fit to y.

    This approximates "the arm/finger points upward at some frame determines angle".
    """
    points = [
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_second_finger_mcp",
        "right_second_finger_pip",
        "right_second_finger_dip",
        "right_second_finger_distal",
        "right_third_finger_mcp",
        "right_third_finger_pip",
        "right_third_finger_dip",
        "right_third_finger_distal",
        "right_fourth_finger_mcp",
        "right_fourth_finger_pip",
        "right_fourth_finger_dip",
        "right_fourth_finger_distal",
    ]

    N = X.shape[0]
    pos: Dict[str, np.ndarray] = {}
    for p in points:
        arr = np.empty((N, NUM_FRAMES, 3), dtype=np.float32)
        for i, ts in enumerate(X):
            arr[i] = get_keypoint_data(ts, p)
        pos[p] = arr.astype(np.float64)

    best = None
    for a_pt, b_pt in combinations(points, 2):
        A = pos[a_pt]
        B = pos[b_pt]
        dx = B[:, :, 0] - A[:, :, 0]
        dy = B[:, :, 1] - A[:, :, 1]
        dz = B[:, :, 2] - A[:, :, 2]
        horiz = np.sqrt(dx * dx + dy * dy)
        ang = np.degrees(np.arctan2(dz, horiz))
        ang_abs = np.degrees(np.arctan2(np.abs(dz), horiz))

        for t in range(NUM_FRAMES):
            for label, arr in (("pitch", ang[:, t]), ("pitch_abs", ang_abs[:, t])):
                fit = _fit_affine(arr, y)
                if fit is None:
                    continue
                a, b = fit
                pred = a * arr + b
                res = _metrics(
                    pred=pred,
                    y=y,
                    name="arm_vector_pitch_affine",
                    meta={"a_pt": a_pt, "b_pt": b_pt, "frame": t, "variant": label, "a": a, "b": b},
                )
                if best is None or res.rmse < best.rmse:
                    best = res

    assert best is not None
    best.meta["note"] = "Best pair/frame/variant from a brute-force search over right arm/hand vectors."
    return best


def best_velocity_pitch_affine(X: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Search over (keypoint proxy, frame, abs/not) for velocity pitch, then affine-fit to y.
    """
    # Velocity in ft/frame (dt=1) to avoid assuming FPS.
    def vel_per_frame(p: np.ndarray) -> np.ndarray:
        return np.gradient(p, axis=0)

    kps = [
        "right_wrist",
        "right_second_finger_distal",
        "right_third_finger_distal",
        "right_fourth_finger_distal",
    ]
    N = X.shape[0]

    vel: Dict[str, np.ndarray] = {}
    for kp in kps:
        v = np.empty((N, NUM_FRAMES, 3), dtype=np.float32)
        for i, ts in enumerate(X):
            p = get_keypoint_data(ts, kp)
            v[i] = vel_per_frame(p)
        vel[kp] = v.astype(np.float64)

    # Average fingertips (index/middle/ring) as a ball-contact proxy.
    vel["tips_avg"] = (vel["right_second_finger_distal"] + vel["right_third_finger_distal"] + vel["right_fourth_finger_distal"]) / 3.0

    best = None
    for name, v in vel.items():
        vx = v[:, :, 0]
        vy = v[:, :, 1]
        vz = v[:, :, 2]
        horiz = np.sqrt(vx * vx + vy * vy)
        pitch = np.degrees(np.arctan2(vz, horiz))
        pitch_abs = np.degrees(np.arctan2(np.abs(vz), horiz))

        for t in range(NUM_FRAMES):
            for label, arr in (("pitch", pitch[:, t]), ("pitch_abs", pitch_abs[:, t])):
                fit = _fit_affine(arr, y)
                if fit is None:
                    continue
                a, b = fit
                pred = a * arr + b
                res = _metrics(
                    pred=pred,
                    y=y,
                    name="velocity_pitch_affine",
                    meta={"kp": name, "frame": t, "variant": label, "a": a, "b": b},
                )
                if best is None or res.rmse < best.rmse:
                    best = res

    assert best is not None
    best.meta["note"] = "Best keypoint/frame/variant from brute-force search over velocity pitch."
    return best


def best_projectile_entry_angle_physics(X: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Try a simplified projectile model:
      - choose a release frame (heuristic)
      - choose a proxy point for ball velocity
      - assume v_ball = k * v_proxy (k tuned)
      - assume constant gravity g_frame in ft/frame^2 (tuned)
      - compute entry angle when z(t) hits 10ft

    This is included because it's the closest match to the dataset's definition,
    but it's expected to fail if the angle label comes from tracked ball flight.
    """
    def vel_per_frame(p: np.ndarray) -> np.ndarray:
        return np.gradient(p, axis=0)

    N = X.shape[0]
    search_start = NUM_FRAMES // 3

    # Candidate proxy points.
    # Proxy name -> key in pos/vel dicts.
    proxies = {
        "wrist": "right_wrist",
        "tips_avg": "tips_avg",
        "index_tip": "right_second_finger_distal",
        "middle_tip": "right_third_finger_distal",
        "ring_tip": "right_fourth_finger_distal",
        # Finger-geometry sphere fit (attempt at reconstructing the ball center while in hand).
        "sphere_center": "sphere_center",
    }

    pos: Dict[str, np.ndarray] = {}
    vel: Dict[str, np.ndarray] = {}
    for p in {"right_wrist", "right_second_finger_distal", "right_third_finger_distal", "right_fourth_finger_distal"}:
        arr = np.empty((N, NUM_FRAMES, 3), dtype=np.float32)
        v = np.empty((N, NUM_FRAMES, 3), dtype=np.float32)
        for i, ts in enumerate(X):
            pp = get_keypoint_data(ts, p)
            arr[i] = pp
            v[i] = vel_per_frame(pp)
        pos[p] = arr.astype(np.float64)
        vel[p] = v.astype(np.float64)

    pos["tips_avg"] = (pos["right_second_finger_distal"] + pos["right_third_finger_distal"] + pos["right_fourth_finger_distal"]) / 3.0
    vel["tips_avg"] = (vel["right_second_finger_distal"] + vel["right_third_finger_distal"] + vel["right_fourth_finger_distal"]) / 3.0

    # Sphere-fit center from all right-hand distal points (cheap enough at this dataset size).
    centers = np.full((N, NUM_FRAMES, 3), np.nan, dtype=np.float64)
    for i, ts in enumerate(X):
        c, _r, _e = estimate_ball_center_series(ts, RIGHT_HAND_DEFAULT_POINTS, robust_iters=2, min_points=4)
        centers[i] = c
    pos["sphere_center"] = centers
    vel["sphere_center"] = np.gradient(centers, axis=1)

    # Release frame heuristics (all per-shot, using only kinematics).
    release_methods: Dict[str, np.ndarray] = {}
    speed_wrist = np.linalg.norm(vel["right_wrist"], axis=2)
    release_methods["wrist_max_speed"] = search_start + np.nanargmax(speed_wrist[:, search_start:], axis=1)
    release_methods["wrist_max_z"] = search_start + np.nanargmax(pos["right_wrist"][:, search_start:, 2], axis=1)
    release_methods["tips_max_vz"] = search_start + np.nanargmax(vel["tips_avg"][:, search_start:, 2], axis=1)

    def entry_angle_from_release(z0: np.ndarray, v0: np.ndarray, g_frame: float) -> np.ndarray:
        vx = v0[:, 0]
        vy = v0[:, 1]
        vz = v0[:, 2]

        a = -0.5 * g_frame
        b = vz
        c = z0 - 10.0

        t = np.full_like(z0, np.nan, dtype=np.float64)
        if abs(a) < 1e-12:
            t_lin = (10.0 - z0) / vz
            t = t_lin
        else:
            disc = b * b - 4.0 * a * c
            valid = (disc >= 0) & np.isfinite(disc)
            sqrt_disc = np.sqrt(disc[valid])
            b_valid = b[valid]
            t1 = (-b_valid + sqrt_disc) / (2.0 * a)
            t2 = (-b_valid - sqrt_disc) / (2.0 * a)
            t_sel = np.maximum(t1, t2)
            # if the larger root is negative, try the smaller.
            t_alt = np.minimum(t1, t2)
            t[valid] = np.where(t_sel > 0, t_sel, t_alt)

        valid_t = (t > 0) & np.isfinite(t)
        vz_entry = vz - g_frame * t
        horiz = np.sqrt(vx * vx + vy * vy)
        ang = np.degrees(np.arctan2(np.abs(vz_entry), horiz))
        ang[~valid_t] = np.nan
        return ang

    # Search includes very small g_frame because time base is unknown (frames are time-normalized).
    g_grid = np.concatenate([np.linspace(0.0, 0.002, 81), np.linspace(0.0025, 0.02, 71)])
    k_grid = np.linspace(0.5, 40.0, 158)  # speed scale from proxy -> ball

    best: Optional[FitResult] = None
    for rname, ridx in release_methods.items():
        ridx = ridx.astype(int)
        idx = np.arange(N)
        for proxy_name, pkey in proxies.items():
            z0 = pos[pkey][idx, ridx, 2]
            vbase = vel[pkey][idx, ridx]
            base_ok = np.isfinite(z0) & np.isfinite(vbase).all(axis=1)
            if base_ok.sum() < 250:
                continue

            for g in g_grid:
                for k in k_grid:
                    pred = entry_angle_from_release(z0, vbase * k, float(g))
                    res = _metrics(
                    pred=pred,
                    y=y,
                    name="projectile_entry_angle",
                    meta={"release": rname, "proxy": proxy_name, "g_frame": float(g), "k": float(k)},
                )
                    # Require most rows to be valid; otherwise this is just exploiting NaN filtering.
                    if res.n < 300:
                        continue
                    if best is None or res.mae < best.mae:
                        best = res

    assert best is not None
    best.meta["note"] = "Best (release heuristic, proxy, g_frame, k) by MAE."
    return best


def ridge_overfit_on_flattened(X: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> FitResult:
    """Interpolating baseline: fit Ridge on flattened input and report TRAIN error."""
    from sklearn.linear_model import Ridge

    Xf = X.reshape(X.shape[0], -1).astype(np.float64)
    Xf = _impute_nan_with_col_mean(Xf)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xf, y.astype(np.float64))
    pred = model.predict(Xf)
    return _metrics(
        pred=pred,
        y=y,
        name="ridge_overfit_flattened_train",
        meta={"alpha": float(alpha), "n_features": int(Xf.shape[1])},
    )


def ridge_cv_on_flattened(X: np.ndarray, y: np.ndarray, alpha: float = 100.0, n_splits: int = 5) -> FitResult:
    """Non-interpolating check: KFold CV error for Ridge on flattened input."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    Xf = X.reshape(X.shape[0], -1).astype(np.float64)
    Xf = _impute_nan_with_col_mean(Xf)
    yy = y.astype(np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = np.full_like(yy, np.nan, dtype=np.float64)
    for tr, va in kf.split(Xf):
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xf[tr], yy[tr])
        preds[va] = model.predict(Xf[va])

    return _metrics(
        pred=preds,
        y=yy,
        name="ridge_flattened_kfold_cv",
        meta={"alpha": float(alpha), "n_splits": int(n_splits)},
    )


def best_fingertip_sphere_ballistics_angle(X: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Physics attempt:
      1) Fit a sphere to fingertip points per frame -> ball center proxy
      2) Detect release as last good-fit frame
      3) Estimate release velocity from center near release
      4) Propagate to rim plane z=10 ft and compute entry angle there

    Searches a small grid of hyperparameters (global constants) and reports best MAE.
    """
    N = X.shape[0]
    start_search = NUM_FRAMES // 3
    end_search = NUM_FRAMES - 5

    # We keep this attempt as the earlier "unconstrained sphere + MAD thresholds" baseline.
    point_sets = {
        "right_only": list(RIGHT_HAND_DEFAULT_POINTS),
        "both_hands": list(RIGHT_HAND_DEFAULT_POINTS + LEFT_HAND_DEFAULT_POINTS),
    }
    k_mad_grid = [2.0, 3.0, 4.0, 5.0]
    min_run_grid = [3, 5, 7]
    vel_window_grid = [3, 5, 7]
    g_scale_grid = [0.95, 1.0, 1.05]

    def _velocity_from_centers(centers: np.ndarray, rel: int, vel_window: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Local linear fit on last `vel_window` frames before rel (inclusive).
        w = int(max(3, vel_window))
        t0 = max(0, rel - w + 1)
        idx = np.arange(t0, rel + 1)
        C = centers[idx]
        ok = np.isfinite(C).all(axis=1)
        if ok.sum() < 3:
            return None
        idx = idx[ok]
        C = C[ok]
        tt = (idx.astype(np.float64) - float(rel)) * (1.0 / 60.0)
        A = np.stack([np.ones_like(tt), tt], axis=1)
        v = np.zeros(3, dtype=np.float64)
        p0 = np.zeros(3, dtype=np.float64)
        for ax in range(3):
            coef, *_ = np.linalg.lstsq(A, C[:, ax], rcond=None)
            p0[ax] = coef[0]
            v[ax] = coef[1]
        return p0, v

    best: Optional[FitResult] = None

    # Precompute sphere-fit series once per point set (expensive part).
    precomp: Dict[str, Dict[str, np.ndarray]] = {}
    for pset_name, pnames in point_sets.items():
        centers_all = np.full((N, NUM_FRAMES, 3), np.nan, dtype=np.float64)
        radii_all = np.full((N, NUM_FRAMES), np.nan, dtype=np.float64)
        rms_all = np.full((N, NUM_FRAMES), np.nan, dtype=np.float64)
        for i in range(N):
            c, r, e = estimate_ball_center_series(X[i], pnames, robust_iters=2, min_points=4)
            centers_all[i] = c
            radii_all[i] = r
            rms_all[i] = e
        precomp[pset_name] = {"centers": centers_all, "radii": radii_all, "rms": rms_all}

    for pset_name in point_sets.keys():
        centers_all = precomp[pset_name]["centers"]
        radii_all = precomp[pset_name]["radii"]
        rms_all = precomp[pset_name]["rms"]

        for k_mad in k_mad_grid:
            for min_run in min_run_grid:
                for vel_window in vel_window_grid:
                    # Release frame depends on detection thresholds; cache per (k_mad, min_run) per shot.
                    release_idx = np.full((N,), -1, dtype=np.int32)
                    for i in range(N):
                        rel = detect_release_from_sphere_fit(
                            radii=radii_all[i],
                            rms=rms_all[i],
                            start_search=start_search,
                            end_search=end_search,
                            k_mad=k_mad,
                            min_run=min_run,
                        )
                        if rel is not None:
                            release_idx[i] = int(rel)

                    for g_scale in g_scale_grid:
                        preds = np.full((N,), np.nan, dtype=np.float64)
                        n_success = 0
                        for i in range(N):
                            rel = int(release_idx[i])
                            if rel < 0:
                                continue
                            pv = _velocity_from_centers(centers_all[i], rel, vel_window)
                            if pv is None:
                                continue
                            p0, v0 = pv
                            ang = entry_angle_at_rim_from_release(
                                p0,
                                v0,
                                g_ft_s2=float(G_FT_S2 * g_scale),
                            )
                            if ang is None or not np.isfinite(ang):
                                continue
                            preds[i] = float(ang)
                            n_success += 1

                        res = _metrics(
                            pred=preds,
                            y=y,
                            name="fingertip_sphere_ballistics",
                            meta={
                                "point_set": pset_name,
                                "k_mad": float(k_mad),
                                "min_run": int(min_run),
                                "vel_window": int(vel_window),
                                "g_scale": float(g_scale),
                                "n_success": int(n_success),
                            },
                        )
                        if res.n < 250:
                            continue
                        if best is None or res.mae < best.mae:
                            best = res

    if best is None:
        return FitResult(
            name="fingertip_sphere_ballistics",
            n=0,
            mae=float("nan"),
            rmse=float("nan"),
            maxe=float("nan"),
            pct_within_0p01=0.0,
            meta={"note": "No valid predictions (sphere-fit release failed)."},
        )
    best.meta["note"] = "Best global settings (grid search) by MAE; uses no labels per shot."
    return best


def best_fingertip_fixed_radius_ballistics_angle(X: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Improved physics attempt:
    - Fit a sphere with FIXED radius to right-hand fingertips to estimate ball center.
    - Detect release via strict contact constraints (fit residual + center-to-wrist distance).
    - Enforce consistent gravity scaling: g_frame = 32.174 / k^2 where k is effective fps.
    """
    N = X.shape[0]
    start_search = NUM_FRAMES // 3
    end_search = NUM_FRAMES - 5

    # Effective ball radius in feet is ~0.39. We tune an "effective" radius because fingertip
    # markers are noisy and may not lie exactly on the ball surface.
    radius_grid = [0.25, 0.30, 0.35, 0.39, 0.42, 0.45, 0.50]
    rms_thr_grid = [0.002, 0.003, 0.004, 0.006, 0.008]
    dist_min_grid = [0.05, 0.10, 0.15]
    dist_max_grid = [0.50, 0.70, 0.90]
    min_run_grid = [3, 5]
    vel_window_grid = [3, 5, 7]
    # k = effective fps scale converting ft/frame -> ft/s; we enforce g_frame accordingly.
    k_grid = [120.0, 180.0, 240.0, 300.0, 360.0, 420.0]

    # Preload wrist trajectories (needed for contact constraints).
    wrist = np.empty((N, NUM_FRAMES, 3), dtype=np.float64)
    for i, ts in enumerate(X):
        wrist[i] = get_keypoint_data(ts, "right_wrist").astype(np.float64)

    best: Optional[FitResult] = None

    # Heavy part: compute fixed-radius sphere centers + rms for each radius.
    for radius in radius_grid:
        centers_all = np.full((N, NUM_FRAMES, 3), np.nan, dtype=np.float64)
        rms_all = np.full((N, NUM_FRAMES), np.nan, dtype=np.float64)
        for i in range(N):
            c, _r, e = estimate_ball_center_series(
                X[i],
                RIGHT_HAND_DEFAULT_POINTS,
                robust_iters=2,
                min_points=4,
                radius_fixed=float(radius),
            )
            centers_all[i] = c
            rms_all[i] = e

        # For speed, we reuse release indices across k_grid (only depends on thresholds).
        for rms_thr in rms_thr_grid:
            for dist_min in dist_min_grid:
                for dist_max in dist_max_grid:
                    if dist_max <= dist_min:
                        continue
                    for min_run in min_run_grid:
                        release_idx = np.full((N,), -1, dtype=np.int32)
                        for i in range(N):
                            rel = detect_release_from_contact_constraints(
                                centers_all[i],
                                rms_all[i],
                                wrist[i],
                                start_search=start_search,
                                end_search=end_search,
                                rms_thr=float(rms_thr),
                                dist_min=float(dist_min),
                                dist_max=float(dist_max),
                                min_run=int(min_run),
                            )
                            if rel is not None:
                                release_idx[i] = int(rel)

                        for vel_window in vel_window_grid:
                            w = int(max(3, vel_window))
                            # Precompute per-shot (p0, v_frame) at detected release.
                            p0_all = np.full((N, 3), np.nan, dtype=np.float64)
                            v_frame_all = np.full((N, 3), np.nan, dtype=np.float64)
                            for i in range(N):
                                rel = int(release_idx[i])
                                if rel < 0:
                                    continue
                                t0 = max(0, rel - w + 1)
                                idx = np.arange(t0, rel + 1)
                                C = centers_all[i, idx]
                                ok = np.isfinite(C).all(axis=1)
                                if ok.sum() < 3:
                                    continue
                                idx = idx[ok]
                                C = C[ok]
                                # Time in frames relative to release (so slope is ft/frame).
                                tt = (idx.astype(np.float64) - float(rel))
                                A = np.stack([np.ones_like(tt), tt], axis=1)
                                for ax in range(3):
                                    coef, *_ = np.linalg.lstsq(A, C[:, ax], rcond=None)
                                    p0_all[i, ax] = coef[0]
                                    v_frame_all[i, ax] = coef[1]

                            # Evaluate per k_eff globally.
                            preds = np.full((N,), np.nan, dtype=np.float64)
                            for k_eff in k_grid:
                                preds[:] = np.nan
                                n_success = 0
                                for i in range(N):
                                    p0 = p0_all[i]
                                    v_frame = v_frame_all[i]
                                    if not np.isfinite(p0).all() or not np.isfinite(v_frame).all():
                                        continue
                                    v0 = v_frame * float(k_eff)  # ft/s
                                    ang = entry_angle_at_rim_from_release(p0, v0, g_ft_s2=float(G_FT_S2))
                                    if ang is None or not np.isfinite(ang):
                                        continue
                                    preds[i] = float(ang)
                                    n_success += 1

                                res = _metrics(
                                    pred=preds,
                                    y=y,
                                    name="fingertip_fixed_radius_ballistics",
                                    meta={
                                        "radius_eff_ft": float(radius),
                                        "rms_thr": float(rms_thr),
                                        "dist_min": float(dist_min),
                                        "dist_max": float(dist_max),
                                        "min_run": int(min_run),
                                        "vel_window": int(w),
                                        "k_eff_fps": float(k_eff),
                                        "n_success": int(n_success),
                                    },
                                )
                                if res.n < 250:
                                    continue
                                if best is None or res.mae < best.mae:
                                    best = res

    if best is None:
        return FitResult(
            name="fingertip_fixed_radius_ballistics",
            n=0,
            mae=float("nan"),
            rmse=float("nan"),
            maxe=float("nan"),
            pct_within_0p01=0.0,
            meta={"note": "No valid predictions."},
        )
    best.meta["note"] = "Fixed-radius sphere fit + contact constraints + global time scale; best by MAE."
    return best


def main() -> None:
    X, y, _meta = load_all_as_arrays(train=True)
    angle = y[:, 0].astype(np.float64)

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    results: List[FitResult] = []

    results.append(best_single_scalar_feature_affine(X, angle, keypoint_cols))
    results.append(best_arm_vector_pitch_affine(X, angle))
    results.append(best_velocity_pitch_affine(X, angle))
    results.append(best_projectile_entry_angle_physics(X, angle))
    results.append(best_fingertip_sphere_ballistics_angle(X, angle))
    results.append(best_fingertip_fixed_radius_ballistics_angle(X, angle))
    results.append(ridge_overfit_on_flattened(X, angle, alpha=1e-6))
    results.append(ridge_cv_on_flattened(X, angle, alpha=100.0, n_splits=5))

    # Print compact report
    results_sorted = sorted(
        results, key=lambda r: (np.nan_to_num(r.rmse, nan=1e9), np.nan_to_num(r.mae, nan=1e9))
    )
    print("\nAngle reconstruction attempts (train.csv)")
    print("name\tn\tmae\trmse\tmaxe\t%<=0.01\tmeta")
    for r in results_sorted:
        meta_str = ", ".join(f"{k}={v}" for k, v in r.meta.items())
        print(
            f"{r.name}\t{r.n}\t{r.mae:.6f}\t{r.rmse:.6f}\t{r.maxe:.6f}\t{r.pct_within_0p01:.2f}\t{meta_str}"
        )


if __name__ == "__main__":
    main()
