"""
Estimate ball release state from finger keypoints, then compute entry angle at the rim plane.

This is a "solve it" attempt (physics-based) given the dataset does not include ball tracking.

Idea:
- While the ball is in the hands, multiple fingertip/distal points should lie on (approximately)
  a common sphere (the ball surface). Fit a sphere per frame to estimate ball center.
- Detect release as the last frame where the sphere fit is stable/good.
- Estimate release velocity from the ball-center time series near release.
- Propagate the projectile to the rim plane z=10 ft and compute entry angle from velocity there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.physics_features import DT, NUM_FRAMES, get_keypoint_data


HOOP_Z_FT = 10.0
G_FT_S2 = 32.174  # gravitational acceleration in ft/s^2


RIGHT_HAND_DEFAULT_POINTS: Tuple[str, ...] = (
    "right_thumb",
    "right_first_finger_distal",
    "right_second_finger_distal",
    "right_third_finger_distal",
    "right_fourth_finger_distal",
    "right_fifth_finger_distal",
)

LEFT_HAND_DEFAULT_POINTS: Tuple[str, ...] = (
    "left_thumb",
    "left_first_finger_distal",
    "left_second_finger_distal",
    "left_third_finger_distal",
    "left_fourth_finger_distal",
    "left_fifth_finger_distal",
)


@dataclass(frozen=True)
class SphereFit:
    center: np.ndarray  # (3,)
    radius: float
    rms_residual: float
    n_points: int


@dataclass(frozen=True)
class ReleaseEstimate:
    release_frame: int
    center_release: np.ndarray  # (3,)
    velocity_release: np.ndarray  # (3,)
    debug: Dict[str, object]


def fit_sphere_algebraic(points: np.ndarray) -> Optional[SphereFit]:
    """
    Fit a sphere to 3D points using a simple linear least squares formulation.

    Solves: x^2 + y^2 + z^2 = 2*c_x*x + 2*c_y*y + 2*c_z*z + d
    Then:
      center = (c_x, c_y, c_z)
      r^2 = ||center||^2 + d
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    m = int(points.shape[0])
    if m < 4:
        return None

    P = points.astype(np.float64)
    A = np.concatenate([2.0 * P, np.ones((m, 1), dtype=np.float64)], axis=1)  # (m, 4)
    b = np.sum(P * P, axis=1)  # (m,)

    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)  # (4,)
    except np.linalg.LinAlgError:
        return None

    c = x[:3]
    d = float(x[3])
    r2 = float(np.dot(c, c) + d)
    if not np.isfinite(r2):
        return None
    r = float(np.sqrt(max(r2, 0.0)))

    dist = np.linalg.norm(P - c[None, :], axis=1)
    resid = dist - r
    rms = float(np.sqrt(np.mean(resid * resid)))

    return SphereFit(center=c.astype(np.float64), radius=r, rms_residual=rms, n_points=m)


def fit_sphere_fixed_radius(points: np.ndarray, radius: float) -> Optional[SphereFit]:
    """
    Fit sphere center assuming a known radius.

    Uses linear least squares on:
      ||p - c||^2 = r^2  ->  2 p·c - s = ||p||^2 - r^2, where s = ||c||^2
    Unknowns: [c_x, c_y, c_z, s].
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    m = int(points.shape[0])
    if m < 4:
        return None
    if not np.isfinite(radius) or radius <= 0:
        return None

    P = points.astype(np.float64)
    A = np.concatenate([2.0 * P, -np.ones((m, 1), dtype=np.float64)], axis=1)  # (m,4)
    b = np.sum(P * P, axis=1) - float(radius) * float(radius)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    c = x[:3]
    dist = np.linalg.norm(P - c[None, :], axis=1)
    resid = dist - float(radius)
    rms = float(np.sqrt(np.mean(resid * resid)))
    return SphereFit(center=c.astype(np.float64), radius=float(radius), rms_residual=rms, n_points=m)


def fit_sphere_robust(points: np.ndarray, iters: int = 2, keep_frac: float = 0.7) -> Optional[SphereFit]:
    """Iteratively fit sphere and drop the worst residual points."""
    pts = points
    fit = None
    for _ in range(max(1, int(iters))):
        fit = fit_sphere_algebraic(pts)
        if fit is None:
            return None
        if pts.shape[0] < 6:
            return fit
        dist = np.linalg.norm(pts.astype(np.float64) - fit.center[None, :], axis=1)
        resid = np.abs(dist - fit.radius)
        k = max(4, int(np.ceil(pts.shape[0] * float(keep_frac))))
        keep_idx = np.argsort(resid)[:k]
        pts = pts[keep_idx]
    return fit


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def estimate_ball_center_series(
    timeseries: np.ndarray,
    point_names: Sequence[str],
    robust_iters: int = 2,
    min_points: int = 4,
    radius_fixed: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate ball center per frame by fitting a sphere to finger points.

    Returns:
      centers: (T, 3) float64 (NaN if fit fails)
      radii:   (T,) float64 (NaN if fit fails)
      rms:     (T,) float64 (NaN if fit fails)
    """
    T = timeseries.shape[0]
    centers = np.full((T, 3), np.nan, dtype=np.float64)
    radii = np.full((T,), np.nan, dtype=np.float64)
    rms = np.full((T,), np.nan, dtype=np.float64)

    # Preload point trajectories.
    trajs: List[np.ndarray] = []
    for name in point_names:
        try:
            trajs.append(get_keypoint_data(timeseries, name).astype(np.float64))
        except Exception:
            # Missing keypoint: skip it.
            continue

    if not trajs:
        return centers, radii, rms

    for t in range(T):
        pts = np.stack([tr[t] for tr in trajs], axis=0)  # (n, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < int(min_points):
            continue
        if radius_fixed is None:
            fit = fit_sphere_robust(pts, iters=robust_iters, keep_frac=0.7)
        else:
            # With a fixed radius, robust trimming can help reject outliers but we need an initial fit.
            fit = fit_sphere_fixed_radius(pts, float(radius_fixed))
            if fit is not None and robust_iters > 1 and pts.shape[0] >= 6:
                dist = np.linalg.norm(pts.astype(np.float64) - fit.center[None, :], axis=1)
                resid = np.abs(dist - float(radius_fixed))
                k = max(4, int(np.ceil(pts.shape[0] * 0.7)))
                keep_idx = np.argsort(resid)[:k]
                fit = fit_sphere_fixed_radius(pts[keep_idx], float(radius_fixed))
        if fit is None:
            continue
        centers[t] = fit.center
        radii[t] = fit.radius
        rms[t] = fit.rms_residual

    return centers, radii, rms


def detect_release_from_sphere_fit(
    radii: np.ndarray,
    rms: np.ndarray,
    start_search: int,
    end_search: int,
    k_mad: float = 4.0,
    min_run: int = 5,
) -> Optional[int]:
    """
    Detect release as the last frame in a stable "good-fit" run before the breakdown.

    Heuristic:
    - Estimate baseline sphere-fit quality from the early portion (where ball is likely held).
    - Mark frames as "good" when rms is low and radius is close to baseline.
    - Pick the last frame of the last sufficiently-long good run in the search window.
    """
    T = int(rms.shape[0])
    s0 = int(max(0, min(start_search, T - 1)))
    s1 = int(max(s0 + 1, min(end_search, T)))

    # Baseline from early frames (first third), restricted to finite entries.
    early = slice(0, max(10, T // 3))
    rms0 = rms[early]
    rad0 = radii[early]
    rms_med = float(np.nanmedian(rms0))
    rms_mad = _mad(rms0)
    rad_med = float(np.nanmedian(rad0))
    rad_mad = _mad(rad0)

    if not np.isfinite(rms_med) or not np.isfinite(rad_med):
        return None

    # Robust thresholds.
    rms_thr = rms_med + float(k_mad) * (rms_mad if np.isfinite(rms_mad) and rms_mad > 1e-9 else 0.0)
    # If MAD collapses, still cap rms_thr to something reasonable.
    rms_thr = float(min(rms_thr, rms_med + 0.20))

    rad_thr = float(k_mad) * (rad_mad if np.isfinite(rad_mad) and rad_mad > 1e-9 else 0.0)
    rad_thr = float(max(rad_thr, 0.10 * max(rad_med, 1e-3)))  # allow at least +-10%

    good = np.isfinite(rms) & np.isfinite(radii)
    good &= (rms <= rms_thr) & (np.abs(radii - rad_med) <= rad_thr)

    good[:s0] = False
    good[s1:] = False

    if not np.any(good):
        return None

    # Find last run of Trues with length >= min_run, return its last index.
    idx = np.where(good)[0]
    # Convert to runs.
    last_end = None
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        # close run
        if prev - run_start + 1 >= int(min_run):
            last_end = prev
        run_start = i
        prev = i
    # final run
    if prev - run_start + 1 >= int(min_run):
        last_end = prev

    return int(last_end) if last_end is not None else None


def detect_release_from_contact_constraints(
    centers: np.ndarray,
    rms: np.ndarray,
    wrist_xyz: np.ndarray,
    *,
    start_search: int,
    end_search: int,
    rms_thr: float,
    dist_min: float,
    dist_max: float,
    min_run: int = 3,
) -> Optional[int]:
    """
    Release = last frame in the window where:
      - sphere-fit rms is small
      - center is within a plausible distance to the wrist

    This is stricter than MAD-based detection and works better when sphere-fit
    remains numerically stable after release.
    """
    T = int(rms.shape[0])
    s0 = int(max(0, min(start_search, T - 1)))
    s1 = int(max(s0 + 1, min(end_search, T)))

    ok = np.isfinite(rms) & (rms <= float(rms_thr))
    ok &= np.isfinite(centers).all(axis=1) & np.isfinite(wrist_xyz).all(axis=1)
    if np.any(ok):
        dist = np.linalg.norm(centers - wrist_xyz.astype(np.float64), axis=1)
        ok &= (dist >= float(dist_min)) & (dist <= float(dist_max))

    ok[:s0] = False
    ok[s1:] = False
    if not np.any(ok):
        return None

    idx = np.where(ok)[0]
    last_end = None
    run_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        if prev - run_start + 1 >= int(min_run):
            last_end = prev
        run_start = i
        prev = i
    if prev - run_start + 1 >= int(min_run):
        last_end = prev
    return int(last_end) if last_end is not None else None


def estimate_release_from_fingers(
    timeseries: np.ndarray,
    point_names: Sequence[str],
    *,
    start_search: int,
    end_search: int,
    k_mad: float = 4.0,
    min_run: int = 5,
    robust_iters: int = 2,
    vel_window: int = 5,
    g_ft_s2: float = G_FT_S2,
) -> Optional[ReleaseEstimate]:
    centers, radii, rms = estimate_ball_center_series(
        timeseries,
        point_names=point_names,
        robust_iters=robust_iters,
        min_points=4,
    )
    rel = detect_release_from_sphere_fit(
        radii=radii,
        rms=rms,
        start_search=start_search,
        end_search=end_search,
        k_mad=k_mad,
        min_run=min_run,
    )
    if rel is None:
        return None

    # Estimate velocity via local linear fit on the last `vel_window` frames before rel (inclusive).
    w = int(max(3, vel_window))
    t0 = max(0, rel - w + 1)
    idx = np.arange(t0, rel + 1)
    C = centers[idx]
    ok = np.isfinite(C).all(axis=1)
    if ok.sum() < 3:
        return None

    idx = idx[ok]
    C = C[ok]
    tt = (idx.astype(np.float64) - float(rel)) * DT  # seconds relative to release
    # Fit C ~= p0 + v * t
    A = np.stack([np.ones_like(tt), tt], axis=1)  # (n,2)
    v = np.zeros(3, dtype=np.float64)
    p0 = np.zeros(3, dtype=np.float64)
    for ax in range(3):
        coef, *_ = np.linalg.lstsq(A, C[:, ax], rcond=None)  # [p0, v]
        p0[ax] = coef[0]
        v[ax] = coef[1]

    debug = {
        "rms_median_early": float(np.nanmedian(rms[: max(10, len(rms) // 3)])),
        "radii_median_early": float(np.nanmedian(radii[: max(10, len(radii) // 3)])),
        "rms_at_release": float(rms[rel]) if np.isfinite(rms[rel]) else None,
        "radius_at_release": float(radii[rel]) if np.isfinite(radii[rel]) else None,
        "vel_window": int(w),
        "k_mad": float(k_mad),
        "min_run": int(min_run),
        "g_ft_s2": float(g_ft_s2),
        "point_names": list(point_names),
    }
    return ReleaseEstimate(release_frame=rel, center_release=p0, velocity_release=v, debug=debug)


def entry_angle_at_rim_from_release(
    p0: np.ndarray,
    v0: np.ndarray,
    *,
    hoop_z_ft: float = HOOP_Z_FT,
    g_ft_s2: float = G_FT_S2,
) -> Optional[float]:
    """
    Compute entry angle at the plane z=hoop_z_ft for ballistic flight (no drag).

    Returns angle in degrees (positive), or None if the rim plane isn't reached.
    """
    z0 = float(p0[2])
    vz0 = float(v0[2])
    # Solve z0 + vz0*t - 0.5*g*t^2 = hoop_z
    a = -0.5 * float(g_ft_s2)
    b = float(vz0)
    c = float(z0 - hoop_z_ft)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)
    # Pick the later positive time (descending if two crossings exist).
    ts = [t for t in (t1, t2) if t > 1e-6 and np.isfinite(t)]
    if not ts:
        return None
    t = float(max(ts))

    vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2]) - float(g_ft_s2) * t
    v_h = float(np.sqrt(vx * vx + vy * vy))
    if v_h < 1e-9:
        return None
    ang = float(np.degrees(np.arctan2(-vz, v_h)))
    return float(abs(ang))
