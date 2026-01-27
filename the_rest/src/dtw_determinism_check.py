#!/usr/bin/env python3
"""
DTW-based nearest-neighbor determinism check for `angle`.

Motivation:
- Euclidean distance on fixed-frame sequences can miss "same motion, shifted in time".
- DTW aligns sequences, so if angle were deterministic from some release-related signal
  up to time warps, DTW-NN should produce very small angle differences.

We keep this computationally feasible by:
1) building low-dimensional time series per shot (few signals)
2) downsampling to a modest length (e.g. 60)
3) using a two-stage search:
   - Euclidean kNN in flattened space to get candidate neighbors
   - compute DTW only to those candidates; pick DTW nearest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data_loader import load_all_as_arrays, get_keypoint_columns
from src.physics_features import init_keypoint_mapping, get_keypoint_data, compute_joint_angle


NUM_FRAMES = 240


def _nan_impute_per_dim(X: np.ndarray) -> np.ndarray:
    """Impute NaNs per-dimension with that dimension's median."""
    X = X.astype(np.float64, copy=True)
    for d in range(X.shape[1]):
        col = X[:, d]
        m = np.isfinite(col)
        if not np.any(m):
            X[:, d] = 0.0
            continue
        med = float(np.median(col[m]))
        col[~m] = med
        X[:, d] = col
    return X


def _zscore_per_dim(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=True)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def _downsample_linear(X: np.ndarray, n: int) -> np.ndarray:
    """Downsample (T,D) to (n,D) by linear interpolation."""
    T = X.shape[0]
    if T == n:
        return X.astype(np.float32, copy=False)
    if T < 2:
        return np.full((n, X.shape[1]), np.nan, dtype=np.float32)
    xi = np.linspace(0.0, float(T - 1), int(n))
    x = np.arange(T, dtype=np.float64)
    out = np.empty((int(n), X.shape[1]), dtype=np.float64)
    for d in range(X.shape[1]):
        out[:, d] = np.interp(xi, x, X[:, d].astype(np.float64))
    return out.astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int) -> float:
    """
    Sakoe-Chiba band DTW (squared euclidean per-step cost), returns sqrt(total_cost).
    a: (T,D), b: (T,D) float32/float64
    """
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    T1, D1 = a.shape
    T2, D2 = b.shape
    if D1 != D2:
        raise ValueError("dtw_distance requires same feature dimension")
    band = int(max(band, abs(T1 - T2)))
    INF = 1e30
    prev = np.full((T2 + 1,), INF, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, T1 + 1):
        curr = np.full((T2 + 1,), INF, dtype=np.float64)
        j0 = max(1, i - band)
        j1 = min(T2, i + band)
        ai = a[i - 1]
        for j in range(j0, j1 + 1):
            bj = b[j - 1]
            diff = ai - bj
            cost = float(np.dot(diff, diff))
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev = curr
    return float(np.sqrt(prev[T2]))


def build_signals(ts: np.ndarray, kind: str) -> np.ndarray:
    """
    Return (T,D) signals for one shot.
    Kinds are designed to be low-D and release-relevant.
    """
    if kind == "wrist_speed":
        w = get_keypoint_data(ts, "right_wrist").astype(np.float64)
        v = np.gradient(w, axis=0)
        s = np.linalg.norm(v, axis=1, keepdims=True)
        return s

    if kind == "wrist_speed_tips_speed":
        w = get_keypoint_data(ts, "right_wrist").astype(np.float64)
        v = np.gradient(w, axis=0)
        ws = np.linalg.norm(v, axis=1)
        idx_tip = get_keypoint_data(ts, "right_second_finger_distal").astype(np.float64)
        mid_tip = get_keypoint_data(ts, "right_third_finger_distal").astype(np.float64)
        ring_tip = get_keypoint_data(ts, "right_fourth_finger_distal").astype(np.float64)
        tips = (idx_tip + mid_tip + ring_tip) / 3.0
        tv = np.gradient(tips, axis=0)
        tspeed = np.linalg.norm(tv, axis=1)
        return np.stack([ws, tspeed], axis=1)

    if kind == "angles_elbow_wrist_index":
        rs = get_keypoint_data(ts, "right_shoulder").astype(np.float64)
        re = get_keypoint_data(ts, "right_elbow").astype(np.float64)
        rw = get_keypoint_data(ts, "right_wrist").astype(np.float64)
        r_index_mcp = get_keypoint_data(ts, "right_second_finger_mcp").astype(np.float64)
        r_index_tip = get_keypoint_data(ts, "right_second_finger_distal").astype(np.float64)
        elbow = compute_joint_angle(rs, re, rw)
        wrist = compute_joint_angle(re, rw, r_index_mcp)
        index_mcp = compute_joint_angle(rw, r_index_mcp, r_index_tip)
        return np.stack([elbow, wrist, index_mcp], axis=1)

    if kind == "angles_plus_wrist_speed":
        ang = build_signals(ts, "angles_elbow_wrist_index")
        ws = build_signals(ts, "wrist_speed")
        return np.concatenate([ang, ws], axis=1)

    raise ValueError(f"Unknown signal kind: {kind}")


@dataclass(frozen=True)
class DTWResult:
    kind: str
    n: int
    band: int
    downsample: int
    k_candidates: int
    nn1_abs_dangle_median: float
    nn1_abs_dangle_p95: float
    pct_le_0p1: float
    pct_le_0p25: float
    pct_gt_1: float


def dtw_nn_eval(
    X: np.ndarray,
    angles: np.ndarray,
    participants: np.ndarray,
    *,
    kind: str,
    downsample: int,
    band: int,
    k_candidates: int,
    within_participant: bool,
) -> Tuple[DTWResult, pd.DataFrame]:
    """Compute DTW-NN mapping using candidate pruning; also return a pairs table."""
    N = X.shape[0]
    series: List[np.ndarray] = []
    for i in range(N):
        s = build_signals(X[i], kind)
        s = _nan_impute_per_dim(s)
        s = _downsample_linear(s, int(downsample))
        s = _nan_impute_per_dim(s)
        s = _zscore_per_dim(s)
        series.append(s.astype(np.float32))

    # Candidate stage: Euclidean kNN in flattened series.
    Xflat = np.stack([s.reshape(-1) for s in series], axis=0)
    # Global zscore for candidate search.
    mu = np.mean(Xflat, axis=0, keepdims=True)
    sd = np.std(Xflat, axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xcand = ((Xflat - mu) / sd).astype(np.float32)

    if within_participant:
        # Fit separate neighbor indexes per participant to ensure candidates come from same player.
        idx = np.full((N, min(int(k_candidates) + 1, N)), -1, dtype=np.int32)
        for pid in np.unique(participants):
            m = participants == pid
            if m.sum() < 2:
                continue
            Xp = Xcand[m]
            nn = NearestNeighbors(n_neighbors=min(int(k_candidates) + 1, Xp.shape[0]), metric="euclidean", n_jobs=1)
            nn.fit(Xp)
            _d, idx_p = nn.kneighbors(Xp, return_distance=True)
            # Map local indices back to global.
            glob = np.where(m)[0]
            idx[m, : idx_p.shape[1]] = glob[idx_p]
    else:
        nn = NearestNeighbors(n_neighbors=min(int(k_candidates) + 1, N), metric="euclidean", n_jobs=1)
        nn.fit(Xcand)
        _d, idx = nn.kneighbors(Xcand, return_distance=True)

    nn1 = np.full((N,), -1, dtype=np.int32)
    nn1_dist = np.full((N,), np.nan, dtype=np.float64)
    pairs = []

    for i in range(N):
        best_j = None
        best_d = float("inf")
        for j in idx[i, 1:]:
            j = int(j)
            if j < 0:
                continue
            d = dtw_distance(series[i], series[j], band=int(band))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            continue
        nn1[i] = int(best_j)
        nn1_dist[i] = float(best_d)
        pairs.append((i, int(best_j), float(best_d), float(abs(angles[i] - angles[int(best_j)]))))

    ok = nn1 >= 0
    nn1_dangle = np.abs(angles[ok] - angles[nn1[ok]])

    res = DTWResult(
        kind=kind,
        n=int(ok.sum()),
        band=int(band),
        downsample=int(downsample),
        k_candidates=int(k_candidates),
        nn1_abs_dangle_median=float(np.median(nn1_dangle)),
        nn1_abs_dangle_p95=float(np.quantile(nn1_dangle, 0.95)),
        pct_le_0p1=float(np.mean(nn1_dangle <= 0.1) * 100.0),
        pct_le_0p25=float(np.mean(nn1_dangle <= 0.25) * 100.0),
        pct_gt_1=float(np.mean(nn1_dangle > 1.0) * 100.0),
    )

    df_pairs = pd.DataFrame(pairs, columns=["i", "j", "dtw_dist", "abs_angle_diff"]).sort_values("dtw_dist")
    return res, df_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/dtw_determinism.csv")
    ap.add_argument("--k-candidates", type=int, default=40)
    ap.add_argument("--downsample", type=int, default=60)
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--within-participant", action="store_true", help="Restrict DTW NN search to same participant_id")
    args = ap.parse_args()

    X, y, meta = load_all_as_arrays(train=True)
    angles = y[:, 0].astype(np.float64)
    participants = meta["participant_id"].to_numpy()

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    kinds = [
        "wrist_speed",
        "wrist_speed_tips_speed",
        "angles_elbow_wrist_index",
        "angles_plus_wrist_speed",
    ]

    rows = []
    best_pairs = None
    best_name = None
    best_pct = -1.0
    for kind in kinds:
        res, pairs = dtw_nn_eval(
            X,
            angles,
            participants,
            kind=kind,
            downsample=int(args.downsample),
            band=int(args.band),
            k_candidates=int(args.k_candidates),
            within_participant=bool(args.within_participant),
        )
        rows.append(res.__dict__)
        if best_pairs is None or res.pct_le_0p1 > best_pct:
            best_pairs = pairs
            best_name = kind
            best_pct = float(res.pct_le_0p1)

    df = pd.DataFrame(rows).sort_values(["pct_le_0p1", "nn1_abs_dangle_median"], ascending=[False, True])
    df.to_csv(args.out, index=False)

    # Write pairs for the best kind.
    pairs_out = args.out.replace(".csv", f"_{best_name}_pairs.csv")
    assert best_pairs is not None
    best_pairs.to_csv(pairs_out, index=False)

    print("DTW determinism check (NN inconsistency)")
    print(f"- wrote: {args.out}")
    print(f"- wrote: {pairs_out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
