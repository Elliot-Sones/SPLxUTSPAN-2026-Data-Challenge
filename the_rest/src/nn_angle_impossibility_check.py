#!/usr/bin/env python3
"""
Nearest-neighbor "impossibility" sanity check for achieving ~0.1° angle accuracy.

If we can find pairs of shots that are extremely similar in kinematics space
(after reasonable invariances like translation + yaw/heading normalization),
but whose ground-truth `angle` differs by >> 0.1°, then predicting to 0.1°
from kinematics alone is likely impossible (missing variables / label noise).

This is not a formal proof, but it provides strong evidence either way.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data_loader import META_COLS, load_all_as_arrays, get_keypoint_columns
from src.physics_features import init_keypoint_mapping, get_keypoint_data


@dataclass(frozen=True)
class Pair:
    i: int
    j: int
    dist: float
    angle_i: float
    angle_j: float
    d_angle: float
    rms_per_coord: float
    max_abs_coord: float


def _rotation_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_representation(
    X: np.ndarray,
    *,
    root_relative: bool,
    heading_normalize: bool,
    heading_frame: int,
) -> np.ndarray:
    """
    Build a per-shot representation with optional translation (root) and yaw invariance.

    - root_relative: subtract mid_hip per frame from every keypoint
    - heading_normalize: rotate around z so left_shoulder->right_shoulder points along +x
      (using a single frame for stability)
    """
    N = X.shape[0]
    out = X.astype(np.float64).copy()

    if root_relative or heading_normalize:
        mid_hip = np.stack([get_keypoint_data(ts, "mid_hip") for ts in X], axis=0).astype(np.float64)
        if root_relative:
            out = out.reshape(N, 240, -1, 3)
            out = out - mid_hip[:, :, None, :]
            out = out.reshape(N, 240, -1)

        if heading_normalize:
            ls = np.stack([get_keypoint_data(ts, "left_shoulder") for ts in X], axis=0).astype(np.float64)
            rs = np.stack([get_keypoint_data(ts, "right_shoulder") for ts in X], axis=0).astype(np.float64)

            out_ = out.reshape(N, 240, -1, 3)
            for i in range(N):
                t = int(np.clip(heading_frame, 0, 239))
                v = rs[i, t] - ls[i, t]
                # Yaw angle in the x-y plane; vertical is z.
                theta = float(np.arctan2(v[1], v[0]))
                R = _rotation_z(-theta)
                out_[i] = out_[i] @ R.T
            out = out_.reshape(N, 240, -1)

    return out


def standardize_features(Xf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score per feature with NaN-imputation to feature mean."""
    X = Xf.astype(np.float64)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(mean, np.where(bad)[1])
    X = (X - mean) / std
    return X.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def compute_pairs(
    Xrep: np.ndarray,
    angles: np.ndarray,
    meta: pd.DataFrame,
    *,
    k_neighbors: int,
    max_pairs: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    N = Xrep.shape[0]
    Xf = Xrep.reshape(N, -1)
    Xs, _, _ = standardize_features(Xf)

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, N), metric="euclidean", n_jobs=1)
    nn.fit(Xs)
    dists, idxs = nn.kneighbors(Xs, return_distance=True)

    # Build candidate unique pairs from the neighbor lists (skip self at column 0).
    seen = set()
    pairs: List[Pair] = []
    for i in range(N):
        for rank in range(1, idxs.shape[1]):
            j = int(idxs[i, rank])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            dist = float(dists[i, rank])

            # Additional interpretable similarity stats in the *unstandardized* rep.
            diff = (Xf[i] - Xf[j]).astype(np.float64)
            m = np.isfinite(diff)
            if not np.any(m):
                continue
            diff = diff[m]
            rms = float(np.sqrt(np.mean(diff * diff)))
            mx = float(np.max(np.abs(diff)))

            ai = float(angles[i])
            aj = float(angles[j])
            da = float(abs(ai - aj))
            pairs.append(Pair(i=a, j=b, dist=dist, angle_i=ai, angle_j=aj, d_angle=da, rms_per_coord=rms, max_abs_coord=mx))

    # Sort by distance; keep top max_pairs.
    pairs.sort(key=lambda p: p.dist)
    pairs = pairs[: int(max_pairs)]

    rows = []
    for p in pairs:
        mi = meta.iloc[p.i]
        mj = meta.iloc[p.j]
        rows.append(
            {
                "i": p.i,
                "j": p.j,
                "id_i": mi["id"],
                "id_j": mj["id"],
                "shot_id_i": mi["shot_id"],
                "shot_id_j": mj["shot_id"],
                "participant_i": int(mi["participant_id"]),
                "participant_j": int(mj["participant_id"]),
                "angle_i": p.angle_i,
                "angle_j": p.angle_j,
                "abs_angle_diff": p.d_angle,
                "nn_dist_zscore": p.dist,
                "rms_coord_diff_ft": p.rms_per_coord,
                "max_abs_coord_diff_ft": p.max_abs_coord,
            }
        )

    df = pd.DataFrame(rows)

    # Summary stats: how inconsistent are angles among nearest neighbors?
    nn1 = idxs[:, 1] if idxs.shape[1] > 1 else idxs[:, 0]
    nn1_dist = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
    nn1_dangle = np.abs(angles - angles[nn1])
    summary = {
        "n": float(N),
        "nn1_dist_median": float(np.median(nn1_dist)),
        "nn1_dist_p01": float(np.quantile(nn1_dist, 0.01)),
        "nn1_dist_p05": float(np.quantile(nn1_dist, 0.05)),
        "nn1_abs_dangle_median": float(np.median(nn1_dangle)),
        "nn1_abs_dangle_p90": float(np.quantile(nn1_dangle, 0.90)),
        "nn1_abs_dangle_p95": float(np.quantile(nn1_dangle, 0.95)),
        "pct_nn1_abs_dangle_le_0p1": float(np.mean(nn1_dangle <= 0.1) * 100.0),
        "pct_nn1_abs_dangle_gt_0p5": float(np.mean(nn1_dangle > 0.5) * 100.0),
        "pct_nn1_abs_dangle_gt_1": float(np.mean(nn1_dangle > 1.0) * 100.0),
    }
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="Neighbors to scan for candidate pairs")
    ap.add_argument("--max-pairs", type=int, default=200, help="Keep this many closest unique pairs")
    ap.add_argument("--heading-frame", type=int, default=90, help="Frame used for heading normalization")
    ap.add_argument("--out", type=str, default="output/nn_angle_pairs.csv", help="Output CSV path")
    args = ap.parse_args()

    X, y, meta = load_all_as_arrays(train=True)
    angles = y[:, 0].astype(np.float64)

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    reps = [
        ("raw", dict(root_relative=False, heading_normalize=False)),
        ("root", dict(root_relative=True, heading_normalize=False)),
        ("root_heading", dict(root_relative=True, heading_normalize=True)),
    ]

    out_rows = []
    best_name = None
    best_summary = None
    best_df = None

    for name, kw in reps:
        Xrep = build_representation(X, heading_frame=int(args.heading_frame), **kw)
        df_pairs, summary = compute_pairs(
            Xrep,
            angles,
            meta,
            k_neighbors=int(args.k),
            max_pairs=int(args.max_pairs),
        )
        summary_row = dict(representation=name, **summary)
        out_rows.append(summary_row)

        # Pick the representation that yields the smallest nn1 distance p01
        # (i.e., the strongest chance to find near-duplicates).
        if best_summary is None or summary["nn1_dist_p01"] < best_summary["nn1_dist_p01"]:
            best_summary = summary
            best_name = name
            best_df = df_pairs

    assert best_df is not None

    # Write the closest-pair table (for the "best" representation).
    best_df.to_csv(args.out, index=False)

    # Also write the summary table.
    summary_path = args.out.replace(".csv", "_summary.csv")
    pd.DataFrame(out_rows).to_csv(summary_path, index=False)

    print("Nearest-neighbor impossibility check (train.csv)")
    print(f"- Wrote closest pairs: {args.out}")
    print(f"- Wrote summary:       {summary_path}")
    print(f"- Best rep (smallest 1% nn distance): {best_name}")
    # Print key summary numbers for best rep.
    print(
        f"- NN1 abs dAngle: median={best_summary['nn1_abs_dangle_median']:.3f} deg, "
        f"p95={best_summary['nn1_abs_dangle_p95']:.3f} deg, "
        f"%<=0.1 deg={best_summary['pct_nn1_abs_dangle_le_0p1']:.1f}%"
    )


if __name__ == "__main__":
    main()
