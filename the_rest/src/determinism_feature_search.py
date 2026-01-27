#!/usr/bin/env python3
"""
Search for a (near-)deterministic representation of `angle` from kinematics.

We do this without fitting a flexible predictor by using a nearest-neighbor
inconsistency test:
  - build a feature representation per shot
  - find each shot's nearest neighbor in that feature space
  - if angle is (approximately) a deterministic function of the features,
    then nearest neighbors should have very similar angles (e.g., <= 0.1 deg)

If no representation yields small NN angle differences, that is evidence that
0.1 deg accuracy is not achievable from these inputs alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data_loader import load_all_as_arrays, get_keypoint_columns
from src.physics_features import init_keypoint_mapping, get_keypoint_data, compute_joint_angle


NUM_FRAMES = 240


def _rotation_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _standardize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    bad = ~np.isfinite(X)
    if bad.any():
        X = X.copy()
        X[bad] = np.take(mean, np.where(bad)[1])
    X = (X - mean) / std
    return X.astype(np.float32)


def _finite_mask(X: np.ndarray) -> np.ndarray:
    return np.isfinite(X).all(axis=1)


def _release_idx_wrist_speed(ts: np.ndarray, start: int = 80, end: int = 170) -> int:
    wrist = get_keypoint_data(ts, "right_wrist").astype(np.float64)
    v = np.gradient(wrist, axis=0)
    s = np.linalg.norm(v, axis=1)
    a = int(np.clip(start, 0, NUM_FRAMES - 1))
    b = int(np.clip(end, a + 1, NUM_FRAMES))
    return a + int(np.nanargmax(s[a:b]))


def _release_idx_tips_vz(ts: np.ndarray, start: int = 80, end: int = 200) -> int:
    idx_tip = get_keypoint_data(ts, "right_second_finger_distal").astype(np.float64)
    mid_tip = get_keypoint_data(ts, "right_third_finger_distal").astype(np.float64)
    ring_tip = get_keypoint_data(ts, "right_fourth_finger_distal").astype(np.float64)
    tips = (idx_tip + mid_tip + ring_tip) / 3.0
    v = np.gradient(tips, axis=0)
    vz = v[:, 2]
    a = int(np.clip(start, 0, NUM_FRAMES - 1))
    b = int(np.clip(end, a + 1, NUM_FRAMES))
    return a + int(np.nanargmax(vz[a:b]))


def _kp_block(ts: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Return (T, 3*len(names)) stack of xyz for keypoints in order."""
    out = []
    for n in names:
        out.append(get_keypoint_data(ts, n).astype(np.float64))
    return np.concatenate(out, axis=1)


def _vel_block(ts_block: np.ndarray) -> np.ndarray:
    return np.gradient(ts_block, axis=0)


def _acc_block(ts_block: np.ndarray) -> np.ndarray:
    return np.gradient(np.gradient(ts_block, axis=0), axis=0)


def _apply_root_and_heading(
    block: np.ndarray,
    *,
    ts: np.ndarray,
    root_relative: bool,
    heading_normalize: bool,
    heading_frame: int,
) -> np.ndarray:
    """
    block: (T, 3*K) in world coords.
    If root_relative: subtract mid_hip per frame (xyz applied to each keypoint).
    If heading_normalize: rotate around z using shoulders at heading_frame.
    """
    T = block.shape[0]
    K = block.shape[1] // 3
    X = block.reshape(T, K, 3).copy()

    if root_relative:
        hip = get_keypoint_data(ts, "mid_hip").astype(np.float64)
        X -= hip[:, None, :]

    if heading_normalize:
        t = int(np.clip(heading_frame, 0, T - 1))
        ls = get_keypoint_data(ts, "left_shoulder").astype(np.float64)[t]
        rs = get_keypoint_data(ts, "right_shoulder").astype(np.float64)[t]
        v = rs - ls
        theta = float(np.arctan2(v[1], v[0]))
        R = _rotation_z(-theta)
        X = X @ R.T

    return X.reshape(T, 3 * K)


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def _hand_frame_features(ts: np.ndarray, *, side: str = "right") -> np.ndarray:
    """
    Build simple hand orientation features over time.

    Outputs per frame:
      - palm_normal (3): normal of (index - wrist) x (pinky - wrist)
      - palm_dir   (3): direction wrist -> middle fingertip
      - thumb_dir  (3): direction wrist -> thumb tip
    """
    if side not in ("right", "left"):
        raise ValueError("side must be right/left")

    wrist = get_keypoint_data(ts, f"{side}_wrist").astype(np.float64)
    index = get_keypoint_data(ts, f"{side}_second_finger_distal").astype(np.float64)
    pinky = get_keypoint_data(ts, f"{side}_fifth_finger_distal").astype(np.float64)
    middle = get_keypoint_data(ts, f"{side}_third_finger_distal").astype(np.float64)
    thumb = get_keypoint_data(ts, f"{side}_thumb").astype(np.float64)

    v_index = index - wrist
    v_pinky = pinky - wrist
    palm_normal = _unit(np.cross(v_index, v_pinky))
    palm_dir = _unit(middle - wrist)
    thumb_dir = _unit(thumb - wrist)

    return np.concatenate([palm_normal, palm_dir, thumb_dir], axis=1)  # (T, 9)


def _joint_angle_features(ts: np.ndarray) -> np.ndarray:
    """
    Compute a small set of joint-angle time series (degrees).

    Outputs per frame:
      - right_elbow_angle
      - right_wrist_angle (elbow-wrist-index_mcp)
      - right_index_mcp_flex (wrist-index_mcp-index_distal)
    """
    rs = get_keypoint_data(ts, "right_shoulder").astype(np.float64)
    re = get_keypoint_data(ts, "right_elbow").astype(np.float64)
    rw = get_keypoint_data(ts, "right_wrist").astype(np.float64)
    r_index_mcp = get_keypoint_data(ts, "right_second_finger_mcp").astype(np.float64)
    r_index_tip = get_keypoint_data(ts, "right_second_finger_distal").astype(np.float64)

    elbow = compute_joint_angle(rs, re, rw)  # (T,)
    wrist = compute_joint_angle(re, rw, r_index_mcp)
    index_mcp = compute_joint_angle(rw, r_index_mcp, r_index_tip)

    return np.stack([elbow, wrist, index_mcp], axis=1).astype(np.float64)  # (T, 3)


def _reparam_by_arclen(block: np.ndarray, n_samples: int = 60) -> np.ndarray:
    """
    Time-warp invariance via arc-length parameterization.

    Given a multivariate time series (T, D), we compute a pseudo-time s(t) based on
    cumulative L2 speed in this feature space and resample at fixed percentiles.
    """
    T = block.shape[0]
    if T < 2:
        return np.full((n_samples, block.shape[1]), np.nan, dtype=np.float64)
    v = np.linalg.norm(np.diff(block, axis=0), axis=1)  # (T-1,)
    s = np.concatenate([[0.0], np.cumsum(v)])  # (T,)
    if not np.isfinite(s).all() or float(s[-1]) < 1e-9:
        # Degenerate: fall back to uniform sampling.
        idx = np.linspace(0, T - 1, n_samples)
        return np.stack([np.interp(idx, np.arange(T), block[:, d]) for d in range(block.shape[1])], axis=1)

    q = np.linspace(0.0, float(s[-1]), int(n_samples))
    # Interpolate each dimension over s.
    out = np.empty((int(n_samples), block.shape[1]), dtype=np.float64)
    for d in range(block.shape[1]):
        out[:, d] = np.interp(q, s, block[:, d])
    return out


def build_repr_full(
    X: np.ndarray,
    *,
    keypoints: Sequence[str],
    use_vel: bool,
    use_acc: bool,
    root_relative: bool,
    heading_normalize: bool,
    heading_frame: int,
) -> np.ndarray:
    """Full sequence representation over all 240 frames."""
    N = X.shape[0]
    feats = []
    for i in range(N):
        ts = X[i]
        pos = _kp_block(ts, keypoints)
        pos = _apply_root_and_heading(
            pos,
            ts=ts,
            root_relative=root_relative,
            heading_normalize=heading_normalize,
            heading_frame=heading_frame,
        )
        parts = [pos]
        if use_vel:
            parts.append(_vel_block(pos))
        if use_acc:
            parts.append(_acc_block(pos))
        feats.append(np.concatenate(parts, axis=1).reshape(-1))
    return np.stack(feats, axis=0).astype(np.float32)


def build_repr_angles_and_handframe(
    X: np.ndarray,
    *,
    root_relative: bool,
    heading_normalize: bool,
    heading_frame: int,
    use_vel: bool,
    arclen_samples: Optional[int],
) -> np.ndarray:
    """Combine joint angles + (right) hand frame unit vectors; optionally arc-length reparameterize."""
    N = X.shape[0]
    feats = []
    for i in range(N):
        ts = X[i]
        ang = _joint_angle_features(ts)  # (T,3)
        hf = _hand_frame_features(ts, side="right")  # (T,9)

        # Hand frame is defined from points, so apply root/heading by transforming the underlying points.
        # For simplicity, we apply root/heading to the wrist/index/pinky/middle/thumb positions and then
        # recompute hand frame.
        if root_relative or heading_normalize:
            # Build a minimal ts slice containing just required points, applying transforms via a kp block.
            names = [
                "mid_hip",
                "left_shoulder",
                "right_shoulder",
                "right_wrist",
                "right_second_finger_distal",
                "right_fifth_finger_distal",
                "right_third_finger_distal",
                "right_thumb",
                "right_shoulder",
                "right_elbow",
                "right_second_finger_mcp",
            ]
            blk = _kp_block(ts, names)
            blk = _apply_root_and_heading(
                blk,
                ts=ts,
                root_relative=root_relative,
                heading_normalize=heading_normalize,
                heading_frame=heading_frame,
            )
            # Unpack transformed points.
            P = blk.reshape(NUM_FRAMES, len(names), 3)
            name_to_idx = {n: k for k, n in enumerate(names)}
            rw = P[:, name_to_idx["right_wrist"]]
            idx = P[:, name_to_idx["right_second_finger_distal"]]
            pky = P[:, name_to_idx["right_fifth_finger_distal"]]
            mid = P[:, name_to_idx["right_third_finger_distal"]]
            thb = P[:, name_to_idx["right_thumb"]]
            palm_normal = _unit(np.cross(idx - rw, pky - rw))
            palm_dir = _unit(mid - rw)
            thumb_dir = _unit(thb - rw)
            hf = np.concatenate([palm_normal, palm_dir, thumb_dir], axis=1)

            # Recompute angles from transformed points for consistency.
            rs = P[:, name_to_idx["right_shoulder"]]
            re = P[:, name_to_idx["right_elbow"]]
            r_index_mcp = P[:, name_to_idx["right_second_finger_mcp"]]
            elbow = compute_joint_angle(rs, re, rw)
            wrist_ang = compute_joint_angle(re, rw, r_index_mcp)
            index_mcp = compute_joint_angle(rw, r_index_mcp, idx)
            ang = np.stack([elbow, wrist_ang, index_mcp], axis=1)

        base = np.concatenate([ang, hf], axis=1)  # (T,12)
        parts = [base]
        if use_vel:
            parts.append(_vel_block(base))
        series = np.concatenate(parts, axis=1)
        if arclen_samples is not None:
            series = _reparam_by_arclen(series, n_samples=int(arclen_samples))
        feats.append(series.reshape(-1))
    return np.stack(feats, axis=0).astype(np.float32)


def build_repr_release_window(
    X: np.ndarray,
    *,
    keypoints: Sequence[str],
    release_method: str,
    window: int,
    use_vel: bool,
    use_acc: bool,
    root_relative: bool,
    heading_normalize: bool,
    heading_frame: int,
) -> np.ndarray:
    """Window around a per-shot release estimate; flatten."""
    N = X.shape[0]
    w = int(max(1, window))
    feats = []
    for i in range(N):
        ts = X[i]
        if release_method == "wrist_speed":
            r = _release_idx_wrist_speed(ts)
        elif release_method == "tips_vz":
            r = _release_idx_tips_vz(ts)
        else:
            raise ValueError(f"unknown release_method: {release_method}")
        a = max(0, r - w)
        b = min(NUM_FRAMES, r + w + 1)
        tsw = ts[a:b]
        pos = _kp_block(tsw, keypoints)
        pos = _apply_root_and_heading(
            pos,
            ts=tsw,
            root_relative=root_relative,
            heading_normalize=heading_normalize,
            heading_frame=heading_frame,
        )
        parts = [pos]
        if use_vel:
            parts.append(_vel_block(pos))
        if use_acc:
            parts.append(_acc_block(pos))
        feats.append(np.concatenate(parts, axis=1).reshape(-1))

    # Pad to constant length by right-padding with NaNs (then standardized/imputed).
    max_len = max(f.shape[0] for f in feats)
    Xf = np.full((N, max_len), np.nan, dtype=np.float64)
    for i, f in enumerate(feats):
        Xf[i, : f.shape[0]] = f
    return Xf.astype(np.float32)


def nn_inconsistency_metrics(Xf: np.ndarray, angles: np.ndarray, k: int = 1) -> Dict[str, float]:
    """Compute NN1 angle inconsistency metrics (after feature standardization)."""
    Xs = _standardize(Xf)
    ok = _finite_mask(Xs) & np.isfinite(angles)
    Xs = Xs[ok]
    a = angles[ok].astype(np.float64)
    if Xs.shape[0] < 10:
        return {"n": float(Xs.shape[0])}

    nn = NearestNeighbors(n_neighbors=min(k + 1, Xs.shape[0]), metric="euclidean", n_jobs=1)
    nn.fit(Xs)
    dists, idxs = nn.kneighbors(Xs, return_distance=True)
    nn1 = idxs[:, 1] if idxs.shape[1] > 1 else idxs[:, 0]
    nn1_dist = dists[:, 1] if dists.shape[1] > 1 else dists[:, 0]
    nn1_dangle = np.abs(a - a[nn1])

    return {
        "n": float(Xs.shape[0]),
        "nn1_dist_p01": float(np.quantile(nn1_dist, 0.01)),
        "nn1_dist_median": float(np.median(nn1_dist)),
        "nn1_abs_dangle_median": float(np.median(nn1_dangle)),
        "nn1_abs_dangle_p90": float(np.quantile(nn1_dangle, 0.90)),
        "nn1_abs_dangle_p95": float(np.quantile(nn1_dangle, 0.95)),
        "pct_nn1_abs_dangle_le_0p1": float(np.mean(nn1_dangle <= 0.1) * 100.0),
        "pct_nn1_abs_dangle_le_0p25": float(np.mean(nn1_dangle <= 0.25) * 100.0),
        "pct_nn1_abs_dangle_gt_1": float(np.mean(nn1_dangle > 1.0) * 100.0),
    }


def nn_inconsistency_metrics_within_groups(
    Xf: np.ndarray,
    angles: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, float]:
    """
    NN1 inconsistency where the nearest neighbor must come from the same group.

    This matches the competition setting where participants repeat in train/test.
    """
    Xs = _standardize(Xf)
    ok = _finite_mask(Xs) & np.isfinite(angles) & np.isfinite(groups)
    Xs = Xs[ok]
    a = angles[ok].astype(np.float64)
    g = groups[ok].astype(np.int64)
    if Xs.shape[0] < 10:
        return {"n": float(Xs.shape[0])}

    nn1_dangle = []
    nn1_dist = []
    n_total = 0
    for gid in np.unique(g):
        m = g == gid
        if m.sum() < 2:
            continue
        Xg = Xs[m]
        ag = a[m]
        nn = NearestNeighbors(n_neighbors=min(2, Xg.shape[0]), metric="euclidean", n_jobs=1)
        nn.fit(Xg)
        dists, idxs = nn.kneighbors(Xg, return_distance=True)
        if idxs.shape[1] < 2:
            continue
        j = idxs[:, 1]
        da = np.abs(ag - ag[j])
        nn1_dangle.append(da)
        nn1_dist.append(dists[:, 1])
        n_total += int(m.sum())

    if not nn1_dangle:
        return {"n": float(0)}
    nn1_dangle = np.concatenate(nn1_dangle)
    nn1_dist = np.concatenate(nn1_dist)

    return {
        "n": float(n_total),
        "nn1_dist_p01": float(np.quantile(nn1_dist, 0.01)),
        "nn1_dist_median": float(np.median(nn1_dist)),
        "nn1_abs_dangle_median": float(np.median(nn1_dangle)),
        "nn1_abs_dangle_p90": float(np.quantile(nn1_dangle, 0.90)),
        "nn1_abs_dangle_p95": float(np.quantile(nn1_dangle, 0.95)),
        "pct_nn1_abs_dangle_le_0p1": float(np.mean(nn1_dangle <= 0.1) * 100.0),
        "pct_nn1_abs_dangle_le_0p25": float(np.mean(nn1_dangle <= 0.25) * 100.0),
        "pct_nn1_abs_dangle_gt_1": float(np.mean(nn1_dangle > 1.0) * 100.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/determinism_search.csv")
    ap.add_argument("--heading-frame", type=int, default=90)
    ap.add_argument("--within-participant", action="store_true", help="Restrict NN search to same participant_id")
    args = ap.parse_args()

    X, y, meta = load_all_as_arrays(train=True)
    angles = y[:, 0].astype(np.float64)
    groups = meta["participant_id"].to_numpy()

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Keypoint subsets: broad -> targeted.
    kps_all = [c[:-2] for c in keypoint_cols if c.endswith("_x")]
    kps_right_arm = ["right_shoulder", "right_elbow", "right_wrist"]
    kps_right_hand = [
        "right_wrist",
        "right_thumb",
        "right_first_finger_distal",
        "right_second_finger_distal",
        "right_third_finger_distal",
        "right_fourth_finger_distal",
        "right_fifth_finger_distal",
        "right_second_finger_mcp",
        "right_third_finger_mcp",
        "right_fourth_finger_mcp",
        "right_fifth_finger_mcp",
    ]
    kps_both_hands = kps_right_hand + [
        "left_wrist",
        "left_thumb",
        "left_first_finger_distal",
        "left_second_finger_distal",
        "left_third_finger_distal",
        "left_fourth_finger_distal",
        "left_fifth_finger_distal",
    ]

    configs = []

    def add_full(name: str, keypoints: Sequence[str], **kw):
        configs.append(("full", name, dict(keypoints=keypoints, **kw)))

    def add_win(name: str, keypoints: Sequence[str], **kw):
        configs.append(("win", name, dict(keypoints=keypoints, **kw)))

    def add_custom(name: str, builder: Callable[[], np.ndarray]):
        configs.append(("custom", name, {"builder": builder}))

    for root_relative in (False, True):
        for heading_normalize in (False, True):
            add_full(
                f"all_pos_rr={root_relative}_hd={heading_normalize}",
                kps_all,
                use_vel=False,
                use_acc=False,
                root_relative=root_relative,
                heading_normalize=heading_normalize,
                heading_frame=int(args.heading_frame),
            )
            add_full(
                f"right_hand_posvel_rr={root_relative}_hd={heading_normalize}",
                kps_right_hand,
                use_vel=True,
                use_acc=False,
                root_relative=root_relative,
                heading_normalize=heading_normalize,
                heading_frame=int(args.heading_frame),
            )
            add_full(
                f"both_hands_posvel_rr={root_relative}_hd={heading_normalize}",
                kps_both_hands,
                use_vel=True,
                use_acc=False,
                root_relative=root_relative,
                heading_normalize=heading_normalize,
                heading_frame=int(args.heading_frame),
            )
            add_full(
                f"right_arm_posvelacc_rr={root_relative}_hd={heading_normalize}",
                kps_right_arm,
                use_vel=True,
                use_acc=True,
                root_relative=root_relative,
                heading_normalize=heading_normalize,
                heading_frame=int(args.heading_frame),
            )

    for release_method in ("wrist_speed", "tips_vz"):
        for window in (2, 4, 6, 8):
            add_win(
                f"right_hand_win{window}_{release_method}",
                kps_right_hand,
                release_method=release_method,
                window=window,
                use_vel=True,
                use_acc=True,
                root_relative=True,
                heading_normalize=True,
                heading_frame=int(args.heading_frame),
            )
            add_win(
                f"both_hands_win{window}_{release_method}",
                kps_both_hands,
                release_method=release_method,
                window=window,
                use_vel=True,
                use_acc=True,
                root_relative=True,
                heading_normalize=True,
                heading_frame=int(args.heading_frame),
            )

    # Joint angles + hand orientation; and an arc-length reparameterized version (DTW-like).
    for root_relative in (False, True):
        for heading_normalize in (False, True):
            add_custom(
                f"angles_handframe_rr={root_relative}_hd={heading_normalize}",
                builder=lambda rr=root_relative, hd=heading_normalize: build_repr_angles_and_handframe(
                    X,
                    root_relative=rr,
                    heading_normalize=hd,
                    heading_frame=int(args.heading_frame),
                    use_vel=True,
                    arclen_samples=None,
                ),
            )
            add_custom(
                f"angles_handframe_arclen60_rr={root_relative}_hd={heading_normalize}",
                builder=lambda rr=root_relative, hd=heading_normalize: build_repr_angles_and_handframe(
                    X,
                    root_relative=rr,
                    heading_normalize=hd,
                    heading_frame=int(args.heading_frame),
                    use_vel=True,
                    arclen_samples=60,
                ),
            )

    rows = []
    for kind, name, kw in configs:
        if kind == "full":
            Xf = build_repr_full(X, **kw)
        elif kind == "win":
            Xf = build_repr_release_window(X, **kw)
        else:
            Xf = kw["builder"]()
        if args.within_participant:
            met = nn_inconsistency_metrics_within_groups(Xf, angles, groups)
        else:
            met = nn_inconsistency_metrics(Xf, angles, k=1)
        rows.append({"name": name, "kind": kind, **met})

    df = pd.DataFrame(rows).sort_values(
        ["pct_nn1_abs_dangle_le_0p1", "nn1_abs_dangle_median"], ascending=[False, True]
    )
    df.to_csv(args.out, index=False)

    print("Determinism feature search (NN inconsistency)")
    print(f"- wrote: {args.out}")
    top = df.head(10)[
        [
            "name",
            "kind",
            "n",
            "nn1_abs_dangle_median",
            "nn1_abs_dangle_p95",
            "pct_nn1_abs_dangle_le_0p1",
            "pct_nn1_abs_dangle_le_0p25",
        ]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
