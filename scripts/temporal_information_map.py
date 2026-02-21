"""
Temporal Information Map: where does predictive signal live across 240 frames?

For each player x target x frame, compute LOO MSE using features from THAT frame.
This produces a heatmap showing which temporal regions contain information.

Key question: is the signal concentrated at one frame (current assumption) or
spread across multiple disjoint phases (wind-up, load, release, follow-through)?

If multi-phase, we can extract features from multiple windows and concatenate
into a richer representation for the Ridge pipeline.

Data format: each row is one shot. Each keypoint column (e.g. nose_x) is a JSON
array of 240 values (one per frame at 60fps).
"""

from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

HOOP = np.array([5.25, -25.0, 10.0])
N_FRAMES = 240
TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    arr = np.array(json.loads(s), dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def load_data():
    """Load train data: returns (X_3d, players, targets) where X_3d is (N, 240, n_kp, 3)."""
    df = pd.read_csv(DATA_DIR / "train.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]

    kp_names = []
    for col in kp_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    n_kp = len(kp_names)
    n = len(df)

    X_3d = np.zeros((n, N_FRAMES, n_kp, 3), dtype=np.float32)
    for idx, row in df.iterrows():
        for col_i, col in enumerate(kp_cols):
            arr = parse_array_string(row[col])
            X_3d[idx, :, col_i // 3, col_i % 3] = arr
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx+1}/{n}...")

    players = df["participant_id"].values
    targets = {t: df[t].values for t in TARGETS}

    # Scale targets to [0,1] for comparable MSE
    import joblib
    scalers = {}
    y_scaled = {}
    for t in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{t}.pkl"
        if scaler_path.exists():
            scalers[t] = joblib.load(scaler_path)
            y_scaled[t] = scalers[t].transform(targets[t].reshape(-1, 1)).ravel()
        else:
            # Manual min-max if no scaler
            mn, mx = targets[t].min(), targets[t].max()
            y_scaled[t] = (targets[t] - mn) / (mx - mn + 1e-10)

    return X_3d, players, y_scaled, df


def extract_frame_features(X_3d, frame_idx):
    """Extract hoop-relative keypoint features at a specific frame.
    X_3d: (N, 240, n_kp, 3) -> returns (N, n_kp*3) hoop-relative features.
    """
    n = X_3d.shape[0]
    kp = X_3d[:, frame_idx, :, :]  # (N, n_kp, 3)

    # Hoop-relative
    hoop_rel = kp - HOOP[np.newaxis, np.newaxis, :]
    return hoop_rel.reshape(n, -1)  # (N, n_kp*3)


def fast_loo_mse(features, player_ids, target_values, bw_quantile=0.3):
    """
    Fast LOO using Nadaraya-Watson (kernel-weighted average).
    Per-player models. Returns global MSE.
    """
    n = len(target_values)
    predictions = np.full(n, np.nan)
    unique_players = np.unique(player_ids)

    for pid in unique_players:
        mask = player_ids == pid
        idx = np.where(mask)[0]
        X_p = features[mask]
        y_p = target_values[mask]
        n_p = len(y_p)

        if n_p < 3:
            predictions[idx] = np.mean(y_p)
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)

        D = pairwise_distances(X_scaled)
        dists_flat = D[np.triu_indices(n_p, k=1)]
        if len(dists_flat) == 0:
            predictions[idx] = np.mean(y_p)
            continue
        bw = np.quantile(dists_flat, bw_quantile)
        if bw < 1e-10:
            bw = 1.0

        weights = np.exp(-0.5 * (D / bw) ** 2)
        np.fill_diagonal(weights, 0)  # LOO

        w_sum = weights.sum(axis=1)
        w_sum = np.maximum(w_sum, 1e-10)
        preds = (weights @ y_p) / w_sum
        predictions[idx] = preds

    mse = np.mean((predictions - target_values) ** 2)
    return mse


def main():
    print("=" * 70)
    print("TEMPORAL INFORMATION MAP")
    print("Scanning all 240 frames to find where predictive signal lives")
    print("=" * 70)

    print("\nLoading data...")
    X_3d, players, targets, df = load_data()
    unique_players = np.unique(players)
    n_kp = X_3d.shape[2]
    print(f"  {len(df)} shots, {len(unique_players)} players, {n_kp} keypoints")

    # Scan every 5th frame (48 evaluations)
    frame_step = 5
    frames_to_scan = list(range(0, N_FRAMES, frame_step))
    n_scan = len(frames_to_scan)

    print(f"\nScanning {n_scan} frames (every {frame_step}th)...")
    print(f"Using Nadaraya-Watson LOO per player\n")

    # Results storage
    results_global = {t: np.zeros(n_scan) for t in TARGETS}
    results_per_player = {
        t: {pid: np.zeros(n_scan) for pid in unique_players}
        for t in TARGETS
    }

    for fi, frame_idx in enumerate(frames_to_scan):
        if fi % 10 == 0:
            print(f"  Frame {frame_idx:3d} ({fi+1}/{n_scan})...")

        feats = extract_frame_features(X_3d, frame_idx)

        for tname in TARGETS:
            y = targets[tname]

            # Global
            mse = fast_loo_mse(feats, players, y)
            results_global[tname][fi] = mse

            # Per-player
            for pid in unique_players:
                mask = players == pid
                if mask.sum() < 5:
                    results_per_player[tname][pid][fi] = np.nan
                    continue
                mse_p = fast_loo_mse(feats[mask], players[mask], y[mask])
                results_per_player[tname][pid][fi] = mse_p

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("GLOBAL: Best frames per target")
    print("=" * 70)

    for tname in TARGETS:
        mses = results_global[tname]
        best_fi = np.argmin(mses)
        best_frame = frames_to_scan[best_fi]
        best_mse = mses[best_fi]

        # Top 5 sorted
        sorted_idx = np.argsort(mses)
        print(f"\n{tname}: best_frame={best_frame} (MSE={best_mse:.6f})")
        print(f"  Top 5: {[(frames_to_scan[i], f'{mses[i]:.6f}') for i in sorted_idx[:5]]}")

        # Disjoint regions (>25 frames apart)
        used = set()
        regions = []
        for idx in sorted_idx:
            f = frames_to_scan[idx]
            if any(abs(f - u) < 25 for u in used):
                continue
            used.add(f)
            regions.append((f, mses[idx]))
            if len(regions) >= 5:
                break

        print(f"  Disjoint peaks (>25f apart):")
        for rank, (f, m) in enumerate(regions):
            gap = (m - best_mse) / best_mse * 100
            print(f"    #{rank+1}: frame {f:3d} MSE={m:.6f} (+{gap:.1f}%)")

    # Per-player
    print("\n" + "=" * 70)
    print("PER-PLAYER: Disjoint peaks per target")
    print("=" * 70)

    for tname in TARGETS:
        print(f"\n{tname}:")
        for pid in sorted(unique_players):
            mses = results_per_player[tname][pid]
            if np.all(np.isnan(mses)):
                continue

            sorted_idx = np.argsort(mses)
            used = set()
            regions = []
            for idx in sorted_idx:
                if np.isnan(mses[idx]):
                    continue
                f = frames_to_scan[idx]
                if any(abs(f - u) < 25 for u in used):
                    continue
                used.add(f)
                regions.append((f, mses[idx]))
                if len(regions) >= 4:
                    break

            best_mse = regions[0][1] if regions else 0
            region_str = " | ".join([
                f"f{f}({m:.6f}, +{(m-best_mse)/best_mse*100:.0f}%)"
                for f, m in regions
            ])
            print(f"  P{pid}: {region_str}")

    # Signal spread analysis
    print("\n" + "=" * 70)
    print("SIGNAL SPREAD: concentrated or multi-phase?")
    print("=" * 70)

    for tname in TARGETS:
        mses = results_global[tname]
        best_mse = np.min(mses)
        best_frame = frames_to_scan[np.argmin(mses)]

        within_5 = np.sum(mses < best_mse * 1.05)
        within_10 = np.sum(mses < best_mse * 1.10)
        within_20 = np.sum(mses < best_mse * 1.20)

        # Secondary peak far from primary
        far_mask = np.array([abs(frames_to_scan[i] - best_frame) > 40 for i in range(n_scan)])
        if far_mask.any():
            sec_mses = mses.copy()
            sec_mses[~far_mask] = np.inf
            sec_fi = np.argmin(sec_mses)
            sec_frame = frames_to_scan[sec_fi]
            sec_gap = (mses[sec_fi] - best_mse) / best_mse * 100
        else:
            sec_frame, sec_gap = -1, 999

        print(f"\n{tname}: primary=f{best_frame}")
        print(f"  Within 5%:  {within_5}/{n_scan}")
        print(f"  Within 10%: {within_10}/{n_scan}")
        print(f"  Within 20%: {within_20}/{n_scan}")
        if sec_frame >= 0:
            print(f"  Secondary peak: f{sec_frame} (+{sec_gap:.1f}% above best)")
            if sec_gap < 5:
                print(f"  >>> STRONG multi-phase: secondary nearly as good as primary")
            elif sec_gap < 15:
                print(f"  >>> MODERATE multi-phase signal")
            else:
                print(f"  >>> Concentrated near primary")

    # Multi-frame combination test
    print("\n" + "=" * 70)
    print("COMBINATION TEST: can multiple frames beat single best?")
    print("=" * 70)

    for tname in TARGETS:
        y = targets[tname]
        mses = results_global[tname]
        best_fi = np.argmin(mses)
        best_frame = frames_to_scan[best_fi]
        best_mse = mses[best_fi]

        # Get top disjoint frames
        sorted_idx = np.argsort(mses)
        used = set()
        top_frames = []
        for idx in sorted_idx:
            f = frames_to_scan[idx]
            if any(abs(f - u) < 25 for u in used):
                continue
            used.add(f)
            top_frames.append(f)
            if len(top_frames) >= 4:
                break

        print(f"\n{tname}: top disjoint frames = {top_frames}")
        print(f"  Single best: f{best_frame} MSE={best_mse:.6f}")

        for n_combo in [2, 3, 4]:
            if n_combo > len(top_frames):
                break
            for combo in combinations(top_frames, n_combo):
                feats_list = [extract_frame_features(X_3d, f) for f in combo]
                combined = np.concatenate(feats_list, axis=1)
                mse = fast_loo_mse(combined, players, y)
                delta = (mse - best_mse) / best_mse * 100
                marker = " <<< BETTER" if delta < -1 else (" < marginal" if delta < 0 else "")
                print(f"  {combo}: MSE={mse:.6f} ({delta:+.1f}%){marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
