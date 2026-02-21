"""
Commitment Point Analysis: When is a free throw's outcome biomechanically determined?

For each player x target, compute the best single-feature Pearson |r| at every frame
from 60 to 220. The "commitment point" is when predictive information first reaches
80% of its peak - the moment at which the shot outcome is already decided.

Key question: Is depth committed to 200ms before release, while angle only commits
at release?
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]
TARGET_COL = {"angle": "scaled_angle", "depth": "scaled_depth", "left_right": "scaled_left_right"}
RAW_COL = {"angle": "angle", "depth": "depth", "left_right": "left_right"}

RELEASE_FRAME = 153  # nominal release frame
FRAMES = list(range(60, 221, 3))  # every 3 frames = 50ms at 60fps

KEYPOINTS_BASE = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
COORDS = ["x", "y", "z"]


def load_trajectories(df):
    """Load full 240-frame trajectories from train.csv."""
    n = len(df)
    kp_cols = []
    for kp in KEYPOINTS_BASE:
        for c in COORDS:
            col = f"{kp}_{c}"
            if col in df.columns:
                kp_cols.append(col)

    print(f"  Loading {len(kp_cols)} trajectory columns for {n} shots...", flush=True)
    # Shape: (n_shots, 240, n_kp_cols)
    trajs = np.zeros((n, 240, len(kp_cols)), dtype=np.float32)
    for j, col in enumerate(kp_cols):
        for i, val in enumerate(df[col]):
            if pd.isna(val):
                trajs[i, :, j] = np.nan
            else:
                arr = json.loads(str(val).replace("nan", "null"))
                trajs[i, :, j] = np.nan_to_num(arr, nan=np.nan)

    return trajs, kp_cols


def get_features_at_frame(trajs, frame, kp_cols):
    """Extract hoop-relative positions at a given frame. Returns (n_shots, n_features)."""
    f = int(np.clip(frame, 0, 239))
    feats = trajs[:, f, :].copy()  # (n_shots, n_kp_cols)

    # Hip-center the coordinates
    left_hip_x_idx = next((j for j, c in enumerate(kp_cols) if c == "left_hip_x"), None)
    right_hip_x_idx = next((j for j, c in enumerate(kp_cols) if c == "right_hip_x"), None)
    left_hip_y_idx = next((j for j, c in enumerate(kp_cols) if c == "left_hip_y"), None)
    right_hip_y_idx = next((j for j, c in enumerate(kp_cols) if c == "right_hip_y"), None)
    left_hip_z_idx = next((j for j, c in enumerate(kp_cols) if c == "left_hip_z"), None)
    right_hip_z_idx = next((j for j, c in enumerate(kp_cols) if c == "right_hip_z"), None)

    if all(idx is not None for idx in [left_hip_x_idx, right_hip_x_idx]):
        cx = (feats[:, left_hip_x_idx] + feats[:, right_hip_x_idx]) / 2
        cy = (feats[:, left_hip_y_idx] + feats[:, right_hip_y_idx]) / 2
        cz = (feats[:, left_hip_z_idx] + feats[:, right_hip_z_idx]) / 2
        for j, col in enumerate(kp_cols):
            if col.endswith("_x"):
                feats[:, j] -= cx
            elif col.endswith("_y"):
                feats[:, j] -= cy
            elif col.endswith("_z"):
                feats[:, j] -= cz

    return feats


def best_r_at_frame(feats, y, min_n=15):
    """Return max |r| across all features."""
    if len(y) < min_n:
        return 0.0, -1
    best = 0.0
    best_idx = -1
    for j in range(feats.shape[1]):
        x = feats[:, j]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n:
            continue
        try:
            r, _ = pearsonr(x[mask], y[mask])
            if abs(r) > best:
                best = abs(r)
                best_idx = j
        except Exception:
            pass
    return best, best_idx


def main():
    print("Loading data...", flush=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    import joblib
    scalers = {}
    for t in TARGETS:
        try:
            scalers[t] = joblib.load(DATA_DIR / f"scaler_{t}.pkl")
        except Exception:
            scalers[t] = None

    players = sorted(train["participant_id"].unique())
    print(f"Players: {players}", flush=True)
    print(f"Frames to test: {len(FRAMES)} from {FRAMES[0]} to {FRAMES[-1]}", flush=True)

    results = {}

    for player in players:
        pdata = train[train["participant_id"] == player].copy().reset_index(drop=True)
        n = len(pdata)
        print(f"\n--- Player {player} (n={n}) ---", flush=True)

        trajs, kp_cols = load_trajectories(pdata)
        results[str(player)] = {}

        for tname in TARGETS:
            # Get scaled targets
            raw_col = RAW_COL[tname]
            if raw_col not in pdata.columns:
                continue
            raw_y = pdata[raw_col].values.astype(float)
            if scalers.get(tname) is not None:
                y = scalers[tname].transform(raw_y.reshape(-1, 1))[:, 0]
            else:
                # Normalize manually
                y = (raw_y - raw_y.min()) / (raw_y.max() - raw_y.min() + 1e-8)

            frame_r = []
            for frame in FRAMES:
                feats = get_features_at_frame(trajs, frame, kp_cols)
                r, _ = best_r_at_frame(feats, y)
                frame_r.append(r)

            max_r = max(frame_r) if frame_r else 0.0
            peak_frame_idx = frame_r.index(max_r)
            peak_frame = FRAMES[peak_frame_idx]

            # 80% commitment point - first frame where r >= 0.8 * peak
            threshold = 0.8 * max_r
            commit_frame = FRAMES[-1]
            for i, r in enumerate(frame_r):
                if r >= threshold:
                    commit_frame = FRAMES[i]
                    break

            # 50% commitment point
            threshold50 = 0.5 * max_r
            commit_frame50 = FRAMES[-1]
            for i, r in enumerate(frame_r):
                if r >= threshold50:
                    commit_frame50 = FRAMES[i]
                    break

            ms_to_peak = (RELEASE_FRAME - peak_frame) * (1000 / 60)
            ms_commit80 = (RELEASE_FRAME - commit_frame) * (1000 / 60)
            ms_commit50 = (RELEASE_FRAME - commit_frame50) * (1000 / 60)

            results[str(player)][tname] = {
                "frames": FRAMES,
                "r_curve": [round(r, 4) for r in frame_r],
                "peak_r": round(max_r, 4),
                "peak_frame": peak_frame,
                "peak_ms_before_release": round(ms_to_peak, 1),
                "commit_frame_80pct": commit_frame,
                "commit_ms_before_release_80pct": round(ms_commit80, 1),
                "commit_frame_50pct": commit_frame50,
                "commit_ms_before_release_50pct": round(ms_commit50, 1),
            }

            direction = "before" if ms_to_peak > 0 else "AFTER"
            print(f"  {tname}: peak r={max_r:.3f} at frame {peak_frame} "
                  f"({abs(ms_to_peak):.0f}ms {direction} release), "
                  f"80% commit at {abs(ms_commit80):.0f}ms {'before' if ms_commit80 > 0 else 'after'} release",
                  flush=True)

    # Summary
    print("\n" + "="*60, flush=True)
    print("SUMMARY: Commitment Points Across All Players", flush=True)
    print("="*60, flush=True)
    print(f"{'Target':<12} {'Peak r':>8} {'Peak (ms)':>12} {'80% (ms)':>12} {'50% (ms)':>12}", flush=True)
    print("-"*60, flush=True)

    for tname in TARGETS:
        peak_rs = []
        peak_mss = []
        commit80s = []
        commit50s = []
        for player in players:
            d = results.get(str(player), {}).get(tname, {})
            if d:
                peak_rs.append(d["peak_r"])
                peak_mss.append(d["peak_ms_before_release"])
                commit80s.append(d["commit_ms_before_release_80pct"])
                commit50s.append(d["commit_ms_before_release_50pct"])
        if peak_rs:
            print(f"{tname:<12} {np.mean(peak_rs):>8.3f} {np.mean(peak_mss):>11.0f}ms "
                  f"{np.mean(commit80s):>11.0f}ms {np.mean(commit50s):>11.0f}ms", flush=True)

    # Per-player detail
    print("\nPer-player detail:", flush=True)
    for tname in TARGETS:
        print(f"\n  {tname.upper()}:", flush=True)
        for player in players:
            d = results.get(str(player), {}).get(tname, {})
            if d:
                print(f"    P{player}: peak r={d['peak_r']:.3f} at "
                      f"{abs(d['peak_ms_before_release']):.0f}ms "
                      f"{'before' if d['peak_ms_before_release'] > 0 else 'AFTER'} release, "
                      f"80% commit at {abs(d['commit_ms_before_release_80pct']):.0f}ms "
                      f"{'before' if d['commit_ms_before_release_80pct'] > 0 else 'AFTER'} release",
                      flush=True)

    print("\n" + "="*60, flush=True)
    print("HEADLINE FINDING", flush=True)
    print("="*60, flush=True)

    for tname in TARGETS:
        commit80s = [results[str(p)][tname]["commit_ms_before_release_80pct"]
                     for p in players if tname in results.get(str(p), {})]
        peak_mss = [results[str(p)][tname]["peak_ms_before_release"]
                    for p in players if tname in results.get(str(p), {})]
        peak_rs = [results[str(p)][tname]["peak_r"]
                   for p in players if tname in results.get(str(p), {})]
        if commit80s:
            print(f"{tname}: outcome 80% committed at {np.mean(commit80s):.0f}ms before release "
                  f"(peak r={np.mean(peak_rs):.3f} at {np.mean(peak_mss):.0f}ms before release)",
                  flush=True)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = OUTPUT_DIR / f"commitment_point_analysis_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}", flush=True)

    # Write research file
    write_research(results, players)


def write_research(results, players):
    lines = ["# Commitment Point Analysis: When Is a Free Throw Outcome Determined?\n",
             "\n**Date**: 2026-02-20\n",
             "**Script**: scripts/commitment_point_analysis.py\n\n",
             "## Method\n\n",
             "For each player x target, compute max |Pearson r| between any single keypoint "
             "position (hip-centered, 33 keypoints x 3 coords = 99 features) and the shot "
             "outcome at every frame from 60 to 220 (3-frame steps = 50ms resolution at 60fps). "
             "The 'commitment point' is when this best-r first reaches 80% of its session peak.\n\n",
             f"Release frame: {RELEASE_FRAME} (frame 153 = t=0)\n\n",
             "## Results\n\n",
             "### Summary Table\n\n",
             "| Target | Mean Peak r | Mean Peak (ms before release) | Mean 80% Commit (ms before release) |\n",
             "|--------|------------|-------------------------------|--------------------------------------|\n"]

    for tname in ["angle", "depth", "left_right"]:
        peak_rs = [results[str(p)][tname]["peak_r"] for p in players if tname in results.get(str(p), {})]
        peak_mss = [results[str(p)][tname]["peak_ms_before_release"] for p in players if tname in results.get(str(p), {})]
        commit80s = [results[str(p)][tname]["commit_ms_before_release_80pct"] for p in players if tname in results.get(str(p), {})]
        if peak_rs:
            lines.append(f"| {tname} | {np.mean(peak_rs):.3f} | {np.mean(peak_mss):.0f}ms | {np.mean(commit80s):.0f}ms |\n")

    lines += ["\n### Per-Player Detail\n\n"]
    for tname in ["angle", "depth", "left_right"]:
        lines.append(f"#### {tname.upper()}\n\n")
        lines.append("| Player | Peak r | Peak frame | ms before release | 80% commit ms |\n")
        lines.append("|--------|--------|------------|-------------------|---------------|\n")
        for p in players:
            d = results.get(str(p), {}).get(tname, {})
            if d:
                lines.append(f"| P{p} | {d['peak_r']:.3f} | {d['peak_frame']} | "
                              f"{d['peak_ms_before_release']:.0f}ms | "
                              f"{d['commit_ms_before_release_80pct']:.0f}ms |\n")
        lines.append("\n")

    outpath = PROJECT_DIR / "Research" / "COMMITMENT_POINT_ANALYSIS_2026-02-20.md"
    with open(outpath, "w") as f:
        f.writelines(lines)
    print(f"Research written to {outpath}", flush=True)


if __name__ == "__main__":
    main()
