"""
Mega Optimal Blend - Combine ALL diverse quality sources optimally.

We now have multiple quality diverse sources:
- Sub 3411: Ridge pipeline (our best standalone)
- Sub 3516: Improved CNN (CV 0.006451, different architecture)
- Sub 3550: GP Bayesian (diverse on LR)
- Sub 3582: LightGBM (diverse on depth/LR)

Strategy: Per-target optimal blend using known diversity structure.
For angle (all sources similar): mostly Ridge
For depth (CNN/LightGBM diverse): spread weight to diverse sources
For left_right (LightGBM very diverse): use more LightGBM

Then apply per-player CNN allocation on top (like Sub 3558).
"""

from __future__ import annotations

import fcntl
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"

TARGETS = ["angle", "depth", "left_right"]
SUB_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def get_next_submission_number():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    lock_path = SUBMISSION_DIR / ".submission_lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        nums = []
        for p in existing:
            try:
                nums.append(int(p.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
        next_num = max(nums) + 1 if nums else 1
        (SUBMISSION_DIR / f"submission_{next_num}.csv").touch()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return next_num


def save_submission(df, desc):
    num = get_next_submission_number()
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved submission_{num}: {desc}", flush=True)
    return num


def main():
    print("=" * 70, flush=True)
    print("Mega Optimal Blend", flush=True)
    print("=" * 70, flush=True)

    # Load all source submissions
    sources = {
        "ridge": 3411,
        "cnn": 3516,
        "gp": 3550,
        "lgbm": 3582,
    }

    subs = {}
    for name, num in sources.items():
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        subs[name] = pd.read_csv(path)
        print(f"  Loaded {name} (Sub {num}): {len(subs[name])} rows", flush=True)

    # Also load Sub3558 (current best) for comparison
    sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")

    # Load test data for player IDs
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    pids_test = test_df["participant_id"].values
    unique_pids = np.unique(pids_test)

    # Compute pairwise diversity
    print("\n--- Pairwise Diversity ---", flush=True)
    source_names = list(sources.keys())
    for tc in SUB_COLS:
        print(f"\n  {tc}:", flush=True)
        for i in range(len(source_names)):
            for j in range(i+1, len(source_names)):
                r, _ = pearsonr(subs[source_names[i]][tc], subs[source_names[j]][tc])
                print(f"    {source_names[i]} vs {source_names[j]}: r={r:.4f}", flush=True)

    # ============================================================
    # Strategy 1: Global per-target optimal weights
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Strategy 1: Global Per-Target Blend", flush=True)
    print(f"{'='*60}", flush=True)

    # Based on diversity analysis:
    # - angle: all sources highly correlated (r>0.96), stick with ridge
    # - depth: lgbm diverse (r=0.87), cnn somewhat diverse
    # - left_right: lgbm very diverse (r=0.79), gp diverse (r=0.80)

    per_target_weights = {
        "scaled_angle":      {"ridge": 0.90, "cnn": 0.05, "gp": 0.02, "lgbm": 0.03},
        "scaled_depth":      {"ridge": 0.82, "cnn": 0.06, "gp": 0.04, "lgbm": 0.08},
        "scaled_left_right": {"ridge": 0.78, "cnn": 0.05, "gp": 0.05, "lgbm": 0.12},
    }

    result_s1 = pd.DataFrame({"id": subs["ridge"]["id"]})
    for tc in SUB_COLS:
        vals = np.zeros(len(subs["ridge"]))
        w = per_target_weights[tc]
        for src, weight in w.items():
            vals += weight * subs[src][tc].values
        result_s1[tc] = np.clip(vals, 0, 1)
        print(f"  {tc}: weights={w}", flush=True)

    # Diversity vs Sub3558
    for tc in SUB_COLS:
        r, _ = pearsonr(result_s1[tc], sub3558[tc])
        print(f"  {tc} vs Sub3558: r={r:.4f}", flush=True)

    save_submission(result_s1, "Global per-target 4-source blend")

    # ============================================================
    # Strategy 2: Per-player-per-target allocation
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Strategy 2: Per-Player Per-Target Allocation", flush=True)
    print(f"{'='*60}", flush=True)

    # Load calibration data for player quality
    cal_files = sorted(OUTPUT_DIR.glob("per_player_calibration_*.json"))
    if cal_files:
        with open(cal_files[-1]) as f:
            calibrations = json.load(f)
    else:
        calibrations = None

    result_s2 = pd.DataFrame({"id": subs["ridge"]["id"]})
    for tidx, (tname, tc) in enumerate(zip(TARGETS, SUB_COLS)):
        vals = np.zeros(len(subs["ridge"]))

        for pid in unique_pids:
            te_mask = pids_test == pid
            te_idx = np.where(te_mask)[0]

            # Get player quality
            if calibrations and tname in calibrations:
                cal = calibrations[tname].get(str(int(pid)), {})
                mse = cal.get("mse", 0.008)
            else:
                mse = 0.008

            # Determine blend based on quality
            # Worse players get more diversity (more non-ridge sources)
            if mse < 0.006:
                # Good player: mostly ridge
                w = {"ridge": 0.92, "cnn": 0.04, "gp": 0.02, "lgbm": 0.02}
            elif mse < 0.010:
                # Medium
                w = {"ridge": 0.82, "cnn": 0.06, "gp": 0.04, "lgbm": 0.08}
            elif mse < 0.015:
                # Bad
                w = {"ridge": 0.72, "cnn": 0.10, "gp": 0.06, "lgbm": 0.12}
            else:
                # Very bad
                w = {"ridge": 0.60, "cnn": 0.15, "gp": 0.10, "lgbm": 0.15}

            blend = np.zeros(te_mask.sum())
            for src, weight in w.items():
                blend += weight * subs[src][tc].values[te_idx]

            vals[te_idx] = blend
            print(f"  P{pid} {tname}: MSE={mse:.4f}, w={dict((k,f'{v:.2f}') for k,v in w.items())}",
                  flush=True)

        result_s2[tc] = np.clip(vals, 0, 1)

    for tc in SUB_COLS:
        r, _ = pearsonr(result_s2[tc], sub3558[tc])
        print(f"  {tc} vs Sub3558: r={r:.4f}", flush=True)

    save_submission(result_s2, "Per-player-per-target 4-source blend")

    # ============================================================
    # Strategy 3: Grid search over per-target weights
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Strategy 3: Grid Search Per-Target Weights", flush=True)
    print(f"{'='*60}", flush=True)

    # Try many weight combos, evaluate by diversity from Sub3558
    # More diverse = likely to be a different (potentially better) prediction
    best_blends = {}
    for tc in SUB_COLS:
        best_div = 1.0  # lower r = more diverse
        best_w = None

        for w_ridge in np.arange(0.70, 0.96, 0.02):
            for w_cnn in np.arange(0.02, 0.15, 0.02):
                for w_lgbm in np.arange(0.02, 0.20, 0.02):
                    w_gp = 1.0 - w_ridge - w_cnn - w_lgbm
                    if w_gp < 0 or w_gp > 0.15:
                        continue

                    blend = (w_ridge * subs["ridge"][tc].values +
                             w_cnn * subs["cnn"][tc].values +
                             w_gp * subs["gp"][tc].values +
                             w_lgbm * subs["lgbm"][tc].values)

                    r, _ = pearsonr(blend, sub3558[tc])
                    # Want: diverse from 3558 but not TOO diverse (means noise)
                    # Sweet spot: r around 0.97-0.99 (slightly different from best)
                    score = abs(r - 0.98)  # minimize distance from 0.98

                    if score < best_div:
                        best_div = score
                        best_w = {"ridge": w_ridge, "cnn": w_cnn, "gp": w_gp, "lgbm": w_lgbm}
                        best_r = r

        print(f"  {tc}: best r={best_r:.4f}, weights={dict((k,f'{v:.2f}') for k,v in best_w.items())}",
              flush=True)
        best_blends[tc] = best_w

    result_s3 = pd.DataFrame({"id": subs["ridge"]["id"]})
    for tc in SUB_COLS:
        vals = np.zeros(len(subs["ridge"]))
        for src, w in best_blends[tc].items():
            vals += w * subs[src][tc].values
        result_s3[tc] = np.clip(vals, 0, 1)

    save_submission(result_s3, "Grid-searched per-target 4-source blend")

    # ============================================================
    # Strategy 4: Direct replacement of Sub3558 with 4-source version
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Strategy 4: Rebuild Sub3558 with LightGBM added", flush=True)
    print(f"{'='*60}", flush=True)

    # Sub3558 was: per-player CNN allocation on top of Sub3411
    # Now add LightGBM on top
    # Base: Sub3558, add small LightGBM signal

    result_s4 = sub3558.copy()
    for tc in SUB_COLS:
        # Add 3% LightGBM signal to Sub3558
        result_s4[tc] = np.clip(
            0.97 * sub3558[tc].values + 0.03 * subs["lgbm"][tc].values,
            0, 1
        )

    save_submission(result_s4, "Sub3558 + 3% LightGBM")

    # 5% LightGBM
    result_s4b = sub3558.copy()
    for tc in SUB_COLS:
        result_s4b[tc] = np.clip(
            0.95 * sub3558[tc].values + 0.05 * subs["lgbm"][tc].values,
            0, 1
        )
    save_submission(result_s4b, "Sub3558 + 5% LightGBM")

    # Per-target: more LightGBM for diverse targets
    result_s4c = sub3558.copy()
    lgbm_weights = {"scaled_angle": 0.02, "scaled_depth": 0.05, "scaled_left_right": 0.08}
    for tc in SUB_COLS:
        w = lgbm_weights[tc]
        result_s4c[tc] = np.clip(
            (1-w) * sub3558[tc].values + w * subs["lgbm"][tc].values,
            0, 1
        )
    save_submission(result_s4c, "Sub3558 + per-target LightGBM (2/5/8%)")

    # Also: Sub3558 + 2% GP for LR specifically
    result_s4d = sub3558.copy()
    result_s4d["scaled_left_right"] = np.clip(
        0.95 * sub3558["scaled_left_right"].values +
        0.03 * subs["lgbm"]["scaled_left_right"].values +
        0.02 * subs["gp"]["scaled_left_right"].values,
        0, 1
    )
    result_s4d["scaled_depth"] = np.clip(
        0.96 * sub3558["scaled_depth"].values +
        0.04 * subs["lgbm"]["scaled_depth"].values,
        0, 1
    )
    save_submission(result_s4d, "Sub3558 + targeted LGBM+GP for depth/LR")

    print(f"\nDone!", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"mega_optimal_blend_{ts}.json", "w") as f:
        json.dump({
            "per_target_weights": per_target_weights,
            "grid_search_weights": {k: {kk: float(vv) for kk, vv in v.items()}
                                    for k, v in best_blends.items()},
            "lgbm_per_target": lgbm_weights,
        }, f, indent=2)


if __name__ == "__main__":
    main()
