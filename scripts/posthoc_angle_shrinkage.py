"""
Post-hoc Angle Shrinkage

The angle target overfits 2.57x because 72% of variance is between-player.
The model overconfidently predicts within-player deviations that are noise.

Solution: shrink Sub 2503's angle predictions toward player means.
  angle_final = shrink * sub_angle + (1 - shrink) * player_mean

This is the simplest possible intervention - no new model, just a post-hoc
correction to the current best submission.

Also test: shrinkage on ALL targets (depth/LR overfit too, just less).
"""

import json
import fcntl
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split('_')[1]) for fp in existing
                    if fp.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    print("=" * 70)
    print("POST-HOC ANGLE SHRINKAGE")
    print("=" * 70)

    # Load data to get player means
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Compute player means on scaled targets
    player_means = {}
    for target in TARGETS:
        col_map = {"angle": "angle", "depth": "depth", "left_right": "left_right"}
        raw_col = col_map[target]
        player_means[target] = {}
        for pid in sorted(train_df['participant_id'].unique()):
            mask = train_df['participant_id'] == pid
            raw_vals = train_df.loc[mask, raw_col].values
            scaled_vals = scalers[target].transform(raw_vals.reshape(-1, 1)).ravel()
            player_means[target][pid] = np.mean(scaled_vals)
            print(f"  P{pid} {target} mean: {player_means[target][pid]:.4f} "
                  f"(std={np.std(scaled_vals):.4f}, n={mask.sum()})")

    # Test player IDs
    test_pids = test_df['participant_id'].values

    # Load Sub 2503
    sub = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")
    print(f"\n  Sub 2503 loaded: {len(sub)} rows")

    # =========================================================================
    # APPROACH 1: Shrink angle only
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 1: Shrink angle only")
    print("=" * 70)

    for shrink in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sub_new = sub.copy()
        for i in range(len(sub_new)):
            pid = test_pids[i]
            pm = player_means["angle"][pid]
            orig = sub_new.loc[i, 'scaled_angle']
            sub_new.loc[i, 'scaled_angle'] = shrink * orig + (1 - shrink) * pm
        sub_new['scaled_angle'] = sub_new['scaled_angle'].clip(0, 1)

        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_new.to_csv(sub_path, index=False)

        # Show how much the predictions changed
        delta = np.abs(sub_new['scaled_angle'].values - sub['scaled_angle'].values)
        print(f"  Sub {sub_num}: angle shrink={shrink:.1f}, "
              f"mean |delta|={np.mean(delta):.6f}, max |delta|={np.max(delta):.6f}")

    # =========================================================================
    # APPROACH 2: Shrink all targets
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 2: Shrink all targets (angle more, depth/LR less)")
    print("=" * 70)

    # Based on overfit ratios: angle 2.57x, depth 1.51x, LR 1.62x
    # Shrink amount should be proportional to overfit degree
    configs = [
        {"angle": 0.5, "depth": 0.85, "left_right": 0.80},  # mild
        {"angle": 0.4, "depth": 0.80, "left_right": 0.75},  # moderate
        {"angle": 0.3, "depth": 0.75, "left_right": 0.70},  # aggressive
        {"angle": 0.5, "depth": 0.90, "left_right": 0.90},  # angle-focused
        {"angle": 0.6, "depth": 0.90, "left_right": 0.90},  # angle-focused mild
        {"angle": 0.7, "depth": 0.90, "left_right": 0.90},  # angle-focused minimal
        {"angle": 0.8, "depth": 0.95, "left_right": 0.95},  # very conservative
    ]

    for cfg in configs:
        sub_new = sub.copy()
        for i in range(len(sub_new)):
            pid = test_pids[i]
            for target, col in [("angle", "scaled_angle"), ("depth", "scaled_depth"),
                                ("left_right", "scaled_left_right")]:
                pm = player_means[target][pid]
                orig = sub_new.loc[i, col]
                sub_new.loc[i, col] = cfg[target] * orig + (1 - cfg[target]) * pm
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            sub_new[col] = sub_new[col].clip(0, 1)

        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_new.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: angle={cfg['angle']:.1f}, depth={cfg['depth']:.2f}, "
              f"LR={cfg['left_right']:.2f}")

    # =========================================================================
    # APPROACH 3: Per-player shrinkage (different shrink per player)
    # =========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 3: Per-player angle shrinkage")
    print("Based on within-player std (lower std = more shrinkage)")
    print("=" * 70)

    # Compute per-player within-player std for angle
    player_stds = {}
    for pid in sorted(train_df['participant_id'].unique()):
        mask = train_df['participant_id'] == pid
        raw_vals = train_df.loc[mask, 'angle'].values
        scaled_vals = scalers["angle"].transform(raw_vals.reshape(-1, 1)).ravel()
        player_stds[pid] = np.std(scaled_vals)

    print("  Per-player angle stds:")
    for pid, std in sorted(player_stds.items()):
        print(f"    P{pid}: std={std:.4f}")

    # Shrink more for low-std players (they're more consistent, model deviations are more noise)
    # P1: std=0.0431 (most consistent) -> more shrinkage
    # P5: std=0.1357 (most variable) -> less shrinkage
    max_std = max(player_stds.values())
    min_std = min(player_stds.values())

    for base_shrink in [0.3, 0.5, 0.7]:
        sub_new = sub.copy()
        player_shrinks = {}
        for pid in sorted(train_df['participant_id'].unique()):
            # Linear interpolation: low std -> low shrink, high std -> high shrink
            normalized = (player_stds[pid] - min_std) / (max_std - min_std + 1e-8)
            player_shrink = base_shrink + (1.0 - base_shrink) * normalized
            player_shrinks[pid] = player_shrink

        for i in range(len(sub_new)):
            pid = test_pids[i]
            pm = player_means["angle"][pid]
            orig = sub_new.loc[i, 'scaled_angle']
            sub_new.loc[i, 'scaled_angle'] = player_shrinks[pid] * orig + (1 - player_shrinks[pid]) * pm

        sub_new['scaled_angle'] = sub_new['scaled_angle'].clip(0, 1)
        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_new.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: per-player angle shrinkage (base={base_shrink:.1f})")
        for pid in sorted(player_shrinks.keys()):
            print(f"    P{pid}: shrink={player_shrinks[pid]:.3f} (std={player_stds[pid]:.4f})")

    print("\n" + "=" * 70)
    print("SUMMARY OF GENERATED SUBMISSIONS")
    print("=" * 70)
    print("\n  All submissions modify Sub 2503 by shrinking predictions toward player means.")
    print("  The thesis: model is overconfident in within-player deviations (especially angle).")
    print("  Shrinkage corrects this by pulling predictions back toward the player mean.")
    print("\n  Key LB tests to prioritize:")
    print("  1. Conservative angle-only shrink (0.7-0.8) - smallest change, safest")
    print("  2. Moderate angle-only shrink (0.5) - best LOO, moderate risk")
    print("  3. All-target shrink (angle=0.5, depth=0.90, LR=0.90) - if angle-only works")
    print("  4. Per-player shrink (base=0.5) - most principled but complex")


if __name__ == "__main__":
    main()
