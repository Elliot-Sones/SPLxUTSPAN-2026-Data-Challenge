"""
Target-Specific Blending - Different blend weights for each target.

Since different submissions are good at different targets, blend them
optimally per target:
- angle: prioritize lower angle_std
- depth: prioritize depth_mean closer to 0.505
- left_right: prioritize accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"

# Load best submissions
subs = {
    9: pd.read_csv(SUBMISSION_DIR / "submission_9.csv"),
    10: pd.read_csv(SUBMISSION_DIR / "submission_10.csv"),
    25: pd.read_csv(SUBMISSION_DIR / "submission_25.csv"),
    111: pd.read_csv(SUBMISSION_DIR / "submission_111.csv"),
    113: pd.read_csv(SUBMISSION_DIR / "submission_113.csv"),  # LB 0.008031
    114: pd.read_csv(SUBMISSION_DIR / "submission_114.csv"),  # Biomech
}

# Known LB scores
KNOWN_LB = {9: 0.009109, 10: 0.008907, 25: 0.008305, 111: 0.008703, 113: 0.008031}

# Optimal targets
OPTIMAL_DEPTH_MEAN = 0.505
TARGET_ANGLE_STD = 0.1370  # Lower than Sub 25's 0.1380


def create_target_blend(angle_blend, depth_blend, lr_blend):
    """Create a blend with different weights per target."""
    result = subs[9][['id']].copy()

    # Angle blend
    angle_preds = sum(w * subs[s]['scaled_angle'] for s, w in angle_blend)
    result['scaled_angle'] = angle_preds

    # Depth blend
    depth_preds = sum(w * subs[s]['scaled_depth'] for s, w in depth_blend)
    result['scaled_depth'] = depth_preds

    # Left-right blend
    lr_preds = sum(w * subs[s]['scaled_left_right'] for s, w in lr_blend)
    result['scaled_left_right'] = lr_preds

    return result


def evaluate_blend(df):
    """Compute metrics for a blend."""
    return {
        'angle_std': df['scaled_angle'].std(),
        'angle_mean': df['scaled_angle'].mean(),
        'depth_mean': df['scaled_depth'].mean(),
        'depth_std': df['scaled_depth'].std(),
        'lr_mean': df['scaled_left_right'].mean(),
        'lr_std': df['scaled_left_right'].std(),
    }


def score_blend(metrics):
    """
    Score a blend based on how close it is to optimal.
    Lower is better.
    """
    # Depth_mean distance from 0.505 (weight: 0.74 based on correlation)
    depth_score = abs(metrics['depth_mean'] - OPTIMAL_DEPTH_MEAN) * 0.74

    # Angle_std (weight: 0.20 based on correlation)
    # Lower is better, but too low might mean underfitting
    angle_score = abs(metrics['angle_std'] - TARGET_ANGLE_STD) * 0.20

    return depth_score + angle_score


def main():
    print("=" * 80)
    print("TARGET-SPECIFIC BLENDING")
    print("=" * 80)

    # Analyze individual submissions
    print("\nIndividual submission profiles:")
    print(f"{'Sub':>5} {'angle_std':>12} {'depth_mean':>12} {'LB':>12}")
    print("-" * 45)

    for s, df in sorted(subs.items()):
        m = evaluate_blend(df)
        lb = KNOWN_LB.get(s, '-')
        print(f"{s:>5} {m['angle_std']:>12.4f} {m['depth_mean']:>12.4f} {str(lb):>12}")

    # Search for optimal target-specific blends
    print("\n" + "=" * 60)
    print("Searching for optimal target-specific blend...")
    print("=" * 60)

    best_score = float('inf')
    best_blend = None
    best_metrics = None

    # Try different combinations for each target
    sub_keys = list(subs.keys())

    # For angle: prioritize Sub 9 and 10 (lower angle_std when blended)
    angle_candidates = [
        [(9, 0.5), (10, 0.5)],  # Sub 25 style
        [(9, 0.6), (10, 0.4)],
        [(9, 0.4), (10, 0.6)],
        [(9, 0.5), (10, 0.4), (111, 0.1)],  # Sub 113 style
        [(9, 0.45), (10, 0.45), (111, 0.1)],
        [(9, 0.5), (10, 0.45), (111, 0.05)],
        [(113, 1.0)],  # Just use Sub 113
    ]

    # For depth: prioritize getting close to 0.505
    depth_candidates = [
        [(9, 0.5), (10, 0.5)],
        [(9, 0.6), (10, 0.4)],
        [(9, 0.55), (10, 0.45)],
        [(9, 0.5), (10, 0.4), (111, 0.1)],
        [(9, 0.5), (10, 0.45), (114, 0.05)],  # Add biomech diversity
        [(113, 1.0)],
    ]

    # For left_right: use best performing
    lr_candidates = [
        [(9, 0.5), (10, 0.5)],
        [(9, 0.5), (10, 0.4), (111, 0.1)],
        [(9, 0.5), (10, 0.45), (111, 0.05)],
        [(113, 1.0)],
        [(9, 0.4), (10, 0.4), (111, 0.2)],
    ]

    count = 0
    for a_blend in angle_candidates:
        for d_blend in depth_candidates:
            for lr_blend in lr_candidates:
                count += 1
                df = create_target_blend(a_blend, d_blend, lr_blend)
                metrics = evaluate_blend(df)
                score = score_blend(metrics)

                if score < best_score:
                    best_score = score
                    best_blend = (a_blend, d_blend, lr_blend)
                    best_metrics = metrics

    print(f"Tested {count} combinations")
    print(f"\nBest blend:")
    print(f"  Angle: {best_blend[0]}")
    print(f"  Depth: {best_blend[1]}")
    print(f"  Left-Right: {best_blend[2]}")
    print(f"\nMetrics:")
    print(f"  angle_std: {best_metrics['angle_std']:.4f}")
    print(f"  depth_mean: {best_metrics['depth_mean']:.4f}")
    print(f"  Score: {best_score:.6f}")

    # Compare with Sub 113
    sub113_metrics = evaluate_blend(subs[113])
    sub113_score = score_blend(sub113_metrics)
    print(f"\nSub 113 comparison:")
    print(f"  angle_std: {sub113_metrics['angle_std']:.4f}")
    print(f"  depth_mean: {sub113_metrics['depth_mean']:.4f}")
    print(f"  Score: {sub113_score:.6f}")

    if best_score < sub113_score:
        print(f"\n*** New blend is {sub113_score - best_score:.6f} better! ***")

        # Create and save
        df = create_target_blend(*best_blend)

        existing = list(SUBMISSION_DIR.glob("submission*.csv"))
        nums = []
        for f in existing:
            name = f.stem
            if name.startswith("submission_"):
                try:
                    nums.append(int(name.split('_')[1]))
                except:
                    pass

        next_num = max(nums) + 1
        filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
        df.to_csv(filepath, index=False)

        print(f"\nSaved as: {filepath}")
        print(f"Submission {next_num}: Target-specific blend")
        return filepath

    print("\nSub 113 is already optimal or near-optimal.")
    return None


if __name__ == "__main__":
    main()
