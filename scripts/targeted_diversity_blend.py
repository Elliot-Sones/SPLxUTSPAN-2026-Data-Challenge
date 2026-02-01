"""
Targeted Diversity Blend

Goal: Create a submission with meaningful diversity (<95% corr with Sub 133)
while maintaining profile constraints.

Strategy:
1. Use per-player specialized model (Sub 163) which has 85% angle correlation
2. Blend directly with Sub 133 to satisfy profile constraints
3. Find maximum diversity weight that keeps angle_std < 0.14
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Target constraints from Sub 133 (best submission)
TARGET_ANGLE_STD = 0.14
TARGET_DEPTH_MEAN_MIN = 0.495
TARGET_DEPTH_MEAN_MAX = 0.515


def load_submission(num):
    """Load a submission file."""
    return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")


def compute_metrics(angle, depth, lr, ref_angle):
    """Compute profile metrics and correlation."""
    return {
        'angle_std': np.std(angle),
        'depth_mean': np.mean(depth),
        'lr_mean': np.mean(lr),
        'angle_corr': np.corrcoef(angle, ref_angle)[0, 1]
    }


def find_optimal_blend(new_angle, new_depth, new_lr, ref_angle, ref_depth, ref_lr,
                        target_corr_range=(0.85, 0.95)):
    """
    Find optimal blend weight that maximizes diversity while satisfying constraints.

    Binary search for maximum weight on new predictions (diversity) while keeping:
    - angle_std < TARGET_ANGLE_STD
    - depth_mean in [TARGET_DEPTH_MEAN_MIN, TARGET_DEPTH_MEAN_MAX]
    - correlation in target_corr_range
    """
    def is_valid(w):
        blend_angle = w * new_angle + (1 - w) * ref_angle
        blend_depth = w * new_depth + (1 - w) * ref_depth

        angle_std = np.std(blend_angle)
        depth_mean = np.mean(blend_depth)
        corr = np.corrcoef(blend_angle, ref_angle)[0, 1]

        std_ok = angle_std <= TARGET_ANGLE_STD
        depth_ok = TARGET_DEPTH_MEAN_MIN <= depth_mean <= TARGET_DEPTH_MEAN_MAX
        corr_ok = target_corr_range[0] <= corr <= target_corr_range[1]

        return std_ok and depth_ok, corr_ok, corr

    # Binary search for maximum valid weight
    lo, hi = 0.0, 1.0
    best_w = 0.0

    while hi - lo > 0.001:
        mid = (lo + hi) / 2
        profile_ok, _, _ = is_valid(mid)

        if profile_ok:
            best_w = mid
            lo = mid
        else:
            hi = mid

    return best_w


def main():
    print("="*80)
    print("TARGETED DIVERSITY BLEND")
    print("="*80)

    # Load submissions
    sub133 = load_submission(133)
    sub163 = load_submission(163)  # Per-player specialized

    ref_angle = sub133['scaled_angle'].values
    ref_depth = sub133['scaled_depth'].values
    ref_lr = sub133['scaled_left_right'].values

    new_angle = sub163['scaled_angle'].values
    new_depth = sub163['scaled_depth'].values
    new_lr = sub163['scaled_left_right'].values

    print("\nSub 133 profile:")
    m133 = compute_metrics(ref_angle, ref_depth, ref_lr, ref_angle)
    for k, v in m133.items():
        print(f"  {k}: {v:.4f}")

    print("\nSub 163 (per-player specialized) profile:")
    m163 = compute_metrics(new_angle, new_depth, new_lr, ref_angle)
    for k, v in m163.items():
        print(f"  {k}: {v:.4f}")

    # Find optimal blend weight
    print("\n" + "="*80)
    print("SEARCHING FOR OPTIMAL BLEND WEIGHT")
    print("="*80)

    # Test different weights
    print("\nWeight scan:")
    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        blend_angle = w * new_angle + (1 - w) * ref_angle
        blend_depth = w * new_depth + (1 - w) * ref_depth
        blend_lr = w * new_lr + (1 - w) * ref_lr

        metrics = compute_metrics(blend_angle, blend_depth, blend_lr, ref_angle)
        profile_ok = (metrics['angle_std'] <= TARGET_ANGLE_STD and
                      TARGET_DEPTH_MEAN_MIN <= metrics['depth_mean'] <= TARGET_DEPTH_MEAN_MAX)

        status = "OK" if profile_ok else "FAIL"
        print(f"  w={w:.1f}: angle_std={metrics['angle_std']:.4f}, "
              f"depth_mean={metrics['depth_mean']:.4f}, "
              f"corr={metrics['angle_corr']:.4f} [{status}]")

    # Find exact optimal weight
    optimal_w = find_optimal_blend(new_angle, new_depth, new_lr,
                                    ref_angle, ref_depth, ref_lr,
                                    target_corr_range=(0.0, 1.0))  # Any correlation

    print(f"\nOptimal weight (max diversity while satisfying profile): {optimal_w:.3f}")

    # Create final blend
    final_angle = optimal_w * new_angle + (1 - optimal_w) * ref_angle
    final_depth = optimal_w * new_depth + (1 - optimal_w) * ref_depth
    final_lr = optimal_w * new_lr + (1 - optimal_w) * ref_lr

    final_metrics = compute_metrics(final_angle, final_depth, final_lr, ref_angle)

    print("\nFinal blend profile:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Per-player correlation analysis
    print("\n" + "="*80)
    print("PER-PLAYER CORRELATION ANALYSIS")
    print("="*80)

    # Infer player IDs from test data patterns
    # Based on angle ranges (observed from training data patterns)
    n_samples = len(final_angle)

    print(f"\nTotal test samples: {n_samples}")
    print("Cannot determine per-player correlations without test player IDs")

    # Save submission
    print("\n" + "="*80)
    print("CREATING SUBMISSION")
    print("="*80)

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 165

    submission = pd.DataFrame({
        'id': sub133['id'].values,
        'scaled_angle': final_angle,
        'scaled_depth': final_depth,
        'scaled_left_right': final_lr
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"\nSubmission {next_num} details:")
    print(f"  Model: {optimal_w:.1%} Sub163 (per-player) + {1-optimal_w:.1%} Sub133")
    print(f"  Profile: angle_std={final_metrics['angle_std']:.4f}, depth_mean={final_metrics['depth_mean']:.4f}")
    print(f"  Correlation with Sub 133: {final_metrics['angle_corr']:.4f}")

    # Assessment
    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)

    if final_metrics['angle_corr'] > 0.98:
        print("  WARNING: Very similar to Sub 133 - unlikely to improve")
    elif final_metrics['angle_corr'] > 0.95:
        print("  MODERATE: Slightly different from Sub 133 - small chance of improvement")
    elif final_metrics['angle_corr'] > 0.90:
        print("  GOOD: Meaningful diversity - reasonable chance of improvement")
    else:
        print("  RISKY: Very different from Sub 133 - could be better or worse")

    # Also create a more aggressive blend for comparison
    print("\n" + "="*80)
    print("CREATING AGGRESSIVE BLEND (relaxed constraints)")
    print("="*80)

    # Find weight that gives ~90% correlation even if profile is slightly off
    for target_std in [0.145, 0.15, 0.155, 0.16]:
        w = 0.0
        for test_w in np.arange(0.0, 1.0, 0.01):
            blend_angle = test_w * new_angle + (1 - test_w) * ref_angle
            if np.std(blend_angle) <= target_std:
                w = test_w

        blend_angle = w * new_angle + (1 - w) * ref_angle
        blend_depth = w * new_depth + (1 - w) * ref_depth
        corr = np.corrcoef(blend_angle, ref_angle)[0, 1]
        std = np.std(blend_angle)
        dmean = np.mean(blend_depth)

        print(f"  target_std={target_std:.3f}: w={w:.2f}, actual_std={std:.4f}, "
              f"corr={corr:.4f}, depth_mean={dmean:.4f}")

    # Save with relaxed constraint (angle_std <= 0.15)
    aggressive_w = 0.0
    for test_w in np.arange(0.0, 1.0, 0.01):
        blend_angle = test_w * new_angle + (1 - test_w) * ref_angle
        blend_depth = test_w * new_depth + (1 - test_w) * ref_depth
        if np.std(blend_angle) <= 0.15 and 0.49 <= np.mean(blend_depth) <= 0.52:
            aggressive_w = test_w

    if aggressive_w > optimal_w + 0.05:
        agg_angle = aggressive_w * new_angle + (1 - aggressive_w) * ref_angle
        agg_depth = aggressive_w * new_depth + (1 - aggressive_w) * ref_depth
        agg_lr = aggressive_w * new_lr + (1 - aggressive_w) * ref_lr

        agg_metrics = compute_metrics(agg_angle, agg_depth, agg_lr, ref_angle)

        print(f"\nAggressive blend (angle_std <= 0.15):")
        print(f"  Weight: {aggressive_w:.2f}")
        for k, v in agg_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save aggressive submission
        next_num += 1
        agg_submission = pd.DataFrame({
            'id': sub133['id'].values,
            'scaled_angle': agg_angle,
            'scaled_depth': agg_depth,
            'scaled_left_right': agg_lr
        })

        agg_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        agg_submission.to_csv(agg_file, index=False)

        print(f"\nSaved aggressive blend: {agg_file}")
        print(f"  Model: {aggressive_w:.1%} Sub163 + {1-aggressive_w:.1%} Sub133")
        print(f"  Profile: angle_std={agg_metrics['angle_std']:.4f}, depth_mean={agg_metrics['depth_mean']:.4f}")
        print(f"  Correlation with Sub 133: {agg_metrics['angle_corr']:.4f}")


if __name__ == "__main__":
    main()
