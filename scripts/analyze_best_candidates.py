"""
Analyze best candidate submissions and rank them.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'

# Target profile (from Sub 133, the best)
TARGET_ANGLE_STD = 0.1377
TARGET_DEPTH_MEAN = 0.5055


def load_submission(num):
    try:
        return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")
    except:
        return None


def analyze_submission(sub, num, ref_sub):
    """Analyze a submission relative to Sub 133."""
    angle = sub['scaled_angle'].values
    depth = sub['scaled_depth'].values
    lr = sub['scaled_left_right'].values

    ref_angle = ref_sub['scaled_angle'].values

    angle_std = np.std(angle)
    depth_mean = np.mean(depth)
    lr_mean = np.mean(lr)

    corr = np.corrcoef(angle, ref_angle)[0, 1]

    # Profile distance (weighted by LB correlation)
    # depth_mean is 3.7x more important than angle_std
    profile_dist = abs(angle_std - TARGET_ANGLE_STD) + 3.7 * abs(depth_mean - TARGET_DEPTH_MEAN)

    # Predicted LB (simplified model)
    # Lower angle_std and closer depth_mean to 0.5055 -> better LB
    pred_lb = 0.0078 + 0.15 * abs(angle_std - TARGET_ANGLE_STD) + 0.55 * abs(depth_mean - TARGET_DEPTH_MEAN)

    return {
        'num': num,
        'angle_std': angle_std,
        'depth_mean': depth_mean,
        'lr_mean': lr_mean,
        'corr_133': corr,
        'profile_dist': profile_dist,
        'pred_lb': pred_lb,
        'angle_std_vs_133': angle_std - 0.137117,  # Sub 133's angle_std
    }


def main():
    print("="*80)
    print("CANDIDATE SUBMISSION ANALYSIS")
    print("="*80)

    # Load Sub 133 as reference
    sub133 = load_submission(133)

    # Analyze recent submissions
    candidates = list(range(163, 181))  # Recent submissions

    results = []
    for num in candidates:
        sub = load_submission(num)
        if sub is not None:
            metrics = analyze_submission(sub, num, sub133)
            results.append(metrics)

    # Sort by predicted LB
    results.sort(key=lambda x: x['pred_lb'])

    print("\nTop candidates (sorted by predicted LB):")
    print("-"*100)
    print(f"{'Sub':>5} | {'angle_std':>10} | {'depth_mean':>10} | {'corr_133':>9} | {'pred_lb':>9} | {'vs_133':>10}")
    print("-"*100)

    for r in results[:15]:
        vs_133 = "BETTER" if r['angle_std'] < 0.137117 else "WORSE"
        print(f"{r['num']:>5} | {r['angle_std']:>10.6f} | {r['depth_mean']:>10.6f} | "
              f"{r['corr_133']:>9.4f} | {r['pred_lb']:>9.6f} | {vs_133:>10}")

    print("\n" + "="*80)
    print("BEST CANDIDATES TO SUBMIT")
    print("="*80)

    # Filter candidates with good profile
    good_candidates = [r for r in results if r['angle_std'] <= 0.14 and
                       0.50 <= r['depth_mean'] <= 0.51]

    if good_candidates:
        print("\nCandidates with good profile (angle_std <= 0.14, depth_mean in [0.50, 0.51]):")
        for r in good_candidates[:5]:
            print(f"\n  Sub {r['num']}:")
            print(f"    angle_std:  {r['angle_std']:.6f} ({'better' if r['angle_std'] < 0.137117 else 'worse'} than Sub 133)")
            print(f"    depth_mean: {r['depth_mean']:.6f}")
            print(f"    corr_133:   {r['corr_133']:.4f}")
            print(f"    pred_lb:    {r['pred_lb']:.6f}")
    else:
        print("\nNo candidates with good profile found.")

    # Find submissions with lower angle_std than Sub 133
    print("\n" + "="*80)
    print("SUBMISSIONS WITH LOWER angle_std THAN SUB 133 (0.137117)")
    print("="*80)

    lower_std = [r for r in results if r['angle_std'] < 0.137117]
    if lower_std:
        for r in lower_std:
            improvement = (0.137117 - r['angle_std']) / 0.137117 * 100
            print(f"\n  Sub {r['num']}: angle_std={r['angle_std']:.6f} ({improvement:.2f}% lower)")
            print(f"    depth_mean: {r['depth_mean']:.6f}")
            print(f"    corr_133:   {r['corr_133']:.4f}")
    else:
        print("\nNo submissions with lower angle_std than Sub 133.")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Best by predicted LB with reasonable correlation
    best = [r for r in results if r['corr_133'] > 0.95][:3]
    print("\nTop 3 by predicted LB (with >95% correlation to Sub 133):")
    for r in best:
        print(f"  Sub {r['num']}: pred_lb={r['pred_lb']:.6f}, corr={r['corr_133']:.4f}")

    # Most diverse with good profile
    diverse = [r for r in results if r['corr_133'] < 0.99 and r['angle_std'] < 0.145]
    if diverse:
        diverse.sort(key=lambda x: x['corr_133'])
        print("\nMost diverse with reasonable profile:")
        for r in diverse[:3]:
            print(f"  Sub {r['num']}: corr={r['corr_133']:.4f}, angle_std={r['angle_std']:.6f}")


if __name__ == "__main__":
    main()
