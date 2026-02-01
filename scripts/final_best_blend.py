"""
Final Best Blend

Combines all our learnings:
1. Selective amplification of high-difference samples (from 133 vs 151)
2. Optimized component weights
3. Depth calibration
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'


def load_submission(num):
    return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")


def main():
    print("="*80)
    print("FINAL BEST BLEND")
    print("="*80)

    sub133 = load_submission(133)
    sub151 = load_submission(151)
    sub9 = load_submission(9)
    sub10 = load_submission(10)
    sub111 = load_submission(111)
    sub25 = load_submission(25)

    test_ids = sub133['id'].values
    n = len(test_ids)

    ref_angle_std = np.std(sub133['scaled_angle'].values)
    print(f"\nSub 133 angle_std: {ref_angle_std:.6f}")

    # Compute difference vectors
    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values

    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    # High-difference mask (top 10%)
    threshold = np.percentile(total_diff, 90)
    high_diff_mask = total_diff > threshold

    # Create best submission combining strategies
    best_submissions = []

    # Strategy A: Selective amplification alpha=1.0
    sel_angle = sub133['scaled_angle'].values.copy()
    sel_depth = sub133['scaled_depth'].values.copy()
    sel_lr = sub133['scaled_left_right'].values.copy()

    alpha = 1.0
    sel_angle[high_diff_mask] += alpha * diff_angle[high_diff_mask]
    sel_depth[high_diff_mask] += alpha * diff_depth[high_diff_mask]
    sel_lr[high_diff_mask] += alpha * diff_lr[high_diff_mask]

    sel_angle = np.clip(sel_angle, 0, 1)
    sel_depth = np.clip(sel_depth, 0, 1)
    sel_lr = np.clip(sel_lr, 0, 1)

    # Calibrate depth
    sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
    sel_depth = np.clip(sel_depth, 0, 1)

    sel_std = np.std(sel_angle)
    print(f"\nSelective amplification (alpha=1.0): angle_std={sel_std:.6f}")

    best_submissions.append(('selective_amp', sel_angle.copy(), sel_depth.copy(), sel_lr.copy(), sel_std))

    # Strategy B: w10=48% blend
    w10 = 0.48
    w9 = 0.30
    w111 = 0.21
    w25 = 1.0 - w9 - w10 - w111

    emp_angle = (w25 * sub25['scaled_angle'].values +
                w9 * sub9['scaled_angle'].values +
                w10 * sub10['scaled_angle'].values +
                w111 * sub111['scaled_angle'].values)

    emp_depth = (w25 * sub25['scaled_depth'].values +
                w9 * sub9['scaled_depth'].values +
                w10 * sub10['scaled_depth'].values +
                w111 * sub111['scaled_depth'].values)

    emp_lr = (w25 * sub25['scaled_left_right'].values +
             w9 * sub9['scaled_left_right'].values +
             w10 * sub10['scaled_left_right'].values +
             w111 * sub111['scaled_left_right'].values)

    emp_depth = emp_depth - np.mean(emp_depth) + 0.5055
    emp_depth = np.clip(emp_depth, 0, 1)

    emp_std = np.std(emp_angle)
    print(f"w10=48% blend: angle_std={emp_std:.6f}")

    best_submissions.append(('w10_48', emp_angle.copy(), emp_depth.copy(), emp_lr.copy(), emp_std))

    # Strategy C: Combine A and B
    comb_angle = 0.5 * sel_angle + 0.5 * emp_angle
    comb_depth = 0.5 * sel_depth + 0.5 * emp_depth
    comb_lr = 0.5 * sel_lr + 0.5 * emp_lr

    comb_depth = comb_depth - np.mean(comb_depth) + 0.5055
    comb_depth = np.clip(comb_depth, 0, 1)

    comb_std = np.std(comb_angle)
    print(f"Combined A+B: angle_std={comb_std:.6f}")

    best_submissions.append(('combined', comb_angle.copy(), comb_depth.copy(), comb_lr.copy(), comb_std))

    # Find best
    best_submissions.sort(key=lambda x: x[4])  # Sort by angle_std

    print("\n" + "="*80)
    print("SAVING BEST SUBMISSIONS")
    print("="*80)

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    for name, angle, depth, lr, std in best_submissions:
        if std < ref_angle_std:
            corr = np.corrcoef(angle, sub133['scaled_angle'].values)[0, 1]
            improvement = (ref_angle_std - std) / ref_angle_std * 100

            print(f"\n{name}: angle_std={std:.6f} ({improvement:.3f}% better than Sub 133)")
            print(f"  corr with Sub 133: {corr:.4f}")
            print(f"  depth_mean: {np.mean(depth):.6f}")

            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': angle,
                'scaled_depth': depth,
                'scaled_left_right': lr
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")
            next_num += 1

    # Summary
    print("\n" + "="*80)
    print("FINAL RANKING")
    print("="*80)

    print("\nSubmissions with lower angle_std than Sub 133 (0.137117):")
    for name, angle, depth, lr, std in best_submissions:
        if std < ref_angle_std:
            print(f"  {name}: {std:.6f} ({(ref_angle_std - std) * 100:.4f}% lower)")


if __name__ == "__main__":
    main()
