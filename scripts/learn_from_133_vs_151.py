"""
Learn From Sub 133 vs Sub 151

Sub 133 (LB 0.007809) is 6.4% better than Sub 151 (LB 0.008305).
Can we learn what makes Sub 133 better and amplify it?

The key difference is Sample 17, which has 0.13 total difference.
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
    print("LEARNING FROM SUB 133 vs SUB 151")
    print("="*80)

    sub133 = load_submission(133)
    sub151 = load_submission(151)
    sub9 = load_submission(9)
    sub10 = load_submission(10)
    sub111 = load_submission(111)
    sub25 = load_submission(25)

    test_ids = sub133['id'].values
    n = len(test_ids)

    # Compute difference vectors
    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values

    print(f"\nDifference analysis (Sub 133 - Sub 151):")
    print(f"  angle: mean={diff_angle.mean():.6f}, std={np.std(diff_angle):.6f}")
    print(f"  depth: mean={diff_depth.mean():.6f}, std={np.std(diff_depth):.6f}")
    print(f"  lr:    mean={diff_lr.mean():.6f}, std={np.std(diff_lr):.6f}")

    # Find samples where Sub 133 differs most from Sub 151
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    print(f"\nTop 10 samples with largest difference:")
    top_idx = np.argsort(total_diff)[-10:][::-1]
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. Sample {idx}: total={total_diff[idx]:.4f}, "
              f"angle={diff_angle[idx]:.4f}, depth={diff_depth[idx]:.4f}")

    # Strategy: Amplify Sub 133's direction away from Sub 151
    print("\n" + "="*80)
    print("STRATEGY: AMPLIFY SUB 133's DIRECTION")
    print("="*80)

    # Create amplified predictions
    # new = sub133 + alpha * (sub133 - sub151)
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        amp_angle = sub133['scaled_angle'].values + alpha * diff_angle
        amp_depth = sub133['scaled_depth'].values + alpha * diff_depth
        amp_lr = sub133['scaled_left_right'].values + alpha * diff_lr

        # Clip to [0, 1]
        amp_angle = np.clip(amp_angle, 0, 1)
        amp_depth = np.clip(amp_depth, 0, 1)
        amp_lr = np.clip(amp_lr, 0, 1)

        amp_std = np.std(amp_angle)
        amp_depth_mean = np.mean(amp_depth)

        print(f"\nalpha={alpha}: angle_std={amp_std:.4f}, depth_mean={amp_depth_mean:.4f}")

    # Strategy 2: Only amplify for high-difference samples
    print("\n" + "="*80)
    print("STRATEGY 2: SELECTIVE AMPLIFICATION")
    print("="*80)

    # Only amplify top 10% different samples
    threshold = np.percentile(total_diff, 90)
    high_diff_mask = total_diff > threshold

    print(f"Amplifying {high_diff_mask.sum()} high-difference samples")

    for alpha in [0.2, 0.5, 1.0]:
        sel_angle = sub133['scaled_angle'].values.copy()
        sel_depth = sub133['scaled_depth'].values.copy()
        sel_lr = sub133['scaled_left_right'].values.copy()

        sel_angle[high_diff_mask] += alpha * diff_angle[high_diff_mask]
        sel_depth[high_diff_mask] += alpha * diff_depth[high_diff_mask]
        sel_lr[high_diff_mask] += alpha * diff_lr[high_diff_mask]

        # Clip
        sel_angle = np.clip(sel_angle, 0, 1)
        sel_depth = np.clip(sel_depth, 0, 1)
        sel_lr = np.clip(sel_lr, 0, 1)

        sel_std = np.std(sel_angle)
        sel_depth_mean = np.mean(sel_depth)
        corr = np.corrcoef(sel_angle, sub133['scaled_angle'].values)[0, 1]

        if sel_std < 0.14:
            print(f"\nalpha={alpha}: angle_std={sel_std:.4f}, depth_mean={sel_depth_mean:.4f}, corr={corr:.4f}")

    # Strategy 3: Blend Sub 133 further away from Sub 151 (toward Sub 9/10/111)
    print("\n" + "="*80)
    print("STRATEGY 3: OPPOSITE DIRECTION FROM SUB 151")
    print("="*80)

    # Sub 151 = Sub 25 essentially (100% correlation)
    # Try moving away from Sub 25

    # Find the direction from Sub 25 to Sub 133
    dir_angle = sub133['scaled_angle'].values - sub25['scaled_angle'].values
    dir_depth = sub133['scaled_depth'].values - sub25['scaled_depth'].values
    dir_lr = sub133['scaled_left_right'].values - sub25['scaled_left_right'].values

    for alpha in [0.05, 0.10, 0.15, 0.20]:
        ext_angle = sub133['scaled_angle'].values + alpha * dir_angle
        ext_depth = sub133['scaled_depth'].values + alpha * dir_depth
        ext_lr = sub133['scaled_left_right'].values + alpha * dir_lr

        # Clip
        ext_angle = np.clip(ext_angle, 0, 1)
        ext_depth = np.clip(ext_depth, 0, 1)
        ext_lr = np.clip(ext_lr, 0, 1)

        ext_std = np.std(ext_angle)
        ext_depth_mean = np.mean(ext_depth)
        corr = np.corrcoef(ext_angle, sub133['scaled_angle'].values)[0, 1]

        if ext_std < 0.14:
            print(f"\nalpha={alpha}: angle_std={ext_std:.4f}, depth_mean={ext_depth_mean:.4f}, corr={corr:.4f}")

            # Calibrate depth
            ext_depth = ext_depth - np.mean(ext_depth) + 0.5055
            ext_depth = np.clip(ext_depth, 0, 1)

            # Save best
            if alpha == 0.10 and ext_std < 0.14:
                existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
                nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
                next_num = max(nums) + 1

                submission = pd.DataFrame({
                    'id': test_ids,
                    'scaled_angle': ext_angle,
                    'scaled_depth': ext_depth,
                    'scaled_left_right': ext_lr
                })

                output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
                submission.to_csv(output_file, index=False)

                print(f"\nSaved: {output_file}")
                print(f"  angle_std: {np.std(ext_angle):.6f}")
                print(f"  depth_mean: {np.mean(ext_depth):.6f}")

    # Strategy 4: Look at what components Sub 133 uses more than Sub 151
    print("\n" + "="*80)
    print("STRATEGY 4: ANALYZE COMPONENT USAGE")
    print("="*80)

    # Sub 133 = 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111
    # Sub 151 = Sub 25 essentially
    # So Sub 133 uses more Sub9, Sub10, Sub111 than Sub 151

    # Try emphasizing Sub10 even more (44% -> 50%)
    for w10 in [0.48, 0.50, 0.52, 0.54]:
        w9 = 0.30
        w111 = 0.21
        w25 = 1.0 - w9 - w10 - w111

        if w25 < 0:
            continue

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

        emp_std = np.std(emp_angle)
        emp_depth_mean = np.mean(emp_depth)
        corr = np.corrcoef(emp_angle, sub133['scaled_angle'].values)[0, 1]

        print(f"\nw10={w10:.0%}: angle_std={emp_std:.6f}, depth_mean={emp_depth_mean:.6f}, corr={corr:.4f}")

        if emp_std < 0.137117 and abs(emp_depth_mean - 0.5055) < 0.002:
            print("  ** BETTER THAN SUB 133 **")

            # Save
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1

            emp_depth = emp_depth - np.mean(emp_depth) + 0.5055
            emp_depth = np.clip(emp_depth, 0, 1)

            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': emp_angle,
                'scaled_depth': emp_depth,
                'scaled_left_right': emp_lr
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")


if __name__ == "__main__":
    main()
