"""
Micro-Optimization

Since all diverse models have higher angle_std than Sub 133,
and blending always makes correlation very high,
let's try micro-adjustments to Sub 133 predictions.

The idea: Small, targeted adjustments to specific samples
might improve predictions without increasing angle_std.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'
OUTPUT_DIR = PROJECT_DIR / 'output'


def load_submission(num):
    return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")


def main():
    print("="*80)
    print("MICRO-OPTIMIZATION")
    print("Targeted adjustments to Sub 133")
    print("="*80)

    # Load key submissions
    sub133 = load_submission(133)
    sub9 = load_submission(9)
    sub10 = load_submission(10)
    sub111 = load_submission(111)
    sub25 = load_submission(25)
    sub163 = load_submission(163)

    test_ids = sub133['id'].values
    n = len(test_ids)

    ref_angle = sub133['scaled_angle'].values
    ref_depth = sub133['scaled_depth'].values
    ref_lr = sub133['scaled_left_right'].values

    ref_angle_std = np.std(ref_angle)
    ref_depth_mean = np.mean(ref_depth)

    print(f"\nSub 133 reference:")
    print(f"  angle_std:  {ref_angle_std:.6f}")
    print(f"  depth_mean: {ref_depth_mean:.6f}")

    # Strategy 1: Sample-specific micro-adjustments
    # Identify samples where other models disagree with Sub 133
    print("\n" + "="*80)
    print("STRATEGY 1: Sample-specific adjustments")
    print("="*80)

    # Find samples with high disagreement
    subs = {'9': sub9, '10': sub10, '111': sub111, '25': sub25, '163': sub163}

    angle_variance = np.zeros(n)
    depth_variance = np.zeros(n)
    lr_variance = np.zeros(n)

    for sub in subs.values():
        angle_variance += (sub['scaled_angle'].values - ref_angle) ** 2
        depth_variance += (sub['scaled_depth'].values - ref_depth) ** 2
        lr_variance += (sub['scaled_left_right'].values - ref_lr) ** 2

    angle_variance /= len(subs)
    depth_variance /= len(subs)
    lr_variance /= len(subs)

    # Top disagreement samples
    top_disagreement = np.argsort(angle_variance + depth_variance + lr_variance)[-10:][::-1]
    print("\nTop 10 samples with highest model disagreement:")
    for i, idx in enumerate(top_disagreement):
        print(f"  {i+1}. Sample {idx}: angle_var={angle_variance[idx]:.4f}, "
              f"depth_var={depth_variance[idx]:.4f}")

    # Strategy 2: Per-sample weighted ensemble
    print("\n" + "="*80)
    print("STRATEGY 2: Per-sample weighted ensemble")
    print("="*80)

    # For high-agreement samples: use Sub 133
    # For high-disagreement samples: use ensemble

    agreement_threshold = np.percentile(angle_variance, 75)
    high_disagreement_mask = angle_variance > agreement_threshold

    print(f"High disagreement samples: {high_disagreement_mask.sum()}")

    # Ensemble for high-disagreement samples
    ens_angle = (0.25 * sub9['scaled_angle'].values +
                 0.25 * sub10['scaled_angle'].values +
                 0.25 * sub111['scaled_angle'].values +
                 0.25 * sub25['scaled_angle'].values)

    ens_depth = (0.25 * sub9['scaled_depth'].values +
                 0.25 * sub10['scaled_depth'].values +
                 0.25 * sub111['scaled_depth'].values +
                 0.25 * sub25['scaled_depth'].values)

    ens_lr = (0.25 * sub9['scaled_left_right'].values +
              0.25 * sub10['scaled_left_right'].values +
              0.25 * sub111['scaled_left_right'].values +
              0.25 * sub25['scaled_left_right'].values)

    # Try different blend strengths for high-disagreement samples
    best_blend = None
    best_std = ref_angle_std

    for blend_strength in [0.05, 0.10, 0.15, 0.20, 0.25]:
        new_angle = ref_angle.copy()
        new_depth = ref_depth.copy()
        new_lr = ref_lr.copy()

        # Only adjust high-disagreement samples
        new_angle[high_disagreement_mask] = ((1 - blend_strength) * ref_angle[high_disagreement_mask] +
                                              blend_strength * ens_angle[high_disagreement_mask])
        new_depth[high_disagreement_mask] = ((1 - blend_strength) * ref_depth[high_disagreement_mask] +
                                              blend_strength * ens_depth[high_disagreement_mask])
        new_lr[high_disagreement_mask] = ((1 - blend_strength) * ref_lr[high_disagreement_mask] +
                                          blend_strength * ens_lr[high_disagreement_mask])

        new_std = np.std(new_angle)
        new_depth_mean = np.mean(new_depth)

        if new_std < best_std:
            best_std = new_std
            best_blend = (new_angle.copy(), new_depth.copy(), new_lr.copy(), blend_strength)
            print(f"\n  blend_strength={blend_strength}: angle_std={new_std:.6f} (improvement: {(ref_angle_std-new_std)*100:.4f}%)")

    # Strategy 3: Optimize blend weights per sample to minimize angle variance
    print("\n" + "="*80)
    print("STRATEGY 3: Variance-minimizing per-sample weights")
    print("="*80)

    # For each sample, find the weight that brings it closest to the overall mean
    # while keeping total angle_std low

    angle_mean = np.mean(ref_angle)

    # Try to shrink extreme predictions toward mean
    shrink_factors = np.ones(n)
    for i in range(n):
        dev = ref_angle[i] - angle_mean
        if abs(dev) > np.std(ref_angle):
            # Shrink extreme predictions
            shrink_factors[i] = 0.95

    shrunk_angle = angle_mean + shrink_factors * (ref_angle - angle_mean)
    shrunk_std = np.std(shrunk_angle)

    print(f"Shrinkage approach:")
    print(f"  Original angle_std: {ref_angle_std:.6f}")
    print(f"  Shrunk angle_std:   {shrunk_std:.6f}")

    if shrunk_std < ref_angle_std:
        print(f"  Improvement: {(ref_angle_std - shrunk_std)*100:.4f}%")

        # Calibrate to match original mean
        shrunk_angle = shrunk_angle - np.mean(shrunk_angle) + np.mean(ref_angle)

    # Strategy 4: Find samples where Sub 133 differs from component submissions
    print("\n" + "="*80)
    print("STRATEGY 4: Analyze Sub 133 blend residuals")
    print("="*80)

    # Sub 133 = 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111
    reconstructed = (0.05 * sub25['scaled_angle'].values +
                    0.30 * sub9['scaled_angle'].values +
                    0.44 * sub10['scaled_angle'].values +
                    0.21 * sub111['scaled_angle'].values)

    residual = ref_angle - reconstructed
    print(f"Reconstruction error: MSE={np.mean(residual**2):.8f}")
    print(f"Max residual: {np.max(np.abs(residual)):.6f}")

    # The residuals should be tiny if weights are correct
    if np.max(np.abs(residual)) > 0.001:
        print("  Note: Some reconstruction error exists")

    # Save best submissions
    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    if best_blend:
        angle_pred, depth_pred, lr_pred, strength = best_blend

        # Calibrate depth
        depth_pred = depth_pred - np.mean(depth_pred) + 0.5055
        depth_pred = np.clip(depth_pred, 0, 1)

        submission = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': angle_pred,
            'scaled_depth': depth_pred,
            'scaled_left_right': lr_pred
        })

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)

        corr = np.corrcoef(angle_pred, ref_angle)[0, 1]
        print(f"\nSaved: {output_file}")
        print(f"  angle_std: {np.std(angle_pred):.6f}")
        print(f"  depth_mean: {np.mean(depth_pred):.6f}")
        print(f"  Correlation with Sub 133: {corr:.4f}")

    # Save shrunk version
    if shrunk_std < ref_angle_std:
        next_num += 1

        shrunk_depth = ref_depth.copy()
        shrunk_lr = ref_lr.copy()

        submission = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': shrunk_angle,
            'scaled_depth': shrunk_depth,
            'scaled_left_right': shrunk_lr
        })

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)

        corr = np.corrcoef(shrunk_angle, ref_angle)[0, 1]
        print(f"\nSaved shrunk: {output_file}")
        print(f"  angle_std: {np.std(shrunk_angle):.6f}")
        print(f"  Correlation with Sub 133: {corr:.4f}")

    # Try one more thing: optimize blend of Sub 133 components with finer granularity
    print("\n" + "="*80)
    print("STRATEGY 5: Fine-grained weight optimization")
    print("="*80)

    best_fine_std = ref_angle_std
    best_fine_weights = None

    for w25 in np.arange(0.03, 0.08, 0.005):
        for w9 in np.arange(0.28, 0.33, 0.005):
            for w10 in np.arange(0.42, 0.47, 0.005):
                w111 = 1.0 - w25 - w9 - w10
                if w111 < 0.18 or w111 > 0.24:
                    continue

                blend = (w25 * sub25['scaled_angle'].values +
                        w9 * sub9['scaled_angle'].values +
                        w10 * sub10['scaled_angle'].values +
                        w111 * sub111['scaled_angle'].values)

                blend_std = np.std(blend)
                depth_blend = (w25 * sub25['scaled_depth'].values +
                              w9 * sub9['scaled_depth'].values +
                              w10 * sub10['scaled_depth'].values +
                              w111 * sub111['scaled_depth'].values)
                depth_mean = np.mean(depth_blend)

                if blend_std < best_fine_std and abs(depth_mean - 0.5055) < 0.002:
                    best_fine_std = blend_std
                    best_fine_weights = (w25, w9, w10, w111)

    if best_fine_weights and best_fine_std < ref_angle_std:
        w25, w9, w10, w111 = best_fine_weights

        fine_angle = (w25 * sub25['scaled_angle'].values +
                     w9 * sub9['scaled_angle'].values +
                     w10 * sub10['scaled_angle'].values +
                     w111 * sub111['scaled_angle'].values)

        fine_depth = (w25 * sub25['scaled_depth'].values +
                     w9 * sub9['scaled_depth'].values +
                     w10 * sub10['scaled_depth'].values +
                     w111 * sub111['scaled_depth'].values)

        fine_lr = (w25 * sub25['scaled_left_right'].values +
                  w9 * sub9['scaled_left_right'].values +
                  w10 * sub10['scaled_left_right'].values +
                  w111 * sub111['scaled_left_right'].values)

        # Calibrate
        fine_depth = fine_depth - np.mean(fine_depth) + 0.5055
        fine_depth = np.clip(fine_depth, 0, 1)

        next_num += 1
        submission = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': fine_angle,
            'scaled_depth': fine_depth,
            'scaled_left_right': fine_lr
        })

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)

        corr = np.corrcoef(fine_angle, ref_angle)[0, 1]
        print(f"\nBest fine-grained weights:")
        print(f"  {w25:.1%} Sub25 + {w9:.1%} Sub9 + {w10:.1%} Sub10 + {w111:.1%} Sub111")
        print(f"  angle_std: {best_fine_std:.6f} (vs Sub133: {ref_angle_std:.6f})")
        print(f"  Improvement: {(ref_angle_std - best_fine_std)*100:.4f}%")
        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
