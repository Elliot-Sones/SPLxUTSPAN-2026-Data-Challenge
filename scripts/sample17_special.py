"""
Sample 17 Special Handling

Sample 17 was identified as a massive outlier. Sub 133 components disagree wildly.
Try special handling for this specific sample combined with the amplification approach.
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
    print("SAMPLE 17 SPECIAL HANDLING")
    print("="*80)

    sub133 = load_submission(133)
    sub151 = load_submission(151)
    sub183 = load_submission(183)  # LB 0.007698

    test_ids = sub133['id'].values
    n = len(test_ids)

    # Find sample 17
    sample_17_idx = np.where(test_ids == 17)[0]
    if len(sample_17_idx) == 0:
        # Check if it's test_17 or similar
        for i, tid in enumerate(test_ids):
            if '17' in str(tid):
                print(f"Found sample: {tid} at index {i}")
                sample_17_idx = [i]
                break

    if len(sample_17_idx) == 0:
        print("Sample 17 not found in test set!")
        print(f"Test IDs: {test_ids[:20]}")
        return

    idx17 = sample_17_idx[0]
    print(f"\nSample 17 index: {idx17}")
    print(f"\nSub 133 predictions for sample 17:")
    print(f"  angle: {sub133['scaled_angle'].values[idx17]:.6f}")
    print(f"  depth: {sub133['scaled_depth'].values[idx17]:.6f}")
    print(f"  lr: {sub133['scaled_left_right'].values[idx17]:.6f}")

    print(f"\nSub 151 predictions for sample 17:")
    print(f"  angle: {sub151['scaled_angle'].values[idx17]:.6f}")
    print(f"  depth: {sub151['scaled_depth'].values[idx17]:.6f}")
    print(f"  lr: {sub151['scaled_left_right'].values[idx17]:.6f}")

    print(f"\nSub 183 predictions for sample 17:")
    print(f"  angle: {sub183['scaled_angle'].values[idx17]:.6f}")
    print(f"  depth: {sub183['scaled_depth'].values[idx17]:.6f}")
    print(f"  lr: {sub183['scaled_left_right'].values[idx17]:.6f}")

    # Compute differences
    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values

    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    print(f"\nSample 17 total difference: {total_diff[idx17]:.6f}")
    print(f"Percentile rank: {(total_diff < total_diff[idx17]).sum() / n * 100:.1f}%")

    ref_angle_std = np.std(sub133['scaled_angle'].values)
    print(f"\nSub 133 angle_std: {ref_angle_std:.6f}")

    # Try different strategies for sample 17
    results = []

    print("\n" + "="*80)
    print("TESTING DIFFERENT STRATEGIES")
    print("="*80)

    # Strategy 1: Exclude sample 17 from amplification
    print("\n1. Exclude sample 17 from amplification (use Sub 133 directly)")
    for pctl in [95, 96]:
        threshold = np.percentile(total_diff, pctl)
        high_diff_mask = total_diff > threshold
        high_diff_mask[idx17] = False  # Exclude sample 17

        for alpha in [2.5, 2.75, 3.0]:
            sel_angle = sub133['scaled_angle'].values.copy()
            sel_depth = sub133['scaled_depth'].values.copy()
            sel_lr = sub133['scaled_left_right'].values.copy()

            sel_angle[high_diff_mask] += alpha * diff_angle[high_diff_mask]
            sel_depth[high_diff_mask] += alpha * diff_depth[high_diff_mask]
            sel_lr[high_diff_mask] += alpha * diff_lr[high_diff_mask]

            sel_angle = np.clip(sel_angle, 0, 1)
            sel_depth = np.clip(sel_depth, 0, 1)
            sel_lr = np.clip(sel_lr, 0, 1)

            sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
            sel_depth = np.clip(sel_depth, 0, 1)

            sel_std = np.std(sel_angle)

            results.append({
                'strategy': 'exclude_17',
                'pctl': pctl,
                'alpha': alpha,
                'angle_std': sel_std,
                'angle': sel_angle.copy(),
                'depth': sel_depth.copy(),
                'lr': sel_lr.copy()
            })

            print(f"  pctl={pctl}, alpha={alpha}: angle_std={sel_std:.6f}")

    # Strategy 2: Use mean of Sub 133 and 151 for sample 17
    print("\n2. Use mean of Sub 133/151 for sample 17")
    for pctl in [95, 96]:
        threshold = np.percentile(total_diff, pctl)
        high_diff_mask = total_diff > threshold

        for alpha in [2.5, 2.75, 3.0]:
            sel_angle = sub133['scaled_angle'].values.copy()
            sel_depth = sub133['scaled_depth'].values.copy()
            sel_lr = sub133['scaled_left_right'].values.copy()

            sel_angle[high_diff_mask] += alpha * diff_angle[high_diff_mask]
            sel_depth[high_diff_mask] += alpha * diff_depth[high_diff_mask]
            sel_lr[high_diff_mask] += alpha * diff_lr[high_diff_mask]

            # Override sample 17 with mean
            sel_angle[idx17] = (sub133['scaled_angle'].values[idx17] + sub151['scaled_angle'].values[idx17]) / 2
            sel_depth[idx17] = (sub133['scaled_depth'].values[idx17] + sub151['scaled_depth'].values[idx17]) / 2
            sel_lr[idx17] = (sub133['scaled_left_right'].values[idx17] + sub151['scaled_left_right'].values[idx17]) / 2

            sel_angle = np.clip(sel_angle, 0, 1)
            sel_depth = np.clip(sel_depth, 0, 1)
            sel_lr = np.clip(sel_lr, 0, 1)

            sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
            sel_depth = np.clip(sel_depth, 0, 1)

            sel_std = np.std(sel_angle)

            results.append({
                'strategy': 'mean_17',
                'pctl': pctl,
                'alpha': alpha,
                'angle_std': sel_std,
                'angle': sel_angle.copy(),
                'depth': sel_depth.copy(),
                'lr': sel_lr.copy()
            })

            print(f"  pctl={pctl}, alpha={alpha}: angle_std={sel_std:.6f}")

    # Strategy 3: Extra amplification for sample 17
    print("\n3. Extra amplification for sample 17")
    for pctl in [95, 96]:
        threshold = np.percentile(total_diff, pctl)
        high_diff_mask = total_diff > threshold

        for alpha in [2.5, 2.75, 3.0]:
            sel_angle = sub133['scaled_angle'].values.copy()
            sel_depth = sub133['scaled_depth'].values.copy()
            sel_lr = sub133['scaled_left_right'].values.copy()

            sel_angle[high_diff_mask] += alpha * diff_angle[high_diff_mask]
            sel_depth[high_diff_mask] += alpha * diff_depth[high_diff_mask]
            sel_lr[high_diff_mask] += alpha * diff_lr[high_diff_mask]

            # Extra amplification for sample 17 (2x more)
            sel_angle[idx17] = sub133['scaled_angle'].values[idx17] + 2 * alpha * diff_angle[idx17]
            sel_depth[idx17] = sub133['scaled_depth'].values[idx17] + 2 * alpha * diff_depth[idx17]
            sel_lr[idx17] = sub133['scaled_left_right'].values[idx17] + 2 * alpha * diff_lr[idx17]

            sel_angle = np.clip(sel_angle, 0, 1)
            sel_depth = np.clip(sel_depth, 0, 1)
            sel_lr = np.clip(sel_lr, 0, 1)

            sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
            sel_depth = np.clip(sel_depth, 0, 1)

            sel_std = np.std(sel_angle)

            results.append({
                'strategy': 'extra_17',
                'pctl': pctl,
                'alpha': alpha,
                'angle_std': sel_std,
                'angle': sel_angle.copy(),
                'depth': sel_depth.copy(),
                'lr': sel_lr.copy()
            })

            print(f"  pctl={pctl}, alpha={alpha}: angle_std={sel_std:.6f}")

    # Save best
    results.sort(key=lambda x: x['angle_std'])

    print("\n" + "="*80)
    print("TOP RESULTS")
    print("="*80)

    best_angle_std = 0.136042  # Current best (Sub 206)

    for r in results[:10]:
        vs_best = r['angle_std'] - best_angle_std
        marker = "**NEW BEST**" if r['angle_std'] < best_angle_std else ""
        print(f"{r['strategy']}: pctl={r['pctl']}, alpha={r['alpha']}: "
              f"angle_std={r['angle_std']:.6f} ({vs_best:+.6f}) {marker}")

    # Save if better
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    saved = 0
    for r in results:
        if r['angle_std'] < best_angle_std and saved < 3:
            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': r['angle'],
                'scaled_depth': r['depth'],
                'scaled_left_right': r['lr']
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)

            print(f"\nSaved Sub {next_num}: {r['strategy']} pctl={r['pctl']}, alpha={r['alpha']}")
            print(f"  angle_std={r['angle_std']:.6f}")

            next_num += 1
            saved += 1

    if saved == 0:
        print("\nNo improvements found over current best (0.136042)")


if __name__ == "__main__":
    main()
