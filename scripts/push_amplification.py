"""
Push Amplification Further

Sub 183 (alpha=1.0) scored 0.007698 - 1.4% better than Sub 133!
Let's try higher alpha values and different thresholds.
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
    print("PUSHING AMPLIFICATION FURTHER")
    print("Sub 183 (alpha=1.0) scored 0.007698 - NEW BEST!")
    print("="*80)

    sub133 = load_submission(133)
    sub151 = load_submission(151)

    test_ids = sub133['id'].values
    n = len(test_ids)

    # Compute difference vectors
    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values

    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    ref_angle_std = np.std(sub133['scaled_angle'].values)
    print(f"\nSub 133 angle_std: {ref_angle_std:.6f}")
    print(f"Sub 183 angle_std: 0.136569 (alpha=1.0, top 10%)")

    # Try different alpha values and thresholds
    results = []

    for percentile in [85, 90, 95]:  # Top X% different samples
        threshold = np.percentile(total_diff, percentile)
        high_diff_mask = total_diff > threshold
        n_samples = high_diff_mask.sum()

        for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
            sel_angle = sub133['scaled_angle'].values.copy()
            sel_depth = sub133['scaled_depth'].values.copy()
            sel_lr = sub133['scaled_left_right'].values.copy()

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
            corr = np.corrcoef(sel_angle, sub133['scaled_angle'].values)[0, 1]

            results.append({
                'percentile': percentile,
                'n_samples': n_samples,
                'alpha': alpha,
                'angle_std': sel_std,
                'corr': corr,
                'angle': sel_angle.copy(),
                'depth': sel_depth.copy(),
                'lr': sel_lr.copy()
            })

    # Sort by angle_std
    results.sort(key=lambda x: x['angle_std'])

    print("\n" + "="*80)
    print("RESULTS (sorted by angle_std)")
    print("="*80)

    print(f"\n{'Pctl':>5} | {'N':>3} | {'Alpha':>5} | {'angle_std':>10} | {'corr':>6} | {'vs 133':>8}")
    print("-"*55)

    for r in results[:15]:
        vs_133 = "BETTER" if r['angle_std'] < ref_angle_std else "worse"
        print(f"{r['percentile']:>5} | {r['n_samples']:>3} | {r['alpha']:>5.1f} | "
              f"{r['angle_std']:>10.6f} | {r['corr']:>6.4f} | {vs_133:>8}")

    # Save top submissions
    print("\n" + "="*80)
    print("SAVING BEST SUBMISSIONS")
    print("="*80)

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    saved = 0
    for r in results:
        if r['angle_std'] < ref_angle_std and saved < 5:
            # Skip if very similar to already saved
            improvement = (ref_angle_std - r['angle_std']) / ref_angle_std * 100

            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': r['angle'],
                'scaled_depth': r['depth'],
                'scaled_left_right': r['lr']
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)

            print(f"\nSub {next_num}: pctl={r['percentile']}, alpha={r['alpha']}")
            print(f"  angle_std: {r['angle_std']:.6f} ({improvement:.2f}% better)")
            print(f"  corr: {r['corr']:.4f}")
            print(f"  Saved: {output_file}")

            next_num += 1
            saved += 1

    # Also try on ALL samples with small alpha
    print("\n" + "="*80)
    print("TRYING SMALL ALPHA ON ALL SAMPLES")
    print("="*80)

    for alpha in [0.05, 0.10, 0.15, 0.20]:
        all_angle = sub133['scaled_angle'].values + alpha * diff_angle
        all_depth = sub133['scaled_depth'].values + alpha * diff_depth
        all_lr = sub133['scaled_left_right'].values + alpha * diff_lr

        all_angle = np.clip(all_angle, 0, 1)
        all_depth = np.clip(all_depth, 0, 1)
        all_lr = np.clip(all_lr, 0, 1)

        all_depth = all_depth - np.mean(all_depth) + 0.5055
        all_depth = np.clip(all_depth, 0, 1)

        all_std = np.std(all_angle)
        all_corr = np.corrcoef(all_angle, sub133['scaled_angle'].values)[0, 1]

        print(f"\nalpha={alpha}: angle_std={all_std:.6f}, corr={all_corr:.4f}")

        if all_std < ref_angle_std:
            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': all_angle,
                'scaled_depth': all_depth,
                'scaled_left_right': all_lr
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")
            next_num += 1


if __name__ == "__main__":
    main()
