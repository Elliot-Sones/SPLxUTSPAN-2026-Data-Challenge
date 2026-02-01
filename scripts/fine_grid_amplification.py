"""
Fine Grid Amplification - Find Optimal Sweet Spot

Sub 186 (pctl=95, alpha=2.5) = angle_std 0.136207 - best so far
Sub 183 (pctl=90, alpha=1.0) = LB 0.007698 - confirmed improvement

Do a finer grid around pctl=95, alpha=2.5
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
    print("FINE GRID AMPLIFICATION")
    print("Finding optimal sweet spot around pctl=95, alpha=2.5")
    print("="*80)

    sub133 = load_submission(133)
    sub151 = load_submission(151)

    test_ids = sub133['id'].values

    # Compute difference vectors
    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values

    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    ref_angle_std = np.std(sub133['scaled_angle'].values)
    best_angle_std = 0.136207  # Sub 186

    print(f"\nSub 133 angle_std: {ref_angle_std:.6f}")
    print(f"Sub 186 angle_std: {best_angle_std:.6f} (current best)")

    # Fine grid search
    results = []

    print("\n" + "="*80)
    print("FINE GRID SEARCH")
    print("="*80)

    for percentile in np.arange(92, 98, 0.5):  # 92, 92.5, 93, ..., 97.5
        threshold = np.percentile(total_diff, percentile)
        high_diff_mask = total_diff > threshold
        n_samples = high_diff_mask.sum()

        for alpha in np.arange(1.5, 4.0, 0.25):  # 1.5, 1.75, 2.0, ..., 3.75
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

    print(f"\n{'Pctl':>6} | {'N':>3} | {'Alpha':>5} | {'angle_std':>10} | {'corr':>6} | vs 186")
    print("-"*60)

    for r in results[:20]:
        vs_186 = r['angle_std'] - best_angle_std
        marker = "**" if r['angle_std'] < best_angle_std else "  "
        print(f"{r['percentile']:>6.1f} | {r['n_samples']:>3} | {r['alpha']:>5.2f} | "
              f"{r['angle_std']:>10.6f} | {r['corr']:>6.4f} | {vs_186:>+.6f} {marker}")

    # Save best submissions that beat Sub 186
    print("\n" + "="*80)
    print("SAVING BEST SUBMISSIONS")
    print("="*80)

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    saved = 0
    for r in results:
        if r['angle_std'] < best_angle_std and saved < 5:
            improvement = (ref_angle_std - r['angle_std']) / ref_angle_std * 100

            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': r['angle'],
                'scaled_depth': r['depth'],
                'scaled_left_right': r['lr']
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)

            print(f"\nSub {next_num}: pctl={r['percentile']:.1f}, alpha={r['alpha']:.2f}")
            print(f"  angle_std: {r['angle_std']:.6f} ({improvement:.2f}% better than 133)")
            print(f"  vs Sub 186: {r['angle_std'] - best_angle_std:.6f}")
            print(f"  corr: {r['corr']:.4f}")
            print(f"  Saved: {output_file}")

            next_num += 1
            saved += 1

    if saved == 0:
        print("\nNo submissions beat Sub 186's angle_std=0.136207")
        print("Sub 186 remains the top candidate")


if __name__ == "__main__":
    main()
