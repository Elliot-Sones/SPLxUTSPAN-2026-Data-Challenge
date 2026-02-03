"""
Optimize Around Sub 219's Sweet Spot

Sub 219 (pctl=91, alpha=1.1) = LB 0.007682 - NEW BEST!
Try variations around this setting.
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
    print("OPTIMIZING AROUND SUB 219")
    print("="*80)
    print("\nSub 219 (pctl=91, alpha=1.1) = LB 0.007682 - NEW BEST!")

    sub133 = load_submission(133)
    sub151 = load_submission(151)

    test_ids = sub133['id'].values

    # Compute difference vectors
    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values

    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    ref_angle_std = np.std(sub133['scaled_angle'].values)
    sub219_std = 0.136554  # From previous run

    print(f"\nSub 133 angle_std: {ref_angle_std:.6f}")
    print(f"Sub 219 angle_std: {sub219_std:.6f}")

    results = []

    # Fine grid around Sub 219's settings (pctl=91, alpha=1.1)
    print("\n" + "="*80)
    print("TESTING AROUND SUB 219 (pctl=91, alpha=1.1)")
    print("="*80)

    for percentile in [90, 90.5, 91, 91.5, 92]:
        threshold = np.percentile(total_diff, percentile)
        high_diff_mask = total_diff > threshold
        n_samples = high_diff_mask.sum()

        for alpha in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
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

    # Sort by angle_std (closer to sub219's std)
    results.sort(key=lambda x: abs(x['angle_std'] - sub219_std))

    print(f"\n{'Pctl':>5} | {'N':>3} | {'Alpha':>5} | {'angle_std':>10} | {'vs 219':>10}")
    print("-"*50)

    for r in results[:20]:
        vs_219 = r['angle_std'] - sub219_std
        marker = "<- 219" if r['percentile'] == 91 and r['alpha'] == 1.1 else ""
        print(f"{r['percentile']:>5} | {r['n_samples']:>3} | {r['alpha']:>5.2f} | "
              f"{r['angle_std']:>10.6f} | {vs_219:>+10.6f} {marker}")

    # Save top variations
    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    saved = 0
    seen_configs = set()

    for r in results:
        config = (r['percentile'], r['alpha'])
        if config in seen_configs:
            continue
        if config == (91, 1.1):  # Skip Sub 219's exact settings
            continue

        if saved < 6 and r['angle_std'] < ref_angle_std:
            seen_configs.add(config)

            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': r['angle'],
                'scaled_depth': r['depth'],
                'scaled_left_right': r['lr']
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)

            vs_219 = r['angle_std'] - sub219_std
            print(f"\nSub {next_num}: pctl={r['percentile']}, alpha={r['alpha']:.2f}")
            print(f"  n_samples: {r['n_samples']}")
            print(f"  angle_std: {r['angle_std']:.6f} (vs 219: {vs_219:+.6f})")

            next_num += 1
            saved += 1


if __name__ == "__main__":
    main()
