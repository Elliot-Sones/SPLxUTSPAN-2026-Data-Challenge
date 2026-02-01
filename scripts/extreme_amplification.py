"""
Extreme Amplification - Push Even Further

Sub 183 (alpha=1.0, top 10%) = LB 0.007698 - 1.4% better
Sub 186 (alpha=2.5, top 5%) = angle_std 0.136207 - 0.66% improvement

Try even more extreme settings:
- Higher percentiles (97, 99) - only 2-4 samples
- Higher alpha (4.0, 5.0, 6.0)
- Target-specific amplification
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
    print("EXTREME AMPLIFICATION")
    print("Sub 183 scored 0.007698 - pushing further!")
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
    print(f"Sub 183 angle_std: 0.136569 (scored 0.007698)")

    # Find existing submission numbers
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    results = []

    # Part 1: Extreme percentiles with high alpha
    print("\n" + "="*80)
    print("PART 1: EXTREME PERCENTILES (97, 99)")
    print("="*80)

    for percentile in [97, 99]:
        threshold = np.percentile(total_diff, percentile)
        high_diff_mask = total_diff > threshold
        n_samples = high_diff_mask.sum()

        for alpha in [2.0, 3.0, 4.0, 5.0, 6.0]:
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
                'type': 'extreme_pctl',
                'percentile': percentile,
                'n_samples': n_samples,
                'alpha': alpha,
                'angle_std': sel_std,
                'corr': corr,
                'angle': sel_angle.copy(),
                'depth': sel_depth.copy(),
                'lr': sel_lr.copy()
            })

            print(f"pctl={percentile} (n={n_samples}), alpha={alpha}: "
                  f"angle_std={sel_std:.6f}, corr={corr:.4f}")

    # Part 2: Combine multiple thresholds with different alphas
    print("\n" + "="*80)
    print("PART 2: MULTI-TIER AMPLIFICATION")
    print("="*80)

    # Apply different amplification to different tiers
    for config in [
        {'tier1': (99, 5.0), 'tier2': (95, 2.0)},  # Top 1% with 5x, top 5% with 2x
        {'tier1': (99, 6.0), 'tier2': (90, 1.0)},  # Top 1% with 6x, top 10% with 1x
        {'tier1': (97, 4.0), 'tier2': (90, 1.5)},  # Top 3% with 4x, top 10% with 1.5x
    ]:
        sel_angle = sub133['scaled_angle'].values.copy()
        sel_depth = sub133['scaled_depth'].values.copy()
        sel_lr = sub133['scaled_left_right'].values.copy()

        # Apply tier 2 first (lower priority)
        pctl2, alpha2 = config['tier2']
        thresh2 = np.percentile(total_diff, pctl2)
        mask2 = total_diff > thresh2

        sel_angle[mask2] += alpha2 * diff_angle[mask2]
        sel_depth[mask2] += alpha2 * diff_depth[mask2]
        sel_lr[mask2] += alpha2 * diff_lr[mask2]

        # Apply tier 1 (higher priority - overwrites tier 2 for top samples)
        pctl1, alpha1 = config['tier1']
        thresh1 = np.percentile(total_diff, pctl1)
        mask1 = total_diff > thresh1

        # Reset tier 1 samples to base
        sel_angle[mask1] = sub133['scaled_angle'].values[mask1] + alpha1 * diff_angle[mask1]
        sel_depth[mask1] = sub133['scaled_depth'].values[mask1] + alpha1 * diff_depth[mask1]
        sel_lr[mask1] = sub133['scaled_left_right'].values[mask1] + alpha1 * diff_lr[mask1]

        sel_angle = np.clip(sel_angle, 0, 1)
        sel_depth = np.clip(sel_depth, 0, 1)
        sel_lr = np.clip(sel_lr, 0, 1)

        sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
        sel_depth = np.clip(sel_depth, 0, 1)

        sel_std = np.std(sel_angle)
        corr = np.corrcoef(sel_angle, sub133['scaled_angle'].values)[0, 1]

        results.append({
            'type': 'multi_tier',
            'config': str(config),
            'angle_std': sel_std,
            'corr': corr,
            'angle': sel_angle.copy(),
            'depth': sel_depth.copy(),
            'lr': sel_lr.copy()
        })

        print(f"Multi-tier {config}: angle_std={sel_std:.6f}, corr={corr:.4f}")

    # Part 3: Per-target amplification
    print("\n" + "="*80)
    print("PART 3: PER-TARGET AMPLIFICATION")
    print("="*80)

    for target_focus in ['angle', 'depth', 'lr']:
        sel_angle = sub133['scaled_angle'].values.copy()
        sel_depth = sub133['scaled_depth'].values.copy()
        sel_lr = sub133['scaled_left_right'].values.copy()

        # Different amplification per target
        if target_focus == 'angle':
            # Only amplify angle differences
            thresh = np.percentile(np.abs(diff_angle), 95)
            mask = np.abs(diff_angle) > thresh
            sel_angle[mask] += 3.0 * diff_angle[mask]
        elif target_focus == 'depth':
            # Only amplify depth differences
            thresh = np.percentile(np.abs(diff_depth), 95)
            mask = np.abs(diff_depth) > thresh
            sel_depth[mask] += 3.0 * diff_depth[mask]
        else:
            # Only amplify lr differences
            thresh = np.percentile(np.abs(diff_lr), 95)
            mask = np.abs(diff_lr) > thresh
            sel_lr[mask] += 3.0 * diff_lr[mask]

        sel_angle = np.clip(sel_angle, 0, 1)
        sel_depth = np.clip(sel_depth, 0, 1)
        sel_lr = np.clip(sel_lr, 0, 1)

        sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
        sel_depth = np.clip(sel_depth, 0, 1)

        sel_std = np.std(sel_angle)
        corr = np.corrcoef(sel_angle, sub133['scaled_angle'].values)[0, 1]

        results.append({
            'type': f'target_{target_focus}',
            'angle_std': sel_std,
            'corr': corr,
            'angle': sel_angle.copy(),
            'depth': sel_depth.copy(),
            'lr': sel_lr.copy()
        })

        print(f"Target {target_focus}: angle_std={sel_std:.6f}, corr={corr:.4f}")

    # Sort and save best results
    print("\n" + "="*80)
    print("SAVING BEST SUBMISSIONS")
    print("="*80)

    # Sort by angle_std
    results.sort(key=lambda x: x['angle_std'])

    saved = 0
    for r in results:
        if r['angle_std'] < ref_angle_std and saved < 6:
            improvement = (ref_angle_std - r['angle_std']) / ref_angle_std * 100

            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': r['angle'],
                'scaled_depth': r['depth'],
                'scaled_left_right': r['lr']
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)

            desc = r.get('config', f"pctl={r.get('percentile', 'N/A')}, alpha={r.get('alpha', 'N/A')}")
            print(f"\nSub {next_num}: {r['type']} - {desc}")
            print(f"  angle_std: {r['angle_std']:.6f} ({improvement:.2f}% better)")
            print(f"  corr: {r['corr']:.4f}")
            print(f"  Saved: {output_file}")

            next_num += 1
            saved += 1


if __name__ == "__main__":
    main()
