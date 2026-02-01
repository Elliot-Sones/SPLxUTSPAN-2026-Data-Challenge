"""
Combined Amplify and Blend

Try combining the amplification approach with blending other good submissions.
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
    print("COMBINED AMPLIFY AND BLEND")
    print("="*80)

    # Load key submissions
    sub133 = load_submission(133)
    sub151 = load_submission(151)
    sub183 = load_submission(183)  # Confirmed LB 0.007698
    sub201 = load_submission(201)  # Best angle_std so far

    test_ids = sub133['id'].values
    n = len(test_ids)

    ref_angle_std = np.std(sub133['scaled_angle'].values)
    sub183_std = np.std(sub183['scaled_angle'].values)
    sub201_std = np.std(sub201['scaled_angle'].values)

    print(f"\nSub 133 angle_std: {ref_angle_std:.6f}")
    print(f"Sub 183 angle_std: {sub183_std:.6f} (LB 0.007698)")
    print(f"Sub 201 angle_std: {sub201_std:.6f} (best candidate)")

    # Check correlation between Sub 183 and Sub 201
    corr = np.corrcoef(sub183['scaled_angle'].values, sub201['scaled_angle'].values)[0, 1]
    print(f"\nCorrelation 183 vs 201: {corr:.4f}")

    results = []

    # Strategy 1: Blend Sub 183 and Sub 201
    print("\n" + "="*80)
    print("STRATEGY 1: BLEND 183 (confirmed) AND 201 (best angle_std)")
    print("="*80)

    for w183 in np.arange(0, 1.05, 0.1):
        w201 = 1 - w183

        blend_angle = w183 * sub183['scaled_angle'].values + w201 * sub201['scaled_angle'].values
        blend_depth = w183 * sub183['scaled_depth'].values + w201 * sub201['scaled_depth'].values
        blend_lr = w183 * sub183['scaled_left_right'].values + w201 * sub201['scaled_left_right'].values

        # Calibrate depth
        blend_depth = blend_depth - np.mean(blend_depth) + 0.5055
        blend_depth = np.clip(blend_depth, 0, 1)

        blend_std = np.std(blend_angle)

        results.append({
            'strategy': 'blend_183_201',
            'w183': w183,
            'w201': w201,
            'angle_std': blend_std,
            'angle': blend_angle.copy(),
            'depth': blend_depth.copy(),
            'lr': blend_lr.copy()
        })

        print(f"  183:{w183:.0%} + 201:{w201:.0%} = angle_std={blend_std:.6f}")

    # Strategy 2: Amplify Sub 183 further (since it's confirmed)
    print("\n" + "="*80)
    print("STRATEGY 2: AMPLIFY SUB 183 (confirmed LB 0.007698)")
    print("="*80)

    diff_angle = sub183['scaled_angle'].values - sub133['scaled_angle'].values
    diff_depth = sub183['scaled_depth'].values - sub133['scaled_depth'].values
    diff_lr = sub183['scaled_left_right'].values - sub133['scaled_left_right'].values

    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    for pctl in [90, 95]:
        threshold = np.percentile(total_diff, pctl)
        high_diff_mask = total_diff > threshold
        n_samples = high_diff_mask.sum()

        for alpha in [0.5, 1.0, 1.5]:
            # Amplify 183's direction relative to 133
            sel_angle = sub183['scaled_angle'].values + alpha * diff_angle
            sel_depth = sub183['scaled_depth'].values + alpha * diff_depth
            sel_lr = sub183['scaled_left_right'].values + alpha * diff_lr

            sel_angle = np.clip(sel_angle, 0, 1)
            sel_depth = np.clip(sel_depth, 0, 1)
            sel_lr = np.clip(sel_lr, 0, 1)

            sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
            sel_depth = np.clip(sel_depth, 0, 1)

            sel_std = np.std(sel_angle)

            results.append({
                'strategy': 'amplify_183',
                'pctl': pctl,
                'alpha': alpha,
                'angle_std': sel_std,
                'angle': sel_angle.copy(),
                'depth': sel_depth.copy(),
                'lr': sel_lr.copy()
            })

            print(f"  pctl={pctl}, alpha={alpha}: angle_std={sel_std:.6f}")

    # Strategy 3: Use Sub 183 as base, apply 201's amplification pattern
    print("\n" + "="*80)
    print("STRATEGY 3: 183 BASE + 201 PATTERN")
    print("="*80)

    # Find what samples 201 modified from 133
    diff_201_133 = sub201['scaled_angle'].values - sub133['scaled_angle'].values
    modified_mask = np.abs(diff_201_133) > 0.001

    print(f"  201 modified {modified_mask.sum()} samples from 133")

    # Apply the same modification pattern to 183
    for scale in [0.5, 1.0, 1.5, 2.0]:
        sel_angle = sub183['scaled_angle'].values.copy()
        sel_depth = sub183['scaled_depth'].values.copy()
        sel_lr = sub183['scaled_left_right'].values.copy()

        sel_angle[modified_mask] += scale * diff_201_133[modified_mask]

        diff_depth_201 = sub201['scaled_depth'].values - sub133['scaled_depth'].values
        diff_lr_201 = sub201['scaled_left_right'].values - sub133['scaled_left_right'].values

        sel_depth[modified_mask] += scale * diff_depth_201[modified_mask]
        sel_lr[modified_mask] += scale * diff_lr_201[modified_mask]

        sel_angle = np.clip(sel_angle, 0, 1)
        sel_depth = np.clip(sel_depth, 0, 1)
        sel_lr = np.clip(sel_lr, 0, 1)

        sel_depth = sel_depth - np.mean(sel_depth) + 0.5055
        sel_depth = np.clip(sel_depth, 0, 1)

        sel_std = np.std(sel_angle)

        results.append({
            'strategy': '183_plus_201_pattern',
            'scale': scale,
            'angle_std': sel_std,
            'angle': sel_angle.copy(),
            'depth': sel_depth.copy(),
            'lr': sel_lr.copy()
        })

        print(f"  scale={scale}: angle_std={sel_std:.6f}")

    # Sort and report
    results.sort(key=lambda x: x['angle_std'])

    print("\n" + "="*80)
    print("TOP RESULTS")
    print("="*80)

    best_angle_std = 0.136042

    for r in results[:15]:
        vs_best = r['angle_std'] - best_angle_std
        marker = "**NEW**" if r['angle_std'] < best_angle_std else ""
        desc = f"{r['strategy']}"
        if 'w183' in r:
            desc += f" w183={r['w183']:.0%}"
        elif 'pctl' in r:
            desc += f" pctl={r.get('pctl', 'N/A')}, alpha={r.get('alpha', 'N/A')}"
        elif 'scale' in r:
            desc += f" scale={r['scale']}"
        print(f"{desc}: angle_std={r['angle_std']:.6f} ({vs_best:+.6f}) {marker}")

    # Save best
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    print("\n" + "="*80)
    print("SAVING SUBMISSIONS")
    print("="*80)

    saved = 0
    for r in results:
        if saved < 5 and r['angle_std'] <= sub183_std:
            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': r['angle'],
                'scaled_depth': r['depth'],
                'scaled_left_right': r['lr']
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            submission.to_csv(output_file, index=False)

            improvement = (ref_angle_std - r['angle_std']) / ref_angle_std * 100
            print(f"\nSub {next_num}: {r['strategy']}")
            print(f"  angle_std: {r['angle_std']:.6f} ({improvement:.2f}% better than 133)")

            next_num += 1
            saved += 1


if __name__ == "__main__":
    main()
