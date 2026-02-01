"""
Aggressive Ensemble Search

Try many different ensemble combinations to find one that:
1. Has lower angle_std than Sub 133 (0.1377)
2. Has depth_mean close to 0.5055
3. Is different enough to potentially improve LB
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'
OUTPUT_DIR = PROJECT_DIR / 'output'


def load_submission(num):
    try:
        return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")
    except:
        return None


def compute_metrics(subs_dict, weights):
    """Compute blended predictions and metrics."""
    n = len(list(subs_dict.values())[0])
    angle = np.zeros(n)
    depth = np.zeros(n)
    lr = np.zeros(n)

    for sub_num, w in weights.items():
        angle += w * subs_dict[sub_num]['scaled_angle'].values
        depth += w * subs_dict[sub_num]['scaled_depth'].values
        lr += w * subs_dict[sub_num]['scaled_left_right'].values

    return {
        'angle': angle,
        'depth': depth,
        'lr': lr,
        'angle_std': np.std(angle),
        'angle_mean': np.mean(angle),
        'depth_mean': np.mean(depth),
        'lr_mean': np.mean(lr)
    }


def main():
    print("="*80)
    print("AGGRESSIVE ENSEMBLE SEARCH")
    print("="*80)

    # Load all available submissions
    sub_nums = [9, 10, 25, 111, 133]

    subs = {}
    for num in sub_nums:
        sub = load_submission(num)
        if sub is not None:
            subs[num] = sub
            print(f"Loaded Sub {num}")

    print(f"\nLoaded {len(subs)} submissions")

    # Reference: Sub 133
    sub133 = subs[133]
    ref_angle_std = np.std(sub133['scaled_angle'].values)
    ref_depth_mean = np.mean(sub133['scaled_depth'].values)

    print(f"\nSub 133 reference:")
    print(f"  angle_std:  {ref_angle_std:.6f}")
    print(f"  depth_mean: {ref_depth_mean:.6f}")

    # Try many weight combinations
    print("\n" + "="*80)
    print("SEARCHING FOR BETTER BLENDS")
    print("="*80)

    best_blends = []

    # Grid search with fine granularity
    # Sub 133 = 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111
    # Try variations around this

    for w25 in np.arange(0.0, 0.15, 0.02):
        for w9 in np.arange(0.20, 0.45, 0.02):
            for w10 in np.arange(0.35, 0.55, 0.02):
                w111 = 1.0 - w25 - w9 - w10
                if w111 < 0.10 or w111 > 0.35:
                    continue

                weights = {25: w25, 9: w9, 10: w10, 111: w111}
                metrics = compute_metrics(subs, weights)

                # Check if better than Sub 133
                if metrics['angle_std'] < ref_angle_std:
                    depth_dist = abs(metrics['depth_mean'] - 0.5055)
                    if depth_dist < 0.003:  # Within 0.003 of optimal
                        improvement = ref_angle_std - metrics['angle_std']

                        # Compute correlation with Sub 133
                        corr = np.corrcoef(metrics['angle'], sub133['scaled_angle'].values)[0, 1]

                        best_blends.append({
                            'weights': weights.copy(),
                            'angle_std': metrics['angle_std'],
                            'depth_mean': metrics['depth_mean'],
                            'improvement': improvement,
                            'corr_133': corr,
                            'predictions': (metrics['angle'].copy(),
                                          metrics['depth'].copy(),
                                          metrics['lr'].copy())
                        })

    print(f"\nFound {len(best_blends)} blends with lower angle_std than Sub 133")

    if best_blends:
        # Sort by angle_std (lower is better)
        best_blends.sort(key=lambda x: x['angle_std'])

        print("\nTop 10 blends by angle_std:")
        for i, blend in enumerate(best_blends[:10]):
            w = blend['weights']
            print(f"\n{i+1}. angle_std={blend['angle_std']:.6f} "
                  f"(improvement: {blend['improvement']*100:.4f}%)")
            print(f"   depth_mean={blend['depth_mean']:.6f}")
            print(f"   corr_133={blend['corr_133']:.4f}")
            print(f"   weights: {w[25]:.0%} Sub25 + {w[9]:.0%} Sub9 + "
                  f"{w[10]:.0%} Sub10 + {w[111]:.0%} Sub111")

        # Save the best one
        best = best_blends[0]
        print("\n" + "="*80)
        print("SAVING BEST BLEND")
        print("="*80)

        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1

        # Calibrate depth to exactly 0.5055
        angle_pred, depth_pred, lr_pred = best['predictions']
        depth_pred = depth_pred - np.mean(depth_pred) + 0.5055
        depth_pred = np.clip(depth_pred, 0, 1)

        submission = pd.DataFrame({
            'id': sub133['id'].values,
            'scaled_angle': angle_pred,
            'scaled_depth': depth_pred,
            'scaled_left_right': lr_pred
        })

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)

        w = best['weights']
        print(f"\nSaved: {output_file}")
        print(f"\nSubmission {next_num} details:")
        print(f"  Blend: {w[25]:.0%} Sub25 + {w[9]:.0%} Sub9 + {w[10]:.0%} Sub10 + {w[111]:.0%} Sub111")
        print(f"  angle_std: {np.std(angle_pred):.6f} (vs Sub133: {ref_angle_std:.6f})")
        print(f"  depth_mean: {np.mean(depth_pred):.6f}")
        print(f"  Improvement over Sub133 angle_std: {best['improvement']*100:.4f}%")

        # Also try adding a small amount of Sub 163 (per-player) for diversity
        print("\n" + "="*80)
        print("ADDING DIVERSITY FROM SUB 163")
        print("="*80)

        sub163 = load_submission(163)
        if sub163 is not None:
            # Try small amounts of Sub 163
            for div_weight in [0.02, 0.03, 0.05]:
                div_angle = div_weight * sub163['scaled_angle'].values + (1 - div_weight) * angle_pred
                div_depth = div_weight * sub163['scaled_depth'].values + (1 - div_weight) * depth_pred
                div_lr = div_weight * sub163['scaled_left_right'].values + (1 - div_weight) * lr_pred

                div_std = np.std(div_angle)
                div_corr = np.corrcoef(div_angle, sub133['scaled_angle'].values)[0, 1]

                if div_std < 0.14:
                    print(f"\n{div_weight:.0%} Sub163 + {1-div_weight:.0%} best_blend:")
                    print(f"  angle_std: {div_std:.6f}")
                    print(f"  corr_133: {div_corr:.4f}")

                    # Save this variant too
                    next_num += 1

                    # Calibrate depth
                    div_depth = div_depth - np.mean(div_depth) + 0.5055
                    div_depth = np.clip(div_depth, 0, 1)

                    sub_div = pd.DataFrame({
                        'id': sub133['id'].values,
                        'scaled_angle': div_angle,
                        'scaled_depth': div_depth,
                        'scaled_left_right': div_lr
                    })

                    div_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
                    sub_div.to_csv(div_file, index=False)
                    print(f"  Saved: {div_file}")
    else:
        print("No blends found that improve on Sub 133's angle_std")

    # Also try 5-way blend including Sub 133 itself
    print("\n" + "="*80)
    print("5-WAY BLEND INCLUDING SUB 133")
    print("="*80)

    best_5way = None
    best_5way_std = ref_angle_std

    for w133 in np.arange(0.30, 0.70, 0.05):
        remaining = 1.0 - w133
        for w25 in np.arange(0.0, remaining * 0.3, 0.02):
            for w9 in np.arange(0.0, remaining * 0.5, 0.02):
                for w10 in np.arange(0.0, remaining * 0.6, 0.02):
                    w111 = remaining - w25 - w9 - w10
                    if w111 < 0 or w111 > remaining * 0.4:
                        continue

                    weights = {133: w133, 25: w25, 9: w9, 10: w10, 111: w111}
                    metrics = compute_metrics(subs, weights)

                    if metrics['angle_std'] < best_5way_std:
                        depth_dist = abs(metrics['depth_mean'] - 0.5055)
                        if depth_dist < 0.003:
                            best_5way_std = metrics['angle_std']
                            best_5way = {
                                'weights': weights.copy(),
                                'angle_std': metrics['angle_std'],
                                'depth_mean': metrics['depth_mean'],
                                'predictions': (metrics['angle'].copy(),
                                              metrics['depth'].copy(),
                                              metrics['lr'].copy())
                            }

    if best_5way:
        w = best_5way['weights']
        print(f"\nBest 5-way blend:")
        print(f"  {w[133]:.0%} Sub133 + {w[25]:.0%} Sub25 + {w[9]:.0%} Sub9 + "
              f"{w[10]:.0%} Sub10 + {w[111]:.0%} Sub111")
        print(f"  angle_std: {best_5way['angle_std']:.6f}")
        print(f"  depth_mean: {best_5way['depth_mean']:.6f}")

        # Save
        next_num += 1
        angle_pred, depth_pred, lr_pred = best_5way['predictions']
        depth_pred = depth_pred - np.mean(depth_pred) + 0.5055
        depth_pred = np.clip(depth_pred, 0, 1)

        sub_5way = pd.DataFrame({
            'id': sub133['id'].values,
            'scaled_angle': angle_pred,
            'scaled_depth': depth_pred,
            'scaled_left_right': lr_pred
        })

        file_5way = SUBMISSION_DIR / f"submission_{next_num}.csv"
        sub_5way.to_csv(file_5way, index=False)
        print(f"\nSaved: {file_5way}")


if __name__ == "__main__":
    main()
