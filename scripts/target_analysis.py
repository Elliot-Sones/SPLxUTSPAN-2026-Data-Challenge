"""
TARGET ANALYSIS

Investigate the target variables:
1. What do angle, depth, left_right represent?
2. Are they correlated with each other?
3. Is there measurement noise in the targets?
4. What's the variance within vs between players?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def main():
    print("=" * 80)
    print("TARGET ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    participant_ids = meta['participant_id'].values

    angle = y[:, 0]
    depth = y[:, 1]
    lr = y[:, 2]

    print(f"Total samples: {len(y)}")

    # ==========================================================================
    # Basic statistics
    # ==========================================================================
    print()
    print("=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)

    for name, target in [('angle', angle), ('depth', depth), ('left_right', lr)]:
        print(f"\n{name.upper()}:")
        print(f"  Mean: {target.mean():.4f}")
        print(f"  Std:  {target.std():.4f}")
        print(f"  Min:  {target.min():.4f}")
        print(f"  Max:  {target.max():.4f}")
        print(f"  Range: {target.max() - target.min():.4f}")

    # ==========================================================================
    # Correlations between targets
    # ==========================================================================
    print()
    print("=" * 80)
    print("TARGET CORRELATIONS")
    print("=" * 80)
    print()
    print("If targets are highly correlated, predicting one helps predict others")
    print()

    corr_ad = np.corrcoef(angle, depth)[0, 1]
    corr_al = np.corrcoef(angle, lr)[0, 1]
    corr_dl = np.corrcoef(depth, lr)[0, 1]

    print(f"Correlation(angle, depth):      {corr_ad:+.4f}")
    print(f"Correlation(angle, left_right): {corr_al:+.4f}")
    print(f"Correlation(depth, left_right): {corr_dl:+.4f}")

    # ==========================================================================
    # Per-player statistics
    # ==========================================================================
    print()
    print("=" * 80)
    print("PER-PLAYER TARGET STATISTICS")
    print("=" * 80)
    print()
    print("Higher variance = harder to predict")
    print()

    print(f"{'Player':<8} {'N':<6} {'angle_std':<12} {'depth_std':<12} {'lr_std':<12}")
    print("-" * 60)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        n = mask.sum()

        a_std = angle[mask].std()
        d_std = depth[mask].std()
        l_std = lr[mask].std()

        print(f"{pid:<8} {n:<6} {a_std:<12.4f} {d_std:<12.4f} {l_std:<12.4f}")

    # ==========================================================================
    # Variance decomposition
    # ==========================================================================
    print()
    print("=" * 80)
    print("VARIANCE DECOMPOSITION")
    print("=" * 80)
    print()
    print("How much variance is BETWEEN players vs WITHIN players?")
    print()

    for name, target in [('angle', angle), ('depth', depth), ('left_right', lr)]:
        # Total variance
        total_var = target.var()

        # Between-player variance (variance of player means)
        player_means = [target[participant_ids == pid].mean() for pid in np.unique(participant_ids)]
        between_var = np.var(player_means)

        # Within-player variance (average of within-player variances)
        within_vars = [target[participant_ids == pid].var() for pid in np.unique(participant_ids)]
        within_var = np.mean(within_vars)

        between_pct = between_var / total_var * 100
        within_pct = within_var / total_var * 100

        print(f"{name.upper()}:")
        print(f"  Total variance:   {total_var:.4f}")
        print(f"  Between players:  {between_var:.4f} ({between_pct:.1f}%)")
        print(f"  Within players:   {within_var:.4f} ({within_pct:.1f}%)")
        print()

    # ==========================================================================
    # Check for outliers
    # ==========================================================================
    print()
    print("=" * 80)
    print("OUTLIER ANALYSIS")
    print("=" * 80)
    print()

    for name, target in [('angle', angle), ('depth', depth), ('left_right', lr)]:
        z_scores = np.abs(stats.zscore(target))
        outliers = (z_scores > 3).sum()
        print(f"{name}: {outliers} outliers (|z| > 3)")

    # ==========================================================================
    # Per-player correlations
    # ==========================================================================
    print()
    print("=" * 80)
    print("PER-PLAYER TARGET CORRELATIONS")
    print("=" * 80)
    print()
    print("If correlation varies by player, the physics relationship differs")
    print()

    print(f"{'Player':<8} {'angle-depth':<15} {'angle-lr':<15} {'depth-lr':<15}")
    print("-" * 60)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        a = angle[mask]
        d = depth[mask]
        l = lr[mask]

        c_ad = np.corrcoef(a, d)[0, 1]
        c_al = np.corrcoef(a, l)[0, 1]
        c_dl = np.corrcoef(d, l)[0, 1]

        print(f"{pid:<8} {c_ad:+.4f}         {c_al:+.4f}         {c_dl:+.4f}")

    # ==========================================================================
    # What does each target likely represent?
    # ==========================================================================
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("Based on the statistics:")
    print()

    # Check if targets are centered around 0.5 (likely normalized error from center)
    for name, target in [('angle', angle), ('depth', depth), ('left_right', lr)]:
        mean = target.mean()
        if 0.4 < mean < 0.6:
            print(f"{name}: Mean={mean:.3f} - likely centered/normalized (0.5 = center/target)")
        else:
            print(f"{name}: Mean={mean:.3f} - may represent actual physical measurement")

    print()
    print("Likely interpretation:")
    print("- angle: Release angle or vertical deviation from ideal")
    print("- depth: Shot distance - short vs long (0.5 might be perfect distance)")
    print("- left_right: Horizontal deviation (0.5 might be center)")
    print()
    print("If targets represent DEVIATION from ideal, lower variance = better player")


if __name__ == "__main__":
    main()
