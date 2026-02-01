"""
Analyze Sub 133 vs Sub 151

Sub 151 was our "optimized blend" with profile distance 0.0003 (nearly perfect)
and 99.11% correlation with Sub 133, but scored 6.4% WORSE on LB (0.008305 vs 0.007809).

Goal: Understand what makes Sub 133 special vs slight variations
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'
OUTPUT_DIR = PROJECT_DIR / 'output'


def load_submission(num):
    return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")


def analyze_differences(sub_a, sub_b, name_a, name_b):
    """Analyze differences between two submissions."""
    print(f"\n{'='*60}")
    print(f"COMPARING {name_a} vs {name_b}")
    print(f"{'='*60}")

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        a = sub_a[col].values
        b = sub_b[col].values

        diff = a - b
        abs_diff = np.abs(diff)

        corr = np.corrcoef(a, b)[0, 1]
        mse = np.mean(diff ** 2)
        mae = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        max_diff_idx = np.argmax(abs_diff)

        target = col.replace('scaled_', '')
        print(f"\n{target}:")
        print(f"  Correlation: {corr:.6f}")
        print(f"  MSE: {mse:.8f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Max diff: {max_diff:.6f} at sample {max_diff_idx}")
        print(f"  {name_a} range: [{a.min():.4f}, {a.max():.4f}]")
        print(f"  {name_b} range: [{b.min():.4f}, {b.max():.4f}]")

        # Distribution of differences
        print(f"  Diff percentiles: p10={np.percentile(diff, 10):.5f}, "
              f"p50={np.percentile(diff, 50):.5f}, p90={np.percentile(diff, 90):.5f}")

        # Find samples with largest differences
        largest_diff_idx = np.argsort(abs_diff)[-5:][::-1]
        print(f"  Top 5 different samples:")
        for idx in largest_diff_idx:
            print(f"    Sample {idx}: {name_a}={a[idx]:.4f}, {name_b}={b[idx]:.4f}, diff={diff[idx]:.4f}")


def main():
    print("="*80)
    print("ANALYSIS: SUB 133 vs SUB 151")
    print("="*80)

    # Load submissions
    sub133 = load_submission(133)
    sub151 = load_submission(151)

    print("\nSubmission 133 (LB 0.007809 - BEST):")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        vals = sub133[col].values
        print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    print("\nSubmission 151 (LB 0.008305 - 6.4% worse):")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        vals = sub151[col].values
        print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Detailed comparison
    analyze_differences(sub133, sub151, "Sub133", "Sub151")

    # Which samples differ most?
    print("\n" + "="*80)
    print("SAMPLE-LEVEL ANALYSIS")
    print("="*80)

    # Compute total difference per sample
    angle_diff = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    depth_diff = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    lr_diff = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values

    total_diff = np.sqrt(angle_diff**2 + depth_diff**2 + lr_diff**2)

    print(f"\nTotal diff per sample (Euclidean):")
    print(f"  Mean: {total_diff.mean():.6f}")
    print(f"  Std: {total_diff.std():.6f}")
    print(f"  Max: {total_diff.max():.6f}")

    # Top 10 most different samples
    top_diff_idx = np.argsort(total_diff)[-10:][::-1]
    print(f"\nTop 10 most different samples:")
    for i, idx in enumerate(top_diff_idx):
        print(f"  {i+1}. Sample {idx} (ID: {sub133['id'].iloc[idx][:8]}...): "
              f"total_diff={total_diff[idx]:.4f}")
        print(f"      angle: {sub133['scaled_angle'].iloc[idx]:.4f} vs {sub151['scaled_angle'].iloc[idx]:.4f}")
        print(f"      depth: {sub133['scaled_depth'].iloc[idx]:.4f} vs {sub151['scaled_depth'].iloc[idx]:.4f}")
        print(f"      lr:    {sub133['scaled_left_right'].iloc[idx]:.4f} vs {sub151['scaled_left_right'].iloc[idx]:.4f}")

    # Is the difference systematic (bias) or random?
    print("\n" + "="*80)
    print("BIAS ANALYSIS")
    print("="*80)

    print(f"\nAngle: Sub133 - Sub151 mean bias = {angle_diff.mean():.6f}")
    print(f"Depth: Sub133 - Sub151 mean bias = {depth_diff.mean():.6f}")
    print(f"LR:    Sub133 - Sub151 mean bias = {lr_diff.mean():.6f}")

    if abs(angle_diff.mean()) > 0.001:
        print(f"\n  Angle has systematic bias: Sub133 predicts {angle_diff.mean():.4f} higher")
    if abs(depth_diff.mean()) > 0.001:
        print(f"  Depth has systematic bias: Sub133 predicts {depth_diff.mean():.4f} higher")
    if abs(lr_diff.mean()) > 0.001:
        print(f"  LR has systematic bias: Sub133 predicts {lr_diff.mean():.4f} higher")

    # Segment analysis
    print("\n" + "="*80)
    print("SEGMENT ANALYSIS")
    print("="*80)

    # Split by angle prediction quartile
    angle_133 = sub133['scaled_angle'].values
    for q, (lo, hi) in enumerate([(0, 25), (25, 50), (50, 75), (75, 100)]):
        lo_val = np.percentile(angle_133, lo)
        hi_val = np.percentile(angle_133, hi)
        mask = (angle_133 >= lo_val) & (angle_133 < hi_val + 0.001)

        segment_diff = angle_diff[mask]
        print(f"\nAngle Q{q+1} ({lo_val:.3f}-{hi_val:.3f}): n={mask.sum()}")
        print(f"  Mean diff: {segment_diff.mean():.6f}")
        print(f"  Std diff: {segment_diff.std():.6f}")

    # Compare other submissions to understand the winning blend
    print("\n" + "="*80)
    print("COMPARING OTHER SUBMISSIONS")
    print("="*80)

    other_subs = [9, 10, 25, 111]
    for num in other_subs:
        try:
            sub = load_submission(num)
            angle = sub['scaled_angle'].values
            corr_133 = np.corrcoef(angle, sub133['scaled_angle'].values)[0, 1]
            corr_151 = np.corrcoef(angle, sub151['scaled_angle'].values)[0, 1]
            print(f"\nSub {num}:")
            print(f"  Corr with 133: {corr_133:.4f}")
            print(f"  Corr with 151: {corr_151:.4f}")
        except Exception as e:
            print(f"\nSub {num}: Error loading - {e}")

    # Save analysis
    analysis_df = pd.DataFrame({
        'id': sub133['id'],
        'angle_133': sub133['scaled_angle'],
        'angle_151': sub151['scaled_angle'],
        'angle_diff': angle_diff,
        'depth_133': sub133['scaled_depth'],
        'depth_151': sub151['scaled_depth'],
        'depth_diff': depth_diff,
        'lr_133': sub133['scaled_left_right'],
        'lr_151': sub151['scaled_left_right'],
        'lr_diff': lr_diff,
        'total_diff': total_diff
    })

    analysis_df = analysis_df.sort_values('total_diff', ascending=False)
    analysis_df.to_csv(OUTPUT_DIR / 'sub133_vs_sub151_analysis.csv', index=False)
    print(f"\nSaved detailed analysis: {OUTPUT_DIR / 'sub133_vs_sub151_analysis.csv'}")


if __name__ == "__main__":
    main()
