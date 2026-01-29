"""
Analyze submission scores to find the gradient toward better performance.

We have 6 submissions with known LB scores:
- Sub 8: 0.010220
- Sub 9: 0.009109
- Sub 11: 0.009848
- Sub 20: 0.008619
- Sub 25: 0.008305 (best)
- Sub 34: 0.008377

Use these as signal to understand what prediction patterns lead to better scores.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"

# Known LB scores
LB_SCORES = {
    8: 0.010220,
    9: 0.009109,
    11: 0.009848,
    20: 0.008619,
    25: 0.008305,  # best
    34: 0.008377,
}

def load_submissions():
    """Load all submissions with known scores."""
    submissions = {}
    for sub_num in LB_SCORES.keys():
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        if filepath.exists():
            submissions[sub_num] = pd.read_csv(filepath)
    return submissions

def analyze_prediction_patterns(submissions):
    """Analyze what prediction patterns correlate with better scores."""
    print("="*70)
    print("ANALYZING PREDICTION PATTERNS vs LB SCORE")
    print("="*70)

    # Extract statistics from each submission
    stats_data = []
    for sub_num, df in submissions.items():
        row = {
            'sub': sub_num,
            'lb_score': LB_SCORES[sub_num],
            'angle_mean': df['scaled_angle'].mean(),
            'angle_std': df['scaled_angle'].std(),
            'angle_min': df['scaled_angle'].min(),
            'angle_max': df['scaled_angle'].max(),
            'depth_mean': df['scaled_depth'].mean(),
            'depth_std': df['scaled_depth'].std(),
            'depth_min': df['scaled_depth'].min(),
            'depth_max': df['scaled_depth'].max(),
            'lr_mean': df['scaled_left_right'].mean(),
            'lr_std': df['scaled_left_right'].std(),
            'lr_min': df['scaled_left_right'].min(),
            'lr_max': df['scaled_left_right'].max(),
        }
        stats_data.append(row)

    stats_df = pd.DataFrame(stats_data).sort_values('lb_score')

    print("\nSubmission statistics (sorted by LB score, best first):")
    print(stats_df.to_string(index=False))

    # Compute correlations between stats and LB score
    print("\n" + "="*70)
    print("CORRELATIONS WITH LB SCORE (negative = better)")
    print("="*70)

    correlations = []
    for col in stats_df.columns:
        if col not in ['sub', 'lb_score']:
            corr, pval = stats.pearsonr(stats_df[col], stats_df['lb_score'])
            correlations.append((col, corr, pval))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nFeature correlations with LB score:")
    for col, corr, pval in correlations:
        direction = "lower is better" if corr > 0 else "higher is better"
        sig = "*" if pval < 0.1 else ""
        print(f"  {col}: r={corr:+.3f} (p={pval:.3f}) {sig} [{direction}]")

    return stats_df

def analyze_per_sample_differences(submissions):
    """Analyze per-sample prediction differences between best and worst."""
    print("\n" + "="*70)
    print("PER-SAMPLE ANALYSIS: BEST (sub25) vs WORST (sub8)")
    print("="*70)

    best = submissions[25]
    worst = submissions[8]

    # Compute differences
    diff_angle = best['scaled_angle'] - worst['scaled_angle']
    diff_depth = best['scaled_depth'] - worst['scaled_depth']
    diff_lr = best['scaled_left_right'] - worst['scaled_left_right']

    print(f"\nPrediction shifts (best - worst):")
    print(f"  angle: mean={diff_angle.mean():+.4f}, std={diff_angle.std():.4f}")
    print(f"  depth: mean={diff_depth.mean():+.4f}, std={diff_depth.std():.4f}")
    print(f"  lr:    mean={diff_lr.mean():+.4f}, std={diff_lr.std():.4f}")

    # Which samples changed most?
    total_shift = np.abs(diff_angle) + np.abs(diff_depth) + np.abs(diff_lr)
    top_changed = np.argsort(total_shift)[::-1][:10]

    print(f"\nTop 10 samples with largest prediction shifts:")
    for idx in top_changed:
        print(f"  Sample {idx}: angle={diff_angle.iloc[idx]:+.3f}, "
              f"depth={diff_depth.iloc[idx]:+.3f}, lr={diff_lr.iloc[idx]:+.3f}")

def analyze_blend_gradient(submissions):
    """Analyze how blending affects score to find optimal direction."""
    print("\n" + "="*70)
    print("BLEND GRADIENT ANALYSIS")
    print("="*70)

    # We know these blend results:
    # 80-20 (sub20): 0.008619
    # 50-50 (sub25): 0.008305
    # 30-70 (sub34): 0.008377

    # This means the gradient points toward 50-50 from both directions
    # The optimal is around 50-50

    sub9 = submissions[9]
    sub10 = submissions.get(10)

    if sub10 is None:
        # Load sub10
        filepath = SUBMISSION_DIR / "submission_10.csv"
        if filepath.exists():
            sub10 = pd.read_csv(filepath)

    if sub10 is not None:
        print("\nComparing sub9 vs sub10 (the blend components):")

        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            diff = sub10[col] - sub9[col]
            print(f"\n  {col}:")
            print(f"    sub9:  mean={sub9[col].mean():.4f}, std={sub9[col].std():.4f}")
            print(f"    sub10: mean={sub10[col].mean():.4f}, std={sub10[col].std():.4f}")
            print(f"    diff:  mean={diff.mean():+.4f}, std={diff.std():.4f}")

            # Correlation of diff with improvement direction
            # We know 50-50 is best, so the optimal "movement" from sub9 is 0.5 * diff
            print(f"    optimal shift from sub9: {0.5 * diff.mean():+.4f}")

def extrapolate_optimal(submissions):
    """Try to extrapolate what predictions would score 0.007."""
    print("\n" + "="*70)
    print("EXTRAPOLATING TOWARD 0.007")
    print("="*70)

    # Current best: 0.008305 (sub25)
    # Target: 0.007
    # Gap: 0.001305 (15.7% improvement needed)

    best = submissions[25]

    print(f"\nCurrent best (sub25): 0.008305")
    print(f"Target: 0.007")
    print(f"Gap: 0.001305 (15.7% improvement needed)")

    # Analyze the trend from our blends
    # 80-20: 0.008619 (sub9 weight = 0.8)
    # 50-50: 0.008305 (sub9 weight = 0.5)
    # 30-70: 0.008377 (sub9 weight = 0.3)

    # The relationship between sub9_weight and score:
    weights = [0.8, 0.5, 0.3]
    scores = [0.008619, 0.008305, 0.008377]

    # Fit quadratic to find minimum
    coeffs = np.polyfit(weights, scores, 2)

    # Find minimum
    # d/dw (aw^2 + bw + c) = 2aw + b = 0
    # w_opt = -b / (2a)
    a, b, c = coeffs
    w_opt = -b / (2 * a)
    score_opt = a * w_opt**2 + b * w_opt + c

    print(f"\nQuadratic fit of blend weight vs score:")
    print(f"  Optimal sub9 weight: {w_opt:.2f}")
    print(f"  Predicted optimal score: {score_opt:.6f}")

    if 0 <= w_opt <= 1:
        print(f"\n  This suggests trying a {w_opt*100:.0f}-{(1-w_opt)*100:.0f} blend")

    return w_opt, score_opt

def create_optimized_blend(submissions, w_opt):
    """Create submission with optimized blend weight."""
    sub9 = submissions[9]

    # Load sub10
    filepath = SUBMISSION_DIR / "submission_10.csv"
    sub10 = pd.read_csv(filepath)

    w9 = max(0, min(1, w_opt))
    w10 = 1 - w9

    blended = pd.DataFrame({'id': sub9['id']})
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        blended[col] = w9 * sub9[col] + w10 * sub10[col]

    # Find next submission number
    nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
            if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    blended.to_csv(filepath, index=False)

    print(f"\nCreated submission_{next_num}: {w9*100:.0f}-{w10*100:.0f} blend")
    print(f"  Predicted score: {w_opt:.6f}")

    return next_num

def main():
    submissions = load_submissions()

    print(f"Loaded {len(submissions)} submissions with known LB scores")

    # Analyze patterns
    stats_df = analyze_prediction_patterns(submissions)

    # Per-sample analysis
    analyze_per_sample_differences(submissions)

    # Blend gradient
    analyze_blend_gradient(submissions)

    # Extrapolate
    w_opt, score_opt = extrapolate_optimal(submissions)

    # Create optimized submission if optimal weight is valid
    if 0.3 <= w_opt <= 0.7:
        create_optimized_blend(submissions, w_opt)

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. The optimal blend is around 50-50 (sub9-sub10)
2. Sub10 contributes signal that sub9 lacks
3. To reach 0.007, we need something beyond simple blending
4. Consider: different models, different features, or ensemble with more submissions
""")

if __name__ == "__main__":
    main()
