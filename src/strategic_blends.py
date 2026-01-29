"""
Strategic Blending: Create intelligent blends using high-depth_max candidates.

Based on our analysis:
- depth_max has r=-0.986 correlation with LB score
- Sub25 (0.008305): depth_max=0.7447
- Sub41 (depth_boosted): depth_max=0.7806 (+0.036)
- Sub43 (GB): depth_max=0.7829 (+0.038)

If the relationship is approximately linear:
  delta_score ~ -0.055 * delta_depth_max
Then:
  Sub41 expected improvement: 0.036 * 0.055 = 0.002 -> 0.0063
  Sub43 expected improvement: 0.038 * 0.055 = 0.002 -> 0.0062

But this assumes perfect correlation. More realistic: blend with proven performers.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Known LB scores
LB_SCORES = {
    8: 0.010220,
    9: 0.009109,
    10: 0.008907,
    11: 0.009848,
    20: 0.008619,
    25: 0.008305,  # best - 50/50 blend
    34: 0.008377,
}


def load_submission(num):
    """Load a submission by number."""
    return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")


def save_submission(df, description=""):
    """Save submission with next number."""
    nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
            if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    df.to_csv(filepath, index=False)

    print(f"\nSubmission {next_num}: {description}")
    print(f"File: {filepath}")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
              f"min={df[col].min():.4f}, max={df[col].max():.4f}")

    return next_num


def blend(subs, weights):
    """Blend multiple submissions with given weights."""
    result = subs[0].copy()
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        result[col] = sum(w * s[col] for w, s in zip(weights, subs))
    return result


def main():
    print("="*70)
    print("STRATEGIC BLENDING")
    print("="*70)

    # Load key submissions
    sub9 = load_submission(9)   # Ensemble: 0.009109
    sub10 = load_submission(10) # Optuna: 0.008907
    sub25 = load_submission(25) # Best: 0.008305 (50-50 blend of 9+10)
    sub41 = load_submission(41) # Depth-boosted: high depth_max
    sub43 = load_submission(43) # Gradient Boosting: highest depth_max

    print("\nSubmission statistics:")
    print(f"{'Sub':<6} {'depth_max':>12} {'depth_mean':>12} {'angle_std':>12}")
    print("-"*48)
    for name, df in [('9', sub9), ('10', sub10), ('25', sub25), ('41', sub41), ('43', sub43)]:
        print(f"{name:<6} {df['scaled_depth'].max():>12.4f} "
              f"{df['scaled_depth'].mean():>12.4f} "
              f"{df['scaled_angle'].std():>12.4f}")

    candidates = []

    # Strategy 1: Blend sub25 (proven best) with sub41 (high depth)
    print("\n" + "="*70)
    print("STRATEGY 1: Blend sub25 + sub41")
    print("="*70)

    for w25 in [0.7, 0.5, 0.3]:
        w41 = 1 - w25
        blended = blend([sub25, sub41], [w25, w41])
        desc = f"{w25*100:.0f}% sub25 + {w41*100:.0f}% sub41"
        print(f"\n{desc}:")
        print(f"  depth_max = {blended['scaled_depth'].max():.4f}")
        candidates.append((blended, desc))

    # Strategy 2: Blend sub25 with sub43
    print("\n" + "="*70)
    print("STRATEGY 2: Blend sub25 + sub43")
    print("="*70)

    for w25 in [0.7, 0.5, 0.3]:
        w43 = 1 - w25
        blended = blend([sub25, sub43], [w25, w43])
        desc = f"{w25*100:.0f}% sub25 + {w43*100:.0f}% sub43"
        print(f"\n{desc}:")
        print(f"  depth_max = {blended['scaled_depth'].max():.4f}")
        candidates.append((blended, desc))

    # Strategy 3: Three-way blend: sub9 + sub10 + sub43
    print("\n" + "="*70)
    print("STRATEGY 3: Three-way blend sub9 + sub10 + sub43")
    print("="*70)

    # Equal weights
    blended = blend([sub9, sub10, sub43], [1/3, 1/3, 1/3])
    desc = "33% sub9 + 33% sub10 + 33% sub43"
    print(f"\n{desc}:")
    print(f"  depth_max = {blended['scaled_depth'].max():.4f}")
    candidates.append((blended, desc))

    # Weight toward sub43
    blended = blend([sub9, sub10, sub43], [0.25, 0.25, 0.5])
    desc = "25% sub9 + 25% sub10 + 50% sub43"
    print(f"\n{desc}:")
    print(f"  depth_max = {blended['scaled_depth'].max():.4f}")
    candidates.append((blended, desc))

    # Strategy 4: Target-specific blending
    print("\n" + "="*70)
    print("STRATEGY 4: Target-specific blend")
    print("="*70)

    # Use sub25 for angle/left_right, but sub43 for depth
    target_blend = sub25.copy()
    target_blend['scaled_depth'] = 0.3 * sub25['scaled_depth'] + 0.7 * sub43['scaled_depth']
    desc = "sub25 angle/lr + 70% sub43 depth"
    print(f"\n{desc}:")
    print(f"  depth_max = {target_blend['scaled_depth'].max():.4f}")
    candidates.append((target_blend, desc))

    # Strategy 5: Use sub41's depth directly with sub25's angle/lr
    target_blend2 = sub25.copy()
    target_blend2['scaled_depth'] = sub41['scaled_depth']
    desc = "sub25 angle/lr + sub41 depth"
    print(f"\n{desc}:")
    print(f"  depth_max = {target_blend2['scaled_depth'].max():.4f}")
    candidates.append((target_blend2, desc))

    # Strategy 6: Conservative blend - 80% sub25 + 20% sub43
    blended = blend([sub25, sub43], [0.8, 0.2])
    desc = "80% sub25 + 20% sub43 (conservative)"
    print(f"\n{desc}:")
    print(f"  depth_max = {blended['scaled_depth'].max():.4f}")
    candidates.append((blended, desc))

    # Analyze all candidates
    print("\n" + "="*70)
    print("CANDIDATE ANALYSIS")
    print("="*70)

    # Sort by depth_max
    candidates_sorted = sorted(candidates, key=lambda x: x[0]['scaled_depth'].max(), reverse=True)

    print(f"\n{'Rank':<6} {'depth_max':>10} {'Description':<50}")
    print("-"*70)
    for i, (df, desc) in enumerate(candidates_sorted):
        print(f"{i+1:<6} {df['scaled_depth'].max():>10.4f} {desc:<50}")

    # Select top candidates to save
    print("\n" + "="*70)
    print("SAVING TOP CANDIDATES")
    print("="*70)

    # Save top 3 by depth_max
    for i, (df, desc) in enumerate(candidates_sorted[:3]):
        save_submission(df, desc)

    # Also save the target-specific blend if not in top 3
    for df, desc in candidates:
        if "sub41 depth" in desc or "70% sub43 depth" in desc:
            already_saved = any(desc == d for _, d in candidates_sorted[:3])
            if not already_saved:
                save_submission(df, desc)

    print("\n" + "="*70)
    print("RECOMMENDATION FOR SUBMISSION")
    print("="*70)

    best_df, best_desc = candidates_sorted[0]
    print(f"\nHighest depth_max candidate: {best_desc}")
    print(f"depth_max = {best_df['scaled_depth'].max():.4f}")

    # Estimate expected score based on correlation
    # sub25: depth_max=0.7447, score=0.008305
    # Correlation coefficient: -0.986
    # Rough estimate: each 0.01 increase in depth_max reduces score by ~0.0003

    depth_increase = best_df['scaled_depth'].max() - 0.7447
    estimated_improvement = depth_increase * 0.05  # Conservative estimate
    estimated_score = 0.008305 - estimated_improvement

    print(f"\nEstimated LB score (assuming linear relationship):")
    print(f"  depth_max increase: {depth_increase:+.4f}")
    print(f"  Estimated score: {estimated_score:.4f}")

    if estimated_score < 0.007:
        print(f"\n*** This could potentially beat 0.007! ***")
    else:
        print(f"\n  Gap to 0.007: {estimated_score - 0.007:.4f}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
