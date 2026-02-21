"""
Angle-Diverse Ensemble

Key insight: Most submissions reuse Sub 784 angle predictions
Sub 1109 has different (physics-based) angles

Strategy:
1. Ensemble across angle-diverse models (784, 1109)
2. Combine with best depth/LR from 1350/1421
3. Per-target adaptive weighting
4. Uncertainty-based per-shot weighting

Target: Break angle duplication bottleneck
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submission")
RESEARCH_DIR = Path("Research")

TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]

def load_submission(sub_num: int) -> pd.DataFrame:
    """Load submission file"""
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission {sub_num} not found")
    return pd.read_csv(sub_path)

def create_angle_diverse_ensemble(
    submissions: Dict[int, pd.DataFrame],
    angle_weights: Dict[int, float],
    depth_lr_sub: int
) -> pd.DataFrame:
    """
    Create ensemble with angle diversity

    Args:
        submissions: Dict of submission dataframes
        angle_weights: Weights for angle predictions per submission
        depth_lr_sub: Submission to use for depth/LR

    Returns:
        Ensemble predictions
    """
    first_sub = list(submissions.values())[0]
    shot_ids = first_sub['id'].values
    n_test = len(shot_ids)

    ensemble_preds = np.zeros((n_test, 3))

    for i, shot_id in enumerate(shot_ids):
        # Weighted angle
        angle_pred = 0.0
        for sub_num, weight in angle_weights.items():
            sub_df = submissions[sub_num]
            pred_row = sub_df[sub_df['id'] == shot_id]
            angle_pred += weight * pred_row['scaled_angle'].values[0]

        # Depth and LR from best model
        depth_lr_df = submissions[depth_lr_sub]
        pred_row = depth_lr_df[depth_lr_df['id'] == shot_id]
        depth_pred = pred_row['scaled_depth'].values[0]
        lr_pred = pred_row['scaled_left_right'].values[0]

        ensemble_preds[i] = [angle_pred, depth_pred, lr_pred]

    result = pd.DataFrame({
        'id': shot_ids,
        'scaled_angle': ensemble_preds[:, 0],
        'scaled_depth': ensemble_preds[:, 1],
        'scaled_left_right': ensemble_preds[:, 2]
    })

    return result

def main():
    print("=" * 80)
    print("ANGLE-DIVERSE ENSEMBLE")
    print("=" * 80)

    # Load submissions
    print("\n1. Loading submissions...")

    # Key submissions:
    # 784: baseline (LB 0.007224)
    # 1109: physics angle blend (LB 0.007223) - 10% physics angle
    # 1350: best (LB 0.006776) - per-example V1, uses 784 angle
    # 1421: V2 (LB 0.006789) - per-example V2, uses 784 angle

    submissions = {}
    for sub_num in [784, 1109, 1350, 1421]:
        submissions[sub_num] = load_submission(sub_num)
        print(f"   Loaded Sub {sub_num}")

    # Check angle diversity
    print("\n2. Analyzing angle diversity...")
    first_shot_id = submissions[784]['id'].values[0]
    for sub_num in [784, 1109, 1350, 1421]:
        pred_row = submissions[sub_num][submissions[sub_num]['id'] == first_shot_id]
        angle = pred_row['scaled_angle'].values[0]
        print(f"   Sub {sub_num} angle (shot 1): {angle:.10f}")

    # Compute angle correlation
    angle_784 = submissions[784]['scaled_angle'].values
    angle_1109 = submissions[1109]['scaled_angle'].values
    angle_1350 = submissions[1350]['scaled_angle'].values

    corr_1109_784 = np.corrcoef(angle_1109, angle_784)[0, 1]
    corr_1350_784 = np.corrcoef(angle_1350, angle_784)[0, 1]

    print(f"\n   Angle correlations:")
    print(f"   - Sub 1109 vs 784: {corr_1109_784:.6f}")
    print(f"   - Sub 1350 vs 784: {corr_1350_784:.6f}")

    # Mean absolute difference
    diff_1109_784 = np.mean(np.abs(angle_1109 - angle_784))
    print(f"   - Mean |angle_1109 - angle_784|: {diff_1109_784:.6f}")

    # Generate angle-diverse ensembles
    print("\n3. Generating angle-diverse submissions...")
    submission_id = 1461

    strategies = [
        # Angle: blend 784 and 1109, Depth/LR: 1350 (best)
        ("angle_10pct_physics_depth_1350", {784: 0.90, 1109: 0.10}, 1350),
        ("angle_20pct_physics_depth_1350", {784: 0.80, 1109: 0.20}, 1350),
        ("angle_30pct_physics_depth_1350", {784: 0.70, 1109: 0.30}, 1350),
        ("angle_50pct_physics_depth_1350", {784: 0.50, 1109: 0.50}, 1350),

        # Pure physics angle with best depth/LR
        ("angle_1109_depth_1350", {1109: 1.0}, 1350),

        # Angle: blend 784 and 1109, Depth/LR: 1421 (V2)
        ("angle_10pct_physics_depth_1421", {784: 0.90, 1109: 0.10}, 1421),
        ("angle_20pct_physics_depth_1421", {784: 0.80, 1109: 0.20}, 1421),
        ("angle_50pct_physics_depth_1421", {784: 0.50, 1109: 0.50}, 1421),

        # Three-way angle blend
        ("angle_3way_uniform_depth_1350", {784: 0.33, 1109: 0.33, 1350: 0.34}, 1350),
        ("angle_3way_quality_depth_1350", {784: 0.10, 1109: 0.10, 1350: 0.80}, 1350),
    ]

    results_summary = []

    for strategy_name, angle_weights, depth_lr_sub in strategies:
        result_df = create_angle_diverse_ensemble(
            submissions, angle_weights, depth_lr_sub
        )

        # Save submission
        sub_path = SUBMISSION_DIR / f"submission_{submission_id}.csv"
        result_df.to_csv(sub_path, index=False)
        print(f"   Saved {strategy_name} to submission_{submission_id}.csv")

        # Compute correlations with Sub 1350
        ref_df = submissions[1350]
        correlations = []
        for t_idx, target in enumerate(TARGETS):
            corr = np.corrcoef(
                result_df[target].values,
                ref_df[target].values
            )[0, 1]
            correlations.append(corr)

        results_summary.append({
            'submission': submission_id,
            'strategy': strategy_name,
            'angle_weights': angle_weights,
            'depth_lr_source': depth_lr_sub,
            'correlation_with_1350': f"angle={correlations[0]:.3f}, depth={correlations[1]:.3f}, lr={correlations[2]:.3f}"
        })

        submission_id += 1

    # Save results
    print("\n4. Saving results...")
    results_md = RESEARCH_DIR / "ANGLE_DIVERSE_ENSEMBLE_RESULTS.md"

    with open(results_md, 'w') as f:
        f.write("# Angle-Diverse Ensemble Results\n\n")
        f.write("## Problem\n\n")
        f.write("Most submissions (784, 1350, 1421, 1366, etc.) use IDENTICAL angle predictions.\n")
        f.write("This limits ensemble diversity and potential improvement.\n\n")

        f.write("## Solution\n\n")
        f.write("Sub 1109 (physics-based angle) provides angle diversity:\n")
        f.write(f"- Angle correlation with Sub 784: {corr_1109_784:.6f}\n")
        f.write(f"- Mean absolute difference: {diff_1109_784:.6f}\n")
        f.write("- Sub 1109 LB: 0.007223 (nearly tied with 784 at 0.007224)\n\n")

        f.write("## Strategy\n\n")
        f.write("1. Blend angle predictions from Sub 784 (ML) and Sub 1109 (physics)\n")
        f.write("2. Use best depth/LR from Sub 1350 (LB 0.006776) or Sub 1421 (LB 0.006789)\n")
        f.write("3. Test different angle blend ratios (10%, 20%, 30%, 50%, 100% physics)\n\n")

        f.write("## Generated Submissions\n\n")
        for result in results_summary:
            f.write(f"### Sub {result['submission']}: {result['strategy']}\n")
            f.write(f"- Angle weights: {result['angle_weights']}\n")
            f.write(f"- Depth/LR source: Sub {result['depth_lr_source']}\n")
            f.write(f"- Correlation with Sub 1350: {result['correlation_with_1350']}\n\n")

        f.write("## Expected Performance\n\n")
        f.write("- If physics angle adds value: may beat Sub 1350 (0.006776)\n")
        f.write("- If physics angle hurts: will be worse than Sub 1350\n")
        f.write("- Optimal blend likely in 10-30% physics range (Sub 1109 used 10%)\n\n")

        f.write("## Key Insights\n\n")
        f.write("1. Sub 1109 angle differs from Sub 784 by mean 0.0048 (small but non-zero)\n")
        f.write("2. Physics angle correlation 0.996 with ML angle (high but not 1.0)\n")
        f.write("3. Breaking angle duplication may unlock ensemble potential\n")
        f.write("4. Per-shot adaptive angle selection could be next step\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Test submissions 1461-1470 on leaderboard\n")
        f.write("2. If improvement, implement per-shot angle selection (ML vs physics)\n")
        f.write("3. Find or create more angle-diverse models\n")
        f.write("4. Try per-shot adaptive weighting using shot similarity\n\n")

    print(f"   Saved results to {results_md}")

    print("\n" + "=" * 80)
    print(f"COMPLETED: Generated {len(strategies)} angle-diverse submissions")
    print(f"Submissions: 1461-{submission_id-1}")
    print("=" * 80)

if __name__ == "__main__":
    main()
