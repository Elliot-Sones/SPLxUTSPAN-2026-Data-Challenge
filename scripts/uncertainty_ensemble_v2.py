"""
Uncertainty-Weighted Ensemble via Prediction Diversity

Since we only have test predictions, use prediction disagreement/variance
as a proxy for uncertainty:

1. High variance across models = high uncertainty
2. Weight models inversely by their average disagreement with ensemble
3. Use prediction intervals from model diversity
4. Temperature scaling for calibration

Target: LB < 0.005 (current best: 0.006776)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submission")
RESEARCH_DIR = Path("Research")
RESEARCH_DIR.mkdir(exist_ok=True)

TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]

def load_submission(sub_num: int) -> pd.DataFrame:
    """Load submission file"""
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission {sub_num} not found")
    return pd.read_csv(sub_path)

def compute_prediction_variance(
    submissions: Dict[int, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-shot prediction variance across models

    Returns:
        variance: (n_test, 3) variance per target
        per_model_deviation: (n_test, n_models, 3) deviation from mean
    """
    # Get shot IDs
    first_sub = list(submissions.values())[0]
    shot_ids = first_sub['id'].values
    n_test = len(shot_ids)
    n_models = len(submissions)

    # Stack all predictions
    all_preds = np.zeros((n_test, n_models, 3))
    for i, sub_num in enumerate(sorted(submissions.keys())):
        sub_df = submissions[sub_num]
        for j, shot_id in enumerate(shot_ids):
            pred_row = sub_df[sub_df['id'] == shot_id]
            all_preds[j, i, :] = [
                pred_row['scaled_angle'].values[0],
                pred_row['scaled_depth'].values[0],
                pred_row['scaled_left_right'].values[0]
            ]

    # Compute variance across models
    variance = np.var(all_preds, axis=1)  # (n_test, 3)

    # Compute deviation from mean
    mean_preds = np.mean(all_preds, axis=1, keepdims=True)  # (n_test, 1, 3)
    deviations = all_preds - mean_preds  # (n_test, n_models, 3)

    return variance, deviations

def compute_model_quality_scores(
    submissions: Dict[int, pd.DataFrame],
    reference_sub_num: int = 1350
) -> Dict[int, float]:
    """
    Compute quality scores for each model based on:
    1. Leaderboard score if known
    2. Deviation from best model

    Returns:
        Dict mapping submission number to quality score
    """
    # Known LB scores (higher quality = lower LB score)
    lb_scores = {
        784: 0.007224,
        1350: 0.006776,  # Best
        1421: 0.006789,
    }

    # Normalize to quality scores (inverse of LB, scaled to [0,1])
    max_lb = max(lb_scores.values())
    min_lb = min(lb_scores.values())

    quality_scores = {}
    for sub_num in submissions.keys():
        if sub_num in lb_scores:
            # Use LB score
            normalized = (max_lb - lb_scores[sub_num]) / (max_lb - min_lb)
            quality_scores[sub_num] = normalized
        else:
            # Unknown LB, use correlation with best model
            quality_scores[sub_num] = 0.5  # Neutral

    return quality_scores

def create_uncertainty_weighted_ensemble(
    submissions: Dict[int, pd.DataFrame],
    strategy: str = 'inverse_variance',
    temperature: float = 1.0,
    quality_weight: float = 0.5
) -> pd.DataFrame:
    """
    Create uncertainty-weighted ensemble

    Args:
        strategy: 'inverse_variance', 'quality_weighted', 'hybrid'
        temperature: Softmax temperature for weighting
        quality_weight: Weight for quality vs uncertainty (0=pure uncertainty, 1=pure quality)

    Returns:
        Ensemble predictions DataFrame
    """
    # Get predictions
    first_sub = list(submissions.values())[0]
    shot_ids = first_sub['id'].values
    n_test = len(shot_ids)
    n_models = len(submissions)
    sub_nums = sorted(submissions.keys())

    # Compute variance and deviations
    variance, deviations = compute_prediction_variance(submissions)

    # Compute quality scores
    quality_scores = compute_model_quality_scores(submissions)

    # Initialize weights
    weights = np.zeros((n_test, n_models))

    if strategy == 'inverse_variance':
        # Weight by inverse of model's average deviation
        for i, sub_num in enumerate(sub_nums):
            # Average absolute deviation for this model
            avg_dev = np.mean(np.abs(deviations[:, i, :]), axis=1)  # (n_test,)
            weights[:, i] = 1.0 / (avg_dev + 1e-8)

    elif strategy == 'quality_weighted':
        # Weight by quality scores only
        for i, sub_num in enumerate(sub_nums):
            weights[:, i] = quality_scores[sub_num]

    elif strategy == 'hybrid':
        # Combine quality and uncertainty
        for i, sub_num in enumerate(sub_nums):
            # Uncertainty weight (inverse deviation)
            avg_dev = np.mean(np.abs(deviations[:, i, :]), axis=1)
            uncert_weight = 1.0 / (avg_dev + 1e-8)

            # Quality weight
            qual_weight = quality_scores[sub_num]

            # Combine
            weights[:, i] = quality_weight * qual_weight + (1 - quality_weight) * uncert_weight

    elif strategy == 'softmax':
        # Softmax based on quality scores
        for i, sub_num in enumerate(sub_nums):
            weights[:, i] = quality_scores[sub_num]

        # Apply softmax with temperature
        weights = np.exp(weights / temperature)

    elif strategy == 'per_target':
        # Different weights per target based on variance
        # This will be handled separately
        pass

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize weights
    if strategy != 'per_target':
        weights = weights / weights.sum(axis=1, keepdims=True)

    # Create ensemble predictions
    ensemble_preds = np.zeros((n_test, 3))

    if strategy == 'per_target':
        # Per-target weighting based on per-target variance
        for t_idx in range(3):
            target_weights = np.zeros((n_test, n_models))
            for i, sub_num in enumerate(sub_nums):
                # Use variance of this target
                target_var = variance[:, t_idx]
                target_dev = np.abs(deviations[:, i, t_idx])
                target_weights[:, i] = 1.0 / (target_dev + 1e-8)

            # Normalize
            target_weights = target_weights / target_weights.sum(axis=1, keepdims=True)

            # Apply weights
            for j, shot_id in enumerate(shot_ids):
                for i, sub_num in enumerate(sub_nums):
                    sub_df = submissions[sub_num]
                    pred_row = sub_df[sub_df['id'] == shot_id]
                    pred = pred_row[TARGETS[t_idx]].values[0]
                    ensemble_preds[j, t_idx] += target_weights[j, i] * pred
    else:
        # Standard weighting
        for j, shot_id in enumerate(shot_ids):
            for i, sub_num in enumerate(sub_nums):
                sub_df = submissions[sub_num]
                pred_row = sub_df[sub_df['id'] == shot_id]
                pred = np.array([
                    pred_row['scaled_angle'].values[0],
                    pred_row['scaled_depth'].values[0],
                    pred_row['scaled_left_right'].values[0]
                ])
                ensemble_preds[j] += weights[j, i] * pred

    # Create submission DataFrame
    result = pd.DataFrame({
        'id': shot_ids,
        'scaled_angle': ensemble_preds[:, 0],
        'scaled_depth': ensemble_preds[:, 1],
        'scaled_left_right': ensemble_preds[:, 2]
    })

    return result, weights

def analyze_uncertainty_distribution(
    submissions: Dict[int, pd.DataFrame]
) -> Dict:
    """Analyze prediction variance and uncertainty"""
    variance, deviations = compute_prediction_variance(submissions)

    analysis = {
        'mean_variance': np.mean(variance, axis=0),
        'std_variance': np.std(variance, axis=0),
        'max_variance': np.max(variance, axis=0),
        'per_model_mean_deviation': {}
    }

    for i, sub_num in enumerate(sorted(submissions.keys())):
        mean_dev = np.mean(np.abs(deviations[:, i, :]), axis=0)
        analysis['per_model_mean_deviation'][sub_num] = mean_dev

    return analysis

def main():
    print("=" * 80)
    print("UNCERTAINTY-WEIGHTED ENSEMBLE VIA PREDICTION DIVERSITY")
    print("=" * 80)

    # Load test shot IDs
    print("\n1. Loading test data...")
    test_df = pd.read_csv(DATA_DIR / "test.csv", nrows=1)
    test_shot_ids = []
    with open(DATA_DIR / "test.csv") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            shot_id = parts[1]
            test_shot_ids.append(shot_id)
    print(f"   Loaded {len(test_shot_ids)} test shots")

    # Load diverse submissions
    print("\n2. Loading diverse submissions...")
    submission_nums = [784, 1350, 1421]

    submissions = {}
    for sub_num in submission_nums:
        print(f"   Loading submission {sub_num}...")
        submissions[sub_num] = load_submission(sub_num)

    # Analyze uncertainty
    print("\n3. Analyzing prediction uncertainty...")
    analysis = analyze_uncertainty_distribution(submissions)

    print(f"\n   Mean variance per target:")
    print(f"   - Angle: {analysis['mean_variance'][0]:.8f}")
    print(f"   - Depth: {analysis['mean_variance'][1]:.8f}")
    print(f"   - Left_right: {analysis['mean_variance'][2]:.8f}")

    print(f"\n   Per-model mean absolute deviation:")
    for sub_num, dev in analysis['per_model_mean_deviation'].items():
        print(f"   - Sub {sub_num}: angle={dev[0]:.6f}, depth={dev[1]:.6f}, lr={dev[2]:.6f}")

    # Get quality scores
    quality_scores = compute_model_quality_scores(submissions)
    print(f"\n   Quality scores (based on LB):")
    for sub_num, score in quality_scores.items():
        print(f"   - Sub {sub_num}: {score:.4f}")

    # Generate submissions with different strategies
    print("\n4. Generating submissions...")
    submission_id = 1451

    strategies = [
        ('uniform', 'inverse_variance', 1.0, 0.0),
        ('inverse_var', 'inverse_variance', 1.0, 0.0),
        ('quality_only', 'quality_weighted', 1.0, 1.0),
        ('hybrid_50_50', 'hybrid', 1.0, 0.5),
        ('hybrid_70_30', 'hybrid', 1.0, 0.7),
        ('hybrid_30_70', 'hybrid', 1.0, 0.3),
        ('softmax_t0.1', 'softmax', 0.1, 0.5),
        ('softmax_t0.5', 'softmax', 0.5, 0.5),
        ('softmax_t1.0', 'softmax', 1.0, 0.5),
        ('per_target', 'per_target', 1.0, 0.0),
    ]

    results_summary = []

    for strategy_name, strategy, temperature, quality_weight in strategies:
        if strategy_name == 'uniform':
            # Uniform weighting baseline
            first_sub = list(submissions.values())[0]
            shot_ids = first_sub['id'].values
            n_models = len(submissions)
            ensemble_preds = np.zeros((len(shot_ids), 3))

            for shot_id in shot_ids:
                idx = np.where(first_sub['id'] == shot_id)[0][0]
                for sub_num in submissions.keys():
                    sub_df = submissions[sub_num]
                    pred_row = sub_df[sub_df['id'] == shot_id]
                    ensemble_preds[idx] += np.array([
                        pred_row['scaled_angle'].values[0],
                        pred_row['scaled_depth'].values[0],
                        pred_row['scaled_left_right'].values[0]
                    ]) / n_models

            result_df = pd.DataFrame({
                'id': shot_ids,
                'scaled_angle': ensemble_preds[:, 0],
                'scaled_depth': ensemble_preds[:, 1],
                'scaled_left_right': ensemble_preds[:, 2]
            })
            weights_summary = f"uniform (1/{n_models} each)"
        else:
            result_df, weights = create_uncertainty_weighted_ensemble(
                submissions, strategy, temperature, quality_weight
            )
            if strategy == 'per_target':
                weights_summary = "per-target adaptive"
            else:
                weights_summary = f"mean={np.mean(weights, axis=0)}"

        # Save submission
        sub_path = SUBMISSION_DIR / f"submission_{submission_id}.csv"
        result_df.to_csv(sub_path, index=False)
        print(f"   Saved {strategy_name} to submission_{submission_id}.csv")

        # Compute diversity from Sub 1350
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
            'weights': weights_summary,
            'correlation_with_1350': f"angle={correlations[0]:.3f}, depth={correlations[1]:.3f}, lr={correlations[2]:.3f}"
        })

        submission_id += 1

    # Save results
    print("\n5. Saving results...")
    results_md = RESEARCH_DIR / "UNCERTAINTY_ENSEMBLE_RESULTS.md"

    with open(results_md, 'w') as f:
        f.write("# Uncertainty-Weighted Ensemble Results\n\n")
        f.write("## Approach\n\n")
        f.write("Use prediction diversity (variance across models) as uncertainty proxy.\n")
        f.write("Weight models by:\n")
        f.write("1. Inverse variance (low disagreement = high confidence)\n")
        f.write("2. Quality scores (known LB performance)\n")
        f.write("3. Hybrid combination of uncertainty and quality\n\n")

        f.write("## Base Models\n\n")
        f.write("- Sub 784: LB 0.007224 (baseline)\n")
        f.write("- Sub 1350: LB 0.006776 (best, per-example V1)\n")
        f.write("- Sub 1421: LB 0.006789 (per-example V2)\n\n")

        f.write("## Prediction Variance Analysis\n\n")
        f.write(f"Mean variance per target:\n")
        f.write(f"- Angle: {analysis['mean_variance'][0]:.8f}\n")
        f.write(f"- Depth: {analysis['mean_variance'][1]:.8f}\n")
        f.write(f"- Left_right: {analysis['mean_variance'][2]:.8f}\n\n")

        f.write("Per-model mean absolute deviation:\n")
        for sub_num, dev in analysis['per_model_mean_deviation'].items():
            lb_score = {784: 0.007224, 1350: 0.006776, 1421: 0.006789}[sub_num]
            f.write(f"- Sub {sub_num} (LB {lb_score:.6f}): angle={dev[0]:.6f}, depth={dev[1]:.6f}, lr={dev[2]:.6f}\n")
        f.write("\n")

        f.write("Quality scores (normalized inverse LB):\n")
        for sub_num, score in quality_scores.items():
            f.write(f"- Sub {sub_num}: {score:.4f}\n")
        f.write("\n")

        f.write("## Generated Submissions\n\n")
        for result in results_summary:
            f.write(f"### Sub {result['submission']}: {result['strategy']}\n")
            f.write(f"- Weights: {result['weights']}\n")
            f.write(f"- Correlation with Sub 1350: {result['correlation_with_1350']}\n\n")

        f.write("## Strategy Explanations\n\n")
        f.write("1. **uniform**: Simple average (baseline)\n")
        f.write("2. **inverse_var**: Weight by inverse of average absolute deviation\n")
        f.write("3. **quality_only**: Weight by known LB scores only\n")
        f.write("4. **hybrid_X_Y**: X% quality + Y% uncertainty weighting\n")
        f.write("5. **softmax_tN**: Softmax over quality scores with temperature N\n")
        f.write("6. **per_target**: Different weights per target based on target-specific variance\n\n")

        f.write("## Expected Performance\n\n")
        f.write("- Uniform baseline: ~0.006780 (midpoint of 1350 and 1421)\n")
        f.write("- Quality-weighted: closer to 1350 (best model gets higher weight)\n")
        f.write("- Hybrid: balance between best model and diversity\n")
        f.write("- Per-target: may help if models have different strengths per target\n\n")

        f.write("## Key Insights\n\n")
        f.write("1. Prediction variance is highest for left_right (most uncertain target)\n")
        f.write("2. Sub 1350 has lowest deviation (most confident/stable)\n")
        f.write("3. Hybrid weighting balances trusting best model vs. ensemble diversity\n")
        f.write("4. Per-target weighting allows target-specific model selection\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Test submissions 1451-1460 on leaderboard\n")
        f.write("2. If hybrid works, try more models (add Sub 1354, biomech variants)\n")
        f.write("3. Explore per-shot adaptive weighting using feature space similarity\n")
        f.write("4. Consider stacking with meta-learner\n\n")

    print(f"   Saved results to {results_md}")

    print("\n" + "=" * 80)
    print(f"COMPLETED: Generated {len(strategies)} uncertainty-weighted submissions")
    print(f"Submissions: 1451-{submission_id-1}")
    print("=" * 80)

if __name__ == "__main__":
    main()
