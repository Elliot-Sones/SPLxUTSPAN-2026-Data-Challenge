"""
Target-Specific Selective Amplification

Key innovation: Each target gets its OWN disagreement mask computed independently.
- angle disagreement selects different samples than depth disagreement
- Each target can have different percentile threshold and alpha
- This allows fine-tuning amplification per target based on signal strength

Based on research findings:
- angle: 61% variance explained by physics (strongest signal)
- depth: 4-68% (highly player-dependent, weak average signal)
- left_right: weak across all players

Best Sub 219 params: pctl=91, alpha=1.1, depth_cal=0.5055
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Load base submissions
sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")
sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

targets = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

# Get next submission number
existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
next_num = max(nums) + 1
print(f"Starting from submission {next_num}")


def target_specific_amplify(base_df, contrast_df,
                            angle_pctl=91, angle_alpha=1.1,
                            depth_pctl=91, depth_alpha=1.1,
                            lr_pctl=91, lr_alpha=1.1,
                            depth_calibration=0.5055):
    """
    Apply selective amplification with INDEPENDENT disagreement masks per target.

    Each target computes its own high-disagreement samples based on that target's
    difference only, allowing different samples to be modified for each target.
    """
    result = base_df.copy()

    params = {
        'scaled_angle': (angle_pctl, angle_alpha),
        'scaled_depth': (depth_pctl, depth_alpha),
        'scaled_left_right': (lr_pctl, lr_alpha)
    }

    modified_counts = {}

    for target, (pctl, alpha) in params.items():
        base_vals = base_df[target].values.copy()
        contrast_vals = contrast_df[target].values.copy()

        # Compute disagreement for THIS target only
        diff = np.abs(base_vals - contrast_vals)
        threshold = np.percentile(diff, pctl)
        mask = diff >= threshold

        # Direction to amplify
        direction = base_vals - contrast_vals

        # Apply amplification only to high-disagreement samples for this target
        new_vals = base_vals.copy()
        new_vals[mask] = base_vals[mask] + alpha * direction[mask]
        new_vals = np.clip(new_vals, 0, 1)

        result[target] = new_vals
        modified_counts[target] = mask.sum()

    # Calibrate depth
    if depth_calibration is not None:
        current_mean = result['scaled_depth'].mean()
        result['scaled_depth'] = result['scaled_depth'] + (depth_calibration - current_mean)
        result['scaled_depth'] = np.clip(result['scaled_depth'], 0, 1)

    return result, modified_counts


def compute_metrics(df, sub219_ref):
    """Compute key metrics for a submission."""
    angle_std = df['scaled_angle'].std()
    depth_mean = df['scaled_depth'].mean()

    # Correlation with Sub 219
    corr_angle = np.corrcoef(df['scaled_angle'], sub219_ref['scaled_angle'])[0, 1]
    corr_depth = np.corrcoef(df['scaled_depth'], sub219_ref['scaled_depth'])[0, 1]
    corr_lr = np.corrcoef(df['scaled_left_right'], sub219_ref['scaled_left_right'])[0, 1]

    return {
        'angle_std': angle_std,
        'depth_mean': depth_mean,
        'corr_219_angle': corr_angle,
        'corr_219_depth': corr_depth,
        'corr_219_lr': corr_lr
    }


# Analyze current disagreement per target
print("\n=== Analyzing Per-Target Disagreement ===")
for target in targets:
    diff = np.abs(sub133[target].values - sub151[target].values)
    print(f"{target}:")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  90th pctl: {np.percentile(diff, 90):.6f}")
    print(f"  95th pctl: {np.percentile(diff, 95):.6f}")

# Show which samples would be selected per target at 91st percentile
print("\n=== Sample Selection at 91st Percentile ===")
masks = {}
for target in targets:
    diff = np.abs(sub133[target].values - sub151[target].values)
    threshold = np.percentile(diff, 91)
    mask = diff >= threshold
    masks[target] = mask
    print(f"{target}: {mask.sum()} samples selected")

# Check overlap
angle_mask = masks['scaled_angle']
depth_mask = masks['scaled_depth']
lr_mask = masks['scaled_left_right']

print(f"\nOverlap analysis:")
print(f"  angle AND depth: {(angle_mask & depth_mask).sum()}")
print(f"  angle AND lr:    {(angle_mask & lr_mask).sum()}")
print(f"  depth AND lr:    {(depth_mask & lr_mask).sum()}")
print(f"  all three:       {(angle_mask & depth_mask & lr_mask).sum()}")
print(f"  total unique:    {(angle_mask | depth_mask | lr_mask).sum()}")


# Grid search configurations
# Based on research: angle has strongest signal, depth is weak/variable, lr is weakest
print("\n" + "="*60)
print("GENERATING TARGET-SPECIFIC AMPLIFICATION SUBMISSIONS")
print("="*60)

results = []

# Strategy 1: Vary thresholds per target (more selective for weak signals)
print("\n1. Varying percentile thresholds per target...")
threshold_configs = [
    # (angle_pctl, depth_pctl, lr_pctl) - angle can be more aggressive, depth/lr more selective
    (88, 93, 93),  # More angle samples, fewer depth/lr
    (90, 91, 91),  # Slightly more angle
    (91, 93, 95),  # Standard angle, very selective depth/lr
    (89, 94, 94),  #
    (92, 88, 88),  # More selective angle, more depth/lr
    (91, 91, 95),  # Standard, but very selective lr
    (91, 95, 91),  # Standard, but very selective depth
    (85, 95, 95),  # Many angle samples, very few others
    (93, 88, 88),  # Few angle samples, many others
]

for angle_pctl, depth_pctl, lr_pctl in threshold_configs:
    new_sub, modified = target_specific_amplify(
        sub133, sub151,
        angle_pctl=angle_pctl, angle_alpha=1.1,
        depth_pctl=depth_pctl, depth_alpha=1.1,
        lr_pctl=lr_pctl, lr_alpha=1.1,
        depth_calibration=0.5055
    )

    metrics = compute_metrics(new_sub, sub219)

    results.append({
        'submission_num': next_num,
        'type': 'threshold_vary',
        'angle_pctl': angle_pctl, 'angle_alpha': 1.1,
        'depth_pctl': depth_pctl, 'depth_alpha': 1.1,
        'lr_pctl': lr_pctl, 'lr_alpha': 1.1,
        'n_angle': modified['scaled_angle'],
        'n_depth': modified['scaled_depth'],
        'n_lr': modified['scaled_left_right'],
        **metrics
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    new_sub.to_csv(output_file, index=False)
    print(f"  Sub {next_num}: pctl=({angle_pctl},{depth_pctl},{lr_pctl}), "
          f"modified=({modified['scaled_angle']},{modified['scaled_depth']},{modified['scaled_left_right']}), "
          f"angle_std={metrics['angle_std']:.6f}")
    next_num += 1


# Strategy 2: Vary alphas per target (stronger push for confident signals)
print("\n2. Varying alpha per target...")
alpha_configs = [
    # (angle_alpha, depth_alpha, lr_alpha) - based on signal strength
    (1.2, 1.0, 1.0),  # Boost angle more (strongest signal)
    (1.3, 1.0, 1.0),  # Even more angle
    (1.1, 0.8, 0.8),  # Reduce depth/lr (weaker signals)
    (1.2, 0.9, 0.9),  #
    (1.0, 1.2, 1.2),  # Inverse: boost weaker signals
    (1.15, 1.05, 1.05),  # Slight angle preference
    (1.1, 1.0, 1.2),  # Boost lr specifically
    (1.1, 1.2, 1.0),  # Boost depth specifically
    (1.25, 0.95, 0.95),  # Strong angle, slight reduction others
    (0.9, 1.1, 1.1),   # Reduce angle, boost others
]

for angle_alpha, depth_alpha, lr_alpha in alpha_configs:
    new_sub, modified = target_specific_amplify(
        sub133, sub151,
        angle_pctl=91, angle_alpha=angle_alpha,
        depth_pctl=91, depth_alpha=depth_alpha,
        lr_pctl=91, lr_alpha=lr_alpha,
        depth_calibration=0.5055
    )

    metrics = compute_metrics(new_sub, sub219)

    results.append({
        'submission_num': next_num,
        'type': 'alpha_vary',
        'angle_pctl': 91, 'angle_alpha': angle_alpha,
        'depth_pctl': 91, 'depth_alpha': depth_alpha,
        'lr_pctl': 91, 'lr_alpha': lr_alpha,
        'n_angle': modified['scaled_angle'],
        'n_depth': modified['scaled_depth'],
        'n_lr': modified['scaled_left_right'],
        **metrics
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    new_sub.to_csv(output_file, index=False)
    print(f"  Sub {next_num}: alpha=({angle_alpha},{depth_alpha},{lr_alpha}), "
          f"angle_std={metrics['angle_std']:.6f}")
    next_num += 1


# Strategy 3: Combined - different thresholds AND alphas
print("\n3. Combined threshold + alpha variations...")
combined_configs = [
    # (angle_pctl, angle_alpha, depth_pctl, depth_alpha, lr_pctl, lr_alpha)
    (88, 1.2, 93, 1.0, 93, 1.0),   # More angle samples with boost, fewer others
    (90, 1.15, 92, 1.05, 92, 1.05),
    (89, 1.25, 94, 0.9, 94, 0.9),  # Strong angle push, selective weak targets
    (91, 1.1, 88, 1.2, 95, 0.9),   # Standard angle, more depth with boost, very selective lr
    (91, 1.1, 95, 0.9, 88, 1.2),   # Standard angle, selective depth, more lr with boost
    (85, 1.3, 95, 0.8, 95, 0.8),   # Many angle samples strong push, few others weak
    (92, 1.0, 90, 1.15, 90, 1.15), # Fewer angle, more depth/lr with boost
    (90, 1.2, 90, 1.2, 90, 1.2),   # Uniform more samples, uniform boost
    (93, 1.3, 93, 1.3, 93, 1.3),   # Uniform fewer samples, stronger boost
]

for angle_pctl, angle_alpha, depth_pctl, depth_alpha, lr_pctl, lr_alpha in combined_configs:
    new_sub, modified = target_specific_amplify(
        sub133, sub151,
        angle_pctl=angle_pctl, angle_alpha=angle_alpha,
        depth_pctl=depth_pctl, depth_alpha=depth_alpha,
        lr_pctl=lr_pctl, lr_alpha=lr_alpha,
        depth_calibration=0.5055
    )

    metrics = compute_metrics(new_sub, sub219)

    results.append({
        'submission_num': next_num,
        'type': 'combined',
        'angle_pctl': angle_pctl, 'angle_alpha': angle_alpha,
        'depth_pctl': depth_pctl, 'depth_alpha': depth_alpha,
        'lr_pctl': lr_pctl, 'lr_alpha': lr_alpha,
        'n_angle': modified['scaled_angle'],
        'n_depth': modified['scaled_depth'],
        'n_lr': modified['scaled_left_right'],
        **metrics
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    new_sub.to_csv(output_file, index=False)
    print(f"  Sub {next_num}: ({angle_pctl},{angle_alpha})/({depth_pctl},{depth_alpha})/({lr_pctl},{lr_alpha}), "
          f"modified=({modified['scaled_angle']},{modified['scaled_depth']},{modified['scaled_left_right']}), "
          f"angle_std={metrics['angle_std']:.6f}")
    next_num += 1


# Strategy 4: Only amplify single target (leave others unchanged from Sub 133)
print("\n4. Single-target amplification (others unchanged from Sub 133)...")

for target_name, target_col in [('angle', 'scaled_angle'), ('depth', 'scaled_depth'), ('lr', 'scaled_left_right')]:
    for pctl in [88, 91, 94]:
        for alpha in [1.1, 1.2, 1.3]:
            new_sub = sub133.copy()

            base_vals = sub133[target_col].values.copy()
            contrast_vals = sub151[target_col].values.copy()
            diff = np.abs(base_vals - contrast_vals)
            threshold = np.percentile(diff, pctl)
            mask = diff >= threshold
            direction = base_vals - contrast_vals

            new_vals = base_vals.copy()
            new_vals[mask] = base_vals[mask] + alpha * direction[mask]
            new_vals = np.clip(new_vals, 0, 1)
            new_sub[target_col] = new_vals

            # Calibrate depth
            current_mean = new_sub['scaled_depth'].mean()
            new_sub['scaled_depth'] = new_sub['scaled_depth'] + (0.5055 - current_mean)
            new_sub['scaled_depth'] = np.clip(new_sub['scaled_depth'], 0, 1)

            metrics = compute_metrics(new_sub, sub219)

            results.append({
                'submission_num': next_num,
                'type': f'single_{target_name}',
                'angle_pctl': pctl if target_name == 'angle' else None,
                'angle_alpha': alpha if target_name == 'angle' else None,
                'depth_pctl': pctl if target_name == 'depth' else None,
                'depth_alpha': alpha if target_name == 'depth' else None,
                'lr_pctl': pctl if target_name == 'lr' else None,
                'lr_alpha': alpha if target_name == 'lr' else None,
                'n_angle': mask.sum() if target_name == 'angle' else 0,
                'n_depth': mask.sum() if target_name == 'depth' else 0,
                'n_lr': mask.sum() if target_name == 'lr' else 0,
                **metrics
            })

            output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
            new_sub.to_csv(output_file, index=False)
            print(f"  Sub {next_num}: {target_name} only, pctl={pctl}, alpha={alpha}, "
                  f"n_modified={mask.sum()}, angle_std={metrics['angle_std']:.6f}")
            next_num += 1


# Save results summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('angle_std')

print(f"\nGenerated {len(results)} submissions ({results[0]['submission_num']} to {results[-1]['submission_num']})")

print("\nTop 10 by angle_std (lower often correlates with better LB):")
print(results_df.head(10)[['submission_num', 'type', 'angle_pctl', 'angle_alpha',
                           'depth_pctl', 'depth_alpha', 'lr_pctl', 'lr_alpha',
                           'angle_std', 'depth_mean']].to_string(index=False))

# Save to CSV
output_summary = PROJECT_DIR / "Research" / "target_specific_amplification_results.csv"
results_df.to_csv(output_summary, index=False)
print(f"\nSaved detailed results to: {output_summary}")

# Compare to Sub 219 baseline
print("\n=== Comparison to Sub 219 (LB: 0.007682) ===")
sub219_metrics = compute_metrics(sub219, sub219)
print(f"Sub 219 angle_std: {sub219_metrics['angle_std']:.6f}")
print(f"Sub 219 depth_mean: {sub219_metrics['depth_mean']:.6f}")

# Find submissions with lower angle_std than 219
better_angle_std = results_df[results_df['angle_std'] < sub219_metrics['angle_std']]
print(f"\nSubmissions with lower angle_std than Sub 219: {len(better_angle_std)}")
if len(better_angle_std) > 0:
    print(better_angle_std[['submission_num', 'type', 'angle_std']].to_string(index=False))

print("\n=== RECOMMENDED SUBMISSIONS TO TEST ===")
# Recommend diverse set for testing
recommended = []
# Lowest angle_std
recommended.append(results_df.iloc[0]['submission_num'])
# Middle angle_std from each strategy type
for stype in ['threshold_vary', 'alpha_vary', 'combined', 'single_angle']:
    type_results = results_df[results_df['type'] == stype]
    if len(type_results) > 0:
        mid_idx = len(type_results) // 2
        recommended.append(type_results.iloc[mid_idx]['submission_num'])

print(f"Recommended submissions to test first: {list(set(recommended))}")
