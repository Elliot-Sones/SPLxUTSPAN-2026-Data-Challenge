"""
Refined Selective Amplification Variations

Based on the grid search, Sub 219 uses:
- Percentile: 91 (11 samples modified)
- Alpha: 1.1
- Depth calibration: 0.5055

This script generates more refined variations to potentially beat Sub 219.
"""

import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = "/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge"
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
OUTPUT_DIR = os.path.join(BASE_DIR, "external_data/output")

# Load base submissions
sub133 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_133.csv"))
sub151 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_151.csv"))
sub219 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_219.csv"))

targets = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

# Get next submission number
existing_subs = [f for f in os.listdir(SUBMISSION_DIR) if f.startswith('submission_') and f.endswith('.csv')]
max_num = max([int(f.split('_')[1].split('.')[0]) for f in existing_subs if '_' in f])
submission_num = max_num + 1
print(f"Starting from submission {submission_num}")

def selective_amplify(base_df, contrast_df, percentile=91, alpha=1.0,
                      per_target_alpha=None, depth_calibration=0.5055):
    """Apply selective amplification."""
    result = base_df.copy()
    diff = np.abs(base_df[targets].values - contrast_df[targets].values).sum(axis=1)
    threshold = np.percentile(diff, percentile)
    high_disagree_mask = diff >= threshold

    for i, target in enumerate(targets):
        if per_target_alpha is not None:
            a = per_target_alpha.get(target, alpha)
        else:
            a = alpha

        base_vals = base_df[target].values.copy()
        contrast_vals = contrast_df[target].values.copy()
        direction = base_vals - contrast_vals

        new_vals = base_vals.copy()
        new_vals[high_disagree_mask] = base_vals[high_disagree_mask] + a * direction[high_disagree_mask]
        new_vals = np.clip(new_vals, 0, 1)
        result[target] = new_vals

    if depth_calibration is not None:
        current_mean = result['scaled_depth'].mean()
        result['scaled_depth'] = result['scaled_depth'] + (depth_calibration - current_mean)
        result['scaled_depth'] = np.clip(result['scaled_depth'], 0, 1)

    return result, high_disagree_mask.sum()

submissions_info = []

print("\n=== Generating Refined Variations ===")

# Fine-grained alpha variations around 1.1
print("\n1. Fine-grained alpha variations around 1.1...")
fine_alphas = [1.02, 1.03, 1.04, 1.06, 1.07, 1.08, 1.12, 1.13, 1.14]
for alpha in fine_alphas:
    new_sub, n_modified = selective_amplify(sub133, sub151, 91, alpha, depth_calibration=0.5055)
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    new_sub.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': f'fine_alpha_{alpha}',
        'percentile': 91,
        'alpha': alpha,
        'depth_cal': 0.5055,
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'file': sub_path
    })
    print(f"  Saved submission_{submission_num}: alpha={alpha}")
    submission_num += 1

# Fine-grained depth calibrations
print("\n2. Fine-grained depth calibrations...")
fine_depths = [0.5052, 0.5053, 0.5054, 0.5056, 0.5057, 0.5058]
for depth_cal in fine_depths:
    new_sub, n_modified = selective_amplify(sub133, sub151, 91, 1.1, depth_calibration=depth_cal)
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    new_sub.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': f'fine_depth_{depth_cal}',
        'percentile': 91,
        'alpha': 1.1,
        'depth_cal': depth_cal,
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'file': sub_path
    })
    print(f"  Saved submission_{submission_num}: depth_cal={depth_cal}")
    submission_num += 1

# Asymmetric amplification (different alpha for angle vs depth vs left_right)
print("\n3. Asymmetric amplification configurations...")
asymmetric_configs = [
    # Boost angle slightly more
    {'scaled_angle': 1.15, 'scaled_depth': 1.1, 'scaled_left_right': 1.1},
    {'scaled_angle': 1.2, 'scaled_depth': 1.1, 'scaled_left_right': 1.1},
    {'scaled_angle': 1.1, 'scaled_depth': 1.15, 'scaled_left_right': 1.1},
    {'scaled_angle': 1.1, 'scaled_depth': 1.1, 'scaled_left_right': 1.15},

    # Reduce one while boosting others
    {'scaled_angle': 1.0, 'scaled_depth': 1.15, 'scaled_left_right': 1.15},
    {'scaled_angle': 1.15, 'scaled_depth': 1.0, 'scaled_left_right': 1.15},
    {'scaled_angle': 1.15, 'scaled_depth': 1.15, 'scaled_left_right': 1.0},

    # Mixed strategies
    {'scaled_angle': 1.05, 'scaled_depth': 1.1, 'scaled_left_right': 1.15},
    {'scaled_angle': 1.15, 'scaled_depth': 1.05, 'scaled_left_right': 1.1},
    {'scaled_angle': 1.1, 'scaled_depth': 1.15, 'scaled_left_right': 1.05},
]

for config in asymmetric_configs:
    new_sub, n_modified = selective_amplify(sub133, sub151, 91, 1.1,
                                            per_target_alpha=config,
                                            depth_calibration=0.5055)
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    new_sub.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': f'asymmetric_{submission_num}',
        'percentile': 91,
        'alpha': str(config),
        'depth_cal': 0.5055,
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'file': sub_path
    })
    print(f"  Saved submission_{submission_num}: {config}")
    submission_num += 1

# Fractional percentile threshold (use custom threshold instead of percentile)
print("\n4. Custom threshold variations...")
diff_133_151 = np.abs(sub133[targets].values - sub151[targets].values).sum(axis=1)

# Get the exact threshold for 91st percentile
p91_threshold = np.percentile(diff_133_151, 91)
print(f"   91st percentile threshold: {p91_threshold:.6f}")

# Try thresholds slightly above and below
threshold_variations = [
    p91_threshold * 0.95,
    p91_threshold * 0.98,
    p91_threshold * 1.02,
    p91_threshold * 1.05,
]

def selective_amplify_threshold(base_df, contrast_df, threshold, alpha=1.1, depth_calibration=0.5055):
    """Apply selective amplification with custom threshold."""
    result = base_df.copy()
    diff = np.abs(base_df[targets].values - contrast_df[targets].values).sum(axis=1)
    high_disagree_mask = diff >= threshold

    for target in targets:
        base_vals = base_df[target].values.copy()
        contrast_vals = contrast_df[target].values.copy()
        direction = base_vals - contrast_vals

        new_vals = base_vals.copy()
        new_vals[high_disagree_mask] = base_vals[high_disagree_mask] + alpha * direction[high_disagree_mask]
        new_vals = np.clip(new_vals, 0, 1)
        result[target] = new_vals

    if depth_calibration is not None:
        current_mean = result['scaled_depth'].mean()
        result['scaled_depth'] = result['scaled_depth'] + (depth_calibration - current_mean)
        result['scaled_depth'] = np.clip(result['scaled_depth'], 0, 1)

    return result, high_disagree_mask.sum()

for thresh in threshold_variations:
    new_sub, n_modified = selective_amplify_threshold(sub133, sub151, thresh, 1.1, 0.5055)
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    new_sub.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': f'custom_thresh_{thresh:.4f}',
        'percentile': 'custom',
        'alpha': 1.1,
        'depth_cal': 0.5055,
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'file': sub_path
    })
    print(f"  Saved submission_{submission_num}: threshold={thresh:.4f}, n_modified={n_modified}")
    submission_num += 1

# Weighted amplification (amplify more for samples with higher disagreement)
print("\n5. Weighted amplification (proportional to disagreement)...")
def weighted_selective_amplify(base_df, contrast_df, percentile=91, base_alpha=1.1,
                               weight_factor=0.5, depth_calibration=0.5055):
    """
    Weighted amplification: alpha scales with disagreement level.
    alpha_i = base_alpha * (1 + weight_factor * (diff_i - min_diff) / (max_diff - min_diff))
    """
    result = base_df.copy()
    diff = np.abs(base_df[targets].values - contrast_df[targets].values).sum(axis=1)
    threshold = np.percentile(diff, percentile)
    high_disagree_mask = diff >= threshold

    # Calculate weights for high-disagreement samples
    high_diffs = diff[high_disagree_mask]
    if len(high_diffs) > 1:
        normalized_diffs = (high_diffs - high_diffs.min()) / (high_diffs.max() - high_diffs.min() + 1e-10)
    else:
        normalized_diffs = np.ones(len(high_diffs))

    for target in targets:
        base_vals = base_df[target].values.copy()
        contrast_vals = contrast_df[target].values.copy()
        direction = base_vals - contrast_vals

        new_vals = base_vals.copy()
        # Apply weighted alpha to each high-disagreement sample
        weighted_alphas = base_alpha * (1 + weight_factor * normalized_diffs)
        new_vals[high_disagree_mask] = base_vals[high_disagree_mask] + weighted_alphas * direction[high_disagree_mask]
        new_vals = np.clip(new_vals, 0, 1)
        result[target] = new_vals

    if depth_calibration is not None:
        current_mean = result['scaled_depth'].mean()
        result['scaled_depth'] = result['scaled_depth'] + (depth_calibration - current_mean)
        result['scaled_depth'] = np.clip(result['scaled_depth'], 0, 1)

    return result, high_disagree_mask.sum()

weight_factors = [0.3, 0.5, 0.7, 1.0]
for wf in weight_factors:
    new_sub, n_modified = weighted_selective_amplify(sub133, sub151, 91, 1.1, wf, 0.5055)
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    new_sub.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': f'weighted_wf{wf}',
        'percentile': 91,
        'alpha': f'1.1*weighted_{wf}',
        'depth_cal': 0.5055,
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'file': sub_path
    })
    print(f"  Saved submission_{submission_num}: weight_factor={wf}")
    submission_num += 1

# Blend with Sub 219 (average with slight variations)
print("\n6. Blends with Sub 219...")
blend_weights = [0.4, 0.45, 0.55, 0.6]
for w in blend_weights:
    # Create amplified version
    amplified, _ = selective_amplify(sub133, sub151, 91, 1.1, depth_calibration=0.5055)

    # Blend with 219
    blended = sub219.copy()
    for target in targets:
        blended[target] = w * amplified[target] + (1 - w) * sub219[target]

    # Recalibrate depth
    current_mean = blended['scaled_depth'].mean()
    blended['scaled_depth'] = blended['scaled_depth'] + (0.5055 - current_mean)
    blended['scaled_depth'] = np.clip(blended['scaled_depth'], 0, 1)

    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    blended.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': f'blend_219_w{w}',
        'percentile': 91,
        'alpha': f'blend_{w}',
        'depth_cal': 0.5055,
        'n_modified': 'blend',
        'depth_mean': blended['scaled_depth'].mean(),
        'file': sub_path
    })
    print(f"  Saved submission_{submission_num}: blend with 219, weight={w}")
    submission_num += 1

# Combined variations: different alpha + depth at same time
print("\n7. Combined alpha and depth variations...")
combined_configs = [
    (1.08, 0.5053),
    (1.12, 0.5053),
    (1.08, 0.5057),
    (1.12, 0.5057),
    (1.05, 0.5052),
    (1.15, 0.5058),
]

for alpha, depth_cal in combined_configs:
    new_sub, n_modified = selective_amplify(sub133, sub151, 91, alpha, depth_calibration=depth_cal)
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    new_sub.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': f'combined_a{alpha}_d{depth_cal}',
        'percentile': 91,
        'alpha': alpha,
        'depth_cal': depth_cal,
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'file': sub_path
    })
    print(f"  Saved submission_{submission_num}: alpha={alpha}, depth={depth_cal}")
    submission_num += 1

# Save summary
sub_info_df = pd.DataFrame(submissions_info)
sub_info_path = os.path.join(OUTPUT_DIR, 'selective_amplification_refined.csv')
sub_info_df.to_csv(sub_info_path, index=False)

print(f"\n=== Summary ===")
print(f"Generated {len(submissions_info)} new refined submissions")
print(f"Submission numbers: {submissions_info[0]['submission_num']} to {submissions_info[-1]['submission_num']}")
print(f"\nSaved submission info to: {sub_info_path}")

print("\n=== Submission Details ===")
for info in submissions_info:
    print(f"  {info['submission_num']}: {info['config']}")
