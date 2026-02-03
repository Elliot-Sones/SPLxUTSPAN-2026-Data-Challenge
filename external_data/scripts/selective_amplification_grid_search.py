"""
Selective Amplification Grid Search

Based on analysis that Sub 219 uses selective amplification:
1. Find samples where Sub 133 and Sub 151 disagree most (above 91st percentile)
2. Amplify in Sub 133's direction: new = Sub133 + alpha * (Sub133 - Sub151)
3. Calibrate depth_mean to 0.5055

This script tests variations:
- Different percentile thresholds: 85, 88, 90, 91, 92, 93, 95
- Different alpha values: 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5
- Per-target alpha
- Different contrast submissions
- Two-stage amplification
"""

import pandas as pd
import numpy as np
from itertools import product
import os

# Paths
BASE_DIR = "/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge"
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
OUTPUT_DIR = os.path.join(BASE_DIR, "external_data/output")

# Load base submissions
sub133 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_133.csv"))
sub151 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_151.csv"))
sub219 = pd.read_csv(os.path.join(SUBMISSION_DIR, "submission_219.csv"))

# Verify they have the same IDs in same order
assert list(sub133['id']) == list(sub151['id']) == list(sub219['id']), "IDs don't match"

targets = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

# Load other potential contrast submissions
contrast_subs = {}
for sub_num in [9, 25, 84, 100, 150, 152, 160]:
    path = os.path.join(SUBMISSION_DIR, f"submission_{sub_num}.csv")
    if os.path.exists(path):
        contrast_subs[sub_num] = pd.read_csv(path)
        print(f"Loaded submission_{sub_num}")

print(f"\nSub 133 depth mean: {sub133['scaled_depth'].mean():.6f}")
print(f"Sub 151 depth mean: {sub151['scaled_depth'].mean():.6f}")
print(f"Sub 219 depth mean: {sub219['scaled_depth'].mean():.6f}")
print(f"Target depth mean: 0.5055")

# Analyze Sub 219 to verify our understanding
print("\n--- Analyzing Sub 219 ---")
diff_133_151 = np.abs(sub133[targets].values - sub151[targets].values).sum(axis=1)
print(f"Total disagreement range: {diff_133_151.min():.4f} to {diff_133_151.max():.4f}")
print(f"Mean disagreement: {diff_133_151.mean():.4f}")

# Check what percentile threshold was used
for pct in [85, 88, 90, 91, 92, 93, 95]:
    threshold = np.percentile(diff_133_151, pct)
    n_samples = (diff_133_151 >= threshold).sum()
    print(f"  {pct}th percentile threshold: {threshold:.4f}, {n_samples} samples above")

def selective_amplify(base_df, contrast_df, percentile=91, alpha=1.0,
                      per_target_alpha=None, depth_calibration=0.5055):
    """
    Apply selective amplification.

    For high-disagreement samples:
        new = base + alpha * (base - contrast)

    Then calibrate depth mean.
    """
    result = base_df.copy()

    # Calculate per-sample disagreement (sum across all targets)
    diff = np.abs(base_df[targets].values - contrast_df[targets].values).sum(axis=1)
    threshold = np.percentile(diff, percentile)
    high_disagree_mask = diff >= threshold

    for i, target in enumerate(targets):
        if per_target_alpha is not None:
            a = per_target_alpha.get(target, alpha)
        else:
            a = alpha

        # Apply amplification only to high-disagreement samples
        base_vals = base_df[target].values.copy()
        contrast_vals = contrast_df[target].values.copy()
        direction = base_vals - contrast_vals

        # Amplify
        new_vals = base_vals.copy()
        new_vals[high_disagree_mask] = base_vals[high_disagree_mask] + a * direction[high_disagree_mask]

        # Clip to [0, 1]
        new_vals = np.clip(new_vals, 0, 1)
        result[target] = new_vals

    # Calibrate depth
    if depth_calibration is not None:
        current_mean = result['scaled_depth'].mean()
        result['scaled_depth'] = result['scaled_depth'] + (depth_calibration - current_mean)
        result['scaled_depth'] = np.clip(result['scaled_depth'], 0, 1)

    return result, high_disagree_mask.sum()

def two_stage_amplify(base_df, contrast1_df, contrast2_df,
                      percentile1=91, alpha1=1.0,
                      percentile2=85, alpha2=0.5,
                      depth_calibration=0.5055):
    """
    Two-stage amplification:
    Stage 1: Amplify using contrast1
    Stage 2: Further refine using contrast2
    """
    # Stage 1
    result, n1 = selective_amplify(base_df, contrast1_df, percentile1, alpha1,
                                   depth_calibration=None)

    # Stage 2
    result, n2 = selective_amplify(result, contrast2_df, percentile2, alpha2,
                                   depth_calibration=depth_calibration)

    return result, n1, n2

def score_similarity_to_219(new_sub):
    """Calculate similarity to Sub 219 (lower = more similar)"""
    diff = np.abs(new_sub[targets].values - sub219[targets].values).mean()
    return diff

# Grid search parameters
percentiles = [85, 88, 90, 91, 92, 93, 95]
alphas = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
depth_targets = [0.5055, 0.5050, 0.5060, None]  # None = no calibration

# Results storage
results = []

print("\n=== Running Grid Search ===")
print("Testing uniform alpha across all targets...")

# Test 1: Uniform alpha with Sub 151 as contrast
for pct, alpha, depth_cal in product(percentiles, alphas, depth_targets):
    new_sub, n_modified = selective_amplify(sub133, sub151, pct, alpha,
                                            depth_calibration=depth_cal)
    sim = score_similarity_to_219(new_sub)

    results.append({
        'type': 'uniform',
        'contrast': 151,
        'percentile': pct,
        'alpha': alpha,
        'depth_cal': depth_cal,
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'angle_mean': new_sub['scaled_angle'].mean(),
        'lr_mean': new_sub['scaled_left_right'].mean(),
        'sim_to_219': sim
    })

print(f"Tested {len(results)} uniform configurations")

# Test 2: Per-target alpha
per_target_configs = [
    {'scaled_angle': 1.0, 'scaled_depth': 0.5, 'scaled_left_right': 1.0},
    {'scaled_angle': 1.2, 'scaled_depth': 0.8, 'scaled_left_right': 1.0},
    {'scaled_angle': 0.8, 'scaled_depth': 1.0, 'scaled_left_right': 0.8},
    {'scaled_angle': 1.5, 'scaled_depth': 0.5, 'scaled_left_right': 1.2},
    {'scaled_angle': 1.0, 'scaled_depth': 1.5, 'scaled_left_right': 0.8},
]

for pct in percentiles:
    for config in per_target_configs:
        for depth_cal in [0.5055, None]:
            new_sub, n_modified = selective_amplify(sub133, sub151, pct, 1.0,
                                                    per_target_alpha=config,
                                                    depth_calibration=depth_cal)
            sim = score_similarity_to_219(new_sub)

            results.append({
                'type': 'per_target',
                'contrast': 151,
                'percentile': pct,
                'alpha': str(config),
                'depth_cal': depth_cal,
                'n_modified': n_modified,
                'depth_mean': new_sub['scaled_depth'].mean(),
                'angle_mean': new_sub['scaled_angle'].mean(),
                'lr_mean': new_sub['scaled_left_right'].mean(),
                'sim_to_219': sim
            })

print(f"Tested {len(results)} total configurations (added per-target)")

# Test 3: Different contrast submissions
print("\nTesting different contrast submissions...")
for contrast_num, contrast_df in contrast_subs.items():
    if list(contrast_df['id']) != list(sub133['id']):
        print(f"  Skipping sub {contrast_num} - IDs don't match")
        continue

    for pct, alpha in product([88, 91, 93], [0.9, 1.0, 1.1, 1.2]):
        new_sub, n_modified = selective_amplify(sub133, contrast_df, pct, alpha,
                                                depth_calibration=0.5055)
        sim = score_similarity_to_219(new_sub)

        results.append({
            'type': 'alt_contrast',
            'contrast': contrast_num,
            'percentile': pct,
            'alpha': alpha,
            'depth_cal': 0.5055,
            'n_modified': n_modified,
            'depth_mean': new_sub['scaled_depth'].mean(),
            'angle_mean': new_sub['scaled_angle'].mean(),
            'lr_mean': new_sub['scaled_left_right'].mean(),
            'sim_to_219': sim
        })

print(f"Tested {len(results)} total configurations (added alt contrasts)")

# Test 4: Two-stage amplification
print("\nTesting two-stage amplification...")
if 9 in contrast_subs and list(contrast_subs[9]['id']) == list(sub133['id']):
    sub9 = contrast_subs[9]
    for pct1, alpha1, pct2, alpha2 in product([90, 91, 92], [1.0, 1.1, 1.2],
                                               [85, 88], [0.3, 0.5, 0.7]):
        new_sub, n1, n2 = two_stage_amplify(sub133, sub151, sub9,
                                            pct1, alpha1, pct2, alpha2,
                                            depth_calibration=0.5055)
        sim = score_similarity_to_219(new_sub)

        results.append({
            'type': 'two_stage',
            'contrast': '151+9',
            'percentile': f"{pct1}+{pct2}",
            'alpha': f"{alpha1}+{alpha2}",
            'depth_cal': 0.5055,
            'n_modified': f"{n1}+{n2}",
            'depth_mean': new_sub['scaled_depth'].mean(),
            'angle_mean': new_sub['scaled_angle'].mean(),
            'lr_mean': new_sub['scaled_left_right'].mean(),
            'sim_to_219': sim
        })

print(f"Tested {len(results)} total configurations")

# Convert to DataFrame and sort by similarity to 219
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sim_to_219')

# Save results
results_path = os.path.join(OUTPUT_DIR, 'selective_amplification_grid_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nSaved grid search results to: {results_path}")

# Print top configurations closest to Sub 219
print("\n=== Top 20 Configurations Most Similar to Sub 219 ===")
print(results_df.head(20).to_string(index=False))

# Print configurations that might beat Sub 219
print("\n=== Potential Improvements (slight variations from 219) ===")
# We want configurations that are similar but not identical to 219
best_configs = results_df[(results_df['sim_to_219'] > 0.0001) &
                          (results_df['sim_to_219'] < 0.01)].head(30)
print(best_configs.to_string(index=False))

# Generate promising submissions
print("\n=== Generating Promising Submissions ===")
submission_num = 335

# 1. Best match to 219 approach
print("\n1. Recreating Sub 219 approach...")
best_match = results_df.iloc[0]
print(f"   Best match: {dict(best_match)}")

# 2. Generate variations that might beat 219
variation_configs = [
    # Slight alpha variations from 219
    {'percentile': 91, 'alpha': 1.05, 'depth_cal': 0.5055, 'name': 'alpha_1.05'},
    {'percentile': 91, 'alpha': 0.95, 'depth_cal': 0.5055, 'name': 'alpha_0.95'},
    {'percentile': 91, 'alpha': 1.15, 'depth_cal': 0.5055, 'name': 'alpha_1.15'},

    # Different percentile thresholds
    {'percentile': 89, 'alpha': 1.0, 'depth_cal': 0.5055, 'name': 'pct_89'},
    {'percentile': 90, 'alpha': 1.0, 'depth_cal': 0.5055, 'name': 'pct_90'},
    {'percentile': 92, 'alpha': 1.0, 'depth_cal': 0.5055, 'name': 'pct_92'},
    {'percentile': 93, 'alpha': 1.0, 'depth_cal': 0.5055, 'name': 'pct_93'},

    # Different depth calibrations
    {'percentile': 91, 'alpha': 1.0, 'depth_cal': 0.5050, 'name': 'depth_0.5050'},
    {'percentile': 91, 'alpha': 1.0, 'depth_cal': 0.5060, 'name': 'depth_0.5060'},
    {'percentile': 91, 'alpha': 1.0, 'depth_cal': 0.5045, 'name': 'depth_0.5045'},

    # Combined variations
    {'percentile': 90, 'alpha': 1.1, 'depth_cal': 0.5055, 'name': 'pct90_alpha1.1'},
    {'percentile': 92, 'alpha': 0.9, 'depth_cal': 0.5055, 'name': 'pct92_alpha0.9'},
    {'percentile': 91, 'alpha': 1.2, 'depth_cal': 0.5050, 'name': 'alpha1.2_depth0.505'},

    # More aggressive amplification
    {'percentile': 88, 'alpha': 1.3, 'depth_cal': 0.5055, 'name': 'aggressive_pct88'},
    {'percentile': 85, 'alpha': 1.5, 'depth_cal': 0.5055, 'name': 'very_aggressive'},

    # Conservative amplification
    {'percentile': 93, 'alpha': 0.8, 'depth_cal': 0.5055, 'name': 'conservative_pct93'},
    {'percentile': 95, 'alpha': 0.7, 'depth_cal': 0.5055, 'name': 'very_conservative'},
]

# Per-target variations
per_target_variations = [
    {'scaled_angle': 1.1, 'scaled_depth': 0.9, 'scaled_left_right': 1.0},
    {'scaled_angle': 0.9, 'scaled_depth': 1.1, 'scaled_left_right': 1.0},
    {'scaled_angle': 1.0, 'scaled_depth': 1.0, 'scaled_left_right': 1.2},
    {'scaled_angle': 1.2, 'scaled_depth': 0.8, 'scaled_left_right': 1.1},
]

submissions_info = []

for config in variation_configs:
    new_sub, n_modified = selective_amplify(sub133, sub151,
                                            config['percentile'],
                                            config['alpha'],
                                            depth_calibration=config['depth_cal'])

    # Save submission
    sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
    new_sub.to_csv(sub_path, index=False)

    submissions_info.append({
        'submission_num': submission_num,
        'config': config['name'],
        'percentile': config['percentile'],
        'alpha': config['alpha'],
        'depth_cal': config['depth_cal'],
        'n_modified': n_modified,
        'depth_mean': new_sub['scaled_depth'].mean(),
        'angle_mean': new_sub['scaled_angle'].mean(),
        'lr_mean': new_sub['scaled_left_right'].mean(),
        'file': sub_path
    })

    print(f"Saved submission_{submission_num}: {config['name']}")
    submission_num += 1

# Per-target alpha submissions
for i, pt_config in enumerate(per_target_variations):
    for pct in [90, 91, 92]:
        new_sub, n_modified = selective_amplify(sub133, sub151, pct, 1.0,
                                                per_target_alpha=pt_config,
                                                depth_calibration=0.5055)

        sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
        new_sub.to_csv(sub_path, index=False)

        submissions_info.append({
            'submission_num': submission_num,
            'config': f'per_target_{i}_pct{pct}',
            'percentile': pct,
            'alpha': str(pt_config),
            'depth_cal': 0.5055,
            'n_modified': n_modified,
            'depth_mean': new_sub['scaled_depth'].mean(),
            'angle_mean': new_sub['scaled_angle'].mean(),
            'lr_mean': new_sub['scaled_left_right'].mean(),
            'file': sub_path
        })

        print(f"Saved submission_{submission_num}: per_target_{i}_pct{pct}")
        submission_num += 1

# Two-stage submissions (if available)
if 9 in contrast_subs and list(contrast_subs[9]['id']) == list(sub133['id']):
    sub9 = contrast_subs[9]
    two_stage_configs = [
        (91, 1.0, 85, 0.5),
        (91, 1.1, 88, 0.3),
        (90, 1.0, 85, 0.5),
        (92, 0.9, 88, 0.4),
    ]

    for pct1, alpha1, pct2, alpha2 in two_stage_configs:
        new_sub, n1, n2 = two_stage_amplify(sub133, sub151, sub9,
                                            pct1, alpha1, pct2, alpha2,
                                            depth_calibration=0.5055)

        sub_path = os.path.join(SUBMISSION_DIR, f"submission_{submission_num}.csv")
        new_sub.to_csv(sub_path, index=False)

        submissions_info.append({
            'submission_num': submission_num,
            'config': f'two_stage_p{pct1}a{alpha1}_p{pct2}a{alpha2}',
            'percentile': f'{pct1}+{pct2}',
            'alpha': f'{alpha1}+{alpha2}',
            'depth_cal': 0.5055,
            'n_modified': f'{n1}+{n2}',
            'depth_mean': new_sub['scaled_depth'].mean(),
            'angle_mean': new_sub['scaled_angle'].mean(),
            'lr_mean': new_sub['scaled_left_right'].mean(),
            'file': sub_path
        })

        print(f"Saved submission_{submission_num}: two_stage_p{pct1}a{alpha1}_p{pct2}a{alpha2}")
        submission_num += 1

# Save submission info
sub_info_df = pd.DataFrame(submissions_info)
sub_info_path = os.path.join(OUTPUT_DIR, 'selective_amplification_submissions.csv')
sub_info_df.to_csv(sub_info_path, index=False)
print(f"\nSaved submission info to: {sub_info_path}")

print("\n=== Summary ===")
print(f"Generated {len(submissions_info)} new submissions")
print(f"Submission numbers: 335 to {submission_num - 1}")
print("\nSubmission Details:")
for info in submissions_info[:10]:
    print(f"  {info['submission_num']}: {info['config']} "
          f"(pct={info['percentile']}, alpha={info['alpha']}, "
          f"depth_mean={info['depth_mean']:.4f})")
if len(submissions_info) > 10:
    print(f"  ... and {len(submissions_info) - 10} more")
