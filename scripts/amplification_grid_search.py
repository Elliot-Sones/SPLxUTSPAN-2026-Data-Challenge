"""
Grid Search for Selective Amplification

Sub 219 used: pctl=91, alpha=1.1
Try variations to beat it.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Load submissions
sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")
sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

# Also load other potential contrast submissions
contrast_subs = {
    '151': sub151,
}

# Try to load more contrast submissions
for num in [113, 145, 146, 147, 148, 149, 150]:
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    if path.exists():
        contrast_subs[str(num)] = pd.read_csv(path)

print(f"Loaded {len(contrast_subs)} contrast submissions")

# Get next submission number
existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
next_num = max(nums) + 1

# Grid search parameters
percentiles = [85, 88, 90, 91, 92, 93, 95]
alphas = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

results = []

print("\nRunning grid search...")
print("="*60)

for contrast_name, contrast_sub in contrast_subs.items():
    # Compute differences
    diff_angle = sub133['scaled_angle'].values - contrast_sub['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - contrast_sub['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - contrast_sub['scaled_left_right'].values
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    for pctl in percentiles:
        threshold = np.percentile(total_diff, pctl)
        mask = total_diff > threshold
        n_samples = mask.sum()

        for alpha in alphas:
            # Apply amplification
            new_angle = sub133['scaled_angle'].values.copy()
            new_depth = sub133['scaled_depth'].values.copy()
            new_lr = sub133['scaled_left_right'].values.copy()

            new_angle[mask] += alpha * diff_angle[mask]
            new_depth[mask] += alpha * diff_depth[mask]
            new_lr[mask] += alpha * diff_lr[mask]

            # Calibrate
            new_depth = new_depth - np.mean(new_depth) + 0.5055
            new_angle = np.clip(new_angle, 0, 1)
            new_depth = np.clip(new_depth, 0, 1)
            new_lr = np.clip(new_lr, 0, 1)

            # Compute metrics
            angle_std = np.std(new_angle)
            depth_mean = np.mean(new_depth)

            # Compare to Sub 219
            corr_219 = np.corrcoef(new_angle, sub219['scaled_angle'].values)[0, 1]

            results.append({
                'contrast': contrast_name,
                'pctl': pctl,
                'alpha': alpha,
                'n_samples': n_samples,
                'angle_std': angle_std,
                'depth_mean': depth_mean,
                'corr_219': corr_219,
            })

# Sort by angle_std (lower is often better based on our findings)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('angle_std')

print("\nTop 20 configurations by angle_std:")
print(results_df.head(20).to_string(index=False))

# Save promising submissions
print("\n" + "="*60)
print("SAVING PROMISING SUBMISSIONS")
print("="*60)

# Save top 5 unique configurations
saved = set()
for _, row in results_df.head(20).iterrows():
    key = (row['contrast'], row['pctl'], row['alpha'])
    if key in saved:
        continue

    contrast_sub = contrast_subs[row['contrast']]
    diff_angle = sub133['scaled_angle'].values - contrast_sub['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - contrast_sub['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - contrast_sub['scaled_left_right'].values
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    threshold = np.percentile(total_diff, row['pctl'])
    mask = total_diff > threshold

    new_angle = sub133['scaled_angle'].values.copy()
    new_depth = sub133['scaled_depth'].values.copy()
    new_lr = sub133['scaled_left_right'].values.copy()

    new_angle[mask] += row['alpha'] * diff_angle[mask]
    new_depth[mask] += row['alpha'] * diff_depth[mask]
    new_lr[mask] += row['alpha'] * diff_lr[mask]

    new_depth = new_depth - np.mean(new_depth) + 0.5055
    new_angle = np.clip(new_angle, 0, 1)
    new_depth = np.clip(new_depth, 0, 1)
    new_lr = np.clip(new_lr, 0, 1)

    submission = pd.DataFrame({
        'id': sub133['id'],
        'scaled_angle': new_angle,
        'scaled_depth': new_depth,
        'scaled_left_right': new_lr
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file} (contrast={row['contrast']}, pctl={row['pctl']}, alpha={row['alpha']}, angle_std={row['angle_std']:.6f})")

    saved.add(key)
    next_num += 1

    if len(saved) >= 5:
        break

# Also try per-target amplification
print("\n" + "="*60)
print("PER-TARGET AMPLIFICATION")
print("="*60)

# Amplify only angle (often most important)
contrast_sub = sub151
diff_angle = sub133['scaled_angle'].values - contrast_sub['scaled_angle'].values
total_diff = np.abs(diff_angle)

for pctl in [88, 90, 92]:
    threshold = np.percentile(total_diff, pctl)
    mask = total_diff > threshold

    for alpha in [1.0, 1.2, 1.5]:
        new_angle = sub133['scaled_angle'].values.copy()
        new_angle[mask] += alpha * diff_angle[mask]
        new_angle = np.clip(new_angle, 0, 1)

        submission = pd.DataFrame({
            'id': sub133['id'],
            'scaled_angle': new_angle,
            'scaled_depth': sub133['scaled_depth'].values,  # Keep original
            'scaled_left_right': sub133['scaled_left_right'].values  # Keep original
        })

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)
        print(f"Saved: {output_file} (angle-only, pctl={pctl}, alpha={alpha}, angle_std={np.std(new_angle):.6f})")
        next_num += 1

print("\nDone!")
