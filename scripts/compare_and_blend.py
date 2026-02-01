"""
Compare physics submission to sub25 and create blends.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'
NEW_SUBMISSION_DIR = PROJECT_DIR / 'submissions'

# Load submissions
physics = pd.read_csv(NEW_SUBMISSION_DIR / 'submission_1.csv')
sub25 = pd.read_csv(SUBMISSION_DIR / 'submission_25.csv')

print("=" * 80)
print("COMPARISON: Physics vs Sub25 (Best: LB 0.008305)")
print("=" * 80)
print()

# Map column names
col_map = {
    'angle': 'scaled_angle',
    'depth': 'scaled_depth',
    'left_right': 'scaled_left_right'
}

for target in ['angle', 'depth', 'left_right']:
    sub25_col = col_map[target]
    print(f"{target.upper()}:")
    print(f"  Physics: mean={physics[target].mean():.4f}, std={physics[target].std():.4f}, min={physics[target].min():.4f}, max={physics[target].max():.4f}")
    print(f"  Sub25:   mean={sub25[sub25_col].mean():.4f}, std={sub25[sub25_col].std():.4f}, min={sub25[sub25_col].min():.4f}, max={sub25[sub25_col].max():.4f}")

    # Correlation
    corr = np.corrcoef(physics[target], sub25[sub25_col])[0, 1]
    print(f"  Correlation: {corr:.4f}")

    # Mean absolute difference
    mad = np.abs(physics[target] - sub25[sub25_col]).mean()
    print(f"  Mean Abs Diff: {mad:.4f}")
    print()

# Create blends
print("=" * 80)
print("CREATING BLENDS")
print("=" * 80)
print()

blend_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

next_sub_num = 74  # Based on existing submissions

for ratio in blend_ratios:
    blend = pd.DataFrame({
        'id': physics['id'],
        'scaled_angle': ratio * physics['angle'] + (1 - ratio) * sub25['scaled_angle'],
        'scaled_depth': ratio * physics['depth'] + (1 - ratio) * sub25['scaled_depth'],
        'scaled_left_right': ratio * physics['left_right'] + (1 - ratio) * sub25['scaled_left_right'],
    })

    # Clip to valid range
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        blend[col] = np.clip(blend[col], 0.0, 1.0)

    blend_path = SUBMISSION_DIR / f'submission_{next_sub_num}.csv'
    blend.to_csv(blend_path, index=False)

    print(f"Sub {next_sub_num}: {int(ratio*100)}% physics + {int((1-ratio)*100)}% sub25")
    print(f"  angle: mean={blend['scaled_angle'].mean():.4f}, std={blend['scaled_angle'].std():.4f}")
    print(f"  depth: mean={blend['scaled_depth'].mean():.4f}, std={blend['scaled_depth'].std():.4f}")
    print(f"  left_right: mean={blend['scaled_left_right'].mean():.4f}, std={blend['scaled_left_right'].std():.4f}")
    print(f"  Saved to: {blend_path}")
    print()

    next_sub_num += 1

# Also save pure physics to submission folder with correct column names
physics_renamed = pd.DataFrame({
    'id': physics['id'],
    'scaled_angle': physics['angle'],
    'scaled_depth': physics['depth'],
    'scaled_left_right': physics['left_right'],
})
physics_renamed.to_csv(SUBMISSION_DIR / f'submission_{next_sub_num}.csv', index=False)
print(f"Sub {next_sub_num}: 100% physics")
print(f"  Saved to: {SUBMISSION_DIR / f'submission_{next_sub_num}.csv'}")

print()
print("=" * 80)
print("SUMMARY OF NEW SUBMISSIONS")
print("=" * 80)
print()
print("Sub 74: 10% physics + 90% sub25")
print("Sub 75: 20% physics + 80% sub25")
print("Sub 76: 30% physics + 70% sub25")
print("Sub 77: 40% physics + 60% sub25")
print("Sub 78: 50% physics + 50% sub25")
print("Sub 79: 100% physics")
print()
print("Recommended to test: Sub 76 (30% physics) - conservative blend")
print("High risk/high reward: Sub 79 (100% physics)")
