"""
Optimal Blend with Submission 121

Based on analysis:
- Sub 121 has optimal depth_mean (0.5050) after calibration
- Blending at 5% with Sub 113 gives angle_std=0.1374 (better than 0.1379)
- This should marginally improve over Sub 113's LB of 0.008031
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

def main():
    # Load submissions
    sub113 = pd.read_csv(SUBMISSION_DIR / "submission_113.csv")
    sub121 = pd.read_csv(SUBMISSION_DIR / "submission_121.csv")

    cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

    # Test different blend weights
    print("Blend analysis:")
    print("=" * 60)

    best_w = 0.05
    best_profile = float('inf')

    for w in [0.03, 0.05, 0.07, 0.10]:
        blend = (1 - w) * sub113[cols].values + w * sub121[cols].values
        angle_std = np.std(blend[:, 0])
        depth_mean = np.mean(blend[:, 1])

        # Profile score: lower is better
        profile = abs(angle_std - 0.1379) + abs(depth_mean - 0.505)

        print(f"w={w:.2f}: angle_std={angle_std:.4f}, depth_mean={depth_mean:.4f}, profile_score={profile:.6f}")

        if profile < best_profile:
            best_profile = profile
            best_w = w

    print(f"\nBest weight: {best_w}")

    # Create optimal blend
    blend = (1 - best_w) * sub113[cols].values + best_w * sub121[cols].values

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        try:
            name = f.stem
            if name.startswith('submission_'):
                nums.append(int(name.split('_')[1]))
            elif name.startswith('submission'):
                nums.append(int(name.replace('submission', '')))
        except:
            pass
    next_num = max(nums) + 1 if nums else 1

    submission = pd.DataFrame({
        'id': sub113['id'],
        'scaled_angle': blend[:, 0],
        'scaled_depth': blend[:, 1],
        'scaled_left_right': blend[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: {100-int(best_w*100)}% sub113 + {int(best_w*100)}% sub121")
    print(f"{'='*60}")
    print(f"  angle_std: {angle_std:.4f} (Sub 113: 0.1379)")
    print(f"  depth_mean: {depth_mean:.4f} (Sub 113: 0.5050)")
    print(f"  File: {filepath}")

    # Comparison with Sub 113
    print("\nComparison with Sub 113:")
    print(f"  angle_std: {0.1379:.4f} -> {angle_std:.4f} ({'+' if angle_std > 0.1379 else ''}{(angle_std - 0.1379)*100:.2f}%)")
    print(f"  depth_mean: {0.5050:.4f} -> {depth_mean:.4f}")

    return filepath


if __name__ == "__main__":
    main()
