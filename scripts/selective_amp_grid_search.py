"""
Grid Search for Optimal Selective Amplification Parameters

Best known: Sub 219 with pctl=91, alpha=1.1 (LB: 0.007682)

This script explores parameter combinations to find potentially better settings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"


def selective_amplify(base_df, contrast_df, percentile=91, alpha=1.1):
    """
    Selective amplification: amplify in the direction of base
    for samples where base and contrast disagree.
    """
    result = base_df.copy()
    target_cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

    for col in target_cols:
        diff = np.abs(base_df[col].values - contrast_df[col].values)
        threshold = np.percentile(diff, percentile)
        mask = diff >= threshold

        direction = np.sign(base_df[col].values - contrast_df[col].values)
        amplification = direction * diff * alpha

        result[col] = base_df[col].values.copy()
        result.loc[mask, col] = result.loc[mask, col] + amplification[mask]

    return result


def target_specific_amplify(base_df, contrast_df,
                            angle_pctl, angle_alpha,
                            depth_pctl, depth_alpha,
                            lr_pctl, lr_alpha):
    """Target-specific amplification parameters."""
    result = base_df.copy()

    params = {
        'scaled_angle': (angle_pctl, angle_alpha),
        'scaled_depth': (depth_pctl, depth_alpha),
        'scaled_left_right': (lr_pctl, lr_alpha)
    }

    for col, (pctl, alpha) in params.items():
        diff = np.abs(base_df[col].values - contrast_df[col].values)
        threshold = np.percentile(diff, pctl)
        mask = diff >= threshold

        direction = np.sign(base_df[col].values - contrast_df[col].values)
        amplification = direction * diff * alpha

        result[col] = base_df[col].values.copy()
        result.loc[mask, col] = result.loc[mask, col] + amplification[mask]

    return result


def get_next_submission_number():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            pass
    return max(numbers) + 1 if numbers else 1


def main():
    print("=" * 80)
    print("SELECTIVE AMPLIFICATION GRID SEARCH")
    print("=" * 80)

    # Load base submissions
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")  # 4-way blend (0.007809)
    sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")  # Contrast for Sub219

    # Check if sub151 exists
    if not (SUBMISSION_DIR / "submission_151.csv").exists():
        print("submission_151.csv not found - using sub133 as contrast")
        sub151 = sub133.copy()

    print(f"Base: Sub133 (LB: 0.007809)")
    print(f"Contrast: Sub151")

    # Grid search parameters
    percentiles = [88, 89, 90, 91, 92, 93]
    alphas = [0.8, 0.9, 1.0, 1.1, 1.2]

    print(f"\nSearching {len(percentiles)} percentiles x {len(alphas)} alphas = {len(percentiles)*len(alphas)} combinations")

    sub_num = get_next_submission_number()
    results = []

    print("\n" + "=" * 80)
    print("UNIFORM PARAMETER SEARCH")
    print("=" * 80)

    for pctl, alpha in product(percentiles, alphas):
        result = selective_amplify(sub133, sub151, percentile=pctl, alpha=alpha)

        # Calibrate depth
        result['scaled_depth'] = result['scaled_depth'] - result['scaled_depth'].mean() + 0.5055
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            result[col] = result[col].clip(0, 1)

        # Save
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        result.to_csv(sub_path, index=False)

        profile = {
            'sub': sub_num,
            'pctl': pctl,
            'alpha': alpha,
            'angle_std': result['scaled_angle'].std(),
            'depth_mean': result['scaled_depth'].mean()
        }
        results.append(profile)

        print(f"  Sub {sub_num}: pctl={pctl}, alpha={alpha}, "
              f"angle_std={profile['angle_std']:.4f}")
        sub_num += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find submissions closest to target profile
    target_angle_std = 0.137
    target_depth_mean = 0.5055

    print(f"\nSubmissions closest to target profile (angle_std={target_angle_std}, depth_mean={target_depth_mean}):")

    for r in sorted(results, key=lambda x: abs(x['angle_std'] - target_angle_std))[:5]:
        diff = abs(r['angle_std'] - target_angle_std)
        print(f"  Sub {r['sub']}: pctl={r['pctl']}, alpha={r['alpha']}, "
              f"angle_std={r['angle_std']:.4f} (diff={diff:.4f})")

    print("\n" + "=" * 80)
    print("TARGET-SPECIFIC SEARCH")
    print("=" * 80)

    # Try target-specific parameters based on what we learned
    # Angle: strong physics signal, might benefit from higher alpha
    # Depth: temporal features better, maybe lower alpha
    # Left_right: weak signal, conservative

    configs = [
        # (angle_pctl, angle_alpha, depth_pctl, depth_alpha, lr_pctl, lr_alpha)
        (91, 1.2, 91, 1.0, 91, 1.0),  # Higher angle amplification
        (92, 1.1, 90, 1.0, 90, 1.0),  # Higher angle threshold
        (91, 1.1, 89, 0.9, 91, 1.1),  # Lower depth threshold
        (90, 1.0, 92, 1.2, 90, 1.0),  # Higher depth amplification
        (92, 1.2, 90, 0.8, 91, 1.0),  # Strong angle, weak depth
    ]

    for config in configs:
        result = target_specific_amplify(
            sub133, sub151,
            angle_pctl=config[0], angle_alpha=config[1],
            depth_pctl=config[2], depth_alpha=config[3],
            lr_pctl=config[4], lr_alpha=config[5]
        )

        # Calibrate
        result['scaled_depth'] = result['scaled_depth'] - result['scaled_depth'].mean() + 0.5055
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            result[col] = result[col].clip(0, 1)

        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        result.to_csv(sub_path, index=False)

        print(f"  Sub {sub_num}: angle=({config[0]},{config[1]}), "
              f"depth=({config[2]},{config[3]}), lr=({config[4]},{config[5]}), "
              f"angle_std={result['scaled_angle'].std():.4f}")
        sub_num += 1


if __name__ == "__main__":
    main()
