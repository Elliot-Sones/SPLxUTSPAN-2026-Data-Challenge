"""
Calibrated Ensemble - Combine Diverse Models with Profile Calibration

Strategy:
1. Combine predictions from multiple diverse models
2. Calibrate final predictions to achieve optimal profile
3. Focus on improving depth (where we have most diversity)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"


def calibrate_to_profile(predictions, target_angle_std=0.1379, target_depth_mean=0.505):
    """Calibrate predictions to target profile."""
    calibrated = predictions.copy()

    # Calibrate depth mean (simple shift)
    current_depth_mean = calibrated[:, 1].mean()
    calibrated[:, 1] += (target_depth_mean - current_depth_mean)

    # Calibrate angle std (shrinkage toward mean)
    current_angle_std = calibrated[:, 0].std()
    if current_angle_std > target_angle_std:
        angle_mean = calibrated[:, 0].mean()
        shrinkage = target_angle_std / current_angle_std
        calibrated[:, 0] = angle_mean + shrinkage * (calibrated[:, 0] - angle_mean)

    return calibrated


def main():
    print("=" * 70)
    print("CALIBRATED ENSEMBLE")
    print("=" * 70)

    # Load best submission for reference
    sub113 = pd.read_csv(SUBMISSION_DIR / "submission_113.csv")

    # Load diverse models
    submissions = {
        120: pd.read_csv(SUBMISSION_DIR / "submission_120.csv"),  # Innovative model
        121: pd.read_csv(SUBMISSION_DIR / "submission_121.csv"),  # Profile-optimized
        124: pd.read_csv(SUBMISSION_DIR / "submission_124.csv"),  # Body segment
    }

    cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

    # Get correlation of each submission with Sub 113
    print("\nCorrelations with Sub 113:")
    for sub_num, df in submissions.items():
        for col in cols:
            r = np.corrcoef(sub113[col], df[col])[0, 1]
            print(f"  Sub {sub_num} {col}: r={r:.4f}")

    # Try different ensemble combinations
    print("\n" + "=" * 70)
    print("Testing ensemble combinations:")
    print("=" * 70)

    best_profile_score = float('inf')
    best_config = None
    best_predictions = None

    # Test various weight combinations
    for w113 in [0.85, 0.90, 0.92, 0.95]:
        remaining = 1 - w113
        for w120 in [0, remaining/3, remaining/2, remaining]:
            for w121 in [0, remaining/3, remaining/2]:
                w124 = remaining - w120 - w121
                if w124 < 0:
                    continue

                # Create ensemble
                ensemble = (
                    w113 * sub113[cols].values +
                    w120 * submissions[120][cols].values +
                    w121 * submissions[121][cols].values +
                    w124 * submissions[124][cols].values
                )

                # Calibrate
                calibrated = calibrate_to_profile(ensemble)

                angle_std = np.std(calibrated[:, 0])
                depth_mean = np.mean(calibrated[:, 1])

                # Score: prefer low angle_std and depth_mean close to 0.505
                profile_score = abs(angle_std - 0.1379) + abs(depth_mean - 0.505)

                if profile_score < best_profile_score:
                    best_profile_score = profile_score
                    best_config = (w113, w120, w121, w124)
                    best_predictions = calibrated.copy()
                    print(f"  w113={w113:.2f}, w120={w120:.3f}, w121={w121:.3f}, w124={w124:.3f}: "
                          f"angle_std={angle_std:.4f}, depth_mean={depth_mean:.4f}")

    print(f"\nBest config: w113={best_config[0]:.2f}, w120={best_config[1]:.3f}, "
          f"w121={best_config[2]:.3f}, w124={best_config[3]:.3f}")

    # Create uncalibrated version with best weights
    w113, w120, w121, w124 = best_config
    uncalibrated = (
        w113 * sub113[cols].values +
        w120 * submissions[120][cols].values +
        w121 * submissions[121][cols].values +
        w124 * submissions[124][cols].values
    )

    # Also try without calibration
    print("\nUncalibrated version:")
    print(f"  angle_std: {np.std(uncalibrated[:, 0]):.4f}")
    print(f"  depth_mean: {np.mean(uncalibrated[:, 1]):.4f}")

    # Create both calibrated and uncalibrated submissions
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

    # Save uncalibrated (often performs better on actual LB)
    submission_uncal = pd.DataFrame({
        'id': sub113['id'],
        'scaled_angle': uncalibrated[:, 0],
        'scaled_depth': uncalibrated[:, 1],
        'scaled_left_right': uncalibrated[:, 2],
    })

    filepath_uncal = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission_uncal.to_csv(filepath_uncal, index=False)

    angle_std = submission_uncal['scaled_angle'].std()
    depth_mean = submission_uncal['scaled_depth'].mean()

    print(f"\n{'='*70}")
    print(f"SUBMISSION {next_num}: calibrated_ensemble (uncalibrated)")
    print(f"{'='*70}")
    print(f"  Weights: {w113:.0%} sub113 + {w120:.1%} sub120 + {w121:.1%} sub121 + {w124:.1%} sub124")
    print(f"  angle_std: {angle_std:.4f} (Sub 113: 0.1379)")
    print(f"  depth_mean: {depth_mean:.4f} (Sub 113: 0.5050)")
    print(f"  File: {filepath_uncal}")

    # Also save calibrated version
    next_num += 1
    submission_cal = pd.DataFrame({
        'id': sub113['id'],
        'scaled_angle': best_predictions[:, 0],
        'scaled_depth': best_predictions[:, 1],
        'scaled_left_right': best_predictions[:, 2],
    })

    filepath_cal = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission_cal.to_csv(filepath_cal, index=False)

    angle_std_cal = submission_cal['scaled_angle'].std()
    depth_mean_cal = submission_cal['scaled_depth'].mean()

    print(f"\n{'='*70}")
    print(f"SUBMISSION {next_num}: calibrated_ensemble (calibrated)")
    print(f"{'='*70}")
    print(f"  angle_std: {angle_std_cal:.4f} (target: 0.1379)")
    print(f"  depth_mean: {depth_mean_cal:.4f} (target: 0.505)")
    print(f"  File: {filepath_cal}")

    # Compare with Sub 113
    print("\nComparison with Sub 113:")
    print(f"  Sub 113: angle_std=0.1379, depth_mean=0.5050, LB=0.008031")
    print(f"  New (uncal): angle_std={angle_std:.4f}, depth_mean={depth_mean:.4f}")
    print(f"  New (cal): angle_std={angle_std_cal:.4f}, depth_mean={depth_mean_cal:.4f}")

    return filepath_uncal, filepath_cal


if __name__ == "__main__":
    main()
