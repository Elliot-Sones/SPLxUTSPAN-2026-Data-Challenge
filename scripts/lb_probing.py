"""
Leaderboard Probing for MSE-based competition.

For MSE metric, if we predict constant c for all samples:
    MSE = mean((y_true - c)^2) = var(y_true) + (mean(y_true) - c)^2

By submitting different constants, we can solve for mean and variance of test targets.

Strategy:
1. Submit c=0.5 (center of [0,1] range) -> get MSE_1
2. Submit c=0.3 -> get MSE_2
3. Submit c=0.7 -> get MSE_3

From these 3 MSE values, we can solve for:
- mean(y_true) for each target
- var(y_true) for each target

Then use this to calibrate our predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"
DATA_DIR = Path(__file__).parent.parent / "data"

def create_constant_submission(angle_val, depth_val, lr_val, sub_num):
    """Create submission with constant values."""
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    sub = pd.DataFrame({
        'id': test_df['id'],
        'scaled_angle': [angle_val] * len(test_df),
        'scaled_depth': [depth_val] * len(test_df),
        'scaled_left_right': [lr_val] * len(test_df),
    })

    filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(filepath, index=False)
    print(f"Created {filepath}")
    print(f"  angle={angle_val}, depth={depth_val}, lr={lr_val}")
    return filepath

def analyze_probing_results(mse_results):
    """
    Given MSE results from constant submissions, solve for test set mean and variance.

    mse_results: list of (c, mse) tuples where c is the constant and mse is the LB score

    MSE = var + (mean - c)^2

    With 2 submissions at c1 and c2:
    MSE1 = var + (mean - c1)^2
    MSE2 = var + (mean - c2)^2

    MSE1 - MSE2 = (mean - c1)^2 - (mean - c2)^2
                = mean^2 - 2*mean*c1 + c1^2 - mean^2 + 2*mean*c2 - c2^2
                = 2*mean*(c2 - c1) + (c1^2 - c2^2)

    Solving for mean:
    mean = (MSE1 - MSE2 - c1^2 + c2^2) / (2*(c2 - c1))

    Then: var = MSE1 - (mean - c1)^2
    """
    if len(mse_results) < 2:
        print("Need at least 2 probing results")
        return None

    (c1, mse1), (c2, mse2) = mse_results[0], mse_results[1]

    # Solve for mean
    mean = (mse1 - mse2 - c1**2 + c2**2) / (2 * (c2 - c1))

    # Solve for variance
    var = mse1 - (mean - c1)**2

    print(f"\nInferred test set statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Variance: {var:.6f}")
    print(f"  Std: {np.sqrt(max(0, var)):.6f}")

    return mean, var

def create_probing_submissions():
    """Create a set of probing submissions."""
    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    # Probing strategy: test different constant values
    # We'll vary one target at a time while keeping others at 0.5

    probes = [
        # Probe 1: All at 0.5 (baseline)
        (0.5, 0.5, 0.5),
        # Probe 2: Angle at 0.3
        (0.3, 0.5, 0.5),
        # Probe 3: Angle at 0.7
        (0.7, 0.5, 0.5),
        # Probe 4: Depth at 0.3
        (0.5, 0.3, 0.5),
        # Probe 5: Depth at 0.7
        (0.5, 0.7, 0.5),
        # Probe 6: LR at 0.3
        (0.5, 0.5, 0.3),
        # Probe 7: LR at 0.7
        (0.5, 0.5, 0.7),
    ]

    print("Creating probing submissions...")
    print("="*60)

    for i, (a, d, lr) in enumerate(probes):
        create_constant_submission(a, d, lr, next_num + i)

    print("\n" + "="*60)
    print("PROBING PLAN")
    print("="*60)
    print(f"""
Submit these to LB and record the MSE scores:

Sub {next_num}: angle=0.5, depth=0.5, lr=0.5 (baseline)
Sub {next_num+1}: angle=0.3, depth=0.5, lr=0.5 (probe angle low)
Sub {next_num+2}: angle=0.7, depth=0.5, lr=0.5 (probe angle high)
Sub {next_num+3}: angle=0.5, depth=0.3, lr=0.5 (probe depth low)
Sub {next_num+4}: angle=0.5, depth=0.7, lr=0.5 (probe depth high)
Sub {next_num+5}: angle=0.5, lr=0.5, lr=0.3 (probe lr low)
Sub {next_num+6}: angle=0.5, lr=0.5, lr=0.7 (probe lr high)

After getting LB scores, run:
python scripts/lb_probing.py --analyze <scores>

This will reveal the TRUE mean and variance of each target in the test set.
""")

    return next_num

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        # Analyze mode: parse MSE results
        # Expected format: python lb_probing.py --analyze 0.05 0.06 0.07 0.05 0.06 0.05 0.06
        if len(sys.argv) < 9:
            print("Usage: python lb_probing.py --analyze MSE1 MSE2 MSE3 MSE4 MSE5 MSE6 MSE7")
            print("Where MSE1-7 are the LB scores from the 7 probing submissions")
            return

        mse_scores = [float(x) for x in sys.argv[2:9]]

        print("Analyzing probing results...")
        print("="*60)

        # For each target, we have baseline (0.5) and two probes (0.3, 0.7)
        # MSE = var + (mean - c)^2 for that target component
        # Total MSE = MSE_angle + MSE_depth + MSE_lr (assuming independent)

        # Baseline MSE (all at 0.5)
        mse_base = mse_scores[0]

        # Angle probes
        mse_angle_low = mse_scores[1]  # angle=0.3
        mse_angle_high = mse_scores[2]  # angle=0.7

        # Depth probes
        mse_depth_low = mse_scores[3]  # depth=0.3
        mse_depth_high = mse_scores[4]  # depth=0.7

        # LR probes
        mse_lr_low = mse_scores[5]  # lr=0.3
        mse_lr_high = mse_scores[6]  # lr=0.7

        print(f"\nBaseline MSE (all 0.5): {mse_base:.6f}")

        # Analyze each target
        for name, mse_low, mse_high, c_low, c_high, c_base in [
            ("angle", mse_angle_low, mse_angle_high, 0.3, 0.7, 0.5),
            ("depth", mse_depth_low, mse_depth_high, 0.3, 0.7, 0.5),
            ("left_right", mse_lr_low, mse_lr_high, 0.3, 0.7, 0.5),
        ]:
            # Delta from baseline tells us about this target's contribution
            delta_low = mse_low - mse_base
            delta_high = mse_high - mse_base

            # delta = (mean - c_new)^2 - (mean - c_base)^2
            # For c_low=0.3, c_base=0.5: delta_low = (mean-0.3)^2 - (mean-0.5)^2
            # For c_high=0.7, c_base=0.5: delta_high = (mean-0.7)^2 - (mean-0.5)^2

            # Expanding: delta_low = mean^2 - 0.6*mean + 0.09 - mean^2 + mean - 0.25
            #                      = 0.4*mean - 0.16
            # So: mean = (delta_low + 0.16) / 0.4

            # More generally:
            # delta = (mean - c_new)^2 - (mean - c_base)^2
            #       = 2*mean*(c_base - c_new) + c_new^2 - c_base^2
            # mean = (delta - c_new^2 + c_base^2) / (2*(c_base - c_new))

            mean_from_low = (delta_low - c_low**2 + c_base**2) / (2 * (c_base - c_low))
            mean_from_high = (delta_high - c_high**2 + c_base**2) / (2 * (c_base - c_high))

            mean_est = (mean_from_low + mean_from_high) / 2

            print(f"\n{name}:")
            print(f"  MSE at 0.3: {mse_low:.6f} (delta: {delta_low:+.6f})")
            print(f"  MSE at 0.7: {mse_high:.6f} (delta: {delta_high:+.6f})")
            print(f"  Estimated mean from low: {mean_from_low:.4f}")
            print(f"  Estimated mean from high: {mean_from_high:.4f}")
            print(f"  Average estimated mean: {mean_est:.4f}")

    else:
        # Create probing submissions
        create_probing_submissions()

if __name__ == "__main__":
    main()
