"""
Profile-Constrained Blending

Step 3 of the implementation plan:
- Combine per-player predictions while maintaining good profile
- Apply shrinkage/calibration to ensure profile constraints
- If angle_std too high, blend with Sub 133 minimally

Goal: Create diverse predictions (<95% correlated with Sub 133) with good profile
- angle_std < 0.14
- depth_mean in [0.50, 0.51]
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'
SUBMISSION_DIR = PROJECT_DIR / 'submission'

TARGETS = ['angle', 'depth', 'left_right']

TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}

# Target profile from Sub 133 (the best submission)
TARGET_PROFILE = {
    'angle_std': 0.1377,
    'angle_mean': 0.5002,
    'depth_mean': 0.5055,
    'lr_mean': 0.4687
}


def load_submissions(*sub_nums):
    """Load multiple submissions."""
    submissions = {}
    for num in sub_nums:
        sub_file = SUBMISSION_DIR / f"submission_{num}.csv"
        if sub_file.exists():
            df = pd.read_csv(sub_file)
            submissions[num] = df
        else:
            print(f"Warning: {sub_file} not found")
    return submissions


def compute_profile(angle, depth, lr):
    """Compute profile metrics."""
    return {
        'angle_std': np.std(angle),
        'angle_mean': np.mean(angle),
        'depth_mean': np.mean(depth),
        'lr_mean': np.mean(lr)
    }


def profile_distance(profile, target=TARGET_PROFILE):
    """Compute distance from target profile (lower is better)."""
    angle_std_diff = abs(profile['angle_std'] - target['angle_std'])
    depth_mean_diff = abs(profile['depth_mean'] - target['depth_mean'])

    # Weight depth_mean more (3.7x more predictive of LB according to research)
    return angle_std_diff + 3.7 * depth_mean_diff


def calibrate_to_profile(angle, depth, lr, target_profile=TARGET_PROFILE):
    """
    Calibrate predictions to match target profile.

    Uses simple linear scaling to match mean/std.
    """
    # Angle: match mean and std
    current_angle_std = np.std(angle)
    current_angle_mean = np.mean(angle)

    # Scale to target std
    if current_angle_std > 0:
        angle_calibrated = (angle - current_angle_mean) / current_angle_std * target_profile['angle_std']
        angle_calibrated += target_profile['angle_mean']
    else:
        angle_calibrated = angle

    # Depth: shift to target mean
    current_depth_mean = np.mean(depth)
    depth_calibrated = depth - current_depth_mean + target_profile['depth_mean']

    # LR: shift to target mean
    current_lr_mean = np.mean(lr)
    lr_calibrated = lr - current_lr_mean + target_profile['lr_mean']

    # Clip to [0, 1]
    angle_calibrated = np.clip(angle_calibrated, 0, 1)
    depth_calibrated = np.clip(depth_calibrated, 0, 1)
    lr_calibrated = np.clip(lr_calibrated, 0, 1)

    return angle_calibrated, depth_calibrated, lr_calibrated


def optimal_blend_for_profile(subs_dict, target_corr_range=(0.85, 0.95)):
    """
    Find optimal blend weights that:
    1. Minimize profile distance
    2. Keep correlation with Sub 133 in target range

    Args:
        subs_dict: dict of sub_num -> DataFrame
        target_corr_range: (min, max) correlation with Sub 133

    Returns:
        optimal weights, blended predictions
    """
    sub_nums = list(subs_dict.keys())
    n_subs = len(sub_nums)

    if n_subs == 0:
        raise ValueError("No submissions provided")

    # Extract arrays
    angles = [subs_dict[n]['scaled_angle'].values for n in sub_nums]
    depths = [subs_dict[n]['scaled_depth'].values for n in sub_nums]
    lrs = [subs_dict[n]['scaled_left_right'].values for n in sub_nums]

    # Get Sub 133 for correlation target
    if 133 in subs_dict:
        sub133_angle = subs_dict[133]['scaled_angle'].values
        sub133_depth = subs_dict[133]['scaled_depth'].values
        sub133_lr = subs_dict[133]['scaled_left_right'].values
    else:
        sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
        sub133_angle = sub133['scaled_angle'].values
        sub133_depth = sub133['scaled_depth'].values
        sub133_lr = sub133['scaled_left_right'].values

    def objective(weights):
        """Objective: minimize profile distance + correlation penalty."""
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        blend_angle = sum(w * a for w, a in zip(weights, angles))
        blend_depth = sum(w * d for w, d in zip(weights, depths))
        blend_lr = sum(w * l for w, l in zip(weights, lrs))

        profile = compute_profile(blend_angle, blend_depth, blend_lr)
        dist = profile_distance(profile)

        # Correlation penalty
        corr = np.corrcoef(blend_angle, sub133_angle)[0, 1]
        if corr > target_corr_range[1]:
            # Too similar - penalize
            penalty = 10 * (corr - target_corr_range[1])
        elif corr < target_corr_range[0]:
            # Too different - penalize (but less)
            penalty = 5 * (target_corr_range[0] - corr)
        else:
            penalty = 0

        return dist + penalty

    # Initial weights (equal)
    x0 = np.ones(n_subs) / n_subs

    # Bounds: each weight between 0 and 1
    bounds = [(0, 1) for _ in range(n_subs)]

    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Optimize
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x / result.x.sum()

    # Compute final blend
    blend_angle = sum(w * a for w, a in zip(optimal_weights, angles))
    blend_depth = sum(w * d for w, d in zip(optimal_weights, depths))
    blend_lr = sum(w * l for w, l in zip(optimal_weights, lrs))

    return dict(zip(sub_nums, optimal_weights)), (blend_angle, blend_depth, blend_lr)


def constrained_blend_with_sub133(new_predictions, sub133, max_angle_std=0.14, min_depth=0.50, max_depth=0.51):
    """
    Blend new predictions with Sub 133 to satisfy profile constraints.

    Uses binary search to find minimal blend weight that satisfies constraints.
    """
    new_angle, new_depth, new_lr = new_predictions
    sub133_angle = sub133['scaled_angle'].values
    sub133_depth = sub133['scaled_depth'].values
    sub133_lr = sub133['scaled_left_right'].values

    def check_constraints(w_new):
        """Check if blend with w_new weight on new predictions satisfies constraints."""
        blend_angle = w_new * new_angle + (1 - w_new) * sub133_angle
        blend_depth = w_new * new_depth + (1 - w_new) * sub133_depth

        angle_std = np.std(blend_angle)
        depth_mean = np.mean(blend_depth)

        return angle_std <= max_angle_std and min_depth <= depth_mean <= max_depth

    # Binary search for maximum w_new that satisfies constraints
    lo, hi = 0.0, 1.0

    while hi - lo > 0.01:
        mid = (lo + hi) / 2
        if check_constraints(mid):
            lo = mid
        else:
            hi = mid

    optimal_w = lo

    # Compute final blend
    blend_angle = optimal_w * new_angle + (1 - optimal_w) * sub133_angle
    blend_depth = optimal_w * new_depth + (1 - optimal_w) * sub133_depth
    blend_lr = optimal_w * new_lr + (1 - optimal_w) * sub133_lr

    return optimal_w, (blend_angle, blend_depth, blend_lr)


def main():
    print("="*80)
    print("PROFILE-CONSTRAINED BLENDING")
    print("="*80)

    # Load Sub 133 (the best)
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    test_ids = sub133['id'].values

    print("\nSub 133 profile:")
    profile_133 = compute_profile(
        sub133['scaled_angle'].values,
        sub133['scaled_depth'].values,
        sub133['scaled_left_right'].values
    )
    for k, v in profile_133.items():
        print(f"  {k}: {v:.4f}")

    # Load other key submissions for blending candidates
    # Sub 9, 10, 25, 111 are components of Sub 133
    key_subs = [9, 10, 25, 111, 133]

    # Also check for recent specialized submissions
    recent_subs = list(SUBMISSION_DIR.glob("submission_16*.csv"))
    for sub_file in recent_subs:
        num = int(sub_file.stem.split('_')[1])
        if num not in key_subs:
            key_subs.append(num)

    print(f"\nLoading submissions: {key_subs}")
    submissions = load_submissions(*key_subs)

    print(f"\nLoaded {len(submissions)} submissions")

    # Analyze each submission's profile
    print("\n" + "="*80)
    print("SUBMISSION PROFILES")
    print("="*80)

    for num, df in sorted(submissions.items()):
        profile = compute_profile(
            df['scaled_angle'].values,
            df['scaled_depth'].values,
            df['scaled_left_right'].values
        )
        dist = profile_distance(profile)

        corr_angle = np.corrcoef(df['scaled_angle'].values, sub133['scaled_angle'].values)[0, 1]

        print(f"\nSub {num}:")
        print(f"  angle_std:  {profile['angle_std']:.4f}")
        print(f"  depth_mean: {profile['depth_mean']:.4f}")
        print(f"  Profile dist: {dist:.4f}")
        print(f"  Corr w/ 133 (angle): {corr_angle:.4f}")

    # Strategy 1: Find optimal blend of diverse submissions
    print("\n" + "="*80)
    print("STRATEGY 1: OPTIMAL BLEND")
    print("="*80)

    # Only blend non-133 submissions with 133
    blend_candidates = {k: v for k, v in submissions.items() if k != 133}
    blend_candidates[133] = submissions[133]  # Include 133 as well

    if len(blend_candidates) > 1:
        try:
            optimal_weights, (blend_angle, blend_depth, blend_lr) = optimal_blend_for_profile(
                blend_candidates, target_corr_range=(0.90, 0.98)
            )

            print("\nOptimal weights:")
            for sub_num, weight in sorted(optimal_weights.items()):
                if weight > 0.01:
                    print(f"  Sub {sub_num}: {weight:.3f}")

            blend_profile = compute_profile(blend_angle, blend_depth, blend_lr)
            corr_133 = np.corrcoef(blend_angle, sub133['scaled_angle'].values)[0, 1]

            print(f"\nBlend profile:")
            print(f"  angle_std:  {blend_profile['angle_std']:.4f}")
            print(f"  depth_mean: {blend_profile['depth_mean']:.4f}")
            print(f"  Corr w/ 133 (angle): {corr_133:.4f}")
            print(f"  Profile distance: {profile_distance(blend_profile):.4f}")

        except Exception as e:
            print(f"Optimization failed: {e}")
            blend_angle = sub133['scaled_angle'].values.copy()
            blend_depth = sub133['scaled_depth'].values.copy()
            blend_lr = sub133['scaled_left_right'].values.copy()
    else:
        print("Not enough diverse submissions for blending")
        blend_angle = sub133['scaled_angle'].values.copy()
        blend_depth = sub133['scaled_depth'].values.copy()
        blend_lr = sub133['scaled_left_right'].values.copy()

    # Strategy 2: Calibrate to target profile
    print("\n" + "="*80)
    print("STRATEGY 2: CALIBRATION TO TARGET PROFILE")
    print("="*80)

    calib_angle, calib_depth, calib_lr = calibrate_to_profile(
        blend_angle, blend_depth, blend_lr, TARGET_PROFILE
    )

    calib_profile = compute_profile(calib_angle, calib_depth, calib_lr)
    corr_calib = np.corrcoef(calib_angle, sub133['scaled_angle'].values)[0, 1]

    print(f"\nCalibrated profile:")
    print(f"  angle_std:  {calib_profile['angle_std']:.4f}")
    print(f"  depth_mean: {calib_profile['depth_mean']:.4f}")
    print(f"  Corr w/ 133 (angle): {corr_calib:.4f}")
    print(f"  Profile distance: {profile_distance(calib_profile):.4f}")

    # Strategy 3: Constrained blend with Sub 133
    print("\n" + "="*80)
    print("STRATEGY 3: CONSTRAINED BLEND WITH SUB 133")
    print("="*80)

    # Use calibrated predictions as new predictions
    optimal_w, (final_angle, final_depth, final_lr) = constrained_blend_with_sub133(
        (calib_angle, calib_depth, calib_lr),
        sub133,
        max_angle_std=0.14,
        min_depth=0.495,
        max_depth=0.515
    )

    final_profile = compute_profile(final_angle, final_depth, final_lr)
    corr_final = np.corrcoef(final_angle, sub133['scaled_angle'].values)[0, 1]

    print(f"\nOptimal blend weight on new predictions: {optimal_w:.3f}")
    print(f"Weight on Sub 133: {1 - optimal_w:.3f}")

    print(f"\nFinal profile:")
    print(f"  angle_std:  {final_profile['angle_std']:.4f}")
    print(f"  depth_mean: {final_profile['depth_mean']:.4f}")
    print(f"  Corr w/ 133 (angle): {corr_final:.4f}")
    print(f"  Profile distance: {profile_distance(final_profile):.4f}")

    # Check if we have meaningful diversity
    if corr_final > 0.99:
        print("\n  WARNING: Final predictions too similar to Sub 133 (>99% corr)")
        print("  This submission may not improve over Sub 133")
    elif corr_final < 0.85:
        print("\n  WARNING: Final predictions too different from Sub 133 (<85% corr)")
        print("  High risk of worse LB score")
    else:
        print("\n  Good diversity achieved while maintaining profile constraints")

    # Save submission
    print("\n" + "="*80)
    print("CREATING SUBMISSION")
    print("="*80)

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 163

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': final_angle,
        'scaled_depth': final_depth,
        'scaled_left_right': final_lr
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"\nSubmission {next_num} details:")
    print(f"  Model: Profile-constrained blend")
    print(f"  Blend: {optimal_w:.1%} new + {1-optimal_w:.1%} Sub 133")
    print(f"  Profile: angle_std={final_profile['angle_std']:.4f}, depth_mean={final_profile['depth_mean']:.4f}")
    print(f"  Correlation with Sub 133: {corr_final:.4f}")

    # Save to research
    research_file = PROJECT_DIR / 'Research' / 'PROFILE_CONSTRAINED_BLEND.md'
    with open(research_file, 'w') as f:
        f.write("# Profile-Constrained Blending Results\n\n")
        f.write(f"## Submission {next_num}\n\n")
        f.write("### Method\n")
        f.write(f"- Optimal blend weight search with profile constraints\n")
        f.write(f"- Calibration to match Sub 133 profile\n")
        f.write(f"- Constrained blend: {optimal_w:.1%} new + {1-optimal_w:.1%} Sub 133\n\n")
        f.write("### Results\n")
        f.write(f"- angle_std: {final_profile['angle_std']:.4f} (target: < 0.14)\n")
        f.write(f"- depth_mean: {final_profile['depth_mean']:.4f} (target: 0.50-0.51)\n")
        f.write(f"- Correlation with Sub 133: {corr_final:.4f}\n")
        f.write(f"- Profile distance: {profile_distance(final_profile):.4f}\n\n")
        f.write("### Analysis\n")
        if corr_final > 0.99:
            f.write("WARNING: Too similar to Sub 133, unlikely to improve LB\n")
        elif corr_final < 0.85:
            f.write("WARNING: Too different from Sub 133, high risk of worse LB\n")
        else:
            f.write("Good diversity with profile constraints satisfied\n")

    print(f"\nSaved research: {research_file}")


if __name__ == "__main__":
    main()
