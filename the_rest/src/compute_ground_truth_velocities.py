"""
Compute ground truth velocities for all training shots using inverse ballistics.

This script:
1. Loads all training shots
2. Extracts release positions and outcomes
3. Solves inverse ballistics for each shot
4. Saves results to output/ground_truth_velocities.csv
5. Analyzes convergence and compares to observed velocities
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_metadata, iterate_shots, get_keypoint_columns
from src.physics_features import init_keypoint_mapping, extract_physics_features
from src.inverse_ballistics import solve_inverse_ballistics, calibrate_hoop_position


def compute_all_ground_truth_velocities(
    max_shots: int = None,
    use_global_optimization: bool = False,
    calibrate_hoop: bool = True
):
    """
    Compute ground truth velocities for all training shots.

    Args:
        max_shots: Maximum number of shots to process (None for all)
        use_global_optimization: Use global optimization (slower but more robust)
        calibrate_hoop: Whether to calibrate hoop position from data
    """
    print("Computing ground truth velocities...")

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Load metadata
    meta_df = load_metadata(train=True)
    n_shots = len(meta_df) if max_shots is None else min(max_shots, len(meta_df))

    print(f"Processing {n_shots} shots...")

    # First pass: collect release positions for hoop calibration
    if calibrate_hoop:
        print("Calibrating hoop position...")
        release_positions = []
        depths = []
        left_rights = []

        for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
            if i >= n_shots:
                break

            try:
                feats = extract_physics_features(timeseries, smooth=True)
                release_pos = np.array([
                    feats['wrist_x_release'],
                    feats['wrist_y_release'],
                    feats['wrist_z_release']
                ])

                if not np.any(np.isnan(release_pos)):
                    release_positions.append(release_pos)
                    depths.append(metadata['depth'])
                    left_rights.append(metadata['left_right'])

            except Exception as e:
                continue

        release_positions = np.array(release_positions)
        depths = np.array(depths)
        left_rights = np.array(left_rights)

        # Calibrate hoop position
        hoop_pos = calibrate_hoop_position(
            release_positions,
            depths,
            left_rights,
            release_height_mean=np.mean(release_positions[:, 2])
        )

        print(f"  Calibrated hoop position: {hoop_pos}")
    else:
        # Use default hoop position
        hoop_pos = np.array([18.5, -21.0, 0.0])  # Rough guess from data
        print(f"  Using default hoop position: {hoop_pos}")

    # Estimate gravity constant from coordinate system
    # If typical release height is ~5 units and velocity is ~7 units/s,
    # we can estimate g from the scale
    # For now, use g=9.81 (will adjust if needed)
    g = 9.81

    # Second pass: solve inverse ballistics for each shot
    results = []
    failed_count = 0

    print("Computing velocities for each shot...")
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= n_shots:
            break

        # Progress indicator
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        try:
            # Extract release features
            feats = extract_physics_features(timeseries, smooth=True)

            release_pos = np.array([
                feats['wrist_x_release'],
                feats['wrist_y_release'],
                feats['wrist_z_release']
            ])

            observed_vel = np.array([
                feats['wrist_vx_release'],
                feats['wrist_vy_release'],
                feats['wrist_vz_release']
            ])

            # Target outcomes
            target_angle = metadata['angle']
            target_depth = metadata['depth']
            target_left_right = metadata['left_right']

            # Check for NaN values
            if np.any(np.isnan(release_pos)) or np.isnan(target_angle):
                failed_count += 1
                continue

            # Solve inverse ballistics
            method = "global" if use_global_optimization else "local"
            inverse_result = solve_inverse_ballistics(
                release_pos,
                target_angle,
                target_depth,
                target_left_right,
                hoop_pos,
                g=g,
                initial_guess=observed_vel,
                method=method
            )

            # Store results
            result = {
                # Metadata
                "id": metadata["id"],
                "shot_id": metadata["shot_id"],
                "participant_id": metadata["participant_id"],

                # Ground truth targets
                "angle": target_angle,
                "depth": target_depth,
                "left_right": target_left_right,

                # Release position
                "release_x": release_pos[0],
                "release_y": release_pos[1],
                "release_z": release_pos[2],

                # Observed velocity (from tracking)
                "observed_vx": observed_vel[0],
                "observed_vy": observed_vel[1],
                "observed_vz": observed_vel[2],
                "observed_vel_mag": np.linalg.norm(observed_vel),

                # Solved velocity (ground truth)
                "gt_vx": inverse_result['vx'],
                "gt_vy": inverse_result['vy'],
                "gt_vz": inverse_result['vz'],
                "gt_vel_mag": inverse_result['vel_magnitude'],

                # Optimization metrics
                "convergence_error": inverse_result['error'],
                "convergence_success": inverse_result['success'],

                # Predicted outcomes (should match targets if convergence is good)
                "predicted_angle": inverse_result['predicted_angle'],
                "predicted_depth": inverse_result['predicted_depth'],
                "predicted_left_right": inverse_result['predicted_left_right'],

                # Physics
                "time_of_flight": inverse_result['time_of_flight'],

                # Feature for analysis
                "velocity_difference": np.linalg.norm(inverse_result['velocity'] - observed_vel),
            }

            results.append(result)

        except Exception as e:
            failed_count += 1
            print(f"\nError processing shot {i}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print(f"\nSuccessfully processed: {len(df)}/{n_shots} shots")
    print(f"Failed: {failed_count} shots")

    if len(df) == 0:
        print("No valid results, exiting")
        return None

    # Analysis
    print("\n=== Convergence Analysis ===")
    converged = df['convergence_success'].sum()
    print(f"Converged: {converged}/{len(df)} ({100*converged/len(df):.1f}%)")

    # Low error threshold
    low_error = (df['convergence_error'] < 0.1).sum()
    print(f"Low error (<0.1): {low_error}/{len(df)} ({100*low_error/len(df):.1f}%)")

    print(f"\nError statistics:")
    print(f"  Mean: {df['convergence_error'].mean():.6f}")
    print(f"  Median: {df['convergence_error'].median():.6f}")
    print(f"  Max: {df['convergence_error'].max():.6f}")
    print(f"  Std: {df['convergence_error'].std():.6f}")

    print("\n=== Velocity Analysis ===")
    print("Ground truth velocity magnitude:")
    print(f"  Mean: {df['gt_vel_mag'].mean():.2f}")
    print(f"  Std: {df['gt_vel_mag'].std():.2f}")
    print(f"  Min: {df['gt_vel_mag'].min():.2f}")
    print(f"  Max: {df['gt_vel_mag'].max():.2f}")

    print("\nObserved velocity magnitude:")
    print(f"  Mean: {df['observed_vel_mag'].mean():.2f}")
    print(f"  Std: {df['observed_vel_mag'].std():.2f}")

    print("\nVelocity difference (GT vs Observed):")
    print(f"  Mean: {df['velocity_difference'].mean():.2f}")
    print(f"  Median: {df['velocity_difference'].median():.2f}")

    # Correlation between observed and GT velocity
    corr_vx = df[['observed_vx', 'gt_vx']].corr().iloc[0, 1]
    corr_vy = df[['observed_vy', 'gt_vy']].corr().iloc[0, 1]
    corr_vz = df[['observed_vz', 'gt_vz']].corr().iloc[0, 1]

    print("\nCorrelation (Observed vs GT):")
    print(f"  vx: {corr_vx:.3f}")
    print(f"  vy: {corr_vy:.3f}")
    print(f"  vz: {corr_vz:.3f}")

    print("\n=== Prediction Quality ===")
    # Check how well the solved velocity reproduces targets
    angle_mae = np.abs(df['predicted_angle'] - df['angle']).mean()
    depth_mae = np.abs(df['predicted_depth'] - df['depth']).mean()
    left_right_mae = np.abs(df['predicted_left_right'] - df['left_right']).mean()

    print(f"Prediction error (solved velocity should perfectly match targets):")
    print(f"  Angle MAE: {angle_mae:.4f}°")
    print(f"  Depth MAE: {depth_mae:.4f}")
    print(f"  Left/Right MAE: {left_right_mae:.4f}")

    # Save results
    output_path = Path(__file__).parent.parent / "output" / "ground_truth_velocities.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

    # Also save hoop position for reference
    hoop_info = pd.DataFrame({
        'hoop_x': [hoop_pos[0]],
        'hoop_y': [hoop_pos[1]],
        'hoop_z': [hoop_pos[2]],
        'gravity': [g]
    })
    hoop_path = Path(__file__).parent.parent / "output" / "calibrated_hoop_position.csv"
    hoop_info.to_csv(hoop_path, index=False)
    print(f"✓ Saved hoop calibration to {hoop_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shots", type=int, default=None, help="Maximum shots to process")
    parser.add_argument("--global-opt", action="store_true", help="Use global optimization")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip hoop calibration")
    parser.add_argument("--quick", action="store_true", help="Quick test on 10 shots")

    args = parser.parse_args()

    if args.quick:
        max_shots = 10
    else:
        max_shots = args.max_shots

    df = compute_all_ground_truth_velocities(
        max_shots=max_shots,
        use_global_optimization=args.global_opt,
        calibrate_hoop=not args.no_calibrate
    )
