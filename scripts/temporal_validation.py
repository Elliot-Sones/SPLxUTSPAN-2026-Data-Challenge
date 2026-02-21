"""
Temporal Data Validation

Validate that competition data truly has 240 unique temporal frames.
Critical questions:
1. Are frames unique or duplicated?
2. Is motion smooth (continuous) or stepwise?
3. Do SPL and competition temporal patterns match?
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
DATA_DIR = Path("data")
EXTERNAL_DIR = Path("external_data/SPL-Open-Data/basketball/freethrow")
OUTPUT_DIR = Path("output/temporal_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_temporal_array(s):
    """Parse string-serialized array"""
    s_clean = s.replace('nan', 'null')
    arr = json.loads(s_clean)
    arr = [np.nan if x is None else x for x in arr]
    return np.array(arr, dtype=np.float32)


def analyze_frame_uniqueness(df, n_samples=10):
    """
    Analyze if 240 frames are unique or duplicated.
    """
    print("="*80)
    print("Frame Uniqueness Analysis")
    print("="*80)

    results = []

    for idx in range(min(n_samples, len(df))):
        row = df.iloc[idx]
        shot_id = row["shot_id"]

        # Parse temporal array for a key keypoint
        nose_x = parse_temporal_array(row["nose_x"])
        nose_y = parse_temporal_array(row["nose_y"])
        nose_z = parse_temporal_array(row["nose_z"])

        # Check for duplicates
        unique_frames = len(np.unique(nose_x[~np.isnan(nose_x)]))
        total_frames = len(nose_x)
        non_nan_frames = np.sum(~np.isnan(nose_x))

        # Compute frame-to-frame differences (velocity proxy)
        diffs = np.abs(np.diff(nose_x))
        non_zero_diffs = np.sum(diffs > 1e-6)

        # Compute temporal variance
        temporal_variance = np.nanvar(nose_x)

        results.append({
            "shot_id": shot_id,
            "total_frames": total_frames,
            "non_nan_frames": non_nan_frames,
            "unique_values": unique_frames,
            "non_zero_diffs": non_zero_diffs,
            "temporal_variance": temporal_variance,
            "is_static": unique_frames < 10,
        })

        print(f"\nShot {shot_id}:")
        print(f"  Total frames: {total_frames}")
        print(f"  Non-NaN frames: {non_nan_frames}")
        print(f"  Unique values: {unique_frames}")
        print(f"  Non-zero differences: {non_zero_diffs} / {total_frames-1}")
        print(f"  Temporal variance: {temporal_variance:.6f}")
        print(f"  Assessment: {'STATIC/DUPLICATED' if unique_frames < 10 else 'DYNAMIC'}")

    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total shots analyzed: {len(results_df)}")
    print(f"Static/duplicated shots: {results_df['is_static'].sum()}")
    print(f"Dynamic shots: {(~results_df['is_static']).sum()}")
    print(f"\nAverage unique frames: {results_df['unique_values'].mean():.1f} / 240")
    print(f"Average non-zero diffs: {results_df['non_zero_diffs'].mean():.1f} / 239")
    print(f"Average temporal variance: {results_df['temporal_variance'].mean():.6f}")

    if results_df['is_static'].mean() > 0.5:
        print("\n⚠️  WARNING: Majority of shots appear to be static/duplicated!")
        print("   Temporal data may be single-frame replicated.")
    else:
        print("\n✓ CONFIRMED: Temporal data is dynamic (true motion sequences)")

    return results_df


def visualize_temporal_patterns(df, n_samples=3):
    """
    Visualize temporal patterns for sample shots.
    """
    print("\n" + "="*80)
    print("Temporal Pattern Visualization")
    print("="*80)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(min(n_samples, len(df))):
        row = df.iloc[idx]
        shot_id = row["shot_id"]

        # Parse temporal arrays
        nose_x = parse_temporal_array(row["nose_x"])
        nose_y = parse_temporal_array(row["nose_y"])
        wrist_r_z = parse_temporal_array(row["right_wrist_z"])

        # Plot X coordinate
        axes[idx, 0].plot(nose_x, label="nose_x", alpha=0.7)
        axes[idx, 0].set_title(f"Shot {shot_id} - X Position")
        axes[idx, 0].set_xlabel("Frame")
        axes[idx, 0].set_ylabel("Position (feet)")
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        # Plot Y coordinate
        axes[idx, 1].plot(nose_y, label="nose_y", alpha=0.7)
        axes[idx, 1].set_title(f"Shot {shot_id} - Y Position")
        axes[idx, 1].set_xlabel("Frame")
        axes[idx, 1].set_ylabel("Position (feet)")
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)

        # Plot wrist height (z)
        axes[idx, 2].plot(wrist_r_z, label="right_wrist_z", color="red", alpha=0.7)
        axes[idx, 2].set_title(f"Shot {shot_id} - Wrist Height")
        axes[idx, 2].set_xlabel("Frame")
        axes[idx, 2].set_ylabel("Height (feet)")
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "temporal_patterns.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved visualization to {plot_path}")


def compute_velocity_features(df, n_samples=10):
    """
    Compute velocity and acceleration features from temporal data.
    """
    print("\n" + "="*80)
    print("Velocity Feature Analysis")
    print("="*80)

    velocity_stats = []

    for idx in range(min(n_samples, len(df))):
        row = df.iloc[idx]
        shot_id = row["shot_id"]

        # Parse wrist trajectory (key for release)
        wrist_x = parse_temporal_array(row["right_wrist_x"])
        wrist_y = parse_temporal_array(row["right_wrist_y"])
        wrist_z = parse_temporal_array(row["right_wrist_z"])

        # Compute 3D velocity magnitude
        dx = np.diff(wrist_x)
        dy = np.diff(wrist_y)
        dz = np.diff(wrist_z)
        velocity_mag = np.sqrt(dx**2 + dy**2 + dz**2)

        # Remove NaN
        velocity_mag_clean = velocity_mag[~np.isnan(velocity_mag)]

        if len(velocity_mag_clean) > 0:
            max_vel = np.max(velocity_mag_clean)
            mean_vel = np.mean(velocity_mag_clean)
            std_vel = np.std(velocity_mag_clean)
            peak_frame = np.argmax(velocity_mag_clean)
        else:
            max_vel = mean_vel = std_vel = peak_frame = 0

        velocity_stats.append({
            "shot_id": shot_id,
            "max_velocity": max_vel,
            "mean_velocity": mean_vel,
            "std_velocity": std_vel,
            "peak_frame": peak_frame,
        })

        print(f"\nShot {shot_id}:")
        print(f"  Max velocity: {max_vel:.3f} ft/frame")
        print(f"  Mean velocity: {mean_vel:.3f} ft/frame")
        print(f"  Std velocity: {std_vel:.3f} ft/frame")
        print(f"  Peak frame: {peak_frame}")

    velocity_df = pd.DataFrame(velocity_stats)

    print("\n" + "="*80)
    print("Velocity Summary")
    print("="*80)
    print(f"Average max velocity: {velocity_df['max_velocity'].mean():.3f} ft/frame")
    print(f"Average peak frame: {velocity_df['peak_frame'].mean():.1f}")

    return velocity_df


def compare_spl_vs_competition():
    """
    Compare temporal patterns between SPL and competition data.
    """
    print("\n" + "="*80)
    print("SPL vs Competition Temporal Comparison")
    print("="*80)

    # Load one SPL shot
    spl_file = EXTERNAL_DIR / "data/P0001/BB_FT_P0001_T0001.json"
    with open(spl_file) as f:
        spl_shot = json.load(f)

    # Extract SPL wrist trajectory
    spl_wrist_z = []
    for frame in spl_shot["tracking"]:
        player = frame["data"]["player"]
        if "R_WRIST" in player:
            spl_wrist_z.append(player["R_WRIST"][2])
        else:
            spl_wrist_z.append(np.nan)
    spl_wrist_z = np.array(spl_wrist_z, dtype=np.float32)

    # Load competition shot
    comp_df = pd.read_csv(DATA_DIR / "train.csv")
    comp_wrist_z = parse_temporal_array(comp_df.iloc[0]["right_wrist_z"])

    # Compare
    print(f"SPL shot:")
    print(f"  Frames: {len(spl_wrist_z)}")
    print(f"  Wrist height range: [{np.nanmin(spl_wrist_z):.2f}, {np.nanmax(spl_wrist_z):.2f}]")
    print(f"  Wrist height variance: {np.nanvar(spl_wrist_z):.4f}")

    print(f"\nCompetition shot:")
    print(f"  Frames: {len(comp_wrist_z)}")
    print(f"  Wrist height range: [{np.nanmin(comp_wrist_z):.2f}, {np.nanmax(comp_wrist_z):.2f}]")
    print(f"  Wrist height variance: {np.nanvar(comp_wrist_z):.4f}")

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(spl_wrist_z, label="SPL wrist height", alpha=0.7)
    axes[0].set_title("SPL Temporal Pattern")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Wrist Height (feet)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(comp_wrist_z, label="Competition wrist height", color="orange", alpha=0.7)
    axes[1].set_title("Competition Temporal Pattern")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Wrist Height (feet)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = OUTPUT_DIR / "spl_vs_competition.png"
    plt.savefig(comparison_path, dpi=150)
    print(f"\nSaved comparison to {comparison_path}")


def main():
    print("="*80)
    print("TEMPORAL DATA VALIDATION")
    print("="*80)

    # Load competition data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Loaded {len(train_df)} training shots")

    # 1. Analyze frame uniqueness
    uniqueness_results = analyze_frame_uniqueness(train_df, n_samples=20)
    uniqueness_results.to_csv(OUTPUT_DIR / "frame_uniqueness.csv", index=False)

    # 2. Visualize temporal patterns
    visualize_temporal_patterns(train_df, n_samples=3)

    # 3. Compute velocity features
    velocity_results = compute_velocity_features(train_df, n_samples=20)
    velocity_results.to_csv(OUTPUT_DIR / "velocity_features.csv", index=False)

    # 4. Compare SPL vs competition
    compare_spl_vs_competition()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Results saved to {OUTPUT_DIR}/")
    print("\nKey Outputs:")
    print(f"  - frame_uniqueness.csv: Frame duplication analysis")
    print(f"  - velocity_features.csv: Velocity statistics")
    print(f"  - temporal_patterns.png: Sample trajectories")
    print(f"  - spl_vs_competition.png: SPL vs competition comparison")


if __name__ == "__main__":
    main()
