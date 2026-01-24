"""
Batch analysis and validation tool for ball tracking
Tests ball position calculation and release detection across multiple shots
"""

import pandas as pd
import numpy as np
import json
import ball_tracking as bt
from typing import List, Dict


def parse_array_json(s):
    """Convert string representation of array to numpy array"""
    s = s.replace('nan', 'null')
    return np.array(json.loads(s), dtype=np.float32)


def load_shot_data(csv_path, shot_index=0):
    """Load and parse a single shot from the dataset"""
    df = pd.read_csv(csv_path)

    # Get keypoint column names
    keypoint_names = df.columns[3:-3]

    # Parse arrays for the selected shot
    shot = df.iloc[shot_index].copy()
    for col in keypoint_names:
        shot[col] = parse_array_json(shot[col])

    return shot, keypoint_names


def analyze_shot(shot, shot_index: int) -> Dict:
    """
    Analyze a single shot and return metrics

    Args:
        shot: Shot data from DataFrame
        shot_index: Index of the shot

    Returns:
        Dictionary with analysis results
    """
    num_frames = len(shot['nose_x'])

    # Detect release
    release_frame, release_scores = bt.detect_release_frame(shot)

    # Calculate trajectory statistics
    trajectory = bt.get_ball_trajectory(shot, 0, num_frames - 1)
    valid_trajectory = trajectory[~np.isnan(trajectory).any(axis=1)]

    # Count valid ball positions
    num_valid_positions = len(valid_trajectory)

    # Calculate trajectory metrics (pre-release only)
    pre_release_traj = trajectory[:release_frame+1]
    pre_release_valid = pre_release_traj[~np.isnan(pre_release_traj).any(axis=1)]

    results = {
        'shot_index': shot_index,
        'shot_id': shot['shot_id'],
        'participant_id': shot['participant_id'],
        'num_frames': num_frames,
        'release_frame': release_frame,
        'release_time_s': release_frame / 60.0,
        'release_score': release_scores[release_frame],
        'num_valid_ball_positions': num_valid_positions,
        'ball_coverage_percent': (num_valid_positions / num_frames) * 100.0,
    }

    # Add trajectory statistics if sufficient data
    if len(pre_release_valid) >= 2:
        start_pos = pre_release_valid[0]
        end_pos = pre_release_valid[-1]

        results['distance_traveled_ft'] = np.linalg.norm(end_pos - start_pos)
        results['height_gain_ft'] = end_pos[2] - start_pos[2]
        results['max_height_ft'] = np.max(pre_release_valid[:, 2])
        results['min_height_ft'] = np.min(pre_release_valid[:, 2])
        results['height_range_ft'] = results['max_height_ft'] - results['min_height_ft']

        # Calculate average velocity
        if len(pre_release_valid) > 1:
            total_time = (len(pre_release_valid) - 1) / 60.0
            results['avg_velocity_ft_s'] = results['distance_traveled_ft'] / total_time if total_time > 0 else 0.0
        else:
            results['avg_velocity_ft_s'] = 0.0
    else:
        results['distance_traveled_ft'] = np.nan
        results['height_gain_ft'] = np.nan
        results['max_height_ft'] = np.nan
        results['min_height_ft'] = np.nan
        results['height_range_ft'] = np.nan
        results['avg_velocity_ft_s'] = np.nan

    # Check for anomalies
    anomalies = []
    if release_frame < 5:
        anomalies.append('very_early_release')
    if release_frame > num_frames - 10:
        anomalies.append('very_late_release')
    if num_valid_positions < num_frames * 0.5:
        anomalies.append('low_ball_coverage')
    if not np.isnan(results['height_gain_ft']) and results['height_gain_ft'] < -0.5:
        anomalies.append('large_downward_motion')
    if not np.isnan(results['height_gain_ft']) and results['height_gain_ft'] > 5.0:
        anomalies.append('excessive_upward_motion')

    results['anomalies'] = ', '.join(anomalies) if anomalies else 'none'

    return results


def analyze_multiple_shots(csv_path: str, num_shots: int = 5) -> pd.DataFrame:
    """
    Analyze multiple shots and return summary DataFrame

    Args:
        csv_path: Path to training CSV
        num_shots: Number of shots to analyze

    Returns:
        DataFrame with analysis results for all shots
    """
    results_list = []

    print(f"Analyzing {num_shots} shots...")
    print("="*80)

    for shot_index in range(num_shots):
        print(f"\nShot {shot_index + 1}/{num_shots}...")

        try:
            shot, _ = load_shot_data(csv_path, shot_index)
            results = analyze_shot(shot, shot_index)
            results_list.append(results)

            # Print summary
            print(f"  Shot ID: {results['shot_id']}")
            print(f"  Release Frame: {results['release_frame']} ({results['release_time_s']:.3f}s)")
            print(f"  Release Score: {results['release_score']:.3f}")
            print(f"  Ball Coverage: {results['ball_coverage_percent']:.1f}%")
            if not np.isnan(results['height_gain_ft']):
                print(f"  Height Gain: {results['height_gain_ft']:.3f} ft")
                print(f"  Avg Velocity: {results['avg_velocity_ft_s']:.3f} ft/s")
            if results['anomalies'] != 'none':
                print(f"  ANOMALIES: {results['anomalies']}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue

    print("\n" + "="*80)
    print("Analysis complete!")

    df_results = pd.DataFrame(results_list)
    return df_results


def print_summary_statistics(df_results: pd.DataFrame):
    """Print summary statistics across all analyzed shots"""

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal shots analyzed: {len(df_results)}")

    print("\nRelease Frame Statistics:")
    print(f"  Mean: {df_results['release_frame'].mean():.1f}")
    print(f"  Median: {df_results['release_frame'].median():.1f}")
    print(f"  Std Dev: {df_results['release_frame'].std():.1f}")
    print(f"  Min: {df_results['release_frame'].min()}")
    print(f"  Max: {df_results['release_frame'].max()}")

    print("\nRelease Time Statistics:")
    print(f"  Mean: {df_results['release_time_s'].mean():.3f}s")
    print(f"  Median: {df_results['release_time_s'].median():.3f}s")
    print(f"  Range: {df_results['release_time_s'].min():.3f}s - {df_results['release_time_s'].max():.3f}s")

    print("\nBall Coverage Statistics:")
    print(f"  Mean: {df_results['ball_coverage_percent'].mean():.1f}%")
    print(f"  Min: {df_results['ball_coverage_percent'].min():.1f}%")

    if 'height_gain_ft' in df_results.columns:
        valid_height = df_results['height_gain_ft'].dropna()
        if len(valid_height) > 0:
            print("\nHeight Gain Statistics:")
            print(f"  Mean: {valid_height.mean():.3f} ft")
            print(f"  Median: {valid_height.median():.3f} ft")
            print(f"  Range: {valid_height.min():.3f} ft - {valid_height.max():.3f} ft")

    # Count anomalies
    anomaly_counts = {}
    for anomalies_str in df_results['anomalies']:
        if anomalies_str != 'none':
            for anomaly in anomalies_str.split(', '):
                anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1

    if anomaly_counts:
        print("\nAnomaly Counts:")
        for anomaly, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            print(f"  {anomaly}: {count}")
    else:
        print("\nNo anomalies detected!")

    print("\n" + "="*80)


def main():
    csv_path = '../data/train.csv'

    # Analyze first 5 shots
    df_results = analyze_multiple_shots(csv_path, num_shots=5)

    # Print summary statistics
    print_summary_statistics(df_results)

    # Save results to CSV
    output_path = 'ball_analysis_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
