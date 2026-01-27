#!/usr/bin/env python3
"""
Build Release Prediction Dataset from SkillMimic Data

Process multiple SkillMimic shots to create a dataset for training:
    Input: Hand kinematics before/at release
    Output: Ball velocity at release

This dataset will be used to train a model that predicts ball release
parameters from hand movements alone (for SPL data which has no ball tracking).
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import json


def load_joint_mapping(map_path: str = "visualisation/skillmimic_joint_map.json") -> dict:
    with open(map_path, 'r') as f:
        return json.load(f)['joint_map']


def detect_release_frame(contact_flag: np.ndarray, hand_height: np.ndarray) -> int:
    """
    Detect release frame using multiple methods.

    Method 1 (if available): Contact flag transition
    Method 2 (fallback): Peak hand height
    """
    # Method 1: Contact flag transition
    for i in range(len(contact_flag) - 1):
        if contact_flag[i] == 1 and contact_flag[i+1] == 0:
            return i + 1

    # Method 2: Peak hand height (for SPL data without contact flag)
    release_candidates = []
    for i in range(5, len(hand_height) - 5):
        # Look for local maximum
        if hand_height[i] > hand_height[i-1] and hand_height[i] > hand_height[i+1]:
            # Verify it's a significant peak
            if hand_height[i] > np.percentile(hand_height, 75):
                release_candidates.append(i)

    if release_candidates:
        return release_candidates[0]  # First peak

    # Fallback: Frame with max height
    return int(np.argmax(hand_height))


def calculate_velocity(positions: np.ndarray, window: int = 1) -> np.ndarray:
    """Calculate velocity using finite differences"""
    vel = np.zeros_like(positions)
    vel[window:-window] = (positions[2*window:] - positions[:-2*window]) / (2 * window)
    vel[:window] = positions[window] - positions[0]
    vel[-window:] = positions[-1] - positions[-window-1]
    return vel


def extract_features_for_modeling(filepath: str, pre_release_window: int = 20) -> dict:
    """
    Extract features from one SkillMimic shot.

    Args:
        filepath: Path to .pt file
        pre_release_window: Frames before release to include

    Returns:
        Dictionary with input features (hand kinematics) and target (ball velocity)
    """
    # Load data
    data = torch.load(filepath, map_location='cpu')
    joint_map = load_joint_mapping()

    # Extract positions
    ball_position = data[:, 318:321].numpy()
    ball_velocity = data[:, 325:328].numpy()
    contact_flag = data[:, 336].numpy()

    # Hand positions
    l_wrist = joint_map['L_Wrist']
    r_wrist = joint_map['R_Wrist']
    l_elbow = joint_map['L_Elbow']
    r_elbow = joint_map['R_Elbow']
    l_shoulder = joint_map['L_Shoulder']
    r_shoulder = joint_map['R_Shoulder']

    left_hand = data[:, l_wrist].numpy()
    right_hand = data[:, r_wrist].numpy()
    left_elbow = data[:, l_elbow].numpy()
    right_elbow = data[:, r_elbow].numpy()
    left_shoulder = data[:, l_shoulder].numpy()
    right_shoulder = data[:, r_shoulder].numpy()

    avg_hand = (left_hand + right_hand) / 2

    # Detect release
    release_frame = detect_release_frame(contact_flag, avg_hand[:, 1])

    # Calculate kinematics
    hand_velocity = calculate_velocity(avg_hand)
    hand_speed = np.linalg.norm(hand_velocity, axis=1)

    left_hand_vel = calculate_velocity(left_hand)
    right_hand_vel = calculate_velocity(right_hand)

    # Pre-release window
    start_frame = max(0, release_frame - pre_release_window)
    pre_release_indices = range(start_frame, release_frame)

    # --- INPUT FEATURES (what SPL data will have) ---

    # 1. Release timing
    features = {
        'shot_id': Path(filepath).stem,
        'release_frame': release_frame,
        'total_frames': len(data),
        'release_frame_pct': release_frame / len(data),
    }

    # 2. Hand position at release
    features.update({
        'hand_x_rel': avg_hand[release_frame, 0],
        'hand_y_rel': avg_hand[release_frame, 1],
        'hand_z_rel': avg_hand[release_frame, 2],
        'hand_height_rel': avg_hand[release_frame, 1],
    })

    # 3. Hand velocity at release (nearly zero, but include)
    features.update({
        'hand_vx_rel': hand_velocity[release_frame, 0],
        'hand_vy_rel': hand_velocity[release_frame, 1],
        'hand_vz_rel': hand_velocity[release_frame, 2],
        'hand_speed_rel': hand_speed[release_frame],
    })

    # 4. Pre-release hand dynamics (the key predictive features)
    pre_speeds = hand_speed[pre_release_indices]
    pre_heights = avg_hand[pre_release_indices, 1]
    pre_vels = hand_velocity[pre_release_indices]

    features.update({
        # Speed statistics
        'pre_speed_mean': np.mean(pre_speeds),
        'pre_speed_max': np.max(pre_speeds),
        'pre_speed_std': np.std(pre_speeds),
        'pre_speed_min': np.min(pre_speeds),

        # Height statistics
        'pre_height_mean': np.mean(pre_heights),
        'pre_height_max': np.max(pre_heights),
        'pre_height_std': np.std(pre_heights),
        'pre_height_range': np.max(pre_heights) - np.min(pre_heights),

        # Vertical velocity statistics (key for ball release)
        'pre_vy_mean': np.mean(pre_vels[:, 1]),
        'pre_vy_max': np.max(pre_vels[:, 1]),
        'pre_vy_std': np.std(pre_vels[:, 1]),

        # When did peak speed occur (relative to release)
        'frames_from_peak_speed': release_frame - (start_frame + np.argmax(pre_speeds)),
    })

    # 5. Hand separation (shooting technique)
    hand_separation = np.linalg.norm(left_hand - right_hand, axis=1)
    features.update({
        'hand_separation_rel': hand_separation[release_frame],
        'hand_separation_mean': np.mean(hand_separation[pre_release_indices]),
    })

    # 6. Arm configuration (elbow angles approximated by positions)
    left_arm_length = np.linalg.norm(left_elbow[release_frame] - left_shoulder[release_frame])
    right_arm_length = np.linalg.norm(right_elbow[release_frame] - right_shoulder[release_frame])
    left_forearm = np.linalg.norm(left_hand[release_frame] - left_elbow[release_frame])
    right_forearm = np.linalg.norm(right_hand[release_frame] - right_elbow[release_frame])

    features.update({
        'left_arm_length': left_arm_length,
        'right_arm_length': right_arm_length,
        'left_forearm_length': left_forearm,
        'right_forearm_length': right_forearm,
        'elbow_height_left': left_elbow[release_frame, 1],
        'elbow_height_right': right_elbow[release_frame, 1],
        'shoulder_height_left': left_shoulder[release_frame, 1],
        'shoulder_height_right': right_shoulder[release_frame, 1],
    })

    # --- TARGET FEATURES (what we want to predict) ---

    # Ball velocity at release (this is what determines landing outcome)
    features.update({
        'target_ball_vx': ball_velocity[release_frame, 0],
        'target_ball_vy': ball_velocity[release_frame, 1],
        'target_ball_vz': ball_velocity[release_frame, 2],
        'target_ball_speed': np.linalg.norm(ball_velocity[release_frame]),
    })

    # Ball position at release (for reference)
    features.update({
        'ball_x_rel': ball_position[release_frame, 0],
        'ball_y_rel': ball_position[release_frame, 1],
        'ball_z_rel': ball_position[release_frame, 2],
    })

    return features


def build_dataset(filepaths: List[str], output_csv: str):
    """
    Process multiple SkillMimic files and create training dataset.

    Args:
        filepaths: List of .pt file paths
        output_csv: Output CSV file path
    """
    all_features = []

    print(f"Processing {len(filepaths)} SkillMimic shots...")

    for i, filepath in enumerate(filepaths):
        try:
            features = extract_features_for_modeling(filepath)
            all_features.append(features)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(filepaths)}")

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    # Save
    df.to_csv(output_csv, index=False)

    print(f"\nDataset created: {output_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Input features: {df.shape[1] - 4}")  # Exclude targets
    print(f"  Target features: 4 (ball vx, vy, vz, speed)")

    # Show summary
    print(f"\nTarget statistics:")
    print(df[['target_ball_vx', 'target_ball_vy', 'target_ball_vz', 'target_ball_speed']].describe())

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build release prediction dataset from SkillMimic data"
    )
    parser.add_argument("--input-dir", required=True, help="Directory with .pt files")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--pattern", default="*.pt", help="File pattern to match")

    args = parser.parse_args()

    # Find all files
    input_path = Path(args.input_dir)
    filepaths = sorted(input_path.glob(args.pattern))
    filepaths = [str(f) for f in filepaths]

    if not filepaths:
        print(f"No files found matching {args.pattern} in {args.input_dir}")
        return

    print(f"Found {len(filepaths)} files")

    # Build dataset
    df = build_dataset(filepaths, args.output)

    print("\nDone!")
    print(f"\nNext steps:")
    print(f"1. Train model: hand kinematics → ball velocity")
    print(f"2. Apply to SPL data to predict release parameters")
    print(f"3. Use ballistic physics to predict landing outcomes")


if __name__ == "__main__":
    main()
