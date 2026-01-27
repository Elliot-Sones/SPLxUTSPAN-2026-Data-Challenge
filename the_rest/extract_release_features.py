#!/usr/bin/env python3
"""
Extract Release Features from SkillMimic Data

Goal: Learn the relationship between hand kinematics and ball release
parameters to apply to SPL data (which has no ball tracking).

Output features for each shot:
1. Release detection: When did ball leave hands?
2. Release kinematics: Hand positions, velocities at release
3. Release parameters: Ball initial velocity at release
4. Pre-release dynamics: Hand motion leading up to release
"""

import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path


def load_joint_mapping(map_path: str = "visualisation/skillmimic_joint_map.json") -> dict:
    with open(map_path, 'r') as f:
        return json.load(f)['joint_map']


def extract_release_features(filepath: str) -> dict:
    """
    Extract all features needed to model release physics.

    Returns:
        Dictionary with release frame, hand kinematics, ball velocity, etc.
    """
    # Load data
    data = torch.load(filepath, map_location='cpu')
    joint_map = load_joint_mapping()

    # Extract features
    ball_position = data[:, 318:321].numpy()  # Ball position
    ball_velocity = data[:, 325:328].numpy()  # Ball velocity (3D)
    contact_flag = data[:, 336].numpy()       # Contact flag

    # Hand positions
    l_wrist = joint_map['L_Wrist']
    r_wrist = joint_map['R_Wrist']
    left_hand = data[:, l_wrist].numpy()
    right_hand = data[:, r_wrist].numpy()
    avg_hand = (left_hand + right_hand) / 2

    # Key body positions for shooting mechanics
    l_elbow = joint_map['L_Elbow']
    r_elbow = joint_map['R_Elbow']
    l_shoulder = joint_map['L_Shoulder']
    r_shoulder = joint_map['R_Shoulder']

    left_elbow = data[:, l_elbow].numpy()
    right_elbow = data[:, r_elbow].numpy()
    left_shoulder = data[:, l_shoulder].numpy()
    right_shoulder = data[:, r_shoulder].numpy()

    # Find release frame
    release_frame = None
    for i in range(len(contact_flag) - 1):
        if contact_flag[i] == 1 and contact_flag[i+1] == 0:
            release_frame = i + 1
            break

    if release_frame is None:
        release_frame = np.where(contact_flag == 0)[0][0] if np.any(contact_flag == 0) else 0

    # Calculate velocities (finite difference)
    def calculate_velocity(positions):
        """Calculate velocity using central difference"""
        vel = np.zeros_like(positions)
        vel[1:-1] = (positions[2:] - positions[:-2]) / 2  # Central difference
        vel[0] = positions[1] - positions[0]  # Forward difference
        vel[-1] = positions[-1] - positions[-2]  # Backward difference
        return vel

    hand_velocity = calculate_velocity(avg_hand)
    left_hand_vel = calculate_velocity(left_hand)
    right_hand_vel = calculate_velocity(right_hand)

    # Calculate accelerations
    def calculate_acceleration(velocity):
        """Calculate acceleration from velocity"""
        acc = np.zeros_like(velocity)
        acc[1:-1] = (velocity[2:] - velocity[:-2]) / 2
        acc[0] = velocity[1] - velocity[0]
        acc[-1] = velocity[-1] - velocity[-2]
        return acc

    hand_acceleration = calculate_acceleration(hand_velocity)

    # Extract pre-release window (10 frames before release)
    window_size = 10
    pre_release_start = max(0, release_frame - window_size)
    pre_release_frames = range(pre_release_start, release_frame)

    # Features at release frame
    release_features = {
        'release_frame': release_frame,
        'total_frames': len(data),

        # Ball state at release
        'ball_position': ball_position[release_frame],
        'ball_velocity': ball_velocity[release_frame],
        'ball_speed': np.linalg.norm(ball_velocity[release_frame]),

        # Hand state at release
        'left_hand_position': left_hand[release_frame],
        'right_hand_position': right_hand[release_frame],
        'avg_hand_position': avg_hand[release_frame],
        'hand_velocity': hand_velocity[release_frame],
        'hand_speed': np.linalg.norm(hand_velocity[release_frame]),
        'hand_acceleration': hand_acceleration[release_frame],

        # Hand separation (shooting hand technique)
        'hand_separation': np.linalg.norm(left_hand[release_frame] - right_hand[release_frame]),

        # Arm angles at release (elbow and shoulder)
        'left_elbow_position': left_elbow[release_frame],
        'right_elbow_position': right_elbow[release_frame],
        'left_shoulder_position': left_shoulder[release_frame],
        'right_shoulder_position': right_shoulder[release_frame],

        # Ball-hand relationship at release
        'ball_hand_distance': np.linalg.norm(ball_position[release_frame] - avg_hand[release_frame]),
        'ball_hand_relative_velocity': ball_velocity[release_frame] - hand_velocity[release_frame],

        # Pre-release dynamics (10 frames before)
        'pre_release_hand_positions': avg_hand[pre_release_frames],
        'pre_release_hand_velocities': hand_velocity[pre_release_frames],
        'pre_release_hand_speeds': np.linalg.norm(hand_velocity[pre_release_frames], axis=1),

        # Peak hand speed before release
        'peak_hand_speed_pre_release': np.max(np.linalg.norm(hand_velocity[pre_release_frames], axis=1)),

        # Hand height at release (Y coordinate, assuming Y is vertical)
        'release_height': avg_hand[release_frame, 1],

        # Trajectory after release (for validation)
        'post_release_ball_positions': ball_position[release_frame:release_frame+10] if release_frame + 10 < len(data) else ball_position[release_frame:],
    }

    return release_features


def compute_trajectory_features(ball_positions: np.ndarray) -> dict:
    """
    Compute trajectory features from ball positions.
    These would be your TARGET features to predict from hand kinematics.
    """
    if len(ball_positions) < 5:
        return {}

    # Fit parabolic trajectory (simplified)
    # In real physics: x = x0 + vx*t, y = y0 + vy*t - 0.5*g*t^2, z = z0 + vz*t

    # Extract trajectory characteristics
    start_pos = ball_positions[0]
    end_pos = ball_positions[-1]

    # Angle (elevation angle)
    displacement = end_pos - start_pos
    horizontal_dist = np.sqrt(displacement[0]**2 + displacement[2]**2)  # X-Z plane
    vertical_dist = displacement[1]  # Y axis
    angle = np.arctan2(vertical_dist, horizontal_dist) * 180 / np.pi

    # Depth (distance traveled)
    depth = np.linalg.norm(displacement)

    # Left/Right (lateral deviation)
    # Assuming Z is lateral direction
    left_right = displacement[2]

    # Peak height
    peak_height = np.max(ball_positions[:, 1])

    return {
        'angle': angle,
        'depth': depth,
        'left_right': left_right,
        'peak_height': peak_height,
        'displacement': displacement,
    }


def analyze_release_prediction(features: dict):
    """
    Analyze what features are most predictive of release parameters.
    This guides what to extract from SPL data.
    """
    print("=" * 70)
    print("Release Feature Analysis")
    print("=" * 70)

    print(f"\n{'RELEASE DETECTION':^70}")
    print("-" * 70)
    print(f"Release frame: {features['release_frame']} / {features['total_frames']}")
    print(f"Release height: {features['release_height']:.3f}")

    print(f"\n{'HAND KINEMATICS AT RELEASE':^70}")
    print("-" * 70)
    print(f"Hand position: ({features['avg_hand_position'][0]:.3f}, "
          f"{features['avg_hand_position'][1]:.3f}, {features['avg_hand_position'][2]:.3f})")
    print(f"Hand velocity: ({features['hand_velocity'][0]:.3f}, "
          f"{features['hand_velocity'][1]:.3f}, {features['hand_velocity'][2]:.3f})")
    print(f"Hand speed: {features['hand_speed']:.3f}")
    print(f"Hand acceleration: {np.linalg.norm(features['hand_acceleration']):.3f}")
    print(f"Hand separation: {features['hand_separation']:.3f}")

    print(f"\n{'BALL PARAMETERS AT RELEASE':^70}")
    print("-" * 70)
    print(f"Ball velocity: ({features['ball_velocity'][0]:.3f}, "
          f"{features['ball_velocity'][1]:.3f}, {features['ball_velocity'][2]:.3f})")
    print(f"Ball speed: {features['ball_speed']:.3f}")
    print(f"Ball-hand distance: {features['ball_hand_distance']:.3f}")

    rel_vel = features['ball_hand_relative_velocity']
    print(f"Ball velocity relative to hand: ({rel_vel[0]:.3f}, {rel_vel[1]:.3f}, {rel_vel[2]:.3f})")
    print(f"Relative speed: {np.linalg.norm(rel_vel):.3f}")

    print(f"\n{'PRE-RELEASE DYNAMICS':^70}")
    print("-" * 70)
    print(f"Peak hand speed (10 frames before): {features['peak_hand_speed_pre_release']:.3f}")
    print(f"Mean hand speed (10 frames before): {np.mean(features['pre_release_hand_speeds']):.3f}")

    # Trajectory analysis
    if len(features['post_release_ball_positions']) > 5:
        traj = compute_trajectory_features(features['post_release_ball_positions'])
        print(f"\n{'PREDICTED TRAJECTORY (from release velocity)':^70}")
        print("-" * 70)
        print(f"Launch angle: {traj['angle']:.1f} degrees")
        print(f"Depth (distance): {traj['depth']:.3f}")
        print(f"Left/Right deviation: {traj['left_right']:.3f}")
        print(f"Peak height: {traj['peak_height']:.3f}")

    print("=" * 70)


def extract_modeling_features(filepath: str) -> pd.DataFrame:
    """
    Extract features in format suitable for ML modeling.

    Returns DataFrame with one row per shot containing:
    - Input features: Hand kinematics before/at release
    - Target features: Ball velocity at release (to predict landing)
    """
    features = extract_release_features(filepath)

    # Flatten pre-release time series into statistics
    pre_release_speeds = features['pre_release_hand_speeds']
    pre_release_positions = features['pre_release_hand_positions']

    # Build feature vector
    feature_dict = {
        # Release frame timing
        'release_frame': features['release_frame'],
        'release_frame_pct': features['release_frame'] / features['total_frames'],

        # Hand position at release
        'hand_x': features['avg_hand_position'][0],
        'hand_y': features['avg_hand_position'][1],
        'hand_z': features['avg_hand_position'][2],

        # Hand velocity at release
        'hand_vx': features['hand_velocity'][0],
        'hand_vy': features['hand_velocity'][1],
        'hand_vz': features['hand_velocity'][2],
        'hand_speed': features['hand_speed'],

        # Hand acceleration at release
        'hand_ax': features['hand_acceleration'][0],
        'hand_ay': features['hand_acceleration'][1],
        'hand_az': features['hand_acceleration'][2],

        # Hand separation
        'hand_separation': features['hand_separation'],

        # Pre-release dynamics (statistics)
        'pre_release_speed_mean': np.mean(pre_release_speeds),
        'pre_release_speed_max': np.max(pre_release_speeds),
        'pre_release_speed_std': np.std(pre_release_speeds),
        'pre_release_height_mean': np.mean(pre_release_positions[:, 1]),
        'pre_release_height_max': np.max(pre_release_positions[:, 1]),

        # TARGET: Ball velocity at release (what we want to predict from hand kinematics)
        'target_ball_vx': features['ball_velocity'][0],
        'target_ball_vy': features['ball_velocity'][1],
        'target_ball_vz': features['ball_velocity'][2],
        'target_ball_speed': features['ball_speed'],
    }

    return pd.DataFrame([feature_dict])


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract release features for SPL mapping"
    )
    parser.add_argument("filepath", help="Path to SkillMimic .pt file")
    parser.add_argument("--analyze", action="store_true", help="Print analysis")
    parser.add_argument("--save-csv", help="Save modeling features to CSV")

    args = parser.parse_args()

    # Extract features
    features = extract_release_features(args.filepath)

    # Analyze
    if args.analyze:
        analyze_release_prediction(features)

    # Save for modeling
    if args.save_csv:
        df = extract_modeling_features(args.filepath)
        df.to_csv(args.save_csv, index=False)
        print(f"\nFeatures saved to: {args.save_csv}")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
