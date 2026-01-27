#!/usr/bin/env python3
"""
Extract Ball-Hand Features from SkillMimic Data

Use this as a template for extracting similar features from SPL data.
"""

import torch
import numpy as np
import json


def extract_ball_hand_features(filepath: str):
    """
    Extract ball position, hand positions, and contact features.

    Returns dict with:
    - ball_position: (N, 3) array
    - left_hand_position: (N, 3) array
    - right_hand_position: (N, 3) array
    - ball_hand_distance: (N,) array
    - contact_flag: (N,) array
    - ball_velocity: (N, 3) array
    - release_frame: int
    """

    # Load data
    data = torch.load(filepath, map_location='cpu')

    # Load joint mapping
    with open("visualisation/skillmimic_joint_map.json", 'r') as f:
        joint_map = json.load(f)['joint_map']

    # Extract ball features
    ball_position = data[:, 318:321].numpy()  # Features 318-320
    ball_velocity = data[:, 325:328].numpy()  # Features 325-327
    contact_flag = data[:, 336].numpy()       # Feature 336

    # Extract hand positions from skeleton
    l_wrist = joint_map['L_Wrist']
    r_wrist = joint_map['R_Wrist']

    left_hand = data[:, l_wrist].numpy()
    right_hand = data[:, r_wrist].numpy()

    # Calculate average hand position
    avg_hand = (left_hand + right_hand) / 2

    # Ball-hand distance
    ball_hand_distance = np.linalg.norm(ball_position - avg_hand, axis=1)

    # Find release frame
    release_frame = None
    for i in range(len(contact_flag) - 1):
        if contact_flag[i] == 1 and contact_flag[i+1] == 0:
            release_frame = i + 1
            break

    if release_frame is None:
        release_frame = np.where(contact_flag == 0)[0][0] if np.any(contact_flag == 0) else 0

    return {
        'ball_position': ball_position,
        'ball_velocity': ball_velocity,
        'left_hand_position': left_hand,
        'right_hand_position': right_hand,
        'avg_hand_position': avg_hand,
        'ball_hand_distance': ball_hand_distance,
        'contact_flag': contact_flag,
        'release_frame': release_frame,
        'num_frames': len(data)
    }


def analyze_features(features: dict):
    """Analyze extracted features"""

    print("="*70)
    print("Ball-Hand Feature Analysis")
    print("="*70)

    release_frame = features['release_frame']
    contact_flag = features['contact_flag']
    distance = features['ball_hand_distance']

    # Contact phase statistics
    contact_frames = np.where(contact_flag == 1)[0]
    release_frames = np.where(contact_flag == 0)[0]

    print(f"\nData Summary:")
    print(f"  Total frames: {features['num_frames']}")
    print(f"  Contact frames: {len(contact_frames)} (0-{contact_frames[-1] if len(contact_frames) > 0 else 'N/A'})")
    print(f"  Release frame: {release_frame}")
    print(f"  Flight frames: {len(release_frames)} ({release_frame}-{len(contact_flag)-1})")

    print(f"\nBall-Hand Distance:")
    if len(contact_frames) > 0:
        print(f"  During contact: {distance[contact_frames].mean():.3f} ± {distance[contact_frames].std():.3f}")
        print(f"  Min: {distance[contact_frames].min():.3f}, Max: {distance[contact_frames].max():.3f}")

    if len(release_frames) > 0:
        print(f"  After release: {distance[release_frames].mean():.3f} ± {distance[release_frames].std():.3f}")

    print(f"\nBall Position at Key Frames:")
    for frame in [0, release_frame-1, release_frame, release_frame+5, -1]:
        if 0 <= frame < len(contact_flag):
            pos = features['ball_position'][frame]
            print(f"  Frame {frame:3d}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print(f"\nBall Velocity at Release:")
    if 0 <= release_frame < len(features['ball_velocity']):
        vel = features['ball_velocity'][release_frame]
        speed = np.linalg.norm(vel)
        print(f"  Velocity: ({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
        print(f"  Speed: {speed:.3f} units/frame")

    print("="*70)


def validate_features(features: dict):
    """Validate extracted features"""

    print("\nFeature Validation:")
    print("-"*70)

    issues = []

    # Check 1: Ball should be close to hands during contact
    contact_frames = features['contact_flag'] == 1
    if np.any(contact_frames):
        max_distance = features['ball_hand_distance'][contact_frames].max()
        if max_distance > 0.3:
            issues.append(f"Ball too far from hands during contact: {max_distance:.3f}")
        else:
            print("✓ Ball distance during contact < 0.3")

    # Check 2: Ball should move away after release
    release_frame = features['release_frame']
    if release_frame < len(features['ball_hand_distance']) - 5:
        dist_at_release = features['ball_hand_distance'][release_frame]
        dist_after = features['ball_hand_distance'][release_frame + 5]
        if dist_after <= dist_at_release:
            issues.append(f"Ball not moving away after release")
        else:
            print("✓ Ball moves away after release")

    # Check 3: Contact flag should transition
    transitions = np.diff(features['contact_flag'].astype(int))
    if not np.any(transitions == -1):
        issues.append("No contact-to-release transition found")
    else:
        print("✓ Contact flag transitions correctly")

    # Check 4: Ball velocity should be non-zero after release
    if release_frame < len(features['ball_velocity']):
        vel = features['ball_velocity'][release_frame]
        speed = np.linalg.norm(vel)
        if speed < 0.01:
            issues.append(f"Ball velocity too low at release: {speed:.3f}")
        else:
            print(f"✓ Ball has velocity at release: {speed:.3f}")

    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All validation checks passed!")

    print("-"*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ball-hand features from SkillMimic data"
    )
    parser.add_argument("filepath", help="Path to .pt file")
    parser.add_argument("--save", help="Save features to .npz file")

    args = parser.parse_args()

    # Extract features
    features = extract_ball_hand_features(args.filepath)

    # Analyze
    analyze_features(features)

    # Validate
    validate_features(features)

    # Save if requested
    if args.save:
        np.savez(
            args.save,
            ball_position=features['ball_position'],
            ball_velocity=features['ball_velocity'],
            left_hand=features['left_hand_position'],
            right_hand=features['right_hand_position'],
            distance=features['ball_hand_distance'],
            contact=features['contact_flag'],
            release_frame=features['release_frame']
        )
        print(f"\n✓ Features saved to: {args.save}")


if __name__ == "__main__":
    main()
