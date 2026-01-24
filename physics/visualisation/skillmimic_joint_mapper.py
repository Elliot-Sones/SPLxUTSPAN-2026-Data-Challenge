#!/usr/bin/env python3
"""
SkillMimic Joint Mapper

Analyzes SkillMimic data to map 337 features to anatomical joints.
Based on mocap_humanoid.xml structure (61 bodies).

The MJCF humanoid has this hierarchy:
- Pelvis (root with freejoint: 6 DOF)
- Legs: L_Hip → L_Knee → L_Ankle → L_Toe (and R_ versions)
- Torso: Pelvis → Torso → Spine → Spine2 → Chest
- Head: Chest → Neck → Head
- Arms: Chest → L/R_Thorax → L/R_Shoulder → L/R_Elbow → L/R_Wrist
- Fingers: 5 fingers per hand × 3 segments = 30 finger bodies

Total: 61 bodies
Expected positions: 61 × 3 = 183 features
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

# Known body structure from mocap_humanoid.xml
BODY_HIERARCHY = [
    "Pelvis",
    # Legs
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    # Torso/spine
    "Torso", "Spine", "Spine2", "Chest",
    # Head
    "Neck", "Head",
    # Left arm
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    # Left hand fingers
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    # Right arm
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    # Right hand fingers
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]

# Anatomical connections for skeleton
SKELETON_CONNECTIONS = [
    # Spine/torso
    ("Pelvis", "Torso"),
    ("Torso", "Spine"),
    ("Spine", "Spine2"),
    ("Spine2", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),

    # Left leg
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Toe"),

    # Right leg
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Toe"),

    # Left arm
    ("Chest", "L_Thorax"),
    ("L_Thorax", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),

    # Left hand
    ("L_Wrist", "L_Index1"),
    ("L_Index1", "L_Index2"),
    ("L_Index2", "L_Index3"),
    ("L_Wrist", "L_Middle1"),
    ("L_Middle1", "L_Middle2"),
    ("L_Middle2", "L_Middle3"),
    ("L_Wrist", "L_Pinky1"),
    ("L_Pinky1", "L_Pinky2"),
    ("L_Pinky2", "L_Pinky3"),
    ("L_Wrist", "L_Ring1"),
    ("L_Ring1", "L_Ring2"),
    ("L_Ring2", "L_Ring3"),
    ("L_Wrist", "L_Thumb1"),
    ("L_Thumb1", "L_Thumb2"),
    ("L_Thumb2", "L_Thumb3"),

    # Right arm
    ("Chest", "R_Thorax"),
    ("R_Thorax", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),

    # Right hand
    ("R_Wrist", "R_Index1"),
    ("R_Index1", "R_Index2"),
    ("R_Index2", "R_Index3"),
    ("R_Wrist", "R_Middle1"),
    ("R_Middle1", "R_Middle2"),
    ("R_Middle2", "R_Middle3"),
    ("R_Wrist", "R_Pinky1"),
    ("R_Pinky1", "R_Pinky2"),
    ("R_Pinky2", "R_Pinky3"),
    ("R_Wrist", "R_Ring1"),
    ("R_Ring1", "R_Ring2"),
    ("R_Ring2", "R_Ring3"),
    ("R_Wrist", "R_Thumb1"),
    ("R_Thumb1", "R_Thumb2"),
    ("R_Thumb2", "R_Thumb3"),
]


def analyze_feature_characteristics(data: torch.Tensor):
    """Analyze statistical properties of each feature"""
    print(f"\nAnalyzing {data.shape[1]} features across {data.shape[0]} frames...")

    stats = {}
    for i in range(data.shape[1]):
        feature = data[:, i].numpy()
        stats[i] = {
            'min': float(feature.min()),
            'max': float(feature.max()),
            'mean': float(feature.mean()),
            'std': float(feature.std()),
            'range': float(feature.max() - feature.min()),
        }

    return stats


def find_position_features(stats, num_bodies=61):
    """
    Identify which features are likely 3D positions.

    Strategy:
    - Position features typically come in consecutive triplets (x, y, z)
    - They have reasonable ranges (not too large, not too small)
    - Standard deviation > 0 (moving joints)
    """
    print("\nIdentifying position features...")

    # Look for consecutive triplets with similar characteristics
    position_triplets = []

    # Start from feature 0 and look for 61 bodies × 3 = 183 features
    expected_positions = num_bodies * 3

    for start_idx in range(0, min(200, len(stats) - expected_positions + 1)):
        triplets = []
        for body_idx in range(num_bodies):
            base = start_idx + body_idx * 3
            if base + 2 >= len(stats):
                break

            # Check if these 3 consecutive features form a valid position triplet
            x_idx, y_idx, z_idx = base, base + 1, base + 2
            x_stat = stats[x_idx]
            y_stat = stats[y_idx]
            z_stat = stats[z_idx]

            # Heuristics for position data:
            # 1. All have non-zero std (movement)
            # 2. Reasonable ranges (0.01 to 10.0 units)
            # 3. Y (vertical) typically has largest range for standing humanoid

            if (x_stat['std'] > 0.001 and y_stat['std'] > 0.001 and z_stat['std'] > 0.001 and
                0.01 < x_stat['range'] < 10.0 and 0.01 < y_stat['range'] < 10.0 and
                0.01 < z_stat['range'] < 10.0):
                triplets.append([x_idx, y_idx, z_idx])

        if len(triplets) == num_bodies:
            print(f"Found {len(triplets)} position triplets starting at feature {start_idx}")
            position_triplets = triplets
            break

    return position_triplets


def map_features_to_bodies(position_triplets, data: torch.Tensor):
    """Map position triplets to anatomical body names"""

    if len(position_triplets) != len(BODY_HIERARCHY):
        print(f"WARNING: Found {len(position_triplets)} triplets, expected {len(BODY_HIERARCHY)}")
        print("Will attempt to map what we have...")

    # Create mapping
    joint_map = {}
    for i, body_name in enumerate(BODY_HIERARCHY):
        if i < len(position_triplets):
            joint_map[body_name] = position_triplets[i]

    # Validate by checking if pelvis is near center and stationary
    if "Pelvis" in joint_map:
        pelvis_idx = joint_map["Pelvis"]
        pelvis_pos = data[:, pelvis_idx].numpy()
        pelvis_movement = pelvis_pos.std(axis=0)
        print(f"\nPelvis position variability: x={pelvis_movement[0]:.4f}, "
              f"y={pelvis_movement[1]:.4f}, z={pelvis_movement[2]:.4f}")
        print("(Root joint should have lower variability than limbs)")

    # Check limb positions make sense
    if "L_Wrist" in joint_map and "R_Wrist" in joint_map:
        l_wrist = data[0, joint_map["L_Wrist"]].numpy()
        r_wrist = data[0, joint_map["R_Wrist"]].numpy()
        print(f"\nFrame 0 wrist positions:")
        print(f"  L_Wrist: {l_wrist}")
        print(f"  R_Wrist: {r_wrist}")
        print("(Should be symmetric around body center)")

    return joint_map


def validate_skeleton(joint_map, data: torch.Tensor):
    """Validate that the skeleton makes anatomical sense"""
    print("\n" + "="*70)
    print("Skeleton Validation")
    print("="*70)

    issues = []

    # Check symmetry: left and right limbs should be mirror images
    frame_idx = 0
    pairs = [
        ("L_Wrist", "R_Wrist"),
        ("L_Elbow", "R_Elbow"),
        ("L_Knee", "R_Knee"),
        ("L_Ankle", "R_Ankle"),
    ]

    for left, right in pairs:
        if left in joint_map and right in joint_map:
            l_pos = data[frame_idx, joint_map[left]].numpy()
            r_pos = data[frame_idx, joint_map[right]].numpy()

            # X coordinates should be opposite sign (if centered at 0)
            # Y and Z should be similar
            y_diff = abs(l_pos[1] - r_pos[1])
            z_diff = abs(l_pos[2] - r_pos[2])

            print(f"{left} vs {right}:")
            print(f"  Y difference: {y_diff:.4f} (should be small)")
            print(f"  Z difference: {z_diff:.4f} (should be small)")

            if y_diff > 0.5 or z_diff > 0.5:
                issues.append(f"Large asymmetry between {left} and {right}")

    # Check limb lengths are reasonable
    limb_connections = [
        ("L_Shoulder", "L_Elbow"),
        ("L_Elbow", "L_Wrist"),
        ("L_Hip", "L_Knee"),
        ("L_Knee", "L_Ankle"),
    ]

    print("\nLimb lengths:")
    for start, end in limb_connections:
        if start in joint_map and end in joint_map:
            start_pos = data[frame_idx, joint_map[start]].numpy()
            end_pos = data[frame_idx, joint_map[end]].numpy()
            length = np.linalg.norm(end_pos - start_pos)
            print(f"  {start} → {end}: {length:.4f}")

            if length < 0.05 or length > 2.0:
                issues.append(f"Unusual limb length for {start} → {end}: {length:.4f}")

    if issues:
        print("\n⚠️  Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Skeleton structure looks good!")

    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Map SkillMimic features to anatomical joints")
    parser.add_argument("filepath", help="Path to .pt file")
    parser.add_argument("--output", default="visualisation/skillmimic_joint_map.json",
                        help="Output JSON file for joint mapping")

    args = parser.parse_args()

    print("="*70)
    print("SkillMimic Joint Mapper")
    print("="*70)
    print(f"Loading: {args.filepath}")

    # Load data
    data = torch.load(args.filepath, map_location='cpu')
    if not isinstance(data, torch.Tensor):
        raise RuntimeError(f"Expected torch.Tensor, got {type(data)}")

    print(f"Data shape: {data.shape}")
    print(f"Expected structure: {len(BODY_HIERARCHY)} bodies × 3 = {len(BODY_HIERARCHY) * 3} position features")

    # Analyze features
    stats = analyze_feature_characteristics(data)

    # Find position features
    position_triplets = find_position_features(stats, num_bodies=len(BODY_HIERARCHY))

    if not position_triplets:
        print("\n❌ Could not identify position features!")
        print("Falling back to manual inspection...")

        # Show feature ranges to help manual identification
        print("\nFeature ranges (first 200):")
        for i in range(min(200, len(stats))):
            s = stats[i]
            print(f"Feature {i:3d}: range={s['range']:8.4f}, std={s['std']:8.4f}, "
                  f"mean={s['mean']:8.4f}")
        return

    # Map to anatomical names
    joint_map = map_features_to_bodies(position_triplets, data)

    # Validate
    is_valid = validate_skeleton(joint_map, data)

    # Save mapping
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "joint_map": joint_map,
        "skeleton_connections": [[a, b] for a, b in SKELETON_CONNECTIONS],
        "num_bodies": len(BODY_HIERARCHY),
        "num_features": data.shape[1],
        "validated": is_valid
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Joint mapping saved to: {output_path}")
    print(f"  - Mapped {len(joint_map)} bodies")
    print(f"  - {len(SKELETON_CONNECTIONS)} skeleton connections")
    print("="*70)


if __name__ == "__main__":
    main()
