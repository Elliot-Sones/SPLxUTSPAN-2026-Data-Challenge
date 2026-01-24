#!/usr/bin/env python3
"""
Compare old (broken) vs new (fixed) joint extraction
"""

import json
import torch
import numpy as np

# Old implementation (broken)
OLD_JOINT_TRIPLETS = [
    [51, 52, 53],    # 0
    [54, 55, 56],    # 1
    [57, 58, 59],    # 2
    [108, 109, 110], # 3
    [111, 112, 113], # 4
    [114, 115, 116], # 5
    [220, 221, 222], # 6
    [223, 224, 225], # 7
    [226, 227, 228], # 8
    [229, 230, 231], # 9
    [232, 233, 234], # 10
    [235, 236, 237], # 11
    [238, 239, 240], # 12
    [241, 242, 243], # 13
    [244, 245, 246], # 14
    [247, 248, 249], # 15
    [250, 251, 252], # 16
    [253, 254, 255], # 17
    [259, 260, 261], # 18
    [262, 263, 264], # 19
]

def main():
    print("="*70)
    print("Old vs New Joint Extraction Comparison")
    print("="*70)

    # Load new joint map
    with open("visualisation/skillmimic_joint_map.json", 'r') as f:
        new_map = json.load(f)

    # Load sample data
    data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

    print(f"\nData shape: {data.shape}")
    print(f"Total features: {data.shape[1]}")

    # Old extraction
    print(f"\n{'OLD (BROKEN) IMPLEMENTATION':-^70}")
    print(f"Joints extracted: {len(OLD_JOINT_TRIPLETS)}")
    print(f"Feature ranges: {min(min(t) for t in OLD_JOINT_TRIPLETS)} - {max(max(t) for t in OLD_JOINT_TRIPLETS)}")
    print("\nFeatures used:")
    all_old_features = sorted(set(f for triplet in OLD_JOINT_TRIPLETS for f in triplet))
    print(f"  {all_old_features}")
    print(f"\nMissing critical features (165-323):")
    print(f"  These contain the actual skeleton positions!")

    # New extraction
    print(f"\n{'NEW (FIXED) IMPLEMENTATION':-^70}")
    print(f"Joints extracted: {len(new_map['joint_map'])}")
    joint_names = list(new_map['joint_map'].keys())
    print(f"\nBody parts included:")

    # Group by category
    categories = {
        'Core/Spine': ['Pelvis', 'Torso', 'Spine', 'Spine2', 'Chest', 'Neck', 'Head'],
        'Left Leg': ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe'],
        'Right Leg': ['R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe'],
        'Left Arm': ['L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist'],
        'Right Arm': ['R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist'],
        'Left Hand': [n for n in joint_names if n.startswith('L_') and any(f in n for f in ['Index', 'Middle', 'Pinky', 'Ring', 'Thumb'])],
        'Right Hand': [n for n in joint_names if n.startswith('R_') and any(f in n for f in ['Index', 'Middle', 'Pinky', 'Ring', 'Thumb'])],
    }

    for category, joints in categories.items():
        found = [j for j in joints if j in new_map['joint_map']]
        print(f"  {category}: {len(found)} joints")
        if category in ['Core/Spine', 'Left Leg', 'Right Leg', 'Left Arm', 'Right Arm']:
            print(f"    {', '.join(found)}")

    # Check what old features actually represent
    print(f"\n{'WHAT OLD FEATURES ACTUALLY ARE':-^70}")
    print("\nOld feature 51-59 (claimed to be 'pelvis/torso/arms'):")
    frame_0_data = []
    for triplet in OLD_JOINT_TRIPLETS[:3]:
        pos = [data[0, triplet[i]].item() for i in range(3)]
        frame_0_data.append(pos)
        print(f"  Features {triplet}: {pos}")

    print("\nOld feature 220-264 (claimed to be 'hand fingers'):")
    old_hand_start = 6  # Joint index in OLD_JOINT_TRIPLETS
    for i in range(old_hand_start, min(old_hand_start + 3, len(OLD_JOINT_TRIPLETS))):
        triplet = OLD_JOINT_TRIPLETS[i]
        pos = [data[0, triplet[j]].item() for j in range(3)]
        print(f"  Features {triplet}: {pos}")

    # Compare with new mapping
    print(f"\n{'ACTUAL JOINTS (NEW MAPPING)':-^70}")
    print("\nActual L_Wrist (features 219-221):")
    if 'L_Wrist' in new_map['joint_map']:
        indices = new_map['joint_map']['L_Wrist']
        pos = [data[0, indices[i]].item() for i in range(3)]
        print(f"  Features {indices}: {pos}")

    print("\nActual L_Index1 (features 222-224):")
    if 'L_Index1' in new_map['joint_map']:
        indices = new_map['joint_map']['L_Index1']
        pos = [data[0, indices[i]].item() for i in range(3)]
        print(f"  Features {indices}: {pos}")

    print("\nActual Pelvis (features 165-167):")
    if 'Pelvis' in new_map['joint_map']:
        indices = new_map['joint_map']['Pelvis']
        pos = [data[0, indices[i]].item() for i in range(3)]
        print(f"  Features {indices}: {pos}")

    print("\nActual Head (features 207-209):")
    if 'Head' in new_map['joint_map']:
        indices = new_map['joint_map']['Head']
        pos = [data[0, indices[i]].item() for i in range(3)]
        print(f"  Features {indices}: {pos}")

    # Calculate coverage
    print(f"\n{'COVERAGE ANALYSIS':-^70}")
    total_bodies = len(new_map['joint_map'])
    old_coverage = len(OLD_JOINT_TRIPLETS)
    new_coverage = total_bodies

    print(f"\nOld implementation:")
    print(f"  - Extracted: {old_coverage} joints")
    print(f"  - Missing: {total_bodies - old_coverage} joints ({(total_bodies - old_coverage) / total_bodies * 100:.1f}%)")
    print(f"  - Missing: Complete legs, torso, head, neck")

    print(f"\nNew implementation:")
    print(f"  - Extracted: {new_coverage} joints")
    print(f"  - Coverage: 100%")
    print(f"  - Includes: All body parts with proper anatomical structure")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
