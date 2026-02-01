"""
KEYPOINT EXPLORATION

First, understand ALL available keypoints in the dataset.
Then we can design proper physics-based feature groups.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns


def main():
    print("=" * 80)
    print("KEYPOINT EXPLORATION")
    print("=" * 80)
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()

    print(f"Data shape: {X.shape}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Frames: {X.shape[1]}")
    print(f"  Keypoints: {X.shape[2]}")
    print()

    # Parse keypoint names
    print("=" * 80)
    print("ALL KEYPOINT COLUMNS")
    print("=" * 80)
    print()

    # Group by body part
    body_parts = {}
    for col in keypoint_cols:
        # Split into body part and axis
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            body_part, axis = parts
            if body_part not in body_parts:
                body_parts[body_part] = []
            body_parts[body_part].append(axis)

    print(f"Total keypoints: {len(keypoint_cols)}")
    print(f"Unique body parts: {len(body_parts)}")
    print()

    # List all body parts
    print("BODY PARTS WITH AVAILABLE AXES:")
    print("-" * 60)

    for body_part in sorted(body_parts.keys()):
        axes = sorted(body_parts[body_part])
        print(f"  {body_part}: {axes}")

    # Categorize by body region
    print()
    print("=" * 80)
    print("CATEGORIZED BY BODY REGION")
    print("=" * 80)

    categories = {
        'HEAD/FACE': [],
        'TORSO': [],
        'RIGHT_ARM': [],
        'LEFT_ARM': [],
        'RIGHT_HAND': [],
        'LEFT_HAND': [],
        'RIGHT_LEG': [],
        'LEFT_LEG': [],
        'OTHER': []
    }

    for body_part in body_parts.keys():
        bp_lower = body_part.lower()

        if any(x in bp_lower for x in ['nose', 'eye', 'ear', 'mouth', 'head', 'face']):
            categories['HEAD/FACE'].append(body_part)
        elif any(x in bp_lower for x in ['hip', 'spine', 'chest', 'neck', 'pelvis', 'torso']):
            categories['TORSO'].append(body_part)
        elif 'right' in bp_lower and any(x in bp_lower for x in ['shoulder', 'elbow', 'arm']):
            categories['RIGHT_ARM'].append(body_part)
        elif 'left' in bp_lower and any(x in bp_lower for x in ['shoulder', 'elbow', 'arm']):
            categories['LEFT_ARM'].append(body_part)
        elif 'right' in bp_lower and any(x in bp_lower for x in ['wrist', 'hand', 'thumb', 'index', 'middle', 'ring', 'pinky', 'finger']):
            categories['RIGHT_HAND'].append(body_part)
        elif 'left' in bp_lower and any(x in bp_lower for x in ['wrist', 'hand', 'thumb', 'index', 'middle', 'ring', 'pinky', 'finger']):
            categories['LEFT_HAND'].append(body_part)
        elif 'right' in bp_lower and any(x in bp_lower for x in ['knee', 'ankle', 'foot', 'toe', 'heel', 'leg', 'thigh']):
            categories['RIGHT_LEG'].append(body_part)
        elif 'left' in bp_lower and any(x in bp_lower for x in ['knee', 'ankle', 'foot', 'toe', 'heel', 'leg', 'thigh']):
            categories['LEFT_LEG'].append(body_part)
        else:
            categories['OTHER'].append(body_part)

    for category, parts in categories.items():
        if parts:
            print(f"\n{category}:")
            for part in sorted(parts):
                print(f"  - {part}")

    # Check for finger/hand keypoints specifically
    print()
    print("=" * 80)
    print("HAND/FINGER KEYPOINTS (CRITICAL FOR ANGLE)")
    print("=" * 80)

    hand_parts = [bp for bp in body_parts.keys() if any(x in bp.lower() for x in
                  ['wrist', 'hand', 'thumb', 'index', 'middle', 'ring', 'pinky', 'finger', 'palm'])]

    print(f"\nFound {len(hand_parts)} hand-related keypoints:")
    for part in sorted(hand_parts):
        axes = body_parts[part]
        print(f"  {part}: {axes}")

    # Sample data statistics
    print()
    print("=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)

    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    # Check for NaN patterns
    nan_counts = np.isnan(X).sum(axis=(0, 1))
    nan_pct = nan_counts / (X.shape[0] * X.shape[1]) * 100

    print("\nKeypoints with >1% NaN:")
    for i, col in enumerate(keypoint_cols):
        if nan_pct[i] > 1:
            print(f"  {col}: {nan_pct[i]:.1f}% NaN")

    if (nan_pct > 1).sum() == 0:
        print("  None - data is complete!")

    # Check variance of key features
    print()
    print("=" * 80)
    print("FEATURE VARIANCE CHECK")
    print("=" * 80)
    print("\nChecking which features have meaningful variance at release frame (155)")

    frame = 155
    low_var_features = []
    high_var_features = []

    for col in keypoint_cols:
        idx = col_to_idx[col]
        values = X[:, frame, idx]
        var = np.nanvar(values)

        if var < 0.001:
            low_var_features.append((col, var))
        elif var > 1.0:
            high_var_features.append((col, var))

    print(f"\nLow variance features (<0.001): {len(low_var_features)}")
    for col, var in low_var_features[:10]:
        print(f"  {col}: var={var:.6f}")

    print(f"\nHigh variance features (>1.0): {len(high_var_features)}")
    for col, var in sorted(high_var_features, key=lambda x: -x[1])[:10]:
        print(f"  {col}: var={var:.4f}")


if __name__ == "__main__":
    main()
