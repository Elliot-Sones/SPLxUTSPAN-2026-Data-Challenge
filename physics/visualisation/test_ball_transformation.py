#!/usr/bin/env python3
"""
Test if ball position is in root-relative coordinates
"""

import json
import torch
import numpy as np

# Load data
data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

# Load joint mapping
with open("visualisation/skillmimic_joint_map.json", 'r') as f:
    joint_map = json.load(f)['joint_map']

# Original ball position (features 119, 325, 326)
ball_x_raw = data[:, 325].numpy()
ball_y_raw = data[:, 119].numpy()
ball_z_raw = data[:, 326].numpy()

# Pelvis (root) position
pelvis_indices = joint_map['Pelvis']
pelvis_x = data[:, pelvis_indices[0]].numpy()
pelvis_y = data[:, pelvis_indices[1]].numpy()
pelvis_z = data[:, pelvis_indices[2]].numpy()

print("="*70)
print("Testing Ball Position Transformations")
print("="*70)

# Test different transformations
print("\nFrame 0 (ball in hand):")
print(f"  Pelvis: ({pelvis_x[0]:.4f}, {pelvis_y[0]:.4f}, {pelvis_z[0]:.4f})")
print(f"  L_Wrist: ", end="")
l_wrist = joint_map['L_Wrist']
print(f"({data[0, l_wrist[0]]:.4f}, {data[0, l_wrist[1]]:.4f}, {data[0, l_wrist[2]]:.4f})")

print(f"\n  Ball raw: ({ball_x_raw[0]:.4f}, {ball_y_raw[0]:.4f}, {ball_z_raw[0]:.4f})")

# Try adding pelvis
ball_x_global = ball_x_raw + pelvis_x
ball_y_global = ball_y_raw + pelvis_y
ball_z_global = ball_z_raw + pelvis_z
print(f"  Ball + pelvis: ({ball_x_global[0]:.4f}, {ball_y_global[0]:.4f}, {ball_z_global[0]:.4f})")

# Calculate distance
dist = np.sqrt((ball_x_global[0] - data[0, l_wrist[0]].item())**2 +
               (ball_y_global[0] - data[0, l_wrist[1]].item())**2 +
               (ball_z_global[0] - data[0, l_wrist[2]].item())**2)
print(f"  Distance to L_Wrist: {dist:.4f}")

# Try other transformations
print("\n" + "="*70)
print("Testing various coordinate transformations:")
print("="*70)

transformations = [
    ("Raw (no transform)", ball_x_raw, ball_y_raw, ball_z_raw),
    ("+ Pelvis", ball_x_raw + pelvis_x, ball_y_raw + pelvis_y, ball_z_raw + pelvis_z),
    ("Swap XY", ball_y_raw, ball_x_raw, ball_z_raw),
    ("Swap XZ", ball_z_raw, ball_y_raw, ball_x_raw),
    ("Swap YZ", ball_x_raw, ball_z_raw, ball_y_raw),
]

for name, bx, by, bz in transformations:
    dist0 = np.sqrt((bx[0] - data[0, l_wrist[0]].item())**2 +
                    (by[0] - data[0, l_wrist[1]].item())**2 +
                    (bz[0] - data[0, l_wrist[2]].item())**2)

    dist59 = np.sqrt((bx[59] - data[59, l_wrist[0]].item())**2 +
                     (by[59] - data[59, l_wrist[1]].item())**2 +
                     (bz[59] - data[59, l_wrist[2]].item())**2)

    print(f"\n{name}:")
    print(f"  Frame 0 dist:  {dist0:.4f} (should be < 0.3 if ball in hand)")
    print(f"  Frame 59 dist: {dist59:.4f} (release, can be larger)")

# Try searching all possible feature combinations
print("\n" + "="*70)
print("Searching all possible XYZ feature combinations...")
print("="*70)

best_combos = []

# Only check features that have reasonable range for position data
for x_feat in range(data.shape[1]):
    x_range = (data[:, x_feat].max() - data[:, x_feat].min()).item()
    if x_range < 0.1 or x_range > 5.0:
        continue

    for y_feat in range(data.shape[1]):
        if y_feat == x_feat:
            continue
        y_range = (data[:, y_feat].max() - data[:, y_feat].min()).item()
        if y_range < 0.1 or y_range > 5.0:
            continue

        for z_feat in range(data.shape[1]):
            if z_feat == x_feat or z_feat == y_feat:
                continue
            z_range = (data[:, z_feat].max() - data[:, z_feat].min()).item()
            if z_range < 0.1 or z_range > 5.0:
                continue

            # Test this combination
            bx = data[:, x_feat].numpy()
            by = data[:, y_feat].numpy()
            bz = data[:, z_feat].numpy()

            # Distance at frame 0 (should be close)
            dist0 = np.sqrt((bx[0] - data[0, l_wrist[0]].item())**2 +
                           (by[0] - data[0, l_wrist[1]].item())**2 +
                           (bz[0] - data[0, l_wrist[2]].item())**2)

            if dist0 < 0.25:
                # Distance at frame 59 (release)
                dist59 = np.sqrt((bx[59] - data[59, l_wrist[0]].item())**2 +
                                (by[59] - data[59, l_wrist[1]].item())**2 +
                                (bz[59] - data[59, l_wrist[2]].item())**2)

                best_combos.append({
                    'x_feat': x_feat,
                    'y_feat': y_feat,
                    'z_feat': z_feat,
                    'dist0': dist0,
                    'dist59': dist59
                })

if best_combos:
    print(f"\nFound {len(best_combos)} feature combinations with ball close to hand at frame 0:\n")
    best_combos = sorted(best_combos, key=lambda x: x['dist0'])
    for combo in best_combos[:5]:
        print(f"Features X={combo['x_feat']}, Y={combo['y_feat']}, Z={combo['z_feat']}:")
        print(f"  Frame 0 distance: {combo['dist0']:.4f}")
        print(f"  Frame 59 distance: {combo['dist59']:.4f}")
else:
    print("\nNo feature combinations found with ball close to hand!")
    print("The ball position may require a more complex transformation.")
