#!/usr/bin/env python3
"""
Verify ball position relative to hand/wrist positions
"""

import json
import torch
import numpy as np

# Load data
data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

# Load joint mapping
with open("visualisation/skillmimic_joint_map.json", 'r') as f:
    joint_map = json.load(f)['joint_map']

# Ball position (from analysis)
ball_x = data[:, 325].numpy()
ball_y = data[:, 119].numpy()
ball_z = data[:, 326].numpy()

# Contact flag
contact = data[:, 336].numpy()

print("="*70)
print("Ball Position vs Hand Positions Verification")
print("="*70)

# Check frame 0 (ball should be in hands)
frame = 0
print(f"\nFrame {frame} (ball in hand, contact={contact[frame]}):")

# Get wrist and hand positions
body_parts = ['L_Wrist', 'R_Wrist', 'L_Index1', 'R_Index1', 'L_Thumb1', 'R_Thumb1']

for part in body_parts:
    if part in joint_map:
        indices = joint_map[part]
        x = data[frame, indices[0]].item()
        y = data[frame, indices[1]].item()
        z = data[frame, indices[2]].item()
        print(f"  {part:12s}: x={x:7.4f}, y={y:7.4f}, z={z:7.4f}")

print(f"\n  Ball position:  x={ball_x[frame]:7.4f}, y={ball_y[frame]:7.4f}, z={ball_z[frame]:7.4f}")

# Calculate distances from ball to each hand part
print("\nDistances from ball to hands:")
for part in body_parts:
    if part in joint_map:
        indices = joint_map[part]
        x = data[frame, indices[0]].item()
        y = data[frame, indices[1]].item()
        z = data[frame, indices[2]].item()

        dist = np.sqrt((x - ball_x[frame])**2 + (y - ball_y[frame])**2 + (z - ball_z[frame])**2)
        print(f"  {part:12s}: {dist:.4f} units")

# Check release frame
release_frame = 59
print(f"\n{'='*70}")
print(f"Frame {release_frame} (release frame, contact={contact[release_frame]}):")

for part in body_parts:
    if part in joint_map:
        indices = joint_map[part]
        x = data[release_frame, indices[0]].item()
        y = data[release_frame, indices[1]].item()
        z = data[release_frame, indices[2]].item()
        print(f"  {part:12s}: x={x:7.4f}, y={y:7.4f}, z={z:7.4f}")

print(f"\n  Ball position:  x={ball_x[release_frame]:7.4f}, y={ball_y[release_frame]:7.4f}, z={ball_z[release_frame]:7.4f}")

print("\nDistances from ball to hands:")
for part in body_parts:
    if part in joint_map:
        indices = joint_map[part]
        x = data[release_frame, indices[0]].item()
        y = data[release_frame, indices[1]].item()
        z = data[release_frame, indices[2]].item()

        dist = np.sqrt((x - ball_x[release_frame])**2 + (y - ball_y[release_frame])**2 + (z - ball_z[release_frame])**2)
        print(f"  {part:12s}: {dist:.4f} units")

# Check coordinate ranges
print(f"\n{'='*70}")
print("Coordinate ranges across all frames:")
print("="*70)

print("\nSkeleton (from joint positions):")
all_x, all_y, all_z = [], [], []
for name, indices in joint_map.items():
    for frame in range(len(data)):
        all_x.append(data[frame, indices[0]].item())
        all_y.append(data[frame, indices[1]].item())
        all_z.append(data[frame, indices[2]].item())

print(f"  X range: {min(all_x):.4f} to {max(all_x):.4f} (range={max(all_x)-min(all_x):.4f})")
print(f"  Y range: {min(all_y):.4f} to {max(all_y):.4f} (range={max(all_y)-min(all_y):.4f})")
print(f"  Z range: {min(all_z):.4f} to {max(all_z):.4f} (range={max(all_z)-min(all_z):.4f})")

print("\nBall:")
print(f"  X range: {ball_x.min():.4f} to {ball_x.max():.4f} (range={ball_x.max()-ball_x.min():.4f})")
print(f"  Y range: {ball_y.min():.4f} to {ball_y.max():.4f} (range={ball_y.max()-ball_y.min():.4f})")
print(f"  Z range: {ball_z.min():.4f} to {ball_z.max():.4f} (range={ball_z.max()-ball_z.min():.4f})")

print("\n" + "="*70)
print("Analysis:")
if abs(ball_x[0] - 0.31) < 0.1 and abs(ball_y[0] - 1.35) < 0.5:
    print("✓ Ball coordinates appear to match skeleton coordinate system")
else:
    print("✗ Ball coordinates DO NOT match skeleton coordinate system")
    print("  Ball may be using a different coordinate frame or encoding")

print("="*70)
