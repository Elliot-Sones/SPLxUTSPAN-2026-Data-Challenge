#!/usr/bin/env python3
"""
Find actual ball position features by proximity to hands during contact
"""

import json
import torch
import numpy as np

# Load data
data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

# Load joint mapping
with open("visualisation/skillmimic_joint_map.json", 'r') as f:
    joint_map = json.load(f)['joint_map']

# Contact flag
contact = data[:, 336].numpy()

print("="*70)
print("Finding Ball Position Features")
print("="*70)

# Get average hand position during contact frames (frames 0-58)
contact_frames = np.where(contact == 1)[0]
print(f"\nContact frames: {contact_frames[0]} to {contact_frames[-1]}")

# Average hand position during contact
hand_parts = ['L_Wrist', 'R_Wrist', 'L_Index1', 'R_Index1']
avg_hand_x, avg_hand_y, avg_hand_z = [], [], []

for frame in contact_frames[:30]:  # Use first 30 contact frames
    for part in hand_parts:
        if part in joint_map:
            indices = joint_map[part]
            avg_hand_x.append(data[frame, indices[0]].item())
            avg_hand_y.append(data[frame, indices[1]].item())
            avg_hand_z.append(data[frame, indices[2]].item())

avg_hand_x = np.mean(avg_hand_x)
avg_hand_y = np.mean(avg_hand_y)
avg_hand_z = np.mean(avg_hand_z)

print(f"\nAverage hand position during contact:")
print(f"  X: {avg_hand_x:.4f}")
print(f"  Y: {avg_hand_y:.4f}")
print(f"  Z: {avg_hand_z:.4f}")

# Search for features that:
# 1. Are close to hand position during contact
# 2. Move away after release
# 3. Form triplets (x, y, z)

print(f"\n{'='*70}")
print("Searching for position features close to hands during contact...")
print("="*70)

candidates = []

for i in range(0, data.shape[1] - 2):
    # Check if this could be a position triplet
    feat_x = data[:, i]
    feat_y = data[:, i+1]
    feat_z = data[:, i+2]

    # Average position during contact
    contact_x = feat_x[contact_frames].mean().item()
    contact_y = feat_y[contact_frames].mean().item()
    contact_z = feat_z[contact_frames].mean().item()

    # Distance from average hand position
    dist = np.sqrt((contact_x - avg_hand_x)**2 +
                   (contact_y - avg_hand_y)**2 +
                   (contact_z - avg_hand_z)**2)

    # Check if it has good range (ball moves after release)
    range_x = (feat_x.max() - feat_x.min()).item()
    range_y = (feat_y.max() - feat_y.min()).item()
    range_z = (feat_z.max() - feat_z.min()).item()

    # Ball should be close to hands during contact and have good movement range
    if dist < 0.3 and range_y > 0.5:  # Y should have large range (vertical motion)
        candidates.append({
            'start_idx': i,
            'dist_to_hand': dist,
            'contact_pos': (contact_x, contact_y, contact_z),
            'ranges': (range_x, range_y, range_z),
            'mean_range': (range_x + range_y + range_z) / 3
        })

# Sort by distance to hand
candidates = sorted(candidates, key=lambda x: x['dist_to_hand'])

print(f"\nFound {len(candidates)} candidate triplets:\n")
for c in candidates[:10]:
    print(f"Features [{c['start_idx']}, {c['start_idx']+1}, {c['start_idx']+2}]:")
    print(f"  Distance to hand: {c['dist_to_hand']:.4f}")
    print(f"  Contact position: ({c['contact_pos'][0]:.3f}, {c['contact_pos'][1]:.3f}, {c['contact_pos'][2]:.3f})")
    print(f"  Ranges: ({c['ranges'][0]:.3f}, {c['ranges'][1]:.3f}, {c['ranges'][2]:.3f})")
    print()

if candidates:
    best = candidates[0]
    print("="*70)
    idx = best['start_idx']
    print(f"BEST CANDIDATE: Features [{idx}, {idx+1}, {idx+2}]")
    print("="*70)

    # Verify by plotting trajectory
    ball_x_feat = data[:, best['start_idx']].numpy()
    ball_y_feat = data[:, best['start_idx']+1].numpy()
    ball_z_feat = data[:, best['start_idx']+2].numpy()

    print(f"\nFrame-by-frame analysis:")
    print(f"{'Frame':<8} {'Contact':<8} {'X':<10} {'Y':<10} {'Z':<10} {'Dist to L_Wrist':<15}")
    print("-"*70)

    for frame in [0, 20, 40, 58, 59, 60, 70, 80, 90, 100]:
        if frame < len(data):
            # Get L_Wrist position
            l_wrist_indices = joint_map['L_Wrist']
            wrist_x = data[frame, l_wrist_indices[0]].item()
            wrist_y = data[frame, l_wrist_indices[1]].item()
            wrist_z = data[frame, l_wrist_indices[2]].item()

            ball_x = ball_x_feat[frame]
            ball_y = ball_y_feat[frame]
            ball_z = ball_z_feat[frame]

            dist = np.sqrt((ball_x - wrist_x)**2 + (ball_y - wrist_y)**2 + (ball_z - wrist_z)**2)

            c_flag = int(contact[frame])
            print(f"{frame:<8} {c_flag:<8} {ball_x:<10.4f} {ball_y:<10.4f} {ball_z:<10.4f} {dist:<15.4f}")
else:
    print("\nNo candidates found! Trying looser criteria...")

    # Try again with looser distance requirement
    for i in range(0, data.shape[1] - 2):
        feat_x = data[:, i]
        feat_y = data[:, i+1]
        feat_z = data[:, i+2]

        contact_x = feat_x[contact_frames].mean().item()
        contact_y = feat_y[contact_frames].mean().item()
        contact_z = feat_z[contact_frames].mean().item()

        dist = np.sqrt((contact_x - avg_hand_x)**2 +
                       (contact_y - avg_hand_y)**2 +
                       (contact_z - avg_hand_z)**2)

        range_y = (feat_y.max() - feat_y.min()).item()

        if dist < 0.5 and range_y > 1.0:
            print(f"Features [{i}, {i+1}, {i+2}]: dist={dist:.4f}, Y_range={range_y:.4f}")
