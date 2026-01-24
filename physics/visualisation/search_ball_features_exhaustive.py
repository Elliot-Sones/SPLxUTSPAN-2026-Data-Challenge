#!/usr/bin/env python3
"""
Exhaustive search for ball position features by analyzing contact frames
"""

import json
import torch
import numpy as np

# Load data
data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

with open("visualisation/skillmimic_joint_map.json", 'r') as f:
    joint_map = json.load(f)['joint_map']

# Contact flag
contact = data[:, 336].numpy()
contact_frames = np.where(contact == 1)[0]
release_frame = np.where(np.diff(contact.astype(int)) == -1)[0]
if len(release_frame) > 0:
    release_frame = release_frame[0] + 1
else:
    release_frame = 59

print("="*70)
print("Exhaustive Ball Position Feature Search")
print("="*70)
print(f"\nContact frames: 0-{contact_frames[-1]}")
print(f"Release frame: {release_frame}")

# Get hand position range during contact
hand_parts = ['L_Wrist', 'R_Wrist']
hand_x_range = []
hand_y_range = []
hand_z_range = []

for frame in contact_frames:
    for part in hand_parts:
        indices = joint_map[part]
        hand_x_range.append(data[frame, indices[0]].item())
        hand_y_range.append(data[frame, indices[1]].item())
        hand_z_range.append(data[frame, indices[2]].item())

hand_x_min, hand_x_max = min(hand_x_range), max(hand_x_range)
hand_y_min, hand_y_max = min(hand_y_range), max(hand_y_range)
hand_z_min, hand_z_max = min(hand_z_range), max(hand_z_range)

print(f"\nHand position range during contact:")
print(f"  X: [{hand_x_min:.3f}, {hand_x_max:.3f}]")
print(f"  Y: [{hand_y_min:.3f}, {hand_y_max:.3f}]")
print(f"  Z: [{hand_z_min:.3f}, {hand_z_max:.3f}]")

# Strategy: Find features where:
# 1. During contact: values are within hand range
# 2. After release: values show parabolic motion (Y decreases then increases for shot)
# 3. Form reasonable triplets

print("\n" + "="*70)
print("Searching individual features...")
print("="*70)

candidates_x = []
candidates_y = []
candidates_z = []

for i in range(data.shape[1]):
    feat = data[:, i].numpy()

    # Check if feature is within hand range during contact
    contact_values = feat[contact_frames]
    contact_mean = contact_values.mean()
    contact_std = contact_values.std()

    # Check post-release motion
    release_values = feat[release_frame:release_frame+20] if release_frame+20 < len(feat) else feat[release_frame:]
    if len(release_values) > 0:
        release_range = release_values.max() - release_values.min()
    else:
        release_range = 0

    # X candidates: within hand X range during contact, moderate motion after
    if hand_x_min - 0.3 <= contact_mean <= hand_x_max + 0.3 and release_range > 0.1:
        candidates_x.append({
            'idx': i,
            'contact_mean': contact_mean,
            'contact_std': contact_std,
            'release_range': release_range,
            'dist_to_hand_x': min(abs(contact_mean - hand_x_min), abs(contact_mean - hand_x_max))
        })

    # Y candidates: within hand Y range, large motion after (ball rises then falls)
    if hand_y_min - 0.3 <= contact_mean <= hand_y_max + 0.3 and release_range > 0.5:
        candidates_y.append({
            'idx': i,
            'contact_mean': contact_mean,
            'contact_std': contact_std,
            'release_range': release_range,
            'dist_to_hand_y': min(abs(contact_mean - hand_y_min), abs(contact_mean - hand_y_max))
        })

    # Z candidates: within hand Z range, moderate motion after
    if hand_z_min - 0.3 <= contact_mean <= hand_z_max + 0.3 and release_range > 0.2:
        candidates_z.append({
            'idx': i,
            'contact_mean': contact_mean,
            'contact_std': contact_std,
            'release_range': release_range,
            'dist_to_hand_z': min(abs(contact_mean - hand_z_min), abs(contact_mean - hand_z_max))
        })

# Sort by proximity to hand range
candidates_x = sorted(candidates_x, key=lambda x: x['dist_to_hand_x'])
candidates_y = sorted(candidates_y, key=lambda x: x['dist_to_hand_y'])
candidates_z = sorted(candidates_z, key=lambda x: x['dist_to_hand_z'])

print(f"\nFound {len(candidates_x)} X candidates, {len(candidates_y)} Y candidates, {len(candidates_z)} Z candidates")

print("\nTop 5 X candidates:")
for c in candidates_x[:5]:
    print(f"  Feature {c['idx']:3d}: contact_mean={c['contact_mean']:.3f}, release_range={c['release_range']:.3f}")

print("\nTop 5 Y candidates:")
for c in candidates_y[:5]:
    print(f"  Feature {c['idx']:3d}: contact_mean={c['contact_mean']:.3f}, release_range={c['release_range']:.3f}")

print("\nTop 5 Z candidates:")
for c in candidates_z[:5]:
    print(f"  Feature {c['idx']:3d}: contact_mean={c['contact_mean']:.3f}, release_range={c['release_range']:.3f}")

# Try best combinations
print("\n" + "="*70)
print("Testing best feature combinations...")
print("="*70)

best_combos = []

for cx in candidates_x[:3]:
    for cy in candidates_y[:3]:
        for cz in candidates_z[:3]:
            if cx['idx'] == cy['idx'] or cx['idx'] == cz['idx'] or cy['idx'] == cz['idx']:
                continue

            # Test at multiple contact frames
            total_dist = 0
            for test_frame in [0, 10, 20, 30, 40, 50]:
                if test_frame >= len(contact_frames):
                    break
                frame = contact_frames[test_frame]

                ball_x = data[frame, cx['idx']].item()
                ball_y = data[frame, cy['idx']].item()
                ball_z = data[frame, cz['idx']].item()

                # Average distance to both wrists
                dist_sum = 0
                for part in hand_parts:
                    indices = joint_map[part]
                    wx = data[frame, indices[0]].item()
                    wy = data[frame, indices[1]].item()
                    wz = data[frame, indices[2]].item()

                    dist = np.sqrt((ball_x - wx)**2 + (ball_y - wy)**2 + (ball_z - wz)**2)
                    dist_sum += dist

                total_dist += dist_sum / len(hand_parts)

            avg_dist = total_dist / 6

            best_combos.append({
                'x': cx['idx'],
                'y': cy['idx'],
                'z': cz['idx'],
                'avg_contact_dist': avg_dist
            })

best_combos = sorted(best_combos, key=lambda x: x['avg_contact_dist'])

print(f"\nTop 10 feature combinations (by avg distance to hands during contact):\n")
for i, combo in enumerate(best_combos[:10], 1):
    print(f"{i}. Features X={combo['x']}, Y={combo['y']}, Z={combo['z']}")
    print(f"   Avg distance to hands: {combo['avg_contact_dist']:.3f}")

    # Show a few sample positions
    print(f"   Frame 0:  ({data[0, combo['x']]:.3f}, {data[0, combo['y']]:.3f}, {data[0, combo['z']]:.3f})")
    print(f"   Frame {release_frame}: ({data[release_frame, combo['x']]:.3f}, {data[release_frame, combo['y']]:.3f}, {data[release_frame, combo['z']]:.3f})")
    print()

if best_combos:
    best = best_combos[0]
    print("="*70)
    print(f"BEST COMBINATION: X={best['x']}, Y={best['y']}, Z={best['z']}")
    print(f"Average distance to hands during contact: {best['avg_contact_dist']:.3f}")
    print("="*70)

    # Save for use in visualization
    with open("visualisation/ball_features.json", 'w') as f:
        json.dump({
            'x_feature': best['x'],
            'y_feature': best['y'],
            'z_feature': best['z'],
            'avg_contact_distance': best['avg_contact_dist'],
            'contact_flag': 336,
            'release_frame': int(release_frame)
        }, f, indent=2)

    print("\nSaved to visualisation/ball_features.json")
