#!/usr/bin/env python3
"""
Find ball position by looking for parabolic motion after release
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

# Contact and release
contact = data[:, 336].numpy()
release_frame = np.where(np.diff(contact.astype(int)) == -1)[0][0] + 1

print("="*70)
print("Finding Ball Position via Parabolic Motion Analysis")
print("="*70)
print(f"Release frame: {release_frame}")

# After release, ball should follow physics:
# - X: roughly constant velocity (or slight deceleration)
# - Y: parabolic (rise then fall, or just fall)
# - Z: roughly constant velocity (forward motion)

# Look at frames 60-90 (after release)
post_release = slice(release_frame, min(release_frame + 30, len(data)))

print("\nSearching for features with parabolic Y motion after release...")

candidates = []

for i in range(data.shape[1]):
    feat = data[:, i].numpy()

    # Skip if no motion
    if feat.std() < 0.01:
        continue

    # Look at post-release motion
    post = feat[post_release]

    # Check if it has the right characteristics:
    # 1. Significant change after release
    release_change = abs(post[-1] - post[0])
    if release_change < 0.3:
        continue

    # 2. For Y (vertical): should rise then fall, or just fall
    # Check if there's a peak (derivative changes sign)
    diffs = np.diff(post)
    sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]

    # 3. Smooth motion (not jittery)
    smoothness = diffs.std()

    # For potential Y coordinate: should have peak or steady decrease
    has_peak = len(sign_changes) > 0

    candidates.append({
        'idx': i,
        'release_change': release_change,
        'has_peak': has_peak,
        'num_sign_changes': len(sign_changes),
        'smoothness': smoothness,
        'post_mean': post.mean(),
        'post_range': post.max() - post.min()
    })

# Sort by likelihood of being Y coordinate
y_candidates = [c for c in candidates if c['release_change'] > 0.5]
y_candidates = sorted(y_candidates, key=lambda x: x['post_range'], reverse=True)

print(f"\nFound {len(y_candidates)} Y candidates with significant vertical motion:\n")
for i, c in enumerate(y_candidates[:10], 1):
    print(f"{i}. Feature {c['idx']:3d}: range={c['post_range']:.3f}, peak={c['has_peak']}, "
          f"changes={c['num_sign_changes']}")

# Now for each Y candidate, find matching X and Z
print("\n" + "="*70)
print("Finding X and Z for each Y candidate...")
print("="*70)

best_triplets = []

for yc in y_candidates[:5]:
    y_idx = yc['idx']
    y_feat = data[:, y_idx].numpy()

    # Find X (should have some horizontal motion but less than Y)
    for x_idx in range(data.shape[1]):
        if x_idx == y_idx:
            continue
        x_feat = data[:, x_idx].numpy()
        x_post = x_feat[post_release]
        x_range = x_post.max() - x_post.min()

        # X should have some motion but less dramatic than Y
        if not (0.1 < x_range < 2.0):
            continue

        # Find Z (forward motion, steady)
        for z_idx in range(data.shape[1]):
            if z_idx == y_idx or z_idx == x_idx:
                continue
            z_feat = data[:, z_idx].numpy()
            z_post = z_feat[post_release]
            z_range = z_post.max() - z_post.min()

            # Z should have moderate motion
            if not (0.1 < z_range < 2.0):
                continue

            # Check if trajectory makes sense
            # After release, ball should generally move up/forward initially
            # Y should increase initially (for a shot)
            y_initial_change = y_feat[release_frame+5] - y_feat[release_frame]

            # Store this combination
            best_triplets.append({
                'x': x_idx,
                'y': y_idx,
                'z': z_idx,
                'y_initial_rise': y_initial_change,
                'x_range': x_range,
                'y_range': yc['post_range'],
                'z_range': z_range,
                'y_has_peak': yc['has_peak']
            })

# Sort by Y initial rise (shots should rise initially)
best_triplets = sorted(best_triplets, key=lambda x: abs(x['y_initial_rise']), reverse=True)

print(f"\nTop 10 triplet combinations:\n")
for i, t in enumerate(best_triplets[:10], 1):
    print(f"{i}. X={t['x']}, Y={t['y']}, Z={t['z']}")
    print(f"   Y rises by {t['y_initial_rise']:.3f} after release")
    print(f"   Ranges: X={t['x_range']:.3f}, Y={t['y_range']:.3f}, Z={t['z_range']:.3f}")
    print(f"   Y has peak: {t['y_has_peak']}")

    # Show trajectory snippet
    x_vals = data[release_frame:release_frame+10, t['x']].numpy()
    y_vals = data[release_frame:release_frame+10, t['y']].numpy()
    z_vals = data[release_frame:release_frame+10, t['z']].numpy()
    print(f"   First 10 frames after release:")
    for j in range(min(5, len(x_vals))):
        print(f"     Frame {release_frame+j}: ({x_vals[j]:.3f}, {y_vals[j]:.3f}, {z_vals[j]:.3f})")
    print()

if best_triplets:
    # Visualize top 3 candidates
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, t in enumerate(best_triplets[:3]):
        ax = axes[idx]

        x_vals = data[:, t['x']].numpy()
        y_vals = data[:, t['y']].numpy()

        # Plot trajectory
        ax.plot(x_vals[:release_frame], y_vals[:release_frame], 'o-',
                label='In hand', color='orange', markersize=3)
        ax.plot(x_vals[release_frame:], y_vals[release_frame:], 'o-',
                label='After release', color='blue', markersize=3)
        ax.axvline(x=x_vals[release_frame], color='red', linestyle='--',
                   label=f'Release (frame {release_frame})')

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'Candidate {idx+1}: Features X={t["x"]}, Y={t["y"]}, Z={t["z"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualisation/ball_trajectory_candidates.png', dpi=150)
    print(f"\nSaved trajectory plots to: visualisation/ball_trajectory_candidates.png")

    # Print best candidate
    best = best_triplets[0]
    print("\n" + "="*70)
    print(f"BEST CANDIDATE: X={best['x']}, Y={best['y']}, Z={best['z']}")
    print(f"  Y rises {best['y_initial_rise']:.3f} units after release")
    print(f"  Motion ranges: X={best['x_range']:.3f}, Y={best['y_range']:.3f}, Z={best['z_range']:.3f}")
    print("="*70)
else:
    print("\nNo good candidates found!")
