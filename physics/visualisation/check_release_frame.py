import torch

data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

release_frame = 59
print(f"BALL RELEASE AT FRAME {release_frame}")
print(f"Feature 336: 1→0 (ball in hand → ball released)\n")

for feat_idx in [119, 325, 326]:
    feat = data[:, feat_idx]
    
    print(f"\nFeature {feat_idx} around release (frames 54-64):")
    for i in range(54, 65):
        marker = " <-- RELEASE" if i == release_frame else ""
        print(f"  Frame {i:3d}: {feat[i].item():7.4f}{marker}")
    
    # Velocity before and after
    vel_before = feat[release_frame] - feat[release_frame-1]
    vel_after = feat[release_frame+1] - feat[release_frame]
    accel = vel_after - vel_before
    
    print(f"  Velocity before release: {vel_before.item():7.4f}")
    print(f"  Velocity at release:     {vel_after.item():7.4f}")
    print(f"  Acceleration:            {accel.item():7.4f}")

# Check if features 325-326 show parabolic motion after release (ball trajectory)
print("\n\nChecking for parabolic motion after release...")
print("If these are ball position, we'd expect:")
print("- Horizontal (X): roughly constant velocity")
print("- Vertical (Y/Z): constant acceleration (gravity)")

for feat_idx in [325, 326]:
    feat = data[release_frame:, feat_idx]  # After release
    if len(feat) > 2:
        velocities = torch.diff(feat)
        accelerations = torch.diff(velocities)
        
        print(f"\nFeature {feat_idx} after release:")
        print(f"  Mean velocity: {velocities.mean().item():.4f}")
        print(f"  Std velocity:  {velocities.std().item():.4f}")
        print(f"  Mean accel:    {accelerations.mean().item():.4f}")
        print(f"  Std accel:     {accelerations.std().item():.4f}")

