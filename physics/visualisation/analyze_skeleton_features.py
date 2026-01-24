import torch
import numpy as np
import pandas as pd

data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

print("Analyzing features to identify skeleton joints...\n")

# Joint positions should have:
# 1. Moderate range (not tiny, not huge)
# 2. Smooth temporal evolution
# 3. Likely grouped in triplets (x, y, z)

results = []
for i in range(data.shape[1]):
    feat = data[:, i]
    
    # Calculate metrics
    value_range = feat.max().item() - feat.min().item()
    std_val = feat.std().item()
    
    # Temporal smoothness (low second derivative = position-like)
    if len(feat) > 2:
        first_deriv = torch.diff(feat)
        second_deriv = torch.diff(first_deriv)
        smoothness = second_deriv.abs().mean().item()
    else:
        smoothness = 0
    
    # Mean frame-to-frame change
    mean_change = torch.diff(feat).abs().mean().item()
    
    results.append({
        'feature': i,
        'range': value_range,
        'std': std_val,
        'smoothness': smoothness,
        'mean_change': mean_change
    })

df = pd.DataFrame(results)

# Position-like features: range > 0.3, std > 0.05, smoothness < 0.02
position_candidates = df[
    (df['range'] > 0.3) & 
    (df['std'] > 0.05) & 
    (df['smoothness'] < 0.02)
]['feature'].tolist()

print(f"Found {len(position_candidates)} position-like features")
print(f"\nPosition candidates: {position_candidates[:30]}...")

# Check for triplet patterns (consecutive groups of 3)
print("\n\nLooking for triplet patterns (x,y,z groups):")
triplets = []
i = 0
while i < len(position_candidates) - 2:
    f1, f2, f3 = position_candidates[i], position_candidates[i+1], position_candidates[i+2]
    
    # Check if consecutive
    if f2 == f1 + 1 and f3 == f2 + 1:
        triplets.append((f1, f2, f3))
        print(f"  Triplet: [{f1}, {f2}, {f3}]")
        i += 3
    else:
        i += 1

print(f"\nFound {len(triplets)} potential joint triplets (x,y,z)")

# Analyze first few triplets
print("\n\nAnalyzing first 10 triplets:")
for idx, (f1, f2, f3) in enumerate(triplets[:10]):
    x_range = data[:, f1].max() - data[:, f1].min()
    y_range = data[:, f2].max() - data[:, f2].min()
    z_range = data[:, f3].max() - data[:, f3].min()
    
    print(f"Joint {idx} (features {f1}-{f3}):")
    print(f"  X range: {x_range:.3f}, Y range: {y_range:.3f}, Z range: {z_range:.3f}")

# Check if there are exactly 52 joints (156 features / 3)
print(f"\n\nTotal triplets found: {len(triplets)}")
print(f"Expected for 52 joints: 52 triplets (156 features)")

# Save triplet indices
with open('/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/visualisation/skillmimic_joint_indices.txt', 'w') as f:
    f.write("SkillMimic Joint Position Indices (triplets of x,y,z)\n")
    f.write("="*60 + "\n\n")
    for idx, (f1, f2, f3) in enumerate(triplets):
        f.write(f"Joint {idx:2d}: features [{f1:3d}, {f2:3d}, {f3:3d}]\n")

print("\nSaved joint indices to: skillmimic_joint_indices.txt")

