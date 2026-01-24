import torch
import numpy as np
import pandas as pd

# Load the data
data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

print("Searching for ball-related features in 337-feature tensor...\n")

# Ball features likely have specific characteristics:
# 1. Ball position should have larger range of motion than body parts (falls from ~10ft)
# 2. Ball velocity should spike at release
# 3. Should be 3 consecutive features for x,y,z

# Calculate metrics for each feature
results = []
for i in range(data.shape[1]):
    feature = data[:, i]
    
    # Metrics
    value_range = feature.max().item() - feature.min().item()
    mean_val = feature.mean().item()
    std_val = feature.std().item()
    
    # Frame-to-frame change
    diffs = torch.diff(feature)
    max_diff = diffs.abs().max().item()
    mean_diff = diffs.abs().mean().item()
    
    results.append({
        'feature_idx': i,
        'min': feature.min().item(),
        'max': feature.max().item(),
        'range': value_range,
        'mean': mean_val,
        'std': std_val,
        'max_frame_diff': max_diff,
        'mean_frame_diff': mean_diff
    })

df = pd.DataFrame(results)

# Look for features with high range (ball falls/rises significantly)
print("Top 20 features by value range (ball position candidates):")
print(df.nlargest(20, 'range')[['feature_idx', 'min', 'max', 'range', 'std']])

print("\n\nTop 20 features by max frame-to-frame change (ball velocity candidates):")
print(df.nlargest(20, 'max_frame_diff')[['feature_idx', 'max_frame_diff', 'mean_frame_diff', 'range']])

# Look for groups of 3 consecutive features (x,y,z position)
print("\n\nLooking for groups of 3 consecutive features with similar characteristics...")
for i in range(0, min(50, data.shape[1]-2)):
    feat0, feat1, feat2 = data[:, i], data[:, i+1], data[:, i+2]
    
    # Check if they have similar range and high variance (position-like)
    ranges = [feat0.max()-feat0.min(), feat1.max()-feat1.min(), feat2.max()-feat2.min()]
    stds = [feat0.std(), feat1.std(), feat2.std()]
    
    # Ball position likely has high std and range
    if all(r > 0.3 for r in ranges) and all(s > 0.15 for s in stds):
        print(f"  Features [{i}, {i+1}, {i+2}]: ranges={[f'{r:.3f}' for r in ranges]}, stds={[f'{s:.3f}' for s in stds]}")

