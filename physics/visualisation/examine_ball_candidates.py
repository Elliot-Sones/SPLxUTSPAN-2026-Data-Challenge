import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

print("BALL POSITION CANDIDATES:\n")

# Check features 325, 326 (highest range)
print("Features 325-326 (highest range):")
for i in [325, 326]:
    feat = data[:, i]
    print(f"  Feature {i}: min={feat.min():.3f}, max={feat.max():.3f}, range={feat.max()-feat.min():.3f}")

# Check if there's a feature 327 to complete the xyz triplet
if data.shape[1] > 327:
    feat = data[:, 327]
    print(f"  Feature 327: min={feat.min():.3f}, max={feat.max():.3f}, range={feat.max()-feat.min():.3f}")

# Check feature 119 (appears in both high range and high change)
print("\nFeature 119 (high range AND high frame-to-frame change):")
feat119 = data[:, 119]
print(f"  min={feat119.min():.3f}, max={feat119.max():.3f}, range={feat119.max()-feat119.min():.3f}")
print(f"  mean frame-to-frame change: {torch.diff(feat119).abs().mean():.3f}")

# Check surrounding features (118, 120, 121) - might be ball xyz
print("\nFeatures around 119 (potential ball position xyz):")
for i in range(118, 122):
    if i < data.shape[1]:
        feat = data[:, i]
        diff = torch.diff(feat).abs().mean()
        print(f"  Feature {i}: min={feat.min():.3f}, max={feat.max():.3f}, range={feat.max()-feat.min():.3f}, mean_diff={diff:.3f}")

# Plot features 119-121 to see if they look like ball trajectory
print("\nGenerating ball trajectory plot...")
fig = make_subplots(rows=3, cols=1, subplot_titles=['Feature 119 (X?)', 'Feature 120 (Y?)', 'Feature 121 (Z?)'])

for idx, i in enumerate([119, 120, 121]):
    if i < data.shape[1]:
        fig.add_trace(
            go.Scatter(x=list(range(104)), y=data[:, i].numpy(), mode='lines+markers', name=f'Feature {i}'),
            row=idx+1, col=1
        )

fig.update_layout(height=800, title_text="Potential Ball Position Features (119-121)")
fig.write_html("/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/visualisation/ball_position_candidates.html")
print("Saved plot to: visualisation/ball_position_candidates.html")

# Also check the last feature (336) which had suspicious 1.0 values
print("\nFeature 336 (last feature, suspicious values):")
feat336 = data[:, 336]
print(f"  Unique values: {torch.unique(feat336).numpy()}")
print(f"  Value counts: {[(v.item(), (feat336==v).sum().item()) for v in torch.unique(feat336)]}")

