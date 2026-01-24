import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')

# Plot top candidates: 325, 326 (highest range)
# and 119 (high range + high velocity)
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        'Feature 325 (highest range: 2.84)',
        'Feature 326 (2nd highest range: 1.93)',
        'Feature 119 (high range + velocity)',
        'Feature 336 (binary - ball released flag?)'
    ],
    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
)

features_to_plot = [325, 326, 119, 336]
for idx, feat_idx in enumerate(features_to_plot):
    feat = data[:, feat_idx]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(104)),
            y=feat.numpy(),
            mode='lines+markers',
            name=f'Feature {feat_idx}',
            marker=dict(size=4)
        ),
        row=idx+1, col=1
    )

fig.update_xaxes(title_text="Frame", row=4, col=1)
fig.update_layout(height=1000, title_text="Ball Position Candidates Analysis", showlegend=False)

fig.write_html("/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/visualisation/ball_candidates_detailed.html")
print("Saved detailed plot")

# Analyze feature 336 (binary) - might indicate ball contact/release
feat336 = data[:, 336]
release_frame = None
for i in range(len(feat336)-1):
    if feat336[i] == 0 and feat336[i+1] == 1:
        release_frame = i+1
        print(f"\nBall release transition at frame {release_frame} (0→1)")
        break

# Check what happens to features 325, 326, 119 around the release frame
if release_frame:
    print(f"\nFeature values around release frame {release_frame}:")
    for feat_idx in [119, 325, 326]:
        feat = data[:, feat_idx]
        window = 5
        start = max(0, release_frame - window)
        end = min(len(feat), release_frame + window)
        
        print(f"\n  Feature {feat_idx} (frames {start}-{end}):")
        for i in range(start, end):
            marker = " <-- RELEASE" if i == release_frame else ""
            print(f"    Frame {i}: {feat[i].item():.4f}{marker}")
        
        # Calculate velocity at release
        if release_frame > 0 and release_frame < len(feat) - 1:
            vel_before = feat[release_frame] - feat[release_frame-1]
            vel_after = feat[release_frame+1] - feat[release_frame]
            print(f"    Velocity before release: {vel_before.item():.4f}")
            print(f"    Velocity after release: {vel_after.item():.4f}")

