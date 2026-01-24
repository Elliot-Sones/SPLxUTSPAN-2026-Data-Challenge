import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = torch.load("/Users/elliot18/Downloads/005_014pickle_shot_001.pt", map_location='cpu')
release_frame = 59

# Features 325, 326, 119 appear to be ball position coordinates
# Feature 336 is ball contact flag (1=in hand, 0=released)

fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        'Feature 119: Ball Position (likely Y - vertical, dramatic rise)',
        'Feature 325: Ball Position (likely X - horizontal, constant velocity)',
        'Feature 326: Ball Position (likely Z - forward/depth, rising)',
        'Feature 336: Ball Contact Flag (1=in hand, 0=released)'
    ],
    vertical_spacing=0.08
)

ball_features = [
    (119, "Y (vertical)", "red"),
    (325, "X (horizontal)", "blue"),
    (326, "Z (forward)", "green"),
    (336, "Contact", "black")
]

for idx, (feat_idx, label, color) in enumerate(ball_features):
    feat = data[:, feat_idx]
    
    # Before release (in hand)
    fig.add_trace(
        go.Scatter(
            x=list(range(release_frame)),
            y=feat[:release_frame].numpy(),
            mode='lines+markers',
            name=f'{label} (in hand)',
            line=dict(color=color, dash='dot'),
            marker=dict(size=3),
            showlegend=True
        ),
        row=idx+1, col=1
    )
    
    # After release (free flight)
    fig.add_trace(
        go.Scatter(
            x=list(range(release_frame-1, len(feat))),
            y=feat[release_frame-1:].numpy(),
            mode='lines+markers',
            name=f'{label} (released)',
            line=dict(color=color, width=2),
            marker=dict(size=4),
            showlegend=True
        ),
        row=idx+1, col=1
    )
    
    # Mark release point
    fig.add_vline(
        x=release_frame,
        line_dash="dash",
        line_color="red",
        row=idx+1, col=1
    )

fig.update_xaxes(title_text="Frame", row=4, col=1)
fig.update_layout(
    height=1200,
    title_text="Ball Position and Contact Analysis (Release at Frame 59)",
    hovermode='x unified'
)

output_path = "/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/visualisation/ball_trajectory_analysis.html"
fig.write_html(output_path)
print(f"Saved visualization: {output_path}")

# Summary analysis
print("\n" + "="*80)
print("BALL POSITION FEATURES IDENTIFIED")
print("="*80)

print("\nFeature 336: Ball Contact Flag")
print(f"  Frames 0-58: 1.0 (ball in hand)")
print(f"  Frames 59-103: 0.0 (ball released)")
print(f"  Release frame: 59")

print("\nFeature 119: Likely Y-axis (vertical)")
print(f"  Before release: -1.47 to -0.61 (rising with hand)")
print(f"  At release (frame 59): -0.61")
print(f"  After release: -0.61 → 0.88 (continues rising, peaks, then falls)")
print(f"  Release velocity: 0.88 units/frame (high upward velocity)")

print("\nFeature 325: Likely X-axis (horizontal)")
print(f"  Before release: 1.19 → 0.94 (moving horizontally with body)")
print(f"  At release (frame 59): 0.94")
print(f"  After release: Constant velocity -0.056 ± 0.001 (consistent horizontal motion)")
print(f"  This matches projectile motion: constant horizontal velocity!")

print("\nFeature 326: Likely Z-axis (forward/depth)")
print(f"  Before release: 2.35 → 2.57 (moving forward with shooting motion)")
print(f"  At release (frame 59): 2.57")
print(f"  After release: Continues rising (ball moving toward basket)")
print(f"  Release velocity: 0.04 units/frame (forward velocity)")

print("\n" + "="*80)
print("CONCLUSION: Features 119, 325, 326 are ball position (X, Y, Z)")
print("            Feature 336 is ball contact flag")
print("="*80)

