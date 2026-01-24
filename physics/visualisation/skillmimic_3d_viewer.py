#!/usr/bin/env python3
"""
SkillMimic 3D Visualization
Creates interactive 3D animation similar to SPL ball_viewer.py but for SkillMimic data

Features:
- Ball trajectory visualization
- Release frame detection (from feature 336)
- Frame-by-frame animation with slider
- Ball colored by phase (in-hand vs released)
"""

import argparse
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Ball radius (normalized units - SkillMimic uses normalized coordinates)
BALL_RADIUS = 0.03


def load_skillmimic_file(filepath: str) -> torch.Tensor:
    """Load SkillMimic .pt file"""
    data = torch.load(filepath, map_location='cpu')
    if not isinstance(data, torch.Tensor):
        raise RuntimeError(f"Expected torch.Tensor, got {type(data)}")
    return data


def extract_ball_data(data: torch.Tensor) -> dict:
    """
    Extract ball-related features from SkillMimic tensor

    Returns:
        dict with 'position', 'contact_flag', 'release_frame'
    """
    # Ball position: features 119 (Y-vertical), 325 (X-horizontal), 326 (Z-forward)
    # Note: Reorder to [X, Y, Z] for standard 3D plotting
    ball_x = data[:, 325].numpy()  # Horizontal
    ball_y = data[:, 119].numpy()  # Vertical
    ball_z = data[:, 326].numpy()  # Forward/depth

    ball_position = np.stack([ball_x, ball_y, ball_z], axis=1)  # (num_frames, 3)

    # Contact flag: feature 336 (1=in hand, 0=released)
    contact_flag = data[:, 336].numpy()

    # Find release frame (transition from 1→0)
    release_frame = None
    for i in range(len(contact_flag) - 1):
        if contact_flag[i] == 1 and contact_flag[i+1] == 0:
            release_frame = i + 1
            break

    # If no transition found, assume released throughout
    if release_frame is None:
        release_frame = 0

    return {
        'position': ball_position,
        'contact_flag': contact_flag,
        'release_frame': release_frame,
        'num_frames': len(data)
    }


def create_ball_sphere_trace(ball_pos, in_hand=True, is_release=False, num_points=15):
    """
    Create sphere mesh for basketball

    Args:
        ball_pos: [x, y, z] position
        in_hand: Whether ball is in hand (affects color)
        is_release: Whether this is the release frame
        num_points: Sphere resolution
    """
    # Generate sphere surface
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = BALL_RADIUS * np.outer(np.cos(u), np.sin(v)) + ball_pos[0]
    y = BALL_RADIUS * np.outer(np.sin(u), np.sin(v)) + ball_pos[1]
    z = BALL_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v)) + ball_pos[2]

    # Flatten for scatter3d
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Color based on phase
    if is_release:
        color = '#FFD700'  # Gold (release frame)
        size = 5
        opacity = 1.0
    elif in_hand:
        color = '#FF8C00'  # Orange (in hand)
        size = 3
        opacity = 0.7
    else:
        color = '#4169E1'  # Royal blue (in flight)
        size = 2
        opacity = 0.5

    return go.Scatter3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        mode='markers',
        marker=dict(size=size, color=color, opacity=opacity),
        name='Basketball',
        hoverinfo='text',
        text=f"Ball<br>({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f})"
    )


def create_ball_trajectory_trace(trajectory, current_frame, release_frame):
    """
    Create trajectory trail showing ball path

    Args:
        trajectory: Array of positions (num_frames, 3)
        current_frame: Current frame index
        release_frame: Release frame index
    """
    # Show trajectory up to current frame
    traj = trajectory[:current_frame+1]

    if len(traj) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode='lines', showlegend=False)

    # Color gradient based on release
    colors = []
    for i in range(len(traj)):
        if i < release_frame:
            colors.append(0)  # Pre-release (blue)
        else:
            colors.append((i - release_frame) / max(1, current_frame - release_frame))  # Post-release (red)

    return go.Scatter3d(
        x=traj[:, 0],
        y=traj[:, 1],
        z=traj[:, 2],
        mode='lines+markers',
        line=dict(
            width=4,
            color=colors,
            colorscale=[[0, '#FF8C00'], [0.5, '#FFD700'], [1, '#4169E1']],
            showscale=False
        ),
        marker=dict(size=2),
        name='Ball Trajectory',
        hoverinfo='text',
        text=[f"Frame {i}" for i in range(len(traj))]
    )


def create_court_reference(ball_data):
    """Create reference lines for court (origin, axes)"""
    traces = []

    # Get data bounds
    positions = ball_data['position']
    x_range = [positions[:, 0].min() - 0.5, positions[:, 0].max() + 0.5]
    y_range = [positions[:, 1].min() - 0.5, positions[:, 1].max() + 0.5]
    z_range = [positions[:, 2].min() - 0.5, positions[:, 2].max() + 0.5]

    # Origin point
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='x'),
        name='Origin',
        hoverinfo='text',
        text='Origin (0,0,0)'
    ))

    # Ground plane grid
    grid_points = np.linspace(-2, 2, 9)
    for x in grid_points:
        traces.append(go.Scatter3d(
            x=[x, x],
            y=[y_range[0], y_range[1]],
            z=[z_range[0], z_range[0]],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    for y in grid_points:
        traces.append(go.Scatter3d(
            x=[x_range[0], x_range[1]],
            y=[y, y],
            z=[z_range[0], z_range[0]],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))

    return traces


def create_interactive_visualization(data, ball_data, filepath, output_path='skillmimic_3d_viewer.html'):
    """Create interactive 3D visualization with animation"""

    num_frames = ball_data['num_frames']
    release_frame = ball_data['release_frame']
    trajectory = ball_data['position']
    contact_flags = ball_data['contact_flag']

    # Create figure
    fig = go.Figure()

    # Create frames for animation
    frames = []
    for frame_idx in range(num_frames):
        traces = []

        # Add court reference
        traces.extend(create_court_reference(ball_data))

        # Add ball
        ball_pos = trajectory[frame_idx]
        in_hand = contact_flags[frame_idx] == 1
        is_release = (frame_idx == release_frame)
        ball_trace = create_ball_sphere_trace(ball_pos, in_hand, is_release)
        traces.append(ball_trace)

        # Add trajectory trail
        traj_trace = create_ball_trajectory_trace(trajectory, frame_idx, release_frame)
        traces.append(traj_trace)

        # Title with release indication
        release_indicator = " [RELEASE]" if is_release else ""
        phase = "In Hand" if in_hand else "Released"
        title = (f"Frame {frame_idx}/{num_frames-1}{release_indicator}<br>"
                f"Phase: {phase}<br>"
                f"Position: ({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f})")

        frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            layout=go.Layout(title_text=title)
        ))

    # Add initial frame traces
    initial_traces = create_court_reference(ball_data)
    initial_ball = create_ball_sphere_trace(trajectory[0], contact_flags[0] == 1, False)
    initial_traces.append(initial_ball)

    for trace in initial_traces:
        fig.add_trace(trace)

    # Add frames to figure
    fig.frames = frames

    # Create slider
    sliders = [dict(
        steps=[dict(
            args=[[f.name],
                  dict(frame=dict(duration=0, redraw=True),
                       mode='immediate',
                       transition=dict(duration=0))],
            method='animate',
            label=str(k)
        ) for k, f in enumerate(fig.frames)],
        active=0,
        y=0,
        x=0.1,
        currentvalue=dict(
            prefix='Frame: ',
            visible=True,
            xanchor='center'
        ),
        len=0.9,
        xanchor='left',
        yanchor='top',
        transition=dict(duration=0)
    )]

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"SkillMimic Basketball Shooting Motion<br>"
                 f"File: {filepath.split('/')[-1]}<br>"
                 f"<b>Release Frame: {release_frame}</b>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (horizontal)',
            yaxis_title='Y (vertical)',
            zaxis_title='Z (forward)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2)
            )
        ),
        sliders=sliders,
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0.05,
                x=0.05,
                xanchor='left',
                yanchor='bottom',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=33, redraw=True),  # ~30 FPS
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        height=900,
        showlegend=True
    )

    # Save
    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")


def create_ball_metrics_plot(ball_data, output_path='skillmimic_ball_metrics.html'):
    """Create plots showing ball trajectory metrics over time"""

    num_frames = ball_data['num_frames']
    release_frame = ball_data['release_frame']
    trajectory = ball_data['position']
    contact_flags = ball_data['contact_flag']

    frames = np.arange(num_frames)

    # Calculate velocities
    velocities = np.diff(trajectory, axis=0)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    velocity_magnitudes = np.concatenate([[0], velocity_magnitudes])  # Pad to match frames

    # Calculate accelerations
    accelerations = np.diff(velocities, axis=0)
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)
    accel_magnitudes = np.concatenate([[0, 0], accel_magnitudes])  # Pad to match frames

    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=(
            'Ball Position Components',
            'Ball Velocity (Frame-to-Frame)',
            'Ball Acceleration',
            'Ball Height (Y-axis)',
            'Ball Contact Flag'
        ),
        vertical_spacing=0.06
    )

    # Plot 1: Position components
    fig.add_trace(
        go.Scatter(x=frames, y=trajectory[:, 0], mode='lines', name='X (horizontal)',
                  line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=frames, y=trajectory[:, 1], mode='lines', name='Y (vertical)',
                  line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=frames, y=trajectory[:, 2], mode='lines', name='Z (forward)',
                  line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_vline(x=release_frame, line_dash="dash", line_color="orange",
                 annotation_text="Release", row=1, col=1)

    # Plot 2: Velocity magnitude
    fig.add_trace(
        go.Scatter(x=frames, y=velocity_magnitudes, mode='lines', name='Velocity',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    fig.add_vline(x=release_frame, line_dash="dash", line_color="orange", row=2, col=1)

    # Plot 3: Acceleration magnitude
    fig.add_trace(
        go.Scatter(x=frames, y=accel_magnitudes, mode='lines', name='Acceleration',
                  line=dict(color='brown', width=2)),
        row=3, col=1
    )
    fig.add_vline(x=release_frame, line_dash="dash", line_color="orange", row=3, col=1)

    # Plot 4: Height (Y) trajectory
    fig.add_trace(
        go.Scatter(x=frames, y=trajectory[:, 1], mode='lines', name='Height',
                  line=dict(color='green', width=2)),
        row=4, col=1
    )
    fig.add_vline(x=release_frame, line_dash="dash", line_color="orange", row=4, col=1)

    # Find peak height after release
    post_release_y = trajectory[release_frame:, 1]
    if len(post_release_y) > 0:
        peak_idx = release_frame + np.argmax(post_release_y)
        fig.add_vline(x=peak_idx, line_dash="dot", line_color="darkgreen",
                     annotation_text="Peak", row=4, col=1)

    # Plot 5: Contact flag
    fig.add_trace(
        go.Scatter(x=frames, y=contact_flags, mode='lines', name='Contact',
                  line=dict(color='black', width=2)),
        row=5, col=1
    )
    fig.add_vline(x=release_frame, line_dash="dash", line_color="orange", row=5, col=1)

    # Update axes
    fig.update_xaxes(title_text="Frame", row=5, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Velocity", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration", row=3, col=1)
    fig.update_yaxes(title_text="Height", row=4, col=1)
    fig.update_yaxes(title_text="Contact (1=In Hand)", row=5, col=1)

    fig.update_layout(
        title_text=f"SkillMimic Ball Trajectory Metrics (Release: Frame {release_frame})",
        height=1400,
        showlegend=True
    )

    fig.write_html(output_path)
    print(f"Metrics plot saved to: {output_path}")


def create_trajectory_comparison(ball_data, output_path='skillmimic_trajectory_comparison.html'):
    """Create 2D trajectory plots (side view, top view)"""

    trajectory = ball_data['position']
    release_frame = ball_data['release_frame']
    contact_flags = ball_data['contact_flag']

    # Split into pre-release and post-release
    pre_release = trajectory[:release_frame+1]
    post_release = trajectory[release_frame:]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Side View (X-Y)', 'Top View (X-Z)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    # Side view (X-Y)
    fig.add_trace(
        go.Scatter(x=pre_release[:, 0], y=pre_release[:, 1],
                  mode='lines+markers', name='Pre-Release',
                  line=dict(color='orange', width=3),
                  marker=dict(size=4)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=post_release[:, 0], y=post_release[:, 1],
                  mode='lines+markers', name='Post-Release',
                  line=dict(color='blue', width=3),
                  marker=dict(size=4)),
        row=1, col=1
    )
    # Mark release point
    fig.add_trace(
        go.Scatter(x=[trajectory[release_frame, 0]], y=[trajectory[release_frame, 1]],
                  mode='markers', name='Release Point',
                  marker=dict(size=15, color='gold', symbol='star')),
        row=1, col=1
    )

    # Top view (X-Z)
    fig.add_trace(
        go.Scatter(x=pre_release[:, 0], y=pre_release[:, 2],
                  mode='lines+markers', name='Pre-Release',
                  line=dict(color='orange', width=3),
                  marker=dict(size=4),
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=post_release[:, 0], y=post_release[:, 2],
                  mode='lines+markers', name='Post-Release',
                  line=dict(color='blue', width=3),
                  marker=dict(size=4),
                  showlegend=False),
        row=1, col=2
    )
    # Mark release point
    fig.add_trace(
        go.Scatter(x=[trajectory[release_frame, 0]], y=[trajectory[release_frame, 2]],
                  mode='markers', name='Release Point',
                  marker=dict(size=15, color='gold', symbol='star'),
                  showlegend=False),
        row=1, col=2
    )

    fig.update_xaxes(title_text="X (horizontal)", row=1, col=1)
    fig.update_yaxes(title_text="Y (vertical)", row=1, col=1)
    fig.update_xaxes(title_text="X (horizontal)", row=1, col=2)
    fig.update_yaxes(title_text="Z (forward)", row=1, col=2)

    fig.update_layout(
        title_text=f"Ball Trajectory Views (Release: Frame {release_frame})",
        height=600,
        showlegend=True
    )

    fig.write_html(output_path)
    print(f"Trajectory comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D interactive visualization of SkillMimic basketball data"
    )
    parser.add_argument("filepath", help="Path to .pt file")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    print("="*70)
    print("SkillMimic 3D Visualization")
    print("="*70)
    print(f"Loading: {args.filepath}")

    # Load data
    data = load_skillmimic_file(args.filepath)
    print(f"Data shape: {data.shape}")

    # Extract ball data
    ball_data = extract_ball_data(data)
    print(f"\nBall Data:")
    print(f"  Num frames: {ball_data['num_frames']}")
    print(f"  Release frame: {ball_data['release_frame']}")
    print(f"  Ball position range:")
    print(f"    X: [{ball_data['position'][:, 0].min():.3f}, {ball_data['position'][:, 0].max():.3f}]")
    print(f"    Y: [{ball_data['position'][:, 1].min():.3f}, {ball_data['position'][:, 1].max():.3f}]")
    print(f"    Z: [{ball_data['position'][:, 2].min():.3f}, {ball_data['position'][:, 2].max():.3f}]")

    # Calculate trajectory statistics
    release_frame = ball_data['release_frame']
    trajectory = ball_data['position']

    if release_frame > 0:
        pre_release_dist = np.linalg.norm(trajectory[release_frame] - trajectory[0])
        print(f"\n  Distance traveled before release: {pre_release_dist:.3f} units")

    if release_frame < len(trajectory) - 1:
        post_release_traj = trajectory[release_frame:]
        max_height = post_release_traj[:, 1].max()
        print(f"  Max height after release: {max_height:.3f}")
        print(f"  Height gain from release: {max_height - trajectory[release_frame, 1]:.3f}")

    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)

    # Create visualizations
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_interactive_visualization(
        data, ball_data, args.filepath,
        output_dir / "skillmimic_3d_viewer.html"
    )
    create_ball_metrics_plot(
        ball_data,
        output_dir / "skillmimic_ball_metrics.html"
    )
    create_trajectory_comparison(
        ball_data,
        output_dir / "skillmimic_trajectory_views.html"
    )

    print("\nVisualization complete!")
    print("Files generated:")
    print(f"  - {output_dir / 'skillmimic_3d_viewer.html'}: 3D animation")
    print(f"  - {output_dir / 'skillmimic_ball_metrics.html'}: Trajectory metrics")
    print(f"  - {output_dir / 'skillmimic_trajectory_views.html'}: 2D trajectory views")
    print("="*70)


if __name__ == "__main__":
    main()
