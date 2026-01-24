"""
Enhanced frame-by-frame visualization with ball tracking
Displays skeleton + basketball position + release detection
"""

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ball_tracking as bt


# Skeleton connections (same as frame_by_frame_viewer.py)
SKELETON_CONNECTIONS = [
    # Head
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),

    # Torso
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'neck'),
    ('right_shoulder', 'neck'),
    ('neck', 'mid_hip'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'mid_hip'),
    ('right_hip', 'mid_hip'),

    # Left arm
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('left_wrist', 'left_wrist_2'),

    # Right arm
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('right_wrist', 'right_wrist_2'),

    # Left leg
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('left_ankle', 'left_heel'),
    ('left_heel', 'left_big_toe'),
    ('left_big_toe', 'left_small_toe'),

    # Right leg
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('right_ankle', 'right_heel'),
    ('right_heel', 'right_big_toe'),
    ('right_big_toe', 'right_small_toe'),

    # Left hand - thumb
    ('left_wrist_2', 'left_first_finger_cmc'),
    ('left_first_finger_cmc', 'left_first_finger_mcp'),
    ('left_first_finger_mcp', 'left_first_finger_ip'),
    ('left_first_finger_ip', 'left_first_finger_distal'),
    ('left_wrist_2', 'left_thumb'),

    # Left hand - index
    ('left_wrist_2', 'left_second_finger_mcp'),
    ('left_second_finger_mcp', 'left_second_finger_pip'),
    ('left_second_finger_pip', 'left_second_finger_dip'),
    ('left_second_finger_dip', 'left_second_finger_distal'),

    # Left hand - middle
    ('left_wrist_2', 'left_third_finger_mcp'),
    ('left_third_finger_mcp', 'left_third_finger_pip'),
    ('left_third_finger_pip', 'left_third_finger_dip'),
    ('left_third_finger_dip', 'left_third_finger_distal'),

    # Left hand - ring
    ('left_wrist_2', 'left_fourth_finger_mcp'),
    ('left_fourth_finger_mcp', 'left_fourth_finger_pip'),
    ('left_fourth_finger_pip', 'left_fourth_finger_dip'),
    ('left_fourth_finger_dip', 'left_fourth_finger_distal'),

    # Left hand - pinky
    ('left_wrist_2', 'left_fifth_finger_mcp'),
    ('left_fifth_finger_mcp', 'left_fifth_finger_pip'),
    ('left_fifth_finger_pip', 'left_fifth_finger_dip'),
    ('left_fifth_finger_dip', 'left_fifth_finger_distal'),
    ('left_wrist_2', 'left_pinky'),

    # Right hand - thumb
    ('right_wrist_2', 'right_first_finger_cmc'),
    ('right_first_finger_cmc', 'right_first_finger_mcp'),
    ('right_first_finger_mcp', 'right_first_finger_ip'),
    ('right_first_finger_ip', 'right_first_finger_distal'),
    ('right_wrist_2', 'right_thumb'),

    # Right hand - index
    ('right_wrist_2', 'right_second_finger_mcp'),
    ('right_second_finger_mcp', 'right_second_finger_pip'),
    ('right_second_finger_pip', 'right_second_finger_dip'),
    ('right_second_finger_dip', 'right_second_finger_distal'),

    # Right hand - middle
    ('right_wrist_2', 'right_third_finger_mcp'),
    ('right_third_finger_mcp', 'right_third_finger_pip'),
    ('right_third_finger_pip', 'right_third_finger_dip'),
    ('right_third_finger_dip', 'right_third_finger_distal'),

    # Right hand - ring
    ('right_wrist_2', 'right_fourth_finger_mcp'),
    ('right_fourth_finger_mcp', 'right_fourth_finger_pip'),
    ('right_fourth_finger_pip', 'right_fourth_finger_dip'),
    ('right_fourth_finger_dip', 'right_fourth_finger_distal'),

    # Right hand - pinky
    ('right_wrist_2', 'right_fifth_finger_mcp'),
    ('right_fifth_finger_mcp', 'right_fifth_finger_pip'),
    ('right_fifth_finger_pip', 'right_fifth_finger_dip'),
    ('right_fifth_finger_dip', 'right_fifth_finger_distal'),
    ('right_wrist_2', 'right_pinky'),
]


def parse_array_json(s):
    """Convert string representation of array to numpy array"""
    s = s.replace('nan', 'null')
    return np.array(json.loads(s), dtype=np.float32)


def load_shot_data(csv_path, shot_index=0):
    """Load and parse a single shot from the dataset"""
    df = pd.read_csv(csv_path)

    # Get keypoint column names
    keypoint_names = df.columns[3:-3]

    # Parse arrays for the selected shot
    shot = df.iloc[shot_index].copy()
    for col in keypoint_names:
        shot[col] = parse_array_json(shot[col])

    return shot, keypoint_names


def get_keypoint_positions(shot, frame_idx):
    """Extract all keypoint positions for a specific frame"""
    keypoints = {}

    # Extract unique keypoint names (remove _x, _y, _z suffix)
    keypoint_base_names = set()
    for col in shot.index:
        if col.endswith('_x'):
            keypoint_base_names.add(col[:-2])

    # Get positions for each keypoint
    for name in keypoint_base_names:
        try:
            x = shot[f'{name}_x'][frame_idx]
            y = shot[f'{name}_y'][frame_idx]
            z = shot[f'{name}_z'][frame_idx]

            # Only add if not NaN
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                keypoints[name] = np.array([x, y, z])
        except (KeyError, IndexError):
            continue

    return keypoints


def create_skeleton_traces(keypoints, frame_idx):
    """Create plotly traces for skeleton visualization"""
    traces = []

    # Add all keypoints as scatter points
    xs, ys, zs = [], [], []
    labels = []
    for name, pos in keypoints.items():
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])
        labels.append(f"{name}<br>({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")

    # Keypoint scatter
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=5, color='red'),
        text=labels,
        hoverinfo='text',
        name='Keypoints'
    ))

    # Add skeleton connections
    for conn in SKELETON_CONNECTIONS:
        start, end = conn
        if start in keypoints and end in keypoints:
            start_pos = keypoints[start]
            end_pos = keypoints[end]

            traces.append(go.Scatter3d(
                x=[start_pos[0], end_pos[0]],
                y=[start_pos[1], end_pos[1]],
                z=[start_pos[2], end_pos[2]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    return traces


def create_ball_sphere_trace(ball_pos, frame_idx, release_frame, num_points=20):
    """
    Create a sphere mesh to represent the basketball

    Args:
        ball_pos: [x, y, z] position of ball center
        frame_idx: Current frame index
        release_frame: Release frame index
        num_points: Resolution of sphere mesh

    Returns:
        go.Scatter3d trace representing the ball
    """
    if ball_pos is None or np.any(np.isnan(ball_pos)):
        # Return empty trace if no ball data
        return go.Scatter3d(x=[], y=[], z=[], mode='markers', showlegend=False)

    # Generate sphere surface
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = bt.NBA_BALL_RADIUS_FEET * np.outer(np.cos(u), np.sin(v)) + ball_pos[0]
    y = bt.NBA_BALL_RADIUS_FEET * np.outer(np.sin(u), np.sin(v)) + ball_pos[1]
    z = bt.NBA_BALL_RADIUS_FEET * np.outer(np.ones(np.size(u)), np.cos(v)) + ball_pos[2]

    # Flatten for scatter3d
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Color based on phase
    if frame_idx < release_frame:
        color = '#FF8C00'  # Orange (pre-release)
        size = 3
        opacity = 0.6
    elif frame_idx == release_frame:
        color = '#FFD700'  # Gold (release frame)
        size = 5
        opacity = 0.9
    else:
        color = '#808080'  # Gray (post-release)
        size = 2
        opacity = 0.3

    return go.Scatter3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        mode='markers',
        marker=dict(size=size, color=color, opacity=opacity),
        name='Basketball',
        hoverinfo='text',
        text=f"Ball<br>Frame {frame_idx}<br>({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f})"
    )


def create_ball_trajectory_trace(trajectory, release_frame, start_frame=0):
    """
    Create trajectory trail showing ball path up to release

    Args:
        trajectory: Array of ball positions (num_frames, 3)
        release_frame: Frame index of release
        start_frame: Starting frame for trajectory

    Returns:
        go.Scatter3d trace for trajectory
    """
    # Filter out NaN values and only show up to release
    valid_indices = []
    xs, ys, zs = [], [], []

    for i, pos in enumerate(trajectory):
        frame_idx = start_frame + i
        if frame_idx > release_frame:
            break
        if not np.any(np.isnan(pos)):
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            valid_indices.append(i)

    if len(xs) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode='lines', showlegend=False)

    # Color gradient from blue (start) to red (release)
    colors = np.linspace(0, 1, len(xs))

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines+markers',
        line=dict(
            width=4,
            color=colors,
            colorscale='Bluered',
            showscale=False
        ),
        marker=dict(size=2),
        name='Ball Trajectory',
        hoverinfo='text',
        text=[f"Frame {start_frame + i}" for i in valid_indices]
    )


def create_interactive_visualization(shot, keypoint_names, release_frame, release_scores,
                                     output_path='ball_frame_viewer.html'):
    """Create interactive frame-by-frame visualization with ball tracking"""

    # Get number of frames
    num_frames = len(shot[keypoint_names[0]])

    # Calculate ball trajectory for all frames
    print("Calculating ball trajectory...")
    trajectory = bt.get_ball_trajectory(shot, 0, num_frames - 1)

    # Create figure with frames
    fig = go.Figure()

    # Create frames for animation/slider
    frames = []
    for frame_idx in range(num_frames):
        # Get skeleton traces
        keypoints = get_keypoint_positions(shot, frame_idx)
        traces = create_skeleton_traces(keypoints, frame_idx)

        # Add ball sphere
        ball_pos = bt.calculate_ball_position(shot, frame_idx)
        ball_trace = create_ball_sphere_trace(ball_pos, frame_idx, release_frame)
        traces.append(ball_trace)

        # Add trajectory trail (up to current frame)
        traj_trace = create_ball_trajectory_trace(trajectory[:frame_idx+1], release_frame, 0)
        traces.append(traj_trace)

        # Title with release indication
        release_indicator = " [RELEASE]" if frame_idx == release_frame else ""
        title = (f"Frame {frame_idx}/{num_frames-1} | Time: {frame_idx/60:.3f}s{release_indicator}<br>"
                f"Release Score: {release_scores[frame_idx]:.3f}")

        frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            layout=go.Layout(title_text=title)
        ))

    # Add initial frame
    initial_keypoints = get_keypoint_positions(shot, 0)
    initial_traces = create_skeleton_traces(initial_keypoints, 0)
    initial_ball = create_ball_sphere_trace(bt.calculate_ball_position(shot, 0), 0, release_frame)
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
        x=0,
        currentvalue=dict(
            prefix='Frame: ',
            visible=True,
            xanchor='center'
        ),
        len=1.0,
        xanchor='left',
        yanchor='top',
        transition=dict(duration=0)
    )]

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Shot Visualization with Ball Tracking<br>Shot ID: {shot['shot_id']} | Participant: {shot['participant_id']}<br>"
                 f"Angle: {shot['angle']:.2f}° | Depth: {shot['depth']:.2f}\" | Left/Right: {shot['left_right']:.2f}\"<br>"
                 f"<b>Release Frame: {release_frame} ({release_frame/60:.3f}s)</b>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (feet)',
            yaxis_title='Y (feet)',
            zaxis_title='Z (feet)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        sliders=sliders,
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0.05,
                x=0,
                xanchor='left',
                yanchor='bottom',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=16.67, redraw=True),  # ~60 FPS
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

    # Save to HTML
    fig.write_html(output_path)
    print(f"Visualization saved to: {output_path}")


def create_metrics_plot(shot, release_frame, release_scores, output_path='ball_metrics.html'):
    """Create plots showing release detection metrics over time"""

    num_frames = len(shot['nose_x'])

    # Calculate metrics for each frame
    frames = np.arange(num_frames)
    times = frames / 60.0

    hand_velocities_left = []
    hand_velocities_right = []
    hand_separations = []
    apertures_left = []
    apertures_right = []

    for frame_idx in range(num_frames):
        # Hand velocities (magnitude)
        vel_left = bt.calculate_hand_velocity(shot, frame_idx, 'left', window=1)
        vel_right = bt.calculate_hand_velocity(shot, frame_idx, 'right', window=1)

        hand_velocities_left.append(np.linalg.norm(vel_left) if vel_left is not None else np.nan)
        hand_velocities_right.append(np.linalg.norm(vel_right) if vel_right is not None else np.nan)

        # Hand separation
        sep = bt.calculate_hand_separation(shot, frame_idx)
        hand_separations.append(sep if sep is not None else np.nan)

        # Hand apertures
        ap_left = bt.calculate_hand_aperture(shot, frame_idx, 'left')
        ap_right = bt.calculate_hand_aperture(shot, frame_idx, 'right')

        apertures_left.append(ap_left if ap_left is not None else np.nan)
        apertures_right.append(ap_right if ap_right is not None else np.nan)

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Release Score Over Time',
            'Hand Velocity (Magnitude)',
            'Hand Separation Distance',
            'Hand Aperture (Thumb-Index Distance)'
        ),
        vertical_spacing=0.08
    )

    # Plot 1: Release score
    fig.add_trace(
        go.Scatter(x=times, y=release_scores, mode='lines', name='Release Score',
                  line=dict(color='purple', width=2)),
        row=1, col=1
    )
    fig.add_vline(x=release_frame/60.0, line_dash="dash", line_color="red",
                 annotation_text=f"Release: {release_frame/60.0:.3f}s", row=1, col=1)

    # Plot 2: Hand velocities
    fig.add_trace(
        go.Scatter(x=times, y=hand_velocities_left, mode='lines', name='Left Hand Vel',
                  line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=times, y=hand_velocities_right, mode='lines', name='Right Hand Vel',
                  line=dict(color='green')),
        row=2, col=1
    )
    fig.add_vline(x=release_frame/60.0, line_dash="dash", line_color="red", row=2, col=1)

    # Plot 3: Hand separation
    fig.add_trace(
        go.Scatter(x=times, y=hand_separations, mode='lines', name='Hand Separation',
                  line=dict(color='orange', width=2)),
        row=3, col=1
    )
    fig.add_vline(x=release_frame/60.0, line_dash="dash", line_color="red", row=3, col=1)

    # Plot 4: Hand apertures
    fig.add_trace(
        go.Scatter(x=times, y=apertures_left, mode='lines', name='Left Aperture',
                  line=dict(color='cyan')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=times, y=apertures_right, mode='lines', name='Right Aperture',
                  line=dict(color='magenta')),
        row=4, col=1
    )
    fig.add_vline(x=release_frame/60.0, line_dash="dash", line_color="red", row=4, col=1)

    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (ft/s)", row=2, col=1)
    fig.update_yaxes(title_text="Distance (ft)", row=3, col=1)
    fig.update_yaxes(title_text="Distance (ft)", row=4, col=1)

    fig.update_layout(
        title_text=f"Ball Release Detection Metrics - Shot ID: {shot['shot_id']}",
        height=1200,
        showlegend=True
    )

    fig.write_html(output_path)
    print(f"Metrics plot saved to: {output_path}")


def create_data_table(shot, keypoint_names, release_frame, release_scores,
                     output_path='ball_data_table.html'):
    """Create detailed data table with ball coordinates and metrics"""

    num_frames = len(shot[keypoint_names[0]])

    # Create table data
    table_data = []
    for frame_idx in range(num_frames):
        ball_pos = bt.calculate_ball_position(shot, frame_idx)

        table_data.append({
            'Frame': frame_idx,
            'Time (s)': f"{frame_idx/60:.6f}",
            'Ball_X': f"{ball_pos[0]:.10f}" if ball_pos is not None else "N/A",
            'Ball_Y': f"{ball_pos[1]:.10f}" if ball_pos is not None else "N/A",
            'Ball_Z': f"{ball_pos[2]:.10f}" if ball_pos is not None else "N/A",
            'Release_Score': f"{release_scores[frame_idx]:.6f}",
            'Is_Release': "YES" if frame_idx == release_frame else ""
        })

    df_table = pd.DataFrame(table_data)

    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_table.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df_table[col] for col in df_table.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=10, color='black'),
            height=25
        )
    )])

    fig.update_layout(
        title=f"Ball Position Data - Shot ID: {shot['shot_id']} | Release Frame: {release_frame}",
        height=800
    )

    fig.write_html(output_path)
    print(f"Data table saved to: {output_path}")


def main():
    # Load first shot from training data
    csv_path = '../data/train.csv'
    shot, keypoint_names = load_shot_data(csv_path, shot_index=0)

    print("="*60)
    print("Ball Tracking Visualization - SPLxUTSPAN 2026")
    print("="*60)
    print(f"Shot ID: {shot['shot_id']}")
    print(f"Participant ID: {shot['participant_id']}")
    print(f"Target Angle: {shot['angle']:.2f}°")
    print(f"Target Depth: {shot['depth']:.2f}\"")
    print(f"Target Left/Right: {shot['left_right']:.2f}\"")
    print("="*60)

    # Detect release frame
    print("\nDetecting ball release...")
    release_frame, release_scores = bt.detect_release_frame(shot)

    print(f"Release Frame: {release_frame}")
    print(f"Release Time: {release_frame/60:.3f} seconds")
    print(f"Release Score: {release_scores[release_frame]:.3f}")

    # Calculate trajectory statistics
    trajectory = bt.get_ball_trajectory(shot, 0, release_frame)
    valid_trajectory = trajectory[~np.isnan(trajectory).any(axis=1)]

    if len(valid_trajectory) > 0:
        print(f"\nBall Trajectory Statistics (frames 0-{release_frame}):")
        print(f"  Distance traveled: {np.linalg.norm(valid_trajectory[-1] - valid_trajectory[0]):.3f} feet")
        print(f"  Max height (Z): {np.max(valid_trajectory[:, 2]):.3f} feet")
        print(f"  Min height (Z): {np.min(valid_trajectory[:, 2]):.3f} feet")
        print(f"  Height gain: {valid_trajectory[-1, 2] - valid_trajectory[0, 2]:.3f} feet")

    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)

    # Create visualizations
    create_interactive_visualization(shot, keypoint_names, release_frame, release_scores,
                                    'ball_frame_viewer.html')
    create_metrics_plot(shot, release_frame, release_scores, 'ball_metrics.html')
    create_data_table(shot, keypoint_names, release_frame, release_scores,
                     'ball_data_table.html')

    print("\nVisualization complete!")
    print("Files generated:")
    print("  - ball_frame_viewer.html: 3D animation with basketball")
    print("  - ball_metrics.html: Release detection metrics")
    print("  - ball_data_table.html: Ball coordinate data")


if __name__ == '__main__':
    main()
