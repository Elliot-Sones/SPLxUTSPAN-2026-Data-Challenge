#!/usr/bin/env python3
"""
SkillMimic 3D Visualization with Skeleton
Interactive 3D animation showing humanoid skeleton + basketball
"""

import argparse
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Ball radius (normalized units)
BALL_RADIUS = 0.03

# Joint indices (triplets of x,y,z features)
JOINT_TRIPLETS = [
    [51, 52, 53],    # 0
    [54, 55, 56],    # 1
    [57, 58, 59],    # 2
    [108, 109, 110], # 3
    [111, 112, 113], # 4
    [114, 115, 116], # 5
    [220, 221, 222], # 6
    [223, 224, 225], # 7
    [226, 227, 228], # 8
    [229, 230, 231], # 9
    [232, 233, 234], # 10
    [235, 236, 237], # 11
    [238, 239, 240], # 12
    [241, 242, 243], # 13
    [244, 245, 246], # 14
    [247, 248, 249], # 15
    [250, 251, 252], # 16
    [253, 254, 255], # 17
    [259, 260, 261], # 18
    [262, 263, 264], # 19
]

# Skeleton connections (based on analysis)
# Core body structure + hand connections
SKELETON_CONNECTIONS = [
    # Lower body
    (1, 3),  # Left foot to pelvis
    (2, 3),  # Right foot to pelvis
    (3, 0),  # Pelvis to torso

    # Upper body
    (0, 4),  # Torso to right shoulder/arm
    (0, 5),  # Torso to left shoulder/arm

    # Right hand finger chain (joints 6-19)
    # Connect all hand joints to form palm/fingers
    (4, 6),   # Arm to hand base
    (6, 7), (7, 8), (8, 9),    # Finger chain 1
    (9, 10), (10, 11), (11, 12), # Finger chain 2
    (12, 13), (13, 14), (14, 15), # Finger chain 3
    (15, 16), (16, 17),  # Finger chain 4
    (6, 18), (18, 19),   # Thumb
]


def load_skillmimic_file(filepath: str) -> torch.Tensor:
    """Load SkillMimic .pt file"""
    data = torch.load(filepath, map_location='cpu')
    if not isinstance(data, torch.Tensor):
        raise RuntimeError(f"Expected torch.Tensor, got {type(data)}")
    return data


def extract_joint_positions(data: torch.Tensor, frame_idx: int) -> np.ndarray:
    """
    Extract joint positions for a specific frame

    Returns:
        Array of shape (num_joints, 3) with [x, y, z] positions
    """
    joints = []
    for triplet in JOINT_TRIPLETS:
        x = data[frame_idx, triplet[0]].item()
        y = data[frame_idx, triplet[1]].item()
        z = data[frame_idx, triplet[2]].item()
        joints.append([x, y, z])

    return np.array(joints)


def extract_ball_data(data: torch.Tensor) -> dict:
    """Extract ball-related features from SkillMimic tensor"""
    # Ball position: features 119 (Y), 325 (X), 326 (Z)
    ball_x = data[:, 325].numpy()
    ball_y = data[:, 119].numpy()
    ball_z = data[:, 326].numpy()
    ball_position = np.stack([ball_x, ball_y, ball_z], axis=1)

    # Contact flag: feature 336 (1=in hand, 0=released)
    contact_flag = data[:, 336].numpy()

    # Find release frame
    release_frame = None
    for i in range(len(contact_flag) - 1):
        if contact_flag[i] == 1 and contact_flag[i+1] == 0:
            release_frame = i + 1
            break
    if release_frame is None:
        release_frame = 0

    return {
        'position': ball_position,
        'contact_flag': contact_flag,
        'release_frame': release_frame,
        'num_frames': len(data)
    }


def create_skeleton_traces(joints, frame_idx):
    """Create plotly traces for skeleton visualization"""
    traces = []

    # Add all keypoints as scatter points
    traces.append(go.Scatter3d(
        x=joints[:, 0],
        y=joints[:, 1],
        z=joints[:, 2],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Joints',
        text=[f"Joint {i}" for i in range(len(joints))],
        hoverinfo='text'
    ))

    # Add skeleton connections
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start_pos = joints[start_idx]
        end_pos = joints[end_idx]

        traces.append(go.Scatter3d(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]],
            z=[start_pos[2], end_pos[2]],
            mode='lines',
            line=dict(color='blue', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

    return traces


def create_ball_sphere_trace(ball_pos, in_hand=True, is_release=False, num_points=15):
    """Create sphere mesh for basketball"""
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = BALL_RADIUS * np.outer(np.cos(u), np.sin(v)) + ball_pos[0]
    y = BALL_RADIUS * np.outer(np.sin(u), np.sin(v)) + ball_pos[1]
    z = BALL_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v)) + ball_pos[2]

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    if is_release:
        color = '#FFD700'  # Gold
        size = 5
        opacity = 1.0
    elif in_hand:
        color = '#FF8C00'  # Orange
        size = 3
        opacity = 0.7
    else:
        color = '#4169E1'  # Royal blue
        size = 2
        opacity = 0.5

    return go.Scatter3d(
        x=x_flat, y=y_flat, z=z_flat,
        mode='markers',
        marker=dict(size=size, color=color, opacity=opacity),
        name='Basketball',
        hoverinfo='text',
        text=f"Ball<br>({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f})"
    )


def create_ball_trajectory_trace(trajectory, current_frame, release_frame):
    """Create trajectory trail showing ball path"""
    traj = trajectory[:current_frame+1]

    if len(traj) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode='lines', showlegend=False)

    colors = []
    for i in range(len(traj)):
        if i < release_frame:
            colors.append(0)
        else:
            colors.append((i - release_frame) / max(1, current_frame - release_frame))

    return go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
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


def create_ground_plane(joints_all_frames):
    """Create ground plane reference"""
    traces = []

    # Find ground level (min Y across all frames)
    min_y = joints_all_frames[:, :, 1].min()

    # Create grid
    x_range = [joints_all_frames[:, :, 0].min() - 0.5, joints_all_frames[:, :, 0].max() + 0.5]
    z_range = [joints_all_frames[:, :, 2].min() - 0.5, joints_all_frames[:, :, 2].max() + 0.5]

    grid_points = np.linspace(-2, 2, 9)
    for x in grid_points:
        if x >= x_range[0] and x <= x_range[1]:
            traces.append(go.Scatter3d(
                x=[x, x],
                y=[min_y, min_y],
                z=[z_range[0], z_range[1]],
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
    for z in grid_points:
        if z >= z_range[0] and z <= z_range[1]:
            traces.append(go.Scatter3d(
                x=[x_range[0], x_range[1]],
                y=[min_y, min_y],
                z=[z, z],
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

    return traces


def create_interactive_visualization(data, ball_data, filepath, output_path='skillmimic_3d_skeleton.html'):
    """Create interactive 3D visualization with skeleton and ball"""

    num_frames = ball_data['num_frames']
    release_frame = ball_data['release_frame']
    trajectory = ball_data['position']
    contact_flags = ball_data['contact_flag']

    # Extract all joint positions for ground plane calculation
    joints_all_frames = np.array([extract_joint_positions(data, i) for i in range(num_frames)])

    # Create figure
    fig = go.Figure()

    # Create frames for animation
    frames = []
    for frame_idx in range(num_frames):
        traces = []

        # Add ground plane
        traces.extend(create_ground_plane(joints_all_frames))

        # Add skeleton
        joints = extract_joint_positions(data, frame_idx)
        traces.extend(create_skeleton_traces(joints, frame_idx))

        # Add ball
        ball_pos = trajectory[frame_idx]
        in_hand = contact_flags[frame_idx] == 1
        is_release = (frame_idx == release_frame)
        ball_trace = create_ball_sphere_trace(ball_pos, in_hand, is_release)
        traces.append(ball_trace)

        # Add trajectory trail
        traj_trace = create_ball_trajectory_trace(trajectory, frame_idx, release_frame)
        traces.append(traj_trace)

        # Title
        release_indicator = " [RELEASE]" if is_release else ""
        phase = "In Hand" if in_hand else "Released"
        title = (f"Frame {frame_idx}/{num_frames-1}{release_indicator}<br>"
                f"Phase: {phase}")

        frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            layout=go.Layout(title_text=title)
        ))

    # Add initial frame traces
    initial_traces = create_ground_plane(joints_all_frames)
    initial_joints = extract_joint_positions(data, 0)
    initial_traces.extend(create_skeleton_traces(initial_joints, 0))
    initial_ball = create_ball_sphere_trace(trajectory[0], contact_flags[0] == 1, False)
    initial_traces.append(initial_ball)

    for trace in initial_traces:
        fig.add_trace(trace)

    # Add frames
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
        currentvalue=dict(prefix='Frame: ', visible=True, xanchor='center'),
        len=0.9,
        xanchor='left',
        yanchor='top',
        transition=dict(duration=0)
    )]

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"SkillMimic: Skeleton + Ball Visualization<br>"
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
            camera=dict(eye=dict(x=2.0, y=1.5, z=1.5))
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
                            frame=dict(duration=33, redraw=True),
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

    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D visualization of SkillMimic skeleton + basketball"
    )
    parser.add_argument("filepath", help="Path to .pt file")
    parser.add_argument("--output-dir", default=".", help="Output directory")

    args = parser.parse_args()

    print("="*70)
    print("SkillMimic 3D Skeleton + Ball Visualization")
    print("="*70)
    print(f"Loading: {args.filepath}")

    # Load data
    data = load_skillmimic_file(args.filepath)
    print(f"Data shape: {data.shape}")
    print(f"Extracted {len(JOINT_TRIPLETS)} joints")

    # Extract ball data
    ball_data = extract_ball_data(data)
    print(f"\nRelease frame: {ball_data['release_frame']}")

    # Create visualization
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_interactive_visualization(
        data, ball_data, args.filepath,
        output_dir / "skillmimic_3d_skeleton.html"
    )

    print("\nVisualization complete!")
    print(f"Open: {output_dir / 'skillmimic_3d_skeleton.html'}")
    print("="*70)


if __name__ == "__main__":
    main()
