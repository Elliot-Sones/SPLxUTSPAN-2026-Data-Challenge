#!/usr/bin/env python3
"""
SkillMimic 3D Visualization - Fixed Skeleton

Shows complete humanoid skeleton (53 joints) with proper anatomical structure.
Based on mocap_humanoid.xml from SkillMimic repository.
"""

import argparse
import json
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Ball radius (normalized units)
BALL_RADIUS = 0.03


def load_joint_mapping(map_path: str = "visualisation/skillmimic_joint_map.json") -> dict:
    """Load joint mapping from JSON file"""
    with open(map_path, 'r') as f:
        return json.load(f)


def load_skillmimic_file(filepath: str) -> torch.Tensor:
    """Load SkillMimic .pt file"""
    data = torch.load(filepath, map_location='cpu')
    if not isinstance(data, torch.Tensor):
        raise RuntimeError(f"Expected torch.Tensor, got {type(data)}")
    return data


def extract_joint_positions(data: torch.Tensor, frame_idx: int, joint_map: dict) -> dict:
    """
    Extract joint positions for a specific frame using anatomical names

    Returns:
        Dictionary mapping joint names to [x, y, z] positions
    """
    joints = {}
    for name, indices in joint_map.items():
        x = data[frame_idx, indices[0]].item()
        y = data[frame_idx, indices[1]].item()
        z = data[frame_idx, indices[2]].item()
        joints[name] = np.array([x, y, z])

    return joints


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


def get_body_part_color(joint_name: str) -> tuple:
    """Assign color based on anatomical body part"""
    if any(part in joint_name for part in ["Pelvis", "Torso", "Spine", "Chest"]):
        return ('#1f77b4', 5)  # Blue, thick
    elif any(part in joint_name for part in ["Head", "Neck"]):
        return ('#FFD700', 5)  # Gold, thick
    elif any(part in joint_name for part in ["Hip", "Knee", "Ankle", "Toe"]):
        return ('#2ca02c', 5)  # Green, thick
    elif any(part in joint_name for part in ["Thorax", "Shoulder", "Elbow", "Wrist"]):
        return ('#d62728', 4)  # Red, medium
    else:  # Fingers
        return ('#ff7f0e', 2)  # Orange, thin


def create_skeleton_traces(joints: dict, skeleton_connections: list, frame_idx: int):
    """Create plotly traces for skeleton visualization with color-coded body parts"""
    traces = []

    # Group connections by body part for organized rendering
    main_body = []
    legs = []
    arms = []
    fingers = []

    for start_name, end_name in skeleton_connections:
        if start_name not in joints or end_name not in joints:
            continue

        # Categorize connection
        if any(part in start_name or part in end_name for part in ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"]):
            main_body.append((start_name, end_name))
        elif any(part in start_name or part in end_name for part in ["Hip", "Knee", "Ankle", "Toe"]):
            legs.append((start_name, end_name))
        elif any(part in start_name or part in end_name for part in ["Thorax", "Shoulder", "Elbow", "Wrist"]):
            arms.append((start_name, end_name))
        else:  # Fingers
            fingers.append((start_name, end_name))

    # Draw connections by category
    connection_groups = [
        (main_body, '#1f77b4', 6, 'Torso/Head'),
        (legs, '#2ca02c', 5, 'Legs'),
        (arms, '#d62728', 5, 'Arms'),
        (fingers, '#ff7f0e', 2, 'Hands'),
    ]

    for connections, color, width, label in connection_groups:
        if not connections:
            continue

        # Collect all line segments for this group
        x_coords = []
        y_coords = []
        z_coords = []

        for start_name, end_name in connections:
            start_pos = joints[start_name]
            end_pos = joints[end_name]

            x_coords.extend([start_pos[0], end_pos[0], None])
            y_coords.extend([start_pos[1], end_pos[1], None])
            z_coords.extend([start_pos[2], end_pos[2], None])

        traces.append(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(color=color, width=width),
            name=label,
            showlegend=(frame_idx == 0),
            hoverinfo='skip'
        ))

    # Add joint markers (major joints only - not all fingers)
    major_joints = [
        "Pelvis", "Torso", "Chest", "Head",
        "L_Hip", "L_Knee", "L_Ankle",
        "R_Hip", "R_Knee", "R_Ankle",
        "L_Shoulder", "L_Elbow", "L_Wrist",
        "R_Shoulder", "R_Elbow", "R_Wrist",
    ]

    marker_x = []
    marker_y = []
    marker_z = []
    marker_text = []
    marker_colors = []

    for name in major_joints:
        if name in joints:
            pos = joints[name]
            marker_x.append(pos[0])
            marker_y.append(pos[1])
            marker_z.append(pos[2])
            marker_text.append(name)
            color, _ = get_body_part_color(name)
            marker_colors.append(color)

    traces.append(go.Scatter3d(
        x=marker_x,
        y=marker_y,
        z=marker_z,
        mode='markers',
        marker=dict(size=4, color=marker_colors, line=dict(width=1, color='white')),
        name='Key Joints',
        text=marker_text,
        hoverinfo='text',
        showlegend=(frame_idx == 0)
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
        text=f"Ball<br>({ball_pos[0]:.4f}, {ball_pos[1]:.4f}, {ball_pos[2]:.4f})",
        showlegend=False
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
        text=[f"Frame {i}" for i in range(len(traj))],
        showlegend=False
    )


def create_ground_plane(joints_all_frames):
    """Create ground plane reference"""
    traces = []

    # Find ground level (min Y across all frames)
    all_y = []
    for frame_joints in joints_all_frames:
        for pos in frame_joints.values():
            all_y.append(pos[1])
    min_y = min(all_y)

    # Find x and z ranges
    all_x = []
    all_z = []
    for frame_joints in joints_all_frames:
        for pos in frame_joints.values():
            all_x.append(pos[0])
            all_z.append(pos[2])

    x_range = [min(all_x) - 0.5, max(all_x) + 0.5]
    z_range = [min(all_z) - 0.5, max(all_z) + 0.5]

    # Create grid
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


def create_coordinate_axes(origin, scale=0.2):
    """Create XYZ coordinate axes for reference"""
    traces = []

    axes = [
        ('X', [scale, 0, 0], 'red'),
        ('Y', [0, scale, 0], 'green'),
        ('Z', [0, 0, scale], 'blue'),
    ]

    for name, direction, color in axes:
        traces.append(go.Scatter3d(
            x=[origin[0], origin[0] + direction[0]],
            y=[origin[1], origin[1] + direction[1]],
            z=[origin[2], origin[2] + direction[2]],
            mode='lines+text',
            line=dict(color=color, width=4),
            text=['', name],
            textposition='top center',
            name=f'{name}-axis',
            showlegend=False,
            hoverinfo='skip'
        ))

    return traces


def create_interactive_visualization(data, ball_data, joint_mapping, filepath,
                                     output_path='skillmimic_3d_skeleton_fixed.html'):
    """Create interactive 3D visualization with proper skeleton"""

    joint_map = joint_mapping['joint_map']
    skeleton_connections = joint_mapping['skeleton_connections']
    num_frames = ball_data['num_frames']
    release_frame = ball_data['release_frame']
    trajectory = ball_data['position']
    contact_flags = ball_data['contact_flag']

    # Extract all joint positions for ground plane calculation
    joints_all_frames = [extract_joint_positions(data, i, joint_map) for i in range(num_frames)]

    # Create figure
    fig = go.Figure()

    # Create frames for animation
    frames = []
    for frame_idx in range(num_frames):
        traces = []

        # Add ground plane
        traces.extend(create_ground_plane(joints_all_frames))

        # Add coordinate axes at pelvis
        joints = joints_all_frames[frame_idx]
        if "Pelvis" in joints:
            traces.extend(create_coordinate_axes(joints["Pelvis"]))

        # Add skeleton
        traces.extend(create_skeleton_traces(joints, skeleton_connections, frame_idx))

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
    initial_joints = joints_all_frames[0]
    if "Pelvis" in initial_joints:
        initial_traces.extend(create_coordinate_axes(initial_joints["Pelvis"]))
    initial_traces.extend(create_skeleton_traces(initial_joints, skeleton_connections, 0))
    initial_ball = create_ball_sphere_trace(trajectory[0], contact_flags[0] == 1, False)
    initial_traces.append(initial_ball)
    initial_traj = create_ball_trajectory_trace(trajectory, 0, release_frame)
    initial_traces.append(initial_traj)

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
            text=f"SkillMimic: Complete Skeleton (53 Joints)<br>"
                 f"File: {filepath.split('/')[-1]}<br>"
                 f"Release Frame: {release_frame}",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y (up)',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=2.5, y=1.2, z=2.0),
                center=dict(x=0, y=0, z=0)
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
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D visualization of SkillMimic with complete skeleton (53 joints)"
    )
    parser.add_argument("filepath", help="Path to .pt file")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--joint-map", default="visualisation/skillmimic_joint_map.json",
                        help="Path to joint mapping JSON")

    args = parser.parse_args()

    print("="*70)
    print("SkillMimic 3D Visualization - Complete Skeleton")
    print("="*70)
    print(f"Loading: {args.filepath}")

    # Load joint mapping
    joint_mapping = load_joint_mapping(args.joint_map)
    print(f"Joint mapping: {joint_mapping['num_bodies']} bodies, "
          f"{len(joint_mapping['skeleton_connections'])} connections")

    # Load data
    data = load_skillmimic_file(args.filepath)
    print(f"Data shape: {data.shape}")

    # Extract ball data
    ball_data = extract_ball_data(data)
    print(f"Release frame: {ball_data['release_frame']}")

    # Create visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_interactive_visualization(
        data, ball_data, joint_mapping, args.filepath,
        output_dir / "skillmimic_3d_skeleton_fixed.html"
    )

    print("\nVisualization complete!")
    print(f"Open: {output_dir / 'skillmimic_3d_skeleton_fixed.html'}")
    print("="*70)


if __name__ == "__main__":
    main()
