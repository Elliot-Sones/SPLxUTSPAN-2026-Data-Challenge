"""
Frame-by-frame visualization tool for SPLxUTSPAN Data Challenge 2026
Displays all keypoints with high precision and interactive navigation
"""

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Skeleton connections for drawing body structure
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


def create_interactive_visualization(shot, keypoint_names, output_path='frame_viewer.html'):
    """Create interactive frame-by-frame visualization"""

    # Get number of frames
    num_frames = len(shot[keypoint_names[0]])

    # Create figure with frames
    fig = go.Figure()

    # Create frames for animation/slider
    frames = []
    for frame_idx in range(num_frames):
        keypoints = get_keypoint_positions(shot, frame_idx)
        traces = create_skeleton_traces(keypoints, frame_idx)

        frames.append(go.Frame(
            data=traces,
            name=str(frame_idx),
            layout=go.Layout(
                title_text=f"Frame {frame_idx}/{num_frames-1} | Time: {frame_idx/60:.3f}s"
            )
        ))

    # Add initial frame
    initial_keypoints = get_keypoint_positions(shot, 0)
    initial_traces = create_skeleton_traces(initial_keypoints, 0)
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
            text=f"Shot Visualization<br>Shot ID: {shot['shot_id']} | Participant: {shot['participant_id']}<br>"
                 f"Angle: {shot['angle']:.2f}° | Depth: {shot['depth']:.2f}\" | Left/Right: {shot['left_right']:.2f}\"",
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
    print(f"Total frames: {num_frames}")
    print(f"Duration: {num_frames/60:.2f} seconds")
    print(f"Frame rate: 60 FPS")


def create_data_table(shot, keypoint_names, output_path='frame_data_table.html'):
    """Create detailed data table showing all coordinates for all frames"""

    num_frames = len(shot[keypoint_names[0]])

    # Create table data
    table_data = []
    for frame_idx in range(num_frames):
        keypoints = get_keypoint_positions(shot, frame_idx)
        for name, pos in keypoints.items():
            table_data.append({
                'Frame': frame_idx,
                'Time (s)': f"{frame_idx/60:.6f}",
                'Keypoint': name,
                'X': f"{pos[0]:.10f}",
                'Y': f"{pos[1]:.10f}",
                'Z': f"{pos[2]:.10f}"
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
        title=f"Complete Frame Data - Shot ID: {shot['shot_id']} | Participant: {shot['participant_id']}",
        height=800
    )

    fig.write_html(output_path)
    print(f"Data table saved to: {output_path}")


def main():
    # Load first shot from training data
    csv_path = '../data/train.csv'
    shot, keypoint_names = load_shot_data(csv_path, shot_index=0)

    print("="*60)
    print("SPLxUTSPAN Data Challenge 2026 - Frame-by-Frame Viewer")
    print("="*60)
    print(f"Shot ID: {shot['shot_id']}")
    print(f"Participant ID: {shot['participant_id']}")
    print(f"Target Angle: {shot['angle']:.2f}°")
    print(f"Target Depth: {shot['depth']:.2f}\"")
    print(f"Target Left/Right: {shot['left_right']:.2f}\"")
    print(f"Total Keypoints: {len(keypoint_names)//3}")
    print("="*60)

    # Create visualizations
    create_interactive_visualization(shot, keypoint_names, 'frame_viewer.html')
    create_data_table(shot, keypoint_names, 'frame_data_table.html')

    print("\nVisualization complete!")
    print("Open 'frame_viewer.html' in a browser to view the 3D animation")
    print("Open 'frame_data_table.html' to see all coordinate values")


if __name__ == '__main__':
    main()
