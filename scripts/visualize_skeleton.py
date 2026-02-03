"""
Visualize skeleton data to understand coordinate system.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent


def get_keypoint_info():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return keypoints


# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Torso
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('neck', 'mid_hip'),
    ('left_shoulder', 'neck'),
    ('right_shoulder', 'neck'),

    # Left arm
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),

    # Right arm
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),

    # Left leg
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),

    # Right leg
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),

    # Head
    ('nose', 'neck'),
]


def visualize_shot(timeseries, keypoints, shot_idx=0, frames=[145, 150, 153, 156, 160]):
    """Visualize skeleton at multiple frames."""
    keypoint_idx = {name: i for i, name in enumerate(keypoints)}

    fig = plt.figure(figsize=(16, 4))

    for i, frame in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), i+1, projection='3d')

        # Extract positions for this frame
        positions = {}
        for name, idx in keypoint_idx.items():
            pos = timeseries[frame, idx*3:(idx+1)*3]
            if np.any(pos):
                positions[name] = pos

        # Plot joints
        xs, ys, zs = [], [], []
        for name, pos in positions.items():
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

        ax.scatter(xs, ys, zs, c='blue', s=20)

        # Plot skeleton connections
        for j1, j2 in SKELETON_CONNECTIONS:
            if j1 in positions and j2 in positions:
                p1, p2 = positions[j1], positions[j2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=1)

        # Highlight wrist
        if 'right_wrist' in positions:
            w = positions['right_wrist']
            ax.scatter([w[0]], [w[1]], [w[2]], c='red', s=100, marker='*')

        ax.set_title(f'Frame {frame}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set consistent limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)

    plt.suptitle(f'Shot {shot_idx} - Skeleton at Release')
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / 'output' / 'skeleton_visualization.png', dpi=150)
    print(f"Saved: output/skeleton_visualization.png")


def analyze_coordinate_system(timeseries, keypoints):
    """Analyze the coordinate system of our data."""
    keypoint_idx = {name: i for i, name in enumerate(keypoints)}

    print("\n=== COORDINATE SYSTEM ANALYSIS ===")

    # Analyze a few key joints at release frame
    frame = 153

    joints_to_check = ['nose', 'left_shoulder', 'right_shoulder',
                       'left_hip', 'right_hip', 'right_wrist', 'right_ankle']

    print(f"\nJoint positions at frame {frame}:")
    for name in joints_to_check:
        if name in keypoint_idx:
            idx = keypoint_idx[name]
            pos = timeseries[frame, idx*3:(idx+1)*3]
            print(f"  {name:20s}: x={pos[0]:7.3f}, y={pos[1]:7.3f}, z={pos[2]:7.3f}")

    # Check axis orientations
    print("\n=== AXIS INTERPRETATION ===")

    # Get shoulder and hip positions
    ls = timeseries[frame, keypoint_idx['left_shoulder']*3:(keypoint_idx['left_shoulder']+1)*3]
    rs = timeseries[frame, keypoint_idx['right_shoulder']*3:(keypoint_idx['right_shoulder']+1)*3]
    lh = timeseries[frame, keypoint_idx['left_hip']*3:(keypoint_idx['left_hip']+1)*3]
    rh = timeseries[frame, keypoint_idx['right_hip']*3:(keypoint_idx['right_hip']+1)*3]
    nose = timeseries[frame, keypoint_idx['nose']*3:(keypoint_idx['nose']+1)*3]
    ankle = timeseries[frame, keypoint_idx['right_ankle']*3:(keypoint_idx['right_ankle']+1)*3]

    # Left-right axis
    shoulder_diff = rs - ls
    print(f"Right - Left shoulder: x={shoulder_diff[0]:.3f}, y={shoulder_diff[1]:.3f}, z={shoulder_diff[2]:.3f}")

    # Up-down axis
    nose_ankle_diff = nose - ankle
    print(f"Nose - Ankle: x={nose_ankle_diff[0]:.3f}, y={nose_ankle_diff[1]:.3f}, z={nose_ankle_diff[2]:.3f}")

    # Determine axes
    print("\n=== AXIS MAPPING ===")
    if abs(shoulder_diff[0]) > abs(shoulder_diff[1]) and abs(shoulder_diff[0]) > abs(shoulder_diff[2]):
        print("X axis: Left-Right")
    elif abs(shoulder_diff[1]) > abs(shoulder_diff[0]) and abs(shoulder_diff[1]) > abs(shoulder_diff[2]):
        print("Y axis: Left-Right")
    else:
        print("Z axis: Left-Right")

    if abs(nose_ankle_diff[0]) > abs(nose_ankle_diff[1]) and abs(nose_ankle_diff[0]) > abs(nose_ankle_diff[2]):
        print("X axis: Up-Down")
    elif abs(nose_ankle_diff[1]) > abs(nose_ankle_diff[1]) and abs(nose_ankle_diff[1]) > abs(nose_ankle_diff[2]):
        print("Y axis: Up-Down")
    else:
        print("Z axis: Up-Down")


def analyze_wrist_trajectory(timeseries, keypoints):
    """Analyze wrist trajectory during release."""
    keypoint_idx = {name: i for i, name in enumerate(keypoints)}
    wrist_idx = keypoint_idx['right_wrist']

    print("\n=== WRIST TRAJECTORY ===")

    frames = list(range(145, 165))
    positions = []

    for f in frames:
        pos = timeseries[f, wrist_idx*3:(wrist_idx+1)*3]
        positions.append(pos)
        if f in [150, 153, 156]:
            print(f"Frame {f}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")

    positions = np.array(positions)

    # Velocity at release
    dt = 1/60
    vel_153 = (positions[8] - positions[5]) / (3*dt)  # frame 153 - 150
    print(f"\nVelocity at release (frame 153): vx={vel_153[0]:.3f}, vy={vel_153[1]:.3f}, vz={vel_153[2]:.3f}")
    print(f"Speed: {np.linalg.norm(vel_153):.3f} units/sec")

    # Direction
    speed = np.linalg.norm(vel_153)
    if speed > 0:
        direction = vel_153 / speed
        print(f"Direction: x={direction[0]:.3f}, y={direction[1]:.3f}, z={direction[2]:.3f}")


def main():
    keypoints = get_keypoint_info()
    print(f"Keypoints: {len(keypoints)}")

    # Load first few shots
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i == 0:
            print(f"\nShot {i}: Player {metadata['participant_id']}")
            print(f"Timeseries shape: {timeseries.shape}")

            analyze_coordinate_system(timeseries, keypoints)
            analyze_wrist_trajectory(timeseries, keypoints)
            visualize_shot(timeseries, keypoints, i)

        if i >= 2:
            break


if __name__ == "__main__":
    main()
