"""
Proper MuJoCo visualization of skeleton data.

Creates a model with all keypoints as spheres, connected by capsules.
Exports keyframes that can be stepped through in MuJoCo viewer.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

FEET_TO_METERS = 0.3048
FPS = 60


def get_keypoint_map(keypoint_cols):
    keypoint_map = {}
    for i, col in enumerate(keypoint_cols):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in keypoint_map:
                keypoint_map[name] = {}
            keypoint_map[name][axis] = i
    return keypoint_map


def get_position(timeseries, keypoint_map, name, frame):
    if name not in keypoint_map:
        return None
    km = keypoint_map[name]
    if 'x' not in km or 'y' not in km or 'z' not in km:
        return None
    pos = np.array([
        timeseries[frame, km['x']],
        timeseries[frame, km['y']],
        timeseries[frame, km['z']]
    ])
    if np.any(np.isnan(pos)):
        return None
    return pos * FEET_TO_METERS  # Convert to meters


# Key skeleton connections for visualization
SKELETON_CONNECTIONS = [
    # Spine
    ('mid_hip', 'mid_spine'),
    ('mid_spine', 'neck'),
    ('neck', 'nose'),

    # Right arm
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('right_wrist', 'right_third_finger_distal'),

    # Left arm
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),

    # Right leg
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),

    # Left leg
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),

    # Shoulders
    ('left_shoulder', 'right_shoulder'),

    # Hips
    ('left_hip', 'right_hip'),
]


def create_skeleton_xml(shot_data, shot_id, targets, hoop_direction='x'):
    """Generate MuJoCo XML with skeleton keyframes."""

    frames = shot_data['frames']
    positions = shot_data['positions']  # dict of keypoint -> array of positions

    # Find approximate center of player
    hip_pos = positions.get('mid_hip', positions.get('right_hip', None))
    if hip_pos is None:
        return None

    center = hip_pos[0]  # First frame position as reference

    # Determine hoop position (15 feet = 4.57m from player)
    hoop_distance = 4.57
    if hoop_direction == 'x':
        hoop_pos = [center[0] - hoop_distance, center[1], 3.05]
    else:
        hoop_pos = [center[0], center[1] - hoop_distance, 3.05]

    # Start building XML
    xml_parts = [f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="skeleton_{shot_id}">
    <!--
    Skeleton Visualization - Shot {shot_id}
    Targets: angle={targets.get('angle', 0):.2f}, depth={targets.get('depth', 0):.2f}, left_right={targets.get('left_right', 0):.2f}

    INSTRUCTIONS:
    1. Set Key to 0, click Load key
    2. Increment Key (1, 2, 3...) and Load key to step through frames
    3. Or click Run to animate

    Watch the hand (tan sphere) and ball (orange) - notice how the hand
    moves mostly UPWARD with very little horizontal movement toward the hoop.
    -->

    <option gravity="0 0 -9.81" timestep="0.002"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8"/>
    </visual>

    <asset>
        <material name="court" rgba="0.76 0.60 0.42 1"/>
        <material name="ball" rgba="1 0.5 0 1"/>
        <material name="joint" rgba="0.2 0.6 0.9 1"/>
        <material name="hand" rgba="0.9 0.75 0.6 1"/>
        <material name="rim" rgba="1 0.3 0 1"/>
        <material name="bone" rgba="0.3 0.5 0.8 0.8"/>
        <material name="velocity" rgba="1 0 0 1"/>
    </asset>

    <worldbody>
        <light pos="{center[0]:.2f} {center[1]:.2f} 10" dir="0 0 -1" diffuse="1 1 1"/>

        <!-- Court floor -->
        <geom name="court" type="plane" size="20 20 0.1" material="court"/>

        <!-- Player position marker -->
        <geom name="player_marker" type="cylinder" pos="{center[0]:.3f} {center[1]:.3f} 0.01"
              size="0.3 0.01" rgba="0 0.8 0 0.5"/>

        <!-- Hoop -->
        <body name="hoop" pos="{hoop_pos[0]:.3f} {hoop_pos[1]:.3f} {hoop_pos[2]:.3f}">
            <geom name="rim" type="cylinder" size="0.229 0.02" material="rim"/>
            <geom name="backboard" type="box" pos="{-0.2 if hoop_direction == 'x' else 0:.2f} {0 if hoop_direction == 'x' else -0.2:.2f} 0.45"
                  size="{'0.02 0.91 0.61' if hoop_direction == 'x' else '0.91 0.02 0.61'}" rgba="1 1 1 0.9"/>
        </body>

        <!-- Distance marker on floor (line from player to hoop) -->
        <geom name="distance_line" type="capsule"
              fromto="{center[0]:.3f} {center[1]:.3f} 0.02 {hoop_pos[0]:.3f} {hoop_pos[1]:.3f} 0.02"
              size="0.02" rgba="1 1 0 0.5"/>
''']

    # Add skeleton keypoints as bodies with freejoints
    keypoints_to_add = [
        ('mid_hip', 0.06, 'joint'),
        ('mid_spine', 0.05, 'joint'),
        ('neck', 0.04, 'joint'),
        ('nose', 0.08, 'joint'),  # Head
        ('right_shoulder', 0.05, 'joint'),
        ('right_elbow', 0.04, 'joint'),
        ('right_wrist', 0.04, 'hand'),
        ('right_third_finger_distal', 0.03, 'hand'),
        ('left_shoulder', 0.05, 'joint'),
        ('left_elbow', 0.04, 'joint'),
        ('left_wrist', 0.04, 'joint'),
        ('right_hip', 0.05, 'joint'),
        ('right_knee', 0.04, 'joint'),
        ('right_ankle', 0.04, 'joint'),
        ('left_hip', 0.05, 'joint'),
        ('left_knee', 0.04, 'joint'),
        ('left_ankle', 0.04, 'joint'),
    ]

    valid_keypoints = []
    for kp_name, size, material in keypoints_to_add:
        if kp_name in positions and len(positions[kp_name]) > 0:
            init_pos = positions[kp_name][0]
            xml_parts.append(f'''
        <body name="{kp_name}" pos="{init_pos[0]:.4f} {init_pos[1]:.4f} {init_pos[2]:.4f}">
            <freejoint name="{kp_name}_joint"/>
            <geom type="sphere" size="{size}" material="{material}"/>
        </body>''')
            valid_keypoints.append(kp_name)

    # Add basketball
    finger_pos = positions.get('right_third_finger_distal', positions.get('right_wrist'))
    if finger_pos is not None:
        ball_init = finger_pos[0] + np.array([0, 0, 0.15])
        xml_parts.append(f'''
        <body name="ball" pos="{ball_init[0]:.4f} {ball_init[1]:.4f} {ball_init[2]:.4f}">
            <freejoint name="ball_joint"/>
            <geom type="sphere" size="0.12" material="ball" mass="0.625"/>
        </body>''')

    xml_parts.append('''
    </worldbody>

    <keyframe>''')

    # Generate keyframes - sample every 2 frames for smoother playback
    frame_step = 2
    sampled_indices = list(range(0, len(frames), frame_step))

    # Calculate velocities for ball trajectory after release
    if 'right_third_finger_distal' in positions:
        finger_positions = positions['right_third_finger_distal']
        finger_vel = np.zeros_like(finger_positions)
        for j in range(3):
            finger_vel[:, j] = savgol_filter(finger_positions[:, j], 11, 3, deriv=1) * FPS
        peak_idx = np.argmax(finger_vel[:, 2])  # Peak upward velocity
    else:
        peak_idx = len(frames) // 2
        finger_vel = None

    for key_idx, sample_idx in enumerate(sampled_indices):
        frame_num = frames[sample_idx]

        # Build qpos string - 7 DOF per body (3 pos + 4 quat)
        qpos_parts = []

        for kp_name in valid_keypoints:
            pos = positions[kp_name][sample_idx]
            qpos_parts.append(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} 1 0 0 0")

        # Ball position - follows hand until release, then projectile
        if finger_pos is not None:
            if sample_idx <= peak_idx:
                ball_pos = positions['right_third_finger_distal'][sample_idx] + np.array([0, 0, 0.15])
            else:
                # Projectile motion after release
                release_pos = positions['right_third_finger_distal'][peak_idx] + np.array([0, 0, 0.15])
                release_vel = finger_vel[peak_idx] if finger_vel is not None else np.array([0, 0, 3])
                dt = (sample_idx - peak_idx) / FPS * frame_step
                ball_pos = release_pos + release_vel * dt + np.array([0, 0, -0.5 * 9.81 * dt * dt])

            qpos_parts.append(f"{ball_pos[0]:.4f} {ball_pos[1]:.4f} {ball_pos[2]:.4f} 1 0 0 0")

        qpos_str = "  ".join(qpos_parts)

        xml_parts.append(f'''
        <key name="f{frame_num}" time="{key_idx * frame_step / FPS:.3f}" qpos="{qpos_str}"/>''')

    xml_parts.append('''
    </keyframe>
</mujoco>''')

    return '\n'.join(xml_parts)


def main():
    print("=" * 80)
    print("CREATING MUJOCO SKELETON VISUALIZATION")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Get first shot
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        shot_id = metadata['id'][:8]
        targets = {
            'angle': metadata.get('angle', 0),
            'depth': metadata.get('depth', 0),
            'left_right': metadata.get('left_right', 0),
        }

        print(f"\nProcessing shot: {shot_id}")
        print(f"Targets: angle={targets['angle']:.2f}, depth={targets['depth']:.2f}, left_right={targets['left_right']:.2f}")

        # Extract all keypoint positions
        frames = list(range(50, 190))
        positions = {}

        keypoints = [
            'mid_hip', 'mid_spine', 'neck', 'nose',
            'right_shoulder', 'right_elbow', 'right_wrist', 'right_third_finger_distal',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle',
        ]

        for kp in keypoints:
            pos_list = []
            for frame in frames:
                pos = get_position(timeseries, keypoint_map, kp, frame)
                if pos is not None:
                    pos_list.append(pos)
                elif len(pos_list) > 0:
                    pos_list.append(pos_list[-1])  # Repeat last valid

            if len(pos_list) == len(frames):
                positions[kp] = np.array(pos_list)

        print(f"Valid keypoints: {list(positions.keys())}")

        shot_data = {
            'frames': frames,
            'positions': positions,
        }

        # Generate XML
        xml = create_skeleton_xml(shot_data, shot_id, targets, hoop_direction='x')

        if xml:
            output_path = PROJECT_DIR / "physics_engine" / "skeleton_viewer.xml"
            with open(output_path, 'w') as f:
                f.write(xml)

            print(f"\nSaved to: {output_path}")
            print(f"\nTo view:")
            print(f"  1. Open MuJoCo app")
            print(f"  2. Drag {output_path.name} into the window")
            print(f"  3. Set Key=0, Load key, then increment Key and Load to step through")
            print(f"  4. Watch hand (tan) and ball (orange) vs hoop (red ring)")

            # Print key velocity info
            if 'right_third_finger_distal' in positions:
                finger = positions['right_third_finger_distal']
                vel = np.zeros_like(finger)
                for j in range(3):
                    vel[:, j] = savgol_filter(finger[:, j], 11, 3, deriv=1) * FPS

                peak_idx = np.argmax(vel[:, 2])
                print(f"\n  Peak velocity at frame {frames[peak_idx]}:")
                print(f"    Vx: {vel[peak_idx, 0]:.2f} m/s ({vel[peak_idx, 0]/FEET_TO_METERS:.1f} ft/s)")
                print(f"    Vy: {vel[peak_idx, 1]:.2f} m/s ({vel[peak_idx, 1]/FEET_TO_METERS:.1f} ft/s)")
                print(f"    Vz: {vel[peak_idx, 2]:.2f} m/s ({vel[peak_idx, 2]/FEET_TO_METERS:.1f} ft/s)")
                horiz = np.sqrt(vel[peak_idx, 0]**2 + vel[peak_idx, 1]**2)
                print(f"    Horizontal: {horiz:.2f} m/s ({horiz/FEET_TO_METERS:.1f} ft/s)")
                print(f"    Required: ~6.5 m/s (21 ft/s) horizontal")

        break

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
