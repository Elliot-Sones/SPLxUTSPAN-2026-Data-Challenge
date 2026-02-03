"""
Export skeleton trajectory data to MuJoCo XML with keyframes.

This creates an animated visualization showing the actual skeleton motion
from the data, so we can see exactly what the hand is doing during the shot.
"""

import numpy as np
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


def extract_trajectory(timeseries, keypoint_map, start_frame=60, end_frame=180):
    """Extract key body positions for animation."""
    trajectory = {
        'frames': [],
        'right_wrist': [],
        'right_elbow': [],
        'right_shoulder': [],
        'left_wrist': [],
        'left_elbow': [],
        'left_shoulder': [],
        'head': [],
        'pelvis': [],  # torso base
        'right_finger': [],  # shooting hand fingertip
    }

    for frame in range(start_frame, end_frame):
        # Right arm (shooting hand)
        rw = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        re = get_position(timeseries, keypoint_map, 'right_elbow', frame)
        rs = get_position(timeseries, keypoint_map, 'right_shoulder', frame)
        rf = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)

        # Left arm (guide hand)
        lw = get_position(timeseries, keypoint_map, 'left_wrist', frame)
        le = get_position(timeseries, keypoint_map, 'left_elbow', frame)
        ls = get_position(timeseries, keypoint_map, 'left_shoulder', frame)

        # Body
        head = get_position(timeseries, keypoint_map, 'nose', frame)  # Use nose for head
        pelvis = get_position(timeseries, keypoint_map, 'mid_hip', frame)  # Use mid_hip for pelvis

        # Only add if we have the key positions
        if rw is not None and rs is not None and pelvis is not None:
            trajectory['frames'].append(frame)
            trajectory['right_wrist'].append(rw)
            trajectory['right_elbow'].append(re if re is not None else rw)
            trajectory['right_shoulder'].append(rs)
            trajectory['left_wrist'].append(lw if lw is not None else ls)
            trajectory['left_elbow'].append(le if le is not None else lw if lw is not None else ls)
            trajectory['left_shoulder'].append(ls if ls is not None else rs - np.array([0.4, 0, 0]))
            trajectory['head'].append(head if head is not None else (pelvis + np.array([0, 0, 0.5]) if pelvis is not None else np.array([0, 0, 2])))
            trajectory['pelvis'].append(pelvis)
            trajectory['right_finger'].append(rf if rf is not None else rw + np.array([0.1, 0, 0]))

    # Convert to arrays
    for key in trajectory:
        if key != 'frames':
            trajectory[key] = np.array(trajectory[key])

    return trajectory


def compute_velocities(trajectory):
    """Compute velocities using Savgol filter."""
    velocities = {}
    window = 11

    for key in ['right_wrist', 'right_finger', 'pelvis']:
        pos = trajectory[key]
        if len(pos) > window:
            vel = np.zeros_like(pos)
            for j in range(3):
                vel[:, j] = savgol_filter(pos[:, j], window, 3, deriv=1) * FPS
            velocities[key] = vel

    return velocities


def generate_mujoco_xml(trajectory, velocities, shot_id, targets):
    """Generate MuJoCo XML with skeleton and keyframes."""

    # Find key moments
    finger_vz = velocities['right_finger'][:, 2]
    peak_idx = np.argmax(finger_vz)
    peak_frame = trajectory['frames'][peak_idx]

    # Ball should release around peak velocity
    release_pos = trajectory['right_finger'][peak_idx]
    release_vel = velocities['right_finger'][peak_idx]

    # Estimate hoop position (15 feet from player in -X direction based on typical setup)
    pelvis_pos = trajectory['pelvis'][peak_idx]
    hoop_pos = [pelvis_pos[0] - 4.57, pelvis_pos[1], 3.05]  # 15 feet toward -X

    # Generate keyframes for every 2nd frame (30 fps animation)
    keyframes_xml = []

    # Sample frames for keyframes
    sample_indices = list(range(0, len(trajectory['frames']), 2))

    for i, idx in enumerate(sample_indices):
        # Positions
        pelvis = trajectory['pelvis'][idx]
        rs = trajectory['right_shoulder'][idx]
        re = trajectory['right_elbow'][idx]
        rw = trajectory['right_wrist'][idx]
        rf = trajectory['right_finger'][idx]
        ls = trajectory['left_shoulder'][idx]
        le = trajectory['left_elbow'][idx]
        lw = trajectory['left_wrist'][idx]
        head = trajectory['head'][idx]

        # Ball follows hand until release, then free falls
        if idx <= peak_idx:
            ball_pos = rf + np.array([0, 0, 0.12])  # On top of finger
            ball_vel = velocities['right_finger'][idx] if idx < len(velocities['right_finger']) else np.zeros(3)
        else:
            # After release - compute projectile motion
            dt = (idx - peak_idx) / FPS
            ball_pos = release_pos + release_vel * dt + np.array([0, 0, -0.5 * 9.81 * dt * dt])
            ball_vel = release_vel + np.array([0, 0, -9.81 * dt])

        frame_num = trajectory['frames'][idx]

        keyframes_xml.append(f'''        <key name="f{frame_num}" time="{i * 2 / FPS:.3f}"
             qpos="{pelvis[0]:.4f} {pelvis[1]:.4f} {pelvis[2]:.4f}
                    {rs[0]-pelvis[0]-0.2:.4f} {rs[1]-pelvis[1]:.4f} {rs[2]-pelvis[2]-0.4:.4f}
                    {re[0]-rs[0]-0.15:.4f} {re[1]-rs[1]:.4f} {re[2]-rs[2]+0.15:.4f}
                    {rw[0]-re[0]-0.1:.4f} {rw[1]-re[1]:.4f} {rw[2]-re[2]+0.15:.4f}
                    {rf[0]-rw[0]-0.05:.4f} {rf[1]-rw[1]:.4f} {rf[2]-rw[2]+0.05:.4f}
                    {ls[0]-pelvis[0]+0.2:.4f} {ls[1]-pelvis[1]:.4f} {ls[2]-pelvis[2]-0.4:.4f}
                    {le[0]-ls[0]+0.15:.4f} {le[1]-ls[1]:.4f} {le[2]-ls[2]+0.15:.4f}
                    {lw[0]-le[0]+0.1:.4f} {lw[1]-le[1]:.4f} {lw[2]-le[2]+0.15:.4f}
                    {ball_pos[0]:.4f} {ball_pos[1]:.4f} {ball_pos[2]:.4f} 1 0 0 0"/>''')

    keyframes_str = '\n'.join(keyframes_xml)

    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="skeleton_shot_{shot_id[:8]}">
    <!--
    Actual Skeleton Trajectory from Data

    Shot ID: {shot_id}
    Targets: angle={targets['angle']:.2f}, depth={targets['depth']:.2f}, left_right={targets['left_right']:.2f}

    Peak hand velocity at frame {peak_frame}
    Release position: [{release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f}] m
    Release velocity: [{release_vel[0]:.2f}, {release_vel[1]:.2f}, {release_vel[2]:.2f}] m/s

    KEY OBSERVATION: Look at the release velocity!
    - Horizontal speed: {np.sqrt(release_vel[0]**2 + release_vel[1]**2):.2f} m/s ({np.sqrt(release_vel[0]**2 + release_vel[1]**2)/FEET_TO_METERS:.1f} ft/s)
    - Vertical speed: {release_vel[2]:.2f} m/s ({release_vel[2]/FEET_TO_METERS:.1f} ft/s)
    - Total speed: {np.linalg.norm(release_vel):.2f} m/s ({np.linalg.norm(release_vel)/FEET_TO_METERS:.1f} ft/s)

    To reach hoop 15ft away, need ~6-7 m/s horizontal velocity!

    USE: Click through keyframes (0, 1, 2...) and Load key to step through animation
         Or set Key to 0, Load key, then Run for continuous playback
    -->

    <option gravity="0 0 -9.81" timestep="0.002"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
    </visual>

    <asset>
        <material name="court" rgba="0.76 0.60 0.42 1"/>
        <material name="ball" rgba="1 0.5 0 1"/>
        <material name="body" rgba="0.2 0.6 0.9 1"/>
        <material name="hand" rgba="0.85 0.7 0.55 1"/>
        <material name="rim" rgba="1 0.3 0 1"/>
        <material name="trajectory" rgba="0 1 0 0.3"/>
    </asset>

    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>

        <!-- Court -->
        <geom name="court" type="plane" size="15 15 0.1" material="court"/>

        <!-- Player position marker -->
        <geom name="player_marker" type="cylinder" pos="{pelvis_pos[0]:.2f} {pelvis_pos[1]:.2f} 0.01" size="0.3 0.01" rgba="0 0.8 0 0.5"/>

        <!-- Hoop (15 feet from player in -X direction) -->
        <body name="hoop" pos="{hoop_pos[0]:.2f} {hoop_pos[1]:.2f} {hoop_pos[2]:.2f}">
            <geom name="rim" type="cylinder" size="0.229 0.015" material="rim"/>
            <geom name="backboard" type="box" pos="-0.15 0 0.45" size="0.02 0.91 0.61" rgba="1 1 1 0.8"/>
            <geom name="pole" type="cylinder" pos="-0.4 0 -1.5" size="0.08 1.5" rgba="0.3 0.3 0.3 1"/>
        </body>

        <!-- Trajectory markers showing where ball actually goes -->
        <geom name="release_marker" type="sphere" pos="{release_pos[0]:.3f} {release_pos[1]:.3f} {release_pos[2]:.3f}" size="0.05" rgba="1 1 0 0.8"/>

        <!-- SKELETON -->
        <body name="torso" pos="0 0 0">
            <joint name="torso_x" type="slide" axis="1 0 0"/>
            <joint name="torso_y" type="slide" axis="0 1 0"/>
            <joint name="torso_z" type="slide" axis="0 0 1"/>
            <geom name="torso_geom" type="capsule" size="0.12" fromto="0 0 0 0 0 0.5" material="body"/>
            <geom name="head_geom" type="sphere" size="0.1" pos="0 0 0.6" material="body"/>

            <!-- Right arm -->
            <body name="right_shoulder" pos="0.2 0 0.4">
                <joint name="rs_x" type="slide" axis="1 0 0"/>
                <joint name="rs_y" type="slide" axis="0 1 0"/>
                <joint name="rs_z" type="slide" axis="0 0 1"/>
                <geom name="rs_geom" type="sphere" size="0.05" material="body"/>

                <body name="right_elbow" pos="0.15 0 -0.15">
                    <joint name="re_x" type="slide" axis="1 0 0"/>
                    <joint name="re_y" type="slide" axis="0 1 0"/>
                    <joint name="re_z" type="slide" axis="0 0 1"/>
                    <geom name="r_upperarm" type="capsule" size="0.035" fromto="-0.15 0 0.15 0 0 0" material="body"/>
                    <geom name="re_geom" type="sphere" size="0.04" material="body"/>

                    <body name="right_wrist" pos="0.1 0 -0.15">
                        <joint name="rw_x" type="slide" axis="1 0 0"/>
                        <joint name="rw_y" type="slide" axis="0 1 0"/>
                        <joint name="rw_z" type="slide" axis="0 0 1"/>
                        <geom name="r_forearm" type="capsule" size="0.03" fromto="-0.1 0 0.15 0 0 0" material="body"/>
                        <geom name="rw_geom" type="sphere" size="0.035" material="hand"/>

                        <body name="right_hand" pos="0.05 0 -0.05">
                            <joint name="rh_x" type="slide" axis="1 0 0"/>
                            <joint name="rh_y" type="slide" axis="0 1 0"/>
                            <joint name="rh_z" type="slide" axis="0 0 1"/>
                            <geom name="rh_geom" type="cylinder" size="0.06 0.012" material="hand"/>
                        </body>
                    </body>
                </body>
            </body>

            <!-- Left arm -->
            <body name="left_shoulder" pos="-0.2 0 0.4">
                <joint name="ls_x" type="slide" axis="1 0 0"/>
                <joint name="ls_y" type="slide" axis="0 1 0"/>
                <joint name="ls_z" type="slide" axis="0 0 1"/>
                <geom name="ls_geom" type="sphere" size="0.05" material="body"/>

                <body name="left_elbow" pos="-0.15 0 -0.15">
                    <joint name="le_x" type="slide" axis="1 0 0"/>
                    <joint name="le_y" type="slide" axis="0 1 0"/>
                    <joint name="le_z" type="slide" axis="0 0 1"/>
                    <geom name="l_upperarm" type="capsule" size="0.035" fromto="0.15 0 0.15 0 0 0" material="body"/>
                    <geom name="le_geom" type="sphere" size="0.04" material="body"/>

                    <body name="left_wrist" pos="-0.1 0 -0.15">
                        <joint name="lw_x" type="slide" axis="1 0 0"/>
                        <joint name="lw_y" type="slide" axis="0 1 0"/>
                        <joint name="lw_z" type="slide" axis="0 0 1"/>
                        <geom name="l_forearm" type="capsule" size="0.03" fromto="0.1 0 0.15 0 0 0" material="body"/>
                        <geom name="lw_geom" type="sphere" size="0.035" material="hand"/>
                    </body>
                </body>
            </body>
        </body>

        <!-- Basketball -->
        <body name="ball" pos="0 0 2">
            <freejoint name="ball_free"/>
            <geom name="basketball" type="sphere" size="0.12" material="ball" mass="0.625"/>
        </body>

    </worldbody>

    <keyframe>
{keyframes_str}
    </keyframe>

</mujoco>
'''
    return xml


def main():
    print("=" * 80)
    print("EXPORTING SKELETON TRAJECTORY TO MUJOCO")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Get first valid shot
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        trajectory = extract_trajectory(timeseries, keypoint_map)

        if len(trajectory['frames']) < 30:
            continue

        velocities = compute_velocities(trajectory)

        if 'right_finger' not in velocities:
            continue

        shot_id = metadata['id']
        targets = {
            'angle': metadata.get('angle', 0),
            'depth': metadata.get('depth', 0),
            'left_right': metadata.get('left_right', 0),
        }

        print(f"\nProcessing shot: {shot_id[:8]}")
        print(f"  Targets: angle={targets['angle']:.2f}, depth={targets['depth']:.2f}, left_right={targets['left_right']:.2f}")
        print(f"  Frames: {trajectory['frames'][0]} to {trajectory['frames'][-1]}")

        # Find peak velocity
        finger_vz = velocities['right_finger'][:, 2]
        peak_idx = np.argmax(finger_vz)
        peak_frame = trajectory['frames'][peak_idx]
        release_vel = velocities['right_finger'][peak_idx]

        print(f"\n  Peak velocity at frame {peak_frame}:")
        print(f"    Vx: {release_vel[0]:.2f} m/s ({release_vel[0]/FEET_TO_METERS:.1f} ft/s)")
        print(f"    Vy: {release_vel[1]:.2f} m/s ({release_vel[1]/FEET_TO_METERS:.1f} ft/s)")
        print(f"    Vz: {release_vel[2]:.2f} m/s ({release_vel[2]/FEET_TO_METERS:.1f} ft/s)")

        horiz_speed = np.sqrt(release_vel[0]**2 + release_vel[1]**2)
        print(f"\n  Horizontal speed: {horiz_speed:.2f} m/s ({horiz_speed/FEET_TO_METERS:.1f} ft/s)")
        print(f"  REQUIRED for 15ft shot: ~6.5 m/s (21 ft/s)")
        print(f"  MISSING: {6.5 - horiz_speed:.2f} m/s ({(6.5 - horiz_speed)/FEET_TO_METERS:.1f} ft/s)")

        # Generate XML
        xml = generate_mujoco_xml(trajectory, velocities, shot_id, targets)

        output_path = PROJECT_DIR / "physics_engine" / "skeleton_shot.xml"
        with open(output_path, 'w') as f:
            f.write(xml)

        print(f"\n  Saved to: {output_path}")
        print(f"\n  To visualize:")
        print(f"    1. Open MuJoCo app")
        print(f"    2. Drag {output_path.name} into the window")
        print(f"    3. Use Key field (0, 1, 2...) and Load key to step through frames")

        break

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
