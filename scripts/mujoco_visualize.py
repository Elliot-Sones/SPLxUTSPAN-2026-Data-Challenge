"""
Visualize the MuJoCo simulation to understand the data-physics disconnect.

This will show:
1. The hand trajectory from skeleton data
2. The ball being pushed and released
3. The actual trajectory vs what a real free throw should look like
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from scipy.signal import savgol_filter
import time
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

BALL_RADIUS = 0.12
FEET_TO_METERS = 0.3048
FPS = 60


def create_visualization_model():
    """
    Create a model with:
    - Court floor
    - Hoop (approximate position)
    - Hand platform
    - Ball
    - Visual markers for trajectory
    """
    xml = """
    <mujoco model="basketball_viz">
        <option gravity="0 0 -9.81" timestep="0.001"/>

        <visual>
            <global offwidth="1280" offheight="720"/>
        </visual>

        <asset>
            <material name="floor" rgba="0.4 0.3 0.2 1"/>
            <material name="hoop" rgba="1.0 0.5 0.0 1"/>
            <material name="ball" rgba="1.0 0.5 0.0 1"/>
            <material name="hand" rgba="0.9 0.75 0.6 1"/>
            <material name="target" rgba="0.0 1.0 0.0 0.5"/>
        </asset>

        <worldbody>
            <!-- Court floor -->
            <geom type="plane" size="15 15 0.1" material="floor"/>

            <!-- Free throw line marker (at player position) -->
            <geom type="box" pos="5.7 -7.6 0.01" size="0.5 0.05 0.01" rgba="1 1 1 1"/>

            <!-- Hoop (approximate - we'll see where it should be) -->
            <!-- If player is at x=5.7, y=-7.6 and shoots toward -X direction -->
            <!-- Hoop would be at approximately x=5.7-4.57=-9.1 or toward y direction -->
            <!-- Let's put a marker where we THINK the hoop should be -->

            <!-- Option 1: Hoop in -X direction (15 feet from player) -->
            <body name="hoop_x" pos="1.1 -7.6 3.05">
                <geom type="cylinder" size="0.23 0.02" rgba="1 0.5 0 1" euler="0 90 0"/>
                <geom type="box" size="0.05 0.05 0.5" pos="0 0.3 0.5" rgba="0.5 0.5 0.5 1"/>
            </body>

            <!-- Option 2: Hoop in -Y direction (15 feet from player) -->
            <body name="hoop_y" pos="5.7 -12.2 3.05">
                <geom type="cylinder" size="0.23 0.02" rgba="0 1 0 0.5" euler="90 0 0"/>
            </body>

            <!-- Hand platform -->
            <body name="hand" pos="5.7 -7.6 1.8">
                <joint name="hx" type="slide" axis="1 0 0"/>
                <joint name="hy" type="slide" axis="0 1 0"/>
                <joint name="hz" type="slide" axis="0 0 1"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" material="hand"/>
            </body>

            <!-- Ball -->
            <body name="ball" pos="5.7 -7.6 2.0">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" material="ball"/>
            </body>

            <!-- Trajectory markers (we'll show where ball goes) -->
            <body name="release_marker" pos="0 0 0">
                <geom type="sphere" size="0.05" rgba="0 1 0 0.8"/>
            </body>

            <!-- Camera -->
            <camera name="side_view" pos="12 -7.6 4" xyaxes="0 -1 0 0 0 1"/>
            <camera name="top_view" pos="5.7 -7.6 12" xyaxes="1 0 0 0 1 0"/>
            <camera name="shooter_view" pos="8 -7.6 2" xyaxes="0 -1 0 0.3 0 1"/>

        </worldbody>

        <actuator>
            <position name="hx_ctrl" joint="hx" kp="50000" kv="5000"/>
            <position name="hy_ctrl" joint="hy" kp="50000" kv="5000"/>
            <position name="hz_ctrl" joint="hz" kp="50000" kv="5000"/>
        </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


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
    return pos


def get_hand_trajectory(timeseries, keypoint_map):
    positions = []
    frames = []
    for frame in range(50, 200):
        pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
        if pos is None:
            pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if pos is not None:
            positions.append(pos * FEET_TO_METERS)  # Convert to meters
            frames.append(frame)
    return (np.array(positions), np.array(frames)) if len(positions) > 20 else (None, None)


def main():
    print("=" * 80)
    print("MUJOCO VISUALIZATION")
    print("=" * 80)
    print("\nThis will open an interactive viewer showing:")
    print("  - Orange hoop: Expected hoop position in -X direction")
    print("  - Green hoop: Expected hoop position in -Y direction")
    print("  - Orange ball: Basketball being shot")
    print("  - White line: Free throw line marker")
    print("\nControls:")
    print("  - Mouse drag: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Double-click: Reset view")
    print("  - ESC or close window: Exit")

    model = create_visualization_model()
    data = mujoco.MjData(model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Get first shot
    shot_data = None
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
        if hand_pos is not None:
            shot_data = {
                'id': metadata['id'],
                'hand_pos': hand_pos,
                'frames': frames,
                'angle': metadata.get('angle'),
                'depth': metadata.get('depth'),
                'left_right': metadata.get('left_right'),
            }
            break

    if shot_data is None:
        print("No valid shot data found!")
        return

    print(f"\nLoaded shot: {shot_data['id'][:8]}")
    print(f"  Targets: angle={shot_data['angle']:.1f}, depth={shot_data['depth']:.1f}, left_right={shot_data['left_right']:.1f}")

    # Compute velocities
    hand_pos = shot_data['hand_pos']
    frames = shot_data['frames']

    window = 11
    hand_vel = np.zeros_like(hand_pos)
    for j in range(3):
        hand_vel[:, j] = savgol_filter(hand_pos[:, j], window, 3, deriv=1) * FPS

    # Find start
    start_idx = 0
    for j in range(len(hand_vel)):
        if hand_vel[j, 2] > 0.3:
            start_idx = max(0, j - 10)
            break

    print(f"\n  Hand position range:")
    print(f"    X: {hand_pos[:, 0].min():.2f} to {hand_pos[:, 0].max():.2f} m")
    print(f"    Y: {hand_pos[:, 1].min():.2f} to {hand_pos[:, 1].max():.2f} m")
    print(f"    Z: {hand_pos[:, 2].min():.2f} to {hand_pos[:, 2].max():.2f} m")

    print(f"\n  Hand velocity at peak:")
    peak_idx = np.argmax(hand_vel[:, 2])
    print(f"    Vx: {hand_vel[peak_idx, 0]:.2f} m/s")
    print(f"    Vy: {hand_vel[peak_idx, 1]:.2f} m/s")
    print(f"    Vz: {hand_vel[peak_idx, 2]:.2f} m/s")

    # Initialize simulation
    mujoco.mj_resetData(model, data)

    init_hand = hand_pos[start_idx]
    # Hand body default is at [5.7, -7.6, 1.8]
    hand_default = np.array([5.7, -7.6, 1.8])
    hand_offset = init_hand - hand_default

    data.qpos[0:3] = hand_offset  # Hand joints
    palm_top_z = hand_default[2] + hand_offset[2] + 0.02
    ball_z = palm_top_z + BALL_RADIUS - 0.005

    data.qpos[3:6] = [init_hand[0], init_hand[1], ball_z]  # Ball position
    data.qpos[6:10] = [1, 0, 0, 0]  # Ball quaternion

    data.ctrl[0:3] = hand_offset

    mujoco.mj_forward(model, data)

    print("\nStarting visualization...")
    print("Watch the ball trajectory and compare to hoop positions!")

    # Run with viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera
        viewer.cam.lookat[:] = [5.7, -7.6, 2.5]
        viewer.cam.distance = 8.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        frame_idx = start_idx
        sim_time = 0
        frame_dt = 1.0 / FPS
        released = False
        release_pos = None

        print("\nSimulation running...")
        print("  Frame | Hand Vz | Ball Vz | Ball Pos")
        print("  " + "-" * 50)

        while viewer.is_running():
            step_start = time.time()

            # Update hand position
            if frame_idx < len(hand_pos) - 1:
                target = hand_pos[frame_idx]
                target_offset = target - hand_default

                data.ctrl[0] = target_offset[0]
                data.ctrl[1] = target_offset[1]
                data.ctrl[2] = target_offset[2]

                frame_idx += 1

            # Step simulation
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep

            # Check for release
            ball_vel = data.qvel[3:6]
            ball_pos = data.qpos[3:6]

            if not released and sim_time > 0.5 and ball_vel[2] < 0:
                released = True
                release_pos = ball_pos.copy()
                print(f"\n  RELEASE detected at {ball_pos}")
                print(f"  Ball velocity: [{ball_vel[0]:.2f}, {ball_vel[1]:.2f}, {ball_vel[2]:.2f}] m/s")

            # Print status every 0.1 seconds
            if int(sim_time * 10) != int((sim_time - model.opt.timestep) * 10):
                if frame_idx < len(frames):
                    h_vz = hand_vel[min(frame_idx, len(hand_vel)-1), 2]
                    print(f"  {frames[min(frame_idx, len(frames)-1)]:5d} | {h_vz:7.2f} | {ball_vel[2]:7.2f} | "
                          f"[{ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f}]")

            # Sync to realtime (slowed down for visibility)
            viewer.sync()

            # Slow down for visibility
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep * 2 - elapsed  # 2x slower than realtime
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Stop after ball falls below ground or goes too far
            if ball_pos[2] < -0.5 or sim_time > 3.0:
                print(f"\n  Simulation ended at t={sim_time:.2f}s")
                print(f"  Final ball position: [{ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f}]")

                if release_pos is not None:
                    # Calculate where ball went relative to hoops
                    hoop_x_pos = np.array([1.1, -7.6, 3.05])  # Orange hoop
                    hoop_y_pos = np.array([5.7, -12.2, 3.05])  # Green hoop

                    dist_to_hoop_x = np.linalg.norm(release_pos[:2] - hoop_x_pos[:2])
                    dist_to_hoop_y = np.linalg.norm(release_pos[:2] - hoop_y_pos[:2])

                    print(f"\n  Distance from release to hoops:")
                    print(f"    Orange hoop (-X direction): {dist_to_hoop_x:.2f} m")
                    print(f"    Green hoop (-Y direction):  {dist_to_hoop_y:.2f} m")

                # Wait for user to close
                print("\n  Close the viewer window to exit...")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.1)
                break


if __name__ == "__main__":
    main()
