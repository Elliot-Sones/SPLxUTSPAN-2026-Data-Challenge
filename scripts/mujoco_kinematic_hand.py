"""
MuJoCo simulation with kinematically-controlled hand.

Instead of using actuators (which have tracking lag), directly set
the hand qpos and qvel to match the target trajectory. This ensures
the hand follows the exact motion from the data while still applying
contact forces to the ball through physics.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

BALL_RADIUS = 0.12
FEET_TO_METERS = 0.3048
FPS = 60


def create_model():
    """Create model. Hand will be controlled directly via qpos/qvel."""
    xml = """
    <mujoco model="ball_hand_kinematic">
        <option gravity="0 0 -9.81" timestep="0.0002"/>

        <worldbody>
            <geom type="plane" size="20 20 0.1"/>

            <!-- Hand with slide joints (no actuators - direct control) -->
            <body name="hand" pos="0 0 1.5">
                <joint name="hx" type="slide" axis="1 0 0"/>
                <joint name="hy" type="slide" axis="0 1 0"/>
                <joint name="hz" type="slide" axis="0 0 1"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
            </body>

            <!-- Ball with freejoint -->
            <body name="ball" pos="0 0 1.66">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"/>
            </body>
        </worldbody>
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
    return np.array([
        timeseries[frame, km['x']],
        timeseries[frame, km['y']],
        timeseries[frame, km['z']]
    ])


def get_hand_trajectory(timeseries, keypoint_map):
    positions = []
    frames = []
    for frame in range(50, 200):
        pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
        if pos is None:
            pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if pos is not None and not np.any(np.isnan(pos)):
            positions.append(pos * FEET_TO_METERS)
            frames.append(frame)
    return (np.array(positions), np.array(frames)) if len(positions) > 20 else (None, None)


def check_contact(model, data):
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            return True
    return False


def simulate_shot(timeseries, keypoint_map, model, data, verbose=False):
    """Simulate ball-hand contact with kinematic hand control."""

    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    # Compute target velocities from data
    window = min(11, len(hand_pos) - 2)
    if window % 2 == 0:
        window -= 1
    hand_vel_target = np.zeros_like(hand_pos)
    for i in range(3):
        hand_vel_target[:, i] = savgol_filter(hand_pos[:, i], window, 3, deriv=1) * FPS

    # Find shot start
    start_idx = 0
    for i in range(len(hand_vel_target)):
        if hand_vel_target[i, 2] > 0.5:
            start_idx = max(0, i - 3)
            break

    target_max_speed = np.max(np.linalg.norm(hand_vel_target, axis=1))

    # Reset
    mujoco.mj_resetData(model, data)

    init_hand = hand_pos[start_idx]
    init_hand_vel = hand_vel_target[start_idx]
    hand_offset = init_hand - np.array([0, 0, 1.5])

    # Set initial state
    # qpos[0:3] = hand joints, qpos[3:6] = ball pos, qpos[6:10] = ball quat
    data.qpos[0:3] = hand_offset

    palm_top_z = 1.5 + hand_offset[2] + 0.02
    ball_z = palm_top_z + BALL_RADIUS - 0.005  # Small penetration

    data.qpos[3:6] = [init_hand[0], init_hand[1], ball_z]
    data.qpos[6:10] = [1, 0, 0, 0]

    # Set initial velocities
    # qvel[0:3] = hand vel, qvel[3:6] = ball linear vel, qvel[6:9] = ball angular vel
    data.qvel[0:3] = init_hand_vel
    data.qvel[3:6] = init_hand_vel  # Ball starts with hand velocity
    data.qvel[6:9] = [0, 0, 0]

    mujoco.mj_forward(model, data)

    initial_contact = check_contact(model, data)

    if verbose:
        print(f"  Target max speed: {target_max_speed:.2f} m/s ({target_max_speed/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Start frame: {frames[start_idx]}, Contact: {initial_contact}")

    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS

    had_contact = initial_contact
    contact_count = 1 if initial_contact else 0
    no_contact_count = 0
    release_data = None
    max_ball_speed = 0
    max_hand_speed = 0

    # For interpolation between frames
    current_time = 0.0
    frame_time = 0.0

    for idx in range(start_idx, len(hand_pos) - 1):
        # Get current and next frame data
        pos_curr = hand_pos[idx] - np.array([0, 0, 1.5])
        pos_next = hand_pos[idx + 1] - np.array([0, 0, 1.5])
        vel_curr = hand_vel_target[idx]
        vel_next = hand_vel_target[idx + 1]

        frame_time = 0.0

        while frame_time < frame_dt:
            # Interpolate hand position and velocity within frame
            t = frame_time / frame_dt
            hand_offset_interp = pos_curr * (1 - t) + pos_next * t
            hand_vel_interp = vel_curr * (1 - t) + vel_next * t

            # KINEMATIC CONTROL: Directly set hand position and velocity
            data.qpos[0:3] = hand_offset_interp
            data.qvel[0:3] = hand_vel_interp

            # Step physics (this updates ball position based on contact)
            mujoco.mj_step(model, data)

            frame_time += sim_dt
            current_time += sim_dt

        # Record ball state at frame boundary
        ball_vel = data.qvel[3:6].copy()
        ball_speed = np.linalg.norm(ball_vel)
        hand_speed = np.linalg.norm(data.qvel[0:3])

        max_ball_speed = max(max_ball_speed, ball_speed)
        max_hand_speed = max(max_hand_speed, hand_speed)

        in_contact = check_contact(model, data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
            had_contact = True
        else:
            no_contact_count += 1

        if had_contact and contact_count > 3 and no_contact_count >= 3:
            release_data = {
                'frame': frames[idx],
                'ball_pos': data.qpos[3:6].copy(),
                'ball_vel': ball_vel,
                'speed': ball_speed,
                'max_speed': max_ball_speed,
                'max_hand_speed': max_hand_speed,
                'target_max_speed': target_max_speed,
                'contact_frames': contact_count,
            }

            if verbose:
                print(f"  RELEASE at frame {frames[idx]}")
                print(f"  Ball vel: {ball_vel}")
                print(f"  Ball speed: {ball_speed:.2f} m/s ({ball_speed/FEET_TO_METERS:.1f} ft/s)")
                print(f"  Max ball speed: {max_ball_speed:.2f} m/s ({max_ball_speed/FEET_TO_METERS:.1f} ft/s)")
                print(f"  Max hand speed: {max_hand_speed:.2f} m/s ({max_hand_speed/FEET_TO_METERS:.1f} ft/s)")
                print(f"  Tracking: {max_hand_speed/target_max_speed*100:.1f}%")

            break

    if release_data is None:
        ball_vel = data.qvel[3:6].copy()
        ball_speed = np.linalg.norm(ball_vel)

        release_data = {
            'frame': frames[-1],
            'ball_pos': data.qpos[3:6].copy(),
            'ball_vel': ball_vel,
            'speed': ball_speed,
            'max_speed': max_ball_speed,
            'max_hand_speed': max_hand_speed,
            'target_max_speed': target_max_speed,
            'contact_frames': contact_count,
            'no_release': True,
        }

        if verbose:
            print(f"  No release - final speed: {ball_speed:.2f} m/s ({ball_speed/FEET_TO_METERS:.1f} ft/s)")
            print(f"  Max ball speed: {max_ball_speed:.2f} m/s ({max_ball_speed/FEET_TO_METERS:.1f} ft/s)")
            print(f"  Max hand speed: {max_hand_speed:.2f} m/s ({max_hand_speed/FEET_TO_METERS:.1f} ft/s)")
            print(f"  Tracking: {max_hand_speed/target_max_speed*100:.1f}%")
            print(f"  Contact frames: {contact_count}")

    return release_data


def main():
    print("=" * 80)
    print("MUJOCO KINEMATIC HAND SIMULATION")
    print("=" * 80)
    print("\nDirect qpos/qvel control - hand follows exact trajectory from data")

    model = create_model()
    data = mujoco.MjData(model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Quick test with controlled hand motion
    print("\n" + "=" * 60)
    print("TEST: Controlled upward motion at 5 m/s")
    print("=" * 60)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0]  # Hand at default
    data.qpos[3:6] = [0, 0, 1.63]  # Ball on hand
    data.qpos[6:10] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    print(f"Initial contact: {check_contact(model, data)}")

    # Move hand up at 5 m/s
    hand_vel = 5.0  # m/s
    sim_dt = 0.0002
    duration = 0.1  # seconds
    steps = int(duration / sim_dt)

    hand_z = 0.0
    for step in range(steps):
        # Set hand position and velocity kinematically
        hand_z += hand_vel * sim_dt
        data.qpos[0:3] = [0, 0, hand_z]
        data.qvel[0:3] = [0, 0, hand_vel]

        mujoco.mj_step(model, data)

        if step % 100 == 0:
            ball_vel = data.qvel[3:6]
            ball_vz = ball_vel[2]
            contact = check_contact(model, data)
            print(f"  Step {step}: hand_z={hand_z:.3f}, ball_vz={ball_vz:.2f} m/s, contact={contact}")

    ball_vel = data.qvel[3:6]
    print(f"\nFinal ball velocity: {ball_vel}")
    print(f"Final ball speed: {np.linalg.norm(ball_vel):.2f} m/s ({np.linalg.norm(ball_vel)/FEET_TO_METERS:.1f} ft/s)")
    print(f"Velocity transfer: {np.linalg.norm(ball_vel)/hand_vel*100:.1f}%")

    # Process real shots
    print("\n" + "=" * 80)
    print("Processing real shots...")
    print("=" * 80)

    results = []
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 20:
            break

        print(f"\nShot {i+1}: {metadata['id'][:8]}")
        result = simulate_shot(timeseries, keypoint_map, model, data, verbose=True)

        if result:
            results.append(result)

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        ball_speeds = [r['max_speed'] / FEET_TO_METERS for r in results]
        release_speeds = [r['speed'] / FEET_TO_METERS for r in results]
        target_speeds = [r['target_max_speed'] / FEET_TO_METERS for r in results]
        tracking = [r['max_hand_speed'] / r['target_max_speed'] * 100 for r in results]

        print(f"\nTarget hand speeds: {np.mean(target_speeds):.1f} +/- {np.std(target_speeds):.1f} ft/s (max {np.max(target_speeds):.1f})")
        print(f"Max ball speeds: {np.mean(ball_speeds):.1f} +/- {np.std(ball_speeds):.1f} ft/s (max {np.max(ball_speeds):.1f})")
        print(f"Release speeds: {np.mean(release_speeds):.1f} +/- {np.std(release_speeds):.1f} ft/s")
        print(f"Hand tracking: {np.mean(tracking):.1f}% (should be ~100%)")
        print(f"\nRequired: ~22 ft/s")


if __name__ == "__main__":
    main()
