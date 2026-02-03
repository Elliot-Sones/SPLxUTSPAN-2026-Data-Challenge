"""
MuJoCo Ball-Hand Contact Simulation v3

Fixed: Ensure ball actually contacts hand at start.
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

BALL_RADIUS = 0.12  # meters
FEET_TO_METERS = 0.3048
FPS = 60


def create_model():
    """
    Create MuJoCo model with larger hand geometry to ensure contact.
    """
    xml = """
    <mujoco model="ball_hand_v3">
        <option gravity="0 0 -9.81" timestep="0.001" integrator="implicit">
            <flag contact="enable"/>
        </option>

        <default>
            <geom condim="4" friction="1.0 0.005 0.0001" solref="0.01 1"/>
        </default>

        <worldbody>
            <!-- Ground -->
            <geom type="plane" size="10 10 0.1" rgba="0.3 0.5 0.3 1"/>

            <!-- Hand - mocap body (kinematic, we control it directly) -->
            <!-- Made larger to ensure contact -->
            <body name="hand" mocap="true" pos="0 0 1.5">
                <!-- Large cupped surface to hold ball -->
                <geom name="palm" type="cylinder" size="0.12 0.02"
                      rgba="0.9 0.75 0.6 1" euler="0 0 0"/>
            </body>

            <!-- Ball -->
            <body name="ball" pos="0 0 1.66">
                <freejoint name="ball_free"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"
                      rgba="1.0 0.5 0.0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


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


def get_hand_trajectory(timeseries, keypoint_map, start_frame=50, end_frame=200):
    """Get hand trajectory in meters."""
    positions = []
    frames = []

    for frame in range(start_frame, end_frame):
        pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
        if pos is None:
            pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if pos is not None:
            positions.append(pos * FEET_TO_METERS)
            frames.append(frame)

    if len(positions) < 20:
        return None, None

    return np.array(positions), np.array(frames)


def check_contact(model, data):
    """Check if ball contacts hand."""
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            return True
    return False


def get_contact_force(model, data):
    """Get contact force magnitude between ball and hand."""
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            # Get contact force
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            return np.linalg.norm(force[:3])  # Normal force magnitude
    return 0.0


def simulate_shot(timeseries, keypoint_map, model, data, verbose=False):
    """Simulate ball-hand contact throughout shot."""

    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    # Reset
    mujoco.mj_resetData(model, data)

    # Compute velocities
    window = min(11, len(hand_pos) - 2)
    if window % 2 == 0:
        window -= 1
    hand_vel = np.zeros_like(hand_pos)
    for i in range(3):
        hand_vel[:, i] = savgol_filter(hand_pos[:, i], window, 3, deriv=1) * FPS

    # Find when hand starts moving up
    start_idx = 0
    for i in range(len(hand_vel)):
        if hand_vel[i, 2] > 0.5:
            start_idx = max(0, i - 5)
            break

    # Initial hand position
    init_hand = hand_pos[start_idx]

    # Set hand position
    data.mocap_pos[0] = init_hand

    # Ball position: on top of hand
    # Hand is cylinder with height 0.02, ball radius 0.12
    # Ball center should be at hand_z + 0.02 + 0.12 = hand_z + 0.14
    ball_init_z = init_hand[2] + 0.02 + BALL_RADIUS
    data.qpos[0] = init_hand[0]
    data.qpos[1] = init_hand[1]
    data.qpos[2] = ball_init_z
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0

    # Forward to establish contact
    mujoco.mj_forward(model, data)

    # Step a few times to let ball settle
    for _ in range(100):
        mujoco.mj_step(model, data)

    # Check contact established
    mujoco.mj_forward(model, data)
    initial_contact = check_contact(model, data)
    initial_force = get_contact_force(model, data)

    if verbose:
        print(f"  Start frame: {frames[start_idx]}")
        print(f"  Hand init: {init_hand}")
        print(f"  Ball init: {data.qpos[0:3]}")
        print(f"  Initial contact: {initial_contact}, force: {initial_force:.2f} N")

    if not initial_contact:
        # Try to establish contact by dropping ball
        if verbose:
            print("  Dropping ball to establish contact...")

        for _ in range(500):
            mujoco.mj_step(model, data)
            if check_contact(model, data):
                break

        initial_contact = check_contact(model, data)
        if verbose:
            print(f"  Contact after drop: {initial_contact}")

    # Simulation parameters
    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS
    steps_per_frame = int(frame_dt / sim_dt)

    # State tracking
    had_contact = initial_contact
    contact_count = 0 if not initial_contact else 1
    no_contact_count = 0
    release_data = None
    max_contact_force = 0

    # Simulate
    for idx in range(start_idx, len(hand_pos)):
        # Move hand
        data.mocap_pos[0] = hand_pos[idx]

        # Step
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # Check contact
        in_contact = check_contact(model, data)
        force = get_contact_force(model, data)
        max_contact_force = max(max_contact_force, force)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
            had_contact = True
        else:
            no_contact_count += 1

        # Release detection: had significant contact, now lost
        if had_contact and contact_count > 5 and no_contact_count >= 3:
            ball_pos = data.qpos[0:3].copy()
            ball_vel = data.qvel[0:3].copy()

            release_data = {
                'frame': frames[idx],
                'ball_pos': ball_pos,
                'ball_vel': ball_vel,
                'speed': np.linalg.norm(ball_vel),
                'contact_frames': contact_count,
                'max_force': max_contact_force,
            }

            if verbose:
                print(f"  RELEASE at frame {frames[idx]}")
                print(f"  Ball pos: {ball_pos}")
                print(f"  Ball vel: {ball_vel}")
                print(f"  Speed: {np.linalg.norm(ball_vel):.2f} m/s ({np.linalg.norm(ball_vel)/FEET_TO_METERS:.1f} ft/s)")
                print(f"  Contact frames: {contact_count}, Max force: {max_contact_force:.1f} N")

            break

    if release_data is None:
        ball_pos = data.qpos[0:3].copy()
        ball_vel = data.qvel[0:3].copy()

        release_data = {
            'frame': frames[-1] if len(frames) > 0 else 0,
            'ball_pos': ball_pos,
            'ball_vel': ball_vel,
            'speed': np.linalg.norm(ball_vel),
            'contact_frames': contact_count,
            'max_force': max_contact_force,
            'no_release': True,
        }

        if verbose:
            print(f"  No release detected")
            print(f"  Contact frames: {contact_count}, Max force: {max_contact_force:.1f} N")
            print(f"  Final vel: {ball_vel}, speed: {np.linalg.norm(ball_vel):.2f} m/s")

    return release_data


def main():
    print("=" * 80)
    print("MUJOCO BALL-HAND CONTACT SIMULATION v3")
    print("=" * 80)

    model, data = create_model()
    print(f"\nModel: timestep={model.opt.timestep}s, nq={model.nq}, nmocap={model.nmocap}")

    # Test contact detection
    print("\nTesting contact detection...")
    mujoco.mj_resetData(model, data)
    data.mocap_pos[0] = [0, 0, 1.5]  # Hand at z=1.5
    data.qpos[0:3] = [0, 0, 1.5 + 0.02 + 0.12]  # Ball on top
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # Let settle
    for _ in range(100):
        mujoco.mj_step(model, data)

    print(f"  Contact: {check_contact(model, data)}")
    print(f"  Force: {get_contact_force(model, data):.2f} N")
    print(f"  Ball pos: {data.qpos[0:3]}")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    print("\nProcessing shots...")
    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 10:
            break

        print(f"\nShot {i+1}: {metadata['id'][:8]} (Player {metadata['participant_id']})")

        result = simulate_shot(timeseries, keypoint_map, model, data, verbose=True)

        if result is not None:
            result['shot_id'] = metadata['id']
            results.append(result)

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        speeds = [r['speed'] for r in results]
        speeds_fps = [s / FEET_TO_METERS for s in speeds]

        print(f"\nRelease speeds: {np.mean(speeds_fps):.1f} +/- {np.std(speeds_fps):.1f} ft/s")
        print(f"Range: {np.min(speeds_fps):.1f} to {np.max(speeds_fps):.1f} ft/s")
        print(f"Required: ~24 ft/s")

        # Contact stats
        contact_frames = [r['contact_frames'] for r in results]
        max_forces = [r['max_force'] for r in results]
        print(f"\nContact frames: {np.mean(contact_frames):.1f} +/- {np.std(contact_frames):.1f}")
        print(f"Max contact force: {np.mean(max_forces):.1f} +/- {np.std(max_forces):.1f} N")


if __name__ == "__main__":
    main()
