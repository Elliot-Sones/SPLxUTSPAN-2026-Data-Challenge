"""
MuJoCo Ball-Hand Contact Simulation v2

Fixed version with proper initialization and joint indexing.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

# Physical constants (SI units)
BALL_RADIUS = 0.12  # meters
BALL_MASS = 0.625   # kg
FEET_TO_METERS = 0.3048
FPS = 60


def create_model():
    """
    Create a simple MuJoCo model:
    - Ball: free body with sphere geom
    - Hand: kinematic platform we move directly
    """
    xml = """
    <mujoco model="ball_hand">
        <option gravity="0 0 -9.81" timestep="0.002">
            <flag contact="enable"/>
        </option>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>
        </asset>

        <default>
            <geom condim="4" friction="1.0 0.005 0.0001"/>
        </default>

        <worldbody>
            <!-- Ground -->
            <geom type="plane" size="10 10 0.1" rgba="0.3 0.5 0.3 1"/>

            <!-- Hand platform - mocap body that we control directly -->
            <body name="hand" mocap="true" pos="0 0 1.5">
                <geom name="palm" type="box" size="0.07 0.05 0.015"
                      rgba="0.9 0.75 0.6 1" friction="1.0 0.005 0.0001"/>
                <geom name="finger_rim" type="capsule" size="0.012" fromto="-0.05 0.05 0.01 0.05 0.05 0.01"
                      rgba="0.9 0.75 0.6 1"/>
            </body>

            <!-- Ball - free body -->
            <body name="ball" pos="0 0 1.65">
                <freejoint name="ball_free"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"
                      rgba="1.0 0.5 0.0 1" friction="1.0 0.005 0.0001"/>
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


def get_hand_trajectory(timeseries, keypoint_map):
    """
    Get hand trajectory from fingertip positions.
    Returns positions in METERS.
    """
    positions = []
    frames = []

    for frame in range(50, 200):
        # Use middle finger as reference
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
    """Check if ball is in contact with hand."""
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)

        hand_geoms = ['palm', 'finger_rim']
        if (g1 == 'ball_geom' and g2 in hand_geoms) or \
           (g2 == 'ball_geom' and g1 in hand_geoms):
            return True
    return False


def simulate_shot(timeseries, keypoint_map, model, data, verbose=False):
    """
    Simulate ball-hand contact throughout the shot.
    """
    # Get hand trajectory
    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Compute hand velocity
    window = min(11, len(hand_pos) - 2)
    if window % 2 == 0:
        window -= 1
    hand_vel = np.zeros_like(hand_pos)
    for i in range(3):
        hand_vel[:, i] = savgol_filter(hand_pos[:, i], window, 3, deriv=1) * FPS

    # Find shot start (when hand starts moving up significantly)
    start_idx = 0
    for i in range(len(hand_vel)):
        if hand_vel[i, 2] > 1.0:  # 1 m/s upward threshold
            start_idx = max(0, i - 3)
            break

    # Initial positions
    init_hand = hand_pos[start_idx]

    # Set hand (mocap body) position
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    data.mocap_pos[0] = init_hand

    # Set ball on top of hand
    data.qpos[0:3] = init_hand + np.array([0, 0, BALL_RADIUS + 0.02])  # ball pos
    data.qpos[3:7] = [1, 0, 0, 0]  # ball quaternion
    data.qvel[:] = 0

    # Forward kinematics
    mujoco.mj_forward(model, data)

    if verbose:
        print(f"  Start frame: {frames[start_idx]}")
        print(f"  Hand init: {init_hand}")
        print(f"  Ball init: {data.qpos[0:3]}")
        print(f"  Contact: {check_contact(model, data)}")

    # Simulation parameters
    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS
    steps_per_frame = int(frame_dt / sim_dt)

    # Track state
    in_contact_prev = True
    contact_count = 0
    no_contact_count = 0
    release_detected = False
    release_data = None

    # Simulate frame by frame
    for idx in range(start_idx, len(hand_pos)):
        # Move hand to new position
        target_pos = hand_pos[idx]
        data.mocap_pos[0] = target_pos

        # Step simulation
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # Check contact
        in_contact = check_contact(model, data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
        else:
            no_contact_count += 1

        # Detect release: had contact, now lost it
        if contact_count > 3 and no_contact_count >= 2 and not release_detected:
            release_detected = True

            # Get ball state
            ball_pos = data.qpos[0:3].copy()
            ball_vel = data.qvel[0:3].copy()

            release_data = {
                'frame': frames[idx],
                'ball_pos': ball_pos,
                'ball_vel': ball_vel,
                'speed': np.linalg.norm(ball_vel),
                'hand_pos': target_pos,
                'contact_frames': contact_count,
            }

            if verbose:
                print(f"  Release at frame {frames[idx]}")
                print(f"  Ball pos: {ball_pos}")
                print(f"  Ball vel: {ball_vel}")
                print(f"  Speed: {np.linalg.norm(ball_vel):.2f} m/s ({np.linalg.norm(ball_vel)/FEET_TO_METERS:.1f} ft/s)")

            break

        in_contact_prev = in_contact

    if not release_detected:
        # Use final state
        ball_pos = data.qpos[0:3].copy()
        ball_vel = data.qvel[0:3].copy()

        release_data = {
            'frame': frames[-1],
            'ball_pos': ball_pos,
            'ball_vel': ball_vel,
            'speed': np.linalg.norm(ball_vel),
            'hand_pos': hand_pos[-1],
            'contact_frames': contact_count,
            'no_release': True,
        }

        if verbose:
            print(f"  No release detected")
            print(f"  Final ball vel: {ball_vel}")
            print(f"  Speed: {np.linalg.norm(ball_vel):.2f} m/s")
            print(f"  Contact frames: {contact_count}")

    return release_data


def main():
    print("=" * 80)
    print("MUJOCO BALL-HAND CONTACT SIMULATION v2")
    print("=" * 80)

    # Create model
    print("\nCreating MuJoCo model...")
    model, data = create_model()

    # Print model info
    print(f"  Timestep: {model.opt.timestep} s")
    print(f"  nq (positions): {model.nq}")
    print(f"  nv (velocities): {model.nv}")
    print(f"  nmocap: {model.nmocap}")

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
            result['true_angle'] = metadata['angle']
            result['true_depth'] = metadata['depth']
            result['true_lr'] = metadata['left_right']
            results.append(result)

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        speeds_ms = [r['speed'] for r in results]
        speeds_fps = [s / FEET_TO_METERS for s in speeds_ms]

        print(f"\nRelease speeds (m/s): {np.mean(speeds_ms):.2f} +/- {np.std(speeds_ms):.2f}")
        print(f"Release speeds (ft/s): {np.mean(speeds_fps):.1f} +/- {np.std(speeds_fps):.1f}")
        print(f"Min: {np.min(speeds_fps):.1f} ft/s, Max: {np.max(speeds_fps):.1f} ft/s")
        print(f"\nRequired: ~24 ft/s (7.3 m/s)")

        # Show velocity components
        vels = np.array([r['ball_vel'] for r in results])
        vels_fps = vels / FEET_TO_METERS

        print(f"\nVelocity components (ft/s):")
        print(f"  vx: {np.mean(vels_fps[:,0]):.1f} +/- {np.std(vels_fps[:,0]):.1f}")
        print(f"  vy: {np.mean(vels_fps[:,1]):.1f} +/- {np.std(vels_fps[:,1]):.1f}")
        print(f"  vz: {np.mean(vels_fps[:,2]):.1f} +/- {np.std(vels_fps[:,2]):.1f}")


if __name__ == "__main__":
    main()
