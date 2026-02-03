"""
MuJoCo Ball-Hand Contact Simulation v4

Use position-controlled actuated joints instead of mocap.
This ensures the hand actually PUSHES the ball through physics.
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
    """
    Model with position-controlled hand that pushes through physics.
    High-gain position actuators ensure hand tracks target closely
    while still applying physical forces to ball.
    """
    xml = """
    <mujoco model="ball_hand_v4">
        <option gravity="0 0 -9.81" timestep="0.0005" integrator="implicit">
            <flag contact="enable"/>
        </option>

        <default>
            <geom condim="4" friction="1.0 0.005 0.0001" solref="0.002 1"/>
            <joint damping="50" armature="0.1"/>
        </default>

        <worldbody>
            <geom type="plane" size="10 10 0.1"/>

            <!-- Hand with high-gain position control -->
            <body name="hand_base" pos="0 0 0">
                <joint name="hand_x" type="slide" axis="1 0 0" range="-20 20"/>
                <joint name="hand_y" type="slide" axis="0 1 0" range="-20 20"/>
                <joint name="hand_z" type="slide" axis="0 0 1" range="0 10"/>

                <!-- Hand surface - cup shape -->
                <geom name="palm" type="cylinder" size="0.10 0.015"
                      pos="0 0 0" rgba="0.9 0.7 0.5 1" mass="2.0"/>
            </body>

            <!-- Ball -->
            <body name="ball" pos="0 0 0.5">
                <freejoint name="ball_free"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"
                      rgba="1.0 0.5 0.0 1"/>
            </body>
        </worldbody>

        <actuator>
            <!-- Very high gain position control -->
            <position name="pos_x" joint="hand_x" kp="10000" kv="1000"/>
            <position name="pos_y" joint="hand_y" kp="10000" kv="1000"/>
            <position name="pos_z" joint="hand_z" kp="10000" kv="1000"/>
        </actuator>
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
    positions = []
    frames = []
    for frame in range(50, 200):
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
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            return True
    return False


def simulate_shot(timeseries, keypoint_map, model, data, verbose=False):
    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    mujoco.mj_resetData(model, data)

    # Compute velocities for smooth control
    window = 9
    hand_vel = np.zeros_like(hand_pos)
    for i in range(3):
        hand_vel[:, i] = savgol_filter(hand_pos[:, i], window, 3, deriv=1) * FPS

    # Find shot start
    start_idx = 0
    for i in range(len(hand_vel)):
        if hand_vel[i, 2] > 0.3:
            start_idx = max(0, i - 3)
            break

    init_hand = hand_pos[start_idx]

    # Set hand position (joint positions)
    data.qpos[7] = init_hand[0]  # hand_x
    data.qpos[8] = init_hand[1]  # hand_y
    data.qpos[9] = init_hand[2]  # hand_z

    # Set actuator targets to same position
    data.ctrl[0] = init_hand[0]
    data.ctrl[1] = init_hand[1]
    data.ctrl[2] = init_hand[2]

    # Ball on top of hand
    ball_z = init_hand[2] + 0.015 + BALL_RADIUS + 0.005
    data.qpos[0] = init_hand[0]
    data.qpos[1] = init_hand[1]
    data.qpos[2] = ball_z
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:] = 0

    mujoco.mj_forward(model, data)

    # Let settle
    for _ in range(200):
        mujoco.mj_step(model, data)

    initial_contact = check_contact(model, data)

    if verbose:
        print(f"  Start frame: {frames[start_idx]}")
        print(f"  Hand init: {init_hand}")
        print(f"  Ball init: {data.qpos[0:3]}")
        print(f"  Contact: {initial_contact}")

    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS
    steps_per_frame = int(frame_dt / sim_dt)

    had_contact = initial_contact
    contact_count = 0 if not initial_contact else 1
    no_contact_count = 0
    release_data = None
    max_ball_speed = 0

    for idx in range(start_idx, len(hand_pos)):
        # Set position targets for actuators
        target = hand_pos[idx]
        data.ctrl[0] = target[0]
        data.ctrl[1] = target[1]
        data.ctrl[2] = target[2]

        # Step simulation
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # Track ball velocity
        ball_vel = data.qvel[0:3].copy()
        ball_speed = np.linalg.norm(ball_vel)
        max_ball_speed = max(max_ball_speed, ball_speed)

        in_contact = check_contact(model, data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
            had_contact = True
        else:
            no_contact_count += 1

        # Release detection
        if had_contact and contact_count > 3 and no_contact_count >= 3:
            ball_pos = data.qpos[0:3].copy()

            release_data = {
                'frame': frames[idx],
                'ball_pos': ball_pos,
                'ball_vel': ball_vel,
                'speed': ball_speed,
                'max_speed': max_ball_speed,
                'contact_frames': contact_count,
            }

            if verbose:
                print(f"  RELEASE at frame {frames[idx]}")
                print(f"  Ball vel: {ball_vel}")
                print(f"  Speed: {ball_speed:.2f} m/s ({ball_speed/FEET_TO_METERS:.1f} ft/s)")
                print(f"  Max speed seen: {max_ball_speed:.2f} m/s ({max_ball_speed/FEET_TO_METERS:.1f} ft/s)")

            break

    if release_data is None:
        ball_pos = data.qpos[0:3].copy()
        ball_vel = data.qvel[0:3].copy()
        ball_speed = np.linalg.norm(ball_vel)

        release_data = {
            'frame': frames[-1],
            'ball_pos': ball_pos,
            'ball_vel': ball_vel,
            'speed': ball_speed,
            'max_speed': max_ball_speed,
            'contact_frames': contact_count,
            'no_release': True,
        }

        if verbose:
            print(f"  No release - final speed: {ball_speed:.2f} m/s")
            print(f"  Max speed seen: {max_ball_speed:.2f} m/s ({max_ball_speed/FEET_TO_METERS:.1f} ft/s)")

    return release_data


def main():
    print("=" * 80)
    print("MUJOCO BALL-HAND CONTACT v4 (Position-Controlled)")
    print("=" * 80)

    model, data = create_model()
    print(f"\nModel: dt={model.opt.timestep}s, nq={model.nq}, nu={model.nu}")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 10:
            break

        print(f"\nShot {i+1}: {metadata['id'][:8]}")
        result = simulate_shot(timeseries, keypoint_map, model, data, verbose=True)

        if result is not None:
            results.append(result)

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        release_speeds = [r['speed'] / FEET_TO_METERS for r in results]
        max_speeds = [r['max_speed'] / FEET_TO_METERS for r in results]

        print(f"\nRelease speeds: {np.mean(release_speeds):.1f} +/- {np.std(release_speeds):.1f} ft/s")
        print(f"Max speeds seen: {np.mean(max_speeds):.1f} +/- {np.std(max_speeds):.1f} ft/s")
        print(f"Best max speed: {np.max(max_speeds):.1f} ft/s")
        print(f"\nRequired: ~24 ft/s")


if __name__ == "__main__":
    main()
