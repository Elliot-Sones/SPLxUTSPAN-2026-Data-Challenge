"""
MuJoCo ball-hand contact simulation with CORRECT qpos indexing.

qpos layout (hand first in XML, then ball):
  qpos[0:3] = hand joints [hx, hy, hz] (offsets from body default [0,0,1.5])
  qpos[3:6] = ball position [x, y, z] (world coordinates)
  qpos[6:10] = ball quaternion [w, x, y, z]

qvel layout:
  qvel[0:3] = hand joint velocities
  qvel[3:6] = ball linear velocity
  qvel[6:9] = ball angular velocity
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
    Simple model: ball + moving platform (hand).
    IMPORTANT: hand body is defined BEFORE ball body in XML,
    so hand joints come first in qpos.
    """
    xml = """
    <mujoco model="ball_hand_fixed">
        <option gravity="0 0 -9.81" timestep="0.0005"/>

        <worldbody>
            <geom type="plane" size="20 20 0.1"/>

            <!-- Hand/platform that moves (FIRST body -> qpos[0:3]) -->
            <body name="hand" pos="0 0 1.5">
                <joint name="hx" type="slide" axis="1 0 0" damping="200"/>
                <joint name="hy" type="slide" axis="0 1 0" damping="200"/>
                <joint name="hz" type="slide" axis="0 0 1" damping="200"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
            </body>

            <!-- Ball (SECOND body -> qpos[3:10]) -->
            <body name="ball" pos="0 0 1.66">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"/>
            </body>
        </worldbody>

        <actuator>
            <position name="hx_ctrl" joint="hx" kp="20000" kv="2000"/>
            <position name="hy_ctrl" joint="hy" kp="20000" kv="2000"/>
            <position name="hz_ctrl" joint="hz" kp="20000" kv="2000"/>
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
    """Simulate ball-hand contact with CORRECT qpos indexing."""

    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    # Reset
    mujoco.mj_resetData(model, data)

    # Compute hand velocity for finding shot start
    window = min(11, len(hand_pos) - 2)
    if window % 2 == 0:
        window -= 1
    hand_vel = np.zeros_like(hand_pos)
    for i in range(3):
        hand_vel[:, i] = savgol_filter(hand_pos[:, i], window, 3, deriv=1) * FPS

    # Find shot start (when hand starts moving up)
    start_idx = 0
    for i in range(len(hand_vel)):
        if hand_vel[i, 2] > 0.5:
            start_idx = max(0, i - 3)
            break

    # Initial hand position in world coordinates
    init_hand = hand_pos[start_idx]

    # Hand joints are offsets from body default position [0, 0, 1.5]
    hand_offset = init_hand - np.array([0, 0, 1.5])

    # CORRECT qpos layout:
    # qpos[0:3] = hand joints [hx, hy, hz]
    # qpos[3:6] = ball position [x, y, z]
    # qpos[6:10] = ball quaternion [w, x, y, z]

    # Set hand joint positions
    data.qpos[0] = hand_offset[0]  # hx
    data.qpos[1] = hand_offset[1]  # hy
    data.qpos[2] = hand_offset[2]  # hz

    # Ball on top of hand (with slight penetration to ensure contact)
    # Hand world z = 1.5 + hand_offset[2]
    # Palm top z = hand world z + 0.02 (half height of cylinder)
    # Ball center for contact = palm top z + ball radius
    # Subtract 0.01 for penetration to ensure contact
    palm_top_z = 1.5 + hand_offset[2] + 0.02
    ball_z = palm_top_z + BALL_RADIUS - 0.01

    # Set ball position (world coordinates)
    data.qpos[3] = init_hand[0]  # ball x
    data.qpos[4] = init_hand[1]  # ball y
    data.qpos[5] = ball_z        # ball z

    # Set ball quaternion (identity)
    data.qpos[6:10] = [1, 0, 0, 0]

    # Zero all velocities
    data.qvel[:] = 0

    # Set actuator targets to hold hand in position
    data.ctrl[0] = hand_offset[0]
    data.ctrl[1] = hand_offset[1]
    data.ctrl[2] = hand_offset[2]

    mujoco.mj_forward(model, data)

    # Let settle briefly
    for _ in range(50):
        mujoco.mj_step(model, data)

    initial_contact = check_contact(model, data)

    if verbose:
        print(f"  Start frame: {frames[start_idx]}")
        print(f"  Hand init (world): {init_hand}")
        print(f"  Hand offset: {hand_offset}")
        print(f"  Ball init: {data.qpos[3:6]}")
        print(f"  Contact: {initial_contact}")

    # Simulation loop
    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS
    steps_per_frame = int(frame_dt / sim_dt)

    had_contact = initial_contact
    contact_count = 1 if initial_contact else 0
    no_contact_count = 0
    release_data = None
    max_ball_speed = 0

    for idx in range(start_idx, len(hand_pos)):
        # Target hand position (as offset from [0,0,1.5])
        target = hand_pos[idx]
        target_offset = target - np.array([0, 0, 1.5])

        data.ctrl[0] = target_offset[0]
        data.ctrl[1] = target_offset[1]
        data.ctrl[2] = target_offset[2]

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        # CORRECT: ball velocity is qvel[3:6], not qvel[0:3]
        ball_vel = data.qvel[3:6].copy()
        ball_speed = np.linalg.norm(ball_vel)
        max_ball_speed = max(max_ball_speed, ball_speed)

        in_contact = check_contact(model, data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
            had_contact = True
        else:
            no_contact_count += 1

        # Release: had contact, now lost
        if had_contact and contact_count > 5 and no_contact_count >= 3:
            # CORRECT: ball position is qpos[3:6]
            release_data = {
                'frame': frames[idx],
                'ball_pos': data.qpos[3:6].copy(),
                'ball_vel': ball_vel,
                'speed': ball_speed,
                'max_speed': max_ball_speed,
                'contact_frames': contact_count,
            }

            if verbose:
                print(f"  RELEASE at frame {frames[idx]}")
                print(f"  Ball vel: {ball_vel}")
                print(f"  Speed: {ball_speed:.2f} m/s ({ball_speed/FEET_TO_METERS:.1f} ft/s)")
                print(f"  Max speed: {max_ball_speed:.2f} m/s ({max_ball_speed/FEET_TO_METERS:.1f} ft/s)")

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
            'contact_frames': contact_count,
            'no_release': True,
        }

        if verbose:
            print(f"  No release - final speed: {ball_speed:.2f} m/s ({ball_speed/FEET_TO_METERS:.1f} ft/s)")
            print(f"  Max speed: {max_ball_speed:.2f} m/s ({max_ball_speed/FEET_TO_METERS:.1f} ft/s)")
            print(f"  Contact frames: {contact_count}")

    return release_data


def main():
    print("=" * 80)
    print("MUJOCO BALL-HAND CONTACT (Fixed Indexing)")
    print("=" * 80)

    model = create_model()
    data = mujoco.MjData(model)

    print(f"\nModel: dt={model.opt.timestep}s, nq={model.nq}, nv={model.nv}, nu={model.nu}")

    # Verify qpos layout
    print("\nVerifying qpos layout...")
    mujoco.mj_resetData(model, data)
    print(f"  After reset: qpos = {data.qpos}")
    print(f"  Expected: hand[0,0,0] + ball_pos[0,0,1.66] + quat[1,0,0,0]")

    # Quick contact test
    print("\nQuick contact test...")
    mujoco.mj_resetData(model, data)

    # Hand at default position [0, 0, 1.5]
    # Palm top at z = 1.5 + 0.02 = 1.52
    # Ball center for contact = 1.52 + 0.12 = 1.64
    # With 1cm penetration = 1.63

    data.qpos[0:3] = [0, 0, 0]     # Hand joints at default
    data.qpos[3:6] = [0, 0, 1.63]  # Ball with 1cm penetration
    data.qpos[6:10] = [1, 0, 0, 0] # Ball quaternion

    mujoco.mj_forward(model, data)

    # Get geom positions to verify
    palm_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

    print(f"  Hand joints: {data.qpos[0:3]}")
    print(f"  Ball qpos: {data.qpos[3:6]}")
    print(f"  Palm geom pos: {data.geom_xpos[palm_geom_id]}")
    print(f"  Ball geom pos: {data.geom_xpos[ball_geom_id]}")
    print(f"  Initial contact: {check_contact(model, data)}, ncon={data.ncon}")

    for _ in range(100):
        mujoco.mj_step(model, data)

    print(f"  After settle:")
    print(f"    Ball pos: {data.qpos[3:6]}")
    print(f"    Ball vel: {data.qvel[3:6]}")
    print(f"    Contact: {check_contact(model, data)}")

    # Test pushing ball upward
    print("\nTest pushing ball upward...")
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0]
    data.qpos[3:6] = [0, 0, 1.63]
    data.qpos[6:10] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # Push hand up
    print("  Moving hand up at 5 m/s for 0.1s...")
    target_z = 0
    for step in range(200):  # 0.1s at 0.0005s timestep
        target_z += 5.0 * 0.0005  # 5 m/s
        data.ctrl[0] = 0
        data.ctrl[1] = 0
        data.ctrl[2] = target_z
        mujoco.mj_step(model, data)

        if step % 40 == 0:
            ball_vel = data.qvel[3:6]
            ball_speed = np.linalg.norm(ball_vel)
            contact = check_contact(model, data)
            print(f"    Step {step}: ball_z={data.qpos[5]:.3f}, ball_vz={ball_vel[2]:.2f}, "
                  f"speed={ball_speed:.2f} m/s, contact={contact}")

    ball_vel = data.qvel[3:6]
    print(f"\n  Final ball velocity: {ball_vel}")
    print(f"  Final ball speed: {np.linalg.norm(ball_vel):.2f} m/s ({np.linalg.norm(ball_vel)/FEET_TO_METERS:.1f} ft/s)")

    # Process real shots
    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    print("\n" + "=" * 80)
    print("Processing real shots...")
    print("=" * 80)
    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 10:
            break

        print(f"\nShot {i+1}: {metadata['id'][:8]}")
        result = simulate_shot(timeseries, keypoint_map, model, data, verbose=True)

        if result:
            results.append(result)

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        release_speeds = [r['speed'] / FEET_TO_METERS for r in results]
        max_speeds = [r['max_speed'] / FEET_TO_METERS for r in results]

        print(f"\nRelease speeds: {np.mean(release_speeds):.1f} +/- {np.std(release_speeds):.1f} ft/s")
        print(f"Max speeds: {np.mean(max_speeds):.1f} +/- {np.std(max_speeds):.1f} ft/s")
        print(f"Best: {np.max(max_speeds):.1f} ft/s")
        print(f"\nRequired: ~24 ft/s")


if __name__ == "__main__":
    main()
