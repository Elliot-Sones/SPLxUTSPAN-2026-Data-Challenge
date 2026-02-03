"""
MuJoCo simulation with higher actuator gains for better velocity tracking.
Test on shots with high hand velocities.
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


def create_model(kp=50000, kv=5000):
    """Create model with configurable actuator gains."""
    xml = f"""
    <mujoco model="ball_hand_highgain">
        <option gravity="0 0 -9.81" timestep="0.0002"/>

        <worldbody>
            <geom type="plane" size="20 20 0.1"/>

            <body name="hand" pos="0 0 1.5">
                <joint name="hx" type="slide" axis="1 0 0" damping="100"/>
                <joint name="hy" type="slide" axis="0 1 0" damping="100"/>
                <joint name="hz" type="slide" axis="0 0 1" damping="100"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
            </body>

            <body name="ball" pos="0 0 1.66">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"/>
            </body>
        </worldbody>

        <actuator>
            <position name="hx_ctrl" joint="hx" kp="{kp}" kv="{kv}"/>
            <position name="hy_ctrl" joint="hy" kp="{kp}" kv="{kv}"/>
            <position name="hz_ctrl" joint="hz" kp="{kp}" kv="{kv}"/>
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
    """Simulate ball-hand contact."""

    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    # Compute target velocities
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

    # Calculate target max speed
    target_max_speed = np.max(np.linalg.norm(hand_vel_target, axis=1))

    # Reset
    mujoco.mj_resetData(model, data)

    init_hand = hand_pos[start_idx]
    hand_offset = init_hand - np.array([0, 0, 1.5])

    # Set initial state
    data.qpos[0:3] = hand_offset
    palm_top_z = 1.5 + hand_offset[2] + 0.02
    ball_z = palm_top_z + BALL_RADIUS - 0.01
    data.qpos[3:6] = [init_hand[0], init_hand[1], ball_z]
    data.qpos[6:10] = [1, 0, 0, 0]
    data.qvel[:] = 0
    data.ctrl[:] = hand_offset

    mujoco.mj_forward(model, data)

    # Let settle
    for _ in range(50):
        mujoco.mj_step(model, data)

    initial_contact = check_contact(model, data)

    if verbose:
        print(f"  Target max speed: {target_max_speed:.2f} m/s ({target_max_speed/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Start frame: {frames[start_idx]}, Contact: {initial_contact}")

    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS
    steps_per_frame = int(frame_dt / sim_dt)

    had_contact = initial_contact
    contact_count = 1 if initial_contact else 0
    no_contact_count = 0
    release_data = None
    max_ball_speed = 0
    max_hand_speed = 0

    for idx in range(start_idx, len(hand_pos)):
        target = hand_pos[idx]
        target_offset = target - np.array([0, 0, 1.5])

        data.ctrl[:] = target_offset

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)

        hand_vel = data.qvel[0:3]
        ball_vel = data.qvel[3:6].copy()

        hand_speed = np.linalg.norm(hand_vel)
        ball_speed = np.linalg.norm(ball_vel)

        max_hand_speed = max(max_hand_speed, hand_speed)
        max_ball_speed = max(max_ball_speed, ball_speed)

        in_contact = check_contact(model, data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
            had_contact = True
        else:
            no_contact_count += 1

        if had_contact and contact_count > 5 and no_contact_count >= 3:
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

    return release_data


def main():
    print("=" * 80)
    print("MUJOCO HIGH-GAIN SIMULATION")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Test different gain settings
    gain_configs = [
        (20000, 2000, "Low (default)"),
        (50000, 5000, "Medium"),
        (100000, 10000, "High"),
    ]

    # Find shots with high hand velocity
    print("\nFinding high-velocity shots...")
    high_vel_shots = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 50:
            break

        hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
        if hand_pos is None:
            continue

        window = min(11, len(hand_pos) - 2)
        if window % 2 == 0:
            window -= 1
        hand_vel = np.zeros_like(hand_pos)
        for j in range(3):
            hand_vel[:, j] = savgol_filter(hand_pos[:, j], window, 3, deriv=1) * FPS

        max_speed = np.max(np.linalg.norm(hand_vel, axis=1))

        if max_speed > 5.0:  # > 16 ft/s
            high_vel_shots.append({
                'idx': i,
                'metadata': metadata,
                'timeseries': timeseries,
                'max_speed': max_speed,
            })

    print(f"Found {len(high_vel_shots)} shots with hand speed > 5 m/s (16 ft/s)")

    if not high_vel_shots:
        print("No high-velocity shots found!")
        return

    # Sort by velocity
    high_vel_shots.sort(key=lambda x: -x['max_speed'])

    # Test on top 3 fastest shots
    print("\n" + "=" * 80)
    print("TESTING GAIN CONFIGURATIONS")
    print("=" * 80)

    for shot_info in high_vel_shots[:3]:
        print(f"\n{'='*60}")
        print(f"Shot: {shot_info['metadata']['id'][:10]}")
        print(f"Target hand speed: {shot_info['max_speed']:.2f} m/s ({shot_info['max_speed']/FEET_TO_METERS:.1f} ft/s)")
        print(f"{'='*60}")

        for kp, kv, name in gain_configs:
            print(f"\n  {name} gains (kp={kp}, kv={kv}):")

            try:
                model = create_model(kp=kp, kv=kv)
                data = mujoco.MjData(model)

                result = simulate_shot(
                    shot_info['timeseries'],
                    keypoint_map,
                    model, data,
                    verbose=True
                )
            except Exception as e:
                print(f"    ERROR: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS")
    print("=" * 80)

    print("\nUsing medium gains (kp=50000) on all high-velocity shots:")

    model = create_model(kp=50000, kv=5000)
    data = mujoco.MjData(model)

    results = []
    for shot_info in high_vel_shots:
        result = simulate_shot(
            shot_info['timeseries'],
            keypoint_map,
            model, data,
            verbose=False
        )
        if result:
            results.append(result)

    if results:
        ball_speeds = [r['max_speed'] / FEET_TO_METERS for r in results]
        tracking = [r['max_hand_speed'] / r['target_max_speed'] * 100 for r in results]

        print(f"\n  Ball speeds:")
        print(f"    Mean: {np.mean(ball_speeds):.1f} ft/s")
        print(f"    Max:  {np.max(ball_speeds):.1f} ft/s")
        print(f"    Min:  {np.min(ball_speeds):.1f} ft/s")

        print(f"\n  Hand tracking efficiency:")
        print(f"    Mean: {np.mean(tracking):.1f}%")
        print(f"    Max:  {np.max(tracking):.1f}%")

        print(f"\n  Required: ~22 ft/s")


if __name__ == "__main__":
    main()
