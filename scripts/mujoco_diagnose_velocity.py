"""
Diagnose why ball velocity is too low in MuJoCo simulation.

Questions:
1. Is the hand actuator tracking the target trajectory?
2. What is the actual hand velocity during simulation?
3. How much of hand velocity transfers to ball?
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
    xml = """
    <mujoco model="ball_hand_diag">
        <option gravity="0 0 -9.81" timestep="0.0005"/>

        <worldbody>
            <geom type="plane" size="20 20 0.1"/>

            <body name="hand" pos="0 0 1.5">
                <joint name="hx" type="slide" axis="1 0 0" damping="200"/>
                <joint name="hy" type="slide" axis="0 1 0" damping="200"/>
                <joint name="hz" type="slide" axis="0 0 1" damping="200"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
            </body>

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


def main():
    print("=" * 80)
    print("MUJOCO VELOCITY DIAGNOSTIC")
    print("=" * 80)

    model = create_model()
    data = mujoco.MjData(model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Get first shot
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 1:
            break

        print(f"\nAnalyzing Shot: {metadata['id'][:8]}")

        hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
        if hand_pos is None:
            print("  No hand trajectory")
            continue

        # Compute target velocities from data
        window = min(11, len(hand_pos) - 2)
        if window % 2 == 0:
            window -= 1
        hand_vel_target = np.zeros_like(hand_pos)
        for j in range(3):
            hand_vel_target[:, j] = savgol_filter(hand_pos[:, j], window, 3, deriv=1) * FPS

        # Find shot start
        start_idx = 0
        for j in range(len(hand_vel_target)):
            if hand_vel_target[j, 2] > 0.5:
                start_idx = max(0, j - 3)
                break

        print(f"\n1. TARGET HAND VELOCITY FROM DATA:")
        print(f"   Start frame: {frames[start_idx]}")
        print(f"   Frames {frames[start_idx]} to {frames[-1]}")

        # Print target velocities
        peak_vz = 0
        peak_speed = 0
        for j in range(start_idx, len(hand_vel_target)):
            vz = hand_vel_target[j, 2]
            speed = np.linalg.norm(hand_vel_target[j])
            peak_vz = max(peak_vz, vz)
            peak_speed = max(peak_speed, speed)
            if j < start_idx + 10 or j > len(hand_vel_target) - 5:
                print(f"   Frame {frames[j]}: vz={vz:.2f} m/s, speed={speed:.2f} m/s")

        print(f"   Peak vz: {peak_vz:.2f} m/s ({peak_vz/FEET_TO_METERS:.1f} ft/s)")
        print(f"   Peak speed: {peak_speed:.2f} m/s ({peak_speed/FEET_TO_METERS:.1f} ft/s)")

        # Now simulate and compare
        print(f"\n2. ACTUAL SIMULATION:")

        mujoco.mj_resetData(model, data)

        init_hand = hand_pos[start_idx]
        hand_offset = init_hand - np.array([0, 0, 1.5])

        # Set up initial state
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

        print(f"   Initial contact: {check_contact(model, data)}")

        sim_dt = model.opt.timestep
        frame_dt = 1.0 / FPS
        steps_per_frame = int(frame_dt / sim_dt)

        print(f"\n   Frame-by-frame comparison:")
        print(f"   {'Frame':<8} {'Target vz':<12} {'Actual vz':<12} {'Ball vz':<12} {'Contact':<8}")
        print(f"   {'-'*60}")

        max_hand_vel = 0
        max_ball_vel = 0

        for idx in range(start_idx, min(start_idx + 40, len(hand_pos))):
            target = hand_pos[idx]
            target_offset = target - np.array([0, 0, 1.5])
            target_vz = hand_vel_target[idx, 2]

            data.ctrl[:] = target_offset

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # Actual hand velocity
            hand_vel_actual = data.qvel[0:3]
            hand_vz_actual = hand_vel_actual[2]

            # Ball velocity
            ball_vel = data.qvel[3:6]
            ball_vz = ball_vel[2]

            contact = check_contact(model, data)

            max_hand_vel = max(max_hand_vel, np.linalg.norm(hand_vel_actual))
            max_ball_vel = max(max_ball_vel, np.linalg.norm(ball_vel))

            if idx < start_idx + 15 or idx > min(start_idx + 35, len(hand_pos) - 1):
                print(f"   {frames[idx]:<8} {target_vz:<12.2f} {hand_vz_actual:<12.2f} {ball_vz:<12.2f} {str(contact):<8}")

        print(f"\n   PEAK VELOCITIES:")
        print(f"   Target hand speed: {peak_speed:.2f} m/s ({peak_speed/FEET_TO_METERS:.1f} ft/s)")
        print(f"   Actual hand speed: {max_hand_vel:.2f} m/s ({max_hand_vel/FEET_TO_METERS:.1f} ft/s)")
        print(f"   Ball speed:        {max_ball_vel:.2f} m/s ({max_ball_vel/FEET_TO_METERS:.1f} ft/s)")

        ratio = max_hand_vel / peak_speed if peak_speed > 0 else 0
        print(f"\n   Hand tracking efficiency: {ratio*100:.1f}%")
        print(f"   Velocity transfer (ball/hand): {max_ball_vel/max_hand_vel*100:.1f}%")

        # Compute what gains would be needed
        print(f"\n3. ANALYSIS:")
        print(f"   The hand actuator is NOT tracking the target velocity!")
        print(f"   Target vz peaks at {peak_vz:.2f} m/s but actuator only achieves ~{max_hand_vel:.2f} m/s")
        print(f"   Need to increase actuator gains significantly.")

        # Test with higher gains
        print(f"\n4. TESTING HIGHER GAINS:")

        for kp in [50000, 100000, 200000]:
            # Create new model with higher gains
            xml = f"""
            <mujoco model="ball_hand_highgain">
                <option gravity="0 0 -9.81" timestep="0.0005"/>
                <worldbody>
                    <geom type="plane" size="20 20 0.1"/>
                    <body name="hand" pos="0 0 1.5">
                        <joint name="hx" type="slide" axis="1 0 0" damping="200"/>
                        <joint name="hy" type="slide" axis="0 1 0" damping="200"/>
                        <joint name="hz" type="slide" axis="0 0 1" damping="200"/>
                        <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
                    </body>
                    <body name="ball" pos="0 0 1.66">
                        <freejoint name="ball_joint"/>
                        <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"/>
                    </body>
                </worldbody>
                <actuator>
                    <position name="hx_ctrl" joint="hx" kp="{kp}" kv="{kp//10}"/>
                    <position name="hy_ctrl" joint="hy" kp="{kp}" kv="{kp//10}"/>
                    <position name="hz_ctrl" joint="hz" kp="{kp}" kv="{kp//10}"/>
                </actuator>
            </mujoco>
            """
            model_hg = mujoco.MjModel.from_xml_string(xml)
            data_hg = mujoco.MjData(model_hg)

            mujoco.mj_resetData(model_hg, data_hg)
            data_hg.qpos[0:3] = hand_offset
            data_hg.qpos[3:6] = [init_hand[0], init_hand[1], ball_z]
            data_hg.qpos[6:10] = [1, 0, 0, 0]
            data_hg.qvel[:] = 0
            data_hg.ctrl[:] = hand_offset
            mujoco.mj_forward(model_hg, data_hg)

            for _ in range(50):
                mujoco.mj_step(model_hg, data_hg)

            max_hand_vel_hg = 0
            max_ball_vel_hg = 0

            for idx in range(start_idx, min(start_idx + 40, len(hand_pos))):
                target = hand_pos[idx]
                target_offset = target - np.array([0, 0, 1.5])
                data_hg.ctrl[:] = target_offset

                for _ in range(steps_per_frame):
                    mujoco.mj_step(model_hg, data_hg)

                hand_vel_actual = data_hg.qvel[0:3]
                ball_vel = data_hg.qvel[3:6]

                max_hand_vel_hg = max(max_hand_vel_hg, np.linalg.norm(hand_vel_actual))
                max_ball_vel_hg = max(max_ball_vel_hg, np.linalg.norm(ball_vel))

            print(f"   kp={kp}: hand={max_hand_vel_hg:.2f} m/s, ball={max_ball_vel_hg:.2f} m/s ({max_ball_vel_hg/FEET_TO_METERS:.1f} ft/s)")


if __name__ == "__main__":
    main()
