"""
Compare direct kinematic features vs MuJoCo features.

Earlier analysis showed:
- Kinematic vx correlates 0.448 with angle
- MuJoCo features have negative correlations

Something is getting flipped. Let's investigate.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import pandas as pd
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

BALL_RADIUS = 0.12
FEET_TO_METERS = 0.3048
FPS = 60


def create_model():
    xml = """
    <mujoco model="compare">
        <option gravity="0 0 -9.81" timestep="0.0002"/>
        <worldbody>
            <geom type="plane" size="20 20 0.1"/>
            <body name="hand" pos="0 0 1.5">
                <joint name="hx" type="slide" axis="1 0 0"/>
                <joint name="hy" type="slide" axis="0 1 0"/>
                <joint name="hz" type="slide" axis="0 0 1"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
            </body>
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
    print("COMPARING KINEMATIC vs MUJOCO VELOCITIES")
    print("=" * 80)

    mj_model = create_model()
    mj_data = mujoco.MjData(mj_model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 50:
            break

        # Get hand trajectory
        hand_positions = []
        frames = []
        for frame in range(50, 200):
            pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
            if pos is not None and not np.any(np.isnan(pos)):
                hand_positions.append(pos)  # Keep in FEET
                frames.append(frame)

        if len(hand_positions) < 20:
            continue

        hand_positions = np.array(hand_positions)
        frames = np.array(frames)

        # Compute kinematic velocity (in feet/s)
        window = min(11, len(hand_positions) - 2)
        if window % 2 == 0:
            window -= 1
        hand_vel = np.zeros_like(hand_positions)
        for j in range(3):
            hand_vel[:, j] = savgol_filter(hand_positions[:, j], window, 3, deriv=1) * FPS

        # Find peak upward velocity (kinematic release)
        vz = hand_vel[:, 2]
        peak_idx = np.argmax(vz)

        if peak_idx < 5 or peak_idx >= len(hand_vel) - 5:
            continue

        kinematic_vel = hand_vel[peak_idx]  # In feet/s
        kinematic_frame = frames[peak_idx]

        # Now run MuJoCo simulation
        hand_pos_m = hand_positions * FEET_TO_METERS  # Convert to meters
        hand_vel_m = hand_vel * FEET_TO_METERS  # Velocity in m/s

        # Find start
        start_idx = 0
        for j in range(len(hand_vel_m)):
            if hand_vel_m[j, 2] > 0.3 * FEET_TO_METERS:
                start_idx = max(0, j - 5)
                break

        mujoco.mj_resetData(mj_model, mj_data)

        init_hand = hand_pos_m[start_idx]
        init_hand_vel = hand_vel_m[start_idx]
        hand_offset = init_hand - np.array([0, 0, 1.5])

        mj_data.qpos[0:3] = hand_offset
        palm_top_z = 1.5 + hand_offset[2] + 0.02
        ball_z = palm_top_z + BALL_RADIUS - 0.005

        mj_data.qpos[3:6] = [init_hand[0], init_hand[1], ball_z]
        mj_data.qpos[6:10] = [1, 0, 0, 0]
        mj_data.qvel[0:3] = init_hand_vel
        mj_data.qvel[3:6] = init_hand_vel
        mj_data.qvel[6:9] = [0, 0, 0]

        mujoco.mj_forward(mj_model, mj_data)

        if not check_contact(mj_model, mj_data):
            continue

        sim_dt = mj_model.opt.timestep
        frame_dt = 1.0 / FPS

        had_contact = True
        contact_count = 1
        no_contact_count = 0
        mujoco_vel = None
        mujoco_frame = None

        for idx in range(start_idx, len(hand_pos_m) - 1):
            pos_curr = hand_pos_m[idx] - np.array([0, 0, 1.5])
            pos_next = hand_pos_m[idx + 1] - np.array([0, 0, 1.5])
            vel_curr = hand_vel_m[idx]
            vel_next = hand_vel_m[idx + 1]

            frame_time = 0.0
            while frame_time < frame_dt:
                t = frame_time / frame_dt
                mj_data.qpos[0:3] = pos_curr * (1 - t) + pos_next * t
                mj_data.qvel[0:3] = vel_curr * (1 - t) + vel_next * t
                mujoco.mj_step(mj_model, mj_data)
                frame_time += sim_dt

            in_contact = check_contact(mj_model, mj_data)

            if in_contact:
                contact_count += 1
                no_contact_count = 0
            else:
                no_contact_count += 1

            if had_contact and contact_count > 3 and no_contact_count >= 2:
                mujoco_vel = mj_data.qvel[3:6].copy() / FEET_TO_METERS  # Convert back to ft/s
                mujoco_frame = frames[idx]
                break

        if mujoco_vel is None:
            mujoco_vel = mj_data.qvel[3:6].copy() / FEET_TO_METERS
            mujoco_frame = frames[-1]

        data.append({
            'id': metadata['id'],
            # Kinematic (direct from data)
            'kin_vx': kinematic_vel[0],
            'kin_vy': kinematic_vel[1],
            'kin_vz': kinematic_vel[2],
            'kin_frame': kinematic_frame,
            # MuJoCo (from simulation)
            'mj_vx': mujoco_vel[0],
            'mj_vy': mujoco_vel[1],
            'mj_vz': mujoco_vel[2],
            'mj_frame': mujoco_frame,
            # Targets
            'angle': metadata.get('angle'),
            'depth': metadata.get('depth'),
            'left_right': metadata.get('left_right'),
        })

    df = pd.DataFrame(data)
    print(f"\nAnalyzed {len(df)} shots")

    print("\n" + "=" * 60)
    print("VELOCITY COMPARISON")
    print("=" * 60)

    print("\nKinematic velocities (ft/s):")
    print(f"  vx: {df['kin_vx'].mean():.2f} +/- {df['kin_vx'].std():.2f}")
    print(f"  vy: {df['kin_vy'].mean():.2f} +/- {df['kin_vy'].std():.2f}")
    print(f"  vz: {df['kin_vz'].mean():.2f} +/- {df['kin_vz'].std():.2f}")

    print("\nMuJoCo velocities (ft/s):")
    print(f"  vx: {df['mj_vx'].mean():.2f} +/- {df['mj_vx'].std():.2f}")
    print(f"  vy: {df['mj_vy'].mean():.2f} +/- {df['mj_vy'].std():.2f}")
    print(f"  vz: {df['mj_vz'].mean():.2f} +/- {df['mj_vz'].std():.2f}")

    print("\n" + "=" * 60)
    print("CORRELATION WITH TARGETS")
    print("=" * 60)

    print(f"\n{'Velocity':<15} {'angle':>10} {'depth':>10} {'left_right':>12}")
    print("-" * 50)

    for vel_col in ['kin_vx', 'kin_vy', 'kin_vz', 'mj_vx', 'mj_vy', 'mj_vz']:
        corrs = []
        for target in ['angle', 'depth', 'left_right']:
            corr = df[vel_col].corr(df[target])
            corrs.append(corr)
        print(f"{vel_col:<15} {corrs[0]:>10.3f} {corrs[1]:>10.3f} {corrs[2]:>12.3f}")

    print("\n" + "=" * 60)
    print("KINEMATIC vs MUJOCO CORRELATION")
    print("=" * 60)

    print(f"\nvx: kin vs mj correlation: {df['kin_vx'].corr(df['mj_vx']):.3f}")
    print(f"vy: kin vs mj correlation: {df['kin_vy'].corr(df['mj_vy']):.3f}")
    print(f"vz: kin vs mj correlation: {df['kin_vz'].corr(df['mj_vz']):.3f}")

    print("\n" + "=" * 60)
    print("RELEASE FRAME COMPARISON")
    print("=" * 60)

    print(f"\nKinematic release frame: {df['kin_frame'].mean():.1f} +/- {df['kin_frame'].std():.1f}")
    print(f"MuJoCo release frame:    {df['mj_frame'].mean():.1f} +/- {df['mj_frame'].std():.1f}")
    print(f"Frame difference:        {(df['mj_frame'] - df['kin_frame']).mean():.1f} +/- {(df['mj_frame'] - df['kin_frame']).std():.1f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)

    kin_angle_corr = df['kin_vx'].corr(df['angle'])
    mj_angle_corr = df['mj_vx'].corr(df['angle'])

    print(f"\nKinematic vx -> angle: {kin_angle_corr:.3f}")
    print(f"MuJoCo vx -> angle:    {mj_angle_corr:.3f}")

    if abs(kin_angle_corr) > abs(mj_angle_corr):
        print("\n=> Kinematic features are MORE predictive!")
        print("   MuJoCo simulation may be adding noise or detecting release at wrong time.")
    else:
        print("\n=> MuJoCo features are MORE predictive!")


if __name__ == "__main__":
    main()
