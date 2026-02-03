"""
Debug when the ball releases in MuJoCo simulation.

Hypothesis: Ball separates when hand DECELERATES because contact force drops.
The ball should release at or near PEAK VELOCITY, not after.
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
    <mujoco model="debug_release">
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
    print("DEBUG: WHEN DOES BALL RELEASE IN MUJOCO?")
    print("=" * 80)

    mj_model = create_model()
    mj_data = mujoco.MjData(mj_model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Process first shot in detail
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 1:
            break

        print(f"\nShot: {metadata['id'][:8]}")

        # Get hand trajectory
        hand_positions = []
        frames = []
        for frame in range(50, 200):
            pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
            if pos is not None and not np.any(np.isnan(pos)):
                hand_positions.append(pos * FEET_TO_METERS)  # Convert to meters
                frames.append(frame)

        hand_positions = np.array(hand_positions)
        frames = np.array(frames)

        # Compute velocities and accelerations
        window = 11
        hand_vel = np.zeros_like(hand_positions)
        hand_acc = np.zeros_like(hand_positions)
        for j in range(3):
            hand_vel[:, j] = savgol_filter(hand_positions[:, j], window, 3, deriv=1) * FPS
            hand_acc[:, j] = savgol_filter(hand_positions[:, j], window, 3, deriv=2) * FPS * FPS

        # Find key moments in kinematic data
        vz = hand_vel[:, 2]
        az = hand_acc[:, 2]

        peak_vz_idx = np.argmax(vz)
        peak_vz = vz[peak_vz_idx]

        print(f"\nKinematic analysis:")
        print(f"  Peak upward velocity: {peak_vz:.2f} m/s at frame {frames[peak_vz_idx]}")

        # Show velocity profile around peak
        print(f"\n  Frame-by-frame around peak:")
        print(f"  {'Frame':<8} {'Hand Vz':<12} {'Hand Az':<12} {'Phase'}")
        print(f"  {'-'*50}")

        for idx in range(max(0, peak_vz_idx - 10), min(len(frames), peak_vz_idx + 10)):
            phase = ""
            if idx < peak_vz_idx:
                phase = "accelerating"
            elif idx == peak_vz_idx:
                phase = "PEAK"
            else:
                phase = "decelerating"
            print(f"  {frames[idx]:<8} {vz[idx]:<12.3f} {az[idx]:<12.3f} {phase}")

        # Now run MuJoCo simulation
        print("\n" + "=" * 60)
        print("MuJoCo Simulation:")
        print("=" * 60)

        # Start before peak
        start_idx = max(0, peak_vz_idx - 20)

        mujoco.mj_resetData(mj_model, mj_data)

        init_hand = hand_positions[start_idx]
        init_hand_vel = hand_vel[start_idx]
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

        print(f"\n  Starting at frame {frames[start_idx]}")
        print(f"  Initial contact: {check_contact(mj_model, mj_data)}")

        sim_dt = mj_model.opt.timestep
        frame_dt = 1.0 / FPS

        print(f"\n  {'Frame':<8} {'Hand Vz':<10} {'Ball Vz':<10} {'Contact':<10} {'Phase'}")
        print(f"  {'-'*55}")

        contact_lost_frame = None
        contact_lost_ball_vel = None

        for idx in range(start_idx, min(len(hand_positions) - 1, peak_vz_idx + 15)):
            pos_curr = hand_positions[idx] - np.array([0, 0, 1.5])
            pos_next = hand_positions[idx + 1] - np.array([0, 0, 1.5])
            vel_curr = hand_vel[idx]
            vel_next = hand_vel[idx + 1]

            frame_time = 0.0
            while frame_time < frame_dt:
                t = frame_time / frame_dt
                mj_data.qpos[0:3] = pos_curr * (1 - t) + pos_next * t
                mj_data.qvel[0:3] = vel_curr * (1 - t) + vel_next * t
                mujoco.mj_step(mj_model, mj_data)
                frame_time += sim_dt

            ball_vel = mj_data.qvel[3:6]
            hand_vz_current = hand_vel[idx, 2]
            ball_vz = ball_vel[2]
            contact = check_contact(mj_model, mj_data)

            phase = ""
            if idx < peak_vz_idx:
                phase = "accel"
            elif idx == peak_vz_idx:
                phase = "PEAK"
            else:
                phase = "decel"

            print(f"  {frames[idx]:<8} {hand_vz_current:<10.3f} {ball_vz:<10.3f} {str(contact):<10} {phase}")

            if contact_lost_frame is None and not contact:
                contact_lost_frame = frames[idx]
                contact_lost_ball_vel = ball_vel.copy()

        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        print(f"\nKinematic peak velocity at frame: {frames[peak_vz_idx]}")
        print(f"MuJoCo contact lost at frame:     {contact_lost_frame}")

        if contact_lost_frame is not None:
            frame_diff = contact_lost_frame - frames[peak_vz_idx]
            print(f"\nDifference: {frame_diff} frames")

            if frame_diff < 0:
                print("=> Ball releases BEFORE peak velocity!")
                print("   This explains why MuJoCo ball velocity < kinematic hand velocity")
            elif frame_diff > 0:
                print("=> Ball releases AFTER peak velocity")
                print("   Hand is decelerating, ball has lower velocity")
            else:
                print("=> Ball releases AT peak velocity - optimal!")

            if contact_lost_ball_vel is not None:
                print(f"\nBall velocity at release: {contact_lost_ball_vel}")
                print(f"Hand velocity at peak:    {hand_vel[peak_vz_idx]}")


if __name__ == "__main__":
    main()
