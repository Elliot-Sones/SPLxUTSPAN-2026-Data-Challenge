"""
Diagnose velocity extraction - what are we actually measuring?

The whole point of simulation is that we simulate it.
If velocity is off, we're missing something in the calculation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

HOOP_POSITION = np.array([5.25, -25.0, 10.0])  # feet
FPS = 60


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


def analyze_shot_kinematics(timeseries, keypoint_map):
    """
    Analyze the full kinematic chain during a shot.
    Track position and velocity of key joints through the shot.
    """
    # Key joints in the kinetic chain
    joints = [
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'right_second_finger_distal',  # Index fingertip
        'right_third_finger_distal',   # Middle fingertip
    ]

    # Track positions for each joint
    joint_data = {j: {'pos': [], 'frames': []} for j in joints}

    for frame in range(50, 200):
        for joint in joints:
            pos = get_position(timeseries, keypoint_map, joint, frame)
            if pos is not None:
                joint_data[joint]['pos'].append(pos)
                joint_data[joint]['frames'].append(frame)

    # Compute velocities
    results = {}
    for joint in joints:
        if len(joint_data[joint]['pos']) < 15:
            continue

        pos = np.array(joint_data[joint]['pos'])
        frames = np.array(joint_data[joint]['frames'])

        # Compute velocity using Savitzky-Golay
        window = min(15, len(pos) - 2)
        if window % 2 == 0:
            window -= 1
        if window < 5:
            window = 5

        try:
            vx = savgol_filter(pos[:, 0], window, 3, deriv=1) * FPS
            vy = savgol_filter(pos[:, 1], window, 3, deriv=1) * FPS
            vz = savgol_filter(pos[:, 2], window, 3, deriv=1) * FPS
        except:
            vx = np.gradient(pos[:, 0]) * FPS
            vy = np.gradient(pos[:, 1]) * FPS
            vz = np.gradient(pos[:, 2]) * FPS

        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        # Find peak velocity
        peak_idx = np.argmax(speed)
        peak_frame = frames[peak_idx]
        peak_speed = speed[peak_idx]
        peak_vel = np.array([vx[peak_idx], vy[peak_idx], vz[peak_idx]])

        results[joint] = {
            'peak_frame': peak_frame,
            'peak_speed': peak_speed,
            'peak_vel': peak_vel,
            'pos_at_peak': pos[peak_idx],
        }

    return results


def main():
    print("=" * 80)
    print("VELOCITY EXTRACTION DIAGNOSIS")
    print("=" * 80)
    print()
    print("Question: Why are extracted velocities ~1-4 ft/s when free throws need ~22 ft/s?")
    print()

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Analyze a few shots in detail
    all_results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 20:  # Analyze first 20 shots
            break

        shot_id = metadata['id']
        player_id = metadata['participant_id']

        kinematics = analyze_shot_kinematics(timeseries, keypoint_map)

        if 'right_wrist' in kinematics and 'right_third_finger_distal' in kinematics:
            wrist = kinematics['right_wrist']
            finger = kinematics['right_third_finger_distal']

            all_results.append({
                'shot_id': shot_id,
                'player_id': player_id,
                'wrist_peak_speed': wrist['peak_speed'],
                'wrist_peak_frame': wrist['peak_frame'],
                'wrist_peak_vx': wrist['peak_vel'][0],
                'wrist_peak_vy': wrist['peak_vel'][1],
                'wrist_peak_vz': wrist['peak_vel'][2],
                'finger_peak_speed': finger['peak_speed'],
                'finger_peak_frame': finger['peak_frame'],
                'finger_peak_vx': finger['peak_vel'][0],
                'finger_peak_vy': finger['peak_vel'][1],
                'finger_peak_vz': finger['peak_vel'][2],
            })

            if i < 5:
                print(f"Shot {shot_id} (Player {player_id}):")
                print(f"  Wrist:  peak speed = {wrist['peak_speed']:.2f} ft/s at frame {wrist['peak_frame']}")
                print(f"          velocity = [{wrist['peak_vel'][0]:.2f}, {wrist['peak_vel'][1]:.2f}, {wrist['peak_vel'][2]:.2f}]")
                print(f"          position = {wrist['pos_at_peak']}")
                print(f"  Finger: peak speed = {finger['peak_speed']:.2f} ft/s at frame {finger['peak_frame']}")
                print(f"          velocity = [{finger['peak_vel'][0]:.2f}, {finger['peak_vel'][1]:.2f}, {finger['peak_vel'][2]:.2f}]")
                print(f"          position = {finger['pos_at_peak']}")
                print()

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nWrist peak speeds (ft/s):")
        print(f"  Mean: {df['wrist_peak_speed'].mean():.2f}")
        print(f"  Std:  {df['wrist_peak_speed'].std():.2f}")
        print(f"  Min:  {df['wrist_peak_speed'].min():.2f}")
        print(f"  Max:  {df['wrist_peak_speed'].max():.2f}")

        print(f"\nFinger peak speeds (ft/s):")
        print(f"  Mean: {df['finger_peak_speed'].mean():.2f}")
        print(f"  Std:  {df['finger_peak_speed'].std():.2f}")
        print(f"  Min:  {df['finger_peak_speed'].min():.2f}")
        print(f"  Max:  {df['finger_peak_speed'].max():.2f}")

        print(f"\nVelocity components at wrist peak (ft/s):")
        print(f"  vx (toward hoop): {df['wrist_peak_vx'].mean():.2f} +/- {df['wrist_peak_vx'].std():.2f}")
        print(f"  vy (lateral):     {df['wrist_peak_vy'].mean():.2f} +/- {df['wrist_peak_vy'].std():.2f}")
        print(f"  vz (vertical):    {df['wrist_peak_vz'].mean():.2f} +/- {df['wrist_peak_vz'].std():.2f}")

        # Check if data might be in different units
        print("\n" + "=" * 80)
        print("UNIT CHECK")
        print("=" * 80)

        # Get a sample shot and check positions
        for metadata, timeseries in iterate_shots(train=True):
            # Check wrist height - should be ~6-7 feet during shot
            for frame in [100, 120, 140, 160]:
                wrist_pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
                if wrist_pos is not None:
                    print(f"Frame {frame}: wrist position = {wrist_pos}")

            # Check ankle to see floor level
            ankle_pos = get_position(timeseries, keypoint_map, 'right_ankle', 100)
            if ankle_pos is not None:
                print(f"Right ankle at frame 100: {ankle_pos}")

            # Check shoulder width to validate scale
            l_shoulder = get_position(timeseries, keypoint_map, 'left_shoulder', 100)
            r_shoulder = get_position(timeseries, keypoint_map, 'right_shoulder', 100)
            if l_shoulder is not None and r_shoulder is not None:
                width = np.linalg.norm(l_shoulder - r_shoulder)
                print(f"Shoulder width: {width:.2f} feet (expected ~1.5 feet)")

            break

        # Calculate what the ball velocity SHOULD be
        print("\n" + "=" * 80)
        print("PHYSICS CHECK: What velocity is needed?")
        print("=" * 80)

        # Free throw line is ~15 feet from hoop center
        # Release height ~7 feet, hoop at 10 feet
        # Using projectile motion:
        # For 45-degree release: v = sqrt(g*d / sin(2*theta)) where d is horizontal distance
        # For higher angles (typical 50-55 degrees), need slightly more speed

        horizontal_dist = 15.0  # feet (approximate)
        height_diff = 3.0  # feet (10 - 7)
        g = 32.174  # ft/s^2

        for angle in [45, 50, 52, 55]:
            theta = np.radians(angle)
            # Time to reach horizontal distance
            # t = d / (v * cos(theta))
            # Height: h = v*sin(theta)*t - 0.5*g*t^2 = height_diff
            # Solving: v^2 = g*d^2 / (2*cos^2(theta) * (d*tan(theta) - height_diff))

            denom = horizontal_dist * np.tan(theta) - height_diff
            if denom > 0:
                v_squared = g * horizontal_dist**2 / (2 * np.cos(theta)**2 * denom)
                v = np.sqrt(v_squared)
                vh = v * np.cos(theta)
                vz = v * np.sin(theta)
                print(f"  {angle} degrees: v = {v:.1f} ft/s (vh = {vh:.1f}, vz = {vz:.1f})")


if __name__ == "__main__":
    main()
