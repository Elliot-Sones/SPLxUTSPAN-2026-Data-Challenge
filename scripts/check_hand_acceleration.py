"""
Check the actual hand acceleration in the data.

For the ball to reach 24 ft/s (7.3 m/s) during a ~0.3 second push:
  Required acceleration = 7.3 / 0.3 = 24 m/s^2

The contact force on the ball = m * (g + a)
  If a = 24 m/s^2: F = 0.625 * (9.81 + 24) = 21 N

Let's see what acceleration the hand data actually shows.
"""

import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

FEET_TO_METERS = 0.3048
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


def analyze_hand_kinematics(timeseries, keypoint_map):
    """Analyze hand position, velocity, and acceleration."""

    positions = []
    frames = []

    for frame in range(50, 200):
        pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
        if pos is None:
            pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if pos is not None:
            positions.append(pos * FEET_TO_METERS)  # Convert to meters
            frames.append(frame)

    if len(positions) < 20:
        return None

    positions = np.array(positions)
    frames = np.array(frames)

    # Compute velocity (m/s)
    window = 9
    vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS

    # Compute acceleration (m/s^2)
    ax = savgol_filter(positions[:, 0], window, 3, deriv=2) * FPS * FPS
    ay = savgol_filter(positions[:, 1], window, 3, deriv=2) * FPS * FPS
    az = savgol_filter(positions[:, 2], window, 3, deriv=2) * FPS * FPS

    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    return {
        'frames': frames,
        'positions': positions,
        'vx': vx, 'vy': vy, 'vz': vz,
        'ax': ax, 'ay': ay, 'az': az,
        'speed': speed,
        'accel_mag': accel_mag,
    }


def main():
    print("=" * 80)
    print("HAND ACCELERATION ANALYSIS")
    print("=" * 80)
    print("\nRequired for 24 ft/s release:")
    print("  Acceleration: ~24 m/s^2")
    print("  Contact force: ~21 N (vs 6 N for just supporting weight)")
    print()

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    all_max_speed = []
    all_max_accel = []
    all_max_az = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 20:
            break

        data = analyze_hand_kinematics(timeseries, keypoint_map)
        if data is None:
            continue

        # Find peaks
        max_speed = np.max(data['speed'])
        max_accel = np.max(data['accel_mag'])
        max_az = np.max(data['az'])  # Upward acceleration

        all_max_speed.append(max_speed)
        all_max_accel.append(max_accel)
        all_max_az.append(max_az)

        if i < 5:
            print(f"Shot {i+1}:")
            print(f"  Max speed: {max_speed:.2f} m/s ({max_speed/FEET_TO_METERS:.1f} ft/s)")
            print(f"  Max total accel: {max_accel:.1f} m/s^2")
            print(f"  Max upward accel (az): {max_az:.1f} m/s^2")

            # What contact force would this create?
            ball_mass = 0.625
            g = 9.81
            contact_force = ball_mass * (g + max_az)
            print(f"  Implied contact force: {contact_force:.1f} N")
            print()

    print("=" * 80)
    print("SUMMARY ACROSS ALL SHOTS")
    print("=" * 80)

    print(f"\nMax hand speed:")
    print(f"  Mean: {np.mean(all_max_speed):.2f} m/s ({np.mean(all_max_speed)/FEET_TO_METERS:.1f} ft/s)")
    print(f"  Max:  {np.max(all_max_speed):.2f} m/s ({np.max(all_max_speed)/FEET_TO_METERS:.1f} ft/s)")

    print(f"\nMax upward acceleration (az):")
    print(f"  Mean: {np.mean(all_max_az):.1f} m/s^2")
    print(f"  Max:  {np.max(all_max_az):.1f} m/s^2")
    print(f"  Required: ~24 m/s^2")

    print(f"\nMax total acceleration:")
    print(f"  Mean: {np.mean(all_max_accel):.1f} m/s^2")
    print(f"  Max:  {np.max(all_max_accel):.1f} m/s^2")

    # What velocity would we get with observed accelerations?
    # v = a * t, assuming t = 0.3s push duration
    push_time = 0.3
    max_possible_vel = np.mean(all_max_az) * push_time
    print(f"\nWith {push_time}s push at mean acceleration ({np.mean(all_max_az):.1f} m/s^2):")
    print(f"  Achievable velocity: {max_possible_vel:.1f} m/s ({max_possible_vel/FEET_TO_METERS:.0f} ft/s)")
    print(f"  Required: 7.3 m/s (24 ft/s)")

    # Gap analysis
    ratio = max_possible_vel / 7.3
    print(f"\nWe have {ratio*100:.0f}% of the required acceleration in the data.")


if __name__ == "__main__":
    main()
