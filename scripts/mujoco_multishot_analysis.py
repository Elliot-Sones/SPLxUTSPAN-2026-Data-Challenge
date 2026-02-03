"""
Analyze hand velocities across multiple shots to understand the distribution.
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


def analyze_shot_velocities(timeseries, keypoint_map):
    """Analyze velocities of different keypoints."""
    results = {}

    keypoints_to_check = [
        'right_third_finger_distal',
        'right_wrist',
        'right_elbow',
        'right_shoulder',
    ]

    for kp_name in keypoints_to_check:
        positions = []
        frames = []
        for frame in range(50, 200):
            pos = get_position(timeseries, keypoint_map, kp_name, frame)
            if pos is not None:
                positions.append(pos * FEET_TO_METERS)
                frames.append(frame)

        if len(positions) < 20:
            results[kp_name] = None
            continue

        positions = np.array(positions)
        frames = np.array(frames)

        # Compute velocities
        window = min(11, len(positions) - 2)
        if window % 2 == 0:
            window -= 1

        vel = np.zeros_like(positions)
        for i in range(3):
            vel[:, i] = savgol_filter(positions[:, i], window, 3, deriv=1) * FPS

        speeds = np.linalg.norm(vel, axis=1)
        vz = vel[:, 2]

        results[kp_name] = {
            'max_speed': np.max(speeds),
            'max_vz': np.max(vz),
            'mean_speed': np.mean(speeds),
            'peak_frame': frames[np.argmax(speeds)],
            'peak_vz_frame': frames[np.argmax(vz)],
        }

    return results


def main():
    print("=" * 80)
    print("MULTI-SHOT VELOCITY ANALYSIS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Collect stats across shots
    all_finger_speeds = []
    all_wrist_speeds = []
    all_finger_vz = []
    all_wrist_vz = []

    print("\nPer-shot analysis (first 50 shots):")
    print(f"{'Shot':<12} {'Finger Speed':<15} {'Wrist Speed':<15} {'Finger vz':<15} {'Wrist vz':<15}")
    print("-" * 75)

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 50:
            break

        results = analyze_shot_velocities(timeseries, keypoint_map)

        finger = results.get('right_third_finger_distal')
        wrist = results.get('right_wrist')

        if finger:
            all_finger_speeds.append(finger['max_speed'])
            all_finger_vz.append(finger['max_vz'])
        if wrist:
            all_wrist_speeds.append(wrist['max_speed'])
            all_wrist_vz.append(wrist['max_vz'])

        if i < 20:  # Print first 20
            fs = f"{finger['max_speed']:.2f} ({finger['max_speed']/FEET_TO_METERS:.1f}ft/s)" if finger else "N/A"
            ws = f"{wrist['max_speed']:.2f} ({wrist['max_speed']/FEET_TO_METERS:.1f}ft/s)" if wrist else "N/A"
            fvz = f"{finger['max_vz']:.2f}" if finger else "N/A"
            wvz = f"{wrist['max_vz']:.2f}" if wrist else "N/A"
            print(f"{metadata['id'][:10]:<12} {fs:<15} {ws:<15} {fvz:<15} {wvz:<15}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if all_finger_speeds:
        print(f"\nFinger (right_third_finger_distal):")
        print(f"  Max speed - Mean: {np.mean(all_finger_speeds):.2f} m/s ({np.mean(all_finger_speeds)/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Max speed - Max:  {np.max(all_finger_speeds):.2f} m/s ({np.max(all_finger_speeds)/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Max speed - Min:  {np.min(all_finger_speeds):.2f} m/s ({np.min(all_finger_speeds)/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Max vz - Mean:    {np.mean(all_finger_vz):.2f} m/s")
        print(f"  Max vz - Max:     {np.max(all_finger_vz):.2f} m/s")

    if all_wrist_speeds:
        print(f"\nWrist (right_wrist):")
        print(f"  Max speed - Mean: {np.mean(all_wrist_speeds):.2f} m/s ({np.mean(all_wrist_speeds)/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Max speed - Max:  {np.max(all_wrist_speeds):.2f} m/s ({np.max(all_wrist_speeds)/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Max speed - Min:  {np.min(all_wrist_speeds):.2f} m/s ({np.min(all_wrist_speeds)/FEET_TO_METERS:.1f} ft/s)")
        print(f"  Max vz - Mean:    {np.mean(all_wrist_vz):.2f} m/s")
        print(f"  Max vz - Max:     {np.max(all_wrist_vz):.2f} m/s")

    print("\n" + "=" * 80)
    print("REQUIRED VELOCITY ANALYSIS")
    print("=" * 80)

    # For a free throw trajectory
    print("\nFree throw physics:")
    print("  Distance to hoop: ~15 feet (4.57m)")
    print("  Height change: ~2 feet (0.61m)")
    print("  Typical flight time: 0.8-1.0 seconds")

    # Calculate required velocity
    # Assuming 45 degree launch angle (optimal for height)
    # v = sqrt(g * d / sin(2*theta)) where theta=45 deg
    g = 9.81
    d = 4.57  # horizontal distance
    theta = np.radians(50)  # typical launch angle

    # For projectile motion: d = v*cos(theta)*t, where t = 2*v*sin(theta)/g (time to peak and back)
    # Simplified: v = sqrt(g*d / sin(2*theta))
    v_required = np.sqrt(g * d / np.sin(2 * theta))

    print(f"\n  Required release speed (approx): {v_required:.2f} m/s ({v_required/FEET_TO_METERS:.1f} ft/s)")
    print(f"  Observed finger speeds: {np.mean(all_finger_speeds):.2f} - {np.max(all_finger_speeds):.2f} m/s")

    if np.max(all_finger_speeds) >= v_required:
        print("\n  FINDING: Some shots have sufficient velocity for realistic trajectories!")
    else:
        print(f"\n  GAP: Need {v_required:.2f} m/s, max observed is {np.max(all_finger_speeds):.2f} m/s")
        print(f"  Deficit: {(v_required - np.max(all_finger_speeds)):.2f} m/s")


if __name__ == "__main__":
    main()
