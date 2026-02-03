"""
Trace a shot frame-by-frame to understand the motion.

The ball's velocity comes from the kinetic chain:
1. Elbow extension (pushes ball forward)
2. Wrist flick (adds arc)
3. Finger snap (final touch)

We need to track ball position through the ENTIRE shot motion
and compute velocity at the RIGHT moment (release frame).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

HOOP_POSITION = np.array([5.25, -25.0, 10.0])
BALL_RADIUS = 4.7 / 12.0  # feet
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


def calculate_ball_position(timeseries, keypoint_map, frame):
    """Calculate ball center from fingertip positions."""
    fingertips = [
        'right_first_finger_distal',
        'right_second_finger_distal',
        'right_third_finger_distal',
        'right_fourth_finger_distal',
        'right_fifth_finger_distal',
    ]
    weights = [0.15, 0.25, 0.30, 0.20, 0.10]

    positions = []
    valid_weights = []

    for tip, w in zip(fingertips, weights):
        pos = get_position(timeseries, keypoint_map, tip, frame)
        if pos is not None:
            positions.append(pos)
            valid_weights.append(w)

    if len(positions) < 3:
        return None

    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()

    centroid = np.zeros(3)
    for pos, w in zip(positions, valid_weights):
        centroid += pos * w

    # Ball center is above fingertips
    wrist = get_position(timeseries, keypoint_map, 'right_wrist', frame)
    if wrist is not None:
        palm_dir = centroid - wrist
        palm_dir = palm_dir / (np.linalg.norm(palm_dir) + 1e-6)
        ball_center = centroid + palm_dir * BALL_RADIUS * 0.5
    else:
        ball_center = centroid + np.array([0, 0, BALL_RADIUS])

    return ball_center


def trace_shot(timeseries, keypoint_map, shot_id, player_id):
    """
    Trace ball position and velocity through the entire shot.
    """
    print(f"\n{'='*80}")
    print(f"SHOT {shot_id[:8]} (Player {player_id})")
    print(f"{'='*80}")

    # Track ball position through frames
    frames = []
    positions = []

    for frame in range(50, 200):
        pos = calculate_ball_position(timeseries, keypoint_map, frame)
        if pos is not None:
            frames.append(frame)
            positions.append(pos)

    if len(positions) < 20:
        print("  Not enough data")
        return None

    frames = np.array(frames)
    positions = np.array(positions)

    # Compute velocity using Savitzky-Golay
    window = 11
    try:
        vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
        vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
        vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS
    except:
        vx = np.gradient(positions[:, 0]) * FPS
        vy = np.gradient(positions[:, 1]) * FPS
        vz = np.gradient(positions[:, 2]) * FPS

    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Find key moments:
    # 1. Peak height (release is near this)
    # 2. Peak upward velocity (release)
    # 3. Peak total speed

    peak_height_idx = np.argmax(positions[:, 2])
    peak_height_frame = frames[peak_height_idx]

    # Peak upward velocity (vz > 0)
    upward_vz = np.where(vz > 0, vz, 0)
    peak_up_idx = np.argmax(upward_vz)
    peak_up_frame = frames[peak_up_idx]

    # Peak total speed
    peak_speed_idx = np.argmax(speed)
    peak_speed_frame = frames[peak_speed_idx]

    print(f"\nKey moments:")
    print(f"  Peak height:   frame {peak_height_frame}, Z = {positions[peak_height_idx, 2]:.2f} ft")
    print(f"  Peak up vel:   frame {peak_up_frame}, vz = {vz[peak_up_idx]:.2f} ft/s")
    print(f"  Peak speed:    frame {peak_speed_frame}, speed = {speed[peak_speed_idx]:.2f} ft/s")

    # The release frame should be near peak upward velocity
    release_idx = peak_up_idx

    print(f"\nAt release (frame {frames[release_idx]}):")
    print(f"  Position: [{positions[release_idx, 0]:.2f}, {positions[release_idx, 1]:.2f}, {positions[release_idx, 2]:.2f}]")
    print(f"  Velocity: [{vx[release_idx]:.2f}, {vy[release_idx]:.2f}, {vz[release_idx]:.2f}] ft/s")
    print(f"  Speed:    {speed[release_idx]:.2f} ft/s")

    # Analyze the velocity direction
    vel_at_release = np.array([vx[release_idx], vy[release_idx], vz[release_idx]])
    speed_at_release = speed[release_idx]

    # Direction toward hoop
    to_hoop = HOOP_POSITION - positions[release_idx]
    to_hoop_horizontal = np.array([to_hoop[0], to_hoop[1], 0])
    to_hoop_dir = to_hoop_horizontal / (np.linalg.norm(to_hoop_horizontal) + 1e-6)

    # Project velocity onto hoop direction
    vel_horizontal = np.array([vx[release_idx], vy[release_idx], 0])
    vel_toward_hoop = np.dot(vel_horizontal, to_hoop_dir)

    print(f"\n  Distance to hoop: {np.linalg.norm(to_hoop_horizontal):.2f} ft horizontal")
    print(f"  Velocity toward hoop: {vel_toward_hoop:.2f} ft/s")
    print(f"  Velocity upward: {vz[release_idx]:.2f} ft/s")

    # What angle is this?
    if speed_at_release > 1:
        angle = np.degrees(np.arctan2(vz[release_idx], np.linalg.norm(vel_horizontal)))
        print(f"  Release angle: {angle:.1f} degrees")

    # Show the motion sequence around release
    print(f"\nFrame-by-frame around release (frames {frames[max(0,release_idx-5)]} to {frames[min(len(frames)-1, release_idx+5)]}):")
    print(f"  {'Frame':>6} {'X':>8} {'Y':>8} {'Z':>8} {'vx':>8} {'vy':>8} {'vz':>8} {'speed':>8}")

    for i in range(max(0, release_idx - 5), min(len(frames), release_idx + 6)):
        print(f"  {frames[i]:>6} {positions[i,0]:>8.2f} {positions[i,1]:>8.2f} {positions[i,2]:>8.2f} "
              f"{vx[i]:>8.2f} {vy[i]:>8.2f} {vz[i]:>8.2f} {speed[i]:>8.2f}")

    # Check the shoulder and elbow motion
    print(f"\nKinetic chain analysis:")

    # Get shoulder, elbow, wrist positions at release
    shoulder = get_position(timeseries, keypoint_map, 'right_shoulder', frames[release_idx])
    elbow = get_position(timeseries, keypoint_map, 'right_elbow', frames[release_idx])
    wrist = get_position(timeseries, keypoint_map, 'right_wrist', frames[release_idx])

    if shoulder is not None and elbow is not None and wrist is not None:
        upper_arm = np.linalg.norm(elbow - shoulder)
        forearm = np.linalg.norm(wrist - elbow)
        print(f"  Upper arm length: {upper_arm:.2f} ft ({upper_arm*12:.1f} inches)")
        print(f"  Forearm length: {forearm:.2f} ft ({forearm*12:.1f} inches)")

        # Arm extension (angle at elbow)
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        elbow_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        print(f"  Elbow angle: {elbow_angle:.1f} degrees (180 = fully extended)")

    return {
        'shot_id': shot_id,
        'release_frame': frames[release_idx],
        'position': positions[release_idx],
        'velocity': vel_at_release,
        'speed': speed_at_release,
        'vel_toward_hoop': vel_toward_hoop,
        'vel_up': vz[release_idx],
    }


def main():
    print("=" * 80)
    print("SHOT TRACING ANALYSIS")
    print("=" * 80)
    print("\nTracing ball motion through shots to understand velocity extraction")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 10:  # Analyze first 10 shots
            break

        result = trace_shot(timeseries, keypoint_map, metadata['id'], metadata['participant_id'])
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        speeds = [r['speed'] for r in results]
        toward_hoop = [r['vel_toward_hoop'] for r in results]
        up_vel = [r['vel_up'] for r in results]

        print(f"\nTotal speed at release: {np.mean(speeds):.2f} +/- {np.std(speeds):.2f} ft/s")
        print(f"Velocity toward hoop: {np.mean(toward_hoop):.2f} +/- {np.std(toward_hoop):.2f} ft/s")
        print(f"Upward velocity: {np.mean(up_vel):.2f} +/- {np.std(up_vel):.2f} ft/s")

        print(f"\nRequired for free throw: ~24 ft/s total, ~15 ft/s horizontal, ~19 ft/s vertical")


if __name__ == "__main__":
    main()
