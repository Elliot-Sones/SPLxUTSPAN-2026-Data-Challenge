"""
Analyze the actual release moment vs what we're detecting.

The ball leaves the hand when:
1. Forward velocity starts decreasing (hand slowing down)
2. The fingers extend/snap
3. Ball separates from fingertips

We should NOT use peak upward velocity - that's AFTER release.
The release happens when TOTAL velocity (especially forward) is highest.
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
BALL_RADIUS = 4.7 / 12.0
FPS = 60
GRAVITY = 32.174  # ft/s^2


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

    return centroid


def find_true_release(positions, frames, vx, vy, vz, speed, hoop_pos):
    """
    Find the TRUE release frame based on physical constraints.

    The ball is released when:
    1. It's still moving toward the hoop (vx < 0 in our coords)
    2. It's moving upward (vz > 0)
    3. The motion is consistent with actually reaching the hoop

    We know the ball MUST have enough velocity to reach the hoop.
    Let's work backwards from that constraint.
    """
    # Distance to hoop
    results = []

    for i in range(len(frames)):
        if vz[i] < 0:  # Skip if moving down
            continue

        # Position and velocity at this frame
        pos = positions[i]
        vel = np.array([vx[i], vy[i], vz[i]])

        # Direction to hoop
        to_hoop = hoop_pos - pos
        horiz_dist = np.sqrt(to_hoop[0]**2 + to_hoop[1]**2)
        height_diff = to_hoop[2]

        # Current velocity toward hoop
        horiz_dir = to_hoop[:2] / (horiz_dist + 1e-6)
        vel_toward = vel[0] * horiz_dir[0] + vel[1] * horiz_dir[1]
        vel_up = vz[i]

        # Would this velocity reach the hoop?
        # Simple check: can projectile motion cover the distance?
        # x = vh * t, z = vz * t - 0.5 * g * t^2
        # At hoop: x = horiz_dist, z = height_diff
        # t = horiz_dist / vh
        # height_diff = vz * t - 0.5 * g * t^2

        if vel_toward > 0.5:  # Must be moving toward hoop
            t = horiz_dist / vel_toward
            z_at_hoop = vel_up * t - 0.5 * GRAVITY * t * t

            # How close would it get to hoop height?
            height_error = z_at_hoop - height_diff

            results.append({
                'frame': frames[i],
                'idx': i,
                'vel_toward': vel_toward,
                'vel_up': vel_up,
                'speed': speed[i],
                't_to_hoop': t,
                'z_at_hoop': z_at_hoop,
                'height_error': height_error,
                'horiz_dist': horiz_dist,
                'height_diff': height_diff,
            })

    return results


def analyze_shot(timeseries, keypoint_map, metadata):
    """Analyze a single shot for release point detection."""

    frames = []
    positions = []

    for frame in range(50, 200):
        pos = calculate_ball_position(timeseries, keypoint_map, frame)
        if pos is not None:
            frames.append(frame)
            positions.append(pos)

    if len(positions) < 20:
        return None

    frames = np.array(frames)
    positions = np.array(positions)

    # Compute velocity
    window = 11
    vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Find where velocity toward hoop is maximum (not upward velocity!)
    # Velocity toward hoop is in -X direction (shooter at X~18, hoop at X~5.25)
    vel_toward_hoop = -vx  # Positive when moving toward hoop

    # Find peak forward velocity (while still going up)
    forward_peaks = []
    for i in range(len(frames)):
        if vz[i] > 1.0:  # Must be moving upward
            forward_peaks.append((i, vel_toward_hoop[i]))

    if not forward_peaks:
        return None

    # Find max forward velocity
    best_idx = max(forward_peaks, key=lambda x: x[1])[0]
    peak_forward_frame = frames[best_idx]

    # Compare to peak upward velocity
    peak_up_idx = np.argmax(np.where(vz > 0, vz, 0))
    peak_up_frame = frames[peak_up_idx]

    print(f"\nShot {metadata['id'][:8]} (Player {metadata['participant_id']}):")
    print(f"  Peak forward velocity: frame {peak_forward_frame}")
    print(f"    vx = {vx[best_idx]:.2f}, vz = {vz[best_idx]:.2f}, speed = {speed[best_idx]:.2f}")
    print(f"    vel toward hoop = {vel_toward_hoop[best_idx]:.2f} ft/s")
    print(f"  Peak upward velocity: frame {peak_up_frame}")
    print(f"    vx = {vx[peak_up_idx]:.2f}, vz = {vz[peak_up_idx]:.2f}, speed = {speed[peak_up_idx]:.2f}")
    print(f"    vel toward hoop = {vel_toward_hoop[peak_up_idx]:.2f} ft/s")

    # The TRUE release should be closer to peak forward, not peak upward
    # because at peak upward, horizontal velocity has already decreased

    # Calculate required velocity to reach hoop from each candidate frame
    print(f"\n  Required velocity analysis:")

    for idx, frame in [(best_idx, peak_forward_frame), (peak_up_idx, peak_up_frame)]:
        pos = positions[idx]
        to_hoop = HOOP_POSITION - pos
        horiz_dist = np.sqrt(to_hoop[0]**2 + to_hoop[1]**2)
        height_diff = to_hoop[2]

        # For a 50 degree release angle, what speed is needed?
        theta = np.radians(50)
        denom = horiz_dist * np.tan(theta) - height_diff
        if denom > 0:
            v_req_squared = GRAVITY * horiz_dist**2 / (2 * np.cos(theta)**2 * denom)
            v_required = np.sqrt(v_req_squared)
        else:
            v_required = float('inf')

        v_actual = speed[idx]
        print(f"    Frame {frame}: actual speed = {v_actual:.1f} ft/s, required = {v_required:.1f} ft/s ({v_actual/v_required*100:.0f}%)")

    # What if we use a SMALLER window for velocity calculation?
    print(f"\n  Effect of window size on peak velocity:")
    for win in [5, 7, 9, 11, 15]:
        if win >= len(positions):
            continue
        try:
            vx_w = savgol_filter(positions[:, 0], win, 3, deriv=1) * FPS
            vz_w = savgol_filter(positions[:, 2], win, 3, deriv=1) * FPS
            speed_w = np.sqrt(vx_w**2 + vy**2 + vz_w**2)
            peak_speed = np.max(speed_w)
            print(f"    Window {win}: peak speed = {peak_speed:.2f} ft/s")
        except:
            pass

    return {
        'shot_id': metadata['id'],
        'peak_forward_frame': peak_forward_frame,
        'peak_forward_speed': speed[best_idx],
        'vel_toward_hoop': vel_toward_hoop[best_idx],
        'peak_up_frame': peak_up_frame,
        'peak_up_speed': speed[peak_up_idx],
    }


def main():
    print("=" * 80)
    print("RELEASE POINT ANALYSIS")
    print("=" * 80)
    print("\nComparing peak FORWARD velocity vs peak UPWARD velocity")
    print("The true release happens when ball is moving forward AND up, not just up.")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    results = []
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 5:
            break
        result = analyze_shot(timeseries, keypoint_map, metadata)
        if result:
            results.append(result)

    if results:
        print("\n" + "=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)

        forward_speeds = [r['vel_toward_hoop'] for r in results]
        print(f"\nVelocity toward hoop at peak forward frame:")
        print(f"  Mean: {np.mean(forward_speeds):.2f} ft/s")
        print(f"  Max:  {np.max(forward_speeds):.2f} ft/s")
        print(f"  Required: ~15-17 ft/s")
        print(f"\nThe maximum forward velocity we can extract is only {np.max(forward_speeds):.1f} ft/s,")
        print(f"which is about {np.max(forward_speeds)/16*100:.0f}% of what's needed.")


if __name__ == "__main__":
    main()
