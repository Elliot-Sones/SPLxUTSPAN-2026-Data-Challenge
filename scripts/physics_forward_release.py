"""
Better release detection: find when forward velocity is maximized while still ascending.

The ball needs FORWARD momentum to reach the hoop. The release must happen
while the hand still has forward velocity, not after it's stopped.

Physics insight: the ball releases when the hand can no longer accelerate it.
But we should detect this in the DIRECTION of the shot, not just vertical.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

HOOP_POSITION = np.array([5.25, -25.0, 10.0])
BALL_RADIUS = 4.7 / 12.0
FPS = 60
GRAVITY = 32.174


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


def find_best_release_frame(positions, frames, vx, vy, vz, hoop_pos):
    """
    Find the frame that gives the best shot at reaching the hoop.

    The ball must have:
    1. Forward velocity (toward hoop)
    2. Upward velocity
    3. Enough total energy to reach the hoop

    We'll find the frame where the velocity DIRECTION best matches
    what's needed for the shot, weighted by speed.
    """
    best_score = -np.inf
    best_idx = None

    for i in range(len(frames)):
        pos = positions[i]
        vel = np.array([vx[i], vy[i], vz[i]])
        speed = np.linalg.norm(vel)

        if speed < 5.0:  # Too slow to be release
            continue

        # Direction to hoop
        to_hoop = hoop_pos - pos
        horiz_dist = np.sqrt(to_hoop[0]**2 + to_hoop[1]**2)
        height_diff = to_hoop[2]

        # Required direction for 50-degree shot
        # Horizontal component: toward hoop
        # Vertical component: upward at ~50 degrees

        # Check if moving toward hoop
        horiz_vel = np.sqrt(vx[i]**2 + vy[i]**2)
        vel_toward_hoop = -(vx[i] * to_hoop[0] + vy[i] * to_hoop[1]) / (horiz_dist + 1e-6)

        if vel_toward_hoop < 1.0:  # Must be moving toward hoop
            continue

        if vz[i] < 3.0:  # Must be moving upward
            continue

        # Score: how good is this velocity for reaching the hoop?
        # We want: forward velocity AND upward velocity
        score = vel_toward_hoop * 2 + vz[i]  # Weight forward more since it's what's missing

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def simulate_trajectory(pos, vel, target_height=10.0, max_time=2.0, dt=0.001):
    """Simulate projectile motion."""
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)

    t = 0
    max_height = pos[2]

    while t < max_time:
        vel[2] -= GRAVITY * dt
        pos += vel * dt
        t += dt

        max_height = max(max_height, pos[2])

        # Crossed target height going down
        if pos[2] <= target_height and vel[2] < 0 and t > 0.3:
            return {
                'success': True,
                'landing_pos': pos.copy(),
                'max_height': max_height,
                'time': t,
            }

        if pos[2] < 0:
            return {'success': False, 'reason': 'ground', 'max_height': max_height}

    return {'success': False, 'reason': 'timeout', 'max_height': max_height}


def analyze_shot(timeseries, keypoint_map, metadata, verbose=False):
    """Analyze a shot with multiple release detection methods."""

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

    # Velocity with different windows
    window = 7
    vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Method 1: Peak upward velocity (old method)
    peak_up_idx = np.argmax(vz)

    # Method 2: Peak total speed
    peak_speed_idx = np.argmax(speed)

    # Method 3: Peak forward velocity while moving up
    vel_toward_hoop = -vx  # negative X is toward hoop
    forward_while_up = np.where(vz > 2.0, vel_toward_hoop, -100)
    peak_forward_idx = np.argmax(forward_while_up)

    # Method 4: Best combined velocity for reaching hoop
    best_idx = find_best_release_frame(positions, frames, vx, vy, vz, HOOP_POSITION)

    results = {
        'shot_id': metadata['id'],
        'player_id': metadata['participant_id'],
        'true_angle': metadata['angle'],
        'true_depth': metadata['depth'],
        'true_lr': metadata['left_right'],
    }

    for method_name, idx in [
        ('peak_up', peak_up_idx),
        ('peak_speed', peak_speed_idx),
        ('peak_forward', peak_forward_idx),
        ('best_combined', best_idx),
    ]:
        if idx is None:
            continue

        pos = positions[idx]
        vel = np.array([vx[idx], vy[idx], vz[idx]])
        spd = speed[idx]

        # Simulate
        traj = simulate_trajectory(pos, vel)

        to_hoop = HOOP_POSITION - pos
        horiz_dist = np.sqrt(to_hoop[0]**2 + to_hoop[1]**2)

        results[f'{method_name}_frame'] = frames[idx]
        results[f'{method_name}_speed'] = spd
        results[f'{method_name}_vx'] = vx[idx]
        results[f'{method_name}_vz'] = vz[idx]
        results[f'{method_name}_success'] = traj['success']
        results[f'{method_name}_max_height'] = traj['max_height']

        if traj['success']:
            landing = traj['landing_pos']
            # Entry angle (degrees from horizontal)
            results[f'{method_name}_entry_angle'] = np.degrees(np.arctan2(abs(vel[2] - GRAVITY * traj['time']), np.sqrt(vel[0]**2 + vel[1]**2)))

            # Depth and left_right relative to hoop
            results[f'{method_name}_depth'] = (HOOP_POSITION[0] - landing[0]) * 12  # inches
            results[f'{method_name}_lr'] = (landing[1] - HOOP_POSITION[1]) * 12  # inches

        if verbose:
            print(f"  {method_name}: frame {frames[idx]}, speed {spd:.1f} ft/s, "
                  f"vx={vx[idx]:.1f}, vz={vz[idx]:.1f}, success={traj['success']}")

    return results


def main():
    print("=" * 80)
    print("COMPARING RELEASE DETECTION METHODS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    all_results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        result = analyze_shot(timeseries, keypoint_map, metadata, verbose=(i < 3))
        if result:
            all_results.append(result)

        if i >= 2:
            print("...", flush=True)
        if i >= 50:
            break

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    methods = ['peak_up', 'peak_speed', 'peak_forward', 'best_combined']

    for method in methods:
        success_col = f'{method}_success'
        if success_col not in df.columns:
            continue

        success_rate = df[success_col].mean() * 100

        print(f"\n{method}:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Avg speed: {df[f'{method}_speed'].mean():.1f} +/- {df[f'{method}_speed'].std():.1f} ft/s")
        print(f"  Avg vx: {df[f'{method}_vx'].mean():.1f} ft/s (need ~-15)")
        print(f"  Avg vz: {df[f'{method}_vz'].mean():.1f} ft/s (need ~19)")

        # Check correlations with targets for successful shots
        success_mask = df[success_col] == True
        if success_mask.sum() > 5:
            for target, pred_col in [('true_angle', f'{method}_entry_angle'),
                                     ('true_depth', f'{method}_depth'),
                                     ('true_lr', f'{method}_lr')]:
                if pred_col in df.columns:
                    valid = df[success_mask][[target, pred_col]].dropna()
                    if len(valid) > 5:
                        r, p = pearsonr(valid[target], valid[pred_col])
                        print(f"  {target.split('_')[1]} correlation: r={r:.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
