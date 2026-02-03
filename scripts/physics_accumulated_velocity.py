"""
Accumulated velocity approach.

The ball doesn't just have the instantaneous fingertip velocity at release.
It has ACCUMULATED momentum from the entire shooting motion:
1. Arm swing forward (adds horizontal velocity)
2. Elbow extension (adds more horizontal + vertical)
3. Wrist flick (adds vertical velocity)
4. Finger snap (adds final push)

We should integrate the ball's acceleration over the motion, not just
take the instantaneous velocity at release.

The ball acceleration = hand acceleration (while in contact)
Ball velocity at release = integral of all accelerations

This is different from taking the derivative at one point because
the hand ROTATES during the shot. The forward velocity from the arm swing
gets "hidden" when the wrist rotates upward.
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
DT = 1.0 / FPS


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


def find_shot_window(positions, frames, vz):
    """
    Find the shooting window: when the ball is being pushed upward.

    Start: when upward velocity becomes significant
    End: when vertical acceleration goes negative (release)
    """
    # Find when vz first exceeds threshold (shot starting)
    start_idx = None
    for i in range(len(vz)):
        if vz[i] > 3.0:  # Significant upward motion
            start_idx = max(0, i - 5)  # Go back a bit to capture run-up
            break

    if start_idx is None:
        start_idx = 0

    # Compute acceleration
    window = 7
    az = savgol_filter(positions[:, 2], window, 3, deriv=2) * FPS * FPS

    # Find when az goes negative while vz still positive (release)
    end_idx = None
    for i in range(start_idx + 10, len(az)):
        if az[i] < 0 and vz[i] > 2.0:
            end_idx = i
            break

    if end_idx is None:
        # Use peak vz
        end_idx = np.argmax(vz)

    return start_idx, end_idx


def simulate_ball_with_hand(positions, frames, start_idx, end_idx):
    """
    Simulate the ball being pushed by the hand.

    Instead of just taking velocity at one frame, we:
    1. Start the ball at rest (or with the hand's initial velocity)
    2. At each frame, the hand exerts force on the ball
    3. Accumulate the velocity from all the pushes

    The key insight: the hand's ACCELERATION adds to the ball's velocity,
    not the hand's instantaneous velocity.
    """

    # Compute hand velocity and acceleration
    window = 7
    vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS

    ax = savgol_filter(positions[:, 0], window, 3, deriv=2) * FPS * FPS
    ay = savgol_filter(positions[:, 1], window, 3, deriv=2) * FPS * FPS
    az = savgol_filter(positions[:, 2], window, 3, deriv=2) * FPS * FPS

    # Method 1: Just use instantaneous velocity at release (baseline)
    vel_instant = np.array([vx[end_idx], vy[end_idx], vz[end_idx]])

    # Method 2: Accumulate only POSITIVE accelerations
    # The hand pushes the ball (positive acceleration toward hoop and up)
    # Negative accelerations (deceleration) mean the hand is stopping, not the ball

    ball_vel = np.array([0.0, 0.0, 0.0])

    # Start with initial hand velocity
    ball_vel = np.array([vx[start_idx], vy[start_idx], vz[start_idx]])

    for i in range(start_idx + 1, end_idx + 1):
        # Hand acceleration at this frame
        hand_acc = np.array([ax[i], ay[i], az[i]])

        # The hand can only PUSH the ball, not pull it
        # So we only add positive accelerations in the direction of motion

        # In Z (vertical): positive acceleration pushes up
        if hand_acc[2] > 0:
            ball_vel[2] += hand_acc[2] * DT

        # In X (toward hoop, negative direction): negative ax pushes ball toward hoop
        if hand_acc[0] < 0:  # Accelerating toward hoop (negative X)
            ball_vel[0] += hand_acc[0] * DT

        # In Y (lateral): any direction
        ball_vel[1] += hand_acc[1] * DT

    # Method 3: Track when each component's velocity is maximum
    # The ball "remembers" its peak velocity in each direction

    peak_vx = np.min(vx[start_idx:end_idx+1])  # Most negative = most toward hoop
    peak_vy = vy[end_idx]  # Just use final lateral
    peak_vz = np.max(vz[start_idx:end_idx+1])  # Most positive = most upward

    vel_peak = np.array([peak_vx, peak_vy, peak_vz])

    return {
        'vel_instant': vel_instant,
        'vel_accumulated': ball_vel,
        'vel_peak': vel_peak,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_frame': frames[start_idx],
        'end_frame': frames[end_idx],
    }


def simulate_trajectory(pos, vel, target_height=10.0, max_time=2.0, dt=0.001):
    """Simulate projectile motion."""
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)

    t = 0
    while t < max_time:
        vel[2] -= GRAVITY * dt
        pos += vel * dt
        t += dt

        if pos[2] <= target_height and vel[2] < 0 and t > 0.3:
            return {'success': True, 'landing_pos': pos.copy(), 'time': t}

        if pos[2] < 0:
            return {'success': False, 'reason': 'ground'}

    return {'success': False, 'reason': 'timeout'}


def analyze_shot(timeseries, keypoint_map, metadata, verbose=False):
    """Analyze a shot with accumulated velocity."""

    frames = []
    positions = []

    for frame in range(50, 200):
        pos = calculate_ball_position(timeseries, keypoint_map, frame)
        if pos is not None:
            frames.append(frame)
            positions.append(pos)

    if len(positions) < 30:
        return None

    frames = np.array(frames)
    positions = np.array(positions)

    window = 7
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS

    start_idx, end_idx = find_shot_window(positions, frames, vz)

    sim = simulate_ball_with_hand(positions, frames, start_idx, end_idx)

    if verbose:
        print(f"\nShot {metadata['id'][:8]}:")
        print(f"  Window: frames {sim['start_frame']} to {sim['end_frame']}")
        print(f"  Instant velocity: [{sim['vel_instant'][0]:.1f}, {sim['vel_instant'][1]:.1f}, {sim['vel_instant'][2]:.1f}]")
        print(f"  Accumulated velocity: [{sim['vel_accumulated'][0]:.1f}, {sim['vel_accumulated'][1]:.1f}, {sim['vel_accumulated'][2]:.1f}]")
        print(f"  Peak velocity: [{sim['vel_peak'][0]:.1f}, {sim['vel_peak'][1]:.1f}, {sim['vel_peak'][2]:.1f}]")

    result = {
        'shot_id': metadata['id'],
        'true_angle': metadata['angle'],
        'true_depth': metadata['depth'],
        'true_lr': metadata['left_right'],
    }

    release_pos = positions[end_idx]

    for method, vel in [('instant', sim['vel_instant']),
                        ('accumulated', sim['vel_accumulated']),
                        ('peak', sim['vel_peak'])]:
        speed = np.linalg.norm(vel)
        result[f'{method}_speed'] = speed
        result[f'{method}_vx'] = vel[0]
        result[f'{method}_vz'] = vel[2]

        # Simulate trajectory
        traj = simulate_trajectory(release_pos, vel)
        result[f'{method}_success'] = traj['success']

        if traj['success']:
            landing = traj['landing_pos']
            # Calculate targets
            # Entry angle at crossing
            t = traj['time']
            vel_at_landing = vel.copy()
            vel_at_landing[2] -= GRAVITY * t
            horiz_speed = np.sqrt(vel_at_landing[0]**2 + vel_at_landing[1]**2)
            if horiz_speed > 0.1:
                entry_angle = np.degrees(np.arctan2(abs(vel_at_landing[2]), horiz_speed))
            else:
                entry_angle = 90.0

            result[f'{method}_entry_angle'] = entry_angle
            result[f'{method}_depth'] = (HOOP_POSITION[0] - landing[0]) * 12
            result[f'{method}_lr'] = (landing[1] - HOOP_POSITION[1]) * 12

    return result


def main():
    print("=" * 80)
    print("ACCUMULATED VELOCITY APPROACH")
    print("=" * 80)
    print("\nIdea: ball accumulates momentum throughout shot, not just at release")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    all_results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        result = analyze_shot(timeseries, keypoint_map, metadata, verbose=(i < 3))
        if result:
            all_results.append(result)

        if i >= 2:
            print(".", end="", flush=True)
        if i >= 100:
            break

    print()

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    methods = ['instant', 'accumulated', 'peak']

    for method in methods:
        print(f"\n{method.upper()}:")
        print(f"  Speed: {df[f'{method}_speed'].mean():.1f} +/- {df[f'{method}_speed'].std():.1f} ft/s")
        print(f"  vx: {df[f'{method}_vx'].mean():.1f} ft/s (need ~-15)")
        print(f"  vz: {df[f'{method}_vz'].mean():.1f} ft/s (need ~19)")

        success_col = f'{method}_success'
        if success_col in df.columns:
            print(f"  Success rate: {df[success_col].mean() * 100:.1f}%")

            success_mask = df[success_col] == True
            if success_mask.sum() > 10:
                print("  Correlations with targets:")
                for target in ['angle', 'depth', 'lr']:
                    pred_col = f'{method}_{"entry_angle" if target == "angle" else target}'
                    true_col = f'true_{target}'
                    if pred_col in df.columns:
                        valid = df[success_mask][[true_col, pred_col]].dropna()
                        if len(valid) > 5:
                            r, p = pearsonr(valid[true_col], valid[pred_col])
                            sig = "*" if p < 0.05 else ""
                            print(f"    {target}: r={r:.3f} (p={p:.3f}) {sig}")

    # Show the best approach
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nThe 'peak' method uses maximum velocity in each direction separately.")
    print("This approximates the ball 'remembering' its peak momentum in each axis.")


if __name__ == "__main__":
    main()
