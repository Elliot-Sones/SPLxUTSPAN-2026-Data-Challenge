"""
Simulate ball in contact with hand to detect EXACT release moment.

Physics model:
1. Ball sits on fingertips, following hand position
2. Hand accelerates ball through contact force
3. Release occurs when:
   - Hand starts decelerating (can't push ball anymore)
   - The required contact force would need to be negative (pulling)
   - Ball separates and follows projectile motion

At release, ball velocity = hand velocity at that instant.
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
BALL_MASS = 0.625  # kg = 1.38 lb
FPS = 60
GRAVITY = 32.174  # ft/s^2 (in feet)
GRAVITY_M = 9.81  # m/s^2

# Convert gravity to feet: 9.81 m/s^2 * 3.281 ft/m = 32.17 ft/s^2


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

    # Ball center is slightly above fingertips (ball rests on fingers)
    wrist = get_position(timeseries, keypoint_map, 'right_wrist', frame)
    if wrist is not None:
        palm_dir = centroid - wrist
        palm_dir = palm_dir / (np.linalg.norm(palm_dir) + 1e-6)
        ball_center = centroid + palm_dir * BALL_RADIUS * 0.5
    else:
        ball_center = centroid + np.array([0, 0, BALL_RADIUS * 0.5])

    return ball_center


def simulate_ball_contact(timeseries, keypoint_map, verbose=False):
    """
    Simulate ball following hand, detect release via physics.

    The ball is in contact with the hand. At each frame:
    1. Ball position = hand position (follows hand)
    2. We compute the CONTACT FORCE required to keep ball with hand
    3. Contact force = m * (a_hand + g)  where a_hand is hand acceleration
    4. Release when contact force would be negative (can't pull the ball)

    At release:
    - Ball velocity = hand velocity at that frame
    - Ball then follows projectile motion
    """

    # Collect hand (ball) positions
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

    # Compute velocity (1st derivative)
    # Use smaller window for more accurate peak detection
    window = 7
    vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS

    # Compute acceleration (2nd derivative)
    ax = savgol_filter(positions[:, 0], window, 3, deriv=2) * FPS * FPS
    ay = savgol_filter(positions[:, 1], window, 3, deriv=2) * FPS * FPS
    az = savgol_filter(positions[:, 2], window, 3, deriv=2) * FPS * FPS

    # Contact force in Z direction (vertical)
    # F_contact = m * (a_z + g)
    # - When hand accelerates up faster than gravity, F_contact > 0 (pushing)
    # - When hand decelerates (a_z < -g), F_contact < 0 (would need to pull)

    # The ball releases when the hand can no longer push it up
    # This is when a_z drops below -g (hand decelerating faster than gravity)

    # Actually, for a ball sitting on the hand:
    # - If hand is stationary, contact force = m*g (just supporting weight)
    # - If hand accelerates up, contact force = m*(g + a_z) > m*g
    # - If hand accelerates down (or decelerates from upward motion), contact force decreases
    # - Release when contact force would be 0: a_z = -g

    # However, the hand direction also matters. The contact force is along the
    # hand normal (palm direction). Let's simplify: assume ball releases when
    # vertical acceleration is most negative (hand decelerating fastest).

    # Find candidate release frames:
    # 1. Must be moving upward (vz > 0)
    # 2. Vertical acceleration is becoming negative (decelerating)

    if verbose:
        print(f"\nFrame-by-frame analysis:")
        print(f"  Frame    Z      vz      az    Contact Force")

    release_frame = None
    release_idx = None

    # Look for release: hand moving up but decelerating
    for i in range(5, len(frames) - 5):
        # Check if ball is ascending
        if vz[i] < 2.0:  # Must be moving up significantly
            continue

        # Check if hand is decelerating (az < 0)
        # The exact threshold depends on many factors, but let's look for
        # where az goes from positive to negative (inflection point)

        # Contact force per unit mass: f = az + g
        # In our units (feet), g = 32.17 ft/s^2
        contact_force_per_mass = az[i] + GRAVITY

        if verbose and 100 <= frames[i] <= 130:
            print(f"  {frames[i]:5d} {positions[i,2]:6.2f} {vz[i]:7.2f} {az[i]:7.1f} {contact_force_per_mass:8.1f}")

        # Release criterion: contact force approaches zero
        # This means az approaches -g
        # We look for where contact force first drops significantly
        if i > 10:
            # Check if contact force is dropping and getting low
            prev_cf = az[i-3] + GRAVITY
            curr_cf = az[i] + GRAVITY

            # Release when:
            # 1. Contact force is positive but decreasing
            # 2. Ball is high enough (Z > 5 feet = above head level)
            # 3. Ball is moving up
            if positions[i, 2] > 5.0 and vz[i] > 3.0:
                # Look for the inflection point where acceleration peaks
                # Actually, let's look for where the hand starts decelerating significantly
                if az[i] < 0 and az[i-1] >= 0:
                    release_frame = frames[i]
                    release_idx = i
                    break

    # If no release found, use peak upward velocity
    if release_frame is None:
        peak_vz_idx = np.argmax(vz)
        # Go back a few frames to when hand still had forward momentum
        release_idx = max(0, peak_vz_idx - 3)
        release_frame = frames[release_idx]

    # Get ball state at release
    release_pos = positions[release_idx]
    release_vel = np.array([vx[release_idx], vy[release_idx], vz[release_idx]])
    release_speed = np.linalg.norm(release_vel)

    return {
        'release_frame': release_frame,
        'release_pos': release_pos,
        'release_vel': release_vel,
        'release_speed': release_speed,
        'all_frames': frames,
        'all_positions': positions,
        'all_vx': vx,
        'all_vy': vy,
        'all_vz': vz,
        'all_ax': ax,
        'all_ay': ay,
        'all_az': az,
    }


def simulate_trajectory(pos, vel, max_time=2.0, dt=0.001):
    """
    Simulate ball trajectory after release.
    """
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)

    t = 0
    while t < max_time:
        # Gravity
        vel[2] -= GRAVITY * dt
        pos += vel * dt
        t += dt

        # Check if ball crosses rim plane (Z = 10)
        if pos[2] >= 10.0 and vel[2] < 0:  # Descending through rim height
            # Linear interpolate to exact crossing
            return {
                'success': True,
                'landing_pos': pos.copy(),
                'time': t,
            }

        # Check if fell below ground
        if pos[2] < 0:
            return {'success': False, 'reason': 'ground'}

    return {'success': False, 'reason': 'timeout'}


def main():
    print("=" * 80)
    print("BALL CONTACT SIMULATION")
    print("=" * 80)
    print("\nSimulating ball following hand to detect exact release moment")
    print("Release occurs when hand decelerates (contact force drops)")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 10:
            break

        sim = simulate_ball_contact(timeseries, keypoint_map, verbose=(i==0))

        if sim is None:
            continue

        print(f"\nShot {metadata['id'][:8]} (Player {metadata['participant_id']}):")
        print(f"  Release frame: {sim['release_frame']}")
        print(f"  Release pos: [{sim['release_pos'][0]:.2f}, {sim['release_pos'][1]:.2f}, {sim['release_pos'][2]:.2f}] ft")
        print(f"  Release vel: [{sim['release_vel'][0]:.2f}, {sim['release_vel'][1]:.2f}, {sim['release_vel'][2]:.2f}] ft/s")
        print(f"  Release speed: {sim['release_speed']:.2f} ft/s")

        # Calculate what's needed to reach hoop
        to_hoop = HOOP_POSITION - sim['release_pos']
        horiz_dist = np.sqrt(to_hoop[0]**2 + to_hoop[1]**2)
        height_diff = to_hoop[2]

        print(f"  Distance to hoop: {horiz_dist:.1f} ft horizontal, {height_diff:.1f} ft up")

        # Compute required velocity for 50 degree angle
        theta = np.radians(50)
        denom = horiz_dist * np.tan(theta) - height_diff
        if denom > 0:
            v_req = np.sqrt(GRAVITY * horiz_dist**2 / (2 * np.cos(theta)**2 * denom))
            print(f"  Required speed (50 deg): {v_req:.1f} ft/s")
            print(f"  Measured / Required: {sim['release_speed']/v_req*100:.0f}%")

        # Simulate trajectory with measured velocity
        traj = simulate_trajectory(sim['release_pos'], sim['release_vel'])
        if traj['success']:
            landing = traj['landing_pos']
            dx = landing[0] - HOOP_POSITION[0]
            dy = landing[1] - HOOP_POSITION[1]
            print(f"  Trajectory lands at: X offset = {dx:.1f} ft, Y offset = {dy:.1f} ft from hoop")
        else:
            print(f"  Trajectory: {traj['reason']}")

        results.append({
            'shot_id': metadata['id'],
            'release_speed': sim['release_speed'],
            'vx': sim['release_vel'][0],
            'vy': sim['release_vel'][1],
            'vz': sim['release_vel'][2],
        })

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nRelease speeds: {df['release_speed'].mean():.1f} +/- {df['release_speed'].std():.1f} ft/s")
        print(f"  vx (toward hoop): {df['vx'].mean():.1f} +/- {df['vx'].std():.1f} ft/s")
        print(f"  vy (lateral): {df['vy'].mean():.1f} +/- {df['vy'].std():.1f} ft/s")
        print(f"  vz (up): {df['vz'].mean():.1f} +/- {df['vz'].std():.1f} ft/s")

        print(f"\nRequired for free throw:")
        print(f"  Total: ~24 ft/s")
        print(f"  Horizontal (toward hoop): ~15 ft/s")
        print(f"  Vertical (up): ~19 ft/s")

        print(f"\nDeficit:")
        print(f"  Horizontal: {-df['vx'].mean():.1f} vs 15 = {-df['vx'].mean()/15*100:.0f}%")
        print(f"  Vertical: {df['vz'].mean():.1f} vs 19 = {df['vz'].mean()/19*100:.0f}%")


if __name__ == "__main__":
    main()
