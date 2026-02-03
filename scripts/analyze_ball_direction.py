"""
Analyze the direction the ball should be traveling based on targets.

If targets (depth, left_right) represent deviations from hoop center,
we can infer:
1. The approximate hoop position
2. What the "correct" trajectory should look like
"""

import numpy as np
import pandas as pd
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


def main():
    print("=" * 80)
    print("BALL DIRECTION ANALYSIS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Collect release data and targets
    data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 100:
            break

        # Get hand positions
        hand_positions = []
        for frame in range(80, 180):
            pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
            if pos is not None:
                hand_positions.append(pos)

        if len(hand_positions) < 20:
            continue

        hand_positions = np.array(hand_positions)

        # Compute velocities
        vel = np.zeros_like(hand_positions)
        for j in range(3):
            vel[:, j] = savgol_filter(hand_positions[:, j], 11, 3, deriv=1) * FPS

        # Find peak upward velocity (likely release point)
        vz = vel[:, 2]
        peak_idx = np.argmax(vz)

        if peak_idx < 5 or peak_idx >= len(vel) - 5:
            continue

        release_pos = hand_positions[peak_idx]
        release_vel = vel[peak_idx]

        data.append({
            'id': metadata['id'],
            'release_x': release_pos[0],
            'release_y': release_pos[1],
            'release_z': release_pos[2],
            'vx': release_vel[0],
            'vy': release_vel[1],
            'vz': release_vel[2],
            'angle': metadata.get('angle'),
            'depth': metadata.get('depth'),
            'left_right': metadata.get('left_right'),
        })

    df = pd.DataFrame(data)
    print(f"\nAnalyzed {len(df)} shots")

    print("\n" + "=" * 60)
    print("RELEASE POSITION (in feet)")
    print("=" * 60)
    print(f"X: {df['release_x'].mean():.2f} +/- {df['release_x'].std():.2f}")
    print(f"Y: {df['release_y'].mean():.2f} +/- {df['release_y'].std():.2f}")
    print(f"Z: {df['release_z'].mean():.2f} +/- {df['release_z'].std():.2f}")

    print("\n" + "=" * 60)
    print("RELEASE VELOCITY (in feet/second)")
    print("=" * 60)
    print(f"Vx: {df['vx'].mean():.2f} +/- {df['vx'].std():.2f}")
    print(f"Vy: {df['vy'].mean():.2f} +/- {df['vy'].std():.2f}")
    print(f"Vz: {df['vz'].mean():.2f} +/- {df['vz'].std():.2f}")

    horizontal_speed = np.sqrt(df['vx']**2 + df['vy']**2)
    print(f"\nHorizontal speed: {horizontal_speed.mean():.2f} +/- {horizontal_speed.std():.2f}")
    print(f"Vertical speed:   {df['vz'].mean():.2f} +/- {df['vz'].std():.2f}")

    total_speed = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    print(f"Total speed:      {total_speed.mean():.2f} +/- {total_speed.std():.2f}")

    print("\n" + "=" * 60)
    print("TARGETS")
    print("=" * 60)
    print(f"angle:      {df['angle'].mean():.2f} +/- {df['angle'].std():.2f} degrees")
    print(f"depth:      {df['depth'].mean():.2f} +/- {df['depth'].std():.2f} inches")
    print(f"left_right: {df['left_right'].mean():.2f} +/- {df['left_right'].std():.2f} inches")

    print("\n" + "=" * 60)
    print("CORRELATION: VELOCITY vs TARGETS")
    print("=" * 60)

    print(f"\n{'Velocity':<20} {'angle':>10} {'depth':>10} {'left_right':>12}")
    print("-" * 55)

    for vel_col in ['vx', 'vy', 'vz']:
        corrs = []
        for target in ['angle', 'depth', 'left_right']:
            corr = df[vel_col].corr(df[target])
            corrs.append(corr)
        print(f"{vel_col:<20} {corrs[0]:>10.3f} {corrs[1]:>10.3f} {corrs[2]:>12.3f}")

    # Horizontal speed
    print(f"{'horizontal_speed':<20} {horizontal_speed.corr(df['angle']):>10.3f} "
          f"{horizontal_speed.corr(df['depth']):>10.3f} "
          f"{horizontal_speed.corr(df['left_right']):>12.3f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: What velocity components correlate with targets?")
    print("=" * 60)

    # The targets should relate to where the ball lands
    # - angle: launch angle (higher = more arc)
    # - depth: forward/back deviation
    # - left_right: lateral deviation

    vz_angle_corr = df['vz'].corr(df['angle'])
    print(f"\nVz (vertical velocity) correlates {vz_angle_corr:.3f} with angle")
    print("  => Higher upward velocity = higher angle (makes sense!)")

    vx_lr_corr = df['vx'].corr(df['left_right'])
    vy_lr_corr = df['vy'].corr(df['left_right'])
    print(f"\nVx correlates {vx_lr_corr:.3f} with left_right")
    print(f"Vy correlates {vy_lr_corr:.3f} with left_right")

    if abs(vx_lr_corr) > abs(vy_lr_corr):
        print("  => X-axis is the lateral direction!")
        print("  => Hoop is likely in the Y direction from player")
    else:
        print("  => Y-axis is the lateral direction!")
        print("  => Hoop is likely in the X direction from player")

    vx_depth_corr = df['vx'].corr(df['depth'])
    vy_depth_corr = df['vy'].corr(df['depth'])
    print(f"\nVx correlates {vx_depth_corr:.3f} with depth")
    print(f"Vy correlates {vy_depth_corr:.3f} with depth")

    if abs(vx_depth_corr) > abs(vy_depth_corr):
        print("  => X-axis is the depth (forward) direction")
    else:
        print("  => Y-axis is the depth (forward) direction")

    print("\n" + "=" * 60)
    print("PHYSICAL REQUIREMENTS FOR FREE THROW")
    print("=" * 60)

    # Free throw: 15 feet horizontal, ~2 feet up from release to hoop
    # With ~1 second flight time:
    # - Horizontal velocity: 15 ft/s
    # - Vertical velocity (initial): ~15-20 ft/s

    print("\nRequired for realistic trajectory:")
    print("  Horizontal: ~15-20 ft/s")
    print("  Vertical:   ~15-20 ft/s")

    print(f"\nObserved in data:")
    print(f"  Horizontal: {horizontal_speed.mean():.1f} ft/s (missing ~{15 - horizontal_speed.mean():.0f} ft/s)")
    print(f"  Vertical:   {df['vz'].mean():.1f} ft/s")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The skeleton data shows:
1. Strong vertical velocity (~10 ft/s upward) - good!
2. Minimal horizontal velocity (~1-2 ft/s) - PROBLEM!

Possible explanations:
1. The player is shooting almost straight up (unlikely for free throw)
2. The horizontal release motion (wrist flick toward hoop) is not captured
3. There's a coordinate system issue

The MuJoCo simulation correctly models the physics, but the INPUT
motion data lacks the horizontal velocity component toward the hoop.

To predict targets accurately, we may need to:
1. Infer the horizontal velocity from other cues (body angle, etc.)
2. Or use the RELATIVE velocities to predict RELATIVE outcomes
   (not absolute trajectory, but deviations from player's typical shot)
""")


if __name__ == "__main__":
    main()
