"""
Analyze the coordinate system to understand:
1. Where is the player positioned?
2. What direction is the player facing?
3. Where should the hoop be?
4. What direction should the ball travel?
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


def main():
    print("=" * 80)
    print("COORDINATE SYSTEM ANALYSIS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Collect data from multiple shots
    all_positions = {
        'pelvis': [],
        'right_shoulder': [],
        'right_wrist': [],
        'right_third_finger_distal': [],
    }

    all_velocities = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 20:
            break

        # Get positions at key frames
        for frame in [100, 120, 140, 160]:
            for kp_name in all_positions.keys():
                pos = get_position(timeseries, keypoint_map, kp_name, frame)
                if pos is not None:
                    all_positions[kp_name].append(pos)

        # Compute release velocity direction
        hand_positions = []
        for frame in range(80, 180):
            pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
            if pos is not None:
                hand_positions.append(pos)

        if len(hand_positions) > 20:
            hand_positions = np.array(hand_positions)
            # Velocity using savgol
            vel = np.zeros_like(hand_positions)
            for j in range(3):
                vel[:, j] = savgol_filter(hand_positions[:, j], 11, 3, deriv=1) * FPS

            # Find peak upward velocity
            vz = vel[:, 2]
            peak_idx = np.argmax(vz)

            if peak_idx > 5 and peak_idx < len(vel) - 5:
                release_vel = vel[peak_idx]
                all_velocities.append(release_vel)

    print("\n" + "=" * 60)
    print("PLAYER POSITION (in feet)")
    print("=" * 60)

    for kp_name, positions in all_positions.items():
        if positions:
            positions = np.array(positions)
            mean_pos = np.mean(positions, axis=0)
            print(f"\n{kp_name}:")
            print(f"  X: {mean_pos[0]:.2f} feet (range: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f})")
            print(f"  Y: {mean_pos[1]:.2f} feet (range: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f})")
            print(f"  Z: {mean_pos[2]:.2f} feet (range: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f})")

    print("\n" + "=" * 60)
    print("RELEASE VELOCITY DIRECTION (in feet/sec)")
    print("=" * 60)

    if all_velocities:
        all_velocities = np.array(all_velocities)
        mean_vel = np.mean(all_velocities, axis=0)
        print(f"\nMean release velocity:")
        print(f"  Vx: {mean_vel[0]:.2f} ft/s")
        print(f"  Vy: {mean_vel[1]:.2f} ft/s")
        print(f"  Vz: {mean_vel[2]:.2f} ft/s (upward)")

        # Which direction is the ball going?
        print(f"\nVelocity direction analysis:")
        print(f"  Horizontal speed (sqrt(vx^2+vy^2)): {np.sqrt(mean_vel[0]**2 + mean_vel[1]**2):.2f} ft/s")
        print(f"  Vertical speed (vz): {mean_vel[2]:.2f} ft/s")

        if abs(mean_vel[0]) > abs(mean_vel[1]):
            print(f"  Primary horizontal direction: {'positive X' if mean_vel[0] > 0 else 'negative X'}")
        else:
            print(f"  Primary horizontal direction: {'positive Y' if mean_vel[1] > 0 else 'negative Y'}")

    print("\n" + "=" * 60)
    print("COORDINATE SYSTEM INTERPRETATION")
    print("=" * 60)

    # Free throw setup:
    # - Player stands at free throw line (15 feet from basket)
    # - Basket rim is 10 feet high
    # - Player faces the basket

    pelvis_pos = np.mean(all_positions['pelvis'], axis=0) if all_positions['pelvis'] else None

    if pelvis_pos is not None:
        print(f"\nPlayer pelvis at: ({pelvis_pos[0]:.2f}, {pelvis_pos[1]:.2f}, {pelvis_pos[2]:.2f}) feet")

        # Assuming standard court setup:
        # If player is at free throw line, basket is 15 feet away horizontally
        print("\nPossible hoop locations (assuming 15 feet from player):")

        # Option 1: Hoop is in negative X direction
        print(f"  If hoop is in -X direction: ({pelvis_pos[0] - 15:.2f}, {pelvis_pos[1]:.2f}, 10.00) feet")

        # Option 2: Hoop is in positive X direction
        print(f"  If hoop is in +X direction: ({pelvis_pos[0] + 15:.2f}, {pelvis_pos[1]:.2f}, 10.00) feet")

        # Option 3: Hoop is in negative Y direction
        print(f"  If hoop is in -Y direction: ({pelvis_pos[0]:.2f}, {pelvis_pos[1] - 15:.2f}, 10.00) feet")

        # Option 4: Hoop is in positive Y direction
        print(f"  If hoop is in +Y direction: ({pelvis_pos[0]:.2f}, {pelvis_pos[1] + 15:.2f}, 10.00) feet")

    print("\n" + "=" * 60)
    print("VELOCITY DIRECTION vs HOOP")
    print("=" * 60)

    if all_velocities and pelvis_pos is not None:
        mean_vel = np.mean(all_velocities, axis=0)

        # The ball should be going TOWARD the hoop
        # If Vx is positive, ball is going in +X direction
        # If Vy is positive, ball is going in +Y direction

        print(f"\nBall is moving:")
        if mean_vel[0] > 0.5:
            print(f"  TOWARD +X direction ({mean_vel[0]:.2f} ft/s)")
            print(f"  => Hoop is likely in +X direction from player")
            print(f"  => Hoop X = {pelvis_pos[0] + 15:.2f} feet")
        elif mean_vel[0] < -0.5:
            print(f"  TOWARD -X direction ({mean_vel[0]:.2f} ft/s)")
            print(f"  => Hoop is likely in -X direction from player")
            print(f"  => Hoop X = {pelvis_pos[0] - 15:.2f} feet")
        else:
            print(f"  Minimal X velocity ({mean_vel[0]:.2f} ft/s)")

        if mean_vel[1] > 0.5:
            print(f"  TOWARD +Y direction ({mean_vel[1]:.2f} ft/s)")
            print(f"  => Hoop is likely in +Y direction from player")
            print(f"  => Hoop Y = {pelvis_pos[1] + 15:.2f} feet")
        elif mean_vel[1] < -0.5:
            print(f"  TOWARD -Y direction ({mean_vel[1]:.2f} ft/s)")
            print(f"  => Hoop is likely in -Y direction from player")
            print(f"  => Hoop Y = {pelvis_pos[1] - 15:.2f} feet")
        else:
            print(f"  Minimal Y velocity ({mean_vel[1]:.2f} ft/s)")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("\nTo fix trajectory prediction, need to determine:")
    print("1. Exact hoop position in data coordinates")
    print("2. Scale factor if data is not in actual feet")
    print("3. Whether coordinate axes are swapped")


if __name__ == "__main__":
    main()
