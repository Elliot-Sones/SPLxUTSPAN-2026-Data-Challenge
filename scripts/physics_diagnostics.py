"""
Physics Simulation Diagnostics

Investigates why 87.5% of simulations fail to reach the hoop.
Analyzes:
1. Scale factor calibration
2. Release velocities being extracted
3. Coordinate transformations
4. What would be needed to reach the hoop
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "physics_engine"))

from data_loader import iterate_shots, get_keypoint_columns
from core import (
    BasketballSimulator,
    calibrate_scale_factor,
    get_keypoint_indices,
    extract_all_release_params,
    PLAYER_RELEASE_WINDOWS
)
from core.scale_calibration import get_joint_position, validate_scale


def analyze_physics_requirements():
    """
    Calculate what release parameters are physically required to reach the hoop.
    """
    print("=" * 80)
    print("PART 1: PHYSICS REQUIREMENTS FOR A FREE THROW")
    print("=" * 80)

    # Free throw geometry
    distance = 4.57  # meters (15 feet)
    hoop_height = 3.05  # meters (10 feet)
    release_height = 2.1  # meters (typical release height)

    # For projectile motion: y = y0 + vy*t - 0.5*g*t^2, x = x0 + vx*t
    # At hoop: x = 4.57, y = 3.05
    # Time to reach hoop: t = 4.57 / vx
    # Height at hoop: 3.05 = 2.1 + vy*(4.57/vx) - 0.5*9.81*(4.57/vx)^2

    print("\nFree throw geometry:")
    print(f"  Distance to hoop: {distance} m")
    print(f"  Hoop height: {hoop_height} m")
    print(f"  Typical release height: {release_height} m")
    print(f"  Height gain needed: {hoop_height - release_height} m")

    print("\nRequired velocities for different release angles:")
    print("-" * 60)
    print(f"{'Angle (deg)':<12} {'Vx (m/s)':<12} {'Vz (m/s)':<12} {'Speed (m/s)':<12} {'Time (s)':<12}")
    print("-" * 60)

    g = 9.81
    for angle_deg in [30, 35, 40, 45, 50, 55, 60]:
        angle_rad = np.radians(angle_deg)

        # Solve for initial speed
        # Using projectile equations:
        # x = v*cos(theta)*t
        # y = y0 + v*sin(theta)*t - 0.5*g*t^2
        # t = x / (v*cos(theta))
        # y = y0 + x*tan(theta) - 0.5*g*x^2 / (v^2*cos^2(theta))
        # Solving for v:
        # v^2 = g*x^2 / (2*cos^2(theta) * (x*tan(theta) - (y - y0)))

        delta_y = hoop_height - release_height
        x = distance

        denominator = x * np.tan(angle_rad) - delta_y
        if denominator <= 0:
            continue

        v_squared = (g * x**2) / (2 * np.cos(angle_rad)**2 * denominator)
        if v_squared <= 0:
            continue

        v = np.sqrt(v_squared)
        vx = v * np.cos(angle_rad)
        vz = v * np.sin(angle_rad)
        t = x / vx

        print(f"{angle_deg:<12} {vx:<12.2f} {vz:<12.2f} {v:<12.2f} {t:<12.2f}")

    print("-" * 60)
    print("\nKey insight: Need vx ~ 5-7 m/s, vz ~ 4-7 m/s, total speed ~ 7-9 m/s")


def analyze_extracted_velocities():
    """
    Analyze what velocities are being extracted from the skeleton data.
    """
    print("\n" + "=" * 80)
    print("PART 2: EXTRACTED RELEASE VELOCITIES FROM SKELETON DATA")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_idx = get_keypoint_indices(keypoint_cols)

    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 50:  # Analyze first 50 shots
            break

        player_id = metadata['participant_id']

        # Get scale factor
        scale_factor = calibrate_scale_factor(timeseries, keypoint_idx)

        # Extract release params
        params = extract_all_release_params(timeseries, keypoint_idx, player_id, scale_factor)

        # Raw velocity (before MuJoCo coordinate transform)
        vel = params['velocity']
        speed = np.linalg.norm(vel)

        results.append({
            'shot_id': metadata['id'],
            'player_id': player_id,
            'scale_factor': scale_factor,
            'release_frame': params['release_frame'],
            'vx': vel[0],
            'vy': vel[1],
            'vz': vel[2],
            'speed': speed,
            'pos_x': params['position'][0],
            'pos_y': params['position'][1],
            'pos_z': params['position'][2],
            'backspin': params['backspin'],
        })

    df = pd.DataFrame(results)

    print("\nExtracted velocity statistics:")
    print("-" * 60)
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 60)
    for col in ['vx', 'vy', 'vz', 'speed', 'scale_factor', 'pos_z']:
        print(f"{col:<20} {df[col].mean():<12.3f} {df[col].std():<12.3f} {df[col].min():<12.3f} {df[col].max():<12.3f}")
    print("-" * 60)

    print("\nPer-player velocity breakdown:")
    print("-" * 60)
    for pid in sorted(df['player_id'].unique()):
        player_df = df[df['player_id'] == pid]
        print(f"\nPlayer {pid}:")
        print(f"  n_shots: {len(player_df)}")
        print(f"  vx: {player_df['vx'].mean():.2f} +/- {player_df['vx'].std():.2f}")
        print(f"  vy: {player_df['vy'].mean():.2f} +/- {player_df['vy'].std():.2f}")
        print(f"  vz: {player_df['vz'].mean():.2f} +/- {player_df['vz'].std():.2f}")
        print(f"  speed: {player_df['speed'].mean():.2f} +/- {player_df['speed'].std():.2f}")
        print(f"  scale: {player_df['scale_factor'].mean():.3f}")
        print(f"  release_height (pos_z): {player_df['pos_z'].mean():.2f}")

    return df


def analyze_raw_wrist_motion():
    """
    Look at the raw wrist motion in the data to understand coordinate system.
    """
    print("\n" + "=" * 80)
    print("PART 3: RAW WRIST MOTION ANALYSIS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_idx = get_keypoint_indices(keypoint_cols)

    print("\nAnalyzing raw wrist positions and velocities (no scaling)...")

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 5:  # Just first 5 shots
            break

        player_id = metadata['participant_id']
        print(f"\n--- Shot {i+1}, Player {player_id} ---")

        # Get wrist positions at key frames
        frames = [100, 120, 140, 150, 160, 170, 180]

        print(f"{'Frame':<8} {'X':<10} {'Y':<10} {'Z':<10} {'dX/dt':<10} {'dZ/dt':<10}")
        print("-" * 58)

        prev_pos = None
        for frame in frames:
            pos = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame)
            if pos is not None:
                if prev_pos is not None:
                    # Velocity in units/frame * 60 fps = units/second
                    dx = (pos[0] - prev_pos[0]) * 60 / 20  # 20 frame gap
                    dz = (pos[2] - prev_pos[2]) * 60 / 20
                    print(f"{frame:<8} {pos[0]:<10.3f} {pos[1]:<10.3f} {pos[2]:<10.3f} {dx:<10.3f} {dz:<10.3f}")
                else:
                    print(f"{frame:<8} {pos[0]:<10.3f} {pos[1]:<10.3f} {pos[2]:<10.3f} {'--':<10} {'--':<10}")
                prev_pos = pos

        # Get ankle position for ground reference
        ankle = get_joint_position(timeseries, keypoint_idx, "right_ankle", 150)
        if ankle is not None:
            print(f"\nAnkle Z at frame 150: {ankle[2]:.3f}")
            wrist = get_joint_position(timeseries, keypoint_idx, "right_wrist", 150)
            if wrist is not None:
                print(f"Wrist-Ankle height diff: {wrist[2] - ankle[2]:.3f} units")

        # Scale factor
        scale = calibrate_scale_factor(timeseries, keypoint_idx)
        print(f"Scale factor: {scale:.4f} m/unit")
        print(f"Implied height diff in meters: {(wrist[2] - ankle[2]) * scale:.2f} m")


def simulate_with_diagnostics():
    """
    Run simulations with detailed diagnostics on why they fail.
    """
    print("\n" + "=" * 80)
    print("PART 4: SIMULATION FAILURE ANALYSIS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_idx = get_keypoint_indices(keypoint_cols)
    simulator = BasketballSimulator()

    success_count = 0
    failure_reasons = {
        'hit_ground': 0,
        'timeout': 0,
        'wrong_direction': 0,
    }

    print("\nDetailed analysis of first 20 shots:")
    print("-" * 80)

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 20:
            break

        player_id = metadata['participant_id']
        scale_factor = calibrate_scale_factor(timeseries, keypoint_idx)
        params = extract_all_release_params(timeseries, keypoint_idx, player_id, scale_factor)

        pos = params['position']
        vel = params['velocity']

        # Simulate
        landing, entry_angle, trajectory = simulator.simulate_shot(pos, vel, params['backspin'])

        # Analyze trajectory
        if len(trajectory) > 0:
            final_pos = trajectory[-1]
            max_z = max(t[2] for t in trajectory)
            final_x = final_pos[0]

            if landing is not None:
                success_count += 1
                status = "SUCCESS"
            elif final_pos[2] < 0.15:
                failure_reasons['hit_ground'] += 1
                status = "HIT_GROUND"
            elif final_x < -2:
                failure_reasons['wrong_direction'] += 1
                status = "WRONG_DIR"
            else:
                failure_reasons['timeout'] += 1
                status = "TIMEOUT"
        else:
            status = "NO_TRAJ"

        print(f"Shot {i+1:2d} P{player_id}: pos=({pos[0]:.1f},{pos[1]:.2f},{pos[2]:.1f}) "
              f"vel=({vel[0]:.1f},{vel[1]:.1f},{vel[2]:.1f}) "
              f"speed={np.linalg.norm(vel):.1f} -> {status}")

    print("-" * 80)
    print(f"\nSummary (first 20 shots):")
    print(f"  Success: {success_count}")
    print(f"  Hit ground: {failure_reasons['hit_ground']}")
    print(f"  Wrong direction: {failure_reasons['wrong_direction']}")
    print(f"  Timeout: {failure_reasons['timeout']}")


def analyze_coordinate_system():
    """
    Determine the correct coordinate system mapping.
    """
    print("\n" + "=" * 80)
    print("PART 5: COORDINATE SYSTEM ANALYSIS")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_idx = get_keypoint_indices(keypoint_cols)

    print("\nAnalyzing body part positions to determine coordinate axes...")

    # Get positions of key body parts at a mid-shot frame
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 1:  # Just first shot
            break

        frame = 150

        # Get multiple body parts
        joints = ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip',
                  'right_wrist', 'right_ankle', 'nose', 'right_elbow']

        positions = {}
        for joint in joints:
            pos = get_joint_position(timeseries, keypoint_idx, joint, frame)
            if pos is not None:
                positions[joint] = pos

        print(f"\nBody part positions at frame {frame}:")
        print(f"{'Joint':<20} {'X':<12} {'Y':<12} {'Z':<12}")
        print("-" * 56)
        for joint, pos in positions.items():
            print(f"{joint:<20} {pos[0]:<12.3f} {pos[1]:<12.3f} {pos[2]:<12.3f}")

        # Analyze coordinate system
        print("\nCoordinate system analysis:")

        # Shoulder width (should be horizontal)
        if 'right_shoulder' in positions and 'left_shoulder' in positions:
            rs, ls = positions['right_shoulder'], positions['left_shoulder']
            diff = rs - ls
            print(f"\nShoulder diff (R-L): X={diff[0]:.3f}, Y={diff[1]:.3f}, Z={diff[2]:.3f}")
            print(f"  -> Largest diff in {'XYZ'[np.argmax(np.abs(diff))]} axis")
            print(f"  -> This axis is LATERAL (left-right)")

        # Vertical (ankle to nose should be mostly vertical)
        if 'right_ankle' in positions and 'nose' in positions:
            ankle, nose = positions['right_ankle'], positions['nose']
            diff = nose - ankle
            print(f"\nNose-Ankle diff: X={diff[0]:.3f}, Y={diff[1]:.3f}, Z={diff[2]:.3f}")
            print(f"  -> Largest diff in {'XYZ'[np.argmax(np.abs(diff))]} axis")
            print(f"  -> This axis is VERTICAL (up-down)")

        # Forward (wrist should be in front during shot)
        if 'right_wrist' in positions and 'right_shoulder' in positions:
            wrist, shoulder = positions['right_wrist'], positions['right_shoulder']
            diff = wrist - shoulder
            print(f"\nWrist-Shoulder diff: X={diff[0]:.3f}, Y={diff[1]:.3f}, Z={diff[2]:.3f}")

        # Check wrist motion during shot
        print("\nWrist X position over time (to find forward direction):")
        wrist_x = []
        for f in range(100, 200, 10):
            pos = get_joint_position(timeseries, keypoint_idx, "right_wrist", f)
            if pos is not None:
                wrist_x.append((f, pos[0]))
                print(f"  Frame {f}: X = {pos[0]:.3f}")

        # See if X increases or decreases (toward hoop)
        if len(wrist_x) > 1:
            x_start = wrist_x[0][1]
            x_end = wrist_x[-1][1]
            direction = "INCREASING" if x_end > x_start else "DECREASING"
            print(f"\nWrist X {direction} during shot ({x_start:.3f} -> {x_end:.3f})")
            print(f"  -> If toward hoop, X is {'positive' if x_end > x_start else 'negative'} toward hoop")


def main():
    analyze_physics_requirements()
    analyze_raw_wrist_motion()
    analyze_coordinate_system()
    df = analyze_extracted_velocities()
    simulate_with_diagnostics()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
