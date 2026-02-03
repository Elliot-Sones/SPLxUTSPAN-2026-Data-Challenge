"""
MuJoCo Physics Feature Extraction Pipeline.

Extract physics-based features from ball-hand contact simulation:
- Ball velocity at release (vx, vy, vz, speed)
- Ball position at release
- Contact duration
- Release frame
- Velocity transfer efficiency
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import pandas as pd
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

BALL_RADIUS = 0.12
FEET_TO_METERS = 0.3048
FPS = 60


def create_model():
    """Create MuJoCo model for ball-hand simulation."""
    xml = """
    <mujoco model="ball_hand_features">
        <option gravity="0 0 -9.81" timestep="0.0002"/>

        <worldbody>
            <geom type="plane" size="20 20 0.1"/>

            <body name="hand" pos="0 0 1.5">
                <joint name="hx" type="slide" axis="1 0 0"/>
                <joint name="hy" type="slide" axis="0 1 0"/>
                <joint name="hz" type="slide" axis="0 0 1"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
            </body>

            <body name="ball" pos="0 0 1.66">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


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
    return np.array([
        timeseries[frame, km['x']],
        timeseries[frame, km['y']],
        timeseries[frame, km['z']]
    ])


def get_hand_trajectory(timeseries, keypoint_map):
    positions = []
    frames = []
    for frame in range(50, 200):
        pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
        if pos is None:
            pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if pos is not None and not np.any(np.isnan(pos)):
            positions.append(pos * FEET_TO_METERS)
            frames.append(frame)
    return (np.array(positions), np.array(frames)) if len(positions) > 20 else (None, None)


def check_contact(model, data):
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            return True
    return False


def extract_physics_features(timeseries, keypoint_map, model, data):
    """Extract physics features from MuJoCo simulation."""

    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    # Compute target velocities
    window = min(11, len(hand_pos) - 2)
    if window % 2 == 0:
        window -= 1
    hand_vel = np.zeros_like(hand_pos)
    for i in range(3):
        hand_vel[:, i] = savgol_filter(hand_pos[:, i], window, 3, deriv=1) * FPS

    # Find shot start
    start_idx = 0
    for i in range(len(hand_vel)):
        if hand_vel[i, 2] > 0.3:
            start_idx = max(0, i - 5)
            break

    # Reset simulation
    mujoco.mj_resetData(model, data)

    init_hand = hand_pos[start_idx]
    init_hand_vel = hand_vel[start_idx]
    hand_offset = init_hand - np.array([0, 0, 1.5])

    # Initial state
    data.qpos[0:3] = hand_offset
    palm_top_z = 1.5 + hand_offset[2] + 0.02
    ball_z = palm_top_z + BALL_RADIUS - 0.005

    data.qpos[3:6] = [init_hand[0], init_hand[1], ball_z]
    data.qpos[6:10] = [1, 0, 0, 0]
    data.qvel[0:3] = init_hand_vel
    data.qvel[3:6] = init_hand_vel
    data.qvel[6:9] = [0, 0, 0]

    mujoco.mj_forward(model, data)

    initial_contact = check_contact(model, data)
    if not initial_contact:
        return None  # Skip if no initial contact

    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS

    # Track state during simulation
    had_contact = True
    contact_count = 1
    no_contact_count = 0
    release_frame = None
    release_ball_vel = None
    release_ball_pos = None
    max_ball_speed = 0
    max_hand_speed = 0
    max_ball_vz = 0

    # Also track ball trajectory for computing features
    ball_velocities = []
    ball_positions = []

    for idx in range(start_idx, len(hand_pos) - 1):
        pos_curr = hand_pos[idx] - np.array([0, 0, 1.5])
        pos_next = hand_pos[idx + 1] - np.array([0, 0, 1.5])
        vel_curr = hand_vel[idx]
        vel_next = hand_vel[idx + 1]

        frame_time = 0.0
        while frame_time < frame_dt:
            t = frame_time / frame_dt
            hand_offset_interp = pos_curr * (1 - t) + pos_next * t
            hand_vel_interp = vel_curr * (1 - t) + vel_next * t

            data.qpos[0:3] = hand_offset_interp
            data.qvel[0:3] = hand_vel_interp

            mujoco.mj_step(model, data)
            frame_time += sim_dt

        # Record state
        ball_vel_frame = data.qvel[3:6].copy()
        ball_pos_frame = data.qpos[3:6].copy()
        ball_speed = np.linalg.norm(ball_vel_frame)
        hand_speed = np.linalg.norm(data.qvel[0:3])

        ball_velocities.append(ball_vel_frame)
        ball_positions.append(ball_pos_frame)

        max_ball_speed = max(max_ball_speed, ball_speed)
        max_hand_speed = max(max_hand_speed, hand_speed)
        max_ball_vz = max(max_ball_vz, ball_vel_frame[2])

        in_contact = check_contact(model, data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
        else:
            no_contact_count += 1

        # Release detection
        if had_contact and contact_count > 3 and no_contact_count >= 3:
            release_frame = frames[idx]
            release_ball_vel = ball_vel_frame
            release_ball_pos = ball_pos_frame
            break

    # If no clear release, use final state
    if release_frame is None:
        release_frame = frames[-1]
        release_ball_vel = data.qvel[3:6].copy()
        release_ball_pos = data.qpos[3:6].copy()

    # Compute features
    release_speed = np.linalg.norm(release_ball_vel)
    release_vx = release_ball_vel[0]
    release_vy = release_ball_vel[1]
    release_vz = release_ball_vel[2]

    release_x = release_ball_pos[0]
    release_y = release_ball_pos[1]
    release_z = release_ball_pos[2]

    # Compute hand velocity at release (from data)
    release_idx = np.searchsorted(frames, release_frame)
    release_idx = min(release_idx, len(hand_vel) - 1)
    hand_vel_at_release = hand_vel[release_idx]
    hand_speed_at_release = np.linalg.norm(hand_vel_at_release)

    # Velocity transfer
    vel_transfer = release_speed / hand_speed_at_release if hand_speed_at_release > 0.1 else 0

    # Launch angle (from horizontal)
    horizontal_speed = np.sqrt(release_vx**2 + release_vy**2)
    launch_angle = np.degrees(np.arctan2(release_vz, horizontal_speed)) if horizontal_speed > 0.01 else 90.0

    features = {
        # Release velocity components (meters per second)
        'mj_release_vx': release_vx,
        'mj_release_vy': release_vy,
        'mj_release_vz': release_vz,
        'mj_release_speed': release_speed,

        # Release velocity in ft/s
        'mj_release_speed_fps': release_speed / FEET_TO_METERS,
        'mj_release_vz_fps': release_vz / FEET_TO_METERS,

        # Release position (meters)
        'mj_release_x': release_x,
        'mj_release_y': release_y,
        'mj_release_z': release_z,

        # Timing
        'mj_release_frame': release_frame,
        'mj_contact_frames': contact_count,

        # Peak values
        'mj_max_ball_speed': max_ball_speed,
        'mj_max_ball_speed_fps': max_ball_speed / FEET_TO_METERS,
        'mj_max_ball_vz': max_ball_vz,
        'mj_max_hand_speed': max_hand_speed,

        # Derived
        'mj_velocity_transfer': vel_transfer,
        'mj_launch_angle': launch_angle,
        'mj_horizontal_speed': horizontal_speed,
    }

    return features


def main():
    print("=" * 80)
    print("MUJOCO PHYSICS FEATURE EXTRACTION")
    print("=" * 80)

    model = create_model()
    data = mujoco.MjData(model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    print("\nExtracting features from training data...")

    all_features = []
    all_targets = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}...")

        features = extract_physics_features(timeseries, keypoint_map, model, data)

        if features is not None:
            features['id'] = metadata['id']
            features['angle'] = metadata.get('angle')
            features['depth'] = metadata.get('depth')
            features['left_right'] = metadata.get('left_right')
            all_features.append(features)

    print(f"\nExtracted features from {len(all_features)} shots")

    # Create DataFrame
    merged = pd.DataFrame(all_features)
    print(f"Total shots with features: {len(merged)}")

    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION WITH TARGETS")
    print("=" * 80)

    feature_cols = [c for c in merged.columns if c.startswith('mj_')]
    target_cols = ['angle', 'depth', 'left_right']

    print(f"\n{'Feature':<30} {'angle':>10} {'depth':>10} {'left_right':>12}")
    print("-" * 65)

    best_correlations = {}
    for feature in feature_cols:
        corrs = []
        for target in target_cols:
            if feature in merged.columns and target in merged.columns:
                # Remove NaN
                mask = ~(merged[feature].isna() | merged[target].isna())
                if mask.sum() > 10:
                    corr = merged.loc[mask, feature].corr(merged.loc[mask, target])
                else:
                    corr = 0
            else:
                corr = 0
            corrs.append(corr)

            if target not in best_correlations or abs(corr) > abs(best_correlations[target][1]):
                best_correlations[target] = (feature, corr)

        print(f"{feature:<30} {corrs[0]:>10.3f} {corrs[1]:>10.3f} {corrs[2]:>12.3f}")

    print("\n" + "=" * 80)
    print("BEST CORRELATIONS PER TARGET")
    print("=" * 80)
    for target, (feature, corr) in best_correlations.items():
        print(f"  {target}: {feature} (r = {corr:.3f})")

    # Save features
    output_path = PROJECT_DIR / "output" / "mujoco_physics_features.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    for col in ['mj_release_speed_fps', 'mj_max_ball_speed_fps', 'mj_launch_angle', 'mj_velocity_transfer']:
        if col in merged.columns:
            values = merged[col].dropna()
            print(f"\n{col}:")
            print(f"  Mean: {values.mean():.2f}")
            print(f"  Std:  {values.std():.2f}")
            print(f"  Min:  {values.min():.2f}")
            print(f"  Max:  {values.max():.2f}")


if __name__ == "__main__":
    main()
