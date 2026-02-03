"""
MuJoCo model extracting ball velocity at PEAK HAND VELOCITY.

Key insight: Ball velocity at peak hand velocity closely matches hand velocity.
We should extract features at this moment, not at contact loss.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

BALL_RADIUS = 0.12
FEET_TO_METERS = 0.3048
FPS = 60


def create_model():
    xml = """
    <mujoco model="peak_velocity">
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


def check_contact(model, data):
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            return True
    return False


def extract_features(timeseries, keypoint_map, mj_model, mj_data):
    """Extract ball velocity at peak hand velocity."""

    # Get hand trajectory
    hand_positions = []
    frames = []
    for frame in range(50, 200):
        pos = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
        if pos is None:
            pos = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if pos is not None and not np.any(np.isnan(pos)):
            hand_positions.append(pos * FEET_TO_METERS)
            frames.append(frame)

    if len(hand_positions) < 30:
        return None

    hand_positions = np.array(hand_positions)
    frames = np.array(frames)

    # Compute velocities
    window = 11
    hand_vel = np.zeros_like(hand_positions)
    for j in range(3):
        hand_vel[:, j] = savgol_filter(hand_positions[:, j], window, 3, deriv=1) * FPS

    # Find peak upward velocity
    vz = hand_vel[:, 2]
    peak_idx = np.argmax(vz)

    if peak_idx < 10 or peak_idx >= len(frames) - 5:
        return None

    peak_frame = frames[peak_idx]
    kinematic_vel_at_peak = hand_vel[peak_idx]  # m/s

    # Run MuJoCo simulation to get ball velocity at peak frame
    start_idx = max(0, peak_idx - 25)

    mujoco.mj_resetData(mj_model, mj_data)

    init_hand = hand_positions[start_idx]
    init_hand_vel = hand_vel[start_idx]
    hand_offset = init_hand - np.array([0, 0, 1.5])

    mj_data.qpos[0:3] = hand_offset
    palm_top_z = 1.5 + hand_offset[2] + 0.02
    ball_z = palm_top_z + BALL_RADIUS - 0.005

    mj_data.qpos[3:6] = [init_hand[0], init_hand[1], ball_z]
    mj_data.qpos[6:10] = [1, 0, 0, 0]
    mj_data.qvel[0:3] = init_hand_vel
    mj_data.qvel[3:6] = init_hand_vel
    mj_data.qvel[6:9] = [0, 0, 0]

    mujoco.mj_forward(mj_model, mj_data)

    if not check_contact(mj_model, mj_data):
        return None

    sim_dt = mj_model.opt.timestep
    frame_dt = 1.0 / FPS

    ball_vel_at_peak = None

    for idx in range(start_idx, peak_idx + 1):
        if idx >= len(hand_positions) - 1:
            break

        pos_curr = hand_positions[idx] - np.array([0, 0, 1.5])
        pos_next = hand_positions[idx + 1] - np.array([0, 0, 1.5])
        vel_curr = hand_vel[idx]
        vel_next = hand_vel[idx + 1]

        frame_time = 0.0
        while frame_time < frame_dt:
            t = frame_time / frame_dt
            mj_data.qpos[0:3] = pos_curr * (1 - t) + pos_next * t
            mj_data.qvel[0:3] = vel_curr * (1 - t) + vel_next * t
            mujoco.mj_step(mj_model, mj_data)
            frame_time += sim_dt

        if idx == peak_idx:
            ball_vel_at_peak = mj_data.qvel[3:6].copy()

    if ball_vel_at_peak is None:
        return None

    # Convert to ft/s
    kinematic_vel_fps = kinematic_vel_at_peak / FEET_TO_METERS
    ball_vel_fps = ball_vel_at_peak / FEET_TO_METERS

    # Also get ball position at peak
    ball_pos_at_peak = mj_data.qpos[3:6].copy()

    # Compute derived features
    kinematic_speed = np.linalg.norm(kinematic_vel_fps)
    ball_speed = np.linalg.norm(ball_vel_fps)
    kinematic_horizontal = np.sqrt(kinematic_vel_fps[0]**2 + kinematic_vel_fps[1]**2)
    ball_horizontal = np.sqrt(ball_vel_fps[0]**2 + ball_vel_fps[1]**2)

    features = {
        # Kinematic (hand) velocities at peak
        'kin_vx': kinematic_vel_fps[0],
        'kin_vy': kinematic_vel_fps[1],
        'kin_vz': kinematic_vel_fps[2],
        'kin_speed': kinematic_speed,
        'kin_horizontal': kinematic_horizontal,

        # MuJoCo (ball) velocities at peak
        'ball_vx': ball_vel_fps[0],
        'ball_vy': ball_vel_fps[1],
        'ball_vz': ball_vel_fps[2],
        'ball_speed': ball_speed,
        'ball_horizontal': ball_horizontal,

        # Ball position at peak
        'ball_x': ball_pos_at_peak[0],
        'ball_y': ball_pos_at_peak[1],
        'ball_z': ball_pos_at_peak[2],

        # Timing
        'peak_frame': peak_frame,

        # Velocity transfer
        'vel_transfer': ball_speed / kinematic_speed if kinematic_speed > 0.1 else 1.0,
    }

    return features


def main():
    print("=" * 80)
    print("MUJOCO PEAK VELOCITY MODEL")
    print("=" * 80)
    print("\nExtracting ball velocity at PEAK HAND VELOCITY (not at contact loss)")

    mj_model = create_model()
    mj_data = mujoco.MjData(mj_model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    print("\nExtracting features...")

    data = []
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i % 50 == 0:
            print(f"  Shot {i+1}...")

        features = extract_features(timeseries, keypoint_map, mj_model, mj_data)

        if features is not None:
            features['id'] = metadata['id']
            features['participant_id'] = metadata.get('participant_id')
            features['angle'] = metadata.get('angle')
            features['depth'] = metadata.get('depth')
            features['left_right'] = metadata.get('left_right')
            data.append(features)

    df = pd.DataFrame(data)
    print(f"\nExtracted features from {len(df)} shots")

    # Verify velocity transfer
    print("\n" + "=" * 60)
    print("VELOCITY TRANSFER VERIFICATION")
    print("=" * 60)

    print(f"\nKinematic hand vz: {df['kin_vz'].mean():.2f} +/- {df['kin_vz'].std():.2f} ft/s")
    print(f"MuJoCo ball vz:    {df['ball_vz'].mean():.2f} +/- {df['ball_vz'].std():.2f} ft/s")
    print(f"Transfer ratio:    {df['vel_transfer'].mean():.2f} +/- {df['vel_transfer'].std():.2f}")

    corr_vz = df['kin_vz'].corr(df['ball_vz'])
    print(f"\nCorrelation (kin_vz vs ball_vz): {corr_vz:.3f}")

    # Correlation with targets
    print("\n" + "=" * 60)
    print("CORRELATION WITH TARGETS")
    print("=" * 60)

    feature_cols = ['kin_vx', 'kin_vy', 'kin_vz', 'kin_speed',
                    'ball_vx', 'ball_vy', 'ball_vz', 'ball_speed',
                    'ball_z', 'peak_frame']

    print(f"\n{'Feature':<15} {'angle':>10} {'depth':>10} {'left_right':>12}")
    print("-" * 50)

    for feat in feature_cols:
        corrs = []
        for target in ['angle', 'depth', 'left_right']:
            corr = df[feat].corr(df[target])
            corrs.append(corr)
        print(f"{feat:<15} {corrs[0]:>10.3f} {corrs[1]:>10.3f} {corrs[2]:>12.3f}")

    # Cross-validation
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (GroupKFold by player)")
    print("=" * 60)

    X = df[feature_cols].values
    y = df[['angle', 'depth', 'left_right']].values
    groups = df['participant_id'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=5)

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y_target = y[:, i]
        y_pred = np.zeros_like(y_target)

        for train_idx, val_idx in gkf.split(X_scaled, y_target, groups):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled[train_idx], y_target[train_idx])
            y_pred[val_idx] = model.predict(X_scaled[val_idx])

        mse = mean_squared_error(y_target, y_pred)
        rmse = np.sqrt(mse)
        corr = np.corrcoef(y_target, y_pred)[0, 1]

        print(f"\n{target}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Corr: {corr:.4f}")

    # Save
    output_path = PROJECT_DIR / "output" / "mujoco_peak_velocity_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
