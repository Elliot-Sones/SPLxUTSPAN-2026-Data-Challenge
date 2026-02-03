"""
MuJoCo Velocity-Based Prediction Model.

Key insight: We can't compute full trajectory (missing horizontal velocity),
but the velocity components at release directly predict targets.

The MuJoCo simulation provides accurate ball velocities - use them as features.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
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
    <mujoco model="ball_velocity">
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


def extract_mujoco_features(timeseries, keypoint_map, mj_model, mj_data):
    """Extract ball velocity features from MuJoCo simulation."""

    hand_pos, frames = get_hand_trajectory(timeseries, keypoint_map)
    if hand_pos is None:
        return None

    # Compute hand velocities
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
    mujoco.mj_resetData(mj_model, mj_data)

    init_hand = hand_pos[start_idx]
    init_hand_vel = hand_vel[start_idx]
    hand_offset = init_hand - np.array([0, 0, 1.5])

    # Initial state
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

    # Track state
    had_contact = True
    contact_count = 1
    no_contact_count = 0
    max_ball_speed = 0
    max_ball_vz = 0

    release_vel = None
    release_pos = None
    release_frame = None

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

            mj_data.qpos[0:3] = hand_offset_interp
            mj_data.qvel[0:3] = hand_vel_interp

            mujoco.mj_step(mj_model, mj_data)
            frame_time += sim_dt

        ball_vel_frame = mj_data.qvel[3:6].copy()
        ball_speed = np.linalg.norm(ball_vel_frame)

        max_ball_speed = max(max_ball_speed, ball_speed)
        max_ball_vz = max(max_ball_vz, ball_vel_frame[2])

        in_contact = check_contact(mj_model, mj_data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
        else:
            no_contact_count += 1

        if had_contact and contact_count > 3 and no_contact_count >= 2:
            release_vel = ball_vel_frame
            release_pos = mj_data.qpos[3:6].copy()
            release_frame = frames[idx]
            break

    if release_vel is None:
        release_vel = mj_data.qvel[3:6].copy()
        release_pos = mj_data.qpos[3:6].copy()
        release_frame = frames[-1]

    # Extract features (in ft/s for consistency with data)
    release_vel_fps = release_vel / FEET_TO_METERS

    horizontal_speed = np.sqrt(release_vel_fps[0]**2 + release_vel_fps[1]**2)
    total_speed = np.linalg.norm(release_vel_fps)

    features = {
        # Velocity components (ft/s)
        'mj_vx': release_vel_fps[0],
        'mj_vy': release_vel_fps[1],
        'mj_vz': release_vel_fps[2],
        'mj_speed': total_speed,
        'mj_horizontal_speed': horizontal_speed,

        # Position at release
        'mj_release_z': release_pos[2],

        # Peak values
        'mj_max_speed': max_ball_speed / FEET_TO_METERS,
        'mj_max_vz': max_ball_vz / FEET_TO_METERS,

        # Timing
        'mj_release_frame': release_frame,
        'mj_contact_frames': contact_count,
    }

    return features


def main():
    print("=" * 80)
    print("MUJOCO VELOCITY-BASED PREDICTION MODEL")
    print("=" * 80)

    mj_model = create_model()
    mj_data = mujoco.MjData(mj_model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    print("\nExtracting MuJoCo features...")

    data = []
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i % 50 == 0:
            print(f"  Shot {i+1}...")

        features = extract_mujoco_features(timeseries, keypoint_map, mj_model, mj_data)

        if features is not None:
            features['id'] = metadata['id']
            features['participant_id'] = metadata.get('participant_id')
            features['angle'] = metadata.get('angle')
            features['depth'] = metadata.get('depth')
            features['left_right'] = metadata.get('left_right')
            data.append(features)

    df = pd.DataFrame(data)
    print(f"\nExtracted features from {len(df)} shots")

    # Feature columns
    feature_cols = ['mj_vx', 'mj_vy', 'mj_vz', 'mj_speed', 'mj_horizontal_speed',
                    'mj_release_z', 'mj_max_speed', 'mj_max_vz',
                    'mj_release_frame', 'mj_contact_frames']

    target_cols = ['angle', 'depth', 'left_right']

    # Remove rows with missing values
    df_clean = df.dropna(subset=feature_cols + target_cols)
    print(f"Clean samples: {len(df_clean)}")

    X = df_clean[feature_cols].values
    y = df_clean[target_cols].values
    groups = df_clean['participant_id'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (GroupKFold by player)")
    print("=" * 60)

    # GroupKFold by player
    gkf = GroupKFold(n_splits=5)

    for i, target in enumerate(target_cols):
        y_target = y[:, i]

        # Cross-validation predictions
        y_pred = np.zeros_like(y_target)

        for train_idx, val_idx in gkf.split(X_scaled, y_target, groups):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled[train_idx], y_target[train_idx])
            y_pred[val_idx] = model.predict(X_scaled[val_idx])

        mse = mean_squared_error(y_target, y_pred)
        rmse = np.sqrt(mse)
        corr = np.corrcoef(y_target, y_pred)[0, 1]

        print(f"\n{target}:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Corr: {corr:.4f}")

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Ridge coefficients)")
    print("=" * 60)

    for i, target in enumerate(target_cols):
        y_target = y[:, i]
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_target)

        print(f"\n{target}:")
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coef': model.coef_
        }).sort_values('coef', key=abs, ascending=False)

        for _, row in coef_df.head(5).iterrows():
            print(f"  {row['feature']:<25}: {row['coef']:>8.4f}")

    # Combined MSE across all targets
    print("\n" + "=" * 60)
    print("COMBINED METRICS")
    print("=" * 60)

    total_mse = 0
    for i, target in enumerate(target_cols):
        y_target = y[:, i]
        y_pred = np.zeros_like(y_target)

        for train_idx, val_idx in gkf.split(X_scaled, y_target, groups):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled[train_idx], y_target[train_idx])
            y_pred[val_idx] = model.predict(X_scaled[val_idx])

        mse = mean_squared_error(y_target, y_pred)
        total_mse += mse

    avg_mse = total_mse / 3
    print(f"\nAverage MSE across targets: {avg_mse:.6f}")
    print(f"\nCurrent best model MSE: 0.008305")
    print(f"Target MSE: < 0.007")

    # Save features
    output_path = PROJECT_DIR / "output" / "mujoco_velocity_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved features to {output_path}")


if __name__ == "__main__":
    main()
