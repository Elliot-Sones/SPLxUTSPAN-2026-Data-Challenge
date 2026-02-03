"""
MuJoCo Physics-Based Trajectory Prediction.

Use physics to DIRECTLY predict where the ball lands relative to the hoop:
1. Simulate ball-hand contact until release
2. Get exact release position and velocity
3. Compute projectile trajectory to hoop plane
4. Calculate deviation from hoop center = target prediction

No ML needed - pure physics.
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

BALL_RADIUS = 0.12  # meters
FEET_TO_METERS = 0.3048
METERS_TO_FEET = 1 / FEET_TO_METERS
METERS_TO_INCHES = METERS_TO_FEET * 12
FPS = 60
GRAVITY = 9.81  # m/s^2

# Hoop position (from data analysis)
# Hoop center is at approximately [5.25, -25, 10] feet
HOOP_X = 5.25 * FEET_TO_METERS  # ~1.6 m
HOOP_Y = -25 * FEET_TO_METERS   # ~-7.62 m (negative = toward hoop)
HOOP_Z = 10 * FEET_TO_METERS    # ~3.05 m


def create_model():
    """Create MuJoCo model for ball-hand simulation."""
    xml = """
    <mujoco model="ball_trajectory">
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
    """Extract hand trajectory in meters."""
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
    """Check if ball is in contact with palm."""
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            return True
    return False


def projectile_trajectory(pos, vel, target_y):
    """
    Compute where projectile crosses the target y-plane.

    Using projectile motion equations:
    x(t) = x0 + vx*t
    y(t) = y0 + vy*t
    z(t) = z0 + vz*t - 0.5*g*t^2

    Solve for t when y(t) = target_y, then compute x(t) and z(t).
    """
    x0, y0, z0 = pos
    vx, vy, vz = vel

    # Time to reach target y-plane: y0 + vy*t = target_y
    if abs(vy) < 0.01:
        return None  # Ball not moving toward hoop

    t = (target_y - y0) / vy

    if t < 0:
        return None  # Ball moving away from hoop

    # Position at target y-plane
    x_final = x0 + vx * t
    z_final = z0 + vz * t - 0.5 * GRAVITY * t * t

    return {
        'time_of_flight': t,
        'x_at_hoop': x_final,
        'z_at_hoop': z_final,
        'y_at_hoop': target_y,
    }


def compute_trajectory_deviation(release_pos, release_vel):
    """
    Compute how far the ball lands from hoop center.

    Returns:
    - depth: distance past/short of hoop center (positive = long)
    - left_right: lateral deviation (positive = right of center)
    - angle: launch angle deviation from optimal
    """
    # Compute trajectory to hoop plane
    result = projectile_trajectory(release_pos, release_vel, HOOP_Y)

    if result is None:
        return None

    x_at_hoop = result['x_at_hoop']
    z_at_hoop = result['z_at_hoop']

    # Deviation from hoop center
    # left_right: x deviation (positive = ball lands right of center)
    left_right_m = x_at_hoop - HOOP_X
    left_right_inches = left_right_m * METERS_TO_INCHES

    # depth: This is trickier - it's about the trajectory arc, not just final position
    # For now, use z deviation as proxy (positive = ball too high/long)
    # Actually, depth likely refers to how far past the front of the rim
    # Let's compute based on z position relative to hoop height
    z_deviation_m = z_at_hoop - HOOP_Z
    depth_inches = z_deviation_m * METERS_TO_INCHES

    # Launch angle
    horizontal_speed = np.sqrt(release_vel[0]**2 + release_vel[1]**2)
    if horizontal_speed > 0.01:
        launch_angle = np.degrees(np.arctan2(release_vel[2], horizontal_speed))
    else:
        launch_angle = 90.0

    # Optimal launch angle for free throw is approximately 45-52 degrees
    # Deviation from 48 degrees (middle of optimal range)
    angle_deviation = launch_angle - 48.0

    return {
        'left_right': left_right_inches,
        'depth': depth_inches,
        'angle': angle_deviation,
        'x_at_hoop': x_at_hoop,
        'z_at_hoop': z_at_hoop,
        'time_of_flight': result['time_of_flight'],
        'launch_angle': launch_angle,
    }


def simulate_and_predict(timeseries, keypoint_map, model, data):
    """
    Simulate ball-hand contact and predict trajectory to hoop.
    """
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

    if not check_contact(model, data):
        return None

    sim_dt = model.opt.timestep
    frame_dt = 1.0 / FPS

    # Track contact state
    had_contact = True
    contact_count = 1
    no_contact_count = 0

    release_pos = None
    release_vel = None
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

            data.qpos[0:3] = hand_offset_interp
            data.qvel[0:3] = hand_vel_interp

            mujoco.mj_step(model, data)
            frame_time += sim_dt

        in_contact = check_contact(model, data)

        if in_contact:
            contact_count += 1
            no_contact_count = 0
        else:
            no_contact_count += 1

        # Release: had sufficient contact, now lost
        if had_contact and contact_count > 3 and no_contact_count >= 2:
            release_pos = data.qpos[3:6].copy()
            release_vel = data.qvel[3:6].copy()
            release_frame = frames[idx]
            break

    if release_pos is None:
        # Use final state
        release_pos = data.qpos[3:6].copy()
        release_vel = data.qvel[3:6].copy()
        release_frame = frames[-1]

    # Compute trajectory prediction
    prediction = compute_trajectory_deviation(release_pos, release_vel)

    if prediction is None:
        return None

    prediction['release_frame'] = release_frame
    prediction['release_pos'] = release_pos.tolist()
    prediction['release_vel'] = release_vel.tolist()
    prediction['release_speed'] = np.linalg.norm(release_vel)
    prediction['contact_frames'] = contact_count

    return prediction


def main():
    print("=" * 80)
    print("MUJOCO PHYSICS-BASED TRAJECTORY PREDICTION")
    print("=" * 80)
    print(f"\nHoop position: ({HOOP_X:.2f}, {HOOP_Y:.2f}, {HOOP_Z:.2f}) meters")
    print(f"               ({HOOP_X*METERS_TO_FEET:.2f}, {HOOP_Y*METERS_TO_FEET:.2f}, {HOOP_Z*METERS_TO_FEET:.2f}) feet")

    model = create_model()
    data = mujoco.MjData(model)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    print("\nProcessing training shots...")

    results = []
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i % 50 == 0:
            print(f"  Shot {i+1}...")

        prediction = simulate_and_predict(timeseries, keypoint_map, model, data)

        if prediction is not None:
            prediction['id'] = metadata['id']
            prediction['actual_angle'] = metadata.get('angle')
            prediction['actual_depth'] = metadata.get('depth')
            prediction['actual_left_right'] = metadata.get('left_right')
            results.append(prediction)

    print(f"\nProcessed {len(results)} shots successfully")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Compare predictions to actual targets
    print("\n" + "=" * 80)
    print("PREDICTION vs ACTUAL COMPARISON")
    print("=" * 80)

    # For each target, compute error
    for target in ['angle', 'depth', 'left_right']:
        pred_col = target  # Our prediction column has same name
        actual_col = f'actual_{target}'

        if pred_col in df.columns and actual_col in df.columns:
            # Remove NaN
            mask = ~(df[pred_col].isna() | df[actual_col].isna())
            pred = df.loc[mask, pred_col].values
            actual = df.loc[mask, actual_col].values

            # Compute metrics
            error = pred - actual
            mse = np.mean(error ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(error))
            corr = np.corrcoef(pred, actual)[0, 1]

            print(f"\n{target}:")
            print(f"  Correlation: {corr:.4f}")
            print(f"  MSE:  {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  Predicted range: [{pred.min():.2f}, {pred.max():.2f}]")
            print(f"  Actual range:    [{actual.min():.2f}, {actual.max():.2f}]")

    # Detailed analysis
    print("\n" + "=" * 80)
    print("TRAJECTORY STATISTICS")
    print("=" * 80)

    print(f"\nRelease speed (m/s):")
    print(f"  Mean: {df['release_speed'].mean():.2f}")
    print(f"  Max:  {df['release_speed'].max():.2f}")

    print(f"\nLaunch angle (degrees):")
    print(f"  Mean: {df['launch_angle'].mean():.1f}")
    print(f"  Range: [{df['launch_angle'].min():.1f}, {df['launch_angle'].max():.1f}]")

    print(f"\nTime of flight (seconds):")
    print(f"  Mean: {df['time_of_flight'].mean():.2f}")
    print(f"  Range: [{df['time_of_flight'].min():.2f}, {df['time_of_flight'].max():.2f}]")

    print(f"\nPredicted landing at hoop (meters):")
    print(f"  X: {df['x_at_hoop'].mean():.2f} +/- {df['x_at_hoop'].std():.2f} (hoop center: {HOOP_X:.2f})")
    print(f"  Z: {df['z_at_hoop'].mean():.2f} +/- {df['z_at_hoop'].std():.2f} (hoop height: {HOOP_Z:.2f})")

    # Save results
    output_path = PROJECT_DIR / "output" / "mujoco_trajectory_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

    # Show first few predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (first 10)")
    print("=" * 80)

    cols_to_show = ['id', 'angle', 'actual_angle', 'depth', 'actual_depth',
                    'left_right', 'actual_left_right', 'launch_angle', 'time_of_flight']
    print(df[cols_to_show].head(10).to_string())


if __name__ == "__main__":
    main()
