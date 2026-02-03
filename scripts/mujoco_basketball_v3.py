"""
MuJoCo Basketball Simulation v3

Correct coordinate mapping based on data analysis:
- X axis: Forward/Backward (negative X = toward hoop)
- Y axis: Left/Right
- Z axis: Up/Down

Data scale: ~0.29 m/unit (7 units height ≈ 2m)
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Court constants (meters)
HOOP_HEIGHT = 3.05
HOOP_RADIUS = 0.2286
FREE_THROW_DIST = 4.57

# Data scale factor
UNIT_TO_METERS = 0.29


def create_scene():
    xml = """
    <mujoco model="basketball">
        <option gravity="0 0 -9.81" timestep="0.001"/>
        <worldbody>
            <geom type="plane" size="20 20 0.1"/>
            <body name="hoop" pos="0 0 3.05">
                <geom type="cylinder" size="0.23 0.01" euler="90 0 0" rgba="1 0.3 0 1"/>
            </body>
            <body name="ball" pos="-4 0 2">
                <freejoint name="ball_joint"/>
                <geom type="sphere" size="0.12" mass="0.625" rgba="1 0.5 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml), mujoco.MjData(mujoco.MjModel.from_xml_string(xml))


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def extract_release_data(timeseries, keypoint_idx):
    """Extract wrist position and velocity at release."""
    def get_joint(name, frame):
        idx = keypoint_idx.get(name)
        if idx is None or frame >= len(timeseries):
            return None
        pos = timeseries[frame, idx*3:(idx+1)*3]
        return pos if np.any(pos) else None

    # Get wrist at multiple frames
    wrist_150 = get_joint("right_wrist", 150)
    wrist_153 = get_joint("right_wrist", 153)
    wrist_156 = get_joint("right_wrist", 156)

    if wrist_150 is None or wrist_153 is None:
        return None

    # Also get reference point (ankle) for ground level
    ankle = get_joint("right_ankle", 153)
    ground_z = ankle[2] if ankle is not None else 0.9

    return {
        'pos_150': wrist_150,
        'pos_153': wrist_153,
        'pos_156': wrist_156,
        'ground_z': ground_z,
        'velocity': (wrist_153 - wrist_150) / (3/60)  # 3 frames at 60fps
    }


def simulate_shot(model, data, release_pos, release_vel):
    """
    Simulate shot and return landing position at hoop plane.

    Returns: (y_landing, z_landing) or None if miss
    """
    mujoco.mj_resetData(model, data)

    data.qpos[0:3] = release_pos
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[0:3] = release_vel
    data.qvel[3:6] = [0, 0, 0]

    prev_x = release_pos[0]

    while data.time < 3.0:
        mujoco.mj_step(model, data)
        pos = data.qpos[0:3].copy()

        # Check if crossed hoop plane (x goes from negative to ~0)
        if prev_x < 0.1 and pos[0] >= 0.1:
            return pos[1], pos[2]  # y, z at crossing

        if pos[2] < 0.12:  # Hit ground
            return None

        prev_x = pos[0]

    return None


def physics_predict(release_data, params, model, data):
    """
    Map keypoint data to physics prediction using tunable parameters.

    params: [pos_scale, vel_scale, vel_x_mult, vel_z_mult, height_offset, lateral_scale]
    """
    pos_scale, vel_scale, vel_x_mult, vel_z_mult, height_offset, lateral_scale = params

    pos = release_data['pos_153']
    vel = release_data['velocity']
    ground = release_data['ground_z']

    # Convert to meters with correct coordinate mapping
    # Our X (forward) -> MuJoCo X
    # Our Y (left/right) -> MuJoCo Y
    # Our Z (up) -> MuJoCo Z

    # Release position
    height_above_ground = (pos[2] - ground) * UNIT_TO_METERS * pos_scale
    release_z = height_above_ground + height_offset

    # Lateral position
    release_y = pos[1] * UNIT_TO_METERS * lateral_scale

    # Start from free throw distance
    release_x = -FREE_THROW_DIST

    release_pos = np.array([release_x, release_y, release_z])

    # Release velocity
    # Our data has velocity in units/sec, convert to m/s
    # Negative X in our data = toward hoop = positive X in simulation
    vel_x = -vel[0] * UNIT_TO_METERS * vel_scale * vel_x_mult  # flip X
    vel_y = vel[1] * UNIT_TO_METERS * vel_scale * 0.3  # reduce lateral
    vel_z = vel[2] * UNIT_TO_METERS * vel_scale * vel_z_mult

    # Ensure reasonable velocity
    speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    if speed < 5:
        # Too slow, scale up
        factor = 7 / (speed + 0.1)
        vel_x *= factor
        vel_z *= factor

    release_vel = np.array([vel_x, vel_y, vel_z])

    # Simulate
    result = simulate_shot(model, data, release_pos, release_vel)

    if result is None:
        return None

    y_land, z_land = result

    # Convert landing position to our target space [0, 1]
    # Hoop center is at y=0, z=3.05

    # Lateral error -> affects angle and left_right
    lateral_error = y_land  # deviation from center in meters

    # Height error -> affects depth (arc)
    height_error = z_land - HOOP_HEIGHT

    return lateral_error, height_error


def objective(params, training_data, model, data, scalers):
    """Evaluate parameters on training data."""
    total_error = 0
    count = 0

    for release_data, targets in training_data:
        result = physics_predict(release_data, params, model, data)

        if result is None:
            total_error += 0.5  # Penalty for missed simulation
            count += 1
            continue

        lateral_error, height_error = result

        # Map to predictions
        pred_angle = 0.5 + lateral_error * 0.2  # Tune this mapping
        pred_lr = 0.5 - lateral_error * 0.3
        pred_depth = 0.5 + height_error * 0.5

        pred_angle = np.clip(pred_angle, 0, 1)
        pred_lr = np.clip(pred_lr, 0, 1)
        pred_depth = np.clip(pred_depth, 0, 1)

        err = ((pred_angle - targets['angle'])**2 +
               (pred_depth - targets['depth'])**2 +
               (pred_lr - targets['left_right'])**2) / 3

        total_error += err
        count += 1

    return total_error / count if count > 0 else 1.0


def main():
    print("="*80)
    print("MUJOCO BASKETBALL SIMULATION v3 - Correct Mapping")
    print("="*80)

    model, data = create_scene()
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Load training data
    print("\nLoading training data...")
    training_data = []

    for metadata, timeseries in iterate_shots(train=True):
        release_data = extract_release_data(timeseries, keypoint_idx)
        if release_data is None:
            continue

        targets = {
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        }
        training_data.append((release_data, targets))

    print(f"Loaded {len(training_data)} samples")

    # Optimize parameters
    print("\nOptimizing simulation parameters...")

    # params: [pos_scale, vel_scale, vel_x_mult, vel_z_mult, height_offset, lateral_scale]
    bounds = [
        (0.5, 3.0),   # pos_scale
        (0.5, 5.0),   # vel_scale
        (0.5, 3.0),   # vel_x_mult
        (0.5, 3.0),   # vel_z_mult
        (1.5, 2.5),   # height_offset
        (0.1, 1.0),   # lateral_scale
    ]

    result = differential_evolution(
        lambda p: objective(p, training_data[:100], model, data, scalers),
        bounds,
        maxiter=50,
        seed=42,
        disp=True,
        workers=1
    )

    best_params = result.x
    print(f"\nBest params: {best_params}")
    print(f"Best MSE: {result.fun:.4f}")

    # Evaluate on full training set
    full_mse = objective(best_params, training_data, model, data, scalers)
    print(f"Full training MSE: {full_mse:.4f}")

    # Learn output mapping using training data
    print("\nLearning output mapping...")

    train_physics = []
    train_targets_list = []

    for release_data, targets in training_data:
        result = physics_predict(release_data, best_params, model, data)
        if result is not None:
            train_physics.append(result)
            train_targets_list.append([targets['angle'], targets['depth'], targets['left_right']])

    train_physics = np.array(train_physics)
    train_targets_arr = np.array(train_targets_list)

    print(f"Valid physics predictions: {len(train_physics)}")

    # Fit linear mapping from physics output to targets
    from sklearn.linear_model import Ridge

    X_physics = np.column_stack([
        train_physics[:, 0],  # lateral error
        train_physics[:, 1],  # height error
        train_physics[:, 0]**2,
        train_physics[:, 1]**2,
        np.ones(len(train_physics))
    ])

    models = []
    for i in range(3):
        reg = Ridge(alpha=1.0)
        reg.fit(X_physics, train_targets_arr[:, i])
        models.append(reg)
        score = reg.score(X_physics, train_targets_arr[:, i])
        print(f"  Target {i} R2: {score:.4f}")

    # Generate test predictions
    print("\n" + "="*60)
    print("GENERATING TEST PREDICTIONS")
    print("="*60)

    test_predictions = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        release_data = extract_release_data(timeseries, keypoint_idx)
        test_ids.append(metadata['id'])

        if release_data is None:
            test_predictions.append([0.5, 0.5, 0.5])
            continue

        result = physics_predict(release_data, best_params, model, data)

        if result is None:
            test_predictions.append([0.5, 0.5, 0.5])
            continue

        lateral_error, height_error = result
        X = np.array([[lateral_error, height_error, lateral_error**2, height_error**2, 1]])

        preds = [np.clip(models[i].predict(X)[0], 0, 1) for i in range(3)]
        test_predictions.append(preds)

    predictions = np.array(test_predictions)

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055

    print(f"Predictions shape: {predictions.shape}")
    print(f"angle_std: {np.std(predictions[:, 0]):.4f}")
    print(f"depth_mean: {np.mean(predictions[:, 1]):.4f}")

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions[:, 0],
        'scaled_depth': predictions[:, 1],
        'scaled_left_right': predictions[:, 2]
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Compare
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Blend with Sub 219
    print("\nBlending with Sub 219...")
    for w in [0.1, 0.2, 0.3]:
        blend = submission.copy()
        blend['scaled_angle'] = w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values
        blend['scaled_depth'] = w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values
        blend['scaled_left_right'] = w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"  {w:.0%} physics + {1-w:.0%} Sub219: angle_std={blend['scaled_angle'].std():.4f} -> {blend_file}")


if __name__ == "__main__":
    main()
