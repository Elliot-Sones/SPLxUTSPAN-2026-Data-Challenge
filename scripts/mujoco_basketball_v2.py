"""
MuJoCo Basketball Simulation v2

Improved coordinate mapping and parameter optimization.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.optimize import minimize
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent

# Court constants (meters)
HOOP_HEIGHT = 3.05
HOOP_RADIUS = 0.2286
BALL_RADIUS = 0.12
FREE_THROW_DIST = 4.57


def create_basketball_scene():
    """Create MuJoCo XML for basketball scene."""
    xml = """
    <mujoco model="basketball">
        <option gravity="0 0 -9.81" timestep="0.002"/>

        <worldbody>
            <geom type="plane" size="10 10 0.1"/>

            <!-- Hoop at origin, facing -x direction -->
            <body name="hoop" pos="0 0 3.05">
                <geom type="cylinder" size="0.2286 0.01" euler="90 0 0" rgba="1 0.3 0 1"/>
            </body>

            <!-- Basketball - free body -->
            <body name="ball" pos="-4 0 2">
                <freejoint name="ball_joint"/>
                <geom type="sphere" size="0.12" mass="0.625" rgba="1 0.5 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_wrist_data(timeseries, keypoint_idx):
    """Extract wrist position and velocity around release."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return None
        idx = keypoint_idx[name]
        pos = timeseries[frame, idx*3:(idx+1)*3]
        return pos if np.any(pos) else None

    # Get wrist positions around release
    positions = []
    for f in range(148, 158):
        pos = get_joint("right_wrist", f)
        if pos is not None:
            positions.append((f, pos))

    if len(positions) < 3:
        return None

    # Release position (frame 153)
    release_pos = get_joint("right_wrist", 153)
    pre_pos = get_joint("right_wrist", 150)

    if release_pos is None or pre_pos is None:
        return None

    # Velocity
    dt = 3 / 60.0
    velocity = (release_pos - pre_pos) / dt

    return {
        'position': release_pos,
        'velocity': velocity,
        'positions': positions
    }


def simulate_shot(model, data, release_pos, release_vel, hoop_pos=np.array([0, 0, 3.05])):
    """Simulate shot and return where it crosses hoop plane."""
    mujoco.mj_resetData(model, data)

    # Set ball position and velocity
    data.qpos[0:3] = release_pos
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[0:3] = release_vel
    data.qvel[3:6] = [0, 0, 0]

    # Track trajectory
    prev_x = release_pos[0]
    landing = None

    while data.time < 3.0:
        mujoco.mj_step(model, data)
        ball_pos = data.qpos[0:3].copy()

        # Check if crossed hoop x-plane (going from negative to positive x)
        if prev_x < 0 and ball_pos[0] >= 0:
            # Interpolate to exact crossing
            t = -prev_x / (ball_pos[0] - prev_x)
            landing = {
                'y': ball_pos[1],  # lateral position
                'z': ball_pos[2],  # height
            }
            break

        # Check ground collision
        if ball_pos[2] < BALL_RADIUS:
            break

        prev_x = ball_pos[0]

    return landing


def map_keypoints_to_physics(wrist_data, params):
    """
    Map keypoint data to physics parameters using tunable params.

    params: [scale_pos, scale_vel, vel_angle_adjust, vel_up_boost, x_offset]
    """
    scale_pos, scale_vel, vel_angle, vel_up, x_offset = params

    pos = wrist_data['position']
    vel = wrist_data['velocity']

    # Scale position to meters
    # Assume z=1.0 in our data corresponds to ~2m release height
    release_pos = np.array([
        -FREE_THROW_DIST + x_offset,  # Start at free throw line
        pos[1] * scale_pos,  # Lateral position
        1.8 + pos[2] * scale_pos * 0.5  # Height (base + scaled)
    ])

    # Scale and adjust velocity
    speed = np.linalg.norm(vel) * scale_vel

    # Direction: primarily forward and up
    vel_dir = vel / (np.linalg.norm(vel) + 1e-6)

    release_vel = np.array([
        speed * 0.7,  # Forward (toward hoop)
        vel_dir[1] * speed * 0.3,  # Lateral (from wrist motion)
        speed * (0.5 + vel_up)  # Upward
    ])

    return release_pos, release_vel


def evaluate_params(params, training_data, model, data, scalers):
    """Evaluate how well params predict training outcomes."""
    errors = []

    for wrist_data, actual_targets in training_data:
        release_pos, release_vel = map_keypoints_to_physics(wrist_data, params)
        landing = simulate_shot(model, data, release_pos, release_vel)

        if landing is None:
            errors.append(1.0)  # Penalty for missed shots
            continue

        # Compare landing position to actual targets
        # Our targets are scaled 0-1, landing is in meters

        # Lateral error -> relates to angle and left_right targets
        # Hoop center is at y=0, radius=0.23m
        lateral_error = landing['y']  # Deviation from center

        # Height error -> relates to depth (arc)
        height_error = landing['z'] - HOOP_HEIGHT

        # Map to our target space (rough approximation)
        pred_lr = 0.5 + lateral_error / 1.0  # Scale to 0-1
        pred_angle = 0.5 + lateral_error / 2.0  # Different scaling
        pred_depth = 0.5 + height_error / 0.5

        # Clamp
        pred_lr = np.clip(pred_lr, 0, 1)
        pred_angle = np.clip(pred_angle, 0, 1)
        pred_depth = np.clip(pred_depth, 0, 1)

        # Error vs actual
        err = (
            (pred_angle - actual_targets['angle'])**2 +
            (pred_depth - actual_targets['depth'])**2 +
            (pred_lr - actual_targets['left_right'])**2
        )
        errors.append(err)

    return np.mean(errors)


def main():
    print("="*80)
    print("MUJOCO BASKETBALL SIMULATION v2")
    print("="*80)

    # Create simulation
    xml = create_basketball_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Load all training data
    print("\nLoading training data...")
    training_data = []

    for metadata, timeseries in iterate_shots(train=True):
        wrist_data = extract_wrist_data(timeseries, keypoint_idx)
        if wrist_data is None:
            continue

        targets = {
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        }
        training_data.append((wrist_data, targets))

    print(f"Loaded {len(training_data)} valid training samples")

    # Initial parameters
    # [scale_pos, scale_vel, vel_angle, vel_up, x_offset]
    initial_params = [1.0, 1.0, 0.0, 0.3, 0.0]

    print("\nEvaluating initial parameters...")
    initial_error = evaluate_params(initial_params, training_data[:50], model, data, scalers)
    print(f"Initial MSE: {initial_error:.4f}")

    # Optimize parameters
    print("\nOptimizing simulation parameters...")

    def objective(params):
        return evaluate_params(params, training_data[:100], model, data, scalers)

    result = minimize(
        objective,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': 100, 'disp': True}
    )

    best_params = result.x
    print(f"\nOptimized params: {best_params}")
    print(f"Optimized MSE: {result.fun:.4f}")

    # Evaluate on all training data
    print("\nEvaluating on full training set...")
    full_error = evaluate_params(best_params, training_data, model, data, scalers)
    print(f"Full training MSE: {full_error:.4f}")

    # Generate predictions for test set
    print("\n" + "="*60)
    print("GENERATING TEST PREDICTIONS")
    print("="*60)

    test_predictions = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        wrist_data = extract_wrist_data(timeseries, keypoint_idx)

        if wrist_data is None:
            # Fallback: use mean predictions
            test_predictions.append([0.5, 0.5, 0.5])
            test_ids.append(metadata['id'])
            continue

        release_pos, release_vel = map_keypoints_to_physics(wrist_data, best_params)
        landing = simulate_shot(model, data, release_pos, release_vel)

        if landing is None:
            test_predictions.append([0.5, 0.5, 0.5])
        else:
            pred_lr = np.clip(0.5 + landing['y'] / 1.0, 0, 1)
            pred_angle = np.clip(0.5 + landing['y'] / 2.0, 0, 1)
            pred_depth = np.clip(0.5 + (landing['z'] - HOOP_HEIGHT) / 0.5, 0, 1)
            test_predictions.append([pred_angle, pred_depth, pred_lr])

        test_ids.append(metadata['id'])

    predictions = np.array(test_predictions)

    # Calibrate depth mean
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055

    print(f"Test predictions shape: {predictions.shape}")
    print(f"angle_std: {np.std(predictions[:, 0]):.4f}")
    print(f"depth_mean: {np.mean(predictions[:, 1]):.4f}")

    # Save submission
    import pandas as pd
    SUBMISSION_DIR = PROJECT_DIR / "submission"
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

    # Compare with best submissions
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")


if __name__ == "__main__":
    main()
