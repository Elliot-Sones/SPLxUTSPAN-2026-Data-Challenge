"""
Basketball Physics Simulation

Generate synthetic training data by simulating basketball trajectories.
Learn the relationship between release parameters and shot outcomes.

Physics model:
- Ball follows parabolic trajectory under gravity
- Success depends on entry angle, position relative to hoop
- We simulate thousands of shots with varying release params
- Then apply learned patterns to real body pose data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)

# Basketball court constants (in feet)
HOOP_HEIGHT = 10.0  # 10 feet
HOOP_RADIUS = 0.75  # 9 inches = 0.75 feet
BALL_RADIUS = 0.39  # ~4.7 inches
FREE_THROW_DISTANCE = 15.0  # feet from hoop
THREE_POINT_DISTANCE = 23.75  # feet (NBA)

# Physics constants
GRAVITY = 32.174  # ft/s^2


class BasketballPhysics:
    """Simulate basketball shot trajectories."""

    def __init__(self, hoop_x=0, hoop_y=0, hoop_z=HOOP_HEIGHT):
        self.hoop_pos = np.array([hoop_x, hoop_y, hoop_z])
        self.hoop_radius = HOOP_RADIUS
        self.ball_radius = BALL_RADIUS

    def simulate_trajectory(self, release_pos, release_vel, dt=0.01, max_time=3.0):
        """
        Simulate ball trajectory from release to hoop plane.

        Args:
            release_pos: [x, y, z] position at release (feet)
            release_vel: [vx, vy, vz] velocity at release (ft/s)
            dt: time step
            max_time: maximum simulation time

        Returns:
            trajectory: list of positions
            entry_point: where ball crosses hoop plane (z = hoop_z)
            entry_vel: velocity at entry
        """
        pos = np.array(release_pos, dtype=float)
        vel = np.array(release_vel, dtype=float)

        trajectory = [pos.copy()]

        t = 0
        while t < max_time:
            # Update velocity (gravity in -z direction)
            vel[2] -= GRAVITY * dt

            # Update position
            pos = pos + vel * dt
            trajectory.append(pos.copy())

            # Check if ball has descended to hoop height
            if pos[2] <= self.hoop_pos[2] and vel[2] < 0:
                # Interpolate to exact hoop height
                # Find where z = hoop_z
                prev_pos = trajectory[-2]
                t_cross = (self.hoop_pos[2] - prev_pos[2]) / (pos[2] - prev_pos[2])
                entry_point = prev_pos + t_cross * (pos - prev_pos)

                # Entry velocity
                entry_vel = vel.copy()

                return trajectory, entry_point, entry_vel

            # Check if ball went below ground
            if pos[2] < 0:
                return trajectory, None, None

            t += dt

        return trajectory, None, None

    def evaluate_shot(self, release_pos, release_vel):
        """
        Evaluate a shot and return outcome metrics.

        Returns:
            dict with:
                - made: bool, whether shot would go in
                - angle_error: horizontal angle error (degrees)
                - depth_error: front/back error (feet, positive = long)
                - left_right_error: side error (feet, positive = right)
                - entry_angle: angle of ball entering hoop (degrees from vertical)
        """
        trajectory, entry_point, entry_vel = self.simulate_trajectory(
            release_pos, release_vel
        )

        if entry_point is None:
            return {
                'made': False,
                'angle_error': 90,
                'depth_error': 10,
                'left_right_error': 10,
                'entry_angle': 0
            }

        # Calculate errors relative to hoop center
        dx = entry_point[0] - self.hoop_pos[0]  # depth error (toward/away from shooter)
        dy = entry_point[1] - self.hoop_pos[1]  # left/right error

        # Entry angle (from vertical)
        horizontal_vel = np.sqrt(entry_vel[0]**2 + entry_vel[1]**2)
        entry_angle = np.degrees(np.arctan2(horizontal_vel, -entry_vel[2]))

        # Distance from hoop center
        distance_from_center = np.sqrt(dx**2 + dy**2)

        # Ball must fit through hoop with margin
        effective_radius = self.hoop_radius - self.ball_radius * 0.5

        # Check if shot goes in (simplified - ignores rim bounces)
        # Need good entry angle (> 30 degrees from horizontal) and position
        made = (distance_from_center < effective_radius) and (entry_angle > 30)

        # Angle error: how far off-center horizontally
        angle_error = np.degrees(np.arctan2(dy, 15))  # assuming 15ft distance

        return {
            'made': made,
            'angle_error': angle_error,
            'depth_error': dx,
            'left_right_error': dy,
            'entry_angle': entry_angle,
            'distance_from_center': distance_from_center
        }


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic shooting data with known outcomes."""
    print(f"Generating {n_samples} synthetic shots...")

    physics = BasketballPhysics()

    data = []

    for i in range(n_samples):
        # Random release position (simulating different shooters/positions)
        # Assume shooting from ~15 feet away, varying positions
        release_x = -15 + np.random.uniform(-2, 2)  # distance from hoop
        release_y = np.random.uniform(-3, 3)  # left/right position
        release_z = np.random.uniform(7, 9)  # release height (7-9 feet)

        # Random release velocity
        # Need ~20-25 ft/s to reach hoop from 15 feet
        speed = np.random.uniform(18, 28)

        # Release angle (from horizontal)
        release_angle = np.random.uniform(45, 65)  # degrees

        # Horizontal aiming (toward hoop with some error)
        aim_error = np.random.uniform(-5, 5)  # degrees

        # Convert to velocity components
        horizontal_speed = speed * np.cos(np.radians(release_angle))
        vz = speed * np.sin(np.radians(release_angle))
        vx = horizontal_speed * np.cos(np.radians(aim_error))
        vy = horizontal_speed * np.sin(np.radians(aim_error))

        release_pos = [release_x, release_y, release_z]
        release_vel = [vx, vy, vz]

        # Simulate shot
        result = physics.evaluate_shot(release_pos, release_vel)

        # Store features and outcomes
        data.append({
            # Release features
            'release_x': release_x,
            'release_y': release_y,
            'release_z': release_z,
            'release_speed': speed,
            'release_angle': release_angle,
            'release_vx': vx,
            'release_vy': vy,
            'release_vz': vz,
            'aim_error': aim_error,

            # Outcomes
            'made': result['made'],
            'angle_error': result['angle_error'],
            'depth_error': result['depth_error'],
            'left_right_error': result['left_right_error'],
            'entry_angle': result['entry_angle'],
        })

    df = pd.DataFrame(data)

    # Stats
    print(f"  Made shots: {df['made'].sum()} ({df['made'].mean()*100:.1f}%)")
    print(f"  Angle error range: [{df['angle_error'].min():.2f}, {df['angle_error'].max():.2f}]")
    print(f"  Depth error range: [{df['depth_error'].min():.2f}, {df['depth_error'].max():.2f}]")

    return df


def train_physics_model(synthetic_data):
    """Train model to predict shot outcomes from release parameters."""
    print("\nTraining physics-based predictor...")

    feature_cols = ['release_x', 'release_y', 'release_z',
                    'release_speed', 'release_angle',
                    'release_vx', 'release_vy', 'release_vz']

    X = synthetic_data[feature_cols].values

    # Normalize outcomes to [0, 1] range similar to our targets
    y_angle = (synthetic_data['angle_error'] - synthetic_data['angle_error'].min()) / \
              (synthetic_data['angle_error'].max() - synthetic_data['angle_error'].min())
    y_depth = (synthetic_data['depth_error'] - synthetic_data['depth_error'].min()) / \
              (synthetic_data['depth_error'].max() - synthetic_data['depth_error'].min())
    y_lr = (synthetic_data['left_right_error'] - synthetic_data['left_right_error'].min()) / \
           (synthetic_data['left_right_error'].max() - synthetic_data['left_right_error'].min())

    y = np.column_stack([y_angle, y_depth, y_lr])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train neural network to learn physics mapping
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Check training performance
    pred = model.predict(X_scaled)
    mse = np.mean((pred - y)**2)
    print(f"  Training MSE: {mse:.6f}")

    return scaler, model, feature_cols


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_release_features(timeseries, keypoint_idx):
    """
    Extract release parameters from body pose.
    Use wrist position/velocity as proxy for ball release.
    """
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Release frame (based on our analysis, frame 153 is release)
    release_frame = 153
    pre_release_frame = 150

    # Get wrist positions
    wrist_release = get_joint("right_wrist", release_frame)
    wrist_pre = get_joint("right_wrist", pre_release_frame)

    if np.any(wrist_release) and np.any(wrist_pre):
        # Position at release (convert from normalized to approximate feet)
        # Our data is normalized, so we estimate scale
        # Typical wrist height at release ~7-8 feet
        scale = 8.0 / max(wrist_release[2], 0.1)  # rough scale factor

        features['release_x'] = wrist_release[0] * scale
        features['release_y'] = wrist_release[1] * scale
        features['release_z'] = wrist_release[2] * scale

        # Velocity (frames at 60fps, so dt = 3/60 = 0.05 sec)
        dt = 3 / 60
        vel = (wrist_release - wrist_pre) / dt * scale

        features['release_vx'] = vel[0]
        features['release_vy'] = vel[1]
        features['release_vz'] = vel[2]

        # Speed and angle
        features['release_speed'] = np.linalg.norm(vel)
        horizontal_speed = np.sqrt(vel[0]**2 + vel[1]**2)
        features['release_angle'] = np.degrees(np.arctan2(vel[2], horizontal_speed))

    return features


def main():
    print("="*80)
    print("BASKETBALL PHYSICS SIMULATION")
    print("="*80)

    # Step 1: Generate synthetic data
    synthetic = generate_synthetic_data(n_samples=50000)

    # Step 2: Train physics model
    physics_scaler, physics_model, feature_cols = train_physics_model(synthetic)

    # Step 3: Extract release features from real data
    print("\n" + "="*60)
    print("EXTRACTING RELEASE FEATURES FROM REAL DATA")
    print("="*60)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_release_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])

    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_release_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train_raw = pd.DataFrame(train_features)
    X_test_raw = pd.DataFrame(test_features)

    # Fill missing with median
    for col in feature_cols:
        if col not in X_train_raw.columns:
            X_train_raw[col] = 0
            X_test_raw[col] = 0

    X_train_raw = X_train_raw[feature_cols].fillna(X_train_raw.median())
    X_test_raw = X_test_raw[feature_cols].fillna(X_test_raw.median())

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Training samples: {len(X_train_raw)}")
    print(f"Test samples: {len(X_test_raw)}")

    # Step 4: Use physics model to generate features
    print("\n" + "="*60)
    print("APPLYING PHYSICS MODEL")
    print("="*60)

    X_train_scaled = physics_scaler.transform(X_train_raw)
    X_test_scaled = physics_scaler.transform(X_test_raw)

    # Get physics predictions as features
    physics_pred_train = physics_model.predict(X_train_scaled)
    physics_pred_test = physics_model.predict(X_test_scaled)

    print("Physics predictions (train):")
    print(f"  Angle range: [{physics_pred_train[:, 0].min():.3f}, {physics_pred_train[:, 0].max():.3f}]")
    print(f"  Depth range: [{physics_pred_train[:, 1].min():.3f}, {physics_pred_train[:, 1].max():.3f}]")

    # Correlations with actual targets
    print("\nCorrelation with actual targets:")
    for i, name in enumerate(['angle', 'depth', 'left_right']):
        corr = np.corrcoef(physics_pred_train[:, i], y_train[:, i])[0, 1]
        print(f"  {name}: {corr:.4f}")

    # Step 5: Combine raw features + physics features
    X_train_combined = np.hstack([X_train_raw.values, physics_pred_train])
    X_test_combined = np.hstack([X_test_raw.values, physics_pred_test])

    print(f"\nCombined features: {X_train_combined.shape[1]}")

    # Step 6: Train final model
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)

    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train_combined)
    X_test_final = final_scaler.transform(X_test_combined)

    predictions = np.zeros((len(X_test_final), 3))
    cv_scores = []

    for alpha in [1, 10, 50, 100]:
        scores = []
        for i in range(3):
            y = y_train[:, i]
            gkf = GroupKFold(n_splits=5)
            fold_scores = []
            for train_idx, val_idx in gkf.split(X_train_final, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_final[train_idx], y[train_idx])
                pred = model.predict(X_train_final[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                fold_scores.append(mse)
            scores.append(np.mean(fold_scores))
        print(f"Ridge alpha={alpha}: angle={scores[0]:.4f}, depth={scores[1]:.4f}, lr={scores[2]:.4f}, mean={np.mean(scores):.4f}")

    # Use best alpha
    best_alpha = 50
    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        gkf = GroupKFold(n_splits=5)
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train_final, y, players):
            model = Ridge(alpha=best_alpha)
            model.fit(X_train_final[train_idx], y[train_idx])
            pred = model.predict(X_train_final[val_idx])
            mse = np.mean((pred - y[val_idx])**2)
            fold_scores.append(mse)

        cv_score = np.mean(fold_scores)
        cv_scores.append(cv_score)

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_final, y)
        predictions[:, i] = model.predict(X_test_final)

    print(f"\nFinal CV: angle={cv_scores[0]:.4f}, depth={cv_scores[1]:.4f}, lr={cv_scores[2]:.4f}")
    print(f"Mean CV: {np.mean(cv_scores):.6f}")

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    angle_std = np.std(predictions[:, 0])
    print(f"angle_std: {angle_std:.6f}")

    # Save submission
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

    # Compare with existing
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    corr219 = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    corr133 = np.corrcoef(predictions[:, 0], sub133['scaled_angle'].values)[0, 1]

    print(f"\nCorrelation with Sub 219: {corr219:.4f}")
    print(f"Correlation with Sub 133: {corr133:.4f}")

    # Blend with Sub 219
    print("\n" + "="*60)
    print("BLENDING WITH SUB 219")
    print("="*60)

    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = submission.copy()
        blend['scaled_angle'] = w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values
        blend['scaled_depth'] = w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values
        blend['scaled_left_right'] = w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

        std = blend['scaled_angle'].std()

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"  {w:.0%} physics + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
