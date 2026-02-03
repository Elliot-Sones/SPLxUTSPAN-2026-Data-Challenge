"""
Physics-Informed Features (Direct, No Simulation)

Uses the release parameter extraction but skips the MuJoCo simulation.
Maps release velocity, position, and angle directly to targets.

This tests whether the physics-informed feature extraction is valuable
even without the full trajectory simulation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "physics_engine"))

from data_loader import iterate_shots, load_scalers, get_keypoint_columns
from core import (
    calibrate_scale_factor,
    get_keypoint_indices,
    extract_all_release_params,
)

SUBMISSION_DIR = PROJECT_DIR / "submission"


def extract_physics_features(params: dict) -> np.ndarray:
    """
    Extract physics-informed features from release parameters.
    """
    pos = params['position']
    vel = params['velocity']

    # Basic features
    features = [
        # Position
        pos[1],  # Lateral position (Y)
        pos[2],  # Release height (Z)

        # Velocity
        vel[0],  # Forward velocity (vx)
        vel[1],  # Lateral velocity (vy)
        vel[2],  # Vertical velocity (vz)

        # Derived
        np.linalg.norm(vel),  # Total speed
        np.arctan2(vel[2], vel[0]),  # Release angle (radians)
        np.arctan2(vel[1], vel[0]),  # Lateral angle

        # Ratios
        vel[2] / (vel[0] + 0.01),  # vz/vx ratio
        vel[1] / (vel[0] + 0.01),  # vy/vx ratio

        # Interactions
        pos[2] * vel[2],  # Height x vertical velocity
        pos[1] * vel[1],  # Lateral position x lateral velocity

        # Release frame (timing)
        params['release_frame'],
        params['release_frame'] / 240.0,  # Normalized timing

        # Backspin
        params['backspin'],
    ]

    return np.array(features)


def run_physics_features_model():
    """
    Train a model using physics-informed features directly.
    """
    print("=" * 80)
    print("PHYSICS-INFORMED FEATURES (DIRECT, NO SIMULATION)")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_idx = get_keypoint_indices(keypoint_cols)
    scalers = load_scalers()

    # Extract features for training data
    print("\n1. Extracting physics features from training data...")

    X_train = []
    y_train = []
    player_ids = []
    shot_ids = []

    for metadata, timeseries in iterate_shots(train=True):
        player_id = metadata['participant_id']
        scale_factor = calibrate_scale_factor(timeseries, keypoint_idx)
        params = extract_all_release_params(timeseries, keypoint_idx, player_id, scale_factor)

        features = extract_physics_features(params)
        X_train.append(features)

        # Scaled targets
        y_train.append([
            scalers['angle'].transform([[metadata['angle']]])[0, 0],
            scalers['depth'].transform([[metadata['depth']]])[0, 0],
            scalers['left_right'].transform([[metadata['left_right']]])[0, 0],
        ])

        player_ids.append(player_id)
        shot_ids.append(metadata['id'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    player_ids = np.array(player_ids)

    print(f"   Training samples: {len(X_train)}")
    print(f"   Features per sample: {X_train.shape[1]}")

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    # Cross-validation
    print("\n2. Cross-validation...")

    gkf = GroupKFold(n_splits=5)
    cv_scores = {'angle': [], 'depth': [], 'left_right': []}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_scaled, groups=player_ids)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        for i, target in enumerate(['angle', 'depth', 'left_right']):
            model = Ridge(alpha=1.0)
            model.fit(X_tr, y_tr[:, i])
            pred = model.predict(X_val)
            mse = np.mean((pred - y_val[:, i])**2)
            cv_scores[target].append(mse)

    print(f"   CV MSE:")
    total_cv = 0
    for target in ['angle', 'depth', 'left_right']:
        mean_mse = np.mean(cv_scores[target])
        std_mse = np.std(cv_scores[target])
        print(f"     {target}: {mean_mse:.6f} +/- {std_mse:.6f}")
        total_cv += mean_mse

    mean_cv = total_cv / 3
    print(f"   Mean CV MSE: {mean_cv:.6f}")

    # Fit final models
    print("\n3. Fitting final models...")

    models = {}
    for i, target in enumerate(['angle', 'depth', 'left_right']):
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train[:, i])
        models[target] = model

        # Training R²
        train_pred = model.predict(X_train_scaled)
        ss_res = np.sum((y_train[:, i] - train_pred)**2)
        ss_tot = np.sum((y_train[:, i] - np.mean(y_train[:, i]))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"   {target} training R²: {r2:.4f}")

    # Generate test predictions
    print("\n4. Generating test predictions...")

    X_test = []
    test_ids = []
    test_player_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        player_id = metadata['participant_id']
        scale_factor = calibrate_scale_factor(timeseries, keypoint_idx)
        params = extract_all_release_params(timeseries, keypoint_idx, player_id, scale_factor)

        features = extract_physics_features(params)
        X_test.append(features)
        test_ids.append(metadata['id'])
        test_player_ids.append(player_id)

    X_test = np.array(X_test)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = scaler_X.transform(X_test)

    predictions = np.zeros((len(X_test), 3))
    for i, target in enumerate(['angle', 'depth', 'left_right']):
        predictions[:, i] = models[target].predict(X_test_scaled)

    # Clip to valid range
    predictions = np.clip(predictions, 0, 1)

    print(f"   Test predictions: {len(predictions)}")
    print(f"   angle_std: {np.std(predictions[:, 0]):.4f}")
    print(f"   depth_mean: {np.mean(predictions[:, 1]):.4f}")
    print(f"   left_right_std: {np.std(predictions[:, 2]):.4f}")

    # Save submission
    print("\n5. Saving submission...")

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions[:, 0],
        'scaled_depth': predictions[:, 1],
        'scaled_left_right': predictions[:, 2]
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"   Saved: {output_file}")

    # Compare with Sub 219
    print("\n6. Comparison with Sub 219...")

    sub219_path = SUBMISSION_DIR / "submission_219.csv"
    if sub219_path.exists():
        sub219 = pd.read_csv(sub219_path)
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            corr = np.corrcoef(submission[col].values, sub219[col].values)[0, 1]
            print(f"   {col}: r = {corr:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CV MSE: {mean_cv:.6f}")
    print(f"Target: < 0.007")
    print("=" * 80)

    return mean_cv


if __name__ == "__main__":
    run_physics_features_model()
