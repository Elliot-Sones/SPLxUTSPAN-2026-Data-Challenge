"""
Quick empirical test: Physics-based vs Direct ML prediction

Test setup:
- 100 training shots (fast)
- Physics features only (~40 features, not 3000+)
- Single 80/20 train/test split
- Compare MSE on test set

Approaches:
A) Direct ML: Features → Outcomes (angle, depth, left_right)
B) Physics: Features → Velocity → Physics → Outcomes
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_keypoint_columns, iterate_shots
from src.physics_features import init_keypoint_mapping, extract_physics_features
from src.inverse_ballistics import forward_ballistics, calculate_outcomes_from_landing


def load_data_quick(max_shots=100):
    """Load data with physics features only (fast)."""
    print(f"Loading {max_shots} shots with physics features...")

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= max_shots:
            break

        if (i + 1) % 25 == 0:
            print(f"  Loaded {i+1}/{max_shots}...")

        try:
            # Extract physics features (40 features)
            feats = extract_physics_features(timeseries, participant_id=metadata['participant_id'], smooth=True)

            # Add targets
            feats['angle'] = metadata['angle']
            feats['depth'] = metadata['depth']
            feats['left_right'] = metadata['left_right']
            feats['shot_id'] = metadata['shot_id']

            data.append(feats)

        except Exception as e:
            print(f"Error on shot {i}: {e}")
            continue

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} shots with {len(df.columns)-4} features")

    return df


def approach_a_direct_ml(X_train, y_train, X_test, y_test):
    """Approach A: Direct ML prediction (features → outcomes)."""
    print("\n=== Approach A: Direct ML ===")

    # Train separate model for each target
    models = {}
    predictions = {}

    for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train[:, target_idx], verbose=False)
        pred = model.predict(X_test)

        models[target_name] = model
        predictions[target_name] = pred

        mse = mean_squared_error(y_test[:, target_idx], pred)
        print(f"  {target_name}: MSE = {mse:.4f}")

    # Overall MSE
    y_pred = np.column_stack([predictions['angle'], predictions['depth'], predictions['left_right']])
    overall_mse = mean_squared_error(y_test, y_pred)
    print(f"  Overall MSE: {overall_mse:.4f}")

    return models, y_pred, overall_mse


def approach_b_physics(X_train, y_train, X_test, y_test, df_train, df_test):
    """Approach B: Physics-based prediction (features → velocity → physics → outcomes)."""
    print("\n=== Approach B: Physics-Based ===")

    # Load ground truth velocities
    gt_path = Path(__file__).parent.parent / "output" / "ground_truth_velocities.csv"
    if not gt_path.exists():
        print("Ground truth velocities not found, skipping")
        return None, None, None

    df_gt = pd.read_csv(gt_path)

    # Merge with train/test data
    df_train_merged = df_train.merge(df_gt[['shot_id', 'gt_vx', 'gt_vy', 'gt_vz', 'convergence_error']],
                                      on='shot_id', how='inner')
    df_test_merged = df_test.merge(df_gt[['shot_id', 'gt_vx', 'gt_vy', 'gt_vz', 'convergence_error']],
                                    on='shot_id', how='inner')

    # Filter out poor convergence
    df_train_merged = df_train_merged[df_train_merged['convergence_error'] < 1.0]
    df_test_merged = df_test_merged[df_test_merged['convergence_error'] < 1.0]

    print(f"  Train: {len(df_train_merged)} shots with GT velocities")
    print(f"  Test: {len(df_test_merged)} shots with GT velocities")

    if len(df_train_merged) < 20 or len(df_test_merged) < 5:
        print("Insufficient data for physics approach")
        return None, None, None

    # Get feature columns (exclude metadata and targets)
    feature_cols = [c for c in df_train.columns if c not in ['shot_id', 'angle', 'depth', 'left_right']]

    # Train velocity predictors
    X_train_vel = df_train_merged[feature_cols].values
    X_test_vel = df_test_merged[feature_cols].values

    X_train_vel = np.nan_to_num(X_train_vel, nan=0.0)
    X_test_vel = np.nan_to_num(X_test_vel, nan=0.0)

    vel_models = {}
    vel_predictions = {}

    for vel_component in ['vx', 'vy', 'vz']:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )

        y_vel = df_train_merged[f'gt_{vel_component}'].values
        model.fit(X_train_vel, y_vel, verbose=False)

        pred = model.predict(X_test_vel)
        vel_models[vel_component] = model
        vel_predictions[vel_component] = pred

        r2 = model.score(X_test_vel, df_test_merged[f'gt_{vel_component}'].values)
        print(f"  {vel_component} predictor: R² = {r2:.4f}")

    # Use physics to convert velocities to outcomes
    print("  Applying ballistic physics...")

    # Load hoop position
    hoop_path = Path(__file__).parent.parent / "output" / "calibrated_hoop_position.csv"
    hoop_df = pd.read_csv(hoop_path)
    hoop_pos = np.array([hoop_df['hoop_x'].values[0], hoop_df['hoop_y'].values[0], hoop_df['hoop_z'].values[0]])
    g = hoop_df['gravity'].values[0]

    predictions_physics = []

    for idx in range(len(df_test_merged)):
        row = df_test_merged.iloc[idx]

        # Release position
        release_pos = np.array([row['wrist_x_release'], row['wrist_y_release'], row['wrist_z_release']])

        # Predicted velocity
        pred_vel = np.array([
            vel_predictions['vx'][idx],
            vel_predictions['vy'][idx],
            vel_predictions['vz'][idx]
        ])

        # Forward ballistics
        result = forward_ballistics(release_pos, pred_vel, g)

        if result.get('valid', False):
            landing_pos = np.array([result['landing_x'], result['landing_y'], result['landing_z']])
            landing_vel = np.array([result['landing_vx'], result['landing_vy'], result['landing_vz']])
            outcomes = calculate_outcomes_from_landing(landing_pos, landing_vel, hoop_pos)

            predictions_physics.append([outcomes['angle'], outcomes['depth'], outcomes['left_right']])
        else:
            # Fallback to NaN
            predictions_physics.append([np.nan, np.nan, np.nan])

    y_pred_physics = np.array(predictions_physics)

    # Filter out NaN predictions
    valid_mask = ~np.isnan(y_pred_physics).any(axis=1)
    y_pred_physics_valid = y_pred_physics[valid_mask]
    y_test_physics = df_test_merged[['angle', 'depth', 'left_right']].values[valid_mask]

    print(f"  Valid predictions: {valid_mask.sum()}/{len(y_pred_physics)}")

    if valid_mask.sum() < 3:
        print("Too few valid predictions")
        return None, None, None

    # Calculate MSE
    for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
        mse = mean_squared_error(y_test_physics[:, target_idx], y_pred_physics_valid[:, target_idx])
        print(f"  {target_name}: MSE = {mse:.4f}")

    overall_mse = mean_squared_error(y_test_physics, y_pred_physics_valid)
    print(f"  Overall MSE: {overall_mse:.4f}")

    return vel_models, y_pred_physics_valid, overall_mse


def main():
    # Load data
    df = load_data_quick(max_shots=100)

    # Prepare features and targets
    feature_cols = [c for c in df.columns if c not in ['shot_id', 'angle', 'depth', 'left_right']]
    X = df[feature_cols].values
    y = df[['angle', 'depth', 'left_right']].values

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    # Train/test split (80/20)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"\nTrain: {len(X_train)} shots")
    print(f"Test: {len(X_test)} shots")
    print(f"Features: {X.shape[1]}")

    # Approach A: Direct ML
    models_a, pred_a, mse_a = approach_a_direct_ml(X_train, y_train, X_test, y_test)

    # Approach B: Physics
    models_b, pred_b, mse_b = approach_b_physics(X_train, y_train, X_test, y_test, df_train, df_test)

    # Comparison
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Approach A (Direct ML):    MSE = {mse_a:.4f}")

    if mse_b is not None:
        print(f"Approach B (Physics):      MSE = {mse_b:.4f}")

        if mse_b < mse_a:
            improvement = (mse_a - mse_b) / mse_a * 100
            print(f"\n✓ Physics wins by {improvement:.1f}% improvement!")
        else:
            degradation = (mse_b - mse_a) / mse_a * 100
            print(f"\n✗ Direct ML wins (Physics is {degradation:.1f}% worse)")
    else:
        print("Approach B (Physics):      Failed")

    print("\nNote: This is a quick test on 100 shots with physics features only.")
    print("Full comparison would use all 345 shots and more features.")


if __name__ == "__main__":
    main()
