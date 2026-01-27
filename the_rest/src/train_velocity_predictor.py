"""
Train velocity predictor: Hand kinematics → Release velocity

Uses ground truth velocities from inverse ballistics as targets.
Trains separate models for vx, vy, vz using existing feature engineering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_keypoint_columns, iterate_shots
from src.feature_engineering import init_keypoint_mapping, extract_all_features


def load_ground_truth_velocities():
    """Load ground truth velocities computed by inverse ballistics."""
    path = Path(__file__).parent.parent / "output" / "ground_truth_velocities.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth velocities not found at {path}. "
            "Run compute_ground_truth_velocities.py first."
        )
    df = pd.read_csv(path)
    return df


def extract_features_for_all_shots(shot_ids, participant_ids, max_shots=None):
    """
    Extract features for all shots.

    Args:
        shot_ids: List of shot IDs to extract
        participant_ids: List of participant IDs
        max_shots: Maximum number of shots to process

    Returns:
        features_list: List of feature dictionaries
        feature_names: List of feature names
    """
    print("Extracting features for all shots...")

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    features_list = []
    shot_id_to_index = {sid: idx for idx, sid in enumerate(shot_ids)}

    n_shots = len(shot_ids) if max_shots is None else min(max_shots, len(shot_ids))

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= n_shots:
            break

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_shots} shots...")

        shot_id = metadata['shot_id']

        # Only extract for shots in our ground truth data
        if shot_id not in shot_id_to_index:
            continue

        try:
            # Extract all features (tiers 1, 2, 3)
            features = extract_all_features(
                timeseries,
                participant_id=metadata['participant_id'],
                tiers=[1, 2, 3]
            )

            # Add shot_id for matching
            features['shot_id'] = shot_id

            features_list.append(features)

        except Exception as e:
            print(f"Error extracting features for shot {shot_id}: {e}")
            continue

    print(f"Extracted features for {len(features_list)} shots")

    # Get feature names (exclude shot_id and participant_id to avoid duplication)
    if len(features_list) > 0:
        feature_names = [k for k in features_list[0].keys() if k not in ['shot_id', 'participant_id']]
    else:
        feature_names = []

    return features_list, feature_names


def train_velocity_models(
    X_train, y_train,
    X_val, y_val,
    component_name: str,
    model_type: str = 'xgboost'
):
    """
    Train a single velocity component model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        component_name: 'vx', 'vy', or 'vz'
        model_type: 'xgboost' or 'lightgbm'

    Returns:
        trained model
    """
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:  # lightgbm
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)

    print(f"  {component_name}:")
    print(f"    Train R²: {r2_train:.4f}, MAE: {mae_train:.4f}")
    print(f"    Val R²: {r2_val:.4f}, MAE: {mae_val:.4f}")

    return model, r2_val, mae_val


def train_velocity_predictor_cv(
    df_gt_velocities,
    features_list,
    feature_names,
    model_type='xgboost',
    n_folds=5
):
    """
    Train velocity predictor with cross-validation.

    Args:
        df_gt_velocities: Ground truth velocities DataFrame
        features_list: List of feature dictionaries
        feature_names: List of feature names
        model_type: 'xgboost' or 'lightgbm'
        n_folds: Number of CV folds

    Returns:
        models: Dictionary of trained models for each component
        cv_results: Cross-validation results
    """
    print(f"\nTraining {model_type} velocity predictors with {n_folds}-fold CV...")

    # Convert features to DataFrame
    df_features = pd.DataFrame(features_list)

    # Merge with ground truth velocities (drop participant_id from features to avoid duplication)
    df_features_clean = df_features.drop(columns=['participant_id'] if 'participant_id' in df_features.columns else [])
    df = df_gt_velocities.merge(df_features_clean, on='shot_id', how='inner')

    print(f"Matched {len(df)} shots with features and ground truth velocities")

    # Filter out low-quality convergences
    df = df[df['convergence_error'] < 1.0]
    print(f"After filtering convergence errors: {len(df)} shots")

    if len(df) < 50:
        print("Insufficient data for training")
        return None, None

    # Prepare features and targets
    # Make sure we only select feature columns that exist
    available_features = [f for f in feature_names if f in df.columns]
    missing_features = [f for f in feature_names if f not in df.columns]

    if len(missing_features) > 0:
        print(f"Warning: {len(missing_features)} features not found in data")
        print(f"First few missing: {missing_features[:5]}")

    X = df[available_features].values
    y_vx = df['gt_vx'].values
    y_vy = df['gt_vy'].values
    y_vz = df['gt_vz'].values
    groups = df['participant_id'].values

    # Replace NaN features with 0
    X = np.nan_to_num(X, nan=0.0)

    # Cross-validation
    gkf = GroupKFold(n_splits=n_folds)

    cv_results = {
        'vx': {'r2': [], 'mae': []},
        'vy': {'r2': [], 'mae': []},
        'vz': {'r2': [], 'mae': []},
    }

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_vx_train, y_vx_val = y_vx[train_idx], y_vx[val_idx]
        y_vy_train, y_vy_val = y_vy[train_idx], y_vy[val_idx]
        y_vz_train, y_vz_val = y_vz[train_idx], y_vz[val_idx]

        # Train models for each component
        _, r2_vx, mae_vx = train_velocity_models(
            X_train, y_vx_train, X_val, y_vx_val, 'vx', model_type
        )
        _, r2_vy, mae_vy = train_velocity_models(
            X_train, y_vy_train, X_val, y_vy_val, 'vy', model_type
        )
        _, r2_vz, mae_vz = train_velocity_models(
            X_train, y_vz_train, X_val, y_vz_val, 'vz', model_type
        )

        cv_results['vx']['r2'].append(r2_vx)
        cv_results['vx']['mae'].append(mae_vx)
        cv_results['vy']['r2'].append(r2_vy)
        cv_results['vy']['mae'].append(mae_vy)
        cv_results['vz']['r2'].append(r2_vz)
        cv_results['vz']['mae'].append(mae_vz)

    # Print CV summary
    print("\n=== Cross-Validation Summary ===")
    for component in ['vx', 'vy', 'vz']:
        r2_mean = np.mean(cv_results[component]['r2'])
        r2_std = np.std(cv_results[component]['r2'])
        mae_mean = np.mean(cv_results[component]['mae'])
        mae_std = np.std(cv_results[component]['mae'])

        print(f"{component}:")
        print(f"  R² = {r2_mean:.4f} ± {r2_std:.4f}")
        print(f"  MAE = {mae_mean:.4f} ± {mae_std:.4f}")

    # Train final models on all data
    print("\n=== Training Final Models ===")
    final_models = {}

    model_vx, _, _ = train_velocity_models(X, y_vx, X, y_vx, 'vx (full)', model_type)
    model_vy, _, _ = train_velocity_models(X, y_vy, X, y_vy, 'vy (full)', model_type)
    model_vz, _, _ = train_velocity_models(X, y_vz, X, y_vz, 'vz (full)', model_type)

    final_models['vx'] = model_vx
    final_models['vy'] = model_vy
    final_models['vz'] = model_vz
    final_models['feature_names'] = available_features

    # Save models
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    for component, model in [('vx', model_vx), ('vy', model_vy), ('vz', model_vz)]:
        model_path = models_dir / f"velocity_predictor_{component}_{model_type}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved {component} model to {model_path}")

    # Save feature names
    feature_names_path = models_dir / "velocity_predictor_feature_names.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(available_features, f)
    print(f"✓ Saved feature names to {feature_names_path}")

    # Save CV results
    cv_results_path = Path(__file__).parent.parent / "output" / f"velocity_predictor_cv_results_{model_type}.csv"
    cv_df = pd.DataFrame({
        'component': ['vx', 'vy', 'vz'],
        'r2_mean': [np.mean(cv_results[c]['r2']) for c in ['vx', 'vy', 'vz']],
        'r2_std': [np.std(cv_results[c]['r2']) for c in ['vx', 'vy', 'vz']],
        'mae_mean': [np.mean(cv_results[c]['mae']) for c in ['vx', 'vy', 'vz']],
        'mae_std': [np.std(cv_results[c]['mae']) for c in ['vx', 'vy', 'vz']],
    })
    cv_df.to_csv(cv_results_path, index=False)
    print(f"✓ Saved CV results to {cv_results_path}")

    return final_models, cv_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=['xgboost', 'lightgbm'], default='xgboost')
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-shots", type=int, default=None)

    args = parser.parse_args()

    # Load ground truth velocities
    print("Loading ground truth velocities...")
    df_gt = load_ground_truth_velocities()
    print(f"Loaded {len(df_gt)} shots with ground truth velocities")

    # Extract features
    features_list, feature_names = extract_features_for_all_shots(
        df_gt['shot_id'].values,
        df_gt['participant_id'].values,
        max_shots=args.max_shots
    )

    print(f"Extracted {len(feature_names)} features")

    # Train models
    models, cv_results = train_velocity_predictor_cv(
        df_gt,
        features_list,
        feature_names,
        model_type=args.model_type,
        n_folds=args.n_folds
    )

    if models is not None:
        print("\n✓ Velocity predictor training complete!")
