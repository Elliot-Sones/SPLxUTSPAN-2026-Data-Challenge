"""
Train angle-specific model with strong regularization and feature selection.

Trains a model ONLY for angle prediction (not multi-output).
Uses:
- Angle-specific features
- Physics features
- Engineering features
- Feature selection (top predictive features)
- Strong regularization to prevent overfitting
- Group K-Fold cross-validation (by participant)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Import feature extractors
from data_loader import load_metadata, iterate_shots, get_keypoint_columns
from feature_engineering import init_keypoint_mapping as init_fe, extract_all_features
from physics_features import init_keypoint_mapping as init_pf, extract_physics_features
from angle_features import init_keypoint_mapping as init_af, extract_all_angle_features


def load_all_features(
    use_physics: bool = True,
    use_engineering: bool = True,
    use_angle_specific: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load and extract all features for all training shots.

    Returns:
        df: DataFrame with all features
        y: Target array (angle only)
        groups: Participant IDs for GroupKFold
    """
    print("Loading all training shots and extracting features...")

    # Initialize keypoint mappings
    keypoint_cols = get_keypoint_columns()
    if use_engineering:
        init_fe(keypoint_cols)
    if use_physics:
        init_pf(keypoint_cols)
    if use_angle_specific:
        init_af(keypoint_cols)

    # Extract features for all shots
    all_features = []
    angles = []
    groups = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True, chunk_size=20)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}...")

        feature_dict = {}

        # Extract different feature types
        if use_engineering:
            eng_features = extract_all_features(
                timeseries,
                participant_id=metadata["participant_id"],
                tiers=[1, 2, 3]
            )
            feature_dict.update(eng_features)

        if use_physics:
            phys_features = extract_physics_features(
                timeseries,
                participant_id=metadata["participant_id"],
                smooth=True
            )
            feature_dict.update(phys_features)

        if use_angle_specific:
            angle_features = extract_all_angle_features(
                timeseries,
                participant_id=metadata["participant_id"]
            )
            feature_dict.update(angle_features)

        all_features.append(feature_dict)
        angles.append(metadata["angle"])
        groups.append(metadata["participant_id"])

    print(f"Extracted features for {len(all_features)} shots")

    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    y = np.array(angles)

    print(f"Total features: {len(df.columns)}")
    print(f"Target (angle) range: [{y.min():.2f}, {y.max():.2f}]")

    return df, y, np.array(groups)


def train_angle_specific_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    model_type: str = "xgboost",
    n_features: int = 100,
    hyperparams: Optional[Dict] = None
) -> Tuple[List, Dict]:
    """
    Train angle-only model with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (angle only)
        groups: Participant IDs for GroupKFold
        feature_names: List of feature names
        model_type: "xgboost", "lightgbm", or "ridge"
        n_features: Number of top features to keep
        hyperparams: Optional hyperparameter dict

    Returns:
        models: List of trained models (one per fold)
        results: Dict with CV results
    """
    print(f"\nTraining {model_type} angle-specific model...")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Feature selection: keep only most predictive
    print(f"\nSelecting top {n_features} features...")
    feature_selector = SelectKBest(f_regression, k=min(n_features, X.shape[1]))
    X_selected = feature_selector.fit_transform(X, y)

    # Get selected feature names
    selected_mask = feature_selector.get_support()
    selected_features = [name for name, selected in zip(feature_names, selected_mask) if selected]
    print(f"Selected {len(selected_features)} features")

    # Cross-validation with GroupKFold (by participant)
    gkf = GroupKFold(n_splits=5)

    models = []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_selected, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")

        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

        # Train model based on type
        if model_type == "xgboost":
            model = train_xgboost_model(X_train, y_train, X_val, y_val, hyperparams)
        elif model_type == "lightgbm":
            model = train_lightgbm_model(X_train, y_train, X_val, y_val, hyperparams)
        elif model_type == "ridge":
            model = train_ridge_model(X_train, y_train, hyperparams)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        models.append(model)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_val = mean_squared_error(y_val, y_pred_val)
        r2_train = r2_score(y_train, y_pred_train)
        r2_val = r2_score(y_val, y_pred_val)

        print(f"Train MSE: {mse_train:.6f}, R²: {r2_train:.4f}")
        print(f"Val   MSE: {mse_val:.6f}, R²: {r2_val:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "train_mse": mse_train,
            "val_mse": mse_val,
            "train_r2": r2_train,
            "val_r2": r2_val
        })

    # Aggregate results
    results = {
        "fold_results": fold_results,
        "mean_train_mse": np.mean([r["train_mse"] for r in fold_results]),
        "mean_val_mse": np.mean([r["val_mse"] for r in fold_results]),
        "std_val_mse": np.std([r["val_mse"] for r in fold_results]),
        "mean_train_r2": np.mean([r["train_r2"] for r in fold_results]),
        "mean_val_r2": np.mean([r["val_r2"] for r in fold_results]),
        "selected_features": selected_features,
        "feature_selector": feature_selector
    }

    print("\n=== Cross-Validation Results ===")
    print(f"Mean Train MSE: {results['mean_train_mse']:.6f} (RMSE: {np.sqrt(results['mean_train_mse']):.4f}°)")
    print(f"Mean Val MSE:   {results['mean_val_mse']:.6f} (RMSE: {np.sqrt(results['mean_val_mse']):.4f}°)")
    print(f"Std Val MSE:    {results['std_val_mse']:.6f}")
    print(f"Mean Train R²:  {results['mean_train_r2']:.4f}")
    print(f"Mean Val R²:    {results['mean_val_r2']:.4f}")

    return models, results


def train_xgboost_model(X_train, y_train, X_val, y_val, hyperparams=None):
    """Train XGBoost model with strong regularization."""
    if hyperparams is None:
        hyperparams = {
            "n_estimators": 300,
            "max_depth": 4,  # Shallow trees to prevent overfitting
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,  # L1 regularization
            "reg_lambda": 10.0,  # Strong L2 regularization
            "min_child_weight": 5,  # Require more samples per leaf
            "random_state": 42,
            "tree_method": "hist",
            "early_stopping_rounds": 20
        }

    model = xgb.XGBRegressor(**hyperparams)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def train_lightgbm_model(X_train, y_train, X_val, y_val, hyperparams=None):
    """Train LightGBM model with strong regularization."""
    if hyperparams is None:
        hyperparams = {
            "n_estimators": 300,
            "max_depth": 4,
            "num_leaves": 15,  # Limit complexity
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "min_child_weight": 5,
            "random_state": 42,
            "verbose": -1
        }

    model = lgb.LGBMRegressor(**hyperparams)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )

    return model


def train_ridge_model(X_train, y_train, hyperparams=None):
    """Train Ridge regression with polynomial features."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

    if hyperparams is None:
        hyperparams = {
            "poly_degree": 2,
            "alpha": 10.0
        }

    # Create pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=hyperparams["poly_degree"])),
        ("ridge", Ridge(alpha=hyperparams["alpha"]))
    ])

    model.fit(X_train, y_train)

    return model


def save_model(models, results, feature_names, output_dir: Path, model_type: str):
    """Save trained models and results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    for i, model in enumerate(models):
        model_path = output_dir / f"{model_type}_angle_fold{i+1}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Save results
    results_path = output_dir / f"{model_type}_angle_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # Save feature names
    features_path = output_dir / f"{model_type}_angle_features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)

    print(f"\nSaved models to {output_dir}")


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Train angle-specific model")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm", "ridge"],
                        help="Model type")
    parser.add_argument("--n-features", type=int, default=100,
                        help="Number of top features to keep")
    parser.add_argument("--use-physics", action="store_true", default=True,
                        help="Use physics features")
    parser.add_argument("--use-engineering", action="store_true", default=True,
                        help="Use engineering features")
    parser.add_argument("--use-angle-specific", action="store_true", default=True,
                        help="Use angle-specific features")
    parser.add_argument("--output-dir", type=str, default="models/angle_specific",
                        help="Output directory for models")
    args = parser.parse_args()

    # Load features
    df, y, groups = load_all_features(
        use_physics=args.use_physics,
        use_engineering=args.use_engineering,
        use_angle_specific=args.use_angle_specific
    )

    # Remove non-numeric columns
    exclude_cols = ["participant_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    feature_names = feature_cols

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Groups (participants): {np.unique(groups)}")

    # Train model
    models, results = train_angle_specific_model(
        X=X,
        y=y,
        groups=groups,
        feature_names=feature_names,
        model_type=args.model,
        n_features=args.n_features
    )

    # Save models
    output_dir = Path(args.output_dir)
    save_model(models, results, feature_names, output_dir, args.model)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Features: {args.n_features} selected from {X.shape[1]}")
    print(f"Val MSE: {results['mean_val_mse']:.6f} (RMSE: {np.sqrt(results['mean_val_mse']):.4f}°)")
    print(f"Val R²: {results['mean_val_r2']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
