"""
Training script for SPLxUTSPAN 2026 Data Challenge.

Implements GroupKFold cross-validation by participant and model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import pickle
import time

from data_loader import (
    iterate_shots, load_metadata, get_keypoint_columns,
    load_scalers, scale_targets, DATA_DIR
)
from feature_engineering import (
    init_keypoint_mapping, extract_all_features
)
from models.baseline import XGBoostBaseline, LightGBMBaseline, SklearnBaseline, get_top_features


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_features_for_all_shots(
    train: bool = True,
    tiers: List[int] = [1, 2, 3],
    max_shots: Optional[int] = None,
    cache_file: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Extract features for all shots.

    Args:
        train: Whether to use training data
        tiers: Which feature tiers to extract
        max_shots: Maximum shots to process (for testing)
        cache_file: Optional path to cache extracted features

    Returns:
        X: Feature matrix (n_shots, n_features)
        y: Target matrix (n_shots, 3) or None for test
        feature_names: List of feature names
        meta: Metadata DataFrame
    """
    # Check cache
    if cache_file and cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["X"], cached["y"], cached["feature_names"], cached["meta"]

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Load metadata
    meta = load_metadata(train)
    n_shots = len(meta) if max_shots is None else min(max_shots, len(meta))

    print(f"Extracting features for {n_shots} shots (tiers={tiers})...")

    # Extract features
    all_features = []
    all_targets = []
    processed = 0

    start_time = time.time()
    for metadata, timeseries in iterate_shots(train, chunk_size=20):
        if processed >= n_shots:
            break

        features = extract_all_features(
            timeseries,
            participant_id=metadata["participant_id"],
            tiers=tiers
        )
        all_features.append(features)

        if train:
            all_targets.append([
                metadata["angle"],
                metadata["depth"],
                metadata["left_right"]
            ])

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {processed}/{n_shots} shots ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"Feature extraction complete: {processed} shots in {elapsed:.1f}s")

    # Convert to arrays
    feature_names = sorted(all_features[0].keys())
    X = np.array([
        [f.get(name, np.nan) for name in feature_names]
        for f in all_features
    ], dtype=np.float32)

    y = np.array(all_targets, dtype=np.float32) if train else None

    # Handle NaN values - replace with column median
    for i in range(X.shape[1]):
        col = X[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            X[nan_mask, i] = median_val

    print(f"Feature matrix shape: {X.shape}")
    print(f"NaN values remaining: {np.isnan(X).sum()}")

    # Cache results
    if cache_file:
        print(f"Caching features to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "meta": meta.iloc[:n_shots]
            }, f)

    return X, y, feature_names, meta.iloc[:n_shots]


def train_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    model_type: str = "xgboost",
    n_folds: int = 5,
) -> Tuple[object, Dict]:
    """
    Train model with GroupKFold cross-validation.

    Args:
        X: Feature matrix
        y: Target matrix
        groups: Group labels (participant_id)
        feature_names: List of feature names
        model_type: "xgboost" or "lightgbm"
        n_folds: Number of CV folds

    Returns:
        model: Trained model on full data
        cv_results: Dictionary with CV metrics
    """
    print(f"\nTraining {model_type} with {n_folds}-fold GroupKFold CV...")

    gkf = GroupKFold(n_splits=n_folds)

    fold_mses = []
    fold_mses_per_target = {"angle": [], "depth": [], "left_right": []}
    oof_predictions = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        val_participant = np.unique(groups[val_idx])[0]
        print(f"\nFold {fold + 1}: Validating on participant {val_participant}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Create and train model with improved hyperparameters
        if model_type == "xgboost":
            model = XGBoostBaseline(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
            )
        elif model_type == "lightgbm":
            model = LightGBMBaseline(
                n_estimators=500,
                num_leaves=20,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
            )
        else:  # sklearn fallback
            model = SklearnBaseline(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.7,
            )

        model.fit(X_train, y_train, feature_names)

        # Predict
        y_pred = model.predict(X_val)
        oof_predictions[val_idx] = y_pred

        # Calculate MSE
        mse = mean_squared_error(y_val, y_pred)
        fold_mses.append(mse)

        target_names = ["angle", "depth", "left_right"]
        for i, name in enumerate(target_names):
            target_mse = mean_squared_error(y_val[:, i], y_pred[:, i])
            fold_mses_per_target[name].append(target_mse)
            print(f"  {name} MSE: {target_mse:.6f}")

        print(f"  Total MSE: {mse:.6f}")

    # Overall CV results
    cv_results = {
        "mean_mse": np.mean(fold_mses),
        "std_mse": np.std(fold_mses),
        "fold_mses": fold_mses,
        "per_target_mses": {
            name: {
                "mean": np.mean(mses),
                "std": np.std(mses),
                "folds": mses
            }
            for name, mses in fold_mses_per_target.items()
        },
        "oof_predictions": oof_predictions,
    }

    print(f"\n{'='*50}")
    print(f"CV Results ({model_type}):")
    print(f"  Mean MSE: {cv_results['mean_mse']:.6f} +/- {cv_results['std_mse']:.6f}")
    for name, stats in cv_results["per_target_mses"].items():
        print(f"  {name}: {stats['mean']:.6f} +/- {stats['std']:.6f}")

    # Train final model on all data with same improved hyperparameters
    print(f"\nTraining final model on all data...")
    if model_type == "xgboost":
        final_model = XGBoostBaseline(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
    elif model_type == "lightgbm":
        final_model = LightGBMBaseline(
            n_estimators=500,
            num_leaves=20,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
    else:  # sklearn fallback
        final_model = SklearnBaseline(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.7,
        )

    final_model.fit(X, y, feature_names)

    # Feature importance
    importances = final_model.get_feature_importance()
    top_features = get_top_features(importances, feature_names, top_k=15)

    print("\nTop 15 features per target:")
    for target, features in top_features.items():
        print(f"\n{target}:")
        for fname, imp in features[:10]:
            print(f"  {fname}: {imp:.4f}")

    return final_model, cv_results


def main():
    """Main training pipeline."""
    print("SPLxUTSPAN 2026 Data Challenge - Training Pipeline")
    print("=" * 60)

    # Extract features (with caching)
    cache_file = OUTPUT_DIR / "features_train.pkl"
    X, y, feature_names, meta = extract_features_for_all_shots(
        train=True,
        tiers=[1, 2, 3],
        cache_file=cache_file
    )

    # Get participant groups
    groups = meta["participant_id"].values

    print(f"\nDataset summary:")
    print(f"  Shots: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Participants: {np.unique(groups)}")

    # Load scalers for later use
    scalers = load_scalers()

    # Try to determine which model types are available
    model_types_to_train = []

    # Try sklearn first (always works)
    model_types_to_train.append("sklearn")

    # Try LightGBM (usually works without extra deps)
    try:
        import lightgbm
        model_types_to_train.append("lightgbm")
    except Exception as e:
        print(f"LightGBM not available: {e}")

    # Try XGBoost (may need libomp on Mac)
    try:
        import xgboost
        # Quick check if it actually works
        xgboost.XGBRegressor()
        model_types_to_train.append("xgboost")
    except Exception as e:
        print(f"XGBoost not available: {e}")

    results = {"feature_names": feature_names}

    for model_type in model_types_to_train:
        print("\n" + "=" * 60)
        model, cv_results = train_with_cv(
            X, y, groups, feature_names,
            model_type=model_type,
            n_folds=5
        )

        # Save model
        model_path = OUTPUT_DIR / f"{model_type}_model.pkl"
        model.save(model_path)
        print(f"\n{model_type} model saved to {model_path}")

        results[model_type] = cv_results

    # Save CV results
    results_path = OUTPUT_DIR / "cv_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nCV results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Training complete!")
    for model_type in model_types_to_train:
        if model_type in results:
            print(f"{model_type} CV MSE: {results[model_type]['mean_mse']:.6f}")


if __name__ == "__main__":
    main()
