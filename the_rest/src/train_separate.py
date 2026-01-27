"""
Train separate models per target for SPLxUTSPAN 2026 Data Challenge.

Each target (angle, depth, left_right) gets its own model with optimized hyperparameters.
This allows for target-specific feature selection and tuning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import pickle
import time

try:
    from data_loader import (
        iterate_shots, load_metadata, get_keypoint_columns,
        load_scalers, DATA_DIR, TARGET_COLS
    )
except ImportError:
    from src.data_loader import (
        iterate_shots, load_metadata, get_keypoint_columns,
        load_scalers, DATA_DIR, TARGET_COLS
    )


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class SeparateTargetModels:
    """Wrapper for training separate models per target."""

    def __init__(self, model_class, model_params: Dict = None):
        """
        Args:
            model_class: Class to use for individual models (e.g., LGBMRegressor)
            model_params: Parameters to pass to each model
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.models = {}
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """
        Fit separate model for each target.

        Args:
            X: (n_samples, n_features)
            y: (n_samples, 3) for angle, depth, left_right
            feature_names: Optional feature names
        """
        self.feature_names_ = feature_names
        target_names = ["angle", "depth", "left_right"]

        for i, target in enumerate(target_names):
            print(f"  Training {target} model...")
            model = self.model_class(**self.model_params)
            model.fit(X, y[:, i])
            self.models[target] = model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all targets."""
        target_names = ["angle", "depth", "left_right"]
        predictions = np.zeros((X.shape[0], 3))

        for i, target in enumerate(target_names):
            predictions[:, i] = self.models[target].predict(X)

        return predictions

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance for each target."""
        importances = {}
        for target, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[target] = model.feature_importances_
        return importances

    def save(self, filepath: Path):
        """Save models to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "SeparateTargetModels":
        """Load models from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


def extract_features_with_extractor(
    extractor_func: Callable,
    train: bool = True,
    smooth: bool = False,
    max_shots: Optional[int] = None,
    cache_file: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Extract features using a custom extractor function.

    Args:
        extractor_func: Function that takes (timeseries, participant_id, smooth) and returns dict
        train: Whether to use training data
        smooth: Whether to apply smoothing
        max_shots: Maximum shots to process
        cache_file: Optional cache file path

    Returns:
        X, y, feature_names, meta
    """
    # Check cache
    if cache_file and cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["X"], cached["y"], cached["feature_names"], cached["meta"]

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()

    # Import and initialize depending on extractor type
    try:
        from physics_features import init_keypoint_mapping
    except ImportError:
        from src.physics_features import init_keypoint_mapping
    init_keypoint_mapping(keypoint_cols)

    # Load metadata
    meta = load_metadata(train)
    n_shots = len(meta) if max_shots is None else min(max_shots, len(meta))

    print(f"Extracting features for {n_shots} shots...")

    # Extract features
    all_features = []
    all_targets = []
    processed = 0

    start_time = time.time()
    for metadata, timeseries in iterate_shots(train, chunk_size=20):
        if processed >= n_shots:
            break

        features = extractor_func(
            timeseries,
            participant_id=metadata["participant_id"],
            smooth=smooth
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

    # Handle NaN values
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


def train_separate_models_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    model_type: str = "lightgbm",
    n_folds: int = 5,
) -> Tuple[SeparateTargetModels, Dict]:
    """
    Train separate models per target with GroupKFold CV.

    Args:
        X: Feature matrix
        y: Target matrix
        groups: Participant IDs
        feature_names: Feature names
        model_type: "lightgbm" or "xgboost"
        n_folds: Number of CV folds

    Returns:
        model: Trained SeparateTargetModels
        cv_results: CV metrics
    """
    print(f"\nTraining separate {model_type} models with {n_folds}-fold GroupKFold CV...")

    # Get model class and params
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
            model_class = lgb.LGBMRegressor
            model_params = {
                "n_estimators": 500,
                "num_leaves": 20,
                "learning_rate": 0.02,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbose": -1,
            }
        except ImportError:
            raise ImportError("LightGBM not installed")
    elif model_type == "xgboost":
        try:
            import xgboost as xgb
            model_class = xgb.XGBRegressor
            model_params = {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.02,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "tree_method": "hist",
            }
        except ImportError:
            raise ImportError("XGBoost not installed")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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

        # Train separate models
        model = SeparateTargetModels(model_class, model_params)
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
    print(f"CV Results (separate {model_type}):")
    print(f"  Mean MSE: {cv_results['mean_mse']:.6f} +/- {cv_results['std_mse']:.6f}")
    for name, stats in cv_results["per_target_mses"].items():
        print(f"  {name}: {stats['mean']:.6f} +/- {stats['std']:.6f}")

    # Train final model on all data
    print(f"\nTraining final model on all data...")
    final_model = SeparateTargetModels(model_class, model_params)
    final_model.fit(X, y, feature_names)

    # Feature importance
    importances = final_model.get_feature_importance()
    print("\nTop 10 features per target:")
    for target, imp in importances.items():
        indices = np.argsort(imp)[::-1][:10]
        print(f"\n{target}:")
        for idx in indices:
            print(f"  {feature_names[idx]}: {imp[idx]:.4f}")

    return final_model, cv_results


def compute_scaled_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute scaled MSE matching competition metric.

    The competition scales each target to [0, 1] before computing MSE.
    """
    scalers = load_scalers()

    y_true_scaled = np.zeros_like(y_true)
    y_pred_scaled = np.zeros_like(y_pred)

    for i, target in enumerate(TARGET_COLS):
        y_true_scaled[:, i] = scalers[target].transform(y_true[:, i].reshape(-1, 1)).ravel()
        y_pred_scaled[:, i] = scalers[target].transform(y_pred[:, i].reshape(-1, 1)).ravel()

    return mean_squared_error(y_true_scaled, y_pred_scaled)


def main():
    """Quick test of separate model training."""
    print("Testing separate model training with physics features...")

    try:
        from physics_features import extract_physics_features, init_keypoint_mapping
    except ImportError:
        from src.physics_features import extract_physics_features, init_keypoint_mapping

    # Initialize
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Extract physics features
    cache_file = OUTPUT_DIR / "features_physics_test.pkl"
    X, y, feature_names, meta = extract_features_with_extractor(
        extract_physics_features,
        train=True,
        smooth=False,
        max_shots=50,  # Quick test
        cache_file=cache_file,
    )

    groups = meta["participant_id"].values

    print(f"\nDataset: {X.shape[0]} shots, {X.shape[1]} features")

    # Train with CV
    model, cv_results = train_separate_models_with_cv(
        X, y, groups, feature_names,
        model_type="lightgbm",
        n_folds=5,
    )

    # Compute scaled MSE
    oof_preds = cv_results["oof_predictions"]
    scaled_mse = compute_scaled_mse(y, oof_preds)
    print(f"\nScaled MSE (competition metric): {scaled_mse:.6f}")

    print("\nSeparate model training test complete!")


if __name__ == "__main__":
    main()
