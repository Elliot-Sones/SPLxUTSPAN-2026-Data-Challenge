"""
Prediction script for SPLxUTSPAN 2026 Data Challenge.

Generates submission.csv for test data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import pickle

from data_loader import (
    iterate_shots, load_metadata, get_keypoint_columns,
    load_scalers, scale_targets, DATA_DIR
)
from feature_engineering import (
    init_keypoint_mapping, extract_all_features
)
from models.baseline import XGBoostBaseline, LightGBMBaseline, SklearnBaseline


OUTPUT_DIR = Path(__file__).parent.parent / "output"


def extract_test_features(
    tiers: List[int] = [1, 2, 3],
    cache_file: Optional[Path] = None
) -> tuple:
    """Extract features for test data."""
    # Check cache
    if cache_file and cache_file.exists():
        print(f"Loading cached test features from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["X"], cached["feature_names"], cached["meta"]

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Load metadata
    meta = load_metadata(train=False)
    n_shots = len(meta)

    print(f"Extracting features for {n_shots} test shots...")

    all_features = []
    processed = 0

    for metadata, timeseries in iterate_shots(train=False, chunk_size=20):
        features = extract_all_features(
            timeseries,
            participant_id=metadata["participant_id"],
            tiers=tiers
        )
        all_features.append(features)

        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed}/{n_shots}")

    print(f"Processed {processed} test shots")

    # Convert to array
    feature_names = sorted(all_features[0].keys())
    X = np.array([
        [f.get(name, np.nan) for name in feature_names]
        for f in all_features
    ], dtype=np.float32)

    # Handle NaN values
    for i in range(X.shape[1]):
        col = X[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            X[nan_mask, i] = median_val

    # Cache
    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump({
                "X": X,
                "feature_names": feature_names,
                "meta": meta
            }, f)

    return X, feature_names, meta


def generate_submission(
    model_path: Path,
    output_path: Path,
    model_type: str = "xgboost"
) -> pd.DataFrame:
    """
    Generate submission file using trained model.

    Args:
        model_path: Path to saved model
        output_path: Path for output submission.csv
        model_type: "xgboost" or "lightgbm"

    Returns:
        Submission DataFrame
    """
    print(f"Loading model from {model_path}")
    if model_type == "xgboost":
        model = XGBoostBaseline.load(model_path)
    elif model_type == "lightgbm":
        model = LightGBMBaseline.load(model_path)
    else:  # sklearn
        model = SklearnBaseline.load(model_path)

    # Extract test features
    cache_file = OUTPUT_DIR / "features_test.pkl"
    X_test, feature_names, meta = extract_test_features(cache_file=cache_file)

    print(f"Test set: {X_test.shape[0]} shots, {X_test.shape[1]} features")

    # Predict
    print("Generating predictions...")
    y_pred = model.predict(X_test)

    # Load scalers and scale predictions
    print("Scaling predictions...")
    scalers = load_scalers()

    y_scaled = np.zeros_like(y_pred)
    target_names = ["angle", "depth", "left_right"]
    for i, name in enumerate(target_names):
        y_scaled[:, i] = scalers[name].transform(y_pred[:, i].reshape(-1, 1)).ravel()

    # Clip to [0, 1] range
    y_scaled = np.clip(y_scaled, 0, 1)

    # Create submission DataFrame (use scaled_ prefix as per template)
    submission = pd.DataFrame({
        "id": meta["id"],
        "scaled_angle": y_scaled[:, 0],
        "scaled_depth": y_scaled[:, 1],
        "scaled_left_right": y_scaled[:, 2]
    })

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    # Print summary
    print("\nSubmission summary:")
    print(submission.describe())

    return submission


def ensemble_predictions(
    model_paths: List[Path],
    model_types: List[str],
    weights: Optional[List[float]] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate ensemble predictions from multiple models.

    Args:
        model_paths: List of model file paths
        model_types: List of model types ("xgboost" or "lightgbm")
        weights: Optional weights for each model (default: equal)
        output_path: Optional path to save submission

    Returns:
        Submission DataFrame
    """
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)

    assert len(model_paths) == len(model_types) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

    # Extract test features (cached)
    cache_file = OUTPUT_DIR / "features_test.pkl"
    X_test, feature_names, meta = extract_test_features(cache_file=cache_file)

    # Collect predictions from all models
    all_predictions = []

    for model_path, model_type in zip(model_paths, model_types):
        print(f"Loading {model_type} from {model_path}")
        if model_type == "xgboost":
            model = XGBoostBaseline.load(model_path)
        elif model_type == "lightgbm":
            model = LightGBMBaseline.load(model_path)
        else:  # sklearn
            model = SklearnBaseline.load(model_path)

        y_pred = model.predict(X_test)
        all_predictions.append(y_pred)

    # Weighted average
    y_ensemble = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, weights):
        y_ensemble += weight * pred

    # Scale predictions
    scalers = load_scalers()
    y_scaled = np.zeros_like(y_ensemble)
    target_names = ["angle", "depth", "left_right"]
    for i, name in enumerate(target_names):
        y_scaled[:, i] = scalers[name].transform(y_ensemble[:, i].reshape(-1, 1)).ravel()

    y_scaled = np.clip(y_scaled, 0, 1)

    # Create submission (use scaled_ prefix as per template)
    submission = pd.DataFrame({
        "id": meta["id"],
        "scaled_angle": y_scaled[:, 0],
        "scaled_depth": y_scaled[:, 1],
        "scaled_left_right": y_scaled[:, 2]
    })

    if output_path:
        submission.to_csv(output_path, index=False)
        print(f"Ensemble submission saved to {output_path}")

    return submission


def main():
    """Generate submission from best model."""
    print("SPLxUTSPAN 2026 - Generating Submission")
    print("=" * 50)

    # Check for trained models
    sklearn_path = OUTPUT_DIR / "sklearn_model.pkl"
    lgb_path = OUTPUT_DIR / "lightgbm_model.pkl"
    xgb_path = OUTPUT_DIR / "xgboost_model.pkl"

    # Collect available models
    available_models = []
    if sklearn_path.exists():
        available_models.append((sklearn_path, "sklearn"))
    if lgb_path.exists():
        available_models.append((lgb_path, "lightgbm"))
    if xgb_path.exists():
        available_models.append((xgb_path, "xgboost"))

    if len(available_models) == 0:
        print("No trained models found. Run train.py first.")
        return

    if len(available_models) >= 2:
        # Ensemble available models
        model_paths = [m[0] for m in available_models]
        model_types = [m[1] for m in available_models]
        weights = [1.0 / len(available_models)] * len(available_models)

        print(f"Using ensemble of {model_types}")
        submission = ensemble_predictions(
            model_paths=model_paths,
            model_types=model_types,
            weights=weights,
            output_path=OUTPUT_DIR / "submission.csv"
        )
    else:
        # Single model
        model_path, model_type = available_models[0]
        print(f"Using single model: {model_type}")
        submission = generate_submission(
            model_path,
            OUTPUT_DIR / "submission.csv",
            model_type=model_type
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
