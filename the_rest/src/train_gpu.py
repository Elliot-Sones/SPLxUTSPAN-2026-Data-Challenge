"""
GPU Training script for SPLxUTSPAN 2026 Data Challenge.

Supports gradient boosting (XGBoost/LightGBM with GPU) and deep learning models.
Run on vast.ai or any GPU machine.

Usage:
    python train_gpu.py --model xgboost_gpu
    python train_gpu.py --model lightgbm_gpu
    python train_gpu.py --model cnn_lstm
    python train_gpu.py --model transformer
    python train_gpu.py --model all --tune
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import pickle
import time
import json

# Local imports
from data_loader import (
    iterate_shots, load_metadata, get_keypoint_columns,
    load_scalers, DATA_DIR
)
from feature_engineering import (
    init_keypoint_mapping, extract_all_features
)


OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_FRAMES = 240
NUM_FEATURES = 207


def load_raw_timeseries(train: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load raw time series data for deep learning.

    Returns:
        X: (n_samples, n_timesteps, n_features) array
        y: (n_samples, 3) targets
        meta: Metadata DataFrame
    """
    cache_file = OUTPUT_DIR / f"raw_timeseries_{'train' if train else 'test'}.pkl"

    if cache_file.exists():
        print(f"Loading cached raw timeseries from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["X"], cached["y"], cached["meta"]

    meta = load_metadata(train)
    n_shots = len(meta)

    print(f"Loading raw timeseries for {n_shots} shots...")

    X_list = []
    y_list = []

    for metadata, timeseries in iterate_shots(train, chunk_size=50):
        X_list.append(timeseries)
        if train:
            y_list.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if train else None

    # Handle NaN values - replace with 0
    X = np.nan_to_num(X, nan=0.0)

    print(f"Raw timeseries shape: {X.shape}")

    # Cache
    with open(cache_file, "wb") as f:
        pickle.dump({"X": X, "y": y, "meta": meta}, f)

    return X, y, meta


def load_features(train: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Load extracted features for gradient boosting."""
    cache_file = OUTPUT_DIR / f"features_{'train' if train else 'test'}.pkl"

    if cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["X"], cached["y"], cached["feature_names"], cached["meta"]

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    meta = load_metadata(train)
    n_shots = len(meta)

    print(f"Extracting features for {n_shots} shots...")

    all_features = []
    all_targets = []

    start_time = time.time()
    for metadata, timeseries in iterate_shots(train, chunk_size=50):
        features = extract_all_features(
            timeseries,
            participant_id=metadata["participant_id"],
            tiers=[1, 2, 3]
        )
        all_features.append(features)

        if train:
            all_targets.append([
                metadata["angle"],
                metadata["depth"],
                metadata["left_right"]
            ])

    elapsed = time.time() - start_time
    print(f"Feature extraction: {len(all_features)} shots in {elapsed:.1f}s")

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

    print(f"Feature matrix: {X.shape}")

    # Cache
    with open(cache_file, "wb") as f:
        pickle.dump({
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "meta": meta
        }, f)

    return X, y, feature_names, meta


def train_xgboost_gpu(X, y, groups, feature_names, n_folds=5, **params):
    """Train XGBoost with GPU acceleration."""
    import xgboost as xgb

    print("\nTraining XGBoost (GPU)...")

    default_params = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }
    default_params.update(params)

    gkf = GroupKFold(n_splits=n_folds)
    fold_mses = []
    fold_mses_per_target = {"angle": [], "depth": [], "left_right": []}
    oof_predictions = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        val_participant = np.unique(groups[val_idx])[0]
        print(f"\nFold {fold + 1}: Val participant {val_participant}")

        # Train separate model for each target
        preds = np.zeros_like(y_val)
        for i, target_name in enumerate(["angle", "depth", "left_right"]):
            model = xgb.XGBRegressor(**default_params)
            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                verbose=False
            )
            preds[:, i] = model.predict(X_val)

        oof_predictions[val_idx] = preds

        # Calculate MSE
        mse = mean_squared_error(y_val, preds)
        fold_mses.append(mse)

        for i, name in enumerate(["angle", "depth", "left_right"]):
            target_mse = mean_squared_error(y_val[:, i], preds[:, i])
            fold_mses_per_target[name].append(target_mse)
            print(f"  {name} MSE: {target_mse:.4f}")

        print(f"  Total MSE: {mse:.4f}")

    print(f"\nXGBoost GPU - Mean MSE: {np.mean(fold_mses):.4f} +/- {np.std(fold_mses):.4f}")

    return {
        "model": "xgboost_gpu",
        "mean_mse": np.mean(fold_mses),
        "std_mse": np.std(fold_mses),
        "per_target": fold_mses_per_target,
        "params": default_params,
    }


def train_lightgbm_gpu(X, y, groups, feature_names, n_folds=5, **params):
    """Train LightGBM with GPU acceleration."""
    import lightgbm as lgb

    print("\nTraining LightGBM (GPU)...")

    default_params = {
        "n_estimators": 1000,
        "num_leaves": 31,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "device": "gpu",
        "random_state": 42,
        "verbose": -1,
    }
    default_params.update(params)

    gkf = GroupKFold(n_splits=n_folds)
    fold_mses = []
    fold_mses_per_target = {"angle": [], "depth": [], "left_right": []}
    oof_predictions = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        val_participant = np.unique(groups[val_idx])[0]
        print(f"\nFold {fold + 1}: Val participant {val_participant}")

        # Train separate model for each target
        preds = np.zeros_like(y_val)
        for i, target_name in enumerate(["angle", "depth", "left_right"]):
            model = lgb.LGBMRegressor(**default_params)
            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
            )
            preds[:, i] = model.predict(X_val)

        oof_predictions[val_idx] = preds

        # Calculate MSE
        mse = mean_squared_error(y_val, preds)
        fold_mses.append(mse)

        for i, name in enumerate(["angle", "depth", "left_right"]):
            target_mse = mean_squared_error(y_val[:, i], preds[:, i])
            fold_mses_per_target[name].append(target_mse)
            print(f"  {name} MSE: {target_mse:.4f}")

        print(f"  Total MSE: {mse:.4f}")

    print(f"\nLightGBM GPU - Mean MSE: {np.mean(fold_mses):.4f} +/- {np.std(fold_mses):.4f}")

    return {
        "model": "lightgbm_gpu",
        "mean_mse": np.mean(fold_mses),
        "std_mse": np.std(fold_mses),
        "per_target": fold_mses_per_target,
        "params": default_params,
    }


def train_deep_learning(X, y, groups, model_type="cnn_lstm", n_folds=5, **params):
    """Train deep learning model with cross-validation."""
    from models.deep_learning import DeepLearningTrainer

    print(f"\nTraining {model_type} (GPU)...")

    gkf = GroupKFold(n_splits=n_folds)
    fold_mses = []
    fold_mses_per_target = {"angle": [], "depth": [], "left_right": []}
    oof_predictions = np.zeros((len(y), 3))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        val_participant = np.unique(groups[val_idx])[0]
        print(f"\nFold {fold + 1}: Val participant {val_participant}")

        trainer = DeepLearningTrainer(model_type=model_type, **params)
        trainer.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=200,
            batch_size=32,
            lr=1e-3,
            patience=20,
        )

        preds = trainer.predict(X_val)
        oof_predictions[val_idx] = preds

        # Calculate MSE
        mse = mean_squared_error(y_val, preds)
        fold_mses.append(mse)

        for i, name in enumerate(["angle", "depth", "left_right"]):
            target_mse = mean_squared_error(y_val[:, i], preds[:, i])
            fold_mses_per_target[name].append(target_mse)
            print(f"  {name} MSE: {target_mse:.4f}")

        print(f"  Total MSE: {mse:.4f}")

    print(f"\n{model_type} - Mean MSE: {np.mean(fold_mses):.4f} +/- {np.std(fold_mses):.4f}")

    return {
        "model": model_type,
        "mean_mse": np.mean(fold_mses),
        "std_mse": np.std(fold_mses),
        "per_target": fold_mses_per_target,
        "params": params,
    }


def hyperparameter_search(X, y, groups, model_type="xgboost_gpu", n_trials=50):
    """Optuna hyperparameter search."""
    import optuna

    print(f"\nHyperparameter search for {model_type} ({n_trials} trials)...")

    def objective(trial):
        if model_type == "xgboost_gpu":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            }
            result = train_xgboost_gpu(X, y, groups, None, n_folds=3, **params)
        elif model_type == "lightgbm_gpu":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            }
            result = train_lightgbm_gpu(X, y, groups, None, n_folds=3, **params)
        else:
            raise ValueError(f"Unknown model type for tuning: {model_type}")

        return result["mean_mse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: MSE = {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    return study.best_trial.params


def calculate_scaled_mse(mse_per_target: Dict) -> float:
    """Calculate scaled MSE (competition metric)."""
    # Scaling ranges from data_loader
    angle_range = 30  # [30, 60]
    depth_range = 42  # [-12, 30]
    lr_range = 32     # [-16, 16]

    angle_scaled = np.mean(mse_per_target["angle"]) / (angle_range ** 2)
    depth_scaled = np.mean(mse_per_target["depth"]) / (depth_range ** 2)
    lr_scaled = np.mean(mse_per_target["left_right"]) / (lr_range ** 2)

    return (angle_scaled + depth_scaled + lr_scaled) / 3


def main():
    parser = argparse.ArgumentParser(description="GPU Training for SPLxUTSPAN 2026")
    parser.add_argument("--model", type=str, default="xgboost_gpu",
                       choices=["xgboost_gpu", "lightgbm_gpu", "cnn_lstm", "transformer", "all"],
                       help="Model type to train")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=50, help="Number of tuning trials")
    args = parser.parse_args()

    print("=" * 60)
    print("SPLxUTSPAN 2026 - GPU Training")
    print("=" * 60)

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = []

    # Gradient boosting models use extracted features
    if args.model in ["xgboost_gpu", "lightgbm_gpu", "all"]:
        X, y, feature_names, meta = load_features(train=True)
        groups = meta["participant_id"].values

        print(f"\nFeature data: {X.shape[0]} shots, {X.shape[1]} features")

        if args.model in ["xgboost_gpu", "all"]:
            if args.tune:
                best_params = hyperparameter_search(X, y, groups, "xgboost_gpu", args.trials)
                result = train_xgboost_gpu(X, y, groups, feature_names, **best_params)
            else:
                result = train_xgboost_gpu(X, y, groups, feature_names)
            result["scaled_mse"] = calculate_scaled_mse(result["per_target"])
            results.append(result)

        if args.model in ["lightgbm_gpu", "all"]:
            if args.tune:
                best_params = hyperparameter_search(X, y, groups, "lightgbm_gpu", args.trials)
                result = train_lightgbm_gpu(X, y, groups, feature_names, **best_params)
            else:
                result = train_lightgbm_gpu(X, y, groups, feature_names)
            result["scaled_mse"] = calculate_scaled_mse(result["per_target"])
            results.append(result)

    # Deep learning models use raw time series
    if args.model in ["cnn_lstm", "transformer", "all"]:
        X, y, meta = load_raw_timeseries(train=True)
        groups = meta["participant_id"].values

        print(f"\nRaw data: {X.shape[0]} shots, {X.shape[1]} timesteps, {X.shape[2]} features")

        if args.model in ["cnn_lstm", "all"]:
            result = train_deep_learning(X, y, groups, "cnn_lstm")
            result["scaled_mse"] = calculate_scaled_mse(result["per_target"])
            results.append(result)

        if args.model in ["transformer", "all"]:
            result = train_deep_learning(X, y, groups, "transformer")
            result["scaled_mse"] = calculate_scaled_mse(result["per_target"])
            results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'MSE':<12} {'Scaled MSE':<12} {'vs Leader':<12}")
    print("-" * 56)

    leader_score = 0.008381
    for r in results:
        scaled = r["scaled_mse"]
        gap = scaled / leader_score
        print(f"{r['model']:<20} {r['mean_mse']:<12.4f} {scaled:<12.6f} {gap:.2f}x")

    # Save results
    results_file = OUTPUT_DIR / "gpu_training_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
