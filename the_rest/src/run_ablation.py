"""
Ablation study for physics-based feature engineering.

Compares multiple approaches:
- A1: Physics-only, unsmoothed
- A2: Physics-only, smoothed (Savitzky-Golay)
- B1: Hybrid (physics + z-coords), unsmoothed
- B2: Hybrid (physics + z-coords), smoothed
- C: Current baseline (all 3365 features)

All experiments use separate models per target with 5-fold GroupKFold CV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import pickle
import time
import argparse
import json

try:
    from data_loader import (
        iterate_shots, load_metadata, get_keypoint_columns,
        load_scalers, DATA_DIR, TARGET_COLS
    )
    from train_separate import (
        SeparateTargetModels,
        extract_features_with_extractor,
        compute_scaled_mse,
    )
except ImportError:
    from src.data_loader import (
        iterate_shots, load_metadata, get_keypoint_columns,
        load_scalers, DATA_DIR, TARGET_COLS
    )
    from src.train_separate import (
        SeparateTargetModels,
        extract_features_with_extractor,
        compute_scaled_mse,
    )


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class ExperimentResult:
    """Container for experiment results."""

    def __init__(self, name: str):
        self.name = name
        self.n_features = 0
        self.raw_mse = 0.0
        self.raw_mse_std = 0.0
        self.scaled_mse = 0.0
        self.per_target_mse = {}
        self.feature_time = 0.0
        self.train_time = 0.0
        self.top_features = {}

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "n_features": self.n_features,
            "raw_mse": self.raw_mse,
            "raw_mse_std": self.raw_mse_std,
            "scaled_mse": self.scaled_mse,
            "per_target_mse": self.per_target_mse,
            "feature_time": self.feature_time,
            "train_time": self.train_time,
            "top_features": self.top_features,
        }


def run_experiment(
    name: str,
    extractor_func,
    smooth: bool,
    model_type: str = "lightgbm",
    n_folds: int = 5,
    cache_prefix: str = None,
) -> ExperimentResult:
    """
    Run a single ablation experiment.

    Args:
        name: Experiment name
        extractor_func: Feature extraction function
        smooth: Whether to apply smoothing
        model_type: Model type
        n_folds: CV folds
        cache_prefix: Prefix for cache file

    Returns:
        ExperimentResult with metrics
    """
    result = ExperimentResult(name)
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"Smoothing: {smooth}, Model: {model_type}")
    print("="*60)

    # Feature extraction
    cache_file = None
    if cache_prefix:
        smooth_suffix = "_smooth" if smooth else "_unsmooth"
        cache_file = OUTPUT_DIR / f"features_{cache_prefix}{smooth_suffix}.pkl"

    feature_start = time.time()
    X, y, feature_names, meta = extract_features_with_extractor(
        extractor_func,
        train=True,
        smooth=smooth,
        cache_file=cache_file,
    )
    result.feature_time = time.time() - feature_start
    result.n_features = X.shape[1]

    groups = meta["participant_id"].values

    print(f"\nDataset: {X.shape[0]} shots, {X.shape[1]} features")

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
                "n_jobs": -1,
            }
        except ImportError:
            print("LightGBM not available, falling back to sklearn")
            from sklearn.ensemble import GradientBoostingRegressor
            model_class = GradientBoostingRegressor
            model_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.03,
                "subsample": 0.7,
                "random_state": 42,
            }
    else:
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
                "n_jobs": -1,
            }
        except ImportError:
            print("XGBoost not available, falling back to sklearn")
            from sklearn.ensemble import GradientBoostingRegressor
            model_class = GradientBoostingRegressor
            model_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.03,
                "subsample": 0.7,
                "random_state": 42,
            }

    # Cross-validation
    gkf = GroupKFold(n_splits=n_folds)

    fold_mses = []
    fold_mses_per_target = {"angle": [], "depth": [], "left_right": []}
    oof_predictions = np.zeros_like(y)

    train_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        val_participant = np.unique(groups[val_idx])[0]
        print(f"\nFold {fold + 1}: Validating on participant {val_participant}")

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
        for i, tname in enumerate(target_names):
            target_mse = mean_squared_error(y_val[:, i], y_pred[:, i])
            fold_mses_per_target[tname].append(target_mse)
            print(f"  {tname} MSE: {target_mse:.6f}")

        print(f"  Total MSE: {mse:.6f}")

    result.train_time = time.time() - train_start

    # Store results
    result.raw_mse = np.mean(fold_mses)
    result.raw_mse_std = np.std(fold_mses)

    for tname, mses in fold_mses_per_target.items():
        result.per_target_mse[tname] = {
            "mean": np.mean(mses),
            "std": np.std(mses),
        }

    # Compute scaled MSE
    result.scaled_mse = compute_scaled_mse(y, oof_predictions)

    # Train final model for feature importance
    final_model = SeparateTargetModels(model_class, model_params)
    final_model.fit(X, y, feature_names)

    importances = final_model.get_feature_importance()
    for target, imp in importances.items():
        indices = np.argsort(imp)[::-1][:10]
        result.top_features[target] = [
            (feature_names[idx], float(imp[idx])) for idx in indices
        ]

    print(f"\n{'='*50}")
    print(f"Results for {name}:")
    print(f"  Features: {result.n_features}")
    print(f"  Raw MSE: {result.raw_mse:.6f} +/- {result.raw_mse_std:.6f}")
    print(f"  Scaled MSE: {result.scaled_mse:.6f}")
    print(f"  Feature extraction time: {result.feature_time:.1f}s")
    print(f"  Training time: {result.train_time:.1f}s")

    return result


def run_baseline_experiment(
    model_type: str = "lightgbm",
    n_folds: int = 5,
) -> ExperimentResult:
    """Run baseline experiment with current 3365 features."""
    try:
        from feature_engineering import init_keypoint_mapping, extract_all_features
    except ImportError:
        from src.feature_engineering import init_keypoint_mapping, extract_all_features

    result = ExperimentResult("C_baseline")
    print(f"\n{'='*60}")
    print("Running experiment: C_baseline (current 3365 features)")
    print("="*60)

    # Initialize
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Use existing cached features if available
    cache_file = OUTPUT_DIR / "features_train.pkl"

    feature_start = time.time()
    if cache_file.exists():
        print(f"Loading cached baseline features from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        X = cached["X"]
        y = cached["y"]
        feature_names = cached["feature_names"]
        meta = cached["meta"]
    else:
        # Extract features
        X, y, feature_names, meta = extract_features_with_extractor(
            extract_all_features,
            train=True,
            smooth=False,
            cache_file=cache_file,
        )

    result.feature_time = time.time() - feature_start
    result.n_features = X.shape[1]

    groups = meta["participant_id"].values
    print(f"\nDataset: {X.shape[0]} shots, {X.shape[1]} features")

    # Get model class
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
                "n_jobs": -1,
            }
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model_class = GradientBoostingRegressor
            model_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.03,
                "subsample": 0.7,
                "random_state": 42,
            }
    else:
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
                "n_jobs": -1,
            }
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model_class = GradientBoostingRegressor
            model_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.03,
                "subsample": 0.7,
                "random_state": 42,
            }

    # CV
    gkf = GroupKFold(n_splits=n_folds)

    fold_mses = []
    fold_mses_per_target = {"angle": [], "depth": [], "left_right": []}
    oof_predictions = np.zeros_like(y)

    train_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        val_participant = np.unique(groups[val_idx])[0]
        print(f"\nFold {fold + 1}: Validating on participant {val_participant}")

        model = SeparateTargetModels(model_class, model_params)
        model.fit(X_train, y_train, feature_names)

        y_pred = model.predict(X_val)
        oof_predictions[val_idx] = y_pred

        mse = mean_squared_error(y_val, y_pred)
        fold_mses.append(mse)

        for i, tname in enumerate(["angle", "depth", "left_right"]):
            target_mse = mean_squared_error(y_val[:, i], y_pred[:, i])
            fold_mses_per_target[tname].append(target_mse)
            print(f"  {tname} MSE: {target_mse:.6f}")

        print(f"  Total MSE: {mse:.6f}")

    result.train_time = time.time() - train_start

    result.raw_mse = np.mean(fold_mses)
    result.raw_mse_std = np.std(fold_mses)

    for tname, mses in fold_mses_per_target.items():
        result.per_target_mse[tname] = {
            "mean": np.mean(mses),
            "std": np.std(mses),
        }

    result.scaled_mse = compute_scaled_mse(y, oof_predictions)

    print(f"\n{'='*50}")
    print(f"Results for C_baseline:")
    print(f"  Features: {result.n_features}")
    print(f"  Raw MSE: {result.raw_mse:.6f} +/- {result.raw_mse_std:.6f}")
    print(f"  Scaled MSE: {result.scaled_mse:.6f}")

    return result


def print_results_table(results: List[ExperimentResult]):
    """Print a comparison table of all experiments."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)

    # Header
    print(f"\n{'Experiment':<20} {'Features':>10} {'Scaled MSE':>12} {'Raw MSE':>12} {'Angle':>10} {'Depth':>10} {'L/R':>10}")
    print("-"*80)

    # Sort by scaled MSE
    sorted_results = sorted(results, key=lambda r: r.scaled_mse)

    for r in sorted_results:
        angle_mse = r.per_target_mse.get("angle", {}).get("mean", 0)
        depth_mse = r.per_target_mse.get("depth", {}).get("mean", 0)
        lr_mse = r.per_target_mse.get("left_right", {}).get("mean", 0)

        print(f"{r.name:<20} {r.n_features:>10} {r.scaled_mse:>12.6f} {r.raw_mse:>12.6f} "
              f"{angle_mse:>10.6f} {depth_mse:>10.6f} {lr_mse:>10.6f}")

    print("-"*80)

    # Best result
    best = sorted_results[0]
    print(f"\nBest: {best.name} with scaled MSE = {best.scaled_mse:.6f}")

    # Comparison to baseline
    baseline = next((r for r in results if "baseline" in r.name.lower()), None)
    if baseline and best.name != baseline.name:
        improvement = (baseline.scaled_mse - best.scaled_mse) / baseline.scaled_mse * 100
        print(f"Improvement over baseline: {improvement:.1f}%")


def run_all_experiments(model_type: str = "lightgbm", n_folds: int = 5) -> List[ExperimentResult]:
    """Run all ablation experiments."""
    try:
        from physics_features import extract_physics_features, init_keypoint_mapping
        from hybrid_features import extract_hybrid_features
    except ImportError:
        from src.physics_features import extract_physics_features, init_keypoint_mapping
        from src.hybrid_features import extract_hybrid_features

    # Initialize keypoint mapping once
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    results = []

    # A1: Physics-only, unsmoothed
    results.append(run_experiment(
        name="A1_physics_unsmooth",
        extractor_func=extract_physics_features,
        smooth=False,
        model_type=model_type,
        n_folds=n_folds,
        cache_prefix="physics",
    ))

    # A2: Physics-only, smoothed
    results.append(run_experiment(
        name="A2_physics_smooth",
        extractor_func=extract_physics_features,
        smooth=True,
        model_type=model_type,
        n_folds=n_folds,
        cache_prefix="physics",
    ))

    # B1: Hybrid, unsmoothed
    results.append(run_experiment(
        name="B1_hybrid_unsmooth",
        extractor_func=extract_hybrid_features,
        smooth=False,
        model_type=model_type,
        n_folds=n_folds,
        cache_prefix="hybrid",
    ))

    # B2: Hybrid, smoothed
    results.append(run_experiment(
        name="B2_hybrid_smooth",
        extractor_func=extract_hybrid_features,
        smooth=True,
        model_type=model_type,
        n_folds=n_folds,
        cache_prefix="hybrid",
    ))

    # C: Baseline
    results.append(run_baseline_experiment(
        model_type=model_type,
        n_folds=n_folds,
    ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Which experiments to run: all, physics, hybrid, baseline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "xgboost"],
        help="Model type to use"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    args = parser.parse_args()

    print("Physics-Based Feature Engineering Ablation Study")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"CV Folds: {args.folds}")
    print(f"Experiments: {args.experiments}")

    try:
        from physics_features import init_keypoint_mapping
    except ImportError:
        from src.physics_features import init_keypoint_mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    results = []

    if args.experiments in ["all", "physics"]:
        try:
            from physics_features import extract_physics_features
        except ImportError:
            from src.physics_features import extract_physics_features

        results.append(run_experiment(
            name="A1_physics_unsmooth",
            extractor_func=extract_physics_features,
            smooth=False,
            model_type=args.model,
            n_folds=args.folds,
            cache_prefix="physics",
        ))

        results.append(run_experiment(
            name="A2_physics_smooth",
            extractor_func=extract_physics_features,
            smooth=True,
            model_type=args.model,
            n_folds=args.folds,
            cache_prefix="physics",
        ))

    if args.experiments in ["all", "hybrid"]:
        try:
            from hybrid_features import extract_hybrid_features
        except ImportError:
            from src.hybrid_features import extract_hybrid_features

        results.append(run_experiment(
            name="B1_hybrid_unsmooth",
            extractor_func=extract_hybrid_features,
            smooth=False,
            model_type=args.model,
            n_folds=args.folds,
            cache_prefix="hybrid",
        ))

        results.append(run_experiment(
            name="B2_hybrid_smooth",
            extractor_func=extract_hybrid_features,
            smooth=True,
            model_type=args.model,
            n_folds=args.folds,
            cache_prefix="hybrid",
        ))

    if args.experiments in ["all", "baseline"]:
        results.append(run_baseline_experiment(
            model_type=args.model,
            n_folds=args.folds,
        ))

    # Print results
    print_results_table(results)

    # Save results
    results_file = OUTPUT_DIR / "ablation_results.json"
    results_dict = {r.name: r.to_dict() for r in results}
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Save best model
    best = min(results, key=lambda r: r.scaled_mse)
    print(f"\nBest experiment: {best.name}")

    return results


if __name__ == "__main__":
    main()
