"""
Hyperparameter optimization for angle-specific model.

Uses Bayesian optimization (Optuna) to find optimal hyperparameters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: uv pip install optuna")


def optimize_xgboost_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 100
) -> Dict:
    """
    Use Optuna to find optimal XGBoost hyperparameters.

    Args:
        X: Feature matrix
        y: Target (angle)
        groups: Participant IDs for GroupKFold
        n_trials: Number of optimization trials

    Returns:
        Best hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not installed")

    def objective(trial):
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "tree_method": "hist",
            "early_stopping_rounds": 20
        }

        # Cross-validation
        gkf = GroupKFold(n_splits=5)
        val_mses = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            val_mses.append(mse)

        return np.mean(val_mses)

    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest MSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def optimize_lightgbm_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 100
) -> Dict:
    """Use Optuna to find optimal LightGBM hyperparameters."""
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not installed")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "verbose": -1
        }

        gkf = GroupKFold(n_splits=5)
        val_mses = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            val_mses.append(mse)

        return np.mean(val_mses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest MSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def grid_search_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray
) -> Dict:
    """
    Fallback grid search for XGBoost (when Optuna not available).

    Tests a small grid of promising hyperparameters.
    """
    from itertools import product

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "reg_lambda": [5.0, 10.0, 20.0],
        "min_child_weight": [3, 5, 7]
    }

    # Fixed params
    fixed_params = {
        "n_estimators": 300,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "random_state": 42,
        "tree_method": "hist"
    }

    best_mse = float("inf")
    best_params = None

    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    total_combos = np.prod([len(v) for v in values])
    print(f"Testing {total_combos} parameter combinations...")

    for i, combo in enumerate(product(*values)):
        params = {**fixed_params}
        for k, v in zip(keys, combo):
            params[k] = v

        # Cross-validation
        gkf = GroupKFold(n_splits=5)
        val_mses = []

        # Add early stopping to params
        params["early_stopping_rounds"] = 20

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            val_mses.append(mse)

        mean_mse = np.mean(val_mses)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total_combos}] Current best MSE: {best_mse:.6f}")

        if mean_mse < best_mse:
            best_mse = mean_mse
            best_params = params.copy()

    print(f"\nBest MSE: {best_mse:.6f}")
    print(f"Best params: {best_params}")

    return best_params


def main():
    """Main optimization pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize angle model hyperparameters")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"],
                        help="Model type to optimize")
    parser.add_argument("--method", type=str, default="bayesian",
                        choices=["bayesian", "grid"],
                        help="Optimization method")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of trials for Bayesian optimization")
    parser.add_argument("--n-features", type=int, default=100,
                        help="Number of features to select")
    parser.add_argument("--output", type=str, default="models/angle_specific/best_params.pkl",
                        help="Output path for best parameters")
    args = parser.parse_args()

    # Load features (reuse train_angle_model logic)
    from train_angle_model import load_all_features

    print("Loading features...")
    df, y, groups = load_all_features(
        use_physics=True,
        use_engineering=True,
        use_angle_specific=True
    )

    # Prepare features
    exclude_cols = ["participant_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)

    # Feature selection
    print(f"\nSelecting top {args.n_features} features...")
    feature_selector = SelectKBest(f_regression, k=min(args.n_features, X.shape[1]))
    X_selected = feature_selector.fit_transform(X, y)
    print(f"Selected features shape: {X_selected.shape}")

    # Optimize
    print(f"\nOptimizing {args.model} hyperparameters using {args.method} method...")

    if args.method == "bayesian":
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Falling back to grid search.")
            best_params = grid_search_xgboost(X_selected, y, groups)
        else:
            if args.model == "xgboost":
                best_params = optimize_xgboost_hyperparams(
                    X_selected, y, groups, n_trials=args.n_trials
                )
            elif args.model == "lightgbm":
                best_params = optimize_lightgbm_hyperparams(
                    X_selected, y, groups, n_trials=args.n_trials
                )
    elif args.method == "grid":
        if args.model == "xgboost":
            best_params = grid_search_xgboost(X_selected, y, groups)
        else:
            raise ValueError("Grid search only implemented for XGBoost")

    # Save best parameters
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(best_params, f)

    print(f"\nSaved best parameters to {output_path}")


if __name__ == "__main__":
    main()
