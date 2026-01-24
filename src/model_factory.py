"""
Model factory for exhaustive grid search.

Defines 8 different model configurations (M1-M8) for testing.
All models are gradient boosting or linear - no deep learning.
"""

from typing import Dict, Any, Tuple, Type
import numpy as np


# Model definitions
MODEL_CONFIGS = {
    "M1": "LightGBM_default",
    "M2": "LightGBM_conservative",
    "M3": "XGBoost_default",
    "M4": "XGBoost_conservative",
    "M5": "CatBoost_default",
    "M6": "CatBoost_conservative",
    "M7": "RandomForest",
    "M8": "Ridge",
}


def get_model_config(model_id: str) -> Tuple[Type, Dict[str, Any]]:
    """
    Get model class and parameters for a given model ID.

    Args:
        model_id: One of M1-M8 or the full name

    Returns:
        (model_class, model_params)
    """
    configs = {
        # =================================================================
        # M1: LightGBM Default
        # =================================================================
        "M1": _get_lightgbm_default,
        "LightGBM_default": _get_lightgbm_default,

        # =================================================================
        # M2: LightGBM Conservative (less overfitting)
        # =================================================================
        "M2": _get_lightgbm_conservative,
        "LightGBM_conservative": _get_lightgbm_conservative,

        # =================================================================
        # M3: XGBoost Default
        # =================================================================
        "M3": _get_xgboost_default,
        "XGBoost_default": _get_xgboost_default,

        # =================================================================
        # M4: XGBoost Conservative
        # =================================================================
        "M4": _get_xgboost_conservative,
        "XGBoost_conservative": _get_xgboost_conservative,

        # =================================================================
        # M5: CatBoost Default
        # =================================================================
        "M5": _get_catboost_default,
        "CatBoost_default": _get_catboost_default,

        # =================================================================
        # M6: CatBoost Conservative
        # =================================================================
        "M6": _get_catboost_conservative,
        "CatBoost_conservative": _get_catboost_conservative,

        # =================================================================
        # M7: Random Forest
        # =================================================================
        "M7": _get_random_forest,
        "RandomForest": _get_random_forest,

        # =================================================================
        # M8: Ridge Regression (linear baseline)
        # =================================================================
        "M8": _get_ridge,
        "Ridge": _get_ridge,
    }

    if model_id not in configs:
        raise ValueError(f"Unknown model: {model_id}. Choose from {list(MODEL_CONFIGS.keys())}")

    return configs[model_id]()


def _get_lightgbm_default() -> Tuple[Type, Dict[str, Any]]:
    """LightGBM with default hyperparameters."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    params = {
        "n_estimators": 500,
        "num_leaves": 20,
        "learning_rate": 0.02,
        "max_depth": -1,  # No limit
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 10,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    return lgb.LGBMRegressor, params


def _get_lightgbm_conservative() -> Tuple[Type, Dict[str, Any]]:
    """LightGBM with conservative hyperparameters to reduce overfitting."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    params = {
        "n_estimators": 300,
        "num_leaves": 10,
        "learning_rate": 0.01,
        "max_depth": 5,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    return lgb.LGBMRegressor, params


def _get_xgboost_default() -> Tuple[Type, Dict[str, Any]]:
    """XGBoost with default hyperparameters."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }
    return xgb.XGBRegressor, params


def _get_xgboost_conservative() -> Tuple[Type, Dict[str, Any]]:
    """XGBoost with conservative hyperparameters."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_weight": 10,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }
    return xgb.XGBRegressor, params


def _get_catboost_default() -> Tuple[Type, Dict[str, Any]]:
    """CatBoost with default hyperparameters."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("CatBoost not installed. Run: pip install catboost")

    params = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    return CatBoostRegressor, params


def _get_catboost_conservative() -> Tuple[Type, Dict[str, Any]]:
    """CatBoost with conservative hyperparameters."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("CatBoost not installed. Run: pip install catboost")

    params = {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.01,
        "l2_leaf_reg": 10.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    return CatBoostRegressor, params


def _get_random_forest() -> Tuple[Type, Dict[str, Any]]:
    """Random Forest regressor."""
    from sklearn.ensemble import RandomForestRegressor

    params = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    return RandomForestRegressor, params


def _get_ridge() -> Tuple[Type, Dict[str, Any]]:
    """Ridge regression (linear baseline)."""
    from sklearn.linear_model import RidgeCV

    # RidgeCV automatically selects best alpha
    params = {
        "alphas": np.logspace(-3, 3, 50),
        "cv": 5,
    }
    return RidgeCV, params


# =============================================================================
# Model with custom hyperparameters (for Optuna tuning)
# =============================================================================

def get_model_with_params(model_id: str, custom_params: Dict[str, Any]) -> Tuple[Type, Dict[str, Any]]:
    """
    Get model class with custom parameters (for hyperparameter tuning).

    Args:
        model_id: Base model ID (M1-M8)
        custom_params: Custom parameters to override defaults

    Returns:
        (model_class, merged_params)
    """
    model_class, default_params = get_model_config(model_id)
    merged_params = {**default_params, **custom_params}
    return model_class, merged_params


def get_optuna_search_space(model_id: str) -> Dict[str, Any]:
    """
    Get Optuna search space for hyperparameter tuning.

    Args:
        model_id: Model ID (M1-M8)

    Returns:
        Dictionary describing search space for Optuna
    """
    if model_id in ["M1", "M2", "LightGBM_default", "LightGBM_conservative"]:
        return {
            "n_estimators": ("int", 100, 1000),
            "num_leaves": ("int", 5, 50),
            "learning_rate": ("float_log", 0.005, 0.1),
            "max_depth": ("int", 3, 12),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "reg_alpha": ("float_log", 1e-4, 10),
            "reg_lambda": ("float_log", 1e-4, 10),
            "min_child_samples": ("int", 5, 50),
        }
    elif model_id in ["M3", "M4", "XGBoost_default", "XGBoost_conservative"]:
        return {
            "n_estimators": ("int", 100, 1000),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 0.005, 0.1),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "reg_alpha": ("float_log", 1e-4, 10),
            "reg_lambda": ("float_log", 1e-4, 10),
            "min_child_weight": ("int", 1, 20),
        }
    elif model_id in ["M5", "M6", "CatBoost_default", "CatBoost_conservative"]:
        return {
            "iterations": ("int", 100, 1000),
            "depth": ("int", 3, 10),
            "learning_rate": ("float_log", 0.005, 0.1),
            "l2_leaf_reg": ("float_log", 0.1, 30),
        }
    elif model_id in ["M7", "RandomForest"]:
        return {
            "n_estimators": ("int", 100, 1000),
            "max_depth": ("int", 3, 20),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 20),
            "max_features": ("categorical", ["sqrt", "log2", 0.5, 0.7, 1.0]),
        }
    elif model_id in ["M8", "Ridge"]:
        # Ridge uses RidgeCV which auto-selects alpha
        return {}
    else:
        return {}


def sample_optuna_params(trial, model_id: str) -> Dict[str, Any]:
    """
    Sample hyperparameters using Optuna trial.

    Args:
        trial: Optuna trial object
        model_id: Model ID (M1-M8)

    Returns:
        Dictionary of sampled parameters
    """
    search_space = get_optuna_search_space(model_id)
    params = {}

    for param_name, spec in search_space.items():
        if spec[0] == "int":
            params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
        elif spec[0] == "float":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2])
        elif spec[0] == "float_log":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=True)
        elif spec[0] == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, spec[1])

    return params


def get_model_description(model_id: str) -> str:
    """Get description of a model configuration."""
    descriptions = {
        "M1": "LightGBM default: n_est=500, lr=0.02, leaves=20",
        "M2": "LightGBM conservative: n_est=300, lr=0.01, leaves=10",
        "M3": "XGBoost default: n_est=500, lr=0.02, depth=5",
        "M4": "XGBoost conservative: n_est=300, lr=0.01, depth=3",
        "M5": "CatBoost default: iter=500, lr=0.03, depth=6",
        "M6": "CatBoost conservative: iter=300, lr=0.01, depth=4",
        "M7": "Random Forest: n_est=500, depth=10",
        "M8": "Ridge: linear baseline with CV alpha selection",
    }
    return descriptions.get(model_id, "Unknown model")


def is_model_available(model_id: str) -> bool:
    """Check if a model's dependencies are installed."""
    try:
        get_model_config(model_id)
        return True
    except ImportError:
        return False


def get_available_models() -> Dict[str, str]:
    """Get dictionary of available models (dependencies installed)."""
    available = {}
    for model_id, name in MODEL_CONFIGS.items():
        if is_model_available(model_id):
            available[model_id] = name
    return available


if __name__ == "__main__":
    # Test model factory
    print("Testing model factory...")

    # Check available models
    available = get_available_models()
    print(f"\nAvailable models: {list(available.keys())}")

    # Test each available model
    for model_id in available.keys():
        try:
            model_class, params = get_model_config(model_id)
            print(f"\n{model_id}: {model_class.__name__}")
            print(f"  Description: {get_model_description(model_id)}")
            print(f"  Params: {list(params.keys())}")
        except Exception as e:
            print(f"\n{model_id}: ERROR - {e}")

    print("\nModel factory test complete!")
