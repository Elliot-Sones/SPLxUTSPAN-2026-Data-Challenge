"""
Model definitions for data augmentation test.

7 models tested:
1. LightGBM
2. XGBoost
3. CatBoost
4. RandomForest
5. Ridge
6. k-NN (distribution shift detector)
7. MLP
"""

import numpy as np
from typing import Dict, Callable
from sklearn.multioutput import MultiOutputRegressor


def get_lightgbm_model():
    """LightGBM with default parameters."""
    import lightgbm as lgb

    base = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def get_xgboost_model():
    """XGBoost with default parameters."""
    import xgboost as xgb

    base = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def get_catboost_model():
    """CatBoost with default parameters using sklearn wrapper."""
    from sklearn.base import BaseEstimator, RegressorMixin
    from catboost import CatBoostRegressor as _CatBoostRegressor

    class CatBoostWrapper(BaseEstimator, RegressorMixin):
        """Sklearn-compatible CatBoost wrapper."""
        def __init__(self, iterations=100, depth=6, learning_rate=0.1,
                     random_state=42, verbose=0, thread_count=-1):
            self.iterations = iterations
            self.depth = depth
            self.learning_rate = learning_rate
            self.random_state = random_state
            self.verbose = verbose
            self.thread_count = thread_count

        def fit(self, X, y):
            self._model = _CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbose=self.verbose,
                thread_count=self.thread_count,
            )
            self._model.fit(X, y)
            return self

        def predict(self, X):
            return self._model.predict(X)

    base = CatBoostWrapper()
    return MultiOutputRegressor(base, n_jobs=1)


def get_randomforest_model():
    """RandomForest - reference baseline that achieved 0.010 MSE."""
    from sklearn.ensemble import RandomForestRegressor

    base = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def get_ridge_model():
    """Ridge regression - linear baseline."""
    from sklearn.linear_model import Ridge

    base = Ridge(alpha=1.0, random_state=42)
    return MultiOutputRegressor(base, n_jobs=1)


def get_knn_model():
    """k-NN - distribution shift detector."""
    from sklearn.neighbors import KNeighborsRegressor

    base = KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def get_mlp_model():
    """MLP - neural network baseline."""
    from sklearn.neural_network import MLPRegressor

    base = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def get_all_models() -> Dict[str, Callable]:
    """Return dict of model name -> model factory function."""
    return {
        "LightGBM": get_lightgbm_model,
        "XGBoost": get_xgboost_model,
        "CatBoost": get_catboost_model,
        "RandomForest": get_randomforest_model,
        "Ridge": get_ridge_model,
        "k-NN": get_knn_model,
        "MLP": get_mlp_model,
    }


if __name__ == "__main__":
    # Quick test that all models can be instantiated
    print("Testing model instantiation...")

    models = get_all_models()
    for name, factory in models.items():
        try:
            model = factory()
            print(f"  [OK] {name}: {type(model.estimator).__name__}")
        except ImportError as e:
            print(f"  [SKIP] {name}: {e}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print("\nModel instantiation test complete.")
