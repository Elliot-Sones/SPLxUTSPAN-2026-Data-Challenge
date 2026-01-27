"""
Baseline gradient boosting models for SPLxUTSPAN 2026 Data Challenge.

Provides XGBoost and LightGBM wrappers for multi-output regression.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pickle
from pathlib import Path


class XGBoostBaseline(BaseEstimator, RegressorMixin):
    """XGBoost multi-output regressor for predicting angle, depth, left_right."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit the model.

        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples, 3) target matrix [angle, depth, left_right]
            feature_names: Optional list of feature names
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: uv pip install xgboost")

        self.feature_names_ = feature_names

        base_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",  # Faster training
        )

        self.model = MultiOutputRegressor(base_model, n_jobs=1)
        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance for each target."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        importances = {}
        target_names = ["angle", "depth", "left_right"]

        for i, estimator in enumerate(self.model.estimators_):
            importances[target_names[i]] = estimator.feature_importances_

        return importances

    def save(self, filepath: Path):
        """Save model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "XGBoostBaseline":
        """Load model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class LightGBMBaseline(BaseEstimator, RegressorMixin):
    """LightGBM multi-output regressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit the model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: uv pip install lightgbm")

        self.feature_names_ = feature_names

        base_model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

        self.model = MultiOutputRegressor(base_model, n_jobs=1)
        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance for each target."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        importances = {}
        target_names = ["angle", "depth", "left_right"]

        for i, estimator in enumerate(self.model.estimators_):
            importances[target_names[i]] = estimator.feature_importances_

        return importances

    def save(self, filepath: Path):
        """Save model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "LightGBMBaseline":
        """Load model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class SklearnBaseline(BaseEstimator, RegressorMixin):
    """Sklearn GradientBoostingRegressor - fallback when XGBoost/LightGBM unavailable."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.model = None
        self.feature_names_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit the model."""
        from sklearn.ensemble import GradientBoostingRegressor

        self.feature_names_ = feature_names

        base_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance for each target."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")

        importances = {}
        target_names = ["angle", "depth", "left_right"]

        for i, estimator in enumerate(self.model.estimators_):
            importances[target_names[i]] = estimator.feature_importances_

        return importances

    def save(self, filepath: Path):
        """Save model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "SklearnBaseline":
        """Load model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


def get_top_features(
    importances: Dict[str, np.ndarray],
    feature_names: List[str],
    top_k: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """Get top-k most important features for each target."""
    result = {}

    for target, imp in importances.items():
        indices = np.argsort(imp)[::-1][:top_k]
        result[target] = [(feature_names[i], imp[i]) for i in indices]

    return result
