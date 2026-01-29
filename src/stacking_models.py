"""
Stacking and ensemble models for the overnight experiment system.

Implements:
1. StackingRegressor: Level 1 models + meta learner
2. BlendingEnsemble: Weighted average of models
3. OptimizedBlend: Scipy-optimized weights
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble with cross-validated base model predictions.

    Level 1: Base models trained on folds, predictions collected OOF
    Level 2: Meta learner trained on OOF predictions
    """

    def __init__(
        self,
        base_models: List[Tuple[str, BaseEstimator]],
        meta_model: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        use_original_features: bool = False,
    ):
        """
        Args:
            base_models: List of (name, model) tuples
            meta_model: Meta learner (default: RidgeCV)
            n_folds: Number of CV folds for OOF predictions
            use_original_features: Include original features in meta input
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model else RidgeCV(alphas=np.logspace(-3, 3, 50))
        self.n_folds = n_folds
        self.use_original_features = use_original_features

        # Fitted models
        self._fitted_base_models = None
        self._fitted_meta_model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> "StackingRegressor":
        """
        Fit the stacking ensemble.

        Args:
            X: Training features
            y: Training targets
            groups: Group labels for GroupKFold (e.g., participant_id)
        """
        n_samples = X.shape[0]
        n_base = len(self.base_models)

        # OOF predictions for meta learner
        oof_predictions = np.zeros((n_samples, n_base))

        # Store fitted models per fold
        self._fitted_base_models = {name: [] for name, _ in self.base_models}

        # Cross-validation
        if groups is not None:
            cv = GroupKFold(n_splits=self.n_folds)
            splits = cv.split(X, y, groups)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            splits = cv.split(X, y)

        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for i, (name, model) in enumerate(self.base_models):
                # Clone and fit
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)

                # Store fitted model
                self._fitted_base_models[name].append(model_clone)

                # OOF predictions
                oof_predictions[val_idx, i] = model_clone.predict(X_val)

        # Prepare meta features
        if self.use_original_features:
            meta_X = np.hstack([X, oof_predictions])
        else:
            meta_X = oof_predictions

        # Fit meta learner
        self._fitted_meta_model = clone(self.meta_model)
        self._fitted_meta_model.fit(meta_X, y)

        # Refit base models on full data for prediction
        self._final_base_models = {}
        for name, model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self._final_base_models[name] = model_clone

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the stacking ensemble."""
        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, (name, _) in enumerate(self.base_models):
            base_predictions[:, i] = self._final_base_models[name].predict(X)

        # Prepare meta features
        if self.use_original_features:
            meta_X = np.hstack([X, base_predictions])
        else:
            meta_X = base_predictions

        # Meta learner prediction
        return self._fitted_meta_model.predict(meta_X)

    def get_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get OOF predictions for analysis (re-runs CV)."""
        n_samples = X.shape[0]
        n_base = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_base))

        if groups is not None:
            cv = GroupKFold(n_splits=self.n_folds)
            splits = cv.split(X, y, groups)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            splits = cv.split(X, y)

        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for i, (name, model) in enumerate(self.base_models):
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                oof_predictions[val_idx, i] = model_clone.predict(X_val)

        return oof_predictions


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """
    Simple weighted average ensemble.

    Models are trained independently, predictions are blended.
    """

    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            models: List of (name, model) tuples
            weights: Optional weights (default: equal weights)
        """
        self.models = models
        self.weights = weights
        self._fitted_models = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlendingEnsemble":
        """Fit all models."""
        for name, model in self.models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self._fitted_models[name] = model_clone
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate blended predictions."""
        predictions = []
        for name, _ in self.models:
            predictions.append(self._fitted_models[name].predict(X))

        predictions = np.array(predictions)  # (n_models, n_samples)

        if self.weights is None:
            return np.mean(predictions, axis=0)
        else:
            weights = np.array(self.weights).reshape(-1, 1)
            return np.sum(predictions * weights, axis=0)


class OptimizedBlend(BaseEstimator, RegressorMixin):
    """
    Ensemble with scipy-optimized weights.

    Finds optimal weights by minimizing validation MSE.
    """

    def __init__(
        self,
        models: List[Tuple[str, BaseEstimator]],
        n_folds: int = 5,
    ):
        self.models = models
        self.n_folds = n_folds
        self._fitted_models = {}
        self._optimal_weights = None

    def _get_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get OOF predictions for all models."""
        n_samples = X.shape[0]
        n_models = len(self.models)
        oof_predictions = np.zeros((n_samples, n_models))

        if groups is not None:
            cv = GroupKFold(n_splits=self.n_folds)
            splits = cv.split(X, y, groups)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            splits = cv.split(X, y)

        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for i, (name, model) in enumerate(self.models):
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                oof_predictions[val_idx, i] = model_clone.predict(X_val)

        return oof_predictions

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> "OptimizedBlend":
        """Fit models and find optimal weights."""
        # Get OOF predictions
        oof_predictions = self._get_oof_predictions(X, y, groups)

        # Optimize weights
        n_models = len(self.models)

        def objective(weights):
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

            # Blend predictions
            blended = np.dot(oof_predictions, weights)

            # MSE
            return mean_squared_error(y, blended)

        # Initial weights (equal)
        x0 = np.ones(n_models) / n_models

        # Constraints: weights sum to 1, all positive
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        self._optimal_weights = result.x / result.x.sum()

        # Fit all models on full data
        for name, model in self.models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self._fitted_models[name] = model_clone

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate optimally blended predictions."""
        predictions = []
        for name, _ in self.models:
            predictions.append(self._fitted_models[name].predict(X))

        predictions = np.array(predictions)  # (n_models, n_samples)
        return np.dot(predictions.T, self._optimal_weights)

    def get_weights(self) -> Dict[str, float]:
        """Get the optimized weights."""
        return {name: w for (name, _), w in zip(self.models, self._optimal_weights)}


class MultiTargetStacker:
    """
    Stacking ensemble that handles multi-target prediction.

    Trains separate stackers for each target.
    """

    def __init__(
        self,
        base_models: List[Tuple[str, BaseEstimator]],
        meta_model: Optional[BaseEstimator] = None,
        targets: List[str] = ["angle", "depth", "left_right"],
        n_folds: int = 5,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.targets = targets
        self.n_folds = n_folds

        self._stackers = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> "MultiTargetStacker":
        """
        Fit separate stackers for each target.

        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples, n_targets)
            groups: Optional group labels
        """
        for i, target in enumerate(self.targets):
            stacker = StackingRegressor(
                base_models=[(name, clone(model)) for name, model in self.base_models],
                meta_model=clone(self.meta_model) if self.meta_model else None,
                n_folds=self.n_folds,
            )
            stacker.fit(X, y[:, i], groups)
            self._stackers[target] = stacker

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for all targets."""
        predictions = np.zeros((X.shape[0], len(self.targets)))

        for i, target in enumerate(self.targets):
            predictions[:, i] = self._stackers[target].predict(X)

        return predictions


def create_default_stacking_ensemble(include_dl: bool = False):
    """
    Create a default stacking ensemble with common models.

    Args:
        include_dl: Whether to include DL model predictions

    Returns:
        List of (name, model) tuples for base models
    """
    try:
        import lightgbm as lgb
        lgbm_available = True
    except ImportError:
        lgbm_available = False

    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        xgb_available = False

    try:
        from catboost import CatBoostRegressor
        catboost_available = True
    except ImportError:
        catboost_available = False

    base_models = []

    if lgbm_available:
        base_models.append((
            "lgbm",
            lgb.LGBMRegressor(
                n_estimators=300,
                num_leaves=20,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_lambda=1.0,
                verbose=-1,
                n_jobs=-1,
            )
        ))

    if xgb_available:
        base_models.append((
            "xgb",
            xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.02,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_lambda=1.0,
                verbosity=0,
                n_jobs=-1,
            )
        ))

    if catboost_available:
        base_models.append((
            "catboost",
            CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                verbose=False,
            )
        ))

    # Always include Ridge as baseline
    base_models.append((
        "ridge",
        Ridge(alpha=1.0)
    ))

    return base_models


if __name__ == "__main__":
    print("Testing stacking models...")

    # Generate test data
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    groups = np.repeat(np.arange(5), 20)

    # Test StackingRegressor
    print("\n1. StackingRegressor:")
    base_models = create_default_stacking_ensemble()
    print(f"   Base models: {[name for name, _ in base_models]}")

    stacker = StackingRegressor(base_models, n_folds=5)
    stacker.fit(X, y, groups)
    preds = stacker.predict(X)
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Train MSE: {mean_squared_error(y, preds):.4f}")

    # Test BlendingEnsemble
    print("\n2. BlendingEnsemble:")
    blender = BlendingEnsemble(base_models)
    blender.fit(X, y)
    preds = blender.predict(X)
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Train MSE: {mean_squared_error(y, preds):.4f}")

    # Test OptimizedBlend
    print("\n3. OptimizedBlend:")
    opt_blend = OptimizedBlend(base_models, n_folds=5)
    opt_blend.fit(X, y, groups)
    preds = opt_blend.predict(X)
    print(f"   Predictions shape: {preds.shape}")
    print(f"   Train MSE: {mean_squared_error(y, preds):.4f}")
    print(f"   Optimal weights: {opt_blend.get_weights()}")

    # Test MultiTargetStacker
    print("\n4. MultiTargetStacker:")
    y_multi = np.random.randn(n_samples, 3)
    mt_stacker = MultiTargetStacker(base_models, n_folds=5)
    mt_stacker.fit(X, y_multi, groups)
    preds = mt_stacker.predict(X)
    print(f"   Predictions shape: {preds.shape}")

    print("\nAll stacking model tests passed!")
