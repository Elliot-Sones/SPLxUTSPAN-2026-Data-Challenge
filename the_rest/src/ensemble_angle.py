"""
Ensemble of angle-specific models for robust prediction.

Trains multiple diverse models and combines them via weighted averaging.
Models in ensemble:
1. XGBoost with top 100 features
2. LightGBM with top 100 features
3. XGBoost with angle-specific features only
4. Ridge regression with polynomial features
5. (Optional) XGBoost with physics features only
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize


def train_diverse_models(
    X_all: np.ndarray,
    X_angle: np.ndarray,
    X_physics: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names_all: List[str],
    feature_names_angle: List[str],
    feature_names_physics: List[str]
) -> Dict:
    """
    Train ensemble of diverse models.

    Args:
        X_all: All features
        X_angle: Angle-specific features only
        X_physics: Physics features only
        y: Target (angle)
        groups: Participant IDs
        feature_names_*: Feature name lists

    Returns:
        Dict with trained models and CV results
    """
    print("Training ensemble of diverse models...")

    gkf = GroupKFold(n_splits=5)
    ensemble_models = {
        "xgb_top100": [],
        "lgb_top100": [],
        "xgb_angle_only": [],
        "ridge_poly": [],
        "xgb_physics_only": []
    }

    all_fold_predictions = {name: {"train": [], "val": [], "y_train": [], "y_val": []}
                            for name in ensemble_models.keys()}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")

        # =====================================================================
        # Model 1: XGBoost with top 100 features (from all features)
        # =====================================================================
        print("  Training XGBoost (top 100 features)...")
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Feature selection
        selector = SelectKBest(f_regression, k=min(100, X_train.shape[1]))
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_val_sel = selector.transform(X_val)

        model_xgb = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=10.0,
            min_child_weight=5,
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=20
        )
        model_xgb.fit(
            X_train_sel, y_train,
            eval_set=[(X_val_sel, y_val)],
            verbose=False
        )

        ensemble_models["xgb_top100"].append((model_xgb, selector))
        all_fold_predictions["xgb_top100"]["train"].append(model_xgb.predict(X_train_sel))
        all_fold_predictions["xgb_top100"]["val"].append(model_xgb.predict(X_val_sel))
        all_fold_predictions["xgb_top100"]["y_train"].append(y_train)
        all_fold_predictions["xgb_top100"]["y_val"].append(y_val)

        # =====================================================================
        # Model 2: LightGBM with top 100 features
        # =====================================================================
        print("  Training LightGBM (top 100 features)...")
        # Reuse same feature selection
        model_lgb = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=10.0,
            min_child_weight=5,
            random_state=42,
            verbose=-1
        )
        model_lgb.fit(
            X_train_sel, y_train,
            eval_set=[(X_val_sel, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False)]
        )

        ensemble_models["lgb_top100"].append((model_lgb, selector))
        all_fold_predictions["lgb_top100"]["train"].append(model_lgb.predict(X_train_sel))
        all_fold_predictions["lgb_top100"]["val"].append(model_lgb.predict(X_val_sel))
        all_fold_predictions["lgb_top100"]["y_train"].append(y_train)
        all_fold_predictions["lgb_top100"]["y_val"].append(y_val)

        # =====================================================================
        # Model 3: XGBoost with angle-specific features only
        # =====================================================================
        print("  Training XGBoost (angle-specific features)...")
        X_train_angle, X_val_angle = X_angle[train_idx], X_angle[val_idx]

        model_xgb_angle = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5.0,
            min_child_weight=3,
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=20
        )
        model_xgb_angle.fit(
            X_train_angle, y_train,
            eval_set=[(X_val_angle, y_val)],
            verbose=False
        )

        ensemble_models["xgb_angle_only"].append(model_xgb_angle)
        all_fold_predictions["xgb_angle_only"]["train"].append(model_xgb_angle.predict(X_train_angle))
        all_fold_predictions["xgb_angle_only"]["val"].append(model_xgb_angle.predict(X_val_angle))
        all_fold_predictions["xgb_angle_only"]["y_train"].append(y_train)
        all_fold_predictions["xgb_angle_only"]["y_val"].append(y_val)

        # =====================================================================
        # Model 4: Ridge regression with polynomial features
        # =====================================================================
        print("  Training Ridge (polynomial)...")
        model_ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2)),
            ("ridge", Ridge(alpha=10.0))
        ])
        model_ridge.fit(X_train_sel, y_train)

        ensemble_models["ridge_poly"].append((model_ridge, selector))
        all_fold_predictions["ridge_poly"]["train"].append(model_ridge.predict(X_train_sel))
        all_fold_predictions["ridge_poly"]["val"].append(model_ridge.predict(X_val_sel))
        all_fold_predictions["ridge_poly"]["y_train"].append(y_train)
        all_fold_predictions["ridge_poly"]["y_val"].append(y_val)

        # =====================================================================
        # Model 5: XGBoost with physics features only
        # =====================================================================
        print("  Training XGBoost (physics features)...")
        X_train_physics, X_val_physics = X_physics[train_idx], X_physics[val_idx]

        model_xgb_physics = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=5.0,
            min_child_weight=3,
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=20
        )
        model_xgb_physics.fit(
            X_train_physics, y_train,
            eval_set=[(X_val_physics, y_val)],
            verbose=False
        )

        ensemble_models["xgb_physics_only"].append(model_xgb_physics)
        all_fold_predictions["xgb_physics_only"]["train"].append(model_xgb_physics.predict(X_train_physics))
        all_fold_predictions["xgb_physics_only"]["val"].append(model_xgb_physics.predict(X_val_physics))
        all_fold_predictions["xgb_physics_only"]["y_train"].append(y_train)
        all_fold_predictions["xgb_physics_only"]["y_val"].append(y_val)

    # Compute individual model performance
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)

    for model_name in ensemble_models.keys():
        # Concatenate all folds
        y_train_all = np.concatenate(all_fold_predictions[model_name]["y_train"])
        y_val_all = np.concatenate(all_fold_predictions[model_name]["y_val"])
        pred_train_all = np.concatenate(all_fold_predictions[model_name]["train"])
        pred_val_all = np.concatenate(all_fold_predictions[model_name]["val"])

        mse_train = mean_squared_error(y_train_all, pred_train_all)
        mse_val = mean_squared_error(y_val_all, pred_val_all)
        r2_val = r2_score(y_val_all, pred_val_all)

        print(f"{model_name:25} Train MSE: {mse_train:.6f}, Val MSE: {mse_val:.6f}, Val R²: {r2_val:.4f}")

    return ensemble_models, all_fold_predictions


def optimize_ensemble_weights(all_fold_predictions: Dict) -> np.ndarray:
    """
    Find optimal weights for ensemble using validation predictions.

    Uses constrained optimization to minimize MSE on validation set.
    """
    print("\n" + "="*60)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("="*60)

    model_names = list(all_fold_predictions.keys())
    n_models = len(model_names)

    # Concatenate validation predictions from all folds
    val_predictions = []
    for model_name in model_names:
        pred = np.concatenate(all_fold_predictions[model_name]["val"])
        val_predictions.append(pred)

    val_predictions = np.array(val_predictions).T  # Shape: (n_samples, n_models)
    y_val = np.concatenate(all_fold_predictions[model_names[0]]["y_val"])

    # Objective: minimize MSE with weighted average
    def objective(weights):
        ensemble_pred = val_predictions @ weights
        return mean_squared_error(y_val, ensemble_pred)

    # Constraints: weights sum to 1, all non-negative
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    # Initial weights: uniform
    x0 = np.ones(n_models) / n_models

    # Optimize
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    optimal_weights = result.x

    print("\nOptimal weights:")
    for name, weight in zip(model_names, optimal_weights):
        print(f"  {name:25} {weight:.4f}")

    # Compute ensemble performance
    ensemble_pred = val_predictions @ optimal_weights
    ensemble_mse = mean_squared_error(y_val, ensemble_pred)
    ensemble_r2 = r2_score(y_val, ensemble_pred)

    print(f"\nEnsemble Val MSE: {ensemble_mse:.6f} (RMSE: {np.sqrt(ensemble_mse):.4f}°)")
    print(f"Ensemble Val R²: {ensemble_r2:.4f}")

    return optimal_weights


def save_ensemble(ensemble_models, weights, output_dir: Path):
    """Save ensemble models and weights."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each model type
    for model_name, models in ensemble_models.items():
        model_path = output_dir / f"ensemble_{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(models, f)

    # Save weights
    weights_path = output_dir / "ensemble_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(weights, f)

    print(f"\nSaved ensemble to {output_dir}")


def main():
    """Main ensemble training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ensemble of angle models")
    parser.add_argument("--output-dir", type=str, default="models/angle_ensemble",
                        help="Output directory")
    args = parser.parse_args()

    # Load features
    from train_angle_model import load_all_features

    print("Loading features...")
    df_all, y, groups = load_all_features(
        use_physics=True,
        use_engineering=True,
        use_angle_specific=True
    )

    # Load angle-specific features only
    df_angle, _, _ = load_all_features(
        use_physics=False,
        use_engineering=False,
        use_angle_specific=True
    )

    # Load physics features only
    df_physics, _, _ = load_all_features(
        use_physics=True,
        use_engineering=False,
        use_angle_specific=False
    )

    # Prepare feature matrices
    exclude_cols = ["participant_id"]

    feature_names_all = [c for c in df_all.columns if c not in exclude_cols]
    X_all = np.nan_to_num(df_all[feature_names_all].values, nan=0.0)

    feature_names_angle = [c for c in df_angle.columns if c not in exclude_cols]
    X_angle = np.nan_to_num(df_angle[feature_names_angle].values, nan=0.0)

    feature_names_physics = [c for c in df_physics.columns if c not in exclude_cols]
    X_physics = np.nan_to_num(df_physics[feature_names_physics].values, nan=0.0)

    print(f"\nAll features: {X_all.shape}")
    print(f"Angle features: {X_angle.shape}")
    print(f"Physics features: {X_physics.shape}")

    # Train ensemble
    ensemble_models, all_fold_predictions = train_diverse_models(
        X_all=X_all,
        X_angle=X_angle,
        X_physics=X_physics,
        y=y,
        groups=groups,
        feature_names_all=feature_names_all,
        feature_names_angle=feature_names_angle,
        feature_names_physics=feature_names_physics
    )

    # Optimize ensemble weights
    optimal_weights = optimize_ensemble_weights(all_fold_predictions)

    # Save ensemble
    output_dir = Path(args.output_dir)
    save_ensemble(ensemble_models, optimal_weights, output_dir)

    print("\n" + "="*60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
