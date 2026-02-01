"""
Player 1 Angle Optimization

Intensive optimization for a single player (ID=1) and single target (angle).
Goal: Find the best possible model configuration for this specific combination
to understand what's achievable and what features/models work best.

This will help understand if there's hidden signal we're missing.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from scipy.stats import pearsonr
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "angle"
PLAYER_ID = 1

# Load target scaler
ANGLE_SCALER = joblib.load(DATA_DIR / "scaler_angle.pkl")
ANGLE_RANGE = ANGLE_SCALER.data_range_[0]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_player1_data():
    """Load only Player 1 data with comprehensive features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    print("Loading Player 1 data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    # Filter to Player 1 only
    train_df = train_df[train_df["participant_id"] == PLAYER_ID].reset_index(drop=True)
    print(f"Player 1 has {len(train_df)} samples")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    # Build keypoint index for raw feature extraction
    keypoint_index = {}
    for idx, col in enumerate(keypoint_cols):
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            keypoint_name = parts[0]
            coord = parts[1]
            if keypoint_name not in keypoint_index:
                keypoint_index[keypoint_name] = {}
            keypoint_index[keypoint_name][coord] = idx

    all_hybrid_features = []
    all_advanced_features = []
    all_timeseries = []
    all_targets = []
    shot_ids = []

    for idx, row in train_df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        hybrid_feats = extract_hybrid_features(timeseries, row["participant_id"], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row["participant_id"])

        all_hybrid_features.append(hybrid_feats)
        all_advanced_features.append(advanced_feats)
        all_timeseries.append(timeseries)
        all_targets.append(row["angle"])
        shot_ids.append(row["id"])

    # Build feature matrix (hybrid + advanced)
    hybrid_names = sorted(all_hybrid_features[0].keys())
    advanced_names = sorted(all_advanced_features[0].keys())
    all_names = sorted(set(hybrid_names + advanced_names))

    X = np.zeros((len(train_df), len(all_names)), dtype=np.float32)
    for i, (hf, af) in enumerate(zip(all_hybrid_features, all_advanced_features)):
        combined = {**hf, **af}
        for j, name in enumerate(all_names):
            X[i, j] = combined.get(name, 0.0)

    y = np.array(all_targets, dtype=np.float32)

    print(f"Features shape: {X.shape}")
    print(f"Target (angle) range: {y.min():.2f} to {y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

    return {
        "X": X,
        "y": y,
        "feature_names": all_names,
        "timeseries": all_timeseries,
        "keypoint_index": keypoint_index,
        "keypoint_cols": keypoint_cols,
        "shot_ids": shot_ids,
    }


def compute_cv_score(X, y, model_class, model_params, n_splits=5):
    """Compute 5-fold CV score (raw MSE)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = np.zeros(len(y))

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train = y[train_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        preds[val_idx] = model.predict(X_val)

    raw_mse = np.mean((preds - y) ** 2)
    scaled_mse = raw_mse / (ANGLE_RANGE ** 2)
    r2 = 1 - (np.sum((preds - y) ** 2) / np.sum((y - y.mean()) ** 2))

    return {"raw_mse": raw_mse, "scaled_mse": scaled_mse, "r2": r2, "preds": preds}


def extract_frame_specific_features(timeseries_list, keypoint_index, keypoints, frames):
    """Extract features at specific frames."""
    features = []
    feature_names = []

    for kp in keypoints:
        for frame in frames:
            for coord in ["x", "y", "z"]:
                feature_names.append(f"{kp}_{coord}_f{frame}")

    for ts in timeseries_list:
        row = []
        for kp in keypoints:
            if kp not in keypoint_index:
                for _ in frames:
                    for _ in ["x", "y", "z"]:
                        row.append(0.0)
                continue
            for frame in frames:
                for coord in ["x", "y", "z"]:
                    if coord in keypoint_index[kp]:
                        col_idx = keypoint_index[kp][coord]
                        row.append(ts[frame, col_idx] if frame < len(ts) else 0.0)
                    else:
                        row.append(0.0)
        features.append(row)

    return np.array(features, dtype=np.float32), feature_names


def main():
    print("=" * 80)
    print("PLAYER 1 ANGLE OPTIMIZATION")
    print("=" * 80)

    data = load_player1_data()
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"]

    results = []

    # =============================================================================
    # PHASE 1: Model Architecture Search
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: MODEL ARCHITECTURE SEARCH")
    print("=" * 80)

    models_to_test = {
        "Ridge_1": (Ridge, {"alpha": 1.0, "random_state": 42}),
        "Ridge_10": (Ridge, {"alpha": 10.0, "random_state": 42}),
        "Ridge_100": (Ridge, {"alpha": 100.0, "random_state": 42}),
        "Ridge_1000": (Ridge, {"alpha": 1000.0, "random_state": 42}),
        "Lasso_0.01": (Lasso, {"alpha": 0.01, "random_state": 42, "max_iter": 5000}),
        "Lasso_0.1": (Lasso, {"alpha": 0.1, "random_state": 42, "max_iter": 5000}),
        "Lasso_1": (Lasso, {"alpha": 1.0, "random_state": 42, "max_iter": 5000}),
        "ElasticNet_0.5": (ElasticNet, {"alpha": 0.5, "l1_ratio": 0.5, "random_state": 42, "max_iter": 5000}),
        "ElasticNet_0.1": (ElasticNet, {"alpha": 0.1, "l1_ratio": 0.5, "random_state": 42, "max_iter": 5000}),
        "BayesianRidge": (BayesianRidge, {}),
        "LGB_conservative": (lgb.LGBMRegressor, {"n_estimators": 50, "num_leaves": 5, "learning_rate": 0.1, "reg_alpha": 1.0, "reg_lambda": 1.0, "random_state": 42, "verbose": -1, "n_jobs": -1}),
        "LGB_standard": (lgb.LGBMRegressor, {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 0.5, "random_state": 42, "verbose": -1, "n_jobs": -1}),
        "LGB_aggressive": (lgb.LGBMRegressor, {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.03, "max_depth": 8, "random_state": 42, "verbose": -1, "n_jobs": -1}),
        "XGB_standard": (xgb.XGBRegressor, {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 0.5, "random_state": 42, "verbosity": 0, "n_jobs": -1}),
        "CatBoost": (CatBoostRegressor, {"iterations": 100, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0, "random_state": 42, "verbose": False}),
        "RandomForest": (RandomForestRegressor, {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1}),
        "ExtraTrees": (ExtraTreesRegressor, {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1}),
        "GradientBoosting": (GradientBoostingRegressor, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}),
        "KNN_5": (KNeighborsRegressor, {"n_neighbors": 5}),
        "KNN_10": (KNeighborsRegressor, {"n_neighbors": 10}),
        "SVR_rbf": (SVR, {"kernel": "rbf", "C": 1.0}),
    }

    print("\nTesting all model architectures...")
    for model_name, (model_class, params) in models_to_test.items():
        try:
            scores = compute_cv_score(X, y, model_class, params)
            print(f"  {model_name:25s}: scaled_MSE={scores['scaled_mse']:.6f}, R2={scores['r2']:.4f}")
            results.append({
                "phase": "model_search",
                "config": model_name,
                "scaled_mse": scores["scaled_mse"],
                "raw_mse": scores["raw_mse"],
                "r2": scores["r2"],
            })
        except Exception as e:
            print(f"  {model_name:25s}: ERROR - {e}")

    # Find best model
    best_model = min([r for r in results if r["phase"] == "model_search"], key=lambda x: x["scaled_mse"])
    print(f"\nBest model so far: {best_model['config']} with scaled_MSE={best_model['scaled_mse']:.6f}")

    # =============================================================================
    # PHASE 2: Feature Selection
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: FEATURE SELECTION")
    print("=" * 80)

    # Use Ridge as base model for feature selection experiments
    base_model_class = Ridge
    base_params = {"alpha": 100.0, "random_state": 42}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # SelectKBest with different k values
    print("\nSelectKBest (f_regression):")
    for k in [20, 50, 100, 150, 200, 250]:
        if k > X.shape[1]:
            continue
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X_scaled, y)

        scores = compute_cv_score(X_selected, y, base_model_class, base_params)
        print(f"  k={k:3d}: scaled_MSE={scores['scaled_mse']:.6f}, R2={scores['r2']:.4f}")
        results.append({
            "phase": "feature_selection",
            "config": f"SelectKBest_f_reg_k{k}",
            "scaled_mse": scores["scaled_mse"],
            "raw_mse": scores["raw_mse"],
            "r2": scores["r2"],
        })

    print("\nSelectKBest (mutual_info):")
    for k in [20, 50, 100, 150]:
        if k > X.shape[1]:
            continue
        selector = SelectKBest(mutual_info_regression, k=k)
        X_selected = selector.fit_transform(X_scaled, y)

        scores = compute_cv_score(X_selected, y, base_model_class, base_params)
        print(f"  k={k:3d}: scaled_MSE={scores['scaled_mse']:.6f}, R2={scores['r2']:.4f}")
        results.append({
            "phase": "feature_selection",
            "config": f"SelectKBest_mi_k{k}",
            "scaled_mse": scores["scaled_mse"],
            "raw_mse": scores["raw_mse"],
            "r2": scores["r2"],
        })

    # =============================================================================
    # PHASE 3: Correlation-based Feature Analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: TOP FEATURES BY CORRELATION")
    print("=" * 80)

    correlations = []
    for i, name in enumerate(feature_names):
        r, p = pearsonr(X_scaled[:, i], y)
        correlations.append((name, abs(r), r, p))

    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 30 features by absolute correlation with angle:")
    for name, abs_r, r, p in correlations[:30]:
        print(f"  {name:45s}: r={r:+.4f}, p={p:.4f}")

    # Test models with top correlated features
    print("\nTesting with top correlated features:")
    for top_n in [10, 20, 30, 50, 100]:
        top_features = [c[0] for c in correlations[:top_n]]
        top_indices = [feature_names.index(f) for f in top_features]
        X_top = X_scaled[:, top_indices]

        scores = compute_cv_score(X_top, y, base_model_class, base_params)
        print(f"  Top {top_n:3d}: scaled_MSE={scores['scaled_mse']:.6f}, R2={scores['r2']:.4f}")
        results.append({
            "phase": "correlation_selection",
            "config": f"top{top_n}_corr",
            "scaled_mse": scores["scaled_mse"],
            "raw_mse": scores["raw_mse"],
            "r2": scores["r2"],
        })

    # =============================================================================
    # PHASE 4: Optuna Hyperparameter Optimization
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    # Use top correlated features
    top_features = [c[0] for c in correlations[:50]]
    top_indices = [feature_names.index(f) for f in top_features]
    X_opt = X_scaled[:, top_indices]

    def lgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 300),
            "num_leaves": trial.suggest_int("num_leaves", 3, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        scores = compute_cv_score(X_opt, y, lgb.LGBMRegressor, params)
        return scores["scaled_mse"]

    print("\nOptimizing LightGBM (100 trials)...")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(lgb_objective, n_trials=100, show_progress_bar=True)

    print(f"\nBest LightGBM params: {study_lgb.best_params}")
    print(f"Best LightGBM scaled_MSE: {study_lgb.best_value:.6f}")

    results.append({
        "phase": "optuna_lgb",
        "config": "optimized_lgb",
        "scaled_mse": study_lgb.best_value,
        "raw_mse": study_lgb.best_value * (ANGLE_RANGE ** 2),
        "r2": 0,  # Would need to recalculate
    })

    # Optuna for Ridge
    def ridge_objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 10000.0, log=True)
        params = {"alpha": alpha, "random_state": 42}

        # Also try different feature counts
        k = trial.suggest_int("k_features", 10, min(200, X.shape[1]))
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X_scaled, y)

        scores = compute_cv_score(X_selected, y, Ridge, params)
        return scores["scaled_mse"]

    print("\nOptimizing Ridge with feature selection (100 trials)...")
    study_ridge = optuna.create_study(direction="minimize")
    study_ridge.optimize(ridge_objective, n_trials=100, show_progress_bar=True)

    print(f"\nBest Ridge params: {study_ridge.best_params}")
    print(f"Best Ridge scaled_MSE: {study_ridge.best_value:.6f}")

    results.append({
        "phase": "optuna_ridge",
        "config": "optimized_ridge",
        "scaled_mse": study_ridge.best_value,
        "raw_mse": study_ridge.best_value * (ANGLE_RANGE ** 2),
        "r2": 0,
    })

    # =============================================================================
    # PHASE 5: Frame-Specific Feature Engineering
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: FRAME-SPECIFIC FEATURES")
    print("=" * 80)

    # Based on research, frame 153 is peak for angle prediction
    # Test features at different frames around this

    keypoints = ["right_wrist", "right_elbow", "right_shoulder", "right_knee", "right_ankle",
                 "left_wrist", "left_elbow", "mid_hip", "neck", "nose"]

    frame_ranges = {
        "release_tight": list(range(145, 165, 2)),  # Frames 145-163
        "release_wide": list(range(130, 180, 5)),   # Frames 130-175
        "full_shot": list(range(100, 180, 10)),     # Frames 100-170
        "all_phases": list(range(0, 240, 20)),      # Every 20 frames
    }

    for range_name, frames in frame_ranges.items():
        X_frames, _ = extract_frame_specific_features(
            data["timeseries"], data["keypoint_index"], keypoints, frames
        )

        scaler_f = StandardScaler()
        X_frames_scaled = scaler_f.fit_transform(X_frames)
        X_frames_scaled = np.nan_to_num(X_frames_scaled, nan=0.0)

        # Test with Ridge
        scores = compute_cv_score(X_frames_scaled, y, Ridge, {"alpha": 100.0, "random_state": 42})
        print(f"  {range_name:20s} ({X_frames.shape[1]:3d} features): scaled_MSE={scores['scaled_mse']:.6f}, R2={scores['r2']:.4f}")

        results.append({
            "phase": "frame_specific",
            "config": range_name,
            "scaled_mse": scores["scaled_mse"],
            "raw_mse": scores["raw_mse"],
            "r2": scores["r2"],
        })

    # =============================================================================
    # PHASE 6: Ensemble Methods
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 6: ENSEMBLE METHODS")
    print("=" * 80)

    # Simple averaging ensemble
    print("\nTesting ensemble of best models...")

    ensemble_models = [
        (Ridge, {"alpha": study_ridge.best_params.get("alpha", 100.0), "random_state": 42}),
        (lgb.LGBMRegressor, {**study_lgb.best_params, "random_state": 42, "verbose": -1, "n_jobs": -1}),
        (CatBoostRegressor, {"iterations": 100, "depth": 4, "learning_rate": 0.05, "random_state": 42, "verbose": False}),
    ]

    # Get predictions from each model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = []

    for model_class, params in ensemble_models:
        preds = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train = y[train_idx]
            model = model_class(**params)
            model.fit(X_train, y_train)
            preds[val_idx] = model.predict(X_val)
        all_preds.append(preds)

    # Simple average
    ensemble_pred = np.mean(all_preds, axis=0)
    ensemble_mse = np.mean((ensemble_pred - y) ** 2)
    ensemble_scaled_mse = ensemble_mse / (ANGLE_RANGE ** 2)
    ensemble_r2 = 1 - (np.sum((ensemble_pred - y) ** 2) / np.sum((y - y.mean()) ** 2))

    print(f"  Ensemble (avg): scaled_MSE={ensemble_scaled_mse:.6f}, R2={ensemble_r2:.4f}")

    results.append({
        "phase": "ensemble",
        "config": "simple_avg",
        "scaled_mse": ensemble_scaled_mse,
        "raw_mse": ensemble_mse,
        "r2": ensemble_r2,
    })

    # Weighted ensemble (optimize weights)
    def ensemble_objective(trial):
        w1 = trial.suggest_float("w1", 0, 1)
        w2 = trial.suggest_float("w2", 0, 1)
        w3 = 1 - w1 - w2
        if w3 < 0:
            return 1.0

        pred = w1 * all_preds[0] + w2 * all_preds[1] + w3 * all_preds[2]
        mse = np.mean((pred - y) ** 2)
        return mse / (ANGLE_RANGE ** 2)

    study_ens = optuna.create_study(direction="minimize")
    study_ens.optimize(ensemble_objective, n_trials=50, show_progress_bar=False)

    print(f"  Ensemble (weighted): scaled_MSE={study_ens.best_value:.6f}, weights={study_ens.best_params}")

    results.append({
        "phase": "ensemble",
        "config": "weighted",
        "scaled_mse": study_ens.best_value,
        "raw_mse": study_ens.best_value * (ANGLE_RANGE ** 2),
        "r2": 0,
    })

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - PLAYER 1 ANGLE OPTIMIZATION")
    print("=" * 80)

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("scaled_mse")

    print("\nTop 10 Best Configurations:")
    for i, row in df_sorted.head(10).iterrows():
        print(f"  {row['phase']}/{row['config']}: scaled_MSE={row['scaled_mse']:.6f}, R2={row['r2']:.4f}")

    # Save results
    output_file = OUTPUT_DIR / "player1_angle_optimization.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Best configuration
    best = df_sorted.iloc[0]
    print(f"\n*** BEST CONFIGURATION ***")
    print(f"  Phase: {best['phase']}")
    print(f"  Config: {best['config']}")
    print(f"  Scaled MSE: {best['scaled_mse']:.6f}")
    print(f"  Raw MSE: {best['raw_mse']:.4f}")
    print(f"  R2: {best['r2']:.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"\nPlayer 1 has {len(y)} samples with angle ranging from {y.min():.2f} to {y.max():.2f}")
    print(f"Angle std: {y.std():.2f}")
    print(f"\nBest achievable scaled MSE on this data: {best['scaled_mse']:.6f}")
    print(f"This corresponds to raw RMSE of: {np.sqrt(best['raw_mse']):.4f} degrees")

    # Compare to baseline
    baseline_mse = np.var(y) / (ANGLE_RANGE ** 2)
    print(f"\nBaseline (predicting mean): scaled_MSE={baseline_mse:.6f}")
    print(f"Best model improvement over baseline: {(1 - best['scaled_mse']/baseline_mse)*100:.1f}%")


if __name__ == "__main__":
    main()
