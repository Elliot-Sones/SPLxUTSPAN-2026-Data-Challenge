"""
Player 1 Angle Deep Optimization

Goal: Maximize prediction quality for Player 1 angle specifically.
This builds on the initial optimization findings and goes deeper.

Key findings from initial optimization:
- Best config: Optuna Ridge with feature selection (scaled MSE = 0.001273)
- SelectKBest with k=20 features works well (R2=0.24)
- Top correlated features are the key signal

This script will:
1. Identify the exact best features for Player 1 angle
2. Find the optimal model configuration
3. Create a test prediction using the best model
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
from scipy.stats import pearsonr
import lightgbm as lgb
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


def load_data():
    """Load all data (train and test) with comprehensive features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    print("Loading all data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    def extract_features(df, is_train=True):
        features = []
        targets = []
        pids = []
        ids = []

        for idx, row in df.iterrows():
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            hybrid = extract_hybrid_features(ts, row["participant_id"], smooth=False)
            advanced = extract_advanced_features(ts, row["participant_id"])
            combined = {**hybrid, **advanced}
            features.append(combined)
            pids.append(row["participant_id"])
            ids.append(row["id"])
            if is_train:
                targets.append(row["angle"])

        return features, targets, pids, ids

    print("  Processing training data...")
    train_features, train_targets, train_pids, train_ids = extract_features(train_df, True)

    print("  Processing test data...")
    test_features, _, test_pids, test_ids = extract_features(test_df, False)

    # Get feature names
    feature_names = sorted(train_features[0].keys())

    # Convert to arrays
    X_train = np.array([[f.get(n, 0.0) for n in feature_names] for f in train_features], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)
    pids_train = np.array(train_pids)
    ids_train = np.array(train_ids)

    X_test = np.array([[f.get(n, 0.0) for n in feature_names] for f in test_features], dtype=np.float32)
    pids_test = np.array(test_pids)
    ids_test = np.array(test_ids)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "pids_train": pids_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "pids_test": pids_test,
        "ids_test": ids_test,
        "feature_names": feature_names,
    }


def compute_cv_score(X, y, model_class, model_params, n_splits=5, n_repeats=3):
    """Compute repeated k-fold CV score."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    all_preds = []
    all_true = []

    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(**model_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            all_preds.extend(preds)
            all_true.extend(y_val)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    raw_mse = np.mean((all_preds - all_true) ** 2)
    scaled_mse = raw_mse / (ANGLE_RANGE ** 2)
    r2 = 1 - (np.sum((all_preds - all_true) ** 2) / np.sum((all_true - all_true.mean()) ** 2))

    return {"raw_mse": raw_mse, "scaled_mse": scaled_mse, "r2": r2}


def main():
    print("=" * 80)
    print("PLAYER 1 ANGLE DEEP OPTIMIZATION")
    print("=" * 80)

    # Load data
    data = load_data()

    # Filter to Player 1
    p1_train_mask = data["pids_train"] == PLAYER_ID
    p1_test_mask = data["pids_test"] == PLAYER_ID

    X_p1 = data["X_train"][p1_train_mask]
    y_p1 = data["y_train"][p1_train_mask]

    X_test_p1 = data["X_test"][p1_test_mask]
    ids_test_p1 = data["ids_test"][p1_test_mask]

    feature_names = data["feature_names"]

    print(f"\nPlayer 1 Training: {len(y_p1)} samples")
    print(f"Player 1 Test: {len(X_test_p1)} samples")
    print(f"Angle range: {y_p1.min():.2f} to {y_p1.max():.2f}, mean={y_p1.mean():.2f}, std={y_p1.std():.2f}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_p1)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # =============================================================================
    # PHASE 1: Identify Best Features
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: FEATURE ANALYSIS")
    print("=" * 80)

    # Calculate correlations
    correlations = []
    for i, name in enumerate(feature_names):
        try:
            r, p = pearsonr(X_scaled[:, i], y_p1)
            if not np.isnan(r):
                correlations.append((name, abs(r), r, p, i))
        except:
            pass

    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 50 features by correlation with angle:")
    for name, abs_r, r, p, idx in correlations[:50]:
        print(f"  {name:50s}: r={r:+.4f}, p={p:.4f}")

    # Save top features
    top_features_file = OUTPUT_DIR / "player1_angle_top_features.csv"
    top_df = pd.DataFrame([(c[0], c[2], c[3]) for c in correlations[:100]],
                          columns=["feature", "correlation", "p_value"])
    top_df.to_csv(top_features_file, index=False)
    print(f"\nTop 100 features saved to: {top_features_file}")

    # =============================================================================
    # PHASE 2: Exhaustive Feature Count Search
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: OPTIMAL FEATURE COUNT SEARCH")
    print("=" * 80)

    results = []

    # Test with top k features (by correlation)
    for k in range(5, min(101, len(correlations)), 5):
        top_indices = [c[4] for c in correlations[:k]]
        X_top = X_scaled[:, top_indices]

        scores = compute_cv_score(X_top, y_p1, Ridge, {"alpha": 100.0, "random_state": 42})
        print(f"  Top {k:3d} features: scaled_MSE={scores['scaled_mse']:.6f}, R2={scores['r2']:.4f}")

        results.append({
            "method": "top_corr",
            "k": k,
            "scaled_mse": scores["scaled_mse"],
            "r2": scores["r2"],
        })

    # Find optimal k
    best_k_result = min(results, key=lambda x: x["scaled_mse"])
    print(f"\nOptimal k by correlation: {best_k_result['k']} (scaled_MSE={best_k_result['scaled_mse']:.6f})")

    # =============================================================================
    # PHASE 3: Optuna Deep Optimization
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: OPTUNA DEEP OPTIMIZATION (500 trials)")
    print("=" * 80)

    def objective(trial):
        # Feature selection
        k = trial.suggest_int("k_features", 5, min(80, len(correlations)))
        top_indices = [c[4] for c in correlations[:k]]
        X_selected = X_scaled[:, top_indices]

        # Model selection
        model_type = trial.suggest_categorical("model", ["ridge", "elasticnet", "lgb"])

        if model_type == "ridge":
            alpha = trial.suggest_float("ridge_alpha", 0.01, 10000.0, log=True)
            model_class = Ridge
            params = {"alpha": alpha, "random_state": 42}

        elif model_type == "elasticnet":
            alpha = trial.suggest_float("en_alpha", 0.01, 100.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
            model_class = ElasticNet
            params = {"alpha": alpha, "l1_ratio": l1_ratio, "random_state": 42, "max_iter": 5000}

        else:  # lgb
            model_class = lgb.LGBMRegressor
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "num_leaves": trial.suggest_int("num_leaves", 3, 30),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 3, 30),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }

        scores = compute_cv_score(X_selected, y_p1, model_class, params, n_splits=5, n_repeats=1)
        return scores["scaled_mse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  Value (scaled_MSE): {study.best_value:.6f}")
    print(f"  Params: {study.best_params}")

    # =============================================================================
    # PHASE 4: Train Final Model and Generate Predictions
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: FINAL MODEL AND TEST PREDICTIONS")
    print("=" * 80)

    best_params = study.best_params
    k = best_params["k_features"]
    top_indices = [c[4] for c in correlations[:k]]
    top_feature_names = [c[0] for c in correlations[:k]]

    # Get selected features
    X_final = X_scaled[:, top_indices]

    # Build final model
    model_type = best_params["model"]
    if model_type == "ridge":
        final_model = Ridge(alpha=best_params["ridge_alpha"], random_state=42)
    elif model_type == "elasticnet":
        final_model = ElasticNet(
            alpha=best_params["en_alpha"],
            l1_ratio=best_params["l1_ratio"],
            random_state=42,
            max_iter=5000
        )
    else:
        final_model = lgb.LGBMRegressor(
            n_estimators=best_params["n_estimators"],
            num_leaves=best_params["num_leaves"],
            learning_rate=best_params["learning_rate"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            min_child_samples=best_params["min_child_samples"],
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

    # Train on all Player 1 data
    final_model.fit(X_final, y_p1)

    # Get CV predictions for analysis
    cv_preds = np.zeros(len(y_p1))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_final):
        if model_type == "ridge":
            cv_model = Ridge(alpha=best_params["ridge_alpha"], random_state=42)
        elif model_type == "elasticnet":
            cv_model = ElasticNet(alpha=best_params["en_alpha"], l1_ratio=best_params["l1_ratio"], random_state=42, max_iter=5000)
        else:
            cv_model = lgb.LGBMRegressor(**{k: v for k, v in best_params.items() if k.startswith("n_") or k in ["learning_rate", "reg_alpha", "reg_lambda", "min_child_samples", "num_leaves"]}, random_state=42, verbose=-1, n_jobs=-1)

        cv_model.fit(X_final[train_idx], y_p1[train_idx])
        cv_preds[val_idx] = cv_model.predict(X_final[val_idx])

    cv_mse = np.mean((cv_preds - y_p1) ** 2)
    cv_scaled_mse = cv_mse / (ANGLE_RANGE ** 2)
    cv_r2 = 1 - (np.sum((cv_preds - y_p1) ** 2) / np.sum((y_p1 - y_p1.mean()) ** 2))

    print(f"\nFinal CV performance:")
    print(f"  Scaled MSE: {cv_scaled_mse:.6f}")
    print(f"  R2: {cv_r2:.4f}")
    print(f"  Raw RMSE: {np.sqrt(cv_mse):.4f} degrees")

    # Generate test predictions
    X_test_scaled = scaler.transform(X_test_p1)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
    X_test_final = X_test_scaled[:, top_indices]

    test_preds = final_model.predict(X_test_final)

    print(f"\nTest predictions for Player 1 angle:")
    print(f"  N samples: {len(test_preds)}")
    print(f"  Range: {test_preds.min():.2f} to {test_preds.max():.2f}")
    print(f"  Mean: {test_preds.mean():.2f}, Std: {test_preds.std():.2f}")

    # Save predictions
    pred_df = pd.DataFrame({
        "id": ids_test_p1,
        "angle_pred": test_preds,
        "angle_pred_scaled": ANGLE_SCALER.transform(test_preds.reshape(-1, 1)).flatten(),
    })
    pred_file = OUTPUT_DIR / "player1_angle_predictions.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"\nPredictions saved to: {pred_file}")

    # Save best features
    features_df = pd.DataFrame({
        "feature": top_feature_names,
        "correlation": [c[2] for c in correlations[:k]],
    })
    features_file = OUTPUT_DIR / "player1_angle_best_features.csv"
    features_df.to_csv(features_file, index=False)
    print(f"Best features saved to: {features_file}")

    # =============================================================================
    # PHASE 5: Analysis and Recommendations
    # =============================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS AND RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n1. Best model type: {model_type}")
    print(f"2. Optimal number of features: {k}")
    print(f"3. CV scaled MSE: {cv_scaled_mse:.6f}")

    # Compare to baseline
    baseline_mse = np.var(y_p1) / (ANGLE_RANGE ** 2)
    improvement = (1 - cv_scaled_mse / baseline_mse) * 100
    print(f"\n4. Baseline (predicting mean): {baseline_mse:.6f}")
    print(f"5. Improvement over baseline: {improvement:.1f}%")

    # Feature importance
    print(f"\n6. Most important features (by correlation):")
    for i, (name, abs_r, r, p, idx) in enumerate(correlations[:10]):
        print(f"   {i+1}. {name}: r={r:+.4f}")

    # Training sample statistics
    print(f"\n7. Training data statistics:")
    print(f"   - N samples: {len(y_p1)}")
    print(f"   - Angle mean: {y_p1.mean():.2f}")
    print(f"   - Angle std: {y_p1.std():.2f}")

    # Expected test performance
    print(f"\n8. Expected test performance:")
    print(f"   - CV scaled MSE for Player 1 angle: {cv_scaled_mse:.6f}")
    print(f"   - If all players/targets had this performance:")
    print(f"     Overall score would be: {cv_scaled_mse:.6f}")
    print(f"   - Current best LB score is: 0.008305")
    print(f"   - This represents a potential improvement of {(0.008305 - cv_scaled_mse)/0.008305*100:.1f}%")
    print(f"     IF this performance generalizes to test set")

    # Caveats
    print(f"\n9. IMPORTANT CAVEATS:")
    print(f"   - This is CV performance, NOT test performance")
    print(f"   - Player 1 has only {len(y_p1)} training samples")
    print(f"   - Overfitting is a risk with small samples")
    print(f"   - Test set distribution may differ")

    # Save summary
    summary = {
        "player_id": PLAYER_ID,
        "target": TARGET,
        "n_train_samples": len(y_p1),
        "n_test_samples": len(X_test_p1),
        "best_model": model_type,
        "n_features": k,
        "cv_scaled_mse": cv_scaled_mse,
        "cv_r2": cv_r2,
        "baseline_mse": baseline_mse,
        "improvement_pct": improvement,
        "best_params": str(best_params),
    }

    summary_file = OUTPUT_DIR / "player1_angle_summary.json"
    import json as json_module
    with open(summary_file, "w") as f:
        json_module.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
