#!/usr/bin/env python3
"""
Test the Kaggle baseline approach with cross-validation.
This extracts simple features (mean + last value) from timeseries.
"""

import numpy as np
import pandas as pd
import ast
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# --- Configuration ---
BASE_PATH = Path("data")
TRAIN_PATH = BASE_PATH / "train.csv"

SCALER_BOUNDS = {
    'angle': {'min': 30, 'max': 60},
    'depth': {'min': -12, 'max': 30},
    'left_right': {'min': -16, 'max': 16}
}

# --- Data Loading ---
def parse_array_column(x):
    """Safely parses string-lists into numpy arrays."""
    try:
        if isinstance(x, str):
            if x.strip().startswith('[') and x.strip().endswith(']'):
                return np.array(ast.literal_eval(x))
        if isinstance(x, (list, np.ndarray)):
            return np.array(x)
        return np.zeros(1)
    except:
        return np.zeros(1)

def load_data():
    """Load training data from CSV and parse arrays."""
    print("Loading training data...")
    df = pd.read_csv(TRAIN_PATH)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} shots")

    # Identify and parse timeseries columns
    exclude_cols = ['id', 'shot_id', 'participant_id', 'angle', 'depth', 'left_right', 'Unnamed: 0']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Found {len(feature_cols)} feature columns")
    print("Parsing timeseries data...")

    for col in feature_cols:
        if len(df) > 0 and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(parse_array_column)

    return df, feature_cols

def extract_simple_features(df, feature_cols):
    """
    Extract simple statistical features from timeseries.
    For each keypoint coordinate: [mean, last_value]
    """
    print("\nExtracting features...")
    print(f"Processing {len(feature_cols)} timeseries columns")

    features = []

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Processing shot {idx+1}/{len(df)}...")

        row_feats = []

        for col in feature_cols:
            ts_data = row[col]

            # Convert to 1D array
            if hasattr(ts_data, '__len__') and len(ts_data) > 0:
                arr = np.array(ts_data).flatten()
            else:
                arr = np.array([0.0])

            # Extract statistics
            if len(arr) > 0:
                mean_val = float(np.mean(arr))
                last_val = float(arr[-1])
            else:
                mean_val = 0.0
                last_val = 0.0

            row_feats.extend([mean_val, last_val])

        features.append(row_feats)

    X = np.array(features)
    print(f"\nExtracted features shape: {X.shape}")
    print(f"Features per shot: {X.shape[1]}")

    return X

def train_and_evaluate(X, y, groups, model_type="randomforest"):
    """Train model with cross-validation."""
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=5)

    # Results for all targets
    all_results = {
        'angle': [],
        'depth': [],
        'left_right': []
    }

    target_names = ['angle', 'depth', 'left_right']

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Train separate model for each target
        for t, target_name in enumerate(target_names):
            # Create model
            if model_type == "randomforest":
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == "xgboost":
                model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    random_state=42
                )
            elif model_type == "lightgbm":
                model = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    random_state=42,
                    verbose=-1
                )

            # Train
            model.fit(X_train, y_train[:, t])

            # Predict
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)

            # Metrics
            mse_train = mean_squared_error(y_train[:, t], pred_train)
            mse_val = mean_squared_error(y_val[:, t], pred_val)
            r2_train = r2_score(y_train[:, t], pred_train)
            r2_val = r2_score(y_val[:, t], pred_val)

            print(f"  {target_name:12} Train MSE: {mse_train:.4f}, Val MSE: {mse_val:.4f}, Val R²: {r2_val:.4f}")

            all_results[target_name].append(mse_val)

    # Aggregate results
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION RESULTS - {model_type.upper()}")
    print(f"{'='*70}")

    target_mses = {}
    for target_name in target_names:
        mean_mse = np.mean(all_results[target_name])
        std_mse = np.std(all_results[target_name])
        target_mses[target_name] = mean_mse
        print(f"{target_name:12} Mean Val MSE: {mean_mse:.6f} (±{std_mse:.6f}), RMSE: {np.sqrt(mean_mse):.4f}")

    overall_mse = np.mean(list(target_mses.values()))
    print(f"\n{'Overall':12} Mean MSE: {overall_mse:.6f}, RMSE: {np.sqrt(overall_mse):.4f}")

    return overall_mse, target_mses

def main():
    """Main evaluation pipeline."""
    print("="*70)
    print("KAGGLE BASELINE APPROACH EVALUATION")
    print("="*70)
    print("\nApproach: Extract [mean, last_value] from each timeseries column")
    print("Models: RandomForest, XGBoost, LightGBM")

    # Load data
    df, feature_cols = load_data()

    # Extract features
    X = extract_simple_features(df, feature_cols)

    # Targets and groups
    targets = ['angle', 'depth', 'left_right']
    y = df[targets].values
    groups = df['participant_id'].values

    print(f"\nDataset summary:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Targets: {y.shape[1]}")
    print(f"  Participants: {len(np.unique(groups))}")

    # Test multiple models
    results = {}

    # Model 1: Random Forest (from Kaggle baseline)
    print("\n\n" + "#"*70)
    print("MODEL 1: RANDOM FOREST")
    print("#"*70)
    rf_mse, rf_target_mses = train_and_evaluate(X, y, groups, model_type="randomforest")
    results["RandomForest"] = rf_mse

    # Model 2: XGBoost
    print("\n\n" + "#"*70)
    print("MODEL 2: XGBOOST")
    print("#"*70)
    xgb_mse, xgb_target_mses = train_and_evaluate(X, y, groups, model_type="xgboost")
    results["XGBoost"] = xgb_mse

    # Model 3: LightGBM
    print("\n\n" + "#"*70)
    print("MODEL 3: LIGHTGBM")
    print("#"*70)
    lgb_mse, lgb_target_mses = train_and_evaluate(X, y, groups, model_type="lightgbm")
    results["LightGBM"] = lgb_mse

    # Final comparison
    print("\n\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    baseline = 0.021
    winner = 0.008

    print(f"\n{'Model':<20} {'Overall MSE':>15} {'RMSE':>10} {'vs Baseline':>15} {'vs Winner':>15}")
    print("-"*80)

    for model_name, mse in sorted(results.items(), key=lambda x: x[1]):
        rmse = np.sqrt(mse)

        if mse < baseline:
            vs_baseline = f"{(1-mse/baseline)*100:.1f}% better"
        else:
            vs_baseline = f"{(mse/baseline-1)*100:.1f}% worse"

        if mse < winner:
            vs_winner = "BEATS!"
        else:
            vs_winner = f"{mse/winner:.2f}x worse"

        print(f"{model_name:<20} {mse:>15.6f} {rmse:>10.4f} {vs_baseline:>15} {vs_winner:>15}")

    # Best model
    best_model = min(results, key=results.get)
    best_mse = results[best_model]

    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model}")
    print(f"MSE: {best_mse:.6f} (RMSE: {np.sqrt(best_mse):.4f})")

    if best_mse < baseline:
        improvement = (1 - best_mse/baseline) * 100
        print(f"✓ Improvement over your baseline (0.021): {improvement:.1f}%")
    else:
        decline = (best_mse/baseline - 1) * 100
        print(f"✗ Worse than your baseline (0.021): {decline:.1f}%")

    if best_mse < winner:
        print(f"🏆 BEATS competition winner (0.008)!")
    else:
        gap = best_mse / winner
        print(f"⚠️  {gap:.1f}x worse than winner (0.008)")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
