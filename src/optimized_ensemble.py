"""
Optimized ensemble with Optuna hyperparameter tuning.

Strategy:
1. Target-specific feature selection
2. Per-player per-target hyperparameter optimization
3. Weighted ensemble with optimal weights
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
import optuna
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_all_data():
    """Load train and test data with all features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    def extract_features(df, is_train=True):
        all_features = []
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
            combined = {**hybrid_feats, **advanced_feats}
            all_features.append(combined)

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        return all_features, ids, pids, targets if is_train else None

    print(f"Processing {len(train_df)} training shots...")
    train_feats, train_ids, train_pids, train_targets = extract_features(train_df, True)

    print(f"Processing {len(test_df)} test shots...")
    test_feats, test_ids, test_pids, _ = extract_features(test_df, False)

    feature_names = sorted(train_feats[0].keys())
    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "X_test": X_test,
        "test_ids": np.array(test_ids),
        "test_pids": np.array(test_pids),
        "feature_names": feature_names,
    }


def select_features_for_target(X, y, feature_names, target_name, k=100):
    """Select top k features for a specific target."""
    selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
    selector.fit(np.nan_to_num(X, nan=0.0), y)
    mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_names, mask) if m]
    return mask, selected_features


def optimize_lgb_for_player_target(X, y, n_trials=30):
    """Optuna optimization for LightGBM."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = lgb.LGBMRegressor(**params)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        return scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_catboost_for_player_target(X, y, n_trials=20):
    """Optuna optimization for CatBoost with manual CV."""
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "random_state": 42,
            "verbose": False,
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            mse = np.mean((pred - y_val) ** 2)
            scores.append(mse)

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_optimized_ensemble(data):
    """Train optimized ensemble with per-player per-target tuning."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    pids = data["train_pids"]
    feature_names = data["feature_names"]

    unique_pids = sorted(np.unique(pids))

    # Storage
    all_models = {}
    all_scalers = {}
    all_feature_masks = {}
    oof_preds = np.zeros_like(y_train)

    # Target-specific feature selection
    print("\n" + "="*70)
    print("FEATURE SELECTION PER TARGET")
    print("="*70)

    target_feature_masks = {}
    for target_idx, target in enumerate(TARGETS):
        mask, selected = select_features_for_target(
            X_train, y_train[:, target_idx], feature_names, target, k=150
        )
        target_feature_masks[target] = mask
        print(f"  {target}: {sum(mask)} features selected")

    # Train per player per target
    print("\n" + "="*70)
    print("TRAINING WITH OPTUNA OPTIMIZATION")
    print("="*70)

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X_train[pid_mask]
        y_player = y_train[pid_mask]
        n_samples = len(X_player)

        print(f"\n--- Player {pid} ({n_samples} samples) ---")

        # Standardize
        scaler = StandardScaler()
        X_scaled_full = scaler.fit_transform(X_player)
        X_scaled_full = np.nan_to_num(X_scaled_full, nan=0.0)
        all_scalers[pid] = scaler

        player_indices = np.where(pid_mask)[0]

        for target_idx, target in enumerate(TARGETS):
            print(f"  Optimizing {target}...")

            # Apply target-specific feature selection
            feature_mask = target_feature_masks[target]
            X_scaled = X_scaled_full[:, feature_mask]
            y_target = y_player[:, target_idx]

            all_feature_masks[(pid, target)] = feature_mask

            # Optimize hyperparameters
            best_lgb_params = optimize_lgb_for_player_target(X_scaled, y_target, n_trials=20)
            best_cat_params = optimize_catboost_for_player_target(X_scaled, y_target, n_trials=15)

            # 5-fold CV for OOF predictions
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            fold_preds = np.zeros(n_samples)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y_target[train_idx], y_target[val_idx]

                # Train LGB
                lgb_model = lgb.LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1, n_jobs=-1)
                lgb_model.fit(X_tr, y_tr)
                lgb_pred = lgb_model.predict(X_val)

                # Train CatBoost
                cat_model = CatBoostRegressor(**best_cat_params, random_state=42, verbose=False)
                cat_model.fit(X_tr, y_tr)
                cat_pred = cat_model.predict(X_val)

                # Train Ridge
                ridge_model = Ridge(alpha=1.0, random_state=42)
                ridge_model.fit(X_tr, y_tr)
                ridge_pred = ridge_model.predict(X_val)

                # Weighted average (learn optimal weights from validation)
                # Simple average for now
                fold_preds[val_idx] = 0.45 * lgb_pred + 0.45 * cat_pred + 0.1 * ridge_pred

            # Store OOF predictions
            oof_preds[player_indices, target_idx] = fold_preds

            # Train final models on all player data
            lgb_final = lgb.LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1, n_jobs=-1)
            lgb_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'lgb')] = lgb_final

            cat_final = CatBoostRegressor(**best_cat_params, random_state=42, verbose=False)
            cat_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'cat')] = cat_final

            ridge_final = Ridge(alpha=1.0, random_state=42)
            ridge_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_final

            # Report CV score
            mse = np.mean((fold_preds - y_target) ** 2)
            print(f"    {target} CV MSE: {mse:.4f}")

    # Overall CV evaluation
    print("\n" + "="*70)
    print("OVERALL CV RESULTS")
    print("="*70)

    # Load scalers for final metrics
    angle_scaler = joblib.load(DATA_DIR / 'scaler_angle.pkl')
    depth_scaler = joblib.load(DATA_DIR / 'scaler_depth.pkl')
    lr_scaler = joblib.load(DATA_DIR / 'scaler_left_right.pkl')

    ranges = {
        "angle": angle_scaler.data_range_[0],
        "depth": depth_scaler.data_range_[0],
        "left_right": lr_scaler.data_range_[0],
    }

    total_scaled_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((oof_preds[:, target_idx] - y_train[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_scaled_mse += scaled_mse
        print(f"  {target}: raw MSE = {raw_mse:.4f}, scaled MSE = {scaled_mse:.6f}")

    avg_scaled_mse = total_scaled_mse / 3
    print(f"\n  AVERAGE SCALED MSE: {avg_scaled_mse:.6f}")

    return {
        "models": all_models,
        "scalers": all_scalers,
        "feature_masks": all_feature_masks,
        "target_feature_masks": target_feature_masks,
        "oof_preds": oof_preds,
        "cv_score": avg_scaled_mse,
    }


def predict(data, trained):
    """Generate predictions for test set."""
    X_test = data["X_test"]
    test_pids = data["test_pids"]

    models = trained["models"]
    scalers = trained["scalers"]
    feature_masks = trained["feature_masks"]

    predictions = np.zeros((len(X_test), 3))

    for i, (x, pid) in enumerate(zip(X_test, test_pids)):
        x_scaled_full = scalers[pid].transform(x.reshape(1, -1))
        x_scaled_full = np.nan_to_num(x_scaled_full, nan=0.0)

        for target_idx, target in enumerate(TARGETS):
            feature_mask = feature_masks[(pid, target)]
            x_scaled = x_scaled_full[:, feature_mask]

            lgb_pred = models[(pid, target, 'lgb')].predict(x_scaled)[0]
            cat_pred = models[(pid, target, 'cat')].predict(x_scaled)[0]
            ridge_pred = models[(pid, target, 'ridge')].predict(x_scaled)[0]

            predictions[i, target_idx] = 0.45 * lgb_pred + 0.45 * cat_pred + 0.1 * ridge_pred

    return predictions


def create_submission(test_ids, predictions, submission_num, cv_score):
    """Create submission CSV."""
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions[:, 0],
        'scaled_depth': scaled_predictions[:, 1],
        'scaled_left_right': scaled_predictions[:, 2],
    })

    filename = f"submission_{submission_num}.csv"
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\nSubmission saved to: {filepath}")
    print(f"CV Score: {cv_score:.6f}")

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}, "
              f"min={submission[col].min():.4f}, max={submission[col].max():.4f}")

    return filepath


def main():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    print("="*70)
    print(f"OPTIMIZED ENSEMBLE SUBMISSION {next_num}")
    print("="*70)

    # Load data
    data = load_all_data()

    # Train
    trained = train_optimized_ensemble(data)

    # Predict
    predictions = predict(data, trained)

    # Submit
    filepath = create_submission(
        data["test_ids"], predictions, next_num, trained["cv_score"]
    )

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    return filepath


if __name__ == "__main__":
    main()
