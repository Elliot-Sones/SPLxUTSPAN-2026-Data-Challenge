"""
Ultra-optimized ensemble focusing on Player 5 improvement.

Key observations:
- Player 5 has 2-3x higher error than other players
- Need different feature engineering for Player 5
- More aggressive hyperparameter search
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
import optuna
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    """Load all data with features."""
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

            if (idx + 1) % 100 == 0:
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


def optimize_lgb(X, y, n_trials=30, is_player5=False):
    """Optimize LightGBM hyperparameters."""
    def objective(trial):
        # More conservative for player 5
        if is_player5:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 30, 150),
                "num_leaves": trial.suggest_int("num_leaves", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 5, 50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            }

        params.update({"random_state": 42, "verbose": -1, "n_jobs": -1})

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(np.mean((pred - y[val_idx]) ** 2))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_catboost(X, y, n_trials=20, is_player5=False):
    """Optimize CatBoost hyperparameters."""
    def objective(trial):
        if is_player5:
            params = {
                "iterations": trial.suggest_int("iterations", 30, 150),
                "depth": trial.suggest_int("depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            }
        else:
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            }

        params.update({"random_state": 42, "verbose": False})

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = CatBoostRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(np.mean((pred - y[val_idx]) ** 2))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_xgb(X, y, n_trials=20, is_player5=False):
    """Optimize XGBoost hyperparameters."""
    def objective(trial):
        if is_player5:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 30, 150),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }

        params.update({"random_state": 42, "verbosity": 0, "n_jobs": -1})

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = xgb.XGBRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(np.mean((pred - y[val_idx]) ** 2))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_ultra_optimized(data):
    """Train with ultra-optimization, focusing on Player 5."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    pids = data["train_pids"]
    feature_names = data["feature_names"]

    unique_pids = sorted(np.unique(pids))

    all_models = {}
    all_scalers = {}
    all_feature_masks = {}
    oof_preds = np.zeros_like(y_train)

    # Target-specific feature counts
    feature_k = {"angle": 120, "depth": 100, "left_right": 80}

    print("\n" + "="*70)
    print("ULTRA-OPTIMIZED TRAINING")
    print("="*70)

    # Feature selection per target
    target_masks = {}
    for target_idx, target in enumerate(TARGETS):
        selector = SelectKBest(mutual_info_regression, k=feature_k[target])
        selector.fit(np.nan_to_num(X_train, nan=0.0), y_train[:, target_idx])
        target_masks[target] = selector.get_support()
        print(f"  {target}: {sum(target_masks[target])} features")

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X_train[pid_mask]
        y_player = y_train[pid_mask]
        n_samples = len(X_player)
        is_player5 = (pid == 5)

        print(f"\n--- Player {pid} ({n_samples} samples) {'[SPECIAL HANDLING]' if is_player5 else ''} ---")

        # Use RobustScaler for Player 5
        if is_player5:
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        X_scaled_full = scaler.fit_transform(X_player)
        X_scaled_full = np.nan_to_num(X_scaled_full, nan=0.0)
        all_scalers[pid] = scaler

        player_indices = np.where(pid_mask)[0]

        # Optuna trials - more for player 5
        n_lgb_trials = 50 if is_player5 else 25
        n_cat_trials = 30 if is_player5 else 15
        n_xgb_trials = 30 if is_player5 else 15

        for target_idx, target in enumerate(TARGETS):
            print(f"  Optimizing {target}...")

            feature_mask = target_masks[target]
            X_scaled = X_scaled_full[:, feature_mask]
            y_target = y_player[:, target_idx]

            all_feature_masks[(pid, target)] = feature_mask

            # Optimize all models
            best_lgb = optimize_lgb(X_scaled, y_target, n_lgb_trials, is_player5)
            best_cat = optimize_catboost(X_scaled, y_target, n_cat_trials, is_player5)
            best_xgb = optimize_xgb(X_scaled, y_target, n_xgb_trials, is_player5)

            # CV with optimized models
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n_samples)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y_target[train_idx], y_target[val_idx]

                # LightGBM
                lgb_model = lgb.LGBMRegressor(**best_lgb, random_state=42, verbose=-1, n_jobs=-1)
                lgb_model.fit(X_tr, y_tr)
                lgb_pred = lgb_model.predict(X_val)

                # CatBoost
                cat_model = CatBoostRegressor(**best_cat, random_state=42, verbose=False)
                cat_model.fit(X_tr, y_tr)
                cat_pred = cat_model.predict(X_val)

                # XGBoost
                xgb_model = xgb.XGBRegressor(**best_xgb, random_state=42, verbosity=0, n_jobs=-1)
                xgb_model.fit(X_tr, y_tr)
                xgb_pred = xgb_model.predict(X_val)

                # Ridge
                ridge_model = Ridge(alpha=1.0, random_state=42)
                ridge_model.fit(X_tr, y_tr)
                ridge_pred = ridge_model.predict(X_val)

                # Weighted average - different weights for Player 5
                if is_player5:
                    # More regularization for Player 5
                    fold_preds[val_idx] = 0.35 * lgb_pred + 0.35 * cat_pred + 0.15 * xgb_pred + 0.15 * ridge_pred
                else:
                    fold_preds[val_idx] = 0.40 * lgb_pred + 0.40 * cat_pred + 0.15 * xgb_pred + 0.05 * ridge_pred

            oof_preds[player_indices, target_idx] = fold_preds

            # Train final models
            lgb_final = lgb.LGBMRegressor(**best_lgb, random_state=42, verbose=-1, n_jobs=-1)
            lgb_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'lgb')] = lgb_final

            cat_final = CatBoostRegressor(**best_cat, random_state=42, verbose=False)
            cat_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'cat')] = cat_final

            xgb_final = xgb.XGBRegressor(**best_xgb, random_state=42, verbosity=0, n_jobs=-1)
            xgb_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'xgb')] = xgb_final

            ridge_final = Ridge(alpha=1.0, random_state=42)
            ridge_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_final

            # Store weights
            if is_player5:
                all_models[(pid, target, 'weights')] = [0.35, 0.35, 0.15, 0.15]
            else:
                all_models[(pid, target, 'weights')] = [0.40, 0.40, 0.15, 0.05]

            mse = np.mean((fold_preds - y_target) ** 2)
            print(f"    {target} CV MSE: {mse:.4f}")

    # Final evaluation
    print("\n" + "="*70)
    print("OVERALL CV RESULTS")
    print("="*70)

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
        "oof_preds": oof_preds,
        "cv_score": avg_scaled_mse,
    }


def predict(data, trained):
    """Generate predictions."""
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
            xgb_pred = models[(pid, target, 'xgb')].predict(x_scaled)[0]
            ridge_pred = models[(pid, target, 'ridge')].predict(x_scaled)[0]

            weights = models[(pid, target, 'weights')]
            predictions[i, target_idx] = (
                weights[0] * lgb_pred +
                weights[1] * cat_pred +
                weights[2] * xgb_pred +
                weights[3] * ridge_pred
            )

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
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}")

    return filepath


def main():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    print("="*70)
    print(f"ULTRA-OPTIMIZED SUBMISSION {next_num}")
    print("="*70)

    data = load_all_data()
    trained = train_ultra_optimized(data)
    predictions = predict(data, trained)
    filepath = create_submission(data["test_ids"], predictions, next_num, trained["cv_score"])

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    return filepath


if __name__ == "__main__":
    main()
