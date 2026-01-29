"""
Ensemble submission: Multiple models with stacking.

Strategy:
1. Base models: LightGBM, XGBoost, CatBoost, Ridge
2. Per-player per-target models (15 x 4 = 60 base models)
3. Stacking meta-model: Ridge regression on base predictions
4. Target-specific feature selection
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    """Parse string array to numpy array."""
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    arr = np.array(json.loads(s), dtype=np.float32)
    return arr


def load_train_data():
    """Load training data and extract advanced features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    print("Loading training data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    print(f"Processing {len(train_df)} training shots...")

    all_features = []
    all_targets = []
    all_pids = []

    for idx, row in train_df.iterrows():
        # Parse timeseries
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        # Extract both feature sets
        hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row['participant_id'])

        # Combine
        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)

        # Targets
        all_targets.append([row['angle'], row['depth'], row['left_right']])
        all_pids.append(row['participant_id'])

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    # Convert to arrays
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    return X, y, pids, feature_names


def load_test_data(feature_names):
    """Load test data and extract features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    print("Loading test data...")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in test_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    print(f"Processing {len(test_df)} test shots...")

    all_features = []
    test_ids = []
    test_pids = []

    for idx, row in test_df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row['participant_id'])

        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)

        test_ids.append(row['id'])
        test_pids.append(row['participant_id'])

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)}")

    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)

    print(f"Test features shape: {X.shape}")

    return X, np.array(test_ids), np.array(test_pids)


def get_model_configs():
    """Return model configurations."""
    return {
        "lgb": {
            "class": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 100,
                "num_leaves": 10,
                "learning_rate": 0.05,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
        },
        "lgb_deep": {
            "class": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 200,
                "num_leaves": 31,
                "learning_rate": 0.03,
                "max_depth": 8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "min_child_samples": 10,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
        },
        "xgb": {
            "class": xgb.XGBRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
            }
        },
        "ridge": {
            "class": Ridge,
            "params": {
                "alpha": 1.0,
                "random_state": 42,
            }
        },
        "catboost": {
            "class": CatBoostRegressor,
            "params": {
                "iterations": 100,
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "random_state": 42,
                "verbose": False,
            }
        },
    }


def train_ensemble_cv(X, y, pids, feature_names):
    """
    Train ensemble with cross-validation.
    Returns OOF predictions and trained models.
    """
    print("\n" + "="*70)
    print("TRAINING ENSEMBLE WITH 5-FOLD CV")
    print("="*70)

    model_configs = get_model_configs()
    n_models = len(model_configs)

    # Storage
    oof_preds = {target: np.zeros((len(X), n_models)) for target in TARGETS}
    all_models = {}  # {(player, target, model_name): model}
    all_scalers = {}  # {player: scaler}

    # Group by player
    unique_pids = sorted(np.unique(pids))

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        n_samples = len(X_player)

        print(f"\n--- Player {pid} ({n_samples} samples) ---")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[pid] = scaler

        # 5-fold CV within player
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        player_indices = np.where(pid_mask)[0]

        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_target[train_idx], y_target[val_idx]

                for model_idx, (model_name, config) in enumerate(model_configs.items()):
                    model = config["class"](**config["params"])
                    model.fit(X_train, y_train)

                    # OOF predictions
                    pred = model.predict(X_val)

                    # Map back to global indices
                    global_val_idx = player_indices[val_idx]
                    oof_preds[target][global_val_idx, model_idx] = pred

        # Train final models on all player data
        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]

            for model_name, config in model_configs.items():
                model = config["class"](**config["params"])
                model.fit(X_scaled, y_target)
                all_models[(pid, target, model_name)] = model

    # Evaluate OOF predictions
    print("\n" + "="*70)
    print("OOF EVALUATION")
    print("="*70)

    for target in TARGETS:
        target_idx = TARGETS.index(target)
        y_true = y[:, target_idx]

        for model_idx, model_name in enumerate(model_configs.keys()):
            preds = oof_preds[target][:, model_idx]
            mse = np.mean((preds - y_true) ** 2)
            print(f"  {target} - {model_name}: MSE = {mse:.6f}")

        # Ensemble mean
        mean_pred = np.mean(oof_preds[target], axis=1)
        mse_mean = np.mean((mean_pred - y_true) ** 2)
        print(f"  {target} - ENSEMBLE MEAN: MSE = {mse_mean:.6f}")

    return oof_preds, all_models, all_scalers


def train_meta_model(oof_preds, y):
    """Train meta-model (stacking) on OOF predictions."""
    print("\n" + "="*70)
    print("TRAINING META-MODEL (STACKING)")
    print("="*70)

    meta_models = {}

    for target_idx, target in enumerate(TARGETS):
        X_meta = oof_preds[target]
        y_meta = y[:, target_idx]

        # Use Ridge as meta-learner
        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(X_meta, y_meta)
        meta_models[target] = meta_model

        # Evaluate
        pred = meta_model.predict(X_meta)
        mse = np.mean((pred - y_meta) ** 2)
        print(f"  {target} - META MODEL MSE: {mse:.6f}")

        # Print weights
        print(f"    Weights: {dict(zip(get_model_configs().keys(), meta_model.coef_))}")

    return meta_models


def predict_ensemble(X_test, test_pids, all_models, all_scalers, meta_models):
    """Generate ensemble predictions."""
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)

    model_configs = get_model_configs()
    n_models = len(model_configs)
    n_test = len(X_test)

    predictions = np.zeros((n_test, 3))

    for target_idx, target in enumerate(TARGETS):
        base_preds = np.zeros((n_test, n_models))

        for i, (x, pid) in enumerate(zip(X_test, test_pids)):
            x_scaled = all_scalers[pid].transform(x.reshape(1, -1))
            x_scaled = np.nan_to_num(x_scaled, nan=0.0)

            for model_idx, model_name in enumerate(model_configs.keys()):
                model = all_models[(pid, target, model_name)]
                base_preds[i, model_idx] = model.predict(x_scaled)[0]

        # Apply meta-model
        predictions[:, target_idx] = meta_models[target].predict(base_preds)

    return predictions


def create_submission(test_ids, predictions, submission_num):
    """Create submission CSV."""
    print("\nLoading target scalers...")
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    print("Scaling predictions to [0, 1] range...")
    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions[:, 0],
        'scaled_depth': scaled_predictions[:, 1],
        'scaled_left_right': scaled_predictions[:, 2],
    })

    # Save
    filename = f"submission_{submission_num}.csv"
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\nSubmission saved to: {filepath}")
    print(f"  Rows: {len(submission)}")

    # Stats
    print(f"\nPrediction statistics:")
    for i, col in enumerate(TARGETS):
        print(f"  {col}: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}")

    print(f"\nScaled prediction statistics:")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}, "
              f"min={submission[col].min():.4f}, max={submission[col].max():.4f}")

    return filepath


def main():
    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    print("="*70)
    print(f"ENSEMBLE SUBMISSION {next_num}")
    print("="*70)

    # Load training data
    X_train, y_train, pids_train, feature_names = load_train_data()

    # Train ensemble
    oof_preds, all_models, all_scalers = train_ensemble_cv(
        X_train, y_train, pids_train, feature_names
    )

    # Train meta-model
    meta_models = train_meta_model(oof_preds, y_train)

    # Load test data
    X_test, test_ids, test_pids = load_test_data(feature_names)

    # Generate predictions
    predictions = predict_ensemble(X_test, test_pids, all_models, all_scalers, meta_models)

    # Create submission
    filepath = create_submission(test_ids, predictions, next_num)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    return filepath


if __name__ == "__main__":
    main()
