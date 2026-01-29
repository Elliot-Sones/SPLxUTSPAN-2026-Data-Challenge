"""
Minimal ensemble - same approach as submission 9 but ONLY hybrid features.

Hypothesis: Advanced features cause overfitting. Simpler features generalize better.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

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


def load_data_hybrid_only():
    """Load data with ONLY hybrid features - no advanced features."""
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping

    print("Loading data with hybrid features ONLY...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)

    def extract_features(df, is_train=True):
        all_features = []
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            all_features.append(feats)

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        return all_features, ids, pids, targets if is_train else None

    print(f"Processing {len(train_df)} training shots...")
    train_feats, train_ids, train_pids, train_targets = extract_features(train_df, True)

    print(f"Processing {len(test_df)} test shots...")
    test_feats, test_ids, test_pids, _ = extract_features(test_df, False)

    feature_names = sorted(train_feats[0].keys())
    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)

    print(f"Features: {X_train.shape[1]} (hybrid only)")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "X_test": X_test,
        "test_ids": np.array(test_ids),
        "test_pids": np.array(test_pids),
        "feature_names": feature_names,
    }


# Same model configs as submission 9
def get_model_configs():
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


def train_minimal_ensemble(data):
    """Train minimal ensemble - same as sub9 but fewer features."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    pids = data["train_pids"]

    unique_pids = sorted(np.unique(pids))
    model_configs = get_model_configs()
    n_models = len(model_configs)

    oof_preds = {target: np.zeros((len(X_train), n_models)) for target in TARGETS}
    all_models = {}
    all_scalers = {}

    print("\n" + "="*70)
    print("MINIMAL ENSEMBLE (hybrid features only)")
    print("="*70)

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X_train[pid_mask]
        y_player = y_train[pid_mask]
        n_samples = len(X_player)

        print(f"\n--- Player {pid} ({n_samples} samples) ---")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[pid] = scaler

        player_indices = np.where(pid_mask)[0]

        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y_target[train_idx], y_target[val_idx]

                for model_idx, (model_name, config) in enumerate(model_configs.items()):
                    model = config["class"](**config["params"])
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    global_val_idx = player_indices[val_idx]
                    oof_preds[target][global_val_idx, model_idx] = pred

        # Train final models
        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]
            for model_name, config in model_configs.items():
                model = config["class"](**config["params"])
                model.fit(X_scaled, y_target)
                all_models[(pid, target, model_name)] = model

    # Train meta-models
    print("\n" + "="*70)
    print("TRAINING META-MODELS")
    print("="*70)

    meta_models = {}
    for target_idx, target in enumerate(TARGETS):
        X_meta = oof_preds[target]
        y_meta = y_train[:, target_idx]

        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(X_meta, y_meta)
        meta_models[target] = meta_model

        pred = meta_model.predict(X_meta)
        mse = np.mean((pred - y_meta) ** 2)
        print(f"  {target} meta MSE: {mse:.4f}")
        print(f"    Weights: {dict(zip(model_configs.keys(), meta_model.coef_))}")

    # Evaluation
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
        pred = meta_models[target].predict(oof_preds[target])
        raw_mse = np.mean((pred - y_train[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_scaled_mse += scaled_mse
        print(f"  {target}: raw MSE = {raw_mse:.4f}, scaled MSE = {scaled_mse:.6f}")

    avg_scaled_mse = total_scaled_mse / 3
    print(f"\n  AVERAGE SCALED MSE: {avg_scaled_mse:.6f}")

    return {
        "models": all_models,
        "scalers": all_scalers,
        "meta_models": meta_models,
        "cv_score": avg_scaled_mse,
    }


def predict(data, trained):
    """Generate predictions."""
    X_test = data["X_test"]
    test_pids = data["test_pids"]

    models = trained["models"]
    scalers = trained["scalers"]
    meta_models = trained["meta_models"]
    model_configs = get_model_configs()
    n_models = len(model_configs)

    predictions = np.zeros((len(X_test), 3))

    for target_idx, target in enumerate(TARGETS):
        base_preds = np.zeros((len(X_test), n_models))

        for i, (x, pid) in enumerate(zip(X_test, test_pids)):
            x_scaled = scalers[pid].transform(x.reshape(1, -1))
            x_scaled = np.nan_to_num(x_scaled, nan=0.0)

            for model_idx, model_name in enumerate(model_configs.keys()):
                model = models[(pid, target, model_name)]
                base_preds[i, model_idx] = model.predict(x_scaled)[0]

        predictions[:, target_idx] = meta_models[target].predict(base_preds)

    return predictions


def create_submission(test_ids, predictions, submission_num, cv_score):
    """Create submission."""
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
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    print("="*70)
    print(f"MINIMAL ENSEMBLE SUBMISSION {next_num}")
    print("="*70)

    data = load_data_hybrid_only()
    trained = train_minimal_ensemble(data)
    predictions = predict(data, trained)
    create_submission(data["test_ids"], predictions, next_num, trained["cv_score"])

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
