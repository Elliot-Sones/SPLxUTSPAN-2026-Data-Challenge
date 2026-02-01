"""
Sub9 Model CV Score

Runs the exact sub9 model architecture with CV to get a comparable score.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

TARGETS = ['angle', 'depth', 'left_right']

# Load target scalers
TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_model_configs():
    """Same configs as sub9."""
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
            "params": {"alpha": 1.0, "random_state": 42}
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


def load_train_data():
    """Load training data with sub9 features."""
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
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)

        all_targets.append([row['angle'], row['depth'], row['left_right']])
        all_pids.append(row['participant_id'])

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    print(f"Features shape: {X.shape}")
    return X, y, pids, feature_names


def main():
    print("=" * 80)
    print("SUB9 MODEL CV SCORE")
    print("Using exact sub9 architecture and features")
    print("=" * 80)

    X, y, pids, feature_names = load_train_data()

    model_configs = get_model_configs()
    n_models = len(model_configs)
    unique_pids = sorted(np.unique(pids))

    # OOF predictions for each base model
    oof_preds = {target: np.zeros((len(X), n_models)) for target in TARGETS}

    print("\nTraining with 5-fold CV per player...")

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        n_samples = len(X_player)

        print(f"  Player {pid} ({n_samples} samples)")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

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
                    pred = model.predict(X_val)
                    global_val_idx = player_indices[val_idx]
                    oof_preds[target][global_val_idx, model_idx] = pred

    # Meta-stacking with CV
    print("\nMeta-stacking...")
    final_preds = {}

    for target_idx, target in enumerate(TARGETS):
        X_meta = oof_preds[target]
        y_target = y[:, target_idx]

        meta_preds = np.zeros(len(y_target))
        kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf_meta.split(X_meta):
            X_train, X_val = X_meta[train_idx], X_meta[val_idx]
            y_train = y_target[train_idx]

            meta_model = Ridge(alpha=1.0, random_state=42)
            meta_model.fit(X_train, y_train)
            meta_preds[val_idx] = meta_model.predict(X_val)

        final_preds[target] = meta_preds

    # Calculate MSE in scaled [0,1] space
    print("\n" + "=" * 80)
    print("SUB9 CV SCORE (scaled to [0,1])")
    print("=" * 80)

    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        scaler = TARGET_SCALERS[target]
        scaled_preds = scaler.transform(final_preds[target].reshape(-1, 1)).flatten()
        scaled_truth = scaler.transform(y[:, target_idx].reshape(-1, 1)).flatten()

        scaled_preds = np.clip(scaled_preds, 0, 1)
        scaled_truth = np.clip(scaled_truth, 0, 1)

        target_mse = np.mean((scaled_preds - scaled_truth) ** 2)
        print(f"  {target}: MSE = {target_mse:.6f}")
        total_mse += target_mse

    avg_mse = total_mse / 3
    print(f"\n  OVERALL CV MSE: {avg_mse:.6f}")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"  Sub9 CV Score:  {avg_mse:.6f}")
    print(f"  Sub9 LB Score:  0.009109")
    print(f"  Physics CV:     0.006372")

    if avg_mse < 0.006372:
        print(f"\n  SUB9 CV WINS vs Physics by {(0.006372 - avg_mse)/0.006372*100:.1f}%")
    else:
        print(f"\n  PHYSICS CV WINS vs Sub9 CV by {(avg_mse - 0.006372)/avg_mse*100:.1f}%")


if __name__ == "__main__":
    main()
