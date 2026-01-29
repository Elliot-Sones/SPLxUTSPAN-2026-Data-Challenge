"""
Simple robust model - prioritize generalization over CV score.

Key insight: Complex models overfit. Simpler models generalize better.
Submission 9 (simpler) beat submission 11 (complex) on leaderboard.
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
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

# Very conservative parameters - prevent overfitting
LGB_PARAMS = {
    "n_estimators": 50,
    "num_leaves": 5,
    "learning_rate": 0.03,
    "max_depth": 3,
    "reg_alpha": 5.0,
    "reg_lambda": 5.0,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
}

CAT_PARAMS = {
    "iterations": 50,
    "depth": 3,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.5,
    "verbose": False,
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data_simple_features():
    """Load data with only hybrid features (no advanced features - they cause overfitting)."""
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping

    print("Loading data with SIMPLE features only...")
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

            # Only hybrid features - simpler, more robust
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

    print(f"Features: {X_train.shape[1]} (simple hybrid only)")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "X_test": X_test,
        "test_ids": np.array(test_ids),
        "test_pids": np.array(test_pids),
        "feature_names": feature_names,
    }


def train_simple_robust(data):
    """Train simple, robust models."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    pids = data["train_pids"]
    feature_names = data["feature_names"]

    unique_pids = sorted(np.unique(pids))

    all_models = {}
    all_scalers = {}
    all_selectors = {}
    oof_preds = np.zeros_like(y_train)

    # Use only top 50 features per target
    N_FEATURES = 50

    print("\n" + "="*70)
    print(f"SIMPLE ROBUST MODEL (top {N_FEATURES} features)")
    print("="*70)

    # Feature selection per target (global, not per-player)
    target_selectors = {}
    for target_idx, target in enumerate(TARGETS):
        selector = SelectKBest(f_regression, k=N_FEATURES)
        selector.fit(np.nan_to_num(X_train, nan=0.0), y_train[:, target_idx])
        target_selectors[target] = selector
        selected = [f for f, m in zip(feature_names, selector.get_support()) if m]
        print(f"  {target}: {len(selected)} features selected")

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X_train[pid_mask]
        y_player = y_train[pid_mask]
        n_samples = len(X_player)

        print(f"\n--- Player {pid} ({n_samples} samples) ---")

        scaler = StandardScaler()
        X_scaled_full = scaler.fit_transform(X_player)
        X_scaled_full = np.nan_to_num(X_scaled_full, nan=0.0)
        all_scalers[pid] = scaler

        player_indices = np.where(pid_mask)[0]

        for target_idx, target in enumerate(TARGETS):
            selector = target_selectors[target]
            X_scaled = selector.transform(X_scaled_full)
            y_target = y_player[:, target_idx]

            all_selectors[(pid, target)] = selector

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n_samples)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y_target[train_idx], y_target[val_idx]

                # LightGBM
                lgb_model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=42)
                lgb_model.fit(X_tr, y_tr)
                lgb_pred = lgb_model.predict(X_val)

                # CatBoost
                cat_model = CatBoostRegressor(**CAT_PARAMS, random_state=42)
                cat_model.fit(X_tr, y_tr)
                cat_pred = cat_model.predict(X_val)

                # Ridge (high alpha for regularization)
                ridge_model = Ridge(alpha=10.0, random_state=42)
                ridge_model.fit(X_tr, y_tr)
                ridge_pred = ridge_model.predict(X_val)

                # Equal weight - simple averaging
                fold_preds[val_idx] = (lgb_pred + cat_pred + ridge_pred) / 3

            oof_preds[player_indices, target_idx] = fold_preds

            # Final models
            lgb_final = lgb.LGBMRegressor(**LGB_PARAMS, random_state=42)
            lgb_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'lgb')] = lgb_final

            cat_final = CatBoostRegressor(**CAT_PARAMS, random_state=42)
            cat_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'cat')] = cat_final

            ridge_final = Ridge(alpha=10.0, random_state=42)
            ridge_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_final

            mse = np.mean((fold_preds - y_target) ** 2)
            print(f"  {target} CV MSE: {mse:.4f}")

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
        raw_mse = np.mean((oof_preds[:, target_idx] - y_train[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_scaled_mse += scaled_mse
        print(f"  {target}: raw MSE = {raw_mse:.4f}, scaled MSE = {scaled_mse:.6f}")

    avg_scaled_mse = total_scaled_mse / 3
    print(f"\n  AVERAGE SCALED MSE: {avg_scaled_mse:.6f}")
    print("  (Note: Higher CV is expected - prioritizing generalization)")

    return {
        "models": all_models,
        "scalers": all_scalers,
        "selectors": target_selectors,
        "cv_score": avg_scaled_mse,
    }


def predict(data, trained):
    """Generate predictions."""
    X_test = data["X_test"]
    test_pids = data["test_pids"]

    models = trained["models"]
    scalers = trained["scalers"]
    selectors = trained["selectors"]

    predictions = np.zeros((len(X_test), 3))

    for i, (x, pid) in enumerate(zip(X_test, test_pids)):
        x_scaled_full = scalers[pid].transform(x.reshape(1, -1))
        x_scaled_full = np.nan_to_num(x_scaled_full, nan=0.0)

        for target_idx, target in enumerate(TARGETS):
            x_scaled = selectors[target].transform(x_scaled_full)

            lgb_pred = models[(pid, target, 'lgb')].predict(x_scaled)[0]
            cat_pred = models[(pid, target, 'cat')].predict(x_scaled)[0]
            ridge_pred = models[(pid, target, 'ridge')].predict(x_scaled)[0]

            predictions[i, target_idx] = (lgb_pred + cat_pred + ridge_pred) / 3

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
    print(f"SIMPLE ROBUST SUBMISSION {next_num}")
    print("="*70)

    data = load_data_simple_features()
    trained = train_simple_robust(data)
    predictions = predict(data, trained)
    create_submission(data["test_ids"], predictions, next_num, trained["cv_score"])

    print("\n" + "="*70)
    print("DONE - Prioritized generalization over CV score")
    print("="*70)


if __name__ == "__main__":
    main()
