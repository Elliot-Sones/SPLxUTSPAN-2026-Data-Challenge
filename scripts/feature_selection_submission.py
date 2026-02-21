"""
Submission using aggressive feature selection (top 20 features per target).
This technique showed 18% CV improvement in testing.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    return np.array(json.loads(s.replace("nan", "null")), dtype=np.float32)

def extract_features(X_raw):
    """Extract comprehensive features."""
    features = np.concatenate([
        np.nanmean(X_raw, axis=1),
        np.nanstd(X_raw, axis=1),
        np.nanmax(X_raw, axis=1) - np.nanmin(X_raw, axis=1),
        np.nanpercentile(X_raw, 25, axis=1),
        np.nanpercentile(X_raw, 75, axis=1),
    ], axis=1)
    return np.nan_to_num(features, nan=0.0)

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols + TARGETS]

    n_train = len(train_df)
    X_train_raw = np.zeros((n_train, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in train_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_train_raw[idx, :, i] = parse_array_string(row[col])

    y_train = train_df[TARGETS].values.astype(np.float32)

    n_test = len(test_df)
    X_test_raw = np.zeros((n_test, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in test_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_test_raw[idx, :, i] = parse_array_string(row[col])

    test_ids = test_df["id"].values

    return X_train_raw, y_train, X_test_raw, test_ids

def main():
    X_train_raw, y_train, X_test_raw, test_ids = load_data()

    print("Extracting features...")
    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)
    print(f"  Features: {X_train.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining with top 20 features per target...")
    predictions = np.zeros((len(X_test), 3))

    for t_idx, target in enumerate(TARGETS):
        print(f"\n  {target}:")

        # Select top 20 features
        selector = SelectKBest(mutual_info_regression, k=20)
        X_train_sel = selector.fit_transform(X_train_scaled, y_train[:, t_idx])
        X_test_sel = selector.transform(X_test_scaled)

        # Train ensemble
        preds = []

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=8, learning_rate=0.03,
            reg_alpha=1.0, reg_lambda=1.0, verbose=-1
        )
        lgb_model.fit(X_train_sel, y_train[:, t_idx])
        preds.append(lgb_model.predict(X_test_sel))

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            reg_alpha=1.0, reg_lambda=1.0, verbosity=0
        )
        xgb_model.fit(X_train_sel, y_train[:, t_idx])
        preds.append(xgb_model.predict(X_test_sel))

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=200, depth=3, learning_rate=0.03,
            l2_leaf_reg=5.0, verbose=False
        )
        cat_model.fit(X_train_sel, y_train[:, t_idx])
        preds.append(cat_model.predict(X_test_sel))

        predictions[:, t_idx] = np.mean(preds, axis=0)
        print(f"    mean={predictions[:, t_idx].mean():.4f}, std={predictions[:, t_idx].std():.4f}")

    # Scale predictions
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled[:, i] = scalers[target].transform(predictions[:, i].reshape(-1, 1)).flatten()

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    # Save
    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled[:, 0],
        'scaled_depth': scaled[:, 1],
        'scaled_left_right': scaled[:, 2],
    })
    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    sub.to_csv(filepath, index=False)

    print(f"\nSubmission saved: {filepath}")
    print(f"  angle_std = {sub['scaled_angle'].std():.6f}")
    print(f"  depth_mean = {sub['scaled_depth'].mean():.6f}")

    print(f"\n=== Submission {next_num} ===")
    print("Method: Top 20 features per target + LGB/XGB/Cat ensemble")
    print(f"File: {filepath}")

if __name__ == "__main__":
    main()
