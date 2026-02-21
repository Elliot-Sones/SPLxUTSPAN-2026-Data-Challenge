"""
Submission using Savitzky-Golay denoising + existing best features.
Applies Savitzky-Golay filter before feature extraction.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
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

def apply_savgol(X, window=11, poly=3):
    """Apply Savitzky-Golay denoising."""
    X_out = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            s = X[i, :, j]
            valid = ~np.isnan(s)
            if valid.sum() >= window:
                s_clean = np.interp(np.arange(len(s)), np.where(valid)[0], s[valid])
                X_out[i, :, j] = savgol_filter(s_clean, window, poly)
            else:
                X_out[i, :, j] = s
    return X_out

def extract_features(X):
    """Extract comprehensive features from denoised data."""
    features = []
    for i in range(X.shape[0]):
        f = {}
        for j in range(X.shape[2]):
            s = X[i, :, j]

            # Basic stats
            f[f'f{j}_mean'] = np.nanmean(s)
            f[f'f{j}_std'] = np.nanstd(s)
            f[f'f{j}_min'] = np.nanmin(s)
            f[f'f{j}_max'] = np.nanmax(s)
            f[f'f{j}_range'] = f[f'f{j}_max'] - f[f'f{j}_min']

            # Velocity (first derivative)
            vel = np.gradient(s)
            f[f'f{j}_vel_mean'] = np.nanmean(vel)
            f[f'f{j}_vel_std'] = np.nanstd(vel)
            f[f'f{j}_vel_max'] = np.nanmax(np.abs(vel))

            # Frame of max velocity
            f[f'f{j}_vel_max_frame'] = np.nanargmax(np.abs(vel))

        features.append(f)

    # Convert to DataFrame
    df = pd.DataFrame(features)
    return df.values, list(df.columns)

def load_data():
    print("Loading training data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = ["id", "shot_id", "participant_id"] + TARGETS
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    n = len(train_df)
    X_train = np.zeros((n, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in train_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_train[idx, :, i] = parse_array_string(row[col])

    y_train = train_df[TARGETS].values.astype(np.float32)
    pids_train = train_df["participant_id"].values

    print("Loading test data...")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    n_test = len(test_df)
    X_test = np.zeros((n_test, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in test_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_test[idx, :, i] = parse_array_string(row[col])

    test_ids = test_df["id"].values

    return X_train, y_train, pids_train, X_test, test_ids

def train_and_predict():
    X_train_raw, y_train, pids_train, X_test_raw, test_ids = load_data()

    print("Applying Savitzky-Golay denoising...")
    X_train_sg = apply_savgol(X_train_raw)
    X_test_sg = apply_savgol(X_test_raw)

    print("Extracting features...")
    X_train, feature_names = extract_features(X_train_sg)
    X_test, _ = extract_features(X_test_sg)
    print(f"Features: {X_train.shape[1]}")

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    print("\nTraining ensemble...")
    predictions = np.zeros((len(X_test), 3))

    for t_idx, target in enumerate(TARGETS):
        print(f"\n  {target}:")

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=150, num_leaves=15, learning_rate=0.03,
            reg_alpha=0.5, reg_lambda=0.5, verbose=-1, n_jobs=-1
        )
        lgb_model.fit(X_train, y_train[:, t_idx])
        lgb_pred = lgb_model.predict(X_test)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            reg_alpha=0.5, reg_lambda=0.5, verbosity=0, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train[:, t_idx])
        xgb_pred = xgb_model.predict(X_test)

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=150, depth=4, learning_rate=0.03,
            l2_leaf_reg=3.0, verbose=False
        )
        cat_model.fit(X_train, y_train[:, t_idx])
        cat_pred = cat_model.predict(X_test)

        # Ensemble average
        predictions[:, t_idx] = (lgb_pred + xgb_pred + cat_pred) / 3
        print(f"    pred mean={predictions[:, t_idx].mean():.4f}, std={predictions[:, t_idx].std():.4f}")

    return predictions, test_ids

def create_submission(predictions, test_ids):
    # Load scalers
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Scale predictions
    scaled = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled[:, i] = scalers[target].transform(predictions[:, i].reshape(-1, 1)).flatten()

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    # Create submission
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

    return filepath, next_num

if __name__ == "__main__":
    print("="*60)
    print("SAVITZKY-GOLAY DENOISED SUBMISSION")
    print("="*60)

    predictions, test_ids = train_and_predict()
    filepath, sub_num = create_submission(predictions, test_ids)

    print(f"\n=== Submission {sub_num} ===")
    print("Method: Savitzky-Golay denoising (window=11, poly=3) + LGB/XGB/Cat ensemble")
    print(f"File: {filepath}")
