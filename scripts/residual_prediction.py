"""
Residual Prediction - A fundamentally different formulation

Instead of predicting: angle = f(shot)
Predict: angle - player_mean_angle = f(shot)

This removes the player-specific baseline that causes overfitting.
The model only learns what makes THIS shot different from the player's typical shot.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_target_ranges():
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }


def load_data():
    """Load data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def extract_series(df):
        all_series = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])
            all_series.append(ts)
        return np.array(all_series)

    print("Loading training data...")
    train_series = extract_series(train_df)
    train_targets = train_df[["angle", "depth", "left_right"]].values
    train_pids = train_df["participant_id"].values
    train_ids = train_df["id"].values

    print("Loading test data...")
    test_series = extract_series(test_df)
    test_pids = test_df["participant_id"].values
    test_ids = test_df["id"].values

    return {
        "train_series": train_series,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_series": test_series,
        "test_pids": test_pids,
        "test_ids": test_ids,
        "keypoint_cols": keypoint_cols,
    }


def traditional_prediction(data):
    """
    Traditional approach: predict absolute values.
    """
    print("\n" + "="*60)
    print("TRADITIONAL PREDICTION (baseline)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # PCA + Ridge
        n_comp = min(15, len(X_train) - 1)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx, t_idx]

                model = Ridge(alpha=100.0)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, oof_preds


def residual_prediction(data):
    """
    Residual approach: predict deviation from player mean.

    y_residual = y - player_mean
    model predicts y_residual
    final prediction = player_mean + predicted_residual
    """
    print("\n" + "="*60)
    print("RESIDUAL PREDICTION (new formulation)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)
    player_means = {}

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # Compute player means
        player_mean = y_train.mean(axis=0)
        player_means[pid] = player_mean

        # Convert to residuals
        y_residual = y_train - player_mean

        # PCA + Ridge on residuals
        n_comp = min(15, len(X_train) - 1)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_residual[train_idx, t_idx]

                model = Ridge(alpha=100.0)
                model.fit(X_tr, y_tr)

                # Predict residual, then add back player mean
                pred_residual = model.predict(X_val)
                oof_preds[player_indices[val_idx], t_idx] = pred_residual + player_mean[t_idx]

                test_residual = model.predict(X_test_scaled)
                fold_test_preds.append(test_residual + player_mean[t_idx])

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, oof_preds, player_means


def residual_with_shrinkage(data, shrink_factor=0.5):
    """
    Residual prediction with shrinkage.

    Shrink the predicted residual toward zero (conservative).
    This reduces overconfidence in residual predictions.
    """
    print("\n" + "="*60)
    print(f"RESIDUAL WITH SHRINKAGE (factor={shrink_factor})")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        player_mean = y_train.mean(axis=0)
        y_residual = y_train - player_mean

        n_comp = min(15, len(X_train) - 1)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_residual[train_idx, t_idx]

                model = Ridge(alpha=100.0)
                model.fit(X_tr, y_tr)

                # Predict residual, shrink it, then add back player mean
                pred_residual = model.predict(X_val)
                shrunk_residual = pred_residual * (1 - shrink_factor)
                oof_preds[player_indices[val_idx], t_idx] = shrunk_residual + player_mean[t_idx]

                test_residual = model.predict(X_test_scaled)
                shrunk_test_residual = test_residual * (1 - shrink_factor)
                fold_test_preds.append(shrunk_test_residual + player_mean[t_idx])

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Compute CV
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, oof_preds


def create_submission(test_ids, predictions, cv_score, approach_name):
    """Create submission."""
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        name = f.stem
        if name.startswith("submission_"):
            try:
                nums.append(int(name.split('_')[1]))
            except:
                pass
        elif name.startswith("submission"):
            try:
                nums.append(int(name[10:]))
            except:
                pass

    next_num = max(nums) + 1 if nums else 1

    scaled_preds = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    pred_lb = 0.0038 + 0.034 * angle_std

    print(f"\nSubmission {next_num}: {approach_name}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")
    print(f"  Predicted LB: {pred_lb:.6f}")

    return filepath, next_num


def main():
    print("="*80)
    print("RESIDUAL PREDICTION - DIFFERENT PROBLEM FORMULATION")
    print("="*80)
    print()
    print("Key idea: Predict deviation from player mean, not absolute value")
    print("This removes player-specific bias that causes overfitting")

    data = load_data()

    results = []

    # Traditional approach
    trad_preds, trad_cv, _ = traditional_prediction(data)
    results.append(("traditional", trad_cv, trad_preds))

    # Residual approach
    resid_preds, resid_cv, _, _ = residual_prediction(data)
    results.append(("residual", resid_cv, resid_preds))

    # Residual with different shrinkage factors
    for shrink in [0.3, 0.5, 0.7]:
        shrink_preds, shrink_cv, _ = residual_with_shrinkage(data, shrink_factor=shrink)
        results.append((f"residual_shrink_{shrink}", shrink_cv, shrink_preds))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Approach':<25} {'CV Score':>12}")
    print("-" * 40)
    for name, cv, _ in sorted(results, key=lambda x: x[1]):
        print(f"{name:<25} {cv:>12.6f}")

    # Create submission with best approach
    best_name, best_cv, best_preds = min(results, key=lambda x: x[1])
    print(f"\nBest approach: {best_name}")

    create_submission(data["test_ids"], best_preds, best_cv, best_name)

    # Also create residual submission for comparison
    if best_name != "residual":
        create_submission(data["test_ids"], resid_preds, resid_cv, "residual")


if __name__ == "__main__":
    main()
