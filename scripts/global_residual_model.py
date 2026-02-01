"""
Global Model with Residual Prediction

Instead of per-player models, use ONE global model.
This tests if residual prediction helps when the model
doesn't have player-specific parameters.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

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


def global_traditional(data):
    """
    Global model predicting absolute values.
    Uses player ID as a feature.
    """
    print("\n" + "="*60)
    print("GLOBAL MODEL - TRADITIONAL (absolute values)")
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

    # Add player ID as one-hot features
    unique_pids = sorted(np.unique(train_pids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    def add_player_features(X, pids):
        player_onehot = np.zeros((len(X), len(unique_pids)))
        for i, pid in enumerate(pids):
            player_onehot[i, pid_to_idx[pid]] = 1
        return np.hstack([X, player_onehot])

    X_train = add_player_features(train_flat, train_pids)
    X_test = add_player_features(test_flat, test_pids)

    # PCA on motion features only
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(train_flat)
    X_test_pca = pca.transform(test_flat)

    # Add player features back
    X_train_final = add_player_features(X_train_pca, train_pids)
    X_test_final = add_player_features(X_test_pca, test_pids)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    # Use GroupKFold to ensure player appears in both train and val
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for t_idx in range(3):
        fold_test_preds = []

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = train_targets[train_idx, t_idx]

            model = Ridge(alpha=100.0)
            model.fit(X_tr, y_tr)

            oof_preds[val_idx, t_idx] = model.predict(X_val)
            fold_test_preds.append(model.predict(X_test_scaled))

        predictions[:, t_idx] = np.mean(fold_test_preds, axis=0)

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


def global_residual(data):
    """
    Global model predicting residuals from player mean.
    This should help the model focus on shot-specific variations.
    """
    print("\n" + "="*60)
    print("GLOBAL MODEL - RESIDUAL (deviation from player mean)")
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

    # Compute player means
    unique_pids = sorted(np.unique(train_pids))
    player_means = {}
    for pid in unique_pids:
        mask = train_pids == pid
        player_means[pid] = train_targets[mask].mean(axis=0)

    # Convert targets to residuals
    train_residuals = np.zeros_like(train_targets)
    for i, pid in enumerate(train_pids):
        train_residuals[i] = train_targets[i] - player_means[pid]

    # Add player features
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    def add_player_features(X, pids):
        player_onehot = np.zeros((len(X), len(unique_pids)))
        for i, pid in enumerate(pids):
            player_onehot[i, pid_to_idx[pid]] = 1
        return np.hstack([X, player_onehot])

    # PCA on motion features
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(train_flat)
    X_test_pca = pca.transform(test_flat)

    X_train_final = add_player_features(X_train_pca, train_pids)
    X_test_final = add_player_features(X_test_pca, test_pids)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for t_idx in range(3):
        fold_test_preds = []

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = train_residuals[train_idx, t_idx]

            model = Ridge(alpha=100.0)
            model.fit(X_tr, y_tr)

            # Predict residual
            val_residual = model.predict(X_val)
            test_residual = model.predict(X_test_scaled)

            # Add back player means
            for i, idx in enumerate(val_idx):
                pid = train_pids[idx]
                oof_preds[idx, t_idx] = val_residual[i] + player_means[pid][t_idx]

            test_preds_fold = np.zeros(len(test_series))
            for i, pid in enumerate(test_pids):
                test_preds_fold[i] = test_residual[i] + player_means[pid][t_idx]
            fold_test_preds.append(test_preds_fold)

        predictions[:, t_idx] = np.mean(fold_test_preds, axis=0)

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


def global_residual_no_player_features(data):
    """
    Global model predicting residuals WITHOUT player ID features.
    The model must learn purely from motion data.
    """
    print("\n" + "="*60)
    print("GLOBAL MODEL - RESIDUAL (no player ID features)")
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

    # Compute player means
    unique_pids = sorted(np.unique(train_pids))
    player_means = {}
    for pid in unique_pids:
        mask = train_pids == pid
        player_means[pid] = train_targets[mask].mean(axis=0)

    # Convert targets to residuals
    train_residuals = np.zeros_like(train_targets)
    for i, pid in enumerate(train_pids):
        train_residuals[i] = train_targets[i] - player_means[pid]

    # PCA on motion features only
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(train_flat)
    X_test_pca = pca.transform(test_flat)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for t_idx in range(3):
        fold_test_preds = []

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = train_residuals[train_idx, t_idx]

            model = Ridge(alpha=100.0)
            model.fit(X_tr, y_tr)

            # Predict residual
            val_residual = model.predict(X_val)
            test_residual = model.predict(X_test_scaled)

            # Add back player means
            for i, idx in enumerate(val_idx):
                pid = train_pids[idx]
                oof_preds[idx, t_idx] = val_residual[i] + player_means[pid][t_idx]

            test_preds_fold = np.zeros(len(test_series))
            for i, pid in enumerate(test_pids):
                test_preds_fold[i] = test_residual[i] + player_means[pid][t_idx]
            fold_test_preds.append(test_preds_fold)

        predictions[:, t_idx] = np.mean(fold_test_preds, axis=0)

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
    print("GLOBAL MODEL WITH RESIDUAL PREDICTION")
    print("="*80)

    data = load_data()

    results = []

    # Global traditional
    trad_preds, trad_cv, _ = global_traditional(data)
    results.append(("global_traditional", trad_cv, trad_preds))

    # Global residual with player features
    resid_preds, resid_cv, _ = global_residual(data)
    results.append(("global_residual", resid_cv, resid_preds))

    # Global residual without player features
    resid_no_pid_preds, resid_no_pid_cv, _ = global_residual_no_player_features(data)
    results.append(("global_residual_no_pid", resid_no_pid_cv, resid_no_pid_preds))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Approach':<30} {'CV Score':>12}")
    print("-" * 45)
    for name, cv, _ in sorted(results, key=lambda x: x[1]):
        print(f"{name:<30} {cv:>12.6f}")

    # Create submission with best
    best_name, best_cv, best_preds = min(results, key=lambda x: x[1])
    print(f"\nBest approach: {best_name}")

    create_submission(data["test_ids"], best_preds, best_cv, best_name)


if __name__ == "__main__":
    main()
