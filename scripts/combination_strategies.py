"""
Combination Strategies - Comprehensive Test

Test many different ways to combine models/predictions.
Goal: Find strategy that minimizes both CV and angle_std.
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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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


def compute_cv_and_std(oof_preds, train_targets, test_preds):
    """Compute CV score and angle_std."""
    ranges = get_target_ranges()

    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        total_mse += scaled_mse

    cv_score = total_mse / 3

    # Compute angle_std on scaled predictions
    scaled_angle = TARGET_SCALERS["angle"].transform(
        test_preds[:, 0].reshape(-1, 1)
    ).flatten()
    angle_std = np.std(scaled_angle)

    return cv_score, angle_std


def base_per_player_model(data, frames, alpha=100.0, n_components=15):
    """Base per-player model - returns OOF and test predictions."""
    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

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

        n_comp = min(n_components, len(X_train) - 1)
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

                model = Ridge(alpha=alpha)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    return oof_preds, predictions


# ============================================================
# STRATEGY 1: Regularization Ladder
# Combine models with different regularization strengths
# ============================================================
def strategy_regularization_ladder(data):
    """Blend models with different alphas."""
    print("\n  Strategy 1: Regularization Ladder")

    frames = list(range(140, 170))
    alphas = [10, 100, 500, 1000]

    all_oof = []
    all_test = []

    for alpha in alphas:
        oof, test = base_per_player_model(data, frames, alpha=alpha)
        all_oof.append(oof)
        all_test.append(test)

    # Simple average
    oof_blend = np.mean(all_oof, axis=0)
    test_blend = np.mean(all_test, axis=0)

    cv, angle_std = compute_cv_and_std(oof_blend, data["train_targets"], test_blend)
    return cv, angle_std, oof_blend, test_blend


# ============================================================
# STRATEGY 2: Frame Window Ensemble
# Combine models from different temporal windows
# ============================================================
def strategy_frame_window_ensemble(data):
    """Blend models from different frame windows."""
    print("\n  Strategy 2: Frame Window Ensemble")

    windows = [
        list(range(135, 160)),  # Early
        list(range(145, 165)),  # Narrow
        list(range(140, 170)),  # Standard
        list(range(150, 180)),  # Late
    ]

    all_oof = []
    all_test = []

    for frames in windows:
        oof, test = base_per_player_model(data, frames, alpha=100)
        all_oof.append(oof)
        all_test.append(test)

    oof_blend = np.mean(all_oof, axis=0)
    test_blend = np.mean(all_test, axis=0)

    cv, angle_std = compute_cv_and_std(oof_blend, data["train_targets"], test_blend)
    return cv, angle_std, oof_blend, test_blend


# ============================================================
# STRATEGY 3: PCA Component Ladder
# Combine models with different numbers of PCA components
# ============================================================
def strategy_pca_ladder(data):
    """Blend models with different PCA components."""
    print("\n  Strategy 3: PCA Component Ladder")

    frames = list(range(140, 170))
    n_components_list = [5, 10, 15, 20]

    all_oof = []
    all_test = []

    for n_comp in n_components_list:
        oof, test = base_per_player_model(data, frames, alpha=100, n_components=n_comp)
        all_oof.append(oof)
        all_test.append(test)

    oof_blend = np.mean(all_oof, axis=0)
    test_blend = np.mean(all_test, axis=0)

    cv, angle_std = compute_cv_and_std(oof_blend, data["train_targets"], test_blend)
    return cv, angle_std, oof_blend, test_blend


# ============================================================
# STRATEGY 4: Hierarchical Target Prediction
# Predict one target first, use as feature for others
# ============================================================
def strategy_hierarchical(data):
    """Predict depth first, use for angle prediction."""
    print("\n  Strategy 4: Hierarchical Target Prediction")

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    # First pass: predict all targets independently and get OOF
    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        n_comp = min(15, len(X_train) - 1)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # First predict depth (t_idx=1)
        fold_test_preds_depth = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx, 1]  # depth

            model = Ridge(alpha=100)
            model.fit(X_tr, y_tr)

            oof_preds[player_indices[val_idx], 1] = model.predict(X_val)
            fold_test_preds_depth.append(model.predict(X_test_scaled))

        predictions[test_mask, 1] = np.mean(fold_test_preds_depth, axis=0)

        # Now predict angle using depth as additional feature
        X_train_aug = np.hstack([X_train_scaled, oof_preds[train_mask, 1:2]])

        fold_test_preds_angle = []
        for train_idx, val_idx in kf.split(X_train_aug):
            X_tr = X_train_aug[train_idx]
            X_val = X_train_aug[val_idx]
            y_tr = y_train[train_idx, 0]  # angle

            model = Ridge(alpha=100)
            model.fit(X_tr, y_tr)

            oof_preds[player_indices[val_idx], 0] = model.predict(X_val)

            # For test, use predicted depth
            X_test_aug = np.hstack([X_test_scaled, predictions[test_mask, 1:2]])
            fold_test_preds_angle.append(model.predict(X_test_aug))

        predictions[test_mask, 0] = np.mean(fold_test_preds_angle, axis=0)

        # Predict left_right normally
        fold_test_preds_lr = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx, 2]

            model = Ridge(alpha=100)
            model.fit(X_tr, y_tr)

            oof_preds[player_indices[val_idx], 2] = model.predict(X_val)
            fold_test_preds_lr.append(model.predict(X_test_scaled))

        predictions[test_mask, 2] = np.mean(fold_test_preds_lr, axis=0)

    cv, angle_std = compute_cv_and_std(oof_preds, train_targets, predictions)
    return cv, angle_std, oof_preds, predictions


# ============================================================
# STRATEGY 5: Body Part Ensemble
# Separate models for different body parts
# ============================================================
def strategy_body_part_ensemble(data):
    """Combine predictions from different body part subsets."""
    print("\n  Strategy 5: Body Part Ensemble")

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]
    keypoint_cols = data["keypoint_cols"]

    # Define body part groups
    body_parts = {
        "arms": [i for i, c in enumerate(keypoint_cols) if any(x in c.lower() for x in ["wrist", "elbow", "shoulder"])],
        "hands": [i for i, c in enumerate(keypoint_cols) if any(x in c.lower() for x in ["thumb", "pinky", "index"])],
        "torso": [i for i, c in enumerate(keypoint_cols) if any(x in c.lower() for x in ["hip", "spine", "chest"])],
    }

    frames = list(range(140, 170))
    unique_pids = sorted(np.unique(train_pids))

    all_oof = []
    all_test = []

    for part_name, part_indices in body_parts.items():
        if len(part_indices) == 0:
            continue

        def flatten(series):
            flat = series[:, frames, :][:, :, part_indices].reshape(len(series), -1)
            return np.nan_to_num(flat, nan=0.0)

        train_flat = flatten(train_series)
        test_flat = flatten(test_series)

        predictions = np.zeros((len(test_series), 3))
        oof_preds = np.zeros_like(train_targets)

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_flat[train_mask]
            y_train = train_targets[train_mask]
            X_test = test_flat[test_mask]

            player_indices = np.where(train_mask)[0]

            n_comp = min(10, len(X_train) - 1, X_train.shape[1] - 1)
            if n_comp < 2:
                continue

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

                    model = Ridge(alpha=100)
                    model.fit(X_tr, y_tr)

                    oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                    fold_test_preds.append(model.predict(X_test_scaled))

                predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        all_oof.append(oof_preds)
        all_test.append(predictions)

    oof_blend = np.mean(all_oof, axis=0)
    test_blend = np.mean(all_test, axis=0)

    cv, angle_std = compute_cv_and_std(oof_blend, train_targets, test_blend)
    return cv, angle_std, oof_blend, test_blend


# ============================================================
# STRATEGY 6: Residual Boosting
# Model 1 predicts baseline, Model 2 predicts residual
# ============================================================
def strategy_residual_boosting(data):
    """Two-stage: base prediction + residual correction."""
    print("\n  Strategy 6: Residual Boosting")

    train_targets = data["train_targets"]

    # Stage 1: Conservative model (high regularization)
    frames1 = list(range(140, 170))
    oof1, test1 = base_per_player_model(data, frames1, alpha=500, n_components=10)

    # Stage 2: Predict residuals with aggressive model
    residuals = train_targets - oof1

    # Create modified data with residuals as targets
    data_resid = data.copy()
    data_resid["train_targets"] = residuals

    frames2 = list(range(145, 165))
    oof2, test2 = base_per_player_model(data_resid, frames2, alpha=50, n_components=15)

    # Combine: base + 0.5 * residual (shrinkage)
    shrink = 0.3
    oof_final = oof1 + shrink * oof2
    test_final = test1 + shrink * test2

    cv, angle_std = compute_cv_and_std(oof_final, train_targets, test_final)
    return cv, angle_std, oof_final, test_final


# ============================================================
# STRATEGY 7: Player Mean Shrinkage Blend
# Blend prediction with player mean
# ============================================================
def strategy_player_mean_shrinkage(data):
    """Blend model prediction with player mean."""
    print("\n  Strategy 7: Player Mean Shrinkage")

    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    frames = list(range(140, 170))
    oof_model, test_model = base_per_player_model(data, frames, alpha=100)

    # Compute player means
    unique_pids = sorted(np.unique(train_pids))
    player_means = {}
    for pid in unique_pids:
        mask = train_pids == pid
        player_means[pid] = train_targets[mask].mean(axis=0)

    # Create mean predictions
    oof_mean = np.zeros_like(oof_model)
    test_mean = np.zeros((len(test_pids), 3))

    for i, pid in enumerate(train_pids):
        oof_mean[i] = player_means[pid]
    for i, pid in enumerate(test_pids):
        test_mean[i] = player_means[pid]

    # Blend: weight toward mean for stability
    best_cv = float('inf')
    best_blend = None
    best_weight = 0

    for weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
        oof_blend = (1 - weight) * oof_model + weight * oof_mean
        test_blend = (1 - weight) * test_model + weight * test_mean

        cv, angle_std = compute_cv_and_std(oof_blend, train_targets, test_blend)

        if cv < best_cv:
            best_cv = cv
            best_blend = (oof_blend, test_blend)
            best_weight = weight

    cv, angle_std = compute_cv_and_std(best_blend[0], train_targets, best_blend[1])
    print(f"    Best weight toward mean: {best_weight}")
    return cv, angle_std, best_blend[0], best_blend[1]


# ============================================================
# STRATEGY 8: KNN + Ridge Ensemble
# Combine different model types
# ============================================================
def strategy_model_type_ensemble(data):
    """Combine Ridge and KNN predictions."""
    print("\n  Strategy 8: Model Type Ensemble (Ridge + KNN)")

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    # Ridge predictions
    oof_ridge, test_ridge = base_per_player_model(data, frames, alpha=100)

    # KNN predictions
    oof_knn = np.zeros_like(train_targets)
    test_knn = np.zeros((len(test_series), 3))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

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

                k = min(5, len(X_tr) - 1)
                model = KNeighborsRegressor(n_neighbors=k, weights='distance')
                model.fit(X_tr, y_tr)

                oof_knn[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            test_knn[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Blend Ridge and KNN
    oof_blend = 0.7 * oof_ridge + 0.3 * oof_knn
    test_blend = 0.7 * test_ridge + 0.3 * test_knn

    cv, angle_std = compute_cv_and_std(oof_blend, train_targets, test_blend)
    return cv, angle_std, oof_blend, test_blend


# ============================================================
# STRATEGY 9: Target-Specific Optimal Config
# Use best config for each target independently
# ============================================================
def strategy_target_specific(data):
    """Different optimal settings per target."""
    print("\n  Strategy 9: Target-Specific Optimal Config")

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    # Best configs from per_target_best.py
    target_configs = {
        0: {"frames": list(range(145, 165)), "alpha": 100, "n_comp": 15},  # angle
        1: {"frames": list(range(145, 165)), "alpha": 10, "n_comp": 15},   # depth
        2: {"frames": list(range(130, 180)), "alpha": 10, "n_comp": 15},   # left_right
    }

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for t_idx, config in target_configs.items():
        frames = config["frames"]
        alpha = config["alpha"]
        n_comp = config["n_comp"]

        def flatten(series):
            flat = series[:, frames, :].reshape(len(series), -1)
            return np.nan_to_num(flat, nan=0.0)

        train_flat = flatten(train_series)
        test_flat = flatten(test_series)

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_flat[train_mask]
            y_train = train_targets[train_mask, t_idx]
            X_test = test_flat[test_mask]

            player_indices = np.where(train_mask)[0]

            n_c = min(n_comp, len(X_train) - 1)
            pca = PCA(n_components=n_c)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_pca)
            X_test_scaled = scaler.transform(X_test_pca)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx]

                model = Ridge(alpha=alpha)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    cv, angle_std = compute_cv_and_std(oof_preds, train_targets, predictions)
    return cv, angle_std, oof_preds, predictions


# ============================================================
# STRATEGY 10: Bootstrap Ensemble
# Multiple models on bootstrap samples
# ============================================================
def strategy_bootstrap_ensemble(data, n_bootstraps=5):
    """Ensemble of models trained on bootstrap samples."""
    print("\n  Strategy 10: Bootstrap Ensemble")

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    unique_pids = sorted(np.unique(train_pids))

    all_test_preds = []

    for bootstrap_idx in range(n_bootstraps):
        np.random.seed(bootstrap_idx)

        predictions = np.zeros((len(test_series), 3))

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_flat[train_mask]
            y_train = train_targets[train_mask]
            X_test = test_flat[test_mask]

            # Bootstrap sample within player
            n_samples = len(X_train)
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train[boot_idx]
            y_boot = y_train[boot_idx]

            n_comp = min(15, len(X_boot) - 1)
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_boot)
            X_test_pca = pca.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_pca)
            X_test_scaled = scaler.transform(X_test_pca)

            for t_idx in range(3):
                model = Ridge(alpha=100)
                model.fit(X_train_scaled, y_boot[:, t_idx])
                predictions[test_mask, t_idx] = model.predict(X_test_scaled)

        all_test_preds.append(predictions)

    test_blend = np.mean(all_test_preds, axis=0)

    # For OOF, use standard CV
    oof_preds, _ = base_per_player_model(data, frames, alpha=100)

    cv, angle_std = compute_cv_and_std(oof_preds, train_targets, test_blend)
    return cv, angle_std, oof_preds, test_blend


# ============================================================
# STRATEGY 11: Confidence-Weighted Prediction
# Weight predictions by model confidence (inverse variance)
# ============================================================
def strategy_confidence_weighted(data):
    """Weight predictions by cross-validation variance."""
    print("\n  Strategy 11: Confidence-Weighted Prediction")

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

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
            fold_val_errors = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr, y_val = y_train[train_idx, t_idx], y_train[val_idx, t_idx]

                model = Ridge(alpha=100)
                model.fit(X_tr, y_tr)

                val_pred = model.predict(X_val)
                oof_preds[player_indices[val_idx], t_idx] = val_pred

                fold_test_preds.append(model.predict(X_test_scaled))
                fold_val_errors.append(np.mean((val_pred - y_val)**2))

            # Weight by inverse error
            weights = 1.0 / (np.array(fold_val_errors) + 1e-6)
            weights = weights / weights.sum()

            weighted_pred = np.zeros(len(X_test))
            for w, pred in zip(weights, fold_test_preds):
                weighted_pred += w * pred

            predictions[test_mask, t_idx] = weighted_pred

    cv, angle_std = compute_cv_and_std(oof_preds, train_targets, predictions)
    return cv, angle_std, oof_preds, predictions


# ============================================================
# STRATEGY 12: Global + Per-Player Blend
# Combine global model with per-player models
# ============================================================
def strategy_global_local_blend(data):
    """Blend global model with per-player models."""
    print("\n  Strategy 12: Global + Per-Player Blend")

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    # Per-player predictions
    oof_local, test_local = base_per_player_model(data, frames, alpha=100)

    # Global model
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(train_flat)
    X_test_pca = pca.transform(test_flat)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    oof_global = np.zeros_like(train_targets)
    test_global = np.zeros((len(test_series), 3))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for t_idx in range(3):
        fold_test_preds = []

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = train_targets[train_idx, t_idx]

            model = Ridge(alpha=100)
            model.fit(X_tr, y_tr)

            oof_global[val_idx, t_idx] = model.predict(X_val)
            fold_test_preds.append(model.predict(X_test_scaled))

        test_global[:, t_idx] = np.mean(fold_test_preds, axis=0)

    # Blend: mostly local, small global contribution
    oof_blend = 0.9 * oof_local + 0.1 * oof_global
    test_blend = 0.9 * test_local + 0.1 * test_global

    cv, angle_std = compute_cv_and_std(oof_blend, train_targets, test_blend)
    return cv, angle_std, oof_blend, test_blend


def main():
    print("=" * 80)
    print("COMBINATION STRATEGIES - COMPREHENSIVE TEST")
    print("=" * 80)

    data = load_data()

    strategies = [
        ("1. Regularization Ladder", strategy_regularization_ladder),
        ("2. Frame Window Ensemble", strategy_frame_window_ensemble),
        ("3. PCA Component Ladder", strategy_pca_ladder),
        ("4. Hierarchical Target", strategy_hierarchical),
        ("5. Body Part Ensemble", strategy_body_part_ensemble),
        ("6. Residual Boosting", strategy_residual_boosting),
        ("7. Player Mean Shrinkage", strategy_player_mean_shrinkage),
        ("8. Model Type Ensemble", strategy_model_type_ensemble),
        ("9. Target-Specific Config", strategy_target_specific),
        ("10. Bootstrap Ensemble", strategy_bootstrap_ensemble),
        ("11. Confidence-Weighted", strategy_confidence_weighted),
        ("12. Global + Local Blend", strategy_global_local_blend),
    ]

    results = []

    print("\n" + "=" * 60)
    print("TESTING ALL STRATEGIES")
    print("=" * 60)

    for name, func in strategies:
        try:
            cv, angle_std, oof, test = func(data)
            results.append({
                "strategy": name,
                "cv": cv,
                "angle_std": angle_std,
                "oof": oof,
                "test": test
            })
            print(f"  {name}: CV={cv:.6f}, angle_std={angle_std:.4f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY - SORTED BY CV")
    print("=" * 80)

    print(f"\n{'Strategy':<35} {'CV':>10} {'angle_std':>12} {'Pred LB':>10}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x["cv"]):
        pred_lb = 0.0038 + 0.034 * r["angle_std"]
        print(f"{r['strategy']:<35} {r['cv']:>10.6f} {r['angle_std']:>12.4f} {pred_lb:>10.6f}")

    # Find best by predicted LB
    print("\n" + "=" * 80)
    print("SUMMARY - SORTED BY PREDICTED LB")
    print("=" * 80)

    for r in results:
        r["pred_lb"] = 0.0038 + 0.034 * r["angle_std"]

    print(f"\n{'Strategy':<35} {'CV':>10} {'angle_std':>12} {'Pred LB':>10}")
    print("-" * 70)

    for r in sorted(results, key=lambda x: x["pred_lb"]):
        print(f"{r['strategy']:<35} {r['cv']:>10.6f} {r['angle_std']:>12.4f} {r['pred_lb']:>10.6f}")

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    df_results = pd.DataFrame([
        {"strategy": r["strategy"], "cv": r["cv"], "angle_std": r["angle_std"], "pred_lb": r["pred_lb"]}
        for r in results
    ])
    df_results.to_csv(OUTPUT_DIR / "combination_strategies.csv", index=False)
    print(f"\nResults saved to output/combination_strategies.csv")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Best by CV
    best_cv = min(results, key=lambda x: x["cv"])
    print(f"\nBest by CV: {best_cv['strategy']}")
    print(f"  CV: {best_cv['cv']:.6f}, angle_std: {best_cv['angle_std']:.4f}")

    # Best by predicted LB
    best_lb = min(results, key=lambda x: x["pred_lb"])
    print(f"\nBest by Predicted LB: {best_lb['strategy']}")
    print(f"  CV: {best_lb['cv']:.6f}, angle_std: {best_lb['angle_std']:.4f}, Pred LB: {best_lb['pred_lb']:.6f}")

    # Balanced recommendation
    # Score = 0.5 * normalized_cv + 0.5 * normalized_pred_lb
    cv_values = [r["cv"] for r in results]
    lb_values = [r["pred_lb"] for r in results]

    cv_min, cv_max = min(cv_values), max(cv_values)
    lb_min, lb_max = min(lb_values), max(lb_values)

    for r in results:
        r["cv_norm"] = (r["cv"] - cv_min) / (cv_max - cv_min + 1e-9)
        r["lb_norm"] = (r["pred_lb"] - lb_min) / (lb_max - lb_min + 1e-9)
        r["balanced"] = 0.5 * r["cv_norm"] + 0.5 * r["lb_norm"]

    best_balanced = min(results, key=lambda x: x["balanced"])
    print(f"\nBest Balanced (CV + Pred LB): {best_balanced['strategy']}")
    print(f"  CV: {best_balanced['cv']:.6f}, angle_std: {best_balanced['angle_std']:.4f}, Pred LB: {best_balanced['pred_lb']:.6f}")


if __name__ == "__main__":
    main()
