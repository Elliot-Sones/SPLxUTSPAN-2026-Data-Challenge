"""
Radical Approaches - More aggressive paradigm shifts

1. 1D CNN on raw sequences
2. Dynamic Time Warping similarity
3. Target-space optimization
4. Constrained player-specific models
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
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
    """Load data with minimal processing."""
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


def dtw_distance(s1, s2, window=10):
    """
    Simplified DTW with Sakoe-Chiba band constraint.
    Only computes on a subset of dimensions for speed.
    """
    n, m = len(s1), len(s2)

    # Use infinity matrix with constraint
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            d = np.sqrt(np.nansum((s1[i-1] - s2[j-1])**2))
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

    return cost[n, m]


def approach_dtw_knn(data, k=3, n_dims=20):
    """
    Approach: KNN with Dynamic Time Warping distance

    DTW handles temporal variations in shot timing better than Euclidean distance.
    """
    print("\n" + "="*60)
    print(f"APPROACH: DTW-KNN (k={k}, dims={n_dims})")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Use only release window and reduce dimensions
    frames = list(range(130, 180))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        print(f"  Player {pid}...")
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        player_train = train_series[train_mask][:, frames, :n_dims]
        player_test = test_series[test_mask][:, frames, :n_dims]
        player_targets = train_targets[train_mask]

        n_train = len(player_train)
        n_test = len(player_test)

        # Compute DTW distance matrix for test
        test_distances = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                test_distances[i, j] = dtw_distance(player_test[i], player_train[j], window=5)

        # KNN prediction
        for i in range(n_test):
            nearest = np.argsort(test_distances[i])[:k]
            weights = 1.0 / (test_distances[i, nearest] + 1e-6)
            weights /= weights.sum()
            predictions[test_mask][i] = np.average(player_targets[nearest], axis=0, weights=weights)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(player_train):
            tr_series = player_train[train_idx]
            val_series = player_train[val_idx]
            tr_targets = player_targets[train_idx]

            for vi, val_shot in enumerate(val_series):
                distances = np.array([dtw_distance(val_shot, tr_shot, window=5) for tr_shot in tr_series])
                nearest = np.argsort(distances)[:k]
                weights = 1.0 / (distances[nearest] + 1e-6)
                weights /= weights.sum()
                oof_preds[player_indices[val_idx[vi]]] = np.average(tr_targets[nearest], axis=0, weights=weights)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, f"dtw_k{k}"


def approach_target_space_interpolation(data):
    """
    Approach: Target-Space Interpolation

    Idea: Map shots to target space, then interpolate.
    If shot A is between shots B and C in feature space,
    its target should be between B's and C's targets.
    """
    print("\n" + "="*60)
    print("APPROACH: Target-Space Interpolation")
    print("="*60)

    from sklearn.decomposition import PCA

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

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        n_samples = X_train.shape[0]

        # PCA to reduce dimensions
        n_components = min(15, n_samples - 1)
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # For each test point, find its position as a convex combination of training points
        # This is like a soft KNN where weights are based on distance in PCA space

        from scipy.spatial.distance import cdist

        for i, x_test in enumerate(X_test_pca):
            distances = cdist([x_test], X_train_pca)[0]

            # Soft weights (exponential decay)
            sigma = np.median(distances)
            weights = np.exp(-distances / (2 * sigma))
            weights /= weights.sum()

            predictions[test_mask][i] = np.average(y_train, axis=0, weights=weights)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(X_train_pca):
            X_tr = X_train_pca[train_idx]
            X_val = X_train_pca[val_idx]
            y_tr = y_train[train_idx]

            for vi, x_val in enumerate(X_val):
                distances = cdist([x_val], X_tr)[0]
                sigma = np.median(distances) + 1e-6
                weights = np.exp(-distances / (2 * sigma))
                weights /= weights.sum()
                oof_preds[player_indices[val_idx[vi]]] = np.average(y_tr, axis=0, weights=weights)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "target_space"


def approach_constrained_ridge(data, constraint_alpha=0.5):
    """
    Approach: Constrained Ridge Regression

    Add constraints that predictions should be "reasonable" for each player.
    - Predictions should be within observed range
    - Predictions should maintain player-specific correlations
    """
    print("\n" + "="*60)
    print("APPROACH: Constrained Ridge")
    print("="*60)

    from sklearn.decomposition import PCA

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

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        # Get player's target statistics
        player_means = y_train.mean(axis=0)
        player_stds = y_train.std(axis=0)
        player_mins = y_train.min(axis=0)
        player_maxs = y_train.max(axis=0)

        # PCA
        n_components = min(20, X_train.shape[0] - 1)
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        for t_idx in range(3):
            # Ridge regression
            model = Ridge(alpha=100.0)
            model.fit(X_train_scaled, y_train[:, t_idx])
            raw_preds = model.predict(X_test_scaled)

            # Constraint: clip to observed range with some margin
            margin = 0.5 * player_stds[t_idx]
            constrained_preds = np.clip(raw_preds,
                                        player_mins[t_idx] - margin,
                                        player_maxs[t_idx] + margin)

            # Blend raw and constrained
            predictions[test_mask, t_idx] = (
                constraint_alpha * constrained_preds +
                (1 - constraint_alpha) * raw_preds
            )

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx]

            fold_means = y_tr.mean(axis=0)
            fold_stds = y_tr.std(axis=0)
            fold_mins = y_tr.min(axis=0)
            fold_maxs = y_tr.max(axis=0)

            for t_idx in range(3):
                model = Ridge(alpha=100.0)
                model.fit(X_tr, y_tr[:, t_idx])
                raw_preds = model.predict(X_val)

                margin = 0.5 * fold_stds[t_idx]
                constrained_preds = np.clip(raw_preds,
                                            fold_mins[t_idx] - margin,
                                            fold_maxs[t_idx] + margin)

                oof_preds[player_indices[val_idx], t_idx] = (
                    constraint_alpha * constrained_preds +
                    (1 - constraint_alpha) * raw_preds
                )

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "constrained_ridge"


def approach_weighted_ensemble(data):
    """
    Approach: Learn optimal weights for combining multiple simple predictors.

    Instead of using a single model, combine:
    1. Player mean
    2. KNN
    3. Ridge
    4. Distance-weighted interpolation

    Learn weights that minimize CV error.
    """
    print("\n" + "="*60)
    print("APPROACH: Weighted Ensemble")
    print("="*60)

    from sklearn.decomposition import PCA

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

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        n_samples = X_train.shape[0]

        # PCA
        n_components = min(15, n_samples - 1)
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        # Multiple base predictions
        def get_base_predictions(X_tr, y_tr, X_val, scaler_fit=None):
            preds = []

            # 1. Mean
            mean_pred = np.tile(y_tr.mean(axis=0), (len(X_val), 1))
            preds.append(mean_pred)

            # 2. KNN k=3
            from sklearn.neighbors import KNeighborsRegressor
            knn3_pred = np.zeros((len(X_val), 3))
            for t_idx in range(3):
                knn = KNeighborsRegressor(n_neighbors=min(3, len(X_tr)-1), weights='distance')
                knn.fit(X_tr, y_tr[:, t_idx])
                knn3_pred[:, t_idx] = knn.predict(X_val)
            preds.append(knn3_pred)

            # 3. KNN k=5
            knn5_pred = np.zeros((len(X_val), 3))
            for t_idx in range(3):
                knn = KNeighborsRegressor(n_neighbors=min(5, len(X_tr)-1), weights='distance')
                knn.fit(X_tr, y_tr[:, t_idx])
                knn5_pred[:, t_idx] = knn.predict(X_val)
            preds.append(knn5_pred)

            # 4. Ridge
            ridge_pred = np.zeros((len(X_val), 3))
            for t_idx in range(3):
                ridge = Ridge(alpha=100.0)
                ridge.fit(X_tr, y_tr[:, t_idx])
                ridge_pred[:, t_idx] = ridge.predict(X_val)
            preds.append(ridge_pred)

            return np.array(preds)  # (n_models, n_val, 3)

        # Cross-validation to find best weights
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        all_oof_base = []
        all_oof_true = []

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx]

            base_preds = get_base_predictions(X_tr, y_tr, X_val)
            all_oof_base.append(base_preds)
            all_oof_true.append(y_train[val_idx])

        # Stack all OOF predictions
        all_oof_base = np.concatenate(all_oof_base, axis=1)  # (n_models, n_train, 3)
        all_oof_true = np.concatenate(all_oof_true, axis=0)  # (n_train, 3)

        # Find optimal weights per target
        optimal_weights = np.zeros((3, all_oof_base.shape[0]))

        for t_idx in range(3):
            from scipy.optimize import minimize

            def loss(weights):
                weights = np.abs(weights)
                weights = weights / weights.sum()
                ensemble = np.average(all_oof_base[:, :, t_idx], axis=0, weights=weights)
                return np.mean((ensemble - all_oof_true[:, t_idx])**2)

            n_models = all_oof_base.shape[0]
            init = np.ones(n_models) / n_models
            result = minimize(loss, init, method='Nelder-Mead')
            optimal_weights[t_idx] = np.abs(result.x) / np.abs(result.x).sum()

        # Generate final predictions
        test_base = get_base_predictions(X_train_scaled, y_train, X_test_scaled)

        for t_idx in range(3):
            predictions[test_mask, t_idx] = np.average(
                test_base[:, :, t_idx],
                axis=0,
                weights=optimal_weights[t_idx]
            )

        # Regenerate OOF with optimal weights
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx]

            base_preds = get_base_predictions(X_tr, y_tr, X_val)

            for t_idx in range(3):
                oof_preds[player_indices[val_idx], t_idx] = np.average(
                    base_preds[:, :, t_idx],
                    axis=0,
                    weights=optimal_weights[t_idx]
                )

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "weighted_ensemble"


def approach_residual_correction(data):
    """
    Approach: Residual Correction

    Train model to predict targets.
    Then train a second model to predict the residuals.
    This can capture patterns the first model misses.
    """
    print("\n" + "="*60)
    print("APPROACH: Residual Correction")
    print("="*60)

    from sklearn.decomposition import PCA

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Use two different frame windows
    frames1 = list(range(140, 170))
    frames2 = list(range(120, 180))

    def flatten(series, frames):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat1 = flatten(train_series, frames1)
    test_flat1 = flatten(test_series, frames1)

    train_flat2 = flatten(train_series, frames2)
    test_flat2 = flatten(test_series, frames2)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train1 = train_flat1[train_mask]
        X_train2 = train_flat2[train_mask]
        y_train = train_targets[train_mask]
        X_test1 = test_flat1[test_mask]
        X_test2 = test_flat2[test_mask]

        n_samples = X_train1.shape[0]

        # PCA for first model
        n_comp1 = min(15, n_samples - 1)
        pca1 = PCA(n_components=n_comp1)
        X_train1_pca = pca1.fit_transform(X_train1)
        X_test1_pca = pca1.transform(X_test1)

        scaler1 = StandardScaler()
        X_train1_scaled = scaler1.fit_transform(X_train1_pca)
        X_test1_scaled = scaler1.transform(X_test1_pca)

        # PCA for residual model (different features)
        n_comp2 = min(20, n_samples - 1)
        pca2 = PCA(n_components=n_comp2)
        X_train2_pca = pca2.fit_transform(X_train2)
        X_test2_pca = pca2.transform(X_test2)

        scaler2 = StandardScaler()
        X_train2_scaled = scaler2.fit_transform(X_train2_pca)
        X_test2_scaled = scaler2.transform(X_test2_pca)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(X_train1_scaled):
            X_tr1, X_val1 = X_train1_scaled[train_idx], X_train1_scaled[val_idx]
            X_tr2, X_val2 = X_train2_scaled[train_idx], X_train2_scaled[val_idx]
            y_tr = y_train[train_idx]

            for t_idx in range(3):
                # First model
                model1 = Ridge(alpha=100.0)
                model1.fit(X_tr1, y_tr[:, t_idx])
                pred1 = model1.predict(X_tr1)
                val_pred1 = model1.predict(X_val1)

                # Residuals
                residuals = y_tr[:, t_idx] - pred1

                # Second model on residuals
                model2 = Ridge(alpha=500.0)
                model2.fit(X_tr2, residuals)
                val_pred2 = model2.predict(X_val2)

                # Combined prediction
                oof_preds[player_indices[val_idx], t_idx] = val_pred1 + 0.5 * val_pred2

        # Final predictions
        for t_idx in range(3):
            model1 = Ridge(alpha=100.0)
            model1.fit(X_train1_scaled, y_train[:, t_idx])
            pred1 = model1.predict(X_train1_scaled)
            test_pred1 = model1.predict(X_test1_scaled)

            residuals = y_train[:, t_idx] - pred1

            model2 = Ridge(alpha=500.0)
            model2.fit(X_train2_scaled, residuals)
            test_pred2 = model2.predict(X_test2_scaled)

            predictions[test_mask, t_idx] = test_pred1 + 0.5 * test_pred2

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "residual_correction"


def create_submission(test_ids, predictions, cv_score, approach_name):
    """Create submission CSV."""
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

    # Scale predictions
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

    print(f"\nSubmission {next_num} created: {filepath}")
    print(f"  Approach: {approach_name}")
    print(f"  CV Score: {cv_score:.6f}")

    return filepath, next_num


def main():
    print("="*80)
    print("RADICAL APPROACHES")
    print("="*80)

    data = load_data()

    results = []
    all_predictions = {}

    # Test approaches
    approaches = [
        (approach_target_space_interpolation, {}),
        (approach_constrained_ridge, {"constraint_alpha": 0.5}),
        (approach_weighted_ensemble, {}),
        (approach_residual_correction, {}),
    ]

    for approach_func, kwargs in approaches:
        try:
            preds, cv, name = approach_func(data, **kwargs) if kwargs else approach_func(data)
            results.append({"approach": name, "cv_score": cv})
            all_predictions[name] = preds
        except Exception as e:
            print(f"Error in {approach_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    # DTW is slow, test with fewer dimensions
    try:
        preds, cv, name = approach_dtw_knn(data, k=3, n_dims=10)
        results.append({"approach": name, "cv_score": cv})
        all_predictions[name] = preds
    except Exception as e:
        print(f"Error in DTW: {e}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF RADICAL APPROACHES")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("cv_score")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(OUTPUT_DIR / "radical_approaches_results.csv", index=False)

    # Create submission with best approach
    if results:
        best = results_df.iloc[0]
        best_name = best["approach"]
        best_preds = all_predictions[best_name]

        print(f"\nCreating submission with best approach: {best_name}")
        filepath, sub_num = create_submission(
            data["test_ids"],
            best_preds,
            best["cv_score"],
            best_name
        )

    return results_df


if __name__ == "__main__":
    main()
