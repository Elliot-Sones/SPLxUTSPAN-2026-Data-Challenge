"""
Fundamentally Different Approaches - Testing paradigm shifts, not incremental improvements.

Approaches:
1. Raw Sequence Matching - KNN on raw trajectories
2. Prototype Matching - Distance to player's mean shot
3. Shot Space Interpolation - Interpolate in manifold space
4. Ensemble of Fundamentals - Combine above approaches
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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


def load_raw_data():
    """Load raw time series data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def extract_raw_series(df):
        all_series = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])
            all_series.append(ts)
        return np.array(all_series)

    print("Loading training data...")
    train_series = extract_raw_series(train_df)
    train_targets = train_df[["angle", "depth", "left_right"]].values
    train_pids = train_df["participant_id"].values
    train_ids = train_df["id"].values

    print("Loading test data...")
    test_series = extract_raw_series(test_df)
    test_pids = test_df["participant_id"].values
    test_ids = test_df["id"].values

    return {
        "train_series": train_series,  # (N, 240, n_keypoints)
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_series": test_series,
        "test_pids": test_pids,
        "test_ids": test_ids,
        "keypoint_cols": keypoint_cols,
    }


def compute_trajectory_distance(ts1, ts2, frames=None):
    """Compute distance between two trajectories using key frames."""
    if frames is None:
        frames = list(range(130, 180))  # Release window

    # Use only valid (non-nan) values
    t1 = ts1[frames].flatten()
    t2 = ts2[frames].flatten()

    valid = ~(np.isnan(t1) | np.isnan(t2))
    if valid.sum() < 10:
        return np.inf

    return np.sqrt(np.mean((t1[valid] - t2[valid])**2))


def approach_1_knn_raw(data, k=3):
    """
    Approach 1: K-Nearest Neighbors on raw trajectories

    Idea: Find the k most similar training shots (by trajectory) and average their targets.
    This is fundamentally different because it uses NO feature engineering.
    """
    print("\n" + "="*60)
    print("APPROACH 1: KNN on Raw Trajectories")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Key frames around release
    key_frames = list(range(140, 170))

    # Flatten trajectories for distance computation
    def flatten_key_frames(series, frames):
        flat = series[:, frames, :].reshape(len(series), -1)
        # Handle NaN
        flat = np.nan_to_num(flat, nan=0.0)
        return flat

    train_flat = flatten_key_frames(train_series, key_frames)
    test_flat = flatten_key_frames(test_series, key_frames)

    # Per-player KNN
    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        # Normalize within player
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # KNN for each target
        for t_idx in range(3):
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train_scaled, y_train[:, t_idx])
            predictions[test_mask, t_idx] = knn.predict(X_test_scaled)

        # Cross-validation for this player
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx]

            for t_idx in range(3):
                knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                knn.fit(X_tr, y_tr[:, t_idx])
                oof_preds[player_indices[val_idx], t_idx] = knn.predict(X_val)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "knn_raw"


def approach_2_prototype_distance(data):
    """
    Approach 2: Prototype-based prediction

    Idea: For each player, compute the "prototype" shot (mean trajectory).
    Then predict based on how each shot differs from the prototype.
    """
    print("\n" + "="*60)
    print("APPROACH 2: Prototype Distance")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    key_frames = list(range(140, 170))

    def flatten_key_frames(series, frames):
        flat = series[:, frames, :].reshape(len(series), -1)
        flat = np.nan_to_num(flat, nan=0.0)
        return flat

    train_flat = flatten_key_frames(train_series, key_frames)
    test_flat = flatten_key_frames(test_series, key_frames)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        # Compute prototype (mean shot)
        prototype = X_train.mean(axis=0)

        # Compute distance from prototype for each shot
        train_dist = np.linalg.norm(X_train - prototype, axis=1, keepdims=True)
        test_dist = np.linalg.norm(X_test - prototype, axis=1, keepdims=True)

        # Also compute direction (PCA components)
        pca = PCA(n_components=min(10, X_train.shape[0]-1))
        train_pca = pca.fit_transform(X_train)
        test_pca = pca.transform(X_test)

        # Combine distance and direction
        X_train_feat = np.hstack([train_dist, train_pca])
        X_test_feat = np.hstack([test_dist, test_pca])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_test_scaled = scaler.transform(X_test_feat)

        # Simple linear model
        from sklearn.linear_model import Ridge

        for t_idx in range(3):
            model = Ridge(alpha=10.0)
            model.fit(X_train_scaled, y_train[:, t_idx])
            predictions[test_mask, t_idx] = model.predict(X_test_scaled)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx]

            for t_idx in range(3):
                model = Ridge(alpha=10.0)
                model.fit(X_tr, y_tr[:, t_idx])
                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "prototype"


def approach_3_manifold_interpolation(data):
    """
    Approach 3: Manifold Interpolation

    Idea: Embed shots in a low-dimensional manifold using PCA/t-SNE.
    Then interpolate targets based on position in manifold.
    """
    print("\n" + "="*60)
    print("APPROACH 3: Manifold Interpolation")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    key_frames = list(range(140, 170))

    def flatten_key_frames(series, frames):
        flat = series[:, frames, :].reshape(len(series), -1)
        flat = np.nan_to_num(flat, nan=0.0)
        return flat

    train_flat = flatten_key_frames(train_series, key_frames)
    test_flat = flatten_key_frames(test_series, key_frames)

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
        n_components = min(20, n_samples - 1)

        # PCA for manifold embedding
        pca = PCA(n_components=n_components)
        train_embedded = pca.fit_transform(X_train)
        test_embedded = pca.transform(X_test)

        # Weighted KNN in manifold space
        from sklearn.neighbors import KNeighborsRegressor

        for t_idx in range(3):
            knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
            knn.fit(train_embedded, y_train[:, t_idx])
            predictions[test_mask, t_idx] = knn.predict(test_embedded)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(train_embedded):
            X_tr, X_val = train_embedded[train_idx], train_embedded[val_idx]
            y_tr = y_train[train_idx]

            for t_idx in range(3):
                knn = KNeighborsRegressor(n_neighbors=min(5, len(train_idx)-1), weights='distance')
                knn.fit(X_tr, y_tr[:, t_idx])
                oof_preds[player_indices[val_idx], t_idx] = knn.predict(X_val)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "manifold"


def approach_4_temporal_attention(data):
    """
    Approach 4: Learn which frames matter most

    Idea: Instead of using fixed frames, learn per-player which frames
    have the strongest correlation with targets, then use only those.
    """
    print("\n" + "="*60)
    print("APPROACH 4: Temporal Attention (Frame Selection)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]
    keypoint_cols = data["keypoint_cols"]

    ranges = get_target_ranges()

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        player_series = train_series[train_mask]
        player_targets = train_targets[train_mask]
        player_test = test_series[test_mask]

        n_samples = player_series.shape[0]

        # For each target, find the most correlated frame-keypoint combinations
        best_features_per_target = {}

        for t_idx, target in enumerate(TARGETS):
            y = player_targets[:, t_idx]

            correlations = []
            for frame in range(100, 200):  # Focus on release region
                for kp_idx in range(player_series.shape[2]):
                    values = player_series[:, frame, kp_idx]
                    if np.std(values) > 0.01 and not np.any(np.isnan(values)):
                        corr, _ = pearsonr(values, y)
                        if not np.isnan(corr):
                            correlations.append((abs(corr), frame, kp_idx, corr))

            # Select top 20 features
            correlations.sort(reverse=True)
            best_features_per_target[t_idx] = correlations[:20]

        # Build feature matrix using selected features
        def build_features(series, feature_list):
            features = []
            for _, frame, kp_idx, _ in feature_list:
                features.append(series[:, frame, kp_idx])
            return np.array(features).T

        from sklearn.linear_model import Ridge

        for t_idx in range(3):
            if not best_features_per_target[t_idx]:
                # Fallback to mean
                predictions[test_mask, t_idx] = player_targets[:, t_idx].mean()
                continue

            X_train = build_features(player_series, best_features_per_target[t_idx])
            X_test = build_features(player_test, best_features_per_target[t_idx])
            y_train = player_targets[:, t_idx]

            # Handle NaN
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=10.0)
            model.fit(X_train_scaled, y_train)
            predictions[test_mask, t_idx] = model.predict(X_test_scaled)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for fold_train_idx, fold_val_idx in kf.split(player_series):
            # Re-select features on fold training data
            fold_series = player_series[fold_train_idx]
            fold_targets = player_targets[fold_train_idx]
            val_series = player_series[fold_val_idx]

            for t_idx in range(3):
                y = fold_targets[:, t_idx]

                correlations = []
                for frame in range(100, 200):
                    for kp_idx in range(fold_series.shape[2]):
                        values = fold_series[:, frame, kp_idx]
                        if np.std(values) > 0.01 and not np.any(np.isnan(values)):
                            corr, _ = pearsonr(values, y)
                            if not np.isnan(corr):
                                correlations.append((abs(corr), frame, kp_idx, corr))

                correlations.sort(reverse=True)
                top_features = correlations[:20]

                if not top_features:
                    oof_preds[player_indices[fold_val_idx], t_idx] = y.mean()
                    continue

                X_tr = build_features(fold_series, top_features)
                X_val = build_features(val_series, top_features)

                X_tr = np.nan_to_num(X_tr, nan=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0)

                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)

                model = Ridge(alpha=10.0)
                model.fit(X_tr_scaled, y)
                oof_preds[player_indices[fold_val_idx], t_idx] = model.predict(X_val_scaled)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "temporal_attention"


def approach_5_target_mean_baseline(data):
    """
    Approach 5: Per-player target mean baseline

    The simplest possible model - just predict the player's mean target.
    This is the floor that any model should beat.
    """
    print("\n" + "="*60)
    print("APPROACH 5: Per-Player Mean Baseline")
    print("="*60)

    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    predictions = np.zeros((len(test_pids), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        player_targets = train_targets[train_mask]
        player_mean = player_targets.mean(axis=0)

        predictions[test_mask] = player_mean
        oof_preds[train_mask] = player_mean

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "mean_baseline"


def approach_6_gaussian_process(data):
    """
    Approach 6: Gaussian Process Regression

    Idea: Model uncertainty explicitly. GP can capture non-linear relationships
    and provide uncertainty estimates.
    """
    print("\n" + "="*60)
    print("APPROACH 6: Gaussian Process Regression")
    print("="*60)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    key_frames = list(range(140, 170))

    def flatten_key_frames(series, frames):
        flat = series[:, frames, :].reshape(len(series), -1)
        flat = np.nan_to_num(flat, nan=0.0)
        return flat

    train_flat = flatten_key_frames(train_series, key_frames)
    test_flat = flatten_key_frames(test_series, key_frames)

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        # Reduce dimensionality for GP (expensive)
        pca = PCA(n_components=min(10, X_train.shape[0]-1))
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)

        for t_idx in range(3):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
            gp.fit(X_train_scaled, y_train[:, t_idx])
            predictions[test_mask, t_idx] = gp.predict(X_test_scaled)

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(train_mask)[0]

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx]

            for t_idx in range(3):
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42)
                gp.fit(X_tr, y_tr[:, t_idx])
                oof_preds[player_indices[val_idx], t_idx] = gp.predict(X_val)

    # Compute CV score
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "gaussian_process"


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
    print("FUNDAMENTALLY DIFFERENT APPROACHES")
    print("Testing paradigm shifts, not incremental improvements")
    print("="*80)

    data = load_raw_data()

    results = []
    all_predictions = {}

    # Test all approaches
    approaches = [
        (approach_5_target_mean_baseline, {}),  # Baseline first
        (approach_1_knn_raw, {"k": 3}),
        (approach_1_knn_raw, {"k": 5}),
        (approach_1_knn_raw, {"k": 7}),
        (approach_2_prototype_distance, {}),
        (approach_3_manifold_interpolation, {}),
        (approach_4_temporal_attention, {}),
        (approach_6_gaussian_process, {}),
    ]

    for approach_func, kwargs in approaches:
        try:
            preds, cv, name = approach_func(data, **kwargs) if kwargs else approach_func(data)
            results.append({
                "approach": name + (f"_k{kwargs.get('k', '')}" if 'k' in kwargs else ""),
                "cv_score": cv,
            })
            all_predictions[name + (f"_k{kwargs.get('k', '')}" if 'k' in kwargs else "")] = preds
        except Exception as e:
            print(f"Error in {approach_func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF FUNDAMENTAL APPROACHES")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("cv_score")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(OUTPUT_DIR / "fundamental_approaches_results.csv", index=False)

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

        # Also try ensemble of all approaches
        print("\n" + "="*60)
        print("ENSEMBLE OF ALL APPROACHES")
        print("="*60)

        # Average all predictions
        ensemble_preds = np.mean(list(all_predictions.values()), axis=0)

        # Evaluate ensemble (use oof from best as proxy)
        print("Ensemble created by averaging all approaches")

        filepath2, sub_num2 = create_submission(
            data["test_ids"],
            ensemble_preds,
            0.0,  # Unknown CV for ensemble
            "ensemble_fundamental"
        )

    return results_df


if __name__ == "__main__":
    main()
