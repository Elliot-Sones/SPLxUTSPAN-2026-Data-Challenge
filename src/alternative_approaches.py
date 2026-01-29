"""
Alternative Approaches: Fundamentally different modeling strategies.

Previous approaches all used similar feature engineering + tree/linear models.
This script tries completely different approaches:

1. K-Nearest Neighbors (instance-based, no parametric model)
2. Critical Frame Only (use only the most important frames)
3. Gaussian Process Regression (Bayesian, good for small data)
4. Simple Mean/Median baseline (sanity check)
5. Per-player mean (simplest personalized model)
6. Weighted KNN with learned weights
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

# Critical frames discovered in research
CRITICAL_FRAMES = {
    "angle": [153, 150, 155, 160, 145],  # Around frame 153
    "depth": [102, 100, 105, 110, 95],   # Around frame 102
    "left_right": [237, 235, 239, 230, 225],  # Around frame 237
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_raw_data():
    """Load raw time series data."""
    print("Loading raw data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def extract_timeseries(df, is_train=True):
        all_ts = []
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            all_ts.append(ts)
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        return np.array(all_ts), np.array(ids), np.array(pids), np.array(targets) if is_train else None

    train_ts, train_ids, train_pids, train_targets = extract_timeseries(train_df, True)
    test_ts, test_ids, test_pids, _ = extract_timeseries(test_df, False)

    print(f"Train: {train_ts.shape}, Test: {test_ts.shape}")

    return {
        "train_ts": train_ts,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "test_ts": test_ts,
        "test_ids": test_ids,
        "test_pids": test_pids,
        "keypoint_cols": keypoint_cols,
    }


def extract_critical_frame_features(ts_data, target_name):
    """Extract features only from critical frames for a specific target."""
    frames = CRITICAL_FRAMES[target_name]
    n_samples = ts_data.shape[0]
    n_keypoints = ts_data.shape[2]

    # Extract values at critical frames
    features = []
    for frame in frames:
        if frame < 240:
            features.append(ts_data[:, frame, :])

    # Also add velocity at critical frames
    for frame in frames:
        if frame > 0 and frame < 240:
            velocity = ts_data[:, frame, :] - ts_data[:, frame-1, :]
            features.append(velocity)

    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0.0)

    return X


def extract_simple_features(ts_data):
    """Extract very simple features: mean, std, min, max per keypoint."""
    n_samples = ts_data.shape[0]

    features = []

    # Global statistics per keypoint
    features.append(np.nanmean(ts_data, axis=1))  # Mean over time
    features.append(np.nanstd(ts_data, axis=1))   # Std over time
    features.append(np.nanmin(ts_data, axis=1))   # Min over time
    features.append(np.nanmax(ts_data, axis=1))   # Max over time

    # Range
    features.append(np.nanmax(ts_data, axis=1) - np.nanmin(ts_data, axis=1))

    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0.0)

    return X


def approach_knn(data):
    """K-Nearest Neighbors approach."""
    print("\n" + "="*70)
    print("APPROACH 1: K-Nearest Neighbors")
    print("="*70)

    train_ts = data["train_ts"]
    test_ts = data["test_ts"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    # Extract simple features
    X_train = extract_simple_features(train_ts)
    X_test = extract_simple_features(test_ts)

    print(f"Features: {X_train.shape[1]}")

    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train[train_mask]
        y_p_train = train_targets[train_mask]
        X_p_test = X_test[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        n_neighbors = min(5, len(X_p_train) - 1)

        for target_idx, target in enumerate(TARGETS):
            y_target = y_p_train[:, target_idx]

            # Try different k values
            best_k = n_neighbors
            best_score = float('inf')

            for k in range(2, min(10, len(X_p_train))):
                model = KNeighborsRegressor(n_neighbors=k, weights='distance')
                kf = KFold(n_splits=min(5, len(X_p_train)), shuffle=True, random_state=42)
                try:
                    scores = -cross_val_score(model, X_p_train_scaled, y_target,
                                            cv=kf, scoring='neg_mean_squared_error')
                    if scores.mean() < best_score:
                        best_score = scores.mean()
                        best_k = k
                except:
                    pass

            # Train final model
            model = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
            model.fit(X_p_train_scaled, y_target)

            test_indices = np.where(test_mask)[0]
            predictions[test_indices, target_idx] = model.predict(X_p_test_scaled)

            cv_scores.append(best_score)

    print(f"Mean CV MSE: {np.mean(cv_scores):.6f}")

    return predictions, np.mean(cv_scores)


def approach_critical_frames(data):
    """Use only critical frames for each target."""
    print("\n" + "="*70)
    print("APPROACH 2: Critical Frames Only")
    print("="*70)

    train_ts = data["train_ts"]
    test_ts = data["test_ts"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    predictions = np.zeros((len(test_ts), 3))
    cv_scores = []

    for target_idx, target in enumerate(TARGETS):
        print(f"\n  {target}: using frames {CRITICAL_FRAMES[target]}")

        # Extract target-specific features
        X_train = extract_critical_frame_features(train_ts, target)
        X_test = extract_critical_frame_features(test_ts, target)

        print(f"    Features: {X_train.shape[1]}")

        for pid in sorted(np.unique(train_pids)):
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_p_train = X_train[train_mask]
            y_p_train = train_targets[train_mask, target_idx]
            X_p_test = X_test[test_mask]

            # Standardize
            scaler = StandardScaler()
            X_p_train_scaled = scaler.fit_transform(X_p_train)
            X_p_test_scaled = scaler.transform(X_p_test)

            # Use Ridge (simple, less overfitting)
            model = Ridge(alpha=10.0, random_state=42)
            model.fit(X_p_train_scaled, y_p_train)

            test_indices = np.where(test_mask)[0]
            predictions[test_indices, target_idx] = model.predict(X_p_test_scaled)

            # CV score
            kf = KFold(n_splits=min(5, len(X_p_train)), shuffle=True, random_state=42)
            try:
                scores = -cross_val_score(model, X_p_train_scaled, y_p_train,
                                        cv=kf, scoring='neg_mean_squared_error')
                cv_scores.append(scores.mean())
            except:
                pass

    print(f"\nMean CV MSE: {np.mean(cv_scores):.6f}")

    return predictions, np.mean(cv_scores)


def approach_bayesian_ridge(data):
    """Bayesian Ridge Regression - good for small datasets."""
    print("\n" + "="*70)
    print("APPROACH 3: Bayesian Ridge Regression")
    print("="*70)

    train_ts = data["train_ts"]
    test_ts = data["test_ts"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    # Use simple features + PCA
    X_train = extract_simple_features(train_ts)
    X_test = extract_simple_features(test_ts)

    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train[train_mask]
        y_p_train = train_targets[train_mask]
        X_p_test = X_test[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        # PCA to reduce dimensionality
        n_components = min(30, X_p_train_scaled.shape[0] - 1, X_p_train_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_p_train_pca = pca.fit_transform(X_p_train_scaled)
        X_p_test_pca = pca.transform(X_p_test_scaled)

        for target_idx, target in enumerate(TARGETS):
            y_target = y_p_train[:, target_idx]

            model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
            model.fit(X_p_train_pca, y_target)

            test_indices = np.where(test_mask)[0]
            predictions[test_indices, target_idx] = model.predict(X_p_test_pca)

            # CV
            kf = KFold(n_splits=min(5, len(X_p_train)), shuffle=True, random_state=42)
            try:
                scores = -cross_val_score(model, X_p_train_pca, y_target,
                                        cv=kf, scoring='neg_mean_squared_error')
                cv_scores.append(scores.mean())
            except:
                pass

    print(f"Mean CV MSE: {np.mean(cv_scores):.6f}")

    return predictions, np.mean(cv_scores)


def approach_per_player_mean(data):
    """Simplest baseline: predict per-player mean."""
    print("\n" + "="*70)
    print("APPROACH 4: Per-Player Mean (Baseline)")
    print("="*70)

    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    predictions = np.zeros((len(test_pids), 3))

    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        # Just predict the mean for each target
        for target_idx in range(3):
            player_mean = train_targets[train_mask, target_idx].mean()
            test_indices = np.where(test_mask)[0]
            predictions[test_indices, target_idx] = player_mean

    # Calculate "CV" score (leave-one-out style)
    cv_scores = []
    for pid in sorted(np.unique(train_pids)):
        mask = train_pids == pid
        y = train_targets[mask]
        mean_pred = y.mean(axis=0)
        mse = np.mean((y - mean_pred) ** 2)
        cv_scores.append(mse)

    print(f"Mean CV MSE: {np.mean(cv_scores):.6f}")

    return predictions, np.mean(cv_scores)


def approach_distance_weighted(data):
    """Distance-weighted prediction from similar training samples."""
    print("\n" + "="*70)
    print("APPROACH 5: Distance-Weighted Ensemble")
    print("="*70)

    train_ts = data["train_ts"]
    test_ts = data["test_ts"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    # Use simple features
    X_train = extract_simple_features(train_ts)
    X_test = extract_simple_features(test_ts)

    predictions = np.zeros((len(X_test), 3))

    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train[train_mask]
        y_p_train = train_targets[train_mask]
        X_p_test = X_test[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        # Compute distances
        distances = cdist(X_p_test_scaled, X_p_train_scaled, metric='euclidean')

        # Convert to weights (inverse distance)
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted prediction
        test_indices = np.where(test_mask)[0]
        for target_idx in range(3):
            pred = np.sum(weights * y_p_train[:, target_idx], axis=1)
            predictions[test_indices, target_idx] = pred

    print("No CV score for this approach (uses all training data)")

    return predictions, None


def approach_ensemble_diverse(data):
    """Ensemble of diverse simple models."""
    print("\n" + "="*70)
    print("APPROACH 6: Diverse Model Ensemble")
    print("="*70)

    train_ts = data["train_ts"]
    test_ts = data["test_ts"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]

    # Multiple feature sets
    X_train_simple = extract_simple_features(train_ts)
    X_test_simple = extract_simple_features(test_ts)

    all_predictions = []

    # Model 1: KNN on simple features
    print("  Training KNN...")
    preds1 = np.zeros((len(X_test_simple), 3))
    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train_simple[train_mask]
        y_p_train = train_targets[train_mask]
        X_p_test = X_test_simple[test_mask]

        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        for target_idx in range(3):
            k = min(5, len(X_p_train) - 1)
            model = KNeighborsRegressor(n_neighbors=k, weights='distance')
            model.fit(X_p_train_scaled, y_p_train[:, target_idx])
            test_indices = np.where(test_mask)[0]
            preds1[test_indices, target_idx] = model.predict(X_p_test_scaled)

    all_predictions.append(preds1)

    # Model 2: Ridge on simple features
    print("  Training Ridge...")
    preds2 = np.zeros((len(X_test_simple), 3))
    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train_simple[train_mask]
        y_p_train = train_targets[train_mask]
        X_p_test = X_test_simple[test_mask]

        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        for target_idx in range(3):
            model = Ridge(alpha=10.0)
            model.fit(X_p_train_scaled, y_p_train[:, target_idx])
            test_indices = np.where(test_mask)[0]
            preds2[test_indices, target_idx] = model.predict(X_p_test_scaled)

    all_predictions.append(preds2)

    # Model 3: Bayesian Ridge on PCA features
    print("  Training Bayesian Ridge + PCA...")
    preds3 = np.zeros((len(X_test_simple), 3))
    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train_simple[train_mask]
        y_p_train = train_targets[train_mask]
        X_p_test = X_test_simple[test_mask]

        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        n_comp = min(20, X_p_train_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_p_train_pca = pca.fit_transform(X_p_train_scaled)
        X_p_test_pca = pca.transform(X_p_test_scaled)

        for target_idx in range(3):
            model = BayesianRidge()
            model.fit(X_p_train_pca, y_p_train[:, target_idx])
            test_indices = np.where(test_mask)[0]
            preds3[test_indices, target_idx] = model.predict(X_p_test_pca)

    all_predictions.append(preds3)

    # Model 4: Critical frame features + Ridge
    print("  Training Critical Frame models...")
    preds4 = np.zeros((len(test_ts), 3))
    for target_idx, target in enumerate(TARGETS):
        X_train_cf = extract_critical_frame_features(train_ts, target)
        X_test_cf = extract_critical_frame_features(test_ts, target)

        for pid in sorted(np.unique(train_pids)):
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_p_train = X_train_cf[train_mask]
            y_p_train = train_targets[train_mask, target_idx]
            X_p_test = X_test_cf[test_mask]

            scaler = StandardScaler()
            X_p_train_scaled = scaler.fit_transform(X_p_train)
            X_p_test_scaled = scaler.transform(X_p_test)

            model = Ridge(alpha=10.0)
            model.fit(X_p_train_scaled, y_p_train)
            test_indices = np.where(test_mask)[0]
            preds4[test_indices, target_idx] = model.predict(X_p_test_scaled)

    all_predictions.append(preds4)

    # Ensemble: simple average
    ensemble_pred = np.mean(all_predictions, axis=0)

    print(f"  Ensemble of {len(all_predictions)} models")

    return ensemble_pred, None


def create_submission(test_ids, predictions, name):
    """Create submission file."""
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    # Clip to valid range
    scaled_predictions = np.clip(scaled_predictions, 0, 1)

    df = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions[:, 0],
        'scaled_depth': scaled_predictions[:, 1],
        'scaled_left_right': scaled_predictions[:, 2],
    })

    nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
            if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    df.to_csv(filepath, index=False)

    print(f"\nSubmission {next_num} ({name}):")
    print(f"  angle_std:  {df['scaled_angle'].std():.4f}")
    print(f"  depth_mean: {df['scaled_depth'].mean():.4f}")
    print(f"  depth_max:  {df['scaled_depth'].max():.4f}")

    return next_num, df


def main():
    print("="*70)
    print("ALTERNATIVE APPROACHES")
    print("="*70)

    # Load data
    data = load_raw_data()

    results = {}

    # Run each approach
    preds_knn, cv_knn = approach_knn(data)
    results['knn'] = {'preds': preds_knn, 'cv': cv_knn}

    preds_cf, cv_cf = approach_critical_frames(data)
    results['critical_frames'] = {'preds': preds_cf, 'cv': cv_cf}

    preds_br, cv_br = approach_bayesian_ridge(data)
    results['bayesian_ridge'] = {'preds': preds_br, 'cv': cv_br}

    preds_mean, cv_mean = approach_per_player_mean(data)
    results['player_mean'] = {'preds': preds_mean, 'cv': cv_mean}

    preds_dw, _ = approach_distance_weighted(data)
    results['distance_weighted'] = {'preds': preds_dw, 'cv': None}

    preds_ens, _ = approach_ensemble_diverse(data)
    results['diverse_ensemble'] = {'preds': preds_ens, 'cv': None}

    # Create submissions for promising approaches
    print("\n" + "="*70)
    print("CREATING SUBMISSIONS")
    print("="*70)

    submissions = []

    # Save diverse ensemble (combines multiple approaches)
    num, df = create_submission(data["test_ids"], preds_ens, "diverse_ensemble")
    submissions.append((num, "diverse_ensemble", df))

    # Save KNN
    num, df = create_submission(data["test_ids"], preds_knn, "knn")
    submissions.append((num, "knn", df))

    # Save critical frames
    num, df = create_submission(data["test_ids"], preds_cf, "critical_frames")
    submissions.append((num, "critical_frames", df))

    # Blend diverse ensemble with sub25 (the proven best)
    print("\n--- Blending with sub25 ---")
    sub25 = pd.read_csv(SUBMISSION_DIR / "submission_25.csv")

    for w_new in [0.3, 0.5]:
        w_25 = 1 - w_new
        blended = pd.DataFrame({'id': data["test_ids"]})

        # Get scaled predictions for ensemble
        target_scalers = {}
        for target in TARGETS:
            target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

        scaled_ens = np.zeros_like(preds_ens)
        for i, target in enumerate(TARGETS):
            scaled_ens[:, i] = target_scalers[target].transform(
                preds_ens[:, i].reshape(-1, 1)
            ).flatten()
        scaled_ens = np.clip(scaled_ens, 0, 1)

        for i, col in enumerate(['scaled_angle', 'scaled_depth', 'scaled_left_right']):
            blended[col] = w_new * scaled_ens[:, i] + w_25 * sub25[col]

        nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
                if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1

        filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blended.to_csv(filepath, index=False)

        print(f"\nSubmission {next_num}: {w_new*100:.0f}% diverse_ensemble + {w_25*100:.0f}% sub25")
        print(f"  angle_std:  {blended['scaled_angle'].std():.4f}")
        print(f"  depth_mean: {blended['scaled_depth'].mean():.4f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nApproach CV scores:")
    for name, result in results.items():
        if result['cv'] is not None:
            print(f"  {name}: {result['cv']:.4f}")

    print("\nRecommendation: Try the diverse_ensemble + sub25 blend")
    print("This combines fundamentally different approaches with proven performance.")


if __name__ == "__main__":
    main()
