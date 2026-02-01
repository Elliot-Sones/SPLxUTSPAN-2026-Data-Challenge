"""
Optimal Blend - Find the best combination of all approaches

Also tries direct optimization of predictions.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.optimize import minimize
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


def model_prototype(X_train, y_train, X_test, n_components=10):
    """Prototype-based model."""
    n_samples = X_train.shape[0]
    n_components = min(n_components, n_samples - 1)

    prototype = X_train.mean(axis=0)

    train_dist = np.linalg.norm(X_train - prototype, axis=1, keepdims=True)
    test_dist = np.linalg.norm(X_test - prototype, axis=1, keepdims=True)

    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(X_train)
    test_pca = pca.transform(X_test)

    X_train_feat = np.hstack([train_dist, train_pca])
    X_test_feat = np.hstack([test_dist, test_pca])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    preds = np.zeros((len(X_test), 3))
    for t_idx in range(3):
        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y_train[:, t_idx])
        preds[:, t_idx] = model.predict(X_test_scaled)

    return preds, X_train_scaled, X_test_scaled


def model_knn(X_train, y_train, X_test, k=5):
    """KNN-based model."""
    preds = np.zeros((len(X_test), 3))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for t_idx in range(3):
        knn = KNeighborsRegressor(n_neighbors=min(k, len(X_train)-1), weights='distance')
        knn.fit(X_train_scaled, y_train[:, t_idx])
        preds[:, t_idx] = knn.predict(X_test_scaled)

    return preds


def model_ridge_pca(X_train, y_train, X_test, n_components=15, alpha=100.0):
    """Ridge with PCA."""
    n_samples = X_train.shape[0]
    n_components = min(n_components, n_samples - 1)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    preds = np.zeros((len(X_test), 3))
    for t_idx in range(3):
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train[:, t_idx])
        preds[:, t_idx] = model.predict(X_test_scaled)

    return preds


def model_mean(y_train, n_test):
    """Mean baseline."""
    return np.tile(y_train.mean(axis=0), (n_test, 1))


def optimal_blend(data):
    """
    Find optimal blend of multiple base models.
    """
    print("\n" + "="*60)
    print("OPTIMAL BLEND")
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

    # Collect OOF predictions from all models
    n_models = 5
    oof_all = np.zeros((len(train_series), 3, n_models))
    test_all = np.zeros((len(test_series), 3, n_models))

    for pid in unique_pids:
        print(f"  Player {pid}...")
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # Cross-validation for each model
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_test_preds = {m: [] for m in range(n_models)}

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            # Model 0: Mean
            mean_preds = model_mean(y_tr, len(X_val))
            oof_all[player_indices[val_idx], :, 0] = mean_preds

            # Model 1: Prototype
            proto_preds, _, _ = model_prototype(X_tr, y_tr, X_val)
            oof_all[player_indices[val_idx], :, 1] = proto_preds

            # Model 2: KNN k=3
            knn3_preds = model_knn(X_tr, y_tr, X_val, k=3)
            oof_all[player_indices[val_idx], :, 2] = knn3_preds

            # Model 3: KNN k=5
            knn5_preds = model_knn(X_tr, y_tr, X_val, k=5)
            oof_all[player_indices[val_idx], :, 3] = knn5_preds

            # Model 4: Ridge
            ridge_preds = model_ridge_pca(X_tr, y_tr, X_val)
            oof_all[player_indices[val_idx], :, 4] = ridge_preds

            # Collect test predictions
            fold_test_preds[0].append(model_mean(y_tr, len(X_test)))
            fold_test_preds[1].append(model_prototype(X_tr, y_tr, X_test)[0])
            fold_test_preds[2].append(model_knn(X_tr, y_tr, X_test, k=3))
            fold_test_preds[3].append(model_knn(X_tr, y_tr, X_test, k=5))
            fold_test_preds[4].append(model_ridge_pca(X_tr, y_tr, X_test))

        # Average test predictions
        for m in range(n_models):
            test_all[test_mask, :, m] = np.mean(fold_test_preds[m], axis=0)

    # Find optimal weights per target
    print("\nOptimizing blend weights...")

    optimal_weights = np.zeros((3, n_models))
    cv_per_target = []

    for t_idx, target in enumerate(TARGETS):
        y_true = train_targets[:, t_idx]

        def loss(weights):
            weights = np.abs(weights)
            weights = weights / (weights.sum() + 1e-8)
            ensemble = np.sum(oof_all[:, t_idx, :] * weights, axis=1)
            return np.mean((ensemble - y_true)**2)

        init = np.ones(n_models) / n_models
        result = minimize(loss, init, method='Nelder-Mead', options={'maxiter': 1000})
        optimal_weights[t_idx] = np.abs(result.x) / (np.abs(result.x).sum() + 1e-8)

        # Compute CV score for this target
        ensemble_oof = np.sum(oof_all[:, t_idx, :] * optimal_weights[t_idx], axis=1)
        mse = np.mean((ensemble_oof - y_true)**2)
        scaled_mse = mse / (ranges[target]**2)
        cv_per_target.append(scaled_mse)
        print(f"  {target}: weights = {optimal_weights[t_idx]}, CV = {scaled_mse:.6f}")

    # Generate final predictions
    predictions = np.zeros((len(test_series), 3))
    for t_idx in range(3):
        predictions[:, t_idx] = np.sum(test_all[:, t_idx, :] * optimal_weights[t_idx], axis=1)

    cv_score = np.mean(cv_per_target)
    print(f"\n  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "optimal_blend", optimal_weights


def direct_optimization(data):
    """
    Direct optimization: Find predictions that minimize expected error.

    Idea: Instead of training models, directly optimize prediction values
    to minimize cross-validation error using gradient-free optimization.
    """
    print("\n" + "="*60)
    print("DIRECT OPTIMIZATION")
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

    for pid in unique_pids:
        print(f"  Player {pid}...")
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        n_test = len(X_test)

        # PCA for similarity computation
        pca = PCA(n_components=min(15, len(X_train)-1))
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Compute similarity matrix (test to train)
        from scipy.spatial.distance import cdist
        distances = cdist(X_test_pca, X_train_pca)

        for t_idx in range(3):
            y = y_train[:, t_idx]

            def objective(test_preds):
                """
                For each test prediction, compute expected error based on
                similar training samples.
                """
                total_error = 0
                for i in range(n_test):
                    # Weight by similarity
                    sigma = np.median(distances[i]) + 1e-6
                    weights = np.exp(-distances[i] / (2 * sigma))
                    weights /= weights.sum()

                    # Expected squared error
                    expected_error = np.sum(weights * (test_preds[i] - y)**2)
                    total_error += expected_error

                # Regularization: predictions should be close to weighted mean
                for i in range(n_test):
                    sigma = np.median(distances[i]) + 1e-6
                    weights = np.exp(-distances[i] / (2 * sigma))
                    weights /= weights.sum()
                    weighted_mean = np.sum(weights * y)
                    total_error += 0.1 * (test_preds[i] - weighted_mean)**2

                return total_error / n_test

            # Initialize with weighted mean
            init = np.zeros(n_test)
            for i in range(n_test):
                sigma = np.median(distances[i]) + 1e-6
                weights = np.exp(-distances[i] / (2 * sigma))
                weights /= weights.sum()
                init[i] = np.sum(weights * y)

            # Optimize
            result = minimize(objective, init, method='L-BFGS-B',
                            options={'maxiter': 100})
            predictions[test_mask, t_idx] = result.x

    # Compute CV using leave-one-out within each player
    print("\nComputing CV...")
    oof_preds = np.zeros_like(train_targets)

    for pid in unique_pids:
        train_mask = train_pids == pid
        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        player_indices = np.where(train_mask)[0]

        pca = PCA(n_components=min(15, len(X_train)-1))
        X_train_pca = pca.fit_transform(X_train)

        for i in range(len(X_train)):
            # Leave one out
            mask = np.ones(len(X_train), dtype=bool)
            mask[i] = False
            X_tr = X_train_pca[mask]
            y_tr = y_train[mask]
            x_test = X_train_pca[i:i+1]

            distances = cdist(x_test, X_tr)[0]
            sigma = np.median(distances) + 1e-6
            weights = np.exp(-distances / (2 * sigma))
            weights /= weights.sum()

            oof_preds[player_indices[i]] = np.sum(weights[:, None] * y_tr, axis=0)

    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "direct_optimization"


def bayesian_model_averaging(data):
    """
    Bayesian Model Averaging: Weight models by their posterior probability.
    """
    print("\n" + "="*60)
    print("BAYESIAN MODEL AVERAGING")
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
        print(f"  Player {pid}...")
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # Define model configurations
        configs = [
            ('pca10_ridge10', 10, 10),
            ('pca10_ridge100', 10, 100),
            ('pca15_ridge10', 15, 10),
            ('pca15_ridge100', 15, 100),
            ('pca15_ridge500', 15, 500),
            ('pca20_ridge100', 20, 100),
        ]

        # Cross-validation to get model errors
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            model_errors = []
            model_test_preds = []

            for name, n_comp, alpha in configs:
                n_comp = min(n_comp, X_train.shape[0] - 2)

                fold_errors = []
                fold_test_preds = []

                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr = y_train[train_idx, t_idx]
                    y_val = y_train[val_idx, t_idx]

                    pca = PCA(n_components=n_comp)
                    X_tr_pca = pca.fit_transform(X_tr)
                    X_val_pca = pca.transform(X_val)
                    X_test_pca = pca.transform(X_test)

                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr_pca)
                    X_val_scaled = scaler.transform(X_val_pca)
                    X_test_scaled = scaler.transform(X_test_pca)

                    model = Ridge(alpha=alpha)
                    model.fit(X_tr_scaled, y_tr)

                    val_pred = model.predict(X_val_scaled)
                    fold_errors.extend((val_pred - y_val)**2)
                    fold_test_preds.append(model.predict(X_test_scaled))

                model_errors.append(np.mean(fold_errors))
                model_test_preds.append(np.mean(fold_test_preds, axis=0))

            # Compute Bayesian weights (inversely proportional to error)
            errors = np.array(model_errors)
            # Avoid numerical issues
            errors = np.clip(errors, 1e-6, None)

            # Posterior probability proportional to exp(-error/2sigma^2)
            sigma = np.std(errors) + 1e-6
            log_weights = -errors / (2 * sigma**2)
            weights = np.exp(log_weights - log_weights.max())
            weights /= weights.sum()

            # Weighted average of predictions
            predictions[test_mask, t_idx] = sum(w * p for w, p in zip(weights, model_test_preds))

        # OOF predictions using best config
        best_config = configs[np.argmin(model_errors)]
        n_comp, alpha = best_config[1], best_config[2]
        n_comp = min(n_comp, X_train.shape[0] - 2)

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            pca = PCA(n_components=n_comp)
            X_tr_pca = pca.fit_transform(X_tr)
            X_val_pca = pca.transform(X_val)

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_pca)
            X_val_scaled = scaler.transform(X_val_pca)

            for t_idx in range(3):
                model = Ridge(alpha=alpha)
                model.fit(X_tr_scaled, y_tr[:, t_idx])
                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val_scaled)

    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, "bayesian_avg"


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
    print("OPTIMAL BLEND AND DIRECT OPTIMIZATION")
    print("="*80)

    data = load_data()

    results = []
    all_predictions = {}

    # Test approaches
    try:
        preds, cv, name, _ = optimal_blend(data)
        results.append({"approach": name, "cv_score": cv})
        all_predictions[name] = preds
    except Exception as e:
        print(f"Error in optimal_blend: {e}")
        import traceback
        traceback.print_exc()

    try:
        preds, cv, name = direct_optimization(data)
        results.append({"approach": name, "cv_score": cv})
        all_predictions[name] = preds
    except Exception as e:
        print(f"Error in direct_optimization: {e}")
        import traceback
        traceback.print_exc()

    try:
        preds, cv, name = bayesian_model_averaging(data)
        results.append({"approach": name, "cv_score": cv})
        all_predictions[name] = preds
    except Exception as e:
        print(f"Error in bayesian_avg: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("cv_score")
        print(results_df.to_string(index=False))

        # Create submission with best
        best = results_df.iloc[0]
        best_name = best["approach"]
        best_preds = all_predictions[best_name]

        filepath, sub_num = create_submission(
            data["test_ids"],
            best_preds,
            best["cv_score"],
            best_name
        )

        # Also save results
        results_df.to_csv(OUTPUT_DIR / "optimal_blend_results.csv", index=False)

    return results_df if results else None


if __name__ == "__main__":
    main()
