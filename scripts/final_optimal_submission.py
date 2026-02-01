"""
Final Optimal Submission

Combines all the best approaches discovered:
1. Prototype model (best for left_right)
2. Target-specific features (best for angle and depth)
3. Optimal blend weights per target
4. Multiple frame windows
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

    # Build keypoint index
    kp_idx = {}
    for idx, col in enumerate(keypoint_cols):
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            kp_name = parts[0]
            coord = parts[1]
            if kp_name not in kp_idx:
                kp_idx[kp_name] = {}
            kp_idx[kp_name][coord] = idx

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
        "kp_idx": kp_idx,
    }


def prototype_model(X_train, y_train, X_test, t_idx, alpha=10.0, n_comp=10):
    """Prototype-based model."""
    n_samples = X_train.shape[0]
    n_comp = min(n_comp, n_samples - 1)

    prototype = X_train.mean(axis=0)
    train_dist = np.linalg.norm(X_train - prototype, axis=1, keepdims=True)
    test_dist = np.linalg.norm(X_test - prototype, axis=1, keepdims=True)

    pca = PCA(n_components=n_comp)
    train_pca = pca.fit_transform(X_train)
    test_pca = pca.transform(X_test)

    X_train_feat = np.hstack([train_dist, train_pca])
    X_test_feat = np.hstack([test_dist, test_pca])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train[:, t_idx])

    return model.predict(X_test_scaled), X_train_scaled, X_test_scaled, model


def knn_model(X_train, y_train, X_test, t_idx, k=5):
    """KNN model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=min(k, len(X_train)-1), weights='distance')
    knn.fit(X_train_scaled, y_train[:, t_idx])

    return knn.predict(X_test_scaled)


def ridge_pca_model(X_train, y_train, X_test, t_idx, n_comp=15, alpha=100.0):
    """Ridge with PCA."""
    n_samples = X_train.shape[0]
    n_comp = min(n_comp, n_samples - 1)

    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train[:, t_idx])

    return model.predict(X_test_scaled)


def run_final_model(data):
    """
    Final optimized model.
    """
    print("\n" + "="*60)
    print("FINAL OPTIMAL MODEL")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Multiple frame windows
    frame_windows = {
        "release": list(range(140, 170)),
        "extended": list(range(130, 180)),
        "narrow": list(range(145, 165)),
    }

    unique_pids = sorted(np.unique(train_pids))

    # Collect predictions from all models
    n_models = 6
    oof_all = np.zeros((len(train_series), 3, n_models))
    test_all = np.zeros((len(test_series), 3, n_models))

    print("\nRunning base models...")

    for pid in unique_pids:
        print(f"  Player {pid}...")
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        player_train = train_series[train_mask]
        player_test = test_series[test_mask]
        player_targets = train_targets[train_mask]
        player_indices = np.where(train_mask)[0]

        # Prepare data for different windows
        def flatten(series, frames):
            flat = series[:, frames, :].reshape(len(series), -1)
            return np.nan_to_num(flat, nan=0.0)

        X_release = flatten(player_train, frame_windows["release"])
        X_test_release = flatten(player_test, frame_windows["release"])

        X_extended = flatten(player_train, frame_windows["extended"])
        X_test_extended = flatten(player_test, frame_windows["extended"])

        X_narrow = flatten(player_train, frame_windows["narrow"])
        X_test_narrow = flatten(player_test, frame_windows["narrow"])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for t_idx in range(3):
            fold_test_preds = {m: [] for m in range(n_models)}

            for train_idx, val_idx in kf.split(X_release):
                # Model 0: Mean
                mean_pred = np.full(len(val_idx), player_targets[train_idx, t_idx].mean())
                oof_all[player_indices[val_idx], t_idx, 0] = mean_pred
                fold_test_preds[0].append(np.full(len(X_test_release), player_targets[train_idx, t_idx].mean()))

                # Model 1: Prototype (release window)
                X_tr, X_val = X_release[train_idx], X_release[val_idx]
                y_tr = player_targets[train_idx]
                proto_pred, X_tr_s, X_val_s, _ = prototype_model(X_tr, y_tr, X_val, t_idx, alpha=10)
                oof_all[player_indices[val_idx], t_idx, 1] = proto_pred
                fold_test_preds[1].append(prototype_model(X_tr, y_tr, X_test_release, t_idx, alpha=10)[0])

                # Model 2: Prototype (extended window)
                X_tr, X_val = X_extended[train_idx], X_extended[val_idx]
                proto_pred, _, _, _ = prototype_model(X_tr, y_tr, X_val, t_idx, alpha=10)
                oof_all[player_indices[val_idx], t_idx, 2] = proto_pred
                fold_test_preds[2].append(prototype_model(X_tr, y_tr, X_test_extended, t_idx, alpha=10)[0])

                # Model 3: KNN k=3
                X_tr, X_val = X_release[train_idx], X_release[val_idx]
                knn_pred = knn_model(X_tr, y_tr, X_val, t_idx, k=3)
                oof_all[player_indices[val_idx], t_idx, 3] = knn_pred
                fold_test_preds[3].append(knn_model(X_tr, y_tr, X_test_release, t_idx, k=3))

                # Model 4: KNN k=5
                knn_pred = knn_model(X_tr, y_tr, X_val, t_idx, k=5)
                oof_all[player_indices[val_idx], t_idx, 4] = knn_pred
                fold_test_preds[4].append(knn_model(X_tr, y_tr, X_test_release, t_idx, k=5))

                # Model 5: Ridge PCA
                ridge_pred = ridge_pca_model(X_tr, y_tr, X_val, t_idx, n_comp=15, alpha=100)
                oof_all[player_indices[val_idx], t_idx, 5] = ridge_pred
                fold_test_preds[5].append(ridge_pca_model(X_tr, y_tr, X_test_release, t_idx, n_comp=15, alpha=100))

            # Average test predictions
            for m in range(n_models):
                test_all[test_mask, t_idx, m] = np.mean(fold_test_preds[m], axis=0)

    # Find optimal weights per target
    print("\nOptimizing blend weights...")

    optimal_weights = np.zeros((3, n_models))
    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for t_idx, target in enumerate(TARGETS):
        y_true = train_targets[:, t_idx]

        def loss(weights):
            weights = np.abs(weights)
            weights = weights / (weights.sum() + 1e-8)
            ensemble = np.sum(oof_all[:, t_idx, :] * weights, axis=1)
            return np.mean((ensemble - y_true)**2)

        # Multiple starts for optimization
        best_loss = float('inf')
        best_weights = None

        for _ in range(10):
            init = np.random.dirichlet(np.ones(n_models))
            result = minimize(loss, init, method='Nelder-Mead', options={'maxiter': 1000})
            if result.fun < best_loss:
                best_loss = result.fun
                best_weights = np.abs(result.x) / (np.abs(result.x).sum() + 1e-8)

        optimal_weights[t_idx] = best_weights

        # Generate predictions
        predictions[:, t_idx] = np.sum(test_all[:, t_idx, :] * optimal_weights[t_idx], axis=1)
        oof_preds[:, t_idx] = np.sum(oof_all[:, t_idx, :] * optimal_weights[t_idx], axis=1)

        mse = np.mean((oof_preds[:, t_idx] - y_true)**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: CV = {scaled_mse:.6f}, weights = {optimal_weights[t_idx].round(3)}")

    total_cv = sum(
        np.mean((oof_preds[:, i] - train_targets[:, i])**2) / (ranges[t]**2)
        for i, t in enumerate(TARGETS)
    ) / 3

    print(f"\n  TOTAL CV: {total_cv:.6f}")

    return predictions, total_cv, oof_preds


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

    # Save prediction stats
    print(f"\n  Prediction statistics:")
    for i, target in enumerate(TARGETS):
        print(f"    {target}: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}")

    return filepath, next_num


def main():
    print("="*80)
    print("FINAL OPTIMAL SUBMISSION")
    print("="*80)

    data = load_data()

    predictions, cv_score, oof_preds = run_final_model(data)

    filepath, sub_num = create_submission(
        data["test_ids"],
        predictions,
        cv_score,
        "final_optimal"
    )

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    ranges = get_target_ranges()
    train_targets = data["train_targets"]

    print("\nPer-target CV breakdown:")
    for i, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, i] - train_targets[:, i])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")

    print(f"\nTOTAL CV: {cv_score:.6f}")
    print(f"TARGET: 0.007000")
    print(f"GAP: {(cv_score - 0.007) * 100:.2f}%")

    return cv_score


if __name__ == "__main__":
    main()
