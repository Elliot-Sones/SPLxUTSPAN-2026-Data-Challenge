"""
Outlier-Robust Model

Handle outlier samples specially to avoid bad predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.covariance import EllipticEnvelope
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def main():
    print("=" * 70)
    print("OUTLIER-ROBUST MODEL")
    print("=" * 70)

    scalers_dict = load_scalers()

    # Build features
    print("\nLoading data...")
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    train_features = []
    train_targets = []
    train_pids = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        train_features.append(features)
        train_targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        train_pids.append(metadata["participant_id"])

    train_df = pd.DataFrame(train_features).fillna(0)
    train_df = train_df.loc[:, train_df.nunique() > 1]

    X_train = train_df.values
    y_train = np.array(train_targets)
    pids_train = np.array(train_pids)

    test_features = []
    test_pids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        test_features.append(features)
        test_pids.append(metadata["participant_id"])

    test_df = pd.DataFrame(test_features).fillna(0)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    X_test = test_df[train_df.columns].values
    pids_test = np.array(test_pids)

    target_names = ["angle", "depth", "left_right"]

    # Detect test outliers using feature space
    print("\nDetecting outliers in feature space...")

    test_meta = load_metadata(train=False)
    outlier_scores = np.zeros(len(X_test))

    for pid in np.unique(pids_test):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_tr = X_train[train_mask]
        X_te = X_test[test_mask]

        # Use RobustScaler to reduce impact of outliers
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Compute distance from training mean
        mean = X_tr_scaled.mean(axis=0)
        std = X_tr_scaled.std(axis=0) + 1e-8

        test_z_scores = np.abs((X_te_scaled - mean) / std)
        outlier_scores[test_mask] = test_z_scores.max(axis=1)

    print(f"Outlier score range: [{outlier_scores.min():.2f}, {outlier_scores.max():.2f}]")

    # Identify outlier samples
    outlier_threshold = np.percentile(outlier_scores, 90)  # Top 10% are outliers
    is_outlier = outlier_scores > outlier_threshold

    print(f"Outlier threshold: {outlier_threshold:.2f}")
    print(f"Number of outliers: {is_outlier.sum()} out of {len(is_outlier)}")

    print("\nOutlier samples:")
    for idx in np.where(is_outlier)[0]:
        pid = test_meta.iloc[idx]["participant_id"]
        print(f"  Sample {idx} (Player {pid}): score={outlier_scores[idx]:.2f}")

    # Strategy 1: Use Huber regression (robust to outliers)
    print("\n" + "=" * 70)
    print("STRATEGY 1: HUBER REGRESSION")
    print("=" * 70)

    huber_preds = np.zeros((len(X_test), 3))

    for pid in np.unique(pids_train):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            huber = HuberRegressor(epsilon=1.5, alpha=0.01, max_iter=200)
            huber.fit(X_tr_scaled, y_tr[:, t_idx])
            huber_preds[test_mask, t_idx] = huber.predict(X_te_scaled)

    # Scale
    scaled_huber = np.zeros_like(huber_preds)
    for i, target in enumerate(target_names):
        scaled_huber[:, i] = scalers_dict[target].transform(huber_preds[:, i].reshape(-1, 1)).ravel()

    print(f"Huber profile: angle_std={scaled_huber[:, 0].std():.4f}, depth_mean={scaled_huber[:, 1].mean():.4f}")

    # Strategy 2: Replace outliers with Sub 133 predictions
    print("\n" + "=" * 70)
    print("STRATEGY 2: REPLACE OUTLIERS WITH SUB 133")
    print("=" * 70)

    # Get our standard predictions
    ridge_preds = np.zeros((len(X_test), 3))

    for pid in np.unique(pids_train):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            ridge = Ridge(alpha=100)
            ridge.fit(X_tr_scaled, y_tr[:, t_idx])
            ridge_preds[test_mask, t_idx] = ridge.predict(X_te_scaled)

    # Scale
    scaled_ridge = np.zeros_like(ridge_preds)
    for i, target in enumerate(target_names):
        scaled_ridge[:, i] = scalers_dict[target].transform(ridge_preds[:, i].reshape(-1, 1)).ravel()

    # Load Sub 133
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    # Replace outliers
    replaced = scaled_ridge.copy()
    for idx in np.where(is_outlier)[0]:
        replaced[idx, :] = sub133[cols].values[idx, :]

    print(f"After replacing outliers: angle_std={replaced[:, 0].std():.4f}, depth_mean={replaced[:, 1].mean():.4f}")

    # Correlation with Sub 133
    corr = np.corrcoef(replaced.ravel(), sub133[cols].values.ravel())[0, 1]
    print(f"Correlation with Sub 133: {corr:.4f}")

    # Strategy 3: Weighted blend based on outlier score
    print("\n" + "=" * 70)
    print("STRATEGY 3: OUTLIER-WEIGHTED BLEND")
    print("=" * 70)

    # Higher outlier score -> trust Sub 133 more
    norm_outlier = outlier_scores / outlier_scores.max()
    ridge_weight = 0.2 * (1 - norm_outlier)

    weighted_blend = np.zeros_like(scaled_ridge)
    for i in range(3):
        weighted_blend[:, i] = (1 - ridge_weight) * sub133[cols[i]].values + ridge_weight * scaled_ridge[:, i]

    print(f"Outlier-weighted profile: angle_std={weighted_blend[:, 0].std():.4f}, depth_mean={weighted_blend[:, 1].mean():.4f}")

    corr = np.corrcoef(weighted_blend.ravel(), sub133[cols].values.ravel())[0, 1]
    print(f"Correlation with Sub 133: {corr:.4f}")

    # Save best submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    # Save Huber model
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": scaled_huber[:, 0],
        "scaled_depth": scaled_huber[:, 1],
        "scaled_left_right": scaled_huber[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved Huber model: {path}")

    next_num += 1

    # Save outlier-replaced
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": replaced[:, 0],
        "scaled_depth": replaced[:, 1],
        "scaled_left_right": replaced[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved outlier-replaced: {path}")

    next_num += 1

    # Save outlier-weighted blend
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": weighted_blend[:, 0],
        "scaled_depth": weighted_blend[:, 1],
        "scaled_left_right": weighted_blend[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved outlier-weighted blend: {path}")


if __name__ == "__main__":
    main()
