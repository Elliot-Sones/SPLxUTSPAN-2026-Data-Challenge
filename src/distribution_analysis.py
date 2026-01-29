"""
Distribution Analysis: Understand train/test shift and create adjusted predictions.

The idea is that test samples might have different characteristics than training.
If we can identify and adjust for this, we might improve predictions.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]


def load_data():
    """Load data with features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    def parse_array_string(s):
        if pd.isna(s):
            return np.full(240, np.nan, dtype=np.float32)
        s = s.replace("nan", "null")
        return np.array(json.loads(s), dtype=np.float32)

    def extract_features(df, is_train=True):
        all_features = []
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
            combined = {**hybrid_feats, **advanced_feats}
            all_features.append(combined)

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        return all_features, ids, pids, targets if is_train else None

    print("Extracting features...")
    train_feats, train_ids, train_pids, train_targets = extract_features(train_df, True)
    test_feats, test_ids, test_pids, _ = extract_features(test_df, False)

    feature_names = sorted(train_feats[0].keys())
    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "X_test": X_test,
        "test_ids": np.array(test_ids),
        "test_pids": np.array(test_pids),
        "feature_names": feature_names,
    }


def adversarial_validation(X_train, X_test):
    """Check if train and test are distinguishable."""
    print("\n" + "="*70)
    print("ADVERSARIAL VALIDATION")
    print("="*70)

    # Combine train and test
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Train classifier to distinguish train from test
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_scaled, y_combined, cv=5, scoring='roc_auc')

    print(f"AUC scores: {scores}")
    print(f"Mean AUC: {scores.mean():.4f} (0.5 = no shift, 1.0 = complete shift)")

    if scores.mean() > 0.6:
        print("WARNING: Significant train/test distribution shift detected!")
    else:
        print("Train/test distributions are similar.")

    # Fit on all data to get feature importances
    clf.fit(X_scaled, y_combined)

    return clf, scores.mean()


def get_test_similarity_weights(X_train, X_test, train_pids, test_pids):
    """
    Compute similarity weights: how similar each training sample is to test distribution.
    Samples more similar to test should get higher weight.
    """
    print("\n" + "="*70)
    print("COMPUTING SIMILARITY WEIGHTS")
    print("="*70)

    weights = np.ones(len(X_train))

    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train[train_mask]
        X_p_test = X_test[test_mask]

        if len(X_p_test) == 0:
            continue

        # Compute mean of test samples for this player
        test_mean = X_p_test.mean(axis=0)

        # Compute distance of each training sample to test mean
        distances = np.linalg.norm(X_p_train - test_mean, axis=1)

        # Convert to weights (inverse distance)
        max_dist = distances.max() + 1e-6
        sample_weights = 1 - (distances / max_dist)

        # Assign weights
        train_indices = np.where(train_mask)[0]
        weights[train_indices] = sample_weights

    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Weight mean: {weights.mean():.4f}")

    return weights


def train_with_weights(X_train, y_train, X_test, train_pids, test_pids, weights):
    """Train models using sample weights."""
    print("\n" + "="*70)
    print("TRAINING WITH SIMILARITY WEIGHTS")
    print("="*70)

    predictions = np.zeros((len(X_test), 3))

    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train[train_mask]
        y_p_train = y_train[train_mask]
        X_p_test = X_test[test_mask]
        w_p = weights[train_mask]

        if len(X_p_test) == 0:
            continue

        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        for target_idx in range(3):
            y_target = y_p_train[:, target_idx]

            # LightGBM with sample weights
            model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=10,
                learning_rate=0.05,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_p_train_scaled, y_target, sample_weight=w_p)

            test_indices = np.where(test_mask)[0]
            predictions[test_indices, target_idx] = model.predict(X_p_test_scaled)

    return predictions


def per_player_calibration(X_train, y_train, X_test, train_pids, test_pids):
    """
    Per-player calibration: adjust predictions based on player statistics.
    """
    print("\n" + "="*70)
    print("PER-PLAYER CALIBRATED PREDICTIONS")
    print("="*70)

    predictions = np.zeros((len(X_test), 3))

    for pid in sorted(np.unique(train_pids)):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_p_train = X_train[train_mask]
        y_p_train = y_train[train_mask]
        X_p_test = X_test[test_mask]

        if len(X_p_test) == 0:
            continue

        scaler = StandardScaler()
        X_p_train_scaled = scaler.fit_transform(X_p_train)
        X_p_test_scaled = scaler.transform(X_p_test)

        # Calculate player stats from training data
        player_means = y_p_train.mean(axis=0)
        player_stds = y_p_train.std(axis=0)

        for target_idx in range(3):
            y_target = y_p_train[:, target_idx]

            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=10,
                learning_rate=0.05,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_p_train_scaled, y_target)

            raw_pred = model.predict(X_p_test_scaled)

            # Calibrate: ensure predictions have similar distribution to training
            pred_mean = raw_pred.mean()
            pred_std = raw_pred.std() + 1e-6

            # Normalize and rescale to match training distribution
            calibrated = (raw_pred - pred_mean) / pred_std
            calibrated = calibrated * player_stds[target_idx] + player_means[target_idx]

            test_indices = np.where(test_mask)[0]
            predictions[test_indices, target_idx] = calibrated

        print(f"Player {pid}: train samples={len(X_p_train)}, test samples={len(X_p_test)}")

    return predictions


def simple_shrinkage(sub25_path, shrinkage_factor=0.95):
    """
    Apply shrinkage: pull predictions toward player mean.
    This is a regularization technique that can reduce variance.
    """
    print("\n" + "="*70)
    print(f"APPLYING SHRINKAGE (factor={shrinkage_factor})")
    print("="*70)

    sub25 = pd.read_csv(sub25_path)

    # Load test data to get player IDs
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_pids = test_df['participant_id'].values

    # Load training data to get player means
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    # Compute player means from training
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    player_means = {}
    for pid in sorted(train_df['participant_id'].unique()):
        mask = train_df['participant_id'] == pid
        player_means[pid] = {}
        for target in TARGETS:
            raw_mean = train_df.loc[mask, target].mean()
            # Scale to [0, 1]
            scaled_mean = target_scalers[target].transform([[raw_mean]])[0, 0]
            player_means[pid][target] = scaled_mean

    # Apply shrinkage
    shrunk = sub25.copy()
    for i, pid in enumerate(test_pids):
        for target, col in zip(TARGETS, ['scaled_angle', 'scaled_depth', 'scaled_left_right']):
            current = sub25[col].iloc[i]
            mean = player_means[pid][target]
            # Shrink toward mean
            shrunk.loc[i, col] = mean + shrinkage_factor * (current - mean)

    return shrunk


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
    print("DISTRIBUTION ANALYSIS AND ADJUSTED PREDICTIONS")
    print("="*70)

    data = load_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    train_pids = data["train_pids"]
    test_pids = data["test_pids"]
    test_ids = data["test_ids"]

    # 1. Adversarial validation
    clf, auc = adversarial_validation(X_train, X_test)

    # 2. Compute similarity weights
    weights = get_test_similarity_weights(X_train, X_test, train_pids, test_pids)

    # 3. Train with weights
    preds_weighted = train_with_weights(X_train, y_train, X_test, train_pids, test_pids, weights)
    num_w, df_w = create_submission(test_ids, preds_weighted, "weighted")

    # 4. Per-player calibration
    preds_calib = per_player_calibration(X_train, y_train, X_test, train_pids, test_pids)
    num_c, df_c = create_submission(test_ids, preds_calib, "calibrated")

    # 5. Shrinkage on sub25
    sub25_path = SUBMISSION_DIR / "submission_25.csv"
    for factor in [0.95, 0.90]:
        shrunk = simple_shrinkage(sub25_path, factor)

        nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
                if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1

        filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
        shrunk.to_csv(filepath, index=False)

        print(f"\nSubmission {next_num} (shrinkage {factor}):")
        print(f"  angle_std:  {shrunk['scaled_angle'].std():.4f}")
        print(f"  depth_mean: {shrunk['scaled_depth'].mean():.4f}")

    # 6. Blend weighted predictions with sub25
    sub25 = pd.read_csv(sub25_path)

    for w_new in [0.3]:
        w_25 = 1 - w_new
        blended = pd.DataFrame({'id': test_ids})

        for i, col in enumerate(['scaled_angle', 'scaled_depth', 'scaled_left_right']):
            blended[col] = w_new * df_w[col] + w_25 * sub25[col]

        nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
                if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1

        filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blended.to_csv(filepath, index=False)

        print(f"\nSubmission {next_num}: {w_new*100:.0f}% weighted + {w_25*100:.0f}% sub25")
        print(f"  angle_std:  {blended['scaled_angle'].std():.4f}")
        print(f"  depth_mean: {blended['scaled_depth'].mean():.4f}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
