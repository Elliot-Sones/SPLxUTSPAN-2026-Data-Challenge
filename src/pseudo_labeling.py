"""
Pseudo-labeling: Use current best predictions as labels for test data.

Strategy:
1. Use submission_20 predictions as pseudo-labels for test
2. Combine train + pseudo-labeled test
3. Retrain on combined data
4. Generate new predictions
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data_with_pseudo_labels():
    """Load train data and test data with pseudo-labels from best submission."""
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping

    print("Loading data with pseudo-labels...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Load best submission for pseudo-labels
    best_sub = pd.read_csv(SUBMISSION_DIR / "submission_20.csv")  # LB 0.008619
    print(f"Using submission_20 as pseudo-labels (LB: 0.008619)")

    # Load target scalers to convert back to original scale
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)

    def extract_features(df):
        all_features = []
        ids = []
        pids = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            all_features.append(feats)
            ids.append(row['id'])
            pids.append(row['participant_id'])

        return all_features, ids, pids

    print("Processing training data...")
    train_feats, train_ids, train_pids = extract_features(train_df)

    print("Processing test data...")
    test_feats, test_ids, test_pids = extract_features(test_df)

    feature_names = sorted(train_feats[0].keys())

    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)

    # Train targets (actual)
    y_train = np.zeros((len(train_df), 3), dtype=np.float32)
    y_train[:, 0] = train_df['angle'].values
    y_train[:, 1] = train_df['depth'].values
    y_train[:, 2] = train_df['left_right'].values

    # Test pseudo-labels (from best submission, convert back to original scale)
    y_test_pseudo = np.zeros((len(test_df), 3), dtype=np.float32)

    # Map test ids to submission
    sub_dict = best_sub.set_index('id').to_dict('index')
    for i, test_id in enumerate(test_ids):
        row = sub_dict[test_id]
        # Convert scaled predictions back to original scale
        y_test_pseudo[i, 0] = target_scalers['angle'].inverse_transform([[row['scaled_angle']]])[0, 0]
        y_test_pseudo[i, 1] = target_scalers['depth'].inverse_transform([[row['scaled_depth']]])[0, 0]
        y_test_pseudo[i, 2] = target_scalers['left_right'].inverse_transform([[row['scaled_left_right']]])[0, 0]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Pseudo-label stats:")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: mean={y_test_pseudo[:, i].mean():.2f}, std={y_test_pseudo[:, i].std():.2f}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "X_test": X_test,
        "y_test_pseudo": y_test_pseudo,
        "test_ids": np.array(test_ids),
        "test_pids": np.array(test_pids),
        "feature_names": feature_names,
    }


def train_with_pseudo_labels(data, pseudo_weight=0.5):
    """Train on combined train + weighted pseudo-labeled test data."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    train_pids = data["train_pids"]
    X_test = data["X_test"]
    y_test_pseudo = data["y_test_pseudo"]
    test_pids = data["test_pids"]

    unique_pids = sorted(np.unique(train_pids))

    all_models = {}
    all_scalers = {}

    print("\n" + "="*70)
    print(f"TRAINING WITH PSEUDO-LABELS (weight={pseudo_weight})")
    print("="*70)

    for pid in unique_pids:
        # Get train data for this player
        train_mask = train_pids == pid
        X_player_train = X_train[train_mask]
        y_player_train = y_train[train_mask]

        # Get test data for this player (pseudo-labeled)
        test_mask = test_pids == pid
        X_player_test = X_test[test_mask]
        y_player_test = y_test_pseudo[test_mask]

        n_train = len(X_player_train)
        n_test = len(X_player_test)

        print(f"\n--- Player {pid} (train={n_train}, test={n_test}) ---")

        # Combine train + pseudo-labeled test
        X_combined = np.vstack([X_player_train, X_player_test])
        y_combined = np.vstack([y_player_train, y_player_test])

        # Sample weights: train=1.0, pseudo=pseudo_weight
        sample_weights = np.concatenate([
            np.ones(n_train),
            np.full(n_test, pseudo_weight)
        ])

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[pid] = scaler

        for target_idx, target in enumerate(TARGETS):
            y_target = y_combined[:, target_idx]

            # LightGBM with sample weights
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=10,
                learning_rate=0.05,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            lgb_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'lgb')] = lgb_model

            # CatBoost with sample weights
            cat_model = CatBoostRegressor(
                iterations=100,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_state=42,
                verbose=False
            )
            cat_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'cat')] = cat_model

            # Ridge (no sample weights, use weighted data)
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_model

            # Evaluate on original train data only
            X_train_scaled = scaler.transform(X_player_train)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

            lgb_pred = lgb_model.predict(X_train_scaled)
            cat_pred = cat_model.predict(X_train_scaled)
            ridge_pred = ridge_model.predict(X_train_scaled)

            pred = (lgb_pred + cat_pred + ridge_pred) / 3
            mse = np.mean((pred - y_player_train[:, target_idx]) ** 2)
            print(f"  {target} train MSE: {mse:.4f}")

    return {
        "models": all_models,
        "scalers": all_scalers,
    }


def predict(data, trained):
    """Generate final predictions."""
    X_test = data["X_test"]
    test_pids = data["test_pids"]

    models = trained["models"]
    scalers = trained["scalers"]

    predictions = np.zeros((len(X_test), 3))

    for i, (x, pid) in enumerate(zip(X_test, test_pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        x_scaled = np.nan_to_num(x_scaled, nan=0.0)

        for target_idx, target in enumerate(TARGETS):
            lgb_pred = models[(pid, target, 'lgb')].predict(x_scaled)[0]
            cat_pred = models[(pid, target, 'cat')].predict(x_scaled)[0]
            ridge_pred = models[(pid, target, 'ridge')].predict(x_scaled)[0]

            predictions[i, target_idx] = (lgb_pred + cat_pred + ridge_pred) / 3

    return predictions


def create_submission(test_ids, predictions, submission_num):
    """Create submission."""
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions[:, 0],
        'scaled_depth': scaled_predictions[:, 1],
        'scaled_left_right': scaled_predictions[:, 2],
    })

    filename = f"submission_{submission_num}.csv"
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\nSubmission saved to: {filepath}")

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}")

    return filepath


def main():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    print("="*70)
    print(f"PSEUDO-LABELING SUBMISSION {next_num}")
    print("="*70)

    data = load_data_with_pseudo_labels()

    # Try different pseudo-label weights
    for weight in [0.3, 0.5, 0.7]:
        print(f"\n{'='*70}")
        print(f"Weight = {weight}")
        print('='*70)

        trained = train_with_pseudo_labels(data, pseudo_weight=weight)
        predictions = predict(data, trained)

        create_submission(data["test_ids"], predictions, next_num)
        next_num += 1


if __name__ == "__main__":
    main()
