"""
Create submission using per-player per-target models with tuned hyperparameters.

Uses the optimal hyperparameters discovered in per_target_experiment.py:
- angle: n_est=111, lr=0.065, aggressive learning
- depth: n_est=176, lr=0.007, conservative learning
- left_right: n_est=154, lr=0.005, most conservative
"""

import json
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM required. Run: uv pip install lightgbm")

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"
SUBMISSION_DIR = Path(__file__).parent.parent / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

# Optimal hyperparameters from per-target tuning experiment (30 trials)
TUNED_PARAMS = {
    "angle": {
        "n_estimators": 111,
        "num_leaves": 30,
        "learning_rate": 0.0646,
        "max_depth": 10,
        "subsample": 0.862,
        "colsample_bytree": 0.766,
        "reg_alpha": 0.0045,
        "reg_lambda": 0.00026,
        "min_child_samples": 23,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    },
    "depth": {
        "n_estimators": 176,
        "num_leaves": 5,
        "learning_rate": 0.00724,
        "max_depth": 12,
        "subsample": 0.755,
        "colsample_bytree": 0.563,
        "reg_alpha": 0.00137,
        "reg_lambda": 0.011,
        "min_child_samples": 30,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    },
    "left_right": {
        "n_estimators": 154,
        "num_leaves": 21,
        "learning_rate": 0.00502,
        "max_depth": 5,
        "subsample": 0.857,
        "colsample_bytree": 0.706,
        "reg_alpha": 0.00132,
        "reg_lambda": 0.041,
        "min_child_samples": 33,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    },
}


try:
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping
    from physics_features import init_keypoint_mapping as physics_init
except ImportError:
    from src.hybrid_features import extract_hybrid_features, init_keypoint_mapping
    from src.physics_features import init_keypoint_mapping as physics_init


def parse_array_string(s):
    """Parse string array to numpy array."""
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    arr = np.array(json.loads(s), dtype=np.float32)
    return arr


def get_keypoint_columns_from_test():
    """Get keypoint column names from test.csv."""
    test_df = pd.read_csv(DATA_DIR / "test.csv", nrows=0)
    meta_cols = ["id", "shot_id", "participant_id"]
    return [c for c in test_df.columns if c not in meta_cols]


def extract_f4_features(timeseries: np.ndarray, participant_id: int, smooth: bool = True) -> dict:
    """
    Extract F4 (hybrid with participant ID) features using the actual hybrid_features module.
    """
    features = extract_hybrid_features(timeseries, participant_id, smooth=smooth)

    # Ensure participant features are included
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    return features


def load_train_data():
    """Load training data from cache."""
    cache_file = Path(__file__).parent.parent / "the_rest" / "output" / "feature_cache" / "features_F4_smooth.pkl"

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']

    if isinstance(data['meta'], pd.DataFrame):
        groups = data['meta']['participant_id'].values
    else:
        groups = data['meta'][:, 2]

    return X, y, groups


def extract_test_features():
    """Extract features from test.csv."""
    print("Extracting test features...")

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in test_df.columns if c not in meta_cols]

    # Initialize keypoint mapping (required for physics features)
    init_keypoint_mapping(keypoint_cols)

    all_features = []
    test_ids = []
    test_participants = []

    for idx, row in test_df.iterrows():
        # Parse timeseries
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        # Extract features using hybrid_features (same as training)
        features = extract_f4_features(timeseries, row['participant_id'], smooth=True)
        all_features.append(features)
        test_ids.append(row['id'])
        test_participants.append(row['participant_id'])

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)} test shots")

    # Convert to array - need to match training feature order
    # Load training to get feature names
    with open(Path(__file__).parent.parent / "the_rest" / "output" / "feature_cache" / "features_F4_smooth.pkl", 'rb') as f:
        train_data = pickle.load(f)
    train_feature_names = train_data['feature_names']

    X_test = np.zeros((len(all_features), len(train_feature_names)), dtype=np.float32)
    for i, feat_dict in enumerate(all_features):
        for j, name in enumerate(train_feature_names):
            X_test[i, j] = feat_dict.get(name, 0.0)

    # Check for feature coverage
    missing_features = []
    for name in train_feature_names:
        if all(feat_dict.get(name, 0.0) == 0.0 for feat_dict in all_features):
            missing_features.append(name)
    if missing_features:
        print(f"  Warning: {len(missing_features)} features have all zeros")

    print(f"  Test features: {X_test.shape}")

    return X_test, np.array(test_ids), np.array(test_participants)


def train_and_predict():
    """Train per-player per-target models and predict on test."""

    # Load training data
    print("Loading training data...")
    X_train, y_train, train_groups = load_train_data()
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Extract test features
    X_test, test_ids, test_groups = extract_test_features()

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train per-player per-target models
    print("\nTraining per-player per-target models...")

    predictions = np.zeros((len(X_test), 3))

    for target_idx, target in enumerate(TARGETS):
        print(f"\n  {target}:")
        params = TUNED_PARAMS[target]
        y_target = y_train[:, target_idx]

        # Train one model per participant
        models = {}
        for pid in np.unique(train_groups):
            mask = train_groups == pid
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_scaled[mask], y_target[mask])
            models[pid] = model
            print(f"    Player {pid}: trained on {mask.sum()} samples")

        # Also train fallback on all data
        fallback = lgb.LGBMRegressor(**params)
        fallback.fit(X_train_scaled, y_target)

        # Predict for test
        for pid in np.unique(test_groups):
            test_mask = test_groups == pid
            if pid in models:
                predictions[test_mask, target_idx] = models[pid].predict(X_test_scaled[test_mask])
            else:
                predictions[test_mask, target_idx] = fallback.predict(X_test_scaled[test_mask])

    return test_ids, predictions


def create_submission(test_ids, predictions, submission_num):
    """Create submission CSV with properly scaled values."""
    import joblib

    # Load scalers
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Scale predictions to [0, 1]
    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        # Use the scaler's transform method
        scaled_predictions[:, i] = scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).ravel()

    # Create submission with correct column names
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions[:, 0],
        'scaled_depth': scaled_predictions[:, 1],
        'scaled_left_right': scaled_predictions[:, 2],
    })

    # Save
    filename = f"submission_{submission_num}.csv"
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\nSubmission saved to: {filepath}")
    print(f"  Rows: {len(submission)}")
    print(f"  Columns: {submission.columns.tolist()}")

    # Stats for raw predictions
    print(f"\nRaw prediction statistics:")
    for i, col in enumerate(TARGETS):
        print(f"  {col}: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}, "
              f"min={predictions[:, i].min():.4f}, max={predictions[:, i].max():.4f}")

    # Stats for scaled predictions
    print(f"\nScaled prediction statistics (should be in [0, 1]):")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}, "
              f"min={submission[col].min():.4f}, max={submission[col].max():.4f}")

    # Verify scaling bounds
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        if submission[col].min() < 0 or submission[col].max() > 1:
            print(f"  WARNING: {col} has values outside [0, 1]!")

    return filepath


def main():
    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing]
        next_num = max(nums) + 1
    else:
        next_num = 1

    print("="*70)
    print(f"CREATING SUBMISSION {next_num}")
    print("="*70)
    print("\nModel: Per-player per-target LightGBM with tuned hyperparameters")
    print("Features: F4 (hybrid with participant ID), 132 features")
    print("Strategy: 15 models (5 players x 3 targets)")
    print("\nTuned hyperparameters:")
    for target, params in TUNED_PARAMS.items():
        print(f"  {target}: n_est={params['n_estimators']}, lr={params['learning_rate']:.4f}")

    # Train and predict
    test_ids, predictions = train_and_predict()

    # Create submission
    filepath = create_submission(test_ids, predictions, next_num)

    # Log submission details
    log_entry = f"""
## Submission {next_num}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Model**: Per-player per-target LightGBM

**Configuration**:
- Features: F4 (hybrid with participant ID), 132 features
- Strategy: 15 models (5 players x 3 targets)
- Preprocessing: StandardScaler

**Tuned Hyperparameters**:
- angle: n_est=111, lr=0.0646, max_depth=10, num_leaves=30
- depth: n_est=176, lr=0.00724, max_depth=12, num_leaves=5
- left_right: n_est=154, lr=0.00502, max_depth=5, num_leaves=21

**CV Score** (from per_target_experiment.py):
- Total: 0.0179
- angle: 0.0207
- depth: 0.0177
- left_right: 0.0154

**File**: {filepath}

**Submission Score**: TBD (update after submission)
"""

    print("\n" + "="*70)
    print("SUBMISSION DETAILS (copy to TEST_RESULTS.md after scoring)")
    print("="*70)
    print(log_entry)

    return filepath


if __name__ == "__main__":
    main()
