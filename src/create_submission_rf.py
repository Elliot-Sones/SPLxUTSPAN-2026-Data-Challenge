"""
Create submission using Random Forest - the best model from overnight experiments.

Best overnight result: Random Forest with F4 features, MSE 0.01456
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_SPLIT_DIR = PROJECT_DIR / "data_split"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output" / "overnight"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

# Random Forest parameters (from overnight experiments)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_leaf': 5,
    'n_jobs': -1,
    'random_state': 42,
}


def parse_array_string(s):
    """Parse string array to numpy array."""
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    arr = np.array(json.loads(s), dtype=np.float32)
    return arr


def load_train_features():
    """Load training features from data_split."""
    # Load from data_split (consistent with test feature extraction)
    print("Loading training features from data_split...")
    all_X = []
    all_y = []

    for player_dir in sorted(DATA_SPLIT_DIR.iterdir()):
        if not player_dir.is_dir():
            continue

        angle_file = player_dir / "angle.csv"
        if angle_file.exists():
            df = pd.read_csv(angle_file)

            # Feature columns (exclude meta and target)
            exclude = ['id', 'target', 'participant_id', 'participant_1',
                      'participant_2', 'participant_3', 'participant_4', 'participant_5']
            feature_cols = [c for c in df.columns if c not in exclude]

            X_player = df[feature_cols].values
            all_X.append(X_player)

            # Targets
            y_row = np.zeros((len(df), 3))
            y_row[:, 0] = df['target'].values

            depth_file = player_dir / "depth.csv"
            if depth_file.exists():
                y_row[:, 1] = pd.read_csv(depth_file)['target'].values

            lr_file = player_dir / "left_right.csv"
            if lr_file.exists():
                y_row[:, 2] = pd.read_csv(lr_file)['target'].values

            all_y.append(y_row)

    X = np.vstack(all_X)
    y = np.vstack(all_y)

    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def get_feature_names():
    """Get feature names from data_split."""
    for player_dir in sorted(DATA_SPLIT_DIR.iterdir()):
        if not player_dir.is_dir():
            continue
        angle_file = player_dir / "angle.csv"
        if angle_file.exists():
            df = pd.read_csv(angle_file)
            exclude = ['id', 'target', 'participant_id', 'participant_1',
                      'participant_2', 'participant_3', 'participant_4', 'participant_5']
            return [c for c in df.columns if c not in exclude]
    return []


def extract_test_features():
    """Extract features from test.csv matching training features."""
    print("\nExtracting test features...")

    # Get feature names from training
    feature_names = get_feature_names()
    print(f"  Using {len(feature_names)} features from training")

    # Load test data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in test_df.columns if c not in meta_cols]

    # We need to extract the same features as training
    # Import the hybrid feature extraction functions (same as used for data_split)
    try:
        from hybrid_features import extract_hybrid_features, init_keypoint_mapping
    except ImportError:
        from src.hybrid_features import extract_hybrid_features, init_keypoint_mapping

    # Initialize keypoint mapping
    init_keypoint_mapping(keypoint_cols)

    all_features = []
    test_ids = []

    for idx, row in test_df.iterrows():
        # Parse timeseries
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        # Extract features (same hybrid features used for data_split)
        features = extract_hybrid_features(timeseries, participant_id=row['participant_id'], smooth=False)
        all_features.append(features)
        test_ids.append(row['id'])

        if (idx + 1) % 20 == 0:
            print(f"    Processed {idx + 1}/{len(test_df)} test shots")

    # Convert to array matching training feature order
    X_test = np.zeros((len(all_features), len(feature_names)), dtype=np.float32)
    for i, feat_dict in enumerate(all_features):
        for j, name in enumerate(feature_names):
            X_test[i, j] = feat_dict.get(name, 0.0)

    print(f"  Test features: {X_test.shape}")
    return X_test, np.array(test_ids)


def train_and_predict():
    """Train Random Forest and predict on test."""

    # Load training data
    X_train, y_train = load_train_features()

    # Extract test features
    X_test, test_ids = extract_test_features()

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle NaN
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    # Train per-target Random Forest models
    print("\nTraining Random Forest models...")
    predictions = np.zeros((len(X_test), 3))

    for target_idx, target in enumerate(TARGETS):
        print(f"  Training {target}...")
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train_scaled, y_train[:, target_idx])
        predictions[:, target_idx] = model.predict(X_test_scaled)
        print(f"    Done. Predictions range: [{predictions[:, target_idx].min():.4f}, {predictions[:, target_idx].max():.4f}]")

    return test_ids, predictions


def create_submission(test_ids, predictions, submission_num):
    """Create submission CSV with scaled predictions."""

    # Load the MinMaxScalers that were used on the original targets
    print("\nLoading target scalers...")
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Scale predictions to [0, 1] range using the same scalers
    print("Scaling predictions to [0, 1] range...")
    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        # MinMaxScaler.transform expects 2D array
        scaled_predictions[:, i] = scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    # Create submission DataFrame with correct column names
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

    # Stats for original predictions
    print(f"\nOriginal prediction statistics:")
    for i, col in enumerate(TARGETS):
        print(f"  {col}: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}, "
              f"min={predictions[:, i].min():.4f}, max={predictions[:, i].max():.4f}")

    # Stats for scaled predictions
    print(f"\nScaled prediction statistics (should be in [0, 1] range):")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}, "
              f"min={submission[col].min():.4f}, max={submission[col].max():.4f}")

    return filepath


def main():
    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing]
        next_num = max(nums) + 1
    else:
        next_num = 1

    print("=" * 70)
    print(f"CREATING SUBMISSION {next_num}")
    print("=" * 70)
    print("\nModel: Random Forest (best from overnight experiments)")
    print("Features: F4 hybrid features")
    print(f"Parameters: {RF_PARAMS}")
    print("\nCV Score from overnight: 0.01456 (20.9% better than baseline)")

    # Train and predict
    test_ids, predictions = train_and_predict()

    # Create submission
    filepath = create_submission(test_ids, predictions, next_num)

    # Log details
    log_entry = f"""
## Submission {next_num}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Model**: Random Forest

**Configuration**:
- Features: F4 hybrid features (from data_split)
- Model: RandomForestRegressor
- n_estimators: 200
- max_depth: 10
- min_samples_leaf: 5

**CV Score** (from overnight experiments):
- Total MSE: 0.01456
- Improvement over baseline: 20.9%

**File**: {filepath}

**Submission Score**: TBD (update after submission)
"""

    print("\n" + "=" * 70)
    print("SUBMISSION DETAILS")
    print("=" * 70)
    print(log_entry)

    return filepath


if __name__ == "__main__":
    main()
