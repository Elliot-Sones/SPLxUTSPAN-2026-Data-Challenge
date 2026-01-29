"""
Replicate the 0.0104 CV score approach from TEST_PLAN.md.

Configuration:
- Per-participant models with internal 5-fold CV
- LightGBM: n_estimators=100, num_leaves=10, learning_rate=0.05, reg_alpha=0.5, reg_lambda=0.5
- 15 models total (5 participants x 3 targets)
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_SPLIT_DIR = PROJECT_DIR / "data_split"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

# Exact params from TEST_PLAN.md that achieved 0.0104
LGBM_PARAMS = {
    'n_estimators': 100,
    'num_leaves': 10,
    'learning_rate': 0.05,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

try:
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping
except ImportError:
    from src.hybrid_features import extract_hybrid_features, init_keypoint_mapping


def parse_array_string(s):
    """Parse string array to numpy array."""
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    arr = np.array(json.loads(s), dtype=np.float32)
    return arr


def load_train_data_per_player():
    """Load training data grouped by player from data_split."""
    print("Loading training data per player from data_split...")

    player_data = {}

    for player_dir in sorted(DATA_SPLIT_DIR.iterdir()):
        if not player_dir.is_dir():
            continue

        player_id = int(player_dir.name.split('_')[1])

        angle_file = player_dir / "angle.csv"
        if angle_file.exists():
            df = pd.read_csv(angle_file)

            # Feature columns
            exclude = ['id', 'target', 'participant_id', 'participant_1',
                      'participant_2', 'participant_3', 'participant_4', 'participant_5']
            feature_cols = [c for c in df.columns if c not in exclude]

            X = df[feature_cols].values

            # Load all targets
            y = np.zeros((len(df), 3))
            y[:, 0] = df['target'].values

            depth_file = player_dir / "depth.csv"
            if depth_file.exists():
                y[:, 1] = pd.read_csv(depth_file)['target'].values

            lr_file = player_dir / "left_right.csv"
            if lr_file.exists():
                y[:, 2] = pd.read_csv(lr_file)['target'].values

            player_data[player_id] = {
                'X': X,
                'y': y,
                'feature_names': feature_cols,
                'n_samples': len(df)
            }
            print(f"  Player {player_id}: {len(df)} samples, {len(feature_cols)} features")

    return player_data


def run_cv_and_train_final(player_data):
    """
    Run internal 5-fold CV for each player and train final models.
    Returns both CV scores and trained models.
    """
    print("\nRunning per-participant 5-fold CV...")

    all_cv_scores = {t: [] for t in TARGETS}
    all_models = {}  # {(player_id, target): model}
    all_scalers = {}  # {player_id: scaler}

    for player_id, data in sorted(player_data.items()):
        X = data['X']
        y = data['y']
        n_samples = data['n_samples']

        print(f"\n--- Player {player_id} ({n_samples} samples) ---")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[player_id] = scaler

        # 5-fold CV within this player
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        player_cv_scores = {t: [] for t in TARGETS}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for target_idx, target in enumerate(TARGETS):
                model = lgb.LGBMRegressor(**LGBM_PARAMS)
                model.fit(X_train, y_train[:, target_idx])

                pred = model.predict(X_val)
                mse = np.mean((pred - y_val[:, target_idx]) ** 2)
                player_cv_scores[target].append(mse)

        # Print player CV results
        for target in TARGETS:
            mean_mse = np.mean(player_cv_scores[target])
            all_cv_scores[target].append(mean_mse)
            print(f"  {target}: CV MSE = {mean_mse:.6f}")

        # Train final models on all player data
        for target_idx, target in enumerate(TARGETS):
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X_scaled, y[:, target_idx])
            all_models[(player_id, target)] = model

    # Compute overall CV scores (weighted by samples)
    print("\n" + "="*60)
    print("OVERALL CV RESULTS (per-participant internal 5-fold)")
    print("="*60)

    total_samples = sum(d['n_samples'] for d in player_data.values())

    overall_scores = {}
    for target in TARGETS:
        # Weight by number of samples per player
        weighted_mse = sum(
            player_data[pid]['n_samples'] * all_cv_scores[target][i]
            for i, pid in enumerate(sorted(player_data.keys()))
        ) / total_samples
        overall_scores[target] = weighted_mse
        print(f"  {target}: {weighted_mse:.6f}")

    total_scaled_mse = np.mean(list(overall_scores.values()))
    print(f"\n  TOTAL SCALED MSE: {total_scaled_mse:.6f}")

    return all_models, all_scalers, overall_scores, total_scaled_mse


def extract_test_features(feature_names):
    """Extract features from test.csv."""
    print("\nExtracting test features...")

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in test_df.columns if c not in meta_cols]

    # Initialize keypoint mapping
    init_keypoint_mapping(keypoint_cols)

    all_features = []
    test_ids = []
    test_participants = []

    for idx, row in test_df.iterrows():
        # Parse timeseries
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        # Extract hybrid features (same as training)
        features = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        all_features.append(features)
        test_ids.append(row['id'])
        test_participants.append(row['participant_id'])

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)} test shots")

    # Convert to array matching training feature order
    X_test = np.zeros((len(all_features), len(feature_names)), dtype=np.float32)
    for i, feat_dict in enumerate(all_features):
        for j, name in enumerate(feature_names):
            X_test[i, j] = feat_dict.get(name, 0.0)

    print(f"  Test features: {X_test.shape}")
    return X_test, np.array(test_ids), np.array(test_participants)


def predict_test(X_test, test_participants, models, scalers, feature_names):
    """Generate predictions using per-participant models."""
    print("\nGenerating predictions...")

    predictions = np.zeros((len(X_test), 3))

    for i, (x, pid) in enumerate(zip(X_test, test_participants)):
        # Scale using this participant's scaler
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        x_scaled = np.nan_to_num(x_scaled, nan=0.0)

        # Predict using this participant's models
        for target_idx, target in enumerate(TARGETS):
            model = models[(pid, target)]
            predictions[i, target_idx] = model.predict(x_scaled)[0]

    return predictions


def create_submission(test_ids, predictions, submission_num, cv_score):
    """Create submission CSV with scaled predictions."""

    # Load scalers
    print("\nLoading target scalers...")
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Scale predictions to [0, 1]
    print("Scaling predictions to [0, 1] range...")
    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    # Create submission DataFrame
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

    # Stats
    print(f"\nOriginal prediction statistics:")
    for i, col in enumerate(TARGETS):
        print(f"  {col}: mean={predictions[:, i].mean():.4f}, std={predictions[:, i].std():.4f}, "
              f"min={predictions[:, i].min():.4f}, max={predictions[:, i].max():.4f}")

    print(f"\nScaled prediction statistics:")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}, "
              f"min={submission[col].min():.4f}, max={submission[col].max():.4f}")

    return filepath


def main():
    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    print("="*70)
    print(f"REPLICATING 0.0104 CV APPROACH - SUBMISSION {next_num}")
    print("="*70)
    print("\nApproach: Per-participant internal 5-fold CV")
    print(f"Model: LightGBM")
    print(f"Parameters: {LGBM_PARAMS}")
    print("Strategy: 15 models (5 players x 3 targets)")

    # Load training data per player
    player_data = load_train_data_per_player()

    # Get feature names from first player
    feature_names = list(player_data.values())[0]['feature_names']

    # Run CV and train final models
    models, scalers, cv_scores, total_cv_score = run_cv_and_train_final(player_data)

    # Extract test features
    X_test, test_ids, test_participants = extract_test_features(feature_names)

    # Predict
    predictions = predict_test(X_test, test_participants, models, scalers, feature_names)

    # Create submission
    filepath = create_submission(test_ids, predictions, next_num, total_cv_score)

    # Log
    log_entry = f"""
## Submission {next_num}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Approach**: Per-participant internal 5-fold CV (replicating 0.0104 from TEST_PLAN.md)

**Model**: LightGBM

**Parameters**:
- n_estimators: 100
- num_leaves: 10
- learning_rate: 0.05
- reg_alpha: 0.5
- reg_lambda: 0.5

**Strategy**: 15 models (5 players x 3 targets)

**CV Scores**:
- angle: {cv_scores['angle']:.6f}
- depth: {cv_scores['depth']:.6f}
- left_right: {cv_scores['left_right']:.6f}
- **TOTAL: {total_cv_score:.6f}**

**File**: {filepath}

**Leaderboard Score**: TBD
"""

    print("\n" + "="*70)
    print("SUBMISSION DETAILS")
    print("="*70)
    print(log_entry)

    return filepath


if __name__ == "__main__":
    main()
