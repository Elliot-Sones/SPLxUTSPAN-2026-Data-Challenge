"""
Stable Physics Features Submission

Uses simple, generalizable features that don't overfit:
- ANGLE: both_wrists @ frame 150 (alpha=5.0)
- DEPTH: right_wrist + velocity @ frame 140 (alpha=1.0)
- LEFT_RIGHT: both_wrists @ frame 160 (alpha=5.0)
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
SUBMISSION_DIR = PROJECT_DIR / 'submission'

TARGETS = ['angle', 'depth', 'left_right']

TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}

# Stable configs
CONFIGS = {
    'angle': {
        'parts': ['right_wrist', 'left_wrist'],
        'frame': 150,
        'velocity': False,
        'alpha': 5.0
    },
    'depth': {
        'parts': ['right_wrist'],
        'frame': 140,
        'velocity': True,
        'vel_window': 5,
        'alpha': 1.0
    },
    'left_right': {
        'parts': ['right_wrist', 'left_wrist'],
        'frame': 160,
        'velocity': False,
        'alpha': 5.0
    },
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def extract_features(X, col_to_idx, config):
    """Extract features based on config."""
    features = []
    frame = config['frame']

    for part in config['parts']:
        cols = get_cols(col_to_idx, part)
        if cols:
            features.append(X[:, frame, cols])

            if config.get('velocity', False):
                window = config.get('vel_window', 5)
                if frame + window < X.shape[1]:
                    vel = (X[:, frame + window, cols] - X[:, frame, cols]) / window
                    features.append(vel)

    if features:
        return np.hstack(features)
    return None


def load_test_data():
    """Load test data."""
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in test_df.columns if c not in meta_cols]

    n_samples = len(test_df)
    n_frames = 240
    n_keypoints = len(keypoint_cols)

    X_test = np.zeros((n_samples, n_frames, n_keypoints), dtype=np.float32)
    test_ids = []
    test_pids = []

    for idx, row in test_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_test[idx, :, i] = parse_array_string(row[col])
        test_ids.append(row['id'])
        test_pids.append(row['participant_id'])

    return X_test, np.array(test_ids), np.array(test_pids), keypoint_cols


def main():
    print("=" * 70)
    print("STABLE PHYSICS FEATURES SUBMISSION")
    print("=" * 70)

    # Load training data
    X_train, y_train, meta_train = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}
    train_pids = meta_train['participant_id'].values

    # Load test data
    X_test, test_ids, test_pids, _ = load_test_data()
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Store predictions and CV scores
    predictions = np.zeros((len(test_ids), 3))
    cv_scores = {}

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n{target_name.upper()}:")

        config = CONFIGS[target_name]
        y_target = y_train[:, target_idx]

        # Extract features
        train_features = extract_features(X_train, col_to_idx, config)
        test_features = extract_features(X_test, col_to_idx, config)

        print(f"  Config: {config['parts']} @ frame {config['frame']}")
        print(f"  Features: {train_features.shape[1]}")

        # Per-player training
        target_preds = np.zeros(len(test_ids))
        cv_preds = np.zeros(len(y_target))

        for pid in sorted(np.unique(train_pids)):
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_tr = np.nan_to_num(train_features[train_mask], nan=0.0)
            y_tr = y_target[train_mask]

            # CV for this player
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            player_cv_preds = np.zeros(len(y_tr))

            for tr_idx, val_idx in kf.split(X_tr):
                scaler = StandardScaler()
                X_tr_fold = scaler.fit_transform(X_tr[tr_idx])
                X_val_fold = scaler.transform(X_tr[val_idx])

                model = Ridge(alpha=config['alpha'])
                model.fit(X_tr_fold, y_tr[tr_idx])
                player_cv_preds[val_idx] = model.predict(X_val_fold)

            cv_preds[train_mask] = player_cv_preds

            # Train final model on all player data
            if test_mask.sum() > 0:
                X_te = np.nan_to_num(test_features[test_mask], nan=0.0)

                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_te_scaled = scaler.transform(X_te)

                model = Ridge(alpha=config['alpha'])
                model.fit(X_tr_scaled, y_tr)
                target_preds[test_mask] = model.predict(X_te_scaled)

                print(f"  Player {pid}: {train_mask.sum()} train, {test_mask.sum()} test")

        predictions[:, target_idx] = target_preds

        # Compute CV score
        scaler = TARGET_SCALERS[target_name]
        scaled_cv = np.clip(scaler.transform(cv_preds.reshape(-1, 1)).flatten(), 0, 1)
        scaled_truth = np.clip(scaler.transform(y_target.reshape(-1, 1)).flatten(), 0, 1)
        cv_mse = np.mean((scaled_cv - scaled_truth) ** 2)
        cv_std = np.std(scaled_cv)

        cv_scores[target_name] = {'mse': cv_mse, 'std': cv_std}
        print(f"  CV MSE: {cv_mse:.6f}, pred_std: {cv_std:.4f}")

    # Scale predictions to [0, 1]
    print("\nScaling predictions...")
    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()
        scaled_predictions[:, i] = np.clip(scaled_predictions[:, i], 0, 1)

    # Create submission
    submission_df = pd.DataFrame({
        'id': test_ids,
        'angle': scaled_predictions[:, 0],
        'depth': scaled_predictions[:, 1],
        'left_right': scaled_predictions[:, 2],
    })

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    next_num = len(existing) + 1

    submission_path = SUBMISSION_DIR / f"submission{next_num}.csv"
    submission_df.to_csv(submission_path, index=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUBMISSION SAVED: {submission_path}")
    print(f"{'='*70}")

    print("\nCV Scores:")
    total_mse = 0
    for target in TARGETS:
        print(f"  {target}: MSE={cv_scores[target]['mse']:.6f}, std={cv_scores[target]['std']:.4f}")
        total_mse += cv_scores[target]['mse']

    avg_mse = total_mse / 3
    print(f"\n  OVERALL CV MSE: {avg_mse:.6f}")

    print("\nSubmission Statistics:")
    for col in ['angle', 'depth', 'left_right']:
        print(f"  {col}: mean={submission_df[col].mean():.4f}, std={submission_df[col].std():.4f}, "
              f"min={submission_df[col].min():.4f}, max={submission_df[col].max():.4f}")

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  This submission CV:  {avg_mse:.6f}")
    print(f"  Sub9 CV:             0.008773")
    print(f"  Sub9 LB:             0.009109")
    print(f"  Previous physics CV: 0.006372 (but angle_std=0.1822)")
    print(f"  This angle_std:      {cv_scores['angle']['std']:.4f}")
    print(f"  Sub9 angle_std:      0.1390")

    print(f"\n{'='*70}")
    print("SUBMISSION DETAILS")
    print(f"{'='*70}")
    print(f"""
Submission {next_num}: Stable Physics Features

Method: Simple, generalizable features with high regularization
- ANGLE: both_wrists @ frame 150 (Ridge alpha=5.0)
- DEPTH: right_wrist + velocity(w=5) @ frame 140 (Ridge alpha=1.0)
- LEFT_RIGHT: both_wrists @ frame 160 (Ridge alpha=5.0)

Key improvements over previous physics submission:
- Simpler features (6-9 features per target vs 20+)
- Higher regularization (alpha=5 vs alpha=1)
- Lower angle_std ({cv_scores['angle']['std']:.4f} vs 0.1822)
- Should generalize better to test set

File: {submission_path}
""")


if __name__ == "__main__":
    main()
