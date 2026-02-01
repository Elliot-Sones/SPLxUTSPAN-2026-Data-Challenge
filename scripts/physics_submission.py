"""
Physics Features Submission

Creates a submission using the best physics features for each player-target combination.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
SUBMISSION_DIR = PROJECT_DIR / 'submission'
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ['angle', 'depth', 'left_right']

# Load target scalers
TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}

# Best configs from DEFINITIVE_PHYSICS_RESULTS.md
BEST_CONFIGS = {
    'angle': {
        1: {'parts': ['right_second_finger_distal', 'right_third_finger_distal', 'left_wrist'],
            'frame': 165, 'add_velocity': True, 'window': 3, 'add_wrist_snap': True},
        2: {'parts': ['left_knee'], 'frame': 110, 'add_velocity': True, 'window': 5, 'add_pos': True},
        3: {'parts': ['left_fourth_finger_distal'], 'frame': 105, 'add_velocity': True, 'window': 5},
        4: {'parts': ['right_shoulder', 'right_knee'], 'frame': 160, 'add_velocity': False},
        5: {'parts': ['right_wrist', 'left_wrist', 'right_hip'], 'frame': 155, 'add_lr_asymmetry': True},
    },
    'depth': {
        1: {'parts': ['right_second_finger_distal'], 'frames': [150, 155, 160], 'multi_frame': True},
        2: {'parts': ['right_elbow'], 'frame': 95, 'add_velocity': True, 'window': 10},
        3: {'parts': ['left_ankle'], 'frame': 130, 'add_velocity': True, 'window': 2},
        4: {'parts': ['right_wrist', 'right_elbow', 'right_shoulder'], 'frame': 135, 'add_velocity': True, 'window': 5},
        5: {'parts': ['right_second_finger_mcp'], 'frame': 145, 'add_velocity': False},
    },
    'left_right': {
        1: {'parts': ['left_wrist', 'right_wrist'], 'frame': 175, 'add_velocity': True, 'window': 5, 'add_guide_hand': True},
        2: {'parts': ['right_wrist', 'left_wrist', 'right_hip'], 'frame': 150, 'add_lr_asymmetry': True},
        3: {'parts': ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow', 'right_hip', 'left_hip'],
            'frame': 175, 'add_lr_asymmetry': True},
        4: {'parts': ['left_elbow'], 'frame': 150, 'add_velocity': True, 'window': 10},
        5: {'parts': ['right_ear'], 'frame': 175, 'add_velocity': True, 'window': 2},
    },
}

# Best model type per player-target from CV results
BEST_MODELS = {
    'angle': {1: 'ridge', 2: 'ridge', 3: 'ridge', 4: 'ridge', 5: 'ridge'},
    'depth': {1: 'ridge', 2: 'ridge', 3: 'rf', 4: 'rf', 5: 'ridge'},
    'left_right': {1: 'lgb', 2: 'ridge', 3: 'lgb', 4: 'ridge', 5: 'rf'},
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def extract_features_for_config(X_data, col_to_idx, config):
    """Extract features based on config."""
    features = []

    # Multi-frame position features
    if config.get('multi_frame', False):
        for frame in config['frames']:
            for part in config['parts']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_data[:, frame, cols])
    else:
        # Single frame position features
        frame = config['frame']
        for part in config['parts']:
            cols = get_cols(col_to_idx, part)
            if cols:
                features.append(X_data[:, frame, cols])

    # Add position explicitly if requested
    if config.get('add_pos', False):
        frame = config['frame']
        for part in config['parts']:
            cols = get_cols(col_to_idx, part)
            if cols:
                features.append(X_data[:, frame, cols])

    # Velocity features
    if config.get('add_velocity', False):
        frame = config['frame']
        window = config.get('window', 5)
        for part in config['parts'][:2]:
            cols = get_cols(col_to_idx, part)
            if cols and frame + window < X_data.shape[1]:
                vel = (X_data[:, frame + window, cols] - X_data[:, frame, cols]) / window
                features.append(vel)

    # Wrist snap
    if config.get('add_wrist_snap', False):
        frame = config['frame']
        window = config.get('window', 3)
        wrist_cols = get_cols(col_to_idx, 'right_wrist')
        elbow_cols = get_cols(col_to_idx, 'right_elbow')
        if wrist_cols and elbow_cols and frame + window < X_data.shape[1]:
            wrist_vel = (X_data[:, frame + window, wrist_cols] - X_data[:, frame, wrist_cols]) / window
            elbow_vel = (X_data[:, frame + window, elbow_cols] - X_data[:, frame, elbow_cols]) / window
            snap = wrist_vel - elbow_vel
            features.append(snap)

    # Guide hand
    if config.get('add_guide_hand', False):
        frame = config['frame']
        cols = get_cols(col_to_idx, 'left_wrist')
        if cols:
            features.append(X_data[:, frame, cols])

    # Left-right asymmetry
    if config.get('add_lr_asymmetry', False):
        frame = config['frame']
        lr_pairs = [
            ('right_wrist', 'left_wrist'),
            ('right_shoulder', 'left_shoulder'),
            ('right_hip', 'left_hip'),
            ('right_elbow', 'left_elbow'),
        ]
        for right, left in lr_pairs:
            for axis in ['x', 'y', 'z']:
                r_col = f"{right}_{axis}"
                l_col = f"{left}_{axis}"
                if r_col in col_to_idx and l_col in col_to_idx:
                    diff = X_data[:, frame, col_to_idx[r_col]] - X_data[:, frame, col_to_idx[l_col]]
                    features.append(diff.reshape(-1, 1))

    if features:
        feat_list = []
        for f in features:
            if f.ndim == 1:
                feat_list.append(f.reshape(-1, 1))
            else:
                feat_list.append(f)
        return np.column_stack(feat_list)
    return None


def get_model(model_type):
    if model_type == 'ridge':
        return Ridge(alpha=1.0)
    elif model_type == 'rf':
        return RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42)
    else:  # lgb
        return lgb.LGBMRegressor(n_estimators=100, num_leaves=10, learning_rate=0.05,
                                  reg_alpha=0.5, reg_lambda=0.5, verbose=-1, random_state=42)


def load_test_data():
    """Load test data."""
    print("Loading test data...")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in test_df.columns if c not in meta_cols]

    print(f"Processing {len(test_df)} test shots...")

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
    print("=" * 80)
    print("PHYSICS FEATURES SUBMISSION")
    print("=" * 80)

    # Load training data
    X_train, y_train, meta_train = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}
    train_pids = meta_train['participant_id'].values

    targets = {
        'angle': y_train[:, 0],
        'depth': y_train[:, 1],
        'left_right': y_train[:, 2]
    }

    # Load test data
    X_test, test_ids, test_pids, _ = load_test_data()

    # Store predictions
    predictions = np.zeros((len(test_ids), 3))

    print("\nTraining and predicting...")

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n{target_name.upper()}:")
        target_values = targets[target_name]

        for pid in sorted(np.unique(train_pids)):
            # Training data for this player
            train_mask = train_pids == pid
            X_train_player = X_train[train_mask]
            y_train_player = target_values[train_mask]

            # Test data for this player
            test_mask = test_pids == pid
            X_test_player = X_test[test_mask]

            if test_mask.sum() == 0:
                continue

            # Get config and extract features
            config = BEST_CONFIGS[target_name][pid]
            train_features = extract_features_for_config(X_train_player, col_to_idx, config)
            test_features = extract_features_for_config(X_test_player, col_to_idx, config)

            if train_features is None or test_features is None:
                print(f"  Player {pid}: Feature extraction failed, using mean")
                predictions[test_mask, target_idx] = np.mean(y_train_player)
                continue

            # Handle NaN
            train_features = np.nan_to_num(train_features, nan=0.0)
            test_features = np.nan_to_num(test_features, nan=0.0)

            # Scale features
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)

            # Train model
            model_type = BEST_MODELS[target_name][pid]
            model = get_model(model_type)
            model.fit(train_features_scaled, y_train_player)

            # Predict
            preds = model.predict(test_features_scaled)
            predictions[test_mask, target_idx] = preds

            print(f"  Player {pid}: {model_type}, {train_features.shape[1]} features, {test_mask.sum()} test samples")

    # Scale predictions to [0, 1]
    print("\nScaling predictions to [0, 1]...")
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

    print(f"\n{'='*80}")
    print(f"SUBMISSION SAVED: {submission_path}")
    print(f"{'='*80}")

    print("\nSubmission statistics:")
    for col in ['angle', 'depth', 'left_right']:
        print(f"  {col}: mean={submission_df[col].mean():.4f}, std={submission_df[col].std():.4f}, "
              f"min={submission_df[col].min():.4f}, max={submission_df[col].max():.4f}")

    print(f"\n{'='*80}")
    print("SUBMISSION DETAILS")
    print(f"{'='*80}")
    print(f"""
Submission {next_num}: Physics Features Model

Method: Per-player, per-target, per-frame optimal physics features
- Features derived from exhaustive testing of all 69 body parts
- Optimal frames identified per player-target combination
- Best model type (Ridge/RF/LGB) selected per player-target based on CV

CV Score: 0.006372 (vs Sub9 CV: 0.008773, Sub9 LB: 0.009109)

Feature configs:
ANGLE:
  P1: fingertip + wrist snap @ frame 165 (18 feats, ridge)
  P2: left_knee pos+vel @ frame 110 (9 feats, ridge)
  P3: left_fourth_finger_distal vel @ frame 105 (6 feats, ridge)
  P4: right_shoulder + right_knee @ frame 160 (6 feats, ridge)
  P5: wrists + hip + LR asymmetry @ frame 155 (21 feats, ridge)

DEPTH:
  P1: fingertip multi-frame @ 150-160 (9 feats, ridge)
  P2: right_elbow vel w10 @ frame 95 (6 feats, ridge)
  P3: left_ankle vel w2 @ frame 130 (6 feats, rf)
  P4: arm chain + vel @ frame 135 (15 feats, rf)
  P5: right_second_finger_mcp @ frame 145 (3 feats, ridge)

LEFT_RIGHT:
  P1: wrists + guide hand @ frame 175 (15 feats, lgb)
  P2: wrists + hip + LR asymmetry @ frame 150 (21 feats, ridge)
  P3: body alignment (shoulders, elbows, hips) @ frame 175 (30 feats, lgb)
  P4: left_elbow vel w10 @ frame 150 (6 feats, ridge)
  P5: right_ear vel w2 @ frame 175 (6 feats, rf)

File: {submission_path}
""")


if __name__ == "__main__":
    main()
