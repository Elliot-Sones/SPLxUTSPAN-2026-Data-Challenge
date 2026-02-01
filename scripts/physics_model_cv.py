"""
Physics Model CV Score

Uses the EXACT best physics features for each player-target combination
from our exhaustive testing. Compares CV score to sub9 LB score (0.009109).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Load target scalers (used to scale predictions to [0,1] for LB comparison)
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


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def extract_features_for_config(X_player, col_to_idx, config):
    """Extract features based on config for a player."""
    features = []

    # Multi-frame position features
    if config.get('multi_frame', False):
        for frame in config['frames']:
            for part in config['parts']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
    else:
        # Single frame position features
        frame = config['frame']
        for part in config['parts']:
            cols = get_cols(col_to_idx, part)
            if cols:
                features.append(X_player[:, frame, cols])

    # Add position explicitly if requested
    if config.get('add_pos', False):
        frame = config['frame']
        for part in config['parts']:
            cols = get_cols(col_to_idx, part)
            if cols:
                features.append(X_player[:, frame, cols])

    # Velocity features
    if config.get('add_velocity', False):
        frame = config['frame']
        window = config.get('window', 5)
        for part in config['parts'][:2]:  # First two parts
            cols = get_cols(col_to_idx, part)
            if cols and frame + window < X_player.shape[1]:
                vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                features.append(vel)

    # Wrist snap (wrist_vel - elbow_vel)
    if config.get('add_wrist_snap', False):
        frame = config['frame']
        window = config.get('window', 3)
        wrist_cols = get_cols(col_to_idx, 'right_wrist')
        elbow_cols = get_cols(col_to_idx, 'right_elbow')
        if wrist_cols and elbow_cols and frame + window < X_player.shape[1]:
            wrist_vel = (X_player[:, frame + window, wrist_cols] - X_player[:, frame, wrist_cols]) / window
            elbow_vel = (X_player[:, frame + window, elbow_cols] - X_player[:, frame, elbow_cols]) / window
            snap = wrist_vel - elbow_vel
            features.append(snap)

    # Guide hand
    if config.get('add_guide_hand', False):
        frame = config['frame']
        cols = get_cols(col_to_idx, 'left_wrist')
        if cols:
            features.append(X_player[:, frame, cols])

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
                    diff = X_player[:, frame, col_to_idx[r_col]] - X_player[:, frame, col_to_idx[l_col]]
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


def train_and_evaluate_cv(X, y, model_type='ridge'):
    """Train model with 5-fold CV and return OOF predictions."""
    n = len(y)
    preds = np.zeros(n)

    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    if valid.sum() < 10:
        return np.full(n, np.nan)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X[valid]):
        # Map back to original indices
        valid_indices = np.where(valid)[0]
        train_orig = valid_indices[train_idx]
        val_orig = valid_indices[val_idx]

        X_train, X_val = X[train_orig], X[val_orig]
        y_train = y[train_orig]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42)
        else:  # lgb
            model = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, learning_rate=0.05,
                                       reg_alpha=0.5, reg_lambda=0.5, verbose=-1, random_state=42)

        model.fit(X_train_s, y_train)
        preds[val_orig] = model.predict(X_val_s)

    # Set invalid samples to mean
    preds[~valid] = np.nanmean(y)

    return preds


def main():
    print("=" * 80)
    print("PHYSICS MODEL CV SCORE")
    print("Using per-player, per-target, per-frame optimal features")
    print("=" * 80)

    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}
    participant_ids = meta['participant_id'].values

    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2]
    }

    # Store all OOF predictions
    all_oof_preds = {target: np.zeros(len(y)) for target in targets}

    results = []

    for target_name, target_values in targets.items():
        print(f"\n{'='*70}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*70}")

        target_preds = np.zeros(len(target_values))

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]
            n_samples = mask.sum()

            config = BEST_CONFIGS[target_name][pid]
            features = extract_features_for_config(X_player, col_to_idx, config)

            if features is None:
                print(f"  Player {pid}: Feature extraction failed")
                target_preds[mask] = np.mean(y_player)
                continue

            # Try multiple models and pick best
            best_mse = float('inf')
            best_preds = None
            best_model = None

            for model_type in ['ridge', 'rf', 'lgb']:
                preds = train_and_evaluate_cv(features, y_player, model_type)
                mse = np.mean((preds - y_player) ** 2)

                if mse < best_mse:
                    best_mse = mse
                    best_preds = preds
                    best_model = model_type

            target_preds[mask] = best_preds

            print(f"  Player {pid} (n={n_samples}): MSE={best_mse:.6f}, R2={1 - best_mse/np.var(y_player):.3f} ({best_model}, {features.shape[1]} feats)")

            results.append({
                'target': target_name,
                'player': pid,
                'mse': best_mse,
                'r2': 1 - best_mse/np.var(y_player),
                'model': best_model,
                'n_features': features.shape[1],
                'n_samples': n_samples,
            })

        all_oof_preds[target_name] = target_preds

        # Target-level MSE
        target_mse = np.mean((target_preds - target_values) ** 2)
        print(f"\n  {target_name.upper()} OVERALL MSE: {target_mse:.6f}")

    # Calculate overall MSE in SCALED [0,1] space (same as LB metric)
    print("\n" + "=" * 80)
    print("FINAL CV SCORE (comparable to LB - scaled to [0,1])")
    print("=" * 80)

    total_mse = 0
    for target_name, target_values in targets.items():
        # Scale both predictions and ground truth to [0,1]
        scaler = TARGET_SCALERS[target_name]
        scaled_preds = scaler.transform(all_oof_preds[target_name].reshape(-1, 1)).flatten()
        scaled_truth = scaler.transform(target_values.reshape(-1, 1)).flatten()

        # Clip to [0,1] as done in submissions
        scaled_preds = np.clip(scaled_preds, 0, 1)
        scaled_truth = np.clip(scaled_truth, 0, 1)

        target_mse = np.mean((scaled_preds - scaled_truth) ** 2)
        print(f"  {target_name}: MSE = {target_mse:.6f}")
        total_mse += target_mse

    avg_mse = total_mse / 3
    print(f"\n  OVERALL CV MSE (avg of 3 targets): {avg_mse:.6f}")

    print("\n" + "=" * 80)
    print("COMPARISON TO SUB9")
    print("=" * 80)

    sub9_lb = 0.009109
    print(f"  Sub9 LB Score:     {sub9_lb:.6f}")
    print(f"  Physics CV Score:  {avg_mse:.6f}")

    if avg_mse < sub9_lb:
        improvement = (sub9_lb - avg_mse) / sub9_lb * 100
        print(f"\n  PHYSICS WINS by {improvement:.1f}% ({sub9_lb - avg_mse:.6f} lower MSE)")
    else:
        gap = (avg_mse - sub9_lb) / sub9_lb * 100
        print(f"\n  SUB9 WINS by {gap:.1f}% ({avg_mse - sub9_lb:.6f} lower MSE)")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'physics_model_cv.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'physics_model_cv.csv'}")


if __name__ == "__main__":
    main()
