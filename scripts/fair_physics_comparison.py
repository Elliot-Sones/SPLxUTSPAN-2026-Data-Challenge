"""
Fair Physics vs Sub9 Features Comparison

Uses IDENTICAL methodology for both:
- Same 5-fold CV per player
- Same model ensemble (LGB + XGB + CatBoost + Ridge)
- Same meta-stacking

This gives us a true apples-to-apples comparison.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

TARGETS = ['angle', 'depth', 'left_right']


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    arr = np.array(json.loads(s), dtype=np.float32)
    return arr


def get_model_configs():
    """Same configs as sub9."""
    return {
        "lgb": {
            "class": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 100,
                "num_leaves": 10,
                "learning_rate": 0.05,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
        },
        "xgb": {
            "class": xgb.XGBRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
            }
        },
        "ridge": {
            "class": Ridge,
            "params": {"alpha": 1.0, "random_state": 42}
        },
        "catboost": {
            "class": CatBoostRegressor,
            "params": {
                "iterations": 100,
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "random_state": 42,
                "verbose": False,
            }
        },
    }


def evaluate_with_ensemble(X, y, pids, feature_set_name):
    """
    Run the exact same ensemble CV as sub9.
    Returns per-target MSE and overall MSE.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {feature_set_name}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"{'='*70}")

    model_configs = get_model_configs()
    n_models = len(model_configs)
    unique_pids = sorted(np.unique(pids))

    # OOF predictions for each base model
    oof_preds = {target: np.zeros((len(X), n_models)) for target in TARGETS}

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        n_samples = len(X_player)

        # Remove constant features for this player
        non_constant_mask = np.std(X_player, axis=0) > 1e-10
        X_player_filtered = X_player[:, non_constant_mask]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player_filtered)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # 5-fold CV within player
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(pid_mask)[0]

        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_target[train_idx], y_target[val_idx]

                for model_idx, (model_name, config) in enumerate(model_configs.items()):
                    try:
                        model = config["class"](**config["params"])
                        model.fit(X_train, y_train)
                        pred = model.predict(X_val)
                    except Exception as e:
                        # Fallback to mean prediction if model fails (e.g., constant features)
                        pred = np.full(len(X_val), np.mean(y_train))
                    global_val_idx = player_indices[val_idx]
                    oof_preds[target][global_val_idx, model_idx] = pred

    # Evaluate base models
    print("\nBase model OOF MSE:")
    base_mses = {}
    for target in TARGETS:
        target_idx = TARGETS.index(target)
        y_true = y[:, target_idx]
        base_mses[target] = {}

        for model_idx, model_name in enumerate(model_configs.keys()):
            preds = oof_preds[target][:, model_idx]
            mse = np.mean((preds - y_true) ** 2)
            base_mses[target][model_name] = mse

        # Ensemble mean
        mean_pred = np.mean(oof_preds[target], axis=1)
        mse_mean = np.mean((mean_pred - y_true) ** 2)
        base_mses[target]['ensemble_mean'] = mse_mean

    # Train meta-model with CV
    print("\nMeta-model (stacking) OOF MSE:")
    meta_mses = {}
    overall_mse = 0.0

    for target_idx, target in enumerate(TARGETS):
        X_meta = oof_preds[target]
        y_true = y[:, target_idx]

        # 5-fold CV for meta-model
        meta_preds = np.zeros(len(y_true))
        kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf_meta.split(X_meta):
            X_train, X_val = X_meta[train_idx], X_meta[val_idx]
            y_train = y_true[train_idx]

            meta_model = Ridge(alpha=1.0, random_state=42)
            meta_model.fit(X_train, y_train)
            meta_preds[val_idx] = meta_model.predict(X_val)

        mse = np.mean((meta_preds - y_true) ** 2)
        meta_mses[target] = mse
        overall_mse += mse
        print(f"  {target}: {mse:.6f}")

    overall_mse /= 3
    print(f"\n  OVERALL MSE (avg of 3 targets): {overall_mse:.6f}")

    return {
        'base_mses': base_mses,
        'meta_mses': meta_mses,
        'overall_mse': overall_mse
    }


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def extract_physics_features(X_raw, pids, col_to_idx):
    """
    Extract physics features (per-sample) using per-player-per-target configs.
    """
    BEST_CONFIGS = {
        'angle': {
            1: {'parts': ['right_second_finger_distal', 'left_wrist'], 'frame': 165, 'add_velocity': True, 'window': 3},
            2: {'parts': ['left_knee'], 'frame': 110, 'add_velocity': True, 'window': 5},
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
            1: {'parts': ['left_wrist', 'right_wrist'], 'frame': 175, 'add_velocity': True, 'window': 5},
            2: {'parts': ['right_wrist', 'left_wrist', 'right_hip'], 'frame': 150, 'add_lr_asymmetry': True},
            3: {'parts': ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow'], 'frame': 175, 'add_lr_asymmetry': True},
            4: {'parts': ['left_elbow'], 'frame': 150, 'add_velocity': True, 'window': 10},
            5: {'parts': ['right_ear'], 'frame': 175, 'add_velocity': True, 'window': 2},
        },
    }

    # Get max feature count across all configs
    max_feats = 50  # Generous upper bound

    # Create unified feature matrix
    n_samples = X_raw.shape[0]
    all_features = []

    for i in range(n_samples):
        pid = pids[i]
        X_sample = X_raw[i]  # Shape: (240, 207)
        sample_feats = {}

        for target in TARGETS:
            config = BEST_CONFIGS[target][pid]
            prefix = f"{target}_"

            # Position features
            if config.get('multi_frame', False):
                for fi, frame in enumerate(config['frames']):
                    for part in config['parts']:
                        cols = get_cols(col_to_idx, part)
                        if cols:
                            for ci, c in enumerate(cols):
                                sample_feats[f"{prefix}{part}_f{frame}_{ci}"] = X_sample[frame, c]
            else:
                frame = config['frame']
                for part in config['parts']:
                    cols = get_cols(col_to_idx, part)
                    if cols:
                        for ci, c in enumerate(cols):
                            sample_feats[f"{prefix}{part}_pos_{ci}"] = X_sample[frame, c]

            # Velocity features
            if config.get('add_velocity', False):
                frame = config['frame']
                window = config.get('window', 5)
                for part in config['parts'][:2]:
                    cols = get_cols(col_to_idx, part)
                    if cols and frame + window < X_sample.shape[0]:
                        for ci, c in enumerate(cols):
                            vel = (X_sample[frame + window, c] - X_sample[frame, c]) / window
                            sample_feats[f"{prefix}{part}_vel_{ci}"] = vel

            # LR asymmetry
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
                            diff = X_sample[frame, col_to_idx[r_col]] - X_sample[frame, col_to_idx[l_col]]
                            sample_feats[f"{prefix}{right}_{left}_{axis}_diff"] = diff

        all_features.append(sample_feats)

    # Convert to array
    feature_names = sorted(all_features[0].keys())
    X_physics = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)

    return X_physics, feature_names


def load_sub9_features():
    """Load features exactly as sub9 does."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    all_features = []
    all_targets = []
    all_pids = []

    for idx, row in train_df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)
        all_targets.append([row['angle'], row['depth'], row['left_right']])
        all_pids.append(row['participant_id'])

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    return X, y, pids, feature_names


def main():
    print("=" * 80)
    print("FAIR COMPARISON: PHYSICS FEATURES vs SUB9 FEATURES")
    print("Using identical CV methodology for both")
    print("=" * 80)

    # Load raw data for physics features
    X_raw, y_raw, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}
    pids = meta['participant_id'].values

    # Extract physics features
    print("\nExtracting physics features...")
    X_physics, physics_feat_names = extract_physics_features(X_raw, pids, col_to_idx)
    print(f"Physics features: {X_physics.shape}")

    # Load sub9 features
    print("\nLoading sub9 features...")
    X_sub9, y_sub9, pids_sub9, sub9_feat_names = load_sub9_features()
    print(f"Sub9 features: {X_sub9.shape}")

    # Make sure y matches
    y = np.column_stack([y_raw[:, 0], y_raw[:, 1], y_raw[:, 2]])

    # Evaluate both
    results = {}

    results['physics'] = evaluate_with_ensemble(X_physics, y, pids, "PHYSICS FEATURES")
    results['sub9'] = evaluate_with_ensemble(X_sub9, y_sub9, pids_sub9, "SUB9 FEATURES")

    # Summary comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print("\nPer-target MSE (meta-stacking):")
    print(f"{'Target':<12} {'Physics':<12} {'Sub9':<12} {'Winner':<15} {'Diff':<10}")
    print("-" * 60)

    for target in TARGETS:
        physics_mse = results['physics']['meta_mses'][target]
        sub9_mse = results['sub9']['meta_mses'][target]
        diff = physics_mse - sub9_mse
        winner = "PHYSICS" if diff < 0 else "SUB9"
        print(f"{target:<12} {physics_mse:.6f}     {sub9_mse:.6f}     {winner:<15} {abs(diff):.6f}")

    print("-" * 60)
    physics_overall = results['physics']['overall_mse']
    sub9_overall = results['sub9']['overall_mse']
    diff = physics_overall - sub9_overall
    winner = "PHYSICS" if diff < 0 else "SUB9"
    print(f"{'OVERALL':<12} {physics_overall:.6f}     {sub9_overall:.6f}     {winner:<15} {abs(diff):.6f}")

    # Save comparison
    comparison = []
    for target in TARGETS + ['overall']:
        if target == 'overall':
            physics_mse = results['physics']['overall_mse']
            sub9_mse = results['sub9']['overall_mse']
        else:
            physics_mse = results['physics']['meta_mses'][target]
            sub9_mse = results['sub9']['meta_mses'][target]

        comparison.append({
            'target': target,
            'physics_mse': physics_mse,
            'sub9_mse': sub9_mse,
            'diff': physics_mse - sub9_mse,
            'winner': 'physics' if physics_mse < sub9_mse else 'sub9'
        })

    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / 'fair_physics_comparison.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'fair_physics_comparison.csv'}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if physics_overall < sub9_overall:
        improvement = (sub9_overall - physics_overall) / sub9_overall * 100
        print(f"Physics features achieve {improvement:.1f}% lower MSE than sub9!")
        print("Recommendation: Use physics features instead of sub9 features")
    else:
        gap = (physics_overall - sub9_overall) / sub9_overall * 100
        print(f"Sub9 features achieve {gap:.1f}% lower MSE than physics features.")
        print("\nNext steps:")
        print("1. Combine physics features WITH sub9 features")
        print("2. Use physics features for targets where they win")


if __name__ == "__main__":
    main()
