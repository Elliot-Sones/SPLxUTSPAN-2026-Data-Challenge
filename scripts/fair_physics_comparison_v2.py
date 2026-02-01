"""
Fair Physics vs Sub9 Features Comparison V2

Uses generalized physics features that work for all players.
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


def evaluate_with_ensemble(X, y_target, pids, feature_set_name, target_name):
    """
    Run the exact same ensemble CV as sub9 for a single target.
    """
    print(f"\n  {target_name}: ", end="")

    model_configs = get_model_configs()
    n_models = len(model_configs)
    unique_pids = sorted(np.unique(pids))

    # OOF predictions for each base model
    oof_preds = np.zeros((len(X), n_models))

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask].copy()
        y_player = y_target[pid_mask]

        # Remove constant and nan features
        non_constant_mask = np.std(X_player, axis=0) > 1e-10
        nan_mask = ~np.any(np.isnan(X_player), axis=0)
        valid_mask = non_constant_mask & nan_mask

        if valid_mask.sum() == 0:
            # No valid features - use mean prediction
            player_indices = np.where(pid_mask)[0]
            for model_idx in range(n_models):
                oof_preds[player_indices, model_idx] = np.mean(y_player)
            continue

        X_player = X_player[:, valid_mask]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # 5-fold CV within player
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_indices = np.where(pid_mask)[0]

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_player[train_idx], y_player[val_idx]

            for model_idx, (model_name, config) in enumerate(model_configs.items()):
                try:
                    model = config["class"](**config["params"])
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                except Exception:
                    pred = np.full(len(X_val), np.mean(y_train))
                global_val_idx = player_indices[val_idx]
                oof_preds[global_val_idx, model_idx] = pred

    # Meta-stacking with CV
    meta_preds = np.zeros(len(y_target))
    kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf_meta.split(oof_preds):
        X_train, X_val = oof_preds[train_idx], oof_preds[val_idx]
        y_train = y_target[train_idx]

        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(X_train, y_train)
        meta_preds[val_idx] = meta_model.predict(X_val)

    mse = np.mean((meta_preds - y_target) ** 2)
    print(f"MSE = {mse:.6f}")

    return mse


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def extract_unified_physics_features(X_raw, col_to_idx):
    """
    Extract physics features that work for all players.
    Based on the best findings from exhaustive testing.
    """
    n_samples = X_raw.shape[0]

    # Key body parts and frames from research
    key_parts = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'right_hip', 'left_hip',
        'right_knee', 'right_ankle',
        'right_second_finger_distal', 'right_second_finger_mcp',
    ]

    key_frames = [95, 105, 110, 130, 135, 145, 150, 155, 160, 165, 175]
    velocity_windows = [2, 3, 5, 10]

    all_features = []

    for i in range(n_samples):
        X_sample = X_raw[i]  # Shape: (240, n_keypoints)
        sample_feats = {}

        # Position features at key frames
        for part in key_parts:
            cols = get_cols(col_to_idx, part)
            if not cols:
                continue

            for frame in key_frames:
                for ci, c in enumerate(cols):
                    axis = ['x', 'y', 'z'][ci]
                    sample_feats[f"{part}_{axis}_f{frame}"] = X_sample[frame, c]

        # Velocity features
        for part in ['right_wrist', 'right_elbow', 'left_wrist', 'right_knee']:
            cols = get_cols(col_to_idx, part)
            if not cols:
                continue

            for frame in [110, 150, 160]:
                for window in velocity_windows:
                    if frame + window >= X_sample.shape[0]:
                        continue
                    for ci, c in enumerate(cols):
                        axis = ['x', 'y', 'z'][ci]
                        vel = (X_sample[frame + window, c] - X_sample[frame, c]) / window
                        sample_feats[f"{part}_{axis}_vel_w{window}_f{frame}"] = vel

        # LR asymmetry features
        lr_pairs = [
            ('right_wrist', 'left_wrist'),
            ('right_shoulder', 'left_shoulder'),
            ('right_hip', 'left_hip'),
            ('right_elbow', 'left_elbow'),
        ]
        for frame in [150, 160, 175]:
            for right, left in lr_pairs:
                for axis in ['x', 'y', 'z']:
                    r_col = f"{right}_{axis}"
                    l_col = f"{left}_{axis}"
                    if r_col in col_to_idx and l_col in col_to_idx:
                        diff = X_sample[frame, col_to_idx[r_col]] - X_sample[frame, col_to_idx[l_col]]
                        sample_feats[f"{right}_{left}_{axis}_diff_f{frame}"] = diff

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
    print("\nExtracting unified physics features...")
    X_physics, physics_feat_names = extract_unified_physics_features(X_raw, col_to_idx)
    print(f"Physics features: {X_physics.shape}")

    # Load sub9 features
    print("\nLoading sub9 features...")
    X_sub9, y_sub9, pids_sub9, sub9_feat_names = load_sub9_features()
    print(f"Sub9 features: {X_sub9.shape}")

    # Make sure y matches
    y = np.column_stack([y_raw[:, 0], y_raw[:, 1], y_raw[:, 2]])

    # Evaluate both
    results = {'physics': {}, 'sub9': {}}

    print("\n" + "=" * 70)
    print("PHYSICS FEATURES EVALUATION")
    print("=" * 70)
    for target_idx, target in enumerate(TARGETS):
        results['physics'][target] = evaluate_with_ensemble(
            X_physics, y[:, target_idx], pids, "PHYSICS", target
        )

    print("\n" + "=" * 70)
    print("SUB9 FEATURES EVALUATION")
    print("=" * 70)
    for target_idx, target in enumerate(TARGETS):
        results['sub9'][target] = evaluate_with_ensemble(
            X_sub9, y_sub9[:, target_idx], pids_sub9, "SUB9", target
        )

    # Summary comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(f"\n{'Target':<12} {'Physics':<12} {'Sub9':<12} {'Winner':<15} {'Diff':<10}")
    print("-" * 60)

    total_physics = 0
    total_sub9 = 0

    for target in TARGETS:
        physics_mse = results['physics'][target]
        sub9_mse = results['sub9'][target]
        diff = physics_mse - sub9_mse
        winner = "PHYSICS" if diff < 0 else "SUB9"
        print(f"{target:<12} {physics_mse:.6f}     {sub9_mse:.6f}     {winner:<15} {abs(diff):.6f}")
        total_physics += physics_mse
        total_sub9 += sub9_mse

    print("-" * 60)
    physics_overall = total_physics / 3
    sub9_overall = total_sub9 / 3
    diff = physics_overall - sub9_overall
    winner = "PHYSICS" if diff < 0 else "SUB9"
    print(f"{'OVERALL':<12} {physics_overall:.6f}     {sub9_overall:.6f}     {winner:<15} {abs(diff):.6f}")

    # Save comparison
    comparison = []
    for target in TARGETS + ['overall']:
        if target == 'overall':
            physics_mse = physics_overall
            sub9_mse = sub9_overall
        else:
            physics_mse = results['physics'][target]
            sub9_mse = results['sub9'][target]

        comparison.append({
            'target': target,
            'physics_mse': physics_mse,
            'sub9_mse': sub9_mse,
            'diff': physics_mse - sub9_mse,
            'winner': 'physics' if physics_mse < sub9_mse else 'sub9',
            'physics_n_feats': X_physics.shape[1],
            'sub9_n_feats': X_sub9.shape[1],
        })

    pd.DataFrame(comparison).to_csv(OUTPUT_DIR / 'fair_physics_comparison.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'fair_physics_comparison.csv'}")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    print(f"\nPhysics features: {X_physics.shape[1]} features")
    print(f"Sub9 features: {X_sub9.shape[1]} features")

    if physics_overall < sub9_overall:
        improvement = (sub9_overall - physics_overall) / sub9_overall * 100
        print(f"\nPhysics features achieve {improvement:.1f}% lower MSE than sub9!")
        print("With only {:.1f}% of the features ({} vs {})".format(
            X_physics.shape[1] / X_sub9.shape[1] * 100,
            X_physics.shape[1],
            X_sub9.shape[1]
        ))
    else:
        gap = (physics_overall - sub9_overall) / sub9_overall * 100
        print(f"\nSub9 features achieve {gap:.1f}% lower MSE than physics features.")
        print("\nPossible next steps:")
        print("1. Combine physics features WITH sub9 features")
        print("2. Use physics features for targets where they win")
        print("3. Add more physics-based features")


if __name__ == "__main__":
    main()
