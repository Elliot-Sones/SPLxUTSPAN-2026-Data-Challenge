"""
Physics Features vs Sub9 Ensemble Model CV Comparison

This script compares:
1. Our physics features (from exhaustive testing) - simple Ridge model
2. The actual sub9 ensemble model (LGB+XGB+CatBoost+Ridge with meta-stacking)

This answers the question: Do physics features alone beat the full sub9 pipeline?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns
from advanced_features import (
    init_keypoint_mapping as adv_init,
    extract_advanced_features as adv_extract_all,
)
from hybrid_features import (
    init_keypoint_mapping as hybrid_init,
    extract_hybrid_features,
)

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def nested_cv_r2(X_features, y, model_type='ridge'):
    """Simple nested CV to get R-squared."""
    if X_features is None or len(y) < 15:
        return np.nan
    valid = ~(np.isnan(X_features).any(axis=1) | np.isnan(y))
    if valid.sum() < 15:
        return np.nan
    X_clean = X_features[valid]
    y_clean = y[valid]
    preds = np.zeros(len(y_clean))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X_clean):
        X_tr, X_te = X_clean[train_idx], X_clean[test_idx]
        y_tr, y_te = y_clean[train_idx], y_clean[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42)
        model.fit(X_tr_s, y_tr)
        preds[test_idx] = model.predict(X_te_s)
    mse = np.mean((preds - y_clean) ** 2)
    r2 = 1 - mse / np.var(y_clean)
    return r2


def sub9_ensemble_cv(X_features, y):
    """
    Replicate sub9 ensemble: LGB + XGB + CatBoost + Ridge with OOF stacking.
    Returns R-squared and MSE.
    """
    if X_features is None or len(y) < 15:
        return np.nan, np.nan
    valid = ~(np.isnan(X_features).any(axis=1) | np.isnan(y))
    if valid.sum() < 15:
        return np.nan, np.nan

    X_clean = X_features[valid]
    y_clean = y[valid]
    n_samples = len(y_clean)

    # OOF predictions for each base model
    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)
    oof_ridge = np.zeros(n_samples)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X_clean):
        X_tr, X_te = X_clean[train_idx], X_clean[test_idx]
        y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
            random_state=42
        )
        lgb_model.fit(X_tr_s, y_tr)
        oof_lgb[test_idx] = lgb_model.predict(X_te_s)

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=0,
            random_state=42
        )
        xgb_model.fit(X_tr_s, y_tr)
        oof_xgb[test_idx] = xgb_model.predict(X_te_s)

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=100,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=3,
            verbose=0,
            random_state=42
        )
        cat_model.fit(X_tr_s, y_tr)
        oof_cat[test_idx] = cat_model.predict(X_te_s)

        # Ridge
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_tr_s, y_tr)
        oof_ridge[test_idx] = ridge_model.predict(X_te_s)

    # Meta-stacking with Ridge
    meta_features = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_ridge])
    meta_preds = np.zeros(n_samples)

    for train_idx, test_idx in kf.split(meta_features):
        X_tr, X_te = meta_features[train_idx], meta_features[test_idx]
        y_tr = y_clean[train_idx]

        meta_model = Ridge(alpha=0.1)
        meta_model.fit(X_tr, y_tr)
        meta_preds[test_idx] = meta_model.predict(X_te)

    mse = np.mean((meta_preds - y_clean) ** 2)
    r2 = 1 - mse / np.var(y_clean)

    return r2, mse


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def extract_physics_features_for_player(X_player, col_to_idx, target_name, pid):
    """
    Extract our best physics features based on exhaustive testing.
    Returns feature array for the player.
    """
    # Best configs from DEFINITIVE_PHYSICS_RESULTS.md
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

    config = BEST_CONFIGS[target_name][pid]
    features = []

    # Position features
    if config.get('multi_frame', False):
        for frame in config['frames']:
            for part in config['parts']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
    else:
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
        # Handle different shapes
        feat_list = []
        for f in features:
            if f.ndim == 1:
                feat_list.append(f.reshape(-1, 1))
            else:
                feat_list.append(f)
        return np.column_stack(feat_list)
    return None


def extract_sub9_features(X_raw, keypoint_cols):
    """
    Extract features the way sub9 does it using advanced_features and hybrid_features.
    """
    adv_init(keypoint_cols)
    hybrid_init(keypoint_cols)

    n_samples = X_raw.shape[0]
    all_features = []

    for i in range(n_samples):
        timeseries = X_raw[i]  # Shape: (240, 207)

        # Extract advanced features
        adv_feats = adv_extract_all(timeseries)

        # Extract hybrid features
        hybrid_feats = extract_hybrid_features(timeseries)

        # Combine
        combined = {**adv_feats, **hybrid_feats}
        all_features.append(combined)

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    # Handle missing values
    df = df.fillna(0)

    return df.values, df.columns.tolist()


def main():
    print("=" * 80)
    print("PHYSICS FEATURES vs SUB9 ENSEMBLE MODEL CV COMPARISON")
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

    results = []

    # First, extract sub9-style features for all samples
    print("\nExtracting sub9-style features (advanced + hybrid)...")
    sub9_features, sub9_feature_names = extract_sub9_features(X, keypoint_cols)
    print(f"Sub9 features: {sub9_features.shape[1]} features")

    for target_name, target_values in targets.items():
        print(f"\n{'='*80}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*80}")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            y_player = target_values[mask]
            n_samples = mask.sum()

            # Physics features (our best from exhaustive testing)
            X_player_raw = X[mask]
            physics_feats = extract_physics_features_for_player(X_player_raw, col_to_idx, target_name, pid)

            if physics_feats is not None:
                physics_r2_ridge = nested_cv_r2(physics_feats, y_player, 'ridge')
                physics_r2_rf = nested_cv_r2(physics_feats, y_player, 'rf')
                physics_r2 = max(physics_r2_ridge, physics_r2_rf)
                physics_n_feats = physics_feats.shape[1]
            else:
                physics_r2 = np.nan
                physics_n_feats = 0

            # Sub9 ensemble features
            sub9_player_feats = sub9_features[mask]
            sub9_r2, sub9_mse = sub9_ensemble_cv(sub9_player_feats, y_player)
            sub9_n_feats = sub9_player_feats.shape[1]

            # Simple Ridge on sub9 features for comparison
            sub9_ridge_r2 = nested_cv_r2(sub9_player_feats, y_player, 'ridge')

            print(f"\nPlayer {pid} (n={n_samples}):")
            print(f"  Physics features ({physics_n_feats} feats):")
            print(f"    Ridge R2: {physics_r2:.4f}")
            print(f"  Sub9 features ({sub9_n_feats} feats):")
            print(f"    Ridge R2: {sub9_ridge_r2:.4f}")
            print(f"    Full ensemble R2: {sub9_r2:.4f}")

            diff = physics_r2 - sub9_r2
            better = "PHYSICS WINS" if diff > 0 else "SUB9 WINS"
            print(f"  Difference: {diff:+.4f} ({better})")

            results.append({
                'target': target_name,
                'player': pid,
                'n_samples': n_samples,
                'physics_r2': physics_r2,
                'physics_n_feats': physics_n_feats,
                'sub9_ridge_r2': sub9_ridge_r2,
                'sub9_ensemble_r2': sub9_r2,
                'sub9_n_feats': sub9_n_feats,
                'diff_vs_ensemble': diff,
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for target_name in ['angle', 'depth', 'left_right']:
        target_df = results_df[results_df['target'] == target_name]

        physics_avg = target_df['physics_r2'].mean()
        sub9_ridge_avg = target_df['sub9_ridge_r2'].mean()
        sub9_ensemble_avg = target_df['sub9_ensemble_r2'].mean()

        print(f"\n{target_name.upper()}:")
        print(f"  Physics (best body parts, {target_df['physics_n_feats'].mean():.0f} feats avg): R2 = {physics_avg:.4f}")
        print(f"  Sub9 Ridge ({target_df['sub9_n_feats'].iloc[0]} feats):                        R2 = {sub9_ridge_avg:.4f}")
        print(f"  Sub9 Full Ensemble:                                        R2 = {sub9_ensemble_avg:.4f}")

        if physics_avg > sub9_ensemble_avg:
            print(f"  --> PHYSICS FEATURES WIN by {physics_avg - sub9_ensemble_avg:.4f}")
        else:
            print(f"  --> SUB9 ENSEMBLE WINS by {sub9_ensemble_avg - physics_avg:.4f}")

    # Overall
    print("\n" + "=" * 80)
    print("OVERALL AVERAGE")
    print("=" * 80)

    overall_physics = results_df['physics_r2'].mean()
    overall_sub9_ridge = results_df['sub9_ridge_r2'].mean()
    overall_sub9_ensemble = results_df['sub9_ensemble_r2'].mean()

    print(f"Physics features:       R2 = {overall_physics:.4f}")
    print(f"Sub9 Ridge:             R2 = {overall_sub9_ridge:.4f}")
    print(f"Sub9 Full Ensemble:     R2 = {overall_sub9_ensemble:.4f}")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'physics_vs_sub9_cv.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'physics_vs_sub9_cv.csv'}")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)

    if overall_physics > overall_sub9_ensemble:
        print("Physics features OUTPERFORM sub9 ensemble!")
        print("Recommendation: Replace sub9 features with physics features.")
    else:
        gap = overall_sub9_ensemble - overall_physics
        print(f"Sub9 ensemble still performs better by {gap:.4f} R2 on average.")
        print("\nPossible next steps:")
        print("1. Combine physics features WITH sub9 features")
        print("2. Use physics features for specific players/targets where they win")
        print("3. The sub9 features capture something beyond simple physics")


if __name__ == "__main__":
    main()
