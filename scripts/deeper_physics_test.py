"""
DEEPER PHYSICS TEST

The linear models only explain 17-39% of variance.
If physics determines the shot, there should be more signal.

Testing:
1. Velocity in all axes (X, Y, Z) - not just position
2. Acceleration and jerk
3. Nonlinear models (Random Forest)
4. More comprehensive feature combinations
5. Different time windows for velocity

Goal: Find the MAXIMUM explainable variance with physics features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def nested_cv_test(X_features: np.ndarray, y: np.ndarray, model_type='ridge') -> dict:
    """Test features with nested CV."""
    if len(y) < 15 or np.isnan(X_features).all():
        return {'r2': np.nan, 'mse': np.nan}

    # Remove samples with NaN
    valid_mask = ~(np.isnan(X_features).any(axis=1) | np.isnan(y))
    if valid_mask.sum() < 15:
        return {'r2': np.nan, 'mse': np.nan}

    X_clean = X_features[valid_mask]
    y_clean = y[valid_mask]

    preds = np.zeros(len(y_clean))
    preds[:] = np.nan

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X_clean):
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train, y_test = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=3, random_state=42)
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, min_samples_leaf=3, random_state=42)

        model.fit(X_tr_s, y_train)
        preds[test_idx] = model.predict(X_te_s)

    mse = np.mean((preds - y_clean) ** 2)
    baseline_mse = np.var(y_clean)
    r2 = 1 - mse / baseline_mse

    return {'r2': r2, 'mse': mse}


def extract_comprehensive_features(X_player, col_to_idx, frame_range=(100, 180)):
    """Extract comprehensive physics features."""
    n_samples = X_player.shape[0]
    features = []
    feature_names = []

    # Key body parts
    parts = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist', 'right_hip', 'right_knee']
    axes = ['x', 'y', 'z']

    # Sample frames
    frames = list(range(frame_range[0], frame_range[1], 10))

    for part in parts:
        for axis in axes:
            col = f"{part}_{axis}"
            if col not in col_to_idx:
                continue
            idx = col_to_idx[col]

            for frame in frames:
                if frame >= X_player.shape[1]:
                    continue

                # Position
                pos = X_player[:, frame, idx]
                features.append(pos)
                feature_names.append(f"{col}_pos_f{frame}")

                # Velocity (5-frame window)
                if frame + 5 < X_player.shape[1]:
                    vel = X_player[:, frame + 5, idx] - X_player[:, frame, idx]
                    features.append(vel)
                    feature_names.append(f"{col}_vel_f{frame}")

                # Acceleration
                if frame + 10 < X_player.shape[1]:
                    v1 = X_player[:, frame + 5, idx] - X_player[:, frame, idx]
                    v2 = X_player[:, frame + 10, idx] - X_player[:, frame + 5, idx]
                    acc = v2 - v1
                    features.append(acc)
                    feature_names.append(f"{col}_acc_f{frame}")

    # Joint angles and differences
    for frame in frames:
        if frame >= X_player.shape[1]:
            continue

        # Wrist-elbow difference (arm extension)
        for axis in axes:
            if f"right_wrist_{axis}" in col_to_idx and f"right_elbow_{axis}" in col_to_idx:
                diff = X_player[:, frame, col_to_idx[f"right_wrist_{axis}"]] - X_player[:, frame, col_to_idx[f"right_elbow_{axis}"]]
                features.append(diff)
                feature_names.append(f"wrist_elbow_{axis}_diff_f{frame}")

        # Shoulder alignment (right-left)
        for axis in axes:
            if f"right_shoulder_{axis}" in col_to_idx and f"left_shoulder_{axis}" in col_to_idx:
                diff = X_player[:, frame, col_to_idx[f"right_shoulder_{axis}"]] - X_player[:, frame, col_to_idx[f"left_shoulder_{axis}"]]
                features.append(diff)
                feature_names.append(f"shoulder_align_{axis}_f{frame}")

    if len(features) == 0:
        return None, None

    return np.column_stack(features), feature_names


def main():
    print("=" * 80)
    print("DEEPER PHYSICS TEST")
    print("=" * 80)
    print()
    print("Testing if there's more signal with:")
    print("1. Velocity in all axes")
    print("2. Acceleration")
    print("3. Nonlinear models (RF, GBM)")
    print("4. Comprehensive feature combinations")
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2]
    }

    print(f"Data: {len(y)} samples, {len(np.unique(participant_ids))} players")

    results = []

    # ==========================================================================
    # Test 1: Comprehensive features with different models
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 1: COMPREHENSIVE FEATURES + DIFFERENT MODELS")
    print("=" * 80)

    for target_name, target in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target[mask]

            # Extract comprehensive features
            features, feature_names = extract_comprehensive_features(X_player, col_to_idx)

            if features is None:
                print(f"  Player {pid}: Feature extraction failed")
                continue

            # Test with different models
            ridge_result = nested_cv_test(features, y_player, 'ridge')
            rf_result = nested_cv_test(features, y_player, 'rf')
            gbm_result = nested_cv_test(features, y_player, 'gbm')

            best_r2 = max(ridge_result['r2'], rf_result['r2'], gbm_result['r2'])
            best_model = 'ridge' if ridge_result['r2'] == best_r2 else ('rf' if rf_result['r2'] == best_r2 else 'gbm')

            print(f"  Player {pid}: Ridge={ridge_result['r2']:.3f}, RF={rf_result['r2']:.3f}, GBM={gbm_result['r2']:.3f} -> Best: {best_model} R²={best_r2:.3f}")

            results.append({
                'test': 'comprehensive',
                'target': target_name,
                'player': pid,
                'ridge_r2': ridge_result['r2'],
                'rf_r2': rf_result['r2'],
                'gbm_r2': gbm_result['r2'],
                'best_r2': best_r2,
                'best_model': best_model,
                'n_features': features.shape[1]
            })

    # ==========================================================================
    # Test 2: Release window focus (frames 150-180)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 2: RELEASE WINDOW FOCUS (frames 150-180)")
    print("=" * 80)

    for target_name, target in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target[mask]

            # Focus on release window
            features, _ = extract_comprehensive_features(X_player, col_to_idx, frame_range=(150, 185))

            if features is None:
                continue

            rf_result = nested_cv_test(features, y_player, 'rf')
            print(f"  Player {pid}: RF R²={rf_result['r2']:.3f} ({features.shape[1]} features)")

            results.append({
                'test': 'release_window',
                'target': target_name,
                'player': pid,
                'rf_r2': rf_result['r2'],
                'n_features': features.shape[1]
            })

    # ==========================================================================
    # Test 3: Power phase focus for depth (frames 80-140)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 3: POWER PHASE FOR DEPTH (frames 80-140)")
    print("=" * 80)

    target = targets['depth']

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = target[mask]

        # Focus on power phase
        features, _ = extract_comprehensive_features(X_player, col_to_idx, frame_range=(80, 145))

        if features is None:
            continue

        rf_result = nested_cv_test(features, y_player, 'rf')
        print(f"  Player {pid}: RF R²={rf_result['r2']:.3f} ({features.shape[1]} features)")

    # ==========================================================================
    # Test 4: Maximum features (all frames, all body parts)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 4: MAXIMUM FEATURES (upper bound test)")
    print("=" * 80)
    print()
    print("Using ALL available features to find theoretical upper bound")
    print("(May overfit, but shows maximum possible signal)")

    for target_name, target in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target[mask]

            # Use ALL frames with many body parts
            features, _ = extract_comprehensive_features(X_player, col_to_idx, frame_range=(50, 220))

            if features is None:
                continue

            # Use stronger RF
            rf_strong = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=2, random_state=42)

            # Nested CV
            valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(y_player))
            X_clean = features[valid_mask]
            y_clean = y_player[valid_mask]

            if len(y_clean) < 15:
                continue

            preds = np.zeros(len(y_clean))
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, test_idx in kf.split(X_clean):
                X_tr, X_te = X_clean[train_idx], X_clean[test_idx]
                y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                rf_strong.fit(X_tr_s, y_tr)
                preds[test_idx] = rf_strong.predict(X_te_s)

            mse = np.mean((preds - y_clean) ** 2)
            r2 = 1 - mse / np.var(y_clean)

            print(f"  Player {pid}: R²={r2:.3f} ({features.shape[1]} features)")

            results.append({
                'test': 'maximum_features',
                'target': target_name,
                'player': pid,
                'rf_r2': r2,
                'n_features': features.shape[1]
            })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY: MAXIMUM ACHIEVABLE R² PER TARGET")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for target_name in ['angle', 'depth', 'left_right']:
        target_results = results_df[results_df['target'] == target_name]
        if len(target_results) == 0:
            continue

        # Get best R² per player across all tests
        best_per_player = target_results.groupby('player').apply(
            lambda x: x['rf_r2'].max() if 'rf_r2' in x.columns else x['best_r2'].max()
        )

        print(f"\n{target_name.upper()}:")
        for pid, best_r2 in best_per_player.items():
            print(f"  Player {pid}: Max R² = {best_r2:.3f} ({best_r2*100:.1f}% variance)")

        avg_r2 = best_per_player.mean()
        print(f"  AVERAGE: {avg_r2:.3f} ({avg_r2*100:.1f}% variance)")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("If physics fully determined the outcome, R² should approach 1.0")
    print()
    print("Possible reasons for lower R²:")
    print("1. Motion capture noise/resolution limits")
    print("2. Missing data (fingers, ball contact point, spin)")
    print("3. Target measurement noise")
    print("4. Relevant physics happens outside captured frames")
    print("5. Player inconsistency (same form, different outcome)")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'deeper_physics_test.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'deeper_physics_test.csv'}")


if __name__ == "__main__":
    main()
