"""
ANGLE OPTIMAL FEATURES

Based on XYZ investigation findings:
- Y-axis (forward position) matters more than Z for angle
- Temporal change (prep -> release) gives strong signal
- Different players have different optimal features:
  - Player 1: arm_vector at frame 135 (R²=0.123)
  - Player 2: right_wrist XYZ at frame 110 (R²=0.146)
  - Player 3: left_wrist XYZ at frame 130 (R²=0.030)
  - Player 4: temporal_change prep=100, release=170 (R²=0.225)
  - Player 5: left_wrist XYZ at frame 150 (R²=0.104)

This script tests combinations and finds the optimal feature set per player.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def nested_cv_test(X_features: np.ndarray, y: np.ndarray) -> dict:
    """Test features with nested CV."""
    if len(y) < 15 or np.isnan(X_features).all():
        return {'r2': np.nan, 'mse': np.nan}

    preds = np.zeros(len(y))
    preds[:] = np.nan

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X_features):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        if valid.sum() < 5:
            preds[test_idx] = np.nanmean(y_train)
            continue

        X_clean = X_train[valid]
        y_clean = y_train[valid]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_clean)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_clean)

        X_test_clean = np.nan_to_num(X_test, nan=0.0)
        X_te_s = scaler.transform(X_test_clean)
        preds[test_idx] = model.predict(X_te_s)

    preds[np.isnan(preds)] = np.nanmean(y)

    mse = np.mean((preds - y) ** 2)
    baseline_mse = np.var(y)
    r2 = 1 - mse / baseline_mse

    return {'r2': r2, 'mse': mse, 'preds': preds}


def extract_features(X_player, col_to_idx, feature_type, frame=None, prep_frame=None, release_frame=None):
    """Extract specific feature types."""
    n_samples = X_player.shape[0]

    if feature_type == 'right_wrist_xyz':
        return np.column_stack([
            X_player[:, frame, col_to_idx['right_wrist_x']],
            X_player[:, frame, col_to_idx['right_wrist_y']],
            X_player[:, frame, col_to_idx['right_wrist_z']]
        ])

    elif feature_type == 'left_wrist_xyz':
        return np.column_stack([
            X_player[:, frame, col_to_idx['left_wrist_x']],
            X_player[:, frame, col_to_idx['left_wrist_y']],
            X_player[:, frame, col_to_idx['left_wrist_z']]
        ])

    elif feature_type == 'right_elbow_xyz':
        return np.column_stack([
            X_player[:, frame, col_to_idx['right_elbow_x']],
            X_player[:, frame, col_to_idx['right_elbow_y']],
            X_player[:, frame, col_to_idx['right_elbow_z']]
        ])

    elif feature_type == 'arm_vector':
        shoulder = np.column_stack([
            X_player[:, frame, col_to_idx['right_shoulder_x']],
            X_player[:, frame, col_to_idx['right_shoulder_y']],
            X_player[:, frame, col_to_idx['right_shoulder_z']]
        ])
        wrist = np.column_stack([
            X_player[:, frame, col_to_idx['right_wrist_x']],
            X_player[:, frame, col_to_idx['right_wrist_y']],
            X_player[:, frame, col_to_idx['right_wrist_z']]
        ])
        arm_vector = wrist - shoulder
        arm_length = np.sqrt(np.sum(arm_vector**2, axis=1))
        arm_angle_z = np.arccos(np.clip(arm_vector[:, 2] / (arm_length + 1e-8), -1, 1))
        xy_length = np.sqrt(arm_vector[:, 0]**2 + arm_vector[:, 1]**2)
        arm_angle_xy = np.arctan2(arm_vector[:, 2], xy_length)
        arm_azimuth = np.arctan2(arm_vector[:, 1], arm_vector[:, 0])
        return np.column_stack([arm_angle_z, arm_angle_xy, arm_azimuth, arm_length])

    elif feature_type == 'temporal_change':
        features = []
        for part in ['right_wrist', 'right_elbow']:
            for axis in ['x', 'y', 'z']:
                col = f"{part}_{axis}"
                change = X_player[:, release_frame, col_to_idx[col]] - X_player[:, prep_frame, col_to_idx[col]]
                features.append(change)
        return np.column_stack(features)

    elif feature_type == 'full_arm_xyz':
        features = []
        for part in ['right_shoulder', 'right_elbow', 'right_wrist']:
            for axis in ['x', 'y', 'z']:
                col = f"{part}_{axis}"
                features.append(X_player[:, frame, col_to_idx[col]])
        return np.column_stack(features)

    elif feature_type == 'both_wrists_xyz':
        return np.column_stack([
            X_player[:, frame, col_to_idx['right_wrist_x']],
            X_player[:, frame, col_to_idx['right_wrist_y']],
            X_player[:, frame, col_to_idx['right_wrist_z']],
            X_player[:, frame, col_to_idx['left_wrist_x']],
            X_player[:, frame, col_to_idx['left_wrist_y']],
            X_player[:, frame, col_to_idx['left_wrist_z']]
        ])

    elif feature_type == 'y_axis_key':
        # Key Y-axis features (forward position matters for angle)
        return np.column_stack([
            X_player[:, frame, col_to_idx['right_wrist_y']],
            X_player[:, frame, col_to_idx['right_elbow_y']],
            X_player[:, frame, col_to_idx['right_shoulder_y']]
        ])

    return None


def main():
    print("=" * 80)
    print("ANGLE OPTIMAL FEATURES")
    print("=" * 80)
    print()
    print("Finding optimal feature combinations per player for angle prediction")
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    participant_ids = meta['participant_id'].values
    angle_target = y[:, 0]

    print(f"Data: {len(y)} samples, {len(np.unique(participant_ids))} players")

    # Feature configurations to test
    FEATURE_CONFIGS = [
        # Single body part XYZ
        ('right_wrist_xyz', {'frames': [100, 110, 120, 130, 140, 150, 160, 170]}),
        ('left_wrist_xyz', {'frames': [100, 110, 120, 130, 140, 150, 160, 170]}),
        ('right_elbow_xyz', {'frames': [100, 110, 120, 130, 140, 150, 160, 170]}),

        # Derived features
        ('arm_vector', {'frames': [120, 130, 135, 140, 150, 160, 170]}),
        ('full_arm_xyz', {'frames': [110, 120, 130, 140, 150, 160, 170]}),
        ('both_wrists_xyz', {'frames': [100, 110, 120, 130, 140, 150, 160]}),
        ('y_axis_key', {'frames': [100, 110, 120, 130, 140, 150, 160, 170, 180]}),

        # Temporal features
        ('temporal_change', {'prep_frames': [80, 100, 120], 'release_frames': [150, 160, 170]}),
    ]

    results = []
    best_per_player = {}

    # ==========================================================================
    # Test all configurations per player
    # ==========================================================================

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        print(f"\n{'='*80}")
        print(f"PLAYER {pid}")
        print(f"{'='*80}")

        player_best_r2 = -999
        player_best_config = None

        for feature_type, config in FEATURE_CONFIGS:

            if feature_type == 'temporal_change':
                # Test all prep/release combinations
                for prep in config['prep_frames']:
                    for release in config['release_frames']:
                        if release <= prep:
                            continue

                        features = extract_features(
                            X_player, col_to_idx, feature_type,
                            prep_frame=prep, release_frame=release
                        )

                        result = nested_cv_test(features, y_player)

                        config_str = f"{feature_type}_prep{prep}_rel{release}"
                        results.append({
                            'player': pid,
                            'feature_type': feature_type,
                            'config': config_str,
                            'r2': result['r2'],
                            'mse': result['mse']
                        })

                        if result['r2'] > player_best_r2:
                            player_best_r2 = result['r2']
                            player_best_config = config_str

                        if result['r2'] > 0.1:
                            print(f"  {config_str}: R²={result['r2']:.4f} ***")
            else:
                # Test all frames
                for frame in config['frames']:
                    features = extract_features(
                        X_player, col_to_idx, feature_type, frame=frame
                    )

                    if features is None:
                        continue

                    result = nested_cv_test(features, y_player)

                    config_str = f"{feature_type}_f{frame}"
                    results.append({
                        'player': pid,
                        'feature_type': feature_type,
                        'config': config_str,
                        'r2': result['r2'],
                        'mse': result['mse']
                    })

                    if result['r2'] > player_best_r2:
                        player_best_r2 = result['r2']
                        player_best_config = config_str

                    if result['r2'] > 0.1:
                        print(f"  {config_str}: R²={result['r2']:.4f} ***")

        best_per_player[pid] = {
            'config': player_best_config,
            'r2': player_best_r2
        }

        print(f"\n  BEST: {player_best_config} -> R²={player_best_r2:.4f}")

    # ==========================================================================
    # Test combined features (best single + additional)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TESTING FEATURE COMBINATIONS")
    print("=" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        print(f"\nPlayer {pid}:")

        # Get best single feature
        best_single = best_per_player[pid]

        # Try combining with other feature types
        combo_results = []

        # Base features to combine
        base_configs = [
            ('right_wrist_xyz', 110),
            ('left_wrist_xyz', 130),
            ('arm_vector', 135),
            ('y_axis_key', 150),
        ]

        for config1_name, config1_frame in base_configs:
            for config2_name, config2_frame in base_configs:
                if config1_name == config2_name:
                    continue

                features1 = extract_features(X_player, col_to_idx, config1_name, frame=config1_frame)
                features2 = extract_features(X_player, col_to_idx, config2_name, frame=config2_frame)

                if features1 is None or features2 is None:
                    continue

                combined = np.column_stack([features1, features2])
                result = nested_cv_test(combined, y_player)

                combo_name = f"{config1_name}_f{config1_frame} + {config2_name}_f{config2_frame}"
                combo_results.append({
                    'combo': combo_name,
                    'r2': result['r2']
                })

        # Add temporal combo
        for prep, rel in [(100, 170), (120, 160)]:
            temporal = extract_features(X_player, col_to_idx, 'temporal_change', prep_frame=prep, release_frame=rel)

            for config_name, config_frame in base_configs[:2]:
                single = extract_features(X_player, col_to_idx, config_name, frame=config_frame)
                if single is None or temporal is None:
                    continue

                combined = np.column_stack([single, temporal])
                result = nested_cv_test(combined, y_player)

                combo_name = f"{config_name}_f{config_frame} + temporal_prep{prep}_rel{rel}"
                combo_results.append({
                    'combo': combo_name,
                    'r2': result['r2']
                })

        # Show best combos
        combo_df = pd.DataFrame(combo_results)
        if len(combo_df) > 0:
            combo_df = combo_df.sort_values('r2', ascending=False)

            for _, row in combo_df.head(3).iterrows():
                indicator = "***" if row['r2'] > best_single['r2'] else ""
                print(f"  {row['combo']}: R²={row['r2']:.4f} {indicator}")

            # Update best if combo is better
            best_combo = combo_df.iloc[0]
            if best_combo['r2'] > best_per_player[pid]['r2']:
                best_per_player[pid] = {
                    'config': best_combo['combo'],
                    'r2': best_combo['r2']
                }

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("FINAL SUMMARY: BEST ANGLE FEATURES PER PLAYER")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for pid in sorted(np.unique(participant_ids)):
        best = best_per_player[pid]
        print(f"\nPlayer {pid}: R² = {best['r2']:.4f}")
        print(f"  Config: {best['config']}")

    # Compare with baseline
    print()
    print("=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)
    print()
    print("Baseline (mean prediction) has R² = 0")
    print()

    total_signal = 0
    for pid, best in best_per_player.items():
        if best['r2'] > 0:
            total_signal += 1
            print(f"Player {pid}: R²={best['r2']:.4f} -> {best['r2']*100:.1f}% variance explained")

    print(f"\nPlayers with positive R²: {total_signal}/5")

    # Average R² across players
    avg_r2 = np.mean([v['r2'] for v in best_per_player.values() if not np.isnan(v['r2'])])
    print(f"Average R² across players: {avg_r2:.4f}")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'angle_optimal_features.csv', index=False)

    # Save best per player
    best_df = pd.DataFrame([
        {'player': pid, 'config': v['config'], 'r2': v['r2']}
        for pid, v in best_per_player.items()
    ])
    best_df.to_csv(OUTPUT_DIR / 'angle_best_per_player.csv', index=False)

    print(f"\nResults saved to {OUTPUT_DIR / 'angle_optimal_features.csv'}")
    print(f"Best configs saved to {OUTPUT_DIR / 'angle_best_per_player.csv'}")


if __name__ == "__main__":
    main()
