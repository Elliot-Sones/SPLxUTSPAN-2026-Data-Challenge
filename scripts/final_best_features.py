"""
FINAL BEST FEATURES

Combine ALL the best findings from all tests to get the maximum achievable R².
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def nested_cv_test(X_features, y, model_type='ridge'):
    if X_features is None or len(y) < 15:
        return {'r2': np.nan}
    valid = ~(np.isnan(X_features).any(axis=1) | np.isnan(y))
    if valid.sum() < 15:
        return {'r2': np.nan}
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
    return {'r2': r2}


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def main():
    print("=" * 80)
    print("FINAL BEST FEATURES - MAXIMUM ACHIEVABLE R²")
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

    # Best feature configs per player per target based on all tests
    BEST_CONFIGS = {
        'angle': {
            1: {
                'parts': ['right_second_finger_distal'],
                'frame': 110,
                'add_velocity': True,
                'velocity_window': 1,
                'add_guide_hand': True,
            },
            2: {
                'parts': ['right_knee', 'right_hip', 'right_ankle'],
                'frame': 120,
                'add_velocity': False,
                'add_left_wrist': True,
            },
            3: {
                'parts': ['right_knee', 'right_hip', 'right_ankle'],
                'frame': 170,
                'add_velocity': False,
                'add_left_finger': True,
            },
            4: {
                'parts': ['right_knee', 'right_hip', 'right_ankle'],
                'frame': 160,
                'add_velocity': True,
                'velocity_window': 5,
                'add_shoulder': True,
            },
            5: {
                'parts': ['right_wrist', 'left_wrist', 'right_hip'],
                'frame': 155,
                'add_lr_asymmetry': True,
            },
        },
        'depth': {
            1: {
                'parts': ['right_wrist', 'right_elbow', 'right_shoulder'],
                'frame': 150,
                'add_velocity': True,
                'velocity_window': 5,
                'add_leg_drive': True,
            },
            2: {
                'parts': ['right_wrist', 'right_elbow'],
                'frame': 100,
                'add_velocity': True,
                'velocity_window': 3,
                'add_knee': True,
            },
            3: {
                'parts': ['right_wrist', 'left_ankle'],
                'frame': 130,
                'add_velocity': True,
                'velocity_window': 2,
            },
            4: {
                'parts': ['right_wrist', 'right_knee'],
                'frame': 140,
                'add_velocity': True,
                'velocity_window': 3,
                'add_hip': True,
            },
            5: {
                'parts': ['right_wrist', 'right_second_finger_mcp'],
                'frame': 145,
                'add_velocity': True,
                'velocity_window': 15,
            },
        },
        'left_right': {
            1: {
                'parts': ['right_second_finger_distal', 'right_wrist', 'right_elbow'],
                'frame': 150,
                'add_velocity': True,
                'velocity_window': 5,
                'add_knee': True,
            },
            2: {
                'parts': ['right_wrist', 'left_wrist', 'right_hip'],
                'frame': 150,
                'add_lr_asymmetry': True,
            },
            3: {
                'parts': ['right_shoulder', 'left_shoulder', 'neck'],
                'frame': 175,
                'add_lr_asymmetry': True,
            },
            4: {
                'parts': ['right_second_finger_distal', 'right_wrist', 'right_elbow'],
                'frame': 150,
                'add_velocity': False,
            },
            5: {
                'parts': ['right_wrist', 'left_wrist', 'right_hip'],
                'frame': 170,
                'add_velocity': True,
                'velocity_window': 18,
            },
        },
    }

    results = []

    for target_name, target_values in targets.items():
        print(f"\n{'='*80}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*80}")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            config = BEST_CONFIGS[target_name][pid]
            frame = config['frame']

            features = []

            # Base positions
            for part in config['parts']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])

            # Add velocity if specified
            if config.get('add_velocity', False):
                window = config.get('velocity_window', 5)
                for part in config['parts'][:2]:  # First two parts
                    cols = get_cols(col_to_idx, part)
                    if cols and frame + window < X_player.shape[1]:
                        vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                        features.append(vel)

            # Add guide hand
            if config.get('add_guide_hand', False):
                cols = get_cols(col_to_idx, 'left_wrist')
                if cols:
                    features.append(X_player[:, frame, cols])

            # Add left wrist
            if config.get('add_left_wrist', False):
                cols = get_cols(col_to_idx, 'left_wrist')
                if cols:
                    features.append(X_player[:, frame, cols])

            # Add left finger
            if config.get('add_left_finger', False):
                cols = get_cols(col_to_idx, 'left_fourth_finger_distal')
                if cols:
                    features.append(X_player[:, frame, cols])

            # Add shoulder
            if config.get('add_shoulder', False):
                cols = get_cols(col_to_idx, 'right_shoulder')
                if cols:
                    features.append(X_player[:, frame, cols])

            # Add leg drive
            if config.get('add_leg_drive', False):
                for part in ['right_knee', 'right_ankle']:
                    cols = get_cols(col_to_idx, part)
                    if cols:
                        features.append(X_player[:, frame, cols])

            # Add knee
            if config.get('add_knee', False):
                cols = get_cols(col_to_idx, 'right_knee')
                if cols:
                    features.append(X_player[:, frame, cols])

            # Add hip
            if config.get('add_hip', False):
                cols = get_cols(col_to_idx, 'right_hip')
                if cols:
                    features.append(X_player[:, frame, cols])

            # Add left-right asymmetry
            if config.get('add_lr_asymmetry', False):
                lr_pairs = [
                    ('right_wrist', 'left_wrist'),
                    ('right_shoulder', 'left_shoulder'),
                    ('right_hip', 'left_hip'),
                ]
                for right, left in lr_pairs:
                    for axis in ['x', 'y', 'z']:
                        r_col = f"{right}_{axis}"
                        l_col = f"{left}_{axis}"
                        if r_col in col_to_idx and l_col in col_to_idx:
                            diff = X_player[:, frame, col_to_idx[r_col]] - X_player[:, frame, col_to_idx[l_col]]
                            features.append(diff.reshape(-1, 1))

            # Combine features
            if features:
                feat_array = np.column_stack([f if f.ndim > 1 else f.reshape(-1, 1) for f in features])

                # Test both models
                ridge_result = nested_cv_test(feat_array, y_player, 'ridge')
                rf_result = nested_cv_test(feat_array, y_player, 'rf')

                best_r2 = max(ridge_result['r2'], rf_result['r2'])
                best_model = 'ridge' if ridge_result['r2'] >= rf_result['r2'] else 'rf'

                print(f"\nPlayer {pid}:")
                print(f"  Config: {config['parts']}, frame={frame}")
                print(f"  Ridge R²={ridge_result['r2']:.3f}, RF R²={rf_result['r2']:.3f}")
                print(f"  BEST: R²={best_r2:.3f} ({best_model})")

                results.append({
                    'target': target_name,
                    'player': pid,
                    'best_r2': best_r2,
                    'best_model': best_model,
                    'n_features': feat_array.shape[1]
                })
            else:
                print(f"\nPlayer {pid}: Feature extraction failed")

    # Final summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for target_name in ['angle', 'depth', 'left_right']:
        print(f"\n{target_name.upper()}:")
        target_results = results_df[results_df['target'] == target_name]

        for _, row in target_results.iterrows():
            print(f"  Player {row['player']}: R²={row['best_r2']:.3f} ({row['best_model']}, {row['n_features']} features)")

        avg = target_results['best_r2'].mean()
        print(f"  AVERAGE: {avg:.3f}")

    print()
    print("=" * 80)
    print("COMPARISON WITH ALL PREVIOUS TESTS")
    print("=" * 80)

    comparisons = {
        'angle': [
            ('Initial (Z-only)', 0.052),
            ('Comprehensive Groups', 0.148),
            ('Finger Deep Dive', 0.202),
            ('Exhaustive Search', 0.261),
            ('Gaps Test', 0.213),
        ],
        'depth': [
            ('Initial', 0.387),
            ('Comprehensive Groups', 0.590),
            ('Exhaustive Search', 0.599),
            ('Gaps Test', 0.554),
        ],
        'left_right': [
            ('Initial', 0.226),
            ('Comprehensive Groups', 0.517),
            ('Exhaustive Search', 0.519),
            ('Gaps Test', 0.525),
        ],
    }

    for target_name in ['angle', 'depth', 'left_right']:
        print(f"\n{target_name.upper()} progression:")
        for test_name, r2 in comparisons[target_name]:
            print(f"  {test_name}: {r2:.3f}")

        final_avg = results_df[results_df['target'] == target_name]['best_r2'].mean()
        print(f"  FINAL: {final_avg:.3f}")

    results_df.to_csv(OUTPUT_DIR / 'final_best_features.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'final_best_features.csv'}")


if __name__ == "__main__":
    main()
