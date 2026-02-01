"""
EXHAUSTIVE GAPS TEST

Thoroughly test all remaining gaps:
1. All velocity windows 1-20
2. Feature interactions (products, ratios, differences)
3. Three body part combinations
4. Finer frame granularity (every frame)
5. Left vs right comparisons
6. Different derivatives (jerk = acceleration change)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from itertools import combinations
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
            model = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=3, random_state=42)
        model.fit(X_tr_s, y_tr)
        preds[test_idx] = model.predict(X_te_s)
    mse = np.mean((preds - y_clean) ** 2)
    r2 = 1 - mse / np.var(y_clean)
    return {'r2': r2}


def main():
    print("=" * 80)
    print("EXHAUSTIVE GAPS TEST")
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

    # Key body parts for thorough testing
    key_parts = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'left_elbow', 'left_shoulder',
        'right_hip', 'right_knee', 'right_ankle',
        'right_second_finger_distal', 'right_third_finger_distal',
        'left_second_finger_distal', 'neck', 'nose'
    ]
    key_parts = [p for p in key_parts if any(f"{p}_{a}" in col_to_idx for a in ['x', 'y', 'z'])]

    all_results = []

    # ==========================================================================
    # GAP 1: All velocity windows 1-20
    # ==========================================================================
    print()
    print("=" * 80)
    print("GAP 1: ALL VELOCITY WINDOWS 1-20")
    print("=" * 80)

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_config = ""

            for body_part in ['right_wrist', 'right_second_finger_distal', 'right_knee', 'left_wrist']:
                cols = [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]
                if len(cols) < 3:
                    continue

                for window in range(1, 21):
                    for frame in range(100, 180, 10):
                        if frame + window >= X_player.shape[1]:
                            continue

                        vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                        result = nested_cv_test(vel, y_player)

                        if not np.isnan(result['r2']) and result['r2'] > best_r2:
                            best_r2 = result['r2']
                            best_config = f"{body_part}_vel_w{window}_f{frame}"

            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_config}")
            all_results.append({'gap': 'velocity_windows', 'target': target_name, 'player': pid, 'best_r2': best_r2, 'config': best_config})

    # ==========================================================================
    # GAP 2: Feature interactions (products, ratios)
    # ==========================================================================
    print()
    print("=" * 80)
    print("GAP 2: FEATURE INTERACTIONS")
    print("=" * 80)

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_config = ""

            for frame in range(140, 180, 10):
                # Interaction: wrist_z * elbow_z, wrist_z / elbow_z
                wrist_z = f"right_wrist_z"
                elbow_z = f"right_elbow_z"

                if wrist_z in col_to_idx and elbow_z in col_to_idx:
                    w = X_player[:, frame, col_to_idx[wrist_z]]
                    e = X_player[:, frame, col_to_idx[elbow_z]]

                    features = np.column_stack([
                        w, e,
                        w * e,  # product
                        w - e,  # difference
                        w / (e + 0.01),  # ratio
                        (w - e) ** 2,  # squared difference
                    ])

                    result = nested_cv_test(features, y_player)
                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_config = f"wrist_elbow_interactions_f{frame}"

                # Finger interactions
                idx_z = f"right_second_finger_distal_z"
                mid_z = f"right_third_finger_distal_z"

                if idx_z in col_to_idx and mid_z in col_to_idx:
                    i = X_player[:, frame, col_to_idx[idx_z]]
                    m = X_player[:, frame, col_to_idx[mid_z]]

                    features = np.column_stack([
                        i, m,
                        i - m,  # finger spread
                        (i + m) / 2,  # average
                    ])

                    result = nested_cv_test(features, y_player)
                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_config = f"finger_interactions_f{frame}"

            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_config}")
            all_results.append({'gap': 'interactions', 'target': target_name, 'player': pid, 'best_r2': best_r2, 'config': best_config})

    # ==========================================================================
    # GAP 3: Three body part combinations
    # ==========================================================================
    print()
    print("=" * 80)
    print("GAP 3: THREE BODY PART COMBINATIONS")
    print("=" * 80)

    three_part_combos = [
        ['right_wrist', 'right_elbow', 'right_shoulder'],
        ['right_wrist', 'left_wrist', 'right_hip'],
        ['right_second_finger_distal', 'right_wrist', 'right_elbow'],
        ['right_knee', 'right_hip', 'right_ankle'],
        ['right_shoulder', 'left_shoulder', 'neck'],
    ]

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_config = ""

            for combo in three_part_combos:
                all_cols = []
                valid_combo = True
                for part in combo:
                    cols = [col_to_idx[f"{part}_{a}"] for a in ['x', 'y', 'z'] if f"{part}_{a}" in col_to_idx]
                    if len(cols) < 3:
                        valid_combo = False
                        break
                    all_cols.extend(cols)

                if not valid_combo:
                    continue

                for frame in range(120, 180, 10):
                    features = X_player[:, frame, all_cols]
                    result = nested_cv_test(features, y_player)

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_config = f"{'+'.join(combo)}_f{frame}"

            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_config}")
            all_results.append({'gap': 'three_parts', 'target': target_name, 'player': pid, 'best_r2': best_r2, 'config': best_config})

    # ==========================================================================
    # GAP 4: Finer frame granularity (every 2 frames in critical window)
    # ==========================================================================
    print()
    print("=" * 80)
    print("GAP 4: FINE FRAME GRANULARITY (every 2 frames)")
    print("=" * 80)

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_config = ""

            for body_part in ['right_wrist', 'right_second_finger_distal']:
                cols = [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]
                if len(cols) < 3:
                    continue

                # Fine search: every 2 frames from 140-180
                for frame in range(140, 180, 2):
                    features = X_player[:, frame, cols]
                    result = nested_cv_test(features, y_player)

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_config = f"{body_part}_f{frame}"

            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_config}")
            all_results.append({'gap': 'fine_frames', 'target': target_name, 'player': pid, 'best_r2': best_r2, 'config': best_config})

    # ==========================================================================
    # GAP 5: Left-right asymmetry features
    # ==========================================================================
    print()
    print("=" * 80)
    print("GAP 5: LEFT-RIGHT ASYMMETRY")
    print("=" * 80)

    lr_pairs = [
        ('right_wrist', 'left_wrist'),
        ('right_elbow', 'left_elbow'),
        ('right_shoulder', 'left_shoulder'),
        ('right_hip', 'left_hip'),
        ('right_knee', 'left_knee'),
        ('right_second_finger_distal', 'left_second_finger_distal'),
    ]

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_config = ""

            for frame in range(140, 180, 5):
                features = []

                for right, left in lr_pairs:
                    for axis in ['x', 'y', 'z']:
                        r_col = f"{right}_{axis}"
                        l_col = f"{left}_{axis}"
                        if r_col in col_to_idx and l_col in col_to_idx:
                            diff = X_player[:, frame, col_to_idx[r_col]] - X_player[:, frame, col_to_idx[l_col]]
                            features.append(diff)

                if len(features) > 0:
                    feat_array = np.column_stack(features)
                    result = nested_cv_test(feat_array, y_player)

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_config = f"lr_asymmetry_f{frame}"

            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_config}")
            all_results.append({'gap': 'lr_asymmetry', 'target': target_name, 'player': pid, 'best_r2': best_r2, 'config': best_config})

    # ==========================================================================
    # GAP 6: Jerk (acceleration change)
    # ==========================================================================
    print()
    print("=" * 80)
    print("GAP 6: JERK (acceleration change)")
    print("=" * 80)

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_config = ""

            for body_part in ['right_wrist', 'right_second_finger_distal', 'right_knee']:
                cols = [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]
                if len(cols) < 3:
                    continue

                window = 5
                for frame in range(100, 170, 10):
                    if frame + 3 * window >= X_player.shape[1]:
                        continue

                    # Velocity at t, t+w, t+2w
                    v1 = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                    v2 = (X_player[:, frame + 2*window, cols] - X_player[:, frame + window, cols]) / window
                    v3 = (X_player[:, frame + 3*window, cols] - X_player[:, frame + 2*window, cols]) / window

                    # Acceleration
                    a1 = (v2 - v1) / window
                    a2 = (v3 - v2) / window

                    # Jerk
                    jerk = (a2 - a1) / window

                    features = np.column_stack([v1, a1, jerk])
                    result = nested_cv_test(features, y_player)

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_config = f"{body_part}_vel+acc+jerk_f{frame}"

            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_config}")
            all_results.append({'gap': 'jerk', 'target': target_name, 'player': pid, 'best_r2': best_r2, 'config': best_config})

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print()
    print("=" * 80)
    print("FINAL SUMMARY: BEST FROM ALL GAPS")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)

    for target_name in ['angle', 'depth', 'left_right']:
        print(f"\n{target_name.upper()}:")
        target_results = results_df[results_df['target'] == target_name]

        for pid in sorted(np.unique(participant_ids)):
            player_results = target_results[target_results['player'] == pid]
            if len(player_results) > 0:
                best = player_results.loc[player_results['best_r2'].idxmax()]
                print(f"  Player {pid}: R²={best['best_r2']:.3f} | {best['gap']}: {best['config']}")

        avg = target_results.groupby('player')['best_r2'].max().mean()
        print(f"  AVERAGE MAX: {avg:.3f}")

    results_df.to_csv(OUTPUT_DIR / 'exhaustive_gaps_test.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'exhaustive_gaps_test.csv'}")


if __name__ == "__main__":
    main()
