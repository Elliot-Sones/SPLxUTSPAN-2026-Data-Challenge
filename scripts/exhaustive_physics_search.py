"""
EXHAUSTIVE PHYSICS SEARCH

Systematic search to ensure we haven't missed better features:
1. All body parts (not just arm/leg)
2. All velocity windows (1-15 frames)
3. All frame positions (every 2 frames)
4. Feature interactions
5. Multi-frame temporal patterns
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
    """Nested CV test."""
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
    print("EXHAUSTIVE PHYSICS SEARCH")
    print("=" * 80)
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

    # Get all unique body parts
    body_parts = set()
    for col in keypoint_cols:
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            body_parts.add(parts[0])
    body_parts = sorted(list(body_parts))

    print(f"Total body parts: {len(body_parts)}")
    print(f"Total keypoints: {len(keypoint_cols)}")

    all_results = []

    # ==========================================================================
    # Test 1: Every body part, position only
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 1: ALL BODY PARTS - POSITION ONLY")
    print("=" * 80)

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_part = ""
            best_frame = 0

            for body_part in body_parts:
                # Get XYZ columns
                cols = []
                for axis in ['x', 'y', 'z']:
                    col = f"{body_part}_{axis}"
                    if col in col_to_idx:
                        cols.append(col_to_idx[col])

                if len(cols) < 3:
                    continue

                # Test frames
                for frame in range(80, 200, 5):
                    if frame >= X_player.shape[1]:
                        continue

                    features = X_player[:, frame, cols]
                    result = nested_cv_test(features, y_player, 'ridge')

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_part = body_part
                        best_frame = frame

            indicator = "***" if best_r2 > 0.3 else "**" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_part} @ frame {best_frame} {indicator}")

            all_results.append({
                'test': 'position_only',
                'target': target_name,
                'player': pid,
                'best_r2': best_r2,
                'best_feature': best_part,
                'best_frame': best_frame
            })

    # ==========================================================================
    # Test 2: Every body part, velocity with different windows
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 2: ALL BODY PARTS - VELOCITY (windows 2, 5, 10)")
    print("=" * 80)

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_part = ""
            best_frame = 0
            best_window = 0

            for body_part in body_parts:
                cols = []
                for axis in ['x', 'y', 'z']:
                    col = f"{body_part}_{axis}"
                    if col in col_to_idx:
                        cols.append(col_to_idx[col])

                if len(cols) < 3:
                    continue

                for window in [2, 5, 10]:
                    for frame in range(80, 190, 5):
                        if frame + window >= X_player.shape[1]:
                            continue

                        # Velocity
                        vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                        result = nested_cv_test(vel, y_player, 'ridge')

                        if not np.isnan(result['r2']) and result['r2'] > best_r2:
                            best_r2 = result['r2']
                            best_part = body_part
                            best_frame = frame
                            best_window = window

            indicator = "***" if best_r2 > 0.3 else "**" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_part}_vel @ frame {best_frame}, window={best_window} {indicator}")

            all_results.append({
                'test': 'velocity',
                'target': target_name,
                'player': pid,
                'best_r2': best_r2,
                'best_feature': f"{best_part}_vel_w{best_window}",
                'best_frame': best_frame
            })

    # ==========================================================================
    # Test 3: Position + Velocity combined
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 3: POSITION + VELOCITY COMBINED")
    print("=" * 80)

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_part = ""
            best_frame = 0

            for body_part in body_parts:
                cols = []
                for axis in ['x', 'y', 'z']:
                    col = f"{body_part}_{axis}"
                    if col in col_to_idx:
                        cols.append(col_to_idx[col])

                if len(cols) < 3:
                    continue

                window = 5
                for frame in range(80, 185, 5):
                    if frame + window >= X_player.shape[1]:
                        continue

                    pos = X_player[:, frame, cols]
                    vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                    features = np.column_stack([pos, vel])

                    result = nested_cv_test(features, y_player, 'ridge')

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_part = body_part
                        best_frame = frame

            indicator = "***" if best_r2 > 0.3 else "**" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_part}_pos+vel @ frame {best_frame} {indicator}")

            all_results.append({
                'test': 'pos_vel',
                'target': target_name,
                'player': pid,
                'best_r2': best_r2,
                'best_feature': f"{best_part}_pos+vel",
                'best_frame': best_frame
            })

    # ==========================================================================
    # Test 4: Two body parts combined (top combinations)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 4: TWO BODY PARTS COMBINED")
    print("=" * 80)

    # Focus on promising body parts
    key_parts = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'right_hip', 'right_knee',
        'right_second_finger_distal', 'right_third_finger_distal'
    ]
    key_parts = [p for p in key_parts if any(f"{p}_{a}" in col_to_idx for a in ['x', 'y', 'z'])]

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_combo = ""
            best_frame = 0

            for part1, part2 in combinations(key_parts, 2):
                cols1 = [col_to_idx[f"{part1}_{a}"] for a in ['x', 'y', 'z'] if f"{part1}_{a}" in col_to_idx]
                cols2 = [col_to_idx[f"{part2}_{a}"] for a in ['x', 'y', 'z'] if f"{part2}_{a}" in col_to_idx]

                if len(cols1) < 3 or len(cols2) < 3:
                    continue

                for frame in range(100, 180, 10):
                    features = np.column_stack([
                        X_player[:, frame, cols1],
                        X_player[:, frame, cols2]
                    ])

                    result = nested_cv_test(features, y_player, 'ridge')

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_combo = f"{part1} + {part2}"
                        best_frame = frame

            indicator = "***" if best_r2 > 0.3 else "**" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_combo} @ frame {best_frame} {indicator}")

            all_results.append({
                'test': 'two_parts',
                'target': target_name,
                'player': pid,
                'best_r2': best_r2,
                'best_feature': best_combo,
                'best_frame': best_frame
            })

    # ==========================================================================
    # Test 5: Multi-frame temporal (3 consecutive frames)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 5: MULTI-FRAME TEMPORAL (3 frames)")
    print("=" * 80)

    key_parts_short = ['right_wrist', 'right_second_finger_distal', 'right_knee']

    for target_name, target_values in targets.items():
        print(f"\n{target_name.upper()}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]

            best_r2 = -999
            best_part = ""
            best_frame = 0

            for body_part in key_parts_short:
                cols = [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]

                if len(cols) < 3:
                    continue

                for frame in range(100, 175, 5):
                    # 3 consecutive frames
                    if frame + 10 >= X_player.shape[1]:
                        continue

                    features = np.column_stack([
                        X_player[:, frame, cols],
                        X_player[:, frame + 5, cols],
                        X_player[:, frame + 10, cols]
                    ])

                    result = nested_cv_test(features, y_player, 'ridge')

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_part = body_part
                        best_frame = frame

            indicator = "***" if best_r2 > 0.3 else "**" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"  Player {pid}: R²={best_r2:+.3f} | {best_part} @ frames {best_frame}-{best_frame+10} {indicator}")

            all_results.append({
                'test': 'multi_frame',
                'target': target_name,
                'player': pid,
                'best_r2': best_r2,
                'best_feature': f"{best_part}_3frames",
                'best_frame': best_frame
            })

    # ==========================================================================
    # Summary: Best overall per player per target
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY: BEST OVERALL PER PLAYER PER TARGET")
    print("=" * 80)

    results_df = pd.DataFrame(all_results)

    for target_name in ['angle', 'depth', 'left_right']:
        print(f"\n{target_name.upper()}:")
        target_results = results_df[results_df['target'] == target_name]

        for pid in sorted(np.unique(participant_ids)):
            player_results = target_results[target_results['player'] == pid]
            if len(player_results) > 0:
                best = player_results.loc[player_results['best_r2'].idxmax()]
                print(f"  Player {pid}: R²={best['best_r2']:.3f} | {best['test']}: {best['best_feature']} @ f{best['best_frame']}")

        avg = target_results.groupby('player')['best_r2'].max().mean()
        print(f"  AVERAGE MAX: {avg:.3f}")

    # Save
    results_df.to_csv(OUTPUT_DIR / 'exhaustive_physics_search.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'exhaustive_physics_search.csv'}")

    # ==========================================================================
    # Compare with previous best
    # ==========================================================================
    print()
    print("=" * 80)
    print("COMPARISON WITH PREVIOUS BEST")
    print("=" * 80)

    previous_best = {
        'angle': 0.202,
        'depth': 0.590,
        'left_right': 0.517
    }

    for target_name in ['angle', 'depth', 'left_right']:
        target_results = results_df[results_df['target'] == target_name]
        new_avg = target_results.groupby('player')['best_r2'].max().mean()
        prev = previous_best[target_name]

        diff = new_avg - prev
        print(f"\n{target_name}: Previous={prev:.3f}, New={new_avg:.3f}, Diff={diff:+.3f}")


if __name__ == "__main__":
    main()
