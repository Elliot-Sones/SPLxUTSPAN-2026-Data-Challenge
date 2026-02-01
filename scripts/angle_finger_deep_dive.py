"""
ANGLE FINGER DEEP DIVE

The fingertip features showed some signal. Let's explore more variations:
1. Different velocity window sizes (1, 2, 3, 5, 10 frames)
2. Different finger combinations
3. Finger joint angles (curl/extension)
4. Finger trajectory (position + velocity + acceleration)
5. Guide hand variations
6. Wrist snap (wrist velocity relative to elbow)
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
    print("ANGLE FINGER DEEP DIVE")
    print("=" * 80)
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    participant_ids = meta['participant_id'].values
    angle_target = y[:, 0]

    # Finger keypoints
    FINGERS = {
        'thumb': 'right_first_finger',
        'index': 'right_second_finger',
        'middle': 'right_third_finger',
        'ring': 'right_fourth_finger',
        'pinky': 'right_fifth_finger',
    }

    JOINTS = ['mcp', 'pip', 'dip', 'distal']  # knuckle to tip

    results = []

    # ==========================================================================
    # Test 1: Different velocity window sizes
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: VELOCITY WINDOW SIZE")
    print("=" * 80)
    print("\nTesting index+middle fingertip velocity with different windows")

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        print(f"\nPlayer {pid}:")

        for window in [1, 2, 3, 5, 10]:
            best_r2 = -999
            best_frame = 155

            for frame in range(145, 175, 5):
                if frame + window >= X_player.shape[1]:
                    continue

                features = []
                for finger in ['index', 'middle']:
                    prefix = FINGERS[finger]
                    for axis in ['x', 'y', 'z']:
                        col = f"{prefix}_distal_{axis}"
                        if col in col_to_idx:
                            idx = col_to_idx[col]
                            vel = (X_player[:, frame + window, idx] - X_player[:, frame, idx]) / window
                            features.append(vel)

                if len(features) == 0:
                    continue

                feat_array = np.column_stack(features)
                result = nested_cv_test(feat_array, y_player, 'ridge')

                if not np.isnan(result['r2']) and result['r2'] > best_r2:
                    best_r2 = result['r2']
                    best_frame = frame

            indicator = "***" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"  window={window}: R²={best_r2:+.3f} @ frame {best_frame} {indicator}")

            results.append({
                'test': 'velocity_window',
                'player': pid,
                'window': window,
                'best_r2': best_r2,
                'best_frame': best_frame
            })

    # ==========================================================================
    # Test 2: Different finger combinations
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 2: FINGER COMBINATIONS")
    print("=" * 80)

    finger_combos = [
        (['index'], 'Index only'),
        (['middle'], 'Middle only'),
        (['index', 'middle'], 'Index + Middle'),
        (['thumb', 'index', 'middle'], 'Thumb + Index + Middle'),
        (['index', 'middle', 'ring'], 'Index + Middle + Ring'),
        (['thumb', 'index', 'middle', 'ring', 'pinky'], 'All fingers'),
    ]

    for fingers, desc in finger_combos:
        print(f"\n{desc}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = angle_target[mask]

            best_r2 = -999
            best_frame = 155

            for frame in range(145, 175, 5):
                features = []
                for finger in fingers:
                    prefix = FINGERS[finger]
                    for axis in ['x', 'y', 'z']:
                        col = f"{prefix}_distal_{axis}"
                        if col in col_to_idx:
                            features.append(X_player[:, frame, col_to_idx[col]])

                if len(features) == 0:
                    continue

                feat_array = np.column_stack(features)

                for model_type in ['ridge', 'rf']:
                    result = nested_cv_test(feat_array, y_player, model_type)
                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_frame = frame

            indicator = "***" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"  Player {pid}: R²={best_r2:+.3f} @ frame {best_frame} {indicator}")

            results.append({
                'test': 'finger_combo',
                'player': pid,
                'fingers': ','.join(fingers),
                'best_r2': best_r2,
                'best_frame': best_frame
            })

    # ==========================================================================
    # Test 3: Finger joint angles (curl)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 3: FINGER CURL ANGLE")
    print("=" * 80)
    print("\nAngle at each finger joint (how curled the fingers are)")

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        best_r2 = -999
        best_frame = 155

        for frame in range(140, 180, 5):
            features = []

            # Calculate finger curl for index and middle fingers
            for finger in ['index', 'middle']:
                prefix = FINGERS[finger]

                # Get joint positions
                joints = {}
                for joint in ['mcp', 'pip', 'dip', 'distal']:
                    joint_name = f"{prefix}_{joint}"
                    pos = []
                    for axis in ['x', 'y', 'z']:
                        col = f"{joint_name}_{axis}"
                        if col in col_to_idx:
                            pos.append(X_player[:, frame, col_to_idx[col]])
                    if len(pos) == 3:
                        joints[joint] = np.column_stack(pos)

                if len(joints) < 3:
                    continue

                # Calculate angles at PIP and DIP joints
                for joint_name, joint_pos in [('pip', joints.get('pip')), ('dip', joints.get('dip'))]:
                    if joint_pos is None:
                        continue

                    # Get adjacent joints
                    if joint_name == 'pip' and 'mcp' in joints and 'dip' in joints:
                        v1 = joints['mcp'] - joint_pos
                        v2 = joints['dip'] - joint_pos
                    elif joint_name == 'dip' and 'pip' in joints and 'distal' in joints:
                        v1 = joints['pip'] - joint_pos
                        v2 = joints['distal'] - joint_pos
                    else:
                        continue

                    # Calculate angle
                    dot = np.sum(v1 * v2, axis=1)
                    mag = np.sqrt(np.sum(v1**2, axis=1)) * np.sqrt(np.sum(v2**2, axis=1))
                    angle = np.arccos(np.clip(dot / (mag + 1e-8), -1, 1))
                    features.append(angle)

            if len(features) == 0:
                continue

            feat_array = np.column_stack(features)

            for model_type in ['ridge', 'rf']:
                result = nested_cv_test(feat_array, y_player, model_type)
                if not np.isnan(result['r2']) and result['r2'] > best_r2:
                    best_r2 = result['r2']
                    best_frame = frame

        indicator = "***" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
        print(f"  Player {pid}: R²={best_r2:+.3f} @ frame {best_frame} {indicator}")

        results.append({
            'test': 'finger_curl',
            'player': pid,
            'best_r2': best_r2,
            'best_frame': best_frame
        })

    # ==========================================================================
    # Test 4: Wrist snap (wrist velocity relative to elbow)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 4: WRIST SNAP")
    print("=" * 80)
    print("\nWrist velocity relative to elbow (the snap motion)")

    for window in [2, 3, 5]:
        print(f"\n  Window = {window} frames:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = angle_target[mask]

            best_r2 = -999
            best_frame = 155

            for frame in range(150, 175, 3):
                if frame + window >= X_player.shape[1]:
                    continue

                features = []

                # Wrist velocity
                for axis in ['x', 'y', 'z']:
                    wrist_col = f"right_wrist_{axis}"
                    elbow_col = f"right_elbow_{axis}"

                    if wrist_col in col_to_idx and elbow_col in col_to_idx:
                        wrist_vel = (X_player[:, frame + window, col_to_idx[wrist_col]] -
                                    X_player[:, frame, col_to_idx[wrist_col]]) / window
                        elbow_vel = (X_player[:, frame + window, col_to_idx[elbow_col]] -
                                    X_player[:, frame, col_to_idx[elbow_col]]) / window

                        # Wrist velocity relative to elbow
                        rel_vel = wrist_vel - elbow_vel
                        features.append(rel_vel)

                        # Also add absolute wrist velocity
                        features.append(wrist_vel)

                if len(features) == 0:
                    continue

                feat_array = np.column_stack(features)

                for model_type in ['ridge', 'rf']:
                    result = nested_cv_test(feat_array, y_player, model_type)
                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_frame = frame

            indicator = "***" if best_r2 > 0.1 else "+" if best_r2 > 0 else ""
            print(f"    Player {pid}: R²={best_r2:+.3f} @ frame {best_frame} {indicator}")

            results.append({
                'test': 'wrist_snap',
                'player': pid,
                'window': window,
                'best_r2': best_r2,
                'best_frame': best_frame
            })

    # ==========================================================================
    # Test 5: Combined best features
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 5: COMBINED FEATURES")
    print("=" * 80)
    print("\nCombining fingertip position + velocity + wrist snap + guide hand")

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        best_r2 = -999
        best_config = ""

        for frame in range(145, 175, 5):
            for window in [3, 5]:
                if frame + window >= X_player.shape[1]:
                    continue

                features = []

                # Fingertip positions (index, middle)
                for finger in ['index', 'middle']:
                    prefix = FINGERS[finger]
                    for axis in ['x', 'y', 'z']:
                        col = f"{prefix}_distal_{axis}"
                        if col in col_to_idx:
                            features.append(X_player[:, frame, col_to_idx[col]])

                # Fingertip velocities
                for finger in ['index', 'middle']:
                    prefix = FINGERS[finger]
                    for axis in ['x', 'y', 'z']:
                        col = f"{prefix}_distal_{axis}"
                        if col in col_to_idx:
                            vel = (X_player[:, frame + window, col_to_idx[col]] -
                                  X_player[:, frame, col_to_idx[col]]) / window
                            features.append(vel)

                # Wrist snap
                for axis in ['x', 'y', 'z']:
                    wrist_col = f"right_wrist_{axis}"
                    elbow_col = f"right_elbow_{axis}"
                    if wrist_col in col_to_idx and elbow_col in col_to_idx:
                        wrist_vel = (X_player[:, frame + window, col_to_idx[wrist_col]] -
                                    X_player[:, frame, col_to_idx[wrist_col]]) / window
                        elbow_vel = (X_player[:, frame + window, col_to_idx[elbow_col]] -
                                    X_player[:, frame, col_to_idx[elbow_col]]) / window
                        features.append(wrist_vel - elbow_vel)

                # Guide hand position
                for axis in ['x', 'y', 'z']:
                    col = f"left_wrist_{axis}"
                    if col in col_to_idx:
                        features.append(X_player[:, frame, col_to_idx[col]])

                if len(features) == 0:
                    continue

                feat_array = np.column_stack(features)

                for model_type in ['ridge', 'rf']:
                    result = nested_cv_test(feat_array, y_player, model_type)
                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_config = f"frame={frame}, window={window}, model={model_type}"

        indicator = "***" if best_r2 > 0.15 else "+" if best_r2 > 0 else ""
        print(f"  Player {pid}: R²={best_r2:+.3f} ({best_config}) {indicator}")

        results.append({
            'test': 'combined',
            'player': pid,
            'best_r2': best_r2,
            'best_config': best_config
        })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY: BEST ANGLE PHYSICS PER PLAYER")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for pid in sorted(np.unique(participant_ids)):
        player_results = results_df[results_df['player'] == pid]
        if len(player_results) > 0:
            best = player_results.loc[player_results['best_r2'].idxmax()]
            print(f"\nPlayer {pid}: Max R² = {best['best_r2']:.3f}")
            print(f"  Test: {best['test']}")
            if 'best_config' in best and pd.notna(best['best_config']):
                print(f"  Config: {best['best_config']}")
            elif 'window' in best and pd.notna(best['window']):
                print(f"  Window: {best['window']}, Frame: {best['best_frame']}")
            elif 'fingers' in best and pd.notna(best['fingers']):
                print(f"  Fingers: {best['fingers']}, Frame: {best['best_frame']}")

    avg_best = results_df.groupby('player')['best_r2'].max().mean()
    print(f"\nAVERAGE MAX R² ACROSS PLAYERS: {avg_best:.3f}")

    # Save
    results_df.to_csv(OUTPUT_DIR / 'angle_finger_deep_dive.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'angle_finger_deep_dive.csv'}")


if __name__ == "__main__":
    main()
