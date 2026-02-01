"""
ANGLE XYZ INVESTIGATION

The Z-coordinate alone shows weak signal for angle.
Angle is determined by the DIRECTION of ball release, which depends on:
1. Wrist position in 3D space (X, Y, Z)
2. Arm orientation (shoulder-elbow-wrist vector)
3. Velocity direction at release

This script tests all three coordinate axes and their combinations.
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

    return {'r2': r2, 'mse': mse}


def main():
    print("=" * 80)
    print("ANGLE XYZ INVESTIGATION")
    print("=" * 80)
    print()
    print("Testing X, Y, Z coordinates and combinations for angle prediction")
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    participant_ids = meta['participant_id'].values
    angle_target = y[:, 0]

    print(f"Data: {len(y)} samples, {len(np.unique(participant_ids))} players")
    print(f"Available keypoints: {len(keypoint_cols)}")

    # Find available coordinates
    x_cols = [c for c in keypoint_cols if c.endswith('_x')]
    y_cols = [c for c in keypoint_cols if c.endswith('_y')]
    z_cols = [c for c in keypoint_cols if c.endswith('_z')]

    print(f"X coords: {len(x_cols)}, Y coords: {len(y_cols)}, Z coords: {len(z_cols)}")

    # Key body parts for angle
    key_parts = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist', 'left_elbow', 'left_shoulder']

    results = []

    # ==========================================================================
    # Test 1: Individual coordinate axes at different frames
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 1: Individual Coordinate Axes")
    print("=" * 80)

    for axis in ['x', 'y', 'z']:
        print(f"\n--- {axis.upper()} Axis ---")

        for part in key_parts:
            col_name = f"{part}_{axis}"
            if col_name not in col_to_idx:
                continue

            col_idx = col_to_idx[col_name]

            for pid in sorted(np.unique(participant_ids)):
                mask = participant_ids == pid
                X_player = X[mask]
                y_player = angle_target[mask]

                best_r2 = -999
                best_frame = 0

                # Test frames 100-200
                for frame in range(100, 200, 5):
                    features = X_player[:, frame, col_idx:col_idx+1]
                    result = nested_cv_test(features, y_player)

                    if not np.isnan(result['r2']) and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_frame = frame

                if best_r2 > 0:
                    print(f"  Player {pid} {col_name}: R²={best_r2:.4f} at frame {best_frame} ***")
                    results.append({
                        'test': 'individual_axis',
                        'player': pid,
                        'feature': col_name,
                        'frame': best_frame,
                        'r2': best_r2
                    })

    # ==========================================================================
    # Test 2: Combined XYZ for each body part
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 2: Combined XYZ Per Body Part")
    print("=" * 80)

    for part in key_parts:
        x_col = f"{part}_x"
        y_col = f"{part}_y"
        z_col = f"{part}_z"

        if x_col not in col_to_idx or y_col not in col_to_idx or z_col not in col_to_idx:
            continue

        print(f"\n{part}:")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = angle_target[mask]

            best_r2 = -999
            best_frame = 0

            for frame in range(100, 200, 5):
                features = np.column_stack([
                    X_player[:, frame, col_to_idx[x_col]],
                    X_player[:, frame, col_to_idx[y_col]],
                    X_player[:, frame, col_to_idx[z_col]]
                ])

                result = nested_cv_test(features, y_player)

                if not np.isnan(result['r2']) and result['r2'] > best_r2:
                    best_r2 = result['r2']
                    best_frame = frame

            indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
            print(f"  Player {pid}: R²={best_r2:+.4f} at frame {best_frame} {indicator}")

            results.append({
                'test': 'xyz_combined',
                'player': pid,
                'feature': part,
                'frame': best_frame,
                'r2': best_r2
            })

    # ==========================================================================
    # Test 3: Arm Vector (shoulder to wrist direction)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 3: Arm Vector Direction (determines release trajectory)")
    print("=" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]
        n_samples = mask.sum()

        best_r2 = -999
        best_frame = 0
        best_features = None

        for frame in range(120, 180, 5):
            # Calculate arm vector components
            features = []

            # Vector from shoulder to wrist (right arm)
            if all(f"right_{p}_{a}" in col_to_idx for p in ['shoulder', 'wrist'] for a in ['x', 'y', 'z']):
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

                # Calculate arm angle relative to vertical (z-axis)
                arm_length = np.sqrt(np.sum(arm_vector**2, axis=1))
                arm_angle_z = np.arccos(np.clip(arm_vector[:, 2] / (arm_length + 1e-8), -1, 1))

                # Calculate arm angle relative to horizontal (x-y plane)
                xy_length = np.sqrt(arm_vector[:, 0]**2 + arm_vector[:, 1]**2)
                arm_angle_xy = np.arctan2(arm_vector[:, 2], xy_length)

                # Horizontal direction (azimuth)
                arm_azimuth = np.arctan2(arm_vector[:, 1], arm_vector[:, 0])

                features = np.column_stack([
                    arm_angle_z,
                    arm_angle_xy,
                    arm_azimuth,
                    arm_length
                ])

            if len(features) == 0:
                continue

            result = nested_cv_test(features, y_player)

            if not np.isnan(result['r2']) and result['r2'] > best_r2:
                best_r2 = result['r2']
                best_frame = frame

        indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
        print(f"  Player {pid}: R²={best_r2:+.4f} at frame {best_frame} {indicator}")

        results.append({
            'test': 'arm_vector',
            'player': pid,
            'feature': 'arm_direction',
            'frame': best_frame,
            'r2': best_r2
        })

    # ==========================================================================
    # Test 4: Elbow Angle (bend angle at elbow)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 4: Elbow Bend Angle (arm flexion)")
    print("=" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        best_r2 = -999
        best_frame = 0

        for frame in range(100, 180, 5):
            # Calculate elbow angle using shoulder-elbow-wrist
            if all(f"right_{p}_{a}" in col_to_idx for p in ['shoulder', 'elbow', 'wrist'] for a in ['x', 'y', 'z']):
                shoulder = np.column_stack([
                    X_player[:, frame, col_to_idx['right_shoulder_x']],
                    X_player[:, frame, col_to_idx['right_shoulder_y']],
                    X_player[:, frame, col_to_idx['right_shoulder_z']]
                ])
                elbow = np.column_stack([
                    X_player[:, frame, col_to_idx['right_elbow_x']],
                    X_player[:, frame, col_to_idx['right_elbow_y']],
                    X_player[:, frame, col_to_idx['right_elbow_z']]
                ])
                wrist = np.column_stack([
                    X_player[:, frame, col_to_idx['right_wrist_x']],
                    X_player[:, frame, col_to_idx['right_wrist_y']],
                    X_player[:, frame, col_to_idx['right_wrist_z']]
                ])

                # Vectors
                upper_arm = shoulder - elbow
                forearm = wrist - elbow

                # Elbow angle (dot product)
                dot = np.sum(upper_arm * forearm, axis=1)
                mag1 = np.sqrt(np.sum(upper_arm**2, axis=1))
                mag2 = np.sqrt(np.sum(forearm**2, axis=1))

                elbow_angle = np.arccos(np.clip(dot / (mag1 * mag2 + 1e-8), -1, 1))

                features = elbow_angle.reshape(-1, 1)

                result = nested_cv_test(features, y_player)

                if not np.isnan(result['r2']) and result['r2'] > best_r2:
                    best_r2 = result['r2']
                    best_frame = frame

        indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
        print(f"  Player {pid}: R²={best_r2:+.4f} at frame {best_frame} {indicator}")

        results.append({
            'test': 'elbow_angle',
            'player': pid,
            'feature': 'elbow_bend',
            'frame': best_frame,
            'r2': best_r2
        })

    # ==========================================================================
    # Test 5: Wrist Velocity at Release (velocity direction = ball direction)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 5: Wrist Velocity Components (velocity direction = release angle)")
    print("=" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        best_r2 = -999
        best_frame = 0

        for frame in range(140, 180, 5):
            # Calculate velocity (position change over 5 frames)
            dt = 5
            if frame + dt >= X_player.shape[1]:
                continue

            if all(f"right_wrist_{a}" in col_to_idx for a in ['x', 'y', 'z']):
                wrist_t0 = np.column_stack([
                    X_player[:, frame, col_to_idx['right_wrist_x']],
                    X_player[:, frame, col_to_idx['right_wrist_y']],
                    X_player[:, frame, col_to_idx['right_wrist_z']]
                ])
                wrist_t1 = np.column_stack([
                    X_player[:, frame + dt, col_to_idx['right_wrist_x']],
                    X_player[:, frame + dt, col_to_idx['right_wrist_y']],
                    X_player[:, frame + dt, col_to_idx['right_wrist_z']]
                ])

                velocity = (wrist_t1 - wrist_t0) / dt

                # Velocity magnitude and direction
                vel_mag = np.sqrt(np.sum(velocity**2, axis=1))

                # Velocity angle (elevation angle - determines ball trajectory)
                vel_angle_z = np.arctan2(velocity[:, 2], np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2))

                # Velocity azimuth (horizontal direction)
                vel_azimuth = np.arctan2(velocity[:, 1], velocity[:, 0])

                features = np.column_stack([
                    velocity[:, 0],  # vx
                    velocity[:, 1],  # vy
                    velocity[:, 2],  # vz
                    vel_mag,
                    vel_angle_z,
                    vel_azimuth
                ])

                result = nested_cv_test(features, y_player)

                if not np.isnan(result['r2']) and result['r2'] > best_r2:
                    best_r2 = result['r2']
                    best_frame = frame

        indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
        print(f"  Player {pid}: R²={best_r2:+.4f} at frame {best_frame} {indicator}")

        results.append({
            'test': 'wrist_velocity',
            'player': pid,
            'feature': 'velocity_components',
            'frame': best_frame,
            'r2': best_r2
        })

    # ==========================================================================
    # Test 6: Full Kinetic Chain (multiple joints)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 6: Full Kinetic Chain (all arm joints)")
    print("=" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        best_r2 = -999
        best_frame = 0

        for frame in range(120, 180, 5):
            feature_list = []

            # All arm joints x, y, z
            for part in ['right_shoulder', 'right_elbow', 'right_wrist']:
                for axis in ['x', 'y', 'z']:
                    col = f"{part}_{axis}"
                    if col in col_to_idx:
                        feature_list.append(X_player[:, frame, col_to_idx[col]])

            if len(feature_list) < 9:
                continue

            features = np.column_stack(feature_list)

            result = nested_cv_test(features, y_player)

            if not np.isnan(result['r2']) and result['r2'] > best_r2:
                best_r2 = result['r2']
                best_frame = frame

        indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
        print(f"  Player {pid}: R²={best_r2:+.4f} at frame {best_frame} {indicator}")

        results.append({
            'test': 'full_kinetic_chain',
            'player': pid,
            'feature': 'all_arm_joints_xyz',
            'frame': best_frame,
            'r2': best_r2
        })

    # ==========================================================================
    # Test 7: Temporal features (change from preparation to release)
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 7: Temporal Features (preparation -> release change)")
    print("=" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = angle_target[mask]

        best_r2 = -999
        best_config = ""

        prep_frames = [80, 100, 120]
        release_frames = [150, 160, 170]

        for prep in prep_frames:
            for release in release_frames:
                feature_list = []

                # Change in wrist position
                for axis in ['x', 'y', 'z']:
                    col = f"right_wrist_{axis}"
                    if col in col_to_idx:
                        change = X_player[:, release, col_to_idx[col]] - X_player[:, prep, col_to_idx[col]]
                        feature_list.append(change)

                # Change in elbow position
                for axis in ['x', 'y', 'z']:
                    col = f"right_elbow_{axis}"
                    if col in col_to_idx:
                        change = X_player[:, release, col_to_idx[col]] - X_player[:, prep, col_to_idx[col]]
                        feature_list.append(change)

                if len(feature_list) < 6:
                    continue

                features = np.column_stack(feature_list)

                result = nested_cv_test(features, y_player)

                if not np.isnan(result['r2']) and result['r2'] > best_r2:
                    best_r2 = result['r2']
                    best_config = f"prep={prep}, release={release}"

        indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
        print(f"  Player {pid}: R²={best_r2:+.4f} ({best_config}) {indicator}")

        results.append({
            'test': 'temporal_change',
            'player': pid,
            'feature': 'prep_to_release',
            'frame': best_config,
            'r2': best_r2
        })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY: BEST ANGLE PHYSICS BY PLAYER")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for pid in sorted(np.unique(participant_ids)):
        player_results = results_df[results_df['player'] == pid]
        if len(player_results) > 0:
            best = player_results.loc[player_results['r2'].idxmax()]
            print(f"\nPlayer {pid}: Best R² = {best['r2']:.4f}")
            print(f"  Test: {best['test']}")
            print(f"  Feature: {best['feature']}")
            print(f"  Frame: {best['frame']}")

    # Overall statistics
    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    positive = results_df[results_df['r2'] > 0]
    strong = results_df[results_df['r2'] > 0.05]

    print(f"\nPositive R² results: {len(positive)}/{len(results_df)}")
    print(f"Strong R² (>0.05): {len(strong)}/{len(results_df)}")

    # By test type
    print("\nBy test type:")
    for test_type in results_df['test'].unique():
        test_results = results_df[results_df['test'] == test_type]
        mean_r2 = test_results['r2'].mean()
        max_r2 = test_results['r2'].max()
        pos_count = (test_results['r2'] > 0).sum()
        print(f"  {test_type:25} Mean R²={mean_r2:+.4f}, Max R²={max_r2:+.4f}, Positive={pos_count}/5")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'angle_xyz_investigation.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'angle_xyz_investigation.csv'}")


if __name__ == "__main__":
    main()
