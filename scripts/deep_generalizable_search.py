"""
Deep Search for Generalizable Features

More extensive search including:
1. All frames in fine granularity
2. All body part combinations
3. Feature interactions
4. Temporal patterns
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

TARGETS = ['angle', 'depth', 'left_right']

TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def compute_scaled_mse(preds, truth, target_name):
    scaler = TARGET_SCALERS[target_name]
    scaled_preds = np.clip(scaler.transform(preds.reshape(-1, 1)).flatten(), 0, 1)
    scaled_truth = np.clip(scaler.transform(truth.reshape(-1, 1)).flatten(), 0, 1)
    return np.mean((scaled_preds - scaled_truth) ** 2)


def lopo_cv(X_features, y_target, pids, target_name, alpha=10.0):
    """Leave-One-Player-Out CV."""
    unique_pids = sorted(np.unique(pids))
    all_preds = np.zeros(len(y_target))

    for holdout_pid in unique_pids:
        train_mask = pids != holdout_pid
        test_mask = pids == holdout_pid

        X_train = np.nan_to_num(X_features[train_mask], nan=0.0)
        X_test = np.nan_to_num(X_features[test_mask], nan=0.0)
        y_train = y_target[train_mask]

        std = np.std(X_train, axis=0)
        valid_cols = std > 1e-10
        if valid_cols.sum() == 0:
            all_preds[test_mask] = np.mean(y_train)
            continue

        X_train = X_train[:, valid_cols]
        X_test = X_test[:, valid_cols]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)
        all_preds[test_mask] = model.predict(X_test_s)

    return compute_scaled_mse(all_preds, y_target, target_name)


def main():
    print("=" * 70)
    print("DEEP SEARCH FOR GENERALIZABLE FEATURES")
    print("=" * 70)

    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}
    pids = meta['participant_id'].values

    # All available body parts
    all_parts = list(set([col.rsplit('_', 1)[0] for col in keypoint_cols if col.endswith('_x')]))
    print(f"Total body parts: {len(all_parts)}")

    best_configs = {}

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n{'='*70}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*70}")

        y_target = y[:, target_idx]

        baseline_mse = compute_scaled_mse(np.full(len(y_target), np.mean(y_target)), y_target, target_name)
        print(f"Baseline LOPO MSE: {baseline_mse:.6f}")

        best_mse = baseline_mse
        best_config = None
        best_features = None

        # Search 1: Single body part, all frames
        print("\n--- Single body parts (fine frame search) ---")
        for part in all_parts:
            cols = get_cols(col_to_idx, part)
            if not cols:
                continue

            for frame in range(80, 200, 5):  # Fine granularity
                features = X[:, frame, cols]
                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse - 0.0001:  # Meaningful improvement
                    best_mse = mse
                    best_config = f"{part}_f{frame}"
                    best_features = features.copy()
                    print(f"  NEW BEST: {best_config}: {mse:.6f} ({(baseline_mse-mse)/baseline_mse*100:.1f}% vs baseline)")

        # Search 2: Velocity features
        print("\n--- Velocity features ---")
        key_parts = ['right_wrist', 'right_elbow', 'left_wrist', 'right_shoulder', 'right_knee', 'right_hip']
        for part in key_parts:
            cols = get_cols(col_to_idx, part)
            if not cols:
                continue

            for frame in range(100, 180, 10):
                for window in [3, 5, 7, 10, 15]:
                    if frame + window >= X.shape[1]:
                        continue

                    pos = X[:, frame, cols]
                    vel = (X[:, frame + window, cols] - X[:, frame, cols]) / window
                    features = np.hstack([pos, vel])

                    mse = lopo_cv(features, y_target, pids, target_name)

                    if mse < best_mse - 0.0001:
                        best_mse = mse
                        best_config = f"{part}_f{frame}_posvel_w{window}"
                        best_features = features.copy()
                        print(f"  NEW BEST: {best_config}: {mse:.6f}")

        # Search 3: Body part pairs
        print("\n--- Body part pairs ---")
        important_pairs = [
            ('right_wrist', 'right_elbow'),
            ('right_wrist', 'left_wrist'),
            ('right_elbow', 'right_shoulder'),
            ('right_shoulder', 'left_shoulder'),
            ('right_hip', 'left_hip'),
            ('right_wrist', 'right_hip'),
            ('right_shoulder', 'right_hip'),
            ('neck', 'mid_hip'),
        ]

        for p1, p2 in important_pairs:
            cols1 = get_cols(col_to_idx, p1)
            cols2 = get_cols(col_to_idx, p2)
            if not cols1 or not cols2:
                continue

            for frame in range(100, 180, 10):
                features = np.hstack([X[:, frame, cols1], X[:, frame, cols2]])
                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse - 0.0001:
                    best_mse = mse
                    best_config = f"{p1}+{p2}_f{frame}"
                    best_features = features.copy()
                    print(f"  NEW BEST: {best_config}: {mse:.6f}")

        # Search 4: Difference features (asymmetry)
        print("\n--- Asymmetry features ---")
        lr_pairs = [
            ('right_wrist', 'left_wrist'),
            ('right_elbow', 'left_elbow'),
            ('right_shoulder', 'left_shoulder'),
            ('right_hip', 'left_hip'),
            ('right_knee', 'left_knee'),
        ]

        for frame in range(100, 180, 10):
            diff_features = []
            for r, l in lr_pairs:
                for axis in ['x', 'y', 'z']:
                    rc = f"{r}_{axis}"
                    lc = f"{l}_{axis}"
                    if rc in col_to_idx and lc in col_to_idx:
                        diff = X[:, frame, col_to_idx[rc]] - X[:, frame, col_to_idx[lc]]
                        diff_features.append(diff)

            if diff_features:
                features = np.column_stack(diff_features)
                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse - 0.0001:
                    best_mse = mse
                    best_config = f"lr_asymmetry_f{frame}"
                    best_features = features.copy()
                    print(f"  NEW BEST: {best_config}: {mse:.6f}")

        # Search 5: Multi-frame temporal
        print("\n--- Multi-frame temporal ---")
        for part in ['right_wrist', 'right_elbow', 'neck']:
            cols = get_cols(col_to_idx, part)
            if not cols:
                continue

            for start in range(100, 160, 10):
                frames = [start, start + 10, start + 20, start + 30]
                frames = [f for f in frames if f < X.shape[1]]

                features = np.hstack([X[:, f, cols] for f in frames])
                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse - 0.0001:
                    best_mse = mse
                    best_config = f"{part}_temporal_{start}"
                    best_features = features.copy()
                    print(f"  NEW BEST: {best_config}: {mse:.6f}")

        # Search 6: Z-coordinate statistics (like sub9)
        print("\n--- Z-coordinate statistics ---")
        for part in ['right_wrist', 'right_elbow', 'right_shoulder', 'right_knee', 'right_hip', 'neck']:
            z_col = f"{part}_z"
            if z_col not in col_to_idx:
                continue

            z_series = X[:, :, col_to_idx[z_col]]

            # Statistics over time
            z_mean = np.nanmean(z_series, axis=1)
            z_std = np.nanstd(z_series, axis=1)
            z_max = np.nanmax(z_series, axis=1)
            z_min = np.nanmin(z_series, axis=1)
            z_range = z_max - z_min

            features = np.column_stack([z_mean, z_std, z_max, z_min, z_range])
            mse = lopo_cv(features, y_target, pids, target_name)

            if mse < best_mse - 0.0001:
                best_mse = mse
                best_config = f"{part}_z_stats"
                best_features = features.copy()
                print(f"  NEW BEST: {best_config}: {mse:.6f}")

        # Search 7: Combined Z stats from multiple parts
        print("\n--- Combined Z statistics ---")
        parts_for_z = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_knee', 'right_hip']
        all_z_features = []
        for part in parts_for_z:
            z_col = f"{part}_z"
            if z_col not in col_to_idx:
                continue
            z_series = X[:, :, col_to_idx[z_col]]
            all_z_features.extend([
                np.nanmean(z_series, axis=1),
                np.nanstd(z_series, axis=1),
                np.nanmax(z_series, axis=1),
            ])

        if all_z_features:
            features = np.column_stack(all_z_features)
            mse = lopo_cv(features, y_target, pids, target_name)

            if mse < best_mse - 0.0001:
                best_mse = mse
                best_config = "combined_z_stats"
                best_features = features.copy()
                print(f"  NEW BEST: {best_config}: {mse:.6f}")

        print(f"\n*** BEST for {target_name}: {best_config}")
        print(f"    LOPO MSE: {best_mse:.6f}")
        print(f"    vs baseline: {(baseline_mse - best_mse) / baseline_mse * 100:.1f}%")

        best_configs[target_name] = {
            'config': best_config,
            'mse': best_mse,
            'baseline': baseline_mse,
            'improvement': (baseline_mse - best_mse) / baseline_mse * 100
        }

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_mse = 0
    for target_name in TARGETS:
        cfg = best_configs[target_name]
        print(f"\n{target_name.upper()}:")
        print(f"  Config: {cfg['config']}")
        print(f"  LOPO MSE: {cfg['mse']:.6f}")
        print(f"  Improvement: {cfg['improvement']:.1f}%")
        total_mse += cfg['mse']

    avg_mse = total_mse / 3
    print(f"\n{'='*70}")
    print(f"OVERALL AVERAGE LOPO MSE: {avg_mse:.6f}")
    print(f"Sub9 LB Score: 0.009109")
    print(f"Gap: {(avg_mse - 0.009109) / 0.009109 * 100:.1f}%")


if __name__ == "__main__":
    main()
