"""
Find Generalizable Features

Use Leave-One-Player-Out CV as the metric to find features with TRUE signal.
Features that work across all players = real physics signal, not noise.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from itertools import combinations
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


def lopo_cv(X_features, y_target, pids, target_name):
    """
    Leave-One-Player-Out CV - the TRUE generalization test.
    Returns MSE that reflects how well features generalize.
    """
    unique_pids = sorted(np.unique(pids))
    all_preds = np.zeros(len(y_target))

    for holdout_pid in unique_pids:
        train_mask = pids != holdout_pid
        test_mask = pids == holdout_pid

        X_train = X_features[train_mask]
        X_test = X_features[test_mask]
        y_train = y_target[train_mask]

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Remove constant features
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

        # Use high regularization to prevent overfitting
        model = Ridge(alpha=10.0)
        model.fit(X_train_s, y_train)
        all_preds[test_mask] = model.predict(X_test_s)

    return compute_scaled_mse(all_preds, y_target, target_name)


def extract_simple_features(X, col_to_idx, body_parts, frame, add_velocity=False, vel_window=5):
    """Extract simple position (and optionally velocity) features."""
    features = []

    for part in body_parts:
        cols = get_cols(col_to_idx, part)
        if cols:
            features.append(X[:, frame, cols])

            if add_velocity and frame + vel_window < X.shape[1]:
                vel = (X[:, frame + vel_window, cols] - X[:, frame, cols]) / vel_window
                features.append(vel)

    if features:
        return np.hstack(features)
    return None


def main():
    print("=" * 70)
    print("FINDING GENERALIZABLE FEATURES")
    print("Using Leave-One-Player-Out CV to find TRUE signal")
    print("=" * 70)

    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}
    pids = meta['participant_id'].values

    # Key body parts to test (physics-based)
    body_parts = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'left_elbow', 'left_shoulder',
        'right_hip', 'left_hip', 'right_knee', 'right_ankle',
        'neck', 'nose', 'mid_hip',
    ]
    body_parts = [p for p in body_parts if any(f"{p}_{a}" in col_to_idx for a in ['x', 'y', 'z'])]

    # Frames to test
    frames = [100, 110, 120, 130, 140, 150, 160, 170, 180]

    results = []

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n{'='*70}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*70}")

        y_target = y[:, target_idx]

        # Baseline: predict mean
        baseline_preds = np.full(len(y_target), np.mean(y_target))
        baseline_mse = compute_scaled_mse(baseline_preds, y_target, target_name)
        print(f"\nBaseline (mean prediction) LOPO MSE: {baseline_mse:.6f}")

        best_mse = baseline_mse
        best_config = "baseline"

        # Test 1: Single body parts
        print("\n--- Testing single body parts ---")
        for part in body_parts:
            for frame in frames:
                features = extract_simple_features(X, col_to_idx, [part], frame)
                if features is None:
                    continue

                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse:
                    best_mse = mse
                    best_config = f"{part}_f{frame}"
                    print(f"  NEW BEST: {part} @ frame {frame}: LOPO MSE = {mse:.6f}")

                results.append({
                    'target': target_name,
                    'config': f"{part}_f{frame}",
                    'lopo_mse': mse,
                    'vs_baseline': (baseline_mse - mse) / baseline_mse * 100
                })

        # Test 2: Single body parts with velocity
        print("\n--- Testing single body parts + velocity ---")
        for part in ['right_wrist', 'right_elbow', 'left_wrist', 'right_knee']:
            for frame in [120, 140, 150, 160]:
                for vel_window in [3, 5, 10]:
                    features = extract_simple_features(X, col_to_idx, [part], frame,
                                                       add_velocity=True, vel_window=vel_window)
                    if features is None:
                        continue

                    mse = lopo_cv(features, y_target, pids, target_name)

                    if mse < best_mse:
                        best_mse = mse
                        best_config = f"{part}_f{frame}_vel{vel_window}"
                        print(f"  NEW BEST: {part} @ frame {frame} + vel(w={vel_window}): LOPO MSE = {mse:.6f}")

                    results.append({
                        'target': target_name,
                        'config': f"{part}_f{frame}_vel{vel_window}",
                        'lopo_mse': mse,
                        'vs_baseline': (baseline_mse - mse) / baseline_mse * 100
                    })

        # Test 3: Two body parts (arm chain)
        print("\n--- Testing body part pairs ---")
        pairs = [
            ('right_wrist', 'right_elbow'),
            ('right_wrist', 'right_shoulder'),
            ('right_elbow', 'right_shoulder'),
            ('right_wrist', 'left_wrist'),
            ('right_hip', 'right_knee'),
            ('right_shoulder', 'right_hip'),
        ]

        for p1, p2 in pairs:
            for frame in [140, 150, 160]:
                features = extract_simple_features(X, col_to_idx, [p1, p2], frame)
                if features is None:
                    continue

                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse:
                    best_mse = mse
                    best_config = f"{p1}+{p2}_f{frame}"
                    print(f"  NEW BEST: {p1}+{p2} @ frame {frame}: LOPO MSE = {mse:.6f}")

                results.append({
                    'target': target_name,
                    'config': f"{p1}+{p2}_f{frame}",
                    'lopo_mse': mse,
                    'vs_baseline': (baseline_mse - mse) / baseline_mse * 100
                })

        # Test 4: Arm chain (3 parts)
        print("\n--- Testing arm chain ---")
        for frame in [140, 150, 160]:
            features = extract_simple_features(X, col_to_idx,
                                               ['right_wrist', 'right_elbow', 'right_shoulder'],
                                               frame)
            if features is not None:
                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse:
                    best_mse = mse
                    best_config = f"arm_chain_f{frame}"
                    print(f"  NEW BEST: arm_chain @ frame {frame}: LOPO MSE = {mse:.6f}")

                results.append({
                    'target': target_name,
                    'config': f"arm_chain_f{frame}",
                    'lopo_mse': mse,
                    'vs_baseline': (baseline_mse - mse) / baseline_mse * 100
                })

        # Test 5: Body alignment (left-right differences)
        print("\n--- Testing body alignment (L-R differences) ---")
        for frame in [140, 150, 160, 170]:
            lr_features = []
            lr_pairs = [
                ('right_wrist', 'left_wrist'),
                ('right_elbow', 'left_elbow'),
                ('right_shoulder', 'left_shoulder'),
                ('right_hip', 'left_hip'),
            ]
            for right, left in lr_pairs:
                for axis in ['x', 'y', 'z']:
                    r_col = f"{right}_{axis}"
                    l_col = f"{left}_{axis}"
                    if r_col in col_to_idx and l_col in col_to_idx:
                        diff = X[:, frame, col_to_idx[r_col]] - X[:, frame, col_to_idx[l_col]]
                        lr_features.append(diff)

            if lr_features:
                features = np.column_stack(lr_features)
                mse = lopo_cv(features, y_target, pids, target_name)

                if mse < best_mse:
                    best_mse = mse
                    best_config = f"body_alignment_f{frame}"
                    print(f"  NEW BEST: body_alignment @ frame {frame}: LOPO MSE = {mse:.6f}")

                results.append({
                    'target': target_name,
                    'config': f"body_alignment_f{frame}",
                    'lopo_mse': mse,
                    'vs_baseline': (baseline_mse - mse) / baseline_mse * 100
                })

        # Test 6: Multi-frame features (temporal)
        print("\n--- Testing multi-frame (temporal) ---")
        for part in ['right_wrist', 'right_elbow']:
            for start_frame in [140, 150]:
                frames_to_use = [start_frame, start_frame + 5, start_frame + 10]
                multi_features = []
                cols = get_cols(col_to_idx, part)
                if cols:
                    for f in frames_to_use:
                        if f < X.shape[1]:
                            multi_features.append(X[:, f, cols])

                if multi_features:
                    features = np.hstack(multi_features)
                    mse = lopo_cv(features, y_target, pids, target_name)

                    if mse < best_mse:
                        best_mse = mse
                        best_config = f"{part}_multiframe_{start_frame}"
                        print(f"  NEW BEST: {part} multi-frame @ {start_frame}: LOPO MSE = {mse:.6f}")

                    results.append({
                        'target': target_name,
                        'config': f"{part}_multiframe_{start_frame}",
                        'lopo_mse': mse,
                        'vs_baseline': (baseline_mse - mse) / baseline_mse * 100
                    })

        print(f"\n*** BEST for {target_name}: {best_config} with LOPO MSE = {best_mse:.6f}")
        print(f"    Improvement over baseline: {(baseline_mse - best_mse) / baseline_mse * 100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: BEST GENERALIZABLE FEATURES")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    for target_name in TARGETS:
        target_results = results_df[results_df['target'] == target_name]
        best = target_results.loc[target_results['lopo_mse'].idxmin()]

        print(f"\n{target_name.upper()}:")
        print(f"  Best config: {best['config']}")
        print(f"  LOPO MSE: {best['lopo_mse']:.6f}")
        print(f"  vs baseline: {best['vs_baseline']:.1f}% better")

        # Top 5
        print(f"  Top 5 configs:")
        top5 = target_results.nsmallest(5, 'lopo_mse')
        for _, row in top5.iterrows():
            print(f"    {row['config']}: {row['lopo_mse']:.6f} ({row['vs_baseline']:.1f}% vs baseline)")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'generalizable_features.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'generalizable_features.csv'}")

    # Overall comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    best_per_target = {}
    for target_name in TARGETS:
        target_results = results_df[results_df['target'] == target_name]
        best_per_target[target_name] = target_results['lopo_mse'].min()

    avg_lopo = np.mean(list(best_per_target.values()))

    print(f"\n  Previous physics features LOPO MSE: 0.044452")
    print(f"  New generalizable features LOPO MSE: {avg_lopo:.6f}")
    print(f"  Sub9 LB Score: 0.009109")

    if avg_lopo < 0.009109:
        print(f"\n  Generalizable features BEAT sub9 on LOPO!")
    else:
        print(f"\n  Generalizable features still worse than sub9 on LOPO")
        print(f"  Gap: {(avg_lopo - 0.009109) / 0.009109 * 100:.1f}%")


if __name__ == "__main__":
    main()
