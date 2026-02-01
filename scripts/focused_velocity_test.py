"""
FOCUSED VELOCITY TEST

Previous tests show:
- Single frame features: negative test R²
- Multi-frame velocity features: negative test R²

This test tries an even more focused approach:
1. Very simple 1-2 feature velocity models
2. Per-player optimal velocity window search
3. Extremely regularized models
4. Test if ANY frame combination yields positive signal
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


class KeypointExtractor:
    def __init__(self, keypoint_cols: List[str]):
        self.keypoint_cols = keypoint_cols
        self.col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    def get(self, ts: np.ndarray, name: str, frame: int = None) -> np.ndarray:
        idx = self.col_to_idx.get(name)
        if idx is None:
            return np.nan if frame is not None else np.full(ts.shape[0], np.nan)
        if frame is not None:
            if frame < 0 or frame >= ts.shape[0]:
                return np.nan
            return ts[frame, idx]
        return ts[:, idx]


def per_player_cv_with_high_reg(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                                  alpha: float = 100.0, n_splits: int = 5) -> Tuple[float, float]:
    """Within-player CV with very high regularization."""
    all_test_r2 = []

    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = y[mask]

        valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
        X_clean = X_player[valid]
        y_clean = y_player[valid]

        if len(y_clean) < n_splits * 2:
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_clean):
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)

            all_test_r2.append(model.score(X_test_scaled, y_test))

    return np.mean(all_test_r2), np.std(all_test_r2)


def per_player_cv_detail(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                         alpha: float = 100.0, n_splits: int = 5) -> Dict[int, float]:
    """Get CV result per player."""
    results = {}

    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = y[mask]

        valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
        X_clean = X_player[valid]
        y_clean = y_player[valid]

        if len(y_clean) < n_splits * 2:
            results[pid] = np.nan
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        test_r2s = []

        for train_idx, test_idx in kf.split(X_clean):
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)
            test_r2s.append(model.score(X_test_scaled, y_test))

        results[pid] = np.mean(test_r2s)

    return results


def main():
    print("=" * 80)
    print("FOCUSED VELOCITY TEST")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    angle_target = y[:, 0]
    participant_ids = meta['participant_id'].values

    print(f"Loaded {len(angle_target)} shots from {len(np.unique(participant_ids))} players")
    print()

    # ==========================================================================
    # TEST 1: Exhaustive search for positive single-velocity features
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: Exhaustive Search - All Velocity Windows (single feature)")
    print("=" * 80)
    print()
    print("Testing ALL combinations of start frame, end frame, and keypoint")
    print("with high regularization (alpha=100) to avoid overfitting")
    print()

    keypoints = [
        'right_wrist_z', 'left_wrist_z',
        'right_elbow_z', 'left_elbow_z',
        'right_knee_z', 'left_knee_z',
        'right_ankle_z', 'left_ankle_z',
        'right_hip_z', 'left_hip_z',
        'right_shoulder_z', 'left_shoulder_z',
    ]

    all_results = []

    for kp_name in keypoints:
        for start in range(60, 180, 10):
            for end in range(start + 5, min(start + 60, 220), 5):
                # Compute velocity for all samples
                velocities = []
                for i in range(len(X)):
                    val_start = kp.get(X[i], kp_name, start)
                    val_end = kp.get(X[i], kp_name, end)
                    if np.isnan(val_start) or np.isnan(val_end):
                        velocities.append(np.nan)
                    else:
                        velocities.append((val_end - val_start) / (end - start))

                X_vel = np.array(velocities).reshape(-1, 1)

                valid_pct = (~np.isnan(X_vel)).mean()
                if valid_pct < 0.9:
                    continue

                mean_r2, std_r2 = per_player_cv_with_high_reg(X_vel, angle_target, participant_ids, alpha=100.0)

                all_results.append({
                    'Keypoint': kp_name,
                    'Start': start,
                    'End': end,
                    'Window': end - start,
                    'Test R²': mean_r2,
                    'Std': std_r2,
                })

    results_df = pd.DataFrame(all_results).sort_values('Test R²', ascending=False)

    print("Top 20 velocity features (single feature, high regularization):")
    print(results_df.head(20).to_string(index=False))
    print()

    best_r2 = results_df['Test R²'].max()
    print(f"Best single-velocity Test R²: {best_r2:.4f}")

    # Check how many are positive
    positive_count = (results_df['Test R²'] > 0).sum()
    print(f"Positive Test R²: {positive_count} / {len(results_df)}")
    print()

    # ==========================================================================
    # TEST 2: Per-player optimal velocity search
    # ==========================================================================
    print("=" * 80)
    print("TEST 2: Per-Player Optimal Velocity Window")
    print("=" * 80)
    print()
    print("Finding the best velocity window for EACH player separately")
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]

        best_r2_player = -999
        best_config = None

        for kp_name in ['right_wrist_z', 'right_elbow_z', 'right_knee_z', 'right_ankle_z']:
            for start in range(80, 170, 10):
                for end in range(start + 10, min(start + 50, 200), 10):
                    velocities = []
                    for i in np.where(mask)[0]:
                        val_start = kp.get(X[i], kp_name, start)
                        val_end = kp.get(X[i], kp_name, end)
                        if np.isnan(val_start) or np.isnan(val_end):
                            velocities.append(np.nan)
                        else:
                            velocities.append((val_end - val_start) / (end - start))

                    X_vel = np.array(velocities).reshape(-1, 1)

                    valid = ~(np.isnan(X_vel.ravel()) | np.isnan(y_player))
                    X_clean = X_vel[valid]
                    y_clean = y_player[valid]

                    if len(y_clean) < 20:
                        continue

                    # 5-fold CV within this player
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    test_r2s = []

                    for train_idx, test_idx in kf.split(X_clean):
                        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
                        y_train, y_test = y_clean[train_idx], y_clean[test_idx]

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        model = Ridge(alpha=100.0)
                        model.fit(X_train_scaled, y_train)
                        test_r2s.append(model.score(X_test_scaled, y_test))

                    mean_r2 = np.mean(test_r2s)
                    if mean_r2 > best_r2_player:
                        best_r2_player = mean_r2
                        best_config = (kp_name, start, end, np.std(test_r2s))

        if best_config:
            print(f"Player {pid}: Best velocity = {best_config[0]} [{best_config[1]}-{best_config[2]}]")
            print(f"           Test R² = {best_r2_player:.4f} (±{best_config[3]:.4f})")
        else:
            print(f"Player {pid}: No valid configuration found")
        print()

    # ==========================================================================
    # TEST 3: Test simple position differences (not velocity - raw difference)
    # ==========================================================================
    print("=" * 80)
    print("TEST 3: Raw Position Differences (Frame B - Frame A)")
    print("=" * 80)
    print()
    print("Testing if raw position change (not normalized by time) has signal")
    print()

    position_results = []

    for kp_name in ['right_wrist_z', 'right_elbow_z', 'right_knee_z']:
        for frame_a in [100, 120, 140, 150, 153]:
            for frame_b in [155, 160, 165, 170, 180]:
                if frame_b <= frame_a:
                    continue

                diffs = []
                for i in range(len(X)):
                    val_a = kp.get(X[i], kp_name, frame_a)
                    val_b = kp.get(X[i], kp_name, frame_b)
                    if np.isnan(val_a) or np.isnan(val_b):
                        diffs.append(np.nan)
                    else:
                        diffs.append(val_b - val_a)

                X_diff = np.array(diffs).reshape(-1, 1)

                valid_pct = (~np.isnan(X_diff)).mean()
                if valid_pct < 0.9:
                    continue

                mean_r2, std_r2 = per_player_cv_with_high_reg(X_diff, angle_target, participant_ids, alpha=100.0)

                position_results.append({
                    'Keypoint': kp_name,
                    'Frame A': frame_a,
                    'Frame B': frame_b,
                    'Test R²': mean_r2,
                    'Std': std_r2,
                })

    pos_df = pd.DataFrame(position_results).sort_values('Test R²', ascending=False)
    print("Top 15 position difference features:")
    print(pos_df.head(15).to_string(index=False))
    print()

    # ==========================================================================
    # TEST 4: Combining 2 best velocity features
    # ==========================================================================
    print("=" * 80)
    print("TEST 4: Combining Top 2 Velocity Features")
    print("=" * 80)
    print()

    # Get top 5 velocity configs
    top_configs = results_df.head(5)[['Keypoint', 'Start', 'End']].values.tolist()

    from itertools import combinations

    combo_results = []
    for config1, config2 in combinations(top_configs, 2):
        # Extract both features
        features_1 = []
        features_2 = []

        for i in range(len(X)):
            # Feature 1
            val_start = kp.get(X[i], config1[0], config1[1])
            val_end = kp.get(X[i], config1[0], config1[2])
            if np.isnan(val_start) or np.isnan(val_end):
                features_1.append(np.nan)
            else:
                features_1.append((val_end - val_start) / (config1[2] - config1[1]))

            # Feature 2
            val_start = kp.get(X[i], config2[0], config2[1])
            val_end = kp.get(X[i], config2[0], config2[2])
            if np.isnan(val_start) or np.isnan(val_end):
                features_2.append(np.nan)
            else:
                features_2.append((val_end - val_start) / (config2[2] - config2[1]))

        X_combo = np.column_stack([features_1, features_2])

        valid_pct = (~np.isnan(X_combo).any(axis=1)).mean()
        if valid_pct < 0.9:
            continue

        mean_r2, std_r2 = per_player_cv_with_high_reg(X_combo, angle_target, participant_ids, alpha=100.0)

        combo_results.append({
            'Config 1': f"{config1[0]}[{config1[1]}-{config1[2]}]",
            'Config 2': f"{config2[0]}[{config2[1]}-{config2[2]}]",
            'Test R²': mean_r2,
            'Std': std_r2,
        })

    combo_df = pd.DataFrame(combo_results).sort_values('Test R²', ascending=False)
    print("Top 5 two-feature combinations:")
    print(combo_df.head(5).to_string(index=False))
    print()

    # ==========================================================================
    # TEST 5: Different regularization strengths
    # ==========================================================================
    print("=" * 80)
    print("TEST 5: Effect of Regularization Strength")
    print("=" * 80)
    print()

    # Use the best single velocity feature
    best_row = results_df.iloc[0]
    best_kp = best_row['Keypoint']
    best_start = int(best_row['Start'])
    best_end = int(best_row['End'])

    velocities = []
    for i in range(len(X)):
        val_start = kp.get(X[i], best_kp, best_start)
        val_end = kp.get(X[i], best_kp, best_end)
        if np.isnan(val_start) or np.isnan(val_end):
            velocities.append(np.nan)
        else:
            velocities.append((val_end - val_start) / (best_end - best_start))

    X_best = np.array(velocities).reshape(-1, 1)

    print(f"Best feature: {best_kp}[{best_start}-{best_end}]")
    print()
    print("Regularization strength vs Test R²:")

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        mean_r2, std_r2 = per_player_cv_with_high_reg(X_best, angle_target, participant_ids, alpha=alpha)
        print(f"  alpha={alpha:>8.2f}: Test R² = {mean_r2:.4f} (±{std_r2:.4f})")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    best_velocity_r2 = results_df['Test R²'].max()
    best_position_r2 = pos_df['Test R²'].max() if len(pos_df) > 0 else np.nan
    best_combo_r2 = combo_df['Test R²'].max() if len(combo_df) > 0 else np.nan

    print(f"Best single velocity feature:   Test R² = {best_velocity_r2:.4f}")
    print(f"Best position difference:       Test R² = {best_position_r2:.4f}")
    print(f"Best 2-feature combination:     Test R² = {best_combo_r2:.4f}")
    print()

    overall_best = max(best_velocity_r2, best_position_r2, best_combo_r2)

    if overall_best < 0:
        print("=" * 80)
        print("CONCLUSION: NO POSITIVE TEST R² FOUND")
        print("=" * 80)
        print()
        print("Even with:")
        print("- Exhaustive frame combination search")
        print("- Per-player optimization")
        print("- High regularization (alpha=100-10000)")
        print("- Single and double feature models")
        print()
        print("All velocity/motion features show NEGATIVE test R².")
        print()
        print("This means shot-to-shot variation within each player is:")
        print("LARGER than any physics-based signal we can extract.")
        print()
        print("The physics IS there, but it's swamped by:")
        print("1. Natural shot-to-shot variability in technique")
        print("2. Noise in motion capture data")
        print("3. Too few samples per player (~70) to learn the pattern")
    else:
        print(f"FOUND POSITIVE TEST R²: {overall_best:.4f}")
        print("There may be some usable signal.")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'focused_velocity_results.csv', index=False)
    pos_df.to_csv(OUTPUT_DIR / 'position_difference_results.csv', index=False)

    print()
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
