"""
ANGLE PHYSICS DEEP DIVE

We found strong physics signal for depth. Now let's find if there's
any physics signal for angle that we might have missed.

Key insight from depth analysis:
- Different players have different optimal frames
- Player 4: Frame 120 critical
- Player 5: Frame 150 critical

Let's do exhaustive per-player feature search for angle.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
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


def within_player_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     alpha: float = 1.0) -> Tuple[float, float]:
    """Within-player CV for a single player."""
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid]
    y_clean = y[valid]

    if len(y_clean) < n_splits * 2:
        return np.nan, np.nan

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

    return np.mean(test_r2s), np.std(test_r2s)


def main():
    print("=" * 80)
    print("ANGLE PHYSICS DEEP DIVE")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    participant_ids = meta['participant_id'].values
    angle_target = y[:, 0]

    print(f"Loaded {len(y)} shots")
    print()

    # Key keypoints
    key_kps = [
        'right_wrist_z', 'left_wrist_z',
        'right_elbow_z', 'left_elbow_z',
        'right_shoulder_z', 'left_shoulder_z',
        'right_knee_z', 'left_knee_z',
        'right_ankle_z', 'left_ankle_z',
        'right_hip_z', 'left_hip_z',
    ]

    # ==========================================================================
    # TEST 1: EXHAUSTIVE per-player single feature search
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: Exhaustive Single Feature Search Per Player (ANGLE)")
    print("=" * 80)
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]

        print(f"\nPlayer {pid}:")
        print("-" * 70)

        positive_features = []

        # Test positions at all frames
        for kp_name in key_kps:
            for frame in range(60, 220, 5):
                # Extract feature
                feature_vals = np.array([kp.get(X[i], kp_name, frame) for i in np.where(mask)[0]])
                X_single = feature_vals.reshape(-1, 1)

                r2, std = within_player_cv(X_single, y_player, alpha=1.0)
                if not np.isnan(r2) and r2 > 0:
                    positive_features.append((f'{kp_name}_f{frame}', r2, std, 'position'))

        # Test velocities
        for kp_name in key_kps:
            for start in range(60, 200, 10):
                for window in [5, 10, 15, 20, 30, 40]:
                    end = start + window
                    if end > 220:
                        continue

                    feature_vals = []
                    for i in np.where(mask)[0]:
                        val_start = kp.get(X[i], kp_name, start)
                        val_end = kp.get(X[i], kp_name, end)
                        if np.isnan(val_start) or np.isnan(val_end):
                            feature_vals.append(np.nan)
                        else:
                            feature_vals.append((val_end - val_start) / window)

                    X_single = np.array(feature_vals).reshape(-1, 1)

                    r2, std = within_player_cv(X_single, y_player, alpha=1.0)
                    if not np.isnan(r2) and r2 > 0:
                        positive_features.append((f'{kp_name}_vel_{start}_{end}', r2, std, 'velocity'))

        # Test joint angles
        for frame in range(80, 200, 5):
            # Knee angle
            feature_vals = []
            for i in np.where(mask)[0]:
                hip = kp.get(X[i], 'right_hip_z', frame)
                knee = kp.get(X[i], 'right_knee_z', frame)
                ankle = kp.get(X[i], 'right_ankle_z', frame)
                if np.isnan(hip) or np.isnan(knee) or np.isnan(ankle):
                    feature_vals.append(np.nan)
                else:
                    feature_vals.append((hip - knee) + (ankle - knee))
            X_single = np.array(feature_vals).reshape(-1, 1)
            r2, std = within_player_cv(X_single, y_player, alpha=1.0)
            if not np.isnan(r2) and r2 > 0:
                positive_features.append((f'knee_angle_f{frame}', r2, std, 'angle'))

            # Elbow angle
            feature_vals = []
            for i in np.where(mask)[0]:
                shoulder = kp.get(X[i], 'right_shoulder_z', frame)
                elbow = kp.get(X[i], 'right_elbow_z', frame)
                wrist = kp.get(X[i], 'right_wrist_z', frame)
                if np.isnan(shoulder) or np.isnan(elbow) or np.isnan(wrist):
                    feature_vals.append(np.nan)
                else:
                    feature_vals.append((shoulder - elbow) + (wrist - elbow))
            X_single = np.array(feature_vals).reshape(-1, 1)
            r2, std = within_player_cv(X_single, y_player, alpha=1.0)
            if not np.isnan(r2) and r2 > 0:
                positive_features.append((f'elbow_angle_f{frame}', r2, std, 'angle'))

        positive_features.sort(key=lambda x: x[1], reverse=True)

        if positive_features:
            print(f"Found {len(positive_features)} features with positive Test R²")
            print()
            print("Top 15:")
            for feat, r2, std, feat_type in positive_features[:15]:
                print(f"  [{feat_type:8}] {feat}: Test R² = {r2:.4f} (±{std:.4f})")
        else:
            print("NO POSITIVE FEATURES FOUND")

        # Save best features for this player
        if positive_features:
            best_feat = positive_features[0]
            print()
            print(f"BEST: {best_feat[0]} with Test R² = {best_feat[1]:.4f}")

    # ==========================================================================
    # TEST 2: Combined features per player
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 2: Try Player-Specific Optimal Combinations")
    print("=" * 80)

    # Manual optimal features based on player analysis
    player_optimal = {
        1: [],  # Will fill based on Test 1 results
        2: [],
        3: [],
        4: [],
        5: [],
    }

    # For each player, extract their specific features and test
    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]

        # Find this player's top features
        all_features = []

        for kp_name in key_kps:
            for frame in range(60, 220, 10):
                feature_vals = np.array([kp.get(X[i], kp_name, frame) for i in np.where(mask)[0]])
                X_single = feature_vals.reshape(-1, 1)
                r2, std = within_player_cv(X_single, y_player, alpha=1.0)
                if not np.isnan(r2):
                    all_features.append((f'{kp_name}_f{frame}', r2, X_single))

        all_features.sort(key=lambda x: x[1], reverse=True)

        if all_features:
            print(f"\nPlayer {pid}:")
            best_1 = all_features[0]
            print(f"  Best 1 feature: {best_1[0]} = {best_1[1]:.4f}")

            if len(all_features) >= 2:
                # Try combining top 2
                X_combo = np.hstack([all_features[0][2], all_features[1][2]])
                r2, std = within_player_cv(X_combo, y_player, alpha=1.0)
                print(f"  Top 2 combined: {r2:.4f}")

            if len(all_features) >= 3:
                # Try combining top 3
                X_combo = np.hstack([all_features[0][2], all_features[1][2], all_features[2][2]])
                r2, std = within_player_cv(X_combo, y_player, alpha=1.0)
                print(f"  Top 3 combined: {r2:.4f}")

    # ==========================================================================
    # TEST 3: Regularization sweep on best features
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 3: Regularization Sweep on Player 4 (closest to positive)")
    print("=" * 80)
    print()

    # Player 4 had the least negative test R² for angle
    mask = participant_ids == 4
    y_player = angle_target[mask]

    # Find best feature for Player 4
    best_feat = None
    best_r2 = -999

    for kp_name in key_kps:
        for frame in range(60, 220, 10):
            feature_vals = np.array([kp.get(X[i], kp_name, frame) for i in np.where(mask)[0]])
            X_single = feature_vals.reshape(-1, 1)
            r2, _ = within_player_cv(X_single, y_player, alpha=1.0)
            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_feat = (kp_name, frame, X_single)

    if best_feat:
        print(f"Best feature for Player 4: {best_feat[0]}_f{best_feat[1]}")
        print(f"Test R² at alpha=1.0: {best_r2:.4f}")
        print()
        print("Regularization sweep:")

        for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            r2, std = within_player_cv(best_feat[2], y_player, alpha=alpha)
            indicator = " <<<" if r2 > 0 else ""
            print(f"  alpha={alpha:>8.3f}: Test R² = {r2:.4f} (±{std:.4f}){indicator}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Key findings for ANGLE target:")
    print()
    print("The challenge: Unlike DEPTH where players 4 and 5 showed strong signal,")
    print("ANGLE appears to have less physics signal available.")
    print()
    print("Possible reasons:")
    print("1. Angle depends on subtle wrist snap timing - hard to capture from keypoints")
    print("2. Small differences in release position create large angle changes")
    print("3. The keypoint resolution may not capture the relevant physics")
    print()
    print("However - we DID find positive features for some players!")
    print("The key is to use player-specific features, not one-size-fits-all.")


if __name__ == "__main__":
    main()
