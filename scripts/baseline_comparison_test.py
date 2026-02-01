"""
BASELINE COMPARISON TEST

Goal: Find where physics features beat the baseline
Compare physics features against what sub9 actually uses.

Also test all 3 targets (angle, depth, left_right) - maybe some targets
have more physics signal than others.
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
from advanced_features import extract_advanced_features as extract_baseline_features
from advanced_features import init_keypoint_mapping as init_adv_mapping
from hybrid_features import extract_hybrid_features, init_keypoint_mapping as init_hybrid_mapping

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


def extract_simple_physics_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """Extract very simple physics features - positions and velocities at key frames."""
    features = {}

    key_frames = [100, 120, 140, 150, 153, 160, 170, 180]
    key_kps = ['right_wrist_z', 'right_elbow_z', 'right_knee_z', 'right_ankle_z', 'right_hip_z']

    # Positions at key frames
    for kp_name in key_kps:
        for frame in key_frames:
            features[f'{kp_name}_f{frame}'] = kp.get(ts, kp_name, frame)

    # Velocities between key frame pairs
    velocity_pairs = [(100, 120), (120, 140), (140, 160), (150, 170), (153, 180)]
    for kp_name in key_kps:
        for start, end in velocity_pairs:
            val_start = kp.get(ts, kp_name, start)
            val_end = kp.get(ts, kp_name, end)
            if not np.isnan(val_start) and not np.isnan(val_end):
                features[f'{kp_name}_vel_{start}_{end}'] = (val_end - val_start) / (end - start)
            else:
                features[f'{kp_name}_vel_{start}_{end}'] = np.nan

    # Simple joint angles (knee angle, elbow angle)
    for frame in [150, 153, 160]:
        # Knee angle proxy
        hip_z = kp.get(ts, 'right_hip_z', frame)
        knee_z = kp.get(ts, 'right_knee_z', frame)
        ankle_z = kp.get(ts, 'right_ankle_z', frame)
        if not np.isnan(hip_z) and not np.isnan(knee_z) and not np.isnan(ankle_z):
            features[f'knee_angle_f{frame}'] = (hip_z - knee_z) + (ankle_z - knee_z)
        else:
            features[f'knee_angle_f{frame}'] = np.nan

        # Elbow angle proxy
        shoulder_z = kp.get(ts, 'right_shoulder_z', frame)
        elbow_z = kp.get(ts, 'right_elbow_z', frame)
        wrist_z = kp.get(ts, 'right_wrist_z', frame)
        if not np.isnan(shoulder_z) and not np.isnan(elbow_z) and not np.isnan(wrist_z):
            features[f'elbow_angle_f{frame}'] = (shoulder_z - elbow_z) + (wrist_z - elbow_z)
        else:
            features[f'elbow_angle_f{frame}'] = np.nan

    return features


def within_player_cv(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                     alpha: float = 1.0, n_splits: int = 5) -> Tuple[float, float]:
    """Within-player CV."""
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


def per_player_cv(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                  alpha: float = 1.0, n_splits: int = 5) -> Dict[int, Tuple[float, float]]:
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
            results[pid] = (np.nan, np.nan)
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

        results[pid] = (np.mean(test_r2s), np.std(test_r2s))

    return results


def main():
    print("=" * 80)
    print("BASELINE COMPARISON TEST")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2],
    }

    print(f"Loaded {len(y)} shots from {len(np.unique(participant_ids))} players")
    print()

    # Initialize keypoint mappings
    init_adv_mapping(keypoint_cols)
    init_hybrid_mapping(keypoint_cols)

    # ==========================================================================
    # Extract features
    # ==========================================================================
    print("Extracting features...")

    # Baseline features (what sub9 uses)
    baseline_features = []
    for i in range(len(X)):
        baseline_features.append(extract_baseline_features(X[i]))
    df_baseline = pd.DataFrame(baseline_features)

    # Hybrid features
    hybrid_features = []
    for i in range(len(X)):
        hybrid_features.append(extract_hybrid_features(X[i]))
    df_hybrid = pd.DataFrame(hybrid_features)

    # Simple physics features
    physics_features = []
    for i in range(len(X)):
        physics_features.append(extract_simple_physics_features(X[i], kp))
    df_physics = pd.DataFrame(physics_features)

    print(f"Baseline features: {df_baseline.shape[1]}")
    print(f"Hybrid features: {df_hybrid.shape[1]}")
    print(f"Physics features: {df_physics.shape[1]}")
    print()

    # Clean NaN columns
    df_baseline_clean = df_baseline.dropna(axis=1, how='all')
    df_hybrid_clean = df_hybrid.dropna(axis=1, how='all')
    df_physics_clean = df_physics.dropna(axis=1, how='all')

    # ==========================================================================
    # TEST 1: Compare feature sets across ALL targets
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: Feature Sets vs All Targets (Within-Player 5-Fold CV)")
    print("=" * 80)
    print()

    feature_sets = [
        ('Baseline (sub9)', df_baseline_clean),
        ('Hybrid', df_hybrid_clean),
        ('Simple Physics', df_physics_clean),
    ]

    for target_name, target in targets.items():
        print(f"\n{target_name.upper()}:")
        print("-" * 60)

        for feat_name, df in feature_sets:
            X_feat = df.values
            mean_r2, std_r2 = within_player_cv(X_feat, target, participant_ids, alpha=1.0)
            print(f"  {feat_name}: Test R² = {mean_r2:.4f} (±{std_r2:.4f})")

    # ==========================================================================
    # TEST 2: Per-player comparison for angle target
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 2: Per-Player Comparison (Angle Target)")
    print("=" * 80)
    print()

    print("Test R² by Player:")
    print("-" * 70)
    print(f"{'Player':<10}", end="")
    for feat_name, _ in feature_sets:
        print(f"{feat_name:<25}", end="")
    print()
    print("-" * 70)

    for pid in sorted(np.unique(participant_ids)):
        print(f"Player {pid:<3}", end="")

        for feat_name, df in feature_sets:
            mask = participant_ids == pid
            X_player = df.values[mask]
            y_player = targets['angle'][mask]

            valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
            X_clean = X_player[valid]
            y_clean = y_player[valid]

            if len(y_clean) < 10:
                print(f"{'N/A':<25}", end="")
                continue

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            test_r2s = []

            for train_idx, test_idx in kf.split(X_clean):
                X_train, X_test = X_clean[train_idx], X_clean[test_idx]
                y_train, y_test = y_clean[train_idx], y_clean[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)
                test_r2s.append(model.score(X_test_scaled, y_test))

            mean_r2 = np.mean(test_r2s)
            print(f"{mean_r2:>+.4f}                ", end="")

        print()

    # ==========================================================================
    # TEST 3: Find individual physics features that beat baseline per player
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 3: Individual Physics Features Per Player (Angle Target)")
    print("=" * 80)
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = targets['angle'][mask]

        print(f"\nPlayer {pid}:")
        print("-" * 50)

        # Get baseline score for this player
        X_baseline_player = df_baseline_clean.values[mask]
        valid_b = ~(np.isnan(X_baseline_player).any(axis=1) | np.isnan(y_player))

        if valid_b.sum() < 10:
            print("  Not enough valid samples")
            continue

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        baseline_r2s = []

        for train_idx, test_idx in kf.split(np.where(valid_b)[0]):
            train_mask = np.zeros(len(valid_b), dtype=bool)
            test_mask = np.zeros(len(valid_b), dtype=bool)
            train_mask[np.where(valid_b)[0][train_idx]] = True
            test_mask[np.where(valid_b)[0][test_idx]] = True

            X_train = X_baseline_player[train_mask]
            X_test = X_baseline_player[test_mask]
            y_train = y_player[train_mask]
            y_test = y_player[test_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            baseline_r2s.append(model.score(X_test_scaled, y_test))

        baseline_mean = np.mean(baseline_r2s)
        print(f"  Baseline Test R²: {baseline_mean:.4f}")

        # Test individual physics features
        positive_features = []
        for col in df_physics_clean.columns:
            X_single = df_physics_clean[[col]].values[mask]

            valid = ~(np.isnan(X_single.ravel()) | np.isnan(y_player))
            X_clean = X_single[valid]
            y_clean = y_player[valid]

            if len(y_clean) < 10:
                continue

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            test_r2s = []

            for train_idx, test_idx in kf.split(X_clean):
                X_train, X_test = X_clean[train_idx], X_clean[test_idx]
                y_train, y_test = y_clean[train_idx], y_clean[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)
                test_r2s.append(model.score(X_test_scaled, y_test))

            mean_r2 = np.mean(test_r2s)
            if mean_r2 > 0:
                positive_features.append((col, mean_r2, np.std(test_r2s)))

        if positive_features:
            positive_features.sort(key=lambda x: x[1], reverse=True)
            print(f"  Positive physics features ({len(positive_features)} found):")
            for feat, r2, std in positive_features[:5]:
                indicator = " <<<" if r2 > baseline_mean else ""
                print(f"    {feat}: {r2:.4f} (±{std:.4f}){indicator}")
        else:
            print("  No positive physics features found")

    # ==========================================================================
    # TEST 4: Test depth and left_right targets
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 4: Physics Features for Depth and Left-Right")
    print("=" * 80)

    for target_name in ['depth', 'left_right']:
        print(f"\n{target_name.upper()}:")
        print("-" * 60)

        target = targets[target_name]

        # Overall
        for feat_name, df in [('Baseline', df_baseline_clean), ('Physics', df_physics_clean)]:
            mean_r2, std_r2 = within_player_cv(df.values, target, participant_ids, alpha=1.0)
            print(f"  {feat_name}: Test R² = {mean_r2:.4f} (±{std_r2:.4f})")

        # Per player
        print()
        print("  Per player (Physics only):")
        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = df_physics_clean.values[mask]
            y_player = target[mask]

            valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
            X_clean = X_player[valid]
            y_clean = y_player[valid]

            if len(y_clean) < 10:
                print(f"    Player {pid}: N/A")
                continue

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            test_r2s = []

            for train_idx, test_idx in kf.split(X_clean):
                X_train, X_test = X_clean[train_idx], X_clean[test_idx]
                y_train, y_test = y_clean[train_idx], y_clean[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)
                test_r2s.append(model.score(X_test_scaled, y_test))

            mean_r2 = np.mean(test_r2s)
            indicator = " <<<" if mean_r2 > 0 else ""
            print(f"    Player {pid}: Test R² = {mean_r2:.4f}{indicator}")

    # ==========================================================================
    # TEST 5: Combine physics + baseline features
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 5: Physics + Baseline Combined")
    print("=" * 80)
    print()

    # Combine features
    df_combined = pd.concat([df_baseline_clean, df_physics_clean], axis=1)
    print(f"Combined features: {df_combined.shape[1]}")
    print()

    for target_name, target in targets.items():
        baseline_r2, _ = within_player_cv(df_baseline_clean.values, target, participant_ids)
        combined_r2, _ = within_player_cv(df_combined.values, target, participant_ids)

        improvement = combined_r2 - baseline_r2
        indicator = " BETTER" if improvement > 0 else ""
        print(f"{target_name}:")
        print(f"  Baseline only: {baseline_r2:.4f}")
        print(f"  Combined:      {combined_r2:.4f} (diff: {improvement:+.4f}){indicator}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)


if __name__ == "__main__":
    main()
