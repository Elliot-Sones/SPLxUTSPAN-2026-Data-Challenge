"""
PHYSICS MODEL TEST

Use the discovered player-specific physics features to build a model
and compare against baseline.

This tests whether the physics findings translate to actual improvements.
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
from advanced_features import extract_advanced_features, init_keypoint_mapping

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


# Optimal features discovered from deep dive analysis
PLAYER_OPTIMAL_FEATURES = {
    # DEPTH features
    'depth': {
        1: [
            ('right_wrist_z', 120, 'position'),
            ('left_elbow_z', 80, 120, 'velocity'),
            ('right_elbow_z', 80, 120, 'velocity'),
        ],
        2: [
            ('right_shoulder_z', 120, 'position'),
            ('left_elbow_z', 80, 120, 'velocity'),
            ('right_wrist_z', 80, 120, 'velocity'),
        ],
        3: [
            ('right_shoulder_z', 120, 'position'),
            ('right_hip_z', 120, 'position'),
            ('right_elbow_z', 80, 120, 'velocity'),
        ],
        4: [
            ('right_shoulder_z', 120, 'position'),
            ('left_elbow_z', 80, 120, 'velocity'),
            ('right_wrist_z', 80, 120, 'velocity'),
            ('right_elbow_z', 80, 120, 'velocity'),
            ('right_hip_z', 120, 'position'),
        ],
        5: [
            ('right_hip_z', 150, 160, 'velocity'),
            ('left_hip_z', 150, 160, 'velocity'),
            ('left_shoulder_z', 150, 160, 'velocity'),
            ('right_hip_z', 150, 'position'),
            ('right_shoulder_z', 150, 'position'),
        ],
    },
    # ANGLE features
    'angle': {
        1: [
            ('left_elbow_z', 110, 120, 'velocity'),
            ('left_wrist_z', 110, 115, 'velocity'),
            ('right_elbow_z', 110, 125, 'velocity'),
        ],
        2: [
            ('right_knee_z', 'knee_angle', 80, 'joint_angle'),
            ('right_knee_z', 110, 115, 'velocity'),
            ('right_ankle_z', 110, 115, 'velocity'),
        ],
        3: [
            ('right_ankle_z', 180, 220, 'velocity'),
            ('right_ankle_z', 180, 210, 'velocity'),
        ],
        4: [
            ('right_elbow_z', 170, 175, 'velocity'),
            ('right_shoulder_z', 170, 180, 'velocity'),
            ('left_elbow_z', 170, 175, 'velocity'),
        ],
        5: [
            # Player 5 has no positive features for angle - use depth features as proxy
            ('right_hip_z', 150, 160, 'velocity'),
            ('left_hip_z', 150, 160, 'velocity'),
        ],
    },
    # LEFT_RIGHT features (weak signal, use generic)
    'left_right': {
        1: [('right_wrist_z', 153, 'position')],
        2: [('right_wrist_z', 153, 'position')],
        3: [('right_wrist_z', 153, 'position')],
        4: [('right_wrist_z', 153, 'position')],
        5: [('right_wrist_z', 153, 'position')],
    }
}


def extract_player_optimal_features(ts: np.ndarray, kp: KeypointExtractor,
                                    player_id: int, target: str) -> Dict[str, float]:
    """Extract optimal features for a specific player and target."""
    features = {}

    optimal = PLAYER_OPTIMAL_FEATURES.get(target, {}).get(player_id, [])

    for feat_spec in optimal:
        if len(feat_spec) == 3:
            # Position feature
            kp_name, frame, feat_type = feat_spec
            if feat_type == 'position':
                features[f'{kp_name}_f{frame}'] = kp.get(ts, kp_name, frame)
        elif len(feat_spec) == 4:
            if feat_spec[3] == 'velocity':
                # Velocity feature
                kp_name, start, end, feat_type = feat_spec
                val_start = kp.get(ts, kp_name, start)
                val_end = kp.get(ts, kp_name, end)
                if not np.isnan(val_start) and not np.isnan(val_end):
                    features[f'{kp_name}_vel_{start}_{end}'] = (val_end - val_start) / (end - start)
                else:
                    features[f'{kp_name}_vel_{start}_{end}'] = np.nan
            elif feat_spec[3] == 'joint_angle':
                # Joint angle
                _, _, frame, _ = feat_spec
                hip = kp.get(ts, 'right_hip_z', frame)
                knee = kp.get(ts, 'right_knee_z', frame)
                ankle = kp.get(ts, 'right_ankle_z', frame)
                if not (np.isnan(hip) or np.isnan(knee) or np.isnan(ankle)):
                    features[f'knee_angle_f{frame}'] = (hip - knee) + (ankle - knee)
                else:
                    features[f'knee_angle_f{frame}'] = np.nan

    return features


def extract_generic_physics_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """Extract generic physics features (for comparison)."""
    features = {}

    key_kps = ['right_wrist_z', 'right_elbow_z', 'right_knee_z', 'right_hip_z']

    for kp_name in key_kps:
        for frame in [100, 120, 140, 153, 160, 180]:
            features[f'{kp_name}_f{frame}'] = kp.get(ts, kp_name, frame)

        for start, end in [(80, 120), (120, 160), (150, 180)]:
            val_start = kp.get(ts, kp_name, start)
            val_end = kp.get(ts, kp_name, end)
            if not np.isnan(val_start) and not np.isnan(val_end):
                features[f'{kp_name}_vel_{start}_{end}'] = (val_end - val_start) / (end - start)
            else:
                features[f'{kp_name}_vel_{start}_{end}'] = np.nan

    return features


def within_player_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Within-player CV."""
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid]
    y_clean = y[valid]

    if len(y_clean) < n_splits * 2:
        return np.nan, np.nan, np.nan, np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_r2s = []
    test_r2s = []

    for train_idx, test_idx in kf.split(X_clean):
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train, y_test = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        train_r2s.append(model.score(X_train_scaled, y_train))
        test_r2s.append(model.score(X_test_scaled, y_test))

    return np.mean(train_r2s), np.std(train_r2s), np.mean(test_r2s), np.std(test_r2s)


def main():
    print("=" * 80)
    print("PHYSICS MODEL TEST")
    print("=" * 80)
    print()
    print("Testing player-specific optimal physics features vs generic vs baseline")
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)
    init_keypoint_mapping(keypoint_cols)

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2],
    }

    print(f"Loaded {len(y)} shots from {len(np.unique(participant_ids))} players")
    print()

    # ==========================================================================
    # Extract features for each approach
    # ==========================================================================
    print("Extracting features...")

    # Baseline features
    baseline_features = []
    for i in range(len(X)):
        baseline_features.append(extract_advanced_features(X[i]))
    df_baseline = pd.DataFrame(baseline_features).dropna(axis=1, how='all')

    # Generic physics features
    generic_features = []
    for i in range(len(X)):
        generic_features.append(extract_generic_physics_features(X[i], kp))
    df_generic = pd.DataFrame(generic_features).dropna(axis=1, how='all')

    print(f"Baseline features: {df_baseline.shape[1]}")
    print(f"Generic physics features: {df_generic.shape[1]}")
    print()

    # ==========================================================================
    # TEST: Compare approaches per player per target
    # ==========================================================================
    print("=" * 80)
    print("COMPARISON: Baseline vs Generic Physics vs Optimal Physics")
    print("=" * 80)
    print()

    results = []

    for target_name, target in targets.items():
        print(f"\n{'='*80}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*80}")

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            y_player = target[mask]

            # Extract optimal features for this player/target
            optimal_features = []
            for i in np.where(mask)[0]:
                optimal_features.append(extract_player_optimal_features(X[i], kp, pid, target_name))
            df_optimal = pd.DataFrame(optimal_features)

            # Baseline
            X_baseline = df_baseline.values[mask]
            _, _, base_r2, base_std = within_player_cv(X_baseline, y_player, alpha=1.0)

            # Generic physics
            X_generic = df_generic.values[mask]
            _, _, gen_r2, gen_std = within_player_cv(X_generic, y_player, alpha=1.0)

            # Optimal physics
            X_optimal = df_optimal.values
            _, _, opt_r2, opt_std = within_player_cv(X_optimal, y_player, alpha=1.0)

            # Determine best
            best = 'Baseline'
            best_r2 = base_r2
            if gen_r2 > best_r2 or (np.isnan(best_r2) and not np.isnan(gen_r2)):
                best = 'Generic'
                best_r2 = gen_r2
            if opt_r2 > best_r2 or (np.isnan(best_r2) and not np.isnan(opt_r2)):
                best = 'Optimal'
                best_r2 = opt_r2

            indicator = " <<<" if opt_r2 > base_r2 and not np.isnan(opt_r2) else ""

            print(f"\nPlayer {pid}:")
            print(f"  Baseline ({df_baseline.shape[1]} feats): Test R² = {base_r2:.4f}")
            print(f"  Generic  ({df_generic.shape[1]} feats): Test R² = {gen_r2:.4f}")
            print(f"  Optimal  ({df_optimal.shape[1]} feats): Test R² = {opt_r2:.4f}{indicator}")
            print(f"  BEST: {best}")

            results.append({
                'Target': target_name,
                'Player': pid,
                'Baseline R²': base_r2,
                'Generic R²': gen_r2,
                'Optimal R²': opt_r2,
                'Best': best,
                'Optimal Beats Baseline': opt_r2 > base_r2 if not (np.isnan(opt_r2) or np.isnan(base_r2)) else False,
            })

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    results_df = pd.DataFrame(results)

    # Count wins
    wins = results_df.groupby('Best').size()
    print("Wins by approach:")
    for approach, count in wins.items():
        print(f"  {approach}: {count}")
    print()

    # Where optimal beats baseline
    optimal_wins = results_df[results_df['Optimal Beats Baseline']]
    print(f"Cases where Optimal Physics beats Baseline: {len(optimal_wins)} / {len(results_df)}")
    if len(optimal_wins) > 0:
        print()
        print("Details:")
        for _, row in optimal_wins.iterrows():
            improvement = row['Optimal R²'] - row['Baseline R²']
            print(f"  {row['Target']} Player {row['Player']}: {row['Baseline R²']:.4f} -> {row['Optimal R²']:.4f} (+{improvement:.4f})")

    # Per target summary
    print()
    print("Per-target summary:")
    for target in ['angle', 'depth', 'left_right']:
        target_data = results_df[results_df['Target'] == target]
        opt_wins = target_data['Optimal Beats Baseline'].sum()
        print(f"  {target}: Optimal beats Baseline in {opt_wins}/{len(target_data)} players")

    # Save
    results_df.to_csv(OUTPUT_DIR / 'physics_model_comparison.csv', index=False)
    print()
    print(f"Results saved to {OUTPUT_DIR / 'physics_model_comparison.csv'}")


if __name__ == "__main__":
    main()
