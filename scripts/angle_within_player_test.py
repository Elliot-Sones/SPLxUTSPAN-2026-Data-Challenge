"""
ANGLE Within-Player Generalization Test

The previous test showed cross-player generalization is impossible.
This test checks: do features generalize to unseen shots from the SAME player?

This is the actual scenario for the competition:
- Train on ~55 shots from Player X
- Test on ~15 shots from Player X (different shots, same player)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'

# Import feature extractors from previous script
from angle_physics_test import (
    KeypointExtractor,
    extract_leg_drive_features,
    extract_arm_features,
    extract_timing_features,
    extract_posture_features,
    extract_datamined_baseline,
)


def within_player_cv(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                     n_splits: int = 5) -> Tuple[float, float, float, float]:
    """
    Cross-validation WITHIN each player, then average across players.

    This mimics the actual train/test scenario:
    - Train on some shots from a player
    - Test on other shots from the SAME player

    Returns: (mean_train_r2, std_train_r2, mean_test_r2, std_test_r2)
    """
    all_train_r2 = []
    all_test_r2 = []

    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = y[mask]

        # Remove NaN rows
        valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
        X_clean = X_player[valid]
        y_clean = y_player[valid]

        if len(y_clean) < n_splits * 2:
            continue

        # K-fold within this player
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        player_train_r2 = []
        player_test_r2 = []

        for train_idx, test_idx in kf.split(X_clean):
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            # Standardize using train stats
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)

            # Score
            train_r2 = model.score(X_train_scaled, y_train)
            test_r2 = model.score(X_test_scaled, y_test)

            player_train_r2.append(train_r2)
            player_test_r2.append(test_r2)

        all_train_r2.extend(player_train_r2)
        all_test_r2.extend(player_test_r2)

    return (
        np.mean(all_train_r2),
        np.std(all_train_r2),
        np.mean(all_test_r2),
        np.std(all_test_r2)
    )


def per_player_analysis(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                        group_name: str) -> pd.DataFrame:
    """Detailed analysis per player."""
    results = []

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = y[mask]

        # Remove NaN
        valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
        X_clean = X_player[valid]
        y_clean = y_player[valid]

        n_shots = len(y_clean)

        if n_shots < 10:
            continue

        # 5-fold CV within player
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        train_r2s = []
        test_r2s = []

        for train_idx, test_idx in kf.split(X_clean):
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)

            train_r2s.append(model.score(X_train_scaled, y_train))
            test_r2s.append(model.score(X_test_scaled, y_test))

        results.append({
            'Player': pid,
            'Feature Group': group_name,
            'N Shots': n_shots,
            'Train R² (mean)': np.mean(train_r2s),
            'Test R² (mean)': np.mean(test_r2s),
            'Gap': np.mean(train_r2s) - np.mean(test_r2s),
            'Generalization %': (np.mean(test_r2s) / np.mean(train_r2s) * 100)
                                if np.mean(train_r2s) > 0.01 else 0,
        })

    return pd.DataFrame(results)


def test_individual_features(X_df: pd.DataFrame, y: np.ndarray,
                             participant_ids: np.ndarray) -> pd.DataFrame:
    """Test each individual feature with within-player CV."""
    results = []

    for col in X_df.columns:
        X_single = X_df[[col]].values

        train_mean, train_std, test_mean, test_std = within_player_cv(
            X_single, y, participant_ids, n_splits=5
        )

        if not np.isnan(test_mean):
            results.append({
                'Feature': col,
                'Train R²': train_mean,
                'Test R²': test_mean,
                'Gap': train_mean - test_mean,
                'Generalization %': (test_mean / train_mean * 100) if train_mean > 0.01 else 0,
            })

    return pd.DataFrame(results).sort_values('Test R²', ascending=False)


def main():
    print("=" * 80)
    print("ANGLE WITHIN-PLAYER GENERALIZATION TEST")
    print("=" * 80)
    print()
    print("This tests: do features generalize to UNSEEN SHOTS from the SAME PLAYER?")
    print("This is the actual competition scenario.")
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    angle_target = y[:, 0]
    participant_ids = meta['participant_id'].values

    print(f"Loaded {len(angle_target)} shots")
    print()

    # Extract features
    print("Extracting features...")
    all_leg_drive = []
    all_arm = []
    all_timing = []
    all_posture = []
    all_baseline = []

    for i in range(len(X)):
        ts = X[i]
        all_leg_drive.append(extract_leg_drive_features(ts, kp))
        all_arm.append(extract_arm_features(ts, kp))
        all_timing.append(extract_timing_features(ts, kp))
        all_posture.append(extract_posture_features(ts, kp))
        all_baseline.append(extract_datamined_baseline(ts, kp))

    df_leg_drive = pd.DataFrame(all_leg_drive)
    df_arm = pd.DataFrame(all_arm)
    df_timing = pd.DataFrame(all_timing)
    df_posture = pd.DataFrame(all_posture)
    df_baseline = pd.DataFrame(all_baseline)
    df_all_physics = pd.concat([df_leg_drive, df_arm, df_timing, df_posture], axis=1)

    print("Features extracted.")
    print()

    # ==========================================================================
    # TEST 1: Within-player CV for each feature group
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: Within-Player Cross-Validation (5-fold per player)")
    print("=" * 80)
    print()

    results = []

    for name, df in [
        ('Leg Drive (11 features)', df_leg_drive),
        ('Arm (12 features)', df_arm),
        ('Timing (8 features)', df_timing),
        ('Posture (6 features)', df_posture),
        ('Baseline data-mined (30 features)', df_baseline),
        ('All Physics Combined (37 features)', df_all_physics),
    ]:
        X_group = df.values
        train_mean, train_std, test_mean, test_std = within_player_cv(
            X_group, angle_target, participant_ids, n_splits=5
        )

        gap = train_mean - test_mean
        gen_pct = (test_mean / train_mean * 100) if train_mean > 0.01 else 0

        results.append({
            'Feature Group': name,
            'Train R²': f"{train_mean:.4f} (±{train_std:.4f})",
            'Test R²': f"{test_mean:.4f} (±{test_std:.4f})",
            'Gap': gap,
            'Generalization %': gen_pct,
        })

        print(f"{name}:")
        print(f"  Train R²: {train_mean:.4f} (±{train_std:.4f})")
        print(f"  Test R²:  {test_mean:.4f} (±{test_std:.4f})")
        print(f"  Gap: {gap:.4f}")
        print(f"  Generalization: {gen_pct:.1f}%")
        print()

    # ==========================================================================
    # TEST 2: Per-player breakdown
    # ==========================================================================
    print("=" * 80)
    print("TEST 2: Per-Player Breakdown")
    print("=" * 80)
    print()

    all_player_results = []

    for name, df in [
        ('Leg Drive', df_leg_drive),
        ('Arm', df_arm),
        ('Timing', df_timing),
        ('Posture', df_posture),
        ('Baseline', df_baseline),
    ]:
        player_df = per_player_analysis(df.values, angle_target, participant_ids, name)
        all_player_results.append(player_df)

    player_results_df = pd.concat(all_player_results, ignore_index=True)

    # Pivot for display
    pivot = player_results_df.pivot(index='Player', columns='Feature Group', values='Test R²')
    print("Test R² by Player and Feature Group:")
    print(pivot.round(3).to_string())
    print()

    # Find best feature group per player
    print("Best Feature Group per Player:")
    for pid in sorted(player_results_df['Player'].unique()):
        player_data = player_results_df[player_results_df['Player'] == pid]
        best_row = player_data.loc[player_data['Test R²'].idxmax()]
        print(f"  Player {pid}: {best_row['Feature Group']} (Test R²={best_row['Test R²']:.3f})")
    print()

    # ==========================================================================
    # TEST 3: Individual feature analysis
    # ==========================================================================
    print("=" * 80)
    print("TEST 3: Top Individual Features (within-player CV)")
    print("=" * 80)
    print()

    all_features_df = pd.concat([df_all_physics, df_baseline], axis=1)
    feature_results = test_individual_features(all_features_df, angle_target, participant_ids)

    print("Top 15 features by TEST R² (within-player generalization):")
    print(feature_results.head(15).to_string(index=False))
    print()

    print("Bottom 10 features (worst generalization):")
    print(feature_results.tail(10).to_string(index=False))
    print()

    # ==========================================================================
    # TEST 4: Compare fewer features (simpler model)
    # ==========================================================================
    print("=" * 80)
    print("TEST 4: Effect of Feature Count on Generalization")
    print("=" * 80)
    print()
    print("Hypothesis: Fewer features = less overfitting = better generalization")
    print()

    # Sort features by test R²
    top_features = feature_results.head(20)['Feature'].tolist()

    for n_features in [1, 2, 3, 5, 10, 15, 20]:
        selected = top_features[:n_features]
        X_selected = all_features_df[selected].values

        train_mean, train_std, test_mean, test_std = within_player_cv(
            X_selected, angle_target, participant_ids, n_splits=5
        )

        gap = train_mean - test_mean
        print(f"Top {n_features} features: Train R²={train_mean:.4f}, Test R²={test_mean:.4f}, Gap={gap:.4f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Best overall result
    best_result = max(results, key=lambda x: float(x['Test R²'].split()[0]))
    print(f"Best Feature Group: {best_result['Feature Group']}")
    print(f"  Test R²: {best_result['Test R²']}")
    print(f"  Generalization: {best_result['Generalization %']:.1f}%")
    print()

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'angle_within_player_results.csv', index=False)
    feature_results.to_csv(OUTPUT_DIR / 'angle_individual_feature_results.csv', index=False)
    player_results_df.to_csv(OUTPUT_DIR / 'angle_per_player_results.csv', index=False)

    print(f"Results saved to {OUTPUT_DIR}")
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
