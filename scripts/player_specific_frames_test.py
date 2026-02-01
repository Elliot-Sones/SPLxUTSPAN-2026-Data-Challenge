"""
PLAYER-SPECIFIC FRAMES TEST

Goal: Determine if player-specific frame selection provides REAL signal
that generalizes, or if it's just overfitting noise.

Method: Nested CV
- Outer loop: Evaluate generalization (held-out test fold)
- Inner loop: Select optimal frame (on training fold only)

This prevents data leakage and gives realistic estimates.
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


def find_best_frame_on_train(X_train: np.ndarray, y_train: np.ndarray,
                              kp: KeypointExtractor, keypoint: str,
                              frame_range: Tuple[int, int],
                              feat_type: str = 'position') -> int:
    """Find best frame using only training data (inner CV)."""
    best_frame = frame_range[0]
    best_score = -999

    for frame in range(frame_range[0], frame_range[1], 5):
        if feat_type == 'position':
            features = np.array([kp.get(X_train[i], keypoint, frame)
                                for i in range(len(X_train))]).reshape(-1, 1)
        elif feat_type == 'velocity':
            features = []
            for i in range(len(X_train)):
                val_start = kp.get(X_train[i], keypoint, frame)
                val_end = kp.get(X_train[i], keypoint, frame + 10)
                if np.isnan(val_start) or np.isnan(val_end):
                    features.append(np.nan)
                else:
                    features.append(val_end - val_start)
            features = np.array(features).reshape(-1, 1)

        valid = ~(np.isnan(features.ravel()) | np.isnan(y_train))
        if valid.sum() < 10:
            continue

        X_clean = features[valid]
        y_clean = y_train[valid]

        # Inner 3-fold CV to evaluate this frame
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in kf.split(X_clean):
            X_tr, X_val = X_clean[tr_idx], X_clean[val_idx]
            y_tr, y_val = y_clean[tr_idx], y_clean[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            model = Ridge(alpha=1.0)
            model.fit(X_tr_s, y_tr)
            scores.append(model.score(X_val_s, y_val))

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_frame = frame

    return best_frame


def nested_cv_test(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                   kp: KeypointExtractor, keypoint: str,
                   frame_range: Tuple[int, int], fixed_frame: int,
                   feat_type: str = 'position', n_outer_splits: int = 5) -> Dict:
    """
    Nested CV test comparing:
    1. Fixed frame (e.g., frame 153 for all)
    2. Player-specific optimal frame (selected on training data only)
    """

    fixed_preds = np.zeros(len(y))
    optimal_preds = np.zeros(len(y))
    fixed_preds[:] = np.nan
    optimal_preds[:] = np.nan

    selected_frames = {}  # Track which frames were selected

    # Process each player separately
    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = y[mask]
        indices = np.where(mask)[0]

        if len(y_player) < 15:
            continue

        # Outer CV for this player
        kf_outer = KFold(n_splits=n_outer_splits, shuffle=True, random_state=42)

        player_selected_frames = []

        for fold, (train_idx, test_idx) in enumerate(kf_outer.split(X_player)):
            X_train = X_player[train_idx]
            X_test = X_player[test_idx]
            y_train = y_player[train_idx]
            y_test = y_player[test_idx]

            # ============================================================
            # Method 1: Fixed frame
            # ============================================================
            if feat_type == 'position':
                X_train_fixed = np.array([kp.get(X_train[i], keypoint, fixed_frame)
                                         for i in range(len(X_train))]).reshape(-1, 1)
                X_test_fixed = np.array([kp.get(X_test[i], keypoint, fixed_frame)
                                        for i in range(len(X_test))]).reshape(-1, 1)
            else:
                X_train_fixed = []
                for i in range(len(X_train)):
                    v1 = kp.get(X_train[i], keypoint, fixed_frame)
                    v2 = kp.get(X_train[i], keypoint, fixed_frame + 10)
                    X_train_fixed.append(v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)
                X_train_fixed = np.array(X_train_fixed).reshape(-1, 1)

                X_test_fixed = []
                for i in range(len(X_test)):
                    v1 = kp.get(X_test[i], keypoint, fixed_frame)
                    v2 = kp.get(X_test[i], keypoint, fixed_frame + 10)
                    X_test_fixed.append(v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)
                X_test_fixed = np.array(X_test_fixed).reshape(-1, 1)

            valid_train = ~(np.isnan(X_train_fixed.ravel()) | np.isnan(y_train))
            if valid_train.sum() >= 5:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train_fixed[valid_train])

                model = Ridge(alpha=1.0)
                model.fit(X_tr_s, y_train[valid_train])

                X_test_clean = np.nan_to_num(X_test_fixed, nan=0.0)
                X_te_s = scaler.transform(X_test_clean)
                preds = model.predict(X_te_s)

                for i, idx in enumerate(test_idx):
                    fixed_preds[indices[idx]] = preds[i]

            # ============================================================
            # Method 2: Optimal frame (selected on training data only)
            # ============================================================
            optimal_frame = find_best_frame_on_train(X_train, y_train, kp, keypoint,
                                                      frame_range, feat_type)
            player_selected_frames.append(optimal_frame)

            if feat_type == 'position':
                X_train_opt = np.array([kp.get(X_train[i], keypoint, optimal_frame)
                                       for i in range(len(X_train))]).reshape(-1, 1)
                X_test_opt = np.array([kp.get(X_test[i], keypoint, optimal_frame)
                                      for i in range(len(X_test))]).reshape(-1, 1)
            else:
                X_train_opt = []
                for i in range(len(X_train)):
                    v1 = kp.get(X_train[i], keypoint, optimal_frame)
                    v2 = kp.get(X_train[i], keypoint, optimal_frame + 10)
                    X_train_opt.append(v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)
                X_train_opt = np.array(X_train_opt).reshape(-1, 1)

                X_test_opt = []
                for i in range(len(X_test)):
                    v1 = kp.get(X_test[i], keypoint, optimal_frame)
                    v2 = kp.get(X_test[i], keypoint, optimal_frame + 10)
                    X_test_opt.append(v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)
                X_test_opt = np.array(X_test_opt).reshape(-1, 1)

            valid_train = ~(np.isnan(X_train_opt.ravel()) | np.isnan(y_train))
            if valid_train.sum() >= 5:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train_opt[valid_train])

                model = Ridge(alpha=1.0)
                model.fit(X_tr_s, y_train[valid_train])

                X_test_clean = np.nan_to_num(X_test_opt, nan=0.0)
                X_te_s = scaler.transform(X_test_clean)
                preds = model.predict(X_te_s)

                for i, idx in enumerate(test_idx):
                    optimal_preds[indices[idx]] = preds[i]

        selected_frames[pid] = player_selected_frames

    # Fill NaN with mean
    fixed_preds[np.isnan(fixed_preds)] = y.mean()
    optimal_preds[np.isnan(optimal_preds)] = y.mean()

    # Calculate MSE
    fixed_mse = np.mean((fixed_preds - y) ** 2)
    optimal_mse = np.mean((optimal_preds - y) ** 2)
    baseline_mse = np.mean((y - y.mean()) ** 2)  # Predicting mean

    return {
        'fixed_mse': fixed_mse,
        'optimal_mse': optimal_mse,
        'baseline_mse': baseline_mse,
        'fixed_r2': 1 - fixed_mse / baseline_mse,
        'optimal_r2': 1 - optimal_mse / baseline_mse,
        'selected_frames': selected_frames,
        'improvement': (fixed_mse - optimal_mse) / fixed_mse * 100,
    }


def main():
    print("=" * 80)
    print("PLAYER-SPECIFIC FRAMES TEST (Nested CV)")
    print("=" * 80)
    print()
    print("Testing if player-specific frame selection provides REAL signal")
    print("Using nested CV to prevent data leakage")
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2],
    }

    print(f"Data: {len(y)} samples, {len(np.unique(participant_ids))} players")
    print()

    # ==========================================================================
    # Test configurations
    # ==========================================================================

    test_configs = [
        # (keypoint, frame_range, fixed_frame, feat_type, target)
        ('right_wrist_z', (100, 200), 153, 'position', 'angle'),
        ('right_wrist_z', (100, 180), 140, 'velocity', 'angle'),
        ('right_elbow_z', (100, 200), 153, 'position', 'angle'),
        ('right_elbow_z', (100, 180), 140, 'velocity', 'angle'),

        ('right_knee_z', (80, 180), 102, 'position', 'depth'),
        ('right_knee_z', (80, 160), 100, 'velocity', 'depth'),
        ('right_hip_z', (80, 180), 120, 'position', 'depth'),
        ('right_hip_z', (100, 170), 120, 'velocity', 'depth'),

        ('right_shoulder_z', (100, 200), 153, 'position', 'left_right'),
        ('right_elbow_z', (100, 200), 153, 'position', 'left_right'),
    ]

    results = []

    for keypoint, frame_range, fixed_frame, feat_type, target_name in test_configs:
        print(f"\nTesting: {keypoint} ({feat_type}) for {target_name}")
        print(f"  Fixed frame: {fixed_frame}, Search range: {frame_range}")

        target = targets[target_name]

        result = nested_cv_test(X, target, participant_ids, kp, keypoint,
                               frame_range, fixed_frame, feat_type)

        print(f"  Fixed frame MSE:    {result['fixed_mse']:.4f} (R² = {result['fixed_r2']:.4f})")
        print(f"  Optimal frame MSE:  {result['optimal_mse']:.4f} (R² = {result['optimal_r2']:.4f})")
        print(f"  Baseline (mean) MSE: {result['baseline_mse']:.4f}")
        print(f"  Improvement: {result['improvement']:+.2f}%")

        # Show selected frames per player
        print("  Selected frames per player (across folds):")
        for pid, frames in sorted(result['selected_frames'].items()):
            unique_frames = list(set(frames))
            consistency = len(unique_frames) == 1
            print(f"    Player {pid}: {unique_frames} {'(consistent)' if consistency else '(varies)'}")

        results.append({
            'keypoint': keypoint,
            'feat_type': feat_type,
            'target': target_name,
            'fixed_frame': fixed_frame,
            'fixed_mse': result['fixed_mse'],
            'optimal_mse': result['optimal_mse'],
            'fixed_r2': result['fixed_r2'],
            'optimal_r2': result['optimal_r2'],
            'improvement': result['improvement'],
        })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    results_df = pd.DataFrame(results)

    # Group by target
    for target_name in ['angle', 'depth', 'left_right']:
        target_results = results_df[results_df['target'] == target_name]
        print(f"\n{target_name.upper()}:")
        print("-" * 70)

        for _, row in target_results.iterrows():
            indicator = "BETTER" if row['improvement'] > 0 else "WORSE"
            print(f"  {row['keypoint']:20} ({row['feat_type']:8}): "
                  f"Fixed R²={row['fixed_r2']:+.4f}, Optimal R²={row['optimal_r2']:+.4f}, "
                  f"Diff={row['improvement']:+.1f}% [{indicator}]")

    # Overall conclusion
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    positive_improvements = results_df[results_df['improvement'] > 0]
    print(f"Configurations where player-specific frames help: {len(positive_improvements)}/{len(results_df)}")

    if len(positive_improvements) > 0:
        print()
        print("Best improvements:")
        best = positive_improvements.nlargest(3, 'improvement')
        for _, row in best.iterrows():
            print(f"  {row['target']}: {row['keypoint']} ({row['feat_type']}) - {row['improvement']:+.1f}%")

    # Check if any have positive R²
    positive_r2 = results_df[results_df['optimal_r2'] > 0]
    print()
    print(f"Configurations with positive R² (better than mean): {len(positive_r2)}/{len(results_df)}")

    if len(positive_r2) > 0:
        print("These features have REAL signal:")
        for _, row in positive_r2.iterrows():
            print(f"  {row['target']}: {row['keypoint']} ({row['feat_type']}) R²={row['optimal_r2']:.4f}")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'player_specific_frames_test.csv', index=False)
    print()
    print(f"Results saved to {OUTPUT_DIR / 'player_specific_frames_test.csv'}")


if __name__ == "__main__":
    main()
