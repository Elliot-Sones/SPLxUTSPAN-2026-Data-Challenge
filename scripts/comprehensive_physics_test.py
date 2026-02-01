"""
COMPREHENSIVE PHYSICS TEST

Test all combinations of:
- Player (1-5)
- Frame (search optimal per player)
- Feature Group (leg drive, arm, posture, timing)
- Target (angle, depth, left_right)
- Model (Ridge with different alphas)

Using nested CV to get unbiased estimates.
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

    def get(self, ts: np.ndarray, name: str, frame: int = None):
        idx = self.col_to_idx.get(name)
        if idx is None:
            return np.nan if frame is not None else np.full(ts.shape[0], np.nan)
        if frame is not None:
            if frame < 0 or frame >= ts.shape[0]:
                return np.nan
            return ts[frame, idx]
        return ts[:, idx]


# Feature groups based on basketball physics
FEATURE_GROUPS = {
    'leg_drive': {
        'keypoints': ['right_knee_z', 'left_knee_z', 'right_ankle_z', 'left_ankle_z', 'right_hip_z', 'left_hip_z'],
        'frame_range': (80, 170),
        'description': 'Leg push/extension during shot'
    },
    'arm_release': {
        'keypoints': ['right_wrist_z', 'left_wrist_z', 'right_elbow_z', 'left_elbow_z'],
        'frame_range': (120, 200),
        'description': 'Arm extension at release'
    },
    'shoulder': {
        'keypoints': ['right_shoulder_z', 'left_shoulder_z'],
        'frame_range': (100, 200),
        'description': 'Shoulder position/alignment'
    },
    'full_body': {
        'keypoints': ['right_hip_z', 'right_knee_z', 'right_elbow_z', 'right_wrist_z'],
        'frame_range': (100, 180),
        'description': 'Key joints in kinetic chain'
    },
}

FEATURE_TYPES = ['position', 'velocity']
MODELS = {
    'ridge_low': lambda: Ridge(alpha=0.1),
    'ridge_med': lambda: Ridge(alpha=1.0),
    'ridge_high': lambda: Ridge(alpha=10.0),
    'ridge_vhigh': lambda: Ridge(alpha=100.0),
}


def extract_group_features(ts: np.ndarray, kp: KeypointExtractor,
                           group_name: str, frame: int, feat_type: str) -> np.ndarray:
    """Extract features for a group at a specific frame."""
    group = FEATURE_GROUPS[group_name]
    features = []

    for keypoint in group['keypoints']:
        if feat_type == 'position':
            features.append(kp.get(ts, keypoint, frame))
        elif feat_type == 'velocity':
            v1 = kp.get(ts, keypoint, frame)
            v2 = kp.get(ts, keypoint, frame + 10)
            if np.isnan(v1) or np.isnan(v2):
                features.append(np.nan)
            else:
                features.append(v2 - v1)

    return np.array(features)


def find_optimal_frame_inner_cv(X_train: np.ndarray, y_train: np.ndarray,
                                 kp: KeypointExtractor, group_name: str,
                                 feat_type: str, model_fn) -> int:
    """Find optimal frame using inner CV on training data only."""
    group = FEATURE_GROUPS[group_name]
    frame_range = group['frame_range']

    best_frame = frame_range[0]
    best_score = -999

    for frame in range(frame_range[0], frame_range[1], 10):
        # Extract features at this frame
        features = np.array([extract_group_features(X_train[i], kp, group_name, frame, feat_type)
                            for i in range(len(X_train))])

        valid = ~(np.isnan(features).any(axis=1) | np.isnan(y_train))
        if valid.sum() < 10:
            continue

        X_clean = features[valid]
        y_clean = y_train[valid]

        # Inner 3-fold CV
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for tr_idx, val_idx in kf.split(X_clean):
            X_tr, X_val = X_clean[tr_idx], X_clean[val_idx]
            y_tr, y_val = y_clean[tr_idx], y_clean[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            model = model_fn()
            model.fit(X_tr_s, y_tr)
            scores.append(model.score(X_val_s, y_val))

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_frame = frame

    return best_frame


def nested_cv_per_player(X: np.ndarray, y: np.ndarray, kp: KeypointExtractor,
                         group_name: str, feat_type: str, model_name: str,
                         fixed_frame: int = None) -> Dict:
    """
    Nested CV for a single player.
    If fixed_frame is None, find optimal frame on each training fold.
    """
    model_fn = MODELS[model_name]

    if len(y) < 15:
        return {'mse': np.nan, 'r2': np.nan, 'selected_frames': []}

    preds = np.zeros(len(y))
    preds[:] = np.nan
    selected_frames = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Determine frame to use
        if fixed_frame is not None:
            frame = fixed_frame
        else:
            frame = find_optimal_frame_inner_cv(X_train, y_train, kp, group_name, feat_type, model_fn)
        selected_frames.append(frame)

        # Extract features
        X_train_feat = np.array([extract_group_features(X_train[i], kp, group_name, frame, feat_type)
                                for i in range(len(X_train))])
        X_test_feat = np.array([extract_group_features(X_test[i], kp, group_name, frame, feat_type)
                               for i in range(len(X_test))])

        valid = ~(np.isnan(X_train_feat).any(axis=1) | np.isnan(y_train))
        if valid.sum() < 5:
            preds[test_idx] = y_train.mean()
            continue

        X_clean = X_train_feat[valid]
        y_clean = y_train[valid]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_clean)

        model = model_fn()
        model.fit(X_tr_s, y_clean)

        X_test_clean = np.nan_to_num(X_test_feat, nan=0.0)
        X_te_s = scaler.transform(X_test_clean)
        fold_preds = model.predict(X_te_s)

        preds[test_idx] = fold_preds

    # Fill NaN with mean
    preds[np.isnan(preds)] = y.mean()

    mse = np.mean((preds - y) ** 2)
    baseline_mse = np.var(y)
    r2 = 1 - mse / baseline_mse

    return {
        'mse': mse,
        'r2': r2,
        'baseline_mse': baseline_mse,
        'selected_frames': selected_frames
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE PHYSICS TEST")
    print("=" * 80)
    print()
    print("Testing: Player x Frame x Group x Target x Model")
    print("Using nested CV to get unbiased estimates")
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

    # Fixed frames (current baseline)
    FIXED_FRAMES = {
        'angle': 153,
        'depth': 102,
        'left_right': 153,
    }

    results = []

    # ==========================================================================
    # Test all combinations
    # ==========================================================================

    for target_name, target in targets.items():
        print(f"\n{'='*80}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*80}")

        for group_name in FEATURE_GROUPS.keys():
            print(f"\n  Group: {group_name} - {FEATURE_GROUPS[group_name]['description']}")

            for feat_type in FEATURE_TYPES:
                for model_name in ['ridge_med', 'ridge_high']:  # Test two regularization levels

                    print(f"    {feat_type} + {model_name}:")

                    for pid in sorted(np.unique(participant_ids)):
                        mask = participant_ids == pid
                        X_player = X[mask]
                        y_player = target[mask]

                        # Test with fixed frame
                        fixed_result = nested_cv_per_player(
                            X_player, y_player, kp, group_name, feat_type, model_name,
                            fixed_frame=FIXED_FRAMES[target_name]
                        )

                        # Test with optimal frame (found per fold)
                        optimal_result = nested_cv_per_player(
                            X_player, y_player, kp, group_name, feat_type, model_name,
                            fixed_frame=None
                        )

                        improvement = ((fixed_result['mse'] - optimal_result['mse']) / fixed_result['mse'] * 100
                                      if fixed_result['mse'] > 0 else 0)

                        # Determine if frames are consistent across folds
                        frames = optimal_result['selected_frames']
                        frame_consistency = len(set(frames)) == 1 if frames else False

                        indicator = ""
                        if optimal_result['r2'] > 0:
                            indicator = " ** POSITIVE R² **"
                        elif improvement > 5:
                            indicator = " (improved)"

                        print(f"      P{pid}: Fixed R²={fixed_result['r2']:+.3f}, "
                              f"Optimal R²={optimal_result['r2']:+.3f}, "
                              f"Frames={list(set(frames))}{indicator}")

                        results.append({
                            'target': target_name,
                            'group': group_name,
                            'feat_type': feat_type,
                            'model': model_name,
                            'player': pid,
                            'fixed_r2': fixed_result['r2'],
                            'optimal_r2': optimal_result['r2'],
                            'fixed_mse': fixed_result['mse'],
                            'optimal_mse': optimal_result['mse'],
                            'improvement_pct': improvement,
                            'selected_frames': str(list(set(frames))),
                            'frame_consistent': frame_consistency,
                        })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY: BEST CONFIGURATIONS")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    # Find configurations with positive R²
    positive_r2 = results_df[results_df['optimal_r2'] > 0].copy()
    print(f"\nConfigurations with POSITIVE R² (real signal): {len(positive_r2)}/{len(results_df)}")

    if len(positive_r2) > 0:
        # Group by player and target, show best
        print("\nBest per Player per Target:")
        print("-" * 90)

        for target_name in ['angle', 'depth', 'left_right']:
            print(f"\n{target_name.upper()}:")
            target_positive = positive_r2[positive_r2['target'] == target_name]

            for pid in sorted(np.unique(participant_ids)):
                player_positive = target_positive[target_positive['player'] == pid]
                if len(player_positive) > 0:
                    best = player_positive.loc[player_positive['optimal_r2'].idxmax()]
                    print(f"  Player {pid}: {best['group']:12} {best['feat_type']:8} "
                          f"R²={best['optimal_r2']:.4f} frames={best['selected_frames']}")
                else:
                    print(f"  Player {pid}: No positive R² found")

    # Overall stats
    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    for target_name in ['angle', 'depth', 'left_right']:
        target_results = results_df[results_df['target'] == target_name]
        pos_count = (target_results['optimal_r2'] > 0).sum()
        mean_r2 = target_results['optimal_r2'].mean()
        max_r2 = target_results['optimal_r2'].max()

        print(f"\n{target_name}:")
        print(f"  Positive R² configs: {pos_count}/{len(target_results)}")
        print(f"  Mean R²: {mean_r2:.4f}")
        print(f"  Max R²: {max_r2:.4f}")

    # Where does player-specific frame help?
    print()
    print("=" * 80)
    print("PLAYER-SPECIFIC FRAMES ANALYSIS")
    print("=" * 80)

    improved = results_df[results_df['improvement_pct'] > 5]
    print(f"\nConfigs where player-specific frame improves >5%: {len(improved)}/{len(results_df)}")

    if len(improved) > 0:
        print("\nTop improvements:")
        top_improved = improved.nlargest(10, 'improvement_pct')
        for _, row in top_improved.iterrows():
            print(f"  {row['target']:10} P{row['player']} {row['group']:12} {row['feat_type']:8}: "
                  f"+{row['improvement_pct']:.1f}% (frames: {row['selected_frames']})")

    # Save full results
    results_df.to_csv(OUTPUT_DIR / 'comprehensive_physics_test.csv', index=False)
    print(f"\nFull results saved to {OUTPUT_DIR / 'comprehensive_physics_test.csv'}")

    # ==========================================================================
    # Final Recommendation
    # ==========================================================================
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    # Find the most promising configurations
    best_configs = positive_r2.nlargest(5, 'optimal_r2') if len(positive_r2) > 0 else pd.DataFrame()

    if len(best_configs) > 0:
        print("Most promising physics configurations (positive R², nested CV validated):")
        for _, row in best_configs.iterrows():
            print(f"  {row['target']:10} Player {row['player']}: {row['group']} ({row['feat_type']})")
            print(f"    R² = {row['optimal_r2']:.4f}, Frames = {row['selected_frames']}")
    else:
        print("No configurations found with positive R² in nested CV.")
        print("Physics features may not generalize beyond training data.")


if __name__ == "__main__":
    main()
