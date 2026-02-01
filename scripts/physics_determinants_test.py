"""
PHYSICS DETERMINANTS TEST

Groups based on what PHYSICALLY determines each target:

ANGLE (release angle):
- Wrist position at release (determines ball trajectory angle)
- Elbow extension (arm angle at release)
- Release height (higher = steeper angle typically)

DEPTH (distance/power):
- Leg drive (knee/ankle extension = power source)
- Hip thrust (core power transfer)
- Follow-through velocity (momentum transfer)

LEFT_RIGHT (lateral accuracy):
- Shoulder alignment (rotation = lateral deviation)
- Elbow position relative to shoulder (inside/outside)
- Wrist deviation from center line

Testing per player with nested CV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
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

    def get(self, ts: np.ndarray, name: str, frame: int = None):
        idx = self.col_to_idx.get(name)
        if idx is None:
            return np.nan if frame is not None else np.full(ts.shape[0], np.nan)
        if frame is not None:
            if frame < 0 or frame >= ts.shape[0]:
                return np.nan
            return ts[frame, idx]
        return ts[:, idx]


# Physics-based feature definitions for each target
TARGET_PHYSICS = {
    'angle': {
        'description': 'Release angle depends on arm position and wrist angle at release',
        'groups': {
            'wrist_release': {
                'description': 'Wrist height at release determines initial ball angle',
                'features': [
                    {'name': 'wrist_z_position', 'keypoint': 'right_wrist_z', 'type': 'position'},
                ],
                'frame_range': (140, 180),  # Release window
            },
            'elbow_extension': {
                'description': 'Elbow angle affects release trajectory',
                'features': [
                    {'name': 'elbow_z_position', 'keypoint': 'right_elbow_z', 'type': 'position'},
                    {'name': 'elbow_velocity', 'keypoint': 'right_elbow_z', 'type': 'velocity'},
                ],
                'frame_range': (130, 180),
            },
            'arm_angle': {
                'description': 'Wrist-elbow relationship determines arm angle',
                'features': [
                    {'name': 'wrist_elbow_diff', 'keypoint1': 'right_wrist_z', 'keypoint2': 'right_elbow_z', 'type': 'diff'},
                ],
                'frame_range': (140, 175),
            },
        },
    },
    'depth': {
        'description': 'Depth/power comes from leg drive and body thrust',
        'groups': {
            'leg_drive': {
                'description': 'Knee extension provides primary power',
                'features': [
                    {'name': 'knee_velocity', 'keypoint': 'right_knee_z', 'type': 'velocity'},
                    {'name': 'ankle_velocity', 'keypoint': 'right_ankle_z', 'type': 'velocity'},
                ],
                'frame_range': (90, 160),  # Push phase
            },
            'hip_thrust': {
                'description': 'Hip extension transfers leg power to upper body',
                'features': [
                    {'name': 'hip_velocity', 'keypoint': 'right_hip_z', 'type': 'velocity'},
                    {'name': 'hip_position', 'keypoint': 'right_hip_z', 'type': 'position'},
                ],
                'frame_range': (100, 160),
            },
            'body_extension': {
                'description': 'Full body extension at peak',
                'features': [
                    {'name': 'wrist_hip_diff', 'keypoint1': 'right_wrist_z', 'keypoint2': 'right_hip_z', 'type': 'diff'},
                ],
                'frame_range': (120, 170),
            },
        },
    },
    'left_right': {
        'description': 'Lateral accuracy depends on body alignment and rotation',
        'groups': {
            'shoulder_alignment': {
                'description': 'Shoulder rotation causes lateral deviation',
                'features': [
                    {'name': 'shoulder_z', 'keypoint': 'right_shoulder_z', 'type': 'position'},
                    {'name': 'shoulder_diff', 'keypoint1': 'right_shoulder_z', 'keypoint2': 'left_shoulder_z', 'type': 'diff'},
                ],
                'frame_range': (140, 180),
            },
            'elbow_alignment': {
                'description': 'Elbow position relative to body center',
                'features': [
                    {'name': 'elbow_shoulder_diff', 'keypoint1': 'right_elbow_z', 'keypoint2': 'right_shoulder_z', 'type': 'diff'},
                ],
                'frame_range': (140, 175),
            },
            'hip_rotation': {
                'description': 'Hip rotation affects aim',
                'features': [
                    {'name': 'hip_diff', 'keypoint1': 'right_hip_z', 'keypoint2': 'left_hip_z', 'type': 'diff'},
                ],
                'frame_range': (130, 170),
            },
        },
    },
}


def extract_feature(ts: np.ndarray, kp: KeypointExtractor, feat_def: Dict, frame: int) -> float:
    """Extract a single feature at a specific frame."""
    feat_type = feat_def['type']

    if feat_type == 'position':
        return kp.get(ts, feat_def['keypoint'], frame)

    elif feat_type == 'velocity':
        v1 = kp.get(ts, feat_def['keypoint'], frame)
        v2 = kp.get(ts, feat_def['keypoint'], frame + 10)
        if np.isnan(v1) or np.isnan(v2):
            return np.nan
        return v2 - v1

    elif feat_type == 'diff':
        v1 = kp.get(ts, feat_def['keypoint1'], frame)
        v2 = kp.get(ts, feat_def['keypoint2'], frame)
        if np.isnan(v1) or np.isnan(v2):
            return np.nan
        return v1 - v2

    return np.nan


def extract_group_features(ts: np.ndarray, kp: KeypointExtractor,
                           group_def: Dict, frame: int) -> np.ndarray:
    """Extract all features for a group at a specific frame."""
    features = []
    for feat_def in group_def['features']:
        features.append(extract_feature(ts, kp, feat_def, frame))
    return np.array(features)


def find_optimal_frame(X_train: np.ndarray, y_train: np.ndarray,
                       kp: KeypointExtractor, group_def: Dict) -> int:
    """Find optimal frame using inner CV on training data only."""
    frame_range = group_def['frame_range']
    best_frame = frame_range[0]
    best_score = -999

    for frame in range(frame_range[0], frame_range[1], 5):
        features = np.array([extract_group_features(X_train[i], kp, group_def, frame)
                            for i in range(len(X_train))])

        valid = ~(np.isnan(features).any(axis=1) | np.isnan(y_train))
        if valid.sum() < 8:
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

            model = Ridge(alpha=1.0)
            model.fit(X_tr_s, y_tr)
            scores.append(model.score(X_val_s, y_val))

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_frame = frame

    return best_frame


def nested_cv_test(X: np.ndarray, y: np.ndarray, kp: KeypointExtractor,
                   group_def: Dict) -> Dict:
    """Nested CV test for a single player with a specific feature group."""
    if len(y) < 15:
        return {'r2': np.nan, 'mse': np.nan, 'frames': []}

    preds = np.zeros(len(y))
    preds[:] = np.nan
    selected_frames = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Find optimal frame on training data
        frame = find_optimal_frame(X_train, y_train, kp, group_def)
        selected_frames.append(frame)

        # Extract features
        X_train_feat = np.array([extract_group_features(X_train[i], kp, group_def, frame)
                                for i in range(len(X_train))])
        X_test_feat = np.array([extract_group_features(X_test[i], kp, group_def, frame)
                               for i in range(len(X_test))])

        valid = ~(np.isnan(X_train_feat).any(axis=1) | np.isnan(y_train))
        if valid.sum() < 5:
            preds[test_idx] = y_train.mean()
            continue

        X_clean = X_train_feat[valid]
        y_clean = y_train[valid]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_clean)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_clean)

        X_test_clean = np.nan_to_num(X_test_feat, nan=0.0)
        X_te_s = scaler.transform(X_test_clean)
        preds[test_idx] = model.predict(X_te_s)

    preds[np.isnan(preds)] = y.mean()

    mse = np.mean((preds - y) ** 2)
    baseline_mse = np.var(y)
    r2 = 1 - mse / baseline_mse

    return {'r2': r2, 'mse': mse, 'baseline_mse': baseline_mse, 'frames': selected_frames}


def main():
    print("=" * 80)
    print("PHYSICS DETERMINANTS TEST")
    print("=" * 80)
    print()
    print("Testing physics features based on what DETERMINES each target")
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

    results = []

    # ==========================================================================
    # Test each target with its physics-based groups
    # ==========================================================================

    for target_name, target in targets.items():
        target_physics = TARGET_PHYSICS[target_name]

        print()
        print("=" * 80)
        print(f"TARGET: {target_name.upper()}")
        print(f"Physics: {target_physics['description']}")
        print("=" * 80)

        for group_name, group_def in target_physics['groups'].items():
            print(f"\n  {group_name}: {group_def['description']}")
            print(f"  Features: {[f['name'] for f in group_def['features']]}")
            print(f"  Frame search range: {group_def['frame_range']}")
            print()

            for pid in sorted(np.unique(participant_ids)):
                mask = participant_ids == pid
                X_player = X[mask]
                y_player = target[mask]

                result = nested_cv_test(X_player, y_player, kp, group_def)

                # Determine frame consistency
                frames = result['frames']
                unique_frames = list(set(frames)) if frames else []
                consistent = len(unique_frames) == 1

                indicator = ""
                if result['r2'] > 0.05:
                    indicator = " *** SIGNAL ***"
                elif result['r2'] > 0:
                    indicator = " (positive)"

                print(f"    Player {pid}: R² = {result['r2']:+.4f}, "
                      f"Frames = {unique_frames} "
                      f"{'(consistent)' if consistent else '(varies)'}{indicator}")

                results.append({
                    'target': target_name,
                    'group': group_name,
                    'player': pid,
                    'r2': result['r2'],
                    'mse': result['mse'],
                    'baseline_mse': result['baseline_mse'],
                    'frames': str(unique_frames),
                    'frame_consistent': consistent,
                    'n_features': len(group_def['features']),
                })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY: PHYSICS FEATURES WITH REAL SIGNAL")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    # Find positive R² results
    positive = results_df[results_df['r2'] > 0]
    strong = results_df[results_df['r2'] > 0.05]

    print(f"\nPositive R² (any signal): {len(positive)}/{len(results_df)}")
    print(f"Strong R² (> 0.05): {len(strong)}/{len(results_df)}")

    # Best per target
    print()
    print("BEST PHYSICS GROUP PER PLAYER PER TARGET:")
    print("-" * 80)

    for target_name in ['angle', 'depth', 'left_right']:
        print(f"\n{target_name.upper()}:")
        target_results = results_df[results_df['target'] == target_name]

        for pid in sorted(np.unique(participant_ids)):
            player_results = target_results[target_results['player'] == pid]
            if len(player_results) == 0:
                continue

            best = player_results.loc[player_results['r2'].idxmax()]
            indicator = "***" if best['r2'] > 0.05 else "+" if best['r2'] > 0 else ""

            print(f"  Player {pid}: {best['group']:20} R²={best['r2']:+.4f} "
                  f"frames={best['frames']:15} {indicator}")

    # Per target summary
    print()
    print("AVERAGE R² BY TARGET AND GROUP:")
    print("-" * 60)

    summary = results_df.groupby(['target', 'group'])['r2'].agg(['mean', 'std', 'max']).round(4)
    print(summary.to_string())

    # Overall verdict
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    for target_name in ['angle', 'depth', 'left_right']:
        target_results = results_df[results_df['target'] == target_name]
        mean_r2 = target_results['r2'].mean()
        max_r2 = target_results['r2'].max()
        pos_count = (target_results['r2'] > 0).sum()

        print(f"\n{target_name}:")
        print(f"  Mean R²: {mean_r2:.4f}")
        print(f"  Max R²:  {max_r2:.4f}")
        print(f"  Positive R² configs: {pos_count}/{len(target_results)}")

        if max_r2 > 0.05:
            print(f"  -> USABLE physics signal exists")
        elif max_r2 > 0:
            print(f"  -> Weak signal, may not generalize")
        else:
            print(f"  -> No physics signal found")

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'physics_determinants_test.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'physics_determinants_test.csv'}")


if __name__ == "__main__":
    main()
