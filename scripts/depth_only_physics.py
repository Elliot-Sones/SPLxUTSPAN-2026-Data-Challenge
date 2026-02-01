"""
DEPTH-ONLY PHYSICS SUBMISSION

Finding: Only depth benefits from physics features (+40% over baseline).
Strategy: Use sub25 for angle and left_right, physics only for depth.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

DATA_DIR = Path(__file__).parent.parent / 'data'
SUBMISSION_DIR = Path(__file__).parent.parent / 'submission'


class KeypointExtractor:
    def __init__(self, keypoint_cols):
        self.keypoint_cols = keypoint_cols
        self.col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    def get(self, ts, name, frame=None):
        idx = self.col_to_idx.get(name)
        if idx is None:
            return np.nan if frame is not None else np.full(ts.shape[0], np.nan)
        if frame is not None:
            if frame < 0 or frame >= ts.shape[0]:
                return np.nan
            return ts[frame, idx]
        return ts[:, idx]


# Best depth features found (from deep dive - these showed 0.63-0.79 R²)
DEPTH_FEATURES_BY_PLAYER = {
    1: [
        {'type': 'position', 'keypoint': 'right_elbow_z', 'frame': 150},
        {'type': 'velocity', 'keypoint': 'left_ankle_z', 'start': 110, 'end': 130},
    ],
    2: [
        {'type': 'velocity', 'keypoint': 'left_ankle_z', 'start': 70, 'end': 110},
        {'type': 'velocity', 'keypoint': 'left_wrist_z', 'start': 140, 'end': 160},
    ],
    3: [
        {'type': 'velocity', 'keypoint': 'left_hip_z', 'start': 130, 'end': 135},
        {'type': 'velocity', 'keypoint': 'left_hip_z', 'start': 120, 'end': 140},
    ],
    4: [
        {'type': 'position', 'keypoint': 'right_shoulder_z', 'frame': 120},
        {'type': 'velocity', 'keypoint': 'right_wrist_z', 'start': 170, 'end': 180},
    ],
    5: [
        {'type': 'position', 'keypoint': 'left_wrist_z', 'frame': 145},
        {'type': 'velocity', 'keypoint': 'left_elbow_z', 'start': 80, 'end': 95},
    ],
}


def extract_feature(ts, kp, feat_spec):
    if feat_spec['type'] == 'position':
        return kp.get(ts, feat_spec['keypoint'], feat_spec['frame'])
    elif feat_spec['type'] == 'velocity':
        val_start = kp.get(ts, feat_spec['keypoint'], feat_spec['start'])
        val_end = kp.get(ts, feat_spec['keypoint'], feat_spec['end'])
        if np.isnan(val_start) or np.isnan(val_end):
            return np.nan
        return (val_end - val_start) / (feat_spec['end'] - feat_spec['start'])
    return np.nan


def main():
    print("=" * 80)
    print("DEPTH-ONLY PHYSICS SUBMISSION")
    print("=" * 80)
    print()

    # Load data
    X_train, y_train, meta_train = load_all_as_arrays(train=True)
    X_test, _, meta_test = load_all_as_arrays(train=False)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    train_pids = meta_train['participant_id'].values
    test_pids = meta_test['participant_id'].values
    depth_train = y_train[:, 1]

    # Load scaler
    depth_scaler = joblib.load(DATA_DIR / 'scaler_depth.pkl')

    # Load sub25 for angle and left_right
    sub25 = pd.read_csv(SUBMISSION_DIR / 'submission_25.csv')

    print(f"Training: {len(y_train)}, Test: {len(X_test)}")
    print()

    # ==========================================================================
    # Predict depth using physics features
    # ==========================================================================
    depth_predictions = np.zeros(len(X_test))

    for pid in np.unique(train_pids):
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        if test_mask.sum() == 0:
            continue

        features = DEPTH_FEATURES_BY_PLAYER.get(pid, [])
        if len(features) == 0:
            depth_predictions[test_mask] = depth_train[train_mask].mean()
            continue

        # Extract training features
        X_feat_train = np.zeros((train_mask.sum(), len(features)))
        for i, idx in enumerate(np.where(train_mask)[0]):
            for j, feat_spec in enumerate(features):
                X_feat_train[i, j] = extract_feature(X_train[idx], kp, feat_spec)

        # Extract test features
        X_feat_test = np.zeros((test_mask.sum(), len(features)))
        for i, idx in enumerate(np.where(test_mask)[0]):
            for j, feat_spec in enumerate(features):
                X_feat_test[i, j] = extract_feature(X_test[idx], kp, feat_spec)

        y_train_p = depth_train[train_mask]

        # Handle NaN
        valid = ~(np.isnan(X_feat_train).any(axis=1) | np.isnan(y_train_p))
        if valid.sum() < 5:
            depth_predictions[test_mask] = y_train_p.mean()
            continue

        X_clean = X_feat_train[valid]
        y_clean = y_train_p[valid]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        # Try different regularization
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_clean)

        X_test_clean = np.nan_to_num(X_feat_test, nan=0.0)
        X_test_scaled = scaler.transform(X_test_clean)
        preds = model.predict(X_test_scaled)

        depth_predictions[test_mask] = preds
        print(f"Player {pid}: {test_mask.sum()} test samples, pred mean={preds.mean():.2f}")

    # Scale depth predictions
    depth_scaled = depth_scaler.transform(depth_predictions.reshape(-1, 1)).ravel()
    depth_scaled = np.clip(depth_scaled, 0.0, 1.0)

    print()
    print(f"Physics depth: mean={depth_scaled.mean():.4f}, std={depth_scaled.std():.4f}")
    print(f"Sub25 depth:   mean={sub25['scaled_depth'].mean():.4f}, std={sub25['scaled_depth'].std():.4f}")

    # ==========================================================================
    # Create submissions: pure hybrid and blends
    # ==========================================================================

    # Sub 84: Pure hybrid (sub25 angle/lr + physics depth)
    hybrid = pd.DataFrame({
        'id': meta_test['id'].values,
        'scaled_angle': sub25['scaled_angle'].values,
        'scaled_depth': depth_scaled,
        'scaled_left_right': sub25['scaled_left_right'].values,
    })
    hybrid.to_csv(SUBMISSION_DIR / 'submission_84.csv', index=False)
    print()
    print("Sub 84: sub25 angle/lr + physics depth (100%)")

    # Sub 85-87: Blends of physics depth with sub25 depth
    for ratio, sub_num in [(0.3, 85), (0.5, 86), (0.7, 87)]:
        blend = pd.DataFrame({
            'id': meta_test['id'].values,
            'scaled_angle': sub25['scaled_angle'].values,
            'scaled_depth': ratio * depth_scaled + (1 - ratio) * sub25['scaled_depth'].values,
            'scaled_left_right': sub25['scaled_left_right'].values,
        })
        blend.to_csv(SUBMISSION_DIR / f'submission_{sub_num}.csv', index=False)
        print(f"Sub {sub_num}: sub25 angle/lr + {int(ratio*100)}% physics depth + {int((1-ratio)*100)}% sub25 depth")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Sub 84: Full physics depth")
    print("Sub 85: 30% physics depth + 70% sub25 depth")
    print("Sub 86: 50% physics depth + 50% sub25 depth")
    print("Sub 87: 70% physics depth + 30% sub25 depth")
    print()
    print("All use sub25 for angle and left_right (physics hurt those)")
    print()
    print("Recommendation: Test Sub 85 first (conservative)")


if __name__ == "__main__":
    main()
