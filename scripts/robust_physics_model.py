"""
ROBUST PHYSICS MODEL

Sub 79 got 0.015604 (worse than baseline 0.008305).
The 5-feature greedy selection overfit to training data.

New approach:
1. Use only 1-2 features per player (less overfitting)
2. Use pre-defined robust features based on physics understanding
3. Higher regularization
4. Test nested CV to get realistic estimates
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DATA_DIR = Path(__file__).parent.parent / 'data'
SUBMISSION_DIR = Path(__file__).parent.parent / 'submission'


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


# ROBUST features - based on basketball physics, not data mining
# These are features that SHOULD matter based on domain knowledge
ROBUST_FEATURES = {
    'angle': {
        # Release angle depends on wrist position at release
        'all': [
            {'type': 'position', 'keypoint': 'right_wrist_z', 'frame': 153, 'name': 'wrist_z_release'},
            {'type': 'velocity', 'keypoint': 'right_wrist_z', 'start': 140, 'end': 160, 'name': 'wrist_vel_release'},
        ]
    },
    'depth': {
        # Depth depends on push force from legs
        'all': [
            {'type': 'velocity', 'keypoint': 'right_knee_z', 'start': 100, 'end': 150, 'name': 'knee_push_vel'},
            {'type': 'position', 'keypoint': 'right_hip_z', 'frame': 120, 'name': 'hip_height_push'},
        ]
    },
    'left_right': {
        # Left-right depends on shoulder alignment
        'all': [
            {'type': 'position', 'keypoint': 'right_shoulder_z', 'frame': 153, 'name': 'shoulder_z_release'},
            {'type': 'position', 'keypoint': 'right_elbow_z', 'frame': 153, 'name': 'elbow_z_release'},
        ]
    }
}


def extract_single_feature(ts: np.ndarray, kp: KeypointExtractor,
                           feat_spec: Dict[str, Any]) -> float:
    """Extract a single feature."""
    feat_type = feat_spec['type']

    if feat_type == 'position':
        return kp.get(ts, feat_spec['keypoint'], feat_spec['frame'])

    elif feat_type == 'velocity':
        val_start = kp.get(ts, feat_spec['keypoint'], feat_spec['start'])
        val_end = kp.get(ts, feat_spec['keypoint'], feat_spec['end'])
        if np.isnan(val_start) or np.isnan(val_end):
            return np.nan
        return (val_end - val_start) / (feat_spec['end'] - feat_spec['start'])

    return np.nan


def nested_cv_score(X_all: np.ndarray, y_all: np.ndarray, participant_ids: np.ndarray,
                    alpha: float = 10.0) -> float:
    """Nested CV - outer loop for evaluation, inner loop for training."""
    outer_preds = np.zeros(len(y_all))
    outer_preds[:] = np.nan

    # Outer loop: 5-fold CV
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf_outer.split(X_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        pids_train = participant_ids[train_idx]
        pids_test = participant_ids[test_idx]

        # Train per-player models on train fold
        for pid in np.unique(pids_train):
            train_mask = pids_train == pid
            test_mask = pids_test == pid

            if test_mask.sum() == 0:
                continue

            X_p_train = X_train[train_mask]
            y_p_train = y_train[train_mask]
            X_p_test = X_test[test_mask]

            # Handle NaN
            valid = ~(np.isnan(X_p_train).any(axis=1) | np.isnan(y_p_train))
            if valid.sum() < 5:
                # Fallback to mean
                outer_preds[test_idx[test_mask]] = y_p_train.mean()
                continue

            X_clean = X_p_train[valid]
            y_clean = y_p_train[valid]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            model = Ridge(alpha=alpha)
            model.fit(X_scaled, y_clean)

            # Predict on test
            X_p_test_clean = np.nan_to_num(X_p_test, nan=0.0)
            X_p_test_scaled = scaler.transform(X_p_test_clean)
            preds = model.predict(X_p_test_scaled)

            outer_preds[test_idx[test_mask]] = preds

    # Fill remaining NaN with overall mean
    outer_preds[np.isnan(outer_preds)] = y_all.mean()

    # Calculate MSE
    mse = np.mean((outer_preds - y_all) ** 2)
    return mse


def main():
    print("=" * 80)
    print("ROBUST PHYSICS MODEL")
    print("=" * 80)
    print()
    print("Using pre-defined robust features based on physics, not data mining")
    print()

    # Load data
    print("Loading data...")
    X_train, y_train, meta_train = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    train_pids = meta_train['participant_id'].values
    targets_train = {
        'angle': y_train[:, 0],
        'depth': y_train[:, 1],
        'left_right': y_train[:, 2],
    }

    print(f"Training: {len(y_train)} samples")

    # Load test data
    X_test, _, meta_test = load_all_as_arrays(train=False)
    test_pids = meta_test['participant_id'].values
    print(f"Test: {len(X_test)} samples")
    print()

    # Load scalers
    scalers = {}
    for t in ['angle', 'depth', 'left_right']:
        scalers[t] = joblib.load(DATA_DIR / f'scaler_{t}.pkl')

    # ==========================================================================
    # Test 1: Nested CV with robust features
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: Nested CV with Robust Features (2 features per target)")
    print("=" * 80)
    print()

    for target_name, target in targets_train.items():
        features = ROBUST_FEATURES[target_name]['all']

        # Extract features
        X_feat = np.zeros((len(X_train), len(features)))
        for i in range(len(X_train)):
            for j, feat_spec in enumerate(features):
                X_feat[i, j] = extract_single_feature(X_train[i], kp, feat_spec)

        # Nested CV
        mse = nested_cv_score(X_feat, target, train_pids, alpha=10.0)

        # Scale to compare with LB
        y_scaled = scalers[target_name].transform(target.reshape(-1, 1)).ravel()
        # Rough estimate: CV MSE in original scale -> scaled MSE
        y_var_orig = np.var(target)
        y_var_scaled = np.var(y_scaled)
        mse_scaled_est = mse * (y_var_scaled / y_var_orig)

        print(f"{target_name}:")
        print(f"  Features: {[f['name'] for f in features]}")
        print(f"  Nested CV MSE (original): {mse:.4f}")
        print(f"  Estimated scaled MSE: {mse_scaled_est:.6f}")
        print()

    # ==========================================================================
    # Test 2: Compare with baseline (just mean prediction)
    # ==========================================================================
    print("=" * 80)
    print("TEST 2: Compare with Mean Baseline")
    print("=" * 80)
    print()

    for target_name, target in targets_train.items():
        # Mean prediction per player
        mean_preds = np.zeros(len(target))
        for pid in np.unique(train_pids):
            mask = train_pids == pid
            mean_preds[mask] = target[mask].mean()

        mse_mean = np.mean((mean_preds - target) ** 2)

        features = ROBUST_FEATURES[target_name]['all']
        X_feat = np.zeros((len(X_train), len(features)))
        for i in range(len(X_train)):
            for j, feat_spec in enumerate(features):
                X_feat[i, j] = extract_single_feature(X_train[i], kp, feat_spec)

        mse_physics = nested_cv_score(X_feat, target, train_pids, alpha=10.0)

        improvement = (mse_mean - mse_physics) / mse_mean * 100

        print(f"{target_name}:")
        print(f"  Mean baseline MSE: {mse_mean:.4f}")
        print(f"  Physics MSE: {mse_physics:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")
        print()

    # ==========================================================================
    # Test 3: Higher regularization sweep
    # ==========================================================================
    print("=" * 80)
    print("TEST 3: Regularization Sweep")
    print("=" * 80)
    print()

    target = targets_train['depth']  # Depth had best signal
    features = ROBUST_FEATURES['depth']['all']
    X_feat = np.zeros((len(X_train), len(features)))
    for i in range(len(X_train)):
        for j, feat_spec in enumerate(features):
            X_feat[i, j] = extract_single_feature(X_train[i], kp, feat_spec)

    print("Depth target - regularization sweep:")
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        mse = nested_cv_score(X_feat, target, train_pids, alpha=alpha)
        print(f"  alpha={alpha:>7.1f}: MSE = {mse:.4f}")
    print()

    # ==========================================================================
    # Generate submission with robust features
    # ==========================================================================
    print("=" * 80)
    print("GENERATING ROBUST SUBMISSION")
    print("=" * 80)
    print()

    test_predictions = {}

    for target_name, target in targets_train.items():
        features = ROBUST_FEATURES[target_name]['all']

        all_preds = np.zeros(len(X_test))

        for pid in np.unique(train_pids):
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            if test_mask.sum() == 0:
                continue

            # Extract training features
            X_feat_train = np.zeros((train_mask.sum(), len(features)))
            for i, idx in enumerate(np.where(train_mask)[0]):
                for j, feat_spec in enumerate(features):
                    X_feat_train[i, j] = extract_single_feature(X_train[idx], kp, feat_spec)

            # Extract test features
            X_feat_test = np.zeros((test_mask.sum(), len(features)))
            for i, idx in enumerate(np.where(test_mask)[0]):
                for j, feat_spec in enumerate(features):
                    X_feat_test[i, j] = extract_single_feature(X_test[idx], kp, feat_spec)

            y_train_p = target[train_mask]

            # Handle NaN
            valid = ~(np.isnan(X_feat_train).any(axis=1) | np.isnan(y_train_p))
            if valid.sum() < 5:
                all_preds[test_mask] = y_train_p.mean()
                continue

            X_clean = X_feat_train[valid]
            y_clean = y_train_p[valid]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            model = Ridge(alpha=100.0)  # High regularization
            model.fit(X_scaled, y_clean)

            X_test_clean = np.nan_to_num(X_feat_test, nan=0.0)
            X_test_scaled = scaler.transform(X_test_clean)
            preds = model.predict(X_test_scaled)

            all_preds[test_mask] = preds

        test_predictions[target_name] = all_preds

    # Scale predictions
    scaled_predictions = {}
    for target_name in ['angle', 'depth', 'left_right']:
        preds = test_predictions[target_name]
        preds_scaled = scalers[target_name].transform(preds.reshape(-1, 1)).ravel()
        preds_scaled = np.clip(preds_scaled, 0.0, 1.0)
        scaled_predictions[target_name] = preds_scaled

        print(f"{target_name}: mean={preds_scaled.mean():.4f}, std={preds_scaled.std():.4f}")

    # Save submission
    submission = pd.DataFrame({
        'id': meta_test['id'].values,
        'scaled_angle': scaled_predictions['angle'],
        'scaled_depth': scaled_predictions['depth'],
        'scaled_left_right': scaled_predictions['left_right'],
    })

    submission_path = SUBMISSION_DIR / 'submission_80.csv'
    submission.to_csv(submission_path, index=False)
    print()
    print(f"Saved to: {submission_path}")

    # ==========================================================================
    # Also create blend with sub25
    # ==========================================================================
    print()
    print("Creating blends with sub25...")

    sub25 = pd.read_csv(SUBMISSION_DIR / 'submission_25.csv')

    for ratio in [0.1, 0.2, 0.3]:
        blend = pd.DataFrame({
            'id': submission['id'],
            'scaled_angle': ratio * submission['scaled_angle'] + (1 - ratio) * sub25['scaled_angle'],
            'scaled_depth': ratio * submission['scaled_depth'] + (1 - ratio) * sub25['scaled_depth'],
            'scaled_left_right': ratio * submission['scaled_left_right'] + (1 - ratio) * sub25['scaled_left_right'],
        })

        sub_num = 80 + int(ratio * 10)
        blend_path = SUBMISSION_DIR / f'submission_{sub_num}.csv'
        blend.to_csv(blend_path, index=False)
        print(f"Sub {sub_num}: {int(ratio*100)}% robust physics + {int((1-ratio)*100)}% sub25 -> {blend_path}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Sub 80: 100% robust physics (2 features per target, high regularization)")
    print("Sub 81: 10% robust + 90% sub25")
    print("Sub 82: 20% robust + 80% sub25")
    print("Sub 83: 30% robust + 80% sub25")
    print()
    print("Recommendation: Test Sub 81 first (safest)")


if __name__ == "__main__":
    main()
