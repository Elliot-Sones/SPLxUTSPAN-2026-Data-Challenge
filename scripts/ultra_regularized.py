"""
Ultra-Regularized Model

Use very high regularization (alpha=1000+) to force low variance predictions.
Goal: Get angle_std < 0.137 while maintaining good predictions.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'
SUBMISSION_DIR = PROJECT_DIR / 'submission'

TARGETS = ['angle', 'depth', 'left_right']

TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def extract_minimal_features(timeseries, pid, keypoint_idx):
    """Extract minimal features - only the most stable predictors."""
    features = {}

    # Key frames from research
    frames = [150, 153, 155]

    # Key joints only
    key_joints = ['right_wrist', 'right_elbow', 'right_knee', 'mid_hip', 'left_wrist']

    for joint in key_joints:
        if joint in keypoint_idx:
            idx = keypoint_idx[joint]
            for frame in frames:
                data = timeseries[frame, idx*3:(idx+1)*3]
                features[f'{joint}_f{frame}_z'] = data[2]  # z only

    # Add player ID
    for p in range(1, 6):
        features[f'player_{p}'] = 1.0 if pid == p else 0.0

    return features


def load_data_minimal(train=True):
    """Load data with minimal features."""
    filename = "train.csv" if train else "test.csv"
    print(f"Loading {filename}...")
    df = pd.read_csv(DATA_DIR / filename)

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in df.columns if c not in meta_cols]

    # Build keypoint index
    keypoint_idx = {}
    for col in keypoint_cols:
        if col.endswith('_x'):
            name = col[:-2]
            keypoint_idx[name] = len(keypoint_idx)

    all_features = []
    all_targets = []
    all_pids = []
    all_ids = []

    for idx, row in df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        feats = extract_minimal_features(timeseries, row['participant_id'], keypoint_idx)
        all_features.append(feats)

        if train:
            all_targets.append([row['angle'], row['depth'], row['left_right']])
        all_pids.append(row['participant_id'])
        all_ids.append(row['id'])

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)

    if train:
        y = np.array(all_targets, dtype=np.float32)
    else:
        y = None

    pids = np.array(all_pids)

    print(f"Features: {X.shape[1]}")
    return X, y, pids, feature_names, all_ids


def main():
    print("="*80)
    print("ULTRA-REGULARIZED MODEL")
    print("="*80)

    # Load data
    X_train, y_train, pids_train, feature_names, train_ids = load_data_minimal(train=True)
    X_test, _, pids_test, _, test_ids = load_data_minimal(train=False)

    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    # Try different regularization strengths
    best_submission = None
    best_angle_std = float('inf')

    for alpha in [100, 500, 1000, 2000, 5000]:
        print(f"\n" + "="*60)
        print(f"ALPHA = {alpha}")
        print("="*60)

        predictions = {target: np.zeros(len(X_test)) for target in TARGETS}

        unique_pids = sorted(np.unique(pids_train))

        for target_idx, target in enumerate(TARGETS):
            for pid in unique_pids:
                train_mask = pids_train == pid
                test_mask = pids_test == pid

                if not np.any(test_mask):
                    continue

                X_player = X_train[train_mask]
                y_player = y_train[train_mask, target_idx]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_player)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_scaled, y_player)

                X_test_player = X_test[test_mask]
                X_test_scaled = scaler.transform(X_test_player)
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

                predictions[target][test_mask] = model.predict(X_test_scaled)

        # Convert to scaled space
        scaled_preds = {}
        for target in TARGETS:
            scaler = TARGET_SCALERS[target]
            scaled = scaler.transform(predictions[target].reshape(-1, 1)).flatten()
            scaled = np.clip(scaled, 0, 1)
            scaled_preds[target] = scaled

        angle_std = np.std(scaled_preds['angle'])
        depth_mean = np.mean(scaled_preds['depth'])

        print(f"angle_std:  {angle_std:.4f}")
        print(f"depth_mean: {depth_mean:.4f}")

        # Blend with Sub 133 to get best angle_std
        for w in np.arange(0.0, 1.0, 0.01):
            blend_angle = w * scaled_preds['angle'] + (1 - w) * sub133['scaled_angle'].values
            blend_std = np.std(blend_angle)
            if blend_std <= 0.1371:  # Target: match Sub 133
                break

        blend_angle = w * scaled_preds['angle'] + (1 - w) * sub133['scaled_angle'].values
        blend_depth = w * scaled_preds['depth'] + (1 - w) * sub133['scaled_depth'].values
        blend_lr = w * scaled_preds['left_right'] + (1 - w) * sub133['scaled_left_right'].values

        # Calibrate depth
        blend_depth = blend_depth - np.mean(blend_depth) + 0.5055
        blend_depth = np.clip(blend_depth, 0, 1)

        final_std = np.std(blend_angle)
        final_depth = np.mean(blend_depth)
        corr = np.corrcoef(blend_angle, sub133['scaled_angle'].values)[0, 1]

        print(f"\nBlend: {w:.0%} new + {1-w:.0%} Sub133")
        print(f"Final angle_std:  {final_std:.4f}")
        print(f"Final depth_mean: {final_depth:.4f}")
        print(f"Correlation w/ 133: {corr:.4f}")

        if final_std < best_angle_std and w > 0:
            best_angle_std = final_std
            best_submission = (blend_angle.copy(), blend_depth.copy(), blend_lr.copy(), alpha, w)

    # Save best
    if best_submission:
        angle_pred, depth_pred, lr_pred, alpha, w = best_submission

        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1

        submission = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': angle_pred,
            'scaled_depth': depth_pred,
            'scaled_left_right': lr_pred
        })

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)

        print(f"\n" + "="*80)
        print(f"SAVED BEST: {output_file}")
        print(f"  Alpha: {alpha}")
        print(f"  Blend: {w:.0%} new + {1-w:.0%} Sub133")
        print(f"  angle_std: {np.std(angle_pred):.4f}")
        print("="*80)

    # Also try ensemble of different alphas
    print("\n" + "="*80)
    print("ENSEMBLE OF DIFFERENT REGULARIZATIONS")
    print("="*80)

    ensemble_preds = {target: np.zeros(len(X_test)) for target in TARGETS}
    alphas = [100, 500, 1000, 2000]

    for alpha in alphas:
        predictions = {target: np.zeros(len(X_test)) for target in TARGETS}

        for target_idx, target in enumerate(TARGETS):
            for pid in unique_pids:
                train_mask = pids_train == pid
                test_mask = pids_test == pid

                if not np.any(test_mask):
                    continue

                X_player = X_train[train_mask]
                y_player = y_train[train_mask, target_idx]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_player)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_scaled, y_player)

                X_test_player = X_test[test_mask]
                X_test_scaled = scaler.transform(X_test_player)
                X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

                predictions[target][test_mask] = model.predict(X_test_scaled)

        for target in TARGETS:
            ensemble_preds[target] += predictions[target] / len(alphas)

    # Convert to scaled
    scaled_ens = {}
    for target in TARGETS:
        scaler = TARGET_SCALERS[target]
        scaled = scaler.transform(ensemble_preds[target].reshape(-1, 1)).flatten()
        scaled = np.clip(scaled, 0, 1)
        scaled_ens[target] = scaled

    print(f"\nEnsemble profile:")
    print(f"  angle_std:  {np.std(scaled_ens['angle']):.4f}")
    print(f"  depth_mean: {np.mean(scaled_ens['depth']):.4f}")

    # Blend with Sub 133
    for w in np.arange(0.0, 1.0, 0.01):
        blend_angle = w * scaled_ens['angle'] + (1 - w) * sub133['scaled_angle'].values
        if np.std(blend_angle) <= 0.1371:
            break

    blend_angle = w * scaled_ens['angle'] + (1 - w) * sub133['scaled_angle'].values
    blend_depth = w * scaled_ens['depth'] + (1 - w) * sub133['scaled_depth'].values
    blend_lr = w * scaled_ens['left_right'] + (1 - w) * sub133['scaled_left_right'].values

    blend_depth = blend_depth - np.mean(blend_depth) + 0.5055
    blend_depth = np.clip(blend_depth, 0, 1)

    corr = np.corrcoef(blend_angle, sub133['scaled_angle'].values)[0, 1]

    print(f"\nEnsemble blend: {w:.0%} new + {1-w:.0%} Sub133")
    print(f"  angle_std:  {np.std(blend_angle):.4f}")
    print(f"  depth_mean: {np.mean(blend_depth):.4f}")
    print(f"  Correlation: {corr:.4f}")

    if w > 0:
        next_num += 1
        submission = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': blend_angle,
            'scaled_depth': blend_depth,
            'scaled_left_right': blend_lr
        })
        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)
        print(f"\nSaved ensemble: {output_file}")


if __name__ == "__main__":
    main()
