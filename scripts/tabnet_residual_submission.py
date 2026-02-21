"""
TabNet Residual Submission

Best approach from experiments: Residual prediction showed 62% improvement for angle.
This script creates a submission using residual mode.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from pytorch_tabnet.tab_model import TabNetRegressor
from data_loader import iterate_shots, get_keypoint_columns


def get_keypoint_map(keypoint_cols):
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def extract_features(timeseries, keypoint_map):
    """Extract features."""
    features = {}

    key_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'mid_hip', 'right_knee', 'right_ankle',
        'right_third_finger_distal', 'right_second_finger_mcp',
        'left_shoulder', 'neck'
    ]

    for joint in key_joints:
        if joint not in keypoint_map:
            continue
        i = keypoint_map[joint]

        x = timeseries[:, i*3]
        y = timeseries[:, i*3 + 1]
        z = timeseries[:, i*3 + 2]

        features[f'{joint}_z_mean'] = np.nanmean(z)
        features[f'{joint}_z_std'] = np.nanstd(z)
        features[f'{joint}_z_max'] = np.nanmax(z)
        features[f'{joint}_z_range'] = np.nanmax(z) - np.nanmin(z)

        for frame in [110, 120, 130, 140, 150]:
            if frame < len(z):
                features[f'{joint}_z_f{frame}'] = z[frame]

        vz = np.diff(z) * 60
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2) * 60

        features[f'{joint}_vz_max'] = np.nanmax(vz)
        features[f'{joint}_speed_max'] = np.nanmax(speed)

        if len(vz) > 0:
            peak_idx = np.nanargmax(vz)
            features[f'{joint}_vz_at_peak'] = vz[peak_idx]

        if len(z) > 160:
            features[f'{joint}_vz_release_max'] = np.nanmax(vz[100:160]) if len(vz) > 160 else 0

    return features


def get_next_submission_number():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            pass
    return max(numbers) + 1 if numbers else 1


def main():
    print("=" * 80)
    print("TABNET RESIDUAL SUBMISSION")
    print("=" * 80)
    print("Using residual prediction mode (best from experiments)")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Load Sub 219 (best LB submission)
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

    # Extract training features
    print("\nExtracting training features...")
    train_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_features(timeseries, keypoint_map)
        features['player_id'] = metadata['participant_id']
        features['shot_id'] = metadata['id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']
        train_data.append(features)

    train_df = pd.DataFrame(train_data)
    print(f"  Training samples: {len(train_df)}")

    feature_cols = [c for c in train_df.columns
                   if c not in ['player_id', 'shot_id', 'angle', 'depth', 'left_right']]
    print(f"  Features: {len(feature_cols)}")

    X_train = train_df[feature_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0)

    # Scale targets to 0-1
    targets = ['angle', 'depth', 'left_right']
    target_cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']
    target_stats = {}
    y_train = {}

    for target in targets:
        vals = train_df[target].values
        target_stats[target] = {'min': vals.min(), 'max': vals.max()}
        y_train[target] = (vals - vals.min()) / (vals.max() - vals.min())

    groups = train_df['player_id'].values

    # Extract test features
    print("\nExtracting test features...")
    test_data = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_features(timeseries, keypoint_map)
        test_data.append(features)
        test_ids.append(metadata['id'])

    test_df = pd.DataFrame(test_data)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test = test_df[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Get Sub 219 predictions for test (as baseline)
    sub219_indexed = sub219.set_index('id')
    baseline_preds = {}
    for target, col in zip(targets, target_cols):
        baseline_preds[target] = sub219_indexed.loc[test_ids, col].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train residual models
    print("\n" + "=" * 80)
    print("TRAINING RESIDUAL MODELS")
    print("=" * 80)

    predictions = {}
    gkf = GroupKFold(n_splits=5)

    for target, col in zip(targets, target_cols):
        print(f"\n{target.upper()}:")
        y = y_train[target]

        # For training, use per-player mean as baseline (since we don't have Sub 219 train predictions)
        player_means = train_df.groupby('player_id')[target].transform('mean')
        player_means_scaled = (player_means - target_stats[target]['min']) / \
                             (target_stats[target]['max'] - target_stats[target]['min'])
        residuals = y - player_means_scaled.values

        # Cross-validate to find best epochs
        best_epochs = []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_scaled, y, groups)):
            X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
            res_tr, res_val = residuals[tr_idx], residuals[val_idx]

            model = TabNetRegressor(
                n_d=8, n_a=8,
                n_steps=3,
                gamma=1.5,
                lambda_sparse=1e-2,
                optimizer_params=dict(lr=1e-2),
                mask_type='entmax',
                verbose=0
            )

            model.fit(
                X_tr, res_tr.reshape(-1, 1),
                eval_set=[(X_val, res_val.reshape(-1, 1))],
                max_epochs=100,
                patience=20,
                batch_size=32,
                virtual_batch_size=16,
                drop_last=False
            )

            best_epochs.append(model.best_epoch)

        avg_epoch = int(np.mean(best_epochs))
        print(f"  Best epochs: {best_epochs}, avg={avg_epoch}")

        # Train final model
        final_model = TabNetRegressor(
            n_d=8, n_a=8,
            n_steps=3,
            gamma=1.5,
            lambda_sparse=1e-2,
            optimizer_params=dict(lr=1e-2),
            mask_type='entmax',
            verbose=0
        )

        final_model.fit(
            X_train_scaled, residuals.reshape(-1, 1),
            max_epochs=max(avg_epoch + 10, 30),
            patience=100,  # No early stopping
            batch_size=32,
            virtual_batch_size=16,
            drop_last=False
        )

        # Predict residuals for test
        pred_residuals = final_model.predict(X_test_scaled).flatten()

        # Final prediction = baseline + residual
        final_pred = baseline_preds[target] + pred_residuals

        # Clip to valid range
        final_pred = np.clip(final_pred, 0, 1)
        predictions[target] = final_pred

        print(f"  Residual stats: mean={pred_residuals.mean():.4f}, std={pred_residuals.std():.4f}")
        print(f"  Final pred: mean={final_pred.mean():.4f}, std={final_pred.std():.4f}")

    # Create submission
    sub_num = get_next_submission_number()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    })

    # Calibrate depth
    submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
    submission['scaled_depth'] = np.clip(submission['scaled_depth'], 0, 1)

    print(f"\nProfile:")
    print(f"  angle_std: {submission['scaled_angle'].std():.6f} (target ~0.137)")
    print(f"  depth_mean: {submission['scaled_depth'].mean():.6f} (target 0.5055)")

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nResidual TabNet submission saved: {sub_path}")

    # Compare with Sub 219
    print("\n" + "=" * 80)
    print("COMPARISON WITH SUB 219")
    print("=" * 80)

    diff = submission.set_index('id')[target_cols].values - sub219.set_index('id')[target_cols].values
    print(f"Mean absolute difference from Sub 219:")
    for i, col in enumerate(target_cols):
        print(f"  {col}: {np.mean(np.abs(diff[:, i])):.6f}")

    correlation = np.corrcoef(
        submission['scaled_angle'].values,
        sub219.set_index('id').loc[test_ids, 'scaled_angle'].values
    )[0, 1]
    print(f"\nAngle correlation with Sub 219: {correlation:.4f}")


if __name__ == "__main__":
    main()
