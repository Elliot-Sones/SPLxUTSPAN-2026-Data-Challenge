"""
Temporal Features for Depth Prediction

Key insight from within-player correlation analysis:
- set_point_frame: r=0.56 within-player
- release_frame: r=0.52 within-player

This is much better than physics features for depth!
Strategy: Keep Sub219 for angle/left_right, improve depth with temporal features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from data_loader import iterate_shots, get_keypoint_columns


def get_keypoint_map(keypoint_cols):
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def extract_temporal_features(timeseries, kp_idx):
    """Extract temporal features that show within-player signal for depth."""
    features = {}

    # Get wrist trajectory
    if 'right_wrist' not in kp_idx:
        return features

    i = kp_idx['right_wrist']
    wrist_z = timeseries[:, i*3 + 2]
    wrist_vz = np.diff(wrist_z) * 60

    # Release frame: maximum upward velocity
    try:
        release_frame = np.nanargmax(wrist_vz[80:160]) + 80
    except:
        release_frame = 120

    # Set point frame: wrist at highest point before release
    try:
        # Look for local max in z before release
        search_window = wrist_z[max(0, release_frame-40):release_frame]
        if len(search_window) > 0:
            set_point_frame = np.nanargmax(search_window) + max(0, release_frame-40)
        else:
            set_point_frame = release_frame - 10
    except:
        set_point_frame = release_frame - 10

    # Dip frame: minimum z before set point
    try:
        dip_window = wrist_z[60:set_point_frame]
        if len(dip_window) > 0:
            dip_frame = np.nanargmin(dip_window) + 60
        else:
            dip_frame = 80
    except:
        dip_frame = 80

    features['release_frame'] = release_frame
    features['set_point_frame'] = set_point_frame
    features['dip_frame'] = dip_frame
    features['dip_to_release_duration'] = (release_frame - dip_frame) / 60.0
    features['set_to_release_duration'] = (release_frame - set_point_frame) / 60.0

    # Values at key frames
    features['wrist_z_at_dip'] = wrist_z[min(dip_frame, len(wrist_z)-1)]
    features['wrist_z_at_set'] = wrist_z[min(set_point_frame, len(wrist_z)-1)]
    features['wrist_z_at_release'] = wrist_z[min(release_frame, len(wrist_z)-1)]

    # Height changes
    features['dip_to_set_rise'] = features['wrist_z_at_set'] - features['wrist_z_at_dip']
    features['set_to_release_rise'] = features['wrist_z_at_release'] - features['wrist_z_at_set']

    # Add some velocity features at key frames
    if release_frame < len(wrist_vz):
        features['vz_at_release'] = wrist_vz[release_frame]
    else:
        features['vz_at_release'] = np.nan

    # Knee features (also showed within-player signal)
    if 'right_knee' in kp_idx:
        ki = kp_idx['right_knee']
        knee_z = timeseries[:, ki*3 + 2]

        # Knee flexion at key frames
        if dip_frame < len(knee_z):
            features['knee_z_at_dip'] = knee_z[dip_frame]
        if release_frame < len(knee_z):
            features['knee_z_at_release'] = knee_z[release_frame]

        features['knee_rise'] = features.get('knee_z_at_release', 0) - features.get('knee_z_at_dip', 0)

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
    print("TEMPORAL FEATURES FOR DEPTH PREDICTION")
    print("=" * 80)
    print("Using features with high within-player correlation:")
    print("  - set_point_frame (r=0.56)")
    print("  - release_frame (r=0.52)")

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_map(keypoint_cols)

    # Extract training features
    print("\nExtracting training features...")
    train_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_temporal_features(timeseries, kp_idx)
        features['player_id'] = metadata['participant_id']
        features['shot_id'] = metadata['id']
        features['depth'] = metadata['depth']
        train_data.append(features)

    train_df = pd.DataFrame(train_data)
    print(f"  Training samples: {len(train_df)}")

    feature_cols = [c for c in train_df.columns
                   if c not in ['player_id', 'shot_id', 'depth']]
    print(f"  Features: {len(feature_cols)}")

    X_train = train_df[feature_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0)

    # Scale depth to 0-1
    depth_vals = train_df['depth'].values
    depth_min, depth_max = depth_vals.min(), depth_vals.max()
    y_train = (depth_vals - depth_min) / (depth_max - depth_min)

    groups = train_df['player_id'].values

    # Extract test features
    print("\nExtracting test features...")
    test_data = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_temporal_features(timeseries, kp_idx)
        test_data.append(features)
        test_ids.append(metadata['id'])

    test_df = pd.DataFrame(test_data)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test = test_df[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-validate
    print("\n" + "=" * 80)
    print("GROUPKFOLD CV FOR DEPTH")
    print("=" * 80)

    gkf = GroupKFold(n_splits=5)
    mses = []
    oof_preds = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_scaled, y_train, groups)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        pred = np.clip(pred, 0, 1)

        oof_preds[val_idx] = pred
        mse = mean_squared_error(y_val, pred)
        mses.append(mse)
        print(f"  Fold {fold+1}: MSE = {mse:.6f}")

    print(f"\n  Overall CV MSE: {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

    # Train final model
    print("\nTraining final model...")
    final_model = Ridge(alpha=10.0)
    final_model.fit(X_train_scaled, y_train)

    # Make predictions
    depth_pred = final_model.predict(X_test_scaled)
    depth_pred = np.clip(depth_pred, 0, 1)

    # Load Sub219 for angle and left_right
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub219_indexed = sub219.set_index('id')

    # Create submissions with different blend weights
    print("\n" + "=" * 80)
    print("CREATING SUBMISSIONS")
    print("=" * 80)

    sub_num = get_next_submission_number()

    # Pure temporal model for depth
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': sub219_indexed.loc[test_ids, 'scaled_angle'].values,
        'scaled_depth': depth_pred,
        'scaled_left_right': sub219_indexed.loc[test_ids, 'scaled_left_right'].values
    })

    # Calibrate depth
    submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
    submission['scaled_depth'] = np.clip(submission['scaled_depth'], 0, 1)

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Sub {sub_num}: Pure temporal depth, depth_mean={submission['scaled_depth'].mean():.4f}")
    sub_num += 1

    # Blended models
    for weight in [0.1, 0.2, 0.3, 0.5]:
        blended_depth = weight * depth_pred + (1 - weight) * sub219_indexed.loc[test_ids, 'scaled_depth'].values

        submission_blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': sub219_indexed.loc[test_ids, 'scaled_angle'].values,
            'scaled_depth': blended_depth,
            'scaled_left_right': sub219_indexed.loc[test_ids, 'scaled_left_right'].values
        })

        # Calibrate
        submission_blend['scaled_depth'] = submission_blend['scaled_depth'] - submission_blend['scaled_depth'].mean() + 0.5055
        submission_blend['scaled_depth'] = np.clip(submission_blend['scaled_depth'], 0, 1)

        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        submission_blend.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: {int(weight*100)}% temporal + {int((1-weight)*100)}% Sub219 depth")
        sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Temporal features (set_point_frame, release_frame) have r=0.5+ within-player")
    print("correlation for depth. This is much better than physics velocity features.")
    print("If depth CV improves, the blended submissions may help LB score.")


if __name__ == "__main__":
    main()
