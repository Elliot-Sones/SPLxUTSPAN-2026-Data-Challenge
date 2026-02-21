"""
Target-Specific Ensemble

Uses the best approach for each target based on our comprehensive testing:
- Angle: Actual KAN (CV 0.010 - best we've seen)
- Depth: Temporal features (r=0.56 within-player signal)
- Left_right: Sub219 baseline (nothing significantly better found)

This combines the strengths of different approaches.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from data_loader import iterate_shots, get_keypoint_columns

# Import KAN
try:
    from kan import KAN
    import torch
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    print("KAN not available.")


def get_keypoint_map(keypoint_cols):
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def extract_angle_features(timeseries, kp_idx):
    """Extract features optimized for angle prediction (for KAN)."""
    features = {}

    key_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'mid_hip', 'right_knee', 'right_ankle', 'neck'
    ]

    for joint in key_joints:
        if joint not in kp_idx:
            continue

        i = kp_idx[joint]
        x = timeseries[:, i*3]
        y = timeseries[:, i*3 + 1]
        z = timeseries[:, i*3 + 2]

        # Statistics
        features[f'{joint}_z_mean'] = np.nanmean(z)
        features[f'{joint}_z_std'] = np.nanstd(z)
        features[f'{joint}_z_max'] = np.nanmax(z)

        # Velocities
        vz = np.diff(z) * 60
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2) * 60

        features[f'{joint}_vz_max'] = np.nanmax(vz)
        features[f'{joint}_speed_max'] = np.nanmax(speed)

        # Values at key frames
        for frame in [100, 110, 120, 130]:
            if frame < len(z):
                features[f'{joint}_z_f{frame}'] = z[frame]

    return features


def extract_depth_features(timeseries, kp_idx):
    """Extract temporal features optimized for depth prediction."""
    features = {}

    # Get wrist trajectory for key frame detection
    if 'right_wrist' in kp_idx:
        i = kp_idx['right_wrist']
        wrist_z = timeseries[:, i*3 + 2]
        wrist_vz = np.diff(wrist_z) * 60

        # Release frame: max upward velocity
        try:
            release_frame = np.nanargmax(wrist_vz[80:160]) + 80
        except:
            release_frame = 120

        # Set point frame: local max before release
        try:
            set_point_frame = np.nanargmax(wrist_z[60:release_frame]) + 60
        except:
            set_point_frame = release_frame - 10

        # Dip frame: min before set point
        try:
            dip_frame = np.nanargmin(wrist_z[40:set_point_frame]) + 40
        except:
            dip_frame = 80

        # TEMPORAL FEATURES (these have within-player signal)
        features['release_frame'] = release_frame
        features['set_point_frame'] = set_point_frame
        features['dip_frame'] = dip_frame
        features['dip_to_release_duration'] = (release_frame - dip_frame) / 60.0
        features['set_to_release_duration'] = (release_frame - set_point_frame) / 60.0
        features['dip_to_set_duration'] = (set_point_frame - dip_frame) / 60.0

        # Location of maximum (tsfresh showed r=0.56 for this)
        try:
            features['wrist_z_max_location'] = np.nanargmax(wrist_z[60:180]) + 60
        except:
            features['wrist_z_max_location'] = 120

        # Longest strike below mean
        mean_z = np.nanmean(wrist_z)
        below_mean = wrist_z < mean_z
        longest_strike = 0
        current_strike = 0
        for val in below_mean:
            if val:
                current_strike += 1
                longest_strike = max(longest_strike, current_strike)
            else:
                current_strike = 0
        features['wrist_longest_below_mean'] = longest_strike

    # Hip temporal features (also showed signal)
    if 'mid_hip' in kp_idx:
        i = kp_idx['mid_hip']
        hip_z = timeseries[:, i*3 + 2]

        try:
            features['hip_z_max_location'] = np.nanargmax(hip_z[60:180]) + 60
        except:
            features['hip_z_max_location'] = 120

        features['hip_z_range'] = np.nanmax(hip_z) - np.nanmin(hip_z)

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
    print("TARGET-SPECIFIC ENSEMBLE")
    print("=" * 80)
    print("Using best approach for each target:")
    print("  - Angle: KAN (CV 0.010)")
    print("  - Depth: Temporal features (r=0.56 within-player)")
    print("  - Left_right: Sub219 baseline")

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_map(keypoint_cols)

    # Extract features
    print("\nExtracting features...")
    train_angle_features = []
    train_depth_features = []
    train_targets = {'angle': [], 'depth': [], 'left_right': []}
    train_players = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        angle_feat = extract_angle_features(timeseries, kp_idx)
        depth_feat = extract_depth_features(timeseries, kp_idx)

        train_angle_features.append(angle_feat)
        train_depth_features.append(depth_feat)
        train_players.append(metadata['participant_id'])

        for t in train_targets:
            train_targets[t].append(metadata[t])

    df_angle = pd.DataFrame(train_angle_features)
    df_depth = pd.DataFrame(train_depth_features)

    print(f"  Training samples: {len(df_angle)}")
    print(f"  Angle features: {len(df_angle.columns)}")
    print(f"  Depth features: {len(df_depth.columns)}")

    # Prepare data
    angle_cols = df_angle.columns.tolist()
    depth_cols = df_depth.columns.tolist()

    X_angle = df_angle.values
    X_depth = df_depth.values

    X_angle = np.nan_to_num(X_angle, nan=0.0)
    X_depth = np.nan_to_num(X_depth, nan=0.0)

    # Scale
    scaler_angle = StandardScaler()
    scaler_depth = StandardScaler()

    X_angle_scaled = scaler_angle.fit_transform(X_angle)
    X_depth_scaled = scaler_depth.fit_transform(X_depth)

    # Scale targets
    y_data = {}
    target_stats = {}
    for t in ['angle', 'depth', 'left_right']:
        vals = np.array(train_targets[t])
        target_stats[t] = {'min': vals.min(), 'max': vals.max()}
        y_data[t] = (vals - vals.min()) / (vals.max() - vals.min())

    groups = np.array(train_players)
    gkf = GroupKFold(n_splits=5)

    # =========================================================================
    # ANGLE: Train KAN
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANGLE: TRAINING KAN")
    print("=" * 80)

    if KAN_AVAILABLE:
        # Select top features for KAN (to avoid overfitting)
        n_features_kan = 20
        feature_importance = np.abs(X_angle_scaled).std(axis=0)
        top_indices = np.argsort(feature_importance)[-n_features_kan:]
        X_angle_kan = X_angle_scaled[:, top_indices]

        y_angle = y_data['angle']
        angle_mses = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_angle_kan, y_angle, groups)):
            X_tr = torch.FloatTensor(X_angle_kan[train_idx])
            X_val = torch.FloatTensor(X_angle_kan[val_idx])
            y_tr = torch.FloatTensor(y_angle[train_idx]).reshape(-1, 1)
            y_val = y_angle[val_idx]

            model = KAN(width=[n_features_kan, 8, 1], grid=3, k=3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_fn = torch.nn.MSELoss()

            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                pred = model(X_tr)
                loss = loss_fn(pred, y_tr)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(X_val).numpy().flatten()
                pred = np.clip(pred, 0, 1)

            mse = mean_squared_error(y_val, pred)
            angle_mses.append(mse)
            print(f"  Fold {fold+1}: MSE = {mse:.6f}")

        print(f"  KAN Angle CV: {np.mean(angle_mses):.6f} +/- {np.std(angle_mses):.6f}")

        # Train final KAN for angle
        X_angle_kan_tensor = torch.FloatTensor(X_angle_kan)
        y_angle_tensor = torch.FloatTensor(y_angle).reshape(-1, 1)

        kan_angle = KAN(width=[n_features_kan, 8, 1], grid=3, k=3)
        optimizer = torch.optim.Adam(kan_angle.parameters(), lr=0.01)

        kan_angle.train()
        for epoch in range(150):
            optimizer.zero_grad()
            pred = kan_angle(X_angle_kan_tensor)
            loss = loss_fn(pred, y_angle_tensor)
            loss.backward()
            optimizer.step()

        kan_angle.eval()
    else:
        print("  KAN not available, using Ridge fallback")
        kan_angle = None
        top_indices = None

    # =========================================================================
    # DEPTH: Train Temporal Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("DEPTH: TRAINING TEMPORAL MODEL")
    print("=" * 80)

    y_depth = y_data['depth']
    depth_mses = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_depth_scaled, y_depth, groups)):
        X_tr, X_val = X_depth_scaled[train_idx], X_depth_scaled[val_idx]
        y_tr, y_val = y_depth[train_idx], y_depth[val_idx]

        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        pred = np.clip(pred, 0, 1)

        mse = mean_squared_error(y_val, pred)
        depth_mses.append(mse)
        print(f"  Fold {fold+1}: MSE = {mse:.6f}")

    print(f"  Temporal Depth CV: {np.mean(depth_mses):.6f} +/- {np.std(depth_mses):.6f}")

    # Train final depth model
    depth_model = Ridge(alpha=10.0)
    depth_model.fit(X_depth_scaled, y_depth)

    # =========================================================================
    # EXTRACT TEST FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING TEST FEATURES")
    print("=" * 80)

    test_angle_features = []
    test_depth_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        angle_feat = extract_angle_features(timeseries, kp_idx)
        depth_feat = extract_depth_features(timeseries, kp_idx)

        test_angle_features.append(angle_feat)
        test_depth_features.append(depth_feat)
        test_ids.append(metadata['id'])

    df_test_angle = pd.DataFrame(test_angle_features)
    df_test_depth = pd.DataFrame(test_depth_features)

    # Ensure same columns
    for col in angle_cols:
        if col not in df_test_angle.columns:
            df_test_angle[col] = 0.0
    for col in depth_cols:
        if col not in df_test_depth.columns:
            df_test_depth[col] = 0.0

    X_test_angle = df_test_angle[angle_cols].values
    X_test_depth = df_test_depth[depth_cols].values

    X_test_angle = np.nan_to_num(X_test_angle, nan=0.0)
    X_test_depth = np.nan_to_num(X_test_depth, nan=0.0)

    X_test_angle_scaled = scaler_angle.transform(X_test_angle)
    X_test_depth_scaled = scaler_depth.transform(X_test_depth)

    print(f"  Test samples: {len(test_ids)}")

    # =========================================================================
    # MAKE PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS")
    print("=" * 80)

    # Angle: KAN
    if KAN_AVAILABLE and kan_angle is not None:
        X_test_angle_kan = X_test_angle_scaled[:, top_indices]
        with torch.no_grad():
            angle_pred = kan_angle(torch.FloatTensor(X_test_angle_kan)).numpy().flatten()
        angle_pred = np.clip(angle_pred, 0, 1)
        print(f"  Angle: KAN predictions, std={np.std(angle_pred):.4f}")
    else:
        # Fallback to Ridge
        angle_model = Ridge(alpha=10.0)
        angle_model.fit(X_angle_scaled, y_data['angle'])
        angle_pred = angle_model.predict(X_test_angle_scaled)
        angle_pred = np.clip(angle_pred, 0, 1)
        print(f"  Angle: Ridge fallback, std={np.std(angle_pred):.4f}")

    # Depth: Temporal
    depth_pred = depth_model.predict(X_test_depth_scaled)
    depth_pred = np.clip(depth_pred, 0, 1)
    print(f"  Depth: Temporal predictions, mean={np.mean(depth_pred):.4f}")

    # Left_right: Use Sub219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub219_indexed = sub219.set_index('id')
    lr_pred = sub219_indexed.loc[test_ids, 'scaled_left_right'].values
    print(f"  Left_right: Sub219 baseline")

    # =========================================================================
    # CREATE SUBMISSIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CREATING SUBMISSIONS")
    print("=" * 80)

    sub_num = get_next_submission_number()

    # Pure target-specific ensemble
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': angle_pred,
        'scaled_depth': depth_pred,
        'scaled_left_right': lr_pred
    })

    # Calibrate depth
    submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        submission[col] = submission[col].clip(0, 1)

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Sub {sub_num}: Target-specific ensemble (KAN angle, temporal depth, Sub219 lr)")
    print(f"    angle_std={submission['scaled_angle'].std():.4f}")
    print(f"    depth_mean={submission['scaled_depth'].mean():.4f}")
    sub_num += 1

    # Blended versions with Sub219
    for angle_weight in [0.3, 0.5, 0.7]:
        for depth_weight in [0.3, 0.5, 0.7]:
            blended = pd.DataFrame({'id': test_ids})

            # Blend angle
            blended['scaled_angle'] = angle_weight * angle_pred + (1 - angle_weight) * sub219_indexed.loc[test_ids, 'scaled_angle'].values

            # Blend depth
            blended['scaled_depth'] = depth_weight * depth_pred + (1 - depth_weight) * sub219_indexed.loc[test_ids, 'scaled_depth'].values

            # Keep Sub219 left_right
            blended['scaled_left_right'] = lr_pred

            # Calibrate
            blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
            for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
                blended[col] = blended[col].clip(0, 1)

            sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            blended.to_csv(sub_path, index=False)
            print(f"  Sub {sub_num}: {int(angle_weight*100)}% KAN angle, {int(depth_weight*100)}% temporal depth")
            sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Target-specific ensemble combines:")
    print(f"  - Angle: KAN (CV {np.mean(angle_mses):.4f})" if KAN_AVAILABLE else "  - Angle: Ridge fallback")
    print(f"  - Depth: Temporal (CV {np.mean(depth_mses):.4f})")
    print("  - Left_right: Sub219 (best available)")


if __name__ == "__main__":
    main()
