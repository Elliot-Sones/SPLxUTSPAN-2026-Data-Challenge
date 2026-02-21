"""
Bone Length Optimization for Anatomical Denoising

The key insight: bone lengths should be constant throughout a shot.
Variations in measured bone lengths are measurement noise.

This script:
1. Computes bone lengths for each frame
2. Optimizes joint positions to enforce constant bone lengths
3. Uses denoised trajectories for feature extraction
4. Tests if anatomically-consistent data improves predictions

Bone length optimization acts as a physics-informed denoising method.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import sys
import warnings
warnings.filterwarnings('ignore')

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


# Define skeleton structure (bones)
SKELETON_BONES = [
    ('right_wrist', 'right_elbow'),
    ('right_elbow', 'right_shoulder'),
    ('right_shoulder', 'neck'),
    ('left_shoulder', 'neck'),
    ('left_wrist', 'left_elbow'),
    ('left_elbow', 'left_shoulder'),
    ('neck', 'mid_hip'),
    ('mid_hip', 'right_hip'),
    ('mid_hip', 'left_hip'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
]


def compute_bone_length(p1, p2):
    """Compute Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2)**2))


def get_joint_position(timeseries, kp_idx, joint, frame):
    """Get 3D position of a joint at a frame."""
    if joint not in kp_idx:
        return None
    i = kp_idx[joint]
    return timeseries[frame, i*3:i*3+3].copy()


def estimate_bone_lengths(timeseries, kp_idx):
    """
    Estimate constant bone lengths from trajectory data.
    Uses median across frames for robustness.
    """
    bone_lengths = {}

    for joint1, joint2 in SKELETON_BONES:
        if joint1 not in kp_idx or joint2 not in kp_idx:
            continue

        lengths = []
        for frame in range(len(timeseries)):
            p1 = get_joint_position(timeseries, kp_idx, joint1, frame)
            p2 = get_joint_position(timeseries, kp_idx, joint2, frame)

            if p1 is not None and p2 is not None:
                if not np.any(np.isnan(p1)) and not np.any(np.isnan(p2)):
                    lengths.append(compute_bone_length(p1, p2))

        if lengths:
            # Use median for robustness to outliers
            bone_lengths[(joint1, joint2)] = np.median(lengths)

    return bone_lengths


def denoise_with_bone_constraints(timeseries, kp_idx, bone_lengths, weight=0.5):
    """
    Denoise joint positions using bone length constraints.

    Simple approach: For each frame, adjust positions to better match
    expected bone lengths while staying close to original measurements.

    Uses iterative adjustment rather than full optimization for speed.
    """
    denoised = timeseries.copy()
    n_frames = len(timeseries)

    # First, apply temporal smoothing
    for joint in kp_idx:
        i = kp_idx[joint]
        for coord in range(3):
            col = denoised[:, i*3 + coord]
            if not np.any(np.isnan(col)):
                # Savitzky-Golay filter for smooth derivatives
                try:
                    denoised[:, i*3 + coord] = savgol_filter(col, window_length=11, polyorder=3)
                except:
                    pass

    # Then, enforce bone length constraints
    for iteration in range(3):  # Few iterations
        for frame in range(n_frames):
            for (joint1, joint2), target_length in bone_lengths.items():
                if joint1 not in kp_idx or joint2 not in kp_idx:
                    continue

                i1, i2 = kp_idx[joint1], kp_idx[joint2]
                p1 = denoised[frame, i1*3:i1*3+3]
                p2 = denoised[frame, i2*3:i2*3+3]

                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue

                current_length = compute_bone_length(p1, p2)
                if current_length < 1e-6:
                    continue

                # Compute correction
                error = current_length - target_length
                direction = (p2 - p1) / current_length

                # Distribute correction between both joints
                correction = direction * error * weight * 0.5
                denoised[frame, i1*3:i1*3+3] += correction
                denoised[frame, i2*3:i2*3+3] -= correction

    return denoised


def extract_features(timeseries, kp_idx):
    """Extract features from trajectory."""
    features = {}

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'mid_hip', 'right_knee']

    # Detect key frames
    if 'right_wrist' in kp_idx:
        i = kp_idx['right_wrist']
        wrist_z = timeseries[:, i*3 + 2]
        wrist_vz = np.gradient(wrist_z) * 60

        try:
            release_frame = np.nanargmax(wrist_vz[80:160]) + 80
        except:
            release_frame = 120

        features['release_frame'] = release_frame
    else:
        release_frame = 120

    for joint in key_joints:
        if joint not in kp_idx:
            continue

        i = kp_idx[joint]
        z = timeseries[:, i*3 + 2]
        vz = np.gradient(z) * 60

        features[f'{joint}_z_mean'] = np.nanmean(z)
        features[f'{joint}_z_std'] = np.nanstd(z)
        features[f'{joint}_z_max'] = np.nanmax(z)
        features[f'{joint}_vz_max'] = np.nanmax(vz)

        if release_frame < len(z):
            features[f'{joint}_z_release'] = z[release_frame]
        if release_frame < len(vz):
            features[f'{joint}_vz_release'] = vz[release_frame]

    return features


def compute_bone_length_variance(timeseries, kp_idx):
    """Compute variance in bone lengths (measure of noise)."""
    variances = {}

    for joint1, joint2 in SKELETON_BONES:
        if joint1 not in kp_idx or joint2 not in kp_idx:
            continue

        lengths = []
        for frame in range(len(timeseries)):
            p1 = get_joint_position(timeseries, kp_idx, joint1, frame)
            p2 = get_joint_position(timeseries, kp_idx, joint2, frame)

            if p1 is not None and p2 is not None:
                if not np.any(np.isnan(p1)) and not np.any(np.isnan(p2)):
                    lengths.append(compute_bone_length(p1, p2))

        if lengths:
            variances[(joint1, joint2)] = np.std(lengths)

    return variances


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
    print("BONE LENGTH OPTIMIZATION FOR ANATOMICAL DENOISING")
    print("=" * 80)
    print("Key insight: Bone lengths should be constant. Variations are noise.")

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_map(keypoint_cols)

    # =========================================================================
    # ANALYSIS: Bone length variance
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS: BONE LENGTH VARIANCE")
    print("=" * 80)

    all_variances = []
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        variances = compute_bone_length_variance(timeseries, kp_idx)
        all_variances.append(variances)
        if i >= 10:  # Sample
            break

    print("\nBone length standard deviations (should be ~0 for perfect data):")
    bones_to_show = [
        ('right_wrist', 'right_elbow'),
        ('right_elbow', 'right_shoulder'),
        ('right_knee', 'right_ankle'),
    ]

    for bone in bones_to_show:
        if bone in all_variances[0]:
            stds = [v.get(bone, 0) for v in all_variances]
            print(f"  {bone[0]:20s} - {bone[1]:20s}: {np.mean(stds):.4f} m")

    # =========================================================================
    # EXPERIMENT 1: Original vs Denoised Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ORIGINAL VS DENOISED FEATURES")
    print("=" * 80)

    # Extract features from original data
    print("\nExtracting features from original data...")
    train_original = []
    train_denoised = []
    train_targets = {'angle': [], 'depth': [], 'left_right': []}
    train_players = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        # Original features
        feat_orig = extract_features(timeseries, kp_idx)

        # Denoised features
        bone_lengths = estimate_bone_lengths(timeseries, kp_idx)
        denoised_ts = denoise_with_bone_constraints(timeseries, kp_idx, bone_lengths)
        feat_denoised = extract_features(denoised_ts, kp_idx)

        train_original.append(feat_orig)
        train_denoised.append(feat_denoised)
        train_players.append(metadata['participant_id'])

        for t in train_targets:
            train_targets[t].append(metadata[t])

        if i % 100 == 0:
            print(f"  Processed {i+1} shots...")

    df_original = pd.DataFrame(train_original)
    df_denoised = pd.DataFrame(train_denoised)

    print(f"  Training samples: {len(df_original)}")
    print(f"  Features: {len(df_original.columns)}")

    # Scale targets
    targets = ['angle', 'depth', 'left_right']
    y_data = {}
    for t in targets:
        vals = np.array(train_targets[t])
        y_data[t] = (vals - vals.min()) / (vals.max() - vals.min())

    groups = np.array(train_players)

    # Prepare feature matrices
    feature_cols = df_original.columns.tolist()
    X_original = df_original[feature_cols].values
    X_denoised = df_denoised[feature_cols].values

    X_original = np.nan_to_num(X_original, nan=0.0)
    X_denoised = np.nan_to_num(X_denoised, nan=0.0)

    scaler_orig = StandardScaler()
    scaler_denoise = StandardScaler()

    X_orig_scaled = scaler_orig.fit_transform(X_original)
    X_denoise_scaled = scaler_denoise.fit_transform(X_denoised)

    gkf = GroupKFold(n_splits=5)

    print("\nOriginal data CV:")
    results_orig = {}
    for target in targets:
        y = y_data[target]
        mses = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_orig_scaled, y, groups)):
            X_tr, X_val = X_orig_scaled[train_idx], X_orig_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=10.0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            pred = np.clip(pred, 0, 1)

            mse = mean_squared_error(y_val, pred)
            mses.append(mse)

        results_orig[target] = np.mean(mses)
        print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

    print(f"    Total: {sum(results_orig.values())/3:.6f}")

    print("\nDenoised data CV:")
    results_denoise = {}
    for target in targets:
        y = y_data[target]
        mses = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_denoise_scaled, y, groups)):
            X_tr, X_val = X_denoise_scaled[train_idx], X_denoise_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=10.0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            pred = np.clip(pred, 0, 1)

            mse = mean_squared_error(y_val, pred)
            mses.append(mse)

        results_denoise[target] = np.mean(mses)
        print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

    print(f"    Total: {sum(results_denoise.values())/3:.6f}")

    # =========================================================================
    # EXPERIMENT 2: Add bone length features
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: BONE LENGTH AS FEATURES")
    print("=" * 80)
    print("Using estimated bone lengths as player-specific features...")

    # Extract bone length features
    train_bone_features = []

    for metadata, timeseries in iterate_shots(train=True):
        bone_lengths = estimate_bone_lengths(timeseries, kp_idx)

        # Use bone lengths as features
        bone_feat = {}
        for (j1, j2), length in bone_lengths.items():
            bone_feat[f'bone_{j1}_{j2}'] = length

        train_bone_features.append(bone_feat)

    df_bone = pd.DataFrame(train_bone_features)
    X_bone = df_bone.values
    X_bone = np.nan_to_num(X_bone, nan=0.0)

    scaler_bone = StandardScaler()
    X_bone_scaled = scaler_bone.fit_transform(X_bone)

    # Combine with original features
    X_combined = np.hstack([X_orig_scaled, X_bone_scaled])

    print(f"  Original features: {X_orig_scaled.shape[1]}")
    print(f"  Bone features: {X_bone_scaled.shape[1]}")
    print(f"  Combined features: {X_combined.shape[1]}")

    print("\nCombined features CV:")
    results_combined = {}
    for target in targets:
        y = y_data[target]
        mses = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_combined, y, groups)):
            X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=10.0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            pred = np.clip(pred, 0, 1)

            mse = mean_squared_error(y_val, pred)
            mses.append(mse)

        results_combined[target] = np.mean(mses)
        print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

    print(f"    Total: {sum(results_combined.values())/3:.6f}")

    # =========================================================================
    # CREATE SUBMISSIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CREATING SUBMISSIONS")
    print("=" * 80)

    # Extract test features
    test_original = []
    test_denoised = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        feat_orig = extract_features(timeseries, kp_idx)

        bone_lengths = estimate_bone_lengths(timeseries, kp_idx)
        denoised_ts = denoise_with_bone_constraints(timeseries, kp_idx, bone_lengths)
        feat_denoised = extract_features(denoised_ts, kp_idx)

        test_original.append(feat_orig)
        test_denoised.append(feat_denoised)
        test_ids.append(metadata['id'])

    df_test_orig = pd.DataFrame(test_original)
    df_test_denoise = pd.DataFrame(test_denoised)

    for col in feature_cols:
        if col not in df_test_orig.columns:
            df_test_orig[col] = 0.0
        if col not in df_test_denoise.columns:
            df_test_denoise[col] = 0.0

    X_test_orig = df_test_orig[feature_cols].values
    X_test_denoise = df_test_denoise[feature_cols].values

    X_test_orig = np.nan_to_num(X_test_orig, nan=0.0)
    X_test_denoise = np.nan_to_num(X_test_denoise, nan=0.0)

    X_test_orig_scaled = scaler_orig.transform(X_test_orig)
    X_test_denoise_scaled = scaler_denoise.transform(X_test_denoise)

    sub_num = get_next_submission_number()

    # Use denoised data for predictions
    predictions = {}
    for target in targets:
        model = Ridge(alpha=10.0)
        model.fit(X_denoise_scaled, y_data[target])
        pred = model.predict(X_test_denoise_scaled)
        predictions[target] = np.clip(pred, 0, 1)

    # Pure denoised submission
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    })

    submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        submission[col] = submission[col].clip(0, 1)

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Sub {sub_num}: Bone-denoised features, angle_std={submission['scaled_angle'].std():.4f}")
    sub_num += 1

    # Blend with Sub219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub219_indexed = sub219.set_index('id')

    for weight in [0.1, 0.2, 0.3]:
        blended = pd.DataFrame({'id': test_ids})

        for target, col in [('angle', 'scaled_angle'), ('depth', 'scaled_depth'), ('left_right', 'scaled_left_right')]:
            bone_pred = predictions[target]
            sub219_pred = sub219_indexed.loc[test_ids, col].values
            blended[col] = weight * bone_pred + (1 - weight) * sub219_pred

        blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blended[col] = blended[col].clip(0, 1)

        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: {int(weight*100)}% bone-denoised + {int((1-weight)*100)}% Sub219")
        sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Bone length optimization enforces anatomical constraints.")
    print("If denoised CV improves, the constraints help remove noise.")

    # Print comparison
    print("\nCV Comparison:")
    print(f"  Original:  {sum(results_orig.values())/3:.6f}")
    print(f"  Denoised:  {sum(results_denoise.values())/3:.6f}")
    print(f"  Combined:  {sum(results_combined.values())/3:.6f}")


if __name__ == "__main__":
    main()
