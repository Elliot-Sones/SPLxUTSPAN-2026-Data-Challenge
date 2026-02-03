"""
SkillMimic-Inspired Feature Engineering for SPL Basketball Challenge

Based on analysis of SkillMimic shot style data:
- Release occurs at 52-62% through the motion
- Release angle is consistently ~70 degrees vertical
- Peak wrist velocity occurs at or near release
- Elbow extension varies significantly between styles (15-69 degrees)

This script:
1. Maps SPL keypoints to biomechanical joints
2. Extracts features inspired by SkillMimic patterns
3. Tests feature predictive power for angle/depth/left_right targets
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# SPL Keypoint Mapping (derived from the 69 keypoints)
# Standard body pose format: COCO + hand keypoints
# These indices are approximate and should be verified against actual data
SPL_KEYPOINT_INDICES = {
    # Core body (MediaPipe/COCO style)
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
    # Hand keypoints follow (21 per hand)
}

# Key frames based on SkillMimic analysis
# Release timing: 52-62% of motion, SPL has 240 frames
RELEASE_FRAME_RANGE = (125, 155)  # ~52-65% of 240 frames


def get_keypoint_3d(keypoints, frame, keypoint_name, col_to_idx):
    """Get 3D position of a keypoint at a specific frame."""
    kp_idx = SPL_KEYPOINT_INDICES.get(keypoint_name)
    if kp_idx is None:
        return np.array([0.0, 0.0, 0.0])

    # Column indices for x, y, z
    # Assuming columns are ordered as: kp0_x, kp0_y, kp0_z, kp1_x, ...
    base_idx = kp_idx * 3
    if base_idx + 2 >= keypoints.shape[2]:
        return np.array([0.0, 0.0, 0.0])

    return keypoints[frame, base_idx:base_idx+3]


def compute_joint_angle(p1, p2, p3):
    """Compute angle at p2 between p1-p2-p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-8 or v2_norm < 1e-8:
        return 0.0

    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle) * 180 / np.pi


def compute_velocity(keypoints, frame, keypoint_idx, fps=60):
    """Compute velocity of a keypoint at a specific frame."""
    if frame < 1:
        return np.array([0.0, 0.0, 0.0])

    base_idx = keypoint_idx * 3
    if base_idx + 2 >= keypoints.shape[1]:
        return np.array([0.0, 0.0, 0.0])

    pos_curr = keypoints[frame, base_idx:base_idx+3]
    pos_prev = keypoints[frame-1, base_idx:base_idx+3]

    return (pos_curr - pos_prev) * fps


def extract_skillmimic_features_single(keypoints, keypoint_cols):
    """
    Extract SkillMimic-inspired features from a single shot's keypoint data.

    Args:
        keypoints: np.ndarray of shape (240, 207) - frames x features
        keypoint_cols: list of column names

    Returns:
        dict: Feature name -> value
    """
    features = {}
    n_frames = keypoints.shape[0]

    # Build column to index mapping
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    # Parse keypoint columns to understand structure
    # Columns are like: "nose_x", "nose_y", "nose_z", "left_eye_inner_x", ...
    kp_positions = {}
    for col in keypoint_cols:
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in kp_positions:
                kp_positions[name] = {}
            kp_positions[name][axis] = col_to_idx[col]

    def get_pos(frame, kp_name):
        """Get 3D position of keypoint at frame."""
        if kp_name not in kp_positions:
            return np.array([np.nan, np.nan, np.nan])
        pos = kp_positions[kp_name]
        x = keypoints[frame, pos.get('x', 0)] if 'x' in pos else np.nan
        y = keypoints[frame, pos.get('y', 0)] if 'y' in pos else np.nan
        z = keypoints[frame, pos.get('z', 0)] if 'z' in pos else np.nan
        return np.array([x, y, z])

    # Key frames for analysis (based on SkillMimic timing)
    release_frames = [125, 140, 155]  # 52%, 58%, 65% of motion
    peak_frames = [135, 145, 155]     # Around peak velocity

    # ========================================
    # POSITION-BASED FEATURES
    # ========================================

    for frame in release_frames:
        suffix = f"_f{frame}"

        # Right arm positions
        r_shoulder = get_pos(frame, 'right_shoulder')
        r_elbow = get_pos(frame, 'right_elbow')
        r_wrist = get_pos(frame, 'right_wrist')
        r_index = get_pos(frame, 'right_index')

        # Left arm for comparison
        l_shoulder = get_pos(frame, 'left_shoulder')
        l_elbow = get_pos(frame, 'left_elbow')
        l_wrist = get_pos(frame, 'left_wrist')

        # Hip and core
        r_hip = get_pos(frame, 'right_hip')
        l_hip = get_pos(frame, 'left_hip')

        # 1. Wrist height (relative to hip center)
        hip_center = (r_hip + l_hip) / 2
        if not np.isnan(r_wrist[1]) and not np.isnan(hip_center[1]):
            features[f'r_wrist_height{suffix}'] = r_wrist[1] - hip_center[1]

        # 2. Elbow angle (angle at elbow between upper arm and forearm)
        if not np.any(np.isnan([r_shoulder, r_elbow, r_wrist])):
            features[f'elbow_angle{suffix}'] = compute_joint_angle(
                r_shoulder, r_elbow, r_wrist
            )

        # 3. Shoulder angle (angle at shoulder between torso and upper arm)
        if not np.any(np.isnan([hip_center, r_shoulder, r_elbow])):
            features[f'shoulder_angle{suffix}'] = compute_joint_angle(
                hip_center, r_shoulder, r_elbow
            )

        # 4. Arm extension ratio
        if not np.any(np.isnan([r_shoulder, r_elbow, r_wrist])):
            upper_arm_len = np.linalg.norm(r_elbow - r_shoulder)
            forearm_len = np.linalg.norm(r_wrist - r_elbow)
            total_len = upper_arm_len + forearm_len
            direct_len = np.linalg.norm(r_wrist - r_shoulder)
            if total_len > 1e-8:
                features[f'arm_extension_ratio{suffix}'] = direct_len / total_len

        # 5. Wrist position relative to shoulder (shooting angle indicator)
        if not np.any(np.isnan([r_shoulder, r_wrist])):
            wrist_shoulder_vec = r_wrist - r_shoulder
            # Vertical elevation angle
            horizontal = np.linalg.norm([wrist_shoulder_vec[0], wrist_shoulder_vec[2]])
            if horizontal > 1e-8:
                features[f'arm_elevation{suffix}'] = np.arctan2(
                    wrist_shoulder_vec[1], horizontal
                ) * 180 / np.pi

        # 6. Hand symmetry (left vs right wrist positions)
        if not np.any(np.isnan([r_wrist, l_wrist])):
            features[f'wrist_y_diff{suffix}'] = r_wrist[1] - l_wrist[1]
            features[f'wrist_x_diff{suffix}'] = r_wrist[0] - l_wrist[0]

        # 7. Index finger position (fingertip for release point)
        if not np.any(np.isnan([r_index, r_wrist])):
            finger_vec = r_index - r_wrist
            features[f'finger_extension{suffix}'] = np.linalg.norm(finger_vec)
            if np.linalg.norm(finger_vec) > 1e-8:
                features[f'finger_angle_y{suffix}'] = np.arctan2(
                    finger_vec[1], np.linalg.norm([finger_vec[0], finger_vec[2]])
                ) * 180 / np.pi

    # ========================================
    # VELOCITY-BASED FEATURES
    # ========================================

    for frame in peak_frames:
        suffix = f"_v{frame}"

        # Compute velocities
        if frame > 0:
            # Right wrist velocity
            r_wrist_curr = get_pos(frame, 'right_wrist')
            r_wrist_prev = get_pos(frame - 1, 'right_wrist')

            if not np.any(np.isnan([r_wrist_curr, r_wrist_prev])):
                wrist_vel = (r_wrist_curr - r_wrist_prev) * 60  # 60 FPS
                features[f'wrist_velocity_mag{suffix}'] = np.linalg.norm(wrist_vel)
                features[f'wrist_velocity_y{suffix}'] = wrist_vel[1]
                features[f'wrist_velocity_z{suffix}'] = wrist_vel[2]  # Forward velocity

            # Right elbow velocity
            r_elbow_curr = get_pos(frame, 'right_elbow')
            r_elbow_prev = get_pos(frame - 1, 'right_elbow')

            if not np.any(np.isnan([r_elbow_curr, r_elbow_prev])):
                elbow_vel = (r_elbow_curr - r_elbow_prev) * 60
                features[f'elbow_velocity_mag{suffix}'] = np.linalg.norm(elbow_vel)

    # ========================================
    # TEMPORAL PATTERN FEATURES
    # ========================================

    # Find peak wrist velocity frame
    wrist_velocities = []
    for f in range(1, n_frames):
        r_wrist_curr = get_pos(f, 'right_wrist')
        r_wrist_prev = get_pos(f - 1, 'right_wrist')

        if not np.any(np.isnan([r_wrist_curr, r_wrist_prev])):
            vel = np.linalg.norm((r_wrist_curr - r_wrist_prev) * 60)
            wrist_velocities.append((f, vel))

    if wrist_velocities:
        peak_frame, peak_vel = max(wrist_velocities, key=lambda x: x[1])
        features['peak_wrist_velocity'] = peak_vel
        features['peak_velocity_frame_ratio'] = peak_frame / n_frames

        # Velocity at key percentages of motion
        for pct in [50, 60, 70]:
            target_frame = int(n_frames * pct / 100)
            matches = [v for f, v in wrist_velocities if abs(f - target_frame) < 3]
            if matches:
                features[f'wrist_velocity_pct{pct}'] = np.mean(matches)

    # Wrist height trajectory
    wrist_heights = []
    for f in range(n_frames):
        r_wrist = get_pos(f, 'right_wrist')
        if not np.isnan(r_wrist[1]):
            wrist_heights.append((f, r_wrist[1]))

    if wrist_heights:
        heights = [h for _, h in wrist_heights]
        features['wrist_height_max'] = max(heights)
        features['wrist_height_min'] = min(heights)
        features['wrist_height_range'] = max(heights) - min(heights)

        # Frame of max height
        max_height_frame = max(wrist_heights, key=lambda x: x[1])[0]
        features['max_height_frame_ratio'] = max_height_frame / n_frames

    # ========================================
    # KINETIC CHAIN FEATURES
    # ========================================

    # Sequential activation: hip -> shoulder -> elbow -> wrist
    # Based on SkillMimic observation of kinetic chain

    # Find peak velocity frame for each joint
    joints_to_track = ['right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
    peak_frames_by_joint = {}

    for joint in joints_to_track:
        velocities = []
        for f in range(1, n_frames):
            curr = get_pos(f, joint)
            prev = get_pos(f - 1, joint)
            if not np.any(np.isnan([curr, prev])):
                vel = np.linalg.norm((curr - prev) * 60)
                velocities.append((f, vel))

        if velocities:
            peak_frame = max(velocities, key=lambda x: x[1])[0]
            peak_frames_by_joint[joint] = peak_frame

    # Kinetic chain timing score (should be sequential: hip first, wrist last)
    if len(peak_frames_by_joint) == 4:
        expected_order = ['right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
        actual_order = sorted(peak_frames_by_joint.keys(),
                             key=lambda x: peak_frames_by_joint[x])

        correct_pairs = sum(1 for i in range(3)
                          if expected_order.index(actual_order[i]) <
                             expected_order.index(actual_order[i+1]))
        features['kinetic_chain_score'] = correct_pairs / 3.0

        # Time delay between hip and wrist peak
        features['hip_to_wrist_delay'] = (
            peak_frames_by_joint['right_wrist'] -
            peak_frames_by_joint['right_hip']
        )

    return features


def extract_all_features(X, keypoint_cols):
    """Extract features from all shots."""
    n_samples = X.shape[0]
    all_features = []

    for i in range(n_samples):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}/{n_samples}...")
        features = extract_skillmimic_features_single(X[i], keypoint_cols)
        all_features.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    # Fill NaN with column median
    for col in df.columns:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)

    return df


def evaluate_features(X_features, y, participant_ids, target_names):
    """Evaluate feature predictive power using LOPO CV."""
    results = []

    logo = LeaveOneGroupOut()
    scaler = StandardScaler()

    for target_idx, target_name in enumerate(target_names):
        y_target = y[:, target_idx]

        # Scale features
        X_scaled = scaler.fit_transform(X_features)

        # LOPO CV
        model = Ridge(alpha=1.0)
        scores = cross_val_score(
            model, X_scaled, y_target,
            cv=logo.split(X_scaled, y_target, participant_ids),
            scoring='r2'
        )

        results.append({
            'target': target_name,
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'min_r2': scores.min(),
            'max_r2': scores.max(),
        })

        print(f"  {target_name}: R2 = {scores.mean():.4f} +/- {scores.std():.4f}")

    return pd.DataFrame(results)


def main():
    print("="*80)
    print("SKILLMIMIC-INSPIRED FEATURE ENGINEERING FOR SPL")
    print("="*80)

    # Load SPL data
    print("\n[Step 1] Loading SPL data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Keypoint columns: {len(keypoint_cols)}")

    # Print sample keypoint columns
    print("\n  Sample keypoint columns:")
    for col in keypoint_cols[:10]:
        print(f"    {col}")

    # Extract features
    print("\n[Step 2] Extracting SkillMimic-inspired features...")
    feature_df = extract_all_features(X, keypoint_cols)

    print(f"\n  Extracted {len(feature_df.columns)} features")
    print("\n  Feature list:")
    for col in sorted(feature_df.columns):
        print(f"    {col}")

    # Save features
    print("\n[Step 3] Saving features...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    feature_df.to_csv(OUTPUT_DIR / "skillmimic_features.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'skillmimic_features.csv'}")

    # Evaluate features
    print("\n[Step 4] Evaluating feature predictive power (LOPO CV)...")
    participant_ids = meta['participant_id'].values
    target_names = ['angle', 'depth', 'left_right']

    results = evaluate_features(
        feature_df.values, y, participant_ids, target_names
    )

    print("\n[Step 5] Results summary:")
    print(results.to_string(index=False))

    # Save results
    results.to_csv(OUTPUT_DIR / "skillmimic_feature_results.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'skillmimic_feature_results.csv'}")

    # Feature importance analysis
    print("\n[Step 6] Feature importance (correlation with targets)...")

    correlations = {}
    for target_idx, target_name in enumerate(target_names):
        y_target = y[:, target_idx]
        corrs = []
        for col in feature_df.columns:
            corr = np.corrcoef(feature_df[col].values, y_target)[0, 1]
            if not np.isnan(corr):
                corrs.append((col, abs(corr)))
        corrs.sort(key=lambda x: -x[1])
        correlations[target_name] = corrs[:10]

        print(f"\n  Top 10 features for {target_name}:")
        for feat, corr in corrs[:10]:
            print(f"    {feat}: |r| = {corr:.4f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
