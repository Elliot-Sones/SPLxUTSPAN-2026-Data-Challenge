"""
SkillMimic-Enhanced Basketball Prediction Model

Combines:
1. SkillMimic biomechanical insights (elbow angle, wrist velocity, kinetic chain)
2. Proven feature engineering from existing models
3. Per-player modeling (which works well)
4. Conservative regularization to prevent overfitting

Key insight from SkillMimic analysis:
- Release occurs at 52-65% of motion (frames 125-155 for SPL's 240 frames)
- Release angle consistently ~70 degrees
- Peak wrist velocity at or near release
- Kinetic chain timing: hip -> shoulder -> elbow -> wrist

Strategy:
- Focus on relative/normalized features (more generalizable across players)
- Use velocity ratios rather than raw velocities
- Combine with player-mean baseline (improves generalization)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_keypoint_columns(keypoint_cols):
    """Parse keypoint columns into a structured format."""
    kp_positions = {}
    for i, col in enumerate(keypoint_cols):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in kp_positions:
                kp_positions[name] = {}
            kp_positions[name][axis] = i
    return kp_positions


def get_pos(keypoints, frame, kp_positions, kp_name):
    """Get 3D position of keypoint at frame."""
    if kp_name not in kp_positions:
        return np.array([np.nan, np.nan, np.nan])
    pos = kp_positions[kp_name]
    x = keypoints[frame, pos.get('x', 0)] if 'x' in pos else np.nan
    y = keypoints[frame, pos.get('y', 0)] if 'y' in pos else np.nan
    z = keypoints[frame, pos.get('z', 0)] if 'z' in pos else np.nan
    return np.array([x, y, z])


def compute_angle(p1, p2, p3):
    """Compute angle at p2 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    cos_ang = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos_ang, -1, 1)) * 180 / np.pi


def extract_skillmimic_enhanced_features(keypoints, kp_positions):
    """
    Extract SkillMimic-enhanced features focusing on generalizability.

    Key design principles:
    1. Use RELATIVE features (normalized by body size/position)
    2. Use RATIOS rather than absolute values
    3. Focus on BIOMECHANICAL patterns that should generalize
    """
    features = {}
    n_frames = keypoints.shape[0]

    # Key body parts
    joints = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_index',
              'left_shoulder', 'left_elbow', 'left_wrist',
              'right_hip', 'left_hip', 'nose']

    # Check available joints
    available_joints = [j for j in joints if j in kp_positions]

    # Key frames based on SkillMimic timing analysis
    # Release: 52-65% of motion
    release_frames = [125, 140, 155]

    for frame in release_frames:
        suffix = f"_f{frame}"

        # Get positions
        r_shoulder = get_pos(keypoints, frame, kp_positions, 'right_shoulder')
        r_elbow = get_pos(keypoints, frame, kp_positions, 'right_elbow')
        r_wrist = get_pos(keypoints, frame, kp_positions, 'right_wrist')
        r_index = get_pos(keypoints, frame, kp_positions, 'right_index')
        l_shoulder = get_pos(keypoints, frame, kp_positions, 'left_shoulder')
        l_wrist = get_pos(keypoints, frame, kp_positions, 'left_wrist')
        r_hip = get_pos(keypoints, frame, kp_positions, 'right_hip')
        l_hip = get_pos(keypoints, frame, kp_positions, 'left_hip')
        nose = get_pos(keypoints, frame, kp_positions, 'nose')

        # Body reference points
        hip_center = (r_hip + l_hip) / 2
        shoulder_center = (r_shoulder + l_shoulder) / 2

        # === RELATIVE POSITION FEATURES ===

        # 1. Wrist height relative to shoulder height (normalized)
        if not np.any(np.isnan([r_wrist, r_shoulder, hip_center])):
            shoulder_height = r_shoulder[1] - hip_center[1]
            if abs(shoulder_height) > 1e-8:
                features[f'wrist_rel_shoulder{suffix}'] = (r_wrist[1] - r_shoulder[1]) / shoulder_height

        # 2. Arm extension ratio (generalizes across body sizes)
        if not np.any(np.isnan([r_shoulder, r_elbow, r_wrist])):
            upper_arm = np.linalg.norm(r_elbow - r_shoulder)
            forearm = np.linalg.norm(r_wrist - r_elbow)
            total_arm = upper_arm + forearm
            direct = np.linalg.norm(r_wrist - r_shoulder)
            if total_arm > 1e-8:
                features[f'arm_extension{suffix}'] = direct / total_arm

        # 3. Elbow angle (biomechanical - important from SkillMimic)
        if not np.any(np.isnan([r_shoulder, r_elbow, r_wrist])):
            elbow_ang = compute_angle(r_shoulder, r_elbow, r_wrist)
            if not np.isnan(elbow_ang):
                features[f'elbow_angle{suffix}'] = elbow_ang

        # 4. Shoulder elevation angle (relative to vertical)
        if not np.any(np.isnan([r_shoulder, r_wrist])):
            arm_vec = r_wrist - r_shoulder
            vertical = np.array([0, 1, 0])
            arm_len = np.linalg.norm(arm_vec)
            if arm_len > 1e-8:
                cos_elev = np.dot(arm_vec, vertical) / arm_len
                features[f'arm_vertical_angle{suffix}'] = np.arccos(np.clip(cos_elev, -1, 1)) * 180 / np.pi

        # 5. Hand position relative to nose (release point indicator)
        if not np.any(np.isnan([r_wrist, nose, hip_center])):
            body_height = nose[1] - hip_center[1]
            if abs(body_height) > 1e-8:
                features[f'wrist_above_nose{suffix}'] = (r_wrist[1] - nose[1]) / body_height

        # 6. Asymmetry features (left vs right)
        if not np.any(np.isnan([r_wrist, l_wrist, shoulder_center])):
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            if shoulder_width > 1e-8:
                features[f'wrist_asymmetry_x{suffix}'] = (r_wrist[0] - l_wrist[0]) / shoulder_width
                features[f'wrist_asymmetry_y{suffix}'] = (r_wrist[1] - l_wrist[1]) / shoulder_width

    # === VELOCITY FEATURES (relative) ===

    # Get player's average arm length for normalization
    arm_lengths = []
    for f in range(0, n_frames, 10):
        r_shoulder = get_pos(keypoints, f, kp_positions, 'right_shoulder')
        r_elbow = get_pos(keypoints, f, kp_positions, 'right_elbow')
        r_wrist = get_pos(keypoints, f, kp_positions, 'right_wrist')
        if not np.any(np.isnan([r_shoulder, r_elbow, r_wrist])):
            arm_len = np.linalg.norm(r_elbow - r_shoulder) + np.linalg.norm(r_wrist - r_elbow)
            arm_lengths.append(arm_len)

    avg_arm_length = np.median(arm_lengths) if arm_lengths else 1.0

    for frame in [135, 145, 155]:
        suffix = f"_v{frame}"
        if frame > 0:
            r_wrist_curr = get_pos(keypoints, frame, kp_positions, 'right_wrist')
            r_wrist_prev = get_pos(keypoints, frame - 1, kp_positions, 'right_wrist')

            if not np.any(np.isnan([r_wrist_curr, r_wrist_prev])):
                vel = (r_wrist_curr - r_wrist_prev) * 60  # 60 FPS

                # Velocity normalized by arm length (generalizable)
                vel_mag = np.linalg.norm(vel)
                features[f'wrist_vel_norm{suffix}'] = vel_mag / avg_arm_length

                # Vertical velocity ratio
                if vel_mag > 1e-8:
                    features[f'wrist_vel_y_ratio{suffix}'] = vel[1] / vel_mag
                    features[f'wrist_vel_z_ratio{suffix}'] = vel[2] / vel_mag

    # === TEMPORAL PATTERN FEATURES ===

    # Find peak velocity frame
    velocities = []
    for f in range(1, n_frames):
        r_wrist_curr = get_pos(keypoints, f, kp_positions, 'right_wrist')
        r_wrist_prev = get_pos(keypoints, f - 1, kp_positions, 'right_wrist')
        if not np.any(np.isnan([r_wrist_curr, r_wrist_prev])):
            vel = np.linalg.norm((r_wrist_curr - r_wrist_prev) * 60)
            velocities.append((f, vel))

    if velocities:
        peak_frame, peak_vel = max(velocities, key=lambda x: x[1])
        features['peak_vel_frame_ratio'] = peak_frame / n_frames
        features['peak_vel_normalized'] = peak_vel / avg_arm_length

    # Kinetic chain timing (from SkillMimic insight)
    joints_chain = ['right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
    peak_times = {}

    for joint in joints_chain:
        if joint not in kp_positions:
            continue
        vels = []
        for f in range(1, n_frames):
            curr = get_pos(keypoints, f, kp_positions, joint)
            prev = get_pos(keypoints, f - 1, kp_positions, joint)
            if not np.any(np.isnan([curr, prev])):
                vels.append((f, np.linalg.norm((curr - prev) * 60)))
        if vels:
            peak_times[joint] = max(vels, key=lambda x: x[1])[0]

    if len(peak_times) >= 3:
        # Timing delay: wrist peak should come after hip peak
        if 'right_hip' in peak_times and 'right_wrist' in peak_times:
            features['kinetic_chain_delay'] = (
                peak_times['right_wrist'] - peak_times['right_hip']
            ) / n_frames

    # Wrist height trajectory
    heights = []
    for f in range(n_frames):
        r_wrist = get_pos(keypoints, f, kp_positions, 'right_wrist')
        hip_center = (
            get_pos(keypoints, f, kp_positions, 'right_hip') +
            get_pos(keypoints, f, kp_positions, 'left_hip')
        ) / 2
        if not np.any(np.isnan([r_wrist, hip_center])):
            heights.append((f, r_wrist[1] - hip_center[1]))

    if heights:
        max_h = max(heights, key=lambda x: x[1])
        features['max_height_frame_ratio'] = max_h[0] / n_frames

    return features


def run_lopo_cv(X, y, groups):
    """Run Leave-One-Player-Out CV."""
    logo = LeaveOneGroupOut()
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=10.0)  # Conservative regularization

    scores = cross_val_score(
        model, X_scaled, y,
        cv=logo.split(X_scaled, y, groups),
        scoring='neg_mean_squared_error'
    )

    return -scores.mean(), scores.std()


def main():
    print("="*80)
    print("SKILLMIMIC-ENHANCED MODEL FOR SPL")
    print("="*80)

    # Load data
    print("\n[Step 1] Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    participant_ids = meta['participant_id'].values

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Participants: {np.unique(participant_ids)}")

    # Parse keypoint structure
    kp_positions = parse_keypoint_columns(keypoint_cols)
    print(f"\n  Available body parts: {list(kp_positions.keys())[:10]}...")

    # Extract features
    print("\n[Step 2] Extracting SkillMimic-enhanced features...")
    all_features = []
    for i in range(X.shape[0]):
        if i % 50 == 0:
            print(f"  Shot {i+1}/{X.shape[0]}...")
        features = extract_skillmimic_enhanced_features(X[i], kp_positions)
        all_features.append(features)

    feature_df = pd.DataFrame(all_features)

    # Fill NaN
    for col in feature_df.columns:
        med = feature_df[col].median()
        feature_df[col] = feature_df[col].fillna(med if not pd.isna(med) else 0)

    print(f"\n  Extracted {len(feature_df.columns)} features")
    print("\n  Feature list:")
    for col in sorted(feature_df.columns):
        print(f"    {col}")

    # Evaluate
    print("\n[Step 3] LOPO CV Evaluation...")
    target_names = ['angle', 'depth', 'left_right']
    results = []

    for target_idx, target_name in enumerate(target_names):
        mse, std = run_lopo_cv(
            feature_df.values,
            y[:, target_idx],
            participant_ids
        )
        results.append({
            'target': target_name,
            'mse': mse,
            'std': std,
            'rmse': np.sqrt(mse)
        })
        print(f"  {target_name}: MSE = {mse:.6f} (+/- {std:.6f}), RMSE = {np.sqrt(mse):.4f}")

    # Combined with player-mean baseline
    print("\n[Step 4] Testing player-mean augmented model...")

    # Add player mean as features
    for target_idx, target_name in enumerate(target_names):
        y_target = y[:, target_idx]
        player_means = {}
        for pid in np.unique(participant_ids):
            mask = participant_ids == pid
            player_means[pid] = y_target[mask].mean()

        feature_df[f'player_mean_{target_name}'] = [
            player_means[pid] for pid in participant_ids
        ]

    # Re-evaluate with player means
    results_augmented = []
    for target_idx, target_name in enumerate(target_names):
        mse, std = run_lopo_cv(
            feature_df.values,
            y[:, target_idx],
            participant_ids
        )
        results_augmented.append({
            'target': target_name,
            'mse': mse,
            'std': std,
            'rmse': np.sqrt(mse)
        })
        print(f"  {target_name} (augmented): MSE = {mse:.6f}, RMSE = {np.sqrt(mse):.4f}")

    # Save features
    print("\n[Step 5] Saving results...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    feature_df.to_csv(OUTPUT_DIR / "skillmimic_enhanced_features.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'skillmimic_enhanced_features.csv'}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "skillmimic_enhanced_results.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'skillmimic_enhanced_results.csv'}")

    # Feature importance
    print("\n[Step 6] Feature correlations with targets...")
    for target_idx, target_name in enumerate(target_names):
        y_target = y[:, target_idx]
        corrs = []
        for col in feature_df.columns:
            if 'player_mean' in col:
                continue
            corr = np.corrcoef(feature_df[col].values, y_target)[0, 1]
            if not np.isnan(corr):
                corrs.append((col, abs(corr)))
        corrs.sort(key=lambda x: -x[1])

        print(f"\n  Top 5 for {target_name}:")
        for feat, corr in corrs[:5]:
            print(f"    {feat}: |r| = {corr:.4f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
