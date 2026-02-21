"""
Spatial Graph Features for Basketball Free Throw Prediction

Extracts features based on skeleton graph structure:
1. Bone lengths (distances between connected joints)
2. Relative joint positions (wrist relative to shoulder, etc.)
3. Body alignment angles (trunk lean, arm alignment)
4. Pose configuration at key frames (dip, set-point, release)

These capture "how the body is positioned" rather than "how fast things move".
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


def get_joint_position(timeseries, kp_idx, joint_name, frame):
    """Get 3D position of a joint at a specific frame."""
    if joint_name not in kp_idx:
        return np.array([np.nan, np.nan, np.nan])
    i = kp_idx[joint_name]
    frame = min(frame, len(timeseries) - 1)
    return timeseries[frame, i*3:i*3+3]


def compute_distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2)**2))


def compute_angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 (in degrees)."""
    v1 = p1 - p2
    v2 = p3 - p2

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return np.nan

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def detect_key_frames(timeseries, kp_idx):
    """Detect dip, set-point, and release frames."""
    if 'right_wrist' not in kp_idx:
        return 80, 100, 120

    i = kp_idx['right_wrist']
    wrist_z = timeseries[:, i*3 + 2]
    wrist_vz = np.diff(wrist_z) * 60

    # Release: max upward velocity
    try:
        release_frame = np.nanargmax(wrist_vz[80:160]) + 80
    except:
        release_frame = 120

    # Set-point: local max in z before release
    try:
        search_start = max(60, release_frame - 40)
        search_window = wrist_z[search_start:release_frame]
        if len(search_window) > 5:
            set_point_frame = np.nanargmax(search_window) + search_start
        else:
            set_point_frame = release_frame - 10
    except:
        set_point_frame = release_frame - 10

    # Dip: min z before set-point
    try:
        dip_window = wrist_z[40:set_point_frame]
        if len(dip_window) > 5:
            dip_frame = np.nanargmin(dip_window) + 40
        else:
            dip_frame = 80
    except:
        dip_frame = 80

    return dip_frame, set_point_frame, release_frame


def extract_spatial_features(timeseries, kp_idx):
    """Extract spatial graph features."""
    features = {}

    # Detect key frames
    dip_frame, set_point_frame, release_frame = detect_key_frames(timeseries, kp_idx)
    features['dip_frame'] = dip_frame
    features['set_point_frame'] = set_point_frame
    features['release_frame'] = release_frame

    # Define skeleton graph edges (bones)
    bones = [
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

    key_frames = {
        'dip': dip_frame,
        'set': set_point_frame,
        'release': release_frame
    }

    # 1. BONE LENGTHS at each key frame
    for frame_name, frame in key_frames.items():
        for joint1, joint2 in bones:
            p1 = get_joint_position(timeseries, kp_idx, joint1, frame)
            p2 = get_joint_position(timeseries, kp_idx, joint2, frame)
            dist = compute_distance(p1, p2)
            features[f'bone_{joint1}_{joint2}_{frame_name}'] = dist

    # 2. RELATIVE JOINT POSITIONS at key frames
    reference_joints = ['mid_hip', 'right_shoulder', 'neck']
    target_joints = ['right_wrist', 'right_elbow', 'right_ankle']

    for frame_name, frame in key_frames.items():
        for ref_joint in reference_joints:
            ref_pos = get_joint_position(timeseries, kp_idx, ref_joint, frame)
            for target_joint in target_joints:
                target_pos = get_joint_position(timeseries, kp_idx, target_joint, frame)

                # Relative position (x, y, z)
                rel = target_pos - ref_pos
                features[f'rel_{target_joint}_to_{ref_joint}_x_{frame_name}'] = rel[0]
                features[f'rel_{target_joint}_to_{ref_joint}_y_{frame_name}'] = rel[1]
                features[f'rel_{target_joint}_to_{ref_joint}_z_{frame_name}'] = rel[2]

                # Distance
                features[f'dist_{target_joint}_to_{ref_joint}_{frame_name}'] = compute_distance(target_pos, ref_pos)

    # 3. BODY ALIGNMENT ANGLES at key frames
    angle_triplets = [
        # Arm alignment
        ('right_shoulder', 'right_elbow', 'right_wrist', 'elbow'),
        ('neck', 'right_shoulder', 'right_elbow', 'shoulder'),
        # Trunk
        ('right_shoulder', 'mid_hip', 'right_knee', 'trunk_lean'),
        ('neck', 'mid_hip', 'right_hip', 'spine'),
        # Legs
        ('right_hip', 'right_knee', 'right_ankle', 'knee'),
        ('mid_hip', 'right_hip', 'right_knee', 'hip'),
    ]

    for frame_name, frame in key_frames.items():
        for j1, j2, j3, angle_name in angle_triplets:
            p1 = get_joint_position(timeseries, kp_idx, j1, frame)
            p2 = get_joint_position(timeseries, kp_idx, j2, frame)
            p3 = get_joint_position(timeseries, kp_idx, j3, frame)
            angle = compute_angle(p1, p2, p3)
            features[f'angle_{angle_name}_{frame_name}'] = angle

    # 4. POSE CHANGES between key frames
    for joint in ['right_wrist', 'right_elbow', 'right_shoulder', 'mid_hip']:
        # Dip to set-point
        p_dip = get_joint_position(timeseries, kp_idx, joint, dip_frame)
        p_set = get_joint_position(timeseries, kp_idx, joint, set_point_frame)
        p_rel = get_joint_position(timeseries, kp_idx, joint, release_frame)

        features[f'{joint}_rise_dip_to_set'] = p_set[2] - p_dip[2]
        features[f'{joint}_rise_set_to_release'] = p_rel[2] - p_set[2]
        features[f'{joint}_forward_dip_to_release'] = p_rel[1] - p_dip[1]

    # 5. SYMMETRY FEATURES (left vs right)
    for frame_name, frame in key_frames.items():
        # Shoulder symmetry
        l_shoulder = get_joint_position(timeseries, kp_idx, 'left_shoulder', frame)
        r_shoulder = get_joint_position(timeseries, kp_idx, 'right_shoulder', frame)
        features[f'shoulder_symmetry_z_{frame_name}'] = r_shoulder[2] - l_shoulder[2]
        features[f'shoulder_symmetry_y_{frame_name}'] = r_shoulder[1] - l_shoulder[1]

        # Hip symmetry
        l_hip = get_joint_position(timeseries, kp_idx, 'left_hip', frame)
        r_hip = get_joint_position(timeseries, kp_idx, 'right_hip', frame)
        features[f'hip_symmetry_z_{frame_name}'] = r_hip[2] - l_hip[2]

    # 6. WRIST POSITION RELATIVE TO HEAD (release point)
    if 'nose' in kp_idx or 'neck' in kp_idx:
        head_joint = 'nose' if 'nose' in kp_idx else 'neck'
        head_pos = get_joint_position(timeseries, kp_idx, head_joint, release_frame)
        wrist_pos = get_joint_position(timeseries, kp_idx, 'right_wrist', release_frame)

        features['wrist_above_head_release'] = wrist_pos[2] - head_pos[2]
        features['wrist_forward_of_head_release'] = wrist_pos[1] - head_pos[1]

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
    print("SPATIAL GRAPH FEATURES")
    print("=" * 80)
    print("Extracting features based on skeleton graph structure:")
    print("  - Bone lengths")
    print("  - Relative joint positions")
    print("  - Body alignment angles")
    print("  - Pose at key frames (dip, set-point, release)")

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_map(keypoint_cols)

    # Extract training features
    print("\nExtracting training features...")
    train_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_spatial_features(timeseries, kp_idx)
        features['player_id'] = metadata['participant_id']
        features['shot_id'] = metadata['id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']
        train_data.append(features)

        if i % 100 == 0:
            print(f"  Processed {i+1} shots...")

    train_df = pd.DataFrame(train_data)
    print(f"  Training samples: {len(train_df)}")

    # Get feature columns
    exclude_cols = ['player_id', 'shot_id', 'angle', 'depth', 'left_right']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"  Spatial features: {len(feature_cols)}")

    # Prepare data
    X_train = train_df[feature_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0)

    groups = train_df['player_id'].values

    # Scale targets
    targets = ['angle', 'depth', 'left_right']
    y_data = {}
    target_stats = {}
    for t in targets:
        vals = train_df[t].values
        target_stats[t] = {'min': vals.min(), 'max': vals.max()}
        y_data[t] = (vals - vals.min()) / (vals.max() - vals.min())

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Cross-validate
    print("\n" + "=" * 80)
    print("GROUPKFOLD CV WITH RIDGE REGRESSION")
    print("=" * 80)

    gkf = GroupKFold(n_splits=5)
    results = {}

    for target in targets:
        y = y_data[target]
        mses = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_scaled, y, groups)):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=10.0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            pred = np.clip(pred, 0, 1)

            mse = mean_squared_error(y_val, pred)
            mses.append(mse)

        results[target] = np.mean(mses)
        print(f"  {target}: CV MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

    print(f"\n  Total: {sum(results.values())/3:.6f}")

    # Within-player correlation analysis
    print("\n" + "=" * 80)
    print("WITHIN-PLAYER CORRELATIONS (TOP 5 PER TARGET)")
    print("=" * 80)

    for target in targets:
        print(f"\n{target.upper()}:")

        # Calculate within-player correlations
        feat_corrs = {}
        for feat in feature_cols:
            player_corrs = []
            for player in train_df['player_id'].unique():
                player_df = train_df[train_df['player_id'] == player]
                if len(player_df) < 10:
                    continue

                feat_vals = player_df[feat].values
                target_vals = player_df[target].values

                # Skip if no variance
                if np.std(feat_vals) < 1e-6 or np.std(target_vals) < 1e-6:
                    continue

                r = np.corrcoef(feat_vals, target_vals)[0, 1]
                if not np.isnan(r):
                    player_corrs.append(r)

            if len(player_corrs) >= 3:
                feat_corrs[feat] = (np.mean(player_corrs), np.std(player_corrs))

        # Sort by absolute correlation
        sorted_feats = sorted(feat_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)

        for feat, (mean_r, std_r) in sorted_feats[:5]:
            print(f"  {feat:45s}: r={mean_r:+.4f} (std={std_r:.4f})")

    # Extract test features
    print("\n" + "=" * 80)
    print("EXTRACTING TEST FEATURES")
    print("=" * 80)

    test_data = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_spatial_features(timeseries, kp_idx)
        test_data.append(features)
        test_ids.append(metadata['id'])

    test_df = pd.DataFrame(test_data)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test = test_df[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Test samples: {len(test_df)}")

    # Train final models and predict
    print("\n" + "=" * 80)
    print("CREATING SUBMISSIONS")
    print("=" * 80)

    predictions = {}
    for target in targets:
        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y_data[target])
        pred = model.predict(X_test_scaled)
        predictions[target] = np.clip(pred, 0, 1)

    # Load Sub219 for blending
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub219_indexed = sub219.set_index('id')

    sub_num = get_next_submission_number()

    # Pure spatial features submission
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    })

    # Calibrate
    submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        submission[col] = submission[col].clip(0, 1)

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Sub {sub_num}: Pure spatial features, angle_std={submission['scaled_angle'].std():.4f}")
    sub_num += 1

    # Blended submissions
    for weight in [0.1, 0.2, 0.3]:
        blended = pd.DataFrame({'id': test_ids})

        for target, col in [('angle', 'scaled_angle'), ('depth', 'scaled_depth'), ('left_right', 'scaled_left_right')]:
            spatial_pred = predictions[target]
            sub219_pred = sub219_indexed.loc[test_ids, col].values
            blended[col] = weight * spatial_pred + (1 - weight) * sub219_pred

        # Calibrate
        blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blended[col] = blended[col].clip(0, 1)

        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: {int(weight*100)}% spatial + {int((1-weight)*100)}% Sub219")
        sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Spatial graph features capture body configuration at key frames.")
    print("If within-player correlations are higher than velocity features,")
    print("these may help improve predictions.")


if __name__ == "__main__":
    main()
