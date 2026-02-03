"""
Skeleton Graph Features

Treat the skeleton as a graph and extract graph-based features.
This captures the structural relationships between joints that
simple position features miss.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)

# Define skeleton graph structure (edges between connected joints)
SKELETON_EDGES = [
    # Spine
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),

    # Right arm (shooting arm)
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),

    # Left arm
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),

    # Right leg
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),

    # Left leg
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),

    # Hands (if available)
    ('right_wrist', 'right_thumb'),
    ('right_wrist', 'right_index'),
    ('right_wrist', 'right_pinky'),
    ('left_wrist', 'left_thumb'),
    ('left_wrist', 'left_index'),
    ('left_wrist', 'left_pinky'),
]


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def extract_graph_features(timeseries, keypoint_idx, frame):
    """Extract graph-based features from skeleton at a specific frame."""
    features = {}
    n_frames = len(timeseries)

    if frame >= n_frames:
        return features

    def get_joint(name):
        if name not in keypoint_idx:
            return None
        idx = keypoint_idx[name]
        pos = timeseries[frame, idx*3:(idx+1)*3]
        if np.all(pos == 0):
            return None
        return pos

    # 1. Edge lengths (bone lengths)
    edge_lengths = []
    for j1, j2 in SKELETON_EDGES:
        p1 = get_joint(j1)
        p2 = get_joint(j2)
        if p1 is not None and p2 is not None:
            length = np.linalg.norm(p2 - p1)
            features[f'edge_{j1}_{j2}_f{frame}'] = length
            edge_lengths.append(length)

    if edge_lengths:
        features[f'mean_edge_length_f{frame}'] = np.mean(edge_lengths)
        features[f'std_edge_length_f{frame}'] = np.std(edge_lengths)
        features[f'max_edge_length_f{frame}'] = np.max(edge_lengths)

    # 2. Joint angles at key vertices
    # Right arm chain
    shoulder = get_joint('right_shoulder')
    elbow = get_joint('right_elbow')
    wrist = get_joint('right_wrist')

    if shoulder is not None and elbow is not None and wrist is not None:
        # Elbow angle
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features[f'elbow_angle_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

    # Shoulder angle
    hip = get_joint('right_hip')
    if hip is not None and shoulder is not None and elbow is not None:
        v1 = hip - shoulder
        v2 = elbow - shoulder
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features[f'shoulder_angle_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

    # 3. Cross-body symmetry
    left_shoulder = get_joint('left_shoulder')
    right_shoulder = get_joint('right_shoulder')
    left_hip = get_joint('left_hip')
    right_hip = get_joint('right_hip')

    if all(p is not None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
        # Shoulder line vs hip line angle
        shoulder_vec = right_shoulder - left_shoulder
        hip_vec = right_hip - left_hip
        cos_ang = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8)
        features[f'body_twist_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

        # Shoulder tilt
        features[f'shoulder_tilt_z_f{frame}'] = right_shoulder[2] - left_shoulder[2]
        features[f'hip_tilt_z_f{frame}'] = right_hip[2] - left_hip[2]

    # 4. Arm-body relationship
    if shoulder is not None and elbow is not None and wrist is not None:
        # Arm plane normal
        v1 = elbow - shoulder
        v2 = wrist - elbow
        arm_normal = np.cross(v1, v2)
        if np.linalg.norm(arm_normal) > 1e-8:
            arm_normal = arm_normal / np.linalg.norm(arm_normal)
            features[f'arm_plane_nx_f{frame}'] = arm_normal[0]
            features[f'arm_plane_ny_f{frame}'] = arm_normal[1]
            features[f'arm_plane_nz_f{frame}'] = arm_normal[2]

    # 5. Shooting arm extension ratio
    if shoulder is not None and wrist is not None and elbow is not None:
        direct_dist = np.linalg.norm(wrist - shoulder)
        arm_length = np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow)
        if arm_length > 0:
            features[f'arm_extension_ratio_f{frame}'] = direct_dist / arm_length

    # 6. Hand position relative to body center
    if left_hip is not None and right_hip is not None and left_shoulder is not None and right_shoulder is not None:
        body_center = (left_hip + right_hip + left_shoulder + right_shoulder) / 4
        if wrist is not None:
            rel_wrist = wrist - body_center
            features[f'wrist_rel_x_f{frame}'] = rel_wrist[0]
            features[f'wrist_rel_y_f{frame}'] = rel_wrist[1]
            features[f'wrist_rel_z_f{frame}'] = rel_wrist[2]

    # 7. Graph centrality - distance from wrist to body center
    if wrist is not None and left_hip is not None and right_hip is not None:
        center_hip = (left_hip + right_hip) / 2
        features[f'wrist_to_center_f{frame}'] = np.linalg.norm(wrist - center_hip)

    return features


def extract_temporal_graph_features(timeseries, keypoint_idx):
    """Extract graph features across multiple frames."""
    features = {}

    # Key frames
    frames = [80, 100, 120, 140, 153, 160, 180]

    for frame in frames:
        frame_features = extract_graph_features(timeseries, keypoint_idx, frame)
        features.update(frame_features)

    # Temporal changes in key graph properties
    for frame_pair in [(100, 153), (120, 160), (140, 170)]:
        f1, f2 = frame_pair
        feat1 = extract_graph_features(timeseries, keypoint_idx, f1)
        feat2 = extract_graph_features(timeseries, keypoint_idx, f2)

        # Change in elbow angle
        if f'elbow_angle_f{f1}' in feat1 and f'elbow_angle_f{f2}' in feat2:
            features[f'elbow_angle_change_{f1}_{f2}'] = feat2[f'elbow_angle_f{f2}'] - feat1[f'elbow_angle_f{f1}']

        # Change in arm extension
        if f'arm_extension_ratio_f{f1}' in feat1 and f'arm_extension_ratio_f{f2}' in feat2:
            features[f'arm_extension_change_{f1}_{f2}'] = feat2[f'arm_extension_ratio_f{f2}'] - feat1[f'arm_extension_ratio_f{f1}']

    return features


def main():
    print("="*80)
    print("SKELETON GRAPH FEATURES")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    print(f"\nKeypoints: {len(keypoint_idx)}")
    print(f"Skeleton edges: {len(SKELETON_EDGES)}")

    # Extract features
    print("\nExtracting graph features...")
    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_temporal_graph_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])

    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_temporal_graph_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Graph features: {len(common_cols)}")
    print(f"Training samples: {len(X_train)}")

    # Feature correlations
    print("\nTop feature correlations with angle:")
    corrs = []
    for col in common_cols:
        if X_train[col].std() > 0.001:
            corr = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
            if not np.isnan(corr):
                corrs.append((col, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs[:20]:
        print(f"  {col}: {corr:.4f}")

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining models...")
    predictions = np.zeros((len(X_test), 3))

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        best_alpha = 100
        best_score = float('inf')

        for alpha in [10, 50, 100, 200, 500]:
            gkf = GroupKFold(n_splits=5)
            scores = []
            for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled[train_idx], y[train_idx])
                pred = model.predict(X_train_scaled[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                scores.append(mse)
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        print(f"  {target}: best_alpha={best_alpha}, CV MSE={best_score:.4f}")

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    print(f"\nangle_std: {np.std(predictions[:, 0]):.6f}")

    # Compare with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions[:, 0],
        'scaled_depth': predictions[:, 1],
        'scaled_left_right': predictions[:, 2]
    })
    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Blend with Sub 219
    for w in [0.15, 0.25, 0.35]:
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"Saved: {blend_file} ({w:.0%} graph + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")


if __name__ == "__main__":
    main()
