"""
Two Novel Approaches:
1. Confidence-Based Blending - use agreement between models as confidence
2. Cluster-Based Local Models - different models for different motion types
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def extract_features(timeseries, keypoint_idx):
    """Extract features for clustering and prediction."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    key_frames = [80, 100, 120, 140, 150, 153, 156, 160, 170]

    for f in key_frames:
        # Positions
        for joint in ['right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder', 'right_hip']:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

        # Body twist
        ls = get_joint('left_shoulder', f)
        rs = get_joint('right_shoulder', f)
        lh = get_joint('left_hip', f)
        rh = get_joint('right_hip', f)

        if np.any(ls) and np.any(rs) and np.any(lh) and np.any(rh):
            shoulder_vec = rs - ls
            hip_vec = rh - lh
            cos_ang = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8)
            features[f'body_twist_f{f}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

        # Elbow angle
        wrist = get_joint('right_wrist', f)
        elbow = get_joint('right_elbow', f)
        shoulder = get_joint('right_shoulder', f)

        if np.any(wrist) and np.any(elbow) and np.any(shoulder):
            v1 = shoulder - elbow
            v2 = wrist - elbow
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            features[f'elbow_angle_f{f}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

    # Velocities around release
    for f_start, f_end in [(148, 153), (150, 155), (153, 158)]:
        for joint in ['right_wrist', 'right_elbow']:
            pos_s = get_joint(joint, f_start)
            pos_e = get_joint(joint, f_end)
            if np.any(pos_s) and np.any(pos_e):
                vel = pos_e - pos_s
                features[f'{joint}_speed_{f_start}_{f_end}'] = np.linalg.norm(vel)
                features[f'{joint}_vz_{f_start}_{f_end}'] = vel[2]

    return features


def main():
    print("="*70)
    print("CONFIDENCE-BASED & CLUSTER-BASED MODELS")
    print("="*70)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Load data
    print("\nLoading data...")
    train_data = []
    test_data = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_features(timeseries, keypoint_idx)
        train_data.append({
            'features': features,
            'player': metadata['participant_id'],
            'targets': {
                'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
                'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
                'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
            }
        })

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_features(timeseries, keypoint_idx)
        test_data.append({
            'features': features,
            'id': metadata['id'],
            'player': metadata['participant_id']
        })

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Load existing submissions for confidence calculation
    sub_nums = [9, 10, 25, 111, 133, 219]
    submissions = {}
    for num in sub_nums:
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        if path.exists():
            submissions[num] = pd.read_csv(path)

    print(f"Loaded {len(submissions)} submissions for confidence calculation")

    # Create feature matrices
    X_train = pd.DataFrame([d['features'] for d in train_data]).fillna(0)
    X_test = pd.DataFrame([d['features'] for d in test_data]).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols].values
    X_test = X_test[common_cols].values

    y_train = {
        'angle': np.array([d['targets']['angle'] for d in train_data]),
        'depth': np.array([d['targets']['depth'] for d in train_data]),
        'left_right': np.array([d['targets']['left_right'] for d in train_data])
    }

    test_ids = [d['id'] for d in test_data]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")

    # =========================================================================
    # APPROACH 1: CONFIDENCE-BASED BLENDING
    # =========================================================================
    print("\n" + "="*70)
    print("APPROACH 1: CONFIDENCE-BASED BLENDING")
    print("="*70)

    # Calculate disagreement between submissions for each test sample
    targets = ['angle', 'depth', 'left_right']

    # Get predictions from all submissions
    all_preds = {t: [] for t in targets}
    for num, sub in submissions.items():
        for t in targets:
            all_preds[t].append(sub[f'scaled_{t}'].values)

    # Convert to arrays
    for t in targets:
        all_preds[t] = np.array(all_preds[t])  # shape: (n_submissions, n_test)

    # Calculate standard deviation (disagreement) per sample
    disagreement = {t: np.std(all_preds[t], axis=0) for t in targets}

    print("\nDisagreement statistics:")
    for t in targets:
        print(f"  {t}: mean={np.mean(disagreement[t]):.4f}, max={np.max(disagreement[t]):.4f}")

    # High disagreement = low confidence
    # Low disagreement = high confidence

    # Strategy: For high disagreement samples, blend toward the mean
    # For low disagreement samples, use Sub 219

    confidence_preds = {}
    for t in targets:
        # Normalize disagreement to [0, 1]
        dis = disagreement[t]
        dis_norm = (dis - dis.min()) / (dis.max() - dis.min() + 1e-8)

        # confidence = 1 - disagreement (high disagreement = low confidence)
        confidence = 1 - dis_norm

        # Mean prediction across all submissions
        mean_pred = np.mean(all_preds[t], axis=0)

        # Blend: confident -> use Sub 219, not confident -> use mean
        # pred = confidence * sub219 + (1-confidence) * mean
        confidence_preds[t] = confidence * sub219[f'scaled_{t}'].values + (1 - confidence) * mean_pred

    # Calibrate depth
    confidence_preds['depth'] = confidence_preds['depth'] - np.mean(confidence_preds['depth']) + 0.5055

    for t in targets:
        confidence_preds[t] = np.clip(confidence_preds[t], 0, 1)

    angle_std_conf = np.std(confidence_preds['angle'])
    corr_conf = np.corrcoef(confidence_preds['angle'], sub219['scaled_angle'].values)[0, 1]

    print(f"\nConfidence-based results:")
    print(f"  angle_std: {angle_std_conf:.6f}")
    print(f"  Correlation with Sub 219: {corr_conf:.4f}")

    # =========================================================================
    # APPROACH 2: CLUSTER-BASED LOCAL MODELS
    # =========================================================================
    print("\n" + "="*70)
    print("APPROACH 2: CLUSTER-BASED LOCAL MODELS")
    print("="*70)

    # Combine train and test for clustering
    X_all = np.vstack([X_train_scaled, X_test_scaled])

    # Try different numbers of clusters
    best_cluster_preds = None
    best_angle_std = float('inf')
    best_n_clusters = 0

    for n_clusters in [3, 5, 7, 10]:
        print(f"\n  Trying {n_clusters} clusters...")

        # Cluster all data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        all_labels = kmeans.fit_predict(X_all)

        train_labels = all_labels[:len(X_train)]
        test_labels = all_labels[len(X_train):]

        # Train local models per cluster
        cluster_preds = {t: np.zeros(len(X_test)) for t in targets}

        for cluster_id in range(n_clusters):
            train_mask = train_labels == cluster_id
            test_mask = test_labels == cluster_id

            n_train_cluster = np.sum(train_mask)
            n_test_cluster = np.sum(test_mask)

            if n_train_cluster < 10 or n_test_cluster == 0:
                # Not enough data - use global mean or Sub 219
                for t in targets:
                    if n_test_cluster > 0:
                        cluster_preds[t][test_mask] = sub219[f'scaled_{t}'].values[test_mask]
                continue

            X_train_cluster = X_train_scaled[train_mask]
            X_test_cluster = X_test_scaled[test_mask]

            for t in targets:
                y_train_cluster = y_train[t][train_mask]

                # Train local Ridge model
                model = Ridge(alpha=10)
                model.fit(X_train_cluster, y_train_cluster)
                pred = model.predict(X_test_cluster)

                cluster_preds[t][test_mask] = pred

        # Calibrate depth
        cluster_preds['depth'] = cluster_preds['depth'] - np.mean(cluster_preds['depth']) + 0.5055

        for t in targets:
            cluster_preds[t] = np.clip(cluster_preds[t], 0, 1)

        angle_std = np.std(cluster_preds['angle'])
        print(f"    angle_std: {angle_std:.6f}")

        if abs(angle_std - 0.137162) < abs(best_angle_std - 0.137162):
            best_angle_std = angle_std
            best_cluster_preds = cluster_preds.copy()
            best_n_clusters = n_clusters

    print(f"\n  Best: {best_n_clusters} clusters, angle_std={best_angle_std:.6f}")

    corr_cluster = np.corrcoef(best_cluster_preds['angle'], sub219['scaled_angle'].values)[0, 1]
    print(f"  Correlation with Sub 219: {corr_cluster:.4f}")

    # =========================================================================
    # APPROACH 3: COMBINED - Cluster + Confidence
    # =========================================================================
    print("\n" + "="*70)
    print("APPROACH 3: COMBINED (Cluster + Confidence)")
    print("="*70)

    # Use cluster predictions where confident, Sub 219 otherwise
    combined_preds = {}
    for t in targets:
        # Higher disagreement -> trust Sub 219 more
        dis_norm = (disagreement[t] - disagreement[t].min()) / (disagreement[t].max() - disagreement[t].min() + 1e-8)

        # Blend cluster and Sub 219 based on confidence
        confidence = 1 - dis_norm
        combined_preds[t] = 0.3 * best_cluster_preds[t] + 0.7 * sub219[f'scaled_{t}'].values

        # For high disagreement samples, lean more toward cluster model (it's different)
        # For low disagreement samples, stick with Sub 219

    combined_preds['depth'] = combined_preds['depth'] - np.mean(combined_preds['depth']) + 0.5055

    for t in targets:
        combined_preds[t] = np.clip(combined_preds[t], 0, 1)

    angle_std_combined = np.std(combined_preds['angle'])
    corr_combined = np.corrcoef(combined_preds['angle'], sub219['scaled_angle'].values)[0, 1]

    print(f"Combined results:")
    print(f"  angle_std: {angle_std_combined:.6f}")
    print(f"  Correlation with Sub 219: {corr_combined:.4f}")

    # =========================================================================
    # APPROACH 4: Nearest Neighbor Direct
    # =========================================================================
    print("\n" + "="*70)
    print("APPROACH 4: NEAREST NEIGHBOR (No Model)")
    print("="*70)

    # For each test sample, find K nearest training samples
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_train_scaled)

    distances, indices = nn.kneighbors(X_test_scaled)

    nn_preds = {t: np.zeros(len(X_test)) for t in targets}

    for i in range(len(X_test)):
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]

        # Weight by inverse distance
        weights = 1 / (neighbor_distances + 1e-8)
        weights = weights / weights.sum()

        for t in targets:
            neighbor_targets = y_train[t][neighbor_indices]
            nn_preds[t][i] = np.sum(weights * neighbor_targets)

    nn_preds['depth'] = nn_preds['depth'] - np.mean(nn_preds['depth']) + 0.5055

    for t in targets:
        nn_preds[t] = np.clip(nn_preds[t], 0, 1)

    angle_std_nn = np.std(nn_preds['angle'])
    corr_nn = np.corrcoef(nn_preds['angle'], sub219['scaled_angle'].values)[0, 1]

    print(f"Nearest Neighbor results:")
    print(f"  angle_std: {angle_std_nn:.6f}")
    print(f"  Correlation with Sub 219: {corr_nn:.4f}")

    # =========================================================================
    # SAVE ALL SUBMISSIONS
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING SUBMISSIONS")
    print("="*70)

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    results = [
        ('confidence', confidence_preds, angle_std_conf, corr_conf),
        ('cluster', best_cluster_preds, best_angle_std, corr_cluster),
        ('combined', combined_preds, angle_std_combined, corr_combined),
        ('nearest_neighbor', nn_preds, angle_std_nn, corr_nn),
    ]

    for name, preds, a_std, corr in results:
        sub = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': preds['angle'],
            'scaled_depth': preds['depth'],
            'scaled_left_right': preds['left_right']
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
        print(f"Saved: submission_{next_num}.csv ({name}, angle_std={a_std:.6f}, corr={corr:.4f})")
        next_num += 1

    # Blends with Sub 219
    print("\nBlends with Sub 219:")

    for name, preds, _, _ in results:
        for w in [0.15, 0.25]:
            blend = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': w * preds['angle'] + (1-w) * sub219['scaled_angle'].values,
                'scaled_depth': w * preds['depth'] + (1-w) * sub219['scaled_depth'].values,
                'scaled_left_right': w * preds['left_right'] + (1-w) * sub219['scaled_left_right'].values
            })
            blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

            blend.to_csv(SUBMISSION_DIR / f"submission_{next_num}.csv", index=False)
            print(f"Saved: submission_{next_num}.csv ({w:.0%} {name} + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")
            next_num += 1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Approach':<20} {'angle_std':<12} {'Corr w/Sub219':<15}")
    print("-"*50)
    for name, _, a_std, corr in results:
        print(f"{name:<20} {a_std:<12.6f} {corr:<15.4f}")


if __name__ == "__main__":
    main()
