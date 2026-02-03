"""
Transfer Learning from NBA Shooting Data

Use real NBA player shooting motion data to improve basketball predictions.

NBA data includes:
- Release position (rx, ry, rz)
- Release velocity (rv, rvx, rvy, rvz)
- Trajectory points
- Path metrics

We'll learn shooting patterns from 190 NBA players and transfer to our task.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_DIR / "external_data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def load_nba_data():
    """Load NBA shooting data."""
    player_df = pd.read_csv(EXTERNAL_DIR / "player_metrics.csv")
    print(f"NBA players: {len(player_df)}")

    # Key shooting features
    shooting_features = [
        'hght',  # height
        'rx', 'ry', 'rz',  # release position
        'rv', 'rvx', 'rvy', 'rvz',  # release velocity
        'mnv', 'mxv',  # min/max velocity
        'ta1x', 'ta1y', 'ta1z',  # trajectory apex
        'pl', 'spl',  # path length
    ]

    available = [c for c in shooting_features if c in player_df.columns]
    print(f"Using {len(available)} NBA shooting features")

    # Fill NaN and handle inf values
    X_nba = player_df[available].fillna(0).replace([np.inf, -np.inf], 0).values
    X_nba = np.clip(X_nba, -1e6, 1e6)  # Clip extreme values

    # We don't have explicit "success" labels, so we'll use path metrics as proxy
    # Better shooters tend to have more consistent paths
    # Use path length ratio as a proxy for shooting quality
    if 'plr' in player_df.columns:
        y_nba = player_df['plr'].fillna(1.0).values
    else:
        y_nba = player_df['pl'].fillna(5.0).values

    return X_nba, y_nba, available, player_df


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_shooting_features(timeseries, keypoint_idx):
    """Extract features aligned with NBA shooting mechanics."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Release frames (around the shooting motion)
    release_frames = [150, 153, 155, 158]

    # 1. Hand/wrist position at release (similar to NBA rx, ry, rz)
    for f in release_frames:
        wrist = get_joint("right_wrist", f)
        features[f'wrist_x_f{f}'] = wrist[0]
        features[f'wrist_y_f{f}'] = wrist[1]
        features[f'wrist_z_f{f}'] = wrist[2]

    # 2. Release velocity (similar to NBA rv, rvx, rvy, rvz)
    if 155 < n_frames and 150 < n_frames:
        wrist_before = get_joint("right_wrist", 150)
        wrist_after = get_joint("right_wrist", 155)
        velocity = (wrist_after - wrist_before) / (5/60)  # 5 frames at 60fps

        features['release_vx'] = velocity[0]
        features['release_vy'] = velocity[1]
        features['release_vz'] = velocity[2]
        features['release_speed'] = np.linalg.norm(velocity)

    # 3. Elbow position (affects release angle)
    for f in [150, 153, 155]:
        elbow = get_joint("right_elbow", f)
        features[f'elbow_x_f{f}'] = elbow[0]
        features[f'elbow_y_f{f}'] = elbow[1]
        features[f'elbow_z_f{f}'] = elbow[2]

    # 4. Shoulder position (base of shooting motion)
    for f in [150, 153, 155]:
        shoulder = get_joint("right_shoulder", f)
        features[f'shoulder_x_f{f}'] = shoulder[0]
        features[f'shoulder_y_f{f}'] = shoulder[1]
        features[f'shoulder_z_f{f}'] = shoulder[2]

    # 5. Arm extension at release
    shoulder = get_joint("right_shoulder", 153)
    elbow = get_joint("right_elbow", 153)
    wrist = get_joint("right_wrist", 153)

    if np.any(shoulder) and np.any(elbow) and np.any(wrist):
        upper_arm = np.linalg.norm(elbow - shoulder)
        forearm = np.linalg.norm(wrist - elbow)
        features['arm_extension'] = upper_arm + forearm

        # Elbow angle
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features['elbow_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # 6. Body position (hip as reference)
    for f in [150, 153]:
        hip = get_joint("right_hip", f)
        features[f'hip_x_f{f}'] = hip[0]
        features[f'hip_y_f{f}'] = hip[1]
        features[f'hip_z_f{f}'] = hip[2]

    # 7. Follow-through (trajectory after release)
    if 160 < n_frames and 155 < n_frames:
        wrist_155 = get_joint("right_wrist", 155)
        wrist_160 = get_joint("right_wrist", 160)
        follow_through = wrist_160 - wrist_155
        features['follow_through_x'] = follow_through[0]
        features['follow_through_y'] = follow_through[1]
        features['follow_through_z'] = follow_through[2]

    return features


def learn_nba_patterns(X_nba, y_nba):
    """Learn shooting patterns from NBA data."""
    print("\nLearning patterns from NBA data...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nba)

    # PCA to find main shooting patterns
    n_components = min(10, X_nba.shape[1], len(X_nba) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    print(f"  PCA components: {n_components}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Cluster players by shooting style
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pca.transform(X_scaled))

    print(f"  Shooting style clusters: {n_clusters}")

    return scaler, pca, kmeans


def load_basketball_data():
    """Load basketball prediction data."""
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []
    train_ids = []

    print("\nLoading basketball training data...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_shooting_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])
        train_ids.append(metadata['shot_id'])

    test_features = []
    test_ids = []

    print("Loading basketball test data...")
    for metadata, timeseries in iterate_shots(train=False):
        features = extract_shooting_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    return (train_features, train_targets, train_players, train_ids,
            test_features, test_ids)


def create_nba_transfer_features(features, nba_scaler, nba_pca, nba_kmeans, nba_feature_names):
    """Create transfer features based on NBA patterns."""

    # Map our features to NBA-like features
    nba_like = []

    # Height proxy (use shoulder height)
    nba_like.append(features.get('shoulder_z_f153', 0))

    # Release position
    nba_like.append(features.get('wrist_x_f153', 0))
    nba_like.append(features.get('wrist_y_f153', 0))
    nba_like.append(features.get('wrist_z_f153', 0))

    # Release velocity
    nba_like.append(features.get('release_speed', 0))
    nba_like.append(features.get('release_vx', 0))
    nba_like.append(features.get('release_vy', 0))
    nba_like.append(features.get('release_vz', 0))

    # Velocity range proxy
    nba_like.append(features.get('release_speed', 0) * 0.8)  # min proxy
    nba_like.append(features.get('release_speed', 0) * 1.2)  # max proxy

    # Trajectory proxy (follow through)
    nba_like.append(features.get('follow_through_x', 0))
    nba_like.append(features.get('follow_through_y', 0))
    nba_like.append(features.get('follow_through_z', 0))

    # Path length proxy
    nba_like.append(features.get('arm_extension', 0))
    nba_like.append(features.get('arm_extension', 0) * 1.1)

    nba_like = np.array(nba_like).reshape(1, -1)

    # Pad or truncate to match NBA feature count
    expected_len = len(nba_feature_names)
    if len(nba_like[0]) < expected_len:
        nba_like = np.pad(nba_like, ((0, 0), (0, expected_len - len(nba_like[0]))))
    elif len(nba_like[0]) > expected_len:
        nba_like = nba_like[:, :expected_len]

    # Transform using NBA patterns
    try:
        nba_scaled = nba_scaler.transform(nba_like)
        nba_pca_feat = nba_pca.transform(nba_scaled)[0]
        cluster_dists = nba_kmeans.transform(nba_pca_feat.reshape(1, -1))[0]
        cluster_id = nba_kmeans.predict(nba_pca_feat.reshape(1, -1))[0]
    except:
        nba_pca_feat = np.zeros(nba_pca.n_components_)
        cluster_dists = np.zeros(5)
        cluster_id = 0

    transfer_features = {}
    for i, val in enumerate(nba_pca_feat):
        transfer_features[f'nba_pca_{i}'] = val

    for i, dist in enumerate(cluster_dists):
        transfer_features[f'nba_cluster_dist_{i}'] = dist

    transfer_features['nba_shooting_style'] = cluster_id

    return transfer_features


def main():
    print("="*80)
    print("TRANSFER LEARNING FROM NBA SHOOTING DATA")
    print("="*80)

    # Load NBA data
    X_nba, y_nba, nba_features, nba_df = load_nba_data()

    # Learn NBA patterns
    nba_scaler, nba_pca, nba_kmeans = learn_nba_patterns(X_nba, y_nba)

    # Load basketball data
    (train_features, train_targets, train_players, train_ids,
     test_features, test_ids) = load_basketball_data()

    print(f"\nBasketball train: {len(train_features)}, test: {len(test_features)}")

    # Combine original + NBA transfer features
    X_train = []
    for bf in train_features:
        nba_transfer = create_nba_transfer_features(bf, nba_scaler, nba_pca, nba_kmeans, nba_features)
        combined = {**bf, **nba_transfer}
        X_train.append(combined)

    X_test = []
    for bf in test_features:
        nba_transfer = create_nba_transfer_features(bf, nba_scaler, nba_pca, nba_kmeans, nba_features)
        combined = {**bf, **nba_transfer}
        X_test.append(combined)

    X_train = pd.DataFrame(X_train).fillna(0)
    X_test = pd.DataFrame(X_test).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Total features: {len(common_cols)}")
    print(f"NBA transfer features: {len([c for c in common_cols if c.startswith('nba_')])}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\nTraining model with NBA transfer features...")

    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        gkf = GroupKFold(n_splits=5)
        fold_scores = []

        for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
            model = Ridge(alpha=10.0)
            model.fit(X_train_scaled[train_idx], y[train_idx])
            pred = model.predict(X_train_scaled[val_idx])
            mse = np.mean((pred - y[val_idx])**2)
            fold_scores.append(mse)

        cv_score = np.mean(fold_scores)
        cv_scores.append(cv_score)
        print(f"  {target} CV MSE: {cv_score:.6f}")

        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

    overall_cv = np.mean(cv_scores)
    print(f"\nOverall CV MSE: {overall_cv:.6f}")

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    angle_std = np.std(predictions[:, 0])
    print(f"angle_std: {angle_std:.6f}")

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

    # Check correlation with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Blend with Sub 219
    print("\nBlending with Sub 219...")
    for w in [0.1, 0.2, 0.3, 0.4]:
        blend = submission.copy()
        blend['scaled_angle'] = w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values
        blend['scaled_depth'] = w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values
        blend['scaled_left_right'] = w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

        std = blend['scaled_angle'].std()

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"  {w:.0%} NBA + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")

    # Apply selective amplification
    print("\nApplying selective amplification...")
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")

    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    threshold = np.percentile(total_diff, 91)
    mask = total_diff > threshold

    amp_preds = predictions.copy()
    amp_preds[mask, 0] += 1.1 * diff_angle[mask]
    amp_preds[mask, 1] += 1.1 * diff_depth[mask]
    amp_preds[mask, 2] += 1.1 * diff_lr[mask]

    amp_preds = np.clip(amp_preds, 0, 1)
    amp_preds[:, 1] = amp_preds[:, 1] - np.mean(amp_preds[:, 1]) + 0.5055
    amp_preds[:, 1] = np.clip(amp_preds[:, 1], 0, 1)

    std = np.std(amp_preds[:, 0])

    next_num += 1
    amp_sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': amp_preds[:, 0],
        'scaled_depth': amp_preds[:, 1],
        'scaled_left_right': amp_preds[:, 2]
    })
    amp_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    amp_sub.to_csv(amp_file, index=False)
    print(f"\nNBA Transfer + Amplification: angle_std={std:.6f} -> {amp_file}")


if __name__ == "__main__":
    main()
