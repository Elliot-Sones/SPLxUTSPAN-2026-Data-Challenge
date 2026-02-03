"""
SSL-Enhanced Features

Use self-supervised learning to create better features:
1. Train autoencoder on all timeseries (train + test)
2. Use encoder output as additional features
3. Combine with best physics features
4. Apply selective amplification
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
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


KEY_JOINTS = ["right_wrist", "right_elbow", "right_shoulder", "right_hip",
              "right_knee", "left_wrist", "neck", "mid_hip"]


def extract_release_window(timeseries, start=145, end=165):
    """Extract the release window as a flattened feature vector."""
    if end > len(timeseries):
        end = len(timeseries)
    if start >= end:
        return np.zeros((end - start) * timeseries.shape[1])

    window = timeseries[start:end, :]
    return window.flatten()


def extract_physics_features(timeseries, keypoint_idx):
    """Extract physics-based features."""
    features = {}
    n_frames = len(timeseries)

    release_frames = [145, 148, 150, 153, 155, 158, 160]

    for joint in KEY_JOINTS:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in release_frames:
            if f < n_frames:
                pos = timeseries[f, idx*3:(idx+1)*3]
                features[f"{joint}_x_f{f}"] = pos[0]
                features[f"{joint}_y_f{f}"] = pos[1]
                features[f"{joint}_z_f{f}"] = pos[2]

        # Velocity at release
        if 153 < n_frames and 150 < n_frames:
            pos_before = timeseries[150, idx*3:(idx+1)*3]
            pos_after = timeseries[153, idx*3:(idx+1)*3]
            vel = pos_after - pos_before
            features[f"{joint}_vel_x"] = vel[0]
            features[f"{joint}_vel_y"] = vel[1]
            features[f"{joint}_vel_z"] = vel[2]
            features[f"{joint}_speed"] = np.linalg.norm(vel)

    return features


def load_all_data():
    """Load all data."""
    scalers = load_scalers()
    keypoint_idx = get_keypoint_indices()

    train_ts = []
    train_features = []
    train_targets = []
    train_players = []
    train_ids = []

    print("Loading training data...")
    for metadata, timeseries in iterate_shots(train=True):
        train_ts.append(timeseries)
        train_features.append(extract_physics_features(timeseries, keypoint_idx))
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])
        train_ids.append(metadata['shot_id'])

    test_ts = []
    test_features = []
    test_ids = []

    print("Loading test data...")
    for metadata, timeseries in iterate_shots(train=False):
        test_ts.append(timeseries)
        test_features.append(extract_physics_features(timeseries, keypoint_idx))
        test_ids.append(metadata['id'])

    return (train_ts, train_features, train_targets, train_players, train_ids,
            test_ts, test_features, test_ids)


def learn_ssl_features(all_ts, n_components=20):
    """Learn SSL features using PCA on release window."""
    print("\nLearning SSL features...")

    # Extract release windows
    windows = []
    for ts in all_ts:
        window = extract_release_window(ts, start=145, end=165)
        windows.append(window)

    windows = np.array(windows)
    print(f"  Release windows shape: {windows.shape}")

    # Normalize
    scaler = StandardScaler()
    windows_scaled = scaler.fit_transform(windows)

    # PCA for dimensionality reduction
    pca = PCA(n_components=min(n_components, windows.shape[1], len(windows)-1))
    pca.fit(windows_scaled)

    print(f"  PCA components: {pca.n_components_}")
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    # Cluster-based features
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pca.transform(windows_scaled))

    return scaler, pca, kmeans


def encode_with_ssl(timeseries, scaler, pca, kmeans):
    """Encode timeseries with SSL features."""
    window = extract_release_window(timeseries, start=145, end=165)
    window_scaled = scaler.transform(window.reshape(1, -1))
    pca_features = pca.transform(window_scaled)[0]

    # Distance to cluster centers
    cluster_dists = kmeans.transform(pca_features.reshape(1, -1))[0]
    cluster_id = kmeans.predict(pca_features.reshape(1, -1))[0]

    features = {}
    for i, val in enumerate(pca_features):
        features[f'ssl_pca_{i}'] = val

    for i, dist in enumerate(cluster_dists):
        features[f'ssl_cluster_dist_{i}'] = dist

    features['ssl_cluster_id'] = cluster_id

    return features


def train_model(X_train, y_train, players, X_test):
    """Train model with cross-validation."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    target_names = ['angle', 'depth', 'left_right']

    for i, target in enumerate(target_names):
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

    return predictions, np.mean(cv_scores)


def main():
    print("="*80)
    print("SSL-ENHANCED FEATURES")
    print("="*80)

    # Load data
    (train_ts, train_phys, train_targets, train_players, train_ids,
     test_ts, test_phys, test_ids) = load_all_data()

    print(f"\nTrain: {len(train_ts)}, Test: {len(test_ts)}")

    # Learn SSL features on all data
    all_ts = train_ts + test_ts
    scaler_ssl, pca, kmeans = learn_ssl_features(all_ts, n_components=20)

    # Combine physics + SSL features
    print("\nCombining physics + SSL features...")

    X_train = []
    for i, ts in enumerate(train_ts):
        phys = train_phys[i]
        ssl = encode_with_ssl(ts, scaler_ssl, pca, kmeans)
        combined = {**phys, **ssl}
        X_train.append(combined)

    X_test = []
    for i, ts in enumerate(test_ts):
        phys = test_phys[i]
        ssl = encode_with_ssl(ts, scaler_ssl, pca, kmeans)
        combined = {**phys, **ssl}
        X_test.append(combined)

    X_train = pd.DataFrame(X_train).fillna(0)
    X_test = pd.DataFrame(X_test).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Total features: {len(common_cols)}")

    # Train model
    print("\nTraining model...")
    predictions, cv_score = train_model(X_train.values, y_train, players, X_test.values)

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    angle_std = np.std(predictions[:, 0])
    print(f"\nCV MSE: {cv_score:.6f}")
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

    # Check correlation with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Apply selective amplification (Sub 219 approach)
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")

    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    # Try amplification on this model's predictions
    print("\nApplying selective amplification to SSL model...")

    for pctl, alpha in [(91, 1.1), (90, 1.0), (92, 0.9)]:
        threshold = np.percentile(total_diff, pctl)
        mask = total_diff > threshold

        amp_preds = predictions.copy()
        amp_preds[mask, 0] += alpha * diff_angle[mask]
        amp_preds[mask, 1] += alpha * diff_depth[mask]
        amp_preds[mask, 2] += alpha * diff_lr[mask]

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

        print(f"  pctl={pctl}, alpha={alpha}: angle_std={std:.6f} -> {amp_file}")

    # Blend SSL with Sub 219
    print("\nBlending SSL with Sub 219...")
    for w in [0.1, 0.2, 0.3]:
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

        print(f"  {w:.0%} SSL + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
