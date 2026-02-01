"""
Teacher-Student Learning

Idea: Train a model to predict Sub 133's test predictions using training data.
This captures what makes Sub 133 successful while potentially improving generalization.

Steps:
1. Use CV to generate pseudo-labels on training data that match Sub 133's behavior
2. Train student model on these pseudo-labels
3. Blend student predictions with Sub 133
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def smooth_ts(ts, window=5):
    smoothed = np.zeros_like(ts)
    for col in range(ts.shape[1]):
        smoothed[:, col] = np.convolve(ts[:, col], np.ones(window)/window, mode='same')
    return smoothed


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_simple_features(timeseries, keypoint_idx):
    """Simple but effective features."""
    ts = smooth_ts(timeseries, window=5)
    features = {}

    joints = ["right_wrist", "right_elbow", "right_shoulder", "neck", "mid_hip"]
    frames = [148, 150, 153, 155, 158]

    for joint in joints:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in frames:
            if f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]
                features[f"{joint}_y_f{f}"] = pos[1]
                features[f"{joint}_z_f{f}"] = pos[2]

    return features


def main():
    print("=" * 70)
    print("TEACHER-STUDENT LEARNING")
    print("=" * 70)

    scalers_dict = load_scalers()
    keypoint_idx = get_keypoint_indices()

    # Load Sub 133 (teacher)
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    # Build features
    print("\nExtracting features...")

    train_features = []
    train_targets = []
    train_pids = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_simple_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        train_pids.append(metadata["participant_id"])

    train_df = pd.DataFrame(train_features).fillna(0)
    train_df = train_df.loc[:, train_df.nunique() > 1]

    X_train = train_df.values
    y_train = np.array(train_targets)
    pids_train = np.array(train_pids)

    test_features = []
    test_pids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_simple_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_pids.append(metadata["participant_id"])

    test_df = pd.DataFrame(test_features).fillna(0)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    X_test = test_df[train_df.columns].values
    pids_test = np.array(test_pids)

    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} samples")

    target_names = ["angle", "depth", "left_right"]

    # Step 1: Generate pseudo-labels by training on true labels and predicting on training data
    # This simulates what Sub 133 might have predicted for training samples
    print("\nStep 1: Generating pseudo-labels on training data...")

    pseudo_labels = np.zeros_like(y_train)
    cv_preds_true = np.zeros_like(y_train)

    for pid in np.unique(pids_train):
        mask = pids_train == pid
        X_pid = X_train[mask]
        y_pid = y_train[mask]
        indices = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_scaled):
            for t_idx in range(3):
                ridge = Ridge(alpha=50)
                ridge.fit(X_scaled[train_idx], y_pid[train_idx, t_idx])
                cv_preds_true[indices[test_idx], t_idx] = ridge.predict(X_scaled[test_idx])

    # Scale pseudo-labels to match Sub 133's profile
    scaled_cv = np.zeros_like(cv_preds_true)
    for i, target in enumerate(target_names):
        scaled_cv[:, i] = scalers_dict[target].transform(cv_preds_true[:, i].reshape(-1, 1)).ravel()

    print(f"CV pseudo-label profile: angle_std={scaled_cv[:, 0].std():.4f}, depth_mean={scaled_cv[:, 1].mean():.4f}")

    # Step 2: Analyze training samples most similar to test samples
    print("\nStep 2: Finding training neighbors of test samples...")

    # For each test sample, find k nearest training neighbors from same player
    # This helps understand what kinds of training samples Sub 133 predictions are based on

    k = 5  # neighbors
    neighbor_similarity = np.zeros(len(X_test))

    for pid in np.unique(pids_test):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_tr_pid = X_train[train_mask]
        X_te_pid = X_test[test_mask]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_pid)
        X_te_scaled = scaler.transform(X_te_pid)

        # KNN to find neighbors
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(X_tr_scaled, np.zeros(len(X_tr_scaled)))  # dummy target

        # Get distances to neighbors
        distances, _ = knn.kneighbors(X_te_scaled)
        mean_dist = distances.mean(axis=1)

        neighbor_similarity[test_mask] = 1 / (1 + mean_dist)

    print(f"Mean neighbor similarity: {neighbor_similarity.mean():.4f}")
    print(f"Min: {neighbor_similarity.min():.4f}, Max: {neighbor_similarity.max():.4f}")

    # Step 3: Train student model on true labels
    print("\nStep 3: Training student model...")

    student_preds = np.zeros((len(X_test), 3))
    student_cv = np.zeros_like(y_train)

    for pid in np.unique(pids_train):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]
        indices = np.where(train_mask)[0]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            ridge = Ridge(alpha=50)
            ridge.fit(X_tr_scaled, y_tr[:, t_idx])

            student_preds[test_mask, t_idx] = ridge.predict(X_te_scaled)

            # CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, te_idx in kf.split(X_tr_scaled):
                ridge_cv = Ridge(alpha=50)
                ridge_cv.fit(X_tr_scaled[tr_idx], y_tr[tr_idx, t_idx])
                student_cv[indices[te_idx], t_idx] = ridge_cv.predict(X_tr_scaled[te_idx])

    # Scale student predictions
    scaled_student = np.zeros_like(student_preds)
    scaled_student_cv = np.zeros_like(student_cv)
    scaled_true = np.zeros_like(y_train)

    for i, target in enumerate(target_names):
        scaled_student[:, i] = scalers_dict[target].transform(student_preds[:, i].reshape(-1, 1)).ravel()
        scaled_student_cv[:, i] = scalers_dict[target].transform(student_cv[:, i].reshape(-1, 1)).ravel()
        scaled_true[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

    print(f"Student CV MSE: {np.mean((scaled_student_cv - scaled_true) ** 2):.6f}")
    print(f"Student test profile: angle_std={scaled_student[:, 0].std():.4f}, depth_mean={scaled_student[:, 1].mean():.4f}")

    # Step 4: Confidence-weighted blend
    print("\nStep 4: Confidence-weighted blending...")

    # Use neighbor similarity as confidence
    # High similarity = trust student more, low similarity = trust Sub 133 more

    # Normalize similarity to [0, 0.3] range
    confidence = 0.3 * (neighbor_similarity - neighbor_similarity.min()) / \
                 (neighbor_similarity.max() - neighbor_similarity.min() + 1e-8)

    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")

    weighted_blend = np.zeros_like(scaled_student)
    for i in range(3):
        weighted_blend[:, i] = (1 - confidence) * sub133[cols[i]].values + confidence * scaled_student[:, i]

    print(f"Weighted blend profile: angle_std={weighted_blend[:, 0].std():.4f}, depth_mean={weighted_blend[:, 1].mean():.4f}")

    # Correlation with Sub 133
    corr = np.corrcoef(weighted_blend.ravel(), sub133[cols].values.ravel())[0, 1]
    print(f"Correlation with Sub 133: {corr:.4f}")

    # Step 5: Try different blend strategies
    print("\n" + "=" * 70)
    print("BLEND STRATEGIES")
    print("=" * 70)

    strategies = []

    # Strategy 1: Uniform blend
    for w in [0.05, 0.1, 0.15, 0.2]:
        uniform = (1 - w) * sub133[cols].values + w * scaled_student
        angle_std = uniform[:, 0].std()
        depth_mean = uniform[:, 1].mean()
        profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)
        strategies.append((f"Uniform {w:.2f}", uniform, profile_dist))

    # Strategy 2: Per-target optimal weight
    per_target = np.zeros_like(scaled_student)
    for i, col in enumerate(cols):
        # Find weight that minimizes profile distance for this target
        best_w = 0
        best_diff = float("inf")
        target_val = 0.1377 if i == 0 else (0.5055 if i == 1 else 0.4687)
        for w in np.linspace(0, 0.3, 31):
            blended = (1 - w) * sub133[col].values + w * scaled_student[:, i]
            diff = abs((blended.std() if i == 0 else blended.mean()) - target_val)
            if diff < best_diff:
                best_diff = diff
                best_w = w
                per_target[:, i] = blended
        print(f"  Optimal weight for {col}: {best_w:.2f}")

    angle_std = per_target[:, 0].std()
    depth_mean = per_target[:, 1].mean()
    profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)
    strategies.append(("Per-target optimal", per_target, profile_dist))

    # Strategy 3: Weighted by confidence
    strategies.append(("Confidence-weighted", weighted_blend, abs(weighted_blend[:, 0].std() - 0.1377) + abs(weighted_blend[:, 1].mean() - 0.5055)))

    # Sort by profile distance
    strategies.sort(key=lambda x: x[2])

    print("\nStrategies sorted by profile distance:")
    for name, preds, dist in strategies:
        corr = np.corrcoef(preds.ravel(), sub133[cols].values.ravel())[0, 1]
        print(f"  {name}: profile_dist={dist:.4f}, corr_with_133={corr:.4f}")

    # Save best submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    test_meta = load_metadata(train=False)
    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    # Save top 2 strategies
    for name, preds, dist in strategies[:2]:
        submission = pd.DataFrame({
            "id": test_meta["id"].values,
            "scaled_angle": preds[:, 0],
            "scaled_depth": preds[:, 1],
            "scaled_left_right": preds[:, 2]
        })
        path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(path, index=False)
        print(f"Saved {name}: {path}")
        next_num += 1


if __name__ == "__main__":
    main()
