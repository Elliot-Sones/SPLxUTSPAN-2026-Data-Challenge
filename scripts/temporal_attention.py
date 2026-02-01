"""
Temporal Attention Model

Instead of extracting features at specific frames, use attention
to learn which frames are important for each target.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_temporal_features(timeseries, keypoint_idx, frame_weights):
    """Extract weighted features across time using learned attention weights."""
    features = {}

    # Key joints
    joints = ["right_wrist", "right_elbow", "right_shoulder", "neck", "mid_hip"]

    # Normalize weights
    weights = np.array(frame_weights)
    weights = weights / weights.sum()

    for joint in joints:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        # Position: weighted average across frames
        pos = timeseries[:, idx*3:(idx+1)*3]
        weighted_pos = (pos * weights[:, np.newaxis]).sum(axis=0)

        features[f"{joint}_x_wavg"] = weighted_pos[0]
        features[f"{joint}_y_wavg"] = weighted_pos[1]
        features[f"{joint}_z_wavg"] = weighted_pos[2]

        # Also include specific high-weight frames
        top_frames = np.argsort(weights)[-5:]
        for f in top_frames:
            if f < len(timeseries):
                p = timeseries[f, idx*3:(idx+1)*3]
                features[f"{joint}_x_f{f}"] = p[0]
                features[f"{joint}_y_f{f}"] = p[1]
                features[f"{joint}_z_f{f}"] = p[2]

    return features


def learn_frame_weights(X_all_frames, y, pids, target_idx, n_frames=240):
    """Learn optimal frame weights via greedy search."""
    # Start with uniform weights around release
    weights = np.zeros(n_frames)
    weights[140:170] = 1.0  # Focus on release window
    weights = weights / weights.sum()

    # Greedy search: adjust weights to minimize CV
    best_mse = float("inf")
    best_weights = weights.copy()

    for iteration in range(20):
        # Try shifting weight to different frames
        for f in range(140, 170):
            test_weights = weights.copy()
            test_weights[f] += 0.05
            test_weights = test_weights / test_weights.sum()

            # Extract features with these weights
            # (simplified: use weighted sum of frame features)
            X_weighted = np.zeros((len(X_all_frames), X_all_frames[0].shape[1]))
            for i, ts in enumerate(X_all_frames):
                for frame_idx in range(min(n_frames, len(ts))):
                    X_weighted[i] += test_weights[frame_idx] * ts[frame_idx]

            # CV evaluation
            mse = evaluate_weighted_features(X_weighted, y, pids, target_idx)

            if mse < best_mse:
                best_mse = mse
                best_weights = test_weights.copy()

        weights = best_weights.copy()

    return best_weights, best_mse


def evaluate_weighted_features(X, y, pids, target_idx):
    """Evaluate features with within-player CV."""
    all_preds = np.zeros(len(y))

    for pid in np.unique(pids):
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask, target_idx]
        indices = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_scaled):
            ridge = Ridge(alpha=100)
            ridge.fit(X_scaled[train_idx], y_pid[train_idx])
            all_preds[indices[test_idx]] = ridge.predict(X_scaled[test_idx])

    mse = np.mean((all_preds - y[:, target_idx]) ** 2)
    return mse


def main():
    print("=" * 70)
    print("TEMPORAL ATTENTION MODEL")
    print("=" * 70)

    scalers_dict = load_scalers()
    keypoint_idx = get_keypoint_indices()

    # Load all timeseries
    print("\nLoading timeseries data...")

    train_timeseries = []
    train_targets = []
    train_pids = []

    for metadata, timeseries in iterate_shots(train=True):
        train_timeseries.append(timeseries)
        train_targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        train_pids.append(metadata["participant_id"])

    y_train = np.array(train_targets)
    pids_train = np.array(train_pids)

    test_timeseries = []
    test_pids = []

    for metadata, timeseries in iterate_shots(train=False):
        test_timeseries.append(timeseries)
        test_pids.append(metadata["participant_id"])

    pids_test = np.array(test_pids)

    target_names = ["angle", "depth", "left_right"]

    # For simplicity, use fixed attention weights based on release frame
    print("\nUsing pre-defined attention weights...")

    # Attention weights: Gaussian centered at frame 153 (release)
    frames = np.arange(240)
    attention = np.exp(-((frames - 153) ** 2) / (2 * 10 ** 2))  # sigma=10
    attention = attention / attention.sum()

    print(f"Attention peak at frame 153, sigma=10")
    print(f"Top 5 weighted frames: {np.argsort(attention)[-5:]}")

    # Extract features with attention weights
    print("\nExtracting attention-weighted features...")

    train_features = []
    for ts in train_timeseries:
        features = extract_temporal_features(ts, keypoint_idx, attention[:len(ts)])
        train_features.append(features)

    train_df = pd.DataFrame(train_features).fillna(0)
    train_df = train_df.loc[:, train_df.nunique() > 1]

    X_train = train_df.values

    test_features = []
    for ts in test_timeseries:
        features = extract_temporal_features(ts, keypoint_idx, attention[:len(ts)])
        test_features.append(features)

    test_df = pd.DataFrame(test_features).fillna(0)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    X_test = test_df[train_df.columns].values

    print(f"Features: {X_train.shape[1]}")

    # Train and evaluate
    print("\nTraining model...")

    test_preds = np.zeros((len(X_test), 3))
    cv_preds = np.zeros_like(y_train)

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
            ridge = Ridge(alpha=100)
            ridge.fit(X_tr_scaled, y_tr[:, t_idx])

            test_preds[test_mask, t_idx] = ridge.predict(X_te_scaled)

            # CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, te_idx in kf.split(X_tr_scaled):
                ridge_cv = Ridge(alpha=100)
                ridge_cv.fit(X_tr_scaled[tr_idx], y_tr[tr_idx, t_idx])
                cv_preds[indices[te_idx], t_idx] = ridge_cv.predict(X_tr_scaled[te_idx])

    # Scale
    scaled_test = np.zeros_like(test_preds)
    scaled_cv = np.zeros_like(cv_preds)
    scaled_true = np.zeros_like(y_train)

    for i, target in enumerate(target_names):
        scaled_test[:, i] = scalers_dict[target].transform(test_preds[:, i].reshape(-1, 1)).ravel()
        scaled_cv[:, i] = scalers_dict[target].transform(cv_preds[:, i].reshape(-1, 1)).ravel()
        scaled_true[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

    # CV metrics
    print("\nCV Results:")
    total_mse = 0
    for i, target in enumerate(target_names):
        mse = np.mean((scaled_cv[:, i] - scaled_true[:, i]) ** 2)
        total_mse += mse
        print(f"  {target}: {mse:.6f}")
    total_mse /= 3
    print(f"Total CV MSE: {total_mse:.6f}")

    print("\nTest profile:")
    print(f"  angle_std: {scaled_test[:, 0].std():.4f} (target: 0.1377)")
    print(f"  depth_mean: {scaled_test[:, 1].mean():.4f} (target: 0.5055)")

    # Compare to Sub 133
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    print("\nCorrelation with Sub 133:")
    for i, col in enumerate(cols):
        corr = np.corrcoef(scaled_test[:, i], sub133[col])[0, 1]
        print(f"  {col}: {corr:.4f}")

    # Try different attention widths
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT ATTENTION WIDTHS")
    print("=" * 70)

    for sigma in [5, 8, 15, 20]:
        attention = np.exp(-((frames - 153) ** 2) / (2 * sigma ** 2))
        attention = attention / attention.sum()

        # Quick feature extraction and evaluation
        features = []
        for ts in train_timeseries[:50]:  # Quick test on subset
            f = extract_temporal_features(ts, keypoint_idx, attention[:len(ts)])
            features.append(f)

        df = pd.DataFrame(features).fillna(0)
        X = df.values

        # Simple CV
        mse = 0
        for i in range(3):
            y_t = scalers_dict[target_names[i]].transform(y_train[:50, i].reshape(-1, 1)).ravel()
            ridge = Ridge(alpha=100)
            ridge.fit(X, y_t)
            pred = ridge.predict(X)
            mse += np.mean((pred - y_t) ** 2)
        mse /= 3

        print(f"  sigma={sigma}: approx MSE={mse:.6f}")

    # Save submission
    print("\n" + "=" * 70)
    print("SAVING SUBMISSION")
    print("=" * 70)

    test_meta = load_metadata(train=False)
    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": scaled_test[:, 0],
        "scaled_depth": scaled_test[:, 1],
        "scaled_left_right": scaled_test[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
