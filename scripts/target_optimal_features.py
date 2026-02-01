"""
Target-Specific Optimal Feature Selection

For each target (angle, depth, left_right), find the optimal:
1. Frames (which time points)
2. Joints (which body parts)
3. Axes (x, y, z)
4. Derived features (velocity, acceleration, angles)

Use within-player CV as the optimization objective.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def smooth_timeseries(ts, window=5):
    """Apply smoothing."""
    smoothed = np.zeros_like(ts)
    for col in range(ts.shape[1]):
        smoothed[:, col] = np.convolve(ts[:, col], np.ones(window)/window, mode='same')
    return smoothed


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


KEY_JOINTS = [
    "right_wrist", "right_elbow", "right_shoulder", "right_hip",
    "right_knee", "left_wrist", "neck", "mid_hip", "right_ankle",
    "left_elbow", "left_shoulder"
]


def extract_all_features(timeseries, keypoint_idx):
    """Extract comprehensive feature set for feature selection."""
    ts = smooth_timeseries(timeseries, window=5)
    features = {}

    # Frames around release
    frames = list(range(140, 170))

    for joint in KEY_JOINTS:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in frames:
            if f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]

                # Position features
                features[f"{joint}_x_f{f}"] = pos[0]
                features[f"{joint}_y_f{f}"] = pos[1]
                features[f"{joint}_z_f{f}"] = pos[2]

                # Velocity (if we have enough frames)
                if f > 0 and f < len(ts) - 1:
                    vel = (ts[f+1, idx*3:(idx+1)*3] - ts[f-1, idx*3:(idx+1)*3]) / 2
                    features[f"{joint}_vx_f{f}"] = vel[0]
                    features[f"{joint}_vy_f{f}"] = vel[1]
                    features[f"{joint}_vz_f{f}"] = vel[2]
                    features[f"{joint}_speed_f{f}"] = np.linalg.norm(vel)

    # Joint distances at key frames
    key_frames = [148, 150, 153, 155, 158]
    pairs = [
        ("right_wrist", "right_elbow"),
        ("right_elbow", "right_shoulder"),
        ("right_shoulder", "right_hip"),
        ("right_wrist", "mid_hip"),
        ("right_wrist", "neck"),
    ]

    for j1, j2 in pairs:
        if j1 not in keypoint_idx or j2 not in keypoint_idx:
            continue
        idx1, idx2 = keypoint_idx[j1], keypoint_idx[j2]

        for f in key_frames:
            if f < len(ts):
                p1 = ts[f, idx1*3:(idx1+1)*3]
                p2 = ts[f, idx2*3:(idx2+1)*3]
                dist = np.linalg.norm(p1 - p2)
                features[f"dist_{j1}_{j2}_f{f}"] = dist

    # Elbow angle at key frames
    if all(j in keypoint_idx for j in ["right_wrist", "right_elbow", "right_shoulder"]):
        for f in key_frames:
            if f < len(ts):
                wrist = ts[f, keypoint_idx["right_wrist"]*3:(keypoint_idx["right_wrist"]+1)*3]
                elbow = ts[f, keypoint_idx["right_elbow"]*3:(keypoint_idx["right_elbow"]+1)*3]
                shoulder = ts[f, keypoint_idx["right_shoulder"]*3:(keypoint_idx["right_shoulder"]+1)*3]

                v1 = wrist - elbow
                v2 = shoulder - elbow
                cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                features[f"elbow_angle_f{f}"] = np.arccos(np.clip(cos_ang, -1, 1))

    return features


def evaluate_feature_subset(X, y, pids, feature_mask, alpha=100):
    """Evaluate a feature subset using within-player CV."""
    X_sub = X[:, feature_mask]

    all_preds = np.zeros(len(y))

    for pid in np.unique(pids):
        mask = pids == pid
        X_pid = X_sub[mask]
        y_pid = y[mask]
        indices = np.where(mask)[0]

        if X_pid.shape[1] == 0:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_pid):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_scaled[train_idx], y_pid[train_idx])
            all_preds[indices[test_idx]] = ridge.predict(X_scaled[test_idx])

    mse = np.mean((all_preds - y) ** 2)
    return mse, all_preds


def greedy_feature_selection(X, y, pids, feature_names, target_name, max_features=50):
    """Greedy forward feature selection for a single target."""
    print(f"\n  Selecting features for {target_name}...")

    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))

    best_mse = float("inf")

    for step in range(max_features):
        best_feature = None
        best_feature_mse = float("inf")

        for feat_idx in remaining:
            current = selected + [feat_idx]
            mask = np.zeros(n_features, dtype=bool)
            mask[current] = True

            mse, _ = evaluate_feature_subset(X, y, pids, mask)

            if mse < best_feature_mse:
                best_feature_mse = mse
                best_feature = feat_idx

        if best_feature is None or best_feature_mse >= best_mse:
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_mse = best_feature_mse

        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}: {len(selected)} features, MSE={best_mse:.6f}")

    print(f"    Final: {len(selected)} features, MSE={best_mse:.6f}")

    return selected, best_mse


def main():
    print("=" * 70)
    print("TARGET-SPECIFIC OPTIMAL FEATURE SELECTION")
    print("=" * 70)

    scalers_dict = load_scalers()
    keypoint_idx = get_keypoint_indices()

    # Build full feature matrix
    print("\nExtracting features...")
    feature_list = []
    y_list = []
    pids_list = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_all_features(timeseries, keypoint_idx)
        feature_list.append(features)
        y_list.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        pids_list.append(metadata["participant_id"])

    df = pd.DataFrame(feature_list).fillna(0)
    df = df.loc[:, df.nunique() > 1]
    feature_names = df.columns.tolist()

    X = df.values
    y = np.array(y_list)
    pids = np.array(pids_list)

    print(f"Total features: {X.shape[1]}")

    # Target-specific feature selection
    target_names = ["angle", "depth", "left_right"]
    selected_features = {}
    target_mse = {}

    for t_idx, target in enumerate(target_names):
        # Scale target
        y_scaled = scalers_dict[target].transform(y[:, t_idx].reshape(-1, 1)).ravel()

        # Greedy selection
        selected, mse = greedy_feature_selection(X, y_scaled, pids, feature_names, target, max_features=40)
        selected_features[target] = [feature_names[i] for i in selected]
        target_mse[target] = mse

        print(f"\n  Top 10 features for {target}:")
        for i, feat in enumerate(selected_features[target][:10]):
            print(f"    {i+1}. {feat}")

    # Now train final model with target-specific features
    print("\n" + "=" * 70)
    print("FINAL MODEL WITH TARGET-SPECIFIC FEATURES")
    print("=" * 70)

    # Build test features
    print("\nBuilding test features...")
    test_feature_list = []
    test_pids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_all_features(timeseries, keypoint_idx)
        test_feature_list.append(features)
        test_pids.append(metadata["participant_id"])

    test_df = pd.DataFrame(test_feature_list).fillna(0)
    test_pids = np.array(test_pids)

    # Train and predict for each target
    predictions = np.zeros((len(test_df), 3))
    cv_predictions = np.zeros((len(y), 3))

    for t_idx, target in enumerate(target_names):
        print(f"\n  Training for {target}...")

        y_target = scalers_dict[target].transform(y[:, t_idx].reshape(-1, 1)).ravel()

        # Get selected features for this target
        feat_cols = selected_features[target]
        X_train = df[feat_cols].values
        X_test = test_df[feat_cols].fillna(0).values

        # CV predictions
        cv_preds = np.zeros(len(y))

        for pid in np.unique(pids):
            train_mask = pids == pid
            test_mask = test_pids == pid

            X_tr = X_train[train_mask]
            y_tr = y_target[train_mask]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_test[test_mask])

            ridge = Ridge(alpha=100)
            ridge.fit(X_tr_scaled, y_tr)

            predictions[test_mask, t_idx] = ridge.predict(X_te_scaled)

            # Within-player CV
            indices = np.where(train_mask)[0]
            X_pid_scaled = scaler.fit_transform(X_train[train_mask])

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, te_idx in kf.split(X_pid_scaled):
                ridge = Ridge(alpha=100)
                ridge.fit(X_pid_scaled[tr_idx], y_tr[tr_idx])
                cv_preds[indices[te_idx]] = ridge.predict(X_pid_scaled[te_idx])

        cv_predictions[:, t_idx] = cv_preds

        cv_mse = np.mean((cv_preds - y_target) ** 2)
        print(f"    CV MSE: {cv_mse:.6f}")

    # Overall metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total_cv_mse = 0
    for t_idx, target in enumerate(target_names):
        y_target = scalers_dict[target].transform(y[:, t_idx].reshape(-1, 1)).ravel()
        mse = np.mean((cv_predictions[:, t_idx] - y_target) ** 2)
        total_cv_mse += mse
        print(f"  {target} CV MSE: {mse:.6f}")

    total_cv_mse /= 3
    print(f"\nTotal CV MSE: {total_cv_mse:.6f}")

    # Profile
    print(f"\nProfile:")
    print(f"  angle_std: {predictions[:, 0].std():.4f} (target: 0.1377)")
    print(f"  depth_mean: {predictions[:, 1].mean():.4f} (target: 0.5055)")

    # Compare to Sub 133
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    print(f"\nCorrelation with Sub 133:")
    for t_idx, col in enumerate(["scaled_angle", "scaled_depth", "scaled_left_right"]):
        corr = np.corrcoef(predictions[:, t_idx], sub133[col])[0, 1]
        print(f"  {col}: {corr:.4f}")

    # Save submission
    test_meta = load_metadata(train=False)

    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": predictions[:, 0],
        "scaled_depth": predictions[:, 1],
        "scaled_left_right": predictions[:, 2]
    })

    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Save selected features
    features_path = OUTPUT_DIR / "target_optimal_features.csv"
    max_len = max(len(v) for v in selected_features.values())
    feat_df = pd.DataFrame({
        target: selected_features[target] + [None] * (max_len - len(selected_features[target]))
        for target in target_names
    })
    feat_df.to_csv(features_path, index=False)
    print(f"Saved features: {features_path}")


if __name__ == "__main__":
    main()
