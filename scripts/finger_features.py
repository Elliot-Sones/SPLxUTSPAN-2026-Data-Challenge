"""
Finger and Hand Features

Focus on hand and finger keypoints at release, which directly affect ball trajectory.
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


def smooth_ts(ts, window=5):
    smoothed = np.zeros_like(ts)
    for col in range(ts.shape[1]):
        smoothed[:, col] = np.convolve(ts[:, col], np.ones(window)/window, mode='same')
    return smoothed


def extract_finger_features(timeseries, keypoint_idx):
    """Extract features focused on hands and fingers."""
    ts = smooth_ts(timeseries, window=5)
    features = {}

    # Hand keypoints
    hand_joints = [
        "right_wrist", "right_thumb", "right_pinky",
        "left_wrist", "left_thumb", "left_pinky"
    ]

    # Arm joints for context
    arm_joints = ["right_elbow", "right_shoulder"]

    # Release frames (critical window)
    release_frames = [150, 151, 152, 153, 154, 155, 156]

    # 1. Hand positions at each release frame
    for joint in hand_joints + arm_joints:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in release_frames:
            if f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]
                features[f"{joint}_x_f{f}"] = pos[0]
                features[f"{joint}_y_f{f}"] = pos[1]
                features[f"{joint}_z_f{f}"] = pos[2]

    # 2. Hand velocity at release
    for joint in hand_joints:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in [152, 153, 154]:
            if f > 0 and f < len(ts) - 1:
                vel = (ts[f+1, idx*3:(idx+1)*3] - ts[f-1, idx*3:(idx+1)*3]) / 2
                features[f"{joint}_vx_f{f}"] = vel[0]
                features[f"{joint}_vy_f{f}"] = vel[1]
                features[f"{joint}_vz_f{f}"] = vel[2]
                features[f"{joint}_speed_f{f}"] = np.linalg.norm(vel)

    # 3. Finger spread (thumb to pinky distance)
    if "right_thumb" in keypoint_idx and "right_pinky" in keypoint_idx:
        thumb_idx = keypoint_idx["right_thumb"]
        pinky_idx = keypoint_idx["right_pinky"]

        for f in release_frames:
            if f < len(ts):
                thumb = ts[f, thumb_idx*3:(thumb_idx+1)*3]
                pinky = ts[f, pinky_idx*3:(pinky_idx+1)*3]
                features[f"finger_spread_f{f}"] = np.linalg.norm(thumb - pinky)

    # 4. Wrist to finger distances
    for finger in ["right_thumb", "right_pinky"]:
        if finger not in keypoint_idx or "right_wrist" not in keypoint_idx:
            continue
        finger_idx = keypoint_idx[finger]
        wrist_idx = keypoint_idx["right_wrist"]

        for f in release_frames:
            if f < len(ts):
                finger_pos = ts[f, finger_idx*3:(finger_idx+1)*3]
                wrist_pos = ts[f, wrist_idx*3:(wrist_idx+1)*3]
                features[f"{finger}_wrist_dist_f{f}"] = np.linalg.norm(finger_pos - wrist_pos)

    # 5. Hand orientation (thumb to pinky vector direction)
    if "right_thumb" in keypoint_idx and "right_pinky" in keypoint_idx:
        thumb_idx = keypoint_idx["right_thumb"]
        pinky_idx = keypoint_idx["right_pinky"]

        for f in [152, 153, 154]:
            if f < len(ts):
                thumb = ts[f, thumb_idx*3:(thumb_idx+1)*3]
                pinky = ts[f, pinky_idx*3:(pinky_idx+1)*3]
                hand_vec = thumb - pinky
                norm = np.linalg.norm(hand_vec) + 1e-8
                features[f"hand_orient_x_f{f}"] = hand_vec[0] / norm
                features[f"hand_orient_y_f{f}"] = hand_vec[1] / norm
                features[f"hand_orient_z_f{f}"] = hand_vec[2] / norm

    # 6. Arm extension (wrist height relative to shoulder)
    if "right_wrist" in keypoint_idx and "right_shoulder" in keypoint_idx:
        wrist_idx = keypoint_idx["right_wrist"]
        shoulder_idx = keypoint_idx["right_shoulder"]

        for f in release_frames:
            if f < len(ts):
                wrist = ts[f, wrist_idx*3:(wrist_idx+1)*3]
                shoulder = ts[f, shoulder_idx*3:(shoulder_idx+1)*3]
                features[f"wrist_above_shoulder_f{f}"] = wrist[1] - shoulder[1]
                features[f"arm_extension_f{f}"] = np.linalg.norm(wrist - shoulder)

    # 7. Two-hand coordination (for guide hand)
    if "right_wrist" in keypoint_idx and "left_wrist" in keypoint_idx:
        r_idx = keypoint_idx["right_wrist"]
        l_idx = keypoint_idx["left_wrist"]

        for f in release_frames:
            if f < len(ts):
                r_wrist = ts[f, r_idx*3:(r_idx+1)*3]
                l_wrist = ts[f, l_idx*3:(l_idx+1)*3]
                features[f"hand_separation_f{f}"] = np.linalg.norm(r_wrist - l_wrist)

    return features


def main():
    print("=" * 70)
    print("FINGER AND HAND FEATURES")
    print("=" * 70)

    scalers_dict = load_scalers()
    keypoint_idx = get_keypoint_indices()

    # List available hand/finger keypoints
    print("\nAvailable hand/finger keypoints:")
    for name in keypoint_idx:
        if any(x in name.lower() for x in ["wrist", "thumb", "pinky", "hand"]):
            print(f"  {name}")

    # Build features
    print("\nExtracting finger features...")

    train_features = []
    train_targets = []
    train_pids = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_finger_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        train_pids.append(metadata["participant_id"])

    train_df = pd.DataFrame(train_features).fillna(0)
    train_df = train_df.loc[:, train_df.nunique() > 1]

    X_train = train_df.values
    y_train = np.array(train_targets)
    pids_train = np.array(train_pids)

    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    test_features = []
    test_pids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_finger_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_pids.append(metadata["participant_id"])

    test_df = pd.DataFrame(test_features).fillna(0)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    X_test = test_df[train_df.columns].values
    pids_test = np.array(test_pids)

    print(f"Test: {X_test.shape[0]} samples")

    target_names = ["angle", "depth", "left_right"]

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

    # Scale predictions
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
        print(f"  {target} MSE: {mse:.6f}")
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

    # Try blending with Sub 133
    print("\n" + "=" * 70)
    print("BLENDING WITH SUB 133")
    print("=" * 70)

    best_blend = None
    best_profile_dist = float("inf")
    best_weight = 0

    for w in np.linspace(0, 0.3, 31):
        blended = (1 - w) * sub133[cols].values + w * scaled_test

        angle_std = blended[:, 0].std()
        depth_mean = blended[:, 1].mean()
        profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

        if profile_dist < best_profile_dist:
            best_profile_dist = profile_dist
            best_weight = w
            best_blend = blended.copy()

    print(f"Best weight: {best_weight:.2f}")
    print(f"Profile dist: {best_profile_dist:.6f}")

    if best_weight > 0:
        corr = np.corrcoef(best_blend.ravel(), sub133[cols].values.ravel())[0, 1]
        print(f"Blend correlation with Sub 133: {corr:.4f}")

    # Save submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    test_meta = load_metadata(train=False)
    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    # Save finger-only model
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": scaled_test[:, 0],
        "scaled_depth": scaled_test[:, 1],
        "scaled_left_right": scaled_test[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved finger model: {path}")

    if best_weight > 0:
        next_num += 1
        submission = pd.DataFrame({
            "id": test_meta["id"].values,
            "scaled_angle": best_blend[:, 0],
            "scaled_depth": best_blend[:, 1],
            "scaled_left_right": best_blend[:, 2]
        })
        path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(path, index=False)
        print(f"Saved finger blend: {path}")


if __name__ == "__main__":
    main()
