"""
Comprehensive Feature Engineering Test

Goal: Find a strategy that will 100% score < 0.007 on LB
Current best: 0.007809 (Sub 133)
Required improvement: 10.4%

Approach: Test EVERY novel feature engineering approach systematically,
measuring both CV and profile match to Sub 133.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy import signal
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


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


def smooth_timeseries(ts, window=5):
    """Apply smoothing to reduce noise."""
    smoothed = np.zeros_like(ts)
    for col in range(ts.shape[1]):
        smoothed[:, col] = np.convolve(ts[:, col], np.ones(window)/window, mode='same')
    return smoothed


def detect_release_frame(timeseries, keypoint_idx):
    """Detect release frame based on wrist velocity peak."""
    if "right_wrist" not in keypoint_idx:
        return 153  # default

    idx = keypoint_idx["right_wrist"]
    wrist_pos = timeseries[:, idx*3:(idx+1)*3]

    # Compute velocity
    vel = np.diff(wrist_pos, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Find peak after frame 100
    search_start = 100
    if len(speed) > search_start + 10:
        peak_idx = search_start + np.argmax(speed[search_start:search_start+80])
        return peak_idx
    return 153


def extract_features_v1(timeseries, keypoint_idx, participant_id):
    """V1: Standard frame-based features with smoothing."""
    ts = smooth_timeseries(timeseries, window=5)
    features = {}

    frames = [145, 148, 150, 153, 155, 158, 160]

    for joint in KEY_JOINTS:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in frames:
            if f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]
                features[f"{joint}_x_f{f}"] = pos[0]
                features[f"{joint}_y_f{f}"] = pos[1]
                features[f"{joint}_z_f{f}"] = pos[2]

    return features


def extract_features_v2(timeseries, keypoint_idx, participant_id):
    """V2: Player-normalized features."""
    ts = smooth_timeseries(timeseries, window=5)
    features = {}

    # Features relative to player's mean position
    frames = [145, 148, 150, 153, 155, 158, 160]

    for joint in KEY_JOINTS:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        # Compute mean position over all frames
        joint_mean = np.mean(ts[:, idx*3:(idx+1)*3], axis=0)

        for f in frames:
            if f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]
                # Deviation from mean
                dev = pos - joint_mean
                features[f"{joint}_dev_x_f{f}"] = dev[0]
                features[f"{joint}_dev_y_f{f}"] = dev[1]
                features[f"{joint}_dev_z_f{f}"] = dev[2]

    return features


def extract_features_v3(timeseries, keypoint_idx, participant_id):
    """V3: Release-aligned features."""
    ts = smooth_timeseries(timeseries, window=5)
    release = detect_release_frame(ts, keypoint_idx)

    features = {}

    # Frames relative to release
    offsets = [-8, -5, -3, 0, 2, 5, 7]

    for joint in KEY_JOINTS:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for offset in offsets:
            f = release + offset
            if 0 <= f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]
                features[f"{joint}_x_rel{offset:+d}"] = pos[0]
                features[f"{joint}_y_rel{offset:+d}"] = pos[1]
                features[f"{joint}_z_rel{offset:+d}"] = pos[2]

    features["release_frame"] = release
    return features


def extract_features_v4(timeseries, keypoint_idx, participant_id):
    """V4: Velocity and acceleration at key moments."""
    ts = smooth_timeseries(timeseries, window=7)  # More smoothing for derivatives

    features = {}
    release = 153

    for joint in KEY_JOINTS:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        pos = ts[:, idx*3:(idx+1)*3]

        # Velocity (central difference)
        vel = np.zeros_like(pos)
        vel[1:-1] = (pos[2:] - pos[:-2]) / 2

        # Acceleration
        acc = np.zeros_like(pos)
        acc[1:-1] = pos[2:] - 2*pos[1:-1] + pos[:-2]

        # At release
        if release < len(ts):
            features[f"{joint}_vel_x_release"] = vel[release, 0]
            features[f"{joint}_vel_y_release"] = vel[release, 1]
            features[f"{joint}_vel_z_release"] = vel[release, 2]
            features[f"{joint}_speed_release"] = np.linalg.norm(vel[release])

            features[f"{joint}_acc_x_release"] = acc[release, 0]
            features[f"{joint}_acc_y_release"] = acc[release, 1]
            features[f"{joint}_acc_z_release"] = acc[release, 2]

    return features


def extract_features_v5(timeseries, keypoint_idx, participant_id):
    """V5: Joint angles and distances (relative features only)."""
    ts = smooth_timeseries(timeseries, window=5)

    features = {}
    frames = [148, 150, 153, 155, 158]

    # Joint pairs for distance
    pairs = [
        ("right_wrist", "right_elbow"),
        ("right_elbow", "right_shoulder"),
        ("right_shoulder", "right_hip"),
        ("right_wrist", "mid_hip"),
        ("left_wrist", "right_wrist"),
    ]

    for j1, j2 in pairs:
        if j1 not in keypoint_idx or j2 not in keypoint_idx:
            continue

        idx1, idx2 = keypoint_idx[j1], keypoint_idx[j2]

        for f in frames:
            if f < len(ts):
                p1 = ts[f, idx1*3:(idx1+1)*3]
                p2 = ts[f, idx2*3:(idx2+1)*3]
                dist = np.linalg.norm(p1 - p2)
                features[f"dist_{j1}_{j2}_f{f}"] = dist

    # Elbow angle
    if all(j in keypoint_idx for j in ["right_wrist", "right_elbow", "right_shoulder"]):
        for f in frames:
            if f < len(ts):
                wrist = ts[f, keypoint_idx["right_wrist"]*3:(keypoint_idx["right_wrist"]+1)*3]
                elbow = ts[f, keypoint_idx["right_elbow"]*3:(keypoint_idx["right_elbow"]+1)*3]
                shoulder = ts[f, keypoint_idx["right_shoulder"]*3:(keypoint_idx["right_shoulder"]+1)*3]

                v1 = wrist - elbow
                v2 = shoulder - elbow
                cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                features[f"elbow_angle_f{f}"] = np.arccos(np.clip(cos_ang, -1, 1))

    return features


def extract_features_v6(timeseries, keypoint_idx, participant_id):
    """V6: Combined best features from all approaches."""
    f1 = extract_features_v1(timeseries, keypoint_idx, participant_id)
    f4 = extract_features_v4(timeseries, keypoint_idx, participant_id)
    f5 = extract_features_v5(timeseries, keypoint_idx, participant_id)

    # Combine with prefixes to avoid collision
    combined = {}
    for k, v in f1.items():
        combined[f"pos_{k}"] = v
    for k, v in f4.items():
        combined[f"vel_{k}"] = v
    for k, v in f5.items():
        combined[f"rel_{k}"] = v

    return combined


def extract_features_v7(timeseries, keypoint_idx, participant_id):
    """V7: Minimal stable features (from drift analysis)."""
    ts = smooth_timeseries(timeseries, window=5)

    features = {}

    # Only the most stable features from drift analysis
    stable_joints = ["right_knee", "right_elbow", "mid_hip", "neck"]
    frames = [150, 153, 155]

    for joint in stable_joints:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in frames:
            if f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]
                features[f"{joint}_z_f{f}"] = pos[2]  # Z only (most stable)

        # Velocity at release
        if 153 < len(ts) - 1:
            vel = ts[154, idx*3:(idx+1)*3] - ts[152, idx*3:(idx+1)*3]
            features[f"{joint}_vel_z_release"] = vel[2]

    return features


def evaluate_feature_set(feature_extractor, name, max_shots=None):
    """Evaluate a feature extraction approach."""
    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    keypoint_idx = get_keypoint_indices()

    print(f"\nEvaluating: {name}")
    print("  Extracting features...")

    feature_list = []
    y_list = []
    pids_list = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if max_shots and i >= max_shots:
            break

        features = feature_extractor(timeseries, keypoint_idx, metadata["participant_id"])
        feature_list.append(features)
        y_list.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        pids_list.append(metadata["participant_id"])

    df = pd.DataFrame(feature_list).fillna(0)
    X = df.values
    y = np.array(y_list)
    pids = np.array(pids_list)

    print(f"  Features: {X.shape[1]}")

    # Within-player CV
    all_preds = np.zeros_like(y)

    for pid in np.unique(pids):
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask]
        indices = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_pid):
            for t_idx in range(3):
                ridge = Ridge(alpha=100)
                ridge.fit(X_scaled[train_idx], y_pid[train_idx, t_idx])
                all_preds[indices[test_idx], t_idx] = ridge.predict(X_scaled[test_idx])

    # Metrics
    mse_per_target = {}
    for i, target in enumerate(target_names):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))

    # Profile
    scaled_preds = np.zeros_like(all_preds)
    for i, target in enumerate(target_names):
        scaled_preds[:, i] = scalers_dict[target].transform(all_preds[:, i].reshape(-1, 1)).ravel()

    angle_std = np.std(scaled_preds[:, 0])
    depth_mean = np.mean(scaled_preds[:, 1])

    # Profile distance from Sub 133 (LB 0.007809)
    profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

    print(f"  CV MSE: {total_mse:.6f}")
    print(f"  angle_std: {angle_std:.4f} (target: 0.1377)")
    print(f"  depth_mean: {depth_mean:.4f} (target: 0.5055)")
    print(f"  Profile distance: {profile_dist:.4f}")

    return {
        "name": name,
        "cv_mse": total_mse,
        "angle_mse": mse_per_target["angle"],
        "depth_mse": mse_per_target["depth"],
        "lr_mse": mse_per_target["left_right"],
        "angle_std": angle_std,
        "depth_mean": depth_mean,
        "profile_dist": profile_dist,
        "n_features": X.shape[1]
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE FEATURE ENGINEERING TEST")
    print("=" * 70)
    print("\nGoal: Find features that achieve < 0.007 LB")
    print("Current best: Sub 133 with LB 0.007809")
    print("Sub 133 profile: angle_std=0.1377, depth_mean=0.5055")

    results = []

    # Test all feature extraction approaches
    approaches = [
        (extract_features_v1, "V1: Standard frame-based"),
        (extract_features_v2, "V2: Player-normalized"),
        (extract_features_v3, "V3: Release-aligned"),
        (extract_features_v4, "V4: Velocity/Acceleration"),
        (extract_features_v5, "V5: Relative features only"),
        (extract_features_v6, "V6: Combined (pos+vel+rel)"),
        (extract_features_v7, "V7: Minimal stable"),
    ]

    for extractor, name in approaches:
        result = evaluate_feature_set(extractor, name)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("profile_dist")

    print("\nRanked by Profile Match (lower = closer to Sub 133):")
    print(results_df[["name", "cv_mse", "angle_std", "depth_mean", "profile_dist"]].to_string(index=False))

    # Save
    results_df.to_csv(OUTPUT_DIR / "comprehensive_feature_test.csv", index=False)

    # Best approach
    best = results_df.iloc[0]

    print(f"\n{'=' * 70}")
    print("BEST APPROACH")
    print(f"{'=' * 70}")
    print(f"  {best['name']}")
    print(f"  CV MSE: {best['cv_mse']:.6f}")
    print(f"  Profile: angle_std={best['angle_std']:.4f}, depth_mean={best['depth_mean']:.4f}")

    # Can this beat Sub 133?
    sub133_profile = 0.1377 + 0.5055  # Just for reference

    if best['profile_dist'] < 0.01 and best['cv_mse'] < 0.010:
        print("\n*** PROMISING: Profile matches Sub 133 well ***")
        print("Worth creating submission to test on LB")
    else:
        print("\nNot clearly better than Sub 133 based on profile match.")
        print("Need to explore more approaches.")


if __name__ == "__main__":
    main()
