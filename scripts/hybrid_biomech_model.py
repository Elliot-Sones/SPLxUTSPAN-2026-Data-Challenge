"""
Hybrid Biomechanical + PCA Model

Combine:
1. PCA features from raw keypoints (captures general patterns)
2. Advanced biomech features (captures physics/mechanics)
3. Key frame features (captures release timing)

Goal: Create a more accurate model that blends well with Sub 113.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
from tqdm import tqdm
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}

KEYPOINT_MAP = {}


def init_keypoint_mapping(keypoint_cols):
    global KEYPOINT_MAP
    for i, col in enumerate(keypoint_cols):
        KEYPOINT_MAP[col] = i


def get_kp(series, name, frame=None):
    if name not in KEYPOINT_MAP:
        return None
    idx = KEYPOINT_MAP[name]
    if frame is not None:
        return series[frame, idx]
    return series[:, idx]


def smooth_signal(signal, window=5):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < window:
        return signal
    try:
        return savgol_filter(signal, window, 2)
    except:
        return signal


def compute_velocity(signal):
    return np.gradient(smooth_signal(signal))


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_target_ranges():
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }


def extract_hybrid_features(series, participant_id):
    """Extract hybrid features combining PCA, biomech, and key frames."""
    features = {}

    # Key frames based on research
    release_frames = list(range(145, 165))  # Release window
    early_frames = list(range(80, 120))     # Load phase
    late_frames = list(range(165, 200))     # Follow-through

    # =====================================================
    # 1. KEY JOINT VELOCITIES (most important features)
    # =====================================================
    key_joints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "right_knee", "right_hip", "left_wrist"
    ]

    for joint in key_joints:
        for axis in ["x", "y", "z"]:
            data = get_kp(series, f"{joint}_{axis}")
            if data is not None:
                vel = compute_velocity(data)

                # Velocity at key frames
                features[f"{joint}_{axis}_vel_release"] = np.nanmean(vel[release_frames])
                features[f"{joint}_{axis}_vel_early"] = np.nanmean(vel[early_frames])

                # Position at key frames
                features[f"{joint}_{axis}_pos_release"] = np.nanmean(data[release_frames])

                # Velocity timing
                peak_frame = np.argmax(np.abs(vel))
                features[f"{joint}_{axis}_vel_peak_time"] = peak_frame / 240.0

    # =====================================================
    # 2. KINETIC CHAIN FEATURES
    # =====================================================
    right_hip_z = get_kp(series, "right_hip_z")
    right_knee_z = get_kp(series, "right_knee_z")
    right_shoulder_z = get_kp(series, "right_shoulder_z")
    right_elbow_z = get_kp(series, "right_elbow_z")
    right_wrist_z = get_kp(series, "right_wrist_z")

    if all(x is not None for x in [right_hip_z, right_knee_z, right_shoulder_z, right_elbow_z, right_wrist_z]):
        # Velocity peaks for kinetic chain timing
        hip_vel = compute_velocity(right_hip_z)
        knee_vel = compute_velocity(right_knee_z)
        shoulder_vel = compute_velocity(right_shoulder_z)
        elbow_vel = compute_velocity(right_elbow_z)
        wrist_vel = compute_velocity(right_wrist_z)

        features["kinetic_hip_peak"] = np.argmax(np.abs(hip_vel)) / 240.0
        features["kinetic_knee_peak"] = np.argmax(np.abs(knee_vel)) / 240.0
        features["kinetic_shoulder_peak"] = np.argmax(np.abs(shoulder_vel)) / 240.0
        features["kinetic_elbow_peak"] = np.argmax(np.abs(elbow_vel)) / 240.0
        features["kinetic_wrist_peak"] = np.argmax(np.abs(wrist_vel)) / 240.0

        # Velocity ratios (distal should be faster than proximal)
        features["vel_ratio_wrist_elbow"] = np.nanmax(np.abs(wrist_vel)) / (np.nanmax(np.abs(elbow_vel)) + 1e-6)
        features["vel_ratio_elbow_shoulder"] = np.nanmax(np.abs(elbow_vel)) / (np.nanmax(np.abs(shoulder_vel)) + 1e-6)

        # Wrist snap (key for angle prediction)
        wrist_snap = wrist_vel - elbow_vel
        features["wrist_snap_max"] = np.nanmax(wrist_snap)
        features["wrist_snap_release"] = np.nanmean(wrist_snap[release_frames])

    # =====================================================
    # 3. ELBOW-WRIST COORDINATION (critical for accuracy)
    # =====================================================
    if right_elbow_z is not None and right_wrist_z is not None:
        elbow_vel = compute_velocity(right_elbow_z)
        wrist_vel = compute_velocity(right_wrist_z)

        # Coordination in release window (research: more coordination = better)
        corr = np.corrcoef(elbow_vel[130:175], wrist_vel[130:175])[0, 1]
        features["elbow_wrist_coord"] = corr if not np.isnan(corr) else 0

        # Variability in last frames (research: more variability = worse)
        features["release_variability"] = np.nanstd(elbow_vel[160:175]) + np.nanstd(wrist_vel[160:175])

    # =====================================================
    # 4. BODY ALIGNMENT AND STABILITY
    # =====================================================
    mid_hip_z = get_kp(series, "mid_hip_z")
    left_shoulder_z = get_kp(series, "left_shoulder_z")
    right_shoulder_z = get_kp(series, "right_shoulder_z")

    if all(x is not None for x in [mid_hip_z, left_shoulder_z, right_shoulder_z]):
        # Center of mass stability
        mid_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2
        com_z = 0.6 * mid_hip_z + 0.4 * mid_shoulder_z

        features["com_stability_release"] = np.nanstd(com_z[release_frames])
        features["com_z_release"] = np.nanmean(com_z[release_frames])

        # Shoulder alignment (symmetry)
        shoulder_diff = right_shoulder_z - left_shoulder_z
        features["shoulder_symmetry"] = np.nanmean(shoulder_diff[release_frames])

    # =====================================================
    # 5. GUIDE HAND (LEFT WRIST) FEATURES
    # =====================================================
    left_wrist_x = get_kp(series, "left_wrist_x")
    left_wrist_y = get_kp(series, "left_wrist_y")
    left_wrist_z = get_kp(series, "left_wrist_z")

    if all(x is not None for x in [left_wrist_x, left_wrist_y, left_wrist_z]):
        left_vx = compute_velocity(left_wrist_x)
        left_vz = compute_velocity(left_wrist_z)

        features["guide_hand_vx_release"] = np.nanmean(left_vx[release_frames])
        features["guide_hand_vz_release"] = np.nanmean(left_vz[release_frames])
        features["guide_hand_z_release"] = np.nanmean(left_wrist_z[release_frames])

    # Clean up NaN/Inf
    for k, v in list(features.items()):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


def load_and_extract_features():
    """Load data and extract features."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    init_keypoint_mapping(keypoint_cols)

    # Key frame windows for PCA
    pca_frames = list(range(140, 170))

    def process_df(df, desc="Processing"):
        all_pca = []
        all_hybrid = []
        all_pids = []
        all_ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            # PCA features (raw keypoints at key frames)
            pca_data = ts[pca_frames, :].flatten()
            pca_data = np.nan_to_num(pca_data, nan=0.0)
            all_pca.append(pca_data)

            # Hybrid features
            hybrid = extract_hybrid_features(ts, row['participant_id'])
            all_hybrid.append(hybrid)

            all_pids.append(row['participant_id'])
            all_ids.append(row['id'])

        return np.array(all_pca), pd.DataFrame(all_hybrid), np.array(all_pids), np.array(all_ids)

    print("Processing training data...")
    train_pca, train_hybrid, train_pids, train_ids = process_df(train_df, "Train")
    train_targets = train_df[["angle", "depth", "left_right"]].values

    print("Processing test data...")
    test_pca, test_hybrid, test_pids, test_ids = process_df(test_df, "Test")

    return {
        "train_pca": train_pca,
        "train_hybrid": train_hybrid,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_pca": test_pca,
        "test_hybrid": test_hybrid,
        "test_pids": test_pids,
        "test_ids": test_ids,
    }


def train_hybrid_model(data):
    """Train hybrid model using both PCA and biomech features."""
    print("\n" + "=" * 60)
    print("HYBRID BIOMECH + PCA MODEL")
    print("=" * 60)

    train_pca = data["train_pca"]
    train_hybrid = np.nan_to_num(data["train_hybrid"].values, nan=0.0)
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]

    test_pca = data["test_pca"]
    test_hybrid = np.nan_to_num(data["test_hybrid"].values, nan=0.0)
    test_pids = data["test_pids"]

    print(f"PCA features: {train_pca.shape[1]}")
    print(f"Hybrid features: {train_hybrid.shape[1]}")

    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_pca), 3))
    oof_preds = np.zeros_like(train_targets)

    for t_idx, target in enumerate(TARGETS):
        print(f"\n{target}:")

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            # PCA data for this player
            X_pca_train = train_pca[train_mask]
            X_pca_test = test_pca[test_mask]

            # Apply PCA
            n_comp = min(20, X_pca_train.shape[0] - 1, X_pca_train.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca_train_reduced = pca.fit_transform(X_pca_train)
            X_pca_test_reduced = pca.transform(X_pca_test)

            # Hybrid features
            X_hybrid_train = train_hybrid[train_mask]
            X_hybrid_test = test_hybrid[test_mask]

            # Combine
            X_train = np.hstack([X_pca_train_reduced, X_hybrid_train])
            X_test = np.hstack([X_pca_test_reduced, X_hybrid_test])

            y_train = train_targets[train_mask, t_idx]
            player_indices = np.where(train_mask)[0]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 5-fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for fold_train_idx, fold_val_idx in kf.split(X_train_scaled):
                X_tr = X_train_scaled[fold_train_idx]
                X_val = X_train_scaled[fold_val_idx]
                y_tr = y_train[fold_train_idx]

                # Use Ridge for stability
                model = Ridge(alpha=50)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[fold_val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  CV: {scaled_mse:.6f}")

    # Total CV
    print("\nFinal CV:")
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return oof_preds, predictions, cv_score


def main():
    print("=" * 80)
    print("HYBRID BIOMECHANICAL + PCA MODEL")
    print("=" * 80)

    data = load_and_extract_features()
    oof, test_preds, cv_score = train_hybrid_model(data)

    # Create submission
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.startswith('submission_')]
    next_num = max(nums) + 1 if nums else 1

    scaled_preds = np.zeros_like(test_preds)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            test_preds[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': data["test_ids"],
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: hybrid_biomech_pca")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")
    print(f"  depth_mean: {depth_mean:.4f}")
    print(f"  File: {filepath}")

    # Check correlation with Sub 113
    sub113 = pd.read_csv(SUBMISSION_DIR / "submission_113.csv")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        r = np.corrcoef(sub113[col], submission[col])[0, 1]
        print(f"  {col} vs Sub113: r={r:.4f}")

    return filepath


if __name__ == "__main__":
    main()
