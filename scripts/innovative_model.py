"""
Innovative Model - Pushing the Limits

Key innovations:
1. Player-specific optimal features (from physics research)
2. Wavelet features for multi-resolution analysis
3. Joint coordination cross-correlation features
4. Movement signature features (unique per player)
5. Shooting consistency metrics
6. Target-specific frame attention

Goal: Create a diverse model with good profile for blending with Sub 113.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import pywt

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


def compute_acceleration(signal):
    return np.gradient(compute_velocity(signal))


def compute_jerk(signal):
    return np.gradient(compute_acceleration(signal))


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


# =====================================================
# PLAYER-SPECIFIC OPTIMAL FEATURES (from research)
# =====================================================

PLAYER_OPTIMAL_FEATURES = {
    # Depth features (strongest signal)
    "depth": {
        1: {
            "frames": [(80, 120), (145, 165)],
            "joints": ["right_elbow_z", "right_wrist_z", "right_shoulder_z"],
            "type": "velocity"
        },
        2: {
            "frames": [(115, 125)],
            "joints": ["right_shoulder_z", "left_elbow_z", "right_hip_z"],
            "type": "position"
        },
        3: {
            "frames": [(115, 125)],
            "joints": ["right_shoulder_z", "left_elbow_z", "right_hip_z"],
            "type": "position"
        },
        4: {
            "frames": [(115, 125)],
            "joints": ["right_shoulder_z", "left_elbow_z", "right_hip_z"],
            "type": "position"
        },
        5: {
            "frames": [(150, 160)],
            "joints": ["right_hip_z", "left_hip_z", "left_shoulder_z"],
            "type": "velocity"
        },
    },
    # Angle features (moderate signal)
    "angle": {
        1: {
            "frames": [(110, 120)],
            "joints": ["left_elbow_z", "left_wrist_z", "right_elbow_z"],
            "type": "velocity"
        },
        2: {
            "frames": [(75, 85)],
            "joints": ["right_knee_z", "left_knee_z"],
            "type": "position"
        },
        3: {
            "frames": [(180, 220)],
            "joints": ["right_ankle_z", "left_ankle_z"],
            "type": "velocity"
        },
        4: {
            "frames": [(170, 180)],
            "joints": ["right_elbow_z", "right_shoulder_z", "left_elbow_z"],
            "type": "velocity"
        },
        5: {
            "frames": [(145, 165)],
            "joints": ["right_wrist_z", "right_elbow_z"],
            "type": "velocity"
        },
    },
    # Left-right features (weakest signal - use general)
    "left_right": {
        1: {"frames": [(145, 175)], "joints": ["right_wrist_x", "right_elbow_x"], "type": "velocity"},
        2: {"frames": [(145, 175)], "joints": ["right_wrist_x", "right_elbow_x"], "type": "velocity"},
        3: {"frames": [(145, 175)], "joints": ["right_wrist_x", "right_elbow_x"], "type": "velocity"},
        4: {"frames": [(145, 175)], "joints": ["right_wrist_x", "right_elbow_x"], "type": "velocity"},
        5: {"frames": [(145, 175)], "joints": ["right_wrist_x", "right_elbow_x"], "type": "velocity"},
    },
}


def extract_wavelet_features(signal, prefix, wavelet='sym4', level=3):
    """Extract multi-resolution wavelet features."""
    features = {}
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        for i, c in enumerate(coeffs):
            if len(c) > 0:
                features[f'{prefix}_wavelet_energy_L{i}'] = np.sum(c**2)
                features[f'{prefix}_wavelet_std_L{i}'] = np.std(c)
                if len(c) > 3:
                    features[f'{prefix}_wavelet_kurtosis_L{i}'] = kurtosis(c)
    except:
        pass

    return features


def extract_cross_correlation_features(vel1, vel2, prefix):
    """Extract coordination features between two joint velocities."""
    features = {}
    vel1 = np.nan_to_num(vel1, nan=0.0, posinf=0.0, neginf=0.0)
    vel2 = np.nan_to_num(vel2, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        correlation = np.correlate(vel1 - vel1.mean(), vel2 - vel2.mean(), mode='full')
        if len(correlation) > 0:
            max_idx = np.argmax(np.abs(correlation))
            features[f'{prefix}_xcorr_lag'] = max_idx - len(vel1) + 1
            features[f'{prefix}_xcorr_max'] = np.max(np.abs(correlation)) / (np.std(vel1) * np.std(vel2) * len(vel1) + 1e-10)
    except:
        pass

    return features


def compute_joint_angle(p1, p2, p3, frame):
    """Compute angle at p2 between vectors p1->p2 and p2->p3."""
    v1 = p1[frame] - p2[frame]
    v2 = p3[frame] - p2[frame]

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle) * 180 / np.pi


def extract_shooting_consistency_features(series, prefix=""):
    """Extract features related to shooting consistency (research: velocity SD is key)."""
    features = {}

    # Wrist velocity consistency
    right_wrist_z = get_kp(series, "right_wrist_z")
    if right_wrist_z is not None:
        vel = compute_velocity(right_wrist_z)
        vel_mag = np.abs(vel)

        features[f'{prefix}wrist_vel_sd'] = np.std(vel_mag)
        features[f'{prefix}wrist_vel_cv'] = np.std(vel_mag) / (np.mean(vel_mag) + 1e-10)
        features[f'{prefix}wrist_vel_range'] = np.max(vel_mag) - np.min(vel_mag)

        # Jerk (smoothness measure)
        jerk = compute_jerk(right_wrist_z)
        features[f'{prefix}wrist_jerk_mean'] = np.mean(np.abs(jerk))
        features[f'{prefix}wrist_jerk_max'] = np.max(np.abs(jerk))

    return features


def extract_movement_signature(series, prefix=""):
    """Extract unique movement signature features per player."""
    features = {}

    # Kinetic chain timing (research: ankle -> knee -> hip -> shoulder -> elbow -> wrist)
    joints = [
        ("right_ankle_z", "ankle"),
        ("right_knee_z", "knee"),
        ("right_hip_z", "hip"),
        ("right_shoulder_z", "shoulder"),
        ("right_elbow_z", "elbow"),
        ("right_wrist_z", "wrist"),
    ]

    peak_times = {}
    peak_velocities = {}

    for joint_name, short_name in joints:
        data = get_kp(series, joint_name)
        if data is not None:
            vel = compute_velocity(data)
            peak_times[short_name] = np.argmax(np.abs(vel))
            peak_velocities[short_name] = np.max(np.abs(vel))
            features[f'{prefix}{short_name}_peak_time'] = peak_times[short_name] / 240.0
            features[f'{prefix}{short_name}_peak_vel'] = peak_velocities[short_name]

    # Timing lags (proximal-to-distal sequencing)
    if len(peak_times) == 6:
        features[f'{prefix}hip_shoulder_lag'] = peak_times["shoulder"] - peak_times["hip"]
        features[f'{prefix}shoulder_elbow_lag'] = peak_times["elbow"] - peak_times["shoulder"]
        features[f'{prefix}elbow_wrist_lag'] = peak_times["wrist"] - peak_times["elbow"]
        features[f'{prefix}total_chain_time'] = peak_times["wrist"] - peak_times["ankle"]

        # Velocity amplification (distal should be faster)
        if peak_velocities["shoulder"] > 0:
            features[f'{prefix}elbow_shoulder_amp'] = peak_velocities["elbow"] / peak_velocities["shoulder"]
        if peak_velocities["elbow"] > 0:
            features[f'{prefix}wrist_elbow_amp'] = peak_velocities["wrist"] / peak_velocities["elbow"]

    return features


def extract_player_specific_features(series, player_id, target):
    """Extract features optimized for specific player-target combination."""
    features = {}

    if player_id not in PLAYER_OPTIMAL_FEATURES[target]:
        return features

    config = PLAYER_OPTIMAL_FEATURES[target][player_id]

    for frame_range in config["frames"]:
        start, end = frame_range
        frames = list(range(start, min(end, 240)))

        for joint_name in config["joints"]:
            data = get_kp(series, joint_name)
            if data is None:
                continue

            if config["type"] == "velocity":
                vel = compute_velocity(data)
                features[f'{joint_name}_vel_{start}_{end}_mean'] = np.nanmean(vel[frames])
                features[f'{joint_name}_vel_{start}_{end}_max'] = np.nanmax(np.abs(vel[frames]))
                features[f'{joint_name}_vel_{start}_{end}_std'] = np.nanstd(vel[frames])
            else:  # position
                features[f'{joint_name}_pos_{start}_{end}_mean'] = np.nanmean(data[frames])
                features[f'{joint_name}_pos_{start}_{end}_std'] = np.nanstd(data[frames])
                features[f'{joint_name}_pos_f{start}'] = data[start] if start < 240 else np.nan

    return features


def extract_innovative_features(series, participant_id):
    """Extract all innovative features."""
    features = {}

    # Release window
    release_frames = list(range(145, 165))

    # =====================================================
    # 1. WAVELET FEATURES (multi-resolution)
    # =====================================================
    key_joints = ["right_wrist_z", "right_elbow_z", "right_shoulder_z"]
    for joint in key_joints:
        data = get_kp(series, joint)
        if data is not None:
            features.update(extract_wavelet_features(data, joint))

    # =====================================================
    # 2. JOINT COORDINATION FEATURES
    # =====================================================
    right_elbow_z = get_kp(series, "right_elbow_z")
    right_wrist_z = get_kp(series, "right_wrist_z")
    right_shoulder_z = get_kp(series, "right_shoulder_z")

    if right_elbow_z is not None and right_wrist_z is not None:
        elbow_vel = compute_velocity(right_elbow_z)
        wrist_vel = compute_velocity(right_wrist_z)
        features.update(extract_cross_correlation_features(elbow_vel, wrist_vel, "elbow_wrist"))

        # Coordination in release window
        if len(elbow_vel) > 175 and len(wrist_vel) > 175:
            corr = np.corrcoef(elbow_vel[130:175], wrist_vel[130:175])[0, 1]
            features["elbow_wrist_release_corr"] = corr if not np.isnan(corr) else 0

    if right_shoulder_z is not None and right_elbow_z is not None:
        shoulder_vel = compute_velocity(right_shoulder_z)
        elbow_vel = compute_velocity(right_elbow_z)
        features.update(extract_cross_correlation_features(shoulder_vel, elbow_vel, "shoulder_elbow"))

    # =====================================================
    # 3. SHOOTING CONSISTENCY FEATURES
    # =====================================================
    features.update(extract_shooting_consistency_features(series))

    # =====================================================
    # 4. MOVEMENT SIGNATURE FEATURES
    # =====================================================
    features.update(extract_movement_signature(series))

    # =====================================================
    # 5. JOINT ANGLES AT KEY FRAMES
    # =====================================================
    # Get 3D positions for angle computation
    right_shoulder = np.zeros((240, 3))
    right_elbow = np.zeros((240, 3))
    right_wrist = np.zeros((240, 3))
    right_hip = np.zeros((240, 3))
    right_knee = np.zeros((240, 3))
    right_ankle = np.zeros((240, 3))

    for axis_idx, axis in enumerate(['x', 'y', 'z']):
        data = get_kp(series, f"right_shoulder_{axis}")
        if data is not None:
            right_shoulder[:, axis_idx] = data
        data = get_kp(series, f"right_elbow_{axis}")
        if data is not None:
            right_elbow[:, axis_idx] = data
        data = get_kp(series, f"right_wrist_{axis}")
        if data is not None:
            right_wrist[:, axis_idx] = data
        data = get_kp(series, f"right_hip_{axis}")
        if data is not None:
            right_hip[:, axis_idx] = data
        data = get_kp(series, f"right_knee_{axis}")
        if data is not None:
            right_knee[:, axis_idx] = data
        data = get_kp(series, f"right_ankle_{axis}")
        if data is not None:
            right_ankle[:, axis_idx] = data

    # Elbow angle at key frames
    for f in [80, 120, 150, 160]:
        if f < 240:
            features[f'elbow_angle_f{f}'] = compute_joint_angle(right_shoulder, right_elbow, right_wrist, f)
            features[f'knee_angle_f{f}'] = compute_joint_angle(right_hip, right_knee, right_ankle, f)

    # Elbow angle velocity
    elbow_angles = [compute_joint_angle(right_shoulder, right_elbow, right_wrist, f) for f in range(240)]
    elbow_angles = np.array(elbow_angles)
    elbow_angle_vel = np.gradient(smooth_signal(elbow_angles))
    features['elbow_angle_vel_release'] = np.nanmean(elbow_angle_vel[release_frames])
    features['elbow_angle_vel_max'] = np.nanmax(np.abs(elbow_angle_vel))

    # =====================================================
    # 6. TARGET-SPECIFIC PLAYER FEATURES
    # =====================================================
    for target in TARGETS:
        target_features = extract_player_specific_features(series, participant_id, target)
        for k, v in target_features.items():
            features[f'{target}_{k}'] = v

    # =====================================================
    # 7. CENTER OF MASS FEATURES
    # =====================================================
    mid_hip_z = get_kp(series, "mid_hip_z")
    left_shoulder_z = get_kp(series, "left_shoulder_z")
    right_shoulder_z = get_kp(series, "right_shoulder_z")

    if all(x is not None for x in [mid_hip_z, left_shoulder_z, right_shoulder_z]):
        mid_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2
        com_z = 0.6 * mid_hip_z + 0.4 * mid_shoulder_z

        com_vel = compute_velocity(com_z)
        features['com_vel_release'] = np.nanmean(com_vel[release_frames])
        features['com_vel_max'] = np.nanmax(com_vel)
        features['com_stability'] = np.nanstd(com_z[release_frames])

    # =====================================================
    # 8. STANDARD RELEASE FEATURES (for stability)
    # =====================================================
    key_joints_xyz = [
        "right_wrist", "right_elbow", "right_shoulder",
        "right_knee", "right_hip", "left_wrist"
    ]

    for joint in key_joints_xyz:
        for axis in ["x", "y", "z"]:
            data = get_kp(series, f"{joint}_{axis}")
            if data is not None:
                vel = compute_velocity(data)

                # Position at release
                features[f'{joint}_{axis}_release'] = np.nanmean(data[release_frames])

                # Velocity at release
                features[f'{joint}_{axis}_vel_release'] = np.nanmean(vel[release_frames])

                # Key frame values
                for f in [150, 155, 160]:
                    if f < 240:
                        features[f'{joint}_{axis}_f{f}'] = data[f]

    # Clean up NaN/Inf
    for k, v in list(features.items()):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0

    return features


def load_and_extract_features():
    """Load data and extract features."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    init_keypoint_mapping(keypoint_cols)

    def process_df(df, desc="Processing"):
        all_features = []
        all_pids = []
        all_ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            features = extract_innovative_features(ts, row['participant_id'])
            all_features.append(features)
            all_pids.append(row['participant_id'])
            all_ids.append(row['id'])

        return pd.DataFrame(all_features), np.array(all_pids), np.array(all_ids)

    print("Processing training data...")
    train_features, train_pids, train_ids = process_df(train_df, "Train")
    train_targets = train_df[["angle", "depth", "left_right"]].values

    print("Processing test data...")
    test_features, test_pids, test_ids = process_df(test_df, "Test")

    return {
        "train_features": train_features,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_features": test_features,
        "test_pids": test_pids,
        "test_ids": test_ids,
    }


def train_innovative_model(data):
    """Train innovative model with player-specific features."""
    print("\n" + "=" * 60)
    print("INNOVATIVE MODEL - Player-Specific Features")
    print("=" * 60)

    X_train = np.nan_to_num(data["train_features"].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = data["train_targets"]
    train_pids = data["train_pids"]

    X_test = np.nan_to_num(data["test_features"].values, nan=0.0, posinf=0.0, neginf=0.0)
    test_pids = data["test_pids"]

    print(f"Features: {X_train.shape[1]}")

    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(X_test), 3))
    oof_preds = np.zeros_like(y_train)

    for t_idx, target in enumerate(TARGETS):
        print(f"\n{target}:")

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_tr = X_train[train_mask]
            X_te = X_test[test_mask]
            y_tr = y_train[train_mask, t_idx]
            player_indices = np.where(train_mask)[0]

            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for fold_train_idx, fold_val_idx in kf.split(X_tr_scaled):
                X_fold_tr = X_tr_scaled[fold_train_idx]
                X_fold_val = X_tr_scaled[fold_val_idx]
                y_fold_tr = y_tr[fold_train_idx]

                # Use Ridge with moderate regularization
                model = Ridge(alpha=100)
                model.fit(X_fold_tr, y_fold_tr)

                oof_preds[player_indices[fold_val_idx], t_idx] = model.predict(X_fold_val)
                fold_test_preds.append(model.predict(X_te_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        mse = np.mean((oof_preds[:, t_idx] - y_train[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  CV: {scaled_mse:.6f}")

    # Total CV
    print("\nFinal CV:")
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - y_train[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return oof_preds, predictions, cv_score


def main():
    print("=" * 80)
    print("INNOVATIVE MODEL - Pushing the Limits")
    print("=" * 80)
    print("\nInnovations included:")
    print("  1. Player-specific optimal features (from physics research)")
    print("  2. Wavelet features for multi-resolution analysis")
    print("  3. Joint coordination cross-correlation features")
    print("  4. Movement signature features (kinetic chain timing)")
    print("  5. Shooting consistency metrics (velocity SD)")
    print("  6. Joint angle features")

    data = load_and_extract_features()
    oof, test_preds, cv_score = train_innovative_model(data)

    # Create submission
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        try:
            # Handle both submission_N.csv and submissionN.csv formats
            name = f.stem
            if name.startswith('submission_'):
                nums.append(int(name.split('_')[1]))
            elif name.startswith('submission'):
                nums.append(int(name.replace('submission', '')))
        except:
            pass
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
    print(f"SUBMISSION {next_num}: innovative_model")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f} (target: <0.14)")
    print(f"  depth_mean: {depth_mean:.4f} (target: ~0.505)")
    print(f"  File: {filepath}")

    # Check correlation with Sub 113 if it exists
    sub113_path = SUBMISSION_DIR / "submission_113.csv"
    if sub113_path.exists():
        sub113 = pd.read_csv(sub113_path)
        print("\nCorrelation with Sub 113:")
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            r = np.corrcoef(sub113[col], submission[col])[0, 1]
            print(f"  {col}: r={r:.4f}")

    # Predicted LB based on profile
    # From research: LB ~ 0.0065 + 0.0006 * (depth_mean - 0.505) + 0.002 * (angle_std - 0.14)
    predicted_lb = 0.008 + 5 * abs(depth_mean - 0.505) + 1 * (angle_std - 0.14)
    print(f"\nPredicted LB (rough): {predicted_lb:.6f}")

    return filepath


if __name__ == "__main__":
    main()
