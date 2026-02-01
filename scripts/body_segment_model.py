"""
Body Segment Model - Focus on Angles and Efficiency

Key innovations:
1. Body segment angles (elbow, shoulder, knee, trunk)
2. Movement symmetry (left vs right side)
3. Trajectory smoothness (jerk metrics)
4. Movement efficiency (path length ratios)
5. Temporal consistency (velocity profiles)

Goal: Capture fundamentally different aspects of shooting motion.
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
from tqdm import tqdm

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


def compute_jerk(signal):
    """Compute jerk (rate of acceleration change) - measure of smoothness."""
    vel = np.gradient(smooth_signal(signal))
    acc = np.gradient(vel)
    jerk = np.gradient(acc)
    return jerk


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


def get_3d_position(series, joint_name, frame=None):
    """Get 3D position of a joint."""
    x = get_kp(series, f"{joint_name}_x", frame)
    y = get_kp(series, f"{joint_name}_y", frame)
    z = get_kp(series, f"{joint_name}_z", frame)

    if x is None or y is None or z is None:
        if frame is not None:
            return np.array([0, 0, 0])
        return np.zeros((240, 3))

    if frame is not None:
        return np.array([x, y, z])
    return np.column_stack([x, y, z])


def compute_angle_between_vectors(v1, v2):
    """Compute angle between two 3D vectors."""
    v1 = np.nan_to_num(v1, nan=0.0)
    v2 = np.nan_to_num(v2, nan=0.0)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle) * 180 / np.pi


def compute_joint_angle_series(p1_series, p2_series, p3_series):
    """Compute joint angle at p2 for all frames."""
    angles = np.zeros(240)
    for f in range(240):
        v1 = p1_series[f] - p2_series[f]
        v2 = p3_series[f] - p2_series[f]
        angles[f] = compute_angle_between_vectors(v1, v2)
    return angles


def extract_body_segment_features(series, participant_id):
    """Extract body segment and efficiency features."""
    features = {}

    release_frames = list(range(145, 165))
    load_frames = list(range(80, 120))
    propulsion_frames = list(range(120, 160))

    # =====================================================
    # 1. JOINT ANGLES OVER TIME
    # =====================================================

    # Get 3D positions
    right_shoulder = get_3d_position(series, "right_shoulder")
    right_elbow = get_3d_position(series, "right_elbow")
    right_wrist = get_3d_position(series, "right_wrist")
    right_hip = get_3d_position(series, "right_hip")
    right_knee = get_3d_position(series, "right_knee")
    right_ankle = get_3d_position(series, "right_ankle")
    left_shoulder = get_3d_position(series, "left_shoulder")
    left_elbow = get_3d_position(series, "left_elbow")
    left_wrist = get_3d_position(series, "left_wrist")
    mid_hip = get_3d_position(series, "mid_hip")
    neck = get_3d_position(series, "neck")

    # Elbow angle (shoulder-elbow-wrist)
    elbow_angles = compute_joint_angle_series(right_shoulder, right_elbow, right_wrist)
    features['elbow_angle_release'] = np.nanmean(elbow_angles[release_frames])
    features['elbow_angle_load'] = np.nanmean(elbow_angles[load_frames])
    features['elbow_angle_range'] = np.nanmax(elbow_angles) - np.nanmin(elbow_angles)
    features['elbow_angle_max'] = np.nanmax(elbow_angles)
    features['elbow_angle_min'] = np.nanmin(elbow_angles)

    # Elbow angular velocity
    elbow_angle_vel = np.gradient(smooth_signal(elbow_angles))
    features['elbow_angle_vel_release'] = np.nanmean(elbow_angle_vel[release_frames])
    features['elbow_angle_vel_max'] = np.nanmax(elbow_angle_vel)
    features['elbow_angle_vel_max_time'] = np.argmax(elbow_angle_vel) / 240.0

    # Shoulder angle (hip-shoulder-elbow)
    shoulder_angles = compute_joint_angle_series(right_hip, right_shoulder, right_elbow)
    features['shoulder_angle_release'] = np.nanmean(shoulder_angles[release_frames])
    features['shoulder_angle_range'] = np.nanmax(shoulder_angles) - np.nanmin(shoulder_angles)

    # Knee angle (hip-knee-ankle)
    knee_angles = compute_joint_angle_series(right_hip, right_knee, right_ankle)
    features['knee_angle_load'] = np.nanmean(knee_angles[load_frames])
    features['knee_angle_release'] = np.nanmean(knee_angles[release_frames])
    features['knee_angle_range'] = np.nanmax(knee_angles) - np.nanmin(knee_angles)

    # Trunk lean angle (mid_hip to neck vs vertical)
    trunk_vectors = neck - mid_hip
    vertical = np.array([0, 0, 1])
    trunk_angles = np.array([compute_angle_between_vectors(trunk_vectors[f], vertical)
                            for f in range(240)])
    features['trunk_angle_release'] = np.nanmean(trunk_angles[release_frames])
    features['trunk_angle_max'] = np.nanmax(trunk_angles)

    # =====================================================
    # 2. MOVEMENT SYMMETRY (left vs right)
    # =====================================================

    # Shoulder symmetry
    left_shoulder_z = get_kp(series, "left_shoulder_z")
    right_shoulder_z = get_kp(series, "right_shoulder_z")
    if left_shoulder_z is not None and right_shoulder_z is not None:
        shoulder_diff = right_shoulder_z - left_shoulder_z
        features['shoulder_symmetry_release'] = np.nanmean(shoulder_diff[release_frames])
        features['shoulder_symmetry_std'] = np.nanstd(shoulder_diff)

    # Hip symmetry
    left_hip_z = get_kp(series, "left_hip_z")
    right_hip_z = get_kp(series, "right_hip_z")
    if left_hip_z is not None and right_hip_z is not None:
        hip_diff = right_hip_z - left_hip_z
        features['hip_symmetry_release'] = np.nanmean(hip_diff[release_frames])
        features['hip_symmetry_std'] = np.nanstd(hip_diff)

    # =====================================================
    # 3. MOVEMENT SMOOTHNESS (jerk metrics)
    # =====================================================

    # Wrist jerk (measure of movement smoothness)
    right_wrist_z = get_kp(series, "right_wrist_z")
    if right_wrist_z is not None:
        jerk = compute_jerk(right_wrist_z)
        features['wrist_jerk_mean'] = np.nanmean(np.abs(jerk))
        features['wrist_jerk_release'] = np.nanmean(np.abs(jerk[release_frames]))
        features['wrist_jerk_max'] = np.nanmax(np.abs(jerk))

    # Elbow jerk
    right_elbow_z = get_kp(series, "right_elbow_z")
    if right_elbow_z is not None:
        jerk = compute_jerk(right_elbow_z)
        features['elbow_jerk_mean'] = np.nanmean(np.abs(jerk))
        features['elbow_jerk_release'] = np.nanmean(np.abs(jerk[release_frames]))

    # =====================================================
    # 4. MOVEMENT EFFICIENCY
    # =====================================================

    # Path length ratio (actual path / straight line distance)
    if right_wrist_z is not None:
        # Total path length
        path_lengths = np.sqrt(np.diff(right_wrist[:, 0])**2 +
                              np.diff(right_wrist[:, 1])**2 +
                              np.diff(right_wrist[:, 2])**2)
        total_path = np.nansum(path_lengths)

        # Straight line distance (start to end of release window)
        start_pos = right_wrist[release_frames[0]]
        end_pos = right_wrist[release_frames[-1]]
        straight_dist = np.linalg.norm(end_pos - start_pos)

        if straight_dist > 0.01:
            features['wrist_path_efficiency'] = straight_dist / (total_path + 1e-6)
        else:
            features['wrist_path_efficiency'] = 0

    # Velocity profile consistency
    if right_wrist_z is not None:
        vel = compute_velocity(right_wrist_z)
        features['velocity_consistency'] = np.nanmean(vel) / (np.nanstd(vel) + 1e-6)

    # =====================================================
    # 5. TIMING FEATURES
    # =====================================================

    # Peak velocity timing for different joints
    joints = [("right_wrist_z", "wrist"), ("right_elbow_z", "elbow"),
              ("right_shoulder_z", "shoulder"), ("right_knee_z", "knee")]

    for joint_name, short_name in joints:
        data = get_kp(series, joint_name)
        if data is not None:
            vel = compute_velocity(data)
            features[f'{short_name}_peak_vel_time'] = np.argmax(np.abs(vel)) / 240.0
            features[f'{short_name}_peak_vel'] = np.nanmax(np.abs(vel))

    # Timing sequence (proper kinetic chain should go proximal to distal)
    if all(k in features for k in ['knee_peak_vel_time', 'shoulder_peak_vel_time',
                                    'elbow_peak_vel_time', 'wrist_peak_vel_time']):
        features['kinetic_chain_order'] = (
            features['knee_peak_vel_time'] +
            features['shoulder_peak_vel_time'] * 2 +
            features['elbow_peak_vel_time'] * 3 +
            features['wrist_peak_vel_time'] * 4
        )

    # =====================================================
    # 6. RELEASE WINDOW FEATURES
    # =====================================================

    # Key joint positions at release
    key_joints = ["right_wrist", "right_elbow", "right_shoulder", "right_knee"]
    for joint in key_joints:
        for axis in ["x", "y", "z"]:
            data = get_kp(series, f"{joint}_{axis}")
            if data is not None:
                features[f'{joint}_{axis}_release'] = np.nanmean(data[release_frames])
                vel = compute_velocity(data)
                features[f'{joint}_{axis}_vel_release'] = np.nanmean(vel[release_frames])

    # =====================================================
    # 7. BODY ALIGNMENT
    # =====================================================

    # Elbow lateral deviation (from vertical line through shoulder)
    right_shoulder_x = get_kp(series, "right_shoulder_x")
    right_elbow_x = get_kp(series, "right_elbow_x")
    if right_shoulder_x is not None and right_elbow_x is not None:
        lateral_dev = right_elbow_x - right_shoulder_x
        features['elbow_lateral_dev_release'] = np.nanmean(lateral_dev[release_frames])
        features['elbow_lateral_dev_std'] = np.nanstd(lateral_dev)

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

            features = extract_body_segment_features(ts, row['participant_id'])
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


def train_body_segment_model(data):
    """Train model with body segment features."""
    print("\n" + "=" * 60)
    print("BODY SEGMENT MODEL")
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
    print("BODY SEGMENT MODEL - Angles and Efficiency")
    print("=" * 80)
    print("\nFeatures:")
    print("  1. Joint angles (elbow, shoulder, knee, trunk)")
    print("  2. Movement symmetry (left vs right)")
    print("  3. Movement smoothness (jerk)")
    print("  4. Movement efficiency (path ratios)")
    print("  5. Kinetic chain timing")

    data = load_and_extract_features()
    oof, test_preds, cv_score = train_body_segment_model(data)

    # Create submission
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        try:
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
    print(f"SUBMISSION {next_num}: body_segment_model")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f} (target: <0.14)")
    print(f"  depth_mean: {depth_mean:.4f} (target: ~0.505)")
    print(f"  File: {filepath}")

    # Check correlation with Sub 113
    sub113_path = SUBMISSION_DIR / "submission_113.csv"
    if sub113_path.exists():
        sub113 = pd.read_csv(sub113_path)
        print("\nCorrelation with Sub 113:")
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            r = np.corrcoef(sub113[col], submission[col])[0, 1]
            print(f"  {col}: r={r:.4f}")

    return filepath


if __name__ == "__main__":
    main()
