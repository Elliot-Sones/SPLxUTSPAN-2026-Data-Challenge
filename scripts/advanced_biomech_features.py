"""
Advanced Biomechanical Features - Based on Latest Research

Key insights from research:
1. Elbow-wrist coordination in last 0.01 seconds is critical
2. Less ROM (more controlled movement) = better shooting
3. Proximal-to-distal energy sequencing matters
4. Velocity features 10x more important than positions

New features:
- Center of mass dynamics
- Angular momentum and rotational dynamics
- Kinetic chain (sequential power transfer)
- Joint angle velocities and accelerations
- Elbow-wrist coordination (phase coupling)
- Guide hand dynamics
- Symmetry indices
- Phase duration measurements
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

# Keypoint mapping
KEYPOINT_MAP = {}


def init_keypoint_mapping(keypoint_cols):
    global KEYPOINT_MAP
    for i, col in enumerate(keypoint_cols):
        KEYPOINT_MAP[col] = i


def get_kp(series, name, frame=None):
    """Get keypoint data."""
    if name not in KEYPOINT_MAP:
        return None
    idx = KEYPOINT_MAP[name]
    if frame is not None:
        return series[frame, idx]
    return series[:, idx]


def smooth_signal(signal, window=5):
    """Apply Savitzky-Golay filter for smoothing."""
    # Handle NaN values
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < window:
        return signal
    try:
        return savgol_filter(signal, window, 2)
    except:
        return signal


def compute_velocity(signal, smooth=True):
    """Compute velocity (first derivative)."""
    if smooth:
        signal = smooth_signal(signal)
    return np.gradient(signal)


def compute_acceleration(signal, smooth=True):
    """Compute acceleration (second derivative)."""
    vel = compute_velocity(signal, smooth)
    return np.gradient(vel)


def compute_jerk(signal, smooth=True):
    """Compute jerk (third derivative)."""
    acc = compute_acceleration(signal, smooth)
    return np.gradient(acc)


def extract_advanced_biomech_features(series, participant_id):
    """
    Extract advanced biomechanical features based on latest research.
    """
    features = {}

    # ===================================================================
    # 1. CENTER OF MASS DYNAMICS
    # ===================================================================
    # Approximate CoM from hip and shoulder midpoints
    mid_hip_x = get_kp(series, "mid_hip_x")
    mid_hip_y = get_kp(series, "mid_hip_y")
    mid_hip_z = get_kp(series, "mid_hip_z")

    left_shoulder_x = get_kp(series, "left_shoulder_x")
    left_shoulder_y = get_kp(series, "left_shoulder_y")
    left_shoulder_z = get_kp(series, "left_shoulder_z")
    right_shoulder_x = get_kp(series, "right_shoulder_x")
    right_shoulder_y = get_kp(series, "right_shoulder_y")
    right_shoulder_z = get_kp(series, "right_shoulder_z")

    if all(x is not None for x in [mid_hip_x, left_shoulder_x, right_shoulder_x]):
        # Shoulder midpoint
        mid_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
        mid_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        mid_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2

        # Approximate CoM (weighted average of hip and shoulder)
        com_x = 0.6 * mid_hip_x + 0.4 * mid_shoulder_x
        com_y = 0.6 * mid_hip_y + 0.4 * mid_shoulder_y
        com_z = 0.6 * mid_hip_z + 0.4 * mid_shoulder_z

        # CoM velocity
        com_vx = compute_velocity(com_x)
        com_vy = compute_velocity(com_y)
        com_vz = compute_velocity(com_z)
        com_speed = np.sqrt(com_vx**2 + com_vy**2 + com_vz**2)

        # CoM features
        features["com_z_mean"] = np.nanmean(com_z)
        features["com_z_range"] = np.nanmax(com_z) - np.nanmin(com_z)
        features["com_speed_max"] = np.nanmax(com_speed)
        features["com_speed_release"] = np.nanmean(com_speed[140:170])
        features["com_vy_release"] = np.nanmean(com_vy[140:170])
        features["com_vz_release"] = np.nanmean(com_vz[140:170])

        # CoM stability (variance during setup)
        features["com_stability_setup"] = np.nanstd(com_z[20:60])
        features["com_stability_release"] = np.nanstd(com_z[140:170])

    # ===================================================================
    # 2. ANGULAR MOMENTUM / ROTATIONAL DYNAMICS
    # ===================================================================
    right_shoulder_z = get_kp(series, "right_shoulder_z")
    right_elbow_z = get_kp(series, "right_elbow_z")
    right_wrist_z = get_kp(series, "right_wrist_z")

    if all(x is not None for x in [right_shoulder_z, right_elbow_z, right_wrist_z]):
        # Upper arm rotation (shoulder to elbow)
        upper_arm_angle = np.arctan2(
            right_elbow_z - right_shoulder_z,
            get_kp(series, "right_elbow_y") - get_kp(series, "right_shoulder_y")
        )
        upper_arm_angular_vel = compute_velocity(upper_arm_angle)

        # Forearm rotation (elbow to wrist)
        forearm_angle = np.arctan2(
            right_wrist_z - right_elbow_z,
            get_kp(series, "right_wrist_y") - get_kp(series, "right_elbow_y")
        )
        forearm_angular_vel = compute_velocity(forearm_angle)

        features["upper_arm_angular_vel_max"] = np.nanmax(np.abs(upper_arm_angular_vel))
        features["upper_arm_angular_vel_release"] = np.nanmean(upper_arm_angular_vel[150:170])
        features["forearm_angular_vel_max"] = np.nanmax(np.abs(forearm_angular_vel))
        features["forearm_angular_vel_release"] = np.nanmean(forearm_angular_vel[150:170])

        # Angular acceleration
        upper_arm_angular_acc = compute_acceleration(upper_arm_angle)
        forearm_angular_acc = compute_acceleration(forearm_angle)
        features["upper_arm_angular_acc_max"] = np.nanmax(np.abs(upper_arm_angular_acc))
        features["forearm_angular_acc_max"] = np.nanmax(np.abs(forearm_angular_acc))

    # ===================================================================
    # 3. KINETIC CHAIN (PROXIMAL-TO-DISTAL SEQUENCING)
    # ===================================================================
    right_hip_z = get_kp(series, "right_hip_z")
    right_knee_z = get_kp(series, "right_knee_z")
    right_ankle_z = get_kp(series, "right_ankle_z")

    if all(x is not None for x in [right_hip_z, right_knee_z, right_shoulder_z, right_elbow_z, right_wrist_z]):
        # Velocity of each joint
        hip_vel = compute_velocity(right_hip_z)
        knee_vel = compute_velocity(right_knee_z)
        shoulder_vel = compute_velocity(right_shoulder_z)
        elbow_vel = compute_velocity(right_elbow_z)
        wrist_vel = compute_velocity(right_wrist_z)

        # Time of peak velocity for each joint
        hip_peak_time = np.argmax(np.abs(hip_vel))
        knee_peak_time = np.argmax(np.abs(knee_vel))
        shoulder_peak_time = np.argmax(np.abs(shoulder_vel))
        elbow_peak_time = np.argmax(np.abs(elbow_vel))
        wrist_peak_time = np.argmax(np.abs(wrist_vel))

        features["kinetic_chain_hip_to_wrist"] = wrist_peak_time - hip_peak_time
        features["kinetic_chain_knee_to_elbow"] = elbow_peak_time - knee_peak_time
        features["kinetic_chain_shoulder_to_wrist"] = wrist_peak_time - shoulder_peak_time
        features["kinetic_chain_elbow_to_wrist"] = wrist_peak_time - elbow_peak_time

        # Velocity ratios (proximal to distal should increase)
        features["vel_ratio_wrist_shoulder"] = np.nanmax(np.abs(wrist_vel)) / (np.nanmax(np.abs(shoulder_vel)) + 1e-6)
        features["vel_ratio_wrist_elbow"] = np.nanmax(np.abs(wrist_vel)) / (np.nanmax(np.abs(elbow_vel)) + 1e-6)
        features["vel_ratio_elbow_shoulder"] = np.nanmax(np.abs(elbow_vel)) / (np.nanmax(np.abs(shoulder_vel)) + 1e-6)

    # ===================================================================
    # 4. ELBOW-WRIST COORDINATION (CRITICAL - from research)
    # ===================================================================
    if right_elbow_z is not None and right_wrist_z is not None:
        elbow_vel = compute_velocity(right_elbow_z)
        wrist_vel = compute_velocity(right_wrist_z)

        # Wrist snap = wrist velocity - elbow velocity (the "flick")
        wrist_snap = wrist_vel - elbow_vel
        features["wrist_snap_max"] = np.nanmax(wrist_snap)
        features["wrist_snap_release"] = np.nanmean(wrist_snap[155:170])
        features["wrist_snap_timing"] = np.argmax(wrist_snap) / 240.0

        # Coordination variability in last frames (research: higher = worse)
        # Last 0.01 seconds at 60fps = ~0.6 frames, use last 3-5 frames
        release_window = slice(165, 175)
        elbow_release_std = np.nanstd(elbow_vel[release_window])
        wrist_release_std = np.nanstd(wrist_vel[release_window])
        features["coordination_variability_release"] = elbow_release_std + wrist_release_std

        # Phase coupling (correlation between elbow and wrist velocities)
        corr = np.corrcoef(elbow_vel[100:180], wrist_vel[100:180])[0, 1]
        features["elbow_wrist_coupling"] = corr if not np.isnan(corr) else 0

        # Relative phase (timing offset between elbow and wrist peak)
        elbow_peak = np.argmax(np.abs(elbow_vel[100:180]))
        wrist_peak = np.argmax(np.abs(wrist_vel[100:180]))
        features["elbow_wrist_phase_offset"] = wrist_peak - elbow_peak

    # ===================================================================
    # 5. JOINT ANGLE VELOCITIES AND ACCELERATIONS
    # ===================================================================
    # Elbow angle (shoulder-elbow-wrist)
    if all(x is not None for x in [right_shoulder_z, right_elbow_z, right_wrist_z]):
        # Calculate elbow angle in 3D
        shoulder_y = get_kp(series, "right_shoulder_y")
        elbow_y = get_kp(series, "right_elbow_y")
        wrist_y = get_kp(series, "right_wrist_y")

        # Vectors
        v1_y = shoulder_y - elbow_y
        v1_z = right_shoulder_z - right_elbow_z
        v2_y = wrist_y - elbow_y
        v2_z = right_wrist_z - right_elbow_z

        # Angle between vectors
        dot = v1_y * v2_y + v1_z * v2_z
        mag1 = np.sqrt(v1_y**2 + v1_z**2)
        mag2 = np.sqrt(v2_y**2 + v2_z**2)
        elbow_angle = np.arccos(np.clip(dot / (mag1 * mag2 + 1e-6), -1, 1))

        elbow_angle_vel = compute_velocity(elbow_angle)
        elbow_angle_acc = compute_acceleration(elbow_angle)

        features["elbow_angle_mean"] = np.nanmean(elbow_angle)
        features["elbow_angle_range"] = np.nanmax(elbow_angle) - np.nanmin(elbow_angle)
        features["elbow_angle_vel_max"] = np.nanmax(np.abs(elbow_angle_vel))
        features["elbow_angle_vel_release"] = np.nanmean(elbow_angle_vel[155:170])
        features["elbow_angle_acc_max"] = np.nanmax(np.abs(elbow_angle_acc))

        # ROM (Range of Motion) - less is better per research
        features["elbow_rom"] = np.nanmax(elbow_angle) - np.nanmin(elbow_angle)

    # ===================================================================
    # 6. GUIDE HAND (LEFT WRIST) DYNAMICS
    # ===================================================================
    left_wrist_x = get_kp(series, "left_wrist_x")
    left_wrist_y = get_kp(series, "left_wrist_y")
    left_wrist_z = get_kp(series, "left_wrist_z")

    if all(x is not None for x in [left_wrist_x, left_wrist_y, left_wrist_z, right_wrist_z]):
        # Guide hand velocity
        left_vx = compute_velocity(left_wrist_x)
        left_vy = compute_velocity(left_wrist_y)
        left_vz = compute_velocity(left_wrist_z)
        left_speed = np.sqrt(left_vx**2 + left_vy**2 + left_vz**2)

        features["guide_hand_speed_release"] = np.nanmean(left_speed[150:170])
        features["guide_hand_vz_release"] = np.nanmean(left_vz[150:170])

        # Guide hand separation (distance between wrists)
        right_wrist_x = get_kp(series, "right_wrist_x")
        right_wrist_y = get_kp(series, "right_wrist_y")

        wrist_distance = np.sqrt(
            (right_wrist_x - left_wrist_x)**2 +
            (right_wrist_y - left_wrist_y)**2 +
            (right_wrist_z - left_wrist_z)**2
        )

        features["wrist_separation_release"] = np.nanmean(wrist_distance[155:170])
        features["wrist_separation_rate"] = np.nanmean(compute_velocity(wrist_distance)[150:170])

        # Guide hand decoupling timing (when distance starts increasing rapidly)
        sep_vel = compute_velocity(wrist_distance)
        sep_peak = np.argmax(sep_vel[100:180])
        features["guide_hand_decouple_timing"] = (100 + sep_peak) / 240.0

    # ===================================================================
    # 7. SYMMETRY INDICES
    # ===================================================================
    left_hip_z = get_kp(series, "left_hip_z")
    right_hip_z = get_kp(series, "right_hip_z")

    if left_shoulder_z is not None and right_shoulder_z is not None:
        # Shoulder symmetry
        shoulder_diff = right_shoulder_z - left_shoulder_z
        features["shoulder_symmetry_mean"] = np.nanmean(shoulder_diff)
        features["shoulder_symmetry_std"] = np.nanstd(shoulder_diff)
        features["shoulder_symmetry_release"] = np.nanmean(shoulder_diff[150:170])

    if left_hip_z is not None and right_hip_z is not None:
        # Hip symmetry
        hip_diff = right_hip_z - left_hip_z
        features["hip_symmetry_mean"] = np.nanmean(hip_diff)
        features["hip_symmetry_std"] = np.nanstd(hip_diff)
        features["hip_symmetry_release"] = np.nanmean(hip_diff[150:170])

    # Body twist (shoulder vs hip alignment)
    if all(x is not None for x in [left_shoulder_z, right_shoulder_z, left_hip_z, right_hip_z]):
        shoulder_align = right_shoulder_z - left_shoulder_z
        hip_align = right_hip_z - left_hip_z
        body_twist = shoulder_align - hip_align
        features["body_twist_mean"] = np.nanmean(body_twist)
        features["body_twist_release"] = np.nanmean(body_twist[150:170])

    # ===================================================================
    # 8. PHASE DURATION MEASUREMENTS
    # ===================================================================
    if right_wrist_z is not None:
        wrist_vel = compute_velocity(right_wrist_z)
        wrist_speed = np.abs(wrist_vel)

        # Find key phase transitions based on velocity
        threshold = np.nanmax(wrist_speed) * 0.3
        above_threshold = wrist_speed > threshold

        # Find first time above threshold (start of propulsion)
        propulsion_start = np.argmax(above_threshold)

        # Find peak velocity (release moment)
        peak_vel_frame = np.argmax(wrist_speed)

        features["setup_duration"] = propulsion_start / 240.0
        features["propulsion_duration"] = (peak_vel_frame - propulsion_start) / 240.0
        features["release_frame_normalized"] = peak_vel_frame / 240.0

    # ===================================================================
    # 9. MOTION SMOOTHNESS (JERK METRICS)
    # ===================================================================
    if right_wrist_z is not None:
        wrist_jerk = compute_jerk(right_wrist_z)

        # Normalized jerk (smoothness metric)
        movement_time = 240 / 60.0  # seconds
        movement_distance = np.nanmax(right_wrist_z) - np.nanmin(right_wrist_z)

        if movement_distance > 0:
            # Dimensionless jerk (lower = smoother)
            mean_jerk_sq = np.nanmean(wrist_jerk**2)
            normalized_jerk = np.sqrt(mean_jerk_sq) * movement_time**3 / movement_distance
            features["normalized_jerk"] = normalized_jerk

        features["jerk_release_mean"] = np.nanmean(np.abs(wrist_jerk[150:170]))
        features["jerk_release_std"] = np.nanstd(wrist_jerk[150:170])

    # ===================================================================
    # 10. ADVANCED VELOCITY PROFILES
    # ===================================================================
    if right_wrist_z is not None:
        wrist_vel = compute_velocity(right_wrist_z)
        wrist_acc = compute_acceleration(right_wrist_z)

        # Velocity consistency (std during phases)
        features["vel_consistency_setup"] = np.nanstd(wrist_vel[20:60])
        features["vel_consistency_load"] = np.nanstd(wrist_vel[60:120])
        features["vel_consistency_release"] = np.nanstd(wrist_vel[140:180])

        # Acceleration profile
        features["acc_peak_value"] = np.nanmax(wrist_acc)
        features["acc_peak_timing"] = np.argmax(wrist_acc) / 240.0

        # Deceleration after release
        peak_frame = np.argmax(wrist_vel)
        if peak_frame < 220:
            decel = wrist_acc[peak_frame:min(peak_frame+20, 240)]
            features["deceleration_rate"] = np.nanmean(decel)

    # Replace NaN with 0
    for k, v in features.items():
        if np.isnan(v) if isinstance(v, float) else False:
            features[k] = 0.0

    return features


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


def load_and_extract_features():
    """Load data and extract advanced biomechanical features."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    init_keypoint_mapping(keypoint_cols)

    def extract_all_features(df, desc="Processing"):
        all_features = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            features = extract_advanced_biomech_features(ts, row['participant_id'])
            all_features.append(features)

        return pd.DataFrame(all_features)

    print("Extracting features for training data...")
    train_features = extract_all_features(train_df, "Training")
    train_targets = train_df[["angle", "depth", "left_right"]].values
    train_pids = train_df["participant_id"].values
    train_ids = train_df["id"].values

    print("Extracting features for test data...")
    test_features = extract_all_features(test_df, "Test")
    test_pids = test_df["participant_id"].values
    test_ids = test_df["id"].values

    return {
        "train_features": train_features,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_features": test_features,
        "test_pids": test_pids,
        "test_ids": test_ids,
    }


def train_and_evaluate(data):
    """Train per-player Ridge model and evaluate."""
    print("\n" + "=" * 60)
    print("ADVANCED BIOMECHANICAL FEATURES MODEL")
    print("=" * 60)

    X_train = np.nan_to_num(data["train_features"].values, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = data["train_targets"]
    train_pids = data["train_pids"]
    X_test = np.nan_to_num(data["test_features"].values, nan=0.0, posinf=0.0, neginf=0.0)
    test_pids = data["test_pids"]

    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Feature names: {list(data['train_features'].columns)[:20]}...")

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
            y_tr = y_train[train_mask, t_idx]
            X_te = X_test[test_mask]

            player_indices = np.where(train_mask)[0]

            # Scale features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            # 5-fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_tr_scaled):
                X_fold_tr = X_tr_scaled[train_idx]
                X_fold_val = X_tr_scaled[val_idx]
                y_fold_tr = y_tr[train_idx]

                model = Ridge(alpha=100)
                model.fit(X_fold_tr, y_fold_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_fold_val)
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


def create_submission(test_ids, predictions, cv_score, approach_name):
    """Create submission file."""
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        name = f.stem
        if name.startswith("submission_"):
            try:
                nums.append(int(name.split('_')[1]))
            except:
                pass
        elif name.startswith("submission"):
            try:
                nums.append(int(name[10:]))
            except:
                pass

    next_num = max(nums) + 1 if nums else 1

    scaled_preds = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: {approach_name}")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")
    print(f"  depth_mean: {depth_mean:.4f}")
    print(f"  File: {filepath}")

    return filepath, next_num


def main():
    print("=" * 80)
    print("ADVANCED BIOMECHANICAL FEATURES")
    print("=" * 80)
    print("\nNew features based on latest research:")
    print("  - Center of mass dynamics")
    print("  - Angular momentum / rotational dynamics")
    print("  - Kinetic chain (proximal-to-distal)")
    print("  - Elbow-wrist coordination")
    print("  - Joint angle velocities/accelerations")
    print("  - Guide hand dynamics")
    print("  - Symmetry indices")
    print("  - Phase durations")
    print("  - Motion smoothness (jerk)")

    data = load_and_extract_features()

    # Train and evaluate
    oof, test_preds, cv_score = train_and_evaluate(data)

    # Create submission
    filepath, sub_num = create_submission(
        data["test_ids"], test_preds, cv_score, "advanced_biomech"
    )

    # Test blending with Sub 25
    print("\n" + "=" * 60)
    print("BLENDING WITH SUB 25")
    print("=" * 60)

    sub25 = pd.read_csv(SUBMISSION_DIR / "submission_25.csv")
    new_sub = pd.read_csv(filepath)

    # Check correlation
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        r = np.corrcoef(sub25[col], new_sub[col])[0, 1]
        print(f"{col}: correlation with Sub 25 = {r:.4f}")

    # Test blends
    print("\nBlend tests:")
    cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']
    for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        blend = (1-w) * sub25[cols] + w * new_sub[cols]
        print(f"  w={w:.2f}: angle_std={blend.scaled_angle.std():.4f}, "
              f"depth_mean={blend.scaled_depth.mean():.4f}")

    return filepath, sub_num


if __name__ == "__main__":
    main()
