"""
Advanced Biomechanics Features - Physics-Informed Engineering

Implements strategies for small-data regime:
1. Joint angles (knee flexion, elbow extension, wrist snap)
2. Center of Mass (CoM) stability
3. Kinetic chain features
4. Anatomical denoising (bone length optimization, Butterworth filter)
5. Temporal segmentation (critical window, set-point detection)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import cdist
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

FPS = 60


def get_keypoint_indices(keypoint_cols):
    """Build keypoint name to index mapping."""
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def get_position(timeseries, kp_idx, name, frame):
    """Get 3D position of a keypoint at a frame."""
    if name not in kp_idx:
        return None
    i = kp_idx[name]
    pos = timeseries[frame, i*3:i*3+3]
    if np.any(np.isnan(pos)):
        return None
    return pos


def get_trajectory(timeseries, kp_idx, name, start=0, end=240):
    """Get full trajectory of a keypoint."""
    if name not in kp_idx:
        return None
    i = kp_idx[name]
    traj = timeseries[start:end, i*3:i*3+3]
    return traj


def butterworth_filter(data, cutoff=8, fs=60, order=4):
    """Zero-lag Butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Handle NaN by interpolating
    if np.any(np.isnan(data)):
        mask = ~np.isnan(data)
        if mask.sum() < 10:
            return data
        data = np.interp(np.arange(len(data)), np.where(mask)[0], data[mask])

    # Apply zero-phase filter
    try:
        filtered = filtfilt(b, a, data)
        return filtered
    except:
        return data


def compute_joint_angle(p1, p2, p3):
    """
    Compute angle at joint p2 formed by segments p1-p2 and p2-p3.
    Returns angle in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return np.nan

    cos_angle = np.clip(dot / (norm1 * norm2), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compute_bone_length(p1, p2):
    """Compute distance between two points."""
    return np.linalg.norm(p2 - p1)


def detect_set_point(wrist_z, eye_level=5.5):
    """
    Detect set-point: when wrist crosses eye level during upward motion.
    Returns frame index.
    """
    # Find frames where wrist is near eye level and moving up
    for i in range(60, len(wrist_z) - 1):
        if wrist_z[i] < eye_level < wrist_z[i+1]:
            return i
    return 100  # Default


def detect_release_frame(wrist_z_vel):
    """Detect release: peak upward velocity."""
    if len(wrist_z_vel) < 10:
        return 110
    return 60 + np.argmax(wrist_z_vel[60:160])


def detect_dip_frame(hip_z):
    """Detect dip: lowest hip position before release."""
    if len(hip_z) < 100:
        return 80
    return np.argmin(hip_z[40:120]) + 40


def compute_center_of_mass(timeseries, kp_idx, frame):
    """
    Estimate Center of Mass from key body segments.
    Simplified: weighted average of major joints.
    """
    # Approximate body segment weights (normalized)
    segments = {
        'mid_hip': 0.4,      # Pelvis/trunk
        'right_shoulder': 0.15,
        'left_shoulder': 0.15,
        'right_knee': 0.1,
        'left_knee': 0.1,
        'head': 0.1
    }

    com = np.zeros(3)
    total_weight = 0

    for joint, weight in segments.items():
        pos = get_position(timeseries, kp_idx, joint, frame)
        if pos is not None:
            com += weight * pos
            total_weight += weight

    if total_weight > 0:
        com /= total_weight

    return com


def extract_advanced_features(timeseries, kp_idx):
    """Extract physics-informed biomechanical features."""
    features = {}

    # Get key joint trajectories
    joints = {
        'right_wrist': get_trajectory(timeseries, kp_idx, 'right_wrist'),
        'right_elbow': get_trajectory(timeseries, kp_idx, 'right_elbow'),
        'right_shoulder': get_trajectory(timeseries, kp_idx, 'right_shoulder'),
        'right_hip': get_trajectory(timeseries, kp_idx, 'right_hip'),
        'right_knee': get_trajectory(timeseries, kp_idx, 'right_knee'),
        'right_ankle': get_trajectory(timeseries, kp_idx, 'right_ankle'),
        'mid_hip': get_trajectory(timeseries, kp_idx, 'mid_hip'),
        'right_third_finger_distal': get_trajectory(timeseries, kp_idx, 'right_third_finger_distal'),
    }

    # Check if we have essential joints
    if joints['right_wrist'] is None or joints['right_elbow'] is None:
        return None

    wrist = joints['right_wrist']
    elbow = joints['right_elbow']
    shoulder = joints['right_shoulder']

    # =========================================================================
    # 1. ANATOMICAL DENOISING - Apply Butterworth filter
    # =========================================================================
    wrist_z_filtered = butterworth_filter(wrist[:, 2], cutoff=8)
    wrist_x_filtered = butterworth_filter(wrist[:, 0], cutoff=8)
    wrist_y_filtered = butterworth_filter(wrist[:, 1], cutoff=8)

    # Compute filtered velocities
    wrist_vz = np.gradient(wrist_z_filtered) * FPS
    wrist_vx = np.gradient(wrist_x_filtered) * FPS
    wrist_vy = np.gradient(wrist_y_filtered) * FPS
    wrist_speed = np.sqrt(wrist_vx**2 + wrist_vy**2 + wrist_vz**2)

    features['wrist_vz_max_filtered'] = np.max(wrist_vz)
    features['wrist_speed_max_filtered'] = np.max(wrist_speed)

    # =========================================================================
    # 2. TEMPORAL SEGMENTATION - Critical window detection
    # =========================================================================

    # Detect key frames
    release_frame = detect_release_frame(wrist_vz)
    features['release_frame'] = release_frame

    set_point_frame = detect_set_point(wrist_z_filtered)
    features['set_point_frame'] = set_point_frame

    if joints['mid_hip'] is not None:
        hip_z = joints['mid_hip'][:, 2]
        dip_frame = detect_dip_frame(hip_z)
        features['dip_frame'] = dip_frame
        features['dip_to_release_duration'] = (release_frame - dip_frame) / FPS
    else:
        dip_frame = 80
        features['dip_frame'] = dip_frame
        features['dip_to_release_duration'] = 0.5

    # Critical window features (dip to release)
    critical_start = max(dip_frame, 60)
    critical_end = min(release_frame + 10, 200)

    if critical_end > critical_start:
        features['critical_wrist_vz_max'] = np.max(wrist_vz[critical_start:critical_end])
        features['critical_wrist_vz_mean'] = np.mean(wrist_vz[critical_start:critical_end])
        features['critical_speed_max'] = np.max(wrist_speed[critical_start:critical_end])

    # =========================================================================
    # 3. JOINT ANGLES - Knee flexion, elbow extension, wrist snap
    # =========================================================================

    # Elbow angle over time
    if shoulder is not None:
        elbow_angles = []
        for f in range(critical_start, critical_end):
            if f < len(shoulder) and f < len(elbow) and f < len(wrist):
                angle = compute_joint_angle(shoulder[f], elbow[f], wrist[f])
                if not np.isnan(angle):
                    elbow_angles.append(angle)

        if elbow_angles:
            features['elbow_angle_min'] = np.min(elbow_angles)  # Max flexion
            features['elbow_angle_max'] = np.max(elbow_angles)  # Max extension
            features['elbow_angle_range'] = features['elbow_angle_max'] - features['elbow_angle_min']
            features['elbow_angle_at_release'] = elbow_angles[-1] if elbow_angles else np.nan

    # Knee angle (flexion during loading)
    if joints['right_hip'] is not None and joints['right_knee'] is not None and joints['right_ankle'] is not None:
        hip = joints['right_hip']
        knee = joints['right_knee']
        ankle = joints['right_ankle']

        knee_angles = []
        for f in range(dip_frame - 20, release_frame):
            if 0 <= f < len(hip) and f < len(knee) and f < len(ankle):
                angle = compute_joint_angle(hip[f], knee[f], ankle[f])
                if not np.isnan(angle):
                    knee_angles.append(angle)

        if knee_angles:
            features['knee_angle_min'] = np.min(knee_angles)  # Max flexion (deepest squat)
            features['knee_angle_at_release'] = knee_angles[-1] if knee_angles else np.nan
            features['knee_flexion_depth'] = 180 - features['knee_angle_min']

    # Wrist snap (finger vs wrist velocity differential)
    if joints['right_third_finger_distal'] is not None:
        finger = joints['right_third_finger_distal']
        finger_vz = np.gradient(butterworth_filter(finger[:, 2], cutoff=8)) * FPS

        features['wrist_snap_velocity'] = np.max(finger_vz) - np.max(wrist_vz)
        features['finger_vz_max'] = np.max(finger_vz)

    # =========================================================================
    # 4. CENTER OF MASS (CoM) STABILITY
    # =========================================================================

    com_positions = []
    for f in range(critical_start, critical_end):
        com = compute_center_of_mass(timeseries, kp_idx, f)
        if np.any(com != 0):
            com_positions.append(com)

    if len(com_positions) > 5:
        com_array = np.array(com_positions)

        # CoM stability (less movement = more stable)
        features['com_x_range'] = np.max(com_array[:, 0]) - np.min(com_array[:, 0])
        features['com_y_range'] = np.max(com_array[:, 1]) - np.min(com_array[:, 1])
        features['com_z_range'] = np.max(com_array[:, 2]) - np.min(com_array[:, 2])
        features['com_stability'] = 1.0 / (features['com_x_range'] + features['com_y_range'] + 0.01)

        # CoM trajectory (upward = good leg drive)
        features['com_z_rise'] = com_array[-1, 2] - com_array[0, 2]

    # =========================================================================
    # 5. KINETIC CHAIN - Sequential energy transfer
    # =========================================================================

    # Peak velocity timing for each joint
    peak_times = {}

    if joints['mid_hip'] is not None:
        hip_vz = np.gradient(butterworth_filter(joints['mid_hip'][:, 2], cutoff=8)) * FPS
        peak_times['hip'] = np.argmax(hip_vz[60:160]) + 60

    if shoulder is not None:
        shoulder_vz = np.gradient(butterworth_filter(shoulder[:, 2], cutoff=8)) * FPS
        peak_times['shoulder'] = np.argmax(shoulder_vz[60:160]) + 60

    if elbow is not None:
        elbow_vz = np.gradient(butterworth_filter(elbow[:, 2], cutoff=8)) * FPS
        peak_times['elbow'] = np.argmax(elbow_vz[60:160]) + 60

    peak_times['wrist'] = np.argmax(wrist_vz[60:160]) + 60

    # Kinetic chain timing
    if 'hip' in peak_times:
        features['kc_hip_to_wrist'] = (peak_times['wrist'] - peak_times['hip']) / FPS
    if 'shoulder' in peak_times and 'elbow' in peak_times:
        features['kc_shoulder_to_elbow'] = (peak_times['elbow'] - peak_times['shoulder']) / FPS
    if 'elbow' in peak_times:
        features['kc_elbow_to_wrist'] = (peak_times['wrist'] - peak_times['elbow']) / FPS

    # Check if kinetic chain is in correct order (proximal to distal)
    order = ['hip', 'shoulder', 'elbow', 'wrist']
    times = [peak_times.get(j, -1) for j in order if j in peak_times]
    features['kc_correct_order'] = 1.0 if times == sorted(times) else 0.0

    # =========================================================================
    # 6. BONE LENGTH CONSISTENCY (Anatomical constraint)
    # =========================================================================

    # Compute bone lengths at release (should be consistent)
    if shoulder is not None and release_frame < len(shoulder):
        features['upper_arm_length'] = compute_bone_length(shoulder[release_frame], elbow[release_frame])
        features['forearm_length'] = compute_bone_length(elbow[release_frame], wrist[release_frame])

    # =========================================================================
    # 7. RELEASE KINEMATICS
    # =========================================================================

    # Velocity at release
    if release_frame < len(wrist_vz):
        features['release_vz'] = wrist_vz[release_frame]
        features['release_vx'] = wrist_vx[release_frame]
        features['release_vy'] = wrist_vy[release_frame]
        features['release_speed'] = wrist_speed[release_frame]

        # Release angle
        horiz_speed = np.sqrt(wrist_vx[release_frame]**2 + wrist_vy[release_frame]**2)
        if horiz_speed > 0.1:
            features['release_angle'] = np.degrees(np.arctan2(wrist_vz[release_frame], horiz_speed))
        else:
            features['release_angle'] = 85.0

    # Release height
    features['release_height'] = wrist_z_filtered[release_frame]

    return features


def main():
    print("=" * 80)
    print("ADVANCED BIOMECHANICS FEATURE EXTRACTION")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_indices(keypoint_cols)

    print(f"Keypoints available: {len(kp_idx)}")

    # Extract features for all training shots
    print("\nExtracting features...")
    data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_advanced_features(timeseries, kp_idx)
        if features is None:
            continue

        features['player_id'] = metadata['participant_id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']

        data.append(features)

        if i % 50 == 0:
            print(f"  Processed {i+1} shots...")

    df = pd.DataFrame(data)
    print(f"\nTotal samples: {len(df)}")
    print(f"Features extracted: {len(df.columns) - 4}")

    # Save features
    output_path = PROJECT_DIR / "output" / "advanced_biomech_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")

    # Print feature statistics
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    feature_cols = [c for c in df.columns if c not in ['player_id', 'angle', 'depth', 'left_right']]

    for feat in feature_cols[:20]:  # First 20
        vals = df[feat].dropna()
        if len(vals) > 0:
            print(f"{feat:35s}: mean={vals.mean():8.3f}, std={vals.std():8.3f}")

    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION WITH TARGETS")
    print("=" * 80)

    from scipy.stats import pearsonr

    for target in ['angle', 'depth', 'left_right']:
        print(f"\n{target.upper()}:")
        correlations = []

        for feat in feature_cols:
            valid = df[[feat, target]].dropna()
            if len(valid) > 10:
                r, p = pearsonr(valid[feat], valid[target])
                correlations.append((feat, r, p))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, r, p in correlations[:10]:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:35s}: r={r:+.4f} {sig}")


if __name__ == "__main__":
    main()
