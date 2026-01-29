"""
Advanced feature engineering with frame-specific features.

Based on research findings:
- Frame 153: ankle/knee z predicts ANGLE (R2=0.45)
- Frame 102: hand positions predict DEPTH (R2=0.08)
- Frame 237: finger positions predict LEFT_RIGHT (R2=0.025)
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats as sp_stats
from scipy.signal import savgol_filter


# Constants
FRAME_RATE = 60
NUM_FRAMES = 240
DT = 1.0 / FRAME_RATE

# Critical frames from research
ANGLE_FRAME = 153  # Best predictability for angle
DEPTH_FRAME = 102  # Best predictability for depth
LEFT_RIGHT_FRAME = 237  # Best predictability for left_right

# Keypoint mapping (initialized at runtime)
KEYPOINT_NAMES = None
KEYPOINT_INDEX = None


def init_keypoint_mapping(keypoint_cols: List[str]):
    """Initialize keypoint name to index mapping."""
    global KEYPOINT_NAMES, KEYPOINT_INDEX

    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])

    KEYPOINT_NAMES = keypoints
    KEYPOINT_INDEX = {name: i for i, name in enumerate(keypoints)}


def get_keypoint_data(timeseries: np.ndarray, keypoint: str) -> np.ndarray:
    """Extract x, y, z data for a specific keypoint."""
    if KEYPOINT_INDEX is None:
        raise RuntimeError("Call init_keypoint_mapping first")

    idx = KEYPOINT_INDEX[keypoint]
    return timeseries[:, idx*3:(idx+1)*3]


def smooth_timeseries(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply Savitzky-Golay smoothing."""
    if window < 3 or np.any(np.isnan(data)):
        return data
    if window % 2 == 0:
        window += 1
    polyorder = min(2, window - 1)
    try:
        if data.ndim == 1:
            return savgol_filter(data, window, polyorder)
        else:
            return savgol_filter(data, window, polyorder, axis=0)
    except (ValueError, np.linalg.LinAlgError):
        return data


def compute_velocity(series: np.ndarray, smooth: bool = False) -> np.ndarray:
    """Compute velocity using central differences."""
    if smooth:
        series = smooth_timeseries(series)
    return np.gradient(series, DT, axis=0) if series.ndim > 1 else np.gradient(series, DT)


def compute_acceleration(series: np.ndarray, smooth: bool = False) -> np.ndarray:
    """Compute acceleration (second derivative)."""
    vel = compute_velocity(series, smooth)
    return np.gradient(vel, DT, axis=0) if vel.ndim > 1 else np.gradient(vel, DT)


def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Compute angle at joint p2 formed by p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2

    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)

    denom = norm1 * norm2
    denom[denom == 0] = 1e-10

    cos_angle = np.clip(dot / denom, -1, 1)
    return np.arccos(cos_angle) * 180 / np.pi


def extract_frame_features(timeseries: np.ndarray, frame: int, prefix: str) -> Dict[str, float]:
    """Extract all keypoint positions at a specific frame."""
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    for keypoint in KEYPOINT_NAMES:
        try:
            data = get_keypoint_data(timeseries, keypoint)
            features[f"{prefix}_{keypoint}_x"] = data[frame, 0]
            features[f"{prefix}_{keypoint}_y"] = data[frame, 1]
            features[f"{prefix}_{keypoint}_z"] = data[frame, 2]
        except (KeyError, IndexError):
            pass

    return features


def extract_angle_critical_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract features critical for ANGLE prediction.
    Research shows frame 153 with ankle/knee z-positions are most predictive (R2=0.45).
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    frame = ANGLE_FRAME

    # Top predictors for angle at frame 153
    critical_keypoints = [
        "left_ankle", "right_ankle",
        "left_knee", "right_knee",
        "left_heel", "right_heel",
        "left_big_toe", "right_big_toe",
        "left_small_toe", "right_small_toe",
        "mid_hip", "left_hip", "right_hip"
    ]

    for kp in critical_keypoints:
        try:
            data = get_keypoint_data(timeseries, kp)
            # Position at critical frame
            features[f"angle_f{frame}_{kp}_z"] = data[frame, 2]
            features[f"angle_f{frame}_{kp}_y"] = data[frame, 1]
            features[f"angle_f{frame}_{kp}_x"] = data[frame, 0]

            # Also get velocity at critical frame
            vel = compute_velocity(data, smooth=True)
            features[f"angle_f{frame}_{kp}_vz"] = vel[frame, 2]
        except (KeyError, IndexError):
            pass

    # Multi-frame window around critical frame (smoothing)
    window_start = max(0, frame - 5)
    window_end = min(NUM_FRAMES, frame + 5)

    for kp in ["left_ankle", "right_ankle", "left_knee", "right_knee"]:
        try:
            data = get_keypoint_data(timeseries, kp)
            z_window = data[window_start:window_end, 2]
            features[f"angle_window_{kp}_z_mean"] = np.nanmean(z_window)
            features[f"angle_window_{kp}_z_std"] = np.nanstd(z_window)
        except (KeyError, IndexError):
            pass

    # Joint angles at critical frame
    try:
        left_hip = get_keypoint_data(timeseries, "left_hip")
        left_knee = get_keypoint_data(timeseries, "left_knee")
        left_ankle = get_keypoint_data(timeseries, "left_ankle")

        left_knee_angle = compute_joint_angle(left_hip, left_knee, left_ankle)
        features["angle_f153_left_knee_angle"] = left_knee_angle[frame]
    except (KeyError, IndexError):
        pass

    try:
        right_hip = get_keypoint_data(timeseries, "right_hip")
        right_knee = get_keypoint_data(timeseries, "right_knee")
        right_ankle = get_keypoint_data(timeseries, "right_ankle")

        right_knee_angle = compute_joint_angle(right_hip, right_knee, right_ankle)
        features["angle_f153_right_knee_angle"] = right_knee_angle[frame]
    except (KeyError, IndexError):
        pass

    return features


def extract_depth_critical_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract features critical for DEPTH prediction.
    Research shows frame 102 with left hand finger positions are most predictive (R2=0.08).
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    frame = DEPTH_FRAME

    # Top predictors for depth at frame 102 - left hand (guide hand) fingers
    critical_keypoints = [
        "left_second_finger_pip", "left_first_finger_cmc", "left_first_finger_mcp",
        "left_thumb", "left_second_finger_mcp", "left_third_finger_pip",
        "left_wrist", "left_first_finger_pip", "left_third_finger_mcp",
        # Also right hand
        "right_wrist", "right_elbow", "right_shoulder"
    ]

    for kp in critical_keypoints:
        try:
            data = get_keypoint_data(timeseries, kp)
            features[f"depth_f{frame}_{kp}_x"] = data[frame, 0]
            features[f"depth_f{frame}_{kp}_y"] = data[frame, 1]
            features[f"depth_f{frame}_{kp}_z"] = data[frame, 2]

            # Velocity
            vel = compute_velocity(data, smooth=True)
            features[f"depth_f{frame}_{kp}_vel_mag"] = np.linalg.norm(vel[frame])
        except (KeyError, IndexError):
            pass

    # Multi-frame window
    window_start = max(0, frame - 5)
    window_end = min(NUM_FRAMES, frame + 5)

    for kp in ["left_wrist", "right_wrist"]:
        try:
            data = get_keypoint_data(timeseries, kp)
            y_window = data[window_start:window_end, 1]
            features[f"depth_window_{kp}_y_mean"] = np.nanmean(y_window)
            features[f"depth_window_{kp}_y_std"] = np.nanstd(y_window)
        except (KeyError, IndexError):
            pass

    return features


def extract_leftright_critical_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract features critical for LEFT_RIGHT prediction.
    Research shows frame 237 with right hand finger z-positions are marginally predictive (R2=0.025).
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    frame = LEFT_RIGHT_FRAME

    # Top predictors for left_right at frame 237 - right hand fingers z
    critical_keypoints = [
        "right_third_finger_distal", "right_third_finger_dip",
        "right_second_finger_dip", "right_third_finger_pip",
        "right_first_finger_mcp", "right_second_finger_pip",
        "right_wrist", "right_elbow",
        # Balance indicators
        "mid_hip", "left_hip", "right_hip",
        "left_ankle", "right_ankle"
    ]

    for kp in critical_keypoints:
        try:
            data = get_keypoint_data(timeseries, kp)
            features[f"lr_f{frame}_{kp}_x"] = data[frame, 0]
            features[f"lr_f{frame}_{kp}_y"] = data[frame, 1]
            features[f"lr_f{frame}_{kp}_z"] = data[frame, 2]
        except (KeyError, IndexError):
            pass

    # Lateral balance throughout shot
    try:
        hip = get_keypoint_data(timeseries, "mid_hip")
        features["lr_hip_x_range"] = np.nanmax(hip[:, 0]) - np.nanmin(hip[:, 0])
        features["lr_hip_x_std"] = np.nanstd(hip[:, 0])

        # Hip x at different phases
        features["lr_hip_x_setup"] = np.nanmean(hip[0:60, 0])
        features["lr_hip_x_release"] = np.nanmean(hip[120:180, 0])
        features["lr_hip_x_follow"] = np.nanmean(hip[180:240, 0])
    except (KeyError, IndexError):
        pass

    # Arm alignment
    try:
        shoulder = get_keypoint_data(timeseries, "right_shoulder")
        elbow = get_keypoint_data(timeseries, "right_elbow")
        wrist = get_keypoint_data(timeseries, "right_wrist")

        # Elbow lateral deviation from shoulder-wrist line
        midpoint_x = (shoulder[:, 0] + wrist[:, 0]) / 2
        elbow_deviation = elbow[:, 0] - midpoint_x

        features["lr_elbow_deviation_mean"] = np.nanmean(elbow_deviation)
        features["lr_elbow_deviation_f153"] = elbow_deviation[153]
        features["lr_elbow_deviation_f237"] = elbow_deviation[237] if len(elbow_deviation) > 237 else np.nan
    except (KeyError, IndexError):
        pass

    return features


def extract_release_features(timeseries: np.ndarray) -> Dict[str, float]:
    """Extract features at detected release point."""
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    try:
        wrist = get_keypoint_data(timeseries, "right_wrist")
        wrist_vel = compute_velocity(wrist, smooth=True)
        vel_mag = np.linalg.norm(wrist_vel, axis=1)

        # Find release frame (peak velocity in second half)
        search_start = NUM_FRAMES // 3
        release_frame = search_start + np.argmax(vel_mag[search_start:])

        features["release_frame_detected"] = release_frame
        features["release_velocity_mag"] = vel_mag[release_frame]
        features["release_vx"] = wrist_vel[release_frame, 0]
        features["release_vy"] = wrist_vel[release_frame, 1]
        features["release_vz"] = wrist_vel[release_frame, 2]

        # Position at release
        features["release_wrist_x"] = wrist[release_frame, 0]
        features["release_wrist_y"] = wrist[release_frame, 1]
        features["release_wrist_z"] = wrist[release_frame, 2]

        # Elevation angle
        vel_horiz = np.sqrt(wrist_vel[release_frame, 0]**2 + wrist_vel[release_frame, 1]**2)
        features["release_elevation_angle"] = np.arctan2(wrist_vel[release_frame, 2], vel_horiz) * 180 / np.pi

        # Azimuth angle
        features["release_azimuth_angle"] = np.arctan2(wrist_vel[release_frame, 0], wrist_vel[release_frame, 1]) * 180 / np.pi

    except (KeyError, IndexError):
        pass

    return features


def extract_phase_features(timeseries: np.ndarray) -> Dict[str, float]:
    """Extract features from different phases of the shot."""
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    # Phase definitions
    phases = {
        "setup": (0, 60),
        "load": (60, 120),
        "propulsion": (120, 180),
        "follow": (180, 240)
    }

    key_joints = ["right_wrist", "right_elbow", "right_knee", "mid_hip"]

    for joint in key_joints:
        try:
            data = get_keypoint_data(timeseries, joint)
            vel = compute_velocity(data, smooth=True)

            for phase_name, (start, end) in phases.items():
                # Position stats per phase
                features[f"phase_{phase_name}_{joint}_z_mean"] = np.nanmean(data[start:end, 2])
                features[f"phase_{phase_name}_{joint}_z_range"] = np.nanmax(data[start:end, 2]) - np.nanmin(data[start:end, 2])

                # Velocity stats per phase
                vel_mag = np.linalg.norm(vel[start:end], axis=1)
                features[f"phase_{phase_name}_{joint}_vel_mean"] = np.nanmean(vel_mag)
                features[f"phase_{phase_name}_{joint}_vel_max"] = np.nanmax(vel_mag)
        except (KeyError, IndexError):
            pass

    return features


def extract_consistency_features(timeseries: np.ndarray) -> Dict[str, float]:
    """Extract features related to movement consistency/smoothness."""
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    try:
        wrist = get_keypoint_data(timeseries, "right_wrist")
        vel = compute_velocity(wrist, smooth=True)
        acc = compute_acceleration(wrist, smooth=True)
        jerk = np.gradient(acc, DT, axis=0)

        # Smoothness metrics
        vel_mag = np.linalg.norm(vel, axis=1)
        acc_mag = np.linalg.norm(acc, axis=1)
        jerk_mag = np.linalg.norm(jerk, axis=1)

        features["wrist_vel_smoothness"] = np.nanstd(vel_mag)
        features["wrist_acc_smoothness"] = np.nanstd(acc_mag)
        features["wrist_jerk_mean"] = np.nanmean(jerk_mag)
        features["wrist_jerk_max"] = np.nanmax(jerk_mag)

        # Trajectory straightness
        trajectory = wrist[60:180] - wrist[60]  # During propulsion
        if len(trajectory) > 1:
            # Compute path length vs direct distance
            path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            direct_dist = np.linalg.norm(trajectory[-1])
            if direct_dist > 0.01:
                features["trajectory_straightness"] = direct_dist / (path_length + 1e-6)
            else:
                features["trajectory_straightness"] = 0.0
    except (KeyError, IndexError):
        pass

    return features


def extract_advanced_features(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None
) -> Dict[str, float]:
    """
    Extract all advanced features.

    Returns ~200+ features specifically designed for each target.
    """
    features = {}

    # Target-specific critical features
    features.update(extract_angle_critical_features(timeseries))
    features.update(extract_depth_critical_features(timeseries))
    features.update(extract_leftright_critical_features(timeseries))

    # Release features
    features.update(extract_release_features(timeseries))

    # Phase features
    features.update(extract_phase_features(timeseries))

    # Consistency features
    features.update(extract_consistency_features(timeseries))

    # Participant one-hot
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    return features


if __name__ == "__main__":
    # Test
    print("Testing advanced feature extraction...")

    try:
        from data_loader import load_single_shot, get_keypoint_columns
    except ImportError:
        from src.data_loader import load_single_shot, get_keypoint_columns

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    print(f"Initialized {len(KEYPOINT_NAMES)} keypoints")

    metadata, timeseries = load_single_shot(0, train=True)
    print(f"Loaded shot: {metadata['id']}")

    features = extract_advanced_features(timeseries, metadata["participant_id"])
    print(f"Advanced features: {len(features)}")

    # Show feature counts by category
    angle_features = sum(1 for k in features if k.startswith("angle_"))
    depth_features = sum(1 for k in features if k.startswith("depth_"))
    lr_features = sum(1 for k in features if k.startswith("lr_"))
    release_features = sum(1 for k in features if k.startswith("release_"))
    phase_features = sum(1 for k in features if k.startswith("phase_"))

    print(f"\nFeature breakdown:")
    print(f"  Angle-critical: {angle_features}")
    print(f"  Depth-critical: {depth_features}")
    print(f"  Left/Right-critical: {lr_features}")
    print(f"  Release: {release_features}")
    print(f"  Phase: {phase_features}")
