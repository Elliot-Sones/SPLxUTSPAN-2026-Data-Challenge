"""
Physics-based feature engineering for basketball free throw prediction.

Once the ball leaves the hand, its trajectory is completely determined by 6 quantities at release:
- Position: (x, y, z) - where the ball starts
- Velocity: (vx, vy, vz) - direction and speed

This module extracts ~40 physics-based features that directly relate to shot outcome:
- angle: Entry angle = f(release arc) -> wrist_vz / wrist_vy at release
- depth: Short/long = forward velocity -> wrist_vy at release
- left_right: Lateral = sideways velocity -> wrist_vx at release + elbow alignment
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.signal import savgol_filter


# Constants
FRAME_RATE = 60
NUM_FRAMES = 240
DT = 1.0 / FRAME_RATE


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
    """
    Extract x, y, z data for a specific keypoint.

    Args:
        timeseries: (240, 207) array
        keypoint: e.g., "right_wrist"

    Returns:
        (240, 3) array with x, y, z coordinates
    """
    if KEYPOINT_INDEX is None:
        raise RuntimeError("Call init_keypoint_mapping first")

    idx = KEYPOINT_INDEX[keypoint]
    return timeseries[:, idx*3:(idx+1)*3]


def smooth_timeseries(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to reduce noise before velocity computation.

    Args:
        data: Time series data (n_frames,) or (n_frames, n_features)
        window: Window size for smoothing (must be odd)

    Returns:
        Smoothed data
    """
    if window < 3:
        return data

    # Check for NaN values
    if np.any(np.isnan(data)):
        # Skip smoothing if there are NaN values (savgol_filter can't handle them)
        return data

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # polyorder must be less than window
    polyorder = min(2, window - 1)

    try:
        if data.ndim == 1:
            return savgol_filter(data, window, polyorder)
        else:
            return savgol_filter(data, window, polyorder, axis=0)
    except (ValueError, np.linalg.LinAlgError):
        # If savgol_filter fails, return original data
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
    """
    Compute angle at joint p2 formed by p1-p2-p3.

    Args:
        p1, p2, p3: (240, 3) arrays representing 3D positions over time

    Returns:
        (240,) array of angles in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)

    denom = norm1 * norm2
    denom[denom == 0] = 1e-10

    cos_angle = np.clip(dot / denom, -1, 1)
    return np.arccos(cos_angle) * 180 / np.pi


def detect_release_frame(timeseries: np.ndarray, smooth: bool = False) -> int:
    """
    Detect the ball release frame using peak wrist velocity magnitude.

    The release occurs at peak velocity - when the ball leaves the hand.
    """
    if KEYPOINT_INDEX is None:
        return NUM_FRAMES // 2

    try:
        wrist_data = get_keypoint_data(timeseries, "right_wrist")

        # Compute velocity in all 3 axes
        vel = compute_velocity(wrist_data, smooth=smooth)

        # Compute velocity magnitude
        vel_magnitude = np.linalg.norm(vel, axis=1)

        # Search from 1/3 into the shot (release typically in second half)
        search_start = NUM_FRAMES // 3
        release_idx = search_start + np.argmax(vel_magnitude[search_start:])
        return int(release_idx)
    except (KeyError, IndexError):
        return NUM_FRAMES // 2


def extract_physics_features(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    Extract ~40 physics-based features from a single shot.

    Feature categories:
    1. Release Velocity Vector (6 features)
    2. Release Position (6 features)
    3. Arm Alignment at Release (6 features)
    4. Velocity Derivatives (4 features)
    5. Pre-Release Mechanics (8 features)
    6. Participant Features (5 features)

    Args:
        timeseries: (240, 207) array
        participant_id: Optional participant ID
        smooth: Whether to apply smoothing before velocity computation

    Returns:
        Dictionary of ~40 physics-based features
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    # Detect release frame
    release_frame = detect_release_frame(timeseries, smooth=smooth)
    features["release_frame"] = release_frame
    features["release_time"] = release_frame / FRAME_RATE

    try:
        # Get key body part positions
        wrist = get_keypoint_data(timeseries, "right_wrist")
        elbow = get_keypoint_data(timeseries, "right_elbow")
        shoulder = get_keypoint_data(timeseries, "right_shoulder")
        hip = get_keypoint_data(timeseries, "mid_hip")

        # =====================================================================
        # Category 1: Release Velocity Vector (6 features)
        # The most important features - directly determine trajectory
        # =====================================================================
        wrist_vel = compute_velocity(wrist, smooth=smooth)

        # Velocity components at release
        vx = wrist_vel[release_frame, 0]
        vy = wrist_vel[release_frame, 1]
        vz = wrist_vel[release_frame, 2]

        features["wrist_vx_release"] = vx
        features["wrist_vy_release"] = vy  # Forward velocity - predicts depth
        features["wrist_vz_release"] = vz  # Vertical velocity - predicts angle

        # Velocity magnitude
        vel_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        features["velocity_magnitude_release"] = vel_magnitude

        # Elevation angle - predicts entry angle
        # atan2(vz, vy) gives angle of velocity vector in vertical plane
        vel_horiz = np.sqrt(vx**2 + vy**2)
        features["elevation_angle"] = np.arctan2(vz, vel_horiz) * 180 / np.pi

        # Azimuth angle - predicts left/right
        # atan2(vx, vy) gives lateral direction
        features["azimuth_angle"] = np.arctan2(vx, vy) * 180 / np.pi

        # =====================================================================
        # Category 2: Release Position (6 features)
        # Where the ball starts its trajectory
        # =====================================================================
        features["wrist_x_release"] = wrist[release_frame, 0]
        features["wrist_y_release"] = wrist[release_frame, 1]
        features["wrist_z_release"] = wrist[release_frame, 2]

        # Release height relative to hip (accounts for body height)
        features["release_height_relative"] = wrist[release_frame, 2] - hip[release_frame, 2]

        # Lateral offset from shoulder (shooting alignment)
        features["lateral_offset"] = wrist[release_frame, 0] - shoulder[release_frame, 0]

        # Forward position (how far in front of body)
        features["forward_position"] = wrist[release_frame, 1] - shoulder[release_frame, 1]

        # =====================================================================
        # Category 3: Arm Alignment at Release (6 features)
        # Body mechanics that affect velocity direction
        # =====================================================================

        # Elbow alignment - should be ~0 for straight shot
        # Lateral deviation of elbow from shoulder-wrist line
        shoulder_x = shoulder[release_frame, 0]
        wrist_x = wrist[release_frame, 0]
        elbow_x = elbow[release_frame, 0]
        midpoint_x = (shoulder_x + wrist_x) / 2
        features["elbow_alignment"] = elbow_x - midpoint_x

        # Shoulder-elbow-wrist angle (arm extension)
        arm_angle = compute_joint_angle(shoulder, elbow, wrist)
        features["arm_extension_angle"] = arm_angle[release_frame]

        # Wrist snap angle (elbow-wrist-finger if available)
        try:
            finger = get_keypoint_data(timeseries, "right_second_finger_mcp")
            wrist_snap_angle = compute_joint_angle(elbow, wrist, finger)
            features["wrist_snap_angle"] = wrist_snap_angle[release_frame]
        except (KeyError, IndexError):
            features["wrist_snap_angle"] = np.nan

        # Shoulder rotation - angle of shoulder line vs forward direction
        try:
            left_shoulder = get_keypoint_data(timeseries, "left_shoulder")
            shoulder_vec = left_shoulder[release_frame] - shoulder[release_frame]
            # Angle in horizontal plane
            features["shoulder_rotation"] = np.arctan2(shoulder_vec[1], shoulder_vec[0]) * 180 / np.pi
        except (KeyError, IndexError):
            features["shoulder_rotation"] = np.nan

        # Arm plane angle - is arm moving in vertical plane?
        # Compute velocity direction in xy plane
        arm_plane = np.arctan2(vx, vy) * 180 / np.pi
        features["arm_plane_angle"] = arm_plane

        # Elbow height relative to shoulder
        features["elbow_height_relative"] = elbow[release_frame, 2] - shoulder[release_frame, 2]

        # =====================================================================
        # Category 4: Velocity Derivatives (4 features)
        # Timing and consistency indicators
        # =====================================================================

        # Acceleration at release (is velocity still increasing?)
        wrist_acc = compute_acceleration(wrist, smooth=smooth)
        acc_magnitude = np.linalg.norm(wrist_acc[release_frame])
        features["acceleration_at_release"] = acc_magnitude

        # Jerk at release (smoothness)
        jerk = np.gradient(wrist_acc, DT, axis=0)
        jerk_magnitude = np.linalg.norm(jerk[release_frame])
        features["jerk_at_release"] = jerk_magnitude

        # Time to peak velocity
        vel_magnitude_series = np.linalg.norm(wrist_vel, axis=1)
        peak_vel_frame = np.argmax(vel_magnitude_series)
        features["time_to_peak_velocity"] = peak_vel_frame / FRAME_RATE

        # Peak velocity magnitude
        features["peak_velocity_magnitude"] = vel_magnitude_series[peak_vel_frame]

        # =====================================================================
        # Category 5: Pre-Release Mechanics (8 features)
        # Upstream factors that affect release quality
        # =====================================================================

        # Knee angle features
        try:
            right_knee = get_keypoint_data(timeseries, "right_knee")
            right_hip = get_keypoint_data(timeseries, "right_hip")
            right_ankle = get_keypoint_data(timeseries, "right_ankle")

            knee_angle = compute_joint_angle(right_hip, right_knee, right_ankle)
            knee_angle_vel = compute_velocity(knee_angle, smooth=smooth)

            # Min knee angle (maximum bend depth during loading)
            features["knee_bend_depth"] = np.nanmin(knee_angle)

            # Max knee extension rate (power generation)
            # Look in propulsion phase: from 30 frames before release to release
            # This captures the leg drive that powers the shot
            prop_start = max(0, release_frame - 60)
            prop_end = release_frame
            if prop_end > prop_start:
                features["knee_extension_rate_max"] = np.nanmax(knee_angle_vel[prop_start:prop_end])
            else:
                features["knee_extension_rate_max"] = np.nan
        except (KeyError, IndexError):
            features["knee_bend_depth"] = np.nan
            features["knee_extension_rate_max"] = np.nan

        # Set point height (wrist z at pause before release)
        # Look for minimum velocity in frames 60-150 (loading phase)
        try:
            vel_mag = np.linalg.norm(wrist_vel, axis=1)
            load_start, load_end = 60, min(150, release_frame - 10)
            if load_end > load_start:
                set_point_frame = load_start + np.argmin(vel_mag[load_start:load_end])
                features["set_point_height"] = wrist[set_point_frame, 2]
            else:
                features["set_point_height"] = np.nan
        except (IndexError, ValueError):
            features["set_point_height"] = np.nan

        # Hip vertical velocity (upward momentum)
        hip_vel = compute_velocity(hip, smooth=smooth)
        features["hip_vertical_velocity_max"] = np.nanmax(hip_vel[:release_frame, 2])

        # Shoulder elevation rate
        shoulder_vel = compute_velocity(shoulder, smooth=smooth)
        features["shoulder_elevation_rate"] = np.nanmax(shoulder_vel[:release_frame, 2])

        # Guide hand interference - off-hand wrist velocity at release
        try:
            left_wrist = get_keypoint_data(timeseries, "left_wrist")
            left_wrist_vel = compute_velocity(left_wrist, smooth=smooth)
            features["guide_hand_vx"] = left_wrist_vel[release_frame, 0]
        except (KeyError, IndexError):
            features["guide_hand_vx"] = np.nan

        # Balance - hip lateral movement during shot
        hip_lateral_range = np.nanmax(hip[:release_frame, 0]) - np.nanmin(hip[:release_frame, 0])
        features["hip_lateral_range"] = hip_lateral_range

        # Consistency - variance of wrist z during set position
        try:
            set_start, set_end = 60, 120
            wrist_z_std = np.nanstd(wrist[set_start:set_end, 2])
            features["set_position_stability"] = wrist_z_std
        except (IndexError, ValueError):
            features["set_position_stability"] = np.nan

    except (KeyError, IndexError) as e:
        # If essential keypoints are missing, return minimal features
        pass

    # =========================================================================
    # Category 6: Participant Features (5 features)
    # Account for individual shooting styles
    # =========================================================================
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    return features


def get_physics_feature_names() -> List[str]:
    """Return list of all physics feature names in consistent order."""
    feature_names = [
        # Category 1: Release Velocity Vector
        "release_frame", "release_time",
        "wrist_vx_release", "wrist_vy_release", "wrist_vz_release",
        "velocity_magnitude_release", "elevation_angle", "azimuth_angle",
        # Category 2: Release Position
        "wrist_x_release", "wrist_y_release", "wrist_z_release",
        "release_height_relative", "lateral_offset", "forward_position",
        # Category 3: Arm Alignment
        "elbow_alignment", "arm_extension_angle", "wrist_snap_angle",
        "shoulder_rotation", "arm_plane_angle", "elbow_height_relative",
        # Category 4: Velocity Derivatives
        "acceleration_at_release", "jerk_at_release",
        "time_to_peak_velocity", "peak_velocity_magnitude",
        # Category 5: Pre-Release Mechanics
        "knee_bend_depth", "knee_extension_rate_max",
        "set_point_height", "hip_vertical_velocity_max",
        "shoulder_elevation_rate", "guide_hand_vx",
        "hip_lateral_range", "set_position_stability",
        # Category 6: Participant
        "participant_id",
        "participant_1", "participant_2", "participant_3",
        "participant_4", "participant_5",
    ]
    return feature_names


if __name__ == "__main__":
    # Test physics feature extraction
    try:
        from data_loader import load_single_shot, get_keypoint_columns
    except ImportError:
        from src.data_loader import load_single_shot, get_keypoint_columns

    print("Testing physics feature extraction...")

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    print(f"Initialized {len(KEYPOINT_NAMES)} keypoints")

    # Load a single shot
    metadata, timeseries = load_single_shot(0, train=True)
    print(f"Loaded shot: {metadata['id']}")
    print(f"Timeseries shape: {timeseries.shape}")

    # Extract physics features (unsmoothed)
    print("\nExtracting physics features (unsmoothed)...")
    features_unsmoothed = extract_physics_features(
        timeseries,
        participant_id=metadata["participant_id"],
        smooth=False
    )
    print(f"Features extracted: {len(features_unsmoothed)}")

    # Extract physics features (smoothed)
    print("\nExtracting physics features (smoothed)...")
    features_smoothed = extract_physics_features(
        timeseries,
        participant_id=metadata["participant_id"],
        smooth=True
    )
    print(f"Features extracted: {len(features_smoothed)}")

    # Check for NaN values
    nan_count = sum(1 for v in features_unsmoothed.values() if np.isnan(v) if isinstance(v, float))
    print(f"\nNaN values (unsmoothed): {nan_count}")

    # Print key features
    print("\nKey velocity features:")
    for key in ["wrist_vx_release", "wrist_vy_release", "wrist_vz_release",
                "velocity_magnitude_release", "elevation_angle", "azimuth_angle"]:
        print(f"  {key}: {features_unsmoothed.get(key, 'N/A'):.4f}")

    print("\nPhysics feature extraction test complete!")
