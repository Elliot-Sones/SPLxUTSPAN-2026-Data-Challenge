"""
Angle-specific feature engineering for basketball free throw entry angle prediction.

Entry angle is determined by the velocity ratio at the hoop:
    angle = arctan(-vz / sqrt(vx^2 + vy^2))

This depends on:
1. Release trajectory (initial angle)
2. Gravity effect (time of flight)
3. Initial velocity magnitude
4. Arc height
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.signal import savgol_filter

# Constants
FRAME_RATE = 60
NUM_FRAMES = 240
DT = 1.0 / FRAME_RATE
GRAVITY = 9.81  # m/s^2

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


def compute_velocity(series: np.ndarray, smooth: bool = True) -> np.ndarray:
    """Compute velocity using central differences."""
    if smooth and not np.any(np.isnan(series)):
        try:
            if series.ndim == 1:
                series = savgol_filter(series, 5, 2)
            else:
                series = savgol_filter(series, 5, 2, axis=0)
        except (ValueError, np.linalg.LinAlgError):
            # If savgol_filter fails, skip smoothing
            pass
    return np.gradient(series, DT, axis=0) if series.ndim > 1 else np.gradient(series, DT)


def compute_acceleration(series: np.ndarray, smooth: bool = True) -> np.ndarray:
    """Compute acceleration (second derivative)."""
    vel = compute_velocity(series, smooth)
    return np.gradient(vel, DT, axis=0) if vel.ndim > 1 else np.gradient(vel, DT)


def detect_release_frame(timeseries: np.ndarray) -> int:
    """Detect ball release frame using peak wrist velocity magnitude."""
    if KEYPOINT_INDEX is None:
        return NUM_FRAMES // 2

    try:
        wrist_data = get_keypoint_data(timeseries, "right_wrist")
        vel = compute_velocity(wrist_data, smooth=True)
        vel_magnitude = np.linalg.norm(vel, axis=1)
        search_start = NUM_FRAMES // 3
        release_idx = search_start + np.argmax(vel_magnitude[search_start:])
        return int(release_idx)
    except (KeyError, IndexError):
        return NUM_FRAMES // 2


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


def extract_angle_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract features specifically designed for angle prediction.

    Entry angle is determined by velocity ratio at hoop:
    angle = arctan(-vz / sqrt(vx^2 + vy^2))

    This is affected by:
    1. Release trajectory (initial angle)
    2. Gravity effect (time of flight)
    3. Initial velocity magnitude
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    try:
        # Get key body parts
        wrist = get_keypoint_data(timeseries, "right_wrist")
        elbow = get_keypoint_data(timeseries, "right_elbow")
        shoulder = get_keypoint_data(timeseries, "right_shoulder")

        # Detect release frame
        release_frame = detect_release_frame(timeseries)
        features["release_frame"] = release_frame

        # Compute velocities
        wrist_vel = compute_velocity(wrist, smooth=True)

        # =====================================================================
        # 1. RELEASE TRAJECTORY ANGLE
        # =====================================================================
        vx_rel = wrist_vel[release_frame, 0]
        vy_rel = wrist_vel[release_frame, 1]
        vz_rel = wrist_vel[release_frame, 2]

        vel_horizontal = np.sqrt(vx_rel**2 + vy_rel**2)
        if vel_horizontal > 0:
            release_elevation = np.arctan2(vz_rel, vel_horizontal) * 180 / np.pi
            features["release_elevation_angle"] = release_elevation
        else:
            features["release_elevation_angle"] = np.nan

        # =====================================================================
        # 2. ARC TRAJECTORY FEATURES
        # =====================================================================
        # Maximum height reached (affects entry angle)
        max_wrist_height = np.nanmax(wrist[:, 2])
        release_height = wrist[release_frame, 2]
        features["arc_height"] = max_wrist_height - release_height

        # Peak height timing (early peak = steeper entry)
        peak_height_frame = np.nanargmax(wrist[:, 2])
        features["peak_height_frame"] = peak_height_frame
        features["peak_height_relative_timing"] = (peak_height_frame - release_frame) / FRAME_RATE

        # =====================================================================
        # 3. VELOCITY MAGNITUDE (affects time of flight)
        # =====================================================================
        vel_magnitude = np.linalg.norm([vx_rel, vy_rel, vz_rel])
        features["release_velocity_magnitude"] = vel_magnitude

        # Estimate time to hoop (forward distance / forward velocity)
        forward_distance = 4.0  # Approximate free throw distance (meters)
        if abs(vy_rel) > 0.1:
            estimated_flight_time = forward_distance / abs(vy_rel)
            features["estimated_flight_time"] = estimated_flight_time

            # Estimate velocity at hoop using physics
            vz_at_hoop = vz_rel - GRAVITY * estimated_flight_time
            if vel_horizontal > 0:
                estimated_entry_angle = np.arctan2(-vz_at_hoop, vel_horizontal) * 180 / np.pi
                features["estimated_entry_angle_physics"] = estimated_entry_angle
            else:
                features["estimated_entry_angle_physics"] = np.nan
        else:
            features["estimated_flight_time"] = np.nan
            features["estimated_entry_angle_physics"] = np.nan

        # =====================================================================
        # 4. VELOCITY DECAY/CONSISTENCY
        # =====================================================================
        # How smoothly does velocity change near release?
        pre_release_window = slice(max(0, release_frame-10), release_frame)
        vel_mag_series = np.linalg.norm(wrist_vel, axis=1)

        if release_frame > 10:
            vel_pre_release = vel_mag_series[pre_release_window]
            valid_mask = ~np.isnan(vel_pre_release)

            if valid_mask.sum() > 2:
                # Linear fit to velocity profile
                times = np.arange(len(vel_pre_release))
                slope, intercept = np.polyfit(times[valid_mask], vel_pre_release[valid_mask], 1)
                features["velocity_decay_rate"] = slope

                # Smoothness (low std = consistent acceleration)
                features["velocity_consistency"] = np.nanstd(vel_pre_release)
            else:
                features["velocity_decay_rate"] = np.nan
                features["velocity_consistency"] = np.nan
        else:
            features["velocity_decay_rate"] = np.nan
            features["velocity_consistency"] = np.nan

        # =====================================================================
        # 5. BODY ANGLE AT RELEASE
        # =====================================================================
        # Shooter lean affects trajectory
        try:
            hip = get_keypoint_data(timeseries, "mid_hip")
            body_vector = shoulder[release_frame] - hip[release_frame]
            body_lean = np.arctan2(body_vector[2], body_vector[1]) * 180 / np.pi
            features["body_lean_at_release"] = body_lean
        except (KeyError, IndexError):
            features["body_lean_at_release"] = np.nan

        # =====================================================================
        # 6. ARM ANGLE AT RELEASE
        # =====================================================================
        # High release angle tends to produce steeper entry
        arm_vector = wrist[release_frame] - elbow[release_frame]
        arm_horiz = np.linalg.norm(arm_vector[:2])
        if arm_horiz > 0:
            arm_elevation = np.arctan2(arm_vector[2], arm_horiz) * 180 / np.pi
            features["arm_elevation_at_release"] = arm_elevation
        else:
            features["arm_elevation_at_release"] = np.nan

        # Elbow angle (extension)
        elbow_angle = compute_joint_angle(shoulder, elbow, wrist)
        features["elbow_angle_at_release"] = elbow_angle[release_frame]

        # =====================================================================
        # 7. FOLLOW-THROUGH TRAJECTORY
        # =====================================================================
        # Post-release wrist motion indicates initial ball trajectory
        if release_frame < NUM_FRAMES - 10:
            post_release = slice(release_frame, min(release_frame+10, NUM_FRAMES))
            wrist_z_post = wrist[post_release, 2]

            if len(wrist_z_post) > 3:
                # Is wrist still rising after release?
                post_slope = (wrist_z_post[-1] - wrist_z_post[0]) / len(wrist_z_post)
                features["follow_through_rise"] = post_slope

                # Follow-through consistency
                features["follow_through_smoothness"] = np.nanstd(np.diff(wrist_z_post))
            else:
                features["follow_through_rise"] = np.nan
                features["follow_through_smoothness"] = np.nan
        else:
            features["follow_through_rise"] = np.nan
            features["follow_through_smoothness"] = np.nan

        # =====================================================================
        # 8. WRIST SNAP (AFFECTS BACKSPIN AND TRAJECTORY)
        # =====================================================================
        # Wrist snap angle change
        try:
            finger = get_keypoint_data(timeseries, "right_second_finger_mcp")
            wrist_snap = compute_joint_angle(elbow, wrist, finger)

            features["wrist_snap_at_release"] = wrist_snap[release_frame]

            # Wrist snap rate (angular velocity)
            if release_frame > 5:
                wrist_snap_vel = compute_velocity(wrist_snap, smooth=True)
                features["wrist_snap_rate"] = wrist_snap_vel[release_frame]
            else:
                features["wrist_snap_rate"] = np.nan
        except (KeyError, IndexError):
            features["wrist_snap_at_release"] = np.nan
            features["wrist_snap_rate"] = np.nan

    except (KeyError, IndexError) as e:
        pass

    return features


def extract_trajectory_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Capture pre-release hand trajectory that determines ball velocity.

    The acceleration phase before release is critical for determining
    the release velocity, which in turn determines the entry angle.
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    try:
        wrist = get_keypoint_data(timeseries, "right_wrist")
        release_frame = detect_release_frame(timeseries)

        # Define acceleration phase: ~60 frames before release
        accel_start = max(80, release_frame - 60)
        accel_end = release_frame

        if accel_end > accel_start + 10:
            # Wrist trajectory during acceleration
            wrist_accel_phase = wrist[accel_start:accel_end, :]

            # Vertical acceleration profile
            z_vals = wrist_accel_phase[:, 2]
            z_velocity = np.gradient(z_vals, DT)
            z_accel = np.gradient(z_velocity, DT)

            features["accel_phase_mean_vz"] = np.nanmean(z_velocity)
            features["accel_phase_max_vz"] = np.nanmax(z_velocity)
            features["accel_phase_mean_az"] = np.nanmean(z_accel)
            features["accel_phase_max_az"] = np.nanmax(z_accel)

            # Trajectory curvature (2nd derivative magnitude)
            curvature_z = np.abs(z_accel)
            features["trajectory_curvature"] = np.nanmean(curvature_z)

            # Consistency (variance in acceleration)
            features["trajectory_smoothness"] = np.nanstd(z_accel)

            # Path length during acceleration (total distance traveled)
            displacements = np.linalg.norm(np.diff(wrist_accel_phase, axis=0), axis=1)
            features["accel_phase_path_length"] = np.nansum(displacements)

            # Straightness ratio (direct distance / path length)
            direct_distance = np.linalg.norm(wrist_accel_phase[-1] - wrist_accel_phase[0])
            path_length = np.nansum(displacements)
            if path_length > 0:
                features["accel_phase_straightness"] = direct_distance / path_length
            else:
                features["accel_phase_straightness"] = np.nan
        else:
            features["accel_phase_mean_vz"] = np.nan
            features["accel_phase_max_vz"] = np.nan
            features["accel_phase_mean_az"] = np.nan
            features["accel_phase_max_az"] = np.nan
            features["trajectory_curvature"] = np.nan
            features["trajectory_smoothness"] = np.nan
            features["accel_phase_path_length"] = np.nan
            features["accel_phase_straightness"] = np.nan

    except (KeyError, IndexError):
        pass

    return features


def extract_all_angle_features(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None
) -> Dict[str, float]:
    """
    Extract all angle-specific features.

    Args:
        timeseries: (240, 207) array
        participant_id: Optional participant ID

    Returns:
        Dictionary of angle-specific features
    """
    features = {}

    # Participant features
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    # Angle-specific features
    features.update(extract_angle_features(timeseries))

    # Trajectory features
    features.update(extract_trajectory_features(timeseries))

    return features


if __name__ == "__main__":
    # Test angle feature extraction
    from data_loader import load_single_shot, get_keypoint_columns

    print("Testing angle feature extraction...")

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    print(f"Initialized {len(KEYPOINT_NAMES)} keypoints")

    # Load a single shot
    metadata, timeseries = load_single_shot(0, train=True)
    print(f"Loaded shot: {metadata['id']}")
    print(f"Ground truth angle: {metadata['angle']:.2f} degrees")

    # Extract angle features
    print("\nExtracting angle-specific features...")
    features = extract_all_angle_features(timeseries, metadata["participant_id"])
    print(f"Features extracted: {len(features)}")

    # Check for NaN values
    nan_count = sum(1 for v in features.values() if isinstance(v, (int, float)) and np.isnan(v))
    print(f"Features with NaN: {nan_count}")

    # Print key features
    print("\nKey angle features:")
    for key in ["release_elevation_angle", "arc_height", "estimated_entry_angle_physics",
                "arm_elevation_at_release", "follow_through_rise"]:
        val = features.get(key, np.nan)
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}")

    print("\nAngle feature extraction test complete!")
