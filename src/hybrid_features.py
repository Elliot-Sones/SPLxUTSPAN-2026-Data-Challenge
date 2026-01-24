"""
Hybrid feature engineering: Physics features + top z-coordinate statistics.

Combines ~40 physics-based features with ~40 carefully selected statistical features
based on empirical feature importance analysis. Total: ~80 features.

The top generic features in the baseline were primarily z-coordinates of key joints.
This module adds mean, std, min, max, range for:
- right_wrist_z, right_elbow_z, right_shoulder_z
- right_knee_z, right_hip_z, mid_hip_z
- left_wrist_z (guide hand)
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats as sp_stats

try:
    from physics_features import (
        init_keypoint_mapping as physics_init_keypoint_mapping,
        extract_physics_features,
        get_keypoint_data,
        compute_velocity,
        KEYPOINT_INDEX,
        KEYPOINT_NAMES,
        NUM_FRAMES,
        FRAME_RATE,
        DT,
    )
except ImportError:
    from src.physics_features import (
        init_keypoint_mapping as physics_init_keypoint_mapping,
        extract_physics_features,
        get_keypoint_data,
        compute_velocity,
        KEYPOINT_INDEX,
        KEYPOINT_NAMES,
        NUM_FRAMES,
        FRAME_RATE,
        DT,
    )


# Re-export init function
def init_keypoint_mapping(keypoint_cols: List[str]):
    """Initialize keypoint mapping (delegates to physics_features)."""
    physics_init_keypoint_mapping(keypoint_cols)


def extract_zcoord_stats(timeseries: np.ndarray, keypoint: str, prefix: str = None) -> Dict[str, float]:
    """
    Extract statistics for a keypoint's z-coordinate.

    Args:
        timeseries: (240, 207) array
        keypoint: e.g., "right_wrist"
        prefix: Optional prefix for feature names

    Returns:
        Dictionary with mean, std, min, max, range, q25, q75, energy
    """
    features = {}
    name_prefix = prefix if prefix else keypoint

    try:
        data = get_keypoint_data(timeseries, keypoint)
        z = data[:, 2]  # z coordinate

        valid = z[~np.isnan(z)]
        if len(valid) == 0:
            return features

        features[f"{name_prefix}_z_mean"] = np.mean(valid)
        features[f"{name_prefix}_z_std"] = np.std(valid)
        features[f"{name_prefix}_z_min"] = np.min(valid)
        features[f"{name_prefix}_z_max"] = np.max(valid)
        features[f"{name_prefix}_z_range"] = np.max(valid) - np.min(valid)
        features[f"{name_prefix}_z_q25"] = np.percentile(valid, 25)
        features[f"{name_prefix}_z_q75"] = np.percentile(valid, 75)
        features[f"{name_prefix}_z_energy"] = np.sum(valid ** 2)

    except (KeyError, IndexError):
        pass

    return features


def extract_velocity_stats(timeseries: np.ndarray, keypoint: str, smooth: bool = False) -> Dict[str, float]:
    """
    Extract velocity statistics for a keypoint.

    Args:
        timeseries: (240, 207) array
        keypoint: e.g., "right_wrist"
        smooth: Whether to smooth before velocity computation

    Returns:
        Dictionary with velocity statistics
    """
    features = {}

    try:
        data = get_keypoint_data(timeseries, keypoint)

        # Compute velocity magnitude
        vel = compute_velocity(data, smooth=smooth)
        vel_mag = np.linalg.norm(vel, axis=1)

        valid = vel_mag[~np.isnan(vel_mag)]
        if len(valid) == 0:
            return features

        features[f"{keypoint}_vel_mean"] = np.mean(valid)
        features[f"{keypoint}_vel_std"] = np.std(valid)
        features[f"{keypoint}_vel_max"] = np.max(valid)
        features[f"{keypoint}_vel_min"] = np.min(valid)
        features[f"{keypoint}_vel_range"] = np.max(valid) - np.min(valid)

        # Time to max velocity (normalized)
        features[f"{keypoint}_vel_max_time"] = np.argmax(vel_mag) / NUM_FRAMES

        # Velocity at different phases
        phase_bounds = [(0, 60), (60, 120), (120, 180), (180, 240)]
        phase_names = ["prep", "load", "prop", "release"]
        for (start, end), phase_name in zip(phase_bounds, phase_names):
            phase_vel = vel_mag[start:end]
            features[f"{keypoint}_vel_{phase_name}_mean"] = np.nanmean(phase_vel)

    except (KeyError, IndexError):
        pass

    return features


def extract_joint_angle_stats(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract statistics for key joint angles.
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    # Import the compute_joint_angle function
    from physics_features import compute_joint_angle

    # Key joints for shooting
    joints = [
        ("right_elbow", "right_shoulder", "right_elbow", "right_wrist"),
        ("right_knee", "right_hip", "right_knee", "right_ankle"),
        ("right_shoulder", "neck", "right_shoulder", "right_elbow"),
    ]

    for joint_name, kp1, kp2, kp3 in joints:
        try:
            p1 = get_keypoint_data(timeseries, kp1)
            p2 = get_keypoint_data(timeseries, kp2)
            p3 = get_keypoint_data(timeseries, kp3)

            angle = compute_joint_angle(p1, p2, p3)
            valid = angle[~np.isnan(angle)]

            if len(valid) == 0:
                continue

            features[f"{joint_name}_angle_mean"] = np.mean(valid)
            features[f"{joint_name}_angle_std"] = np.std(valid)
            features[f"{joint_name}_angle_min"] = np.min(valid)
            features[f"{joint_name}_angle_max"] = np.max(valid)
            features[f"{joint_name}_angle_range"] = np.max(valid) - np.min(valid)

        except (KeyError, IndexError):
            continue

    return features


def extract_hybrid_features(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    Extract hybrid features: physics + top z-coord statistics.

    Total: ~80 features

    Args:
        timeseries: (240, 207) array
        participant_id: Optional participant ID
        smooth: Whether to apply smoothing

    Returns:
        Dictionary of ~80 features
    """
    features = {}

    # =========================================================================
    # Part 1: All physics-based features (~40)
    # =========================================================================
    physics = extract_physics_features(timeseries, participant_id, smooth=smooth)
    features.update(physics)

    # =========================================================================
    # Part 2: Z-coordinate statistics for key joints (~48)
    # These were the top generic features in baseline
    # =========================================================================
    key_joints_z = [
        "right_wrist",
        "right_elbow",
        "right_shoulder",
        "right_knee",
        "right_hip",
        "mid_hip",
        "left_wrist",  # guide hand
        "neck",
    ]

    for joint in key_joints_z:
        features.update(extract_zcoord_stats(timeseries, joint))

    # =========================================================================
    # Part 3: Velocity statistics for key joints (~30)
    # =========================================================================
    key_joints_vel = [
        "right_wrist",
        "right_elbow",
        "right_knee",
    ]

    for joint in key_joints_vel:
        features.update(extract_velocity_stats(timeseries, joint, smooth=smooth))

    # =========================================================================
    # Part 4: Joint angle statistics (~15)
    # =========================================================================
    features.update(extract_joint_angle_stats(timeseries))

    return features


def get_hybrid_feature_names() -> List[str]:
    """Return list of all hybrid feature names."""
    from physics_features import get_physics_feature_names

    # Start with physics features
    names = get_physics_feature_names()

    # Add z-coord stats
    key_joints_z = [
        "right_wrist", "right_elbow", "right_shoulder",
        "right_knee", "right_hip", "mid_hip", "left_wrist", "neck"
    ]
    z_stats = ["z_mean", "z_std", "z_min", "z_max", "z_range", "z_q25", "z_q75", "z_energy"]
    for joint in key_joints_z:
        for stat in z_stats:
            names.append(f"{joint}_{stat}")

    # Add velocity stats
    key_joints_vel = ["right_wrist", "right_elbow", "right_knee"]
    vel_stats = ["vel_mean", "vel_std", "vel_max", "vel_min", "vel_range", "vel_max_time",
                 "vel_prep_mean", "vel_load_mean", "vel_prop_mean", "vel_release_mean"]
    for joint in key_joints_vel:
        for stat in vel_stats:
            names.append(f"{joint}_{stat}")

    # Add joint angle stats
    joints = ["right_elbow", "right_knee", "right_shoulder"]
    angle_stats = ["angle_mean", "angle_std", "angle_min", "angle_max", "angle_range"]
    for joint in joints:
        for stat in angle_stats:
            names.append(f"{joint}_{stat}")

    return names


if __name__ == "__main__":
    # Test hybrid feature extraction
    try:
        from data_loader import load_single_shot, get_keypoint_columns
    except ImportError:
        from src.data_loader import load_single_shot, get_keypoint_columns

    print("Testing hybrid feature extraction...")

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    print(f"Initialized keypoint mapping")

    # Load a single shot
    metadata, timeseries = load_single_shot(0, train=True)
    print(f"Loaded shot: {metadata['id']}")

    # Extract hybrid features (unsmoothed)
    print("\nExtracting hybrid features (unsmoothed)...")
    features_unsmoothed = extract_hybrid_features(
        timeseries,
        participant_id=metadata["participant_id"],
        smooth=False
    )
    print(f"Features extracted: {len(features_unsmoothed)}")

    # Extract hybrid features (smoothed)
    print("\nExtracting hybrid features (smoothed)...")
    features_smoothed = extract_hybrid_features(
        timeseries,
        participant_id=metadata["participant_id"],
        smooth=True
    )
    print(f"Features extracted: {len(features_smoothed)}")

    # Check for NaN values
    nan_count = sum(1 for v in features_unsmoothed.values()
                    if isinstance(v, float) and np.isnan(v))
    print(f"\nNaN values (unsmoothed): {nan_count}")

    # Show feature categories
    physics_count = sum(1 for k in features_unsmoothed.keys()
                        if k.startswith(("wrist_v", "velocity", "elevation", "azimuth",
                                         "release", "lateral", "forward", "elbow_align",
                                         "arm_", "shoulder_rot", "accel", "jerk", "time_to",
                                         "peak_vel", "knee_", "set_point", "hip_", "guide",
                                         "participant")))
    z_count = sum(1 for k in features_unsmoothed.keys() if "_z_" in k)
    vel_count = sum(1 for k in features_unsmoothed.keys() if "_vel_" in k)
    angle_count = sum(1 for k in features_unsmoothed.keys() if "_angle_" in k)

    print(f"\nFeature breakdown:")
    print(f"  Physics-based: ~{physics_count}")
    print(f"  Z-coordinate stats: {z_count}")
    print(f"  Velocity stats: {vel_count}")
    print(f"  Angle stats: {angle_count}")

    print("\nHybrid feature extraction test complete!")
