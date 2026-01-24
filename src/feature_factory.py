"""
Feature factory for exhaustive grid search.

Defines 8 different feature set extractors (F1-F8) for testing.
Each extractor returns a dictionary of features from a single shot.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from scipy.signal import savgol_filter

try:
    from data_loader import get_keypoint_columns, NUM_FRAMES, FRAME_RATE
    from physics_features import (
        init_keypoint_mapping,
        extract_physics_features,
        get_keypoint_data,
        compute_velocity,
        compute_acceleration,
        compute_joint_angle,
        detect_release_frame,
        KEYPOINT_INDEX,
        DT,
    )
    from hybrid_features import extract_hybrid_features
    from feature_engineering import (
        extract_all_features,
        extract_tier1_features,
        extract_basic_stats,
    )
except ImportError:
    from src.data_loader import get_keypoint_columns, NUM_FRAMES, FRAME_RATE
    from src.physics_features import (
        init_keypoint_mapping,
        extract_physics_features,
        get_keypoint_data,
        compute_velocity,
        compute_acceleration,
        compute_joint_angle,
        detect_release_frame,
        KEYPOINT_INDEX,
        DT,
    )
    from src.hybrid_features import extract_hybrid_features
    from src.feature_engineering import (
        extract_all_features,
        extract_tier1_features,
        extract_basic_stats,
    )


# Feature set definitions
FEATURE_SETS = {
    "F1": "tier1_baseline",
    "F2": "physics_only",
    "F3": "hybrid",
    "F4": "hybrid_with_pid",
    "F5": "top100_importance",
    "F6": "release_window",
    "F7": "z_coords_only",
    "F8": "velocity_focused",
}


def ensure_keypoint_mapping():
    """Ensure keypoint mapping is initialized."""
    if KEYPOINT_INDEX is None:
        keypoint_cols = get_keypoint_columns()
        init_keypoint_mapping(keypoint_cols)


# =============================================================================
# F1: Tier 1 Baseline (~3365 features)
# =============================================================================

def extract_f1_tier1_baseline(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F1: Full tier 1 baseline features.

    Includes basic statistics for all 207 time series.
    ~3365 features (207 x ~14 stats + participant).
    """
    ensure_keypoint_mapping()
    return extract_all_features(timeseries, participant_id, tiers=[1, 2, 3])


# =============================================================================
# F2: Physics Only (~40 features)
# =============================================================================

def extract_f2_physics_only(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F2: Physics-based features only.

    ~40 features focused on release mechanics.
    """
    ensure_keypoint_mapping()
    return extract_physics_features(timeseries, participant_id, smooth=smooth)


# =============================================================================
# F3: Hybrid (~130 features)
# =============================================================================

def extract_f3_hybrid(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F3: Hybrid features - physics + z-coord statistics.

    ~130 features combining physics with top statistical features.
    """
    ensure_keypoint_mapping()
    features = extract_hybrid_features(timeseries, participant_id, smooth=smooth)
    # Remove participant one-hot (will be added in F4)
    for pid in range(1, 6):
        features.pop(f"participant_{pid}", None)
    features.pop("participant_id", None)
    return features


# =============================================================================
# F4: Hybrid with Participant ID (~136 features)
# =============================================================================

def extract_f4_hybrid_with_pid(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F4: Hybrid features with participant one-hot encoding.

    ~136 features (hybrid + 6 participant features).
    """
    ensure_keypoint_mapping()
    features = extract_hybrid_features(timeseries, participant_id, smooth=smooth)
    # Ensure participant features are included
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0
    return features


# =============================================================================
# F5: Top 100 by Importance (placeholder - requires prior training)
# =============================================================================

# This will be populated during runtime based on feature importance analysis
TOP_100_FEATURES = None


def extract_f5_top100_importance(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F5: Top 100 features by LightGBM importance.

    Requires pre-computed feature importance list.
    Falls back to hybrid features if not available.
    """
    ensure_keypoint_mapping()

    if TOP_100_FEATURES is None:
        # Fall back to hybrid
        return extract_f4_hybrid_with_pid(timeseries, participant_id, smooth)

    # Extract all features and filter to top 100
    all_features = extract_all_features(timeseries, participant_id, tiers=[1, 2, 3])
    return {k: v for k, v in all_features.items() if k in TOP_100_FEATURES}


def set_top100_features(feature_names: List[str]):
    """Set the top 100 feature names from importance analysis."""
    global TOP_100_FEATURES
    TOP_100_FEATURES = set(feature_names[:100])


# =============================================================================
# F6: Release Window Only (~80 features)
# =============================================================================

def extract_f6_release_window(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F6: Features from release window only (frames 100-180).

    Focuses on the critical release phase.
    """
    ensure_keypoint_mapping()
    features = {}

    # Window around release
    window_start, window_end = 100, 180
    window_ts = timeseries[window_start:window_end]

    # Key keypoints for release
    key_keypoints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "right_knee", "right_hip", "mid_hip", "neck"
    ]

    for kp_name in key_keypoints:
        try:
            data = get_keypoint_data(timeseries, kp_name)
            window_data = data[window_start:window_end]

            # Position stats in window
            for coord_idx, coord in enumerate(["x", "y", "z"]):
                series = window_data[:, coord_idx]
                valid = series[~np.isnan(series)]
                if len(valid) == 0:
                    continue

                prefix = f"{kp_name}_{coord}_release"
                features[f"{prefix}_mean"] = np.mean(valid)
                features[f"{prefix}_std"] = np.std(valid)
                features[f"{prefix}_range"] = np.max(valid) - np.min(valid)
                features[f"{prefix}_first"] = series[0]
                features[f"{prefix}_last"] = series[-1]

            # Velocity in window
            vel = compute_velocity(data, smooth=smooth)
            window_vel = vel[window_start:window_end]
            vel_mag = np.linalg.norm(window_vel, axis=1)

            features[f"{kp_name}_vel_release_max"] = np.nanmax(vel_mag)
            features[f"{kp_name}_vel_release_mean"] = np.nanmean(vel_mag)

        except (KeyError, IndexError):
            continue

    # Add release-specific physics
    physics = extract_physics_features(timeseries, participant_id, smooth=smooth)
    for key in ["wrist_vx_release", "wrist_vy_release", "wrist_vz_release",
                "velocity_magnitude_release", "elevation_angle", "azimuth_angle",
                "elbow_alignment", "arm_extension_angle"]:
        if key in physics:
            features[key] = physics[key]

    # Participant
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    return features


# =============================================================================
# F7: Z-Coordinates Only (~345 features)
# =============================================================================

def extract_f7_z_coords_only(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F7: Only z-coordinates with comprehensive statistics.

    Z-coordinates (heights) were top features in baseline analysis.
    ~345 features (69 keypoints x 5 stats).
    """
    ensure_keypoint_mapping()
    features = {}

    # All z-coordinates are at indices 2, 5, 8, 11, ... (every 3rd starting at 2)
    n_keypoints = timeseries.shape[1] // 3

    for kp_idx in range(n_keypoints):
        z_idx = kp_idx * 3 + 2
        z_series = timeseries[:, z_idx]

        if smooth:
            try:
                z_series = savgol_filter(z_series, 5, 2)
            except (ValueError, np.linalg.LinAlgError):
                pass

        valid = z_series[~np.isnan(z_series)]
        if len(valid) == 0:
            continue

        prefix = f"kp{kp_idx}_z"
        features[f"{prefix}_mean"] = np.mean(valid)
        features[f"{prefix}_std"] = np.std(valid)
        features[f"{prefix}_min"] = np.min(valid)
        features[f"{prefix}_max"] = np.max(valid)
        features[f"{prefix}_range"] = np.max(valid) - np.min(valid)

    # Participant
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    return features


# =============================================================================
# F8: Velocity Focused (~80 features)
# =============================================================================

def extract_f8_velocity_focused(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    smooth: bool = False
) -> Dict[str, float]:
    """
    F8: Velocity and acceleration focused features.

    Ball trajectory is determined by velocity at release.
    ~80 features focusing on movement dynamics.
    """
    ensure_keypoint_mapping()
    features = {}

    # Key body parts for shooting motion
    key_keypoints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "right_knee", "right_hip", "mid_hip", "left_wrist"
    ]

    release_frame = detect_release_frame(timeseries, smooth=smooth)

    for kp_name in key_keypoints:
        try:
            data = get_keypoint_data(timeseries, kp_name)
            vel = compute_velocity(data, smooth=smooth)
            acc = compute_acceleration(data, smooth=smooth)

            vel_mag = np.linalg.norm(vel, axis=1)
            acc_mag = np.linalg.norm(acc, axis=1)

            # Global velocity stats
            features[f"{kp_name}_vel_mean"] = np.nanmean(vel_mag)
            features[f"{kp_name}_vel_max"] = np.nanmax(vel_mag)
            features[f"{kp_name}_vel_std"] = np.nanstd(vel_mag)

            # Velocity at release
            features[f"{kp_name}_vel_at_release"] = vel_mag[release_frame]

            # Time to peak velocity (normalized)
            peak_frame = np.nanargmax(vel_mag)
            features[f"{kp_name}_time_to_peak_vel"] = peak_frame / NUM_FRAMES

            # Acceleration stats
            features[f"{kp_name}_acc_mean"] = np.nanmean(acc_mag)
            features[f"{kp_name}_acc_max"] = np.nanmax(acc_mag)
            features[f"{kp_name}_acc_at_release"] = acc_mag[release_frame]

            # Direction at release (for key points)
            if kp_name == "right_wrist":
                features["wrist_vx_release"] = vel[release_frame, 0]
                features["wrist_vy_release"] = vel[release_frame, 1]
                features["wrist_vz_release"] = vel[release_frame, 2]

                # Elevation and azimuth
                vel_horiz = np.sqrt(vel[release_frame, 0]**2 + vel[release_frame, 1]**2)
                features["elevation_angle"] = np.arctan2(vel[release_frame, 2], vel_horiz) * 180 / np.pi
                features["azimuth_angle"] = np.arctan2(vel[release_frame, 0], vel[release_frame, 1]) * 180 / np.pi

        except (KeyError, IndexError):
            continue

    # Joint angular velocities
    joint_configs = [
        ("right_elbow", "right_shoulder", "right_elbow", "right_wrist"),
        ("right_knee", "right_hip", "right_knee", "right_ankle"),
    ]

    for joint_name, kp1, kp2, kp3 in joint_configs:
        try:
            p1 = get_keypoint_data(timeseries, kp1)
            p2 = get_keypoint_data(timeseries, kp2)
            p3 = get_keypoint_data(timeseries, kp3)

            angle = compute_joint_angle(p1, p2, p3)
            ang_vel = np.gradient(angle, DT)

            features[f"{joint_name}_angvel_mean"] = np.nanmean(np.abs(ang_vel))
            features[f"{joint_name}_angvel_max"] = np.nanmax(np.abs(ang_vel))
            features[f"{joint_name}_angvel_at_release"] = ang_vel[release_frame]

        except (KeyError, IndexError):
            continue

    # Participant
    if participant_id is not None:
        features["participant_id"] = participant_id
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    return features


# =============================================================================
# Factory Interface
# =============================================================================

def get_feature_extractor(feature_set_id: str) -> Callable:
    """
    Get the feature extractor function for a given feature set ID.

    Args:
        feature_set_id: One of F1-F8 or the full name

    Returns:
        Feature extractor function
    """
    extractors = {
        "F1": extract_f1_tier1_baseline,
        "tier1_baseline": extract_f1_tier1_baseline,
        "F2": extract_f2_physics_only,
        "physics_only": extract_f2_physics_only,
        "F3": extract_f3_hybrid,
        "hybrid": extract_f3_hybrid,
        "F4": extract_f4_hybrid_with_pid,
        "hybrid_with_pid": extract_f4_hybrid_with_pid,
        "F5": extract_f5_top100_importance,
        "top100_importance": extract_f5_top100_importance,
        "F6": extract_f6_release_window,
        "release_window": extract_f6_release_window,
        "F7": extract_f7_z_coords_only,
        "z_coords_only": extract_f7_z_coords_only,
        "F8": extract_f8_velocity_focused,
        "velocity_focused": extract_f8_velocity_focused,
    }

    if feature_set_id not in extractors:
        raise ValueError(f"Unknown feature set: {feature_set_id}. Choose from {list(FEATURE_SETS.keys())}")

    return extractors[feature_set_id]


def get_feature_set_description(feature_set_id: str) -> str:
    """Get description of a feature set."""
    descriptions = {
        "F1": "Tier 1 baseline: ~3365 features (full statistical)",
        "F2": "Physics only: ~40 features (release mechanics)",
        "F3": "Hybrid: ~130 features (physics + z-stats)",
        "F4": "Hybrid + PID: ~136 features (hybrid + participant one-hot)",
        "F5": "Top 100: 100 features (by importance)",
        "F6": "Release window: ~80 features (frames 100-180)",
        "F7": "Z-coords only: ~345 features (heights)",
        "F8": "Velocity focused: ~80 features (dynamics)",
    }
    return descriptions.get(feature_set_id, "Unknown feature set")


if __name__ == "__main__":
    # Test feature extraction
    try:
        from data_loader import load_single_shot
    except ImportError:
        from src.data_loader import load_single_shot

    print("Testing feature factory...")

    # Initialize
    ensure_keypoint_mapping()

    # Load a test shot
    metadata, timeseries = load_single_shot(0, train=True)
    print(f"Loaded shot: {metadata['id']}")

    # Test each feature set
    for fset_id in FEATURE_SETS.keys():
        extractor = get_feature_extractor(fset_id)
        features = extractor(timeseries, metadata["participant_id"], smooth=False)
        print(f"\n{fset_id} ({FEATURE_SETS[fset_id]}): {len(features)} features")

    print("\nFeature factory test complete!")
