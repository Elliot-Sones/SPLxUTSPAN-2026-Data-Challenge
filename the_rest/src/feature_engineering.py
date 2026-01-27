"""
Feature engineering for biomechanics time series data.

Extracts statistical, physics-based, and biomechanics-specific features
from motion capture data.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle

# Feature extraction constants
FRAME_RATE = 60
NUM_FRAMES = 240
DT = 1.0 / FRAME_RATE  # Time step in seconds


# Keypoint indices for important body parts (will be set during initialization)
KEYPOINT_NAMES = None
KEYPOINT_INDEX = None


def init_keypoint_mapping(keypoint_cols: List[str]):
    """Initialize keypoint name to index mapping."""
    global KEYPOINT_NAMES, KEYPOINT_INDEX

    # Extract unique keypoint names (remove _x, _y, _z suffix)
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
    # Each keypoint has 3 consecutive columns (x, y, z)
    return timeseries[:, idx*3:(idx+1)*3]


# =============================================================================
# Tier 1: Basic Statistical Features
# =============================================================================

def extract_basic_stats(series: np.ndarray) -> Dict[str, float]:
    """Extract basic statistical features from a 1D time series."""
    # Handle NaN values
    valid = series[~np.isnan(series)]
    if len(valid) == 0:
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "range": np.nan, "median": np.nan, "q10": np.nan, "q25": np.nan,
            "q75": np.nan, "q90": np.nan, "iqr": np.nan, "energy": np.nan,
            "first": np.nan, "last": np.nan
        }

    return {
        "mean": np.mean(valid),
        "std": np.std(valid),
        "min": np.min(valid),
        "max": np.max(valid),
        "range": np.max(valid) - np.min(valid),
        "median": np.median(valid),
        "q10": np.percentile(valid, 10),
        "q25": np.percentile(valid, 25),
        "q75": np.percentile(valid, 75),
        "q90": np.percentile(valid, 90),
        "iqr": np.percentile(valid, 75) - np.percentile(valid, 25),
        "energy": np.sum(valid ** 2),
        "first": series[0] if not np.isnan(series[0]) else np.nan,
        "last": series[-1] if not np.isnan(series[-1]) else np.nan,
    }


def extract_tier1_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract Tier 1 features: basic statistics for all 207 time series.

    Args:
        timeseries: (240, 207) array

    Returns:
        Dictionary with ~2800 features (207 series * ~14 stats)
    """
    features = {}
    n_features = timeseries.shape[1]

    for i in range(n_features):
        series = timeseries[:, i]
        stats_dict = extract_basic_stats(series)
        for stat_name, value in stats_dict.items():
            features[f"f{i}_{stat_name}"] = value

    return features


# =============================================================================
# Tier 1b: Velocity and Acceleration Features
# =============================================================================

def compute_velocity(series: np.ndarray) -> np.ndarray:
    """Compute velocity (first derivative) using central differences."""
    velocity = np.gradient(series, DT)
    return velocity


def compute_acceleration(series: np.ndarray) -> np.ndarray:
    """Compute acceleration (second derivative)."""
    velocity = compute_velocity(series)
    acceleration = np.gradient(velocity, DT)
    return acceleration


def extract_velocity_features(timeseries: np.ndarray, key_indices: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Extract velocity-based features.

    Args:
        timeseries: (240, 207) array
        key_indices: Optional list of feature indices to process (for efficiency)

    Returns:
        Dictionary with velocity features
    """
    features = {}

    if key_indices is None:
        # Process all features (expensive)
        key_indices = range(timeseries.shape[1])

    for i in key_indices:
        series = timeseries[:, i]
        if np.isnan(series).all():
            continue

        vel = compute_velocity(series)
        valid_vel = vel[~np.isnan(vel)]

        if len(valid_vel) == 0:
            continue

        features[f"f{i}_vel_mean"] = np.mean(valid_vel)
        features[f"f{i}_vel_std"] = np.std(valid_vel)
        features[f"f{i}_vel_max"] = np.max(valid_vel)
        features[f"f{i}_vel_min"] = np.min(valid_vel)
        features[f"f{i}_vel_max_abs"] = np.max(np.abs(valid_vel))

        # Time of max velocity (proxy for release timing)
        max_vel_idx = np.argmax(np.abs(vel))
        features[f"f{i}_vel_max_time"] = max_vel_idx / FRAME_RATE

    return features


def extract_acceleration_features(timeseries: np.ndarray, key_indices: Optional[List[int]] = None) -> Dict[str, float]:
    """Extract acceleration-based features."""
    features = {}

    if key_indices is None:
        key_indices = range(timeseries.shape[1])

    for i in key_indices:
        series = timeseries[:, i]
        if np.isnan(series).all():
            continue

        acc = compute_acceleration(series)
        valid_acc = acc[~np.isnan(acc)]

        if len(valid_acc) == 0:
            continue

        features[f"f{i}_acc_mean"] = np.mean(valid_acc)
        features[f"f{i}_acc_std"] = np.std(valid_acc)
        features[f"f{i}_acc_max"] = np.max(valid_acc)
        features[f"f{i}_acc_min"] = np.min(valid_acc)

    return features


# =============================================================================
# Tier 2: Biomechanics-Specific Features
# =============================================================================

def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Compute angle at joint p2 formed by p1-p2-p3.

    Args:
        p1, p2, p3: (240, 3) arrays representing 3D positions over time

    Returns:
        (240,) array of angles in degrees
    """
    v1 = p1 - p2  # Vector from p2 to p1
    v2 = p3 - p2  # Vector from p2 to p3

    # Compute angle using dot product
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)

    # Avoid division by zero
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10

    cos_angle = np.clip(dot / denom, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi

    return angle


def extract_joint_angle_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract joint angle features for key biomechanical joints.

    Requires init_keypoint_mapping to have been called.
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    # Define joints: (joint_name, point1, center, point2)
    joints = [
        ("right_elbow", "right_shoulder", "right_elbow", "right_wrist"),
        ("left_elbow", "left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "neck", "right_shoulder", "right_elbow"),
        ("left_shoulder", "neck", "left_shoulder", "left_elbow"),
        ("right_knee", "right_hip", "right_knee", "right_ankle"),
        ("left_knee", "left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_shoulder", "right_hip", "right_knee"),
        ("left_hip", "left_shoulder", "left_hip", "left_knee"),
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

            # Angular velocity
            ang_vel = compute_velocity(angle)
            valid_vel = ang_vel[~np.isnan(ang_vel)]
            if len(valid_vel) > 0:
                features[f"{joint_name}_angvel_max"] = np.max(np.abs(valid_vel))
                features[f"{joint_name}_angvel_mean"] = np.mean(valid_vel)

        except (KeyError, IndexError):
            continue

    return features


def detect_release_frame(timeseries: np.ndarray) -> int:
    """
    Detect the ball release frame.

    Uses maximum right wrist VELOCITY MAGNITUDE as proxy for release point.
    Release occurs at peak velocity, not peak height.
    """
    if KEYPOINT_INDEX is None:
        return NUM_FRAMES // 2  # Default to middle

    try:
        wrist_data = get_keypoint_data(timeseries, "right_wrist")
        # Compute velocity in all 3 axes
        vel = np.gradient(wrist_data, DT, axis=0)
        # Compute velocity magnitude
        vel_magnitude = np.linalg.norm(vel, axis=1)
        # Search from 1/3 into the shot (release typically in second half)
        search_start = NUM_FRAMES // 3
        release_idx = search_start + np.argmax(vel_magnitude[search_start:])
        return int(release_idx)
    except (KeyError, IndexError):
        return NUM_FRAMES // 2


def extract_release_features(timeseries: np.ndarray) -> Dict[str, float]:
    """Extract features at the detected release point."""
    features = {}

    release_frame = detect_release_frame(timeseries)
    features["release_frame"] = release_frame
    features["release_time"] = release_frame / FRAME_RATE

    if KEYPOINT_INDEX is None:
        return features

    # Key body parts at release
    key_parts = ["right_wrist", "right_elbow", "right_shoulder", "right_hip", "neck"]

    for part in key_parts:
        try:
            data = get_keypoint_data(timeseries, part)

            # Position at release
            features[f"{part}_release_x"] = data[release_frame, 0]
            features[f"{part}_release_y"] = data[release_frame, 1]
            features[f"{part}_release_z"] = data[release_frame, 2]

            # Velocity at release
            vel_x = compute_velocity(data[:, 0])
            vel_y = compute_velocity(data[:, 1])
            vel_z = compute_velocity(data[:, 2])

            features[f"{part}_release_vel_x"] = vel_x[release_frame]
            features[f"{part}_release_vel_y"] = vel_y[release_frame]
            features[f"{part}_release_vel_z"] = vel_z[release_frame]

            # CRITICAL: Release velocity magnitude (for depth prediction)
            vel_magnitude = np.sqrt(vel_x[release_frame]**2 + vel_y[release_frame]**2 + vel_z[release_frame]**2)
            features[f"{part}_release_vel_magnitude"] = vel_magnitude

        except (KeyError, IndexError):
            continue

    # === NEW BIOMECHANICS FEATURES ===

    try:
        # Get key body part positions
        wrist = get_keypoint_data(timeseries, "right_wrist")
        elbow = get_keypoint_data(timeseries, "right_elbow")
        shoulder = get_keypoint_data(timeseries, "right_shoulder")

        # 1. WRIST SNAP ANGLE (for angle prediction)
        # Try to use finger tip if available, otherwise use wrist-elbow extension
        try:
            finger = get_keypoint_data(timeseries, "right_first_finger_distal")
            wrist_snap = compute_joint_angle(elbow, wrist, finger)
            features["wrist_snap_angle_at_release"] = wrist_snap[release_frame]
            features["wrist_snap_angle_mean"] = np.nanmean(wrist_snap)
            features["wrist_snap_angle_max"] = np.nanmax(wrist_snap)
        except (KeyError, IndexError):
            pass

        # 2. ELBOW ALIGNMENT (for left_right prediction)
        # Is the elbow under the ball (centered) or out to the side?
        # Compute lateral deviation: elbow_x relative to shoulder-wrist line
        shoulder_x = shoulder[release_frame, 0]
        wrist_x = wrist[release_frame, 0]
        elbow_x = elbow[release_frame, 0]
        midpoint_x = (shoulder_x + wrist_x) / 2
        features["elbow_lateral_deviation"] = elbow_x - midpoint_x

        # Wrist lateral position relative to shoulder (for left_right)
        features["wrist_lateral_from_shoulder"] = wrist_x - shoulder_x

        # 3. ARM EXTENSION at release (for angle prediction)
        # Distance from shoulder to wrist
        arm_vector = wrist[release_frame] - shoulder[release_frame]
        arm_extension = np.linalg.norm(arm_vector)
        features["arm_extension_at_release"] = arm_extension

        # 4. RELEASE VELOCITY DIRECTION (for angle prediction)
        # Elevation angle of velocity vector
        wrist_vel = np.array([
            compute_velocity(wrist[:, 0])[release_frame],
            compute_velocity(wrist[:, 1])[release_frame],
            compute_velocity(wrist[:, 2])[release_frame]
        ])
        vel_horiz = np.sqrt(wrist_vel[0]**2 + wrist_vel[1]**2)
        vel_vert = wrist_vel[2]
        release_elevation_angle = np.arctan2(vel_vert, vel_horiz) * 180 / np.pi
        features["release_elevation_angle"] = release_elevation_angle

        # 5. TRUNK ANGLE (for stability/consistency)
        try:
            left_shoulder = get_keypoint_data(timeseries, "left_shoulder")
            mid_hip = get_keypoint_data(timeseries, "mid_hip")
            # Trunk lean: angle of shoulder midpoint relative to hip in vertical plane
            shoulder_mid = (shoulder[release_frame] + left_shoulder[release_frame]) / 2
            trunk_vector = shoulder_mid - mid_hip[release_frame]
            trunk_lean = np.arctan2(trunk_vector[2], np.sqrt(trunk_vector[0]**2 + trunk_vector[1]**2)) * 180 / np.pi
            features["trunk_lean_at_release"] = trunk_lean
        except (KeyError, IndexError):
            pass

    except (KeyError, IndexError):
        pass

    return features


def extract_propulsion_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract features from the propulsion phase (frames 120-180).
    Critical for depth prediction - leg drive generates power.
    """
    features = {}

    if KEYPOINT_INDEX is None:
        return features

    try:
        # Get knee data during propulsion phase
        right_knee = get_keypoint_data(timeseries, "right_knee")
        right_hip = get_keypoint_data(timeseries, "right_hip")
        right_ankle = get_keypoint_data(timeseries, "right_ankle")

        # Compute knee angle over time
        knee_angle = compute_joint_angle(right_hip, right_knee, right_ankle)

        # Propulsion phase: frames 120-180
        prop_start, prop_end = 120, 180
        prop_angle = knee_angle[prop_start:prop_end]

        # KNEE EXTENSION RATE (for depth prediction)
        # Faster extension = more power = longer shot
        knee_extension_rate = compute_velocity(knee_angle)
        prop_extension_rate = knee_extension_rate[prop_start:prop_end]

        features["knee_angle_propulsion_mean"] = np.nanmean(prop_angle)
        features["knee_angle_propulsion_range"] = np.nanmax(prop_angle) - np.nanmin(prop_angle)
        features["knee_extension_rate_max"] = np.nanmax(prop_extension_rate)
        features["knee_extension_rate_mean"] = np.nanmean(prop_extension_rate)

        # Vertical center of mass trajectory (proxy using mid_hip)
        mid_hip = get_keypoint_data(timeseries, "mid_hip")
        hip_z = mid_hip[:, 2]
        hip_z_vel = compute_velocity(hip_z)

        # Max upward velocity during propulsion
        features["hip_vertical_vel_max"] = np.nanmax(hip_z_vel[prop_start:prop_end])
        features["hip_vertical_displacement"] = hip_z[prop_end] - hip_z[prop_start]

    except (KeyError, IndexError):
        pass

    return features


# =============================================================================
# Tier 3: Phase-Based Features
# =============================================================================

def extract_phase_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract features for each phase of the shot.

    Phases:
    - Phase 0 (frames 0-60): Preparation
    - Phase 1 (frames 60-120): Loading
    - Phase 2 (frames 120-180): Propulsion
    - Phase 3 (frames 180-240): Release/Follow-through
    """
    features = {}
    phase_bounds = [(0, 60), (60, 120), (120, 180), (180, 240)]

    if KEYPOINT_INDEX is None:
        return features

    # Focus on key body parts for phase features
    key_parts = ["right_wrist", "right_elbow", "right_shoulder", "right_knee"]

    for part in key_parts:
        try:
            data = get_keypoint_data(timeseries, part)

            for phase_idx, (start, end) in enumerate(phase_bounds):
                phase_data = data[start:end]

                for coord_idx, coord in enumerate(["x", "y", "z"]):
                    series = phase_data[:, coord_idx]
                    valid = series[~np.isnan(series)]

                    if len(valid) == 0:
                        continue

                    features[f"{part}_{coord}_phase{phase_idx}_mean"] = np.mean(valid)
                    features[f"{part}_{coord}_phase{phase_idx}_std"] = np.std(valid)
                    features[f"{part}_{coord}_phase{phase_idx}_range"] = np.max(valid) - np.min(valid)

        except (KeyError, IndexError):
            continue

    return features


# =============================================================================
# Master Feature Extraction
# =============================================================================

def extract_all_features(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    tiers: List[int] = [1, 2, 3]
) -> Dict[str, float]:
    """
    Extract all features from a single shot.

    Args:
        timeseries: (240, 207) array
        participant_id: Optional participant ID for participant-specific features
        tiers: Which feature tiers to extract (1, 2, 3)

    Returns:
        Dictionary of features
    """
    features = {}

    # Add participant as a feature (for models that can use it)
    if participant_id is not None:
        features["participant_id"] = participant_id
        # One-hot encoding for participant (helps tree models)
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    # Tier 1: Basic statistics
    if 1 in tiers:
        features.update(extract_tier1_features(timeseries))

        # Key feature indices for velocity/acceleration
        # Focus on shooting-relevant body parts only (no duplicates)
        # right_shoulder (6), right_elbow (8), right_wrist (10)
        # right_hip (12), right_knee (14), neck (22), mid_hip (21)
        key_keypoint_ids = [6, 8, 10, 12, 14, 21, 22]  # Key body parts
        key_indices = []
        for kp_id in key_keypoint_ids:
            key_indices.extend([kp_id * 3, kp_id * 3 + 1, kp_id * 3 + 2])

        features.update(extract_velocity_features(timeseries, key_indices))
        features.update(extract_acceleration_features(timeseries, key_indices))

    # Tier 2: Biomechanics-specific
    if 2 in tiers:
        features.update(extract_joint_angle_features(timeseries))
        features.update(extract_release_features(timeseries))
        features.update(extract_propulsion_features(timeseries))

    # Tier 3: Phase-based
    if 3 in tiers:
        features.update(extract_phase_features(timeseries))

    return features


def features_to_array(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """Convert feature dictionary to numpy array with consistent ordering."""
    return np.array([features.get(name, np.nan) for name in feature_names], dtype=np.float32)


def save_features(features_list: List[Dict], filepath: Path):
    """Save extracted features to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(features_list, f)


def load_features(filepath: Path) -> List[Dict]:
    """Load extracted features from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Test feature extraction
    from data_loader import load_single_shot, get_keypoint_columns

    print("Testing feature engineering...")

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    print(f"Initialized {len(KEYPOINT_NAMES)} keypoints")

    # Load a single shot
    metadata, timeseries = load_single_shot(0, train=True)
    print(f"Loaded shot: {metadata['id']}")
    print(f"Timeseries shape: {timeseries.shape}")

    # Extract features
    print("\nExtracting Tier 1 features...")
    t1_features = extract_tier1_features(timeseries)
    print(f"Tier 1 features: {len(t1_features)}")

    print("\nExtracting Tier 2 features...")
    t2_features = extract_joint_angle_features(timeseries)
    t2_features.update(extract_release_features(timeseries))
    print(f"Tier 2 features: {len(t2_features)}")

    print("\nExtracting Tier 3 features...")
    t3_features = extract_phase_features(timeseries)
    print(f"Tier 3 features: {len(t3_features)}")

    print("\nExtracting all features...")
    all_features = extract_all_features(timeseries, metadata["participant_id"])
    print(f"Total features: {len(all_features)}")

    # Check for NaN values
    nan_count = sum(1 for v in all_features.values() if np.isnan(v))
    print(f"Features with NaN: {nan_count}")

    print("\nFeature engineering test complete!")
