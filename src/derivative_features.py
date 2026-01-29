"""
Derivative feature extraction from raw 240x207 frame sequences.

Extracts velocity (1st derivative), acceleration (2nd derivative),
and jerk (3rd derivative) features that capture motion dynamics.
Also extracts features at critical frames identified by prior analysis:
- Frame 102: peak R^2 for depth
- Frame 153: peak R^2 for angle
- Frame 237: peak R^2 for left_right
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.signal import savgol_filter

# Constants
FRAME_RATE = 60
NUM_FRAMES = 240
DT = 1.0 / FRAME_RATE

# Critical frames identified by R^2 analysis
CRITICAL_FRAMES = {
    "depth": 102,      # 1.70s - left hand guide
    "angle": 153,      # 2.55s - lower body stance
    "left_right": 237  # 3.95s - finger control
}

# Key keypoints for motion analysis
KEY_KEYPOINT_INDICES = {
    "right_wrist": 16,
    "right_elbow": 14,
    "right_shoulder": 12,
    "left_wrist": 15,
    "left_elbow": 13,
    "left_shoulder": 11,
    "right_hip": 24,
    "left_hip": 23,
    "mid_hip": 0,  # Placeholder - may vary
    "right_knee": 26,
    "left_knee": 25,
    "right_ankle": 28,
    "left_ankle": 27,
    "neck": 0,  # Placeholder
}


def _smooth_series(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply Savitzky-Golay smoothing."""
    if window < 3 or np.any(np.isnan(series)):
        return series
    window = window if window % 2 == 1 else window + 1
    try:
        if series.ndim == 1:
            return savgol_filter(series, window, 2)
        else:
            return savgol_filter(series, window, 2, axis=0)
    except (ValueError, np.linalg.LinAlgError):
        return series


def _compute_derivative(series: np.ndarray, smooth: bool = True) -> np.ndarray:
    """Compute first derivative (velocity)."""
    if smooth:
        series = _smooth_series(series)
    return np.gradient(series, DT, axis=0) if series.ndim > 1 else np.gradient(series, DT)


def _compute_magnitude(vector_series: np.ndarray) -> np.ndarray:
    """Compute magnitude of vector time series."""
    return np.linalg.norm(vector_series, axis=1)


def _safe_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute safe statistics handling NaN values."""
    features = {}
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return features

    features[f"{prefix}_mean"] = float(np.mean(valid))
    features[f"{prefix}_std"] = float(np.std(valid))
    features[f"{prefix}_max"] = float(np.max(valid))
    features[f"{prefix}_min"] = float(np.min(valid))
    features[f"{prefix}_range"] = float(np.max(valid) - np.min(valid))

    # Time to max (normalized 0-1)
    max_idx = np.nanargmax(arr)
    features[f"{prefix}_time_to_max"] = float(max_idx / len(arr))

    return features


def extract_velocity_features(
    timeseries: np.ndarray,
    keypoint_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Extract velocity (1st derivative) features from raw sequences.

    Args:
        timeseries: (240, 207) array of raw keypoint positions
        keypoint_indices: List of keypoint indices to process (default: key body parts)

    Returns:
        Dictionary of velocity features
    """
    features = {}

    # Default to key keypoints
    if keypoint_indices is None:
        # Use every 5th keypoint to keep feature count manageable
        keypoint_indices = list(range(0, 69, 5))

    for kp_idx in keypoint_indices:
        # Extract x, y, z for this keypoint
        base_idx = kp_idx * 3
        if base_idx + 3 > timeseries.shape[1]:
            continue

        kp_data = timeseries[:, base_idx:base_idx + 3]

        # Compute velocity
        velocity = _compute_derivative(kp_data, smooth=True)
        vel_magnitude = _compute_magnitude(velocity)

        prefix = f"kp{kp_idx}_vel"

        # Global velocity stats
        features.update(_safe_stats(vel_magnitude, prefix))

        # Velocity at critical frames
        for target, frame in CRITICAL_FRAMES.items():
            if frame < len(vel_magnitude):
                features[f"{prefix}_at_f{frame}"] = float(vel_magnitude[frame])

        # Per-axis velocity stats
        for axis_idx, axis in enumerate(["x", "y", "z"]):
            axis_vel = velocity[:, axis_idx]
            features.update(_safe_stats(axis_vel, f"{prefix}_{axis}"))

    return features


def extract_acceleration_features(
    timeseries: np.ndarray,
    keypoint_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Extract acceleration (2nd derivative) features from raw sequences.

    Args:
        timeseries: (240, 207) array of raw keypoint positions
        keypoint_indices: List of keypoint indices to process

    Returns:
        Dictionary of acceleration features
    """
    features = {}

    if keypoint_indices is None:
        keypoint_indices = list(range(0, 69, 5))

    for kp_idx in keypoint_indices:
        base_idx = kp_idx * 3
        if base_idx + 3 > timeseries.shape[1]:
            continue

        kp_data = timeseries[:, base_idx:base_idx + 3]

        # Compute velocity then acceleration
        velocity = _compute_derivative(kp_data, smooth=True)
        acceleration = _compute_derivative(velocity, smooth=False)
        acc_magnitude = _compute_magnitude(acceleration)

        prefix = f"kp{kp_idx}_acc"

        # Global acceleration stats
        features.update(_safe_stats(acc_magnitude, prefix))

        # Acceleration at critical frames
        for target, frame in CRITICAL_FRAMES.items():
            if frame < len(acc_magnitude):
                features[f"{prefix}_at_f{frame}"] = float(acc_magnitude[frame])

        # Per-axis acceleration stats
        for axis_idx, axis in enumerate(["x", "y", "z"]):
            axis_acc = acceleration[:, axis_idx]
            features.update(_safe_stats(axis_acc, f"{prefix}_{axis}"))

    return features


def extract_jerk_features(
    timeseries: np.ndarray,
    keypoint_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Extract jerk (3rd derivative) features from raw sequences.
    Jerk measures smoothness - lower jerk = smoother motion.

    Args:
        timeseries: (240, 207) array of raw keypoint positions
        keypoint_indices: List of keypoint indices to process

    Returns:
        Dictionary of jerk features
    """
    features = {}

    if keypoint_indices is None:
        keypoint_indices = list(range(0, 69, 5))

    for kp_idx in keypoint_indices:
        base_idx = kp_idx * 3
        if base_idx + 3 > timeseries.shape[1]:
            continue

        kp_data = timeseries[:, base_idx:base_idx + 3]

        # Compute velocity -> acceleration -> jerk
        velocity = _compute_derivative(kp_data, smooth=True)
        acceleration = _compute_derivative(velocity, smooth=False)
        jerk = _compute_derivative(acceleration, smooth=False)
        jerk_magnitude = _compute_magnitude(jerk)

        prefix = f"kp{kp_idx}_jerk"

        # Jerk stats (lower = smoother motion)
        features.update(_safe_stats(jerk_magnitude, prefix))

        # Jerk at critical frames
        for target, frame in CRITICAL_FRAMES.items():
            if frame < len(jerk_magnitude):
                features[f"{prefix}_at_f{frame}"] = float(jerk_magnitude[frame])

    return features


def extract_critical_frame_features(
    timeseries: np.ndarray,
    window_size: int = 5
) -> Dict[str, float]:
    """
    Extract raw values and windowed stats around critical frames.

    Args:
        timeseries: (240, 207) array of raw keypoint positions
        window_size: Number of frames on each side for windowed stats

    Returns:
        Dictionary of critical frame features
    """
    features = {}

    # Key keypoints for critical frames
    key_kp_indices = [0, 5, 10, 15, 16, 20, 25, 30]  # Subset for efficiency

    for frame_name, frame_idx in CRITICAL_FRAMES.items():
        # Raw values at critical frame
        for kp_idx in key_kp_indices:
            base_idx = kp_idx * 3
            if base_idx + 3 > timeseries.shape[1]:
                continue

            for axis_idx, axis in enumerate(["x", "y", "z"]):
                col_idx = base_idx + axis_idx
                if col_idx < timeseries.shape[1]:
                    features[f"f{frame_idx}_kp{kp_idx}_{axis}"] = float(
                        timeseries[frame_idx, col_idx]
                    )

        # Windowed stats around critical frame
        start = max(0, frame_idx - window_size)
        end = min(NUM_FRAMES, frame_idx + window_size + 1)

        for kp_idx in key_kp_indices:
            base_idx = kp_idx * 3
            if base_idx + 3 > timeseries.shape[1]:
                continue

            window_data = timeseries[start:end, base_idx:base_idx + 3]
            window_magnitude = np.linalg.norm(window_data, axis=1)

            prefix = f"f{frame_idx}_kp{kp_idx}_window"
            valid = window_magnitude[~np.isnan(window_magnitude)]
            if len(valid) > 0:
                features[f"{prefix}_mean"] = float(np.mean(valid))
                features[f"{prefix}_std"] = float(np.std(valid))

    return features


def extract_frame_difference_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract inter-frame differences at critical points.

    These capture how the motion changes between the key moments
    for each target prediction.

    Args:
        timeseries: (240, 207) array of raw keypoint positions

    Returns:
        Dictionary of frame difference features
    """
    features = {}

    f_depth = CRITICAL_FRAMES["depth"]      # 102
    f_angle = CRITICAL_FRAMES["angle"]      # 153
    f_lr = CRITICAL_FRAMES["left_right"]    # 237

    # Key keypoints for difference computation
    key_kp_indices = [0, 5, 10, 15, 16, 20, 25, 30]

    # Frame differences
    frame_pairs = [
        ("depth_to_angle", f_depth, f_angle),
        ("angle_to_lr", f_angle, f_lr),
        ("depth_to_lr", f_depth, f_lr),
    ]

    for pair_name, f1, f2 in frame_pairs:
        for kp_idx in key_kp_indices:
            base_idx = kp_idx * 3
            if base_idx + 3 > timeseries.shape[1]:
                continue

            # Position difference
            pos1 = timeseries[f1, base_idx:base_idx + 3]
            pos2 = timeseries[f2, base_idx:base_idx + 3]
            diff = pos2 - pos1

            features[f"diff_{pair_name}_kp{kp_idx}_mag"] = float(np.linalg.norm(diff))

            for axis_idx, axis in enumerate(["x", "y", "z"]):
                features[f"diff_{pair_name}_kp{kp_idx}_{axis}"] = float(diff[axis_idx])

    # Velocity at each critical frame
    for frame_name, frame_idx in CRITICAL_FRAMES.items():
        # Compute velocity as finite difference around frame
        if frame_idx > 0 and frame_idx < NUM_FRAMES - 1:
            for kp_idx in key_kp_indices:
                base_idx = kp_idx * 3
                if base_idx + 3 > timeseries.shape[1]:
                    continue

                vel = (timeseries[frame_idx + 1, base_idx:base_idx + 3] -
                       timeseries[frame_idx - 1, base_idx:base_idx + 3]) / (2 * DT)
                features[f"vel_{frame_name}_kp{kp_idx}_mag"] = float(np.linalg.norm(vel))

    return features


def extract_target_specific_window_features(
    timeseries: np.ndarray,
    target: str
) -> Dict[str, float]:
    """
    Extract features from target-specific frame windows.

    Based on R^2 analysis:
    - depth: frames 50-150 (peaks at 102)
    - angle: frames 100-175 (peaks at 153)
    - left_right: frames 175-240 (peaks at 237)

    Args:
        timeseries: (240, 207) array of raw keypoint positions
        target: One of "depth", "angle", "left_right"

    Returns:
        Dictionary of window-specific features
    """
    features = {}

    # Target-specific windows
    windows = {
        "depth": (50, 150),
        "angle": (100, 175),
        "left_right": (175, 240),
    }

    if target not in windows:
        return features

    start, end = windows[target]
    window_ts = timeseries[start:end]

    # Key keypoints
    key_kp_indices = list(range(0, 69, 5))

    for kp_idx in key_kp_indices:
        base_idx = kp_idx * 3
        if base_idx + 3 > timeseries.shape[1]:
            continue

        kp_data = window_ts[:, base_idx:base_idx + 3]
        prefix = f"{target}_win_kp{kp_idx}"

        # Position stats in window
        for axis_idx, axis in enumerate(["x", "y", "z"]):
            axis_data = kp_data[:, axis_idx]
            valid = axis_data[~np.isnan(axis_data)]
            if len(valid) > 0:
                features[f"{prefix}_{axis}_mean"] = float(np.mean(valid))
                features[f"{prefix}_{axis}_std"] = float(np.std(valid))
                features[f"{prefix}_{axis}_range"] = float(np.max(valid) - np.min(valid))

        # Velocity in window
        velocity = _compute_derivative(kp_data, smooth=True)
        vel_magnitude = _compute_magnitude(velocity)
        valid_vel = vel_magnitude[~np.isnan(vel_magnitude)]
        if len(valid_vel) > 0:
            features[f"{prefix}_vel_mean"] = float(np.mean(valid_vel))
            features[f"{prefix}_vel_max"] = float(np.max(valid_vel))

    return features


def extract_all_derivative_features(
    timeseries: np.ndarray,
    participant_id: Optional[int] = None,
    include_velocity: bool = True,
    include_acceleration: bool = True,
    include_jerk: bool = True,
    include_critical_frames: bool = True,
    include_frame_diffs: bool = True,
) -> Dict[str, float]:
    """
    Extract all derivative-based features from raw sequence.

    Args:
        timeseries: (240, 207) array of raw keypoint positions
        participant_id: Optional participant ID for one-hot encoding
        include_*: Flags to control which feature groups to include

    Returns:
        Dictionary of all derivative features
    """
    features = {}

    # Use subset of keypoints for efficiency
    kp_indices = list(range(0, 69, 5))  # Every 5th keypoint = 14 keypoints

    if include_velocity:
        features.update(extract_velocity_features(timeseries, kp_indices))

    if include_acceleration:
        features.update(extract_acceleration_features(timeseries, kp_indices))

    if include_jerk:
        features.update(extract_jerk_features(timeseries, kp_indices))

    if include_critical_frames:
        features.update(extract_critical_frame_features(timeseries))

    if include_frame_diffs:
        features.update(extract_frame_difference_features(timeseries))

    # Participant encoding
    if participant_id is not None:
        features["participant_id"] = float(participant_id)
        for pid in range(1, 6):
            features[f"participant_{pid}"] = 1.0 if participant_id == pid else 0.0

    return features


if __name__ == "__main__":
    # Test derivative feature extraction
    import sys
    sys.path.insert(0, str(__file__).replace("/src/derivative_features.py", ""))

    try:
        from src.data_loader import load_single_shot
    except ImportError:
        from data_loader import load_single_shot

    print("Testing derivative feature extraction...")

    # Load a test shot
    metadata, timeseries = load_single_shot(0, train=True)
    print(f"Loaded shot: {metadata['id']}, shape: {timeseries.shape}")

    # Test each feature type
    print("\nVelocity features:")
    vel_feats = extract_velocity_features(timeseries)
    print(f"  Count: {len(vel_feats)}")

    print("\nAcceleration features:")
    acc_feats = extract_acceleration_features(timeseries)
    print(f"  Count: {len(acc_feats)}")

    print("\nJerk features:")
    jerk_feats = extract_jerk_features(timeseries)
    print(f"  Count: {len(jerk_feats)}")

    print("\nCritical frame features:")
    cf_feats = extract_critical_frame_features(timeseries)
    print(f"  Count: {len(cf_feats)}")

    print("\nFrame difference features:")
    fd_feats = extract_frame_difference_features(timeseries)
    print(f"  Count: {len(fd_feats)}")

    print("\nAll derivative features:")
    all_feats = extract_all_derivative_features(
        timeseries,
        participant_id=metadata["participant_id"]
    )
    print(f"  Total count: {len(all_feats)}")

    # Check for NaN values
    nan_count = sum(1 for v in all_feats.values() if isinstance(v, float) and np.isnan(v))
    print(f"  NaN values: {nan_count}")

    print("\nDerivative feature extraction test complete!")
