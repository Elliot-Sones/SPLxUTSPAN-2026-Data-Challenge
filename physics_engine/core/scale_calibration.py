"""
Scale Calibration: Convert normalized skeleton units to meters.

Uses multiple body segments with known anthropometric lengths for robust estimation.
"""

import numpy as np
from typing import Dict, Optional, Tuple


# Anthropometric reference lengths (meters)
# Based on average adult male basketball player measurements
REFERENCE_LENGTHS = {
    'forearm': 0.265,      # Elbow to wrist
    'upper_arm': 0.300,    # Shoulder to elbow
    'shin': 0.400,         # Knee to ankle
    'thigh': 0.430,        # Hip to knee
    'hand': 0.190,         # Wrist to fingertip
    'shoulder_width': 0.450,  # Left to right shoulder
}


def get_joint_position(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    joint_name: str,
    frame: int
) -> Optional[np.ndarray]:
    """
    Extract 3D position of a joint at a specific frame.

    Args:
        timeseries: Shape (240, 207) - frames x features
        keypoint_idx: Dict mapping joint names to column indices
        joint_name: Name of the joint (e.g., 'right_wrist')
        frame: Frame number (0-239)

    Returns:
        Position [x, y, z] or None if not available
    """
    if joint_name not in keypoint_idx:
        return None

    idx = keypoint_idx[joint_name]
    if frame >= len(timeseries):
        return None

    # Each joint has 3 coordinates (x, y, z) stored consecutively
    pos = timeseries[frame, idx*3:(idx+1)*3]

    # Check for NaN or zero values
    if np.any(np.isnan(pos)) or not np.any(pos):
        return None

    return pos


def calibrate_scale_factor(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    frame_range: Tuple[int, int] = (100, 180)
) -> float:
    """
    Compute meters-per-unit scale factor using multiple body segments.

    Uses median across segments and frames for robustness to noise.

    Args:
        timeseries: Shape (240, 207)
        keypoint_idx: Dict mapping joint names to indices
        frame_range: Frames to sample (default: shooting motion phase)

    Returns:
        Scale factor (meters per normalized unit)
    """
    scales = []

    for frame in range(frame_range[0], frame_range[1], 5):
        # --- Right forearm ---
        elbow = get_joint_position(timeseries, keypoint_idx, "right_elbow", frame)
        wrist = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame)
        if elbow is not None and wrist is not None:
            forearm_len = np.linalg.norm(wrist - elbow)
            if forearm_len > 0.01:
                scales.append(REFERENCE_LENGTHS['forearm'] / forearm_len)

        # --- Right upper arm ---
        shoulder = get_joint_position(timeseries, keypoint_idx, "right_shoulder", frame)
        if shoulder is not None and elbow is not None:
            upper_arm_len = np.linalg.norm(elbow - shoulder)
            if upper_arm_len > 0.01:
                scales.append(REFERENCE_LENGTHS['upper_arm'] / upper_arm_len)

        # --- Right shin ---
        knee = get_joint_position(timeseries, keypoint_idx, "right_knee", frame)
        ankle = get_joint_position(timeseries, keypoint_idx, "right_ankle", frame)
        if knee is not None and ankle is not None:
            shin_len = np.linalg.norm(ankle - knee)
            if shin_len > 0.01:
                scales.append(REFERENCE_LENGTHS['shin'] / shin_len)

        # --- Right thigh ---
        hip = get_joint_position(timeseries, keypoint_idx, "right_hip", frame)
        if hip is not None and knee is not None:
            thigh_len = np.linalg.norm(knee - hip)
            if thigh_len > 0.01:
                scales.append(REFERENCE_LENGTHS['thigh'] / thigh_len)

        # --- Left side (for redundancy) ---
        l_elbow = get_joint_position(timeseries, keypoint_idx, "left_elbow", frame)
        l_wrist = get_joint_position(timeseries, keypoint_idx, "left_wrist", frame)
        if l_elbow is not None and l_wrist is not None:
            l_forearm_len = np.linalg.norm(l_wrist - l_elbow)
            if l_forearm_len > 0.01:
                scales.append(REFERENCE_LENGTHS['forearm'] / l_forearm_len)

        # --- Shoulder width ---
        l_shoulder = get_joint_position(timeseries, keypoint_idx, "left_shoulder", frame)
        if shoulder is not None and l_shoulder is not None:
            shoulder_width = np.linalg.norm(shoulder - l_shoulder)
            if shoulder_width > 0.01:
                scales.append(REFERENCE_LENGTHS['shoulder_width'] / shoulder_width)

    # Use median for robustness
    if scales:
        return float(np.median(scales))
    else:
        # Fallback to previously validated value
        return 0.29


def validate_scale(
    scale_factor: float,
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    frame: int = 150
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that scale factor produces physically plausible values.

    Args:
        scale_factor: Computed scale factor
        timeseries: Shape (240, 207)
        keypoint_idx: Dict mapping joint names to indices
        frame: Frame to validate at

    Returns:
        (is_valid, measurements_dict)
    """
    measurements = {}
    is_valid = True

    # Check wrist height above ankle (should be ~2.0-2.4m at release)
    wrist = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame)
    ankle = get_joint_position(timeseries, keypoint_idx, "right_ankle", frame)

    if wrist is not None and ankle is not None:
        release_height = (wrist[2] - ankle[2]) * scale_factor
        measurements['release_height'] = release_height

        # Free throw release height typically 1.8m - 2.5m
        if not (1.5 < release_height < 2.8):
            is_valid = False

    # Check arm span (should be ~1.8-2.2m)
    r_wrist = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame)
    l_wrist = get_joint_position(timeseries, keypoint_idx, "left_wrist", frame)

    if r_wrist is not None and l_wrist is not None:
        arm_span = np.linalg.norm(r_wrist - l_wrist) * scale_factor
        measurements['arm_span'] = arm_span

        # Arm span typically 1.5m - 2.5m
        if not (1.0 < arm_span < 3.0):
            is_valid = False

    return is_valid, measurements


def get_keypoint_indices(keypoint_cols):
    """
    Build a dict mapping joint names to their indices in the timeseries.

    The timeseries has 207 columns: 69 joints x 3 coordinates (x, y, z).
    """
    keypoints = []
    for col in keypoint_cols:
        # Column format: "joint_name_x", "joint_name_y", "joint_name_z"
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)

    return {name: i for i, name in enumerate(keypoints)}


if __name__ == "__main__":
    # Test with sample data
    print("Scale calibration module loaded.")
    print(f"Reference lengths: {REFERENCE_LENGTHS}")
