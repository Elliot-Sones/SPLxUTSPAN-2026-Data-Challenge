"""
Geometric Feature Extraction for Bowling Motion Analysis.

Extracts spatial relationship features:
- Pairwise distances between key body parts (posture indicators)
- Trajectory geometry (path length, curvature, tortuosity)
- Joint angles and limb segment relationships
- Body alignment and orientation features
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Constants
FRAME_RATE = 60  # Hz
NUM_FRAMES = 240
NUM_KEYPOINTS = 69
COORDS_PER_KEYPOINT = 3

# Keypoint indices (from COCO + BlazePose format)
KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}

# Key distance pairs for bowling analysis
DISTANCE_PAIRS = [
    # Arm reach and extension
    ('right_wrist', 'right_hip'),           # Bowling arm reach
    ('right_wrist', 'right_shoulder'),      # Arm extension
    ('right_elbow', 'right_hip'),           # Elbow position relative to body
    ('right_wrist', 'right_elbow'),         # Forearm length (for normalization)

    # Cross-body coordination
    ('right_wrist', 'left_wrist'),          # Hand separation
    ('right_shoulder', 'left_shoulder'),    # Shoulder width (stance)
    ('right_hip', 'left_hip'),              # Hip width

    # Balance and stance
    ('right_ankle', 'left_ankle'),          # Foot separation
    ('right_knee', 'right_hip'),            # Thigh length
    ('right_ankle', 'right_hip'),           # Leg extension

    # Vertical body alignment
    ('nose', 'right_hip'),                  # Trunk height
    ('right_shoulder', 'right_hip'),        # Torso length

    # Release mechanics
    ('right_wrist', 'nose'),                # Wrist to head (release height)
    ('right_index', 'right_wrist'),         # Finger extension
]


def get_keypoint_position(sequence: np.ndarray, keypoint_name: str,
                          frame: Optional[int] = None) -> np.ndarray:
    """
    Get 3D position of a keypoint.

    Args:
        sequence: Shape (n_frames, 207)
        keypoint_name: Name of keypoint
        frame: Specific frame index, or None for all frames

    Returns:
        Position array shape (3,) if frame specified, else (n_frames, 3)
    """
    idx = KEYPOINT_INDICES.get(keypoint_name)
    if idx is None:
        raise ValueError(f"Unknown keypoint: {keypoint_name}")

    col_start = idx * 3
    if frame is not None:
        return sequence[frame, col_start:col_start+3]
    return sequence[:, col_start:col_start+3]


def compute_mid_hip(sequence: np.ndarray, frame: Optional[int] = None) -> np.ndarray:
    """Compute midpoint between left and right hip."""
    left_hip = get_keypoint_position(sequence, 'left_hip', frame)
    right_hip = get_keypoint_position(sequence, 'right_hip', frame)
    return (left_hip + right_hip) / 2


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between two 3D points or arrays of points."""
    return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1))


def extract_distance_features(sequence: np.ndarray) -> Dict[str, float]:
    """
    Extract pairwise distance features between key body parts.

    Args:
        sequence: Shape (n_frames, 207)

    Returns:
        Dictionary of distance-based features
    """
    features = {}

    # Detect release frame (peak wrist velocity)
    right_wrist_pos = get_keypoint_position(sequence, 'right_wrist')
    wrist_velocity = np.diff(right_wrist_pos, axis=0)
    wrist_speed = np.linalg.norm(wrist_velocity, axis=1)
    release_frame = np.argmax(wrist_speed) + 1  # +1 because diff reduces length by 1
    release_frame = np.clip(release_frame, 10, len(sequence) - 10)

    features['release_frame_detected'] = float(release_frame)

    for kp1_name, kp2_name in DISTANCE_PAIRS:
        # Get positions for both keypoints
        try:
            pos1 = get_keypoint_position(sequence, kp1_name)
            pos2 = get_keypoint_position(sequence, kp2_name)
        except ValueError:
            continue

        # Compute distance over time
        distances = euclidean_distance(pos1, pos2)

        # Handle NaN values
        valid_mask = ~np.isnan(distances)
        if not valid_mask.any():
            continue
        distances_clean = distances[valid_mask]

        pair_name = f'{kp1_name}_{kp2_name}'

        # Basic statistics
        features[f'dist_{pair_name}_mean'] = np.mean(distances_clean)
        features[f'dist_{pair_name}_std'] = np.std(distances_clean)
        features[f'dist_{pair_name}_max'] = np.max(distances_clean)
        features[f'dist_{pair_name}_min'] = np.min(distances_clean)
        features[f'dist_{pair_name}_range'] = np.max(distances_clean) - np.min(distances_clean)

        # At release frame
        if valid_mask[release_frame]:
            features[f'dist_{pair_name}_at_release'] = distances[release_frame]
        else:
            features[f'dist_{pair_name}_at_release'] = np.mean(distances_clean)

        # Trajectory length (total distance traveled)
        dist_diff = np.diff(distances_clean)
        features[f'dist_{pair_name}_trajectory_length'] = np.sum(np.abs(dist_diff))

        # Rate of change at release
        if release_frame > 0 and release_frame < len(distances) - 1:
            rate = (distances[min(release_frame + 1, len(distances)-1)] -
                   distances[max(release_frame - 1, 0)]) / 2
            features[f'dist_{pair_name}_rate_at_release'] = rate

    return features


def extract_distance_ratios(sequence: np.ndarray) -> Dict[str, float]:
    """
    Extract normalized distance ratios (body-size invariant features).
    """
    features = {}

    # Reference distances for normalization
    shoulder_width = get_keypoint_position(sequence, 'right_shoulder') - \
                     get_keypoint_position(sequence, 'left_shoulder')
    shoulder_dist = np.mean(np.linalg.norm(shoulder_width, axis=1))

    if shoulder_dist < 0.01:  # Avoid division by zero
        shoulder_dist = 1.0

    # Arm reach relative to shoulder width
    wrist_pos = get_keypoint_position(sequence, 'right_wrist')
    shoulder_pos = get_keypoint_position(sequence, 'right_shoulder')
    arm_extension = euclidean_distance(wrist_pos, shoulder_pos)

    features['ratio_arm_extension_mean'] = np.nanmean(arm_extension) / shoulder_dist
    features['ratio_arm_extension_max'] = np.nanmax(arm_extension) / shoulder_dist

    # Stance width relative to shoulder width
    ankle_dist = euclidean_distance(
        get_keypoint_position(sequence, 'right_ankle'),
        get_keypoint_position(sequence, 'left_ankle')
    )
    features['ratio_stance_width_mean'] = np.nanmean(ankle_dist) / shoulder_dist

    # Hand separation relative to shoulder width
    hand_dist = euclidean_distance(
        get_keypoint_position(sequence, 'right_wrist'),
        get_keypoint_position(sequence, 'left_wrist')
    )
    features['ratio_hand_separation_mean'] = np.nanmean(hand_dist) / shoulder_dist
    features['ratio_hand_separation_max'] = np.nanmax(hand_dist) / shoulder_dist

    return features


def compute_angle_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Compute angle at p2 formed by p1-p2-p3.

    Returns angle in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    # Normalize vectors
    v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True)

    # Avoid division by zero
    v1_norm = np.clip(v1_norm, 1e-10, None)
    v2_norm = np.clip(v2_norm, 1e-10, None)

    v1_normalized = v1 / v1_norm
    v2_normalized = v2 / v2_norm

    # Compute angle via dot product
    cos_angle = np.sum(v1_normalized * v2_normalized, axis=-1)
    cos_angle = np.clip(cos_angle, -1, 1)

    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def extract_joint_angle_features(sequence: np.ndarray) -> Dict[str, float]:
    """
    Extract joint angle features for key body joints.
    """
    features = {}

    # Joint definitions: (parent, joint, child)
    joints = [
        ('right_shoulder', 'right_elbow', 'right_wrist'),  # Right elbow angle
        ('right_hip', 'right_shoulder', 'right_elbow'),    # Right shoulder angle
        ('right_hip', 'right_knee', 'right_ankle'),        # Right knee angle
        ('right_shoulder', 'right_hip', 'right_knee'),     # Right hip angle
        ('left_shoulder', 'left_elbow', 'left_wrist'),     # Left elbow angle
        ('left_hip', 'left_knee', 'left_ankle'),           # Left knee angle
    ]

    # Detect release frame
    right_wrist_pos = get_keypoint_position(sequence, 'right_wrist')
    wrist_velocity = np.diff(right_wrist_pos, axis=0)
    wrist_speed = np.linalg.norm(wrist_velocity, axis=1)
    release_frame = np.argmax(wrist_speed) + 1
    release_frame = np.clip(release_frame, 10, len(sequence) - 10)

    for parent, joint, child in joints:
        try:
            p1 = get_keypoint_position(sequence, parent)
            p2 = get_keypoint_position(sequence, joint)
            p3 = get_keypoint_position(sequence, child)
        except ValueError:
            continue

        angles = compute_angle_3points(p1, p2, p3)

        # Handle NaN
        valid_mask = ~np.isnan(angles)
        if not valid_mask.any():
            continue
        angles_clean = angles[valid_mask]

        joint_name = joint.replace('right_', 'r_').replace('left_', 'l_')

        features[f'angle_{joint_name}_mean'] = np.mean(angles_clean)
        features[f'angle_{joint_name}_std'] = np.std(angles_clean)
        features[f'angle_{joint_name}_max'] = np.max(angles_clean)
        features[f'angle_{joint_name}_min'] = np.min(angles_clean)
        features[f'angle_{joint_name}_range'] = np.max(angles_clean) - np.min(angles_clean)

        if valid_mask[release_frame]:
            features[f'angle_{joint_name}_at_release'] = angles[release_frame]

        # Angular velocity (degrees per frame)
        angle_diff = np.diff(angles_clean)
        features[f'angle_{joint_name}_velocity_mean'] = np.mean(np.abs(angle_diff))
        features[f'angle_{joint_name}_velocity_max'] = np.max(np.abs(angle_diff))

    return features


def extract_trajectory_features(sequence: np.ndarray) -> Dict[str, float]:
    """
    Extract trajectory geometry features for key body parts.

    Features:
    - Path length: Total distance traveled
    - Tortuosity: Path length / straight-line distance (1 = straight line)
    - Curvature: Local curvature along trajectory
    """
    features = {}

    keypoints_to_track = ['right_wrist', 'right_elbow', 'right_shoulder',
                          'right_hip', 'right_knee', 'right_ankle']

    for kp_name in keypoints_to_track:
        try:
            positions = get_keypoint_position(sequence, kp_name)
        except ValueError:
            continue

        # Handle NaN values
        valid_mask = ~np.any(np.isnan(positions), axis=1)
        if np.sum(valid_mask) < 10:
            continue
        positions_clean = positions[valid_mask]

        # Path length
        path_segments = np.diff(positions_clean, axis=0)
        segment_lengths = np.linalg.norm(path_segments, axis=1)
        path_length = np.sum(segment_lengths)

        # Straight-line distance (start to end)
        straight_dist = np.linalg.norm(positions_clean[-1] - positions_clean[0])

        # Tortuosity
        if straight_dist > 0.01:
            tortuosity = path_length / straight_dist
        else:
            tortuosity = 1.0

        kp_short = kp_name.replace('right_', 'r_')
        features[f'traj_{kp_short}_path_length'] = path_length
        features[f'traj_{kp_short}_straight_dist'] = straight_dist
        features[f'traj_{kp_short}_tortuosity'] = tortuosity

        # Curvature (approximated by change in direction)
        if len(path_segments) > 1:
            # Normalize segments
            seg_norms = np.linalg.norm(path_segments, axis=1, keepdims=True)
            seg_norms = np.clip(seg_norms, 1e-10, None)
            normalized_segments = path_segments / seg_norms

            # Angle between consecutive segments
            dot_products = np.sum(normalized_segments[:-1] * normalized_segments[1:], axis=1)
            dot_products = np.clip(dot_products, -1, 1)
            angles = np.arccos(dot_products)

            features[f'traj_{kp_short}_curvature_mean'] = np.mean(angles)
            features[f'traj_{kp_short}_curvature_max'] = np.max(angles)
            features[f'traj_{kp_short}_curvature_std'] = np.std(angles)

    return features


def extract_body_orientation_features(sequence: np.ndarray) -> Dict[str, float]:
    """
    Extract body orientation and alignment features.
    """
    features = {}

    # Shoulder alignment (relative to lane direction - assumed to be Y axis)
    left_shoulder = get_keypoint_position(sequence, 'left_shoulder')
    right_shoulder = get_keypoint_position(sequence, 'right_shoulder')
    shoulder_vec = right_shoulder - left_shoulder

    # Angle of shoulder line in XZ plane (horizontal rotation)
    shoulder_angle_xz = np.arctan2(shoulder_vec[:, 2], shoulder_vec[:, 0])

    features['orient_shoulder_angle_mean'] = np.nanmean(np.degrees(shoulder_angle_xz))
    features['orient_shoulder_angle_std'] = np.nanstd(np.degrees(shoulder_angle_xz))

    # Hip alignment
    left_hip = get_keypoint_position(sequence, 'left_hip')
    right_hip = get_keypoint_position(sequence, 'right_hip')
    hip_vec = right_hip - left_hip

    hip_angle_xz = np.arctan2(hip_vec[:, 2], hip_vec[:, 0])
    features['orient_hip_angle_mean'] = np.nanmean(np.degrees(hip_angle_xz))

    # Shoulder-hip alignment difference (body twist)
    body_twist = shoulder_angle_xz - hip_angle_xz
    features['orient_body_twist_mean'] = np.nanmean(np.degrees(body_twist))
    features['orient_body_twist_max'] = np.nanmax(np.abs(np.degrees(body_twist)))

    # Trunk lean (forward/backward tilt)
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2
    trunk_vec = mid_shoulder - mid_hip

    # Angle from vertical (Y-axis assumed up)
    trunk_from_vertical = np.arctan2(
        np.sqrt(trunk_vec[:, 0]**2 + trunk_vec[:, 2]**2),
        trunk_vec[:, 1]
    )
    features['orient_trunk_lean_mean'] = np.nanmean(np.degrees(trunk_from_vertical))
    features['orient_trunk_lean_max'] = np.nanmax(np.degrees(trunk_from_vertical))

    return features


def extract_all_geometric_features(sequence: np.ndarray) -> Dict[str, float]:
    """
    Extract all geometric features.

    Args:
        sequence: Shape (n_frames, 207)

    Returns:
        Dictionary of all geometric features
    """
    features = {}

    features.update(extract_distance_features(sequence))
    features.update(extract_distance_ratios(sequence))
    features.update(extract_joint_angle_features(sequence))
    features.update(extract_trajectory_features(sequence))
    features.update(extract_body_orientation_features(sequence))

    return features


def get_geometric_feature_names() -> List[str]:
    """Return list of all geometric feature names."""
    # Generate a sample to get all feature names
    np.random.seed(42)
    sample = np.random.randn(240, 207)
    features = extract_all_geometric_features(sample)
    return list(features.keys())


if __name__ == '__main__':
    # Test with random data
    np.random.seed(42)
    test_sequence = np.random.randn(240, 207)

    features = extract_all_geometric_features(test_sequence)

    print(f"Total geometric features: {len(features)}")
    print("\nSample features:")
    for name, value in list(features.items())[:15]:
        print(f"  {name}: {value:.4f}")
