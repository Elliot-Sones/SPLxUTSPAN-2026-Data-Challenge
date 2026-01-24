"""
Ball position calculation and release detection for basketball shooting analysis
Uses hand keypoint data to track ball position and detect release point
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# NBA Basketball Specifications
NBA_BALL_DIAMETER_INCHES = 9.43
NBA_BALL_DIAMETER_FEET = NBA_BALL_DIAMETER_INCHES / 12.0  # 0.786 feet
NBA_BALL_RADIUS_FEET = NBA_BALL_DIAMETER_FEET / 2.0  # 0.393 feet

# Frame rate
FPS = 60
DT = 1.0 / FPS  # 0.01667 seconds

# Keypoint names for hand tracking
PALM_KEYPOINTS = {
    'left': [
        'left_wrist_2',
        'left_first_finger_mcp',
        'left_second_finger_mcp',
        'left_third_finger_mcp',
        'left_fourth_finger_mcp',
        'left_fifth_finger_mcp'
    ],
    'right': [
        'right_wrist_2',
        'right_first_finger_mcp',
        'right_second_finger_mcp',
        'right_third_finger_mcp',
        'right_fourth_finger_mcp',
        'right_fifth_finger_mcp'
    ]
}

FINGERTIP_KEYPOINTS = {
    'left': [
        'left_first_finger_distal',
        'left_second_finger_distal',
        'left_third_finger_distal',
        'left_fourth_finger_distal',
        'left_fifth_finger_distal'
    ],
    'right': [
        'right_first_finger_distal',
        'right_second_finger_distal',
        'right_third_finger_distal',
        'right_fourth_finger_distal',
        'right_fifth_finger_distal'
    ]
}

# Finger joint keypoints for angle calculation
FINGER_JOINTS = {
    'left': {
        'first': ['left_first_finger_cmc', 'left_first_finger_mcp', 'left_first_finger_ip', 'left_first_finger_distal'],
        'second': ['left_wrist_2', 'left_second_finger_mcp', 'left_second_finger_pip', 'left_second_finger_dip'],
        'third': ['left_wrist_2', 'left_third_finger_mcp', 'left_third_finger_pip', 'left_third_finger_dip'],
        'fourth': ['left_wrist_2', 'left_fourth_finger_mcp', 'left_fourth_finger_pip', 'left_fourth_finger_dip'],
        'fifth': ['left_wrist_2', 'left_fifth_finger_mcp', 'left_fifth_finger_pip', 'left_fifth_finger_dip']
    },
    'right': {
        'first': ['right_first_finger_cmc', 'right_first_finger_mcp', 'right_first_finger_ip', 'right_first_finger_distal'],
        'second': ['right_wrist_2', 'right_second_finger_mcp', 'right_second_finger_pip', 'right_second_finger_dip'],
        'third': ['right_wrist_2', 'right_third_finger_mcp', 'right_third_finger_pip', 'right_third_finger_dip'],
        'fourth': ['right_wrist_2', 'right_fourth_finger_mcp', 'right_fourth_finger_pip', 'right_fourth_finger_dip'],
        'fifth': ['right_wrist_2', 'right_fifth_finger_mcp', 'right_fifth_finger_pip', 'right_fifth_finger_dip']
    }
}


def get_keypoint_3d(shot, keypoint_name: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Extract 3D position of a keypoint at a specific frame

    Args:
        shot: Shot data from DataFrame
        keypoint_name: Name of the keypoint (e.g., 'left_wrist_2')
        frame_idx: Frame index

    Returns:
        np.ndarray of [x, y, z] or None if data is missing/invalid
    """
    try:
        x = shot[f'{keypoint_name}_x'][frame_idx]
        y = shot[f'{keypoint_name}_y'][frame_idx]
        z = shot[f'{keypoint_name}_z'][frame_idx]

        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            return None

        return np.array([x, y, z], dtype=np.float32)
    except (KeyError, IndexError):
        return None


def calculate_palm_centroid(shot, frame_idx: int, hand: str = 'left') -> Optional[np.ndarray]:
    """
    Calculate palm centroid from wrist + 5 finger MCP joints

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index
        hand: 'left' or 'right'

    Returns:
        np.ndarray of [x, y, z] centroid or None if insufficient data
    """
    palm_points = []

    for keypoint_name in PALM_KEYPOINTS[hand]:
        pos = get_keypoint_3d(shot, keypoint_name, frame_idx)
        if pos is not None:
            palm_points.append(pos)

    if len(palm_points) < 3:  # Need at least 3 points for reasonable centroid
        return None

    return np.mean(palm_points, axis=0)


def calculate_fingertip_centroid(shot, frame_idx: int, hand: str = 'left') -> Optional[np.ndarray]:
    """
    Calculate fingertip centroid from all 5 finger distal keypoints

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index
        hand: 'left' or 'right'

    Returns:
        np.ndarray of [x, y, z] centroid or None if insufficient data
    """
    fingertip_points = []

    for keypoint_name in FINGERTIP_KEYPOINTS[hand]:
        pos = get_keypoint_3d(shot, keypoint_name, frame_idx)
        if pos is not None:
            fingertip_points.append(pos)

    if len(fingertip_points) < 3:  # Need at least 3 fingertips
        return None

    return np.mean(fingertip_points, axis=0)


def calculate_ball_position(shot, frame_idx: int) -> Optional[np.ndarray]:
    """
    Calculate ball position using hybrid weighted approach with both hands

    Method:
    - For each hand: ball_center = 0.6 * palm_centroid + 0.4 * fingertip_centroid
    - Combined: ball_position = (left_ball + right_ball) / 2

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index

    Returns:
        np.ndarray of [x, y, z] ball center or None if insufficient data
    """
    PALM_WEIGHT = 0.6
    FINGERTIP_WEIGHT = 0.4

    hand_ball_positions = []

    for hand in ['left', 'right']:
        palm = calculate_palm_centroid(shot, frame_idx, hand)
        fingertips = calculate_fingertip_centroid(shot, frame_idx, hand)

        if palm is not None and fingertips is not None:
            hand_ball = PALM_WEIGHT * palm + FINGERTIP_WEIGHT * fingertips
            hand_ball_positions.append(hand_ball)

    if len(hand_ball_positions) == 0:
        return None

    # Average positions from both hands (or use single hand if only one available)
    return np.mean(hand_ball_positions, axis=0)


def calculate_hand_velocity(shot, frame_idx: int, hand: str = 'left', window: int = 1) -> Optional[np.ndarray]:
    """
    Calculate velocity of hand centroid using finite differences

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index
        hand: 'left' or 'right'
        window: Number of frames to use for velocity calculation

    Returns:
        np.ndarray of [vx, vy, vz] in feet/second or None if insufficient data
    """
    if frame_idx < window:
        return None

    pos_current = calculate_palm_centroid(shot, frame_idx, hand)
    pos_previous = calculate_palm_centroid(shot, frame_idx - window, hand)

    if pos_current is None or pos_previous is None:
        return None

    velocity = (pos_current - pos_previous) / (window * DT)
    return velocity


def calculate_hand_acceleration(shot, frame_idx: int, hand: str = 'left', window: int = 1) -> Optional[float]:
    """
    Calculate vertical acceleration (z-component) of hand

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index
        hand: 'left' or 'right'
        window: Number of frames to use for calculation

    Returns:
        float acceleration in feet/s^2 or None if insufficient data
    """
    if frame_idx < window:
        return None

    v_current = calculate_hand_velocity(shot, frame_idx, hand, window)
    v_previous = calculate_hand_velocity(shot, frame_idx - window, hand, window)

    if v_current is None or v_previous is None:
        return None

    # Use Z-component (vertical) acceleration
    accel_z = (v_current[2] - v_previous[2]) / (window * DT)
    return accel_z


def calculate_angle_at_joint(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at joint p2 formed by points p1-p2-p3

    Args:
        p1, p2, p3: 3D positions as numpy arrays

    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    # Normalize
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    # Calculate angle
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_finger_angles(shot, frame_idx: int, hand: str = 'left') -> Dict[str, float]:
    """
    Calculate angles at PIP joint for each finger

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index
        hand: 'left' or 'right'

    Returns:
        Dictionary mapping finger name to PIP angle in degrees
    """
    angles = {}

    for finger_name, joint_names in FINGER_JOINTS[hand].items():
        if len(joint_names) < 3:
            continue

        # Get positions for angle calculation (use joints 1, 2, 3 for PIP angle)
        p1 = get_keypoint_3d(shot, joint_names[1], frame_idx)
        p2 = get_keypoint_3d(shot, joint_names[2], frame_idx)
        p3 = get_keypoint_3d(shot, joint_names[3], frame_idx)

        if p1 is not None and p2 is not None and p3 is not None:
            angle = calculate_angle_at_joint(p1, p2, p3)
            angles[finger_name] = angle

    return angles


def calculate_hand_aperture(shot, frame_idx: int, hand: str = 'left') -> Optional[float]:
    """
    Calculate hand aperture as distance between thumb tip and index finger tip

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index
        hand: 'left' or 'right'

    Returns:
        Distance in feet or None if data unavailable
    """
    thumb_tip = get_keypoint_3d(shot, f'{hand}_first_finger_distal', frame_idx)
    index_tip = get_keypoint_3d(shot, f'{hand}_second_finger_distal', frame_idx)

    if thumb_tip is None or index_tip is None:
        return None

    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance


def calculate_hand_separation(shot, frame_idx: int) -> Optional[float]:
    """
    Calculate distance between left and right hand palm centroids

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index

    Returns:
        Distance in feet or None if data unavailable
    """
    left_palm = calculate_palm_centroid(shot, frame_idx, 'left')
    right_palm = calculate_palm_centroid(shot, frame_idx, 'right')

    if left_palm is None or right_palm is None:
        return None

    distance = np.linalg.norm(left_palm - right_palm)
    return distance


def detect_release_frame(shot, start_frame: int = 0, end_frame: Optional[int] = None,
                         velocity_threshold: float = -10.0, finger_extension_rate: float = 5.0,
                         aperture_rate: float = 0.1, separation_rate: float = 0.05,
                         combined_threshold: float = 0.7) -> Tuple[int, np.ndarray]:
    """
    Detect ball release frame using multi-criteria approach

    Combines 4 detection methods:
    1. Hand velocity change (deceleration)
    2. Finger extension (angle rate of change)
    3. Hand aperture increase
    4. Hand separation

    Args:
        shot: Shot data from DataFrame
        start_frame: First frame to consider
        end_frame: Last frame to consider (None = all frames)
        velocity_threshold: Acceleration threshold for release (feet/s^2)
        finger_extension_rate: Finger angle change rate threshold (deg/frame)
        aperture_rate: Hand aperture change rate threshold (feet/frame)
        separation_rate: Hand separation rate threshold (feet/frame)
        combined_threshold: Combined score threshold for release detection

    Returns:
        Tuple of (release_frame_index, release_scores_array)
    """
    # Get total frames
    num_frames = len(shot['nose_x'])  # Use any keypoint to get frame count
    if end_frame is None:
        end_frame = num_frames - 1

    # Initialize score arrays
    release_scores = np.zeros(num_frames)
    velocity_scores = np.zeros(num_frames)
    finger_scores = np.zeros(num_frames)
    aperture_scores = np.zeros(num_frames)
    separation_scores = np.zeros(num_frames)

    # Calculate metrics for each frame
    for frame_idx in range(start_frame + 2, end_frame):  # Need history for derivatives

        # 1. Velocity-based detection (deceleration in Z)
        accel_left = calculate_hand_acceleration(shot, frame_idx, 'left', window=1)
        accel_right = calculate_hand_acceleration(shot, frame_idx, 'right', window=1)

        if accel_left is not None and accel_right is not None:
            avg_accel = (accel_left + accel_right) / 2.0
            if avg_accel < velocity_threshold:
                velocity_scores[frame_idx] = 1.0

        # 2. Finger extension detection
        for hand in ['left', 'right']:
            angles_current = calculate_finger_angles(shot, frame_idx, hand)
            angles_previous = calculate_finger_angles(shot, frame_idx - 1, hand)

            if len(angles_current) > 0 and len(angles_previous) > 0:
                # Calculate average angle change rate
                angle_changes = []
                for finger in angles_current:
                    if finger in angles_previous:
                        change = angles_current[finger] - angles_previous[finger]
                        angle_changes.append(abs(change))

                if len(angle_changes) > 0:
                    avg_change = np.mean(angle_changes)
                    if avg_change > finger_extension_rate:
                        finger_scores[frame_idx] = 1.0

        # 3. Hand aperture increase
        for hand in ['left', 'right']:
            aperture_current = calculate_hand_aperture(shot, frame_idx, hand)
            aperture_previous = calculate_hand_aperture(shot, frame_idx - 1, hand)

            if aperture_current is not None and aperture_previous is not None:
                aperture_change = aperture_current - aperture_previous
                if aperture_change > aperture_rate:
                    aperture_scores[frame_idx] = 1.0

        # 4. Hand separation
        separation_current = calculate_hand_separation(shot, frame_idx)
        separation_previous = calculate_hand_separation(shot, frame_idx - 1)

        if separation_current is not None and separation_previous is not None:
            separation_change = separation_current - separation_previous
            if separation_change > separation_rate:
                separation_scores[frame_idx] = 1.0

        # Combined score
        release_scores[frame_idx] = (
            0.3 * velocity_scores[frame_idx] +
            0.3 * finger_scores[frame_idx] +
            0.2 * aperture_scores[frame_idx] +
            0.2 * separation_scores[frame_idx]
        )

    # Find first frame exceeding combined threshold
    release_candidates = np.where(release_scores[start_frame:end_frame] > combined_threshold)[0]

    if len(release_candidates) > 0:
        release_frame = start_frame + release_candidates[0]
    else:
        # Fallback: use frame with maximum score
        release_frame = start_frame + np.argmax(release_scores[start_frame:end_frame])

    return release_frame, release_scores


def get_ball_trajectory(shot, start_frame: int = 0, end_frame: Optional[int] = None) -> np.ndarray:
    """
    Calculate ball trajectory for a range of frames

    Args:
        shot: Shot data from DataFrame
        start_frame: First frame
        end_frame: Last frame (None = all frames)

    Returns:
        np.ndarray of shape (num_frames, 3) with ball positions [x, y, z]
        Missing frames will have NaN values
    """
    num_frames = len(shot['nose_x'])
    if end_frame is None:
        end_frame = num_frames - 1

    trajectory = np.full((end_frame - start_frame + 1, 3), np.nan)

    for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        ball_pos = calculate_ball_position(shot, frame_idx)
        if ball_pos is not None:
            trajectory[i] = ball_pos

    return trajectory


def calculate_ball_to_hand_distance(shot, frame_idx: int, hand: str = 'left') -> Optional[float]:
    """
    Calculate distance from ball center to hand palm centroid

    Args:
        shot: Shot data from DataFrame
        frame_idx: Frame index
        hand: 'left' or 'right'

    Returns:
        Distance in feet or None if data unavailable
    """
    ball_pos = calculate_ball_position(shot, frame_idx)
    palm_pos = calculate_palm_centroid(shot, frame_idx, hand)

    if ball_pos is None or palm_pos is None:
        return None

    distance = np.linalg.norm(ball_pos - palm_pos)
    return distance
