"""
Release Parameter Extraction: Extract ball position, velocity, and spin from skeleton data.

Uses Savitzky-Golay filtering for robust velocity estimation.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, Optional, Tuple
from .scale_calibration import get_joint_position


# Player-specific release frame search windows (from DEFINITIVE_PHYSICS_RESULTS.md)
PLAYER_RELEASE_WINDOWS = {
    1: (110, 165),
    2: (80, 195),
    3: (60, 220),
    4: (125, 195),
    5: (65, 200),
}


def detect_release_frame(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    player_id: int,
    scale_factor: float
) -> Tuple[int, float]:
    """
    Detect the ball release frame using wrist kinematics.

    Release is detected as the frame with peak upward wrist velocity,
    within player-specific search windows.

    Args:
        timeseries: Shape (240, 207)
        keypoint_idx: Dict mapping joint names to indices
        player_id: Player ID (1-5)
        scale_factor: Meters per unit

    Returns:
        (release_frame, peak_velocity_z)
    """
    # Get player-specific search window
    start, end = PLAYER_RELEASE_WINDOWS.get(player_id, (100, 180))

    # Extract wrist z-positions in the window
    wrist_z = []
    valid_frames = []

    for frame in range(start, min(end, 240)):
        pos = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame)
        if pos is not None:
            wrist_z.append(pos[2])
            valid_frames.append(frame)

    if len(wrist_z) < 10:
        # Not enough data - fall back to middle of window
        return (start + end) // 2, 0.0

    wrist_z = np.array(wrist_z)

    # Smooth and compute velocity using Savitzky-Golay filter
    # window=7 at 60fps = ~0.12s window, polyorder=3 for cubic fit
    try:
        # deriv=1 gives first derivative, multiplied by frame rate
        wrist_vz = savgol_filter(wrist_z, window_length=7, polyorder=3, deriv=1) * 60
    except ValueError:
        # Fall back if window is too small
        wrist_vz = np.gradient(wrist_z) * 60

    # Release is at peak upward velocity
    peak_idx = np.argmax(wrist_vz)
    release_frame = valid_frames[peak_idx]
    peak_velocity = wrist_vz[peak_idx] * scale_factor

    # Validate: velocity should decrease after release
    if peak_idx < len(wrist_vz) - 5:
        post_release_vel = np.mean(wrist_vz[peak_idx+1:peak_idx+5])
        if post_release_vel > wrist_vz[peak_idx] * 0.9:
            # Velocity didn't decrease - might be wrong frame
            # Fall back to maximum height
            peak_height_idx = np.argmax(wrist_z)
            release_frame = valid_frames[peak_height_idx]

    return release_frame, peak_velocity


def extract_release_position(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    release_frame: int,
    scale_factor: float
) -> np.ndarray:
    """
    Extract ball position at release.

    Uses weighted centroid of palm and fingertips.

    Args:
        timeseries: Shape (240, 207)
        keypoint_idx: Dict mapping joint names to indices
        release_frame: Detected release frame
        scale_factor: Meters per unit

    Returns:
        Ball position [x, y, z] in meters (MuJoCo coordinates)
    """
    # Get wrist position (base of palm)
    wrist = get_joint_position(timeseries, keypoint_idx, "right_wrist", release_frame)
    if wrist is None:
        wrist = np.array([0, 0, 7])  # Fallback

    # Get fingertip positions (index and middle finger)
    index_tip = get_joint_position(timeseries, keypoint_idx, "right_second_finger_distal", release_frame)
    middle_tip = get_joint_position(timeseries, keypoint_idx, "right_third_finger_distal", release_frame)

    # Weighted centroid: 60% palm, 40% fingertips
    if index_tip is not None and middle_tip is not None:
        fingertip_center = (index_tip + middle_tip) / 2
        ball_pos = 0.6 * wrist + 0.4 * fingertip_center
    else:
        # Fallback: use wrist with small forward offset
        ball_pos = wrist.copy()
        ball_pos[0] -= 0.1 / scale_factor  # Ball slightly forward

    # Get ground reference (ankle position)
    ankle = get_joint_position(timeseries, keypoint_idx, "right_ankle", release_frame)
    ground_z = ankle[2] if ankle is not None else 0.9

    # Get body center for lateral reference (use hip midpoint)
    right_hip = get_joint_position(timeseries, keypoint_idx, "right_hip", release_frame)
    left_hip = get_joint_position(timeseries, keypoint_idx, "left_hip", release_frame)
    if right_hip is not None and left_hip is not None:
        body_center_y = (right_hip[1] + left_hip[1]) / 2
    else:
        body_center_y = ball_pos[1]  # Fallback: assume centered

    # Convert to MuJoCo coordinates
    # Data coordinate system (from diagnostics):
    #   X = forward/backward (positive X = toward hoop during shot)
    #   Y = lateral (left-right)
    #   Z = vertical (up-down)
    # MuJoCo: X = toward hoop (positive), Y = lateral, Z = up

    # Height above ground
    height = (ball_pos[2] - ground_z) * scale_factor

    # Lateral position: deviation from body center (should be small for straight shot)
    # The hoop is at Y = 0.15m in MuJoCo, so we center around that
    lateral_deviation = (ball_pos[1] - body_center_y) * scale_factor
    lateral = 0.15 + lateral_deviation  # Center at hoop Y position

    # X position: start at free throw line
    x_pos = -4.57  # 4.57m from hoop

    return np.array([x_pos, lateral, height])


def extract_release_velocity(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    release_frame: int,
    scale_factor: float,
    release_height: float = 2.0
) -> np.ndarray:
    """
    Extract ball release velocity using Savitzky-Golay differentiation.

    The raw data velocities are too small (1-2 m/s when we need 7-9 m/s).
    This appears to be a data normalization artifact. We extract the DIRECTION
    from the data and apply a physically correct magnitude.

    Args:
        timeseries: Shape (240, 207)
        keypoint_idx: Dict mapping joint names to indices
        release_frame: Detected release frame
        scale_factor: Meters per unit
        release_height: Height above ground at release (meters)

    Returns:
        Release velocity [vx, vy, vz] in m/s (MuJoCo coordinates)
    """
    # Extract wrist positions around release (+-10 frames)
    window_start = max(0, release_frame - 10)
    window_end = min(240, release_frame + 10)

    positions = []
    valid_frames = []

    for frame in range(window_start, window_end):
        pos = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame)
        if pos is not None:
            positions.append(pos)
            valid_frames.append(frame)

    if len(positions) < 7:
        # Not enough data - return default velocity for 45 degree shot
        return np.array([5.3, 0.0, 5.3])  # ~7.5 m/s at 45 degrees

    positions = np.array(positions)

    # Savitzky-Golay derivative for smooth velocity estimate
    try:
        vx = savgol_filter(positions[:, 0], window_length=7, polyorder=3, deriv=1) * 60
        vy = savgol_filter(positions[:, 1], window_length=7, polyorder=3, deriv=1) * 60
        vz = savgol_filter(positions[:, 2], window_length=7, polyorder=3, deriv=1) * 60
    except ValueError:
        vx = np.gradient(positions[:, 0]) * 60
        vy = np.gradient(positions[:, 1]) * 60
        vz = np.gradient(positions[:, 2]) * 60

    # Get velocity at release frame
    idx = np.searchsorted(valid_frames, release_frame)
    idx = min(idx, len(vx) - 1)

    # Raw velocity in data units/second (NOT meters - data velocities are too small)
    raw_velocity = np.array([vx[idx], vy[idx], vz[idx]])

    # Coordinate transformation based on diagnostics:
    # Data: positive X = toward hoop (wrist X increases during shot follow-through)
    #       But at release moment, wrist is moving in NEGATIVE X direction (toward hoop)
    #       because the shooting motion goes forward
    # Data: Y = lateral (shoulder width is in Y)
    # Data: Z = vertical (ankle to nose is in Z)
    #
    # MuJoCo: positive X = toward hoop, Y = lateral, Z = up
    #
    # Key insight from diagnostics: at frame 150 (release), dX/dt is NEGATIVE
    # This means the ball goes in negative X direction in data coords
    # In MuJoCo, we need positive X to go toward hoop
    # So: vx_mujoco = -vx_data

    velocity_direction = np.array([-raw_velocity[0], raw_velocity[1], raw_velocity[2]])

    # Normalize to get direction only
    dir_magnitude = np.linalg.norm(velocity_direction)
    if dir_magnitude < 0.01:
        # No clear direction - use default 45 degree toward hoop
        velocity_direction = np.array([1.0, 0.0, 1.0])
        dir_magnitude = np.sqrt(2)

    direction_unit = velocity_direction / dir_magnitude

    # APPROACH: Preserve variation in the extracted direction
    # Instead of calculating velocity to hit the hoop (which removes variation),
    # we use a base velocity and let the direction determine where the ball lands.
    # The landing position deviation from hoop center predicts shot outcome.

    # Softly constrain angle to reasonable range (25-65 degrees)
    # but preserve variation within that range
    if direction_unit[0] > 0.1:  # Moving toward hoop
        current_angle = np.arctan2(direction_unit[2], direction_unit[0])

        # Soft clamping: compress angles outside 25-65 degree range
        # but keep the relative ordering
        min_angle = np.radians(25)
        max_angle = np.radians(65)
        mid_angle = np.radians(45)

        if current_angle < min_angle:
            # Compress angles below 25 degrees
            current_angle = min_angle - (min_angle - current_angle) * 0.3
        elif current_angle > max_angle:
            # Compress angles above 65 degrees
            current_angle = max_angle + (current_angle - max_angle) * 0.3

        # Reconstruct direction with adjusted angle
        vx_ratio = np.cos(current_angle)
        vz_ratio = np.sin(current_angle)

        # Preserve lateral component with moderate reduction
        vy_ratio = direction_unit[1] * 0.5  # Keep 50% of lateral variation

        # Renormalize
        norm = np.sqrt(vx_ratio**2 + vy_ratio**2 + vz_ratio**2)
        direction_unit = np.array([vx_ratio, vy_ratio, vz_ratio]) / norm
    else:
        # Not moving toward hoop - use default direction
        direction_unit = np.array([0.707, 0.0, 0.707])

    # Use a BASE velocity that reaches the hoop for average shot
    # Then add variation from the extracted direction
    # This ensures most shots reach the hoop while preserving angular variation

    # Calculate base speed for 45-degree shot from release_height
    g = 9.81
    x_target = 4.57
    z_target = 3.05
    z0 = release_height

    # For 45 degree shot: v^2 = g*x / (sin(2*theta) * (1 + (z_target-z0)/(x*tan(theta))))
    # Simplified for 45 degrees: v^2 ~ g * x / (1 - (z_target-z0)/x)
    base_speed = np.sqrt(g * x_target / (1 - (z_target - z0) / x_target + 0.1))
    base_speed = np.clip(base_speed, 7.0, 9.0)

    # Modulate speed slightly based on angle
    # Steeper angles need more speed, shallower need less
    angle = np.arctan2(direction_unit[2], direction_unit[0])
    angle_factor = 1.0 + (angle - np.radians(45)) * 0.3  # Small adjustment
    target_speed = base_speed * angle_factor
    target_speed = np.clip(target_speed, 6.5, 10.0)

    velocity_mujoco = direction_unit * target_speed

    return velocity_mujoco


def estimate_backspin(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    release_frame: int,
    scale_factor: float
) -> float:
    """
    Estimate ball backspin from fingertip-wrist differential motion.

    Backspin is created by fingers flicking downward relative to wrist.

    Args:
        timeseries: Shape (240, 207)
        keypoint_idx: Dict mapping joint names to indices
        release_frame: Detected release frame
        scale_factor: Meters per unit

    Returns:
        Backspin in rad/s (positive = backspin)
    """
    dt = 3  # 3 frames = 0.05 seconds

    # Get fingertip and wrist positions before and after release
    frame_pre = max(0, release_frame - dt)
    frame_post = min(239, release_frame + dt)

    index_pre = get_joint_position(timeseries, keypoint_idx, "right_second_finger_distal", frame_pre)
    index_post = get_joint_position(timeseries, keypoint_idx, "right_second_finger_distal", frame_post)
    wrist_pre = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame_pre)
    wrist_post = get_joint_position(timeseries, keypoint_idx, "right_wrist", frame_post)

    if any(p is None for p in [index_pre, index_post, wrist_pre, wrist_post]):
        # Default moderate backspin
        return 15.0

    # Relative fingertip motion (finger velocity - wrist velocity)
    fingertip_motion = (index_post - index_pre) - (wrist_post - wrist_pre)

    # Convert to angular velocity
    ball_radius = 0.12  # meters

    # Tangential velocity at contact point
    frame_diff = frame_post - frame_pre
    tangential_vel = fingertip_motion[2] * scale_factor * 60 / frame_diff

    # Angular velocity (rad/s)
    backspin = tangential_vel / ball_radius

    # Typical backspin: 2-4 Hz = 12-25 rad/s
    backspin = np.clip(backspin, -30, 30)

    return float(backspin)


def extract_all_release_params(
    timeseries: np.ndarray,
    keypoint_idx: Dict[str, int],
    player_id: int,
    scale_factor: float
) -> Dict:
    """
    Extract all release parameters for a shot.

    Args:
        timeseries: Shape (240, 207)
        keypoint_idx: Dict mapping joint names to indices
        player_id: Player ID (1-5)
        scale_factor: Meters per unit

    Returns:
        Dict with release_frame, position, velocity, backspin
    """
    # Detect release frame
    release_frame, peak_vel = detect_release_frame(
        timeseries, keypoint_idx, player_id, scale_factor
    )

    # Extract position
    position = extract_release_position(
        timeseries, keypoint_idx, release_frame, scale_factor
    )

    # Extract velocity (pass actual release height for physics calculation)
    velocity = extract_release_velocity(
        timeseries, keypoint_idx, release_frame, scale_factor,
        release_height=position[2]  # Z component is height
    )

    # Estimate backspin
    backspin = estimate_backspin(
        timeseries, keypoint_idx, release_frame, scale_factor
    )

    return {
        'release_frame': release_frame,
        'position': position,
        'velocity': velocity,
        'backspin': backspin,
        'peak_wrist_velocity': peak_vel,
        'scale_factor': scale_factor,
    }


if __name__ == "__main__":
    print("Release extraction module loaded.")
    print(f"Player-specific release windows: {PLAYER_RELEASE_WINDOWS}")
