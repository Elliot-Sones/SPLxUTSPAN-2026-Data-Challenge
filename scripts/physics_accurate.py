"""
Accurate Physics Simulation

Key corrections:
1. Data is in FEET (not normalized units)
2. Hoop at [5.25, -25, 10] feet
3. Use ALL finger keypoints to calculate ball position
4. Targets: angle (degrees), depth (inches), left_right (inches)
5. Universal physics - no per-player calibration
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, load_scalers, get_keypoint_columns

# Constants
HOOP_POSITION = np.array([5.25, -25.0, 10.0])  # feet
HOOP_RADIUS = 9.0 / 12.0  # 9 inches = 0.75 feet
BALL_RADIUS = 4.7 / 12.0  # 9.4 inch diameter = 4.7 inch radius = 0.39 feet
BALL_MASS = 0.625  # kg (22 oz)
GRAVITY = 32.174  # ft/s^2
FPS = 60

# Air resistance parameters (for a basketball)
AIR_DENSITY = 0.0765  # lb/ft^3 at sea level
DRAG_COEFFICIENT = 0.47  # sphere
BALL_AREA = np.pi * BALL_RADIUS**2  # cross-sectional area

SUBMISSION_DIR = PROJECT_DIR / "submission"


def get_keypoint_map(keypoint_cols):
    """Build mapping from keypoint names to column indices."""
    keypoint_map = {}
    for i, col in enumerate(keypoint_cols):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in keypoint_map:
                keypoint_map[name] = {}
            keypoint_map[name][axis] = i
    return keypoint_map


def get_position(timeseries, keypoint_map, name, frame):
    """Get 3D position of a keypoint at a frame."""
    if name not in keypoint_map:
        return None
    km = keypoint_map[name]
    if 'x' not in km or 'y' not in km or 'z' not in km:
        return None
    pos = np.array([
        timeseries[frame, km['x']],
        timeseries[frame, km['y']],
        timeseries[frame, km['z']]
    ])
    if np.any(np.isnan(pos)):
        return None
    return pos


def calculate_ball_position(timeseries, keypoint_map, frame):
    """
    Calculate ball center position from finger keypoints.

    The ball rests on the fingertips during the shooting motion.
    We use a weighted average of fingertip positions plus an offset
    for the ball center being above the fingers.
    """
    # Right hand fingertips (distal phalanges)
    fingertips = [
        'right_first_finger_distal',   # Thumb
        'right_second_finger_distal',  # Index
        'right_third_finger_distal',   # Middle
        'right_fourth_finger_distal',  # Ring
        'right_fifth_finger_distal',   # Pinky
    ]

    positions = []
    weights = [0.15, 0.25, 0.30, 0.20, 0.10]  # Middle and index finger most important

    for tip in fingertips:
        pos = get_position(timeseries, keypoint_map, tip, frame)
        if pos is not None:
            positions.append(pos)
        else:
            weights.pop(len(positions))  # Remove weight if position missing

    if len(positions) < 3:
        # Fallback to wrist
        wrist = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if wrist is not None:
            return wrist + np.array([0, 0, BALL_RADIUS])
        return None

    # Weighted centroid of fingertips
    weights = np.array(weights[:len(positions)])
    weights = weights / weights.sum()

    fingertip_center = np.zeros(3)
    for pos, w in zip(positions, weights):
        fingertip_center += pos * w

    # Ball center is above fingertips by approximately ball radius
    # Direction: perpendicular to palm, which we estimate from wrist-to-fingertips
    wrist = get_position(timeseries, keypoint_map, 'right_wrist', frame)
    if wrist is not None:
        palm_direction = fingertip_center - wrist
        palm_direction = palm_direction / (np.linalg.norm(palm_direction) + 1e-6)
        ball_center = fingertip_center + palm_direction * BALL_RADIUS * 0.5
    else:
        ball_center = fingertip_center + np.array([0, 0, BALL_RADIUS])

    return ball_center


def detect_release_frame(timeseries, keypoint_map):
    """
    Detect ball release frame by finding when fingertip velocity peaks.

    Release occurs when the fingers are moving fastest (just before they
    lose contact with the ball).
    """
    # Use middle finger distal as primary indicator
    finger_name = 'right_third_finger_distal'

    positions = []
    for frame in range(60, 200):
        pos = get_position(timeseries, keypoint_map, finger_name, frame)
        if pos is not None:
            positions.append((frame, pos))

    if len(positions) < 20:
        return 150  # Fallback

    frames = np.array([p[0] for p in positions])
    z_positions = np.array([p[1][2] for p in positions])

    # Compute velocity using Savitzky-Golay
    try:
        z_velocity = savgol_filter(z_positions, window_length=11, polyorder=3, deriv=1) * FPS
    except:
        z_velocity = np.gradient(z_positions) * FPS

    # Release is at peak upward velocity
    peak_idx = np.argmax(z_velocity)
    return frames[peak_idx]


def calculate_release_velocity(timeseries, keypoint_map, release_frame):
    """
    Calculate ball velocity at release.

    The ball velocity combines:
    1. Forward momentum from arm swing (elbow/shoulder motion)
    2. Upward/forward flick from wrist/fingers

    We need to capture the FULL velocity, not just fingertip motion at the
    exact release moment.
    """
    # Get ball positions around release - use asymmetric window
    # More frames BEFORE release (where forward motion happens)
    window_before = 20
    window_after = 5
    start = max(0, release_frame - window_before)
    end = min(240, release_frame + window_after)

    positions = []
    valid_frames = []

    for frame in range(start, end):
        pos = calculate_ball_position(timeseries, keypoint_map, frame)
        if pos is not None:
            positions.append(pos)
            valid_frames.append(frame)

    if len(positions) < 11:
        # Fallback: estimate from hoop direction
        ball_pos = calculate_ball_position(timeseries, keypoint_map, release_frame)
        if ball_pos is None:
            return np.array([-15, 0, 12])  # Default toward hoop with arc

        to_hoop = HOOP_POSITION - ball_pos
        horizontal_dist = np.sqrt(to_hoop[0]**2 + to_hoop[1]**2)

        # For projectile motion to hit target at same height:
        # v = sqrt(g * d) for 45 degrees
        # But we're shooting at a higher target, so need more speed
        speed = np.sqrt(GRAVITY * horizontal_dist * 1.2)

        # Direction: toward hoop with ~50 degree angle
        horizontal_dir = to_hoop[:2] / (np.linalg.norm(to_hoop[:2]) + 1e-6)
        angle_rad = np.radians(50)
        velocity = np.array([
            horizontal_dir[0] * speed * np.cos(angle_rad),
            horizontal_dir[1] * speed * np.cos(angle_rad),
            speed * np.sin(angle_rad)
        ])
        return velocity

    positions = np.array(positions)

    # Use Savitzky-Golay to get smooth velocity
    # Longer window for better capture of motion
    window_len = min(15, len(positions) - 2)
    if window_len % 2 == 0:
        window_len -= 1
    if window_len < 5:
        window_len = 5

    try:
        vx = savgol_filter(positions[:, 0], window_length=window_len, polyorder=3, deriv=1) * FPS
        vy = savgol_filter(positions[:, 1], window_length=window_len, polyorder=3, deriv=1) * FPS
        vz = savgol_filter(positions[:, 2], window_length=window_len, polyorder=3, deriv=1) * FPS
    except:
        vx = np.gradient(positions[:, 0]) * FPS
        vy = np.gradient(positions[:, 1]) * FPS
        vz = np.gradient(positions[:, 2]) * FPS

    # Get velocity at release frame
    idx = np.searchsorted(valid_frames, release_frame)
    idx = min(idx, len(vx) - 1)

    raw_velocity = np.array([vx[idx], vy[idx], vz[idx]])

    # Calculate required velocities using projectile motion
    ball_pos = positions[idx]
    to_hoop = HOOP_POSITION - ball_pos

    x_dist = to_hoop[0]  # Distance in X direction (negative = toward hoop)
    y_dist = to_hoop[1]  # Distance in Y direction
    horizontal_dist = np.sqrt(x_dist**2 + y_dist**2)
    height_diff = to_hoop[2]  # Height difference (positive = hoop is higher)

    # For projectile motion with release angle theta:
    # x = v*cos(theta)*t
    # z = z0 + v*sin(theta)*t - 0.5*g*t^2
    #
    # At hoop: x = horizontal_dist, z = z0 + height_diff
    # Solving: v^2 = g*x^2 / (2*cos^2(theta) * (x*tan(theta) - height_diff))
    #
    # For a given angle, this gives the required speed.

    # Extract release angle from raw velocity (the DIRECTION is the key signal)
    raw_horizontal = np.sqrt(raw_velocity[0]**2 + raw_velocity[1]**2)
    raw_vertical = raw_velocity[2]

    if raw_horizontal > 0.5 and raw_vertical > 0.5:
        # Use extracted angle - this captures the variation in release angle
        extracted_angle = np.arctan2(raw_vertical, raw_horizontal)
    else:
        # Default 52 degrees (typical free throw)
        extracted_angle = np.radians(52)

    # Allow wider angle range to preserve variation (35-65 degrees)
    # This lets shots be "too flat" or "too steep" which affects depth
    release_angle = np.clip(extracted_angle, np.radians(35), np.radians(65))

    # Calculate required speed for this angle
    cos_theta = np.cos(release_angle)
    tan_theta = np.tan(release_angle)

    denominator = horizontal_dist * tan_theta - height_diff
    if denominator <= 0:
        # Can't reach hoop with this angle - use steeper angle
        release_angle = np.radians(55)
        cos_theta = np.cos(release_angle)
        tan_theta = np.tan(release_angle)
        denominator = horizontal_dist * tan_theta - height_diff

    if denominator > 0:
        v_squared = (GRAVITY * horizontal_dist**2) / (2 * cos_theta**2 * denominator)
        speed = np.sqrt(v_squared)
    else:
        # Fallback
        speed = 22.0  # ft/s typical free throw

    # Clamp speed to reasonable range
    speed = np.clip(speed, 18.0, 26.0)

    # Horizontal direction (toward hoop)
    horizontal_dir = to_hoop[:2] / (horizontal_dist + 1e-6)

    # Compute final velocity
    vh = speed * np.cos(release_angle)
    vz = speed * np.sin(release_angle)

    # The main velocity is toward the hoop
    # Lateral variation affects left/right aim (perpendicular to hoop direction)

    # Extract lateral component from raw velocity
    # The perpendicular direction to hoop is [-horizontal_dir[1], horizontal_dir[0]]
    perp_dir = np.array([-horizontal_dir[1], horizontal_dir[0]])

    # Project raw velocity onto perpendicular direction to get lateral component
    # This captures the left/right aim variation in the shooting motion
    if raw_horizontal > 0.5:
        raw_xy = np.array([raw_velocity[0], raw_velocity[1]])
        lateral_component = np.dot(raw_xy, perp_dir)
        # Allow more lateral variation - up to 10% of horizontal velocity
        # A 10% lateral gives ~6 degree angle which causes ~13 inch deviation at 12 feet
        max_lateral = vh * 0.15  # 15% max
        lateral_vel = np.clip(lateral_component, -max_lateral, max_lateral)
    else:
        lateral_vel = 0

    # Also add variation from the Y component of ball position
    # If the shooter's hand is off-center, that affects where the ball goes
    ball_pos = positions[idx]
    shooter_center_y = -25.0  # Approximate Y position of shooter
    y_offset = ball_pos[1] - shooter_center_y
    # Convert Y offset to lateral velocity (roughly: 1 foot offset = 1 ft/s lateral)
    lateral_from_position = y_offset * 0.5

    lateral_vel += lateral_from_position
    lateral_vel = np.clip(lateral_vel, -vh * 0.2, vh * 0.2)  # Cap at 20%

    # Final velocity: main direction + perpendicular lateral + vertical
    velocity = np.array([
        horizontal_dir[0] * vh + perp_dir[0] * lateral_vel,
        horizontal_dir[1] * vh + perp_dir[1] * lateral_vel,
        vz
    ])

    return velocity


def calculate_spin(timeseries, keypoint_map, release_frame):
    """
    Calculate ball spin from differential finger motion.

    Backspin is created by fingers flicking - the index/middle fingers
    move faster than the ring/pinky at release.
    """
    dt = 3  # frames

    # Get index and ring finger velocities
    index_pre = get_position(timeseries, keypoint_map, 'right_second_finger_distal', release_frame - dt)
    index_post = get_position(timeseries, keypoint_map, 'right_second_finger_distal', release_frame + dt)
    ring_pre = get_position(timeseries, keypoint_map, 'right_fourth_finger_distal', release_frame - dt)
    ring_post = get_position(timeseries, keypoint_map, 'right_fourth_finger_distal', release_frame + dt)

    if any(p is None for p in [index_pre, index_post, ring_pre, ring_post]):
        return 3.0  # Default backspin in Hz

    index_vel = (index_post - index_pre) * FPS / (2 * dt)
    ring_vel = (ring_post - ring_pre) * FPS / (2 * dt)

    # Differential velocity (tangential)
    diff_vel = np.linalg.norm(index_vel - ring_vel)

    # Convert to angular velocity
    # Assuming fingers are ~3 inches apart on ball surface
    finger_separation = 3.0 / 12.0  # feet
    angular_vel = diff_vel / (BALL_RADIUS + finger_separation / 2)  # rad/s
    spin_hz = angular_vel / (2 * np.pi)

    # Typical backspin: 2-4 Hz
    spin_hz = np.clip(spin_hz, 0, 6)

    return spin_hz


def simulate_trajectory(pos, vel, spin_hz, dt=0.001, max_time=2.0):
    """
    Simulate ball trajectory with gravity and air resistance.

    Args:
        pos: Initial position [x, y, z] in feet
        vel: Initial velocity [vx, vy, vz] in ft/s
        spin_hz: Backspin in Hz (affects Magnus force)
        dt: Time step in seconds
        max_time: Maximum simulation time

    Returns:
        Dictionary with landing position and entry angle at rim plane
    """
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)

    # Convert spin to rad/s
    spin_omega = spin_hz * 2 * np.pi  # rad/s

    # Simulation loop
    t = 0
    prev_pos = pos.copy()

    while t < max_time:
        # Gravity
        accel = np.array([0, 0, -GRAVITY])

        # Air resistance (drag)
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            # F_drag = 0.5 * rho * Cd * A * v^2
            drag_magnitude = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * BALL_AREA * speed**2
            drag_accel = -drag_magnitude / BALL_MASS * vel / speed
            accel += drag_accel * 0.00685  # Convert lb to mass units

        # Magnus force (simplified - backspin creates lift)
        if spin_omega > 0 and speed > 0.1:
            # Magnus force perpendicular to velocity and spin axis
            # For backspin, this creates upward force when moving forward
            # Simplified: lift proportional to spin and forward velocity
            horizontal_speed = np.sqrt(vel[0]**2 + vel[1]**2)
            magnus_lift = 0.1 * spin_omega * horizontal_speed / GRAVITY  # Simplified coefficient
            accel[2] += magnus_lift * 0.5  # Small effect

        # Euler integration
        vel = vel + accel * dt
        pos = pos + vel * dt
        t += dt

        # Check if ball is at or near rim height (Z = 10 feet)
        # and moving downward (descending into hoop area)
        # We want to catch when ball crosses the rim plane from above

        rim_height = HOOP_POSITION[2]  # 10 feet
        prev_z = prev_pos[2] if 'prev_pos' in dir() else pos[2] - vel[2] * dt

        # Ball crosses rim height going DOWN
        crossed_down = prev_z > rim_height and pos[2] <= rim_height

        if crossed_down:
            # Linear interpolation to exact crossing at Z = rim_height
            alpha = (rim_height - prev_z) / (pos[2] - prev_z + 1e-9)

            crossing_x = prev_pos[0] + vel[0] * dt * alpha
            crossing_y = prev_pos[1] + vel[1] * dt * alpha
            crossing_z = rim_height

            # Check if the ball is near the hoop horizontally
            # Hoop is at [5.25, -25], ball should be within reasonable range
            dist_to_hoop_center = np.sqrt((crossing_x - HOOP_POSITION[0])**2 +
                                          (crossing_y - HOOP_POSITION[1])**2)

            # If too far from hoop (> 5 feet), this is a miss, not a crossing
            if dist_to_hoop_center > 5.0:
                prev_y = pos[1]
                prev_pos = pos.copy()
                continue

            # Entry angle (degrees from horizontal)
            # This is the angle at which the ball descends into the hoop
            horizontal_vel = np.sqrt(vel[0]**2 + vel[1]**2)
            if horizontal_vel > 0.1:
                entry_angle = np.degrees(np.arctan2(abs(vel[2]), horizontal_vel))
            else:
                entry_angle = 90.0  # Vertical drop

            # Convert to output units
            # The coordinate system has:
            # - Origin [0, 0, 0] at top-left of court
            # - Hoop at [5.25, -25, 10]
            #
            # depth: distance from FRONT of hoop along hoop's AXIS (toward/away from shooter)
            #        The hoop axis points from hoop toward shooter (increasing X)
            #        Positive depth = ball went past front of rim (long)
            #        Negative depth = ball short of rim
            #        Since we cross at Y=-25 (the rim plane), depth relates to X position
            #        Front of rim is at X = 5.25 + HOOP_RADIUS = 6.0 feet
            #        (rim extends toward shooter)
            #
            # left_right: lateral displacement from CENTER of hoop
            #        Hoop center X = 5.25, but that's the basket center
            #        Wait, the problem says depth is Y axis and left_right is X axis
            #        Let me re-read...
            #
            # From problem: "depth is identified as the y axis", "left_right is identified as x axis"
            # This means in THEIR coordinate system:
            #   - y axis = depth (toward/away from shooter)
            #   - x axis = left_right (lateral)
            #
            # In OUR coordinate system:
            #   - Our Y is lateral (both shooter and hoop at Y ~ -25)
            #   - Our X is depth direction (shooter at X~18, hoop at X~5)
            #
            # Wait, let me check the hoop center again: [5.25, -25, 10]
            # This means the hoop center is at X=5.25, Y=-25
            # The shooter is at X~18, Y~-25
            # So the shot goes from X~18 toward X~5 (decreasing X = toward hoop)
            #
            # For depth (their Y axis = our X axis):
            #   - The ball crosses the rim plane (our Y = -25)
            #   - At that point, depth = how far forward/back along X relative to hoop center
            #   - If ball X < 5.25, it's past the hoop center (long)
            #   - If ball X > 5.25, it's short
            #   - Depth = 5.25 - crossing_x (positive = short, negative = long)
            #   - Wait, the problem says positive depth = ... let me check
            #
            # Actually, looking at typical basketball stats:
            #   - depth > 0 means ball is long (past center toward back)
            #   - depth < 0 means ball is short (front of rim)
            #
            # For left_right (their X axis):
            #   - At the rim plane, lateral deviation from center
            #   - In our coords, we need to see how far in Y the ball is from -25
            #   - Wait, we're CROSSING at Y=-25, so Y deviation is by definition 0
            #   - That doesn't work...
            #
            # I think the "rim plane" in the problem means the horizontal plane at Z=10 (rim height)
            # NOT the vertical plane at Y=-25.
            #
            # Let me reconsider: the rim is at Z=10 feet. When the ball crosses this plane:
            #   - depth = Y position relative to rim center (forward/back)
            #   - left_right = X position relative to rim center (left/right)
            #   - angle = entry angle
            #
            # Rim center: [5.25, -25, 10]
            # When ball crosses Z=10:
            #   - depth = crossing_Y - (-25) = crossing_Y + 25 (but wait, both are ~-25)
            #   - This still doesn't make sense...

            # Let me just use simple interpretation:
            # At the crossing point (where ball is at the rim plane):
            # - depth: how far short/long (along the shot direction)
            # - left_right: how far left/right of center

            # The shot direction is from shooter to hoop: approximately [-1, 0, varies]
            # At rim plane (Y=-25), the ball has some X position.
            # depth = distance from rim center along X axis
            # left_right = distance from rim center perpendicular to shot (approximately Y deviation)

            # But we crossed at Y=-25, so Y at crossing is exactly -25...
            # Unless we're crossing the Z=10 plane instead.

            # I'll use the simpler interpretation:
            # - At crossing (Y=-25), depth = X deviation, lr = Z deviation? No...

            # Now we're crossing at Z = 10 (rim height)
            # Hoop center is at [5.25, -25, 10]
            #
            # From the problem definition:
            # - depth (Y axis in their system): distance from front of hoop along hoop axis
            #   The hoop axis points from backboard toward shooter
            #   In our coords, that's the +X direction (from X=5.25 toward shooter at X~18)
            #   So depth = crossing_x - 5.25, but this needs to be from FRONT of hoop
            #   Front of hoop = center + radius toward shooter = 5.25 + 0.75 = 6.0
            #   depth = crossing_x - 6.0, but that would be negative for shots going in
            #   Actually: positive depth = ball center is past front (into hoop)
            #   So: depth = 6.0 - crossing_x (positive if ball is before the front rim, i.e., "in")
            #   Hmm, this is confusing. Let me just use:
            #   depth = crossing_y - HOOP_POSITION[1] (how far from hoop center in Y)
            #
            # - left_right (X axis in their system): lateral displacement from center
            #   In our coords, lateral from shot direction would be perpendicular
            #   Since shot goes mainly in -X direction, perpendicular is Y direction
            #   So left_right relates to crossing_y deviation from hoop center Y

            # Actually, let me just directly compute:
            # The hoop center in XY is at [5.25, -25]
            # depth = how far the ball is from the hoop center along the SHOT direction
            # left_right = how far the ball is from center perpendicular to shot

            # Shot direction: from shooter (~[18, -25]) to hoop ([5.25, -25])
            # This is [-12.75, 0] normalized = [-1, 0]
            # So shot direction is purely in -X
            # Perpendicular is Y direction

            # At crossing:
            # depth (along shot direction) = -(crossing_x - 5.25) = 5.25 - crossing_x
            #   Positive depth = ball hasn't reached hoop center yet (short)
            #   Negative depth = ball went past hoop center (long)
            #   Wait, in basketball stats, positive depth usually means long (past center)
            #   So: depth = crossing_x - 5.25 (positive if ball went past, i.e., long)
            #   But crossing_x < 5.25 for shots that go "into" the hoop
            #   I'll use: depth = HOOP_POSITION[0] - crossing_x (positive = ball is short of center)

            # left_right = crossing_y - HOOP_POSITION[1] (positive = right of center)

            # In our coordinate system:
            # - Shot direction is -X (from shooter at X~18 to hoop at X~5.25)
            # - Lateral direction is Y
            #
            # In the problem's coordinate system (at hoop):
            # - depth (their Y) = along hoop axis = our X direction
            # - left_right (their X) = lateral = our Y direction
            #
            # At crossing (Z=10):
            # - depth = X deviation from hoop center (positive = ball past hoop toward shooter, short)
            #   Wait, if ball is at X=6 and hoop center at X=5.25, the ball is 0.75 feet
            #   toward the shooter from center. Is that "short" or "long"?
            #   In basketball, "depth" typically means front-to-back of the rim.
            #   Positive depth often means the ball went past the front of the rim (deep into hoop).
            #   So depth = hoop_center_x - crossing_x (positive if ball is past center toward back)
            #
            # - left_right = Y deviation from hoop center
            #   Positive = ball is to the right (depends on which way you're facing)
            #   If facing the hoop from the free throw line, right is... -Y direction? +Y direction?
            #   The problem says "positive is right" in their X axis.
            #   Let's assume: left_right = crossing_y - hoop_center_y, and adjust sign if needed.

            # For now, let me try the simple mapping and check correlations:
            depth_feet = HOOP_POSITION[0] - crossing_x  # Positive = ball past hoop center (long)
            left_right_feet = crossing_y - HOOP_POSITION[1]  # Y deviation from center

            # Convert to inches
            depth_inches = depth_feet * 12
            left_right_inches = left_right_feet * 12

            return {
                'success': True,
                'crossing_pos': [crossing_x, crossing_y, crossing_z],
                'entry_angle': entry_angle,
                'depth_inches': depth_inches,
                'left_right_inches': left_right_inches,
                'time': t,
            }

        prev_pos = pos.copy()

        # Ball hit ground
        if pos[2] < 0:
            return {'success': False, 'reason': 'ground'}

    return {'success': False, 'reason': 'timeout'}


def main():
    print("=" * 80)
    print("ACCURATE PHYSICS SIMULATION")
    print("=" * 80)
    print(f"Hoop position: {HOOP_POSITION} feet")
    print(f"Ball radius: {BALL_RADIUS:.3f} feet ({BALL_RADIUS*12:.1f} inches)")
    print()

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)
    scalers = load_scalers()

    # Process training data
    print("Processing training data...")

    results = []

    for metadata, timeseries in iterate_shots(train=True):
        shot_id = metadata['id']
        player_id = metadata['participant_id']

        # Ground truth targets
        true_angle = metadata['angle']
        true_depth = metadata['depth']
        true_lr = metadata['left_right']

        # Detect release frame
        release_frame = detect_release_frame(timeseries, keypoint_map)

        # Calculate ball position at release
        ball_pos = calculate_ball_position(timeseries, keypoint_map, release_frame)
        if ball_pos is None:
            results.append({
                'id': shot_id, 'success': False, 'reason': 'no_position'
            })
            continue

        # Calculate velocity
        ball_vel = calculate_release_velocity(timeseries, keypoint_map, release_frame)

        # Calculate spin
        spin_hz = calculate_spin(timeseries, keypoint_map, release_frame)

        # Simulate
        sim_result = simulate_trajectory(ball_pos, ball_vel, spin_hz)

        if sim_result['success']:
            results.append({
                'id': shot_id,
                'player_id': player_id,
                'success': True,
                'release_frame': release_frame,
                'ball_pos': ball_pos,
                'ball_vel': ball_vel,
                'speed': np.linalg.norm(ball_vel),
                'spin_hz': spin_hz,
                'pred_angle': sim_result['entry_angle'],
                'pred_depth': sim_result['depth_inches'],
                'pred_lr': sim_result['left_right_inches'],
                'true_angle': true_angle,
                'true_depth': true_depth,
                'true_lr': true_lr,
            })
        else:
            results.append({
                'id': shot_id,
                'success': False,
                'reason': sim_result.get('reason', 'unknown'),
                'ball_pos': ball_pos,
                'ball_vel': ball_vel,
            })

    # Analyze results
    df = pd.DataFrame(results)
    success_df = df[df['success'] == True]

    print(f"\nResults:")
    print(f"  Total shots: {len(df)}")
    print(f"  Successful simulations: {len(success_df)} ({100*len(success_df)/len(df):.1f}%)")

    if len(success_df) > 0:
        print(f"\nPrediction analysis:")

        # Correlations
        from scipy.stats import pearsonr

        for target in ['angle', 'depth', 'lr']:
            pred_col = f'pred_{target}'
            true_col = f'true_{target}'

            if pred_col in success_df.columns:
                corr, pval = pearsonr(success_df[pred_col], success_df[true_col])
                mse = np.mean((success_df[pred_col] - success_df[true_col])**2)
                print(f"  {target}: r={corr:.4f}, MSE={mse:.4f}")

        print(f"\nVelocity statistics:")
        print(f"  Speed: {success_df['speed'].mean():.2f} +/- {success_df['speed'].std():.2f} ft/s")

        print(f"\nPrediction ranges:")
        print(f"  angle: {success_df['pred_angle'].min():.1f} to {success_df['pred_angle'].max():.1f} degrees")
        print(f"  depth: {success_df['pred_depth'].min():.1f} to {success_df['pred_depth'].max():.1f} inches")
        print(f"  left_right: {success_df['pred_lr'].min():.1f} to {success_df['pred_lr'].max():.1f} inches")

        print(f"\nTrue ranges:")
        print(f"  angle: {success_df['true_angle'].min():.1f} to {success_df['true_angle'].max():.1f} degrees")
        print(f"  depth: {success_df['true_depth'].min():.1f} to {success_df['true_depth'].max():.1f} inches")
        print(f"  left_right: {success_df['true_lr'].min():.1f} to {success_df['true_lr'].max():.1f} inches")

    # Check failed simulations
    failed = df[df['success'] == False]
    if len(failed) > 0:
        print(f"\nFailed simulations:")
        reasons = failed['reason'].value_counts()
        for reason, count in reasons.items():
            print(f"  {reason}: {count}")

        # Check velocities of failed shots
        failed_with_vel = failed[failed['ball_vel'].notna()]
        if len(failed_with_vel) > 0:
            print(f"\nFailed shot velocities (sample):")
            for _, row in failed_with_vel.head(5).iterrows():
                vel = row['ball_vel']
                pos = row['ball_pos']
                print(f"  pos={pos}, vel={vel}, speed={np.linalg.norm(vel):.1f} ft/s")


if __name__ == "__main__":
    main()
