"""
Inverse ballistics solver for basketball shots.

Given:
- Release position (from wrist tracking)
- Landing outcomes (angle, depth, left_right from competition data)
- Hoop position (calibrated from data)

Solve for:
- Release velocity (vx, vy, vz) that produces the observed outcomes

This provides "ground truth" velocities to train the velocity predictor.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional
import warnings

# Gravity constant (will be calibrated from data if units are unknown)
G_DEFAULT = 9.81  # m/s^2 in SI units

# Hoop parameters (will be calibrated from data)
HOOP_HEIGHT_DEFAULT = 3.05  # meters (10 feet)
HOOP_DISTANCE_DEFAULT = 4.19  # meters (free throw line distance)


def forward_ballistics(
    release_pos: np.ndarray,
    release_vel: np.ndarray,
    g: float = G_DEFAULT
) -> Dict[str, float]:
    """
    Forward ballistic physics: given release position and velocity, calculate outcomes.

    Args:
        release_pos: [x, y, z] release position
        release_vel: [vx, vy, vz] release velocity
        g: gravity constant in data units

    Returns:
        Dictionary with:
        - landing_x, landing_y, landing_z: landing position
        - landing_vx, landing_vy, landing_vz: landing velocity
        - entry_angle: angle of entry (degrees)
        - time_of_flight: time to reach hoop (seconds)
    """
    x0, y0, z0 = release_pos
    vx, vy, vz = release_vel

    # Solve quadratic equation for time to reach hoop height z=0
    # z(t) = z0 + vz*t - 0.5*g*t^2 = 0
    a = -0.5 * g
    b = vz
    c = z0

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        # Ball doesn't reach ground
        return {"valid": False}

    # Two solutions: take the positive one that's further in time (descending arc)
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    t = max(t1, t2)

    if t <= 0:
        return {"valid": False}

    # Landing position
    landing_x = x0 + vx * t
    landing_y = y0 + vy * t
    landing_z = 0.0  # Ground level

    # Landing velocity
    landing_vx = vx
    landing_vy = vy
    landing_vz = vz - g * t

    # Entry angle: angle of velocity vector with horizontal plane
    vel_horiz = np.sqrt(landing_vx**2 + landing_vy**2)
    entry_angle = np.arctan2(-landing_vz, vel_horiz) * 180 / np.pi  # Negative because downward

    return {
        "valid": True,
        "landing_x": landing_x,
        "landing_y": landing_y,
        "landing_z": landing_z,
        "landing_vx": landing_vx,
        "landing_vy": landing_vy,
        "landing_vz": landing_vz,
        "entry_angle": entry_angle,
        "time_of_flight": t
    }


def calculate_outcomes_from_landing(
    landing_pos: np.ndarray,
    landing_vel: np.ndarray,
    hoop_pos: np.ndarray
) -> Dict[str, float]:
    """
    Calculate competition outcomes (angle, depth, left_right) from landing physics.

    Args:
        landing_pos: [x, y, z] where ball lands
        landing_vel: [vx, vy, vz] velocity at landing
        hoop_pos: [x, y, z] hoop center position

    Returns:
        angle, depth, left_right in competition units
    """
    # Entry angle: angle of velocity vector with horizontal
    vel_horiz = np.sqrt(landing_vel[0]**2 + landing_vel[1]**2)
    entry_angle = np.arctan2(-landing_vel[2], vel_horiz) * 180 / np.pi

    # Depth: distance from hoop center in forward direction (y-axis)
    # Positive depth = ball lands past hoop (long)
    # Negative depth = ball lands short of hoop (short)
    depth = landing_pos[1] - hoop_pos[1]

    # Left/Right: lateral deviation from hoop center (x-axis)
    # Negative = left, positive = right
    left_right = landing_pos[0] - hoop_pos[0]

    return {
        "angle": entry_angle,
        "depth": depth,
        "left_right": left_right
    }


def inverse_ballistics_objective(
    velocity: np.ndarray,
    release_pos: np.ndarray,
    target_angle: float,
    target_depth: float,
    target_left_right: float,
    hoop_pos: np.ndarray,
    g: float,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Objective function for inverse ballistics optimization.

    Minimize squared error between predicted and actual outcomes.
    """
    # Forward physics
    result = forward_ballistics(release_pos, velocity, g)

    if not result["valid"]:
        return 1e10  # Large penalty for invalid trajectories

    # Calculate outcomes
    landing_pos = np.array([result["landing_x"], result["landing_y"], result["landing_z"]])
    landing_vel = np.array([result["landing_vx"], result["landing_vy"], result["landing_vz"]])

    outcomes = calculate_outcomes_from_landing(landing_pos, landing_vel, hoop_pos)

    # Weighted squared error
    error = (
        weights[0] * (outcomes["angle"] - target_angle)**2 +
        weights[1] * (outcomes["depth"] - target_depth)**2 +
        weights[2] * (outcomes["left_right"] - target_left_right)**2
    )

    return error


def solve_inverse_ballistics(
    release_pos: np.ndarray,
    target_angle: float,
    target_depth: float,
    target_left_right: float,
    hoop_pos: np.ndarray,
    g: float = G_DEFAULT,
    initial_guess: Optional[np.ndarray] = None,
    method: str = "local"
) -> Dict[str, any]:
    """
    Solve for release velocity that produces observed outcomes.

    Args:
        release_pos: [x, y, z] release position
        target_angle: observed entry angle (degrees)
        target_depth: observed depth (inches or data units)
        target_left_right: observed lateral deviation
        hoop_pos: [x, y, z] hoop position
        g: gravity constant
        initial_guess: optional initial velocity guess
        method: "local" (L-BFGS-B) or "global" (differential evolution)

    Returns:
        Dictionary with:
        - velocity: [vx, vy, vz] solved velocity
        - error: final optimization error
        - success: whether optimization converged
        - predicted_angle, predicted_depth, predicted_left_right: outcomes from solved velocity
    """
    # Initial guess if not provided
    if initial_guess is None:
        # Simple projectile guess: aim at hoop with reasonable velocity
        dx = hoop_pos[0] - release_pos[0]
        dy = hoop_pos[1] - release_pos[1]
        dz = hoop_pos[2] - release_pos[2]

        # Estimate time of flight (typical ~0.6-1.0 seconds for free throw)
        t_guess = 0.7

        vx_guess = dx / t_guess
        vy_guess = dy / t_guess
        vz_guess = (dz + 0.5 * g * t_guess**2) / t_guess

        initial_guess = np.array([vx_guess, vy_guess, vz_guess])

    # Bounds: reasonable velocity ranges for basketball shots
    # Magnitude typically 3-10 m/s or equivalent in data units
    vel_mag_max = 15.0
    bounds = [(-vel_mag_max, vel_mag_max) for _ in range(3)]

    # Weights: normalize by typical std of each target
    # angle std ~5°, depth std ~5 units, left_right std ~4 units
    weights = (1.0, 1.0, 1.0)  # Equal weights for now

    if method == "global":
        # Global optimization (slower but more robust)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                inverse_ballistics_objective,
                bounds=bounds,
                args=(release_pos, target_angle, target_depth, target_left_right, hoop_pos, g, weights),
                maxiter=1000,
                atol=0.01,
                seed=42
            )
        velocity = result.x
        error = result.fun
        success = result.success
    else:
        # Local optimization (faster)
        result = minimize(
            inverse_ballistics_objective,
            initial_guess,
            args=(release_pos, target_angle, target_depth, target_left_right, hoop_pos, g, weights),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500}
        )
        velocity = result.x
        error = result.fun
        success = result.success

    # Calculate predicted outcomes
    forward_result = forward_ballistics(release_pos, velocity, g)

    if forward_result["valid"]:
        landing_pos = np.array([forward_result["landing_x"], forward_result["landing_y"], forward_result["landing_z"]])
        landing_vel = np.array([forward_result["landing_vx"], forward_result["landing_vy"], forward_result["landing_vz"]])
        predicted_outcomes = calculate_outcomes_from_landing(landing_pos, landing_vel, hoop_pos)
    else:
        predicted_outcomes = {"angle": np.nan, "depth": np.nan, "left_right": np.nan}

    return {
        "velocity": velocity,
        "vx": velocity[0],
        "vy": velocity[1],
        "vz": velocity[2],
        "vel_magnitude": np.linalg.norm(velocity),
        "error": error,
        "success": success,
        "predicted_angle": predicted_outcomes["angle"],
        "predicted_depth": predicted_outcomes["depth"],
        "predicted_left_right": predicted_outcomes["left_right"],
        "time_of_flight": forward_result.get("time_of_flight", np.nan)
    }


def calibrate_hoop_position(
    release_positions: np.ndarray,
    observed_depths: np.ndarray,
    observed_left_rights: np.ndarray,
    release_height_mean: float
) -> np.ndarray:
    """
    Estimate hoop position from data.

    Assumption: shots with depth=0 and left_right=0 go through hoop center.

    Args:
        release_positions: (n_shots, 3) release positions
        observed_depths: (n_shots,) depth values
        observed_left_rights: (n_shots,) lateral deviations
        release_height_mean: typical release height

    Returns:
        [hoop_x, hoop_y, hoop_z] estimated hoop position
    """
    # Find shots close to perfect (depth ~ 0, left_right ~ 0)
    tolerance = 2.0  # within 2 units of perfect
    good_shots = (np.abs(observed_depths) < tolerance) & (np.abs(observed_left_rights) < tolerance)

    if good_shots.sum() == 0:
        # Fallback: use median landing position
        print("Warning: No shots near perfect, using median as hoop estimate")
        hoop_x = np.median(release_positions[:, 0])
        hoop_y = np.median(release_positions[:, 1]) + 4.0  # Assume 4 units forward
        hoop_z = 0.0  # Ground level
    else:
        # Use mean of near-perfect shots
        good_releases = release_positions[good_shots]
        hoop_x = np.mean(good_releases[:, 0])
        hoop_y = np.mean(good_releases[:, 1]) + 4.0  # Assume hoop is ~4 units forward from release
        hoop_z = 0.0

    return np.array([hoop_x, hoop_y, hoop_z])


def calibrate_gravity(
    release_positions: np.ndarray,
    release_velocities: np.ndarray,
    time_of_flight_estimates: np.ndarray
) -> float:
    """
    Calibrate gravity constant from observed trajectories.

    Args:
        release_positions: (n_shots, 3) release positions
        release_velocities: (n_shots, 3) measured velocities from tracking
        time_of_flight_estimates: (n_shots,) estimated flight times

    Returns:
        Calibrated gravity constant in data units
    """
    # z(t) = z0 + vz*t - 0.5*g*t^2
    # At landing: 0 = z0 + vz*t - 0.5*g*t^2
    # Solve for g: g = 2*(z0 + vz*t) / t^2

    z0 = release_positions[:, 2]
    vz = release_velocities[:, 2]
    t = time_of_flight_estimates

    # Filter valid data
    valid = (t > 0.1) & (t < 2.0) & ~np.isnan(z0) & ~np.isnan(vz)

    if valid.sum() < 10:
        print("Warning: Insufficient data for gravity calibration, using default")
        return G_DEFAULT

    g_estimates = 2 * (z0[valid] + vz[valid] * t[valid]) / (t[valid]**2)

    # Use median to be robust to outliers
    g_calibrated = np.median(g_estimates)

    # Sanity check: should be positive and reasonable magnitude
    if g_calibrated < 1.0 or g_calibrated > 50.0:
        print(f"Warning: Calibrated gravity {g_calibrated:.2f} is unreasonable, using default")
        return G_DEFAULT

    return g_calibrated


if __name__ == "__main__":
    print("Testing inverse ballistics solver...")

    # Test 1: Sanity check with known trajectory
    print("\n=== Test 1: Sanity Check ===")

    # Known parameters
    release_pos = np.array([0.0, 0.0, 2.0])
    true_velocity = np.array([0.5, 5.0, 8.0])
    hoop_pos = np.array([0.0, 4.5, 0.0])
    g = 9.81

    # Forward: calculate outcomes from known velocity
    forward_result = forward_ballistics(release_pos, true_velocity, g)
    print(f"Forward ballistics from true velocity:")
    print(f"  Entry angle: {forward_result['entry_angle']:.2f}°")
    print(f"  Landing: ({forward_result['landing_x']:.2f}, {forward_result['landing_y']:.2f}, {forward_result['landing_z']:.2f})")
    print(f"  Time of flight: {forward_result['time_of_flight']:.3f}s")

    landing_pos = np.array([forward_result['landing_x'], forward_result['landing_y'], forward_result['landing_z']])
    landing_vel = np.array([forward_result['landing_vx'], forward_result['landing_vy'], forward_result['landing_vz']])
    outcomes = calculate_outcomes_from_landing(landing_pos, landing_vel, hoop_pos)

    print(f"  Outcomes: angle={outcomes['angle']:.2f}°, depth={outcomes['depth']:.2f}, left_right={outcomes['left_right']:.2f}")

    # Inverse: recover velocity from outcomes
    print(f"\nInverse ballistics from outcomes:")
    inverse_result = solve_inverse_ballistics(
        release_pos,
        outcomes['angle'],
        outcomes['depth'],
        outcomes['left_right'],
        hoop_pos,
        g,
        method="local"
    )

    print(f"  Solved velocity: ({inverse_result['vx']:.2f}, {inverse_result['vy']:.2f}, {inverse_result['vz']:.2f})")
    print(f"  True velocity:   ({true_velocity[0]:.2f}, {true_velocity[1]:.2f}, {true_velocity[2]:.2f})")
    print(f"  Error: {inverse_result['error']:.6f}")
    print(f"  Velocity difference: {np.linalg.norm(inverse_result['velocity'] - true_velocity):.4f}")
    print(f"  Convergence: {inverse_result['success']}")

    # Test 2: Load real data and try inverse ballistics
    print("\n=== Test 2: Real Data ===")

    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from src.data_loader import load_single_shot, get_keypoint_columns
        from src.physics_features import init_keypoint_mapping, extract_physics_features

        # Initialize
        keypoint_cols = get_keypoint_columns()
        init_keypoint_mapping(keypoint_cols)

        # Load shot
        metadata, timeseries = load_single_shot(0, train=True)

        # Extract features
        feats = extract_physics_features(timeseries, smooth=True)

        # Release position and observed velocity
        release_pos_real = np.array([
            feats['wrist_x_release'],
            feats['wrist_y_release'],
            feats['wrist_z_release']
        ])

        observed_vel = np.array([
            feats['wrist_vx_release'],
            feats['wrist_vy_release'],
            feats['wrist_vz_release']
        ])

        # Target outcomes
        target_angle = metadata['angle']
        target_depth = metadata['depth']
        target_left_right = metadata['left_right']

        # Estimate hoop position (rough guess for now)
        hoop_pos_real = np.array([
            release_pos_real[0],  # Same x as release (centered)
            release_pos_real[1] + 4.0,  # 4 units forward
            0.0  # Ground level
        ])

        print(f"Shot {metadata['id']}:")
        print(f"  Release pos: {release_pos_real}")
        print(f"  Observed vel: {observed_vel}")
        print(f"  Targets: angle={target_angle:.2f}°, depth={target_depth:.2f}, left_right={target_left_right:.2f}")

        # Solve inverse ballistics with g=9.81 (will need calibration)
        inverse_result_real = solve_inverse_ballistics(
            release_pos_real,
            target_angle,
            target_depth,
            target_left_right,
            hoop_pos_real,
            g=9.81,
            initial_guess=observed_vel,
            method="local"
        )

        print(f"\nInverse ballistics solution:")
        print(f"  Solved vel: ({inverse_result_real['vx']:.2f}, {inverse_result_real['vy']:.2f}, {inverse_result_real['vz']:.2f})")
        print(f"  Error: {inverse_result_real['error']:.6f}")
        print(f"  Predicted: angle={inverse_result_real['predicted_angle']:.2f}°, "
              f"depth={inverse_result_real['predicted_depth']:.2f}, "
              f"left_right={inverse_result_real['predicted_left_right']:.2f}")

    except ImportError as e:
        print(f"Skipping real data test: {e}")

    print("\n✓ Inverse ballistics tests complete")
