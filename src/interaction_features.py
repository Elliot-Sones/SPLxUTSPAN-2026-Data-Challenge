"""
Interaction Feature Extraction for Bowling Motion Analysis.

Creates multiplicative feature interactions and ratios that may capture
combined effects not represented by individual features.

Key interaction types:
- Velocity * Extension products (power transfer)
- Directional ratios (trajectory shape)
- Temporal relationships (timing)
"""

import numpy as np
from typing import Dict, List, Optional
import warnings


def extract_interaction_features(base_features: Dict[str, float]) -> Dict[str, float]:
    """
    Extract multiplicative feature interactions from base features.

    Assumes base_features contains F4 hybrid features or derivative features
    with names like:
    - release_velocity_*, release_velocity_y, etc.
    - angle_*, knee_extension_rate, etc.

    Args:
        base_features: Dictionary of pre-computed features

    Returns:
        Dictionary of interaction features
    """
    interactions = {}

    # Helper to safely get features
    def get(name: str, default: float = 0.0) -> float:
        return base_features.get(name, default)

    # Helper for safe division
    def safe_div(a: float, b: float, default: float = 0.0) -> float:
        if abs(b) < 1e-10:
            return default
        return a / b

    # === Power Transfer Interactions ===
    # Velocity * knee extension (leg-to-arm power transfer)
    release_vel = get('release_velocity_magnitude', get('release_velocity', 0))
    knee_ext = get('knee_extension_rate', get('propulsion_knee_extension_rate', 0))
    interactions['power_velocity_x_knee_ext'] = release_vel * knee_ext

    # Velocity * hip drive
    hip_vel = get('hip_vertical_velocity', get('propulsion_hip_vertical_velocity', 0))
    interactions['power_velocity_x_hip_drive'] = release_vel * hip_vel

    # Arm extension * velocity (full extension at release)
    arm_ext = get('arm_extension_at_release', get('release_arm_extension', 0))
    interactions['power_arm_ext_x_velocity'] = arm_ext * release_vel

    # === Trajectory Shape Ratios ===
    # Horizontal vs forward velocity ratio (predicts left_right)
    vx = get('release_velocity_x', get('release_vx', 0))
    vy = get('release_velocity_y', get('release_vy', 0))
    vz = get('release_velocity_z', get('release_vz', 0))

    interactions['ratio_vx_vy'] = safe_div(vx, vy)  # Lateral vs forward
    interactions['ratio_vz_vy'] = safe_div(vz, vy)  # Arc vs distance
    interactions['ratio_vx_vz'] = safe_div(vx, vz)  # Lateral vs vertical

    # Combined horizontal component
    horizontal_v = np.sqrt(vx**2 + vz**2) if (vx != 0 or vz != 0) else 0
    interactions['ratio_horizontal_forward'] = safe_div(horizontal_v, abs(vy))

    # === Angle-Velocity Interactions ===
    # Release elevation * velocity (affects depth)
    elev_angle = get('release_elevation_angle', get('elevation_angle', 0))
    interactions['angle_x_velocity'] = elev_angle * release_vel

    # Wrist snap * velocity (spin control)
    wrist_snap = get('wrist_snap_angle', get('release_wrist_snap', 0))
    interactions['wrist_snap_x_velocity'] = wrist_snap * release_vel

    # Trunk lean * velocity (balance at release)
    trunk_lean = get('trunk_lean_at_release', get('trunk_lean', 0))
    interactions['trunk_lean_x_velocity'] = trunk_lean * release_vel

    # === Timing Interactions ===
    # Time-to-max-velocity * velocity (acceleration profile)
    time_to_max = get('time_to_max_velocity', get('velocity_time_to_max', 0))
    interactions['timing_x_velocity'] = time_to_max * release_vel

    # Release frame * velocity (late vs early release)
    release_frame = get('release_frame', get('detected_release_frame', 0))
    interactions['release_timing_x_velocity'] = (release_frame / 240.0) * release_vel

    # === Stability Interactions ===
    # Shoulder stability * velocity (controlled power)
    shoulder_std = get('right_shoulder_x_std', get('shoulder_stability', 0))
    interactions['stability_shoulder_x_velocity'] = safe_div(release_vel, shoulder_std + 0.01)

    # Hip stability * velocity
    hip_std = get('right_hip_x_std', get('hip_stability', 0))
    interactions['stability_hip_x_velocity'] = safe_div(release_vel, hip_std + 0.01)

    # === Body Coordination ===
    # Elbow angle * shoulder angle (arm position)
    elbow_angle = get('right_elbow_angle_mean', get('elbow_angle', 0))
    shoulder_angle = get('right_shoulder_angle_mean', get('shoulder_angle', 0))
    interactions['arm_position_elbow_x_shoulder'] = elbow_angle * shoulder_angle

    # Knee angle * hip angle (leg coordination)
    knee_angle = get('right_knee_angle_mean', get('knee_angle', 0))
    hip_angle = get('right_hip_angle_mean', get('hip_angle', 0))
    interactions['leg_position_knee_x_hip'] = knee_angle * hip_angle

    # === Phase Transitions ===
    # Velocity increase from loading to release
    loading_vel = get('loading_velocity_mean', get('phase1_velocity', 0))
    release_vel_phase = get('release_velocity_mean', get('phase3_velocity', release_vel))
    interactions['phase_velocity_increase'] = release_vel_phase - loading_vel
    interactions['phase_velocity_ratio'] = safe_div(release_vel_phase, loading_vel)

    # === Squared Terms (Non-linear effects) ===
    interactions['velocity_squared'] = release_vel ** 2
    interactions['knee_ext_squared'] = knee_ext ** 2
    interactions['arm_ext_squared'] = arm_ext ** 2

    # === Three-way Interactions (Most Important) ===
    # Velocity * arm extension * knee extension (full kinetic chain)
    interactions['kinetic_chain'] = release_vel * arm_ext * knee_ext

    # Velocity * elevation * horizontal ratio (trajectory control)
    interactions['trajectory_control'] = release_vel * elev_angle * interactions['ratio_horizontal_forward']

    return interactions


def extract_derived_ratios(base_features: Dict[str, float]) -> Dict[str, float]:
    """
    Extract ratio features that normalize values.
    """
    ratios = {}

    def get(name: str, default: float = 0.0) -> float:
        return base_features.get(name, default)

    def safe_div(a: float, b: float, default: float = 0.0) -> float:
        if abs(b) < 1e-10:
            return default
        return a / b

    # Velocity component ratios
    vel_mag = get('release_velocity_magnitude', get('release_velocity', 1))
    ratios['velocity_x_fraction'] = safe_div(get('release_velocity_x', get('release_vx', 0)), vel_mag)
    ratios['velocity_y_fraction'] = safe_div(get('release_velocity_y', get('release_vy', 0)), vel_mag)
    ratios['velocity_z_fraction'] = safe_div(get('release_velocity_z', get('release_vz', 0)), vel_mag)

    # Variability ratios (CV = std/mean)
    wrist_mean = get('right_wrist_x_mean', 1)
    wrist_std = get('right_wrist_x_std', 0)
    ratios['wrist_x_cv'] = safe_div(wrist_std, abs(wrist_mean))

    wrist_y_mean = get('right_wrist_y_mean', 1)
    wrist_y_std = get('right_wrist_y_std', 0)
    ratios['wrist_y_cv'] = safe_div(wrist_y_std, abs(wrist_y_mean))

    # Joint angle ratios
    elbow_mean = get('right_elbow_angle_mean', 1)
    elbow_range = get('right_elbow_angle_range', 0)
    ratios['elbow_rom_ratio'] = safe_div(elbow_range, elbow_mean)

    knee_mean = get('right_knee_angle_mean', 1)
    knee_range = get('right_knee_angle_range', 0)
    ratios['knee_rom_ratio'] = safe_div(knee_range, knee_mean)

    # Phase ratios
    phase0_mean = get('phase0_velocity_mean', 1)
    phase3_mean = get('phase3_velocity_mean', 0)
    ratios['velocity_phase_ratio'] = safe_div(phase3_mean, phase0_mean)

    return ratios


def extract_polynomial_features(base_features: Dict[str, float],
                                 key_features: Optional[List[str]] = None,
                                 degree: int = 2) -> Dict[str, float]:
    """
    Extract polynomial features for key variables.

    Args:
        base_features: Dictionary of base features
        key_features: List of feature names to create polynomials for
        degree: Polynomial degree (2 or 3)

    Returns:
        Dictionary of polynomial features
    """
    if key_features is None:
        # Default key features for polynomial expansion
        key_features = [
            'release_velocity_magnitude',
            'release_velocity',
            'knee_extension_rate',
            'arm_extension_at_release',
            'release_elevation_angle',
            'wrist_snap_angle',
        ]

    poly_features = {}

    for name in key_features:
        val = base_features.get(name, 0)
        if val != 0:
            if degree >= 2:
                poly_features[f'{name}_sq'] = val ** 2
            if degree >= 3:
                poly_features[f'{name}_cube'] = val ** 3

    return poly_features


def extract_all_interaction_features(base_features: Dict[str, float]) -> Dict[str, float]:
    """
    Extract all interaction features.

    Args:
        base_features: Dictionary of pre-computed features (F4 hybrid or derivatives)

    Returns:
        Dictionary of all interaction features
    """
    all_features = {}

    all_features.update(extract_interaction_features(base_features))
    all_features.update(extract_derived_ratios(base_features))
    all_features.update(extract_polynomial_features(base_features))

    return all_features


def get_interaction_feature_names() -> List[str]:
    """Return list of interaction feature names."""
    # Create dummy features to extract names
    dummy_base = {
        'release_velocity_magnitude': 1.0,
        'release_velocity': 1.0,
        'release_velocity_x': 0.5,
        'release_velocity_y': 0.8,
        'release_velocity_z': 0.3,
        'release_vx': 0.5,
        'release_vy': 0.8,
        'release_vz': 0.3,
        'knee_extension_rate': 0.1,
        'propulsion_knee_extension_rate': 0.1,
        'hip_vertical_velocity': 0.2,
        'propulsion_hip_vertical_velocity': 0.2,
        'arm_extension_at_release': 0.9,
        'release_arm_extension': 0.9,
        'release_elevation_angle': 15.0,
        'elevation_angle': 15.0,
        'wrist_snap_angle': 30.0,
        'release_wrist_snap': 30.0,
        'trunk_lean_at_release': 10.0,
        'trunk_lean': 10.0,
        'time_to_max_velocity': 0.6,
        'velocity_time_to_max': 0.6,
        'release_frame': 180.0,
        'detected_release_frame': 180.0,
        'right_shoulder_x_std': 0.05,
        'shoulder_stability': 0.05,
        'right_hip_x_std': 0.03,
        'hip_stability': 0.03,
        'right_elbow_angle_mean': 120.0,
        'elbow_angle': 120.0,
        'right_shoulder_angle_mean': 90.0,
        'shoulder_angle': 90.0,
        'right_knee_angle_mean': 150.0,
        'knee_angle': 150.0,
        'right_hip_angle_mean': 160.0,
        'hip_angle': 160.0,
        'loading_velocity_mean': 0.3,
        'phase1_velocity': 0.3,
        'release_velocity_mean': 1.0,
        'phase3_velocity': 1.0,
        'right_wrist_x_mean': 0.5,
        'right_wrist_x_std': 0.1,
        'right_wrist_y_mean': 0.8,
        'right_wrist_y_std': 0.15,
        'right_elbow_angle_range': 40.0,
        'right_knee_angle_range': 30.0,
        'phase0_velocity_mean': 0.2,
        'phase3_velocity_mean': 0.9,
    }

    features = extract_all_interaction_features(dummy_base)
    return list(features.keys())


if __name__ == '__main__':
    # Test with sample features
    sample_base = {
        'release_velocity_magnitude': 1.5,
        'release_velocity_x': 0.3,
        'release_velocity_y': 1.4,
        'release_velocity_z': 0.2,
        'knee_extension_rate': 0.15,
        'arm_extension_at_release': 0.95,
        'release_elevation_angle': 12.0,
        'wrist_snap_angle': 25.0,
        'right_elbow_angle_mean': 115.0,
        'right_shoulder_angle_mean': 85.0,
    }

    features = extract_all_interaction_features(sample_base)

    print(f"Total interaction features: {len(features)}")
    print("\nSample features:")
    for name, value in list(features.items())[:15]:
        print(f"  {name}: {value:.4f}")
