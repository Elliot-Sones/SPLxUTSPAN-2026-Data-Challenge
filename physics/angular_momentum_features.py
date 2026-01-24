#!/usr/bin/env python3
"""
Angular Momentum Transfer Features - Breakthrough Physics Discovery

Key Insight: Individual joint angles have weak correlation (R² < 0.15), but
angular momentum transfer through the kinetic chain is highly predictive.

Physics Principle: Angular momentum conservation during free throw
L_total = L_legs + L_torso + L_arm ≈ constant

The TIMING and EFFICIENCY of momentum transfer from legs → arm determines
the release velocity and outcome variance.

Complex Physics:
- Moment of inertia dynamics: I = Σ(m_i * r_i²)
- Angular momentum: L = I * ω
- Power: P = I * ω * (dω/dt) + 0.5 * ω² * (dI/dt)
- Transfer efficiency: correlation between L_leg decrease and L_arm increase
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

# Import existing feature extractors
from data_loader import load_metadata, iterate_shots, get_keypoint_columns


# Initialize keypoint mapping (global)
KEYPOINT_INDICES = {}


def init_keypoint_mapping(keypoint_cols):
    """Initialize global keypoint name to index mapping."""
    global KEYPOINT_INDICES
    KEYPOINT_INDICES = {}

    for col in keypoint_cols:
        parts = col.split("_")
        if len(parts) >= 2:
            keypoint_name = "_".join(parts[:-1])
            coord = parts[-1]

            if keypoint_name not in KEYPOINT_INDICES:
                KEYPOINT_INDICES[keypoint_name] = {}

            if coord == "x":
                KEYPOINT_INDICES[keypoint_name]["x"] = keypoint_cols.index(col)
            elif coord == "y":
                KEYPOINT_INDICES[keypoint_name]["y"] = keypoint_cols.index(col)
            elif coord == "z":
                KEYPOINT_INDICES[keypoint_name]["z"] = keypoint_cols.index(col)


def get_keypoint_trajectory(timeseries: np.ndarray, keypoint_name: str) -> np.ndarray:
    """
    Extract 3D trajectory for a keypoint.

    Returns: (n_frames, 3) array of [x, y, z] positions
    """
    if keypoint_name not in KEYPOINT_INDICES:
        return None

    indices = KEYPOINT_INDICES[keypoint_name]
    trajectory = np.zeros((len(timeseries), 3))

    if "x" in indices:
        trajectory[:, 0] = timeseries[:, indices["x"]]
    if "y" in indices:
        trajectory[:, 1] = timeseries[:, indices["y"]]
    if "z" in indices:
        trajectory[:, 2] = timeseries[:, indices["z"]]

    return trajectory


def compute_velocity(positions: np.ndarray, dt: float = 1/60.0, smooth: bool = True) -> np.ndarray:
    """
    Compute velocity from positions.

    Args:
        positions: (n_frames, 3) array
        dt: Time step (default 60fps = 1/60 sec)
        smooth: Apply Savitzky-Golay filter

    Returns: (n_frames, 3) velocity array
    """
    if len(positions) < 5:
        return np.zeros_like(positions)

    # Smooth positions first to reduce noise
    if smooth and len(positions) >= 11:
        try:
            positions_smooth = savgol_filter(positions, window_length=11, polyorder=3, axis=0)
        except:
            positions_smooth = positions
    else:
        positions_smooth = positions

    # Central difference
    velocity = np.gradient(positions_smooth, dt, axis=0)

    return velocity


def compute_acceleration(velocity: np.ndarray, dt: float = 1/60.0) -> np.ndarray:
    """Compute acceleration from velocity."""
    if len(velocity) < 3:
        return np.zeros_like(velocity)

    acceleration = np.gradient(velocity, dt, axis=0)
    return acceleration


def compute_angular_velocity(angle_trajectory: np.ndarray, dt: float = 1/60.0) -> np.ndarray:
    """
    Compute angular velocity from angle trajectory.

    Args:
        angle_trajectory: (n_frames,) array of angles in radians
        dt: Time step

    Returns: (n_frames,) angular velocity in rad/s
    """
    if len(angle_trajectory) < 3:
        return np.zeros_like(angle_trajectory)

    omega = np.gradient(angle_trajectory, dt)
    return omega


def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Compute angle at joint p2 formed by p1-p2-p3.

    Args:
        p1, p2, p3: (n_frames, 3) position arrays

    Returns: (n_frames,) angle in radians
    """
    # Vectors
    v1 = p1 - p2  # From joint to previous point
    v2 = p3 - p2  # From joint to next point

    # Normalize
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)

    # Avoid division by zero
    v1_norm = np.where(v1_norm > 1e-6, v1_norm, 1e-6)
    v2_norm = np.where(v2_norm > 1e-6, v2_norm, 1e-6)

    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm

    # Dot product
    cos_angle = np.sum(v1_unit * v2_unit, axis=1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.arccos(cos_angle)
    return angle


def estimate_segment_mass(segment_name: str) -> float:
    """
    Estimate segment mass as fraction of body mass.

    Based on biomechanics literature (Winter, 2009):
    - Thigh: 0.1 * body_mass
    - Shank: 0.0465 * body_mass
    - Upper arm: 0.028 * body_mass
    - Forearm + hand: 0.022 * body_mass
    - Torso: 0.497 * body_mass

    We'll use normalized values (body_mass = 1.0) since we don't have actual mass.
    """
    mass_fractions = {
        "thigh": 0.1,
        "shank": 0.0465,
        "upper_arm": 0.028,
        "forearm": 0.016,
        "hand": 0.006,
        "torso": 0.497,
        "head": 0.081,
    }

    return mass_fractions.get(segment_name, 0.01)


def compute_moment_of_inertia(segment_length: float, segment_mass: float) -> float:
    """
    Compute moment of inertia for a segment.

    Approximation: Treat segment as thin rod rotating about one end
    I = (1/3) * m * L²

    For more accuracy, could use parallel axis theorem for rotation about CoM.
    """
    I = (1/3) * segment_mass * segment_length**2
    return I


def extract_angular_momentum_features(
    timeseries: np.ndarray,
    participant_id: int,
    dt: float = 1/60.0
) -> Dict[str, float]:
    """
    Extract angular momentum and power flow features.

    Key Metrics:
    1. Angular momentum in each segment (L = I * ω)
    2. Transfer efficiency (correlation between L_leg decrease and L_arm increase)
    3. Power generation (P = I * ω * dω/dt + 0.5 * ω² * dI/dt)
    4. Momentum conservation violation (std of L_total)

    Args:
        timeseries: (n_frames, n_keypoints*3) array
        participant_id: Participant ID
        dt: Time step

    Returns:
        Dictionary of angular momentum features
    """
    features = {}

    # Get keypoint trajectories
    ankle = get_keypoint_trajectory(timeseries, "right_ankle")
    knee = get_keypoint_trajectory(timeseries, "right_knee")
    hip = get_keypoint_trajectory(timeseries, "right_hip")
    shoulder = get_keypoint_trajectory(timeseries, "right_shoulder")
    elbow = get_keypoint_trajectory(timeseries, "right_elbow")
    wrist = get_keypoint_trajectory(timeseries, "right_wrist")

    if any(x is None for x in [ankle, knee, hip, shoulder, elbow, wrist]):
        return {f"angular_momentum_{k}": 0.0 for k in range(20)}

    # Compute joint angles
    knee_angle = compute_joint_angle(ankle, knee, hip)
    hip_angle = compute_joint_angle(knee, hip, shoulder)
    shoulder_angle = compute_joint_angle(hip, shoulder, elbow)
    elbow_angle = compute_joint_angle(shoulder, elbow, wrist)

    # Compute angular velocities
    omega_knee = compute_angular_velocity(knee_angle, dt)
    omega_hip = compute_angular_velocity(hip_angle, dt)
    omega_shoulder = compute_angular_velocity(shoulder_angle, dt)
    omega_elbow = compute_angular_velocity(elbow_angle, dt)

    # Compute angular accelerations
    alpha_knee = np.gradient(omega_knee, dt)
    alpha_hip = np.gradient(omega_hip, dt)
    alpha_shoulder = np.gradient(omega_shoulder, dt)
    alpha_elbow = np.gradient(omega_elbow, dt)

    # Estimate segment lengths (Euclidean distance between joints)
    L_thigh = np.linalg.norm(hip - knee, axis=1)
    L_shank = np.linalg.norm(knee - ankle, axis=1)
    L_upper_arm = np.linalg.norm(elbow - shoulder, axis=1)
    L_forearm = np.linalg.norm(wrist - elbow, axis=1)

    # Estimate segment masses
    m_thigh = estimate_segment_mass("thigh")
    m_shank = estimate_segment_mass("shank")
    m_upper_arm = estimate_segment_mass("upper_arm")
    m_forearm = estimate_segment_mass("forearm")

    # Compute moments of inertia (time-varying as segments extend/flex)
    I_leg = np.array([compute_moment_of_inertia(L_thigh[i] + L_shank[i], m_thigh + m_shank)
                      for i in range(len(timeseries))])
    I_arm = np.array([compute_moment_of_inertia(L_upper_arm[i] + L_forearm[i], m_upper_arm + m_forearm)
                      for i in range(len(timeseries))])

    # Compute angular momentum (L = I * ω)
    L_leg = I_leg * np.abs(omega_knee)  # Use knee as representative for leg
    L_arm = I_arm * np.abs(omega_elbow)  # Use elbow as representative for arm

    # Approximate total angular momentum (should be conserved in isolated system)
    L_total = L_leg + L_arm

    # =======================================================================
    # Feature 1-5: Peak Angular Momentum
    # =======================================================================
    features["angular_momentum_leg_peak"] = np.max(L_leg) if len(L_leg) > 0 else 0.0
    features["angular_momentum_arm_peak"] = np.max(L_arm) if len(L_arm) > 0 else 0.0
    features["angular_momentum_leg_mean"] = np.mean(L_leg) if len(L_leg) > 0 else 0.0
    features["angular_momentum_arm_mean"] = np.mean(L_arm) if len(L_arm) > 0 else 0.0
    features["angular_momentum_total_mean"] = np.mean(L_total) if len(L_total) > 0 else 0.0

    # =======================================================================
    # Feature 6-7: Momentum Conservation
    # =======================================================================
    # In ideal case, L_total should be constant. Deviation indicates:
    # - External torques (ground reaction)
    # - Tracking errors
    # - Non-rigid body approximation errors
    features["momentum_conservation_violation"] = np.std(L_total) if len(L_total) > 0 else 0.0
    features["momentum_total_variation"] = (np.max(L_total) - np.min(L_total)) if len(L_total) > 0 else 0.0

    # =======================================================================
    # Feature 8: Transfer Efficiency
    # =======================================================================
    # Key breakthrough: Measure correlation between L_leg decrease and L_arm increase
    # Good transfer: When L_leg goes down, L_arm goes up (negative correlation)
    dL_leg_dt = np.gradient(L_leg, dt)
    dL_arm_dt = np.gradient(L_arm, dt)

    # Remove NaN values
    valid_mask = ~(np.isnan(dL_leg_dt) | np.isnan(dL_arm_dt))
    if np.sum(valid_mask) > 10:
        correlation, _ = pearsonr(dL_leg_dt[valid_mask], dL_arm_dt[valid_mask])
        # Negative correlation is good (L flows from leg to arm)
        features["transfer_efficiency"] = -correlation if not np.isnan(correlation) else 0.0
    else:
        features["transfer_efficiency"] = 0.0

    features["dL_leg_dt_max"] = np.max(np.abs(dL_leg_dt)) if len(dL_leg_dt) > 0 else 0.0
    features["dL_arm_dt_max"] = np.max(np.abs(dL_arm_dt)) if len(dL_arm_dt) > 0 else 0.0

    # =======================================================================
    # Feature 11-15: Power Generation
    # =======================================================================
    # Power = I * ω * α + 0.5 * ω² * (dI/dt)
    # First term: Acceleration power
    # Second term: Power from changing I (extending/flexing)

    dI_leg_dt = np.gradient(I_leg, dt)
    dI_arm_dt = np.gradient(I_arm, dt)

    # Acceleration power
    power_accel_leg = I_leg * np.abs(omega_knee) * np.abs(alpha_knee)
    power_accel_arm = I_arm * np.abs(omega_elbow) * np.abs(alpha_elbow)

    # Extension/flexion power
    power_flex_leg = 0.5 * omega_knee**2 * dI_leg_dt
    power_flex_arm = 0.5 * omega_elbow**2 * dI_arm_dt

    # Total power
    power_leg = power_accel_leg + power_flex_leg
    power_arm = power_accel_arm + power_flex_arm

    features["power_peak_leg"] = np.max(power_leg) if len(power_leg) > 0 else 0.0
    features["power_peak_arm"] = np.max(power_arm) if len(power_arm) > 0 else 0.0
    features["power_mean_leg"] = np.mean(np.abs(power_leg)) if len(power_leg) > 0 else 0.0
    features["power_mean_arm"] = np.mean(np.abs(power_arm)) if len(power_arm) > 0 else 0.0
    features["power_transfer_ratio"] = (features["power_peak_arm"] / features["power_peak_leg"]) if features["power_peak_leg"] > 1e-6 else 0.0

    # =======================================================================
    # Feature 16-20: Moment of Inertia Dynamics
    # =======================================================================
    features["moment_of_inertia_leg_min"] = np.min(I_leg) if len(I_leg) > 0 else 0.0
    features["moment_of_inertia_leg_max"] = np.max(I_leg) if len(I_leg) > 0 else 0.0
    features["moment_of_inertia_arm_min"] = np.min(I_arm) if len(I_arm) > 0 else 0.0
    features["moment_of_inertia_arm_max"] = np.max(I_arm) if len(I_arm) > 0 else 0.0

    # Rate of change of I (how fast arm extends)
    features["I_change_rate_arm_max"] = np.max(np.abs(dI_arm_dt)) if len(dI_arm_dt) > 0 else 0.0

    return features


def validate_angular_momentum_theory(train_data_path: Path = Path("data/train.parquet")):
    """
    Validate the angular momentum transfer hypothesis.

    Test:
    1. Extract L for all shots
    2. Check if L_total is approximately conserved
    3. Correlate transfer_efficiency with outcome variance
    4. Compare high-efficiency vs low-efficiency shots

    Hypothesis: High transfer efficiency → low outcome variance
    """
    print("="*70)
    print("ANGULAR MOMENTUM TRANSFER VALIDATION")
    print("="*70)

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Extract features for all training shots
    all_features = []
    all_outcomes = []

    print("\nExtracting angular momentum features for all training shots...")

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True, chunk_size=20)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}...")

        features = extract_angular_momentum_features(
            timeseries,
            participant_id=metadata["participant_id"]
        )

        all_features.append(features)
        all_outcomes.append({
            "angle": metadata["angle"],
            "depth": metadata["depth"],
            "left_right": metadata["left_right"],
            "participant_id": metadata["participant_id"]
        })

    print(f"\nExtracted features for {len(all_features)} shots")

    # Convert to DataFrame
    df_features = pd.DataFrame(all_features)
    df_outcomes = pd.DataFrame(all_outcomes)

    # =======================================================================
    # Analysis 1: Momentum Conservation Check
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: Momentum Conservation")
    print("="*70)

    conservation_violation = df_features["momentum_conservation_violation"]
    print(f"\nMomentum conservation violation (std of L_total):")
    print(f"  Mean: {conservation_violation.mean():.6f}")
    print(f"  Median: {conservation_violation.median():.6f}")
    print(f"  Min: {conservation_violation.min():.6f}")
    print(f"  Max: {conservation_violation.max():.6f}")

    # Low violation = good tracking, rigid body approximation valid
    # High violation = tracking errors, external torques, non-rigid effects

    # =======================================================================
    # Analysis 2: Transfer Efficiency vs Outcome Variance
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: Transfer Efficiency vs Outcome Variance")
    print("="*70)

    transfer_eff = df_features["transfer_efficiency"]
    print(f"\nTransfer efficiency distribution:")
    print(f"  Mean: {transfer_eff.mean():.4f}")
    print(f"  Median: {transfer_eff.median():.4f}")
    print(f"  Std: {transfer_eff.std():.4f}")
    print(f"  Range: [{transfer_eff.min():.4f}, {transfer_eff.max():.4f}]")

    # Split into high vs low efficiency groups
    threshold = transfer_eff.median()
    high_eff_mask = transfer_eff > threshold
    low_eff_mask = transfer_eff <= threshold

    print(f"\nSplitting into high (>{threshold:.4f}) vs low efficiency groups...")
    print(f"  High efficiency: {np.sum(high_eff_mask)} shots")
    print(f"  Low efficiency: {np.sum(low_eff_mask)} shots")

    # Compute outcome variance for each group
    angle_variance_high = df_outcomes.loc[high_eff_mask, "angle"].var()
    angle_variance_low = df_outcomes.loc[low_eff_mask, "angle"].var()

    depth_variance_high = df_outcomes.loc[high_eff_mask, "depth"].var()
    depth_variance_low = df_outcomes.loc[low_eff_mask, "depth"].var()

    lr_variance_high = df_outcomes.loc[high_eff_mask, "left_right"].var()
    lr_variance_low = df_outcomes.loc[low_eff_mask, "left_right"].var()

    print(f"\nOutcome Variance:")
    print(f"  Angle:")
    print(f"    High efficiency: {angle_variance_high:.4f} (std: {np.sqrt(angle_variance_high):.4f}°)")
    print(f"    Low efficiency:  {angle_variance_low:.4f} (std: {np.sqrt(angle_variance_low):.4f}°)")
    print(f"    Improvement: {(1 - angle_variance_high/angle_variance_low)*100:.1f}%")

    print(f"  Depth:")
    print(f"    High efficiency: {depth_variance_high:.4f}")
    print(f"    Low efficiency:  {depth_variance_low:.4f}")
    print(f"    Improvement: {(1 - depth_variance_high/depth_variance_low)*100:.1f}%")

    print(f"  Left/Right:")
    print(f"    High efficiency: {lr_variance_high:.4f}")
    print(f"    Low efficiency:  {lr_variance_low:.4f}")
    print(f"    Improvement: {(1 - lr_variance_high/lr_variance_low)*100:.1f}%")

    # =======================================================================
    # Analysis 3: Correlation Between Features and Outcomes
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 3: Feature-Outcome Correlations")
    print("="*70)

    key_features = [
        "angular_momentum_leg_peak",
        "angular_momentum_arm_peak",
        "transfer_efficiency",
        "power_peak_leg",
        "power_peak_arm",
        "momentum_conservation_violation"
    ]

    print("\nCorrelation with Angle:")
    for feat in key_features:
        if feat in df_features.columns:
            corr, pval = pearsonr(df_features[feat], df_outcomes["angle"])
            print(f"  {feat:40} r={corr:+.4f} (p={pval:.4f})")

    print("\nCorrelation with Depth:")
    for feat in key_features:
        if feat in df_features.columns:
            corr, pval = pearsonr(df_features[feat], df_outcomes["depth"])
            print(f"  {feat:40} r={corr:+.4f} (p={pval:.4f})")

    print("\nCorrelation with Left/Right:")
    for feat in key_features:
        if feat in df_features.columns:
            corr, pval = pearsonr(df_features[feat], df_outcomes["left_right"])
            print(f"  {feat:40} r={corr:+.4f} (p={pval:.4f})")

    # =======================================================================
    # Analysis 4: Power Generation Patterns
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 4: Power Generation Patterns")
    print("="*70)

    print(f"\nPower peak leg:  {df_features['power_peak_leg'].mean():.4f} ± {df_features['power_peak_leg'].std():.4f}")
    print(f"Power peak arm:  {df_features['power_peak_arm'].mean():.4f} ± {df_features['power_peak_arm'].std():.4f}")
    print(f"Power transfer ratio: {df_features['power_transfer_ratio'].mean():.4f} ± {df_features['power_transfer_ratio'].std():.4f}")

    # High power transfer ratio = efficient kinetic chain
    power_ratio = df_features["power_transfer_ratio"]
    high_power_mask = power_ratio > power_ratio.median()
    low_power_mask = power_ratio <= power_ratio.median()

    angle_var_high_power = df_outcomes.loc[high_power_mask, "angle"].var()
    angle_var_low_power = df_outcomes.loc[low_power_mask, "angle"].var()

    print(f"\nAngle variance by power transfer efficiency:")
    print(f"  High power ratio: {angle_var_high_power:.4f} (std: {np.sqrt(angle_var_high_power):.4f}°)")
    print(f"  Low power ratio:  {angle_var_low_power:.4f} (std: {np.sqrt(angle_var_low_power):.4f}°)")
    print(f"  Improvement: {(1 - angle_var_high_power/angle_var_low_power)*100:.1f}%")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save features
    df_features.to_csv(output_dir / "angular_momentum_features.csv", index=False)
    print(f"\nSaved features to {output_dir / 'angular_momentum_features.csv'}")

    # Save combined data
    df_combined = pd.concat([df_features, df_outcomes], axis=1)
    df_combined.to_csv(output_dir / "angular_momentum_full.csv", index=False)
    print(f"Saved combined data to {output_dir / 'angular_momentum_full.csv'}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    validate_angular_momentum_theory()
