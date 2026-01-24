#!/usr/bin/env python3
"""
Power Flow Timing Signatures - Breakthrough Discovery Test 3

Key Insight: The TIMING of peak power in each kinetic chain segment predicts
shooting efficiency and outcome variance.

Physics Principle: Optimal kinetic chain sequencing
  Legs → Hips → Torso → Shoulders → Elbow → Wrist

Optimal phase lags (from biomechanics literature):
  - Knee → Hip: 50-80ms (3-5 frames at 60fps)
  - Hip → Shoulder: 30-50ms (2-3 frames)
  - Shoulder → Elbow: 30-50ms (2-3 frames)
  - Elbow → Wrist: 20-30ms (1-2 frames)

Hypothesis: Deviations from optimal timing → increased outcome variance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

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
    """Extract 3D trajectory for a keypoint."""
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
    """Compute velocity from positions with optional smoothing."""
    if len(positions) < 5:
        return np.zeros_like(positions)

    # Smooth positions first
    if smooth and len(positions) >= 11:
        try:
            positions_smooth = savgol_filter(positions, window_length=11, polyorder=3, axis=0)
        except:
            positions_smooth = positions
    else:
        positions_smooth = positions

    velocity = np.gradient(positions_smooth, dt, axis=0)
    return velocity


def compute_power(velocity: np.ndarray, mass: float = 1.0, dt: float = 1/60.0) -> np.ndarray:
    """
    Compute instantaneous power as P = F · v ≈ m * a · v

    Args:
        velocity: (n_frames, 3) velocity array
        mass: Segment mass (normalized)
        dt: Time step

    Returns:
        (n_frames,) power array
    """
    if len(velocity) < 3:
        return np.zeros(len(velocity))

    # Acceleration
    acceleration = np.gradient(velocity, dt, axis=0)

    # Force ≈ m * a
    force = mass * acceleration

    # Power = F · v (dot product)
    power = np.sum(force * velocity, axis=1)

    return np.abs(power)  # Use magnitude


def find_peak_power_timing(power: np.ndarray, min_power: float = 1.0) -> int:
    """
    Find frame of peak power.

    Args:
        power: (n_frames,) power array
        min_power: Minimum power threshold

    Returns:
        Frame index of peak power (or -1 if not found)
    """
    if len(power) < 5:
        return -1

    # Find peaks
    peaks, _ = find_peaks(power, height=min_power, distance=5)

    if len(peaks) == 0:
        # No peaks found, use global maximum
        return int(np.argmax(power))

    # Return highest peak
    peak_powers = power[peaks]
    highest_peak_idx = peaks[np.argmax(peak_powers)]

    return int(highest_peak_idx)


def extract_power_flow_timing_features(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract power flow timing features through the kinetic chain.

    Measures:
    1. Timing of peak power in each segment
    2. Phase lags between segments
    3. Deviations from optimal timing
    4. Power transfer efficiency
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
        return {f"power_timing_{i}": 0.0 for i in range(20)}

    # Compute velocities
    vel_knee = compute_velocity(knee)
    vel_hip = compute_velocity(hip)
    vel_shoulder = compute_velocity(shoulder)
    vel_elbow = compute_velocity(elbow)
    vel_wrist = compute_velocity(wrist)

    # Compute power in each segment
    # Using normalized masses from biomechanics literature
    power_knee = compute_power(vel_knee, mass=0.10)  # Leg ~10% body mass
    power_hip = compute_power(vel_hip, mass=0.50)    # Torso ~50% body mass
    power_shoulder = compute_power(vel_shoulder, mass=0.05)  # Shoulder ~5%
    power_elbow = compute_power(vel_elbow, mass=0.03)  # Upper arm ~3%
    power_wrist = compute_power(vel_wrist, mass=0.02)  # Forearm+hand ~2%

    # Find timing of peak power in each segment
    t_peak_knee = find_peak_power_timing(power_knee)
    t_peak_hip = find_peak_power_timing(power_hip)
    t_peak_shoulder = find_peak_power_timing(power_shoulder)
    t_peak_elbow = find_peak_power_timing(power_elbow)
    t_peak_wrist = find_peak_power_timing(power_wrist)

    # Peak timings as features
    features["timing_peak_knee"] = float(t_peak_knee)
    features["timing_peak_hip"] = float(t_peak_hip)
    features["timing_peak_shoulder"] = float(t_peak_shoulder)
    features["timing_peak_elbow"] = float(t_peak_elbow)
    features["timing_peak_wrist"] = float(t_peak_wrist)

    # Phase lags (time differences)
    lag_knee_hip = t_peak_hip - t_peak_knee
    lag_hip_shoulder = t_peak_shoulder - t_peak_hip
    lag_shoulder_elbow = t_peak_elbow - t_peak_shoulder
    lag_elbow_wrist = t_peak_wrist - t_peak_elbow

    features["lag_knee_hip"] = float(lag_knee_hip)
    features["lag_hip_shoulder"] = float(lag_hip_shoulder)
    features["lag_shoulder_elbow"] = float(lag_shoulder_elbow)
    features["lag_elbow_wrist"] = float(lag_elbow_wrist)

    # Optimal phase lags (from biomechanics literature)
    # At 60fps: 1 frame = 16.67ms
    optimal_lags = {
        "knee_hip": 4.0,        # 60-70ms → 4 frames
        "hip_shoulder": 2.5,    # 40ms → 2.5 frames
        "shoulder_elbow": 2.5,  # 40ms → 2.5 frames
        "elbow_wrist": 1.5      # 25ms → 1.5 frames
    }

    # Timing errors (deviation from optimal)
    timing_error_knee_hip = abs(lag_knee_hip - optimal_lags["knee_hip"])
    timing_error_hip_shoulder = abs(lag_hip_shoulder - optimal_lags["hip_shoulder"])
    timing_error_shoulder_elbow = abs(lag_shoulder_elbow - optimal_lags["shoulder_elbow"])
    timing_error_elbow_wrist = abs(lag_elbow_wrist - optimal_lags["elbow_wrist"])

    features["timing_error_knee_hip"] = timing_error_knee_hip
    features["timing_error_hip_shoulder"] = timing_error_hip_shoulder
    features["timing_error_shoulder_elbow"] = timing_error_shoulder_elbow
    features["timing_error_elbow_wrist"] = timing_error_elbow_wrist

    # Total timing error
    features["timing_error_total"] = (timing_error_knee_hip + timing_error_hip_shoulder +
                                      timing_error_shoulder_elbow + timing_error_elbow_wrist)

    # Power magnitudes
    features["power_peak_knee_magnitude"] = float(np.max(power_knee)) if len(power_knee) > 0 else 0.0
    features["power_peak_hip_magnitude"] = float(np.max(power_hip)) if len(power_hip) > 0 else 0.0
    features["power_peak_shoulder_magnitude"] = float(np.max(power_shoulder)) if len(power_shoulder) > 0 else 0.0
    features["power_peak_elbow_magnitude"] = float(np.max(power_elbow)) if len(power_elbow) > 0 else 0.0
    features["power_peak_wrist_magnitude"] = float(np.max(power_wrist)) if len(power_wrist) > 0 else 0.0

    # Power transfer efficiency (output / input)
    if features["power_peak_knee_magnitude"] > 1e-6:
        features["power_transfer_efficiency"] = features["power_peak_wrist_magnitude"] / features["power_peak_knee_magnitude"]
    else:
        features["power_transfer_efficiency"] = 0.0

    return features


def validate_power_flow_timing():
    """
    Validate power flow timing hypothesis.

    Test:
    1. Extract timing features for all shots
    2. Correlate timing errors with outcome variance
    3. Compare optimal vs non-optimal timing groups

    Hypothesis: Lower timing errors → lower outcome variance
    """
    print("="*70)
    print("POWER FLOW TIMING VALIDATION")
    print("="*70)

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Extract features
    print("\nExtracting power flow timing features for all training shots...")

    all_features = []
    all_outcomes = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True, chunk_size=20)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}...")

        features = extract_power_flow_timing_features(timeseries)

        all_features.append(features)
        all_outcomes.append({
            "angle": metadata["angle"],
            "depth": metadata["depth"],
            "left_right": metadata["left_right"]
        })

    print(f"\nExtracted features for {len(all_features)} shots")

    # Convert to DataFrames
    df_features = pd.DataFrame(all_features)
    df_outcomes = pd.DataFrame(all_outcomes)

    # =======================================================================
    # Analysis 1: Timing Error Distribution
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: Timing Error Distribution")
    print("="*70)

    timing_error_total = df_features["timing_error_total"]
    print(f"\nTotal timing error distribution:")
    print(f"  Mean: {timing_error_total.mean():.2f} frames")
    print(f"  Median: {timing_error_total.median():.2f} frames")
    print(f"  Std: {timing_error_total.std():.2f} frames")
    print(f"  Range: [{timing_error_total.min():.2f}, {timing_error_total.max():.2f}] frames")

    # =======================================================================
    # Analysis 2: Timing Error vs Outcome Variance
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: Timing Error vs Outcome Variance")
    print("="*70)

    # Split into low vs high timing error groups
    threshold = timing_error_total.median()
    low_error_mask = timing_error_total <= threshold
    high_error_mask = timing_error_total > threshold

    print(f"\nSplitting into low (≤{threshold:.2f}) vs high timing error groups...")
    print(f"  Low timing error: {np.sum(low_error_mask)} shots")
    print(f"  High timing error: {np.sum(high_error_mask)} shots")

    # Compute outcome variance for each group
    angle_var_low = df_outcomes.loc[low_error_mask, "angle"].var()
    angle_var_high = df_outcomes.loc[high_error_mask, "angle"].var()

    depth_var_low = df_outcomes.loc[low_error_mask, "depth"].var()
    depth_var_high = df_outcomes.loc[high_error_mask, "depth"].var()

    lr_var_low = df_outcomes.loc[low_error_mask, "left_right"].var()
    lr_var_high = df_outcomes.loc[high_error_mask, "left_right"].var()

    print(f"\nOutcome Variance:")
    print(f"  Angle:")
    print(f"    Low timing error:  {angle_var_low:.4f} (std: {np.sqrt(angle_var_low):.4f}°)")
    print(f"    High timing error: {angle_var_high:.4f} (std: {np.sqrt(angle_var_high):.4f}°)")
    improvement_angle = (1 - angle_var_low/angle_var_high)*100 if angle_var_high > 0 else 0
    print(f"    Improvement: {improvement_angle:.1f}%")

    print(f"  Depth:")
    print(f"    Low timing error:  {depth_var_low:.4f}")
    print(f"    High timing error: {depth_var_high:.4f}")
    improvement_depth = (1 - depth_var_low/depth_var_high)*100 if depth_var_high > 0 else 0
    print(f"    Improvement: {improvement_depth:.1f}%")

    print(f"  Left/Right:")
    print(f"    Low timing error:  {lr_var_low:.4f}")
    print(f"    High timing error: {lr_var_high:.4f}")
    improvement_lr = (1 - lr_var_low/lr_var_high)*100 if lr_var_high > 0 else 0
    print(f"    Improvement: {improvement_lr:.1f}%")

    # =======================================================================
    # Analysis 3: Correlation Between Timing Errors and Outcomes
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 3: Timing Error - Outcome Correlations")
    print("="*70)

    timing_features = [
        "timing_error_total",
        "timing_error_knee_hip",
        "timing_error_hip_shoulder",
        "timing_error_shoulder_elbow",
        "timing_error_elbow_wrist",
        "power_transfer_efficiency"
    ]

    print("\nCorrelation with Angle:")
    for feat in timing_features:
        if feat in df_features.columns:
            valid_mask = ~np.isnan(df_features[feat])
            if np.sum(valid_mask) > 10:
                corr, pval = pearsonr(df_features.loc[valid_mask, feat],
                                     df_outcomes.loc[valid_mask, "angle"])
                print(f"  {feat:40} r={corr:+.4f} (p={pval:.4f})")

    print("\nCorrelation with Depth:")
    for feat in timing_features:
        if feat in df_features.columns:
            valid_mask = ~np.isnan(df_features[feat])
            if np.sum(valid_mask) > 10:
                corr, pval = pearsonr(df_features.loc[valid_mask, feat],
                                     df_outcomes.loc[valid_mask, "depth"])
                print(f"  {feat:40} r={corr:+.4f} (p={pval:.4f})")

    # =======================================================================
    # Analysis 4: Predictive Power of Timing Features
    # =======================================================================
    print("\n" + "="*70)
    print("ANALYSIS 4: Predictive Power (XGBoost Model)")
    print("="*70)

    # Prepare data
    X = df_features.values
    X = np.nan_to_num(X, nan=0.0)
    y = df_outcomes.values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models for each target
    results = {}
    for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
        model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train[:, target_idx])
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test[:, target_idx], pred)

        # Feature importance
        importance = model.feature_importances_
        top_features_idx = np.argsort(importance)[-5:][::-1]
        top_features = [(df_features.columns[idx], importance[idx])
                       for idx in top_features_idx]

        results[target_name] = {
            "mse": mse,
            "top_features": top_features
        }

        print(f"\n{target_name.upper()}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  Top 5 features:")
        for feat_name, importance_val in top_features:
            print(f"    {feat_name:40} {importance_val:.4f}")

    # Overall MSE
    overall_mse = (results["angle"]["mse"] + results["depth"]["mse"] + results["left_right"]["mse"]) / 3
    print(f"\n  Overall MSE: {overall_mse:.4f}")

    # Save features
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    df_features.to_csv(output_dir / "power_flow_timing_features.csv", index=False)
    print(f"\nSaved features to {output_dir / 'power_flow_timing_features.csv'}")

    # Save combined data
    df_combined = pd.concat([df_features, df_outcomes], axis=1)
    df_combined.to_csv(output_dir / "power_flow_timing_full.csv", index=False)
    print(f"Saved combined data to {output_dir / 'power_flow_timing_full.csv'}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    validate_power_flow_timing()
