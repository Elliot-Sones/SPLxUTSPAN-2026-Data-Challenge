#!/usr/bin/env python3
"""
Phase-Specific Feature Windows - Breakthrough Discovery Test 2

Key Insight: Different targets are determined at different temporal phases:
- DEPTH: Peaks at frame 102 (loading phase) - determined by leg power
- ANGLE: Peaks at frame 153 (propulsion phase) - determined by arm trajectory
- LEFT/RIGHT: Peaks at frame 237 (follow-through) - determined by wrist deviation

This suggests extracting features from TARGET-SPECIFIC temporal windows rather
than using a single release frame for all targets.

Physics Principle: Causal chain in shooting mechanics
  Depth ← Energy generation (frames 50-150)
  Angle ← Trajectory elevation (frames 100-175)
  Left/Right ← Release mechanics & spin (frames 175-240)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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


def compute_velocity(positions: np.ndarray, dt: float = 1/60.0) -> np.ndarray:
    """Compute velocity from positions using gradient."""
    if len(positions) < 3:
        return np.zeros_like(positions)
    return np.gradient(positions, dt, axis=0)


def extract_depth_features(timeseries: np.ndarray, window_start: int = 50, window_end: int = 150) -> Dict[str, float]:
    """
    Extract features for DEPTH prediction from loading phase (frames 50-150).

    Depth is determined by energy generation in the legs during loading.

    Key features:
    - Knee bend depth (maximum squat)
    - Hip drop (center of mass lowering)
    - Leg power generation
    - Ankle-knee-hip alignment
    """
    features = {}

    # Get keypoints
    ankle = get_keypoint_trajectory(timeseries, "right_ankle")
    knee = get_keypoint_trajectory(timeseries, "right_knee")
    hip = get_keypoint_trajectory(timeseries, "right_hip")

    if any(x is None for x in [ankle, knee, hip]):
        return {f"depth_{i}": 0.0 for i in range(10)}

    # Ensure window is valid
    window_start = max(0, min(window_start, len(timeseries) - 1))
    window_end = max(window_start + 1, min(window_end, len(timeseries)))

    # Extract window
    ankle_w = ankle[window_start:window_end]
    knee_w = knee[window_start:window_end]
    hip_w = hip[window_start:window_end]

    # 1. Knee bend depth (minimum knee height relative to ankle)
    knee_height_rel = knee_w[:, 2] - ankle_w[:, 2]
    features["depth_knee_min_height"] = np.min(knee_height_rel) if len(knee_height_rel) > 0 else 0.0
    features["depth_knee_mean_height"] = np.mean(knee_height_rel) if len(knee_height_rel) > 0 else 0.0

    # 2. Hip drop (maximum CoM drop)
    hip_drop = hip_w[0, 2] - np.min(hip_w[:, 2])
    features["depth_hip_drop_max"] = hip_drop

    # 3. Hip vertical velocity (power generation proxy)
    hip_vel = compute_velocity(hip_w)
    hip_vel_z = hip_vel[:, 2]
    features["depth_hip_vel_z_min"] = np.min(hip_vel_z) if len(hip_vel_z) > 0 else 0.0
    features["depth_hip_vel_z_max"] = np.max(hip_vel_z) if len(hip_vel_z) > 0 else 0.0

    # 4. Knee extension rate (how fast legs extend)
    knee_vel = compute_velocity(knee_w)
    knee_vel_z = knee_vel[:, 2]
    features["depth_knee_extension_rate"] = np.max(knee_vel_z) if len(knee_vel_z) > 0 else 0.0

    # 5. Ankle stability (should be minimal x,y movement)
    ankle_xy_movement = np.std(ankle_w[:, :2], axis=0)
    features["depth_ankle_stability_x"] = ankle_xy_movement[0] if len(ankle_xy_movement) > 0 else 0.0
    features["depth_ankle_stability_y"] = ankle_xy_movement[1] if len(ankle_xy_movement) > 0 else 0.0

    # 6. Loading duration
    features["depth_loading_duration"] = window_end - window_start

    return features


def extract_angle_features(timeseries: np.ndarray, window_start: int = 100, window_end: int = 175) -> Dict[str, float]:
    """
    Extract features for ANGLE prediction from propulsion phase (frames 100-175).

    Angle is determined by arm trajectory elevation during propulsion.

    Key features:
    - Shoulder elevation
    - Torso lean
    - Arm extension rate
    - Wrist trajectory elevation
    """
    features = {}

    # Get keypoints
    hip = get_keypoint_trajectory(timeseries, "right_hip")
    shoulder = get_keypoint_trajectory(timeseries, "right_shoulder")
    elbow = get_keypoint_trajectory(timeseries, "right_elbow")
    wrist = get_keypoint_trajectory(timeseries, "right_wrist")

    if any(x is None for x in [hip, shoulder, elbow, wrist]):
        return {f"angle_{i}": 0.0 for i in range(10)}

    # Ensure window is valid
    window_start = max(0, min(window_start, len(timeseries) - 1))
    window_end = max(window_start + 1, min(window_end, len(timeseries)))

    # Extract window
    hip_w = hip[window_start:window_end]
    shoulder_w = shoulder[window_start:window_end]
    elbow_w = elbow[window_start:window_end]
    wrist_w = wrist[window_start:window_end]

    # At frame 153 specifically (peak frame for angle)
    frame_153_idx = min(153 - window_start, len(wrist_w) - 1)
    if frame_153_idx >= 0 and frame_153_idx < len(wrist_w):
        # 1. Shoulder elevation at peak frame
        shoulder_height = shoulder_w[frame_153_idx, 2] - hip_w[frame_153_idx, 2]
        features["angle_shoulder_elevation_f153"] = shoulder_height

        # 2. Torso lean (forward tilt)
        torso_vector = shoulder_w[frame_153_idx] - hip_w[frame_153_idx]
        torso_lean_x = torso_vector[0]  # Forward lean
        features["angle_torso_lean_f153"] = torso_lean_x

        # 3. Arm elevation (wrist height relative to shoulder)
        arm_elevation = wrist_w[frame_153_idx, 2] - shoulder_w[frame_153_idx, 2]
        features["angle_arm_elevation_f153"] = arm_elevation
    else:
        features["angle_shoulder_elevation_f153"] = 0.0
        features["angle_torso_lean_f153"] = 0.0
        features["angle_arm_elevation_f153"] = 0.0

    # 4. Mean shoulder elevation over window
    shoulder_heights = shoulder_w[:, 2] - hip_w[:, 2]
    features["angle_shoulder_elevation_mean"] = np.mean(shoulder_heights) if len(shoulder_heights) > 0 else 0.0

    # 5. Arm extension rate (elbow straightening)
    elbow_shoulder_dist = np.linalg.norm(elbow_w - shoulder_w, axis=1)
    wrist_elbow_dist = np.linalg.norm(wrist_w - elbow_w, axis=1)
    arm_length = elbow_shoulder_dist + wrist_elbow_dist
    arm_extension_rate = np.gradient(arm_length, 1/60.0)
    features["angle_arm_extension_rate_max"] = np.max(arm_extension_rate) if len(arm_extension_rate) > 0 else 0.0
    features["angle_arm_extension_rate_mean"] = np.mean(arm_extension_rate) if len(arm_extension_rate) > 0 else 0.0

    # 6. Wrist trajectory elevation angle
    wrist_vel = compute_velocity(wrist_w)
    # Elevation angle = arctan(vz / sqrt(vx² + vy²))
    wrist_vel_xy = np.linalg.norm(wrist_vel[:, :2], axis=1)
    wrist_vel_z = wrist_vel[:, 2]
    elevation_angles = np.arctan2(wrist_vel_z, wrist_vel_xy + 1e-6)  # Avoid division by zero
    features["angle_wrist_trajectory_elevation_mean"] = np.mean(elevation_angles) if len(elevation_angles) > 0 else 0.0
    features["angle_wrist_trajectory_elevation_max"] = np.max(elevation_angles) if len(elevation_angles) > 0 else 0.0

    # 7. Propulsion duration
    features["angle_propulsion_duration"] = window_end - window_start

    return features


def extract_left_right_features(timeseries: np.ndarray, window_start: int = 175, window_end: int = 240) -> Dict[str, float]:
    """
    Extract features for LEFT/RIGHT prediction from release + follow-through (frames 175-240).

    Left/Right is determined by lateral wrist deviation and spin during release.

    Key features:
    - Wrist lateral deviation
    - Shoulder-wrist alignment
    - Follow-through direction
    - Asymmetry in arm motion
    """
    features = {}

    # Get keypoints
    shoulder = get_keypoint_trajectory(timeseries, "right_shoulder")
    elbow = get_keypoint_trajectory(timeseries, "right_elbow")
    wrist = get_keypoint_trajectory(timeseries, "right_wrist")

    if any(x is None for x in [shoulder, elbow, wrist]):
        return {f"lr_{i}": 0.0 for i in range(10)}

    # Ensure window is valid
    window_start = max(0, min(window_start, len(timeseries) - 1))
    window_end = max(window_start + 1, min(window_end, len(timeseries)))

    # Extract window
    shoulder_w = shoulder[window_start:window_end]
    elbow_w = elbow[window_start:window_end]
    wrist_w = wrist[window_start:window_end]

    # At frame 237 specifically (peak frame for left/right)
    frame_237_idx = min(237 - window_start, len(wrist_w) - 1)
    if frame_237_idx >= 0 and frame_237_idx < len(wrist_w):
        # 1. Wrist lateral deviation at peak frame
        wrist_x_dev = wrist_w[frame_237_idx, 0] - shoulder_w[frame_237_idx, 0]
        features["lr_wrist_lateral_dev_f237"] = wrist_x_dev

        # 2. Wrist forward position
        wrist_y_dev = wrist_w[frame_237_idx, 1] - shoulder_w[frame_237_idx, 1]
        features["lr_wrist_forward_dev_f237"] = wrist_y_dev
    else:
        features["lr_wrist_lateral_dev_f237"] = 0.0
        features["lr_wrist_forward_dev_f237"] = 0.0

    # 3. Wrist lateral deviation std (consistency)
    wrist_x_relative = wrist_w[:, 0] - shoulder_w[:, 0]
    features["lr_wrist_lateral_std"] = np.std(wrist_x_relative) if len(wrist_x_relative) > 0 else 0.0
    features["lr_wrist_lateral_mean"] = np.mean(wrist_x_relative) if len(wrist_x_relative) > 0 else 0.0

    # 4. Elbow-wrist alignment (should be straight for consistent release)
    elbow_wrist_vector = wrist_w - elbow_w
    elbow_wrist_lateral = elbow_wrist_vector[:, 0]  # X component
    features["lr_elbow_wrist_alignment_std"] = np.std(elbow_wrist_lateral) if len(elbow_wrist_lateral) > 0 else 0.0
    features["lr_elbow_wrist_alignment_mean"] = np.mean(elbow_wrist_lateral) if len(elbow_wrist_lateral) > 0 else 0.0

    # 5. Follow-through trajectory (wrist path after release)
    if len(wrist_w) > 10:
        followthrough_path = wrist_w[-10:]  # Last 10 frames
        followthrough_x_range = np.max(followthrough_path[:, 0]) - np.min(followthrough_path[:, 0])
        features["lr_followthrough_x_range"] = followthrough_x_range
    else:
        features["lr_followthrough_x_range"] = 0.0

    # 6. Wrist velocity lateral component
    wrist_vel = compute_velocity(wrist_w)
    wrist_vel_x = wrist_vel[:, 0]
    features["lr_wrist_vel_x_max"] = np.max(np.abs(wrist_vel_x)) if len(wrist_vel_x) > 0 else 0.0
    features["lr_wrist_vel_x_mean"] = np.mean(wrist_vel_x) if len(wrist_vel_x) > 0 else 0.0

    # 7. Follow-through duration
    features["lr_followthrough_duration"] = window_end - window_start

    return features


def extract_all_phase_specific_features(timeseries: np.ndarray) -> Dict[str, float]:
    """Extract all phase-specific features for all three targets."""
    features = {}

    # Depth features (loading phase: frames 50-150)
    depth_features = extract_depth_features(timeseries, window_start=50, window_end=150)
    features.update(depth_features)

    # Angle features (propulsion phase: frames 100-175)
    angle_features = extract_angle_features(timeseries, window_start=100, window_end=175)
    features.update(angle_features)

    # Left/Right features (release + follow-through: frames 175-240)
    lr_features = extract_left_right_features(timeseries, window_start=175, window_end=240)
    features.update(lr_features)

    return features


def validate_phase_specific_approach():
    """
    Validate phase-specific feature windows hypothesis.

    Test: Train 3 separate models using different temporal windows:
    - Model_depth: Uses frames 50-150 only
    - Model_angle: Uses frames 100-175 only
    - Model_lr: Uses frames 175-240 only

    Compare to baseline (all frames).

    Hypothesis: Phase-specific models improve by 15-25% per target.
    """
    print("="*70)
    print("PHASE-SPECIFIC FEATURE WINDOWS VALIDATION")
    print("="*70)

    # Initialize keypoint mapping
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Extract features
    print("\nExtracting phase-specific features for all training shots...")

    all_features = []
    all_outcomes = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True, chunk_size=20)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}...")

        features = extract_all_phase_specific_features(timeseries)

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

    print(f"\nFeature counts:")
    depth_features = [c for c in df_features.columns if c.startswith("depth_")]
    angle_features = [c for c in df_features.columns if c.startswith("angle_")]
    lr_features = [c for c in df_features.columns if c.startswith("lr_")]

    print(f"  Depth features (frames 50-150): {len(depth_features)}")
    print(f"  Angle features (frames 100-175): {len(angle_features)}")
    print(f"  Left/Right features (frames 175-240): {len(lr_features)}")

    # Train-test split (80/20)
    indices = np.arange(len(df_features))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train = df_features.iloc[train_idx].values
    X_test = df_features.iloc[test_idx].values
    y_train = df_outcomes.iloc[train_idx].values
    y_test = df_outcomes.iloc[test_idx].values

    # Replace NaN with 0
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # =======================================================================
    # Test 1: Target-specific models (separate feature subsets)
    # =======================================================================
    print("\n" + "="*70)
    print("TEST 1: Target-Specific Models (Separate Features)")
    print("="*70)

    # Get feature indices
    depth_feat_idx = [df_features.columns.get_loc(c) for c in depth_features]
    angle_feat_idx = [df_features.columns.get_loc(c) for c in angle_features]
    lr_feat_idx = [df_features.columns.get_loc(c) for c in lr_features]

    results_target_specific = {}

    # Model for DEPTH (using only depth features)
    print("\nTraining depth model (frames 50-150)...")
    X_train_depth = X_train[:, depth_feat_idx]
    X_test_depth = X_test[:, depth_feat_idx]
    model_depth = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model_depth.fit(X_train_depth, y_train[:, 1])  # y[:, 1] is depth
    pred_depth = model_depth.predict(X_test_depth)
    mse_depth = mean_squared_error(y_test[:, 1], pred_depth)
    r2_depth = r2_score(y_test[:, 1], pred_depth)
    print(f"  Depth MSE: {mse_depth:.4f}, R²: {r2_depth:.4f}")
    results_target_specific["depth"] = {"mse": mse_depth, "r2": r2_depth}

    # Model for ANGLE (using only angle features)
    print("\nTraining angle model (frames 100-175)...")
    X_train_angle = X_train[:, angle_feat_idx]
    X_test_angle = X_test[:, angle_feat_idx]
    model_angle = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model_angle.fit(X_train_angle, y_train[:, 0])  # y[:, 0] is angle
    pred_angle = model_angle.predict(X_test_angle)
    mse_angle = mean_squared_error(y_test[:, 0], pred_angle)
    r2_angle = r2_score(y_test[:, 0], pred_angle)
    print(f"  Angle MSE: {mse_angle:.4f}, R²: {r2_angle:.4f}")
    results_target_specific["angle"] = {"mse": mse_angle, "r2": r2_angle}

    # Model for LEFT/RIGHT (using only lr features)
    print("\nTraining left/right model (frames 175-240)...")
    X_train_lr = X_train[:, lr_feat_idx]
    X_test_lr = X_test[:, lr_feat_idx]
    model_lr = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model_lr.fit(X_train_lr, y_train[:, 2])  # y[:, 2] is left_right
    pred_lr = model_lr.predict(X_test_lr)
    mse_lr = mean_squared_error(y_test[:, 2], pred_lr)
    r2_lr = r2_score(y_test[:, 2], pred_lr)
    print(f"  Left/Right MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}")
    results_target_specific["left_right"] = {"mse": mse_lr, "r2": r2_lr}

    # Overall MSE (average across targets)
    overall_mse_target_specific = (mse_angle + mse_depth + mse_lr) / 3
    print(f"\n  Overall MSE (target-specific): {overall_mse_target_specific:.4f}")

    # =======================================================================
    # Test 2: Multi-output model (all features)
    # =======================================================================
    print("\n" + "="*70)
    print("TEST 2: Multi-Output Model (All Features)")
    print("="*70)

    print("\nTraining multi-output model...")
    model_multi = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)

    results_multi = {}
    for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
        model_multi.fit(X_train, y_train[:, target_idx])
        pred = model_multi.predict(X_test)
        mse = mean_squared_error(y_test[:, target_idx], pred)
        r2 = r2_score(y_test[:, target_idx], pred)
        print(f"  {target_name:12} MSE: {mse:.4f}, R²: {r2:.4f}")
        results_multi[target_name] = {"mse": mse, "r2": r2}

    overall_mse_multi = (results_multi["angle"]["mse"] + results_multi["depth"]["mse"] + results_multi["left_right"]["mse"]) / 3
    print(f"\n  Overall MSE (multi-output): {overall_mse_multi:.4f}")

    # =======================================================================
    # Comparison
    # =======================================================================
    print("\n" + "="*70)
    print("COMPARISON: Target-Specific vs Multi-Output")
    print("="*70)

    for target_name in ["angle", "depth", "left_right"]:
        mse_specific = results_target_specific[target_name]["mse"]
        mse_multi = results_multi[target_name]["mse"]
        improvement = (1 - mse_specific / mse_multi) * 100

        print(f"\n{target_name.upper()}:")
        print(f"  Target-specific: MSE = {mse_specific:.4f}")
        print(f"  Multi-output:    MSE = {mse_multi:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")

    overall_improvement = (1 - overall_mse_target_specific / overall_mse_multi) * 100
    print(f"\nOVERALL:")
    print(f"  Target-specific: MSE = {overall_mse_target_specific:.4f}")
    print(f"  Multi-output:    MSE = {overall_mse_multi:.4f}")
    print(f"  Improvement: {overall_improvement:+.1f}%")

    # Save features
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    df_features.to_csv(output_dir / "phase_specific_features.csv", index=False)
    print(f"\nSaved features to {output_dir / 'phase_specific_features.csv'}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    return results_target_specific, results_multi


if __name__ == "__main__":
    validate_phase_specific_approach()
