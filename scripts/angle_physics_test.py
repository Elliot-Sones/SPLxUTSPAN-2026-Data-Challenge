"""
ANGLE Physics-Based Feature Test

This script tests physics-based feature groups for predicting ANGLE from first principles.
Each feature group is derived from basketball shooting biomechanics, not data mining.

Test methodology:
1. Extract physics-based features (leg drive, arm, timing, posture)
2. Test each group per player to identify shooting styles
3. Test stability across random splits
4. Compare to baseline data-mined features
5. Report with full transparency

Author: Claude
Date: 2024-01-29
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns, NUM_FRAMES

# Constants
FRAME_RATE = 60
DT = 1.0 / FRAME_RATE

# Output
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


class KeypointExtractor:
    """Helper class to extract keypoint data by name."""

    def __init__(self, keypoint_cols: List[str]):
        self.keypoint_cols = keypoint_cols
        self.keypoint_to_idx = {}

        for i, col in enumerate(keypoint_cols):
            self.keypoint_to_idx[col] = i

    def get_xyz(self, timeseries: np.ndarray, keypoint: str) -> np.ndarray:
        """Get x, y, z coordinates for a keypoint. Shape: (240, 3)"""
        x_idx = self.keypoint_to_idx.get(f'{keypoint}_x')
        y_idx = self.keypoint_to_idx.get(f'{keypoint}_y')
        z_idx = self.keypoint_to_idx.get(f'{keypoint}_z')

        if x_idx is None:
            raise ValueError(f"Keypoint {keypoint} not found")

        return np.stack([
            timeseries[:, x_idx],
            timeseries[:, y_idx],
            timeseries[:, z_idx]
        ], axis=1)

    def get_coord(self, timeseries: np.ndarray, keypoint: str, coord: str) -> np.ndarray:
        """Get single coordinate. Shape: (240,)"""
        idx = self.keypoint_to_idx.get(f'{keypoint}_{coord}')
        if idx is None:
            raise ValueError(f"Keypoint {keypoint}_{coord} not found")
        return timeseries[:, idx]


def compute_velocity(series: np.ndarray) -> np.ndarray:
    """Compute velocity using central differences."""
    return np.gradient(series, DT, axis=0)


def compute_angle_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle at p2 formed by p1-p2-p3 vectors."""
    v1 = p1 - p2
    v2 = p3 - p2

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return np.nan

    cos_angle = np.clip(dot / (norm1 * norm2), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def detect_release_frame(wrist_xyz: np.ndarray, search_start: int = 80) -> int:
    """Detect release frame as peak wrist velocity after search_start."""
    vel = compute_velocity(wrist_xyz)
    vel_mag = np.linalg.norm(vel, axis=1)

    # Find peak after search_start
    search_region = vel_mag[search_start:]
    peak_idx = search_start + np.argmax(search_region)

    return peak_idx


def extract_leg_drive_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """
    Group 1: Leg Drive Features

    Physics: Leg extension adds upward momentum to the body, which transfers to the ball.
    More leg drive = higher release point = steeper arc needed.
    """
    features = {}

    # Get keypoint data
    left_ankle_z = kp.get_coord(ts, 'left_ankle', 'z')
    right_ankle_z = kp.get_coord(ts, 'right_ankle', 'z')
    left_knee_z = kp.get_coord(ts, 'left_knee', 'z')
    right_knee_z = kp.get_coord(ts, 'right_knee', 'z')
    left_hip_z = kp.get_coord(ts, 'left_hip', 'z')
    right_hip_z = kp.get_coord(ts, 'right_hip', 'z')
    mid_hip_z = kp.get_coord(ts, 'mid_hip', 'z')

    # Frame 153 (identified as critical for angle)
    f = 153
    features['f153_left_ankle_z'] = left_ankle_z[f]
    features['f153_right_ankle_z'] = right_ankle_z[f]
    features['f153_left_knee_z'] = left_knee_z[f]
    features['f153_right_knee_z'] = right_knee_z[f]
    features['f153_mid_hip_z'] = mid_hip_z[f]

    # Change from start (frames 0-30 average) to peak extension (frames 140-160 average)
    start_ankle_z = np.nanmean([left_ankle_z[0:30].mean(), right_ankle_z[0:30].mean()])
    peak_ankle_z = np.nanmean([left_ankle_z[140:160].mean(), right_ankle_z[140:160].mean()])
    features['ankle_z_rise'] = peak_ankle_z - start_ankle_z

    start_knee_z = np.nanmean([left_knee_z[0:30].mean(), right_knee_z[0:30].mean()])
    peak_knee_z = np.nanmean([left_knee_z[140:160].mean(), right_knee_z[140:160].mean()])
    features['knee_z_rise'] = peak_knee_z - start_knee_z

    start_hip_z = mid_hip_z[0:30].mean()
    peak_hip_z = mid_hip_z[140:160].mean()
    features['hip_z_rise'] = peak_hip_z - start_hip_z

    # Hip velocity at frame 153 (positive = still going up)
    hip_vel = compute_velocity(mid_hip_z)
    features['hip_z_velocity_f153'] = hip_vel[153]

    # Knee extension angle at frame 153
    # Angle at knee: hip-knee-ankle
    try:
        left_hip = kp.get_xyz(ts, 'left_hip')[153]
        left_knee = kp.get_xyz(ts, 'left_knee')[153]
        left_ankle = kp.get_xyz(ts, 'left_ankle')[153]
        features['left_knee_angle_f153'] = compute_angle_3points(left_hip, left_knee, left_ankle)

        right_hip = kp.get_xyz(ts, 'right_hip')[153]
        right_knee = kp.get_xyz(ts, 'right_knee')[153]
        right_ankle = kp.get_xyz(ts, 'right_ankle')[153]
        features['right_knee_angle_f153'] = compute_angle_3points(right_hip, right_knee, right_ankle)
    except:
        features['left_knee_angle_f153'] = np.nan
        features['right_knee_angle_f153'] = np.nan

    return features


def extract_arm_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """
    Group 2: Arm Extension Features

    Physics: Arm angle and extension at release determines the direction of force on the ball.
    Elbow angle affects release angle; wrist snap affects final trajectory.
    """
    features = {}

    # Detect release frame
    wrist_xyz = kp.get_xyz(ts, 'right_wrist')
    release_frame = detect_release_frame(wrist_xyz)
    features['release_frame'] = release_frame

    # Wrist position at release
    features['wrist_x_release'] = wrist_xyz[release_frame, 0]
    features['wrist_y_release'] = wrist_xyz[release_frame, 1]
    features['wrist_z_release'] = wrist_xyz[release_frame, 2]

    # Wrist velocity at release
    wrist_vel = compute_velocity(wrist_xyz)
    features['wrist_vx_release'] = wrist_vel[release_frame, 0]
    features['wrist_vy_release'] = wrist_vel[release_frame, 1]
    features['wrist_vz_release'] = wrist_vel[release_frame, 2]
    features['wrist_speed_release'] = np.linalg.norm(wrist_vel[release_frame])

    # Release angle (vertical component of velocity direction)
    v_horizontal = np.sqrt(wrist_vel[release_frame, 0]**2 + wrist_vel[release_frame, 1]**2)
    if v_horizontal > 0.01:
        features['release_angle_computed'] = np.degrees(np.arctan2(wrist_vel[release_frame, 2], v_horizontal))
    else:
        features['release_angle_computed'] = np.nan

    # Elbow angle at release
    try:
        shoulder = kp.get_xyz(ts, 'right_shoulder')[release_frame]
        elbow = kp.get_xyz(ts, 'right_elbow')[release_frame]
        wrist = kp.get_xyz(ts, 'right_wrist')[release_frame]
        features['elbow_angle_release'] = compute_angle_3points(shoulder, elbow, wrist)
    except:
        features['elbow_angle_release'] = np.nan

    # Shoulder elevation at release (shoulder_z - hip_z)
    shoulder_z = kp.get_coord(ts, 'right_shoulder', 'z')[release_frame]
    hip_z = kp.get_coord(ts, 'mid_hip', 'z')[release_frame]
    features['shoulder_elevation_release'] = shoulder_z - hip_z

    # Arm alignment (is elbow under wrist at release?)
    elbow_x = kp.get_coord(ts, 'right_elbow', 'x')[release_frame]
    wrist_x = kp.get_coord(ts, 'right_wrist', 'x')[release_frame]
    features['elbow_wrist_x_diff'] = wrist_x - elbow_x

    return features


def extract_timing_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """
    Group 3: Timing Features

    Physics: When in the jump the ball is released affects the upward velocity component.
    Release on way up = extra upward velocity from body momentum.
    Release at peak = neutral.
    Release on way down = reduced upward velocity.
    """
    features = {}

    # Get hip trajectory (proxy for body center of mass)
    hip_z = kp.get_coord(ts, 'mid_hip', 'z')
    hip_vel = compute_velocity(hip_z)

    # Detect jump start (when hip_vel becomes significantly positive)
    # and jump peak (when hip_vel crosses zero from positive to negative)

    # Find jump start: first frame where hip is rising significantly
    threshold = 0.5  # m/s
    jump_start = 60  # default
    for i in range(60, 180):
        if hip_vel[i] > threshold:
            jump_start = i
            break

    # Find jump peak: where velocity crosses zero
    jump_peak = 150  # default
    for i in range(jump_start, 200):
        if i > 0 and hip_vel[i-1] > 0 and hip_vel[i] <= 0:
            jump_peak = i
            break

    features['jump_start_frame'] = jump_start
    features['jump_peak_frame'] = jump_peak

    # Detect release frame
    wrist_xyz = kp.get_xyz(ts, 'right_wrist')
    release_frame = detect_release_frame(wrist_xyz)
    features['release_frame_timing'] = release_frame

    # Release timing relative to jump
    if jump_peak > jump_start:
        features['release_in_jump_phase'] = (release_frame - jump_start) / (jump_peak - jump_start)
    else:
        features['release_in_jump_phase'] = 0.5

    # Hip velocity at release (positive = still going up, negative = coming down)
    features['hip_velocity_at_release'] = hip_vel[release_frame]

    # Is release before, at, or after peak?
    features['release_before_peak'] = 1.0 if release_frame < jump_peak else 0.0
    features['frames_from_peak'] = release_frame - jump_peak

    # Jump height (max hip_z - initial hip_z)
    features['jump_height'] = np.nanmax(hip_z[60:200]) - np.nanmean(hip_z[0:30])

    return features


def extract_posture_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """
    Group 4: Body Posture Features

    Physics: Body lean affects the direction of force application.
    Leaning back = more vertical release.
    Leaning forward = flatter release.
    """
    features = {}

    # Detect release frame
    wrist_xyz = kp.get_xyz(ts, 'right_wrist')
    release_frame = detect_release_frame(wrist_xyz)

    # Trunk lean at release
    shoulder_z = kp.get_coord(ts, 'right_shoulder', 'z')[release_frame]
    shoulder_y = kp.get_coord(ts, 'right_shoulder', 'y')[release_frame]
    hip_z = kp.get_coord(ts, 'mid_hip', 'z')[release_frame]
    hip_y = kp.get_coord(ts, 'mid_hip', 'y')[release_frame]

    # Trunk angle: how much is upper body leaning forward/back
    dz = shoulder_z - hip_z
    dy = shoulder_y - hip_y
    if abs(dy) > 0.01:
        features['trunk_lean_angle'] = np.degrees(np.arctan2(dz, dy))
    else:
        features['trunk_lean_angle'] = 90.0  # vertical

    # Head position relative to body
    neck_z = kp.get_coord(ts, 'neck', 'z')[release_frame]
    features['head_above_shoulder'] = neck_z - shoulder_z

    # Shoulder alignment (are shoulders level?)
    left_shoulder_z = kp.get_coord(ts, 'left_shoulder', 'z')[release_frame]
    right_shoulder_z = kp.get_coord(ts, 'right_shoulder', 'z')[release_frame]
    features['shoulder_tilt'] = right_shoulder_z - left_shoulder_z

    # Hip alignment
    left_hip_z = kp.get_coord(ts, 'left_hip', 'z')[release_frame]
    right_hip_z = kp.get_coord(ts, 'right_hip', 'z')[release_frame]
    features['hip_tilt'] = right_hip_z - left_hip_z

    # Body sway during shot (stability)
    hip_x = kp.get_coord(ts, 'mid_hip', 'x')
    features['body_sway_x'] = np.nanstd(hip_x[60:180])

    # Eye/head focus (is head stable during release?)
    nose_z = kp.get_coord(ts, 'nose', 'z')
    features['head_stability'] = np.nanstd(nose_z[release_frame-10:release_frame+10])

    return features


def extract_datamined_baseline(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """
    Baseline: Current data-mined features from advanced_features.py

    These are the features that were found to have high R^2 on training data.
    Included for comparison.
    """
    features = {}

    # Frame 153 features (the "critical" frame)
    f = 153

    for keypoint in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee',
                     'left_hip', 'right_hip', 'mid_hip']:
        for coord in ['x', 'y', 'z']:
            try:
                val = kp.get_coord(ts, keypoint, coord)[f]
                features[f'baseline_f153_{keypoint}_{coord}'] = val
            except:
                features[f'baseline_f153_{keypoint}_{coord}'] = np.nan

    # Also frame 133 (the actual peak R^2 frame from analysis)
    f = 133
    for keypoint in ['right_elbow', 'right_wrist', 'right_shoulder']:
        for coord in ['x', 'y', 'z']:
            try:
                val = kp.get_coord(ts, keypoint, coord)[f]
                features[f'baseline_f133_{keypoint}_{coord}'] = val
            except:
                features[f'baseline_f133_{keypoint}_{coord}'] = np.nan

    return features


def compute_r2_simple(X: np.ndarray, y: np.ndarray) -> float:
    """Compute R^2 using simple linear regression."""
    # Remove NaN
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_valid = X[valid]
    y_valid = y[valid]

    if len(y_valid) < 10 or X_valid.shape[1] == 0:
        return np.nan

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Fit and score
    model = Ridge(alpha=1.0)
    try:
        model.fit(X_scaled, y_valid)
        return model.score(X_scaled, y_valid)
    except:
        return np.nan


def compute_cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[float, float]:
    """Compute cross-validated R^2."""
    # Remove NaN
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_valid = X[valid]
    y_valid = y[valid]

    if len(y_valid) < n_splits * 2:
        return np.nan, np.nan

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # CV
    model = Ridge(alpha=1.0)
    try:
        scores = cross_val_score(model, X_scaled, y_valid, cv=n_splits, scoring='r2')
        return scores.mean(), scores.std()
    except:
        return np.nan, np.nan


def compute_stability_score(X: np.ndarray, y: np.ndarray, n_bootstrap: int = 20) -> float:
    """
    Compute stability score: how consistent is R^2 across random 80% subsets?

    High stability = feature is likely signal
    Low stability = feature is likely noise
    """
    r2_scores = []
    n = len(y)

    for seed in range(n_bootstrap):
        np.random.seed(seed)
        idx = np.random.choice(n, size=int(0.8 * n), replace=False)
        r2 = compute_r2_simple(X[idx], y[idx])
        if not np.isnan(r2):
            r2_scores.append(r2)

    if len(r2_scores) < 5:
        return np.nan

    # Stability = mean / std (like Sharpe ratio)
    # Higher = more stable
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    if std_r2 < 0.001:
        return 100.0  # Very stable

    return mean_r2 / std_r2


def main():
    print("=" * 80)
    print("ANGLE PHYSICS-BASED FEATURE TEST")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    angle_target = y[:, 0]  # angle is first target
    participant_ids = meta['participant_id'].values

    print(f"Loaded {len(angle_target)} shots")
    print(f"Participants: {np.unique(participant_ids)}")
    print()

    # Extract all feature groups for all shots
    print("Extracting physics-based features...")

    all_leg_drive = []
    all_arm = []
    all_timing = []
    all_posture = []
    all_baseline = []

    for i in range(len(X)):
        ts = X[i]

        all_leg_drive.append(extract_leg_drive_features(ts, kp))
        all_arm.append(extract_arm_features(ts, kp))
        all_timing.append(extract_timing_features(ts, kp))
        all_posture.append(extract_posture_features(ts, kp))
        all_baseline.append(extract_datamined_baseline(ts, kp))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(X)}")

    # Convert to DataFrames
    df_leg_drive = pd.DataFrame(all_leg_drive)
    df_arm = pd.DataFrame(all_arm)
    df_timing = pd.DataFrame(all_timing)
    df_posture = pd.DataFrame(all_posture)
    df_baseline = pd.DataFrame(all_baseline)

    # Combine all physics features
    df_all_physics = pd.concat([df_leg_drive, df_arm, df_timing, df_posture], axis=1)

    print(f"\nFeature counts:")
    print(f"  Leg Drive: {len(df_leg_drive.columns)}")
    print(f"  Arm: {len(df_arm.columns)}")
    print(f"  Timing: {len(df_timing.columns)}")
    print(f"  Posture: {len(df_posture.columns)}")
    print(f"  Baseline (data-mined): {len(df_baseline.columns)}")
    print(f"  All Physics Combined: {len(df_all_physics.columns)}")

    # ==========================================================================
    # TEST 1: Overall R^2 for each feature group
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Overall R^2 for each feature group (ALL DATA)")
    print("=" * 80)

    results_overall = []

    for name, df in [
        ('Leg Drive', df_leg_drive),
        ('Arm', df_arm),
        ('Timing', df_timing),
        ('Posture', df_posture),
        ('Baseline (data-mined)', df_baseline),
        ('All Physics Combined', df_all_physics),
    ]:
        X_group = df.values
        r2_train = compute_r2_simple(X_group, angle_target)
        cv_mean, cv_std = compute_cv_r2(X_group, angle_target)
        stability = compute_stability_score(X_group, angle_target)

        results_overall.append({
            'Feature Group': name,
            'N Features': X_group.shape[1],
            'Train R²': r2_train,
            'CV R² (mean)': cv_mean,
            'CV R² (std)': cv_std,
            'Stability': stability,
        })

        print(f"\n{name}:")
        print(f"  Features: {X_group.shape[1]}")
        print(f"  Train R²: {r2_train:.4f}")
        print(f"  CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"  Stability: {stability:.2f}")

    # ==========================================================================
    # TEST 2: Per-player R^2 to identify shooting styles
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Per-player R^2 (identify shooting styles)")
    print("=" * 80)

    results_per_player = []

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        n_shots = mask.sum()

        print(f"\n--- Player {pid} ({n_shots} shots) ---")

        for name, df in [
            ('Leg Drive', df_leg_drive),
            ('Arm', df_arm),
            ('Timing', df_timing),
            ('Posture', df_posture),
        ]:
            X_group = df.values[mask]
            y_player = angle_target[mask]

            r2 = compute_r2_simple(X_group, y_player)
            cv_mean, cv_std = compute_cv_r2(X_group, y_player, n_splits=3)

            results_per_player.append({
                'Player': pid,
                'Feature Group': name,
                'N Shots': n_shots,
                'Train R²': r2,
                'CV R² (mean)': cv_mean,
            })

            print(f"  {name}: Train R²={r2:.3f}, CV R²={cv_mean:.3f}")

    # Identify dominant shooting style per player
    print("\n--- SHOOTING STYLE SUMMARY ---")
    df_pp = pd.DataFrame(results_per_player)
    for pid in sorted(np.unique(participant_ids)):
        player_data = df_pp[df_pp['Player'] == pid]
        best_group = player_data.loc[player_data['Train R²'].idxmax(), 'Feature Group']
        best_r2 = player_data['Train R²'].max()
        print(f"  Player {pid}: Dominant style = {best_group} (R²={best_r2:.3f})")

    # ==========================================================================
    # TEST 3: Individual feature correlations
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Top individual features by correlation")
    print("=" * 80)

    all_features_df = pd.concat([df_all_physics, df_baseline], axis=1)

    correlations = []
    for col in all_features_df.columns:
        vals = all_features_df[col].values
        valid = ~(np.isnan(vals) | np.isnan(angle_target))
        if valid.sum() > 30:
            r, p = stats.pearsonr(vals[valid], angle_target[valid])
            correlations.append({
                'Feature': col,
                'Pearson r': r,
                'R²': r**2,
                'p-value': p,
                'N valid': valid.sum(),
            })

    corr_df = pd.DataFrame(correlations).sort_values('R²', ascending=False)

    print("\nTop 20 features by R²:")
    print(corr_df.head(20).to_string(index=False))

    # ==========================================================================
    # TEST 4: Leave-one-player-out cross-validation (generalization test)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Leave-one-player-out validation (GENERALIZATION TEST)")
    print("=" * 80)
    print("\nThis tests if features generalize across players.")
    print("High train, low test = OVERFITTING (noise)")
    print("Similar train and test = SIGNAL (generalizes)")

    lopo_results = []

    for name, df in [
        ('Leg Drive', df_leg_drive),
        ('Arm', df_arm),
        ('Timing', df_timing),
        ('Posture', df_posture),
        ('Baseline (data-mined)', df_baseline),
        ('All Physics Combined', df_all_physics),
    ]:
        X_group = df.values

        train_r2s = []
        test_r2s = []

        for test_pid in sorted(np.unique(participant_ids)):
            train_mask = participant_ids != test_pid
            test_mask = participant_ids == test_pid

            X_train = X_group[train_mask]
            y_train = angle_target[train_mask]
            X_test = X_group[test_mask]
            y_test = angle_target[test_mask]

            # Remove NaN
            valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            valid_test = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))

            X_train_clean = X_train[valid_train]
            y_train_clean = y_train[valid_train]
            X_test_clean = X_test[valid_test]
            y_test_clean = y_test[valid_test]

            if len(y_train_clean) < 10 or len(y_test_clean) < 5:
                continue

            # Standardize using train stats
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test_clean)

            # Fit and evaluate
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train_clean)

            train_r2 = model.score(X_train_scaled, y_train_clean)
            test_r2 = model.score(X_test_scaled, y_test_clean)

            train_r2s.append(train_r2)
            test_r2s.append(test_r2)

        mean_train = np.mean(train_r2s)
        mean_test = np.mean(test_r2s)
        gap = mean_train - mean_test

        lopo_results.append({
            'Feature Group': name,
            'Train R² (mean)': mean_train,
            'Test R² (mean)': mean_test,
            'Gap (overfit)': gap,
            'Generalization %': (mean_test / mean_train * 100) if mean_train > 0 else 0,
        })

        print(f"\n{name}:")
        print(f"  Train R²: {mean_train:.4f}")
        print(f"  Test R² (held-out player): {mean_test:.4f}")
        print(f"  Gap: {gap:.4f}")
        print(f"  Generalization: {mean_test/mean_train*100:.1f}%" if mean_train > 0 else "  Generalization: N/A")

    # ==========================================================================
    # TEST 5: Feature stability across bootstrap samples
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Feature stability across 50 bootstrap samples")
    print("=" * 80)
    print("\nHigh stability = consistent signal")
    print("Low stability = fitting to noise")

    stability_results = []

    for name, df in [
        ('Leg Drive', df_leg_drive),
        ('Arm', df_arm),
        ('Timing', df_timing),
        ('Posture', df_posture),
        ('Baseline (data-mined)', df_baseline),
        ('All Physics Combined', df_all_physics),
    ]:
        X_group = df.values
        stability = compute_stability_score(X_group, angle_target, n_bootstrap=50)

        stability_results.append({
            'Feature Group': name,
            'Stability Score': stability,
        })

        print(f"  {name}: {stability:.2f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Which features should we use for ANGLE?")
    print("=" * 80)

    summary_df = pd.DataFrame(lopo_results)
    summary_df = summary_df.sort_values('Generalization %', ascending=False)

    print("\nRanking by generalization (test R² / train R²):")
    print(summary_df.to_string(index=False))

    print("\n--- KEY FINDINGS ---")

    best_gen = summary_df.iloc[0]
    print(f"\n1. Best generalizing features: {best_gen['Feature Group']}")
    print(f"   Train R²: {best_gen['Train R² (mean)']:.4f}")
    print(f"   Test R²: {best_gen['Test R² (mean)']:.4f}")
    print(f"   Generalization: {best_gen['Generalization %']:.1f}%")

    worst_gen = summary_df.iloc[-1]
    print(f"\n2. Worst generalizing (most overfit): {worst_gen['Feature Group']}")
    print(f"   Train R²: {worst_gen['Train R² (mean)']:.4f}")
    print(f"   Test R²: {worst_gen['Test R² (mean)']:.4f}")
    print(f"   Generalization: {worst_gen['Generalization %']:.1f}%")

    # Save detailed results
    results_file = OUTPUT_DIR / 'angle_physics_test_results.csv'
    summary_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Save correlation results
    corr_file = OUTPUT_DIR / 'angle_feature_correlations.csv'
    corr_df.to_csv(corr_file, index=False)
    print(f"Correlations saved to: {corr_file}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
