"""
MULTI-FRAME PHYSICS TEST

The hypothesis: Single-frame features miss the MOTION physics.
Physics of a shot is about HOW force is applied over time, not just position at one instant.

This test systematically evaluates:
1. Velocity features (dx/dt) - rate of change between frames
2. Acceleration features (d²x/dt²) - change in velocity
3. Trajectory features - the path taken over multiple frames
4. Force proxy features - mass * acceleration approximations

For each player, to see if different players have different optimal motion patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


class KeypointExtractor:
    """Extract specific keypoints from the raw data."""

    def __init__(self, keypoint_cols: List[str]):
        self.keypoint_cols = keypoint_cols
        self._build_index()

    def _build_index(self):
        self.col_to_idx = {col: i for i, col in enumerate(self.keypoint_cols)}

    def get(self, ts: np.ndarray, name: str, frame: int = None) -> np.ndarray:
        """Get keypoint values. If frame is None, return all frames."""
        idx = self.col_to_idx.get(name)
        if idx is None:
            return np.nan if frame is not None else np.full(ts.shape[0], np.nan)
        if frame is not None:
            if frame < 0 or frame >= ts.shape[0]:
                return np.nan
            return ts[frame, idx]
        return ts[:, idx]


def compute_velocity(ts: np.ndarray, kp: KeypointExtractor, keypoint: str,
                     start_frame: int, end_frame: int) -> float:
    """Compute velocity (change per frame) between two frames."""
    val_start = kp.get(ts, keypoint, start_frame)
    val_end = kp.get(ts, keypoint, end_frame)
    if np.isnan(val_start) or np.isnan(val_end):
        return np.nan
    return (val_end - val_start) / (end_frame - start_frame)


def compute_acceleration(ts: np.ndarray, kp: KeypointExtractor, keypoint: str,
                         frame: int, delta: int = 5) -> float:
    """Compute acceleration (change in velocity) around a frame."""
    v1 = compute_velocity(ts, kp, keypoint, frame - delta, frame)
    v2 = compute_velocity(ts, kp, keypoint, frame, frame + delta)
    if np.isnan(v1) or np.isnan(v2):
        return np.nan
    return (v2 - v1) / delta


def compute_trajectory_integral(ts: np.ndarray, kp: KeypointExtractor, keypoint: str,
                                start_frame: int, end_frame: int) -> float:
    """Compute integral of trajectory (area under curve) - captures total displacement effort."""
    values = kp.get(ts, keypoint)
    if np.any(np.isnan(values[start_frame:end_frame+1])):
        return np.nan
    return np.trapezoid(values[start_frame:end_frame+1])


def compute_trajectory_smoothness(ts: np.ndarray, kp: KeypointExtractor, keypoint: str,
                                  start_frame: int, end_frame: int) -> float:
    """Compute trajectory smoothness (inverse of jerk - third derivative)."""
    values = kp.get(ts, keypoint)
    segment = values[start_frame:end_frame+1]
    if len(segment) < 4 or np.any(np.isnan(segment)):
        return np.nan
    # Jerk is third derivative - lower jerk = smoother motion
    jerk = np.diff(segment, n=3)
    return 1.0 / (np.std(jerk) + 1e-6)


def compute_peak_velocity_frame(ts: np.ndarray, kp: KeypointExtractor, keypoint: str,
                                start_frame: int, end_frame: int) -> Tuple[int, float]:
    """Find frame with peak velocity and return (frame, velocity)."""
    values = kp.get(ts, keypoint)
    segment = values[start_frame:end_frame+1]
    if len(segment) < 2 or np.any(np.isnan(segment)):
        return (np.nan, np.nan)
    velocities = np.diff(segment)
    peak_idx = np.argmax(np.abs(velocities))
    return (start_frame + peak_idx, velocities[peak_idx])


def extract_velocity_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """Extract velocity-based features at multiple time windows."""
    features = {}

    # Key keypoints for shooting
    keypoints = [
        'right_wrist_z', 'left_wrist_z',
        'right_elbow_z', 'left_elbow_z',
        'right_knee_z', 'left_knee_z',
        'right_ankle_z', 'left_ankle_z',
        'right_hip_z', 'left_hip_z',
    ]

    # Key phases of the shot (approximate)
    phases = [
        ('setup', 60, 90),      # Early setup
        ('load', 90, 120),      # Loading phase
        ('push', 120, 150),     # Push/acceleration phase
        ('release', 150, 180),  # Release phase
        ('follow', 180, 210),   # Follow through
    ]

    for kp_name in keypoints:
        for phase_name, start, end in phases:
            vel = compute_velocity(ts, kp, kp_name, start, end)
            features[f'{kp_name}_vel_{phase_name}'] = vel

    # Also compute velocity at specific critical windows around frame 153
    critical_windows = [(143, 153), (153, 163), (148, 158), (140, 160)]
    for kp_name in keypoints:
        for i, (start, end) in enumerate(critical_windows):
            vel = compute_velocity(ts, kp, kp_name, start, end)
            features[f'{kp_name}_vel_crit{i}'] = vel

    return features


def extract_acceleration_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """Extract acceleration-based features."""
    features = {}

    keypoints = [
        'right_wrist_z', 'left_wrist_z',
        'right_elbow_z', 'left_elbow_z',
        'right_knee_z', 'left_knee_z',
        'right_ankle_z', 'left_ankle_z',
    ]

    # Key frames to measure acceleration
    frames = [120, 135, 150, 153, 165, 180]
    deltas = [5, 10, 15]

    for kp_name in keypoints:
        for frame in frames:
            for delta in deltas:
                acc = compute_acceleration(ts, kp, kp_name, frame, delta)
                features[f'{kp_name}_acc_f{frame}_d{delta}'] = acc

    return features


def extract_trajectory_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """Extract trajectory-based features."""
    features = {}

    keypoints = [
        'right_wrist_z', 'left_wrist_z',
        'right_elbow_z', 'left_elbow_z',
    ]

    # Key segments of the shot
    segments = [
        ('full_push', 100, 160),
        ('release_window', 140, 170),
        ('follow_through', 160, 200),
    ]

    for kp_name in keypoints:
        for seg_name, start, end in segments:
            # Total displacement
            integral = compute_trajectory_integral(ts, kp, kp_name, start, end)
            features[f'{kp_name}_integral_{seg_name}'] = integral

            # Smoothness
            smoothness = compute_trajectory_smoothness(ts, kp, kp_name, start, end)
            features[f'{kp_name}_smooth_{seg_name}'] = smoothness

            # Peak velocity info
            peak_frame, peak_vel = compute_peak_velocity_frame(ts, kp, kp_name, start, end)
            features[f'{kp_name}_peak_frame_{seg_name}'] = peak_frame
            features[f'{kp_name}_peak_vel_{seg_name}'] = peak_vel

    return features


def extract_force_proxy_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """Extract force-proxy features (acceleration * position as mass proxy)."""
    features = {}

    # The idea: heavier body parts (legs) accelerating contribute more force
    # We proxy this by looking at acceleration of body segments

    # Leg drive force proxy
    for side in ['right', 'left']:
        # Knee extension acceleration (straightening leg)
        knee_z = kp.get(ts, f'{side}_knee_z')
        hip_z = kp.get(ts, f'{side}_hip_z')
        ankle_z = kp.get(ts, f'{side}_ankle_z')

        if not np.any(np.isnan(knee_z)) and not np.any(np.isnan(hip_z)):
            # Leg extension = knee moving away from hip
            leg_ext = knee_z - hip_z
            # Acceleration of leg extension during push phase
            push_segment = leg_ext[120:160]
            if len(push_segment) > 3:
                leg_acc = np.diff(push_segment, n=2)
                features[f'{side}_leg_ext_acc_mean'] = np.mean(leg_acc)
                features[f'{side}_leg_ext_acc_max'] = np.max(leg_acc)
                features[f'{side}_leg_ext_acc_timing'] = np.argmax(leg_acc)

    # Arm extension force proxy
    for side in ['right', 'left']:
        wrist_z = kp.get(ts, f'{side}_wrist_z')
        elbow_z = kp.get(ts, f'{side}_elbow_z')
        shoulder_z = kp.get(ts, f'{side}_shoulder_z')

        if not np.any(np.isnan(wrist_z)) and not np.any(np.isnan(elbow_z)):
            # Arm extension = wrist moving away from elbow
            arm_ext = wrist_z - elbow_z
            release_segment = arm_ext[140:180]
            if len(release_segment) > 3:
                arm_acc = np.diff(release_segment, n=2)
                features[f'{side}_arm_ext_acc_mean'] = np.mean(arm_acc)
                features[f'{side}_arm_ext_acc_max'] = np.max(arm_acc)
                features[f'{side}_arm_ext_acc_timing'] = np.argmax(arm_acc)

    return features


def extract_coordination_features(ts: np.ndarray, kp: KeypointExtractor) -> Dict[str, float]:
    """Extract features about timing coordination between body parts."""
    features = {}

    # Get peak velocity frames for different body parts
    body_parts = [
        ('right_ankle_z', 80, 160),
        ('right_knee_z', 90, 170),
        ('right_hip_z', 100, 180),
        ('right_elbow_z', 120, 180),
        ('right_wrist_z', 130, 190),
    ]

    peak_frames = {}
    peak_vels = {}

    for kp_name, start, end in body_parts:
        frame, vel = compute_peak_velocity_frame(ts, kp, kp_name, start, end)
        peak_frames[kp_name] = frame
        peak_vels[kp_name] = vel

    # Coordination: timing difference between body parts
    # Good shooting = sequential activation from legs to wrist
    if not np.isnan(peak_frames.get('right_ankle_z', np.nan)) and \
       not np.isnan(peak_frames.get('right_wrist_z', np.nan)):
        features['ankle_to_wrist_delay'] = peak_frames['right_wrist_z'] - peak_frames['right_ankle_z']

    if not np.isnan(peak_frames.get('right_knee_z', np.nan)) and \
       not np.isnan(peak_frames.get('right_elbow_z', np.nan)):
        features['knee_to_elbow_delay'] = peak_frames['right_elbow_z'] - peak_frames['right_knee_z']

    # Velocity ratios
    if not np.isnan(peak_vels.get('right_ankle_z', np.nan)) and \
       not np.isnan(peak_vels.get('right_wrist_z', np.nan)) and \
       abs(peak_vels['right_ankle_z']) > 0.001:
        features['wrist_ankle_vel_ratio'] = peak_vels['right_wrist_z'] / peak_vels['right_ankle_z']

    return features


def within_player_cv(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                     n_splits: int = 5) -> Tuple[float, float, float, float]:
    """Within-player cross-validation."""
    all_train_r2 = []
    all_test_r2 = []

    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = y[mask]

        valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
        X_clean = X_player[valid]
        y_clean = y_player[valid]

        if len(y_clean) < n_splits * 2:
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_clean):
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)

            train_r2 = model.score(X_train_scaled, y_train)
            test_r2 = model.score(X_test_scaled, y_test)

            all_train_r2.append(train_r2)
            all_test_r2.append(test_r2)

    return (
        np.mean(all_train_r2),
        np.std(all_train_r2),
        np.mean(all_test_r2),
        np.std(all_test_r2)
    )


def per_player_cv(X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray,
                  n_splits: int = 5) -> Dict[int, Tuple[float, float]]:
    """Get CV results per player."""
    results = {}

    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        X_player = X[mask]
        y_player = y[mask]

        valid = ~(np.isnan(X_player).any(axis=1) | np.isnan(y_player))
        X_clean = X_player[valid]
        y_clean = y_player[valid]

        if len(y_clean) < n_splits * 2:
            results[pid] = (np.nan, np.nan)
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        test_r2s = []

        for train_idx, test_idx in kf.split(X_clean):
            X_train, X_test = X_clean[train_idx], X_clean[test_idx]
            y_train, y_test = y_clean[train_idx], y_clean[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            test_r2s.append(model.score(X_test_scaled, y_test))

        results[pid] = (np.mean(test_r2s), np.std(test_r2s))

    return results


def main():
    print("=" * 80)
    print("MULTI-FRAME PHYSICS TEST")
    print("=" * 80)
    print()
    print("Testing velocity, acceleration, trajectory, and coordination features")
    print("These capture the MOTION physics, not just single-frame positions")
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    angle_target = y[:, 0]
    participant_ids = meta['participant_id'].values

    print(f"Loaded {len(angle_target)} shots from {len(np.unique(participant_ids))} players")
    print()

    # Extract all feature groups
    print("Extracting multi-frame features...")

    all_velocity = []
    all_acceleration = []
    all_trajectory = []
    all_force = []
    all_coordination = []

    for i in range(len(X)):
        ts = X[i]
        all_velocity.append(extract_velocity_features(ts, kp))
        all_acceleration.append(extract_acceleration_features(ts, kp))
        all_trajectory.append(extract_trajectory_features(ts, kp))
        all_force.append(extract_force_proxy_features(ts, kp))
        all_coordination.append(extract_coordination_features(ts, kp))

    df_velocity = pd.DataFrame(all_velocity)
    df_acceleration = pd.DataFrame(all_acceleration)
    df_trajectory = pd.DataFrame(all_trajectory)
    df_force = pd.DataFrame(all_force)
    df_coordination = pd.DataFrame(all_coordination)

    print(f"Velocity features: {df_velocity.shape[1]}")
    print(f"Acceleration features: {df_acceleration.shape[1]}")
    print(f"Trajectory features: {df_trajectory.shape[1]}")
    print(f"Force proxy features: {df_force.shape[1]}")
    print(f"Coordination features: {df_coordination.shape[1]}")
    print()

    # ==========================================================================
    # TEST 1: Each feature group alone
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: Individual Feature Groups (Within-Player 5-Fold CV)")
    print("=" * 80)
    print()

    feature_groups = [
        ('Velocity', df_velocity),
        ('Acceleration', df_acceleration),
        ('Trajectory', df_trajectory),
        ('Force Proxy', df_force),
        ('Coordination', df_coordination),
    ]

    results = []
    for name, df in feature_groups:
        # Remove columns with all NaN
        df_clean = df.dropna(axis=1, how='all')
        if df_clean.shape[1] == 0:
            print(f"{name}: All features are NaN, skipping")
            continue

        X_group = df_clean.values
        train_mean, train_std, test_mean, test_std = within_player_cv(
            X_group, angle_target, participant_ids, n_splits=5
        )

        results.append({
            'Group': name,
            'N Features': df_clean.shape[1],
            'Train R²': train_mean,
            'Test R²': test_mean,
            'Gap': train_mean - test_mean,
        })

        print(f"{name} ({df_clean.shape[1]} features):")
        print(f"  Train R²: {train_mean:.4f} (±{train_std:.4f})")
        print(f"  Test R²:  {test_mean:.4f} (±{test_std:.4f})")
        print(f"  Gap: {train_mean - test_mean:.4f}")
        print()

    # ==========================================================================
    # TEST 2: Combinations of feature groups
    # ==========================================================================
    print("=" * 80)
    print("TEST 2: Combinations of Feature Groups")
    print("=" * 80)
    print()

    # Clean dataframes
    df_velocity_clean = df_velocity.dropna(axis=1, how='all')
    df_acceleration_clean = df_acceleration.dropna(axis=1, how='all')
    df_trajectory_clean = df_trajectory.dropna(axis=1, how='all')
    df_force_clean = df_force.dropna(axis=1, how='all')
    df_coordination_clean = df_coordination.dropna(axis=1, how='all')

    combinations = [
        ('Velocity + Acceleration', pd.concat([df_velocity_clean, df_acceleration_clean], axis=1)),
        ('Velocity + Trajectory', pd.concat([df_velocity_clean, df_trajectory_clean], axis=1)),
        ('Force + Coordination', pd.concat([df_force_clean, df_coordination_clean], axis=1)),
        ('All Multi-Frame', pd.concat([df_velocity_clean, df_acceleration_clean,
                                       df_trajectory_clean, df_force_clean,
                                       df_coordination_clean], axis=1)),
    ]

    for name, df in combinations:
        X_group = df.values
        train_mean, train_std, test_mean, test_std = within_player_cv(
            X_group, angle_target, participant_ids, n_splits=5
        )

        results.append({
            'Group': name,
            'N Features': df.shape[1],
            'Train R²': train_mean,
            'Test R²': test_mean,
            'Gap': train_mean - test_mean,
        })

        print(f"{name} ({df.shape[1]} features):")
        print(f"  Train R²: {train_mean:.4f} (±{train_std:.4f})")
        print(f"  Test R²:  {test_mean:.4f} (±{test_std:.4f})")
        print()

    # ==========================================================================
    # TEST 3: Per-player analysis
    # ==========================================================================
    print("=" * 80)
    print("TEST 3: Per-Player Results for Best Feature Groups")
    print("=" * 80)
    print()

    best_groups = [
        ('Velocity', df_velocity_clean),
        ('Coordination', df_coordination_clean),
        ('All Multi-Frame', pd.concat([df_velocity_clean, df_acceleration_clean,
                                       df_trajectory_clean, df_force_clean,
                                       df_coordination_clean], axis=1)),
    ]

    for name, df in best_groups:
        print(f"\n{name}:")
        player_results = per_player_cv(df.values, angle_target, participant_ids)
        for pid, (mean_r2, std_r2) in sorted(player_results.items()):
            print(f"  Player {pid}: Test R² = {mean_r2:.4f} (±{std_r2:.4f})")

    # ==========================================================================
    # TEST 4: Find individual features with positive signal
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 4: Individual Features with Positive Test R²")
    print("=" * 80)
    print()

    all_features_df = pd.concat([df_velocity_clean, df_acceleration_clean,
                                 df_trajectory_clean, df_force_clean,
                                 df_coordination_clean], axis=1)

    positive_features = []

    for col in all_features_df.columns:
        X_single = all_features_df[[col]].values

        # Skip if too many NaN
        valid_pct = (~np.isnan(X_single)).mean()
        if valid_pct < 0.5:
            continue

        train_mean, train_std, test_mean, test_std = within_player_cv(
            X_single, angle_target, participant_ids, n_splits=5
        )

        if not np.isnan(test_mean) and test_mean > 0:
            positive_features.append({
                'Feature': col,
                'Train R²': train_mean,
                'Test R²': test_mean,
                'Gap': train_mean - test_mean,
            })

    if positive_features:
        positive_df = pd.DataFrame(positive_features).sort_values('Test R²', ascending=False)
        print(f"Found {len(positive_features)} features with positive Test R²:")
        print(positive_df.to_string(index=False))
    else:
        print("No individual features found with positive Test R²")

    # ==========================================================================
    # TEST 5: Test specific velocity windows more granularly
    # ==========================================================================
    print()
    print("=" * 80)
    print("TEST 5: Granular Velocity Windows Around Release (Frames 140-170)")
    print("=" * 80)
    print()

    # Test many velocity windows to find optimal
    best_window_results = []

    key_kps = ['right_wrist_z', 'right_elbow_z', 'right_knee_z']

    for kp_name in key_kps:
        for start in range(140, 161, 5):
            for window_size in [5, 10, 15, 20]:
                end = start + window_size
                if end > 180:
                    continue

                # Extract this specific velocity for all samples
                velocities = []
                for i in range(len(X)):
                    vel = compute_velocity(X[i], kp, kp_name, start, end)
                    velocities.append(vel)

                X_vel = np.array(velocities).reshape(-1, 1)

                valid_pct = (~np.isnan(X_vel)).mean()
                if valid_pct < 0.5:
                    continue

                train_mean, train_std, test_mean, test_std = within_player_cv(
                    X_vel, angle_target, participant_ids, n_splits=5
                )

                if not np.isnan(test_mean):
                    best_window_results.append({
                        'Keypoint': kp_name,
                        'Start': start,
                        'End': end,
                        'Window': window_size,
                        'Train R²': train_mean,
                        'Test R²': test_mean,
                    })

    if best_window_results:
        window_df = pd.DataFrame(best_window_results).sort_values('Test R²', ascending=False)
        print("Top 15 velocity windows by Test R²:")
        print(window_df.head(15).to_string(index=False))
        print()
        print("Bottom 5 velocity windows:")
        print(window_df.tail(5).to_string(index=False))

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    results_df = pd.DataFrame(results)
    print("All Feature Groups Summary:")
    print(results_df.to_string(index=False))
    print()

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'multi_frame_physics_results.csv', index=False)

    if positive_features:
        positive_df.to_csv(OUTPUT_DIR / 'positive_multi_frame_features.csv', index=False)

    if best_window_results:
        window_df.to_csv(OUTPUT_DIR / 'velocity_window_analysis.csv', index=False)

    print(f"Results saved to {OUTPUT_DIR}")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    best_test_r2 = max([r['Test R²'] for r in results if not np.isnan(r['Test R²'])])
    print(f"Best Test R² achieved: {best_test_r2:.4f}")
    print()

    if best_test_r2 < 0:
        print("RESULT: Multi-frame physics features also show NEGATIVE Test R²")
        print()
        print("This suggests:")
        print("1. The motion patterns vary too much shot-to-shot within a player")
        print("2. Even velocity/acceleration don't capture consistent physics")
        print("3. The variance in how each shot is made exceeds the signal")
        print()
        print("The problem may not be WHICH features, but that shot-to-shot")
        print("variation within a player is larger than the signal we can extract.")
    else:
        print(f"RESULT: Found positive Test R² of {best_test_r2:.4f}")
        print("There may be some signal in multi-frame physics features.")


if __name__ == "__main__":
    main()
