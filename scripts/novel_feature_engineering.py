"""
Novel Feature Engineering - Pushing the Boundaries

This script implements advanced biomechanical and signal processing features
that haven't been tried before in this competition.

Key insight from analysis:
- Within-player CV (0.008-0.009) matches LB closely
- Test set has same 5 players as training
- Current best LB: 0.008305, target: 0.007 (need 15%+ improvement)
- Per-player models are correct approach

Novel feature categories:
1. Kinetic chain features (proximal-to-distal sequencing)
2. Movement smoothness (spectral arc length, jerk)
3. Player-normalized features (z-scores within player)
4. Inter-joint coordination (cross-correlations)
5. Release mechanics (estimated ball trajectory)
6. Temporal dynamics (phase transitions, timing)
7. Energy transfer features
8. Shot consistency features (distance from player mean)
9. Balance/stability features
10. Velocity profile shape (PCA on velocity curves)
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import lightgbm as lgb

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

# Load target scalers
TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_target_ranges():
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }


def smooth_signal(x, window=5):
    """Apply moving average smoothing."""
    if len(x) < window:
        return x
    return uniform_filter1d(x, size=window, mode='nearest')


def compute_velocity(pos, dt=1/60):
    """Compute velocity from position using central differences."""
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def compute_acceleration(vel, dt=1/60):
    """Compute acceleration from velocity."""
    return compute_velocity(vel, dt)


def compute_jerk(acc, dt=1/60):
    """Compute jerk (derivative of acceleration)."""
    return compute_velocity(acc, dt)


# =============================================================================
# NOVEL FEATURE EXTRACTORS
# =============================================================================

def extract_kinetic_chain_features(timeseries, keypoint_index):
    """
    Kinetic chain features: proximal-to-distal sequencing

    In basketball shooting, power transfers from:
    hip -> trunk -> shoulder -> elbow -> wrist -> fingers

    Features:
    - Timing of peak velocity for each joint
    - Sequence order correctness
    - Time delays between joints
    """
    features = {}

    # Key joints in kinetic chain (proximal to distal)
    chain_joints = ["mid_hip", "right_shoulder", "right_elbow", "right_wrist"]

    # Get z-velocities (vertical component most relevant for shooting)
    peak_times = {}
    peak_vels = {}

    for joint in chain_joints:
        if joint not in keypoint_index:
            continue

        z_idx = keypoint_index[joint].get("z")
        if z_idx is None:
            continue

        z_pos = timeseries[:, z_idx]
        z_pos_smooth = smooth_signal(z_pos, window=5)
        z_vel = compute_velocity(z_pos_smooth)

        # Find peak velocity in release window (frames 100-180)
        window = z_vel[100:180]
        if len(window) > 0:
            peak_idx = np.argmax(np.abs(window)) + 100
            peak_times[joint] = peak_idx
            peak_vels[joint] = z_vel[peak_idx]

    # Compute timing features
    if len(peak_times) >= 2:
        joints = list(peak_times.keys())
        times = [peak_times[j] for j in joints]

        # Check if sequence is correct (proximal before distal)
        correct_order = all(times[i] <= times[i+1] for i in range(len(times)-1))
        features["kinetic_chain_correct"] = float(correct_order)

        # Time delays
        for i, j1 in enumerate(joints[:-1]):
            j2 = joints[i+1]
            delay = peak_times[j2] - peak_times[j1]
            features[f"kinetic_delay_{j1}_to_{j2}"] = delay

        # Total chain time
        features["kinetic_chain_duration"] = times[-1] - times[0]

        # Peak velocity ratio (distal should be higher)
        if "right_wrist" in peak_vels and "right_shoulder" in peak_vels:
            if peak_vels["right_shoulder"] != 0:
                features["kinetic_vel_ratio_wrist_shoulder"] = peak_vels["right_wrist"] / peak_vels["right_shoulder"]

    return features


def extract_smoothness_features(timeseries, keypoint_index):
    """
    Movement smoothness features based on jerk minimization principle.

    Smoother movements are associated with more skilled/consistent performance.

    Features:
    - Normalized jerk (lower = smoother)
    - Spectral arc length (lower = smoother)
    - Number of velocity peaks (fewer = smoother)
    """
    features = {}

    key_joints = ["right_wrist", "right_elbow"]

    for joint in key_joints:
        if joint not in keypoint_index:
            continue

        for coord in ["x", "y", "z"]:
            idx = keypoint_index[joint].get(coord)
            if idx is None:
                continue

            pos = timeseries[:, idx]
            pos_smooth = smooth_signal(pos, window=3)

            # Focus on release phase
            pos_phase = pos_smooth[100:180]
            if len(pos_phase) < 10:
                continue

            vel = compute_velocity(pos_phase)
            acc = compute_acceleration(vel)
            jerk = compute_jerk(acc)

            # Normalized jerk (dimensionless)
            duration = len(pos_phase) / 60.0
            movement_range = np.ptp(pos_phase)
            if movement_range > 0.001 and duration > 0:
                norm_jerk = np.sqrt(np.mean(jerk**2)) * (duration**2.5) / movement_range
                features[f"smoothness_jerk_{joint}_{coord}"] = norm_jerk

            # Number of velocity peaks
            vel_peaks, _ = signal.find_peaks(np.abs(vel), height=np.std(vel)*0.5)
            features[f"smoothness_n_peaks_{joint}_{coord}"] = len(vel_peaks)

            # Velocity variability
            features[f"smoothness_vel_cv_{joint}_{coord}"] = np.std(vel) / (np.mean(np.abs(vel)) + 1e-6)

    return features


def extract_coordination_features(timeseries, keypoint_index):
    """
    Inter-joint coordination features.

    How well do different joints move together?
    - Cross-correlations between joint velocities
    - Synchrony metrics
    """
    features = {}

    joint_pairs = [
        ("right_wrist", "right_elbow"),
        ("right_elbow", "right_shoulder"),
        ("right_wrist", "left_wrist"),  # Guide hand
        ("right_knee", "right_hip"),
    ]

    for j1, j2 in joint_pairs:
        if j1 not in keypoint_index or j2 not in keypoint_index:
            continue

        for coord in ["z"]:  # Focus on z (vertical)
            idx1 = keypoint_index[j1].get(coord)
            idx2 = keypoint_index[j2].get(coord)
            if idx1 is None or idx2 is None:
                continue

            pos1 = smooth_signal(timeseries[100:180, idx1], window=3)
            pos2 = smooth_signal(timeseries[100:180, idx2], window=3)

            vel1 = compute_velocity(pos1)
            vel2 = compute_velocity(pos2)

            # Cross-correlation at lag 0
            if np.std(vel1) > 0 and np.std(vel2) > 0:
                corr = np.corrcoef(vel1, vel2)[0, 1]
                if not np.isnan(corr):
                    features[f"coord_corr_{j1}_{j2}_{coord}"] = corr

            # Maximum cross-correlation and lag
            try:
                xcorr = np.correlate(vel1 - np.mean(vel1), vel2 - np.mean(vel2), mode='full')
                xcorr = xcorr / (np.std(vel1) * np.std(vel2) * len(vel1))
                max_idx = np.argmax(xcorr)
                lag = max_idx - (len(vel1) - 1)
                features[f"coord_lag_{j1}_{j2}_{coord}"] = lag
                features[f"coord_max_xcorr_{j1}_{j2}_{coord}"] = xcorr[max_idx]
            except:
                pass

    return features


def extract_release_mechanics_features(timeseries, keypoint_index):
    """
    Release mechanics: estimated ball trajectory from hand position.

    Features:
    - Release height (wrist z at release)
    - Release velocity components
    - Release angle (estimated from velocity direction)
    - Set point height
    """
    features = {}

    if "right_wrist" not in keypoint_index:
        return features

    # Get wrist trajectory
    wrist_x = timeseries[:, keypoint_index["right_wrist"]["x"]]
    wrist_y = timeseries[:, keypoint_index["right_wrist"]["y"]]
    wrist_z = timeseries[:, keypoint_index["right_wrist"]["z"]]

    # Smooth
    wrist_x = smooth_signal(wrist_x, window=5)
    wrist_y = smooth_signal(wrist_y, window=5)
    wrist_z = smooth_signal(wrist_z, window=5)

    # Velocities
    vx = compute_velocity(wrist_x)
    vy = compute_velocity(wrist_y)
    vz = compute_velocity(wrist_z)

    # Find release frame (peak upward velocity of wrist, frames 130-170)
    window = vz[130:170]
    if len(window) > 0:
        release_frame = np.argmax(window) + 130

        features["release_frame"] = release_frame
        features["release_height"] = wrist_z[release_frame]
        features["release_vx"] = vx[release_frame]
        features["release_vy"] = vy[release_frame]
        features["release_vz"] = vz[release_frame]

        # Release speed
        speed = np.sqrt(vx[release_frame]**2 + vy[release_frame]**2 + vz[release_frame]**2)
        features["release_speed"] = speed

        # Release angle (from horizontal)
        if speed > 0:
            release_angle = np.arctan2(vz[release_frame], np.sqrt(vx[release_frame]**2 + vy[release_frame]**2))
            features["release_angle_rad"] = release_angle
            features["release_angle_deg"] = np.degrees(release_angle)

        # Set point (highest wrist z before release)
        pre_release = wrist_z[100:release_frame]
        if len(pre_release) > 0:
            set_point_frame = np.argmax(pre_release) + 100
            features["set_point_height"] = wrist_z[set_point_frame]
            features["set_point_frame"] = set_point_frame
            features["release_drop"] = features["set_point_height"] - features["release_height"]

    return features


def extract_balance_features(timeseries, keypoint_index):
    """
    Balance and stability features.

    - Center of mass approximation
    - Base of support (foot positions)
    - Lateral sway
    """
    features = {}

    # Approximate center of mass from mid_hip
    if "mid_hip" in keypoint_index:
        hip_x = timeseries[:, keypoint_index["mid_hip"]["x"]]
        hip_y = timeseries[:, keypoint_index["mid_hip"]["y"]]
        hip_z = timeseries[:, keypoint_index["mid_hip"]["z"]]

        # Stability during setup (frames 0-100)
        features["setup_hip_x_std"] = np.std(hip_x[0:100])
        features["setup_hip_y_std"] = np.std(hip_y[0:100])
        features["setup_hip_z_std"] = np.std(hip_z[0:100])

        # Stability during release (frames 130-160)
        features["release_hip_x_std"] = np.std(hip_x[130:160])
        features["release_hip_y_std"] = np.std(hip_y[130:160])

        # Forward lean during release
        features["forward_lean"] = hip_y[150] - hip_y[100]

        # Lateral sway
        features["lateral_sway_range"] = np.ptp(hip_x[100:180])

    # Foot positions
    for foot in ["left_ankle", "right_ankle"]:
        if foot in keypoint_index:
            z = timeseries[:, keypoint_index[foot]["z"]]
            # Foot lift during shot
            features[f"{foot}_lift"] = np.max(z[100:180]) - np.mean(z[0:50])

    return features


def extract_velocity_profile_features(timeseries, keypoint_index):
    """
    Velocity profile shape features using PCA.

    The shape of the velocity curve contains information about shot technique.
    """
    features = {}

    if "right_wrist" not in keypoint_index:
        return features

    wrist_z = timeseries[:, keypoint_index["right_wrist"]["z"]]
    wrist_z = smooth_signal(wrist_z, window=5)
    vel_z = compute_velocity(wrist_z)

    # Extract velocity profile in release window
    vel_profile = vel_z[100:180]

    if len(vel_profile) >= 10:
        # Normalize profile
        if np.std(vel_profile) > 0:
            vel_norm = (vel_profile - np.mean(vel_profile)) / np.std(vel_profile)

            # Simple shape descriptors
            features["vel_profile_skew"] = stats.skew(vel_profile)
            features["vel_profile_kurtosis"] = stats.kurtosis(vel_profile)

            # Time to peak velocity (as fraction of window)
            peak_idx = np.argmax(vel_profile)
            features["vel_profile_peak_timing"] = peak_idx / len(vel_profile)

            # Asymmetry (area before vs after peak)
            area_before = np.sum(vel_profile[:peak_idx])
            area_after = np.sum(vel_profile[peak_idx:])
            if area_before + area_after != 0:
                features["vel_profile_asymmetry"] = (area_after - area_before) / (area_before + area_after + 1e-6)

    return features


def extract_shot_consistency_features(timeseries, keypoint_index, player_mean_trajectory=None):
    """
    Shot consistency features: how similar is this shot to the player's average?

    Note: This requires computing player means first.
    """
    features = {}

    if player_mean_trajectory is None:
        return features

    # Compute distance from player mean for key joints
    for joint in ["right_wrist", "right_elbow"]:
        if joint not in keypoint_index:
            continue

        for coord in ["x", "y", "z"]:
            idx = keypoint_index[joint].get(coord)
            if idx is None:
                continue

            current = timeseries[:, idx]
            mean = player_mean_trajectory.get(f"{joint}_{coord}")

            if mean is not None and len(mean) == len(current):
                # Mean absolute deviation from player mean
                diff = current - mean
                features[f"consistency_{joint}_{coord}_mad"] = np.mean(np.abs(diff))
                features[f"consistency_{joint}_{coord}_corr"] = np.corrcoef(current, mean)[0, 1] if np.std(current) > 0 and np.std(mean) > 0 else 0

    return features


def extract_energy_features(timeseries, keypoint_index):
    """
    Energy transfer features.

    - Kinetic energy at different phases
    - Energy flow from lower to upper body
    """
    features = {}

    joints = ["right_wrist", "right_elbow", "right_shoulder", "right_knee", "mid_hip"]

    for joint in joints:
        if joint not in keypoint_index:
            continue

        # Get 3D velocity
        vx = compute_velocity(smooth_signal(timeseries[:, keypoint_index[joint].get("x", 0)], 5))
        vy = compute_velocity(smooth_signal(timeseries[:, keypoint_index[joint].get("y", 0)], 5))
        vz = compute_velocity(smooth_signal(timeseries[:, keypoint_index[joint].get("z", 0)], 5))

        # Speed squared (proportional to kinetic energy)
        ke = vx**2 + vy**2 + vz**2

        # Peak energy in different phases
        features[f"energy_{joint}_setup"] = np.max(ke[0:100])
        features[f"energy_{joint}_release"] = np.max(ke[130:170])

        # Energy ratio
        if features[f"energy_{joint}_setup"] > 0:
            features[f"energy_{joint}_ratio"] = features[f"energy_{joint}_release"] / features[f"energy_{joint}_setup"]

    return features


def extract_all_novel_features(timeseries, keypoint_index):
    """Extract all novel features."""
    features = {}

    features.update(extract_kinetic_chain_features(timeseries, keypoint_index))
    features.update(extract_smoothness_features(timeseries, keypoint_index))
    features.update(extract_coordination_features(timeseries, keypoint_index))
    features.update(extract_release_mechanics_features(timeseries, keypoint_index))
    features.update(extract_balance_features(timeseries, keypoint_index))
    features.update(extract_velocity_profile_features(timeseries, keypoint_index))
    features.update(extract_energy_features(timeseries, keypoint_index))

    return features


def load_data_with_novel_features():
    """Load data and extract novel features."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    # Build keypoint index
    keypoint_index = {}
    for idx, col in enumerate(keypoint_cols):
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            keypoint_name = parts[0]
            coord = parts[1]
            if keypoint_name not in keypoint_index:
                keypoint_index[keypoint_name] = {}
            keypoint_index[keypoint_name][coord] = idx

    print(f"Processing {len(train_df)} shots...")

    all_novel_features = []
    all_targets = []
    all_pids = []
    all_timeseries = []

    for idx, row in train_df.iterrows():
        ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            ts[:, i] = parse_array_string(row[col])

        # Extract novel features
        novel_feats = extract_all_novel_features(ts, keypoint_index)
        all_novel_features.append(novel_feats)
        all_timeseries.append(ts)

        all_targets.append([row["angle"], row["depth"], row["left_right"]])
        all_pids.append(row["participant_id"])

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    # Convert to arrays
    feature_names = sorted(all_novel_features[0].keys())
    X_novel = np.array([[f.get(n, 0.0) for n in feature_names] for f in all_novel_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    print(f"Novel features shape: {X_novel.shape}")

    return X_novel, y, pids, feature_names, all_timeseries, keypoint_index


def compute_cv_score(X, y, pids, model_class, params):
    """Compute within-player CV score."""
    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(pids))

    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        player_indices = np.where(pid_mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train = y_player[train_idx]

            for target_idx, target in enumerate(TARGETS):
                model = model_class(**params)
                model.fit(X_train, y_train[:, target_idx])
                all_preds[player_indices[val_idx], target_idx] = model.predict(X_val)

    # Compute scaled MSE
    results = {}
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((all_preds[:, target_idx] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        results[target] = scaled_mse
        total_mse += scaled_mse

    results["total"] = total_mse / 3
    return results


def main():
    print("=" * 80)
    print("NOVEL FEATURE ENGINEERING")
    print("=" * 80)

    # Load data with novel features
    X_novel, y, pids, novel_feature_names, timeseries_list, keypoint_index = load_data_with_novel_features()

    # Also load hybrid features for comparison
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    print("\nExtracting baseline features...")
    all_baseline_features = []
    for idx, row in train_df.iterrows():
        ts = timeseries_list[idx]
        hybrid = extract_hybrid_features(ts, row["participant_id"], smooth=False)
        advanced = extract_advanced_features(ts, row["participant_id"])
        combined = {**hybrid, **advanced}
        all_baseline_features.append(combined)

    baseline_names = sorted(all_baseline_features[0].keys())
    X_baseline = np.array([[f.get(n, 0.0) for n in baseline_names] for f in all_baseline_features], dtype=np.float32)

    print(f"Baseline features: {X_baseline.shape}")
    print(f"Novel features: {X_novel.shape}")

    results = []

    # =============================================================================
    # TEST 1: Novel features only
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: NOVEL FEATURES ONLY")
    print("=" * 80)

    for alpha in [10, 100, 1000]:
        scores = compute_cv_score(X_novel, y, pids, Ridge, {"alpha": alpha, "random_state": 42})
        print(f"  Ridge alpha={alpha}: total={scores['total']:.6f}, angle={scores['angle']:.6f}, depth={scores['depth']:.6f}, lr={scores['left_right']:.6f}")
        results.append({"test": "novel_only", "config": f"ridge_alpha{alpha}", **scores})

    scores = compute_cv_score(X_novel, y, pids, lgb.LGBMRegressor,
                              {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05,
                               "reg_alpha": 0.5, "reg_lambda": 0.5, "random_state": 42, "verbose": -1, "n_jobs": -1})
    print(f"  LightGBM: total={scores['total']:.6f}")
    results.append({"test": "novel_only", "config": "lgb", **scores})

    # =============================================================================
    # TEST 2: Baseline features only (for comparison)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 2: BASELINE FEATURES ONLY")
    print("=" * 80)

    scores = compute_cv_score(X_baseline, y, pids, Ridge, {"alpha": 100, "random_state": 42})
    print(f"  Ridge alpha=100: total={scores['total']:.6f}")
    results.append({"test": "baseline_only", "config": "ridge_alpha100", **scores})

    scores = compute_cv_score(X_baseline, y, pids, lgb.LGBMRegressor,
                              {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05,
                               "reg_alpha": 0.5, "reg_lambda": 0.5, "random_state": 42, "verbose": -1, "n_jobs": -1})
    print(f"  LightGBM: total={scores['total']:.6f}")
    results.append({"test": "baseline_only", "config": "lgb", **scores})

    # =============================================================================
    # TEST 3: Combined features (baseline + novel)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 3: COMBINED FEATURES (BASELINE + NOVEL)")
    print("=" * 80)

    X_combined = np.hstack([X_baseline, X_novel])
    print(f"Combined features: {X_combined.shape}")

    for alpha in [10, 100, 1000]:
        scores = compute_cv_score(X_combined, y, pids, Ridge, {"alpha": alpha, "random_state": 42})
        print(f"  Ridge alpha={alpha}: total={scores['total']:.6f}")
        results.append({"test": "combined", "config": f"ridge_alpha{alpha}", **scores})

    scores = compute_cv_score(X_combined, y, pids, lgb.LGBMRegressor,
                              {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05,
                               "reg_alpha": 0.5, "reg_lambda": 0.5, "random_state": 42, "verbose": -1, "n_jobs": -1})
    print(f"  LightGBM: total={scores['total']:.6f}")
    results.append({"test": "combined", "config": "lgb", **scores})

    # =============================================================================
    # TEST 4: Feature importance analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 4: NOVEL FEATURE IMPORTANCE")
    print("=" * 80)

    # Compute correlations with targets
    scaler = StandardScaler()
    X_novel_scaled = scaler.fit_transform(X_novel)
    X_novel_scaled = np.nan_to_num(X_novel_scaled, nan=0.0)

    print("\nTop novel features by correlation with each target:")
    for target_idx, target in enumerate(TARGETS):
        correlations = []
        for i, name in enumerate(novel_feature_names):
            try:
                r, p = stats.pearsonr(X_novel_scaled[:, i], y[:, target_idx])
                if not np.isnan(r):
                    correlations.append((name, r, p))
            except:
                pass

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  {target.upper()}:")
        for name, r, p in correlations[:10]:
            print(f"    {name:50s}: r={r:+.4f}, p={p:.4f}")

    # =============================================================================
    # Save Results
    # =============================================================================
    output_file = OUTPUT_DIR / "novel_feature_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save feature names
    features_file = OUTPUT_DIR / "novel_feature_names.txt"
    with open(features_file, "w") as f:
        for name in sorted(novel_feature_names):
            f.write(f"{name}\n")
    print(f"Feature names saved to: {features_file}")

    # =============================================================================
    # Summary
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df_sorted = df.sort_values("total")
    print("\nBest configurations:")
    for _, row in df_sorted.head(5).iterrows():
        print(f"  {row['test']}/{row['config']}: total={row['total']:.6f}")

    # Check improvement from novel features
    baseline_best = df[df['test'] == 'baseline_only']['total'].min()
    combined_best = df[df['test'] == 'combined']['total'].min()

    if combined_best < baseline_best:
        improvement = (baseline_best - combined_best) / baseline_best * 100
        print(f"\nNovel features improved score by {improvement:.2f}%")
    else:
        print(f"\nNovel features did not improve score")


if __name__ == "__main__":
    main()
