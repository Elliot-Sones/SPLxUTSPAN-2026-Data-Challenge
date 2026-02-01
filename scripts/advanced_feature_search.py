"""
Advanced Feature Search - Pushing for 0.007

Key finding from novel features:
- release_vx has r=0.61 correlation with angle (very strong!)
- Combined features with LightGBM achieved 0.009112

This script explores:
1. More release mechanics features (the release_vx discovery suggests this is key)
2. Player-normalized features (z-scores within each player)
3. Frame-by-frame velocity analysis
4. Finger/hand orientation features
5. Trajectory shape features
6. Acceleration profiles
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

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
    if len(x) < window:
        return x
    return uniform_filter1d(x, size=window, mode='nearest')


def compute_velocity(pos, dt=1/60):
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt
    return vel


def compute_acceleration(vel, dt=1/60):
    return compute_velocity(vel, dt)


# =============================================================================
# RELEASE MECHANICS - DEEP DIVE
# =============================================================================

def extract_release_deep_features(ts, kp_idx):
    """
    Deep analysis of release mechanics since release_vx showed r=0.61 with angle.
    """
    features = {}

    joints = ["right_wrist", "right_elbow", "right_shoulder", "right_knee", "mid_hip",
              "left_wrist", "neck", "nose"]

    # Get all joint positions and velocities
    positions = {}
    velocities = {}
    accelerations = {}

    for joint in joints:
        if joint not in kp_idx:
            continue
        for coord in ["x", "y", "z"]:
            idx = kp_idx[joint].get(coord)
            if idx is None:
                continue
            key = f"{joint}_{coord}"
            pos = smooth_signal(ts[:, idx], window=5)
            positions[key] = pos
            velocities[key] = compute_velocity(pos)
            accelerations[key] = compute_acceleration(velocities[key])

    # Find release frame using wrist z velocity
    if "right_wrist_z" in velocities:
        vz = velocities["right_wrist_z"]
        # Peak upward velocity in frames 130-170
        window = vz[130:170]
        if len(window) > 0:
            release_frame = np.argmax(window) + 130
        else:
            release_frame = 150
    else:
        release_frame = 150

    features["release_frame"] = release_frame

    # Extract features at release and surrounding frames
    for offset in [-10, -5, 0, 5, 10]:
        frame = release_frame + offset
        if frame < 0 or frame >= 240:
            continue

        suffix = f"_at_r{offset:+d}" if offset != 0 else "_at_release"

        for joint in ["right_wrist", "right_elbow", "right_shoulder"]:
            for coord in ["x", "y", "z"]:
                key = f"{joint}_{coord}"
                if key in positions:
                    features[f"{key}{suffix}"] = positions[key][frame]
                if key in velocities:
                    features[f"{key}_vel{suffix}"] = velocities[key][frame]
                if key in accelerations:
                    features[f"{key}_acc{suffix}"] = accelerations[key][frame]

    # 3D velocity and speed at release
    for joint in ["right_wrist", "right_elbow"]:
        vx = velocities.get(f"{joint}_x", np.zeros(240))[release_frame]
        vy = velocities.get(f"{joint}_y", np.zeros(240))[release_frame]
        vz = velocities.get(f"{joint}_z", np.zeros(240))[release_frame]

        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        features[f"{joint}_speed_at_release"] = speed

        # Velocity direction angles
        if speed > 0:
            features[f"{joint}_vel_elevation"] = np.arcsin(vz / speed)  # Angle from horizontal
            features[f"{joint}_vel_azimuth"] = np.arctan2(vy, vx)  # Horizontal direction

    # Velocity ratios between joints at release
    wrist_speed = features.get("right_wrist_speed_at_release", 0)
    elbow_speed = features.get("right_elbow_speed_at_release", 0)
    if elbow_speed > 0:
        features["wrist_elbow_speed_ratio"] = wrist_speed / elbow_speed

    # Relative positions at release
    if "right_wrist_x_at_release" in features and "right_shoulder_x_at_release" in features:
        features["wrist_shoulder_x_diff"] = features["right_wrist_x_at_release"] - features["right_shoulder_x_at_release"]
        features["wrist_shoulder_y_diff"] = features["right_wrist_y_at_release"] - features.get("right_shoulder_y_at_release", 0)
        features["wrist_shoulder_z_diff"] = features["right_wrist_z_at_release"] - features.get("right_shoulder_z_at_release", 0)

    # Arm extension at release
    if all(k in positions for k in ["right_wrist_x", "right_elbow_x", "right_shoulder_x"]):
        for coord in ["x", "y", "z"]:
            wrist = positions.get(f"right_wrist_{coord}", np.zeros(240))[release_frame]
            elbow = positions.get(f"right_elbow_{coord}", np.zeros(240))[release_frame]
            shoulder = positions.get(f"right_shoulder_{coord}", np.zeros(240))[release_frame]

            # Upper arm vector
            upper_arm = np.array([elbow - shoulder for elbow, shoulder in
                                  [(positions.get(f"right_elbow_{c}", np.zeros(240))[release_frame],
                                    positions.get(f"right_shoulder_{c}", np.zeros(240))[release_frame])
                                   for c in ["x", "y", "z"]]])

            # Forearm vector
            forearm = np.array([wrist - elbow for wrist, elbow in
                               [(positions.get(f"right_wrist_{c}", np.zeros(240))[release_frame],
                                 positions.get(f"right_elbow_{c}", np.zeros(240))[release_frame])
                                for c in ["x", "y", "z"]]])

            # Arm extension angle (180 = fully extended)
            if np.linalg.norm(upper_arm) > 0 and np.linalg.norm(forearm) > 0:
                cos_angle = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))
                cos_angle = np.clip(cos_angle, -1, 1)
                features["arm_extension_angle"] = np.degrees(np.arccos(cos_angle))

    return features


def extract_velocity_profile_deep(ts, kp_idx):
    """
    Deep analysis of velocity profiles.
    """
    features = {}

    if "right_wrist" not in kp_idx:
        return features

    for coord in ["x", "y", "z"]:
        idx = kp_idx["right_wrist"].get(coord)
        if idx is None:
            continue

        pos = smooth_signal(ts[:, idx], window=5)
        vel = compute_velocity(pos)
        acc = compute_acceleration(vel)

        # Velocity in release window
        vel_window = vel[100:180]
        acc_window = acc[100:180]

        # Peak detection
        peaks, props = signal.find_peaks(vel_window, height=0)
        troughs, _ = signal.find_peaks(-vel_window)

        features[f"wrist_{coord}_n_vel_peaks"] = len(peaks)
        features[f"wrist_{coord}_n_vel_troughs"] = len(troughs)

        if len(peaks) > 0:
            features[f"wrist_{coord}_max_vel"] = np.max(vel_window[peaks])
            features[f"wrist_{coord}_peak_frame"] = peaks[np.argmax(vel_window[peaks])] + 100

        # Velocity integral (displacement in window)
        features[f"wrist_{coord}_displacement"] = np.sum(vel_window) / 60  # Convert to position units

        # Acceleration statistics
        features[f"wrist_{coord}_max_acc"] = np.max(acc_window)
        features[f"wrist_{coord}_min_acc"] = np.min(acc_window)
        features[f"wrist_{coord}_acc_range"] = np.ptp(acc_window)

        # Velocity curvature (second derivative of velocity = jerk)
        jerk = compute_velocity(acc_window)
        features[f"wrist_{coord}_max_jerk"] = np.max(np.abs(jerk))
        features[f"wrist_{coord}_mean_jerk"] = np.mean(np.abs(jerk))

    return features


def extract_hand_orientation_features(ts, kp_idx):
    """
    Hand orientation features using finger positions.
    """
    features = {}

    # Check if we have finger keypoints
    finger_joints = ["right_first_finger_mcp", "right_third_finger_mcp", "right_first_finger_pip"]

    has_fingers = all(j in kp_idx for j in finger_joints)
    if not has_fingers:
        return features

    # Use finger positions to estimate hand plane orientation
    release_frame = 150  # Approximate

    # Get finger positions at release
    finger_pos = {}
    for joint in finger_joints:
        for coord in ["x", "y", "z"]:
            idx = kp_idx[joint].get(coord)
            if idx is not None:
                finger_pos[f"{joint}_{coord}"] = ts[release_frame, idx]

    # Compute hand plane normal using cross product
    if len(finger_pos) >= 9:  # Need 3 points x 3 coords
        try:
            p1 = np.array([finger_pos[f"right_first_finger_mcp_{c}"] for c in ["x", "y", "z"]])
            p2 = np.array([finger_pos[f"right_third_finger_mcp_{c}"] for c in ["x", "y", "z"]])
            p3 = np.array([finger_pos[f"right_first_finger_pip_{c}"] for c in ["x", "y", "z"]])

            v1 = p2 - p1
            v2 = p3 - p1

            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)

                features["hand_normal_x"] = normal[0]
                features["hand_normal_y"] = normal[1]
                features["hand_normal_z"] = normal[2]

                # Angle of hand plane from vertical
                features["hand_tilt_from_vertical"] = np.degrees(np.arccos(np.abs(normal[2])))
        except:
            pass

    return features


def extract_player_normalized_features(ts, kp_idx, player_stats):
    """
    Features normalized by player's typical values.

    player_stats: dict of {feature_name: (mean, std)} for this player
    """
    features = {}

    if player_stats is None:
        return features

    # Key positions at release
    release_frame = 150

    for joint in ["right_wrist", "right_elbow", "mid_hip"]:
        if joint not in kp_idx:
            continue

        for coord in ["x", "y", "z"]:
            idx = kp_idx[joint].get(coord)
            if idx is None:
                continue

            key = f"{joint}_{coord}_release"
            value = ts[release_frame, idx]

            if key in player_stats:
                mean, std = player_stats[key]
                if std > 0:
                    features[f"{key}_zscore"] = (value - mean) / std

    return features


def extract_trajectory_shape_features(ts, kp_idx):
    """
    Shape of the shooting trajectory.
    """
    features = {}

    if "right_wrist" not in kp_idx:
        return features

    # Wrist trajectory in release phase
    frames = range(100, 180)

    wrist_x = smooth_signal(ts[:, kp_idx["right_wrist"]["x"]], 5)[100:180]
    wrist_y = smooth_signal(ts[:, kp_idx["right_wrist"]["y"]], 5)[100:180]
    wrist_z = smooth_signal(ts[:, kp_idx["right_wrist"]["z"]], 5)[100:180]

    # Trajectory length
    dx = np.diff(wrist_x)
    dy = np.diff(wrist_y)
    dz = np.diff(wrist_z)
    path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    features["trajectory_length"] = path_length

    # Straightness (direct distance / path length)
    direct_dist = np.sqrt((wrist_x[-1]-wrist_x[0])**2 +
                         (wrist_y[-1]-wrist_y[0])**2 +
                         (wrist_z[-1]-wrist_z[0])**2)
    if path_length > 0:
        features["trajectory_straightness"] = direct_dist / path_length

    # Curvature (mean absolute second derivative)
    d2x = np.diff(wrist_x, n=2)
    d2y = np.diff(wrist_y, n=2)
    d2z = np.diff(wrist_z, n=2)
    curvature = np.mean(np.sqrt(d2x**2 + d2y**2 + d2z**2))
    features["trajectory_curvature"] = curvature

    # Vertical vs horizontal displacement ratio
    vertical_disp = wrist_z[-1] - wrist_z[0]
    horizontal_disp = np.sqrt((wrist_x[-1]-wrist_x[0])**2 + (wrist_y[-1]-wrist_y[0])**2)
    if horizontal_disp > 0:
        features["trajectory_vert_horiz_ratio"] = vertical_disp / horizontal_disp

    return features


def extract_all_advanced_features(ts, kp_idx, player_stats=None):
    """Extract all advanced features."""
    features = {}

    features.update(extract_release_deep_features(ts, kp_idx))
    features.update(extract_velocity_profile_deep(ts, kp_idx))
    features.update(extract_hand_orientation_features(ts, kp_idx))
    features.update(extract_trajectory_shape_features(ts, kp_idx))
    features.update(extract_player_normalized_features(ts, kp_idx, player_stats))

    return features


def load_data_and_extract_features():
    """Load data and extract all feature sets."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    # Build keypoint index
    kp_idx = {}
    for idx, col in enumerate(keypoint_cols):
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            kp_name = parts[0]
            coord = parts[1]
            if kp_name not in kp_idx:
                kp_idx[kp_name] = {}
            kp_idx[kp_name][coord] = idx

    print(f"Processing {len(train_df)} shots...")

    # First pass: collect player statistics for normalization
    print("  First pass: computing player statistics...")
    player_data = {pid: [] for pid in train_df["participant_id"].unique()}

    for idx, row in train_df.iterrows():
        ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            ts[:, i] = parse_array_string(row[col])

        # Collect key values for this player
        release_frame = 150
        values = {}
        for joint in ["right_wrist", "right_elbow", "mid_hip"]:
            if joint not in kp_idx:
                continue
            for coord in ["x", "y", "z"]:
                idx_col = kp_idx[joint].get(coord)
                if idx_col is not None:
                    key = f"{joint}_{coord}_release"
                    values[key] = ts[release_frame, idx_col]

        player_data[row["participant_id"]].append(values)

    # Compute player stats (mean, std)
    player_stats = {}
    for pid, data_list in player_data.items():
        if len(data_list) == 0:
            continue
        player_stats[pid] = {}
        keys = data_list[0].keys()
        for key in keys:
            values = [d[key] for d in data_list if key in d]
            if len(values) > 1:
                player_stats[pid][key] = (np.mean(values), np.std(values))

    # Second pass: extract all features
    print("  Second pass: extracting features...")

    all_advanced_features = []
    all_targets = []
    all_pids = []
    all_timeseries = []

    for idx, row in train_df.iterrows():
        ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            ts[:, i] = parse_array_string(row[col])

        pid = row["participant_id"]
        adv_feats = extract_all_advanced_features(ts, kp_idx, player_stats.get(pid))

        all_advanced_features.append(adv_feats)
        all_timeseries.append(ts)
        all_targets.append([row["angle"], row["depth"], row["left_right"]])
        all_pids.append(pid)

        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(train_df)}")

    # Convert to arrays
    feature_names = sorted(all_advanced_features[0].keys())
    X_adv = np.array([[f.get(n, 0.0) for n in feature_names] for f in all_advanced_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    print(f"Advanced features shape: {X_adv.shape}")

    return X_adv, y, pids, feature_names, all_timeseries, kp_idx


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
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train = y_player[train_idx]

            for target_idx, target in enumerate(TARGETS):
                model = model_class(**params)
                model.fit(X_train, y_train[:, target_idx])
                all_preds[player_indices[val_idx], target_idx] = model.predict(X_val)

    results = {}
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((all_preds[:, target_idx] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        results[target] = scaled_mse
        total_mse += scaled_mse

    results["total"] = total_mse / 3
    return results, all_preds


def main():
    print("=" * 80)
    print("ADVANCED FEATURE SEARCH - PUSHING FOR 0.007")
    print("=" * 80)

    # Load data
    X_adv, y, pids, adv_names, timeseries_list, kp_idx = load_data_and_extract_features()

    # Also load baseline and novel features
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    print("\nExtracting baseline features...")
    all_baseline = []
    for idx, row in train_df.iterrows():
        ts = timeseries_list[idx]
        hybrid = extract_hybrid_features(ts, row["participant_id"], smooth=False)
        advanced = extract_advanced_features(ts, row["participant_id"])
        combined = {**hybrid, **advanced}
        all_baseline.append(combined)

    baseline_names = sorted(all_baseline[0].keys())
    X_base = np.array([[f.get(n, 0.0) for n in baseline_names] for f in all_baseline], dtype=np.float32)

    print(f"Baseline features: {X_base.shape}")
    print(f"New advanced features: {X_adv.shape}")

    results = []

    # =============================================================================
    # TEST 1: New advanced features only
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: NEW ADVANCED FEATURES ONLY")
    print("=" * 80)

    scores, _ = compute_cv_score(X_adv, y, pids, Ridge, {"alpha": 100, "random_state": 42})
    print(f"  Ridge alpha=100: total={scores['total']:.6f}")
    results.append({"test": "advanced_only", "config": "ridge", **scores})

    scores, _ = compute_cv_score(X_adv, y, pids, lgb.LGBMRegressor,
                               {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05,
                                "random_state": 42, "verbose": -1, "n_jobs": -1})
    print(f"  LightGBM: total={scores['total']:.6f}")
    results.append({"test": "advanced_only", "config": "lgb", **scores})

    # =============================================================================
    # TEST 2: All features combined
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 2: ALL FEATURES COMBINED")
    print("=" * 80)

    X_all = np.hstack([X_base, X_adv])
    print(f"All features: {X_all.shape}")

    for alpha in [10, 100, 1000]:
        scores, _ = compute_cv_score(X_all, y, pids, Ridge, {"alpha": alpha, "random_state": 42})
        print(f"  Ridge alpha={alpha}: total={scores['total']:.6f}")
        results.append({"test": "all_combined", "config": f"ridge_a{alpha}", **scores})

    scores, _ = compute_cv_score(X_all, y, pids, lgb.LGBMRegressor,
                               {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05,
                                "random_state": 42, "verbose": -1, "n_jobs": -1})
    print(f"  LightGBM: total={scores['total']:.6f}")
    results.append({"test": "all_combined", "config": "lgb", **scores})

    # =============================================================================
    # TEST 3: Feature importance and correlation analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 3: TOP FEATURES BY CORRELATION")
    print("=" * 80)

    all_names = baseline_names + adv_names
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_all_scaled = np.nan_to_num(X_all_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nTop 20 features by correlation with ANGLE:")
    correlations = []
    for i, name in enumerate(all_names):
        try:
            r, p = stats.pearsonr(X_all_scaled[:, i], y[:, 0])
            if not np.isnan(r):
                correlations.append((name, r, p, i))
        except:
            pass

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r, p, idx in correlations[:20]:
        print(f"  {name:55s}: r={r:+.4f}")

    # =============================================================================
    # TEST 4: Optuna optimization with feature selection
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 4: OPTUNA OPTIMIZATION (300 trials)")
    print("=" * 80)

    # Clean NaN/inf from X_all before Optuna
    X_all_clean = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    def objective(trial):
        # Feature selection
        k = trial.suggest_int("k_features", 20, min(300, len(all_names)))

        # Select top k features by f_regression
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X_all_clean, y[:, 0])  # Use angle as target for selection

        # Model params
        model_type = trial.suggest_categorical("model", ["ridge", "lgb"])

        if model_type == "ridge":
            alpha = trial.suggest_float("alpha", 1, 10000, log=True)
            model_class = Ridge
            params = {"alpha": alpha, "random_state": 42}
        else:
            model_class = lgb.LGBMRegressor
            params = {
                "n_estimators": trial.suggest_int("n_est", 50, 300),
                "num_leaves": trial.suggest_int("leaves", 3, 31),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "reg_alpha": trial.suggest_float("reg_a", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_l", 0.001, 10, log=True),
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }

        # Reconstruct full feature matrix with selection
        X_sel = selector.transform(X_all_clean)
        scores, _ = compute_cv_score(X_sel, y, pids, model_class, params)
        return scores["total"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=300, show_progress_bar=True)

    print(f"\nBest trial:")
    print(f"  Score: {study.best_value:.6f}")
    print(f"  Params: {study.best_params}")

    results.append({"test": "optuna", "config": str(study.best_params),
                    "total": study.best_value, "angle": 0, "depth": 0, "left_right": 0})

    # =============================================================================
    # Save Results
    # =============================================================================
    output_file = OUTPUT_DIR / "advanced_feature_search_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save feature names
    features_file = OUTPUT_DIR / "advanced_feature_names.txt"
    with open(features_file, "w") as f:
        for name in sorted(adv_names):
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
        print(f"  {row['test']}/{row['config'][:50]}: total={row['total']:.6f}")

    best_score = df_sorted.iloc[0]["total"]
    print(f"\n*** BEST SCORE: {best_score:.6f} ***")
    print(f"Current LB best: 0.008305")
    print(f"Target: 0.007000")

    if best_score < 0.008305:
        print(f"\nPotential improvement: {(0.008305 - best_score) / 0.008305 * 100:.2f}%")
    else:
        print(f"\nNo improvement over current best LB")


if __name__ == "__main__":
    main()
