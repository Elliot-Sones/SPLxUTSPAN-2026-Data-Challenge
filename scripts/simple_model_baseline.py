"""
Simple Model Baseline Test

Goal: Test if simpler models generalize better.

Tests:
1. Ridge regression with varying alpha (1, 10, 100, 1000)
2. Simple feature sets:
   - Only wrist positions at key frames (f140, f150, f160)
   - Only elbow positions
   - Combinations
3. Compare within-player CV vs LOPO for each
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
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


def load_raw_data():
    """Load raw training data as timeseries (for simple feature extraction)."""
    print("Loading raw training data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    print(f"Processing {len(train_df)} training shots...")

    # Build keypoint index
    keypoint_index = {}
    for idx, col in enumerate(keypoint_cols):
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            keypoint_name = parts[0]
            coord = parts[1]  # x, y, or z
            if keypoint_name not in keypoint_index:
                keypoint_index[keypoint_name] = {}
            keypoint_index[keypoint_name][coord] = idx

    all_timeseries = []
    all_targets = []
    all_pids = []

    for idx, row in train_df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        all_timeseries.append(timeseries)
        all_targets.append([row["angle"], row["depth"], row["left_right"]])
        all_pids.append(row["participant_id"])

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    print(f"Loaded {len(all_timeseries)} shots")
    return all_timeseries, y, pids, keypoint_index, keypoint_cols


def extract_frame_features(timeseries_list, keypoint_index, keypoints, frames):
    """
    Extract features at specific frames for specific keypoints.

    Args:
        timeseries_list: List of (240, n_keypoints) arrays
        keypoint_index: Dict mapping keypoint -> {x: idx, y: idx, z: idx}
        keypoints: List of keypoint names to extract
        frames: List of frame indices to extract

    Returns:
        X: (n_samples, n_features) array
        feature_names: List of feature names
    """
    feature_names = []
    for kp in keypoints:
        for frame in frames:
            for coord in ["x", "y", "z"]:
                feature_names.append(f"{kp}_{coord}_f{frame}")

    n_samples = len(timeseries_list)
    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, ts in enumerate(timeseries_list):
        feat_idx = 0
        for kp in keypoints:
            if kp not in keypoint_index:
                for _ in frames:
                    for _ in ["x", "y", "z"]:
                        X[i, feat_idx] = 0.0
                        feat_idx += 1
                continue

            for frame in frames:
                for coord in ["x", "y", "z"]:
                    if coord in keypoint_index[kp]:
                        col_idx = keypoint_index[kp][coord]
                        X[i, feat_idx] = ts[frame, col_idx] if frame < len(ts) else 0.0
                    else:
                        X[i, feat_idx] = 0.0
                    feat_idx += 1

    return X, feature_names


def extract_stat_features(timeseries_list, keypoint_index, keypoints):
    """
    Extract statistical features (mean, std, min, max, range) for keypoints.
    """
    feature_names = []
    for kp in keypoints:
        for coord in ["x", "y", "z"]:
            for stat in ["mean", "std", "min", "max", "range"]:
                feature_names.append(f"{kp}_{coord}_{stat}")

    n_samples = len(timeseries_list)
    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, ts in enumerate(timeseries_list):
        feat_idx = 0
        for kp in keypoints:
            if kp not in keypoint_index:
                for _ in ["x", "y", "z"]:
                    for _ in ["mean", "std", "min", "max", "range"]:
                        X[i, feat_idx] = 0.0
                        feat_idx += 1
                continue

            for coord in ["x", "y", "z"]:
                if coord in keypoint_index[kp]:
                    col_idx = keypoint_index[kp][coord]
                    data = ts[:, col_idx]
                    valid = data[~np.isnan(data)]
                    if len(valid) > 0:
                        X[i, feat_idx] = np.mean(valid)
                        X[i, feat_idx + 1] = np.std(valid)
                        X[i, feat_idx + 2] = np.min(valid)
                        X[i, feat_idx + 3] = np.max(valid)
                        X[i, feat_idx + 4] = np.max(valid) - np.min(valid)
                    else:
                        X[i, feat_idx:feat_idx + 5] = 0.0
                else:
                    X[i, feat_idx:feat_idx + 5] = 0.0
                feat_idx += 5

    return X, feature_names


def compute_cv_scores(X, y, pids, model_class, model_params, cv_type="within_player"):
    """
    Compute CV scores for a model configuration.

    Returns: dict with mean scaled MSE per target and overall
    """
    ranges = get_target_ranges()

    if cv_type == "within_player":
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
                y_train, y_val = y_player[train_idx], y_player[val_idx]

                for target_idx, target in enumerate(TARGETS):
                    model = model_class(**model_params)
                    model.fit(X_train, y_train[:, target_idx])
                    all_preds[player_indices[val_idx], target_idx] = model.predict(X_val)

    elif cv_type == "lopo":
        all_preds = np.zeros_like(y)
        logo = LeaveOneGroupOut()

        for train_idx, val_idx in logo.split(X, y, pids):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

            for target_idx, target in enumerate(TARGETS):
                model = model_class(**model_params)
                model.fit(X_train_scaled, y_train[:, target_idx])
                all_preds[val_idx, target_idx] = model.predict(X_val_scaled)

    # Compute scaled MSE
    results = {}
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((all_preds[:, target_idx] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        results[f"{target}_mse"] = scaled_mse
        total_mse += scaled_mse

    results["total_mse"] = total_mse / 3
    return results


def main():
    print("=" * 80)
    print("SIMPLE MODEL BASELINE TEST")
    print("=" * 80)

    # Load data
    timeseries_list, y, pids, keypoint_index, keypoint_cols = load_raw_data()

    results = []

    # =============================================================================
    # TEST 1: Ridge with different alpha values
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Ridge Regression with Different Alpha Values")
    print("=" * 80)

    # Use key frame features for this test
    key_keypoints = ["right_wrist", "right_elbow", "right_shoulder", "right_knee", "mid_hip"]
    key_frames = [140, 145, 150, 155, 160]

    X_frames, frame_names = extract_frame_features(timeseries_list, keypoint_index, key_keypoints, key_frames)
    X_stats, stat_names = extract_stat_features(timeseries_list, keypoint_index, key_keypoints)

    # Combine frame and stat features
    X_combined = np.hstack([X_frames, X_stats])
    combined_names = frame_names + stat_names
    print(f"\nCombined features: {X_combined.shape[1]}")

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    for alpha in alphas:
        print(f"\n  Alpha = {alpha}:")
        params = {"alpha": alpha, "random_state": 42}

        wp_scores = compute_cv_scores(X_combined, y, pids, Ridge, params, "within_player")
        lopo_scores = compute_cv_scores(X_combined, y, pids, Ridge, params, "lopo")

        print(f"    Within-player: {wp_scores['total_mse']:.6f}")
        print(f"    LOPO:          {lopo_scores['total_mse']:.6f}")
        print(f"    Gap:           {lopo_scores['total_mse'] - wp_scores['total_mse']:+.6f}")

        results.append({
            "test": "ridge_alpha",
            "config": f"alpha={alpha}",
            "n_features": X_combined.shape[1],
            "within_player_mse": wp_scores["total_mse"],
            "lopo_mse": lopo_scores["total_mse"],
            "angle_wp": wp_scores["angle_mse"],
            "depth_wp": wp_scores["depth_mse"],
            "lr_wp": wp_scores["left_right_mse"],
            "angle_lopo": lopo_scores["angle_mse"],
            "depth_lopo": lopo_scores["depth_mse"],
            "lr_lopo": lopo_scores["left_right_mse"],
        })

    # =============================================================================
    # TEST 2: Different Feature Sets
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Different Feature Sets (with Ridge alpha=100)")
    print("=" * 80)

    feature_sets = {
        "wrist_only": (["right_wrist"], [140, 145, 150, 155, 160]),
        "wrist_elbow": (["right_wrist", "right_elbow"], [140, 145, 150, 155, 160]),
        "upper_body": (["right_wrist", "right_elbow", "right_shoulder"], [140, 145, 150, 155, 160]),
        "full_arm_leg": (["right_wrist", "right_elbow", "right_shoulder", "right_knee", "mid_hip"], [140, 145, 150, 155, 160]),
        "wrist_many_frames": (["right_wrist"], list(range(100, 180, 5))),
        "left_guide_hand": (["left_wrist", "left_elbow"], [140, 145, 150, 155, 160]),
        "both_hands": (["right_wrist", "left_wrist"], [140, 145, 150, 155, 160]),
    }

    for set_name, (keypoints, frames) in feature_sets.items():
        X_f, _ = extract_frame_features(timeseries_list, keypoint_index, keypoints, frames)
        X_s, _ = extract_stat_features(timeseries_list, keypoint_index, keypoints)
        X_test = np.hstack([X_f, X_s])

        print(f"\n  {set_name} ({X_test.shape[1]} features):")

        params = {"alpha": 100.0, "random_state": 42}
        wp_scores = compute_cv_scores(X_test, y, pids, Ridge, params, "within_player")
        lopo_scores = compute_cv_scores(X_test, y, pids, Ridge, params, "lopo")

        print(f"    Within-player: {wp_scores['total_mse']:.6f}")
        print(f"    LOPO:          {lopo_scores['total_mse']:.6f}")

        results.append({
            "test": "feature_set",
            "config": set_name,
            "n_features": X_test.shape[1],
            "within_player_mse": wp_scores["total_mse"],
            "lopo_mse": lopo_scores["total_mse"],
            "angle_wp": wp_scores["angle_mse"],
            "depth_wp": wp_scores["depth_mse"],
            "lr_wp": wp_scores["left_right_mse"],
            "angle_lopo": lopo_scores["angle_mse"],
            "depth_lopo": lopo_scores["depth_mse"],
            "lr_lopo": lopo_scores["left_right_mse"],
        })

    # =============================================================================
    # TEST 3: Model Comparison (on same features)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Model Comparison (full_arm_leg features)")
    print("=" * 80)

    # Use full_arm_leg feature set
    kps = ["right_wrist", "right_elbow", "right_shoulder", "right_knee", "mid_hip"]
    frs = [140, 145, 150, 155, 160]
    X_f, _ = extract_frame_features(timeseries_list, keypoint_index, kps, frs)
    X_s, _ = extract_stat_features(timeseries_list, keypoint_index, kps)
    X_model_test = np.hstack([X_f, X_s])

    models_to_test = {
        "Ridge_1": (Ridge, {"alpha": 1.0, "random_state": 42}),
        "Ridge_100": (Ridge, {"alpha": 100.0, "random_state": 42}),
        "Ridge_1000": (Ridge, {"alpha": 1000.0, "random_state": 42}),
        "Lasso_0.1": (Lasso, {"alpha": 0.1, "random_state": 42, "max_iter": 5000}),
        "Lasso_1": (Lasso, {"alpha": 1.0, "random_state": 42, "max_iter": 5000}),
        "ElasticNet": (ElasticNet, {"alpha": 0.5, "l1_ratio": 0.5, "random_state": 42, "max_iter": 5000}),
        "LightGBM_conservative": (
            lgb.LGBMRegressor,
            {"n_estimators": 50, "num_leaves": 5, "learning_rate": 0.1, "reg_alpha": 1.0, "reg_lambda": 1.0, "random_state": 42, "verbose": -1, "n_jobs": -1}
        ),
        "LightGBM_standard": (
            lgb.LGBMRegressor,
            {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 0.5, "random_state": 42, "verbose": -1, "n_jobs": -1}
        ),
        "RandomForest": (
            RandomForestRegressor,
            {"n_estimators": 50, "max_depth": 5, "random_state": 42, "n_jobs": -1}
        ),
    }

    for model_name, (model_class, params) in models_to_test.items():
        print(f"\n  {model_name}:")

        wp_scores = compute_cv_scores(X_model_test, y, pids, model_class, params, "within_player")
        lopo_scores = compute_cv_scores(X_model_test, y, pids, model_class, params, "lopo")

        print(f"    Within-player: {wp_scores['total_mse']:.6f}")
        print(f"    LOPO:          {lopo_scores['total_mse']:.6f}")

        results.append({
            "test": "model_comparison",
            "config": model_name,
            "n_features": X_model_test.shape[1],
            "within_player_mse": wp_scores["total_mse"],
            "lopo_mse": lopo_scores["total_mse"],
            "angle_wp": wp_scores["angle_mse"],
            "depth_wp": wp_scores["depth_mse"],
            "lr_wp": wp_scores["left_right_mse"],
            "angle_lopo": lopo_scores["angle_mse"],
            "depth_lopo": lopo_scores["depth_mse"],
            "lr_lopo": lopo_scores["left_right_mse"],
        })

    # =============================================================================
    # Save Results
    # =============================================================================
    output_file = OUTPUT_DIR / "simple_model_baseline.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # =============================================================================
    # Summary
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nBest Within-Player Models:")
    df_sorted = df.sort_values("within_player_mse")
    for _, row in df_sorted.head(5).iterrows():
        print(f"  {row['test']}/{row['config']}: {row['within_player_mse']:.6f} (LOPO: {row['lopo_mse']:.6f})")

    print("\nBest LOPO Models (most generalizable):")
    df_sorted = df.sort_values("lopo_mse")
    for _, row in df_sorted.head(5).iterrows():
        print(f"  {row['test']}/{row['config']}: {row['lopo_mse']:.6f} (WP: {row['within_player_mse']:.6f})")

    print("\nSmallest WP-LOPO Gap (best generalization):")
    df["gap"] = df["lopo_mse"] - df["within_player_mse"]
    df_sorted = df.sort_values("gap")
    for _, row in df_sorted.head(5).iterrows():
        print(f"  {row['test']}/{row['config']}: gap={row['gap']:.6f} (WP: {row['within_player_mse']:.6f}, LOPO: {row['lopo_mse']:.6f})")

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("The large gap between within-player CV and LOPO CV suggests:")
    print("1. Player-specific patterns are important for good predictions")
    print("2. The test set likely contains samples from the same players as training")
    print("3. Per-player models are appropriate for this competition")


if __name__ == "__main__":
    main()
