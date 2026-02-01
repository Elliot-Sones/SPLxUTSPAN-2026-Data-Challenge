"""
Depth and Left-Right Optimization

Key insight: Angle is already at 0.0074 (close to target).
Depth (0.0094) and Left-Right (0.0097) are the bottleneck.

This script focuses specifically on improving depth and left_right predictions
using targeted feature engineering and model optimization.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

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


def extract_depth_specific_features(ts, kp_idx):
    """
    Features specifically designed for depth prediction.

    From research: depth correlated with smoothness, forward_lean, set_point_height
    Try:
    - Guide hand features (left hand position/velocity)
    - Forward/backward balance
    - Shot arc features
    - Finger positions (release details)
    """
    features = {}

    # Guide hand (left hand) features - may affect depth control
    for joint in ["left_wrist", "left_elbow", "left_shoulder"]:
        if joint not in kp_idx:
            continue

        for coord in ["x", "y", "z"]:
            idx = kp_idx[joint].get(coord)
            if idx is None:
                continue

            data = smooth_signal(ts[:, idx], window=5)

            # Statistics in different phases
            features[f"depth_{joint}_{coord}_setup_mean"] = np.mean(data[0:100])
            features[f"depth_{joint}_{coord}_release_mean"] = np.mean(data[130:170])
            features[f"depth_{joint}_{coord}_release_std"] = np.std(data[130:170])

            # Guide hand retraction (pull back during shot)
            features[f"depth_{joint}_{coord}_retraction"] = data[100] - data[150]

    # Balance features (forward/backward motion)
    if "mid_hip" in kp_idx:
        hip_y = smooth_signal(ts[:, kp_idx["mid_hip"]["y"]], window=5)
        hip_z = smooth_signal(ts[:, kp_idx["mid_hip"]["z"]], window=5)

        # Forward lean during shot
        features["depth_forward_motion_release"] = hip_y[160] - hip_y[130]
        features["depth_hip_drop"] = hip_z[130] - hip_z[160]

    # Knee bend (affects power and depth control)
    for knee in ["right_knee", "left_knee"]:
        if knee not in kp_idx:
            continue

        z = smooth_signal(ts[:, kp_idx[knee]["z"]], window=5)

        features[f"depth_{knee}_bend_setup"] = z[50] - z[100]  # Bend down
        features[f"depth_{knee}_extension"] = z[150] - z[100]   # Push up
        features[f"depth_{knee}_max_bend"] = np.min(z[80:130])

    # Shooting arc features
    if "right_wrist" in kp_idx:
        wrist_z = smooth_signal(ts[:, kp_idx["right_wrist"]["z"]], window=5)
        wrist_y = smooth_signal(ts[:, kp_idx["right_wrist"]["y"]], window=5)

        # Arc height
        max_height_frame = np.argmax(wrist_z[100:180]) + 100
        features["depth_arc_peak_height"] = wrist_z[max_height_frame]
        features["depth_arc_peak_frame"] = max_height_frame

        # Release point relative to body
        if "mid_hip" in kp_idx:
            hip_y_at_release = ts[150, kp_idx["mid_hip"]["y"]]
            features["depth_wrist_forward_of_hip"] = wrist_y[150] - hip_y_at_release

    return features


def extract_leftright_specific_features(ts, kp_idx):
    """
    Features specifically designed for left-right prediction.

    From research: left-right correlated with hip stability, lateral movement
    Try:
    - Lateral hip/shoulder movement
    - Foot placement asymmetry
    - Head/eye alignment
    - Elbow flare
    """
    features = {}

    # Lateral stability (x-axis movement)
    for joint in ["mid_hip", "neck", "nose", "right_shoulder", "left_shoulder"]:
        if joint not in kp_idx:
            continue

        x = smooth_signal(ts[:, kp_idx[joint]["x"]], window=5)

        # Lateral drift during shot
        features[f"lr_{joint}_x_drift"] = x[160] - x[100]
        features[f"lr_{joint}_x_range"] = np.ptp(x[100:170])
        features[f"lr_{joint}_x_std"] = np.std(x[100:170])

    # Shoulder alignment
    if "right_shoulder" in kp_idx and "left_shoulder" in kp_idx:
        r_shoulder_x = ts[:, kp_idx["right_shoulder"]["x"]]
        l_shoulder_x = ts[:, kp_idx["left_shoulder"]["x"]]

        # Shoulder tilt
        features["lr_shoulder_diff_setup"] = r_shoulder_x[50] - l_shoulder_x[50]
        features["lr_shoulder_diff_release"] = r_shoulder_x[150] - l_shoulder_x[150]
        features["lr_shoulder_rotation"] = (r_shoulder_x[150] - l_shoulder_x[150]) - (r_shoulder_x[50] - l_shoulder_x[50])

    # Elbow flare (common cause of left-right error)
    if "right_elbow" in kp_idx and "right_shoulder" in kp_idx:
        elbow_x = ts[:, kp_idx["right_elbow"]["x"]]
        shoulder_x = ts[:, kp_idx["right_shoulder"]["x"]]

        # Elbow position relative to shoulder
        elbow_flare = elbow_x - shoulder_x
        features["lr_elbow_flare_setup"] = elbow_flare[100]
        features["lr_elbow_flare_release"] = elbow_flare[150]
        features["lr_elbow_flare_change"] = elbow_flare[150] - elbow_flare[100]

    # Foot placement
    for foot in ["right_ankle", "left_ankle"]:
        if foot not in kp_idx:
            continue

        x = ts[:, kp_idx[foot]["x"]]
        features[f"lr_{foot}_x_mean"] = np.mean(x[0:100])  # Setup position

    # Head alignment (looking at rim)
    if "nose" in kp_idx:
        nose_x = ts[:, kp_idx["nose"]["x"]]
        features["lr_head_alignment"] = np.mean(nose_x[100:170])

    return features


def load_data_with_targeted_features():
    """Load data with depth and left-right specific features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

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

    all_features = []
    targets = []
    pids = []

    for idx, row in train_df.iterrows():
        ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            ts[:, i] = parse_array_string(row[col])

        # Extract all feature types
        hybrid = extract_hybrid_features(ts, row["participant_id"], smooth=False)
        advanced = extract_advanced_features(ts, row["participant_id"])
        depth_feats = extract_depth_specific_features(ts, kp_idx)
        lr_feats = extract_leftright_specific_features(ts, kp_idx)

        combined = {**hybrid, **advanced, **depth_feats, **lr_feats}
        all_features.append(combined)

        targets.append([row["angle"], row["depth"], row["left_right"]])
        pids.append(row["participant_id"])

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(n, 0.0) for n in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    pids = np.array(pids)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features: {X.shape}")

    return X, y, pids, feature_names


def compute_cv_score_single_target(X, y, pids, model_class, params, target_idx, target_name):
    """Compute CV score for a single target."""
    range_sq = TARGET_SCALERS[target_name].data_range_[0] ** 2
    unique_pids = sorted(np.unique(pids))

    all_preds = np.zeros(len(y))

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask, target_idx]
        player_indices = np.where(pid_mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_scaled):
            model = model_class(**params)
            model.fit(X_scaled[train_idx], y_player[train_idx])
            all_preds[player_indices[val_idx]] = model.predict(X_scaled[val_idx])

    raw_mse = np.mean((all_preds - y[:, target_idx]) ** 2)
    scaled_mse = raw_mse / range_sq
    return scaled_mse


def main():
    print("=" * 80)
    print("DEPTH AND LEFT-RIGHT OPTIMIZATION")
    print("=" * 80)

    X, y, pids, feature_names = load_data_with_targeted_features()

    # Analyze feature correlations for depth and left_right
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
        print(f"\n{target_name.upper()}:")
        correlations = []
        for i, name in enumerate(feature_names):
            try:
                r, p = stats.pearsonr(X_scaled[:, i], y[:, target_idx])
                if not np.isnan(r):
                    correlations.append((name, r, p))
            except:
                pass

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"  Top 15 features:")
        for name, r, p in correlations[:15]:
            print(f"    {name:55s}: r={r:+.4f}")

    # Test different models for DEPTH
    print("\n" + "=" * 80)
    print("MODEL COMPARISON FOR DEPTH")
    print("=" * 80)

    models = {
        "Ridge_100": (Ridge, {"alpha": 100, "random_state": 42}),
        "Ridge_1000": (Ridge, {"alpha": 1000, "random_state": 42}),
        "Lasso_1": (Lasso, {"alpha": 1.0, "random_state": 42, "max_iter": 5000}),
        "ElasticNet": (ElasticNet, {"alpha": 0.5, "l1_ratio": 0.5, "random_state": 42, "max_iter": 5000}),
        "LightGBM": (lgb.LGBMRegressor, {"n_estimators": 100, "num_leaves": 10, "learning_rate": 0.05, "random_state": 42, "verbose": -1, "n_jobs": -1}),
        "XGBoost": (xgb.XGBRegressor, {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "random_state": 42, "verbosity": 0, "n_jobs": -1}),
        "CatBoost": (CatBoostRegressor, {"iterations": 100, "depth": 4, "learning_rate": 0.05, "random_state": 42, "verbose": False}),
        "RandomForest": (RandomForestRegressor, {"n_estimators": 100, "max_depth": 5, "random_state": 42, "n_jobs": -1}),
        "GradientBoosting": (GradientBoostingRegressor, {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}),
    }

    results = []

    for model_name, (model_class, params) in models.items():
        depth_score = compute_cv_score_single_target(X, y, pids, model_class, params, 1, "depth")
        lr_score = compute_cv_score_single_target(X, y, pids, model_class, params, 2, "left_right")
        angle_score = compute_cv_score_single_target(X, y, pids, model_class, params, 0, "angle")
        total = (angle_score + depth_score + lr_score) / 3

        print(f"  {model_name:20s}: angle={angle_score:.6f}, depth={depth_score:.6f}, lr={lr_score:.6f}, total={total:.6f}")

        results.append({
            "model": model_name,
            "angle": angle_score,
            "depth": depth_score,
            "left_right": lr_score,
            "total": total,
        })

    # Optuna optimization for DEPTH
    print("\n" + "=" * 80)
    print("OPTUNA OPTIMIZATION FOR DEPTH (200 trials)")
    print("=" * 80)

    def depth_objective(trial):
        model_type = trial.suggest_categorical("model", ["lgb", "xgb", "catboost"])

        if model_type == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_est", 50, 300),
                "num_leaves": trial.suggest_int("leaves", 3, 31),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "reg_alpha": trial.suggest_float("reg_a", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_l", 0.001, 10, log=True),
                "random_state": 42, "verbose": -1, "n_jobs": -1,
            }
            model_class = lgb.LGBMRegressor
        elif model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_est", 50, 300),
                "max_depth": trial.suggest_int("depth", 2, 8),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "reg_alpha": trial.suggest_float("reg_a", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_l", 0.001, 10, log=True),
                "random_state": 42, "verbosity": 0, "n_jobs": -1,
            }
            model_class = xgb.XGBRegressor
        else:
            params = {
                "iterations": trial.suggest_int("iter", 50, 300),
                "depth": trial.suggest_int("depth", 2, 8),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2", 0.1, 10, log=True),
                "random_state": 42, "verbose": False,
            }
            model_class = CatBoostRegressor

        return compute_cv_score_single_target(X, y, pids, model_class, params, 1, "depth")

    study_depth = optuna.create_study(direction="minimize")
    study_depth.optimize(depth_objective, n_trials=200, show_progress_bar=True)

    print(f"\nBest depth score: {study_depth.best_value:.6f}")
    print(f"Best params: {study_depth.best_params}")

    # Optuna optimization for LEFT_RIGHT
    print("\n" + "=" * 80)
    print("OPTUNA OPTIMIZATION FOR LEFT_RIGHT (200 trials)")
    print("=" * 80)

    def lr_objective(trial):
        model_type = trial.suggest_categorical("model", ["lgb", "xgb", "catboost"])

        if model_type == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_est", 50, 300),
                "num_leaves": trial.suggest_int("leaves", 3, 31),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "reg_alpha": trial.suggest_float("reg_a", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_l", 0.001, 10, log=True),
                "random_state": 42, "verbose": -1, "n_jobs": -1,
            }
            model_class = lgb.LGBMRegressor
        elif model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_est", 50, 300),
                "max_depth": trial.suggest_int("depth", 2, 8),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "reg_alpha": trial.suggest_float("reg_a", 0.001, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_l", 0.001, 10, log=True),
                "random_state": 42, "verbosity": 0, "n_jobs": -1,
            }
            model_class = xgb.XGBRegressor
        else:
            params = {
                "iterations": trial.suggest_int("iter", 50, 300),
                "depth": trial.suggest_int("depth", 2, 8),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2", 0.1, 10, log=True),
                "random_state": 42, "verbose": False,
            }
            model_class = CatBoostRegressor

        return compute_cv_score_single_target(X, y, pids, model_class, params, 2, "left_right")

    study_lr = optuna.create_study(direction="minimize")
    study_lr.optimize(lr_objective, n_trials=200, show_progress_bar=True)

    print(f"\nBest left_right score: {study_lr.best_value:.6f}")
    print(f"Best params: {study_lr.best_params}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "depth_lr_optimization.csv", index=False)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_row = df.loc[df["total"].idxmin()]
    print(f"\nBest overall model: {best_row['model']}")
    print(f"  Total: {best_row['total']:.6f}")
    print(f"  Angle: {best_row['angle']:.6f}")
    print(f"  Depth: {best_row['depth']:.6f}")
    print(f"  Left-Right: {best_row['left_right']:.6f}")

    print(f"\nOptuna best depth: {study_depth.best_value:.6f}")
    print(f"Optuna best left_right: {study_lr.best_value:.6f}")

    # Check if we can reach 0.007
    best_angle = df["angle"].min()
    best_depth = study_depth.best_value
    best_lr = study_lr.best_value
    optimal_total = (best_angle + best_depth + best_lr) / 3

    print(f"\n*** OPTIMAL POTENTIAL ***")
    print(f"Best angle: {best_angle:.6f}")
    print(f"Best depth: {best_depth:.6f}")
    print(f"Best left_right: {best_lr:.6f}")
    print(f"Optimal total: {optimal_total:.6f}")
    print(f"\nTarget: 0.007000")
    print(f"Gap: {optimal_total - 0.007:.6f}")


if __name__ == "__main__":
    main()
