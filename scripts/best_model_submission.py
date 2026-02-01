"""
Best Model Submission - Comprehensive Feature Engineering

Based on all research findings:
1. CV of 0.008810 achieved with LightGBM + combined features
2. Strong correlations found (r=0.78 for phase_load features)
3. Per-player models are appropriate (within-player CV matches LB)
4. Combined baseline + advanced features work best

This script creates a submission using the best configuration found.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

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


def extract_release_features(ts, kp_idx):
    """Extract release mechanics features (found to have strong correlations)."""
    features = {}

    joints = ["right_wrist", "right_elbow", "right_shoulder"]
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

    # Find release frame
    if "right_wrist_z" in velocities:
        vz = velocities["right_wrist_z"]
        window = vz[130:170]
        if len(window) > 0:
            release_frame = np.argmax(window) + 130
        else:
            release_frame = 150
    else:
        release_frame = 150

    features["release_frame"] = release_frame

    # Features at release and offsets
    for offset in [-10, -5, 0, 5, 10]:
        frame = release_frame + offset
        if frame < 0 or frame >= 240:
            continue

        suffix = f"_at_r{offset:+d}" if offset != 0 else "_at_release"

        for joint in joints:
            for coord in ["x", "y", "z"]:
                key = f"{joint}_{coord}"
                if key in positions:
                    features[f"{key}{suffix}"] = positions[key][frame]
                if key in velocities:
                    features[f"{key}_vel{suffix}"] = velocities[key][frame]
                if key in accelerations:
                    features[f"{key}_acc{suffix}"] = accelerations[key][frame]

    # Speed and direction at release
    for joint in joints:
        vx = velocities.get(f"{joint}_x", np.zeros(240))[release_frame]
        vy = velocities.get(f"{joint}_y", np.zeros(240))[release_frame]
        vz = velocities.get(f"{joint}_z", np.zeros(240))[release_frame]

        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        features[f"{joint}_speed_at_release"] = speed

        if speed > 0:
            features[f"{joint}_vel_elevation"] = np.arcsin(np.clip(vz / speed, -1, 1))
            features[f"{joint}_vel_azimuth"] = np.arctan2(vy, vx)

    return features


def load_all_data():
    """Load train and test data with all features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

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

    def process_df(df, is_train):
        print(f"  Processing {len(df)} shots...")
        all_features = []
        targets = []
        pids = []
        ids = []

        for idx, row in df.iterrows():
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            # Extract all feature types
            hybrid = extract_hybrid_features(ts, row["participant_id"], smooth=False)
            advanced = extract_advanced_features(ts, row["participant_id"])
            release = extract_release_features(ts, kp_idx)

            combined = {**hybrid, **advanced, **release}
            all_features.append(combined)

            pids.append(row["participant_id"])
            ids.append(row["id"])
            if is_train:
                targets.append([row["angle"], row["depth"], row["left_right"]])

            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{len(df)}")

        return all_features, targets, pids, ids

    print("Loading training data...")
    train_features, train_targets, train_pids, train_ids = process_df(train_df, True)

    print("Loading test data...")
    test_features, _, test_pids, test_ids = process_df(test_df, False)

    # Convert to arrays
    feature_names = sorted(train_features[0].keys())
    X_train = np.array([[f.get(n, 0.0) for n in feature_names] for f in train_features], dtype=np.float32)
    X_test = np.array([[f.get(n, 0.0) for n in feature_names] for f in test_features], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)

    # Clean NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features: {X_train.shape[1]}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "train_ids": train_ids,
        "X_test": X_test,
        "test_pids": np.array(test_pids),
        "test_ids": test_ids,
        "feature_names": feature_names,
    }


def train_and_evaluate(data):
    """Train models and compute CV score."""
    X = data["X_train"]
    y = data["y_train"]
    pids = data["train_pids"]
    ranges = get_target_ranges()

    unique_pids = sorted(np.unique(pids))

    # Storage
    all_models = {}  # {(pid, target): model}
    all_scalers = {}  # {pid: scaler}
    oof_preds = np.zeros_like(y)

    print("\nTraining per-player models...")

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        player_indices = np.where(pid_mask)[0]
        n_samples = len(X_player)

        print(f"\n  Player {pid} ({n_samples} samples)")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        all_scalers[pid] = scaler

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]
            fold_preds = np.zeros(n_samples)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train = y_target[train_idx]

                # LightGBM with optimized params
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    num_leaves=10,
                    learning_rate=0.05,
                    reg_alpha=0.5,
                    reg_lambda=0.5,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                fold_preds[val_idx] = model.predict(X_val)

            oof_preds[player_indices, target_idx] = fold_preds

            # Train final model on all player data
            final_model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=10,
                learning_rate=0.05,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            final_model.fit(X_scaled, y_target)
            all_models[(pid, target)] = final_model

    # Compute CV scores
    print("\n" + "=" * 60)
    print("CV RESULTS")
    print("=" * 60)

    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((oof_preds[:, target_idx] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        print(f"  {target}: scaled_MSE = {scaled_mse:.6f}")
        total_mse += scaled_mse

    avg_mse = total_mse / 3
    print(f"\n  TOTAL: {avg_mse:.6f}")

    return all_models, all_scalers, avg_mse


def generate_predictions(data, models, scalers):
    """Generate test predictions."""
    X_test = data["X_test"]
    test_pids = data["test_pids"]
    test_ids = data["test_ids"]

    predictions = np.zeros((len(X_test), 3))

    for i, (x, pid) in enumerate(zip(X_test, test_pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))

        for target_idx, target in enumerate(TARGETS):
            model = models[(pid, target)]
            predictions[i, target_idx] = model.predict(x_scaled)[0]

    return predictions, test_ids


def create_submission(test_ids, predictions, cv_score):
    """Create submission CSV."""
    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    # Scale predictions
    scaled_preds = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    # Create DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    # Save
    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    print(f"\n" + "=" * 60)
    print(f"SUBMISSION {next_num} CREATED")
    print("=" * 60)
    print(f"  File: {filepath}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"\n  Prediction statistics:")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"    {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}")

    return filepath, next_num


def main():
    print("=" * 80)
    print("BEST MODEL SUBMISSION")
    print("Based on comprehensive feature engineering research")
    print("=" * 80)

    # Load data
    data = load_all_data()

    # Train and evaluate
    models, scalers, cv_score = train_and_evaluate(data)

    # Generate predictions
    predictions, test_ids = generate_predictions(data, models, scalers)

    # Create submission
    filepath, sub_num = create_submission(test_ids, predictions, cv_score)

    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Submission number: {sub_num}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  Current best LB: 0.008305")
    print(f"  Target: 0.007000")

    if cv_score < 0.008305:
        print(f"\n  Potential improvement: {(0.008305 - cv_score) / 0.008305 * 100:.2f}%")
    else:
        print(f"\n  CV is {(cv_score - 0.008305) / 0.008305 * 100:.2f}% worse than current LB")

    return filepath


if __name__ == "__main__":
    main()
