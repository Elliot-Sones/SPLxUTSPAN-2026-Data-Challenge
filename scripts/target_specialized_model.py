"""
Target-Specialized Model

Based on research, each target has different optimal features:
- Angle: frame 153, lower body (ankle, knee z), R2=0.45
- Depth: frame 102, left hand positions, R2=0.08
- Left_right: frame 237, right hand fingers, R2=0.025

Strategy: Build ultra-focused models using ONLY the most relevant features
for each target to minimize variance.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}

KEYPOINT_MAP = {}


def init_keypoint_mapping(keypoint_cols):
    global KEYPOINT_MAP
    for i, col in enumerate(keypoint_cols):
        KEYPOINT_MAP[col] = i


def get_kp(series, name, frame=None):
    if name not in KEYPOINT_MAP:
        return None
    idx = KEYPOINT_MAP[name]
    if frame is not None:
        return series[frame, idx]
    return series[:, idx]


def smooth_signal(signal, window=5):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < window:
        return signal
    try:
        return savgol_filter(signal, window, 2)
    except:
        return signal


def compute_velocity(signal):
    return np.gradient(smooth_signal(signal))


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


# Target-specific feature configurations (ultra-focused)
ANGLE_CONFIG = {
    "key_frames": [150, 153, 155, 160],
    "joints": ["left_ankle_z", "right_ankle_z", "left_knee_z", "right_knee_z",
               "left_heel_z", "right_heel_z", "mid_hip_z"],
    "use_velocity": True,
}

DEPTH_CONFIG = {
    "key_frames": [100, 102, 105, 108],
    "joints": ["left_wrist_x", "left_wrist_y", "left_wrist_z",
               "left_elbow_x", "left_elbow_z", "right_wrist_z"],
    "use_velocity": True,
}

LEFT_RIGHT_CONFIG = {
    "key_frames": [230, 235, 237, 239],
    "joints": ["right_wrist_x", "right_wrist_y", "right_elbow_x",
               "right_shoulder_x", "mid_hip_x"],
    "use_velocity": True,
}


def extract_target_features(series, config, prefix=""):
    """Extract features for a specific target."""
    features = {}

    for joint in config["joints"]:
        data = get_kp(series, joint)
        if data is None:
            continue

        # Position at key frames
        for f in config["key_frames"]:
            if f < 240:
                features[f'{prefix}{joint}_f{f}'] = data[f]

        # Mean around key frames
        frame_window = range(
            max(0, min(config["key_frames"]) - 5),
            min(240, max(config["key_frames"]) + 5)
        )
        features[f'{prefix}{joint}_window_mean'] = np.nanmean(data[list(frame_window)])

        # Velocity features
        if config["use_velocity"]:
            vel = compute_velocity(data)
            for f in config["key_frames"]:
                if f < 239:
                    features[f'{prefix}{joint}_vel_f{f}'] = vel[f]
            features[f'{prefix}{joint}_vel_window'] = np.nanmean(vel[list(frame_window)])

    # Clean up NaN/Inf
    for k, v in list(features.items()):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                features[k] = 0.0

    return features


def load_and_extract_features():
    """Load data and extract target-specific features."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    init_keypoint_mapping(keypoint_cols)

    def process_df(df, desc="Processing"):
        # Separate feature sets for each target
        angle_features = []
        depth_features = []
        lr_features = []
        all_pids = []
        all_ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            angle_features.append(extract_target_features(ts, ANGLE_CONFIG, "angle_"))
            depth_features.append(extract_target_features(ts, DEPTH_CONFIG, "depth_"))
            lr_features.append(extract_target_features(ts, LEFT_RIGHT_CONFIG, "lr_"))
            all_pids.append(row['participant_id'])
            all_ids.append(row['id'])

        return {
            "angle": pd.DataFrame(angle_features),
            "depth": pd.DataFrame(depth_features),
            "left_right": pd.DataFrame(lr_features),
            "pids": np.array(all_pids),
            "ids": np.array(all_ids),
        }

    print("Processing training data...")
    train_data = process_df(train_df, "Train")
    train_targets = train_df[["angle", "depth", "left_right"]].values

    print("Processing test data...")
    test_data = process_df(test_df, "Test")

    return {
        "train": train_data,
        "train_targets": train_targets,
        "test": test_data,
    }


def train_specialized_model(data):
    """Train target-specialized models."""
    print("\n" + "=" * 60)
    print("TARGET-SPECIALIZED MODEL")
    print("=" * 60)

    train_targets = data["train_targets"]
    train_pids = data["train"]["pids"]
    test_pids = data["test"]["pids"]

    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(data["test"]["pids"]), 3))
    oof_preds = np.zeros_like(train_targets)

    target_configs = {
        "angle": ANGLE_CONFIG,
        "depth": DEPTH_CONFIG,
        "left_right": LEFT_RIGHT_CONFIG,
    }

    for t_idx, target in enumerate(TARGETS):
        # Get target-specific features
        X_train = np.nan_to_num(data["train"][target].values, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(data["test"][target].values, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"\n{target}:")
        print(f"  Features: {X_train.shape[1]}")

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_tr = X_train[train_mask]
            X_te = X_test[test_mask]
            y_tr = train_targets[train_mask, t_idx]
            player_indices = np.where(train_mask)[0]

            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for fold_train_idx, fold_val_idx in kf.split(X_tr_scaled):
                X_fold_tr = X_tr_scaled[fold_train_idx]
                X_fold_val = X_tr_scaled[fold_val_idx]
                y_fold_tr = y_tr[fold_train_idx]

                # Use high regularization for low variance
                model = Ridge(alpha=200)
                model.fit(X_fold_tr, y_fold_tr)

                oof_preds[player_indices[fold_val_idx], t_idx] = model.predict(X_fold_val)
                fold_test_preds.append(model.predict(X_te_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  CV: {scaled_mse:.6f}")

    # Total CV
    print("\nFinal CV:")
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return oof_preds, predictions, cv_score


def main():
    print("=" * 80)
    print("TARGET-SPECIALIZED MODEL")
    print("=" * 80)
    print("\nUltra-focused feature sets:")
    print("  Angle: lower body at frame 153 (ankle, knee, heel z)")
    print("  Depth: left hand at frame 102 (wrist, elbow)")
    print("  Left_right: right arm at frame 237 (wrist, elbow x)")

    data = load_and_extract_features()
    oof, test_preds, cv_score = train_specialized_model(data)

    # Create submission
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        try:
            name = f.stem
            if name.startswith('submission_'):
                nums.append(int(name.split('_')[1]))
            elif name.startswith('submission'):
                nums.append(int(name.replace('submission', '')))
        except:
            pass
    next_num = max(nums) + 1 if nums else 1

    scaled_preds = np.zeros_like(test_preds)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            test_preds[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': data["test"]["ids"],
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: target_specialized")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f} (target: <0.14, Sub 113: 0.1379)")
    print(f"  depth_mean: {depth_mean:.4f} (target: ~0.505, Sub 113: 0.5050)")
    print(f"  File: {filepath}")

    # Check correlation with Sub 113
    sub113_path = SUBMISSION_DIR / "submission_113.csv"
    if sub113_path.exists():
        sub113 = pd.read_csv(sub113_path)
        print("\nCorrelation with Sub 113:")
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            r = np.corrcoef(sub113[col], submission[col])[0, 1]
            print(f"  {col}: r={r:.4f}")

    return filepath


if __name__ == "__main__":
    main()
