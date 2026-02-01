"""
LightGBM + Biomechanical Features Ensemble

Use gradient boosting to capture non-linear patterns in biomech features,
then ensemble with Ridge for stability.
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
import lightgbm as lgb

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


def extract_features(series, participant_id):
    """Extract comprehensive features for LightGBM."""
    features = {}

    release_frames = list(range(145, 165))
    key_frames = [102, 153, 155, 160, 237]

    # Key joints
    joints = [
        ("right_wrist", "x"), ("right_wrist", "y"), ("right_wrist", "z"),
        ("right_elbow", "x"), ("right_elbow", "y"), ("right_elbow", "z"),
        ("right_shoulder", "x"), ("right_shoulder", "y"), ("right_shoulder", "z"),
        ("right_knee", "x"), ("right_knee", "y"), ("right_knee", "z"),
        ("right_hip", "x"), ("right_hip", "y"), ("right_hip", "z"),
        ("left_wrist", "x"), ("left_wrist", "y"), ("left_wrist", "z"),
        ("mid_hip", "x"), ("mid_hip", "y"), ("mid_hip", "z"),
    ]

    for joint, axis in joints:
        data = get_kp(series, f"{joint}_{axis}")
        if data is not None:
            vel = compute_velocity(data)

            # Position features
            features[f"{joint}_{axis}_mean"] = np.nanmean(data)
            features[f"{joint}_{axis}_release"] = np.nanmean(data[release_frames])

            # Velocity features
            features[f"{joint}_{axis}_vel_max"] = np.nanmax(np.abs(vel))
            features[f"{joint}_{axis}_vel_release"] = np.nanmean(vel[release_frames])
            features[f"{joint}_{axis}_vel_peak_time"] = np.argmax(np.abs(vel)) / 240.0

            # Key frame values
            for f in key_frames:
                if f < 240:
                    features[f"{joint}_{axis}_f{f}"] = data[f]
                    features[f"{joint}_{axis}_v{f}"] = vel[f]

    # Kinetic chain timing
    right_wrist_z = get_kp(series, "right_wrist_z")
    right_elbow_z = get_kp(series, "right_elbow_z")
    right_shoulder_z = get_kp(series, "right_shoulder_z")

    if all(x is not None for x in [right_wrist_z, right_elbow_z, right_shoulder_z]):
        wrist_vel = compute_velocity(right_wrist_z)
        elbow_vel = compute_velocity(right_elbow_z)
        shoulder_vel = compute_velocity(right_shoulder_z)

        # Wrist snap
        wrist_snap = wrist_vel - elbow_vel
        features["wrist_snap_max"] = np.nanmax(wrist_snap)
        features["wrist_snap_release"] = np.nanmean(wrist_snap[release_frames])

        # Velocity ratios
        features["vel_ratio_wrist_elbow"] = np.nanmax(np.abs(wrist_vel)) / (np.nanmax(np.abs(elbow_vel)) + 1e-6)
        features["vel_ratio_elbow_shoulder"] = np.nanmax(np.abs(elbow_vel)) / (np.nanmax(np.abs(shoulder_vel)) + 1e-6)

        # Coordination
        corr = np.corrcoef(elbow_vel[130:175], wrist_vel[130:175])[0, 1]
        features["elbow_wrist_coord"] = corr if not np.isnan(corr) else 0

    # Clean up
    for k, v in list(features.items()):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


def load_and_extract_features():
    """Load data and extract features."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    init_keypoint_mapping(keypoint_cols)

    def process_df(df, desc="Processing"):
        all_features = []
        all_pids = []
        all_ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            features = extract_features(ts, row['participant_id'])
            all_features.append(features)
            all_pids.append(row['participant_id'])
            all_ids.append(row['id'])

        return pd.DataFrame(all_features), np.array(all_pids), np.array(all_ids)

    print("Processing training data...")
    train_features, train_pids, train_ids = process_df(train_df, "Train")
    train_targets = train_df[["angle", "depth", "left_right"]].values

    print("Processing test data...")
    test_features, test_pids, test_ids = process_df(test_df, "Test")

    return {
        "train_features": train_features,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_features": test_features,
        "test_pids": test_pids,
        "test_ids": test_ids,
    }


def train_ensemble(data):
    """Train LightGBM + Ridge ensemble."""
    print("\n" + "=" * 60)
    print("LGB + RIDGE ENSEMBLE")
    print("=" * 60)

    X_train = np.nan_to_num(data["train_features"].values, nan=0.0)
    y_train = data["train_targets"]
    train_pids = data["train_pids"]

    X_test = np.nan_to_num(data["test_features"].values, nan=0.0)
    test_pids = data["test_pids"]

    print(f"Features: {X_train.shape[1]}")

    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(X_test), 3))
    oof_preds = np.zeros_like(y_train)

    for t_idx, target in enumerate(TARGETS):
        print(f"\n{target}:")

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_tr = X_train[train_mask]
            X_te = X_test[test_mask]
            y_tr = y_train[train_mask, t_idx]
            player_indices = np.where(train_mask)[0]

            # Scale for Ridge
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            lgb_oof = np.zeros(len(X_tr))
            ridge_oof = np.zeros(len(X_tr))
            lgb_test = []
            ridge_test = []

            for fold_train_idx, fold_val_idx in kf.split(X_tr):
                X_fold_tr, X_fold_val = X_tr[fold_train_idx], X_tr[fold_val_idx]
                X_fold_tr_s, X_fold_val_s = X_tr_scaled[fold_train_idx], X_tr_scaled[fold_val_idx]
                y_fold_tr = y_tr[fold_train_idx]

                # LightGBM
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=42, verbose=-1
                )
                lgb_model.fit(X_fold_tr, y_fold_tr)
                lgb_oof[fold_val_idx] = lgb_model.predict(X_fold_val)
                lgb_test.append(lgb_model.predict(X_te))

                # Ridge
                ridge_model = Ridge(alpha=100)
                ridge_model.fit(X_fold_tr_s, y_fold_tr)
                ridge_oof[fold_val_idx] = ridge_model.predict(X_fold_val_s)
                ridge_test.append(ridge_model.predict(X_te_scaled))

            # Blend: 50% LGB + 50% Ridge
            oof_blend = 0.5 * lgb_oof + 0.5 * ridge_oof
            test_blend = 0.5 * np.mean(lgb_test, axis=0) + 0.5 * np.mean(ridge_test, axis=0)

            oof_preds[player_indices, t_idx] = oof_blend
            predictions[test_mask, t_idx] = test_blend

        mse = np.mean((oof_preds[:, t_idx] - y_train[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  CV: {scaled_mse:.6f}")

    # Total CV
    print("\nFinal CV:")
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - y_train[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"  TOTAL CV: {cv_score:.6f}")

    return oof_preds, predictions, cv_score


def main():
    print("=" * 80)
    print("LGB + RIDGE ENSEMBLE WITH BIOMECH FEATURES")
    print("=" * 80)

    data = load_and_extract_features()
    oof, test_preds, cv_score = train_ensemble(data)

    # Create submission
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.startswith('submission_')]
    next_num = max(nums) + 1

    scaled_preds = np.zeros_like(test_preds)
    for i, target in enumerate(TARGETS):
        scaled_preds[:, i] = TARGET_SCALERS[target].transform(
            test_preds[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': data["test_ids"],
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: lgb_ridge_biomech")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")
    print(f"  depth_mean: {depth_mean:.4f}")
    print(f"  File: {filepath}")

    # Check correlation with Sub 113
    sub113 = pd.read_csv(SUBMISSION_DIR / "submission_113.csv")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        r = np.corrcoef(sub113[col], submission[col])[0, 1]
        print(f"  {col} vs Sub113: r={r:.4f}")

    # Test blending
    print("\nBlending with Sub 113:")
    cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']
    for w in [0.05, 0.10, 0.15, 0.20]:
        blend = (1-w) * sub113[cols] + w * submission[cols]
        print(f"  w={w:.2f}: angle_std={blend.scaled_angle.std():.4f}, "
              f"depth_mean={blend.scaled_depth.mean():.4f}")

    return filepath


if __name__ == "__main__":
    main()
