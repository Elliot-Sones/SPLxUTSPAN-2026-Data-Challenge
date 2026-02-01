"""
ElasticNet Submission - Best CV Score 0.007778

Based on depth_lr_optimization.py results:
- ElasticNet achieved total CV of 0.007778
- This beats our previous best of 0.008810
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
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

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


def extract_all_features(ts, participant_id, kp_idx, keypoint_cols):
    """Extract all features including depth and left-right specific ones."""
    features = {}

    # Participant one-hot
    for i in range(1, 6):
        features[f"participant_{i}"] = 1.0 if participant_id == i else 0.0

    # Key frames for angle
    angle_frames = [148, 150, 153, 155, 158]
    angle_window = (145, 165)

    # Key frames for depth - earlier in shot
    depth_frames = [100, 102, 105, 108, 110]
    depth_window = (95, 115)

    # Key frames for left-right - follow through
    lr_frames = [220, 225, 230, 235, 237]
    lr_window = (215, 240)

    # Phase definitions
    setup_range = (0, 60)
    load_range = (60, 120)
    release_range = (120, 180)
    follow_range = (180, 240)

    # Extract keypoint features
    for kp_name, coords in kp_idx.items():
        for coord, idx in coords.items():
            pos = ts[:, idx]
            pos_smooth = smooth_signal(pos, window=5)
            vel = compute_velocity(pos_smooth)

            # Angle frame features
            for f in angle_frames:
                features[f"angle_f{f}_{kp_name}_{coord}"] = pos_smooth[f]
                features[f"angle_f{f}_{kp_name}_v{coord}"] = vel[f]

            # Angle window stats
            w = pos_smooth[angle_window[0]:angle_window[1]]
            features[f"angle_window_{kp_name}_{coord}_mean"] = np.nanmean(w)
            features[f"angle_window_{kp_name}_{coord}_std"] = np.nanstd(w)

            # Depth frame features
            for f in depth_frames:
                features[f"depth_f{f}_{kp_name}_{coord}"] = pos_smooth[f]

            # Depth window stats
            w = pos_smooth[depth_window[0]:depth_window[1]]
            features[f"depth_window_{kp_name}_{coord}_mean"] = np.nanmean(w)

            # Left-right frame features
            for f in lr_frames:
                if f < 240:
                    features[f"lr_f{f}_{kp_name}_{coord}"] = pos_smooth[f]

            # Phase features
            for phase_name, (start, end) in [
                ("setup", setup_range),
                ("load", load_range),
                ("release", release_range),
                ("follow", follow_range)
            ]:
                phase_pos = pos_smooth[start:end]
                phase_vel = vel[start:end]
                features[f"phase_{phase_name}_{kp_name}_{coord}_mean"] = np.nanmean(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_{coord}_std"] = np.nanstd(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_{coord}_range"] = np.nanmax(phase_pos) - np.nanmin(phase_pos)
                features[f"phase_{phase_name}_{kp_name}_vel_max"] = np.nanmax(np.abs(phase_vel))

    # Depth-specific features
    if "right_wrist" in kp_idx and "mid_hip" in kp_idx:
        if "x" in kp_idx["right_wrist"] and "x" in kp_idx["mid_hip"]:
            wrist_x = ts[:, kp_idx["right_wrist"]["x"]]
            hip_x = ts[:, kp_idx["mid_hip"]["x"]]
            features["depth_wrist_forward_of_hip"] = np.nanmean(wrist_x[100:120] - hip_x[100:120])

    if "left_wrist" in kp_idx and "x" in kp_idx.get("left_wrist", {}):
        left_wrist_x = ts[:, kp_idx["left_wrist"]["x"]]
        features["depth_left_wrist_x_retraction"] = np.nanmean(left_wrist_x[60:100]) - np.nanmean(left_wrist_x[100:140])

    if "mid_hip" in kp_idx and "z" in kp_idx.get("mid_hip", {}):
        hip_z = ts[:, kp_idx["mid_hip"]["z"]]
        features["depth_hip_drop"] = np.nanmax(hip_z[60:120]) - np.nanmin(hip_z[60:120])

    # Left-right specific features
    if "right_elbow" in kp_idx and "z" in kp_idx.get("right_elbow", {}):
        elbow_z = ts[:, kp_idx["right_elbow"]["z"]]
        features["lr_elbow_z_follow"] = np.nanmean(elbow_z[180:220])

    if "mid_hip" in kp_idx and "x" in kp_idx.get("mid_hip", {}):
        hip_x = ts[:, kp_idx["mid_hip"]["x"]]
        features["lr_hip_x_setup"] = np.nanstd(hip_x[0:60])

    # Guide hand features
    if "left_wrist" in kp_idx:
        lw = kp_idx["left_wrist"]
        if "x" in lw and "y" in lw and "z" in lw:
            lw_x = smooth_signal(ts[:, lw["x"]], 5)
            lw_y = smooth_signal(ts[:, lw["y"]], 5)
            lw_z = smooth_signal(ts[:, lw["z"]], 5)

            lw_vx = compute_velocity(lw_x)
            lw_vy = compute_velocity(lw_y)
            lw_vz = compute_velocity(lw_z)

            release_frame = 150
            features["guide_hand_x_at_release"] = lw_x[release_frame]
            features["guide_hand_y_at_release"] = lw_y[release_frame]
            features["guide_hand_z_at_release"] = lw_z[release_frame]
            features["guide_hand_vx"] = lw_vx[release_frame]
            features["guide_hand_vy"] = lw_vy[release_frame]
            features["guide_hand_vz"] = lw_vz[release_frame]

    # Shoulder alignment
    if "left_shoulder" in kp_idx and "right_shoulder" in kp_idx:
        ls = kp_idx["left_shoulder"]
        rs = kp_idx["right_shoulder"]
        if "y" in ls and "y" in rs:
            ls_y = ts[:, ls["y"]]
            rs_y = ts[:, rs["y"]]
            features["depth_shoulder_alignment"] = np.nanmean((ls_y - rs_y)[100:150])
        if "z" in ls and "z" in rs:
            ls_z = ts[:, ls["z"]]
            rs_z = ts[:, rs["z"]]
            features["lr_shoulder_z_diff"] = np.nanmean((ls_z - rs_z)[150:200])

    return features


def load_data():
    """Load train and test data with all features."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

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

    def process_df(df, is_train):
        all_features = []
        targets = []
        pids = []
        ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])

            feat = extract_all_features(ts, row["participant_id"], kp_idx, keypoint_cols)
            all_features.append(feat)
            pids.append(row["participant_id"])
            ids.append(row["id"])

            if is_train:
                targets.append([row["angle"], row["depth"], row["left_right"]])

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
    """Train ElasticNet models per player."""
    X = data["X_train"]
    y = data["y_train"]
    pids = data["train_pids"]
    ranges = get_target_ranges()

    unique_pids = sorted(np.unique(pids))

    all_models = {}
    all_scalers = {}
    oof_preds = np.zeros_like(y)

    print("\nTraining ElasticNet per-player models...")

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

                # ElasticNet with default parameters (works well based on our tests)
                model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
                model.fit(X_train, y_train)
                fold_preds[val_idx] = model.predict(X_val)

            oof_preds[player_indices, target_idx] = fold_preds

            # Train final model
            final_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
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
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        name = f.stem
        # Handle both submission_X and submissionX formats
        if name.startswith("submission_"):
            try:
                nums.append(int(name.split('_')[1]))
            except:
                pass
        elif name.startswith("submission"):
            try:
                nums.append(int(name[10:]))
            except:
                pass

    next_num = max(nums) + 1 if nums else 1

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
    print("ELASTICNET SUBMISSION")
    print("Best CV Score: 0.007778")
    print("=" * 80)

    # Load data
    data = load_data()

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

    return filepath


if __name__ == "__main__":
    main()
