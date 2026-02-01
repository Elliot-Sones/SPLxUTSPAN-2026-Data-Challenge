"""
Simple Robust Submission - Focus on generalization over CV score

The 7539 feature ElasticNet overfit badly (CV 0.0069, LB 0.0677).
This script uses minimal, robust features with strong regularization.
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
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
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


def extract_simple_features(ts, participant_id, kp_idx):
    """Extract only the most robust, generalizable features."""
    features = {}

    # Key joints only - the most important ones for basketball shooting
    key_joints = ["right_wrist", "right_elbow", "right_shoulder", "mid_hip"]

    # Only a few key frames around release (most predictive)
    key_frames = [145, 150, 155, 160]

    for joint in key_joints:
        if joint not in kp_idx:
            continue
        for coord in ["x", "y", "z"]:
            if coord not in kp_idx[joint]:
                continue
            idx = kp_idx[joint][coord]
            pos = smooth_signal(ts[:, idx], window=7)

            # Position at key frames
            for f in key_frames:
                features[f"{joint}_{coord}_f{f}"] = pos[f]

            # Simple statistics over release window
            window = pos[140:170]
            features[f"{joint}_{coord}_release_mean"] = np.nanmean(window)
            features[f"{joint}_{coord}_release_std"] = np.nanstd(window)
            features[f"{joint}_{coord}_release_range"] = np.nanmax(window) - np.nanmin(window)

    # Add guide hand (left wrist) - important for left-right
    if "left_wrist" in kp_idx:
        for coord in ["x", "y", "z"]:
            if coord in kp_idx["left_wrist"]:
                idx = kp_idx["left_wrist"][coord]
                pos = smooth_signal(ts[:, idx], window=7)
                features[f"left_wrist_{coord}_f150"] = pos[150]
                features[f"left_wrist_{coord}_release_mean"] = np.nanmean(pos[140:170])

    return features


def load_data():
    """Load data with simple features."""
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

            feat = extract_simple_features(ts, row["participant_id"], kp_idx)
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


def train_and_evaluate(data, alpha=1000.0):
    """Train Ridge models with strong regularization."""
    X = data["X_train"]
    y = data["y_train"]
    pids = data["train_pids"]
    ranges = get_target_ranges()

    unique_pids = sorted(np.unique(pids))

    all_models = {}
    all_scalers = {}
    oof_preds = np.zeros_like(y)

    print(f"\nTraining Ridge (alpha={alpha}) per-player models...")

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        player_indices = np.where(pid_mask)[0]
        n_samples = len(X_player)

        print(f"  Player {pid} ({n_samples} samples)")

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

                # Ridge with strong regularization
                model = Ridge(alpha=alpha, random_state=42)
                model.fit(X_train, y_train)
                fold_preds[val_idx] = model.predict(X_val)

            oof_preds[player_indices, target_idx] = fold_preds

            # Train final model
            final_model = Ridge(alpha=alpha, random_state=42)
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
    existing = list(SUBMISSION_DIR.glob("submission*.csv"))
    nums = []
    for f in existing:
        name = f.stem
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

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_preds[:, 0],
        'scaled_depth': scaled_preds[:, 1],
        'scaled_left_right': scaled_preds[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(filepath, index=False)

    print(f"\nSUBMISSION {next_num} CREATED")
    print(f"  File: {filepath}")
    print(f"  CV Score: {cv_score:.6f}")

    return filepath, next_num


def main():
    print("=" * 80)
    print("SIMPLE ROBUST SUBMISSION")
    print("Focus: Generalization over CV score")
    print("=" * 80)

    data = load_data()

    # Try different alpha values
    best_cv = float('inf')
    best_alpha = None

    for alpha in [100, 500, 1000, 2000, 5000]:
        print(f"\n--- Testing alpha={alpha} ---")
        _, _, cv = train_and_evaluate(data, alpha=alpha)
        if cv < best_cv:
            best_cv = cv
            best_alpha = alpha

    print(f"\n{'='*60}")
    print(f"Best alpha: {best_alpha} with CV: {best_cv:.6f}")
    print(f"{'='*60}")

    # Train final model with best alpha
    models, scalers, cv_score = train_and_evaluate(data, alpha=best_alpha)
    predictions, test_ids = generate_predictions(data, models, scalers)
    filepath, sub_num = create_submission(test_ids, predictions, cv_score)

    print(f"\nSummary:")
    print(f"  Features: {data['X_train'].shape[1]}")
    print(f"  Alpha: {best_alpha}")
    print(f"  CV Score: {cv_score:.6f}")

    return filepath


if __name__ == "__main__":
    main()
