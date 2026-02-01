"""
Submission: Target-Specific Optimal Config (Strategy 9)

Use different optimal settings for each target independently.
Pure per-player models - no global component.

Best CV among all strategies tested: 0.007580
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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


def load_data():
    """Load data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    def extract_series(df):
        all_series = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading"):
            ts = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                ts[:, i] = parse_array_string(row[col])
            all_series.append(ts)
        return np.array(all_series)

    print("Loading training data...")
    train_series = extract_series(train_df)
    train_targets = train_df[["angle", "depth", "left_right"]].values
    train_pids = train_df["participant_id"].values
    train_ids = train_df["id"].values

    print("Loading test data...")
    test_series = extract_series(test_df)
    test_pids = test_df["participant_id"].values
    test_ids = test_df["id"].values

    return {
        "train_series": train_series,
        "train_targets": train_targets,
        "train_pids": train_pids,
        "train_ids": train_ids,
        "test_series": test_series,
        "test_pids": test_pids,
        "test_ids": test_ids,
        "keypoint_cols": keypoint_cols,
    }


def target_specific_model(data):
    """
    Use optimal config for each target independently.
    Pure per-player models.
    """
    print("\nTarget-Specific Optimal Config")
    print("=" * 60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    # Optimal configs per target (from per_target_best.py results)
    target_configs = {
        0: {"name": "angle", "frames": list(range(145, 165)), "alpha": 100, "n_comp": 15},
        1: {"name": "depth", "frames": list(range(145, 165)), "alpha": 10, "n_comp": 15},
        2: {"name": "left_right", "frames": list(range(130, 180)), "alpha": 10, "n_comp": 15},
    }

    unique_pids = sorted(np.unique(train_pids))

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for t_idx, config in target_configs.items():
        target_name = config["name"]
        frames = config["frames"]
        alpha = config["alpha"]
        n_comp = config["n_comp"]

        print(f"\n  {target_name}:")
        print(f"    frames: {frames[0]}-{frames[-1]}, alpha: {alpha}, n_comp: {n_comp}")

        def flatten(series):
            flat = series[:, frames, :].reshape(len(series), -1)
            return np.nan_to_num(flat, nan=0.0)

        train_flat = flatten(train_series)
        test_flat = flatten(test_series)

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_flat[train_mask]
            y_train = train_targets[train_mask, t_idx]
            X_test = test_flat[test_mask]

            player_indices = np.where(train_mask)[0]

            n_c = min(n_comp, len(X_train) - 1)
            pca = PCA(n_components=n_c)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_pca)
            X_test_scaled = scaler.transform(X_test_pca)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx]

                model = Ridge(alpha=alpha)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        # Report CV for this target
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target_name]**2)
        print(f"    CV: {scaled_mse:.6f}")

    # Total CV
    print("\n  Final CV:")
    total_mse = 0
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"    {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"    TOTAL CV: {cv_score:.6f}")

    return oof_preds, predictions, cv_score


def create_submission(test_ids, predictions, cv_score, approach_name):
    """Create submission."""
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

    angle_std = submission['scaled_angle'].std()
    depth_mean = submission['scaled_depth'].mean()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {next_num}: {approach_name}")
    print(f"{'='*60}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")
    print(f"  depth_mean: {depth_mean:.4f}")
    print(f"  File: {filepath}")

    return filepath, next_num


def main():
    print("=" * 80)
    print("TARGET-SPECIFIC OPTIMAL CONFIG SUBMISSION")
    print("=" * 80)
    print("\nApproach: Different optimal settings for each target")
    print("  - angle: narrow frames (145-165), alpha=100")
    print("  - depth: narrow frames (145-165), alpha=10")
    print("  - left_right: wide frames (130-180), alpha=10")
    print("\nPure per-player models - no global component")

    data = load_data()

    # Run the model
    oof, test_preds, cv_score = target_specific_model(data)

    # Create submission
    create_submission(data["test_ids"], test_preds, cv_score, "target_specific_optimal")


if __name__ == "__main__":
    main()
