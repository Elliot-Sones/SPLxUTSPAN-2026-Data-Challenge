"""
Target-Specific Optimization

Focus on improving left_right which is the bottleneck.
Try different feature sets and models specifically for each target.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

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
    """Load data with keypoint index."""
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
        "kp_idx": kp_idx,
    }


def extract_left_right_features(ts, kp_idx):
    """
    Extract features specifically for left_right prediction.

    left_right is about lateral deviation, so focus on:
    - X coordinates (lateral position)
    - Differences between left and right body parts
    - Hip and shoulder alignment
    - Guide hand (left wrist) position
    """
    features = {}

    frames = list(range(140, 170))

    # Lateral (X) positions of key joints
    lateral_joints = [
        "right_wrist", "left_wrist", "right_elbow", "left_elbow",
        "right_shoulder", "left_shoulder", "mid_hip"
    ]

    for joint in lateral_joints:
        if joint in kp_idx and "x" in kp_idx[joint]:
            idx = kp_idx[joint]["x"]
            x_vals = ts[frames, idx]
            features[f"{joint}_x_mean"] = np.nanmean(x_vals)
            features[f"{joint}_x_std"] = np.nanstd(x_vals)
            features[f"{joint}_x_range"] = np.nanmax(x_vals) - np.nanmin(x_vals)

            # Key frames
            for f in [145, 150, 155, 160]:
                features[f"{joint}_x_f{f}"] = ts[f, idx]

    # Left-right differences
    if "right_wrist" in kp_idx and "left_wrist" in kp_idx:
        if "x" in kp_idx["right_wrist"] and "x" in kp_idx["left_wrist"]:
            r_idx = kp_idx["right_wrist"]["x"]
            l_idx = kp_idx["left_wrist"]["x"]
            diff = ts[frames, r_idx] - ts[frames, l_idx]
            features["wrist_lr_diff_mean"] = np.nanmean(diff)
            features["wrist_lr_diff_std"] = np.nanstd(diff)

    if "right_shoulder" in kp_idx and "left_shoulder" in kp_idx:
        if "x" in kp_idx["right_shoulder"] and "x" in kp_idx["left_shoulder"]:
            r_idx = kp_idx["right_shoulder"]["x"]
            l_idx = kp_idx["left_shoulder"]["x"]
            diff = ts[frames, r_idx] - ts[frames, l_idx]
            features["shoulder_lr_diff_mean"] = np.nanmean(diff)
            features["shoulder_lr_diff_std"] = np.nanstd(diff)

    # Hip alignment
    if "mid_hip" in kp_idx and "x" in kp_idx["mid_hip"]:
        hip_idx = kp_idx["mid_hip"]["x"]
        hip_x = ts[frames, hip_idx]
        features["hip_x_mean"] = np.nanmean(hip_x)
        features["hip_x_std"] = np.nanstd(hip_x)

        # Hip relative to shoulder center
        if "right_shoulder" in kp_idx and "left_shoulder" in kp_idx:
            rs_idx = kp_idx["right_shoulder"]["x"]
            ls_idx = kp_idx["left_shoulder"]["x"]
            shoulder_center = (ts[frames, rs_idx] + ts[frames, ls_idx]) / 2
            hip_offset = hip_x - shoulder_center
            features["hip_shoulder_offset_mean"] = np.nanmean(hip_offset)
            features["hip_shoulder_offset_std"] = np.nanstd(hip_offset)

    return features


def extract_depth_features(ts, kp_idx):
    """
    Extract features specifically for depth prediction.

    Depth is about how far the ball goes, so focus on:
    - Z coordinates (depth/forward)
    - Forward lean
    - Arm extension
    """
    features = {}

    frames = list(range(140, 170))

    # Z positions (depth)
    depth_joints = [
        "right_wrist", "right_elbow", "right_shoulder", "mid_hip"
    ]

    for joint in depth_joints:
        if joint in kp_idx and "z" in kp_idx[joint]:
            idx = kp_idx[joint]["z"]
            z_vals = ts[frames, idx]
            features[f"{joint}_z_mean"] = np.nanmean(z_vals)
            features[f"{joint}_z_std"] = np.nanstd(z_vals)
            features[f"{joint}_z_range"] = np.nanmax(z_vals) - np.nanmin(z_vals)

            for f in [145, 150, 155, 160]:
                features[f"{joint}_z_f{f}"] = ts[f, idx]

    # Forward lean (wrist relative to hip in Z)
    if "right_wrist" in kp_idx and "mid_hip" in kp_idx:
        if "z" in kp_idx["right_wrist"] and "z" in kp_idx["mid_hip"]:
            w_idx = kp_idx["right_wrist"]["z"]
            h_idx = kp_idx["mid_hip"]["z"]
            lean = ts[frames, w_idx] - ts[frames, h_idx]
            features["forward_lean_mean"] = np.nanmean(lean)
            features["forward_lean_std"] = np.nanstd(lean)

    # Arm extension (elbow angle proxy)
    if "right_wrist" in kp_idx and "right_elbow" in kp_idx and "right_shoulder" in kp_idx:
        for coord in ["x", "y", "z"]:
            if coord in kp_idx["right_wrist"] and coord in kp_idx["right_elbow"]:
                w_idx = kp_idx["right_wrist"][coord]
                e_idx = kp_idx["right_elbow"][coord]
                s_idx = kp_idx["right_shoulder"][coord]

                forearm = ts[frames, w_idx] - ts[frames, e_idx]
                upperarm = ts[frames, e_idx] - ts[frames, s_idx]

                features[f"forearm_{coord}_mean"] = np.nanmean(forearm)
                features[f"upperarm_{coord}_mean"] = np.nanmean(upperarm)

    return features


def extract_angle_features(ts, kp_idx):
    """
    Extract features specifically for angle prediction.

    Angle is about vertical deviation, so focus on:
    - Y coordinates (vertical)
    - Release height
    - Elbow lift
    """
    features = {}

    frames = list(range(140, 170))

    # Y positions (height)
    height_joints = [
        "right_wrist", "right_elbow", "right_shoulder"
    ]

    for joint in height_joints:
        if joint in kp_idx and "y" in kp_idx[joint]:
            idx = kp_idx[joint]["y"]
            y_vals = ts[frames, idx]
            features[f"{joint}_y_mean"] = np.nanmean(y_vals)
            features[f"{joint}_y_std"] = np.nanstd(y_vals)
            features[f"{joint}_y_range"] = np.nanmax(y_vals) - np.nanmin(y_vals)

            for f in [145, 150, 155, 160]:
                features[f"{joint}_y_f{f}"] = ts[f, idx]

    # Release height
    if "right_wrist" in kp_idx and "y" in kp_idx["right_wrist"]:
        w_idx = kp_idx["right_wrist"]["y"]
        features["release_height"] = ts[155, w_idx]

    # Elbow angle proxy
    if "right_wrist" in kp_idx and "right_elbow" in kp_idx and "right_shoulder" in kp_idx:
        if "y" in kp_idx["right_wrist"]:
            w_y = ts[frames, kp_idx["right_wrist"]["y"]]
            e_y = ts[frames, kp_idx["right_elbow"]["y"]]
            features["wrist_elbow_y_diff_mean"] = np.nanmean(w_y - e_y)
            features["wrist_elbow_y_diff_std"] = np.nanstd(w_y - e_y)

    return features


def run_target_specific_models(data):
    """
    Run target-specific feature extraction and modeling.
    """
    print("\n" + "="*60)
    print("TARGET-SPECIFIC OPTIMIZATION")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]
    kp_idx = data["kp_idx"]

    ranges = get_target_ranges()

    # Extract target-specific features
    print("\nExtracting target-specific features...")

    feature_extractors = {
        0: extract_angle_features,  # angle
        1: extract_depth_features,  # depth
        2: extract_left_right_features,  # left_right
    }

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for t_idx, target in enumerate(TARGETS):
        print(f"\n  Target: {target}")

        extractor = feature_extractors[t_idx]

        # Extract features
        train_features = [extractor(ts, kp_idx) for ts in train_series]
        test_features = [extractor(ts, kp_idx) for ts in test_series]

        feature_names = sorted(train_features[0].keys())
        X_train = np.array([[f.get(n, 0) for n in feature_names] for f in train_features])
        X_test = np.array([[f.get(n, 0) for n in feature_names] for f in test_features])

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        print(f"    Features: {X_train.shape[1]}")

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_tr = X_train[train_mask]
            y_tr = train_targets[train_mask, t_idx]
            X_te = X_test[test_mask]

            player_indices = np.where(train_mask)[0]

            # Try multiple model configurations
            configs = [
                ("pca10_ridge10", 10, 10),
                ("pca15_ridge50", 15, 50),
                ("pca10_ridge100", 10, 100),
                ("pca15_ridge100", 15, 100),
            ]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Find best config
            best_cv = float('inf')
            best_config = None

            for name, n_comp, alpha in configs:
                n_comp = min(n_comp, len(X_tr) - 2)
                fold_errors = []

                for train_idx, val_idx in kf.split(X_tr):
                    Xtr, Xval = X_tr[train_idx], X_tr[val_idx]
                    ytr, yval = y_tr[train_idx], y_tr[val_idx]

                    scaler = StandardScaler()
                    Xtr_s = scaler.fit_transform(Xtr)
                    Xval_s = scaler.transform(Xval)

                    pca = PCA(n_components=n_comp)
                    Xtr_p = pca.fit_transform(Xtr_s)
                    Xval_p = pca.transform(Xval_s)

                    model = Ridge(alpha=alpha)
                    model.fit(Xtr_p, ytr)
                    pred = model.predict(Xval_p)
                    fold_errors.append(np.mean((pred - yval)**2))

                cv = np.mean(fold_errors)
                if cv < best_cv:
                    best_cv = cv
                    best_config = (name, n_comp, alpha)

            # Use best config
            name, n_comp, alpha = best_config
            n_comp = min(n_comp, len(X_tr) - 2)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            pca = PCA(n_components=n_comp)
            X_tr_p = pca.fit_transform(X_tr_s)
            X_te_p = pca.transform(X_te_s)

            model = Ridge(alpha=alpha)
            model.fit(X_tr_p, y_tr)
            predictions[test_mask, t_idx] = model.predict(X_te_p)

            # OOF predictions
            for train_idx, val_idx in kf.split(X_tr):
                Xtr, Xval = X_tr_s[train_idx], X_tr_s[val_idx]
                ytr = y_tr[train_idx]

                pca_fold = PCA(n_components=n_comp)
                Xtr_p = pca_fold.fit_transform(Xtr)
                Xval_p = pca_fold.transform(Xval)

                model_fold = Ridge(alpha=alpha)
                model_fold.fit(Xtr_p, ytr)
                oof_preds[player_indices[val_idx], t_idx] = model_fold.predict(Xval_p)

        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"    CV: {scaled_mse:.6f}")

    total_cv = sum(
        np.mean((oof_preds[:, i] - train_targets[:, i])**2) / (ranges[t]**2)
        for i, t in enumerate(TARGETS)
    ) / 3

    print(f"\n  TOTAL CV: {total_cv:.6f}")

    return predictions, total_cv, oof_preds


def combine_with_general_model(data, specific_preds, specific_oof):
    """
    Combine target-specific model with general model.
    """
    print("\n" + "="*60)
    print("COMBINING WITH GENERAL MODEL")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]

    ranges = get_target_ranges()

    frames = list(range(140, 170))

    def flatten(series):
        flat = series[:, frames, :].reshape(len(series), -1)
        return np.nan_to_num(flat, nan=0.0)

    train_flat = flatten(train_series)
    test_flat = flatten(test_series)

    # General prototype model
    general_preds = np.zeros((len(test_series), 3))
    general_oof = np.zeros_like(train_targets)

    unique_pids = sorted(np.unique(train_pids))

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_train = train_flat[train_mask]
        y_train = train_targets[train_mask]
        X_test = test_flat[test_mask]

        player_indices = np.where(train_mask)[0]

        # Prototype model
        prototype = X_train.mean(axis=0)
        train_dist = np.linalg.norm(X_train - prototype, axis=1, keepdims=True)
        test_dist = np.linalg.norm(X_test - prototype, axis=1, keepdims=True)

        pca = PCA(n_components=min(10, len(X_train)-1))
        train_pca = pca.fit_transform(X_train)
        test_pca = pca.transform(X_test)

        X_train_feat = np.hstack([train_dist, train_pca])
        X_test_feat = np.hstack([test_dist, test_pca])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_test_scaled = scaler.transform(X_test_feat)

        for t_idx in range(3):
            model = Ridge(alpha=10.0)
            model.fit(X_train_scaled, y_train[:, t_idx])
            general_preds[test_mask, t_idx] = model.predict(X_test_scaled)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr = y_train[train_idx]

            for t_idx in range(3):
                model = Ridge(alpha=10.0)
                model.fit(X_tr, y_tr[:, t_idx])
                general_oof[player_indices[val_idx], t_idx] = model.predict(X_val)

    # Find optimal blend weights
    print("\nFinding optimal blend weights...")

    final_preds = np.zeros_like(specific_preds)
    final_oof = np.zeros_like(specific_oof)

    for t_idx, target in enumerate(TARGETS):
        y_true = train_targets[:, t_idx]

        def loss(w):
            w = np.clip(w, 0, 1)
            ensemble = w * specific_oof[:, t_idx] + (1 - w) * general_oof[:, t_idx]
            return np.mean((ensemble - y_true)**2)

        result = minimize(loss, 0.5, method='L-BFGS-B', bounds=[(0, 1)])
        best_w = result.x[0]

        final_preds[:, t_idx] = best_w * specific_preds[:, t_idx] + (1 - best_w) * general_preds[:, t_idx]
        final_oof[:, t_idx] = best_w * specific_oof[:, t_idx] + (1 - best_w) * general_oof[:, t_idx]

        mse = np.mean((final_oof[:, t_idx] - y_true)**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: weight={best_w:.3f}, CV={scaled_mse:.6f}")

    total_cv = sum(
        np.mean((final_oof[:, i] - train_targets[:, i])**2) / (ranges[t]**2)
        for i, t in enumerate(TARGETS)
    ) / 3

    print(f"\n  TOTAL CV: {total_cv:.6f}")

    return final_preds, total_cv


def create_submission(test_ids, predictions, cv_score, approach_name):
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

    print(f"\nSubmission {next_num} created: {filepath}")
    print(f"  Approach: {approach_name}")
    print(f"  CV Score: {cv_score:.6f}")

    return filepath, next_num


def main():
    print("="*80)
    print("TARGET-SPECIFIC OPTIMIZATION")
    print("="*80)

    data = load_data()

    # Run target-specific models
    specific_preds, specific_cv, specific_oof = run_target_specific_models(data)

    # Combine with general model
    final_preds, final_cv = combine_with_general_model(data, specific_preds, specific_oof)

    # Create submission
    create_submission(
        data["test_ids"],
        final_preds,
        final_cv,
        "target_specific_combined"
    )

    return final_cv


if __name__ == "__main__":
    main()
