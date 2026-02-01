"""
Multi-Model Combination

Instead of predicting targets directly, predict intermediate physical quantities
and combine them.

For ANGLE:
- Predict wrist velocity → correlates with shot power/arc
- Predict release height → correlates with shot trajectory
- Predict elbow extension → correlates with follow-through
- Combine these to predict angle

For DEPTH:
- Predict forward lean
- Predict arm extension
- Predict wrist snap velocity
- Combine for depth

For LEFT_RIGHT:
- Predict lateral wrist position
- Predict hip alignment
- Predict shoulder rotation
- Combine for left_right
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.stats import pearsonr
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


def compute_intermediate_features(ts, kp_idx):
    """
    Compute physical intermediate features from time series.
    These are the 'auxiliary targets' that we'll predict first.
    """
    features = {}

    release_frame = 155
    window = 5

    # 1. WRIST VELOCITY at release (magnitude)
    if "right_wrist" in kp_idx:
        wrist_vel = []
        for coord in ["x", "y", "z"]:
            if coord in kp_idx["right_wrist"]:
                idx = kp_idx["right_wrist"][coord]
                vel = np.gradient(ts[:, idx])
                wrist_vel.append(vel[release_frame])
        if wrist_vel:
            features["wrist_velocity"] = np.sqrt(np.sum(np.array(wrist_vel)**2))
            features["wrist_vel_y"] = wrist_vel[1] if len(wrist_vel) > 1 else 0

    # 2. RELEASE HEIGHT (wrist Y at release)
    if "right_wrist" in kp_idx and "y" in kp_idx["right_wrist"]:
        idx = kp_idx["right_wrist"]["y"]
        features["release_height"] = ts[release_frame, idx]

    # 3. ELBOW EXTENSION (distance from shoulder to wrist)
    if "right_wrist" in kp_idx and "right_shoulder" in kp_idx:
        dist = 0
        for coord in ["x", "y", "z"]:
            if coord in kp_idx["right_wrist"] and coord in kp_idx["right_shoulder"]:
                w_idx = kp_idx["right_wrist"][coord]
                s_idx = kp_idx["right_shoulder"][coord]
                dist += (ts[release_frame, w_idx] - ts[release_frame, s_idx])**2
        features["arm_extension"] = np.sqrt(dist)

    # 4. FORWARD LEAN (wrist Z relative to hip Z)
    if "right_wrist" in kp_idx and "mid_hip" in kp_idx:
        if "z" in kp_idx["right_wrist"] and "z" in kp_idx["mid_hip"]:
            w_idx = kp_idx["right_wrist"]["z"]
            h_idx = kp_idx["mid_hip"]["z"]
            features["forward_lean"] = ts[release_frame, w_idx] - ts[release_frame, h_idx]

    # 5. WRIST SNAP (acceleration at release)
    if "right_wrist" in kp_idx:
        wrist_acc = []
        for coord in ["x", "y", "z"]:
            if coord in kp_idx["right_wrist"]:
                idx = kp_idx["right_wrist"][coord]
                acc = np.gradient(np.gradient(ts[:, idx]))
                wrist_acc.append(acc[release_frame])
        if wrist_acc:
            features["wrist_snap"] = np.sqrt(np.sum(np.array(wrist_acc)**2))

    # 6. LATERAL POSITION (wrist X relative to hip X)
    if "right_wrist" in kp_idx and "mid_hip" in kp_idx:
        if "x" in kp_idx["right_wrist"] and "x" in kp_idx["mid_hip"]:
            w_idx = kp_idx["right_wrist"]["x"]
            h_idx = kp_idx["mid_hip"]["x"]
            features["lateral_offset"] = ts[release_frame, w_idx] - ts[release_frame, h_idx]

    # 7. SHOULDER ROTATION (difference between shoulders in X)
    if "right_shoulder" in kp_idx and "left_shoulder" in kp_idx:
        if "x" in kp_idx["right_shoulder"] and "x" in kp_idx["left_shoulder"]:
            r_idx = kp_idx["right_shoulder"]["x"]
            l_idx = kp_idx["left_shoulder"]["x"]
            features["shoulder_rotation"] = ts[release_frame, r_idx] - ts[release_frame, l_idx]

    # 8. HIP ALIGNMENT
    if "right_hip" in kp_idx and "left_hip" in kp_idx:
        if "x" in kp_idx["right_hip"] and "x" in kp_idx["left_hip"]:
            r_idx = kp_idx["right_hip"]["x"]
            l_idx = kp_idx["left_hip"]["x"]
            features["hip_alignment"] = ts[release_frame, r_idx] - ts[release_frame, l_idx]

    # 9. GUIDE HAND position (left wrist)
    if "left_wrist" in kp_idx:
        for coord in ["x", "y", "z"]:
            if coord in kp_idx["left_wrist"]:
                idx = kp_idx["left_wrist"][coord]
                features[f"guide_hand_{coord}"] = ts[release_frame, idx]

    # 10. ELBOW LIFT (elbow Y relative to shoulder Y)
    if "right_elbow" in kp_idx and "right_shoulder" in kp_idx:
        if "y" in kp_idx["right_elbow"] and "y" in kp_idx["right_shoulder"]:
            e_idx = kp_idx["right_elbow"]["y"]
            s_idx = kp_idx["right_shoulder"]["y"]
            features["elbow_lift"] = ts[release_frame, e_idx] - ts[release_frame, s_idx]

    return features


def extract_raw_features(ts, frames):
    """Extract raw flattened features for a frame window."""
    flat = ts[frames, :].flatten()
    return np.nan_to_num(flat, nan=0.0)


def multi_model_approach(data):
    """
    Multi-model combination approach.

    For each target:
    1. Compute intermediate physical features
    2. Train models to predict these intermediate features from raw data
    3. Use predicted intermediate features to predict final target
    """
    print("\n" + "="*60)
    print("MULTI-MODEL COMBINATION APPROACH")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]
    kp_idx = data["kp_idx"]

    ranges = get_target_ranges()

    # Step 1: Compute intermediate features for all samples
    print("\nComputing intermediate physical features...")

    train_intermediate = []
    for ts in tqdm(train_series, desc="Train"):
        train_intermediate.append(compute_intermediate_features(ts, kp_idx))

    test_intermediate = []
    for ts in tqdm(test_series, desc="Test"):
        test_intermediate.append(compute_intermediate_features(ts, kp_idx))

    # Convert to arrays
    feature_names = sorted(train_intermediate[0].keys())
    print(f"  Intermediate features: {feature_names}")

    train_inter_arr = np.array([[f.get(n, 0) for n in feature_names] for f in train_intermediate])
    test_inter_arr = np.array([[f.get(n, 0) for n in feature_names] for f in test_intermediate])

    train_inter_arr = np.nan_to_num(train_inter_arr, nan=0.0)
    test_inter_arr = np.nan_to_num(test_inter_arr, nan=0.0)

    # Step 2: Check correlations between intermediate features and targets
    print("\nCorrelations between intermediate features and targets:")
    for t_idx, target in enumerate(TARGETS):
        print(f"\n  {target}:")
        correlations = []
        for f_idx, fname in enumerate(feature_names):
            corr, _ = pearsonr(train_inter_arr[:, f_idx], train_targets[:, t_idx])
            correlations.append((fname, corr))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for fname, corr in correlations[:5]:
            print(f"    {fname}: r={corr:.3f}")

    # Step 3: Prepare raw features
    frames = list(range(140, 170))

    def get_raw_features(series):
        return np.array([extract_raw_features(ts, frames) for ts in series])

    train_raw = get_raw_features(train_series)
    test_raw = get_raw_features(test_series)

    unique_pids = sorted(np.unique(train_pids))

    # Step 4: For each target, train multi-stage model
    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for t_idx, target in enumerate(TARGETS):
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print("="*60)

        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_raw_train = train_raw[train_mask]
            X_inter_train = train_inter_arr[train_mask]
            y_train = train_targets[train_mask, t_idx]

            X_raw_test = test_raw[test_mask]
            X_inter_test = test_inter_arr[test_mask]

            player_indices = np.where(train_mask)[0]

            # PCA on raw features
            n_comp_raw = min(20, len(X_raw_train) - 2)
            pca_raw = PCA(n_components=n_comp_raw)
            X_raw_train_pca = pca_raw.fit_transform(X_raw_train)
            X_raw_test_pca = pca_raw.transform(X_raw_test)

            # Combine: raw PCA + intermediate features
            X_train_combined = np.hstack([X_raw_train_pca, X_inter_train])
            X_test_combined = np.hstack([X_raw_test_pca, X_inter_test])

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_combined)
            X_test_scaled = scaler.transform(X_test_combined)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx]

                # Train model on combined features
                model = Ridge(alpha=50.0)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target} CV: {scaled_mse:.6f}")

    # Final CV
    total_mse = 0
    print("\n" + "="*60)
    print("FINAL CV")
    print("="*60)
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"\n  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, oof_preds


def multi_model_staged(data):
    """
    Staged multi-model: predict intermediate features first,
    then use those predictions to predict targets.
    """
    print("\n" + "="*60)
    print("STAGED MULTI-MODEL (predict intermediates first)")
    print("="*60)

    train_series = data["train_series"]
    train_targets = data["train_targets"]
    train_pids = data["train_pids"]
    test_series = data["test_series"]
    test_pids = data["test_pids"]
    kp_idx = data["kp_idx"]

    ranges = get_target_ranges()

    # Compute actual intermediate features (these are our auxiliary targets)
    print("\nComputing intermediate features...")

    train_intermediate = []
    for ts in tqdm(train_series, desc="Train"):
        train_intermediate.append(compute_intermediate_features(ts, kp_idx))

    test_intermediate = []
    for ts in tqdm(test_series, desc="Test"):
        test_intermediate.append(compute_intermediate_features(ts, kp_idx))

    feature_names = sorted(train_intermediate[0].keys())

    train_inter_arr = np.array([[f.get(n, 0) for n in feature_names] for f in train_intermediate])
    test_inter_arr = np.array([[f.get(n, 0) for n in feature_names] for f in test_intermediate])

    train_inter_arr = np.nan_to_num(train_inter_arr, nan=0.0)
    test_inter_arr = np.nan_to_num(test_inter_arr, nan=0.0)

    # Prepare raw features
    frames = list(range(140, 170))

    def get_raw_features(series):
        return np.array([extract_raw_features(ts, frames) for ts in series])

    train_raw = get_raw_features(train_series)
    test_raw = get_raw_features(test_series)

    unique_pids = sorted(np.unique(train_pids))

    # Stage 1: Predict intermediate features from raw data
    print("\nStage 1: Predicting intermediate features...")

    train_inter_pred = np.zeros_like(train_inter_arr)
    test_inter_pred = np.zeros_like(test_inter_arr)

    for pid in unique_pids:
        train_mask = train_pids == pid
        test_mask = test_pids == pid

        X_raw_train = train_raw[train_mask]
        Y_inter_train = train_inter_arr[train_mask]
        X_raw_test = test_raw[test_mask]

        player_indices = np.where(train_mask)[0]

        n_comp = min(20, len(X_raw_train) - 2)
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_raw_train)
        X_test_pca = pca.transform(X_raw_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for f_idx in range(len(feature_names)):
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = Y_inter_train[train_idx, f_idx]

                model = Ridge(alpha=10.0)
                model.fit(X_tr, y_tr)

                train_inter_pred[player_indices[val_idx], f_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            test_inter_pred[test_mask, f_idx] = np.mean(fold_test_preds, axis=0)

    # Stage 2: Predict targets from predicted intermediate features
    print("\nStage 2: Predicting targets from intermediate predictions...")

    predictions = np.zeros((len(test_series), 3))
    oof_preds = np.zeros_like(train_targets)

    for t_idx, target in enumerate(TARGETS):
        for pid in unique_pids:
            train_mask = train_pids == pid
            test_mask = test_pids == pid

            X_train = train_inter_pred[train_mask]
            y_train = train_targets[train_mask, t_idx]
            X_test = test_inter_pred[test_mask]

            player_indices = np.where(train_mask)[0]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_test_preds = []

            for train_idx, val_idx in kf.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr = y_train[train_idx]

                model = Ridge(alpha=10.0)
                model.fit(X_tr, y_tr)

                oof_preds[player_indices[val_idx], t_idx] = model.predict(X_val)
                fold_test_preds.append(model.predict(X_test_scaled))

            predictions[test_mask, t_idx] = np.mean(fold_test_preds, axis=0)

    # Final CV
    total_mse = 0
    print("\n" + "="*60)
    print("STAGED MODEL CV")
    print("="*60)
    for t_idx, target in enumerate(TARGETS):
        mse = np.mean((oof_preds[:, t_idx] - train_targets[:, t_idx])**2)
        scaled_mse = mse / (ranges[target]**2)
        print(f"  {target}: {scaled_mse:.6f}")
        total_mse += scaled_mse

    cv_score = total_mse / 3
    print(f"\n  TOTAL CV: {cv_score:.6f}")

    return predictions, cv_score, oof_preds


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

    print(f"\nSubmission {next_num}: {approach_name}")
    print(f"  CV Score: {cv_score:.6f}")
    print(f"  angle_std: {angle_std:.4f}")

    return filepath, next_num


def main():
    print("="*80)
    print("MULTI-MODEL COMBINATION")
    print("="*80)
    print()
    print("Idea: Predict intermediate physical quantities first,")
    print("then combine them to predict final targets.")

    data = load_data()

    results = []

    # Approach 1: Combined features (raw + intermediate)
    preds1, cv1, _ = multi_model_approach(data)
    results.append(("combined", cv1, preds1))

    # Approach 2: Staged prediction
    preds2, cv2, _ = multi_model_staged(data)
    results.append(("staged", cv2, preds2))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Approach':<20} {'CV Score':>12}")
    print("-" * 35)
    for name, cv, _ in sorted(results, key=lambda x: x[1]):
        print(f"{name:<20} {cv:>12.6f}")

    # Create submission with best
    best_name, best_cv, best_preds = min(results, key=lambda x: x[1])
    print(f"\nBest approach: {best_name}")

    create_submission(data["test_ids"], best_preds, best_cv, f"multi_model_{best_name}")


if __name__ == "__main__":
    main()
