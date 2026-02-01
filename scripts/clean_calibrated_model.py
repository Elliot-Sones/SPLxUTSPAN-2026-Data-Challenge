"""
Clean Calibrated Model

Use a simpler, cleaner feature set to get a better base model,
then apply calibration to match Sub 133's profile.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy import stats
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def smooth_ts(ts, window=5):
    """Smooth timeseries."""
    smoothed = np.zeros_like(ts)
    for col in range(ts.shape[1]):
        smoothed[:, col] = np.convolve(ts[:, col], np.ones(window)/window, mode='same')
    return smoothed


def extract_clean_features(timeseries, keypoint_idx):
    """Extract a clean, focused feature set."""
    ts = smooth_ts(timeseries, window=5)
    features = {}

    # Key joints for basketball shooting
    joints = ["right_wrist", "right_elbow", "right_shoulder", "right_hip", "neck"]

    # Key frames around release (153)
    frames = [148, 150, 153, 155, 158]

    for joint in joints:
        if joint not in keypoint_idx:
            continue
        idx = keypoint_idx[joint]

        for f in frames:
            if f < len(ts):
                pos = ts[f, idx*3:(idx+1)*3]
                # Only z-coordinate (depth) is most relevant
                features[f"{joint}_z_f{f}"] = pos[2]

                # Add y-coordinate (vertical) for shooting motion
                features[f"{joint}_y_f{f}"] = pos[1]

        # Velocity at release
        if 153 < len(ts) - 1:
            vel = (ts[154, idx*3:(idx+1)*3] - ts[152, idx*3:(idx+1)*3]) / 2
            features[f"{joint}_vel_y_release"] = vel[1]
            features[f"{joint}_vel_z_release"] = vel[2]

    # Key distance: wrist to shoulder (arm extension)
    if "right_wrist" in keypoint_idx and "right_shoulder" in keypoint_idx:
        w_idx = keypoint_idx["right_wrist"]
        s_idx = keypoint_idx["right_shoulder"]

        for f in frames:
            if f < len(ts):
                wrist = ts[f, w_idx*3:(w_idx+1)*3]
                shoulder = ts[f, s_idx*3:(s_idx+1)*3]
                features[f"arm_extension_f{f}"] = np.linalg.norm(wrist - shoulder)

    return features


def quantile_calibrate(predictions, reference):
    """Map predictions to reference distribution."""
    # Get ranks of predictions
    ranks = stats.rankdata(predictions, method='average') / (len(predictions) + 1)

    # Get quantiles from reference
    ref_sorted = np.sort(reference)
    ref_quantiles = np.linspace(0, 1, len(ref_sorted))

    # Interpolate
    calibrated = np.interp(ranks, ref_quantiles, ref_sorted)
    return calibrated


def main():
    print("=" * 70)
    print("CLEAN CALIBRATED MODEL")
    print("=" * 70)

    scalers_dict = load_scalers()
    keypoint_idx = get_keypoint_indices()

    # Build feature matrices
    print("\nExtracting clean features...")

    train_features = []
    train_targets = []
    train_pids = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_clean_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        train_pids.append(metadata["participant_id"])

    train_df = pd.DataFrame(train_features).fillna(0)
    train_df = train_df.loc[:, train_df.nunique() > 1]

    X_train = train_df.values
    y_train = np.array(train_targets)
    pids_train = np.array(train_pids)

    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    test_features = []
    test_pids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_clean_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_pids.append(metadata["participant_id"])

    test_df = pd.DataFrame(test_features).fillna(0)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    X_test = test_df[train_df.columns].values
    pids_test = np.array(test_pids)

    print(f"Test: {X_test.shape[0]} samples")

    target_names = ["angle", "depth", "left_right"]

    # Load Sub 133 as reference
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    # Train model with within-player CV
    print("\nTraining with within-player approach...")

    test_preds = np.zeros((len(X_test), 3))
    cv_preds = np.zeros_like(y_train)

    for pid in np.unique(pids_train):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            ridge = Ridge(alpha=50)  # Less regularization for cleaner features
            ridge.fit(X_tr_scaled, y_tr[:, t_idx])

            test_preds[test_mask, t_idx] = ridge.predict(X_te_scaled)

            # CV for this player
            indices = np.where(train_mask)[0]
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, te_idx in kf.split(X_tr_scaled):
                ridge_cv = Ridge(alpha=50)
                ridge_cv.fit(X_tr_scaled[tr_idx], y_tr[tr_idx, t_idx])
                cv_preds[indices[te_idx], t_idx] = ridge_cv.predict(X_tr_scaled[te_idx])

    # Scale predictions
    scaled_test = np.zeros_like(test_preds)
    scaled_cv = np.zeros_like(cv_preds)
    scaled_targets = np.zeros_like(y_train)

    for i, target in enumerate(target_names):
        scaled_test[:, i] = scalers_dict[target].transform(test_preds[:, i].reshape(-1, 1)).ravel()
        scaled_cv[:, i] = scalers_dict[target].transform(cv_preds[:, i].reshape(-1, 1)).ravel()
        scaled_targets[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

    # CV metrics
    print("\nBase model CV MSE (per target):")
    for i, target in enumerate(target_names):
        mse = np.mean((scaled_cv[:, i] - scaled_targets[:, i]) ** 2)
        print(f"  {target}: {mse:.6f}")

    total_mse = np.mean([(scaled_cv[:, i] - scaled_targets[:, i]) ** 2 for i in range(3)])
    print(f"Total CV MSE: {total_mse:.6f}")

    print("\nBase model test profile:")
    print(f"  angle_std: {scaled_test[:, 0].std():.4f} (target: 0.1377)")
    print(f"  depth_mean: {scaled_test[:, 1].mean():.4f} (target: 0.5055)")

    # Apply quantile calibration to match Sub 133
    print("\n" + "=" * 70)
    print("QUANTILE CALIBRATION TO SUB 133")
    print("=" * 70)

    calibrated_test = np.zeros_like(scaled_test)
    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    for i, col in enumerate(cols):
        calibrated_test[:, i] = quantile_calibrate(scaled_test[:, i], sub133[col].values)

    print("\nCalibrated profile:")
    print(f"  angle_std: {calibrated_test[:, 0].std():.4f}")
    print(f"  depth_mean: {calibrated_test[:, 1].mean():.4f}")

    # Correlation with Sub 133
    print("\nCorrelation with Sub 133:")
    for i, col in enumerate(cols):
        corr = np.corrcoef(calibrated_test[:, i], sub133[col])[0, 1]
        print(f"  {col}: {corr:.4f}")

    # Also try blending with Sub 133
    print("\n" + "=" * 70)
    print("BLEND WITH SUB 133")
    print("=" * 70)

    best_mse = float("inf")
    best_weight = 0
    best_blend = None

    for w in np.linspace(0, 0.5, 21):
        blended = (1 - w) * sub133[cols].values + w * calibrated_test

        # Use CV MSE as proxy (won't match exactly but shows trend)
        # Can't compute real test MSE, so use profile matching
        angle_std = blended[:, 0].std()
        depth_mean = blended[:, 1].mean()
        profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

        if profile_dist < best_mse:
            best_mse = profile_dist
            best_weight = w
            best_blend = blended.copy()

    print(f"Best weight for calibrated model: {best_weight:.2f}")
    print(f"Profile dist: {best_mse:.6f}")

    # Save submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    test_meta = load_metadata(train=False)
    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    # Save base model (uncalibrated clean features)
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": scaled_test[:, 0],
        "scaled_depth": scaled_test[:, 1],
        "scaled_left_right": scaled_test[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved clean base model: {path}")
    next_num += 1

    # Save calibrated model
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": calibrated_test[:, 0],
        "scaled_depth": calibrated_test[:, 1],
        "scaled_left_right": calibrated_test[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved clean calibrated model: {path}")

    # Save blend
    if best_weight > 0:
        next_num += 1
        submission = pd.DataFrame({
            "id": test_meta["id"].values,
            "scaled_angle": best_blend[:, 0],
            "scaled_depth": best_blend[:, 1],
            "scaled_left_right": best_blend[:, 2]
        })
        path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(path, index=False)
        print(f"Saved blend with Sub 133: {path} (weight={best_weight:.2f})")


if __name__ == "__main__":
    main()
