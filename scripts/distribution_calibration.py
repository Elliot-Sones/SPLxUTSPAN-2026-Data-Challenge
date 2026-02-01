"""
Distribution Calibration

Key insight: Sub 133 predicts depth_mean = 0.5055 while training mean = 0.5157.
The successful models are systematically biased lower.

Approaches:
1. Quantile matching - match prediction quantiles to target distribution
2. Shift calibration - learn optimal bias shift
3. Per-player calibration - calibrate separately for each player
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from scipy import stats
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def build_features(train=True):
    """Build feature matrix."""
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    feature_list = []
    targets = []
    pids = []

    for metadata, timeseries in iterate_shots(train):
        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        feature_list.append(features)
        pids.append(metadata["participant_id"])
        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    df = pd.DataFrame(feature_list).fillna(0)
    df = df.loc[:, df.nunique() > 1]

    return df.values, np.array(targets) if train else None, np.array(pids), df.columns.tolist()


def quantile_calibrate(predictions, target_quantiles, pred_quantiles):
    """Calibrate predictions to match target quantile distribution."""
    # Sort predictions and compute their ranks
    ranks = stats.rankdata(predictions, method='average') / (len(predictions) + 1)

    # Interpolate target values at those ranks
    calibrated = np.interp(ranks, pred_quantiles, target_quantiles)
    return calibrated


def shift_calibrate(predictions, target_mean, target_std):
    """Simple shift and scale calibration."""
    pred_mean = predictions.mean()
    pred_std = predictions.std()

    # Standardize then rescale
    calibrated = (predictions - pred_mean) / (pred_std + 1e-8)
    calibrated = calibrated * target_std + target_mean
    return calibrated


def main():
    print("=" * 70)
    print("DISTRIBUTION CALIBRATION")
    print("=" * 70)

    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    # Load training data
    print("\nLoading data...")
    X_train, y_train, pids_train, feature_names = build_features(True)
    X_test, _, pids_test, _ = build_features(False)

    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test: {X_test.shape[0]} samples")

    # Reference: Sub 133 profile
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    target_profiles = {
        "angle": {"mean": sub133["scaled_angle"].mean(), "std": sub133["scaled_angle"].std()},
        "depth": {"mean": sub133["scaled_depth"].mean(), "std": sub133["scaled_depth"].std()},
        "left_right": {"mean": sub133["scaled_left_right"].mean(), "std": sub133["scaled_left_right"].std()}
    }

    print("\nTarget profiles (from Sub 133):")
    for target, profile in target_profiles.items():
        print(f"  {target}: mean={profile['mean']:.4f}, std={profile['std']:.4f}")

    # Train base model and get predictions
    print("\nTraining base model...")
    base_preds_train = np.zeros_like(y_train)
    base_preds_test = np.zeros((len(X_test), 3))

    for pid in np.unique(pids_train):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train[train_mask])
        X_te_scaled = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            y_tr = y_train[train_mask, t_idx]

            ridge = Ridge(alpha=100)
            ridge.fit(X_tr_scaled, y_tr)

            base_preds_train[train_mask, t_idx] = ridge.predict(X_tr_scaled)
            base_preds_test[test_mask, t_idx] = ridge.predict(X_te_scaled)

    # Scale predictions
    scaled_train = np.zeros_like(base_preds_train)
    scaled_test = np.zeros_like(base_preds_test)
    scaled_targets = np.zeros_like(y_train)

    for i, target in enumerate(target_names):
        scaled_train[:, i] = scalers_dict[target].transform(base_preds_train[:, i].reshape(-1, 1)).ravel()
        scaled_test[:, i] = scalers_dict[target].transform(base_preds_test[:, i].reshape(-1, 1)).ravel()
        scaled_targets[:, i] = scalers_dict[target].transform(y_train[:, i].reshape(-1, 1)).ravel()

    print("\nBase model profile:")
    for i, target in enumerate(target_names):
        print(f"  {target}: mean={scaled_test[:, i].mean():.4f}, std={scaled_test[:, i].std():.4f}")

    # Approach 1: Shift calibration (match Sub 133 profile)
    print("\n" + "=" * 70)
    print("APPROACH 1: SHIFT CALIBRATION")
    print("=" * 70)

    shifted_test = np.zeros_like(scaled_test)
    for i, target in enumerate(target_names):
        shifted_test[:, i] = shift_calibrate(
            scaled_test[:, i],
            target_profiles[target]["mean"],
            target_profiles[target]["std"]
        )

    print("Shifted profile:")
    for i, target in enumerate(target_names):
        print(f"  {target}: mean={shifted_test[:, i].mean():.4f}, std={shifted_test[:, i].std():.4f}")

    # Approach 2: Quantile calibration using training distribution
    print("\n" + "=" * 70)
    print("APPROACH 2: QUANTILE CALIBRATION")
    print("=" * 70)

    # Use Sub 133 predictions as the target distribution
    quantile_test = np.zeros_like(scaled_test)
    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    for i, col in enumerate(cols):
        target_sorted = np.sort(sub133[col].values)
        target_quantiles = np.linspace(0, 1, len(target_sorted))

        # Map test predictions to these quantiles
        quantile_test[:, i] = quantile_calibrate(
            scaled_test[:, i],
            target_sorted,
            target_quantiles
        )

    print("Quantile-calibrated profile:")
    for i, target in enumerate(target_names):
        print(f"  {target}: mean={quantile_test[:, i].mean():.4f}, std={quantile_test[:, i].std():.4f}")

    # Approach 3: Blend base + shifted with optimal weights
    print("\n" + "=" * 70)
    print("APPROACH 3: OPTIMAL BLEND (base + shifted)")
    print("=" * 70)

    # Test different blend weights
    best_blend = None
    best_profile_dist = float("inf")
    best_weight = 0

    for w in np.linspace(0, 1, 21):
        blended = (1 - w) * scaled_test + w * shifted_test

        angle_std = blended[:, 0].std()
        depth_mean = blended[:, 1].mean()
        profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

        if profile_dist < best_profile_dist:
            best_profile_dist = profile_dist
            best_blend = blended.copy()
            best_weight = w

    print(f"Optimal weight: {best_weight:.2f}")
    print("Blended profile:")
    for i, target in enumerate(target_names):
        print(f"  {target}: mean={best_blend[:, i].mean():.4f}, std={best_blend[:, i].std():.4f}")

    # Approach 4: Per-player bias correction
    print("\n" + "=" * 70)
    print("APPROACH 4: PER-PLAYER BIAS CORRECTION")
    print("=" * 70)

    # Learn bias per player from training data
    corrected_test = scaled_test.copy()

    for pid in np.unique(pids_train):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        for i, target in enumerate(target_names):
            # Compute prediction bias on training data
            pred_train_player = scaled_train[train_mask, i]
            true_train_player = scaled_targets[train_mask, i]

            bias = np.mean(pred_train_player - true_train_player)

            # Correct test predictions
            corrected_test[test_mask, i] = scaled_test[test_mask, i] - bias

    print("Per-player corrected profile:")
    for i, target in enumerate(target_names):
        print(f"  {target}: mean={corrected_test[:, i].mean():.4f}, std={corrected_test[:, i].std():.4f}")

    # Compare all approaches
    print("\n" + "=" * 70)
    print("COMPARISON WITH SUB 133")
    print("=" * 70)

    approaches = [
        ("Base (uncalibrated)", scaled_test),
        ("Shift calibrated", shifted_test),
        ("Quantile calibrated", quantile_test),
        ("Optimal blend", best_blend),
        ("Per-player corrected", corrected_test),
    ]

    for name, preds in approaches:
        # Profile distance
        angle_std = preds[:, 0].std()
        depth_mean = preds[:, 1].mean()
        profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

        # Correlation with Sub 133
        corr = np.corrcoef(preds.ravel(), sub133[cols].values.ravel())[0, 1]

        print(f"\n{name}:")
        print(f"  Profile dist: {profile_dist:.4f}")
        print(f"  Corr with Sub 133: {corr:.4f}")

    # Save best submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    test_meta = load_metadata(train=False)
    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    submissions_to_save = [
        ("per_player_corrected", corrected_test),
        ("quantile_calibrated", quantile_test),
    ]

    for name, preds in submissions_to_save:
        submission = pd.DataFrame({
            "id": test_meta["id"].values,
            "scaled_angle": preds[:, 0],
            "scaled_depth": preds[:, 1],
            "scaled_left_right": preds[:, 2]
        })

        path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(path, index=False)
        print(f"Saved {name}: {path}")
        next_num += 1


if __name__ == "__main__":
    main()
