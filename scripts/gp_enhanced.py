"""
Enhanced Gaussian Process Model

Improvements over baseline GP:
1. Per-player GP models (like our best submissions)
2. More features with stability filtering
3. Kernel optimization
4. Within-player CV evaluation
5. Profile validation before submission
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, DotProduct
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Load stable features
def get_stable_features(n_features=50):
    """Get top N stable features from drift analysis."""
    drift_file = OUTPUT_DIR / "feature_drift_f4_with_importance.csv"
    if drift_file.exists():
        drift_df = pd.read_csv(drift_file)
        # Sort by stability_adjusted (higher is better)
        drift_df = drift_df.sort_values("stability_adjusted", ascending=False)
        return drift_df["feature"].head(n_features).tolist()
    return None


def build_feature_matrix(train: bool = True, feature_names: list = None):
    """Build feature matrix with specified features."""
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    meta_df = load_metadata(train)
    n_shots = len(meta_df)

    all_features = []
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i % 100 == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    feature_df = pd.DataFrame(all_features)
    feature_df = feature_df.fillna(feature_df.median())

    # Filter to specified features if provided
    if feature_names is not None:
        available = [f for f in feature_names if f in feature_df.columns]
        print(f"  Using {len(available)} of {len(feature_names)} requested features")
        feature_df = feature_df[available]

    # Drop constant columns
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def evaluate_within_player_cv(X, y, pids, kernel_type="rbf", alpha=0.1, n_splits=5):
    """Evaluate GP with within-player CV (more predictive of LB)."""
    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask]

        kf = KFold(n_splits=min(n_splits, len(X_pid)), shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_pid):
            X_train, X_test = X_pid[train_idx], X_pid[test_idx]
            y_train, y_test = y_pid[train_idx], y_pid[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for target_idx, target_name in enumerate(target_names):
                # Create kernel
                if kernel_type == "rbf":
                    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                elif kernel_type == "matern":
                    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
                elif kernel_type == "rbf_linear":
                    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
                else:
                    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    n_restarts_optimizer=2,
                    random_state=42
                )
                gp.fit(X_train_scaled, y_train[:, target_idx])

                # Get indices in original array
                orig_test_idx = np.where(mask)[0][test_idx]
                all_preds[orig_test_idx, target_idx] = gp.predict(X_test_scaled)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(target_names):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def evaluate_lopo_cv(X, y, pids, kernel_type="rbf", alpha=0.1):
    """LOPO CV evaluation."""
    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for test_pid in unique_pids:
        train_mask = pids != test_pid
        test_mask = pids == test_pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for target_idx, target_name in enumerate(target_names):
            if kernel_type == "rbf":
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            elif kernel_type == "matern":
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
            elif kernel_type == "rbf_linear":
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                n_restarts_optimizer=2,
                random_state=42
            )
            gp.fit(X_train_scaled, y_train[:, target_idx])
            all_preds[test_mask, target_idx] = gp.predict(X_test_scaled)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(target_names):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def create_submission(X_train, y_train, pids_train, X_test, pids_test,
                      kernel_type="rbf", alpha=0.1, submission_num=137):
    """Create submission with per-player GP models."""
    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    # Train per-player models
    unique_train_pids = np.unique(pids_train)
    unique_test_pids = np.unique(pids_test)

    predictions = np.zeros((len(X_test), 3))

    for test_pid in unique_test_pids:
        test_mask = pids_test == test_pid
        X_test_pid = X_test[test_mask]

        # Check if player exists in training
        if test_pid in unique_train_pids:
            # Use same-player data for training
            train_mask = pids_train == test_pid
            X_train_pid = X_train[train_mask]
            y_train_pid = y_train[train_mask]
        else:
            # Use all training data
            X_train_pid = X_train
            y_train_pid = y_train

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pid)
        X_test_scaled = scaler.transform(X_test_pid)

        for target_idx, target_name in enumerate(target_names):
            if kernel_type == "rbf":
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            elif kernel_type == "rbf_linear":
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                n_restarts_optimizer=3,
                random_state=42
            )
            gp.fit(X_train_scaled, y_train_pid[:, target_idx])
            predictions[test_mask, target_idx] = gp.predict(X_test_scaled)

    # Check profile
    angle_std = np.std(predictions[:, 0])
    depth_mean = np.mean(predictions[:, 1])
    lr_mean = np.mean(predictions[:, 2])

    print(f"\nSubmission profile:")
    print(f"  angle_std: {angle_std:.4f} (target: < 0.145)")
    print(f"  depth_mean: {depth_mean:.4f} (target: 0.49-0.52)")
    print(f"  lr_mean: {lr_mean:.4f}")

    # Profile check - compare with Sub 133 which has LB 0.007809
    profile_ok = True
    if angle_std > 1.0:  # Relaxed check for now
        print(f"  WARNING: angle_std too high")
        profile_ok = False

    # Create submission
    test_meta = load_metadata(train=False)
    submission_df = pd.DataFrame({
        "id": test_meta["id"].values,
        "angle": predictions[:, 0],
        "depth": predictions[:, 1],
        "left_right": predictions[:, 2]
    })

    submission_path = SUBMISSION_DIR / f"submission_{submission_num}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    return submission_path, profile_ok


def main():
    print("=" * 60)
    print("ENHANCED GAUSSIAN PROCESS MODEL")
    print("=" * 60)

    # Test different feature counts
    results = []

    for n_features in [30, 50, 80]:
        print(f"\n--- Testing with {n_features} stable features ---")

        stable_features = get_stable_features(n_features)
        if stable_features is None:
            print("Could not load stable features")
            continue

        print("Building feature matrix...")
        X, y, pids, feature_names = build_feature_matrix(train=True, feature_names=stable_features)
        print(f"X shape: {X.shape}")

        for kernel_type in ["rbf", "rbf_linear"]:
            for alpha in [0.05, 0.1, 0.2]:
                print(f"\n  kernel={kernel_type}, alpha={alpha}")

                # Within-player CV (more predictive of LB)
                wp_mse, wp_per_target = evaluate_within_player_cv(
                    X, y, pids, kernel_type=kernel_type, alpha=alpha
                )

                # LOPO CV
                lopo_mse, lopo_per_target = evaluate_lopo_cv(
                    X, y, pids, kernel_type=kernel_type, alpha=alpha
                )

                results.append({
                    "n_features": n_features,
                    "kernel": kernel_type,
                    "alpha": alpha,
                    "within_player_mse": wp_mse,
                    "lopo_mse": lopo_mse,
                    "wp_angle": wp_per_target["angle"],
                    "wp_depth": wp_per_target["depth"],
                    "wp_lr": wp_per_target["left_right"],
                })

                print(f"    Within-player MSE: {wp_mse:.6f}")
                print(f"    LOPO MSE: {lopo_mse:.6f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "gp_enhanced_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'gp_enhanced_results.csv'}")

    # Find best config by within-player CV
    best_idx = results_df["within_player_mse"].idxmin()
    best = results_df.iloc[best_idx]

    print(f"\n{'=' * 60}")
    print("BEST CONFIGURATION (by within-player CV)")
    print(f"{'=' * 60}")
    print(f"n_features: {best['n_features']}")
    print(f"kernel: {best['kernel']}")
    print(f"alpha: {best['alpha']}")
    print(f"Within-player MSE: {best['within_player_mse']:.6f}")
    print(f"LOPO MSE: {best['lopo_mse']:.6f}")

    # Create submission with best config
    print(f"\n{'=' * 60}")
    print("CREATING SUBMISSION")
    print(f"{'=' * 60}")

    stable_features = get_stable_features(int(best['n_features']))

    print("\nBuilding train features...")
    X_train, y_train, pids_train, _ = build_feature_matrix(train=True, feature_names=stable_features)

    print("\nBuilding test features...")
    X_test, _, pids_test, _ = build_feature_matrix(train=False, feature_names=stable_features)

    # Find next submission number
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split("_")[1]) for f in existing]
        next_num = max(nums) + 1
    else:
        next_num = 137

    create_submission(
        X_train, y_train, pids_train,
        X_test, pids_test,
        kernel_type=best['kernel'],
        alpha=best['alpha'],
        submission_num=next_num
    )


if __name__ == "__main__":
    main()
