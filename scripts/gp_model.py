"""
Gaussian Processes with ARD Kernel (Plan 2.1)

GPs provide uncertainty estimates + automatic feature relevance.
Use uncertainty to weight predictions or identify unreliable predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers
from data_loader import get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Check for sklearn GP
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    print("sklearn GaussianProcess not available")


def load_stable_features(top_n: int = 30) -> list:
    """Load top N stable features."""
    drift_df = pd.read_csv(OUTPUT_DIR / "feature_drift_f4_with_importance.csv")
    drift_df = drift_df.sort_values("stability_adjusted", ascending=False)
    return drift_df.head(top_n)["feature"].tolist()


def build_feature_matrix(train: bool = True, selected_features: list = None):
    """Build feature matrix."""
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    meta_df = load_metadata(train)
    n_shots = len(meta_df)

    all_features = []
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        features = extract_hybrid_features(timeseries)
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    feature_df = pd.DataFrame(all_features)

    if selected_features:
        available = [f for f in selected_features if f in feature_df.columns]
        print(f"  Using {len(available)} of {len(selected_features)} features")
        feature_df = feature_df[available]

    feature_df = feature_df.fillna(feature_df.median())

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def evaluate_gp_lopo(X, y, pids, kernel_type="rbf", alpha=1e-2):
    """LOPO CV with Gaussian Process."""
    if not GP_AVAILABLE:
        return float("inf"), {}, None, None

    scalers_dict = load_scalers()
    unique_pids = np.unique(pids)

    all_preds = np.zeros_like(y)
    all_stds = np.zeros_like(y)  # Prediction uncertainty

    for pid in unique_pids:
        print(f"  Training GP for participant {pid}...")

        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
            # Scale target
            y_scaler = scalers_dict[target_name]
            y_train_scaled = y_scaler.transform(y_train[:, target_idx].reshape(-1, 1)).ravel()

            # Define kernel with ARD (Automatic Relevance Determination)
            n_features = X_train_scaled.shape[1]

            if kernel_type == "rbf":
                kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features)) + WhiteKernel(noise_level=0.1)
            elif kernel_type == "matern":
                kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=2.5) + WhiteKernel(noise_level=0.1)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

            # Train GP
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                n_restarts_optimizer=5,
                random_state=42,
                normalize_y=True
            )

            try:
                gp.fit(X_train_scaled, y_train_scaled)

                # Predict with uncertainty
                pred_scaled, std_scaled = gp.predict(X_test_scaled, return_std=True)

                # Inverse scale predictions
                all_preds[test_mask, target_idx] = y_scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).ravel()

                # Store uncertainty (in scaled space)
                all_stds[test_mask, target_idx] = std_scaled

            except Exception as e:
                print(f"    GP failed for {target_name}: {e}")
                # Fallback to mean prediction
                all_preds[test_mask, target_idx] = np.mean(y_train[:, target_idx])
                all_stds[test_mask, target_idx] = 1.0

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target, all_preds, all_stds


def evaluate_uncertainty_calibration(y, preds, stds, scalers_dict):
    """Check if GP uncertainty is well-calibrated."""
    results = {}

    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]

        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(preds[:, i].reshape(-1, 1)).ravel()
        std = stds[:, i]

        # Absolute error
        abs_error = np.abs(y_scaled - pred_scaled)

        # Correlation between uncertainty and error
        corr = np.corrcoef(std, abs_error)[0, 1]

        # Are high-uncertainty predictions actually worse?
        median_std = np.median(std)
        low_uncertainty_mask = std < median_std
        high_uncertainty_mask = std >= median_std

        mse_low_uncertainty = np.mean((y_scaled[low_uncertainty_mask] - pred_scaled[low_uncertainty_mask]) ** 2)
        mse_high_uncertainty = np.mean((y_scaled[high_uncertainty_mask] - pred_scaled[high_uncertainty_mask]) ** 2)

        results[target] = {
            "std_error_corr": corr,
            "mse_low_uncertainty": mse_low_uncertainty,
            "mse_high_uncertainty": mse_high_uncertainty,
            "uncertainty_ratio": mse_high_uncertainty / max(mse_low_uncertainty, 1e-6)
        }

    return results


def evaluate_gp_per_player(X, y, pids, kernel_type="rbf", alpha=1e-2, n_folds=5):
    """Per-player GP with internal CV."""
    if not GP_AVAILABLE:
        return float("inf"), {}

    scalers_dict = load_scalers()
    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask]

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_pid):
            X_train, X_val = X_pid[train_idx], X_pid[val_idx]
            y_train = y_pid[train_idx]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
                y_scaler = scalers_dict[target_name]
                y_train_scaled = y_scaler.transform(y_train[:, target_idx].reshape(-1, 1)).ravel()

                n_features = X_train_scaled.shape[1]
                kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features)) + WhiteKernel(noise_level=0.1)

                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    n_restarts_optimizer=3,
                    random_state=42,
                    normalize_y=True
                )

                try:
                    gp.fit(X_train_scaled, y_train_scaled)
                    pred_scaled = gp.predict(X_val_scaled)

                    global_idx = np.where(mask)[0][val_idx]
                    all_preds[global_idx, target_idx] = y_scaler.inverse_transform(
                        pred_scaled.reshape(-1, 1)
                    ).ravel()
                except:
                    global_idx = np.where(mask)[0][val_idx]
                    all_preds[global_idx, target_idx] = np.mean(y_train[:, target_idx])

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def main():
    print("=" * 60)
    print("GAUSSIAN PROCESS WITH ARD KERNEL (Plan 2.1)")
    print("=" * 60)

    if not GP_AVAILABLE:
        print("sklearn GaussianProcess required")
        return

    # Use only top stable features (GPs scale poorly with features)
    n_features = 20  # Reduced for GP computational efficiency
    stable_features = load_stable_features(n_features)
    print(f"\nTop {n_features} stable features: {stable_features[:5]}...")

    print("\nBuilding feature matrix...")
    X, y, pids, feature_names = build_feature_matrix(train=True, selected_features=stable_features)
    print(f"X shape: {X.shape}")

    results = []

    # Test configurations
    configs = [
        {"kernel": "rbf", "alpha": 1e-3},
        {"kernel": "rbf", "alpha": 1e-2},
        {"kernel": "rbf", "alpha": 1e-1},
        {"kernel": "matern", "alpha": 1e-2},
        {"kernel": "matern", "alpha": 1e-1},
    ]

    for config in configs:
        kernel = config["kernel"]
        alpha = config["alpha"]

        print(f"\n--- GP kernel={kernel}, alpha={alpha} ---")

        lopo_mse, lopo_per_target, preds, stds = evaluate_gp_lopo(
            X, y, pids, kernel_type=kernel, alpha=alpha
        )

        print(f"  LOPO MSE: {lopo_mse:.6f}")

        # Check uncertainty calibration
        if preds is not None and stds is not None:
            scalers_dict = load_scalers()
            calibration = evaluate_uncertainty_calibration(y, preds, stds, scalers_dict)

            for target, cal in calibration.items():
                print(f"  {target} uncertainty calibration:")
                print(f"    std-error correlation: {cal['std_error_corr']:.4f}")
                print(f"    MSE low/high uncertainty: {cal['mse_low_uncertainty']:.6f} / {cal['mse_high_uncertainty']:.6f}")
                print(f"    Ratio (>1 = well-calibrated): {cal['uncertainty_ratio']:.2f}")

        results.append({
            "kernel": kernel,
            "alpha": alpha,
            "lopo_mse": lopo_mse,
            "lopo_angle": lopo_per_target.get("angle", np.nan),
            "lopo_depth": lopo_per_target.get("depth", np.nan),
            "lopo_lr": lopo_per_target.get("left_right", np.nan),
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "gp_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'gp_results.csv'}")

    # Find best configuration
    best_idx = results_df["lopo_mse"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n=== BEST GP CONFIGURATION ===")
    print(f"Kernel: {best['kernel']}")
    print(f"Alpha: {best['alpha']}")
    print(f"LOPO MSE: {best['lopo_mse']:.6f}")


if __name__ == "__main__":
    main()
