"""
Optimized GP: Kernel Tuning + Feature Selection

1. Test different kernels (RBF, Matern with different nu)
2. Feature selection optimized for GP (not just stability)
3. Per-target feature selection
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel, DotProduct, WhiteKernel, RationalQuadratic
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def get_stable_features(n_features=50):
    drift_file = OUTPUT_DIR / "feature_drift_f4_with_importance.csv"
    if drift_file.exists():
        drift_df = pd.read_csv(drift_file)
        drift_df = drift_df.sort_values("stability_adjusted", ascending=False)
        return drift_df["feature"].head(n_features).tolist()
    return None


def build_feature_matrix(train: bool = True, feature_names: list = None):
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

    if feature_names is not None:
        available = [f for f in feature_names if f in feature_df.columns]
        feature_df = feature_df[available]

    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def select_features_for_gp(X, y, pids, feature_names, n_features=20, target_idx=0):
    """Select features that work well for GP on a specific target."""
    # Use mutual information to find non-linear relationships
    mi_scores = mutual_info_regression(X, y[:, target_idx], random_state=42)

    # Also check within-player variance (features that vary within player are useful)
    within_var_scores = np.zeros(X.shape[1])
    for pid in np.unique(pids):
        mask = pids == pid
        if mask.sum() > 1:
            within_var_scores += np.var(X[mask], axis=0)
    within_var_scores /= len(np.unique(pids))

    # Combine: high MI + high within-player variance
    combined_scores = mi_scores * (within_var_scores + 0.1)  # +0.1 to avoid zero

    top_indices = np.argsort(combined_scores)[-n_features:]
    return [feature_names[i] for i in top_indices]


def create_kernel(kernel_type):
    """Create kernel based on type."""
    if kernel_type == "rbf":
        return ConstantKernel(1.0) * RBF(length_scale=1.0)
    elif kernel_type == "matern_0.5":
        return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=0.5)
    elif kernel_type == "matern_1.5":
        return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
    elif kernel_type == "matern_2.5":
        return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    elif kernel_type == "rbf_linear":
        return ConstantKernel(1.0) * RBF(length_scale=1.0) + DotProduct(sigma_0=1.0)
    elif kernel_type == "rational_quadratic":
        return ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
    else:
        return ConstantKernel(1.0) * RBF(length_scale=1.0)


def within_player_cv_single_target(X, y_target, pids, kernel_type, alpha):
    """CV for a single target."""
    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y_target)

    for pid in unique_pids:
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y_target[mask]
        indices = np.where(mask)[0]

        feature_scaler = StandardScaler()
        X_pid_scaled = feature_scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_pid):
            X_train = X_pid_scaled[train_idx]
            X_test = X_pid_scaled[test_idx]
            y_train = y_pid[train_idx]

            kernel = create_kernel(kernel_type)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                n_restarts_optimizer=2,
                random_state=42
            )
            gp.fit(X_train, y_train)
            all_preds[indices[test_idx]] = gp.predict(X_test)

    return all_preds


def main():
    print("=" * 60)
    print("OPTIMIZED GP: KERNEL TUNING + FEATURE SELECTION")
    print("=" * 60)

    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    # Get more features initially for selection
    stable_features = get_stable_features(80)

    print("\nBuilding feature matrix...")
    X_train, y_train, pids_train, all_feature_names = build_feature_matrix(
        train=True, feature_names=stable_features
    )
    print(f"Initial X shape: {X_train.shape}")

    # Test kernels with all features first
    print("\n" + "=" * 60)
    print("PHASE 1: KERNEL COMPARISON (all 80 features)")
    print("=" * 60)

    kernel_types = ["rbf", "matern_0.5", "matern_1.5", "matern_2.5", "rational_quadratic"]
    kernel_results = []

    for kernel_type in kernel_types:
        print(f"\nTesting {kernel_type}...")

        mse_per_target = {}
        for target_idx, target_name in enumerate(target_names):
            preds = within_player_cv_single_target(
                X_train, y_train[:, target_idx], pids_train, kernel_type, alpha=0.5
            )
            # Scale and compute MSE
            y_scaled = scalers_dict[target_name].transform(y_train[:, target_idx].reshape(-1, 1)).ravel()
            pred_scaled = scalers_dict[target_name].transform(preds.reshape(-1, 1)).ravel()
            mse_per_target[target_name] = np.mean((y_scaled - pred_scaled) ** 2)

        total_mse = np.mean(list(mse_per_target.values()))
        kernel_results.append({
            "kernel": kernel_type,
            "total_mse": total_mse,
            **mse_per_target
        })
        print(f"  Total MSE: {total_mse:.6f}")
        print(f"  angle={mse_per_target['angle']:.6f}, depth={mse_per_target['depth']:.6f}, lr={mse_per_target['left_right']:.6f}")

    kernel_df = pd.DataFrame(kernel_results)
    best_kernel = kernel_df.loc[kernel_df["total_mse"].idxmin(), "kernel"]
    print(f"\nBest kernel: {best_kernel}")

    # Feature selection for each target
    print("\n" + "=" * 60)
    print("PHASE 2: PER-TARGET FEATURE SELECTION")
    print("=" * 60)

    selected_features = {}
    for target_idx, target_name in enumerate(target_names):
        print(f"\nSelecting features for {target_name}...")
        features = select_features_for_gp(
            X_train, y_train, pids_train, all_feature_names,
            n_features=25, target_idx=target_idx
        )
        selected_features[target_name] = features
        print(f"  Top 5: {features[-5:]}")

    # Test with selected features
    print("\n" + "=" * 60)
    print("PHASE 3: OPTIMIZED MODEL (best kernel + selected features)")
    print("=" * 60)

    # Test different alpha values with best kernel and selected features
    best_config = {"kernel": best_kernel, "alpha": 0.5}
    best_total_mse = float('inf')

    for alpha in [0.1, 0.2, 0.5, 1.0]:
        print(f"\nTesting alpha={alpha}...")

        mse_per_target = {}
        for target_idx, target_name in enumerate(target_names):
            # Get features for this target
            target_features = selected_features[target_name]
            feature_indices = [all_feature_names.index(f) for f in target_features if f in all_feature_names]
            X_target = X_train[:, feature_indices]

            preds = within_player_cv_single_target(
                X_target, y_train[:, target_idx], pids_train, best_kernel, alpha
            )
            y_scaled = scalers_dict[target_name].transform(y_train[:, target_idx].reshape(-1, 1)).ravel()
            pred_scaled = scalers_dict[target_name].transform(preds.reshape(-1, 1)).ravel()
            mse_per_target[target_name] = np.mean((y_scaled - pred_scaled) ** 2)

        total_mse = np.mean(list(mse_per_target.values()))
        print(f"  Total MSE: {total_mse:.6f}")

        if total_mse < best_total_mse:
            best_total_mse = total_mse
            best_config = {"kernel": best_kernel, "alpha": alpha}

    print(f"\nBest config: kernel={best_config['kernel']}, alpha={best_config['alpha']}")
    print(f"Best within-player MSE: {best_total_mse:.6f}")

    # Compare with shared features (same features for all targets)
    print("\n" + "=" * 60)
    print("PHASE 4: SHARED vs PER-TARGET FEATURES")
    print("=" * 60)

    # Union of all selected features
    all_selected = set()
    for features in selected_features.values():
        all_selected.update(features)
    shared_features = list(all_selected)
    print(f"Union of selected features: {len(shared_features)}")

    shared_indices = [all_feature_names.index(f) for f in shared_features if f in all_feature_names]
    X_shared = X_train[:, shared_indices]

    mse_shared = {}
    for target_idx, target_name in enumerate(target_names):
        preds = within_player_cv_single_target(
            X_shared, y_train[:, target_idx], pids_train,
            best_config['kernel'], best_config['alpha']
        )
        y_scaled = scalers_dict[target_name].transform(y_train[:, target_idx].reshape(-1, 1)).ravel()
        pred_scaled = scalers_dict[target_name].transform(preds.reshape(-1, 1)).ravel()
        mse_shared[target_name] = np.mean((y_scaled - pred_scaled) ** 2)

    total_shared = np.mean(list(mse_shared.values()))
    print(f"Shared features MSE: {total_shared:.6f}")

    # Create submission
    print("\n" + "=" * 60)
    print("CREATING SUBMISSION")
    print("=" * 60)

    print("\nBuilding test features...")
    X_test, _, pids_test, _ = build_feature_matrix(train=False, feature_names=stable_features)

    predictions = np.zeros((len(X_test), 3))
    unique_test_pids = np.unique(pids_test)
    unique_train_pids = np.unique(pids_train)

    for target_idx, target_name in enumerate(target_names):
        print(f"\nTraining GP for {target_name}...")

        # Use per-target features
        target_features = selected_features[target_name]
        feature_indices = [all_feature_names.index(f) for f in target_features if f in all_feature_names]

        X_train_target = X_train[:, feature_indices]
        X_test_target = X_test[:, feature_indices]

        for test_pid in unique_test_pids:
            test_mask = pids_test == test_pid

            if test_pid in unique_train_pids:
                train_mask = pids_train == test_pid
                X_tr = X_train_target[train_mask]
                y_tr = y_train[train_mask, target_idx]
            else:
                X_tr = X_train_target
                y_tr = y_train[:, target_idx]

            feature_scaler = StandardScaler()
            X_tr_scaled = feature_scaler.fit_transform(X_tr)
            X_te_scaled = feature_scaler.transform(X_test_target[test_mask])

            kernel = create_kernel(best_config['kernel'])
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=best_config['alpha'],
                n_restarts_optimizer=3,
                random_state=42
            )
            gp.fit(X_tr_scaled, y_tr)
            predictions[test_mask, target_idx] = gp.predict(X_te_scaled)

    # Scale predictions
    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(target_names):
        scaled_predictions[:, i] = scalers_dict[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).ravel()

    # Profile
    print("\n" + "=" * 60)
    print("SUBMISSION PROFILE")
    print("=" * 60)

    print(f"\nOptimized GP predictions:")
    print(f"  angle: mean={scaled_predictions[:, 0].mean():.4f}, std={scaled_predictions[:, 0].std():.4f}")
    print(f"  depth: mean={scaled_predictions[:, 1].mean():.4f}, std={scaled_predictions[:, 1].std():.4f}")
    print(f"  left_right: mean={scaled_predictions[:, 2].mean():.4f}, std={scaled_predictions[:, 2].std():.4f}")

    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    print(f"\nSub 133 (LB 0.007809):")
    print(f"  angle: mean={sub133['scaled_angle'].mean():.4f}, std={sub133['scaled_angle'].std():.4f}")
    print(f"  depth: mean={sub133['scaled_depth'].mean():.4f}, std={sub133['scaled_depth'].std():.4f}")
    print(f"  left_right: mean={sub133['scaled_left_right'].mean():.4f}, std={sub133['scaled_left_right'].std():.4f}")

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split("_")[1]) for f in existing if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1 if nums else 140

    test_meta = load_metadata(train=False)
    submission_df = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": scaled_predictions[:, 0],
        "scaled_depth": scaled_predictions[:, 1],
        "scaled_left_right": scaled_predictions[:, 2]
    })

    submission_path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    # Correlation
    corr_angle = np.corrcoef(scaled_predictions[:, 0], sub133['scaled_angle'].values)[0, 1]
    corr_depth = np.corrcoef(scaled_predictions[:, 1], sub133['scaled_depth'].values)[0, 1]
    corr_lr = np.corrcoef(scaled_predictions[:, 2], sub133['scaled_left_right'].values)[0, 1]

    print(f"\nCorrelation with Sub 133:")
    print(f"  angle: {corr_angle:.4f}")
    print(f"  depth: {corr_depth:.4f}")
    print(f"  left_right: {corr_lr:.4f}")

    # Save results
    results = {
        "best_kernel": best_config['kernel'],
        "best_alpha": best_config['alpha'],
        "within_player_mse": best_total_mse,
        "selected_features": selected_features
    }

    results_df = pd.DataFrame([{
        "kernel": best_config['kernel'],
        "alpha": best_config['alpha'],
        "mse": best_total_mse,
        "angle_features": len(selected_features['angle']),
        "depth_features": len(selected_features['depth']),
        "lr_features": len(selected_features['left_right'])
    }])
    results_df.to_csv(OUTPUT_DIR / "gp_optimized_results.csv", index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'gp_optimized_results.csv'}")


if __name__ == "__main__":
    main()
