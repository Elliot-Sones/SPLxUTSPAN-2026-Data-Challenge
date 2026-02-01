"""
Global GP Submission

Train GP on ALL players (not per-player) to avoid overfitting.
Use the kernel and features that showed best within-player CV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct
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


def get_stable_features(n_features=30):
    """Get top N stable features."""
    drift_file = OUTPUT_DIR / "feature_drift_f4_with_importance.csv"
    if drift_file.exists():
        drift_df = pd.read_csv(drift_file)
        drift_df = drift_df.sort_values("stability_adjusted", ascending=False)
        return drift_df["feature"].head(n_features).tolist()
    return None


def build_feature_matrix(train: bool = True, feature_names: list = None):
    """Build feature matrix."""
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
        print(f"  Using {len(available)} of {len(feature_names)} features")
        feature_df = feature_df[available]

    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def evaluate_within_player_cv(X, y, pids, alpha=0.1, n_splits=5):
    """Within-player CV with global model."""
    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask]
        indices = np.where(mask)[0]

        kf = KFold(n_splits=min(n_splits, len(X_pid)), shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_pid):
            # Use ALL training data from other players + train fold from this player
            other_mask = pids != pid
            train_fold_global_idx = indices[train_idx]

            combined_train_mask = np.zeros(len(X), dtype=bool)
            combined_train_mask[other_mask] = True
            combined_train_mask[train_fold_global_idx] = True

            X_train = X[combined_train_mask]
            y_train = y[combined_train_mask]
            X_test = X_pid[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for target_idx in range(3):
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    n_restarts_optimizer=2,
                    random_state=42
                )
                gp.fit(X_train_scaled, y_train[:, target_idx])

                orig_test_idx = indices[test_idx]
                all_preds[orig_test_idx, target_idx] = gp.predict(X_test_scaled)

    # Calculate MSE
    mse_per_target = {}
    for i, target in enumerate(target_names):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target, all_preds


def main():
    print("=" * 60)
    print("GLOBAL GP SUBMISSION")
    print("=" * 60)

    # Use 30 stable features (best from enhanced GP)
    stable_features = get_stable_features(30)

    print("\nBuilding train features...")
    X_train, y_train, pids_train, feature_names = build_feature_matrix(
        train=True, feature_names=stable_features
    )
    print(f"Train X shape: {X_train.shape}")

    # Test different alpha values
    results = []
    for alpha in [0.05, 0.1, 0.2, 0.5, 1.0]:
        print(f"\nTesting alpha={alpha}...")
        wp_mse, wp_per_target, _ = evaluate_within_player_cv(X_train, y_train, pids_train, alpha=alpha)
        print(f"  Within-player MSE: {wp_mse:.6f}")
        print(f"    angle: {wp_per_target['angle']:.6f}")
        print(f"    depth: {wp_per_target['depth']:.6f}")
        print(f"    left_right: {wp_per_target['left_right']:.6f}")

        results.append({
            "alpha": alpha,
            "wp_mse": wp_mse,
            **wp_per_target
        })

    results_df = pd.DataFrame(results)
    best_alpha = results_df.loc[results_df["wp_mse"].idxmin(), "alpha"]
    print(f"\nBest alpha: {best_alpha}")

    # Create submission with best alpha
    print(f"\n{'=' * 60}")
    print("CREATING SUBMISSION")
    print(f"{'=' * 60}")

    print("\nBuilding test features...")
    X_test, _, pids_test, _ = build_feature_matrix(train=False, feature_names=stable_features)
    print(f"Test X shape: {X_test.shape}")

    # Scale all data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train global GP on all training data
    target_names = ["angle", "depth", "left_right"]
    predictions = np.zeros((len(X_test), 3))

    for target_idx, target_name in enumerate(target_names):
        print(f"\nTraining GP for {target_name}...")
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=best_alpha,
            n_restarts_optimizer=5,
            random_state=42
        )
        gp.fit(X_train_scaled, y_train[:, target_idx])
        predictions[:, target_idx] = gp.predict(X_test_scaled)

        print(f"  Learned kernel: {gp.kernel_}")

    # Check profile
    angle_std = np.std(predictions[:, 0])
    angle_mean = np.mean(predictions[:, 0])
    depth_std = np.std(predictions[:, 1])
    depth_mean = np.mean(predictions[:, 1])
    lr_std = np.std(predictions[:, 2])
    lr_mean = np.mean(predictions[:, 2])

    print(f"\nSubmission profile:")
    print(f"  angle: mean={angle_mean:.4f}, std={angle_std:.4f}")
    print(f"  depth: mean={depth_mean:.4f}, std={depth_std:.4f}")
    print(f"  left_right: mean={lr_mean:.4f}, std={lr_std:.4f}")

    # Compare with Sub 133 profile (our best LB 0.007809)
    # Load Sub 133 for comparison
    sub133_path = SUBMISSION_DIR / "submission_133.csv"
    if sub133_path.exists():
        sub133 = pd.read_csv(sub133_path)
        print(f"\nSub 133 profile (LB 0.007809):")
        print(f"  angle: mean={sub133['angle'].mean():.4f}, std={sub133['angle'].std():.4f}")
        print(f"  depth: mean={sub133['depth'].mean():.4f}, std={sub133['depth'].std():.4f}")
        print(f"  left_right: mean={sub133['left_right'].mean():.4f}, std={sub133['left_right'].std():.4f}")

    # Save submission
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if existing:
        nums = [int(f.stem.split("_")[1]) for f in existing]
        next_num = max(nums) + 1
    else:
        next_num = 138

    test_meta = load_metadata(train=False)
    submission_df = pd.DataFrame({
        "id": test_meta["id"].values,
        "angle": predictions[:, 0],
        "depth": predictions[:, 1],
        "left_right": predictions[:, 2]
    })

    submission_path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    # Save results
    results_df.to_csv(OUTPUT_DIR / "gp_global_results.csv", index=False)
    print(f"Results saved to: {OUTPUT_DIR / 'gp_global_results.csv'}")


if __name__ == "__main__":
    main()
