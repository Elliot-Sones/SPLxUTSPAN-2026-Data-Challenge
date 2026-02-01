"""
GP Submission with Scaled Targets

Train GP on scaled targets to match submission format.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def get_stable_features(n_features=30):
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
        print(f"  Using {len(available)} of {len(feature_names)} features")
        feature_df = feature_df[available]

    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def main():
    print("=" * 60)
    print("GP SCALED SUBMISSION")
    print("=" * 60)

    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    stable_features = get_stable_features(30)

    print("\nBuilding train features...")
    X_train, y_train, pids_train, _ = build_feature_matrix(train=True, feature_names=stable_features)
    print(f"Train X shape: {X_train.shape}")

    print("\nBuilding test features...")
    X_test, _, pids_test, _ = build_feature_matrix(train=False, feature_names=stable_features)
    print(f"Test X shape: {X_test.shape}")

    # Scale targets for training
    y_train_scaled = np.zeros_like(y_train)
    for i, target in enumerate(target_names):
        y_train_scaled[:, i] = scalers_dict[target].transform(
            y_train[:, i].reshape(-1, 1)
        ).ravel()

    print(f"\nScaled target ranges:")
    for i, target in enumerate(target_names):
        print(f"  {target}: min={y_train_scaled[:, i].min():.3f}, max={y_train_scaled[:, i].max():.3f}")

    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Test different alpha values with within-player CV
    print("\n" + "=" * 60)
    print("TESTING ALPHA VALUES")
    print("=" * 60)

    best_alpha = 0.5
    best_mse = float('inf')

    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        print(f"\nTesting alpha={alpha}...")

        unique_pids = np.unique(pids_train)
        all_preds = np.zeros_like(y_train_scaled)

        for pid in unique_pids:
            mask = pids_train == pid
            indices = np.where(mask)[0]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, test_idx in kf.split(indices):
                train_global = indices[train_idx]
                test_global = indices[test_idx]

                # Use other players + train fold
                other_mask = pids_train != pid
                combined_mask = np.zeros(len(X_train), dtype=bool)
                combined_mask[other_mask] = True
                combined_mask[train_global] = True

                X_tr = X_train_scaled[combined_mask]
                X_te = X_train_scaled[test_global]

                for target_idx in range(3):
                    y_tr = y_train_scaled[combined_mask, target_idx]

                    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=alpha,
                        n_restarts_optimizer=2,
                        random_state=42
                    )
                    gp.fit(X_tr, y_tr)
                    all_preds[test_global, target_idx] = gp.predict(X_te)

        # Calculate MSE on scaled targets
        mse = np.mean((all_preds - y_train_scaled) ** 2)
        print(f"  Within-player MSE (scaled): {mse:.6f}")

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha} (MSE: {best_mse:.6f})")

    # Train final model
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

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
        gp.fit(X_train_scaled, y_train_scaled[:, target_idx])
        predictions[:, target_idx] = gp.predict(X_test_scaled)

        print(f"  Kernel: {gp.kernel_}")

    # Check profile
    print(f"\n{'=' * 60}")
    print("SUBMISSION PROFILE")
    print(f"{'=' * 60}")

    print(f"\nGP predictions (scaled):")
    print(f"  angle: mean={predictions[:, 0].mean():.4f}, std={predictions[:, 0].std():.4f}")
    print(f"  depth: mean={predictions[:, 1].mean():.4f}, std={predictions[:, 1].std():.4f}")
    print(f"  left_right: mean={predictions[:, 2].mean():.4f}, std={predictions[:, 2].std():.4f}")

    # Load Sub 133 for comparison
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    print(f"\nSub 133 profile (LB 0.007809):")
    print(f"  angle: mean={sub133['scaled_angle'].mean():.4f}, std={sub133['scaled_angle'].std():.4f}")
    print(f"  depth: mean={sub133['scaled_depth'].mean():.4f}, std={sub133['scaled_depth'].std():.4f}")
    print(f"  left_right: mean={sub133['scaled_left_right'].mean():.4f}, std={sub133['scaled_left_right'].std():.4f}")

    # Save submission
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split("_")[1]) for f in existing if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1 if nums else 138

    test_meta = load_metadata(train=False)
    submission_df = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": predictions[:, 0],
        "scaled_depth": predictions[:, 1],
        "scaled_left_right": predictions[:, 2]
    })

    submission_path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    # Calculate correlation with Sub 133
    corr_angle = np.corrcoef(predictions[:, 0], sub133['scaled_angle'].values)[0, 1]
    corr_depth = np.corrcoef(predictions[:, 1], sub133['scaled_depth'].values)[0, 1]
    corr_lr = np.corrcoef(predictions[:, 2], sub133['scaled_left_right'].values)[0, 1]

    print(f"\nCorrelation with Sub 133:")
    print(f"  angle: {corr_angle:.4f}")
    print(f"  depth: {corr_depth:.4f}")
    print(f"  left_right: {corr_lr:.4f}")

    # Check if profile is similar enough
    angle_std_diff = abs(predictions[:, 0].std() - sub133['scaled_angle'].std())
    depth_mean_diff = abs(predictions[:, 1].mean() - sub133['scaled_depth'].mean())

    if angle_std_diff < 0.05 and depth_mean_diff < 0.05:
        print("\nProfile check: PASS - similar to Sub 133")
    else:
        print(f"\nProfile check: WARN - angle_std_diff={angle_std_diff:.4f}, depth_mean_diff={depth_mean_diff:.4f}")


if __name__ == "__main__":
    main()
