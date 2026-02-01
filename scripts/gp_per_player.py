"""
Per-Player GP Submission

Train separate GP for each player to capture player-specific patterns.
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


def within_player_cv(X, y, pids, alpha=0.1):
    """Evaluate per-player GP with within-player CV."""
    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask]
        indices = np.where(mask)[0]

        # Scale within player
        feature_scaler = StandardScaler()
        X_pid_scaled = feature_scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_pid):
            X_train = X_pid_scaled[train_idx]
            X_test = X_pid_scaled[test_idx]
            y_train = y_pid[train_idx]

            for target_idx in range(3):
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    n_restarts_optimizer=2,
                    random_state=42
                )
                gp.fit(X_train, y_train[:, target_idx])

                global_test_idx = indices[test_idx]
                all_preds[global_test_idx, target_idx] = gp.predict(X_test)

    # Scale and compute MSE
    mse_per_target = {}
    for i, target in enumerate(target_names):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def main():
    print("=" * 60)
    print("PER-PLAYER GP SUBMISSION")
    print("=" * 60)

    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]

    stable_features = get_stable_features(30)

    print("\nBuilding train features...")
    X_train, y_train, pids_train, _ = build_feature_matrix(train=True, feature_names=stable_features)
    print(f"Train shape: {X_train.shape}")

    print("\nBuilding test features...")
    X_test, _, pids_test, _ = build_feature_matrix(train=False, feature_names=stable_features)
    print(f"Test shape: {X_test.shape}")

    # Test alpha values
    print("\n" + "=" * 60)
    print("TESTING ALPHA VALUES (Within-Player CV)")
    print("=" * 60)

    best_alpha = 0.1
    best_mse = float('inf')

    for alpha in [0.01, 0.05, 0.1, 0.2, 0.5]:
        mse, per_target = within_player_cv(X_train, y_train, pids_train, alpha=alpha)
        print(f"\nalpha={alpha}: MSE={mse:.6f}")
        print(f"  angle={per_target['angle']:.6f}, depth={per_target['depth']:.6f}, lr={per_target['left_right']:.6f}")

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha} (MSE: {best_mse:.6f})")

    # Create submission with per-player models
    print("\n" + "=" * 60)
    print("CREATING SUBMISSION")
    print("=" * 60)

    unique_train_pids = np.unique(pids_train)
    unique_test_pids = np.unique(pids_test)

    predictions = np.zeros((len(X_test), 3))

    for test_pid in unique_test_pids:
        test_mask = pids_test == test_pid
        X_test_pid = X_test[test_mask]

        print(f"\nPlayer {test_pid}: {test_mask.sum()} test samples")

        if test_pid in unique_train_pids:
            # Same player - use player-specific model
            train_mask = pids_train == test_pid
            X_train_pid = X_train[train_mask]
            y_train_pid = y_train[train_mask]
            print(f"  Using {train_mask.sum()} player-specific training samples")
        else:
            # Unknown player - use all data
            X_train_pid = X_train
            y_train_pid = y_train
            print(f"  Unknown player - using all {len(X_train)} training samples")

        # Scale features for this player
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train_pid)
        X_test_scaled = feature_scaler.transform(X_test_pid)

        for target_idx, target_name in enumerate(target_names):
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=best_alpha,
                n_restarts_optimizer=3,
                random_state=42
            )
            gp.fit(X_train_scaled, y_train_pid[:, target_idx])
            predictions[test_mask, target_idx] = gp.predict(X_test_scaled)

    # Scale predictions
    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(target_names):
        scaled_predictions[:, i] = scalers_dict[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).ravel()

    # Profile check
    print("\n" + "=" * 60)
    print("SUBMISSION PROFILE")
    print("=" * 60)

    print(f"\nGP predictions (scaled):")
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
    next_num = max(nums) + 1 if nums else 139

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

    # Correlation with Sub 133
    corr_angle = np.corrcoef(scaled_predictions[:, 0], sub133['scaled_angle'].values)[0, 1]
    corr_depth = np.corrcoef(scaled_predictions[:, 1], sub133['scaled_depth'].values)[0, 1]
    corr_lr = np.corrcoef(scaled_predictions[:, 2], sub133['scaled_left_right'].values)[0, 1]

    print(f"\nCorrelation with Sub 133:")
    print(f"  angle: {corr_angle:.4f}")
    print(f"  depth: {corr_depth:.4f}")
    print(f"  left_right: {corr_lr:.4f}")

    avg_corr = (corr_angle + corr_depth + corr_lr) / 3
    print(f"  average: {avg_corr:.4f}")

    if avg_corr > 0.8:
        print("\nHigh correlation with Sub 133 - may not add diversity to ensemble")
    elif avg_corr < 0.5:
        print("\nLow correlation - could provide diversity but may have different error patterns")


if __name__ == "__main__":
    main()
