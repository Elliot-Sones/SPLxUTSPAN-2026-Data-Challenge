"""
Stability-Adjusted Feature Selection (Plan 1.1)

Uses features with high importance + low drift for better generalization.
Features are selected based on stability_adjusted score from feature_drift_f4_with_importance.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, NUM_FRAMES, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Load stability-adjusted scores
def load_stable_features(top_n: int = 50) -> list:
    """Load top N features by stability_adjusted score."""
    drift_df = pd.read_csv(OUTPUT_DIR / "feature_drift_f4_with_importance.csv")

    # Sort by stability_adjusted (higher = more stable + more important)
    drift_df = drift_df.sort_values("stability_adjusted", ascending=False)

    # Get top N features
    stable_features = drift_df.head(top_n)["feature"].tolist()
    return stable_features


def extract_features_for_shot(timeseries: np.ndarray, participant_id: int) -> dict:
    """Extract all features from a single shot."""
    features = extract_hybrid_features(timeseries, participant_id, smooth=True)
    return features


def build_feature_matrix(train: bool = True, stable_features: list = None):
    """Build feature matrix using only stable features."""
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

        features = extract_features_for_shot(timeseries, metadata["participant_id"])
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    # Build DataFrame
    feature_df = pd.DataFrame(all_features)

    # Filter to stable features only
    if stable_features:
        available = [f for f in stable_features if f in feature_df.columns]
        print(f"  Using {len(available)} of {len(stable_features)} stable features")
        feature_df = feature_df[available]

    # Fill NaN with column median
    feature_df = feature_df.fillna(feature_df.median())

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def evaluate_lopo_cv(X, y, pids, alpha=100.0):
    """Leave-One-Participant-Out CV evaluation."""
    scalers_dict = load_scalers()

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train[:, target_idx])
            all_preds[test_mask, target_idx] = model.predict(X_test_scaled)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def evaluate_within_player_cv(X, y, pids, n_folds=5, alpha=100.0):
    """Within-player 5-fold CV (more representative of test distribution)."""
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
            y_train, y_val = y_pid[train_idx], y_pid[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled, y_train[:, target_idx])

                # Map back to global indices
                global_idx = np.where(mask)[0][val_idx]
                all_preds[global_idx, target_idx] = model.predict(X_val_scaled)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def create_submission(X_train, y_train, pids_train, X_test, pids_test, alpha=100.0):
    """Create submission using per-player models."""
    unique_pids = np.unique(pids_train)

    test_preds = np.zeros((len(X_test), 3))

    for pid in unique_pids:
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        if test_mask.sum() == 0:
            continue

        X_train_pid = X_train[train_mask]
        y_train_pid = y_train[train_mask]
        X_test_pid = X_test[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pid)
        X_test_scaled = scaler.transform(X_test_pid)

        for target_idx in range(3):
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train_pid[:, target_idx])
            test_preds[test_mask, target_idx] = model.predict(X_test_scaled)

    return test_preds


def main():
    print("=" * 60)
    print("STABILITY-ADJUSTED FEATURE SELECTION (Plan 1.1)")
    print("=" * 60)

    # Test different numbers of stable features
    results = []

    for n_features in [20, 30, 40, 50, 60, 80, 100]:
        print(f"\n--- Testing with top {n_features} stable features ---")

        stable_features = load_stable_features(n_features)
        print(f"Top 5 stable features: {stable_features[:5]}")

        print("Building feature matrix...")
        X, y, pids, feature_names = build_feature_matrix(train=True, stable_features=stable_features)
        print(f"  X shape: {X.shape}")

        # Test different regularization strengths
        for alpha in [10, 50, 100, 200]:
            lopo_mse, lopo_per_target = evaluate_lopo_cv(X, y, pids, alpha=alpha)
            within_mse, within_per_target = evaluate_within_player_cv(X, y, pids, alpha=alpha)

            results.append({
                "n_features": n_features,
                "alpha": alpha,
                "lopo_mse": lopo_mse,
                "lopo_angle": lopo_per_target["angle"],
                "lopo_depth": lopo_per_target["depth"],
                "lopo_lr": lopo_per_target["left_right"],
                "within_mse": within_mse,
                "within_angle": within_per_target["angle"],
                "within_depth": within_per_target["depth"],
                "within_lr": within_per_target["left_right"],
            })

            print(f"  alpha={alpha}: LOPO={lopo_mse:.6f}, Within-player={within_mse:.6f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "stability_feature_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'stability_feature_results.csv'}")

    # Find best configuration
    best_idx = results_df["lopo_mse"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n=== BEST CONFIGURATION ===")
    print(f"n_features: {best['n_features']}")
    print(f"alpha: {best['alpha']}")
    print(f"LOPO MSE: {best['lopo_mse']:.6f}")
    print(f"  angle: {best['lopo_angle']:.6f}")
    print(f"  depth: {best['lopo_depth']:.6f}")
    print(f"  left_right: {best['lopo_lr']:.6f}")

    # Create submission with best config
    print(f"\n--- Creating submission with best config ---")
    stable_features = load_stable_features(int(best["n_features"]))

    X_train, y_train, pids_train, _ = build_feature_matrix(train=True, stable_features=stable_features)
    X_test, _, pids_test, _ = build_feature_matrix(train=False, stable_features=stable_features)

    test_preds = create_submission(X_train, y_train, pids_train, X_test, pids_test, alpha=best["alpha"])

    # Calculate submission profile
    angle_std = np.std(test_preds[:, 0])
    depth_mean = np.mean(test_preds[:, 1])

    print(f"Submission profile:")
    print(f"  angle_std: {angle_std:.4f}")
    print(f"  depth_mean: {depth_mean:.4f}")

    # Profile check
    profile_ok = angle_std < 0.145 and 0.49 < depth_mean < 0.52
    print(f"  Profile check: {'PASS' if profile_ok else 'FAIL'}")

    if profile_ok:
        # Get next submission number
        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        next_num = max([int(f.stem.split("_")[1]) for f in existing]) + 1 if existing else 137

        # Load test metadata
        test_meta = load_metadata(train=False)

        sub_df = pd.DataFrame({
            "id": test_meta["id"],
            "angle": test_preds[:, 0],
            "depth": test_preds[:, 1],
            "left_right": test_preds[:, 2]
        })

        sub_path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"\nSubmission saved to: {sub_path}")
        print(f"LOPO CV: {best['lopo_mse']:.6f}")
    else:
        print("\nSubmission not created - profile check failed")


if __name__ == "__main__":
    main()
