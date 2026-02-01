"""
Feature Subspace Ensemble with Profile Constraint (Plan 1.3)

Create models with non-overlapping feature subsets.
Only blend those with angle_std < 0.145 (profile check).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers
from data_loader import get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def load_stable_features(top_n: int = 60) -> list:
    """Load top N stable features."""
    drift_df = pd.read_csv(OUTPUT_DIR / "feature_drift_f4_with_importance.csv")
    drift_df = drift_df.sort_values("stability_adjusted", ascending=False)
    return drift_df.head(top_n)["feature"].tolist()


def create_feature_subsets(features: list, n_subsets: int = 4) -> list:
    """Create non-overlapping feature subsets."""
    n_per_subset = len(features) // n_subsets
    subsets = []

    for i in range(n_subsets):
        start = i * n_per_subset
        if i == n_subsets - 1:
            subset = features[start:]
        else:
            subset = features[start:start + n_per_subset]
        subsets.append(subset)

    return subsets


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

        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    feature_df = pd.DataFrame(all_features)

    if selected_features:
        available = [f for f in selected_features if f in feature_df.columns]
        feature_df = feature_df[available]

    feature_df = feature_df.fillna(feature_df.median())

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def train_subspace_model(X, y, pids, alpha=100.0):
    """Train a single model and get predictions."""
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

        for target_idx in range(3):
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train[:, target_idx])
            all_preds[test_mask, target_idx] = model.predict(X_test_scaled)

    return all_preds


def get_test_predictions(X_train, y_train, pids_train, X_test, pids_test, alpha=100.0):
    """Get predictions on test set using per-player models."""
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


def calculate_profile(preds):
    """Calculate submission profile metrics."""
    return {
        "angle_std": np.std(preds[:, 0]),
        "angle_mean": np.mean(preds[:, 0]),
        "depth_mean": np.mean(preds[:, 1]),
        "depth_std": np.std(preds[:, 1]),
        "lr_mean": np.mean(preds[:, 2]),
        "lr_std": np.std(preds[:, 2]),
    }


def calculate_cv_mse(y, preds, scalers_dict):
    """Calculate scaled MSE."""
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)
    return np.mean(list(mse_per_target.values())), mse_per_target


def main():
    print("=" * 60)
    print("FEATURE SUBSPACE ENSEMBLE (Plan 1.3)")
    print("=" * 60)

    scalers_dict = load_scalers()

    # Load stable features
    all_features = load_stable_features(60)
    print(f"Using {len(all_features)} stable features")

    # Create non-overlapping subsets
    n_subsets = 4
    feature_subsets = create_feature_subsets(all_features, n_subsets)
    print(f"\nCreated {n_subsets} feature subsets:")
    for i, subset in enumerate(feature_subsets):
        print(f"  Subset {i+1}: {len(subset)} features - {subset[:3]}...")

    # Build full feature matrix
    X_full, y, pids, all_feature_names = build_feature_matrix(train=True, selected_features=all_features)

    # Train models for each subset
    subset_results = []

    for i, subset in enumerate(feature_subsets):
        print(f"\n--- Subset {i+1} ({len(subset)} features) ---")

        # Build feature matrix for this subset
        X, _, _, feature_names = build_feature_matrix(train=True, selected_features=subset)

        # Train and get CV predictions
        cv_preds = train_subspace_model(X, y, pids, alpha=100.0)

        # Calculate metrics
        cv_mse, per_target = calculate_cv_mse(y, cv_preds, scalers_dict)

        # Get test predictions for profile check
        X_test, _, pids_test, _ = build_feature_matrix(train=False, selected_features=subset)
        test_preds = get_test_predictions(X, y, pids, X_test, pids_test, alpha=100.0)
        profile = calculate_profile(test_preds)

        subset_results.append({
            "subset_id": i + 1,
            "n_features": len(subset),
            "cv_mse": cv_mse,
            "angle_mse": per_target["angle"],
            "depth_mse": per_target["depth"],
            "lr_mse": per_target["left_right"],
            "angle_std": profile["angle_std"],
            "depth_mean": profile["depth_mean"],
            "profile_pass": profile["angle_std"] < 0.145 and 0.49 < profile["depth_mean"] < 0.52,
            "test_preds": test_preds,
        })

        print(f"  CV MSE: {cv_mse:.6f}")
        print(f"  angle_std: {profile['angle_std']:.4f}")
        print(f"  depth_mean: {profile['depth_mean']:.4f}")
        print(f"  Profile: {'PASS' if subset_results[-1]['profile_pass'] else 'FAIL'}")

    # Filter to models that pass profile check
    passing_models = [r for r in subset_results if r["profile_pass"]]
    print(f"\n=== PROFILE CHECK SUMMARY ===")
    print(f"Models passing: {len(passing_models)} / {len(subset_results)}")

    if len(passing_models) > 0:
        # Create ensemble from passing models
        print(f"\n--- Creating ensemble from {len(passing_models)} passing models ---")

        # Equal weight blend
        ensemble_preds = np.mean([r["test_preds"] for r in passing_models], axis=0)
        ensemble_profile = calculate_profile(ensemble_preds)

        print(f"Ensemble profile:")
        print(f"  angle_std: {ensemble_profile['angle_std']:.4f}")
        print(f"  depth_mean: {ensemble_profile['depth_mean']:.4f}")

        # Check correlation between submodel predictions
        print(f"\n--- Submodel correlation analysis ---")
        for i, r1 in enumerate(passing_models):
            for j, r2 in enumerate(passing_models):
                if i < j:
                    corr_angle = np.corrcoef(r1["test_preds"][:, 0], r2["test_preds"][:, 0])[0, 1]
                    corr_depth = np.corrcoef(r1["test_preds"][:, 1], r2["test_preds"][:, 1])[0, 1]
                    print(f"  Subset {r1['subset_id']} vs {r2['subset_id']}: angle_corr={corr_angle:.3f}, depth_corr={corr_depth:.3f}")

        # Save ensemble submission if profile passes
        ensemble_ok = ensemble_profile["angle_std"] < 0.145 and 0.49 < ensemble_profile["depth_mean"] < 0.52
        print(f"  Ensemble profile check: {'PASS' if ensemble_ok else 'FAIL'}")

        if ensemble_ok:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            next_num = max([int(f.stem.split("_")[1]) for f in existing]) + 1 if existing else 137

            test_meta = load_metadata(train=False)
            sub_df = pd.DataFrame({
                "id": test_meta["id"],
                "angle": ensemble_preds[:, 0],
                "depth": ensemble_preds[:, 1],
                "left_right": ensemble_preds[:, 2]
            })

            sub_path = SUBMISSION_DIR / f"submission_{next_num}.csv"
            sub_df.to_csv(sub_path, index=False)
            print(f"\nSubmission saved to: {sub_path}")
    else:
        print("\nNo models passed profile check - no ensemble created")

    # Save results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "test_preds"} for r in subset_results])
    results_df.to_csv(OUTPUT_DIR / "subspace_ensemble_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'subspace_ensemble_results.csv'}")


if __name__ == "__main__":
    main()
