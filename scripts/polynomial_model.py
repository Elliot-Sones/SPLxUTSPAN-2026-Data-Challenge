"""
Polynomial Interaction Features with Heavy Regularization (Plan 2.2)

Degree-2 polynomial on top stable features only, with alpha scaling with feature count.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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


def load_stable_features(top_n: int = 20) -> list:
    """Load top N features by stability_adjusted score."""
    drift_df = pd.read_csv(OUTPUT_DIR / "feature_drift_f4_with_importance.csv")
    drift_df = drift_df.sort_values("stability_adjusted", ascending=False)
    stable_features = drift_df.head(top_n)["feature"].tolist()
    return stable_features


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

    # Build DataFrame
    feature_df = pd.DataFrame(all_features)

    # Filter to selected features
    if selected_features:
        available = [f for f in selected_features if f in feature_df.columns]
        print(f"  Using {len(available)} of {len(selected_features)} features")
        feature_df = feature_df[available]

    # Fill NaN
    feature_df = feature_df.fillna(feature_df.median())

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def create_polynomial_features(X, degree=2, include_bias=False):
    """Create polynomial features."""
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


def evaluate_lopo_cv(X, y, pids, alpha=100.0, use_poly=True, degree=2, n_base_features=20):
    """LOPO CV with optional polynomial features."""
    scalers_dict = load_scalers()

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Scale first
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply polynomial features if requested
        if use_poly:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_test_poly = poly.transform(X_test_scaled)
        else:
            X_train_poly = X_train_scaled
            X_test_poly = X_test_scaled

        for target_idx in range(3):
            # Scale alpha with number of polynomial features
            n_poly_features = X_train_poly.shape[1]
            effective_alpha = alpha * (n_poly_features / n_base_features)

            model = Ridge(alpha=effective_alpha)
            model.fit(X_train_poly, y_train[:, target_idx])
            all_preds[test_mask, target_idx] = model.predict(X_test_poly)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def evaluate_within_player_cv(X, y, pids, alpha=100.0, use_poly=True, degree=2, n_base_features=20, n_folds=5):
    """Within-player CV with polynomial features."""
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

            # Polynomial features
            if use_poly:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_train_poly = poly.fit_transform(X_train_scaled)
                X_val_poly = poly.transform(X_val_scaled)
            else:
                X_train_poly = X_train_scaled
                X_val_poly = X_val_scaled

            for target_idx in range(3):
                n_poly_features = X_train_poly.shape[1]
                effective_alpha = alpha * (n_poly_features / n_base_features)

                model = Ridge(alpha=effective_alpha)
                model.fit(X_train_poly, y_train[:, target_idx])

                global_idx = np.where(mask)[0][val_idx]
                all_preds[global_idx, target_idx] = model.predict(X_val_poly)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def create_submission(X_train, y_train, pids_train, X_test, pids_test, alpha=100.0, degree=2, n_base_features=20):
    """Create submission with polynomial features."""
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

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pid)
        X_test_scaled = scaler.transform(X_test_pid)

        # Polynomial
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        for target_idx in range(3):
            n_poly_features = X_train_poly.shape[1]
            effective_alpha = alpha * (n_poly_features / n_base_features)

            model = Ridge(alpha=effective_alpha)
            model.fit(X_train_poly, y_train_pid[:, target_idx])
            test_preds[test_mask, target_idx] = model.predict(X_test_poly)

    return test_preds


def main():
    print("=" * 60)
    print("POLYNOMIAL INTERACTION FEATURES (Plan 2.2)")
    print("=" * 60)

    results = []

    # Test different numbers of base features and polynomial degrees
    for n_base in [10, 15, 20, 25, 30]:
        print(f"\n--- Testing with {n_base} base features ---")

        stable_features = load_stable_features(n_base)
        X, y, pids, feature_names = build_feature_matrix(train=True, selected_features=stable_features)
        print(f"  X shape: {X.shape}")

        # Baseline (no polynomial)
        for alpha in [50, 100, 200, 500]:
            baseline_lopo, baseline_per_target = evaluate_lopo_cv(
                X, y, pids, alpha=alpha, use_poly=False, n_base_features=n_base
            )

            # Polynomial degree 2
            poly2_lopo, poly2_per_target = evaluate_lopo_cv(
                X, y, pids, alpha=alpha, use_poly=True, degree=2, n_base_features=n_base
            )

            # Interaction only (no squared terms)
            # poly_interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            # ... skipped for brevity, can add if needed

            results.append({
                "n_base_features": n_base,
                "alpha": alpha,
                "baseline_lopo": baseline_lopo,
                "poly2_lopo": poly2_lopo,
                "improvement_pct": (baseline_lopo - poly2_lopo) / baseline_lopo * 100,
                "baseline_angle": baseline_per_target["angle"],
                "baseline_depth": baseline_per_target["depth"],
                "baseline_lr": baseline_per_target["left_right"],
                "poly2_angle": poly2_per_target["angle"],
                "poly2_depth": poly2_per_target["depth"],
                "poly2_lr": poly2_per_target["left_right"],
            })

            improvement = (baseline_lopo - poly2_lopo) / baseline_lopo * 100
            print(f"  alpha={alpha}: baseline={baseline_lopo:.6f}, poly2={poly2_lopo:.6f}, improvement={improvement:.2f}%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "polynomial_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'polynomial_results.csv'}")

    # Find best configuration
    best_idx = results_df["poly2_lopo"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n=== BEST POLYNOMIAL CONFIGURATION ===")
    print(f"n_base_features: {best['n_base_features']}")
    print(f"alpha: {best['alpha']}")
    print(f"Baseline LOPO: {best['baseline_lopo']:.6f}")
    print(f"Poly2 LOPO: {best['poly2_lopo']:.6f}")
    print(f"Improvement: {best['improvement_pct']:.2f}%")

    # Check if polynomial actually helps
    if best["poly2_lopo"] >= best["baseline_lopo"]:
        print("\n*** WARNING: Polynomial features did NOT improve over baseline ***")
        print("This is expected - polynomial features often overfit on small datasets")

    # Create submission with best config if it passes profile check
    print(f"\n--- Creating submission with best config ---")

    n_base = int(best["n_base_features"])
    stable_features = load_stable_features(n_base)

    X_train, y_train, pids_train, _ = build_feature_matrix(train=True, selected_features=stable_features)
    X_test, _, pids_test, _ = build_feature_matrix(train=False, selected_features=stable_features)

    test_preds = create_submission(X_train, y_train, pids_train, X_test, pids_test,
                                   alpha=best["alpha"], degree=2, n_base_features=n_base)

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
        print(f"Poly2 LOPO CV: {best['poly2_lopo']:.6f}")
    else:
        print("\nSubmission not created - profile check failed")


if __name__ == "__main__":
    main()
