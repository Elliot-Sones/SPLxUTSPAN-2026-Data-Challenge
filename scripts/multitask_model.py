"""
Multi-Task Learning (Plan 1.2)

Train single model predicting all 3 targets jointly using MultiTaskElasticNet.
Hypothesis: Shared representations between targets may help generalization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV
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


def build_feature_matrix(train: bool = True):
    """Build feature matrix with all features."""
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
    feature_df = feature_df.fillna(feature_df.median())

    # Drop columns with all zeros or all same value
    feature_df = feature_df.loc[:, (feature_df != feature_df.iloc[0]).any()]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def scale_targets(y, scalers_dict):
    """Scale targets to [0, 1] range using saved scalers."""
    y_scaled = np.zeros_like(y)
    for i, target in enumerate(["angle", "depth", "left_right"]):
        y_scaled[:, i] = scalers_dict[target].transform(y[:, i].reshape(-1, 1)).ravel()
    return y_scaled


def inverse_scale_targets(y_scaled, scalers_dict):
    """Inverse scale targets back to original range."""
    y = np.zeros_like(y_scaled)
    for i, target in enumerate(["angle", "depth", "left_right"]):
        y[:, i] = scalers_dict[target].inverse_transform(y_scaled[:, i].reshape(-1, 1)).ravel()
    return y


def evaluate_multitask_lopo(X, y, pids, l1_ratio=0.5, alpha=0.1):
    """Leave-One-Participant-Out CV with MultiTaskElasticNet."""
    scalers_dict = load_scalers()

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Scale targets for multitask learning
        y_train_scaled = scale_targets(y_train, scalers_dict)

        # Train multitask model
        model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
        model.fit(X_train_scaled, y_train_scaled)

        # Predict and inverse scale
        pred_scaled = model.predict(X_test_scaled)
        all_preds[test_mask] = inverse_scale_targets(pred_scaled, scalers_dict)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target, all_preds


def evaluate_multitask_per_player(X, y, pids, l1_ratio=0.5, alpha=0.1, n_folds=5):
    """Per-player MultiTaskElasticNet with internal CV."""
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

            # Scale features
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)

            # Scale targets
            y_train_scaled = scale_targets(y_train, scalers_dict)

            # Train multitask model
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
            model.fit(X_train_scaled, y_train_scaled)

            # Predict and inverse scale
            pred_scaled = model.predict(X_val_scaled)

            # Map back to global indices
            global_idx = np.where(mask)[0][val_idx]
            all_preds[global_idx] = inverse_scale_targets(pred_scaled, scalers_dict)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def compare_with_separate_models(X, y, pids, l1_ratio=0.5, alpha=0.1):
    """Compare multitask with separate ElasticNet per target."""
    from sklearn.linear_model import ElasticNet

    scalers_dict = load_scalers()

    unique_pids = np.unique(pids)
    all_preds_separate = np.zeros_like(y)

    for pid in unique_pids:
        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Train separate models per target
        for target_idx, target_name in enumerate(["angle", "depth", "left_right"]):
            # Scale target
            scaler_y = scalers_dict[target_name]
            y_train_target = scaler_y.transform(y_train[:, target_idx].reshape(-1, 1)).ravel()

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
            model.fit(X_train_scaled, y_train_target)

            pred_scaled = model.predict(X_test_scaled)
            all_preds_separate[test_mask, target_idx] = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds_separate[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def create_submission(X_train, y_train, pids_train, X_test, pids_test, l1_ratio=0.5, alpha=0.1):
    """Create submission using per-player multitask models."""
    scalers_dict = load_scalers()
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

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_pid)
        X_test_scaled = scaler_X.transform(X_test_pid)

        # Scale targets
        y_train_scaled = scale_targets(y_train_pid, scalers_dict)

        # Train multitask model
        model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
        model.fit(X_train_scaled, y_train_scaled)

        # Predict and inverse scale
        pred_scaled = model.predict(X_test_scaled)
        test_preds[test_mask] = inverse_scale_targets(pred_scaled, scalers_dict)

    return test_preds


def main():
    print("=" * 60)
    print("MULTI-TASK LEARNING (Plan 1.2)")
    print("=" * 60)

    print("\nBuilding feature matrix...")
    X, y, pids, feature_names = build_feature_matrix(train=True)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {len(feature_names)}")

    results = []

    # Test different hyperparameter combinations
    print("\n--- Grid Search over alpha and l1_ratio ---")

    for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
            print(f"\nTesting l1_ratio={l1_ratio}, alpha={alpha}...")

            # LOPO CV with multitask
            lopo_mse, lopo_per_target, _ = evaluate_multitask_lopo(X, y, pids, l1_ratio=l1_ratio, alpha=alpha)

            # Per-player CV with multitask
            pp_mse, pp_per_target = evaluate_multitask_per_player(X, y, pids, l1_ratio=l1_ratio, alpha=alpha)

            # Compare with separate models
            sep_mse, sep_per_target = compare_with_separate_models(X, y, pids, l1_ratio=l1_ratio, alpha=alpha)

            results.append({
                "l1_ratio": l1_ratio,
                "alpha": alpha,
                "multitask_lopo_mse": lopo_mse,
                "multitask_pp_mse": pp_mse,
                "separate_lopo_mse": sep_mse,
                "multitask_angle": lopo_per_target["angle"],
                "multitask_depth": lopo_per_target["depth"],
                "multitask_lr": lopo_per_target["left_right"],
                "separate_angle": sep_per_target["angle"],
                "separate_depth": sep_per_target["depth"],
                "separate_lr": sep_per_target["left_right"],
            })

            print(f"  Multitask LOPO: {lopo_mse:.6f}, Separate: {sep_mse:.6f}")
            print(f"  Multitask benefit: {(sep_mse - lopo_mse) / sep_mse * 100:.2f}%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "multitask_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'multitask_results.csv'}")

    # Find best configuration
    best_idx = results_df["multitask_lopo_mse"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n=== BEST MULTITASK CONFIGURATION ===")
    print(f"l1_ratio: {best['l1_ratio']}")
    print(f"alpha: {best['alpha']}")
    print(f"Multitask LOPO MSE: {best['multitask_lopo_mse']:.6f}")
    print(f"Separate LOPO MSE: {best['separate_lopo_mse']:.6f}")
    print(f"Multitask benefit: {(best['separate_lopo_mse'] - best['multitask_lopo_mse']) / best['separate_lopo_mse'] * 100:.2f}%")

    # Create submission with best config
    print(f"\n--- Creating submission with best config ---")

    X_test, _, pids_test, _ = build_feature_matrix(train=False)
    test_preds = create_submission(X, y, pids, X_test, pids_test,
                                   l1_ratio=best["l1_ratio"], alpha=best["alpha"])

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
        print(f"Multitask LOPO CV: {best['multitask_lopo_mse']:.6f}")
    else:
        print("\nSubmission not created - profile check failed")


if __name__ == "__main__":
    main()
