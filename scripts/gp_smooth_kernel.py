"""
GP with Smoother Kernel for Better Generalization

The Matern 0.5 (rough kernel) overfit. Try smoother kernels
that might generalize better even with slightly worse within-player CV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
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
    all_features = []
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])
        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    feature_df = pd.DataFrame(all_features).fillna(0)
    if feature_names:
        available = [f for f in feature_names if f in feature_df.columns]
        feature_df = feature_df[available]
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    return feature_df.values, np.array(targets) if train else None, np.array(participant_ids)


def within_player_cv(X, y, pids, kernel, alpha):
    """Within-player CV."""
    scalers_dict = load_scalers()
    target_names = ["angle", "depth", "left_right"]
    all_preds = np.zeros_like(y)

    for pid in np.unique(pids):
        mask = pids == pid
        X_pid, y_pid = X[mask], y[mask]
        indices = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X_pid):
            for t_idx in range(3):
                gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=2, random_state=42)
                gp.fit(X_scaled[train_idx], y_pid[train_idx, t_idx])
                all_preds[indices[test_idx], t_idx] = gp.predict(X_scaled[test_idx])

    mse = {}
    for i, name in enumerate(target_names):
        y_s = scalers_dict[name].transform(y[:, i].reshape(-1, 1)).ravel()
        p_s = scalers_dict[name].transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse[name] = np.mean((y_s - p_s) ** 2)

    return np.mean(list(mse.values())), mse


def main():
    print("=" * 60)
    print("GP WITH SMOOTHER KERNELS")
    print("=" * 60)

    scalers_dict = load_scalers()
    stable_features = get_stable_features(30)

    print("\nBuilding features...")
    X_train, y_train, pids_train = build_feature_matrix(True, stable_features)
    X_test, _, pids_test = build_feature_matrix(False, stable_features)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Test smoother kernels with higher alpha (more regularization)
    configs = [
        ("RBF", ConstantKernel(1.0) * RBF(length_scale=1.0), 1.0),
        ("RBF", ConstantKernel(1.0) * RBF(length_scale=1.0), 2.0),
        ("RBF", ConstantKernel(1.0) * RBF(length_scale=1.0), 5.0),
        ("Matern 2.5", ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5), 1.0),
        ("Matern 2.5", ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5), 2.0),
        ("Matern 2.5", ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5), 5.0),
    ]

    results = []
    for name, kernel, alpha in configs:
        print(f"\nTesting {name}, alpha={alpha}...")
        mse, per_target = within_player_cv(X_train, y_train, pids_train, kernel, alpha)
        print(f"  MSE: {mse:.6f} (angle={per_target['angle']:.4f}, depth={per_target['depth']:.4f}, lr={per_target['left_right']:.4f})")
        results.append({"kernel": name, "alpha": alpha, "mse": mse, **per_target})

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["mse"].idxmin()]
    print(f"\nBest: {best['kernel']}, alpha={best['alpha']}, MSE={best['mse']:.6f}")

    # Create submission with best smooth kernel
    print("\n" + "=" * 60)
    print("CREATING SUBMISSION")
    print("=" * 60)

    if best['kernel'] == "RBF":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    else:
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

    predictions = np.zeros((len(X_test), 3))
    target_names = ["angle", "depth", "left_right"]

    for test_pid in np.unique(pids_test):
        test_mask = pids_test == test_pid
        train_mask = pids_train == test_pid

        if train_mask.sum() > 0:
            X_tr, y_tr = X_train[train_mask], y_train[train_mask]
        else:
            X_tr, y_tr = X_train, y_train

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            gp = GaussianProcessRegressor(kernel=kernel, alpha=best['alpha'], n_restarts_optimizer=3, random_state=42)
            gp.fit(X_tr_s, y_tr[:, t_idx])
            predictions[test_mask, t_idx] = gp.predict(X_te_s)

    # Scale
    scaled = np.zeros_like(predictions)
    for i, name in enumerate(target_names):
        scaled[:, i] = scalers_dict[name].transform(predictions[:, i].reshape(-1, 1)).ravel()

    # Profile
    print(f"\nSubmission profile:")
    print(f"  angle: mean={scaled[:, 0].mean():.4f}, std={scaled[:, 0].std():.4f}")
    print(f"  depth: mean={scaled[:, 1].mean():.4f}, std={scaled[:, 1].std():.4f}")
    print(f"  lr: mean={scaled[:, 2].mean():.4f}, std={scaled[:, 2].std():.4f}")

    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    print(f"\nSub 133 (LB 0.007809):")
    print(f"  angle: mean={sub133['scaled_angle'].mean():.4f}, std={sub133['scaled_angle'].std():.4f}")
    print(f"  depth: mean={sub133['scaled_depth'].mean():.4f}, std={sub133['scaled_depth'].std():.4f}")
    print(f"  lr: mean={sub133['scaled_left_right'].mean():.4f}, std={sub133['scaled_left_right'].std():.4f}")

    # Save
    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv") if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        "id": load_metadata(False)["id"].values,
        "scaled_angle": scaled[:, 0],
        "scaled_depth": scaled[:, 1],
        "scaled_left_right": scaled[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Correlation
    print(f"\nCorrelation with Sub 133:")
    print(f"  angle: {np.corrcoef(scaled[:, 0], sub133['scaled_angle'])[0,1]:.4f}")
    print(f"  depth: {np.corrcoef(scaled[:, 1], sub133['scaled_depth'])[0,1]:.4f}")
    print(f"  lr: {np.corrcoef(scaled[:, 2], sub133['scaled_left_right'])[0,1]:.4f}")


if __name__ == "__main__":
    main()
