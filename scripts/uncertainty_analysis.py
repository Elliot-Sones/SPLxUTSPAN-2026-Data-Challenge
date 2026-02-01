"""
Uncertainty Analysis

Identify which samples have high prediction uncertainty and
see if we can improve predictions on those specific samples.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def main():
    print("=" * 70)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 70)

    scalers_dict = load_scalers()

    # Build features
    print("\nLoading data...")
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    train_features = []
    train_targets = []
    train_pids = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        train_features.append(features)
        train_targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        train_pids.append(metadata["participant_id"])

    train_df = pd.DataFrame(train_features).fillna(0)
    train_df = train_df.loc[:, train_df.nunique() > 1]

    X_train = train_df.values
    y_train = np.array(train_targets)
    pids_train = np.array(train_pids)

    test_features = []
    test_pids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        test_features.append(features)
        test_pids.append(metadata["participant_id"])

    test_df = pd.DataFrame(test_features).fillna(0)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    X_test = test_df[train_df.columns].values
    pids_test = np.array(test_pids)

    target_names = ["angle", "depth", "left_right"]

    # Use bagging to estimate prediction uncertainty
    print("\nEstimating prediction uncertainty via bagging...")

    n_estimators = 20
    test_preds_all = np.zeros((len(X_test), 3, n_estimators))
    train_cv_preds_all = np.zeros((len(y_train), 3, n_estimators))

    for pid in np.unique(pids_train):
        train_mask = pids_train == pid
        test_mask = pids_test == pid

        X_tr = X_train[train_mask]
        y_tr = y_train[train_mask]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            # Bagging ensemble
            for e in range(n_estimators):
                # Bootstrap sample
                boot_idx = np.random.choice(len(X_tr_scaled), len(X_tr_scaled), replace=True)
                X_boot = X_tr_scaled[boot_idx]
                y_boot = y_tr[boot_idx, t_idx]

                ridge = Ridge(alpha=100)
                ridge.fit(X_boot, y_boot)

                test_preds_all[test_mask, t_idx, e] = ridge.predict(X_te_scaled)

    # Compute uncertainty (std across ensemble)
    test_preds_mean = test_preds_all.mean(axis=2)
    test_preds_std = test_preds_all.std(axis=2)

    # Scale predictions and uncertainty
    scaled_mean = np.zeros_like(test_preds_mean)
    scaled_std = np.zeros_like(test_preds_std)

    for i, target in enumerate(target_names):
        scaled_mean[:, i] = scalers_dict[target].transform(test_preds_mean[:, i].reshape(-1, 1)).ravel()
        # Scale std by the same factor
        scale = scalers_dict[target].scale_[0]
        scaled_std[:, i] = test_preds_std[:, i] * scale

    print("\nUncertainty statistics (scaled):")
    for i, target in enumerate(target_names):
        print(f"  {target}: mean_std={scaled_std[:, i].mean():.4f}, max_std={scaled_std[:, i].max():.4f}")

    # Load Sub 133 for comparison
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    # Analyze high-uncertainty samples
    print("\n" + "=" * 70)
    print("HIGH UNCERTAINTY SAMPLES")
    print("=" * 70)

    test_meta = load_metadata(train=False)

    for i, (target, col) in enumerate(zip(target_names, cols)):
        print(f"\n{target}:")

        # Get samples with highest uncertainty
        top_uncertain_idx = np.argsort(scaled_std[:, i])[-5:]

        for idx in top_uncertain_idx:
            pid = test_meta.iloc[idx]["participant_id"]
            pred = scaled_mean[idx, i]
            std = scaled_std[idx, i]
            sub133_pred = sub133[col].iloc[idx]
            diff = abs(pred - sub133_pred)

            print(f"  Sample {idx} (P{pid}): pred={pred:.4f}+/-{std:.4f}, "
                  f"Sub133={sub133_pred:.4f}, diff={diff:.4f}")

    # Create uncertainty-weighted blend
    print("\n" + "=" * 70)
    print("UNCERTAINTY-WEIGHTED BLENDING")
    print("=" * 70)

    # For high-uncertainty samples, trust Sub 133 more
    # For low-uncertainty samples, trust our predictions more

    total_std = scaled_std.sum(axis=1)
    norm_uncertainty = total_std / total_std.max()

    # Weight: high uncertainty -> low weight for our predictions
    our_weight = 0.3 * (1 - norm_uncertainty)

    uncertainty_blend = np.zeros_like(scaled_mean)
    for i in range(3):
        uncertainty_blend[:, i] = (1 - our_weight) * sub133[cols[i]].values + our_weight * scaled_mean[:, i]

    print(f"Weight range: [{our_weight.min():.3f}, {our_weight.max():.3f}]")
    print(f"\nUncertainty blend profile:")
    print(f"  angle_std: {uncertainty_blend[:, 0].std():.4f}")
    print(f"  depth_mean: {uncertainty_blend[:, 1].mean():.4f}")

    corr = np.corrcoef(uncertainty_blend.ravel(), sub133[cols].values.ravel())[0, 1]
    print(f"  Correlation with Sub 133: {corr:.4f}")

    # Try opposite: trust our predictions more where Sub 133 disagrees
    print("\n" + "=" * 70)
    print("DISAGREEMENT-BASED BLENDING")
    print("=" * 70)

    disagreement = np.abs(scaled_mean - sub133[cols].values)
    total_disagreement = disagreement.sum(axis=1)
    norm_disagreement = total_disagreement / total_disagreement.max()

    # Where they disagree, average them
    disagree_weight = 0.5 * norm_disagreement

    disagree_blend = np.zeros_like(scaled_mean)
    for i in range(3):
        disagree_blend[:, i] = (1 - disagree_weight) * sub133[cols[i]].values + disagree_weight * scaled_mean[:, i]

    print(f"\nDisagreement blend profile:")
    print(f"  angle_std: {disagree_blend[:, 0].std():.4f}")
    print(f"  depth_mean: {disagree_blend[:, 1].mean():.4f}")

    corr = np.corrcoef(disagree_blend.ravel(), sub133[cols].values.ravel())[0, 1]
    print(f"  Correlation with Sub 133: {corr:.4f}")

    # Save submissions
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    # Save uncertainty blend
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": uncertainty_blend[:, 0],
        "scaled_depth": uncertainty_blend[:, 1],
        "scaled_left_right": uncertainty_blend[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved uncertainty blend: {path}")

    next_num += 1

    # Save disagreement blend
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": disagree_blend[:, 0],
        "scaled_depth": disagree_blend[:, 1],
        "scaled_left_right": disagree_blend[:, 2]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved disagreement blend: {path}")


if __name__ == "__main__":
    main()
