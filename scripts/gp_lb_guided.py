"""
LB-Guided GP Model

Use what we know from LB to guide GP configuration:
1. Match Sub 133's profile (angle_std, depth_mean)
2. Use GP as residual corrector on top of Ridge
3. Blend GP only where it agrees with Sub 133
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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


def approach_1_profile_constrained():
    """
    Approach 1: Train GP but constrain predictions to match Sub 133's profile.
    Post-process GP predictions to have same mean/std as Sub 133.
    """
    print("\n" + "=" * 60)
    print("APPROACH 1: Profile-Constrained GP")
    print("=" * 60)

    scalers_dict = load_scalers()
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    # Target profile from Sub 133
    target_profile = {
        "angle": {"mean": sub133["scaled_angle"].mean(), "std": sub133["scaled_angle"].std()},
        "depth": {"mean": sub133["scaled_depth"].mean(), "std": sub133["scaled_depth"].std()},
        "left_right": {"mean": sub133["scaled_left_right"].mean(), "std": sub133["scaled_left_right"].std()},
    }

    stable_features = get_stable_features(30)
    X_train, y_train, pids_train = build_feature_matrix(True, stable_features)
    X_test, _, pids_test = build_feature_matrix(False, stable_features)

    predictions = np.zeros((len(X_test), 3))
    target_names = ["angle", "depth", "left_right"]

    # Use smoother kernel (Matern 2.5) with high regularization
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

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
            gp = GaussianProcessRegressor(kernel=kernel, alpha=5.0, n_restarts_optimizer=2, random_state=42)
            gp.fit(X_tr_s, y_tr[:, t_idx])
            predictions[test_mask, t_idx] = gp.predict(X_te_s)

    # Scale to [0,1] range
    scaled = np.zeros_like(predictions)
    for i, name in enumerate(target_names):
        scaled[:, i] = scalers_dict[name].transform(predictions[:, i].reshape(-1, 1)).ravel()

    # Now constrain to match Sub 133's profile
    constrained = np.zeros_like(scaled)
    for i, name in enumerate(target_names):
        # Standardize then rescale to target profile
        z_scores = (scaled[:, i] - scaled[:, i].mean()) / (scaled[:, i].std() + 1e-8)
        constrained[:, i] = z_scores * target_profile[name]["std"] + target_profile[name]["mean"]

    print(f"\nOriginal GP profile:")
    print(f"  angle: mean={scaled[:, 0].mean():.4f}, std={scaled[:, 0].std():.4f}")
    print(f"  depth: mean={scaled[:, 1].mean():.4f}, std={scaled[:, 1].std():.4f}")
    print(f"  lr: mean={scaled[:, 2].mean():.4f}, std={scaled[:, 2].std():.4f}")

    print(f"\nConstrained to Sub 133 profile:")
    print(f"  angle: mean={constrained[:, 0].mean():.4f}, std={constrained[:, 0].std():.4f}")
    print(f"  depth: mean={constrained[:, 1].mean():.4f}, std={constrained[:, 1].std():.4f}")
    print(f"  lr: mean={constrained[:, 2].mean():.4f}, std={constrained[:, 2].std():.4f}")

    # Correlation with Sub 133 (should be same as before constraint since we only shift/scale)
    print(f"\nCorrelation with Sub 133:")
    print(f"  angle: {np.corrcoef(constrained[:, 0], sub133['scaled_angle'])[0,1]:.4f}")
    print(f"  depth: {np.corrcoef(constrained[:, 1], sub133['scaled_depth'])[0,1]:.4f}")
    print(f"  lr: {np.corrcoef(constrained[:, 2], sub133['scaled_left_right'])[0,1]:.4f}")

    return constrained, "profile_constrained"


def approach_2_residual_gp():
    """
    Approach 2: Use GP to correct Ridge predictions.
    Train Ridge first, then GP on residuals.
    """
    print("\n" + "=" * 60)
    print("APPROACH 2: Residual GP (Ridge + GP correction)")
    print("=" * 60)

    scalers_dict = load_scalers()
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    stable_features = get_stable_features(30)
    X_train, y_train, pids_train = build_feature_matrix(True, stable_features)
    X_test, _, pids_test = build_feature_matrix(False, stable_features)

    target_names = ["angle", "depth", "left_right"]
    ridge_preds_train = np.zeros_like(y_train)
    ridge_preds_test = np.zeros((len(X_test), 3))
    final_preds = np.zeros((len(X_test), 3))

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

    for test_pid in np.unique(pids_test):
        test_mask = pids_test == test_pid
        train_mask = pids_train == test_pid

        if train_mask.sum() > 0:
            X_tr, y_tr = X_train[train_mask], y_train[train_mask]
            train_indices = np.where(train_mask)[0]
        else:
            X_tr, y_tr = X_train, y_train
            train_indices = np.arange(len(X_train))

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test[test_mask])

        for t_idx in range(3):
            # Step 1: Ridge prediction
            ridge = Ridge(alpha=100)
            ridge.fit(X_tr_s, y_tr[:, t_idx])
            ridge_pred_train = ridge.predict(X_tr_s)
            ridge_pred_test = ridge.predict(X_te_s)

            ridge_preds_train[train_indices, t_idx] = ridge_pred_train
            ridge_preds_test[test_mask, t_idx] = ridge_pred_test

            # Step 2: GP on residuals
            residuals = y_tr[:, t_idx] - ridge_pred_train

            gp = GaussianProcessRegressor(kernel=kernel, alpha=2.0, n_restarts_optimizer=2, random_state=42)
            gp.fit(X_tr_s, residuals)
            residual_correction = gp.predict(X_te_s)

            # Combine: Ridge + small weight * GP correction
            correction_weight = 0.3  # Conservative - don't trust GP too much
            final_preds[test_mask, t_idx] = ridge_pred_test + correction_weight * residual_correction

    # Scale
    scaled = np.zeros_like(final_preds)
    for i, name in enumerate(target_names):
        scaled[:, i] = scalers_dict[name].transform(final_preds[:, i].reshape(-1, 1)).ravel()

    print(f"\nRidge + GP Residual profile:")
    print(f"  angle: mean={scaled[:, 0].mean():.4f}, std={scaled[:, 0].std():.4f}")
    print(f"  depth: mean={scaled[:, 1].mean():.4f}, std={scaled[:, 1].std():.4f}")
    print(f"  lr: mean={scaled[:, 2].mean():.4f}, std={scaled[:, 2].std():.4f}")

    print(f"\nCorrelation with Sub 133:")
    print(f"  angle: {np.corrcoef(scaled[:, 0], sub133['scaled_angle'])[0,1]:.4f}")
    print(f"  depth: {np.corrcoef(scaled[:, 1], sub133['scaled_depth'])[0,1]:.4f}")
    print(f"  lr: {np.corrcoef(scaled[:, 2], sub133['scaled_left_right'])[0,1]:.4f}")

    return scaled, "residual_gp"


def approach_3_selective_blend():
    """
    Approach 3: Use GP only where it agrees with Sub 133.
    If GP and Sub 133 disagree significantly, trust Sub 133.
    """
    print("\n" + "=" * 60)
    print("APPROACH 3: Selective Blend (trust Sub 133 when GP disagrees)")
    print("=" * 60)

    scalers_dict = load_scalers()
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    stable_features = get_stable_features(30)
    X_train, y_train, pids_train = build_feature_matrix(True, stable_features)
    X_test, _, pids_test = build_feature_matrix(False, stable_features)

    target_names = ["angle", "depth", "left_right"]
    gp_preds = np.zeros((len(X_test), 3))

    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

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
            gp = GaussianProcessRegressor(kernel=kernel, alpha=5.0, n_restarts_optimizer=2, random_state=42)
            gp.fit(X_tr_s, y_tr[:, t_idx])
            gp_preds[test_mask, t_idx] = gp.predict(X_te_s)

    # Scale GP predictions
    gp_scaled = np.zeros_like(gp_preds)
    for i, name in enumerate(target_names):
        gp_scaled[:, i] = scalers_dict[name].transform(gp_preds[:, i].reshape(-1, 1)).ravel()

    # Get Sub 133 predictions
    sub133_preds = sub133[["scaled_angle", "scaled_depth", "scaled_left_right"]].values

    # Selective blend: weight GP more when it agrees with Sub 133
    blended = np.zeros_like(gp_scaled)
    col_names = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    for i in range(3):
        gp_col = gp_scaled[:, i]
        sub_col = sub133_preds[:, i]

        # Compute per-sample agreement (smaller diff = more agreement)
        diff = np.abs(gp_col - sub_col)
        diff_normalized = diff / (diff.max() + 1e-8)

        # When diff is small (agreement), weight GP more; when large, trust Sub 133
        # gp_weight = 0.3 when agreement, 0 when disagreement
        gp_weight = 0.3 * (1 - diff_normalized)

        blended[:, i] = (1 - gp_weight) * sub_col + gp_weight * gp_col

    print(f"\nSelective Blend profile:")
    print(f"  angle: mean={blended[:, 0].mean():.4f}, std={blended[:, 0].std():.4f}")
    print(f"  depth: mean={blended[:, 1].mean():.4f}, std={blended[:, 1].std():.4f}")
    print(f"  lr: mean={blended[:, 2].mean():.4f}, std={blended[:, 2].std():.4f}")

    print(f"\nCorrelation with Sub 133:")
    print(f"  angle: {np.corrcoef(blended[:, 0], sub133['scaled_angle'])[0,1]:.4f}")
    print(f"  depth: {np.corrcoef(blended[:, 1], sub133['scaled_depth'])[0,1]:.4f}")
    print(f"  lr: {np.corrcoef(blended[:, 2], sub133['scaled_left_right'])[0,1]:.4f}")

    return blended, "selective_blend"


def main():
    print("=" * 60)
    print("LB-GUIDED GP APPROACHES")
    print("=" * 60)

    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    print("\nSub 133 profile (LB 0.007809):")
    print(f"  angle: mean={sub133['scaled_angle'].mean():.4f}, std={sub133['scaled_angle'].std():.4f}")
    print(f"  depth: mean={sub133['scaled_depth'].mean():.4f}, std={sub133['scaled_depth'].std():.4f}")
    print(f"  lr: mean={sub133['scaled_left_right'].mean():.4f}, std={sub133['scaled_left_right'].std():.4f}")

    # Run all approaches
    results = []

    preds1, name1 = approach_1_profile_constrained()
    results.append((preds1, name1))

    preds2, name2 = approach_2_residual_gp()
    results.append((preds2, name2))

    preds3, name3 = approach_3_selective_blend()
    results.append((preds3, name3))

    # Save all submissions
    print("\n" + "=" * 60)
    print("SAVING SUBMISSIONS")
    print("=" * 60)

    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv") if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    test_meta = load_metadata(False)

    for preds, name in results:
        submission = pd.DataFrame({
            "id": test_meta["id"].values,
            "scaled_angle": preds[:, 0],
            "scaled_depth": preds[:, 1],
            "scaled_left_right": preds[:, 2]
        })
        path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(path, index=False)
        print(f"Saved {name}: {path}")
        next_num += 1

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
    1. Profile-Constrained GP: Forces same mean/std as Sub 133.
       - Same correlation as raw GP, but matches winning profile.
       - Worth trying if profile matching is key to LB.

    2. Residual GP: Ridge baseline + GP correction.
       - More conservative, builds on proven Ridge approach.
       - GP only corrects Ridge's mistakes.

    3. Selective Blend: Trust Sub 133 when GP disagrees.
       - Safest approach - degrades gracefully to Sub 133.
       - Only uses GP where it agrees with winner.
    """)


if __name__ == "__main__":
    main()
