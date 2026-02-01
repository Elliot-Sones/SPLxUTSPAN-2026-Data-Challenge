"""
Optimal Submission Blend Search

Goal: Find blend weights that minimize within-player CV while maintaining good profile.

Approach:
1. Use submissions with good profiles as candidates
2. Optimize blend weights using within-player CV
3. Constraint: keep profile close to Sub 133
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, get_keypoint_columns
from hybrid_features import init_keypoint_mapping, extract_hybrid_features

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"


def get_train_predictions_from_cv():
    """Get CV predictions on training data using same method as successful submissions."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    scalers_dict = load_scalers()

    # Build features
    feature_list = []
    y_list = []
    pids_list = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_hybrid_features(timeseries, metadata["participant_id"])
        feature_list.append(features)
        y_list.append([metadata["angle"], metadata["depth"], metadata["left_right"]])
        pids_list.append(metadata["participant_id"])

    df = pd.DataFrame(feature_list).fillna(0)
    df = df.loc[:, df.nunique() > 1]

    X = df.values
    y = np.array(y_list)
    pids = np.array(pids_list)

    target_names = ["angle", "depth", "left_right"]

    # Within-player CV
    all_preds = np.zeros_like(y)

    for pid in np.unique(pids):
        mask = pids == pid
        X_pid = X[mask]
        y_pid = y[mask]
        indices = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pid)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_pid):
            for t_idx in range(3):
                ridge = Ridge(alpha=100)
                ridge.fit(X_scaled[train_idx], y_pid[train_idx, t_idx])
                all_preds[indices[test_idx], t_idx] = ridge.predict(X_scaled[test_idx])

    # Scale predictions
    scaled_preds = np.zeros_like(all_preds)
    scaled_targets = np.zeros_like(y)
    for i, name in enumerate(target_names):
        scaled_preds[:, i] = scalers_dict[name].transform(all_preds[:, i].reshape(-1, 1)).ravel()
        scaled_targets[:, i] = scalers_dict[name].transform(y[:, i].reshape(-1, 1)).ravel()

    return scaled_preds, scaled_targets, pids


def load_submissions(sub_nums):
    """Load multiple submissions."""
    subs = {}
    for num in sub_nums:
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        if path.exists():
            df = pd.read_csv(path)
            subs[num] = df[["scaled_angle", "scaled_depth", "scaled_left_right"]].values
    return subs


def blend_predictions(weights, predictions_list):
    """Blend predictions with given weights."""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    blended = np.zeros_like(predictions_list[0])
    for w, pred in zip(weights, predictions_list):
        blended += w * pred
    return blended


def main():
    print("=" * 70)
    print("OPTIMAL SUBMISSION BLEND SEARCH")
    print("=" * 70)

    # First, get CV predictions on training data
    print("\nStep 1: Building within-player CV predictions on training data...")
    cv_preds, cv_targets, pids = get_train_predictions_from_cv()

    # Calculate baseline CV MSE
    baseline_mse = np.mean((cv_preds - cv_targets) ** 2)
    print(f"Baseline within-player CV MSE: {baseline_mse:.6f}")

    # Profile of CV predictions
    print(f"CV preds profile: angle_std={cv_preds[:, 0].std():.4f}, depth_mean={cv_preds[:, 1].mean():.4f}")

    # Load candidate submissions
    print("\nStep 2: Loading candidate submissions...")

    # Candidates: submissions with good profiles
    candidates = [25, 29, 37, 40, 54, 81, 115, 118, 125, 127, 131, 132, 133, 135, 136]

    subs = {}
    for num in candidates:
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        if path.exists():
            df = pd.read_csv(path)
            profile_dist = abs(df["scaled_angle"].std() - 0.1377) + abs(df["scaled_depth"].mean() - 0.5055)
            subs[num] = {
                "preds": df[["scaled_angle", "scaled_depth", "scaled_left_right"]].values,
                "profile_dist": profile_dist,
                "angle_std": df["scaled_angle"].std(),
                "depth_mean": df["scaled_depth"].mean()
            }
            print(f"  Sub {num}: angle_std={subs[num]['angle_std']:.4f}, depth_mean={subs[num]['depth_mean']:.4f}")

    # Step 3: Test blending with Sub 133 as anchor
    print("\nStep 3: Testing blends starting from Sub 133...")

    sub133 = subs[133]["preds"]

    # Test each other submission blended with 133
    results = []

    for num in subs:
        if num == 133:
            continue

        other = subs[num]["preds"]

        # Find optimal blend weight
        best_weight = 0
        best_corr = -1

        for w in np.linspace(0, 0.5, 21):
            blended = (1 - w) * sub133 + w * other

            # Compute correlation with CV predictions
            # Higher correlation = better proxy for actual accuracy
            corr = np.corrcoef(blended.ravel(), cv_preds.ravel())[0, 1]

            if corr > best_corr:
                best_corr = corr
                best_weight = w

        blended = (1 - best_weight) * sub133 + best_weight * other
        angle_std = np.std(blended[:, 0])
        depth_mean = np.mean(blended[:, 1])
        profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

        results.append({
            "blend_with": num,
            "weight": best_weight,
            "corr_with_cv": best_corr,
            "angle_std": angle_std,
            "depth_mean": depth_mean,
            "profile_dist": profile_dist
        })

    results_df = pd.DataFrame(results).sort_values("corr_with_cv", ascending=False)
    print("\nBlend results (sorted by correlation with CV):")
    print(results_df.to_string(index=False))

    # Step 4: Multi-submission optimization
    print("\n" + "=" * 70)
    print("Step 4: Multi-submission blend optimization")
    print("=" * 70)

    # Use top submissions that add diversity
    top_subs = [133, 118, 132, 25, 127, 135]
    pred_list = [subs[n]["preds"] for n in top_subs]

    def objective(weights):
        weights = np.abs(weights)  # Keep positive
        weights = weights / weights.sum()

        blended = np.zeros_like(pred_list[0])
        for w, pred in zip(weights, pred_list):
            blended += w * pred

        # Objective: maximize correlation with CV predictions + profile match
        corr = np.corrcoef(blended.ravel(), cv_preds.ravel())[0, 1]

        angle_std = np.std(blended[:, 0])
        depth_mean = np.mean(blended[:, 1])
        profile_penalty = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

        # Minimize negative correlation + profile penalty
        return -corr + 5 * profile_penalty

    # Optimize
    from scipy.optimize import differential_evolution

    n_subs = len(top_subs)
    bounds = [(0, 1)] * n_subs

    result = differential_evolution(objective, bounds, seed=42, maxiter=500, tol=1e-6)

    opt_weights = np.abs(result.x)
    opt_weights = opt_weights / opt_weights.sum()

    print("\nOptimal weights:")
    for num, w in zip(top_subs, opt_weights):
        if w > 0.01:
            print(f"  Sub {num}: {w:.3f}")

    # Create optimized blend
    blended = np.zeros_like(pred_list[0])
    for w, pred in zip(opt_weights, pred_list):
        blended += w * pred

    print(f"\nOptimal blend profile:")
    print(f"  angle_std: {np.std(blended[:, 0]):.4f} (target: 0.1377)")
    print(f"  depth_mean: {np.mean(blended[:, 1]):.4f} (target: 0.5055)")
    print(f"  Correlation with CV: {np.corrcoef(blended.ravel(), cv_preds.ravel())[0, 1]:.4f}")

    # Compare to Sub 133
    corr_133 = np.corrcoef(sub133.ravel(), cv_preds.ravel())[0, 1]
    corr_opt = np.corrcoef(blended.ravel(), cv_preds.ravel())[0, 1]

    print(f"\nComparison:")
    print(f"  Sub 133 corr with CV: {corr_133:.4f}")
    print(f"  Optimal blend corr:   {corr_opt:.4f}")
    print(f"  Improvement: {(corr_opt - corr_133) / corr_133 * 100:.2f}%")

    # Save submission
    if corr_opt > corr_133:
        print("\n*** Optimal blend has better correlation ***")

        test_meta = load_metadata(train=False)

        nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
                if f.stem.split("_")[1].isdigit()]
        next_num = max(nums) + 1

        submission = pd.DataFrame({
            "id": test_meta["id"].values,
            "scaled_angle": blended[:, 0],
            "scaled_depth": blended[:, 1],
            "scaled_left_right": blended[:, 2]
        })

        path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(path, index=False)
        print(f"\nSaved: {path}")
        print(f"Weights: {dict(zip(top_subs, [f'{w:.3f}' for w in opt_weights]))}")
    else:
        print("\nNo improvement over Sub 133")


if __name__ == "__main__":
    main()
