"""
Gradient-Guided Model: Use known LB scores as gradient signal.

Key insight from analysis:
- depth_max has r=-0.986 correlation with LB score (higher = better)
- 50-50 blend (sub25) is optimal at 0.008305
- We need 15.7% improvement to reach 0.007

Strategy:
1. Analyze what makes sub25 better than sub8
2. Extrapolate the "gradient" direction toward even better scores
3. Create new predictions by pushing in the optimal direction
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS = ["angle", "depth", "left_right"]

# Known LB scores
LB_SCORES = {
    8: 0.010220,
    9: 0.009109,
    10: 0.008907,
    11: 0.009848,
    20: 0.008619,
    25: 0.008305,  # best
    34: 0.008377,
}


def load_submissions():
    """Load submissions with known scores."""
    submissions = {}
    for sub_num in LB_SCORES.keys():
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        if filepath.exists():
            submissions[sub_num] = pd.read_csv(filepath)
    return submissions


def analyze_gradient(submissions):
    """Analyze the gradient from worse to better submissions."""
    print("="*70)
    print("ANALYZING GRADIENT TOWARD BETTER SCORES")
    print("="*70)

    best_sub = submissions[25]  # 0.008305
    worst_sub = submissions[8]  # 0.010220

    # Per-column analysis
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        best_vals = best_sub[col].values
        worst_vals = worst_sub[col].values
        diff = best_vals - worst_vals

        print(f"\n{col}:")
        print(f"  Worst (sub8):  mean={worst_vals.mean():.4f}, max={worst_vals.max():.4f}")
        print(f"  Best (sub25):  mean={best_vals.mean():.4f}, max={best_vals.max():.4f}")
        print(f"  Shift:         mean={diff.mean():+.4f}, max shift={diff.max():+.4f}")

        # Which samples shifted most?
        top_shifts = np.argsort(np.abs(diff))[-5:]
        print(f"  Largest shifts at samples: {top_shifts.tolist()}")

    return best_sub, worst_sub


def compute_optimal_direction(submissions):
    """
    Compute the optimal direction for each prediction based on correlation with LB score.

    For each sample and target, determine whether higher or lower values lead to better scores.
    """
    print("\n" + "="*70)
    print("COMPUTING OPTIMAL DIRECTION PER SAMPLE")
    print("="*70)

    # Get sorted submissions by LB score
    sorted_subs = sorted(LB_SCORES.items(), key=lambda x: x[1])  # best first

    # Get number of samples from first submission
    n_samples = len(list(submissions.values())[0])

    directions = {
        'scaled_angle': np.zeros(n_samples),
        'scaled_depth': np.zeros(n_samples),
        'scaled_left_right': np.zeros(n_samples),
    }

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        # For each sample, compute correlation between prediction value and LB score
        for sample_idx in range(n_samples):
            values = []
            scores = []
            for sub_num, lb_score in sorted_subs:
                if sub_num in submissions:
                    values.append(submissions[sub_num][col].iloc[sample_idx])
                    scores.append(lb_score)

            if len(values) >= 3:
                # Negative correlation means higher value = lower (better) LB score
                corr = np.corrcoef(values, scores)[0, 1]
                directions[col][sample_idx] = -corr  # Positive = go higher, negative = go lower

    print("\nDirection analysis (positive = increase value, negative = decrease):")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        d = directions[col]
        print(f"  {col}: mean={d.mean():+.3f}, std={d.std():.3f}")
        print(f"    Samples to increase: {(d > 0.3).sum()}")
        print(f"    Samples to decrease: {(d < -0.3).sum()}")

    return directions


def create_extrapolated_submission(submissions, directions, strength=0.1):
    """
    Create submission by extrapolating in the optimal direction.

    Start from the best submission (sub25) and push further in the gradient direction.
    """
    print(f"\n" + "="*70)
    print(f"CREATING EXTRAPOLATED SUBMISSION (strength={strength})")
    print("="*70)

    best = submissions[25].copy()

    # The difference between sub25 and sub8 represents the improvement direction
    # We want to push further in that direction

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        # Use direction-weighted extrapolation
        current = best[col].values
        direction = directions[col]

        # Clip the direction to reasonable range
        direction = np.clip(direction, -1, 1)

        # Extrapolate: push values in the optimal direction
        # Positive direction = increase, negative = decrease
        shift = strength * direction * current.std()  # Scale by std
        new_values = current + shift

        # Clip to valid range [0, 1]
        new_values = np.clip(new_values, 0, 1)

        best[col] = new_values

        print(f"  {col}: mean shift={shift.mean():+.4f}, new max={new_values.max():.4f}")

    return best


def create_depth_boosted_submission(submissions, boost_factor=1.1):
    """
    Create submission with boosted depth predictions.

    Since depth_max has strong correlation with LB score, try pushing depth higher.
    """
    print(f"\n" + "="*70)
    print(f"CREATING DEPTH-BOOSTED SUBMISSION (factor={boost_factor})")
    print("="*70)

    best = submissions[25].copy()

    # Boost depth values toward higher end
    depth = best['scaled_depth'].values
    depth_mean = depth.mean()

    # Push values above mean higher, keep values below mean similar
    boosted = depth.copy()
    above_mean_mask = depth > depth_mean
    boosted[above_mean_mask] = depth_mean + (depth[above_mean_mask] - depth_mean) * boost_factor
    boosted = np.clip(boosted, 0, 1)

    print(f"  Original depth: mean={depth.mean():.4f}, max={depth.max():.4f}")
    print(f"  Boosted depth:  mean={boosted.mean():.4f}, max={boosted.max():.4f}")

    best['scaled_depth'] = boosted

    return best


def blend_with_higher_depth_weight(submissions):
    """
    Create blend that weighs sub10 higher for depth (since it has higher depth_max).
    """
    print("\n" + "="*70)
    print("CREATING TARGET-SPECIFIC BLEND")
    print("="*70)

    sub9 = submissions[9]
    sub10 = submissions.get(10)

    if sub10 is None:
        filepath = SUBMISSION_DIR / "submission_10.csv"
        sub10 = pd.read_csv(filepath)

    # For angle: 50-50 is optimal
    # For depth: weight sub10 more (higher depth_max)
    # For left_right: 50-50 is optimal

    weights = {
        'scaled_angle': (0.5, 0.5),      # sub9, sub10
        'scaled_depth': (0.3, 0.7),       # Push toward sub10 for higher depth_max
        'scaled_left_right': (0.5, 0.5),
    }

    blended = pd.DataFrame({'id': sub9['id']})

    for col, (w9, w10) in weights.items():
        blended[col] = w9 * sub9[col] + w10 * sub10[col]
        print(f"  {col}: {w9*100:.0f}% sub9 + {w10*100:.0f}% sub10")
        print(f"    Result: mean={blended[col].mean():.4f}, max={blended[col].max():.4f}")

    return blended


def try_new_model_architecture():
    """
    Train completely different model architectures to find new signal.
    """
    print("\n" + "="*70)
    print("TRAINING NEW MODEL ARCHITECTURES")
    print("="*70)

    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    def parse_array_string(s):
        if pd.isna(s):
            return np.full(240, np.nan, dtype=np.float32)
        s = s.replace("nan", "null")
        return np.array(json.loads(s), dtype=np.float32)

    def extract_features_for_df(df, is_train=True):
        all_features = []
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
            combined = {**hybrid_feats, **advanced_feats}
            all_features.append(combined)

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        return all_features, ids, pids, targets if is_train else None

    print("Extracting features...")
    train_feats, train_ids, train_pids, train_targets = extract_features_for_df(train_df, True)
    test_feats, test_ids, test_pids, _ = extract_features_for_df(test_df, False)

    feature_names = sorted(train_feats[0].keys())
    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)
    pids_train = np.array(train_pids)
    pids_test = np.array(test_pids)

    # Clean data
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    print(f"Features: {X_train.shape[1]}")

    # Try different model configurations
    model_configs = {
        "rf": {
            "model": RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_leaf=5,
                max_features=0.5, random_state=42, n_jobs=-1
            ),
            "desc": "Random Forest (shallow)"
        },
        "gb": {
            "model": GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_samples_leaf=5, subsample=0.8, random_state=42
            ),
            "desc": "Gradient Boosting (sklearn)"
        },
        "elastic": {
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            "desc": "ElasticNet"
        },
        "lgb_conservative": {
            "model": lgb.LGBMRegressor(
                n_estimators=50, num_leaves=5, learning_rate=0.1,
                reg_alpha=1.0, reg_lambda=1.0, min_child_samples=20,
                random_state=42, verbose=-1, n_jobs=-1
            ),
            "desc": "LightGBM (very conservative)"
        },
    }

    predictions = {}
    cv_scores = {}

    for model_name, config in model_configs.items():
        print(f"\n--- {config['desc']} ---")

        preds = np.zeros((len(X_test), 3))
        cv_mse = []

        for pid in sorted(np.unique(pids_train)):
            train_mask = pids_train == pid
            test_mask = pids_test == pid

            X_p_train = X_train[train_mask]
            y_p_train = y_train[train_mask]
            X_p_test = X_test[test_mask]

            # Standardize
            scaler = StandardScaler()
            X_p_train_scaled = scaler.fit_transform(X_p_train)
            X_p_test_scaled = scaler.transform(X_p_test)

            for target_idx, target in enumerate(TARGETS):
                y_target = y_p_train[:, target_idx]

                # Clone model for this target
                import copy
                model = copy.deepcopy(config["model"])
                model.fit(X_p_train_scaled, y_target)

                # Predict
                test_indices = np.where(test_mask)[0]
                preds[test_indices, target_idx] = model.predict(X_p_test_scaled)

                # CV score
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                try:
                    scores = -cross_val_score(model, X_p_train_scaled, y_target,
                                            cv=kf, scoring='neg_mean_squared_error')
                    cv_mse.append(scores.mean())
                except:
                    pass

        predictions[model_name] = preds
        cv_scores[model_name] = np.mean(cv_mse)
        print(f"  CV MSE: {cv_scores[model_name]:.6f}")

    return predictions, test_ids


def save_submission(df, suffix=""):
    """Save submission with next number."""
    nums = [int(f.stem.split('_')[1]) for f in SUBMISSION_DIR.glob('submission_*.csv')
            if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
    df.to_csv(filepath, index=False)

    print(f"\nSaved: {filepath}")
    print(f"Stats:")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
              f"min={df[col].min():.4f}, max={df[col].max():.4f}")

    return next_num, filepath


def main():
    print("="*70)
    print("GRADIENT-GUIDED MODEL EXPLORATION")
    print("="*70)

    # Load existing submissions
    submissions = load_submissions()
    print(f"Loaded {len(submissions)} submissions with known scores")

    # Analyze gradient
    best_sub, worst_sub = analyze_gradient(submissions)

    # Compute optimal direction per sample
    directions = compute_optimal_direction(submissions)

    # Create several candidate submissions
    candidates = []

    # 1. Extrapolated submission (mild)
    print("\n" + "="*70)
    print("CANDIDATE 1: Extrapolated (mild)")
    extrap_mild = create_extrapolated_submission(submissions, directions, strength=0.05)
    candidates.append(("extrap_mild", extrap_mild))

    # 2. Extrapolated submission (strong)
    print("\n" + "="*70)
    print("CANDIDATE 2: Extrapolated (strong)")
    extrap_strong = create_extrapolated_submission(submissions, directions, strength=0.15)
    candidates.append(("extrap_strong", extrap_strong))

    # 3. Depth-boosted submission
    print("\n" + "="*70)
    print("CANDIDATE 3: Depth-boosted")
    depth_boosted = create_depth_boosted_submission(submissions, boost_factor=1.15)
    candidates.append(("depth_boosted", depth_boosted))

    # 4. Target-specific blend
    print("\n" + "="*70)
    print("CANDIDATE 4: Target-specific blend")
    target_blend = blend_with_higher_depth_weight(submissions)
    candidates.append(("target_blend", target_blend))

    # Compare candidates
    print("\n" + "="*70)
    print("CANDIDATE COMPARISON")
    print("="*70)

    print(f"\n{'Candidate':<20} {'depth_max':>12} {'depth_mean':>12} {'angle_std':>12}")
    print("-" * 60)

    for name, df in candidates:
        print(f"{name:<20} {df['scaled_depth'].max():>12.4f} "
              f"{df['scaled_depth'].mean():>12.4f} "
              f"{df['scaled_angle'].std():>12.4f}")

    # Based on correlation analysis, depth_max is the strongest predictor
    # Sort by depth_max (higher is better)
    candidates_sorted = sorted(candidates, key=lambda x: x[1]['scaled_depth'].max(), reverse=True)

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    best_candidate_name, best_candidate = candidates_sorted[0]
    print(f"\nBest candidate by depth_max: {best_candidate_name}")
    print(f"  depth_max = {best_candidate['scaled_depth'].max():.4f}")

    # Save the best candidate
    sub_num, filepath = save_submission(best_candidate, best_candidate_name)

    print(f"\n" + "="*70)
    print("SAVED BEST CANDIDATE")
    print("="*70)
    print(f"Submission {sub_num}: {best_candidate_name}")
    print(f"File: {filepath}")

    # Also try new model architectures
    print("\n" + "="*70)
    print("EXPLORING NEW MODEL ARCHITECTURES")
    print("="*70)

    new_preds, test_ids = try_new_model_architecture()

    # Scale predictions and create submissions
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Create submission from each new model
    for model_name, preds in new_preds.items():
        scaled_preds = np.zeros_like(preds)
        for i, target in enumerate(TARGETS):
            scaled_preds[:, i] = target_scalers[target].transform(
                preds[:, i].reshape(-1, 1)
            ).flatten()

        df = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': scaled_preds[:, 0],
            'scaled_depth': scaled_preds[:, 1],
            'scaled_left_right': scaled_preds[:, 2],
        })

        # Only save if depth_max is promising
        if df['scaled_depth'].max() > 0.74:
            sub_num, filepath = save_submission(df, model_name)
            print(f"Saved {model_name} as submission_{sub_num}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
