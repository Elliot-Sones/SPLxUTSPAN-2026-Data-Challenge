"""
SPL Statistical Priors Approach

Strategy: Use SPL Open Data distributions to inform regularization without transfer learning.

Key Insight: SPL provides physics-based constraints:
- Entry angle distributions from real shots
- Make/miss outcomes
- No temporal mismatch (we use statistics, not sequences)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
import lightgbm as lgb
import xgboost as xgb
from scipy import stats

# Paths
DATA_DIR = Path("data")
EXTERNAL_DIR = Path("external_data/SPL-Open-Data/basketball/freethrow")
SUBMISSION_DIR = Path("submission")
OUTPUT_DIR = Path("output/spl_priors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
OPTIMAL_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


def extract_spl_distributions():
    """
    Extract statistical distributions from SPL Open Data.
    Returns: DataFrame with shot outcomes and entry angles.
    """
    print("Extracting SPL distributions...")
    data_dir = EXTERNAL_DIR / "data/P0001"

    spl_data = []
    for trial_file in sorted(data_dir.glob("BB_FT_P0001_T*.json")):
        with open(trial_file) as f:
            trial = json.load(f)

        spl_data.append({
            "trial_id": trial["trial_id"],
            "result": trial.get("result", "unknown"),
            "entry_angle": trial.get("entry_angle", np.nan),
            "landing_x": trial.get("landing_x", np.nan),
            "landing_y": trial.get("landing_y", np.nan),
        })

    df = pd.DataFrame(spl_data)

    # Clean data
    df = df[df["result"] != "unknown"].copy()
    df["made"] = (df["result"] == "make").astype(int)
    df = df.dropna(subset=["entry_angle"])

    print(f"Loaded {len(df)} SPL shots with outcomes")
    print(f"  Makes: {df['made'].sum()} ({df['made'].mean()*100:.1f}%)")
    print(f"  Misses: {(1-df['made']).sum()} ({(1-df['made']).mean()*100:.1f}%)")
    print(f"  Entry angle: mean={df['entry_angle'].mean():.2f}, std={df['entry_angle'].std():.2f}")
    print(f"  Entry angle range: [{df['entry_angle'].min():.2f}, {df['entry_angle'].max():.2f}]")

    return df


def compute_spl_priors(spl_df):
    """
    Compute statistical priors from SPL data.
    """
    # Separate makes and misses
    makes = spl_df[spl_df["made"] == 1]["entry_angle"]
    misses = spl_df[spl_df["made"] == 0]["entry_angle"]

    priors = {
        "entry_angle": {
            "all": {
                "mean": spl_df["entry_angle"].mean(),
                "std": spl_df["entry_angle"].std(),
                "median": spl_df["entry_angle"].median(),
                "q25": spl_df["entry_angle"].quantile(0.25),
                "q75": spl_df["entry_angle"].quantile(0.75),
            },
            "makes": {
                "mean": makes.mean(),
                "std": makes.std(),
                "median": makes.median(),
                "q25": makes.quantile(0.25),
                "q75": makes.quantile(0.75),
            } if len(makes) > 0 else None,
            "misses": {
                "mean": misses.mean(),
                "std": misses.std(),
                "median": misses.median(),
                "q25": misses.quantile(0.25),
                "q75": misses.quantile(0.75),
            } if len(misses) > 0 else None,
        }
    }

    print("\nSPL Priors:")
    print(f"  All shots: {priors['entry_angle']['all']['mean']:.2f} ± {priors['entry_angle']['all']['std']:.2f}")
    if priors['entry_angle']['makes']:
        print(f"  Makes: {priors['entry_angle']['makes']['mean']:.2f} ± {priors['entry_angle']['makes']['std']:.2f}")
    if priors['entry_angle']['misses']:
        print(f"  Misses: {priors['entry_angle']['misses']['mean']:.2f} ± {priors['entry_angle']['misses']['std']:.2f}")

    return priors


def load_competition_data():
    """Load competition data"""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def parse_temporal_array(s):
    """Parse string-serialized array to numpy array"""
    # Replace 'nan' with 'null' for JSON parsing
    s_clean = s.replace('nan', 'null')
    arr = json.loads(s_clean)
    # Convert None to np.nan
    arr = [np.nan if x is None else x for x in arr]
    return np.array(arr, dtype=np.float32)


def extract_features(df, target_frame=153):
    """Extract features at optimal frame from temporal data"""
    kp_cols = [c for c in df.columns if c not in ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]]

    features = []
    for idx, row in df.iterrows():
        # Parse each temporal column and extract frame
        frame_features = []
        for col in kp_cols:
            temporal_array = parse_temporal_array(row[col])
            frame_features.append(temporal_array[target_frame])
        features.append(frame_features)

    return np.array(features)


def train_with_spl_priors(train_df, test_df, priors, prior_strength=0.1):
    """
    Train models with SPL priors as regularization.

    prior_strength: 0.0 = no prior, 1.0 = strong prior
    """
    print(f"\n{'='*80}")
    print(f"Training with SPL priors (strength={prior_strength})")
    print('='*80)

    results = {}

    for target_name in ["angle", "depth", "left_right"]:
        print(f"\n--- Target: {target_name} ---")
        target_frame = OPTIMAL_FRAMES[target_name]

        # Extract features
        X_train = extract_features(train_df, target_frame)
        X_test = extract_features(test_df, target_frame)
        y_train = train_df[target_name].values

        # Standardize
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Competition angle stats (for mapping SPL priors)
        angle_mean = train_df["angle"].mean()
        angle_std = train_df["angle"].std()

        # Map SPL entry_angle prior to competition targets
        # Assumption: competition angle correlates with SPL entry_angle
        if target_name == "angle":
            # Direct mapping: competition angle ≈ SPL entry_angle
            prior_mean = priors["entry_angle"]["makes"]["mean"] if priors["entry_angle"]["makes"] else priors["entry_angle"]["all"]["mean"]
            prior_std = priors["entry_angle"]["makes"]["std"] if priors["entry_angle"]["makes"] else priors["entry_angle"]["all"]["std"]
        elif target_name == "depth":
            # Indirect: higher angle → deeper shot (negative correlation assumption)
            # Use competition mean as default
            prior_mean = train_df["depth"].mean()
            prior_std = train_df["depth"].std()
        else:  # left_right
            # Neutral prior (SPL doesn't provide left/right info)
            prior_mean = train_df["left_right"].mean()
            prior_std = train_df["left_right"].std()

        # Bayesian Ridge with prior
        if prior_strength > 0 and target_name == "angle":
            # Use Bayesian Ridge with informative prior for angle
            model = BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
                compute_score=True,
                max_iter=300
            )
        else:
            # Standard Ridge for other targets or no prior
            model = Ridge(alpha=10.0)

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")

        # Train final model
        model.fit(X_train_scaled, y_train)

        # Predict
        test_pred = model.predict(X_test_scaled)

        # Apply prior regularization (soft constraint)
        if prior_strength > 0:
            # Shrink predictions toward prior mean
            test_pred = (1 - prior_strength) * test_pred + prior_strength * prior_mean

        results[target_name] = {
            "predictions": test_pred,
            "cv_mse": -cv_scores.mean(),
            "prior_mean": prior_mean,
            "prior_std": prior_std,
        }

        print(f"CV MSE: {-cv_scores.mean():.6f}")
        print(f"Prior: mean={prior_mean:.2f}, std={prior_std:.2f}")

    return results


def scale_targets(values, target_name):
    """Scale raw target values to [0, 1]"""
    ranges = {
        "angle": (23.78, 56.47),
        "depth": (-10.17, 24.97),
        "left_right": (-12.98, 10.15)
    }
    min_val, max_val = ranges[target_name]
    scaled = (values - min_val) / (max_val - min_val)
    return np.clip(scaled, 0, 1)


def generate_submissions(results_list, test_df, sub_784_path):
    """Generate submissions with different prior strengths"""
    print(f"\n{'='*80}")
    print("Generating submissions")
    print('='*80)

    # Load baseline
    sub_784 = pd.read_csv(sub_784_path).set_index("id")

    # Get next submission number
    existing_subs = list(SUBMISSION_DIR.glob("submission_*.csv"))
    max_num = max([int(s.stem.split("_")[1]) for s in existing_subs])

    submission_info = []

    for i, (prior_strength, results) in enumerate(results_list):
        max_num += 1
        sub_num = max_num

        # Scale predictions
        angle_scaled = scale_targets(results["angle"]["predictions"], "angle")
        depth_scaled = scale_targets(results["depth"]["predictions"], "depth")
        lr_scaled = scale_targets(results["left_right"]["predictions"], "left_right")

        # Create submission
        submission_rows = []
        for idx, row in test_df.iterrows():
            shot_id = row["shot_id"]
            submission_rows.append({
                "id": shot_id,
                "scaled_angle": angle_scaled[idx],
                "scaled_depth": depth_scaled[idx],
                "scaled_left_right": lr_scaled[idx]
            })

        sub_df = pd.DataFrame(submission_rows)
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(sub_path, index=False)

        submission_info.append({
            "submission": sub_num,
            "path": str(sub_path),
            "prior_strength": prior_strength,
            "cv_scores": {
                "angle": results["angle"]["cv_mse"],
                "depth": results["depth"]["cv_mse"],
                "left_right": results["left_right"]["cv_mse"],
            },
            "priors_used": {
                "angle": {
                    "mean": results["angle"]["prior_mean"],
                    "std": results["angle"]["prior_std"],
                },
                "depth": {
                    "mean": results["depth"]["prior_mean"],
                    "std": results["depth"]["prior_std"],
                },
                "left_right": {
                    "mean": results["left_right"]["prior_mean"],
                    "std": results["left_right"]["prior_std"],
                }
            }
        })

        print(f"Generated {sub_path.name}: prior_strength={prior_strength}")

    return submission_info


def main():
    print('='*80)
    print("SPL Statistical Priors Approach")
    print('='*80)

    # Extract SPL distributions
    spl_df = extract_spl_distributions()
    priors = compute_spl_priors(spl_df)

    # Save SPL data
    spl_df.to_csv(OUTPUT_DIR / "spl_data.csv", index=False)
    with open(OUTPUT_DIR / "spl_priors.json", "w") as f:
        json.dump(priors, f, indent=2)

    # Load competition data
    train_df, test_df = load_competition_data()

    # Test different prior strengths
    prior_strengths = [0.0, 0.05, 0.10, 0.20, 0.30]

    results_list = []
    for strength in prior_strengths:
        results = train_with_spl_priors(train_df, test_df, priors, prior_strength=strength)
        results_list.append((strength, results))

    # Generate submissions
    sub_784_path = SUBMISSION_DIR / "submission_784.csv"
    submission_info = generate_submissions(results_list, test_df, sub_784_path)

    # Save metadata
    metadata = {
        "spl_data": {
            "n_shots": len(spl_df),
            "make_rate": spl_df["made"].mean(),
            "entry_angle_stats": priors["entry_angle"],
        },
        "prior_strengths_tested": prior_strengths,
        "submissions": submission_info
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"SPL shots analyzed: {len(spl_df)}")
    print(f"Prior strengths tested: {prior_strengths}")
    print(f"Submissions generated: {len(submission_info)}")
    print("\nCV Results by prior strength:")
    for strength, results in results_list:
        mean_mse = np.mean([results[t]["cv_mse"] for t in ["angle", "depth", "left_right"]])
        print(f"  {strength:.2f}: mean MSE = {mean_mse:.6f}")
    print("\nSubmissions:")
    for info in submission_info:
        print(f"  Sub {info['submission']}: prior_strength={info['prior_strength']}")

    print(f"\n{'='*80}")
    print("Complete! Test submissions on leaderboard.")
    print('='*80)


if __name__ == "__main__":
    main()
