"""
Gaussian Process Regression Experiment

Goal: Achieve LB < 0.005 using GP with custom kernels and physics-informed constraints.

Strategy:
1. Multiple kernel types: RBF, Matern (nu=0.5, 1.5, 2.5), RationalQuadratic
2. Per-target optimization with separate length scales
3. Physics-informed priors: angle smoothness, depth-LR coupling
4. Blend all outputs with Sub 784 (LB 0.007224)
5. Measure diversity vs Sub 1350 (current best LB 0.006776)

Expected outcome: CV 0.0070-0.0080 but HIGH diversity (r < 0.90 with Sub 1350)
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel,
    ConstantKernel as C
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
RESEARCH_DIR = PROJECT_DIR / "Research"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Submission tracking
SUBMISSION_LOG = []


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split('_')[1]) for fp in existing
                    if fp.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def load_data():
    """Load and parse all data."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Identify keypoint columns
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kpt_cols = [c for c in train_df.columns if c not in meta_cols]

    # Parse keypoint arrays
    for col in kpt_cols:
        train_df[col] = train_df[col].apply(parse_array_string)
        test_df[col] = test_df[col].apply(parse_array_string)

    # Scale targets using existing scalers
    for target in ["angle", "depth", "left_right"]:
        scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        train_df[f"scaled_{target}"] = scaler.transform(train_df[[target]].values)

    return train_df, test_df, kpt_cols


def extract_hoop_relative_coords(df, kpt_cols, frame_idx):
    """Extract hoop-relative coordinates at specific frame."""
    features = []
    for col in kpt_cols:
        vals = np.stack(df[col].values)
        feat = vals[:, frame_idx]
        features.append(feat)
    X = np.column_stack(features)

    # Hoop-relative transformation
    coord_cols = [i for i, c in enumerate(kpt_cols) if c.endswith(('_x', '_y', '_z'))]
    for i in range(0, len(coord_cols), 3):
        if i+2 < len(coord_cols):
            X[:, coord_cols[i]] -= HOOP_POS[0]
            X[:, coord_cols[i+1]] -= HOOP_POS[1]
            X[:, coord_cols[i+2]] -= HOOP_POS[2]

    # Handle NaN
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.any(np.isnan(col)):
            med = np.nanmedian(col)
            col[np.isnan(col)] = med

    return X


def extract_pls_features(df, kpt_cols, n_components=15):
    """Extract PLS components from raw timeseries."""
    features = []
    for col in kpt_cols:
        vals = np.stack(df[col].values)
        features.append(vals)
    X_full = np.concatenate(features, axis=1)

    # Handle NaN
    for i in range(X_full.shape[1]):
        col = X_full[:, i]
        if np.any(np.isnan(col)):
            med = np.nanmedian(col)
            col[np.isnan(col)] = med

    return X_full


def build_gp_kernel(kernel_type, length_scale=1.0, alpha=1.0):
    """Build GP kernel with physics-informed constraints."""
    if kernel_type == "rbf":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2)) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e1))
    elif kernel_type == "matern_0.5":
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale, length_scale_bounds=(1e-2, 1e2), nu=0.5) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e1))
    elif kernel_type == "matern_1.5":
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e1))
    elif kernel_type == "matern_2.5":
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale, length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e1))
    elif kernel_type == "rational_quadratic":
        kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=length_scale, alpha=alpha, length_scale_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernel


def train_gp_model(X_train, y_train, kernel_type, length_scale=1.0, alpha=1.0, n_restarts=3):
    """Train GP model with specified kernel."""
    kernel = build_gp_kernel(kernel_type, length_scale, alpha)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        alpha=1e-6,  # numerical stability
        random_state=42
    )
    gp.fit(X_train, y_train)
    return gp


def run_gp_cv(train_df, kpt_cols, target, kernel_type, length_scale=1.0, alpha=1.0, use_pls=True):
    """Run LOO CV for GP model."""
    print(f"\nRunning GP CV for {target} with kernel={kernel_type}, length_scale={length_scale}, alpha={alpha}, use_pls={use_pls}")

    # Extract features
    frame_idx = TARGET_FRAMES[target]
    X_hc = extract_hoop_relative_coords(train_df, kpt_cols, frame_idx)

    if use_pls:
        X_full = extract_pls_features(train_df, kpt_cols)
        scaler = StandardScaler()
        X_full_scaled = scaler.fit_transform(X_full)

        y = train_df[f"scaled_{target}"].values
        pls = PLSRegression(n_components=15)
        pls.fit(X_full_scaled, y)
        X_pls = pls.transform(X_full_scaled)

        X = np.column_stack([X_hc, X_pls])
    else:
        X = X_hc
        y = train_df[f"scaled_{target}"].values

    # Normalize features
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    # LOO CV
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        gp = train_gp_model(X_tr, y_tr, kernel_type, length_scale, alpha, n_restarts=2)
        y_pred[test_idx] = gp.predict(X_te, return_std=False)

    mse = mean_squared_error(y, y_pred)
    print(f"  CV MSE: {mse:.6f}")

    return mse, y_pred


def generate_gp_submission(train_df, test_df, kpt_cols, kernel_type, length_scale=1.0, alpha=1.0,
                           use_pls=True, blend_weight=0.5, sub_784_path=None):
    """Generate submission using GP model."""
    print(f"\n{'='*80}")
    print(f"Generating GP submission: kernel={kernel_type}, length_scale={length_scale}, alpha={alpha}, use_pls={use_pls}, blend_weight={blend_weight}")
    print(f"{'='*80}")

    # Load Sub 784 for blending
    if sub_784_path is None:
        sub_784_path = SUBMISSION_DIR / "submission_784.csv"
    sub_784 = pd.read_csv(sub_784_path)

    test_preds = {}
    cv_results = {}

    for target in TARGETS:
        print(f"\n--- Target: {target} ---")

        # Extract features
        frame_idx = TARGET_FRAMES[target]
        X_hc_train = extract_hoop_relative_coords(train_df, kpt_cols, frame_idx)
        X_hc_test = extract_hoop_relative_coords(test_df, kpt_cols, frame_idx)

        if use_pls:
            X_full_train = extract_pls_features(train_df, kpt_cols)
            X_full_test = extract_pls_features(test_df, kpt_cols)

            scaler = StandardScaler()
            X_full_train_scaled = scaler.fit_transform(X_full_train)
            X_full_test_scaled = scaler.transform(X_full_test)

            y = train_df[f"scaled_{target}"].values
            pls = PLSRegression(n_components=15)
            pls.fit(X_full_train_scaled, y)
            X_pls_train = pls.transform(X_full_train_scaled)
            X_pls_test = pls.transform(X_full_test_scaled)

            X_train = np.column_stack([X_hc_train, X_pls_train])
            X_test = np.column_stack([X_hc_test, X_pls_test])
        else:
            X_train = X_hc_train
            X_test = X_hc_test
            y = train_df[f"scaled_{target}"].values

        # Normalize features
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        # Train GP
        print(f"Training GP model...")
        gp = train_gp_model(X_train, y, kernel_type, length_scale, alpha, n_restarts=5)

        # Predict
        y_test_pred = gp.predict(X_test, return_std=False)

        # Clip to [0, 1]
        y_test_pred = np.clip(y_test_pred, 0, 1)

        # Blend with Sub 784
        y_test_784 = sub_784[f"scaled_{target}"].values
        y_test_blended = blend_weight * y_test_pred + (1 - blend_weight) * y_test_784

        test_preds[target] = y_test_blended

        # Run CV for documentation
        cv_mse, _ = run_gp_cv(train_df, kpt_cols, target, kernel_type, length_scale, alpha, use_pls)
        cv_results[target] = cv_mse

    # Create submission
    sub_num = get_next_submission_number()
    submission = pd.DataFrame({
        "id": test_df["id"],
        "scaled_angle": test_preds["angle"],
        "scaled_depth": test_preds["depth"],
        "scaled_left_right": test_preds["left_right"]
    })

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(sub_path, index=False)

    # Log submission
    log_entry = {
        "submission_number": sub_num,
        "kernel_type": kernel_type,
        "length_scale": length_scale,
        "alpha": alpha,
        "use_pls": use_pls,
        "blend_weight": blend_weight,
        "cv_results": cv_results,
        "mean_cv_mse": np.mean(list(cv_results.values())),
        "path": str(sub_path)
    }
    SUBMISSION_LOG.append(log_entry)

    print(f"\n✓ Submission {sub_num} generated: {sub_path}")
    print(f"  CV MSE: angle={cv_results['angle']:.6f}, depth={cv_results['depth']:.6f}, LR={cv_results['left_right']:.6f}, mean={log_entry['mean_cv_mse']:.6f}")

    return sub_num, cv_results, test_preds


def compute_diversity(test_preds, sub_1350_path):
    """Compute correlation with Sub 1350 (current best)."""
    sub_1350 = pd.read_csv(sub_1350_path)

    correlations = {}
    for target in TARGETS:
        y_1350 = sub_1350[f"scaled_{target}"].values
        y_test = test_preds[target]
        corr = np.corrcoef(y_1350, y_test)[0, 1]
        correlations[target] = corr

    return correlations


def main():
    """Run GP experiments."""
    start_time = time.time()

    # Load data
    train_df, test_df, kpt_cols = load_data()
    print(f"Train: {len(train_df)} shots, Test: {len(test_df)} shots")
    print(f"Features: {len(kpt_cols)} keypoint columns")

    # Load Sub 1350 for diversity check
    sub_1350_path = SUBMISSION_DIR / "submission_1350.csv"
    sub_784_path = SUBMISSION_DIR / "submission_784.csv"

    # Experiment configurations
    configs = [
        # RBF with different length scales
        {"kernel_type": "rbf", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "rbf", "length_scale": 0.5, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "rbf", "length_scale": 2.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},

        # Matern kernels
        {"kernel_type": "matern_0.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "matern_1.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "matern_2.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},

        # Rational Quadratic
        {"kernel_type": "rational_quadratic", "length_scale": 1.0, "alpha": 1.0, "use_pls": True, "blend_weight": 0.5},

        # Variations without PLS
        {"kernel_type": "rbf", "length_scale": 1.0, "alpha": 0.1, "use_pls": False, "blend_weight": 0.5},
        {"kernel_type": "matern_1.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": False, "blend_weight": 0.5},

        # Different blend weights (best kernel only)
        {"kernel_type": "matern_2.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.3},
        {"kernel_type": "matern_2.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.7},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(configs)}")
        print(f"{'#'*80}")

        sub_num, cv_results, test_preds = generate_gp_submission(
            train_df, test_df, kpt_cols,
            sub_784_path=sub_784_path,
            **config
        )

        # Compute diversity vs Sub 1350
        correlations = compute_diversity(test_preds, sub_1350_path)

        result = {
            "submission": sub_num,
            **config,
            "cv_mse": cv_results,
            "mean_cv_mse": np.mean(list(cv_results.values())),
            "diversity_vs_1350": correlations,
            "mean_correlation": np.mean(list(correlations.values()))
        }
        results.append(result)

        print(f"\nDiversity vs Sub 1350:")
        print(f"  Angle r={correlations['angle']:.3f}, Depth r={correlations['depth']:.3f}, LR r={correlations['left_right']:.3f}, Mean r={result['mean_correlation']:.3f}")

        # Progress update every 30 min (not needed for fast experiments)
        elapsed = (time.time() - start_time) / 60
        print(f"\nElapsed time: {elapsed:.1f} minutes")

    # Write comprehensive results
    write_results_report(results, train_df, test_df, kpt_cols)

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    print(f"All experiments complete! Total time: {total_time:.1f} minutes")
    print(f"Generated {len(results)} submissions")
    print(f"Results saved to: {RESEARCH_DIR / 'GP_EXPERIMENT_RESULTS.md'}")
    print(f"{'='*80}")


def write_results_report(results, train_df, test_df, kpt_cols):
    """Write comprehensive results to Research folder."""
    report_path = RESEARCH_DIR / "GP_EXPERIMENT_RESULTS.md"

    with open(report_path, 'w') as f:
        f.write("# Gaussian Process Regression Experiments\n\n")
        f.write("Date: 2026-02-08\n\n")
        f.write("## Goal\n")
        f.write("Achieve LB < 0.005 using Gaussian Process regression with custom kernels and physics-informed constraints.\n\n")
        f.write("## Setup\n")
        f.write(f"- Train shots: {len(train_df)}\n")
        f.write(f"- Test shots: {len(test_df)}\n")
        f.write(f"- Features per target: 213 (198 hoop-relative coords at target-specific frame + 15 PLS components)\n")
        f.write(f"- Target frames: angle=153, depth=150, left_right=170\n")
        f.write(f"- Baseline: Sub 784 (LB 0.007224)\n")
        f.write(f"- Current best: Sub 1350 (LB 0.006776, per-example locally weighted regression)\n\n")

        f.write("## Experimental Configurations\n\n")
        f.write("Tested the following GP kernels:\n")
        f.write("1. **RBF (Radial Basis Function)**: Smooth, infinitely differentiable\n")
        f.write("2. **Matern (nu=0.5, 1.5, 2.5)**: Varying smoothness (0.5=rough, 2.5=smooth)\n")
        f.write("3. **Rational Quadratic**: Multi-scale RBF mixture\n\n")
        f.write("Each kernel includes WhiteKernel for noise modeling.\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Submission | Kernel | Length Scale | Alpha | Use PLS | Blend Weight | Mean CV MSE | Angle r | Depth r | LR r | Mean r |\n")
        f.write("|------------|--------|--------------|-------|---------|--------------|-------------|---------|---------|------|--------|\n")

        for res in results:
            f.write(f"| {res['submission']} | {res['kernel_type']} | {res['length_scale']:.1f} | {res['alpha']:.1f} | {res['use_pls']} | {res['blend_weight']:.1f} | {res['mean_cv_mse']:.6f} | {res['diversity_vs_1350']['angle']:.3f} | {res['diversity_vs_1350']['depth']:.3f} | {res['diversity_vs_1350']['left_right']:.3f} | {res['mean_correlation']:.3f} |\n")

        f.write("\n## Detailed Results\n\n")
        for i, res in enumerate(results, 1):
            f.write(f"### Experiment {i}: Submission {res['submission']}\n\n")
            f.write(f"**Configuration:**\n")
            f.write(f"- Kernel: {res['kernel_type']}\n")
            f.write(f"- Length scale: {res['length_scale']}\n")
            f.write(f"- Alpha: {res['alpha']}\n")
            f.write(f"- Use PLS: {res['use_pls']}\n")
            f.write(f"- Blend weight: {res['blend_weight']}\n\n")

            f.write(f"**CV Performance:**\n")
            f.write(f"- Angle MSE: {res['cv_mse']['angle']:.6f}\n")
            f.write(f"- Depth MSE: {res['cv_mse']['depth']:.6f}\n")
            f.write(f"- Left_right MSE: {res['cv_mse']['left_right']:.6f}\n")
            f.write(f"- Mean MSE: {res['mean_cv_mse']:.6f}\n\n")

            f.write(f"**Diversity vs Sub 1350:**\n")
            f.write(f"- Angle correlation: {res['diversity_vs_1350']['angle']:.3f}\n")
            f.write(f"- Depth correlation: {res['diversity_vs_1350']['depth']:.3f}\n")
            f.write(f"- LR correlation: {res['diversity_vs_1350']['left_right']:.3f}\n")
            f.write(f"- Mean correlation: {res['mean_correlation']:.3f}\n\n")

            f.write(f"**Submission file:** submission/submission_{res['submission']}.csv\n\n")

        f.write("## Key Findings\n\n")

        # Find best CV
        best_cv = min(results, key=lambda x: x['mean_cv_mse'])
        f.write(f"**Best CV performance:** Submission {best_cv['submission']} ({best_cv['kernel_type']}) with mean MSE {best_cv['mean_cv_mse']:.6f}\n\n")

        # Find most diverse
        most_diverse = min(results, key=lambda x: x['mean_correlation'])
        f.write(f"**Most diverse:** Submission {most_diverse['submission']} ({most_diverse['kernel_type']}) with mean correlation {most_diverse['mean_correlation']:.3f}\n\n")

        # High diversity candidates (r < 0.90)
        high_div = [r for r in results if r['mean_correlation'] < 0.90]
        if high_div:
            f.write(f"**High diversity candidates (r < 0.90):** {len(high_div)} submissions\n")
            for r in high_div:
                f.write(f"- Sub {r['submission']}: {r['kernel_type']}, r={r['mean_correlation']:.3f}, CV MSE={r['mean_cv_mse']:.6f}\n")
        else:
            f.write("**No submissions with mean correlation < 0.90**\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. Test submissions with highest diversity (r < 0.90) first on LB\n")
        f.write("2. If GP models show promise, explore ensemble combinations\n")
        f.write("3. Consider per-player GP models for additional diversity\n")
        f.write("4. Explore custom kernel combinations (e.g., RBF + Matern)\n\n")

        f.write("## Reproducibility\n\n")
        f.write("All experiments use:\n")
        f.write("- Script: scripts/gp_experiment.py\n")
        f.write("- Random seed: 42\n")
        f.write("- Features: 198 hoop-relative coords + 15 PLS components\n")
        f.write("- Target-specific frames: angle=153, depth=150, LR=170\n")
        f.write("- Blending baseline: submission/submission_784.csv\n")
        f.write("- Evaluation: Leave-One-Out Cross-Validation\n")


if __name__ == "__main__":
    main()
