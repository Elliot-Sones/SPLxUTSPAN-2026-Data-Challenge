"""
Fast Gaussian Process Regression Experiment

Optimized version using 5-fold CV instead of LOO for speed.
Goal: Generate 10-15 diverse GP submissions quickly for LB testing.
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel,
    ConstantKernel as C
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
RESEARCH_DIR = PROJECT_DIR / "Research"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


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

    # Scale targets
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
    """Build GP kernel."""
    if kernel_type == "rbf":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2)) + WhiteKernel(noise_level=alpha)
    elif kernel_type.startswith("matern"):
        nu = float(kernel_type.split('_')[1])
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale, length_scale_bounds=(1e-2, 1e2), nu=nu) + WhiteKernel(noise_level=alpha)
    elif kernel_type == "rational_quadratic":
        kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=length_scale, alpha=alpha) + WhiteKernel(noise_level=0.1)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return kernel


def generate_gp_submission(train_df, test_df, kpt_cols, kernel_type, length_scale=1.0, alpha=1.0,
                           use_pls=True, blend_weight=0.5):
    """Generate submission using GP model with 5-fold CV."""
    print(f"\nGenerating GP submission: kernel={kernel_type}, ls={length_scale}, alpha={alpha}, pls={use_pls}, blend={blend_weight}")

    # Load Sub 784 for blending
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    test_preds = {}
    cv_results = {}

    for target in TARGETS:
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

        # Normalize
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        # 5-fold CV for speed
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_cv_pred = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y[train_idx]

            kernel = build_gp_kernel(kernel_type, length_scale, alpha)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True, alpha=1e-6, random_state=42)
            gp.fit(X_tr, y_tr)
            y_cv_pred[val_idx] = gp.predict(X_val)

        cv_mse = mean_squared_error(y, y_cv_pred)
        cv_results[target] = cv_mse

        # Train on full data
        kernel = build_gp_kernel(kernel_type, length_scale, alpha)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True, alpha=1e-6, random_state=42)
        gp.fit(X_train, y)

        # Predict
        y_test_pred = gp.predict(X_test)
        y_test_pred = np.clip(y_test_pred, 0, 1)

        # Blend
        y_test_784 = sub_784[f"scaled_{target}"].values
        y_test_blended = blend_weight * y_test_pred + (1 - blend_weight) * y_test_784

        test_preds[target] = y_test_blended

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

    print(f"✓ Sub {sub_num}: CV MSE = {np.mean(list(cv_results.values())):.6f}")

    return sub_num, cv_results, test_preds


def compute_diversity(test_preds, sub_1350_path):
    """Compute correlation with Sub 1350."""
    sub_1350 = pd.read_csv(sub_1350_path)

    correlations = {}
    for target in TARGETS:
        y_1350 = sub_1350[f"scaled_{target}"].values
        y_test = test_preds[target]
        corr = np.corrcoef(y_1350, y_test)[0, 1]
        correlations[target] = corr

    return correlations


def main():
    """Run fast GP experiments."""
    start_time = time.time()

    train_df, test_df, kpt_cols = load_data()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Features: {len(kpt_cols)}")

    sub_1350_path = SUBMISSION_DIR / "submission_1350.csv"

    # Compact set of diverse configurations
    configs = [
        # Core kernels with optimal settings
        {"kernel_type": "rbf", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "matern_1.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "matern_2.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "rational_quadratic", "length_scale": 1.0, "alpha": 1.0, "use_pls": True, "blend_weight": 0.5},

        # Length scale variations (best kernel)
        {"kernel_type": "matern_2.5", "length_scale": 0.5, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},
        {"kernel_type": "matern_2.5", "length_scale": 2.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.5},

        # Without PLS
        {"kernel_type": "rbf", "length_scale": 1.0, "alpha": 0.1, "use_pls": False, "blend_weight": 0.5},
        {"kernel_type": "matern_2.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": False, "blend_weight": 0.5},

        # Blend weight variations
        {"kernel_type": "matern_2.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.3},
        {"kernel_type": "matern_2.5", "length_scale": 1.0, "alpha": 0.1, "use_pls": True, "blend_weight": 0.7},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(configs)}")
        print(f"{'='*60}")

        sub_num, cv_results, test_preds = generate_gp_submission(train_df, test_df, kpt_cols, **config)

        # Diversity
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

        print(f"Diversity vs Sub 1350: angle={correlations['angle']:.3f}, depth={correlations['depth']:.3f}, LR={correlations['left_right']:.3f}")

    # Write results
    report_path = RESEARCH_DIR / "GP_FAST_EXPERIMENT_RESULTS.md"

    with open(report_path, 'w') as f:
        f.write("# Fast Gaussian Process Regression Experiments\n\n")
        f.write("Date: 2026-02-08\n\n")
        f.write("## Setup\n")
        f.write(f"- Train: {len(train_df)} shots, Test: {len(test_df)} shots\n")
        f.write(f"- Features: 213 per target (198 HC + 15 PLS)\n")
        f.write(f"- Validation: 5-fold CV (faster than LOO)\n")
        f.write(f"- Baseline: Sub 784 (LB 0.007224)\n")
        f.write(f"- Current best: Sub 1350 (LB 0.006776)\n\n")

        f.write("## Results\n\n")
        f.write("| Sub | Kernel | LS | Alpha | PLS | Blend | Mean CV MSE | Angle r | Depth r | LR r | Mean r |\n")
        f.write("|-----|--------|----| ------|-----|-------|-------------|---------|---------|------|--------|\n")

        for res in results:
            f.write(f"| {res['submission']} | {res['kernel_type']} | {res['length_scale']:.1f} | {res['alpha']:.1f} | {res['use_pls']} | {res['blend_weight']:.1f} | {res['mean_cv_mse']:.6f} | {res['diversity_vs_1350']['angle']:.3f} | {res['diversity_vs_1350']['depth']:.3f} | {res['diversity_vs_1350']['left_right']:.3f} | {res['mean_correlation']:.3f} |\n")

        f.write("\n## Key Findings\n\n")

        best_cv = min(results, key=lambda x: x['mean_cv_mse'])
        f.write(f"**Best CV:** Sub {best_cv['submission']} ({best_cv['kernel_type']}) - MSE {best_cv['mean_cv_mse']:.6f}\n\n")

        most_diverse = min(results, key=lambda x: x['mean_correlation'])
        f.write(f"**Most diverse:** Sub {most_diverse['submission']} ({most_diverse['kernel_type']}) - r={most_diverse['mean_correlation']:.3f}\n\n")

        high_div = [r for r in results if r['mean_correlation'] < 0.90]
        f.write(f"**High diversity (r<0.90):** {len(high_div)} submissions\n")
        for r in high_div:
            f.write(f"- Sub {r['submission']}: r={r['mean_correlation']:.3f}, CV MSE={r['mean_cv_mse']:.6f}\n")

        f.write("\n## Recommendations\n\n")
        f.write("Test high diversity submissions on LB. GP models provide different inductive biases vs tree ensembles.\n")

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"Complete! {len(results)} submissions in {total_time:.1f} min")
    print(f"Results: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
