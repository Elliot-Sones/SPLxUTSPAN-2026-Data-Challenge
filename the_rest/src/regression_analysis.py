"""
Regression Coefficient Analysis - Partial Derivatives

Computes actual regression coefficients (dY/dX) to answer:
"If feature X changes by 1 unit, how much does target Y change?"

This extends the correlation analysis by providing interpretable sensitivity metrics.

Relationship between correlation and regression:
    beta = r * (std_Y / std_X)
    beta_standardized = beta * (std_X / std_Y) = r  (verification)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_all_as_arrays, get_keypoint_columns, TARGET_COLS

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def compute_simple_regression(
    X: np.ndarray,
    y: np.ndarray,
    keypoint_cols: List[str]
) -> pd.DataFrame:
    """
    Compute simple linear regression coefficients for all features vs all targets.

    Args:
        X: (n_shots, 240, 207) raw time series
        y: (n_shots, 3) targets [angle, depth, left_right]
        keypoint_cols: list of 207 column names

    Returns:
        DataFrame with beta, se, p-value, R2 for each feature-target pair
    """
    target_names = TARGET_COLS
    results = []
    n_features = len(keypoint_cols)

    print(f"Computing simple regressions for {n_features} features x 2 aggregations x 3 targets...")

    for feat_idx in range(n_features):
        # Compute mean and std across frames for this feature
        feat_mean = np.nanmean(X[:, :, feat_idx], axis=1)
        feat_std = np.nanstd(X[:, :, feat_idx], axis=1)

        for feat_values, feat_type in [(feat_mean, 'raw_mean'), (feat_std, 'raw_std')]:
            for target_idx, target_name in enumerate(target_names):
                target = y[:, target_idx]

                # Remove NaN
                valid = ~(np.isnan(feat_values) | np.isnan(target))
                X_valid = feat_values[valid]
                Y_valid = target[valid]

                if len(X_valid) < 10:
                    continue

                # Fit regression using scipy.stats.linregress
                # Returns: slope, intercept, r, p, stderr
                slope, intercept, r, p, se = stats.linregress(X_valid, Y_valid)

                # R-squared
                r_squared = r ** 2

                # Standard deviations
                std_X = np.std(X_valid, ddof=1)
                std_Y = np.std(Y_valid, ddof=1)

                # Standardized beta (should equal r for verification)
                beta_standardized = slope * (std_X / std_Y) if std_Y > 0 else 0

                # 95% CI
                ci_lower = slope - 1.96 * se
                ci_upper = slope + 1.96 * se

                # T-statistic (for completeness - scipy already computes p)
                t_stat = slope / se if se > 0 else np.nan

                results.append({
                    'feature_idx': feat_idx,
                    'feature_name': keypoint_cols[feat_idx],
                    'feature_type': feat_type,
                    'target': target_name,
                    'beta': slope,
                    'intercept': intercept,
                    'se_beta': se,
                    't_stat': t_stat,
                    'p_value': p,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'r_squared': r_squared,
                    'pearson_r': r,
                    'std_X': std_X,
                    'std_Y': std_Y,
                    'beta_standardized': beta_standardized,
                    'n_samples': len(X_valid)
                })

        if (feat_idx + 1) % 50 == 0:
            print(f"  Processed {feat_idx + 1}/{n_features} features")

    return pd.DataFrame(results)


def compute_per_player_regression(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    keypoint_cols: List[str]
) -> pd.DataFrame:
    """
    Compute regression coefficients separately for each player.

    This isolates within-player variation to answer:
    "Does knee height predict angle FOR THIS PLAYER?"

    Args:
        X: (n_shots, 240, 207) raw time series
        y: (n_shots, 3) targets
        participant_ids: (n_shots,) player IDs
        keypoint_cols: list of 207 column names

    Returns:
        DataFrame with regression results per player-feature-target combination
    """
    target_names = TARGET_COLS
    results = []
    unique_players = np.unique(participant_ids)
    n_features = len(keypoint_cols)

    print(f"Computing per-player regressions for {len(unique_players)} players...")

    for player_id in unique_players:
        mask = participant_ids == player_id
        X_player = X[mask]
        y_player = y[mask]
        n_shots = mask.sum()

        print(f"  Player {player_id}: {n_shots} shots")

        for feat_idx in range(n_features):
            # Compute mean and std across frames
            feat_mean = np.nanmean(X_player[:, :, feat_idx], axis=1)
            feat_std = np.nanstd(X_player[:, :, feat_idx], axis=1)

            for feat_values, feat_type in [(feat_mean, 'raw_mean'), (feat_std, 'raw_std')]:
                for target_idx, target_name in enumerate(target_names):
                    target = y_player[:, target_idx]

                    # Remove NaN
                    valid = ~(np.isnan(feat_values) | np.isnan(target))
                    X_valid = feat_values[valid]
                    Y_valid = target[valid]

                    # Need enough samples for regression
                    if len(X_valid) < 5:
                        continue

                    try:
                        slope, intercept, r, p, se = stats.linregress(X_valid, Y_valid)
                    except Exception:
                        continue

                    r_squared = r ** 2
                    std_X = np.std(X_valid, ddof=1) if len(X_valid) > 1 else 0
                    std_Y = np.std(Y_valid, ddof=1) if len(Y_valid) > 1 else 0
                    beta_standardized = slope * (std_X / std_Y) if std_Y > 0 else 0

                    ci_lower = slope - 1.96 * se
                    ci_upper = slope + 1.96 * se

                    results.append({
                        'player': int(player_id),
                        'feature_idx': feat_idx,
                        'feature_name': keypoint_cols[feat_idx],
                        'feature_type': feat_type,
                        'target': target_name,
                        'beta': slope,
                        'se_beta': se,
                        'p_value': p,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'r_squared': r_squared,
                        'pearson_r': r,
                        'std_X': std_X,
                        'std_Y': std_Y,
                        'beta_standardized': beta_standardized,
                        'n_samples': len(X_valid)
                    })

    return pd.DataFrame(results)


def compute_multivariate_regression(
    X: np.ndarray,
    y: np.ndarray,
    keypoint_cols: List[str],
    simple_results: pd.DataFrame,
    top_n: int = 5,
    alpha: float = 100.0,
    p_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Fit multivariate regression with top N significant features for each target.

    Uses Ridge regression with regularization to handle collinearity.
    Only uses features with p-value < p_threshold from simple regression.

    Args:
        X: (n_shots, 240, 207) raw time series
        y: (n_shots, 3) targets
        keypoint_cols: list of 207 column names
        simple_results: DataFrame from compute_simple_regression
        top_n: number of top features to include
        alpha: Ridge regularization strength

    Returns:
        DataFrame with multivariate model results
    """
    target_names = TARGET_COLS
    results = []

    print(f"Computing multivariate regression with top {top_n} features per target...")

    # Precompute all feature aggregations
    n_shots = X.shape[0]
    n_features = len(keypoint_cols)

    # Build feature matrix: 207 means + 207 stds = 414 features
    all_features = np.zeros((n_shots, n_features * 2))
    feature_names = []

    for feat_idx in range(n_features):
        all_features[:, feat_idx] = np.nanmean(X[:, :, feat_idx], axis=1)
        all_features[:, n_features + feat_idx] = np.nanstd(X[:, :, feat_idx], axis=1)
        feature_names.append(f"{keypoint_cols[feat_idx]}_mean")
        feature_names.append(f"{keypoint_cols[feat_idx]}_std")

    # Reorder feature_names to match all_features layout
    feature_names_ordered = [f"{keypoint_cols[i]}_mean" for i in range(n_features)] + \
                           [f"{keypoint_cols[i]}_std" for i in range(n_features)]

    for target_idx, target_name in enumerate(target_names):
        print(f"  Target: {target_name}")
        target = y[:, target_idx]

        # Get top N significant features by absolute beta from simple regression
        target_results = simple_results[simple_results['target'] == target_name].copy()
        # Filter to significant features only
        significant = target_results[target_results['p_value'] < p_threshold]
        if len(significant) == 0:
            print(f"    No significant features at p<{p_threshold}")
            continue
        significant['abs_beta'] = significant['beta'].abs()
        top_features = significant.nlargest(min(top_n, len(significant)), 'abs_beta')
        print(f"    Using {len(top_features)} significant features (p<{p_threshold})")

        # Build feature indices for selection
        selected_indices = []
        selected_names = []

        for _, row in top_features.iterrows():
            feat_idx = row['feature_idx']
            feat_type = row['feature_type']

            if feat_type == 'raw_mean':
                idx = feat_idx
            else:  # raw_std
                idx = n_features + feat_idx

            selected_indices.append(idx)
            selected_names.append(f"{row['feature_name']}_{feat_type.replace('raw_', '')}")

        # Extract selected features
        X_selected = all_features[:, selected_indices]

        # Remove rows with NaN
        valid = ~(np.isnan(X_selected).any(axis=1) | np.isnan(target))
        X_valid = X_selected[valid]
        y_valid = target[valid]

        if len(y_valid) < top_n + 5:
            print(f"    Skipping: insufficient samples ({len(y_valid)})")
            continue

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)

        # Fit Ridge regression
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y_valid)

        # R-squared
        r_squared = model.score(X_scaled, y_valid)

        # Cross-validated R-squared
        cv_scores = cross_val_score(model, X_scaled, y_valid, cv=5, scoring='r2')

        # Store coefficients
        for i, (name, coef) in enumerate(zip(selected_names, model.coef_)):
            results.append({
                'target': target_name,
                'feature_rank': i + 1,
                'feature_name': name,
                'coefficient_standardized': coef,
                'model_r_squared': r_squared,
                'cv_r_squared_mean': cv_scores.mean(),
                'cv_r_squared_std': cv_scores.std(),
                'n_features': top_n,
                'n_samples': len(y_valid),
                'alpha': alpha
            })

        print(f"    R2: {r_squared:.4f}, CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return pd.DataFrame(results)


def generate_summary_tables(
    simple_df: pd.DataFrame,
    player_df: pd.DataFrame,
    multi_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Generate summary tables for easy interpretation.

    Returns:
        Dictionary of summary DataFrames
    """
    summaries = {}

    # 1. Top 10 features by |beta| for each target (pooled)
    print("\n=== Top 10 Features by |beta| (Pooled) ===")
    for target in TARGET_COLS:
        target_data = simple_df[simple_df['target'] == target].copy()
        target_data['abs_beta'] = target_data['beta'].abs()
        top_10 = target_data.nlargest(10, 'abs_beta')[
            ['feature_name', 'feature_type', 'beta', 'se_beta', 'p_value', 'r_squared', 'ci_lower', 'ci_upper']
        ]
        summaries[f'top10_{target}'] = top_10

        print(f"\n{target.upper()}:")
        print(top_10.to_string(index=False))

    # 2. Top 5 features by |beta| for each target x player
    print("\n=== Top 5 Features by |beta| (Per Player) ===")
    for target in TARGET_COLS:
        for player in sorted(player_df['player'].unique()):
            mask = (player_df['target'] == target) & (player_df['player'] == player)
            player_target = player_df[mask].copy()
            player_target['abs_beta'] = player_target['beta'].abs()
            top_5 = player_target.nlargest(5, 'abs_beta')[
                ['feature_name', 'feature_type', 'beta', 'p_value', 'r_squared']
            ]
            summaries[f'top5_{target}_player{player}'] = top_5

            print(f"\n{target.upper()} - Player {player}:")
            print(top_5.to_string(index=False))

    # 3. Multivariate model performance
    print("\n=== Multivariate Model Performance ===")
    if len(multi_df) > 0:
        perf = multi_df.groupby('target').agg({
            'model_r_squared': 'first',
            'cv_r_squared_mean': 'first',
            'cv_r_squared_std': 'first',
            'n_features': 'first',
            'n_samples': 'first'
        }).reset_index()
        summaries['multivariate_performance'] = perf
        print(perf.to_string(index=False))

    return summaries


def generate_interpretation(simple_df: pd.DataFrame) -> str:
    """
    Generate plain English interpretation of top findings.
    """
    interpretation = []
    interpretation.append("=" * 60)
    interpretation.append("REGRESSION COEFFICIENT INTERPRETATION")
    interpretation.append("=" * 60)
    interpretation.append("")
    interpretation.append("How to read: 'For every 1 unit increase in X, Y changes by beta units'")
    interpretation.append("95% CI shows the range of plausible values for the true effect.")
    interpretation.append("")

    for target in TARGET_COLS:
        target_data = simple_df[simple_df['target'] == target].copy()
        target_data['abs_beta'] = target_data['beta'].abs()

        # Get top 5 significant features
        significant = target_data[target_data['p_value'] < 0.05]
        top_5 = significant.nlargest(5, 'abs_beta')

        interpretation.append(f"\n--- {target.upper()} ---")

        if target == 'angle':
            unit = 'degrees'
        elif target == 'depth':
            unit = 'cm (depth units)'
        else:
            unit = 'units (left-right)'

        for _, row in top_5.iterrows():
            name = row['feature_name']
            feat_type = 'mean' if row['feature_type'] == 'raw_mean' else 'variability'
            beta = row['beta']
            ci_l = row['ci_lower']
            ci_u = row['ci_upper']
            r2 = row['r_squared']

            direction = "increase" if beta > 0 else "decrease"

            interpretation.append(
                f"  {name} ({feat_type}): "
                f"1 unit change -> {beta:.4f} {unit} {direction} "
                f"(95% CI: [{ci_l:.4f}, {ci_u:.4f}], R2={r2:.3f})"
            )

    interpretation.append("")
    interpretation.append("=" * 60)

    return "\n".join(interpretation)


def verify_results(simple_df: pd.DataFrame, corr_df: Optional[pd.DataFrame] = None) -> None:
    """
    Verify regression results against expected relationships.
    """
    print("\n=== VERIFICATION ===")

    # 1. Check beta_standardized ~= pearson_r
    diff = (simple_df['beta_standardized'] - simple_df['pearson_r']).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"1. Standardized beta vs Pearson r:")
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    if max_diff < 0.001:
        print("   PASS: Standardized beta matches Pearson r")
    else:
        print("   WARNING: Some discrepancy (may be due to numerical precision)")

    # 2. Check r_squared = pearson_r^2
    expected_r2 = simple_df['pearson_r'] ** 2
    r2_diff = (simple_df['r_squared'] - expected_r2).abs()
    max_r2_diff = r2_diff.max()
    print(f"\n2. R-squared vs r^2:")
    print(f"   Max difference: {max_r2_diff:.6f}")
    if max_r2_diff < 0.001:
        print("   PASS: R-squared equals r^2")
    else:
        print("   WARNING: Some discrepancy")

    # 3. Cross-check with existing correlations if available
    if corr_df is not None:
        # Merge on feature_idx, target, feature_type
        merged = simple_df.merge(
            corr_df[['feature_idx', 'target', 'feature_type', 'pearson_r']],
            on=['feature_idx', 'target', 'feature_type'],
            suffixes=('_reg', '_corr')
        )
        if len(merged) > 0:
            corr_diff = (merged['pearson_r_reg'] - merged['pearson_r_corr']).abs()
            max_corr_diff = corr_diff.max()
            print(f"\n3. Cross-check with existing correlation analysis:")
            print(f"   Max difference in Pearson r: {max_corr_diff:.6f}")
            if max_corr_diff < 0.001:
                print("   PASS: Results consistent with prior analysis")
            else:
                print("   WARNING: Some discrepancy with prior correlation analysis")


def main():
    """Main function to run all regression analyses."""
    print("=" * 60)
    print("REGRESSION COEFFICIENT ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    participant_ids = meta['participant_id'].values

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {len(keypoint_cols)}")
    print(f"Participants: {np.unique(participant_ids)}")

    # 1. Simple regression (pooled)
    print("\n" + "=" * 60)
    print("1. SIMPLE LINEAR REGRESSION (Pooled)")
    print("=" * 60)
    simple_df = compute_simple_regression(X, y, keypoint_cols)
    simple_df.to_csv(OUTPUT_DIR / "regression_coefficients.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'regression_coefficients.csv'}")
    print(f"Total feature-target pairs: {len(simple_df)}")

    # 2. Per-player regression
    print("\n" + "=" * 60)
    print("2. PER-PLAYER REGRESSION")
    print("=" * 60)
    player_df = compute_per_player_regression(X, y, participant_ids, keypoint_cols)
    player_df.to_csv(OUTPUT_DIR / "regression_per_player.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'regression_per_player.csv'}")
    print(f"Total player-feature-target combinations: {len(player_df)}")

    # 3. Multivariate regression
    print("\n" + "=" * 60)
    print("3. MULTIVARIATE REGRESSION")
    print("=" * 60)
    multi_df = compute_multivariate_regression(X, y, keypoint_cols, simple_df, top_n=5, alpha=100.0)
    multi_df.to_csv(OUTPUT_DIR / "multivariate_regression.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'multivariate_regression.csv'}")

    # 4. Generate summary tables
    print("\n" + "=" * 60)
    print("4. SUMMARY TABLES")
    print("=" * 60)
    summaries = generate_summary_tables(simple_df, player_df, multi_df)

    # Save summaries to CSV files (one per table)
    summary_dir = OUTPUT_DIR / "regression_summaries"
    summary_dir.mkdir(exist_ok=True)
    for name, df in summaries.items():
        df.to_csv(summary_dir / f"{name}.csv", index=False)
    print(f"Saved summary tables to: {summary_dir}/")

    # 5. Generate interpretation
    print("\n" + "=" * 60)
    print("5. INTERPRETATION")
    print("=" * 60)
    interpretation = generate_interpretation(simple_df)
    print(interpretation)

    # Save interpretation
    with open(OUTPUT_DIR / "regression_interpretation.txt", "w") as f:
        f.write(interpretation)
    print(f"Saved: {OUTPUT_DIR / 'regression_interpretation.txt'}")

    # 6. Verification
    print("\n" + "=" * 60)
    print("6. VERIFICATION")
    print("=" * 60)

    # Load existing correlations for cross-check
    corr_path = OUTPUT_DIR / "feature_correlations.csv"
    corr_df = None
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        print(f"Loaded existing correlations from {corr_path}")

    verify_results(simple_df, corr_df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return simple_df, player_df, multi_df


if __name__ == "__main__":
    main()
