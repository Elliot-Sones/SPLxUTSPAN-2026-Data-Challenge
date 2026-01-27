"""
Frame-by-Frame Regression Analysis

Computes regression coefficients per frame to answer:
- "Which frames during the shot are most predictive of the outcome?"
- "When is the release frame?" (should show peak R2)
- "Does post-release pose matter?" (should show low R2)

Four granularities:
1. Pooled per-frame: All shots together, for each of 240 frames
2. Per-player per-frame: Separate regression for each player
3. Binned frames: 10-frame windows (24 bins)
4. Key frames only: 5 representative frames [0, 60, 120, 180, 239]
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import List, Optional, Tuple
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_loader import load_all_as_arrays, get_keypoint_columns, TARGET_COLS, NUM_FRAMES

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def compute_single_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, int]:
    """
    Compute simple linear regression for a single feature vs target.

    Returns:
        (beta, se_beta, p_value, r_squared, n_samples)
    """
    # Remove NaN values
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid]
    y_valid = y[valid]

    n_samples = len(x_valid)
    if n_samples < 5:
        return np.nan, np.nan, np.nan, np.nan, n_samples

    # Check for zero variance
    if np.std(x_valid) < 1e-10 or np.std(y_valid) < 1e-10:
        return np.nan, np.nan, np.nan, np.nan, n_samples

    try:
        slope, intercept, r, p, se = stats.linregress(x_valid, y_valid)
        r_squared = r ** 2
        return slope, se, p, r_squared, n_samples
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, n_samples


def compute_pooled_frame_regression(
    X: np.ndarray,
    y: np.ndarray,
    keypoint_cols: List[str],
    frame_indices: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Compute regression for each frame x feature x target combination (pooled across all shots).

    Args:
        X: (n_shots, 240, 207) raw time series
        y: (n_shots, 3) targets [angle, depth, left_right]
        keypoint_cols: list of 207 column names
        frame_indices: optional list of specific frame indices to analyze (None = all 240)

    Returns:
        DataFrame with frame, feature_idx, feature_name, target, beta, se_beta, p_value, r_squared, n_samples
    """
    if frame_indices is None:
        frame_indices = list(range(NUM_FRAMES))

    n_frames = len(frame_indices)
    n_features = len(keypoint_cols)
    n_targets = len(TARGET_COLS)

    results = []
    total = n_frames * n_features * n_targets
    processed = 0

    for frame_pos, frame_idx in enumerate(frame_indices):
        for feat_idx in range(n_features):
            # Extract feature values at this frame for all shots
            feat_values = X[:, frame_idx, feat_idx]

            for target_idx, target_name in enumerate(TARGET_COLS):
                target_values = y[:, target_idx]

                beta, se, p, r2, n = compute_single_regression(feat_values, target_values)

                results.append({
                    'frame': frame_idx,
                    'feature_idx': feat_idx,
                    'feature_name': keypoint_cols[feat_idx],
                    'target': target_name,
                    'beta': beta,
                    'se_beta': se,
                    'p_value': p,
                    'r_squared': r2,
                    'n_samples': n
                })

                processed += 1

        if (frame_pos + 1) % 20 == 0 or frame_pos == 0:
            print(f"  Frame {frame_idx}: {processed}/{total} regressions ({100*processed/total:.1f}%)")

    return pd.DataFrame(results)


def compute_perplayer_frame_regression(
    X: np.ndarray,
    y: np.ndarray,
    keypoint_cols: List[str],
    participant_ids: np.ndarray,
    frame_indices: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Compute regression for each frame x feature x target x player combination.

    Args:
        X: (n_shots, n_frames, 207) raw time series (n_frames may be 240 or len(frame_indices))
        y: (n_shots, 3) targets
        keypoint_cols: list of 207 column names
        participant_ids: (n_shots,) player IDs
        frame_indices: optional list of frame indices for labeling (None = use 0-based indexing)

    Returns:
        DataFrame with player, frame, feature_idx, feature_name, target, beta, se_beta, p_value, r_squared, n_samples
    """
    unique_players = np.unique(participant_ids)
    n_frames_actual = X.shape[1]

    if frame_indices is None:
        frame_indices = list(range(n_frames_actual))

    n_features = len(keypoint_cols)
    n_targets = len(TARGET_COLS)

    results = []
    total = len(unique_players) * len(frame_indices) * n_features * n_targets
    processed = 0

    for player_id in unique_players:
        mask = participant_ids == player_id
        X_player = X[mask]
        y_player = y[mask]
        n_shots_player = mask.sum()

        print(f"  Player {player_id}: {n_shots_player} shots")

        for frame_pos in range(n_frames_actual):
            frame_label = frame_indices[frame_pos]

            for feat_idx in range(n_features):
                feat_values = X_player[:, frame_pos, feat_idx]

                for target_idx, target_name in enumerate(TARGET_COLS):
                    target_values = y_player[:, target_idx]

                    beta, se, p, r2, n = compute_single_regression(feat_values, target_values)

                    results.append({
                        'player': int(player_id),
                        'frame': frame_label,
                        'feature_idx': feat_idx,
                        'feature_name': keypoint_cols[feat_idx],
                        'target': target_name,
                        'beta': beta,
                        'se_beta': se,
                        'p_value': p,
                        'r_squared': r2,
                        'n_samples': n
                    })

                    processed += 1

        print(f"    Completed: {processed}/{total} ({100*processed/total:.1f}%)")

    return pd.DataFrame(results)


def compute_binned_frame_regression(
    X: np.ndarray,
    y: np.ndarray,
    keypoint_cols: List[str],
    participant_ids: np.ndarray,
    bin_size: int = 10
) -> pd.DataFrame:
    """
    Compute regression with features averaged within frame bins.

    Args:
        X: (n_shots, 240, 207) raw time series
        y: (n_shots, 3) targets
        keypoint_cols: list of 207 column names
        participant_ids: (n_shots,) player IDs
        bin_size: number of frames per bin (default 10)

    Returns:
        DataFrame with player, bin_idx, bin_start, bin_end, feature_idx, feature_name, target, etc.
    """
    n_shots = X.shape[0]
    n_features = X.shape[2]
    n_bins = NUM_FRAMES // bin_size  # 240 / 10 = 24

    # Compute binned features: average within each bin
    print(f"  Creating {n_bins} bins of {bin_size} frames each...")
    X_binned = np.zeros((n_shots, n_bins, n_features), dtype=np.float32)

    for bin_idx in range(n_bins):
        start = bin_idx * bin_size
        end = start + bin_size
        X_binned[:, bin_idx, :] = np.nanmean(X[:, start:end, :], axis=1)

    # Compute per-player regression on binned data
    unique_players = np.unique(participant_ids)
    n_targets = len(TARGET_COLS)

    results = []
    total = len(unique_players) * n_bins * n_features * n_targets
    processed = 0

    for player_id in unique_players:
        mask = participant_ids == player_id
        X_player = X_binned[mask]
        y_player = y[mask]
        n_shots_player = mask.sum()

        print(f"  Player {player_id}: {n_shots_player} shots")

        for bin_idx in range(n_bins):
            bin_start = bin_idx * bin_size
            bin_end = bin_start + bin_size - 1

            for feat_idx in range(n_features):
                feat_values = X_player[:, bin_idx, feat_idx]

                for target_idx, target_name in enumerate(TARGET_COLS):
                    target_values = y_player[:, target_idx]

                    beta, se, p, r2, n = compute_single_regression(feat_values, target_values)

                    results.append({
                        'player': int(player_id),
                        'bin_idx': bin_idx,
                        'bin_start': bin_start,
                        'bin_end': bin_end,
                        'feature_idx': feat_idx,
                        'feature_name': keypoint_cols[feat_idx],
                        'target': target_name,
                        'beta': beta,
                        'se_beta': se,
                        'p_value': p,
                        'r_squared': r2,
                        'n_samples': n
                    })

                    processed += 1

        print(f"    Completed: {processed}/{total} ({100*processed/total:.1f}%)")

    return pd.DataFrame(results)


def generate_frame_r2_summary(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary R2 statistics by frame for each target.

    For each frame, compute:
    - mean_r2: Average R2 across all features
    - max_r2: Maximum R2 (best feature)
    - best_feature: Feature name with highest R2
    """
    summaries = []

    for target in TARGET_COLS:
        target_data = pooled_df[pooled_df['target'] == target]

        for frame in sorted(target_data['frame'].unique()):
            frame_data = target_data[target_data['frame'] == frame]

            # Filter out NaN R2 values
            valid_r2 = frame_data[frame_data['r_squared'].notna()]

            if len(valid_r2) == 0:
                continue

            mean_r2 = valid_r2['r_squared'].mean()
            max_r2 = valid_r2['r_squared'].max()
            best_row = valid_r2.loc[valid_r2['r_squared'].idxmax()]
            best_feature = best_row['feature_name']

            # Also get median and std for robustness
            median_r2 = valid_r2['r_squared'].median()
            std_r2 = valid_r2['r_squared'].std()

            # Count significant features (p < 0.05)
            n_significant = (valid_r2['p_value'] < 0.05).sum()

            summaries.append({
                'frame': frame,
                'target': target,
                'mean_r2': mean_r2,
                'median_r2': median_r2,
                'std_r2': std_r2,
                'max_r2': max_r2,
                'best_feature': best_feature,
                'n_significant_features': n_significant
            })

    return pd.DataFrame(summaries)


def generate_phase_summary(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize top features by phase (4 phases of 60 frames each).
    """
    phases = [
        (0, 59, 'Phase1_Setup'),
        (60, 119, 'Phase2_Windup'),
        (120, 179, 'Phase3_Release'),
        (180, 239, 'Phase4_Follow')
    ]

    summaries = []

    for start, end, phase_name in phases:
        phase_data = pooled_df[(pooled_df['frame'] >= start) & (pooled_df['frame'] <= end)]

        for target in TARGET_COLS:
            target_phase = phase_data[phase_data['target'] == target]

            if len(target_phase) == 0:
                continue

            # Average R2 within phase for each feature
            feature_r2 = target_phase.groupby('feature_name')['r_squared'].mean()

            if len(feature_r2) == 0:
                continue

            best_feature = feature_r2.idxmax()
            best_r2 = feature_r2.max()
            mean_r2 = feature_r2.mean()

            summaries.append({
                'phase': phase_name,
                'frame_start': start,
                'frame_end': end,
                'target': target,
                'best_feature': best_feature,
                'best_feature_r2': best_r2,
                'mean_r2_all_features': mean_r2
            })

    return pd.DataFrame(summaries)


def detect_release_frame(r2_summary_df: pd.DataFrame) -> dict:
    """
    Estimate the release frame based on where R2 peaks.

    Returns dict with estimated release frame per target.
    """
    release_estimates = {}

    for target in TARGET_COLS:
        target_data = r2_summary_df[r2_summary_df['target'] == target]

        if len(target_data) == 0:
            continue

        # Find frame with maximum mean R2
        max_idx = target_data['mean_r2'].idxmax()
        release_frame = target_data.loc[max_idx, 'frame']
        max_r2 = target_data.loc[max_idx, 'mean_r2']

        release_estimates[target] = {
            'estimated_release_frame': int(release_frame),
            'peak_mean_r2': float(max_r2)
        }

    return release_estimates


def print_summary_report(
    pooled_df: pd.DataFrame,
    r2_summary_df: pd.DataFrame,
    phase_summary_df: pd.DataFrame,
    release_estimates: dict
):
    """Print a human-readable summary of findings."""
    print("\n" + "=" * 70)
    print("FRAME-BY-FRAME REGRESSION SUMMARY")
    print("=" * 70)

    # 1. Release frame estimates
    print("\n--- ESTIMATED RELEASE FRAMES (Peak R2) ---")
    for target, info in release_estimates.items():
        frame = info['estimated_release_frame']
        r2 = info['peak_mean_r2']
        time_sec = frame / 60  # 60 fps
        print(f"  {target}: Frame {frame} ({time_sec:.2f}s), Mean R2 = {r2:.4f}")

    # 2. Phase summary
    print("\n--- PHASE SUMMARY ---")
    print("Phase          |   Target   | Best Feature                    | R2")
    print("-" * 70)
    for _, row in phase_summary_df.iterrows():
        print(f"{row['phase']:14} | {row['target']:10} | {row['best_feature'][:30]:31} | {row['best_feature_r2']:.4f}")

    # 3. R2 trend
    print("\n--- R2 TREND BY FRAME (averaged across all features) ---")
    for target in TARGET_COLS:
        target_data = r2_summary_df[r2_summary_df['target'] == target]

        # Sample at key frames
        key_frames = [0, 30, 60, 90, 120, 150, 180, 210, 239]
        print(f"\n  {target.upper()}:")
        print("    Frame:  ", end="")
        for f in key_frames:
            print(f"{f:7}", end="")
        print()
        print("    Mean R2:", end="")
        for f in key_frames:
            row = target_data[target_data['frame'] == f]
            if len(row) > 0:
                r2 = row['mean_r2'].values[0]
                print(f" {r2:.4f}", end="")
            else:
                print("    N/A", end="")
        print()

    # 4. Top features at release frame
    print("\n--- TOP FEATURES AT ESTIMATED RELEASE ---")
    for target, info in release_estimates.items():
        frame = info['estimated_release_frame']
        frame_data = pooled_df[(pooled_df['frame'] == frame) & (pooled_df['target'] == target)]
        top_5 = frame_data.nlargest(5, 'r_squared')

        print(f"\n  {target.upper()} (Frame {frame}):")
        for _, row in top_5.iterrows():
            print(f"    {row['feature_name'][:40]:40} R2={row['r_squared']:.4f} p={row['p_value']:.4f}")


def main():
    """Main function to run all frame-by-frame regression analyses."""
    print("=" * 70)
    print("FRAME-BY-FRAME REGRESSION ANALYSIS")
    print("=" * 70)

    start_time = time.time()

    # Load data
    print("\nLoading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    participant_ids = meta['participant_id'].values

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {len(keypoint_cols)}")
    print(f"Participants: {np.unique(participant_ids)}")
    print(f"Shots per participant: {pd.Series(participant_ids).value_counts().sort_index().to_dict()}")

    # 1. Pooled per-frame regression
    print("\n" + "=" * 70)
    print("1. POOLED PER-FRAME REGRESSION")
    print(f"   240 frames x 207 features x 3 targets = 149,040 regressions")
    print("=" * 70)
    t1 = time.time()
    pooled_df = compute_pooled_frame_regression(X, y, keypoint_cols)
    pooled_df.to_csv(OUTPUT_DIR / "frame_regression_pooled.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'frame_regression_pooled.csv'}")
    print(f"Rows: {len(pooled_df)}, Time: {time.time() - t1:.1f}s")

    # 2. Per-player per-frame regression
    print("\n" + "=" * 70)
    print("2. PER-PLAYER PER-FRAME REGRESSION")
    print(f"   240 frames x 5 players x 207 features x 3 targets = 745,200 regressions")
    print("=" * 70)
    t2 = time.time()
    player_df = compute_perplayer_frame_regression(X, y, keypoint_cols, participant_ids)
    player_df.to_csv(OUTPUT_DIR / "frame_regression_per_player.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'frame_regression_per_player.csv'}")
    print(f"Rows: {len(player_df)}, Time: {time.time() - t2:.1f}s")

    # 3. Binned frames (10-frame windows)
    print("\n" + "=" * 70)
    print("3. BINNED FRAME REGRESSION (10-frame windows)")
    print(f"   24 bins x 5 players x 207 features x 3 targets = 74,520 regressions")
    print("=" * 70)
    t3 = time.time()
    binned_df = compute_binned_frame_regression(X, y, keypoint_cols, participant_ids, bin_size=10)
    binned_df.to_csv(OUTPUT_DIR / "frame_regression_binned.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'frame_regression_binned.csv'}")
    print(f"Rows: {len(binned_df)}, Time: {time.time() - t3:.1f}s")

    # 4. Key frames only
    print("\n" + "=" * 70)
    print("4. KEY FRAME REGRESSION")
    key_frames = [0, 60, 120, 180, 239]
    print(f"   Key frames: {key_frames}")
    print(f"   5 frames x 5 players x 207 features x 3 targets = 15,525 regressions")
    print("=" * 70)
    t4 = time.time()
    X_key = X[:, key_frames, :]
    key_df = compute_perplayer_frame_regression(X_key, y, keypoint_cols, participant_ids, frame_indices=key_frames)
    key_df.to_csv(OUTPUT_DIR / "frame_regression_key_frames.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'frame_regression_key_frames.csv'}")
    print(f"Rows: {len(key_df)}, Time: {time.time() - t4:.1f}s")

    # 5. Generate summaries
    print("\n" + "=" * 70)
    print("5. GENERATING SUMMARIES")
    print("=" * 70)

    # R2 by frame summary
    print("Computing R2 by frame summary...")
    r2_summary_df = generate_frame_r2_summary(pooled_df)
    r2_summary_df.to_csv(OUTPUT_DIR / "frame_r2_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'frame_r2_summary.csv'}")

    # Phase summary
    print("Computing phase summary...")
    phase_summary_df = generate_phase_summary(pooled_df)
    phase_summary_df.to_csv(OUTPUT_DIR / "frame_phase_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'frame_phase_summary.csv'}")

    # Release frame detection
    print("Detecting release frame...")
    release_estimates = detect_release_frame(r2_summary_df)

    # Print summary report
    print_summary_report(pooled_df, r2_summary_df, phase_summary_df, release_estimates)

    # Final stats
    total_time = time.time() - start_time
    total_regressions = len(pooled_df) + len(player_df) + len(binned_df) + len(key_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total regressions computed: {total_regressions:,}")
    print(f"Total time: {total_time:.1f}s ({total_regressions/total_time:.0f} regressions/second)")
    print("\nOutput files:")
    print(f"  - frame_regression_pooled.csv      ({len(pooled_df):,} rows)")
    print(f"  - frame_regression_per_player.csv  ({len(player_df):,} rows)")
    print(f"  - frame_regression_binned.csv      ({len(binned_df):,} rows)")
    print(f"  - frame_regression_key_frames.csv  ({len(key_df):,} rows)")
    print(f"  - frame_r2_summary.csv")
    print(f"  - frame_phase_summary.csv")

    return pooled_df, player_df, binned_df, key_df, r2_summary_df, phase_summary_df, release_estimates


if __name__ == "__main__":
    main()
