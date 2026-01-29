"""
Monitor Brute Force Search Progress.

Utility to view search progress, best results, and generate plots.

Usage:
    uv run python scripts/monitor_search.py           # Basic summary
    uv run python scripts/monitor_search.py --plot    # With improvement plot
    uv run python scripts/monitor_search.py --top 20  # Show top 20 results
"""

import argparse
import json
import pandas as pd
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"
RESULTS_FILE = OUTPUT_DIR / "brute_force_results.csv"
PLOT_FILE = OUTPUT_DIR / "search_progress.png"


def load_results() -> pd.DataFrame:
    """Load results from CSV."""
    if not RESULTS_FILE.exists():
        print(f"No results file found at: {RESULTS_FILE}")
        sys.exit(1)

    df = pd.read_csv(RESULTS_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def print_summary(df: pd.DataFrame, total_trials: int = 10000):
    """Print summary statistics."""
    n_completed = len(df)
    progress_pct = n_completed / total_trials * 100

    print("=" * 60)
    print("BRUTE FORCE SEARCH PROGRESS")
    print("=" * 60)
    print(f"\nTotal trials completed: {n_completed:,}/{total_trials:,} ({progress_pct:.1f}%)")

    if n_completed == 0:
        return

    # Time statistics
    duration_total = df["duration_seconds"].sum()
    duration_avg = df["duration_seconds"].mean()
    print(f"Total runtime: {timedelta(seconds=int(duration_total))}")
    print(f"Average trial time: {duration_avg:.1f}s")

    # Estimate remaining time
    remaining_trials = total_trials - n_completed
    est_remaining_seconds = remaining_trials * duration_avg
    print(f"Estimated remaining: {timedelta(seconds=int(est_remaining_seconds))}")

    # Best results
    best_idx = df["cv_score_combined"].idxmin()
    best = df.loc[best_idx]

    print("\n" + "-" * 60)
    print("BEST RESULT")
    print("-" * 60)
    print(f"Trial ID: {best['trial_id']}")
    print(f"Combined Score: {best['cv_score_combined']:.6f}")
    print(f"  - Angle:      {best['cv_score_angle']:.6f}")
    print(f"  - Depth:      {best['cv_score_depth']:.6f}")
    print(f"  - Left/Right: {best['cv_score_left_right']:.6f}")
    print(f"Model: {best['model_type']}")
    print(f"Features: {best['n_features']} selected")
    print(f"Best params: {best['best_params']}")

    # Best per target
    print("\n" + "-" * 60)
    print("BEST PER TARGET")
    print("-" * 60)

    for target in ["angle", "depth", "left_right"]:
        col = f"cv_score_{target}"
        best_idx = df[col].idxmin()
        best_row = df.loc[best_idx]
        print(f"{target:12s}: {best_row[col]:.6f} (trial {best_row['trial_id']}, "
              f"{best_row['model_type']}, {best_row['n_features']} features)")


def print_model_distribution(df: pd.DataFrame, top_n: int = 100):
    """Print model distribution in top results."""
    print("\n" + "-" * 60)
    print(f"MODEL DISTRIBUTION (Top {top_n})")
    print("-" * 60)

    top_df = df.nsmallest(top_n, "cv_score_combined")
    model_counts = top_df["model_type"].value_counts()

    for model, count in model_counts.items():
        pct = count / top_n * 100
        bar = "#" * int(pct / 2)
        print(f"{model:12s}: {count:4d} ({pct:5.1f}%) {bar}")


def print_feature_size_distribution(df: pd.DataFrame, top_n: int = 100):
    """Print feature size distribution in top results."""
    print("\n" + "-" * 60)
    print(f"FEATURE SIZE DISTRIBUTION (Top {top_n})")
    print("-" * 60)

    top_df = df.nsmallest(top_n, "cv_score_combined")
    size_counts = top_df["n_features"].value_counts().sort_index()

    for size, count in size_counts.items():
        pct = count / top_n * 100
        bar = "#" * int(pct / 2)
        print(f"{size:3d} features: {count:4d} ({pct:5.1f}%) {bar}")


def print_top_results(df: pd.DataFrame, n: int = 10):
    """Print top N results."""
    print("\n" + "-" * 60)
    print(f"TOP {n} RESULTS")
    print("-" * 60)

    top_df = df.nsmallest(n, "cv_score_combined")

    print(f"{'Rank':>4} {'Trial':>6} {'Model':>10} {'Feats':>5} "
          f"{'Combined':>10} {'Angle':>10} {'Depth':>10} {'L/R':>10}")
    print("-" * 78)

    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        print(f"{i:4d} {row['trial_id']:6d} {row['model_type']:>10} {row['n_features']:5d} "
              f"{row['cv_score_combined']:10.6f} {row['cv_score_angle']:10.6f} "
              f"{row['cv_score_depth']:10.6f} {row['cv_score_left_right']:10.6f}")


def plot_progress(df: pd.DataFrame, output_file: Path):
    """Generate improvement-over-time plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot")
        return

    # Sort by timestamp
    df_sorted = df.sort_values("timestamp")

    # Compute running best
    df_sorted["running_best"] = df_sorted["cv_score_combined"].cummin()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Score over time with running best
    ax = axes[0, 0]
    ax.scatter(range(len(df_sorted)), df_sorted["cv_score_combined"],
               alpha=0.3, s=5, label="Trial scores")
    ax.plot(range(len(df_sorted)), df_sorted["running_best"],
            'r-', linewidth=2, label="Best so far")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Combined MSE")
    ax.set_title("Score Improvement Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Score by model type
    ax = axes[0, 1]
    model_scores = df.groupby("model_type")["cv_score_combined"].agg(["min", "mean", "std"])
    x = range(len(model_scores))
    ax.bar(x, model_scores["mean"], yerr=model_scores["std"], alpha=0.7, capsize=5)
    ax.scatter(x, model_scores["min"], color="red", s=100, zorder=5, label="Best")
    ax.set_xticks(x)
    ax.set_xticklabels(model_scores.index, rotation=45)
    ax.set_ylabel("Combined MSE")
    ax.set_title("Score by Model Type")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Score by feature count
    ax = axes[1, 0]
    size_scores = df.groupby("n_features")["cv_score_combined"].agg(["min", "mean", "std"])
    x = range(len(size_scores))
    ax.bar(x, size_scores["mean"], yerr=size_scores["std"], alpha=0.7, capsize=5)
    ax.scatter(x, size_scores["min"], color="red", s=100, zorder=5, label="Best")
    ax.set_xticks(x)
    ax.set_xticklabels(size_scores.index)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Combined MSE")
    ax.set_title("Score by Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Histogram of scores
    ax = axes[1, 1]
    ax.hist(df["cv_score_combined"], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(df["cv_score_combined"].min(), color='r', linestyle='--',
               linewidth=2, label=f"Best: {df['cv_score_combined'].min():.6f}")
    ax.axvline(df["cv_score_combined"].mean(), color='g', linestyle='--',
               linewidth=2, label=f"Mean: {df['cv_score_combined'].mean():.6f}")
    ax.set_xlabel("Combined MSE")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")


def print_recent_trials(df: pd.DataFrame, n: int = 10):
    """Print most recent trials."""
    print("\n" + "-" * 60)
    print(f"MOST RECENT {n} TRIALS")
    print("-" * 60)

    recent = df.nlargest(n, "timestamp")

    for _, row in recent.iterrows():
        ts = row["timestamp"].strftime("%H:%M:%S")
        print(f"[{ts}] Trial {row['trial_id']}: {row['model_type']}, "
              f"{row['n_features']} features -> {row['cv_score_combined']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Monitor brute force search progress")
    parser.add_argument("--plot", action="store_true", help="Generate progress plot")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show")
    parser.add_argument("--total", type=int, default=10000, help="Total planned trials")

    args = parser.parse_args()

    # Load results
    df = load_results()

    # Print summaries
    print_summary(df, args.total)
    print_top_results(df, args.top)
    print_model_distribution(df)
    print_feature_size_distribution(df)
    print_recent_trials(df)

    # Generate plot if requested
    if args.plot:
        plot_progress(df, PLOT_FILE)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
