"""
Analyze Prediction Differences

Since our clean model is 97% correlated with Sub 133,
analyze where they differ to potentially improve.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"


def main():
    print("=" * 70)
    print("ANALYZING PREDICTION DIFFERENCES")
    print("=" * 70)

    # Load submissions
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub147 = pd.read_csv(SUBMISSION_DIR / "submission_147.csv")  # Clean base
    sub148 = pd.read_csv(SUBMISSION_DIR / "submission_148.csv")  # Clean calibrated

    test_meta = load_metadata(train=False)

    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    print("\nPer-target correlations (Sub 133 vs Clean Base):")
    for col in cols:
        corr = np.corrcoef(sub133[col], sub147[col])[0, 1]
        print(f"  {col}: {corr:.4f}")

    print("\nPer-target correlations (Sub 133 vs Clean Calibrated):")
    for col in cols:
        corr = np.corrcoef(sub133[col], sub148[col])[0, 1]
        print(f"  {col}: {corr:.4f}")

    # Analyze differences
    print("\n" + "=" * 70)
    print("DIFFERENCE ANALYSIS")
    print("=" * 70)

    for col in cols:
        diff = sub147[col] - sub133[col]
        print(f"\n{col}:")
        print(f"  Mean diff: {diff.mean():.4f}")
        print(f"  Std diff: {diff.std():.4f}")
        print(f"  Max diff: {diff.max():.4f}")
        print(f"  Min diff: {diff.min():.4f}")

        # Samples with largest differences
        abs_diff = np.abs(diff)
        top_diff_idx = np.argsort(abs_diff)[-5:]

        print(f"  Top 5 differences:")
        for idx in top_diff_idx:
            pid = test_meta.iloc[idx]["participant_id"]
            print(f"    Sample {idx} (Player {pid}): Sub133={sub133[col].iloc[idx]:.4f}, "
                  f"Clean={sub147[col].iloc[idx]:.4f}, Diff={diff.iloc[idx]:.4f}")

    # Per-player analysis
    print("\n" + "=" * 70)
    print("PER-PLAYER CORRELATION")
    print("=" * 70)

    for pid in sorted(test_meta["participant_id"].unique()):
        mask = test_meta["participant_id"] == pid
        print(f"\nPlayer {pid}:")
        for col in cols:
            corr = np.corrcoef(sub133[col][mask], sub147[col][mask])[0, 1]
            print(f"  {col}: {corr:.4f}")

    # Try strategic blend: use our model only where we might be better
    print("\n" + "=" * 70)
    print("STRATEGIC BLENDING")
    print("=" * 70)

    # Idea: For samples where Sub 133 and Clean differ a lot,
    # average them (hedging). Otherwise, trust Sub 133.

    for threshold in [0.05, 0.1, 0.15, 0.2]:
        strategic = sub133[cols].copy()

        for col in cols:
            diff = np.abs(sub147[col] - sub133[col])
            large_diff = diff > threshold

            # Where they differ a lot, average them
            strategic.loc[large_diff, col] = (
                sub133.loc[large_diff, col] + sub147.loc[large_diff, col]
            ) / 2

        # Profile of strategic blend
        angle_std = strategic["scaled_angle"].std()
        depth_mean = strategic["scaled_depth"].mean()
        profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

        n_changed = sum(np.abs(sub147[col] - sub133[col]) > threshold for col in cols)

        print(f"\nThreshold {threshold}: {n_changed[0]} angle, {n_changed[1]} depth, {n_changed[2]} lr changed")
        print(f"  Profile: angle_std={angle_std:.4f}, depth_mean={depth_mean:.4f}")
        print(f"  Profile dist: {profile_dist:.4f}")

    # Save best strategic blend
    print("\n" + "=" * 70)
    print("SAVING STRATEGIC BLEND")
    print("=" * 70)

    # Use threshold 0.1
    strategic = sub133[cols].copy()
    for col in cols:
        diff = np.abs(sub147[col] - sub133[col])
        large_diff = diff > 0.1
        strategic.loc[large_diff, col] = (
            sub133.loc[large_diff, col] + sub147.loc[large_diff, col]
        ) / 2

    nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
            if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": strategic["scaled_angle"],
        "scaled_depth": strategic["scaled_depth"],
        "scaled_left_right": strategic["scaled_left_right"]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved strategic blend: {path}")

    # Also try weighted blend based on per-player correlations
    # Players where our model disagrees more might benefit from our predictions
    print("\n" + "=" * 70)
    print("PER-PLAYER WEIGHTED BLEND")
    print("=" * 70)

    # Weight clean model higher for players where correlation is lower (more disagreement)
    weighted = pd.DataFrame(index=range(len(test_meta)))

    for col in cols:
        weighted[col] = np.zeros(len(test_meta))

        for pid in sorted(test_meta["participant_id"].unique()):
            mask = test_meta["participant_id"] == pid

            # Correlation for this player
            corr = np.corrcoef(sub133[col][mask], sub147[col][mask])[0, 1]

            # Higher weight for clean model when correlation is lower
            clean_weight = max(0, min(0.3, (1 - corr) * 0.5))

            weighted.loc[mask, col] = (
                (1 - clean_weight) * sub133[col][mask].values +
                clean_weight * sub147[col][mask].values
            )

    # Profile
    angle_std = weighted["scaled_angle"].std()
    depth_mean = weighted["scaled_depth"].mean()
    profile_dist = abs(angle_std - 0.1377) + abs(depth_mean - 0.5055)

    print(f"Weighted blend profile: angle_std={angle_std:.4f}, depth_mean={depth_mean:.4f}")
    print(f"Profile dist: {profile_dist:.4f}")

    next_num += 1
    submission = pd.DataFrame({
        "id": test_meta["id"].values,
        "scaled_angle": weighted["scaled_angle"],
        "scaled_depth": weighted["scaled_depth"],
        "scaled_left_right": weighted["scaled_left_right"]
    })
    path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(path, index=False)
    print(f"Saved per-player weighted blend: {path}")


if __name__ == "__main__":
    main()
