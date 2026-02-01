"""
Reverse Engineer Sub 133

Sub 133 is a blend: 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111
Analyze what each component contributes and how they differ.
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
    print("REVERSE ENGINEERING SUB 133")
    print("=" * 70)

    # Load components
    sub9 = pd.read_csv(SUBMISSION_DIR / "submission_9.csv")
    sub10 = pd.read_csv(SUBMISSION_DIR / "submission_10.csv")
    sub25 = pd.read_csv(SUBMISSION_DIR / "submission_25.csv")
    sub111 = pd.read_csv(SUBMISSION_DIR / "submission_111.csv")
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    test_meta = load_metadata(train=False)
    cols = ["scaled_angle", "scaled_depth", "scaled_left_right"]

    # Verify blend formula
    print("\n1. VERIFYING BLEND FORMULA")
    print("-" * 50)

    expected = 0.05 * sub25[cols].values + 0.30 * sub9[cols].values + \
               0.44 * sub10[cols].values + 0.21 * sub111[cols].values
    actual = sub133[cols].values

    diff = np.abs(expected - actual).mean()
    print(f"Average difference from expected blend: {diff:.8f}")

    # Component analysis
    print("\n2. COMPONENT PROFILES")
    print("-" * 50)

    components = [
        ("Sub 9", sub9, 0.30),
        ("Sub 10", sub10, 0.44),
        ("Sub 25", sub25, 0.05),
        ("Sub 111", sub111, 0.21),
        ("Sub 133 (blend)", sub133, 1.0)
    ]

    for name, df, weight in components:
        print(f"\n{name} (weight={weight:.2f}):")
        for col in cols:
            print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    # Per-player analysis
    print("\n3. PER-PLAYER COMPONENT ANALYSIS")
    print("-" * 50)

    for pid in sorted(test_meta["participant_id"].unique()):
        mask = test_meta["participant_id"] == pid
        print(f"\nPlayer {pid} ({mask.sum()} samples):")

        for col in cols:
            print(f"\n  {col}:")
            for name, df, weight in components:
                vals = df[col][mask].values
                print(f"    {name}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Correlation between components
    print("\n4. INTER-COMPONENT CORRELATIONS")
    print("-" * 50)

    for col in cols:
        print(f"\n{col}:")
        for name1, df1, _ in components[:-1]:
            for name2, df2, _ in components[:-1]:
                if name1 < name2:
                    corr = np.corrcoef(df1[col], df2[col])[0, 1]
                    print(f"  {name1} vs {name2}: {corr:.4f}")

    # What makes Sub 133 special?
    print("\n5. DIVERSITY ANALYSIS")
    print("-" * 50)

    # For each sample, compute how much components agree
    for col in cols:
        component_preds = np.column_stack([
            sub9[col].values,
            sub10[col].values,
            sub25[col].values,
            sub111[col].values
        ])

        # Per-sample std of component predictions
        sample_disagreement = np.std(component_preds, axis=1)

        print(f"\n{col} component disagreement:")
        print(f"  Mean: {sample_disagreement.mean():.4f}")
        print(f"  Max:  {sample_disagreement.max():.4f}")

        # Samples with high disagreement
        high_disagree_idx = np.argsort(sample_disagreement)[-5:]
        print(f"  Highest disagreement samples:")
        for idx in high_disagree_idx:
            pid = test_meta.iloc[idx]["participant_id"]
            preds = component_preds[idx]
            blend = sub133[col].iloc[idx]
            print(f"    Sample {idx} (P{pid}): {preds} -> blend={blend:.4f}")

    # Try modified blend weights
    print("\n6. ALTERNATIVE BLEND WEIGHTS")
    print("-" * 50)

    # Target: minimize profile distance while maintaining diversity
    target_profile = {"angle_std": 0.1377, "depth_mean": 0.5055}

    best_weights = None
    best_profile_dist = float("inf")

    # Grid search
    for w9 in np.linspace(0.2, 0.5, 7):
        for w10 in np.linspace(0.3, 0.6, 7):
            for w25 in np.linspace(0, 0.15, 4):
                w111 = 1 - w9 - w10 - w25
                if w111 < 0 or w111 > 0.4:
                    continue

                blend = w9 * sub9[cols].values + w10 * sub10[cols].values + \
                        w25 * sub25[cols].values + w111 * sub111[cols].values

                angle_std = np.std(blend[:, 0])
                depth_mean = np.mean(blend[:, 1])

                profile_dist = abs(angle_std - target_profile["angle_std"]) + \
                               abs(depth_mean - target_profile["depth_mean"])

                if profile_dist < best_profile_dist:
                    best_profile_dist = profile_dist
                    best_weights = (w9, w10, w25, w111)

    print(f"Best weights found: Sub9={best_weights[0]:.2f}, Sub10={best_weights[1]:.2f}, "
          f"Sub25={best_weights[2]:.2f}, Sub111={best_weights[3]:.2f}")
    print(f"Profile dist: {best_profile_dist:.6f}")

    # Create optimized blend
    w9, w10, w25, w111 = best_weights
    optimized_blend = w9 * sub9[cols].values + w10 * sub10[cols].values + \
                      w25 * sub25[cols].values + w111 * sub111[cols].values

    print(f"\nOptimized blend profile:")
    print(f"  angle_std: {np.std(optimized_blend[:, 0]):.4f}")
    print(f"  depth_mean: {np.mean(optimized_blend[:, 1]):.4f}")

    # Compare to Sub 133
    corr = np.corrcoef(optimized_blend.ravel(), sub133[cols].values.ravel())[0, 1]
    print(f"  Correlation with Sub 133: {corr:.4f}")

    # Save if different from Sub 133
    if corr < 0.9999:
        nums = [int(f.stem.split("_")[1]) for f in SUBMISSION_DIR.glob("submission_*.csv")
                if f.stem.split("_")[1].isdigit()]
        next_num = max(nums) + 1

        submission = pd.DataFrame({
            "id": test_meta["id"].values,
            "scaled_angle": optimized_blend[:, 0],
            "scaled_depth": optimized_blend[:, 1],
            "scaled_left_right": optimized_blend[:, 2]
        })
        path = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(path, index=False)
        print(f"\nSaved optimized blend: {path}")
        print(f"Weights: 9={w9:.2f}, 10={w10:.2f}, 25={w25:.2f}, 111={w111:.2f}")
    else:
        print("\nOptimized blend is essentially identical to Sub 133")


if __name__ == "__main__":
    main()
