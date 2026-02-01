"""
Blend GP predictions with Sub 133 to leverage diversity.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

def main():
    # Load submissions
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub140 = pd.read_csv(SUBMISSION_DIR / "submission_140.csv")  # Optimized GP

    print("Sub 133 (LB 0.007809):")
    print(f"  angle: mean={sub133['scaled_angle'].mean():.4f}, std={sub133['scaled_angle'].std():.4f}")
    print(f"  depth: mean={sub133['scaled_depth'].mean():.4f}, std={sub133['scaled_depth'].std():.4f}")
    print(f"  lr: mean={sub133['scaled_left_right'].mean():.4f}, std={sub133['scaled_left_right'].std():.4f}")

    print("\nSub 140 (GP, LB 0.011246):")
    print(f"  angle: mean={sub140['scaled_angle'].mean():.4f}, std={sub140['scaled_angle'].std():.4f}")
    print(f"  depth: mean={sub140['scaled_depth'].mean():.4f}, std={sub140['scaled_depth'].std():.4f}")
    print(f"  lr: mean={sub140['scaled_left_right'].mean():.4f}, std={sub140['scaled_left_right'].std():.4f}")

    # Correlation
    print("\nCorrelation between submissions:")
    print(f"  angle: {np.corrcoef(sub133['scaled_angle'], sub140['scaled_angle'])[0,1]:.4f}")
    print(f"  depth: {np.corrcoef(sub133['scaled_depth'], sub140['scaled_depth'])[0,1]:.4f}")
    print(f"  lr: {np.corrcoef(sub133['scaled_left_right'], sub140['scaled_left_right'])[0,1]:.4f}")

    # Test different blend weights
    # Since Sub 133 is much better, weight it more heavily
    blend_weights = [0.9, 0.85, 0.8, 0.95]

    for w133 in blend_weights:
        w_gp = 1 - w133

        blended = pd.DataFrame({
            "id": sub133["id"],
            "scaled_angle": w133 * sub133["scaled_angle"] + w_gp * sub140["scaled_angle"],
            "scaled_depth": w133 * sub133["scaled_depth"] + w_gp * sub140["scaled_depth"],
            "scaled_left_right": w133 * sub133["scaled_left_right"] + w_gp * sub140["scaled_left_right"],
        })

        print(f"\n--- Blend: {w133:.0%} Sub133 + {w_gp:.0%} GP ---")
        print(f"  angle: mean={blended['scaled_angle'].mean():.4f}, std={blended['scaled_angle'].std():.4f}")
        print(f"  depth: mean={blended['scaled_depth'].mean():.4f}, std={blended['scaled_depth'].std():.4f}")
        print(f"  lr: mean={blended['scaled_left_right'].mean():.4f}, std={blended['scaled_left_right'].std():.4f}")

    # Create submission with 90% Sub133 + 10% GP
    best_weight = 0.9

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split("_")[1]) for f in existing if f.stem.split("_")[1].isdigit()]
    next_num = max(nums) + 1

    blended = pd.DataFrame({
        "id": sub133["id"],
        "scaled_angle": best_weight * sub133["scaled_angle"] + (1-best_weight) * sub140["scaled_angle"],
        "scaled_depth": best_weight * sub133["scaled_depth"] + (1-best_weight) * sub140["scaled_depth"],
        "scaled_left_right": best_weight * sub133["scaled_left_right"] + (1-best_weight) * sub140["scaled_left_right"],
    })

    submission_path = SUBMISSION_DIR / f"submission_{next_num}.csv"
    blended.to_csv(submission_path, index=False)
    print(f"\nBlend saved to: {submission_path}")
    print(f"Weights: {best_weight:.0%} Sub133 + {1-best_weight:.0%} GP")

if __name__ == "__main__":
    main()
