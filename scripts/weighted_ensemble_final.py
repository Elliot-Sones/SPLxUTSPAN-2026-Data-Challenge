"""
Weighted ensemble: learn optimal weights for 8 diverse models.
Simpler than full stacking but principled.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"
TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def load_submission(sub_num: int) -> pd.DataFrame | None:
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    print("=" * 60)
    print("WEIGHTED ENSEMBLE - FINAL SUBMISSION")
    print("=" * 60)

    # Key diverse models (based on LB scores and diversity)
    diverse_subs = [
        (3385, "Biomechanical"),      # NEW - LB 0.006243
        (3190, "Baseline Ridge"),
        (3326, "Position CNN"),
        (3294, "Velocity CNN"),
        (2608, "Pulse features"),
        (2622, "4-way blend"),
        (2716, "TreeAvg+CNN"),
        (2503, "Extended physics"),
    ]

    print(f"\nLoading {len(diverse_subs)} submissions...")
    preds_dict = {}
    for sub_num, name in diverse_subs:
        sub = load_submission(sub_num)
        if sub is None:
            print(f"  Sub {sub_num}: NOT FOUND")
            continue
        preds_dict[sub_num] = sub
        print(f"  Sub {sub_num} ({name}): {len(sub)} rows")

    n_models = len(preds_dict)
    test_ids = list(preds_dict.values())[0]["id"].values
    n_test = len(test_ids)

    # Extract predictions for each model per target
    print(f"\nExtracting predictions for {n_models} models x {len(TARGETS)} targets...")
    preds_per_target = {}

    for target in TARGETS:
        preds_all = []
        for sub_num in preds_dict.keys():
            pred = preds_dict[sub_num][target].values.astype(np.float32)
            preds_all.append(pred)
        preds_per_target[target] = np.column_stack(preds_all)  # (n_test, n_models)
        print(f"  {target}: {preds_per_target[target].shape}")

    # Find good weights by minimizing prediction variance (maximize consensus)
    # where models agree more, we're more confident
    print("\nOptimizing ensemble weights...")

    def score_weights(w):
        """Score function: lower = better. Use model variance as proxy."""
        w = np.abs(w)
        w = w / np.sum(w)  # Normalize
        total_var = 0
        for target in TARGETS:
            preds = preds_per_target[target]  # (n_test, n_models)
            weighted_pred = np.dot(preds, w)
            # Variance of individual model deviations
            var = np.mean(np.var(preds - weighted_pred[:, None], axis=1))
            total_var += var
        return total_var

    # Initial weights: uniform
    w0 = np.ones(n_models) / n_models

    # Optimize
    result = minimize(score_weights, w0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6})
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    print(f"\nOptimal weights:")
    for i, (sub_num, name) in enumerate(diverse_subs):
        if sub_num in preds_dict:
            print(f"  {sub_num} ({name:20s}): {optimal_weights[i]:.4f}")

    # Generate final predictions
    print(f"\nGenerating final predictions with optimal weights...")
    final_preds = []

    for target in TARGETS:
        preds = preds_per_target[target]  # (n_test, n_models)
        weighted = np.dot(preds, optimal_weights)
        weighted = np.clip(weighted, 0, 1)
        final_preds.append(weighted)

    final_preds = np.column_stack(final_preds)

    # Save submission
    nums = []
    for p in SUBMISSION_DIR.glob("submission_*.csv"):
        parts = p.stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    bn = max(nums + [0]) + 1

    sub = pd.DataFrame({
        "id": test_ids,
        "scaled_angle": final_preds[:, 0],
        "scaled_depth": final_preds[:, 1],
        "scaled_left_right": final_preds[:, 2]
    })

    sub.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)

    print(f"\n{'='*60}")
    print(f"FINAL SUBMISSION: Sub {bn}")
    print(f"{'='*60}")
    print(f"  Weighted ensemble of {n_models} diverse models")
    print(f"  Weights optimized to minimize prediction variance")
    print(f"  Current best (Sub 3385): LB 0.006243")
    print(f"  Ready for submission!")

    return bn


if __name__ == "__main__":
    main()
