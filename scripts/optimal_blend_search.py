"""
Optimal Blend Search v2

Given multiple diverse submission-level predictions, find optimal
blend weights for each target independently.

Generates strategic blends combining diverse sources at small weights
with Sub 1640 (current best LB = 0.006698).
"""

import time
import numpy as np
import pandas as pd
import fcntl
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]


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


def main():
    t0 = time.time()
    print("=" * 70)
    print("OPTIMAL BLEND SEARCH v2")
    print("=" * 70)

    # Load base submissions
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
    sub_1640 = pd.read_csv(SUBMISSION_DIR / "submission_1640.csv")
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Load diverse standalone predictions
    candidates = {}
    for name, num in [
        ('feat_weighted', 1707),
        ('pca_subspace', 1708),
        ('multi_kernel', 1709),
        ('huber_weighted', 1710),
        ('power_transform', 1693),
    ]:
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        if path.exists():
            candidates[name] = pd.read_csv(path)
            print(f"  Loaded {name} (Sub {num})")

    # Also check for meta-feature and OOF stack predictions
    for name, num in [('meta_features', 1685), ('oof_stack', 1666)]:
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        if path.exists():
            candidates[name] = pd.read_csv(path)
            print(f"  Loaded {name} (Sub {num})")

    print(f"\n  Total diverse candidates: {len(candidates)}")

    # ==============================================================
    # DIVERSITY ANALYSIS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("DIVERSITY ANALYSIS (r with Sub 1640)")
    print(f"{'=' * 70}")

    diversity = {}
    for name, sub in candidates.items():
        rs = {}
        for target in TARGETS:
            col = f'scaled_{target}'
            r = np.corrcoef(sub[col].values, sub_1640[col].values)[0, 1]
            rs[target] = r
        diversity[name] = rs
        print(f"  {name:20s}: angle={rs['angle']:.4f}, depth={rs['depth']:.4f}, lr={rs['left_right']:.4f}")

    # Also show Sub 1350 vs Sub 1640
    rs_1350 = {}
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_1350[col].values, sub_1640[col].values)[0, 1]
        rs_1350[target] = r
    print(f"  {'sub_1350':20s}: angle={rs_1350['angle']:.4f}, depth={rs_1350['depth']:.4f}, lr={rs_1350['left_right']:.4f}")

    # ==============================================================
    # STRATEGIC BLENDS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING STRATEGIC BLENDS")
    print(f"{'=' * 70}")

    # Strategy 1: Multi-signal equal-weight blend
    diverse_names = [n for n in candidates]
    for total_w in [0.05, 0.08, 0.10, 0.15]:
        per_w = total_w / len(diverse_names)
        sub_num = get_next_submission_number()
        blended = sub_1640.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            val = (1 - total_w) * sub_1640[col].values
            for name in diverse_names:
                val += per_w * candidates[name][col].values
            blended[col] = val
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {total_w*100:.0f}% equal-weight ({len(diverse_names)} sources) + {(1-total_w)*100:.0f}% Sub 1640")

    # Strategy 2: Per-target most diverse
    target_best_diverse = {}
    for target in TARGETS:
        best_name = min(diversity, key=lambda n: diversity[n][target])
        target_best_diverse[target] = best_name
    print(f"\n  Most diverse per target: {target_best_diverse}")

    for total_w in [0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_1640.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            source = target_best_diverse[target]
            blended[col] = (1 - total_w) * sub_1640[col] + total_w * candidates[source][col]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        desc = ", ".join(f"{t}={target_best_diverse[t]}" for t in TARGETS)
        print(f"  Sub {sub_num}: {total_w*100:.0f}% per-target diverse + {(1-total_w)*100:.0f}% Sub 1640 ({desc})")

    # Strategy 3: Only diverse sources (r < 0.95 for at least one target)
    very_diverse = [n for n in candidates if any(diversity[n][t] < 0.95 for t in TARGETS)]
    if very_diverse:
        print(f"\n  Very diverse sources (r<0.95 on any target): {very_diverse}")
        for total_w in [0.05, 0.10]:
            per_w = total_w / len(very_diverse)
            sub_num = get_next_submission_number()
            blended = sub_1640.copy()
            for target in TARGETS:
                col = f'scaled_{target}'
                val = (1 - total_w) * sub_1640[col].values
                for name in very_diverse:
                    val += per_w * candidates[name][col].values
                blended[col] = val
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            print(f"  Sub {sub_num}: {total_w*100:.0f}% very-diverse ({len(very_diverse)} sources) + {(1-total_w)*100:.0f}% Sub 1640")

    # Strategy 4: Diversity-weighted blend (more weight to more diverse)
    for total_w in [0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_1640.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            # Weight inversely by correlation with Sub 1640
            target_weights = {}
            total_inv_r = 0
            for name in candidates:
                inv_r = 1 - diversity[name][target]
                target_weights[name] = inv_r
                total_inv_r += inv_r
            # Normalize to sum to total_w
            val = (1 - total_w) * sub_1640[col].values
            for name in candidates:
                w = total_w * target_weights[name] / (total_inv_r + 1e-10)
                val += w * candidates[name][col].values
            blended[col] = val
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {total_w*100:.0f}% diversity-weighted + {(1-total_w)*100:.0f}% Sub 1640")

    # Strategy 5: 3-way blend with Sub 1350 offset
    # Sub 1640 = 90% Sub 1350 + 10% LASSO
    # Instead of blending with Sub 1640, blend with Sub 1350 + diverse + LASSO
    # This separates the three signals
    for lasso_w, div_w in [(0.08, 0.07), (0.10, 0.05), (0.05, 0.10)]:
        base_w = 1 - lasso_w - div_w
        sub_num = get_next_submission_number()
        # Use the LASSO-only prediction (Sub 1640 - Sub 1350) * 10 to isolate LASSO
        # Actually, just use Sub 1350 as base and add LASSO component from Sub 1640
        lasso_pred = {}
        for target in TARGETS:
            col = f'scaled_{target}'
            lasso_pred[target] = (sub_1640[col].values - 0.90 * sub_1350[col].values) / 0.10

        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            best_diverse = target_best_diverse[target]
            blended[col] = (base_w * sub_1350[col].values +
                           lasso_w * lasso_pred[target] +
                           div_w * candidates[best_diverse][col].values)
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {base_w*100:.0f}% Sub1350 + {lasso_w*100:.0f}% LASSO + {div_w*100:.0f}% diverse")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
