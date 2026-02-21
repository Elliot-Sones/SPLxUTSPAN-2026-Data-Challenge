"""
Mega Blend Optimizer: find optimal per-target weights across ALL diverse models.

Available diverse sources:
- Sub 3190: current best (base)
- Sub 3209: pure velocity (depth r~0.70, LR r~0.60)
- Sub 3204: validated velocity (depth r~0.69, LR r~0.71)
- Sub 3183: scratch velocity CNN (depth r~0.59)
- Sub 2608: pulse features
- Sub 2602: energy wave
- Sub 1557: CNN temporal
- Sub 1507: trajectory (depth r~0.63)
- Sub 2679: XGBoost tree

Uses grid search to find optimal per-target weights.
Constraint: total diverse weight per target <= 15% (avoid overfitting to diversity).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import fcntl
from itertools import product

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"


def get_next_sub_num():
    lock = SUBMISSION_DIR / ".submission_lock"
    lock.touch(exist_ok=True)
    with open(lock, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            nums = [int(p.stem.split("_")[1]) for p in SUBMISSION_DIR.glob("submission_*.csv")
                    if p.stem.split("_")[1].isdigit()]
            n = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{n}.csv").touch()
            return n
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_sub(num):
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    print("=" * 70)
    print("MEGA BLEND OPTIMIZER")
    print("=" * 70)

    # Load all candidate submissions
    candidates = {
        3190: "base (current best LB 0.006331)",
        3209: "pure velocity (LR r=0.60)",
        3204: "validated velocity (depth r=0.69)",
        3183: "scratch CNN standalone",
        2608: "pulse features standalone",
        2602: "energy wave standalone",
        1557: "CNN temporal standalone",
        1507: "trajectory standalone",
        2679: "XGBoost tree standalone",
    }

    subs = {}
    for num, desc in candidates.items():
        df = load_sub(num)
        if df is not None:
            subs[num] = df
            print(f"  Loaded Sub {num}: {desc}")
        else:
            print(f"  MISSING Sub {num}: {desc}")

    base_num = 3190
    base = subs[base_num]
    diverse_nums = [k for k in subs if k != base_num]

    print(f"\n{'=' * 50}")
    print(f"Diversity matrix vs Sub {base_num}:")
    print(f"{'=' * 50}")
    print(f"  {'Sub':<6} {'angle r':>8} {'depth r':>8} {'LR r':>8}")
    for num in diverse_nums:
        df = subs[num]
        r_a = np.corrcoef(base["scaled_angle"], df["scaled_angle"])[0, 1]
        r_d = np.corrcoef(base["scaled_depth"], df["scaled_depth"])[0, 1]
        r_lr = np.corrcoef(base["scaled_left_right"], df["scaled_left_right"])[0, 1]
        print(f"  {num:<6} {r_a:>8.4f} {r_d:>8.4f} {r_lr:>8.4f}")

    # Cross-diversity between diverse sources
    print(f"\n{'=' * 50}")
    print(f"Cross-diversity between diverse sources (depth):")
    print(f"{'=' * 50}")
    depth_r = np.zeros((len(diverse_nums), len(diverse_nums)))
    for i, n1 in enumerate(diverse_nums):
        for j, n2 in enumerate(diverse_nums):
            r = np.corrcoef(subs[n1]["scaled_depth"], subs[n2]["scaled_depth"])[0, 1]
            depth_r[i, j] = r

    print(f"  {'':>6}", end="")
    for n in diverse_nums:
        print(f"  {n:>6}", end="")
    print()
    for i, n1 in enumerate(diverse_nums):
        print(f"  {n1:>6}", end="")
        for j, n2 in enumerate(diverse_nums):
            print(f"  {depth_r[i, j]:>6.3f}", end="")
        print()

    print(f"\n{'=' * 50}")
    print(f"Cross-diversity between diverse sources (LR):")
    print(f"{'=' * 50}")
    lr_r = np.zeros((len(diverse_nums), len(diverse_nums)))
    for i, n1 in enumerate(diverse_nums):
        for j, n2 in enumerate(diverse_nums):
            r = np.corrcoef(subs[n1]["scaled_left_right"], subs[n2]["scaled_left_right"])[0, 1]
            lr_r[i, j] = r

    print(f"  {'':>6}", end="")
    for n in diverse_nums:
        print(f"  {n:>6}", end="")
    print()
    for i, n1 in enumerate(diverse_nums):
        print(f"  {n1:>6}", end="")
        for j, n2 in enumerate(diverse_nums):
            print(f"  {lr_r[i, j]:>6.3f}", end="")
        print()

    # ================================================================
    # STRATEGY: Generate best blend candidates
    # ================================================================
    print(f"\n{'=' * 50}")
    print(f"Generating blend candidates")
    print(f"{'=' * 50}")

    # For each target, pick the best diverse sources
    # Then grid search over weights

    targets = ["angle", "depth", "left_right"]

    # Best diverse sources per target (based on lowest r with base)
    # Sort by diversity (lowest r = most diverse)
    diverse_by_target = {}
    for t in targets:
        col = f"scaled_{t}"
        divs = []
        for num in diverse_nums:
            r = np.corrcoef(base[col], subs[num][col])[0, 1]
            divs.append((abs(r), r, num))
        divs.sort()
        diverse_by_target[t] = divs
        print(f"\n  {t} - most diverse sources:")
        for _, r, num in divs[:5]:
            print(f"    Sub {num}: r = {r:.4f}")

    # Generate blends: pick top 2-3 diverse sources per target
    # and search over weight combinations

    # For depth: top diverse = scratch CNN (3183), trajectory (1507), validated vel (3204)
    # For LR: top diverse = pure velocity (3209), pulse (2608), energy (2602)
    # For angle: top diverse = pulse (2608), pure vel (3209), scratch CNN (3183)

    # Weight grid: 0%, 1%, 2%, 3%, 5%
    weight_options = [0.0, 0.01, 0.02, 0.03, 0.05]

    best_blends = []

    # Strategy 1: per-target single diverse source
    for t in targets:
        col = f"scaled_{t}"
        for _, r, num in diverse_by_target[t][:4]:
            for w in [0.02, 0.03, 0.05, 0.07]:
                blend_spec = {targ: (base_num, 1.0) for targ in targets}
                blend_spec[t] = (num, w)
                # Calculate blend
                blend = {}
                for targ in targets:
                    bcol = f"scaled_{targ}"
                    if blend_spec[targ][0] == base_num:
                        blend[bcol] = base[bcol].values
                    else:
                        src_num, wt = blend_spec[targ]
                        blend[bcol] = (1 - wt) * base[bcol].values + wt * subs[src_num][bcol].values
                desc = f"{t}:{w*100:.0f}%Sub{num}"
                best_blends.append((desc, blend))

    # Strategy 2: multi-target diverse blends
    # Depth + LR simultaneously (angle stays at base)
    depth_sources = [d[2] for d in diverse_by_target["depth"][:3]]
    lr_sources = [d[2] for d in diverse_by_target["left_right"][:3]]

    for d_src in depth_sources:
        for lr_src in lr_sources:
            for d_w in [0.02, 0.03, 0.05]:
                for lr_w in [0.02, 0.03, 0.05]:
                    blend = {}
                    blend["scaled_angle"] = base["scaled_angle"].values
                    blend["scaled_depth"] = (1 - d_w) * base["scaled_depth"].values + d_w * subs[d_src]["scaled_depth"].values
                    blend["scaled_left_right"] = (1 - lr_w) * base["scaled_left_right"].values + lr_w * subs[lr_src]["scaled_left_right"].values
                    desc = f"d:{d_w*100:.0f}%S{d_src}_lr:{lr_w*100:.0f}%S{lr_src}"
                    best_blends.append((desc, blend))

    # Strategy 3: 3-way blend (angle + depth + LR from different sources)
    for a_src in [d[2] for d in diverse_by_target["angle"][:2]]:
        for d_src in depth_sources[:2]:
            for lr_src in lr_sources[:2]:
                for a_w in [0.01, 0.02]:
                    for d_w in [0.03, 0.05]:
                        for lr_w in [0.03, 0.05]:
                            blend = {}
                            blend["scaled_angle"] = (1 - a_w) * base["scaled_angle"].values + a_w * subs[a_src]["scaled_angle"].values
                            blend["scaled_depth"] = (1 - d_w) * base["scaled_depth"].values + d_w * subs[d_src]["scaled_depth"].values
                            blend["scaled_left_right"] = (1 - lr_w) * base["scaled_left_right"].values + lr_w * subs[lr_src]["scaled_left_right"].values
                            desc = f"a:{a_w*100:.0f}%S{a_src}_d:{d_w*100:.0f}%S{d_src}_lr:{lr_w*100:.0f}%S{lr_src}"
                            best_blends.append((desc, blend))

    print(f"\n  Generated {len(best_blends)} blend candidates")

    # Rank by diversity from base (lower correlation = better)
    # Since we can't evaluate on LB, rank by total diversity
    ranked = []
    for desc, blend in best_blends:
        total_div = 0
        for t in targets:
            col = f"scaled_{t}"
            r = np.corrcoef(base[col], blend[col])[0, 1]
            total_div += (1 - abs(r))
        ranked.append((total_div, desc, blend))

    ranked.sort(reverse=True)

    print(f"\n  Top 20 by diversity from base:")
    for i, (div, desc, _) in enumerate(ranked[:20]):
        print(f"    {i+1}. diversity={div:.4f}: {desc}")

    # Save top 10 as submissions
    print(f"\n  Saving top 10 blends as submissions:")
    saved = []
    for i, (div, desc, blend) in enumerate(ranked[:10]):
        bn = get_next_sub_num()
        sub = pd.DataFrame({"id": base["id"]})
        for col in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            sub[col] = blend[col]
        sub.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
        print(f"    Sub {bn}: {desc} (div={div:.4f})")
        saved.append((bn, desc))

    # Also save a few moderate-diversity ones (more conservative)
    print(f"\n  Saving moderate-diversity blends (safer bets):")
    mid = len(ranked) // 3
    for i, (div, desc, blend) in enumerate(ranked[mid:mid+5]):
        bn = get_next_sub_num()
        sub = pd.DataFrame({"id": base["id"]})
        for col in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            sub[col] = blend[col]
        sub.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
        print(f"    Sub {bn}: {desc} (div={div:.4f})")
        saved.append((bn, desc))

    print("\nDone.")


if __name__ == "__main__":
    main()
