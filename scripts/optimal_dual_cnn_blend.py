#!/usr/bin/env python3
"""
Optimal Dual CNN Blend: Combine multi-seed velocity CNN + full-run position CNN + Sub2716.

This script creates the best possible submission by blending:
1. Multi-seed velocity CNN (5-seed ensemble, CV ~0.006618)
2. Full-run position CNN (5-seed, CV 0.006075)
3. Sub 2716 as base (LB 0.006343)
4. Optionally Sub 3294 (5% vel + 95% Sub2716, LB 0.006306)

The key insight: position and velocity CNNs capture different signals
(depth correlation r=0.60, i.e. highly complementary).

Usage:
  uv run python scripts/optimal_dual_cnn_blend.py --vel-sub VEL_STANDALONE --pos-sub 3331
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = [f"scaled_{t}" for t in TARGETS]


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = []
    for f in existing:
        try:
            nums.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums) + 1 if nums else 1
    (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
    return next_num


def load_sub(num):
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Submission {num} not found: {path}")
    return pd.read_csv(path)


def compute_diversity(df1, df2, label1="A", label2="B"):
    for t in TARGETS:
        col = f"scaled_{t}"
        r = np.corrcoef(df1[col].values, df2[col].values)[0, 1]
        print(f"    {t}: r={r:.4f}")


def blend_and_save(components, desc):
    """
    components: list of (sub_df, weight_dict) where weight_dict maps target -> weight
    The weights for each target should sum to 1.0.
    """
    sub_num = get_next_submission_number()
    result = components[0][0][["id"]].copy()

    for t in TARGETS:
        col = f"scaled_{t}"
        blended = np.zeros(len(result))
        total_w = 0
        for df, weights in components:
            w = weights.get(t, weights.get("all", 0))
            blended += w * df[col].values
            total_w += w
        assert abs(total_w - 1.0) < 1e-6, f"Weights for {t} sum to {total_w}, not 1.0"
        result[col] = np.clip(blended, 0.0, 1.0)

    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    result.to_csv(path, index=False)
    print(f"  Sub {sub_num}: {desc}")
    return sub_num, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vel-sub", type=int, required=True,
                   help="Standalone multi-seed velocity CNN submission number")
    p.add_argument("--pos-sub", type=int, default=3331,
                   help="Standalone full-run position CNN submission number")
    p.add_argument("--base-sub", type=int, default=2716,
                   help="Base submission number")
    p.add_argument("--anchor-sub", type=int, default=3294,
                   help="Anchor submission (vel+base)")
    p.add_argument("--lb-best-sub", type=int, default=3326,
                   help="Current LB best submission")
    args = p.parse_args()

    print("=" * 70)
    print("OPTIMAL DUAL CNN BLEND")
    print("=" * 70)

    # Load submissions
    vel_df = load_sub(args.vel_sub)
    pos_df = load_sub(args.pos_sub)
    base_df = load_sub(args.base_sub)
    anchor_df = load_sub(args.anchor_sub)
    lb_best_df = load_sub(args.lb_best_sub)

    print(f"\n  Velocity CNN: Sub {args.vel_sub}")
    print(f"  Position CNN: Sub {args.pos_sub}")
    print(f"  Base: Sub {args.base_sub}")
    print(f"  Anchor: Sub {args.anchor_sub}")
    print(f"  LB Best: Sub {args.lb_best_sub}")

    # Diversity analysis
    print("\n--- DIVERSITY ANALYSIS ---")
    print(f"\n  Velocity vs Base (Sub{args.base_sub}):")
    compute_diversity(vel_df, base_df)
    print(f"\n  Position vs Base (Sub{args.base_sub}):")
    compute_diversity(pos_df, base_df)
    print(f"\n  Velocity vs Position:")
    compute_diversity(vel_df, pos_df)
    print(f"\n  Velocity vs Anchor (Sub{args.anchor_sub}):")
    compute_diversity(vel_df, anchor_df)
    print(f"\n  Position vs Anchor (Sub{args.anchor_sub}):")
    compute_diversity(pos_df, anchor_df)
    print(f"\n  Position vs LB Best (Sub{args.lb_best_sub}):")
    compute_diversity(pos_df, lb_best_df)

    submissions = []

    # ============================================================
    # Strategy 1: Upgrade the winning formula
    # Sub 3326 = 5% pilot_pos + 95% Sub3294 -> replace pilot with full-run pos
    # Sub 3294 = 5% vel + 95% Sub2716
    # Net: 5% full_pos + 4.75% vel + 90.25% Sub2716
    # ============================================================
    print("\n\n--- STRATEGY 1: Upgrade winning formula (replace pilot pos with full-run) ---")

    # 5% full pos + 95% Sub3294
    sub_num, _ = blend_and_save([
        (pos_df, {"all": 0.05}),
        (anchor_df, {"all": 0.95}),
    ], f"5% full-pos + 95% Sub{args.anchor_sub} (upgrade of Sub3326)")
    submissions.append(sub_num)

    # ============================================================
    # Strategy 2: Upgrade vel CNN (replace single-seed with multi-seed)
    # Then layer pos on top
    # ============================================================
    print("\n--- STRATEGY 2: Multi-seed vel + Sub2716, then pos on top ---")

    # First: multi-seed vel + Sub2716
    for vw in [0.03, 0.05, 0.07]:
        sub_num, blend = blend_and_save([
            (vel_df, {"all": vw}),
            (base_df, {"all": 1.0 - vw}),
        ], f"{int(vw*100)}% msVel + {int((1-vw)*100)}% Sub{args.base_sub}")
        submissions.append(sub_num)

        # Layer position on top
        for pw in [0.03, 0.05]:
            sub_num2, _ = blend_and_save([
                (pos_df, {"all": pw}),
                (blend, {"all": 1.0 - pw}),
            ], f"{int(pw*100)}% pos + {int((1-pw)*100)}% x ({int(vw*100)}% msVel + Sub{args.base_sub})")
            submissions.append(sub_num2)

    # ============================================================
    # Strategy 3: Direct 3-way blend (simultaneous weights)
    # ============================================================
    print("\n--- STRATEGY 3: Direct 3-way blend ---")

    weight_configs = [
        # (vel_w, pos_w, base_w)
        (0.05, 0.05, 0.90),  # equal CNN weights
        (0.05, 0.07, 0.88),  # pos-heavy (better CV)
        (0.07, 0.05, 0.88),  # vel-heavy (better diversity)
        (0.07, 0.07, 0.86),  # more CNN
        (0.03, 0.05, 0.92),  # conservative
        (0.05, 0.03, 0.92),  # conservative vel-heavy
        (0.10, 0.05, 0.85),  # aggressive vel
        (0.05, 0.10, 0.85),  # aggressive pos
        (0.10, 0.10, 0.80),  # very aggressive
    ]

    for vw, pw, bw in weight_configs:
        sub_num, _ = blend_and_save([
            (vel_df, {"all": vw}),
            (pos_df, {"all": pw}),
            (base_df, {"all": bw}),
        ], f"{int(vw*100)}% msVel + {int(pw*100)}% pos + {int(bw*100)}% Sub{args.base_sub}")
        submissions.append(sub_num)

    # ============================================================
    # Strategy 4: Per-target optimized 3-way blend
    # Use higher weights for targets where CNN has better diversity
    # ============================================================
    print("\n--- STRATEGY 4: Per-target optimized 3-way blend ---")

    # Compute per-target diversity
    vel_div = {}
    pos_div = {}
    for t in TARGETS:
        col = f"scaled_{t}"
        vel_div[t] = np.corrcoef(vel_df[col].values, base_df[col].values)[0, 1]
        pos_div[t] = np.corrcoef(pos_df[col].values, base_df[col].values)[0, 1]

    print(f"\n  Per-target diversity with Sub{args.base_sub}:")
    for t in TARGETS:
        print(f"    {t}: vel r={vel_div[t]:.4f}, pos r={pos_div[t]:.4f}")

    # Strategy: higher weight for more diverse (lower r) targets
    # Based on known diversity: vel depth r~0.59 (great), pos depth r~0.89 (moderate)
    # Velocity CNN has best depth diversity, position CNN is overall better quality

    per_target_configs = [
        # Config name, vel_weights, pos_weights
        ("depth-heavy-vel",
         {"angle": 0.03, "depth": 0.10, "left_right": 0.05},
         {"angle": 0.05, "depth": 0.03, "left_right": 0.05}),
        ("depth-heavy-vel-v2",
         {"angle": 0.03, "depth": 0.12, "left_right": 0.05},
         {"angle": 0.05, "depth": 0.05, "left_right": 0.05}),
        ("balanced-diverse",
         {"angle": 0.05, "depth": 0.07, "left_right": 0.05},
         {"angle": 0.05, "depth": 0.05, "left_right": 0.05}),
        ("pos-quality-boost",
         {"angle": 0.03, "depth": 0.07, "left_right": 0.03},
         {"angle": 0.07, "depth": 0.05, "left_right": 0.07}),
    ]

    for config_name, vel_w, pos_w in per_target_configs:
        base_w = {t: 1.0 - vel_w[t] - pos_w[t] for t in TARGETS}

        sub_num = get_next_submission_number()
        result = base_df[["id"]].copy()
        desc_parts = []
        for t in TARGETS:
            col = f"scaled_{t}"
            result[col] = np.clip(
                vel_w[t] * vel_df[col].values +
                pos_w[t] * pos_df[col].values +
                base_w[t] * base_df[col].values,
                0.0, 1.0,
            )
            desc_parts.append(f"{t}({int(vel_w[t]*100)}v/{int(pos_w[t]*100)}p)")

        path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        result.to_csv(path, index=False)
        desc = f"PT-{config_name}: {', '.join(desc_parts)} + Sub{args.base_sub}"
        print(f"  Sub {sub_num}: {desc}")
        submissions.append(sub_num)

    # ============================================================
    # Strategy 5: CNN average as single diverse source
    # ============================================================
    print("\n--- STRATEGY 5: Averaged CNN ensemble ---")

    # Average velocity + position CNNs (equal weight)
    avg_cnn = base_df[["id"]].copy()
    for t in TARGETS:
        col = f"scaled_{t}"
        avg_cnn[col] = 0.5 * vel_df[col].values + 0.5 * pos_df[col].values

    # Diversity of averaged CNN
    print("  Avg CNN diversity vs Base:")
    compute_diversity(avg_cnn, base_df)

    for w in [0.05, 0.07, 0.10]:
        sub_num, _ = blend_and_save([
            (avg_cnn, {"all": w}),
            (base_df, {"all": 1.0 - w}),
        ], f"{int(w*100)}% avgCNN(vel+pos) + {int((1-w)*100)}% Sub{args.base_sub}")
        submissions.append(sub_num)

    # Weighted average (position gets more weight due to better CV)
    wavg_cnn = base_df[["id"]].copy()
    for t in TARGETS:
        col = f"scaled_{t}"
        wavg_cnn[col] = 0.4 * vel_df[col].values + 0.6 * pos_df[col].values

    for w in [0.05, 0.07, 0.10]:
        sub_num, _ = blend_and_save([
            (wavg_cnn, {"all": w}),
            (base_df, {"all": 1.0 - w}),
        ], f"{int(w*100)}% wAvgCNN(40v/60p) + {int((1-w)*100)}% Sub{args.base_sub}")
        submissions.append(sub_num)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {len(submissions)} submissions: Sub {min(submissions)}-{max(submissions)}")
    print(f"\nTop candidates for submission (rank by expected LB improvement):")
    print(f"  1. Direct upgrade of winning formula (replace pilot pos -> full pos)")
    print(f"  2. Layered: 5% pos on top of (5% msVel + 95% Sub2716)")
    print(f"  3. 3-way blend with per-target optimization")
    print(f"  4. Aggressive 7%+7% dual CNN")

    # Save metadata
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": ts,
        "vel_sub": args.vel_sub,
        "pos_sub": args.pos_sub,
        "base_sub": args.base_sub,
        "submissions": submissions,
        "n_submissions": len(submissions),
    }
    meta_path = OUTPUT_DIR / f"optimal_dual_cnn_blend_{ts}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")


if __name__ == "__main__":
    main()
