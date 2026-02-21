#!/usr/bin/env python3
"""
Mega Multi-Source Blend: Combine ALL diverse sources for maximum LB improvement.

Available diverse sources (depth correlation with Sub2716):
1. Multi-seed velocity CNN (Sub 3342): depth r=0.66
2. Full-run position CNN (Sub 3331): depth r=0.89
3. Biomech paper features (Sub 3388): depth r=0.92
4. Trajectory model (Sub 1507): depth r=0.61
5. Pulse features (Sub 2608): depth r=0.53 (partially in Sub2716 already)
6. Energy wave (Sub 2602): depth r=0.67 (partially in Sub2716 already)

Strategy: layer multiple diverse sources at small weights on top of Sub2716,
then also try layering on top of Sub3336 (current LB best = 0.006243).

Key insight: diverse sources compound. Even at 1-2% weight, adding sources with
r=0.5-0.6 diversity can give 0.000010-0.000015 improvement each.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import fcntl

PROJECT_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = [f"scaled_{t}" for t in TARGETS]


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


def load_sub(num):
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sub {num} not found")
    return pd.read_csv(path)


def blend(components, ids):
    """Blend multiple submissions with per-target weights.
    components: list of (df, weight_dict) where weight_dict maps target -> weight
    """
    result = pd.DataFrame({"id": ids})
    for t in TARGETS:
        col = f"scaled_{t}"
        blended = np.zeros(len(ids))
        total_w = 0
        for df, weights in components:
            w = weights.get(t, weights.get("all", 0))
            blended += w * df[col].values
            total_w += w
        assert abs(total_w - 1.0) < 1e-6, f"{t}: weights sum to {total_w}"
        result[col] = np.clip(blended, 0.0, 1.0)
    return result


def save_sub(df, desc):
    sub_num = get_next_submission_number()
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    df.to_csv(path, index=False)
    print(f"  Sub {sub_num}: {desc}")
    return sub_num


def diversity_check(df, ref_df, label):
    """Print diversity of df vs ref_df."""
    for t in TARGETS:
        col = f"scaled_{t}"
        r = np.corrcoef(df[col].values, ref_df[col].values)[0, 1]
        print(f"    {t}: r={r:.4f}")


def main():
    print("=" * 70)
    print("MEGA MULTI-SOURCE BLEND")
    print("=" * 70)

    # Load all sources
    base = load_sub(2716)       # LB 0.006343
    vel = load_sub(3342)        # Multi-seed velocity CNN
    pos = load_sub(3331)        # Full-run position CNN
    bio = load_sub(3388)        # Biomech paper features
    traj = load_sub(1507)       # Trajectory model
    pulse = load_sub(2608)      # Pulse features
    energy = load_sub(2602)     # Energy wave
    lb_best = load_sub(3336)    # Current LB best (0.006243)
    sub3294 = load_sub(3294)    # 5% vel + 95% Sub2716

    ids = base["id"].values

    print("\nSources loaded:")
    print("  Sub 2716 (base): LB 0.006343")
    print("  Sub 3342 (multi-seed vel CNN)")
    print("  Sub 3331 (position CNN)")
    print("  Sub 3388 (biomech features)")
    print("  Sub 1507 (trajectory)")
    print("  Sub 2608 (pulse)")
    print("  Sub 2602 (energy wave)")
    print("  Sub 3336 (LB best = 0.006243)")
    print("  Sub 3294 (vel+base = 0.006306)")

    submissions = []

    # =================================================================
    # STRATEGY 1: Layer on top of Sub 3336 (current LB best)
    # Sub 3336 = 5% pos + 4.75% vel + 90.25% Sub2716
    # Add new diverse sources at small weights
    # =================================================================
    print("\n--- STRATEGY 1: Layer on Sub 3336 (LB best) ---")

    # 1a: Add trajectory (depth r=0.61 vs Sub3336)
    for w in [0.01, 0.02, 0.03]:
        df = blend([(traj, {"all": w}), (lb_best, {"all": 1 - w})], ids)
        sn = save_sub(df, f"{int(w*100)}% traj + {int((1-w)*100)}% Sub3336")
        submissions.append(sn)

    # 1b: Add pulse (depth r=0.54 vs Sub3336)
    for w in [0.01, 0.02]:
        df = blend([(pulse, {"all": w}), (lb_best, {"all": 1 - w})], ids)
        sn = save_sub(df, f"{int(w*100)}% pulse + {int((1-w)*100)}% Sub3336")
        submissions.append(sn)

    # 1c: Add biomech (depth r=0.92 vs Sub3336 - less diverse but new signal)
    for w in [0.02, 0.03]:
        df = blend([(bio, {"all": w}), (lb_best, {"all": 1 - w})], ids)
        sn = save_sub(df, f"{int(w*100)}% biomech + {int((1-w)*100)}% Sub3336")
        submissions.append(sn)

    # 1d: Multi-source on top of Sub3336
    # traj + pulse + Sub3336
    df = blend([(traj, {"all": 0.02}), (pulse, {"all": 0.01}), (lb_best, {"all": 0.97})], ids)
    sn = save_sub(df, "2% traj + 1% pulse + 97% Sub3336")
    submissions.append(sn)

    # traj + biomech + Sub3336
    df = blend([(traj, {"all": 0.02}), (bio, {"all": 0.02}), (lb_best, {"all": 0.96})], ids)
    sn = save_sub(df, "2% traj + 2% biomech + 96% Sub3336")
    submissions.append(sn)

    # traj + pulse + biomech + Sub3336
    df = blend([(traj, {"all": 0.02}), (pulse, {"all": 0.01}), (bio, {"all": 0.01}), (lb_best, {"all": 0.96})], ids)
    sn = save_sub(df, "2% traj + 1% pulse + 1% biomech + 96% Sub3336")
    submissions.append(sn)

    # =================================================================
    # STRATEGY 2: Build from scratch - 5+ source blend on Sub2716
    # =================================================================
    print("\n--- STRATEGY 2: Multi-source from scratch on Sub2716 ---")

    configs = [
        # (vel, pos, traj, pulse, bio, energy, base_remaining)
        # Conservative 4-way
        (0.05, 0.05, 0.02, 0.00, 0.00, 0.00, 0.88, "5v+5p+2t+88b"),
        (0.05, 0.05, 0.02, 0.01, 0.00, 0.00, 0.87, "5v+5p+2t+1pu+87b"),
        # 5-way
        (0.05, 0.05, 0.02, 0.01, 0.01, 0.00, 0.86, "5v+5p+2t+1pu+1bio+86b"),
        (0.05, 0.05, 0.02, 0.01, 0.02, 0.00, 0.85, "5v+5p+2t+1pu+2bio+85b"),
        # 6-way mega blend
        (0.05, 0.05, 0.02, 0.01, 0.01, 0.01, 0.85, "5v+5p+2t+1pu+1bio+1en+85b"),
        (0.05, 0.05, 0.03, 0.01, 0.01, 0.01, 0.84, "5v+5p+3t+1pu+1bio+1en+84b"),
        # Trajectory-heavy (depth r=0.61, very diverse)
        (0.05, 0.05, 0.04, 0.01, 0.00, 0.00, 0.85, "5v+5p+4t+1pu+85b"),
        (0.05, 0.05, 0.05, 0.00, 0.00, 0.00, 0.85, "5v+5p+5t+85b"),
        # Velocity-heavy (depth r=0.66)
        (0.07, 0.05, 0.02, 0.01, 0.00, 0.00, 0.85, "7v+5p+2t+1pu+85b"),
        # More CNN, less base
        (0.07, 0.07, 0.02, 0.01, 0.00, 0.00, 0.83, "7v+7p+2t+1pu+83b"),
    ]

    for vw, pw, tw, puw, bw, ew, basew, desc in configs:
        components = [(vel, {"all": vw}), (pos, {"all": pw}), (base, {"all": basew})]
        if tw > 0: components.append((traj, {"all": tw}))
        if puw > 0: components.append((pulse, {"all": puw}))
        if bw > 0: components.append((bio, {"all": bw}))
        if ew > 0: components.append((energy, {"all": ew}))
        df = blend(components, ids)
        sn = save_sub(df, f"From-scratch: {desc}")
        submissions.append(sn)

    # =================================================================
    # STRATEGY 3: Per-target optimized multi-source
    # Use higher weights for targets where each source has best diversity
    # =================================================================
    print("\n--- STRATEGY 3: Per-target optimized ---")

    # Depth-diversity-maximized: heavy traj+vel on depth, light on angle
    pt_configs = [
        ("depth-max-diverse", {
            "vel": {"angle": 0.03, "depth": 0.08, "left_right": 0.05},
            "pos": {"angle": 0.05, "depth": 0.03, "left_right": 0.05},
            "traj": {"angle": 0.01, "depth": 0.05, "left_right": 0.01},
            "pulse": {"angle": 0.01, "depth": 0.02, "left_right": 0.01},
        }),
        ("balanced-5source", {
            "vel": {"angle": 0.04, "depth": 0.06, "left_right": 0.05},
            "pos": {"angle": 0.05, "depth": 0.04, "left_right": 0.05},
            "traj": {"angle": 0.01, "depth": 0.03, "left_right": 0.02},
            "pulse": {"angle": 0.01, "depth": 0.01, "left_right": 0.01},
            "bio": {"angle": 0.01, "depth": 0.01, "left_right": 0.01},
        }),
        ("lr-heavy-pos", {
            "vel": {"angle": 0.03, "depth": 0.07, "left_right": 0.03},
            "pos": {"angle": 0.05, "depth": 0.03, "left_right": 0.07},
            "traj": {"angle": 0.01, "depth": 0.03, "left_right": 0.01},
            "pulse": {"angle": 0.01, "depth": 0.02, "left_right": 0.01},
        }),
    ]

    source_map = {"vel": vel, "pos": pos, "traj": traj, "pulse": pulse, "bio": bio, "energy": energy}

    for config_name, weight_dict in pt_configs:
        sub_num = get_next_submission_number()
        result = pd.DataFrame({"id": ids})

        desc_parts = []
        for t in TARGETS:
            col = f"scaled_{t}"
            blended = np.zeros(len(ids))
            total_w = 0
            for src_name, src_weights in weight_dict.items():
                w = src_weights[t]
                blended += w * source_map[src_name][col].values
                total_w += w
            # Remaining goes to base
            base_w = 1.0 - total_w
            blended += base_w * base[col].values
            result[col] = np.clip(blended, 0.0, 1.0)

        path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        result.to_csv(path, index=False)
        print(f"  Sub {sub_num}: PT-{config_name}")
        submissions.append(sub_num)

    # =================================================================
    # STRATEGY 4: Layered approach - build up from Sub3294
    # Sub3294 = 5% vel + 95% Sub2716 (LB 0.006306)
    # =================================================================
    print("\n--- STRATEGY 4: Layer on Sub3294 ---")

    # 4a: Add pos + traj on Sub3294
    df = blend([(pos, {"all": 0.05}), (traj, {"all": 0.02}), (sub3294, {"all": 0.93})], ids)
    sn = save_sub(df, "5% pos + 2% traj + 93% Sub3294")
    submissions.append(sn)

    df = blend([(pos, {"all": 0.05}), (traj, {"all": 0.02}), (pulse, {"all": 0.01}), (sub3294, {"all": 0.92})], ids)
    sn = save_sub(df, "5% pos + 2% traj + 1% pulse + 92% Sub3294")
    submissions.append(sn)

    # 4b: Replace single-vel in Sub3294 with multi-vel, then add more
    # First build upgraded Sub3294
    upgraded_3294 = blend([(vel, {"all": 0.05}), (base, {"all": 0.95})], ids)
    # Then layer
    df = blend([(pos, {"all": 0.05}), (traj, {"all": 0.02}), (upgraded_3294, {"all": 0.93})], ids)
    sn = save_sub(df, "5% pos + 2% traj + 93% x (5% msVel + 95% Sub2716)")
    submissions.append(sn)

    df = blend([(pos, {"all": 0.05}), (traj, {"all": 0.02}), (pulse, {"all": 0.01}), (upgraded_3294, {"all": 0.92})], ids)
    sn = save_sub(df, "5% pos + 2% traj + 1% pulse + 92% x (5% msVel + 95% Sub2716)")
    submissions.append(sn)

    # =================================================================
    # Diversity check of top candidates vs LB best
    # =================================================================
    print("\n--- DIVERSITY CHECK (top candidates vs Sub3336) ---")
    for sn in submissions[:5]:
        df = load_sub(sn)
        print(f"\n  Sub {sn}:")
        diversity_check(df, lb_best, f"Sub {sn}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generated {len(submissions)} submissions: Sub {min(submissions)}-{max(submissions)}")

    print("\nRANKED CANDIDATES (by expected improvement over Sub 3336):")
    print()
    print("  HIGH CONFIDENCE (safe bets, low risk):")
    print("    - 2% traj + 98% Sub3336  (adds depth r=0.61 diversity)")
    print("    - 1% pulse + 99% Sub3336 (adds depth r=0.54 diversity)")
    print()
    print("  MODERATE CONFIDENCE (higher potential, moderate risk):")
    print("    - 2% traj + 1% pulse + 97% Sub3336 (compound diversity)")
    print("    - 5% pos + 2% traj + 93% Sub3294 (layered on proven formula)")
    print()
    print("  AMBITIOUS (highest potential, higher risk):")
    print("    - 5v+5p+2t+1pu+87b from scratch (5-source)")
    print("    - PT-depth-max-diverse (per-target optimized)")

    # Save metadata
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": ts,
        "submissions": submissions,
        "n_submissions": len(submissions),
        "strategy": "Multi-source mega blend with trajectory + pulse + biomech",
        "current_lb_best": "Sub 3336 = 0.006243",
    }
    meta_path = OUTPUT_DIR / f"mega_multi_source_blend_{ts}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {meta_path}")


if __name__ == "__main__":
    main()
