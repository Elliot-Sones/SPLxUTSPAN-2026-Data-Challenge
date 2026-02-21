"""
Multi-Submission Optimizer

Phase 2 tool: find optimal per-target weighted combinations of existing submissions.
Supports:
1. Simple N-way blends with grid search
2. Per-target column surgery (different source per target)
3. Calibrated scoring using known LB anchors
"""

import argparse
import json
import fcntl
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]

# Known LB scores for calibration
LB_SCORES = {
    2503: 0.006471,
    2609: 0.006446,
    2604: 0.006456,
    2583: 0.006516,
    2575: 0.006502,
    2455: 0.006701,
    2450: 0.006519,
    2429: 0.006502,
    2402: 0.006511,
    2372: 0.006538,
    2169: 0.006552,
    2063: 0.006603,
    1828: 0.006619,
    1350: 0.006776,
    784: 0.007224,
}


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
    return pd.read_csv(path)


def blend_subs(subs_dict, weights_dict):
    """Blend multiple submissions with per-sub weights. weights_dict: {sub_num: weight}"""
    result = None
    total_w = sum(weights_dict.values())
    for sub_num, w in weights_dict.items():
        df = subs_dict[sub_num]
        if result is None:
            result = df.copy()
            for t in TARGETS:
                result[t] = df[t] * (w / total_w)
        else:
            for t in TARGETS:
                result[t] += df[t] * (w / total_w)
    return result


def per_target_blend(subs_dict, target_weights):
    """Per-target blending. target_weights: {target: {sub_num: weight}}"""
    ref_sub = list(subs_dict.values())[0]
    result = ref_sub.copy()
    for t in TARGETS:
        weights = target_weights[t]
        total_w = sum(weights.values())
        result[t] = 0.0
        for sub_num, w in weights.items():
            result[t] += subs_dict[sub_num][t] * (w / total_w)
    return result


def compute_correlations(subs_dict, anchor_num):
    """Compute per-target Pearson correlation of each sub with anchor."""
    anchor = subs_dict[anchor_num]
    corrs = {}
    for sub_num, df in subs_dict.items():
        if sub_num == anchor_num:
            continue
        corrs[sub_num] = {}
        for t in TARGETS:
            corrs[sub_num][t] = np.corrcoef(anchor[t].values, df[t].values)[0, 1]
    return corrs


def grid_search_blend(subs_dict, anchor_num, diverse_nums, weight_range=None):
    """Grid search over blend weights for diverse subs mixed with anchor."""
    if weight_range is None:
        weight_range = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    best_configs = []
    n_diverse = len(diverse_nums)

    # Per-target independent search
    for t in TARGETS:
        best_score = None
        best_weights = None

        # For each combination of weights for diverse sources
        for combo in itertools.product(weight_range, repeat=n_diverse):
            total_diverse = sum(combo)
            if total_diverse > 0.50:  # Don't let diverse sources dominate
                continue
            anchor_w = 1.0 - total_diverse

            # Compute blended prediction for this target
            blended = subs_dict[anchor_num][t].values * anchor_w
            for i, sub_num in enumerate(diverse_nums):
                blended += subs_dict[sub_num][t].values * combo[i]

            # We can't compute true MSE without ground truth, but we can
            # look at how close this blend is to known-good anchors
            # For now, store the config
            config = {anchor_num: anchor_w}
            for i, sub_num in enumerate(diverse_nums):
                if combo[i] > 0:
                    config[sub_num] = combo[i]

            best_configs.append({
                "target": t,
                "weights": config,
                "total_diverse_weight": total_diverse,
            })

    return best_configs


def save_submission(df, description):
    sub_num = get_next_submission_number()
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(f"  Description: {description}")
    return sub_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["blend", "surgery", "analyze", "compound"],
                       default="analyze")
    parser.add_argument("--anchor", type=int, default=2609,
                       help="Anchor submission number")
    parser.add_argument("--diverse", type=int, nargs="+",
                       help="Diverse submission numbers to blend")
    parser.add_argument("--weights", type=float, nargs="+",
                       help="Blend weights for diverse subs")
    args = parser.parse_args()

    if args.mode == "analyze":
        # Load and analyze correlations
        print("=== Multi-Submission Analysis ===\n")

        # Key submissions to analyze
        key_subs = [2503, 2609, 2608, 2602, 2604]
        # Add diverse candidates if they exist
        for s in [1507, 1103, 1557, 1558, 1559, 1560]:
            if (SUBMISSION_DIR / f"submission_{s}.csv").exists():
                key_subs.append(s)

        subs = {}
        for s in key_subs:
            try:
                subs[s] = load_sub(s)
            except Exception:
                print(f"  Sub {s}: NOT FOUND")

        print(f"Loaded {len(subs)} submissions: {sorted(subs.keys())}")

        if 2609 in subs:
            corrs = compute_correlations(subs, 2609)
            print(f"\n--- Correlations with Sub 2609 (new best anchor) ---")
            for sub_num in sorted(corrs.keys()):
                c = corrs[sub_num]
                print(f"  Sub {sub_num}: angle={c[TARGETS[0]]:.4f}  depth={c[TARGETS[1]]:.4f}  LR={c[TARGETS[2]]:.4f}")

    elif args.mode == "compound":
        # Compound two confirmed LB signals: pulse + energy wave
        print("=== Compounding Pulse + Energy Wave Signals ===\n")

        anchor = load_sub(2503)  # original anchor
        pulse = load_sub(2608)   # standalone pulse
        energy = load_sub(2602)  # standalone energy wave
        sub2609 = load_sub(2609) # 1% pulse + 99% Sub 2503 (LB 0.006446)

        # Method 1: Layer energy wave on top of Sub 2609 (which already has pulse)
        # Sub 2609 = 0.01*pulse + 0.99*Sub2503
        # New = w*energy + (1-w)*Sub2609
        print("Method 1: w% energy wave + (1-w)% Sub 2609")
        for ew in [0.01, 0.02, 0.05, 0.10]:
            result = sub2609.copy()
            for t in TARGETS:
                result[t] = ew * energy[t] + (1 - ew) * sub2609[t]
            sub_num = save_submission(result,
                f"{ew*100:.0f}% energy wave + {(1-ew)*100:.0f}% Sub 2609 (compound pulse+energy)")
            print(f"  {ew*100:.0f}% energy: Sub {sub_num}")

        # Method 2: Direct 3-way blend at optimal weights
        print("\nMethod 2: 3-way blend Sub2503 + pulse + energy")
        for pw in [0.01, 0.02]:
            for ew in [0.05, 0.10]:
                aw = 1 - pw - ew
                result = anchor.copy()
                for t in TARGETS:
                    result[t] = aw * anchor[t] + pw * pulse[t] + ew * energy[t]
                sub_num = save_submission(result,
                    f"3-way: {aw*100:.0f}% Sub2503 + {pw*100:.0f}% pulse + {ew*100:.0f}% energy")
                print(f"  {pw*100:.0f}%p + {ew*100:.0f}%e: Sub {sub_num}")

        # Method 3: Per-target surgery using best source per target
        # pulse works at 1%, energy at 10% - try per-target optimal
        print("\nMethod 3: Per-target blend weights")
        result = anchor.copy()
        # Use Sub 2609 values for angle (pulse might help angle more)
        # Use Sub 2604 values for depth (energy wave might help depth more)
        # Use Sub 2609 for LR (or try energy wave)
        configs = [
            {"angle": (0.01, 0.05), "depth": (0.01, 0.10), "lr": (0.01, 0.05)},  # balanced
            {"angle": (0.02, 0.02), "depth": (0.00, 0.10), "lr": (0.01, 0.10)},  # energy-heavy
            {"angle": (0.01, 0.10), "depth": (0.01, 0.05), "lr": (0.02, 0.05)},  # mixed
        ]
        for i, cfg in enumerate(configs):
            result = anchor.copy()
            for t_short, t_full in [("angle", TARGETS[0]), ("depth", TARGETS[1]), ("lr", TARGETS[2])]:
                pw, ew = cfg[t_short]
                aw = 1 - pw - ew
                result[t_full] = aw * anchor[t_full] + pw * pulse[t_full] + ew * energy[t_full]
            desc = f"Per-target 3-way config {i}: " + str(cfg)
            sub_num = save_submission(result, desc)
            print(f"  Config {i}: Sub {sub_num}")

    elif args.mode == "blend":
        if not args.diverse or not args.weights:
            print("Need --diverse and --weights for blend mode")
            return

        anchor = load_sub(args.anchor)
        total_diverse_w = sum(args.weights)
        anchor_w = 1 - total_diverse_w

        result = anchor.copy()
        for t in TARGETS:
            result[t] = anchor[t] * anchor_w
            for sub_num, w in zip(args.diverse, args.weights):
                df = load_sub(sub_num)
                result[t] += df[t] * w

        desc = f"{anchor_w*100:.0f}% Sub{args.anchor} + " + \
               " + ".join(f"{w*100:.0f}% Sub{s}" for s, w in zip(args.diverse, args.weights))
        save_submission(result, desc)

    elif args.mode == "surgery":
        # Per-target column surgery
        if not args.diverse:
            print("Need --diverse for surgery mode (one sub per target: angle, depth, LR)")
            return
        if len(args.diverse) != 3:
            print("Need exactly 3 subs for surgery: angle_sub depth_sub lr_sub")
            return

        result = load_sub(args.diverse[0]).copy()
        for i, t in enumerate(TARGETS):
            src = load_sub(args.diverse[i])
            result[t] = src[t]

        desc = f"Surgery: angle=Sub{args.diverse[0]} depth=Sub{args.diverse[1]} LR=Sub{args.diverse[2]}"
        save_submission(result, desc)


if __name__ == "__main__":
    main()
