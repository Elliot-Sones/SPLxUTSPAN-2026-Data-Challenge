"""
Optimal Blend Search v2

Systematic per-target grid search to find the best blend of diverse
submissions with Sub 3190 (current best anchor, LB 0.006331).

Key insight: different model families at small weights (2-5%) compound
to large diversity gains. This script exhaustively searches per-target
weight allocations across 10 diverse source submissions.

No model training required - purely combines existing CSV predictions.
"""

import time
import itertools
import numpy as np
import pandas as pd
import fcntl
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_SHORT = {"scaled_angle": "angle", "scaled_depth": "depth", "scaled_left_right": "LR"}

ANCHOR_NUM = 3190


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ================================================================
# DIVERSE SOURCE DEFINITIONS
# ================================================================
# Each source has: sub number, short name, known quality info
# quality_penalty: 1.0 = good standalone, lower = weaker standalone
DIVERSE_SOURCES = {
    "bigru":       {"sub": 2870, "quality": 0.7,  "desc": "BiGRU temporal model"},
    "pure_vel":    {"sub": 3209, "quality": 0.6,  "desc": "Pure velocity pipeline"},
    "val_vel":     {"sub": 3204, "quality": 0.7,  "desc": "Validated velocity pipeline"},
    "pulse":       {"sub": 2608, "quality": 0.5,  "desc": "Pulse standalone"},
    "energy":      {"sub": 2602, "quality": 0.6,  "desc": "Energy wave standalone"},
    "xgboost":     {"sub": 2679, "quality": 0.8,  "desc": "XGBoost tree model"},
    "lgbm":        {"sub": 2704, "quality": 0.8,  "desc": "LightGBM tree model"},
    "knn":         {"sub": 2784, "quality": 0.4,  "desc": "kNN bootstrap"},
    "cnn1557":     {"sub": 1557, "quality": 0.7,  "desc": "Temporal CNN"},
    "trajectory":  {"sub": 1507, "quality": 0.5,  "desc": "Trajectory depth model"},
}


def main():
    t0 = time.time()
    print("=" * 80)
    print("OPTIMAL BLEND SEARCH v2")
    print("Anchor: Sub 3190 (LB 0.006331)")
    print(f"Diverse sources: {len(DIVERSE_SOURCES)}")
    print("=" * 80)

    # ================================================================
    # 1. LOAD ALL SUBMISSIONS
    # ================================================================
    print("\n--- Loading submissions ---")

    anchor = pd.read_csv(SUBMISSION_DIR / f"submission_{ANCHOR_NUM}.csv")
    print(f"  Anchor Sub {ANCHOR_NUM}: {len(anchor)} rows")

    sources = {}
    for name, info in DIVERSE_SOURCES.items():
        path = SUBMISSION_DIR / f"submission_{info['sub']}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if len(df) == len(anchor):
                sources[name] = df
                print(f"  Loaded {name:12s} (Sub {info['sub']}): {len(df)} rows")
            else:
                print(f"  SKIP {name:12s} (Sub {info['sub']}): row count mismatch ({len(df)} vs {len(anchor)})")
        else:
            print(f"  SKIP {name:12s} (Sub {info['sub']}): file not found")

    print(f"\n  Total usable sources: {len(sources)}")
    source_names = sorted(sources.keys())

    # ================================================================
    # 2. COMPUTE PER-TARGET CORRELATION MATRIX
    # ================================================================
    print("\n--- Per-target correlation with anchor ---")
    print(f"  {'Source':14s} | {'angle':>8s} | {'depth':>8s} | {'LR':>8s} | {'mean_div':>10s}")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    corr_with_anchor = {}  # {name: {target: r}}
    for name in source_names:
        corr_with_anchor[name] = {}
        divs = []
        for target in TARGETS:
            r = np.corrcoef(
                anchor[target].values,
                sources[name][target].values
            )[0, 1]
            corr_with_anchor[name][target] = r
            divs.append(1 - r)
        mean_div = np.mean(divs)
        ts = TARGET_SHORT
        print(f"  {name:14s} | {corr_with_anchor[name][TARGETS[0]]:8.4f} | "
              f"{corr_with_anchor[name][TARGETS[1]]:8.4f} | "
              f"{corr_with_anchor[name][TARGETS[2]]:8.4f} | {mean_div:10.4f}")

    # ================================================================
    # 3. INTER-SOURCE CORRELATION (detect redundancy)
    # ================================================================
    print("\n--- Inter-source correlation (mean across targets) ---")
    print(f"  {'':14s}", end="")
    for n2 in source_names:
        print(f" | {n2[:6]:>6s}", end="")
    print()
    for n1 in source_names:
        print(f"  {n1:14s}", end="")
        for n2 in source_names:
            if n1 == n2:
                print(f" | {'1.00':>6s}", end="")
            else:
                rs = []
                for target in TARGETS:
                    r = np.corrcoef(
                        sources[n1][target].values,
                        sources[n2][target].values
                    )[0, 1]
                    rs.append(r)
                print(f" | {np.mean(rs):6.3f}", end="")
        print()

    # ================================================================
    # 4. PER-TARGET GRID SEARCH
    # ================================================================
    print("\n" + "=" * 80)
    print("PER-TARGET GRID SEARCH")
    print("=" * 80)

    # Weight grid for each source per target
    WEIGHT_OPTIONS = [0.0, 0.01, 0.02, 0.03, 0.05]
    # Maximum total diverse weight per target
    MAX_TOTAL_WEIGHT = 0.15

    # For each target, find best source combinations
    # Score = diversity_contribution * quality_weight
    # diversity_contribution = weight * (1 - corr_with_anchor)
    # quality_weight = source quality factor

    best_per_target = {}  # {target: list of (score, config_dict)}

    for target in TARGETS:
        tname = TARGET_SHORT[target]
        print(f"\n--- Target: {tname} ---")

        # Rank sources by diversity for this target
        ranked = sorted(source_names,
                        key=lambda n: corr_with_anchor[n][target])
        print(f"  Most diverse sources (lowest r with anchor):")
        for i, name in enumerate(ranked[:5]):
            r = corr_with_anchor[name][target]
            q = DIVERSE_SOURCES[name]["quality"]
            print(f"    {i+1}. {name:14s}: r={r:.4f}, div={1-r:.4f}, quality={q:.1f}")

        # For efficiency: only consider top-6 most diverse sources per target
        # (bottom sources have r>0.95, adding them is just noise)
        top_sources = ranked[:6]

        # Generate all weight combinations for top sources
        # Each source gets a weight from WEIGHT_OPTIONS
        configs = []
        total_combos = len(WEIGHT_OPTIONS) ** len(top_sources)
        print(f"  Searching {total_combos} weight combinations for top {len(top_sources)} sources...")

        for weights in itertools.product(WEIGHT_OPTIONS, repeat=len(top_sources)):
            total_w = sum(weights)
            if total_w == 0:
                continue  # skip all-zero (that's just the anchor)
            if total_w > MAX_TOTAL_WEIGHT:
                continue  # too much diverse weight

            # Compute diversity score for this config
            # Higher = more diverse signal injected, quality-adjusted
            div_score = 0.0
            config = {}
            for src_name, w in zip(top_sources, weights):
                if w > 0:
                    r = corr_with_anchor[src_name][target]
                    q = DIVERSE_SOURCES[src_name]["quality"]
                    # Diversity contribution = weight * (1 - r) * quality
                    div_score += w * (1 - r) * q
                    config[src_name] = w

            # Penalty for too many sources (each adds noise)
            n_sources = len(config)
            noise_penalty = 1.0 - 0.02 * max(0, n_sources - 2)  # small penalty for >2 sources

            # Penalty for high total weight (diminishing returns + overfit risk)
            weight_penalty = 1.0 - 0.5 * (total_w / MAX_TOTAL_WEIGHT) ** 2

            final_score = div_score * noise_penalty * weight_penalty
            configs.append((final_score, total_w, config))

        # Sort by score descending
        configs.sort(key=lambda x: x[0], reverse=True)

        # Show top 10 for this target
        print(f"  Top 10 configs for {tname}:")
        for rank, (score, total_w, config) in enumerate(configs[:10]):
            parts = [f"{n}={w*100:.0f}%" for n, w in sorted(config.items())]
            anchor_w = (1 - total_w) * 100
            print(f"    {rank+1:2d}. score={score:.6f}, anchor={anchor_w:.0f}%, "
                  f"diverse={total_w*100:.0f}%: {', '.join(parts)}")

        best_per_target[target] = configs[:50]  # keep top 50 per target

    # ================================================================
    # 5. COMBINE BEST PER-TARGET CONFIGS INTO FULL SUBMISSIONS
    # ================================================================
    print("\n" + "=" * 80)
    print("COMBINING BEST PER-TARGET CONFIGS")
    print("=" * 80)

    # Strategy A: Take top-K from each target and combine all permutations
    # Use top 5 per target = 5^3 = 125 combos, then rank globally
    TOP_K = 8

    full_configs = []
    for i_angle, (score_a, tw_a, cfg_a) in enumerate(best_per_target[TARGETS[0]][:TOP_K]):
        for i_depth, (score_d, tw_d, cfg_d) in enumerate(best_per_target[TARGETS[1]][:TOP_K]):
            for i_lr, (score_l, tw_l, cfg_l) in enumerate(best_per_target[TARGETS[2]][:TOP_K]):
                combined_score = score_a + score_d + score_l
                total_diverse = tw_a + tw_d + tw_l
                full_configs.append({
                    "combined_score": combined_score,
                    "total_diverse_weight": total_diverse,
                    "angle_config": cfg_a,
                    "angle_anchor_w": 1 - tw_a,
                    "depth_config": cfg_d,
                    "depth_anchor_w": 1 - tw_d,
                    "lr_config": cfg_l,
                    "lr_anchor_w": 1 - tw_l,
                    "rank_a": i_angle,
                    "rank_d": i_depth,
                    "rank_l": i_lr,
                })

    # Sort by combined score
    full_configs.sort(key=lambda x: x["combined_score"], reverse=True)

    # Deduplicate: remove configs that produce identical blends
    seen_fingerprints = set()
    unique_configs = []
    for cfg in full_configs:
        # Fingerprint = frozenset of all (target, source, weight) tuples
        parts = []
        for target, key in [(TARGETS[0], "angle_config"),
                            (TARGETS[1], "depth_config"),
                            (TARGETS[2], "lr_config")]:
            for src, w in sorted(cfg[key].items()):
                parts.append((target, src, w))
        fp = frozenset(parts)
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            unique_configs.append(cfg)

    print(f"  Total combos evaluated: {len(full_configs)}")
    print(f"  Unique configs: {len(unique_configs)}")

    # ================================================================
    # Strategy B: Add some hand-crafted "smart" blends based on memory
    # ================================================================
    # From MEMORY.md: Sub 2622 was 92% + 1% pulse + 5% energy + 2% CNN1557
    # That pattern worked well. Let's try variations.

    hand_crafted = []

    # B1: Known good pattern adapted for Sub 3190
    hand_crafted.append({
        "desc": "Known-good-pattern (1% pulse + 5% energy + 2% CNN)",
        "angle_config": {"pulse": 0.01, "energy": 0.05, "cnn1557": 0.02},
        "depth_config": {"pulse": 0.01, "energy": 0.05, "cnn1557": 0.02},
        "lr_config": {"pulse": 0.01, "energy": 0.05, "cnn1557": 0.02},
    })

    # B2: Tree models (good quality, moderate diversity)
    hand_crafted.append({
        "desc": "Tree-ensemble (2% XGB + 2% LGBM)",
        "angle_config": {"xgboost": 0.02, "lgbm": 0.02},
        "depth_config": {"xgboost": 0.02, "lgbm": 0.02},
        "lr_config": {"xgboost": 0.02, "lgbm": 0.02},
    })

    # B3: Max depth diversity
    hand_crafted.append({
        "desc": "Depth-focused (5% pulse + 3% trajectory depth)",
        "angle_config": {},
        "depth_config": {"pulse": 0.05, "trajectory": 0.03},
        "lr_config": {},
    })

    # B4: Max LR diversity
    hand_crafted.append({
        "desc": "LR-focused (5% kNN + 3% pure_vel)",
        "angle_config": {},
        "depth_config": {},
        "lr_config": {"knn": 0.05, "pure_vel": 0.03},
    })

    # B5: Best-of-each diverse
    hand_crafted.append({
        "desc": "Best-diverse-per-target (depth=pulse+bigru, LR=knn+pure_vel)",
        "angle_config": {"bigru": 0.02},
        "depth_config": {"pulse": 0.03, "bigru": 0.03},
        "lr_config": {"knn": 0.03, "pure_vel": 0.03},
    })

    # B6: All-sources-tiny (1% each of everything)
    hand_crafted.append({
        "desc": "All-sources-1pct",
        "angle_config": {n: 0.01 for n in source_names},
        "depth_config": {n: 0.01 for n in source_names},
        "lr_config": {n: 0.01 for n in source_names},
    })

    # B7: BiGRU-heavy for depth (r=0.505 is very diverse)
    hand_crafted.append({
        "desc": "BiGRU-depth-heavy (5% bigru depth, 2% bigru LR)",
        "angle_config": {},
        "depth_config": {"bigru": 0.05},
        "lr_config": {"bigru": 0.02},
    })

    # B8: Velocity models only
    hand_crafted.append({
        "desc": "Velocity-trio (2% pure_vel + 2% val_vel + 2% bigru)",
        "angle_config": {"pure_vel": 0.02, "val_vel": 0.02, "bigru": 0.02},
        "depth_config": {"pure_vel": 0.02, "val_vel": 0.02, "bigru": 0.02},
        "lr_config": {"pure_vel": 0.02, "val_vel": 0.02, "bigru": 0.02},
    })

    # B9: Conservative 2% total
    hand_crafted.append({
        "desc": "Ultra-conservative (1% energy + 1% bigru, depth+LR only)",
        "angle_config": {},
        "depth_config": {"energy": 0.01, "bigru": 0.01},
        "lr_config": {"energy": 0.01, "bigru": 0.01},
    })

    # B10: Aggressive 15% with quality weighting
    hand_crafted.append({
        "desc": "Aggressive-diverse-15pct",
        "angle_config": {"bigru": 0.03, "energy": 0.02},
        "depth_config": {"pulse": 0.05, "bigru": 0.03, "energy": 0.02, "trajectory": 0.02},
        "lr_config": {"knn": 0.05, "pure_vel": 0.03, "bigru": 0.02},
    })

    # ================================================================
    # 6. GENERATE SUBMISSIONS
    # ================================================================
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION FILES")
    print("=" * 80)

    all_submissions = []

    # Generate top 10 from grid search
    n_grid = min(10, len(unique_configs))
    print(f"\n--- Grid search top {n_grid} ---")
    for rank, cfg in enumerate(unique_configs[:n_grid]):
        sub_num = get_next_submission_number()
        blended = anchor.copy()

        desc_parts = []
        for target, cfg_key, aw_key in [
            (TARGETS[0], "angle_config", "angle_anchor_w"),
            (TARGETS[1], "depth_config", "depth_anchor_w"),
            (TARGETS[2], "lr_config", "lr_anchor_w"),
        ]:
            tname = TARGET_SHORT[target]
            target_cfg = cfg[cfg_key]
            anchor_w = cfg[aw_key]

            if target_cfg:
                # Blend anchor + diverse sources
                val = anchor_w * anchor[target].values
                for src_name, w in target_cfg.items():
                    val += w * sources[src_name][target].values
                blended[target] = val
                src_desc = "+".join(f"{w*100:.0f}%{n}" for n, w in sorted(target_cfg.items()))
                desc_parts.append(f"{tname}:{anchor_w*100:.0f}%anch+{src_desc}")
            else:
                desc_parts.append(f"{tname}:100%anch")

        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        desc = " | ".join(desc_parts)
        print(f"  Sub {sub_num} [grid#{rank+1}]: score={cfg['combined_score']:.6f}")
        print(f"    {desc}")
        all_submissions.append({
            "sub_num": sub_num,
            "type": f"grid#{rank+1}",
            "score": cfg["combined_score"],
            "desc": desc,
            "config": cfg,
        })

    # Generate hand-crafted blends
    print(f"\n--- Hand-crafted blends ({len(hand_crafted)}) ---")
    for hc in hand_crafted:
        sub_num = get_next_submission_number()
        blended = anchor.copy()

        for target, cfg_key in [
            (TARGETS[0], "angle_config"),
            (TARGETS[1], "depth_config"),
            (TARGETS[2], "lr_config"),
        ]:
            target_cfg = hc[cfg_key]
            total_w = sum(target_cfg.values())
            if total_w > 0:
                anchor_w = 1 - total_w
                val = anchor_w * anchor[target].values
                for src_name, w in target_cfg.items():
                    if src_name in sources:
                        val += w * sources[src_name][target].values
                    else:
                        # Source missing, give weight back to anchor
                        val += w * anchor[target].values
                blended[target] = val

        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {hc['desc']}")
        all_submissions.append({
            "sub_num": sub_num,
            "type": "handcrafted",
            "score": 0,  # no grid score for handcrafted
            "desc": hc["desc"],
        })

    # ================================================================
    # 7. DETAILED ANALYSIS OF ALL GENERATED SUBMISSIONS
    # ================================================================
    print("\n" + "=" * 80)
    print("SUBMISSION ANALYSIS")
    print("=" * 80)

    print(f"\n  {'Sub':>6s} | {'Type':>12s} | {'angle_r':>8s} | {'depth_r':>8s} | {'LR_r':>8s} | "
          f"{'mean_div':>10s} | {'max_delta':>10s} | Description")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*40}")

    for entry in all_submissions:
        sub = pd.read_csv(SUBMISSION_DIR / f"submission_{entry['sub_num']}.csv")
        rs = {}
        max_delta = 0
        for target in TARGETS:
            r = np.corrcoef(anchor[target].values, sub[target].values)[0, 1]
            rs[target] = r
            delta = np.max(np.abs(anchor[target].values - sub[target].values))
            max_delta = max(max_delta, delta)

        mean_div = np.mean([1 - r for r in rs.values()])
        desc_short = entry["desc"][:40] if len(entry.get("desc", "")) > 40 else entry.get("desc", "")

        print(f"  {entry['sub_num']:6d} | {entry['type']:>12s} | "
              f"{rs[TARGETS[0]]:8.5f} | {rs[TARGETS[1]]:8.5f} | {rs[TARGETS[2]]:8.5f} | "
              f"{mean_div:10.6f} | {max_delta:10.6f} | {desc_short}")

    # ================================================================
    # 8. RANKED SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 80)
    print("RANKED SUMMARY (by expected improvement potential)")
    print("=" * 80)
    print("  Scoring: diversity from anchor (higher = more potential to improve)")
    print("  But too much diversity = noise. Sweet spot is 0.001-0.010 mean_div.\n")

    # Re-analyze with scoring
    scored = []
    for entry in all_submissions:
        sub = pd.read_csv(SUBMISSION_DIR / f"submission_{entry['sub_num']}.csv")
        divs = []
        for target in TARGETS:
            r = np.corrcoef(anchor[target].values, sub[target].values)[0, 1]
            divs.append(1 - r)
        mean_div = np.mean(divs)

        # Expected improvement heuristic:
        # Small diversity = likely improvement (proven pattern)
        # Too much diversity = likely hurt (Sub 2455 lesson: 50/50 = bad)
        # Sweet spot: 0.001-0.005 mean_div
        if mean_div < 0.0001:
            expected = 0  # basically identical to anchor
        elif mean_div < 0.003:
            expected = mean_div * 1000  # linear in sweet spot
        elif mean_div < 0.010:
            expected = 3.0 - (mean_div - 0.003) * 200  # declining
        else:
            expected = max(0, 1.0 - (mean_div - 0.01) * 100)  # risky territory

        scored.append({
            **entry,
            "mean_div": mean_div,
            "expected_improvement": expected,
        })

    scored.sort(key=lambda x: x["expected_improvement"], reverse=True)

    print(f"  {'Rank':>4s} | {'Sub':>6s} | {'Type':>12s} | {'mean_div':>10s} | {'Exp.Impr':>10s} | Description")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*50}")
    for rank, entry in enumerate(scored):
        desc_short = entry.get("desc", "")[:50]
        print(f"  {rank+1:4d} | {entry['sub_num']:6d} | {entry['type']:>12s} | "
              f"{entry['mean_div']:10.6f} | {entry['expected_improvement']:10.4f} | {desc_short}")

    # ================================================================
    # 9. TOP PICKS FOR LB TESTING
    # ================================================================
    print("\n" + "=" * 80)
    print("TOP PICKS FOR LB TESTING (ordered by priority)")
    print("=" * 80)

    # Pick top 5 diverse recommendations
    top_picks = scored[:5]
    for i, entry in enumerate(top_picks):
        print(f"\n  Priority {i+1}: Sub {entry['sub_num']}")
        print(f"    Type: {entry['type']}")
        print(f"    Description: {entry.get('desc', 'N/A')}")
        print(f"    Mean diversity: {entry['mean_div']:.6f}")
        print(f"    Expected improvement score: {entry['expected_improvement']:.4f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total submissions generated: {len(all_submissions)}")
    print(f"Submission numbers: {[e['sub_num'] for e in all_submissions]}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
