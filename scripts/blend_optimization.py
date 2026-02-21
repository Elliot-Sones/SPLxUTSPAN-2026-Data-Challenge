"""
Blend Optimization: Exhaustive search over top submissions.

Tasks:
A) Three-way combine: Cauchy + multiframe + Sub2063
B) Trajectory distance blends with Sub 2169
C) Fine-grid pairwise and three-way blends among top 6 submissions (5% steps)
D) Diversity analysis and smart combination
"""

import time
import numpy as np
import pandas as pd
import fcntl
from pathlib import Path
from itertools import combinations

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
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
    return pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")


def blend_subs(subs_weights):
    """Blend multiple submission DataFrames with given weights.
    subs_weights: list of (df, weight) tuples. Weights must sum to 1.
    """
    result = subs_weights[0][0].copy()
    for col in TARGET_COLS:
        val = np.zeros(len(result))
        for df, w in subs_weights:
            val += w * df[col].values
        result[col] = val
    return result


def per_target_correlation(sub_a, sub_b):
    """Per-target Pearson correlation between two submissions."""
    corrs = {}
    for col in TARGET_COLS:
        corrs[col] = np.corrcoef(sub_a[col].values, sub_b[col].values)[0, 1]
    return corrs


def mean_pred_diff(sub_a, sub_b):
    """Mean absolute difference per target between two submissions."""
    diffs = {}
    for col in TARGET_COLS:
        diffs[col] = np.mean(np.abs(sub_a[col].values - sub_b[col].values))
    return diffs


def save_blend(blended_df, desc):
    """Save a blended submission and return (sub_num, desc)."""
    sub_num = get_next_submission_number()
    blended_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    return sub_num, desc


def main():
    t0 = time.time()
    print("=" * 70)
    print("BLEND OPTIMIZATION: EXHAUSTIVE SEARCH")
    print("=" * 70)

    # ================================================================
    # LOAD ALL KEY SUBMISSIONS
    # ================================================================
    print("\nLoading submissions...")

    subs = {}
    sub_info = {
        2169: "LB 0.006552 - 30% 3f + 70% Sub2063 (BEST)",
        2166: "LB 0.006584 - 30% multiframe 5f + 70% Sub2063",
        2194: "LB 0.006589 - 10% Cauchy + 90% Sub2063",
        2063: "LB 0.006603 - 50/50 Sub1828 + Sub2020",
        2152: "LB 0.006604",
        2151: "LB 0.006605",
        2191: "LB 0.006681 - Cauchy standalone",
        2163: "Multiframe 5f standalone",
        1828: "LB 0.006619 - angle fix",
        2020: "LB 0.006619 - joint angles + Sub784",
        784:  "LB 0.007224 - global ensemble baseline",
        1507: "Trajectory distance standalone",
        1350: "LB 0.006776 - per-example V1",
        1640: "LB 0.006698 - 10% LASSO + 90% Sub1350",
    }

    for num in sub_info:
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        if path.exists():
            subs[num] = load_sub(num)
            print(f"  Sub {num}: {sub_info[num]} - loaded ({len(subs[num])} rows)")
        else:
            print(f"  Sub {num}: {sub_info[num]} - NOT FOUND")

    # ================================================================
    # EXTRACT STANDALONE 3-FRAME FROM Sub2169 AND Sub2063
    # ================================================================
    # Sub 2169 = 0.30 * 3frame + 0.70 * Sub2063
    # => 3frame = (Sub2169 - 0.70 * Sub2063) / 0.30
    print("\n  Extracting standalone 3-frame from Sub 2169 and Sub 2063...")
    standalone_3f = subs[2169].copy()
    for col in TARGET_COLS:
        standalone_3f[col] = (subs[2169][col].values - 0.70 * subs[2063][col].values) / 0.30
    subs['3f_standalone'] = standalone_3f
    sub_info['3f_standalone'] = "Extracted standalone 3-frame ensemble"
    print("  Done - extracted standalone 3-frame predictions")

    # Verify extraction: reblend should match Sub 2169 exactly
    reblend = blend_subs([(standalone_3f, 0.30), (subs[2063], 0.70)])
    for col in TARGET_COLS:
        max_diff = np.max(np.abs(reblend[col].values - subs[2169][col].values))
        print(f"    Verification {col}: max diff = {max_diff:.2e}")

    # Similarly extract Cauchy standalone from Sub 2194
    # Sub 2194 = 0.10 * Cauchy + 0.90 * Sub2063
    # But we already have Sub 2191 as Cauchy standalone, verify:
    if 2191 in subs:
        reblend_cauchy = blend_subs([(subs[2191], 0.10), (subs[2063], 0.90)])
        for col in TARGET_COLS:
            max_diff = np.max(np.abs(reblend_cauchy[col].values - subs[2194][col].values))
        print(f"    Cauchy reblend verification: max diff = {max_diff:.2e}")

    # ================================================================
    # DIVERSITY ANALYSIS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("DIVERSITY ANALYSIS (correlation with Sub 2169 - current best)")
    print(f"{'=' * 70}")

    # Compare all subs with Sub 2169
    diverse_sources = [2063, 2166, 2194, 2191, 2163, 2152, 2151,
                       1828, 2020, 784, 1507, 1350, 1640, '3f_standalone']
    for num in diverse_sources:
        if num not in subs:
            continue
        corrs = per_target_correlation(subs[2169], subs[num])
        diffs = mean_pred_diff(subs[2169], subs[num])
        label = f"Sub {num}" if isinstance(num, int) else num
        info = sub_info.get(num, "")
        print(f"  {label:20s}: angle r={corrs['scaled_angle']:.4f} diff={diffs['scaled_angle']:.6f}, "
              f"depth r={corrs['scaled_depth']:.4f} diff={diffs['scaled_depth']:.6f}, "
              f"LR r={corrs['scaled_left_right']:.4f} diff={diffs['scaled_left_right']:.6f}")

    # ================================================================
    # TASK A: THREE-WAY COMBINE
    # ================================================================
    print(f"\n{'=' * 70}")
    print("TASK A: THREE-WAY COMBINE (Cauchy + multiframe 3f + Sub2063)")
    print(f"{'=' * 70}")

    task_a_results = []

    # Cauchy standalone = Sub 2191
    # Multiframe 3f standalone = extracted above
    # Sub 2063 = base
    cauchy = subs[2191]
    mf3f = subs['3f_standalone']
    base = subs[2063]

    # Grid search: cauchy_w from 0.05-0.20, mf3f_w from 0.15-0.40, rest = Sub2063
    for cw in [0.05, 0.10, 0.15, 0.20]:
        for mw in [0.15, 0.20, 0.25, 0.30, 0.35]:
            bw = 1.0 - cw - mw
            if bw < 0.40:
                continue  # Need at least 40% base
            blended = blend_subs([(cauchy, cw), (mf3f, mw), (base, bw)])
            # Compute correlation with Sub 2169 per target
            corrs = per_target_correlation(blended, subs[2169])
            mean_corr = np.mean(list(corrs.values()))
            diffs = mean_pred_diff(blended, subs[2169])
            mean_diff = np.mean(list(diffs.values()))
            task_a_results.append({
                'cauchy_w': cw, 'mf3f_w': mw, 'base_w': bw,
                'corr_angle': corrs['scaled_angle'],
                'corr_depth': corrs['scaled_depth'],
                'corr_lr': corrs['scaled_left_right'],
                'mean_corr': mean_corr,
                'mean_diff': mean_diff,
                'desc': f"{cw:.0%} Cauchy + {mw:.0%} 3f + {bw:.0%} Sub2063"
            })

    # Sort by mean_diff (lower is closer to Sub 2169 which is our best)
    task_a_results.sort(key=lambda x: x['mean_diff'])
    print("\n  Results sorted by distance from Sub 2169 (closest first):")
    print(f"  {'Config':<40s} {'r_angle':>8s} {'r_depth':>8s} {'r_LR':>8s} {'mean_r':>8s} {'mean_diff':>10s}")
    for r in task_a_results[:15]:
        print(f"  {r['desc']:<40s} {r['corr_angle']:8.4f} {r['corr_depth']:8.4f} "
              f"{r['corr_lr']:8.4f} {r['mean_corr']:8.4f} {r['mean_diff']:10.6f}")

    # Save top candidates
    saved_a = []
    # Save: closest to Sub 2169, most different from Sub 2169, and a few intermediate
    # Closest 3
    for r in task_a_results[:3]:
        blended = blend_subs([(cauchy, r['cauchy_w']), (mf3f, r['mf3f_w']), (base, r['base_w'])])
        num, desc = save_blend(blended, f"Three-way: {r['desc']}")
        saved_a.append((num, desc, r))
        print(f"\n  Saved Sub {num}: {desc}")

    # Most different (bottom 2)
    for r in task_a_results[-2:]:
        blended = blend_subs([(cauchy, r['cauchy_w']), (mf3f, r['mf3f_w']), (base, r['base_w'])])
        num, desc = save_blend(blended, f"Three-way diverse: {r['desc']}")
        saved_a.append((num, desc, r))
        print(f"  Saved Sub {num}: {desc}")

    # ================================================================
    # TASK A2: Three-way but blended WITH Sub 2169
    # ================================================================
    print(f"\n{'=' * 70}")
    print("TASK A2: THREE-WAY BLENDED WITH SUB 2169")
    print(f"{'=' * 70}")

    # Take the three-way candidates and blend with Sub 2169
    saved_a2 = []
    for r in task_a_results[:3]:
        three_way = blend_subs([(cauchy, r['cauchy_w']), (mf3f, r['mf3f_w']), (base, r['base_w'])])
        for tw_weight in [0.10, 0.20, 0.30]:
            blended = blend_subs([(three_way, tw_weight), (subs[2169], 1.0 - tw_weight)])
            corrs = per_target_correlation(blended, subs[2169])
            mean_corr = np.mean(list(corrs.values()))
            desc = f"{tw_weight:.0%} ({r['desc']}) + {1-tw_weight:.0%} Sub2169"
            num, desc_full = save_blend(blended, desc)
            saved_a2.append((num, desc))
            print(f"  Sub {num}: {desc} (mean_r={mean_corr:.4f})")

    # ================================================================
    # TASK B: TRAJECTORY DISTANCE BLENDS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("TASK B: TRAJECTORY DISTANCE BLENDS")
    print(f"{'=' * 70}")

    saved_b = []
    if 1507 in subs:
        traj = subs[1507]
        # Check diversity with Sub 2169
        corrs = per_target_correlation(traj, subs[2169])
        print(f"\n  Trajectory (Sub 1507) vs Sub 2169:")
        for col, r in corrs.items():
            print(f"    {col}: r = {r:.4f}")

        # Blend trajectory with Sub 2169 at various weights
        for tw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            blended = blend_subs([(traj, tw), (subs[2169], 1.0 - tw)])
            num, desc = save_blend(blended, f"{tw:.0%} Traj + {1-tw:.0%} Sub2169")
            saved_b.append((num, desc))
            print(f"  Sub {num}: {desc}")

        # Per-target optimized: only use trajectory where it's diverse
        # From research: trajectory helps angle and depth, not LR
        for angle_w in [0.0, 0.10, 0.20]:
            for depth_w in [0.10, 0.20, 0.30]:
                lr_w = 0.0  # Trajectory hurts LR
                blended = subs[2169].copy()
                blended['scaled_angle'] = (1 - angle_w) * subs[2169]['scaled_angle'] + angle_w * traj['scaled_angle']
                blended['scaled_depth'] = (1 - depth_w) * subs[2169]['scaled_depth'] + depth_w * traj['scaled_depth']
                blended['scaled_left_right'] = subs[2169]['scaled_left_right']  # Don't touch LR
                num, desc = save_blend(blended, f"Per-target traj: aw={angle_w}, dw={depth_w}, lw={lr_w}")
                saved_b.append((num, desc))
                print(f"  Sub {num}: {desc}")
    else:
        print("  Sub 1507 (trajectory standalone) not found - skipping")

    # ================================================================
    # TASK C: FINE-GRID PAIRWISE AND THREE-WAY BLENDS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("TASK C: FINE-GRID BLEND OPTIMIZATION")
    print(f"{'=' * 70}")

    # Top 6 LB-tested submissions
    top6 = [2169, 2166, 2194, 2063, 2152, 2151]
    top6_present = [n for n in top6 if n in subs]
    print(f"\n  Top LB submissions available: {top6_present}")

    saved_c = []

    # --- Pairwise blends at 5% steps ---
    print("\n  PAIRWISE BLENDS (5% steps):")
    pairwise_results = []
    for i, j in combinations(range(len(top6_present)), 2):
        num_a, num_b = top6_present[i], top6_present[j]
        for w in np.arange(0.05, 1.00, 0.05):
            w = round(w, 2)
            blended = blend_subs([(subs[num_a], w), (subs[num_b], 1.0 - w)])
            # How different is this from Sub 2169?
            corrs = per_target_correlation(blended, subs[2169])
            diffs = mean_pred_diff(blended, subs[2169])
            mean_diff = np.mean(list(diffs.values()))
            mean_corr = np.mean(list(corrs.values()))
            pairwise_results.append({
                'sub_a': num_a, 'sub_b': num_b, 'w_a': w, 'w_b': 1.0 - w,
                'mean_diff': mean_diff, 'mean_corr': mean_corr,
                'corr_angle': corrs['scaled_angle'],
                'corr_depth': corrs['scaled_depth'],
                'corr_lr': corrs['scaled_left_right'],
            })

    # Sort: we want blends that are DIFFERENT from Sub 2169 (lower correlation)
    # but still plausibly good (using top LB submissions)
    pairwise_results.sort(key=lambda x: x['mean_corr'])
    print(f"\n  Most diverse pairwise blends (lowest correlation with Sub 2169):")
    print(f"  {'Config':<35s} {'r_angle':>8s} {'r_depth':>8s} {'r_LR':>8s} {'mean_r':>8s}")
    for r in pairwise_results[:20]:
        desc = f"{r['w_a']:.0%} Sub{r['sub_a']} + {r['w_b']:.0%} Sub{r['sub_b']}"
        print(f"  {desc:<35s} {r['corr_angle']:8.4f} {r['corr_depth']:8.4f} "
              f"{r['corr_lr']:8.4f} {r['mean_corr']:8.4f}")

    # Save the most diverse pairwise blends (top 5 most different from Sub 2169)
    for r in pairwise_results[:5]:
        blended = blend_subs([(subs[r['sub_a']], r['w_a']), (subs[r['sub_b']], r['w_b'])])
        desc = f"Diverse pair: {r['w_a']:.0%} Sub{r['sub_a']} + {r['w_b']:.0%} Sub{r['sub_b']}"
        num, desc_full = save_blend(blended, desc)
        saved_c.append((num, desc))
        print(f"\n  Saved Sub {num}: {desc} (mean_r={r['mean_corr']:.4f})")

    # Also save blends CLOSEST to Sub 2169 but not identical
    # (these may be slight improvements)
    pairwise_results.sort(key=lambda x: x['mean_diff'])
    print(f"\n  Closest pairwise blends to Sub 2169:")
    for r in pairwise_results[:10]:
        desc = f"{r['w_a']:.0%} Sub{r['sub_a']} + {r['w_b']:.0%} Sub{r['sub_b']}"
        print(f"  {desc:<35s} mean_diff={r['mean_diff']:.6f} mean_r={r['mean_corr']:.4f}")

    # Save closest that aren't trivially Sub 2169
    saved_close = 0
    for r in pairwise_results:
        if saved_close >= 3:
            break
        # Skip if one weight is > 0.90 and the sub is 2169
        if (r['sub_a'] == 2169 and r['w_a'] > 0.90) or (r['sub_b'] == 2169 and r['w_b'] > 0.90):
            continue
        blended = blend_subs([(subs[r['sub_a']], r['w_a']), (subs[r['sub_b']], r['w_b'])])
        desc = f"Close pair: {r['w_a']:.0%} Sub{r['sub_a']} + {r['w_b']:.0%} Sub{r['sub_b']}"
        num, desc_full = save_blend(blended, desc)
        saved_c.append((num, desc))
        print(f"  Saved Sub {num}: {desc} (mean_diff={r['mean_diff']:.6f})")
        saved_close += 1

    # --- Three-way blends at 10% steps among top 4 ---
    print(f"\n  THREE-WAY BLENDS (10% steps among top 4):")
    top4 = [n for n in [2169, 2166, 2194, 2063] if n in subs]
    threeway_results = []
    for i, j, k in combinations(range(len(top4)), 3):
        na, nb, nc = top4[i], top4[j], top4[k]
        for wa in np.arange(0.10, 0.90, 0.10):
            for wb in np.arange(0.10, 0.90 - wa, 0.10):
                wc = round(1.0 - wa - wb, 2)
                wa = round(wa, 2)
                wb = round(wb, 2)
                if wc < 0.05:
                    continue
                blended = blend_subs([(subs[na], wa), (subs[nb], wb), (subs[nc], wc)])
                corrs = per_target_correlation(blended, subs[2169])
                diffs = mean_pred_diff(blended, subs[2169])
                mean_diff = np.mean(list(diffs.values()))
                mean_corr = np.mean(list(corrs.values()))
                threeway_results.append({
                    'subs': (na, nb, nc), 'weights': (wa, wb, wc),
                    'mean_diff': mean_diff, 'mean_corr': mean_corr,
                    'corr_angle': corrs['scaled_angle'],
                    'corr_depth': corrs['scaled_depth'],
                    'corr_lr': corrs['scaled_left_right'],
                })

    # Most diverse three-way
    threeway_results.sort(key=lambda x: x['mean_corr'])
    print(f"\n  Most diverse three-way blends:")
    for r in threeway_results[:10]:
        desc = " + ".join(f"{w:.0%} Sub{s}" for s, w in zip(r['subs'], r['weights']))
        print(f"  {desc:<55s} mean_r={r['mean_corr']:.4f}")

    # Save top 3 diverse three-way
    for r in threeway_results[:3]:
        blended = blend_subs([(subs[s], w) for s, w in zip(r['subs'], r['weights'])])
        desc = "Diverse 3-way: " + " + ".join(f"{w:.0%} Sub{s}" for s, w in zip(r['subs'], r['weights']))
        num, desc_full = save_blend(blended, desc)
        saved_c.append((num, desc))
        print(f"\n  Saved Sub {num}: {desc}")

    # Closest three-way
    threeway_results.sort(key=lambda x: x['mean_diff'])
    print(f"\n  Closest three-way blends to Sub 2169:")
    for r in threeway_results[:10]:
        desc = " + ".join(f"{w:.0%} Sub{s}" for s, w in zip(r['subs'], r['weights']))
        print(f"  {desc:<55s} mean_diff={r['mean_diff']:.6f}")

    # Save top 3 closest (non-trivial)
    saved_close3 = 0
    for r in threeway_results:
        if saved_close3 >= 3:
            break
        # Skip if mostly Sub 2169
        if 2169 in r['subs'] and r['weights'][r['subs'].index(2169)] > 0.80:
            continue
        blended = blend_subs([(subs[s], w) for s, w in zip(r['subs'], r['weights'])])
        desc = "Close 3-way: " + " + ".join(f"{w:.0%} Sub{s}" for s, w in zip(r['subs'], r['weights']))
        num, desc_full = save_blend(blended, desc)
        saved_c.append((num, desc))
        print(f"  Saved Sub {num}: {desc}")
        saved_close3 += 1

    # ================================================================
    # TASK D: SMART COMBINATIONS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("TASK D: SMART COMBINATIONS")
    print(f"{'=' * 70}")

    saved_d = []

    # D1: Per-target blend using best LB source per target
    # Use diversity: pick the source with lowest correlation to Sub 2169 PER TARGET
    # and blend at small weight
    print("\n  D1: Per-target diverse source selection")
    for target_idx_i, target in enumerate(TARGETS):
        col = TARGET_COLS[target_idx_i]
        print(f"\n    {target}:")
        for num in top6_present:
            if num == 2169:
                continue
            r = np.corrcoef(subs[2169][col].values, subs[num][col].values)[0, 1]
            diff = np.mean(np.abs(subs[2169][col].values - subs[num][col].values))
            print(f"      Sub {num}: r={r:.4f}, mean_diff={diff:.6f}")

    # D2: Build "anti-2169" - a submission maximally different from Sub 2169
    # using only top LB submissions
    print("\n  D2: Anti-2169 (maximize diversity while using top subs)")
    # For each target, pick the top LB sub most different from 2169
    anti_2169 = subs[2169].copy()
    anti_desc_parts = []
    for target_idx_i, target in enumerate(TARGETS):
        col = TARGET_COLS[target_idx_i]
        best_num = None
        best_r = 1.0
        for num in top6_present:
            if num == 2169:
                continue
            r = np.corrcoef(subs[2169][col].values, subs[num][col].values)[0, 1]
            if r < best_r:
                best_r = r
                best_num = num
        anti_2169[col] = subs[best_num][col].values
        anti_desc_parts.append(f"{target}=Sub{best_num}(r={best_r:.4f})")
        print(f"    {target}: using Sub {best_num} (r={best_r:.4f})")

    # Blend anti-2169 at small weights with Sub 2169
    for aw in [0.05, 0.10, 0.15, 0.20]:
        blended = blend_subs([(anti_2169, aw), (subs[2169], 1.0 - aw)])
        desc = f"{aw:.0%} anti-2169 ({', '.join(anti_desc_parts)}) + {1-aw:.0%} Sub2169"
        num, desc_full = save_blend(blended, desc)
        saved_d.append((num, desc))
        print(f"  Sub {num}: {desc}")

    # D3: Combine ALL top 6 with equal weights
    n_top = len(top6_present)
    eq_w = 1.0 / n_top
    blended = blend_subs([(subs[n], eq_w) for n in top6_present])
    desc = f"Equal-weight top {n_top}: " + " + ".join(f"{eq_w:.1%} Sub{n}" for n in top6_present)
    num, desc_full = save_blend(blended, desc)
    saved_d.append((num, desc))
    print(f"\n  Sub {num}: {desc}")

    # D4: Inverse-LB-score weighted average
    lb_scores = {2169: 0.006552, 2166: 0.006584, 2194: 0.006589,
                 2063: 0.006603, 2152: 0.006604, 2151: 0.006605}
    inv_scores = {n: 1.0/lb_scores[n] for n in top6_present if n in lb_scores}
    total_inv = sum(inv_scores.values())
    weights_inv_lb = {n: inv_scores[n]/total_inv for n in inv_scores}
    blended = blend_subs([(subs[n], weights_inv_lb[n]) for n in weights_inv_lb])
    desc = "Inv-LB weighted: " + " + ".join(f"{weights_inv_lb[n]:.1%} Sub{n}" for n in weights_inv_lb)
    num, desc_full = save_blend(blended, desc)
    saved_d.append((num, desc))
    print(f"  Sub {num}: {desc}")

    # D5: Heavy-tail combination: 60% Sub2169 + 20% Sub2166 + 10% Sub2194 + 10% Sub2063
    blended = blend_subs([
        (subs[2169], 0.60), (subs[2166], 0.20),
        (subs[2194], 0.10), (subs[2063], 0.10)
    ])
    desc = "60% Sub2169 + 20% Sub2166 + 10% Sub2194 + 10% Sub2063"
    num, desc_full = save_blend(blended, desc)
    saved_d.append((num, desc))
    print(f"  Sub {num}: {desc}")

    # D6: Top 3 average
    blended = blend_subs([
        (subs[2169], 1/3), (subs[2166], 1/3), (subs[2194], 1/3)
    ])
    desc = "Equal top 3: 33% Sub2169 + 33% Sub2166 + 33% Sub2194"
    num, desc_full = save_blend(blended, desc)
    saved_d.append((num, desc))
    print(f"  Sub {num}: {desc}")

    # D7: Sub 2169 with small 5% perturbation from various sources
    for source_num in [2191, 1507, 2063, 2152]:
        if source_num not in subs:
            continue
        blended = blend_subs([(subs[source_num], 0.05), (subs[2169], 0.95)])
        desc = f"95% Sub2169 + 5% Sub{source_num}"
        num, desc_full = save_blend(blended, desc)
        saved_d.append((num, desc))
        print(f"  Sub {num}: {desc}")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("SUMMARY OF ALL SAVED SUBMISSIONS")
    print(f"{'=' * 70}")

    all_saved = saved_a + saved_a2 + saved_b + saved_c + saved_d
    print(f"\n  Total submissions generated: {len(all_saved)}")
    for num, desc, *_ in all_saved:
        extra = _[0] if _ else {}
        print(f"  Sub {num}: {desc}")

    # Compute diversity summary for all saved
    print(f"\n  Diversity with Sub 2169:")
    print(f"  {'Sub':>6s} {'r_angle':>8s} {'r_depth':>8s} {'r_LR':>8s} {'mean_r':>8s} {'mean_diff':>10s}")
    for num, desc, *_ in all_saved:
        sub_df = load_sub(num)
        corrs = per_target_correlation(sub_df, subs[2169])
        diffs = mean_pred_diff(sub_df, subs[2169])
        mean_corr = np.mean(list(corrs.values()))
        mean_diff = np.mean(list(diffs.values()))
        print(f"  {num:>6d} {corrs['scaled_angle']:8.4f} {corrs['scaled_depth']:8.4f} "
              f"{corrs['scaled_left_right']:8.4f} {mean_corr:8.4f} {mean_diff:10.6f}")

    print(f"\n  Total time: {elapsed:.1f}s")

    return all_saved


if __name__ == "__main__":
    main()
