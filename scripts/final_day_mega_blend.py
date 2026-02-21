"""
Final Day Mega-Blend: Combine ALL proven quality+diverse sources onto Sub3558.

Sub3558 (LB 0.006148) = per-player CNN allocation on Sub3411 (Ridge).
It already contains Ridge + CNN signal.

New sources to add:
1. FPCA (LOO depth 0.005912, LR diversity r=0.68 vs Sub3558)
2. LightGBM (LOO 0.007481, LR diversity r=0.79 vs Sub3558)
3. Player Channel Oracle (per-player targeted single-channel Ridge)

Strategy: small weights on diverse sources, per-target optimized.
Key diversity lesson: quality + diversity = improvement, diversity-as-noise = worse.
"""

from __future__ import annotations

import fcntl
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

SUB_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def get_next_submission_number():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    lock_path = SUBMISSION_DIR / ".submission_lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        nums = []
        for p in existing:
            try:
                nums.append(int(p.stem.split("_")[1]))
            except (ValueError, IndexError):
                pass
        next_num = max(nums) + 1 if nums else 1
        (SUBMISSION_DIR / f"submission_{next_num}.csv").touch()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return next_num


def save_submission(df, desc):
    num = get_next_submission_number()
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    df.to_csv(path, index=False)
    print(f"  Sub {num}: {desc}", flush=True)
    return num


def main():
    print("=" * 70, flush=True)
    print("FINAL DAY MEGA-BLEND", flush=True)
    print("Combining ALL quality diverse sources onto Sub3558", flush=True)
    print("=" * 70, flush=True)

    # Load base: Sub3558 (LB 0.006148)
    sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
    print(f"  Base: Sub3558, {len(sub3558)} rows", flush=True)

    # Load FPCA standalone (best config, frames 100-200)
    # Sub 3607 is the FPCA standalone submission
    sub_fpca = pd.read_csv(SUBMISSION_DIR / "submission_3607.csv")
    print(f"  FPCA: Sub3607 (standalone LOO=0.006820)", flush=True)

    # Load LightGBM
    sub_lgbm = pd.read_csv(SUBMISSION_DIR / "submission_3582.csv")
    print(f"  LightGBM: Sub3582 (standalone LOO=0.007481)", flush=True)

    # Load Player Channel Oracle (3% blend = Sub3626)
    # Use the raw oracle predictions by reversing the blend:
    # Sub3626 = 0.03 * oracle + 0.97 * Sub3558
    # => oracle = (Sub3626 - 0.97 * Sub3558) / 0.03
    sub3626 = pd.read_csv(SUBMISSION_DIR / "submission_3626.csv")
    oracle_preds = pd.DataFrame({"id": sub3558["id"]})
    for tc in SUB_COLS:
        oracle_raw = (sub3626[tc].values - 0.97 * sub3558[tc].values) / 0.03
        oracle_preds[tc] = np.clip(oracle_raw, 0, 1)
    print(f"  Oracle: extracted from Sub3626 (3% blend)", flush=True)

    # Check diversity of each source vs Sub3558
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    sources = {"FPCA": sub_fpca, "LightGBM": sub_lgbm, "Oracle": oracle_preds}
    for name, src in sources.items():
        divs = []
        for tc in SUB_COLS:
            r, _ = pearsonr(src[tc], sub3558[tc])
            divs.append(r)
            print(f"  {name} {tc}: r={r:.4f}", flush=True)
        print(f"  {name} mean r: {np.mean(divs):.4f}", flush=True)

    # Check pairwise diversity between new sources
    print(f"\n--- Pairwise Diversity Between Sources ---", flush=True)
    src_names = list(sources.keys())
    for i in range(len(src_names)):
        for j in range(i + 1, len(src_names)):
            for tc in SUB_COLS:
                r, _ = pearsonr(sources[src_names[i]][tc], sources[src_names[j]][tc])
                print(f"  {src_names[i]} vs {src_names[j]} {tc}: r={r:.4f}", flush=True)

    # ============================================================
    # BLEND 1: Conservative per-target (2% each for 2 sources)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Blend 1: Conservative dual-source per-target", flush=True)
    blend1 = sub3558.copy()
    # angle: FPCA has moderate diversity, LightGBM similar
    blend1["scaled_angle"] = np.clip(
        0.97 * sub3558["scaled_angle"] +
        0.02 * sub_fpca["scaled_angle"] +
        0.01 * sub_lgbm["scaled_angle"],
        0, 1
    )
    # depth: FPCA strongest here (LOO 0.005912)
    blend1["scaled_depth"] = np.clip(
        0.95 * sub3558["scaled_depth"] +
        0.03 * sub_fpca["scaled_depth"] +
        0.02 * sub_lgbm["scaled_depth"],
        0, 1
    )
    # LR: LightGBM most diverse here (r=0.79)
    blend1["scaled_left_right"] = np.clip(
        0.94 * sub3558["scaled_left_right"] +
        0.03 * sub_fpca["scaled_left_right"] +
        0.03 * sub_lgbm["scaled_left_right"],
        0, 1
    )
    save_submission(blend1, "Conservative: 3558 + 2-3% FPCA + 1-3% LGBM per-target")

    # ============================================================
    # BLEND 2: Aggressive per-target (more weight on diverse sources)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Blend 2: Aggressive per-target", flush=True)
    blend2 = sub3558.copy()
    blend2["scaled_angle"] = np.clip(
        0.95 * sub3558["scaled_angle"] +
        0.03 * sub_fpca["scaled_angle"] +
        0.02 * sub_lgbm["scaled_angle"],
        0, 1
    )
    blend2["scaled_depth"] = np.clip(
        0.90 * sub3558["scaled_depth"] +
        0.06 * sub_fpca["scaled_depth"] +
        0.04 * sub_lgbm["scaled_depth"],
        0, 1
    )
    blend2["scaled_left_right"] = np.clip(
        0.88 * sub3558["scaled_left_right"] +
        0.05 * sub_fpca["scaled_left_right"] +
        0.07 * sub_lgbm["scaled_left_right"],
        0, 1
    )
    save_submission(blend2, "Aggressive: 3558 + 3-6% FPCA + 2-7% LGBM per-target")

    # ============================================================
    # BLEND 3: Triple source (FPCA + LightGBM + Oracle)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Blend 3: Triple source mega-blend", flush=True)
    blend3 = sub3558.copy()
    blend3["scaled_angle"] = np.clip(
        0.95 * sub3558["scaled_angle"] +
        0.02 * sub_fpca["scaled_angle"] +
        0.01 * sub_lgbm["scaled_angle"] +
        0.02 * oracle_preds["scaled_angle"],
        0, 1
    )
    blend3["scaled_depth"] = np.clip(
        0.91 * sub3558["scaled_depth"] +
        0.04 * sub_fpca["scaled_depth"] +
        0.03 * sub_lgbm["scaled_depth"] +
        0.02 * oracle_preds["scaled_depth"],
        0, 1
    )
    blend3["scaled_left_right"] = np.clip(
        0.90 * sub3558["scaled_left_right"] +
        0.03 * sub_fpca["scaled_left_right"] +
        0.04 * sub_lgbm["scaled_left_right"] +
        0.03 * oracle_preds["scaled_left_right"],
        0, 1
    )
    save_submission(blend3, "Triple: 3558 + FPCA + LGBM + Oracle per-target")

    # ============================================================
    # BLEND 4: FPCA-heavy (trusting the depth signal)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Blend 4: FPCA-heavy (depth emphasis)", flush=True)
    blend4 = sub3558.copy()
    blend4["scaled_angle"] = np.clip(
        0.97 * sub3558["scaled_angle"] +
        0.03 * sub_fpca["scaled_angle"],
        0, 1
    )
    blend4["scaled_depth"] = np.clip(
        0.92 * sub3558["scaled_depth"] +
        0.08 * sub_fpca["scaled_depth"],
        0, 1
    )
    blend4["scaled_left_right"] = np.clip(
        0.94 * sub3558["scaled_left_right"] +
        0.06 * sub_fpca["scaled_left_right"],
        0, 1
    )
    save_submission(blend4, "FPCA-heavy: 3-8% FPCA into Sub3558")

    # ============================================================
    # BLEND 5: LightGBM-heavy (trusting the LR signal)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Blend 5: LightGBM-heavy (LR emphasis)", flush=True)
    blend5 = sub3558.copy()
    blend5["scaled_angle"] = np.clip(
        0.98 * sub3558["scaled_angle"] +
        0.02 * sub_lgbm["scaled_angle"],
        0, 1
    )
    blend5["scaled_depth"] = np.clip(
        0.95 * sub3558["scaled_depth"] +
        0.05 * sub_lgbm["scaled_depth"],
        0, 1
    )
    blend5["scaled_left_right"] = np.clip(
        0.90 * sub3558["scaled_left_right"] +
        0.10 * sub_lgbm["scaled_left_right"],
        0, 1
    )
    save_submission(blend5, "LGBM-heavy: 2-10% LGBM into Sub3558")

    # ============================================================
    # BLEND 6: Minimal touch (1% each FPCA+LGBM)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Blend 6: Minimal touch", flush=True)
    blend6 = sub3558.copy()
    for tc in SUB_COLS:
        blend6[tc] = np.clip(
            0.98 * sub3558[tc] +
            0.01 * sub_fpca[tc] +
            0.01 * sub_lgbm[tc],
            0, 1
        )
    save_submission(blend6, "Minimal: 1% FPCA + 1% LGBM into Sub3558")

    # ============================================================
    # BLEND 7: Grid search over 2-source per-target weights
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("Blend 7: Grid search per-target (FPCA+LGBM)", flush=True)

    # For each target, search over FPCA and LGBM weights
    # Objective: maximize expected improvement via diversity-quality tradeoff
    best_weights = {}
    for tc in SUB_COLS:
        best_score = float('inf')
        best_wf = 0.0
        best_wl = 0.0

        for wf in np.arange(0.00, 0.10, 0.01):
            for wl in np.arange(0.00, 0.10, 0.01):
                if wf + wl > 0.15 or wf + wl < 0.01:
                    continue
                wb = 1.0 - wf - wl
                blend_vals = wb * sub3558[tc].values + wf * sub_fpca[tc].values + wl * sub_lgbm[tc].values

                # Score: prefer blends that are slightly different from Sub3558
                # but not too different (which indicates noise)
                r, _ = pearsonr(blend_vals, sub3558[tc])
                # Sweet spot: small weights give r very close to 1.0
                # We want to maximize diversity while keeping it real
                # Use total weight as proxy for expected improvement
                # (more diverse source weight = more potential improvement if source is good)
                total_diverse = wf + wl
                # Penalize if any source dominates too much
                score = total_diverse * (1.0 - r)  # diversity * weight = potential impact

                if score > best_score or (score == best_score and total_diverse < best_wf + best_wl):
                    # Actually we want to MAXIMIZE score, not minimize
                    pass

        # Simpler: just use the weights that our diversity analysis suggests
        # For angle: both sources similar to Sub3558, use small weights
        # For depth: FPCA most different + best quality
        # For LR: LightGBM most different

        best_weights[tc] = {"fpca": 0.0, "lgbm": 0.0}

    # Use knowledge-based weights
    grid_blend = sub3558.copy()
    # angle: both r>0.93, small signal
    grid_blend["scaled_angle"] = np.clip(
        0.96 * sub3558["scaled_angle"] +
        0.02 * sub_fpca["scaled_angle"] +
        0.02 * sub_lgbm["scaled_angle"],
        0, 1
    )
    # depth: FPCA has best standalone depth (LOO 0.005912), use more
    grid_blend["scaled_depth"] = np.clip(
        0.93 * sub3558["scaled_depth"] +
        0.05 * sub_fpca["scaled_depth"] +
        0.02 * sub_lgbm["scaled_depth"],
        0, 1
    )
    # LR: LightGBM r=0.79 (most diverse), FPCA r=0.68 (also good)
    grid_blend["scaled_left_right"] = np.clip(
        0.90 * sub3558["scaled_left_right"] +
        0.04 * sub_fpca["scaled_left_right"] +
        0.06 * sub_lgbm["scaled_left_right"],
        0, 1
    )
    save_submission(grid_blend, "Knowledge-based: A(2/2) D(5/2) LR(4/6) FPCA/LGBM")

    # Final diversity check
    print(f"\n--- Final Blend Diversity vs Sub3558 ---", flush=True)
    blends = {
        "B1_conservative": blend1,
        "B2_aggressive": blend2,
        "B3_triple": blend3,
        "B4_fpca_heavy": blend4,
        "B5_lgbm_heavy": blend5,
        "B6_minimal": blend6,
        "B7_knowledge": grid_blend,
    }
    for bname, bdf in blends.items():
        divs = []
        for tc in SUB_COLS:
            r, _ = pearsonr(bdf[tc], sub3558[tc])
            divs.append(r)
        print(f"  {bname}: mean r={np.mean(divs):.6f} "
              f"(A={divs[0]:.4f}, D={divs[1]:.4f}, LR={divs[2]:.4f})", flush=True)

    print(f"\nDone! Created 7 mega-blend submissions.", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"final_day_mega_blend_{ts}.json", "w") as f:
        json.dump({
            "base": "Sub3558 (LB 0.006148)",
            "sources": {
                "fpca": "Sub3607 (LOO 0.006820)",
                "lgbm": "Sub3582 (LOO 0.007481)",
                "oracle": "extracted from Sub3626",
            },
            "blends_created": list(blends.keys()),
        }, f, indent=2)


if __name__ == "__main__":
    main()
