"""
Optimal Reallocation: Rebuild Sub 2716 from scratch with better diverse sources.

Sub 2716 effective composition:
  87.44% Ridge + 4.80% Energy + 3.92% CNN + 2.00% TreeAvg + 1.84% Pulse

This script replaces the weakest sources (TreeAvg, Pulse) with the strongest
diversity sources discovered (BiGRU for depth, kNN bootstrap for LR), keeping
total diverse weight at ~12.5%.

Evidence:
- BiGRU: depth r=0.505 (vs TreeAvg depth r=0.835) = much better depth diversity
- kNN bootstrap: LR r=0.501 (vs Pulse LR r=0.705) = much better LR diversity
- CNN and Energy: proven LB signal, keep as-is
"""

import fcntl
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


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
    df = pd.read_csv(SUBMISSION_DIR / f"submission_{num}.csv")
    return df[TARGETS].values, df['id'].values


def blend(sources_weights, ids):
    """Blend multiple sources with given weights. sources_weights = [(preds, weight), ...]"""
    result = np.zeros_like(sources_weights[0][0])
    for preds, w in sources_weights:
        result += w * preds
    return result


def save_submission(preds, ids, desc):
    sub_num = get_next_submission_number()
    df = pd.DataFrame({
        'id': ids,
        'scaled_angle': np.clip(preds[:, 0], 0, 1),
        'scaled_depth': np.clip(preds[:, 1], 0, 1),
        'scaled_left_right': np.clip(preds[:, 2], 0, 1)
    })
    df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    return sub_num


def analyze(preds, anchor, name):
    for t, tname in enumerate(["angle", "depth", "LR"]):
        r = np.corrcoef(preds[:, t], anchor[:, t])[0, 1]
        mse_diff = np.mean((preds[:, t] - anchor[:, t]) ** 2)
        print(f"  {tname}: r={r:.4f}, MSE_diff={mse_diff:.8f}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("Optimal Reallocation: Better Diverse Sources", flush=True)
    print("=" * 60, flush=True)

    # Load all sources
    ridge, ids = load_sub(2503)  # Pure Ridge base
    anchor, _ = load_sub(2716)   # Current best anchor
    energy, _ = load_sub(2602)   # Energy wave standalone
    cnn, _ = load_sub(1557)      # Temporal CNN
    pulse, _ = load_sub(2608)    # Pulse standalone
    tree_avg = 0.5 * load_sub(2679)[0] + 0.5 * load_sub(2704)[0]  # XGB+LGBM average
    bigru_5s, _ = load_sub(2870)  # BiGRU 5-seed
    bigru_10s, _ = load_sub(2887) # BiGRU 10-seed
    knn_boot, _ = load_sub(2784)  # kNN bootstrap
    rf_vel, _ = load_sub(2780)    # RF velocity
    trajectory, _ = load_sub(1507)  # Trajectory model

    print("\nSource diversity vs Sub 2716:", flush=True)
    for name, preds in [("Ridge 2503", ridge), ("Energy", energy), ("CNN", cnn),
                         ("Pulse", pulse), ("TreeAvg", tree_avg),
                         ("BiGRU-5s", bigru_5s), ("BiGRU-10s", bigru_10s),
                         ("kNN-boot", knn_boot), ("RF-vel", rf_vel),
                         ("Trajectory", trajectory)]:
        print(f"\n  {name}:", flush=True)
        analyze(preds, anchor, name)

    submissions = []

    # === Strategy 1: Direct reallocation (same 12.5% total, better sources) ===
    # Replace TreeAvg->BiGRU, Pulse->kNN_boot
    print("\n--- Strategy 1: Direct reallocation ---", flush=True)
    preds = blend([
        (ridge, 0.8744),
        (energy, 0.0480),
        (cnn, 0.0392),
        (bigru_10s, 0.0200),  # was TreeAvg
        (knn_boot, 0.0184),   # was Pulse
    ], ids)
    sub_num = save_submission(preds, ids, "realloc_v1")
    print(f"Sub {sub_num}: 87.4%R + 4.8%E + 3.9%CNN + 2.0%BiGRU + 1.8%kNN", flush=True)
    analyze(preds, anchor, "realloc_v1")
    submissions.append(sub_num)

    # === Strategy 2: Per-target optimal reallocation ===
    # For each target column, use the best diverse source
    print("\n--- Strategy 2: Per-target best source ---", flush=True)
    preds2 = np.zeros_like(ridge)
    # Angle: CNN is best per-target diversity (r=0.887)
    preds2[:, 0] = blend([(ridge[:, 0:1], 0.87), (energy[:, 0:1], 0.05),
                           (cnn[:, 0:1], 0.06), (bigru_10s[:, 0:1], 0.02)], ids)[:, 0]
    # Depth: BiGRU is best (r=0.505)
    preds2[:, 1] = blend([(ridge[:, 1:2], 0.87), (energy[:, 1:2], 0.05),
                           (bigru_10s[:, 1:2], 0.05), (cnn[:, 1:2], 0.03)], ids)[:, 0]
    # LR: kNN bootstrap is best (r=0.501)
    preds2[:, 2] = blend([(ridge[:, 2:3], 0.87), (energy[:, 2:3], 0.05),
                           (knn_boot[:, 2:3], 0.05), (cnn[:, 2:3], 0.03)], ids)[:, 0]
    sub_num = save_submission(preds2, ids, "per_target_best")
    print(f"Sub {sub_num}: Per-target best: A=6%CNN+2%BiGRU, D=5%BiGRU+3%CNN, LR=5%kNN+3%CNN", flush=True)
    analyze(preds2, anchor, "per_target_best")
    submissions.append(sub_num)

    # === Strategy 3: Conservative reallocation (2% per source max) ===
    print("\n--- Strategy 3: Conservative 2% per source ---", flush=True)
    preds3 = blend([
        (ridge, 0.87),
        (energy, 0.02),
        (cnn, 0.02),
        (bigru_10s, 0.02),
        (knn_boot, 0.02),
        (rf_vel, 0.02),
        (pulse, 0.01),
        (tree_avg, 0.01),
        (trajectory, 0.01),
    ], ids)
    sub_num = save_submission(preds3, ids, "conservative_2pct")
    print(f"Sub {sub_num}: 87%R + 9 sources at 1-2% each (13% total)", flush=True)
    analyze(preds3, anchor, "conservative_2pct")
    submissions.append(sub_num)

    # === Strategy 4: Per-target optimal with Sub 2716 as anchor ===
    # Start from Sub 2716 (already has 12.5% diverse), add 2% more of BEST sources
    print("\n--- Strategy 4: Sub 2716 + targeted corrections ---", flush=True)
    preds4 = anchor.copy()
    # Depth: add 2% BiGRU (highest diversity)
    preds4[:, 1] = 0.98 * anchor[:, 1] + 0.02 * bigru_10s[:, 1]
    # LR: add 2% kNN (highest diversity)
    preds4[:, 2] = 0.98 * anchor[:, 2] + 0.02 * knn_boot[:, 2]
    sub_num = save_submission(preds4, ids, "2716_plus_best")
    print(f"Sub {sub_num}: Sub2716 + 2%BiGRU(depth) + 2%kNN(LR)", flush=True)
    analyze(preds4, anchor, "2716_plus_best")
    submissions.append(sub_num)

    # === Strategy 5: Replace worst with best in Sub 2716 ===
    # Sub 2716 has TreeAvg and Pulse (least diverse). Replace with BiGRU and kNN.
    # Approximate: subtract out TreeAvg/Pulse contribution, add BiGRU/kNN
    print("\n--- Strategy 5: Swap in Sub 2716 ---", flush=True)
    # Current: anchor = 0.8744*ridge + 0.048*energy + 0.0392*cnn + 0.02*tree + 0.0184*pulse
    # New: subtract tree and pulse, add bigru and knn at same weights
    preds5 = anchor - 0.02 * tree_avg - 0.0184 * pulse + 0.02 * bigru_10s + 0.0184 * knn_boot
    sub_num = save_submission(preds5, ids, "swap_in_2716")
    print(f"Sub {sub_num}: Sub2716 with TreeAvg->BiGRU and Pulse->kNN swapped", flush=True)
    analyze(preds5, anchor, "swap_in_2716")
    submissions.append(sub_num)

    # === Strategy 6: Aggressive BiGRU for depth ===
    # Depth is where BiGRU shines (r=0.505). Try 5% BiGRU for depth only
    print("\n--- Strategy 6: Aggressive BiGRU depth ---", flush=True)
    preds6 = anchor.copy()
    preds6[:, 1] = 0.95 * anchor[:, 1] + 0.05 * bigru_10s[:, 1]
    sub_num = save_submission(preds6, ids, "5pct_bigru_depth")
    print(f"Sub {sub_num}: Sub2716 + 5%BiGRU(depth only)", flush=True)
    analyze(preds6, anchor, "5pct_bigru_depth")
    submissions.append(sub_num)

    # === Strategy 7: Full rebuild with 8 sources ===
    print("\n--- Strategy 7: Full 8-source rebuild ---", flush=True)
    # Per-target weights optimized manually based on diversity rankings
    preds7 = np.zeros_like(ridge)
    # Angle: CNN most diverse, BiGRU second
    preds7[:, 0] = (0.88 * ridge[:, 0] + 0.03 * cnn[:, 0] + 0.02 * bigru_10s[:, 0] +
                     0.02 * knn_boot[:, 0] + 0.02 * energy[:, 0] + 0.02 * rf_vel[:, 0] +
                     0.01 * pulse[:, 0])
    # Depth: BiGRU most diverse, Pulse second, kNN third
    preds7[:, 1] = (0.87 * ridge[:, 1] + 0.03 * bigru_10s[:, 1] + 0.025 * pulse[:, 1] +
                     0.02 * knn_boot[:, 1] + 0.02 * cnn[:, 1] + 0.02 * energy[:, 1] +
                     0.015 * rf_vel[:, 1])
    # LR: kNN most diverse, RF second, BiGRU third
    preds7[:, 2] = (0.87 * ridge[:, 2] + 0.03 * knn_boot[:, 2] + 0.025 * rf_vel[:, 2] +
                     0.02 * bigru_10s[:, 2] + 0.02 * cnn[:, 2] + 0.02 * energy[:, 2] +
                     0.015 * pulse[:, 2])
    sub_num = save_submission(preds7, ids, "8source_rebuild")
    print(f"Sub {sub_num}: 8-source per-target optimized rebuild", flush=True)
    analyze(preds7, anchor, "8source_rebuild")
    submissions.append(sub_num)

    # === Strategy 8: Average of BiGRU 5-seed and 10-seed (more stable) ===
    print("\n--- Strategy 8: Averaged BiGRU ---", flush=True)
    bigru_avg = 0.5 * bigru_5s + 0.5 * bigru_10s
    preds8 = anchor.copy()
    preds8[:, 1] = 0.98 * anchor[:, 1] + 0.02 * bigru_avg[:, 1]
    sub_num = save_submission(preds8, ids, "avg_bigru_depth")
    print(f"Sub {sub_num}: Sub2716 + 2% averaged BiGRU (depth only)", flush=True)
    analyze(preds8, anchor, "avg_bigru_depth")
    submissions.append(sub_num)

    # === Strategy 9: BiGRU averaged for depth + kNN for LR ===
    print("\n--- Strategy 9: Dual fix with averaged BiGRU ---", flush=True)
    preds9 = anchor.copy()
    preds9[:, 1] = 0.98 * anchor[:, 1] + 0.02 * bigru_avg[:, 1]
    preds9[:, 2] = 0.98 * anchor[:, 2] + 0.02 * knn_boot[:, 2]
    sub_num = save_submission(preds9, ids, "dual_fix_avg")
    print(f"Sub {sub_num}: Sub2716 + 2%avgBiGRU(depth) + 2%kNN(LR)", flush=True)
    analyze(preds9, anchor, "dual_fix_avg")
    submissions.append(sub_num)

    # === Strategy 10: Triple 1% additions ===
    print("\n--- Strategy 10: Triple 1% micro-dose ---", flush=True)
    preds10 = anchor.copy()
    preds10[:, 0] = 0.99 * anchor[:, 0] + 0.01 * cnn[:, 0]  # 1% CNN for angle
    preds10[:, 1] = 0.99 * anchor[:, 1] + 0.01 * bigru_avg[:, 1]  # 1% BiGRU for depth
    preds10[:, 2] = 0.99 * anchor[:, 2] + 0.01 * knn_boot[:, 2]  # 1% kNN for LR
    sub_num = save_submission(preds10, ids, "triple_1pct")
    print(f"Sub {sub_num}: Sub2716 + 1%CNN(angle) + 1%avgBiGRU(depth) + 1%kNN(LR)", flush=True)
    analyze(preds10, anchor, "triple_1pct")
    submissions.append(sub_num)

    print(f"\n{'='*60}", flush=True)
    print(f"Generated {len(submissions)} reallocation candidates:", flush=True)
    for s in submissions:
        print(f"  Sub {s}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
