"""
Targeted blend candidates for SPLxUTSPAN competition.
Generates 7 specific per-target blend submissions combining Sub 2716 (best)
with diverse standalone sources at carefully chosen weights.
"""
import fcntl
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT / "submission"
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
    path = SUBMISSION_DIR / f"submission_{num}.csv"
    df = pd.read_csv(path).set_index("id")
    return df


def corr_with_base(blend_df, base_df):
    """Compute per-target Pearson correlation between blend and base."""
    corrs = {}
    for t in TARGETS:
        corrs[t] = np.corrcoef(blend_df[t].values, base_df[t].values)[0, 1]
    return corrs


def make_blend(base_df, specs, ids):
    """
    Create a blend DataFrame.
    specs: dict of {target_col: list of (weight, df)}
    For each target, the weights should sum to 1.0.
    """
    result = pd.DataFrame({"id": ids})
    for t in TARGETS:
        val = np.zeros(len(ids), dtype=np.float64)
        for w, df in specs[t]:
            val += w * df[t].values
        val = np.clip(val, 0.0, 1.0)
        result[t] = val
    return result


def main():
    print("Loading source submissions...", flush=True)

    base = load_sub(2716)
    bigru10 = load_sub(2887)
    knn_boot = load_sub(2784)
    cnn_temp = load_sub(1557)
    pulse = load_sub(2608)
    rf_vel = load_sub(2780)

    ids = base.index.tolist()
    print(f"  Base Sub 2716: {len(ids)} rows", flush=True)

    # Define all 7 blend candidates
    candidates = []

    # 1. BiGRU-depth + kNN-LR combined fix
    candidates.append({
        "name": "BiGRU-depth + kNN-LR combined fix",
        "specs": {
            "scaled_angle":      [(1.00, base)],
            "scaled_depth":      [(0.98, base), (0.02, bigru10)],
            "scaled_left_right": [(0.98, base), (0.02, knn_boot)],
        }
    })

    # 2. BiGRU-depth + kNN-LR + CNN angle
    candidates.append({
        "name": "BiGRU-depth + kNN-LR + CNN angle",
        "specs": {
            "scaled_angle":      [(0.98, base), (0.02, cnn_temp)],
            "scaled_depth":      [(0.98, base), (0.02, bigru10)],
            "scaled_left_right": [(0.98, base), (0.02, knn_boot)],
        }
    })

    # 3. Triple 3% per-target best
    candidates.append({
        "name": "Triple 3% per-target best",
        "specs": {
            "scaled_angle":      [(0.97, base), (0.03, cnn_temp)],
            "scaled_depth":      [(0.97, base), (0.03, bigru10)],
            "scaled_left_right": [(0.97, base), (0.03, knn_boot)],
        }
    })

    # 4. 1% micro-doses all best
    candidates.append({
        "name": "1% micro-doses all best",
        "specs": {
            "scaled_angle":      [(0.97, base), (0.01, cnn_temp), (0.01, bigru10), (0.01, knn_boot)],
            "scaled_depth":      [(0.97, base), (0.01, bigru10), (0.01, pulse), (0.01, knn_boot)],
            "scaled_left_right": [(0.97, base), (0.01, knn_boot), (0.01, rf_vel), (0.01, bigru10)],
        }
    })

    # 5. 2% BiGRU all targets
    candidates.append({
        "name": "2% BiGRU all targets",
        "specs": {
            "scaled_angle":      [(0.98, base), (0.02, bigru10)],
            "scaled_depth":      [(0.98, base), (0.02, bigru10)],
            "scaled_left_right": [(0.98, base), (0.02, bigru10)],
        }
    })

    # 6. 2% kNN all targets
    candidates.append({
        "name": "2% kNN all targets",
        "specs": {
            "scaled_angle":      [(0.98, base), (0.02, knn_boot)],
            "scaled_depth":      [(0.98, base), (0.02, knn_boot)],
            "scaled_left_right": [(0.98, base), (0.02, knn_boot)],
        }
    })

    # 7. Progressive: 1% BiGRU depth, 1% kNN LR only
    candidates.append({
        "name": "Progressive: 1% BiGRU depth, 1% kNN LR",
        "specs": {
            "scaled_angle":      [(1.00, base)],
            "scaled_depth":      [(0.99, base), (0.01, bigru10)],
            "scaled_left_right": [(0.99, base), (0.01, knn_boot)],
        }
    })

    print(f"\nGenerating {len(candidates)} blend submissions...\n", flush=True)

    results = []
    for i, cand in enumerate(candidates, 1):
        blend_df = make_blend(base, cand["specs"], ids)
        sn = get_next_submission_number()
        sp = SUBMISSION_DIR / f"submission_{sn}.csv"
        blend_df.to_csv(sp, index=False)

        # Compute correlations with base
        blend_indexed = blend_df.set_index("id")
        corrs = corr_with_base(blend_indexed, base)

        desc = cand["name"]
        results.append((sn, desc, corrs))

        print(f"  [{i}/7] Sub {sn}: {desc}", flush=True)
        print(f"         File: submission_{sn}.csv", flush=True)
        print(f"         Corr w/ 2716: angle={corrs['scaled_angle']:.6f}  "
              f"depth={corrs['scaled_depth']:.6f}  "
              f"LR={corrs['scaled_left_right']:.6f}", flush=True)

        # Verify shape
        check = pd.read_csv(sp)
        assert len(check) == 113, f"Expected 113 rows, got {len(check)}"
        assert list(check.columns) == ["id", "scaled_angle", "scaled_depth", "scaled_left_right"], \
            f"Bad columns: {list(check.columns)}"
        assert check[TARGETS].min().min() >= 0.0, "Values below 0"
        assert check[TARGETS].max().max() <= 1.0, "Values above 1"
        print(f"         Verified: 113 rows, correct columns, values in [0,1]", flush=True)
        print(flush=True)

    # Summary table
    print("=" * 90, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Sub':>6}  {'Description':<45}  {'angle':>8}  {'depth':>8}  {'LR':>8}", flush=True)
    print("-" * 90, flush=True)
    for sn, desc, corrs in results:
        print(f"{sn:>6}  {desc:<45}  "
              f"{corrs['scaled_angle']:.6f}  "
              f"{corrs['scaled_depth']:.6f}  "
              f"{corrs['scaled_left_right']:.6f}", flush=True)
    print("-" * 90, flush=True)
    print("Lower correlation = more diverse blend (potential for improvement if source is good)", flush=True)
    print(f"\nAll {len(candidates)} submissions generated successfully.", flush=True)


if __name__ == "__main__":
    main()
