"""
Submission Blending Script
--------------------------
Goal: Find existing submissions diverse from Sub 784 (LB 0.007224) and create blended submissions.

Sub 784 context: Created by blending Sub 771 (LB 0.007556) with PLS depth (30% weight)
and hoop-relative left_right (50% weight).
"""

import fcntl
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# ---- Config ----
PROJECT_DIR = Path("/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge")
SUBMISSION_DIR = PROJECT_DIR / "submission"
BASE_SUB_NUM = 784
BASE_SUB_PATH = SUBMISSION_DIR / f"submission_{BASE_SUB_NUM}.csv"
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
EXPECTED_ROWS = 113


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
            placeholder.touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_submission(path):
    """Load a submission CSV and validate it."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Check columns
    if "id" not in df.columns:
        return None
    for col in TARGET_COLS:
        if col not in df.columns:
            return None

    # Check row count
    if len(df) != EXPECTED_ROWS:
        return None

    # Check for NaN
    if df[TARGET_COLS].isna().any().any():
        return None

    return df


def compute_diversity(base_df, other_df):
    """Compute per-target correlation and MSE between two submissions."""
    correlations = {}
    mses = {}
    for col in TARGET_COLS:
        corr = np.corrcoef(base_df[col].values, other_df[col].values)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations[col] = corr
        mse = np.mean((base_df[col].values - other_df[col].values) ** 2)
        mses[col] = mse

    avg_corr = np.mean(list(correlations.values()))
    avg_mse = np.mean(list(mses.values()))
    return correlations, mses, avg_corr, avg_mse


def blend(base_df, other_df, base_weight):
    """Blend two submissions with given weight for the base."""
    blended = base_df.copy()
    other_weight = 1.0 - base_weight
    for col in TARGET_COLS:
        blended[col] = base_weight * base_df[col].values + other_weight * other_df[col].values
    return blended


def multi_blend(dfs, weights=None):
    """Blend multiple submissions with optional weights (equal if None)."""
    if weights is None:
        weights = [1.0 / len(dfs)] * len(dfs)

    blended = dfs[0].copy()
    for col in TARGET_COLS:
        blended[col] = sum(w * df[col].values for w, df in zip(weights, dfs))
    return blended


def save_submission(df, description):
    """Save a blended submission and return its number."""
    sub_num = get_next_submission_number()
    out_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> Saved submission_{sub_num}.csv: {description}")
    return sub_num


def main():
    print("=" * 80)
    print("SUBMISSION BLENDING ANALYSIS")
    print("=" * 80)

    # Load base submission
    base_df = load_submission(BASE_SUB_PATH)
    if base_df is None:
        print(f"ERROR: Could not load base submission at {BASE_SUB_PATH}")
        return

    print(f"\nBase: submission_{BASE_SUB_NUM}.csv (LB 0.007224)")
    print(f"  Rows: {len(base_df)}, Columns: {list(base_df.columns)}")
    for col in TARGET_COLS:
        print(f"  {col}: mean={base_df[col].mean():.6f}, std={base_df[col].std():.6f}, "
              f"min={base_df[col].min():.6f}, max={base_df[col].max():.6f}")

    # ---- Scan all submissions ----
    print("\n" + "=" * 80)
    print("SCANNING ALL SUBMISSIONS FOR DIVERSITY...")
    print("=" * 80)

    all_files = sorted(SUBMISSION_DIR.glob("submission_*.csv"))
    results = []
    loaded_subs = {}  # num -> df

    for fpath in all_files:
        parts = fpath.stem.split('_')
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        sub_num = int(parts[1])
        if sub_num == BASE_SUB_NUM:
            continue

        df = load_submission(fpath)
        if df is None:
            continue

        # Make sure id columns match
        if not (df["id"].values == base_df["id"].values).all():
            # Try sorting by id
            df = df.sort_values("id").reset_index(drop=True)
            base_sorted = base_df.sort_values("id").reset_index(drop=True)
            if not (df["id"].values == base_sorted["id"].values).all():
                continue
            # Use sorted versions
            correlations, mses, avg_corr, avg_mse = compute_diversity(base_sorted, df)
        else:
            correlations, mses, avg_corr, avg_mse = compute_diversity(base_df, df)

        loaded_subs[sub_num] = df
        results.append({
            "sub_num": sub_num,
            "avg_corr": avg_corr,
            "avg_mse": avg_mse,
            "corr_angle": correlations["scaled_angle"],
            "corr_depth": correlations["scaled_depth"],
            "corr_lr": correlations["scaled_left_right"],
            "mse_angle": mses["scaled_angle"],
            "mse_depth": mses["scaled_depth"],
            "mse_lr": mses["scaled_left_right"],
        })

    print(f"\nTotal valid submissions scanned: {len(results)}")

    results_df = pd.DataFrame(results)

    # ---- Top 30 Most Diverse ----
    print("\n" + "=" * 80)
    print("TOP 30 MOST DIVERSE SUBMISSIONS (lowest avg correlation with Sub 784)")
    print("=" * 80)

    diverse = results_df.sort_values("avg_corr", ascending=True).head(30)
    print(f"\n{'Sub':>6} | {'Avg Corr':>9} | {'Avg MSE':>10} | {'Corr Angle':>11} | "
          f"{'Corr Depth':>11} | {'Corr LR':>11} | {'MSE Angle':>10} | {'MSE Depth':>10} | {'MSE LR':>10}")
    print("-" * 120)
    for _, row in diverse.iterrows():
        print(f"{int(row['sub_num']):>6} | {row['avg_corr']:>9.4f} | {row['avg_mse']:>10.6f} | "
              f"{row['corr_angle']:>11.4f} | {row['corr_depth']:>11.4f} | {row['corr_lr']:>11.4f} | "
              f"{row['mse_angle']:>10.6f} | {row['mse_depth']:>10.6f} | {row['mse_lr']:>10.6f}")

    # ---- Top 20 Similar (corr > 0.90 but < 0.999) ----
    print("\n" + "=" * 80)
    print("TOP 20 SIMILAR SUBMISSIONS (avg corr > 0.90 but < 0.999)")
    print("These are independently-arrived-at similar answers - good for noise reduction")
    print("=" * 80)

    similar_mask = (results_df["avg_corr"] > 0.90) & (results_df["avg_corr"] < 0.999)
    similar = results_df[similar_mask].sort_values("avg_corr", ascending=False).head(20)

    if len(similar) == 0:
        print("\nNo submissions found with avg correlation between 0.90 and 0.999.")
        # Relax criteria
        similar_mask = (results_df["avg_corr"] > 0.80) & (results_df["avg_corr"] < 0.999)
        similar = results_df[similar_mask].sort_values("avg_corr", ascending=False).head(20)
        if len(similar) > 0:
            print("Relaxed to avg corr > 0.80:")

    if len(similar) > 0:
        print(f"\n{'Sub':>6} | {'Avg Corr':>9} | {'Avg MSE':>10} | {'Corr Angle':>11} | "
              f"{'Corr Depth':>11} | {'Corr LR':>11}")
        print("-" * 80)
        for _, row in similar.iterrows():
            print(f"{int(row['sub_num']):>6} | {row['avg_corr']:>9.4f} | {row['avg_mse']:>10.6f} | "
                  f"{row['corr_angle']:>11.4f} | {row['corr_depth']:>11.4f} | {row['corr_lr']:>11.4f}")

    # ---- Also show correlation distribution ----
    print("\n" + "=" * 80)
    print("CORRELATION DISTRIBUTION SUMMARY")
    print("=" * 80)

    for bucket_name, lo, hi in [
        ("Very Low (< 0.3)", -1.0, 0.3),
        ("Low (0.3-0.5)", 0.3, 0.5),
        ("Medium (0.5-0.7)", 0.5, 0.7),
        ("High (0.7-0.9)", 0.7, 0.9),
        ("Very High (0.9-0.999)", 0.9, 0.999),
        ("Near Identical (>= 0.999)", 0.999, 1.01),
    ]:
        count = ((results_df["avg_corr"] >= lo) & (results_df["avg_corr"] < hi)).sum()
        print(f"  {bucket_name}: {count} submissions")

    # ---- Create Blended Submissions ----
    print("\n" + "=" * 80)
    print("CREATING BLENDED SUBMISSIONS")
    print("=" * 80)

    saved_subs = []

    # (a) Blend Sub 784 with top-5 most diverse at various weights
    print("\n--- (a) Blending with TOP-5 MOST DIVERSE submissions ---")
    top5_diverse = diverse.head(5)
    diverse_blend_weights = [0.10, 0.20, 0.30]  # weight for OTHER submission

    for _, row in top5_diverse.iterrows():
        other_num = int(row["sub_num"])
        other_df = loaded_subs[other_num]

        # Ensure id alignment
        if not (other_df["id"].values == base_df["id"].values).all():
            other_df = other_df.sort_values("id").reset_index(drop=True)
            base_for_blend = base_df.sort_values("id").reset_index(drop=True)
        else:
            base_for_blend = base_df

        for other_w in diverse_blend_weights:
            base_w = 1.0 - other_w
            blended = blend(base_for_blend, other_df, base_w)
            desc = (f"Sub784({base_w:.0%}) + Sub{other_num}({other_w:.0%}) "
                    f"[diverse, avg_corr={row['avg_corr']:.4f}]")
            sub_num = save_submission(blended, desc)
            saved_subs.append({
                "sub_num": sub_num,
                "description": desc,
                "type": "diverse_blend",
                "components": f"784+{other_num}",
                "weights": f"{base_w:.2f}/{other_w:.2f}",
            })

    # (b) Blend Sub 784 with top-5 similar at weight 0.50
    print("\n--- (b) Blending with TOP-5 SIMILAR submissions (50/50) ---")
    if len(similar) >= 5:
        top5_similar = similar.head(5)
    else:
        # Fall back to highest correlated submissions below 0.999
        fallback = results_df[results_df["avg_corr"] < 0.999].sort_values("avg_corr", ascending=False).head(5)
        top5_similar = fallback
        print("  (Using top-5 most correlated submissions below 0.999)")

    for _, row in top5_similar.iterrows():
        other_num = int(row["sub_num"])
        other_df = loaded_subs[other_num]

        if not (other_df["id"].values == base_df["id"].values).all():
            other_df = other_df.sort_values("id").reset_index(drop=True)
            base_for_blend = base_df.sort_values("id").reset_index(drop=True)
        else:
            base_for_blend = base_df

        blended = blend(base_for_blend, other_df, 0.50)
        desc = (f"Sub784(50%) + Sub{other_num}(50%) "
                f"[similar, avg_corr={row['avg_corr']:.4f}]")
        sub_num = save_submission(blended, desc)
        saved_subs.append({
            "sub_num": sub_num,
            "description": desc,
            "type": "similar_blend",
            "components": f"784+{other_num}",
            "weights": "0.50/0.50",
        })

    # (c) Multi-model ensemble with top 3 and top 5 similar
    print("\n--- (c) Multi-model ensemble with similar submissions ---")

    for ensemble_size in [3, 5]:
        if len(top5_similar) >= ensemble_size:
            ensemble_nums = [int(r["sub_num"]) for _, r in top5_similar.head(ensemble_size).iterrows()]
            ensemble_dfs = [base_df.copy()]

            base_for_ensemble = base_df.copy()
            for num in ensemble_nums:
                edf = loaded_subs[num].copy()
                if not (edf["id"].values == base_for_ensemble["id"].values).all():
                    edf = edf.sort_values("id").reset_index(drop=True)
                    base_for_ensemble = base_for_ensemble.sort_values("id").reset_index(drop=True)
                    ensemble_dfs = [base_for_ensemble]
                    # Reload others with sorted ids
                    ensemble_dfs_temp = [base_for_ensemble]
                    for n2 in ensemble_nums:
                        edf2 = loaded_subs[n2].copy().sort_values("id").reset_index(drop=True)
                        ensemble_dfs_temp.append(edf2)
                    ensemble_dfs = ensemble_dfs_temp
                    break
                ensemble_dfs.append(edf)

            blended = multi_blend(ensemble_dfs)
            ensemble_str = "+".join([str(BASE_SUB_NUM)] + [str(n) for n in ensemble_nums])
            desc = f"Equal-weight ensemble of {ensemble_size + 1} subs: {ensemble_str}"
            sub_num = save_submission(blended, desc)
            saved_subs.append({
                "sub_num": sub_num,
                "description": desc,
                "type": "multi_ensemble",
                "components": ensemble_str,
                "weights": f"equal({1.0/(ensemble_size+1):.4f})",
            })

    # (c2) Also try weighted ensemble: 50% Sub784 + equal split of rest
    print("\n--- (c2) Weighted ensemble: 50% Sub784 + equal split of similar ---")
    for ensemble_size in [3, 5]:
        if len(top5_similar) >= ensemble_size:
            ensemble_nums = [int(r["sub_num"]) for _, r in top5_similar.head(ensemble_size).iterrows()]

            base_for_ensemble = base_df.copy()
            other_dfs = []
            need_sort = False
            for num in ensemble_nums:
                edf = loaded_subs[num].copy()
                if not (edf["id"].values == base_for_ensemble["id"].values).all():
                    need_sort = True
                    break
                other_dfs.append(edf)

            if need_sort:
                base_for_ensemble = base_for_ensemble.sort_values("id").reset_index(drop=True)
                other_dfs = []
                for num in ensemble_nums:
                    edf = loaded_subs[num].copy().sort_values("id").reset_index(drop=True)
                    other_dfs.append(edf)

            # 50% base + equal split of rest for other 50%
            other_w = 0.50 / ensemble_size
            all_dfs = [base_for_ensemble] + other_dfs
            weights = [0.50] + [other_w] * ensemble_size

            blended = multi_blend(all_dfs, weights)
            ensemble_str = "+".join([str(BASE_SUB_NUM)] + [str(n) for n in ensemble_nums])
            desc = (f"Weighted ensemble: 50% Sub784 + {ensemble_size}x similar "
                    f"({other_w:.2%} each): {ensemble_str}")
            sub_num = save_submission(blended, desc)
            saved_subs.append({
                "sub_num": sub_num,
                "description": desc,
                "type": "weighted_ensemble",
                "components": ensemble_str,
                "weights": f"0.50+{ensemble_size}x{other_w:.4f}",
            })

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL CREATED SUBMISSIONS")
    print("=" * 80)

    for info in saved_subs:
        print(f"\n  Sub {info['sub_num']}: {info['description']}")
        print(f"    Type: {info['type']}")
        print(f"    Components: {info['components']}")
        print(f"    Weights: {info['weights']}")

    print(f"\nTotal new submissions created: {len(saved_subs)}")
    print(f"Submission numbers: {[s['sub_num'] for s in saved_subs]}")

    # ---- Verify a sample blend ----
    print("\n" + "=" * 80)
    print("VERIFICATION: Sample blend statistics")
    print("=" * 80)

    if saved_subs:
        first_sub = saved_subs[0]
        verify_df = pd.read_csv(SUBMISSION_DIR / f"submission_{first_sub['sub_num']}.csv")
        print(f"\nVerifying submission_{first_sub['sub_num']}.csv:")
        print(f"  Rows: {len(verify_df)}")
        print(f"  Columns: {list(verify_df.columns)}")
        for col in TARGET_COLS:
            print(f"  {col}: mean={verify_df[col].mean():.6f}, std={verify_df[col].std():.6f}")

        # Correlation with base
        corrs = {}
        for col in TARGET_COLS:
            corr = np.corrcoef(base_df[col].values, verify_df[col].values)[0, 1]
            corrs[col] = corr
        print(f"  Correlation with Sub 784: angle={corrs['scaled_angle']:.4f}, "
              f"depth={corrs['scaled_depth']:.4f}, lr={corrs['scaled_left_right']:.4f}")


if __name__ == "__main__":
    main()
