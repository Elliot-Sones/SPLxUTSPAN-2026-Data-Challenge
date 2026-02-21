"""
N-Way Per-Target Blend Optimizer for SPLxUTSPAN 2026 Kaggle Competition.

Builds Frankenstein submissions by independently optimizing the blend
for each target (angle, depth, LR) from diverse source submissions.

Uses a surrogate MSE model calibrated from known LB scores to guide
the optimization, then generates multiple candidate submissions.

Mathematical framework:
  MSE(wA + (1-w)B) = w^2*MSE_A + (1-w)^2*MSE_B + 2*w*(1-w)*rho*sqrt(MSE_A*MSE_B)
  where rho is the error correlation between A and B.

For N-way blends:
  MSE(sum_i w_i * X_i) = sum_i sum_j w_i * w_j * rho_ij * sqrt(MSE_i * MSE_j)
"""

import time
import fcntl
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.optimize import minimize

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"
RESEARCH_DIR = PROJECT_DIR / "Research"
TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_NAMES = ["angle", "depth", "LR"]


# ============================================================
# KNOWN LB SCORES (calibration anchors)
# ============================================================
KNOWN_LB_SCORES = {
    2716: 0.006343,
    2503: 0.006471,
    2475: 0.006485,
    2429: 0.006502,
    2402: 0.006511,
    2372: 0.006538,
    2169: 0.006552,
    2063: 0.006603,
    1828: 0.006619,
    784: 0.007224,
}


def get_next_submission_number():
    """Atomically get the next submission number using a lock file."""
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


def load_submission(sub_num):
    """Load a single submission CSV."""
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) != 113:
        return None
    if not all(c in df.columns for c in TARGETS):
        return None
    return df


def load_all_submissions(sub_nums):
    """Load multiple submissions, return dict of {num: DataFrame}."""
    subs = {}
    for num in sub_nums:
        df = load_submission(num)
        if df is not None:
            subs[num] = df
    return subs


def compute_per_target_correlations(subs, sub_nums):
    """Compute NxN correlation matrix for each target."""
    n = len(sub_nums)
    corr = {t: np.ones((n, n)) for t in TARGETS}

    # Build data matrices per target
    data = {}
    for t in TARGETS:
        data[t] = np.zeros((113, n))
        for j, num in enumerate(sub_nums):
            data[t][:, j] = subs[num][t].values

    for t in TARGETS:
        for i in range(n):
            for j in range(i + 1, n):
                r = np.corrcoef(data[t][:, i], data[t][:, j])[0, 1]
                if np.isnan(r):
                    r = 0.0
                corr[t][i, j] = r
                corr[t][j, i] = r

    return corr, data


def greedy_diverse_selection(corr, sub_nums, base_idx, max_subs=25, target=None):
    """Greedy forward selection: pick submissions that maximize diversity.

    Starts with base submission, then adds the most diverse one at each step.
    If target is specified, optimize diversity for that target only.
    Otherwise, optimize average diversity across all targets.
    """
    n = len(sub_nums)
    selected = [base_idx]

    for _ in range(max_subs - 1):
        best_score = float('inf')
        best_idx = -1
        for j in range(n):
            if j in selected:
                continue
            # Score = max correlation with any already-selected submission
            if target is not None:
                max_corr = max(abs(corr[target][j, s]) for s in selected)
            else:
                max_corr = max(
                    np.mean([abs(corr[t][j, s]) for t in TARGETS])
                    for s in selected
                )
            if max_corr < best_score:
                best_score = max_corr
                best_idx = j
        if best_idx == -1:
            break
        selected.append(best_idx)

    return selected


def estimate_per_target_mse(known_scores, sub_nums, corr, data):
    """Estimate per-target MSE for known-score submissions.

    We know total MSE = mean(MSE_angle, MSE_depth, MSE_LR).
    We can estimate relative per-target MSE from variance of predictions
    and known relationships between submissions.

    Strategy: Use the prediction variance ratio as a proxy for per-target
    MSE ratio, then calibrate using known total MSE scores.
    """
    # Method: For submissions with known LB scores, estimate per-target MSE
    # using the constraint that mean(MSE_t) = total_MSE for each sub.
    #
    # We use inter-submission prediction differences as a proxy:
    # Submissions that differ more on a target likely have larger errors on it.
    #
    # Simple approach: assume per-target MSE is proportional to prediction variance
    # across diverse submissions (more disagreement = harder target = larger MSE).

    known_nums = [n for n in sub_nums if n in known_scores]
    if len(known_nums) < 2:
        # Fallback: equal split
        return {n: {t: known_scores[n] for t in TARGETS} for n in known_nums}

    # Compute prediction variance per target across all known-score submissions
    known_indices = [sub_nums.index(n) for n in known_nums]

    target_var = {}
    for t in TARGETS:
        # Variance of predictions across known submissions per row
        preds = data[t][:, known_indices]  # (113, n_known)
        target_var[t] = np.mean(np.var(preds, axis=1))

    total_var = sum(target_var.values())
    if total_var < 1e-15:
        target_ratio = {t: 1.0 / 3.0 for t in TARGETS}
    else:
        target_ratio = {t: target_var[t] / total_var for t in TARGETS}

    print(f"\n  Estimated target MSE ratios: "
          f"angle={target_ratio[TARGETS[0]]:.3f}, "
          f"depth={target_ratio[TARGETS[1]]:.3f}, "
          f"LR={target_ratio[TARGETS[2]]:.3f}")

    per_target_mse = {}
    for n in known_nums:
        total = known_scores[n]
        per_target_mse[n] = {}
        for t in TARGETS:
            per_target_mse[n][t] = total * 3.0 * target_ratio[t]

    return per_target_mse


def surrogate_blend_mse(weights, mse_values, corr_matrix, indices):
    """Estimate MSE of a blended submission for a single target.

    MSE(sum_i w_i X_i) ~= sum_i sum_j w_i w_j rho_ij sqrt(MSE_i MSE_j)
    """
    n = len(weights)
    total = 0.0
    for i in range(n):
        for j in range(n):
            ii, jj = indices[i], indices[j]
            rho = corr_matrix[ii, jj]
            total += weights[i] * weights[j] * rho * np.sqrt(mse_values[i] * mse_values[j])
    return total


def optimize_blend_weights(mse_values, corr_matrix, indices, max_nonzero=7):
    """Optimize blend weights for a single target using scipy.

    Returns optimal weights and estimated MSE.
    """
    n = len(mse_values)
    if n == 0:
        return np.array([]), float('inf')
    if n == 1:
        return np.array([1.0]), mse_values[0]

    # Objective: minimize surrogate MSE
    def objective(w):
        return surrogate_blend_mse(w, mse_values, corr_matrix, indices)

    # Constraints: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1)] * n

    # Try multiple random initializations
    best_result = None
    best_mse = float('inf')

    for seed in range(20):
        rng = np.random.RandomState(seed)
        if seed == 0:
            # Start with best single submission
            w0 = np.zeros(n)
            best_idx = np.argmin(mse_values)
            w0[best_idx] = 1.0
        elif seed == 1:
            # Equal weights
            w0 = np.ones(n) / n
        elif seed == 2:
            # Inverse-MSE weights
            inv_mse = 1.0 / (np.array(mse_values) + 1e-10)
            w0 = inv_mse / inv_mse.sum()
        else:
            # Random Dirichlet
            w0 = rng.dirichlet(np.ones(n))

        try:
            result = minimize(objective, w0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 500, 'ftol': 1e-15})
            if result.fun < best_mse:
                best_mse = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        w = np.zeros(n)
        w[np.argmin(mse_values)] = 1.0
        return w, mse_values[np.argmin(mse_values)]

    w = best_result.x

    # Enforce sparsity: keep only top max_nonzero weights
    if np.sum(w > 0.005) > max_nonzero:
        top_idx = np.argsort(w)[-max_nonzero:]
        mask = np.zeros(n, dtype=bool)
        mask[top_idx] = True
        w[~mask] = 0.0
        w = w / (w.sum() + 1e-15)

        # Re-optimize with only the selected sources
        active = np.where(w > 0)[0]
        sub_mse = [mse_values[i] for i in active]
        sub_indices = [indices[i] for i in active]

        def sub_objective(w_sub):
            return surrogate_blend_mse(w_sub, sub_mse, corr_matrix, sub_indices)

        sub_constraints = [{'type': 'eq', 'fun': lambda ws: np.sum(ws) - 1.0}]
        sub_bounds = [(0, 1)] * len(active)
        w_sub0 = w[active] / (w[active].sum() + 1e-15)

        try:
            result2 = minimize(sub_objective, w_sub0, method='SLSQP',
                               bounds=sub_bounds, constraints=sub_constraints,
                               options={'maxiter': 500, 'ftol': 1e-15})
            w = np.zeros(n)
            w[active] = result2.x
            best_mse = result2.fun
        except Exception:
            pass

    # Clean small weights
    w[w < 0.005] = 0.0
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.zeros(n)
        w[np.argmin(mse_values)] = 1.0

    return w, best_mse


def grid_search_2way(target, sub_nums_pool, data_t, mse_dict, corr_t):
    """Exhaustive grid search over 2-way blends for a target."""
    pool = [n for n in sub_nums_pool if n in mse_dict]
    if len(pool) < 2:
        return []

    results = []
    for i, n1 in enumerate(pool):
        for j, n2 in enumerate(pool):
            if j <= i:
                continue
            idx1 = sub_nums_pool.index(n1)
            idx2 = sub_nums_pool.index(n2)
            rho = corr_t[idx1, idx2]
            m1 = mse_dict[n1]
            m2 = mse_dict[n2]

            best_w = None
            best_mse = float('inf')
            for w in np.arange(0, 1.01, 0.01):
                mse = (w ** 2 * m1 + (1 - w) ** 2 * m2 +
                       2 * w * (1 - w) * rho * np.sqrt(m1 * m2))
                if mse < best_mse:
                    best_mse = mse
                    best_w = w

            results.append({
                'subs': (n1, n2),
                'weights': (best_w, 1 - best_w),
                'est_mse': best_mse,
                'rho': rho,
            })

    results.sort(key=lambda x: x['est_mse'])
    return results[:20]


def grid_search_3way(target, sub_nums_pool, data_t, mse_dict, corr_t):
    """Grid search over 3-way blends for a target."""
    pool = [n for n in sub_nums_pool if n in mse_dict]
    if len(pool) < 3:
        return []

    # Only consider top subs by MSE
    pool_sorted = sorted(pool, key=lambda n: mse_dict[n])[:12]

    results = []
    step = 0.05
    weights_grid = np.arange(0, 1.01, step)

    for i, n1 in enumerate(pool_sorted):
        for j, n2 in enumerate(pool_sorted):
            if j <= i:
                continue
            for k, n3 in enumerate(pool_sorted):
                if k <= j:
                    continue

                idx1 = sub_nums_pool.index(n1)
                idx2 = sub_nums_pool.index(n2)
                idx3 = sub_nums_pool.index(n3)
                indices = [idx1, idx2, idx3]
                mse_vals = [mse_dict[n1], mse_dict[n2], mse_dict[n3]]

                best_w = None
                best_mse = float('inf')

                for w1 in weights_grid:
                    for w2 in weights_grid:
                        w3 = 1.0 - w1 - w2
                        if w3 < -0.001 or w3 > 1.001:
                            continue
                        w3 = max(0, w3)
                        ws = [w1, w2, w3]
                        mse = surrogate_blend_mse(ws, mse_vals, corr_t, indices)
                        if mse < best_mse:
                            best_mse = mse
                            best_w = (w1, w2, w3)

                results.append({
                    'subs': (n1, n2, n3),
                    'weights': best_w,
                    'est_mse': best_mse,
                })

    results.sort(key=lambda x: x['est_mse'])
    return results[:15]


def build_blended_predictions(data_t, sub_nums, weights_dict):
    """Build blended predictions for one target given weight dict {sub_num: weight}."""
    pred = np.zeros(113)
    for num, w in weights_dict.items():
        if w > 0:
            idx = sub_nums.index(num)
            pred += w * data_t[:, idx]
    return pred


def save_frankenstein(angle_pred, depth_pred, lr_pred, ids, label):
    """Save a Frankenstein submission (different source per target)."""
    sub_num = get_next_submission_number()
    df = pd.DataFrame({
        'id': ids,
        'scaled_angle': angle_pred,
        'scaled_depth': depth_pred,
        'scaled_left_right': lr_pred,
    })
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    df.to_csv(path, index=False)
    return sub_num


def main():
    t0 = time.time()
    print("=" * 70)
    print("N-WAY PER-TARGET BLEND OPTIMIZER")
    print("=" * 70)
    print(f"  Current best: Sub 2716, LB 0.006343")
    print(f"  Target: 0.0059")

    # ============================================================
    # STEP 1: Define candidate pool
    # ============================================================
    # Start with known-score anchors + diverse submissions found from scan
    anchor_subs = list(KNOWN_LB_SCORES.keys())

    # Diverse submissions from the scan (moderate to high diversity vs 2716)
    diverse_subs = [
        # Extreme diversity (r < 0.5 on some target)
        1507,   # depth r=0.61 (trajectory model)
        1103,   # all targets near-zero (video pose SSL)
        2608,   # depth r=0.53, LR r=0.70 (pulse features)
        2988,   # LR r=0.16, depth r=0.45 (random Fourier features)
        2667,   # angle r=0.00, depth r=0.25 (very diverse)

        # Moderate diversity (r 0.5-0.8)
        1557,   # depth r=0.74, LR r=0.72 (temporal CNN)
        2559,   # depth r=0.58, LR r=0.55
        2870,   # depth r=0.51, LR r=0.72
        2887,   # depth r=0.53, LR r=0.72
        2784,   # depth r=0.78, LR r=0.50
        2602,   # depth r=0.67, LR r=0.72
        2780,   # depth r=0.73, LR r=0.70
        2741,   # depth r=0.78, LR r=0.71

        # Mild diversity (r 0.8-0.95) - may still contribute
        2832,   # depth r=0.90, LR r=0.86 (LASSO k-NN)
        2704,   # depth r=0.84, LR r=0.82
        2679,   # depth r=0.82, LR r=0.88
        2742,   # depth r=0.88, LR r=0.88
        2855,   # depth r=0.90, LR r=0.87
        2987,   # depth r=0.86, LR r=0.75
        2949,   # depth r=0.74, LR r=0.83

        # High-quality close subs (r > 0.95 but known good)
        2502,   # extended physics standalone
        2508,   # similar pipeline variant
    ]

    # Also include some subs from 1000-1600 range known to be diverse
    legacy_diverse = [
        662,    # depth r=-0.57, LR r=-0.27 (most diverse depth!)
        653,    # depth r=-0.22
        963,    # angle r=-0.19
    ]

    all_candidate_nums = sorted(set(anchor_subs + diverse_subs + legacy_diverse))
    print(f"\n  Candidate pool: {len(all_candidate_nums)} submissions")

    # ============================================================
    # STEP 2: Load all submissions
    # ============================================================
    print("\n  Loading submissions...")
    subs = load_all_submissions(all_candidate_nums)
    loaded_nums = sorted(subs.keys())
    print(f"  Successfully loaded: {len(loaded_nums)} submissions")
    missing = set(all_candidate_nums) - set(loaded_nums)
    if missing:
        print(f"  Missing: {sorted(missing)}")

    # Get IDs from reference submission
    ids = subs[2716]['id'].values

    # ============================================================
    # STEP 3: Compute per-target correlations
    # ============================================================
    print("\n  Computing correlation matrices...")
    corr, data = compute_per_target_correlations(subs, loaded_nums)

    # ============================================================
    # STEP 4: Greedy diverse subset selection
    # ============================================================
    print("\n  Greedy diverse subset selection...")
    base_idx = loaded_nums.index(2716)

    for ti, t in enumerate(TARGETS):
        selected = greedy_diverse_selection(corr, loaded_nums, base_idx, max_subs=15, target=t)
        selected_nums = [loaded_nums[s] for s in selected]
        print(f"  {TARGET_NAMES[ti]:>6}: {selected_nums}")

    # ============================================================
    # STEP 5: Estimate per-target MSE
    # ============================================================
    print("\n" + "=" * 70)
    print("PER-TARGET MSE ESTIMATION")
    print("=" * 70)

    per_target_mse = estimate_per_target_mse(KNOWN_LB_SCORES, loaded_nums, corr, data)

    print("\n  Per-target MSE estimates (x1000):")
    print(f"  {'Sub':>6} | {'Total':>8} | {'angle':>8} {'depth':>8} {'LR':>8}")
    print("  " + "-" * 50)
    for n in sorted(per_target_mse.keys()):
        total = KNOWN_LB_SCORES[n]
        a = per_target_mse[n][TARGETS[0]]
        d = per_target_mse[n][TARGETS[1]]
        lr = per_target_mse[n][TARGETS[2]]
        print(f"  {n:>6} | {total*1000:>8.4f} | {a*1000:>8.4f} {d*1000:>8.4f} {lr*1000:>8.4f}")

    # For unknown-score submissions, estimate MSE from correlation with known ones
    # Approach: if a sub is highly correlated with a known sub, its MSE is similar
    # More diverse subs likely have HIGHER MSE (they are different for a reason)
    # Use conservative estimate: MSE proportional to (1 + diversity_penalty)
    print("\n  Estimating MSE for unknown-score submissions...")
    all_mse_per_target = {}
    for n in loaded_nums:
        if n in per_target_mse:
            all_mse_per_target[n] = per_target_mse[n]
        else:
            idx = loaded_nums.index(n)
            all_mse_per_target[n] = {}
            for t in TARGETS:
                # Find most correlated known-score submission
                best_corr_val = -1
                best_known = None
                for kn in per_target_mse:
                    kidx = loaded_nums.index(kn)
                    r = abs(corr[t][idx, kidx])
                    if r > best_corr_val:
                        best_corr_val = r
                        best_known = kn

                # Estimate: MSE scales inversely with correlation
                # High correlation -> similar MSE
                # Low correlation -> assume worse (conservative)
                base_mse = per_target_mse[best_known][t]
                # Penalty: diversity suggests this sub captures different signal
                # but also may be noisier
                if best_corr_val > 0.95:
                    penalty = 1.0
                elif best_corr_val > 0.8:
                    penalty = 1.5
                elif best_corr_val > 0.5:
                    penalty = 2.5
                else:
                    penalty = 5.0
                all_mse_per_target[n][t] = base_mse * penalty

    # ============================================================
    # STEP 6: Per-target optimization
    # ============================================================
    print("\n" + "=" * 70)
    print("PER-TARGET BLEND OPTIMIZATION")
    print("=" * 70)

    best_blends = {}  # {target: {sub_num: weight}}
    best_preds = {}   # {target: prediction_array}

    for ti, t in enumerate(TARGETS):
        print(f"\n--- {TARGET_NAMES[ti].upper()} ---")
        corr_t = corr[t]
        data_t = data[t]

        # Select diverse subset for this target
        selected_indices = greedy_diverse_selection(
            corr, loaded_nums, base_idx, max_subs=20, target=t
        )
        selected_nums = [loaded_nums[s] for s in selected_indices]

        # Get MSE values for selected submissions
        mse_values = [all_mse_per_target[n][t] for n in selected_nums]

        print(f"  Selected {len(selected_nums)} subs for optimization")
        print(f"  MSE range: {min(mse_values)*1000:.4f} - {max(mse_values)*1000:.4f} (x1000)")

        # Method 1: Full N-way optimizer
        print(f"\n  [Method 1] N-way scipy optimizer...")
        w_full, mse_full = optimize_blend_weights(
            mse_values, corr_t, selected_indices, max_nonzero=7
        )
        active_full = [(selected_nums[i], w_full[i]) for i in range(len(w_full)) if w_full[i] > 0.005]
        active_full.sort(key=lambda x: -x[1])
        print(f"  Estimated MSE: {mse_full*1000:.6f} (x1000)")
        for num, w in active_full:
            lb_str = f"  LB={KNOWN_LB_SCORES[num]:.6f}" if num in KNOWN_LB_SCORES else ""
            print(f"    Sub {num:>5}: w={w:.4f}{lb_str}")

        # Method 2: 2-way grid search (all pairs from known-score subs)
        print(f"\n  [Method 2] 2-way grid search...")
        known_in_pool = [n for n in selected_nums if n in KNOWN_LB_SCORES]
        mse_dict_2way = {n: all_mse_per_target[n][t] for n in loaded_nums}
        results_2way = grid_search_2way(t, loaded_nums, data_t, mse_dict_2way, corr_t)
        if results_2way:
            top2 = results_2way[0]
            print(f"  Best 2-way: Sub {top2['subs'][0]} ({top2['weights'][0]:.2f}) + "
                  f"Sub {top2['subs'][1]} ({top2['weights'][1]:.2f}), "
                  f"est_MSE={top2['est_mse']*1000:.6f}, rho={top2['rho']:.4f}")
            for r in results_2way[:5]:
                print(f"    {r['subs']}: w=({r['weights'][0]:.2f},{r['weights'][1]:.2f}), "
                      f"mse={r['est_mse']*1000:.6f}, rho={r['rho']:.4f}")

        # Method 3: 3-way grid search
        print(f"\n  [Method 3] 3-way grid search...")
        results_3way = grid_search_3way(t, loaded_nums, data_t, mse_dict_2way, corr_t)
        if results_3way:
            top3 = results_3way[0]
            print(f"  Best 3-way: Subs {top3['subs']}, w={top3['weights']}, "
                  f"est_MSE={top3['est_mse']*1000:.6f}")
            for r in results_3way[:5]:
                print(f"    {r['subs']}: w=({r['weights'][0]:.2f},{r['weights'][1]:.2f},{r['weights'][2]:.2f}), "
                      f"mse={r['est_mse']*1000:.6f}")

        # Store best blend
        blend_dict = {num: w for num, w in active_full}
        best_blends[t] = blend_dict
        best_preds[t] = build_blended_predictions(data_t, loaded_nums, blend_dict)

    # ============================================================
    # STEP 7: Generate Frankenstein submissions
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING FRANKENSTEIN SUBMISSIONS")
    print("=" * 70)

    submission_log = []

    # --- Strategy 1: Pure optimizer output (each target independent) ---
    label = "pure_optimizer"
    sub_num = save_frankenstein(
        best_preds[TARGETS[0]], best_preds[TARGETS[1]], best_preds[TARGETS[2]],
        ids, label
    )
    print(f"\n  Sub {sub_num}: {label}")
    for ti, t in enumerate(TARGETS):
        active = sorted(best_blends[t].items(), key=lambda x: -x[1])
        desc = " + ".join(f"{w:.3f}*{n}" for n, w in active if w > 0.005)
        print(f"    {TARGET_NAMES[ti]}: {desc}")
    submission_log.append((sub_num, label, dict(best_blends)))

    # --- Strategy 2: Conservative (90% Sub 2716 + 10% optimizer) ---
    label = "conservative_90pct_2716"
    idx_2716 = loaded_nums.index(2716)
    angle_cons = 0.90 * data[TARGETS[0]][:, idx_2716] + 0.10 * best_preds[TARGETS[0]]
    depth_cons = 0.90 * data[TARGETS[1]][:, idx_2716] + 0.10 * best_preds[TARGETS[1]]
    lr_cons = 0.90 * data[TARGETS[2]][:, idx_2716] + 0.10 * best_preds[TARGETS[2]]
    sub_num = save_frankenstein(angle_cons, depth_cons, lr_cons, ids, label)
    print(f"\n  Sub {sub_num}: {label}")
    submission_log.append((sub_num, label, "90% Sub 2716 + 10% optimizer per target"))

    # --- Strategy 3: Conservative (80/20) ---
    label = "conservative_80pct_2716"
    angle_c2 = 0.80 * data[TARGETS[0]][:, idx_2716] + 0.20 * best_preds[TARGETS[0]]
    depth_c2 = 0.80 * data[TARGETS[1]][:, idx_2716] + 0.20 * best_preds[TARGETS[1]]
    lr_c2 = 0.80 * data[TARGETS[2]][:, idx_2716] + 0.20 * best_preds[TARGETS[2]]
    sub_num = save_frankenstein(angle_c2, depth_c2, lr_c2, ids, label)
    print(f"\n  Sub {sub_num}: {label}")
    submission_log.append((sub_num, label, "80% Sub 2716 + 20% optimizer per target"))

    # --- Strategy 4: Best 2-way per target Frankenstein ---
    print(f"\n  Building best-2-way Frankenstein...")
    frank_2way = {}
    for ti, t in enumerate(TARGETS):
        mse_dict_2way = {n: all_mse_per_target[n][t] for n in loaded_nums}
        results_2way = grid_search_2way(t, loaded_nums, data[t], mse_dict_2way, corr[t])
        if results_2way:
            top = results_2way[0]
            n1, n2 = top['subs']
            w1, w2 = top['weights']
            idx1 = loaded_nums.index(n1)
            idx2 = loaded_nums.index(n2)
            pred = w1 * data[t][:, idx1] + w2 * data[t][:, idx2]
            frank_2way[t] = pred
            print(f"    {TARGET_NAMES[ti]}: {w1:.2f}*Sub{n1} + {w2:.2f}*Sub{n2} "
                  f"(est_mse={top['est_mse']*1000:.6f})")
        else:
            frank_2way[t] = data[t][:, idx_2716]

    label = "best_2way_frankenstein"
    sub_num = save_frankenstein(
        frank_2way[TARGETS[0]], frank_2way[TARGETS[1]], frank_2way[TARGETS[2]],
        ids, label
    )
    print(f"  Sub {sub_num}: {label}")
    submission_log.append((sub_num, label, "Best 2-way blend per target"))

    # --- Strategy 5: Best 3-way per target Frankenstein ---
    print(f"\n  Building best-3-way Frankenstein...")
    frank_3way = {}
    for ti, t in enumerate(TARGETS):
        mse_dict_3way = {n: all_mse_per_target[n][t] for n in loaded_nums}
        results_3way = grid_search_3way(t, loaded_nums, data[t], mse_dict_3way, corr[t])
        if results_3way:
            top = results_3way[0]
            n1, n2, n3 = top['subs']
            w1, w2, w3 = top['weights']
            idx1 = loaded_nums.index(n1)
            idx2 = loaded_nums.index(n2)
            idx3 = loaded_nums.index(n3)
            pred = w1 * data[t][:, idx1] + w2 * data[t][:, idx2] + w3 * data[t][:, idx3]
            frank_3way[t] = pred
            print(f"    {TARGET_NAMES[ti]}: {w1:.2f}*Sub{n1} + {w2:.2f}*Sub{n2} + {w3:.2f}*Sub{n3} "
                  f"(est_mse={top['est_mse']*1000:.6f})")
        else:
            frank_3way[t] = data[t][:, idx_2716]

    label = "best_3way_frankenstein"
    sub_num = save_frankenstein(
        frank_3way[TARGETS[0]], frank_3way[TARGETS[1]], frank_3way[TARGETS[2]],
        ids, label
    )
    print(f"  Sub {sub_num}: {label}")
    submission_log.append((sub_num, label, "Best 3-way blend per target"))

    # --- Strategy 6: LR-focused Frankenstein ---
    # LR has the most headroom and genuine diversity opportunity
    print(f"\n  Building LR-focused Frankenstein...")
    # For angle and depth, use Sub 2716 (safest)
    # For LR, use the optimizer result
    label = "lr_focused_frankenstein"
    sub_num = save_frankenstein(
        data[TARGETS[0]][:, idx_2716],
        data[TARGETS[1]][:, idx_2716],
        best_preds[TARGETS[2]],
        ids, label
    )
    print(f"  Sub {sub_num}: {label} (angle+depth from 2716, LR from optimizer)")
    submission_log.append((sub_num, label, "Angle+depth from 2716, LR from optimizer"))

    # --- Strategy 7: Depth-focused Frankenstein ---
    # Sub 1507 has extreme depth diversity (r=0.61)
    print(f"\n  Building depth-focused Frankenstein...")
    label = "depth_focused_frankenstein"
    sub_num = save_frankenstein(
        data[TARGETS[0]][:, idx_2716],
        best_preds[TARGETS[1]],
        data[TARGETS[2]][:, idx_2716],
        ids, label
    )
    print(f"  Sub {sub_num}: {label} (angle+LR from 2716, depth from optimizer)")
    submission_log.append((sub_num, label, "Angle+LR from 2716, depth from optimizer"))

    # --- Strategy 8: Small perturbation blends ---
    # Try blending Sub 2716 with each diverse source at small weights
    print(f"\n  Building small-perturbation blends...")
    diverse_to_try = [1507, 1557, 2608, 2988, 2832, 2949, 2559, 2784]
    for div_sub in diverse_to_try:
        if div_sub not in subs:
            continue
        div_idx = loaded_nums.index(div_sub)
        for w_div in [0.05, 0.10]:
            label = f"perturb_{w_div:.0%}_{div_sub}"
            preds_a = (1 - w_div) * data[TARGETS[0]][:, idx_2716] + w_div * data[TARGETS[0]][:, div_idx]
            preds_d = (1 - w_div) * data[TARGETS[1]][:, idx_2716] + w_div * data[TARGETS[1]][:, div_idx]
            preds_l = (1 - w_div) * data[TARGETS[2]][:, idx_2716] + w_div * data[TARGETS[2]][:, div_idx]
            sub_num = save_frankenstein(preds_a, preds_d, preds_l, ids, label)
            submission_log.append((sub_num, label, f"{(1-w_div)*100:.0f}% Sub 2716 + {w_div*100:.0f}% Sub {div_sub}"))

    # --- Strategy 9: Per-target best diverse blend at small weights ---
    print(f"\n  Building per-target diverse blends...")
    # For each target, find the most diverse sub with reasonable quality
    # and blend at 5-15%
    for w_div in [0.05, 0.10, 0.15]:
        blend_desc = {}
        preds_all = {}
        for ti, t in enumerate(TARGETS):
            # Find most diverse sub for this target
            best_div_sub = None
            best_div_r = 1.0
            for n in loaded_nums:
                if n == 2716:
                    continue
                idx_n = loaded_nums.index(n)
                r = abs(corr[t][idx_2716, idx_n])
                # Only consider subs with some minimal quality
                # (not completely uncorrelated - those are likely garbage)
                if 0.3 < r < best_div_r:
                    best_div_r = r
                    best_div_sub = n
            if best_div_sub is None:
                best_div_sub = 2503  # fallback
                best_div_r = corr[t][idx_2716, loaded_nums.index(2503)]

            div_idx = loaded_nums.index(best_div_sub)
            preds_all[t] = ((1 - w_div) * data[t][:, idx_2716] +
                            w_div * data[t][:, div_idx])
            blend_desc[TARGET_NAMES[ti]] = f"{best_div_sub}(r={best_div_r:.2f})"

        label = f"per_target_diverse_{w_div:.0%}"
        sub_num = save_frankenstein(
            preds_all[TARGETS[0]], preds_all[TARGETS[1]], preds_all[TARGETS[2]],
            ids, label
        )
        desc = ", ".join(f"{k}={v}" for k, v in blend_desc.items())
        print(f"  Sub {sub_num}: {label} [{desc}]")
        submission_log.append((sub_num, label, f"{(1-w_div)*100:.0f}% Sub 2716 + {w_div*100:.0f}% per-target diverse: {desc}"))

    # --- Strategy 10: Cascade blend (blend optimizer output with Sub 2716 at various weights) ---
    print(f"\n  Building cascade blends...")
    for w_opt in [0.05, 0.15, 0.25, 0.30, 0.50]:
        label = f"cascade_{w_opt:.0%}_optimizer"
        preds_a = (1 - w_opt) * data[TARGETS[0]][:, idx_2716] + w_opt * best_preds[TARGETS[0]]
        preds_d = (1 - w_opt) * data[TARGETS[1]][:, idx_2716] + w_opt * best_preds[TARGETS[1]]
        preds_l = (1 - w_opt) * data[TARGETS[2]][:, idx_2716] + w_opt * best_preds[TARGETS[2]]
        sub_num = save_frankenstein(preds_a, preds_d, preds_l, ids, label)
        print(f"  Sub {sub_num}: {label}")
        submission_log.append((sub_num, label, f"{(1-w_opt)*100:.0f}% Sub 2716 + {w_opt*100:.0f}% optimizer"))

    # --- Strategy 11: Column cherry-pick from known-quality submissions ---
    # The purest Frankenstein: for each target, pick the column from whichever
    # known-quality sub performs best on that target.
    # Since we don't know per-target LB, try all combinations of known-quality subs.
    print(f"\n  Building column cherry-pick Frankensteins from known-quality subs...")
    known_sorted = sorted(KNOWN_LB_SCORES.keys(), key=lambda n: KNOWN_LB_SCORES[n])
    top_known = known_sorted[:6]  # Top 6 by LB score

    # Exhaustive per-target cherry-pick: try each known sub for each target
    # That's 6^3 = 216 combos, but many are identical to existing subs
    # Focus on combos where at least one target differs from 2716
    cherry_pick_combos = set()
    for a_sub in top_known:
        for d_sub in top_known:
            for l_sub in top_known:
                # Skip if all same as 2716
                if a_sub == 2716 and d_sub == 2716 and l_sub == 2716:
                    continue
                # Skip if all same sub (that's just that sub)
                if a_sub == d_sub == l_sub:
                    continue
                cherry_pick_combos.add((a_sub, d_sub, l_sub))

    # Score each combo by estimated total MSE
    cherry_results = []
    for a_sub, d_sub, l_sub in cherry_pick_combos:
        est_mse = (all_mse_per_target[a_sub][TARGETS[0]] +
                   all_mse_per_target[d_sub][TARGETS[1]] +
                   all_mse_per_target[l_sub][TARGETS[2]]) / 3.0
        cherry_results.append((a_sub, d_sub, l_sub, est_mse))

    cherry_results.sort(key=lambda x: x[3])
    print(f"  Total cherry-pick combos: {len(cherry_results)}")
    print(f"  Top 10 by estimated MSE:")
    for a, d, l, mse in cherry_results[:10]:
        print(f"    angle=Sub{a}, depth=Sub{d}, LR=Sub{l}, est_MSE={mse*1000:.4f}")

    # Save top 5 cherry-pick combos
    for i, (a_sub, d_sub, l_sub, est_mse) in enumerate(cherry_results[:5]):
        label = f"cherry_pick_{i+1}_a{a_sub}_d{d_sub}_l{l_sub}"
        a_idx = loaded_nums.index(a_sub)
        d_idx = loaded_nums.index(d_sub)
        l_idx = loaded_nums.index(l_sub)
        sub_num = save_frankenstein(
            data[TARGETS[0]][:, a_idx],
            data[TARGETS[1]][:, d_idx],
            data[TARGETS[2]][:, l_idx],
            ids, label
        )
        print(f"  Sub {sub_num}: {label} (est_MSE={est_mse*1000:.4f})")
        submission_log.append((sub_num, label,
            f"angle=Sub{a_sub}(LB={KNOWN_LB_SCORES.get(a_sub,'?')}), "
            f"depth=Sub{d_sub}(LB={KNOWN_LB_SCORES.get(d_sub,'?')}), "
            f"LR=Sub{l_sub}(LB={KNOWN_LB_SCORES.get(l_sub,'?')})"))

    # --- Strategy 12: Known-quality-only 2-way blend per target ---
    # Only blend known-LB-score subs (most reliable surrogate)
    print(f"\n  Building known-quality 2-way blends per target...")
    known_nums_list = sorted(KNOWN_LB_SCORES.keys())
    for w_blend in [0.05, 0.10, 0.15, 0.20]:
        frank_known_2way = {}
        desc_parts = {}
        for ti, t in enumerate(TARGETS):
            # Find best non-2716 known sub to blend with 2716 for this target
            best_partner = None
            best_est_mse = float('inf')
            for n in known_nums_list:
                if n == 2716:
                    continue
                idx_n = loaded_nums.index(n)
                r = corr[t][idx_2716, idx_n]
                m_2716 = all_mse_per_target[2716][t]
                m_n = all_mse_per_target[n][t]
                # MSE of blend
                est = ((1-w_blend)**2 * m_2716 + w_blend**2 * m_n +
                       2 * (1-w_blend) * w_blend * r * np.sqrt(m_2716 * m_n))
                if est < best_est_mse:
                    best_est_mse = est
                    best_partner = n

            idx_partner = loaded_nums.index(best_partner)
            frank_known_2way[t] = ((1 - w_blend) * data[t][:, idx_2716] +
                                    w_blend * data[t][:, idx_partner])
            desc_parts[TARGET_NAMES[ti]] = f"Sub{best_partner}"

        label = f"known_2way_{w_blend:.0%}_per_target"
        desc = ", ".join(f"{k}={v}" for k, v in desc_parts.items())
        sub_num = save_frankenstein(
            frank_known_2way[TARGETS[0]],
            frank_known_2way[TARGETS[1]],
            frank_known_2way[TARGETS[2]],
            ids, label
        )
        print(f"  Sub {sub_num}: {label} [{desc}]")
        submission_log.append((sub_num, label,
            f"{(1-w_blend)*100:.0f}% Sub 2716 + {w_blend*100:.0f}% best known per target: {desc}"))

    # --- Strategy 13: Hedged blend of top N known-quality subs ---
    print(f"\n  Building hedged blends of top known-quality subs...")
    for n_top in [3, 5, 7]:
        top_n = known_sorted[:n_top]
        # Inverse-MSE weighting
        inv_mses = [1.0 / KNOWN_LB_SCORES[n] for n in top_n]
        total_inv = sum(inv_mses)
        weights_hedge = [im / total_inv for im in inv_mses]

        preds_hedge = {t: np.zeros(113) for t in TARGETS}
        for j, n in enumerate(top_n):
            idx_n = loaded_nums.index(n)
            for t in TARGETS:
                preds_hedge[t] += weights_hedge[j] * data[t][:, idx_n]

        label = f"hedged_top{n_top}_inv_mse"
        sub_num = save_frankenstein(
            preds_hedge[TARGETS[0]], preds_hedge[TARGETS[1]], preds_hedge[TARGETS[2]],
            ids, label
        )
        wt_desc = ", ".join(f"Sub{n}:{w:.3f}" for n, w in zip(top_n, weights_hedge))
        print(f"  Sub {sub_num}: {label} [{wt_desc}]")
        submission_log.append((sub_num, label, f"Inv-MSE weighted top {n_top}: {wt_desc}"))

    # --- Strategy 14: Per-target known-sub optimizer (SLSQP on known subs only) ---
    print(f"\n  Building known-sub-only optimizer per target...")
    known_indices = [loaded_nums.index(n) for n in known_nums_list]
    known_mse_per_target = {n: all_mse_per_target[n] for n in known_nums_list}

    frank_known_opt = {}
    for ti, t in enumerate(TARGETS):
        mse_vals = [known_mse_per_target[n][t] for n in known_nums_list]
        w_opt, est_mse = optimize_blend_weights(
            mse_vals, corr[t], known_indices, max_nonzero=5
        )
        active = [(known_nums_list[i], w_opt[i]) for i in range(len(w_opt)) if w_opt[i] > 0.005]
        active.sort(key=lambda x: -x[1])
        print(f"  {TARGET_NAMES[ti]}: est_MSE={est_mse*1000:.6f}")
        for num, w in active:
            print(f"    Sub {num}: {w:.4f} (LB={KNOWN_LB_SCORES[num]:.6f})")

        blend_dict = {num: w for num, w in active}
        frank_known_opt[t] = build_blended_predictions(data[t], loaded_nums, blend_dict)

    label = "known_only_optimizer"
    sub_num = save_frankenstein(
        frank_known_opt[TARGETS[0]], frank_known_opt[TARGETS[1]], frank_known_opt[TARGETS[2]],
        ids, label
    )
    print(f"  Sub {sub_num}: {label}")
    submission_log.append((sub_num, label, "Optimizer using only known-LB-score submissions"))

    # Conservative version: 90% Sub 2716 + 10% known-only optimizer
    label = "known_opt_conservative"
    sub_num = save_frankenstein(
        0.90 * data[TARGETS[0]][:, idx_2716] + 0.10 * frank_known_opt[TARGETS[0]],
        0.90 * data[TARGETS[1]][:, idx_2716] + 0.10 * frank_known_opt[TARGETS[1]],
        0.90 * data[TARGETS[2]][:, idx_2716] + 0.10 * frank_known_opt[TARGETS[2]],
        ids, label
    )
    print(f"  Sub {sub_num}: {label}")
    submission_log.append((sub_num, label, "90% Sub 2716 + 10% known-only optimizer"))

    # --- Strategy 15: Surgical per-target tweaks ---
    # Based on the optimization results, make surgical changes to Sub 2716
    # only for the target where the optimizer found the most improvement
    print(f"\n  Building surgical single-target tweaks...")
    for ti, t in enumerate(TARGETS):
        for w_tweak in [0.05, 0.10, 0.15]:
            preds_tweak = {}
            for tj, t2 in enumerate(TARGETS):
                if tj == ti:
                    # Use optimizer blend for this target
                    preds_tweak[t2] = ((1 - w_tweak) * data[t2][:, idx_2716] +
                                       w_tweak * best_preds[t2])
                else:
                    # Keep Sub 2716 for other targets
                    preds_tweak[t2] = data[t2][:, idx_2716]

            label = f"surgical_{TARGET_NAMES[ti]}_{w_tweak:.0%}"
            sub_num = save_frankenstein(
                preds_tweak[TARGETS[0]], preds_tweak[TARGETS[1]], preds_tweak[TARGETS[2]],
                ids, label
            )
            submission_log.append((sub_num, label,
                f"Sub 2716 with {TARGET_NAMES[ti]} tweaked {w_tweak*100:.0f}% toward optimizer"))

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"\n  Generated {len(submission_log)} submissions:")
    for sub_num, label, desc in submission_log:
        print(f"    Sub {sub_num:>5}: {label}")

    # ============================================================
    # CORRELATION ANALYSIS: How different are the Frankensteins from Sub 2716?
    # ============================================================
    print("\n" + "=" * 70)
    print("DIVERSITY OF GENERATED SUBMISSIONS (vs Sub 2716)")
    print("=" * 70)
    base_df = subs[2716]
    print(f"  {'Sub':>6} {'Label':>35} | {'angle':>8} {'depth':>8} {'LR':>8}")
    print("  " + "-" * 75)
    for sub_num, label, _ in submission_log:
        try:
            gen_df = pd.read_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv")
            rs = {}
            for t in TARGETS:
                r = np.corrcoef(base_df[t].values, gen_df[t].values)[0, 1]
                rs[t] = r if not np.isnan(r) else 0.0
            short_label = label[:35]
            print(f"  {sub_num:>6} {short_label:>35} | {rs[TARGETS[0]]:>8.4f} {rs[TARGETS[1]]:>8.4f} {rs[TARGETS[2]]:>8.4f}")
        except Exception:
            pass

    # ============================================================
    # WRITE RESEARCH REPORT
    # ============================================================
    report_path = RESEARCH_DIR / "NWAY_BLEND_OPTIMIZER_RESULTS.md"
    with open(report_path, 'w') as f:
        f.write("# N-Way Per-Target Blend Optimizer Results\n\n")
        f.write(f"Date: 2026-02-16\n\n")
        f.write(f"## Objective\n")
        f.write(f"Build Frankenstein submissions by independently optimizing blends per target.\n")
        f.write(f"Current best: Sub 2716, LB 0.006343. Target: 0.0059.\n\n")

        f.write(f"## Candidate Pool\n")
        f.write(f"- {len(loaded_nums)} submissions loaded\n")
        f.write(f"- {len(KNOWN_LB_SCORES)} with known LB scores\n")
        f.write(f"- {len(diverse_subs)} diverse submissions from scan\n\n")

        f.write(f"## Per-Target Optimization Results\n\n")
        for ti, t in enumerate(TARGETS):
            f.write(f"### {TARGET_NAMES[ti]}\n")
            active = sorted(best_blends[t].items(), key=lambda x: -x[1])
            for num, w in active:
                lb = f" (LB={KNOWN_LB_SCORES[num]:.6f})" if num in KNOWN_LB_SCORES else ""
                f.write(f"- Sub {num}: weight={w:.4f}{lb}\n")
            f.write("\n")

        f.write(f"## Generated Submissions\n\n")
        f.write(f"| Sub | Strategy | Description |\n")
        f.write(f"|-----|----------|-------------|\n")
        for sub_num, label, desc in submission_log:
            desc_str = str(desc)[:80] if isinstance(desc, str) else str(desc)[:80]
            f.write(f"| {sub_num} | {label} | {desc_str} |\n")

        f.write(f"\n## Key Recommendations\n\n")
        f.write(f"1. Start with conservative blends (90% Sub 2716) for safety\n")
        f.write(f"2. The LR target has the most diversity headroom\n")
        f.write(f"3. Sub 1507 provides extreme depth diversity (r=0.61)\n")
        f.write(f"4. Sub 2988 provides extreme LR diversity (r=0.16)\n")
        f.write(f"5. Very diverse subs (r<0.3) are likely low quality - use at small weights only\n")

    print(f"\n  Report saved to: {report_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
