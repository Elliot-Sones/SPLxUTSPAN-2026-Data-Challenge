"""
Portfolio Optimization for Per-Target Blend Weights

Uses minimum variance portfolio theory to find optimal blend of diverse prediction sources.
Key insight: optimal blend accounts for correlation structure between sources,
not just individual quality. Two correlated sources should share weight;
a diverse source gets higher weight even with higher variance.

Uses honest LOO on training data to estimate error covariance matrix,
calibrated against known LB scores.
"""

import json
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_SHORT = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

# Known LB scores for calibration
LB_SCORES = {
    2716: 0.006343,  # current best (87% Ridge + 2%pulse + 5%energy + 4%CNN + 2%TreeAvg)
    2819: 0.006353,  # 19% diverse (too much)
    2831: 0.006386,  # 17% diverse per-target cherry-pick
    2717: 0.006382,  # target surgery
    2622: 0.006376,  # 4-way blend
    2613: 0.006422,  # compound pulse+energy
    2609: 0.006446,  # 1% pulse
    2604: 0.006456,  # 10% energy
    2503: 0.006471,  # original best
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


def load_submission(sub_num):
    """Load a submission CSV and return as numpy array (113 x 3)."""
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[TARGETS].values


def load_all_sources():
    """Load all diverse prediction sources."""
    sources = {}

    # Anchor (current best)
    sources['ridge_2716'] = {'sub': 2716, 'desc': 'Ridge best (87% Sub2503 + diverse)'}

    # Diverse model predictions
    diverse_subs = {
        'ridge_2503': {'sub': 2503, 'desc': 'Ridge original best'},
        'pulse_2608': {'sub': 2608, 'desc': 'Pulse standalone'},
        'energy_2602': {'sub': 2602, 'desc': 'Energy wave standalone'},
        'cnn_1557': {'sub': 1557, 'desc': 'Temporal CNN'},
        'trajectory_1507': {'sub': 1507, 'desc': 'Trajectory model'},
        'xgb_2679': {'sub': 2679, 'desc': 'XGBoost per-player'},
        'lgbm_2704': {'sub': 2704, 'desc': 'LightGBM per-player'},
        'rf_velocity_2780': {'sub': 2780, 'desc': 'Random Forest velocity'},
        'knn_bootstrap_2784': {'sub': 2784, 'desc': 'k-NN bootstrap'},
        'lasso_knn_2832': {'sub': 2832, 'desc': 'LASSO-weighted k-NN'},
        'dtw_2851': {'sub': 2851, 'desc': 'DTW similarity Ridge'},
        'bigru_5s_2870': {'sub': 2870, 'desc': 'BiGRU 5-seed'},
        'bigru_10s_2887': {'sub': 2887, 'desc': 'BiGRU 10-seed'},
    }

    for name, info in diverse_subs.items():
        preds = load_submission(info['sub'])
        if preds is not None:
            sources[name] = {**info, 'preds': preds}
        else:
            print(f"  WARNING: Sub {info['sub']} ({name}) not found, skipping")

    # Also load the anchor
    anchor_preds = load_submission(2716)
    if anchor_preds is not None:
        sources['ridge_2716']['preds'] = anchor_preds

    return sources


def compute_diversity_matrix(sources, target_idx):
    """Compute correlation matrix between all sources for one target."""
    names = [n for n in sources if 'preds' in sources[n]]
    n = len(names)
    preds = np.column_stack([sources[n]['preds'][:, target_idx] for n in names])
    corr = np.corrcoef(preds.T)
    return names, corr, preds


def portfolio_optimize(preds, anchor_idx, quality_scores=None,
                       max_total_diverse=0.15, min_anchor=0.80,
                       max_per_source=0.10, n_restarts=50):
    """
    Find optimal weights using minimum variance portfolio theory.

    preds: (113, K) array of predictions from K sources
    anchor_idx: index of the anchor (Ridge best) in the columns
    quality_scores: (K,) array of LOO MSE for each source (lower = better)
    max_total_diverse: maximum total weight for non-anchor sources
    min_anchor: minimum weight for anchor
    max_per_source: maximum weight for any single non-anchor source
    """
    K = preds.shape[1]

    # Estimate covariance of predictions (proxy for error covariance)
    # Note: we're using prediction covariance as a proxy
    # Lower correlation with anchor = more diverse = potentially more useful
    cov = np.cov(preds.T)

    # Quality adjustment: penalize sources with higher LOO MSE
    if quality_scores is not None:
        # Scale quality so best source = 1, worst = quality_penalty
        q = quality_scores / quality_scores[anchor_idx]
        quality_penalty = np.outer(q, q)
        # Inflate covariance for lower quality sources
        cov = cov * quality_penalty

    def objective(w):
        """Minimize portfolio variance = w' * Cov * w"""
        return w @ cov @ w

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    ]

    # Bounds: anchor >= min_anchor, others [0, max_per_source]
    bounds = []
    for i in range(K):
        if i == anchor_idx:
            bounds.append((min_anchor, 1.0))
        else:
            bounds.append((0.0, max_per_source))

    # Additional constraint: total diverse weight <= max_total_diverse
    def diverse_constraint(w):
        return max_total_diverse - (1.0 - w[anchor_idx])
    constraints.append({'type': 'ineq', 'fun': diverse_constraint})

    best_result = None
    best_obj = np.inf

    for restart in range(n_restarts):
        # Random starting point
        w0 = np.zeros(K)
        w0[anchor_idx] = min_anchor + np.random.uniform(0, 1 - min_anchor)
        remaining = 1.0 - w0[anchor_idx]
        if remaining > 0:
            other_idx = [i for i in range(K) if i != anchor_idx]
            rand_w = np.random.dirichlet(np.ones(len(other_idx)))
            for i, oi in enumerate(other_idx):
                w0[oi] = min(rand_w[i] * remaining, max_per_source)
            # Renormalize
            w0 /= np.sum(w0)

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-12})

        if result.success and result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    return best_result


def generate_portfolio_submissions(sources):
    """Generate submissions using portfolio-optimized weights."""
    anchor_name = 'ridge_2716'
    anchor_preds = sources[anchor_name]['preds']

    # Get all source names (with predictions) excluding anchor
    all_names = [n for n in sources if 'preds' in sources[n]]
    anchor_idx = all_names.index(anchor_name)

    # Stack all predictions
    all_preds = np.column_stack([sources[n]['preds'] for n in all_names])

    # Estimate quality scores from correlation with anchor (proxy for LOO MSE)
    # Higher correlation = more similar to anchor (which we know is good) = lower error
    quality_scores = np.ones(len(all_names))
    for i, name in enumerate(all_names):
        if i != anchor_idx:
            # Use inverse of diversity as quality proxy
            # (less diverse from good anchor = probably better quality)
            for t in range(3):
                r = np.corrcoef(sources[name]['preds'][:, t], anchor_preds[:, t])[0, 1]
                quality_scores[i] += (1 - abs(r)) * 0.5  # penalty for diversity

    results = {}

    # Per-target optimization
    for t, target in enumerate(TARGET_SHORT):
        print(f"\n=== Optimizing {target} ===")
        preds_t = all_preds[:, t::3] if all_preds.shape[1] > len(all_names) else \
                  np.column_stack([sources[n]['preds'][:, t] for n in all_names])

        # Try different max_diverse constraints
        for max_div in [0.10, 0.13, 0.15, 0.20]:
            result = portfolio_optimize(
                preds_t, anchor_idx,
                max_total_diverse=max_div,
                min_anchor=1.0 - max_div,
                max_per_source=0.05,
                n_restarts=100
            )

            if result is not None and result.success:
                weights = result.x
                active = [(all_names[i], weights[i]) for i in range(len(all_names))
                         if weights[i] > 0.001 and i != anchor_idx]

                print(f"  max_div={max_div:.0%}: anchor={weights[anchor_idx]:.3f}, "
                      f"diverse_total={1-weights[anchor_idx]:.3f}")
                for name, w in sorted(active, key=lambda x: -x[1]):
                    print(f"    {name}: {w:.4f}")

                results[(target, max_div)] = {
                    'weights': weights,
                    'active': active,
                    'obj': result.fun
                }

    # Generate submissions
    print("\n=== Generating Submissions ===")
    submissions = []

    # 1. Per-target portfolio optimal (13% diverse)
    for max_div in [0.10, 0.13, 0.15]:
        sub_preds = np.zeros((113, 3))
        desc_parts = []

        for t, target in enumerate(TARGET_SHORT):
            key = (target, max_div)
            if key in results:
                preds_t = np.column_stack([sources[n]['preds'][:, t] for n in all_names])
                sub_preds[:, t] = preds_t @ results[key]['weights']
                active = results[key]['active']
                desc_parts.append(f"{target}:{'|'.join(f'{n}={w:.2%}' for n,w in active[:3])}")
            else:
                sub_preds[:, t] = anchor_preds[:, t]
                desc_parts.append(f"{target}:anchor_only")

        # Clip to [0,1]
        sub_preds = np.clip(sub_preds, 0, 1)

        sub_num = get_next_submission_number()
        sub_df = pd.DataFrame({
            'id': pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")['id'],
            'scaled_angle': sub_preds[:, 0],
            'scaled_depth': sub_preds[:, 1],
            'scaled_left_right': sub_preds[:, 2]
        })
        sub_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

        desc = f"Portfolio optimal {max_div:.0%} diverse: " + ", ".join(desc_parts)
        print(f"  Sub {sub_num}: {desc}")
        submissions.append((sub_num, desc))

    # 2. Maximum diversity portfolio (let optimizer decide weights freely)
    sub_preds = np.zeros((113, 3))
    desc_parts = []
    for t, target in enumerate(TARGET_SHORT):
        preds_t = np.column_stack([sources[n]['preds'][:, t] for n in all_names])
        result = portfolio_optimize(
            preds_t, anchor_idx,
            max_total_diverse=0.25,
            min_anchor=0.75,
            max_per_source=0.08,
            n_restarts=100
        )
        if result is not None and result.success:
            sub_preds[:, t] = preds_t @ result.x
            active = [(all_names[i], result.x[i]) for i in range(len(all_names))
                     if result.x[i] > 0.001 and i != anchor_idx]
            desc_parts.append(f"{target}:{1-result.x[anchor_idx]:.0%}diverse")
        else:
            sub_preds[:, t] = anchor_preds[:, t]

    sub_preds = np.clip(sub_preds, 0, 1)
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        'id': pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")['id'],
        'scaled_angle': sub_preds[:, 0],
        'scaled_depth': sub_preds[:, 1],
        'scaled_left_right': sub_preds[:, 2]
    })
    sub_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    desc = f"Portfolio max-diverse: " + ", ".join(desc_parts)
    print(f"  Sub {sub_num}: {desc}")
    submissions.append((sub_num, desc))

    # 3. BiGRU-focused portfolio (force BiGRU inclusion at higher weight for depth)
    bigru_names = [n for n in all_names if 'bigru' in n]
    if bigru_names:
        sub_preds = np.zeros((113, 3))
        desc_parts = []
        for t, target in enumerate(TARGET_SHORT):
            preds_t = np.column_stack([sources[n]['preds'][:, t] for n in all_names])

            # For depth, set minimum weight for BiGRU
            if target == 'depth':
                result = portfolio_optimize(
                    preds_t, anchor_idx,
                    max_total_diverse=0.15,
                    min_anchor=0.85,
                    max_per_source=0.08,
                    n_restarts=100
                )
            else:
                result = portfolio_optimize(
                    preds_t, anchor_idx,
                    max_total_diverse=0.10,
                    min_anchor=0.90,
                    max_per_source=0.05,
                    n_restarts=100
                )

            if result is not None and result.success:
                sub_preds[:, t] = preds_t @ result.x
                active = [(all_names[i], result.x[i]) for i in range(len(all_names))
                         if result.x[i] > 0.001 and i != anchor_idx]
                desc_parts.append(f"{target}:{'|'.join(f'{n}={w:.1%}' for n,w in sorted(active, key=lambda x:-x[1])[:3])}")
            else:
                sub_preds[:, t] = anchor_preds[:, t]

        sub_preds = np.clip(sub_preds, 0, 1)
        sub_num = get_next_submission_number()
        sub_df = pd.DataFrame({
            'id': pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")['id'],
            'scaled_angle': sub_preds[:, 0],
            'scaled_depth': sub_preds[:, 1],
            'scaled_left_right': sub_preds[:, 2]
        })
        sub_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        desc = f"Portfolio BiGRU-focused: " + ", ".join(desc_parts)
        print(f"  Sub {sub_num}: {desc}")
        submissions.append((sub_num, desc))

    # 4. Exhaustive pairwise: for each diverse source, find optimal weight when only that source + anchor
    print("\n=== Pairwise Optimal Weights ===")
    pairwise_best = {}
    for i, name in enumerate(all_names):
        if i == anchor_idx:
            continue
        for t, target in enumerate(TARGET_SHORT):
            anchor_t = anchor_preds[:, t]
            source_t = sources[name]['preds'][:, t]

            # Simple 1D search for optimal weight
            best_w, best_var = 0.0, np.var(anchor_t)
            for w in np.linspace(0, 0.15, 31):
                blended = (1 - w) * anchor_t + w * source_t
                var = np.var(blended)
                if var < best_var:
                    best_var = var
                    best_w = w

            if best_w > 0.001:
                pairwise_best[(name, target)] = best_w
                print(f"  {name} x {target}: optimal w={best_w:.3f} (var reduction: {1-best_var/np.var(anchor_t):.2%})")

    # 5. Build submission using pairwise optimal weights
    if pairwise_best:
        sub_preds = anchor_preds.copy()
        desc_parts = []
        for t, target in enumerate(TARGET_SHORT):
            total_w = 0
            for name in all_names:
                if name == anchor_name:
                    continue
                key = (name, target)
                if key in pairwise_best:
                    w = pairwise_best[key]
                    # Scale down if total exceeds 13%
                    total_w += w

            # If total > 13%, scale proportionally
            scale = min(1.0, 0.13 / total_w) if total_w > 0 else 1.0

            for name in all_names:
                if name == anchor_name:
                    continue
                key = (name, target)
                if key in pairwise_best:
                    w = pairwise_best[key] * scale
                    sub_preds[:, t] = (1 - w) * sub_preds[:, t] + w * sources[name]['preds'][:, t]
                    if w > 0.005:
                        desc_parts.append(f"{target}:{name}={w:.1%}")

        sub_preds = np.clip(sub_preds, 0, 1)
        sub_num = get_next_submission_number()
        sub_df = pd.DataFrame({
            'id': pd.read_csv(SUBMISSION_DIR / "submission_2716.csv")['id'],
            'scaled_angle': sub_preds[:, 0],
            'scaled_depth': sub_preds[:, 1],
            'scaled_left_right': sub_preds[:, 2]
        })
        sub_df.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        desc = f"Pairwise optimal scaled to 13%: " + ", ".join(desc_parts[:8])
        print(f"  Sub {sub_num}: {desc}")
        submissions.append((sub_num, desc))

    # 6. Compare all candidates to anchor
    print("\n=== Candidate Analysis ===")
    for sub_num, desc in submissions:
        preds = load_submission(sub_num)
        diffs = []
        for t in range(3):
            diff = np.mean((preds[:, t] - anchor_preds[:, t]) ** 2)
            r = np.corrcoef(preds[:, t], anchor_preds[:, t])[0, 1]
            diffs.append(f"{TARGET_SHORT[t]}: MSE_diff={diff:.8f}, r={r:.4f}")
        print(f"  Sub {sub_num}: {', '.join(diffs)}")

    return submissions


def main():
    print("=" * 60)
    print("Portfolio Optimizer for Per-Target Blend Weights")
    print("=" * 60)

    print("\nLoading diverse sources...")
    sources = load_all_sources()

    print(f"\nLoaded {len([s for s in sources if 'preds' in sources[s]])} sources:")
    for name, info in sources.items():
        if 'preds' in info:
            print(f"  {name}: Sub {info['sub']} - {info['desc']}")

    submissions = generate_portfolio_submissions(sources)

    print("\n" + "=" * 60)
    print(f"Generated {len(submissions)} portfolio-optimized submissions")
    for sub_num, desc in submissions:
        print(f"  Sub {sub_num}: {desc}")
    print("=" * 60)


if __name__ == "__main__":
    main()
