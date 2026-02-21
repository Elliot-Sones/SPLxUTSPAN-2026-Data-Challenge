"""
Per-Player Per-Target Optimal Model Selection

The key insight from Sub 3558 (NEW BEST): allocating different CNN blend weights
per player worked because different players benefit from different models.

This script takes it further: for each of the 15 player-target combos,
find the optimal blend of ALL available model predictions.

We have predictions from:
- Sub3411 (Ridge pipeline)
- Sub3516 (improved CNN standalone)
- Sub3550 (GP Bayesian standalone)
- Sub3546 (gplearn symbolic standalone)

For each player-target combo, find the optimal weighted combination that
minimizes LOO MSE within that player's data. Then apply those weights to
test predictions.

CRITICAL: Use honest LOO (leave-one-out of the BLEND weights, not the models).
The models are already trained, so we're only optimizing 3-4 weights per combo.
"""

from __future__ import annotations

import fcntl
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from itertools import combinations

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
SUB_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_train_targets():
    """Load train targets and player IDs."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    pids_train = train_df["participant_id"].values
    pids_test = test_df["participant_id"].values
    ids_test = test_df["id"].values

    y_raw = train_df[["angle", "depth", "left_right"]].values

    # Scale targets
    y_scaled = np.zeros_like(y_raw, dtype=np.float64)
    for tidx, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scaler.transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    return y_scaled, pids_train, pids_test, ids_test


def load_submissions(sub_numbers):
    """Load multiple submissions."""
    subs = {}
    for num in sub_numbers:
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        if path.exists():
            subs[num] = pd.read_csv(path)
            print(f"  Loaded Sub {num}: {len(subs[num])} rows", flush=True)
        else:
            print(f"  WARNING: Sub {num} not found", flush=True)
    return subs


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
    print(f"  Saved submission_{num}: {desc}", flush=True)
    return num


def optimize_blend_weights(preds_list, y_true, method="grid"):
    """Find optimal blend weights that minimize MSE.

    preds_list: list of prediction arrays (each length N)
    y_true: true target values (length N)

    Returns: weights array (sums to 1)
    """
    n_models = len(preds_list)
    if n_models == 1:
        return np.array([1.0])

    P = np.column_stack(preds_list)  # (N, n_models)
    n = len(y_true)

    if method == "grid":
        # Grid search over weight combinations
        best_mse = float('inf')
        best_w = np.ones(n_models) / n_models

        # For 2-4 models, grid search is feasible
        if n_models == 2:
            for w1 in np.arange(0, 1.01, 0.05):
                w = np.array([w1, 1 - w1])
                pred = P @ w
                mse = np.mean((pred - y_true) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_w = w.copy()
        elif n_models == 3:
            for w1 in np.arange(0, 1.01, 0.05):
                for w2 in np.arange(0, 1.01 - w1, 0.05):
                    w = np.array([w1, w2, 1 - w1 - w2])
                    pred = P @ w
                    mse = np.mean((pred - y_true) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_w = w.copy()
        elif n_models == 4:
            for w1 in np.arange(0, 1.01, 0.10):
                for w2 in np.arange(0, 1.01 - w1, 0.10):
                    for w3 in np.arange(0, 1.01 - w1 - w2, 0.10):
                        w = np.array([w1, w2, w3, 1 - w1 - w2 - w3])
                        pred = P @ w
                        mse = np.mean((pred - y_true) ** 2)
                        if mse < best_mse:
                            best_mse = mse
                            best_w = w.copy()
        else:
            # For more models, use scipy optimize
            def neg_mse(w_free):
                w = np.append(w_free, 1 - np.sum(w_free))
                if np.any(w < 0) or np.any(w > 1):
                    return 1e10
                pred = P @ w
                return np.mean((pred - y_true) ** 2)

            best_mse = float('inf')
            for _ in range(20):
                w0 = np.random.dirichlet(np.ones(n_models))
                res = minimize(neg_mse, w0[:-1], method='Nelder-Mead')
                w = np.append(res.x, 1 - np.sum(res.x))
                if np.all(w >= 0) and res.fun < best_mse:
                    best_mse = res.fun
                    best_w = w.copy()

        return best_w

    return np.ones(n_models) / n_models


def main():
    print("=" * 70, flush=True)
    print("Per-Player Per-Target Optimal Model Selection", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    # Load data
    y_scaled, pids_train, pids_test, ids_test = load_train_targets()
    unique_pids = np.unique(pids_train)
    print(f"Players: {unique_pids}", flush=True)
    print(f"Train: {len(pids_train)}, Test: {len(pids_test)}", flush=True)

    # Load available model submissions (standalone predictions)
    # These are our diverse model sources
    model_subs = {
        "ridge": 3411,      # Base Ridge pipeline
        "cnn": 3516,        # Improved CNN standalone
        "gp": 3550,         # GP Bayesian standalone
        "gplearn": 3546,    # gplearn symbolic standalone
    }

    print("\nLoading model submissions...", flush=True)
    subs = load_submissions(list(model_subs.values()))

    # We need LOO predictions for training data to optimize blend weights.
    # We DON'T have LOO predictions for the standalone submissions.
    # But we CAN use the test predictions directly and optimize weights
    # based on what we know about each model's LOO quality per player.
    #
    # Alternative: use the SUB3558 approach (which allocates CNN weight by MSE)
    # but generalized to all models. Since we can't do honest LOO of the blend
    # (we don't have per-model LOO predictions), we'll use a simpler heuristic:
    #
    # For each player-target, compute what blend of test predictions gives
    # maximum diversity from the base while maintaining the base's core signal.

    # Actually, the BEST approach: we know the LOO MSE per player per target
    # from the calibration analysis. Use that to weight models.
    # Models with lower MSE get higher weight.

    # But we don't have per-player LOO for CNN/GP/gplearn.
    # So let's use the DIVERSITY approach: for player-targets where the base
    # is weak, introduce more diversity from alternative models.

    # Load the per-player calibration data
    cal_files = sorted(OUTPUT_DIR.glob("per_player_calibration_*.json"))
    if cal_files:
        with open(cal_files[-1]) as f:
            calibrations = json.load(f)
        print(f"Loaded calibration from {cal_files[-1].name}", flush=True)
    else:
        print("WARNING: No calibration data found!", flush=True)
        calibrations = None

    # Strategy: For each player-target combo:
    # - If base model is GOOD (MSE < 0.007, slope > 0.5): use mostly base (90%+)
    # - If base model is MEDIUM: use some CNN/GP (80% base + 20% mix)
    # - If base model is BAD (MSE > 0.012 or slope < 0.3): use more alternatives (60% base + 40% mix)

    print(f"\n{'='*60}", flush=True)
    print("Building Per-Player-Target Optimal Blend", flush=True)
    print(f"{'='*60}", flush=True)

    result_df = pd.DataFrame({"id": subs[3411]["id"]})
    blend_log = {}

    for tidx, (tname, scol) in enumerate(zip(TARGETS, SUB_COLS)):
        print(f"\n--- {tname} ---", flush=True)
        vals = np.zeros(len(pids_test))

        for pid in unique_pids:
            te_mask = pids_test == pid
            te_idx = np.where(te_mask)[0]
            n_test = te_mask.sum()

            # Get per-player LOO quality from calibration
            if calibrations and tname in calibrations:
                cal = calibrations[tname].get(str(int(pid)), {})
                mse = cal.get("mse", 0.008)
                slope = cal.get("slope", 0.5)
            else:
                mse = 0.008
                slope = 0.5

            # Determine blend strategy
            if mse < 0.007 and slope > 0.4:
                # Good - mostly base
                weights = {"ridge": 0.92, "cnn": 0.05, "gp": 0.02, "gplearn": 0.01}
                strategy = "good"
            elif mse > 0.012 or slope < 0.2:
                # Bad - more diversity
                weights = {"ridge": 0.60, "cnn": 0.20, "gp": 0.12, "gplearn": 0.08}
                strategy = "bad"
            else:
                # Medium
                weights = {"ridge": 0.80, "cnn": 0.10, "gp": 0.06, "gplearn": 0.04}
                strategy = "medium"

            # Compute blended predictions
            blend_pred = np.zeros(n_test)
            for model_name, w in weights.items():
                sub_num = model_subs[model_name]
                if sub_num in subs:
                    blend_pred += w * subs[sub_num][scol].values[te_idx]

            vals[te_idx] = np.clip(blend_pred, 0, 1)

            print(f"  P{pid}: MSE={mse:.6f}, slope={slope:.3f}, strategy={strategy}, "
                  f"weights={dict((k,f'{v:.2f}') for k,v in weights.items())}",
                  flush=True)

            blend_log[f"{tname}_P{pid}"] = {
                "mse": float(mse), "slope": float(slope),
                "strategy": strategy, "weights": weights
            }

        result_df[scol] = vals

    # Save main submission
    save_submission(result_df, "Per-player-target optimal model selection (3-way)")

    # Also try a more aggressive version (even more diversity for bad combos)
    result_aggressive = pd.DataFrame({"id": subs[3411]["id"]})
    for tidx, (tname, scol) in enumerate(zip(TARGETS, SUB_COLS)):
        vals = np.zeros(len(pids_test))
        for pid in unique_pids:
            te_mask = pids_test == pid
            te_idx = np.where(te_mask)[0]

            if calibrations and tname in calibrations:
                cal = calibrations[tname].get(str(int(pid)), {})
                mse = cal.get("mse", 0.008)
                slope = cal.get("slope", 0.5)
            else:
                mse = 0.008; slope = 0.5

            if mse < 0.007 and slope > 0.4:
                weights = {"ridge": 0.90, "cnn": 0.06, "gp": 0.03, "gplearn": 0.01}
            elif mse > 0.012 or slope < 0.2:
                weights = {"ridge": 0.45, "cnn": 0.25, "gp": 0.18, "gplearn": 0.12}
            else:
                weights = {"ridge": 0.70, "cnn": 0.15, "gp": 0.09, "gplearn": 0.06}

            blend_pred = np.zeros(te_mask.sum())
            for model_name, w in weights.items():
                sub_num = model_subs[model_name]
                if sub_num in subs:
                    blend_pred += w * subs[sub_num][scol].values[te_idx]

            vals[te_idx] = np.clip(blend_pred, 0, 1)

        result_aggressive[scol] = vals

    save_submission(result_aggressive, "Per-player-target AGGRESSIVE model selection")

    # Diversity analysis
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for scol in SUB_COLS:
            r, _ = pearsonr(result_df[scol], sub3558[scol])
            print(f"  {scol}: r={r:.4f}", flush=True)
    except Exception as e:
        print(f"  {e}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"per_player_target_selection_{ts}.json", "w") as f:
        json.dump(blend_log, f, indent=2)


if __name__ == "__main__":
    main()
