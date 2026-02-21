"""
Per-Player Calibration Fix

Problem: P4 angle (33.6% of angle error) and P5 angle/depth have poor calibration.
The model under-predicts variance (calibration slope < 1.0).

Previous approach (P4 hyperparam tuning) FAILED on LB (0.006393 vs 0.006234 base).
Reason: with 67 samples, tuning bw/alpha/frames overfits.

New approach: POST-HOC CALIBRATION
1. Run honest LOO to get predictions for each player-target
2. Fit simple linear calibration: y_true = a * y_pred + b (only 2 params)
3. Apply calibration to test predictions
4. Splice corrected predictions into Sub3507

Also: PER-PLAYER CNN BLEND WEIGHTS
- For players where Ridge is weak (P4/P5), use MORE CNN signal
- For players where Ridge is strong (P1), use LESS CNN signal
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
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window=7, polyorder=2):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if bad.any():
        good = ~bad
        if good.sum() < 3:
            return x
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder)


def load_data():
    print("Loading data...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in train_df.columns if c not in meta_cols]
    kp_names = [c[:-2] for c in kp_cols if c.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        pids, ids = [], []
        targets = []

        for idx in range(n):
            row = df.iloc[idx]
            for ci, col in enumerate(kp_cols):
                X_3d[idx, :, ci // 3, ci % 3] = parse_array_string(row[col])
            pids.append(row["participant_id"])
            ids.append(row["id"])
            if is_train:
                targets.append([row["angle"], row["depth"], row["left_right"]])

        result = {"X_3d": X_3d, "pids": np.array(pids), "ids": np.array(ids),
                  "kp_index": kp_index, "kp_names": kp_names}
        if is_train:
            result["y"] = np.array(targets, dtype=np.float32)
        return result

    print("  Processing train...", flush=True)
    train = process(train_df, True)
    print("  Processing test...", flush=True)
    test = process(test_df, False)
    return train, test


def extract_hoop_relative_features(X_3d, kp_index, frame_idx):
    """Extract hoop-relative features matching the core pipeline."""
    N, T, n_kp, _ = X_3d.shape
    features = []

    # Hip center
    rh = kp_index.get("right_hip", 0)
    lh = kp_index.get("left_hip", 0)
    hip = (X_3d[:, frame_idx, rh, :] + X_3d[:, frame_idx, lh, :]) / 2  # (N, 3)

    # All keypoints hoop-relative
    for kp_name, ki in sorted(kp_index.items(), key=lambda x: x[1]):
        pos = X_3d[:, frame_idx, ki, :]  # (N, 3)
        hoop_rel = pos - HOOP_POS[np.newaxis, :]
        for ci in range(3):
            features.append(hoop_rel[:, ci])

    # Velocity at release frame
    for kp_name, ki in sorted(kp_index.items(), key=lambda x: x[1]):
        if 0 < frame_idx < 239:
            vel = (X_3d[:, frame_idx+1, ki, :] - X_3d[:, frame_idx-1, ki, :]) / (2*DT)
        elif frame_idx > 0:
            vel = (X_3d[:, frame_idx, ki, :] - X_3d[:, frame_idx-1, ki, :]) / DT
        else:
            vel = np.zeros((N, 3))
        for ci in range(3):
            features.append(vel[:, ci])

    # Aggregate stats
    # Distance to hoop
    for ki in range(n_kp):
        d = np.linalg.norm(X_3d[:, frame_idx, ki, :] - HOOP_POS[np.newaxis, :], axis=1)
        features.append(d)

    X = np.column_stack(features)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def honest_loo_per_player(X, y, pids, bw_q=0.3, alpha=10.0):
    """LOO with PLS refit per fold (honest - no leakage)."""
    n = len(y)
    preds = np.zeros(n)

    unique_pids = np.unique(pids)

    for pid in unique_pids:
        pmask = pids == pid
        indices = np.where(pmask)[0]
        n_p = len(indices)

        for i_local, i_global in enumerate(indices):
            # Leave one out
            mask = np.ones(n, dtype=bool)
            mask[i_global] = False

            # Player-only training data
            tr_mask = mask & pmask
            if tr_mask.sum() < 5:
                tr_mask = mask

            # Fit PLS on training data only (honest - no leakage)
            X_tr_raw = X[tr_mask]
            y_tr = y[tr_mask]
            X_te_raw = X[i_global:i_global+1]

            # PLS
            n_comp = min(10, min(X_tr_raw.shape) - 1)
            if n_comp >= 2:
                try:
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(X_tr_raw, y_tr)
                    pls_tr = pls.transform(X_tr_raw)
                    pls_te = pls.transform(X_te_raw)
                    X_tr_aug = np.hstack([X_tr_raw, pls_tr])
                    X_te_aug = np.hstack([X_te_raw, pls_te])
                except Exception:
                    X_tr_aug = X_tr_raw
                    X_te_aug = X_te_raw
            else:
                X_tr_aug = X_tr_raw
                X_te_aug = X_te_raw

            # Scale + predict
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_aug)
            X_te_s = scaler.transform(X_te_aug)

            dists = cdist(X_te_s, X_tr_s, metric='euclidean').flatten()
            bw = np.quantile(dists, bw_q)
            if bw < 1e-8:
                bw = 1.0
            w = np.exp(-0.5 * (dists / bw) ** 2)

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=w)
            preds[i_global] = ridge.predict(X_te_s)[0]

    return preds


def predict_test_per_player(X_train, y_train, pids_train, X_test, pids_test,
                             bw_q=0.3, alpha=10.0):
    """Generate test predictions with PLS (fit on all train data per player)."""
    preds = np.zeros(len(X_test))

    for pid in np.unique(pids_test):
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        if tr_mask.sum() < 5:
            tr_mask = np.ones(len(pids_train), dtype=bool)

        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]

        # PLS
        n_comp = min(10, min(X_tr.shape) - 1)
        if n_comp >= 2:
            try:
                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_tr, y_tr)
                pls_tr = pls.transform(X_tr)
                pls_te = pls.transform(X_te)
                X_tr_aug = np.hstack([X_tr, pls_tr])
                X_te_aug = np.hstack([X_te, pls_te])
            except Exception:
                X_tr_aug = X_tr
                X_te_aug = X_te
        else:
            X_tr_aug = X_tr
            X_te_aug = X_te

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_aug)
        X_te_s = scaler.transform(X_te_aug)

        for i in range(te_mask.sum()):
            dists = cdist(X_te_s[i:i+1], X_tr_s, metric='euclidean').flatten()
            bw = np.quantile(dists, bw_q)
            if bw < 1e-8:
                bw = 1.0
            w = np.exp(-0.5 * (dists / bw) ** 2)

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=w)
            te_indices = np.where(te_mask)[0]
            preds[te_indices[i]] = ridge.predict(X_te_s[i:i+1])[0]

    return preds


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


def main():
    print("=" * 70, flush=True)
    print("Per-Player Calibration Fix", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    train, test = load_data()

    X_3d_train = train["X_3d"]
    X_3d_test = test["X_3d"]
    pids_train = train["pids"]
    pids_test = test["pids"]
    ids_test = test["ids"]
    y_raw = train["y"]
    kp_index = train["kp_index"]

    # Scale targets
    scalers = {}
    y_scaled = np.zeros_like(y_raw)
    for tidx, tname in enumerate(TARGETS):
        scalers[tname] = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scalers[tname].transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    # Load base submission
    sub3507 = pd.read_csv(SUBMISSION_DIR / "submission_3507.csv")
    print(f"Base: Sub3507 (LB 0.006205)", flush=True)

    unique_pids = np.unique(pids_train)
    print(f"Players: {unique_pids}, Train: {len(pids_train)}, Test: {len(pids_test)}", flush=True)

    # ===================================================
    # PHASE 1: Per-player calibration analysis
    # ===================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 1: Per-Player Calibration Analysis", flush=True)
    print(f"{'='*60}", flush=True)

    all_calibrations = {}

    for tidx, tname in enumerate(TARGETS):
        frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        print(f"\n--- {tname} (frame {frame}) ---", flush=True)

        # Extract features
        X_train = extract_hoop_relative_features(X_3d_train, kp_index, frame)
        X_test = extract_hoop_relative_features(X_3d_test, kp_index, frame)
        print(f"  Features: {X_train.shape[1]}", flush=True)

        # Honest LOO
        print(f"  Running honest LOO...", flush=True)
        t0 = time.time()
        loo_preds = honest_loo_per_player(X_train, y, pids_train)
        print(f"  LOO done in {time.time()-t0:.1f}s", flush=True)

        overall_mse = np.mean((loo_preds - y) ** 2)
        print(f"  Overall LOO MSE: {overall_mse:.6f}", flush=True)

        # Per-player calibration
        all_calibrations[tname] = {}
        for pid in unique_pids:
            pmask = pids_train == pid
            y_p = y[pmask]
            pred_p = loo_preds[pmask]
            n_p = pmask.sum()

            mse_p = np.mean((pred_p - y_p) ** 2)

            # Calibration: fit y_true = a * y_pred + b
            lr = LinearRegression()
            lr.fit(pred_p.reshape(-1, 1), y_p)
            slope = lr.coef_[0]
            intercept = lr.intercept_

            # Calibrated predictions
            cal_pred = slope * pred_p + intercept
            cal_mse = np.mean((cal_pred - y_p) ** 2)

            pct_change = (cal_mse - mse_p) / mse_p * 100

            print(f"  P{pid}: n={n_p}, MSE={mse_p:.6f}, slope={slope:.3f}, "
                  f"intercept={intercept:.3f}, cal_MSE={cal_mse:.6f} ({pct_change:+.1f}%)",
                  flush=True)

            all_calibrations[tname][int(pid)] = {
                "n": int(n_p),
                "mse": float(mse_p),
                "slope": float(slope),
                "intercept": float(intercept),
                "cal_mse": float(cal_mse),
                "pct_change": float(pct_change),
                "loo_preds": pred_p.tolist(),
                "y_true": y_p.tolist(),
            }

    # ===================================================
    # PHASE 2: Apply calibration corrections to test predictions
    # ===================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 2: Generate Calibrated Test Predictions", flush=True)
    print(f"{'='*60}", flush=True)

    # First, generate our own test predictions (to calibrate)
    test_preds_raw = {}
    for tidx, tname in enumerate(TARGETS):
        frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        X_train = extract_hoop_relative_features(X_3d_train, kp_index, frame)
        X_test = extract_hoop_relative_features(X_3d_test, kp_index, frame)

        print(f"\n  Generating test predictions for {tname}...", flush=True)
        test_preds = predict_test_per_player(X_train, y, pids_train, X_test, pids_test)
        test_preds_raw[tname] = test_preds

    # Apply per-player calibration to BOTH our predictions and Sub3507
    # Strategy: calibrate our predictions, then blend with Sub3507

    SUB_COLS = {"angle": "scaled_angle", "depth": "scaled_depth", "left_right": "scaled_left_right"}

    # Create calibrated version of Sub3507
    sub3507_calibrated = sub3507.copy()

    for tname in TARGETS:
        scol = SUB_COLS[tname]
        for pid in unique_pids:
            te_mask = pids_test == pid
            cal = all_calibrations[tname][int(pid)]

            # Only calibrate if it helps (cal_mse < mse)
            if cal["pct_change"] < -1.0:  # At least 1% improvement
                slope = cal["slope"]
                intercept = cal["intercept"]

                # Calibrate Sub3507 predictions for this player-target
                te_indices = np.where(te_mask)[0]
                original = sub3507[scol].values[te_indices]
                calibrated = slope * original + intercept
                calibrated = np.clip(calibrated, 0, 1)

                print(f"  Calibrating {tname} P{pid}: slope={slope:.3f}, "
                      f"intercept={intercept:.3f} (LOO improvement: {cal['pct_change']:.1f}%)",
                      flush=True)

                sub3507_calibrated.loc[sub3507_calibrated.index[te_indices], scol] = calibrated
            else:
                print(f"  Skipping {tname} P{pid}: calibration doesn't help ({cal['pct_change']:+.1f}%)",
                      flush=True)

    # Save calibrated Sub3507
    desc = "Sub3507 with per-player calibration correction (only where helps)"
    sub_num_cal = save_submission(sub3507_calibrated, desc)

    # ===================================================
    # PHASE 3: Per-player CNN blend weights
    # ===================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 3: Per-Player CNN Blend Weights", flush=True)
    print(f"{'='*60}", flush=True)

    # Load CNN submissions
    # Sub3507 = 3% CNN + 97% Sub3411
    # Let's load the standalone CNN predictions and Sub3411
    # The improved CNN standalone is Sub 3516
    try:
        sub3411 = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")
        sub3516 = pd.read_csv(SUBMISSION_DIR / "submission_3516.csv")  # Standalone CNN

        # For each player, find optimal CNN weight
        # Sub3507 = 0.03 * sub3516 + 0.97 * sub3411
        # We want to find per-player optimal weights

        # Use LOO analysis to determine which players benefit more from CNN
        for tname in TARGETS:
            scol = SUB_COLS[tname]
            for pid in unique_pids:
                te_mask = pids_test == pid
                cal = all_calibrations[tname][int(pid)]
                mse = cal["mse"]

                # Higher MSE = more room for CNN to help
                # Lower calibration slope = Ridge is under-predicting variation
                slope = cal["slope"]

                # Heuristic: use more CNN for worse-performing players
                if mse > 0.010:
                    cnn_weight = 0.10  # 10% CNN for bad players
                elif mse > 0.007:
                    cnn_weight = 0.05  # 5% CNN for medium players
                else:
                    cnn_weight = 0.02  # 2% CNN for good players

                print(f"  {tname} P{pid}: MSE={mse:.6f}, slope={slope:.3f} -> CNN weight={cnn_weight:.2f}",
                      flush=True)

        # Create per-player CNN weight submission
        sub_per_player_cnn = pd.DataFrame({"id": sub3411["id"]})
        for tname in TARGETS:
            scol = SUB_COLS[tname]
            vals = np.zeros(len(pids_test))
            for pid in unique_pids:
                te_mask = pids_test == pid
                te_indices = np.where(te_mask)[0]

                cal = all_calibrations[tname][int(pid)]
                mse = cal["mse"]

                if mse > 0.010:
                    w = 0.10
                elif mse > 0.007:
                    w = 0.05
                else:
                    w = 0.02

                cnn_vals = sub3516[scol].values[te_indices]
                ridge_vals = sub3411[scol].values[te_indices]
                vals[te_indices] = w * cnn_vals + (1 - w) * ridge_vals

            sub_per_player_cnn[scol] = vals

        save_submission(sub_per_player_cnn, "Per-player CNN weights (2-10% by MSE)")

        # Also create a version with calibration ON TOP of per-player CNN blend
        sub_cal_cnn = sub_per_player_cnn.copy()
        for tname in TARGETS:
            scol = SUB_COLS[tname]
            for pid in unique_pids:
                te_mask = pids_test == pid
                cal = all_calibrations[tname][int(pid)]
                if cal["pct_change"] < -1.0:
                    te_indices = np.where(te_mask)[0]
                    original = sub_cal_cnn[scol].values[te_indices]
                    calibrated = cal["slope"] * original + cal["intercept"]
                    sub_cal_cnn.loc[sub_cal_cnn.index[te_indices], scol] = np.clip(calibrated, 0, 1)

        save_submission(sub_cal_cnn, "Per-player CNN weights + calibration correction")

    except Exception as e:
        print(f"  Error loading CNN/base submissions: {e}", flush=True)

    # ===================================================
    # PHASE 4: Targeted aggressive blend for worst combos
    # ===================================================
    print(f"\n{'='*60}", flush=True)
    print("PHASE 4: Targeted Aggressive Fix for P4 Angle, P5 Angle/Depth", flush=True)
    print(f"{'='*60}", flush=True)

    # Load GP Bayesian predictions (good diversity, especially for LR)
    try:
        sub_gp = pd.read_csv(SUBMISSION_DIR / "submission_3550.csv")  # GP standalone

        # For P4/P5 angle and P5 depth, try splicing in GP predictions
        # at various blend weights into Sub3507
        for blend_w in [0.15, 0.25, 0.40]:
            sub_targeted = sub3507.copy()

            # P4 angle
            p4_te = pids_test == 4
            p4_idx = np.where(p4_te)[0]
            sub_targeted.loc[sub_targeted.index[p4_idx], "scaled_angle"] = (
                blend_w * sub_gp["scaled_angle"].values[p4_idx] +
                (1 - blend_w) * sub3507["scaled_angle"].values[p4_idx]
            )

            # P5 angle
            p5_te = pids_test == 5
            p5_idx = np.where(p5_te)[0]
            sub_targeted.loc[sub_targeted.index[p5_idx], "scaled_angle"] = (
                blend_w * sub_gp["scaled_angle"].values[p5_idx] +
                (1 - blend_w) * sub3507["scaled_angle"].values[p5_idx]
            )

            # P5 depth
            sub_targeted.loc[sub_targeted.index[p5_idx], "scaled_depth"] = (
                blend_w * sub_gp["scaled_depth"].values[p5_idx] +
                (1 - blend_w) * sub3507["scaled_depth"].values[p5_idx]
            )

            save_submission(sub_targeted,
                          f"{int(blend_w*100)}% GP for P4ang+P5ang+P5dep into Sub3507")

    except Exception as e:
        print(f"  Error: {e}", flush=True)

    # ===================================================
    # Summary
    # ===================================================
    print(f"\n{'='*60}", flush=True)
    print("CALIBRATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    for tname in TARGETS:
        print(f"\n  {tname}:", flush=True)
        for pid in unique_pids:
            cal = all_calibrations[tname][int(pid)]
            flag = " <-- FIX TARGET" if cal["pct_change"] < -5 else ""
            print(f"    P{pid}: MSE={cal['mse']:.6f}, slope={cal['slope']:.3f}, "
                  f"cal_change={cal['pct_change']:+.1f}%{flag}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_cal = {}
    for tname in TARGETS:
        save_cal[tname] = {}
        for pid in unique_pids:
            cal = all_calibrations[tname][int(pid)]
            save_cal[tname][int(pid)] = {k: v for k, v in cal.items()
                                          if k not in ["loo_preds", "y_true"]}

    with open(OUTPUT_DIR / f"per_player_calibration_{ts}.json", "w") as f:
        json.dump(save_cal, f, indent=2)


if __name__ == "__main__":
    main()
