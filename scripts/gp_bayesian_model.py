"""
Gaussian Process Model for Basketball Free Throw Prediction

Key idea: GP provides:
1. Principled Bayesian predictions with automatic relevance determination (ARD)
2. The kernel learns which features matter (via length scales)
3. Built-in regularization via marginal likelihood optimization
4. Uncertainty estimates that could help with ensembling

Unlike Ridge (which is a point estimate), GP models the full posterior,
so it naturally handles small sample sizes better.

This uses the same compact features as gplearn but with a GP model.
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
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel, RationalQuadratic
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP_POS = np.array([5.25, -25.0, 10.0])

KEY_JOINTS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_first_finger_mcp", "right_second_finger_mcp",
    "right_third_finger_mcp",
    "left_shoulder", "left_wrist",
    "right_hip", "left_hip",
    "nose",
]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


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
        pids, shot_ids, targets = [], [], []

        for idx in range(n):
            row = df.iloc[idx]
            for ci, col in enumerate(kp_cols):
                X_3d[idx, :, ci // 3, ci % 3] = parse_array_string(row[col])
            pids.append(row["participant_id"])
            shot_ids.append(row["shot_id"])
            if is_train:
                targets.append([row["angle"], row["depth"], row["left_right"]])

        result = {"X_3d": X_3d, "pids": np.array(pids),
                  "shot_ids": np.array(shot_ids), "kp_index": kp_index}
        if is_train:
            result["y"] = np.array(targets, dtype=np.float32)
        return result

    print("  Processing train...", flush=True)
    train = process(train_df, True)
    print("  Processing test...", flush=True)
    test = process(test_df, False)
    return train, test


def extract_features(X_3d, kp_index, frame_idx):
    """Extract compact features at a frame."""
    N = X_3d.shape[0]
    features = []

    rh_idx = kp_index["right_hip"]
    lh_idx = kp_index["left_hip"]
    hip = (X_3d[:, frame_idx, rh_idx, :] + X_3d[:, frame_idx, lh_idx, :]) / 2

    for j in KEY_JOINTS:
        ji = kp_index[j]
        pos = X_3d[:, frame_idx, ji, :]
        rel = pos - hip
        hoop_rel = pos - HOOP_POS[np.newaxis, :]
        for ci in range(3):
            features.append(rel[:, ci])
            features.append(hoop_rel[:, ci])

    # Velocity for shooting arm
    dt = 1.0 / 60.0
    for j in KEY_JOINTS[:6]:
        ji = kp_index[j]
        if 0 < frame_idx < 239:
            vel = (X_3d[:, frame_idx+1, ji, :] - X_3d[:, frame_idx-1, ji, :]) / (2*dt)
        elif frame_idx > 0:
            vel = (X_3d[:, frame_idx, ji, :] - X_3d[:, frame_idx-1, ji, :]) / dt
        else:
            vel = np.zeros((N, 3))
        for ci in range(3):
            features.append(vel[:, ci])

    # Geometric features
    rs = X_3d[:, frame_idx, kp_index["right_shoulder"], :]
    re = X_3d[:, frame_idx, kp_index["right_elbow"], :]
    rw = X_3d[:, frame_idx, kp_index["right_wrist"], :]
    rf = X_3d[:, frame_idx, kp_index["right_first_finger_mcp"], :]
    ls = X_3d[:, frame_idx, kp_index["left_shoulder"], :]

    v1, v2 = rs - re, rw - re
    cos_e = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)+1e-8)
    features.append(np.arccos(np.clip(cos_e, -1, 1)))

    v1, v2 = re - rw, rf - rw
    cos_w = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)+1e-8)
    features.append(np.arccos(np.clip(cos_w, -1, 1)))

    arm = re - rs
    features.append(np.arctan2(arm[:, 2], np.sqrt(arm[:, 0]**2+arm[:, 1]**2)))

    sl = rs - ls
    features.append(np.arctan2(sl[:, 0], -sl[:, 1]))

    features.append(np.linalg.norm(rw - HOOP_POS[np.newaxis, :], axis=1))
    features.append(np.linalg.norm((rw - HOOP_POS[np.newaxis, :])[:, :2], axis=1))
    features.append(rw[:, 2] - HOOP_POS[2])

    X = np.column_stack(features)
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)


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
    print("Gaussian Process Bayesian Model", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    train, test = load_data()
    X_3d_train, X_3d_test = train["X_3d"], test["X_3d"]
    pids_train, pids_test = train["pids"], test["pids"]
    y_raw = train["y"]
    kp_index = train["kp_index"]

    # Scale targets
    scalers = {}
    y_scaled = np.zeros_like(y_raw)
    for tidx, tname in enumerate(TARGETS):
        scalers[tname] = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scalers[tname].transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    all_results = {}
    test_preds_all = {}
    loo_preds_all = {}

    # Test multiple GP kernels
    kernels = {
        "RBF": ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3),
        "Matern32": ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-3),
        "Matern52": ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3),
        "RQ": ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0) + WhiteKernel(1e-3),
    }

    for tidx, tname in enumerate(TARGETS):
        print(f"\n{'='*60}", flush=True)
        print(f"TARGET: {tname} (frame {TARGET_FRAMES[tname]})", flush=True)
        print(f"{'='*60}", flush=True)

        frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]
        X_train = extract_features(X_3d_train, kp_index, frame)
        X_test = extract_features(X_3d_test, kp_index, frame)
        print(f"  Features: {X_train.shape[1]}", flush=True)

        best_kernel_name = None
        best_mse = float('inf')
        best_loo_preds = None

        for kname, kernel in kernels.items():
            print(f"\n  Kernel: {kname}", flush=True)

            # Per-player GP with LOO
            loo_preds = np.zeros(len(y))
            t0 = time.time()

            unique_pids = np.unique(pids_train)
            for pid in unique_pids:
                pmask = pids_train == pid
                X_p = X_train[pmask]
                y_p = y[pmask]
                n_p = len(y_p)

                # LOO within player
                for i in range(n_p):
                    idx_global = np.where(pmask)[0][i]
                    mask = np.ones(n_p, dtype=bool)
                    mask[i] = False

                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_p[mask])
                    X_te = scaler.transform(X_p[i:i+1])

                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=3,
                        alpha=1e-6,
                        normalize_y=True,
                    )
                    gp.fit(X_tr, y_p[mask])
                    pred, std = gp.predict(X_te, return_std=True)
                    loo_preds[idx_global] = np.clip(pred[0], 0, 1)

            mse = np.mean((loo_preds - y) ** 2)
            elapsed = time.time() - t0
            print(f"    LOO MSE: {mse:.6f} ({elapsed:.1f}s)", flush=True)

            if mse < best_mse:
                best_mse = mse
                best_kernel_name = kname
                best_loo_preds = loo_preds.copy()

        print(f"\n  Best kernel: {best_kernel_name} (MSE={best_mse:.6f})", flush=True)
        loo_preds_all[tname] = best_loo_preds

        # Generate test predictions with best kernel
        print(f"  Generating test predictions with {best_kernel_name}...", flush=True)
        test_preds = np.zeros(len(pids_test))
        best_kernel = kernels[best_kernel_name]

        for pid in np.unique(pids_test):
            tr_mask = pids_train == pid
            te_mask = pids_test == pid

            if tr_mask.sum() < 3:
                tr_mask = np.ones(len(pids_train), dtype=bool)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train[tr_mask])
            X_te = scaler.transform(X_test[te_mask])

            gp = GaussianProcessRegressor(
                kernel=best_kernel,
                n_restarts_optimizer=5,
                alpha=1e-6,
                normalize_y=True,
            )
            gp.fit(X_tr, y[tr_mask])
            pred, std = gp.predict(X_te, return_std=True)
            test_preds[te_mask] = np.clip(pred, 0, 1)

        test_preds_all[tname] = test_preds

        all_results[tname] = {
            "best_kernel": best_kernel_name,
            "mse": float(best_mse),
        }

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for tname in TARGETS:
        r = all_results[tname]
        print(f"  {tname}: MSE={r['mse']:.6f} (kernel={r['best_kernel']})", flush=True)
    mean_mse = np.mean([all_results[t]["mse"] for t in TARGETS])
    print(f"  Mean: {mean_mse:.6f}", flush=True)

    # Submissions
    sub_df = pd.DataFrame({
        "shot_id": test["shot_ids"],
        "scaled_angle": test_preds_all["angle"],
        "scaled_depth": test_preds_all["depth"],
        "scaled_left_right": test_preds_all["left_right"],
    })

    save_submission(sub_df, f"GP Bayesian (mean LOO={mean_mse:.6f})")

    # Blends
    try:
        sub3507 = pd.read_csv(SUBMISSION_DIR / "submission_3507.csv")
        for w in [0.03, 0.05, 0.08]:
            blend = sub_df.copy()
            for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
                blend[tc] = w * sub_df[tc] + (1-w) * sub3507[tc]
            save_submission(blend, f"{int(w*100)}% GP + {int((1-w)*100)}% Sub3507")
    except Exception as e:
        print(f"  Blend error: {e}", flush=True)

    # Diversity
    print(f"\n--- Diversity vs Sub3507 ---", flush=True)
    try:
        sub3507 = pd.read_csv(SUBMISSION_DIR / "submission_3507.csv")
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            r, _ = pearsonr(sub_df[tc], sub3507[tc])
            print(f"  {tc}: r={r:.4f}", flush=True)
    except Exception as e:
        print(f"  {e}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"gp_bayesian_run_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
