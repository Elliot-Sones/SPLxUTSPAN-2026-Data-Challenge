"""
TabPFN-v2 for Basketball Free Throw Prediction

TabPFN is a pretrained tabular foundation model (11M params) that does
Bayesian inference in a single forward pass. It requires NO hyperparameter
tuning and is designed for small-N tabular regression.

This is fundamentally different from our Ridge pipeline:
- Ridge: linear model with kernel weighting
- TabPFN: pretrained transformer that has seen millions of synthetic datasets

If TabPFN produces good standalone predictions, it's a completely different
model family for blending.
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
                  "kp_index": kp_index}
        if is_train:
            result["y"] = np.array(targets, dtype=np.float32)
        return result

    print("  Processing train...", flush=True)
    train = process(train_df, True)
    print("  Processing test...", flush=True)
    test = process(test_df, False)
    return train, test


def extract_features(X_3d, kp_index, frame_idx):
    """Extract compact features at release frame."""
    N = X_3d.shape[0]
    features = []
    dt = 1.0 / 60.0

    rh = kp_index["right_hip"]
    lh = kp_index["left_hip"]
    hip = (X_3d[:, frame_idx, rh, :] + X_3d[:, frame_idx, lh, :]) / 2

    # Position relative to hip and hoop
    for j in KEY_JOINTS:
        ji = kp_index[j]
        pos = X_3d[:, frame_idx, ji, :]
        rel = pos - hip
        hoop_rel = pos - HOOP_POS[np.newaxis, :]
        for ci in range(3):
            features.append(rel[:, ci])
            features.append(hoop_rel[:, ci])

    # Velocity
    for j in KEY_JOINTS[:5]:
        ji = kp_index[j]
        if 0 < frame_idx < 239:
            vel = (X_3d[:, frame_idx+1, ji, :] - X_3d[:, frame_idx-1, ji, :]) / (2*dt)
        elif frame_idx > 0:
            vel = (X_3d[:, frame_idx, ji, :] - X_3d[:, frame_idx-1, ji, :]) / dt
        else:
            vel = np.zeros((N, 3))
        for ci in range(3):
            features.append(vel[:, ci])

    # Geometric
    rs = X_3d[:, frame_idx, kp_index["right_shoulder"], :]
    re = X_3d[:, frame_idx, kp_index["right_elbow"], :]
    rw = X_3d[:, frame_idx, kp_index["right_wrist"], :]

    v1, v2 = rs - re, rw - re
    cos_e = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)+1e-8)
    features.append(np.arccos(np.clip(cos_e, -1, 1)))

    features.append(np.linalg.norm(rw - HOOP_POS[np.newaxis, :], axis=1))
    features.append(rw[:, 2] - HOOP_POS[2])

    X = np.column_stack(features)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


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
    from tabpfn import TabPFNRegressor

    print("=" * 70, flush=True)
    print("TabPFN-v2 Foundation Model", flush=True)
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
    y_scaled = np.zeros_like(y_raw)
    for tidx, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scaler.transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    all_results = {}
    test_preds_all = {}

    for tidx, tname in enumerate(TARGETS):
        frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        print(f"\n{'='*60}", flush=True)
        print(f"TARGET: {tname} (frame {frame})", flush=True)
        print(f"{'='*60}", flush=True)

        X_train = extract_features(X_3d_train, kp_index, frame)
        X_test = extract_features(X_3d_test, kp_index, frame)
        print(f"  Features: {X_train.shape[1]}", flush=True)

        # Mode 1: Global TabPFN (all players together)
        print("\n  --- Global TabPFN ---", flush=True)
        model_global = TabPFNRegressor()
        t0 = time.time()
        model_global.fit(X_train, y)
        pred_global = model_global.predict(X_test)
        print(f"  Fit+predict in {time.time()-t0:.1f}s", flush=True)

        # LOO for global
        loo_global = np.zeros(len(y))
        for i in range(len(y)):
            mask = np.ones(len(y), dtype=bool)
            mask[i] = False
            m = TabPFNRegressor()
            m.fit(X_train[mask], y[mask])
            loo_global[i] = m.predict(X_train[i:i+1])[0]
        mse_global = np.mean((loo_global - y) ** 2)
        print(f"  Global LOO MSE: {mse_global:.6f}", flush=True)

        # Mode 2: Per-player TabPFN
        print("\n  --- Per-Player TabPFN ---", flush=True)
        loo_perplayer = np.zeros(len(y))
        pred_perplayer = np.zeros(len(X_test))

        for pid in np.unique(pids_train):
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            n_p = tr_mask.sum()

            if n_p < 10:
                # Too few samples for per-player, use global
                for i in np.where(tr_mask)[0]:
                    loo_perplayer[i] = loo_global[i]
                pred_perplayer[te_mask] = pred_global[te_mask]
                continue

            X_p = X_train[tr_mask]
            y_p = y[tr_mask]

            # LOO within player
            for i_local in range(n_p):
                i_global = np.where(tr_mask)[0][i_local]
                mask_p = np.ones(n_p, dtype=bool)
                mask_p[i_local] = False
                m = TabPFNRegressor()
                m.fit(X_p[mask_p], y_p[mask_p])
                loo_perplayer[i_global] = m.predict(X_p[i_local:i_local+1])[0]

            # Test predictions
            m = TabPFNRegressor()
            m.fit(X_p, y_p)
            pred_perplayer[te_mask] = m.predict(X_test[te_mask])

        mse_perplayer = np.mean((loo_perplayer - y) ** 2)
        print(f"  Per-player LOO MSE: {mse_perplayer:.6f}", flush=True)

        # Pick best
        if mse_perplayer < mse_global:
            test_preds_all[tname] = np.clip(pred_perplayer, 0, 1)
            best_mse = mse_perplayer
            best_mode = "per-player"
        else:
            test_preds_all[tname] = np.clip(pred_global, 0, 1)
            best_mse = mse_global
            best_mode = "global"

        print(f"  Best: {best_mode} (MSE={best_mse:.6f})", flush=True)
        all_results[tname] = {"mse": float(best_mse), "mode": best_mode,
                              "global_mse": float(mse_global),
                              "perplayer_mse": float(mse_perplayer)}

    # Summary
    mean_mse = np.mean([all_results[t]["mse"] for t in TARGETS])
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY (mean={mean_mse:.6f})", flush=True)
    for tname in TARGETS:
        r = all_results[tname]
        print(f"  {tname}: {r['mode']} MSE={r['mse']:.6f} "
              f"(global={r['global_mse']:.6f}, per-player={r['perplayer_mse']:.6f})", flush=True)

    # Submissions
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": test_preds_all["angle"],
        "scaled_depth": test_preds_all["depth"],
        "scaled_left_right": test_preds_all["left_right"],
    })

    save_submission(sub_df, f"TabPFN standalone (mean LOO={mean_mse:.6f})")

    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for w in [0.03, 0.05, 0.08]:
            blend = sub_df.copy()
            for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
                blend[tc] = w * sub_df[tc] + (1-w) * sub3558[tc]
            save_submission(blend, f"{int(w*100)}% TabPFN + {int((1-w)*100)}% Sub3558")
    except Exception as e:
        print(f"  Blend error: {e}", flush=True)

    # Diversity
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            r, _ = pearsonr(sub_df[tc], sub3558[tc])
            print(f"  {tc}: r={r:.4f}", flush=True)
    except Exception as e:
        print(f"  {e}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"tabpfn_run_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
