"""
ROCKET/MiniRocket Regression - PhD-level time series feature extraction.

ROCKET (Dempster et al., 2020, KDD): Random Convolutional Kernel Transform.
- 10,000 random kernels with random length, dilation, padding, weights, bias
- Each kernel produces 2 features: PPV (proportion positive values) and MAX
- Total: 20,000 features -> Ridge regression
- Designed for small sample sizes, state-of-art on UCR benchmarks
- Fundamentally different from learned CNNs or hand-crafted features

Strategy:
- Per-player MiniRocket on shooting-phase keypoint trajectories
- Ridge regression on ROCKET features
- LOO evaluation, then blend with Sub3558
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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP_POS = np.array([5.25, -25.0, 10.0])


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
        pids, ids, targets = [], [], []
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

    train = process(train_df, True)
    test = process(test_df, False)
    return train, test


def prepare_rocket_input(X_3d, kp_index, frame_range=(100, 200), subsample=2):
    """Prepare multivariate time series for ROCKET.

    Output shape: (n_samples, n_channels, n_timepoints)
    Channels = keypoint coords (hip-centered) + velocities
    """
    N = X_3d.shape[0]
    f_start, f_end = frame_range
    frames = np.arange(f_start, f_end, subsample)
    n_frames = len(frames)

    # Hip center for normalization
    rh = kp_index.get("right_hip", 0)
    lh = kp_index.get("left_hip", 0)
    hip = (X_3d[:, frames, rh, :] + X_3d[:, frames, lh, :]) / 2.0  # (N, T, 3)

    # Use key joints (not all 69 - too many channels for 345 samples)
    key_joints = [
        "right_shoulder", "right_elbow", "right_wrist",
        "right_first_finger_mcp", "right_second_finger_mcp",
        "left_shoulder", "left_elbow", "left_wrist",
        "neck", "mid_hip",
        "right_hip", "left_hip",
        "right_knee", "left_knee",
    ]

    channels = []
    for joint in key_joints:
        if joint not in kp_index:
            continue
        ji = kp_index[joint]
        # Hip-centered positions (3 coords)
        pos = X_3d[:, frames, ji, :] - hip  # (N, T, 3)
        pos = np.nan_to_num(pos, nan=0.0)
        for c in range(3):
            channels.append(pos[:, :, c])  # (N, T)
        # Velocities (3 coords)
        vel = np.diff(pos, axis=1, prepend=pos[:, :1, :])
        for c in range(3):
            channels.append(vel[:, :, c])

    # Stack: (N, n_channels, T)
    X_rocket = np.stack(channels, axis=1).astype(np.float32)
    print(f"  ROCKET input: {X_rocket.shape} (samples, channels, timepoints)", flush=True)
    return X_rocket


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
    print("ROCKET/MiniRocket Regression", flush=True)
    print("Random Convolutional Kernel Transform for time series", flush=True)
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
    unique_pids = np.unique(pids_train)

    # Scale targets
    y_scaled = np.zeros_like(y_raw)
    for tidx, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scaler.transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    # Try multiple configs
    configs = [
        {"name": "14j_100_200_s2", "frame_range": (100, 200), "subsample": 2,
         "n_kernels": 5000},
        {"name": "14j_120_180_s2", "frame_range": (120, 180), "subsample": 2,
         "n_kernels": 5000},
        {"name": "14j_100_200_s3", "frame_range": (100, 200), "subsample": 3,
         "n_kernels": 5000},
    ]

    best_test_preds = {}
    best_mses = {}
    best_oof = {}

    for config in configs:
        print(f"\n{'='*50}", flush=True)
        print(f"Config: {config['name']}", flush=True)
        print(f"{'='*50}", flush=True)

        # Prepare ROCKET input
        X_train_rocket = prepare_rocket_input(
            X_3d_train, kp_index,
            frame_range=config["frame_range"],
            subsample=config["subsample"]
        )
        X_test_rocket = prepare_rocket_input(
            X_3d_test, kp_index,
            frame_range=config["frame_range"],
            subsample=config["subsample"]
        )

        # Import ROCKET
        from aeon.transformations.collection.convolution_based import MiniRocket, Rocket
        rocket_cls = MiniRocket
        rocket_name = "MiniRocket"
        print(f"  Using {rocket_name}", flush=True)

        for tidx, tname in enumerate(TARGETS):
            y = y_scaled[:, tidx]

            loo_preds = np.zeros(len(y))
            test_preds_all = np.zeros(len(pids_test))
            n_test_predicted = 0

            for pid in unique_pids:
                tr_mask = pids_train == pid
                te_mask = pids_test == pid
                idx_tr = np.where(tr_mask)[0]
                n_p = len(idx_tr)
                y_p = y[tr_mask]

                X_p_train = X_train_rocket[tr_mask]
                X_p_test = X_test_rocket[te_mask] if te_mask.sum() > 0 else None

                # Fit ROCKET on this player's data
                rocket = rocket_cls(
                    n_kernels=config["n_kernels"],
                    random_state=42
                )

                # Transform all player data
                X_p_transformed = rocket.fit_transform(X_p_train)
                if X_p_test is not None:
                    X_p_test_transformed = rocket.transform(X_p_test)

                # Scale features
                sc = StandardScaler()
                X_p_scaled = sc.fit_transform(X_p_transformed)
                if X_p_test is not None:
                    X_p_test_scaled = sc.transform(X_p_test_transformed)

                # LOO with RidgeCV
                alphas = np.logspace(-3, 4, 30)
                for i in range(n_p):
                    tr_idx = np.array([j for j in range(n_p) if j != i])
                    X_tr_loo = X_p_scaled[tr_idx]
                    y_tr_loo = y_p[tr_idx]

                    ridge = RidgeCV(alphas=alphas)
                    ridge.fit(X_tr_loo, y_tr_loo)
                    pred = ridge.predict(X_p_scaled[[i]])[0]
                    loo_preds[idx_tr[i]] = np.clip(pred, 0, 1)

                # Full model for test
                if X_p_test is not None:
                    ridge_full = RidgeCV(alphas=alphas)
                    ridge_full.fit(X_p_scaled, y_p)
                    te_preds = ridge_full.predict(X_p_test_scaled)
                    test_preds_all[te_mask] = np.clip(te_preds, 0, 1)
                    n_test_predicted += te_mask.sum()

            mse = np.mean((loo_preds - y) ** 2)
            key = f"{config['name']}_{tname}"
            print(f"  {tname}: LOO MSE={mse:.6f}", flush=True)

            if tname not in best_mses or mse < best_mses[tname]:
                best_mses[tname] = mse
                best_test_preds[tname] = test_preds_all.copy()
                best_oof[tname] = loo_preds.copy()

    # Summary
    mean_best = np.mean(list(best_mses.values()))
    print(f"\n{'='*50}", flush=True)
    print(f"BEST per-target: mean LOO={mean_best:.6f}", flush=True)
    for tname in TARGETS:
        print(f"  {tname}: {best_mses[tname]:.6f}", flush=True)

    # Create standalone submission
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": best_test_preds["angle"],
        "scaled_depth": best_test_preds["depth"],
        "scaled_left_right": best_test_preds["left_right"],
    })

    # Diversity vs Sub3558
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
    for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
        r, _ = pearsonr(sub_df[tc], sub3558[tc])
        print(f"  {tc}: r={r:.4f}", flush=True)

    save_submission(sub_df, f"ROCKET standalone (mean LOO={mean_best:.6f})")

    # Blends with Sub3558
    for w in [0.02, 0.03, 0.05]:
        blend = sub_df.copy()
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            blend[tc] = np.clip(w * sub_df[tc] + (1 - w) * sub3558[tc], 0, 1)
        save_submission(blend, f"{int(w*100)}% ROCKET + {int((1-w)*100)}% Sub3558")

    # Per-target optimized blend (more weight where more diverse)
    per_target = sub3558.copy()
    for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
        r, _ = pearsonr(sub_df[tc], sub3558[tc])
        if r < 0.60:
            w = 0.06
        elif r < 0.75:
            w = 0.04
        elif r < 0.85:
            w = 0.03
        else:
            w = 0.02
        per_target[tc] = np.clip(w * sub_df[tc] + (1 - w) * sub3558[tc], 0, 1)
        print(f"  Per-target {tc}: r={r:.4f} -> w={w}", flush=True)
    save_submission(per_target, "Per-target ROCKET blend with Sub3558")

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"rocket_regression_{ts}.json", "w") as f:
        json.dump({
            "best_mses": {k: float(v) for k, v in best_mses.items()},
            "mean_loo": float(mean_best),
            "configs_tested": [c["name"] for c in configs],
        }, f, indent=2)


if __name__ == "__main__":
    main()
