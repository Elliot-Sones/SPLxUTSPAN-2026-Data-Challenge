"""
ROCKET v2 - Full ROCKET (not Mini) + more configurations + combined blend.

v1 results: MiniRocket standalone LOO 0.006592, depth 0.005858 (best ever)
Diversity vs Sub3558: depth r=0.791 (excellent)

v2: Try full Rocket (more features), different kernel counts, all 69 joints.
Then create optimized blends including ROCKET + FPCA + LGBM on Sub3558.
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
from aeon.transformations.collection.convolution_based import MiniRocket, Rocket
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
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

    return process(train_df, True), process(test_df, False)


def prepare_input(X_3d, kp_index, frame_range, subsample, joint_set="14j"):
    N = X_3d.shape[0]
    f_start, f_end = frame_range
    frames = np.arange(f_start, f_end, subsample)

    rh = kp_index.get("right_hip", 0)
    lh = kp_index.get("left_hip", 0)
    hip = (X_3d[:, frames, rh, :] + X_3d[:, frames, lh, :]) / 2.0

    if joint_set == "14j":
        joints = [
            "right_shoulder", "right_elbow", "right_wrist",
            "right_first_finger_mcp", "right_second_finger_mcp",
            "left_shoulder", "left_elbow", "left_wrist",
            "neck", "mid_hip", "right_hip", "left_hip",
            "right_knee", "left_knee",
        ]
    elif joint_set == "7j_arm":
        joints = [
            "right_shoulder", "right_elbow", "right_wrist",
            "right_first_finger_mcp", "right_second_finger_mcp",
            "right_third_finger_mcp", "right_fourth_finger_mcp",
        ]
    else:
        # All joints
        joints = list(kp_index.keys())

    channels = []
    for joint in joints:
        if joint not in kp_index:
            continue
        ji = kp_index[joint]
        pos = np.nan_to_num(X_3d[:, frames, ji, :] - hip, nan=0.0)
        for c in range(3):
            channels.append(pos[:, :, c])
        vel = np.diff(pos, axis=1, prepend=pos[:, :1, :])
        for c in range(3):
            channels.append(vel[:, :, c])

    X = np.stack(channels, axis=1).astype(np.float32)
    return X


def get_next_submission_number():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    lock_path = SUBMISSION_DIR / ".submission_lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
        nums = [int(p.stem.split("_")[1]) for p in existing
                if p.stem.split("_")[1].isdigit()]
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


def run_rocket_loo(X_train, X_test, y, pids_train, pids_test, rocket_cls,
                   n_kernels, tname, random_state=42):
    """Per-player ROCKET LOO + test predictions."""
    unique_pids = np.unique(pids_train)
    loo_preds = np.zeros(len(y))
    test_preds = np.zeros(len(pids_test))
    alphas = np.logspace(-3, 4, 30)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        idx_tr = np.where(tr_mask)[0]
        n_p = len(idx_tr)
        y_p = y[tr_mask]

        X_p = X_train[tr_mask]
        X_p_te = X_test[te_mask] if te_mask.sum() > 0 else None

        rocket = rocket_cls(n_kernels=n_kernels, random_state=random_state)
        X_p_feat = rocket.fit_transform(X_p)
        if X_p_te is not None:
            X_p_te_feat = rocket.transform(X_p_te)

        sc = StandardScaler()
        X_p_s = sc.fit_transform(X_p_feat)
        if X_p_te is not None:
            X_p_te_s = sc.transform(X_p_te_feat)

        # LOO
        for i in range(n_p):
            tr_idx = [j for j in range(n_p) if j != i]
            ridge = RidgeCV(alphas=alphas)
            ridge.fit(X_p_s[tr_idx], y_p[tr_idx])
            loo_preds[idx_tr[i]] = np.clip(ridge.predict(X_p_s[[i]])[0], 0, 1)

        # Test
        if X_p_te is not None:
            ridge_full = RidgeCV(alphas=alphas)
            ridge_full.fit(X_p_s, y_p)
            test_preds[te_mask] = np.clip(ridge_full.predict(X_p_te_s), 0, 1)

    mse = np.mean((loo_preds - y) ** 2)
    return mse, loo_preds, test_preds


def main():
    print("=" * 70, flush=True)
    print("ROCKET v2 - Full ROCKET + Multiple Configs + Combined Blends", flush=True)
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

    y_scaled = np.zeros_like(y_raw)
    for tidx, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scaler.transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    # Configs to test
    configs = [
        # Best from v1
        {"name": "mini_14j_100_200_s3", "cls": MiniRocket, "n_kernels": 5000,
         "frame_range": (100, 200), "subsample": 3, "joints": "14j"},
        # Full ROCKET (more features via PPV+MAX)
        {"name": "full_14j_100_200_s3", "cls": Rocket, "n_kernels": 5000,
         "frame_range": (100, 200), "subsample": 3, "joints": "14j"},
        # Arm-only (shooting arm focus)
        {"name": "mini_7j_100_200_s3", "cls": MiniRocket, "n_kernels": 5000,
         "frame_range": (100, 200), "subsample": 3, "joints": "7j_arm"},
        # More kernels
        {"name": "mini_14j_100_200_s3_10k", "cls": MiniRocket, "n_kernels": 10000,
         "frame_range": (100, 200), "subsample": 3, "joints": "14j"},
        # Multi-seed ensemble (average 3 random states)
        {"name": "mini_14j_100_200_s3_seed7", "cls": MiniRocket, "n_kernels": 5000,
         "frame_range": (100, 200), "subsample": 3, "joints": "14j", "seed": 7},
        {"name": "mini_14j_100_200_s3_seed99", "cls": MiniRocket, "n_kernels": 5000,
         "frame_range": (100, 200), "subsample": 3, "joints": "14j", "seed": 99},
    ]

    all_results = {}
    all_test_preds = {}  # config_target -> preds
    all_oof = {}

    for config in configs:
        print(f"\n{'='*50}", flush=True)
        print(f"Config: {config['name']}", flush=True)
        print(f"{'='*50}", flush=True)

        X_tr = prepare_input(X_3d_train, kp_index, config["frame_range"],
                             config["subsample"], config["joints"])
        X_te = prepare_input(X_3d_test, kp_index, config["frame_range"],
                             config["subsample"], config["joints"])
        print(f"  Input: {X_tr.shape}", flush=True)

        seed = config.get("seed", 42)

        for tidx, tname in enumerate(TARGETS):
            y = y_scaled[:, tidx]
            mse, oof, te_preds = run_rocket_loo(
                X_tr, X_te, y, pids_train, pids_test,
                config["cls"], config["n_kernels"], tname, random_state=seed
            )
            key = f"{config['name']}_{tname}"
            all_results[key] = mse
            all_test_preds[key] = te_preds
            all_oof[key] = oof
            print(f"  {tname}: LOO MSE={mse:.6f}", flush=True)

    # Find best config per target
    print(f"\n{'='*60}", flush=True)
    print("Best per-target configs:", flush=True)
    best_preds = {}
    best_oof_preds = {}
    for tname in TARGETS:
        best_key = None
        best_mse = float('inf')
        for key, mse in all_results.items():
            if key.endswith(f"_{tname}") and mse < best_mse:
                best_mse = mse
                best_key = key
        print(f"  {tname}: {best_key} = {best_mse:.6f}", flush=True)
        best_preds[tname] = all_test_preds[best_key]
        best_oof_preds[tname] = all_oof[best_key]

    # Multi-seed ensemble: average predictions from seeds 42, 7, 99
    print(f"\n--- Multi-seed Ensemble ---", flush=True)
    ensemble_preds = {}
    ensemble_oof = {}
    for tname in TARGETS:
        keys = [k for k in all_test_preds if tname in k and "14j_100_200_s3" in k
                and "10k" not in k and "full" not in k]
        if len(keys) >= 3:
            te_avg = np.mean([all_test_preds[k] for k in keys], axis=0)
            oof_avg = np.mean([all_oof[k] for k in keys], axis=0)
            oof_mse = np.mean((oof_avg - y_scaled[:, TARGETS.index(tname)]) ** 2)
            print(f"  {tname}: ensemble of {len(keys)} seeds, LOO MSE={oof_mse:.6f}", flush=True)
            ensemble_preds[tname] = te_avg
            ensemble_oof[tname] = oof_avg
        else:
            ensemble_preds[tname] = best_preds[tname]
            ensemble_oof[tname] = best_oof_preds[tname]

    # Create submissions
    sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")

    # 1. Best single-config per target
    sub_best = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": np.clip(best_preds["angle"], 0, 1),
        "scaled_depth": np.clip(best_preds["depth"], 0, 1),
        "scaled_left_right": np.clip(best_preds["left_right"], 0, 1),
    })
    print(f"\n--- Diversity: Best Config vs Sub3558 ---", flush=True)
    for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
        r, _ = pearsonr(sub_best[tc], sub3558[tc])
        print(f"  {tc}: r={r:.4f}", flush=True)
    save_submission(sub_best, "ROCKET v2 best-per-target standalone")

    # 2. Multi-seed ensemble
    sub_ens = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": np.clip(ensemble_preds["angle"], 0, 1),
        "scaled_depth": np.clip(ensemble_preds["depth"], 0, 1),
        "scaled_left_right": np.clip(ensemble_preds["left_right"], 0, 1),
    })
    print(f"\n--- Diversity: Ensemble vs Sub3558 ---", flush=True)
    for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
        r, _ = pearsonr(sub_ens[tc], sub3558[tc])
        print(f"  {tc}: r={r:.4f}", flush=True)
    save_submission(sub_ens, "ROCKET v2 3-seed ensemble standalone")

    # 3. Blends with Sub3558 using ensemble
    for w in [0.02, 0.03, 0.05]:
        blend = sub3558.copy()
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            blend[tc] = np.clip(w * sub_ens[tc] + (1 - w) * sub3558[tc], 0, 1)
        save_submission(blend, f"{int(w*100)}% ROCKET-ensemble + {int((1-w)*100)}% Sub3558")

    # 4. Per-target optimized blend
    per_target = sub3558.copy()
    weights_used = {}
    for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
        r, _ = pearsonr(sub_ens[tc], sub3558[tc])
        if r < 0.80:
            w = 0.04
        elif r < 0.90:
            w = 0.03
        else:
            w = 0.02
        per_target[tc] = np.clip(w * sub_ens[tc] + (1 - w) * sub3558[tc], 0, 1)
        weights_used[tc] = w
        print(f"  Per-target {tc}: r={r:.4f} -> w={w}", flush=True)
    save_submission(per_target, f"Per-target ROCKET-ensemble + Sub3558")

    # 5. MEGA: ROCKET + FPCA + LGBM on Sub3558
    print(f"\n--- MEGA BLEND: ROCKET + FPCA + LGBM ---", flush=True)
    try:
        sub_fpca = pd.read_csv(SUBMISSION_DIR / "submission_3607.csv")
        sub_lgbm = pd.read_csv(SUBMISSION_DIR / "submission_3582.csv")

        mega = sub3558.copy()
        # angle: all sources moderately diverse, small weights
        mega["scaled_angle"] = np.clip(
            0.96 * sub3558["scaled_angle"] +
            0.02 * sub_ens["scaled_angle"] +
            0.01 * sub_fpca["scaled_angle"] +
            0.01 * sub_lgbm["scaled_angle"],
            0, 1
        )
        # depth: ROCKET and FPCA both strong here
        mega["scaled_depth"] = np.clip(
            0.92 * sub3558["scaled_depth"] +
            0.04 * sub_ens["scaled_depth"] +
            0.03 * sub_fpca["scaled_depth"] +
            0.01 * sub_lgbm["scaled_depth"],
            0, 1
        )
        # LR: spread across diverse sources
        mega["scaled_left_right"] = np.clip(
            0.93 * sub3558["scaled_left_right"] +
            0.03 * sub_ens["scaled_left_right"] +
            0.02 * sub_fpca["scaled_left_right"] +
            0.02 * sub_lgbm["scaled_left_right"],
            0, 1
        )
        save_submission(mega, "MEGA: Sub3558 + ROCKET + FPCA + LGBM per-target")

        # Also conservative mega
        mega2 = sub3558.copy()
        mega2["scaled_angle"] = np.clip(
            0.97 * sub3558["scaled_angle"] +
            0.01 * sub_ens["scaled_angle"] +
            0.01 * sub_fpca["scaled_angle"] +
            0.01 * sub_lgbm["scaled_angle"],
            0, 1
        )
        mega2["scaled_depth"] = np.clip(
            0.95 * sub3558["scaled_depth"] +
            0.02 * sub_ens["scaled_depth"] +
            0.02 * sub_fpca["scaled_depth"] +
            0.01 * sub_lgbm["scaled_depth"],
            0, 1
        )
        mega2["scaled_left_right"] = np.clip(
            0.95 * sub3558["scaled_left_right"] +
            0.02 * sub_ens["scaled_left_right"] +
            0.02 * sub_fpca["scaled_left_right"] +
            0.01 * sub_lgbm["scaled_left_right"],
            0, 1
        )
        save_submission(mega2, "MEGA conservative: Sub3558 + 1-2% each ROCKET/FPCA/LGBM")

    except Exception as e:
        print(f"  Mega blend error: {e}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"rocket_v2_{ts}.json", "w") as f:
        json.dump({
            "all_results": {k: float(v) for k, v in all_results.items()},
            "weights_used": weights_used,
        }, f, indent=2)


if __name__ == "__main__":
    main()
