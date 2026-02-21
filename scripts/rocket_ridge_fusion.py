"""
ROCKET + Ridge Feature Fusion - Build a NEW base pipeline, not a blend.

Instead of: 98% old_model + 2% new_model (which never works)
Do: Combine ROCKET features + Ridge features into ONE model

Strategy:
1. Extract ROCKET features (random convolutional kernels on trajectories)
2. Extract Ridge pipeline features (hoop-relative + PLS + joint angles)
3. Concatenate both feature sets
4. Per-player RidgeCV on the combined feature space
5. The model learns WHICH features matter - not us guessing blend weights

This is fundamentally different because:
- The model sees BOTH feature sets simultaneously
- It can learn interactions between ROCKET and Ridge features
- No arbitrary blend weight guessing
- Ridge regression with proper regularization handles the high dimensionality

Also try: stacking (use ROCKET OOF + Ridge OOF as meta-features)
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
from aeon.transformations.collection.convolution_based import MiniRocket
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP_POS = np.array([5.25, -25.0, 10.0])

KEY_JOINTS_14 = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_first_finger_mcp", "right_second_finger_mcp",
    "left_shoulder", "left_elbow", "left_wrist",
    "neck", "mid_hip", "right_hip", "left_hip",
    "right_knee", "left_knee",
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


def extract_ridge_features(X_3d, kp_index, target_frame):
    """Extract the same hoop-relative features used in the Ridge pipeline."""
    N = X_3d.shape[0]
    rh = kp_index.get("right_hip", 0)
    lh = kp_index.get("left_hip", 0)

    features = []
    frame = min(target_frame, 239)

    # Hip center
    hip = (X_3d[:, frame, rh, :] + X_3d[:, frame, lh, :]) / 2.0

    # Hoop-relative transform
    forward = HOOP_POS[:2] - hip[:, :2]
    fn = np.linalg.norm(forward, axis=1, keepdims=True)
    fn = np.maximum(fn, 1e-6)
    forward = forward / fn

    for joint_name in kp_index:
        ji = kp_index[joint_name]
        pos = X_3d[:, frame, ji, :] - hip  # hip-relative
        pos = np.nan_to_num(pos, nan=0.0)
        features.append(pos)

        # Velocity at target frame
        if frame > 0:
            vel = X_3d[:, frame, ji, :] - X_3d[:, frame - 1, ji, :]
            vel = np.nan_to_num(vel, nan=0.0)
            features.append(vel)

    # Multi-frame: also grab features at frame-5 and frame+5
    for offset in [-5, 5]:
        f2 = min(max(frame + offset, 0), 239)
        for joint_name in KEY_JOINTS_14:
            if joint_name not in kp_index:
                continue
            ji = kp_index[joint_name]
            pos = X_3d[:, f2, ji, :] - hip
            pos = np.nan_to_num(pos, nan=0.0)
            features.append(pos)

    X = np.hstack(features)
    return X


def extract_rocket_features(X_3d_train, X_3d_test, kp_index,
                            frame_range=(100, 200), subsample=3,
                            n_kernels=5000, seeds=[42, 7, 99]):
    """Multi-seed ROCKET features."""
    def prepare(X_3d):
        N = X_3d.shape[0]
        frames = np.arange(frame_range[0], frame_range[1], subsample)
        rh = kp_index.get("right_hip", 0)
        lh = kp_index.get("left_hip", 0)
        hip = (X_3d[:, frames, rh, :] + X_3d[:, frames, lh, :]) / 2.0

        channels = []
        for joint in KEY_JOINTS_14:
            if joint not in kp_index:
                continue
            ji = kp_index[joint]
            pos = np.nan_to_num(X_3d[:, frames, ji, :] - hip, nan=0.0)
            for c in range(3):
                channels.append(pos[:, :, c])
            vel = np.diff(pos, axis=1, prepend=pos[:, :1, :])
            for c in range(3):
                channels.append(vel[:, :, c])
        return np.stack(channels, axis=1).astype(np.float32)

    X_tr_raw = prepare(X_3d_train)
    X_te_raw = prepare(X_3d_test)

    all_tr_feats = []
    all_te_feats = []

    for seed in seeds:
        rocket = MiniRocket(n_kernels=n_kernels, random_state=seed)
        tr_feat = rocket.fit_transform(X_tr_raw)
        te_feat = rocket.transform(X_te_raw)
        all_tr_feats.append(tr_feat)
        all_te_feats.append(te_feat)

    # Average across seeds for stability
    X_tr_rocket = np.mean(all_tr_feats, axis=0)
    X_te_rocket = np.mean(all_te_feats, axis=0)

    return X_tr_rocket, X_te_rocket


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


def main():
    print("=" * 70, flush=True)
    print("ROCKET + Ridge Feature FUSION Pipeline", flush=True)
    print("Build a new base model, not a blend", flush=True)
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

    y_scaled = np.zeros_like(y_raw)
    for tidx, tname in enumerate(TARGETS):
        scaler = joblib.load(DATA_DIR / f"scaler_{tname}.pkl")
        y_scaled[:, tidx] = scaler.transform(y_raw[:, tidx].reshape(-1, 1)).ravel()

    # Extract ROCKET features (shared across targets)
    print("\n--- Extracting ROCKET features (3-seed ensemble) ---", flush=True)
    X_tr_rocket, X_te_rocket = extract_rocket_features(
        X_3d_train, X_3d_test, kp_index,
        frame_range=(100, 200), subsample=3,
        n_kernels=5000, seeds=[42, 7, 99]
    )
    print(f"  ROCKET features: {X_tr_rocket.shape[1]}", flush=True)

    alphas = np.logspace(-3, 5, 40)
    sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")

    results = {}

    for tidx, tname in enumerate(TARGETS):
        target_frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        # Extract Ridge features for this target's frame
        X_tr_ridge = extract_ridge_features(X_3d_train, kp_index, target_frame)
        X_te_ridge = extract_ridge_features(X_3d_test, kp_index, target_frame)
        print(f"\n  {tname}: Ridge features={X_tr_ridge.shape[1]}, "
              f"ROCKET features={X_tr_rocket.shape[1]}", flush=True)

        # ============================
        # Strategy A: Feature concatenation
        # ============================
        X_tr_fused = np.hstack([X_tr_ridge, X_tr_rocket])
        X_te_fused = np.hstack([X_te_ridge, X_te_rocket])

        loo_preds_fused = np.zeros(len(y))
        test_preds_fused = np.zeros(len(pids_test))

        loo_preds_ridge_only = np.zeros(len(y))
        test_preds_ridge_only = np.zeros(len(pids_test))

        loo_preds_rocket_only = np.zeros(len(y))
        test_preds_rocket_only = np.zeros(len(pids_test))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            idx_tr = np.where(tr_mask)[0]
            n_p = len(idx_tr)
            y_p = y[tr_mask]

            for label, X_tr_src, X_te_src, loo_arr, te_arr in [
                ("fused", X_tr_fused, X_te_fused, loo_preds_fused, test_preds_fused),
                ("ridge", X_tr_ridge, X_te_ridge, loo_preds_ridge_only, test_preds_ridge_only),
                ("rocket", X_tr_rocket, X_te_rocket, loo_preds_rocket_only, test_preds_rocket_only),
            ]:
                sc = StandardScaler()
                X_p = sc.fit_transform(X_tr_src[tr_mask])
                X_p_te = sc.transform(X_te_src[te_mask]) if te_mask.sum() > 0 else None

                # LOO
                for i in range(n_p):
                    tr_idx = [j for j in range(n_p) if j != i]
                    ridge = RidgeCV(alphas=alphas)
                    ridge.fit(X_p[tr_idx], y_p[tr_idx])
                    loo_arr[idx_tr[i]] = np.clip(ridge.predict(X_p[[i]])[0], 0, 1)

                # Test
                if X_p_te is not None:
                    ridge_full = RidgeCV(alphas=alphas)
                    ridge_full.fit(X_p, y_p)
                    te_arr[te_mask] = np.clip(ridge_full.predict(X_p_te), 0, 1)

        mse_fused = np.mean((loo_preds_fused - y) ** 2)
        mse_ridge = np.mean((loo_preds_ridge_only - y) ** 2)
        mse_rocket = np.mean((loo_preds_rocket_only - y) ** 2)

        print(f"  {tname} LOO MSE:", flush=True)
        print(f"    Ridge-only:  {mse_ridge:.6f}", flush=True)
        print(f"    ROCKET-only: {mse_rocket:.6f}", flush=True)
        print(f"    FUSED:       {mse_fused:.6f} ({'BETTER' if mse_fused < min(mse_ridge, mse_rocket) else 'worse'})", flush=True)

        results[tname] = {
            "ridge_only": mse_ridge,
            "rocket_only": mse_rocket,
            "fused": mse_fused,
            "fused_preds": test_preds_fused,
            "ridge_preds": test_preds_ridge_only,
            "rocket_preds": test_preds_rocket_only,
            "fused_oof": loo_preds_fused,
        }

    # ============================
    # Strategy B: Stacking (OOF meta-features)
    # ============================
    print(f"\n{'='*60}", flush=True)
    print("Strategy B: OOF Stacking", flush=True)
    print(f"{'='*60}", flush=True)

    for tidx, tname in enumerate(TARGETS):
        y = y_scaled[:, tidx]

        # Meta-features: OOF predictions from Ridge and ROCKET
        X_meta_tr = np.column_stack([
            results[tname]["fused_oof"],  # fused model OOF
        ])

        # For test, use the fused predictions directly
        # (stacking with 1 base model = identity, skip this)
        print(f"  {tname}: stacking with single base = identity (skip)", flush=True)

    # ============================
    # Create submissions
    # ============================
    print(f"\n{'='*60}", flush=True)
    print("Creating Submissions", flush=True)
    print(f"{'='*60}", flush=True)

    # 1. Fused standalone
    sub_fused = pd.DataFrame({"id": ids_test})
    mean_fused = 0
    for tname in TARGETS:
        col = f"scaled_{tname}" if tname != "left_right" else "scaled_left_right"
        sub_fused[col] = np.clip(results[tname]["fused_preds"], 0, 1)
        mean_fused += results[tname]["fused"] / 3

    print(f"\n  Fused standalone: mean LOO={mean_fused:.6f}", flush=True)
    print(f"  Diversity vs Sub3558:", flush=True)
    for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
        r, _ = pearsonr(sub_fused[tc], sub3558[tc])
        print(f"    {tc}: r={r:.4f}", flush=True)

    save_submission(sub_fused, f"FUSED Ridge+ROCKET (mean LOO={mean_fused:.6f})")

    # 2. Best-of: pick ridge or fused per target based on LOO
    sub_best = pd.DataFrame({"id": ids_test})
    for tname in TARGETS:
        col = f"scaled_{tname}" if tname != "left_right" else "scaled_left_right"
        if results[tname]["fused"] < results[tname]["ridge_only"]:
            sub_best[col] = np.clip(results[tname]["fused_preds"], 0, 1)
            print(f"  {tname}: using FUSED ({results[tname]['fused']:.6f})", flush=True)
        else:
            sub_best[col] = np.clip(results[tname]["ridge_preds"], 0, 1)
            print(f"  {tname}: using RIDGE ({results[tname]['ridge_only']:.6f})", flush=True)
    save_submission(sub_best, "Best-of Ridge/Fused per target")

    # 3. Blend fused with Sub3558 (if fused is better than pure ridge)
    for w in [0.10, 0.20, 0.30, 0.50]:
        blend = sub3558.copy()
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            blend[tc] = np.clip(w * sub_fused[tc] + (1 - w) * sub3558[tc], 0, 1)
        save_submission(blend, f"{int(w*100)}% Fused + {int((1-w)*100)}% Sub3558")

    # 4. Per-player allocation: use fused model more for weak players
    # Like Sub3558 did with CNN, but now with fused ROCKET+Ridge
    print(f"\n--- Per-Player Fused Allocation ---", flush=True)
    per_player_sub = sub3558.copy()
    for tidx, tname in enumerate(TARGETS):
        tc = f"scaled_{tname}" if tname != "left_right" else "scaled_left_right"
        y = y_scaled[:, tidx]

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            if te_mask.sum() == 0:
                continue

            # Player LOO MSE from fused model
            y_p = y[tr_mask]
            oof_p = results[tname]["fused_oof"][tr_mask]
            mse_p = np.mean((oof_p - y_p) ** 2)

            # Compare to Ridge-only LOO for this player
            # Use more fused signal for players where fused is better
            # Use less for players where fused is worse
            if mse_p < 0.006:
                w = 0.15  # good player, fused captures signal well
            elif mse_p < 0.010:
                w = 0.20  # medium player, lean into fused
            else:
                w = 0.25  # weak player, fused may help more

            te_idx = np.where(te_mask)[0]
            per_player_sub.loc[per_player_sub.index[te_idx], tc] = np.clip(
                w * results[tname]["fused_preds"][te_idx] +
                (1 - w) * sub3558[tc].values[te_idx],
                0, 1
            )
            print(f"  P{pid} {tname}: fused LOO MSE={mse_p:.6f}, w={w}", flush=True)

    save_submission(per_player_sub, "Per-player fused ROCKET+Ridge allocation into Sub3558")

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"rocket_ridge_fusion_{ts}.json", "w") as f:
        json.dump({
            k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
            for k, v in results.items()
        }, f, indent=2)


if __name__ == "__main__":
    main()
