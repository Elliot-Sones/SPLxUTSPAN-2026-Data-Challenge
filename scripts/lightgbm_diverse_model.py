"""
LightGBM Diverse Model for Basketball Free Throw Prediction

Tree-based models capture NON-LINEAR interactions that Ridge cannot.
This is fundamentally different from:
- Ridge pipeline (linear model with kernel weighting)
- CNNs (deep learning on temporal sequences)
- GP (kernel-based Bayesian)

Key design:
1. Per-player LightGBM with honest LOO
2. Rich feature set: positions, velocities, accelerations, joint angles,
   multi-frame features (not just one frame)
3. Low leaf count + high regularization to prevent overfit on small N
4. Multi-frame input: features from 5 frames around release
"""

from __future__ import annotations

import fcntl
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
    "nose", "right_knee", "left_knee",
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


def extract_rich_features(X_3d, kp_index, target_frame, multi_frame_offsets=(-6, -3, 0, 3, 6)):
    """Extract rich features for tree models.

    Unlike Ridge which needs compact features, tree models can handle
    more features and will automatically select the most useful ones.

    Features:
    - Multi-frame positions (hip-relative and hoop-relative)
    - Multi-frame velocities
    - Joint angles at release
    - Velocity magnitudes and directions
    - Acceleration features
    """
    N = X_3d.shape[0]
    dt = 1.0 / 60.0
    features = []
    feature_names = []

    rh_idx = kp_index["right_hip"]
    lh_idx = kp_index["left_hip"]

    for offset in multi_frame_offsets:
        f = np.clip(target_frame + offset, 1, 238)
        hip = (X_3d[:, f, rh_idx, :] + X_3d[:, f, lh_idx, :]) / 2.0

        for j in KEY_JOINTS:
            if j not in kp_index:
                continue
            ji = kp_index[j]
            pos = X_3d[:, f, ji, :]
            rel = pos - hip
            hoop_rel = pos - HOOP_POS[np.newaxis, :]

            for ci, cn in enumerate(["x", "y", "z"]):
                features.append(rel[:, ci])
                feature_names.append(f"{j}_rel_{cn}_f{offset}")
                features.append(hoop_rel[:, ci])
                feature_names.append(f"{j}_hoop_{cn}_f{offset}")

            # Velocity at this frame
            vel = (X_3d[:, min(f+1, 239), ji, :] - X_3d[:, max(f-1, 0), ji, :]) / (2*dt)
            vmag = np.linalg.norm(vel, axis=1)
            features.append(vmag)
            feature_names.append(f"{j}_vmag_f{offset}")

            for ci, cn in enumerate(["x", "y", "z"]):
                features.append(vel[:, ci])
                feature_names.append(f"{j}_vel_{cn}_f{offset}")

    # Joint angles at release frame
    f = target_frame
    # Elbow angle
    rs = X_3d[:, f, kp_index.get("right_shoulder", 0), :]
    re = X_3d[:, f, kp_index.get("right_elbow", 0), :]
    rw = X_3d[:, f, kp_index.get("right_wrist", 0), :]

    v1, v2 = rs - re, rw - re
    cos_e = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)+1e-8)
    features.append(np.arccos(np.clip(cos_e, -1, 1)))
    feature_names.append("elbow_angle")

    # Shoulder angle
    rhi = X_3d[:, f, kp_index.get("right_hip", 0), :]
    v1, v2 = rhi - rs, re - rs
    cos_s = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)+1e-8)
    features.append(np.arccos(np.clip(cos_s, -1, 1)))
    feature_names.append("shoulder_angle")

    # Knee angle
    if all(j in kp_index for j in ["right_hip", "right_knee"]):
        rk = X_3d[:, f, kp_index["right_knee"], :]
        v1_k = rhi - rk
        ra_idx = kp_index.get("right_ankle", kp_index.get("right_knee", 0))
        ra = X_3d[:, f, ra_idx, :]
        v2_k = ra - rk
        cos_k = np.sum(v1_k*v2_k, axis=1) / (np.linalg.norm(v1_k, axis=1)*np.linalg.norm(v2_k, axis=1)+1e-8)
        features.append(np.arccos(np.clip(cos_k, -1, 1)))
        feature_names.append("knee_angle")

    # Acceleration at release
    f = target_frame
    for j in KEY_JOINTS[:5]:
        if j not in kp_index:
            continue
        ji = kp_index[j]
        if 1 < f < 238:
            acc = (X_3d[:, f+1, ji, :] - 2*X_3d[:, f, ji, :] + X_3d[:, f-1, ji, :]) / (dt**2)
        else:
            acc = np.zeros((N, 3))
        acc_mag = np.linalg.norm(acc, axis=1)
        features.append(acc_mag)
        feature_names.append(f"{j}_acc_mag")

    # Wrist-to-hoop distance and direction
    wrist = X_3d[:, target_frame, kp_index.get("right_wrist", 0), :]
    to_hoop = HOOP_POS[np.newaxis, :] - wrist
    features.append(np.linalg.norm(to_hoop, axis=1))
    feature_names.append("wrist_hoop_dist")

    # Vertical components
    features.append(wrist[:, 2])
    feature_names.append("wrist_height")
    features.append(wrist[:, 2] - HOOP_POS[2])
    feature_names.append("wrist_vs_hoop_height")

    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


def lgbm_per_player_loo(X_train, y_train, pids_train, params):
    """Per-player LightGBM with honest LOO."""
    N = len(y_train)
    predictions = np.zeros(N)
    unique_pids = np.unique(pids_train)

    for pid in unique_pids:
        mask = pids_train == pid
        idx = np.where(mask)[0]
        n_p = len(idx)
        X_p = X_train[idx]
        y_p = y_train[idx]

        for i_local in range(n_p):
            loo_mask = np.ones(n_p, dtype=bool)
            loo_mask[i_local] = False

            X_loo = X_p[loo_mask]
            y_loo = y_p[loo_mask]

            ds = lgb.Dataset(X_loo, y_loo, free_raw_data=False)
            cb = [lgb.log_evaluation(period=0)]
            model = lgb.train(params, ds, num_boost_round=params.get("n_estimators", 100),
                            callbacks=cb)
            pred = model.predict(X_p[i_local:i_local+1])[0]
            predictions[idx[i_local]] = pred

    return predictions


def lgbm_per_player_predict(X_train, y_train, pids_train,
                             X_test, pids_test, params):
    """Per-player LightGBM test predictions."""
    N_test = len(X_test)
    predictions = np.zeros(N_test)
    unique_pids = np.unique(pids_train)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        if te_mask.sum() == 0:
            continue

        X_p = X_train[tr_mask]
        y_p = y_train[tr_mask]
        X_t = X_test[te_mask]

        ds = lgb.Dataset(X_p, y_p, free_raw_data=False)
        cb = [lgb.log_evaluation(period=0)]
        model = lgb.train(params, ds, num_boost_round=params.get("n_estimators", 100),
                        callbacks=cb)
        pred = model.predict(X_t)
        predictions[te_mask] = pred

    return predictions


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
    print("LightGBM Diverse Model", flush=True)
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

    print(f"Train: {len(pids_train)}, Test: {len(pids_test)}", flush=True)

    # LightGBM parameters optimized for small N (~66-74 per player)
    # Key: very few leaves, lots of regularization, low learning rate
    param_configs = {
        "conservative": {
            "objective": "regression",
            "metric": "mse",
            "n_estimators": 50,
            "num_leaves": 4,  # Very shallow trees
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_samples": 10,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "verbose": -1,
        },
        "moderate": {
            "objective": "regression",
            "metric": "mse",
            "n_estimators": 100,
            "num_leaves": 7,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_samples": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "verbose": -1,
        },
        "aggressive": {
            "objective": "regression",
            "metric": "mse",
            "n_estimators": 200,
            "num_leaves": 10,
            "max_depth": 5,
            "learning_rate": 0.02,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.3,
            "reg_lambda": 3.0,
            "verbose": -1,
        },
    }

    results = {}
    best_test_preds = {}

    for tidx, tname in enumerate(TARGETS):
        tf = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        print(f"\n{'='*60}", flush=True)
        print(f"TARGET: {tname} (frame {tf})", flush=True)
        print(f"{'='*60}", flush=True)

        # Extract rich features
        X_train_feat, feat_names = extract_rich_features(X_3d_train, kp_index, tf)
        X_test_feat, _ = extract_rich_features(X_3d_test, kp_index, tf)
        print(f"  Features: {X_train_feat.shape[1]}", flush=True)

        best_mse = float('inf')
        best_config = None

        for config_name, params in param_configs.items():
            t0 = time.time()
            loo_preds = lgbm_per_player_loo(X_train_feat, y, pids_train, params)
            mse = np.mean((loo_preds - y) ** 2)
            elapsed = time.time() - t0
            print(f"  {config_name}: LOO MSE={mse:.6f} ({elapsed:.1f}s)", flush=True)

            results[f"{tname}_{config_name}"] = {"mse": float(mse)}

            if mse < best_mse:
                best_mse = mse
                best_config = config_name

        # Use best config for test predictions
        print(f"  Best: {best_config} (MSE={best_mse:.6f})", flush=True)

        test_pred = lgbm_per_player_predict(
            X_train_feat, y, pids_train,
            X_test_feat, pids_test,
            param_configs[best_config]
        )
        best_test_preds[tname] = np.clip(test_pred, 0, 1)
        results[f"{tname}_best"] = {"config": best_config, "mse": float(best_mse)}

    # Summary
    mean_mse = np.mean([results[f"{t}_best"]["mse"] for t in TARGETS])
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY (mean LOO = {mean_mse:.6f})", flush=True)
    for tname in TARGETS:
        r = results[f"{tname}_best"]
        print(f"  {tname}: {r['config']} MSE={r['mse']:.6f}", flush=True)

    # Diversity analysis
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": best_test_preds["angle"],
        "scaled_depth": best_test_preds["depth"],
        "scaled_left_right": best_test_preds["left_right"],
    })

    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            r, _ = pearsonr(sub_df[tc], sub3558[tc])
            print(f"  {tc}: r={r:.4f}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

    # Also check vs Sub3411 (Ridge base)
    print(f"\n--- Diversity vs Sub3411 (Ridge) ---", flush=True)
    try:
        sub3411 = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            r, _ = pearsonr(sub_df[tc], sub3411[tc])
            print(f"  {tc}: r={r:.4f}", flush=True)
    except Exception as e:
        print(f"  Error: {e}", flush=True)

    # Save standalone
    save_submission(sub_df, f"LightGBM standalone (mean LOO={mean_mse:.6f})")

    # Blends with Sub3558
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for w in [0.02, 0.05, 0.08, 0.10]:
            blend = sub_df.copy()
            for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
                blend[tc] = np.clip(w * sub_df[tc] + (1-w) * sub3558[tc], 0, 1)
            save_submission(blend, f"{int(w*100)}% LightGBM + {int((1-w)*100)}% Sub3558")
    except Exception as e:
        print(f"  Blend error: {e}", flush=True)

    # Per-target best blend: use LightGBM only for targets where it's diverse
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        per_target = sub3558.copy()
        for tidx, (tname, tc) in enumerate(zip(TARGETS, ["scaled_angle", "scaled_depth", "scaled_left_right"])):
            r, _ = pearsonr(sub_df[tc], sub3558[tc])
            # Use more LightGBM for more diverse targets
            if r < 0.85:
                w = 0.08  # diverse: use 8%
            elif r < 0.92:
                w = 0.05
            else:
                w = 0.02  # similar: use 2%
            per_target[tc] = np.clip(w * sub_df[tc] + (1-w) * sub3558[tc], 0, 1)
            print(f"  Per-target: {tname} r={r:.4f} -> w={w}", flush=True)
        save_submission(per_target, "Per-target LightGBM blend with Sub3558")
    except Exception as e:
        print(f"  Per-target error: {e}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"lightgbm_diverse_run_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
