"""
Path Signature Features for Basketball Free Throw Prediction

Path signatures are iterated integrals that capture the FULL temporal structure
of a multivariate path - including order, interactions between channels, and
nonlinear dynamics. Unlike single-frame features, they encode HOW you get to
the release position, not just WHERE you are.

Level 1: mean increments (like velocity)
Level 2: pairwise channel interactions (like "elbow goes up WHILE wrist extends")
Level 3: triple interactions (higher-order coordination patterns)

Strategy:
1. Extract key joint trajectories around release (frames 100-180)
2. Compute truncated path signatures (level 2-3)
3. Feed into per-player locally-weighted Ridge
4. Blend with Sub3558 (current best)
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
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP_POS = np.array([5.25, -25.0, 10.0])

# Key joints for signatures - focus on shooting mechanics
SIG_JOINTS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_first_finger_mcp", "right_second_finger_mcp",
    "left_shoulder", "left_wrist", "left_elbow",
    "right_hip", "left_hip",
    "nose", "right_knee",
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


def compute_signature(path, depth=2):
    """Compute path signature using esig.

    path: (T, d) array - a d-dimensional path with T time steps
    depth: truncation level (2 or 3)

    Returns: signature feature vector
    """
    import esig
    # esig.stream2sig expects (T, d) float64 array
    path = np.asarray(path, dtype=np.float64)

    # Handle NaN
    if np.any(np.isnan(path)):
        path = np.nan_to_num(path, nan=0.0)

    sig = esig.stream2sig(path, depth)
    return sig


def extract_signature_features(X_3d, kp_index, frame_start, frame_end,
                                subsample=2, sig_depth=2):
    """Extract path signature features for all shots.

    Args:
        X_3d: (N, 240, n_kp, 3) array
        frame_start, frame_end: frame range
        subsample: take every nth frame
        sig_depth: signature truncation level
    """
    N = X_3d.shape[0]

    # Get joint indices
    joint_indices = [kp_index[j] for j in SIG_JOINTS if j in kp_index]
    n_joints = len(joint_indices)

    # Hip center for normalization
    rh_idx = kp_index["right_hip"]
    lh_idx = kp_index["left_hip"]

    features_list = []
    for i in range(N):
        # Extract trajectory for selected joints
        frames = range(frame_start, frame_end, subsample)
        T = len(list(frames))

        # Hip-center the data
        hip = (X_3d[i, frame_start:frame_end:subsample, rh_idx, :] +
               X_3d[i, frame_start:frame_end:subsample, lh_idx, :]) / 2  # (T, 3)

        # Build path: (T, n_joints * 3)
        path_channels = []
        for ji in joint_indices:
            pos = X_3d[i, frame_start:frame_end:subsample, ji, :]  # (T, 3)
            centered = pos - hip
            path_channels.append(centered)

        path = np.concatenate(path_channels, axis=1)  # (T, n_joints*3)

        # Also add time as a channel (helps signatures capture speed)
        t_channel = np.linspace(0, 1, T).reshape(-1, 1)
        path = np.concatenate([path, t_channel], axis=1)  # (T, n_joints*3 + 1)

        # Normalize path to unit scale per channel for numerical stability
        path_std = np.std(path, axis=0, keepdims=True)
        path_std[path_std < 1e-8] = 1.0
        path_norm = path / path_std

        # Compute signature
        sig = compute_signature(path_norm, depth=sig_depth)
        features_list.append(sig)

    X_sig = np.vstack(features_list)
    X_sig = np.nan_to_num(X_sig, nan=0.0, posinf=0.0, neginf=0.0)
    return X_sig


def extract_release_features(X_3d, kp_index, frame_idx):
    """Extract standard single-frame features for comparison."""
    N = X_3d.shape[0]
    features = []

    rh_idx = kp_index["right_hip"]
    lh_idx = kp_index["left_hip"]
    hip = (X_3d[:, frame_idx, rh_idx, :] + X_3d[:, frame_idx, lh_idx, :]) / 2

    for j in SIG_JOINTS:
        if j not in kp_index:
            continue
        ji = kp_index[j]
        pos = X_3d[:, frame_idx, ji, :]
        rel = pos - hip
        hoop_rel = pos - HOOP_POS[np.newaxis, :]
        for ci in range(3):
            features.append(rel[:, ci])
            features.append(hoop_rel[:, ci])

    X = np.column_stack(features)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def locally_weighted_loo(X, y, pids, bw_q=0.3, alpha=10.0, use_pls=True, pls_nc=5):
    """LOO with optional PLS (honest - refit per fold)."""
    n = len(y)
    preds = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        pmask = pids == pids[i]
        tmask = mask & pmask
        if tmask.sum() < 5:
            tmask = mask

        X_tr = X[tmask]
        y_tr = y[tmask]
        X_te = X[i:i+1]

        if use_pls and X_tr.shape[1] > 20:
            nc = min(pls_nc, min(X_tr.shape) - 1)
            if nc >= 2:
                try:
                    pls = PLSRegression(n_components=nc)
                    pls.fit(X_tr, y_tr)
                    pls_tr = pls.transform(X_tr)
                    pls_te = pls.transform(X_te)
                    X_tr = np.hstack([X_tr, pls_tr])
                    X_te = np.hstack([X_te, pls_te])
                except Exception:
                    pass

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        dists = cdist(X_te_s, X_tr_s, metric='euclidean').flatten()
        bw = np.quantile(dists, bw_q)
        if bw < 1e-8:
            bw = 1.0
        w = np.exp(-0.5 * (dists / bw) ** 2)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_s, y_tr, sample_weight=w)
        preds[i] = np.clip(ridge.predict(X_te_s)[0], 0, 1)

    return preds


def predict_test(X_train, y_train, pids_train, X_test, pids_test,
                  bw_q=0.3, alpha=10.0, use_pls=True, pls_nc=5):
    """Predict test with per-player locally-weighted Ridge."""
    preds = np.zeros(len(X_test))

    for pid in np.unique(pids_test):
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        if tr_mask.sum() < 5:
            tr_mask = np.ones(len(pids_train), dtype=bool)

        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]

        if use_pls and X_tr.shape[1] > 20:
            nc = min(pls_nc, min(X_tr.shape) - 1)
            if nc >= 2:
                try:
                    pls = PLSRegression(n_components=nc)
                    pls.fit(X_tr, y_tr)
                    pls_tr = pls.transform(X_tr)
                    pls_te = pls.transform(X_te)
                    X_tr = np.hstack([X_tr, pls_tr])
                    X_te = np.hstack([X_te, pls_te])
                except Exception:
                    pass

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        te_indices = np.where(te_mask)[0]
        for i in range(te_mask.sum()):
            dists = cdist(X_te_s[i:i+1], X_tr_s, metric='euclidean').flatten()
            bw = np.quantile(dists, bw_q)
            if bw < 1e-8:
                bw = 1.0
            w = np.exp(-0.5 * (dists / bw) ** 2)

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=w)
            preds[te_indices[i]] = np.clip(ridge.predict(X_te_s[i:i+1])[0], 0, 1)

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
    print("Path Signature Features for Free Throw Prediction", flush=True)
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
    print(f"Joints for signatures: {len(SIG_JOINTS)}", flush=True)

    all_results = {}
    test_preds_all = {}

    # Test multiple signature configurations
    configs = [
        {"name": "sig2_100_180_s2", "depth": 2, "start": 100, "end": 180, "sub": 2},
        {"name": "sig2_120_170_s2", "depth": 2, "start": 120, "end": 170, "sub": 2},
        {"name": "sig2_130_175_s3", "depth": 2, "start": 130, "end": 175, "sub": 3},
        {"name": "sig3_130_170_s3", "depth": 3, "start": 130, "end": 170, "sub": 3},
    ]

    for cfg in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {cfg['name']} (depth={cfg['depth']}, frames={cfg['start']}-{cfg['end']}, sub={cfg['sub']})",
              flush=True)
        print(f"{'='*60}", flush=True)

        # Compute signatures
        print("  Computing signatures (train)...", flush=True)
        t0 = time.time()
        X_sig_train = extract_signature_features(
            X_3d_train, kp_index, cfg["start"], cfg["end"], cfg["sub"], cfg["depth"])
        print(f"  Train signatures: {X_sig_train.shape} in {time.time()-t0:.1f}s", flush=True)

        print("  Computing signatures (test)...", flush=True)
        X_sig_test = extract_signature_features(
            X_3d_test, kp_index, cfg["start"], cfg["end"], cfg["sub"], cfg["depth"])
        print(f"  Test signatures: {X_sig_test.shape}", flush=True)

        cfg_results = {}

        for tidx, tname in enumerate(TARGETS):
            frame = TARGET_FRAMES[tname]
            y = y_scaled[:, tidx]

            # Also get release-frame features
            X_rel_train = extract_release_features(X_3d_train, kp_index, frame)
            X_rel_test = extract_release_features(X_3d_test, kp_index, frame)

            # Test 3 modes:
            # 1. Signatures only
            # 2. Release features only
            # 3. Signatures + release features

            modes = {
                "sig_only": (X_sig_train, X_sig_test),
                "release_only": (X_rel_train, X_rel_test),
                "sig_plus_release": (
                    np.hstack([X_sig_train, X_rel_train]),
                    np.hstack([X_sig_test, X_rel_test]),
                ),
            }

            print(f"\n  --- {tname} ---", flush=True)

            best_mode = None
            best_mse = float('inf')

            for mode_name, (X_tr, X_te) in modes.items():
                preds = locally_weighted_loo(X_tr, y, pids_train,
                                           bw_q=0.3, alpha=10.0,
                                           use_pls=True, pls_nc=5)
                mse = np.mean((preds - y) ** 2)
                print(f"    {mode_name} ({X_tr.shape[1]} feats): LOO={mse:.6f}", flush=True)

                if mse < best_mse:
                    best_mse = mse
                    best_mode = mode_name

            cfg_results[tname] = {"best_mode": best_mode, "best_mse": float(best_mse)}
            print(f"    Best: {best_mode} (MSE={best_mse:.6f})", flush=True)

        mean_mse = np.mean([cfg_results[t]["best_mse"] for t in TARGETS])
        print(f"\n  Config mean: {mean_mse:.6f}", flush=True)
        all_results[cfg["name"]] = {"mean": float(mean_mse), "per_target": cfg_results}

    # Find best config
    best_cfg_name = min(all_results, key=lambda k: all_results[k]["mean"])
    best_cfg = [c for c in configs if c["name"] == best_cfg_name][0]
    print(f"\n{'='*60}", flush=True)
    print(f"BEST CONFIG: {best_cfg_name} (mean={all_results[best_cfg_name]['mean']:.6f})", flush=True)
    print(f"{'='*60}", flush=True)

    # Generate test predictions with best config
    print("\nGenerating test predictions...", flush=True)
    X_sig_train = extract_signature_features(
        X_3d_train, kp_index, best_cfg["start"], best_cfg["end"],
        best_cfg["sub"], best_cfg["depth"])
    X_sig_test = extract_signature_features(
        X_3d_test, kp_index, best_cfg["start"], best_cfg["end"],
        best_cfg["sub"], best_cfg["depth"])

    for tidx, tname in enumerate(TARGETS):
        frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        X_rel_train = extract_release_features(X_3d_train, kp_index, frame)
        X_rel_test = extract_release_features(X_3d_test, kp_index, frame)

        best_mode = all_results[best_cfg_name]["per_target"][tname]["best_mode"]

        if best_mode == "sig_only":
            X_tr, X_te = X_sig_train, X_sig_test
        elif best_mode == "release_only":
            X_tr, X_te = X_rel_train, X_rel_test
        else:
            X_tr = np.hstack([X_sig_train, X_rel_train])
            X_te = np.hstack([X_sig_test, X_rel_test])

        preds = predict_test(X_tr, y, pids_train, X_te, pids_test,
                           bw_q=0.3, alpha=10.0, use_pls=True, pls_nc=5)
        test_preds_all[tname] = preds

    # Build submissions
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": test_preds_all["angle"],
        "scaled_depth": test_preds_all["depth"],
        "scaled_left_right": test_preds_all["left_right"],
    })

    save_submission(sub_df, f"Path signatures ({best_cfg_name})")

    # Blends with Sub3558 (current best)
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        for w in [0.03, 0.05, 0.08, 0.12]:
            blend = sub_df.copy()
            for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
                blend[tc] = w * sub_df[tc] + (1-w) * sub3558[tc]
            save_submission(blend, f"{int(w*100)}% signatures + {int((1-w)*100)}% Sub3558")
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
    with open(OUTPUT_DIR / f"path_signature_run_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
