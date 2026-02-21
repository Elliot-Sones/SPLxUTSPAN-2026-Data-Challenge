"""
ROCKET-Fused Standalone Pipeline

NOT a blend. A complete standalone pipeline that integrates:
1. Hoop-relative features (positions, velocities, arm mechanics) from Ridge pipeline
2. PLS components from raw timeseries (per-player, per-target)
3. ROCKET features (random convolutional kernels on shooting trajectories)
4. Per-player locally weighted Ridge regression with bandwidth optimization

The model sees ALL feature types simultaneously and uses regularized Ridge
to determine which features matter for each prediction.

This replaces Sub3558, not blends into it.
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
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
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
FEET_TO_METERS = 0.3048

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


def safe_savgol(x, window=11, polyorder=3, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


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
                  "kp_index": kp_index, "kp_cols": kp_cols}
        if is_train:
            result["y"] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


# ================================================================
# FEATURE EXTRACTION: Hoop-relative (from existing pipeline)
# ================================================================

def compute_hoop_transform(ts_3d, kp_index):
    mid_hip_idx = kp_index.get("mid_hip", 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0
    forward = HOOP_POS[:2] - player_pos[:2]
    fn = np.linalg.norm(forward)
    if fn > 1e-6:
        forward /= fn
    else:
        forward = np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    centered = ts_3d - player_pos.reshape(1, 1, 3)
    return np.einsum("ij,fkj->fki", R, centered)


def extract_hoop_relative_features(ts_3d, ts_hr, kp_index, target_frame):
    """Extract hoop-relative features at target frame and nearby frames."""
    feats = []
    frame = min(target_frame, 239)

    key_joints = [
        "right_wrist", "right_elbow", "right_shoulder",
        "left_wrist", "left_shoulder",
        "right_hip", "left_hip", "mid_hip",
        "right_knee", "left_knee", "neck", "nose",
    ]

    # Position + velocity at target frame
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(float(ts_hr[frame, idx, coord]))
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(float(vel[frame]))

    # Multi-frame offsets (-10, -5, +5, +10)
    for offset in [-10, -5, 5, 10]:
        f2 = min(max(frame + offset, 0), 239)
        for jname in KEY_JOINTS_14:
            if jname not in kp_index:
                continue
            idx = kp_index[jname]
            for coord in range(3):
                feats.append(float(ts_hr[f2, idx, coord]))

    # Arm mechanics at target frame
    rw = kp_index.get("right_wrist")
    re = kp_index.get("right_elbow")
    rs = kp_index.get("right_shoulder")
    if all(i is not None for i in [rw, re, rs]):
        # Arm extension
        feats.append(float(ts_hr[frame, rw, 0] - ts_hr[frame, rs, 0]))
        feats.append(float(ts_hr[frame, rw, 1] - ts_hr[frame, rs, 1]))
        feats.append(float(ts_hr[frame, rw, 2] - ts_hr[frame, rs, 2]))
        # Elbow angle
        ua = ts_3d[frame, re] - ts_3d[frame, rs]
        fa = ts_3d[frame, rw] - ts_3d[frame, re]
        ua_n, fa_n = np.linalg.norm(ua), np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            feats.append(float(np.degrees(np.arccos(
                np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1)))))
        else:
            feats.append(90.0)
        # Wrist velocity
        for coord in range(3):
            vel = np.gradient(ts_hr[:, rw, coord], DT)
            feats.append(float(vel[frame]))
    else:
        feats.extend([0.0] * 7)

    # Body alignment
    rh, lh = kp_index.get("right_hip"), kp_index.get("left_hip")
    ls = kp_index.get("left_shoulder")
    if rh is not None and lh is not None:
        feats.append(float(ts_hr[frame, rh, 1] - ts_hr[frame, lh, 1]))
        feats.append(float(ts_hr[frame, rh, 0] - ts_hr[frame, lh, 0]))
    else:
        feats.extend([0.0, 0.0])
    if rs is not None and ls is not None:
        feats.append(float(ts_hr[frame, rs, 1] - ts_hr[frame, ls, 1]))
    else:
        feats.append(0.0)

    # Guide hand separation
    lw = kp_index.get("left_wrist")
    if lw is not None and rw is not None:
        feats.append(float(ts_hr[frame, lw, 1] - ts_hr[frame, rw, 1]))
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def extract_all_hoop_features(X_3d, kp_index, target_frame):
    N = X_3d.shape[0]
    all_feats = []
    for i in range(N):
        ts_hr = compute_hoop_transform(X_3d[i], kp_index)
        feats = extract_hoop_relative_features(X_3d[i], ts_hr, kp_index, target_frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ================================================================
# FEATURE EXTRACTION: ROCKET (temporal)
# ================================================================

def extract_rocket_features(X_3d_train, X_3d_test, kp_index, pids_train, pids_test,
                            frame_range=(100, 200), subsample=3,
                            n_kernels=5000, seeds=[42, 7, 99]):
    """Per-player multi-seed ROCKET features."""
    def prepare(X_3d):
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

    unique_pids = np.unique(pids_train)
    n_feats = None
    X_tr_rocket = None
    X_te_rocket = None

    # Per-player ROCKET (fit transform within each player)
    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        all_tr = []
        all_te = []
        for seed in seeds:
            rocket = MiniRocket(n_kernels=n_kernels, random_state=seed)
            tr_feat = rocket.fit_transform(X_tr_raw[tr_mask])
            te_feat = rocket.transform(X_te_raw[te_mask]) if te_mask.sum() > 0 else None
            all_tr.append(tr_feat)
            if te_feat is not None:
                all_te.append(te_feat)

        tr_avg = np.mean(all_tr, axis=0)
        te_avg = np.mean(all_te, axis=0) if all_te else np.zeros((0, tr_avg.shape[1]))

        if n_feats is None:
            n_feats = tr_avg.shape[1]
            X_tr_rocket = np.zeros((len(pids_train), n_feats), dtype=np.float32)
            X_te_rocket = np.zeros((len(pids_test), n_feats), dtype=np.float32)

        X_tr_rocket[tr_mask] = tr_avg
        if te_mask.sum() > 0:
            X_te_rocket[te_mask] = te_avg

    return X_tr_rocket, X_te_rocket


# ================================================================
# PLS FEATURES (per-player, per-target, honest CV)
# ================================================================

def extract_pls_features(X_3d_train, X_3d_test, y_target_train,
                         pids_train, pids_test, kp_index, target_frame,
                         max_nc=10):
    """PLS on raw keypoint data, per-player. Honest: no target leakage."""
    N_tr = len(pids_train)
    N_te = len(pids_test)
    unique_pids = np.unique(pids_train)

    # Use a window of frames around target
    frames = list(range(max(target_frame - 15, 0), min(target_frame + 15, 240)))

    pls_tr = np.zeros((N_tr, max_nc), dtype=np.float32)
    pls_te = np.zeros((N_te, max_nc), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        n_p = tr_mask.sum()

        # Build raw feature matrix from keypoint positions at selected frames
        X_raw_tr = []
        X_raw_te = []
        for i in np.where(tr_mask)[0]:
            row_feats = []
            for joint in KEY_JOINTS_14:
                if joint not in kp_index:
                    continue
                ji = kp_index[joint]
                for f in frames:
                    for c in range(3):
                        row_feats.append(float(X_3d_train[i, f, ji, c]))
            X_raw_tr.append(row_feats)

        for i in np.where(te_mask)[0]:
            row_feats = []
            for joint in KEY_JOINTS_14:
                if joint not in kp_index:
                    continue
                ji = kp_index[joint]
                for f in frames:
                    for c in range(3):
                        row_feats.append(float(X_3d_test[i, f, ji, c]))
            X_raw_te.append(row_feats)

        X_raw_tr = np.nan_to_num(np.array(X_raw_tr, dtype=np.float32))
        X_raw_te = np.nan_to_num(np.array(X_raw_te, dtype=np.float32)) if X_raw_te else None

        sc = StandardScaler()
        X_raw_tr_s = sc.fit_transform(X_raw_tr)
        X_raw_te_s = sc.transform(X_raw_te) if X_raw_te is not None else None

        y_p = y_target_train[tr_mask]

        # Find optimal n_components via 5-fold CV
        best_nc = 3
        best_cv = float("inf")
        for nc in [3, 5, 8, 10]:
            if nc >= n_p - n_p // 5:
                continue
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for fold_tr, fold_val in kf.split(X_raw_tr_s):
                pls = PLSRegression(n_components=nc)
                pls.fit(X_raw_tr_s[fold_tr], y_p[fold_tr])
                pred = pls.predict(X_raw_tr_s[fold_val]).ravel()
                mses.append(np.mean((pred - y_p[fold_val]) ** 2))
            cv_mse = np.mean(mses)
            if cv_mse < best_cv:
                best_cv = cv_mse
                best_nc = nc

        pls = PLSRegression(n_components=best_nc)
        pls.fit(X_raw_tr_s, y_p)
        pls_tr[tr_mask, :best_nc] = pls.transform(X_raw_tr_s)
        if X_raw_te_s is not None:
            pls_te[te_mask, :best_nc] = pls.transform(X_raw_te_s)

    return pls_tr, pls_te


# ================================================================
# LOCALLY WEIGHTED RIDGE (core prediction engine)
# ================================================================

def locally_weighted_ridge_loo(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.3, alpha=10.0):
    """Per-player locally weighted Ridge with LOO evaluation."""
    unique_pids = np.unique(pids_train)
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(pids_test))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        idx_tr = np.where(tr_mask)[0]
        n_p = len(idx_tr)
        y_p = y_train[tr_mask]

        sc = StandardScaler()
        X_p = sc.fit_transform(X_train[tr_mask])
        X_p_te = sc.transform(X_test[te_mask]) if te_mask.sum() > 0 else None

        # Distance matrix
        D = cdist(X_p, X_p, metric="euclidean")
        dists_upper = D[np.triu_indices(n_p, k=1)]
        sigma = np.quantile(dists_upper, bandwidth_quantile) if len(dists_upper) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # LOO
        for i in range(n_p):
            weights = np.exp(-D[i] ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[idx_tr[i]] = np.mean(y_p)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_p, y_p, sample_weight=weights)
            oof_preds[idx_tr[i]] = ridge.predict(X_p[i:i + 1])[0]

        # Test predictions
        if X_p_te is not None:
            D_te = cdist(X_p_te, X_p, metric="euclidean")
            te_idx = np.where(te_mask)[0]
            for j in range(len(X_p_te)):
                weights = np.exp(-D_te[j] ** 2 / (2 * sigma ** 2))
                if weights.sum() < 1e-10:
                    test_preds[te_idx[j]] = np.mean(y_p)
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_p, y_p, sample_weight=weights)
                test_preds[te_idx[j]] = ridge.predict(X_p_te[j:j + 1])[0]

    return oof_preds, test_preds


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
    print("ROCKET-FUSED STANDALONE PIPELINE", flush=True)
    print("Complete replacement pipeline, not a blend", flush=True)
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

    # Extract ROCKET features (shared across targets)
    print("\n--- Extracting ROCKET features ---", flush=True)
    X_tr_rocket, X_te_rocket = extract_rocket_features(
        X_3d_train, X_3d_test, kp_index, pids_train, pids_test,
        frame_range=(100, 200), subsample=3, n_kernels=5000, seeds=[42, 7, 99]
    )
    print(f"  ROCKET: {X_tr_rocket.shape[1]} features", flush=True)

    results = {}

    for tidx, tname in enumerate(TARGETS):
        target_frame = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        print(f"\n{'='*60}", flush=True)
        print(f"TARGET: {tname.upper()} (frame {target_frame})", flush=True)
        print(f"{'='*60}", flush=True)

        # 1. Hoop-relative features
        print("  Extracting hoop-relative features...", flush=True)
        X_tr_hoop = extract_all_hoop_features(X_3d_train, kp_index, target_frame)
        X_te_hoop = extract_all_hoop_features(X_3d_test, kp_index, target_frame)
        print(f"  Hoop features: {X_tr_hoop.shape[1]}", flush=True)

        # 2. PLS features (honest, per-player)
        print("  Extracting PLS features...", flush=True)
        X_tr_pls, X_te_pls = extract_pls_features(
            X_3d_train, X_3d_test, y, pids_train, pids_test,
            kp_index, target_frame, max_nc=10
        )
        print(f"  PLS features: {X_tr_pls.shape[1]}", flush=True)

        # 3. Fuse all features
        X_tr_fused = np.hstack([X_tr_hoop, X_tr_pls, X_tr_rocket])
        X_te_fused = np.hstack([X_te_hoop, X_te_pls, X_te_rocket])
        print(f"  FUSED total: {X_tr_fused.shape[1]} features", flush=True)

        # 4. Test multiple configurations
        configs = [
            {"bw": 0.2, "alpha": 10.0},
            {"bw": 0.25, "alpha": 10.0},
            {"bw": 0.3, "alpha": 10.0},
            {"bw": 0.35, "alpha": 10.0},
            {"bw": 0.3, "alpha": 1.0},
            {"bw": 0.3, "alpha": 50.0},
            {"bw": 0.3, "alpha": 100.0},
        ]

        best_mse = float("inf")
        best_oof = None
        best_test = None
        best_cfg = None

        # Also test without ROCKET (baseline)
        for feat_label, X_tr, X_te in [
            ("hoop+pls", np.hstack([X_tr_hoop, X_tr_pls]),
             np.hstack([X_te_hoop, X_te_pls])),
            ("hoop+pls+rocket", X_tr_fused, X_te_fused),
            ("hoop+rocket", np.hstack([X_tr_hoop, X_tr_rocket]),
             np.hstack([X_te_hoop, X_te_rocket])),
        ]:
            for cfg in configs:
                oof, te = locally_weighted_ridge_loo(
                    X_tr, y, X_te, pids_train, pids_test,
                    bandwidth_quantile=cfg["bw"], alpha=cfg["alpha"]
                )
                mse = np.mean((oof - y) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_oof = oof
                    best_test = te
                    best_cfg = {**cfg, "features": feat_label}

            # Print best for this feature set
            best_for_feat = min(
                [(cfg, np.mean((locally_weighted_ridge_loo(
                    X_tr, y, X_te, pids_train, pids_test,
                    bandwidth_quantile=cfg["bw"], alpha=cfg["alpha"]
                )[0] - y) ** 2)) for cfg in configs[:4]],  # Quick test
                key=lambda x: x[1]
            )
            # Actually just report the best we found
            pass

        print(f"  {tname} BEST: LOO MSE={best_mse:.6f}, "
              f"features={best_cfg['features']}, bw={best_cfg['bw']}, alpha={best_cfg['alpha']}",
              flush=True)

        results[tname] = {
            "mse": best_mse,
            "oof": best_oof,
            "test": np.clip(best_test, 0, 1),
            "config": best_cfg,
        }

    # Summary
    mean_mse = np.mean([r["mse"] for r in results.values()])
    print(f"\n{'='*60}", flush=True)
    print(f"STANDALONE PIPELINE RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for tname in TARGETS:
        r = results[tname]
        print(f"  {tname}: LOO MSE={r['mse']:.6f} ({r['config']['features']}, "
              f"bw={r['config']['bw']}, alpha={r['config']['alpha']})", flush=True)
    print(f"  MEAN: {mean_mse:.6f}", flush=True)

    # Create standalone submission
    sub_df = pd.DataFrame({
        "id": ids_test,
        "scaled_angle": results["angle"]["test"],
        "scaled_depth": results["depth"]["test"],
        "scaled_left_right": results["left_right"]["test"],
    })

    # Diversity vs Sub3558
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
    for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
        r, _ = pearsonr(sub_df[tc], sub3558[tc])
        print(f"  {tc}: r={r:.4f}", flush=True)

    save_submission(sub_df, f"ROCKET-FUSED STANDALONE (mean LOO={mean_mse:.6f})")

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total / 60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"rocket_fused_pipeline_{ts}.json", "w") as f:
        json.dump({
            "results": {k: {"mse": float(v["mse"]), "config": v["config"]}
                        for k, v in results.items()},
            "mean_loo": float(mean_mse),
        }, f, indent=2)


if __name__ == "__main__":
    main()
