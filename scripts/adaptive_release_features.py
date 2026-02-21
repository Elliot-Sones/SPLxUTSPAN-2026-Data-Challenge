"""
Adaptive Release Features - Two Creative Approaches

Approach 1: Velocity-Weighted Adaptive Frame Features
Instead of extracting features at a FIXED frame (e.g., 153 for angle), compute a
velocity-weighted average of features across frames near release. Each shot gets
its OWN effective release frame based on the actual motion dynamics of that shot.

The key insight: different shots have different release timing. The right hand
velocity profile tells us WHEN the release happens for each shot. By weighting
frames by velocity, we automatically focus on the release point.

Approach 2: Shot Rhythm/Timing Features
Capture the relative timing of kinematic events in the shooting motion:
- Time from peak hip velocity to peak wrist velocity
- Time from peak elbow extension rate to peak wrist velocity
- Duration of the "loading" vs "release" phases
- Relative timing ratios

These features capture the RHYTHM of the shot, which is fundamentally different
from positional features at any single frame.

Both approaches are integrated into the existing per-example Ridge pipeline.
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

# Key joints for feature extraction
KEY_JOINTS_FULL = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_first_finger_mcp", "right_second_finger_mcp",
    "left_shoulder", "left_wrist",
    "right_hip", "left_hip",
    "nose", "right_knee", "left_knee",
    "right_ankle", "left_ankle",
]

# Joints for rhythm detection
RHYTHM_JOINTS = {
    "right_wrist": None,
    "right_elbow": None,
    "right_shoulder": None,
    "right_hip": None,
    "right_knee": None,
    "nose": None,  # proxy for center of mass
}


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


def compute_velocity_profiles(X_3d, kp_index):
    """Compute velocity magnitude profiles for key joints across all frames.

    Returns: dict of joint_name -> (N, 240) velocity magnitude arrays
    """
    N = X_3d.shape[0]
    dt = 1.0 / 60.0
    profiles = {}

    for joint in RHYTHM_JOINTS:
        if joint not in kp_index:
            continue
        ji = kp_index[joint]
        # Central differences for velocity
        vel = np.zeros((N, 240, 3), dtype=np.float32)
        vel[:, 1:-1, :] = (X_3d[:, 2:, ji, :] - X_3d[:, :-2, ji, :]) / (2 * dt)
        vel[:, 0, :] = vel[:, 1, :]
        vel[:, -1, :] = vel[:, -2, :]

        # Velocity magnitude
        vmag = np.linalg.norm(vel, axis=2)  # (N, 240)
        profiles[joint] = vmag

    return profiles


def extract_adaptive_frame_features(X_3d, kp_index, target_frame, window=15):
    """Extract features using velocity-weighted frame averaging.

    Instead of features at a single frame, compute a weighted average of
    features from (target_frame - window) to (target_frame + window),
    where weights are proportional to right wrist velocity magnitude.
    This gives each shot its own effective release frame.
    """
    N = X_3d.shape[0]
    dt = 1.0 / 60.0

    # Compute right wrist velocity profile for weighting
    rw_idx = kp_index["right_wrist"]
    f_start = max(0, target_frame - window)
    f_end = min(239, target_frame + window)

    # Velocity magnitudes in the window
    vel_weights = np.zeros((N, f_end - f_start + 1), dtype=np.float32)
    for fi, f in enumerate(range(f_start, f_end + 1)):
        if 0 < f < 239:
            v = (X_3d[:, f+1, rw_idx, :] - X_3d[:, f-1, rw_idx, :]) / (2*dt)
        elif f > 0:
            v = (X_3d[:, f, rw_idx, :] - X_3d[:, f-1, rw_idx, :]) / dt
        else:
            v = np.zeros((N, 3))
        vel_weights[:, fi] = np.linalg.norm(v, axis=1)

    # Normalize weights (softmax-like with temperature)
    # Higher temperature = more uniform, lower = more peaked
    temperature = 0.5  # favor frames with high velocity
    vel_weights = vel_weights ** (1.0 / temperature)
    weight_sum = vel_weights.sum(axis=1, keepdims=True) + 1e-8
    vel_weights = vel_weights / weight_sum  # (N, n_frames)

    # Hip center for relative coords
    rh = kp_index["right_hip"]
    lh = kp_index["left_hip"]

    # Compute features at each frame, then weight-average
    n_joints = len(KEY_JOINTS_FULL)
    # Features per frame: 6 per joint (3 hip-rel + 3 hoop-rel) = 84
    n_feat_per_frame = n_joints * 6

    weighted_features = np.zeros((N, n_feat_per_frame), dtype=np.float64)

    for fi, f in enumerate(range(f_start, f_end + 1)):
        hip = (X_3d[:, f, rh, :] + X_3d[:, f, lh, :]) / 2.0
        frame_feats = []

        for j in KEY_JOINTS_FULL:
            if j not in kp_index:
                frame_feats.extend([np.zeros(N)] * 6)
                continue
            ji = kp_index[j]
            pos = X_3d[:, f, ji, :]
            rel = pos - hip
            hoop_rel = pos - HOOP_POS[np.newaxis, :]
            for ci in range(3):
                frame_feats.append(rel[:, ci])
                frame_feats.append(hoop_rel[:, ci])

        frame_feats = np.column_stack(frame_feats)  # (N, n_feat_per_frame)
        weighted_features += frame_feats * vel_weights[:, fi:fi+1]

    # Also add velocity features at the velocity-weighted center frame
    # The "effective frame" is the velocity-weighted average frame index
    eff_frame = np.sum(
        vel_weights * np.arange(f_start, f_end + 1)[np.newaxis, :],
        axis=1
    )  # (N,)

    # Interpolate to get features at the effective frame
    # For simplicity, round to nearest integer
    eff_frame_int = np.clip(np.round(eff_frame).astype(int), 1, 238)

    # Velocity features at effective frame
    vel_feats = []
    for j in KEY_JOINTS_FULL[:7]:  # top 7 joints
        if j not in kp_index:
            vel_feats.extend([np.zeros(N)] * 3)
            continue
        ji = kp_index[j]
        v = np.zeros((N, 3))
        for i in range(N):
            ef = eff_frame_int[i]
            v[i] = (X_3d[i, ef+1, ji, :] - X_3d[i, ef-1, ji, :]) / (2*dt)
        for ci in range(3):
            vel_feats.append(v[:, ci])

    vel_feats = np.column_stack(vel_feats)  # (N, 21)

    # Add the effective frame offset as a feature
    frame_offset = (eff_frame - target_frame).reshape(-1, 1)

    all_features = np.hstack([weighted_features, vel_feats, frame_offset])
    return np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_rhythm_features(X_3d, kp_index, vel_profiles):
    """Extract shot rhythm/timing features.

    Captures the relative timing of kinematic events:
    - Peak velocity frame for each joint
    - Time differences between joint peaks
    - Duration of loading/release phases
    - Velocity ratios at different phases
    """
    N = X_3d.shape[0]
    dt = 1.0 / 60.0
    features = []

    # Focus on frames 100-200 (the action happens here)
    f_start, f_end = 100, 200

    # 1. Peak velocity frame for each joint
    peak_frames = {}
    for joint, vmag in vel_profiles.items():
        segment = vmag[:, f_start:f_end]
        peak_local = np.argmax(segment, axis=1)
        peak_global = peak_local + f_start
        peak_frames[joint] = peak_global
        # Peak frame as feature (normalized to [0,1] range)
        features.append((peak_global - f_start) / (f_end - f_start))
        # Peak velocity magnitude
        features.append(np.array([vmag[i, peak_global[i]] for i in range(N)]))

    # 2. Time differences between joint peaks (rhythm features)
    joint_list = list(vel_profiles.keys())
    for i in range(len(joint_list)):
        for j in range(i+1, len(joint_list)):
            j1, j2 = joint_list[i], joint_list[j]
            if j1 in peak_frames and j2 in peak_frames:
                tdiff = (peak_frames[j1] - peak_frames[j2]).astype(np.float32)
                features.append(tdiff * dt)  # in seconds

    # 3. Velocity ratios: peak wrist velocity / peak hip velocity
    if "right_wrist" in vel_profiles and "right_hip" in vel_profiles:
        wrist_peak_v = np.array([vel_profiles["right_wrist"][i, peak_frames["right_wrist"][i]]
                                  for i in range(N)])
        hip_peak_v = np.array([vel_profiles["right_hip"][i, peak_frames["right_hip"][i]]
                                for i in range(N)])
        features.append(wrist_peak_v / (hip_peak_v + 1e-8))

    # 4. Velocity at the standard release frames
    for tname, tf in TARGET_FRAMES.items():
        for joint in ["right_wrist", "right_elbow", "right_shoulder"]:
            if joint in vel_profiles:
                features.append(vel_profiles[joint][:, tf])

    # 5. Acceleration profile features: when does peak acceleration happen?
    for joint in ["right_wrist", "right_elbow"]:
        if joint not in vel_profiles:
            continue
        vmag = vel_profiles[joint][:, f_start:f_end]
        # Acceleration = derivative of velocity
        acc = np.diff(vmag, axis=1) * 60.0  # per second
        peak_acc_frame = np.argmax(acc, axis=1) + f_start
        features.append((peak_acc_frame - f_start) / (f_end - f_start))
        # Peak acceleration magnitude
        features.append(np.array([acc[i, peak_acc_frame[i] - f_start] for i in range(N)]))
        # Frame difference: peak acceleration to peak velocity
        if joint in peak_frames:
            features.append((peak_acc_frame - peak_frames[joint]).astype(np.float32) * dt)

    # 6. "Snap" features: how quickly does velocity change around release?
    for joint in ["right_wrist"]:
        if joint not in vel_profiles:
            continue
        vmag = vel_profiles[joint]
        for tf in [150, 153, 155, 160]:
            if tf >= 5 and tf <= 234:
                # Velocity change rate around this frame
                v_before = vmag[:, tf-5:tf].mean(axis=1)
                v_after = vmag[:, tf:tf+5].mean(axis=1)
                features.append(v_after - v_before)
                features.append(v_after / (v_before + 1e-8))

    # 7. Phase durations (using wrist velocity)
    if "right_wrist" in vel_profiles:
        vmag = vel_profiles["right_wrist"]
        # Find when velocity first exceeds 50% of peak (loading phase end)
        for i in range(N):
            peak_v = vmag[i, f_start:f_end].max()
            threshold = 0.5 * peak_v
            above = np.where(vmag[i, f_start:f_end] > threshold)[0]
            if len(above) > 0:
                load_end = above[0]
                release_start = above[-1]
            else:
                load_end = 50
                release_start = 50
            if i == 0:
                load_ends = [load_end]
                release_starts = [release_start]
            else:
                load_ends.append(load_end)
                release_starts.append(release_start)

        load_ends = np.array(load_ends, dtype=np.float32)
        release_starts = np.array(release_starts, dtype=np.float32)
        features.append(load_ends * dt)  # loading phase duration
        features.append(release_starts * dt)  # release phase end
        features.append((release_starts - load_ends) * dt)  # active phase duration

    X = np.column_stack(features)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def extract_standard_features(X_3d, kp_index, frame_idx):
    """Standard features at a fixed frame (baseline comparison)."""
    N = X_3d.shape[0]
    dt = 1.0 / 60.0
    features = []

    rh = kp_index["right_hip"]
    lh = kp_index["left_hip"]
    hip = (X_3d[:, frame_idx, rh, :] + X_3d[:, frame_idx, lh, :]) / 2.0

    for j in KEY_JOINTS_FULL:
        if j not in kp_index:
            features.extend([np.zeros(N)] * 6)
            continue
        ji = kp_index[j]
        pos = X_3d[:, frame_idx, ji, :]
        rel = pos - hip
        hoop_rel = pos - HOOP_POS[np.newaxis, :]
        for ci in range(3):
            features.append(rel[:, ci])
            features.append(hoop_rel[:, ci])

    # Velocity
    for j in KEY_JOINTS_FULL[:7]:
        if j not in kp_index:
            features.extend([np.zeros(N)] * 3)
            continue
        ji = kp_index[j]
        if 0 < frame_idx < 239:
            vel = (X_3d[:, frame_idx+1, ji, :] - X_3d[:, frame_idx-1, ji, :]) / (2*dt)
        else:
            vel = np.zeros((N, 3))
        for ci in range(3):
            features.append(vel[:, ci])

    X = np.column_stack(features)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def per_example_ridge_loo(X_train, y_train, pids_train, bw_q=0.3, alpha=10.0, n_pls=5):
    """Per-example locally weighted Ridge with PLS, honest LOO."""
    N = len(y_train)
    predictions = np.zeros(N)
    unique_pids = np.unique(pids_train)

    for pid in unique_pids:
        mask = pids_train == pid
        idx = np.where(mask)[0]
        n_p = len(idx)
        X_p = X_train[idx]
        y_p = y_train[idx]

        # Standardize within player
        scaler = StandardScaler()
        X_ps = scaler.fit_transform(X_p)

        # Distance matrix
        dists = np.zeros((n_p, n_p))
        for i in range(n_p):
            for j in range(n_p):
                dists[i, j] = np.linalg.norm(X_ps[i] - X_ps[j])

        # Bandwidth
        bw = np.quantile(dists[dists > 0], bw_q) + 1e-8

        # LOO
        for i_local in range(n_p):
            loo_mask = np.ones(n_p, dtype=bool)
            loo_mask[i_local] = False

            X_loo = X_ps[loo_mask]
            y_loo = y_p[loo_mask]
            x_test = X_ps[i_local:i_local+1]

            # PLS (honest: fit only on LOO data)
            n_pls_actual = min(n_pls, X_loo.shape[1], len(y_loo) - 1)
            if n_pls_actual > 0:
                pls = PLSRegression(n_components=n_pls_actual)
                pls.fit(X_loo, y_loo)
                X_loo_pls = np.hstack([X_loo, pls.transform(X_loo)])
                x_test_pls = np.hstack([x_test, pls.transform(x_test)])
            else:
                X_loo_pls = X_loo
                x_test_pls = x_test

            # Kernel weights
            d = np.linalg.norm(X_loo_pls - x_test_pls, axis=1)
            bw_pls = np.quantile(d[d > 0], bw_q) + 1e-8 if len(d[d > 0]) > 0 else 1.0
            w = np.exp(-0.5 * (d / bw_pls) ** 2)

            # Weighted Ridge
            W = np.diag(np.sqrt(w + 1e-10))
            X_w = W @ X_loo_pls
            y_w = W @ y_loo

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_w, y_w)
            predictions[idx[i_local]] = ridge.predict(x_test_pls)[0]

    return predictions


def per_example_ridge_predict(X_train, y_train, pids_train,
                               X_test, pids_test,
                               bw_q=0.3, alpha=10.0, n_pls=5):
    """Per-example locally weighted Ridge predictions for test data."""
    N_test = len(X_test)
    predictions = np.zeros(N_test)
    unique_pids = np.unique(pids_train)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        X_p = X_train[tr_mask]
        y_p = y_train[tr_mask]
        X_t = X_test[te_mask]
        n_p = len(X_p)
        n_t = te_mask.sum()

        if n_t == 0:
            continue

        scaler = StandardScaler()
        X_ps = scaler.fit_transform(X_p)
        X_ts = scaler.transform(X_t)

        # PLS on full training player data
        n_pls_actual = min(n_pls, X_ps.shape[1], n_p - 1)
        if n_pls_actual > 0:
            pls = PLSRegression(n_components=n_pls_actual)
            pls.fit(X_ps, y_p)
            X_ps_pls = np.hstack([X_ps, pls.transform(X_ps)])
            X_ts_pls = np.hstack([X_ts, pls.transform(X_ts)])
        else:
            X_ps_pls = X_ps
            X_ts_pls = X_ts

        # Bandwidth from training distances
        dists_tr = np.zeros((n_p, n_p))
        for i in range(n_p):
            for j in range(n_p):
                dists_tr[i, j] = np.linalg.norm(X_ps_pls[i] - X_ps_pls[j])
        bw = np.quantile(dists_tr[dists_tr > 0], bw_q) + 1e-8

        for i_test in range(n_t):
            d = np.linalg.norm(X_ps_pls - X_ts_pls[i_test], axis=1)
            w = np.exp(-0.5 * (d / bw) ** 2)

            W = np.diag(np.sqrt(w + 1e-10))
            X_w = W @ X_ps_pls
            y_w = W @ y_p

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_w, y_w)
            predictions[np.where(te_mask)[0][i_test]] = ridge.predict(X_ts_pls[i_test:i_test+1])[0]

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
    print("Adaptive Release Features - Creative Approaches", flush=True)
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

    # Compute velocity profiles (needed for both approaches)
    print("\nComputing velocity profiles...", flush=True)
    vel_profiles_train = compute_velocity_profiles(X_3d_train, kp_index)
    vel_profiles_test = compute_velocity_profiles(X_3d_test, kp_index)

    results = {}
    test_preds = {}

    # ============================================================
    # Approach 1: Adaptive Frame Features
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("APPROACH 1: Velocity-Weighted Adaptive Frame Features", flush=True)
    print(f"{'='*60}", flush=True)

    for tidx, tname in enumerate(TARGETS):
        tf = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        print(f"\n  --- {tname} (center frame {tf}) ---", flush=True)

        # Adaptive features
        X_tr_adapt = extract_adaptive_frame_features(X_3d_train, kp_index, tf, window=15)
        print(f"  Adaptive features: {X_tr_adapt.shape[1]}", flush=True)

        # Standard features for comparison
        X_tr_std = extract_standard_features(X_3d_train, kp_index, tf)
        print(f"  Standard features: {X_tr_std.shape[1]}", flush=True)

        # LOO for adaptive
        t0 = time.time()
        loo_adapt = per_example_ridge_loo(X_tr_adapt, y, pids_train, n_pls=5)
        mse_adapt = np.mean((loo_adapt - y) ** 2)
        print(f"  Adaptive LOO MSE: {mse_adapt:.6f} ({time.time()-t0:.1f}s)", flush=True)

        # LOO for standard
        t0 = time.time()
        loo_std = per_example_ridge_loo(X_tr_std, y, pids_train, n_pls=5)
        mse_std = np.mean((loo_std - y) ** 2)
        print(f"  Standard LOO MSE: {mse_std:.6f} ({time.time()-t0:.1f}s)", flush=True)

        improvement = (mse_std - mse_adapt) / mse_std * 100
        print(f"  Change: {improvement:+.1f}%", flush=True)

        results[f"adapt_{tname}"] = {"mse": float(mse_adapt), "improvement": float(improvement)}
        results[f"std_{tname}"] = {"mse": float(mse_std)}

    # ============================================================
    # Approach 2: Rhythm Features
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("APPROACH 2: Shot Rhythm/Timing Features", flush=True)
    print(f"{'='*60}", flush=True)

    X_rhythm_train = extract_rhythm_features(X_3d_train, kp_index, vel_profiles_train)
    X_rhythm_test = extract_rhythm_features(X_3d_test, kp_index, vel_profiles_test)
    print(f"Rhythm features: {X_rhythm_train.shape[1]}", flush=True)

    for tidx, tname in enumerate(TARGETS):
        tf = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        print(f"\n  --- {tname} ---", flush=True)

        # Rhythm only
        t0 = time.time()
        loo_rhythm = per_example_ridge_loo(X_rhythm_train, y, pids_train, n_pls=5)
        mse_rhythm = np.mean((loo_rhythm - y) ** 2)
        print(f"  Rhythm-only LOO MSE: {mse_rhythm:.6f} ({time.time()-t0:.1f}s)", flush=True)

        # Standard + Rhythm combined
        X_tr_combined = np.hstack([
            extract_standard_features(X_3d_train, kp_index, tf),
            X_rhythm_train
        ])
        t0 = time.time()
        loo_combined = per_example_ridge_loo(X_tr_combined, y, pids_train, n_pls=5)
        mse_combined = np.mean((loo_combined - y) ** 2)
        print(f"  Combined LOO MSE: {mse_combined:.6f} ({time.time()-t0:.1f}s)", flush=True)

        mse_std = results[f"std_{tname}"]["mse"]
        improvement_rhythm = (mse_std - mse_rhythm) / mse_std * 100
        improvement_combined = (mse_std - mse_combined) / mse_std * 100
        print(f"  Rhythm vs std: {improvement_rhythm:+.1f}%", flush=True)
        print(f"  Combined vs std: {improvement_combined:+.1f}%", flush=True)

        results[f"rhythm_{tname}"] = {"mse": float(mse_rhythm)}
        results[f"combined_{tname}"] = {"mse": float(mse_combined),
                                         "improvement": float(improvement_combined)}

    # ============================================================
    # Approach 3: Adaptive + Rhythm combined
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("APPROACH 3: Adaptive + Rhythm Combined", flush=True)
    print(f"{'='*60}", flush=True)

    best_preds_train = {}
    best_preds_test = {}

    for tidx, tname in enumerate(TARGETS):
        tf = TARGET_FRAMES[tname]
        y = y_scaled[:, tidx]

        X_tr_full = np.hstack([
            extract_adaptive_frame_features(X_3d_train, kp_index, tf, window=15),
            X_rhythm_train
        ])
        X_te_full = np.hstack([
            extract_adaptive_frame_features(X_3d_test, kp_index, tf, window=15),
            X_rhythm_test
        ])

        print(f"\n  --- {tname} ({X_tr_full.shape[1]} features) ---", flush=True)

        t0 = time.time()
        loo_full = per_example_ridge_loo(X_tr_full, y, pids_train, n_pls=5)
        mse_full = np.mean((loo_full - y) ** 2)
        print(f"  Full LOO MSE: {mse_full:.6f} ({time.time()-t0:.1f}s)", flush=True)

        mse_std = results[f"std_{tname}"]["mse"]
        improvement = (mse_std - mse_full) / mse_std * 100
        print(f"  vs standard: {improvement:+.1f}%", flush=True)

        results[f"full_{tname}"] = {"mse": float(mse_full), "improvement": float(improvement)}

        # Generate test predictions
        t0 = time.time()
        test_pred = per_example_ridge_predict(
            X_tr_full, y, pids_train, X_te_full, pids_test, n_pls=5
        )
        test_preds[tname] = np.clip(test_pred, 0, 1)
        print(f"  Test predictions in {time.time()-t0:.1f}s", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    for approach in ["std", "adapt", "rhythm", "combined", "full"]:
        mses = []
        for tname in TARGETS:
            key = f"{approach}_{tname}"
            if key in results:
                mses.append(results[key]["mse"])
        if mses:
            print(f"  {approach}: mean MSE = {np.mean(mses):.6f}", flush=True)

    # Diversity analysis
    print(f"\n--- Diversity vs Sub3558 ---", flush=True)
    try:
        sub3558 = pd.read_csv(SUBMISSION_DIR / "submission_3558.csv")
        sub_df = pd.DataFrame({
            "id": ids_test,
            "scaled_angle": test_preds["angle"],
            "scaled_depth": test_preds["depth"],
            "scaled_left_right": test_preds["left_right"],
        })
        for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
            r, _ = pearsonr(sub_df[tc], sub3558[tc])
            print(f"  {tc}: r={r:.4f}", flush=True)

        # Save standalone
        save_submission(sub_df, "Adaptive+Rhythm standalone")

        # Blends with Sub3558
        for w in [0.02, 0.05, 0.08]:
            blend = sub_df.copy()
            for tc in ["scaled_angle", "scaled_depth", "scaled_left_right"]:
                blend[tc] = np.clip(w * sub_df[tc] + (1-w) * sub3558[tc], 0, 1)
            save_submission(blend, f"{int(w*100)}% Adaptive+Rhythm + {int((1-w)*100)}% Sub3558")
    except Exception as e:
        print(f"  Blend error: {e}", flush=True)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"adaptive_release_run_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
