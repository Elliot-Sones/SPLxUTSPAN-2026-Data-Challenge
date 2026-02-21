"""
Measure whether multi-frame features add information beyond single-frame positions.

Method: Partial correlation analysis.
1. Get residuals from our single-frame baseline (per-player Ridge LOO)
2. Compute correlation between candidate features and those residuals
3. If a feature correlates with residuals, it captures something the baseline misses

This is the correct test: no circular evaluation, no feature selection bias.
Features are defined by physics (not by optimizing on this data).
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

HOOP_FEET = np.array([5.25, -25.0, 10.0])
FEET_TO_M = 0.3048
DT = 1.0 / 60.0
N_FRAMES = 240
TARGETS = ["angle", "depth", "left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


def safe_savgol(arr, window=11, poly=3, deriv=0, delta=1.0):
    arr = np.array(arr, dtype=np.float64)
    bad = np.isnan(arr) | np.isinf(arr)
    if np.all(bad):
        return np.zeros_like(arr)
    if np.any(bad):
        good = ~bad
        arr[bad] = np.interp(np.where(bad)[0], np.where(good)[0], arr[good])
    if len(arr) < window:
        return arr
    return savgol_filter(arr, window, poly, deriv=deriv, delta=delta)


def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    return np.nan_to_num(np.array(json.loads(s), dtype=np.float64), nan=0.0)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]
    kp_names = [c[:-2] for c in kp_cols if c.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}
    n_kp = len(kp_names)
    n = len(df)
    X_3d = np.zeros((n, N_FRAMES, n_kp, 3), dtype=np.float32)
    for idx, row in df.iterrows():
        for col_i, col in enumerate(kp_cols):
            X_3d[idx, :, col_i // 3, col_i % 3] = parse_array_string(row[col])
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx+1}/{n}...")
    return X_3d, kp_names, kp_index, df


def get_joint(ts_3d, kp_index, name, smooth=True, window=11):
    idx = kp_index.get(name)
    if idx is None:
        return np.zeros((N_FRAMES, 3))
    traj = ts_3d[:, idx, :].copy()
    for ax in range(3):
        bad = np.isnan(traj[:, ax]) | np.isinf(traj[:, ax])
        if np.any(bad) and not np.all(bad):
            good = ~bad
            traj[bad, ax] = np.interp(np.where(bad)[0], np.where(good)[0], traj[good, ax])
    if smooth:
        for ax in range(3):
            traj[:, ax] = safe_savgol(traj[:, ax], window, 3)
    return traj


def compute_velocity(positions_feet, window=7):
    vel = np.zeros((N_FRAMES, 3))
    for ax in range(3):
        pos_m = positions_feet[:, ax] * FEET_TO_M
        vel[:, ax] = safe_savgol(pos_m, window, 3, deriv=1, delta=DT)
    return vel


def decompose_velocity(vel_3d, player_pos):
    to_hoop = HOOP_FEET[:2] - player_pos[:2]
    d = np.linalg.norm(to_hoop)
    if d > 1e-6:
        fwd = to_hoop / d
    else:
        fwd = np.array([0, -1.0])
    lat = np.array([-fwd[1], fwd[0]])
    v_fwd = vel_3d[0] * fwd[0] + vel_3d[1] * fwd[1]
    v_lat = vel_3d[0] * lat[0] + vel_3d[1] * lat[1]
    v_up = vel_3d[2]
    return v_fwd, v_lat, v_up


def extract_baseline_features(ts_3d, kp_index, frame):
    """Single-frame hoop-relative position features (what our main pipeline uses)."""
    feats = []
    for name in sorted(kp_index.keys()):
        traj = get_joint(ts_3d, kp_index, name, smooth=True)
        pos_rel = traj[frame] - HOOP_FEET
        feats.extend([pos_rel[0], pos_rel[1], pos_rel[2]])
    return np.array(feats, dtype=np.float32)


def baseline_loo_residuals(X_3d, kp_index, y, pids, frame, bw_q=0.3, alpha=10.0):
    """
    Run our baseline single-frame Ridge LOO and return residuals.
    Uses honest PLS (refit per fold).
    """
    n = len(y)

    # Extract features at single frame
    X = np.zeros((n, len(kp_index) * 3), dtype=np.float32)
    for i in range(n):
        X[i] = extract_baseline_features(X_3d[i], kp_index, frame)

    preds = np.full(n, np.nan)

    for pid in sorted(np.unique(pids)):
        mask = pids == pid
        idx = np.where(mask)[0]
        X_p, y_p = X[mask], y[mask]
        n_p = len(y_p)

        if n_p < 5:
            preds[idx] = np.mean(y_p)
            continue

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        n_pls = min(15, n_p - 2, X_s.shape[1] - 1)
        if n_pls < 1:
            n_pls = 1

        for i in range(n_p):
            # Honest PLS
            X_loo = np.delete(X_s, i, axis=0)
            y_loo = np.delete(y_p, i)
            pls = PLSRegression(n_components=n_pls, scale=False)
            pls.fit(X_loo, y_loo)
            X_all_pls = pls.transform(X_s)
            X_comb = np.hstack([X_s, X_all_pls])

            D = pairwise_distances(X_comb)
            bw = np.quantile(D[np.triu_indices(n_p, k=1)], bw_q)
            if bw < 1e-10:
                bw = 1.0
            w = np.exp(-0.5 * (D[i] / bw) ** 2)
            w[i] = 0
            if w.sum() < 1e-10:
                preds[idx[i]] = np.mean(np.delete(y_p, i))
                continue
            sqrt_w = np.sqrt(w)
            X_w = X_comb * sqrt_w[:, None]
            y_w = y_p * sqrt_w
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(np.delete(X_w, i, 0), np.delete(y_w, i))
            preds[idx[i]] = ridge.predict(X_comb[i:i + 1])[0]

    residuals = y - preds
    loo_mse = np.mean(residuals ** 2)
    return residuals, preds, loo_mse


def main():
    print("=" * 70)
    print("MULTI-FRAME SIGNAL MEASUREMENT")
    print("Does adding features from other frames reduce unexplained error?")
    print("=" * 70)

    print("\nLoading data...")
    X_3d, kp_names, kp_index, df = load_data(DATA_DIR / "train.csv")
    pids = df["participant_id"].values
    n = len(df)

    y_scaled = {}
    for t in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{t}.pkl"
        if scaler_path.exists():
            s = joblib.load(scaler_path)
            y_scaled[t] = s.transform(df[t].values.reshape(-1, 1)).ravel()
        else:
            v = df[t].values
            y_scaled[t] = (v - v.min()) / (v.max() - v.min() + 1e-10)

    # ================================================================
    # STEP 1: Get baseline residuals for each target
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 1: Baseline single-frame Ridge LOO residuals")
    print("=" * 50)

    residuals = {}
    for tname in TARGETS:
        frame = TARGET_FRAMES[tname]
        y = y_scaled[tname]
        print(f"\n  {tname} (frame {frame})...")
        res, preds, mse = baseline_loo_residuals(X_3d, kp_index, y, pids, frame)
        residuals[tname] = res
        print(f"    LOO MSE = {mse:.6f}")
        print(f"    Residual mean = {np.mean(res):.6f}, std = {np.std(res):.6f}")

    # ================================================================
    # STEP 2: Precompute candidate features from OTHER frames
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 2: Precompute candidate multi-frame features")
    print("=" * 50)

    # Player positions for velocity decomposition
    player_pos = np.zeros((n, 3))
    for i in range(n):
        hip = get_joint(X_3d[i], kp_index, 'mid_hip', smooth=True)
        player_pos[i] = hip[20]
        player_pos[i, 2] = 0

    # Define candidate features: (name, joint, frame, component)
    # These come from physics, NOT from optimizing on this data
    joints_to_track = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'left_shoulder', 'neck', 'mid_hip',
        'right_knee', 'right_ankle',
        'right_second_finger_distal', 'right_third_finger_distal',
    ]

    # Frames spanning the full shot: windup, acceleration, release, follow-through
    frames_to_check = [60, 80, 100, 120, 130, 140, 145, 150, 153, 155, 160, 170, 180, 200]

    components = ['speed', 'v_fwd', 'v_lat', 'v_up']

    print(f"  {len(joints_to_track)} joints x {len(frames_to_check)} frames x {len(components)} components")
    print(f"  = {len(joints_to_track) * len(frames_to_check) * len(components)} candidate features")

    # Precompute all velocities
    print("  Computing velocities...")
    all_vel = {}
    for jname in joints_to_track:
        vel_arr = np.zeros((n, N_FRAMES, 3))
        for i in range(n):
            traj = get_joint(X_3d[i], kp_index, jname, smooth=True, window=11)
            vel_arr[i] = compute_velocity(traj, window=7)
        all_vel[jname] = vel_arr

    # Also precompute positions at different frames (for position-at-other-frame features)
    all_pos = {}
    for jname in joints_to_track:
        pos_arr = np.zeros((n, N_FRAMES, 3))
        for i in range(n):
            pos_arr[i] = get_joint(X_3d[i], kp_index, jname, smooth=True, window=11)
        all_pos[jname] = pos_arr

    # ================================================================
    # STEP 3: Correlate each candidate feature with residuals
    # ================================================================
    print("\n" + "=" * 50)
    print("STEP 3: Partial correlations (feature vs baseline residuals)")
    print("A feature that correlates with residuals captures NEW information")
    print("=" * 50)

    for tname in TARGETS:
        res = residuals[tname]
        base_frame = TARGET_FRAMES[tname]

        print(f"\n  {tname} (baseline frame = {base_frame}):")
        print(f"  {'Feature':<50} {'Global r':>9} {'Per-player r':>13} {'Player range':>20}")

        results = []

        for jname in joints_to_track:
            for frame in frames_to_check:
                vel = all_vel[jname][:, frame, :]

                # Decompose velocity
                v_fwd = np.zeros(n)
                v_lat = np.zeros(n)
                v_up = np.zeros(n)
                for i in range(n):
                    v_fwd[i], v_lat[i], v_up[i] = decompose_velocity(vel[i], player_pos[i])
                speed = np.linalg.norm(vel, axis=1)

                feat_dict = {'speed': speed, 'v_fwd': v_fwd, 'v_lat': v_lat, 'v_up': v_up}

                for comp_name, feat_vals in feat_dict.items():
                    # Global correlation with residuals
                    if np.std(feat_vals) < 1e-10:
                        continue
                    global_r = np.corrcoef(feat_vals, res)[0, 1]

                    # Per-player correlation with residuals
                    pp_r = []
                    for pid in sorted(np.unique(pids)):
                        mask = pids == pid
                        if np.sum(mask) < 10:
                            continue
                        fv = feat_vals[mask]
                        rv = res[mask]
                        if np.std(fv) < 1e-10 or np.std(rv) < 1e-10:
                            continue
                        r = np.corrcoef(fv, rv)[0, 1]
                        if np.isfinite(r):
                            pp_r.append(r)

                    if pp_r:
                        mean_pp_r = np.mean(pp_r)
                        fname = f"{jname:30s} f{frame:3d} {comp_name}"
                        results.append((abs(mean_pp_r), mean_pp_r, global_r, fname,
                                       min(pp_r), max(pp_r)))

                # Also check: position at a DIFFERENT frame than baseline
                # (does position at another time add info beyond position at target frame?)
                if frame != base_frame:
                    pos = all_pos[jname][:, frame, :]
                    pos_base = all_pos[jname][:, base_frame, :]
                    # Position DIFFERENCE between frames (displacement)
                    disp = pos - pos_base
                    for ax, ax_name in enumerate(['dx', 'dy', 'dz']):
                        feat_vals = disp[:, ax]
                        if np.std(feat_vals) < 1e-10:
                            continue
                        global_r = np.corrcoef(feat_vals, res)[0, 1]
                        pp_r = []
                        for pid in sorted(np.unique(pids)):
                            mask = pids == pid
                            if np.sum(mask) < 10:
                                continue
                            fv = feat_vals[mask]
                            rv = res[mask]
                            if np.std(fv) < 1e-10 or np.std(rv) < 1e-10:
                                continue
                            r = np.corrcoef(fv, rv)[0, 1]
                            if np.isfinite(r):
                                pp_r.append(r)
                        if pp_r:
                            mean_pp_r = np.mean(pp_r)
                            fname = f"{jname:30s} f{frame:3d} {ax_name} (disp from f{base_frame})"
                            results.append((abs(mean_pp_r), mean_pp_r, global_r, fname,
                                           min(pp_r), max(pp_r)))

        # Sort by absolute per-player correlation with residuals
        results.sort(reverse=True)

        for _, pp_r, global_r, fname, pmin, pmax in results[:25]:
            print(f"  {fname:<50} {global_r:>+9.4f} {pp_r:>+13.4f} [{pmin:>+.3f}, {pmax:>+.3f}]")

        # Summary: how much signal is left in residuals?
        print(f"\n  Top feature explains {results[0][1]**2 * 100:.1f}% of per-player residual variance")
        print(f"  Top 5 features (per-player r): ", end="")
        for _, pp_r, _, _, _, _ in results[:5]:
            print(f"{pp_r:+.4f} ", end="")
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
