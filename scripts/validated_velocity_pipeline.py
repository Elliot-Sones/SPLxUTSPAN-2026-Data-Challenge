"""
Validated Velocity Pipeline: features with PROVEN per-player correlations.

Based on diagnostic findings (physics_velocity_diagnostic.py, velocity_per_player_signal.py):

DEPTH (strongest signal):
  - v_up at f150-153 from left wrist, right wrist, elbow, neck (r=0.43-0.46)
  - v_fwd at f140 from right wrist (r=-0.42)
  Physical: upward velocity at release determines depth. Forward speed during
  acceleration inversely correlates (flatter = shallower).

LEFT_RIGHT (moderate signal):
  - neck/shoulder lateral velocity at f170 (r=0.29-0.37)
  Physical: body lateral drift during follow-through predicts ball drift.

ANGLE (weak signal):
  - knee speed at f153 (r=0.18)
  - neck lateral at f145 (r=-0.17)
  Physical: leg drive and body alignment affect launch angle.

Strategy: extract ONLY features with validated physical relationships.
Use per-target feature selection to maximize signal.
"""

from __future__ import annotations
import fcntl
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

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

HOOP_FEET = np.array([5.25, -25.0, 10.0])
FEET_TO_M = 0.3048
DT = 1.0 / 60.0
N_FRAMES = 240

TARGETS = ["angle", "depth", "left_right"]


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


def extract_features(ts_3d, kp_index, target_name):
    """
    Extract target-specific validated velocity features.
    Different features per target - competition mindset.
    """
    hip = get_joint(ts_3d, kp_index, 'mid_hip', smooth=True)
    player_pos = hip[20].copy()
    player_pos[2] = 0

    feats = []

    if target_name == "depth":
        # DEPTH: v_up at f150-153 is the key signal (r=0.43-0.46)
        for jname in ['left_wrist', 'right_wrist', 'right_elbow', 'neck',
                      'right_shoulder', 'left_shoulder']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True, window=11)
            vel = compute_velocity(traj, window=7)
            for ff in [145, 150, 153]:
                v_fwd, v_lat, v_up = decompose_velocity(vel[ff], player_pos)
                feats.append(v_up)
            # Also f140 forward (r=-0.42)
            v_fwd, v_lat, v_up = decompose_velocity(vel[140], player_pos)
            feats.append(v_fwd)

        # Add position context at key frames
        for jname in ['right_wrist', 'right_elbow']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True)
            for ff in [140, 150, 153]:
                rel = traj[ff] - HOOP_FEET
                feats.extend([rel[0], rel[1], rel[2]])

    elif target_name == "left_right":
        # LR: lateral velocity at f170 from neck/shoulders (r=0.29-0.37)
        for jname in ['neck', 'left_shoulder', 'right_shoulder', 'mid_hip',
                      'right_wrist', 'left_wrist']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True, window=11)
            vel = compute_velocity(traj, window=7)
            for ff in [155, 160, 170]:
                v_fwd, v_lat, v_up = decompose_velocity(vel[ff], player_pos)
                feats.append(v_lat)
            # Also add f130 forward (r=-0.20)
            v_fwd, v_lat, v_up = decompose_velocity(vel[130], player_pos)
            feats.append(v_fwd)

        # Finger lateral at f160 (r=-0.21)
        for jname in ['right_second_finger_distal', 'right_third_finger_distal']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True, window=11)
            vel = compute_velocity(traj, window=7)
            v_fwd, v_lat, v_up = decompose_velocity(vel[160], player_pos)
            feats.append(v_lat)

        # Position context
        for jname in ['right_wrist', 'neck']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True)
            for ff in [150, 170]:
                rel = traj[ff] - HOOP_FEET
                feats.extend([rel[0], rel[1], rel[2]])

    elif target_name == "angle":
        # ANGLE: weaker signal. Best are knee speed, neck lat, wrist speed
        for jname in ['right_knee', 'right_ankle', 'mid_hip']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True, window=11)
            vel = compute_velocity(traj, window=7)
            for ff in [145, 150, 153, 155]:
                speed = np.linalg.norm(vel[ff])
                feats.append(speed)
                v_fwd, v_lat, v_up = decompose_velocity(vel[ff], player_pos)
                feats.append(v_up)

        # Neck/shoulder lateral (r=-0.15 to -0.17)
        for jname in ['neck', 'right_shoulder', 'left_shoulder']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True, window=11)
            vel = compute_velocity(traj, window=7)
            for ff in [140, 145, 150]:
                v_fwd, v_lat, v_up = decompose_velocity(vel[ff], player_pos)
                feats.append(v_lat)

        # Arm velocity features
        for jname in ['right_wrist', 'right_elbow']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True, window=11)
            vel = compute_velocity(traj, window=7)
            for ff in [130, 140, 150, 153]:
                speed = np.linalg.norm(vel[ff])
                v_fwd, v_lat, v_up = decompose_velocity(vel[ff], player_pos)
                feats.extend([speed, v_fwd, v_up])

        # Position context
        for jname in ['right_wrist', 'right_elbow']:
            traj = get_joint(ts_3d, kp_index, jname, smooth=True)
            for ff in [140, 150, 153]:
                rel = traj[ff] - HOOP_FEET
                feats.extend([rel[0], rel[1], rel[2]])

    return np.array(feats, dtype=np.float32)


def ridge_loo_per_player(X, y, pids, bw_q=0.3, alpha=10.0):
    """Per-player locally weighted Ridge LOO. No PLS to avoid leakage."""
    n = len(y)
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

        D = pairwise_distances(X_s)
        bw = np.quantile(D[np.triu_indices(n_p, k=1)], bw_q)
        if bw < 1e-10:
            bw = 1.0

        for i in range(n_p):
            w = np.exp(-0.5 * (D[i] / bw) ** 2)
            w[i] = 0
            if w.sum() < 1e-10:
                preds[idx[i]] = np.mean(np.delete(y_p, i))
                continue
            sqrt_w = np.sqrt(w)
            X_w = X_s * sqrt_w[:, None]
            y_w = y_p * sqrt_w
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(np.delete(X_w, i, 0), np.delete(y_w, i))
            preds[idx[i]] = ridge.predict(X_s[i:i + 1])[0]

    return preds


def predict_test(X_train, y_train, pids_train, X_test, pids_test,
                 bw_q=0.3, alpha=10.0):
    n_test = len(X_test)
    preds = np.full(n_test, np.nan)

    for pid in sorted(np.unique(pids_train)):
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        te_idx = np.where(te_mask)[0]

        X_p, y_p = X_train[tr_mask], y_train[tr_mask]
        X_te = X_test[te_mask]
        n_p = len(y_p)

        if n_p < 5 or len(te_idx) == 0:
            if len(te_idx) > 0:
                preds[te_idx] = np.mean(y_p) if n_p > 0 else 0.5
            continue

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)
        X_te_s = scaler.transform(X_te)

        D_train = pairwise_distances(X_s)
        bw = np.quantile(D_train[np.triu_indices(n_p, k=1)], bw_q)
        if bw < 1e-10:
            bw = 1.0

        for j, ti in enumerate(te_idx):
            d = pairwise_distances(X_te_s[j:j + 1], X_s)[0]
            w = np.exp(-0.5 * (d / bw) ** 2)
            if w.sum() < 1e-10:
                preds[ti] = np.mean(y_p)
                continue
            sqrt_w = np.sqrt(w)
            X_w = X_s * sqrt_w[:, None]
            y_w = y_p * sqrt_w
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_w, y_w)
            preds[ti] = ridge.predict(X_te_s[j:j + 1])[0]

    return preds


def get_next_sub_num():
    lock = SUBMISSION_DIR / ".submission_lock"
    lock.touch(exist_ok=True)
    with open(lock, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            nums = [int(p.stem.split("_")[1]) for p in SUBMISSION_DIR.glob("submission_*.csv")
                    if p.stem.split("_")[1].isdigit()]
            n = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{n}.csv").touch()
            return n
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    print("=" * 70)
    print("VALIDATED VELOCITY PIPELINE")
    print("Per-target features with proven per-player correlations")
    print("=" * 70)

    print("\nLoading data...")
    X_3d_tr, kp_names, kp_index, train_df = load_data(DATA_DIR / "train.csv")
    X_3d_te, _, _, test_df = load_data(DATA_DIR / "test.csv")

    pids_tr = train_df["participant_id"].values
    pids_te = test_df["participant_id"].values

    y_scaled = {}
    for t in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{t}.pkl"
        if scaler_path.exists():
            s = joblib.load(scaler_path)
            y_scaled[t] = s.transform(train_df[t].values.reshape(-1, 1)).ravel()
        else:
            v = train_df[t].values
            y_scaled[t] = (v - v.min()) / (v.max() - v.min() + 1e-10)

    # Extract per-target features
    all_loo = {}
    all_test = {}

    for tname in TARGETS:
        print(f"\n{'=' * 50}")
        print(f"TARGET: {tname}")
        print(f"{'=' * 50}")

        print(f"  Extracting features (train)...")
        train_feats = []
        for i in range(len(train_df)):
            feats = extract_features(X_3d_tr[i], kp_index, tname)
            train_feats.append(feats)
        X_train = np.array(train_feats)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  {X_train.shape[1]} features")

        print(f"  Extracting features (test)...")
        test_feats = []
        for i in range(len(test_df)):
            feats = extract_features(X_3d_te[i], kp_index, tname)
            test_feats.append(feats)
        X_test = np.array(test_feats)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # LOO
        y = y_scaled[tname]
        loo_preds = ridge_loo_per_player(X_train, y, pids_tr)
        loo_mse = np.mean((loo_preds - y) ** 2)

        for pid in sorted(np.unique(pids_tr)):
            m = pids_tr == pid
            pmse = np.mean((loo_preds[m] - y[m]) ** 2)
            print(f"  P{pid}: LOO MSE = {pmse:.6f}")
        print(f"  GLOBAL LOO MSE = {loo_mse:.6f}")

        # Test predictions
        test_preds = predict_test(X_train, y, pids_tr, X_test, pids_te)
        all_loo[tname] = loo_preds
        all_test[tname] = test_preds

    mean_loo = np.mean([np.mean((all_loo[t] - y_scaled[t]) ** 2) for t in TARGETS])
    print(f"\nMEAN LOO MSE: {mean_loo:.6f}")
    print(f"  Compare: honest LOO baseline = 0.006830")
    print(f"  Compare: sparse physics = 0.013831")

    # Diversity check
    base_path = SUBMISSION_DIR / "submission_3190.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        print(f"\nDiversity vs Sub 3190:")
        for t in TARGETS:
            col = f"scaled_{t}"
            test_p = np.clip(all_test[t], 0, 1)
            r = np.corrcoef(base[col], test_p)[0, 1]
            print(f"  {t}: r = {r:.4f}")

    # Generate submissions
    sub_num = get_next_sub_num()
    sub = pd.DataFrame({"id": test_df["id"]})
    for t in TARGETS:
        sub[f"scaled_{t}"] = np.clip(all_test[t], 0, 1)
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\nStandalone: submission_{sub_num}.csv")

    # Blends with Sub 3190
    if base_path.exists():
        base = pd.read_csv(base_path)

        # Standard blends
        for pct in [2, 5, 10]:
            w = pct / 100.0
            bn = get_next_sub_num()
            blend = pd.DataFrame({"id": test_df["id"]})
            for t in TARGETS:
                col = f"scaled_{t}"
                blend[col] = (1 - w) * base[col] + w * np.clip(all_test[t], 0, 1)
            blend.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
            print(f"  {pct}% blend: submission_{bn}.csv")

        # Per-target blend: use depth-only velocity features (strongest signal)
        bn = get_next_sub_num()
        blend = pd.DataFrame({"id": test_df["id"]})
        for t in TARGETS:
            col = f"scaled_{t}"
            if t == "depth":
                # Depth has r=0.46 signal - use higher weight
                blend[col] = 0.93 * base[col] + 0.07 * np.clip(all_test[t], 0, 1)
            elif t == "left_right":
                # LR has r=0.37 signal - moderate weight
                blend[col] = 0.95 * base[col] + 0.05 * np.clip(all_test[t], 0, 1)
            else:
                # Angle weak signal - minimal weight
                blend[col] = 0.98 * base[col] + 0.02 * np.clip(all_test[t], 0, 1)
        blend.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
        print(f"  Per-target weighted blend: submission_{bn}.csv (angle 2%, depth 7%, LR 5%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
