"""
Physics-Based Multi-Phase Feature Extraction Pipeline

Instead of statistically optimizing which frames to use (circular/overfit),
detect biomechanical events from the motion data and extract features at each event.

Free throw phases (detected from kinematics, NOT from targets):
1. STANCE: initial body position, alignment to hoop (first stable frame)
2. DIP: lowest point of the wind-up (hip z minimum before shot)
3. LOAD PEAK: maximum knee bend / lowest center of mass
4. SET POINT: wrist crosses eye level on way up
5. RELEASE: peak ball/wrist speed (already detected)
6. FOLLOW-THROUGH: wrist reaches highest point after release

Features from each phase capture different physical quantities:
- Stance: body alignment, foot spacing, initial posture
- Dip: loading depth, energy stored
- Release: velocity, angle, position at moment of truth
- Follow-through: continuation of release mechanics (less noisy measurement)

Frame selection is target-independent (uses only joint kinematics).
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

HOOP = np.array([5.25, -25.0, 10.0])
FEET_TO_METERS = 0.3048
DT = 1.0 / 60.0
N_FRAMES = 240

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["angle", "depth", "left_right"]


def safe_savgol(arr, window, polyorder, deriv=0, delta=1.0):
    arr = np.array(arr, dtype=np.float64)
    bad = np.isnan(arr) | np.isinf(arr)
    if np.all(bad):
        return np.zeros_like(arr)
    if np.any(bad):
        good = ~bad
        arr[bad] = np.interp(np.where(bad)[0], np.where(good)[0], arr[good])
    if len(arr) < window:
        return arr
    return savgol_filter(arr, window, polyorder, deriv=deriv, delta=delta)


def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    arr = np.array(json.loads(s), dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]

    kp_names = []
    for col in kp_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
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


# ================================================================
# PHASE DETECTION (target-independent, kinematics only)
# ================================================================

def detect_phases(ts_3d, kp_index):
    """
    Detect biomechanical phase boundaries from joint kinematics.
    Returns dict of frame indices for each event.
    All detection uses ONLY joint positions/velocities - never targets.
    """
    phases = {}

    # --- Get key trajectories ---
    rw_idx = kp_index.get('right_wrist')
    re_idx = kp_index.get('right_elbow')
    rs_idx = kp_index.get('right_shoulder')
    mh_idx = kp_index.get('mid_hip', 0)
    rk_idx = kp_index.get('right_knee')
    lk_idx = kp_index.get('left_knee')
    neck_idx = kp_index.get('neck')

    wrist = ts_3d[:, rw_idx, :].copy() if rw_idx is not None else ts_3d[:, 0, :].copy()
    hip = ts_3d[:, mh_idx, :].copy()
    elbow = ts_3d[:, re_idx, :].copy() if re_idx is not None else ts_3d[:, 0, :].copy()
    shoulder = ts_3d[:, rs_idx, :].copy() if rs_idx is not None else ts_3d[:, 0, :].copy()

    # Clean NaN
    for traj in [wrist, hip, elbow, shoulder]:
        for ax in range(3):
            vals = traj[:, ax]
            bad = np.isnan(vals) | np.isinf(vals)
            if np.any(bad) and not np.all(bad):
                good = ~bad
                vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
            traj[:, ax] = vals

    # Smooth trajectories
    wrist_z = safe_savgol(wrist[:, 2], 15, 3)
    hip_z = safe_savgol(hip[:, 2], 15, 3)
    wrist_speed_z = safe_savgol(wrist[:, 2], 11, 3, deriv=1, delta=DT)

    # --- 1. STANCE: first stable frame (low velocity, early in sequence) ---
    # Use frame 10-30 where player is settled but hasn't started motion
    wrist_total_speed = np.sqrt(
        safe_savgol(wrist[:, 0], 11, 3, deriv=1, delta=DT)**2 +
        safe_savgol(wrist[:, 1], 11, 3, deriv=1, delta=DT)**2 +
        wrist_speed_z**2
    )
    # Find lowest-motion frame in 10-40 range
    if len(wrist_total_speed) > 40:
        stance_frame = 10 + np.argmin(wrist_total_speed[10:40])
    else:
        stance_frame = 15
    phases['stance'] = int(np.clip(stance_frame, 5, 50))

    # --- 2. DIP: lowest hip position before shot (loading phase) ---
    # Hip z minimum in frames 30-130
    if len(hip_z) > 130:
        dip_frame = 30 + np.argmin(hip_z[30:130])
    else:
        dip_frame = 80
    phases['dip'] = int(np.clip(dip_frame, 30, 130))

    # --- 3. SET POINT: wrist crosses approximate eye level going up ---
    # Eye level ~ neck_z + small offset, or ~6 feet typical
    if neck_idx is not None:
        neck_z = safe_savgol(ts_3d[:, neck_idx, 2], 15, 3)
        eye_level = np.median(neck_z[10:40]) + 0.3  # slightly above neck at stance
    else:
        eye_level = 5.5

    set_point = None
    for i in range(max(60, phases['dip']), min(180, N_FRAMES - 1)):
        if wrist_z[i] < eye_level <= wrist_z[i + 1]:
            set_point = i
            break
    if set_point is None:
        set_point = phases['dip'] + 30  # fallback
    phases['set_point'] = int(np.clip(set_point, 60, 170))

    # --- 4. RELEASE: peak ball speed ---
    # Estimate ball position from wrist + fingertip offset
    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = []
    for k in ft_keys:
        if k in kp_index:
            ft = ts_3d[:, kp_index[k], :].copy()
            for ax in range(3):
                vals = ft[:, ax]
                bad = np.isnan(vals) | np.isinf(vals)
                if np.any(bad) and not np.all(bad):
                    good = ~bad
                    vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
            ft_trajs.append(ft)

    if ft_trajs:
        ft_center = np.mean(ft_trajs, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist + 0.6 * (ft_center - wrist)
    else:
        ball = wrist.copy()

    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    ball_vel = np.zeros_like(ball)
    for ax in range(3):
        ball_vel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_METERS, 9, 3, deriv=1, delta=DT)
    ball_speed = np.linalg.norm(ball_vel, axis=1)

    wrist_peak = 80 + np.argmax(wrist_z[80:200]) if len(wrist_z) > 200 else 140
    search_start = max(80, wrist_peak - 40)
    search_end = min(wrist_peak + 5, 200)
    if search_end > search_start:
        release_frame = search_start + np.argmax(ball_speed[search_start:search_end])
    else:
        release_frame = 140
    phases['release'] = int(np.clip(release_frame, 80, 200))

    # --- 5. FOLLOW-THROUGH: wrist highest point after release ---
    post_release_start = phases['release'] + 5
    post_release_end = min(phases['release'] + 60, N_FRAMES)
    if post_release_end > post_release_start:
        ft_frame = post_release_start + np.argmax(wrist_z[post_release_start:post_release_end])
    else:
        ft_frame = phases['release'] + 20
    phases['follow_through'] = int(np.clip(ft_frame, phases['release'] + 5, 230))

    return phases


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def compute_hoop_transform(ts_3d, kp_index):
    """Hoop-relative coordinate transform."""
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0

    forward = HOOP[:2] - player_pos[:2]
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
    return np.einsum('ij,fkj->fki', R, centered)


def extract_phase_features(ts_3d, ts_hr, kp_index, phases):
    """
    Extract features at each detected phase.
    Returns a flat feature vector combining all phases.
    """
    feats = []

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    joint_indices = [kp_index.get(j) for j in key_joints]
    valid_joints = [(j, idx) for j, idx in zip(key_joints, joint_indices) if idx is not None]

    # Extract features at each phase frame
    for phase_name in ['stance', 'dip', 'set_point', 'release', 'follow_through']:
        f = phases[phase_name]
        f = int(np.clip(f, 0, N_FRAMES - 1))

        # Hoop-relative positions of key joints
        for jname, jidx in valid_joints:
            pos = ts_hr[f, jidx, :]
            feats.extend(pos.tolist())

        # Velocities at this frame (from smoothed trajectories)
        if f > 0 and f < N_FRAMES - 1:
            for jname, jidx in valid_joints:
                vel = (ts_hr[min(f+1, N_FRAMES-1), jidx, :] - ts_hr[max(f-1, 0), jidx, :]) / (2 * DT)
                feats.extend(vel.tolist())
        else:
            feats.extend([0.0] * (len(valid_joints) * 3))

    # Inter-phase dynamics (differences between phases)
    phase_frames = [phases[p] for p in ['stance', 'dip', 'set_point', 'release', 'follow_through']]

    # Timing features (phase durations in frames)
    feats.append(phases['dip'] - phases['stance'])           # wind-up duration
    feats.append(phases['set_point'] - phases['dip'])        # loading duration
    feats.append(phases['release'] - phases['set_point'])    # acceleration duration
    feats.append(phases['follow_through'] - phases['release'])  # follow-through duration
    feats.append(phases['release'])                          # absolute release timing

    # Joint angle features at key phases
    for phase_name in ['dip', 'release', 'follow_through']:
        f = int(np.clip(phases[phase_name], 0, N_FRAMES - 1))

        # Right elbow angle
        if all(k in kp_index for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            s = ts_3d[f, kp_index['right_shoulder'], :]
            e = ts_3d[f, kp_index['right_elbow'], :]
            w = ts_3d[f, kp_index['right_wrist'], :]
            v1 = s - e; v2 = w - e
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            feats.append(float(np.arccos(np.clip(cos_a, -1, 1))))

        # Right knee angle
        if all(k in kp_index for k in ['right_hip', 'right_knee', 'right_ankle']):
            h = ts_3d[f, kp_index['right_hip'], :]
            k_pos = ts_3d[f, kp_index['right_knee'], :]
            a = ts_3d[f, kp_index.get('right_ankle', 0), :]
            v1 = h - k_pos; v2 = a - k_pos
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            feats.append(float(np.arccos(np.clip(cos_a, -1, 1))))

        # Shoulder-hip alignment (lateral tilt)
        if all(k in kp_index for k in ['right_shoulder', 'left_shoulder']):
            rs = ts_hr[f, kp_index['right_shoulder'], :]
            ls = ts_hr[f, kp_index['left_shoulder'], :]
            shoulder_vec = rs - ls
            feats.append(float(np.arctan2(shoulder_vec[2], shoulder_vec[0])))  # tilt angle

    return np.array(feats, dtype=np.float32)


# ================================================================
# PIPELINE: per-player Ridge LOO with multi-phase features
# ================================================================

def run_pipeline(train_data, test_data, kp_index, target_name,
                 bw_quantile=0.3, alpha=10.0, n_pls=10):
    """
    Full pipeline: detect phases -> extract multi-phase features ->
    PLS compress -> per-player locally weighted Ridge.
    Returns LOO predictions, test predictions, and LOO MSE.
    """
    X_3d_train, pids_train, y_train = train_data
    X_3d_test, pids_test = test_data
    n_train = len(y_train)
    n_test = X_3d_test.shape[0]

    unique_pids = np.unique(pids_train)

    # Phase detection + feature extraction
    print(f"    Detecting phases and extracting features...")
    train_features = []
    train_phases_list = []
    for i in range(n_train):
        phases = detect_phases(X_3d_train[i], kp_index)
        train_phases_list.append(phases)
        ts_hr = compute_hoop_transform(X_3d_train[i], kp_index)
        feats = extract_phase_features(X_3d_train[i], ts_hr, kp_index, phases)
        train_features.append(feats)

    test_features = []
    for i in range(n_test):
        phases = detect_phases(X_3d_test[i], kp_index)
        ts_hr = compute_hoop_transform(X_3d_test[i], kp_index)
        feats = extract_phase_features(X_3d_test[i], ts_hr, kp_index, phases)
        test_features.append(feats)

    X_train = np.array(train_features, dtype=np.float32)
    X_test = np.array(test_features, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    n_feats = X_train.shape[1]
    print(f"    {n_feats} multi-phase features extracted")

    # Print phase timing stats
    stance_frames = [p['stance'] for p in train_phases_list]
    dip_frames = [p['dip'] for p in train_phases_list]
    release_frames = [p['release'] for p in train_phases_list]
    ft_frames = [p['follow_through'] for p in train_phases_list]
    print(f"    Phase stats (mean +/- std):")
    print(f"      stance:  {np.mean(stance_frames):.0f} +/- {np.std(stance_frames):.0f}")
    print(f"      dip:     {np.mean(dip_frames):.0f} +/- {np.std(dip_frames):.0f}")
    print(f"      release: {np.mean(release_frames):.0f} +/- {np.std(release_frames):.0f}")
    print(f"      follow:  {np.mean(ft_frames):.0f} +/- {np.std(ft_frames):.0f}")

    # Per-player LOO
    loo_preds = np.full(n_train, np.nan)
    test_preds = np.full(n_test, np.nan)

    for pid in unique_pids:
        train_mask = pids_train == pid
        test_mask = pids_test == pid
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        X_p = X_train[train_mask]
        y_p = y_train[train_mask]
        n_p = len(y_p)

        if n_p < 5:
            loo_preds[train_idx] = np.mean(y_p)
            if test_idx.size > 0:
                test_preds[test_idx] = np.mean(y_p)
            continue

        # PLS compression (fit on all player data for test predictions,
        # refit per LOO fold for honest LOO)
        n_comp = min(n_pls, n_p - 1, X_p.shape[1])
        if n_comp < 1:
            n_comp = 1

        # Scaler for this player
        scaler = StandardScaler()
        X_p_scaled = scaler.fit_transform(X_p)

        # PLS on full player data (for test predictions)
        pls_full = PLSRegression(n_components=n_comp, scale=False)
        pls_full.fit(X_p_scaled, y_p)
        X_p_pls_full = pls_full.transform(X_p_scaled)

        # Combine: raw hoop-relative at release + PLS components
        # Use raw features from release phase only (subset) + PLS from all phases
        # This keeps the distance metric manageable
        X_p_combined_full = np.hstack([X_p_scaled, X_p_pls_full])

        # Distance matrix
        D = pairwise_distances(X_p_combined_full)
        dists_flat = D[np.triu_indices(n_p, k=1)]
        bw = np.quantile(dists_flat, bw_quantile) if len(dists_flat) > 0 else 1.0
        if bw < 1e-10:
            bw = 1.0

        # LOO predictions (honest: refit PLS per fold)
        for i in range(n_p):
            # Honest PLS: refit without held-out point
            X_loo = np.delete(X_p_scaled, i, axis=0)
            y_loo = np.delete(y_p, i, axis=0)

            pls_loo = PLSRegression(n_components=n_comp, scale=False)
            pls_loo.fit(X_loo, y_loo)

            X_all_pls_loo = pls_loo.transform(X_p_scaled)
            X_combined_loo = np.hstack([X_p_scaled, X_all_pls_loo])

            D_loo = pairwise_distances(X_combined_loo)
            bw_loo = np.quantile(D_loo[np.triu_indices(n_p, k=1)], bw_quantile)
            if bw_loo < 1e-10:
                bw_loo = 1.0

            weights = np.exp(-0.5 * (D_loo[i] / bw_loo) ** 2)
            weights[i] = 0

            if weights.sum() < 1e-10:
                loo_preds[train_idx[i]] = np.mean(np.delete(y_p, i))
                continue

            sqrt_w = np.sqrt(weights)
            X_w = X_combined_loo * sqrt_w[:, np.newaxis]
            y_w = y_p * sqrt_w

            X_tr = np.delete(X_w, i, axis=0)
            y_tr = np.delete(y_w, i, axis=0)

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_tr, y_tr)
            loo_preds[train_idx[i]] = ridge.predict(X_combined_loo[i:i+1])[0]

        # Test predictions
        if test_idx.size > 0:
            X_test_p = X_test[test_mask]
            X_test_p_scaled = scaler.transform(X_test_p)
            X_test_p_pls = pls_full.transform(X_test_p_scaled)
            X_test_combined = np.hstack([X_test_p_scaled, X_test_p_pls])

            for j in range(len(test_idx)):
                d_test = pairwise_distances(X_test_combined[j:j+1], X_p_combined_full)[0]
                weights = np.exp(-0.5 * (d_test / bw) ** 2)

                if weights.sum() < 1e-10:
                    test_preds[test_idx[j]] = np.mean(y_p)
                    continue

                sqrt_w = np.sqrt(weights)
                X_w = X_p_combined_full * sqrt_w[:, np.newaxis]
                y_w = y_p * sqrt_w

                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(X_w, y_w)
                test_preds[test_idx[j]] = ridge.predict(X_test_combined[j:j+1])[0]

    loo_mse = np.mean((loo_preds - y_train) ** 2)
    return loo_preds, test_preds, loo_mse


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split("_")[1]) for fp in existing
                    if fp.stem.split("_")[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    print("=" * 70)
    print("PHYSICS MULTI-PHASE PIPELINE")
    print("Phase detection from kinematics (target-independent)")
    print("Multi-frame feature extraction at detected phases")
    print("=" * 70)

    # Load data
    print("\nLoading train data...")
    X_3d_train, kp_names, kp_index, train_df = load_data(DATA_DIR / "train.csv")
    print("Loading test data...")
    X_3d_test, _, _, test_df = load_data(DATA_DIR / "test.csv")

    pids_train = train_df["participant_id"].values
    pids_test = test_df["participant_id"].values

    # Scale targets
    scalers = {}
    y_scaled = {}
    for t in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{t}.pkl"
        if scaler_path.exists():
            scalers[t] = joblib.load(scaler_path)
            y_scaled[t] = scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel()
        else:
            vals = train_df[t].values
            mn, mx = vals.min(), vals.max()
            scalers[t] = (mn, mx)
            y_scaled[t] = (vals - mn) / (mx - mn + 1e-10)

    # Run per-target
    all_loo = {}
    all_test = {}

    for tname in TARGETS:
        print(f"\n{'='*50}")
        print(f"TARGET: {tname}")
        print(f"{'='*50}")

        y = y_scaled[tname]
        train_data = (X_3d_train, pids_train, y)
        test_data = (X_3d_test, pids_test)

        loo_preds, test_preds, loo_mse = run_pipeline(
            train_data, test_data, kp_index, tname,
            bw_quantile=0.3, alpha=10.0, n_pls=10
        )

        all_loo[tname] = loo_preds
        all_test[tname] = test_preds

        # Per-player breakdown
        for pid in sorted(np.unique(pids_train)):
            mask = pids_train == pid
            pmse = np.mean((loo_preds[mask] - y[mask]) ** 2)
            print(f"    P{pid}: LOO MSE = {pmse:.6f} (n={mask.sum()})")

        print(f"  GLOBAL honest LOO MSE = {loo_mse:.6f}")

    # Overall
    mean_loo = np.mean([np.mean((all_loo[t] - y_scaled[t]) ** 2) for t in TARGETS])
    print(f"\n{'='*70}")
    print(f"MEAN LOO MSE across targets: {mean_loo:.6f}")
    print(f"{'='*70}")

    # Generate standalone submission
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({"id": test_df["id"]})
    for t in TARGETS:
        col = f"scaled_{t}"
        preds = all_test[t]
        preds = np.clip(preds, 0, 1)
        sub[col] = preds

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nStandalone submission: {sub_path}")
    print(f"  Predictions stats:")
    for t in TARGETS:
        col = f"scaled_{t}"
        print(f"    {t}: mean={sub[col].mean():.4f}, std={sub[col].std():.4f}, "
              f"min={sub[col].min():.4f}, max={sub[col].max():.4f}")

    # Generate blend with current best (Sub 3190)
    base_sub_path = SUBMISSION_DIR / "submission_3190.csv"
    if base_sub_path.exists():
        base = pd.read_csv(base_sub_path)
        for blend_pct in [2, 5, 10, 15]:
            w = blend_pct / 100.0
            blend_num = get_next_submission_number()
            blend = pd.DataFrame({"id": test_df["id"]})
            for t in TARGETS:
                col = f"scaled_{t}"
                blend[col] = (1 - w) * base[col] + w * np.clip(all_test[t], 0, 1)
            blend_path = SUBMISSION_DIR / f"submission_{blend_num}.csv"
            blend.to_csv(blend_path, index=False)
            print(f"  Blend {blend_pct}%: {blend_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
