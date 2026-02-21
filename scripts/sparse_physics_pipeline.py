"""
Sparse Physics Pipeline: Few features, high signal.

Phase detection via physics (not statistics):
1. T_windup: knee flexion peak (maximum stored energy)
2. T_release: ball acceleration drops to zero (F=ma, when F drops = release)
3. T_follow: arm reaches maximum extension after release

Feature extraction guided by causality:
- At T_windup: energy stored (knee angle, CoM drop), body alignment
- At T_release: ball velocity vector (DIRECTLY determines trajectory),
  release position, joint contributions, wrist snap
- At T_follow: wrist pointing direction, arm extension, body alignment

Each feature has a KNOWN physical relationship to targets:
- angle: release Vz/Vh ratio, elbow angle, forearm elevation
- depth: release speed magnitude, stored energy, release height
- left_right: release Vx, body lateral alignment, wrist lateral direction
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

HOOP_FEET = np.array([5.25, -25.0, 10.0])
HOOP_M = HOOP_FEET * 0.3048
FEET_TO_M = 0.3048
DT = 1.0 / 60.0
G = 9.81  # m/s^2
M_BALL = 0.625  # kg
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


# ================================================================
# PHYSICS-BASED PHASE DETECTION
# ================================================================

def get_joint(ts_3d, kp_index, name, smooth=True):
    """Get a joint trajectory (N_FRAMES, 3), optionally smoothed."""
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
            traj[:, ax] = safe_savgol(traj[:, ax], 15, 3)
    return traj


def compute_joint_angle(p1, p2, p3):
    """Angle at p2 between vectors p2->p1 and p2->p3. Returns radians."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return np.pi
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.arccos(cos_a)


def detect_phases(ts_3d, kp_index):
    """
    Physics-based phase detection. No targets used.

    T_windup: maximum knee flexion (deepest squat before shot)
    T_release: ball acceleration drops below threshold (F~0 = ball free)
    T_follow: maximum wrist height after release
    """
    # Get key trajectories
    wrist = get_joint(ts_3d, kp_index, 'right_wrist')
    hip = get_joint(ts_3d, kp_index, 'mid_hip')
    r_knee = get_joint(ts_3d, kp_index, 'right_knee')
    r_hip = get_joint(ts_3d, kp_index, 'right_hip')
    r_ankle = get_joint(ts_3d, kp_index, 'right_ankle')

    # Fingertip trajectory for ball position
    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = [get_joint(ts_3d, kp_index, k) for k in ft_keys if k in kp_index]
    if ft_trajs:
        ft_center = np.mean(ft_trajs, axis=0)
    else:
        ft_center = wrist.copy()

    # Ball position estimate (between wrist and fingertips)
    ball = wrist + 0.6 * (ft_center - wrist)
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    # === T_WINDUP: Maximum knee flexion ===
    # Compute knee angle at each frame (hip-knee-ankle)
    knee_angles = np.zeros(N_FRAMES)
    for f in range(N_FRAMES):
        knee_angles[f] = compute_joint_angle(r_hip[f], r_knee[f], r_ankle[f])
    knee_angles = safe_savgol(knee_angles, 15, 3)

    # Knee angle MINIMUM in frames 30-140 = maximum flexion = deepest squat
    search_start, search_end = 30, 140
    t_windup = search_start + np.argmin(knee_angles[search_start:search_end])

    # === T_RELEASE: Ball acceleration drops to ~0 (force-based) ===
    # Compute ball acceleration: a = d2x/dt2
    ball_accel = np.zeros((N_FRAMES, 3))
    for ax in range(3):
        ball_accel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_M, 9, 3, deriv=2, delta=DT)

    # Net force on ball = m * a. In contact, force is large. At release, force -> gravity only.
    # We want the frame where the non-gravitational force drops.
    # Gravitational acceleration = [0, 0, -9.81]
    # So non-gravity accel = a - [0, 0, -g]
    gravity = np.array([0, 0, -G])
    non_grav_accel = ball_accel - gravity[np.newaxis, :]
    non_grav_force_mag = np.linalg.norm(non_grav_accel, axis=1) * M_BALL

    # Smooth the force magnitude
    non_grav_force_mag = safe_savgol(non_grav_force_mag, 11, 3)

    # Find where force drops from high to low after the windup
    # First find the peak force (during acceleration phase)
    force_search_start = max(t_windup, 60)
    force_search_end = 200
    if force_search_end > force_search_start:
        peak_force_frame = force_search_start + np.argmax(
            non_grav_force_mag[force_search_start:force_search_end])

        # Release = where force drops below 30% of peak, AFTER peak
        peak_force = non_grav_force_mag[peak_force_frame]
        threshold = peak_force * 0.30

        t_release = peak_force_frame
        for f in range(peak_force_frame + 1, min(peak_force_frame + 40, N_FRAMES)):
            if non_grav_force_mag[f] < threshold:
                t_release = f
                break
    else:
        t_release = 140  # fallback

    t_release = int(np.clip(t_release, 80, 200))

    # === T_FOLLOW: Maximum wrist height after release ===
    wrist_z = safe_savgol(wrist[:, 2], 11, 3)
    follow_start = min(t_release + 3, N_FRAMES - 1)
    follow_end = min(t_release + 60, N_FRAMES)
    if follow_end > follow_start:
        t_follow = follow_start + np.argmax(wrist_z[follow_start:follow_end])
    else:
        t_follow = t_release + 15

    t_follow = int(np.clip(t_follow, t_release + 3, 235))

    return {
        'windup': int(t_windup),
        'release': int(t_release),
        'follow': int(t_follow),
        # Diagnostic
        'peak_force_N': float(non_grav_force_mag[peak_force_frame]) if force_search_end > force_search_start else 0,
        'knee_angle_at_windup': float(np.degrees(knee_angles[t_windup])),
    }


# ================================================================
# PHYSICS-MOTIVATED FEATURE EXTRACTION
# ================================================================

def extract_sparse_features(ts_3d, kp_index, phases):
    """
    Extract physically meaningful features at 3 phase frames.
    Every feature has a known causal relationship to at least one target.
    """
    feats = []

    tw = int(np.clip(phases['windup'], 0, 239))
    tr = int(np.clip(phases['release'], 0, 239))
    tf = int(np.clip(phases['follow'], 0, 239))

    # Get key joint trajectories
    wrist = get_joint(ts_3d, kp_index, 'right_wrist')
    elbow = get_joint(ts_3d, kp_index, 'right_elbow')
    shoulder = get_joint(ts_3d, kp_index, 'right_shoulder')
    l_shoulder = get_joint(ts_3d, kp_index, 'left_shoulder')
    hip = get_joint(ts_3d, kp_index, 'mid_hip')
    r_hip = get_joint(ts_3d, kp_index, 'right_hip')
    r_knee = get_joint(ts_3d, kp_index, 'right_knee')
    r_ankle = get_joint(ts_3d, kp_index, 'right_ankle')
    neck = get_joint(ts_3d, kp_index, 'neck')

    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = [get_joint(ts_3d, kp_index, k) for k in ft_keys if k in kp_index]
    ft_center = np.mean(ft_trajs, axis=0) if ft_trajs else wrist.copy()
    ball = wrist + 0.6 * (ft_center - wrist)
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    # Player position for hoop-relative calculations (at stance)
    player_pos = hip[20, :].copy()
    player_pos[2] = 0

    # Hoop direction
    to_hoop = HOOP_FEET[:2] - player_pos[:2]
    hoop_dist = np.linalg.norm(to_hoop)
    if hoop_dist > 1e-6:
        forward = to_hoop / hoop_dist
    else:
        forward = np.array([0, -1.0])
    lateral = np.array([-forward[1], forward[0]])

    # ==========================================
    # AT WINDUP (T_windup): Stored energy, alignment
    # ==========================================

    # 1. Knee angle at maximum flexion (stored elastic energy in legs)
    #    CAUSES: depth (more bend -> more energy -> higher release speed)
    knee_angle_windup = compute_joint_angle(r_hip[tw], r_knee[tw], r_ankle[tw])
    feats.append(knee_angle_windup)

    # 2. Center of mass drop from stance (gravitational potential energy stored)
    #    CAUSES: depth (more drop -> more energy)
    com_stance_z = hip[20, 2]
    com_windup_z = hip[tw, 2]
    com_drop = com_stance_z - com_windup_z  # positive = deeper squat
    feats.append(com_drop)

    # 3. Body lateral alignment to hoop at windup
    #    CAUSES: left_right (misalignment -> lateral error)
    shoulder_vec = shoulder[tw] - l_shoulder[tw]
    shoulder_lateral = np.dot(shoulder_vec[:2], lateral)
    feats.append(shoulder_lateral)

    # 4. Ball height at windup (how low the dip goes)
    #    CAUSES: depth (lower dip -> more acceleration distance)
    feats.append(ball[tw, 2])

    # 5. Windup timing (frames from stance to dip)
    #    CAUSES: rhythm affects all targets
    feats.append(float(tw))

    # ==========================================
    # AT RELEASE (T_release): Ball velocity, position, form
    # ==========================================

    # 6-8. Ball velocity vector at release (THE most direct cause of all targets)
    #    v_z/v_horizontal -> angle
    #    |v| -> depth (speed determines how far ball goes)
    #    v_lateral -> left_right
    ball_vel = np.zeros(3)
    for ax in range(3):
        ball_vel[ax] = safe_savgol(ball[:, ax] * FEET_TO_M, 9, 3, deriv=1, delta=DT)[tr]

    # Project velocity onto court coordinates
    v_forward = ball_vel[0] * forward[0] + ball_vel[1] * forward[1]
    v_lateral = ball_vel[0] * lateral[0] + ball_vel[1] * lateral[1]
    v_up = ball_vel[2]
    v_horizontal = np.sqrt(v_forward**2 + v_lateral**2)
    v_total = np.linalg.norm(ball_vel)

    feats.append(v_forward)   # 6: toward hoop speed
    feats.append(v_lateral)   # 7: lateral speed -> left_right
    feats.append(v_up)        # 8: vertical speed -> angle

    # 9. Release angle (v_up / v_horizontal)
    #    DIRECTLY determines: angle
    release_angle = np.arctan2(v_up, v_horizontal + 1e-8)
    feats.append(release_angle)

    # 10. Release speed magnitude
    #    DIRECTLY determines: depth
    feats.append(v_total)

    # 11-13. Release position relative to hoop
    #    CAUSES: all targets (closer to hoop = different trajectory needed)
    ball_pos_m = ball[tr] * FEET_TO_M
    rel_to_hoop = ball_pos_m - HOOP_M
    feats.append(rel_to_hoop[0])  # forward distance to hoop
    feats.append(rel_to_hoop[1])  # ... (y component)
    feats.append(rel_to_hoop[2])  # height relative to rim

    # 14. Release height (absolute)
    feats.append(ball[tr, 2])

    # 15. Elbow angle at release
    #    CAUSES: angle (more extended elbow -> higher arc)
    elbow_angle_release = compute_joint_angle(shoulder[tr], elbow[tr], wrist[tr])
    feats.append(elbow_angle_release)

    # 16. Forearm elevation at release
    #    CAUSES: angle (higher forearm -> steeper release)
    forearm = wrist[tr] - elbow[tr]
    forearm_horiz = np.sqrt(forearm[0]**2 + forearm[1]**2)
    forearm_elev = np.arctan2(forearm[2], forearm_horiz + 1e-8)
    feats.append(forearm_elev)

    # 17. Wrist snap: fingertip speed minus wrist speed
    #    CAUSES: angle (more snap -> more backspin -> different arc)
    wrist_vel_mag = np.linalg.norm([
        safe_savgol(wrist[:, ax] * FEET_TO_M, 9, 3, deriv=1, delta=DT)[tr]
        for ax in range(3)
    ])
    ft_vel_mag = np.linalg.norm([
        safe_savgol(ft_center[:, ax] * FEET_TO_M, 9, 3, deriv=1, delta=DT)[tr]
        for ax in range(3)
    ])
    wrist_snap = ft_vel_mag - wrist_vel_mag
    feats.append(wrist_snap)

    # 18. Shoulder lateral tilt at release
    #    CAUSES: left_right
    shoulder_tilt = np.arctan2(
        shoulder[tr, 2] - l_shoulder[tr, 2],
        np.linalg.norm(shoulder[tr, :2] - l_shoulder[tr, :2]) + 1e-8
    )
    feats.append(shoulder_tilt)

    # 19. Release timing (frames from windup to release = acceleration duration)
    #    CAUSES: depth (longer acceleration -> more speed)
    feats.append(float(tr - tw))

    # ==========================================
    # AT FOLLOW-THROUGH (T_follow): Confirmation of release intent
    # ==========================================

    # 20. Wrist-to-hoop direction at follow-through
    #    CAUSES: left_right (where hand points -> where ball goes)
    wrist_to_hoop = HOOP_FEET - wrist[tf]
    wrist_to_hoop_lateral = np.dot(wrist_to_hoop[:2], lateral)
    feats.append(wrist_to_hoop_lateral)

    # 21. Arm full extension at follow-through
    #    CAUSES: angle (full extension -> committed follow-through)
    arm_extension = np.linalg.norm(wrist[tf] - shoulder[tf])
    feats.append(arm_extension)

    # 22. Wrist height at follow-through peak
    #    CAUSES: angle
    feats.append(wrist[tf, 2])

    # 23. Body alignment at follow-through (confirms lateral intent)
    #    CAUSES: left_right
    shoulder_vec_ft = shoulder[tf] - l_shoulder[tf]
    feats.append(np.dot(shoulder_vec_ft[:2], lateral))

    # 24. Follow-through timing
    feats.append(float(tf - tr))

    return np.array(feats, dtype=np.float32)


# ================================================================
# PIPELINE
# ================================================================

def ridge_loo_per_player(X, y, pids, bw_q=0.3, alpha=10.0, n_pls=5):
    """Per-player locally weighted Ridge with honest LOO (PLS refit per fold)."""
    n = len(y)
    preds = np.full(n, np.nan)
    unique_pids = np.unique(pids)

    for pid in unique_pids:
        mask = pids == pid
        idx = np.where(mask)[0]
        X_p, y_p = X[mask], y[mask]
        n_p = len(y_p)

        if n_p < 5:
            preds[idx] = np.mean(y_p)
            continue

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        n_comp = min(n_pls, n_p - 2, X_s.shape[1] - 1)
        if n_comp < 1:
            n_comp = 1

        for i in range(n_p):
            # Honest PLS: refit without point i
            X_loo = np.delete(X_s, i, axis=0)
            y_loo = np.delete(y_p, i)

            pls = PLSRegression(n_components=n_comp, scale=False)
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
            preds[idx[i]] = ridge.predict(X_comb[i:i+1])[0]

    return preds


def predict_test(X_train, y_train, pids_train, X_test, pids_test,
                 bw_q=0.3, alpha=10.0, n_pls=5):
    """Per-player Ridge predictions for test data."""
    n_test = len(X_test)
    preds = np.full(n_test, np.nan)
    unique_pids = np.unique(pids_train)

    for pid in unique_pids:
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

        n_comp = min(n_pls, n_p - 2, X_s.shape[1] - 1)
        if n_comp < 1:
            n_comp = 1

        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X_s, y_p)
        X_pls = pls.transform(X_s)
        X_te_pls = pls.transform(X_te_s)

        X_comb = np.hstack([X_s, X_pls])
        X_te_comb = np.hstack([X_te_s, X_te_pls])

        D_train = pairwise_distances(X_comb)
        bw = np.quantile(D_train[np.triu_indices(n_p, k=1)], bw_q)
        if bw < 1e-10:
            bw = 1.0

        for j, ti in enumerate(te_idx):
            d = pairwise_distances(X_te_comb[j:j+1], X_comb)[0]
            w = np.exp(-0.5 * (d / bw) ** 2)

            if w.sum() < 1e-10:
                preds[ti] = np.mean(y_p)
                continue

            sqrt_w = np.sqrt(w)
            X_w = X_comb * sqrt_w[:, None]
            y_w = y_p * sqrt_w

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_w, y_w)
            preds[ti] = ridge.predict(X_te_comb[j:j+1])[0]

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
    print("SPARSE PHYSICS PIPELINE")
    print("3 phases x ~8 features = 24 total, each physically motivated")
    print("=" * 70)

    print("\nLoading data...")
    X_3d_tr, kp_names, kp_index, train_df = load_data(DATA_DIR / "train.csv")
    X_3d_te, _, _, test_df = load_data(DATA_DIR / "test.csv")

    pids_tr = train_df["participant_id"].values
    pids_te = test_df["participant_id"].values

    # Scale targets
    y_scaled = {}
    for t in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{t}.pkl"
        if scaler_path.exists():
            s = joblib.load(scaler_path)
            y_scaled[t] = s.transform(train_df[t].values.reshape(-1, 1)).ravel()
        else:
            v = train_df[t].values
            y_scaled[t] = (v - v.min()) / (v.max() - v.min() + 1e-10)

    # Phase detection + feature extraction
    print("\nDetecting phases and extracting features (train)...")
    train_feats, train_phases = [], []
    for i in range(len(train_df)):
        phases = detect_phases(X_3d_tr[i], kp_index)
        train_phases.append(phases)
        feats = extract_sparse_features(X_3d_tr[i], kp_index, phases)
        train_feats.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(train_df)}...")

    X_train = np.array(train_feats)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nDetecting phases and extracting features (test)...")
    test_feats = []
    for i in range(len(test_df)):
        phases = detect_phases(X_3d_te[i], kp_index)
        feats = extract_sparse_features(X_3d_te[i], kp_index, phases)
        test_feats.append(feats)

    X_test = np.array(test_feats)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n{X_train.shape[1]} features extracted per shot")

    # Phase stats
    windups = [p['windup'] for p in train_phases]
    releases = [p['release'] for p in train_phases]
    follows = [p['follow'] for p in train_phases]
    forces = [p['peak_force_N'] for p in train_phases]
    knee_angles = [p['knee_angle_at_windup'] for p in train_phases]

    print(f"\nPhase detection stats:")
    print(f"  Windup:  {np.mean(windups):.0f} +/- {np.std(windups):.0f} (knee angle: {np.mean(knee_angles):.1f} +/- {np.std(knee_angles):.1f} deg)")
    print(f"  Release: {np.mean(releases):.0f} +/- {np.std(releases):.0f} (peak force: {np.mean(forces):.1f} +/- {np.std(forces):.1f} N)")
    print(f"  Follow:  {np.mean(follows):.0f} +/- {np.std(follows):.0f}")

    # Per-player phase stats
    for pid in sorted(np.unique(pids_tr)):
        mask = pids_tr == pid
        p_wind = [windups[i] for i in range(len(windups)) if mask[i]]
        p_rel = [releases[i] for i in range(len(releases)) if mask[i]]
        p_fol = [follows[i] for i in range(len(follows)) if mask[i]]
        print(f"  P{pid}: wind={np.mean(p_wind):.0f}+/-{np.std(p_wind):.0f}, "
              f"rel={np.mean(p_rel):.0f}+/-{np.std(p_rel):.0f}, "
              f"follow={np.mean(p_fol):.0f}+/-{np.std(p_fol):.0f}")

    # Run per-target
    all_loo, all_test = {}, {}
    for tname in TARGETS:
        print(f"\n{'='*50}")
        print(f"TARGET: {tname}")
        print(f"{'='*50}")

        y = y_scaled[tname]
        loo_preds = ridge_loo_per_player(X_train, y, pids_tr, n_pls=5)
        loo_mse = np.mean((loo_preds - y) ** 2)

        for pid in sorted(np.unique(pids_tr)):
            m = pids_tr == pid
            pmse = np.mean((loo_preds[m] - y[m]) ** 2)
            print(f"  P{pid}: LOO MSE = {pmse:.6f}")
        print(f"  GLOBAL honest LOO MSE = {loo_mse:.6f}")

        test_preds = predict_test(X_train, y, pids_tr, X_test, pids_te, n_pls=5)
        all_loo[tname] = loo_preds
        all_test[tname] = test_preds

    mean_loo = np.mean([np.mean((all_loo[t] - y_scaled[t]) ** 2) for t in TARGETS])
    print(f"\nMEAN LOO MSE: {mean_loo:.6f}")

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

    if base_path.exists():
        base = pd.read_csv(base_path)
        for pct in [2, 5, 10]:
            w = pct / 100.0
            bn = get_next_sub_num()
            blend = pd.DataFrame({"id": test_df["id"]})
            for t in TARGETS:
                col = f"scaled_{t}"
                blend[col] = (1 - w) * base[col] + w * np.clip(all_test[t], 0, 1)
            blend.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
            print(f"  {pct}% blend: submission_{bn}.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
