"""
Overfit Reduction: Three Proven Approaches

1. Nadaraya-Watson (NW): Replace Ridge with weighted mean of neighbor targets
   for angle. Eliminates regression coefficients that memorize noise.

2. Bagging: Bootstrap aggregate 50 models per prediction. Reduces variance.

3. PLS Leak Fix: Refit PLS excluding the held-out point in LOO. Gives honest
   LOO estimate for proper hyperparameter tuning.

Tests each approach independently and in combination.
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

TARGET_SCALERS = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}

PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}

TARGET_SETTINGS = {
    "angle": {"bw": 0.80, "n_frames": 3, "spacing": 5},
    "depth": {"bw": 0.55, "n_frames": 3, "spacing": 5},
    "left_right": {"bw": 0.30, "n_frames": 1, "spacing": 5},
}

PLAYER_OVERRIDES = {
    (1, "left_right"): {"bw": 0.15},
}


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split('_')[1]) for fp in existing
                    if fp.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


def compute_hoop_transform(ts_3d, kp_index):
    mid_hip_idx = kp_index.get('mid_hip', 0)
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
    return np.einsum('ij,fkj->fki', R, centered)


def detect_release_frame(ts_3d, kp_index):
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 120
    wrist_traj = ts_3d[:, rw_idx, :].copy()
    for ax in range(3):
        vals = wrist_traj[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 120
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, ax] = vals
    wrist_z_smooth = safe_savgol(wrist_traj[:, 2], 11, 3)
    wrist_peak = 80 + np.argmax(wrist_z_smooth[80:200])
    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = [ts_3d[:, kp_index[k], :] for k in ft_keys if k in kp_index]
    if ft_trajs:
        ft_center = np.nanmean(ft_trajs, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)
    vel = np.zeros_like(ball * FEET_TO_METERS)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_METERS, 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)
    s, e = max(80, wrist_peak - 40), min(wrist_peak + 5, 200)
    return int(np.clip(s + np.argmax(speed[s:e]), 80, 200))


def extract_kinetic_chain(ts_3d, kp_index):
    """Extract 40 kinetic chain features."""
    feats = []
    chain_joints = ['right_ankle', 'right_knee', 'right_hip',
                    'right_shoulder', 'right_elbow', 'right_wrist']
    joint_data = {}
    for jname in chain_joints:
        idx = kp_index.get(jname)
        if idx is None:
            return np.zeros(40, dtype=np.float32)
        data = ts_3d[:, idx, :].copy()
        for ax in range(3):
            vals = data[:, ax]
            bad = np.isnan(vals) | np.isinf(vals)
            if np.all(bad):
                data[:, ax] = 0.0
            elif np.any(bad):
                good = ~bad
                vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
                data[:, ax] = vals
        for ax in range(3):
            data[:, ax] = safe_savgol(data[:, ax], 7, 2)
        joint_data[jname] = data

    velocities, vel_mag, vert_vel = {}, {}, {}
    for jname, data in joint_data.items():
        vel = np.gradient(data, DT, axis=0)
        velocities[jname] = vel
        vel_mag[jname] = np.linalg.norm(vel, axis=1)
        vert_vel[jname] = vel[:, 2]

    wrist_speed = vel_mag['right_wrist']
    release_frame = int(np.clip(120 + np.argmax(wrist_speed[120:]), 120, 230))
    feats.append(release_frame)
    prop_start = max(0, release_frame - 60)
    prop_end = min(240, release_frame + 5)
    chain_order = ['right_knee', 'right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
    peak_times, peak_mags = {}, {}
    for jname in chain_order:
        window = vel_mag[jname][prop_start:prop_end]
        if len(window) == 0:
            peak_times[jname] = release_frame; peak_mags[jname] = 0.0
            feats.extend([0.0, 0.0, 0.0]); continue
        if jname == 'right_knee':
            v1 = joint_data['right_hip'] - joint_data['right_knee']
            v2 = joint_data['right_ankle'] - joint_data['right_knee']
            dot = np.sum(v1 * v2, axis=1)
            denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            denom[denom < 1e-10] = 1e-10
            knee_angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            window_ext = np.gradient(knee_angle, DT)[prop_start:prop_end]
            idx_max = np.argmax(window_ext)
            peak_val = window_ext[idx_max]; peak_frame = prop_start + idx_max
        elif jname == 'right_hip':
            window_vert = vert_vel['right_hip'][prop_start:prop_end]
            idx_max = np.argmax(window_vert)
            peak_val = window_vert[idx_max]; peak_frame = prop_start + idx_max
        else:
            idx_max = np.argmax(window)
            peak_val = window[idx_max]; peak_frame = prop_start + idx_max
        peak_times[jname] = peak_frame; peak_mags[jname] = peak_val
        feats.extend([float(peak_frame), float(peak_val), float((peak_frame - release_frame) * DT)])
    pairs = [('right_knee', 'right_hip'), ('right_hip', 'right_shoulder'),
             ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist')]
    for prev_j, curr_j in pairs:
        delta = peak_times[curr_j] - peak_times[prev_j]
        feats.extend([float(delta), float(delta * DT)])
        feats.append(float(peak_mags[curr_j] / peak_mags[prev_j]) if abs(peak_mags[prev_j]) > 1e-6 else 1.0)
    for jname in chain_order:
        feats.append(float(np.sum(vel_mag[jname][prop_start:release_frame] ** 2)))
    v_rel = velocities['right_wrist'][release_frame]
    feats.extend([float(v_rel[0]), float(v_rel[1]), float(v_rel[2]), float(np.linalg.norm(v_rel))])
    v_horiz = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
    feats.extend([float(np.degrees(np.arctan2(v_rel[2], max(v_horiz, 1e-8)))),
                  float(np.degrees(np.arctan2(v_rel[0], max(abs(v_rel[1]), 1e-8))))])
    jerk_mag = np.linalg.norm(np.gradient(np.gradient(velocities['right_wrist'], DT, axis=0), DT, axis=0), axis=1)
    feats.append(float(jerk_mag[release_frame]))
    return np.array(feats[:40], dtype=np.float32)


def extract_base_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """208 features: 198 hoop-relative + 10 joint angles."""
    f = int(np.clip(frame, 0, 239))
    feats = []
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder', 'right_hip', 'left_hip',
                  'mid_hip', 'right_knee', 'left_knee', 'neck', 'nose']
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6); continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            feats.append(np.gradient(ts_hr[:, idx, coord], DT)[f])
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 9); continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            feats.extend([np.nanmean(series), np.nanstd(series), np.nanmax(series) - np.nanmin(series)])
    rw, re, rs = kp_index.get('right_wrist'), kp_index.get('right_elbow'), kp_index.get('right_shoulder')
    if all(i is not None for i in [rw, re, rs]):
        feats.extend([ts_hr[f, rw, c] - ts_hr[f, rs, c] for c in range(3)])
        ua, fa = ts_3d[f, re] - ts_3d[f, rs], ts_3d[f, rw] - ts_3d[f, re]
        ua_n, fa_n = np.linalg.norm(ua), np.linalg.norm(fa)
        feats.append(np.degrees(np.arccos(np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1))) if ua_n > 1e-6 and fa_n > 1e-6 else 90.0)
        feats.extend([np.gradient(ts_hr[:, rw, c], DT)[f] for c in range(3)])
    else:
        feats.extend([0.0] * 7)
    rh, lh, ls = kp_index.get('right_hip'), kp_index.get('left_hip'), kp_index.get('left_shoulder')
    feats.append(ts_hr[f, rh, 1] - ts_hr[f, lh, 1] if rh is not None and lh is not None else 0.0)
    feats.append(ts_hr[f, rh, 0] - ts_hr[f, lh, 0] if rh is not None and lh is not None else 0.0)
    feats.append(ts_hr[f, rs, 1] - ts_hr[f, ls, 1] if rs is not None and ls is not None else 0.0)
    lw = kp_index.get('left_wrist')
    feats.append(ts_hr[f, lw, 1] - ts_hr[f, rw, 1] if lw is not None and rw is not None else 0.0)
    feats.append(release_frame)
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            feats.extend([np.nanmean(series[140:180]), np.nanmax(np.gradient(series, DT)[140:180])])
    else:
        feats.extend([0.0] * 6)
    # Joint angles (10)
    rw_i, re_i, rs_i = rw, re, rs
    rh_i, lh_i = kp_index.get('right_hip'), kp_index.get('left_hip')
    rk_i, lk_i = kp_index.get('right_knee'), kp_index.get('left_knee')
    ra_i, la_i = kp_index.get('right_ankle'), kp_index.get('left_ankle')
    neck_i, mh_i, ls_i = kp_index.get('neck'), kp_index.get('mid_hip'), kp_index.get('left_shoulder')
    feats.append(_angle_between(ts_3d[f, rs_i] - ts_3d[f, rh_i], ts_3d[f, re_i] - ts_3d[f, rs_i]) if all(x is not None for x in [rs_i, rh_i, re_i]) else 90.0)
    if neck_i is not None and mh_i is not None:
        trunk = ts_hr[f, neck_i] - ts_hr[f, mh_i]
        feats.extend([_angle_between(trunk, np.array([0, 0, 1], dtype=np.float32)),
                       np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8))])
    else:
        feats.extend([0.0, 0.0])
    feats.append(_angle_between(ts_3d[f, rh_i] - ts_3d[f, rk_i], ts_3d[f, ra_i] - ts_3d[f, rk_i]) if all(x is not None for x in [rk_i, rh_i, ra_i]) else 90.0)
    feats.append(_angle_between(ts_3d[f, lh_i] - ts_3d[f, lk_i], ts_3d[f, la_i] - ts_3d[f, lk_i]) if all(x is not None for x in [lk_i, lh_i, la_i]) else 90.0)
    feats.append(_angle_between(ts_3d[f, rw_i] - ts_3d[f, re_i], np.array([0, 0, 1], dtype=np.float32)) if re_i is not None and rw_i is not None else 90.0)
    feats.append(_angle_between(ts_hr[f, rw_i] - ts_hr[f, rs_i], np.array([1, 0, 0.5], dtype=np.float32)) if rs_i is not None and rw_i is not None else 90.0)
    feats.append(_angle_between(ts_hr[f, rs_i] - ts_hr[f, ls_i], np.array([0, 1, 0], dtype=np.float32)) if rs_i is not None and ls_i is not None else 90.0)
    if all(x is not None for x in [rh_i, lh_i, rs_i, ls_i]):
        hip_line = ts_hr[f, rh_i, :2] - ts_hr[f, lh_i, :2]
        shoulder_xy = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
        hn, sn = np.linalg.norm(hip_line), np.linalg.norm(shoulder_xy)
        feats.append(np.degrees(np.arccos(np.clip(np.dot(hip_line, shoulder_xy) / (hn * sn), -1, 1))) if hn > 1e-6 and sn > 1e-6 else 0.0)
    else:
        feats.append(0.0)
    feats.append(ts_hr[f, re_i, 2] - ts_hr[f, rs_i, 2] if re_i is not None and rs_i is not None else 0.0)
    return np.array(feats, dtype=np.float32)


def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)
    kp_names = [col[:-2] for col in keypoint_cols if col.endswith("_x")]
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        X_3d = np.zeros((n, 240, len(kp_names), 3), dtype=np.float32)
        X_raw = np.zeros((n, n_kp_cols * 240), dtype=np.float32)
        ids, pids, targets = [], [], []
        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr
            ids.append(row['id']); pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result
    return process(train_df, True), process(test_df, False)


def predict_ridge(X_tr_s, y_tr, weights, x_te, alpha=10.0):
    """Standard Ridge prediction."""
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_tr_s, y_tr, sample_weight=weights)
    return ridge.predict(x_te)[0]


def predict_nw(y_tr, weights):
    """Nadaraya-Watson: weighted mean of targets. No regression."""
    w_sum = np.sum(weights)
    if w_sum < 1e-10:
        return np.mean(y_tr)
    return np.sum(weights * y_tr) / w_sum


def predict_bagged_ridge(X_tr_s, y_tr, weights, x_te, alpha=10.0, n_bags=50, rng=None):
    """Bagged Ridge: bootstrap aggregate multiple Ridge models."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(y_tr)
    preds = []
    for _ in range(n_bags):
        boot_idx = rng.choice(n, size=n, replace=True)
        X_b = X_tr_s[boot_idx]
        y_b = y_tr[boot_idx]
        w_b = weights[boot_idx]
        if np.sum(w_b) < 1e-10:
            preds.append(np.mean(y_tr))
            continue
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_b, y_b, sample_weight=w_b)
        preds.append(ridge.predict(x_te)[0])
    return np.mean(preds)


def predict_bagged_nw(y_tr, weights, n_bags=50, rng=None):
    """Bagged NW: bootstrap aggregate weighted means."""
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(y_tr)
    preds = []
    for _ in range(n_bags):
        boot_idx = rng.choice(n, size=n, replace=True)
        y_b = y_tr[boot_idx]
        w_b = weights[boot_idx]
        w_sum = np.sum(w_b)
        if w_sum < 1e-10:
            preds.append(np.mean(y_tr))
        else:
            preds.append(np.sum(w_b * y_b) / w_sum)
    return np.mean(preds)


def run_target(train_data, test_data, y_train, target, mode="baseline",
               fix_pls_leak=False, n_bags=50):
    """
    Run one target with specified mode.
    mode: "baseline" | "nw" | "bagged_ridge" | "bagged_nw"
    fix_pls_leak: if True, refit PLS excluding held-out point in LOO
    """
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    unique_pids = sorted(np.unique(pids_train))
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}

    tidx = target_idx[target]
    y_raw = y_train[:, tidx]
    scaler = TARGET_SCALERS[target]
    y_sc = scaler.transform(y_raw.reshape(-1, 1)).ravel()

    settings = TARGET_SETTINGS[target]
    bw_default = settings["bw"]
    n_frames = settings["n_frames"]
    spacing = settings["spacing"]
    offsets = [0] if n_frames == 1 else [-spacing, 0, spacing]

    oof_preds = np.zeros(len(pids_train))
    test_preds = np.zeros(len(pids_test))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        n_tr = len(tr_idx)
        n_te = len(te_idx)

        override = PLAYER_OVERRIDES.get((pid, target), {})
        bw = override.get("bw", bw_default)
        center = PLAYER_FRAMES[target][pid]
        frames = [int(np.clip(center + off, 0, 239)) for off in offsets]

        # Pre-extract KC features
        kc_tr = np.array([extract_kinetic_chain(train_data['X_3d'][i], kp_index) for i in tr_idx], dtype=np.float32)
        kc_te = np.array([extract_kinetic_chain(test_data['X_3d'][i], kp_index) for i in te_idx], dtype=np.float32) if n_te > 0 else np.zeros((0, 40))
        kc_tr = np.nan_to_num(kc_tr, nan=0.0, posinf=0.0, neginf=0.0)
        kc_te = np.nan_to_num(kc_te, nan=0.0, posinf=0.0, neginf=0.0)

        # Pre-extract base features per frame
        base_feats_tr = {}
        base_feats_te = {}
        for frame in frames:
            btr = []
            for i in tr_idx:
                ts_3d = train_data['X_3d'][i]
                ts_hr = compute_hoop_transform(ts_3d, kp_index)
                rf = detect_release_frame(ts_3d, kp_index)
                btr.append(extract_base_features(ts_3d, ts_hr, kp_index, rf, frame))
            base_feats_tr[frame] = np.nan_to_num(np.array(btr), nan=0.0, posinf=0.0, neginf=0.0)

            bte = []
            for i in te_idx:
                ts_3d = test_data['X_3d'][i]
                ts_hr = compute_hoop_transform(ts_3d, kp_index)
                rf = detect_release_frame(ts_3d, kp_index)
                bte.append(extract_base_features(ts_3d, ts_hr, kp_index, rf, frame))
            base_feats_te[frame] = np.nan_to_num(np.array(bte), nan=0.0, posinf=0.0, neginf=0.0) if n_te > 0 else np.zeros((0, base_feats_tr[frame].shape[1]))

        raw_tr_player = train_data['X_raw'][tr_mask]
        raw_te_player = test_data['X_raw'][te_mask] if n_te > 0 else np.zeros((0, raw_tr_player.shape[1]))
        y_p_raw = y_raw[tr_mask]
        y_p_sc = y_sc[tr_mask]

        rng = np.random.RandomState(42 + pid)

        if not fix_pls_leak:
            # Standard: fit PLS on all player data (leaky for LOO, correct for test)
            pls_sc = StandardScaler()
            raw_s = pls_sc.fit_transform(raw_tr_player)
            raw_s_te = pls_sc.transform(raw_te_player) if n_te > 0 else np.zeros((0, raw_s.shape[1]))
            nc = max(3, min(10, n_tr - n_tr // 5 - 1))
            pls = PLSRegression(n_components=nc)
            pls.fit(raw_s, y_p_raw)
            pls_tr = pls.transform(raw_s)
            pls_te = pls.transform(raw_s_te) if n_te > 0 else np.zeros((0, nc))

            kc_sc = StandardScaler()
            kc_s = kc_sc.fit_transform(kc_tr)
            kc_s_te = kc_sc.transform(kc_te) if n_te > 0 else np.zeros((0, kc_s.shape[1]))
            kc_nc = max(2, min(3, n_tr - n_tr // 5 - 1))
            kc_pls = PLSRegression(n_components=kc_nc)
            kc_pls.fit(kc_s, y_p_raw)
            kc_pls_tr = kc_pls.transform(kc_s)
            kc_pls_te = kc_pls.transform(kc_s_te) if n_te > 0 else np.zeros((0, kc_nc))

        # --- LOO ---
        frame_oofs = []
        for frame in frames:
            oof = np.zeros(n_tr)
            for li in range(n_tr):
                if fix_pls_leak:
                    # Refit PLS excluding held-out point
                    loo_mask = np.ones(n_tr, dtype=bool)
                    loo_mask[li] = False

                    pls_sc_li = StandardScaler()
                    raw_s_li = pls_sc_li.fit_transform(raw_tr_player[loo_mask])
                    nc_li = max(3, min(10, (n_tr - 1) - (n_tr - 1) // 5 - 1))
                    pls_li = PLSRegression(n_components=nc_li)
                    pls_li.fit(raw_s_li, y_p_raw[loo_mask])
                    # Transform ALL points using LOO-fitted PLS
                    raw_s_all = pls_sc_li.transform(raw_tr_player)
                    pls_tr_li = pls_li.transform(raw_s_all)

                    kc_sc_li = StandardScaler()
                    kc_s_li = kc_sc_li.fit_transform(kc_tr[loo_mask])
                    kc_nc_li = max(2, min(3, (n_tr - 1) - (n_tr - 1) // 5 - 1))
                    kc_pls_li = PLSRegression(n_components=kc_nc_li)
                    kc_pls_li.fit(kc_s_li, y_p_raw[loo_mask])
                    kc_s_all = kc_sc_li.transform(kc_tr)
                    kc_pls_tr_li = kc_pls_li.transform(kc_s_all)

                    X_tr_aug = np.hstack([base_feats_tr[frame], pls_tr_li, kc_pls_tr_li])
                else:
                    X_tr_aug = np.hstack([base_feats_tr[frame], pls_tr, kc_pls_tr])

                sc = StandardScaler()
                X_s = sc.fit_transform(X_tr_aug)

                D = cdist(X_s, X_s, 'euclidean')
                all_dists = D[np.triu_indices(n_tr, k=1)]
                sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
                sigma = max(sigma, 1e-6)

                weights = np.exp(-D[li, :]**2 / (2 * sigma**2))
                weights[li] = 0

                if weights.sum() < 1e-10:
                    oof[li] = np.mean(y_p_sc)
                    continue

                tr_mask_loo = np.ones(n_tr, dtype=bool)
                tr_mask_loo[li] = False

                if mode == "nw":
                    oof[li] = predict_nw(y_p_sc, weights)
                elif mode == "bagged_ridge":
                    oof[li] = predict_bagged_ridge(X_s[tr_mask_loo], y_p_sc[tr_mask_loo],
                                                    weights[tr_mask_loo], X_s[li:li+1],
                                                    n_bags=n_bags, rng=rng)
                elif mode == "bagged_nw":
                    oof[li] = predict_bagged_nw(y_p_sc, weights, n_bags=n_bags, rng=rng)
                else:  # baseline
                    oof[li] = predict_ridge(X_s, y_p_sc, weights, X_s[li:li+1])

            frame_oofs.append(oof)
        oof_preds[tr_idx] = np.mean(frame_oofs, axis=0)

        # --- Test ---
        if n_te > 0:
            if fix_pls_leak:
                # For test, fit PLS on ALL training data (no leak)
                pls_sc_full = StandardScaler()
                raw_s_full = pls_sc_full.fit_transform(raw_tr_player)
                raw_s_te_full = pls_sc_full.transform(raw_te_player)
                nc_full = max(3, min(10, n_tr - n_tr // 5 - 1))
                pls_full = PLSRegression(n_components=nc_full)
                pls_full.fit(raw_s_full, y_p_raw)
                pls_tr_full = pls_full.transform(raw_s_full)
                pls_te_full = pls_full.transform(raw_s_te_full)

                kc_sc_full = StandardScaler()
                kc_s_full = kc_sc_full.fit_transform(kc_tr)
                kc_s_te_full = kc_sc_full.transform(kc_te)
                kc_nc_full = max(2, min(3, n_tr - n_tr // 5 - 1))
                kc_pls_full = PLSRegression(n_components=kc_nc_full)
                kc_pls_full.fit(kc_s_full, y_p_raw)
                kc_pls_tr_full = kc_pls_full.transform(kc_s_full)
                kc_pls_te_full = kc_pls_full.transform(kc_s_te_full)

            frame_tests = []
            for frame in frames:
                if fix_pls_leak:
                    X_tr_aug = np.hstack([base_feats_tr[frame], pls_tr_full, kc_pls_tr_full])
                    X_te_aug = np.hstack([base_feats_te[frame], pls_te_full, kc_pls_te_full])
                else:
                    X_tr_aug = np.hstack([base_feats_tr[frame], pls_tr, kc_pls_tr])
                    X_te_aug = np.hstack([base_feats_te[frame], pls_te, kc_pls_te])

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr_aug)
                X_te_s = sc.transform(X_te_aug)

                D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
                all_dists = D_tr[np.triu_indices(n_tr, k=1)]
                sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
                sigma = max(sigma, 1e-6)

                D_te = cdist(X_te_s, X_tr_s, 'euclidean')
                tpred = np.zeros(n_te)
                for j in range(n_te):
                    weights = np.exp(-D_te[j, :]**2 / (2 * sigma**2))
                    if weights.sum() < 1e-10:
                        tpred[j] = np.mean(y_p_sc)
                        continue
                    if mode == "nw":
                        tpred[j] = predict_nw(y_p_sc, weights)
                    elif mode == "bagged_ridge":
                        tpred[j] = predict_bagged_ridge(X_tr_s, y_p_sc, weights,
                                                         X_te_s[j:j+1], n_bags=n_bags, rng=rng)
                    elif mode == "bagged_nw":
                        tpred[j] = predict_bagged_nw(y_p_sc, weights, n_bags=n_bags, rng=rng)
                    else:
                        tpred[j] = predict_ridge(X_tr_s, y_p_sc, weights, X_te_s[j:j+1])
                frame_tests.append(tpred)
            test_preds[te_idx] = np.mean(frame_tests, axis=0)

    loo_mse = np.mean((oof_preds - y_sc) ** 2)
    return oof_preds, test_preds, loo_mse


def main():
    t0 = time.time()
    print("=" * 70)
    print("OVERFIT REDUCTION: THREE APPROACHES")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']

    # =========================================================================
    # Run all configurations
    # =========================================================================
    configs = [
        # (label, angle_mode, depth_mode, lr_mode, fix_pls)
        ("baseline",           "baseline",     "baseline",     "baseline",     False),
        ("NW angle only",      "nw",           "baseline",     "baseline",     False),
        ("NW all targets",     "nw",           "nw",           "nw",           False),
        ("bagged Ridge angle", "bagged_ridge", "baseline",     "baseline",     False),
        ("bagged Ridge all",   "bagged_ridge", "bagged_ridge", "bagged_ridge", False),
        ("bagged NW angle",    "bagged_nw",    "baseline",     "baseline",     False),
        ("PLS fix only",       "baseline",     "baseline",     "baseline",     True),
        ("NW angle + PLS fix", "nw",           "baseline",     "baseline",     True),
        ("bagged NW angle + PLS fix", "bagged_nw", "baseline", "baseline",     True),
    ]

    results = {}
    all_test_preds = {}

    for label, a_mode, d_mode, lr_mode, fix_pls in configs:
        t1 = time.time()
        print(f"\n{'='*70}")
        print(f"CONFIG: {label}")
        print(f"  angle={a_mode}, depth={d_mode}, LR={lr_mode}, fix_pls={fix_pls}")
        print(f"{'='*70}")

        loos = {}
        tests = {}
        for target, mode in [("angle", a_mode), ("depth", d_mode), ("left_right", lr_mode)]:
            oof, test, mse = run_target(train_data, test_data, y_train, target,
                                        mode=mode, fix_pls_leak=fix_pls, n_bags=50)
            loos[target] = mse
            tests[target] = test
            print(f"  {target}: LOO={mse:.6f} (mode={mode})")

        mean_loo = np.mean(list(loos.values()))
        results[label] = {"loos": loos, "mean_loo": mean_loo}
        all_test_preds[label] = tests
        print(f"  mean LOO: {mean_loo:.6f} [{time.time()-t1:.1f}s]")

    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    baseline_mean = results["baseline"]["mean_loo"]
    print(f"\n  {'Config':<30s} {'angle':>10s} {'depth':>10s} {'LR':>10s} {'mean':>10s} {'%chg':>8s}")
    print(f"  {'-'*78}")
    for label in results:
        r = results[label]
        pct = (r["mean_loo"] - baseline_mean) / baseline_mean * 100
        print(f"  {label:<30s} {r['loos']['angle']:>10.6f} {r['loos']['depth']:>10.6f} "
              f"{r['loos']['left_right']:>10.6f} {r['mean_loo']:>10.6f} {pct:>+7.2f}%")

    # =========================================================================
    # Generate submissions for promising configs
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    sub_2503 = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")

    for label in all_test_preds:
        if label == "baseline":
            continue
        tests = all_test_preds[label]
        r = results[label]

        # Standalone
        sn = get_next_submission_number()
        sub = pd.DataFrame({'id': test_data['ids']})
        for t in TARGETS:
            sub[f"scaled_{t}"] = tests[t]
        sub.to_csv(SUBMISSION_DIR / f"submission_{sn}.csv", index=False)
        print(f"Sub {sn}: {label} standalone (LOO {r['mean_loo']:.6f})")

        # 10% blend with Sub 2503
        sn_b = get_next_submission_number()
        bl = pd.DataFrame({'id': test_data['ids']})
        for t in TARGETS:
            col = f"scaled_{t}"
            bl[col] = 0.10 * tests[t] + 0.90 * sub_2503[col].values
        bl.to_csv(SUBMISSION_DIR / f"submission_{sn_b}.csv", index=False)
        print(f"Sub {sn_b}: 10% {label} + 90% Sub 2503")

    total = time.time() - t0
    print(f"\nTotal runtime: {total:.1f}s")


if __name__ == "__main__":
    main()
