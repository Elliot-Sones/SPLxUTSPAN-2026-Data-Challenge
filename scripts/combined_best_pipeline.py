"""
Combined Best Pipeline

Combines ALL proven improvements into one pipeline:
1. Per-player optimal frames
2. Per-target optimal bandwidth (angle=0.80, depth=0.55, LR=0.30)
3. Joint angle features (10)
4. KC-PLS features (3 PLS components from 40 kinetic chain features)
5. Multi-frame ensemble (per-target: angle=3f, depth=3f, LR=1f)
6. Cross-player transfer for P5 depth (all players, 3x upweight P5)
7. Narrow bandwidth (0.15) for P1 left_right

This is the "kitchen sink" model combining every signal we've found.
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

PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}

# Per-target optimal settings
TARGET_SETTINGS = {
    "angle": {"bw": 0.80, "n_frames": 3, "spacing": 5},
    "depth": {"bw": 0.55, "n_frames": 3, "spacing": 5},
    "left_right": {"bw": 0.30, "n_frames": 1, "spacing": 5},
}

# Player-specific overrides from player-targeted optimization
PLAYER_OVERRIDES = {
    (1, "left_right"): {"bw": 0.15},   # narrow BW for P1 LR (-22% LOO)
    (5, "depth"): {"cross_player": True, "bw": 0.80},  # cross-player for P5 depth (-11.57% LOO)
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


def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)
    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        X_raw = np.zeros((n, n_kp_cols * 240), dtype=np.float32)
        ids, pids, targets = [], [], []
        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result
    return process(train_df, True), process(test_df, False)


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

    velocities = {}
    vel_mag = {}
    vert_vel = {}
    for jname, data in joint_data.items():
        vel = np.gradient(data, DT, axis=0)
        velocities[jname] = vel
        vel_mag[jname] = np.linalg.norm(vel, axis=1)
        vert_vel[jname] = vel[:, 2]

    wrist_speed = vel_mag['right_wrist']
    release_frame = 120 + np.argmax(wrist_speed[120:])
    release_frame = int(np.clip(release_frame, 120, 230))
    feats.append(release_frame)

    prop_start = max(0, release_frame - 60)
    prop_end = min(240, release_frame + 5)

    chain_order = ['right_knee', 'right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
    peak_times = {}
    peak_mags = {}

    for jname in chain_order:
        window = vel_mag[jname][prop_start:prop_end]
        if len(window) == 0:
            peak_times[jname] = release_frame
            peak_mags[jname] = 0.0
            feats.extend([0.0, 0.0, 0.0])
            continue

        if jname == 'right_knee':
            hip_pos = joint_data['right_hip']
            knee_pos = joint_data['right_knee']
            ankle_pos = joint_data['right_ankle']
            v1 = hip_pos - knee_pos
            v2 = ankle_pos - knee_pos
            dot = np.sum(v1 * v2, axis=1)
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            denom = n1 * n2
            denom[denom < 1e-10] = 1e-10
            knee_angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            ext_rate = np.gradient(knee_angle, DT)
            window_ext = ext_rate[prop_start:prop_end]
            idx_max = np.argmax(window_ext)
            peak_val = window_ext[idx_max]
            peak_frame = prop_start + idx_max
        elif jname == 'right_hip':
            window_vert = vert_vel['right_hip'][prop_start:prop_end]
            idx_max = np.argmax(window_vert)
            peak_val = window_vert[idx_max]
            peak_frame = prop_start + idx_max
        else:
            idx_max = np.argmax(window)
            peak_val = window[idx_max]
            peak_frame = prop_start + idx_max

        peak_times[jname] = peak_frame
        peak_mags[jname] = peak_val
        feats.append(float(peak_frame))
        feats.append(float(peak_val))
        feats.append(float((peak_frame - release_frame) * DT))

    pairs = [('right_knee', 'right_hip'), ('right_hip', 'right_shoulder'),
             ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist')]
    for prev_j, curr_j in pairs:
        delta = peak_times[curr_j] - peak_times[prev_j]
        feats.append(float(delta))
        feats.append(float(delta * DT))
        if abs(peak_mags[prev_j]) > 1e-6:
            feats.append(float(peak_mags[curr_j] / peak_mags[prev_j]))
        else:
            feats.append(1.0)

    for jname in chain_order:
        energy = np.sum(vel_mag[jname][prop_start:release_frame] ** 2)
        feats.append(float(energy))

    v_rel = velocities['right_wrist'][release_frame]
    feats.append(float(v_rel[0]))
    feats.append(float(v_rel[1]))
    feats.append(float(v_rel[2]))
    feats.append(float(np.linalg.norm(v_rel)))
    v_horiz = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
    feats.append(float(np.degrees(np.arctan2(v_rel[2], max(v_horiz, 1e-8)))))
    feats.append(float(np.degrees(np.arctan2(v_rel[0], max(abs(v_rel[1]), 1e-8)))))

    acc = np.gradient(velocities['right_wrist'], DT, axis=0)
    jerk = np.gradient(acc, DT, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    feats.append(float(jerk_mag[release_frame]))

    return np.array(feats[:40], dtype=np.float32)


def extract_base_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """208 features: 198 hoop-relative + 10 joint angles."""
    f = int(np.clip(frame, 0, 239))
    feats = []
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 9)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            feats.append(np.nanmean(series))
            feats.append(np.nanstd(series))
            feats.append(np.nanmax(series) - np.nanmin(series))
    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')
    if all(i is not None for i in [rw, re, rs]):
        feats.append(ts_hr[f, rw, 0] - ts_hr[f, rs, 0])
        feats.append(ts_hr[f, rw, 1] - ts_hr[f, rs, 1])
        feats.append(ts_hr[f, rw, 2] - ts_hr[f, rs, 2])
        ua = ts_3d[f, re] - ts_3d[f, rs]
        fa = ts_3d[f, rw] - ts_3d[f, re]
        ua_n, fa_n = np.linalg.norm(ua), np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            feats.append(np.degrees(np.arccos(np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1))))
        else:
            feats.append(90.0)
        for coord in range(3):
            vel = np.gradient(ts_hr[:, rw, coord], DT)
            feats.append(vel[f])
    else:
        feats.extend([0.0] * 7)
    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    ls = kp_index.get('left_shoulder')
    if rh is not None and lh is not None:
        feats.append(ts_hr[f, rh, 1] - ts_hr[f, lh, 1])
        feats.append(ts_hr[f, rh, 0] - ts_hr[f, lh, 0])
    else:
        feats.extend([0.0, 0.0])
    if rs is not None and ls is not None:
        feats.append(ts_hr[f, rs, 1] - ts_hr[f, ls, 1])
    else:
        feats.append(0.0)
    lw = kp_index.get('left_wrist')
    if lw is not None and rw is not None:
        feats.append(ts_hr[f, lw, 1] - ts_hr[f, rw, 1])
    else:
        feats.append(0.0)
    feats.append(release_frame)
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats.append(np.nanmean(series[140:180]))
            feats.append(np.nanmax(vel[140:180]))
    else:
        feats.extend([0.0] * 6)

    # Joint angles (10 features)
    rw_i, re_i, rs_i = kp_index.get('right_wrist'), kp_index.get('right_elbow'), kp_index.get('right_shoulder')
    rh_i, lh_i = kp_index.get('right_hip'), kp_index.get('left_hip')
    rk_i, lk_i = kp_index.get('right_knee'), kp_index.get('left_knee')
    ra_i, la_i = kp_index.get('right_ankle'), kp_index.get('left_ankle')
    neck_i, mh_i, ls_i = kp_index.get('neck'), kp_index.get('mid_hip'), kp_index.get('left_shoulder')
    if all(x is not None for x in [rs_i, rh_i, re_i]):
        feats.append(_angle_between(ts_3d[f, rs_i] - ts_3d[f, rh_i], ts_3d[f, re_i] - ts_3d[f, rs_i]))
    else:
        feats.append(90.0)
    if neck_i is not None and mh_i is not None:
        trunk = ts_hr[f, neck_i] - ts_hr[f, mh_i]
        feats.append(_angle_between(trunk, np.array([0, 0, 1], dtype=np.float32)))
        feats.append(np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8)))
    else:
        feats.extend([0.0, 0.0])
    if all(x is not None for x in [rk_i, rh_i, ra_i]):
        feats.append(_angle_between(ts_3d[f, rh_i] - ts_3d[f, rk_i], ts_3d[f, ra_i] - ts_3d[f, rk_i]))
    else:
        feats.append(90.0)
    if all(x is not None for x in [lk_i, lh_i, la_i]):
        feats.append(_angle_between(ts_3d[f, lh_i] - ts_3d[f, lk_i], ts_3d[f, la_i] - ts_3d[f, lk_i]))
    else:
        feats.append(90.0)
    if re_i is not None and rw_i is not None:
        feats.append(_angle_between(ts_3d[f, rw_i] - ts_3d[f, re_i], np.array([0, 0, 1], dtype=np.float32)))
    else:
        feats.append(90.0)
    if rs_i is not None and rw_i is not None:
        feats.append(_angle_between(ts_hr[f, rw_i] - ts_hr[f, rs_i], np.array([1, 0, 0.5], dtype=np.float32)))
    else:
        feats.append(90.0)
    if rs_i is not None and ls_i is not None:
        feats.append(_angle_between(ts_hr[f, rs_i] - ts_hr[f, ls_i], np.array([0, 1, 0], dtype=np.float32)))
    else:
        feats.append(90.0)
    if all(x is not None for x in [rh_i, lh_i, rs_i, ls_i]):
        hip_line = ts_hr[f, rh_i, :2] - ts_hr[f, lh_i, :2]
        shoulder_xy = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
        hn, sn = np.linalg.norm(hip_line), np.linalg.norm(shoulder_xy)
        if hn > 1e-6 and sn > 1e-6:
            feats.append(np.degrees(np.arccos(np.clip(np.dot(hip_line, shoulder_xy) / (hn * sn), -1, 1))))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)
    if re_i is not None and rs_i is not None:
        feats.append(ts_hr[f, re_i, 2] - ts_hr[f, rs_i, 2])
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def main():
    t0 = time.time()
    print("=" * 70)
    print("COMBINED BEST PIPELINE")
    print("All proven improvements in one model")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    unique_pids = sorted(np.unique(pids_train))

    # Pre-extract KC features for all samples
    print("Extracting KC features...")
    kc_train = np.array([extract_kinetic_chain(train_data['X_3d'][i], kp_index)
                         for i in range(len(pids_train))], dtype=np.float32)
    kc_test = np.array([extract_kinetic_chain(test_data['X_3d'][i], kp_index)
                        for i in range(len(pids_test))], dtype=np.float32)
    kc_train = np.nan_to_num(kc_train, nan=0.0, posinf=0.0, neginf=0.0)
    kc_test = np.nan_to_num(kc_test, nan=0.0, posinf=0.0, neginf=0.0)

    all_oof = {}
    all_test = {}
    all_loo = {}

    for target in TARGETS:
        t1 = time.time()
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        settings = TARGET_SETTINGS[target]
        bw_default = settings["bw"]
        n_frames = settings["n_frames"]
        spacing = settings["spacing"]

        if n_frames == 1:
            offsets = [0]
        elif n_frames == 3:
            offsets = [-spacing, 0, spacing]

        oof_preds = np.zeros(len(pids_train))
        test_preds = np.zeros(len(pids_test))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]
            n_tr = len(tr_idx)
            n_te = len(te_idx)

            # Check for player-specific overrides
            override = PLAYER_OVERRIDES.get((pid, target), {})
            bw = override.get("bw", bw_default)
            cross_player = override.get("cross_player", False)

            center = PLAYER_FRAMES[target][pid]
            frames = [int(np.clip(center + off, 0, 239)) for off in offsets]

            # Standard PLS from raw timeseries (always per-player)
            pls_sc = StandardScaler()
            raw_tr_player = pls_sc.fit_transform(train_data['X_raw'][tr_mask])
            raw_te_player = pls_sc.transform(test_data['X_raw'][te_mask]) if n_te > 0 else np.zeros((0, raw_tr_player.shape[1]))
            nc = min(10, n_tr - n_tr // 5 - 1)
            nc = max(3, nc)
            pls = PLSRegression(n_components=nc)
            pls.fit(raw_tr_player, y_raw[tr_mask])
            pls_tr = pls.transform(raw_tr_player)
            pls_te = pls.transform(raw_te_player) if n_te > 0 else np.zeros((0, nc))

            # KC PLS (3 components)
            kc_sc = StandardScaler()
            kc_tr_player = kc_sc.fit_transform(kc_train[tr_mask])
            kc_te_player = kc_sc.transform(kc_test[te_mask]) if n_te > 0 else np.zeros((0, kc_tr_player.shape[1]))
            kc_nc = min(3, n_tr - n_tr // 5 - 1)
            kc_nc = max(2, kc_nc)
            kc_pls = PLSRegression(n_components=kc_nc)
            kc_pls.fit(kc_tr_player, y_raw[tr_mask])
            kc_pls_tr = kc_pls.transform(kc_tr_player)
            kc_pls_te = kc_pls.transform(kc_te_player) if n_te > 0 else np.zeros((0, kc_nc))

            if cross_player:
                # Use ALL training data, with 3x upweight for this player
                all_tr_idx = np.arange(len(pids_train))
                is_target_player = pids_train == pid

                # PLS for all players
                pls_sc_all = StandardScaler()
                raw_tr_all = pls_sc_all.fit_transform(train_data['X_raw'])
                nc_all = min(10, len(pids_train) - len(pids_train) // 5 - 1)
                pls_all = PLSRegression(n_components=nc_all)
                pls_all.fit(raw_tr_all, y_raw)
                pls_tr_all = pls_all.transform(raw_tr_all)
                pls_te_all = pls_all.transform(pls_sc_all.transform(test_data['X_raw'][te_mask])) if n_te > 0 else np.zeros((0, nc_all))

                # KC PLS for all
                kc_sc_all = StandardScaler()
                kc_tr_all = kc_sc_all.fit_transform(kc_train)
                kc_pls_all = PLSRegression(n_components=3)
                kc_pls_all.fit(kc_tr_all, y_raw)
                kc_pls_tr_all = kc_pls_all.transform(kc_tr_all)
                kc_pls_te_all = kc_pls_all.transform(kc_sc_all.transform(kc_test[te_mask])) if n_te > 0 else np.zeros((0, 3))

            frame_oofs, frame_tests = [], []
            for frame in frames:
                if cross_player:
                    # Extract features for ALL training samples
                    all_feats_tr = []
                    for i in all_tr_idx:
                        ts_3d = train_data['X_3d'][i]
                        ts_hr = compute_hoop_transform(ts_3d, kp_index)
                        rf = detect_release_frame(ts_3d, kp_index)
                        base = extract_base_features(ts_3d, ts_hr, kp_index, rf, frame)
                        all_feats_tr.append(base)
                    X_tr = np.nan_to_num(np.array(all_feats_tr), nan=0.0, posinf=0.0, neginf=0.0)
                    X_tr_aug = np.hstack([X_tr, pls_tr_all, kc_pls_tr_all])
                    y_tr_sc = y_sc  # all players
                else:
                    all_feats_tr = []
                    for i in tr_idx:
                        ts_3d = train_data['X_3d'][i]
                        ts_hr = compute_hoop_transform(ts_3d, kp_index)
                        rf = detect_release_frame(ts_3d, kp_index)
                        base = extract_base_features(ts_3d, ts_hr, kp_index, rf, frame)
                        all_feats_tr.append(base)
                    X_tr = np.nan_to_num(np.array(all_feats_tr), nan=0.0, posinf=0.0, neginf=0.0)
                    X_tr_aug = np.hstack([X_tr, pls_tr, kc_pls_tr])
                    y_tr_sc = y_sc[tr_mask]

                # Test features
                all_feats_te = []
                for i in te_idx:
                    ts_3d = test_data['X_3d'][i]
                    ts_hr = compute_hoop_transform(ts_3d, kp_index)
                    rf = detect_release_frame(ts_3d, kp_index)
                    base = extract_base_features(ts_3d, ts_hr, kp_index, rf, frame)
                    all_feats_te.append(base)
                X_te = np.nan_to_num(np.array(all_feats_te), nan=0.0, posinf=0.0, neginf=0.0) if n_te > 0 else np.zeros((0, X_tr.shape[1]))

                if cross_player:
                    X_te_aug = np.hstack([X_te, pls_te_all, kc_pls_te_all]) if n_te > 0 else np.zeros((0, X_tr_aug.shape[1]))
                else:
                    X_te_aug = np.hstack([X_te, pls_te, kc_pls_te]) if n_te > 0 else np.zeros((0, X_tr_aug.shape[1]))

                n_all_tr = len(X_tr_aug)
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr_aug)
                X_te_s = sc.transform(X_te_aug) if n_te > 0 else np.zeros((0, X_tr_aug.shape[1]))

                D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
                all_dists = D_tr[np.triu_indices(n_all_tr, k=1)]
                sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
                sigma = max(sigma, 1e-6)

                if cross_player:
                    target_in_all = np.where(is_target_player)[0]
                    oof = np.zeros(n_tr)
                    for li, gi in enumerate(target_in_all):
                        weights = np.exp(-D_tr[gi, :]**2 / (2 * sigma**2))
                        weights[gi] = 0
                        weights[is_target_player] *= 3.0
                        weights[gi] = 0
                        if weights.sum() < 1e-10:
                            oof[li] = np.mean(y_tr_sc[target_in_all])
                            continue
                        ridge = Ridge(alpha=10.0)
                        ridge.fit(X_tr_s, y_tr_sc, sample_weight=weights)
                        oof[li] = ridge.predict(X_tr_s[gi:gi+1])[0]

                    tpred = np.zeros(n_te)
                    if n_te > 0:
                        D_te = cdist(X_te_s, X_tr_s, 'euclidean')
                        for j in range(n_te):
                            weights = np.exp(-D_te[j, :]**2 / (2 * sigma**2))
                            weights[is_target_player] *= 3.0
                            if weights.sum() < 1e-10:
                                tpred[j] = np.mean(y_tr_sc[target_in_all])
                                continue
                            ridge = Ridge(alpha=10.0)
                            ridge.fit(X_tr_s, y_tr_sc, sample_weight=weights)
                            tpred[j] = ridge.predict(X_te_s[j:j+1])[0]
                else:
                    oof = np.zeros(n_tr)
                    for i in range(n_tr):
                        weights = np.exp(-D_tr[i, :]**2 / (2 * sigma**2))
                        weights[i] = 0
                        if weights.sum() < 1e-10:
                            oof[i] = np.mean(y_tr_sc)
                            continue
                        ridge = Ridge(alpha=10.0)
                        ridge.fit(X_tr_s, y_tr_sc, sample_weight=weights)
                        oof[i] = ridge.predict(X_tr_s[i:i+1])[0]

                    tpred = np.zeros(n_te)
                    if n_te > 0:
                        D_te = cdist(X_te_s, X_tr_s, 'euclidean')
                        for j in range(n_te):
                            weights = np.exp(-D_te[j, :]**2 / (2 * sigma**2))
                            if weights.sum() < 1e-10:
                                tpred[j] = np.mean(y_tr_sc)
                                continue
                            ridge = Ridge(alpha=10.0)
                            ridge.fit(X_tr_s, y_tr_sc, sample_weight=weights)
                            tpred[j] = ridge.predict(X_te_s[j:j+1])[0]

                frame_oofs.append(oof)
                frame_tests.append(tpred)

            avg_oof = np.mean(frame_oofs, axis=0)
            avg_test = np.mean(frame_tests, axis=0)
            oof_preds[tr_idx] = avg_oof
            if n_te > 0:
                test_preds[te_idx] = avg_test

        loo = np.mean((oof_preds - y_sc)**2)
        all_oof[target] = oof_preds
        all_test[target] = test_preds
        all_loo[target] = loo

        # Per-player breakdown
        print(f"\n{target}: LOO = {loo:.6f} [{time.time()-t1:.1f}s]")
        for pid in unique_pids:
            mask = pids_train == pid
            p_loo = np.mean((oof_preds[mask] - y_sc[mask])**2)
            override = PLAYER_OVERRIDES.get((pid, target), {})
            tag = ""
            if override:
                tag = f" [override: {override}]"
            print(f"  P{pid}: {p_loo:.6f}{tag}")

    mean_loo = np.mean([all_loo[t] for t in TARGETS])
    print(f"\nMean LOO: {mean_loo:.6f}")

    # Generate submissions
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Standalone
    sn = get_next_submission_number()
    sp = SUBMISSION_DIR / f"submission_{sn}.csv"
    sub = pd.DataFrame({'id': test_data['ids']})
    for t in TARGETS:
        sub[f"scaled_{t}"] = all_test[t]
    sub.to_csv(sp, index=False)
    print(f"Sub {sn}: combined best standalone (LOO {mean_loo:.6f})")

    # Blends with Sub 2429
    base = pd.read_csv(SUBMISSION_DIR / "submission_2429.csv").set_index('id')
    for w in [0.10, 0.20, 0.30, 0.50]:
        sn_b = get_next_submission_number()
        sp_b = SUBMISSION_DIR / f"submission_{sn_b}.csv"
        bl = pd.DataFrame({'id': test_data['ids']})
        for t in TARGETS:
            col = f"scaled_{t}"
            vals = []
            for i, tid in enumerate(test_data['ids']):
                vals.append(w * all_test[t][i] + (1-w) * base.loc[tid, col])
            bl[col] = vals
        bl.to_csv(sp_b, index=False)
        print(f"Sub {sn_b}: {int(w*100)}% combined + {int((1-w)*100)}% Sub 2429")

    total = time.time() - t0
    print(f"\nTotal runtime: {total:.1f}s")


if __name__ == "__main__":
    main()
