"""
P4 Angle Fix - Targeted improvement for Player 4 angle predictions.

P4 has the highest angle variance (std=2.68 raw, 0.0895 scaled) among typical players,
but calibration slope is only 0.65 (meaning predictions capture only 65% of true variation).

Strategies:
1. Hyperparameter sweep: bandwidth, alpha, PLS components
2. Feature engineering: P4-specific features, temporal dynamics, flick features
3. Multi-frame ensemble with P4-specific optimal frames
4. Cross-player transfer (use all players' data with P4 upweighting)
5. Different model families: k-NN weighted average, RandomForest, gradient boosting
6. Honest LOO evaluation (refit PLS per fold to avoid leakage)

Goal: Generate improved P4 angle predictions and splice into Sub 3411.
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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048


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
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index,
                  'keypoint_cols': keypoint_cols}
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


def extract_rich_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract comprehensive features including P4-targeted ones.

    Returns ~280+ features: base hoop-relative + joint angles + temporal dynamics +
    finger/hand features + flick window features.
    """
    f = int(np.clip(frame, 0, 239))
    feats = []

    # ---- BASE FEATURES (198) ----
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

    # ---- JOINT ANGLES (10) ----
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

    # ---- FLICK WINDOW FEATURES (frames 135-165, crucial for angle) ----
    flick_start, flick_end = 135, 165
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            acc = np.gradient(vel, DT)
            # Position stats in flick window
            feats.append(np.nanmean(series[flick_start:flick_end]))
            feats.append(np.nanstd(series[flick_start:flick_end]))
            # Velocity stats
            feats.append(np.nanmean(vel[flick_start:flick_end]))
            feats.append(np.nanmax(vel[flick_start:flick_end]))
            feats.append(np.nanmin(vel[flick_start:flick_end]))
            # Acceleration stats
            feats.append(np.nanmean(acc[flick_start:flick_end]))
            feats.append(np.nanmax(acc[flick_start:flick_end]))
    else:
        feats.extend([0.0] * 21)

    # Elbow flick features
    if re is not None:
        for coord in range(3):
            series = ts_hr[:, re, coord]
            vel = np.gradient(series, DT)
            feats.append(np.nanmean(vel[flick_start:flick_end]))
            feats.append(np.nanmax(vel[flick_start:flick_end]))
        # Elbow angle change during flick
        if rs is not None and rw is not None:
            angles_flick = []
            for ff in range(flick_start, flick_end):
                ua = ts_3d[ff, re] - ts_3d[ff, rs]
                fa = ts_3d[ff, rw] - ts_3d[ff, re]
                angles_flick.append(_angle_between(-ua, fa))
            angles_flick = np.array(angles_flick)
            feats.append(angles_flick[-1] - angles_flick[0])  # total change
            feats.append(np.nanstd(angles_flick))  # variability
            feats.append(np.nanmax(np.gradient(angles_flick)))  # peak rate
    else:
        feats.extend([0.0] * 9)

    # ---- FINGERTIP FEATURES (hand shape at release) ----
    finger_tips = ['right_second_finger_distal', 'right_third_finger_distal',
                   'right_fourth_finger_distal', 'right_fifth_finger_distal',
                   'right_thumb_distal']
    finger_proxs = ['right_second_finger_proximal', 'right_third_finger_proximal',
                    'right_fourth_finger_proximal', 'right_fifth_finger_proximal',
                    'right_thumb_proximal']

    # Fingertip positions and spread at release
    tip_positions = []
    for fname in finger_tips:
        fi = kp_index.get(fname)
        if fi is not None:
            tip_positions.append(ts_hr[f, fi, :])
        else:
            tip_positions.append(np.zeros(3))
    tip_positions = np.array(tip_positions)

    # Finger spread (pairwise distances)
    for i in range(len(tip_positions)):
        for j in range(i+1, len(tip_positions)):
            feats.append(np.linalg.norm(tip_positions[i] - tip_positions[j]))

    # Finger curl (distal - proximal distance)
    for fname_t, fname_p in zip(finger_tips, finger_proxs):
        fi_t, fi_p = kp_index.get(fname_t), kp_index.get(fname_p)
        if fi_t is not None and fi_p is not None:
            curl = np.linalg.norm(ts_hr[f, fi_t] - ts_hr[f, fi_p])
            feats.append(curl)
        else:
            feats.append(0.0)

    # Wrist flexion (angle between forearm and hand direction)
    if rw is not None and re is not None:
        mid_finger_idx = kp_index.get('right_third_finger_distal')
        if mid_finger_idx is not None:
            forearm = ts_3d[f, rw] - ts_3d[f, re]
            hand = ts_3d[f, mid_finger_idx] - ts_3d[f, rw]
            feats.append(_angle_between(forearm, hand))
        else:
            feats.append(90.0)
    else:
        feats.append(90.0)

    # Fingertip velocity at release
    for fname in finger_tips[:3]:  # index, middle, ring
        fi = kp_index.get(fname)
        if fi is not None:
            vel = np.gradient(ts_hr[:, fi, :], DT, axis=0)
            feats.append(np.linalg.norm(vel[f]))
            feats.append(vel[f, 2])  # vertical component
        else:
            feats.extend([0.0, 0.0])

    # ---- TEMPORAL DYNAMICS ----
    # Rate of change of key angles near release
    if all(x is not None for x in [re_i, rs_i, rw_i]):
        elbow_angles = []
        for ff in range(max(0, f-10), min(240, f+10)):
            ua = ts_3d[ff, re_i] - ts_3d[ff, rs_i]
            fa = ts_3d[ff, rw_i] - ts_3d[ff, re_i]
            elbow_angles.append(_angle_between(-ua, fa))
        elbow_angles = np.array(elbow_angles)
        feats.append(np.gradient(elbow_angles).mean())  # mean rate
        feats.append(np.gradient(elbow_angles).max())   # peak rate
    else:
        feats.extend([0.0, 0.0])

    # Body center of mass height at release vs start
    if mh_i is not None:
        feats.append(ts_hr[f, mh_i, 2])
        feats.append(ts_hr[f, mh_i, 2] - ts_hr[60, mh_i, 2])  # rise from start
        vel_z = np.gradient(ts_hr[:, mh_i, 2], DT)
        feats.append(vel_z[f])  # upward velocity at release
    else:
        feats.extend([0.0, 0.0, 0.0])

    # ---- MULTI-FRAME RELATIVE FEATURES ----
    # Position change from 10 frames before to release
    if rw is not None:
        for coord in range(3):
            feats.append(ts_hr[f, rw, coord] - ts_hr[max(0, f-10), rw, coord])
            feats.append(ts_hr[f, rw, coord] - ts_hr[max(0, f-5), rw, coord])
    else:
        feats.extend([0.0] * 6)

    return np.array(feats, dtype=np.float32)


def run_honest_loo_p4(X_p4, y_p4, X_raw_p4, y_raw_p4, bw, alpha, pls_nc,
                      use_rich=True, X_all_train=None, y_all_train=None,
                      X_raw_all=None, y_raw_all=None, cross_player=False,
                      pids_all=None):
    """Run honest LOO for P4: refit PLS each fold to avoid leakage."""
    n = len(X_p4)
    oof = np.zeros(n)

    for i in range(n):
        # Leave out sample i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        if cross_player:
            # Use all training data but leave out P4 sample i
            # Find which global index corresponds to P4 sample i
            p4_global_idx = np.where(pids_all == 4)[0]
            all_mask = np.ones(len(pids_all), dtype=bool)
            all_mask[p4_global_idx[i]] = False

            # PLS on all data minus held-out
            pls_sc = StandardScaler()
            raw_tr = pls_sc.fit_transform(X_raw_all[all_mask])
            nc = min(pls_nc, sum(all_mask) - 1)
            nc = max(3, nc)
            pls = PLSRegression(n_components=nc)
            pls.fit(raw_tr, y_raw_all[all_mask])
            pls_tr = pls.transform(raw_tr)
            pls_held = pls.transform(pls_sc.transform(X_raw_all[p4_global_idx[i]:p4_global_idx[i]+1]))

            # Features for all train minus held-out
            X_tr_aug = np.hstack([X_all_train[all_mask], pls_tr])
            X_held_aug = np.hstack([X_all_train[p4_global_idx[i]:p4_global_idx[i]+1], pls_held])
            y_tr = y_all_train[all_mask]

            # Standardize
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_aug)
            X_held_s = sc.transform(X_held_aug)

            # Distance-weighted prediction
            D = cdist(X_held_s, X_tr_s, 'euclidean')
            all_dists = cdist(X_tr_s, X_tr_s, 'euclidean')
            sigma = np.quantile(all_dists[np.triu_indices(len(X_tr_s), k=1)], bw)
            sigma = max(sigma, 1e-6)
            weights = np.exp(-D[0]**2 / (2 * sigma**2))
            # Upweight P4 samples
            p4_in_remaining = np.where(pids_all[all_mask] == 4)[0]
            weights[p4_in_remaining] *= 3.0

            if weights.sum() < 1e-10:
                oof[i] = np.mean(y_tr[p4_in_remaining]) if len(p4_in_remaining) > 0 else np.mean(y_tr)
            else:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=weights)
                oof[i] = ridge.predict(X_held_s)[0]
        else:
            # Standard P4-only LOO
            # Refit PLS on training fold
            pls_sc = StandardScaler()
            raw_tr = pls_sc.fit_transform(X_raw_p4[mask])
            nc = min(pls_nc, sum(mask) - 1)
            nc = max(2, nc)
            pls = PLSRegression(n_components=nc)
            pls.fit(raw_tr, y_raw_p4[mask])
            pls_tr = pls.transform(raw_tr)
            pls_held = pls.transform(pls_sc.transform(X_raw_p4[i:i+1]))

            X_tr_aug = np.hstack([X_p4[mask], pls_tr])
            X_held_aug = np.hstack([X_p4[i:i+1], pls_held])
            y_tr = y_p4[mask]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_aug)
            X_held_s = sc.transform(X_held_aug)

            D = cdist(X_tr_s, X_tr_s, 'euclidean')
            all_dists = D[np.triu_indices(len(X_tr_s), k=1)]
            sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            d_to_held = cdist(X_held_s, X_tr_s, 'euclidean')[0]
            weights = np.exp(-d_to_held**2 / (2 * sigma**2))

            if weights.sum() < 1e-10:
                oof[i] = np.mean(y_tr)
            else:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=weights)
                oof[i] = ridge.predict(X_held_s)[0]

    return oof


def predict_test_p4(X_p4_train, y_p4_train, X_raw_p4_train, y_raw_p4_train,
                    X_p4_test, X_raw_p4_test, bw, alpha, pls_nc,
                    X_all_train=None, y_all_train=None, X_raw_all=None, y_raw_all=None,
                    cross_player=False, pids_all=None, X_all_test_feats=None, X_raw_all_test=None):
    """Generate test predictions for P4."""
    n_te = len(X_p4_test)

    if cross_player:
        # Use all training data
        pls_sc = StandardScaler()
        raw_tr = pls_sc.fit_transform(X_raw_all)
        nc = min(pls_nc, len(X_raw_all) - 1)
        nc = max(3, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(raw_tr, y_raw_all)
        pls_tr = pls.transform(raw_tr)
        pls_te = pls.transform(pls_sc.transform(X_raw_p4_test))

        X_tr_aug = np.hstack([X_all_train, pls_tr])
        X_te_aug = np.hstack([X_p4_test, pls_te])
        y_tr = y_all_train

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_te_s = sc.transform(X_te_aug)

        D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
        sigma = np.quantile(D_tr[np.triu_indices(len(X_tr_s), k=1)], bw)
        sigma = max(sigma, 1e-6)

        D_te = cdist(X_te_s, X_tr_s, 'euclidean')
        p4_mask = pids_all == 4

        preds = np.zeros(n_te)
        for j in range(n_te):
            weights = np.exp(-D_te[j]**2 / (2 * sigma**2))
            weights[p4_mask] *= 3.0
            if weights.sum() < 1e-10:
                preds[j] = np.mean(y_tr[p4_mask])
            else:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=weights)
                preds[j] = ridge.predict(X_te_s[j:j+1])[0]
        return preds
    else:
        # P4-only
        pls_sc = StandardScaler()
        raw_tr = pls_sc.fit_transform(X_raw_p4_train)
        nc = min(pls_nc, len(X_raw_p4_train) - 1)
        nc = max(2, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(raw_tr, y_raw_p4_train)
        pls_tr = pls.transform(raw_tr)
        pls_te = pls.transform(pls_sc.transform(X_raw_p4_test))

        X_tr_aug = np.hstack([X_p4_train, pls_tr])
        X_te_aug = np.hstack([X_p4_test, pls_te])
        y_tr = y_p4_train

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_te_s = sc.transform(X_te_aug)

        D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
        all_dists = D_tr[np.triu_indices(len(X_tr_s), k=1)]
        sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        D_te = cdist(X_te_s, X_tr_s, 'euclidean')
        preds = np.zeros(n_te)
        for j in range(n_te):
            weights = np.exp(-D_te[j]**2 / (2 * sigma**2))
            if weights.sum() < 1e-10:
                preds[j] = np.mean(y_tr)
            else:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=weights)
                preds[j] = ridge.predict(X_te_s[j:j+1])[0]
        return preds


def run_alternative_model_loo(X_p4, y_p4, X_raw_p4, y_raw_p4, model_type, pls_nc=8):
    """Run LOO with alternative model types."""
    n = len(X_p4)
    oof = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        # Honest PLS
        pls_sc = StandardScaler()
        raw_tr = pls_sc.fit_transform(X_raw_p4[mask])
        nc = min(pls_nc, sum(mask) - 1)
        nc = max(2, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(raw_tr, y_raw_p4[mask])
        pls_tr = pls.transform(raw_tr)
        pls_held = pls.transform(pls_sc.transform(X_raw_p4[i:i+1]))

        X_tr_aug = np.hstack([X_p4[mask], pls_tr])
        X_held_aug = np.hstack([X_p4[i:i+1], pls_held])
        y_tr = y_p4[mask]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_held_s = sc.transform(X_held_aug)

        if model_type == 'knn_5':
            model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        elif model_type == 'knn_10':
            model = KNeighborsRegressor(n_neighbors=10, weights='distance')
        elif model_type == 'knn_15':
            model = KNeighborsRegressor(n_neighbors=15, weights='distance')
        elif model_type == 'rf_small':
            model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
        elif model_type == 'rf_medium':
            model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42)
        elif model_type == 'gbr':
            model = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05,
                                              min_samples_leaf=5, random_state=42)
        elif model_type == 'lgb':
            model = lgb.LGBMRegressor(n_estimators=80, num_leaves=8, learning_rate=0.05,
                                       min_child_samples=5, reg_alpha=1.0, reg_lambda=1.0,
                                       random_state=42, verbose=-1)
        elif model_type == 'ridge_low':
            model = Ridge(alpha=1.0)
        elif model_type == 'ridge_high':
            model = Ridge(alpha=100.0)
        elif model_type == 'elastic':
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        else:
            model = Ridge(alpha=10.0)

        model.fit(X_tr_s, y_tr)
        oof[i] = model.predict(X_held_s)[0]

    return oof


def predict_test_alt_model(X_p4_train, y_p4_train, X_raw_p4_train, y_raw_p4_train,
                           X_p4_test, X_raw_p4_test, model_type, pls_nc=8):
    """Test predictions with alternative model."""
    pls_sc = StandardScaler()
    raw_tr = pls_sc.fit_transform(X_raw_p4_train)
    nc = min(pls_nc, len(X_raw_p4_train) - 1)
    nc = max(2, nc)
    pls = PLSRegression(n_components=nc)
    pls.fit(raw_tr, y_raw_p4_train)
    pls_tr = pls.transform(raw_tr)
    pls_te = pls.transform(pls_sc.transform(X_raw_p4_test))

    X_tr_aug = np.hstack([X_p4_train, pls_tr])
    X_te_aug = np.hstack([X_p4_test, pls_te])

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_aug)
    X_te_s = sc.transform(X_te_aug)

    if model_type == 'knn_5':
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    elif model_type == 'knn_10':
        model = KNeighborsRegressor(n_neighbors=10, weights='distance')
    elif model_type == 'knn_15':
        model = KNeighborsRegressor(n_neighbors=15, weights='distance')
    elif model_type == 'rf_small':
        model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
    elif model_type == 'rf_medium':
        model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42)
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05,
                                          min_samples_leaf=5, random_state=42)
    elif model_type == 'lgb':
        model = lgb.LGBMRegressor(n_estimators=80, num_leaves=8, learning_rate=0.05,
                                   min_child_samples=5, reg_alpha=1.0, reg_lambda=1.0,
                                   random_state=42, verbose=-1)
    elif model_type == 'ridge_low':
        model = Ridge(alpha=1.0)
    elif model_type == 'ridge_high':
        model = Ridge(alpha=100.0)
    elif model_type == 'elastic':
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
    else:
        model = Ridge(alpha=10.0)

    model.fit(X_tr_s, y_p4_train)
    return model.predict(X_te_s)


def main():
    t0 = time.time()
    print("=" * 70)
    print("P4 ANGLE FIX - Targeted Player 4 Angle Improvement")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scaler_angle = joblib.load(DATA_DIR / "scaler_angle.pkl")
    y_angle_raw = y_train[:, 0]
    y_angle_sc = scaler_angle.transform(y_angle_raw.reshape(-1, 1)).ravel()

    # P4 masks
    p4_tr_mask = pids_train == 4
    p4_te_mask = pids_test == 4
    p4_tr_idx = np.where(p4_tr_mask)[0]
    p4_te_idx = np.where(p4_te_mask)[0]
    n_p4_train = len(p4_tr_idx)
    n_p4_test = len(p4_te_idx)

    print(f"P4 training samples: {n_p4_train}")
    print(f"P4 test samples: {n_p4_test}")

    y_p4_sc = y_angle_sc[p4_tr_mask]
    y_p4_raw = y_angle_raw[p4_tr_mask]

    # Extract features for multiple frames
    print("\nExtracting rich features at multiple frames...")
    FRAMES_TO_TRY = [140, 145, 148, 150, 153, 155, 160]

    features_by_frame = {}
    for frame in FRAMES_TO_TRY:
        all_feats_tr = []
        all_feats_te = []
        for i in range(len(pids_train)):
            ts_3d = train_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            feats = extract_rich_features(ts_3d, ts_hr, kp_index, rf, frame)
            all_feats_tr.append(feats)
        for i in range(len(pids_test)):
            ts_3d = test_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            feats = extract_rich_features(ts_3d, ts_hr, kp_index, rf, frame)
            all_feats_te.append(feats)

        X_tr = np.nan_to_num(np.array(all_feats_tr), nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(np.array(all_feats_te), nan=0.0, posinf=0.0, neginf=0.0)
        features_by_frame[frame] = {'train': X_tr, 'test': X_te}
        print(f"  Frame {frame}: {X_tr.shape[1]} features extracted")

    # Raw data
    X_raw_p4_train = train_data['X_raw'][p4_tr_mask]
    X_raw_p4_test = test_data['X_raw'][p4_te_mask]
    X_raw_all = train_data['X_raw']

    # ==========================================================
    # EXPERIMENT 1: Hyperparameter sweep on baseline approach
    # ==========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Hyperparameter sweep (honest LOO)")
    print("=" * 70)

    baseline_frame = 150  # P4 angle frame from PLAYER_FRAMES
    X_p4_base = features_by_frame[baseline_frame]['train'][p4_tr_mask]

    results = {}

    # Sweep bandwidth and alpha
    best_loo = float('inf')
    best_config = None

    for bw in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80]:
        for alpha in [1.0, 5.0, 10.0, 20.0, 50.0]:
            for pls_nc in [3, 5, 8, 10]:
                oof = run_honest_loo_p4(X_p4_base, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                        bw, alpha, pls_nc)
                mse = np.mean((oof - y_p4_sc)**2)
                if mse < best_loo:
                    best_loo = mse
                    best_config = {'bw': bw, 'alpha': alpha, 'pls_nc': pls_nc}

    print(f"Best config: bw={best_config['bw']}, alpha={best_config['alpha']}, pls_nc={best_config['pls_nc']}")
    print(f"Best P4 angle honest LOO: {best_loo:.6f}")

    # Also print baseline
    oof_baseline = run_honest_loo_p4(X_p4_base, y_p4_sc, X_raw_p4_train, y_p4_raw, 0.30, 10.0, 10)
    mse_baseline = np.mean((oof_baseline - y_p4_sc)**2)
    print(f"Baseline (bw=0.30, a=10, nc=10) P4 LOO: {mse_baseline:.6f}")
    print(f"Improvement: {(1 - best_loo/mse_baseline)*100:.2f}%")

    results['hyperparam_best'] = {
        'config': best_config,
        'loo': best_loo,
        'baseline_loo': mse_baseline,
    }

    # ==========================================================
    # EXPERIMENT 2: Frame sweep
    # ==========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Frame sweep (using best hyperparams)")
    print("=" * 70)

    bw_best = best_config['bw']
    alpha_best = best_config['alpha']
    pls_nc_best = best_config['pls_nc']

    frame_results = {}
    for frame in FRAMES_TO_TRY:
        X_p4_f = features_by_frame[frame]['train'][p4_tr_mask]
        oof = run_honest_loo_p4(X_p4_f, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                bw_best, alpha_best, pls_nc_best)
        mse = np.mean((oof - y_p4_sc)**2)
        frame_results[frame] = mse
        print(f"  Frame {frame}: P4 LOO = {mse:.6f}")

    best_frame = min(frame_results, key=frame_results.get)
    print(f"Best frame: {best_frame} (LOO = {frame_results[best_frame]:.6f})")
    results['best_frame'] = best_frame

    # ==========================================================
    # EXPERIMENT 3: Multi-frame ensemble
    # ==========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Multi-frame ensemble")
    print("=" * 70)

    # Try top-3 frames
    sorted_frames = sorted(frame_results.items(), key=lambda x: x[1])
    top3_frames = [f for f, _ in sorted_frames[:3]]
    top5_frames = [f for f, _ in sorted_frames[:5]]

    for combo_name, combo_frames in [('top3', top3_frames), ('top5', top5_frames)]:
        oofs = []
        for frame in combo_frames:
            X_p4_f = features_by_frame[frame]['train'][p4_tr_mask]
            oof = run_honest_loo_p4(X_p4_f, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                    bw_best, alpha_best, pls_nc_best)
            oofs.append(oof)
        avg_oof = np.mean(oofs, axis=0)
        mse_avg = np.mean((avg_oof - y_p4_sc)**2)
        print(f"  {combo_name} frames {combo_frames}: P4 LOO = {mse_avg:.6f}")
        results[f'multiframe_{combo_name}'] = {'frames': combo_frames, 'loo': mse_avg}

    # ==========================================================
    # EXPERIMENT 4: Alternative models
    # ==========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Alternative model types")
    print("=" * 70)

    X_p4_best = features_by_frame[best_frame]['train'][p4_tr_mask]

    model_types = ['knn_5', 'knn_10', 'knn_15', 'rf_small', 'rf_medium', 'gbr', 'lgb',
                   'ridge_low', 'ridge_high', 'elastic']

    model_results = {}
    for mt in model_types:
        oof = run_alternative_model_loo(X_p4_best, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                        mt, pls_nc=pls_nc_best)
        mse = np.mean((oof - y_p4_sc)**2)
        model_results[mt] = mse
        print(f"  {mt}: P4 LOO = {mse:.6f}")

    best_model = min(model_results, key=model_results.get)
    print(f"Best model: {best_model} (LOO = {model_results[best_model]:.6f})")
    results['model_results'] = model_results

    # ==========================================================
    # EXPERIMENT 5: Cross-player transfer
    # ==========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Cross-player transfer")
    print("=" * 70)

    X_all_best = features_by_frame[best_frame]['train']

    for bw_cp in [0.50, 0.60, 0.80]:
        oof = run_honest_loo_p4(X_p4_best, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                bw_cp, alpha_best, pls_nc_best,
                                X_all_train=X_all_best, y_all_train=y_angle_sc,
                                X_raw_all=X_raw_all, y_raw_all=y_angle_raw,
                                cross_player=True, pids_all=pids_train)
        mse = np.mean((oof - y_p4_sc)**2)
        print(f"  Cross-player bw={bw_cp}: P4 LOO = {mse:.6f}")
        results[f'cross_player_bw{bw_cp}'] = mse

    # ==========================================================
    # EXPERIMENT 6: Ensemble of approaches
    # ==========================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Ensemble of best approaches")
    print("=" * 70)

    # Collect top approaches
    approach_oofs = {}
    approach_test_preds = {}

    # A) Best hyperparams at best frame
    oof_a = run_honest_loo_p4(features_by_frame[best_frame]['train'][p4_tr_mask], y_p4_sc,
                               X_raw_p4_train, y_p4_raw, bw_best, alpha_best, pls_nc_best)
    test_a = predict_test_p4(features_by_frame[best_frame]['train'][p4_tr_mask], y_p4_sc,
                              X_raw_p4_train, y_p4_raw,
                              features_by_frame[best_frame]['test'][p4_te_mask],
                              X_raw_p4_test, bw_best, alpha_best, pls_nc_best)
    approach_oofs['best_hyper'] = oof_a
    approach_test_preds['best_hyper'] = test_a
    mse_a = np.mean((oof_a - y_p4_sc)**2)
    print(f"  best_hyper: LOO = {mse_a:.6f}")

    # B) Multi-frame ensemble (top3)
    oofs_b, tests_b = [], []
    for frame in top3_frames:
        oof_f = run_honest_loo_p4(features_by_frame[frame]['train'][p4_tr_mask], y_p4_sc,
                                   X_raw_p4_train, y_p4_raw, bw_best, alpha_best, pls_nc_best)
        test_f = predict_test_p4(features_by_frame[frame]['train'][p4_tr_mask], y_p4_sc,
                                  X_raw_p4_train, y_p4_raw,
                                  features_by_frame[frame]['test'][p4_te_mask],
                                  X_raw_p4_test, bw_best, alpha_best, pls_nc_best)
        oofs_b.append(oof_f)
        tests_b.append(test_f)
    oof_b = np.mean(oofs_b, axis=0)
    test_b = np.mean(tests_b, axis=0)
    approach_oofs['multiframe_top3'] = oof_b
    approach_test_preds['multiframe_top3'] = test_b
    mse_b = np.mean((oof_b - y_p4_sc)**2)
    print(f"  multiframe_top3: LOO = {mse_b:.6f}")

    # C) Best alternative model
    oof_c = run_alternative_model_loo(X_p4_best, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                      best_model, pls_nc=pls_nc_best)
    test_c = predict_test_alt_model(X_p4_best, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                     features_by_frame[best_frame]['test'][p4_te_mask],
                                     X_raw_p4_test, best_model, pls_nc=pls_nc_best)
    approach_oofs['alt_model'] = oof_c
    approach_test_preds['alt_model'] = test_c
    mse_c = np.mean((oof_c - y_p4_sc)**2)
    print(f"  alt_model ({best_model}): LOO = {mse_c:.6f}")

    # D) Baseline for comparison
    oof_d = run_honest_loo_p4(features_by_frame[150]['train'][p4_tr_mask], y_p4_sc,
                               X_raw_p4_train, y_p4_raw, 0.30, 10.0, 10)
    test_d = predict_test_p4(features_by_frame[150]['train'][p4_tr_mask], y_p4_sc,
                              X_raw_p4_train, y_p4_raw,
                              features_by_frame[150]['test'][p4_te_mask],
                              X_raw_p4_test, 0.30, 10.0, 10)
    approach_oofs['baseline'] = oof_d
    approach_test_preds['baseline'] = test_d
    mse_d = np.mean((oof_d - y_p4_sc)**2)
    print(f"  baseline: LOO = {mse_d:.6f}")

    # E) 2-way ensemble of top approaches
    sorted_approaches = sorted(
        [(k, np.mean((v - y_p4_sc)**2)) for k, v in approach_oofs.items()],
        key=lambda x: x[1]
    )
    print(f"\n  Approach ranking:")
    for name, mse in sorted_approaches:
        delta = (1 - mse/mse_d) * 100
        print(f"    {name}: LOO = {mse:.6f} ({delta:+.2f}% vs baseline)")

    top2_names = [sorted_approaches[0][0], sorted_approaches[1][0]]
    oof_top2 = np.mean([approach_oofs[n] for n in top2_names], axis=0)
    test_top2 = np.mean([approach_test_preds[n] for n in top2_names], axis=0)
    mse_top2 = np.mean((oof_top2 - y_p4_sc)**2)
    print(f"\n  Top-2 ensemble ({top2_names}): LOO = {mse_top2:.6f}")

    # All-approach ensemble
    oof_all = np.mean([approach_oofs[n] for n, _ in sorted_approaches[:3]], axis=0)
    test_all = np.mean([approach_test_preds[n] for n, _ in sorted_approaches[:3]], axis=0)
    mse_all = np.mean((oof_all - y_p4_sc)**2)
    print(f"  Top-3 ensemble: LOO = {mse_all:.6f}")

    # ==========================================================
    # DETERMINE BEST APPROACH AND GENERATE SUBMISSIONS
    # ==========================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Collect all candidate P4 predictions
    candidates = {
        'best_hyper': (mse_a, test_a),
        'multiframe_top3': (mse_b, test_b),
        f'alt_model_{best_model}': (mse_c, test_c),
        'top2_ensemble': (mse_top2, test_top2),
        'top3_ensemble': (mse_all, test_all),
        'baseline': (mse_d, test_d),
    }

    # Sort by LOO
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0])

    print("\nCandidate rankings (honest P4 angle LOO):")
    for name, (mse, _) in sorted_candidates:
        delta = (1 - mse/mse_d) * 100
        print(f"  {name}: {mse:.6f} ({delta:+.2f}% vs baseline)")

    # Load base submission
    base_sub = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")
    test_ids = test_data['ids']
    p4_test_ids = test_ids[p4_te_mask]

    # Check P5 too - second worst player for angle
    p5_tr_mask = pids_train == 5
    p5_te_mask = pids_test == 5
    p5_tr_idx = np.where(p5_tr_mask)[0]
    p5_te_idx = np.where(p5_te_mask)[0]
    y_p5_sc = y_angle_sc[p5_tr_mask]
    y_p5_raw = y_angle_raw[p5_tr_mask]
    X_raw_p5_train = train_data['X_raw'][p5_tr_mask]
    X_raw_p5_test = test_data['X_raw'][p5_te_mask]

    # Quick P5 angle check with best hyperparams
    X_p5_best = features_by_frame[best_frame]['train'][p5_tr_mask]
    oof_p5 = run_honest_loo_p4(X_p5_best, y_p5_sc, X_raw_p5_train, y_p5_raw,
                                bw_best, alpha_best, pls_nc_best)
    mse_p5_new = np.mean((oof_p5 - y_p5_sc)**2)

    oof_p5_base = run_honest_loo_p4(features_by_frame[150]['train'][p5_tr_mask], y_p5_sc,
                                     X_raw_p5_train, y_p5_raw, 0.30, 10.0, 10)
    mse_p5_base = np.mean((oof_p5_base - y_p5_sc)**2)

    print(f"\nP5 angle baseline LOO: {mse_p5_base:.6f}")
    print(f"P5 angle with P4-optimal params LOO: {mse_p5_new:.6f} ({(1-mse_p5_new/mse_p5_base)*100:+.2f}%)")

    # Generate P5 test predictions with best params
    test_p5 = predict_test_p4(X_p5_best, y_p5_sc, X_raw_p5_train, y_p5_raw,
                               features_by_frame[best_frame]['test'][p5_te_mask],
                               X_raw_p5_test, bw_best, alpha_best, pls_nc_best)

    p5_test_ids = test_ids[p5_te_mask]

    # --- Generate submissions ---
    submission_details = []

    # Take top 3 candidates
    for rank, (cand_name, (cand_mse, cand_test)) in enumerate(sorted_candidates[:3]):
        # Sub A: P4 angle splice only
        sub_num = get_next_submission_number()
        sub = base_sub.copy()
        for j, tid in enumerate(p4_test_ids):
            idx = sub[sub['id'] == tid].index[0]
            sub.loc[idx, 'scaled_angle'] = cand_test[j]
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

        desc = (f"Sub {sub_num}: P4 angle splice ({cand_name}) into Sub 3411. "
                f"Config: {cand_name}, P4 LOO={cand_mse:.6f} vs baseline {mse_d:.6f} "
                f"({(1-cand_mse/mse_d)*100:+.2f}%)")
        print(f"\n{desc}")
        submission_details.append(desc)

        # Sub B: P4+P5 angle splice
        if mse_p5_new < mse_p5_base:
            sub_num2 = get_next_submission_number()
            sub2 = base_sub.copy()
            for j, tid in enumerate(p4_test_ids):
                idx = sub2[sub2['id'] == tid].index[0]
                sub2.loc[idx, 'scaled_angle'] = cand_test[j]
            for j, tid in enumerate(p5_test_ids):
                idx = sub2[sub2['id'] == tid].index[0]
                sub2.loc[idx, 'scaled_angle'] = test_p5[j]
            sub2.to_csv(SUBMISSION_DIR / f"submission_{sub_num2}.csv", index=False)

            desc2 = (f"Sub {sub_num2}: P4+P5 angle splice ({cand_name}+P5opt) into Sub 3411. "
                     f"P4 LOO={cand_mse:.6f}, P5 LOO={mse_p5_new:.6f}")
            print(f"{desc2}")
            submission_details.append(desc2)

    # Sub C: Blend approach - 50/50 blend of best candidate with Sub 3411 P4 angle
    best_cand_name, (best_cand_mse, best_cand_test) = sorted_candidates[0]
    for blend_w in [0.3, 0.5, 0.7]:
        sub_num = get_next_submission_number()
        sub = base_sub.copy()
        for j, tid in enumerate(p4_test_ids):
            idx = sub[sub['id'] == tid].index[0]
            old_val = sub.loc[idx, 'scaled_angle']
            sub.loc[idx, 'scaled_angle'] = blend_w * best_cand_test[j] + (1 - blend_w) * old_val
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

        desc = (f"Sub {sub_num}: P4 angle {int(blend_w*100)}% blend of {best_cand_name} + "
                f"{int((1-blend_w)*100)}% Sub3411. P4 LOO={best_cand_mse:.6f}")
        print(f"\n{desc}")
        submission_details.append(desc)

    # ==========================================================
    # CALIBRATION CHECK
    # ==========================================================
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)

    for name, (mse, test_pred) in sorted_candidates[:3]:
        pred_std = np.std(test_pred)
        train_std = np.std(y_p4_sc)
        print(f"  {name}: pred_std={pred_std:.6f}, train_std={train_std:.6f}, ratio={pred_std/train_std:.4f}")

    print(f"\n  Sub3411 P4 angle: pred_std={base_sub.set_index('id').loc[p4_test_ids, 'scaled_angle'].std():.6f}")

    # ==========================================================
    # SAVE RESEARCH
    # ==========================================================
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Write research file
    research = f"""# P4 Angle Fix Results - 2026-02-19

## Context
- P4 has the highest angle variance (std=2.68 raw, 0.0895 scaled) among typical players
- Sub 3411 P4 angle calibration slope: 0.65 (captures only 65% of true variation)
- P4 contributes 33.6% of all angle error

## Experiment 1: Hyperparameter Sweep (honest LOO)
- Baseline (bw=0.30, alpha=10, PLS nc=10): P4 LOO = {mse_baseline:.6f}
- Best config: bw={best_config['bw']}, alpha={best_config['alpha']}, PLS nc={best_config['pls_nc']}
- Best P4 LOO: {best_loo:.6f} ({(1-best_loo/mse_baseline)*100:+.2f}%)

## Experiment 2: Frame Sweep
- Frames tested: {FRAMES_TO_TRY}
- Best frame: {best_frame} (LOO = {frame_results[best_frame]:.6f})
- Frame results: {dict(sorted(frame_results.items(), key=lambda x: x[1]))}

## Experiment 3: Multi-Frame Ensemble
- Top-3 frames: {top3_frames}
- Top-3 ensemble LOO: {results.get('multiframe_top3', {}).get('loo', 'N/A')}

## Experiment 4: Alternative Models
- Models tested: {list(model_results.keys())}
- Best model: {best_model} (LOO = {model_results[best_model]:.6f})
- All results: {dict(sorted(model_results.items(), key=lambda x: x[1]))}

## Experiment 5: Cross-Player Transfer
- Results: { {k: v for k, v in results.items() if 'cross_player' in k} }

## Experiment 6: Ensemble
- Top-2 ensemble LOO: {mse_top2:.6f}
- Top-3 ensemble LOO: {mse_all:.6f}

## P5 Angle Check
- P5 baseline LOO: {mse_p5_base:.6f}
- P5 with P4-optimal params: {mse_p5_new:.6f} ({(1-mse_p5_new/mse_p5_base)*100:+.2f}%)

## Final Rankings (honest P4 angle LOO)
"""
    for name, (mse, _) in sorted_candidates:
        delta = (1 - mse/mse_d) * 100
        research += f"- {name}: {mse:.6f} ({delta:+.2f}% vs baseline)\n"

    research += f"""
## Submissions Generated
"""
    for detail in submission_details:
        research += f"- {detail}\n"

    research += f"""
## Runtime
Total: {elapsed:.1f}s
"""

    research_path = PROJECT_DIR / "Research" / "P4_ANGLE_FIX_RESULTS_2026-02-19.md"
    research_path.write_text(research)
    print(f"\nResearch saved to {research_path}")

    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    for detail in submission_details:
        print(f"  {detail}")


if __name__ == "__main__":
    main()
