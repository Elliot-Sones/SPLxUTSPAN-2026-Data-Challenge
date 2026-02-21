"""
P4 Angle Fix v2 - Faster version using 5-fold CV for sweep, LOO for final eval.

Strategies:
1. Hyperparameter sweep with 5-fold CV (fast)
2. Multi-frame ensemble
3. Alternative models (k-NN, LightGBM, RF)
4. Cross-player transfer
5. Ensemble of top approaches
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
from sklearn.model_selection import KFold
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


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract rich feature set at a specific frame. ~280 features."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    # ---- BASE FEATURES (198 from combined_best) ----
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
            feats.append(float(np.nanmean(series)))
            feats.append(float(np.nanstd(series)))
            feats.append(float(np.nanmax(series) - np.nanmin(series)))

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
    ls_i = kp_index.get('left_shoulder')
    if rh is not None and lh is not None:
        feats.append(ts_hr[f, rh, 1] - ts_hr[f, lh, 1])
        feats.append(ts_hr[f, rh, 0] - ts_hr[f, lh, 0])
    else:
        feats.extend([0.0, 0.0])
    if rs is not None and ls_i is not None:
        feats.append(ts_hr[f, rs, 1] - ts_hr[f, ls_i, 1])
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
            feats.append(float(np.nanmean(series[140:180])))
            feats.append(float(np.nanmax(vel[140:180])))
    else:
        feats.extend([0.0] * 6)

    # ---- JOINT ANGLES (10 features from combined_best) ----
    rw_i, re_i, rs_i = kp_index.get('right_wrist'), kp_index.get('right_elbow'), kp_index.get('right_shoulder')
    rh_i, lh_i = kp_index.get('right_hip'), kp_index.get('left_hip')
    rk_i, lk_i = kp_index.get('right_knee'), kp_index.get('left_knee')
    ra_i, la_i = kp_index.get('right_ankle'), kp_index.get('left_ankle')
    neck_i, mh_i = kp_index.get('neck'), kp_index.get('mid_hip')

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

    # ---- FLICK WINDOW FEATURES (crucial for angle) ----
    flick_s, flick_e = 135, 165
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            acc = np.gradient(vel, DT)
            feats.append(float(np.nanmean(series[flick_s:flick_e])))
            feats.append(float(np.nanstd(series[flick_s:flick_e])))
            feats.append(float(np.nanmean(vel[flick_s:flick_e])))
            feats.append(float(np.nanmax(vel[flick_s:flick_e])))
            feats.append(float(np.nanmean(acc[flick_s:flick_e])))
    else:
        feats.extend([0.0] * 15)

    # Elbow flick
    if re is not None:
        for coord in range(3):
            vel = np.gradient(ts_hr[:, re, coord], DT)
            feats.append(float(np.nanmax(vel[flick_s:flick_e])))
    else:
        feats.extend([0.0] * 3)

    # Elbow angle change rate during flick
    if all(x is not None for x in [re_i, rs_i, rw_i]):
        angles = []
        for ff in range(flick_s, min(flick_e, 240)):
            ua = ts_3d[ff, re_i] - ts_3d[ff, rs_i]
            fa = ts_3d[ff, rw_i] - ts_3d[ff, re_i]
            angles.append(_angle_between(-ua, fa))
        angles = np.array(angles)
        feats.append(float(angles[-1] - angles[0]))
        feats.append(float(np.nanmax(np.gradient(angles))))
    else:
        feats.extend([0.0, 0.0])

    # ---- FINGERTIP FEATURES ----
    finger_tips = ['right_second_finger_distal', 'right_third_finger_distal',
                   'right_fourth_finger_distal']
    finger_proxs = ['right_second_finger_proximal', 'right_third_finger_proximal',
                    'right_fourth_finger_proximal']
    # Finger spread
    tip_pos = []
    for fname in finger_tips:
        fi = kp_index.get(fname)
        tip_pos.append(ts_hr[f, fi, :] if fi is not None else np.zeros(3))
    tip_pos = np.array(tip_pos)
    for i in range(len(tip_pos)):
        for j in range(i+1, len(tip_pos)):
            feats.append(float(np.linalg.norm(tip_pos[i] - tip_pos[j])))
    # Finger curl
    for ft, fp in zip(finger_tips, finger_proxs):
        fi_t, fi_p = kp_index.get(ft), kp_index.get(fp)
        if fi_t is not None and fi_p is not None:
            feats.append(float(np.linalg.norm(ts_hr[f, fi_t] - ts_hr[f, fi_p])))
        else:
            feats.append(0.0)
    # Wrist flexion
    if rw is not None and re is not None:
        mf = kp_index.get('right_third_finger_distal')
        if mf is not None:
            feats.append(_angle_between(ts_3d[f, rw] - ts_3d[f, re], ts_3d[f, mf] - ts_3d[f, rw]))
        else:
            feats.append(90.0)
    else:
        feats.append(90.0)
    # Fingertip velocity
    for fname in finger_tips:
        fi = kp_index.get(fname)
        if fi is not None:
            vel = np.gradient(ts_hr[:, fi, :], DT, axis=0)
            feats.append(float(np.linalg.norm(vel[f])))
        else:
            feats.append(0.0)

    # ---- BODY DYNAMICS ----
    if mh_i is not None:
        feats.append(float(ts_hr[f, mh_i, 2]))
        feats.append(float(ts_hr[f, mh_i, 2] - ts_hr[60, mh_i, 2]))
        vel_z = np.gradient(ts_hr[:, mh_i, 2], DT)
        feats.append(float(vel_z[f]))
    else:
        feats.extend([0.0, 0.0, 0.0])

    # Position delta from 10 frames before
    if rw is not None:
        for coord in range(3):
            feats.append(float(ts_hr[f, rw, coord] - ts_hr[max(0, f-10), rw, coord]))
    else:
        feats.extend([0.0] * 3)

    return np.array(feats, dtype=np.float32)


def kfold_cv_p4(X_p4, y_p4, X_raw_p4, y_raw_p4, bw, alpha, pls_nc, n_folds=5, seed=42):
    """Fast 5-fold CV for P4 (honest PLS within each fold)."""
    n = len(X_p4)
    oof = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for tr_idx, val_idx in kf.split(X_p4):
        # PLS on training fold only (honest)
        pls_sc = StandardScaler()
        raw_tr = pls_sc.fit_transform(X_raw_p4[tr_idx])
        nc = min(pls_nc, len(tr_idx) - 1)
        nc = max(2, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(raw_tr, y_raw_p4[tr_idx])
        pls_tr = pls.transform(raw_tr)
        pls_val = pls.transform(pls_sc.transform(X_raw_p4[val_idx]))

        X_tr_aug = np.hstack([X_p4[tr_idx], pls_tr])
        X_val_aug = np.hstack([X_p4[val_idx], pls_val])

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_val_s = sc.transform(X_val_aug)

        # Distance computation for kernel
        D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
        all_dists = D_tr[np.triu_indices(len(X_tr_s), k=1)]
        sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        D_val = cdist(X_val_s, X_tr_s, 'euclidean')

        for j, vi in enumerate(val_idx):
            weights = np.exp(-D_val[j]**2 / (2 * sigma**2))
            if weights.sum() < 1e-10:
                oof[vi] = np.mean(y_p4[tr_idx])
            else:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_p4[tr_idx], sample_weight=weights)
                oof[vi] = ridge.predict(X_val_s[j:j+1])[0]

    return oof


def kfold_cv_alt_model(X_p4, y_p4, X_raw_p4, y_raw_p4, model_fn, pls_nc=8, n_folds=5, seed=42):
    """Fast 5-fold CV with alternative model."""
    n = len(X_p4)
    oof = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for tr_idx, val_idx in kf.split(X_p4):
        pls_sc = StandardScaler()
        raw_tr = pls_sc.fit_transform(X_raw_p4[tr_idx])
        nc = min(pls_nc, len(tr_idx) - 1)
        nc = max(2, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(raw_tr, y_raw_p4[tr_idx])
        pls_tr = pls.transform(raw_tr)
        pls_val = pls.transform(pls_sc.transform(X_raw_p4[val_idx]))

        X_tr_aug = np.hstack([X_p4[tr_idx], pls_tr])
        X_val_aug = np.hstack([X_p4[val_idx], pls_val])

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_val_s = sc.transform(X_val_aug)

        model = model_fn()
        model.fit(X_tr_s, y_p4[tr_idx])
        oof[val_idx] = model.predict(X_val_s)

    return oof


def kfold_cv_cross_player(X_all, y_all, pids_all, X_raw_all, y_raw_all,
                           bw, alpha, pls_nc, upweight=3.0, n_folds=5, seed=42):
    """5-fold CV for P4 using cross-player data."""
    p4_mask = pids_all == 4
    p4_idx = np.where(p4_mask)[0]
    n_p4 = len(p4_idx)

    oof = np.zeros(n_p4)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # We split P4 indices into folds
    for tr_fold, val_fold in kf.split(np.arange(n_p4)):
        val_global = p4_idx[val_fold]

        # Training set: all data except P4 validation fold
        train_mask = np.ones(len(pids_all), dtype=bool)
        train_mask[val_global] = False

        pls_sc = StandardScaler()
        raw_tr = pls_sc.fit_transform(X_raw_all[train_mask])
        nc = min(pls_nc, sum(train_mask) - 1)
        nc = max(3, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(raw_tr, y_raw_all[train_mask])
        pls_tr = pls.transform(raw_tr)
        pls_val = pls.transform(pls_sc.transform(X_raw_all[val_global]))

        X_tr_aug = np.hstack([X_all[train_mask], pls_tr])
        X_val_aug = np.hstack([X_all[val_global], pls_val])

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_val_s = sc.transform(X_val_aug)

        D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
        all_dists = D_tr[np.triu_indices(len(X_tr_s), k=1)]
        sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        D_val = cdist(X_val_s, X_tr_s, 'euclidean')

        # Upweight P4 in training
        p4_in_train = np.where(pids_all[train_mask] == 4)[0]

        for j, vi in enumerate(val_fold):
            weights = np.exp(-D_val[j]**2 / (2 * sigma**2))
            weights[p4_in_train] *= upweight
            if weights.sum() < 1e-10:
                oof[vi] = np.mean(y_all[train_mask][p4_in_train])
            else:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_all[train_mask], sample_weight=weights)
                oof[vi] = ridge.predict(X_val_s[j:j+1])[0]

    return oof


def predict_test(X_train, y_train, X_raw_train, y_raw_train, X_test, X_raw_test,
                 bw, alpha, pls_nc):
    """Generate test predictions using locally weighted Ridge."""
    pls_sc = StandardScaler()
    raw_tr = pls_sc.fit_transform(X_raw_train)
    nc = min(pls_nc, len(X_raw_train) - 1)
    nc = max(2, nc)
    pls = PLSRegression(n_components=nc)
    pls.fit(raw_tr, y_raw_train)
    pls_tr = pls.transform(raw_tr)
    pls_te = pls.transform(pls_sc.transform(X_raw_test))

    X_tr_aug = np.hstack([X_train, pls_tr])
    X_te_aug = np.hstack([X_test, pls_te])

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_aug)
    X_te_s = sc.transform(X_te_aug)

    D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
    sigma = np.quantile(D_tr[np.triu_indices(len(X_tr_s), k=1)], bw)
    sigma = max(sigma, 1e-6)

    D_te = cdist(X_te_s, X_tr_s, 'euclidean')
    preds = np.zeros(len(X_test))
    for j in range(len(X_test)):
        weights = np.exp(-D_te[j]**2 / (2 * sigma**2))
        if weights.sum() < 1e-10:
            preds[j] = np.mean(y_train)
        else:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_train, sample_weight=weights)
            preds[j] = ridge.predict(X_te_s[j:j+1])[0]
    return preds


def predict_test_cross_player(X_all, y_all, pids_all, X_raw_all, y_raw_all,
                               X_test, X_raw_test, bw, alpha, pls_nc, upweight=3.0):
    """Generate P4 test predictions using all training data."""
    pls_sc = StandardScaler()
    raw_tr = pls_sc.fit_transform(X_raw_all)
    nc = min(pls_nc, len(X_raw_all) - 1)
    nc = max(3, nc)
    pls = PLSRegression(n_components=nc)
    pls.fit(raw_tr, y_raw_all)
    pls_tr = pls.transform(raw_tr)
    pls_te = pls.transform(pls_sc.transform(X_raw_test))

    X_tr_aug = np.hstack([X_all, pls_tr])
    X_te_aug = np.hstack([X_test, pls_te])

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_aug)
    X_te_s = sc.transform(X_te_aug)

    D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
    sigma = np.quantile(D_tr[np.triu_indices(len(X_tr_s), k=1)], bw)
    sigma = max(sigma, 1e-6)

    D_te = cdist(X_te_s, X_tr_s, 'euclidean')
    p4_mask = pids_all == 4

    preds = np.zeros(len(X_test))
    for j in range(len(X_test)):
        weights = np.exp(-D_te[j]**2 / (2 * sigma**2))
        weights[p4_mask] *= upweight
        if weights.sum() < 1e-10:
            preds[j] = np.mean(y_all[p4_mask])
        else:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_all, sample_weight=weights)
            preds[j] = ridge.predict(X_te_s[j:j+1])[0]
    return preds


def predict_test_alt_model(X_train, y_train, X_raw_train, y_raw_train,
                            X_test, X_raw_test, model_fn, pls_nc=8):
    """Generate test predictions with alternative model."""
    pls_sc = StandardScaler()
    raw_tr = pls_sc.fit_transform(X_raw_train)
    nc = min(pls_nc, len(X_raw_train) - 1)
    nc = max(2, nc)
    pls = PLSRegression(n_components=nc)
    pls.fit(raw_tr, y_raw_train)
    pls_tr = pls.transform(raw_tr)
    pls_te = pls.transform(pls_sc.transform(X_raw_test))

    X_tr_aug = np.hstack([X_train, pls_tr])
    X_te_aug = np.hstack([X_test, pls_te])

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_aug)
    X_te_s = sc.transform(X_te_aug)

    model = model_fn()
    model.fit(X_tr_s, y_train)
    return model.predict(X_te_s)


def main():
    t0 = time.time()
    print("=" * 70)
    print("P4 ANGLE FIX v2 - Fast 5-fold CV Sweep")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scaler_angle = joblib.load(DATA_DIR / "scaler_angle.pkl")
    y_angle_raw = y_train[:, 0]
    y_angle_sc = scaler_angle.transform(y_angle_raw.reshape(-1, 1)).ravel()

    p4_tr_mask = pids_train == 4
    p4_te_mask = pids_test == 4
    p4_tr_idx = np.where(p4_tr_mask)[0]
    p4_te_idx = np.where(p4_te_mask)[0]

    y_p4_sc = y_angle_sc[p4_tr_mask]
    y_p4_raw = y_angle_raw[p4_tr_mask]

    X_raw_p4_train = train_data['X_raw'][p4_tr_mask]
    X_raw_p4_test = test_data['X_raw'][p4_te_mask]

    # Extract features at multiple frames
    print("\nExtracting features at multiple frames...")
    FRAMES = [140, 145, 148, 150, 153, 155, 160]

    feat_cache = {}
    for frame in FRAMES:
        tr_feats, te_feats = [], []
        for i in range(len(pids_train)):
            ts_3d = train_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            tr_feats.append(extract_features(ts_3d, ts_hr, kp_index, rf, frame))
        for i in range(len(pids_test)):
            ts_3d = test_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            te_feats.append(extract_features(ts_3d, ts_hr, kp_index, rf, frame))
        X_tr = np.nan_to_num(np.array(tr_feats), nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(np.array(te_feats), nan=0.0, posinf=0.0, neginf=0.0)
        feat_cache[frame] = {'train': X_tr, 'test': X_te}
        print(f"  Frame {frame}: {X_tr.shape[1]} features")

    print(f"\nFeature extraction: {time.time()-t0:.1f}s")

    # ================================================================
    # PHASE 1: Hyperparameter sweep with 5-fold CV
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Hyperparameter Sweep (5-fold CV)")
    print("=" * 70)

    baseline_frame = 150
    X_p4 = feat_cache[baseline_frame]['train'][p4_tr_mask]

    # Baseline
    oof_base = kfold_cv_p4(X_p4, y_p4_sc, X_raw_p4_train, y_p4_raw, 0.30, 10.0, 10)
    mse_base = np.mean((oof_base - y_p4_sc)**2)
    print(f"Baseline (bw=0.30, a=10, nc=10): P4 CV = {mse_base:.6f}")

    best_mse = mse_base
    best_cfg = {'bw': 0.30, 'alpha': 10.0, 'pls_nc': 10, 'frame': 150}

    # Coarse sweep
    configs = []
    for bw in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]:
        for alpha in [1.0, 5.0, 10.0, 20.0, 50.0]:
            for pls_nc in [3, 5, 8, 10]:
                configs.append((bw, alpha, pls_nc))

    t_sweep = time.time()
    for i, (bw, alpha, pls_nc) in enumerate(configs):
        oof = kfold_cv_p4(X_p4, y_p4_sc, X_raw_p4_train, y_p4_raw, bw, alpha, pls_nc)
        mse = np.mean((oof - y_p4_sc)**2)
        if mse < best_mse:
            best_mse = mse
            best_cfg = {'bw': bw, 'alpha': alpha, 'pls_nc': pls_nc, 'frame': 150}
            print(f"  NEW BEST: bw={bw}, a={alpha}, nc={pls_nc}: {mse:.6f} ({(1-mse/mse_base)*100:+.1f}%)")
        if (i+1) % 40 == 0:
            print(f"  ... {i+1}/{len(configs)} configs tested ({time.time()-t_sweep:.1f}s)")

    print(f"\nBest config: {best_cfg}")
    print(f"Best CV: {best_mse:.6f} ({(1-best_mse/mse_base)*100:+.2f}% vs baseline)")
    print(f"Sweep time: {time.time()-t_sweep:.1f}s")

    bw_best = best_cfg['bw']
    alpha_best = best_cfg['alpha']
    pls_nc_best = best_cfg['pls_nc']

    # ================================================================
    # PHASE 2: Frame sweep with best hyperparams
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Frame Sweep")
    print("=" * 70)

    frame_results = {}
    for frame in FRAMES:
        X_p4_f = feat_cache[frame]['train'][p4_tr_mask]
        oof = kfold_cv_p4(X_p4_f, y_p4_sc, X_raw_p4_train, y_p4_raw,
                           bw_best, alpha_best, pls_nc_best)
        mse = np.mean((oof - y_p4_sc)**2)
        frame_results[frame] = mse
        delta = (1 - mse/mse_base)*100
        print(f"  Frame {frame}: P4 CV = {mse:.6f} ({delta:+.2f}%)")

    best_frame = min(frame_results, key=frame_results.get)
    print(f"Best frame: {best_frame}")

    # ================================================================
    # PHASE 3: Multi-frame ensemble
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Multi-frame Ensemble")
    print("=" * 70)

    sorted_frames = sorted(frame_results.items(), key=lambda x: x[1])

    for n_combo in [2, 3, 4, 5]:
        combo_frames = [f for f, _ in sorted_frames[:n_combo]]
        oofs = []
        for frame in combo_frames:
            X_p4_f = feat_cache[frame]['train'][p4_tr_mask]
            oof = kfold_cv_p4(X_p4_f, y_p4_sc, X_raw_p4_train, y_p4_raw,
                               bw_best, alpha_best, pls_nc_best)
            oofs.append(oof)
        avg_oof = np.mean(oofs, axis=0)
        mse = np.mean((avg_oof - y_p4_sc)**2)
        delta = (1 - mse/mse_base)*100
        print(f"  Top-{n_combo} frames {combo_frames}: CV = {mse:.6f} ({delta:+.2f}%)")

    # ================================================================
    # PHASE 4: Alternative models
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Alternative Models")
    print("=" * 70)

    X_p4_bf = feat_cache[best_frame]['train'][p4_tr_mask]

    model_fns = {
        'knn_5': lambda: KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'knn_10': lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'),
        'knn_15': lambda: KNeighborsRegressor(n_neighbors=15, weights='distance'),
        'knn_20': lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'),
        'rf_small': lambda: RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42),
        'rf_medium': lambda: RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42),
        'gbr': lambda: GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, min_samples_leaf=5, random_state=42),
        'lgb_cons': lambda: lgb.LGBMRegressor(n_estimators=60, num_leaves=6, learning_rate=0.03, min_child_samples=10, reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbose=-1),
        'lgb_mod': lambda: lgb.LGBMRegressor(n_estimators=80, num_leaves=8, learning_rate=0.05, min_child_samples=5, reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1),
        'ridge_1': lambda: Ridge(alpha=1.0),
        'ridge_50': lambda: Ridge(alpha=50.0),
        'ridge_100': lambda: Ridge(alpha=100.0),
    }

    alt_results = {}
    for name, fn in model_fns.items():
        oof = kfold_cv_alt_model(X_p4_bf, y_p4_sc, X_raw_p4_train, y_p4_raw, fn, pls_nc=pls_nc_best)
        mse = np.mean((oof - y_p4_sc)**2)
        alt_results[name] = (mse, fn)
        delta = (1 - mse/mse_base)*100
        print(f"  {name}: CV = {mse:.6f} ({delta:+.2f}%)")

    best_alt_name = min(alt_results, key=lambda x: alt_results[x][0])
    best_alt_mse = alt_results[best_alt_name][0]
    print(f"Best alt model: {best_alt_name} (CV = {best_alt_mse:.6f})")

    # ================================================================
    # PHASE 5: Cross-player transfer
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Cross-Player Transfer")
    print("=" * 70)

    X_all_bf = feat_cache[best_frame]['train']
    cp_results = {}

    for bw_cp in [0.30, 0.50, 0.60, 0.80]:
        for upw in [2.0, 3.0, 5.0]:
            oof = kfold_cv_cross_player(X_all_bf, y_angle_sc, pids_train,
                                         train_data['X_raw'], y_angle_raw,
                                         bw_cp, alpha_best, pls_nc_best, upweight=upw)
            mse = np.mean((oof - y_p4_sc)**2)
            cp_results[(bw_cp, upw)] = mse
            delta = (1 - mse/mse_base)*100
            print(f"  bw={bw_cp}, upw={upw}: CV = {mse:.6f} ({delta:+.2f}%)")

    best_cp = min(cp_results, key=cp_results.get)
    print(f"Best cross-player: bw={best_cp[0]}, upw={best_cp[1]} (CV = {cp_results[best_cp]:.6f})")

    # ================================================================
    # PHASE 6: Collect best approaches and generate test predictions
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: Generate Test Predictions")
    print("=" * 70)

    approaches = {}

    # A) Best hyperparams at best frame (locally weighted Ridge)
    print("  Generating approach A: best_hyper_frame...")
    test_a = predict_test(feat_cache[best_frame]['train'][p4_tr_mask], y_p4_sc,
                           X_raw_p4_train, y_p4_raw,
                           feat_cache[best_frame]['test'][p4_te_mask],
                           X_raw_p4_test, bw_best, alpha_best, pls_nc_best)
    oof_a = kfold_cv_p4(feat_cache[best_frame]['train'][p4_tr_mask], y_p4_sc,
                         X_raw_p4_train, y_p4_raw, bw_best, alpha_best, pls_nc_best)
    mse_a = np.mean((oof_a - y_p4_sc)**2)
    approaches['best_hyper'] = {'oof': oof_a, 'test': test_a, 'mse': mse_a}

    # B) Multi-frame top-3
    print("  Generating approach B: multiframe_top3...")
    top3 = [f for f, _ in sorted_frames[:3]]
    oofs_b, tests_b = [], []
    for frame in top3:
        oof_f = kfold_cv_p4(feat_cache[frame]['train'][p4_tr_mask], y_p4_sc,
                             X_raw_p4_train, y_p4_raw, bw_best, alpha_best, pls_nc_best)
        test_f = predict_test(feat_cache[frame]['train'][p4_tr_mask], y_p4_sc,
                               X_raw_p4_train, y_p4_raw,
                               feat_cache[frame]['test'][p4_te_mask],
                               X_raw_p4_test, bw_best, alpha_best, pls_nc_best)
        oofs_b.append(oof_f)
        tests_b.append(test_f)
    oof_b = np.mean(oofs_b, axis=0)
    test_b = np.mean(tests_b, axis=0)
    mse_b = np.mean((oof_b - y_p4_sc)**2)
    approaches['multiframe3'] = {'oof': oof_b, 'test': test_b, 'mse': mse_b}

    # C) Best alternative model
    print(f"  Generating approach C: {best_alt_name}...")
    oof_c = kfold_cv_alt_model(X_p4_bf, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                alt_results[best_alt_name][1], pls_nc=pls_nc_best)
    test_c = predict_test_alt_model(X_p4_bf, y_p4_sc, X_raw_p4_train, y_p4_raw,
                                     feat_cache[best_frame]['test'][p4_te_mask],
                                     X_raw_p4_test, alt_results[best_alt_name][1], pls_nc=pls_nc_best)
    mse_c = np.mean((oof_c - y_p4_sc)**2)
    approaches[f'alt_{best_alt_name}'] = {'oof': oof_c, 'test': test_c, 'mse': mse_c}

    # D) Cross-player transfer
    print("  Generating approach D: cross_player...")
    oof_d = kfold_cv_cross_player(X_all_bf, y_angle_sc, pids_train,
                                   train_data['X_raw'], y_angle_raw,
                                   best_cp[0], alpha_best, pls_nc_best, upweight=best_cp[1])
    test_d = predict_test_cross_player(X_all_bf, y_angle_sc, pids_train,
                                        train_data['X_raw'], y_angle_raw,
                                        feat_cache[best_frame]['test'][p4_te_mask],
                                        X_raw_p4_test, best_cp[0], alpha_best, pls_nc_best,
                                        upweight=best_cp[1])
    mse_d = np.mean((oof_d - y_p4_sc)**2)
    approaches['cross_player'] = {'oof': oof_d, 'test': test_d, 'mse': mse_d}

    # E) Baseline
    print("  Generating approach E: baseline...")
    oof_e = kfold_cv_p4(feat_cache[150]['train'][p4_tr_mask], y_p4_sc,
                         X_raw_p4_train, y_p4_raw, 0.30, 10.0, 10)
    test_e = predict_test(feat_cache[150]['train'][p4_tr_mask], y_p4_sc,
                           X_raw_p4_train, y_p4_raw,
                           feat_cache[150]['test'][p4_te_mask],
                           X_raw_p4_test, 0.30, 10.0, 10)
    mse_e = np.mean((oof_e - y_p4_sc)**2)
    approaches['baseline'] = {'oof': oof_e, 'test': test_e, 'mse': mse_e}

    # Ensembles of approaches
    print("  Generating ensembles...")
    sorted_approaches = sorted(approaches.items(), key=lambda x: x[1]['mse'])

    # Top-2 ensemble
    top2_names = [sorted_approaches[0][0], sorted_approaches[1][0]]
    oof_t2 = np.mean([approaches[n]['oof'] for n in top2_names], axis=0)
    test_t2 = np.mean([approaches[n]['test'] for n in top2_names], axis=0)
    mse_t2 = np.mean((oof_t2 - y_p4_sc)**2)
    approaches['top2_ens'] = {'oof': oof_t2, 'test': test_t2, 'mse': mse_t2}

    # Top-3 ensemble
    top3_names = [sorted_approaches[0][0], sorted_approaches[1][0], sorted_approaches[2][0]]
    oof_t3 = np.mean([approaches[n]['oof'] for n in top3_names], axis=0)
    test_t3 = np.mean([approaches[n]['test'] for n in top3_names], axis=0)
    mse_t3 = np.mean((oof_t3 - y_p4_sc)**2)
    approaches['top3_ens'] = {'oof': oof_t3, 'test': test_t3, 'mse': mse_t3}

    # All-non-baseline ensemble
    nb_names = [n for n, _ in sorted_approaches if n != 'baseline']
    oof_nb = np.mean([approaches[n]['oof'] for n in nb_names], axis=0)
    test_nb = np.mean([approaches[n]['test'] for n in nb_names], axis=0)
    mse_nb = np.mean((oof_nb - y_p4_sc)**2)
    approaches['all_ens'] = {'oof': oof_nb, 'test': test_nb, 'mse': mse_nb}

    # Print final ranking
    print("\n" + "=" * 70)
    print("FINAL RANKING (5-fold CV)")
    print("=" * 70)

    final_ranking = sorted(approaches.items(), key=lambda x: x[1]['mse'])
    for name, data in final_ranking:
        delta = (1 - data['mse']/mse_base)*100
        cal = np.std(data['test']) / np.std(y_p4_sc) if 'test' in data else 0
        print(f"  {name:20s}: CV = {data['mse']:.6f} ({delta:+.2f}%), pred_std/train_std = {cal:.4f}")

    # ================================================================
    # PHASE 7: Also check P5 angle improvement
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 7: P5 Angle Check")
    print("=" * 70)

    p5_tr_mask = pids_train == 5
    p5_te_mask = pids_test == 5
    y_p5_sc = y_angle_sc[p5_tr_mask]
    y_p5_raw = y_angle_raw[p5_tr_mask]
    X_raw_p5_train = train_data['X_raw'][p5_tr_mask]
    X_raw_p5_test = test_data['X_raw'][p5_te_mask]

    X_p5_bf = feat_cache[best_frame]['train'][p5_tr_mask]
    oof_p5_base = kfold_cv_p4(feat_cache[150]['train'][p5_tr_mask], y_p5_sc,
                               X_raw_p5_train, y_p5_raw, 0.30, 10.0, 10)
    mse_p5_base = np.mean((oof_p5_base - y_p5_sc)**2)

    oof_p5_opt = kfold_cv_p4(X_p5_bf, y_p5_sc, X_raw_p5_train, y_p5_raw,
                              bw_best, alpha_best, pls_nc_best)
    mse_p5_opt = np.mean((oof_p5_opt - y_p5_sc)**2)

    print(f"P5 angle baseline CV: {mse_p5_base:.6f}")
    print(f"P5 angle with P4-opt params CV: {mse_p5_opt:.6f} ({(1-mse_p5_opt/mse_p5_base)*100:+.2f}%)")

    # Also sweep P5 separately
    p5_frame_results = {}
    for frame in FRAMES:
        oof = kfold_cv_p4(feat_cache[frame]['train'][p5_tr_mask], y_p5_sc,
                           X_raw_p5_train, y_p5_raw, bw_best, alpha_best, pls_nc_best)
        mse = np.mean((oof - y_p5_sc)**2)
        p5_frame_results[frame] = mse

    p5_best_frame = min(p5_frame_results, key=p5_frame_results.get)
    print(f"P5 best frame: {p5_best_frame} (CV = {p5_frame_results[p5_best_frame]:.6f}, {(1-p5_frame_results[p5_best_frame]/mse_p5_base)*100:+.2f}%)")

    # Generate P5 test predictions
    test_p5 = predict_test(feat_cache[p5_best_frame]['train'][p5_tr_mask], y_p5_sc,
                            X_raw_p5_train, y_p5_raw,
                            feat_cache[p5_best_frame]['test'][p5_te_mask],
                            X_raw_p5_test, bw_best, alpha_best, pls_nc_best)

    p5_improved = p5_frame_results[p5_best_frame] < mse_p5_base

    # ================================================================
    # PHASE 8: Generate submissions
    # ================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    base_sub = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")
    test_ids = test_data['ids']
    p4_test_ids = test_ids[p4_te_mask]
    p5_test_ids = test_ids[p5_te_mask]

    submission_details = []

    # Take top 3 non-baseline, non-ensemble approaches + top ensemble
    individual_approaches = [(n, d) for n, d in final_ranking if n not in ('baseline', 'top2_ens', 'top3_ens', 'all_ens')][:3]
    ensemble_approaches = [(n, d) for n, d in final_ranking if 'ens' in n][:1]

    top_to_submit = individual_approaches + ensemble_approaches

    for cand_name, cand_data in top_to_submit:
        cand_test = cand_data['test']
        cand_mse = cand_data['mse']

        # Sub A: P4 angle splice only
        sub_num = get_next_submission_number()
        sub = base_sub.copy()
        for j, tid in enumerate(p4_test_ids):
            idx = sub[sub['id'] == tid].index[0]
            sub.loc[idx, 'scaled_angle'] = cand_test[j]
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

        desc = (f"Sub {sub_num}: P4 angle splice [{cand_name}] into Sub3411. "
                f"P4 CV={cand_mse:.6f} vs baseline {mse_base:.6f} "
                f"({(1-cand_mse/mse_base)*100:+.2f}%). "
                f"Config: bw={bw_best}, alpha={alpha_best}, pls_nc={pls_nc_best}, frame={best_frame}.")
        print(f"\n{desc}")
        submission_details.append(desc)

        # Sub B: P4+P5 angle splice (only if P5 improved)
        if p5_improved:
            sub_num2 = get_next_submission_number()
            sub2 = base_sub.copy()
            for j, tid in enumerate(p4_test_ids):
                idx2 = sub2[sub2['id'] == tid].index[0]
                sub2.loc[idx2, 'scaled_angle'] = cand_test[j]
            for j, tid in enumerate(p5_test_ids):
                idx2 = sub2[sub2['id'] == tid].index[0]
                sub2.loc[idx2, 'scaled_angle'] = test_p5[j]
            sub2.to_csv(SUBMISSION_DIR / f"submission_{sub_num2}.csv", index=False)
            desc2 = (f"Sub {sub_num2}: P4+P5 angle splice [{cand_name}+P5opt] into Sub3411. "
                     f"P4 CV={cand_mse:.6f}, P5 CV={p5_frame_results[p5_best_frame]:.6f}")
            print(f"{desc2}")
            submission_details.append(desc2)

    # Blend subs: take best candidate and blend at different weights with Sub 3411
    best_cand_name, best_cand_data = top_to_submit[0]
    best_cand_test = best_cand_data['test']

    for blend_w in [0.3, 0.5, 0.7, 1.0]:
        sub_num = get_next_submission_number()
        sub = base_sub.copy()
        for j, tid in enumerate(p4_test_ids):
            idx = sub[sub['id'] == tid].index[0]
            old_val = sub.loc[idx, 'scaled_angle']
            new_val = blend_w * best_cand_test[j] + (1 - blend_w) * old_val
            sub.loc[idx, 'scaled_angle'] = new_val
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        desc = (f"Sub {sub_num}: P4 angle {int(blend_w*100)}% {best_cand_name} + "
                f"{int((1-blend_w)*100)}% Sub3411 angle.")
        print(f"\n{desc}")
        submission_details.append(desc)

    # Calibration analysis
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)

    sub_3411_p4_angle = base_sub.set_index('id').loc[p4_test_ids, 'scaled_angle'].values
    print(f"  Sub3411 P4: mean={np.mean(sub_3411_p4_angle):.6f}, std={np.std(sub_3411_p4_angle):.6f}")
    print(f"  P4 train:   mean={np.mean(y_p4_sc):.6f}, std={np.std(y_p4_sc):.6f}")
    print(f"  Sub3411 calibration: {np.std(sub_3411_p4_angle)/np.std(y_p4_sc):.4f}")

    for name, data in top_to_submit:
        cal = np.std(data['test']) / np.std(y_p4_sc)
        print(f"  {name}: std={np.std(data['test']):.6f}, calibration={cal:.4f}")

    # ================================================================
    # SAVE RESEARCH
    # ================================================================
    elapsed = time.time() - t0

    research_lines = [
        "# P4 Angle Fix Results - 2026-02-19\n",
        "## Context",
        f"- P4 has highest angle variance (std={np.std(y_p4_raw):.4f} raw, {np.std(y_p4_sc):.4f} scaled)",
        f"- Sub 3411 P4 angle calibration: {np.std(sub_3411_p4_angle)/np.std(y_p4_sc):.4f}",
        f"- P4 has {n_p4_train} training samples, {n_p4_test} test samples\n",
        "## Phase 1: Hyperparameter Sweep",
        f"- Baseline (bw=0.30, a=10, nc=10): P4 CV = {mse_base:.6f}",
        f"- Best config: bw={bw_best}, alpha={alpha_best}, PLS nc={pls_nc_best}",
        f"- Best CV: {best_mse:.6f} ({(1-best_mse/mse_base)*100:+.2f}%)\n",
        "## Phase 2: Frame Sweep",
    ]
    for frame in sorted(frame_results.keys()):
        delta = (1 - frame_results[frame]/mse_base)*100
        marker = " <-- BEST" if frame == best_frame else ""
        research_lines.append(f"- Frame {frame}: {frame_results[frame]:.6f} ({delta:+.2f}%){marker}")

    research_lines.extend([
        f"\n## Phase 4: Alternative Models",
    ])
    for name, (mse, _) in sorted(alt_results.items(), key=lambda x: x[1][0]):
        delta = (1 - mse/mse_base)*100
        research_lines.append(f"- {name}: {mse:.6f} ({delta:+.2f}%)")

    research_lines.extend([
        f"\n## Phase 5: Cross-Player Transfer",
    ])
    for (bw, upw), mse in sorted(cp_results.items(), key=lambda x: x[1]):
        delta = (1 - mse/mse_base)*100
        research_lines.append(f"- bw={bw}, upweight={upw}: {mse:.6f} ({delta:+.2f}%)")

    research_lines.extend([
        f"\n## Final Ranking",
    ])
    for name, data in final_ranking:
        delta = (1 - data['mse']/mse_base)*100
        research_lines.append(f"- {name}: CV = {data['mse']:.6f} ({delta:+.2f}%)")

    research_lines.extend([
        f"\n## P5 Angle",
        f"- Baseline CV: {mse_p5_base:.6f}",
        f"- Best frame ({p5_best_frame}) CV: {p5_frame_results[p5_best_frame]:.6f} ({(1-p5_frame_results[p5_best_frame]/mse_p5_base)*100:+.2f}%)",
        f"- P5 improved: {p5_improved}\n",
        "## Submissions Generated",
    ])
    for d in submission_details:
        research_lines.append(f"- {d}")

    research_lines.append(f"\n## Runtime: {elapsed:.1f}s")

    research_path = PROJECT_DIR / "Research" / "P4_ANGLE_FIX_RESULTS_2026-02-19.md"
    research_path.write_text("\n".join(research_lines))
    print(f"\nResearch saved to {research_path}")
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
