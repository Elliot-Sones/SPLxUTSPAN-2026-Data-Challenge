"""
Adaptive Bandwidth + Stacked Residual Correction

Approach A: Per-player bandwidth optimization
- Sweep bandwidth_quantile from 0.15 to 0.45 for each player independently
- Pick best bandwidth per player via LOO within that player
- Compare to fixed bandwidth_quantile=0.3

Approach B: Stacked residual correction
- Run core pipeline LOO to get OOF predictions + residuals
- Train second-stage Ridge on residuals using DIFFERENT features
  (joint angles, release frame timing, per-player mean residual)
- Nested LOO to avoid data leakage
- Add predicted residual correction to base predictions
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

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


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


# ==============================================================
# DATA LOADING
# ==============================================================

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


# ==============================================================
# FEATURE EXTRACTION (same as per_example_pipeline.py)
# ==============================================================

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


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract compact feature set at a specific frame."""
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

    return np.array(feats, dtype=np.float32)


def extract_joint_angle_features(ts_3d, ts_hr, kp_index, frame):
    """Extract 10 body-proportion-invariant joint angle features."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    rw_i = kp_index.get('right_wrist')
    re_i = kp_index.get('right_elbow')
    rs_i = kp_index.get('right_shoulder')
    rh_i = kp_index.get('right_hip')
    lh_i = kp_index.get('left_hip')
    rk_i = kp_index.get('right_knee')
    lk_i = kp_index.get('left_knee')
    ra_i = kp_index.get('right_ankle')
    la_i = kp_index.get('left_ankle')
    neck_i = kp_index.get('neck')
    mh_i = kp_index.get('mid_hip')
    ls_i = kp_index.get('left_shoulder')

    # 1. Shoulder elevation angle
    if all(x is not None for x in [rs_i, rh_i, re_i]):
        v1 = ts_3d[f, rs_i] - ts_3d[f, rh_i]
        v2 = ts_3d[f, re_i] - ts_3d[f, rs_i]
        feats.append(_angle_between(v1, v2))
    else:
        feats.append(90.0)

    # 2. Trunk forward lean
    if neck_i is not None and mh_i is not None:
        trunk = ts_hr[f, neck_i] - ts_hr[f, mh_i]
        vertical = np.array([0, 0, 1], dtype=np.float32)
        feats.append(_angle_between(trunk, vertical))
        # 3. Trunk lateral lean
        feats.append(np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8)))
    else:
        feats.append(0.0)
        feats.append(0.0)

    # 4. Right knee flexion
    if all(x is not None for x in [rk_i, rh_i, ra_i]):
        v1 = ts_3d[f, rh_i] - ts_3d[f, rk_i]
        v2 = ts_3d[f, ra_i] - ts_3d[f, rk_i]
        feats.append(_angle_between(v1, v2))
    else:
        feats.append(90.0)

    # 5. Left knee flexion
    if all(x is not None for x in [lk_i, lh_i, la_i]):
        v1 = ts_3d[f, lh_i] - ts_3d[f, lk_i]
        v2 = ts_3d[f, la_i] - ts_3d[f, lk_i]
        feats.append(_angle_between(v1, v2))
    else:
        feats.append(90.0)

    # 6. Wrist deviation (forearm angle with vertical)
    if re_i is not None and rw_i is not None:
        forearm = ts_3d[f, rw_i] - ts_3d[f, re_i]
        vertical = np.array([0, 0, 1], dtype=np.float32)
        feats.append(_angle_between(forearm, vertical))
    else:
        feats.append(90.0)

    # 7. Arm line angle with hoop direction
    if rs_i is not None and rw_i is not None:
        arm_line = ts_hr[f, rw_i] - ts_hr[f, rs_i]
        hoop_dir = np.array([1, 0, 0.5], dtype=np.float32)
        feats.append(_angle_between(arm_line, hoop_dir))
    else:
        feats.append(90.0)

    # 8. Shoulder rotation
    if rs_i is not None and ls_i is not None:
        shoulder_line = ts_hr[f, rs_i] - ts_hr[f, ls_i]
        forward = np.array([1, 0, 0], dtype=np.float32)
        feats.append(_angle_between(shoulder_line, forward))
    else:
        feats.append(90.0)

    # 9. Elbow angle (upper arm to forearm)
    if all(x is not None for x in [rs_i, re_i, rw_i]):
        ua = ts_3d[f, rs_i] - ts_3d[f, re_i]
        fa = ts_3d[f, rw_i] - ts_3d[f, re_i]
        feats.append(_angle_between(ua, fa))
    else:
        feats.append(90.0)

    # 10. Hip rotation
    if rh_i is not None and lh_i is not None:
        hip_line = ts_hr[f, rh_i] - ts_hr[f, lh_i]
        forward = np.array([1, 0, 0], dtype=np.float32)
        feats.append(_angle_between(hip_line, forward))
    else:
        feats.append(90.0)

    return np.array(feats, dtype=np.float32)


def extract_all_features(data, target):
    """Extract features for all shots."""
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']

    all_feats = []
    all_ja_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
        ja_feats = extract_joint_angle_features(ts_3d, ts_hr, kp_index, frame)
        all_ja_feats.append(ja_feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_ja = np.array(all_ja_feats, dtype=np.float32)
    X_ja = np.nan_to_num(X_ja, nan=0.0, posinf=0.0, neginf=0.0)
    return X, X_ja, np.array(release_frames)


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test, X_raw_train, X_raw_test):
    """Add PLS components to feature matrices."""
    unique_pids = sorted(np.unique(pids_train))
    max_nc = 15
    pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        n_p = tr_mask.sum()

        scaler = StandardScaler()
        raw_tr = scaler.fit_transform(X_raw_train[tr_mask])
        raw_te = scaler.transform(X_raw_test[te_mask])

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15]:
            if c > nc:
                break
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(raw_tr):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_tr[tr_idx], y_raw_train[tr_mask][tr_idx])
                pred = pls.predict(raw_tr[val_idx]).flatten()
                mses.append(np.mean((pred - y_raw_train[tr_mask][val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_tr, y_raw_train[tr_mask])
        pls_train[tr_mask, :best_nc] = pls.transform(raw_tr)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)

    return np.hstack([X_train, pls_train]), np.hstack([X_test, pls_test])


# ==============================================================
# APPROACH A: ADAPTIVE PER-PLAYER BANDWIDTH
# ==============================================================

def locally_weighted_loo_player(X_s, y, sigma, alpha=10.0):
    """LOO within a single player group, given sigma."""
    n = len(X_s)
    D = cdist(X_s, X_s, metric='euclidean')
    oof = np.zeros(n)
    for i in range(n):
        weights = np.exp(-D[i, :] ** 2 / (2 * sigma ** 2))
        weights[i] = 0  # leave out self
        if weights.sum() < 1e-10:
            oof[i] = np.mean(y)
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_s, y, sample_weight=weights)
        oof[i] = ridge.predict(X_s[i:i+1])[0]
    return oof


def adaptive_bandwidth_experiment(X_train_aug, y_target, pids_train,
                                   X_test_aug, pids_test, alpha=10.0):
    """Test per-player optimal bandwidth vs fixed bandwidth."""
    unique_pids = sorted(np.unique(pids_train))
    bw_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    # Fixed bandwidth baseline (bw=0.3)
    oof_fixed = np.zeros(len(X_train_aug))
    test_fixed = np.zeros(len(X_test_aug))

    # Per-player adaptive
    oof_adaptive = np.zeros(len(X_train_aug))
    test_adaptive = np.zeros(len(X_test_aug))

    # Store per-player results
    player_results = {}

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train_aug[tr_mask]
        y_tr = y_target[tr_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test_aug[te_mask]) if te_mask.sum() > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        all_dists = D_tr[np.triu_indices(n_tr, k=1)]

        # Find best bandwidth for this player via LOO
        best_bw = 0.30
        best_mse = float('inf')
        bw_mses = {}

        for bw in bw_values:
            if len(all_dists) > 0:
                sigma = np.quantile(all_dists, bw)
                sigma = max(sigma, 1e-6)
            else:
                sigma = 1.0

            oof_p = locally_weighted_loo_player(X_tr_s, y_tr, sigma, alpha)
            mse = np.mean((oof_p - y_tr) ** 2)
            bw_mses[bw] = mse

            if mse < best_mse:
                best_mse = mse
                best_bw = bw

        player_results[pid] = {
            'n_samples': n_tr,
            'best_bw': best_bw,
            'bw_mses': bw_mses,
            'best_mse': best_mse,
            'fixed_mse': bw_mses.get(0.30, best_mse),
        }

        # Compute OOF with best bandwidth
        sigma_best = np.quantile(all_dists, best_bw) if len(all_dists) > 0 else 1.0
        sigma_best = max(sigma_best, 1e-6)
        oof_adaptive[tr_indices] = locally_weighted_loo_player(X_tr_s, y_tr, sigma_best, alpha)

        # Compute OOF with fixed bw=0.3
        sigma_fixed = np.quantile(all_dists, 0.30) if len(all_dists) > 0 else 1.0
        sigma_fixed = max(sigma_fixed, 1e-6)
        oof_fixed[tr_indices] = locally_weighted_loo_player(X_tr_s, y_tr, sigma_fixed, alpha)

        # Test predictions with best and fixed bandwidths
        D_te = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te_s) > 0 else np.zeros((0, n_tr))

        for j in range(len(X_te_s)):
            dists = D_te[j, :]

            # Fixed
            w_f = np.exp(-dists ** 2 / (2 * sigma_fixed ** 2))
            if w_f.sum() > 1e-10:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=w_f)
                test_fixed[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]
            else:
                test_fixed[te_indices[j]] = np.mean(y_tr)

            # Adaptive
            w_a = np.exp(-dists ** 2 / (2 * sigma_best ** 2))
            if w_a.sum() > 1e-10:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=w_a)
                test_adaptive[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]
            else:
                test_adaptive[te_indices[j]] = np.mean(y_tr)

    return oof_fixed, test_fixed, oof_adaptive, test_adaptive, player_results


# ==============================================================
# APPROACH B: STACKED RESIDUAL CORRECTION
# ==============================================================

def stacked_residual_correction(X_train_aug, y_target, pids_train,
                                 X_test_aug, pids_test,
                                 X_ja_train, X_ja_test,
                                 release_frames_train, release_frames_test,
                                 alpha=10.0, bw=0.3):
    """
    Two-stage pipeline:
    1. Run base LOO to get OOF predictions and residuals
    2. Train a second-stage model on residuals using different features
    3. Use NESTED LOO to avoid data leakage in residual model
    """
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train_aug)
    n_test = len(X_test_aug)

    # Stage 1: Get base OOF predictions (same as core pipeline)
    oof_base = np.zeros(n_train)
    test_base = np.zeros(n_test)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train_aug[tr_mask]
        y_tr = y_target[tr_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_test_aug[te_mask]) if te_mask.sum() > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        all_dists = D_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        # LOO for base predictions
        for i in range(n_tr):
            weights = np.exp(-D_tr[i, :] ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_base[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_base[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test predictions for base
        D_te = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te_s) > 0 else np.zeros((0, n_tr))
        for j in range(len(X_te_s)):
            dists = D_te[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() > 1e-10:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=weights)
                test_base[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]
            else:
                test_base[te_indices[j]] = np.mean(y_tr)

    # Compute residuals
    residuals = y_target - oof_base
    base_mse = np.mean((oof_base - y_target) ** 2)
    print(f"    Base LOO MSE: {base_mse:.6f}")
    print(f"    Residual stats: mean={residuals.mean():.6f}, std={residuals.std():.6f}")

    # Stage 2: Build residual features
    # Features for residual model:
    # - Joint angles (10 features)
    # - Release frame timing (1 feature)
    # - Per-player ID encoded as mean residual from other samples (nested)
    # We need to be careful about data leakage here

    # Build residual feature matrix
    # Joint angles + release frame
    X_res_train = np.column_stack([X_ja_train, release_frames_train])
    X_res_test = np.column_stack([X_ja_test, release_frames_test])

    # Stage 2: Nested LOO for residual correction
    # For each sample i:
    #   1. Leave out sample i
    #   2. Compute per-player mean residual from remaining samples
    #   3. Train residual model on remaining samples
    #   4. Predict residual correction for sample i
    oof_correction = np.zeros(n_train)
    test_correction = np.zeros(n_test)

    # Try multiple Ridge alphas for residual model
    for res_alpha in [1.0, 10.0, 50.0, 100.0, 500.0]:
        oof_corr_tmp = np.zeros(n_train)

        for pid in unique_pids:
            mask = pids_train == pid
            indices = np.where(mask)[0]
            n_p = len(indices)

            X_res_p = X_res_train[indices]
            res_p = residuals[indices]

            scaler_res = StandardScaler()

            # LOO within player for residual model
            for i in range(n_p):
                # Leave one out
                loo_mask = np.ones(n_p, dtype=bool)
                loo_mask[i] = False

                X_loo = X_res_p[loo_mask]
                r_loo = res_p[loo_mask]

                # Add per-player mean residual as feature
                player_mean_res = np.mean(r_loo)
                X_loo_aug = np.column_stack([X_loo, np.full(len(X_loo), player_mean_res)])
                X_pred = np.column_stack([X_res_p[i:i+1], np.array([[player_mean_res]])])

                # Scale
                sc = StandardScaler()
                X_loo_s = sc.fit_transform(X_loo_aug)
                X_pred_s = sc.transform(X_pred)

                # Train residual model
                ridge_res = Ridge(alpha=res_alpha)
                ridge_res.fit(X_loo_s, r_loo)
                oof_corr_tmp[indices[i]] = ridge_res.predict(X_pred_s)[0]

        # Evaluate corrected predictions
        oof_corrected = oof_base + oof_corr_tmp
        corrected_mse = np.mean((oof_corrected - y_target) ** 2)
        improvement = (corrected_mse - base_mse) / base_mse * 100
        print(f"    Residual alpha={res_alpha:.0f}: corrected MSE={corrected_mse:.6f} ({improvement:+.2f}%)")

        if res_alpha == 50.0 or (corrected_mse < base_mse and res_alpha == 1.0):
            # Save the best one we find
            pass

    # Run final version with multiple alphas and pick best
    best_alpha = None
    best_corr_mse = base_mse
    best_oof_corr = np.zeros(n_train)

    for res_alpha in [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        oof_corr_tmp = np.zeros(n_train)

        for pid in unique_pids:
            mask = pids_train == pid
            indices = np.where(mask)[0]
            n_p = len(indices)
            X_res_p = X_res_train[indices]
            res_p = residuals[indices]

            for i in range(n_p):
                loo_mask = np.ones(n_p, dtype=bool)
                loo_mask[i] = False
                X_loo = X_res_p[loo_mask]
                r_loo = res_p[loo_mask]
                player_mean_res = np.mean(r_loo)
                X_loo_aug = np.column_stack([X_loo, np.full(len(X_loo), player_mean_res)])
                X_pred = np.column_stack([X_res_p[i:i+1], np.array([[player_mean_res]])])
                sc = StandardScaler()
                X_loo_s = sc.fit_transform(X_loo_aug)
                X_pred_s = sc.transform(X_pred)
                ridge_res = Ridge(alpha=res_alpha)
                ridge_res.fit(X_loo_s, r_loo)
                oof_corr_tmp[indices[i]] = ridge_res.predict(X_pred_s)[0]

        corrected_mse = np.mean((oof_base + oof_corr_tmp - y_target) ** 2)
        if corrected_mse < best_corr_mse:
            best_corr_mse = corrected_mse
            best_alpha = res_alpha
            best_oof_corr = oof_corr_tmp.copy()

    if best_alpha is not None:
        print(f"    Best residual alpha: {best_alpha}, MSE: {best_corr_mse:.6f}")
        oof_correction = best_oof_corr

        # Generate test corrections with best alpha
        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            if te_mask.sum() == 0:
                continue
            tr_indices = np.where(tr_mask)[0]
            te_indices = np.where(te_mask)[0]

            X_res_tr = X_res_train[tr_indices]
            res_tr = residuals[tr_indices]
            player_mean_res = np.mean(res_tr)

            X_tr_aug = np.column_stack([X_res_tr, np.full(len(X_res_tr), player_mean_res)])
            X_te_res = X_res_test[te_indices]
            X_te_aug = np.column_stack([X_te_res, np.full(len(X_te_res), player_mean_res)])

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_aug)
            X_te_s = sc.transform(X_te_aug)

            ridge_res = Ridge(alpha=best_alpha)
            ridge_res.fit(X_tr_s, res_tr)
            test_correction[te_indices] = ridge_res.predict(X_te_s)
    else:
        print(f"    No improvement from residual correction")

    # Also try damped corrections (conservative blend)
    oof_corrected_full = oof_base + oof_correction
    test_corrected_full = test_base + test_correction

    best_damp = 0.0
    best_damp_mse = base_mse
    for damp in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        oof_d = oof_base + damp * oof_correction
        mse_d = np.mean((oof_d - y_target) ** 2)
        if mse_d < best_damp_mse:
            best_damp_mse = mse_d
            best_damp = damp

    print(f"    Best damping factor: {best_damp:.1f}, MSE: {best_damp_mse:.6f}")
    oof_final = oof_base + best_damp * oof_correction
    test_final = test_base + best_damp * test_correction

    return {
        'oof_base': oof_base,
        'test_base': test_base,
        'oof_corrected': oof_final,
        'test_corrected': test_final,
        'base_mse': base_mse,
        'corrected_mse': best_damp_mse,
        'best_alpha': best_alpha,
        'best_damp': best_damp,
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ADAPTIVE BANDWIDTH + STACKED RESIDUAL CORRECTION")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # Per-target results storage
    results_a = {}  # Approach A: adaptive bandwidth
    results_b = {}  # Approach B: stacked residual

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc, X_ja_train, rf_train = extract_all_features(train_data, target)
        X_test_hc, X_ja_test, rf_test = extract_all_features(test_data, target)
        print(f"  HC features: {X_train_hc.shape[1]}, JA features: {X_ja_train.shape[1]}")

        # PLS augmentation
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Augmented features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # ---- APPROACH A: Adaptive Bandwidth ----
        print(f"\n  [A] ADAPTIVE PER-PLAYER BANDWIDTH")
        oof_fixed, test_fixed, oof_adaptive, test_adaptive, player_res = \
            adaptive_bandwidth_experiment(X_train_aug, y_target, pids_train,
                                          X_test_aug, pids_test, alpha=10.0)

        mse_fixed = np.mean((oof_fixed - y_target) ** 2)
        mse_adaptive = np.mean((oof_adaptive - y_target) ** 2)
        improvement_a = (mse_adaptive - mse_fixed) / mse_fixed * 100

        print(f"    Fixed bw=0.30 LOO MSE:     {mse_fixed:.6f}")
        print(f"    Adaptive per-player MSE:    {mse_adaptive:.6f}")
        print(f"    Improvement:                {improvement_a:+.2f}%")

        print(f"    Per-player details:")
        for pid in sorted(player_res.keys()):
            pr = player_res[pid]
            delta = (pr['best_mse'] - pr['fixed_mse']) / pr['fixed_mse'] * 100
            print(f"      Player {pid}: n={pr['n_samples']}, best_bw={pr['best_bw']:.2f}, "
                  f"fixed_mse={pr['fixed_mse']:.6f}, best_mse={pr['best_mse']:.6f} ({delta:+.2f}%)")

        results_a[target] = {
            'oof_fixed': oof_fixed,
            'test_fixed': test_fixed,
            'oof_adaptive': oof_adaptive,
            'test_adaptive': test_adaptive,
            'mse_fixed': mse_fixed,
            'mse_adaptive': mse_adaptive,
            'player_results': player_res,
        }

        # ---- APPROACH B: Stacked Residual ----
        print(f"\n  [B] STACKED RESIDUAL CORRECTION")
        res_b = stacked_residual_correction(
            X_train_aug, y_target, pids_train,
            X_test_aug, pids_test,
            X_ja_train, X_ja_test,
            rf_train, rf_test,
            alpha=10.0, bw=0.3)

        improvement_b = (res_b['corrected_mse'] - res_b['base_mse']) / res_b['base_mse'] * 100
        print(f"    Base LOO MSE:      {res_b['base_mse']:.6f}")
        print(f"    Corrected LOO MSE: {res_b['corrected_mse']:.6f}")
        print(f"    Improvement:       {improvement_b:+.2f}%")

        results_b[target] = res_b

    # ============================================================
    # OVERALL RESULTS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("OVERALL RESULTS")
    print(f"{'=' * 70}")

    print("\n  APPROACH A: Adaptive Bandwidth")
    total_fixed, total_adaptive = 0, 0
    for target in TARGETS:
        mf = results_a[target]['mse_fixed']
        ma = results_a[target]['mse_adaptive']
        total_fixed += mf
        total_adaptive += ma
        delta = (ma - mf) / mf * 100
        print(f"    {target}: fixed={mf:.6f}, adaptive={ma:.6f} ({delta:+.2f}%)")
    print(f"    MEAN: fixed={total_fixed/3:.6f}, adaptive={total_adaptive/3:.6f} "
          f"({(total_adaptive - total_fixed) / total_fixed * 100:+.2f}%)")

    print("\n  APPROACH B: Stacked Residual")
    total_base, total_corrected = 0, 0
    for target in TARGETS:
        mb = results_b[target]['base_mse']
        mc = results_b[target]['corrected_mse']
        total_base += mb
        total_corrected += mc
        delta = (mc - mb) / mb * 100
        print(f"    {target}: base={mb:.6f}, corrected={mc:.6f} ({delta:+.2f}%)")
    print(f"    MEAN: base={total_base/3:.6f}, corrected={total_corrected/3:.6f} "
          f"({(total_corrected - total_base) / total_base * 100:+.2f}%)")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Check if either approach improved
    mean_fixed = total_fixed / 3
    mean_adaptive = total_adaptive / 3
    mean_base = total_base / 3
    mean_corrected = total_corrected / 3

    # Sub A: Adaptive bandwidth standalone
    sub_num_a = get_next_submission_number()
    sub_a = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results_a['angle']['test_adaptive'],
        'scaled_depth': results_a['depth']['test_adaptive'],
        'scaled_left_right': results_a['left_right']['test_adaptive'],
    })
    sub_a.to_csv(SUBMISSION_DIR / f"submission_{sub_num_a}.csv", index=False)
    print(f"  Sub {sub_num_a}: Adaptive bandwidth standalone (LOO mean: {mean_adaptive:.6f})")

    # Sub B: Residual corrected standalone
    sub_num_b = get_next_submission_number()
    sub_b = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results_b['angle']['test_corrected'],
        'scaled_depth': results_b['depth']['test_corrected'],
        'scaled_left_right': results_b['left_right']['test_corrected'],
    })
    sub_b.to_csv(SUBMISSION_DIR / f"submission_{sub_num_b}.csv", index=False)
    print(f"  Sub {sub_num_b}: Residual corrected standalone (LOO mean: {mean_corrected:.6f})")

    # Blends with Sub 2169 (current best)
    sub_2169 = pd.read_csv(SUBMISSION_DIR / "submission_2169.csv")

    for weight, label in [(0.10, "10%"), (0.20, "20%"), (0.30, "30%")]:
        # Adaptive blend
        sub_num = get_next_submission_number()
        blended = sub_2169.copy()
        for t_col, t_key in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[t_col] = (1 - weight) * sub_2169[t_col] + weight * results_a[t_key]['test_adaptive']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {label} adaptive + {100-int(weight*100)}% Sub2169")

        # Residual blend
        sub_num = get_next_submission_number()
        blended = sub_2169.copy()
        for t_col, t_key in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[t_col] = (1 - weight) * sub_2169[t_col] + weight * results_b[t_key]['test_corrected']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {label} residual + {100-int(weight*100)}% Sub2169")

    # Best-of-both per target: use whichever approach is better per target
    sub_num = get_next_submission_number()
    sub_best = pd.DataFrame({'id': test_data['ids']})
    for t_col, t_key in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
        mse_a = results_a[t_key]['mse_adaptive']
        mse_b = results_b[t_key]['corrected_mse']
        if mse_a < mse_b:
            sub_best[t_col] = results_a[t_key]['test_adaptive']
            print(f"  {t_key}: using adaptive (MSE={mse_a:.6f} < {mse_b:.6f})")
        else:
            sub_best[t_col] = results_b[t_key]['test_corrected']
            print(f"  {t_key}: using residual (MSE={mse_b:.6f} < {mse_a:.6f})")
    sub_best.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: Best-of-both per target")

    # Blend best-of-both with Sub 2169
    for weight in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_2169.copy()
        for t_col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blended[t_col] = (1 - weight) * sub_2169[t_col] + weight * sub_best[t_col]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(weight*100)}% best-of-both + {100-int(weight*100)}% Sub2169")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
