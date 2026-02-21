"""
Player-Targeted Optimization

Competition mindset: the 3 worst player-target combinations contribute
disproportionately to total error. Instead of improving the global model,
we target SPECIFIC player-target combos with radically different approaches.

Targets (from LOO error analysis):
  - Player 5 depth:  LOO MSE 0.008151, contributes 43.1% of depth error
  - Player 1 LR:     LOO MSE 0.008359, contributes 47.6% of LR error
  - Player 5 angle:  LOO MSE 0.004725, contributes 41.6% of angle error

Approaches per player-target:
  1. Bandwidth sweep (0.10 to 0.95 in fine steps)
  2. Alpha sweep (0.1 to 1000)
  3. Kernel type: Gaussian, Cauchy, Epanechnikov
  4. Frame sweep (fine-grained around per-player optimal)
  5. Multi-frame ensemble (1, 3, 5, 7, 9 frames)
  6. Cross-player transfer (add data from similar players)
  7. Feature subset: full, no-JA, minimal-variance
  8. k-NN baselines (k=3,5,7,11)
  9. Player mean (simplest possible - anti-overfitting)

Output: submissions that SPLICE improved predictions for problem combos
into Sub 2429 (current best LB 0.006502).
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

# Per-player optimal frames
PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}

# Problem player-target combos to optimize
PROBLEM_COMBOS = [
    (5, "depth"),
    (1, "left_right"),
    (5, "angle"),
]


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


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame, include_joint_angles=True):
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

    if include_joint_angles:
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


def extract_player_features(data, indices, kp_index, frame, include_ja=True):
    all_feats = []
    for i in indices:
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame, include_joint_angles=include_ja)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def pls_augment(X_raw_train, X_raw_test, y_target, n_max=15):
    n_p = len(X_raw_train)
    pls_scaler = StandardScaler()
    raw_tr = pls_scaler.fit_transform(X_raw_train)
    raw_te = pls_scaler.transform(X_raw_test) if len(X_raw_test) > 0 else np.zeros((0, raw_tr.shape[1]))
    nc = min(n_max, n_p - n_p // 5 - 1)
    nc = max(3, nc)
    best_nc, best_mse = 3, float('inf')
    for c in [3, 5, 8, 10, 15]:
        if c > n_max or c > nc:
            break
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mses = []
        for ktr, kval in kf.split(raw_tr):
            pls = PLSRegression(n_components=c)
            pls.fit(raw_tr[ktr], y_target[ktr])
            pred = pls.predict(raw_tr[kval]).flatten()
            mses.append(np.mean((pred - y_target[kval]) ** 2))
        if np.mean(mses) < best_mse:
            best_mse = np.mean(mses)
            best_nc = c
    pls_train = np.zeros((n_p, n_max), dtype=np.float32)
    pls_test = np.zeros((len(X_raw_test), n_max), dtype=np.float32)
    pls = PLSRegression(n_components=best_nc)
    pls.fit(raw_tr, y_target)
    pls_train[:, :best_nc] = pls.transform(raw_tr)
    if len(raw_te) > 0:
        pls_test[:, :best_nc] = pls.transform(raw_te)
    return pls_train, pls_test


def run_player_target(train_data, test_data, y_scaled_all, scaler, target, target_idx,
                      pid, config):
    """Run pipeline for a SINGLE player-target with given config.
    Returns (oof_preds, test_preds, loo_mse) for just this player."""

    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    tidx = target_idx[target]

    y_raw = train_data['y'][:, tidx]
    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()

    tr_mask = pids_train == pid
    te_mask = pids_test == pid
    tr_indices = np.where(tr_mask)[0]
    te_indices = np.where(te_mask)[0]
    n_tr = len(tr_indices)
    n_te = len(te_indices)

    bw = config.get('bw', 0.45)
    alpha = config.get('alpha', 10.0)
    n_frames = config.get('n_frames', 1)
    frame_spacing = config.get('frame_spacing', 5)
    pls_max = config.get('pls_max', 15)
    include_ja = config.get('include_ja', True)
    kernel = config.get('kernel', 'gaussian')
    center_frame = config.get('center_frame', PLAYER_FRAMES[target][pid])
    cross_player = config.get('cross_player', False)
    shrinkage = config.get('shrinkage', 0.0)  # blend toward player mean

    # Build frame offsets
    if n_frames == 1:
        offsets = [0]
    elif n_frames == 3:
        offsets = [-frame_spacing, 0, frame_spacing]
    elif n_frames == 5:
        offsets = [-2*frame_spacing, -frame_spacing, 0, frame_spacing, 2*frame_spacing]
    elif n_frames == 7:
        offsets = [-3*frame_spacing, -2*frame_spacing, -frame_spacing, 0,
                   frame_spacing, 2*frame_spacing, 3*frame_spacing]
    elif n_frames == 9:
        offsets = list(range(-4*frame_spacing, 5*frame_spacing, frame_spacing))
    else:
        offsets = [0]

    frames = [int(np.clip(center_frame + off, 0, 239)) for off in offsets]

    # Data selection: use this player only, or add cross-player data
    if cross_player:
        # Use ALL players' data for training, but weight this player more heavily
        all_tr_indices = np.arange(len(pids_train))
        y_tr_raw = y_raw
        y_tr_scaled = y_scaled
        X_raw_tr = train_data['X_raw']
        # Player indicator for weighting later
        is_target_player = pids_train == pid
    else:
        all_tr_indices = tr_indices
        y_tr_raw = y_raw[tr_mask]
        y_tr_scaled = y_scaled[tr_mask]
        X_raw_tr = train_data['X_raw'][tr_mask]

    X_raw_te = test_data['X_raw'][te_mask] if n_te > 0 else np.zeros((0, train_data['X_raw'].shape[1]))

    # PLS augmentation (always on this player's data)
    pls_train_p, pls_test_p = pls_augment(
        train_data['X_raw'][tr_mask], X_raw_te, y_raw[tr_mask], n_max=pls_max
    )

    if cross_player:
        # PLS for all players
        pls_all_train, _ = pls_augment(
            train_data['X_raw'], np.zeros((0, train_data['X_raw'].shape[1])),
            y_raw, n_max=pls_max
        )

    frame_oofs = []
    frame_tests = []

    for frame in frames:
        if cross_player:
            X_tr_hc = extract_player_features(train_data, all_tr_indices, kp_index, frame, include_ja=include_ja)
            X_tr_aug = np.hstack([X_tr_hc, pls_all_train])
        else:
            X_tr_hc = extract_player_features(train_data, tr_indices, kp_index, frame, include_ja=include_ja)
            X_tr_aug = np.hstack([X_tr_hc, pls_train_p])

        X_te_hc = extract_player_features(test_data, te_indices, kp_index, frame, include_ja=include_ja) if n_te > 0 \
            else np.zeros((0, X_tr_hc.shape[1]))
        X_te_aug = np.hstack([X_te_hc, pls_test_p]) if n_te > 0 \
            else np.zeros((0, X_tr_aug.shape[1]))

        n_all_tr = len(X_tr_aug)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_aug)
        X_te_s = sc.transform(X_te_aug) if n_te > 0 else np.zeros((0, X_tr_aug.shape[1]))

        D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        all_dists = D_tr[np.triu_indices(n_all_tr, k=1)]
        sigma = np.quantile(all_dists, bw) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        if cross_player:
            # LOO only for target player samples
            oof = np.zeros(n_tr)
            # Find indices of target player within the full training set
            target_in_all = np.where(is_target_player)[0]

            for li, gi in enumerate(target_in_all):
                if kernel == 'gaussian':
                    weights = np.exp(-D_tr[gi, :] ** 2 / (2 * sigma ** 2))
                elif kernel == 'cauchy':
                    weights = 1.0 / (1.0 + (D_tr[gi, :] / sigma) ** 2)
                elif kernel == 'epanechnikov':
                    u = D_tr[gi, :] / sigma
                    weights = np.maximum(0, 1 - u ** 2)
                weights[gi] = 0  # LOO
                # Upweight same-player samples
                weights[is_target_player] *= 3.0
                weights[gi] = 0  # still exclude self

                if weights.sum() < 1e-10:
                    oof[li] = np.mean(y_tr_scaled[target_in_all])
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr_scaled, sample_weight=weights)
                oof[li] = ridge.predict(X_tr_s[gi:gi+1])[0]

            tpred = np.zeros(n_te)
            if n_te > 0:
                D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
                for j in range(n_te):
                    if kernel == 'gaussian':
                        weights = np.exp(-D_te[j, :] ** 2 / (2 * sigma ** 2))
                    elif kernel == 'cauchy':
                        weights = 1.0 / (1.0 + (D_te[j, :] / sigma) ** 2)
                    elif kernel == 'epanechnikov':
                        u = D_te[j, :] / sigma
                        weights = np.maximum(0, 1 - u ** 2)
                    weights[is_target_player] *= 3.0
                    if weights.sum() < 1e-10:
                        tpred[j] = np.mean(y_tr_scaled[target_in_all])
                        continue
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X_tr_s, y_tr_scaled, sample_weight=weights)
                    tpred[j] = ridge.predict(X_te_s[j:j+1])[0]
        else:
            # Standard single-player LOO
            oof = np.zeros(n_tr)
            for i in range(n_tr):
                if kernel == 'gaussian':
                    weights = np.exp(-D_tr[i, :] ** 2 / (2 * sigma ** 2))
                elif kernel == 'cauchy':
                    weights = 1.0 / (1.0 + (D_tr[i, :] / sigma) ** 2)
                elif kernel == 'epanechnikov':
                    u = D_tr[i, :] / sigma
                    weights = np.maximum(0, 1 - u ** 2)
                weights[i] = 0
                if weights.sum() < 1e-10:
                    oof[i] = np.mean(y_tr_scaled)
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr_scaled, sample_weight=weights)
                oof[i] = ridge.predict(X_tr_s[i:i+1])[0]

            tpred = np.zeros(n_te)
            if n_te > 0:
                D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
                for j in range(n_te):
                    if kernel == 'gaussian':
                        weights = np.exp(-D_te[j, :] ** 2 / (2 * sigma ** 2))
                    elif kernel == 'cauchy':
                        weights = 1.0 / (1.0 + (D_te[j, :] / sigma) ** 2)
                    elif kernel == 'epanechnikov':
                        u = D_te[j, :] / sigma
                        weights = np.maximum(0, 1 - u ** 2)
                    if weights.sum() < 1e-10:
                        tpred[j] = np.mean(y_tr_scaled)
                        continue
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X_tr_s, y_tr_scaled, sample_weight=weights)
                    tpred[j] = ridge.predict(X_te_s[j:j+1])[0]

        frame_oofs.append(oof)
        frame_tests.append(tpred)

    avg_oof = np.mean(frame_oofs, axis=0)
    avg_test = np.mean(frame_tests, axis=0)

    # Apply shrinkage toward player mean
    if shrinkage > 0:
        player_mean = np.mean(y_scaled[tr_mask])
        avg_oof = (1 - shrinkage) * avg_oof + shrinkage * player_mean
        avg_test = (1 - shrinkage) * avg_test + shrinkage * player_mean

    loo_mse = np.mean((avg_oof - y_scaled[tr_mask]) ** 2)
    return avg_oof, avg_test, loo_mse


def run_knn_player(train_data, test_data, y_scaled, scaler, target, target_idx, pid, k=5):
    """Simple k-NN baseline for a player-target combo."""
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    tidx = target_idx[target]

    y_raw = train_data['y'][:, tidx]
    y_sc = scaler.transform(y_raw.reshape(-1, 1)).ravel()

    tr_mask = pids_train == pid
    te_mask = pids_test == pid
    tr_indices = np.where(tr_mask)[0]
    te_indices = np.where(te_mask)[0]
    n_tr = len(tr_indices)
    n_te = len(te_indices)

    frame = PLAYER_FRAMES[target][pid]
    X_tr = extract_player_features(train_data, tr_indices, kp_index, frame)
    X_te = extract_player_features(test_data, te_indices, kp_index, frame) if n_te > 0 \
        else np.zeros((0, X_tr.shape[1]))

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te) if n_te > 0 else np.zeros((0, X_tr.shape[1]))

    D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
    y_player = y_sc[tr_mask]

    oof = np.zeros(n_tr)
    for i in range(n_tr):
        dists = D_tr[i].copy()
        dists[i] = float('inf')
        nn = np.argsort(dists)[:k]
        oof[i] = np.mean(y_player[nn])

    tpred = np.zeros(n_te)
    if n_te > 0:
        D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
        for j in range(n_te):
            nn = np.argsort(D_te[j])[:k]
            tpred[j] = np.mean(y_player[nn])

    loo_mse = np.mean((oof - y_player) ** 2)
    return oof, tpred, loo_mse


def main():
    t0 = time.time()
    print("=" * 70)
    print("PLAYER-TARGETED OPTIMIZATION")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Build comprehensive sweep configs
    configs = []

    # 1. Fine bandwidth sweep
    for bw in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
               0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        configs.append({
            'bw': bw, 'n_frames': 1, 'name': f'bw={bw:.2f}',
        })

    # 2. Alpha sweep at key bandwidths
    for bw in [0.30, 0.45, 0.80]:
        for alpha in [0.1, 1.0, 5.0, 50.0, 100.0, 500.0]:
            configs.append({
                'bw': bw, 'alpha': alpha, 'n_frames': 1,
                'name': f'bw={bw:.2f} a={alpha}',
            })

    # 3. Kernel types
    for kernel in ['cauchy', 'epanechnikov']:
        for bw in [0.30, 0.45, 0.65, 0.80]:
            configs.append({
                'bw': bw, 'kernel': kernel, 'n_frames': 1,
                'name': f'{kernel} bw={bw:.2f}',
            })

    # 4. Multi-frame ensembles at key bandwidths
    for bw in [0.30, 0.45, 0.80]:
        for nf in [3, 5, 7, 9]:
            for sp in [3, 5, 8]:
                configs.append({
                    'bw': bw, 'n_frames': nf, 'frame_spacing': sp,
                    'name': f'bw={bw:.2f} nf={nf} sp={sp}',
                })

    # 5. Fine frame sweep (center frame variation)
    for delta in range(-20, 21, 5):
        if delta == 0:
            continue
        configs.append({
            'bw': 0.45, 'n_frames': 1, 'center_frame_delta': delta,
            'name': f'frame+{delta}',
        })

    # 6. Without joint angles
    for bw in [0.30, 0.45, 0.80]:
        configs.append({
            'bw': bw, 'include_ja': False, 'n_frames': 1,
            'name': f'bw={bw:.2f} no-JA',
        })

    # 7. Shrinkage toward player mean
    for shrink in [0.05, 0.10, 0.20, 0.30, 0.50]:
        configs.append({
            'bw': 0.45, 'shrinkage': shrink, 'n_frames': 1,
            'name': f'shrink={shrink:.2f}',
        })

    # 8. Cross-player transfer
    for bw in [0.30, 0.45, 0.80]:
        configs.append({
            'bw': bw, 'cross_player': True, 'n_frames': 1,
            'name': f'cross-player bw={bw:.2f}',
        })

    # 9. Combined: best bandwidth + multi-frame + cauchy
    for bw in [0.45, 0.65, 0.80]:
        for nf in [3, 5]:
            configs.append({
                'bw': bw, 'kernel': 'cauchy', 'n_frames': nf, 'frame_spacing': 5,
                'name': f'cauchy bw={bw:.2f} nf={nf}',
            })

    print(f"\n{len(configs)} configs x {len(PROBLEM_COMBOS)} player-target combos")
    print(f"= {len(configs) * len(PROBLEM_COMBOS)} total evaluations")

    # Run sweep for each problem combo
    results = {}

    for pid, target in PROBLEM_COMBOS:
        key = f"P{pid}_{target}"
        results[key] = []

        # Get baseline
        baseline_cfg = {'bw': 0.45, 'n_frames': 1, 'name': 'baseline bw=0.45'}
        _, _, baseline_loo = run_player_target(
            train_data, test_data, None, scalers[target], target, target_idx, pid, baseline_cfg
        )

        print(f"\n{'='*60}")
        print(f"Player {pid} - {target} (baseline LOO: {baseline_loo:.6f})")
        print(f"{'='*60}")

        # Also run k-NN baselines
        for k in [3, 5, 7, 11]:
            oof, tpred, loo = run_knn_player(
                train_data, test_data, None, scalers[target], target, target_idx, pid, k=k
            )
            pct = (loo - baseline_loo) / baseline_loo * 100
            results[key].append({
                'name': f'kNN k={k}',
                'loo': loo,
                'pct': pct,
                'oof': oof,
                'test': tpred,
                'type': 'knn',
            })

        # Player mean baseline
        y_raw = train_data['y'][:, target_idx[target]]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()
        tr_mask = train_data['pids'] == pid
        te_mask = test_data['pids'] == pid
        player_mean = np.mean(y_sc[tr_mask])
        mean_oof = np.full(np.sum(tr_mask), player_mean)
        mean_tpred = np.full(np.sum(te_mask), player_mean)
        mean_loo = np.mean((mean_oof - y_sc[tr_mask]) ** 2)
        pct = (mean_loo - baseline_loo) / baseline_loo * 100
        results[key].append({
            'name': 'player_mean',
            'loo': mean_loo,
            'pct': pct,
            'oof': mean_oof,
            'test': mean_tpred,
            'type': 'mean',
        })

        # Run all configs
        for i, cfg in enumerate(configs):
            # Handle center_frame_delta
            run_cfg = dict(cfg)
            if 'center_frame_delta' in run_cfg:
                delta = run_cfg.pop('center_frame_delta')
                run_cfg['center_frame'] = PLAYER_FRAMES[target][pid] + delta

            try:
                oof, tpred, loo = run_player_target(
                    train_data, test_data, None, scalers[target], target, target_idx, pid, run_cfg
                )
            except Exception as e:
                continue

            pct = (loo - baseline_loo) / baseline_loo * 100
            results[key].append({
                'name': cfg['name'],
                'loo': loo,
                'pct': pct,
                'oof': oof,
                'test': tpred,
                'type': 'lw_ridge',
                'config': cfg,
            })

            if (i + 1) % 20 == 0:
                # Show current best
                best_so_far = min(results[key], key=lambda x: x['loo'])
                print(f"  [{i+1}/{len(configs)}] best so far: {best_so_far['name']} "
                      f"LOO={best_so_far['loo']:.6f} ({best_so_far['pct']:+.2f}%)")

        # Sort and display top-10
        sorted_results = sorted(results[key], key=lambda x: x['loo'])
        print(f"\n  Top-10 for Player {pid} - {target}:")
        for rank, r in enumerate(sorted_results[:10]):
            marker = " ***" if r['loo'] < baseline_loo else ""
            print(f"    {rank+1}. {r['name']}: {r['loo']:.6f} ({r['pct']:+.2f}%){marker}")

        print(f"\n  Bottom-3 (worst):")
        for r in sorted_results[-3:]:
            print(f"    {r['name']}: {r['loo']:.6f} ({r['pct']:+.2f}%)")

    # Generate spliced submissions
    print("\n" + "=" * 70)
    print("GENERATING SPLICED SUBMISSIONS")
    print("=" * 70)

    # Load current best (Sub 2429)
    base_sub = pd.read_csv(SUBMISSION_DIR / "submission_2429.csv")

    # Also need to know which test IDs belong to which player
    pid_map_test = dict(zip(test_data['ids'], test_data['pids']))

    # For each problem combo, take the best config and splice into base
    # Strategy 1: splice each combo independently
    for pid, target in PROBLEM_COMBOS:
        key = f"P{pid}_{target}"
        sorted_results = sorted(results[key], key=lambda x: x['loo'])
        best = sorted_results[0]

        # Only splice if improvement
        baseline_cfg = {'bw': 0.45, 'n_frames': 1, 'name': 'baseline'}
        _, _, baseline_loo = run_player_target(
            train_data, test_data, None, scalers[target], target, target_idx, pid, baseline_cfg
        )

        if best['loo'] >= baseline_loo:
            print(f"  P{pid} {target}: no improvement found, skipping")
            continue

        # Create submission with just this combo spliced
        sn = get_next_submission_number()
        sp = SUBMISSION_DIR / f"submission_{sn}.csv"
        spliced = base_sub.copy()
        col = f"scaled_{target}"

        te_mask = test_data['pids'] == pid
        te_indices = np.where(te_mask)[0]

        for local_i, global_i in enumerate(te_indices):
            tid = test_data['ids'][global_i]
            row_idx = spliced[spliced['id'] == tid].index[0]
            spliced.loc[row_idx, col] = best['test'][local_i]

        spliced.to_csv(sp, index=False)
        print(f"  Sub {sn}: P{pid} {target} spliced ({best['name']}, LOO {best['loo']:.6f}, {best['pct']:+.2f}%)")

    # Strategy 2: splice ALL 3 combos at once
    sn = get_next_submission_number()
    sp = SUBMISSION_DIR / f"submission_{sn}.csv"
    spliced_all = base_sub.copy()

    for pid, target in PROBLEM_COMBOS:
        key = f"P{pid}_{target}"
        sorted_results = sorted(results[key], key=lambda x: x['loo'])
        best = sorted_results[0]
        col = f"scaled_{target}"
        te_mask = test_data['pids'] == pid
        te_indices = np.where(te_mask)[0]
        for local_i, global_i in enumerate(te_indices):
            tid = test_data['ids'][global_i]
            row_idx = spliced_all[spliced_all['id'] == tid].index[0]
            spliced_all.loc[row_idx, col] = best['test'][local_i]

    spliced_all.to_csv(sp, index=False)
    print(f"  Sub {sn}: ALL 3 combos spliced into Sub 2429")

    # Strategy 3: conservative blend - 50% best + 50% base for each combo
    sn = get_next_submission_number()
    sp = SUBMISSION_DIR / f"submission_{sn}.csv"
    spliced_blend = base_sub.copy()

    for pid, target in PROBLEM_COMBOS:
        key = f"P{pid}_{target}"
        sorted_results = sorted(results[key], key=lambda x: x['loo'])
        best = sorted_results[0]
        col = f"scaled_{target}"
        te_mask = test_data['pids'] == pid
        te_indices = np.where(te_mask)[0]
        for local_i, global_i in enumerate(te_indices):
            tid = test_data['ids'][global_i]
            row_idx = spliced_blend[spliced_blend['id'] == tid].index[0]
            base_val = spliced_blend.loc[row_idx, col]
            spliced_blend.loc[row_idx, col] = 0.5 * best['test'][local_i] + 0.5 * base_val

    spliced_blend.to_csv(sp, index=False)
    print(f"  Sub {sn}: ALL 3 combos 50/50 blended into Sub 2429")

    # Strategy 4: top-3 ensemble for each combo
    sn = get_next_submission_number()
    sp = SUBMISSION_DIR / f"submission_{sn}.csv"
    spliced_ens = base_sub.copy()

    for pid, target in PROBLEM_COMBOS:
        key = f"P{pid}_{target}"
        sorted_results = sorted(results[key], key=lambda x: x['loo'])
        top3 = sorted_results[:3]
        col = f"scaled_{target}"
        te_mask = test_data['pids'] == pid
        te_indices = np.where(te_mask)[0]
        avg_test = np.mean([r['test'] for r in top3], axis=0)
        for local_i, global_i in enumerate(te_indices):
            tid = test_data['ids'][global_i]
            row_idx = spliced_ens[spliced_ens['id'] == tid].index[0]
            spliced_ens.loc[row_idx, col] = avg_test[local_i]

    spliced_ens.to_csv(sp, index=False)
    print(f"  Sub {sn}: top-3 ensemble per combo spliced into Sub 2429")

    # Strategy 5: top-3 ensemble 50/50 blended
    sn = get_next_submission_number()
    sp = SUBMISSION_DIR / f"submission_{sn}.csv"
    spliced_ens_bl = base_sub.copy()

    for pid, target in PROBLEM_COMBOS:
        key = f"P{pid}_{target}"
        sorted_results = sorted(results[key], key=lambda x: x['loo'])
        top3 = sorted_results[:3]
        col = f"scaled_{target}"
        te_mask = test_data['pids'] == pid
        te_indices = np.where(te_mask)[0]
        avg_test = np.mean([r['test'] for r in top3], axis=0)
        for local_i, global_i in enumerate(te_indices):
            tid = test_data['ids'][global_i]
            row_idx = spliced_ens_bl[spliced_ens_bl['id'] == tid].index[0]
            base_val = spliced_ens_bl.loc[row_idx, col]
            spliced_ens_bl.loc[row_idx, col] = 0.5 * avg_test[local_i] + 0.5 * base_val

    spliced_ens_bl.to_csv(sp, index=False)
    print(f"  Sub {sn}: top-3 ensemble 50/50 blended per combo into Sub 2429")

    total_time = time.time() - t0
    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
