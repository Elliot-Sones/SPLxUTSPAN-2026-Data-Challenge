"""
Regularization Sweep - Attacking the Overfitting Bottleneck

The #1 problem is NOT signal - it's overfitting:
  - Angle: 2.57x overfit (LOO 0.002511 vs LB ~0.006454)
  - Depth: 1.51x overfit
  - LR: 1.62x overfit

Current settings (bw=0.45, alpha=10) were never systematically tuned.
This script tests stronger regularization via:
  1. Extended bandwidth: 0.30 to 0.90 (wider = more neighbors = more regularization)
  2. Per-target Ridge alpha: 10 to 1000 (higher = more shrinkage toward zero)
  3. Post-hoc prediction shrinkage toward player mean (empirical Bayes / James-Stein)
  4. k-NN averaging as extreme regularization baseline

All tested within the unified pipeline framework (per-player frames, joint angles,
3-frame ensemble).

The goal is NOT to improve LOO (that's easy). The goal is to find settings where
the LOO-to-LB transfer ratio is better, which means accepting WORSE LOO for
potentially better LB.
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

PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}
FRAME_OFFSETS = [-5, 0, 5]

# Historical overfitting ratios (LB MSE / LOO MSE)
OVERFIT_RATIOS = {"angle": 2.57, "depth": 1.51, "left_right": 1.62}


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


# ==============================================================
# DATA LOADING (identical to unified pipeline)
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
# FEATURE EXTRACTION (identical to unified pipeline)
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


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """198 hoop-relative + 10 joint angle features."""
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

    # Joint angles
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


def extract_player_features(data, player_mask, kp_index, frame):
    indices = np.where(player_mask)[0]
    all_feats = []
    for i in indices:
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def pls_augment_player(X_raw_train, X_raw_test, y_raw_player, n_max=15):
    n_p = len(X_raw_train)
    pls_scaler = StandardScaler()
    raw_tr = pls_scaler.fit_transform(X_raw_train)
    raw_te = pls_scaler.transform(X_raw_test) if len(X_raw_test) > 0 else np.zeros((0, raw_tr.shape[1]))
    nc = min(n_max, n_p - n_p // 5 - 1)
    nc = max(3, nc)
    best_nc, best_mse = 3, float('inf')
    for c in [3, 5, 8, 10, 15]:
        if c > nc:
            break
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mses = []
        for ktr, kval in kf.split(raw_tr):
            pls = PLSRegression(n_components=c)
            pls.fit(raw_tr[ktr], y_raw_player[ktr])
            pred = pls.predict(raw_tr[kval]).flatten()
            mses.append(np.mean((pred - y_raw_player[kval]) ** 2))
        if np.mean(mses) < best_mse:
            best_mse = np.mean(mses)
            best_nc = c
    pls_train = np.zeros((n_p, n_max), dtype=np.float32)
    pls_test = np.zeros((len(X_raw_test), n_max), dtype=np.float32)
    pls = PLSRegression(n_components=best_nc)
    pls.fit(raw_tr, y_raw_player)
    pls_train[:, :best_nc] = pls.transform(raw_tr)
    if len(raw_te) > 0:
        pls_test[:, :best_nc] = pls.transform(raw_te)
    return pls_train, pls_test


# ==============================================================
# CORE: Locally weighted Ridge with configurable regularization
# ==============================================================

def lw_ridge_player(X_tr_aug, X_te_aug, y_player, bandwidth_quantile, alpha):
    """Locally weighted Ridge with configurable bandwidth and alpha."""
    n_tr = len(X_tr_aug)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_aug)
    X_te_s = scaler.transform(X_te_aug) if len(X_te_aug) > 0 else np.zeros((0, X_tr_aug.shape[1]))

    D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
    all_dists = D_tr[np.triu_indices(n_tr, k=1)]
    sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
    sigma = max(sigma, 1e-6)

    oof = np.zeros(n_tr)
    for i in range(n_tr):
        weights = np.exp(-D_tr[i, :] ** 2 / (2 * sigma ** 2))
        weights[i] = 0
        if weights.sum() < 1e-10:
            oof[i] = np.mean(y_player)
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_s, y_player, sample_weight=weights)
        oof[i] = ridge.predict(X_tr_s[i:i+1])[0]

    test_preds = np.zeros(len(X_te_s))
    if len(X_te_s) > 0:
        D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
        for j in range(len(X_te_s)):
            weights = np.exp(-D_te[j, :] ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                test_preds[j] = np.mean(y_player)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_player, sample_weight=weights)
            test_preds[j] = ridge.predict(X_te_s[j:j+1])[0]

    return oof, test_preds


def knn_player(X_tr_aug, X_te_aug, y_player, k):
    """Simple k-NN regression as extreme regularization baseline."""
    n_tr = len(X_tr_aug)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_aug)
    X_te_s = scaler.transform(X_te_aug) if len(X_te_aug) > 0 else np.zeros((0, X_tr_aug.shape[1]))

    D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')

    oof = np.zeros(n_tr)
    for i in range(n_tr):
        dists = D_tr[i, :].copy()
        dists[i] = np.inf  # exclude self
        nn_idx = np.argsort(dists)[:k]
        oof[i] = np.mean(y_player[nn_idx])

    test_preds = np.zeros(len(X_te_s))
    if len(X_te_s) > 0:
        D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
        for j in range(len(X_te_s)):
            nn_idx = np.argsort(D_te[j, :])[:k]
            test_preds[j] = np.mean(y_player[nn_idx])

    return oof, test_preds


# ==============================================================
# PIPELINE RUNNER
# ==============================================================

def run_pipeline(train_data, test_data, y_train, scalers, target_idx,
                 bandwidth, alpha, shrinkage_factor=None, use_knn=False, knn_k=10):
    """Run full pipeline with given regularization settings."""
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    unique_pids = sorted(np.unique(pids_train))

    results = {}
    for target in TARGETS:
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_scaled = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        oof_preds = np.zeros(len(pids_train))
        test_preds = np.zeros(len(pids_test))

        # Get per-target alpha if dict, else use scalar
        a = alpha[target] if isinstance(alpha, dict) else alpha

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            tr_indices = np.where(tr_mask)[0]
            te_indices = np.where(te_mask)[0]
            n_te = len(te_indices)

            center_frame = PLAYER_FRAMES[target][pid]
            frames = [int(np.clip(center_frame + off, 0, 239)) for off in FRAME_OFFSETS]

            pls_train, pls_test = pls_augment_player(
                train_data['X_raw'][tr_mask],
                test_data['X_raw'][te_mask] if n_te > 0 else np.zeros((0, train_data['X_raw'].shape[1])),
                y_raw[tr_mask]
            )

            frame_oofs, frame_tests = [], []
            for frame in frames:
                X_tr_hc = extract_player_features(train_data, tr_mask, kp_index, frame)
                X_te_hc = extract_player_features(test_data, te_mask, kp_index, frame) if n_te > 0 \
                    else np.zeros((0, X_tr_hc.shape[1]))
                X_tr_aug = np.hstack([X_tr_hc, pls_train])
                X_te_aug = np.hstack([X_te_hc, pls_test]) if n_te > 0 \
                    else np.zeros((0, X_tr_aug.shape[1]))

                if use_knn:
                    oof, tpred = knn_player(X_tr_aug, X_te_aug, y_scaled[tr_mask], knn_k)
                else:
                    oof, tpred = lw_ridge_player(X_tr_aug, X_te_aug, y_scaled[tr_mask], bandwidth, a)
                frame_oofs.append(oof)
                frame_tests.append(tpred)

            avg_oof = np.mean(frame_oofs, axis=0)
            avg_test = np.mean(frame_tests, axis=0)

            # Post-hoc shrinkage toward player mean
            if shrinkage_factor is not None:
                sf = shrinkage_factor[target] if isinstance(shrinkage_factor, dict) else shrinkage_factor
                player_mean = np.mean(y_scaled[tr_mask])
                avg_oof = sf * avg_oof + (1 - sf) * player_mean
                avg_test = sf * avg_test + (1 - sf) * player_mean

            oof_preds[tr_indices] = avg_oof
            if n_te > 0:
                test_preds[te_indices] = avg_test

        loo_mse = np.mean((oof_preds - y_scaled) ** 2)
        results[target] = {'oof': oof_preds, 'test': test_preds,
                           'loo_mse': loo_mse, 'y_scaled': y_scaled}

    return results


def save_submission(test_data, results, tag=""):
    sub_num = get_next_submission_number()
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df = pd.DataFrame({'id': test_data['ids']})
    for target in TARGETS:
        sub_df[f"scaled_{target}"] = results[target]['test']
    sub_df.to_csv(sub_path, index=False)
    return sub_num, sub_path


def save_blend(test_data, results, existing_sub_num, weight):
    existing_path = SUBMISSION_DIR / f"submission_{existing_sub_num}.csv"
    if not existing_path.exists():
        return None, None
    existing_df = pd.read_csv(existing_path).set_index('id')
    sub_num = get_next_submission_number()
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df = pd.DataFrame({'id': test_data['ids']})
    for target in TARGETS:
        col = f"scaled_{target}"
        preds = results[target]['test']
        blended = []
        for i, tid in enumerate(test_data['ids']):
            if tid in existing_df.index:
                blended.append(weight * preds[i] + (1 - weight) * existing_df.loc[tid, col])
            else:
                blended.append(preds[i])
        sub_df[col] = blended
    sub_df.to_csv(sub_path, index=False)
    return sub_num, sub_path


# ==============================================================
# MAIN: SWEEP
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("REGULARIZATION SWEEP - ATTACKING OVERFITTING")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    baseline_loo = {"angle": 0.002645, "depth": 0.004601, "left_right": 0.004331}
    baseline_mean = 0.003859

    # Pre-compute features (expensive part - do once, cache)
    # Actually features depend on frame which depends on target x player,
    # so we can't easily pre-compute. Just run the sweep.

    configs = []

    # ---- Experiment 1: Bandwidth sweep (alpha=10 fixed) ----
    for bw in [0.30, 0.45, 0.55, 0.65, 0.80]:
        configs.append({
            'name': f'bw={bw:.2f} alpha=10',
            'bandwidth': bw, 'alpha': 10, 'shrinkage_factor': None,
            'use_knn': False,
        })

    # ---- Experiment 2: Alpha sweep (bw=0.45 fixed) ----
    for a in [50, 100, 200, 500]:
        configs.append({
            'name': f'bw=0.45 alpha={a}',
            'bandwidth': 0.45, 'alpha': a, 'shrinkage_factor': None,
            'use_knn': False,
        })

    # ---- Experiment 3: Per-target alpha (higher for angle) ----
    configs.append({
        'name': 'per-target alpha (angle=100, depth=10, lr=10)',
        'bandwidth': 0.45, 'alpha': {"angle": 100, "depth": 10, "left_right": 10},
        'shrinkage_factor': None, 'use_knn': False,
    })
    configs.append({
        'name': 'per-target alpha (angle=200, depth=50, lr=50)',
        'bandwidth': 0.45, 'alpha': {"angle": 200, "depth": 50, "left_right": 50},
        'shrinkage_factor': None, 'use_knn': False,
    })

    # ---- Experiment 4: Post-hoc shrinkage ----
    # Shrinkage = 1/overfit_ratio: how much to trust the model vs player mean
    # angle: 1/2.57=0.39, depth: 1/1.51=0.66, lr: 1/1.62=0.62
    configs.append({
        'name': 'shrinkage (calibrated from overfit ratios)',
        'bandwidth': 0.45, 'alpha': 10,
        'shrinkage_factor': {"angle": 1/2.57, "depth": 1/1.51, "left_right": 1/1.62},
        'use_knn': False,
    })
    # Lighter shrinkage (halfway between full model and calibrated)
    configs.append({
        'name': 'shrinkage (light: 0.70/0.83/0.81)',
        'bandwidth': 0.45, 'alpha': 10,
        'shrinkage_factor': {"angle": 0.70, "depth": 0.83, "left_right": 0.81},
        'use_knn': False,
    })

    # ---- Experiment 5: Combined best candidates ----
    configs.append({
        'name': 'bw=0.65 alpha=50 (strong regularization)',
        'bandwidth': 0.65, 'alpha': 50, 'shrinkage_factor': None,
        'use_knn': False,
    })
    configs.append({
        'name': 'bw=0.65 alpha=100 (very strong regularization)',
        'bandwidth': 0.65, 'alpha': 100, 'shrinkage_factor': None,
        'use_knn': False,
    })
    configs.append({
        'name': 'bw=0.55 alpha=50',
        'bandwidth': 0.55, 'alpha': 50, 'shrinkage_factor': None,
        'use_knn': False,
    })

    # ---- Experiment 6: k-NN (extreme regularization) ----
    for k in [10, 15, 20, 30]:
        configs.append({
            'name': f'kNN k={k}',
            'bandwidth': None, 'alpha': None, 'shrinkage_factor': None,
            'use_knn': True, 'knn_k': k,
        })

    print(f"\nRunning {len(configs)} configurations...\n")

    all_results = []
    for i, cfg in enumerate(configs):
        t1 = time.time()
        results = run_pipeline(
            train_data, test_data, y_train, scalers, target_idx,
            bandwidth=cfg.get('bandwidth', 0.45),
            alpha=cfg.get('alpha', 10),
            shrinkage_factor=cfg.get('shrinkage_factor'),
            use_knn=cfg.get('use_knn', False),
            knn_k=cfg.get('knn_k', 10),
        )
        elapsed = time.time() - t1

        mean_loo = np.mean([results[t]['loo_mse'] for t in TARGETS])
        pct = (mean_loo - baseline_mean) / baseline_mean * 100

        per_target = {t: results[t]['loo_mse'] for t in TARGETS}
        per_target_pct = {t: (results[t]['loo_mse'] - baseline_loo[t]) / baseline_loo[t] * 100 for t in TARGETS}

        # Estimated LB score using overfit ratios
        est_lb = np.mean([results[t]['loo_mse'] * OVERFIT_RATIOS[t] for t in TARGETS])

        all_results.append({
            'config': cfg,
            'mean_loo': mean_loo,
            'pct_vs_baseline': pct,
            'per_target': per_target,
            'per_target_pct': per_target_pct,
            'est_lb': est_lb,
            'results': results,
            'time': elapsed,
        })

        print(f"[{i+1}/{len(configs)}] {cfg['name']}")
        print(f"  LOO: {mean_loo:.6f} ({pct:+.2f}%) | Est LB: {est_lb:.6f}")
        print(f"  angle={per_target['angle']:.6f} ({per_target_pct['angle']:+.1f}%) "
              f"depth={per_target['depth']:.6f} ({per_target_pct['depth']:+.1f}%) "
              f"lr={per_target['left_right']:.6f} ({per_target_pct['left_right']:+.1f}%)")
        print(f"  [{elapsed:.1f}s]")

    # ==============================================================
    # ANALYSIS
    # ==============================================================
    print("\n" + "=" * 70)
    print("RESULTS SORTED BY ESTIMATED LB SCORE (lower = better)")
    print("=" * 70)

    sorted_results = sorted(all_results, key=lambda x: x['est_lb'])
    print(f"\n{'Config':<55} {'LOO':>8} {'LOO%':>7} {'EstLB':>8} {'Angle':>8} {'Depth':>8} {'LR':>8}")
    print("-" * 108)
    for r in sorted_results:
        print(f"{r['config']['name']:<55} "
              f"{r['mean_loo']:.6f} {r['pct_vs_baseline']:>+6.1f}% "
              f"{r['est_lb']:.6f} "
              f"{r['per_target']['angle']:.6f} "
              f"{r['per_target']['depth']:.6f} "
              f"{r['per_target']['left_right']:.6f}")

    # Also sort by LOO (what we can measure)
    print("\n" + "=" * 70)
    print("RESULTS SORTED BY LOO MSE (lower = better)")
    print("=" * 70)
    sorted_by_loo = sorted(all_results, key=lambda x: x['mean_loo'])
    print(f"\n{'Config':<55} {'LOO':>8} {'LOO%':>7} {'EstLB':>8}")
    print("-" * 82)
    for r in sorted_by_loo:
        print(f"{r['config']['name']:<55} "
              f"{r['mean_loo']:.6f} {r['pct_vs_baseline']:>+6.1f}% "
              f"{r['est_lb']:.6f}")

    # ==============================================================
    # KEY INSIGHT: estimate which approach has best LB transfer
    # ==============================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHT: ANGLE TARGET ANALYSIS")
    print("=" * 70)
    print("Angle overfits 2.57x. Stronger regularization may HURT LOO but HELP LB.")
    print()
    for r in sorted_results[:10]:
        angle_loo = r['per_target']['angle']
        angle_est_lb = angle_loo * OVERFIT_RATIOS['angle']
        print(f"  {r['config']['name']:<50} angle_LOO={angle_loo:.6f} est_angle_LB={angle_est_lb:.6f}")

    # ==============================================================
    # GENERATE SUBMISSIONS FOR TOP CANDIDATES
    # ==============================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Pick top 3 by estimated LB + top 1 by LOO (if different)
    candidates = []
    seen_names = set()

    # Top 3 by estimated LB
    for r in sorted_results[:3]:
        name = r['config']['name']
        if name not in seen_names:
            candidates.append(r)
            seen_names.add(name)

    # Top 1 by LOO
    for r in sorted_by_loo[:1]:
        name = r['config']['name']
        if name not in seen_names:
            candidates.append(r)
            seen_names.add(name)

    # Also add: the config with lowest ANGLE LOO * overfit_ratio
    best_angle_est = min(all_results, key=lambda x: x['per_target']['angle'] * OVERFIT_RATIOS['angle'])
    if best_angle_est['config']['name'] not in seen_names:
        candidates.append(best_angle_est)
        seen_names.add(best_angle_est['config']['name'])

    for r in candidates:
        name = r['config']['name']
        results = r['results']

        # Standalone
        sn, sp = save_submission(test_data, results)
        print(f"  Sub {sn}: {name} (standalone)")

        # 10% blend with Sub 2402 (current best)
        sn2, sp2 = save_blend(test_data, results, 2402, 0.10)
        if sn2:
            print(f"  Sub {sn2}: 10% {name} + 90% Sub 2402")

        # 30% blend with Sub 2402
        sn3, sp3 = save_blend(test_data, results, 2402, 0.30)
        if sn3:
            print(f"  Sub {sn3}: 30% {name} + 70% Sub 2402")

    total_time = time.time() - t0
    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
