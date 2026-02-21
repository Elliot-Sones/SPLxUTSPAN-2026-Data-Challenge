"""
Per-Target Independent Optimization

Competition mindset: each target is scored independently (mean of 3 MSEs).
We can use COMPLETELY different settings per target:
  - Different bandwidth
  - Different number of ensemble frames
  - Different frame offsets
  - Different PLS components
  - Different Ridge alpha
  - Even different feature sets

This script sweeps all these per target independently, then combines
the best settings into one submission.

From the regularization sweep we know:
  - Angle wants wide bandwidth (bw=0.80 best)
  - LR wants narrow bandwidth (bw=0.30 best)
  - Depth is stable across bandwidths
  - Alpha=10 is optimal for all targets

New things to test per target:
  1. Number of ensemble frames: 1, 3, 5, 7
  2. Frame offset spacing: +/-5, +/-10, +/-15
  3. PLS components: 5, 10, 15, 20
  4. Feature subsets: full (208), no joint angles (198), minimal (top 50 by variance)
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

# Best bandwidths per target from regularization sweep
BEST_BW = {"angle": 0.80, "depth": 0.55, "left_right": 0.30}


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
    """198 hoop-relative + optionally 10 joint angle features."""
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


def extract_player_features(data, player_mask, kp_index, frame, include_ja=True):
    indices = np.where(player_mask)[0]
    all_feats = []
    for i in indices:
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame, include_joint_angles=include_ja)
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
        if c > n_max or c > nc:
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


def run_single_target(train_data, test_data, y_train, scaler, target, target_idx,
                      bandwidth, n_frames, frame_spacing, pls_max, include_ja, alpha=10.0):
    """Run pipeline for a single target with given settings. Returns (oof, test_preds, loo_mse)."""
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    unique_pids = sorted(np.unique(pids_train))

    tidx = target_idx[target]
    y_raw = y_train[:, tidx]
    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()

    oof_preds = np.zeros(len(pids_train))
    test_preds = np.zeros(len(pids_test))

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
    else:
        offsets = [0]

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]
        n_te = len(te_indices)

        center_frame = PLAYER_FRAMES[target][pid]
        frames = [int(np.clip(center_frame + off, 0, 239)) for off in offsets]

        pls_train, pls_test = pls_augment_player(
            train_data['X_raw'][tr_mask],
            test_data['X_raw'][te_mask] if n_te > 0 else np.zeros((0, train_data['X_raw'].shape[1])),
            y_raw[tr_mask],
            n_max=pls_max,
        )

        frame_oofs, frame_tests = [], []
        for frame in frames:
            X_tr_hc = extract_player_features(train_data, tr_mask, kp_index, frame, include_ja=include_ja)
            X_te_hc = extract_player_features(test_data, te_mask, kp_index, frame, include_ja=include_ja) if n_te > 0 \
                else np.zeros((0, X_tr_hc.shape[1]))
            X_tr_aug = np.hstack([X_tr_hc, pls_train])
            X_te_aug = np.hstack([X_te_hc, pls_test]) if n_te > 0 \
                else np.zeros((0, X_tr_aug.shape[1]))

            n_tr = len(X_tr_aug)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_aug)
            X_te_s = sc.transform(X_te_aug) if n_te > 0 else np.zeros((0, X_tr_aug.shape[1]))
            D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
            all_dists = D_tr[np.triu_indices(n_tr, k=1)]
            sigma = np.quantile(all_dists, bandwidth) if len(all_dists) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            oof = np.zeros(n_tr)
            for i in range(n_tr):
                weights = np.exp(-D_tr[i, :] ** 2 / (2 * sigma ** 2))
                weights[i] = 0
                if weights.sum() < 1e-10:
                    oof[i] = np.mean(y_scaled[tr_mask])
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_scaled[tr_mask], sample_weight=weights)
                oof[i] = ridge.predict(X_tr_s[i:i+1])[0]

            tpred = np.zeros(n_te)
            if n_te > 0:
                D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
                for j in range(n_te):
                    weights = np.exp(-D_te[j, :] ** 2 / (2 * sigma ** 2))
                    if weights.sum() < 1e-10:
                        tpred[j] = np.mean(y_scaled[tr_mask])
                        continue
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X_tr_s, y_scaled[tr_mask], sample_weight=weights)
                    tpred[j] = ridge.predict(X_te_s[j:j+1])[0]

            frame_oofs.append(oof)
            frame_tests.append(tpred)

        avg_oof = np.mean(frame_oofs, axis=0)
        avg_test = np.mean(frame_tests, axis=0)
        oof_preds[tr_indices] = avg_oof
        if n_te > 0:
            test_preds[te_indices] = avg_test

    loo_mse = np.mean((oof_preds - y_scaled) ** 2)
    return oof_preds, test_preds, loo_mse


def main():
    t0 = time.time()
    print("=" * 70)
    print("PER-TARGET INDEPENDENT OPTIMIZATION")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    baseline_loo = {"angle": 0.002645, "depth": 0.004601, "left_right": 0.004331}

    # Sweep configs per target
    sweep_configs = []

    # Bandwidth x n_frames x spacing (core sweep)
    for bw in [0.30, 0.45, 0.55, 0.65, 0.80]:
        for nf in [1, 3, 5]:
            for spacing in [5, 10]:
                if nf == 1 and spacing == 10:
                    continue  # skip duplicate single-frame configs
                sweep_configs.append({
                    'bw': bw, 'n_frames': nf, 'spacing': spacing,
                    'pls_max': 15, 'include_ja': True, 'alpha': 10.0,
                    'name': f'bw={bw:.2f} nf={nf} sp={spacing} ja=Y pls=15'
                })

    # Test without joint angles at key bandwidths
    for bw in [0.45, 0.80]:
        sweep_configs.append({
            'bw': bw, 'n_frames': 3, 'spacing': 5,
            'pls_max': 15, 'include_ja': False, 'alpha': 10.0,
            'name': f'bw={bw:.2f} nf=3 sp=5 ja=N pls=15'
        })

    # Test different PLS max
    for pls_max in [5, 10, 20]:
        sweep_configs.append({
            'bw': 0.45, 'n_frames': 3, 'spacing': 5,
            'pls_max': pls_max, 'include_ja': True, 'alpha': 10.0,
            'name': f'bw=0.45 nf=3 sp=5 ja=Y pls={pls_max}'
        })

    # 7-frame ensemble at key bandwidths
    for bw in [0.45, 0.80]:
        sweep_configs.append({
            'bw': bw, 'n_frames': 7, 'spacing': 5,
            'pls_max': 15, 'include_ja': True, 'alpha': 10.0,
            'name': f'bw={bw:.2f} nf=7 sp=5 ja=Y pls=15'
        })

    print(f"\n{len(sweep_configs)} configs x 3 targets = {len(sweep_configs)*3} evaluations")
    print()

    # Run sweep for each target independently
    results_by_target = {t: [] for t in TARGETS}

    for target in TARGETS:
        print(f"\n{'='*50}")
        print(f"TARGET: {target} (baseline LOO: {baseline_loo[target]:.6f})")
        print(f"{'='*50}")

        for i, cfg in enumerate(sweep_configs):
            t1 = time.time()
            oof, tpred, loo = run_single_target(
                train_data, test_data, y_train, scalers[target], target, target_idx,
                bandwidth=cfg['bw'], n_frames=cfg['n_frames'], frame_spacing=cfg['spacing'],
                pls_max=cfg['pls_max'], include_ja=cfg['include_ja'], alpha=cfg['alpha'],
            )
            elapsed = time.time() - t1
            pct = (loo - baseline_loo[target]) / baseline_loo[target] * 100

            results_by_target[target].append({
                'config': cfg,
                'loo': loo,
                'pct': pct,
                'oof': oof,
                'test': tpred,
                'time': elapsed,
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(sweep_configs):
                print(f"  [{i+1}/{len(sweep_configs)}] {cfg['name']}: {loo:.6f} ({pct:+.2f}%) [{elapsed:.1f}s]")

    # Find best config per target
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION PER TARGET")
    print("=" * 70)

    best_per_target = {}
    for target in TARGETS:
        sorted_results = sorted(results_by_target[target], key=lambda x: x['loo'])
        best = sorted_results[0]
        best_per_target[target] = best
        print(f"\n{target} (baseline {baseline_loo[target]:.6f}):")
        for r in sorted_results[:5]:
            print(f"  {r['config']['name']}: {r['loo']:.6f} ({r['pct']:+.2f}%)")

    # Print combined best
    combined_loo = np.mean([best_per_target[t]['loo'] for t in TARGETS])
    baseline_mean = np.mean([baseline_loo[t] for t in TARGETS])
    pct = (combined_loo - baseline_mean) / baseline_mean * 100
    print(f"\nCombined best per-target LOO: {combined_loo:.6f} ({pct:+.2f}% vs baseline {baseline_mean:.6f})")

    for target in TARGETS:
        b = best_per_target[target]
        print(f"  {target}: {b['config']['name']} -> {b['loo']:.6f}")

    # Generate per-target optimized submission
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Standalone per-target optimized
    sub_num = get_next_submission_number()
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df = pd.DataFrame({'id': test_data['ids']})
    for target in TARGETS:
        sub_df[f"scaled_{target}"] = best_per_target[target]['test']
    sub_df.to_csv(sub_path, index=False)
    print(f"Sub {sub_num}: per-target optimized standalone (LOO {combined_loo:.6f})")

    # Blends with Sub 2402
    s_best = pd.read_csv(SUBMISSION_DIR / "submission_2402.csv").set_index('id')
    for w in [0.10, 0.20, 0.30, 0.50]:
        sn = get_next_submission_number()
        sp = SUBMISSION_DIR / f"submission_{sn}.csv"
        blended = pd.DataFrame({'id': test_data['ids']})
        for target in TARGETS:
            col = f"scaled_{target}"
            preds = best_per_target[target]['test']
            bl = []
            for i, tid in enumerate(test_data['ids']):
                bl.append(w * preds[i] + (1 - w) * s_best.loc[tid, col])
            blended[col] = bl
        blended.to_csv(sp, index=False)
        print(f"Sub {sn}: {int(w*100)}% per-target opt + {int((1-w)*100)}% Sub 2402")

    # Also generate: top-2 and top-3 per-target ensemble (average of best configs)
    for n_avg in [2, 3]:
        sn = get_next_submission_number()
        sp = SUBMISSION_DIR / f"submission_{sn}.csv"
        avg_df = pd.DataFrame({'id': test_data['ids']})
        for target in TARGETS:
            sorted_results = sorted(results_by_target[target], key=lambda x: x['loo'])
            avg_test = np.mean([sorted_results[k]['test'] for k in range(n_avg)], axis=0)
            avg_df[f"scaled_{target}"] = avg_test
        avg_df.to_csv(sp, index=False)
        print(f"Sub {sn}: top-{n_avg} avg per target (ensemble of best configs)")

    # Top-3 avg blended with Sub 2402
    for n_avg in [3]:
        for w in [0.10, 0.30]:
            sn = get_next_submission_number()
            sp = SUBMISSION_DIR / f"submission_{sn}.csv"
            blended = pd.DataFrame({'id': test_data['ids']})
            for target in TARGETS:
                col = f"scaled_{target}"
                sorted_results = sorted(results_by_target[target], key=lambda x: x['loo'])
                avg_test = np.mean([sorted_results[k]['test'] for k in range(n_avg)], axis=0)
                bl = []
                for i, tid in enumerate(test_data['ids']):
                    bl.append(w * avg_test[i] + (1 - w) * s_best.loc[tid, col])
                blended[col] = bl
            blended.to_csv(sp, index=False)
            print(f"Sub {sn}: {int(w*100)}% top-{n_avg}-avg + {int((1-w)*100)}% Sub 2402")

    total_time = time.time() - t0
    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
