"""
Unified Winning Pipeline - Combining All Proven Signals

Integrates four confirmed LB-improving techniques into ONE model:
1. Per-player optimal frames (Sub 2372: LB 0.006538, -13.61% LOO)
2. Adaptive bandwidth bw=0.45 (Sub 2354: -5.06% LOO)
3. Joint angle features (Sub 2020: LB 0.006619, -5.7% LOO)
4. Multi-frame ensemble (Sub 2169: LB 0.006552, -4.68% LOO)

Architecture:
  For each player x target:
    - Use per-player optimal center frame
    - Extract features at center, center-5, center+5 (3-frame ensemble)
    - Include 10 joint angle features
    - Use bandwidth_quantile=0.45
    - Per-player PLS augmentation (15 components)
    - Locally weighted Ridge (alpha=10)
    - Average predictions across 3 frames

This is NOT a blend of separate CSVs. It is one integrated model.

Outputs:
  - Standalone submission (unified only)
  - Blend submissions: 10%, 20%, 30% unified + rest Sub 2169
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

# Per-player optimal frames from per_player_frame_optimization.py LOO search
# These were discovered on 2026-02-14 via frame sweep per player x target
PLAYER_FRAMES = {
    "angle": {1: 140, 2: 155, 3: 165, 4: 150, 5: 150},
    "depth": {1: 155, 2: 140, 3: 130, 4: 180, 5: 140},
    "left_right": {1: 185, 2: 130, 3: 140, 4: 180, 5: 145},
}

# Multi-frame offsets (3-frame ensemble: center, center-5, center+5)
FRAME_OFFSETS = [-5, 0, 5]

# Bandwidth: adaptive search showed bw=0.45 optimal for most players
BANDWIDTH_QUANTILE = 0.45
RIDGE_ALPHA = 10.0


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
    """Angle in degrees between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


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
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


# ==============================================================
# FEATURE EXTRACTION
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
    """Extract 198 hoop-relative features + 10 joint angle features at a specific frame."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    # Hoop-relative positions + velocities at target frame
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])

    # Hoop-relative summary stats
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

    # Arm mechanics
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

    # Body alignment
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

    # Guide hand
    lw = kp_index.get('left_wrist')
    if lw is not None and rw is not None:
        feats.append(ts_hr[f, lw, 1] - ts_hr[f, rw, 1])
    else:
        feats.append(0.0)

    # Release frame timing
    feats.append(release_frame)

    # Release window dynamics
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats.append(np.nanmean(series[140:180]))
            feats.append(np.nanmax(vel[140:180]))
    else:
        feats.extend([0.0] * 6)

    # ---- 10 Joint Angle Features (body-proportion invariant) ----
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

    # 6. Wrist deviation from vertical
    if re_i is not None and rw_i is not None:
        forearm = ts_3d[f, rw_i] - ts_3d[f, re_i]
        vertical = np.array([0, 0, 1], dtype=np.float32)
        feats.append(_angle_between(forearm, vertical))
    else:
        feats.append(90.0)

    # 7. Arm line angle vs hoop direction
    if rs_i is not None and rw_i is not None:
        arm_line = ts_hr[f, rw_i] - ts_hr[f, rs_i]
        hoop_dir = np.array([1, 0, 0.5], dtype=np.float32)
        feats.append(_angle_between(arm_line, hoop_dir))
    else:
        feats.append(90.0)

    # 8. Shoulder rotation
    if rs_i is not None and ls_i is not None:
        shoulder_line = ts_hr[f, rs_i] - ts_hr[f, ls_i]
        lateral = np.array([0, 1, 0], dtype=np.float32)
        feats.append(_angle_between(shoulder_line, lateral))
    else:
        feats.append(90.0)

    # 9. Hip-shoulder twist
    if all(x is not None for x in [rh_i, lh_i, rs_i, ls_i]):
        hip_line = ts_hr[f, rh_i, :2] - ts_hr[f, lh_i, :2]
        shoulder_line_xy = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
        hn = np.linalg.norm(hip_line)
        sn = np.linalg.norm(shoulder_line_xy)
        if hn > 1e-6 and sn > 1e-6:
            cos_a = np.clip(np.dot(hip_line, shoulder_line_xy) / (hn * sn), -1, 1)
            feats.append(np.degrees(np.arccos(cos_a)))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)

    # 10. Elbow height relative to shoulder
    if re_i is not None and rs_i is not None:
        feats.append(ts_hr[f, re_i, 2] - ts_hr[f, rs_i, 2])
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


# ==============================================================
# PER-PLAYER FEATURE EXTRACTION AT SPECIFIC FRAME
# ==============================================================

def extract_player_features(data, player_mask, kp_index, frame):
    """Extract features for a set of shots at a specific frame."""
    indices = np.where(player_mask)[0]
    all_feats = []
    for i in indices:
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ==============================================================
# PLS AUGMENTATION (per-player)
# ==============================================================

def pls_augment_player(X_raw_train, X_raw_test, y_raw_player, n_max=15):
    """Compute PLS components for a single player."""
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
# CORE: LOCALLY WEIGHTED RIDGE LOO + TEST (single frame, single player)
# ==============================================================

def lw_ridge_player(X_tr_aug, X_te_aug, y_player,
                    bandwidth_quantile=BANDWIDTH_QUANTILE, alpha=RIDGE_ALPHA):
    """Locally weighted Ridge: LOO predictions for train, predictions for test.
    Returns (oof_preds, test_preds)."""
    n_tr = len(X_tr_aug)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_aug)
    X_te_s = scaler.transform(X_te_aug) if len(X_te_aug) > 0 else np.zeros((0, X_tr_aug.shape[1]))

    D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
    all_dists = D_tr[np.triu_indices(n_tr, k=1)]
    if len(all_dists) > 0:
        sigma = np.quantile(all_dists, bandwidth_quantile)
        sigma = max(sigma, 1e-6)
    else:
        sigma = 1.0

    # LOO
    oof = np.zeros(n_tr)
    for i in range(n_tr):
        dists = D_tr[i, :]
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
        weights[i] = 0
        if weights.sum() < 1e-10:
            oof[i] = np.mean(y_player)
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_s, y_player, sample_weight=weights)
        oof[i] = ridge.predict(X_tr_s[i:i+1])[0]

    # Test
    test_preds = np.zeros(len(X_te_s))
    if len(X_te_s) > 0:
        D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
        for j in range(len(X_te_s)):
            dists = D_te[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                test_preds[j] = np.mean(y_player)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_player, sample_weight=weights)
            test_preds[j] = ridge.predict(X_te_s[j:j+1])[0]

    return oof, test_preds


# ==============================================================
# UNIFIED PIPELINE
# ==============================================================

def run_unified(train_data, test_data, y_train, scalers, target_idx):
    """Run the full unified pipeline with all 4 winning signals.
    Returns per-target LOO predictions, test predictions, and LOO MSE."""

    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    unique_pids = sorted(np.unique(pids_train))

    results = {}

    for target in TARGETS:
        print(f"\n--- Target: {target} ---")
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_scaled = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        oof_preds = np.zeros(len(pids_train))
        test_preds = np.zeros(len(pids_test))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            tr_indices = np.where(tr_mask)[0]
            te_indices = np.where(te_mask)[0]
            n_tr = len(tr_indices)
            n_te = len(te_indices)

            center_frame = PLAYER_FRAMES[target][pid]
            frames = [center_frame + off for off in FRAME_OFFSETS]
            # Clip to valid range
            frames = [int(np.clip(f, 0, 239)) for f in frames]

            # PLS augmentation (shared across frames - uses raw timeseries, frame-independent)
            pls_train, pls_test = pls_augment_player(
                train_data['X_raw'][tr_mask],
                test_data['X_raw'][te_mask] if n_te > 0 else np.zeros((0, train_data['X_raw'].shape[1])),
                y_raw[tr_mask]
            )

            # Multi-frame ensemble: build model at each frame, average predictions
            frame_oofs = []
            frame_tests = []

            for frame in frames:
                # Extract features at this frame for this player
                X_tr_hc = extract_player_features(train_data, tr_mask, kp_index, frame)
                X_te_hc = extract_player_features(test_data, te_mask, kp_index, frame) if n_te > 0 \
                    else np.zeros((0, X_tr_hc.shape[1]))

                # Augment with PLS
                X_tr_aug = np.hstack([X_tr_hc, pls_train])
                X_te_aug = np.hstack([X_te_hc, pls_test]) if n_te > 0 \
                    else np.zeros((0, X_tr_aug.shape[1]))

                # Locally weighted Ridge
                oof, tpred = lw_ridge_player(X_tr_aug, X_te_aug, y_scaled[tr_mask])
                frame_oofs.append(oof)
                frame_tests.append(tpred)

            # Average across frames
            avg_oof = np.mean(frame_oofs, axis=0)
            avg_test = np.mean(frame_tests, axis=0)

            oof_preds[tr_indices] = avg_oof
            if n_te > 0:
                test_preds[te_indices] = avg_test

            loo_mse_player = np.mean((avg_oof - y_scaled[tr_mask]) ** 2)
            print(f"  Player {pid} (n={n_tr}): center_frame={center_frame}, "
                  f"frames={frames}, LOO MSE={loo_mse_player:.6f}")

        loo_mse = np.mean((oof_preds - y_scaled) ** 2)
        print(f"  {target} LOO MSE: {loo_mse:.6f}")

        results[target] = {
            'oof': oof_preds,
            'test': test_preds,
            'loo_mse': loo_mse,
            'y_scaled': y_scaled,
        }

    mean_loo = np.mean([results[t]['loo_mse'] for t in TARGETS])
    print(f"\nMean LOO MSE: {mean_loo:.6f}")
    return results


def save_submission(test_data, results, scalers, tag=""):
    """Save submission CSV in wide format (id, scaled_angle, scaled_depth, scaled_left_right).
    Predictions are already in scaled [0,1] space - do NOT inverse transform."""
    sub_num = get_next_submission_number()
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"

    sub_df = pd.DataFrame({'id': test_data['ids']})
    for target in TARGETS:
        # Predictions are in scaled space already - store directly
        sub_df[f"scaled_{target}"] = results[target]['test']

    sub_df.to_csv(sub_path, index=False)
    print(f"\nSub {sub_num}{tag}: {sub_path}")
    return sub_num, sub_path


def save_blend_with_existing(test_data, results, scalers, existing_sub_num, blend_weight, tag=""):
    """Blend unified predictions with an existing submission (wide format).
    Both predictions are in scaled [0,1] space."""
    existing_path = SUBMISSION_DIR / f"submission_{existing_sub_num}.csv"
    if not existing_path.exists():
        print(f"  WARNING: {existing_path} not found, skipping blend")
        return None, None

    existing_df = pd.read_csv(existing_path)
    sub_num = get_next_submission_number()
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"

    # Build lookup from existing submission
    existing_lookup = existing_df.set_index('id')

    sub_df = pd.DataFrame({'id': test_data['ids']})
    for target in TARGETS:
        scaled_col = f"scaled_{target}"
        preds = results[target]['test']  # already in scaled space

        blended = []
        for i, tid in enumerate(test_data['ids']):
            if tid in existing_lookup.index:
                existing_val = existing_lookup.loc[tid, scaled_col]
                blended.append(blend_weight * preds[i] + (1 - blend_weight) * existing_val)
            else:
                blended.append(preds[i])
        sub_df[scaled_col] = blended

    sub_df.to_csv(sub_path, index=False)
    desc = f"{int(blend_weight*100)}% unified + {int((1-blend_weight)*100)}% Sub {existing_sub_num}"
    print(f"Sub {sub_num} ({desc}){tag}: {sub_path}")
    return sub_num, sub_path


def main():
    t0 = time.time()
    print("=" * 70)
    print("UNIFIED WINNING PIPELINE")
    print("  Per-player frames + bw=0.45 + joint angles + 3-frame ensemble")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']

    # Load scalers
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Run unified pipeline
    results = run_unified(train_data, test_data, y_train, scalers, target_idx)

    # Print LOO comparison vs known baselines
    print("\n" + "=" * 70)
    print("LOO COMPARISON")
    print("=" * 70)
    baseline_loo = {"angle": 0.002645, "depth": 0.004601, "left_right": 0.004331}
    baseline_mean = 0.003859
    for target in TARGETS:
        loo = results[target]['loo_mse']
        bl = baseline_loo[target]
        pct = (loo - bl) / bl * 100
        print(f"  {target:12s}: {loo:.6f} (baseline {bl:.6f}, {pct:+.2f}%)")
    mean_loo = np.mean([results[t]['loo_mse'] for t in TARGETS])
    pct_mean = (mean_loo - baseline_mean) / baseline_mean * 100
    print(f"  {'MEAN':12s}: {mean_loo:.6f} (baseline {baseline_mean:.6f}, {pct_mean:+.2f}%)")

    # Save standalone submission
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)
    standalone_num, _ = save_submission(test_data, results, scalers, tag=" (unified standalone)")

    # Blend with Sub 2169 (current LB best: 0.006552) at multiple weights
    BEST_SUB = 2169
    blend_subs = {}
    for w in [0.10, 0.20, 0.30]:
        sn, _ = save_blend_with_existing(test_data, results, scalers, BEST_SUB, w)
        if sn:
            blend_subs[w] = sn

    # Also blend with Sub 2372 (current actual LB best: 0.006538)
    BEST_SUB_2 = 2372
    for w in [0.10, 0.20, 0.30]:
        sn, _ = save_blend_with_existing(test_data, results, scalers, BEST_SUB_2, w)
        if sn:
            blend_subs[(w, 2372)] = sn

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Print submission summary
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"  Standalone: Sub {standalone_num}")
    for k, v in blend_subs.items():
        if isinstance(k, tuple):
            w, base = k
            print(f"  {int(w*100)}% unified + {int((1-w)*100)}% Sub {base}: Sub {v}")
        else:
            print(f"  {int(k*100)}% unified + {int((1-k)*100)}% Sub {BEST_SUB}: Sub {v}")

    print("\n" + "=" * 70)
    print("LB TESTING PRIORITY")
    print("=" * 70)
    print("  1. 10% unified + 90% Sub 2372 (safest, proven sweet spot)")
    print("  2. 10% unified + 90% Sub 2169")
    print("  3. 20% unified + 80% Sub 2372")
    print("  4. Standalone (riskiest, highest potential)")


if __name__ == "__main__":
    main()
