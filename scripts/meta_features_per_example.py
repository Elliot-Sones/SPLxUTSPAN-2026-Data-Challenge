"""
Meta-Features + Per-Example Regression

Instead of blending OOF stacking predictions post-hoc (which loses accuracy),
USE them as additional features in the per-example locally weighted regression.

The per-example model gets:
- 198 HC features at target-specific frames (standard)
- 15 PLS components (standard)
- N meta-features from diverse models (NEW)

The meta-features are generated via LOPO CV to avoid leakage.
Each meta-feature is a prediction from a diverse model trained on different
feature representations. The per-example Ridge can learn how to weight
the diverse signal locally for each test shot.
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
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
    vel = np.zeros_like(ball * 0.3048)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball[:, ax] * 0.3048, 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)
    s, e = max(80, wrist_peak - 40), min(wrist_peak + 5, 200)
    return int(np.clip(s + np.argmax(speed[s:e]), 80, 200))


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
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


def extract_all_features(data, target):
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test, X_raw_train, X_raw_test):
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


# ================================================================
# DIVERSE META-FEATURE GENERATION
# ================================================================

def extract_hc_at_frame(data, frame):
    """Extract HC features at a specific frame."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def extract_joint_angles(data, frame):
    """Extract joint angles at a specific frame."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    feats_all = []

    joint_triples = [
        ('right_shoulder', 'right_elbow', 'right_wrist'),
        ('right_hip', 'right_shoulder', 'right_elbow'),
        ('right_hip', 'right_knee', 'right_ankle'),
        ('left_shoulder', 'left_elbow', 'left_wrist'),
        ('left_hip', 'left_shoulder', 'left_elbow'),
        ('left_hip', 'left_knee', 'left_ankle'),
        ('neck', 'right_shoulder', 'right_elbow'),
        ('neck', 'left_shoulder', 'left_elbow'),
    ]

    for i in range(n):
        ts_3d = data['X_3d'][i]
        f = int(np.clip(frame, 0, 239))
        shot_feats = []
        for a_name, b_name, c_name in joint_triples:
            a_idx = kp_index.get(a_name)
            b_idx = kp_index.get(b_name)
            c_idx = kp_index.get(c_name)
            if all(idx is not None for idx in [a_idx, b_idx, c_idx]):
                ba = ts_3d[f, a_idx] - ts_3d[f, b_idx]
                bc = ts_3d[f, c_idx] - ts_3d[f, b_idx]
                ba_n = np.linalg.norm(ba)
                bc_n = np.linalg.norm(bc)
                if ba_n > 1e-6 and bc_n > 1e-6:
                    cos_angle = np.clip(np.dot(ba, bc) / (ba_n * bc_n), -1, 1)
                    shot_feats.append(np.degrees(np.arccos(cos_angle)))
                else:
                    shot_feats.append(90.0)
            else:
                shot_feats.append(90.0)
        feats_all.append(shot_feats)
    return np.array(feats_all, dtype=np.float32)


def extract_body_segments(data, frame):
    """Extract body segment lengths and ratios."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    feats_all = []

    segments = [
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_shoulder', 'right_hip'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'left_shoulder'),
        ('right_hip', 'left_hip'),
    ]

    for i in range(n):
        ts_3d = data['X_3d'][i]
        f = int(np.clip(frame, 0, 239))
        shot_feats = []
        for a_name, b_name in segments:
            a_idx = kp_index.get(a_name)
            b_idx = kp_index.get(b_name)
            if a_idx is not None and b_idx is not None:
                shot_feats.append(np.linalg.norm(ts_3d[f, a_idx] - ts_3d[f, b_idx]))
            else:
                shot_feats.append(0.0)
        # Add ratios
        if len(shot_feats) >= 4 and shot_feats[0] > 1e-6:
            shot_feats.append(shot_feats[1] / (shot_feats[0] + 1e-8))  # forearm/upper arm
        else:
            shot_feats.append(1.0)
        if len(shot_feats) >= 6 and shot_feats[4] > 1e-6:
            shot_feats.append(shot_feats[5] / (shot_feats[4] + 1e-8))  # shin/thigh
        else:
            shot_feats.append(1.0)
        feats_all.append(shot_feats)
    return np.array(feats_all, dtype=np.float32)


def extract_velocity_features(data, frame):
    """Extract velocity profiles at key frames."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'right_hip', 'right_knee', 'neck']
    feats_all = []

    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        shot_feats = []
        for jname in key_joints:
            idx = kp_index.get(jname)
            if idx is None:
                shot_feats.extend([0.0] * 6)
                continue
            for coord in range(3):
                series = ts_hr[:, idx, coord]
                vel = np.gradient(series, DT)
                acc = np.gradient(vel, DT)
                shot_feats.append(vel[frame])
                shot_feats.append(acc[frame])
        feats_all.append(shot_feats)
    X = np.array(feats_all, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def generate_meta_features_lopo(train_data, test_data, y_scaled, target):
    """Generate diverse meta-features using LOPO CV.

    For each diverse feature space + model, generate:
    - OOF predictions (345 values) via LOPO
    - Test predictions (113 values) via full training

    Returns: meta_train (345 x N_meta), meta_test (113 x N_meta), meta_names
    """
    pids = train_data['pids']
    pids_test = test_data['pids']
    unique_pids = sorted(np.unique(pids))
    n_train = len(pids)
    n_test = len(pids_test)
    y = y_scaled

    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    frame = TARGET_FRAMES[target]

    # Define diverse feature spaces
    print(f"    Extracting diverse feature spaces...")
    feature_spaces = {}

    # 1. HC at different frames (diverse temporal windows)
    for f in [120, 140, 200]:
        if f != frame:  # Skip the target-specific frame (already in main features)
            feature_spaces[f'hc_f{f}'] = (
                extract_hc_at_frame(train_data, f),
                extract_hc_at_frame(test_data, f)
            )

    # 2. Joint angles at target frame
    feature_spaces['joint_angles'] = (
        extract_joint_angles(train_data, frame),
        extract_joint_angles(test_data, frame)
    )

    # 3. Body segments at target frame
    feature_spaces['body_segments'] = (
        extract_body_segments(train_data, frame),
        extract_body_segments(test_data, frame)
    )

    # 4. Velocity features at target frame
    feature_spaces['velocity'] = (
        extract_velocity_features(train_data, frame),
        extract_velocity_features(test_data, frame)
    )

    # 5. PLS on raw timeseries (target-specific)
    y_raw = train_data['y'][:, target_idx[target]]
    pls_tr = np.zeros((n_train, 15), dtype=np.float32)
    pls_te = np.zeros((n_test, 15), dtype=np.float32)
    for pid in unique_pids:
        tr_mask = pids == pid
        te_mask = pids_test == pid
        scaler = StandardScaler()
        raw_tr = scaler.fit_transform(train_data['X_raw'][tr_mask])
        raw_te = scaler.transform(test_data['X_raw'][te_mask])
        nc = min(15, tr_mask.sum() - 5)
        nc = max(3, nc)
        pls = PLSRegression(n_components=nc)
        pls.fit(raw_tr, y_raw[tr_mask])
        pls_tr[tr_mask, :nc] = pls.transform(raw_tr)
        pls_te[te_mask, :nc] = pls.transform(raw_te)
    feature_spaces[f'pls_{target}'] = (pls_tr, pls_te)

    # Define models to use
    model_configs = [
        ('ridge_10', lambda: Ridge(alpha=10.0)),
        ('ridge_100', lambda: Ridge(alpha=100.0)),
        ('lgb_4', lambda: lgb.LGBMRegressor(
            n_estimators=100, num_leaves=4, learning_rate=0.05,
            min_child_samples=5, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, n_jobs=-1, random_state=42)),
        ('xgb_2', lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            reg_alpha=1.0, reg_lambda=1.0, verbosity=0, n_jobs=-1, random_state=42)),
    ]

    # Generate LOPO OOF + test predictions
    meta_train = []
    meta_test = []
    meta_names = []

    print(f"    Generating LOPO meta-features...")
    for fs_name, (X_fs_train, X_fs_test) in feature_spaces.items():
        for model_name, model_fn in model_configs:
            oof = np.zeros(n_train)
            test_pred = np.zeros(n_test)

            # LOPO CV for OOF
            for val_pid in unique_pids:
                val_mask = pids == val_pid
                train_mask = ~val_mask

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_fs_train[train_mask])
                X_val = scaler.transform(X_fs_train[val_mask])

                model = model_fn()
                model.fit(X_tr, y[train_mask])
                oof[val_mask] = model.predict(X_val)

            # Full model for test
            scaler = StandardScaler()
            X_tr_full = scaler.fit_transform(X_fs_train)
            X_te_full = scaler.transform(X_fs_test)
            model = model_fn()
            model.fit(X_tr_full, y)
            test_pred = model.predict(X_te_full)

            lopo_mse = np.mean((oof - y) ** 2)
            name = f'{fs_name}__{model_name}'

            meta_train.append(oof)
            meta_test.append(test_pred)
            meta_names.append(name)

    meta_train = np.column_stack(meta_train)
    meta_test = np.column_stack(meta_test)

    print(f"    Generated {len(meta_names)} meta-features")
    return meta_train, meta_test, meta_names


def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.5, alpha=10.0):
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))
    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))
        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))
        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(all_dists, bandwidth_quantile) if len(all_dists) > 0 else 1.0
        sigma = max(sigma, 1e-6)
        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]
        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]
    return oof_preds, test_preds


def main():
    t0 = time.time()
    print("=" * 70)
    print("META-FEATURES + PER-EXAMPLE REGRESSION")
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

    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
    sub_1640 = pd.read_csv(SUBMISSION_DIR / "submission_1640.csv")

    baseline_mse = {'angle': 0.002511, 'depth': 0.004510, 'left_right': 0.004209}

    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]

        # Standard features (same as Sub 1350)
        print("  Extracting standard features...")
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        n_standard = X_train_aug.shape[1]
        print(f"  Standard features: {n_standard}")

        # Baseline: standard per-example regression
        print("  Running baseline per-example regression...")
        oof_baseline, test_baseline = locally_weighted_prediction(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_baseline = np.mean((oof_baseline - y_target) ** 2)
        print(f"  Baseline LOO MSE: {mse_baseline:.6f}")

        # Generate diverse meta-features
        print("  Generating meta-features...")
        meta_train, meta_test, meta_names = generate_meta_features_lopo(
            train_data, test_data, y_target, target)

        # Compute diversity of meta-features with baseline
        print(f"\n  Meta-feature diversity with baseline:")
        for i, name in enumerate(meta_names):
            r = np.corrcoef(oof_baseline, meta_train[:, i])[0, 1]
            mse = np.mean((meta_train[:, i] - y_target) ** 2)
            if abs(r) < 0.80:
                print(f"    {name}: r={r:.4f}, LOPO_MSE={mse:.6f} ***DIVERSE***")

        # Test different meta-feature configurations
        configs = {}

        # Config 1: All meta-features appended
        X_train_meta_all = np.hstack([X_train_aug, meta_train])
        X_test_meta_all = np.hstack([X_test_aug, meta_test])
        print(f"\n  Config 1: All meta-features ({X_train_meta_all.shape[1]} features)")
        oof_1, test_1 = locally_weighted_prediction(
            X_train_meta_all, y_target, X_test_meta_all, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_1 = np.mean((oof_1 - y_target) ** 2)
        delta_1 = (mse_1 - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_1:.6f} ({delta_1:+.1f}% vs baseline)")
        configs['all_meta'] = (mse_1, oof_1, test_1)

        # Config 2: Only diverse meta-features (r < 0.70 with baseline)
        diverse_idx = []
        for i, name in enumerate(meta_names):
            r = np.corrcoef(oof_baseline, meta_train[:, i])[0, 1]
            if abs(r) < 0.70:
                diverse_idx.append(i)
        if diverse_idx:
            X_train_meta_div = np.hstack([X_train_aug, meta_train[:, diverse_idx]])
            X_test_meta_div = np.hstack([X_test_aug, meta_test[:, diverse_idx]])
            print(f"\n  Config 2: Diverse meta-features only ({len(diverse_idx)} meta, {X_train_meta_div.shape[1]} total)")
            oof_2, test_2 = locally_weighted_prediction(
                X_train_meta_div, y_target, X_test_meta_div, pids_train, pids_test,
                bandwidth_quantile=0.5, alpha=10.0)
            mse_2 = np.mean((oof_2 - y_target) ** 2)
            delta_2 = (mse_2 - mse_baseline) / mse_baseline * 100
            print(f"  LOO MSE: {mse_2:.6f} ({delta_2:+.1f}% vs baseline)")
            configs['diverse_meta'] = (mse_2, oof_2, test_2)

        # Config 3: Top-5 meta by LOPO MSE
        meta_mses = []
        for i in range(meta_train.shape[1]):
            meta_mses.append(np.mean((meta_train[:, i] - y_target) ** 2))
        top5_idx = np.argsort(meta_mses)[:5]
        X_train_meta_top5 = np.hstack([X_train_aug, meta_train[:, top5_idx]])
        X_test_meta_top5 = np.hstack([X_test_aug, meta_test[:, top5_idx]])
        print(f"\n  Config 3: Top-5 meta by accuracy ({X_train_meta_top5.shape[1]} features)")
        for idx in top5_idx:
            print(f"    {meta_names[idx]}: LOPO_MSE={meta_mses[idx]:.6f}")
        oof_3, test_3 = locally_weighted_prediction(
            X_train_meta_top5, y_target, X_test_meta_top5, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_3 = np.mean((oof_3 - y_target) ** 2)
        delta_3 = (mse_3 - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_3:.6f} ({delta_3:+.1f}% vs baseline)")
        configs['top5_meta'] = (mse_3, oof_3, test_3)

        # Config 4: LASSO-selected features + top-3 diverse meta
        from sklearn.linear_model import Lasso as LassoModel
        lasso_idx = []
        for pid in sorted(np.unique(pids_train)):
            mask = pids_train == pid
            sc = StandardScaler()
            X_s = sc.fit_transform(X_train_aug[mask])
            for b in range(30):
                rng = np.random.RandomState(b)
                idx = rng.choice(len(X_s), len(X_s), replace=True)
                lasso = LassoModel(alpha=0.01, max_iter=5000)
                lasso.fit(X_s[idx], y_target[mask][idx])
                for j in np.where(np.abs(lasso.coef_) > 1e-6)[0]:
                    if j not in lasso_idx:
                        lasso_idx.append(j)
        lasso_idx = sorted(set(lasso_idx))[:30]

        # Select top-3 diverse meta
        diverse_mses = []
        for i, name in enumerate(meta_names):
            r = np.corrcoef(oof_baseline, meta_train[:, i])[0, 1]
            mse_i = meta_mses[i]
            diverse_mses.append((i, abs(r), mse_i))
        # Sort by diversity (low r), then by MSE (low is better)
        diverse_mses.sort(key=lambda x: (x[1], x[2]))
        top3_diverse = [x[0] for x in diverse_mses[:3]]

        X_train_lasso_meta = np.hstack([X_train_aug[:, lasso_idx], meta_train[:, top3_diverse]])
        X_test_lasso_meta = np.hstack([X_test_aug[:, lasso_idx], meta_test[:, top3_diverse]])
        print(f"\n  Config 4: LASSO-30 + top-3 diverse meta ({X_train_lasso_meta.shape[1]} features)")
        for idx in top3_diverse:
            r = np.corrcoef(oof_baseline, meta_train[:, idx])[0, 1]
            print(f"    {meta_names[idx]}: r_baseline={r:.4f}, MSE={meta_mses[idx]:.6f}")
        oof_4, test_4 = locally_weighted_prediction(
            X_train_lasso_meta, y_target, X_test_lasso_meta, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_4 = np.mean((oof_4 - y_target) ** 2)
        delta_4 = (mse_4 - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_4:.6f} ({delta_4:+.1f}% vs baseline)")
        configs['lasso_plus_meta'] = (mse_4, oof_4, test_4)

        # Summary
        print(f"\n  {target.upper()} SUMMARY:")
        print(f"    Baseline (213 features): {mse_baseline:.6f}")
        for name, (mse, _, _) in sorted(configs.items(), key=lambda x: x[1][0]):
            delta = (mse - mse_baseline) / mse_baseline * 100
            print(f"    {name}: {mse:.6f} ({delta:+.1f}%)")

        best_name = min(configs, key=lambda k: configs[k][0])
        all_results[target] = {
            'baseline': (mse_baseline, oof_baseline, test_baseline),
            'configs': configs,
            'best_name': best_name,
            'best': configs[best_name],
        }

    # ================================================================
    # GENERATE SUBMISSIONS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # For each target, use the best config
    best_test = {}
    for target in TARGETS:
        best_name = all_results[target]['best_name']
        best_test[target] = all_results[target]['best'][2]
        delta = (all_results[target]['best'][0] - all_results[target]['baseline'][0]) / all_results[target]['baseline'][0] * 100
        print(f"  {target}: {best_name} ({delta:+.1f}% vs baseline)")

    # Standalone
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': best_test['angle'],
        'scaled_depth': best_test['depth'],
        'scaled_left_right': best_test['left_right'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: STANDALONE best-per-target meta-features")

    # Blends with Sub 1350
    for pct in [0.05, 0.10, 0.15, 0.20]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for col, t in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[col] = (1-pct) * sub_1350[col] + pct * best_test[t]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(pct*100)}% meta-features + {int((1-pct)*100)}% Sub 1350")

    # Blends with Sub 1640
    for pct in [0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_1640.copy()
        for col, t in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[col] = (1-pct) * sub_1640[col] + pct * best_test[t]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(pct*100)}% meta-features + {int((1-pct)*100)}% Sub 1640")

    # Diversity analysis
    print(f"\n  Diversity with Sub 1350:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_1350[col].values, best_test[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    print(f"\n  Diversity with Sub 1640:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_1640[col].values, best_test[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
