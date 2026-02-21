"""
Random Subspace Ensemble for Variance Reduction

Train multiple per-example locally weighted Ridge models on random feature
subsets and average predictions. Classic random subspace method applied to
our best pipeline.

Methodology:
1. Test subset sizes: 50%, 60%, 70%, 80% of features
2. Test stratified vs fully random subset selection
3. Find optimal N (number of subsets): 5, 10, 20, 30
4. Stability analysis across 3 master seeds
5. Generate blended submissions
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
    """Angle in degrees between two 3D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_a))


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract features including joint angles (223 total)."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    # Hoop-relative positions + velocities at target frame (72 features)
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])

    # Hoop-relative summary stats (108 features)
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

    # Arm mechanics (7 features)
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

    # Body alignment (4 features)
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
    lw_kp = kp_index.get('left_wrist')
    if lw_kp is not None and rw is not None:
        feats.append(ts_hr[f, lw_kp, 1] - ts_hr[f, rw, 1])
    else:
        feats.append(0.0)

    # Release timing (1 feature)
    feats.append(release_frame)

    # Release window dynamics (6 features)
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats.append(np.nanmean(series[140:180]))
            feats.append(np.nanmax(vel[140:180]))
    else:
        feats.extend([0.0] * 6)

    # Joint angle features (10 features) - from five_approaches.py
    rw_i, re_i, rs_i = rw, re, rs
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

    # 6. Wrist deviation (forearm vs vertical)
    if re_i is not None and rw_i is not None:
        forearm = ts_3d[f, rw_i] - ts_3d[f, re_i]
        vertical = np.array([0, 0, 1], dtype=np.float32)
        feats.append(_angle_between(forearm, vertical))
    else:
        feats.append(90.0)

    # 7. Arm line angle (shoulder-to-wrist vs hoop direction)
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
        shoulder_line = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
        hn = np.linalg.norm(hip_line)
        sn = np.linalg.norm(shoulder_line)
        if hn > 1e-6 and sn > 1e-6:
            cos_a = np.clip(np.dot(hip_line, shoulder_line) / (hn * sn), -1, 1)
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


# ==============================================================
# FEATURE GROUP DEFINITIONS
# ==============================================================

def get_feature_groups(n_features):
    """Define feature groups for stratified sampling.

    Feature layout (with joint angles, 223 total):
    - Position + velocity at target frame: 12 joints x 6 = 72 (indices 0-71)
    - Summary stats: 12 joints x 9 = 108 (indices 72-179)
    - Arm mechanics: 7 (indices 180-186)
    - Body alignment: 4 (indices 187-190)
    - Release timing: 1 (index 191)
    - Release window dynamics: 6 (indices 192-197)
    - Joint angles: 10 (indices 198-207)
    - PLS components: 15 (indices 208-222)
    """
    groups = {}
    groups['position_velocity'] = list(range(0, 72))
    groups['summary_stats'] = list(range(72, 180))
    groups['arm_mechanics'] = list(range(180, 187))
    groups['body_alignment'] = list(range(187, 191))
    groups['release_timing'] = [191]
    groups['release_dynamics'] = list(range(192, 198))
    groups['joint_angles'] = list(range(198, 208))
    groups['pls'] = list(range(208, min(223, n_features)))

    # Cap to actual n_features
    for g in groups:
        groups[g] = [i for i in groups[g] if i < n_features]

    return groups


def generate_random_subsets(n_features, n_subsets, frac, rng, stratified=False):
    """Generate random feature subsets.

    If stratified=True, sample proportionally from each feature group.
    If stratified=False, fully random.
    """
    n_select = max(1, int(n_features * frac))
    subsets = []

    if stratified:
        groups = get_feature_groups(n_features)
        group_names = list(groups.keys())
        group_sizes = {g: len(groups[g]) for g in group_names}
        total = sum(group_sizes.values())

        for _ in range(n_subsets):
            selected = []
            for g in group_names:
                g_feats = groups[g]
                n_from_group = max(1, int(round(n_select * len(g_feats) / total)))
                n_from_group = min(n_from_group, len(g_feats))
                chosen = rng.choice(g_feats, size=n_from_group, replace=False)
                selected.extend(chosen.tolist())
            # Trim or pad to exact n_select
            if len(selected) > n_select:
                selected = rng.choice(selected, size=n_select, replace=False).tolist()
            elif len(selected) < n_select:
                remaining = [i for i in range(n_features) if i not in selected]
                extra = rng.choice(remaining, size=n_select - len(selected), replace=False)
                selected.extend(extra.tolist())
            subsets.append(sorted(selected))
    else:
        for _ in range(n_subsets):
            selected = rng.choice(n_features, size=n_select, replace=False)
            subsets.append(sorted(selected.tolist()))

    return subsets


# ==============================================================
# LOCALLY WEIGHTED PREDICTION ON FEATURE SUBSET
# ==============================================================

def locally_weighted_subset(X_train, y_train, X_test, pids_train, pids_test,
                            feature_indices, bandwidth_quantile=0.3, alpha=10.0,
                            recompute_bandwidth=True):
    """Run locally weighted Ridge on a feature subset.

    Returns (oof_preds, test_preds).
    """
    X_train_sub = X_train[:, feature_indices]
    X_test_sub = X_test[:, feature_indices]
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train_sub[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test_sub[te_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        # OOF: leave-one-out
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

        # Test predictions
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


def random_subspace_ensemble(X_train, y_train, X_test, pids_train, pids_test,
                             n_subsets=20, frac=0.70, seed=42, stratified=False,
                             bandwidth_quantile=0.3, alpha=10.0):
    """Run random subspace ensemble: train multiple locally weighted Ridge
    on random feature subsets and average.

    Returns:
        oof_preds: averaged OOF predictions
        test_preds: averaged test predictions
        per_subset_oof: list of individual OOF predictions
        per_subset_test: list of individual test predictions
    """
    rng = np.random.RandomState(seed)
    n_features = X_train.shape[1]
    subsets = generate_random_subsets(n_features, n_subsets, frac, rng, stratified=stratified)

    per_subset_oof = []
    per_subset_test = []

    for s_i, feat_idx in enumerate(subsets):
        oof, test = locally_weighted_subset(
            X_train, y_train, X_test, pids_train, pids_test,
            feat_idx, bandwidth_quantile=bandwidth_quantile, alpha=alpha)
        per_subset_oof.append(oof)
        per_subset_test.append(test)

        if (s_i + 1) % 5 == 0 or s_i == 0:
            # Running average
            avg_oof = np.mean(per_subset_oof, axis=0)
            mse = np.mean((avg_oof - y_train) ** 2)
            print(f"      Subset {s_i+1}/{n_subsets}: running avg LOO MSE={mse:.6f}")

    oof_preds = np.mean(per_subset_oof, axis=0)
    test_preds = np.mean(per_subset_test, axis=0)

    return oof_preds, test_preds, per_subset_oof, per_subset_test


# ==============================================================
# ANALYSIS FUNCTIONS
# ==============================================================

def analyze_subset_diversity(per_subset_oof, y_target):
    """Analyze diversity between subset predictions."""
    n_subsets = len(per_subset_oof)
    oof_arr = np.array(per_subset_oof)  # (n_subsets, n_train)

    # Pairwise correlations
    corrs = []
    for i in range(n_subsets):
        for j in range(i+1, n_subsets):
            r = np.corrcoef(oof_arr[i], oof_arr[j])[0, 1]
            corrs.append(r)
    corrs = np.array(corrs)

    # Individual MSEs
    mses = [np.mean((oof_arr[k] - y_target) ** 2) for k in range(n_subsets)]

    # Ensemble MSE at different N
    ensemble_mses = {}
    for n in [1, 3, 5, 10, 15, 20, 25, 30]:
        if n > n_subsets:
            break
        avg = np.mean(oof_arr[:n], axis=0)
        ensemble_mses[n] = np.mean((avg - y_target) ** 2)

    return {
        'pairwise_corr_mean': np.mean(corrs),
        'pairwise_corr_std': np.std(corrs),
        'pairwise_corr_min': np.min(corrs),
        'pairwise_corr_max': np.max(corrs),
        'individual_mse_mean': np.mean(mses),
        'individual_mse_std': np.std(mses),
        'individual_mse_min': np.min(mses),
        'individual_mse_max': np.max(mses),
        'ensemble_mses': ensemble_mses,
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("RANDOM SUBSPACE ENSEMBLE FOR VARIANCE REDUCTION")
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

    # ============================================================
    # PHASE 1: Extract features (with joint angles)
    # ============================================================
    print("\n--- PHASE 1: Feature Extraction ---")
    features_train = {}
    features_test = {}
    for target in TARGETS:
        print(f"  Extracting features for {target} (frame {TARGET_FRAMES[target]})...")
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        features_train[target] = X_train_aug
        features_test[target] = X_test_aug
        print(f"    {target}: {X_train_aug.shape[1]} features")

    # ============================================================
    # PHASE 2: Baseline - full feature locally weighted Ridge
    # ============================================================
    print("\n--- PHASE 2: Baseline (full features) ---")
    baseline_oof = {}
    baseline_test = {}
    for target in TARGETS:
        X_tr = features_train[target]
        X_te = features_test[target]
        y_t = y_scaled[target]
        all_idx = list(range(X_tr.shape[1]))
        oof, test = locally_weighted_subset(
            X_tr, y_t, X_te, pids_train, pids_test,
            all_idx, bandwidth_quantile=0.3, alpha=10.0)
        mse = np.mean((oof - y_t) ** 2)
        baseline_oof[target] = oof
        baseline_test[target] = test
        print(f"  {target}: baseline LOO MSE = {mse:.6f}")

    # ============================================================
    # PHASE 3: Subset size selection (quick test with 10 subsets)
    # ============================================================
    print("\n--- PHASE 3: Subset Size Selection (10 subsets each) ---")
    best_frac_per_target = {}
    for target in TARGETS:
        print(f"\n  {target}:")
        X_tr = features_train[target]
        X_te = features_test[target]
        y_t = y_scaled[target]
        baseline_mse = np.mean((baseline_oof[target] - y_t) ** 2)

        best_frac = 0.70
        best_ens_mse = float('inf')

        for frac in [0.50, 0.60, 0.70, 0.80]:
            oof_avg, _, per_oof, _ = random_subspace_ensemble(
                X_tr, y_t, X_te, pids_train, pids_test,
                n_subsets=10, frac=frac, seed=42, stratified=False,
                bandwidth_quantile=0.3, alpha=10.0)
            ens_mse = np.mean((oof_avg - y_t) ** 2)
            stats = analyze_subset_diversity(per_oof, y_t)
            print(f"    frac={frac:.2f}: ens_MSE={ens_mse:.6f} | "
                  f"indiv_MSE={stats['individual_mse_mean']:.6f} +/- {stats['individual_mse_std']:.6f} | "
                  f"corr={stats['pairwise_corr_mean']:.4f}")
            if ens_mse < best_ens_mse:
                best_ens_mse = ens_mse
                best_frac = frac

        print(f"    BEST frac={best_frac:.2f} (ens_MSE={best_ens_mse:.6f} vs baseline {baseline_mse:.6f})")
        best_frac_per_target[target] = best_frac

    # ============================================================
    # PHASE 4: Stratified vs Fully Random
    # ============================================================
    print("\n--- PHASE 4: Stratified vs Random (10 subsets) ---")
    use_stratified_per_target = {}
    for target in TARGETS:
        frac = best_frac_per_target[target]
        X_tr = features_train[target]
        X_te = features_test[target]
        y_t = y_scaled[target]

        oof_rand, _, _, _ = random_subspace_ensemble(
            X_tr, y_t, X_te, pids_train, pids_test,
            n_subsets=10, frac=frac, seed=42, stratified=False)
        mse_rand = np.mean((oof_rand - y_t) ** 2)

        oof_strat, _, _, _ = random_subspace_ensemble(
            X_tr, y_t, X_te, pids_train, pids_test,
            n_subsets=10, frac=frac, seed=42, stratified=True)
        mse_strat = np.mean((oof_strat - y_t) ** 2)

        use_strat = mse_strat < mse_rand
        use_stratified_per_target[target] = use_strat
        print(f"  {target}: random={mse_rand:.6f}, stratified={mse_strat:.6f} -> {'stratified' if use_strat else 'random'}")

    # ============================================================
    # PHASE 5: Scale up N (optimal subset count)
    # ============================================================
    print("\n--- PHASE 5: Scale Up N (5, 10, 20, 30 subsets) ---")
    best_n_per_target = {}
    final_results = {}
    for target in TARGETS:
        frac = best_frac_per_target[target]
        strat = use_stratified_per_target[target]
        X_tr = features_train[target]
        X_te = features_test[target]
        y_t = y_scaled[target]

        print(f"\n  {target} (frac={frac:.2f}, {'stratified' if strat else 'random'}):")

        best_n = 10
        best_ens_mse = float('inf')

        for n_sub in [5, 10, 20, 30]:
            oof_avg, test_avg, per_oof, per_test = random_subspace_ensemble(
                X_tr, y_t, X_te, pids_train, pids_test,
                n_subsets=n_sub, frac=frac, seed=42, stratified=strat)
            ens_mse = np.mean((oof_avg - y_t) ** 2)
            stats = analyze_subset_diversity(per_oof, y_t)
            print(f"    N={n_sub:2d}: ens_MSE={ens_mse:.6f} | corr={stats['pairwise_corr_mean']:.4f}")
            if ens_mse < best_ens_mse:
                best_ens_mse = ens_mse
                best_n = n_sub
                final_results[target] = {
                    'oof': oof_avg, 'test': test_avg,
                    'per_oof': per_oof, 'per_test': per_test,
                    'stats': stats, 'mse': ens_mse
                }

        best_n_per_target[target] = best_n
        print(f"    BEST N={best_n} (ens_MSE={best_ens_mse:.6f})")

    # ============================================================
    # PHASE 6: Stability analysis (3 master seeds)
    # ============================================================
    print("\n--- PHASE 6: Stability Analysis (3 seeds) ---")
    for target in TARGETS:
        frac = best_frac_per_target[target]
        n_sub = best_n_per_target[target]
        strat = use_stratified_per_target[target]
        X_tr = features_train[target]
        X_te = features_test[target]
        y_t = y_scaled[target]

        seed_oofs = []
        seed_tests = []
        for seed in [42, 123, 999]:
            oof_avg, test_avg, _, _ = random_subspace_ensemble(
                X_tr, y_t, X_te, pids_train, pids_test,
                n_subsets=n_sub, frac=frac, seed=seed, stratified=strat)
            seed_oofs.append(oof_avg)
            seed_tests.append(test_avg)
            mse = np.mean((oof_avg - y_t) ** 2)
            print(f"  {target} seed={seed}: LOO MSE={mse:.6f}")

        # Cross-seed prediction std
        test_std = np.std(seed_tests, axis=0)
        oof_std = np.std(seed_oofs, axis=0)
        print(f"  {target}: test pred std={np.mean(test_std):.6f}, oof pred std={np.mean(oof_std):.6f}")

        # Use average across seeds for final
        final_results[target]['oof'] = np.mean(seed_oofs, axis=0)
        final_results[target]['test'] = np.mean(seed_tests, axis=0)
        final_results[target]['mse'] = np.mean((final_results[target]['oof'] - y_t) ** 2)
        print(f"  {target}: 3-seed avg LOO MSE={final_results[target]['mse']:.6f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    total_mse = 0
    baseline_total = 0
    for target in TARGETS:
        mse = final_results[target]['mse']
        bl = np.mean((baseline_oof[target] - y_scaled[target]) ** 2)
        total_mse += mse
        baseline_total += bl
        pct = (mse - bl) / bl * 100
        print(f"  {target}: ensemble={mse:.6f}, baseline={bl:.6f} ({pct:+.1f}%)")
        print(f"    frac={best_frac_per_target[target]:.2f}, N={best_n_per_target[target]}, "
              f"{'stratified' if use_stratified_per_target[target] else 'random'}")
    print(f"  MEAN: ensemble={total_mse/3:.6f}, baseline={baseline_total/3:.6f} "
          f"({(total_mse - baseline_total) / baseline_total * 100:+.1f}%)")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Load reference submissions
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_2063 = pd.read_csv(SUBMISSION_DIR / "submission_2063.csv")

    # Standalone submission
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': final_results['angle']['test'],
        'scaled_depth': final_results['depth']['test'],
        'scaled_left_right': final_results['left_right']['test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: STANDALONE random subspace ensemble")
    print(f"    angle_std={sub['scaled_angle'].std():.6f}, depth_mean={sub['scaled_depth'].mean():.6f}")

    # Diversity analysis vs references
    print(f"\n  Correlation with Sub 784:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_784[col].values, final_results[target]['test'])[0, 1]
        print(f"    {target}: r={r:.4f}")

    print(f"  Correlation with Sub 2063:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_2063[col].values, final_results[target]['test'])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # Blends with Sub 784 (classic)
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "classic Sub 784 weights"),
        (0.00, 0.20, 0.30, "conservative"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*final_results['angle']['test']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*final_results['depth']['test']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*final_results['left_right']['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: blend with Sub 784 (aw={aw:.2f} dw={dw:.2f} lw={lw:.2f}) {desc}")

    # Blends with Sub 2063 (current best)
    for w in [0.10, 0.20, 0.30, 0.50]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': (1-w)*sub_2063['scaled_angle'].values + w*final_results['angle']['test'],
            'scaled_depth': (1-w)*sub_2063['scaled_depth'].values + w*final_results['depth']['test'],
            'scaled_left_right': (1-w)*sub_2063['scaled_left_right'].values + w*final_results['left_right']['test'],
        })
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(w*100)}% ensemble + {int((1-w)*100)}% Sub 2063")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Return results for reporting
    return {
        'best_frac': best_frac_per_target,
        'best_n': best_n_per_target,
        'use_stratified': use_stratified_per_target,
        'final_results': final_results,
        'baseline_oof': baseline_oof,
    }


if __name__ == "__main__":
    main()
