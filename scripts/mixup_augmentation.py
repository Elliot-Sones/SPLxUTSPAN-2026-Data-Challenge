"""
Mixup Augmentation Pipeline for Per-Example Locally Weighted Ridge

Generates synthetic training samples by blending same-player shots via mixup:
  - Pick two shots from same player
  - Blend ratio lambda ~ Beta(alpha, alpha)
  - Synthetic 3D keypoints = lambda * shot_A + (1-lambda) * shot_B (frame by frame)
  - Synthetic targets = lambda * target_A + (1-lambda) * target_B

Then retrain per-example locally weighted Ridge on augmented data.

CRITICAL: During LOO CV, leave out the original shot AND any synthetic shots
derived from it to prevent data leakage.
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from itertools import combinations
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
    """Load and parse all data."""
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
# MIXUP AUGMENTATION
# ==============================================================

def generate_mixup_samples(X_3d, X_raw, y, pids, alpha=0.5, n_multiplier=2, seed=42):
    """Generate synthetic samples via mixup within each player.

    Args:
        X_3d: (n, 240, n_kp, 3) - 3D keypoint trajectories
        X_raw: (n, n_features) - raw flat features
        y: (n, 3) - targets (angle, depth, left_right)
        pids: (n,) - player IDs
        alpha: Beta distribution parameter. Lower = more extreme blends.
        n_multiplier: How many synthetic samples per original sample.
        seed: Random seed.

    Returns:
        aug_X_3d, aug_X_raw, aug_y, aug_pids, aug_parent_indices
        parent_indices[i] = set of original indices that contributed to synthetic sample i
    """
    rng = np.random.RandomState(seed)
    unique_pids = sorted(np.unique(pids))

    syn_X_3d_list = []
    syn_X_raw_list = []
    syn_y_list = []
    syn_pids_list = []
    syn_parents_list = []  # Track which originals contributed

    for pid in unique_pids:
        mask = pids == pid
        indices = np.where(mask)[0]
        n_player = len(indices)

        # Number of synthetic samples to generate for this player
        n_synthetic = n_player * n_multiplier

        for _ in range(n_synthetic):
            # Pick two different shots from same player
            i, j = rng.choice(n_player, size=2, replace=False)
            idx_a, idx_b = indices[i], indices[j]

            # Sample blend ratio from Beta distribution
            lam = rng.beta(alpha, alpha)

            # Mixup 3D keypoints frame-by-frame
            syn_3d = lam * X_3d[idx_a] + (1 - lam) * X_3d[idx_b]
            syn_raw = lam * X_raw[idx_a] + (1 - lam) * X_raw[idx_b]
            syn_targets = lam * y[idx_a] + (1 - lam) * y[idx_b]

            syn_X_3d_list.append(syn_3d)
            syn_X_raw_list.append(syn_raw)
            syn_y_list.append(syn_targets)
            syn_pids_list.append(pid)
            syn_parents_list.append((idx_a, idx_b))

    if len(syn_X_3d_list) == 0:
        return (np.zeros((0, *X_3d.shape[1:]), dtype=np.float32),
                np.zeros((0, X_raw.shape[1]), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.array([], dtype=pids.dtype),
                [])

    syn_X_3d = np.array(syn_X_3d_list, dtype=np.float32)
    syn_X_raw = np.array(syn_X_raw_list, dtype=np.float32)
    syn_y = np.array(syn_y_list, dtype=np.float32)
    syn_pids = np.array(syn_pids_list)

    return syn_X_3d, syn_X_raw, syn_y, syn_pids, syn_parents_list


# ==============================================================
# FEATURE EXTRACTION (from per_example_pipeline.py)
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


def extract_all_features(X_3d, pids, kp_index, kp_names, target):
    """Extract features for all shots from raw 3D data."""
    frame = TARGET_FRAMES[target]
    n = len(pids)

    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = X_3d[i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


# ==============================================================
# PLS AUGMENTATION
# ==============================================================

def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test,
                     X_raw_train, X_raw_test):
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


def augment_with_pls_mixed(X_orig, y_raw_orig, pids_orig, X_syn, pids_syn,
                           X_test, pids_test,
                           X_raw_orig, X_raw_syn, X_raw_test):
    """PLS fit on original data only, then transform all (original + synthetic + test).

    We fit PLS on original data to avoid leakage from synthetic samples.
    """
    unique_pids = sorted(np.unique(pids_orig))
    max_nc = 15
    pls_orig = np.zeros((len(pids_orig), max_nc), dtype=np.float32)
    pls_syn = np.zeros((len(pids_syn), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        orig_mask = pids_orig == pid
        syn_mask = pids_syn == pid
        te_mask = pids_test == pid
        n_p = orig_mask.sum()

        scaler = StandardScaler()
        raw_orig = scaler.fit_transform(X_raw_orig[orig_mask])
        raw_syn = scaler.transform(X_raw_syn[syn_mask]) if syn_mask.sum() > 0 else np.zeros((0, raw_orig.shape[1]))
        raw_te = scaler.transform(X_raw_test[te_mask])

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15]:
            if c > nc:
                break
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(raw_orig):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_orig[tr_idx], y_raw_orig[orig_mask][tr_idx])
                pred = pls.predict(raw_orig[val_idx]).flatten()
                mses.append(np.mean((pred - y_raw_orig[orig_mask][val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_orig, y_raw_orig[orig_mask])
        pls_orig[orig_mask, :best_nc] = pls.transform(raw_orig)
        if syn_mask.sum() > 0:
            pls_syn[syn_mask, :best_nc] = pls.transform(raw_syn)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)

    X_orig_aug = np.hstack([X_orig, pls_orig])
    X_syn_aug = np.hstack([X_syn, pls_syn])
    X_test_aug = np.hstack([X_test, pls_test])
    return X_orig_aug, X_syn_aug, X_test_aug


# ==============================================================
# LOCALLY WEIGHTED REGRESSION WITH MIXUP AUGMENTATION
# ==============================================================

def locally_weighted_prediction_augmented(
        X_orig, y_orig, pids_orig,
        X_syn, y_syn, pids_syn, syn_parents,
        X_test, pids_test,
        bandwidth_quantile=0.5, alpha=10.0, syn_weight_discount=1.0):
    """Per-example locally weighted Ridge using original + synthetic training data.

    During LOO CV on original samples:
      - Leave out sample i AND any synthetic sample derived from i
      - This prevents data leakage

    For test prediction:
      - Use all original + synthetic samples

    Args:
        syn_weight_discount: Multiplicative discount on synthetic sample weights.
            1.0 = treat same as original. 0.5 = half weight for synthetic.
    """
    unique_pids = sorted(np.unique(pids_orig))
    n_orig = len(X_orig)
    n_syn = len(X_syn)
    n_test = len(X_test)

    # Combined training data
    X_all = np.vstack([X_orig, X_syn])
    y_all = np.concatenate([y_orig, y_syn])
    pids_all = np.concatenate([pids_orig, pids_syn])
    is_synthetic = np.concatenate([np.zeros(n_orig, dtype=bool),
                                   np.ones(n_syn, dtype=bool)])

    # Build parent mapping: for each synthetic sample, which original indices it came from
    parent_of = {}  # syn_index (in combined array) -> set of original indices
    for s_i, (pa, pb) in enumerate(syn_parents):
        parent_of[n_orig + s_i] = {pa, pb}

    oof_preds = np.zeros(n_orig)
    test_preds = np.zeros(n_test)

    for pid in unique_pids:
        # Masks in combined space
        all_mask = pids_all == pid
        orig_mask = (pids_orig == pid)
        te_mask = (pids_test == pid)

        X_all_p = X_all[all_mask]
        y_all_p = y_all[all_mask]
        is_syn_p = is_synthetic[all_mask]
        all_indices = np.where(all_mask)[0]  # indices into combined array
        orig_indices = np.where(orig_mask)[0]  # indices into X_orig
        te_indices = np.where(te_mask)[0]

        n_all_p = len(X_all_p)

        # Standardize within player (fit on original only for stability)
        orig_in_player = ~is_syn_p  # boolean mask within player subset
        scaler = StandardScaler()
        scaler.fit(X_all_p[orig_in_player])
        X_all_p_s = scaler.transform(X_all_p)
        X_te_p_s = scaler.transform(X_test[te_mask]) if te_mask.sum() > 0 else np.zeros((0, X_all_p.shape[1]))

        # Pairwise distances within player
        D_all = cdist(X_all_p_s, X_all_p_s, metric='euclidean')
        D_te = cdist(X_te_p_s, X_all_p_s, metric='euclidean') if te_mask.sum() > 0 else np.zeros((0, n_all_p))

        # Bandwidth from original-only distances
        orig_local_idx = np.where(orig_in_player)[0]
        D_orig_orig = D_all[np.ix_(orig_local_idx, orig_local_idx)]
        n_orig_p = len(orig_local_idx)
        all_dists = D_orig_orig[np.triu_indices(n_orig_p, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        # LOO on original samples
        for i_local, orig_idx in enumerate(orig_local_idx):
            # Which combined indices to exclude? Self + any synthetic derived from this original
            orig_global = all_indices[orig_idx]
            exclude_local = {orig_idx}

            # Find synthetic samples in this player derived from this original
            for j_local in range(n_all_p):
                j_global = all_indices[j_local]
                if j_global in parent_of:
                    if orig_global in parent_of[j_global]:
                        exclude_local.add(j_local)

            # Weights
            dists = D_all[orig_idx, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            # Zero out excluded samples
            for ex in exclude_local:
                weights[ex] = 0

            # Discount synthetic sample weights
            if syn_weight_discount < 1.0:
                syn_local_mask = is_syn_p.copy()
                weights[syn_local_mask] *= syn_weight_discount

            if weights.sum() < 1e-10:
                oof_preds[orig_indices[i_local]] = np.mean(y_all_p[orig_in_player])
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_all_p_s, y_all_p, sample_weight=weights)
            oof_preds[orig_indices[i_local]] = ridge.predict(X_all_p_s[orig_idx:orig_idx+1])[0]

        # Test predictions: use all training data
        for j in range(te_mask.sum()):
            dists = D_te[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            # Discount synthetic
            if syn_weight_discount < 1.0:
                weights[is_syn_p] *= syn_weight_discount

            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_all_p[orig_in_player])
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_all_p_s, y_all_p, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_p_s[j:j+1])[0]

    return oof_preds, test_preds


# ==============================================================
# BASELINE: locally weighted prediction (no augmentation)
# ==============================================================

def locally_weighted_prediction_baseline(X_train, y_train, X_test, pids_train, pids_test,
                                         bandwidth_quantile=0.5, alpha=10.0):
    """Baseline per-example locally weighted Ridge (no augmentation)."""
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
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

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


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("MIXUP AUGMENTATION PIPELINE")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']
    kp_names = train_data['kp_names']

    # Load scalers
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # Load Sub 784 for blending
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # ==============================================================
    # CONFIG SEARCH
    # ==============================================================

    configs = [
        # (alpha, n_multiplier, syn_weight_discount, description)
        (0.5,  2, 1.0, "alpha=0.5, 2x, full weight"),
        (0.2,  2, 1.0, "alpha=0.2, 2x, full weight"),
        (1.0,  2, 1.0, "alpha=1.0, 2x, full weight"),
        (0.5,  1, 1.0, "alpha=0.5, 1x, full weight"),
        (0.5,  3, 1.0, "alpha=0.5, 3x, full weight"),
        (0.5,  2, 0.5, "alpha=0.5, 2x, half weight"),
        (0.2,  3, 0.5, "alpha=0.2, 3x, half weight"),
    ]

    # First get baseline (no augmentation) - bandwidth=0.5, alpha=10
    print("\n" + "=" * 70)
    print("BASELINE (no augmentation)")
    print("=" * 70)

    baseline_results = {}
    for target in TARGETS:
        print(f"\n  Extracting features for {target} (frame {TARGET_FRAMES[target]})...")
        X_train_hc, _ = extract_all_features(
            train_data['X_3d'], pids_train, kp_index, kp_names, target)
        X_test_hc, _ = extract_all_features(
            test_data['X_3d'], pids_test, kp_index, kp_names, target)

        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        y_target = y_scaled[target]

        print(f"  Running baseline locally weighted Ridge...")
        oof_base, test_base = locally_weighted_prediction_baseline(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"  {target} baseline LOO MSE: {mse_base:.6f}")

        baseline_results[target] = {
            'oof': oof_base,
            'test': test_base,
            'mse': mse_base,
            'X_train_hc': X_train_hc,
            'X_test_hc': X_test_hc,
            'X_train_aug': X_train_aug,
            'X_test_aug': X_test_aug,
        }

    base_mean = np.mean([baseline_results[t]['mse'] for t in TARGETS])
    print(f"\n  BASELINE MEAN LOO MSE: {base_mean:.6f}")

    # ==============================================================
    # MIXUP CONFIGS
    # ==============================================================

    all_config_results = []

    for cfg_i, (beta_alpha, n_mult, syn_discount, desc) in enumerate(configs):
        print(f"\n{'=' * 70}")
        print(f"CONFIG {cfg_i+1}/{len(configs)}: {desc}")
        print(f"  Beta alpha={beta_alpha}, n_multiplier={n_mult}, syn_weight_discount={syn_discount}")
        print(f"{'=' * 70}")

        # Generate mixup samples
        print("  Generating mixup samples...")
        syn_X_3d, syn_X_raw, syn_y, syn_pids, syn_parents = generate_mixup_samples(
            train_data['X_3d'], train_data['X_raw'], y_train, pids_train,
            alpha=beta_alpha, n_multiplier=n_mult, seed=42 + cfg_i)

        n_syn = len(syn_pids)
        print(f"  Generated {n_syn} synthetic samples ({n_syn/345:.1f}x original)")

        cfg_results = {}
        for target in TARGETS:
            print(f"\n  TARGET: {target}")

            # Extract features from synthetic shots
            print(f"    Extracting features from {n_syn} synthetic shots...")
            X_syn_hc, _ = extract_all_features(
                syn_X_3d, syn_pids, kp_index, kp_names, target)

            # PLS: fit on original, transform all
            y_raw = y_train[:, target_idx[target]]
            y_raw_syn = syn_y[:, target_idx[target]]

            X_orig_aug, X_syn_aug, X_test_aug = augment_with_pls_mixed(
                baseline_results[target]['X_train_hc'], y_raw, pids_train,
                X_syn_hc, syn_pids,
                baseline_results[target]['X_test_hc'], pids_test,
                train_data['X_raw'], syn_X_raw, test_data['X_raw'])

            y_target = y_scaled[target]
            y_syn_scaled = scalers[target].transform(
                syn_y[:, target_idx[target]].reshape(-1, 1)).ravel()

            # Run augmented locally weighted Ridge
            print(f"    Running augmented locally weighted Ridge...")
            oof_aug, test_aug = locally_weighted_prediction_augmented(
                X_orig_aug, y_target, pids_train,
                X_syn_aug, y_syn_scaled, syn_pids, syn_parents,
                X_test_aug, pids_test,
                bandwidth_quantile=0.5, alpha=10.0,
                syn_weight_discount=syn_discount)

            mse_aug = np.mean((oof_aug - y_target) ** 2)
            mse_base = baseline_results[target]['mse']
            delta_pct = (mse_aug - mse_base) / mse_base * 100

            print(f"    {target}: LOO MSE={mse_aug:.6f} (baseline={mse_base:.6f}, delta={delta_pct:+.2f}%)")

            # Correlation with baseline test predictions
            r_base = np.corrcoef(baseline_results[target]['test'], test_aug)[0, 1]
            # Correlation with Sub 784
            col = f'scaled_{target}'
            r_784 = np.corrcoef(sub_784[col].values, test_aug)[0, 1]
            print(f"    Correlation with baseline: r={r_base:.4f}")
            print(f"    Correlation with Sub 784: r={r_784:.4f}")

            cfg_results[target] = {
                'oof': oof_aug,
                'test': test_aug,
                'mse': mse_aug,
                'delta_pct': delta_pct,
                'r_baseline': r_base,
                'r_784': r_784,
            }

        # Summary for this config
        mean_mse = np.mean([cfg_results[t]['mse'] for t in TARGETS])
        mean_delta = np.mean([cfg_results[t]['delta_pct'] for t in TARGETS])
        print(f"\n  CONFIG SUMMARY: mean LOO MSE={mean_mse:.6f} (delta={mean_delta:+.2f}%)")

        all_config_results.append({
            'desc': desc,
            'beta_alpha': beta_alpha,
            'n_mult': n_mult,
            'syn_discount': syn_discount,
            'results': cfg_results,
            'mean_mse': mean_mse,
            'mean_delta': mean_delta,
        })

    # ==============================================================
    # RANK CONFIGS AND GENERATE SUBMISSIONS
    # ==============================================================

    print(f"\n{'=' * 70}")
    print("CONFIG RANKING")
    print(f"{'=' * 70}")

    sorted_configs = sorted(all_config_results, key=lambda x: x['mean_mse'])
    for i, cfg in enumerate(sorted_configs):
        print(f"  {i+1}. {cfg['desc']}: mean MSE={cfg['mean_mse']:.6f} ({cfg['mean_delta']:+.2f}%)")
        for t in TARGETS:
            r = cfg['results'][t]
            print(f"     {t}: MSE={r['mse']:.6f} ({r['delta_pct']:+.2f}%), r_base={r['r_baseline']:.4f}, r_784={r['r_784']:.4f}")

    # Generate submissions for top 3 configs
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # First: baseline submissions (re-generate fresh)
    sub_num = get_next_submission_number()
    sub_baseline = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': baseline_results['angle']['test'],
        'scaled_depth': baseline_results['depth']['test'],
        'scaled_left_right': baseline_results['left_right']['test'],
    })
    sub_baseline.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: BASELINE (no augmentation) standalone")

    # Baseline blended with Sub 784 (dw=0.30, lw=0.50)
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * baseline_results['depth']['test']
    blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * baseline_results['left_right']['test']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: BASELINE blended with Sub 784 (dw=0.30, lw=0.50)")

    # Top configs
    for i, cfg in enumerate(sorted_configs[:3]):
        print(f"\n  --- Config: {cfg['desc']} ---")

        # Standalone
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': cfg['results']['angle']['test'],
            'scaled_depth': cfg['results']['depth']['test'],
            'scaled_left_right': cfg['results']['left_right']['test'],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {cfg['desc']} standalone")

        # Blended with Sub 784 at dw=0.30, lw=0.50
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * cfg['results']['depth']['test']
        blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * cfg['results']['left_right']['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: {cfg['desc']} blended (dw=0.30, lw=0.50)")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

        # Also blended at lower weights
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_depth'] = 0.80 * sub_784['scaled_depth'] + 0.20 * cfg['results']['depth']['test']
        blended['scaled_left_right'] = 0.65 * sub_784['scaled_left_right'] + 0.35 * cfg['results']['left_right']['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {cfg['desc']} conservative blend (dw=0.20, lw=0.35)")

    # Also: blend best mixup with Sub 1350 (our current LB best)
    print(f"\n  --- Blending best mixup with Sub 1350 ---")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    best_cfg = sorted_configs[0]
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_1350[col] + w * best_cfg['results'][target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        r_1350 = np.mean([np.corrcoef(sub_1350[f'scaled_{t}'].values, best_cfg['results'][t]['test'])[0,1] for t in TARGETS])
        print(f"  Sub {sub_num}: {w*100:.0f}% best mixup + {(1-w)*100:.0f}% Sub 1350 (mean r={r_1350:.4f})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # ==============================================================
    # SUMMARY TABLE
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Baseline mean LOO MSE: {base_mean:.6f}")
    print(f"  Best mixup mean LOO MSE: {sorted_configs[0]['mean_mse']:.6f} ({sorted_configs[0]['desc']})")
    best_delta = sorted_configs[0]['mean_delta']
    print(f"  Improvement: {best_delta:+.2f}%")
    print()
    print("  Per-target breakdown (best config):")
    for t in TARGETS:
        r = sorted_configs[0]['results'][t]
        b = baseline_results[t]['mse']
        print(f"    {t}: {r['mse']:.6f} vs {b:.6f} ({r['delta_pct']:+.2f}%)")

    return sorted_configs, baseline_results


if __name__ == "__main__":
    sorted_configs, baseline_results = main()
