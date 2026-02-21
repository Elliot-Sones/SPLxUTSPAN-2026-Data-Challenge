"""
Semi-Supervised / Transductive Pipeline

Uses the 113 test shots (features known, labels unknown) to improve predictions.
Builds on the per-example locally weighted Ridge from per_example_pipeline.py.

Approaches:
1. Transductive feature normalization: fit scaler/PLS on all 458 shots
2. Self-training with pseudo-labels: iteratively add confident predictions
3. Label propagation: graph-based label spreading from train to test
4. Graph-regularized Ridge: Laplacian smoothness regularization
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
from sklearn.semi_supervised import LabelSpreading

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
# DATA LOADING (same as per_example_pipeline.py)
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
# FEATURE EXTRACTION (copied from per_example_pipeline.py)
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


# ==============================================================
# APPROACH 1: TRANSDUCTIVE FEATURE NORMALIZATION
# ==============================================================

def transductive_pls(X_train_hc, X_test_hc, y_raw_train, pids_train, pids_test,
                     X_raw_train, X_raw_test):
    """Fit PLS on ALL 458 shots (train + test) instead of just 345 train.
    This gives better feature normalization since PLS sees the full distribution."""
    unique_pids = sorted(np.unique(pids_train))
    max_nc = 15
    pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        n_p = tr_mask.sum()

        # TRANSDUCTIVE: fit scaler on ALL shots (train + test)
        raw_all = np.vstack([X_raw_train[tr_mask], X_raw_test[te_mask]])
        scaler = StandardScaler()
        scaler.fit(raw_all)  # fit on all 458

        raw_tr = scaler.transform(X_raw_train[tr_mask])
        raw_te = scaler.transform(X_raw_test[te_mask])

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        # Find optimal components (CV on train only)
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

    return np.hstack([X_train_hc, pls_train]), np.hstack([X_test_hc, pls_test])


def transductive_locally_weighted(X_train, y_train, X_test, pids_train, pids_test,
                                   bandwidth_quantile=0.5, alpha=10.0):
    """Locally weighted Ridge with TRANSDUCTIVE scaler:
    StandardScaler is fit on all shots (train+test) per player."""
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

        # TRANSDUCTIVE: fit scaler on train + test
        X_all = np.vstack([X_tr, X_te]) if len(X_te) > 0 else X_tr
        scaler = StandardScaler()
        scaler.fit(X_all)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        # Pairwise distances
        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        # OOF: LOO locally weighted
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

        # Test
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
# APPROACH 2: SELF-TRAINING WITH PSEUDO-LABELS
# ==============================================================

def self_training_locally_weighted(X_train, y_train, X_test, pids_train, pids_test,
                                    bandwidth_quantile=0.5, alpha=10.0,
                                    n_iterations=3, confidence_quantile=0.25):
    """Iterative self-training:
    1. Train on labeled data, predict test
    2. Add high-confidence test predictions as pseudo-labels
    3. Retrain on labeled + pseudo-labeled
    4. Re-predict test
    """
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)

    # Initial prediction (baseline)
    oof_preds, test_preds = transductive_locally_weighted(
        X_train, y_train, X_test, pids_train, pids_test,
        bandwidth_quantile=bandwidth_quantile, alpha=alpha)

    print(f"      Iteration 0 (baseline): LOO MSE={np.mean((oof_preds - y_train)**2):.6f}")

    for iteration in range(n_iterations):
        # For each player, select high-confidence test predictions
        X_augmented_list = []
        y_augmented_list = []
        pids_augmented_list = []
        n_pseudo = 0

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            X_tr = X_train[tr_mask]
            y_tr = y_train[tr_mask]
            X_te = X_test[te_mask]
            te_indices = np.where(te_mask)[0]

            if len(X_te) == 0:
                X_augmented_list.append(X_tr)
                y_augmented_list.append(y_tr)
                pids_augmented_list.extend([pid] * len(X_tr))
                continue

            # Get test predictions for this player
            te_preds = test_preds[te_indices]

            # Select high-confidence: predictions close to the player's training distribution
            # Use distance from test predictions to the center of the training target distribution
            y_mean = np.mean(y_tr)
            y_std = np.std(y_tr)
            if y_std < 1e-6:
                y_std = 1.0

            # Confidence: how close the prediction is to training distribution center
            # Lower z-score = higher confidence
            z_scores = np.abs((te_preds - y_mean) / y_std)

            # Alternative: use prediction variance across nearby neighbors
            # Select shots in the top/bottom quantile (most "certain" predictions)
            threshold = np.quantile(z_scores, confidence_quantile)
            confident_mask = z_scores <= threshold

            # Include all training data
            X_augmented_list.append(X_tr)
            y_augmented_list.append(y_tr)
            pids_augmented_list.extend([pid] * len(X_tr))

            # Add pseudo-labeled test shots
            if np.any(confident_mask):
                X_pseudo = X_te[confident_mask]
                y_pseudo = te_preds[confident_mask]
                X_augmented_list.append(X_pseudo)
                y_augmented_list.append(y_pseudo)
                pids_augmented_list.extend([pid] * len(X_pseudo))
                n_pseudo += len(X_pseudo)

        X_aug = np.vstack(X_augmented_list)
        y_aug = np.concatenate(y_augmented_list)
        pids_aug = np.array(pids_augmented_list)

        print(f"      Iteration {iteration+1}: {n_pseudo} pseudo-labeled shots added "
              f"(total {len(X_aug)})")

        # Retrain on augmented data, predict test
        # Use transductive scaler but train on augmented data
        test_preds_new = np.zeros(n_test)
        oof_preds_new = np.zeros(n_train)

        for pid in unique_pids:
            aug_mask = pids_aug == pid
            te_mask = pids_test == pid
            tr_mask = pids_train == pid

            X_a = X_aug[aug_mask]
            y_a = y_aug[aug_mask]
            X_te = X_test[te_mask]
            X_tr_orig = X_train[tr_mask]
            y_tr_orig = y_train[tr_mask]
            n_a = len(X_a)
            tr_indices = np.where(tr_mask)[0]
            te_indices = np.where(te_mask)[0]

            # Transductive scaler
            X_all = np.vstack([X_a, X_te]) if len(X_te) > 0 else X_a
            scaler = StandardScaler()
            scaler.fit(X_all)
            X_a_s = scaler.transform(X_a)
            X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_a.shape[1]))
            X_tr_s = scaler.transform(X_tr_orig)

            # Distances within augmented data
            D_a_a = cdist(X_a_s, X_a_s, metric='euclidean')
            D_te_a = cdist(X_te_s, X_a_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_a))

            all_dists = D_a_a[np.triu_indices(n_a, k=1)]
            if len(all_dists) > 0:
                sigma = np.quantile(all_dists, bandwidth_quantile)
                sigma = max(sigma, 1e-6)
            else:
                sigma = 1.0

            # OOF: predict original training shots using augmented data
            n_tr_orig = len(X_tr_orig)
            D_tr_a = cdist(X_tr_s, X_a_s, metric='euclidean')

            for i in range(n_tr_orig):
                dists = D_tr_a[i, :]
                weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
                # Zero-out self (the first n_tr_orig entries in augmented are original train)
                weights[i] = 0

                if weights.sum() < 1e-10:
                    oof_preds_new[tr_indices[i]] = np.mean(y_a)
                    continue

                ridge = Ridge(alpha=alpha)
                ridge.fit(X_a_s, y_a, sample_weight=weights)
                oof_preds_new[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

            # Test predictions
            for j in range(len(X_te)):
                dists = D_te_a[j, :]
                weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

                if weights.sum() < 1e-10:
                    test_preds_new[te_indices[j]] = np.mean(y_a)
                    continue

                ridge = Ridge(alpha=alpha)
                ridge.fit(X_a_s, y_a, sample_weight=weights)
                test_preds_new[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

        mse_new = np.mean((oof_preds_new - y_train) ** 2)
        print(f"      Iteration {iteration+1}: LOO MSE={mse_new:.6f}")

        # Update predictions for next iteration
        test_preds = test_preds_new
        oof_preds = oof_preds_new

    return oof_preds, test_preds


# ==============================================================
# APPROACH 3: LABEL PROPAGATION / SPREADING
# ==============================================================

def label_spreading_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                kernel='rbf', alpha_ls=0.2, n_neighbors=15):
    """Use sklearn LabelSpreading on the full graph of 458 shots.
    Since LabelSpreading is for classification, we discretize then interpolate,
    OR use the raw graph structure for regression."""
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)
    test_preds = np.zeros(n_test)
    oof_preds = np.zeros(n_train)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        if len(X_te) == 0:
            continue

        # Build combined feature matrix
        X_all = np.vstack([X_tr, X_te])
        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)

        # Custom graph-based regression using label spreading logic:
        # Build affinity matrix
        D = cdist(X_all_s, X_all_s, metric='euclidean')
        all_dists = D[np.triu_indices(len(X_all_s), k=1)]
        sigma = np.quantile(all_dists, 0.5)
        sigma = max(sigma, 1e-6)
        W = np.exp(-D ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)

        # Normalize: D^{-1/2} W D^{-1/2}
        d = W.sum(axis=1)
        d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
        S = W * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

        # Label spreading: iterate Y = alpha * S @ Y + (1-alpha) * Y_init
        n_all = len(X_all_s)
        Y = np.zeros(n_all)
        Y[:n_tr] = y_tr
        Y_init = Y.copy()

        # Clamping mask: labeled samples keep their labels
        labeled_mask = np.zeros(n_all, dtype=bool)
        labeled_mask[:n_tr] = True

        for _ in range(50):
            Y_new = alpha_ls * S @ Y + (1 - alpha_ls) * Y_init
            # Clamp labeled samples
            Y_new[labeled_mask] = y_tr
            if np.max(np.abs(Y_new - Y)) < 1e-8:
                break
            Y = Y_new

        test_preds[te_indices] = Y[n_tr:]

        # OOF: LOO label spreading
        for i in range(n_tr):
            Y_loo = np.zeros(n_all)
            Y_loo[:n_tr] = y_tr
            Y_loo_init = Y_loo.copy()
            Y_loo_init[i] = 0  # Remove this sample's label

            loo_mask = np.zeros(n_all, dtype=bool)
            loo_mask[:n_tr] = True
            loo_mask[i] = False

            for _ in range(50):
                Y_new = alpha_ls * S @ Y_loo + (1 - alpha_ls) * Y_loo_init
                Y_new[loo_mask] = y_tr[loo_mask[:n_tr]]
                if np.max(np.abs(Y_new - Y_loo)) < 1e-8:
                    break
                Y_loo = Y_new

            oof_preds[tr_indices[i]] = Y_loo[i]

    return oof_preds, test_preds


# ==============================================================
# APPROACH 4: GRAPH-REGULARIZED RIDGE
# ==============================================================

def graph_regularized_ridge(X_train, y_train, X_test, pids_train, pids_test,
                             alpha=10.0, beta=1.0, bandwidth_quantile=0.5):
    """Ridge regression with graph Laplacian regularization:
    Solve: (X'X + alpha*I + beta*L)^-1 X'y
    where L is the graph Laplacian over ALL shots (train + test).

    For training, we only use labeled data in the loss term,
    but the graph regularization term encourages smooth predictions
    over the full graph including test shots.
    """
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)
    test_preds = np.zeros(n_test)
    oof_preds = np.zeros(n_train)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)
        n_te = len(X_te)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        if n_te == 0:
            continue

        # Combined features
        X_all = np.vstack([X_tr, X_te])
        n_all = n_tr + n_te
        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)
        X_tr_s = X_all_s[:n_tr]
        X_te_s = X_all_s[n_tr:]

        # Build graph Laplacian over all shots
        D = cdist(X_all_s, X_all_s, metric='euclidean')
        all_dists = D[np.triu_indices(n_all, k=1)]
        sigma = np.quantile(all_dists, bandwidth_quantile)
        sigma = max(sigma, 1e-6)
        W = np.exp(-D ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)
        deg = W.sum(axis=1)
        L = np.diag(deg) - W  # Graph Laplacian

        # We want to predict ALL shots (train + test) jointly.
        # The optimization problem:
        # min_f sum_i_in_train (f_i - y_i)^2 + alpha * ||w||^2 + beta * f' L f
        # where f = X_all @ w
        #
        # This becomes: min_w ||M @ X_all @ w - M @ y_ext||^2 + alpha ||w||^2 + beta w' X_all' L X_all w
        # where M is a mask matrix that selects only labeled samples, y_ext is y padded with 0s
        #
        # Simpler formulation: solve the semi-supervised regression directly.
        # (X_tr' X_tr + alpha I + beta X_all' L X_all) w = X_tr' y_tr

        p = X_all_s.shape[1]
        XtX_train = X_tr_s.T @ X_tr_s  # Only training data in data term
        XtLX_all = X_all_s.T @ L @ X_all_s  # All data in graph term
        Xty_train = X_tr_s.T @ y_tr

        A = XtX_train + alpha * np.eye(p) + beta * XtLX_all
        w = np.linalg.solve(A, Xty_train)

        # Predictions
        test_preds[te_indices] = X_te_s @ w

        # OOF (LOO)
        for i in range(n_tr):
            loo_mask = np.ones(n_tr, dtype=bool)
            loo_mask[i] = False
            X_loo = X_tr_s[loo_mask]
            y_loo = y_tr[loo_mask]

            XtX_loo = X_loo.T @ X_loo
            Xty_loo = X_loo.T @ y_loo

            A_loo = XtX_loo + alpha * np.eye(p) + beta * XtLX_all
            w_loo = np.linalg.solve(A_loo, Xty_loo)
            oof_preds[tr_indices[i]] = X_tr_s[i] @ w_loo

    return oof_preds, test_preds


# ==============================================================
# ORIGINAL PIPELINE (for comparison)
# ==============================================================

def original_locally_weighted(X_train, y_train, X_test, pids_train, pids_test,
                               bandwidth_quantile=0.5, alpha=10.0):
    """Original per_example_pipeline.py approach - train-only scaler."""
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

        # ORIGINAL: fit scaler on training only
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


def original_pls(X_train_hc, X_test_hc, y_raw_train, pids_train, pids_test,
                 X_raw_train, X_raw_test):
    """Original PLS: fit only on training data."""
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

    return np.hstack([X_train_hc, pls_train]), np.hstack([X_test_hc, pls_test])


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("SEMI-SUPERVISED / TRANSDUCTIVE PIPELINE")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    # Load scalers
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # Load reference submissions
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    # ============================================================
    # PER-TARGET PIPELINE
    # ============================================================

    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)
        print(f"  HC features: {X_train_hc.shape[1]}")

        y_raw = y_train[:, target_idx[target]]
        y_target = y_scaled[target]

        # --- Build feature sets ---

        # Original PLS (train-only)
        print("  Building feature sets...")
        X_train_orig, X_test_orig = original_pls(
            X_train_hc, X_test_hc, y_raw, pids_train, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"    Original features: {X_train_orig.shape[1]}")

        # Transductive PLS (all 458 shots)
        X_train_trans, X_test_trans = transductive_pls(
            X_train_hc, X_test_hc, y_raw, pids_train, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"    Transductive features: {X_train_trans.shape[1]}")

        results = {}

        # --- BASELINE: Original locally weighted (for comparison) ---
        print("\n  [0] BASELINE: Original locally weighted Ridge...")
        oof_orig, test_orig = original_locally_weighted(
            X_train_orig, y_target, X_test_orig, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_orig = np.mean((oof_orig - y_target) ** 2)
        print(f"      LOO MSE: {mse_orig:.6f}")
        results['baseline'] = (mse_orig, oof_orig, test_orig)

        # --- APPROACH 1: Transductive locally weighted Ridge ---
        print("\n  [1] TRANSDUCTIVE: locally weighted Ridge with transductive scaler...")
        oof_trans, test_trans = transductive_locally_weighted(
            X_train_trans, y_target, X_test_trans, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_trans = np.mean((oof_trans - y_target) ** 2)
        print(f"      LOO MSE: {mse_trans:.6f} (delta: {mse_trans - mse_orig:+.6f}, "
              f"{(mse_trans - mse_orig) / mse_orig * 100:+.2f}%)")
        results['transductive'] = (mse_trans, oof_trans, test_trans)

        # Also try with transductive PLS but original scaler in the Ridge
        print("\n  [1b] TRANSDUCTIVE PLS ONLY: original Ridge scaler, transductive PLS...")
        oof_trans_pls, test_trans_pls = original_locally_weighted(
            X_train_trans, y_target, X_test_trans, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_trans_pls = np.mean((oof_trans_pls - y_target) ** 2)
        print(f"      LOO MSE: {mse_trans_pls:.6f} (delta: {mse_trans_pls - mse_orig:+.6f}, "
              f"{(mse_trans_pls - mse_orig) / mse_orig * 100:+.2f}%)")
        results['transductive_pls_only'] = (mse_trans_pls, oof_trans_pls, test_trans_pls)

        # --- APPROACH 2: Self-training ---
        print("\n  [2] SELF-TRAINING: iterative pseudo-labels...")
        oof_st, test_st = self_training_locally_weighted(
            X_train_trans, y_target, X_test_trans, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0,
            n_iterations=3, confidence_quantile=0.25)
        mse_st = np.mean((oof_st - y_target) ** 2)
        print(f"      Final LOO MSE: {mse_st:.6f} (delta: {mse_st - mse_orig:+.6f}, "
              f"{(mse_st - mse_orig) / mse_orig * 100:+.2f}%)")
        results['self_training'] = (mse_st, oof_st, test_st)

        # --- APPROACH 3: Label spreading ---
        print("\n  [3] LABEL SPREADING: graph-based propagation...")
        for alpha_ls in [0.1, 0.2, 0.5]:
            oof_ls, test_ls = label_spreading_prediction(
                X_train_trans, y_target, X_test_trans, pids_train, pids_test,
                alpha_ls=alpha_ls)
            mse_ls = np.mean((oof_ls - y_target) ** 2)
            print(f"      alpha={alpha_ls}: LOO MSE={mse_ls:.6f} (delta: {mse_ls - mse_orig:+.6f})")
            key = f'label_spreading_a{alpha_ls}'
            results[key] = (mse_ls, oof_ls, test_ls)

        # --- APPROACH 4: Graph-regularized Ridge ---
        print("\n  [4] GRAPH-REGULARIZED RIDGE: Laplacian smoothness...")
        for beta in [0.1, 0.5, 1.0, 5.0]:
            oof_gr, test_gr = graph_regularized_ridge(
                X_train_trans, y_target, X_test_trans, pids_train, pids_test,
                alpha=10.0, beta=beta, bandwidth_quantile=0.5)
            mse_gr = np.mean((oof_gr - y_target) ** 2)
            print(f"      beta={beta}: LOO MSE={mse_gr:.6f} (delta: {mse_gr - mse_orig:+.6f})")
            key = f'graph_ridge_b{beta}'
            results[key] = (mse_gr, oof_gr, test_gr)

        # --- SUMMARY ---
        print(f"\n  {target} SUMMARY:")
        for name, (mse, _, _) in sorted(results.items(), key=lambda x: x[1][0]):
            delta = (mse - mse_orig) / mse_orig * 100
            print(f"    {name:30s}: MSE={mse:.6f} ({delta:+.2f}%)")

        # Pick best approach
        best_name = min(results, key=lambda x: results[x][0])
        best_mse, best_oof, best_test = results[best_name]
        print(f"    BEST: {best_name} (MSE={best_mse:.6f})")

        all_results[target] = {
            'results': results,
            'best_name': best_name,
            'best_test': best_test,
            'best_oof': best_oof,
            'best_mse': best_mse,
        }

    # ============================================================
    # OVERALL CV RESULTS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("OVERALL CV RESULTS")
    print(f"{'=' * 70}")

    total_mse_orig = 0
    total_mse_best = 0
    for target in TARGETS:
        mse_orig = all_results[target]['results']['baseline'][0]
        mse_best = all_results[target]['best_mse']
        total_mse_orig += mse_orig
        total_mse_best += mse_best
        delta = (mse_best - mse_orig) / mse_orig * 100
        print(f"  {target}: baseline={mse_orig:.6f}, best={mse_best:.6f} "
              f"({all_results[target]['best_name']}, {delta:+.2f}%)")
    print(f"  MEAN baseline: {total_mse_orig / 3:.6f}")
    print(f"  MEAN best:     {total_mse_best / 3:.6f}")

    # ============================================================
    # CORRELATION WITH REFERENCE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("CORRELATION WITH REFERENCES")
    print(f"{'=' * 70}")

    for target in TARGETS:
        col = f'scaled_{target}'
        r_784 = np.corrcoef(sub_784[col].values, all_results[target]['best_test'])[0, 1]
        r_1350 = np.corrcoef(sub_1350[col].values, all_results[target]['best_test'])[0, 1]
        print(f"  {target}: r(Sub784)={r_784:.4f}, r(Sub1350)={r_1350:.4f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # 1. Standalone: best approach per target
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': all_results['angle']['best_test'],
        'scaled_depth': all_results['depth']['best_test'],
        'scaled_left_right': all_results['left_right']['best_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: STANDALONE best-per-target")

    # 2. Blend with Sub 784 at various weights
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "optimal"),
        (0.00, 0.20, 0.30, "conservative"),
        (0.00, 0.40, 0.60, "aggressive"),
        (0.10, 0.30, 0.50, "with angle"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*all_results['angle']['best_test']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*all_results['depth']['best_test']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*all_results['left_right']['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: blend aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # 3. Blend with Sub 1350 (to see if transductive improves on top of per-example)
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_1350[col] + w*all_results[target]['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: blend with Sub 1350, w={w:.2f}")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # 4. Per-approach standalone submissions for key methods
    for approach_name in ['transductive', 'transductive_pls_only', 'self_training']:
        if approach_name in all_results['angle']['results']:
            sub_num = get_next_submission_number()
            sub = pd.DataFrame({
                'id': test_data['ids'],
                'scaled_angle': all_results['angle']['results'][approach_name][2],
                'scaled_depth': all_results['depth']['results'][approach_name][2],
                'scaled_left_right': all_results['left_right']['results'][approach_name][2],
            })
            sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

            mse_a = all_results['angle']['results'][approach_name][0]
            mse_d = all_results['depth']['results'][approach_name][0]
            mse_l = all_results['left_right']['results'][approach_name][0]
            print(f"  Sub {sub_num}: {approach_name} standalone")
            print(f"    CV: angle={mse_a:.6f}, depth={mse_d:.6f}, lr={mse_l:.6f}, "
                  f"mean={(mse_a+mse_d+mse_l)/3:.6f}")

            # Also blend these with Sub 784
            sub_num2 = get_next_submission_number()
            blended = sub_784.copy()
            blended['scaled_angle'] = sub_784['scaled_angle']
            blended['scaled_depth'] = 0.70*sub_784['scaled_depth'] + 0.30*all_results['depth']['results'][approach_name][2]
            blended['scaled_left_right'] = 0.50*sub_784['scaled_left_right'] + 0.50*all_results['left_right']['results'][approach_name][2]
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num2}.csv", index=False)
            print(f"  Sub {sub_num2}: {approach_name} blend (aw=0 dw=0.30 lw=0.50)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
