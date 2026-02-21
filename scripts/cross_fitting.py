"""
K-Fold Cross-Fitting for Variance Reduction

Instead of using ALL training data for each per-example Ridge prediction,
split training data into K folds. For each fold, train per-example Ridge
on the other (K-1) folds. Average K predictions.

Benefits:
1. Variance reduction: each sub-model sees different data, errors are somewhat independent
2. Reduced overfitting to specific samples: no single noisy sample in all K models
3. Bias-variance tradeoff: higher bias per sub-model, but averaging reduces variance

Phases:
1. Baseline (standard per-example Ridge on ALL data)
2. K-fold cross-fitting (K=3,5,7,10)
3. Weighted averaging (inverse-MSE weighting)
4. Bootstrap bagging (80% samples, B=10,20)
5. Stability analysis (K=5 with 5 seeds)
6. Effective sample size analysis
7. Diversity analysis (correlation with Sub 2063)
8. Submission generation
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
# CORE: Per-example locally weighted Ridge (reusable)
# ==============================================================

def per_example_ridge_predict(X_tr_s, y_tr, X_query_s, sigma, alpha=10.0):
    """Predict for query points using locally weighted Ridge on given training data.

    Args:
        X_tr_s: scaled training features (n_tr, d)
        y_tr: training targets (n_tr,)
        X_query_s: scaled query features (n_q, d)
        sigma: kernel bandwidth
        alpha: ridge regularization

    Returns:
        predictions: (n_q,) array
    """
    D = cdist(X_query_s, X_tr_s, metric='euclidean')
    preds = np.zeros(len(X_query_s))

    for j in range(len(X_query_s)):
        weights = np.exp(-D[j] ** 2 / (2 * sigma ** 2))
        if weights.sum() < 1e-10:
            preds[j] = np.mean(y_tr)
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_s, y_tr, sample_weight=weights)
        preds[j] = ridge.predict(X_query_s[j:j+1])[0]

    return preds


def compute_sigma_for_player(X_tr_s, bandwidth_quantile=0.3):
    """Compute adaptive bandwidth from pairwise distances."""
    n = len(X_tr_s)
    D = cdist(X_tr_s, X_tr_s, metric='euclidean')
    all_dists = D[np.triu_indices(n, k=1)]
    if len(all_dists) > 0:
        sigma = np.quantile(all_dists, bandwidth_quantile)
        return max(sigma, 1e-6)
    return 1.0


# ==============================================================
# PHASE 1: Baseline (standard per-example Ridge, ALL data)
# ==============================================================

def baseline_prediction(X_train, y_train, X_test, pids_train, pids_test,
                        bandwidth_quantile=0.3, alpha=10.0):
    """Standard per-example Ridge using ALL training data per player."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        sigma = compute_sigma_for_player(X_tr_s, bandwidth_quantile)

        # LOO for OOF
        D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        for i in range(len(X_tr)):
            weights = np.exp(-D_tr[i] ** 2 / (2 * sigma ** 2))
            weights[i] = 0  # Leave out self
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test predictions
        if len(X_te) > 0:
            test_preds[te_indices] = per_example_ridge_predict(
                X_tr_s, y_tr, X_te_s, sigma, alpha)

    return oof_preds, test_preds


# ==============================================================
# PHASE 2: K-Fold Cross-Fitting
# ==============================================================

def cross_fitting_prediction(X_train, y_train, X_test, pids_train, pids_test,
                             K=5, bandwidth_quantile=0.3, alpha=10.0, seed=42):
    """K-fold cross-fitting: split training data per-player into K folds,
    train on (K-1) folds, predict test, average K predictions.

    Returns test predictions, honest LOO predictions, and per-fold predictions.
    """
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)

    # Store per-fold test predictions
    fold_test_preds = np.zeros((K, n_test))
    # For honest LOO: each train sample appears in exactly one held-out fold
    oof_preds = np.zeros(n_train)
    # Per-fold LOO MSE for weighted averaging later
    fold_loo_mses = np.zeros(K)
    fold_loo_counts = np.zeros(K)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]
        n_player = len(X_tr)

        # Standardize within player (using ALL player data for consistent scaling)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        # Compute sigma from ALL player data (consistent across folds)
        sigma = compute_sigma_for_player(X_tr_s, bandwidth_quantile)

        # Split player's training data into K folds
        actual_K = min(K, n_player)
        kf = KFold(n_splits=actual_K, shuffle=True, random_state=seed)

        for fold_idx, (train_fold_idx, held_out_idx) in enumerate(kf.split(X_tr_s)):
            X_fold_tr = X_tr_s[train_fold_idx]
            y_fold_tr = y_tr[train_fold_idx]
            X_fold_ho = X_tr_s[held_out_idx]
            y_fold_ho = y_tr[held_out_idx]

            # Predict test using this fold's training data
            if len(X_te) > 0:
                fold_test_preds[fold_idx, te_indices] = per_example_ridge_predict(
                    X_fold_tr, y_fold_tr, X_te_s, sigma, alpha)

            # Honest LOO for held-out samples:
            # For each held-out sample, it's already excluded from fold training.
            # Just predict it directly using the fold model.
            ho_preds = per_example_ridge_predict(
                X_fold_tr, y_fold_tr, X_fold_ho, sigma, alpha)

            for local_i, global_i in enumerate(held_out_idx):
                oof_preds[tr_indices[global_i]] = ho_preds[local_i]

            # Track per-fold LOO MSE
            fold_mse = np.mean((ho_preds - y_fold_ho) ** 2)
            fold_loo_mses[fold_idx] += fold_mse * len(held_out_idx)
            fold_loo_counts[fold_idx] += len(held_out_idx)

    # Average test predictions across K folds
    test_preds = np.mean(fold_test_preds, axis=0)

    # Compute per-fold LOO MSE
    fold_loo_mses = fold_loo_mses / np.maximum(fold_loo_counts, 1)

    return test_preds, oof_preds, fold_test_preds, fold_loo_mses


# ==============================================================
# PHASE 3: Weighted Averaging
# ==============================================================

def weighted_average_predictions(fold_test_preds, fold_loo_mses):
    """Inverse-MSE weighted averaging of fold predictions."""
    # Inverse MSE weights (lower MSE = higher weight)
    inv_mses = 1.0 / (fold_loo_mses + 1e-10)
    weights = inv_mses / inv_mses.sum()

    test_preds = np.zeros(fold_test_preds.shape[1])
    for k in range(fold_test_preds.shape[0]):
        test_preds += weights[k] * fold_test_preds[k]

    return test_preds, weights


# ==============================================================
# PHASE 4: Bootstrap Bagging
# ==============================================================

def bootstrap_bagging_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                  B=10, sample_frac=0.8, bandwidth_quantile=0.3,
                                  alpha=10.0, seed=42):
    """Bootstrap-style bagging: sample sample_frac of training data B times,
    average B predictions."""
    unique_pids = sorted(np.unique(pids_train))
    n_test = len(X_test)

    bag_test_preds = np.zeros((B, n_test))
    rng = np.random.RandomState(seed)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        te_indices = np.where(te_mask)[0]
        n_player = len(X_tr)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        sigma = compute_sigma_for_player(X_tr_s, bandwidth_quantile)

        n_sample = max(3, int(n_player * sample_frac))

        for b in range(B):
            # Sample WITH replacement
            sample_idx = rng.choice(n_player, size=n_sample, replace=True)
            X_bag = X_tr_s[sample_idx]
            y_bag = y_tr[sample_idx]

            if len(X_te) > 0:
                bag_test_preds[b, te_indices] = per_example_ridge_predict(
                    X_bag, y_bag, X_te_s, sigma, alpha)

    test_preds = np.mean(bag_test_preds, axis=0)
    return test_preds, bag_test_preds


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("K-FOLD CROSS-FITTING FOR VARIANCE REDUCTION")
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

    # Load reference submissions
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_2063 = pd.read_csv(SUBMISSION_DIR / "submission_2063.csv")

    # Collect all results
    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        print("  Extracting features...")
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)
        print(f"  HC features: {X_train_hc.shape[1]}")

        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Augmented features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]
        target_results = {}

        # ----------------------------------------------------------
        # PHASE 1: Baseline
        # ----------------------------------------------------------
        print("\n  PHASE 1: Baseline (ALL data)")
        oof_base, test_base = baseline_prediction(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"    LOO MSE: {mse_base:.6f}")
        target_results['baseline'] = {
            'loo_mse': mse_base, 'test_preds': test_base, 'oof_preds': oof_base
        }

        # ----------------------------------------------------------
        # PHASE 2: K-Fold Cross-Fitting
        # ----------------------------------------------------------
        print("\n  PHASE 2: K-Fold Cross-Fitting")
        for K in [3, 5, 7, 10]:
            test_cf, oof_cf, fold_preds, fold_mses = cross_fitting_prediction(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                K=K, seed=42)
            mse_cf = np.mean((oof_cf - y_target) ** 2)

            # Std of test predictions across folds
            test_std = np.mean(np.std(fold_preds, axis=0))

            # Correlation between fold predictions
            fold_corrs = []
            for i in range(K):
                for j in range(i+1, K):
                    r = np.corrcoef(fold_preds[i], fold_preds[j])[0, 1]
                    fold_corrs.append(r)
            mean_fold_corr = np.mean(fold_corrs) if fold_corrs else 1.0

            print(f"    K={K:2d}: LOO MSE={mse_cf:.6f} (delta={mse_cf - mse_base:+.6f}), "
                  f"test_std={test_std:.6f}, fold_corr={mean_fold_corr:.4f}")

            target_results[f'cf_K{K}'] = {
                'loo_mse': mse_cf, 'test_preds': test_cf, 'oof_preds': oof_cf,
                'fold_preds': fold_preds, 'fold_mses': fold_mses,
                'test_std': test_std, 'mean_fold_corr': mean_fold_corr
            }

        # ----------------------------------------------------------
        # PHASE 3: Weighted Averaging (for best K values)
        # ----------------------------------------------------------
        print("\n  PHASE 3: Weighted Averaging (inverse-MSE)")
        for K in [3, 5, 7, 10]:
            key = f'cf_K{K}'
            fold_preds = target_results[key]['fold_preds']
            fold_mses = target_results[key]['fold_mses']
            test_weighted, weights = weighted_average_predictions(fold_preds, fold_mses)

            # Compare with equal-weight
            test_equal = target_results[key]['test_preds']
            diff = np.mean((test_weighted - test_equal) ** 2)

            print(f"    K={K:2d}: weight_range=[{weights.min():.4f}, {weights.max():.4f}], "
                  f"diff_from_equal={diff:.8f}")

            target_results[f'cf_K{K}_weighted'] = {
                'test_preds': test_weighted, 'weights': weights
            }

        # ----------------------------------------------------------
        # PHASE 4: Bootstrap Bagging
        # ----------------------------------------------------------
        print("\n  PHASE 4: Bootstrap Bagging")
        for B in [10, 20]:
            test_bag, bag_preds = bootstrap_bagging_prediction(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                B=B, seed=42)

            bag_std = np.mean(np.std(bag_preds, axis=0))
            bag_corrs = []
            for i in range(B):
                for j in range(i+1, min(B, i+5)):  # Sample some pairs
                    r = np.corrcoef(bag_preds[i], bag_preds[j])[0, 1]
                    bag_corrs.append(r)
            mean_bag_corr = np.mean(bag_corrs) if bag_corrs else 1.0

            print(f"    B={B:2d}: test_std={bag_std:.6f}, bag_corr={mean_bag_corr:.4f}")

            target_results[f'bag_B{B}'] = {
                'test_preds': test_bag, 'bag_preds': bag_preds,
                'test_std': bag_std, 'mean_bag_corr': mean_bag_corr
            }

        # ----------------------------------------------------------
        # PHASE 5: Stability Analysis (K=5, 5 seeds)
        # ----------------------------------------------------------
        print("\n  PHASE 5: Stability Analysis (K=5, seeds 0-4)")
        seed_test_preds = []
        seed_oof_mses = []
        for seed in range(5):
            test_cf, oof_cf, _, _ = cross_fitting_prediction(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                K=5, seed=seed)
            mse_cf = np.mean((oof_cf - y_target) ** 2)
            seed_test_preds.append(test_cf)
            seed_oof_mses.append(mse_cf)

        seed_test_preds = np.array(seed_test_preds)
        seed_std = np.mean(np.std(seed_test_preds, axis=0))
        seed_mse_std = np.std(seed_oof_mses)

        print(f"    LOO MSE across seeds: {np.mean(seed_oof_mses):.6f} +/- {seed_mse_std:.6f}")
        print(f"    Test pred std across seeds: {seed_std:.6f}")
        print(f"    Seed-averaged vs single-seed diff: {np.mean((np.mean(seed_test_preds, axis=0) - seed_test_preds[0]) ** 2):.8f}")

        # Multi-seed average
        test_multiseed = np.mean(seed_test_preds, axis=0)
        target_results['cf_K5_multiseed'] = {
            'test_preds': test_multiseed,
            'seed_std': seed_std,
            'seed_mse_std': seed_mse_std,
            'seed_oof_mses': seed_oof_mses
        }

        # ----------------------------------------------------------
        # PHASE 6: Effective Sample Size Analysis
        # ----------------------------------------------------------
        print("\n  PHASE 6: Effective Sample Size Analysis")
        for K in [3, 5, 7, 10]:
            # For each K, compute effective number of training samples per sub-model
            eff_sizes = []
            for pid in sorted(np.unique(pids_train)):
                n_player = (pids_train == pid).sum()
                n_fold_train = n_player - n_player // K  # approx (K-1)/K * n
                eff_sizes.append(n_fold_train)
            mean_eff = np.mean(eff_sizes)

            key = f'cf_K{K}'
            test_std = target_results[key]['test_std']
            fold_corr = target_results[key]['mean_fold_corr']

            # Theoretical variance reduction: var_avg = (1/K) * var_single * (1 + (K-1)*rho)
            # where rho = mean fold correlation
            theoretical_reduction = (1.0 / K) * (1 + (K - 1) * fold_corr)

            print(f"    K={K:2d}: mean_eff_samples={mean_eff:.1f}/{np.mean([(pids_train == pid).sum() for pid in np.unique(pids_train)]):.1f}, "
                  f"test_std={test_std:.6f}, fold_corr={fold_corr:.4f}, "
                  f"theoretical_var_frac={theoretical_reduction:.4f}")

        # ----------------------------------------------------------
        # PHASE 7: Diversity Analysis (correlation with Sub 2063)
        # ----------------------------------------------------------
        print("\n  PHASE 7: Diversity Analysis (correlation with Sub 2063)")
        col = f'scaled_{target}'
        ref_2063 = sub_2063[col].values
        ref_784 = sub_784[col].values

        # Baseline
        r_base_2063 = np.corrcoef(test_base, ref_2063)[0, 1]
        r_base_784 = np.corrcoef(test_base, ref_784)[0, 1]
        print(f"    Baseline:     r_2063={r_base_2063:.4f}, r_784={r_base_784:.4f}")

        for K in [3, 5, 7, 10]:
            key = f'cf_K{K}'
            r_2063 = np.corrcoef(target_results[key]['test_preds'], ref_2063)[0, 1]
            r_784 = np.corrcoef(target_results[key]['test_preds'], ref_784)[0, 1]
            print(f"    K={K:2d} equal:   r_2063={r_2063:.4f}, r_784={r_784:.4f}")

        for B in [10, 20]:
            key = f'bag_B{B}'
            r_2063 = np.corrcoef(target_results[key]['test_preds'], ref_2063)[0, 1]
            r_784 = np.corrcoef(target_results[key]['test_preds'], ref_784)[0, 1]
            print(f"    Bag B={B:2d}:    r_2063={r_2063:.4f}, r_784={r_784:.4f}")

        r_ms_2063 = np.corrcoef(test_multiseed, ref_2063)[0, 1]
        r_ms_784 = np.corrcoef(test_multiseed, ref_784)[0, 1]
        print(f"    K=5 multiseed: r_2063={r_ms_2063:.4f}, r_784={r_ms_784:.4f}")

        all_results[target] = target_results

    # ==============================================================
    # PHASE 8: Submission Generation
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 8: SUBMISSION GENERATION")
    print(f"{'=' * 70}")

    # Determine best variant per target
    # We'll generate submissions for: baseline, K=5, K=5 multiseed, K=5 weighted, Bag B=20
    variants = {
        'baseline': {},
        'cf_K5': {},
        'cf_K5_weighted': {},
        'cf_K5_multiseed': {},
        'cf_K10': {},
        'bag_B20': {},
    }

    for target in TARGETS:
        for vname in variants:
            if vname in all_results[target]:
                variants[vname][target] = all_results[target][vname]['test_preds']

    # Print summary table
    print("\n  Summary of test predictions (correlation with Sub 2063):")
    for vname in variants:
        if len(variants[vname]) == 3:
            corrs = []
            for target in TARGETS:
                col = f'scaled_{target}'
                r = np.corrcoef(variants[vname][target], sub_2063[col].values)[0, 1]
                corrs.append(r)
            print(f"    {vname:20s}: angle={corrs[0]:.4f}, depth={corrs[1]:.4f}, lr={corrs[2]:.4f}")

    # Generate standalone submissions
    for vname in variants:
        if len(variants[vname]) < 3:
            continue
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': variants[vname]['angle'],
            'scaled_depth': variants[vname]['depth'],
            'scaled_left_right': variants[vname]['left_right'],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = sub['scaled_angle'].std()
        d_mean = sub['scaled_depth'].mean()
        print(f"\n  Sub {sub_num}: {vname} (standalone)")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # Blend with Sub 784 at standard weights (aw=0.50, dw=0.30, lw=0.50)
    print("\n  Blended submissions (with Sub 784, aw=0.50, dw=0.30, lw=0.50):")
    blend_configs = [
        (0.50, 0.30, 0.50, "standard"),
    ]

    for vname in variants:
        if len(variants[vname]) < 3:
            continue
        for aw, dw, lw, desc in blend_configs:
            sub_num = get_next_submission_number()
            blended = sub_784.copy()
            blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*variants[vname]['angle']
            blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*variants[vname]['depth']
            blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*variants[vname]['left_right']
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

            a_std = blended['scaled_angle'].std()
            d_mean = blended['scaled_depth'].mean()
            print(f"  Sub {sub_num}: {vname} + 784 ({desc})")
            print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # Blend with Sub 2063 at different weights
    print("\n  Blended with Sub 2063:")
    for vname in ['cf_K5', 'cf_K5_multiseed', 'bag_B20']:
        if vname not in variants or len(variants[vname]) < 3:
            continue
        for w_new in [0.10, 0.20, 0.30]:
            sub_num = get_next_submission_number()
            blended = sub_2063.copy()
            for target in TARGETS:
                col = f'scaled_{target}'
                blended[col] = (1-w_new)*sub_2063[col] + w_new*variants[vname][target]
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

            a_std = blended['scaled_angle'].std()
            d_mean = blended['scaled_depth'].mean()
            print(f"  Sub {sub_num}: {w_new:.0%} {vname} + {1-w_new:.0%} Sub2063")
            print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # ==============================================================
    # Write results to research file
    # ==============================================================
    write_research_report(all_results)


def write_research_report(all_results):
    """Write results to Research/CROSS_FITTING_RESULTS.md"""
    report_path = PROJECT_DIR / "Research" / "CROSS_FITTING_RESULTS.md"

    lines = []
    lines.append("# Cross-Fitting Variance Reduction Results")
    lines.append(f"\nDate: 2026-02-12")
    lines.append(f"\nScript: scripts/cross_fitting.py")
    lines.append("")
    lines.append("## Concept")
    lines.append("Split training data into K folds per-player. For each fold, train per-example Ridge")
    lines.append("on (K-1) folds only. Average K predictions for test. Benefits: variance reduction,")
    lines.append("reduced overfitting to specific noisy samples.")
    lines.append("")

    lines.append("## Configuration")
    lines.append("- Per-example locally weighted Ridge (Gaussian kernel, alpha=10)")
    lines.append("- bandwidth_quantile=0.3")
    lines.append("- Features: 198 HC + 15 PLS per target at target-specific frames")
    lines.append("- Per-player splitting (critical: each player's data split separately)")
    lines.append("")

    for target in TARGETS:
        tr = all_results[target]
        lines.append(f"## {target.upper()}")
        lines.append("")

        # Phase 1
        lines.append("### Phase 1: Baseline")
        lines.append(f"- LOO MSE: {tr['baseline']['loo_mse']:.6f}")
        lines.append("")

        # Phase 2
        lines.append("### Phase 2: K-Fold Cross-Fitting")
        lines.append("| K | LOO MSE | Delta | Test Std | Fold Corr |")
        lines.append("|---|---------|-------|----------|-----------|")
        for K in [3, 5, 7, 10]:
            key = f'cf_K{K}'
            if key in tr:
                delta = tr[key]['loo_mse'] - tr['baseline']['loo_mse']
                lines.append(f"| {K} | {tr[key]['loo_mse']:.6f} | {delta:+.6f} | "
                           f"{tr[key]['test_std']:.6f} | {tr[key]['mean_fold_corr']:.4f} |")
        lines.append("")

        # Phase 4
        lines.append("### Phase 4: Bootstrap Bagging")
        for B in [10, 20]:
            key = f'bag_B{B}'
            if key in tr:
                lines.append(f"- B={B}: test_std={tr[key]['test_std']:.6f}, bag_corr={tr[key]['mean_bag_corr']:.4f}")
        lines.append("")

        # Phase 5
        if 'cf_K5_multiseed' in tr:
            lines.append("### Phase 5: Stability (K=5, 5 seeds)")
            ms = tr['cf_K5_multiseed']
            lines.append(f"- LOO MSE mean: {np.mean(ms['seed_oof_mses']):.6f} +/- {ms['seed_mse_std']:.6f}")
            lines.append(f"- Test pred std across seeds: {ms['seed_std']:.6f}")
            lines.append("")

    lines.append("## Key Findings")
    lines.append("- TBD (to be filled after LB testing)")
    lines.append("")

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nResearch report written to: {report_path}")


if __name__ == "__main__":
    main()
