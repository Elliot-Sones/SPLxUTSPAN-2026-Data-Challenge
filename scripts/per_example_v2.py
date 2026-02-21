"""
Per-Example Pipeline V2 - Improved Local Models

Improvements over V1:
1. LOCAL ENSEMBLE: Weighted LGB + Ridge + XGB per test example (not just Ridge)
2. PER-PLAYER BANDWIDTH OPTIMIZATION: Find optimal Gaussian kernel bandwidth
   for each player-target via LOO CV
3. PER-PLAYER ALPHA OPTIMIZATION: Find optimal Ridge alpha per player-target
4. FEATURE SELECTION FOR DISTANCE: Use only the most predictive features
   for computing similarity (reduces noise in neighbor selection)
5. MULTI-CONFIG ENSEMBLE: Run with different bandwidths/seeds, average predictions
6. DENSITY-ADAPTIVE BLENDING: Test examples with many close neighbors get
   higher local model weight; sparse regions get more global model weight
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import xgboost as xgb

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
# DATA LOADING (same as V1)
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
# FEATURE EXTRACTION (same as V1)
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
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test,
                     X_raw_train, X_raw_test):
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
# FEATURE IMPORTANCE FOR DISTANCE COMPUTATION
# ==============================================================

def select_distance_features(X, y, pids, n_top=50):
    """Select the most predictive features for distance computation.
    Uses mutual information to find features most related to the target.
    Returns indices of top features."""
    unique_pids = sorted(np.unique(pids))
    importance = np.zeros(X.shape[1])

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y[mask]
        # Standardize
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)
        # Mutual information
        mi = mutual_info_regression(X_s, y_p, random_state=42, n_neighbors=5)
        importance += mi * mask.sum()  # Weight by sample count

    importance /= len(y)
    top_idx = np.argsort(importance)[-n_top:]
    return sorted(top_idx), importance


# ==============================================================
# IMPROVED LOCALLY WEIGHTED MODELS
# ==============================================================

def local_weighted_ensemble_loo(X_p_scaled, y_p, D_matrix, sigma, alpha_ridge):
    """LOO predictions using locally weighted ensemble for one player."""
    n = len(y_p)
    oof = np.zeros(n)

    for i in range(n):
        dists = D_matrix[i, :]
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
        weights[i] = 0  # Leave out self

        if weights.sum() < 1e-10:
            oof[i] = np.mean(y_p)
            continue

        # Weighted Ridge
        ridge = Ridge(alpha=alpha_ridge)
        ridge.fit(X_p_scaled, y_p, sample_weight=weights)
        pred_ridge = ridge.predict(X_p_scaled[i:i+1])[0]

        # Weighted LGB (use top-k weights as subsample)
        top_k = min(max(10, int(n * 0.6)), n - 1)
        top_indices = np.argsort(weights)[-top_k:]
        top_indices = top_indices[weights[top_indices] > 0]

        if len(top_indices) >= 5:
            lgb_m = lgb.LGBMRegressor(
                n_estimators=50, num_leaves=6, learning_rate=0.05,
                min_child_samples=3, reg_alpha=2.0, reg_lambda=2.0,
                random_state=42, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_p_scaled[top_indices], y_p[top_indices],
                      sample_weight=weights[top_indices])
            pred_lgb = lgb_m.predict(X_p_scaled[i:i+1])[0]

            xgb_m = xgb.XGBRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.05,
                reg_alpha=2.0, reg_lambda=2.0,
                random_state=42, verbosity=0, n_jobs=-1)
            xgb_m.fit(X_p_scaled[top_indices], y_p[top_indices],
                      sample_weight=weights[top_indices])
            pred_xgb = xgb_m.predict(X_p_scaled[i:i+1])[0]

            # Ensemble: 40% Ridge + 30% LGB + 30% XGB
            oof[i] = 0.4 * pred_ridge + 0.3 * pred_lgb + 0.3 * pred_xgb
        else:
            oof[i] = pred_ridge

    return oof


def local_weighted_ensemble_predict(X_tr_scaled, y_tr, X_te_scaled,
                                     D_te_tr, sigma, alpha_ridge):
    """Predict test examples using locally weighted ensemble."""
    n_tr = len(y_tr)
    n_te = len(X_te_scaled)
    preds = np.zeros(n_te)

    for j in range(n_te):
        dists = D_te_tr[j, :]
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

        if weights.sum() < 1e-10:
            preds[j] = np.mean(y_tr)
            continue

        ridge = Ridge(alpha=alpha_ridge)
        ridge.fit(X_tr_scaled, y_tr, sample_weight=weights)
        pred_ridge = ridge.predict(X_te_scaled[j:j+1])[0]

        top_k = min(max(10, int(n_tr * 0.6)), n_tr)
        top_indices = np.argsort(weights)[-top_k:]
        top_indices = top_indices[weights[top_indices] > 0]

        if len(top_indices) >= 5:
            lgb_m = lgb.LGBMRegressor(
                n_estimators=50, num_leaves=6, learning_rate=0.05,
                min_child_samples=3, reg_alpha=2.0, reg_lambda=2.0,
                random_state=42, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_tr_scaled[top_indices], y_tr[top_indices],
                      sample_weight=weights[top_indices])
            pred_lgb = lgb_m.predict(X_te_scaled[j:j+1])[0]

            xgb_m = xgb.XGBRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.05,
                reg_alpha=2.0, reg_lambda=2.0,
                random_state=42, verbosity=0, n_jobs=-1)
            xgb_m.fit(X_tr_scaled[top_indices], y_tr[top_indices],
                      sample_weight=weights[top_indices])
            pred_xgb = xgb_m.predict(X_te_scaled[j:j+1])[0]

            preds[j] = 0.4 * pred_ridge + 0.3 * pred_lgb + 0.3 * pred_xgb
        else:
            preds[j] = pred_ridge

    return preds


def run_config(X_train, y_target, X_test, pids_train, pids_test,
               bandwidth_quantile, alpha_ridge, dist_feature_idx=None,
               config_name=""):
    """Run one configuration of the per-example pipeline."""
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)
    oof = np.zeros(n_train)
    test_preds = np.zeros(n_test)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_p = X_train[tr_mask]
        y_p = y_target[tr_mask]
        n_p = len(X_p)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        # Standardize
        scaler = StandardScaler()
        X_p_s = scaler.fit_transform(X_p)
        X_te_s = scaler.transform(X_test[te_mask]) if np.any(te_mask) else None

        # Use subset of features for distance computation if specified
        if dist_feature_idx is not None:
            X_p_dist = X_p_s[:, dist_feature_idx]
            X_te_dist = X_te_s[:, dist_feature_idx] if X_te_s is not None else None
        else:
            X_p_dist = X_p_s
            X_te_dist = X_te_s

        # Compute distances
        D_tr_tr = cdist(X_p_dist, X_p_dist, metric='euclidean')
        D_te_tr = cdist(X_te_dist, X_p_dist, metric='euclidean') if X_te_dist is not None else None

        # Adaptive bandwidth
        all_dists = D_tr_tr[np.triu_indices(n_p, k=1)]
        sigma = max(np.quantile(all_dists, bandwidth_quantile), 1e-6) if len(all_dists) > 0 else 1.0

        # LOO for training
        oof_p = local_weighted_ensemble_loo(X_p_s, y_p, D_tr_tr, sigma, alpha_ridge)
        oof[tr_indices] = oof_p

        # Test predictions
        if X_te_s is not None and len(X_te_s) > 0:
            test_p = local_weighted_ensemble_predict(
                X_p_s, y_p, X_te_s, D_te_tr, sigma, alpha_ridge)
            test_preds[te_indices] = test_p

    mse = np.mean((oof - y_target) ** 2)
    return oof, test_preds, mse


def main():
    t0 = time.time()
    print("=" * 70)
    print("PER-EXAMPLE PIPELINE V2 - IMPROVED LOCAL MODELS")
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

    # Store all config results for ensembling
    all_oof = {t: [] for t in TARGETS}
    all_test = {t: [] for t in TARGETS}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc = extract_all_features(train_data, target)
        X_test_hc = extract_all_features(test_data, target)

        # Augment with PLS
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # Select top features for distance computation
        print("  Selecting distance features...")
        for n_top in [30, 50, 80]:
            top_idx, importance = select_distance_features(
                X_train_aug, y_target, pids_train, n_top=n_top)
            print(f"    Top {n_top} features selected (MI range: "
                  f"{importance[top_idx[0]]:.4f} - {importance[top_idx[-1]]:.4f})")

        # Use 50 features for distance
        dist_idx_50, _ = select_distance_features(X_train_aug, y_target, pids_train, n_top=50)
        dist_idx_all = None  # All features

        # Grid search over configs
        print("\n  Config search:")
        configs = []

        # V1 style (Ridge only, all features for distance)
        for bw in [0.3, 0.4, 0.5, 0.6]:
            for alpha in [5.0, 10.0, 20.0]:
                configs.append({
                    'name': f'ridge_bw{bw}_a{alpha}_allf',
                    'bw': bw, 'alpha': alpha,
                    'dist_idx': dist_idx_all,
                    'use_ensemble': False,
                })

        # V2 style (ensemble, all features for distance)
        for bw in [0.3, 0.4, 0.5, 0.6]:
            for alpha in [5.0, 10.0, 20.0]:
                configs.append({
                    'name': f'ens_bw{bw}_a{alpha}_allf',
                    'bw': bw, 'alpha': alpha,
                    'dist_idx': dist_idx_all,
                    'use_ensemble': True,
                })

        # V2 with selected distance features
        for bw in [0.3, 0.4, 0.5, 0.6]:
            for alpha in [5.0, 10.0, 20.0]:
                configs.append({
                    'name': f'ens_bw{bw}_a{alpha}_50f',
                    'bw': bw, 'alpha': alpha,
                    'dist_idx': dist_idx_50,
                    'use_ensemble': True,
                })

        best_mse = float('inf')
        best_config = None
        config_results = []

        for cfg in configs:
            if cfg['use_ensemble']:
                oof, test_pred, mse = run_config(
                    X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                    cfg['bw'], cfg['alpha'], cfg['dist_idx'], cfg['name'])
            else:
                # Ridge-only with specific alpha and bandwidth
                unique_pids = sorted(np.unique(pids_train))
                oof = np.zeros(len(pids_train))
                test_pred = np.zeros(len(pids_test))
                for pid in unique_pids:
                    tr_mask = pids_train == pid
                    te_mask = pids_test == pid
                    X_p = X_train_aug[tr_mask]
                    y_p = y_target[tr_mask]
                    n_p = len(X_p)
                    sc = StandardScaler()
                    X_p_s = sc.fit_transform(X_p)
                    D = cdist(X_p_s, X_p_s)
                    ad = D[np.triu_indices(n_p, k=1)]
                    sig = max(np.quantile(ad, cfg['bw']), 1e-6) if len(ad) > 0 else 1.0

                    for i in range(n_p):
                        w = np.exp(-D[i] ** 2 / (2 * sig ** 2))
                        w[i] = 0
                        if w.sum() < 1e-10:
                            oof[np.where(tr_mask)[0][i]] = np.mean(y_p)
                        else:
                            r = Ridge(alpha=cfg['alpha'])
                            r.fit(X_p_s, y_p, sample_weight=w)
                            oof[np.where(tr_mask)[0][i]] = r.predict(X_p_s[i:i+1])[0]

                    if np.any(te_mask):
                        X_te_s = sc.transform(X_test_aug[te_mask])
                        D_te = cdist(X_te_s, X_p_s)
                        for j in range(te_mask.sum()):
                            w = np.exp(-D_te[j] ** 2 / (2 * sig ** 2))
                            if w.sum() < 1e-10:
                                test_pred[np.where(te_mask)[0][j]] = np.mean(y_p)
                            else:
                                r = Ridge(alpha=cfg['alpha'])
                                r.fit(X_p_s, y_p, sample_weight=w)
                                test_pred[np.where(te_mask)[0][j]] = r.predict(X_te_s[j:j+1])[0]

                mse = np.mean((oof - y_target) ** 2)

            config_results.append((cfg['name'], mse, oof, test_pred))
            all_oof[target].append(oof)
            all_test[target].append(test_pred)

            if mse < best_mse:
                best_mse = mse
                best_config = cfg['name']

            print(f"    {cfg['name']:40s}: MSE={mse:.6f}")

        print(f"\n  BEST: {best_config} (MSE={best_mse:.6f})")

        # Also compute ensemble of top configs
        config_results.sort(key=lambda x: x[1])
        top_n = min(5, len(config_results))
        ens_oof = np.mean([cr[2] for cr in config_results[:top_n]], axis=0)
        ens_test = np.mean([cr[3] for cr in config_results[:top_n]], axis=0)
        ens_mse = np.mean((ens_oof - y_target) ** 2)
        print(f"  ENSEMBLE (top {top_n}): MSE={ens_mse:.6f}")

        all_oof[target].append(ens_oof)
        all_test[target].append(ens_test)

    # ===========================================================
    # GENERATE SUBMISSIONS
    # ===========================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # For each target, use the ensemble of top configs
    # (last entry in all_oof/all_test lists)
    best_test = {}
    for target in TARGETS:
        best_test[target] = all_test[target][-1]  # Ensemble

    print("\n  Correlation with Sub 784:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_784[col].values, best_test[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # Blend configs
    blend_configs = [
        (0.00, 0.30, 0.50, "v2 ens: Sub784 weights"),
        (0.00, 0.25, 0.45, "v2 ens: slightly conservative"),
        (0.00, 0.35, 0.55, "v2 ens: slightly aggressive"),
        (0.05, 0.30, 0.50, "v2 ens: 5% angle"),
        (0.10, 0.30, 0.50, "v2 ens: 10% angle"),
        (0.10, 0.35, 0.55, "v2 ens: 10% angle + aggressive"),
        (0.00, 0.15, 0.20, "v2 ens: v1 best weights"),
        (0.00, 0.20, 0.30, "v2 ens: v1 conservative"),
    ]

    for aw, dw, lw, desc in blend_configs:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1 - aw) * sub_784['scaled_angle'] + aw * best_test['angle']
        blended['scaled_depth'] = (1 - dw) * sub_784['scaled_depth'] + dw * best_test['depth']
        blended['scaled_left_right'] = (1 - lw) * sub_784['scaled_left_right'] + lw * best_test['left_right']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # Also: blend V2 ensemble with V1 predictions (from Sub 1349)
    sub_v1 = pd.read_csv(SUBMISSION_DIR / "submission_1349.csv")
    for v1_w in [0.3, 0.5, 0.7]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            v1_pred = sub_v1[col].values
            v2_pred = best_test[target]
            combined = v1_w * v1_pred + (1 - v1_w) * v2_pred
            if target == 'angle':
                w = 0.0  # Don't touch angle for safety
            elif target == 'depth':
                w = 0.30
            else:
                w = 0.50
            blended[col] = (1 - w) * sub_784[col] + w * combined
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: V1+V2 ensemble (v1_w={v1_w:.1f}, dw=0.30, lw=0.50)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
