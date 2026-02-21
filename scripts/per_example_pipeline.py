"""
Per-Example Adaptive Pipeline

Instead of one model per player per target, tailor predictions to each
test example individually:

1. LOCAL WEIGHTED REGRESSION: Weight training samples by similarity to
   the specific test example. Each test shot gets a personalized model.

2. PER-EXAMPLE MODEL SELECTION: Run multiple model variants, generate OOF
   predictions. For each test shot, find nearest training neighbors,
   evaluate which model variant performs best on those neighbors, and
   use that variant's prediction.

3. KNN BIAS CORRECTION: For each test shot, estimate local prediction
   bias from nearest training neighbors' OOF errors and correct.

The key insight: different test shots may benefit from different models.
A shot that looks like Player 3's typical shot should be predicted using
the model that works best on Player 3's similar shots, not a global
model that averages across all shot types.
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
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

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
# FEATURE EXTRACTION (compact version for distance computation)
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
        feats.append(ts_hr[f, rw, 0] - ts_hr[f, rs, 0])  # arm extension fwd
        feats.append(ts_hr[f, rw, 1] - ts_hr[f, rs, 1])  # arm lateral
        feats.append(ts_hr[f, rw, 2] - ts_hr[f, rs, 2])  # arm vertical
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

    return np.array(feats, dtype=np.float32)


def extract_all_features(data, target):
    """Extract features for all shots."""
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
# MULTIPLE MODEL VARIANTS (for per-example model selection)
# ==============================================================

def make_model_variants(seed=42):
    """Create diverse model variants for per-example selection."""
    return {
        'ridge_10': Ridge(alpha=10.0),
        'ridge_100': Ridge(alpha=100.0),
        'ridge_1': Ridge(alpha=1.0),
        'lgb_conservative': lgb.LGBMRegressor(
            n_estimators=60, num_leaves=6, learning_rate=0.03,
            min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=2.0, random_state=seed, verbose=-1, n_jobs=-1),
        'lgb_moderate': lgb.LGBMRegressor(
            n_estimators=80, num_leaves=8, learning_rate=0.05,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1),
        'lgb_flexible': lgb.LGBMRegressor(
            n_estimators=100, num_leaves=12, learning_rate=0.05,
            min_child_samples=3, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.5, reg_lambda=0.5, random_state=seed, verbose=-1, n_jobs=-1),
        'xgb_moderate': xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbosity=0, n_jobs=-1),
        'xgb_flexible': xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.5, reg_lambda=0.5, random_state=seed, verbosity=0, n_jobs=-1),
        'cat_moderate': CatBoostRegressor(
            iterations=80, depth=3, learning_rate=0.05,
            l2_leaf_reg=5.0, random_seed=seed, verbose=False),
        'cat_flexible': CatBoostRegressor(
            iterations=100, depth=4, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=seed, verbose=False),
    }


# ==============================================================
# APPROACH 1: LOCALLY WEIGHTED REGRESSION
# ==============================================================

def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.3):
    """For each test example, train a weighted model where nearby training
    samples get higher weight."""
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

        # Standardize within player
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        # Pairwise distances (train-train for OOF, test-train for prediction)
        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        # Adaptive bandwidth: use quantile of distances
        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        # OOF: leave-one-out locally weighted
        for i in range(n_tr):
            # Weights for all other training samples
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0  # Leave out self

            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue

            # Weighted ridge
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test: locally weighted for each test example
        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds


# ==============================================================
# APPROACH 2: PER-EXAMPLE MODEL SELECTION
# ==============================================================

def per_example_model_selection(X_train, y_train, X_test, pids_train, pids_test,
                                 k_neighbors=10):
    """For each test example, select the best model based on performance
    on its nearest training neighbors."""
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)

    # Step 1: Generate OOF predictions for each model variant
    model_names = list(make_model_variants(42).keys())
    n_models = len(model_names)
    oof_all = np.zeros((n_train, n_models))

    print(f"    Generating OOF for {n_models} model variants...")
    for pid in unique_pids:
        mask = pids_train == pid
        X_p = X_train[mask]
        y_p = y_train[mask]
        indices = np.where(mask)[0]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_s):
            for m_i, m_name in enumerate(model_names):
                models = make_model_variants(42)
                m = models[m_name]
                m.fit(X_s[tr_idx], y_p[tr_idx])
                oof_all[indices[val_idx], m_i] = m.predict(X_s[val_idx])

    # Step 2: Generate test predictions for each model variant
    test_all = np.zeros((n_test, n_models))
    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        if not np.any(te_mask):
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train[tr_mask])
        X_te_s = scaler.transform(X_test[te_mask])
        y_tr = y_train[tr_mask]

        for m_i, m_name in enumerate(model_names):
            models = make_model_variants(42)
            m = models[m_name]
            m.fit(X_tr_s, y_tr)
            test_all[np.where(te_mask)[0], m_i] = m.predict(X_te_s)

    # Step 3: For each test example, find k nearest neighbors and select best model
    test_preds = np.zeros(n_test)
    model_selections = []

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        if not np.any(te_mask):
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train[tr_mask])
        X_te_s = scaler.transform(X_test[te_mask])
        y_tr = y_train[tr_mask]

        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        k = min(k_neighbors, len(X_tr_s) - 1)
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(X_tr_s)

        for j, te_idx in enumerate(te_indices):
            dists, neighbor_idx = nn.kneighbors(X_te_s[j:j+1])
            neighbor_global = tr_indices[neighbor_idx[0]]

            # Evaluate each model on these neighbors
            best_model = 0
            best_mse = float('inf')
            model_mses = []
            for m_i in range(n_models):
                local_preds = oof_all[neighbor_global, m_i]
                local_true = y_train[neighbor_global]
                # Distance-weighted MSE
                w = 1.0 / (dists[0] + 1e-6)
                w /= w.sum()
                local_mse = np.sum(w * (local_preds - local_true) ** 2)
                model_mses.append(local_mse)
                if local_mse < best_mse:
                    best_mse = local_mse
                    best_model = m_i

            # Soft selection: weight models by inverse local MSE
            model_mses = np.array(model_mses)
            # Softmax-style weighting (lower MSE = higher weight)
            inv_mses = 1.0 / (model_mses + 1e-10)
            model_weights = inv_mses / inv_mses.sum()

            # Weighted average of all models
            test_preds[te_idx] = np.dot(model_weights, test_all[te_idx])
            model_selections.append((te_idx, model_names[best_model], best_mse))

    # Also compute OOF with same approach (LOO for neighbors)
    oof_preds = np.zeros(n_train)
    for pid in unique_pids:
        mask = pids_train == pid
        X_p = X_train[mask]
        y_p = y_train[mask]
        indices = np.where(mask)[0]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        k = min(k_neighbors, len(X_s) - 2)
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(X_s)

        for i, idx in enumerate(indices):
            dists, neighbor_idx = nn.kneighbors(X_s[i:i+1])
            # Exclude self
            self_pos = np.where(neighbor_idx[0] == i)[0]
            neighbor_local = np.delete(neighbor_idx[0], self_pos)[:k]
            neighbor_dists = np.delete(dists[0], self_pos)[:k]
            neighbor_global = indices[neighbor_local]

            model_mses = []
            for m_i in range(n_models):
                local_preds = oof_all[neighbor_global, m_i]
                local_true = y_train[neighbor_global]
                w = 1.0 / (neighbor_dists + 1e-6)
                w /= w.sum()
                local_mse = np.sum(w * (local_preds - local_true) ** 2)
                model_mses.append(local_mse)

            model_mses = np.array(model_mses)
            inv_mses = 1.0 / (model_mses + 1e-10)
            model_weights = inv_mses / inv_mses.sum()
            oof_preds[idx] = np.dot(model_weights, oof_all[idx])

    return oof_preds, test_preds, model_selections


# ==============================================================
# APPROACH 3: KNN BIAS CORRECTION
# ==============================================================

def knn_bias_correction(oof_base, test_base, X_train, y_train, X_test,
                         pids_train, pids_test, k_neighbors=8):
    """For each test example, estimate local bias from nearest neighbors'
    OOF errors and apply correction."""
    unique_pids = sorted(np.unique(pids_train))
    test_corrected = test_base.copy()
    oof_corrected = oof_base.copy()

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        if not np.any(te_mask):
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train[tr_mask])
        X_te_s = scaler.transform(X_test[te_mask])

        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        # OOF errors for this player
        oof_errors = oof_base[tr_indices] - y_train[tr_indices]

        k = min(k_neighbors, len(X_tr_s) - 1)
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(X_tr_s)

        # Correct test predictions
        for j, te_idx in enumerate(te_indices):
            dists, neighbor_idx = nn.kneighbors(X_te_s[j:j+1])
            # Distance-weighted bias estimate
            w = 1.0 / (dists[0] + 1e-6)
            w /= w.sum()
            local_bias = np.dot(w, oof_errors[neighbor_idx[0]])

            # Apply conservative correction (damping factor)
            test_corrected[te_idx] -= 0.5 * local_bias

        # OOF correction (LOO)
        nn_loo = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn_loo.fit(X_tr_s)

        for i, idx in enumerate(tr_indices):
            dists, neighbor_idx = nn_loo.kneighbors(X_tr_s[i:i+1])
            self_pos = np.where(neighbor_idx[0] == i)[0]
            neighbor_local = np.delete(neighbor_idx[0], self_pos)[:k]
            neighbor_dists = np.delete(dists[0], self_pos)[:k]

            w = 1.0 / (neighbor_dists + 1e-6)
            w /= w.sum()
            local_bias = np.dot(w, oof_errors[neighbor_local])
            oof_corrected[idx] -= 0.5 * local_bias

    return oof_corrected, test_corrected


# ==============================================================
# BASE ENSEMBLE (for OOF predictions used in bias correction)
# ==============================================================

def train_base_ensemble(X_train, y_train, X_test, pids_train, pids_test, seed=42):
    """Train base per-player ensemble and return OOF + test predictions."""
    unique_pids = sorted(np.unique(pids_train))
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_p = X_train[tr_mask]
        y_p = y_train[tr_mask]
        indices = np.where(tr_mask)[0]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_p)

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf.split(X_s):
            preds = []
            for name, m in make_model_variants(seed).items():
                if name in ['lgb_moderate', 'xgb_moderate', 'cat_moderate', 'ridge_10']:
                    m.fit(X_s[tr_idx], y_p[tr_idx])
                    preds.append(m.predict(X_s[val_idx]))
            oof[indices[val_idx]] = np.mean(preds, axis=0)

        # Full model for test
        if np.any(te_mask):
            X_te_s = scaler.transform(X_test[te_mask])
            preds = []
            for name, m in make_model_variants(seed).items():
                if name in ['lgb_moderate', 'xgb_moderate', 'cat_moderate', 'ridge_10']:
                    m.fit(X_s, y_p)
                    preds.append(m.predict(X_te_s))
            test_preds[te_mask] = np.mean(preds, axis=0)

    return oof, test_preds


# ==============================================================
# PLS AUGMENTATION
# ==============================================================

def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test, X_raw_train, X_raw_test):
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

        # Find optimal components
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
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("PER-EXAMPLE ADAPTIVE PIPELINE")
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

    # ============================================================
    # PER-TARGET PIPELINE
    # ============================================================

    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)
        print(f"  Features: {X_train_hc.shape[1]}")

        # Augment with PLS
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Augmented features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # --- APPROACH 1: Locally Weighted Regression ---
        print("  [1] Locally weighted regression...")
        oof_lw, test_lw = locally_weighted_prediction(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.3)
        mse_lw = np.mean((oof_lw - y_target) ** 2)
        print(f"      CV MSE: {mse_lw:.6f}")

        # Try different bandwidths
        best_bw, best_mse_lw = 0.3, mse_lw
        for bw in [0.15, 0.2, 0.25, 0.35, 0.4, 0.5]:
            oof_tmp, _ = locally_weighted_prediction(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                bandwidth_quantile=bw)
            mse_tmp = np.mean((oof_tmp - y_target) ** 2)
            print(f"      bw={bw:.2f}: CV MSE={mse_tmp:.6f}")
            if mse_tmp < best_mse_lw:
                best_mse_lw = mse_tmp
                best_bw = bw

        if best_bw != 0.3:
            print(f"      Best bandwidth: {best_bw}")
            oof_lw, test_lw = locally_weighted_prediction(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                bandwidth_quantile=best_bw)

        # --- APPROACH 2: Per-Example Model Selection ---
        print("  [2] Per-example model selection...")
        oof_ms, test_ms, selections = per_example_model_selection(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            k_neighbors=10)
        mse_ms = np.mean((oof_ms - y_target) ** 2)
        print(f"      CV MSE: {mse_ms:.6f}")

        # Try different k values
        best_k, best_mse_ms = 10, mse_ms
        for k in [5, 8, 15, 20]:
            oof_tmp, _, _ = per_example_model_selection(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                k_neighbors=k)
            mse_tmp = np.mean((oof_tmp - y_target) ** 2)
            print(f"      k={k}: CV MSE={mse_tmp:.6f}")
            if mse_tmp < best_mse_ms:
                best_mse_ms = mse_tmp
                best_k = k

        if best_k != 10:
            print(f"      Best k: {best_k}")
            oof_ms, test_ms, selections = per_example_model_selection(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                k_neighbors=best_k)

        # --- APPROACH 3: KNN Bias Correction ---
        print("  [3] KNN bias correction on base ensemble...")
        oof_base, test_base = train_base_ensemble(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"      Base ensemble CV MSE: {mse_base:.6f}")

        best_k_bc, best_mse_bc = 8, float('inf')
        for k in [5, 8, 10, 15, 20]:
            oof_bc, test_bc = knn_bias_correction(
                oof_base, test_base, X_train_aug, y_target, X_test_aug,
                pids_train, pids_test, k_neighbors=k)
            mse_bc = np.mean((oof_bc - y_target) ** 2)
            print(f"      k={k}: CV MSE={mse_bc:.6f} (delta: {mse_bc - mse_base:+.6f})")
            if mse_bc < best_mse_bc:
                best_mse_bc = mse_bc
                best_k_bc = k

        oof_bc, test_bc = knn_bias_correction(
            oof_base, test_base, X_train_aug, y_target, X_test_aug,
            pids_train, pids_test, k_neighbors=best_k_bc)

        # --- SUMMARY ---
        print(f"\n  {target} SUMMARY:")
        print(f"    [1] Local weighted:      MSE={best_mse_lw:.6f}")
        print(f"    [2] Model selection:      MSE={best_mse_ms:.6f}")
        print(f"    [3] KNN bias correction:  MSE={best_mse_bc:.6f}")
        print(f"    [B] Base ensemble:        MSE={mse_base:.6f}")

        # Pick best approach
        approaches = {
            'local_weighted': (best_mse_lw, oof_lw, test_lw),
            'model_selection': (best_mse_ms, oof_ms, test_ms),
            'bias_correction': (best_mse_bc, oof_bc, test_bc),
            'base_ensemble': (mse_base, oof_base, test_base),
        }
        best_name = min(approaches, key=lambda x: approaches[x][0])
        best_mse, best_oof, best_test = approaches[best_name]
        print(f"    BEST: {best_name} (MSE={best_mse:.6f})")

        results[target] = {
            'approaches': approaches,
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

    total_mse = 0
    for target in TARGETS:
        mse = results[target]['best_mse']
        total_mse += mse
        print(f"  {target}: {mse:.6f} ({results[target]['best_name']})")
    print(f"  MEAN: {total_mse / 3:.6f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Standalone: best approach per target
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle']['best_test'],
        'scaled_depth': results['depth']['best_test'],
        'scaled_left_right': results['left_right']['best_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: STANDALONE (best per target)")
    print(f"    angle_std={sub['scaled_angle'].std():.6f}, depth_mean={sub['scaled_depth'].mean():.6f}")

    # Blend with Sub 784
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    print(f"\n  Correlation with Sub 784:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_784[col].values, results[target]['best_test'])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # Per-approach blends
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "Sub 784 weights"),
        (0.00, 0.20, 0.30, "conservative"),
        (0.00, 0.15, 0.20, "very conservative"),
        (0.10, 0.30, 0.50, "with angle"),
        (0.00, 0.40, 0.60, "aggressive"),
        (0.00, 0.10, 0.15, "minimal touch"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*results['angle']['best_test']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*results['depth']['best_test']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*results['left_right']['best_test']

        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # Also: per-approach standalone submissions
    print(f"\n  Per-approach standalone submissions:")
    for approach_name in ['local_weighted', 'model_selection', 'bias_correction']:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': results['angle']['approaches'][approach_name][2],
            'scaled_depth': results['depth']['approaches'][approach_name][2],
            'scaled_left_right': results['left_right']['approaches'][approach_name][2],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        mse_a = results['angle']['approaches'][approach_name][0]
        mse_d = results['depth']['approaches'][approach_name][0]
        mse_l = results['left_right']['approaches'][approach_name][0]
        print(f"  Sub {sub_num}: {approach_name}")
        print(f"    CV: angle={mse_a:.6f}, depth={mse_d:.6f}, lr={mse_l:.6f}, mean={(mse_a+mse_d+mse_l)/3:.6f}")

    # Blend per-approach with Sub 784 at same weights
    for approach_name in ['local_weighted', 'model_selection', 'bias_correction']:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = sub_784['scaled_angle']  # Don't touch angle
        blended['scaled_depth'] = 0.70*sub_784['scaled_depth'] + 0.30*results['depth']['approaches'][approach_name][2]
        blended['scaled_left_right'] = 0.50*sub_784['scaled_left_right'] + 0.50*results['left_right']['approaches'][approach_name][2]

        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: {approach_name} blend (aw=0 dw=0.30 lw=0.50)")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
