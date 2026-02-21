"""
Structural Overfitting Attack - 4 Approaches

The entire gap from LOO (0.003743) to LB (0.006619) is overfitting.
Incremental regularization gives incremental gains. These 4 approaches
attack the problem structurally.

1. PSEUDO-LABELING: Add test predictions as soft labels -> 458 samples
2. RANDOM SUBSPACE ENSEMBLE: Average 500 models with random 5-feature subsets
3. LOPO MODEL SELECTION: Use leave-one-player-out (not LOO) for hyperparameters
4. BAGGED PREDICTIONS: Bootstrap training sets, average predictions
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

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
TARGETS = ["angle", "depth", "left_right"]
TARGET_IDX = {"angle": 0, "depth": 1, "left_right": 2}


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


KEY_JOINTS = ['right_wrist', 'right_elbow', 'right_shoulder',
              'left_wrist', 'left_shoulder',
              'right_hip', 'left_hip', 'mid_hip',
              'right_knee', 'left_knee', 'neck', 'nose']


def extract_hc_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    f = int(np.clip(frame, 0, 239))
    feats = []
    for jname in KEY_JOINTS:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])
    for jname in KEY_JOINTS:
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
    lw_kp = kp_index.get('left_wrist')
    if lw_kp is not None and rw is not None:
        feats.append(ts_hr[f, lw_kp, 1] - ts_hr[f, rw, 1])
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


def extract_all_features(data, frame):
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_hc_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test,
                     X_raw_train, X_raw_test, max_nc=15):
    unique_pids = sorted(np.unique(pids_train))
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


def per_example_lw(X_train, y_train, X_test, pids_train, pids_test,
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


# ================================================================
# APPROACH 1: PSEUDO-LABELING
# ================================================================

def pseudo_labeling(X_train, y_train, X_test, pids_train, pids_test,
                    pseudo_labels, pseudo_weight=0.5, n_iterations=3,
                    bandwidth_quantile=0.5, alpha=50.0):
    """
    Add test predictions as soft labels, retrain with increased sample size.
    pseudo_weight: how much to trust pseudo-labels (0=ignore, 1=equal to real)
    """
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)

    # Combine train + test features
    X_all = np.vstack([X_train, X_test])
    pids_all = np.concatenate([pids_train, pids_test])

    # Start with provided pseudo-labels
    current_pseudo = pseudo_labels.copy()

    for iteration in range(n_iterations):
        # Build combined targets: real labels + pseudo-labels
        y_all = np.concatenate([y_train, current_pseudo])
        sample_weights_all = np.concatenate([
            np.ones(n_train),
            np.full(n_test, pseudo_weight)
        ])

        # Train per-example model on combined data, predict test
        test_preds = np.zeros(n_test)

        for pid in unique_pids:
            # All samples for this player (train + test)
            all_mask = pids_all == pid
            te_mask_in_all = np.zeros(len(X_all), dtype=bool)
            te_mask_in_all[n_train:] = pids_test == pid
            tr_mask_in_all = all_mask & ~te_mask_in_all

            X_combined = X_all[tr_mask_in_all]
            y_combined = y_all[tr_mask_in_all]
            w_combined = sample_weights_all[tr_mask_in_all]
            X_te = X_all[te_mask_in_all]

            if len(X_te) == 0:
                continue

            te_indices = np.where(pids_test == pid)[0]

            scaler = StandardScaler()
            X_comb_s = scaler.fit_transform(X_combined)
            X_te_s = scaler.transform(X_te)

            n_comb = len(X_combined)
            D_te_comb = cdist(X_te_s, X_comb_s, metric='euclidean')
            all_dists = cdist(X_comb_s, X_comb_s, metric='euclidean')
            upper = all_dists[np.triu_indices(n_comb, k=1)]
            sigma = np.quantile(upper, bandwidth_quantile) if len(upper) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            for j in range(len(X_te)):
                dists = D_te_comb[j, :]
                kernel_w = np.exp(-dists ** 2 / (2 * sigma ** 2))
                # Combine kernel weights with sample weights
                combined_w = kernel_w * w_combined
                if combined_w.sum() < 1e-10:
                    test_preds[te_indices[j]] = np.mean(y_combined)
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_comb_s, y_combined, sample_weight=combined_w)
                test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

        # Update pseudo-labels for next iteration
        current_pseudo = test_preds.copy()

    return test_preds


# ================================================================
# APPROACH 2: RANDOM SUBSPACE ENSEMBLE
# ================================================================

def random_subspace_ensemble(X_train, y_train, X_test, pids_train, pids_test,
                              n_subspaces=500, n_features=5, alpha=50.0, seed=42):
    """
    Train Ridge on random feature subsets, average predictions.
    Each model sees only 5 features - low capacity, low overfit.
    500 models averaged - high stability.
    """
    rng = np.random.RandomState(seed)
    unique_pids = sorted(np.unique(pids_train))
    n_total_features = X_train.shape[1]

    all_test_preds = []
    all_oof_preds = []

    for s in range(n_subspaces):
        feat_idx = rng.choice(n_total_features, size=n_features, replace=False)
        X_tr_sub = X_train[:, feat_idx]
        X_te_sub = X_test[:, feat_idx]

        test_preds = np.zeros(len(X_test))
        oof_preds = np.zeros(len(X_train))

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            tr_indices = np.where(tr_mask)[0]
            te_indices = np.where(te_mask)[0]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_sub[tr_mask])

            # LOO for OOF
            for i in range(len(tr_indices)):
                mask = np.ones(len(tr_indices), dtype=bool)
                mask[i] = False
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s[mask], y_train[tr_mask][mask])
                oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

            # Test
            if np.any(te_mask):
                X_te_s = scaler.transform(X_te_sub[te_mask])
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_train[tr_mask])
                test_preds[te_indices] = ridge.predict(X_te_s)

        all_test_preds.append(test_preds)
        all_oof_preds.append(oof_preds)

    # Average all models
    avg_test = np.mean(all_test_preds, axis=0)
    avg_oof = np.mean(all_oof_preds, axis=0)
    return avg_oof, avg_test


# ================================================================
# APPROACH 3: LOPO MODEL SELECTION
# ================================================================

def lopo_cv(X_train, y_train, pids_train, bandwidth_quantile=0.5, alpha=10.0):
    """Leave-One-Player-Out cross-validation.
    Train on 4 players, predict 5th. Much harder than LOO."""
    unique_pids = sorted(np.unique(pids_train))
    oof_preds = np.zeros(len(X_train))

    for held_out_pid in unique_pids:
        tr_mask = pids_train != held_out_pid
        val_mask = pids_train == held_out_pid

        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_val = X_train[val_mask]
        val_indices = np.where(val_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Per-example weighted Ridge on training players
        n_tr = len(X_tr)
        D_val_tr = cdist(X_val_s, X_tr_s, metric='euclidean')
        all_dists = cdist(X_tr_s, X_tr_s, metric='euclidean')
        upper = all_dists[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(upper, bandwidth_quantile) if len(upper) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        for j in range(len(X_val)):
            dists = D_val_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                oof_preds[val_indices[j]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[val_indices[j]] = ridge.predict(X_val_s[j:j+1])[0]

    return oof_preds


def lopo_select_and_predict(X_train, y_train, X_test, pids_train, pids_test):
    """Use LOPO to select best hyperparameters, then predict test."""
    configs = [
        (0.5, 10.0), (0.5, 50.0), (0.5, 100.0), (0.5, 200.0), (0.5, 500.0),
        (0.7, 50.0), (0.7, 100.0), (0.7, 200.0),
        (0.8, 100.0), (0.8, 200.0), (0.8, 500.0),
        (0.9, 200.0), (0.9, 500.0), (0.95, 500.0), (0.95, 1000.0),
    ]

    best_mse = float('inf')
    best_config = configs[0]
    all_lopo = {}

    print("    LOPO hyperparameter search:")
    for bw, alpha in configs:
        oof = lopo_cv(X_train, y_train, pids_train,
                      bandwidth_quantile=bw, alpha=alpha)
        mse = np.mean((oof - y_train) ** 2)
        all_lopo[(bw, alpha)] = {'mse': mse, 'oof': oof}
        marker = " <-- BEST" if mse < best_mse else ""
        if mse < best_mse:
            best_mse = mse
            best_config = (bw, alpha)
        print(f"      bw={bw:.2f}, a={alpha:6.0f}: LOPO MSE={mse:.6f}{marker}")

    # Also compute LOO for the best LOPO config
    bw, alpha = best_config
    oof_loo, test_preds = per_example_lw(
        X_train, y_train, X_test, pids_train, pids_test,
        bandwidth_quantile=bw, alpha=alpha)
    loo_mse = np.mean((oof_loo - y_train) ** 2)
    print(f"    Best LOPO config: bw={bw}, alpha={alpha}")
    print(f"    LOPO MSE: {best_mse:.6f}, LOO MSE: {loo_mse:.6f}")

    # Also: average top-3 LOPO configs
    sorted_configs = sorted(all_lopo.items(), key=lambda x: x[1]['mse'])
    top3_configs = [c[0] for c in sorted_configs[:3]]
    print(f"    Top-3 LOPO configs: {top3_configs}")

    top3_tests = []
    for bw, alpha in top3_configs:
        _, t = per_example_lw(X_train, y_train, X_test, pids_train, pids_test,
                               bandwidth_quantile=bw, alpha=alpha)
        top3_tests.append(t)
    avg_top3_test = np.mean(top3_tests, axis=0)

    return test_preds, avg_top3_test, best_config, best_mse


# ================================================================
# APPROACH 4: BAGGED PREDICTIONS
# ================================================================

def bagged_predictions(X_train, y_train, X_test, pids_train, pids_test,
                       n_bags=200, sample_fraction=0.80,
                       bandwidth_quantile=0.5, alpha=50.0, seed=42):
    """Bootstrap training data, predict test, average.
    Each bag uses 80% of training data -> different model each time.
    Average of 200 models reduces variance."""
    rng = np.random.RandomState(seed)
    unique_pids = sorted(np.unique(pids_train))
    n_test = len(X_test)

    all_preds = []

    for b in range(n_bags):
        test_preds = np.zeros(n_test)

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            X_tr = X_train[tr_mask]
            y_tr = y_train[tr_mask]
            X_te = X_test[te_mask]
            te_indices = np.where(te_mask)[0]
            n_tr = len(X_tr)

            if len(X_te) == 0:
                continue

            # Bootstrap sample
            n_sample = max(int(n_tr * sample_fraction), 5)
            sample_idx = rng.choice(n_tr, size=n_sample, replace=True)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr[sample_idx])
            X_te_s = scaler.transform(X_te)
            y_tr_sample = y_tr[sample_idx]

            n_s = len(X_tr_s)
            D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean')
            all_dists = cdist(X_tr_s, X_tr_s, metric='euclidean')
            upper = all_dists[np.triu_indices(n_s, k=1)]
            sigma = np.quantile(upper, bandwidth_quantile) if len(upper) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            for j in range(len(X_te)):
                dists = D_te_tr[j, :]
                weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
                if weights.sum() < 1e-10:
                    test_preds[te_indices[j]] = np.mean(y_tr_sample)
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr_sample, sample_weight=weights)
                test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

        all_preds.append(test_preds)

    avg_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    return avg_preds, std_preds


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("STRUCTURAL OVERFITTING ATTACK")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scalers = {}
    y_scaled = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled[target] = scalers[target].transform(
            y_train[:, TARGET_IDX[target]].reshape(-1, 1)).ravel()

    sub_1828 = pd.read_csv(SUBMISSION_DIR / "submission_1828.csv")

    # Store all approach results
    approach_results = {}  # approach_name -> target -> test_preds

    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        y_t = y_scaled[target]
        y_raw = y_train[:, TARGET_IDX[target]]
        col = f'scaled_{target}'

        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {frame})")
        print(f"{'=' * 70}")

        # Extract features
        X_tr_hc = extract_all_features(train_data, frame)
        X_te_hc = extract_all_features(test_data, frame)
        X_tr_aug, X_te_aug = augment_with_pls(
            X_tr_hc, y_raw, pids_train, X_te_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Features: {X_tr_aug.shape[1]}")

        # Baseline LOO
        oof_base, test_base = per_example_lw(X_tr_aug, y_t, X_te_aug, pids_train, pids_test,
                                              bandwidth_quantile=0.5, alpha=10.0)
        base_mse = np.mean((oof_base - y_t) ** 2)
        print(f"  Baseline LOO MSE: {base_mse:.6f}")

        # ---- APPROACH 1: PSEUDO-LABELING ----
        print(f"\n  [1] PSEUDO-LABELING")
        pseudo_labels_init = sub_1828[col].values

        for pw, n_iter in [(0.3, 2), (0.5, 2), (0.5, 3), (0.7, 3), (0.8, 2)]:
            test_pl = pseudo_labeling(
                X_tr_aug, y_t, X_te_aug, pids_train, pids_test,
                pseudo_labels_init, pseudo_weight=pw, n_iterations=n_iter,
                bandwidth_quantile=0.5, alpha=50.0)
            r_1828 = np.corrcoef(test_pl, sub_1828[col].values)[0, 1]
            key = f'pseudo_w{int(pw*10)}_i{n_iter}'
            if key not in approach_results:
                approach_results[key] = {}
            approach_results[key][target] = test_pl
            print(f"    pw={pw:.1f}, iter={n_iter}: r(1828)={r_1828:.4f}")

        # ---- APPROACH 2: RANDOM SUBSPACE ENSEMBLE ----
        print(f"\n  [2] RANDOM SUBSPACE ENSEMBLE")
        for n_feat, alpha in [(5, 50.0), (5, 100.0), (10, 50.0), (10, 100.0)]:
            oof_rs, test_rs = random_subspace_ensemble(
                X_tr_aug, y_t, X_te_aug, pids_train, pids_test,
                n_subspaces=500, n_features=n_feat, alpha=alpha)
            mse_rs = np.mean((oof_rs - y_t) ** 2)
            r_1828 = np.corrcoef(test_rs, sub_1828[col].values)[0, 1]
            key = f'randsubspace_f{n_feat}_a{int(alpha)}'
            if key not in approach_results:
                approach_results[key] = {}
            approach_results[key][target] = test_rs
            print(f"    n_feat={n_feat}, a={alpha:5.0f}: LOO={mse_rs:.6f}, r(1828)={r_1828:.4f}")

        # ---- APPROACH 3: LOPO MODEL SELECTION ----
        print(f"\n  [3] LOPO MODEL SELECTION")
        test_lopo_best, test_lopo_avg3, best_cfg, lopo_mse = lopo_select_and_predict(
            X_tr_aug, y_t, X_te_aug, pids_train, pids_test)
        r_best = np.corrcoef(test_lopo_best, sub_1828[col].values)[0, 1]
        r_avg3 = np.corrcoef(test_lopo_avg3, sub_1828[col].values)[0, 1]

        if 'lopo_best' not in approach_results:
            approach_results['lopo_best'] = {}
        approach_results['lopo_best'][target] = test_lopo_best
        if 'lopo_avg3' not in approach_results:
            approach_results['lopo_avg3'] = {}
        approach_results['lopo_avg3'][target] = test_lopo_avg3
        print(f"    Best LOPO: r(1828)={r_best:.4f}")
        print(f"    Avg top-3: r(1828)={r_avg3:.4f}")

        # ---- APPROACH 4: BAGGED PREDICTIONS ----
        print(f"\n  [4] BAGGED PREDICTIONS")
        for bw, alpha in [(0.5, 50.0), (0.7, 50.0), (0.7, 100.0)]:
            test_bag, std_bag = bagged_predictions(
                X_tr_aug, y_t, X_te_aug, pids_train, pids_test,
                n_bags=200, sample_fraction=0.80,
                bandwidth_quantile=bw, alpha=alpha)
            r_1828 = np.corrcoef(test_bag, sub_1828[col].values)[0, 1]
            key = f'bagged_bw{int(bw*10)}_a{int(alpha)}'
            if key not in approach_results:
                approach_results[key] = {}
            approach_results[key][target] = test_bag
            print(f"    bw={bw:.1f}, a={alpha:5.0f}: r(1828)={r_1828:.4f}, pred_std={std_bag.mean():.6f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # For each approach, generate blended submissions
    for approach_name, target_preds in approach_results.items():
        if len(target_preds) != 3:
            continue  # Skip incomplete approaches

        # Compute diversity
        diversities = []
        for target in TARGETS:
            col = f'scaled_{target}'
            r = np.corrcoef(target_preds[target], sub_1828[col].values)[0, 1]
            diversities.append(r)
        avg_r = np.mean(diversities)

        # Blend at multiple weights
        for w in [0.20, 0.50, 1.00]:
            sub_num = get_next_submission_number()
            blended = sub_1828.copy()
            for target in TARGETS:
                col = f'scaled_{target}'
                blended[col] = (1 - w) * sub_1828[col] + w * target_preds[target]
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            if w == 0.50:
                print(f"  Sub {sub_num}: {approach_name} w={w:.2f} (avg_r={avg_r:.4f})")

    # Best combination: per-target best approach
    # Pick approach with lowest avg_r (most diverse) per target
    print(f"\n  Per-target best diverse approach:")
    best_diverse = {}
    for target in TARGETS:
        col = f'scaled_{target}'
        best_r = 1.0
        best_approach = None
        for approach_name, target_preds in approach_results.items():
            if target not in target_preds:
                continue
            r = np.corrcoef(target_preds[target], sub_1828[col].values)[0, 1]
            if r < best_r:
                best_r = r
                best_approach = approach_name
        best_diverse[target] = (best_approach, best_r)
        print(f"    {target}: {best_approach} (r={best_r:.4f})")

    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_1828.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            approach_name = best_diverse[target][0]
            blended[col] = (1 - w) * sub_1828[col] + w * approach_results[approach_name][target]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: per-target best diverse w={w:.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
