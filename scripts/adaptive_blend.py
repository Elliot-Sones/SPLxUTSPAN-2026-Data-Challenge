"""
Adaptive Per-Shot Blending

Instead of a global blend weight (e.g., dw=0.30, lw=0.50) applied uniformly
to all 113 test shots, each shot gets its own blend weight based on how
trustworthy the per-example prediction is for that specific shot.

Confidence signals:
1. Effective Sample Size (ESS): how many training shots meaningfully
   contribute to this prediction. High ESS = well-supported prediction.
2. Local LOO error: average squared error of the per-example model on
   the K nearest training neighbors. Low error = model works well locally.
3. Prediction agreement: variance across multiple model configs for this
   shot. Low variance = high confidence.
4. Distance to nearest neighbor: close = interpolation, far = extrapolation.

High-confidence shots get blend weights ABOVE the global optimum.
Low-confidence shots fall back MORE to Sub 784.
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
# DATA LOADING + FEATURE EXTRACTION (inlined from V1)
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
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


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
# LOCALLY WEIGHTED REGRESSION WITH CONFIDENCE METADATA
# ==============================================================

def locally_weighted_with_confidence(X_train, y_train, X_test, pids_train, pids_test,
                                     bandwidth_quantile=0.5, alpha=10.0):
    """Run locally weighted regression and return confidence metadata for each test shot."""
    unique_pids = sorted(np.unique(pids_train))
    n_train = len(X_train)
    n_test = len(X_test)

    test_preds = np.zeros(n_test)
    oof_preds = np.zeros(n_train)

    # Per-shot confidence metadata
    test_meta = {
        'ess': np.zeros(n_test),           # effective sample size
        'local_loo_error': np.zeros(n_test),  # avg LOO error of K nearest neighbors
        'min_dist': np.zeros(n_test),       # distance to nearest neighbor
        'mean_k_dist': np.zeros(n_test),    # mean distance to K neighbors
        'local_bias': np.zeros(n_test),     # mean signed LOO error of neighbors
        'local_std': np.zeros(n_test),      # std of LOO errors of neighbors
    }

    K_NEIGHBORS = 10

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

        # LOO for training
        loo_errors = np.zeros(n_tr)
        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                loo_errors[i] = oof_preds[tr_indices[i]] - y_tr[i]
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]
            loo_errors[i] = oof_preds[tr_indices[i]] - y_tr[i]

        # Test predictions + confidence metadata
        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

            # Effective sample size
            w_norm = weights / weights.sum()
            ess = 1.0 / np.sum(w_norm ** 2)
            test_meta['ess'][te_indices[j]] = ess

            # K nearest neighbors
            k = min(K_NEIGHBORS, n_tr)
            nn_idx = np.argsort(dists)[:k]
            nn_errors = loo_errors[nn_idx]
            nn_sq_errors = nn_errors ** 2

            # Distance-weighted average of neighbor errors
            nn_dists = dists[nn_idx]
            nn_weights = 1.0 / (nn_dists + 1e-6)
            nn_weights /= nn_weights.sum()

            test_meta['local_loo_error'][te_indices[j]] = np.dot(nn_weights, nn_sq_errors)
            test_meta['local_bias'][te_indices[j]] = np.dot(nn_weights, nn_errors)
            test_meta['local_std'][te_indices[j]] = np.sqrt(np.dot(nn_weights, nn_sq_errors))
            test_meta['min_dist'][te_indices[j]] = nn_dists[0]
            test_meta['mean_k_dist'][te_indices[j]] = np.mean(nn_dists)

    return oof_preds, test_preds, test_meta


# ==============================================================
# MULTI-CONFIG PREDICTIONS FOR AGREEMENT SCORE
# ==============================================================

def run_multi_config(X_train, y_train, X_test, pids_train, pids_test):
    """Run multiple locally weighted regression configs and return all predictions."""
    configs = [
        (0.4, 5.0), (0.4, 10.0),
        (0.5, 5.0), (0.5, 10.0),
        (0.6, 5.0), (0.6, 10.0),
    ]
    all_test_preds = []
    all_oof_preds = []
    config_mses = []

    for bw, alpha in configs:
        oof, test_pred, _ = locally_weighted_with_confidence(
            X_train, y_train, X_test, pids_train, pids_test,
            bandwidth_quantile=bw, alpha=alpha)
        all_test_preds.append(test_pred)
        all_oof_preds.append(oof)
        mse = np.mean((oof - y_train) ** 2)
        config_mses.append(mse)
        print(f"    bw={bw:.1f} a={alpha:.0f}: MSE={mse:.6f}")

    return np.array(all_test_preds), np.array(all_oof_preds), configs, config_mses


# ==============================================================
# ADAPTIVE BLEND WEIGHT STRATEGIES
# ==============================================================

def compute_adaptive_weights(test_meta, all_test_preds, w_global, strategy='combined'):
    """Compute per-shot blend weights.

    w_global: the optimal global weight (e.g., 0.30 for depth)
    Returns: array of per-shot weights, same length as test set.
    """
    n_test = len(test_meta['ess'])

    if strategy == 'ess':
        # Effective sample size: higher ESS = more confidence
        ess = test_meta['ess']
        # Normalize ESS to [0, 1] range
        ess_norm = (ess - ess.min()) / (ess.max() - ess.min() + 1e-10)
        # Map to weight range: [w_global*0.5, w_global*1.5]
        weights = w_global * (0.5 + ess_norm)

    elif strategy == 'local_error':
        # Low local LOO error = high confidence
        err = test_meta['local_loo_error']
        # Normalize: high error -> low weight
        err_norm = (err - err.min()) / (err.max() - err.min() + 1e-10)
        # Map: low error -> higher weight
        weights = w_global * (1.5 - err_norm)

    elif strategy == 'agreement':
        # Prediction variance across configs: low variance = high confidence
        pred_std = np.std(all_test_preds, axis=0)
        # Normalize
        std_norm = (pred_std - pred_std.min()) / (pred_std.max() - pred_std.min() + 1e-10)
        # Low variance -> higher weight
        weights = w_global * (1.5 - std_norm)

    elif strategy == 'distance':
        # Close to nearest neighbor = interpolation = high confidence
        min_dist = test_meta['min_dist']
        dist_norm = (min_dist - min_dist.min()) / (min_dist.max() - min_dist.min() + 1e-10)
        # Close -> higher weight
        weights = w_global * (1.5 - dist_norm)

    elif strategy == 'combined':
        # Combine multiple signals
        ess = test_meta['ess']
        ess_norm = (ess - ess.min()) / (ess.max() - ess.min() + 1e-10)

        err = test_meta['local_loo_error']
        err_norm = (err - err.min()) / (err.max() - err.min() + 1e-10)

        pred_std = np.std(all_test_preds, axis=0)
        std_norm = (pred_std - pred_std.min()) / (pred_std.max() - pred_std.min() + 1e-10)

        min_dist = test_meta['min_dist']
        dist_norm = (min_dist - min_dist.min()) / (min_dist.max() - min_dist.min() + 1e-10)

        # Combined confidence: average of 4 normalized signals
        confidence = 0.25 * ess_norm + 0.25 * (1 - err_norm) + 0.25 * (1 - std_norm) + 0.25 * (1 - dist_norm)
        weights = w_global * (0.5 + confidence)

    else:
        weights = np.full(n_test, w_global)

    # Clip to [0, min(2*w_global, 0.95)]
    weights = np.clip(weights, 0.0, min(2 * w_global, 0.95))
    return weights


def compute_bias_corrected_preds(test_preds, test_meta, damping=0.5):
    """Apply local bias correction to predictions."""
    return test_preds - damping * test_meta['local_bias']


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ADAPTIVE PER-SHOT BLENDING")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    target_scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = target_scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Global optimal weights (from LB testing)
    GLOBAL_WEIGHTS = {'angle': 0.00, 'depth': 0.30, 'left_right': 0.50}

    # Store per-target results
    target_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # Extract features
        print("  Extracting features...")
        X_train_hc = extract_all_features(train_data, target)
        X_test_hc = extract_all_features(test_data, target)

        # Add PLS
        print("  Adding PLS components...")
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        print(f"  Features: {X_train_aug.shape[1]}")

        y_target = y_scaled[target]

        # Run multi-config locally weighted regression
        print("\n  Running multi-config search...")
        all_test_preds, all_oof_preds, configs, config_mses = run_multi_config(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)

        # Best config with full metadata
        best_idx = np.argmin(config_mses)
        best_bw, best_alpha = configs[best_idx]
        print(f"\n  Best config: bw={best_bw:.1f} a={best_alpha:.0f} MSE={config_mses[best_idx]:.6f}")

        print(f"  Getting confidence metadata for best config...")
        oof_best, test_best, test_meta = locally_weighted_with_confidence(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=best_bw, alpha=best_alpha)

        # Config-averaged predictions (ensemble of top 3)
        top3 = np.argsort(config_mses)[:3]
        test_ensemble = np.mean(all_test_preds[top3], axis=0)

        # Confidence metadata stats
        print(f"\n  Confidence metadata (test set):")
        print(f"    ESS:         min={test_meta['ess'].min():.1f} "
              f"median={np.median(test_meta['ess']):.1f} max={test_meta['ess'].max():.1f}")
        print(f"    Local error: min={test_meta['local_loo_error'].min():.6f} "
              f"median={np.median(test_meta['local_loo_error']):.6f} max={test_meta['local_loo_error'].max():.6f}")
        print(f"    Pred std:    min={np.std(all_test_preds, axis=0).min():.6f} "
              f"median={np.median(np.std(all_test_preds, axis=0)):.6f} max={np.std(all_test_preds, axis=0).max():.6f}")
        print(f"    Min dist:    min={test_meta['min_dist'].min():.3f} "
              f"median={np.median(test_meta['min_dist']):.3f} max={test_meta['min_dist'].max():.3f}")
        print(f"    Local bias:  mean={test_meta['local_bias'].mean():.6f} "
              f"std={test_meta['local_bias'].std():.6f}")

        # Bias-corrected predictions
        test_bc = compute_bias_corrected_preds(test_best, test_meta, damping=0.3)
        test_bc_strong = compute_bias_corrected_preds(test_best, test_meta, damping=0.5)

        target_results[target] = {
            'test_best': test_best,
            'test_ensemble': test_ensemble,
            'test_bc': test_bc,
            'test_bc_strong': test_bc_strong,
            'test_meta': test_meta,
            'all_test_preds': all_test_preds,
            'best_bw': best_bw,
            'best_alpha': best_alpha,
        }

    # ==============================================================
    # GENERATE SUBMISSIONS
    # ==============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    strategies = ['ess', 'local_error', 'agreement', 'distance', 'combined']

    # 1. Adaptive blend submissions
    print("\n  --- Adaptive Blend Submissions ---")
    for strategy in strategies:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()

        weight_summary = {}
        for target in TARGETS:
            col = f'scaled_{target}'
            w_global = GLOBAL_WEIGHTS[target]

            if w_global == 0:
                continue

            w_adaptive = compute_adaptive_weights(
                target_results[target]['test_meta'],
                target_results[target]['all_test_preds'],
                w_global, strategy=strategy)

            perex_pred = target_results[target]['test_best']
            base_pred = sub_784[col].values

            blended[col] = (1 - w_adaptive) * base_pred + w_adaptive * perex_pred
            weight_summary[target] = {
                'mean': w_adaptive.mean(),
                'min': w_adaptive.min(),
                'max': w_adaptive.max(),
                'std': w_adaptive.std(),
            }

        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: adaptive_{strategy}")
        for t, ws in weight_summary.items():
            print(f"    {t}: w_mean={ws['mean']:.3f} w_min={ws['min']:.3f} "
                  f"w_max={ws['max']:.3f} w_std={ws['std']:.3f}")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # 2. Bias-corrected submissions
    print("\n  --- Bias-Corrected Submissions ---")
    for bc_name, bc_key in [('bc_light', 'test_bc'), ('bc_strong', 'test_bc_strong')]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            w = GLOBAL_WEIGHTS[target]
            if w == 0:
                continue
            blended[col] = (1-w) * sub_784[col].values + w * target_results[target][bc_key]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        print(f"  Sub {sub_num}: {bc_name} (dw=0.30, lw=0.50)")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # 3. Ensemble + adaptive blend
    print("\n  --- Ensemble + Adaptive Blend ---")
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    for target in TARGETS:
        col = f'scaled_{target}'
        w = GLOBAL_WEIGHTS[target]
        if w == 0:
            continue
        w_adaptive = compute_adaptive_weights(
            target_results[target]['test_meta'],
            target_results[target]['all_test_preds'],
            w, strategy='combined')
        blended[col] = (1 - w_adaptive) * sub_784[col].values + w_adaptive * target_results[target]['test_ensemble']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    a_std = blended['scaled_angle'].std()
    d_mean = blended['scaled_depth'].mean()
    print(f"  Sub {sub_num}: ensemble + adaptive_combined")
    print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # 4. Bias-corrected + adaptive
    print("\n  --- Bias-Corrected + Adaptive ---")
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    for target in TARGETS:
        col = f'scaled_{target}'
        w = GLOBAL_WEIGHTS[target]
        if w == 0:
            continue
        w_adaptive = compute_adaptive_weights(
            target_results[target]['test_meta'],
            target_results[target]['all_test_preds'],
            w, strategy='combined')
        blended[col] = (1 - w_adaptive) * sub_784[col].values + w_adaptive * target_results[target]['test_bc']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    a_std = blended['scaled_angle'].std()
    d_mean = blended['scaled_depth'].mean()
    print(f"  Sub {sub_num}: bc_light + adaptive_combined")
    print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # 5. Global uniform (baseline for comparison)
    print("\n  --- Global Uniform (Baseline) ---")
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    for target in TARGETS:
        col = f'scaled_{target}'
        w = GLOBAL_WEIGHTS[target]
        if w == 0:
            continue
        blended[col] = (1-w) * sub_784[col].values + w * target_results[target]['test_best']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    a_std = blended['scaled_angle'].std()
    d_mean = blended['scaled_depth'].mean()
    print(f"  Sub {sub_num}: global_uniform (dw=0.30, lw=0.50) [BASELINE]")
    print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # 6. Correlation analysis: adaptive vs uniform predictions
    print(f"\n{'=' * 70}")
    print("CORRELATION ANALYSIS")
    print(f"{'=' * 70}")
    for target in ['depth', 'left_right']:
        w = GLOBAL_WEIGHTS[target]
        uniform_pred = (1-w) * sub_784[f'scaled_{target}'].values + w * target_results[target]['test_best']
        for strategy in strategies:
            w_adaptive = compute_adaptive_weights(
                target_results[target]['test_meta'],
                target_results[target]['all_test_preds'],
                w, strategy=strategy)
            adaptive_pred = (1 - w_adaptive) * sub_784[f'scaled_{target}'].values + w_adaptive * target_results[target]['test_best']
            r = np.corrcoef(uniform_pred, adaptive_pred)[0, 1]
            diff = np.mean((uniform_pred - adaptive_pred) ** 2)
            print(f"  {target} {strategy}: r={r:.6f} mean_sq_diff={diff:.8f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
