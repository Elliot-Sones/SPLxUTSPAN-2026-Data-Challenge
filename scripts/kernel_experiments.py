"""
Kernel Experiments for Per-Example Locally Weighted Ridge Regression

Tests alternative kernel functions (Laplacian, Tricube, Epanechnikov, Cauchy)
against the baseline Gaussian kernel. Sweeps bandwidth and alpha. Computes
effective number of neighbors. Generates submissions.

Based on per_example_pipeline.py core functions.
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
# KERNEL FUNCTIONS
# ==============================================================

def gaussian_kernel(distances, sigma):
    """Standard Gaussian (RBF) kernel: exp(-d^2 / (2*sigma^2))"""
    return np.exp(-distances ** 2 / (2 * sigma ** 2))


def laplacian_kernel(distances, sigma):
    """Laplacian kernel: exp(-d / sigma). Heavier tails than Gaussian."""
    return np.exp(-distances / sigma)


def tricube_kernel(distances, sigma):
    """Tricube kernel: (1 - (d/sigma)^3)^3 for d < sigma, else 0. Compact support."""
    u = distances / sigma
    weights = np.where(u < 1.0, (1 - u ** 3) ** 3, 0.0)
    return weights


def epanechnikov_kernel(distances, sigma):
    """Epanechnikov kernel: max(0, 1 - (d/sigma)^2). Compact support, optimal for mean."""
    u = distances / sigma
    weights = np.where(u < 1.0, 1 - u ** 2, 0.0)
    return weights


def cauchy_kernel(distances, sigma):
    """Cauchy kernel: 1 / (1 + (d/sigma)^2). Very heavy tails."""
    return 1.0 / (1.0 + (distances / sigma) ** 2)


KERNEL_FUNCTIONS = {
    'gaussian': gaussian_kernel,
    'laplacian': laplacian_kernel,
    'tricube': tricube_kernel,
    'epanechnikov': epanechnikov_kernel,
    'cauchy': cauchy_kernel,
}


def effective_neighbors(weights):
    """Compute effective number of neighbors: (sum w)^2 / sum(w^2).
    This is the 'participation ratio' - equals N for uniform weights, 1 for delta."""
    s = np.sum(weights)
    s2 = np.sum(weights ** 2)
    if s2 < 1e-15:
        return 0.0
    return (s ** 2) / s2


# ==============================================================
# DATA LOADING (copied from per_example_pipeline.py)
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
# CORE: Locally Weighted Prediction with Configurable Kernel
# ==============================================================

def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                kernel_name='gaussian', bandwidth_quantile=0.3,
                                alpha=10.0, return_eff_neighbors=False):
    """Per-example locally weighted Ridge with configurable kernel.

    Returns: (oof_preds, test_preds) or (oof_preds, test_preds, mean_eff_neighbors)
    """
    kernel_fn = KERNEL_FUNCTIONS[kernel_name]
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    all_eff_neighbors = []

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

        # OOF: leave-one-out
        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = kernel_fn(dists, sigma)
            weights[i] = 0  # Leave out self

            if return_eff_neighbors:
                all_eff_neighbors.append(effective_neighbors(weights))

            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test predictions
        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = kernel_fn(dists, sigma)

            if return_eff_neighbors:
                all_eff_neighbors.append(effective_neighbors(weights))

            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue

            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    if return_eff_neighbors:
        return oof_preds, test_preds, np.mean(all_eff_neighbors) if all_eff_neighbors else 0.0
    return oof_preds, test_preds


# ==============================================================
# MAIN EXPERIMENT
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("KERNEL EXPERIMENTS FOR LOCALLY WEIGHTED RIDGE")
    print("=" * 70)

    # Load data
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

    # Extract features for all targets
    print("\nExtracting features for all targets...")
    features = {}
    for target in TARGETS:
        print(f"  {target} (frame {TARGET_FRAMES[target]})...")
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        features[target] = (X_train_aug, X_test_aug)
        print(f"    Features: {X_train_aug.shape[1]}")

    feat_time = time.time() - t0
    print(f"\nFeature extraction: {feat_time:.1f}s")

    # Load reference submissions
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_2063 = pd.read_csv(SUBMISSION_DIR / "submission_2063.csv")

    # ============================================================
    # PHASE 1: Gaussian baseline
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: GAUSSIAN BASELINE (bw=0.3, alpha=10)")
    print(f"{'=' * 70}")

    baseline = {}
    for target in TARGETS:
        X_tr, X_te = features[target]
        oof, test, eff_n = locally_weighted_prediction(
            X_tr, y_scaled[target], X_te, pids_train, pids_test,
            kernel_name='gaussian', bandwidth_quantile=0.3, alpha=10.0,
            return_eff_neighbors=True)
        mse = np.mean((oof - y_scaled[target]) ** 2)
        baseline[target] = {'oof': oof, 'test': test, 'mse': mse, 'eff_n': eff_n}
        print(f"  {target}: LOO MSE={mse:.6f}, eff_neighbors={eff_n:.1f}")

    baseline_mean = np.mean([baseline[t]['mse'] for t in TARGETS])
    print(f"  MEAN LOO MSE: {baseline_mean:.6f}")

    # ============================================================
    # PHASE 2: Laplacian kernel sweep
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: LAPLACIAN KERNEL SWEEP (alpha=10)")
    print(f"{'=' * 70}")

    bw_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    laplacian_results = {}

    for bw in bw_values:
        row = {}
        for target in TARGETS:
            X_tr, X_te = features[target]
            oof, test, eff_n = locally_weighted_prediction(
                X_tr, y_scaled[target], X_te, pids_train, pids_test,
                kernel_name='laplacian', bandwidth_quantile=bw, alpha=10.0,
                return_eff_neighbors=True)
            mse = np.mean((oof - y_scaled[target]) ** 2)
            row[target] = {'oof': oof, 'test': test, 'mse': mse, 'eff_n': eff_n}
        row['mean_mse'] = np.mean([row[t]['mse'] for t in TARGETS])
        row['mean_eff_n'] = np.mean([row[t]['eff_n'] for t in TARGETS])
        laplacian_results[bw] = row

        delta = (row['mean_mse'] - baseline_mean) / baseline_mean * 100
        print(f"  bw={bw:.1f}: mean_MSE={row['mean_mse']:.6f} ({delta:+.2f}%), "
              f"eff_n={row['mean_eff_n']:.1f} "
              f"[a={row['angle']['mse']:.6f} d={row['depth']['mse']:.6f} lr={row['left_right']['mse']:.6f}]")

    # Find best Laplacian bandwidth per target and overall
    best_lap_bw = min(laplacian_results, key=lambda b: laplacian_results[b]['mean_mse'])
    print(f"\n  Best Laplacian bandwidth (overall): {best_lap_bw}")
    for target in TARGETS:
        best_t_bw = min(laplacian_results, key=lambda b: laplacian_results[b][target]['mse'])
        print(f"    {target}: best bw={best_t_bw} (MSE={laplacian_results[best_t_bw][target]['mse']:.6f})")

    # ============================================================
    # PHASE 3: Other kernels at bw=0.3 and at their optimal bw
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 3: OTHER KERNELS (Tricube, Epanechnikov, Cauchy)")
    print(f"{'=' * 70}")

    other_kernels = ['tricube', 'epanechnikov', 'cauchy']
    all_kernel_results = {'gaussian': {0.3: {t: baseline[t] for t in TARGETS}}}
    all_kernel_results['gaussian'][0.3]['mean_mse'] = baseline_mean
    all_kernel_results['laplacian'] = laplacian_results

    for kern in other_kernels:
        print(f"\n  --- {kern.upper()} ---")
        kernel_bw_results = {}
        for bw in bw_values:
            row = {}
            for target in TARGETS:
                X_tr, X_te = features[target]
                oof, test, eff_n = locally_weighted_prediction(
                    X_tr, y_scaled[target], X_te, pids_train, pids_test,
                    kernel_name=kern, bandwidth_quantile=bw, alpha=10.0,
                    return_eff_neighbors=True)
                mse = np.mean((oof - y_scaled[target]) ** 2)
                row[target] = {'oof': oof, 'test': test, 'mse': mse, 'eff_n': eff_n}
            row['mean_mse'] = np.mean([row[t]['mse'] for t in TARGETS])
            row['mean_eff_n'] = np.mean([row[t]['eff_n'] for t in TARGETS])
            kernel_bw_results[bw] = row

            delta = (row['mean_mse'] - baseline_mean) / baseline_mean * 100
            print(f"    bw={bw:.1f}: mean_MSE={row['mean_mse']:.6f} ({delta:+.2f}%), "
                  f"eff_n={row['mean_eff_n']:.1f}")

        all_kernel_results[kern] = kernel_bw_results
        best_bw = min(kernel_bw_results, key=lambda b: kernel_bw_results[b]['mean_mse'])
        print(f"    Best bw: {best_bw} (MSE={kernel_bw_results[best_bw]['mean_mse']:.6f})")

    # ============================================================
    # SUMMARY TABLE: Best config per kernel
    # ============================================================
    print(f"\n{'=' * 70}")
    print("KERNEL COMPARISON SUMMARY (best bandwidth per kernel)")
    print(f"{'=' * 70}")
    print(f"  {'Kernel':<15} {'BW':>4} {'Mean MSE':>10} {'Delta%':>8} {'Eff N':>6} "
          f"{'Angle':>10} {'Depth':>10} {'LR':>10}")
    print(f"  {'-'*13:<15} {'--':>4} {'-'*8:>10} {'-'*6:>8} {'-'*4:>6} "
          f"{'-'*8:>10} {'-'*8:>10} {'-'*8:>10}")

    best_overall = None
    best_overall_mse = float('inf')

    for kern in ['gaussian', 'laplacian', 'tricube', 'epanechnikov', 'cauchy']:
        bw_results = all_kernel_results[kern]
        best_bw = min(bw_results, key=lambda b: bw_results[b]['mean_mse'])
        r = bw_results[best_bw]
        mean_mse = r['mean_mse']
        delta = (mean_mse - baseline_mean) / baseline_mean * 100
        eff_n = r.get('mean_eff_n', np.mean([r[t].get('eff_n', 0) for t in TARGETS]))
        print(f"  {kern:<15} {best_bw:>4.1f} {mean_mse:>10.6f} {delta:>+8.2f} {eff_n:>6.1f} "
              f"{r['angle']['mse']:>10.6f} {r['depth']['mse']:>10.6f} {r['left_right']['mse']:>10.6f}")

        if mean_mse < best_overall_mse:
            best_overall_mse = mean_mse
            best_overall = (kern, best_bw)

    print(f"\n  Best overall: {best_overall[0]} bw={best_overall[1]}")

    # ============================================================
    # PHASE 4: Alpha interaction for best kernel
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"PHASE 4: ALPHA SWEEP FOR BEST KERNEL ({best_overall[0].upper()}, bw={best_overall[1]})")
    print(f"{'=' * 70}")

    alpha_values = [1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0]
    alpha_results = {}

    for alpha in alpha_values:
        row = {}
        for target in TARGETS:
            X_tr, X_te = features[target]
            oof, test, eff_n = locally_weighted_prediction(
                X_tr, y_scaled[target], X_te, pids_train, pids_test,
                kernel_name=best_overall[0], bandwidth_quantile=best_overall[1],
                alpha=alpha, return_eff_neighbors=True)
            mse = np.mean((oof - y_scaled[target]) ** 2)
            row[target] = {'oof': oof, 'test': test, 'mse': mse, 'eff_n': eff_n}
        row['mean_mse'] = np.mean([row[t]['mse'] for t in TARGETS])
        alpha_results[alpha] = row

        delta = (row['mean_mse'] - baseline_mean) / baseline_mean * 100
        print(f"  alpha={alpha:>6.1f}: mean_MSE={row['mean_mse']:.6f} ({delta:+.2f}%) "
              f"[a={row['angle']['mse']:.6f} d={row['depth']['mse']:.6f} lr={row['left_right']['mse']:.6f}]")

    best_alpha = min(alpha_results, key=lambda a: alpha_results[a]['mean_mse'])
    print(f"\n  Best alpha: {best_alpha}")

    # Also sweep alpha for Laplacian (if it's not the best overall)
    if best_overall[0] != 'laplacian':
        print(f"\n  Also sweeping alpha for Laplacian (bw={best_lap_bw})...")
        lap_alpha_results = {}
        for alpha in alpha_values:
            row = {}
            for target in TARGETS:
                X_tr, X_te = features[target]
                oof, test, eff_n = locally_weighted_prediction(
                    X_tr, y_scaled[target], X_te, pids_train, pids_test,
                    kernel_name='laplacian', bandwidth_quantile=best_lap_bw,
                    alpha=alpha, return_eff_neighbors=True)
                mse = np.mean((oof - y_scaled[target]) ** 2)
                row[target] = {'oof': oof, 'test': test, 'mse': mse, 'eff_n': eff_n}
            row['mean_mse'] = np.mean([row[t]['mse'] for t in TARGETS])
            lap_alpha_results[alpha] = row
            delta = (row['mean_mse'] - baseline_mean) / baseline_mean * 100
            print(f"    alpha={alpha:>6.1f}: mean_MSE={row['mean_mse']:.6f} ({delta:+.2f}%)")
        best_lap_alpha = min(lap_alpha_results, key=lambda a: lap_alpha_results[a]['mean_mse'])
        print(f"    Best Laplacian alpha: {best_lap_alpha}")

    # ============================================================
    # PHASE 5: Per-target optimal kernel+bw+alpha
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 5: PER-TARGET OPTIMAL KERNEL+BW+ALPHA")
    print(f"{'=' * 70}")

    # Collect all results into a flat list for each target
    per_target_best = {}
    for target in TARGETS:
        best_config = None
        best_mse_t = float('inf')

        # Check all kernel x bw combos at alpha=10
        for kern in ['gaussian', 'laplacian', 'tricube', 'epanechnikov', 'cauchy']:
            for bw in all_kernel_results.get(kern, {}):
                r = all_kernel_results[kern][bw]
                if target in r and r[target]['mse'] < best_mse_t:
                    best_mse_t = r[target]['mse']
                    best_config = (kern, bw, 10.0)

        # Check alpha sweep for best overall kernel
        for alpha in alpha_results:
            if alpha_results[alpha][target]['mse'] < best_mse_t:
                best_mse_t = alpha_results[alpha][target]['mse']
                best_config = (best_overall[0], best_overall[1], alpha)

        # Check Laplacian alpha sweep if it exists
        if best_overall[0] != 'laplacian':
            for alpha in lap_alpha_results:
                if lap_alpha_results[alpha][target]['mse'] < best_mse_t:
                    best_mse_t = lap_alpha_results[alpha][target]['mse']
                    best_config = ('laplacian', best_lap_bw, alpha)

        per_target_best[target] = (best_config, best_mse_t)
        delta = (best_mse_t - baseline[target]['mse']) / baseline[target]['mse'] * 100
        print(f"  {target}: {best_config[0]} bw={best_config[1]} alpha={best_config[2]} "
              f"MSE={best_mse_t:.6f} ({delta:+.2f}% vs Gaussian baseline)")

    # ============================================================
    # PHASE 5b: Run per-target optimal configs to get predictions
    # ============================================================
    print(f"\n  Generating per-target optimal predictions...")
    per_target_optimal_preds = {}
    for target in TARGETS:
        cfg = per_target_best[target][0]
        X_tr, X_te = features[target]
        oof, test, eff_n = locally_weighted_prediction(
            X_tr, y_scaled[target], X_te, pids_train, pids_test,
            kernel_name=cfg[0], bandwidth_quantile=cfg[1], alpha=cfg[2],
            return_eff_neighbors=True)
        per_target_optimal_preds[target] = {'oof': oof, 'test': test, 'eff_n': eff_n}
        print(f"    {target}: eff_neighbors={eff_n:.1f}")

    # ============================================================
    # PHASE 6: Effective number of neighbors summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 6: EFFECTIVE NEIGHBORS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Kernel':<15} {'BW':>4} {'Angle EffN':>12} {'Depth EffN':>12} {'LR EffN':>12} {'Mean EffN':>10}")
    print(f"  {'-'*80}")

    # For each kernel at its best bw, show effective neighbors
    for kern in ['gaussian', 'laplacian', 'tricube', 'epanechnikov', 'cauchy']:
        bw_results = all_kernel_results[kern]
        best_bw = min(bw_results, key=lambda b: bw_results[b]['mean_mse'])
        r = bw_results[best_bw]
        eff_ns = [r[t].get('eff_n', 0) for t in TARGETS]
        print(f"  {kern:<15} {best_bw:>4.1f} {eff_ns[0]:>12.1f} {eff_ns[1]:>12.1f} {eff_ns[2]:>12.1f} "
              f"{np.mean(eff_ns):>10.1f}")

    # ============================================================
    # PHASE 7: Diversity analysis
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 7: DIVERSITY ANALYSIS (correlation with Sub 2063)")
    print(f"{'=' * 70}")

    # Use per-target optimal predictions
    for target in TARGETS:
        col = f'scaled_{target}'
        cfg = per_target_best[target][0]
        test_pred = per_target_optimal_preds[target]['test']
        r_2063 = np.corrcoef(sub_2063[col].values, test_pred)[0, 1]
        r_784 = np.corrcoef(sub_784[col].values, test_pred)[0, 1]
        print(f"  {target} ({cfg[0]} bw={cfg[1]} a={cfg[2]}): "
              f"r(2063)={r_2063:.4f}, r(784)={r_784:.4f}")

    # Also check each kernel at best bw for diversity
    print(f"\n  Per-kernel diversity (at best bw, alpha=10):")
    for kern in ['laplacian', 'tricube', 'epanechnikov', 'cauchy']:
        bw_results = all_kernel_results[kern]
        best_bw = min(bw_results, key=lambda b: bw_results[b]['mean_mse'])
        r = bw_results[best_bw]
        corrs = []
        for target in TARGETS:
            col = f'scaled_{target}'
            corr = np.corrcoef(sub_2063[col].values, r[target]['test'])[0, 1]
            corrs.append(corr)
        print(f"    {kern} bw={best_bw}: r(2063) = [{corrs[0]:.4f}, {corrs[1]:.4f}, {corrs[2]:.4f}] "
              f"mean={np.mean(corrs):.4f}")

    # ============================================================
    # PHASE 8: Generate submissions
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 8: GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # A: Per-target optimal kernel, standalone blended with Sub 784
    for aw, dw, lw, desc in [
        (0.50, 0.30, 0.50, "standard blend"),
        (0.30, 0.20, 0.30, "conservative blend"),
        (0.70, 0.40, 0.60, "aggressive blend"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*per_target_optimal_preds['angle']['test']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*per_target_optimal_preds['depth']['test']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*per_target_optimal_preds['left_right']['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: Per-target optimal kernel + Sub784 (aw={aw} dw={dw} lw={lw}) - {desc}")
        print(f"    Configs: angle={per_target_best['angle'][0]}, depth={per_target_best['depth'][0]}, LR={per_target_best['left_right'][0]}")

    # B: Per-target optimal kernel, blended with Sub 2063
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_2063.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_2063[col] + w*per_target_optimal_preds[target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {w*100:.0f}% per-target optimal kernel + {(1-w)*100:.0f}% Sub2063")

    # C: Best single kernel at best bw and alpha, blended with Sub 784
    print(f"\n  Best single kernel: {best_overall[0]} bw={best_overall[1]} alpha={best_alpha}")
    best_single = {}
    for target in TARGETS:
        X_tr, X_te = features[target]
        oof, test = locally_weighted_prediction(
            X_tr, y_scaled[target], X_te, pids_train, pids_test,
            kernel_name=best_overall[0], bandwidth_quantile=best_overall[1],
            alpha=best_alpha)
        best_single[target] = test

    for aw, dw, lw, desc in [
        (0.50, 0.30, 0.50, "standard"),
        (0.30, 0.20, 0.30, "conservative"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*best_single['angle']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*best_single['depth']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*best_single['left_right']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {best_overall[0]} bw={best_overall[1]} a={best_alpha} + Sub784 ({desc})")

    # D: Best single kernel blended with Sub 2063
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_2063.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_2063[col] + w*best_single[target]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {w*100:.0f}% {best_overall[0]} + {(1-w)*100:.0f}% Sub2063")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Gaussian baseline (bw=0.3, alpha=10): mean LOO MSE = {baseline_mean:.6f}")

    # Best per-target
    pt_mse = np.mean([per_target_best[t][1] for t in TARGETS])
    pt_delta = (pt_mse - baseline_mean) / baseline_mean * 100
    print(f"  Per-target optimal kernel: mean LOO MSE = {pt_mse:.6f} ({pt_delta:+.2f}%)")
    for target in TARGETS:
        cfg, mse = per_target_best[target]
        print(f"    {target}: {cfg[0]} bw={cfg[1]} alpha={cfg[2]} MSE={mse:.6f}")

    # Best single kernel
    best_alpha_res = alpha_results[best_alpha]
    single_mse = best_alpha_res['mean_mse']
    single_delta = (single_mse - baseline_mean) / baseline_mean * 100
    print(f"  Best single kernel: {best_overall[0]} bw={best_overall[1]} alpha={best_alpha}")
    print(f"    mean LOO MSE = {single_mse:.6f} ({single_delta:+.2f}%)")

    print(f"\n  WARNING: LOO MSE is systematically optimistic (LOO ~0.0037 vs LB ~0.0066).")
    print(f"  Kernel changes that improve LOO may not improve LB.")
    print(f"  Heavy-tailed kernels (Laplacian, Cauchy) use more neighbors = less variance,")
    print(f"  which should help with overfitting gap, but this needs LB validation.")

    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
