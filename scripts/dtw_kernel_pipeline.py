"""
DTW Kernel Pipeline

Replace Euclidean distance in the locally-weighted Ridge regression with
Dynamic Time Warping (DTW) distance on trajectory segments (frames 120-180).

Key insight: Two shots with identical mechanics but different timing look
far apart in Euclidean space (single frame) but close in DTW space
(full trajectory comparison).

This is a DROP-IN replacement for the distance metric in per_example_pipeline.py.
Everything else stays the same: per-player models, PLS augmentation, Gaussian kernel.
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
from dtaidistance import dtw_ndim, dtw

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Key joints for DTW trajectory comparison - MINIMAL set for speed
DTW_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'mid_hip', 'neck',
]

# Frame range for trajectory comparison
DTW_START = 130
DTW_END = 175
DTW_SUBSAMPLE = 3  # Every 3rd frame for speed (15 frames total)


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
    print("Loading data...", flush=True)
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
                print(f"  Processed {idx + 1}/{len(df)}", flush=True)

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
    """Extract compact feature set at a specific frame (same as per_example_pipeline)."""
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


def extract_dtw_trajectories(data, kp_index):
    """Extract trajectory segments for DTW comparison.
    Returns array of shape (n_shots, n_frames, n_channels) where
    n_channels = len(DTW_JOINTS) * 3 coordinates.
    """
    n = len(data['pids'])
    frames = np.arange(DTW_START, DTW_END, DTW_SUBSAMPLE)
    n_frames = len(frames)

    available_joints = [j for j in DTW_JOINTS if j in kp_index]
    n_channels = len(available_joints) * 3

    trajectories = np.zeros((n, n_frames, n_channels), dtype=np.float64)

    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)

        ch = 0
        for jname in available_joints:
            jidx = kp_index[jname]
            for coord in range(3):
                traj = ts_hr[frames, jidx, coord]
                # Clean NaNs
                bad = np.isnan(traj) | np.isinf(traj)
                if np.any(bad) and not np.all(bad):
                    good = ~bad
                    traj[bad] = np.interp(np.where(bad)[0], np.where(good)[0], traj[good])
                elif np.all(bad):
                    traj[:] = 0.0
                trajectories[i, :, ch] = traj
                ch += 1

    return trajectories


def compute_dtw_distance_matrix(trajs_a, trajs_b):
    """Compute DTW distance matrix between two sets of trajectories.
    Uses per-channel 1D DTW summed across channels for speed.
    trajs_a: (n_a, n_frames, n_channels)
    trajs_b: (n_b, n_frames, n_channels)
    Returns: (n_a, n_b) distance matrix
    """
    n_a, n_b = len(trajs_a), len(trajs_b)
    n_channels = trajs_a.shape[2]
    D = np.zeros((n_a, n_b), dtype=np.float64)

    # Per-channel DTW is MUCH faster than multidimensional DTW
    # Sum DTW distances across channels
    for ch in range(n_channels):
        print(f"    DTW channel {ch+1}/{n_channels}...", flush=True)
        # Extract 1D series for this channel
        series_a = [trajs_a[i, :, ch].astype(np.double) for i in range(n_a)]
        series_b = [trajs_b[j, :, ch].astype(np.double) for j in range(n_b)]

        # Use dtaidistance's fast C implementation
        for i in range(n_a):
            for j in range(n_b):
                d = dtw.distance_fast(series_a[i], series_b[j])
                D[i, j] += d

    return D


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


def dtw_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                            D_tr_tr_dtw, D_te_tr_dtw, D_tr_tr_euc, D_te_tr_euc,
                            dtw_weight=0.5, bandwidth_quantile=0.3):
    """Locally weighted Ridge using a BLEND of DTW and Euclidean distances."""
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

        # Get player-specific distance sub-matrices
        tr_global = np.where(tr_mask)[0]
        te_global = np.where(te_mask)[0]

        # Blend DTW and Euclidean distances
        D_dtw_pp = D_tr_tr_dtw[np.ix_(tr_global, tr_global)]
        D_euc_pp = D_tr_tr_euc[np.ix_(tr_global, tr_global)]

        # Normalize both to [0,1] range for fair blending
        d_dtw_max = D_dtw_pp.max() if D_dtw_pp.max() > 0 else 1.0
        d_euc_max = D_euc_pp.max() if D_euc_pp.max() > 0 else 1.0
        D_dtw_norm = D_dtw_pp / d_dtw_max
        D_euc_norm = D_euc_pp / d_euc_max

        D_blend_tr = dtw_weight * D_dtw_norm + (1 - dtw_weight) * D_euc_norm

        # Bandwidth from blended distances
        all_dists = D_blend_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        # OOF: leave-one-out
        for i in range(n_tr):
            weights = np.exp(-D_blend_tr[i, :] ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test predictions
        if len(X_te) > 0:
            D_dtw_te = D_te_tr_dtw[np.ix_(te_global, tr_global)]
            D_euc_te = D_te_tr_euc[np.ix_(te_global, tr_global)]
            D_dtw_te_norm = D_dtw_te / d_dtw_max
            D_euc_te_norm = D_euc_te / d_euc_max
            D_blend_te = dtw_weight * D_dtw_te_norm + (1 - dtw_weight) * D_euc_te_norm

            for j in range(len(X_te)):
                weights = np.exp(-D_blend_te[j, :] ** 2 / (2 * sigma ** 2))
                if weights.sum() < 1e-10:
                    test_preds[te_indices[j]] = np.mean(y_tr)
                    continue
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr_s, y_tr, sample_weight=weights)
                test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds


def euclidean_baseline(X_train, y_train, X_test, pids_train, pids_test,
                       bandwidth_quantile=0.3):
    """Standard Euclidean locally weighted prediction for comparison."""
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
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

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


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("DTW KERNEL PIPELINE", flush=True)
    print("=" * 70, flush=True)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']

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
    # STEP 1: Compute DTW distance matrices (the expensive part)
    # ============================================================
    print("\nExtracting DTW trajectories...", flush=True)
    trajs_train = extract_dtw_trajectories(train_data, kp_index)
    trajs_test = extract_dtw_trajectories(test_data, kp_index)
    print(f"  Train trajectories: {trajs_train.shape}", flush=True)
    print(f"  Test trajectories: {trajs_test.shape}", flush=True)

    print("\nComputing DTW distance matrices...", flush=True)
    t_dtw = time.time()
    D_tr_tr_dtw = compute_dtw_distance_matrix(trajs_train, trajs_train)
    print(f"  Train-train DTW: {time.time() - t_dtw:.1f}s", flush=True)

    t_dtw2 = time.time()
    D_te_tr_dtw = compute_dtw_distance_matrix(trajs_test, trajs_train)
    print(f"  Test-train DTW: {time.time() - t_dtw2:.1f}s", flush=True)

    # Also compute Euclidean distances on features (for blending)
    # We'll compute these per-target since features differ

    # ============================================================
    # STEP 2: Per-target pipeline
    # ============================================================
    results = {}
    sub_3411 = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")

    for target in TARGETS:
        print(f"\n{'=' * 70}", flush=True)
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})", flush=True)
        print(f"{'=' * 70}", flush=True)

        # Extract features (same as baseline)
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)

        # PLS augmentation
        y_raw = y_train[:, target_idx[target]]
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        y_target = y_scaled[target]

        # Compute Euclidean distance matrices on augmented features
        # (We need per-player standardized, so compute full and use sub-matrices)
        n_tr = len(X_train_aug)
        n_te = len(X_test_aug)
        D_tr_tr_euc = np.zeros((n_tr, n_tr))
        D_te_tr_euc = np.zeros((n_te, n_tr))

        for pid in sorted(np.unique(pids_train)):
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train_aug[tr_mask])
            X_te_s = scaler.transform(X_test_aug[te_mask]) if len(te_idx) > 0 else np.zeros((0, X_train_aug.shape[1]))

            D_tr_tr_euc[np.ix_(tr_idx, tr_idx)] = cdist(X_tr_s, X_tr_s, metric='euclidean')
            if len(te_idx) > 0:
                D_te_tr_euc[np.ix_(te_idx, tr_idx)] = cdist(X_te_s, X_tr_s, metric='euclidean')

        # --- BASELINE: Euclidean only ---
        print("  [BASELINE] Euclidean locally weighted regression...", flush=True)
        oof_euc, test_euc = euclidean_baseline(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.3)
        mse_euc = np.mean((oof_euc - y_target) ** 2)
        print(f"      CV MSE: {mse_euc:.6f}", flush=True)

        # --- DTW EXPERIMENTS ---
        best_mse = mse_euc
        best_oof = oof_euc
        best_test = test_euc
        best_config = "euclidean_baseline"

        for dtw_w in [0.2, 0.3, 0.5, 0.7, 1.0]:
            for bw in [0.2, 0.3, 0.4]:
                print(f"  [DTW] dtw_weight={dtw_w}, bw={bw}...", flush=True)
                oof_dtw, test_dtw = dtw_weighted_prediction(
                    X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                    D_tr_tr_dtw, D_te_tr_dtw, D_tr_tr_euc, D_te_tr_euc,
                    dtw_weight=dtw_w, bandwidth_quantile=bw)
                mse_dtw = np.mean((oof_dtw - y_target) ** 2)
                delta = (mse_dtw - mse_euc) / mse_euc * 100
                print(f"      CV MSE: {mse_dtw:.6f} (delta: {delta:+.2f}%)", flush=True)

                if mse_dtw < best_mse:
                    best_mse = mse_dtw
                    best_oof = oof_dtw
                    best_test = test_dtw
                    best_config = f"dtw_w={dtw_w}_bw={bw}"

        print(f"\n  BEST {target}: {best_config} (MSE={best_mse:.6f}, delta={(best_mse-mse_euc)/mse_euc*100:+.2f}%)", flush=True)

        # Diversity vs Sub 3411
        col = f'scaled_{target}'
        r = np.corrcoef(sub_3411[col].values, best_test)[0, 1]
        print(f"  Diversity vs Sub3411: r={r:.4f}", flush=True)

        results[target] = {
            'best_test': best_test,
            'best_oof': best_oof,
            'best_mse': best_mse,
            'best_config': best_config,
            'euc_mse': mse_euc,
        }

    # ============================================================
    # OVERALL RESULTS
    # ============================================================
    print(f"\n{'=' * 70}", flush=True)
    print("OVERALL RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    total_dtw = 0
    total_euc = 0
    for target in TARGETS:
        total_dtw += results[target]['best_mse']
        total_euc += results[target]['euc_mse']
        delta = (results[target]['best_mse'] - results[target]['euc_mse']) / results[target]['euc_mse'] * 100
        print(f"  {target}: DTW best={results[target]['best_mse']:.6f}, Euc={results[target]['euc_mse']:.6f} ({delta:+.2f}%) [{results[target]['best_config']}]", flush=True)

    print(f"  MEAN DTW: {total_dtw/3:.6f}", flush=True)
    print(f"  MEAN EUC: {total_euc/3:.6f}", flush=True)
    print(f"  DELTA: {(total_dtw/3 - total_euc/3)/(total_euc/3)*100:+.2f}%", flush=True)

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}", flush=True)
    print("GENERATING SUBMISSIONS", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Standalone DTW
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle']['best_test'],
        'scaled_depth': results['depth']['best_test'],
        'scaled_left_right': results['left_right']['best_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: DTW STANDALONE", flush=True)

    # Blends with Sub 3411
    for w in [0.03, 0.05, 0.10, 0.15, 0.20]:
        sub_num = get_next_submission_number()
        blended = sub_3411.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_3411[col] + w * results[target]['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(w*100)}% DTW + {int((1-w)*100)}% Sub3411", flush=True)

    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {t: {'best_mse': float(results[t]['best_mse']),
                        'euc_mse': float(results[t]['euc_mse']),
                        'best_config': results[t]['best_config']}
                    for t in TARGETS},
        'mean_dtw_mse': float(total_dtw / 3),
        'mean_euc_mse': float(total_euc / 3),
    }
    with open(OUTPUT_DIR / "dtw_kernel_pipeline_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
