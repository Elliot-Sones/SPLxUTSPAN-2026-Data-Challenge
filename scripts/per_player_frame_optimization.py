"""
Per-Player Per-Target Optimal Frame Search

Instead of using fixed TARGET_FRAMES for all players, find the optimal
frame for each player x target combination. The hypothesis is that
different players have different release timings and biomechanics,
so the optimal feature-extraction frame may differ.

Pipeline is identical to per_example_pipeline.py (locally weighted Ridge
with PLS augmentation), only the frame parameter varies.

WARNING: With only ~66-74 shots per player, per-player frame optimization
could easily overfit. We track this carefully.
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
# FEATURE EXTRACTION
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


def extract_all_features_at_frame(data, frame):
    """Extract features for all shots at a specific frame."""
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


# ==============================================================
# LOO LOCALLY WEIGHTED RIDGE - PER PLAYER ONLY
# ==============================================================

def loo_per_player(X_player, y_player, bandwidth_quantile=0.3, alpha=10.0):
    """Run LOO locally weighted Ridge for a single player's data.
    Returns LOO predictions array."""
    n = len(X_player)
    oof = np.zeros(n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_player)

    D = cdist(X_s, X_s, metric='euclidean')
    all_dists = D[np.triu_indices(n, k=1)]
    if len(all_dists) > 0:
        sigma = np.quantile(all_dists, bandwidth_quantile)
        sigma = max(sigma, 1e-6)
    else:
        sigma = 1.0

    for i in range(n):
        dists = D[i, :]
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
        weights[i] = 0
        if weights.sum() < 1e-10:
            oof[i] = np.mean(y_player)
            continue
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_s, y_player, sample_weight=weights)
        oof[i] = ridge.predict(X_s[i:i+1])[0]

    return oof


def full_loo_with_per_player_frames(train_data, test_data, y_scaled, y_train,
                                     pids_train, pids_test, target,
                                     player_frames, target_idx):
    """Run full LOO + test prediction with per-player frame settings.

    player_frames: dict {pid: frame}
    """
    unique_pids = sorted(np.unique(pids_train))
    kp_index = train_data['kp_index']
    y_target = y_scaled[target]
    y_raw = y_train[:, target_idx[target]]

    oof_preds = np.zeros(len(pids_train))
    test_preds = np.zeros(len(pids_test))

    for pid in unique_pids:
        frame = player_frames[pid]
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        # Extract features at this player's optimal frame
        # Train shots for this player
        train_feats = []
        for i in tr_indices:
            ts_3d = train_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
            train_feats.append(feats)

        # Test shots for this player
        test_feats = []
        for i in te_indices:
            ts_3d = test_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
            test_feats.append(feats)

        X_tr_hc = np.array(train_feats, dtype=np.float32)
        X_tr_hc = np.nan_to_num(X_tr_hc, nan=0.0, posinf=0.0, neginf=0.0)

        if len(test_feats) > 0:
            X_te_hc = np.array(test_feats, dtype=np.float32)
            X_te_hc = np.nan_to_num(X_te_hc, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_te_hc = np.zeros((0, X_tr_hc.shape[1]), dtype=np.float32)

        # PLS augmentation within this player
        n_p = len(tr_indices)
        max_nc = 15

        pls_scaler = StandardScaler()
        raw_tr = pls_scaler.fit_transform(train_data['X_raw'][tr_mask])
        if len(te_indices) > 0:
            raw_te = pls_scaler.transform(test_data['X_raw'][te_mask])
        else:
            raw_te = np.zeros((0, raw_tr.shape[1]))

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15]:
            if c > nc:
                break
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for ktr, kval in kf.split(raw_tr):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_tr[ktr], y_raw[tr_mask][ktr])
                pred = pls.predict(raw_tr[kval]).flatten()
                mses.append(np.mean((pred - y_raw[tr_mask][kval]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls_train = np.zeros((n_p, max_nc), dtype=np.float32)
        pls_test = np.zeros((len(te_indices), max_nc), dtype=np.float32)

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_tr, y_raw[tr_mask])
        pls_train[:, :best_nc] = pls.transform(raw_tr)
        if len(te_indices) > 0:
            pls_test[:, :best_nc] = pls.transform(raw_te)

        X_tr_aug = np.hstack([X_tr_hc, pls_train])
        X_te_aug = np.hstack([X_te_hc, pls_test])

        # Locally weighted Ridge LOO
        y_p = y_target[tr_mask]
        n_tr = len(X_tr_aug)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_aug)
        if len(X_te_aug) > 0:
            X_te_s = scaler.transform(X_te_aug)
        else:
            X_te_s = np.zeros((0, X_tr_s.shape[1]))

        D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        all_dists = D_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, 0.3)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        # LOO
        for i in range(n_tr):
            dists = D_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_p)
                continue
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_p, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test
        if len(X_te_s) > 0:
            D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
            for j in range(len(X_te_s)):
                dists = D_te[j, :]
                weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
                if weights.sum() < 1e-10:
                    test_preds[te_indices[j]] = np.mean(y_p)
                    continue
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr_s, y_p, sample_weight=weights)
                test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    loo_mse = np.mean((oof_preds - y_target) ** 2)
    return oof_preds, test_preds, loo_mse


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("PER-PLAYER PER-TARGET OPTIMAL FRAME SEARCH")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    unique_pids = sorted(np.unique(pids_train))

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
    # PHASE 1: Pre-compute features at all candidate frames
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: Pre-compute features at all candidate frames")
    print(f"{'=' * 70}")

    scan_frames = list(range(120, 191, 5))  # 120, 125, ..., 190
    print(f"Scanning {len(scan_frames)} frames: {scan_frames}")

    # Pre-compute features for each frame (expensive but done once)
    # Store: {frame: (X_train_hc, X_test_hc)}
    features_cache = {}
    for frame in scan_frames:
        print(f"  Extracting features at frame {frame}...")
        X_tr, _ = extract_all_features_at_frame(train_data, frame)
        X_te, _ = extract_all_features_at_frame(test_data, frame)
        features_cache[frame] = (X_tr, X_te)

    # ============================================================
    # PHASE 2: Per-player per-target frame sweep
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: Per-Player Per-Target Frame Sweep")
    print(f"{'=' * 70}")

    # Results: {target: {pid: {frame: loo_mse}}}
    sweep_results = {t: {pid: {} for pid in unique_pids} for t in TARGETS}
    optimal_frames = {t: {} for t in TARGETS}

    for target in TARGETS:
        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]
        print(f"\n  TARGET: {target.upper()} (fixed frame: {TARGET_FRAMES[target]})")

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            n_p = tr_mask.sum()
            tr_indices = np.where(tr_mask)[0]
            y_p = y_target[tr_mask]

            print(f"\n    Player {pid} ({n_p} shots):")

            best_frame = None
            best_mse = float('inf')
            frame_mse_list = []

            for frame in scan_frames:
                # Get pre-computed features for this player
                X_tr_hc = features_cache[frame][0][tr_mask]

                # PLS augmentation for this player only
                pls_scaler = StandardScaler()
                raw_tr = pls_scaler.fit_transform(train_data['X_raw'][tr_mask])

                nc = min(15, n_p - n_p // 5 - 1)
                nc = max(3, nc)

                best_nc, best_nc_mse = 3, float('inf')
                for c in [3, 5, 8, 10, 15]:
                    if c > nc:
                        break
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    mses = []
                    for ktr, kval in kf.split(raw_tr):
                        pls_obj = PLSRegression(n_components=c)
                        pls_obj.fit(raw_tr[ktr], y_raw[tr_mask][ktr])
                        pred = pls_obj.predict(raw_tr[kval]).flatten()
                        mses.append(np.mean((pred - y_raw[tr_mask][kval]) ** 2))
                    if np.mean(mses) < best_nc_mse:
                        best_nc_mse = np.mean(mses)
                        best_nc = c

                pls_train = np.zeros((n_p, 15), dtype=np.float32)
                pls_obj = PLSRegression(n_components=best_nc)
                pls_obj.fit(raw_tr, y_raw[tr_mask])
                pls_train[:, :best_nc] = pls_obj.transform(raw_tr)

                X_tr_aug = np.hstack([X_tr_hc, pls_train])

                # LOO locally weighted Ridge
                oof = loo_per_player(X_tr_aug, y_p, bandwidth_quantile=0.3, alpha=10.0)
                mse = np.mean((oof - y_p) ** 2)

                sweep_results[target][pid][frame] = mse
                frame_mse_list.append((frame, mse))

                if mse < best_mse:
                    best_mse = mse
                    best_frame = frame

            optimal_frames[target][pid] = best_frame
            fixed_mse = sweep_results[target][pid].get(TARGET_FRAMES[target], None)
            if fixed_mse is not None:
                delta_pct = (best_mse - fixed_mse) / fixed_mse * 100
            else:
                delta_pct = 0.0

            # Print top 5 frames
            sorted_frames = sorted(frame_mse_list, key=lambda x: x[1])
            print(f"      Fixed frame {TARGET_FRAMES[target]}: MSE = {fixed_mse:.6f}" if fixed_mse else "")
            print(f"      OPTIMAL frame {best_frame}: MSE = {best_mse:.6f} ({delta_pct:+.2f}% vs fixed)")
            print(f"      Top 5: {', '.join([f'{f}:{m:.6f}' for f, m in sorted_frames[:5]])}")

    # ============================================================
    # PHASE 3: Summary of optimal frames
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 3: Optimal Frame Summary")
    print(f"{'=' * 70}")

    print("\n  Per-player optimal frames vs fixed:")
    for target in TARGETS:
        print(f"\n  {target.upper()} (fixed: {TARGET_FRAMES[target]}):")
        for pid in unique_pids:
            opt = optimal_frames[target][pid]
            fixed_mse = sweep_results[target][pid].get(TARGET_FRAMES[target], float('inf'))
            opt_mse = sweep_results[target][pid][opt]
            delta = (opt_mse - fixed_mse) / fixed_mse * 100 if fixed_mse > 0 else 0
            n_shots = (pids_train == pid).sum()
            print(f"    Player {pid} ({n_shots} shots): fixed={TARGET_FRAMES[target]} -> optimal={opt} "
                  f"(MSE: {fixed_mse:.6f} -> {opt_mse:.6f}, {delta:+.2f}%)")

    # ============================================================
    # PHASE 4: Full LOO with per-player optimal frames
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 4: Full LOO with Per-Player Optimal Frames")
    print(f"{'=' * 70}")

    # Run the full pipeline with per-player frames
    oof_results = {}
    test_results = {}
    mse_results = {}

    for target in TARGETS:
        print(f"\n  Running {target.upper()} with per-player frames: {optimal_frames[target]}")
        oof, test_pred, loo_mse = full_loo_with_per_player_frames(
            train_data, test_data, y_scaled, y_train,
            pids_train, pids_test, target,
            optimal_frames[target], target_idx)
        oof_results[target] = oof
        test_results[target] = test_pred
        mse_results[target] = loo_mse
        print(f"    LOO MSE: {loo_mse:.6f}")

    # Also run fixed-frame baseline for comparison
    print("\n  Running fixed-frame baseline...")
    baseline_mse = {}
    baseline_oof = {}
    baseline_test = {}
    for target in TARGETS:
        fixed_frames = {pid: TARGET_FRAMES[target] for pid in unique_pids}
        oof, test_pred, loo_mse = full_loo_with_per_player_frames(
            train_data, test_data, y_scaled, y_train,
            pids_train, pids_test, target,
            fixed_frames, target_idx)
        baseline_mse[target] = loo_mse
        baseline_oof[target] = oof
        baseline_test[target] = test_pred
        print(f"    {target}: LOO MSE = {loo_mse:.6f}")

    # Comparison
    print(f"\n  COMPARISON:")
    print(f"  {'Target':<12} {'Fixed MSE':<12} {'OptFrame MSE':<14} {'Delta %':<10}")
    total_fixed = 0
    total_opt = 0
    for target in TARGETS:
        fixed = baseline_mse[target]
        opt = mse_results[target]
        delta = (opt - fixed) / fixed * 100
        total_fixed += fixed
        total_opt += opt
        print(f"  {target:<12} {fixed:<12.6f} {opt:<14.6f} {delta:+.2f}%")
    print(f"  {'MEAN':<12} {total_fixed/3:<12.6f} {total_opt/3:<14.6f} "
          f"{(total_opt - total_fixed) / total_fixed * 100:+.2f}%")

    # ============================================================
    # PHASE 5: Smoothed / robust version (median of top-3 frames)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 5: Robust Frame Selection (avg of top-3 closest to optimal)")
    print(f"{'=' * 70}")

    # Instead of picking the single best frame, pick the median of top-3
    # This reduces overfitting risk
    robust_frames = {t: {} for t in TARGETS}
    for target in TARGETS:
        for pid in unique_pids:
            sorted_frames = sorted(sweep_results[target][pid].items(),
                                    key=lambda x: x[1])
            top3 = [f for f, _ in sorted_frames[:3]]
            # Use median frame
            robust_frames[target][pid] = int(np.median(top3))

    print("\n  Robust (median of top-3) per-player frames:")
    for target in TARGETS:
        print(f"\n  {target.upper()}:")
        for pid in unique_pids:
            print(f"    Player {pid}: optimal={optimal_frames[target][pid]}, "
                  f"robust={robust_frames[target][pid]}")

    robust_mse = {}
    robust_oof = {}
    robust_test = {}
    for target in TARGETS:
        print(f"\n  Running {target.upper()} with robust frames: {robust_frames[target]}")
        oof, test_pred, loo_mse = full_loo_with_per_player_frames(
            train_data, test_data, y_scaled, y_train,
            pids_train, pids_test, target,
            robust_frames[target], target_idx)
        robust_mse[target] = loo_mse
        robust_oof[target] = oof
        robust_test[target] = test_pred
        print(f"    LOO MSE: {loo_mse:.6f}")

    print(f"\n  COMPARISON (robust):")
    print(f"  {'Target':<12} {'Fixed MSE':<12} {'Robust MSE':<14} {'Delta %':<10}")
    total_robust = 0
    for target in TARGETS:
        fixed = baseline_mse[target]
        rob = robust_mse[target]
        delta = (rob - fixed) / fixed * 100
        total_robust += rob
        print(f"  {target:<12} {fixed:<12.6f} {rob:<14.6f} {delta:+.2f}%")
    print(f"  {'MEAN':<12} {total_fixed/3:<12.6f} {total_robust/3:<14.6f} "
          f"{(total_robust - total_fixed) / total_fixed * 100:+.2f}%")

    # ============================================================
    # PHASE 6: Generate submissions if improvement found
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 6: Generate Submissions")
    print(f"{'=' * 70}")

    # Use whichever is better: optimal or robust
    use_optimal = (total_opt < total_robust)
    if use_optimal:
        final_test = test_results
        final_oof = oof_results
        final_mse = mse_results
        final_frames = optimal_frames
        method = "optimal"
    else:
        final_test = robust_test
        final_oof = robust_oof
        final_mse = robust_mse
        final_frames = robust_frames
        method = "robust"

    total_best = sum(final_mse[t] for t in TARGETS)
    print(f"\n  Using {method} frames (mean LOO MSE: {total_best/3:.6f})")

    # Also try per-target best selection (optimal for some, robust for others)
    hybrid_test = {}
    hybrid_mse = {}
    for target in TARGETS:
        if mse_results[target] < robust_mse[target]:
            hybrid_test[target] = test_results[target]
            hybrid_mse[target] = mse_results[target]
            print(f"  {target}: using optimal (MSE {mse_results[target]:.6f})")
        else:
            hybrid_test[target] = robust_test[target]
            hybrid_mse[target] = robust_mse[target]
            print(f"  {target}: using robust (MSE {robust_mse[target]:.6f})")

    total_hybrid = sum(hybrid_mse[t] for t in TARGETS)
    print(f"  Hybrid mean LOO MSE: {total_hybrid/3:.6f}")

    # Load reference submissions
    sub_2169 = pd.read_csv(SUBMISSION_DIR / "submission_2169.csv")
    sub_2063 = pd.read_csv(SUBMISSION_DIR / "submission_2063.csv")

    submissions_info = []

    # Sub A: Standalone per-player frame (hybrid)
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': hybrid_test['angle'],
        'scaled_depth': hybrid_test['depth'],
        'scaled_left_right': hybrid_test['left_right'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    info = (f"Sub {sub_num}: STANDALONE per-player-frame ({method}) "
            f"LOO MSE: {total_hybrid/3:.6f}")
    print(f"\n  {info}")
    submissions_info.append(info)

    # Diversity check
    print(f"\n  Diversity with Sub 2169:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_2169[col].values, hybrid_test[target])[0, 1]
        print(f"    {target}: r = {r:.4f}")

    print(f"\n  Diversity with Sub 2063:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_2063[col].values, hybrid_test[target])[0, 1]
        print(f"    {target}: r = {r:.4f}")

    # Blends with Sub 2169
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({'id': test_data['ids']})
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_2169[col].values + w*hybrid_test[target]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        info = f"Sub {sub_num}: {int(w*100)}% per-player-frame + {int((1-w)*100)}% Sub 2169"
        print(f"  {info}")
        submissions_info.append(info)

    # Blends with Sub 2063
    for w in [0.10, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({'id': test_data['ids']})
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_2063[col].values + w*hybrid_test[target]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        info = f"Sub {sub_num}: {int(w*100)}% per-player-frame + {int((1-w)*100)}% Sub 2063"
        print(f"  {info}")
        submissions_info.append(info)

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    print(f"\nFixed frame baseline LOO MSE: {total_fixed/3:.6f}")
    print(f"Per-player optimal LOO MSE:   {total_opt/3:.6f} ({(total_opt-total_fixed)/total_fixed*100:+.2f}%)")
    print(f"Per-player robust LOO MSE:    {total_robust/3:.6f} ({(total_robust-total_fixed)/total_fixed*100:+.2f}%)")
    print(f"Per-player hybrid LOO MSE:    {total_hybrid/3:.6f} ({(total_hybrid-total_fixed)/total_fixed*100:+.2f}%)")

    print(f"\nOptimal per-player frames:")
    for target in TARGETS:
        print(f"  {target}: {optimal_frames[target]}")

    print(f"\nRobust per-player frames:")
    for target in TARGETS:
        print(f"  {target}: {robust_frames[target]}")

    print(f"\nSubmissions generated:")
    for info in submissions_info:
        print(f"  {info}")

    print(f"\nTotal time: {elapsed:.1f}s")

    return {
        'sweep_results': sweep_results,
        'optimal_frames': optimal_frames,
        'robust_frames': robust_frames,
        'baseline_mse': baseline_mse,
        'mse_results': mse_results,
        'robust_mse': robust_mse,
        'hybrid_mse': hybrid_mse,
        'submissions_info': submissions_info,
    }


if __name__ == "__main__":
    main()
