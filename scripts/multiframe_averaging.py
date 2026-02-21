"""
Multi-Frame Prediction Averaging

Reduces prediction variance by:
1. Extracting features at MULTIPLE frames around each target's optimal frame
2. Running per-example locally weighted Ridge for EACH frame independently
3. Averaging predictions across frames

PLS is fitted ONCE (not per frame) to avoid overfitting the augmentation.
Only the HC features change per frame.

Configurations tested:
- Window sizes: 3, 5, 7 frames
- Frame spacings: 1, 2, 3 apart
- Both uniform and distance-weighted averaging
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
    """Extract features for all shots at a specific frame number."""
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
# PLS AUGMENTATION (fitted ONCE, reused across frames)
# ==============================================================

def fit_pls_components(y_raw, pids_train, pids_test, X_raw_train, X_raw_test):
    """Fit PLS ONCE on raw timeseries and return PLS features for train/test.
    These do not depend on the extraction frame."""
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
                pls.fit(raw_tr[tr_idx], y_raw[tr_mask][tr_idx])
                pred = pls.predict(raw_tr[val_idx]).flatten()
                mses.append(np.mean((pred - y_raw[tr_mask][val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_tr, y_raw[tr_mask])
        pls_train[tr_mask, :best_nc] = pls.transform(raw_tr)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)

    return pls_train, pls_test


# ==============================================================
# LOCALLY WEIGHTED PREDICTION (from per_example_pipeline.py)
# ==============================================================

def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.5, alpha=10.0):
    """Per-example locally weighted Ridge regression."""
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

        # OOF: leave-one-out locally weighted
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

        # Test: locally weighted for each test example
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
# MULTI-FRAME AVERAGING CORE
# ==============================================================

def generate_frame_list(center_frame, n_frames, spacing):
    """Generate a symmetric list of frames around center_frame."""
    half = n_frames // 2
    frames = [center_frame + (i - half) * spacing for i in range(n_frames)]
    # Clip to valid range
    frames = [max(0, min(239, f)) for f in frames]
    return frames


def run_multiframe_for_target(train_data, test_data, y_target, pids_train, pids_test,
                               pls_train, pls_test, target_name,
                               n_frames=5, spacing=2, bandwidth_quantile=0.5, alpha=10.0):
    """Run locally weighted prediction at multiple frames and average.

    Returns: oof_avg, test_avg, per-frame results dict
    """
    center = TARGET_FRAMES[target_name]
    frames = generate_frame_list(center, n_frames, spacing)

    print(f"    Frames: {frames} (center={center}, n={n_frames}, spacing={spacing})")

    all_oof = []
    all_test = []
    per_frame_mse = []

    for fi, frame in enumerate(frames):
        # Extract HC features at this frame
        X_train_hc, _ = extract_all_features_at_frame(train_data, frame)
        X_test_hc, _ = extract_all_features_at_frame(test_data, frame)

        # Augment with pre-fitted PLS (same for all frames)
        X_train_aug = np.hstack([X_train_hc, pls_train])
        X_test_aug = np.hstack([X_test_hc, pls_test])

        # Run locally weighted prediction
        oof, test_pred = locally_weighted_prediction(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=bandwidth_quantile, alpha=alpha)

        mse = np.mean((oof - y_target) ** 2)
        per_frame_mse.append(mse)
        all_oof.append(oof)
        all_test.append(test_pred)

        print(f"      Frame {frame}: LOO MSE = {mse:.6f}")

    # Uniform averaging
    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    mse_avg = np.mean((oof_avg - y_target) ** 2)

    print(f"    Uniform average: LOO MSE = {mse_avg:.6f}")

    # Inverse-MSE weighted averaging
    weights = 1.0 / np.array(per_frame_mse)
    weights /= weights.sum()
    oof_wmse = np.average(all_oof, axis=0, weights=weights)
    test_wmse = np.average(all_test, axis=0, weights=weights)
    mse_wmse = np.mean((oof_wmse - y_target) ** 2)

    print(f"    MSE-weighted avg: LOO MSE = {mse_wmse:.6f} (weights: {np.round(weights, 3)})")

    # Optimal CV-based weights (constrained to non-negative, sum to 1)
    # Try a simple grid for the center frame vs average
    best_center_w = 1.0
    best_mse_mix = per_frame_mse[n_frames // 2]  # center-only
    center_oof = all_oof[n_frames // 2]
    center_test = all_test[n_frames // 2]

    for cw in np.arange(0.0, 1.05, 0.1):
        oof_mix = cw * center_oof + (1 - cw) * oof_avg
        mse_mix = np.mean((oof_mix - y_target) ** 2)
        if mse_mix < best_mse_mix:
            best_mse_mix = mse_mix
            best_center_w = cw

    oof_best = best_center_w * center_oof + (1 - best_center_w) * oof_avg
    test_best = best_center_w * center_test + (1 - best_center_w) * test_avg

    print(f"    Center vs avg mix (cw={best_center_w:.1f}): LOO MSE = {best_mse_mix:.6f}")

    return {
        'oof_uniform': oof_avg,
        'test_uniform': test_avg,
        'mse_uniform': mse_avg,
        'oof_wmse': oof_wmse,
        'test_wmse': test_wmse,
        'mse_wmse': mse_wmse,
        'oof_best_mix': oof_best,
        'test_best_mix': test_best,
        'mse_best_mix': best_mse_mix,
        'center_weight': best_center_w,
        'per_frame_mse': per_frame_mse,
        'frames': frames,
        'center_mse': per_frame_mse[n_frames // 2],
    }


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("MULTI-FRAME PREDICTION AVERAGING")
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

    # ===========================================================
    # STEP 1: Fit PLS ONCE per target (frame-independent)
    # ===========================================================
    print("\n--- Fitting PLS components (once per target) ---")
    pls_features = {}
    for target in TARGETS:
        print(f"  PLS for {target}...")
        y_raw = y_train[:, target_idx[target]]
        pls_train, pls_test = fit_pls_components(
            y_raw, pids_train, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        pls_features[target] = (pls_train, pls_test)
        print(f"    PLS shape: {pls_train.shape[1]} components")

    # ===========================================================
    # STEP 2: Single-frame baseline (reproduce Sub 1350)
    # ===========================================================
    print("\n--- Single-frame baseline (reproducing Sub 1350 approach) ---")
    baseline_results = {}
    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        print(f"\n  {target} at frame {frame}:")
        X_train_hc, _ = extract_all_features_at_frame(train_data, frame)
        X_test_hc, _ = extract_all_features_at_frame(test_data, frame)
        pls_tr, pls_te = pls_features[target]
        X_train_aug = np.hstack([X_train_hc, pls_tr])
        X_test_aug = np.hstack([X_test_hc, pls_te])

        oof, test_pred = locally_weighted_prediction(
            X_train_aug, y_scaled[target], X_test_aug, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse = np.mean((oof - y_scaled[target]) ** 2)
        print(f"    LOO MSE = {mse:.6f}")
        baseline_results[target] = {
            'oof': oof, 'test': test_pred, 'mse': mse
        }

    baseline_mean = np.mean([baseline_results[t]['mse'] for t in TARGETS])
    print(f"\n  Baseline mean LOO MSE: {baseline_mean:.6f}")

    # ===========================================================
    # STEP 3: Multi-frame grid search
    # ===========================================================
    print("\n" + "=" * 70)
    print("MULTI-FRAME GRID SEARCH")
    print("=" * 70)

    configs = [
        (3, 1), (3, 2), (3, 3),
        (5, 1), (5, 2), (5, 3),
        (7, 1), (7, 2), (7, 3),
    ]

    all_results = {}

    for n_frames, spacing in configs:
        config_key = f"n{n_frames}_s{spacing}"
        print(f"\n{'=' * 50}")
        print(f"CONFIG: {n_frames} frames, spacing {spacing}")
        print(f"{'=' * 50}")

        config_results = {}
        for target in TARGETS:
            print(f"\n  {target.upper()}:")
            pls_tr, pls_te = pls_features[target]
            result = run_multiframe_for_target(
                train_data, test_data, y_scaled[target], pids_train, pids_test,
                pls_tr, pls_te, target,
                n_frames=n_frames, spacing=spacing,
                bandwidth_quantile=0.5, alpha=10.0)
            config_results[target] = result

        # Summary for this config
        for avg_type in ['uniform', 'wmse', 'best_mix']:
            mses = [config_results[t][f'mse_{avg_type}'] for t in TARGETS]
            mean_mse = np.mean(mses)
            delta = mean_mse - baseline_mean
            print(f"  {avg_type}: mean MSE = {mean_mse:.6f} (delta: {delta:+.6f}, {100*delta/baseline_mean:+.1f}%)")

        all_results[config_key] = config_results

    # ===========================================================
    # STEP 4: Find best config
    # ===========================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Config':<12} {'Uniform':>10} {'MSE-wt':>10} {'Mix':>10} {'Best':>10} {'vs base':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    best_overall_key = None
    best_overall_mse = float('inf')
    best_overall_type = None

    for config_key, config_results in all_results.items():
        for avg_type in ['uniform', 'wmse', 'best_mix']:
            mses = [config_results[t][f'mse_{avg_type}'] for t in TARGETS]
            mean_mse = np.mean(mses)
            if mean_mse < best_overall_mse:
                best_overall_mse = mean_mse
                best_overall_key = config_key
                best_overall_type = avg_type

        u_mse = np.mean([config_results[t]['mse_uniform'] for t in TARGETS])
        w_mse = np.mean([config_results[t]['mse_wmse'] for t in TARGETS])
        m_mse = np.mean([config_results[t]['mse_best_mix'] for t in TARGETS])
        b_mse = min(u_mse, w_mse, m_mse)
        delta = b_mse - baseline_mean
        print(f"  {config_key:<12} {u_mse:>10.6f} {w_mse:>10.6f} {m_mse:>10.6f} {b_mse:>10.6f} {delta:>+10.6f}")

    print(f"\n  Baseline single-frame: {baseline_mean:.6f}")
    print(f"  Best multi-frame: {best_overall_mse:.6f} ({best_overall_key}, {best_overall_type})")
    print(f"  Delta: {best_overall_mse - baseline_mean:+.6f} ({100*(best_overall_mse - baseline_mean)/baseline_mean:+.2f}%)")

    # ===========================================================
    # STEP 5: Generate submissions
    # ===========================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Best multi-frame config
    best_config = all_results[best_overall_key]

    # Also generate for top 3 configs
    sorted_configs = []
    for config_key, config_results in all_results.items():
        for avg_type in ['uniform', 'wmse', 'best_mix']:
            mses = [config_results[t][f'mse_{avg_type}'] for t in TARGETS]
            mean_mse = np.mean(mses)
            sorted_configs.append((mean_mse, config_key, avg_type))
    sorted_configs.sort()

    # Generate submissions for top 5 configs
    generated_subs = []
    for rank, (mean_mse, config_key, avg_type) in enumerate(sorted_configs[:5]):
        config_results = all_results[config_key]

        # Standalone submission
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': config_results['angle'][f'test_{avg_type}'],
            'scaled_depth': config_results['depth'][f'test_{avg_type}'],
            'scaled_left_right': config_results['left_right'][f'test_{avg_type}'],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {config_key} {avg_type} standalone (LOO MSE={mean_mse:.6f})")
        generated_subs.append(sub_num)

        # Blended with Sub 784 at dw=0.30, lw=0.50
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = sub_784['scaled_angle']  # Don't touch angle (aw=0)
        blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * config_results['depth'][f'test_{avg_type}']
        blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * config_results['left_right'][f'test_{avg_type}']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

        r_a = np.corrcoef(sub_784['scaled_angle'], config_results['angle'][f'test_{avg_type}'])[0, 1]
        r_d = np.corrcoef(sub_784['scaled_depth'], config_results['depth'][f'test_{avg_type}'])[0, 1]
        r_l = np.corrcoef(sub_784['scaled_left_right'], config_results['left_right'][f'test_{avg_type}'])[0, 1]
        print(f"  Sub {sub_num}: {config_key} {avg_type} blended dw=0.30 lw=0.50")
        print(f"    Corr with Sub784: angle r={r_a:.4f}, depth r={r_d:.4f}, LR r={r_l:.4f}")
        generated_subs.append(sub_num)

    # Also generate blended baseline single-frame for comparison
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    blended['scaled_angle'] = sub_784['scaled_angle']
    blended['scaled_depth'] = 0.70 * sub_784['scaled_depth'] + 0.30 * baseline_results['depth']['test']
    blended['scaled_left_right'] = 0.50 * sub_784['scaled_left_right'] + 0.50 * baseline_results['left_right']['test']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: BASELINE single-frame blended dw=0.30 lw=0.50 (for comparison)")
    generated_subs.append(sub_num)

    # Per-target best: pick the best config for EACH target independently
    print("\n  Per-target best config selection:")
    per_target_best = {}
    for target in TARGETS:
        best_tmse = float('inf')
        best_tkey = None
        best_ttype = None
        for config_key, config_results in all_results.items():
            for avg_type in ['uniform', 'wmse', 'best_mix']:
                tmse = config_results[target][f'mse_{avg_type}']
                if tmse < best_tmse:
                    best_tmse = tmse
                    best_tkey = config_key
                    best_ttype = avg_type
        per_target_best[target] = (best_tkey, best_ttype, best_tmse)
        base_tmse = baseline_results[target]['mse']
        print(f"    {target}: {best_tkey} {best_ttype} MSE={best_tmse:.6f} (vs baseline {base_tmse:.6f}, delta {best_tmse-base_tmse:+.6f})")

    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    for target in TARGETS:
        config_key, avg_type, _ = per_target_best[target]
        config_results = all_results[config_key]
        col = f'scaled_{target}'
        if target == 'angle':
            blended[col] = sub_784[col]  # aw=0
        elif target == 'depth':
            blended[col] = 0.70 * sub_784[col] + 0.30 * config_results[target][f'test_{avg_type}']
        else:  # left_right
            blended[col] = 0.50 * sub_784[col] + 0.50 * config_results[target][f'test_{avg_type}']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: PER-TARGET BEST multi-frame blended dw=0.30 lw=0.50")
    generated_subs.append(sub_num)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Generated submissions: {generated_subs}")
    print(f"{'=' * 70}")

    # Return results for reporting
    return {
        'baseline_results': baseline_results,
        'baseline_mean': baseline_mean,
        'all_results': all_results,
        'best_overall': (best_overall_key, best_overall_type, best_overall_mse),
        'per_target_best': per_target_best,
        'generated_subs': generated_subs,
    }


if __name__ == "__main__":
    main()
