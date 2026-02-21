"""
Pseudo-Labeling for Ridge Pipeline

Use confident test predictions as pseudo-labels to expand the training set.
Key insight: our model is already good (LB 0.006234). For test shots that are
similar to many training shots, predictions are likely close to true values.

Strategy:
1. Generate test predictions using current best pipeline
2. Measure confidence: how many similar training shots exist (kernel weight sum)
3. Add top-N most confident pseudo-labeled test shots to training
4. Retrain and re-predict
5. Iterate 2-3 rounds

The risk is confirmation bias - mitigated by:
- Only adding the most confident predictions
- Using kernel weight sum as confidence metric
- Stopping if LOO on original 345 degrades
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
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
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
    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal', 'right_fourth_finger_distal']
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


def extract_all_features(data, target, kp_index):
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def predict_with_confidence(X_train, y_train, X_test, pids_train, pids_test,
                            bandwidth_quantile=0.3):
    """Locally weighted Ridge prediction WITH confidence scores.
    Confidence = sum of kernel weights (how much training data support this prediction).
    """
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    test_confidence = np.zeros(len(X_test))
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

        # OOF
        for i in range(n_tr):
            weights = np.exp(-D_tr_tr[i, :] ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test with confidence
        for j in range(len(X_te)):
            weights = np.exp(-D_te_tr[j, :] ** 2 / (2 * sigma ** 2))
            test_confidence[te_indices[j]] = weights.sum()  # Higher = more support
            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds, test_confidence


def augment_with_pls_proper(X_train, y_raw_train, pids_train, X_test, pids_test,
                            X_raw_train, X_raw_test, X_pseudo=None, y_pseudo=None,
                            pids_pseudo=None, X_raw_pseudo=None):
    """PLS augmentation. If pseudo data provided, include in fitting but return
    separate arrays for train/test/pseudo."""
    unique_pids = sorted(np.unique(pids_train))
    max_nc = 15

    # Determine sizes
    n_train = len(pids_train)
    n_test = len(pids_test)
    n_pseudo = len(pids_pseudo) if pids_pseudo is not None else 0

    pls_train = np.zeros((n_train, max_nc), dtype=np.float32)
    pls_test = np.zeros((n_test, max_nc), dtype=np.float32)
    pls_pseudo = np.zeros((n_pseudo, max_nc), dtype=np.float32) if n_pseudo > 0 else None

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        # Combine train + pseudo for PLS fitting
        if pids_pseudo is not None:
            ps_mask = pids_pseudo == pid
            combined_raw = np.vstack([X_raw_train[tr_mask], X_raw_pseudo[ps_mask]])
            combined_y = np.concatenate([y_raw_train[tr_mask], y_pseudo[ps_mask]])
        else:
            combined_raw = X_raw_train[tr_mask]
            combined_y = y_raw_train[tr_mask]

        n_p = len(combined_raw)
        scaler = StandardScaler()
        raw_combined = scaler.fit_transform(combined_raw)
        raw_te = scaler.transform(X_raw_test[te_mask])
        raw_tr = raw_combined[:tr_mask.sum()]
        raw_ps = raw_combined[tr_mask.sum():] if n_pseudo > 0 else None

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)
        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15]:
            if c > nc:
                break
            kf = KFold(n_splits=min(5, n_p), shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(raw_combined):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_combined[tr_idx], combined_y[tr_idx])
                pred = pls.predict(raw_combined[val_idx]).flatten()
                mses.append(np.mean((pred - combined_y[val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_combined, combined_y)
        pls_train[tr_mask, :best_nc] = pls.transform(raw_tr)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)
        if pls_pseudo is not None and raw_ps is not None and len(raw_ps) > 0:
            pls_pseudo[ps_mask, :best_nc] = pls.transform(raw_ps)

    result_train = np.hstack([X_train, pls_train])
    result_test = np.hstack([X_test, pls_test])
    if pls_pseudo is not None:
        result_pseudo = np.hstack([X_pseudo, pls_pseudo])
        return result_train, result_test, result_pseudo
    return result_train, result_test


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("PSEUDO-LABELING RIDGE PIPELINE", flush=True)
    print("=" * 70, flush=True)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']

    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    sub_3411 = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")
    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}", flush=True)
        print(f"TARGET: {target.upper()}", flush=True)
        print(f"{'=' * 70}", flush=True)

        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]

        # Extract features
        X_train_base = extract_all_features(train_data, target, kp_index)
        X_test_base = extract_all_features(test_data, target, kp_index)

        # Baseline: no pseudo-labels
        X_tr_aug, X_te_aug = augment_with_pls_proper(
            X_train_base, y_raw, pids_train, X_test_base, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        oof_base, test_base, test_conf = predict_with_confidence(
            X_tr_aug, y_target, X_te_aug, pids_train, pids_test, bandwidth_quantile=0.3)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"  BASELINE CV MSE: {mse_base:.6f}", flush=True)
        print(f"  Test confidence: min={test_conf.min():.3f}, max={test_conf.max():.3f}, mean={test_conf.mean():.3f}", flush=True)

        # Iterative pseudo-labeling
        best_mse = mse_base
        best_test = test_base.copy()
        best_round = "baseline"

        for n_pseudo in [20, 40, 60, 80]:
            print(f"\n  --- Pseudo-labeling top-{n_pseudo} confident test shots ---", flush=True)

            # Sort test shots by confidence (highest first)
            conf_order = np.argsort(-test_conf)
            selected = conf_order[:n_pseudo]

            # Create pseudo-labeled data
            X_pseudo = X_test_base[selected]
            y_pseudo_raw = np.zeros(n_pseudo)
            y_pseudo_scaled = test_base[selected]  # Use baseline predictions as labels
            pids_pseudo = pids_test[selected]
            X_raw_pseudo = test_data['X_raw'][selected]

            # For PLS, we need raw target values
            # Convert scaled predictions back to raw
            y_pseudo_raw = scalers[target].inverse_transform(
                y_pseudo_scaled.reshape(-1, 1)).ravel()

            # Augment with PLS (including pseudo data)
            X_tr_aug2, X_te_aug2, X_ps_aug2 = augment_with_pls_proper(
                X_train_base, y_raw, pids_train, X_test_base, pids_test,
                train_data['X_raw'], test_data['X_raw'],
                X_pseudo, y_pseudo_raw, pids_pseudo, X_raw_pseudo)

            # Combine train + pseudo
            X_combined = np.vstack([X_tr_aug2, X_ps_aug2])
            y_combined = np.concatenate([y_target, y_pseudo_scaled])
            pids_combined = np.concatenate([pids_train, pids_pseudo])

            # Predict with expanded dataset
            oof_pl, test_pl, _ = predict_with_confidence(
                X_combined, y_combined, X_te_aug2, pids_combined, pids_test,
                bandwidth_quantile=0.3)

            # Evaluate on ORIGINAL train only (first 345)
            mse_pl = np.mean((oof_pl[:len(y_target)] - y_target) ** 2)
            delta = (mse_pl - mse_base) / mse_base * 100
            print(f"  N={n_pseudo}: CV MSE={mse_pl:.6f} (delta: {delta:+.2f}%)", flush=True)

            if mse_pl < best_mse:
                best_mse = mse_pl
                best_test = test_pl.copy()
                best_round = f"pseudo_N{n_pseudo}"

            # Round 2: use updated predictions for pseudo-labels
            if n_pseudo <= 40:
                test_conf2 = np.zeros(len(X_te_aug2))
                for pid in sorted(np.unique(pids_combined)):
                    tr_mask = pids_combined == pid
                    te_mask = pids_test == pid
                    X_tr = X_combined[tr_mask]
                    X_te = X_te_aug2[te_mask]
                    te_idx = np.where(te_mask)[0]
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))
                    D = cdist(X_te_s, X_tr_s, metric='euclidean')
                    all_d = cdist(X_tr_s, X_tr_s, metric='euclidean')
                    sigma = np.quantile(all_d[np.triu_indices(len(X_tr_s), k=1)], 0.3)
                    sigma = max(sigma, 1e-6)
                    for j, idx in enumerate(te_idx):
                        w = np.exp(-D[j] ** 2 / (2 * sigma ** 2))
                        test_conf2[idx] = w.sum()

                # Re-select with updated predictions
                selected2 = np.argsort(-test_conf2)[:n_pseudo]
                y_pseudo_scaled2 = test_pl[selected2]
                y_pseudo_raw2 = scalers[target].inverse_transform(y_pseudo_scaled2.reshape(-1, 1)).ravel()

                X_pseudo2 = X_test_base[selected2]
                pids_pseudo2 = pids_test[selected2]
                X_raw_pseudo2 = test_data['X_raw'][selected2]

                X_tr_aug3, X_te_aug3, X_ps_aug3 = augment_with_pls_proper(
                    X_train_base, y_raw, pids_train, X_test_base, pids_test,
                    train_data['X_raw'], test_data['X_raw'],
                    X_pseudo2, y_pseudo_raw2, pids_pseudo2, X_raw_pseudo2)

                X_combined2 = np.vstack([X_tr_aug3, X_ps_aug3])
                y_combined2 = np.concatenate([y_target, y_pseudo_scaled2])
                pids_combined2 = np.concatenate([pids_train, pids_pseudo2])

                oof_pl2, test_pl2, _ = predict_with_confidence(
                    X_combined2, y_combined2, X_te_aug3, pids_combined2, pids_test,
                    bandwidth_quantile=0.3)

                mse_pl2 = np.mean((oof_pl2[:len(y_target)] - y_target) ** 2)
                delta2 = (mse_pl2 - mse_base) / mse_base * 100
                print(f"  N={n_pseudo} (round 2): CV MSE={mse_pl2:.6f} (delta: {delta2:+.2f}%)", flush=True)

                if mse_pl2 < best_mse:
                    best_mse = mse_pl2
                    best_test = test_pl2.copy()
                    best_round = f"pseudo_N{n_pseudo}_r2"

        # Diversity
        col = f'scaled_{target}'
        r = np.corrcoef(sub_3411[col].values, best_test)[0, 1]
        print(f"\n  BEST {target}: {best_round} (MSE={best_mse:.6f})", flush=True)
        print(f"  Diversity vs Sub3411: r={r:.4f}", flush=True)

        results[target] = {
            'best_test': best_test,
            'best_mse': float(best_mse),
            'baseline_mse': float(mse_base),
            'best_round': best_round,
            'diversity_r': float(r),
        }

    # Overall
    print(f"\n{'=' * 70}", flush=True)
    print("OVERALL RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    total = 0
    total_base = 0
    for target in TARGETS:
        total += results[target]['best_mse']
        total_base += results[target]['baseline_mse']
        delta = (results[target]['best_mse'] - results[target]['baseline_mse']) / results[target]['baseline_mse'] * 100
        print(f"  {target}: PL best={results[target]['best_mse']:.6f}, baseline={results[target]['baseline_mse']:.6f} ({delta:+.2f}%) [{results[target]['best_round']}]", flush=True)
    print(f"  MEAN PL: {total/3:.6f}", flush=True)
    print(f"  MEAN BASE: {total_base/3:.6f}", flush=True)

    # Submissions
    print(f"\n{'=' * 70}", flush=True)
    print("GENERATING SUBMISSIONS", flush=True)
    print(f"{'=' * 70}", flush=True)

    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': results['angle']['best_test'],
        'scaled_depth': results['depth']['best_test'],
        'scaled_left_right': results['left_right']['best_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: PSEUDO-LABEL STANDALONE", flush=True)

    for w in [0.03, 0.05, 0.10]:
        sub_num = get_next_submission_number()
        blended = sub_3411.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_3411[col] + w * results[target]['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(w*100)}% PL + {int((1-w)*100)}% Sub3411", flush=True)

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {t: {'best_mse': results[t]['best_mse'],
                        'baseline_mse': results[t]['baseline_mse'],
                        'diversity_r': results[t]['diversity_r'],
                        'best_round': results[t]['best_round']}
                    for t in TARGETS},
        'mean_pl_mse': float(total / 3),
        'mean_base_mse': float(total_base / 3),
    }
    with open(OUTPUT_DIR / "pseudo_labeling_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
