"""
Quick test: Does fixing RP standardization order matter?

Bug: current RP projects raw features, THEN standardizes per-player.
Fix: standardize per-player FIRST, then project.

Also tests: does RP fundamentally help at any configuration?
"""

import json
import time
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
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
TARGETS = ["angle", "depth", "left_right"]
TARGET_IDX = {"angle": 0, "depth": 1, "left_right": 2}


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
    FEET_TO_METERS = 0.3048
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
    lw_idx = kp_index.get('left_wrist')
    if lw_idx is not None and rw is not None:
        feats.append(ts_hr[f, lw_idx, 1] - ts_hr[f, rw, 1])
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
    all_feats, release_frames = [], []
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


def _weighted_ridge_predict(X_tr_s, y_tr, x_query, weights, alpha):
    """Weighted Ridge with proper intercept handling via centering."""
    w_sum = weights.sum()
    X_mean = (weights[:, None] * X_tr_s).sum(axis=0) / w_sum
    y_mean = (weights * y_tr).sum() / w_sum
    X_c = X_tr_s - X_mean
    y_c = y_tr - y_mean
    W_sqrt = np.sqrt(weights)
    Xw = X_c * W_sqrt[:, None]
    yw = y_c * W_sqrt
    d = X_tr_s.shape[1]
    XtX = Xw.T @ Xw + alpha * np.eye(d)
    Xty = Xw.T @ yw
    beta = np.linalg.solve(XtX, Xty)
    return (x_query - X_mean) @ beta + y_mean


def baseline_lw_ridge(X_train, y_train, X_test, pids_train, pids_test,
                      bw=0.5, alpha=10.0):
    """Baseline per-example locally weighted Ridge (uses sklearn for verification)."""
    unique_pids = sorted(np.unique(pids_train))
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask] if np.any(te_mask) else np.zeros((0, X_train.shape[1]))
        n_tr = len(X_tr)
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr = cdist(X_tr_s, X_tr_s)
        dists_upper = D_tr[np.triu_indices(n_tr, k=1)]
        sigma = np.quantile(dists_upper, bw) if len(dists_upper) > 0 else 1.0
        sigma = max(sigma, 1e-6)

        for i in range(n_tr):
            w = np.exp(-D_tr[i] ** 2 / (2 * sigma ** 2))
            w[i] = 0
            if w.sum() < 1e-10:
                oof[tr_idx[i]] = np.mean(y_tr)
                continue
            # Use sklearn Ridge for ground truth comparison
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=w)
            oof[tr_idx[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        if len(X_te) > 0:
            D_te = cdist(X_te_s, X_tr_s)
            for j in range(len(X_te)):
                w = np.exp(-D_te[j] ** 2 / (2 * sigma ** 2))
                if w.sum() < 1e-10:
                    test_preds[te_idx[j]] = np.mean(y_tr)
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_s, y_tr, sample_weight=w)
                test_preds[te_idx[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof, test_preds


def rp_buggy(X_train, y_train, X_test, pids_train, pids_test,
             n_proj=100, proj_dim=20, bw=0.5, alpha=10.0, seed=42):
    """BUGGY: project raw features, then standardize."""
    rng = np.random.RandomState(seed)
    unique_pids = sorted(np.unique(pids_train))
    n_feat = X_train.shape[1]

    all_oof = np.zeros((n_proj, len(X_train)))
    all_test = np.zeros((n_proj, len(X_test)))

    for p in range(n_proj):
        R = rng.randn(n_feat, proj_dim).astype(np.float32) / np.sqrt(proj_dim)
        X_tr_proj = X_train @ R   # Project RAW features
        X_te_proj = X_test @ R

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            X_tr = X_tr_proj[tr_mask]
            y_tr = y_train[tr_mask]
            n_tr = len(X_tr)
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)  # Standardize AFTER projection

            D_tr = cdist(X_tr_s, X_tr_s)
            dists_upper = D_tr[np.triu_indices(n_tr, k=1)]
            sigma = np.quantile(dists_upper, bw) if len(dists_upper) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            for i in range(n_tr):
                w = np.exp(-D_tr[i] ** 2 / (2 * sigma ** 2))
                w[i] = 0
                if w.sum() < 1e-10:
                    all_oof[p, tr_idx[i]] = np.mean(y_tr)
                    continue
                all_oof[p, tr_idx[i]] = _weighted_ridge_predict(
                    X_tr_s, y_tr, X_tr_s[i], w, alpha)

            if np.any(te_mask):
                X_te = X_te_proj[te_mask]
                X_te_s = scaler.transform(X_te)
                D_te = cdist(X_te_s, X_tr_s)
                for j in range(len(X_te)):
                    w = np.exp(-D_te[j] ** 2 / (2 * sigma ** 2))
                    if w.sum() < 1e-10:
                        all_test[p, te_idx[j]] = np.mean(y_tr)
                        continue
                    all_test[p, te_idx[j]] = _weighted_ridge_predict(
                        X_tr_s, y_tr, X_te_s[j], w, alpha)

    return np.mean(all_oof, axis=0), np.mean(all_test, axis=0)


def rp_fixed(X_train, y_train, X_test, pids_train, pids_test,
             n_proj=100, proj_dim=20, bw=0.5, alpha=10.0, seed=42):
    """FIXED: standardize per-player first, then project."""
    rng = np.random.RandomState(seed)
    unique_pids = sorted(np.unique(pids_train))
    n_feat = X_train.shape[1]

    # Pre-standardize per player
    X_train_std = np.zeros_like(X_train)
    X_test_std = np.zeros_like(X_test)
    scalers = {}
    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        scaler = StandardScaler()
        X_train_std[tr_mask] = scaler.fit_transform(X_train[tr_mask])
        if np.any(te_mask):
            X_test_std[te_mask] = scaler.transform(X_test[te_mask])
        scalers[pid] = scaler

    all_oof = np.zeros((n_proj, len(X_train)))
    all_test = np.zeros((n_proj, len(X_test)))

    for p in range(n_proj):
        R = rng.randn(n_feat, proj_dim).astype(np.float32) / np.sqrt(proj_dim)
        X_tr_proj = X_train_std @ R   # Project STANDARDIZED features
        X_te_proj = X_test_std @ R

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            X_tr = X_tr_proj[tr_mask]
            y_tr = y_train[tr_mask]
            n_tr = len(X_tr)
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            # No additional standardization needed - already standardized before projection
            X_tr_s = X_tr
            D_tr = cdist(X_tr_s, X_tr_s)
            dists_upper = D_tr[np.triu_indices(n_tr, k=1)]
            sigma = np.quantile(dists_upper, bw) if len(dists_upper) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            for i in range(n_tr):
                w = np.exp(-D_tr[i] ** 2 / (2 * sigma ** 2))
                w[i] = 0
                if w.sum() < 1e-10:
                    all_oof[p, tr_idx[i]] = np.mean(y_tr)
                    continue
                all_oof[p, tr_idx[i]] = _weighted_ridge_predict(
                    X_tr_s, y_tr, X_tr_s[i], w, alpha)

            if np.any(te_mask):
                X_te = X_te_proj[te_mask]
                X_te_s = X_te
                D_te = cdist(X_te_s, X_tr_s)
                for j in range(len(X_te)):
                    w = np.exp(-D_te[j] ** 2 / (2 * sigma ** 2))
                    if w.sum() < 1e-10:
                        all_test[p, te_idx[j]] = np.mean(y_tr)
                        continue
                    all_test[p, te_idx[j]] = _weighted_ridge_predict(
                        X_tr_s, y_tr, X_te_s[j], w, alpha)

    return np.mean(all_oof, axis=0), np.mean(all_test, axis=0)


def main():
    t0 = time.time()
    print("=" * 70)
    print("RP FIX TEST: Standardization Order")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scalers, y_scaled = {}, {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_scaled[target] = scalers[target].transform(
            y_train[:, TARGET_IDX[target]].reshape(-1, 1)).ravel()

    sub_1828 = pd.read_csv(SUBMISSION_DIR / "submission_1828.csv")

    # Test on angle only first (fastest to diagnose, highest overfit)
    target = "angle"
    print(f"\nTARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")

    print("  Extracting features...")
    X_train_hc, _ = extract_all_features(train_data, target)
    X_test_hc, _ = extract_all_features(test_data, target)
    y_raw = y_train[:, TARGET_IDX[target]]
    X_train_aug, X_test_aug = augment_with_pls(
        X_train_hc, y_raw, pids_train, X_test_hc, pids_test,
        train_data['X_raw'], test_data['X_raw'])
    n_feat = X_train_aug.shape[1]
    y_target = y_scaled[target]
    sub_vals = sub_1828[f'scaled_{target}'].values

    # 1. Baseline (sklearn Ridge)
    print("\n  [BASELINE] sklearn Ridge per-example (bw=0.5, alpha=10)")
    oof_base, test_base = baseline_lw_ridge(
        X_train_aug, y_target, X_test_aug, pids_train, pids_test)
    base_mse = np.mean((oof_base - y_target) ** 2)
    base_r = np.corrcoef(test_base, sub_vals)[0, 1]
    print(f"    LOO MSE={base_mse:.6f}, r(1828)={base_r:.4f}")

    # 2. Verify manual solve matches sklearn
    print("\n  [VERIFY] Manual solve vs sklearn...")
    import sys
    sys.path.insert(0, str(PROJECT_DIR))
    from scripts.novel_overfit_attack import fast_lw_ridge
    oof_manual, test_manual = fast_lw_ridge(
        X_train_aug, y_target, X_test_aug, pids_train, pids_test)
    manual_mse = np.mean((oof_manual - y_target) ** 2)
    diff = np.max(np.abs(oof_base - oof_manual))
    print(f"    Manual LOO MSE={manual_mse:.6f}")
    print(f"    Max |sklearn - manual| LOO = {diff:.8f}")
    diff_test = np.max(np.abs(test_base - test_manual))
    print(f"    Max |sklearn - manual| test = {diff_test:.8f}")

    # 3. Buggy RP (project raw, then standardize)
    for n_proj, proj_dim, alpha in [(100, 20, 10.0), (100, 20, 50.0)]:
        print(f"\n  [RP BUGGY] n={n_proj}, d={proj_dim}, alpha={alpha}")
        oof_bug, test_bug = rp_buggy(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            n_proj=n_proj, proj_dim=proj_dim, alpha=alpha)
        bug_mse = np.mean((oof_bug - y_target) ** 2)
        bug_r = np.corrcoef(test_bug, sub_vals)[0, 1]
        print(f"    LOO MSE={bug_mse:.6f}, r(1828)={bug_r:.4f}")

    # 4. Fixed RP (standardize first, then project)
    for n_proj, proj_dim, alpha in [(100, 20, 10.0), (100, 20, 50.0)]:
        print(f"\n  [RP FIXED] n={n_proj}, d={proj_dim}, alpha={alpha}")
        oof_fix, test_fix = rp_fixed(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test,
            n_proj=n_proj, proj_dim=proj_dim, alpha=alpha)
        fix_mse = np.mean((oof_fix - y_target) ** 2)
        fix_r = np.corrcoef(test_fix, sub_vals)[0, 1]
        print(f"    LOO MSE={fix_mse:.6f}, r(1828)={fix_r:.4f}")

    # 5. Compare test predictions between buggy and fixed
    print(f"\n  [COMPARISON] Buggy vs Fixed RP (d=20, alpha=10, n=100)")
    oof_bug100, test_bug100 = rp_buggy(
        X_train_aug, y_target, X_test_aug, pids_train, pids_test,
        n_proj=100, proj_dim=20, alpha=10.0, seed=42)
    oof_fix100, test_fix100 = rp_fixed(
        X_train_aug, y_target, X_test_aug, pids_train, pids_test,
        n_proj=100, proj_dim=20, alpha=10.0, seed=42)

    r_bug_fix_oof = np.corrcoef(oof_bug100, oof_fix100)[0, 1]
    r_bug_fix_test = np.corrcoef(test_bug100, test_fix100)[0, 1]
    print(f"    r(buggy, fixed) LOO:  {r_bug_fix_oof:.6f}")
    print(f"    r(buggy, fixed) test: {r_bug_fix_test:.6f}")
    print(f"    Buggy  LOO MSE: {np.mean((oof_bug100 - y_target)**2):.6f}")
    print(f"    Fixed  LOO MSE: {np.mean((oof_fix100 - y_target)**2):.6f}")
    print(f"    Buggy  test mean: {test_bug100.mean():.6f}, std: {test_bug100.std():.6f}")
    print(f"    Fixed  test mean: {test_fix100.mean():.6f}, std: {test_fix100.std():.6f}")

    # 6. Key diagnostic: compute EFFECTIVE overfit ratio for RP
    # LOO on RP is misleadingly low. Test a 2-fold within-player estimate.
    print(f"\n  [DIAGNOSTIC] 2-fold cross-validation for honest error estimate")
    unique_pids = sorted(np.unique(pids_train))
    for label, rp_func in [("buggy", rp_buggy), ("fixed", rp_fixed)]:
        cv_oof = np.zeros(len(X_train_aug))
        for pid in unique_pids:
            pid_idx = np.where(pids_train == pid)[0]
            rng = np.random.RandomState(42)
            rng.shuffle(pid_idx)
            half = len(pid_idx) // 2
            folds = [pid_idx[:half], pid_idx[half:]]
            for fold_id in range(2):
                val_idx = folds[fold_id]
                tr_idx = folds[1 - fold_id]
                X_fold_tr = X_train_aug[tr_idx]
                y_fold_tr = y_target[tr_idx]
                X_fold_val = X_train_aug[val_idx]
                pids_fold_tr = pids_train[tr_idx]
                pids_fold_val = pids_train[val_idx]
                _, val_preds = rp_func(
                    X_fold_tr, y_fold_tr, X_fold_val,
                    pids_fold_tr, pids_fold_val,
                    n_proj=100, proj_dim=20, alpha=10.0)
                cv_oof[val_idx] = val_preds
        cv_mse = np.mean((cv_oof - y_target) ** 2)
        loo_mse_rp = np.mean((oof_bug100 if label == "buggy" else oof_fix100 - y_target) ** 2)
        print(f"    RP {label}: 2-fold CV MSE={cv_mse:.6f}, LOO MSE={loo_mse_rp:.6f}, ratio={cv_mse/max(loo_mse_rp, 1e-10):.2f}x")

    # Also do 2-fold for baseline
    cv_oof_base = np.zeros(len(X_train_aug))
    for pid in unique_pids:
        pid_idx = np.where(pids_train == pid)[0]
        rng = np.random.RandomState(42)
        rng.shuffle(pid_idx)
        half = len(pid_idx) // 2
        folds = [pid_idx[:half], pid_idx[half:]]
        for fold_id in range(2):
            val_idx = folds[fold_id]
            tr_idx = folds[1 - fold_id]
            X_fold_tr = X_train_aug[tr_idx]
            y_fold_tr = y_target[tr_idx]
            X_fold_val = X_train_aug[val_idx]
            pids_fold_tr = pids_train[tr_idx]
            pids_fold_val = pids_train[val_idx]

            for pid2 in sorted(np.unique(pids_fold_tr)):
                ptr = pids_fold_tr == pid2
                pval = pids_fold_val == pid2
                if not np.any(pval):
                    continue
                sc = StandardScaler()
                xtr = sc.fit_transform(X_fold_tr[ptr])
                xval = sc.transform(X_fold_val[pval])
                ytr = y_fold_tr[ptr]
                D = cdist(xtr, xtr)
                dists_upper = D[np.triu_indices(len(xtr), k=1)]
                sigma = np.quantile(dists_upper, 0.5) if len(dists_upper) > 0 else 1.0
                sigma = max(sigma, 1e-6)
                Dv = cdist(xval, xtr)
                global_val = val_idx[pval]
                for j in range(len(xval)):
                    w = np.exp(-Dv[j] ** 2 / (2 * sigma ** 2))
                    if w.sum() < 1e-10:
                        cv_oof_base[global_val[j]] = np.mean(ytr)
                        continue
                    cv_oof_base[global_val[j]] = _weighted_ridge_predict(
                        xtr, ytr, xval[j], w, 10.0)

    cv_mse_base = np.mean((cv_oof_base - y_target) ** 2)
    print(f"    Baseline: 2-fold CV MSE={cv_mse_base:.6f}, LOO MSE={base_mse:.6f}, ratio={cv_mse_base/base_mse:.2f}x")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
