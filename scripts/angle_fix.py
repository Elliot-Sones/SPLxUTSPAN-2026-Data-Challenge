"""
Angle Fix Pipeline

FINDING: Angle overfits 2.57x (LOO -> LB) vs 1.56x for depth/LR.
Sub 784's angle (per-player tree ensemble) is the hidden bottleneck.
Per-example Ridge (used for depth/LR) overfits less.

Strategy:
1. Generate per-example angle predictions with various regularization
2. Test heavy regularization (angle needs it more than depth/LR)
3. Blend angle improvements into Sub 1640 (keep depth/LR unchanged)
4. Try LASSO feature selection for angle
5. Try different bandwidths (wider = less overfit)
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
FEET_TO_METERS = 0.3048
ANGLE_FRAME = 153


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


def extract_all_features(data):
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, ANGLE_FRAME)
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


def per_example_angle(X_train, y_train, X_test, pids_train, pids_test,
                      bandwidth_quantile=0.5, alpha=10.0, feature_select=None):
    """Per-example locally weighted Ridge for angle prediction.

    feature_select: if set, use LASSO to select this many features first.
    """
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

        # Optional LASSO feature selection
        if feature_select is not None and feature_select < X_tr_s.shape[1]:
            lasso = Lasso(alpha=0.001, max_iter=5000)
            lasso.fit(X_tr_s, y_tr)
            importances = np.abs(lasso.coef_)
            top_k = np.argsort(importances)[-feature_select:]
            X_tr_s = X_tr_s[:, top_k]
            X_te_s = X_te_s[:, top_k]

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


def global_ridge_angle(X_train, y_train, X_test, pids_train, pids_test, alpha=100.0):
    """Simple per-player global Ridge - maximum regularization, minimum overfit."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        # LOO for OOF
        for i in range(len(X_tr)):
            mask = np.ones(len(X_tr), dtype=bool)
            mask[i] = False
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s[mask], y_tr[mask])
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        # Test predictions
        if np.any(te_mask):
            X_te_s = scaler.transform(X_test[te_mask])
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr)
            test_preds[te_indices] = ridge.predict(X_te_s)

    return oof_preds, test_preds


def main():
    t0 = time.time()
    print("=" * 70)
    print("ANGLE FIX PIPELINE")
    print("=" * 70)
    print("  Finding: Angle overfits 2.57x vs 1.56x for depth/LR")
    print("  Goal: Reduce angle overfitting to unlock LB improvement")

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scaler_angle = joblib.load(DATA_DIR / "scaler_angle.pkl")
    y_angle_scaled = scaler_angle.transform(y_train[:, 0].reshape(-1, 1)).ravel()
    y_angle_raw = y_train[:, 0]

    # Extract features at angle frame (153)
    X_train_hc, _ = extract_all_features(train_data)
    X_test_hc, _ = extract_all_features(test_data)

    # Augment with PLS
    X_train_aug, X_test_aug = augment_with_pls(
        X_train_hc, y_angle_raw, pids_train,
        X_test_hc, pids_test,
        train_data['X_raw'], test_data['X_raw'])
    print(f"  Features: {X_train_aug.shape[1]}")

    sub_1640 = pd.read_csv(SUBMISSION_DIR / "submission_1640.csv")

    # ============================================================
    # TEST MANY ANGLE CONFIGURATIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("ANGLE CONFIGURATION SWEEP")
    print(f"{'=' * 70}")

    configs = [
        # (name, bandwidth, alpha, feature_select)
        # Current Sub 1350 settings (what was used for depth/LR)
        ("baseline_bw050_a10", 0.50, 10.0, None),
        # Higher regularization (reduce overfit)
        ("high_reg_bw050_a50", 0.50, 50.0, None),
        ("high_reg_bw050_a100", 0.50, 100.0, None),
        ("high_reg_bw050_a200", 0.50, 200.0, None),
        ("high_reg_bw050_a500", 0.50, 500.0, None),
        # Wider bandwidth (smoother kernel = less overfit)
        ("wide_bw070_a10", 0.70, 10.0, None),
        ("wide_bw070_a50", 0.70, 50.0, None),
        ("wide_bw070_a100", 0.70, 100.0, None),
        ("wide_bw080_a50", 0.80, 50.0, None),
        ("wide_bw090_a50", 0.90, 50.0, None),
        # Very wide bandwidth (almost global model)
        ("vwide_bw095_a100", 0.95, 100.0, None),
        # Feature selection (reduce dimensionality = less overfit)
        ("lasso30_bw050_a10", 0.50, 10.0, 30),
        ("lasso30_bw050_a50", 0.50, 50.0, 30),
        ("lasso30_bw070_a50", 0.70, 50.0, 30),
        ("lasso15_bw050_a50", 0.50, 50.0, 15),
        ("lasso15_bw070_a100", 0.70, 100.0, 15),
        # Extreme regularization combos
        ("extreme_bw080_a200", 0.80, 200.0, None),
        ("extreme_bw080_a200_l30", 0.80, 200.0, 30),
    ]

    results = {}
    for name, bw, alpha, fs in configs:
        oof, test = per_example_angle(
            X_train_aug, y_angle_scaled, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=bw, alpha=alpha, feature_select=fs)
        mse = np.mean((oof - y_angle_scaled) ** 2)
        r_1640 = np.corrcoef(test, sub_1640['scaled_angle'].values)[0, 1]
        results[name] = {'mse': mse, 'oof': oof, 'test': test, 'r_1640': r_1640}
        print(f"  {name:35s}: LOO MSE={mse:.6f}, r(Sub1640)={r_1640:.4f}")

    # Also test global Ridge (no local weighting - maximum simplicity)
    for alpha in [10, 50, 100, 200, 500, 1000]:
        name = f"global_ridge_a{alpha}"
        oof, test = global_ridge_angle(
            X_train_aug, y_angle_scaled, X_test_aug, pids_train, pids_test,
            alpha=alpha)
        mse = np.mean((oof - y_angle_scaled) ** 2)
        r_1640 = np.corrcoef(test, sub_1640['scaled_angle'].values)[0, 1]
        results[name] = {'mse': mse, 'oof': oof, 'test': test, 'r_1640': r_1640}
        print(f"  {name:35s}: LOO MSE={mse:.6f}, r(Sub1640)={r_1640:.4f}")

    # Multi-config average (reduce variance)
    print(f"\n  Multi-config averages:")
    # Average of high-reg configs
    high_reg_names = [n for n in results if 'high_reg' in n or 'wide' in n]
    if len(high_reg_names) >= 3:
        avg_oof = np.mean([results[n]['oof'] for n in high_reg_names], axis=0)
        avg_test = np.mean([results[n]['test'] for n in high_reg_names], axis=0)
        mse = np.mean((avg_oof - y_angle_scaled) ** 2)
        r_1640 = np.corrcoef(avg_test, sub_1640['scaled_angle'].values)[0, 1]
        results['avg_high_reg'] = {'mse': mse, 'oof': avg_oof, 'test': avg_test, 'r_1640': r_1640}
        print(f"  {'avg_high_reg':35s}: LOO MSE={mse:.6f}, r(Sub1640)={r_1640:.4f}")

    # Average of global Ridge configs
    global_names = [n for n in results if 'global_ridge' in n]
    if len(global_names) >= 3:
        avg_oof = np.mean([results[n]['oof'] for n in global_names], axis=0)
        avg_test = np.mean([results[n]['test'] for n in global_names], axis=0)
        mse = np.mean((avg_oof - y_angle_scaled) ** 2)
        r_1640 = np.corrcoef(avg_test, sub_1640['scaled_angle'].values)[0, 1]
        results['avg_global_ridge'] = {'mse': mse, 'oof': avg_oof, 'test': avg_test, 'r_1640': r_1640}
        print(f"  {'avg_global_ridge':35s}: LOO MSE={mse:.6f}, r(Sub1640)={r_1640:.4f}")

    # Average ALL configs
    all_names = list(results.keys())
    avg_oof = np.mean([results[n]['oof'] for n in all_names], axis=0)
    avg_test = np.mean([results[n]['test'] for n in all_names], axis=0)
    mse = np.mean((avg_oof - y_angle_scaled) ** 2)
    r_1640 = np.corrcoef(avg_test, sub_1640['scaled_angle'].values)[0, 1]
    results['avg_all'] = {'mse': mse, 'oof': avg_oof, 'test': avg_test, 'r_1640': r_1640}
    print(f"  {'avg_all':35s}: LOO MSE={mse:.6f}, r(Sub1640)={r_1640:.4f}")

    # ============================================================
    # RANKING
    # ============================================================
    print(f"\n{'=' * 70}")
    print("RANKING BY LOO MSE (lower = better, but might overfit)")
    print(f"{'=' * 70}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['mse'])
    for rank, (name, r) in enumerate(sorted_results):
        print(f"  {rank+1:2d}. {name:35s}: LOO={r['mse']:.6f}, r(1640)={r['r_1640']:.4f}")

    # The key insight: LOW LOO MSE = overfit. For angle, we want HIGHER LOO MSE
    # because our problem is overfitting (2.57x ratio).
    # The configs with HIGHEST LOO MSE but still reasonable are likely best on test.
    print(f"\n{'=' * 70}")
    print("RANKING BY LOO MSE (higher = more regularized, less overfit)")
    print(f"{'=' * 70}")
    for rank, (name, r) in enumerate(reversed(sorted_results)):
        print(f"  {rank+1:2d}. {name:35s}: LOO={r['mse']:.6f}, r(1640)={r['r_1640']:.4f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING ANGLE-FIX SUBMISSIONS")
    print(f"{'=' * 70}")

    # Select configs to test:
    # 1. Most regularized per-example (widest bandwidth, highest alpha)
    # 2. Global Ridge (simplest model)
    # 3. LASSO-selected features with high reg
    # 4. Multi-config average (reduce variance)
    # 5. Moderate regularization (sweet spot?)

    candidates = [
        "baseline_bw050_a10",        # What Sub 1350 uses (for reference)
        "high_reg_bw050_a100",       # Same bandwidth, 10x more regularization
        "wide_bw070_a50",            # Wider bandwidth + moderate reg
        "wide_bw080_a50",            # Even wider bandwidth
        "extreme_bw080_a200",        # Extreme regularization
        "lasso30_bw070_a50",         # Feature selection + wide bandwidth
        "lasso15_bw070_a100",        # Aggressive feature selection
        "global_ridge_a100",         # Pure global Ridge
        "global_ridge_a500",         # Very heavy global Ridge
        "avg_high_reg",              # Average of regularized configs
        "avg_global_ridge",          # Average of global Ridge configs
        "avg_all",                   # Average of everything
    ]

    candidates = [c for c in candidates if c in results]

    # For each candidate, blend ONLY the angle into Sub 1640
    # (keep depth and LR from Sub 1640 unchanged)
    for cand_name in candidates:
        test_angle = results[cand_name]['test']
        loo_mse = results[cand_name]['mse']

        for aw in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
            sub_num = get_next_submission_number()
            blended = sub_1640.copy()
            # ONLY change angle
            blended['scaled_angle'] = (1 - aw) * sub_1640['scaled_angle'] + aw * test_angle
            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)

            if aw in [0.30, 1.00]:  # Print only key weights
                print(f"  Sub {sub_num}: aw={aw:.2f} {cand_name} (LOO={loo_mse:.6f})")

    # Special: create submission with averaged angle from multiple regularized configs
    # This is the "safest" bet - averaging reduces both bias and variance
    safe_configs = ["wide_bw070_a50", "wide_bw080_a50", "high_reg_bw050_a100",
                    "global_ridge_a100", "global_ridge_a500", "lasso30_bw070_a50"]
    safe_configs = [c for c in safe_configs if c in results]

    avg_safe_test = np.mean([results[c]['test'] for c in safe_configs], axis=0)
    avg_safe_oof = np.mean([results[c]['oof'] for c in safe_configs], axis=0)
    avg_safe_mse = np.mean((avg_safe_oof - y_angle_scaled) ** 2)

    print(f"\n  Safe average ({len(safe_configs)} configs): LOO MSE={avg_safe_mse:.6f}")

    for aw in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
        sub_num = get_next_submission_number()
        blended = sub_1640.copy()
        blended['scaled_angle'] = (1 - aw) * sub_1640['scaled_angle'] + aw * avg_safe_test
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: aw={aw:.2f} safe_average (depth+LR from Sub 1640)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
