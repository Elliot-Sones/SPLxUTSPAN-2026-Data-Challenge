"""
Power Transform + Per-Example Regression

Theory: Ridge regression assumes Gaussian features. Our HC features may be
skewed. Yeo-Johnson power transforms can make features more Gaussian,
potentially improving Ridge performance.

Approach:
1. Apply Yeo-Johnson transform to all features per-player
2. Run per-example locally weighted Ridge on transformed features
3. Also test: original + transformed features (doubled, then LASSO select)
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
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
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


def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
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


def main():
    t0 = time.time()
    print("=" * 70)
    print("POWER TRANSFORM + PER-EXAMPLE REGRESSION")
    print("=" * 70)

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

    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
    sub_1640 = pd.read_csv(SUBMISSION_DIR / "submission_1640.csv")

    baseline_mse = {'angle': 0.002511, 'depth': 0.004510, 'left_right': 0.004209}

    all_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]

        # Standard features
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        n_feat = X_train_aug.shape[1]
        print(f"  Standard features: {n_feat}")

        # Baseline
        oof_baseline, test_baseline = locally_weighted_prediction(
            X_train_aug, y_target, X_test_aug, pids_train, pids_test)
        mse_baseline = np.mean((oof_baseline - y_target) ** 2)
        print(f"  Baseline LOO MSE: {mse_baseline:.6f}")

        configs = {}
        configs['baseline'] = (mse_baseline, oof_baseline, test_baseline)

        # ================================================================
        # CONFIG 1: Yeo-Johnson power transform
        # ================================================================
        print("\n  Config 1: Yeo-Johnson power transform...")
        unique_pids = sorted(np.unique(pids_train))
        X_tr_yj = np.zeros_like(X_train_aug)
        X_te_yj = np.zeros_like(X_test_aug)
        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            # Fit Yeo-Johnson per-player
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            X_tr_yj[tr_mask] = pt.fit_transform(X_train_aug[tr_mask])
            X_te_yj[te_mask] = pt.transform(X_test_aug[te_mask])
        X_tr_yj = np.nan_to_num(X_tr_yj, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_yj = np.nan_to_num(X_te_yj, nan=0.0, posinf=0.0, neginf=0.0)

        oof_yj, test_yj = locally_weighted_prediction(
            X_tr_yj, y_target, X_te_yj, pids_train, pids_test)
        mse_yj = np.mean((oof_yj - y_target) ** 2)
        delta_yj = (mse_yj - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_yj:.6f} ({delta_yj:+.1f}% vs baseline)")
        configs['yeo_johnson'] = (mse_yj, oof_yj, test_yj)

        # ================================================================
        # CONFIG 2: Quantile transform (uniform -> Gaussian)
        # ================================================================
        print("\n  Config 2: Quantile transform (to Gaussian)...")
        X_tr_qt = np.zeros_like(X_train_aug)
        X_te_qt = np.zeros_like(X_test_aug)
        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            qt = QuantileTransformer(output_distribution='normal', random_state=42,
                                     n_quantiles=min(50, tr_mask.sum()))
            X_tr_qt[tr_mask] = qt.fit_transform(X_train_aug[tr_mask])
            X_te_qt[te_mask] = qt.transform(X_test_aug[te_mask])
        X_tr_qt = np.nan_to_num(X_tr_qt, nan=0.0, posinf=0.0, neginf=0.0)
        X_te_qt = np.nan_to_num(X_te_qt, nan=0.0, posinf=0.0, neginf=0.0)

        oof_qt, test_qt = locally_weighted_prediction(
            X_tr_qt, y_target, X_te_qt, pids_train, pids_test)
        mse_qt = np.mean((oof_qt - y_target) ** 2)
        delta_qt = (mse_qt - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_qt:.6f} ({delta_qt:+.1f}% vs baseline)")
        configs['quantile_gaussian'] = (mse_qt, oof_qt, test_qt)

        # ================================================================
        # CONFIG 3: Original + Yeo-Johnson concatenated, LASSO select 30
        # ================================================================
        print("\n  Config 3: Original + Yeo-Johnson + LASSO select...")
        X_tr_combo = np.hstack([X_train_aug, X_tr_yj])
        X_te_combo = np.hstack([X_test_aug, X_te_yj])
        print(f"  Combined features: {X_tr_combo.shape[1]}")

        # LASSO stability selection
        stability_scores = np.zeros(X_tr_combo.shape[1])
        for pid in unique_pids:
            mask = pids_train == pid
            sc = StandardScaler()
            X_s = sc.fit_transform(X_tr_combo[mask])
            for b in range(30):
                rng = np.random.RandomState(b)
                idx = rng.choice(len(X_s), len(X_s), replace=True)
                lasso = Lasso(alpha=0.01, max_iter=5000)
                lasso.fit(X_s[idx], y_target[mask][idx])
                stability_scores += (np.abs(lasso.coef_) > 1e-6).astype(float)
        stability_scores /= (30 * len(unique_pids))
        top30 = np.argsort(-stability_scores)[:30]

        X_tr_sel = X_tr_combo[:, top30]
        X_te_sel = X_te_combo[:, top30]
        # Count how many come from transformed space
        n_from_orig = np.sum(top30 < n_feat)
        n_from_yj = np.sum(top30 >= n_feat)
        print(f"  Selected: {n_from_orig} original + {n_from_yj} Yeo-Johnson")

        oof_combo, test_combo = locally_weighted_prediction(
            X_tr_sel, y_target, X_te_sel, pids_train, pids_test)
        mse_combo = np.mean((oof_combo - y_target) ** 2)
        delta_combo = (mse_combo - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_combo:.6f} ({delta_combo:+.1f}% vs baseline)")
        configs['combo_lasso30'] = (mse_combo, oof_combo, test_combo)

        # ================================================================
        # CONFIG 4: Smoothed target encoding (player-level features)
        # ================================================================
        print("\n  Config 4: Smoothed target encoding...")
        smoothing_m = 10
        global_mean = np.mean(y_target)
        X_tr_te = np.zeros((len(pids_train), 1), dtype=np.float32)
        X_te_te = np.zeros((len(pids_test), 1), dtype=np.float32)

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            # LOO encoding for training
            for i in np.where(tr_mask)[0]:
                other = y_target[tr_mask & (np.arange(len(y_target)) != i)]
                count = len(other)
                player_mean = np.mean(other) if count > 0 else global_mean
                X_tr_te[i, 0] = (count * player_mean + smoothing_m * global_mean) / (count + smoothing_m)
            # Full encoding for test
            tr_vals = y_target[tr_mask]
            count = len(tr_vals)
            player_mean = np.mean(tr_vals)
            X_te_te[te_mask, 0] = (count * player_mean + smoothing_m * global_mean) / (count + smoothing_m)

        X_tr_with_te = np.hstack([X_train_aug, X_tr_te])
        X_te_with_te = np.hstack([X_test_aug, X_te_te])

        oof_te, test_te = locally_weighted_prediction(
            X_tr_with_te, y_target, X_te_with_te, pids_train, pids_test)
        mse_te = np.mean((oof_te - y_target) ** 2)
        delta_te = (mse_te - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_te:.6f} ({delta_te:+.1f}% vs baseline)")
        configs['target_encoding'] = (mse_te, oof_te, test_te)

        # ================================================================
        # CONFIG 5: Yeo-Johnson + higher alpha (more regularization)
        # ================================================================
        print("\n  Config 5: Yeo-Johnson + alpha=50...")
        oof_yj50, test_yj50 = locally_weighted_prediction(
            X_tr_yj, y_target, X_te_yj, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=50.0)
        mse_yj50 = np.mean((oof_yj50 - y_target) ** 2)
        delta_yj50 = (mse_yj50 - mse_baseline) / mse_baseline * 100
        print(f"  LOO MSE: {mse_yj50:.6f} ({delta_yj50:+.1f}% vs baseline)")
        configs['yeo_johnson_alpha50'] = (mse_yj50, oof_yj50, test_yj50)

        # Summary
        print(f"\n  {target.upper()} SUMMARY:")
        for name, (mse, _, _) in sorted(configs.items(), key=lambda x: x[1][0]):
            delta = (mse - mse_baseline) / mse_baseline * 100
            marker = " <-- BEST" if mse == min(v[0] for v in configs.values()) else ""
            print(f"    {name}: {mse:.6f} ({delta:+.1f}%){marker}")

        best_name = min(configs, key=lambda k: configs[k][0])
        all_results[target] = {
            'configs': configs,
            'best_name': best_name,
            'best': configs[best_name],
        }

    # ================================================================
    # GENERATE SUBMISSIONS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Best per target
    best_test = {}
    for target in TARGETS:
        best_name = all_results[target]['best_name']
        best_test[target] = all_results[target]['best'][2]
        print(f"  {target}: {best_name}")

    # Standalone
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': best_test['angle'],
        'scaled_depth': best_test['depth'],
        'scaled_left_right': best_test['left_right'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: STANDALONE best-per-target power transform")

    # Blends with Sub 1350
    for pct in [0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for col, t in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[col] = (1-pct) * sub_1350[col] + pct * best_test[t]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(pct*100)}% power_transform + {int((1-pct)*100)}% Sub 1350")

    # Blends with Sub 1640
    for pct in [0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_1640.copy()
        for col, t in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[col] = (1-pct) * sub_1640[col] + pct * best_test[t]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(pct*100)}% power_transform + {int((1-pct)*100)}% Sub 1640")

    # Diversity analysis
    print(f"\n  Diversity with Sub 1350:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_1350[col].values, best_test[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    print(f"\n  Diversity with Sub 1640:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r = np.corrcoef(sub_1640[col].values, best_test[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
