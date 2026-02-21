"""
Deep Feature Selection Exploration

Since LASSO stability selection (213->30 features) improved LB from 0.006776 to 0.006698,
explore this direction thoroughly:

1. Different feature counts (15, 20, 30, 40, 50, 80)
2. Per-target feature selection (different features for each target)
3. Different selection methods (LASSO, mutual information, correlation)
4. Test blends with Sub 1350 at multiple percentages
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
from sklearn.feature_selection import mutual_info_regression
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


# ================================================================
# FEATURE SELECTION METHODS
# ================================================================

def select_lasso_stability(X, y, pids, n_features, n_bootstrap=30):
    """LASSO stability selection (same as limiting_factors_fixes.py)."""
    unique_pids = sorted(np.unique(pids))
    selected_features = np.zeros(X.shape[1])
    for pid in unique_pids:
        mask = pids == pid
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X[mask])
        for b in range(n_bootstrap):
            rng = np.random.RandomState(b)
            idx = rng.choice(len(X_s), len(X_s), replace=True)
            lasso = Lasso(alpha=0.01, max_iter=5000)
            lasso.fit(X_s[idx], y[mask][idx])
            selected_features += (np.abs(lasso.coef_) > 1e-6).astype(float)
    stability_scores = selected_features / (n_bootstrap * len(unique_pids))
    top_idx = np.argsort(-stability_scores)[:n_features]
    return top_idx, stability_scores


def select_correlation(X, y, pids, n_features):
    """Select features with highest absolute correlation to target."""
    unique_pids = sorted(np.unique(pids))
    # Per-player correlation, then average
    corrs = np.zeros(X.shape[1])
    for pid in unique_pids:
        mask = pids == pid
        for j in range(X.shape[1]):
            r = np.corrcoef(X[mask, j], y[mask])[0, 1]
            if np.isnan(r):
                r = 0
            corrs[j] += abs(r)
    corrs /= len(unique_pids)
    top_idx = np.argsort(-corrs)[:n_features]
    return top_idx, corrs


def select_mutual_info(X, y, pids, n_features):
    """Select features with highest mutual information with target."""
    unique_pids = sorted(np.unique(pids))
    mi_scores = np.zeros(X.shape[1])
    for pid in unique_pids:
        mask = pids == pid
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X[mask])
        mi = mutual_info_regression(X_s, y[mask], random_state=42, n_neighbors=5)
        mi_scores += mi
    mi_scores /= len(unique_pids)
    top_idx = np.argsort(-mi_scores)[:n_features]
    return top_idx, mi_scores


def select_lasso_path(X, y, pids, n_features):
    """Select features that LASSO keeps at different alpha values.
    More robust than single alpha."""
    unique_pids = sorted(np.unique(pids))
    keep_count = np.zeros(X.shape[1])
    alphas = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    for pid in unique_pids:
        mask = pids == pid
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X[mask])
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=5000)
            lasso.fit(X_s, y[mask])
            keep_count += (np.abs(lasso.coef_) > 1e-6).astype(float)
    keep_count /= (len(unique_pids) * len(alphas))
    top_idx = np.argsort(-keep_count)[:n_features]
    return top_idx, keep_count


def main():
    t0 = time.time()
    print("=" * 70)
    print("DEEP FEATURE SELECTION EXPLORATION")
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
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Baseline LOO MSE (from Sub 1350 pipeline)
    baseline_mse = {'angle': 0.002511, 'depth': 0.004510, 'left_right': 0.004209}

    all_results = {}  # {target: {config_name: {mse, oof, test}}}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]

        # Extract full features (same as Sub 1350)
        X_train_hc, _ = extract_all_features(train_data, target)
        X_test_hc, _ = extract_all_features(test_data, target)
        X_train_aug, X_test_aug = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        n_feat = X_train_aug.shape[1]
        print(f"  Full features: {n_feat}")

        target_results = {}

        # Selection methods
        methods = {
            'lasso_stab': lambda X, y, p, n: select_lasso_stability(X, y, p, n),
            'correlation': lambda X, y, p, n: select_correlation(X, y, p, n),
            'mutual_info': lambda X, y, p, n: select_mutual_info(X, y, p, n),
            'lasso_path': lambda X, y, p, n: select_lasso_path(X, y, p, n),
        }

        # Feature counts to try
        n_features_list = [15, 20, 30, 40, 50, 80]

        for method_name, selector in methods.items():
            print(f"\n  Method: {method_name}")
            for n_feat_sel in n_features_list:
                config_name = f"{method_name}_{n_feat_sel}"
                idx, scores = selector(X_train_aug, y_target, pids_train, n_feat_sel)

                X_tr_sel = X_train_aug[:, idx]
                X_te_sel = X_test_aug[:, idx]

                oof, test = locally_weighted_prediction(
                    X_tr_sel, y_target, X_te_sel, pids_train, pids_test,
                    bandwidth_quantile=0.5, alpha=10.0)
                mse = np.mean((oof - y_target) ** 2)
                delta = (mse - baseline_mse[target]) / baseline_mse[target] * 100

                r_1350 = np.corrcoef(sub_1350[f'scaled_{target}'].values, test)[0, 1]

                print(f"    n={n_feat_sel}: LOO MSE={mse:.6f} ({delta:+.1f}%), r_1350={r_1350:.4f}")
                target_results[config_name] = {'mse': mse, 'oof': oof, 'test': test, 'r_1350': r_1350}

        all_results[target] = target_results

    # ================================================================
    # FIND BEST PER-TARGET CONFIGURATION
    # ================================================================
    print(f"\n{'=' * 70}")
    print("BEST CONFIGURATIONS PER TARGET")
    print(f"{'=' * 70}")

    best_per_target = {}
    for target in TARGETS:
        results = all_results[target]
        best_name = min(results, key=lambda k: results[k]['mse'])
        best = results[best_name]
        delta = (best['mse'] - baseline_mse[target]) / baseline_mse[target] * 100
        print(f"  {target}: {best_name} MSE={best['mse']:.6f} ({delta:+.1f}%), r_1350={best['r_1350']:.4f}")
        best_per_target[target] = best_name

    # Mean across all configs
    print(f"\n  TOP 10 CONFIGS BY MEAN MSE:")
    all_configs = set()
    for target in TARGETS:
        all_configs |= set(all_results[target].keys())

    config_means = {}
    for config in sorted(all_configs):
        if all(config in all_results[t] for t in TARGETS):
            mean_mse = np.mean([all_results[t][config]['mse'] for t in TARGETS])
            config_means[config] = mean_mse

    base_mean = np.mean(list(baseline_mse.values()))
    for i, (config, mean_mse) in enumerate(sorted(config_means.items(), key=lambda x: x[1])[:10]):
        delta = (mean_mse - base_mean) / base_mean * 100
        print(f"    {i+1}. {config}: {mean_mse:.6f} ({delta:+.1f}%)")

    # ================================================================
    # GENERATE SUBMISSIONS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Best uniform config (same selection for all targets)
    best_uniform = min(config_means, key=config_means.get)
    print(f"\n  Best uniform: {best_uniform}")

    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': all_results['angle'][best_uniform]['test'],
        'scaled_depth': all_results['depth'][best_uniform]['test'],
        'scaled_left_right': all_results['left_right'][best_uniform]['test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: {best_uniform} standalone")

    # Blend with Sub 1350 at various percentages
    for pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for col, t in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            blended[col] = (1-pct) * sub_1350[col] + pct * all_results[t][best_uniform]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(pct*100)}% {best_uniform} + {int((1-pct)*100)}% Sub 1350")

    # Best per-target config
    print(f"\n  Best per-target:")
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': all_results['angle'][best_per_target['angle']]['test'],
        'scaled_depth': all_results['depth'][best_per_target['depth']]['test'],
        'scaled_left_right': all_results['left_right'][best_per_target['left_right']]['test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: best per-target standalone")

    # Blend best per-target with Sub 1350
    for pct in [0.05, 0.10, 0.15, 0.20]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for col, t in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'), ('scaled_left_right', 'left_right')]:
            bt = best_per_target[t]
            blended[col] = (1-pct) * sub_1350[col] + pct * all_results[t][bt]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(pct*100)}% best-per-target + {int((1-pct)*100)}% Sub 1350")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
