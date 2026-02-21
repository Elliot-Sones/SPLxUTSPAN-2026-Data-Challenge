"""
Proper Functional PCA (FPCA) Features

Treat each keypoint trajectory as a continuous function using B-spline smoothing.
Extract Functional Principal Component scores as features for the Ridge pipeline.

Previous attempt (scripts/functional_data.py) used Fourier basis + standard PCA
which is NOT proper FPCA and performed 48% WORSE. This uses scikit-fda with
proper B-spline smoothing and roughness penalties.

Output: FPCA scores as new features, blended into Ridge predictions.
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

import skfda
from skfda.representation.basis import BSpline
from skfda.preprocessing.dim_reduction import FPCA as FunctionalPCA
from skfda.preprocessing.smoothing import BasisSmoother

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Key joints for FPCA
FPCA_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_shoulder',
    'mid_hip', 'neck', 'right_knee', 'left_knee',
]

# Frame range for functional analysis
FPCA_START = 100
FPCA_END = 190
FPCA_N_COMPONENTS = 5  # Number of FPC scores per joint-coord


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


def extract_fpca_features(data, kp_index, n_components=5):
    """Extract FPCA scores for each joint-coordinate trajectory.

    For each joint and coordinate (x, y, z):
    1. Extract the trajectory over frames FPCA_START:FPCA_END
    2. Smooth using B-spline basis representation
    3. Compute FPCA on the functional data
    4. Return the FPC scores as features

    Per-player FPCA to respect player-specific motion patterns.
    """
    n = len(data['pids'])
    unique_pids = sorted(np.unique(data['pids']))
    available_joints = [j for j in FPCA_JOINTS if j in kp_index]

    frames = np.arange(FPCA_START, FPCA_END)
    n_frames = len(frames)
    grid_points = frames.astype(float)

    # Total features: n_joints * 3 coords * n_components
    n_channels = len(available_joints) * 3
    total_features = n_channels * n_components

    print(f"  FPCA: {len(available_joints)} joints x 3 coords x {n_components} components = {total_features} features", flush=True)

    all_features = np.zeros((n, total_features), dtype=np.float32)

    for pid in unique_pids:
        mask = data['pids'] == pid
        indices = np.where(mask)[0]
        n_p = len(indices)
        print(f"  Player {pid}: {n_p} shots", flush=True)

        feat_col = 0
        for jname in available_joints:
            jidx = kp_index[jname]
            for coord in range(3):
                # Extract trajectories for this joint-coord for all shots of this player
                trajectories = np.zeros((n_p, n_frames), dtype=np.float64)

                for i, idx in enumerate(indices):
                    ts_hr = compute_hoop_transform(data['X_3d'][idx], kp_index)
                    traj = ts_hr[frames, jidx, coord].astype(np.float64)

                    # Clean NaNs
                    bad = np.isnan(traj) | np.isinf(traj)
                    if np.any(bad) and not np.all(bad):
                        good = ~bad
                        traj[bad] = np.interp(np.where(bad)[0], np.where(good)[0], traj[good])
                    elif np.all(bad):
                        traj[:] = 0.0

                    trajectories[i] = traj

                # Create functional data object
                try:
                    fd = skfda.FDataGrid(
                        data_matrix=trajectories,
                        grid_points=grid_points,
                    )

                    # Smooth with B-spline basis
                    n_basis = min(15, n_frames // 3)
                    basis = BSpline(domain_range=(FPCA_START, FPCA_END - 1), n_basis=n_basis)
                    smoother = BasisSmoother(basis=basis)
                    fd_smooth = smoother.fit_transform(fd)

                    # FPCA
                    nc = min(n_components, n_p - 1)
                    fpca = FunctionalPCA(n_components=nc)
                    scores = fpca.fit_transform(fd_smooth)

                    # Store scores
                    for c in range(nc):
                        all_features[indices, feat_col + c] = scores[:, c].astype(np.float32)

                except Exception as e:
                    print(f"    Warning: FPCA failed for {jname}_{coord} player {pid}: {e}", flush=True)

                feat_col += n_components

    # Clean NaNs
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    return all_features


def extract_base_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Same compact feature set as per_example_pipeline."""
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


def extract_all_base_features(data, target):
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = extract_base_features(ts_3d, ts_hr, kp_index, rf, frame)
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


def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.3):
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
    print("FPCA FEATURES PIPELINE", flush=True)
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

    # Extract FPCA features (shared across all targets)
    print("\nExtracting FPCA features...", flush=True)
    fpca_train = extract_fpca_features(train_data, kp_index, n_components=FPCA_N_COMPONENTS)
    fpca_test = extract_fpca_features(test_data, kp_index, n_components=FPCA_N_COMPONENTS)
    print(f"  FPCA features shape: train={fpca_train.shape}, test={fpca_test.shape}", flush=True)

    sub_3411 = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")
    results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}", flush=True)
        print(f"TARGET: {target.upper()}", flush=True)
        print(f"{'=' * 70}", flush=True)

        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]

        # Extract base features
        X_train_base = extract_all_base_features(train_data, target)
        X_test_base = extract_all_base_features(test_data, target)

        # --- Config A: Base + PLS (baseline) ---
        X_train_A, X_test_A = augment_with_pls(
            X_train_base, y_raw, pids_train, X_test_base, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        oof_A, test_A = locally_weighted_prediction(
            X_train_A, y_target, X_test_A, pids_train, pids_test, bandwidth_quantile=0.3)
        mse_A = np.mean((oof_A - y_target) ** 2)
        print(f"  [A] Base+PLS: CV MSE={mse_A:.6f}", flush=True)

        # --- Config B: Base + PLS + FPCA ---
        X_train_B = np.hstack([X_train_base, fpca_train])
        X_test_B = np.hstack([X_test_base, fpca_test])
        X_train_B, X_test_B = augment_with_pls(
            X_train_B, y_raw, pids_train, X_test_B, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        oof_B, test_B = locally_weighted_prediction(
            X_train_B, y_target, X_test_B, pids_train, pids_test, bandwidth_quantile=0.3)
        mse_B = np.mean((oof_B - y_target) ** 2)
        delta_B = (mse_B - mse_A) / mse_A * 100
        print(f"  [B] Base+FPCA+PLS: CV MSE={mse_B:.6f} (delta: {delta_B:+.2f}%)", flush=True)

        # --- Config C: FPCA only (standalone) ---
        X_train_C = fpca_train.copy()
        X_test_C = fpca_test.copy()
        oof_C, test_C = locally_weighted_prediction(
            X_train_C, y_target, X_test_C, pids_train, pids_test, bandwidth_quantile=0.3)
        mse_C = np.mean((oof_C - y_target) ** 2)
        print(f"  [C] FPCA only: CV MSE={mse_C:.6f}", flush=True)

        # Try different bandwidths for B
        best_mse = mse_B
        best_oof = oof_B
        best_test = test_B
        best_config = "B_bw0.3"

        for bw in [0.2, 0.4, 0.5]:
            oof_tmp, test_tmp = locally_weighted_prediction(
                X_train_B, y_target, X_test_B, pids_train, pids_test, bandwidth_quantile=bw)
            mse_tmp = np.mean((oof_tmp - y_target) ** 2)
            delta_tmp = (mse_tmp - mse_A) / mse_A * 100
            print(f"  [B] bw={bw}: CV MSE={mse_tmp:.6f} (delta: {delta_tmp:+.2f}%)", flush=True)
            if mse_tmp < best_mse:
                best_mse = mse_tmp
                best_oof = oof_tmp
                best_test = test_tmp
                best_config = f"B_bw{bw}"

        # If A (baseline) is still best, use it
        if mse_A < best_mse:
            best_mse = mse_A
            best_oof = oof_A
            best_test = test_A
            best_config = "A_baseline"

        # Diversity vs Sub 3411
        col = f'scaled_{target}'
        r = np.corrcoef(sub_3411[col].values, best_test)[0, 1]
        print(f"\n  BEST {target}: {best_config} (MSE={best_mse:.6f})", flush=True)
        print(f"  Diversity vs Sub3411: r={r:.4f}", flush=True)

        results[target] = {
            'best_test': best_test,
            'best_oof': best_oof,
            'best_mse': float(best_mse),
            'baseline_mse': float(mse_A),
            'best_config': best_config,
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
        print(f"  {target}: FPCA best={results[target]['best_mse']:.6f}, baseline={results[target]['baseline_mse']:.6f} ({delta:+.2f}%) [{results[target]['best_config']}]", flush=True)
    print(f"  MEAN FPCA: {total/3:.6f}", flush=True)
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
    print(f"  Sub {sub_num}: FPCA STANDALONE", flush=True)

    for w in [0.03, 0.05, 0.10, 0.15]:
        sub_num = get_next_submission_number()
        blended = sub_3411.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1 - w) * sub_3411[col] + w * results[target]['best_test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {int(w*100)}% FPCA + {int((1-w)*100)}% Sub3411", flush=True)

    # Save OOF
    oof_df = pd.DataFrame({
        'oof_angle': results['angle']['best_oof'],
        'oof_depth': results['depth']['best_oof'],
        'oof_left_right': results['left_right']['best_oof'],
    })
    oof_df.to_csv(OUTPUT_DIR / "fpca_oof_predictions.csv", index=False)

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {t: {'best_mse': results[t]['best_mse'],
                        'baseline_mse': results[t]['baseline_mse'],
                        'diversity_r': results[t]['diversity_r'],
                        'best_config': results[t]['best_config']}
                    for t in TARGETS},
        'mean_fpca_mse': float(total / 3),
        'mean_base_mse': float(total_base / 3),
    }
    with open(OUTPUT_DIR / "fpca_features_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
