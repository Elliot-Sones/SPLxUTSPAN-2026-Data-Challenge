"""
Multi-Frame Ensemble for Variance Reduction

Instead of extracting features at ONE optimal frame per target,
train separate per-example locally weighted Ridge models at MULTIPLE
frames and average predictions. This reduces variance without
adding features (which causes overfitting on 345 samples).

Methodology:
1. Map LOO MSE vs frame curve (frames 130-180) for each target
2. Select diverse frames based on LOO MSE and prediction correlation
3. Build ensembles incrementally (2, 3, 5, 7 frames)
4. Test with and without joint angle features
5. Generate submissions blended with Sub 2063 and Sub 784
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


def _angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


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


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame,
                     include_joint_angles=False):
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

    # Joint angles (10 features)
    if include_joint_angles:
        rw_i = kp_index.get('right_wrist')
        re_i = kp_index.get('right_elbow')
        rs_i = kp_index.get('right_shoulder')
        rh_i = kp_index.get('right_hip')
        lh_i = kp_index.get('left_hip')
        rk_i = kp_index.get('right_knee')
        lk_i = kp_index.get('left_knee')
        ra_i = kp_index.get('right_ankle')
        la_i = kp_index.get('left_ankle')
        neck_i = kp_index.get('neck')
        mh_i = kp_index.get('mid_hip')
        ls_i = kp_index.get('left_shoulder')

        # 1. Shoulder elevation angle
        if all(x is not None for x in [rs_i, rh_i, re_i]):
            v1 = ts_3d[f, rs_i] - ts_3d[f, rh_i]
            v2 = ts_3d[f, re_i] - ts_3d[f, rs_i]
            feats.append(_angle_between(v1, v2))
        else:
            feats.append(90.0)

        # 2. Trunk forward lean
        if neck_i is not None and mh_i is not None:
            trunk = ts_hr[f, neck_i] - ts_hr[f, mh_i]
            vertical = np.array([0, 0, 1], dtype=np.float32)
            feats.append(_angle_between(trunk, vertical))
            # 3. Trunk lateral lean
            feats.append(np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8)))
        else:
            feats.append(0.0)
            feats.append(0.0)

        # 4. Right knee flexion
        if all(x is not None for x in [rk_i, rh_i, ra_i]):
            v1 = ts_3d[f, rh_i] - ts_3d[f, rk_i]
            v2 = ts_3d[f, ra_i] - ts_3d[f, rk_i]
            feats.append(_angle_between(v1, v2))
        else:
            feats.append(90.0)

        # 5. Left knee flexion
        if all(x is not None for x in [lk_i, lh_i, la_i]):
            v1 = ts_3d[f, lh_i] - ts_3d[f, lk_i]
            v2 = ts_3d[f, la_i] - ts_3d[f, lk_i]
            feats.append(_angle_between(v1, v2))
        else:
            feats.append(90.0)

        # 6. Wrist deviation
        if re_i is not None and rw_i is not None:
            forearm = ts_3d[f, rw_i] - ts_3d[f, re_i]
            vertical = np.array([0, 0, 1], dtype=np.float32)
            feats.append(_angle_between(forearm, vertical))
        else:
            feats.append(90.0)

        # 7. Arm line angle vs hoop direction
        if rs_i is not None and rw_i is not None:
            arm_line = ts_hr[f, rw_i] - ts_hr[f, rs_i]
            hoop_dir = np.array([1, 0, 0.5], dtype=np.float32)
            feats.append(_angle_between(arm_line, hoop_dir))
        else:
            feats.append(90.0)

        # 8. Shoulder rotation
        if rs_i is not None and ls_i is not None:
            shoulder_line = ts_hr[f, rs_i] - ts_hr[f, ls_i]
            lateral = np.array([0, 1, 0], dtype=np.float32)
            feats.append(_angle_between(shoulder_line, lateral))
        else:
            feats.append(90.0)

        # 9. Hip-shoulder twist
        if all(x is not None for x in [rh_i, lh_i, rs_i, ls_i]):
            hip_line = ts_hr[f, rh_i, :2] - ts_hr[f, lh_i, :2]
            shoulder_line_xy = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
            hn = np.linalg.norm(hip_line)
            sn = np.linalg.norm(shoulder_line_xy)
            if hn > 1e-6 and sn > 1e-6:
                cos_a = np.clip(np.dot(hip_line, shoulder_line_xy) / (hn * sn), -1, 1)
                feats.append(np.degrees(np.arccos(cos_a)))
            else:
                feats.append(0.0)
        else:
            feats.append(0.0)

        # 10. Elbow height relative to shoulder
        if re_i is not None and rs_i is not None:
            feats.append(ts_hr[f, re_i, 2] - ts_hr[f, rs_i, 2])
        else:
            feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def extract_all_features_at_frame(data, frame, include_joint_angles=False):
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
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame,
                                 include_joint_angles=include_joint_angles)
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
# LOCALLY WEIGHTED PREDICTION (from per_example_pipeline.py)
# ==============================================================

def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.3, alpha=10.0):
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

        # LOO predictions
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

        # Test predictions
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
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("MULTI-FRAME ENSEMBLE FOR VARIANCE REDUCTION")
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

    # ============================================================
    # PHASE 1: Map LOO MSE vs Frame for each target
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: LOO MSE vs Frame Curve")
    print(f"{'=' * 70}")

    scan_frames = list(range(130, 181, 5))  # 130, 135, 140, ..., 180
    print(f"Scanning {len(scan_frames)} frames: {scan_frames}")

    # Store results per target: {frame: (loo_mse, oof_preds, test_preds)}
    frame_results = {t: {} for t in TARGETS}

    for target in TARGETS:
        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]
        print(f"\n  TARGET: {target.upper()}")

        for frame in scan_frames:
            # Extract features at this frame
            X_train_hc, _ = extract_all_features_at_frame(train_data, frame)
            X_test_hc, _ = extract_all_features_at_frame(test_data, frame)

            # PLS augmentation
            X_train_aug, X_test_aug = augment_with_pls(
                X_train_hc, y_raw, pids_train,
                X_test_hc, pids_test,
                train_data['X_raw'], test_data['X_raw'])

            # Locally weighted prediction
            oof, test_pred = locally_weighted_prediction(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                bandwidth_quantile=0.3, alpha=10.0)

            mse = np.mean((oof - y_target) ** 2)
            frame_results[target][frame] = {
                'mse': mse,
                'oof': oof.copy(),
                'test': test_pred.copy(),
            }
            print(f"    frame={frame}: LOO MSE = {mse:.6f}")

        # Print sorted ranking
        sorted_frames = sorted(frame_results[target].items(), key=lambda x: x[1]['mse'])
        print(f"\n  {target} RANKING (best to worst):")
        for frame, res in sorted_frames:
            marker = " <-- CURRENT BEST" if frame == TARGET_FRAMES[target] else ""
            print(f"    frame={frame}: MSE={res['mse']:.6f}{marker}")

    # ============================================================
    # PHASE 2: Pairwise Prediction Correlations
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: Pairwise Prediction Correlations")
    print(f"{'=' * 70}")

    # For each target, compute pairwise correlations of OOF predictions
    for target in TARGETS:
        frames_sorted = sorted(frame_results[target].keys(),
                               key=lambda f: frame_results[target][f]['mse'])
        top_frames = frames_sorted[:7]  # Top 7 by LOO MSE

        print(f"\n  {target.upper()} - Top {len(top_frames)} frames: {top_frames}")

        # Correlation matrix
        n_top = len(top_frames)
        corr_matrix = np.zeros((n_top, n_top))
        for i in range(n_top):
            for j in range(n_top):
                p_i = frame_results[target][top_frames[i]]['oof']
                p_j = frame_results[target][top_frames[j]]['oof']
                corr_matrix[i, j] = np.corrcoef(p_i, p_j)[0, 1]

        print(f"    Pairwise correlations:")
        header = "         " + "  ".join([f"f{f:3d}" for f in top_frames])
        print(header)
        for i in range(n_top):
            row_str = f"    f{top_frames[i]:3d}: " + "  ".join(
                [f"{corr_matrix[i,j]:.3f}" for j in range(n_top)])
            print(row_str)

    # ============================================================
    # PHASE 3: Build Multi-Frame Ensembles
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 3: Multi-Frame Ensembles")
    print(f"{'=' * 70}")

    # Strategy: For each target, pick frames greedily based on
    # lowest MSE and lowest correlation with already-selected frames
    ensemble_configs = {}

    for target in TARGETS:
        y_target = y_scaled[target]
        frames_by_mse = sorted(frame_results[target].keys(),
                               key=lambda f: frame_results[target][f]['mse'])

        # Greedy selection: start with best frame, add frame that
        # minimizes ensemble LOO MSE
        best_frame = frames_by_mse[0]
        selected = [best_frame]
        remaining = [f for f in frames_by_mse if f != best_frame]

        print(f"\n  {target.upper()} - Greedy frame selection:")
        print(f"    Start: frame={best_frame}, MSE={frame_results[target][best_frame]['mse']:.6f}")

        # Track ensemble MSE as we add frames
        ensemble_mses = [frame_results[target][best_frame]['mse']]

        for step in range(min(6, len(remaining))):
            # For each candidate, compute ensemble MSE
            best_candidate = None
            best_ens_mse = float('inf')

            for cand in remaining:
                # Average OOF predictions of selected + candidate
                oof_list = [frame_results[target][f]['oof'] for f in selected + [cand]]
                ens_oof = np.mean(oof_list, axis=0)
                ens_mse = np.mean((ens_oof - y_target) ** 2)

                if ens_mse < best_ens_mse:
                    best_ens_mse = ens_mse
                    best_candidate = cand

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            ensemble_mses.append(best_ens_mse)

            # Correlation of new frame with existing ensemble
            ens_oof_prev = np.mean([frame_results[target][f]['oof']
                                     for f in selected[:-1]], axis=0)
            new_oof = frame_results[target][best_candidate]['oof']
            r = np.corrcoef(ens_oof_prev, new_oof)[0, 1]
            delta = (best_ens_mse - ensemble_mses[-2]) / ensemble_mses[-2] * 100

            print(f"    +frame={best_candidate}: ensemble MSE={best_ens_mse:.6f} "
                  f"({delta:+.2f}%), r_with_prev={r:.4f}")

        ensemble_configs[target] = {
            'frames': selected,
            'mses': ensemble_mses,
        }

    # ============================================================
    # PHASE 4: Generate ensemble predictions at key sizes
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 4: Ensemble Predictions at Key Sizes")
    print(f"{'=' * 70}")

    # Test both equal-weight and inverse-MSE weighting
    ensemble_sizes = [2, 3, 5]  # Number of frames
    ensemble_predictions = {}  # {target: {size: {method: (oof, test, mse)}}}

    for target in TARGETS:
        y_target = y_scaled[target]
        ensemble_predictions[target] = {}
        frames = ensemble_configs[target]['frames']

        print(f"\n  {target.upper()}:")
        for size in ensemble_sizes:
            if size > len(frames):
                continue
            selected_frames = frames[:size]
            ensemble_predictions[target][size] = {}

            # Equal weight
            oof_list = [frame_results[target][f]['oof'] for f in selected_frames]
            test_list = [frame_results[target][f]['test'] for f in selected_frames]
            ens_oof = np.mean(oof_list, axis=0)
            ens_test = np.mean(test_list, axis=0)
            ens_mse = np.mean((ens_oof - y_target) ** 2)
            ensemble_predictions[target][size]['equal'] = (ens_oof, ens_test, ens_mse)

            # Inverse-MSE weighting
            mses = np.array([frame_results[target][f]['mse'] for f in selected_frames])
            inv_mse_weights = (1.0 / mses)
            inv_mse_weights /= inv_mse_weights.sum()
            ens_oof_w = np.average(oof_list, axis=0, weights=inv_mse_weights)
            ens_test_w = np.average(test_list, axis=0, weights=inv_mse_weights)
            ens_mse_w = np.mean((ens_oof_w - y_target) ** 2)
            ensemble_predictions[target][size]['inv_mse'] = (ens_oof_w, ens_test_w, ens_mse_w)

            print(f"    {size}-frame ({selected_frames})")
            print(f"      equal: MSE={ens_mse:.6f}  inv_mse: MSE={ens_mse_w:.6f}")
            print(f"      weights: {[f'{w:.3f}' for w in inv_mse_weights]}")

        # Also single-best for comparison
        best_f = frames[0]
        single_mse = frame_results[target][best_f]['mse']
        print(f"    single (frame {best_f}): MSE={single_mse:.6f}")

    # ============================================================
    # PHASE 5: Test with joint angles
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 5: Joint Angles Interaction")
    print(f"{'=' * 70}")

    # For the best ensemble size per target, re-run with joint angles
    ja_results = {}
    for target in TARGETS:
        y_target = y_scaled[target]
        y_raw = y_train[:, target_idx[target]]
        # Use 3-frame ensemble
        frames = ensemble_configs[target]['frames'][:3]

        print(f"\n  {target.upper()} - 3-frame with joint angles (frames: {frames})")

        oof_list_ja = []
        test_list_ja = []
        for frame in frames:
            X_train_hc, _ = extract_all_features_at_frame(train_data, frame,
                                                           include_joint_angles=True)
            X_test_hc, _ = extract_all_features_at_frame(test_data, frame,
                                                          include_joint_angles=True)

            X_train_aug, X_test_aug = augment_with_pls(
                X_train_hc, y_raw, pids_train,
                X_test_hc, pids_test,
                train_data['X_raw'], test_data['X_raw'])

            oof, test_pred = locally_weighted_prediction(
                X_train_aug, y_target, X_test_aug, pids_train, pids_test,
                bandwidth_quantile=0.3, alpha=10.0)

            oof_list_ja.append(oof)
            test_list_ja.append(test_pred)
            mse = np.mean((oof - y_target) ** 2)
            print(f"    frame={frame}: LOO MSE = {mse:.6f} (with JA)")

        ens_oof_ja = np.mean(oof_list_ja, axis=0)
        ens_test_ja = np.mean(test_list_ja, axis=0)
        ens_mse_ja = np.mean((ens_oof_ja - y_target) ** 2)

        # Compare with no-JA 3-frame
        ens_mse_no_ja = ensemble_predictions[target][3]['equal'][2]
        delta_pct = (ens_mse_ja - ens_mse_no_ja) / ens_mse_no_ja * 100

        print(f"    3-frame+JA ensemble: MSE={ens_mse_ja:.6f} vs no-JA: {ens_mse_no_ja:.6f} ({delta_pct:+.2f}%)")
        ja_results[target] = {
            'oof': ens_oof_ja,
            'test': ens_test_ja,
            'mse': ens_mse_ja,
        }

    # ============================================================
    # PHASE 6: Per-player breakdown
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 6: Per-Player Breakdown (3-frame equal)")
    print(f"{'=' * 70}")

    for target in TARGETS:
        y_target = y_scaled[target]
        ens_oof = ensemble_predictions[target][3]['equal'][0]
        single_oof = frame_results[target][ensemble_configs[target]['frames'][0]]['oof']

        print(f"\n  {target.upper()}:")
        for pid in sorted(np.unique(pids_train)):
            mask = pids_train == pid
            ens_mse_p = np.mean((ens_oof[mask] - y_target[mask]) ** 2)
            single_mse_p = np.mean((single_oof[mask] - y_target[mask]) ** 2)
            delta = (ens_mse_p - single_mse_p) / single_mse_p * 100
            print(f"    Player {pid}: single={single_mse_p:.6f} -> ensemble={ens_mse_p:.6f} ({delta:+.2f}%)")

    # ============================================================
    # PHASE 7: Generate Submissions
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PHASE 7: Generate Submissions")
    print(f"{'=' * 70}")

    # Load reference submissions
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_2063 = pd.read_csv(SUBMISSION_DIR / "submission_2063.csv")

    # Best ensemble config per target: pick size with lowest LOO MSE
    best_configs = {}
    for target in TARGETS:
        best_size = None
        best_method = None
        best_mse = float('inf')
        for size in ensemble_predictions[target]:
            for method in ensemble_predictions[target][size]:
                mse = ensemble_predictions[target][size][method][2]
                if mse < best_mse:
                    best_mse = mse
                    best_size = size
                    best_method = method
        # Also check JA
        ja_mse = ja_results[target]['mse']
        if ja_mse < best_mse:
            best_configs[target] = {
                'source': 'ja',
                'oof': ja_results[target]['oof'],
                'test': ja_results[target]['test'],
                'mse': ja_mse,
            }
        else:
            best_configs[target] = {
                'source': f'{best_size}f_{best_method}',
                'oof': ensemble_predictions[target][best_size][best_method][0],
                'test': ensemble_predictions[target][best_size][best_method][1],
                'mse': best_mse,
            }
        print(f"  {target}: best = {best_configs[target]['source']}, MSE = {best_configs[target]['mse']:.6f}")

    # Diversity with Sub 2063 and Sub 784
    print(f"\n  Diversity:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r_784 = np.corrcoef(sub_784[col].values, best_configs[target]['test'])[0, 1]
        r_2063 = np.corrcoef(sub_2063[col].values, best_configs[target]['test'])[0, 1]
        print(f"    {target}: r_784={r_784:.4f}, r_2063={r_2063:.4f}")

    # Also compute diversity for the 3-frame equal ensemble specifically
    print(f"\n  3-frame equal ensemble diversity:")
    for target in TARGETS:
        col = f'scaled_{target}'
        test_3f = ensemble_predictions[target][3]['equal'][1]
        r_784 = np.corrcoef(sub_784[col].values, test_3f)[0, 1]
        r_2063 = np.corrcoef(sub_2063[col].values, test_3f)[0, 1]
        print(f"    {target}: r_784={r_784:.4f}, r_2063={r_2063:.4f}")

    # --- Generate submissions ---
    submissions_info = []

    # Sub A: Standalone best per-target multiframe ensemble
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': best_configs['angle']['test'],
        'scaled_depth': best_configs['depth']['test'],
        'scaled_left_right': best_configs['left_right']['test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    info = (f"Sub {sub_num}: STANDALONE multiframe best per target "
            f"(a:{best_configs['angle']['source']}, d:{best_configs['depth']['source']}, "
            f"lr:{best_configs['left_right']['source']})")
    print(f"\n  {info}")
    submissions_info.append(info)

    # Sub B: Blend with Sub 784 at dw=0.30, lw=0.50 (proven weights)
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "proven weights no angle"),
        (0.50, 0.30, 0.50, "with angle fix weight"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*best_configs['angle']['test']
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*best_configs['depth']['test']
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*best_configs['left_right']['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        info = f"Sub {sub_num}: multiframe + Sub 784 (aw={aw}, dw={dw}, lw={lw}) [{desc}]"
        print(f"  {info}")
        submissions_info.append(info)

    # Sub C: Blend with Sub 2063 (current best on LB)
    for w in [0.30, 0.50]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({'id': test_data['ids']})
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_2063[col].values + w*best_configs[target]['test']
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        info = f"Sub {sub_num}: {w:.0%} multiframe + {1-w:.0%} Sub 2063"
        print(f"  {info}")
        submissions_info.append(info)

    # Sub D: 3-frame equal ensemble blended with Sub 784
    sub_num = get_next_submission_number()
    blended = sub_784.copy()
    blended['scaled_angle'] = sub_784['scaled_angle']  # don't touch angle (aw=0)
    blended['scaled_depth'] = 0.70*sub_784['scaled_depth'] + 0.30*ensemble_predictions['depth'][3]['equal'][1]
    blended['scaled_left_right'] = 0.50*sub_784['scaled_left_right'] + 0.50*ensemble_predictions['left_right'][3]['equal'][1]
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    info = f"Sub {sub_num}: 3-frame equal + Sub 784 (aw=0, dw=0.30, lw=0.50)"
    print(f"  {info}")
    submissions_info.append(info)

    # Sub E: 3-frame equal ensemble blended with Sub 2063
    for w in [0.30, 0.50]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({'id': test_data['ids']})
        for target in TARGETS:
            col = f'scaled_{target}'
            test_3f = ensemble_predictions[target][3]['equal'][1]
            blended[col] = (1-w)*sub_2063[col].values + w*test_3f
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        info = f"Sub {sub_num}: {w:.0%} 3-frame equal + {1-w:.0%} Sub 2063"
        print(f"  {info}")
        submissions_info.append(info)

    # Sub F: Joint angles ensemble blended with Sub 2063
    sub_num = get_next_submission_number()
    blended = pd.DataFrame({'id': test_data['ids']})
    for target in TARGETS:
        col = f'scaled_{target}'
        blended[col] = 0.70*sub_2063[col].values + 0.30*ja_results[target]['test']
    blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    info = f"Sub {sub_num}: 30% 3-frame+JA + 70% Sub 2063"
    print(f"  {info}")
    submissions_info.append(info)

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print("\nBest single-frame LOO MSE per target:")
    for target in TARGETS:
        best_f = ensemble_configs[target]['frames'][0]
        mse = frame_results[target][best_f]['mse']
        print(f"  {target}: frame={best_f}, MSE={mse:.6f}")

    print("\nBest multi-frame ensemble LOO MSE per target:")
    for target in TARGETS:
        print(f"  {target}: {best_configs[target]['source']}, MSE={best_configs[target]['mse']:.6f}")

    print(f"\nSingle-frame mean LOO MSE: {np.mean([frame_results[t][ensemble_configs[t]['frames'][0]]['mse'] for t in TARGETS]):.6f}")
    print(f"Multi-frame mean LOO MSE:  {np.mean([best_configs[t]['mse'] for t in TARGETS]):.6f}")

    print(f"\nSubmissions generated:")
    for info in submissions_info:
        print(f"  {info}")

    print(f"\nTotal time: {elapsed:.1f}s")

    # Return results for research doc
    return {
        'frame_results': frame_results,
        'ensemble_configs': ensemble_configs,
        'ensemble_predictions': ensemble_predictions,
        'ja_results': ja_results,
        'best_configs': best_configs,
        'submissions_info': submissions_info,
    }


if __name__ == "__main__":
    main()
