"""
Physics-Constrained Per-Example Regression

Key idea: Instead of predicting targets directly, predict the ball's release
velocity (vx, vy, vz) using per-example locally weighted Ridge, then forward
simulate through ballistic equations to get targets.

Why this might work:
- Same 213 features and per-example method as Sub 1350 (our best)
- But physics equations constrain the output space (3 velocity -> 3 targets)
- This acts as a strong inductive bias / regularizer with only 345 samples
- Velocity is a more natural intermediate than direct target prediction

Previous physics attempt (inverse_projectile_pipeline.py) failed because:
- Only 33 hand geometry features (vs 213 here)
- Per-player Ridge+LGB+PLS (vs per-example weighted Ridge here)
- vy prediction was weak (R=0.56) - should improve with more features

Pipeline:
1. Compute exact release velocity for all 345 training shots (inverse projectile)
2. Extract 213 features (HC at frame 150 + PLS) - same as Sub 1350 approach
3. Per-example locally weighted Ridge to predict (vx, vy, vz) for test shots
4. Forward simulate velocity -> targets via ballistic equations
5. Ensemble physics predictions with Sub 1350 direct predictions
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
from scipy.stats import pearsonr
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
G = 9.81  # m/s^2

# Use frame 150 for velocity features (closest to release, between target frames)
VELOCITY_FRAME = 150
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
# DATA LOADING (same as per_example_pipeline.py)
# ==============================================================

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


def extract_all_features(data, frame):
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


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test, X_raw_train, X_raw_test):
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
# PHYSICS: INVERSE PROJECTILE + FORWARD SIMULATION
# ==============================================================

def estimate_release_position(ts_3d, kp_index, release_frame):
    """Estimate ball position at release (in feet)."""
    rf = release_frame
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return None
    wrist = ts_3d[rf, rw_idx, :]
    if np.any(np.isnan(wrist)):
        return None

    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    fingertips = []
    for k in ft_keys:
        idx = kp_index.get(k)
        if idx is not None:
            pos = ts_3d[rf, idx, :]
            if not np.any(np.isnan(pos)):
                fingertips.append(pos)

    if len(fingertips) == 0:
        return wrist.copy()

    ft_center = np.mean(fingertips, axis=0)
    ball_pos = wrist + 0.6 * (ft_center - wrist)
    return ball_pos


def compute_true_velocity(release_pos_ft, angle_deg, depth_in, lr_in):
    """Compute exact release velocity from known targets and release position."""
    x_land_ft = HOOP_POS[0] + lr_in / 12.0
    y_land_ft = HOOP_POS[1] + depth_in / 12.0
    z_land_ft = HOOP_POS[2]

    dx = (x_land_ft - release_pos_ft[0]) * FEET_TO_METERS
    dy = (y_land_ft - release_pos_ft[1]) * FEET_TO_METERS
    dz = (z_land_ft - release_pos_ft[2]) * FEET_TO_METERS

    D_horiz = np.sqrt(dx**2 + dy**2)
    angle_rad = np.radians(angle_deg)

    t_squared = 2.0 * (dz + D_horiz * np.tan(angle_rad)) / G
    if t_squared <= 0:
        return None

    t = np.sqrt(t_squared)
    vx = dx / t
    vy = dy / t
    vz = (dz + 0.5 * G * t**2) / t

    return np.array([vx, vy, vz])


def forward_simulate(release_pos_ft, vel_ms):
    """Forward simulate projectile from release to hoop plane (z=10ft)."""
    pos_m = np.array(release_pos_ft) * FEET_TO_METERS
    target_z_m = HOOP_POS[2] * FEET_TO_METERS

    dz_m = target_z_m - pos_m[2]
    vz = vel_ms[2]

    a = 0.5 * G
    b = -vz
    c = dz_m

    disc = b**2 - 4*a*c
    if disc < 0:
        return None

    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)

    times = [t for t in [t1, t2] if t > 0]
    if not times:
        return None
    t = max(times)

    if t > 5.0:
        return None

    landing_x_m = pos_m[0] + vel_ms[0] * t
    landing_y_m = pos_m[1] + vel_ms[1] * t

    landing_vz = vz - G * t
    horiz_speed = np.sqrt(vel_ms[0]**2 + vel_ms[1]**2)

    landing_x_ft = landing_x_m / FEET_TO_METERS
    landing_y_ft = landing_y_m / FEET_TO_METERS

    entry_angle = np.degrees(np.arctan2(-landing_vz, horiz_speed))
    depth_in = (landing_y_ft - HOOP_POS[1]) * 12.0
    lr_in = (landing_x_ft - HOOP_POS[0]) * 12.0

    return {'angle': entry_angle, 'depth': depth_in, 'left_right': lr_in}


# ==============================================================
# PER-EXAMPLE LOCALLY WEIGHTED REGRESSION
# ==============================================================

def locally_weighted_regression(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.5, alpha=10.0):
    """Per-example locally weighted Ridge regression.
    Returns LOO OOF predictions and test predictions."""
    unique_pids = sorted(np.unique(pids_train))
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

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

        # LOO OOF
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
# MAIN PIPELINE
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("PHYSICS-CONSTRAINED PER-EXAMPLE REGRESSION")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']  # raw targets: angle (deg), depth (in), lr (in)
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    n_train = len(pids_train)
    n_test = len(pids_test)

    # Load scalers
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # ==============================================================
    # PHASE 1: Compute ground-truth velocities for training shots
    # ==============================================================
    print("\n--- PHASE 1: Inverse Projectile ---")
    kp_index = train_data['kp_index']

    true_vel = np.zeros((n_train, 3))
    release_positions_train = []
    release_frames_train = []
    valid_mask = np.ones(n_train, dtype=bool)

    for i in range(n_train):
        ts_3d = train_data['X_3d'][i]
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames_train.append(rf)
        pos = estimate_release_position(ts_3d, kp_index, rf)
        release_positions_train.append(pos)

        if pos is None:
            valid_mask[i] = False
            continue

        vel = compute_true_velocity(pos, y_train[i, 0], y_train[i, 1], y_train[i, 2])
        if vel is None:
            valid_mask[i] = False
            continue

        true_vel[i] = vel

    n_valid = valid_mask.sum()
    print(f"  Valid inverse solutions: {n_valid}/{n_train}")

    speeds = np.linalg.norm(true_vel[valid_mask], axis=1)
    print(f"  Speed: mean={speeds.mean():.2f}, std={speeds.std():.2f} m/s")

    # For invalid shots, impute player-mean velocity
    for pid in sorted(np.unique(pids_train)):
        pid_mask = (pids_train == pid) & valid_mask
        pid_invalid = (pids_train == pid) & ~valid_mask
        if pid_mask.sum() > 0 and pid_invalid.sum() > 0:
            mean_vel = true_vel[pid_mask].mean(axis=0)
            true_vel[pid_invalid] = mean_vel
            valid_mask[pid_invalid] = True
            print(f"  Player {pid}: imputed {pid_invalid.sum()} shots with mean velocity")

    # Test release positions
    release_positions_test = []
    release_frames_test = []
    for i in range(n_test):
        ts_3d = test_data['X_3d'][i]
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames_test.append(rf)
        pos = estimate_release_position(ts_3d, kp_index, rf)
        if pos is None:
            # Fallback: use player mean position
            pid = pids_test[i]
            pid_mask = pids_train == pid
            positions = [release_positions_train[j] for j in range(n_train)
                         if pid_mask[j] and release_positions_train[j] is not None]
            if positions:
                pos = np.mean(positions, axis=0)
            else:
                pos = np.array([5.25, -20.0, 8.0])  # reasonable default
        release_positions_test.append(pos)

    # ==============================================================
    # PHASE 2: Extract features
    # ==============================================================
    print("\n--- PHASE 2: Feature Extraction ---")

    # Extract HC features at velocity frame (150)
    X_train_hc, _ = extract_all_features(train_data, VELOCITY_FRAME)
    X_test_hc, _ = extract_all_features(test_data, VELOCITY_FRAME)
    print(f"  HC features: {X_train_hc.shape[1]}")

    # Augment with PLS (using depth as reference target for PLS - middle target)
    y_depth_raw = y_train[:, 1]
    X_train_aug, X_test_aug = augment_with_pls(
        X_train_hc, y_depth_raw, pids_train,
        X_test_hc, pids_test,
        train_data['X_raw'], test_data['X_raw'])
    print(f"  Augmented features: {X_train_aug.shape[1]}")

    # Also extract target-specific features for direct prediction comparison
    X_train_ts = {}
    X_test_ts = {}
    for target in TARGETS:
        X_tr_hc, _ = extract_all_features(train_data, TARGET_FRAMES[target])
        X_te_hc, _ = extract_all_features(test_data, TARGET_FRAMES[target])
        y_raw = y_train[:, TARGETS.index(target)]
        X_tr_aug, X_te_aug = augment_with_pls(
            X_tr_hc, y_raw, pids_train,
            X_te_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])
        X_train_ts[target] = X_tr_aug
        X_test_ts[target] = X_te_aug

    # ==============================================================
    # PHASE 3: Per-example velocity prediction
    # ==============================================================
    print("\n--- PHASE 3: Velocity Prediction (Per-Example Weighted Ridge) ---")

    vel_oof = np.zeros((n_train, 3))
    vel_test = np.zeros((n_test, 3))

    for vi, vname in enumerate(['vx', 'vy', 'vz']):
        print(f"\n  Predicting {vname}...")
        y_v = true_vel[:, vi]

        # Try different bandwidths and alphas
        best_config = None
        best_mse = float('inf')

        for bw in [0.3, 0.4, 0.5, 0.6]:
            for alpha in [5.0, 10.0, 20.0]:
                oof, _ = locally_weighted_regression(
                    X_train_aug, y_v, X_test_aug, pids_train, pids_test,
                    bandwidth_quantile=bw, alpha=alpha)
                mse = np.mean((oof - y_v) ** 2)
                r = pearsonr(oof, y_v)[0]
                if mse < best_mse:
                    best_mse = mse
                    best_config = (bw, alpha)
                print(f"    bw={bw:.1f}, alpha={alpha:.0f}: MSE={mse:.6f}, R={r:.4f}")

        bw_best, alpha_best = best_config
        oof, test = locally_weighted_regression(
            X_train_aug, y_v, X_test_aug, pids_train, pids_test,
            bandwidth_quantile=bw_best, alpha=alpha_best)
        vel_oof[:, vi] = oof
        vel_test[:, vi] = test

        r_val = pearsonr(oof, y_v)[0]
        rmse = np.sqrt(np.mean((oof - y_v) ** 2))
        print(f"    BEST {vname}: bw={bw_best}, alpha={alpha_best}, R={r_val:.4f}, RMSE={rmse:.4f} m/s")

    # Overall velocity prediction quality
    print("\n  Velocity prediction summary:")
    for vi, vname in enumerate(['vx', 'vy', 'vz']):
        r = pearsonr(vel_oof[:, vi], true_vel[:, vi])[0]
        rmse = np.sqrt(np.mean((vel_oof[:, vi] - true_vel[:, vi]) ** 2))
        print(f"    {vname}: R={r:.4f}, RMSE={rmse:.4f} m/s")

    speed_pred = np.linalg.norm(vel_oof, axis=1)
    speed_true = np.linalg.norm(true_vel, axis=1)
    print(f"    speed: R={pearsonr(speed_pred, speed_true)[0]:.4f}")

    # ==============================================================
    # PHASE 4: Forward simulate to get physics-predicted targets
    # ==============================================================
    print("\n--- PHASE 4: Forward Simulation ---")

    # OOF physics targets
    oof_physics_raw = np.zeros((n_train, 3))
    oof_sim_fails = 0
    for i in range(n_train):
        if release_positions_train[i] is None:
            oof_physics_raw[i] = y_train[i]  # fallback to true
            oof_sim_fails += 1
            continue
        sim = forward_simulate(release_positions_train[i], vel_oof[i])
        if sim is None:
            oof_physics_raw[i] = y_train[i]
            oof_sim_fails += 1
        else:
            oof_physics_raw[i, 0] = sim['angle']
            oof_physics_raw[i, 1] = sim['depth']
            oof_physics_raw[i, 2] = sim['left_right']

    print(f"  OOF simulation failures: {oof_sim_fails}/{n_train}")

    # Test physics targets
    test_physics_raw = np.zeros((n_test, 3))
    test_sim_fails = 0
    for i in range(n_test):
        sim = forward_simulate(release_positions_test[i], vel_test[i])
        if sim is None:
            test_sim_fails += 1
            # Fallback to mean
            test_physics_raw[i] = y_train.mean(axis=0)
        else:
            test_physics_raw[i, 0] = sim['angle']
            test_physics_raw[i, 1] = sim['depth']
            test_physics_raw[i, 2] = sim['left_right']

    print(f"  Test simulation failures: {test_sim_fails}/{n_test}")

    # Scale physics predictions to [0,1]
    oof_physics_scaled = np.zeros((n_train, 3))
    test_physics_scaled = np.zeros((n_test, 3))
    for ti, target in enumerate(TARGETS):
        oof_physics_scaled[:, ti] = scalers[target].transform(
            oof_physics_raw[:, ti].reshape(-1, 1)).ravel()
        test_physics_scaled[:, ti] = scalers[target].transform(
            test_physics_raw[:, ti].reshape(-1, 1)).ravel()

    # Evaluate physics predictions
    print("\n  Physics predictions (scaled MSE):")
    y_scaled = np.zeros((n_train, 3))
    for ti, target in enumerate(TARGETS):
        y_scaled[:, ti] = scalers[target].transform(
            y_train[:, ti].reshape(-1, 1)).ravel()

    total_phys_mse = 0
    for ti, target in enumerate(TARGETS):
        mse = np.mean((oof_physics_scaled[:, ti] - y_scaled[:, ti]) ** 2)
        total_phys_mse += mse
        print(f"    {target}: MSE={mse:.6f}")
    print(f"    MEAN: {total_phys_mse / 3:.6f}")

    # ==============================================================
    # PHASE 5: Direct prediction (same as Sub 1350) for comparison
    # ==============================================================
    print("\n--- PHASE 5: Direct Prediction (Sub 1350 style) ---")

    oof_direct_scaled = np.zeros((n_train, 3))
    test_direct_scaled = np.zeros((n_test, 3))

    for ti, target in enumerate(TARGETS):
        y_target_scaled = y_scaled[:, ti]
        oof, test = locally_weighted_regression(
            X_train_ts[target], y_target_scaled, X_test_ts[target],
            pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        oof_direct_scaled[:, ti] = oof
        test_direct_scaled[:, ti] = test
        mse = np.mean((oof - y_target_scaled) ** 2)
        print(f"    {target}: MSE={mse:.6f}")

    total_direct_mse = sum(
        np.mean((oof_direct_scaled[:, ti] - y_scaled[:, ti]) ** 2)
        for ti in range(3)) / 3
    print(f"    MEAN: {total_direct_mse:.6f}")

    # ==============================================================
    # PHASE 6: Ensemble physics + direct predictions
    # ==============================================================
    print("\n--- PHASE 6: Physics + Direct Ensemble ---")

    # Try different ensemble weights per target
    best_ensemble_weights = {}
    oof_ensemble_scaled = np.zeros((n_train, 3))
    test_ensemble_scaled = np.zeros((n_test, 3))

    for ti, target in enumerate(TARGETS):
        best_w = 0.0
        best_mse = float('inf')

        for w_phys in np.arange(0.0, 0.55, 0.05):
            oof_blend = w_phys * oof_physics_scaled[:, ti] + (1 - w_phys) * oof_direct_scaled[:, ti]
            mse = np.mean((oof_blend - y_scaled[:, ti]) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w_phys

        best_ensemble_weights[target] = best_w
        oof_ensemble_scaled[:, ti] = best_w * oof_physics_scaled[:, ti] + (1 - best_w) * oof_direct_scaled[:, ti]
        test_ensemble_scaled[:, ti] = best_w * test_physics_scaled[:, ti] + (1 - best_w) * test_direct_scaled[:, ti]

        r_phys_direct = pearsonr(oof_physics_scaled[:, ti], oof_direct_scaled[:, ti])[0]
        print(f"    {target}: best_w_phys={best_w:.2f}, MSE={best_mse:.6f}, "
              f"phys-direct r={r_phys_direct:.4f}")

    total_ensemble_mse = sum(
        np.mean((oof_ensemble_scaled[:, ti] - y_scaled[:, ti]) ** 2)
        for ti in range(3)) / 3
    print(f"    MEAN ensemble MSE: {total_ensemble_mse:.6f}")
    print(f"    vs direct: {total_direct_mse:.6f} ({(total_ensemble_mse/total_direct_mse - 1)*100:+.1f}%)")
    print(f"    vs physics: {total_phys_mse / 3:.6f} ({(total_ensemble_mse/(total_phys_mse/3) - 1)*100:+.1f}%)")

    # ==============================================================
    # PHASE 7: Blend with Sub 784 and generate submissions
    # ==============================================================
    print("\n--- PHASE 7: Submissions ---")

    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    # Check diversity
    print("  Diversity vs Sub 1350:")
    for ti, target in enumerate(TARGETS):
        col = f'scaled_{target}'
        r_phys = pearsonr(sub_1350[col].values, test_physics_scaled[:, ti])[0]
        r_dir = pearsonr(sub_1350[col].values, test_direct_scaled[:, ti])[0]
        r_ens = pearsonr(sub_1350[col].values, test_ensemble_scaled[:, ti])[0]
        print(f"    {target}: physics r={r_phys:.4f}, direct r={r_dir:.4f}, ensemble r={r_ens:.4f}")

    print("\n  Diversity vs Sub 784:")
    for ti, target in enumerate(TARGETS):
        col = f'scaled_{target}'
        r_phys = pearsonr(sub_784[col].values, test_physics_scaled[:, ti])[0]
        r_dir = pearsonr(sub_784[col].values, test_direct_scaled[:, ti])[0]
        print(f"    {target}: physics r={r_phys:.4f}, direct r={r_dir:.4f}")

    # Generate submissions
    submissions = []

    # Sub A: Physics standalone
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': test_physics_scaled[:, 0],
        'scaled_depth': test_physics_scaled[:, 1],
        'scaled_left_right': test_physics_scaled[:, 2],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    submissions.append(('physics_standalone', sub_num))
    print(f"\n  Sub {sub_num}: Physics standalone")

    # Sub B: Physics + Direct ensemble
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': test_ensemble_scaled[:, 0],
        'scaled_depth': test_ensemble_scaled[:, 1],
        'scaled_left_right': test_ensemble_scaled[:, 2],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    submissions.append(('physics_direct_ensemble', sub_num))
    print(f"  Sub {sub_num}: Physics+Direct ensemble (w_phys: "
          f"a={best_ensemble_weights['angle']:.2f}, "
          f"d={best_ensemble_weights['depth']:.2f}, "
          f"lr={best_ensemble_weights['left_right']:.2f})")

    # Sub C-F: Blend physics with Sub 784 at various weights
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "physics_sub784_standard"),
        (0.10, 0.30, 0.50, "physics_sub784_with_angle"),
        (0.00, 0.20, 0.30, "physics_sub784_conservative"),
        (0.00, 0.40, 0.60, "physics_sub784_aggressive"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*test_physics_scaled[:, 0]
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*test_physics_scaled[:, 1]
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*test_physics_scaled[:, 2]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        submissions.append((desc, sub_num))
        print(f"  Sub {sub_num}: {desc} (aw={aw:.2f} dw={dw:.2f} lw={lw:.2f})")

    # Sub G-I: Blend ensemble with Sub 784
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "ensemble_sub784_standard"),
        (0.00, 0.20, 0.30, "ensemble_sub784_conservative"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*test_ensemble_scaled[:, 0]
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*test_ensemble_scaled[:, 1]
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*test_ensemble_scaled[:, 2]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        submissions.append((desc, sub_num))
        print(f"  Sub {sub_num}: {desc} (aw={aw:.2f} dw={dw:.2f} lw={lw:.2f})")

    # Sub J-K: Blend physics with Sub 1350 (small physics correction)
    for w_phys, desc in [(0.10, "10pct_physics_90pct_sub1350"),
                         (0.20, "20pct_physics_80pct_sub1350"),
                         (0.30, "30pct_physics_70pct_sub1350")]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for ti, target in enumerate(TARGETS):
            col = f'scaled_{target}'
            blended[col] = (1 - w_phys) * sub_1350[col] + w_phys * test_physics_scaled[:, ti]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        submissions.append((desc, sub_num))
        print(f"  Sub {sub_num}: {desc}")

    # Sub L-M: Blend ensemble with Sub 1350
    for w_ens, desc in [(0.10, "10pct_ensemble_90pct_sub1350"),
                        (0.20, "20pct_ensemble_80pct_sub1350")]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for ti, target in enumerate(TARGETS):
            col = f'scaled_{target}'
            blended[col] = (1 - w_ens) * sub_1350[col] + w_ens * test_ensemble_scaled[:, ti]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        submissions.append((desc, sub_num))
        print(f"  Sub {sub_num}: {desc}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Physics velocity prediction:")
    for vi, vname in enumerate(['vx', 'vy', 'vz']):
        r = pearsonr(vel_oof[:, vi], true_vel[:, vi])[0]
        print(f"    {vname}: R={r:.4f}")
    print(f"\n  Target prediction (scaled LOO MSE):")
    print(f"    Physics only: {total_phys_mse / 3:.6f}")
    print(f"    Direct only:  {total_direct_mse:.6f}")
    print(f"    Ensemble:     {total_ensemble_mse:.6f}")
    print(f"    Sub 1350 LB:  0.006776")
    print(f"\n  Ensemble weights (physics fraction):")
    for target in TARGETS:
        print(f"    {target}: {best_ensemble_weights[target]:.2f}")
    print(f"\n  Submissions generated: {len(submissions)}")
    for desc, num in submissions:
        print(f"    Sub {num}: {desc}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
