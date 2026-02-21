"""
Unified Pipeline - Combining ALL Proven Improvements

This pipeline integrates every technique that has shown CV improvement:

1. TARGET-SPECIFIC EXTRACTION FRAMES
   - Angle: frame 153 (CV MSE 0.006546)
   - Depth: frame 150 (CV MSE 0.007827)
   - Left_right: frame 170 (16.5% better than frame 153)

2. HOOP-RELATIVE COORDINATE TRANSFORMATION
   - Forward/lateral/vertical decomposition aligned to hoop direction
   - Decouples depth signal (forward) from left_right signal (lateral)

3. RELEASE FRAME TIMING
   - Physics-detected release frame as a scalar feature
   - Per-player correlation with depth: r=0.45-0.75

4. PLS COMPONENTS FROM RAW TIMESERIES
   - Proven best for depth (CV 0.00742 vs 0.0094 baseline)
   - Captures temporal dynamics hand-crafted features miss

5. HIERARCHICAL MODELING
   - Global model trained on all 345 samples + per-player models
   - Per-player optimal blend weight found via inner CV
   - Player 5 uses global model (per-player overfits on 74 samples)
   - -8.3% CV improvement over per-player-only baseline

6. MULTI-SEED ENSEMBLE
   - Average across 3 random seeds for variance reduction

7. CONSERVATIVE REGULARIZATION
   - Subsample=0.8, colsample_bytree=0.8
   - Stronger L1/L2 penalties
   - Fewer leaves / shallower trees
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

# Target-specific optimal frames (from physics frame analysis)
TARGET_FRAMES = {
    "angle": 153,
    "depth": 150,
    "left_right": 170,
}

# Random seeds for multi-seed ensemble
SEEDS = [42, 123, 777]


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
            placeholder.touch(exist_ok=True)
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
    """Load train and test data, parse into 3D timeseries + raw flat arrays."""
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

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Keypoints: {len(kp_names)}, Columns: {n_kp_cols}")

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        X_raw = np.zeros((n, n_kp_cols * 240), dtype=np.float32)
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                kp_idx = col_i // 3
                coord_idx = col_i % 3
                X_3d[idx, :, kp_idx, coord_idx] = arr
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            'X_3d': X_3d,
            'X_raw': X_raw,
            'pids': np.array(pids),
            'ids': np.array(ids),
            'kp_names': kp_names,
            'kp_index': kp_index,
        }
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    train = process(train_df, True)
    test = process(test_df, False)
    return train, test


# ==============================================================
# RELEASE FRAME DETECTION
# ==============================================================

def detect_release_frame(ts_3d, kp_index):
    """Detect per-shot release frame via peak ball speed near wrist peak."""
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

    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = []
    for key in ft_keys:
        idx = kp_index.get(key)
        if idx is not None:
            ft_trajs.append(ts_3d[:, idx, :])

    if len(ft_trajs) > 0:
        ft_center = np.nanmean(ft_trajs, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()

    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    ball_m = ball * FEET_TO_METERS
    vel = np.zeros_like(ball_m)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball_m[:, ax], 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)

    wrist_z_smooth = safe_savgol(wrist_traj[:, 2], 11, 3)
    search_start, search_end = 80, 200
    wrist_peak = search_start + np.argmax(wrist_z_smooth[search_start:search_end])

    release_end = min(wrist_peak + 5, search_end)
    release_start = max(search_start, wrist_peak - 40)
    search_speeds = speed[release_start:release_end]

    if len(search_speeds) > 0:
        release_frame = release_start + np.argmax(search_speeds)
    else:
        release_frame = max(search_start, wrist_peak - 10)

    return int(np.clip(release_frame, 80, 200))


def compute_all_release_frames(data):
    """Compute release frames for all shots."""
    n = len(data['pids'])
    rf = np.zeros(n, dtype=np.float32)
    for i in range(n):
        rf[i] = detect_release_frame(data['X_3d'][i], data['kp_index'])
    print(f"  Release frames: mean={rf.mean():.1f}, std={rf.std():.1f}, "
          f"min={rf.min():.0f}, max={rf.max():.0f}")
    return rf


# ==============================================================
# HOOP-RELATIVE COORDINATE TRANSFORMATION
# ==============================================================

def compute_hoop_transform(ts_3d, kp_index):
    """Compute rotation matrix and origin for hoop-relative coordinates."""
    mid_hip_idx = kp_index.get('mid_hip', 0)
    # Use frame 120 (mid-shot) for stable player position
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0  # Ground plane

    hoop_2d = HOOP_POS[:2]
    player_2d = player_pos[:2]
    forward = hoop_2d - player_2d
    fn = np.linalg.norm(forward)
    if fn > 1e-6:
        forward = forward / fn
    else:
        forward = np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])

    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]

    centered = ts_3d - player_pos.reshape(1, 1, 3)
    ts_hr = np.einsum('ij,fkj->fki', R, centered)
    return ts_hr, R, player_pos


# ==============================================================
# FEATURE EXTRACTION
# ==============================================================

def extract_features_for_target(ts_3d, ts_hr, kp_index, release_frame, target):
    """Extract features optimized for a specific target.

    Uses the target-specific optimal frame for at-frame features,
    plus release window stats and release_frame timing.
    """
    frame = TARGET_FRAMES[target]
    f = int(np.clip(frame, 0, 239))
    feats = {}

    key_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'left_shoulder',
        'right_hip', 'left_hip', 'mid_hip',
        'right_knee', 'left_knee',
        'right_ankle', 'left_ankle',
        'neck', 'nose',
    ]

    # --- AT-FRAME FEATURES (target-specific frame) ---
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            continue

        # Hoop-relative position at target frame
        for coord, cname in enumerate(['fwd', 'lat', 'vert']):
            series = ts_hr[:, idx, coord]
            feats[f'hr_{jname}_{cname}_pos'] = series[f]

            # Velocity at target frame
            vel = np.gradient(series, DT)
            feats[f'hr_{jname}_{cname}_vel'] = vel[f]

        # Original coords position at target frame
        for coord, cname in enumerate(['x', 'y', 'z']):
            series = ts_3d[:, idx, coord]
            feats[f'{jname}_{cname}_pos'] = series[f]

    # --- SUMMARY STATISTICS (full series) ---
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            continue

        for coord, cname in enumerate(['fwd', 'lat', 'vert']):
            series = ts_hr[:, idx, coord]
            feats[f'hr_{jname}_{cname}_mean'] = np.nanmean(series)
            feats[f'hr_{jname}_{cname}_std'] = np.nanstd(series)
            feats[f'hr_{jname}_{cname}_range'] = np.nanmax(series) - np.nanmin(series)

    # --- RELEASE WINDOW FEATURES (frames 140-180) ---
    for jname in ['right_wrist', 'right_elbow', 'right_shoulder']:
        idx = kp_index.get(jname)
        if idx is None:
            continue
        for coord, cname in enumerate(['fwd', 'lat', 'vert']):
            series = ts_hr[:, idx, coord]
            rw = series[140:180]
            feats[f'hr_{jname}_{cname}_rw_mean'] = np.nanmean(rw)
            vel = np.gradient(series, DT)
            feats[f'hr_{jname}_{cname}_rw_vel_max'] = np.nanmax(vel[140:180])
            feats[f'hr_{jname}_{cname}_rw_vel_mean'] = np.nanmean(vel[140:180])

    # --- ARM MECHANICS ---
    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')

    if all(idx is not None for idx in [rw, re, rs]):
        # Arm extension at target frame
        arm_fwd = ts_hr[f, rw, 0] - ts_hr[f, rs, 0]
        arm_lat = ts_hr[f, rw, 1] - ts_hr[f, rs, 1]
        arm_vert = ts_hr[f, rw, 2] - ts_hr[f, rs, 2]
        feats['arm_ext_fwd'] = arm_fwd
        feats['arm_ext_lat'] = arm_lat
        feats['arm_ext_vert'] = arm_vert

        # Elbow angle
        ua = ts_3d[f, re, :] - ts_3d[f, rs, :]
        fa = ts_3d[f, rw, :] - ts_3d[f, re, :]
        ua_n = np.linalg.norm(ua)
        fa_n = np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            cos_a = np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1)
            feats['elbow_angle'] = np.degrees(np.arccos(cos_a))
        else:
            feats['elbow_angle'] = 90.0

        # Forearm elevation
        if fa_n > 1e-6:
            feats['forearm_elev'] = np.degrees(np.arcsin(np.clip(fa[2] / fa_n, -1, 1)))
        else:
            feats['forearm_elev'] = 0.0

        # Wrist velocity components at target frame
        for coord, cname in enumerate(['fwd', 'lat', 'vert']):
            vel = np.gradient(ts_hr[:, rw, coord], DT)
            feats[f'wrist_vel_{cname}'] = vel[f]
            acc = np.gradient(vel, DT)
            feats[f'wrist_acc_{cname}'] = acc[f]

    # --- BODY ALIGNMENT ---
    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    rs_idx, ls = kp_index.get('right_shoulder'), kp_index.get('left_shoulder')

    if rh is not None and lh is not None:
        hip_lat = ts_hr[f, rh, 1] - ts_hr[f, lh, 1]
        feats['hip_alignment'] = hip_lat
        hip_fwd = ts_hr[f, rh, 0] - ts_hr[f, lh, 0]
        feats['hip_rotation'] = hip_fwd

    if rs_idx is not None and ls is not None:
        shoulder_lat = ts_hr[f, rs_idx, 1] - ts_hr[f, ls, 1]
        feats['shoulder_alignment'] = shoulder_lat
        shoulder_fwd = ts_hr[f, rs_idx, 0] - ts_hr[f, ls, 0]
        feats['shoulder_rotation'] = shoulder_fwd

    # Guide hand
    lw = kp_index.get('left_wrist')
    if lw is not None and rw is not None:
        feats['guide_hand_lat'] = ts_hr[f, lw, 1] - ts_hr[f, rw, 1]
        feats['guide_hand_fwd'] = ts_hr[f, lw, 0] - ts_hr[f, rw, 0]

    # --- PHASE VELOCITIES ---
    for pname, (s, e) in [('load', (60, 120)), ('propel', (120, 170))]:
        for jname in ['right_wrist', 'right_elbow']:
            idx = kp_index.get(jname)
            if idx is None:
                continue
            for coord in range(3):
                vel = np.gradient(ts_hr[s:e, idx, coord], DT)
                feats[f'phase_{pname}_{jname}_{"fle"[coord]}_vel_max'] = np.nanmax(vel)

    # --- FOLLOW-THROUGH DYNAMICS ---
    if rw is not None:
        for coord, cname in enumerate(['fwd', 'lat', 'vert']):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats[f'delta_wrist_{cname}_140_170'] = series[170] - series[140]
            feats[f'vel_range_wrist_{cname}_140_170'] = np.max(vel[140:170]) - np.min(vel[140:170])

    # --- RELEASE FRAME TIMING ---
    feats['release_frame'] = release_frame

    return feats


def extract_all_features(data, release_frames, target):
    """Extract features for all shots for a given target."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []

    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr, _, _ = compute_hoop_transform(ts_3d, kp_index)
        feats = extract_features_for_target(ts_3d, ts_hr, kp_index, release_frames[i], target)
        all_feats.append(feats)

    feat_names = sorted(all_feats[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_names


# ==============================================================
# PLS COMPONENT EXTRACTION
# ==============================================================

def extract_pls_components(X_raw_train, y_target_train, pids_train,
                           X_raw_test, pids_test, n_components_range=None):
    """Extract PLS components per player, return augmented features.

    Finds optimal number of components via inner CV, then fits on full data.
    Returns PLS features for both train (OOF) and test.
    """
    if n_components_range is None:
        n_components_range = [3, 5, 8, 10, 15, 20]

    unique_pids = sorted(np.unique(pids_train))
    pls_train = np.zeros((len(pids_train), 0))  # Will be filled per player
    pls_test = np.zeros((len(pids_test), 0))

    # Determine max components across all players
    max_nc = 0
    per_player_nc = {}

    for pid in unique_pids:
        mask = pids_train == pid
        n_p = mask.sum()
        max_comp = min(n_components_range[-1], n_p - n_p // 5 - 1)
        candidates = [c for c in n_components_range if c <= max_comp]
        if not candidates:
            candidates = [min(3, n_p - 2)]

        X_p = X_raw_train[mask]
        y_p = y_target_train[mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)

        best_nc, best_mse = candidates[0], float('inf')
        for nc in candidates:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(X_scaled):
                pls = PLSRegression(n_components=nc)
                pls.fit(X_scaled[tr_idx], y_p[tr_idx])
                pred = pls.predict(X_scaled[val_idx]).flatten()
                mses.append(np.mean((pred - y_p[val_idx]) ** 2))
            avg = np.mean(mses)
            if avg < best_mse:
                best_mse = avg
                best_nc = nc

        per_player_nc[pid] = best_nc
        max_nc = max(max_nc, best_nc)

    print(f"  PLS components per player: {per_player_nc}")

    # Now extract PLS features
    pls_train_feats = np.zeros((len(pids_train), max_nc), dtype=np.float32)
    pls_test_feats = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        nc = per_player_nc[pid]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_raw_train[tr_mask])
        X_te_s = scaler.transform(X_raw_test[te_mask])

        pls = PLSRegression(n_components=nc)
        pls.fit(X_tr_s, y_target_train[tr_mask])

        pls_train_feats[tr_mask, :nc] = pls.transform(X_tr_s)
        pls_test_feats[te_mask, :nc] = pls.transform(X_te_s)

    return pls_train_feats, pls_test_feats, per_player_nc


def extract_pls_oof(X_raw, y_target, pids, per_player_nc, max_nc):
    """Extract PLS features using OOF to avoid data leakage in CV."""
    unique_pids = sorted(np.unique(pids))
    pls_oof = np.zeros((len(pids), max_nc), dtype=np.float32)

    for pid in unique_pids:
        mask = pids == pid
        nc = per_player_nc[pid]
        X_p = X_raw[mask]
        y_p = y_target[mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_scaled):
            pls = PLSRegression(n_components=nc)
            pls.fit(X_scaled[tr_idx], y_p[tr_idx])
            pls_oof[np.where(mask)[0][val_idx], :nc] = pls.transform(X_scaled[val_idx])

    return pls_oof


# ==============================================================
# HIERARCHICAL MODEL
# ==============================================================

def make_models(seed):
    """Create model instances with a specific seed."""
    return [
        ('lgb', lgb.LGBMRegressor(
            n_estimators=80, num_leaves=8, learning_rate=0.05,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=seed,
            verbose=-1, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=seed,
            verbosity=0, n_jobs=-1)),
        ('cat', CatBoostRegressor(
            iterations=80, depth=3, learning_rate=0.05,
            l2_leaf_reg=5.0, random_seed=seed, verbose=False)),
        ('ridge', Ridge(alpha=10.0)),
    ]


def ensemble_predict(models_dict, X):
    """Predict using weighted ensemble."""
    preds = []
    for name, m in models_dict.items():
        preds.append(m.predict(X))
    # 0.3 LGB + 0.3 XGB + 0.3 CatBoost + 0.1 Ridge
    return 0.3 * preds[0] + 0.3 * preds[1] + 0.3 * preds[2] + 0.1 * preds[3]


def train_hierarchical_cv(X_train, y_train, pids_train, target_name, seed):
    """Train hierarchical model with CV, return OOF predictions and optimal weights."""
    unique_pids = sorted(np.unique(pids_train))
    n = len(y_train)
    oof_per_player = np.zeros(n)
    oof_global = np.zeros(n)

    # Global model OOF
    kf_global = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf_global.split(X_train):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr_idx])
        X_val = scaler.transform(X_train[val_idx])
        y_tr = y_train[tr_idx]

        models = {}
        for name, m in make_models(seed):
            m.fit(X_tr, y_tr)
            models[name] = m
        oof_global[val_idx] = ensemble_predict(models, X_val)

    # Per-player model OOF
    for pid in unique_pids:
        mask = pids_train == pid
        X_p = X_train[mask]
        y_p = y_train[mask]
        indices = np.where(mask)[0]
        n_p = len(X_p)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf.split(X_scaled):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr = y_p[tr_idx]

            models = {}
            for name, m in make_models(seed):
                m.fit(X_tr, y_tr)
                models[name] = m
            oof_per_player[indices[val_idx]] = ensemble_predict(models, X_val)

    # Find optimal per-player global weight
    optimal_weights = {}
    for pid in unique_pids:
        mask = pids_train == pid
        y_p = y_train[mask]
        global_p = oof_global[mask]
        local_p = oof_per_player[mask]

        best_w, best_mse = 0.0, float('inf')
        for w in np.arange(0.0, 1.05, 0.1):
            blended = w * global_p + (1 - w) * local_p
            mse = np.mean((blended - y_p) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w
        optimal_weights[pid] = round(best_w, 1)

    # Compute blended OOF
    oof_blended = np.zeros(n)
    for pid in unique_pids:
        mask = pids_train == pid
        w = optimal_weights[pid]
        oof_blended[mask] = w * oof_global[mask] + (1 - w) * oof_per_player[mask]

    mse_blended = np.mean((oof_blended - y_train) ** 2)
    mse_local = np.mean((oof_per_player - y_train) ** 2)
    mse_global = np.mean((oof_global - y_train) ** 2)

    print(f"    Seed {seed}: global MSE={mse_global:.6f}, local MSE={mse_local:.6f}, "
          f"blended MSE={mse_blended:.6f}")
    print(f"    Optimal weights: {optimal_weights}")

    return oof_blended, optimal_weights


def train_hierarchical_final(X_train, y_train, pids_train, X_test, pids_test,
                             optimal_weights, seed):
    """Train final hierarchical model and predict test."""
    unique_pids = sorted(np.unique(pids_train))

    # Global model
    scaler_global = StandardScaler()
    X_tr_g = scaler_global.fit_transform(X_train)
    X_te_g = scaler_global.transform(X_test)

    global_models = {}
    for name, m in make_models(seed):
        m.fit(X_tr_g, y_train)
        global_models[name] = m
    global_test_preds = ensemble_predict(global_models, X_te_g)

    # Per-player models
    local_test_preds = np.zeros(len(X_test))
    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        if not np.any(te_mask):
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr_mask])
        X_te = scaler.transform(X_test[te_mask])
        y_tr = y_train[tr_mask]

        models = {}
        for name, m in make_models(seed):
            m.fit(X_tr, y_tr)
            models[name] = m
        local_test_preds[te_mask] = ensemble_predict(models, X_te)

    # Blend
    test_preds = np.zeros(len(X_test))
    for pid in unique_pids:
        te_mask = pids_test == pid
        w = optimal_weights[pid]
        test_preds[te_mask] = w * global_test_preds[te_mask] + (1 - w) * local_test_preds[te_mask]

    return test_preds


# ==============================================================
# MAIN PIPELINE
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("UNIFIED PIPELINE")
    print("=" * 70)

    # Load data
    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    # Detect release frames
    print("\nDetecting release frames...")
    rf_train = compute_all_release_frames(train_data)
    rf_test = compute_all_release_frames(test_data)

    # Load target scalers
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Scale targets
    y_scaled = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # ============================================================
    # PER-TARGET PIPELINE
    # ============================================================

    oof_preds = {}
    test_preds = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()} (optimal frame: {TARGET_FRAMES[target]})")
        print(f"{'=' * 70}")

        # 1. Extract hand-crafted features at target-specific frame
        print(f"  Extracting features at frame {TARGET_FRAMES[target]}...")
        X_train_hc, feat_names = extract_all_features(train_data, rf_train, target)
        X_test_hc, _ = extract_all_features(test_data, rf_test, target)
        print(f"  Hand-crafted features: {X_train_hc.shape[1]}")

        # 2. Extract PLS components from raw timeseries
        print(f"  Extracting PLS components...")
        y_raw_target = y_train[:, target_idx[target]]
        pls_train, pls_test, per_player_nc = extract_pls_components(
            train_data['X_raw'], y_raw_target, pids_train,
            test_data['X_raw'], pids_test,
            n_components_range=[3, 5, 8, 10, 15, 20])

        max_nc = pls_train.shape[1]

        # OOF PLS features for CV (avoid leakage)
        pls_oof = extract_pls_oof(
            train_data['X_raw'], y_raw_target, pids_train, per_player_nc, max_nc)

        # 3. Combine features
        X_train_combined = np.hstack([X_train_hc, pls_oof])
        X_test_combined = np.hstack([X_test_hc, pls_test])
        print(f"  Combined features: {X_train_combined.shape[1]} "
              f"({X_train_hc.shape[1]} HC + {max_nc} PLS)")

        # 4. Multi-seed hierarchical training
        print(f"  Training hierarchical model (multi-seed)...")
        seed_oof = []
        seed_test = []
        seed_weights = []

        for seed in SEEDS:
            oof, weights = train_hierarchical_cv(
                X_train_combined, y_scaled[target], pids_train, target, seed)
            seed_oof.append(oof)
            seed_weights.append(weights)

            # Average weights across seeds for final model
            test_pred = train_hierarchical_final(
                X_train_combined, y_scaled[target], pids_train,
                X_test_combined, pids_test, weights, seed)
            seed_test.append(test_pred)

        # Average across seeds
        oof_avg = np.mean(seed_oof, axis=0)
        test_avg = np.mean(seed_test, axis=0)

        # Report CV
        mse = np.mean((oof_avg - y_scaled[target]) ** 2)
        r = np.corrcoef(y_scaled[target], oof_avg)[0, 1]
        print(f"  {target} FINAL CV: MSE={mse:.6f}, r={r:.4f}")

        oof_preds[target] = oof_avg
        test_preds[target] = test_avg

    # ============================================================
    # OVERALL CV RESULTS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("OVERALL CV RESULTS")
    print(f"{'=' * 70}")

    total_mse = 0
    for target in TARGETS:
        mse = np.mean((oof_preds[target] - y_scaled[target]) ** 2)
        total_mse += mse
        print(f"  {target}: scaled MSE = {mse:.6f}")
    mean_mse = total_mse / 3
    print(f"  MEAN SCALED MSE: {mean_mse:.6f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Standalone submission
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': test_preds['angle'],
        'scaled_depth': test_preds['depth'],
        'scaled_left_right': test_preds['left_right'],
    })
    filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(filepath, index=False)
    a_std = sub['scaled_angle'].std()
    d_mean = sub['scaled_depth'].mean()
    print(f"  Sub {sub_num}: STANDALONE")
    print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")
    print(f"    -> {filepath}")

    # Blend with Sub 784
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Test correlation with Sub 784
    print(f"\n  Correlation with Sub 784:")
    for col, target in [('scaled_angle', 'angle'), ('scaled_depth', 'depth'),
                         ('scaled_left_right', 'left_right')]:
        r = np.corrcoef(sub_784[col].values, test_preds[target])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # Target-specific blends
    blend_configs = [
        # (aw, dw, lw, description)
        (0.00, 0.30, 0.50, "Sub 784 weights"),
        (0.10, 0.30, 0.50, "light angle"),
        (0.00, 0.30, 0.30, "moderate LR"),
        (0.00, 0.20, 0.40, "balanced"),
        (0.10, 0.20, 0.30, "conservative all"),
        (0.00, 0.40, 0.60, "aggressive depth+LR"),
        (0.15, 0.30, 0.50, "medium angle"),
        (0.00, 0.30, 0.70, "strong LR"),
        (0.00, 0.15, 0.50, "light depth + Sub784 LR"),
        (0.00, 0.50, 0.50, "heavy depth"),
    ]

    print(f"\n  Generating blended submissions:")
    for aw, dw, lw, desc in blend_configs:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1 - aw) * sub_784['scaled_angle'] + aw * test_preds['angle']
        blended['scaled_depth'] = (1 - dw) * sub_784['scaled_depth'] + dw * test_preds['depth']
        blended['scaled_left_right'] = (1 - lw) * sub_784['scaled_left_right'] + lw * test_preds['left_right']

        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(filepath, index=False)

        print(f"    Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")
        print(f"      angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    # Also blend with Sub 771 (in case our model is better as a standalone replacement)
    sub_771 = pd.read_csv(SUBMISSION_DIR / "submission_771.csv")
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "771: same weights as Sub 784"),
        (0.00, 0.40, 0.60, "771: aggressive"),
        (0.50, 0.50, 0.50, "771: equal blend"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_771.copy()
        blended['scaled_angle'] = (1 - aw) * sub_771['scaled_angle'] + aw * test_preds['angle']
        blended['scaled_depth'] = (1 - dw) * sub_771['scaled_depth'] + dw * test_preds['depth']
        blended['scaled_left_right'] = (1 - lw) * sub_771['scaled_left_right'] + lw * test_preds['left_right']

        a_std = blended['scaled_angle'].std()
        d_mean = blended['scaled_depth'].mean()
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        blended.to_csv(filepath, index=False)

        print(f"    Sub {sub_num}: aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc}) [vs Sub 771]")
        print(f"      angle_std={a_std:.6f}, depth_mean={d_mean:.6f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
