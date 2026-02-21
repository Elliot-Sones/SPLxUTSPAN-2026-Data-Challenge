"""
OOF (Out-of-Fold) Stacking Pipeline for SPLxUTSPAN 2026 Data Challenge

Architecture:
  Layer 0 (feature extraction): 10 different feature views of the data
  Layer 1 (base models): Multiple models per feature space -> OOF predictions
  Layer 2 (meta-learner): Ridge on diverse subset of Layer 1 OOF predictions

CV: Leave-One-Player-Out (LOPO) for honest OOF, since players are the
    natural grouping variable in this basketball shooting dataset.

Diversity is the key objective - all previous approaches achieve r>0.85 with
Sub 1350. We need models with fundamentally different feature representations
to break through.
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
from sklearn.linear_model import Ridge, Lasso, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
TARGET_IDX = {"angle": 0, "depth": 1, "left_right": 2}
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


# ================================================================
# UTILITIES
# ================================================================

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


# ================================================================
# DATA LOADING
# ================================================================

def load_data():
    """Load and parse all data into 3D keypoint arrays."""
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


# ================================================================
# FEATURE EXTRACTION FUNCTIONS
# ================================================================

KEY_JOINTS = ['right_wrist', 'right_elbow', 'right_shoulder',
              'left_wrist', 'left_shoulder',
              'right_hip', 'left_hip', 'mid_hip',
              'right_knee', 'left_knee', 'neck', 'nose']


def compute_hoop_transform(ts_3d, kp_index):
    """Transform to hoop-relative coordinates."""
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
    """Detect ball release frame from wrist/finger trajectories."""
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


def _extract_hc_at_frame(ts_3d, ts_hr, kp_index, frame, release_frame):
    """Extract hoop-relative features at a specific frame (compact set)."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    # Position + velocity at frame for key joints
    for jname in KEY_JOINTS:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])

    # Summary stats (mean, std, range) for key joints
    for jname in KEY_JOINTS:
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


# ================================================================
# FEATURE SPACE DEFINITIONS
# ================================================================

def feat_hc_at_frame(data, frame):
    """HC features at a single frame."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats = _extract_hc_at_frame(ts_3d, ts_hr, kp_index, frame, rf)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feat_multi_frame_delta(data, frame_a, frame_b):
    """Delta features: frame_b minus frame_a (velocity-like)."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        feats_a = _extract_hc_at_frame(ts_3d, ts_hr, kp_index, frame_a, rf)
        feats_b = _extract_hc_at_frame(ts_3d, ts_hr, kp_index, frame_b, rf)
        # Delta + both frames concatenated
        delta = feats_b - feats_a
        all_feats.append(np.concatenate([feats_a, feats_b, delta]))
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feat_joint_angles(data, frames):
    """Joint angles (elbow, shoulder, knee) at multiple frames."""
    n = len(data['pids'])
    kp_index = data['kp_index']

    angle_joints = [
        ('right_shoulder', 'right_elbow', 'right_wrist', 'r_elbow'),
        ('right_hip', 'right_shoulder', 'right_elbow', 'r_shoulder_flex'),
        ('right_hip', 'right_knee', 'right_ankle', 'r_knee') if 'right_ankle' in kp_index else None,
        ('left_hip', 'left_knee', 'left_ankle', 'l_knee') if 'left_ankle' in kp_index else None,
        ('neck', 'right_shoulder', 'right_elbow', 'r_shoulder_abd'),
        ('left_shoulder', 'left_elbow', 'left_wrist', 'l_elbow'),
    ]
    angle_joints = [j for j in angle_joints if j is not None]

    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        feats = []
        for j1, j2, j3, name in angle_joints:
            if not all(j in kp_index for j in [j1, j2, j3]):
                feats.extend([0.0] * (len(frames) * 3 + 3))
                continue
            p1 = ts_3d[:, kp_index[j1]]
            p2 = ts_3d[:, kp_index[j2]]
            p3 = ts_3d[:, kp_index[j3]]
            v1 = p1 - p2
            v2 = p3 - p2
            dot = np.sum(v1 * v2, axis=1)
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            denom = n1 * n2
            denom[denom == 0] = 1e-10
            angles = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            angles = np.nan_to_num(angles, nan=90.0)

            for f in frames:
                f = int(np.clip(f, 0, 239))
                feats.append(angles[f])
                # Angular velocity
                ang_vel = np.gradient(angles, DT)
                feats.append(ang_vel[f])
                # Angular acceleration
                ang_acc = np.gradient(ang_vel, DT)
                feats.append(ang_acc[f])

            feats.append(np.nanmax(angles[100:200]) - np.nanmin(angles[100:200]))
            feats.append(np.nanmean(angles[140:170]))
            feats.append(np.nanstd(angles[100:200]))

        all_feats.append(np.array(feats, dtype=np.float32))
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feat_body_segments(data, frames):
    """Body segment lengths and ratios at multiple frames."""
    n = len(data['pids'])
    kp_index = data['kp_index']

    segments = [
        ('right_shoulder', 'right_elbow', 'r_upper_arm'),
        ('right_elbow', 'right_wrist', 'r_forearm'),
        ('right_shoulder', 'right_hip', 'r_torso'),
        ('right_hip', 'right_knee', 'r_thigh'),
        ('left_shoulder', 'left_elbow', 'l_upper_arm'),
        ('left_elbow', 'left_wrist', 'l_forearm'),
        ('left_shoulder', 'left_hip', 'l_torso'),
        ('neck', 'mid_hip', 'spine'),
        ('right_shoulder', 'left_shoulder', 'shoulder_width'),
        ('right_hip', 'left_hip', 'hip_width'),
    ]

    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        feats = []
        seg_lengths = {}
        for j1, j2, name in segments:
            if j1 not in kp_index or j2 not in kp_index:
                for f in frames:
                    feats.append(0.0)
                feats.extend([0.0, 0.0])
                seg_lengths[name] = np.zeros(240)
                continue
            lengths = np.linalg.norm(ts_3d[:, kp_index[j1]] - ts_3d[:, kp_index[j2]], axis=1)
            lengths = np.nan_to_num(lengths, nan=0.0)
            seg_lengths[name] = lengths
            for f in frames:
                f = int(np.clip(f, 0, 239))
                feats.append(lengths[f])
            feats.append(np.nanmean(lengths))
            feats.append(np.nanstd(lengths))

        # Ratios
        for (n1, n2) in [('r_forearm', 'r_upper_arm'), ('r_torso', 'r_thigh'),
                         ('shoulder_width', 'hip_width'), ('spine', 'r_thigh')]:
            if n1 in seg_lengths and n2 in seg_lengths:
                denom = seg_lengths[n2] + 1e-6
                ratio = seg_lengths[n1] / denom
                feats.append(np.nanmean(ratio))
                feats.append(ratio[150] if len(ratio) > 150 else 0.0)
            else:
                feats.extend([0.0, 0.0])

        all_feats.append(np.array(feats, dtype=np.float32))
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feat_velocity_profiles(data, frames):
    """Velocity and acceleration at key frames for key joints."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist', 'mid_hip', 'neck']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        feats = []
        for jname in joints:
            idx = kp_index.get(jname)
            if idx is None:
                feats.extend([0.0] * (len(frames) * 9 + 6))
                continue
            for coord in range(3):
                pos = ts_hr[:, idx, coord]
                pos = np.nan_to_num(pos, nan=0.0)
                vel = np.gradient(pos, DT)
                acc = np.gradient(vel, DT)
                for f in frames:
                    f = int(np.clip(f, 0, 239))
                    feats.append(vel[f])
                    feats.append(acc[f])
                    speed = np.abs(vel[f])
                    feats.append(speed)
            # Speed (3D magnitude) stats
            pos_3d = ts_hr[:, idx, :]
            vel_3d = np.gradient(pos_3d, DT, axis=0)
            speed_3d = np.linalg.norm(vel_3d, axis=1)
            feats.append(np.nanmax(speed_3d[100:200]))
            feats.append(np.nanmean(speed_3d[140:170]))
            frame_max_speed = 100 + np.argmax(speed_3d[100:200])
            feats.append(float(frame_max_speed))
            # Peak speed timing relative to release
            feats.extend([np.nanstd(speed_3d[100:200]),
                         np.nanmean(speed_3d[80:120]),
                         np.nanmean(speed_3d[160:200])])
        all_feats.append(np.array(feats, dtype=np.float32))
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feat_per_player_zscore(data, frame):
    """HC features z-scored within each player group."""
    X_hc = feat_hc_at_frame(data, frame)
    pids = data['pids']
    unique_pids = sorted(np.unique(pids))
    X_z = np.zeros_like(X_hc)
    for pid in unique_pids:
        mask = pids == pid
        scaler = StandardScaler()
        X_z[mask] = scaler.fit_transform(X_hc[mask])
    return np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)


def feat_pls_timeseries(data, y_raw_col, n_components=15):
    """PLS on raw 240-frame timeseries."""
    X_raw = data['X_raw']
    pids = data['pids']
    unique_pids = sorted(np.unique(pids))

    X_pls = np.zeros((len(pids), n_components), dtype=np.float32)
    pls_models = {}

    for pid in unique_pids:
        mask = pids == pid
        n_p = mask.sum()
        scaler = StandardScaler()
        raw_s = scaler.fit_transform(X_raw[mask])
        nc = min(n_components, n_p - n_p // 5 - 1)
        nc = max(3, nc)
        if y_raw_col is not None:
            pls = PLSRegression(n_components=nc)
            pls.fit(raw_s, y_raw_col[mask])
            X_pls[mask, :nc] = pls.transform(raw_s)
        else:
            # Unsupervised: use PCA-like via SVD
            U, S, Vt = np.linalg.svd(raw_s, full_matrices=False)
            X_pls[mask, :nc] = U[:, :nc] * S[:nc]
        pls_models[pid] = {'scaler': scaler, 'nc': nc}

    return X_pls, pls_models


def feat_lasso_selected(data, y_raw_col, frame, n_select=30):
    """LASSO stability selection: run LASSO at multiple alphas, keep features
    that are consistently selected."""
    X_hc = feat_hc_at_frame(data, frame)
    pids = data['pids']
    n_feat = X_hc.shape[1]

    # Selection on full data
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_hc)

    # Run LASSO at multiple alphas and count how often each feature is selected
    selection_counts = np.zeros(n_feat)
    alphas = np.logspace(-4, -1, 20)
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_s, y_raw_col)
        selected = np.abs(lasso.coef_) > 1e-8
        selection_counts += selected.astype(float)

    # Take top n_select features by selection frequency
    top_idx = np.argsort(selection_counts)[-n_select:]
    return X_hc[:, top_idx], top_idx


def feat_concat_multi_frame(data, frames):
    """Concatenated HC features from multiple frames (compact: positions only)."""
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        feats = []
        for f in frames:
            f = int(np.clip(f, 0, 239))
            for jname in KEY_JOINTS:
                idx = kp_index.get(jname)
                if idx is None:
                    feats.extend([0.0] * 3)
                    continue
                for coord in range(3):
                    feats.append(ts_hr[f, idx, coord])
        all_feats.append(np.array(feats, dtype=np.float32))
    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ================================================================
# MODEL DEFINITIONS
# ================================================================

def make_models():
    """Return dict of (name, model_factory) pairs."""
    return {
        'ridge_1': lambda: Ridge(alpha=1.0),
        'ridge_10': lambda: Ridge(alpha=10.0),
        'ridge_100': lambda: Ridge(alpha=100.0),
        'lasso_001': lambda: Lasso(alpha=0.001, max_iter=5000),
        'lasso_01': lambda: Lasso(alpha=0.01, max_iter=5000),
        'lgb_4_001': lambda: lgb.LGBMRegressor(
            n_estimators=100, num_leaves=4, learning_rate=0.01,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1),
        'lgb_8_005': lambda: lgb.LGBMRegressor(
            n_estimators=200, num_leaves=8, learning_rate=0.05,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1),
        'lgb_4_005': lambda: lgb.LGBMRegressor(
            n_estimators=100, num_leaves=4, learning_rate=0.05,
            min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.5, reg_lambda=1.5, random_state=42, verbose=-1, n_jobs=-1),
        'xgb_2_001': lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbosity=0, n_jobs=-1),
        'xgb_3_005': lambda: xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=-1),
        'xgb_2_005': lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.5, reg_lambda=1.5, random_state=42, verbosity=0, n_jobs=-1),
        'knn_5': lambda: KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'knn_10': lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'),
        'knn_20': lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'),
    }


# ================================================================
# OOF GENERATION (LOPO)
# ================================================================

def generate_oof_lopo(X_train, y_train, pids_train, X_test, model_factory):
    """Generate OOF predictions using Leave-One-Player-Out CV.

    Returns:
        oof_preds: (n_train,) array of OOF predictions
        test_preds: (n_test,) array of test predictions (average across folds)
    """
    unique_pids = sorted(np.unique(pids_train))
    oof_preds = np.zeros(len(y_train))
    test_preds_all = []

    for pid in unique_pids:
        tr_mask = pids_train != pid
        val_mask = pids_train == pid

        # Standardize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[tr_mask])
        X_val = scaler.transform(X_train[val_mask])
        X_te = scaler.transform(X_test)

        y_tr = y_train[tr_mask]

        model = model_factory()
        model.fit(X_tr, y_tr)

        oof_preds[val_mask] = model.predict(X_val)
        test_preds_all.append(model.predict(X_te))

    test_preds = np.mean(test_preds_all, axis=0)
    return oof_preds, test_preds


# ================================================================
# DIVERSITY ANALYSIS
# ================================================================

def compute_diversity_matrix(oof_dict):
    """Compute pairwise Pearson r between all OOF predictions."""
    names = list(oof_dict.keys())
    n = len(names)
    R = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.corrcoef(oof_dict[names[i]], oof_dict[names[j]])[0, 1]
            R[i, j] = r
            R[j, i] = r
    return R, names


def greedy_diverse_selection(oof_dict, y_true, max_models=15, min_r_threshold=0.50):
    """Greedily select diverse models: add model that minimizes max pairwise r
    with already-selected models, subject to a minimum quality threshold."""
    names = list(oof_dict.keys())
    # Compute MSE for each model
    mses = {name: np.mean((oof_dict[name] - y_true) ** 2) for name in names}
    # Filter out very poor models (worse than 3x median)
    median_mse = np.median(list(mses.values()))
    candidates = [n for n in names if mses[n] < 3.0 * median_mse]

    if len(candidates) == 0:
        candidates = names[:5]

    # Start with the best model
    selected = [min(candidates, key=lambda n: mses[n])]
    candidates = [n for n in candidates if n not in selected]

    while len(selected) < max_models and candidates:
        best_candidate = None
        best_score = -999

        for c in candidates:
            # Max pairwise r with selected set
            max_r = max(np.corrcoef(oof_dict[c], oof_dict[s])[0, 1] for s in selected)
            # Score: prefer low correlation (diversity) but also good performance
            # Normalize MSE: lower is better
            mse_rank = 1.0 - (mses[c] / (3.0 * median_mse))
            score = (1.0 - max_r) + 0.3 * mse_rank
            if score > best_score:
                best_score = score
                best_candidate = c

        if best_candidate is None:
            break

        # Check if it adds enough diversity
        max_r_with_selected = max(
            np.corrcoef(oof_dict[best_candidate], oof_dict[s])[0, 1]
            for s in selected
        )
        if max_r_with_selected > 0.999 and len(selected) >= 3:
            # Too correlated, skip
            candidates.remove(best_candidate)
            continue

        selected.append(best_candidate)
        candidates.remove(best_candidate)

    return selected, {n: mses[n] for n in selected}


# ================================================================
# META-LEARNER
# ================================================================

def train_meta_learner(oof_matrix, y_train, pids_train, test_matrix, alpha=10.0):
    """Train Ridge meta-learner with nested LOPO CV.

    Returns:
        meta_oof: (n_train,) honest meta-learner OOF predictions
        meta_test: (n_test,) meta-learner test predictions
    """
    unique_pids = sorted(np.unique(pids_train))
    meta_oof = np.zeros(len(y_train))
    meta_test_all = []

    for pid in unique_pids:
        tr_mask = pids_train != pid
        val_mask = pids_train == pid

        scaler = StandardScaler()
        M_tr = scaler.fit_transform(oof_matrix[tr_mask])
        M_val = scaler.transform(oof_matrix[val_mask])
        M_te = scaler.transform(test_matrix)

        meta = Ridge(alpha=alpha)
        meta.fit(M_tr, y_train[tr_mask])

        meta_oof[val_mask] = meta.predict(M_val)
        meta_test_all.append(meta.predict(M_te))

    meta_test = np.mean(meta_test_all, axis=0)
    return meta_oof, meta_test


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("OOF STACKING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    train_data, test_data = load_data()
    y_train_raw = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    # Load scalers
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train_raw[:, TARGET_IDX[target]].reshape(-1, 1)).ravel()

    # Load reference submissions for diversity comparison
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")
    sub_1640 = pd.read_csv(SUBMISSION_DIR / "submission_1640.csv")
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXTRACTING FEATURE SPACES")
    print("=" * 70)

    feature_spaces = {}
    t_feat = time.time()

    # 1. HC at target-specific optimal frames
    for frame in [150, 153, 170]:
        name = f"hc_f{frame}"
        print(f"  [{name}] HC features at frame {frame}...")
        X_tr = feat_hc_at_frame(train_data, frame)
        X_te = feat_hc_at_frame(test_data, frame)
        feature_spaces[name] = (X_tr, X_te)
        print(f"    Shape: {X_tr.shape}")

    # 2. HC at earlier/later frames for diversity
    for frame in [120, 140, 200]:
        name = f"hc_f{frame}"
        print(f"  [{name}] HC features at frame {frame}...")
        X_tr = feat_hc_at_frame(train_data, frame)
        X_te = feat_hc_at_frame(test_data, frame)
        feature_spaces[name] = (X_tr, X_te)
        print(f"    Shape: {X_tr.shape}")

    # 3. Multi-frame deltas
    print("  [delta_150_170] Multi-frame delta f170 - f150...")
    X_tr = feat_multi_frame_delta(train_data, 150, 170)
    X_te = feat_multi_frame_delta(test_data, 150, 170)
    feature_spaces['delta_150_170'] = (X_tr, X_te)
    print(f"    Shape: {X_tr.shape}")

    print("  [delta_120_153] Multi-frame delta f153 - f120...")
    X_tr = feat_multi_frame_delta(train_data, 120, 153)
    X_te = feat_multi_frame_delta(test_data, 120, 153)
    feature_spaces['delta_120_153'] = (X_tr, X_te)
    print(f"    Shape: {X_tr.shape}")

    # 4. Joint angles
    print("  [joint_angles] Joint angles at key frames...")
    X_tr = feat_joint_angles(train_data, [120, 140, 150, 153, 160, 170])
    X_te = feat_joint_angles(test_data, [120, 140, 150, 153, 160, 170])
    feature_spaces['joint_angles'] = (X_tr, X_te)
    print(f"    Shape: {X_tr.shape}")

    # 5. Body segments
    print("  [body_segments] Body segment lengths/ratios...")
    X_tr = feat_body_segments(train_data, [120, 140, 150, 153, 170])
    X_te = feat_body_segments(test_data, [120, 140, 150, 153, 170])
    feature_spaces['body_segments'] = (X_tr, X_te)
    print(f"    Shape: {X_tr.shape}")

    # 6. Velocity profiles
    print("  [velocity] Velocity/acceleration profiles...")
    X_tr = feat_velocity_profiles(train_data, [140, 150, 153, 160, 170])
    X_te = feat_velocity_profiles(test_data, [140, 150, 153, 160, 170])
    feature_spaces['velocity'] = (X_tr, X_te)
    print(f"    Shape: {X_tr.shape}")

    # 7. Per-player z-scored
    print("  [zscore_153] Per-player z-scored HC at f153...")
    X_tr = feat_per_player_zscore(train_data, 153)
    X_te_raw = feat_hc_at_frame(test_data, 153)
    # Z-score test data per player using train stats
    X_te = np.zeros_like(X_te_raw)
    X_tr_raw = feat_hc_at_frame(train_data, 153)
    for pid in sorted(np.unique(pids_train)):
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        scaler_z = StandardScaler()
        scaler_z.fit(X_tr_raw[tr_mask])
        if np.any(te_mask):
            X_te[te_mask] = scaler_z.transform(X_te_raw[te_mask])
    feature_spaces['zscore_153'] = (X_tr, X_te)
    print(f"    Shape: {X_tr.shape}")

    # 8. Concatenated multi-frame positions (compact)
    print("  [multi_pos] Concatenated positions at 5 frames...")
    X_tr = feat_concat_multi_frame(train_data, [120, 140, 153, 170, 200])
    X_te = feat_concat_multi_frame(test_data, [120, 140, 153, 170, 200])
    feature_spaces['multi_pos'] = (X_tr, X_te)
    print(f"    Shape: {X_tr.shape}")

    print(f"\nFeature extraction: {time.time() - t_feat:.1f}s")
    print(f"Total feature spaces: {len(feature_spaces)}")

    # ------------------------------------------------------------------
    # Model selection (subset for speed)
    # ------------------------------------------------------------------
    # Use a focused set of models per feature space
    model_configs = {
        'ridge_10': lambda: Ridge(alpha=10.0),
        'ridge_100': lambda: Ridge(alpha=100.0),
        'lasso_001': lambda: Lasso(alpha=0.001, max_iter=5000),
        'lgb_4_005': lambda: lgb.LGBMRegressor(
            n_estimators=100, num_leaves=4, learning_rate=0.05,
            min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.5, reg_lambda=1.5, random_state=42, verbose=-1, n_jobs=-1),
        'lgb_8_005': lambda: lgb.LGBMRegressor(
            n_estimators=200, num_leaves=8, learning_rate=0.05,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1),
        'xgb_2_005': lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.5, reg_lambda=1.5, random_state=42, verbosity=0, n_jobs=-1),
        'xgb_3_005': lambda: xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0, n_jobs=-1),
        'knn_10': lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'),
    }

    # ------------------------------------------------------------------
    # Per-target OOF generation + stacking
    # ------------------------------------------------------------------
    final_results = {}

    for target in TARGETS:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target.upper()}")
        print(f"{'=' * 70}")

        y_t = y_scaled[target]
        y_raw_t = y_train_raw[:, TARGET_IDX[target]]

        # PLS and LASSO need target-specific extraction
        # 9. PLS on raw timeseries
        print(f"  [pls_{target}] PLS on raw timeseries...")
        X_pls_tr, pls_models = feat_pls_timeseries(train_data, y_raw_t, n_components=15)
        # For test: need to transform using same PLS models
        X_pls_te = np.zeros((len(pids_test), 15), dtype=np.float32)
        for pid in sorted(np.unique(pids_train)):
            te_mask = pids_test == pid
            if not np.any(te_mask):
                continue
            tr_mask = pids_train == pid
            nc = pls_models[pid]['nc']
            raw_scaler = pls_models[pid]['scaler']
            raw_te = raw_scaler.transform(test_data['X_raw'][te_mask])
            raw_tr = raw_scaler.transform(train_data['X_raw'][tr_mask])
            pls = PLSRegression(n_components=nc)
            pls.fit(raw_tr, y_raw_t[tr_mask])
            X_pls_te[te_mask, :nc] = pls.transform(raw_te)

        fs_key_pls = f"pls_{target}"
        feature_spaces_target = dict(feature_spaces)
        feature_spaces_target[fs_key_pls] = (X_pls_tr, X_pls_te)

        # 10. LASSO-selected features
        opt_frame = TARGET_FRAMES[target]
        print(f"  [lasso_{target}] LASSO stability selection at frame {opt_frame}...")
        X_lasso_tr, lasso_idx = feat_lasso_selected(train_data, y_raw_t, opt_frame, n_select=30)
        X_lasso_te = feat_hc_at_frame(test_data, opt_frame)[:, lasso_idx]
        fs_key_lasso = f"lasso_{target}"
        feature_spaces_target[fs_key_lasso] = (X_lasso_tr, X_lasso_te)

        print(f"    LASSO selected {len(lasso_idx)} features")

        # Generate all OOF predictions
        print(f"\n  Generating OOF predictions for {target}...")
        oof_dict = {}
        test_dict = {}
        n_total = len(feature_spaces_target) * len(model_configs)
        count = 0

        for fs_name, (X_tr, X_te) in feature_spaces_target.items():
            for model_name, model_factory in model_configs.items():
                key = f"{fs_name}__{model_name}"
                count += 1
                oof, test_pred = generate_oof_lopo(X_tr, y_t, pids_train, X_te, model_factory)
                mse = np.mean((oof - y_t) ** 2)
                oof_dict[key] = oof
                test_dict[key] = test_pred

                if count % 20 == 0 or count == n_total:
                    print(f"    [{count}/{n_total}] {key}: LOPO MSE={mse:.6f}")

        print(f"\n  Total base models: {len(oof_dict)}")

        # MSE summary
        mses = {k: np.mean((v - y_t) ** 2) for k, v in oof_dict.items()}
        sorted_models = sorted(mses.items(), key=lambda x: x[1])
        print(f"\n  Top 10 base models by LOPO MSE:")
        for i, (name, mse) in enumerate(sorted_models[:10]):
            print(f"    {i+1}. {name}: {mse:.6f}")

        print(f"\n  Bottom 5 base models (worst):")
        for name, mse in sorted_models[-5:]:
            print(f"    {name}: {mse:.6f}")

        # Diversity analysis
        print(f"\n  Diversity analysis:")
        R, names = compute_diversity_matrix(oof_dict)
        mask_upper = np.triu(np.ones_like(R, dtype=bool), k=1)
        r_values = R[mask_upper]
        print(f"    Pairwise r: min={r_values.min():.4f}, max={r_values.max():.4f}, "
              f"mean={r_values.mean():.4f}, median={np.median(r_values):.4f}")

        # Count models with low correlation
        low_div = np.sum(r_values < 0.80)
        total_pairs = len(r_values)
        print(f"    Pairs with r < 0.80: {low_div}/{total_pairs} ({100*low_div/total_pairs:.1f}%)")
        print(f"    Pairs with r < 0.90: {np.sum(r_values < 0.90)}/{total_pairs}")

        # Feature space diversity (average r within each feature space)
        print(f"\n  Feature space diversity (avg within-space r):")
        fs_names = sorted(set(k.split('__')[0] for k in oof_dict.keys()))
        for fs in fs_names:
            fs_models = [k for k in oof_dict.keys() if k.startswith(fs + '__')]
            if len(fs_models) < 2:
                continue
            rs = []
            for i in range(len(fs_models)):
                for j in range(i+1, len(fs_models)):
                    r = np.corrcoef(oof_dict[fs_models[i]], oof_dict[fs_models[j]])[0, 1]
                    rs.append(r)
            print(f"    {fs}: {np.mean(rs):.4f} (n_models={len(fs_models)})")

        # Cross-feature-space diversity
        print(f"\n  Cross-feature-space diversity (avg between-space r):")
        for i, fs1 in enumerate(fs_names):
            for j, fs2 in enumerate(fs_names):
                if j <= i:
                    continue
                models1 = [k for k in oof_dict.keys() if k.startswith(fs1 + '__')]
                models2 = [k for k in oof_dict.keys() if k.startswith(fs2 + '__')]
                rs = []
                for m1 in models1:
                    for m2 in models2:
                        r = np.corrcoef(oof_dict[m1], oof_dict[m2])[0, 1]
                        rs.append(r)
                if np.mean(rs) < 0.85:
                    print(f"    {fs1} x {fs2}: {np.mean(rs):.4f} ***LOW***")

        # Greedy diverse selection
        print(f"\n  Greedy diverse model selection:")
        selected, sel_mses = greedy_diverse_selection(oof_dict, y_t, max_models=15)
        print(f"    Selected {len(selected)} models:")
        for i, name in enumerate(selected):
            # Compute max r with previous models
            if i == 0:
                print(f"      {i+1}. {name}: MSE={sel_mses[name]:.6f} (seed)")
            else:
                max_r = max(np.corrcoef(oof_dict[name], oof_dict[s])[0, 1]
                           for s in selected[:i])
                print(f"      {i+1}. {name}: MSE={sel_mses[name]:.6f}, max_r_prev={max_r:.4f}")

        # Build meta-learner OOF matrix
        oof_matrix = np.column_stack([oof_dict[k] for k in selected])
        test_matrix = np.column_stack([test_dict[k] for k in selected])

        # Train meta-learner with nested LOPO
        print(f"\n  Training meta-learner (Ridge) on {len(selected)} models...")
        best_meta_mse = float('inf')
        best_meta_alpha = 10.0
        best_meta_oof = None
        best_meta_test = None

        for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
            meta_oof, meta_test = train_meta_learner(
                oof_matrix, y_t, pids_train, test_matrix, alpha=alpha)
            meta_mse = np.mean((meta_oof - y_t) ** 2)
            print(f"    alpha={alpha:.1f}: LOPO MSE={meta_mse:.6f}")
            if meta_mse < best_meta_mse:
                best_meta_mse = meta_mse
                best_meta_alpha = alpha
                best_meta_oof = meta_oof
                best_meta_test = meta_test

        print(f"    BEST: alpha={best_meta_alpha}, MSE={best_meta_mse:.6f}")

        # Also try simple average of selected models as baseline
        simple_avg_oof = np.mean(oof_matrix, axis=1)
        simple_avg_test = np.mean(test_matrix, axis=1)
        simple_avg_mse = np.mean((simple_avg_oof - y_t) ** 2)
        print(f"    Simple average: MSE={simple_avg_mse:.6f}")

        # Also try top-1 model
        top1_name = sorted_models[0][0]
        top1_mse = sorted_models[0][1]
        print(f"    Top-1 model ({top1_name}): MSE={top1_mse:.6f}")

        # Choose best
        if best_meta_mse <= simple_avg_mse and best_meta_mse <= top1_mse:
            final_oof = best_meta_oof
            final_test = best_meta_test
            final_mse = best_meta_mse
            final_method = f"meta_ridge_a{best_meta_alpha}"
        elif simple_avg_mse <= top1_mse:
            final_oof = simple_avg_oof
            final_test = simple_avg_test
            final_mse = simple_avg_mse
            final_method = "simple_avg"
        else:
            final_oof = oof_dict[top1_name]
            final_test = test_dict[top1_name]
            final_mse = top1_mse
            final_method = f"top1_{top1_name}"

        print(f"\n  FINAL for {target}: {final_method} (MSE={final_mse:.6f})")

        final_results[target] = {
            'oof': final_oof,
            'test': final_test,
            'mse': final_mse,
            'method': final_method,
            'meta_oof': best_meta_oof,
            'meta_test': best_meta_test,
            'meta_mse': best_meta_mse,
            'avg_oof': simple_avg_oof,
            'avg_test': simple_avg_test,
            'avg_mse': simple_avg_mse,
            'top1_oof': oof_dict[top1_name],
            'top1_test': test_dict[top1_name],
            'top1_mse': top1_mse,
            'all_oof': oof_dict,
            'all_test': test_dict,
            'selected_models': selected,
        }

    # ------------------------------------------------------------------
    # OVERALL RESULTS
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("OVERALL RESULTS")
    print(f"{'=' * 70}")

    for target in TARGETS:
        r = final_results[target]
        print(f"  {target}: MSE={r['mse']:.6f} ({r['method']})")
        print(f"    meta={r['meta_mse']:.6f}, avg={r['avg_mse']:.6f}, top1={r['top1_mse']:.6f}")
    mean_mse = np.mean([final_results[t]['mse'] for t in TARGETS])
    print(f"  MEAN: {mean_mse:.6f}")

    meta_mean = np.mean([final_results[t]['meta_mse'] for t in TARGETS])
    avg_mean = np.mean([final_results[t]['avg_mse'] for t in TARGETS])
    top1_mean = np.mean([final_results[t]['top1_mse'] for t in TARGETS])
    print(f"\n  Meta-learner mean: {meta_mean:.6f}")
    print(f"  Simple avg mean:  {avg_mean:.6f}")
    print(f"  Top-1 mean:       {top1_mean:.6f}")

    # ------------------------------------------------------------------
    # DIVERSITY WITH REFERENCE SUBMISSIONS
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("DIVERSITY WITH REFERENCE SUBMISSIONS")
    print(f"{'=' * 70}")

    # For this we need to compare test predictions
    for label, sub_ref in [("Sub 1350", sub_1350), ("Sub 1640", sub_1640), ("Sub 784", sub_784)]:
        print(f"\n  Correlation with {label}:")
        for target in TARGETS:
            col = f"scaled_{target}"
            for variant, desc in [
                ('meta_test', 'meta-learner'),
                ('avg_test', 'simple avg'),
                ('top1_test', 'top-1'),
            ]:
                r = np.corrcoef(sub_ref[col].values, final_results[target][variant])[0, 1]
                if r < 0.85:
                    marker = " ***LOW CORRELATION***"
                elif r < 0.90:
                    marker = " **somewhat diverse**"
                else:
                    marker = ""
                print(f"    {target} ({desc}): r={r:.4f}{marker}")

    # ------------------------------------------------------------------
    # GENERATE SUBMISSIONS
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    test_ids = test_data['ids']

    # 1. Standalone stacked prediction (meta-learner)
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': final_results['angle']['meta_test'],
        'scaled_depth': final_results['depth']['meta_test'],
        'scaled_left_right': final_results['left_right']['meta_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: STANDALONE META-LEARNER")
    print(f"    LOPO MSE: angle={final_results['angle']['meta_mse']:.6f}, "
          f"depth={final_results['depth']['meta_mse']:.6f}, "
          f"lr={final_results['left_right']['meta_mse']:.6f}, "
          f"mean={meta_mean:.6f}")
    standalone_num = sub_num

    # 2. Standalone simple average
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': final_results['angle']['avg_test'],
        'scaled_depth': final_results['depth']['avg_test'],
        'scaled_left_right': final_results['left_right']['avg_test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: STANDALONE SIMPLE AVERAGE")
    print(f"    LOPO MSE: angle={final_results['angle']['avg_mse']:.6f}, "
          f"depth={final_results['depth']['avg_mse']:.6f}, "
          f"lr={final_results['left_right']['avg_mse']:.6f}, "
          f"mean={avg_mean:.6f}")

    # 3. Standalone best-per-target
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': final_results['angle']['test'],
        'scaled_depth': final_results['depth']['test'],
        'scaled_left_right': final_results['left_right']['test'],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"\n  Sub {sub_num}: STANDALONE BEST-PER-TARGET")
    print(f"    LOPO MSE: angle={final_results['angle']['mse']:.6f}, "
          f"depth={final_results['depth']['mse']:.6f}, "
          f"lr={final_results['left_right']['mse']:.6f}, "
          f"mean={mean_mse:.6f}")

    # 4. Blended with Sub 1350
    print(f"\n  Blends with Sub 1350 (LB 0.006776):")
    for pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': (1-pct)*sub_1350['scaled_angle'].values + pct*final_results['angle']['meta_test'],
            'scaled_depth': (1-pct)*sub_1350['scaled_depth'].values + pct*final_results['depth']['meta_test'],
            'scaled_left_right': (1-pct)*sub_1350['scaled_left_right'].values + pct*final_results['left_right']['meta_test'],
        })
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"    Sub {sub_num}: {int(pct*100)}% stack + {int((1-pct)*100)}% Sub1350")

    # 5. Blended with Sub 1640
    print(f"\n  Blends with Sub 1640 (LB 0.006698):")
    for pct in [0.05, 0.10, 0.15, 0.20]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': (1-pct)*sub_1640['scaled_angle'].values + pct*final_results['angle']['meta_test'],
            'scaled_depth': (1-pct)*sub_1640['scaled_depth'].values + pct*final_results['depth']['meta_test'],
            'scaled_left_right': (1-pct)*sub_1640['scaled_left_right'].values + pct*final_results['left_right']['meta_test'],
        })
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"    Sub {sub_num}: {int(pct*100)}% stack + {int((1-pct)*100)}% Sub1640")

    # 6. Target-specific blends with Sub 1350 (only touch depth and LR)
    print(f"\n  Target-specific blends with Sub 1350:")
    for dw, lw in [(0.10, 0.10), (0.15, 0.20), (0.20, 0.30), (0.30, 0.50)]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': sub_1350['scaled_angle'].values,  # keep angle from 1350
            'scaled_depth': (1-dw)*sub_1350['scaled_depth'].values + dw*final_results['depth']['meta_test'],
            'scaled_left_right': (1-lw)*sub_1350['scaled_left_right'].values + lw*final_results['left_right']['meta_test'],
        })
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"    Sub {sub_num}: angle=1350, depth={int(dw*100)}%stack, lr={int(lw*100)}%stack")

    # 7. Target-specific blends with Sub 1640
    print(f"\n  Target-specific blends with Sub 1640:")
    for dw, lw in [(0.10, 0.10), (0.15, 0.20), (0.20, 0.30)]:
        sub_num = get_next_submission_number()
        blended = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': sub_1640['scaled_angle'].values,  # keep angle from 1640
            'scaled_depth': (1-dw)*sub_1640['scaled_depth'].values + dw*final_results['depth']['meta_test'],
            'scaled_left_right': (1-lw)*sub_1640['scaled_left_right'].values + lw*final_results['left_right']['meta_test'],
        })
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"    Sub {sub_num}: angle=1640, depth={int(dw*100)}%stack, lr={int(lw*100)}%stack")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Feature spaces: {len(feature_spaces)} shared + 2 per-target = "
          f"{len(feature_spaces) + 2 * len(TARGETS)} total")
    print(f"  Models per feature space: {len(model_configs)}")
    for target in TARGETS:
        n_base = len(final_results[target]['all_oof'])
        n_sel = len(final_results[target]['selected_models'])
        print(f"  {target}: {n_base} base models, {n_sel} selected for stacking")
    print(f"\n  Best LOPO MSE (meta-learner): {meta_mean:.6f}")
    print(f"  Standalone submission: Sub {standalone_num}")


if __name__ == "__main__":
    main()
