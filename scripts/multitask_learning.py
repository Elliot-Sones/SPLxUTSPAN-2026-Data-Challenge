"""
Multi-Task Learning for Basketball Shot Prediction

Joint modeling of angle/depth/left_right to leverage shared representations
and target correlations.

Approaches:
1. Shared Ridge Regression (MTL-Ridge): Single model predicts all 3 targets
2. Soft Parameter Sharing (Neural MTL): Shared layers + task-specific heads
3. Auxiliary Tasks: Add make/miss classification, release timing prediction
4. Correlation-Aware Loss: Weight targets by their correlation structure
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
from sklearn.linear_model import Ridge, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

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

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        X_raw = np.zeros((n, len(keypoint_cols) * 240), dtype=np.float32)
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


def extract_features_unified(ts_3d, ts_hr, kp_index, release_frame):
    """Extract features at all three target-specific frames and concatenate."""
    all_feats = []

    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        f = int(np.clip(frame, 0, 239))
        feats = []

        key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                      'left_wrist', 'left_shoulder',
                      'right_hip', 'left_hip', 'mid_hip',
                      'right_knee', 'left_knee', 'neck', 'nose']

        # Hoop-relative positions + velocities at frame
        for jname in key_joints:
            idx = kp_index.get(jname)
            if idx is None:
                feats.extend([0.0] * 6)
                continue
            for coord in range(3):
                feats.append(ts_hr[f, idx, coord])
                vel = np.gradient(ts_hr[:, idx, coord], DT)
                feats.append(vel[f])

        # Summary stats
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

        all_feats.extend(feats)

    # Shared features (only once)
    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')

    # Release frame timing
    all_feats.append(release_frame)

    # Release window dynamics
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            all_feats.append(np.nanmean(series[140:180]))
            all_feats.append(np.nanmax(vel[140:180]))
    else:
        all_feats.extend([0.0] * 6)

    return np.array(all_feats, dtype=np.float32)


def extract_all_features_unified(data):
    """Extract unified feature set for all shots."""
    n = len(data['pids'])
    kp_index = data['kp_index']

    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features_unified(ts_3d, ts_hr, kp_index, rf)
        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


# ==============================================================
# AUXILIARY TASKS
# ==============================================================

def create_auxiliary_targets(train_data, y_train):
    """Create auxiliary task targets."""
    # 1. Make/miss classification (simplified: high angle/depth = make)
    angles = y_train[:, 0]
    depths = y_train[:, 1]
    make_score = (angles + depths) / 2  # Higher = more likely to make
    make_binary = (make_score > np.median(make_score)).astype(np.float32)

    # 2. Release timing
    release_frames = []
    for i in range(len(train_data['pids'])):
        ts_3d = train_data['X_3d'][i]
        rf = detect_release_frame(ts_3d, train_data['kp_index'])
        release_frames.append(rf)
    release_timing = np.array(release_frames, dtype=np.float32) / 240.0  # Normalize

    return make_binary, release_timing


# ==============================================================
# MULTI-TASK MODELS
# ==============================================================

def train_mtl_ridge_per_player(X_train, y_train, X_test, pids_train, pids_test,
                                alpha=10.0, fit_intercept=True):
    """Multi-task ridge regression per player."""
    unique_pids = sorted(np.unique(pids_train))
    n_targets = y_train.shape[1]

    oof = np.zeros((len(X_train), n_targets))
    test_preds = np.zeros((len(X_test), n_targets))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        X_p = X_train[tr_mask]
        y_p = y_train[tr_mask]
        indices = np.where(tr_mask)[0]

        # Standardize
        scaler_x = StandardScaler()
        X_s = scaler_x.fit_transform(X_p)

        # 5-fold CV for OOF
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_s):
            model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
            model.fit(X_s[tr_idx], y_p[tr_idx])
            oof[indices[val_idx]] = model.predict(X_s[val_idx])

        # Test predictions
        if np.any(te_mask):
            X_te_s = scaler_x.transform(X_test[te_mask])
            model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
            model.fit(X_s, y_p)
            test_preds[te_mask] = model.predict(X_te_s)

    return oof, test_preds


def train_mtl_elasticnet_per_player(X_train, y_train, X_test, pids_train, pids_test,
                                     alpha=1.0, l1_ratio=0.5):
    """Multi-task elastic net per player."""
    unique_pids = sorted(np.unique(pids_train))
    n_targets = y_train.shape[1]

    oof = np.zeros((len(X_train), n_targets))
    test_preds = np.zeros((len(X_test), n_targets))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid

        X_p = X_train[tr_mask]
        y_p = y_train[tr_mask]
        indices = np.where(tr_mask)[0]

        scaler_x = StandardScaler()
        X_s = scaler_x.fit_transform(X_p)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_s):
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                        max_iter=2000, random_state=42)
            model.fit(X_s[tr_idx], y_p[tr_idx])
            oof[indices[val_idx]] = model.predict(X_s[val_idx])

        if np.any(te_mask):
            X_te_s = scaler_x.transform(X_test[te_mask])
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                        max_iter=2000, random_state=42)
            model.fit(X_s, y_p)
            test_preds[te_mask] = model.predict(X_te_s)

    return oof, test_preds


def train_mtl_with_auxiliary(X_train, y_train, aux_targets, X_test, pids_train, pids_test):
    """Multi-task with auxiliary tasks."""
    make_binary, release_timing = aux_targets

    # Concatenate auxiliary targets
    y_augmented = np.hstack([y_train, make_binary.reshape(-1, 1), release_timing.reshape(-1, 1)])

    # Train multi-task model
    oof_aug, test_aug = train_mtl_ridge_per_player(
        X_train, y_augmented, X_test, pids_train, pids_test, alpha=10.0)

    # Extract only the main 3 targets
    oof = oof_aug[:, :3]
    test_preds = test_aug[:, :3]

    return oof, test_preds


def train_correlation_weighted_mtl(X_train, y_train, X_test, pids_train, pids_test):
    """Multi-task with correlation-aware loss weighting."""
    # Compute target correlations
    corr_matrix = np.corrcoef(y_train.T)
    print(f"  Target correlation matrix:\n{corr_matrix}")

    # Weight targets inversely by their correlation with others
    # Higher correlation = lower weight (avoid redundant learning)
    weights = 1.0 / (np.abs(corr_matrix).sum(axis=1) - 1 + 1e-6)
    weights /= weights.sum()
    print(f"  Target weights: angle={weights[0]:.3f}, depth={weights[1]:.3f}, lr={weights[2]:.3f}")

    # Train weighted models per target (single-output)
    unique_pids = sorted(np.unique(pids_train))
    oof = np.zeros((len(X_train), 3))
    test_preds = np.zeros((len(X_test), 3))

    for t_idx in range(3):
        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid

            X_p = X_train[tr_mask]
            y_p = y_train[tr_mask, t_idx]
            indices = np.where(tr_mask)[0]

            scaler_x = StandardScaler()
            X_s = scaler_x.fit_transform(X_p)

            # 5-fold CV for OOF
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, val_idx in kf.split(X_s):
                model = Ridge(alpha=10.0 / weights[t_idx], fit_intercept=True)
                model.fit(X_s[tr_idx], y_p[tr_idx])
                oof[indices[val_idx], t_idx] = model.predict(X_s[val_idx])

            # Test predictions
            if np.any(te_mask):
                X_te_s = scaler_x.transform(X_test[te_mask])
                model = Ridge(alpha=10.0 / weights[t_idx], fit_intercept=True)
                model.fit(X_s, y_p)
                test_preds[te_mask, t_idx] = model.predict(X_te_s)

    return oof, test_preds


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("MULTI-TASK LEARNING PIPELINE")
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

    # Scale targets
    y_scaled = np.zeros_like(y_train)
    for i, target in enumerate(TARGETS):
        y_scaled[:, i] = scalers[target].transform(
            y_train[:, i].reshape(-1, 1)).ravel()

    # Extract unified features
    print("\nExtracting unified features...")
    X_train, rf_train = extract_all_features_unified(train_data)
    X_test, rf_test = extract_all_features_unified(test_data)
    print(f"  Features: {X_train.shape[1]}")

    # Add PLS components (per-target)
    print("\nAdding PLS components...")
    pls_train_all = []
    pls_test_all = []

    for t_idx, target in enumerate(TARGETS):
        unique_pids = sorted(np.unique(pids_train))
        max_nc = 15
        pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
        pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

        for pid in unique_pids:
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            n_p = tr_mask.sum()

            scaler = StandardScaler()
            raw_tr = scaler.fit_transform(train_data['X_raw'][tr_mask])
            raw_te = scaler.transform(test_data['X_raw'][te_mask])

            nc = min(10, n_p - n_p // 5 - 1)
            nc = max(3, nc)

            pls = PLSRegression(n_components=nc)
            pls.fit(raw_tr, y_train[tr_mask, t_idx])
            pls_train[tr_mask, :nc] = pls.transform(raw_tr)
            pls_test[te_mask, :nc] = pls.transform(raw_te)

        pls_train_all.append(pls_train)
        pls_test_all.append(pls_test)

    pls_train_concat = np.hstack(pls_train_all)
    pls_test_concat = np.hstack(pls_test_all)

    X_train_aug = np.hstack([X_train, pls_train_concat])
    X_test_aug = np.hstack([X_test, pls_test_concat])
    print(f"  Augmented features: {X_train_aug.shape[1]}")

    # Create auxiliary targets
    print("\nCreating auxiliary targets...")
    aux_targets = create_auxiliary_targets(train_data, y_train)

    # ============================================================
    # MULTI-TASK APPROACHES
    # ============================================================

    results = {}

    # Approach 1: Basic multi-task ridge
    print("\n[1] Multi-task Ridge Regression...")
    oof_mtl_ridge, test_mtl_ridge = train_mtl_ridge_per_player(
        X_train_aug, y_scaled, X_test_aug, pids_train, pids_test, alpha=10.0)

    mse_per_target = []
    for i, target in enumerate(TARGETS):
        mse = np.mean((oof_mtl_ridge[:, i] - y_scaled[:, i]) ** 2)
        mse_per_target.append(mse)
        print(f"  {target}: MSE={mse:.6f}")
    mean_mse = np.mean(mse_per_target)
    print(f"  MEAN: {mean_mse:.6f}")

    results['mtl_ridge'] = {
        'oof': oof_mtl_ridge,
        'test': test_mtl_ridge,
        'mse': mean_mse,
        'mse_per_target': mse_per_target,
    }

    # Try different alpha values
    print("\n  Testing different alpha values...")
    best_alpha, best_mse = 10.0, mean_mse
    for alpha in [1.0, 5.0, 20.0, 50.0, 100.0]:
        oof_tmp, _ = train_mtl_ridge_per_player(
            X_train_aug, y_scaled, X_test_aug, pids_train, pids_test, alpha=alpha)
        mse_tmp = np.mean([np.mean((oof_tmp[:, i] - y_scaled[:, i]) ** 2) for i in range(3)])
        print(f"    alpha={alpha}: MSE={mse_tmp:.6f}")
        if mse_tmp < best_mse:
            best_mse = mse_tmp
            best_alpha = alpha

    if best_alpha != 10.0:
        print(f"  Best alpha: {best_alpha}")
        oof_mtl_ridge, test_mtl_ridge = train_mtl_ridge_per_player(
            X_train_aug, y_scaled, X_test_aug, pids_train, pids_test, alpha=best_alpha)
        results['mtl_ridge']['oof'] = oof_mtl_ridge
        results['mtl_ridge']['test'] = test_mtl_ridge
        results['mtl_ridge']['mse'] = best_mse

    # Approach 2: Multi-task elastic net
    print("\n[2] Multi-task Elastic Net...")
    oof_mtl_enet, test_mtl_enet = train_mtl_elasticnet_per_player(
        X_train_aug, y_scaled, X_test_aug, pids_train, pids_test, alpha=1.0, l1_ratio=0.5)

    mse_per_target = []
    for i, target in enumerate(TARGETS):
        mse = np.mean((oof_mtl_enet[:, i] - y_scaled[:, i]) ** 2)
        mse_per_target.append(mse)
        print(f"  {target}: MSE={mse:.6f}")
    mean_mse = np.mean(mse_per_target)
    print(f"  MEAN: {mean_mse:.6f}")

    results['mtl_enet'] = {
        'oof': oof_mtl_enet,
        'test': test_mtl_enet,
        'mse': mean_mse,
        'mse_per_target': mse_per_target,
    }

    # Approach 3: Multi-task with auxiliary tasks
    print("\n[3] Multi-task with Auxiliary Tasks...")
    oof_mtl_aux, test_mtl_aux = train_mtl_with_auxiliary(
        X_train_aug, y_scaled, aux_targets, X_test_aug, pids_train, pids_test)

    mse_per_target = []
    for i, target in enumerate(TARGETS):
        mse = np.mean((oof_mtl_aux[:, i] - y_scaled[:, i]) ** 2)
        mse_per_target.append(mse)
        print(f"  {target}: MSE={mse:.6f}")
    mean_mse = np.mean(mse_per_target)
    print(f"  MEAN: {mean_mse:.6f}")

    results['mtl_aux'] = {
        'oof': oof_mtl_aux,
        'test': test_mtl_aux,
        'mse': mean_mse,
        'mse_per_target': mse_per_target,
    }

    # Approach 4: Correlation-weighted MTL
    print("\n[4] Correlation-Weighted Multi-task...")
    oof_mtl_corr, test_mtl_corr = train_correlation_weighted_mtl(
        X_train_aug, y_scaled, X_test_aug, pids_train, pids_test)

    mse_per_target = []
    for i, target in enumerate(TARGETS):
        mse = np.mean((oof_mtl_corr[:, i] - y_scaled[:, i]) ** 2)
        mse_per_target.append(mse)
        print(f"  {target}: MSE={mse:.6f}")
    mean_mse = np.mean(mse_per_target)
    print(f"  MEAN: {mean_mse:.6f}")

    results['mtl_corr'] = {
        'oof': oof_mtl_corr,
        'test': test_mtl_corr,
        'mse': mean_mse,
        'mse_per_target': mse_per_target,
    }

    # ============================================================
    # DIVERSITY ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS")
    print("=" * 70)

    # Load Sub 1350 and Sub 784 for comparison
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    for approach_name, res in results.items():
        print(f"\n{approach_name}:")
        for i, target in enumerate(TARGETS):
            col = f'scaled_{target}'
            r_784 = np.corrcoef(sub_784[col].values, res['test'][:, i])[0, 1]
            r_1350 = np.corrcoef(sub_1350[col].values, res['test'][:, i])[0, 1]
            print(f"  {target}: r(Sub784)={r_784:.4f}, r(Sub1350)={r_1350:.4f}")

    # ============================================================
    # GENERATE SUBMISSIONS
    # ============================================================

    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Standalone submissions for each approach
    for approach_name, res in results.items():
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': test_data['ids'],
            'scaled_angle': res['test'][:, 0],
            'scaled_depth': res['test'][:, 1],
            'scaled_left_right': res['test'][:, 2],
        })
        sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {approach_name} (standalone)")
        print(f"    CV MSE: {res['mse']:.6f}")
        print(f"    angle_std={sub['scaled_angle'].std():.6f}, depth_mean={sub['scaled_depth'].mean():.6f}")

    # Blends with Sub 784
    print("\n  Blending with Sub 784:")

    for approach_name, res in results.items():
        for aw, dw, lw, desc in [
            (0.00, 0.30, 0.50, "Sub 784 weights"),
            (0.00, 0.20, 0.30, "conservative"),
            (0.10, 0.30, 0.50, "with angle"),
        ]:
            sub_num = get_next_submission_number()
            blended = sub_784.copy()
            blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*res['test'][:, 0]
            blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*res['test'][:, 1]
            blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*res['test'][:, 2]

            blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
            print(f"  Sub {sub_num}: {approach_name} blend aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")

    # Ensemble across MTL approaches
    print("\n  Ensembling MTL approaches:")
    ensemble_test = np.mean([res['test'] for res in results.values()], axis=0)

    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': ensemble_test[:, 0],
        'scaled_depth': ensemble_test[:, 1],
        'scaled_left_right': ensemble_test[:, 2],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: MTL ensemble (mean)")

    # Blend ensemble with Sub 784
    for aw, dw, lw, desc in [
        (0.00, 0.30, 0.50, "Sub 784 weights"),
        (0.00, 0.20, 0.30, "conservative"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = (1-aw)*sub_784['scaled_angle'] + aw*ensemble_test[:, 0]
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*ensemble_test[:, 1]
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*ensemble_test[:, 2]

        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: MTL ensemble blend aw={aw:.2f} dw={dw:.2f} lw={lw:.2f} ({desc})")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total submissions generated: {len(results) + len(results)*3 + 3}")


if __name__ == "__main__":
    main()
