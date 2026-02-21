"""
Player-Adaptive Metric Learning for Biomechanical Shot Prediction

Research contribution: Instead of a uniform Euclidean distance in the kernel,
learn a player-specific Mahalanobis-like metric from correlation structure.

For each player x target, weight each feature by its empirical correlation
with the target (computed from that player's training shots only):

  w_i(player) = |r(feature_i, target)| for that player

Then the kernel distance becomes:
  d_weighted = sqrt(sum_i w_i * (x_i - y_i)^2 / sum(w))

This means:
- P1's kernel weights hip lateral velocity by |r|=0.669 (the strongest predictor)
- P5's kernel weights wrist lateral velocity by |r|=0.585
- Features uncorrelated with a player's outcomes get near-zero weight

This is fundamentally different from:
1. Standard kernel (equal feature weights) - misses player-specific channels
2. Per-player feature selection (binary threshold) - unstable with small N

Adaptive weighting is a soft, stable version that gracefully incorporates
player-specific information channels discovered in the 1189-feature analysis.

Reference: Related to Metric Learning (Xing et al., 2002) and Locally Weighted
Regression (Atkeson et al., 1997), extended with player-adaptive feature weights.
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
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


def safe_savgol(x, window=11, polyorder=3, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
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
        ids, pids, targets = [], [], []
        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
        X_3d = np.nan_to_num(X_3d, nan=0.0)
        result = {'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    train = process(train_df, True)
    test = process(test_df, False)
    return train, test


# ============================================================
# FEATURE EXTRACTION
# All features extracted from smoothed trajectories to avoid
# NaN-fill discontinuities in velocity computation.
# ============================================================

def get_hoop_transform(ts_smooth, kp_index):
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_smooth[120, mid_hip_idx, :].copy()
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
    centered = ts_smooth - player_pos.reshape(1, 1, 3)
    return np.einsum('ij,fkj->fki', R, centered)


def extract_features_for_shot(ts_3d, kp_index, target_frame):
    """
    Extract combined feature set for one shot:
    - Core: hoop-relative position + velocity for 12 key joints at target frame (72 features)
    - Augmented: lateral velocity/position at follow-through frames (20 features)
    All computed from smoothed trajectories.

    Returns: feature vector, feature names
    """
    # Smooth ALL trajectories first
    n_kp = ts_3d.shape[1]
    ts_smooth = np.zeros_like(ts_3d, dtype=np.float64)
    for k in range(n_kp):
        for ax in range(3):
            ts_smooth[:, k, ax] = safe_savgol(ts_3d[:, k, ax].astype(np.float64))

    ts_hr = get_hoop_transform(ts_smooth, kp_index)

    # Velocity in hoop-relative frame (central difference, smoothed)
    vel_hr = np.zeros_like(ts_hr)
    for k in range(n_kp):
        for ax in range(3):
            vel_hr[:, k, ax] = safe_savgol(ts_hr[:, k, ax], deriv=1, delta=DT)

    f = int(np.clip(target_frame, 1, 238))

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    feats = []
    names = []

    # --- CORE FEATURES: position + velocity at target frame (72 features) ---
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is not None:
            pos = ts_hr[f, idx, :]
            vel = vel_hr[f, idx, :]
        else:
            pos = np.zeros(3)
            vel = np.zeros(3)
        feats.extend(pos.tolist())
        feats.extend(vel.tolist())
        names.extend([f"hr_{jname}_{c}_f{f}" for c in ['x','y','z']] +
                     [f"vel_{jname}_{c}_f{f}" for c in ['x','y','z']])

    # --- AUGMENTED FEATURES: player-specific channels at follow-through ---
    # These were discovered via 1189-feature correlation analysis.
    # Feature specificity: for P1 LR, hip_y at f175 has r=0.669, rank #1.
    #                      For P5 LR, wrist_y at f175 has r=0.573, rank #5.
    #                      Universal features have |r|~0.3, rank ~300-600.

    ft_frames = [153, 155, 160, 165, 170, 175, 180]

    # Right hip lateral (y) velocity - P1 LR channel
    rh_idx = kp_index.get('right_hip')
    mh_idx = kp_index.get('mid_hip')
    for ff in ft_frames:
        ff = int(np.clip(ff, 1, 238))
        v = float(vel_hr[ff, rh_idx, 1]) if rh_idx is not None else 0.0
        feats.append(v)
        names.append(f"vel_rh_y_f{ff}")
        v = float(vel_hr[ff, mh_idx, 1]) if mh_idx is not None else 0.0
        feats.append(v)
        names.append(f"vel_mh_y_f{ff}")

    # Right wrist lateral (y) velocity - P5 LR channel
    rw_idx = kp_index.get('right_wrist')
    for ff in ft_frames:
        ff = int(np.clip(ff, 1, 238))
        v = float(vel_hr[ff, rw_idx, 1]) if rw_idx is not None else 0.0
        feats.append(v)
        names.append(f"vel_rw_y_f{ff}")

    # Right elbow lateral (y) velocity
    re_idx = kp_index.get('right_elbow')
    for ff in [155, 160, 165, 170, 175]:
        ff = int(np.clip(ff, 1, 238))
        v = float(vel_hr[ff, re_idx, 1]) if re_idx is not None else 0.0
        feats.append(v)
        names.append(f"vel_re_y_f{ff}")

    # Neck/left-shoulder lateral (y) position - P4 ANGLE channel
    neck_idx = kp_index.get('neck')
    ls_idx = kp_index.get('left_shoulder')
    for ff in [150, 153, 155, 160, 165, 170]:
        ff = int(np.clip(ff, 1, 238))
        p = float(ts_hr[ff, neck_idx, 1]) if neck_idx is not None else 0.0
        feats.append(p)
        names.append(f"hr_neck_y_f{ff}")
        p = float(ts_hr[ff, ls_idx, 1]) if ls_idx is not None else 0.0
        feats.append(p)
        names.append(f"hr_ls_y_f{ff}")

    # Left wrist lateral (y) position - P5 ANGLE channel (guide hand)
    lw_idx = kp_index.get('left_wrist')
    for ff in [140, 145, 150, 153, 155, 160]:
        ff = int(np.clip(ff, 1, 238))
        p = float(ts_hr[ff, lw_idx, 1]) if lw_idx is not None else 0.0
        feats.append(p)
        names.append(f"hr_lw_y_f{ff}")

    return np.array(feats, dtype=np.float32), names


def build_feature_matrix(data, target_frame):
    n = data['X_3d'].shape[0]
    kp_index = data['kp_index']
    all_feats = []
    feat_names = None
    print(f"  Extracting features for {n} shots (frame={target_frame})...")
    for i in range(n):
        f, names = extract_features_for_shot(data['X_3d'][i], kp_index, target_frame)
        all_feats.append(f)
        if feat_names is None:
            feat_names = names
    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_names


# ============================================================
# PLAYER-ADAPTIVE METRIC LEARNING
#
# Key insight: the Gaussian kernel d(x,y) = exp(-||x-y||^2 / (2*bw^2))
# uses Euclidean distance which weighs all features equally.
#
# We replace this with a player-specific weighted distance:
#   d_w(x,y) = sqrt(sum_i w_i(player) * (x_i - y_i)^2)
#
# where w_i(player) = |r(feature_i, target)| for that player's training shots.
#
# This is a diagonal Mahalanobis metric learned from correlation structure.
# ============================================================

def compute_player_weights(X_tr, y_tr, feat_names, eps=0.01):
    """
    Compute per-feature weights for one player from their training data.
    w_i = |Pearson_r(X_tr[:,i], y_tr)| + eps (eps prevents zero weights)
    """
    n_feats = X_tr.shape[1]
    weights = np.full(n_feats, eps, dtype=np.float64)
    for fi in range(n_feats):
        fv = X_tr[:, fi]
        if np.std(fv) > 1e-8 and np.std(y_tr) > 1e-8:
            r = float(np.corrcoef(fv, y_tr)[0, 1])
            if not np.isnan(r):
                weights[fi] = abs(r) + eps
    return weights


def weighted_euclidean_distance(X1, X2, weights):
    """
    Compute weighted Euclidean distance matrix.
    d_w(x,y) = sqrt(sum_i w_i * (x_i - y_i)^2)
    """
    # weights: (n_feats,)
    # X1: (n1, n_feats), X2: (n2, n_feats)
    w = weights / weights.sum()  # normalize so total weight = 1
    X1_w = X1 * np.sqrt(w)
    X2_w = X2 * np.sqrt(w)
    return cdist(X1_w, X2_w, metric='euclidean')


# ============================================================
# HONEST LOO - STANDARD (uniform kernel) vs ADAPTIVE (player-weighted)
# ============================================================

def run_loo(X_all, y, pids, feat_names, bw_quantile=0.3, alpha=10.0,
            adaptive=False, n_pls=15):
    """
    Honest LOO per player.
    If adaptive=True: use player-specific correlation weights for distance.
    If adaptive=False: standard PLS + uniform Euclidean kernel.
    """
    n = len(y)
    oof_preds = np.zeros(n, dtype=np.float32)
    unique_pids = sorted(set(pids))

    for pid in unique_pids:
        pidx = np.where(pids == pid)[0]
        n_p = len(pidx)

        for i_local, i_global in enumerate(pidx):
            train_local = [j for j in range(n_p) if j != i_local]
            train_global = pidx[train_local]

            X_tr = X_all[train_global]
            X_te = X_all[i_global:i_global+1]
            y_tr = y[train_global]

            if adaptive:
                # --- ADAPTIVE: player-specific feature weights ---
                # Compute weights from training fold only (honest - no test leakage)
                weights = compute_player_weights(X_tr, y_tr, feat_names)

                # Scale features (needed for PLS / distance stability)
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)

                # Weighted distance using player-specific correlation weights
                D_tr = weighted_euclidean_distance(X_tr_s, X_tr_s, weights)
                bw = np.quantile(D_tr[D_tr > 0], bw_quantile)
                if bw < 1e-6:
                    bw = 1.0

                K_tr = np.exp(-0.5 * (D_tr / bw) ** 2)
                D_te = weighted_euclidean_distance(X_te_s, X_tr_s, weights)
                K_te = np.exp(-0.5 * (D_te / bw) ** 2)

            else:
                # --- STANDARD: PLS compression + uniform kernel ---
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)

                n_pls_p = min(n_pls, X_tr_s.shape[1] - 1, n_p - 2)
                pls = PLSRegression(n_components=n_pls_p, max_iter=500)
                pls.fit(X_tr_s, y_tr)
                X_tr_s = pls.transform(X_tr_s)
                X_te_s = pls.transform(X_te_s)

                sc2 = StandardScaler()
                X_tr_s = sc2.fit_transform(X_tr_s)
                X_te_s = sc2.transform(X_te_s)

                D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
                bw = np.quantile(D_tr[D_tr > 0], bw_quantile)
                if bw < 1e-6:
                    bw = 1.0
                K_tr = np.exp(-0.5 * (D_tr / bw) ** 2)
                D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
                K_te = np.exp(-0.5 * (D_te / bw) ** 2)

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(K_tr, y_tr)
            oof_preds[i_global] = float(ridge.predict(K_te)[0])

    mse = float(np.mean((oof_preds - y) ** 2))
    return oof_preds, mse


def make_test_preds(X_tr, y_tr, X_te, pids_tr, pids_te, feat_names,
                    bw_quantile=0.3, alpha=10.0, adaptive=False, n_pls=15):
    """Generate test predictions using full player training sets."""
    n_te = len(X_te)
    preds = np.zeros(n_te, dtype=np.float32)
    unique_pids = sorted(set(pids_tr))

    for pid in unique_pids:
        tr_idx = np.where(pids_tr == pid)[0]
        te_idx = np.where(pids_te == pid)[0]
        if len(te_idx) == 0:
            continue

        X_tr_p = X_tr[tr_idx]
        X_te_p = X_te[te_idx]
        y_tr_p = y_tr[tr_idx]

        if adaptive:
            weights = compute_player_weights(X_tr_p, y_tr_p, feat_names)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_p)
            X_te_s = sc.transform(X_te_p)
            D_tr = weighted_euclidean_distance(X_tr_s, X_tr_s, weights)
            bw = np.quantile(D_tr[D_tr > 0], bw_quantile)
            if bw < 1e-6:
                bw = 1.0
            K_tr = np.exp(-0.5 * (D_tr / bw) ** 2)
            D_te = weighted_euclidean_distance(X_te_s, X_tr_s, weights)
            K_te = np.exp(-0.5 * (D_te / bw) ** 2)
        else:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_p)
            X_te_s = sc.transform(X_te_p)
            n_pls_p = min(n_pls, X_tr_s.shape[1] - 1, len(tr_idx) - 1)
            pls = PLSRegression(n_components=n_pls_p, max_iter=500)
            pls.fit(X_tr_s, y_tr_p)
            X_tr_s = pls.transform(X_tr_s)
            X_te_s = pls.transform(X_te_s)
            sc2 = StandardScaler()
            X_tr_s = sc2.fit_transform(X_tr_s)
            X_te_s = sc2.transform(X_te_s)
            D_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
            bw = np.quantile(D_tr[D_tr > 0], bw_quantile)
            if bw < 1e-6:
                bw = 1.0
            K_tr = np.exp(-0.5 * (D_tr / bw) ** 2)
            D_te = cdist(X_te_s, X_tr_s, metric='euclidean')
            K_te = np.exp(-0.5 * (D_te / bw) ** 2)

        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(K_tr, y_tr_p)
        preds[te_idx] = ridge.predict(K_te).flatten()

    return preds


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    print("=== Player-Adaptive Metric Learning ===")
    print("Honest LOO reference: 0.006830")
    print()

    train, test = load_data()
    y_train = train['y']
    pids_train = train['pids']
    pids_test = test['pids']
    unique_pids = sorted(set(pids_train))

    loo_std = {}
    loo_ada = {}
    test_preds_std = {}
    test_preds_ada = {}

    for target in TARGETS:
        ti = TARGETS.index(target)
        tf = TARGET_FRAMES[target]
        y_t = y_train[:, ti]

        print(f"\n{'='*60}")
        print(f"TARGET: {target.upper()} (frame={tf})")
        print(f"{'='*60}")

        print("  Building features...")
        X_tr, feat_names = build_feature_matrix(train, tf)
        X_te, _ = build_feature_matrix(test, tf)
        n_core = 72  # position + velocity for 12 joints
        n_aug = X_tr.shape[1] - n_core
        print(f"  Total features: {X_tr.shape[1]} ({n_core} core + {n_aug} player-specific channels)")

        # Standard LOO
        print("  LOO standard (PLS + uniform kernel)...")
        t0 = time.time()
        oof_std, mse_std = run_loo(X_tr, y_t, pids_train, feat_names,
                                    adaptive=False, n_pls=15)
        print(f"    MSE = {mse_std:.6f}  ({time.time()-t0:.1f}s)")
        loo_std[target] = mse_std

        # Adaptive LOO
        print("  LOO adaptive (player-weighted metric)...")
        t0 = time.time()
        oof_ada, mse_ada = run_loo(X_tr, y_t, pids_train, feat_names,
                                    adaptive=True)
        delta = (mse_ada - mse_std) / mse_std * 100
        print(f"    MSE = {mse_ada:.6f}  ({time.time()-t0:.1f}s)  delta={delta:+.1f}% vs standard")
        loo_ada[target] = mse_ada

        # Per-player breakdown
        print(f"\n  Per-player breakdown (standard vs adaptive):")
        print(f"  {'Player':<10} {'Standard':>12} {'Adaptive':>12} {'Delta':>8}")
        for pid in unique_pids:
            mask = pids_train == pid
            m_std = float(np.mean((oof_std[mask] - y_t[mask]) ** 2))
            m_ada = float(np.mean((oof_ada[mask] - y_t[mask]) ** 2))
            d = (m_ada - m_std) / m_std * 100
            flag = " <-- IMPROVED" if d < -3 else (" <-- WORSE" if d > 3 else "")
            print(f"  P{pid:<9} {m_std:>12.6f} {m_ada:>12.6f} {d:>7.1f}%{flag}")

        # Show which features got high weight for each player (research insight)
        print(f"\n  Top-5 weighted features per player (adaptive metric):")
        for pid in unique_pids:
            pidx = np.where(pids_train == pid)[0]
            X_p = X_tr[pidx]
            y_p = y_t[pidx]
            weights = compute_player_weights(X_p, y_p, feat_names)
            top5 = np.argsort(weights)[-5:][::-1]
            print(f"  P{pid}: " + " | ".join(
                f"{feat_names[i]}(w={weights[i]:.3f})" for i in top5))

        # Test predictions
        test_preds_std[target] = make_test_preds(X_tr, y_t, X_te, pids_train, pids_test,
                                                   feat_names, adaptive=False, n_pls=15)
        test_preds_ada[target] = make_test_preds(X_tr, y_t, X_te, pids_train, pids_test,
                                                   feat_names, adaptive=True)

    # Overall summary
    mean_std = np.mean(list(loo_std.values()))
    mean_ada = np.mean(list(loo_ada.values()))
    delta_mean = (mean_ada - mean_std) / mean_std * 100

    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Standard (PLS+uniform):  angle={loo_std['angle']:.6f} "
          f"depth={loo_std['depth']:.6f} LR={loo_std['left_right']:.6f} "
          f"mean={mean_std:.6f}")
    print(f"  Adaptive (player-metric): angle={loo_ada['angle']:.6f} "
          f"depth={loo_ada['depth']:.6f} LR={loo_ada['left_right']:.6f} "
          f"mean={mean_ada:.6f}")
    print(f"  Delta: {delta_mean:+.1f}%")
    print(f"  Honest LOO reference: 0.006830")

    # Build submission matrices
    preds_std = np.column_stack([test_preds_std[t] for t in TARGETS])
    preds_ada = np.column_stack([test_preds_ada[t] for t in TARGETS])

    # Load current best for blending
    BEST_SUB = 3507
    best_df = pd.read_csv(SUBMISSION_DIR / f"submission_{BEST_SUB}.csv")
    col_map = {}
    for t in TARGETS:
        if t in best_df.columns:
            col_map[t] = t
        elif f"scaled_{t}" in best_df.columns:
            col_map[t] = f"scaled_{t}"
    best_vals = np.column_stack([best_df[col_map[t]].values for t in TARGETS])

    def save_sub(preds, desc):
        num = get_next_submission_number()
        df = pd.DataFrame({
            'id': test['ids'],
            'angle': np.clip(preds[:, 0], 0, 1),
            'depth': np.clip(preds[:, 1], 0, 1),
            'left_right': np.clip(preds[:, 2], 0, 1),
        })
        path = SUBMISSION_DIR / f"submission_{num}.csv"
        df.to_csv(path, index=False)
        print(f"  Sub {num}: {desc}")
        return num

    print(f"\n--- Creating Submissions ---")
    s1 = save_sub(preds_std, f"Standalone standard (CV={mean_std:.6f})")
    s2 = save_sub(preds_ada, f"Standalone adaptive (CV={mean_ada:.6f})")

    # Decide which model is better for blending
    best_standalone = preds_ada if mean_ada < mean_std else preds_std
    best_cv = min(mean_std, mean_ada)
    best_label = "adaptive" if mean_ada < mean_std else "standard"

    for w in [0.03, 0.05, 0.10]:
        blended = w * best_standalone + (1 - w) * best_vals
        save_sub(blended, f"{int(w*100)}% {best_label} + {int((1-w)*100)}% Sub{BEST_SUB}")

    # Save results
    total = time.time() - t_start
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {
        'loo_standard': loo_std, 'mean_std': mean_std,
        'loo_adaptive': loo_ada, 'mean_ada': mean_ada,
        'delta_pct': delta_mean,
        'honest_loo_reference': 0.006830,
        'total_time_s': total,
        'best_model': best_label,
    }
    out = OUTPUT_DIR / f"player_adaptive_kernel_{ts}.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    print(f"Total time: {total:.1f}s")
