"""
Angle Shrinkage Experiment

Root cause of 2.57x angle overfit: 72% of angle variance is between-player,
only 28% within-player. Within-player std is tiny (0.04-0.09) and features
have max r=0.23. The model memorizes noise.

Fix: shrink angle predictions toward the player mean.
  pred_final = shrink * model_pred + (1 - shrink) * player_mean

Also test: fewer PLS components for angle, feature subset for angle.
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


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


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


def extract_base_features(ts_3d, ts_hr, kp_index, frame):
    """208 features: 198 hoop-relative + 10 joint angles."""
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
    feats.append(frame)
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats.append(np.nanmean(series[140:180]))
            feats.append(np.nanmax(vel[140:180]))
    else:
        feats.extend([0.0] * 6)
    # Joint angles (10 features)
    rw_i, re_i, rs_i = kp_index.get('right_wrist'), kp_index.get('right_elbow'), kp_index.get('right_shoulder')
    rh_i, lh_i = kp_index.get('right_hip'), kp_index.get('left_hip')
    rk_i, lk_i = kp_index.get('right_knee'), kp_index.get('left_knee')
    ra_i, la_i = kp_index.get('right_ankle'), kp_index.get('left_ankle')
    neck_i, mh_i, ls_i = kp_index.get('neck'), kp_index.get('mid_hip'), kp_index.get('left_shoulder')
    if all(x is not None for x in [rs_i, rh_i, re_i]):
        feats.append(_angle_between(ts_3d[f, rs_i] - ts_3d[f, rh_i], ts_3d[f, re_i] - ts_3d[f, rs_i]))
    else:
        feats.append(90.0)
    if neck_i is not None and mh_i is not None:
        trunk = ts_hr[f, neck_i] - ts_hr[f, mh_i]
        feats.append(_angle_between(trunk, np.array([0, 0, 1], dtype=np.float32)))
        feats.append(np.degrees(np.arctan2(trunk[1], trunk[2] + 1e-8)))
    else:
        feats.extend([0.0, 0.0])
    if all(x is not None for x in [rk_i, rh_i, ra_i]):
        feats.append(_angle_between(ts_3d[f, rh_i] - ts_3d[f, rk_i], ts_3d[f, ra_i] - ts_3d[f, rk_i]))
    else:
        feats.append(90.0)
    if all(x is not None for x in [lk_i, lh_i, la_i]):
        feats.append(_angle_between(ts_3d[f, lh_i] - ts_3d[f, lk_i], ts_3d[f, la_i] - ts_3d[f, lk_i]))
    else:
        feats.append(90.0)
    if re_i is not None and rw_i is not None:
        feats.append(_angle_between(ts_3d[f, rw_i] - ts_3d[f, re_i], np.array([0, 0, 1], dtype=np.float32)))
    else:
        feats.append(90.0)
    if rs_i is not None and rw_i is not None:
        feats.append(_angle_between(ts_hr[f, rw_i] - ts_hr[f, rs_i], np.array([1, 0, 0.5], dtype=np.float32)))
    else:
        feats.append(90.0)
    if rs_i is not None and ls_i is not None:
        feats.append(_angle_between(ts_hr[f, rs_i] - ts_hr[f, ls_i], np.array([0, 1, 0], dtype=np.float32)))
    else:
        feats.append(90.0)
    if all(x is not None for x in [rh_i, lh_i, rs_i, ls_i]):
        hip_line = ts_hr[f, rh_i, :2] - ts_hr[f, lh_i, :2]
        shoulder_xy = ts_hr[f, rs_i, :2] - ts_hr[f, ls_i, :2]
        hn, sn = np.linalg.norm(hip_line), np.linalg.norm(shoulder_xy)
        if hn > 1e-6 and sn > 1e-6:
            feats.append(np.degrees(np.arccos(np.clip(np.dot(hip_line, shoulder_xy) / (hn * sn), -1, 1))))
        else:
            feats.append(0.0)
    else:
        feats.append(0.0)
    if re_i is not None and rs_i is not None:
        feats.append(ts_hr[f, re_i, 2] - ts_hr[f, rs_i, 2])
    else:
        feats.append(0.0)
    return np.array(feats, dtype=np.float32)


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
        ids, pids, targets = [], [], []
        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
        result = {'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result
    return process(train_df, True), process(test_df, False)


def loo_with_shrinkage(X_feat, y_sc, pids, bw_q=0.30, alpha=10.0,
                       shrink=1.0, n_pls=15, max_features=None):
    """
    LOO with optional:
    - shrinkage toward player mean
    - PLS component reduction
    - feature count reduction
    """
    unique_pids = sorted(np.unique(pids))
    n = len(y_sc)
    loo_preds = np.full(n, np.nan)

    for pid in unique_pids:
        mask = pids == pid
        idx = np.where(mask)[0]
        Xp = X_feat[mask].copy()
        yp = y_sc[mask]
        n_p = len(idx)
        player_mean = np.mean(yp)

        # Feature selection by correlation with target
        if max_features is not None and max_features < Xp.shape[1]:
            corrs = np.array([abs(np.corrcoef(Xp[:, fi], yp)[0, 1])
                              if np.std(Xp[:, fi]) > 1e-8 else 0.0
                              for fi in range(Xp.shape[1])])
            top_idx = np.argsort(corrs)[::-1][:max_features]
            Xp = Xp[:, top_idx]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xp)

        # PLS
        n_comp = min(n_pls, min(Xs.shape) - 1)
        if n_comp < 1:
            n_comp = 1
        pls = PLSRegression(n_components=n_comp)
        pls.fit(Xs, yp)
        Xpls = pls.transform(Xs)

        D = cdist(Xpls, Xpls, metric='euclidean')
        bw = np.quantile(D[D > 0], bw_q) if np.any(D > 0) else 1.0
        sigma = bw / np.sqrt(2.0)
        if sigma < 1e-12:
            sigma = 1.0

        for li in range(n_p):
            train_mask = np.ones(n_p, dtype=bool)
            train_mask[li] = False

            dists = D[li, train_mask]
            w = np.exp(-0.5 * (dists / sigma) ** 2)
            W = np.diag(np.sqrt(w))

            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(W @ Xpls[train_mask], W @ yp[train_mask])
            raw_pred = model.predict(Xpls[li:li+1])[0]

            # Shrink toward player mean
            final_pred = shrink * raw_pred + (1 - shrink) * player_mean
            loo_preds[idx[li]] = final_pred

    loo_mse = np.mean((loo_preds - y_sc) ** 2)
    return loo_mse, loo_preds


def main():
    t0 = time.time()
    print("=" * 70)
    print("ANGLE SHRINKAGE EXPERIMENT")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    n_train = len(pids_train)
    n_test = len(pids_test)

    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Extract features for angle at frame 153
    print("\nExtracting features...")
    X_train = {}
    X_test = {}
    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        X_train[target] = np.array([
            extract_base_features(train_data['X_3d'][i],
                                  compute_hoop_transform(train_data['X_3d'][i], kp_index),
                                  kp_index, frame)
            for i in range(n_train)
        ], dtype=np.float32)
        X_train[target] = np.nan_to_num(X_train[target], nan=0.0, posinf=0.0, neginf=0.0)

        X_test[target] = np.array([
            extract_base_features(test_data['X_3d'][i],
                                  compute_hoop_transform(test_data['X_3d'][i], kp_index),
                                  kp_index, frame)
            for i in range(n_test)
        ], dtype=np.float32)
        X_test[target] = np.nan_to_num(X_test[target], nan=0.0, posinf=0.0, neginf=0.0)

    # =========================================================================
    # EXPERIMENT 1: Shrinkage sweep for angle
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Shrinkage sweep (angle only)")
    print("pred = shrink * model + (1 - shrink) * player_mean")
    print("=" * 70)

    y_angle_raw = y_train[:, 0]
    y_angle_sc = scalers["angle"].transform(y_angle_raw.reshape(-1, 1)).ravel()

    # Baseline for comparison
    baseline_mse, _ = loo_with_shrinkage(
        X_train["angle"], y_angle_sc, pids_train,
        bw_q=0.30, alpha=10.0, shrink=1.0, n_pls=15)
    print(f"\n  Baseline (shrink=1.0): LOO MSE = {baseline_mse:.6f}")

    # Player mean only
    pm_mse = 0
    for pid in sorted(np.unique(pids_train)):
        mask = pids_train == pid
        ys = y_angle_sc[mask]
        pm = np.mean(ys)
        pm_mse += np.sum((ys - pm) ** 2)
    pm_mse /= n_train
    print(f"  Player mean only (shrink=0.0): LOO MSE = {pm_mse:.6f}")

    results = []
    for shrink in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        mse, _ = loo_with_shrinkage(
            X_train["angle"], y_angle_sc, pids_train,
            bw_q=0.30, alpha=10.0, shrink=shrink, n_pls=15)
        pct_change = (mse - baseline_mse) / baseline_mse * 100
        results.append((shrink, mse, pct_change))
        print(f"  shrink={shrink:.1f}: LOO MSE = {mse:.6f} ({pct_change:+.2f}%)")

    # =========================================================================
    # EXPERIMENT 2: PLS component sweep for angle
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PLS component sweep (angle only)")
    print("=" * 70)

    for n_pls in [2, 3, 5, 7, 10, 15, 20, 30]:
        mse, _ = loo_with_shrinkage(
            X_train["angle"], y_angle_sc, pids_train,
            bw_q=0.30, alpha=10.0, shrink=1.0, n_pls=n_pls)
        pct_change = (mse - baseline_mse) / baseline_mse * 100
        print(f"  n_pls={n_pls:2d}: LOO MSE = {mse:.6f} ({pct_change:+.2f}%)")

    # =========================================================================
    # EXPERIMENT 3: Feature count sweep for angle
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Feature count sweep (angle only)")
    print("=" * 70)

    for max_feat in [10, 20, 30, 50, 75, 100, 150, 208]:
        mse, _ = loo_with_shrinkage(
            X_train["angle"], y_angle_sc, pids_train,
            bw_q=0.30, alpha=10.0, shrink=1.0, n_pls=15,
            max_features=max_feat)
        pct_change = (mse - baseline_mse) / baseline_mse * 100
        print(f"  max_feat={max_feat:3d}: LOO MSE = {mse:.6f} ({pct_change:+.2f}%)")

    # =========================================================================
    # EXPERIMENT 4: Alpha sweep for angle
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Ridge alpha sweep (angle only)")
    print("=" * 70)

    for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
        mse, _ = loo_with_shrinkage(
            X_train["angle"], y_angle_sc, pids_train,
            bw_q=0.30, alpha=alpha, shrink=1.0, n_pls=15)
        pct_change = (mse - baseline_mse) / baseline_mse * 100
        print(f"  alpha={alpha:7.1f}: LOO MSE = {mse:.6f} ({pct_change:+.2f}%)")

    # =========================================================================
    # EXPERIMENT 5: Combined best settings
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Combined settings grid (angle only)")
    print("=" * 70)

    best_combo_mse = baseline_mse
    best_combo = None
    combos = []
    for shrink in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for n_pls in [5, 7, 10, 15]:
            for alpha in [10, 50, 100, 500]:
                mse, _ = loo_with_shrinkage(
                    X_train["angle"], y_angle_sc, pids_train,
                    bw_q=0.30, alpha=alpha, shrink=shrink, n_pls=n_pls)
                combos.append((shrink, n_pls, alpha, mse))

    # Sort by MSE and show top 10
    combos.sort(key=lambda x: x[3])
    print("\n  Top 10 combinations (sorted by LOO MSE):")
    for i, (sh, np_, al, mse) in enumerate(combos[:10]):
        pct = (mse - baseline_mse) / baseline_mse * 100
        print(f"    #{i+1}: shrink={sh:.1f}, n_pls={np_}, alpha={al}, "
              f"LOO MSE={mse:.6f} ({pct:+.2f}%)")

    print("\n  Worst 5 combinations:")
    for i, (sh, np_, al, mse) in enumerate(combos[-5:]):
        pct = (mse - baseline_mse) / baseline_mse * 100
        print(f"    shrink={sh:.1f}, n_pls={np_}, alpha={al}, "
              f"LOO MSE={mse:.6f} ({pct:+.2f}%)")

    # =========================================================================
    # EXPERIMENT 6: Shrinkage applied to submission blend
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Generate angle-shrinkage submissions")
    print("Shrink only the angle predictions, keep depth/LR from Sub 2503")
    print("=" * 70)

    # Read Sub 2503 (current best)
    sub_2503 = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")

    # For generating shrunk angle predictions, we need a standalone pipeline
    # that can produce test predictions with shrinkage
    # Use the baseline pipeline (bw=0.30, alpha=10, 15 PLS) with shrinkage

    target = "angle"
    frame = TARGET_FRAMES[target]
    y_raw = y_train[:, 0]
    y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()
    unique_pids = sorted(np.unique(pids_train))

    for shrink_val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        test_preds = np.full(n_test, np.nan)
        loo_preds = np.full(n_train, np.nan)

        for pid in unique_pids:
            train_mask = pids_train == pid
            test_mask = pids_test == pid
            if not np.any(test_mask):
                continue

            Xp = X_train[target][train_mask]
            yp = y_sc[train_mask]
            n_p = Xp.shape[0]
            player_mean = np.mean(yp)

            Xp_test = X_test[target][test_mask]

            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xp)
            Xs_test = scaler.transform(Xp_test)

            n_comp = min(15, min(Xs.shape) - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(Xs, yp)
            Xpls = pls.transform(Xs)
            Xpls_test = pls.transform(Xs_test)

            D_train = cdist(Xpls, Xpls, metric='euclidean')
            bw = np.quantile(D_train[D_train > 0], 0.30) if np.any(D_train > 0) else 1.0
            sigma = bw / np.sqrt(2.0)
            if sigma < 1e-12:
                sigma = 1.0

            # LOO
            train_idx = np.where(train_mask)[0]
            for li in range(n_p):
                tmask = np.ones(n_p, dtype=bool)
                tmask[li] = False
                dists = D_train[li, tmask]
                w = np.exp(-0.5 * (dists / sigma) ** 2)
                W = np.diag(np.sqrt(w))
                model = Ridge(alpha=10.0, fit_intercept=True)
                model.fit(W @ Xpls[tmask], W @ yp[tmask])
                raw_pred = model.predict(Xpls[li:li+1])[0]
                loo_preds[train_idx[li]] = shrink_val * raw_pred + (1 - shrink_val) * player_mean

            # Test predictions
            D_test = cdist(Xpls_test, Xpls, metric='euclidean')
            test_idx = np.where(test_mask)[0]
            for ti in range(len(test_idx)):
                dists = D_test[ti]
                w = np.exp(-0.5 * (dists / sigma) ** 2)
                W = np.diag(np.sqrt(w))
                model = Ridge(alpha=10.0, fit_intercept=True)
                model.fit(W @ Xpls, W @ yp)
                raw_pred = model.predict(Xpls_test[ti:ti+1])[0]
                test_preds[test_idx[ti]] = shrink_val * raw_pred + (1 - shrink_val) * player_mean

        loo_mse = np.mean((loo_preds - y_sc) ** 2)
        pct = (loo_mse - baseline_mse) / baseline_mse * 100

        # Generate blended submission: replace angle with shrunk angle
        # Use 10% blend weight with Sub 2503
        sub_new = sub_2503.copy()
        for blend_w in [0.10, 0.20, 0.30]:
            sub_blend = sub_2503.copy()
            sub_blend['scaled_angle'] = blend_w * test_preds + (1 - blend_w) * sub_2503['scaled_angle'].values
            sub_blend['scaled_angle'] = sub_blend['scaled_angle'].clip(0, 1)

            sub_num = get_next_submission_number()
            sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            sub_blend.to_csv(sub_path, index=False)
            print(f"  Sub {sub_num}: shrink={shrink_val:.1f}, blend={blend_w:.0%} angle-only into Sub 2503, "
                  f"angle LOO MSE={loo_mse:.6f} ({pct:+.2f}%)")

    # =========================================================================
    # EXPERIMENT 7: Compare depth/LR LOO with same shrinkage
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Shrinkage for depth and LR (for comparison)")
    print("=" * 70)

    for target in ["depth", "left_right"]:
        tidx = {"depth": 1, "left_right": 2}[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        base_mse, _ = loo_with_shrinkage(
            X_train[target], y_sc, pids_train,
            bw_q=0.30, alpha=10.0, shrink=1.0, n_pls=15)

        print(f"\n  {target} baseline: LOO MSE = {base_mse:.6f}")
        for shrink in [0.5, 0.7, 0.9, 1.0]:
            mse, _ = loo_with_shrinkage(
                X_train[target], y_sc, pids_train,
                bw_q=0.30, alpha=10.0, shrink=shrink, n_pls=15)
            pct = (mse - base_mse) / base_mse * 100
            print(f"  shrink={shrink:.1f}: LOO MSE = {mse:.6f} ({pct:+.2f}%)")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
