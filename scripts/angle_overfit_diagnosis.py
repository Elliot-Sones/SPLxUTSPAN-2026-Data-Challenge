"""
Angle Overfit Diagnosis

The angle target overfits 2.57x (LOO 0.002511 vs test 0.006454) while
depth (1.51x) and LR (1.62x) are much more moderate.

This script investigates WHY by examining:
1. Per-player angle LOO error distribution
2. Hardest/easiest examples - what makes them hard?
3. Leave-one-out residual patterns
4. Feature sensitivity analysis (which features drive angle predictions most?)
5. Prediction confidence vs accuracy
6. Effective number of neighbors per example
7. Target distribution analysis
8. LOO prediction variance vs actual target variance
"""

import json
import time
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
import joblib

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}


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
    """198 hoop-relative + 10 joint angle features."""
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
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)
    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}
    n = len(train_df)
    n_kp = len(kp_names)
    X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
    ids, pids, targets = [], [], []
    for idx, row in train_df.iterrows():
        for col_i, col in enumerate(keypoint_cols):
            arr = parse_array_string(row[col])
            X_3d[idx, :, col_i // 3, col_i % 3] = arr
        ids.append(row['id'])
        pids.append(row['participant_id'])
        targets.append([row['angle'], row['depth'], row['left_right']])
    return {
        'X_3d': X_3d, 'pids': np.array(pids), 'ids': np.array(ids),
        'y': np.array(targets, dtype=np.float32),
        'kp_names': kp_names, 'kp_index': kp_index
    }


def run_loo_with_diagnostics(X_feat, y_sc, pids, bw_q=0.30, alpha=10.0):
    """
    Run LOO and return detailed per-example diagnostics.
    Returns: loo_preds, loo_errors, effective_n_neighbors, prediction_variance
    """
    unique_pids = sorted(np.unique(pids))
    n = len(y_sc)
    loo_preds = np.full(n, np.nan)
    eff_neighbors = np.full(n, np.nan)
    pred_variance = np.full(n, np.nan)
    coef_norms = np.full(n, np.nan)
    leverage = np.full(n, np.nan)

    for pid in unique_pids:
        mask = pids == pid
        idx = np.where(mask)[0]
        Xp = X_feat[mask]
        yp = y_sc[mask]
        n_p = len(idx)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xp)

        D = cdist(Xs, Xs, metric='euclidean')
        bw = np.quantile(D[D > 0], bw_q) if np.any(D > 0) else 1.0
        sigma = bw / np.sqrt(2.0)
        if sigma < 1e-12:
            sigma = 1.0

        for li in range(n_p):
            train_mask = np.ones(n_p, dtype=bool)
            train_mask[li] = False
            X_tr = Xs[train_mask]
            y_tr = yp[train_mask]
            x_te = Xs[li:li+1]

            dists = D[li, train_mask]
            w = np.exp(-0.5 * (dists / sigma) ** 2)
            eff_n = (np.sum(w) ** 2) / (np.sum(w ** 2) + 1e-12)
            eff_neighbors[idx[li]] = eff_n

            W = np.diag(np.sqrt(w))
            X_w = W @ X_tr
            y_w = W @ y_tr

            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_w, y_w)
            pred = model.predict(x_te)[0]
            loo_preds[idx[li]] = pred
            coef_norms[idx[li]] = np.linalg.norm(model.coef_)

            # Weighted prediction variance: how much does the prediction
            # change if we perturb neighbors?
            # Use: var(y_weighted) as a proxy for prediction uncertainty
            weighted_mean = np.average(y_tr, weights=w)
            weighted_var = np.average((y_tr - weighted_mean) ** 2, weights=w)
            pred_variance[idx[li]] = weighted_var

            # Leverage: how far is this example from the weighted centroid?
            weighted_center = np.average(X_tr, weights=w, axis=0)
            leverage[idx[li]] = np.linalg.norm(x_te[0] - weighted_center)

    loo_errors = (loo_preds - y_sc) ** 2
    return {
        'preds': loo_preds,
        'errors': loo_errors,
        'eff_neighbors': eff_neighbors,
        'pred_variance': pred_variance,
        'coef_norms': coef_norms,
        'leverage': leverage,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("ANGLE OVERFIT DIAGNOSIS")
    print("Why does angle overfit 2.57x while depth=1.51x and LR=1.62x?")
    print("=" * 70)

    data = load_data()
    y_train = data['y']
    kp_index = data['kp_index']
    pids = data['pids']
    n = len(pids)
    unique_pids = sorted(np.unique(pids))

    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # =========================================================================
    # ANALYSIS 1: Target distribution per player
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Target distributions per player")
    print("=" * 70)

    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()
        print(f"\n  {target} (scaled):")
        print(f"    Overall: mean={np.mean(y_sc):.4f}, std={np.std(y_sc):.4f}, "
              f"range=[{np.min(y_sc):.4f}, {np.max(y_sc):.4f}]")
        for pid in unique_pids:
            mask = pids == pid
            ys = y_sc[mask]
            print(f"    P{pid} (n={mask.sum()}): mean={np.mean(ys):.4f}, std={np.std(ys):.4f}, "
                  f"range=[{np.min(ys):.4f}, {np.max(ys):.4f}], "
                  f"CV={np.std(ys)/max(abs(np.mean(ys)), 1e-6):.3f}")

    # =========================================================================
    # ANALYSIS 2: Extract features and run LOO with diagnostics for each target
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: LOO diagnostics per target")
    print("=" * 70)

    # Extract features at each target frame
    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        print(f"\n  Extracting features for {target} at frame {frame}...")
        X_feat = np.array([
            extract_base_features(data['X_3d'][i],
                                  compute_hoop_transform(data['X_3d'][i], kp_index),
                                  kp_index, frame)
            for i in range(n)
        ], dtype=np.float32)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

        diag = run_loo_with_diagnostics(X_feat, y_sc, pids)

        loo_mse = np.mean(diag['errors'])
        print(f"\n  {target} LOO MSE: {loo_mse:.6f}")

        # Per-player breakdown
        print(f"  Per-player LOO MSE:")
        for pid in unique_pids:
            mask = pids == pid
            player_mse = np.mean(diag['errors'][mask])
            player_eff_n = np.mean(diag['eff_neighbors'][mask])
            player_var = np.mean(diag['pred_variance'][mask])
            player_coef = np.mean(diag['coef_norms'][mask])
            player_lev = np.mean(diag['leverage'][mask])
            print(f"    P{pid} (n={mask.sum()}): MSE={player_mse:.6f}, "
                  f"eff_n={player_eff_n:.1f}, pred_var={player_var:.6f}, "
                  f"coef_norm={player_coef:.2f}, leverage={player_lev:.3f}")

        # Worst examples
        print(f"\n  Top 10 worst LOO errors for {target}:")
        worst_idx = np.argsort(diag['errors'])[::-1][:10]
        for rank, wi in enumerate(worst_idx):
            print(f"    #{rank+1}: idx={wi}, P{pids[wi]}, "
                  f"true={y_sc[wi]:.4f}, pred={diag['preds'][wi]:.4f}, "
                  f"err={diag['errors'][wi]:.6f}, "
                  f"eff_n={diag['eff_neighbors'][wi]:.1f}, "
                  f"leverage={diag['leverage'][wi]:.3f}")

        # Error distribution stats
        print(f"\n  Error distribution for {target}:")
        errs = diag['errors']
        pcts = [50, 75, 90, 95, 99]
        for pct in pcts:
            print(f"    {pct}th percentile: {np.percentile(errs, pct):.6f}")
        print(f"    Max: {np.max(errs):.6f}")
        print(f"    Fraction of total MSE from top 10%: "
              f"{np.sum(np.sort(errs)[-n//10:])/np.sum(errs)*100:.1f}%")
        print(f"    Fraction of total MSE from top 20%: "
              f"{np.sum(np.sort(errs)[-n//5:])/np.sum(errs)*100:.1f}%")

    # =========================================================================
    # ANALYSIS 3: Angle target signal-to-noise ratio
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Signal-to-noise ratio comparison")
    print("=" * 70)

    for target in TARGETS:
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        # Within-player variance vs between-player variance
        within_var = 0
        between_var = 0
        overall_mean = np.mean(y_sc)
        for pid in unique_pids:
            mask = pids == pid
            ys = y_sc[mask]
            within_var += np.sum((ys - np.mean(ys)) ** 2)
            between_var += mask.sum() * (np.mean(ys) - overall_mean) ** 2
        total_var = np.sum((y_sc - overall_mean) ** 2)
        ICC = between_var / total_var  # intraclass correlation
        print(f"\n  {target}:")
        print(f"    Total variance: {total_var/n:.6f}")
        print(f"    Between-player fraction: {ICC:.3f} ({ICC*100:.1f}%)")
        print(f"    Within-player fraction: {1-ICC:.3f} ({(1-ICC)*100:.1f}%)")
        print(f"    Within-player std: {np.sqrt(within_var/n):.4f}")

    # =========================================================================
    # ANALYSIS 4: Feature predictive power per target (simple correlations)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Feature predictive power per target")
    print("=" * 70)

    frame_153 = TARGET_FRAMES["angle"]
    X_feat_153 = np.array([
        extract_base_features(data['X_3d'][i],
                              compute_hoop_transform(data['X_3d'][i], kp_index),
                              kp_index, frame_153)
        for i in range(n)
    ], dtype=np.float32)
    X_feat_153 = np.nan_to_num(X_feat_153, nan=0.0, posinf=0.0, neginf=0.0)

    for target in TARGETS:
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        # Per-player detrended correlations (within-player)
        all_corrs = []
        for pid in unique_pids:
            mask = pids == pid
            ys = y_sc[mask]
            Xs = X_feat_153[mask]
            ys_dm = ys - np.mean(ys)
            for fi in range(Xs.shape[1]):
                xs_dm = Xs[:, fi] - np.mean(Xs[:, fi])
                denom = np.std(xs_dm) * np.std(ys_dm)
                if denom > 1e-8:
                    all_corrs.append(abs(np.corrcoef(xs_dm, ys_dm)[0, 1]))
                else:
                    all_corrs.append(0.0)
        corrs = np.array(all_corrs).reshape(len(unique_pids), -1)
        mean_corrs = np.mean(corrs, axis=0)
        top_feat_idx = np.argsort(mean_corrs)[::-1][:10]
        print(f"\n  {target}: top 10 features by mean within-player |r|:")
        for fi in top_feat_idx:
            print(f"    feat[{fi}]: mean|r|={mean_corrs[fi]:.4f}, "
                  f"per-player={[f'{corrs[pi, fi]:.3f}' for pi in range(len(unique_pids))]}")

    # =========================================================================
    # ANALYSIS 5: Angle LOO error vs frame choice sensitivity
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Frame sensitivity for angle")
    print("=" * 70)

    tidx = 0  # angle
    y_raw = y_train[:, tidx]
    y_sc = scalers["angle"].transform(y_raw.reshape(-1, 1)).ravel()

    for frame in [140, 145, 150, 153, 155, 160, 165, 170]:
        X_feat = np.array([
            extract_base_features(data['X_3d'][i],
                                  compute_hoop_transform(data['X_3d'][i], kp_index),
                                  kp_index, frame)
            for i in range(n)
        ], dtype=np.float32)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # Quick LOO - just compute MSE
        loo_errs = []
        for pid in unique_pids:
            mask = pids == pid
            idx = np.where(mask)[0]
            Xp = X_feat[mask]
            yp = y_sc[mask]
            n_p = len(idx)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xp)
            D = cdist(Xs, Xs, metric='euclidean')
            bw = np.quantile(D[D > 0], 0.30) if np.any(D > 0) else 1.0
            sigma = bw / np.sqrt(2.0)
            if sigma < 1e-12:
                sigma = 1.0
            for li in range(n_p):
                train_mask = np.ones(n_p, dtype=bool)
                train_mask[li] = False
                dists = D[li, train_mask]
                w = np.exp(-0.5 * (dists / sigma) ** 2)
                W = np.diag(np.sqrt(w))
                model = Ridge(alpha=10.0, fit_intercept=True)
                model.fit(W @ Xs[train_mask], W @ yp[train_mask])
                pred = model.predict(Xs[li:li+1])[0]
                loo_errs.append((pred - yp[li]) ** 2)
        loo_mse = np.mean(loo_errs)
        print(f"  Frame {frame}: angle LOO MSE = {loo_mse:.6f}")

    # =========================================================================
    # ANALYSIS 6: Angle target unique characteristics
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Angle target nature")
    print("=" * 70)

    for target in TARGETS:
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()
        print(f"\n  {target}:")
        print(f"    Scaled range: [{np.min(y_sc):.4f}, {np.max(y_sc):.4f}]")
        print(f"    Raw range: [{np.min(y_raw):.4f}, {np.max(y_raw):.4f}]")
        print(f"    Unique values: {len(np.unique(y_raw))}")
        print(f"    Skewness: {pd.Series(y_sc).skew():.3f}")
        print(f"    Kurtosis: {pd.Series(y_sc).kurtosis():.3f}")

        # Check if angle is more "noisy" - adjacent-shot variability
        for pid in unique_pids:
            mask = pids == pid
            ys = y_sc[mask]
            # First-difference std (local noise)
            diff_std = np.std(np.diff(ys))
            # Overall std
            overall_std = np.std(ys)
            print(f"    P{pid}: overall_std={overall_std:.4f}, diff_std={diff_std:.4f}, "
                  f"ratio={diff_std/max(overall_std, 1e-8):.3f}")

    # =========================================================================
    # ANALYSIS 7: Cross-target correlation
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Cross-target correlations")
    print("=" * 70)

    for pid in unique_pids:
        mask = pids == pid
        ya = scalers["angle"].transform(y_train[mask, 0:1]).ravel()
        yd = scalers["depth"].transform(y_train[mask, 1:2]).ravel()
        yl = scalers["left_right"].transform(y_train[mask, 2:3]).ravel()
        print(f"  P{pid}:")
        print(f"    angle-depth: r={np.corrcoef(ya, yd)[0,1]:.3f}")
        print(f"    angle-LR: r={np.corrcoef(ya, yl)[0,1]:.3f}")
        print(f"    depth-LR: r={np.corrcoef(yd, yl)[0,1]:.3f}")

    # =========================================================================
    # ANALYSIS 8: PLS compression quality per target
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 8: PLS compression quality per target")
    print("=" * 70)

    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        tidx = target_idx[target]
        y_raw = y_train[:, tidx]
        y_sc = scalers[target].transform(y_raw.reshape(-1, 1)).ravel()

        X_feat = np.array([
            extract_base_features(data['X_3d'][i],
                                  compute_hoop_transform(data['X_3d'][i], kp_index),
                                  kp_index, frame)
            for i in range(n)
        ], dtype=np.float32)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

        for pid in unique_pids:
            mask = pids == pid
            Xp = X_feat[mask]
            yp = y_sc[mask]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xp)
            n_comp = min(15, min(Xp.shape) - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(Xs, yp)
            Xpls = pls.transform(Xs)
            # How much variance in y is captured by PLS?
            from sklearn.metrics import r2_score
            pls_pred = pls.predict(Xs).ravel()
            r2 = r2_score(yp, pls_pred)
            # Explained variance in X
            Xrec = pls.inverse_transform(Xpls)
            x_r2 = 1 - np.sum((Xs - Xrec) ** 2) / np.sum(Xs ** 2)
            if pid == unique_pids[0]:
                print(f"\n  {target} (frame {frame}):")
            print(f"    P{pid}: PLS R2(y)={r2:.4f}, R2(X)={x_r2:.4f}, "
                  f"y_std={np.std(yp):.4f}, n_comp={n_comp}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Diagnosis complete in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
