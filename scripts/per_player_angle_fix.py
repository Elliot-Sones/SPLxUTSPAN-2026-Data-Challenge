"""
Per-Player Angle Optimization - Optimize hyperparameters AND frame per player.

Key findings from P4 angle fix:
- P4 angle improved 19.3% with bw=0.15, alpha=50, frame=160
- P2 angle improved 12.8% with bw=0.10, alpha=50
- Every player benefits from per-player optimization

This script:
1. Sweeps bw, alpha, PLS nc, and frame per player for angle
2. Generates test predictions using optimal per-player settings
3. Splices ALL improved angle predictions into Sub 3411
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
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048


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
    vel = np.zeros_like(ball * FEET_TO_METERS)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_METERS, 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)
    s, e = max(80, wrist_peak - 40), min(wrist_peak + 5, 200)
    return int(np.clip(s + np.argmax(speed[s:e]), 80, 200))


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract features at a specific frame. Same as p4_angle_fix_v2."""
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
            feats.append(float(np.nanmean(series)))
            feats.append(float(np.nanstd(series)))
            feats.append(float(np.nanmax(series) - np.nanmin(series)))

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
    ls_i = kp_index.get('left_shoulder')
    if rh is not None and lh is not None:
        feats.append(ts_hr[f, rh, 1] - ts_hr[f, lh, 1])
        feats.append(ts_hr[f, rh, 0] - ts_hr[f, lh, 0])
    else:
        feats.extend([0.0, 0.0])
    if rs is not None and ls_i is not None:
        feats.append(ts_hr[f, rs, 1] - ts_hr[f, ls_i, 1])
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
            feats.append(float(np.nanmean(series[140:180])))
            feats.append(float(np.nanmax(vel[140:180])))
    else:
        feats.extend([0.0] * 6)

    # Joint angles (10)
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

    # Flick window features (15)
    flick_s, flick_e = 135, 165
    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            acc = np.gradient(vel, DT)
            feats.append(float(np.nanmean(series[flick_s:flick_e])))
            feats.append(float(np.nanstd(series[flick_s:flick_e])))
            feats.append(float(np.nanmean(vel[flick_s:flick_e])))
            feats.append(float(np.nanmax(vel[flick_s:flick_e])))
            feats.append(float(np.nanmean(acc[flick_s:flick_e])))
    else:
        feats.extend([0.0] * 15)

    # Fingertip spread (3)
    finger_tips = ['right_second_finger_distal', 'right_third_finger_distal',
                   'right_fourth_finger_distal']
    tip_pos = []
    for fname in finger_tips:
        fi = kp_index.get(fname)
        tip_pos.append(ts_hr[f, fi, :] if fi is not None else np.zeros(3))
    tip_pos = np.array(tip_pos)
    for i in range(len(tip_pos)):
        for j in range(i+1, len(tip_pos)):
            feats.append(float(np.linalg.norm(tip_pos[i] - tip_pos[j])))

    # Body dynamics (3)
    if mh_i is not None:
        feats.append(float(ts_hr[f, mh_i, 2]))
        feats.append(float(ts_hr[f, mh_i, 2] - ts_hr[60, mh_i, 2]))
        vel_z = np.gradient(ts_hr[:, mh_i, 2], DT)
        feats.append(float(vel_z[f]))
    else:
        feats.extend([0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def main():
    t0 = time.time()
    print("=" * 70)
    print("PER-PLAYER ANGLE OPTIMIZATION")
    print("=" * 70)

    train_data, test_data = load_data()
    y_train = train_data['y']
    kp_index = train_data['kp_index']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    scaler_angle = joblib.load(DATA_DIR / "scaler_angle.pkl")
    y_angle_raw = y_train[:, 0]
    y_angle_sc = scaler_angle.transform(y_angle_raw.reshape(-1, 1)).ravel()

    # Extract features at multiple frames
    FRAMES = [140, 145, 148, 150, 153, 155, 158, 160, 165, 170]

    print("\nExtracting features...")
    feat_cache = {}
    for frame in FRAMES:
        tr_feats, te_feats = [], []
        for i in range(len(pids_train)):
            ts_3d = train_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            tr_feats.append(extract_features(ts_3d, ts_hr, kp_index, rf, frame))
        for i in range(len(pids_test)):
            ts_3d = test_data['X_3d'][i]
            ts_hr = compute_hoop_transform(ts_3d, kp_index)
            rf = detect_release_frame(ts_3d, kp_index)
            te_feats.append(extract_features(ts_3d, ts_hr, kp_index, rf, frame))
        X_tr = np.nan_to_num(np.array(tr_feats), nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(np.array(te_feats), nan=0.0, posinf=0.0, neginf=0.0)
        feat_cache[frame] = {'train': X_tr, 'test': X_te}
    print(f"  Features per frame: {feat_cache[150]['train'].shape[1]}")
    print(f"  Extraction time: {time.time()-t0:.1f}s")

    X_raw_train = train_data['X_raw']
    X_raw_test = test_data['X_raw']

    # ================================================================
    # SWEEP PER PLAYER
    # ================================================================
    player_configs = {}
    player_baselines = {}
    oof_all_default = np.zeros(len(pids_train))
    oof_all_optimized = np.zeros(len(pids_train))
    test_all_optimized = np.zeros(len(pids_test))

    for pid in sorted(np.unique(pids_train)):
        t_p = time.time()
        p_mask_tr = pids_train == pid
        p_mask_te = pids_test == pid
        p_tr_idx = np.where(p_mask_tr)[0]
        p_te_idx = np.where(p_mask_te)[0]
        n_tr = len(p_tr_idx)
        n_te = len(p_te_idx)

        y_p_sc = y_angle_sc[p_mask_tr]
        y_p_raw = y_angle_raw[p_mask_tr]
        X_raw_p = X_raw_train[p_mask_tr]
        X_raw_p_te = X_raw_test[p_mask_te]

        print(f"\n{'='*60}")
        print(f"Player {pid}: {n_tr} train, {n_te} test samples")
        print(f"{'='*60}")

        # Baseline: default settings
        frame_default = 153  # default angle frame
        X_p_default = feat_cache[frame_default]['train'][p_mask_tr]

        def run_cv(X_p, bw, alpha, pls_nc, seed=42):
            n = len(X_p)
            oof = np.zeros(n)
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            for tr_idx, val_idx in kf.split(X_p):
                pls_sc = StandardScaler()
                raw_tr = pls_sc.fit_transform(X_raw_p[tr_idx])
                nc = min(pls_nc, len(tr_idx) - 1)
                nc = max(2, nc)
                pls = PLSRegression(n_components=nc)
                pls.fit(raw_tr, y_p_raw[tr_idx])
                pls_tr = pls.transform(raw_tr)
                pls_val = pls.transform(pls_sc.transform(X_raw_p[val_idx]))

                X_tr_aug = np.hstack([X_p[tr_idx], pls_tr])
                X_val_aug = np.hstack([X_p[val_idx], pls_val])

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr_aug)
                X_val_s = sc.transform(X_val_aug)

                D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
                all_d = D_tr[np.triu_indices(len(X_tr_s), k=1)]
                sigma = np.quantile(all_d, bw) if len(all_d) > 0 else 1.0
                sigma = max(sigma, 1e-6)

                D_val = cdist(X_val_s, X_tr_s, 'euclidean')
                for j, vi in enumerate(val_idx):
                    w = np.exp(-D_val[j]**2 / (2*sigma**2))
                    if w.sum() < 1e-10:
                        oof[vi] = np.mean(y_p_sc[tr_idx])
                    else:
                        r = Ridge(alpha=alpha)
                        r.fit(X_tr_s, y_p_sc[tr_idx], sample_weight=w)
                        oof[vi] = r.predict(X_val_s[j:j+1])[0]
            return oof

        oof_default = run_cv(X_p_default, 0.30, 10.0, 10)
        mse_default = np.mean((oof_default - y_p_sc)**2)
        player_baselines[pid] = mse_default
        oof_all_default[p_tr_idx] = oof_default
        print(f"  Default (bw=0.30, a=10, nc=10, f=153): CV = {mse_default:.6f}")

        # Phase 1: Sweep bw, alpha, pls_nc at default frame
        best_mse = mse_default
        best_cfg = {'bw': 0.30, 'alpha': 10.0, 'pls_nc': 10}

        bw_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]
        alpha_values = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        nc_values = [3, 5, 8, 10]

        for bw in bw_values:
            for alpha in alpha_values:
                for pls_nc in nc_values:
                    oof = run_cv(X_p_default, bw, alpha, pls_nc)
                    mse = np.mean((oof - y_p_sc)**2)
                    if mse < best_mse:
                        best_mse = mse
                        best_cfg = {'bw': bw, 'alpha': alpha, 'pls_nc': pls_nc}

        print(f"  Best hyper at f=153: {best_cfg} -> CV = {best_mse:.6f} ({(1-best_mse/mse_default)*100:+.2f}%)")

        # Phase 2: Frame sweep with best hyperparams
        frame_results = {}
        for frame in FRAMES:
            X_p_f = feat_cache[frame]['train'][p_mask_tr]
            oof = run_cv(X_p_f, best_cfg['bw'], best_cfg['alpha'], best_cfg['pls_nc'])
            mse = np.mean((oof - y_p_sc)**2)
            frame_results[frame] = mse

        best_frame = min(frame_results, key=frame_results.get)
        best_frame_mse = frame_results[best_frame]
        print(f"  Frame sweep: best={best_frame} -> CV = {best_frame_mse:.6f} ({(1-best_frame_mse/mse_default)*100:+.2f}%)")

        for frame in sorted(frame_results.keys()):
            d = (1-frame_results[frame]/mse_default)*100
            marker = " <-- BEST" if frame == best_frame else ""
            print(f"    f={frame}: {frame_results[frame]:.6f} ({d:+.2f}%){marker}")

        # Phase 3: Multi-frame with top 2-3 frames
        sorted_f = sorted(frame_results.items(), key=lambda x: x[1])
        multi_results = {}
        for n_f in [1, 2, 3]:
            combo = [f for f, _ in sorted_f[:n_f]]
            oofs = [run_cv(feat_cache[f]['train'][p_mask_tr], best_cfg['bw'], best_cfg['alpha'], best_cfg['pls_nc']) for f in combo]
            avg_oof = np.mean(oofs, axis=0)
            mse = np.mean((avg_oof - y_p_sc)**2)
            multi_results[n_f] = (mse, combo)
            print(f"  Top-{n_f} frames {combo}: {mse:.6f} ({(1-mse/mse_default)*100:+.2f}%)")

        # Pick best overall
        best_multi = min(multi_results, key=lambda x: multi_results[x][0])
        best_multi_mse, best_multi_frames = multi_results[best_multi]

        if best_multi_mse < best_frame_mse:
            final_mse = best_multi_mse
            final_frames = best_multi_frames
        else:
            final_mse = best_frame_mse
            final_frames = [best_frame]

        final_cfg = {**best_cfg, 'frames': final_frames}
        player_configs[pid] = final_cfg
        print(f"  FINAL P{pid}: frames={final_frames}, bw={best_cfg['bw']}, alpha={best_cfg['alpha']}, nc={best_cfg['pls_nc']}")
        print(f"  FINAL CV: {final_mse:.6f} ({(1-final_mse/mse_default)*100:+.2f}%)")

        # Generate OOF and test predictions with final config
        oofs_final = []
        tests_final = []
        for frame in final_frames:
            X_p_f_tr = feat_cache[frame]['train'][p_mask_tr]
            X_p_f_te = feat_cache[frame]['test'][p_mask_te]
            oof = run_cv(X_p_f_tr, best_cfg['bw'], best_cfg['alpha'], best_cfg['pls_nc'])
            oofs_final.append(oof)

            # Test predictions
            pls_sc = StandardScaler()
            raw_tr = pls_sc.fit_transform(X_raw_p)
            nc = min(best_cfg['pls_nc'], n_tr - 1)
            nc = max(2, nc)
            pls = PLSRegression(n_components=nc)
            pls.fit(raw_tr, y_p_raw)
            pls_tr = pls.transform(raw_tr)
            pls_te = pls.transform(pls_sc.transform(X_raw_p_te))

            X_tr_aug = np.hstack([X_p_f_tr, pls_tr])
            X_te_aug = np.hstack([X_p_f_te, pls_te])

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_aug)
            X_te_s = sc.transform(X_te_aug)

            D_tr = cdist(X_tr_s, X_tr_s, 'euclidean')
            all_d = D_tr[np.triu_indices(n_tr, k=1)]
            sigma = np.quantile(all_d, best_cfg['bw']) if len(all_d) > 0 else 1.0
            sigma = max(sigma, 1e-6)

            D_te = cdist(X_te_s, X_tr_s, 'euclidean')
            test_pred = np.zeros(n_te)
            for j in range(n_te):
                w = np.exp(-D_te[j]**2 / (2*sigma**2))
                if w.sum() < 1e-10:
                    test_pred[j] = np.mean(y_p_sc)
                else:
                    r = Ridge(alpha=best_cfg['alpha'])
                    r.fit(X_tr_s, y_p_sc, sample_weight=w)
                    test_pred[j] = r.predict(X_te_s[j:j+1])[0]
            tests_final.append(test_pred)

        oof_all_optimized[p_tr_idx] = np.mean(oofs_final, axis=0)
        test_all_optimized[p_te_idx] = np.mean(tests_final, axis=0)

        print(f"  Time: {time.time()-t_p:.1f}s")

    # ================================================================
    # OVERALL RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    mse_default_total = np.mean((oof_all_default - y_angle_sc)**2)
    mse_opt_total = np.mean((oof_all_optimized - y_angle_sc)**2)
    print(f"Default angle CV: {mse_default_total:.6f}")
    print(f"Optimized angle CV: {mse_opt_total:.6f} ({(1-mse_opt_total/mse_default_total)*100:+.2f}%)")

    for pid in sorted(player_configs.keys()):
        mask = pids_train == pid
        mse_d = np.mean((oof_all_default[mask] - y_angle_sc[mask])**2)
        mse_o = np.mean((oof_all_optimized[mask] - y_angle_sc[mask])**2)
        cfg = player_configs[pid]
        print(f"  P{pid}: {mse_d:.6f} -> {mse_o:.6f} ({(1-mse_o/mse_d)*100:+.2f}%) "
              f"[bw={cfg['bw']}, a={cfg['alpha']}, nc={cfg['pls_nc']}, f={cfg['frames']}]")

    # ================================================================
    # GENERATE SUBMISSIONS
    # ================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    base_sub = pd.read_csv(SUBMISSION_DIR / "submission_3411.csv")
    test_ids = test_data['ids']

    submission_details = []

    # Sub 1: All-player angle splice
    sub_num = get_next_submission_number()
    sub = base_sub.copy()
    for i in range(len(test_ids)):
        idx = sub[sub['id'] == test_ids[i]].index[0]
        sub.loc[idx, 'scaled_angle'] = test_all_optimized[i]
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    desc = (f"Sub {sub_num}: ALL-PLAYER angle splice into Sub3411. "
            f"Overall angle CV: {mse_opt_total:.6f} ({(1-mse_opt_total/mse_default_total)*100:+.2f}% vs default). "
            f"Per-player configs: {player_configs}")
    print(f"\n{desc}")
    submission_details.append(desc)

    # Sub 2: Blend 50% with Sub3411 angle
    sub_num2 = get_next_submission_number()
    sub2 = base_sub.copy()
    for i in range(len(test_ids)):
        idx = sub2[sub2['id'] == test_ids[i]].index[0]
        old = sub2.loc[idx, 'scaled_angle']
        sub2.loc[idx, 'scaled_angle'] = 0.5 * test_all_optimized[i] + 0.5 * old
    sub2.to_csv(SUBMISSION_DIR / f"submission_{sub_num2}.csv", index=False)
    desc2 = f"Sub {sub_num2}: 50% all-player angle opt + 50% Sub3411 angle."
    print(f"{desc2}")
    submission_details.append(desc2)

    # Sub 3: Blend 30%
    sub_num3 = get_next_submission_number()
    sub3 = base_sub.copy()
    for i in range(len(test_ids)):
        idx = sub3[sub3['id'] == test_ids[i]].index[0]
        old = sub3.loc[idx, 'scaled_angle']
        sub3.loc[idx, 'scaled_angle'] = 0.3 * test_all_optimized[i] + 0.7 * old
    sub3.to_csv(SUBMISSION_DIR / f"submission_{sub_num3}.csv", index=False)
    desc3 = f"Sub {sub_num3}: 30% all-player angle opt + 70% Sub3411 angle."
    print(f"{desc3}")
    submission_details.append(desc3)

    # Sub 4: Only P4+P5 angle splice (safest - biggest error contributors)
    sub_num4 = get_next_submission_number()
    sub4 = base_sub.copy()
    for pid in [4, 5]:
        p_mask = pids_test == pid
        p_idx = np.where(p_mask)[0]
        for i in p_idx:
            idx = sub4[sub4['id'] == test_ids[i]].index[0]
            sub4.loc[idx, 'scaled_angle'] = test_all_optimized[i]
    sub4.to_csv(SUBMISSION_DIR / f"submission_{sub_num4}.csv", index=False)
    desc4 = f"Sub {sub_num4}: P4+P5 angle splice into Sub3411 (safest, biggest error contributors)."
    print(f"{desc4}")
    submission_details.append(desc4)

    # Sub 5: P2+P4+P5 angle splice (the 3 worst players)
    sub_num5 = get_next_submission_number()
    sub5 = base_sub.copy()
    for pid in [2, 4, 5]:
        p_mask = pids_test == pid
        p_idx = np.where(p_mask)[0]
        for i in p_idx:
            idx = sub5[sub5['id'] == test_ids[i]].index[0]
            sub5.loc[idx, 'scaled_angle'] = test_all_optimized[i]
    sub5.to_csv(SUBMISSION_DIR / f"submission_{sub_num5}.csv", index=False)
    desc5 = f"Sub {sub_num5}: P2+P4+P5 angle splice into Sub3411."
    print(f"{desc5}")
    submission_details.append(desc5)

    # Calibration analysis
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)

    for pid in sorted(np.unique(pids_test)):
        p_mask = pids_test == pid
        p_mask_tr = pids_train == pid
        old_preds = base_sub.set_index('id').loc[test_ids[p_mask], 'scaled_angle'].values
        new_preds = test_all_optimized[p_mask]
        train_std = np.std(y_angle_sc[p_mask_tr])
        print(f"  P{pid}: old_std={np.std(old_preds):.6f}, new_std={np.std(new_preds):.6f}, "
              f"train_std={train_std:.6f}, old_cal={np.std(old_preds)/train_std:.4f}, "
              f"new_cal={np.std(new_preds)/train_std:.4f}")

    # Save research
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Append to research file
    research_path = PROJECT_DIR / "Research" / "P4_ANGLE_FIX_RESULTS_2026-02-19.md"
    with open(research_path, 'a') as f:
        f.write(f"\n\n## Per-Player Angle Optimization (all 5 players)\n")
        f.write(f"Default overall angle CV: {mse_default_total:.6f}\n")
        f.write(f"Optimized overall angle CV: {mse_opt_total:.6f} ({(1-mse_opt_total/mse_default_total)*100:+.2f}%)\n\n")
        f.write("### Per-Player Results\n")
        for pid in sorted(player_configs.keys()):
            mask = pids_train == pid
            mse_d = np.mean((oof_all_default[mask] - y_angle_sc[mask])**2)
            mse_o = np.mean((oof_all_optimized[mask] - y_angle_sc[mask])**2)
            cfg = player_configs[pid]
            f.write(f"- P{pid}: {mse_d:.6f} -> {mse_o:.6f} ({(1-mse_o/mse_d)*100:+.2f}%) "
                    f"bw={cfg['bw']}, alpha={cfg['alpha']}, nc={cfg['pls_nc']}, frames={cfg['frames']}\n")
        f.write("\n### Submissions\n")
        for d in submission_details:
            f.write(f"- {d}\n")
        f.write(f"\nRuntime: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
