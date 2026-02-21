"""
Physics-Informed Feature Extraction.

Key insight: The existing hoop-relative pipeline extracts features at FIXED
frame 153 for all shots. But the actual release frame varies from ~100 to ~195
(std=25 frames). Using physics to detect the correct release frame, then
extracting hoop-relative features at that frame, should improve signal quality.

This uses physics as an INFORMER to the feature extraction, not as a feature source.

Also tests: multi-frame extraction at physics-aligned frames
(release-20, release-10, release, release+5, release+10)
"""

import json
import sys
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
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
HOOP_POS = np.array([5.25, -25.0, 10.0])
FEET_TO_METERS = 0.3048
DT = 1.0 / 60.0

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window, polyorder, **kwargs):
    """Savgol filter with NaN/inf protection."""
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


def detect_release_frame(ts_3d, kp_index):
    """
    Detect per-shot release frame using physics criteria:
    1. Find peak ball speed from Savgol derivative of wrist + fingertip center
    2. Constrain to before wrist peak height
    """
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 153  # fallback

    wrist_traj = ts_3d[:, rw_idx, :].copy()

    # Clean wrist trajectory NaN/inf
    for ax in range(3):
        vals = wrist_traj[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 153  # fallback
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, ax] = vals

    # Fingertip center (average of 3 fingertips to reduce noise)
    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = []
    for key in ft_keys:
        idx = kp_index.get(key)
        if idx is not None:
            ft_trajs.append(ts_3d[:, idx, :])

    if len(ft_trajs) > 0:
        ft_center = np.nanmean(ft_trajs, axis=0)
        # Handle NaN/inf
        for ax in range(3):
            vals = ft_center[:, ax]
            bad = np.isnan(vals) | np.isinf(vals)
            if np.all(bad):
                ft_center[:, ax] = wrist_traj[:, ax]
            elif np.any(bad):
                good = ~bad
                ft_center[bad, ax] = np.interp(
                    np.where(bad)[0], np.where(good)[0], vals[good]
                )
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()

    # Smooth ball position
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    # Compute ball speed (meters/s)
    ball_m = ball * FEET_TO_METERS
    vel = np.zeros_like(ball_m)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball_m[:, ax], 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)

    # Wrist peak height
    wrist_z = wrist_traj[:, 2]
    wrist_z_smooth = safe_savgol(wrist_z, 11, 3)

    # Search for peak ball speed before wrist peak height
    search_start = 80
    search_end = 200
    wrist_peak = search_start + np.argmax(wrist_z_smooth[search_start:search_end])

    # Peak ball speed before wrist peak + 5 frames
    release_end = min(wrist_peak + 5, search_end)
    release_start = max(search_start, wrist_peak - 40)

    search_speeds = speed[release_start:release_end]
    if len(search_speeds) > 0:
        release_frame = release_start + np.argmax(search_speeds)
    else:
        release_frame = max(search_start, wrist_peak - 10)

    return int(np.clip(release_frame, 80, 200))


def extract_features_at_frame(ts_3d, ts_hr, kp_index, frame, prefix=""):
    """Extract hoop-relative features at a specific frame."""
    feats = {}

    key_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'left_shoulder',
        'right_hip', 'left_hip', 'mid_hip',
        'neck', 'nose',
    ]

    f = int(np.clip(frame, 0, 239))

    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            continue

        for coord, cname in enumerate(['fwd', 'lat', 'vert']):
            series = ts_hr[:, idx, coord]
            # Position at frame
            feats[f'{prefix}hr_{jname}_{cname}_pos'] = series[f]
            # Velocity at frame
            vel = np.gradient(series, DT)
            feats[f'{prefix}hr_{jname}_{cname}_vel'] = vel[f]

    # Arm mechanics
    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')

    if all(idx is not None for idx in [rw, re, rs]):
        # Arm extension
        arm_fwd = ts_hr[f, rw, 0] - ts_hr[f, rs, 0]
        arm_lat = ts_hr[f, rw, 1] - ts_hr[f, rs, 1]
        arm_vert = ts_hr[f, rw, 2] - ts_hr[f, rs, 2]
        feats[f'{prefix}arm_ext_fwd'] = arm_fwd
        feats[f'{prefix}arm_ext_lat'] = arm_lat
        feats[f'{prefix}arm_ext_vert'] = arm_vert

        # Elbow angle
        ua = ts_hr[f, re, :] - ts_hr[f, rs, :]
        fa = ts_hr[f, rw, :] - ts_hr[f, re, :]
        ua_n = np.linalg.norm(ua)
        fa_n = np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            cos_a = np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1)
            feats[f'{prefix}elbow_angle'] = np.degrees(np.arccos(cos_a))
        else:
            feats[f'{prefix}elbow_angle'] = 90.0

        # Forearm elevation
        if fa_n > 1e-6:
            feats[f'{prefix}forearm_elev'] = np.degrees(np.arcsin(np.clip(fa[2]/fa_n, -1, 1)))
        else:
            feats[f'{prefix}forearm_elev'] = 0.0

        # Arm velocity
        fwd_ts = ts_hr[:, rw, 0] - ts_hr[:, rs, 0]
        lat_ts = ts_hr[:, rw, 1] - ts_hr[:, rs, 1]
        feats[f'{prefix}arm_ext_fwd_vel'] = np.gradient(fwd_ts, DT)[f]
        feats[f'{prefix}arm_ext_lat_vel'] = np.gradient(lat_ts, DT)[f]

    return feats


def process_all_shots(train_df, test_df):
    """Process all shots with physics-informed feature extraction."""
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process_df(df, is_train=True):
        n = len(df)
        all_feats = []
        n_kp = len(keypoint_cols)
        X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            # Parse 3D timeseries
            ts_3d = np.zeros((240, len(kp_names), 3), dtype=np.float32)
            for j, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                ts_3d[:, j//3, j%3] = arr
                X_raw[i, j*240:(j+1)*240] = arr

            pid = row['participant_id']

            # Physics-informed release frame detection
            release_frame = detect_release_frame(ts_3d, kp_index)

            # Compute hoop-relative transform
            mid_hip_idx = kp_index.get('mid_hip', 0)
            player_pos = ts_3d[120, mid_hip_idx, :].copy()
            player_pos[2] = 0

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

            feats = {'participant_id': pid, 'release_frame': release_frame}

            # Extract features at PHYSICS-DETECTED release frame
            rf = release_frame
            feats.update(extract_features_at_frame(ts_hr, ts_hr, kp_index, rf, prefix='rel_'))

            # Also extract at frames RELATIVE to release
            for offset, name in [(-20, 'pre20_'), (-10, 'pre10_'), (-5, 'pre5_'),
                                  (5, 'post5_'), (10, 'post10_')]:
                f = int(np.clip(rf + offset, 0, 239))
                feats.update(extract_features_at_frame(ts_hr, ts_hr, kp_index, f, prefix=name))

            # Also extract at FIXED frame 153 for comparison
            feats.update(extract_features_at_frame(ts_hr, ts_hr, kp_index, 153, prefix='f153_'))

            # Summary statistics in release window
            for jname in ['right_wrist', 'right_shoulder', 'neck']:
                idx = kp_index.get(jname)
                if idx is None:
                    continue
                for coord, cname in enumerate(['fwd', 'lat', 'vert']):
                    win_start = max(0, rf - 20)
                    win_end = min(240, rf + 10)
                    series = ts_hr[win_start:win_end, idx, coord]
                    feats[f'win_{jname}_{cname}_mean'] = np.nanmean(series)
                    feats[f'win_{jname}_{cname}_std'] = np.nanstd(series)
                    feats[f'win_{jname}_{cname}_range'] = np.nanmax(series) - np.nanmin(series)

                    vel = np.gradient(series, DT)
                    feats[f'win_{jname}_{cname}_vel_mean'] = np.nanmean(vel)
                    feats[f'win_{jname}_{cname}_vel_max'] = np.nanmax(vel)
                    feats[f'win_{jname}_{cname}_vel_min'] = np.nanmin(vel)

            all_feats.append(feats)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n}")

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        feat_df = pd.DataFrame(all_feats)

        return feat_df, X_raw

    print("Processing training shots...")
    train_feats, X_raw_train = process_df(train_df, True)
    print("Processing test shots...")
    test_feats, X_raw_test = process_df(test_df, False)

    return train_feats, test_feats, X_raw_train, X_raw_test


def evaluate_cv(feat_df, X_raw, y_targets, pids, use_pls=False, config_name=""):
    """Within-player CV with optional PLS augmentation."""
    exclude = {'participant_id', 'release_frame'}
    feat_cols = [c for c in feat_df.columns if c not in exclude]
    X = feat_df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    unique_pids = sorted(np.unique(pids))
    results = {}

    for t_idx, target in enumerate(['angle', 'depth', 'left_right']):
        y_raw = y_targets[:, t_idx]
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            y = scaler.transform(y_raw.reshape(-1, 1)).ravel()
        else:
            y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)

        oof = np.full(len(y), np.nan)

        for pid in unique_pids:
            mask = pids == pid
            X_p = X[mask]
            y_p = y[mask]
            raw_p = X_raw[mask] if use_pls else None
            indices = np.where(mask)[0]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for tr_idx, val_idx in kf.split(X_p):
                X_tr, X_val = X_p[tr_idx], X_p[val_idx]
                y_tr = y_p[tr_idx]

                if use_pls and raw_p is not None:
                    ss = StandardScaler()
                    raw_tr_s = ss.fit_transform(raw_p[tr_idx])
                    raw_val_s = ss.transform(raw_p[val_idx])
                    nc = min(20, len(raw_tr_s) - 1)
                    if nc >= 3:
                        pls = PLSRegression(n_components=nc)
                        pls.fit(raw_tr_s, y_tr)
                        X_tr = np.hstack([X_tr, pls.transform(raw_tr_s)])
                        X_val = np.hstack([X_val, pls.transform(raw_val_s)])

                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr, y_tr)
                pred_r = ridge.predict(X_val)
                pred = pred_r

                if lgb is not None and len(X_tr) >= 10:
                    lgb_m = lgb.LGBMRegressor(
                        n_estimators=100, num_leaves=8,
                        min_child_samples=max(5, len(X_tr)//10),
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        verbose=-1, n_jobs=1)
                    lgb_m.fit(X_tr, y_tr)
                    pred_l = lgb_m.predict(X_val)

                    if xgb is not None:
                        xgb_m = xgb.XGBRegressor(
                            n_estimators=100, max_depth=3,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                            verbosity=0, n_jobs=1)
                        xgb_m.fit(X_tr, y_tr)
                        pred_x = xgb_m.predict(X_val)
                        pred = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x
                    else:
                        pred = 0.4 * pred_r + 0.6 * pred_l

                oof[indices[val_idx]] = pred

        nan_mask = np.isnan(oof)
        if np.any(nan_mask):
            oof[nan_mask] = np.mean(y)

        mse = mean_squared_error(y, oof)
        r = np.corrcoef(y, oof)[0, 1] if np.std(oof) > 1e-9 else 0.0
        results[target] = {'mse': mse, 'r': r, 'oof': oof}

    return results


def generate_submission(feat_df, X_raw, y_targets, pids_train,
                        test_feats, X_raw_test, pids_test, use_pls):
    """Train on full data, predict test, blend with Sub 784."""
    exclude = {'participant_id', 'release_frame'}
    feat_cols = [c for c in feat_df.columns if c not in exclude]

    X_train = feat_df[feat_cols].values.astype(np.float32)
    X_test = test_feats[feat_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    unique_pids = sorted(np.unique(pids_train))
    test_preds = {}

    for t_idx, target in enumerate(['angle', 'depth', 'left_right']):
        y_raw = y_targets[:, t_idx]
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            y = scaler.transform(y_raw.reshape(-1, 1)).ravel()
        else:
            y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)

        preds = np.zeros(len(X_test))
        for pid in unique_pids:
            tr_m = pids_train == pid
            te_m = pids_test == pid
            if not np.any(tr_m) or not np.any(te_m):
                continue

            X_tr = X_train[tr_m]
            X_te = X_test[te_m]
            y_tr = y[tr_m]

            if use_pls:
                ss = StandardScaler()
                raw_tr_s = ss.fit_transform(X_raw[tr_m])
                raw_te_s = ss.transform(X_raw_test[te_m])
                nc = min(20, len(raw_tr_s) - 1)
                if nc >= 3:
                    pls = PLSRegression(n_components=nc)
                    pls.fit(raw_tr_s, y_tr)
                    X_tr = np.hstack([X_tr, pls.transform(raw_tr_s)])
                    X_te = np.hstack([X_te, pls.transform(raw_te_s)])

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr, y_tr)
            pred_r = ridge.predict(X_te)

            if lgb is not None and len(X_tr) >= 10:
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=8,
                    min_child_samples=max(5, len(X_tr)//10),
                    learning_rate=0.05, verbose=-1, n_jobs=1)
                lgb_m.fit(X_tr, y_tr)
                pred_l = lgb_m.predict(X_te)
                if xgb is not None:
                    xgb_m = xgb.XGBRegressor(
                        n_estimators=100, max_depth=3,
                        learning_rate=0.05, verbosity=0, n_jobs=1)
                    xgb_m.fit(X_tr, y_tr)
                    pred_x = xgb_m.predict(X_te)
                    preds[te_m] = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x
                else:
                    preds[te_m] = 0.4 * pred_r + 0.6 * pred_l
            else:
                preds[te_m] = pred_r

        test_preds[target] = np.clip(preds, 0, 1)

    # Blend with Sub 784
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    max_num = max([int(p.stem.split('_')[1]) for p in existing]) if existing else 0

    print("\n  Blend weights with Sub 784:")
    for w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        sub = sub_784.copy()
        for target in ['angle', 'depth', 'left_right']:
            col = f'scaled_{target}'
            sub[col] = (1 - w) * sub_784[col] + w * test_preds[target]

        a_std = sub['scaled_angle'].std()
        d_mean = sub['scaled_depth'].mean()

        max_num += 1
        path = SUBMISSION_DIR / f"submission_{max_num}.csv"
        sub.to_csv(path, index=False)
        print(f"    w={w:.2f}: angle_std={a_std:.4f}, depth_mean={d_mean:.4f} -> {path.name}")


def main():
    print("="*70)
    print("PHYSICS-INFORMED FEATURE EXTRACTION")
    print("="*70)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_feats, test_feats, X_raw_train, X_raw_test = process_all_shots(train_df, test_df)

    exclude = {'participant_id', 'release_frame'}
    feat_cols = [c for c in train_feats.columns if c not in exclude]
    print(f"\nExtracted {len(feat_cols)} physics-informed features")

    # Release frame statistics
    rf = train_feats['release_frame'].values
    print(f"Release frame: mean={rf.mean():.1f}, std={rf.std():.1f}, "
          f"range=[{rf.min()}, {rf.max()}]")

    y = train_df[['angle', 'depth', 'left_right']].values
    pids_train = train_df['participant_id'].values
    pids_test = test_df['participant_id'].values

    # Test configurations
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)

    configs = [
        ("Physics-informed HR (no PLS)", False),
        ("Physics-informed HR + PLS", True),
    ]

    for name, use_pls in configs:
        print(f"\n--- {name} ---")
        results = evaluate_cv(train_feats, X_raw_train, y, pids_train, use_pls)
        for t in ['angle', 'depth', 'left_right']:
            print(f"  {t:12s}: MSE={results[t]['mse']:.6f}, r={results[t]['r']:+.4f}")
        mean_mse = np.mean([results[t]['mse'] for t in results])
        print(f"  MEAN MSE = {mean_mse:.6f}")

    # Generate submission with best config (PLS)
    print("\n" + "="*70)
    print("GENERATING SUBMISSION")
    print("="*70)

    generate_submission(train_feats, X_raw_train, y, pids_train,
                        test_feats, X_raw_test, pids_test, use_pls=True)


if __name__ == "__main__":
    main()
