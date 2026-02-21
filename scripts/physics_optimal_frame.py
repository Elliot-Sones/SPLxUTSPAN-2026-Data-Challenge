"""
Physics-Optimal Frame Pipeline.

Key findings from frame analysis:
1. The optimal feature extraction frame varies by target:
   - Angle: frame 150-153 (mid follow-through)
   - Depth: frame 150 + release_frame as feature (r=0.45-0.75 per player)
   - Left_right: frame 160-170 (late follow-through)
2. The physics-detected release frame TIMING is a strong per-player predictor of depth
3. Per-shot release frame detection is too noisy for frame-alignment, but the
   scalar value itself encodes when the player released the ball

Strategy: target-specific frame extraction + release_frame feature + PLS + ensemble
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

import lightgbm as lgb
import xgboost as xgb


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


def detect_release_frame(ts_3d, kp_index):
    """Detect per-shot release frame via peak ball speed."""
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
    search_start = 80
    search_end = 200
    wrist_peak = search_start + np.argmax(wrist_z_smooth[search_start:search_end])

    release_end = min(wrist_peak + 5, search_end)
    release_start = max(search_start, wrist_peak - 40)
    search_speeds = speed[release_start:release_end]

    if len(search_speeds) > 0:
        release_frame = release_start + np.argmax(search_speeds)
    else:
        release_frame = max(search_start, wrist_peak - 10)

    return int(np.clip(release_frame, 80, 200))


def extract_features_at_frame(ts_hr, kp_index, frame):
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
            feats[f'hr_{jname}_{cname}_pos'] = series[f]
            vel = np.gradient(series, DT)
            feats[f'hr_{jname}_{cname}_vel'] = vel[f]

    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')

    if all(idx is not None for idx in [rw, re, rs]):
        arm_fwd = ts_hr[f, rw, 0] - ts_hr[f, rs, 0]
        arm_lat = ts_hr[f, rw, 1] - ts_hr[f, rs, 1]
        arm_vert = ts_hr[f, rw, 2] - ts_hr[f, rs, 2]
        feats['arm_ext_fwd'] = arm_fwd
        feats['arm_ext_lat'] = arm_lat
        feats['arm_ext_vert'] = arm_vert

        ua = ts_hr[f, re, :] - ts_hr[f, rs, :]
        fa = ts_hr[f, rw, :] - ts_hr[f, re, :]
        ua_n = np.linalg.norm(ua)
        fa_n = np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            cos_a = np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1)
            feats['elbow_angle'] = np.degrees(np.arccos(cos_a))
        else:
            feats['elbow_angle'] = 90.0
        if fa_n > 1e-6:
            feats['forearm_elev'] = np.degrees(np.arcsin(np.clip(fa[2]/fa_n, -1, 1)))
        else:
            feats['forearm_elev'] = 0.0

        # Wrist velocity components at this frame
        for coord, cname in enumerate(['fwd', 'lat', 'vert']):
            vel = np.gradient(ts_hr[:, rw, coord], DT)
            feats[f'wrist_vel_{cname}'] = vel[f]
            acc = np.gradient(vel, DT)
            feats[f'wrist_acc_{cname}'] = acc[f]

    return feats


def process_all(df):
    """Process all shots, return per-target feature matrices and physics info."""
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    n = len(df)
    n_kp = len(keypoint_cols)
    X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)
    release_frames = np.zeros(n, dtype=np.float32)

    # Target-specific frames: angle=153, depth=150, left_right=165
    # Also extract at multiple frames for a combined feature set
    target_frames = {
        'angle': 153,
        'depth': 150,
        'left_right': 165,
    }
    extra_frames = [140, 150, 153, 160, 170]

    all_feats_per_target = {t: [] for t in target_frames}
    all_feats_combined = []

    for i, (_, row) in enumerate(df.iterrows()):
        ts_3d = np.zeros((240, len(kp_names), 3), dtype=np.float32)
        for j, col in enumerate(keypoint_cols):
            arr = parse_array_string(row[col])
            ts_3d[:, j//3, j%3] = arr
            X_raw[i, j*240:(j+1)*240] = arr

        rf = detect_release_frame(ts_3d, kp_index)
        release_frames[i] = rf

        # Hoop-relative transform
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

        R_mat = np.eye(3, dtype=np.float32)
        R_mat[0, 0] = forward[0]; R_mat[0, 1] = forward[1]
        R_mat[1, 0] = lateral[0]; R_mat[1, 1] = lateral[1]

        centered = ts_3d - player_pos.reshape(1, 1, 3)
        ts_hr = np.einsum('ij,fkj->fki', R_mat, centered)

        # Per-target optimal frame features
        for target, frame in target_frames.items():
            feats = extract_features_at_frame(ts_hr, kp_index, frame)
            feats['release_frame'] = rf
            all_feats_per_target[target].append(feats)

        # Combined: features from multiple frames + release_frame
        combined = {}
        for frame in extra_frames:
            prefix = f'f{frame}_'
            frame_feats = extract_features_at_frame(ts_hr, kp_index, frame)
            for k, v in frame_feats.items():
                combined[f'{prefix}{k}'] = v
        combined['release_frame'] = rf

        # Add release-window dynamics
        for jname in ['right_wrist', 'right_shoulder']:
            idx = kp_index.get(jname)
            if idx is None:
                continue
            for coord, cname in enumerate(['fwd', 'lat', 'vert']):
                series = ts_hr[:, idx, coord]
                vel = np.gradient(series, DT)
                # Change between frame 140 and 170 (follow-through dynamics)
                combined[f'delta_{jname}_{cname}_140_170'] = series[170] - series[140]
                combined[f'vel_range_{jname}_{cname}_140_170'] = np.max(vel[140:170]) - np.min(vel[140:170])

        all_feats_combined.append(combined)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n}")

    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    result = {
        'per_target': {t: pd.DataFrame(all_feats_per_target[t]) for t in target_frames},
        'combined': pd.DataFrame(all_feats_combined),
        'X_raw': X_raw,
        'release_frames': release_frames,
    }
    return result


def cv_eval_target(X, y_scaled, pids, X_raw=None, use_pls=False):
    """Within-player CV for a single target."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    unique_pids = sorted(np.unique(pids))
    oof = np.full(len(y_scaled), np.nan)

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y_scaled[mask]
        raw_p = X_raw[mask] if use_pls and X_raw is not None else None
        indices = np.where(mask)[0]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(X_p):
            X_tr, X_val = X_p[tr_idx], X_p[val_idx]
            y_tr = y_p[tr_idx]

            if raw_p is not None:
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

            lgb_m = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=8,
                min_child_samples=max(5, len(X_tr)//10),
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                verbose=-1, n_jobs=1)
            lgb_m.fit(X_tr, y_tr)
            pred_l = lgb_m.predict(X_val)

            xgb_m = xgb.XGBRegressor(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                verbosity=0, n_jobs=1)
            xgb_m.fit(X_tr, y_tr)
            pred_x = xgb_m.predict(X_val)

            oof[indices[val_idx]] = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x

    nan_mask = np.isnan(oof)
    if np.any(nan_mask):
        oof[nan_mask] = np.nanmean(y_scaled)

    mse = mean_squared_error(y_scaled, oof)
    r = np.corrcoef(y_scaled, oof)[0, 1] if np.std(oof) > 1e-9 else 0.0
    return mse, r, oof


def predict_test(X_train, y_train, pids_train, X_test, pids_test,
                 X_raw_train=None, X_raw_test=None, use_pls=False):
    """Train on full data, predict test."""
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    unique_pids = sorted(np.unique(pids_train))
    preds = np.zeros(len(X_test))

    for pid in unique_pids:
        tr_m = pids_train == pid
        te_m = pids_test == pid
        if not np.any(tr_m) or not np.any(te_m):
            continue

        X_tr = X_train[tr_m]
        X_te = X_test[te_m]
        y_tr = y_train[tr_m]

        if use_pls and X_raw_train is not None and X_raw_test is not None:
            ss = StandardScaler()
            raw_tr_s = ss.fit_transform(X_raw_train[tr_m])
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

        lgb_m = lgb.LGBMRegressor(
            n_estimators=100, num_leaves=8,
            min_child_samples=max(5, len(X_tr)//10),
            learning_rate=0.05, verbose=-1, n_jobs=1)
        lgb_m.fit(X_tr, y_tr)
        pred_l = lgb_m.predict(X_te)

        xgb_m = xgb.XGBRegressor(
            n_estimators=100, max_depth=3,
            learning_rate=0.05, verbosity=0, n_jobs=1)
        xgb_m.fit(X_tr, y_tr)
        pred_x = xgb_m.predict(X_te)

        preds[te_m] = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x

    return np.clip(preds, 0, 1)


def main():
    print("="*70)
    print("PHYSICS-OPTIMAL FRAME PIPELINE")
    print("="*70)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    pids_train = train_df['participant_id'].values
    pids_test = test_df['participant_id'].values

    print("\nProcessing training shots...")
    train_data = process_all(train_df)
    print("Processing test shots...")
    test_data = process_all(test_df)

    rf_train = train_data['release_frames']
    rf_test = test_data['release_frames']
    print(f"\nRelease frames: train mean={rf_train.mean():.1f}, test mean={rf_test.mean():.1f}")

    # Scale targets
    targets_scaled = {}
    scalers = {}
    for target in ['angle', 'depth', 'left_right']:
        y_raw = train_df[target].values
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        scaler = joblib.load(scaler_path)
        targets_scaled[target] = scaler.transform(y_raw.reshape(-1, 1)).ravel()
        scalers[target] = scaler

    # =====================================================
    # CONFIG A: Per-target optimal frame (no PLS)
    # =====================================================
    print("\n" + "="*70)
    print("CONFIG A: Per-target optimal frame (no PLS)")
    print("="*70)

    for target in ['angle', 'depth', 'left_right']:
        feat_df = train_data['per_target'][target]
        feat_cols = [c for c in feat_df.columns if c != 'release_frame']
        X = feat_df[feat_cols].values.astype(np.float32)
        # Add release_frame
        X = np.hstack([X, rf_train.reshape(-1, 1)])

        mse, r, oof = cv_eval_target(X, targets_scaled[target], pids_train)
        print(f"  {target:12s}: MSE={mse:.6f}, r={r:+.4f}")

    # =====================================================
    # CONFIG B: Per-target optimal frame + PLS
    # =====================================================
    print("\n" + "="*70)
    print("CONFIG B: Per-target optimal frame + PLS")
    print("="*70)

    oof_preds = {}
    for target in ['angle', 'depth', 'left_right']:
        feat_df = train_data['per_target'][target]
        feat_cols = [c for c in feat_df.columns if c != 'release_frame']
        X = feat_df[feat_cols].values.astype(np.float32)
        X = np.hstack([X, rf_train.reshape(-1, 1)])

        mse, r, oof = cv_eval_target(
            X, targets_scaled[target], pids_train,
            X_raw=train_data['X_raw'], use_pls=True)
        oof_preds[target] = oof
        print(f"  {target:12s}: MSE={mse:.6f}, r={r:+.4f}")

    mean_mse = np.mean([mean_squared_error(targets_scaled[t], oof_preds[t])
                        for t in targets_scaled])
    print(f"  MEAN MSE = {mean_mse:.6f}")

    # =====================================================
    # CONFIG C: Combined multi-frame features + PLS
    # =====================================================
    print("\n" + "="*70)
    print("CONFIG C: Combined multi-frame + PLS")
    print("="*70)

    combined_df = train_data['combined']
    feat_cols = [c for c in combined_df.columns]
    X_combined = combined_df[feat_cols].values.astype(np.float32)
    print(f"  Features: {X_combined.shape[1]}")

    for target in ['angle', 'depth', 'left_right']:
        mse, r, oof = cv_eval_target(
            X_combined, targets_scaled[target], pids_train,
            X_raw=train_data['X_raw'], use_pls=True)
        print(f"  {target:12s}: MSE={mse:.6f}, r={r:+.4f}")

    # =====================================================
    # CONFIG D: Frame 153 baseline + release_frame + PLS
    # =====================================================
    print("\n" + "="*70)
    print("CONFIG D: Frame 153 + release_frame + PLS (baseline comparison)")
    print("="*70)

    # Use angle features (frame 153) + release_frame for all targets
    feat_df_153 = train_data['per_target']['angle']  # angle extracts at 153
    feat_cols = [c for c in feat_df_153.columns if c != 'release_frame']
    X_153 = feat_df_153[feat_cols].values.astype(np.float32)
    X_153_rf = np.hstack([X_153, rf_train.reshape(-1, 1)])

    for target in ['angle', 'depth', 'left_right']:
        mse, r, oof = cv_eval_target(
            X_153_rf, targets_scaled[target], pids_train,
            X_raw=train_data['X_raw'], use_pls=True)
        print(f"  {target:12s}: MSE={mse:.6f}, r={r:+.4f}")

    # =====================================================
    # GENERATE SUBMISSIONS
    # =====================================================
    print("\n" + "="*70)
    print("GENERATING SUBMISSIONS")
    print("="*70)

    # Best config: per-target optimal frame + PLS
    test_preds = {}
    for target in ['angle', 'depth', 'left_right']:
        train_feat_df = train_data['per_target'][target]
        test_feat_df = test_data['per_target'][target]
        feat_cols = [c for c in train_feat_df.columns if c != 'release_frame']

        X_tr = train_feat_df[feat_cols].values.astype(np.float32)
        X_te = test_feat_df[feat_cols].values.astype(np.float32)
        X_tr = np.hstack([X_tr, rf_train.reshape(-1, 1)])
        X_te = np.hstack([X_te, rf_test.reshape(-1, 1)])

        preds = predict_test(
            X_tr, targets_scaled[target], pids_train,
            X_te, pids_test,
            X_raw_train=train_data['X_raw'], X_raw_test=test_data['X_raw'],
            use_pls=True)
        test_preds[f'scaled_{target}'] = preds

    # Blend with Sub 784
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    max_num = max([int(p.stem.split('_')[1]) for p in existing]) if existing else 0

    print("\n  Blending with Sub 784:")
    for w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        sub = sub_784.copy()
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            sub[col] = (1 - w) * sub_784[col] + w * test_preds[col]

        a_std = sub['scaled_angle'].std()
        d_mean = sub['scaled_depth'].mean()

        max_num += 1
        path = SUBMISSION_DIR / f"submission_{max_num}.csv"
        sub.to_csv(path, index=False)
        print(f"    w={w:.2f}: angle_std={a_std:.4f}, depth_mean={d_mean:.4f} -> {path.name}")

    # Also: target-specific blend weights
    print("\n  Target-specific blends:")
    for aw, dw, lw in [(0.0, 0.15, 0.15), (0.0, 0.20, 0.20), (0.0, 0.10, 0.30),
                        (0.05, 0.15, 0.25), (0.10, 0.20, 0.30)]:
        sub = sub_784.copy()
        sub['scaled_angle'] = (1-aw) * sub_784['scaled_angle'] + aw * test_preds['scaled_angle']
        sub['scaled_depth'] = (1-dw) * sub_784['scaled_depth'] + dw * test_preds['scaled_depth']
        sub['scaled_left_right'] = (1-lw) * sub_784['scaled_left_right'] + lw * test_preds['scaled_left_right']

        a_std = sub['scaled_angle'].std()
        d_mean = sub['scaled_depth'].mean()

        max_num += 1
        path = SUBMISSION_DIR / f"submission_{max_num}.csv"
        sub.to_csv(path, index=False)
        print(f"    aw={aw:.2f},dw={dw:.2f},lw={lw:.2f}: "
              f"angle_std={a_std:.4f}, depth_mean={d_mean:.4f} -> {path.name}")


if __name__ == "__main__":
    main()
