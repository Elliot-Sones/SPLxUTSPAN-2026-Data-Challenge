"""
Player 5 Targeted Modeling

Player 5 dominates error across all approaches (2-3x higher MSE than others).
Key stats: depth std = 8.16 (vs 4.20 for others), angle std = 4.10 (vs 2.17 for others)

Strategy: Keep Sub 784's blend for Players 1-4. For Player 5, test specialized approaches:
1. Global model: borrow strength from all players (with player encoding)
2. Heavy regularization: simpler models to avoid overfitting high-variance targets
3. PLS with fewer components: reduce overfitting on raw timeseries
4. Ridge-only: maximally simple model
5. Blending multiple random seeds: reduce variance of Player 5 predictions
6. Mean prediction: ultimate baseline - just predict player mean
"""

import json
import sys
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
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
PLAYER5_ID = 5


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


def load_raw_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp = len(keypoint_cols)

    keypoint_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoint_names.append(col[:-2])

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Keypoints: {len(keypoint_names)}")

    def process(df, is_train=True):
        n = len(df)
        X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)
        X_3d = np.zeros((n, 240, len(keypoint_names), 3), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr
                kp_idx = col_i // 3
                coord_idx = col_i % 3
                X_3d[idx, :, kp_idx, coord_idx] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        result = {'X_raw': X_raw, 'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'keypoint_names': keypoint_names}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


# ============================================================
# HOOP-RELATIVE FEATURES
# ============================================================

def compute_hoop_relative_transform(player_pos):
    hoop_2d = HOOP_POS[:2]
    player_2d = player_pos[:2]
    forward = hoop_2d - player_2d
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, -1.0])
    else:
        forward = forward / forward_norm
    lateral = np.array([-forward[1], forward[0]])
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    return R, player_pos


def extract_hoop_relative_features(ts_3d, keypoint_names, pid):
    feats = {'participant_id': pid}
    kp_index = {name: i for i, name in enumerate(keypoint_names)}

    mh_idx = kp_index.get('mid_hip')
    if mh_idx is not None:
        player_pos = np.nanmean(ts_3d[:10, mh_idx, :], axis=0)
    else:
        player_pos = np.nanmean(ts_3d[:10, :, :].mean(axis=1), axis=0)

    R, origin = compute_hoop_relative_transform(player_pos)
    centered = ts_3d - origin.reshape(1, 1, 3)
    ts_hoop = np.einsum('ij,fkj->fki', R, centered)

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist',
                  'left_shoulder', 'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'neck', 'nose']

    for joint in key_joints:
        if joint not in kp_index: continue
        idx = kp_index[joint]
        for coord, cname in enumerate(['forward', 'lateral', 'vertical']):
            s = ts_hoop[:, idx, coord]
            prefix = f"hr_{joint}_{cname}"
            feats[f"{prefix}_mean"] = np.nanmean(s)
            feats[f"{prefix}_std"] = np.nanstd(s)
            feats[f"{prefix}_min"] = np.nanmin(s)
            feats[f"{prefix}_max"] = np.nanmax(s)
            feats[f"{prefix}_range"] = np.nanmax(s) - np.nanmin(s)
            feats[f"{prefix}_release_mean"] = np.nanmean(s[140:180])
            vel = np.gradient(s, 1.0/60.0)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)
            feats[f"{prefix}_vel_at_release"] = vel[153] if len(vel) > 153 else 0.0

        for c, cname in enumerate(['x', 'y', 'z']):
            s = ts_3d[:, idx, c]
            prefix = f"{joint}_{cname}"
            feats[f"{prefix}_mean"] = np.nanmean(s)
            feats[f"{prefix}_std"] = np.nanstd(s)
            feats[f"{prefix}_min"] = np.nanmin(s)
            feats[f"{prefix}_max"] = np.nanmax(s)
            feats[f"{prefix}_range"] = np.nanmax(s) - np.nanmin(s)
            vel = np.gradient(s, 1.0/60.0)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)
            feats[f"f153_{prefix}"] = s[153]

    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    rs, ls = kp_index.get('right_shoulder'), kp_index.get('left_shoulder')
    rw, lw = kp_index.get('right_wrist'), kp_index.get('left_wrist')

    if rh is not None and lh is not None:
        hip_lat = ts_hoop[:, rh, 1] - ts_hoop[:, lh, 1]
        feats['hr_hip_alignment_mean'] = np.nanmean(hip_lat)
        feats['hr_hip_alignment_release'] = hip_lat[153]
    if rs is not None and ls is not None:
        shoulder_lat = ts_hoop[:, rs, 1] - ts_hoop[:, ls, 1]
        feats['hr_shoulder_alignment_mean'] = np.nanmean(shoulder_lat)
        feats['hr_shoulder_alignment_release'] = shoulder_lat[153]
    if rw is not None and lw is not None:
        guide_lat = ts_hoop[:, lw, 1] - ts_hoop[:, rw, 1]
        feats['hr_guide_hand_lateral_release'] = guide_lat[153]
    if rw is not None and rs is not None:
        arm_lat = ts_hoop[:, rw, 1] - ts_hoop[:, rs, 1]
        feats['hr_arm_lateral_dev_release'] = arm_lat[153]

    for j1, j2, j3, name in [('right_shoulder', 'right_elbow', 'right_wrist', 'elbow')]:
        if all(j in kp_index for j in [j1, j2, j3]):
            p1, p2, p3 = ts_3d[:, kp_index[j1]], ts_3d[:, kp_index[j2]], ts_3d[:, kp_index[j3]]
            v1, v2 = p1 - p2, p3 - p2
            dot = np.sum(v1 * v2, axis=1)
            n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
            denom = n1 * n2; denom[denom == 0] = 1e-10
            angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            feats[f"{name}_angle_release"] = angle[153]
            feats[f"{name}_angle_range"] = np.nanmax(angle) - np.nanmin(angle)

    for pname, (s, e) in [('load', (60, 120)), ('propel', (120, 170))]:
        for joint in ['right_wrist', 'right_elbow']:
            if joint not in kp_index: continue
            idx2 = kp_index[joint]
            for c in range(3):
                vel = np.gradient(ts_3d[s:e, idx2, c], 1.0/60.0)
                feats[f"phase_{pname}_{joint}_{'xyz'[c]}_vel_max"] = np.nanmax(vel)
    return feats


# ============================================================
# PLAYER 5 STRATEGIES
# ============================================================

def strategy_global_model(train_data, test_data, target_idx, feat_names):
    """
    Strategy 1: Train on ALL players (not just Player 5).
    Player 5 gets 74 train samples alone, but 345 from all players.
    Uses player_id as a feature. More data = less overfitting.
    """
    X_3d_tr, y, pids_tr = train_data['X_3d'], train_data['y'], train_data['pids']
    X_3d_te, pids_te = test_data['X_3d'], test_data['pids']
    kp_names = train_data['keypoint_names']

    # Extract features for all players
    all_feats_tr = [extract_hoop_relative_features(X_3d_tr[i], kp_names, pids_tr[i]) for i in range(len(X_3d_tr))]
    X_feat_tr = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats_tr], dtype=np.float32)
    X_feat_tr = np.nan_to_num(X_feat_tr, nan=0.0, posinf=0.0, neginf=0.0)

    all_feats_te = [extract_hoop_relative_features(X_3d_te[i], kp_names, pids_te[i]) for i in range(len(X_3d_te))]
    X_feat_te = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats_te], dtype=np.float32)
    X_feat_te = np.nan_to_num(X_feat_te, nan=0.0, posinf=0.0, neginf=0.0)

    y_target = y[:, target_idx]

    # Train global model on ALL data
    scaler = StandardScaler()
    X_tr_scaled = np.nan_to_num(scaler.fit_transform(X_feat_tr), nan=0.0)
    X_te_scaled = np.nan_to_num(scaler.transform(X_feat_te), nan=0.0)

    # CV for Player 5 only
    p5_mask_tr = pids_tr == PLAYER5_ID
    p5_mask_te = pids_te == PLAYER5_ID
    p5_idx_tr = np.where(p5_mask_tr)[0]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_p5 = np.zeros(p5_mask_tr.sum())

    # For CV, train on non-Player5 + train folds of Player5, validate on Player5 val fold
    for fold, (tr_idx_local, val_idx_local) in enumerate(kf.split(np.arange(p5_mask_tr.sum()))):
        # Global indices: all non-P5 + P5 train fold
        non_p5_idx = np.where(~p5_mask_tr)[0]
        p5_tr_global = p5_idx_tr[tr_idx_local]
        train_idx = np.concatenate([non_p5_idx, p5_tr_global])
        val_idx = p5_idx_tr[val_idx_local]

        X_tr, y_tr = X_tr_scaled[train_idx], y_target[train_idx]
        X_val = X_tr_scaled[val_idx]

        preds = []
        for cls, params in [
            (lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=8, learning_rate=0.03,
                reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1)),
            (xgb.XGBRegressor, dict(n_estimators=100, max_depth=3, learning_rate=0.03,
                reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbosity=0, n_jobs=-1)),
            (Ridge, dict(alpha=10.0)),
        ]:
            m = cls(**params)
            m.fit(X_tr, y_tr)
            preds.append(m.predict(X_val))
        oof_p5[val_idx_local] = 0.35 * preds[0] + 0.35 * preds[1] + 0.30 * preds[2]

    cv_mse = np.mean((oof_p5 - y_target[p5_mask_tr]) ** 2)

    # Final model on all data
    preds_list = []
    for cls, params in [
        (lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=8, learning_rate=0.03,
            reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=-1)),
        (xgb.XGBRegressor, dict(n_estimators=100, max_depth=3, learning_rate=0.03,
            reg_alpha=2.0, reg_lambda=2.0, random_state=42, verbosity=0, n_jobs=-1)),
        (Ridge, dict(alpha=10.0)),
    ]:
        m = cls(**params)
        m.fit(X_tr_scaled, y_target)
        preds_list.append(m.predict(X_te_scaled[p5_mask_te]))

    test_preds = 0.35 * preds_list[0] + 0.35 * preds_list[1] + 0.30 * preds_list[2]
    return cv_mse, test_preds, oof_p5


def strategy_heavy_ridge(train_data, test_data, target_idx, feat_names):
    """
    Strategy 2: Ridge-only with heavy regularization for Player 5.
    Maximally simple model - avoids overfitting high variance targets.
    """
    X_3d_tr, y, pids_tr = train_data['X_3d'], train_data['y'], train_data['pids']
    X_3d_te, pids_te = test_data['X_3d'], test_data['pids']
    kp_names = train_data['keypoint_names']

    p5_mask_tr = pids_tr == PLAYER5_ID
    p5_mask_te = pids_te == PLAYER5_ID

    all_feats_tr = [extract_hoop_relative_features(X_3d_tr[i], kp_names, pids_tr[i]) for i in np.where(p5_mask_tr)[0]]
    X_feat = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats_tr], dtype=np.float32)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    all_feats_te = [extract_hoop_relative_features(X_3d_te[i], kp_names, pids_te[i]) for i in np.where(p5_mask_te)[0]]
    X_feat_te = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats_te], dtype=np.float32)
    X_feat_te = np.nan_to_num(X_feat_te, nan=0.0, posinf=0.0, neginf=0.0)

    y_target = y[p5_mask_tr, target_idx]

    scaler = StandardScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(X_feat), nan=0.0)
    X_te_scaled = np.nan_to_num(scaler.transform(X_feat_te), nan=0.0)

    # CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_target))
    for tr_idx, val_idx in kf.split(X_scaled):
        ridge = RidgeCV(alphas=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
        ridge.fit(X_scaled[tr_idx], y_target[tr_idx])
        oof[val_idx] = ridge.predict(X_scaled[val_idx])

    cv_mse = np.mean((oof - y_target) ** 2)

    # Final
    ridge = RidgeCV(alphas=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
    ridge.fit(X_scaled, y_target)
    test_preds = ridge.predict(X_te_scaled)
    return cv_mse, test_preds, oof


def strategy_pls_tuned(train_data, test_data, target_idx):
    """
    Strategy 3: PLS with aggressive component search for Player 5.
    Raw timeseries with very few components to avoid overfitting.
    """
    X_raw, y, pids = train_data['X_raw'], train_data['y'], train_data['pids']
    X_raw_te, pids_te = test_data['X_raw'], test_data['pids']

    p5_mask_tr = pids == PLAYER5_ID
    p5_mask_te = pids_te == PLAYER5_ID

    X_p5 = X_raw[p5_mask_tr]
    y_target = y[p5_mask_tr, target_idx]
    X_p5_te = X_raw_te[p5_mask_te]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_p5)
    X_te_scaled = scaler.transform(X_p5_te)

    # Try very few components (1-15)
    candidates = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    best_nc, best_mse = 3, float('inf')

    for nc in candidates:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mses = []
        for tr_idx, val_idx in kf.split(X_scaled):
            pls = PLSRegression(n_components=nc)
            pls.fit(X_scaled[tr_idx], y_target[tr_idx])
            pls_pred = pls.predict(X_scaled[val_idx]).flatten()

            # Also fit Ridge on PLS components
            X_tr_pls = pls.transform(X_scaled[tr_idx])
            X_val_pls = pls.transform(X_scaled[val_idx])
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_tr_pls, y_target[tr_idx])
            ridge_pred = ridge.predict(X_val_pls)

            pred = 0.5 * pls_pred + 0.5 * ridge_pred
            mses.append(np.mean((pred - y_target[val_idx]) ** 2))
        avg = np.mean(mses)
        if avg < best_mse:
            best_mse = avg
            best_nc = nc

    print(f"    PLS tuned: best components = {best_nc}")

    # Get OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_target))
    for tr_idx, val_idx in kf.split(X_scaled):
        pls = PLSRegression(n_components=best_nc)
        pls.fit(X_scaled[tr_idx], y_target[tr_idx])
        pls_pred = pls.predict(X_scaled[val_idx]).flatten()
        X_tr_pls = pls.transform(X_scaled[tr_idx])
        X_val_pls = pls.transform(X_scaled[val_idx])
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_tr_pls, y_target[tr_idx])
        ridge_pred = ridge.predict(X_val_pls)
        oof[val_idx] = 0.5 * pls_pred + 0.5 * ridge_pred

    cv_mse = np.mean((oof - y_target) ** 2)

    # Final
    pls = PLSRegression(n_components=best_nc)
    pls.fit(X_scaled, y_target)
    pls_pred = pls.predict(X_te_scaled).flatten()
    X_pls_all = pls.transform(X_scaled)
    X_te_pls = pls.transform(X_te_scaled)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_pls_all, y_target)
    ridge_pred = ridge.predict(X_te_pls)
    test_preds = 0.5 * pls_pred + 0.5 * ridge_pred

    return cv_mse, test_preds, oof


def strategy_multi_seed_ensemble(train_data, test_data, target_idx, feat_names):
    """
    Strategy 4: Ensemble across multiple random seeds for Player 5.
    Reduces prediction variance by averaging diverse models.
    """
    X_3d_tr, y, pids_tr = train_data['X_3d'], train_data['y'], train_data['pids']
    X_3d_te, pids_te = test_data['X_3d'], test_data['pids']
    kp_names = train_data['keypoint_names']

    p5_mask_tr = pids_tr == PLAYER5_ID
    p5_mask_te = pids_te == PLAYER5_ID

    all_feats_tr = [extract_hoop_relative_features(X_3d_tr[i], kp_names, pids_tr[i]) for i in np.where(p5_mask_tr)[0]]
    X_feat = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats_tr], dtype=np.float32)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    all_feats_te = [extract_hoop_relative_features(X_3d_te[i], kp_names, pids_te[i]) for i in np.where(p5_mask_te)[0]]
    X_feat_te = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats_te], dtype=np.float32)
    X_feat_te = np.nan_to_num(X_feat_te, nan=0.0, posinf=0.0, neginf=0.0)

    y_target = y[p5_mask_tr, target_idx]

    scaler = StandardScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(X_feat), nan=0.0)
    X_te_scaled = np.nan_to_num(scaler.transform(X_feat_te), nan=0.0)

    seeds = [42, 123, 456, 789, 1024, 2048, 3333, 4444, 5555, 9999]
    all_oof = []
    all_test = []

    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y_target))
        for tr_idx, val_idx in kf.split(X_scaled):
            preds = []
            for cls, params in [
                (lgb.LGBMRegressor, dict(n_estimators=80, num_leaves=8, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1,
                    subsample=0.8, colsample_bytree=0.8)),
                (xgb.XGBRegressor, dict(n_estimators=80, max_depth=3, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbosity=0, n_jobs=-1,
                    subsample=0.8, colsample_bytree=0.8)),
                (Ridge, dict(alpha=10.0)),
            ]:
                m = cls(**params)
                m.fit(X_scaled[tr_idx], y_target[tr_idx])
                preds.append(m.predict(X_scaled[val_idx]))
            oof[val_idx] = 0.35 * preds[0] + 0.35 * preds[1] + 0.30 * preds[2]
        all_oof.append(oof)

        # Test predictions for this seed
        test_preds = []
        for cls, params in [
            (lgb.LGBMRegressor, dict(n_estimators=80, num_leaves=8, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1,
                subsample=0.8, colsample_bytree=0.8)),
            (xgb.XGBRegressor, dict(n_estimators=80, max_depth=3, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbosity=0, n_jobs=-1,
                subsample=0.8, colsample_bytree=0.8)),
            (Ridge, dict(alpha=10.0)),
        ]:
            m = cls(**params)
            m.fit(X_scaled, y_target)
            test_preds.append(m.predict(X_te_scaled))
        all_test.append(0.35 * test_preds[0] + 0.35 * test_preds[1] + 0.30 * test_preds[2])

    oof_avg = np.mean(all_oof, axis=0)
    test_avg = np.mean(all_test, axis=0)
    cv_mse = np.mean((oof_avg - y_target) ** 2)

    return cv_mse, test_avg, oof_avg


def strategy_global_pls(train_data, test_data, target_idx):
    """
    Strategy 5: PLS trained on ALL players' raw timeseries.
    Combines global data (345 samples) with PLS dimensionality reduction.
    """
    X_raw, y, pids = train_data['X_raw'], train_data['y'], train_data['pids']
    X_raw_te, pids_te = test_data['X_raw'], test_data['pids']

    p5_mask_tr = pids == PLAYER5_ID
    p5_mask_te = pids_te == PLAYER5_ID
    p5_idx_tr = np.where(p5_mask_tr)[0]

    y_target = y[:, target_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_te_scaled = scaler.transform(X_raw_te)

    # Find best components using P5-specific CV
    candidates = [2, 3, 5, 8, 10, 15, 20]
    best_nc, best_mse = 5, float('inf')

    for nc in candidates:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mses = []
        for tr_idx_local, val_idx_local in kf.split(np.arange(p5_mask_tr.sum())):
            # Train on non-P5 + P5 train fold
            non_p5_idx = np.where(~p5_mask_tr)[0]
            p5_tr_global = p5_idx_tr[tr_idx_local]
            train_idx = np.concatenate([non_p5_idx, p5_tr_global])
            val_idx = p5_idx_tr[val_idx_local]

            pls = PLSRegression(n_components=nc)
            pls.fit(X_scaled[train_idx], y_target[train_idx])
            pred = pls.predict(X_scaled[val_idx]).flatten()
            mses.append(np.mean((pred - y_target[val_idx]) ** 2))

        avg = np.mean(mses)
        if avg < best_mse:
            best_mse = avg
            best_nc = nc

    print(f"    Global PLS: best components = {best_nc}")

    # OOF
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_p5 = np.zeros(p5_mask_tr.sum())
    for tr_idx_local, val_idx_local in kf.split(np.arange(p5_mask_tr.sum())):
        non_p5_idx = np.where(~p5_mask_tr)[0]
        p5_tr_global = p5_idx_tr[tr_idx_local]
        train_idx = np.concatenate([non_p5_idx, p5_tr_global])
        val_idx = p5_idx_tr[val_idx_local]

        pls = PLSRegression(n_components=best_nc)
        pls.fit(X_scaled[train_idx], y_target[train_idx])
        oof_p5[val_idx_local] = pls.predict(X_scaled[val_idx]).flatten()

    cv_mse = np.mean((oof_p5 - y_target[p5_mask_tr]) ** 2)

    # Final
    pls = PLSRegression(n_components=best_nc)
    pls.fit(X_scaled, y_target)
    test_preds = pls.predict(X_te_scaled[p5_mask_te]).flatten()

    return cv_mse, test_preds, oof_p5


def scale_predictions(raw_preds, target):
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


def main():
    t0 = time.time()
    print("=" * 70)
    print("PLAYER 5 TARGETED MODELING")
    print("=" * 70)

    train_data, test_data = load_raw_data()

    # Get feature names from hoop-relative extraction
    kp_names = train_data['keypoint_names']
    sample_feats = extract_hoop_relative_features(train_data['X_3d'][0], kp_names, train_data['pids'][0])
    feat_names = sorted(sample_feats.keys())

    # Load Sub 784 (current best LB 0.007224)
    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")

    # Test each strategy for each target on Player 5
    print("\n" + "=" * 70)
    print("EVALUATING STRATEGIES FOR PLAYER 5")
    print("=" * 70)

    pids_te = test_data['pids']
    p5_mask_te = pids_te == PLAYER5_ID
    p5_test_ids = test_data['ids'][p5_mask_te]

    # Current baseline: what does Sub 784 have for Player 5?
    p5_sub784 = sub784[sub784['id'].isin(p5_test_ids)]
    print(f"\nPlayer 5 test samples: {p5_mask_te.sum()}")
    print(f"Sub 784 Player 5 predictions:")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={p5_sub784[col].mean():.4f} std={p5_sub784[col].std():.4f}")

    strategies = {
        'global_model': strategy_global_model,
        'heavy_ridge': strategy_heavy_ridge,
        'multi_seed': strategy_multi_seed_ensemble,
    }
    pls_strategies = {
        'pls_tuned': strategy_pls_tuned,
        'global_pls': strategy_global_pls,
    }

    results = {}  # (strategy, target) -> (cv_mse, test_preds)

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n--- TARGET: {target_name} ---")
        for sname, sfunc in strategies.items():
            cv_mse, test_preds, oof = sfunc(train_data, test_data, target_idx, feat_names)
            results[(sname, target_name)] = (cv_mse, test_preds)
            print(f"  {sname:20s}: CV MSE = {cv_mse:.4f}")

        for sname, sfunc in pls_strategies.items():
            cv_mse, test_preds, oof = sfunc(train_data, test_data, target_idx)
            results[(sname, target_name)] = (cv_mse, test_preds)
            print(f"  {sname:20s}: CV MSE = {cv_mse:.4f}")

    # Find best strategy per target for Player 5
    print("\n" + "=" * 70)
    print("BEST STRATEGY PER TARGET (Player 5)")
    print("=" * 70)

    best_per_target = {}
    all_strat_names = list(strategies.keys()) + list(pls_strategies.keys())
    for target_name in TARGETS:
        best_name, best_mse = None, float('inf')
        for sname in all_strat_names:
            cv_mse, _ = results[(sname, target_name)]
            if cv_mse < best_mse:
                best_mse = cv_mse
                best_name = sname
        best_per_target[target_name] = best_name
        print(f"  {target_name:12s}: {best_name:20s} (CV MSE = {best_mse:.4f})")

    # Create submissions: replace Player 5 predictions in Sub 784
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    target_to_col = {'angle': 'scaled_angle', 'depth': 'scaled_depth', 'left_right': 'scaled_left_right'}

    # Approach A: Replace Player 5 with best strategy for each target
    for blend_w in [0.30, 0.50, 0.70, 1.00]:
        sub = sub784.copy()
        desc_parts = []
        for target_name in TARGETS:
            best_sname = best_per_target[target_name]
            _, test_preds_raw = results[(best_sname, target_name)]
            test_preds_scaled = scale_predictions(test_preds_raw, target_name)
            col = target_to_col[target_name]

            # Blend: only for Player 5 rows
            p5_rows = sub['id'].isin(p5_test_ids)
            original = sub.loc[p5_rows, col].values
            sub.loc[p5_rows, col] = (1 - blend_w) * original + blend_w * test_preds_scaled
            desc_parts.append(f"{target_name}:{best_sname}")

        sub_num = get_next_submission_number()
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(filepath, index=False)
        print(f"  Sub {sub_num}: P5 blend_w={blend_w:.2f} best-per-target "
              f"({', '.join(desc_parts)}) "
              f"angle_std={sub['scaled_angle'].std():.6f} "
              f"depth_mean={sub['scaled_depth'].mean():.6f}")

    # Approach B: For each strategy, try replacing just depth or just left_right for Player 5
    for sname in all_strat_names:
        for target_name in ['depth', 'left_right']:
            _, test_preds_raw = results[(sname, target_name)]
            test_preds_scaled = scale_predictions(test_preds_raw, target_name)
            col = target_to_col[target_name]

            for blend_w in [0.50]:
                sub = sub784.copy()
                p5_rows = sub['id'].isin(p5_test_ids)
                original = sub.loc[p5_rows, col].values
                sub.loc[p5_rows, col] = (1 - blend_w) * original + blend_w * test_preds_scaled

                sub_num = get_next_submission_number()
                filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
                sub.to_csv(filepath, index=False)
                print(f"  Sub {sub_num}: P5 {target_name} {sname} blend=0.50 "
                      f"angle_std={sub['scaled_angle'].std():.6f} "
                      f"depth_mean={sub['scaled_depth'].mean():.6f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
