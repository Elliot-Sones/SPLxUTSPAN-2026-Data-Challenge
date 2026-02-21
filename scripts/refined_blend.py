"""
Refined Target-Specific Blending - Fine-tuning around Sub 784's winning config.

Sub 784 scored LB 0.007224 with: angle_w=0.00, depth_w=0.30, left_right_w=0.50
Key finding: left_right correction is the dominant lever.

This script explores:
- Higher left_right weights (0.50 to 1.00)
- Fine-tuned depth weights (0.10 to 0.50)
- Small angle corrections (0.00 to 0.15)
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


def get_next_submission_number():
    """Atomically get the next submission number using a lock file."""
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
# PLS DEPTH MODEL (same as target_specific_blend.py)
# ============================================================

def train_pls_depth(train_data):
    X_raw, y, pids = train_data['X_raw'], train_data['y'], train_data['pids']
    unique_pids = sorted(np.unique(pids))
    depth_idx = 1
    oof_depth = np.zeros(len(y))
    models, scalers = {}, {}

    print("\n--- PLS DEPTH MODEL ---")
    for pid in unique_pids:
        mask = pids == pid
        X_p, y_depth = X_raw[mask], y[mask, depth_idx]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        scalers[pid] = scaler

        candidates = [3, 5, 8, 10, 15, 20, 25, 30]
        max_comp = min(30, n - n // 5 - 1)
        candidates = [c for c in candidates if c <= max_comp]

        best_n, best_mse = 5, float('inf')
        for nc in candidates:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(X_scaled):
                pls = PLSRegression(n_components=nc)
                pls.fit(X_scaled[tr_idx], y_depth[tr_idx])
                pls_pred = pls.predict(X_scaled[val_idx]).flatten()
                X_tr_pls = pls.transform(X_scaled[tr_idx])
                X_val_pls = pls.transform(X_scaled[val_idx])
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_pls, y_depth[tr_idx])
                ridge_pred = ridge.predict(X_val_pls)
                lgb_m = lgb.LGBMRegressor(n_estimators=50, num_leaves=8, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(X_tr_pls, y_depth[tr_idx])
                lgb_pred = lgb_m.predict(X_val_pls)
                pred = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred
                mses.append(np.mean((pred - y_depth[val_idx]) ** 2))
            avg = np.mean(mses)
            if avg < best_mse:
                best_mse = avg
                best_n = nc

        print(f"  Player {pid}: best PLS components = {best_n}, CV MSE = {best_mse:.4f}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            pls = PLSRegression(n_components=best_n)
            pls.fit(X_scaled[tr_idx], y_depth[tr_idx])
            pls_pred = pls.predict(X_scaled[val_idx]).flatten()
            X_tr_pls = pls.transform(X_scaled[tr_idx])
            X_val_pls = pls.transform(X_scaled[val_idx])
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_tr_pls, y_depth[tr_idx])
            ridge_pred = ridge.predict(X_val_pls)
            lgb_m = lgb.LGBMRegressor(n_estimators=50, num_leaves=8, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
            lgb_m.fit(X_tr_pls, y_depth[tr_idx])
            lgb_pred = lgb_m.predict(X_val_pls)
            fold_preds[val_idx] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred
        oof_depth[global_idx] = fold_preds

        pls_final = PLSRegression(n_components=best_n)
        pls_final.fit(X_scaled, y_depth)
        X_pls_all = pls_final.transform(X_scaled)
        ridge_final = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        ridge_final.fit(X_pls_all, y_depth)
        lgb_final = lgb.LGBMRegressor(n_estimators=50, num_leaves=8, learning_rate=0.05,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
        lgb_final.fit(X_pls_all, y_depth)
        models[pid] = {'pls': pls_final, 'ridge': ridge_final, 'lgb': lgb_final}

    mse = np.mean((oof_depth - y[:, depth_idx]) ** 2)
    print(f"  Overall depth CV MSE: {mse:.6f}")
    return oof_depth, models, scalers


def predict_pls_depth(test_data, models, scalers):
    X_raw, pids = test_data['X_raw'], test_data['pids']
    preds = np.zeros(len(X_raw))
    for i, (x, pid) in enumerate(zip(X_raw, pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        pls = models[pid]['pls']
        pls_pred = pls.predict(x_scaled).flatten()[0]
        x_pls = pls.transform(x_scaled)
        ridge_pred = models[pid]['ridge'].predict(x_pls)[0]
        lgb_pred = models[pid]['lgb'].predict(x_pls)[0]
        preds[i] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred
    return preds


# ============================================================
# HOOP-RELATIVE MODEL (same as target_specific_blend.py)
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


def train_hoop_relative_model(train_data, target_idx):
    X_3d, y, pids = train_data['X_3d'], train_data['y'], train_data['pids']
    kp_names = train_data['keypoint_names']
    target_name = TARGETS[target_idx]
    unique_pids = sorted(np.unique(pids))

    print(f"\n--- HOOP-RELATIVE {target_name.upper()} MODEL ---")

    all_feats = [extract_hoop_relative_features(X_3d[i], kp_names, pids[i]) for i in range(len(X_3d))]
    feat_names = sorted(all_feats[0].keys())
    X_feat = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats], dtype=np.float32)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    oof_preds = np.zeros(len(y))
    models, scalers = {}, {}

    for pid in unique_pids:
        mask = pids == pid
        X_p, y_t = X_feat[mask], y[mask, target_idx]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)
        scalers[pid] = scaler

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr = y_t[tr_idx]
            preds = []
            for cls, params in [
                (lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
                (xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
                (CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)),
                (Ridge, dict(alpha=1.0, random_state=42)),
            ]:
                m = cls(**params); m.fit(X_tr, y_tr)
                preds.append(m.predict(X_val))
            fold_preds[val_idx] = 0.3*preds[0] + 0.3*preds[1] + 0.3*preds[2] + 0.1*preds[3]
        oof_preds[global_idx] = fold_preds
        print(f"  Player {pid}: CV MSE = {np.mean((fold_preds - y_t)**2):.4f}")

        for name, cls, params in [
            ('lgb', lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
            ('cat', CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False)),
            ('ridge', Ridge, dict(alpha=1.0, random_state=42)),
        ]:
            m = cls(**params); m.fit(X_scaled, y_t)
            models[(pid, name)] = m

    print(f"  Overall {target_name} CV MSE: {np.mean((oof_preds - y[:, target_idx])**2):.6f}")
    return oof_preds, models, scalers, feat_names


def predict_hoop_relative(test_data, models, scalers, feat_names):
    X_3d, pids, kp_names = test_data['X_3d'], test_data['pids'], test_data['keypoint_names']
    all_feats = [extract_hoop_relative_features(X_3d[i], kp_names, pids[i]) for i in range(len(X_3d))]
    X_feat = np.array([[f.get(name, 0.0) for name in feat_names] for f in all_feats], dtype=np.float32)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    preds = np.zeros(len(X_feat))
    for i, (x, pid) in enumerate(zip(X_feat, pids)):
        x_scaled = np.nan_to_num(scalers[pid].transform(x.reshape(1, -1)), nan=0.0)
        p = [models[(pid, n)].predict(x_scaled)[0] for n in ['lgb', 'xgb', 'cat', 'ridge']]
        preds[i] = 0.3*p[0] + 0.3*p[1] + 0.3*p[2] + 0.1*p[3]
    return preds


# ============================================================
# REFINED BLENDING
# ============================================================

def scale_predictions(raw_preds, target):
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


def refined_blend(test_ids, pls_depth_preds, hr_lr_preds, hr_angle_preds):
    """Refined grid search focused around Sub 784's winning config."""
    sub771 = pd.read_csv(SUBMISSION_DIR / "submission_771.csv")
    pls_depth_scaled = scale_predictions(pls_depth_preds, 'depth')
    hr_lr_scaled = scale_predictions(hr_lr_preds, 'left_right')
    hr_angle_scaled = scale_predictions(hr_angle_preds, 'angle')

    our = pd.DataFrame({'id': test_ids, 'new_depth': pls_depth_scaled,
                         'new_lr': hr_lr_scaled, 'new_angle': hr_angle_scaled})
    merged = sub771.merge(our, on='id')

    print("\n" + "=" * 70)
    print("REFINED BLENDING - focused around Sub 784 winning config")
    print("=" * 70)
    print(f"Sub 784 config: angle_w=0.00, depth_w=0.30, left_right_w=0.50 -> LB 0.007224")
    print()

    for col, nc in [('scaled_angle', 'new_angle'), ('scaled_depth', 'new_depth'),
                     ('scaled_left_right', 'new_lr')]:
        print(f"  {col} correlation with new: {np.corrcoef(merged[col], merged[nc])[0,1]:.4f}")

    # Fine grid around winning config
    # Key insight: left_right at 0.50 was good, try higher
    angle_weights = [0.00, 0.02, 0.05, 0.08, 0.10]
    depth_weights = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    lr_weights = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    print(f"\n  Grid: {len(angle_weights)} angle x {len(depth_weights)} depth x {len(lr_weights)} lr "
          f"= {len(angle_weights)*len(depth_weights)*len(lr_weights)} configs")

    results = []
    for aw in angle_weights:
        for dw in depth_weights:
            for lw in lr_weights:
                ba = (1-aw)*merged['scaled_angle'] + aw*merged['new_angle']
                bd = (1-dw)*merged['scaled_depth'] + dw*merged['new_depth']
                bl = (1-lw)*merged['scaled_left_right'] + lw*merged['new_lr']
                results.append({
                    'aw': aw, 'dw': dw, 'lw': lw,
                    'angle_std': ba.std(), 'depth_mean': bd.mean(),
                    'blend_angle': ba.values, 'blend_depth': bd.values, 'blend_lr': bl.values,
                })

    # Compute diversity from Sub 771
    for r in results:
        da = np.mean((r['blend_angle'] - merged['scaled_angle'].values)**2)
        dd = np.mean((r['blend_depth'] - merged['scaled_depth'].values)**2)
        dl = np.mean((r['blend_lr'] - merged['scaled_left_right'].values)**2)
        r['diversity'] = da + dd + dl
        r['div_angle'] = da
        r['div_depth'] = dd
        r['div_lr'] = dl

    # Relaxed profile constraints (Sub 771 itself has angle_std=0.146, depth_mean=0.514)
    valid = [r for r in results if r['angle_std'] < 0.16 and 0.48 < r['depth_mean'] < 0.53]
    valid.sort(key=lambda x: -x['diversity'])

    print(f"\n  Valid configs: {len(valid)}")
    print(f"\n  {'aw':>5} {'dw':>5} {'lw':>5} | {'a_std':>8} {'d_mean':>8} | {'div_a':>8} {'div_d':>8} {'div_lr':>8} | {'total':>8}")
    print(f"  " + "-" * 85)

    for r in valid[:25]:
        print(f"  {r['aw']:>5.2f} {r['dw']:>5.2f} {r['lw']:>5.2f} | "
              f"{r['angle_std']:>8.4f} {r['depth_mean']:>8.4f} | "
              f"{r['div_angle']:>8.6f} {r['div_depth']:>8.6f} {r['div_lr']:>8.6f} | "
              f"{r['diversity']:>8.6f}")

    return merged, valid


def main():
    t0 = time.time()
    print("=" * 70)
    print("REFINED BLEND EXPERIMENT")
    print("=" * 70)

    train_data, test_data = load_raw_data()

    # Train models
    oof_depth, pls_models, pls_scalers = train_pls_depth(train_data)
    oof_lr, hr_lr_models, hr_lr_scalers, hr_feat_names = train_hoop_relative_model(train_data, target_idx=2)
    oof_angle, hr_angle_models, hr_angle_scalers, _ = train_hoop_relative_model(train_data, target_idx=0)

    # Test predictions
    print("\nGenerating test predictions...")
    pls_depth_test = predict_pls_depth(test_data, pls_models, pls_scalers)
    hr_lr_test = predict_hoop_relative(test_data, hr_lr_models, hr_lr_scalers, hr_feat_names)
    hr_angle_test = predict_hoop_relative(test_data, hr_angle_models, hr_angle_scalers, hr_feat_names)

    # Refined blend
    merged, top_configs = refined_blend(
        test_data['ids'], pls_depth_test, hr_lr_test, hr_angle_test)

    # Save a spread of submissions with varying diversity levels
    print("\n" + "=" * 70)
    print("SAVING SUBMISSIONS")
    print("=" * 70)

    # Select diverse set of configs to submit
    # 1. Most aggressive (highest lr weight that's still valid)
    # 2. Sub 784 clone (aw=0.00, dw=0.30, lw=0.50) for sanity check
    # 3. A few in between
    configs_to_save = []

    # Find configs at different lr weight levels
    target_lrs = [1.00, 0.90, 0.80, 0.70, 0.60, 0.55]
    for target_lw in target_lrs:
        # Find best config at this lr weight (prefer dw=0.30, aw=0.00)
        matches = [r for r in top_configs if abs(r['lw'] - target_lw) < 0.01]
        if matches:
            # Sort by closeness to dw=0.30 and aw=0.00
            matches.sort(key=lambda x: abs(x['dw'] - 0.30) + abs(x['aw']))
            configs_to_save.append(matches[0])

    for config in configs_to_save:
        sub_num = get_next_submission_number()
        sub = pd.DataFrame({
            'id': merged['id'],
            'scaled_angle': config['blend_angle'],
            'scaled_depth': config['blend_depth'],
            'scaled_left_right': config['blend_lr'],
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(filepath, index=False)
        print(f"  Sub {sub_num}: aw={config['aw']:.2f} dw={config['dw']:.2f} lw={config['lw']:.2f} "
              f"angle_std={config['angle_std']:.6f} depth_mean={config['depth_mean']:.6f} "
              f"div={config['diversity']:.8f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
