"""
Hoop-Integrated Ensemble

Goal: Integrate hoop-relative features directly into the full ensemble pipeline
(concatenated with standard features), then blend with Sub 784 (LB 0.007224).

Approach:
1. Two feature sets concatenated:
   a. Standard: per-keypoint stats (mean, std, min, max, range) + release frame (153) values
      + velocity stats over full timeseries. ~207*7 = ~1400 features.
   b. Hoop-relative: transform 14 key joints into hoop-relative coords (forward/lateral/vertical),
      extract stats in both original and hoop-relative coords, plus body alignment features.
      ~500+ features.
2. Per-player per-target 5-fold CV with LGB/XGB/CatBoost/Ridge ensemble (0.3/0.3/0.3/0.1).
3. Blend with Sub 784 at multiple weights.
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
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])

KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist',
    'left_shoulder', 'right_hip', 'left_hip', 'mid_hip',
    'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'neck', 'nose'
]


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
    """Load train and test, returning raw timeseries + metadata."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)

    keypoint_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoint_names.append(col[:-2])

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Keypoint cols: {n_kp_cols}, Keypoints: {len(keypoint_names)}")

    def process(df, is_train=True):
        n = len(df)
        # 3D timeseries: (n_samples, 240, n_keypoints, 3)
        X_3d = np.zeros((n, 240, len(keypoint_names), 3), dtype=np.float32)
        # Also keep the flat 2D timeseries: (n_samples, 240, n_kp_cols)
        X_ts = np.zeros((n, 240, n_kp_cols), dtype=np.float32)

        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_ts[idx, :, col_i] = arr
                kp_idx = col_i // 3
                coord_idx = col_i % 3
                X_3d[idx, :, kp_idx, coord_idx] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        result = {
            'X_ts': X_ts,
            'X_3d': X_3d,
            'pids': np.array(pids),
            'ids': np.array(ids),
            'keypoint_names': keypoint_names,
            'keypoint_cols': keypoint_cols,
        }
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


# ============================================================
# STANDARD FEATURES
# ============================================================

def extract_standard_features(X_ts):
    """
    Per-keypoint statistics over the full timeseries.
    X_ts shape: (n_samples, 240, n_kp_cols)
    Returns: (n_samples, n_features)

    Features per column: mean, std, min, max, range, release_frame_value, velocity_max
    = 7 features per column -> 207 * 7 = 1449 features
    """
    n = X_ts.shape[0]
    n_cols = X_ts.shape[2]

    # Basic stats
    feat_mean = np.nanmean(X_ts, axis=1)           # (n, n_cols)
    feat_std = np.nanstd(X_ts, axis=1)              # (n, n_cols)
    feat_min = np.nanmin(X_ts, axis=1)              # (n, n_cols)
    feat_max = np.nanmax(X_ts, axis=1)              # (n, n_cols)
    feat_range = feat_max - feat_min                 # (n, n_cols)

    # Release frame value (frame 153)
    feat_release = X_ts[:, 153, :]                   # (n, n_cols)

    # Velocity stats: max of absolute velocity
    velocity = np.diff(X_ts, axis=1) * 60.0          # (n, 239, n_cols)
    feat_vel_max = np.nanmax(np.abs(velocity), axis=1)  # (n, n_cols)

    features = np.concatenate([
        feat_mean, feat_std, feat_min, feat_max, feat_range,
        feat_release, feat_vel_max
    ], axis=1)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


# ============================================================
# HOOP-RELATIVE FEATURES
# ============================================================

def compute_hoop_relative_transform(player_pos):
    """Compute rotation matrix to transform into hoop-relative coordinates."""
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


def extract_hoop_relative_features_single(ts_3d, keypoint_names):
    """
    Extract hoop-relative features for a single sample.
    ts_3d shape: (240, n_keypoints, 3)
    Returns: dict of feature_name -> value
    """
    feats = {}
    kp_index = {name: i for i, name in enumerate(keypoint_names)}

    # Get player position from mid_hip average of first 10 frames
    mh_idx = kp_index.get('mid_hip')
    if mh_idx is not None:
        player_pos = np.nanmean(ts_3d[:10, mh_idx, :], axis=0)
    else:
        player_pos = np.nanmean(ts_3d[:10, :, :].mean(axis=1), axis=0)

    R, origin = compute_hoop_relative_transform(player_pos)
    centered = ts_3d - origin.reshape(1, 1, 3)
    ts_hoop = np.einsum('ij,fkj->fki', R, centered)

    # Per-joint features in hoop-relative and original coordinates
    for joint in KEY_JOINTS:
        if joint not in kp_index:
            continue
        idx = kp_index[joint]

        # Hoop-relative coordinates
        for coord, cname in enumerate(['forward', 'lateral', 'vertical']):
            s = ts_hoop[:, idx, coord]
            prefix = f"hr_{joint}_{cname}"
            feats[f"{prefix}_mean"] = np.nanmean(s)
            feats[f"{prefix}_std"] = np.nanstd(s)
            feats[f"{prefix}_min"] = np.nanmin(s)
            feats[f"{prefix}_max"] = np.nanmax(s)
            feats[f"{prefix}_range"] = np.nanmax(s) - np.nanmin(s)
            # Release window mean (frames 140-180)
            feats[f"{prefix}_release_mean"] = np.nanmean(s[140:180])
            # Velocity stats
            vel = np.gradient(s, 1.0 / 60.0)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)
            feats[f"{prefix}_vel_at_release"] = vel[153] if len(vel) > 153 else 0.0

        # Original coordinates
        for c, cname in enumerate(['x', 'y', 'z']):
            s = ts_3d[:, idx, c]
            prefix = f"orig_{joint}_{cname}"
            feats[f"{prefix}_mean"] = np.nanmean(s)
            feats[f"{prefix}_std"] = np.nanstd(s)
            feats[f"{prefix}_min"] = np.nanmin(s)
            feats[f"{prefix}_max"] = np.nanmax(s)
            feats[f"{prefix}_range"] = np.nanmax(s) - np.nanmin(s)
            # Release frame value
            feats[f"{prefix}_f153"] = s[153]
            # Velocity
            vel = np.gradient(s, 1.0 / 60.0)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)

    # Body alignment features
    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    rs, ls = kp_index.get('right_shoulder'), kp_index.get('left_shoulder')
    rw, lw = kp_index.get('right_wrist'), kp_index.get('left_wrist')

    if rh is not None and lh is not None:
        hip_lat = ts_hoop[:, rh, 1] - ts_hoop[:, lh, 1]
        feats['hr_hip_alignment_mean'] = np.nanmean(hip_lat)
        feats['hr_hip_alignment_std'] = np.nanstd(hip_lat)
        feats['hr_hip_alignment_release'] = hip_lat[153]

    if rs is not None and ls is not None:
        shoulder_lat = ts_hoop[:, rs, 1] - ts_hoop[:, ls, 1]
        feats['hr_shoulder_alignment_mean'] = np.nanmean(shoulder_lat)
        feats['hr_shoulder_alignment_std'] = np.nanstd(shoulder_lat)
        feats['hr_shoulder_alignment_release'] = shoulder_lat[153]

    if rw is not None and lw is not None:
        guide_lat = ts_hoop[:, lw, 1] - ts_hoop[:, rw, 1]
        feats['hr_guide_hand_lateral_mean'] = np.nanmean(guide_lat)
        feats['hr_guide_hand_lateral_release'] = guide_lat[153]

    if rw is not None and rs is not None:
        arm_lat = ts_hoop[:, rw, 1] - ts_hoop[:, rs, 1]
        feats['hr_arm_lateral_dev_mean'] = np.nanmean(arm_lat)
        feats['hr_arm_lateral_dev_release'] = arm_lat[153]

    # Elbow angle
    for j1, j2, j3, name in [('right_shoulder', 'right_elbow', 'right_wrist', 'elbow')]:
        if all(j in kp_index for j in [j1, j2, j3]):
            p1 = ts_3d[:, kp_index[j1]]
            p2 = ts_3d[:, kp_index[j2]]
            p3 = ts_3d[:, kp_index[j3]]
            v1, v2 = p1 - p2, p3 - p2
            dot = np.sum(v1 * v2, axis=1)
            n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
            denom = n1 * n2
            denom[denom == 0] = 1e-10
            angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            feats[f"{name}_angle_release"] = angle[153]
            feats[f"{name}_angle_range"] = np.nanmax(angle) - np.nanmin(angle)
            feats[f"{name}_angle_mean"] = np.nanmean(angle)

    # Phase velocity features
    for pname, (s, e) in [('load', (60, 120)), ('propel', (120, 170))]:
        for joint in ['right_wrist', 'right_elbow']:
            if joint not in kp_index:
                continue
            idx2 = kp_index[joint]
            for c in range(3):
                vel = np.gradient(ts_3d[s:e, idx2, c], 1.0 / 60.0)
                feats[f"phase_{pname}_{joint}_{'xyz'[c]}_vel_max"] = np.nanmax(vel)
                feats[f"phase_{pname}_{joint}_{'xyz'[c]}_vel_mean"] = np.nanmean(vel)

    return feats


def extract_hoop_relative_features_batch(X_3d, keypoint_names):
    """Extract hoop-relative features for all samples. Returns (X_feat, feat_names)."""
    all_feats = []
    for i in range(len(X_3d)):
        all_feats.append(extract_hoop_relative_features_single(X_3d[i], keypoint_names))

    feat_names = sorted(all_feats[0].keys())
    X_feat = np.array(
        [[f.get(name, 0.0) for name in feat_names] for f in all_feats],
        dtype=np.float32
    )
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_feat, feat_names


# ============================================================
# COMBINED FEATURES + TRAINING
# ============================================================

def build_combined_features(data, hoop_feat_names=None):
    """
    Build combined feature matrix from standard + hoop-relative features.
    Returns (X_combined, hoop_feat_names).
    """
    print("  Extracting standard features...")
    X_std = extract_standard_features(data['X_ts'])
    print(f"    Standard features: {X_std.shape[1]}")

    print("  Extracting hoop-relative features...")
    X_hoop, feat_names = extract_hoop_relative_features_batch(data['X_3d'], data['keypoint_names'])
    print(f"    Hoop-relative features: {X_hoop.shape[1]}")

    if hoop_feat_names is not None:
        # Ensure same feature order as training
        X_hoop_aligned = np.zeros((len(X_hoop), len(hoop_feat_names)), dtype=np.float32)
        current_map = {name: i for i, name in enumerate(feat_names)}
        for j, name in enumerate(hoop_feat_names):
            if name in current_map:
                X_hoop_aligned[:, j] = X_hoop[:, current_map[name]]
        X_hoop = X_hoop_aligned
        feat_names = hoop_feat_names

    X_combined = np.concatenate([X_std, X_hoop], axis=1)
    print(f"    Combined features: {X_combined.shape[1]}")
    return X_combined, feat_names


def train_per_player_per_target(X, y, pids, target_idx, target_name):
    """
    Train per-player per-target ensemble with 5-fold CV.
    Returns: oof_preds, models_dict, scalers_dict
    """
    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros(len(y))
    models = {}
    scalers = {}

    print(f"\n--- {target_name.upper()} ---")

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_t = y[mask, target_idx]
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

            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1
            )
            lgb_model.fit(X_tr, y_tr)
            preds.append(lgb_model.predict(X_val))

            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1
            )
            xgb_model.fit(X_tr, y_tr)
            preds.append(xgb_model.predict(X_val))

            # CatBoost
            cat_model = CatBoostRegressor(
                iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False
            )
            cat_model.fit(X_tr, y_tr)
            preds.append(cat_model.predict(X_val))

            # Ridge
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_tr, y_tr)
            preds.append(ridge_model.predict(X_val))

            # Ensemble: 0.3*LGB + 0.3*XGB + 0.3*Cat + 0.1*Ridge
            fold_preds[val_idx] = (
                0.3 * preds[0] + 0.3 * preds[1] + 0.3 * preds[2] + 0.1 * preds[3]
            )

        oof_preds[global_idx] = fold_preds
        cv_mse = np.mean((fold_preds - y_t) ** 2)
        print(f"  Player {pid}: n={n}, CV MSE = {cv_mse:.6f}")

        # Train final models on all player data
        pid_models = {}
        for name, cls, params in [
            ('lgb', lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
            ('cat', CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False)),
            ('ridge', Ridge, dict(alpha=1.0)),
        ]:
            m = cls(**params)
            m.fit(X_scaled, y_t)
            pid_models[name] = m

        models[pid] = pid_models

    overall_mse = np.mean((oof_preds - y[:, target_idx]) ** 2)
    print(f"  Overall {target_name} CV MSE: {overall_mse:.6f}")
    return oof_preds, models, scalers, overall_mse


def predict_test(X_test, pids_test, models, scalers):
    """Generate test predictions."""
    preds = np.zeros(len(X_test))
    for i in range(len(X_test)):
        pid = pids_test[i]
        x = X_test[i:i+1]
        x_scaled = np.nan_to_num(scalers[pid].transform(x), nan=0.0)
        p = [models[pid][n].predict(x_scaled)[0] for n in ['lgb', 'xgb', 'cat', 'ridge']]
        preds[i] = 0.3 * p[0] + 0.3 * p[1] + 0.3 * p[2] + 0.1 * p[3]
    return preds


def scale_predictions(raw_preds, target):
    """Scale raw predictions to [0,1] using the target scaler."""
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("HOOP-INTEGRATED ENSEMBLE")
    print("=" * 70)
    print()

    # Load data
    train_data, test_data = load_raw_data()

    # Build combined features
    print("\nBuilding combined features (train)...")
    X_train, hoop_feat_names = build_combined_features(train_data)

    print("\nBuilding combined features (test)...")
    X_test, _ = build_combined_features(test_data, hoop_feat_names=hoop_feat_names)

    y = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    test_ids = test_data['ids']

    print(f"\nFinal feature shapes: train={X_train.shape}, test={X_test.shape}")

    # Train per-player per-target
    print("\n" + "=" * 70)
    print("TRAINING PER-PLAYER PER-TARGET MODELS")
    print("=" * 70)

    results = {}
    all_oof = {}
    all_models = {}
    all_scalers = {}

    for t_idx, target in enumerate(TARGETS):
        oof, models, scalers, cv_mse = train_per_player_per_target(
            X_train, y, pids_train, t_idx, target
        )
        all_oof[target] = oof
        all_models[target] = models
        all_scalers[target] = scalers
        results[target] = cv_mse

    # Overall CV
    total_cv = np.mean([results[t] for t in TARGETS])
    print(f"\n{'=' * 70}")
    print(f"OVERALL CV RESULTS")
    print(f"{'=' * 70}")
    for t in TARGETS:
        print(f"  {t}: CV MSE = {results[t]:.6f}")
    print(f"  Mean CV MSE: {total_cv:.6f}")

    # Per-player per-target breakdown
    print(f"\nPer-player per-target CV MSE:")
    unique_pids = sorted(np.unique(pids_train))
    header = f"  {'Player':>8}"
    for t in TARGETS:
        header += f"  {t:>12}"
    header += f"  {'mean':>12}"
    print(header)
    for pid in unique_pids:
        mask = pids_train == pid
        line = f"  {pid:>8}"
        pid_mses = []
        for t_idx, t in enumerate(TARGETS):
            mse = np.mean((all_oof[t][mask] - y[mask, t_idx]) ** 2)
            pid_mses.append(mse)
            line += f"  {mse:>12.6f}"
        line += f"  {np.mean(pid_mses):>12.6f}"
        print(line)

    # Generate test predictions
    print(f"\n{'=' * 70}")
    print("GENERATING TEST PREDICTIONS")
    print(f"{'=' * 70}")

    raw_preds = {}
    scaled_preds = {}
    for target in TARGETS:
        raw = predict_test(X_test, pids_test, all_models[target], all_scalers[target])
        raw_preds[target] = raw
        scaled = scale_predictions(raw, target)
        scaled_preds[target] = scaled
        print(f"  {target}: raw range [{raw.min():.4f}, {raw.max():.4f}] -> scaled [{scaled.min():.4f}, {scaled.max():.4f}]")

    # Blend with Sub 784
    print(f"\n{'=' * 70}")
    print("BLENDING WITH SUB 784 (LB 0.007224)")
    print(f"{'=' * 70}")

    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    our = pd.DataFrame({
        'id': test_ids,
        'new_angle': scaled_preds['angle'],
        'new_depth': scaled_preds['depth'],
        'new_left_right': scaled_preds['left_right'],
    })
    merged = sub784.merge(our, on='id')

    # Correlations
    print("\nCorrelations with Sub 784:")
    for col, nc in [('scaled_angle', 'new_angle'), ('scaled_depth', 'new_depth'),
                     ('scaled_left_right', 'new_left_right')]:
        r = np.corrcoef(merged[col], merged[nc])[0, 1]
        print(f"  {col}: r = {r:.4f}")

    # Blend at multiple weights
    blend_weights = [0.10, 0.20, 0.30, 0.50]
    submissions = []

    print(f"\n  {'w':>5} | {'angle_std':>10} {'depth_mean':>10} | {'div_angle':>10} {'div_depth':>10} {'div_lr':>10} | {'total_div':>10}")
    print(f"  " + "-" * 80)

    for w in blend_weights:
        ba = (1 - w) * merged['scaled_angle'] + w * merged['new_angle']
        bd = (1 - w) * merged['scaled_depth'] + w * merged['new_depth']
        bl = (1 - w) * merged['scaled_left_right'] + w * merged['new_left_right']

        da = np.mean((ba - merged['scaled_angle'].values) ** 2)
        dd = np.mean((bd - merged['scaled_depth'].values) ** 2)
        dl = np.mean((bl - merged['scaled_left_right'].values) ** 2)
        div_total = da + dd + dl

        print(f"  {w:>5.2f} | {ba.std():>10.6f} {bd.mean():>10.6f} | {da:>10.6f} {dd:>10.6f} {dl:>10.6f} | {div_total:>10.6f}")

        submissions.append({
            'w': w,
            'angle': ba.values,
            'depth': bd.values,
            'left_right': bl.values,
            'angle_std': ba.std(),
            'depth_mean': bd.mean(),
            'diversity': div_total,
        })

    # Save submissions
    print(f"\n{'=' * 70}")
    print("SAVING SUBMISSIONS")
    print(f"{'=' * 70}")

    for s in submissions:
        sub_num = get_next_submission_number()
        sub_df = pd.DataFrame({
            'id': merged['id'],
            'scaled_angle': s['angle'],
            'scaled_depth': s['depth'],
            'scaled_left_right': s['left_right'],
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(filepath, index=False)
        print(f"  Sub {sub_num}: w={s['w']:.2f}, angle_std={s['angle_std']:.6f}, "
              f"depth_mean={s['depth_mean']:.6f}, diversity={s['diversity']:.6f}")
        print(f"    -> {filepath}")
        print(f"    Model: Hoop-integrated ensemble (standard + hoop-relative features concatenated)")
        print(f"    Config: Per-player per-target, 5-fold CV, LGB/XGB/Cat/Ridge (0.3/0.3/0.3/0.1)")
        print(f"    Features: {X_train.shape[1]} combined (standard + hoop-relative)")
        print(f"    Blend: (1-{s['w']:.2f})*sub784 + {s['w']:.2f}*new")
        print(f"    CV MSE: angle={results['angle']:.6f}, depth={results['depth']:.6f}, lr={results['left_right']:.6f}")
        print()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
