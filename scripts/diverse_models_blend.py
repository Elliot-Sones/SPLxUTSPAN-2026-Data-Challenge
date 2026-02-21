"""
Diverse Models Blend - Create predictions using different feature types and model types,
then blend with Sub 784 (current best LB 0.007224).

Feature diversity:
  a. FFT features: top-k frequency magnitudes from each keypoint timeseries
  b. Window-based stats: early/loading/release/follow-through phase statistics
  c. Velocity/acceleration profiles: derivatives of key joints across phases

Model diversity:
  a. SVR with RBF kernel
  b. KNN regressor
  c. ElasticNet
  d. Standard tree ensemble (LGB+XGB+CatBoost+Ridge) with the NEW features

Per-player per-target 5-fold CV, blend with Sub 784 at multiple weights.
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
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])

# Key joints for velocity/acceleration features
KEY_JOINTS = [
    'right_wrist', 'left_wrist',
    'right_elbow', 'left_elbow',
    'right_shoulder', 'left_shoulder',
]

# Frame windows
WINDOWS = {
    'early': (0, 80),
    'loading': (60, 140),
    'release': (130, 180),
    'followthrough': (170, 240),
}

# Phase definitions for velocity profiles
VEL_PHASES = {
    'early': (0, 80),
    'loading': (60, 140),
    'release': (130, 180),
    'followthrough': (170, 240),
    'full': (0, 240),
}


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


def load_data():
    """Load train and test data, return 3D arrays and metadata."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    keypoint_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoint_names.append(col[:-2])

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Keypoints: {len(keypoint_names)}")
    print(f"  Keypoint cols: {len(keypoint_cols)}")

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(keypoint_names)
        # X_3d shape: (n_shots, 240 frames, n_keypoints, 3 coords)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx in range(n):
            row = df.iloc[idx]
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                kp_idx = col_i // 3
                coord_idx = col_i % 3
                X_3d[idx, :, kp_idx, coord_idx] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx + 1}/{n}")

        X_3d = np.nan_to_num(X_3d, nan=0.0, posinf=0.0, neginf=0.0)
        result = {
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
# FEATURE EXTRACTION
# ============================================================

def extract_fft_features(X_3d, keypoint_names, top_k_values=(5, 10)):
    """Extract FFT features: top-k frequency magnitudes per keypoint timeseries."""
    n_shots, n_frames, n_kp, n_coords = X_3d.shape
    features = []

    for shot_idx in range(n_shots):
        feats = {}
        for kp_idx, kp_name in enumerate(keypoint_names):
            for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
                signal = X_3d[shot_idx, :, kp_idx, coord_idx]
                # Apply FFT
                fft_vals = np.fft.rfft(signal)
                magnitudes = np.abs(fft_vals)
                # Skip DC component (index 0)
                magnitudes_no_dc = magnitudes[1:]

                for top_k in top_k_values:
                    # Get top-k magnitudes
                    if len(magnitudes_no_dc) >= top_k:
                        top_indices = np.argsort(magnitudes_no_dc)[-top_k:]
                        top_mags = magnitudes_no_dc[top_indices]
                    else:
                        top_mags = magnitudes_no_dc

                    prefix = f"fft_{kp_name}_{coord_name}_k{top_k}"
                    feats[f"{prefix}_sum"] = np.sum(top_mags)
                    feats[f"{prefix}_mean"] = np.mean(top_mags)
                    feats[f"{prefix}_max"] = np.max(top_mags) if len(top_mags) > 0 else 0.0
                    # Spectral entropy
                    if np.sum(magnitudes_no_dc) > 0:
                        p = magnitudes_no_dc / np.sum(magnitudes_no_dc)
                        p = p[p > 0]
                        feats[f"fft_{kp_name}_{coord_name}_entropy"] = -np.sum(p * np.log(p))
                    else:
                        feats[f"fft_{kp_name}_{coord_name}_entropy"] = 0.0

        features.append(feats)
    return features


def extract_window_features(X_3d, keypoint_names):
    """Extract statistical features per time window per keypoint."""
    n_shots, n_frames, n_kp, n_coords = X_3d.shape
    features = []

    for shot_idx in range(n_shots):
        feats = {}
        for kp_idx, kp_name in enumerate(keypoint_names):
            for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
                signal = X_3d[shot_idx, :, kp_idx, coord_idx]

                for win_name, (start, end) in WINDOWS.items():
                    segment = signal[start:end]
                    prefix = f"win_{win_name}_{kp_name}_{coord_name}"
                    feats[f"{prefix}_mean"] = np.mean(segment)
                    feats[f"{prefix}_std"] = np.std(segment)
                    feats[f"{prefix}_min"] = np.min(segment)
                    feats[f"{prefix}_max"] = np.max(segment)
                    feats[f"{prefix}_range"] = np.max(segment) - np.min(segment)
                    feats[f"{prefix}_median"] = np.median(segment)
                    # Slope (linear trend)
                    if len(segment) > 1:
                        t = np.arange(len(segment), dtype=np.float32)
                        feats[f"{prefix}_slope"] = np.polyfit(t, segment, 1)[0]
                    else:
                        feats[f"{prefix}_slope"] = 0.0

                # Cross-window differences (release vs loading)
                load_seg = signal[WINDOWS['loading'][0]:WINDOWS['loading'][1]]
                rel_seg = signal[WINDOWS['release'][0]:WINDOWS['release'][1]]
                prefix_diff = f"windiff_{kp_name}_{coord_name}"
                feats[f"{prefix_diff}_release_minus_load_mean"] = np.mean(rel_seg) - np.mean(load_seg)
                feats[f"{prefix_diff}_release_minus_load_std"] = np.std(rel_seg) - np.std(load_seg)

        features.append(feats)
    return features


def extract_velocity_features(X_3d, keypoint_names):
    """Extract velocity and acceleration profiles of key joints across phases."""
    n_shots = X_3d.shape[0]
    kp_index = {name: i for i, name in enumerate(keypoint_names)}
    dt = 1.0 / 60.0  # 60 fps
    features = []

    for shot_idx in range(n_shots):
        feats = {}
        for joint in KEY_JOINTS:
            if joint not in kp_index:
                continue
            jidx = kp_index[joint]

            for coord_idx, coord_name in enumerate(['x', 'y', 'z']):
                signal = X_3d[shot_idx, :, jidx, coord_idx]
                velocity = np.gradient(signal, dt)
                acceleration = np.gradient(velocity, dt)

                for phase_name, (start, end) in VEL_PHASES.items():
                    vel_seg = velocity[start:end]
                    acc_seg = acceleration[start:end]

                    vprefix = f"vel_{joint}_{coord_name}_{phase_name}"
                    feats[f"{vprefix}_mean"] = np.mean(vel_seg)
                    feats[f"{vprefix}_std"] = np.std(vel_seg)
                    feats[f"{vprefix}_max"] = np.max(vel_seg)
                    feats[f"{vprefix}_min"] = np.min(vel_seg)
                    feats[f"{vprefix}_absmax"] = np.max(np.abs(vel_seg))
                    feats[f"{vprefix}_rms"] = np.sqrt(np.mean(vel_seg ** 2))

                    aprefix = f"acc_{joint}_{coord_name}_{phase_name}"
                    feats[f"{aprefix}_mean"] = np.mean(acc_seg)
                    feats[f"{aprefix}_std"] = np.std(acc_seg)
                    feats[f"{aprefix}_max"] = np.max(acc_seg)
                    feats[f"{aprefix}_min"] = np.min(acc_seg)
                    feats[f"{aprefix}_absmax"] = np.max(np.abs(acc_seg))

            # 3D speed and acceleration magnitude
            pos_3d = X_3d[shot_idx, :, jidx, :]  # (240, 3)
            speed_3d = np.linalg.norm(np.gradient(pos_3d, dt, axis=0), axis=1)
            acc_3d = np.linalg.norm(np.gradient(np.gradient(pos_3d, dt, axis=0), dt, axis=0), axis=1)

            for phase_name, (start, end) in VEL_PHASES.items():
                sp_seg = speed_3d[start:end]
                ac_seg = acc_3d[start:end]
                feats[f"speed3d_{joint}_{phase_name}_mean"] = np.mean(sp_seg)
                feats[f"speed3d_{joint}_{phase_name}_max"] = np.max(sp_seg)
                feats[f"speed3d_{joint}_{phase_name}_std"] = np.std(sp_seg)
                feats[f"acc3d_{joint}_{phase_name}_mean"] = np.mean(ac_seg)
                feats[f"acc3d_{joint}_{phase_name}_max"] = np.max(ac_seg)

        features.append(feats)
    return features


def build_feature_matrix(feature_dicts_list):
    """Combine multiple feature dict lists into a single feature matrix."""
    # Merge all feature dicts per shot
    n_shots = len(feature_dicts_list[0])
    merged = []
    for i in range(n_shots):
        combined = {}
        for feat_list in feature_dicts_list:
            combined.update(feat_list[i])
        merged.append(combined)

    feat_names = sorted(merged[0].keys())
    X = np.array([[d.get(name, 0.0) for name in feat_names] for d in merged], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_names


# ============================================================
# MODEL TRAINING - Per-player per-target with 5-fold CV
# ============================================================

def train_diverse_models(X, y_target, pids, target_name):
    """Train diverse models per-player with 5-fold CV. Returns OOF predictions and fitted models."""
    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros(len(y_target))
    all_models = {}
    all_scalers = {}

    print(f"\n{'='*60}")
    print(f"Training diverse models for: {target_name}")
    print(f"{'='*60}")

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y_target[mask]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        all_scalers[pid] = scaler

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)

        # Collect per-model OOF for reporting
        model_oofs = {name: np.zeros(n) for name in ['svr', 'knn', 'elasticnet', 'lgb', 'xgb', 'cat', 'ridge']}

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y_p[tr_idx], y_p[val_idx]

            preds_dict = {}

            # SVR with RBF kernel
            svr = SVR(kernel='rbf', C=1.0, epsilon=0.01)
            svr.fit(X_tr, y_tr)
            preds_dict['svr'] = svr.predict(X_val)

            # KNN regressor
            n_neighbors = min(5, len(X_tr) - 1)
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
            knn.fit(X_tr, y_tr)
            preds_dict['knn'] = knn.predict(X_val)

            # ElasticNet
            enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
            enet.fit(X_tr, y_tr)
            preds_dict['elasticnet'] = enet.predict(X_val)

            # LightGBM
            lgb_m = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1
            )
            lgb_m.fit(X_tr, y_tr)
            preds_dict['lgb'] = lgb_m.predict(X_val)

            # XGBoost
            xgb_m = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1
            )
            xgb_m.fit(X_tr, y_tr)
            preds_dict['xgb'] = xgb_m.predict(X_val)

            # CatBoost
            cat_m = CatBoostRegressor(
                iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False
            )
            cat_m.fit(X_tr, y_tr)
            preds_dict['cat'] = cat_m.predict(X_val)

            # Ridge
            ridge_m = Ridge(alpha=1.0)
            ridge_m.fit(X_tr, y_tr)
            preds_dict['ridge'] = ridge_m.predict(X_val)

            # Equal-weight ensemble of all 7 models
            ensemble = (
                preds_dict['svr'] * (1/7) +
                preds_dict['knn'] * (1/7) +
                preds_dict['elasticnet'] * (1/7) +
                preds_dict['lgb'] * (1/7) +
                preds_dict['xgb'] * (1/7) +
                preds_dict['cat'] * (1/7) +
                preds_dict['ridge'] * (1/7)
            )
            fold_preds[val_idx] = ensemble

            for name in model_oofs:
                model_oofs[name][val_idx] = preds_dict[name]

        oof_preds[global_idx] = fold_preds

        # Report per-model MSE for this player
        player_mse = np.mean((fold_preds - y_p) ** 2)
        print(f"  Player {pid} (n={n}): ensemble CV MSE = {player_mse:.6f}")
        for name in ['svr', 'knn', 'elasticnet', 'lgb', 'xgb', 'cat', 'ridge']:
            m_mse = np.mean((model_oofs[name] - y_p) ** 2)
            print(f"    {name:12s}: MSE = {m_mse:.6f}")

        # Train final models on all player data
        player_models = {}
        svr_f = SVR(kernel='rbf', C=1.0, epsilon=0.01)
        svr_f.fit(X_scaled, y_p)
        player_models['svr'] = svr_f

        n_neighbors = min(5, len(X_scaled) - 1)
        knn_f = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        knn_f.fit(X_scaled, y_p)
        player_models['knn'] = knn_f

        enet_f = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
        enet_f.fit(X_scaled, y_p)
        player_models['elasticnet'] = enet_f

        lgb_f = lgb.LGBMRegressor(
            n_estimators=100, num_leaves=10, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1
        )
        lgb_f.fit(X_scaled, y_p)
        player_models['lgb'] = lgb_f

        xgb_f = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1
        )
        xgb_f.fit(X_scaled, y_p)
        player_models['xgb'] = xgb_f

        cat_f = CatBoostRegressor(
            iterations=100, depth=4, learning_rate=0.05,
            l2_leaf_reg=3.0, random_state=42, verbose=False
        )
        cat_f.fit(X_scaled, y_p)
        player_models['cat'] = cat_f

        ridge_f = Ridge(alpha=1.0)
        ridge_f.fit(X_scaled, y_p)
        player_models['ridge'] = ridge_f

        all_models[pid] = player_models

    overall_mse = np.mean((oof_preds - y_target) ** 2)
    print(f"\n  Overall {target_name} ensemble CV MSE: {overall_mse:.6f}")

    return oof_preds, all_models, all_scalers


def predict_diverse_models(X, pids, all_models, all_scalers):
    """Generate test predictions from diverse model ensemble."""
    preds = np.zeros(len(X))
    unique_pids = sorted(np.unique(pids))

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        X_scaled = all_scalers[pid].transform(X_p)
        global_idx = np.where(mask)[0]

        model_preds = {}
        for name in ['svr', 'knn', 'elasticnet', 'lgb', 'xgb', 'cat', 'ridge']:
            model_preds[name] = all_models[pid][name].predict(X_scaled)

        # Equal weight ensemble
        ensemble = sum(model_preds[name] for name in model_preds) / len(model_preds)
        preds[global_idx] = ensemble

    return preds


def scale_predictions(raw_preds, target):
    """Scale raw predictions to [0,1] using saved scaler."""
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("DIVERSE MODELS BLEND EXPERIMENT")
    print("=" * 70)
    print(f"Goal: Create diverse predictions, blend with Sub 784 (LB 0.007224)")
    print()

    # Load data
    train_data, test_data = load_data()
    X_3d_train = train_data['X_3d']
    X_3d_test = test_data['X_3d']
    kp_names = train_data['keypoint_names']
    y = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']

    # ---- Feature extraction ----
    print("\n--- FEATURE EXTRACTION ---")

    print("  Extracting FFT features...")
    t1 = time.time()
    fft_train = extract_fft_features(X_3d_train, kp_names, top_k_values=(5, 10))
    fft_test = extract_fft_features(X_3d_test, kp_names, top_k_values=(5, 10))
    print(f"    Done in {time.time()-t1:.1f}s, {len(fft_train[0])} features per shot")

    print("  Extracting window features...")
    t1 = time.time()
    win_train = extract_window_features(X_3d_train, kp_names)
    win_test = extract_window_features(X_3d_test, kp_names)
    print(f"    Done in {time.time()-t1:.1f}s, {len(win_train[0])} features per shot")

    print("  Extracting velocity/acceleration features...")
    t1 = time.time()
    vel_train = extract_velocity_features(X_3d_train, kp_names)
    vel_test = extract_velocity_features(X_3d_test, kp_names)
    print(f"    Done in {time.time()-t1:.1f}s, {len(vel_train[0])} features per shot")

    # Combine all features
    print("  Building combined feature matrix...")
    X_train, feat_names = build_feature_matrix([fft_train, win_train, vel_train])
    X_test, _ = build_feature_matrix([fft_test, win_test, vel_test])
    print(f"    Total features: {len(feat_names)}")
    print(f"    Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ---- Train models for each target ----
    oof_predictions = {}
    test_predictions = {}
    model_stores = {}

    for target_idx, target_name in enumerate(TARGETS):
        oof, models, scalers = train_diverse_models(
            X_train, y[:, target_idx], pids_train, target_name
        )
        oof_predictions[target_name] = oof
        model_stores[target_name] = (models, scalers)

        test_pred = predict_diverse_models(X_test, pids_test, models, scalers)
        test_predictions[target_name] = test_pred

    # ---- Scale predictions ----
    print("\n--- SCALING PREDICTIONS ---")
    scaled_test = {}
    for target_name in TARGETS:
        raw = test_predictions[target_name]
        scaled = scale_predictions(raw, target_name)
        scaled_test[target_name] = scaled
        print(f"  {target_name}: raw range [{raw.min():.4f}, {raw.max():.4f}] -> "
              f"scaled range [{scaled.min():.4f}, {scaled.max():.4f}]")

    # ---- Load Sub 784 ----
    print("\n--- LOADING SUB 784 ---")
    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    print(f"  Sub 784 shape: {sub784.shape}")

    # Build aligned dataframe
    our_df = pd.DataFrame({
        'id': test_data['ids'],
        'new_angle': scaled_test['angle'],
        'new_depth': scaled_test['depth'],
        'new_lr': scaled_test['left_right'],
    })
    merged = sub784.merge(our_df, on='id')
    assert len(merged) == len(sub784), "ID mismatch between Sub 784 and our predictions"

    # ---- Correlations ----
    print("\n--- CORRELATIONS WITH SUB 784 ---")
    for col_784, col_new in [('scaled_angle', 'new_angle'),
                              ('scaled_depth', 'new_depth'),
                              ('scaled_left_right', 'new_lr')]:
        corr = np.corrcoef(merged[col_784].values, merged[col_new].values)[0, 1]
        diff_mse = np.mean((merged[col_784].values - merged[col_new].values) ** 2)
        print(f"  {col_784:20s} vs new: corr = {corr:.6f}, MSE diff = {diff_mse:.8f}")

    # ---- CV Summary ----
    print("\n--- CV SUMMARY (raw scale) ---")
    for target_idx, target_name in enumerate(TARGETS):
        overall_mse = np.mean((oof_predictions[target_name] - y[:, target_idx]) ** 2)
        print(f"  {target_name}: overall OOF MSE = {overall_mse:.6f}")

        for pid in sorted(np.unique(pids_train)):
            mask = pids_train == pid
            pid_mse = np.mean((oof_predictions[target_name][mask] - y[mask, target_idx]) ** 2)
            print(f"    Player {pid}: MSE = {pid_mse:.6f}")

    # ---- Blending and saving submissions ----
    print("\n" + "=" * 70)
    print("BLENDING WITH SUB 784")
    print("=" * 70)

    blend_weights = [0.10, 0.20, 0.30, 0.40, 0.50]
    submissions = []

    for w in blend_weights:
        blended_angle = (1 - w) * merged['scaled_angle'].values + w * merged['new_angle'].values
        blended_depth = (1 - w) * merged['scaled_depth'].values + w * merged['new_depth'].values
        blended_lr = (1 - w) * merged['scaled_left_right'].values + w * merged['new_lr'].values

        # Stats about the blend
        a_std = np.std(blended_angle)
        d_mean = np.mean(blended_depth)
        lr_std = np.std(blended_lr)

        # Divergence from Sub 784
        div_angle = np.mean((blended_angle - merged['scaled_angle'].values) ** 2)
        div_depth = np.mean((blended_depth - merged['scaled_depth'].values) ** 2)
        div_lr = np.mean((blended_lr - merged['scaled_left_right'].values) ** 2)
        total_div = div_angle + div_depth + div_lr

        sub_num = get_next_submission_number()
        sub_df = pd.DataFrame({
            'id': merged['id'],
            'scaled_angle': blended_angle,
            'scaled_depth': blended_depth,
            'scaled_left_right': blended_lr,
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(filepath, index=False)

        print(f"\n  Sub {sub_num}: blend weight w={w:.2f}")
        print(f"    Formula: final = {1-w:.2f} * sub784 + {w:.2f} * new_predictions")
        print(f"    angle_std={a_std:.6f}, depth_mean={d_mean:.6f}, lr_std={lr_std:.6f}")
        print(f"    Divergence from Sub784: angle={div_angle:.8f}, depth={div_depth:.8f}, lr={div_lr:.8f}, total={total_div:.8f}")
        print(f"    Saved to: {filepath}")

        submissions.append({
            'sub_num': sub_num,
            'weight': w,
            'filepath': str(filepath),
            'angle_std': a_std,
            'depth_mean': d_mean,
            'lr_std': lr_std,
            'div_total': total_div,
        })

    # ---- Also save pure new predictions (w=1.0) ----
    sub_num = get_next_submission_number()
    sub_df = pd.DataFrame({
        'id': merged['id'],
        'scaled_angle': merged['new_angle'],
        'scaled_depth': merged['new_depth'],
        'scaled_left_right': merged['new_lr'],
    })
    filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(filepath, index=False)
    print(f"\n  Sub {sub_num}: PURE new predictions (w=1.00, no blending)")
    print(f"    Saved to: {filepath}")

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"{'Sub':>6} {'Weight':>8} {'angle_std':>12} {'depth_mean':>12} {'div_total':>12}")
    print("-" * 60)
    for s in submissions:
        print(f"{s['sub_num']:>6} {s['weight']:>8.2f} {s['angle_std']:>12.6f} {s['depth_mean']:>12.6f} {s['div_total']:>12.8f}")
    print(f"{sub_num:>6} {'1.00':>8} {'':>12} {'':>12} {'pure':>12}")

    print(f"\n--- MODEL DETAILS ---")
    print(f"  Feature types: FFT (top-k={5,10}), Window stats (4 windows), Velocity/Acceleration (6 key joints)")
    print(f"  Total features: {len(feat_names)}")
    print(f"  Models: SVR(RBF), KNN(k=5,distance), ElasticNet(a=0.01,l1=0.5), LGB, XGB, CatBoost, Ridge")
    print(f"  Ensemble: equal weight (1/7 each)")
    print(f"  Training: per-player per-target, 5-fold KFold(shuffle=True, random_state=42)")
    print(f"  Base submission: Sub 784 (LB 0.007224)")
    print(f"  Blend weights tested: {blend_weights} + pure (1.0)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
