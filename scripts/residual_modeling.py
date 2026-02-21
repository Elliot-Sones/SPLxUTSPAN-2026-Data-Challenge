"""
Experiment: Residual Modeling

Hypothesis: Instead of predicting targets from scratch, train models to predict the
RESIDUAL ERROR of the current best approach. If the base model captures 90%+ of the
signal, we only need to learn the remaining structure in the errors.

Approach:
1. Run base ensemble with 5-fold CV to get OOF predictions
2. Compute residuals = y_true - OOF_pred
3. Train new models to predict these residuals using:
   - PLS components from raw timeseries (captures temporal patterns)
   - Hand-crafted features (existing pipeline)
   - Combination of both
4. Add predicted residuals to base predictions with a conservative weight
5. Also blend residual-corrected predictions with Sub 771 (current LB best)
"""

import json
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cross_decomposition import PLSRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data():
    """Load train and test data with both raw timeseries and hand-crafted features."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp = len(keypoint_cols)

    print(f"  Train: {len(train_df)} shots, Test: {len(test_df)} shots")
    print(f"  {n_kp} keypoint columns")

    def process(df, is_train=True):
        n = len(df)
        X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)
        X_feats = []
        ids = []
        pids = []
        targets = []

        keypoint_names = []
        for col in keypoint_cols:
            if col.endswith("_x"):
                keypoint_names.append(col[:-2])
        kp_index = {name: i for i, name in enumerate(keypoint_names)}

        for idx, row in df.iterrows():
            # Raw timeseries (flattened)
            ts_flat = np.zeros(n_kp * 240, dtype=np.float32)
            ts_3d = np.zeros((240, len(keypoint_names), 3), dtype=np.float32)

            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                ts_flat[col_i * 240:(col_i + 1) * 240] = arr
                kp_idx = col_i // 3
                coord_idx = col_i % 3
                ts_3d[:, kp_idx, coord_idx] = arr

            X_raw[idx] = ts_flat

            # Hand-crafted features
            feats = extract_handcrafted_features(ts_3d, keypoint_names, kp_index, row['participant_id'])
            X_feats.append(feats)

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        feat_names = sorted(X_feats[0].keys())
        X_feat = np.array([[f.get(name, 0.0) for name in feat_names] for f in X_feats], dtype=np.float32)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

        result = {
            'X_raw': X_raw,
            'X_feat': X_feat,
            'pids': np.array(pids),
            'ids': np.array(ids),
            'feat_names': feat_names,
        }
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    print("Processing train...")
    train = process(train_df, True)
    print("Processing test...")
    test = process(test_df, False)

    return train, test


def extract_handcrafted_features(ts_3d, keypoint_names, kp_index, pid):
    """Extract a focused set of hand-crafted features."""
    feats = {}
    feats['participant_id'] = pid

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist',
                  'left_shoulder', 'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'right_ankle', 'left_ankle', 'neck', 'nose']

    for joint in key_joints:
        if joint not in kp_index:
            continue
        idx = kp_index[joint]
        for c, cname in enumerate(['x', 'y', 'z']):
            s = ts_3d[:, idx, c]
            prefix = f"{joint}_{cname}"
            feats[f"{prefix}_mean"] = np.nanmean(s)
            feats[f"{prefix}_std"] = np.nanstd(s)
            feats[f"{prefix}_min"] = np.nanmin(s)
            feats[f"{prefix}_max"] = np.nanmax(s)
            feats[f"{prefix}_range"] = np.nanmax(s) - np.nanmin(s)
            feats[f"{prefix}_q25"] = np.nanpercentile(s, 25)
            feats[f"{prefix}_q75"] = np.nanpercentile(s, 75)

            vel = np.gradient(s, 1.0/60.0)
            feats[f"{prefix}_vel_mean"] = np.nanmean(vel)
            feats[f"{prefix}_vel_max"] = np.nanmax(vel)

            # Key frames
            feats[f"f153_{prefix}"] = s[153]
            feats[f"f102_{prefix}"] = s[102]

    # Joint angles
    for j1, j2, j3, name in [
        ('right_shoulder', 'right_elbow', 'right_wrist', 'elbow'),
        ('right_hip', 'right_shoulder', 'right_elbow', 'shoulder'),
        ('right_hip', 'right_knee', 'right_ankle', 'knee'),
    ]:
        if all(j in kp_index for j in [j1, j2, j3]):
            p1 = ts_3d[:, kp_index[j1]]
            p2 = ts_3d[:, kp_index[j2]]
            p3 = ts_3d[:, kp_index[j3]]
            v1 = p1 - p2
            v2 = p3 - p2
            dot = np.sum(v1 * v2, axis=1)
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            denom = n1 * n2
            denom[denom == 0] = 1e-10
            angle = np.degrees(np.arccos(np.clip(dot / denom, -1, 1)))
            feats[f"{name}_angle_at_release"] = angle[153]
            feats[f"{name}_angle_max"] = np.nanmax(angle)
            feats[f"{name}_angle_range"] = np.nanmax(angle) - np.nanmin(angle)
            feats[f"{name}_angle_vel"] = np.gradient(angle, 1/60.0)[153]

    # Phase-based velocity features
    phases = {'load': (60, 120), 'propel': (120, 170), 'follow': (170, 240)}
    for pname, (s, e) in phases.items():
        for joint in ['right_wrist', 'right_elbow']:
            if joint not in kp_index:
                continue
            idx = kp_index[joint]
            for c in range(3):
                series = ts_3d[s:e, idx, c]
                vel = np.gradient(series, 1.0/60.0)
                feats[f"phase_{pname}_{joint}_{'xyz'[c]}_vel_max"] = np.nanmax(vel)

    return feats


def train_base_ensemble(train_data):
    """
    Train base ensemble (replicating the existing pipeline) to get OOF residuals.
    Returns OOF predictions per target.
    """
    X = train_data['X_feat']
    y = train_data['y']
    pids = train_data['pids']

    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros_like(y)
    base_models = {}
    base_scalers = {}

    print("\n" + "=" * 70)
    print("STEP 1: TRAINING BASE ENSEMBLE FOR OOF PREDICTIONS")
    print("=" * 70)

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y[mask]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        base_scalers[pid] = scaler

        print(f"\nPlayer {pid} ({n} samples)")

        for t_idx, target in enumerate(TARGETS):
            y_t = y_p[:, t_idx]
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n)

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
                y_tr, y_val = y_t[tr_idx], y_t[val_idx]

                lgb_m = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(X_tr, y_tr)

                xgb_m = xgb.XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)
                xgb_m.fit(X_tr, y_tr)

                cat_m = CatBoostRegressor(
                    iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)
                cat_m.fit(X_tr, y_tr)

                ridge_m = Ridge(alpha=1.0, random_state=42)
                ridge_m.fit(X_tr, y_tr)

                pred = (0.3*lgb_m.predict(X_val) + 0.3*xgb_m.predict(X_val) +
                        0.3*cat_m.predict(X_val) + 0.1*ridge_m.predict(X_val))
                fold_preds[val_idx] = pred

            oof_preds[global_idx, t_idx] = fold_preds
            mse = np.mean((fold_preds - y_t) ** 2)
            print(f"  {target}: base CV MSE = {mse:.6f}")

            # Train final models
            for name, cls, params in [
                ('lgb', lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
                ('xgb', xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
                ('cat', CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                    l2_leaf_reg=3.0, random_state=42, verbose=False)),
                ('ridge', Ridge, dict(alpha=1.0, random_state=42)),
            ]:
                m = cls(**params)
                m.fit(X_scaled, y_t)
                base_models[(pid, target, name)] = m

    return oof_preds, base_models, base_scalers


def train_residual_models(train_data, oof_preds):
    """
    Train models to predict the residuals of the base ensemble.
    Uses PLS on raw timeseries as features (different feature space from base model).
    """
    X_raw = train_data['X_raw']
    X_feat = train_data['X_feat']
    y = train_data['y']
    pids = train_data['pids']

    residuals = y - oof_preds
    unique_pids = sorted(np.unique(pids))

    oof_residual_preds = np.zeros_like(residuals)
    residual_models = {}
    residual_scalers = {}

    print("\n" + "=" * 70)
    print("STEP 2: TRAINING RESIDUAL MODELS")
    print("=" * 70)

    for t_idx, target in enumerate(TARGETS):
        r = residuals[:, t_idx]
        print(f"\n{target} residuals: mean={r.mean():.6f}, std={r.std():.6f}")

    for pid in unique_pids:
        mask = pids == pid
        X_raw_p = X_raw[mask]
        X_feat_p = X_feat[mask]
        res_p = residuals[mask]
        n = len(X_raw_p)
        global_idx = np.where(mask)[0]

        # Standardize raw timeseries per player
        scaler_raw = StandardScaler()
        X_raw_scaled = scaler_raw.fit_transform(X_raw_p)
        residual_scalers[(pid, 'raw')] = scaler_raw

        scaler_feat = StandardScaler()
        X_feat_scaled = scaler_feat.fit_transform(X_feat_p)
        X_feat_scaled = np.nan_to_num(X_feat_scaled, nan=0.0)
        residual_scalers[(pid, 'feat')] = scaler_feat

        print(f"\nPlayer {pid} ({n} samples)")

        for t_idx, target in enumerate(TARGETS):
            res_t = res_p[:, t_idx]

            # Find optimal PLS components
            max_comp = min(30, n - n // 5 - 1)
            candidates = [c for c in [3, 5, 8, 10, 15, 20, 25, 30] if c <= max_comp]

            best_n = 5
            best_mse = float('inf')
            for nc in candidates:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                mses = []
                for tr_idx, val_idx in kf.split(X_raw_scaled):
                    pls = PLSRegression(n_components=nc)
                    pls.fit(X_raw_scaled[tr_idx], res_t[tr_idx])
                    pred = pls.predict(X_raw_scaled[val_idx]).flatten()
                    mses.append(np.mean((pred - res_t[val_idx]) ** 2))
                avg_mse = np.mean(mses)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_n = nc

            print(f"  {target}: best PLS components = {best_n}")

            # 5-fold CV for residual predictions
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n)

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_raw_scaled)):
                # PLS on raw timeseries
                pls = PLSRegression(n_components=best_n)
                pls.fit(X_raw_scaled[tr_idx], res_t[tr_idx])
                pls_pred = pls.predict(X_raw_scaled[val_idx]).flatten()

                # Ridge on PLS components
                X_tr_pls = pls.transform(X_raw_scaled[tr_idx])
                X_val_pls = pls.transform(X_raw_scaled[val_idx])
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
                ridge.fit(X_tr_pls, res_t[tr_idx])
                ridge_pred = ridge.predict(X_val_pls)

                # Ridge on hand-crafted features (for diversity)
                ridge_feat = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge_feat.fit(X_feat_scaled[tr_idx], res_t[tr_idx])
                feat_pred = ridge_feat.predict(X_feat_scaled[val_idx])

                # Blend residual predictions
                fold_preds[val_idx] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * feat_pred

            oof_residual_preds[global_idx, t_idx] = fold_preds
            mse_before = np.mean(res_t ** 2)
            mse_after = np.mean((res_t - fold_preds) ** 2)
            pct = (1 - mse_after / mse_before) * 100
            print(f"  {target}: residual MSE before={mse_before:.6f}, after={mse_after:.6f} ({pct:+.1f}%)")

            # Train final residual model
            pls_final = PLSRegression(n_components=best_n)
            pls_final.fit(X_raw_scaled, res_t)
            residual_models[(pid, target, 'pls')] = pls_final

            X_pls_all = pls_final.transform(X_raw_scaled)
            ridge_final = RidgeCV(alphas=[0.1, 1.0, 10.0])
            ridge_final.fit(X_pls_all, res_t)
            residual_models[(pid, target, 'ridge')] = ridge_final

            ridge_feat_final = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            ridge_feat_final.fit(X_feat_scaled, res_t)
            residual_models[(pid, target, 'ridge_feat')] = ridge_feat_final

    return oof_residual_preds, residual_models, residual_scalers


def evaluate_residual_correction(y, oof_base, oof_residual):
    """Evaluate different blending weights for residual correction."""
    print("\n" + "=" * 70)
    print("STEP 3: EVALUATING RESIDUAL CORRECTION WEIGHTS")
    print("=" * 70)

    scalers_target = {}
    for target in TARGETS:
        scalers_target[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    best_alpha = 0.0
    best_score = float('inf')

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        corrected = oof_base + alpha * oof_residual
        total = 0
        for t_idx, target in enumerate(TARGETS):
            raw_mse = np.mean((corrected[:, t_idx] - y[:, t_idx]) ** 2)
            scale_range = scalers_target[target].data_range_[0]
            total += raw_mse / (scale_range ** 2)
        avg = total / 3

        marker = ""
        if avg < best_score:
            best_score = avg
            best_alpha = alpha
            marker = " <-- BEST"

        print(f"  alpha={alpha:.1f}: scaled MSE = {avg:.8f}{marker}")

    # Also try per-target alpha
    print("\n  Per-target alpha search:")
    best_alphas = [0.0, 0.0, 0.0]
    for t_idx, target in enumerate(TARGETS):
        scale_range = scalers_target[target].data_range_[0]
        best_t_score = float('inf')
        for alpha in np.arange(0.0, 1.05, 0.05):
            corrected = oof_base[:, t_idx] + alpha * oof_residual[:, t_idx]
            raw_mse = np.mean((corrected - y[:, t_idx]) ** 2)
            scaled_mse = raw_mse / (scale_range ** 2)
            if scaled_mse < best_t_score:
                best_t_score = scaled_mse
                best_alphas[t_idx] = alpha
        print(f"  {target}: best alpha = {best_alphas[t_idx]:.2f}, scaled MSE = {best_t_score:.8f}")

    # Evaluate with per-target alphas
    total = 0
    for t_idx, target in enumerate(TARGETS):
        corrected = oof_base[:, t_idx] + best_alphas[t_idx] * oof_residual[:, t_idx]
        raw_mse = np.mean((corrected - y[:, t_idx]) ** 2)
        scale_range = scalers_target[target].data_range_[0]
        total += raw_mse / (scale_range ** 2)
    per_target_score = total / 3
    print(f"\n  Per-target alpha score: {per_target_score:.8f}")

    if per_target_score < best_score:
        print("  -> Per-target alphas are better!")
        return best_alphas, per_target_score
    else:
        print(f"  -> Uniform alpha {best_alpha} is better")
        return [best_alpha] * 3, best_score


def predict_test_with_residual(test_data, base_models, base_scalers,
                                residual_models, residual_scalers, alphas):
    """Generate test predictions with residual correction."""
    X_feat = test_data['X_feat']
    X_raw = test_data['X_raw']
    pids = test_data['pids']

    n_test = len(X_feat)
    base_preds = np.zeros((n_test, 3))
    residual_preds = np.zeros((n_test, 3))

    for i, pid in enumerate(pids):
        # Base predictions
        x_feat = base_scalers[pid].transform(X_feat[i:i+1])
        x_feat = np.nan_to_num(x_feat, nan=0.0)

        for t_idx, target in enumerate(TARGETS):
            lgb_p = base_models[(pid, target, 'lgb')].predict(x_feat)[0]
            xgb_p = base_models[(pid, target, 'xgb')].predict(x_feat)[0]
            cat_p = base_models[(pid, target, 'cat')].predict(x_feat)[0]
            ridge_p = base_models[(pid, target, 'ridge')].predict(x_feat)[0]
            base_preds[i, t_idx] = 0.3*lgb_p + 0.3*xgb_p + 0.3*cat_p + 0.1*ridge_p

        # Residual predictions
        x_raw = residual_scalers[(pid, 'raw')].transform(X_raw[i:i+1])
        x_feat_res = residual_scalers[(pid, 'feat')].transform(X_feat[i:i+1])
        x_feat_res = np.nan_to_num(x_feat_res, nan=0.0)

        for t_idx, target in enumerate(TARGETS):
            pls = residual_models[(pid, target, 'pls')]
            pls_pred = pls.predict(x_raw).flatten()[0]
            x_pls = pls.transform(x_raw)
            ridge_pred = residual_models[(pid, target, 'ridge')].predict(x_pls)[0]
            feat_pred = residual_models[(pid, target, 'ridge_feat')].predict(x_feat_res)[0]
            residual_preds[i, t_idx] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * feat_pred

    # Apply corrections with optimal alphas
    final_preds = np.zeros((n_test, 3))
    for t_idx in range(3):
        final_preds[:, t_idx] = base_preds[:, t_idx] + alphas[t_idx] * residual_preds[:, t_idx]

    return final_preds, base_preds


def blend_with_sub771(predictions, test_ids):
    """Try blending residual-corrected predictions with Sub 771."""
    sub771_path = SUBMISSION_DIR / "submission_771.csv"
    if not sub771_path.exists():
        print("\nSub 771 not found, skipping blend")
        return None

    sub771 = pd.read_csv(sub771_path)

    # Scale our predictions to [0,1] space
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    our_scaled = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        our_scaled[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    # Match ordering
    our_df = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': our_scaled[:, 0],
        'scaled_depth': our_scaled[:, 1],
        'scaled_left_right': our_scaled[:, 2],
    })

    merged = sub771.merge(our_df, on='id', suffixes=('_771', '_new'))

    print("\n" + "=" * 70)
    print("BLENDING WITH SUB 771")
    print("=" * 70)

    # Compute correlation between our predictions and Sub 771
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        corr = np.corrcoef(merged[f'{col}_771'], merged[f'{col}_new'])[0, 1]
        print(f"  {col} correlation with Sub 771: {corr:.4f}")

    # Try different blend weights
    print("\n  Blend weight search (our_weight):")
    best_profiles = []
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        blend = {}
        for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
            blend[col] = w * merged[f'{col}_new'] + (1 - w) * merged[f'{col}_771']

        angle_std = np.std(blend['scaled_angle'])
        depth_mean = np.mean(blend['scaled_depth'])
        profile_ok = angle_std < 0.14 and 0.50 <= depth_mean <= 0.51

        print(f"  w={w:.1f}: angle_std={angle_std:.6f} depth_mean={depth_mean:.6f} "
              f"{'PASS' if profile_ok else 'FAIL'}")

        if profile_ok:
            best_profiles.append((w, angle_std, depth_mean))

    return merged


def create_submission(test_ids, predictions, submission_num, cv_score, label=""):
    """Create submission file."""
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled[:, 0],
        'scaled_depth': scaled[:, 1],
        'scaled_left_right': scaled[:, 2],
    })

    filepath = SUBMISSION_DIR / f"submission_{submission_num}.csv"
    sub.to_csv(filepath, index=False)

    print(f"\nSubmission {label} saved: {filepath}")
    print(f"CV Score: {cv_score:.8f}")
    print(f"\nProfile check:")
    print(f"  angle_std:  {sub['scaled_angle'].std():.6f} (need < 0.14)")
    print(f"  depth_mean: {sub['scaled_depth'].mean():.6f} (need 0.50-0.51)")

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={sub[col].mean():.4f} std={sub[col].std():.4f}")

    return filepath


def main():
    t0 = time.time()

    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    print("=" * 70)
    print(f"RESIDUAL MODELING EXPERIMENT (Sub {next_num})")
    print("=" * 70)

    # Load data
    train_data, test_data = load_data()

    # Step 1: Base ensemble OOF
    oof_base, base_models, base_scalers = train_base_ensemble(train_data)

    # Step 2: Train residual models
    oof_residual, residual_models, residual_scalers = train_residual_models(train_data, oof_base)

    # Step 3: Find optimal correction weights
    alphas, cv_score = evaluate_residual_correction(train_data['y'], oof_base, oof_residual)
    print(f"\nOptimal alphas: {alphas}")

    # Step 4: Generate test predictions
    predictions, base_preds = predict_test_with_residual(
        test_data, base_models, base_scalers,
        residual_models, residual_scalers, alphas
    )

    # Create submission (residual-corrected)
    filepath = create_submission(test_data['ids'], predictions, next_num, cv_score, "residual-corrected")

    # Also save base predictions as a separate submission for comparison
    next_num2 = next_num + 1
    create_submission(test_data['ids'], base_preds, next_num2, cv_score, "base-only")

    # Try blending with Sub 771
    blend_with_sub771(predictions, test_data['ids'])

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    return filepath


if __name__ == "__main__":
    main()
