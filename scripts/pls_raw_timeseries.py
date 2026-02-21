"""
Experiment: PLS Regression on Raw Time Series

Hypothesis: Partial Least Squares regression on the raw 49,680-dimensional timeseries
(207 keypoints x 240 frames) can capture temporal patterns that hand-crafted features miss.
PLS is specifically designed for the p >> n regime and finds latent components that maximize
covariance with targets. It's regularized by construction (number of components controls
complexity), making it robust with only 345 samples.

This approach has NOT been tried before in this project. Neural nets failed due to
overfitting, but PLS is linear and heavily regularized.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
import lightgbm as lgb

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


def load_raw_timeseries(csv_path, is_train=True):
    """Load data and flatten each shot's timeseries into a single vector."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)

    print(f"  {len(df)} shots, {n_kp_cols} keypoint columns")

    # Flatten: each shot becomes a vector of length n_kp_cols * 240
    n_samples = len(df)
    n_features = n_kp_cols * 240
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    targets = []
    pids = []
    ids = []

    for idx, row in df.iterrows():
        for col_i, col in enumerate(keypoint_cols):
            ts = parse_array_string(row[col])
            X[idx, col_i * 240:(col_i + 1) * 240] = ts

        ids.append(row['id'])
        pids.append(row['participant_id'])
        if is_train:
            targets.append([row['angle'], row['depth'], row['left_right']])

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    result = {
        'X': X,
        'pids': np.array(pids),
        'ids': np.array(ids),
    }
    if is_train:
        result['y'] = np.array(targets, dtype=np.float32)

    print(f"  Raw feature matrix: {X.shape}")
    return result


def find_optimal_pls_components(X, y, max_components=50, n_splits=5):
    """Find optimal number of PLS components via CV."""
    best_n = 5
    best_mse = float('inf')

    # Test a range of components
    candidates = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
    candidates = [c for c in candidates if c <= min(X.shape[0] - X.shape[0]//n_splits, X.shape[1], max_components)]

    for n_comp in candidates:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_mses = []

        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y_tr)
            pred = pls.predict(X_val).flatten()
            fold_mses.append(np.mean((pred - y_val) ** 2))

        avg_mse = np.mean(fold_mses)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_n = n_comp

    return best_n, best_mse


def train_pls_per_player_target(train_data):
    """Train PLS models per player per target."""
    X = train_data['X']
    y = train_data['y']
    pids = train_data['pids']

    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros_like(y)
    all_models = {}
    all_scalers = {}
    all_pls_components = {}

    print("\n" + "=" * 70)
    print("PLS REGRESSION ON RAW TIME SERIES")
    print("=" * 70)

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y[mask]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        print(f"\nPlayer {pid} ({n} samples)")

        # Standardize per player
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)
        all_scalers[pid] = scaler

        for t_idx, target in enumerate(TARGETS):
            y_t = y_p[:, t_idx]

            # Find optimal PLS components for this player-target
            max_comp = min(50, n - n // 5 - 1)
            best_n, best_mse_inner = find_optimal_pls_components(X_scaled, y_t, max_components=max_comp)
            all_pls_components[(pid, target)] = best_n
            print(f"  {target}: optimal PLS components = {best_n}")

            # 5-fold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n)

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
                y_tr, y_val = y_t[tr_idx], y_t[val_idx]

                # PLS model
                pls = PLSRegression(n_components=best_n)
                pls.fit(X_tr, y_tr)
                pls_pred = pls.predict(X_val).flatten()

                # Also extract PLS components and use Ridge on them
                X_tr_pls = pls.transform(X_tr)
                X_val_pls = pls.transform(X_val)

                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_tr_pls, y_tr)
                ridge_pred = ridge.predict(X_val_pls)

                # Also try LightGBM on PLS components
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=50, num_leaves=8, learning_rate=0.05,
                    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1
                )
                lgb_m.fit(X_tr_pls, y_tr)
                lgb_pred = lgb_m.predict(X_val_pls)

                # Blend: PLS direct, Ridge on components, LGB on components
                fold_preds[val_idx] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred

            oof_preds[global_idx, t_idx] = fold_preds
            mse = np.mean((fold_preds - y_t) ** 2)
            print(f"  {target}: CV MSE = {mse:.6f}")

            # Train final model on all player data
            pls_final = PLSRegression(n_components=best_n)
            pls_final.fit(X_scaled, y_t)
            all_models[(pid, target, 'pls')] = pls_final

            X_pls_all = pls_final.transform(X_scaled)
            ridge_final = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            ridge_final.fit(X_pls_all, y_t)
            all_models[(pid, target, 'ridge')] = ridge_final

            lgb_final = lgb.LGBMRegressor(
                n_estimators=50, num_leaves=8, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1
            )
            lgb_final.fit(X_pls_all, y_t)
            all_models[(pid, target, 'lgb')] = lgb_final

    # Overall CV
    print("\n" + "=" * 70)
    print("OVERALL CV RESULTS")
    print("=" * 70)

    scalers_target = {}
    for target in TARGETS:
        scalers_target[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    total_scaled_mse = 0
    for t_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((oof_preds[:, t_idx] - y[:, t_idx]) ** 2)
        scale_range = scalers_target[target].data_range_[0]
        scaled_mse = raw_mse / (scale_range ** 2)
        total_scaled_mse += scaled_mse
        print(f"  {target}: raw MSE = {raw_mse:.6f}, scaled MSE = {scaled_mse:.8f}")

    avg_scaled_mse = total_scaled_mse / 3
    print(f"\n  AVERAGE SCALED MSE (CV): {avg_scaled_mse:.8f}")

    return {
        'models': all_models,
        'scalers': all_scalers,
        'pls_components': all_pls_components,
        'oof_preds': oof_preds,
        'cv_score': avg_scaled_mse,
    }


def predict_test(test_data, trained):
    """Generate test predictions."""
    X = test_data['X']
    pids = test_data['pids']
    models = trained['models']
    scalers = trained['scalers']

    predictions = np.zeros((len(X), 3))

    for i, (x, pid) in enumerate(zip(X, pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))

        for t_idx, target in enumerate(TARGETS):
            pls_model = models[(pid, target, 'pls')]
            ridge_model = models[(pid, target, 'ridge')]
            lgb_model = models[(pid, target, 'lgb')]

            # PLS direct prediction
            pls_pred = pls_model.predict(x_scaled).flatten()[0]

            # Transform to PLS space and predict with Ridge/LGB
            x_pls = pls_model.transform(x_scaled)
            ridge_pred = ridge_model.predict(x_pls)[0]
            lgb_pred = lgb_model.predict(x_pls)[0]

            predictions[i, t_idx] = 0.4 * pls_pred + 0.3 * ridge_pred + 0.3 * lgb_pred

    return predictions


def create_submission(test_ids, predictions, submission_num, cv_score):
    """Create and save submission file."""
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

    print(f"\nSubmission saved: {filepath}")
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
    print(f"PLS RAW TIMESERIES EXPERIMENT (Sub {next_num})")
    print("=" * 70)

    # Load raw timeseries
    train_data = load_raw_timeseries(DATA_DIR / "train.csv", is_train=True)
    test_data = load_raw_timeseries(DATA_DIR / "test.csv", is_train=False)

    # Train PLS models
    trained = train_pls_per_player_target(train_data)

    # Generate test predictions
    predictions = predict_test(test_data, trained)

    # Create submission
    filepath = create_submission(test_data['ids'], predictions, next_num, trained['cv_score'])

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    return filepath


if __name__ == "__main__":
    main()
