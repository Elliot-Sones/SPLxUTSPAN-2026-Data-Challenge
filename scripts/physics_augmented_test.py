"""
Physics Features + PLS Components: Combined Pipeline Test.

Tests whether physics features (angular velocities, arm geometry at release)
add value ON TOP OF PLS components from raw timeseries.

Compares:
  A) PLS components only (current best approach for depth)
  B) Physics features only (48 features)
  C) Physics + PLS combined
  D) Physics + hoop-relative features combined

If C or D beats both A and B, physics adds orthogonal signal.
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
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


def load_data():
    """Load raw data and physics features."""
    print("Loading data...")

    # Raw data for PLS
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    # Parse raw timeseries (for PLS)
    def parse_raw(df):
        n = len(df)
        X = np.zeros((n, len(keypoint_cols) * 240), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X[i, j*240:(j+1)*240] = arr
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_raw_train = parse_raw(train_df)
    X_raw_test = parse_raw(test_df)
    print(f"  Raw timeseries: {X_raw_train.shape}")

    # Physics features
    physics_train = pd.read_csv(PROJECT_DIR / "output/physics_scratch_v2_train.csv")
    physics_test = pd.read_csv(PROJECT_DIR / "output/physics_scratch_v2_test.csv")

    exclude = {'player_id', 'shot_id', 'angle', 'depth', 'left_right'}
    phys_cols = [c for c in physics_train.columns if c not in exclude]

    X_phys_train = physics_train[phys_cols].values.astype(np.float32)
    X_phys_test = physics_test[phys_cols].values.astype(np.float32)
    X_phys_train = np.nan_to_num(X_phys_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_phys_test = np.nan_to_num(X_phys_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Physics features: {X_phys_train.shape}")

    # Hoop-relative features (extract inline)
    print("  Extracting hoop-relative features...")

    def extract_hoop_features(df, keypoint_cols):
        """Quick hoop-relative features at key frames."""
        kp_names = []
        for col in keypoint_cols:
            if col.endswith("_x"):
                kp_names.append(col[:-2])

        n = len(df)
        all_feats = []

        for i, (_, row) in enumerate(df.iterrows()):
            # Parse 3D timeseries
            n_kp = len(kp_names)
            ts = np.zeros((240, n_kp, 3), dtype=np.float32)
            for j, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                ts[:, j//3, j%3] = arr

            pid = row['participant_id']

            # Compute hoop-relative transform
            # Player center at frame 120 (mid-shot)
            mid_hip_idx = kp_names.index('mid_hip') if 'mid_hip' in kp_names else 0
            player_pos = ts[120, mid_hip_idx, :].copy()
            player_pos[2] = 0  # horizontal plane only

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

            # Transform
            centered = ts - player_pos.reshape(1, 1, 3)
            ts_hr = np.einsum('ij,fkj->fki', R, centered)

            # Key joints
            key_joint_names = [
                'right_wrist', 'right_elbow', 'right_shoulder',
                'left_wrist', 'left_shoulder',
                'right_hip', 'left_hip', 'mid_hip',
                'neck', 'nose',
            ]
            key_joint_idx = {}
            for name in key_joint_names:
                if name in kp_names:
                    key_joint_idx[name] = kp_names.index(name)

            feats = {}
            feats['participant_id'] = pid

            # Frame 153 features (release frame) + stats
            frames = [120, 140, 150, 153, 160, 170, 180]
            for jname, jidx in key_joint_idx.items():
                for coord, cname in enumerate(['fwd', 'lat', 'vert']):
                    series = ts_hr[:, jidx, coord]
                    # Stats
                    feats[f'hr_{jname}_{cname}_mean'] = np.nanmean(series)
                    feats[f'hr_{jname}_{cname}_std'] = np.nanstd(series)
                    feats[f'hr_{jname}_{cname}_range'] = np.nanmax(series) - np.nanmin(series)
                    # Release window
                    feats[f'hr_{jname}_{cname}_release_mean'] = np.nanmean(series[140:180])
                    # Velocity at frame 153
                    vel = np.gradient(series, 1.0/60.0)
                    feats[f'hr_{jname}_{cname}_vel153'] = vel[153] if len(vel) > 153 else 0

                    # Multi-frame features
                    for fr in frames:
                        if fr < len(series):
                            feats[f'hr_{jname}_{cname}_f{fr}'] = series[fr]

            # Arm mechanics in hoop frame
            rw = key_joint_idx.get('right_wrist')
            rs = key_joint_idx.get('right_shoulder')
            if rw is not None and rs is not None:
                arm_fwd = ts_hr[:, rw, 0] - ts_hr[:, rs, 0]
                arm_lat = ts_hr[:, rw, 1] - ts_hr[:, rs, 1]
                feats['hr_arm_ext_fwd_153'] = arm_fwd[153]
                feats['hr_arm_lat_dev_153'] = arm_lat[153]
                feats['hr_arm_ext_fwd_vel'] = np.gradient(arm_fwd, 1/60.)[153]
                feats['hr_arm_lat_dev_vel'] = np.gradient(arm_lat, 1/60.)[153]

            all_feats.append(feats)

        feat_df = pd.DataFrame(all_feats)
        pid_col = feat_df.pop('participant_id')
        return feat_df.values.astype(np.float32), feat_df.columns.tolist(), pid_col.values

    X_hr_train, hr_cols, _ = extract_hoop_features(train_df, keypoint_cols)
    X_hr_test, _, _ = extract_hoop_features(test_df, keypoint_cols)
    X_hr_train = np.nan_to_num(X_hr_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_hr_test = np.nan_to_num(X_hr_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Hoop-relative features: {X_hr_train.shape}")

    # Targets
    y_train = train_df[['angle', 'depth', 'left_right']].values
    pids_train = train_df['participant_id'].values
    pids_test = test_df['participant_id'].values

    return {
        'X_raw_train': X_raw_train, 'X_raw_test': X_raw_test,
        'X_phys_train': X_phys_train, 'X_phys_test': X_phys_test,
        'X_hr_train': X_hr_train, 'X_hr_test': X_hr_test,
        'y_train': y_train,
        'pids_train': pids_train, 'pids_test': pids_test,
        'phys_cols': phys_cols, 'hr_cols': hr_cols,
    }


def evaluate_config(data, config_name, X_train, X_test, use_pls=False):
    """
    Evaluate a feature configuration using within-player CV.

    If use_pls=True, adds PLS components from raw timeseries as features.
    """
    y = data['y_train']
    pids = data['pids_train']
    unique_pids = sorted(np.unique(pids))

    results = {}

    for t_idx, target in enumerate(['angle', 'depth', 'left_right']):
        y_raw = y[:, t_idx]

        # Scale targets
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()
        else:
            y_scaled = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)

        oof = np.full(len(y_scaled), np.nan)

        for pid in unique_pids:
            mask = pids == pid
            X_p = X_train[mask]
            y_p = y_scaled[mask]
            X_raw_p = data['X_raw_train'][mask] if use_pls else None
            indices = np.where(mask)[0]
            n_p = len(X_p)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for tr_idx, val_idx in kf.split(X_p):
                X_tr, X_val = X_p[tr_idx], X_p[val_idx]
                y_tr = y_p[tr_idx]

                # Optionally add PLS components
                if use_pls and X_raw_p is not None:
                    raw_tr = X_raw_p[tr_idx]
                    raw_val = X_raw_p[val_idx]

                    ss = StandardScaler()
                    raw_tr_s = ss.fit_transform(raw_tr)
                    raw_val_s = ss.transform(raw_val)

                    # PLS with optimal components for this player/target
                    max_comp = min(20, len(raw_tr) - 1)
                    if max_comp >= 3:
                        pls = PLSRegression(n_components=max_comp)
                        pls.fit(raw_tr_s, y_tr)
                        pls_tr = pls.transform(raw_tr_s)
                        pls_val = pls.transform(raw_val_s)
                        X_tr = np.hstack([X_tr, pls_tr])
                        X_val = np.hstack([X_val, pls_val])

                # Models
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr, y_tr)
                pred_r = ridge.predict(X_val)
                pred_final = pred_r

                if lgb is not None and len(X_tr) >= 10:
                    lgb_m = lgb.LGBMRegressor(
                        n_estimators=100, num_leaves=8,
                        min_child_samples=max(5, len(X_tr) // 10),
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        verbose=-1, n_jobs=1,
                    )
                    lgb_m.fit(X_tr, y_tr)
                    pred_l = lgb_m.predict(X_val)

                    if xgb is not None:
                        xgb_m = xgb.XGBRegressor(
                            n_estimators=100, max_depth=3,
                            learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                            verbosity=0, n_jobs=1,
                        )
                        xgb_m.fit(X_tr, y_tr)
                        pred_x = xgb_m.predict(X_val)
                        pred_final = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x
                    else:
                        pred_final = 0.4 * pred_r + 0.6 * pred_l

                oof[indices[val_idx]] = pred_final

        nan_mask = np.isnan(oof)
        if np.any(nan_mask):
            oof[nan_mask] = np.mean(y_scaled)

        mse = mean_squared_error(y_scaled, oof)
        r = np.corrcoef(y_scaled, oof)[0, 1] if np.std(oof) > 1e-9 else 0.0
        results[target] = {'mse': mse, 'r': r, 'oof': oof}

    return results


def main():
    print("="*70)
    print("PHYSICS FEATURES + PLS COMPONENTS: COMBINED PIPELINE TEST")
    print("="*70)

    data = load_data()

    configs = [
        ("A) Physics only (48 feats)", data['X_phys_train'], data['X_phys_test'], False),
        ("B) Hoop-relative only (~600 feats)", data['X_hr_train'], data['X_hr_test'], False),
        ("C) Physics + PLS", data['X_phys_train'], data['X_phys_test'], True),
        ("D) Hoop-relative + PLS", data['X_hr_train'], data['X_hr_test'], True),
        ("E) Physics + Hoop-relative",
         np.hstack([data['X_phys_train'], data['X_hr_train']]),
         np.hstack([data['X_phys_test'], data['X_hr_test']]), False),
        ("F) Physics + Hoop-relative + PLS",
         np.hstack([data['X_phys_train'], data['X_hr_train']]),
         np.hstack([data['X_phys_test'], data['X_hr_test']]), True),
    ]

    all_results = {}
    for name, X_tr, X_te, use_pls in configs:
        print(f"\n--- {name} (shape: {X_tr.shape}) ---")
        results = evaluate_config(data, name, X_tr, X_te, use_pls=use_pls)

        for t in ['angle', 'depth', 'left_right']:
            print(f"  {t:12s}: MSE={results[t]['mse']:.6f}, r={results[t]['r']:+.4f}")

        mean_mse = np.mean([results[t]['mse'] for t in results])
        print(f"  MEAN MSE = {mean_mse:.6f}")
        all_results[name] = results

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"\n{'Config':<45} {'Angle':>8} {'Depth':>8} {'LR':>8} {'Mean':>8}")
    print("-" * 80)
    for name, results in all_results.items():
        a = results['angle']['mse']
        d = results['depth']['mse']
        lr = results['left_right']['mse']
        m = np.mean([a, d, lr])
        print(f"{name:<45} {a:.6f} {d:.6f} {lr:.6f} {m:.6f}")

    # Find best config and generate submission
    best_name = min(all_results, key=lambda x: np.mean([
        all_results[x][t]['mse'] for t in ['angle', 'depth', 'left_right']
    ]))
    best_mean = np.mean([all_results[best_name][t]['mse']
                         for t in ['angle', 'depth', 'left_right']])
    print(f"\nBest: {best_name} (Mean MSE = {best_mean:.6f})")

    if best_mean < 0.0090:
        print("\nBest config is promising! Generating submission blended with Sub 784...")
        # Find the best config's details
        for name, X_tr, X_te, use_pls in configs:
            if name == best_name:
                generate_submission(data, X_tr, X_te, use_pls, all_results[best_name])
                break


def generate_submission(data, X_train, X_test, use_pls, cv_results):
    """Generate test predictions and blend with Sub 784."""
    y = data['y_train']
    pids_tr = data['pids_train']
    pids_te = data['pids_test']
    unique_pids = sorted(np.unique(pids_tr))

    test_preds = {}
    for t_idx, target in enumerate(['angle', 'depth', 'left_right']):
        y_raw = y[:, t_idx]
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            y_scaled = scaler.transform(y_raw.reshape(-1, 1)).ravel()
        else:
            y_scaled = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)

        preds = np.zeros(len(data['X_raw_test']))

        for pid in unique_pids:
            tr_mask = pids_tr == pid
            te_mask = pids_te == pid
            if not np.any(tr_mask) or not np.any(te_mask):
                continue

            X_tr = X_train[tr_mask]
            X_te_p = X_test[te_mask]
            y_tr = y_scaled[tr_mask]

            if use_pls:
                raw_tr = data['X_raw_train'][tr_mask]
                raw_te = data['X_raw_test'][te_mask]
                ss = StandardScaler()
                raw_tr_s = ss.fit_transform(raw_tr)
                raw_te_s = ss.transform(raw_te)
                max_comp = min(20, len(raw_tr) - 1)
                if max_comp >= 3:
                    pls = PLSRegression(n_components=max_comp)
                    pls.fit(raw_tr_s, y_tr)
                    X_tr = np.hstack([X_tr, pls.transform(raw_tr_s)])
                    X_te_p = np.hstack([X_te_p, pls.transform(raw_te_s)])

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr, y_tr)
            pred_r = ridge.predict(X_te_p)

            if lgb is not None and len(X_tr) >= 10:
                lgb_m = lgb.LGBMRegressor(
                    n_estimators=100, num_leaves=8,
                    min_child_samples=max(5, len(X_tr) // 10),
                    learning_rate=0.05, verbose=-1, n_jobs=1,
                )
                lgb_m.fit(X_tr, y_tr)
                pred_l = lgb_m.predict(X_te_p)

                if xgb is not None:
                    xgb_m = xgb.XGBRegressor(
                        n_estimators=100, max_depth=3,
                        learning_rate=0.05, verbosity=0, n_jobs=1,
                    )
                    xgb_m.fit(X_tr, y_tr)
                    pred_x = xgb_m.predict(X_te_p)
                    preds[te_mask] = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x
                else:
                    preds[te_mask] = 0.4 * pred_r + 0.6 * pred_l
            else:
                preds[te_mask] = pred_r

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
        print(f"    w={w:.2f}: angle_std={a_std:.4f}, depth_mean={d_mean:.4f}")

        max_num += 1
        path = SUBMISSION_DIR / f"submission_{max_num}.csv"
        sub.to_csv(path, index=False)
        print(f"      -> {path.name}")


if __name__ == "__main__":
    main()
