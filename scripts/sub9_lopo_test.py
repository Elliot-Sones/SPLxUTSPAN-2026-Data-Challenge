"""
Test Sub9 Features with LOPO CV

Check if sub9's 368 features actually generalize across players.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'

TARGETS = ['angle', 'depth', 'left_right']

TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def compute_scaled_mse(preds, truth, target_name):
    scaler = TARGET_SCALERS[target_name]
    scaled_preds = np.clip(scaler.transform(preds.reshape(-1, 1)).flatten(), 0, 1)
    scaled_truth = np.clip(scaler.transform(truth.reshape(-1, 1)).flatten(), 0, 1)
    return np.mean((scaled_preds - scaled_truth) ** 2)


def load_sub9_features():
    """Load features exactly as sub9 does."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    all_features = []
    all_targets = []
    all_pids = []

    for idx, row in train_df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)
        all_targets.append([row['angle'], row['depth'], row['left_right']])
        all_pids.append(row['participant_id'])

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    return X, y, pids, feature_names


def lopo_cv(X_features, y_target, pids, target_name, model_type='ridge'):
    """Leave-One-Player-Out CV."""
    unique_pids = sorted(np.unique(pids))
    all_preds = np.zeros(len(y_target))

    for holdout_pid in unique_pids:
        train_mask = pids != holdout_pid
        test_mask = pids == holdout_pid

        X_train = np.nan_to_num(X_features[train_mask], nan=0.0)
        X_test = np.nan_to_num(X_features[test_mask], nan=0.0)
        y_train = y_target[train_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_type == 'ridge':
            model = Ridge(alpha=10.0)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, learning_rate=0.05,
                                       reg_alpha=1.0, reg_lambda=1.0, verbose=-1, random_state=42)

        model.fit(X_train_s, y_train)
        all_preds[test_mask] = model.predict(X_test_s)

    return compute_scaled_mse(all_preds, y_target, target_name)


def main():
    print("=" * 70)
    print("SUB9 FEATURES - LOPO CV TEST")
    print("Testing if sub9's 368 features generalize across players")
    print("=" * 70)

    print("\nLoading sub9 features...")
    X, y, pids, feature_names = load_sub9_features()
    print(f"Features: {X.shape}")

    results = {}

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n{target_name.upper()}:")

        y_target = y[:, target_idx]

        # Baseline
        baseline_mse = compute_scaled_mse(np.full(len(y_target), np.mean(y_target)), y_target, target_name)
        print(f"  Baseline (mean): {baseline_mse:.6f}")

        # Ridge LOPO
        ridge_mse = lopo_cv(X, y_target, pids, target_name, 'ridge')
        print(f"  Ridge LOPO:      {ridge_mse:.6f} ({(baseline_mse-ridge_mse)/baseline_mse*100:.1f}% vs baseline)")

        # LGB LOPO
        lgb_mse = lopo_cv(X, y_target, pids, target_name, 'lgb')
        print(f"  LGB LOPO:        {lgb_mse:.6f} ({(baseline_mse-lgb_mse)/baseline_mse*100:.1f}% vs baseline)")

        results[target_name] = {
            'baseline': baseline_mse,
            'ridge_lopo': ridge_mse,
            'lgb_lopo': lgb_mse,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_ridge = np.mean([r['ridge_lopo'] for r in results.values()])
    avg_lgb = np.mean([r['lgb_lopo'] for r in results.values()])

    print(f"\nSub9 features LOPO MSE:")
    print(f"  Ridge: {avg_ridge:.6f}")
    print(f"  LGB:   {avg_lgb:.6f}")

    print(f"\nComparison:")
    print(f"  Sub9 LB Score:              0.009109")
    print(f"  Sub9 features LOPO (Ridge): {avg_ridge:.6f}")
    print(f"  Sub9 features LOPO (LGB):   {avg_lgb:.6f}")
    print(f"  Physics features LOPO:      0.015563")

    if avg_lgb < 0.015563:
        print(f"\n  Sub9 features generalize BETTER than physics features")
    else:
        print(f"\n  Sub9 features generalize WORSE than physics features")


if __name__ == "__main__":
    main()
