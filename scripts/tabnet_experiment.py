"""
TabNet Experiment

TabNet is a deep learning model specifically designed for tabular data.
It uses sequential attention to select relevant features at each decision step.

This is Phase 3 from the recommended approach - only keep if it beats Phase 2.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("TabNet not installed. Run: pip install pytorch-tabnet")

from data_loader import iterate_shots, get_keypoint_columns


def extract_features(timeseries, keypoint_cols):
    """Extract basic features for TabNet experiment."""
    features = {}

    # Get keypoint names
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])

    # Release window (frames 100-160)
    release_window = timeseries[100:160]

    # For each keypoint, compute basic statistics
    for i, kp in enumerate(keypoints[:20]):  # Top 20 keypoints for speed
        x = timeseries[:, i*3]
        y = timeseries[:, i*3 + 1]
        z = timeseries[:, i*3 + 2]

        # Position stats
        features[f'{kp}_z_mean'] = np.nanmean(z)
        features[f'{kp}_z_std'] = np.nanstd(z)
        features[f'{kp}_z_max'] = np.nanmax(z)
        features[f'{kp}_z_range'] = np.nanmax(z) - np.nanmin(z)

        # Velocity (simple diff)
        vz = np.diff(z) * 60  # 60 fps
        features[f'{kp}_vz_max'] = np.nanmax(vz) if len(vz) > 0 else 0
        features[f'{kp}_vz_mean'] = np.nanmean(vz) if len(vz) > 0 else 0

        # Release window specific
        z_release = z[100:160]
        features[f'{kp}_z_release_mean'] = np.nanmean(z_release)
        features[f'{kp}_z_release_max'] = np.nanmax(z_release)

    return features


def main():
    print("=" * 80)
    print("TABNET EXPERIMENT")
    print("=" * 80)

    if not TABNET_AVAILABLE:
        print("\nInstalling pytorch-tabnet...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "-q"])

    # Import after potential install
    from pytorch_tabnet.tab_model import TabNetRegressor

    keypoint_cols = get_keypoint_columns()

    # Extract features
    print("\nExtracting features...")
    data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_features(timeseries, keypoint_cols)
        features['player_id'] = metadata['participant_id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']
        data.append(features)

        if i % 50 == 0:
            print(f"  Processed {i+1} shots...")

    df = pd.DataFrame(data)
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(df.columns) - 4}")  # Exclude player_id and targets

    # Prepare data
    feature_cols = [c for c in df.columns if c not in ['player_id', 'angle', 'depth', 'left_right']]
    X = df[feature_cols].values

    # Scale targets to 0-1
    targets = ['angle', 'depth', 'left_right']
    y_dict = {}
    for target in targets:
        vals = df[target].values
        y_dict[target] = (vals - vals.min()) / (vals.max() - vals.min())

    groups = df['player_id'].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Cross-validation
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 80)

    gkf = GroupKFold(n_splits=5)

    results = {
        'tabnet': {t: [] for t in targets},
        'baseline_ridge': {t: [] for t in targets}
    }

    from sklearn.linear_model import Ridge

    for target in targets:
        y = y_dict[target]

        print(f"\n{target.upper()}:")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Baseline: Ridge
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_train_s, y_train)
            ridge_pred = ridge.predict(X_val_s)
            ridge_mse = mean_squared_error(y_val, ridge_pred)
            results['baseline_ridge'][target].append(ridge_mse)

            # TabNet
            try:
                tabnet = TabNetRegressor(
                    n_d=8, n_a=8,  # Width of decision/attention
                    n_steps=3,  # Number of decision steps
                    gamma=1.5,  # Coefficient for feature reusage
                    lambda_sparse=1e-3,  # Sparsity regularization
                    optimizer_params=dict(lr=2e-2),
                    scheduler_params={"step_size": 50, "gamma": 0.9},
                    scheduler_fn=None,
                    mask_type='entmax',
                    verbose=0
                )

                tabnet.fit(
                    X_train_s, y_train.reshape(-1, 1),
                    eval_set=[(X_val_s, y_val.reshape(-1, 1))],
                    max_epochs=100,
                    patience=20,
                    batch_size=64,
                    virtual_batch_size=32,
                    drop_last=False
                )

                tabnet_pred = tabnet.predict(X_val_s).flatten()
                tabnet_mse = mean_squared_error(y_val, tabnet_pred)
                results['tabnet'][target].append(tabnet_mse)

            except Exception as e:
                print(f"  TabNet error in fold {fold}: {e}")
                results['tabnet'][target].append(float('inf'))

        # Summary
        ridge_mean = np.mean(results['baseline_ridge'][target])
        ridge_std = np.std(results['baseline_ridge'][target])

        if results['tabnet'][target] and not all(np.isinf(results['tabnet'][target])):
            tabnet_mean = np.mean([x for x in results['tabnet'][target] if not np.isinf(x)])
            tabnet_std = np.std([x for x in results['tabnet'][target] if not np.isinf(x)])

            print(f"  Ridge:  MSE = {ridge_mean:.6f} +/- {ridge_std:.6f}")
            print(f"  TabNet: MSE = {tabnet_mean:.6f} +/- {tabnet_std:.6f}")

            if tabnet_mean < ridge_mean:
                print(f"  --> TabNet WINS by {(ridge_mean - tabnet_mean) / ridge_mean * 100:.1f}%")
            else:
                print(f"  --> Ridge WINS by {(tabnet_mean - ridge_mean) / tabnet_mean * 100:.1f}%")
        else:
            print(f"  Ridge:  MSE = {ridge_mean:.6f} +/- {ridge_std:.6f}")
            print(f"  TabNet: FAILED")

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    total_ridge = sum(np.mean(results['baseline_ridge'][t]) for t in targets) / 3

    valid_tabnet = [np.mean([x for x in results['tabnet'][t] if not np.isinf(x)])
                   for t in targets if results['tabnet'][t] and not all(np.isinf(results['tabnet'][t]))]

    if valid_tabnet:
        total_tabnet = sum(valid_tabnet) / len(valid_tabnet)
        print(f"Ridge Total MSE:  {total_ridge:.6f}")
        print(f"TabNet Total MSE: {total_tabnet:.6f}")

        if total_tabnet < total_ridge:
            print("\nRECOMMENDATION: TabNet shows improvement - consider for ensemble")
        else:
            print("\nRECOMMENDATION: TabNet does NOT improve over Ridge - skip it")
    else:
        print(f"Ridge Total MSE: {total_ridge:.6f}")
        print("TabNet: All experiments failed")
        print("\nRECOMMENDATION: Skip TabNet")


if __name__ == "__main__":
    main()
