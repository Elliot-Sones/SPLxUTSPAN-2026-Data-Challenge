"""
tsfresh Automated Feature Extraction for Basketball Free Throw Prediction

tsfresh automatically extracts hundreds of time series features including:
- Statistical features (mean, std, skewness, kurtosis)
- Temporal features (autocorrelation, partial autocorrelation)
- Frequency domain features (FFT coefficients, spectral entropy)
- Complexity features (sample entropy, binned entropy)
- Change quantiles, linear trends, etc.

This script:
1. Extracts tsfresh features from key joint trajectories
2. Uses feature selection to reduce dimensionality
3. Tests with GroupKFold CV
4. Analyzes which features have within-player signal
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from data_loader import iterate_shots, get_keypoint_columns

# Install tsfresh if not available
try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    print("Installing tsfresh...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tsfresh", "-q"])
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
        from tsfresh.utilities.dataframe_functions import impute
        TSFRESH_AVAILABLE = True
    except ImportError:
        TSFRESH_AVAILABLE = False
        print("tsfresh installation failed.")


def get_keypoint_map(keypoint_cols):
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def prepare_tsfresh_data(shots_data, kp_idx, key_joints):
    """
    Prepare data in tsfresh format.
    Returns DataFrame with columns: id, time, and feature columns.
    """
    records = []

    for shot_idx, (metadata, timeseries) in enumerate(shots_data):
        shot_id = shot_idx

        # Use frames 60-180 (critical shooting phase)
        start_frame, end_frame = 60, min(180, len(timeseries))

        for t in range(start_frame, end_frame):
            record = {'id': shot_id, 'time': t - start_frame}

            for joint in key_joints:
                if joint not in kp_idx:
                    continue
                i = kp_idx[joint]

                # Add x, y, z coordinates
                record[f'{joint}_x'] = timeseries[t, i*3]
                record[f'{joint}_y'] = timeseries[t, i*3 + 1]
                record[f'{joint}_z'] = timeseries[t, i*3 + 2]

            records.append(record)

    return pd.DataFrame(records)


def get_next_submission_number():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            pass
    return max(numbers) + 1 if numbers else 1


def main():
    print("=" * 80)
    print("TSFRESH AUTOMATED FEATURE EXTRACTION")
    print("=" * 80)

    if not TSFRESH_AVAILABLE:
        print("tsfresh not available. Exiting.")
        return

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_map(keypoint_cols)

    # Key joints to analyze
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'mid_hip', 'right_knee']

    print(f"\nAnalyzing joints: {key_joints}")

    # Load all training data
    print("\nLoading training data...")
    train_shots = []
    train_targets = {'angle': [], 'depth': [], 'left_right': []}
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        train_shots.append((metadata, timeseries))
        train_players.append(metadata['participant_id'])
        for t in train_targets:
            train_targets[t].append(metadata[t])

    print(f"  Training shots: {len(train_shots)}")

    # Prepare tsfresh format
    print("\nPreparing tsfresh format...")
    train_ts_df = prepare_tsfresh_data(train_shots, kp_idx, key_joints)
    print(f"  Time series records: {len(train_ts_df)}")
    print(f"  Features per frame: {len(train_ts_df.columns) - 2}")

    # Scale targets
    targets = ['angle', 'depth', 'left_right']
    y_data = {}
    target_stats = {}
    for t in targets:
        vals = np.array(train_targets[t])
        target_stats[t] = {'min': vals.min(), 'max': vals.max()}
        y_data[t] = (vals - vals.min()) / (vals.max() - vals.min())

    groups = np.array(train_players)

    # =========================================================================
    # EXPERIMENT 1: Minimal Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: MINIMAL TSFRESH FEATURES")
    print("=" * 80)
    print("Using MinimalFCParameters for fast extraction...")

    try:
        X_minimal = extract_features(
            train_ts_df,
            column_id='id',
            column_sort='time',
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=1
        )
        X_minimal = impute(X_minimal)
        print(f"  Extracted features: {X_minimal.shape[1]}")

        # Handle any remaining NaN/inf
        X_minimal = X_minimal.replace([np.inf, -np.inf], np.nan)
        X_minimal = X_minimal.fillna(0)

        # Scale
        scaler_minimal = StandardScaler()
        X_minimal_scaled = scaler_minimal.fit_transform(X_minimal.values)

        gkf = GroupKFold(n_splits=5)
        results_minimal = {}

        for target in targets:
            y = y_data[target]
            mses = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_minimal_scaled, y, groups)):
                X_tr, X_val = X_minimal_scaled[train_idx], X_minimal_scaled[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = Ridge(alpha=10.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                pred = np.clip(pred, 0, 1)

                mse = mean_squared_error(y_val, pred)
                mses.append(mse)

            results_minimal[target] = np.mean(mses)
            print(f"    {target}: CV MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

        print(f"    Total: {sum(results_minimal.values())/3:.6f}")

    except Exception as e:
        print(f"  Minimal extraction failed: {e}")
        X_minimal_scaled = None

    # =========================================================================
    # EXPERIMENT 2: Efficient Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: EFFICIENT TSFRESH FEATURES")
    print("=" * 80)
    print("Using EfficientFCParameters for more features...")

    try:
        X_efficient = extract_features(
            train_ts_df,
            column_id='id',
            column_sort='time',
            default_fc_parameters=EfficientFCParameters(),
            disable_progressbar=True,
            n_jobs=1
        )
        X_efficient = impute(X_efficient)
        print(f"  Extracted features: {X_efficient.shape[1]}")

        # Handle any remaining NaN/inf
        X_efficient = X_efficient.replace([np.inf, -np.inf], np.nan)
        X_efficient = X_efficient.fillna(0)

        # Scale
        scaler_efficient = StandardScaler()
        X_efficient_scaled = scaler_efficient.fit_transform(X_efficient.values)

        results_efficient = {}

        for target in targets:
            y = y_data[target]
            mses = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_efficient_scaled, y, groups)):
                X_tr, X_val = X_efficient_scaled[train_idx], X_efficient_scaled[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = Ridge(alpha=100.0)  # Higher regularization for more features
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                pred = np.clip(pred, 0, 1)

                mse = mean_squared_error(y_val, pred)
                mses.append(mse)

            results_efficient[target] = np.mean(mses)
            print(f"    {target}: CV MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

        print(f"    Total: {sum(results_efficient.values())/3:.6f}")

        # Feature importance analysis
        print("\n  Top features by correlation with each target:")
        for target in targets:
            y = y_data[target]
            correlations = []
            for col in X_efficient.columns:
                r = np.corrcoef(X_efficient[col].values, y)[0, 1]
                if not np.isnan(r):
                    correlations.append((col, abs(r)))

            correlations.sort(key=lambda x: x[1], reverse=True)
            print(f"\n    {target.upper()}:")
            for feat, r in correlations[:5]:
                print(f"      {feat[:50]:50s}: r={r:.4f}")

    except Exception as e:
        print(f"  Efficient extraction failed: {e}")
        X_efficient_scaled = None
        X_efficient = None

    # =========================================================================
    # EXPERIMENT 3: Feature Selection
    # =========================================================================
    if X_efficient is not None:
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: TSFRESH WITH FEATURE SELECTION")
        print("=" * 80)

        for n_features in [20, 50, 100]:
            print(f"\n  Selecting top {n_features} features per target...")

            results_selected = {}

            for target in targets:
                y = y_data[target]
                mses = []

                for fold, (train_idx, val_idx) in enumerate(gkf.split(X_efficient_scaled, y, groups)):
                    X_tr, X_val = X_efficient_scaled[train_idx], X_efficient_scaled[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    # Select features on training data only
                    selector = SelectKBest(f_regression, k=min(n_features, X_tr.shape[1]))
                    X_tr_selected = selector.fit_transform(X_tr, y_tr)
                    X_val_selected = selector.transform(X_val)

                    model = Ridge(alpha=10.0)
                    model.fit(X_tr_selected, y_tr)
                    pred = model.predict(X_val_selected)
                    pred = np.clip(pred, 0, 1)

                    mse = mean_squared_error(y_val, pred)
                    mses.append(mse)

                results_selected[target] = np.mean(mses)
                print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

            print(f"    Total: {sum(results_selected.values())/3:.6f}")

    # =========================================================================
    # EXPERIMENT 4: Within-Player Correlations
    # =========================================================================
    if X_efficient is not None:
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: WITHIN-PLAYER CORRELATIONS")
        print("=" * 80)

        for target in targets:
            print(f"\n  {target.upper()} - Top within-player correlations:")

            y = y_data[target]
            feat_corrs = {}

            for col in X_efficient.columns:
                player_corrs = []
                for player in np.unique(groups):
                    mask = groups == player
                    if mask.sum() < 10:
                        continue

                    feat_vals = X_efficient[col].values[mask]
                    target_vals = y[mask]

                    if np.std(feat_vals) < 1e-6 or np.std(target_vals) < 1e-6:
                        continue

                    r = np.corrcoef(feat_vals, target_vals)[0, 1]
                    if not np.isnan(r):
                        player_corrs.append(r)

                if len(player_corrs) >= 3:
                    feat_corrs[col] = (np.mean(player_corrs), np.std(player_corrs))

            sorted_feats = sorted(feat_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)

            for feat, (mean_r, std_r) in sorted_feats[:5]:
                print(f"    {feat[:50]:50s}: r={mean_r:+.4f} (std={std_r:.4f})")

    # =========================================================================
    # CREATE SUBMISSIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CREATING SUBMISSIONS")
    print("=" * 80)

    # Load test data
    print("\nLoading test data...")
    test_shots = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        test_shots.append((metadata, timeseries))
        test_ids.append(metadata['id'])

    print(f"  Test shots: {len(test_shots)}")

    # Prepare test tsfresh format
    test_ts_df = prepare_tsfresh_data(test_shots, kp_idx, key_joints)

    if X_efficient is not None:
        try:
            # Extract test features
            X_test_efficient = extract_features(
                test_ts_df,
                column_id='id',
                column_sort='time',
                default_fc_parameters=EfficientFCParameters(),
                disable_progressbar=True,
                n_jobs=1
            )
            X_test_efficient = impute(X_test_efficient)

            # Ensure same columns
            for col in X_efficient.columns:
                if col not in X_test_efficient.columns:
                    X_test_efficient[col] = 0.0

            X_test_efficient = X_test_efficient[X_efficient.columns]
            X_test_efficient = X_test_efficient.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_test_scaled = scaler_efficient.transform(X_test_efficient.values)

            # Train final models
            predictions = {}
            for target in targets:
                model = Ridge(alpha=100.0)
                model.fit(X_efficient_scaled, y_data[target])
                pred = model.predict(X_test_scaled)
                predictions[target] = np.clip(pred, 0, 1)

            sub_num = get_next_submission_number()

            # Pure tsfresh submission
            submission = pd.DataFrame({
                'id': test_ids,
                'scaled_angle': predictions['angle'],
                'scaled_depth': predictions['depth'],
                'scaled_left_right': predictions['left_right']
            })

            submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
            for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
                submission[col] = submission[col].clip(0, 1)

            sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            submission.to_csv(sub_path, index=False)
            print(f"  Sub {sub_num}: Pure tsfresh, angle_std={submission['scaled_angle'].std():.4f}")
            sub_num += 1

            # Blend with Sub219
            sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
            sub219_indexed = sub219.set_index('id')

            for weight in [0.1, 0.2, 0.3]:
                blended = pd.DataFrame({'id': test_ids})

                for target, col in [('angle', 'scaled_angle'), ('depth', 'scaled_depth'), ('left_right', 'scaled_left_right')]:
                    ts_pred = predictions[target]
                    sub219_pred = sub219_indexed.loc[test_ids, col].values
                    blended[col] = weight * ts_pred + (1 - weight) * sub219_pred

                blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
                for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
                    blended[col] = blended[col].clip(0, 1)

                sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
                blended.to_csv(sub_path, index=False)
                print(f"  Sub {sub_num}: {int(weight*100)}% tsfresh + {int((1-weight)*100)}% Sub219")
                sub_num += 1

        except Exception as e:
            print(f"  Test feature extraction failed: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("tsfresh automatically extracts time series features including:")
    print("  - Statistical: mean, std, skewness, kurtosis")
    print("  - Temporal: autocorrelation, partial autocorrelation")
    print("  - Frequency: FFT coefficients")
    print("  - Complexity: entropy measures")


if __name__ == "__main__":
    main()
