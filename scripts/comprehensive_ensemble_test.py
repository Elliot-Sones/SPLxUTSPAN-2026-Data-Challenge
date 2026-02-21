"""
Comprehensive Ensemble Test

This script systematically tests:
1. All augmentation techniques combined
2. Full stacking with OOF predictions
3. 10-fold CV (vs current 5-fold)
4. Multiple meta-learners
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import sys

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from data_loader import iterate_shots, get_keypoint_columns


def extract_comprehensive_features(timeseries, keypoint_cols):
    """Extract comprehensive biomechanical features."""
    features = {}

    # Get keypoint names
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])

    # Key joints for analysis
    key_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'mid_hip', 'right_knee', 'right_ankle',
        'right_third_finger_distal', 'right_second_finger_mcp'
    ]

    keypoint_idx = {kp: i for i, kp in enumerate(keypoints)}

    for joint in key_joints:
        if joint not in keypoint_idx:
            continue
        i = keypoint_idx[joint]

        x = timeseries[:, i*3]
        y = timeseries[:, i*3 + 1]
        z = timeseries[:, i*3 + 2]

        # Full trajectory stats
        features[f'{joint}_z_mean'] = np.nanmean(z)
        features[f'{joint}_z_std'] = np.nanstd(z)
        features[f'{joint}_z_max'] = np.nanmax(z)
        features[f'{joint}_z_min'] = np.nanmin(z)
        features[f'{joint}_z_range'] = np.nanmax(z) - np.nanmin(z)

        # Position at key frames
        for frame in [100, 110, 120, 130, 140, 150]:
            if frame < len(z):
                features[f'{joint}_z_f{frame}'] = z[frame]
                features[f'{joint}_x_f{frame}'] = x[frame]

        # Velocity features
        vx = np.diff(x) * 60
        vy = np.diff(y) * 60
        vz = np.diff(z) * 60
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        features[f'{joint}_vz_max'] = np.nanmax(vz)
        features[f'{joint}_vz_mean'] = np.nanmean(vz)
        features[f'{joint}_speed_max'] = np.nanmax(speed)

        # Peak velocity frame
        peak_vz_idx = np.nanargmax(vz) if len(vz) > 0 else 0
        features[f'{joint}_peak_vz_frame'] = peak_vz_idx

        if peak_vz_idx < len(vz):
            features[f'{joint}_vz_at_peak'] = vz[peak_vz_idx]
            features[f'{joint}_speed_at_peak'] = speed[peak_vz_idx] if peak_vz_idx < len(speed) else 0

        # Release window (frames 100-160)
        z_release = z[100:160] if len(z) > 160 else z[100:]
        vz_release = vz[100:160] if len(vz) > 160 else vz[100:]

        features[f'{joint}_z_release_mean'] = np.nanmean(z_release)
        features[f'{joint}_vz_release_max'] = np.nanmax(vz_release) if len(vz_release) > 0 else 0

    # Kinetic chain features
    if 'right_wrist' in keypoint_idx and 'right_elbow' in keypoint_idx:
        wrist_i = keypoint_idx['right_wrist']
        elbow_i = keypoint_idx['right_elbow']

        wrist_vz = np.diff(timeseries[:, wrist_i*3 + 2]) * 60
        elbow_vz = np.diff(timeseries[:, elbow_i*3 + 2]) * 60

        wrist_peak = np.nanargmax(wrist_vz) if len(wrist_vz) > 0 else 0
        elbow_peak = np.nanargmax(elbow_vz) if len(elbow_vz) > 0 else 0

        features['kc_wrist_elbow_delay'] = (wrist_peak - elbow_peak) / 60.0

    return features


def augment_data(X, y, groups):
    """Apply data augmentation."""
    X_aug = [X.copy()]
    y_aug = [y.copy()]
    groups_aug = [groups.copy()]
    weights = [np.ones(len(X))]

    # Jitter augmentation
    for noise_level in [0.01, 0.02]:
        X_noisy = X + np.random.normal(0, noise_level, X.shape)
        X_aug.append(X_noisy)
        y_aug.append(y.copy())
        groups_aug.append(groups.copy())
        weights.append(np.ones(len(X)) * 0.5)

    # Scale augmentation
    for scale in [0.95, 1.05]:
        X_scaled = X * scale
        X_aug.append(X_scaled)
        y_aug.append(y.copy())
        groups_aug.append(groups.copy())
        weights.append(np.ones(len(X)) * 0.5)

    X_combined = np.vstack(X_aug)
    y_combined = np.concatenate(y_aug)
    groups_combined = np.concatenate(groups_aug)
    weights_combined = np.concatenate(weights)

    return X_combined, y_combined, groups_combined, weights_combined


class StackingEnsemble:
    """Full stacking ensemble with OOF predictions."""

    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.base_models = {}
        self.meta_model = None
        self.scalers = {}

    def _get_base_models(self):
        """Create base model instances."""
        models = {
            'lgb': lgb.LGBMRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0, num_leaves=20,
                subsample=0.7, colsample_bytree=0.7,
                verbose=-1, n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=1.0,
                subsample=0.7, colsample_bytree=0.7,
                verbosity=0, n_jobs=-1
            ),
            'ridge': Ridge(alpha=10.0),
            'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }

        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostRegressor(
                iterations=200, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, verbose=0
            )

        return models

    def fit(self, X, y, groups=None):
        """Fit stacking ensemble with OOF predictions."""
        n_samples = len(X)
        n_models = len(self._get_base_models())

        # OOF predictions for each base model
        oof_predictions = np.zeros((n_samples, n_models))

        if groups is not None:
            kf = GroupKFold(n_splits=self.n_folds)
            split_args = (X, y, groups)
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            split_args = (X, y)

        model_names = list(self._get_base_models().keys())

        for fold, (train_idx, val_idx) in enumerate(kf.split(*split_args)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for i, (name, model_template) in enumerate(self._get_base_models().items()):
                # Clone model
                if hasattr(model_template, 'get_params'):
                    model = model_template.__class__(**model_template.get_params())
                else:
                    model = model_template

                model.fit(X_train, y_train)
                oof_predictions[val_idx, i] = model.predict(X_val)

        # Train meta-model on OOF predictions
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_predictions, y)

        # Train final base models on all data
        self.base_models = {}
        for name, model in self._get_base_models().items():
            model.fit(X, y)
            self.base_models[name] = model

        return oof_predictions

    def predict(self, X):
        """Make predictions using stacked ensemble."""
        base_preds = np.column_stack([
            model.predict(X) for model in self.base_models.values()
        ])
        return self.meta_model.predict(base_preds)


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
    print("COMPREHENSIVE ENSEMBLE TEST")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()

    # Extract features
    print("\nExtracting features...")
    data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_comprehensive_features(timeseries, keypoint_cols)
        features['player_id'] = metadata['participant_id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']
        data.append(features)

        if i % 50 == 0:
            print(f"  Processed {i+1} shots...")

    df = pd.DataFrame(data)
    print(f"  Total samples: {len(df)}")

    feature_cols = [c for c in df.columns if c not in ['player_id', 'angle', 'depth', 'left_right']]
    print(f"  Features: {len(feature_cols)}")

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)

    targets = ['angle', 'depth', 'left_right']
    y_dict = {}
    for target in targets:
        vals = df[target].values
        y_dict[target] = (vals - vals.min()) / (vals.max() - vals.min())

    groups = df['player_id'].values

    # Test different CV folds
    print("\n" + "=" * 80)
    print("COMPARING CV STRATEGIES")
    print("=" * 80)

    # GroupKFold is limited to 5 folds (5 players)
    for n_folds, cv_type in [(5, 'GroupKFold'), (10, 'KFold')]:
        print(f"\n{n_folds}-Fold {cv_type}:")

        if cv_type == 'GroupKFold':
            gkf = GroupKFold(n_splits=n_folds)
        else:
            gkf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for target in targets:
            y = y_dict[target]
            mses = []

            if cv_type == 'GroupKFold':
                splits = gkf.split(X, y, groups)
            else:
                splits = gkf.split(X, y)

            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                model = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_lambda=1.0, verbose=-1
                )
                model.fit(X_train_s, y_train)
                pred = model.predict(X_val_s)
                mses.append(mean_squared_error(y_val, pred))

            print(f"  {target}: MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

    # Test augmentation impact
    print("\n" + "=" * 80)
    print("TESTING AUGMENTATION IMPACT")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=5)

    for aug_name, use_aug in [("No Augmentation", False), ("With Augmentation", True)]:
        print(f"\n{aug_name}:")

        for target in targets:
            y = y_dict[target]
            mses = []

            for train_idx, val_idx in gkf.split(X_scaled, y, groups):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                g_train = groups[train_idx]

                if use_aug:
                    X_train, y_train, g_train, weights = augment_data(X_train, y_train, g_train)
                else:
                    weights = None

                model = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    reg_lambda=1.0, verbose=-1
                )
                model.fit(X_train, y_train, sample_weight=weights)
                pred = model.predict(X_val)
                mses.append(mean_squared_error(y_val, pred))

            print(f"  {target}: MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

    # Test full stacking ensemble
    print("\n" + "=" * 80)
    print("TESTING STACKING ENSEMBLE")
    print("=" * 80)

    for target in targets:
        y = y_dict[target]

        # Single model baseline
        baseline_mses = []
        stacking_mses = []

        for train_idx, val_idx in gkf.split(X_scaled, y, groups):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            g_train = groups[train_idx]

            # Baseline: Single LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                reg_lambda=1.0, verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            baseline_pred = lgb_model.predict(X_val)
            baseline_mses.append(mean_squared_error(y_val, baseline_pred))

            # Stacking ensemble
            stacker = StackingEnsemble(n_folds=3)  # Inner CV
            stacker.fit(X_train, y_train, g_train)
            stack_pred = stacker.predict(X_val)
            stacking_mses.append(mean_squared_error(y_val, stack_pred))

        print(f"\n{target}:")
        print(f"  Baseline LGB: MSE = {np.mean(baseline_mses):.6f} +/- {np.std(baseline_mses):.6f}")
        print(f"  Stacking:     MSE = {np.mean(stacking_mses):.6f} +/- {np.std(stacking_mses):.6f}")

        if np.mean(stacking_mses) < np.mean(baseline_mses):
            improvement = (np.mean(baseline_mses) - np.mean(stacking_mses)) / np.mean(baseline_mses) * 100
            print(f"  --> Stacking WINS by {improvement:.1f}%")
        else:
            degradation = (np.mean(stacking_mses) - np.mean(baseline_mses)) / np.mean(baseline_mses) * 100
            print(f"  --> Baseline WINS (stacking {degradation:.1f}% worse)")

    # Generate submission with best approach
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION")
    print("=" * 80)

    # Extract test features
    test_data = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_comprehensive_features(timeseries, keypoint_cols)
        test_data.append(features)
        test_ids.append(metadata['id'])

    test_df = pd.DataFrame(test_data)
    X_test = test_df[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    predictions = {}

    for target in targets:
        y = y_dict[target]

        # Use stacking ensemble
        stacker = StackingEnsemble(n_folds=5)
        stacker.fit(X_train_scaled, y, groups)
        pred = stacker.predict(X_test_scaled)
        pred = np.clip(pred, 0, 1)
        predictions[target] = pred

    # Create submission
    sub_num = get_next_submission_number()
    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions['angle'],
        'scaled_depth': predictions['depth'],
        'scaled_left_right': predictions['left_right']
    })

    # Calibrate depth
    submission['scaled_depth'] = submission['scaled_depth'] - submission['scaled_depth'].mean() + 0.5055
    submission['scaled_depth'] = np.clip(submission['scaled_depth'], 0, 1)

    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    submission.to_csv(sub_path, index=False)

    print(f"\nSubmission saved: {sub_path}")
    print(f"  angle_std: {submission['scaled_angle'].std():.6f}")
    print(f"  depth_mean: {submission['scaled_depth'].mean():.6f}")


if __name__ == "__main__":
    main()
