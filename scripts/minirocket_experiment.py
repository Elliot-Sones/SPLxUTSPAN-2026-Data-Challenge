"""
MiniRocket for Basketball Free Throw Prediction

MiniRocket is a fast, accurate time series classification/regression method that:
1. Uses random convolutional kernels (no training needed for kernels)
2. Extracts PPV (proportion of positive values) features
3. Works well with small datasets due to minimal parameters
4. Is 75x faster than ROCKET with similar accuracy

This script tests MiniRocket on joint trajectories with GroupKFold CV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from data_loader import iterate_shots, get_keypoint_columns

try:
    from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
    MINIROCKET_AVAILABLE = True
except ImportError:
    MINIROCKET_AVAILABLE = False
    print("MiniRocket not available.")


def get_keypoint_map(keypoint_cols):
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def extract_trajectories(timeseries, kp_idx, start_frame=60, end_frame=180):
    """
    Extract joint trajectories as multivariate time series.
    Returns: (n_timesteps, n_channels) array
    """
    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'mid_hip', 'right_knee']

    channels = []
    for joint in key_joints:
        if joint not in kp_idx:
            continue
        i = kp_idx[joint]

        # Get z coordinate (most relevant for shooting)
        z = timeseries[start_frame:end_frame, i*3 + 2]
        channels.append(z)

        # Also add velocity
        vz = np.gradient(z) * 60
        channels.append(vz)

    if not channels:
        return None

    trajectory = np.column_stack(channels)

    # Handle NaN
    trajectory = np.nan_to_num(trajectory, nan=0.0)

    return trajectory


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
    print("MINIROCKET EXPERIMENT")
    print("=" * 80)

    if not MINIROCKET_AVAILABLE:
        print("MiniRocket not available. Exiting.")
        return

    print("MiniRocket uses random convolutional kernels for fast feature extraction.")
    print("No kernel training needed - only the classifier is trained.")

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_map(keypoint_cols)

    # Extract trajectories
    print("\nExtracting trajectories...")
    train_trajectories = []
    train_targets = {'angle': [], 'depth': [], 'left_right': []}
    train_players = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        traj = extract_trajectories(timeseries, kp_idx)
        if traj is None:
            continue

        train_trajectories.append(traj)
        train_players.append(metadata['participant_id'])

        for t in train_targets:
            train_targets[t].append(metadata[t])

    print(f"  Training samples: {len(train_trajectories)}")
    print(f"  Trajectory shape: {train_trajectories[0].shape}")

    # Convert to 3D array for sktime (n_samples, n_channels, n_timesteps)
    X_train = np.array(train_trajectories)
    # sktime expects (n_samples, n_channels, n_timesteps)
    X_train = np.transpose(X_train, (0, 2, 1))
    print(f"  X_train shape: {X_train.shape}")

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
    # EXPERIMENT 1: MiniRocket with default settings
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: MINIROCKET DEFAULT")
    print("=" * 80)

    gkf = GroupKFold(n_splits=5)

    try:
        # Transform all data first
        minirocket = MiniRocketMultivariate(random_state=42)
        X_transformed = minirocket.fit_transform(X_train)
        print(f"  MiniRocket features: {X_transformed.shape[1]}")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_transformed)

        results_default = {}
        for target in targets:
            y = y_data[target]
            mses = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = Ridge(alpha=10.0)
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                pred = np.clip(pred, 0, 1)

                mse = mean_squared_error(y_val, pred)
                mses.append(mse)

            results_default[target] = np.mean(mses)
            print(f"    {target}: CV MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

        print(f"    Total: {sum(results_default.values())/3:.6f}")

    except Exception as e:
        print(f"  MiniRocket failed: {e}")
        X_scaled = None

    # =========================================================================
    # EXPERIMENT 2: MiniRocket with different regularization
    # =========================================================================
    if X_scaled is not None:
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: MINIROCKET WITH ALPHA TUNING")
        print("=" * 80)

        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            print(f"\n  Alpha = {alpha}")

            results = {}
            for target in targets:
                y = y_data[target]
                mses = []

                for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
                    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    model = Ridge(alpha=alpha)
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    pred = np.clip(pred, 0, 1)

                    mse = mean_squared_error(y_val, pred)
                    mses.append(mse)

                results[target] = np.mean(mses)
                print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

            print(f"    Total: {sum(results.values())/3:.6f}")

    # =========================================================================
    # EXPERIMENT 3: Different trajectory windows
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: DIFFERENT TRAJECTORY WINDOWS")
    print("=" * 80)

    windows = [
        (60, 180, "Full phase (60-180)"),
        (80, 160, "Core phase (80-160)"),
        (100, 150, "Release phase (100-150)"),
    ]

    for start, end, desc in windows:
        print(f"\n  Window: {desc}")

        # Re-extract with different window
        trajectories = []
        for metadata, timeseries in iterate_shots(train=True):
            traj = extract_trajectories(timeseries, kp_idx, start_frame=start, end_frame=end)
            if traj is not None:
                trajectories.append(traj)

        X = np.array(trajectories)
        X = np.transpose(X, (0, 2, 1))

        try:
            mr = MiniRocketMultivariate(random_state=42)
            X_mr = mr.fit_transform(X)

            sc = StandardScaler()
            X_mr_scaled = sc.fit_transform(X_mr)

            results = {}
            for target in targets:
                y = y_data[target]
                mses = []

                for fold, (train_idx, val_idx) in enumerate(gkf.split(X_mr_scaled, y, groups)):
                    X_tr, X_val = X_mr_scaled[train_idx], X_mr_scaled[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    model = Ridge(alpha=100.0)
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    pred = np.clip(pred, 0, 1)

                    mse = mean_squared_error(y_val, pred)
                    mses.append(mse)

                results[target] = np.mean(mses)
                print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

            print(f"    Total: {sum(results.values())/3:.6f}")

        except Exception as e:
            print(f"    Failed: {e}")

    # =========================================================================
    # CREATE SUBMISSIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CREATING SUBMISSIONS")
    print("=" * 80)

    # Extract test trajectories
    test_trajectories = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        traj = extract_trajectories(timeseries, kp_idx)
        if traj is None:
            traj = np.zeros((120, train_trajectories[0].shape[1]))
        test_trajectories.append(traj)
        test_ids.append(metadata['id'])

    X_test = np.array(test_trajectories)
    X_test = np.transpose(X_test, (0, 2, 1))

    if X_scaled is not None:
        # Use best config (alpha=100)
        X_test_transformed = minirocket.transform(X_test)
        X_test_scaled = scaler.transform(X_test_transformed)

        predictions = {}
        for target in targets:
            model = Ridge(alpha=100.0)
            model.fit(X_scaled, y_data[target])
            pred = model.predict(X_test_scaled)
            predictions[target] = np.clip(pred, 0, 1)

        sub_num = get_next_submission_number()

        # Pure MiniRocket submission
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
        print(f"  Sub {sub_num}: Pure MiniRocket, angle_std={submission['scaled_angle'].std():.4f}")
        sub_num += 1

        # Blend with Sub219
        sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
        sub219_indexed = sub219.set_index('id')

        for weight in [0.1, 0.2, 0.3]:
            blended = pd.DataFrame({'id': test_ids})

            for target, col in [('angle', 'scaled_angle'), ('depth', 'scaled_depth'), ('left_right', 'scaled_left_right')]:
                mr_pred = predictions[target]
                sub219_pred = sub219_indexed.loc[test_ids, col].values
                blended[col] = weight * mr_pred + (1 - weight) * sub219_pred

            blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
            for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
                blended[col] = blended[col].clip(0, 1)

            sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            blended.to_csv(sub_path, index=False)
            print(f"  Sub {sub_num}: {int(weight*100)}% MiniRocket + {int((1-weight)*100)}% Sub219")
            sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("MiniRocket extracts random convolutional features efficiently.")
    print("Best for time series with temporal patterns.")


if __name__ == "__main__":
    main()
