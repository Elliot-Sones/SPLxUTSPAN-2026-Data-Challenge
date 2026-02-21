"""
Kolmogorov-Arnold Networks (KANs) for Basketball Free Throw Prediction

KANs replace fixed activation functions with learnable spline-based functions.
Potential advantages for small data:
1. More parameter efficient than MLPs
2. Learnable nonlinear transformations
3. Better interpretability

This script tests multiple KAN configurations with proper GroupKFold CV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import sys
import subprocess

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
SUBMISSION_DIR = PROJECT_DIR / "submission"

from data_loader import iterate_shots, get_keypoint_columns

# Install pykan if not available
try:
    from kan import KAN
    KAN_AVAILABLE = True
except ImportError:
    print("Installing pykan...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pykan", "-q"])
    try:
        from kan import KAN
        KAN_AVAILABLE = True
    except ImportError:
        print("pykan installation failed. Trying efficient-kan...")
        subprocess.run([sys.executable, "-m", "pip", "install", "efficient-kan", "-q"])
        try:
            from efficient_kan import KAN
            KAN_AVAILABLE = True
        except ImportError:
            KAN_AVAILABLE = False
            print("KAN not available.")


def get_keypoint_map(keypoint_cols):
    keypoints = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoints.append(col[:-2])
    return {kp: i for i, kp in enumerate(keypoints)}


def extract_features(timeseries, kp_idx):
    """Extract comprehensive features for KAN input."""
    features = {}

    key_joints = [
        'right_wrist', 'right_elbow', 'right_shoulder',
        'left_wrist', 'mid_hip', 'right_knee', 'right_ankle',
        'neck', 'left_shoulder'
    ]

    # Detect key frames
    if 'right_wrist' in kp_idx:
        i = kp_idx['right_wrist']
        wrist_z = timeseries[:, i*3 + 2]
        wrist_vz = np.diff(wrist_z) * 60

        try:
            release_frame = np.nanargmax(wrist_vz[80:160]) + 80
        except:
            release_frame = 120

        try:
            set_point_frame = np.nanargmax(wrist_z[60:release_frame]) + 60
        except:
            set_point_frame = release_frame - 10

        try:
            dip_frame = np.nanargmin(wrist_z[40:set_point_frame]) + 40
        except:
            dip_frame = 80
    else:
        dip_frame, set_point_frame, release_frame = 80, 100, 120

    features['release_frame'] = release_frame
    features['set_point_frame'] = set_point_frame
    features['dip_frame'] = dip_frame

    # Extract features for each joint
    for joint in key_joints:
        if joint not in kp_idx:
            continue

        i = kp_idx[joint]
        x = timeseries[:, i*3]
        y = timeseries[:, i*3 + 1]
        z = timeseries[:, i*3 + 2]

        # Statistics
        features[f'{joint}_z_mean'] = np.nanmean(z)
        features[f'{joint}_z_std'] = np.nanstd(z)
        features[f'{joint}_z_max'] = np.nanmax(z)
        features[f'{joint}_z_range'] = np.nanmax(z) - np.nanmin(z)

        # Values at key frames
        for frame_name, frame in [('dip', dip_frame), ('set', set_point_frame), ('release', release_frame)]:
            if frame < len(z):
                features[f'{joint}_z_{frame_name}'] = z[frame]
                features[f'{joint}_x_{frame_name}'] = x[frame]
                features[f'{joint}_y_{frame_name}'] = y[frame]

        # Velocities
        vz = np.diff(z) * 60
        vx = np.diff(x) * 60
        vy = np.diff(y) * 60
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        features[f'{joint}_vz_max'] = np.nanmax(vz)
        features[f'{joint}_speed_max'] = np.nanmax(speed)

        if len(vz) > release_frame:
            features[f'{joint}_vz_at_release'] = vz[min(release_frame, len(vz)-1)]

    return features


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
    print("KOLMOGOROV-ARNOLD NETWORKS (KANs) EXPERIMENT")
    print("=" * 80)

    if not KAN_AVAILABLE:
        print("\nKAN library not available. Testing with alternative approach...")
        print("Using B-spline feature expansion + Ridge as KAN approximation.")

    keypoint_cols = get_keypoint_columns()
    kp_idx = get_keypoint_map(keypoint_cols)

    # Extract features
    print("\nExtracting features...")
    train_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_features(timeseries, kp_idx)
        features['player_id'] = metadata['participant_id']
        features['angle'] = metadata['angle']
        features['depth'] = metadata['depth']
        features['left_right'] = metadata['left_right']
        train_data.append(features)

    train_df = pd.DataFrame(train_data)
    print(f"  Training samples: {len(train_df)}")

    exclude_cols = ['player_id', 'angle', 'depth', 'left_right']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"  Features: {len(feature_cols)}")

    X = train_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)

    groups = train_df['player_id'].values

    # Scale targets
    targets = ['angle', 'depth', 'left_right']
    y_data = {}
    target_stats = {}
    for t in targets:
        vals = train_df[t].values
        target_stats[t] = {'min': vals.min(), 'max': vals.max()}
        y_data[t] = (vals - vals.min()) / (vals.max() - vals.min())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=5)

    # =========================================================================
    # EXPERIMENT 1: B-Spline Feature Expansion (KAN-like)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: B-SPLINE FEATURE EXPANSION")
    print("=" * 80)
    print("Approximating KAN's learnable splines with explicit B-spline basis.")

    from scipy.interpolate import BSpline

    def create_bspline_features(X, n_knots=5, degree=3):
        """Create B-spline basis expansion for each feature."""
        n_samples, n_features = X.shape
        expanded_features = []

        for j in range(n_features):
            col = X[:, j]
            col_min, col_max = col.min(), col.max()

            if col_max - col_min < 1e-6:
                # No variance, just keep original
                expanded_features.append(col.reshape(-1, 1))
                continue

            # Create knots
            knots = np.linspace(col_min, col_max, n_knots)
            # Add boundary knots for B-spline
            knots = np.concatenate([
                [col_min] * degree,
                knots,
                [col_max] * degree
            ])

            # Create B-spline basis functions
            n_basis = len(knots) - degree - 1
            basis_matrix = np.zeros((n_samples, n_basis))

            for i in range(n_basis):
                coef = np.zeros(n_basis)
                coef[i] = 1
                spline = BSpline(knots, coef, degree)
                basis_matrix[:, i] = spline(col)

            expanded_features.append(basis_matrix)

        return np.hstack(expanded_features)

    # Test different spline configurations
    configs = [
        {'n_knots': 3, 'degree': 2, 'alpha': 100.0},
        {'n_knots': 5, 'degree': 3, 'alpha': 100.0},
        {'n_knots': 7, 'degree': 3, 'alpha': 500.0},
    ]

    best_config = None
    best_total_mse = float('inf')

    for config in configs:
        print(f"\n  Config: n_knots={config['n_knots']}, degree={config['degree']}, alpha={config['alpha']}")

        results = {}
        for target in targets:
            y = y_data[target]
            mses = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                # Create spline features
                X_tr_spline = create_bspline_features(X_tr, config['n_knots'], config['degree'])
                X_val_spline = create_bspline_features(X_val, config['n_knots'], config['degree'])

                model = Ridge(alpha=config['alpha'])
                model.fit(X_tr_spline, y_tr)
                pred = model.predict(X_val_spline)
                pred = np.clip(pred, 0, 1)

                mse = mean_squared_error(y_val, pred)
                mses.append(mse)

            results[target] = np.mean(mses)
            print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

        total_mse = sum(results.values()) / 3
        print(f"    Total: {total_mse:.6f}")

        if total_mse < best_total_mse:
            best_total_mse = total_mse
            best_config = config

    print(f"\n  Best config: {best_config}, Total MSE: {best_total_mse:.6f}")

    # =========================================================================
    # EXPERIMENT 2: Polynomial Feature Expansion
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: POLYNOMIAL FEATURE EXPANSION")
    print("=" * 80)

    from sklearn.preprocessing import PolynomialFeatures

    for degree in [2, 3]:
        print(f"\n  Polynomial degree: {degree}")

        # Use subset of features to avoid explosion
        n_top_features = 20
        feature_importance = np.abs(X_scaled).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-n_top_features:]
        X_subset = X_scaled[:, top_indices]

        poly = PolynomialFeatures(degree=degree, include_bias=False)

        results = {}
        for target in targets:
            y = y_data[target]
            mses = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_subset, y, groups)):
                X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                X_tr_poly = poly.fit_transform(X_tr)
                X_val_poly = poly.transform(X_val)

                model = Ridge(alpha=100.0)
                model.fit(X_tr_poly, y_tr)
                pred = model.predict(X_val_poly)
                pred = np.clip(pred, 0, 1)

                mse = mean_squared_error(y_val, pred)
                mses.append(mse)

            results[target] = np.mean(mses)
            print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

        print(f"    Total: {sum(results.values())/3:.6f}")

    # =========================================================================
    # EXPERIMENT 3: RBF Kernel Expansion
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: RBF KERNEL EXPANSION")
    print("=" * 80)

    from sklearn.kernel_approximation import RBFSampler

    for n_components in [50, 100, 200]:
        print(f"\n  RBF components: {n_components}")

        rbf = RBFSampler(n_components=n_components, gamma=0.1, random_state=42)

        results = {}
        for target in targets:
            y = y_data[target]
            mses = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                X_tr_rbf = rbf.fit_transform(X_tr)
                X_val_rbf = rbf.transform(X_val)

                model = Ridge(alpha=10.0)
                model.fit(X_tr_rbf, y_tr)
                pred = model.predict(X_val_rbf)
                pred = np.clip(pred, 0, 1)

                mse = mean_squared_error(y_val, pred)
                mses.append(mse)

            results[target] = np.mean(mses)
            print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

        print(f"    Total: {sum(results.values())/3:.6f}")

    # =========================================================================
    # EXPERIMENT 4: Actual KAN (if available)
    # =========================================================================
    if KAN_AVAILABLE:
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: ACTUAL KAN NETWORK")
        print("=" * 80)

        import torch

        # Use subset of features
        n_features_kan = 20
        top_indices = np.argsort(np.abs(X_scaled).mean(axis=0))[-n_features_kan:]
        X_kan = X_scaled[:, top_indices]

        for width in [[n_features_kan, 8, 1], [n_features_kan, 16, 8, 1]]:
            print(f"\n  KAN architecture: {width}")

            results = {}
            for target in targets:
                y = y_data[target]
                mses = []

                for fold, (train_idx, val_idx) in enumerate(gkf.split(X_kan, y, groups)):
                    X_tr = torch.FloatTensor(X_kan[train_idx])
                    X_val = torch.FloatTensor(X_kan[val_idx])
                    y_tr = torch.FloatTensor(y[train_idx]).reshape(-1, 1)
                    y_val = y[val_idx]

                    try:
                        model = KAN(width=width, grid=3, k=3)
                        model.train()

                        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                        loss_fn = torch.nn.MSELoss()

                        for epoch in range(100):
                            optimizer.zero_grad()
                            pred = model(X_tr)
                            loss = loss_fn(pred, y_tr)
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            pred = model(X_val).numpy().flatten()
                            pred = np.clip(pred, 0, 1)

                        mse = mean_squared_error(y_val, pred)
                        mses.append(mse)
                    except Exception as e:
                        print(f"      Fold {fold+1} failed: {e}")
                        mses.append(1.0)

                if mses:
                    results[target] = np.mean(mses)
                    print(f"    {target}: CV MSE = {np.mean(mses):.6f}")

            if results:
                print(f"    Total: {sum(results.values())/3:.6f}")

    # =========================================================================
    # CREATE SUBMISSIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CREATING SUBMISSIONS")
    print("=" * 80)

    # Extract test features
    test_data = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_features(timeseries, kp_idx)
        test_data.append(features)
        test_ids.append(metadata['id'])

    test_df = pd.DataFrame(test_data)
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test = test_df[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0)
    X_test_scaled = scaler.transform(X_test)

    # Use best B-spline config
    if best_config:
        X_train_spline = create_bspline_features(X_scaled, best_config['n_knots'], best_config['degree'])
        X_test_spline = create_bspline_features(X_test_scaled, best_config['n_knots'], best_config['degree'])

        predictions = {}
        for target in targets:
            model = Ridge(alpha=best_config['alpha'])
            model.fit(X_train_spline, y_data[target])
            pred = model.predict(X_test_spline)
            predictions[target] = np.clip(pred, 0, 1)

        sub_num = get_next_submission_number()

        # Pure B-spline submission
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
        print(f"  Sub {sub_num}: B-spline KAN-like, angle_std={submission['scaled_angle'].std():.4f}")
        sub_num += 1

        # Blend with Sub219
        sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
        sub219_indexed = sub219.set_index('id')

        for weight in [0.1, 0.2, 0.3]:
            blended = pd.DataFrame({'id': test_ids})

            for target, col in [('angle', 'scaled_angle'), ('depth', 'scaled_depth'), ('left_right', 'scaled_left_right')]:
                kan_pred = predictions[target]
                sub219_pred = sub219_indexed.loc[test_ids, col].values
                blended[col] = weight * kan_pred + (1 - weight) * sub219_pred

            blended['scaled_depth'] = blended['scaled_depth'] - blended['scaled_depth'].mean() + 0.5055
            for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
                blended[col] = blended[col].clip(0, 1)

            sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            blended.to_csv(sub_path, index=False)
            print(f"  Sub {sub_num}: {int(weight*100)}% B-spline + {int((1-weight)*100)}% Sub219")
            sub_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("KAN-like approaches tested:")
    print("  1. B-spline feature expansion - learnable nonlinear basis")
    print("  2. Polynomial features - explicit nonlinear interactions")
    print("  3. RBF kernel approximation - kernel-based nonlinearity")
    if KAN_AVAILABLE:
        print("  4. Actual KAN network")


if __name__ == "__main__":
    main()
