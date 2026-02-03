"""
Launch Angle Focused Model

The launch angle (arctan(vz/vx)) has -0.61 correlation with angle target.
This is the strongest signal we've found from NBA-inspired features.

Test:
1. Different ways to compute launch angle
2. Launch angle + minimal other features
3. Nonlinear transformations
4. Per-player launch angle normalization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import GroupKFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def compute_launch_angles(timeseries, keypoint_idx):
    """Compute launch angle in multiple ways."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}

    # Method 1: Basic wrist velocity (150->153)
    wrist_150 = get_joint("right_wrist", 150)
    wrist_153 = get_joint("right_wrist", 153)

    if np.any(wrist_150) and np.any(wrist_153):
        vel = wrist_153 - wrist_150  # Don't need dt for angle
        if abs(vel[0]) > 0.001:
            features['la_basic'] = np.degrees(np.arctan2(vel[2], abs(vel[0])))

        # Also store components
        features['vel_vx'] = vel[0]
        features['vel_vy'] = vel[1]
        features['vel_vz'] = vel[2]
        features['vel_speed'] = np.linalg.norm(vel)

    # Method 2: Multi-frame average (smoother)
    angles = []
    for f_start, f_end in [(149, 152), (150, 153), (151, 154), (152, 155)]:
        w_start = get_joint("right_wrist", f_start)
        w_end = get_joint("right_wrist", f_end)
        if np.any(w_start) and np.any(w_end):
            v = w_end - w_start
            if abs(v[0]) > 0.001:
                angles.append(np.degrees(np.arctan2(v[2], abs(v[0]))))

    if angles:
        features['la_avg'] = np.mean(angles)
        features['la_std'] = np.std(angles)  # Consistency

    # Method 3: Using elbow velocity (different joint)
    elbow_150 = get_joint("right_elbow", 150)
    elbow_153 = get_joint("right_elbow", 153)

    if np.any(elbow_150) and np.any(elbow_153):
        vel = elbow_153 - elbow_150
        if abs(vel[0]) > 0.001:
            features['la_elbow'] = np.degrees(np.arctan2(vel[2], abs(vel[0])))

    # Method 4: Arm angle (static, not velocity)
    shoulder = get_joint("right_shoulder", 153)
    wrist = get_joint("right_wrist", 153)

    if np.any(shoulder) and np.any(wrist):
        arm_vec = wrist - shoulder
        if np.linalg.norm(arm_vec[:2]) > 0.001:
            features['arm_angle'] = np.degrees(np.arctan2(arm_vec[2], np.linalg.norm(arm_vec[:2])))

    # Method 5: Fingertip approximation (wrist direction)
    wrist_156 = get_joint("right_wrist", 156)
    if np.any(wrist_153) and np.any(wrist_156):
        follow = wrist_156 - wrist_153
        if abs(follow[0]) > 0.001:
            features['la_follow'] = np.degrees(np.arctan2(follow[2], abs(follow[0])))

    # Additional features that might help
    # Height at release
    if np.any(wrist_153):
        features['release_height'] = wrist_153[2]

    # Wrist above shoulder
    if np.any(wrist_153) and np.any(shoulder):
        features['wrist_above_shoulder'] = wrist_153[2] - shoulder[2]

    return features


def extract_minimal_features(timeseries, keypoint_idx):
    """Extract just launch angle + minimal supporting features."""
    features = compute_launch_angles(timeseries, keypoint_idx)

    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Key positions at frame 153 (release)
    for joint in ["right_wrist", "right_shoulder"]:
        pos = get_joint(joint, 153)
        features[f'{joint}_z_153'] = pos[2]

    return features


def main():
    print("="*80)
    print("LAUNCH ANGLE FOCUSED MODEL")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []

    print("\nExtracting features...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_minimal_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])

    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_minimal_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Features: {len(common_cols)}")
    print(f"Feature names: {common_cols}")

    # Correlations
    print("\nFeature correlations with angle:")
    for col in sorted(common_cols):
        corr = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
        print(f"  {col}: {corr:.4f}")

    # Test different model configurations
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    configs = [
        ('Ridge alpha=1', Ridge(alpha=1.0)),
        ('Ridge alpha=10', Ridge(alpha=10.0)),
        ('Ridge alpha=100', Ridge(alpha=100.0)),
        ('Huber', HuberRegressor(epsilon=1.35)),
    ]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_config = None
    best_cv = float('inf')

    for name, model_template in configs:
        cv_scores = []
        for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
            y = y_train[:, target_idx]
            gkf = GroupKFold(n_splits=5)
            scores = []

            for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
                model = Ridge(alpha=model_template.alpha) if hasattr(model_template, 'alpha') else HuberRegressor()
                model.fit(X_train_scaled[train_idx], y[train_idx])
                pred = model.predict(X_train_scaled[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                scores.append(mse)

            cv_scores.append(np.mean(scores))

        overall = np.mean(cv_scores)
        print(f"{name}: angle={cv_scores[0]:.4f}, depth={cv_scores[1]:.4f}, lr={cv_scores[2]:.4f}, mean={overall:.4f}")

        if overall < best_cv:
            best_cv = overall
            best_config = name

    print(f"\nBest config: {best_config}")

    # Test with polynomial features
    print("\n" + "="*60)
    print("POLYNOMIAL FEATURES TEST")
    print("="*60)

    # Select just launch angle features
    la_cols = [c for c in common_cols if 'la_' in c or 'arm_angle' in c]
    print(f"Launch angle features: {la_cols}")

    if la_cols:
        X_la = X_train[la_cols].values

        for degree in [1, 2, 3]:
            if degree > 1:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly = poly.fit_transform(X_la)
            else:
                X_poly = X_la

            scaler_poly = StandardScaler()
            X_poly_scaled = scaler_poly.fit_transform(X_poly)

            gkf = GroupKFold(n_splits=5)
            scores = []

            for train_idx, val_idx in gkf.split(X_poly_scaled, y_train[:, 0], players):
                model = Ridge(alpha=10.0)
                model.fit(X_poly_scaled[train_idx], y_train[:, 0][train_idx])
                pred = model.predict(X_poly_scaled[val_idx])
                mse = np.mean((pred - y_train[:, 0][val_idx])**2)
                scores.append(mse)

            print(f"Degree {degree}: n_features={X_poly.shape[1]}, angle CV={np.mean(scores):.4f}")

    # Train final model
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)

    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, target_idx]
        gkf = GroupKFold(n_splits=5)
        fold_scores = []

        for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
            model = Ridge(alpha=10.0)
            model.fit(X_train_scaled[train_idx], y[train_idx])
            pred = model.predict(X_train_scaled[val_idx])
            mse = np.mean((pred - y[val_idx])**2)
            fold_scores.append(mse)

        cv_score = np.mean(fold_scores)
        cv_scores.append(cv_score)
        print(f"  {target_name} CV MSE: {cv_score:.6f}")

        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y)
        predictions[:, target_idx] = model.predict(X_test_scaled)

    print(f"\nOverall CV MSE: {np.mean(cv_scores):.6f}")

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    angle_std = np.std(predictions[:, 0])
    print(f"angle_std: {angle_std:.6f}")

    # Save
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': predictions[:, 0],
        'scaled_depth': predictions[:, 1],
        'scaled_left_right': predictions[:, 2]
    })

    output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    submission.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Compare with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Also compare with Sub 270 (comprehensive NBA)
    sub270 = pd.read_csv(SUBMISSION_DIR / "submission_270.csv")
    corr270 = np.corrcoef(predictions[:, 0], sub270['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 270: {corr270:.4f}")

    # Blend with Sub 219
    print("\n" + "="*60)
    print("BLENDING")
    print("="*60)

    for w in [0.10, 0.15, 0.20, 0.25]:
        blend = submission.copy()
        blend['scaled_angle'] = w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values
        blend['scaled_depth'] = w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values
        blend['scaled_left_right'] = w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

        std = blend['scaled_angle'].std()

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"  {w:.0%} launch + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")

    # Three-way blend: Launch + NBA comprehensive + Sub219
    print("\nThree-way blends:")
    sub270_preds = sub270[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values

    for w_launch, w_nba in [(0.1, 0.1), (0.15, 0.10), (0.10, 0.15), (0.15, 0.15)]:
        w_219 = 1 - w_launch - w_nba
        blend = submission.copy()
        blend['scaled_angle'] = w_launch * predictions[:, 0] + w_nba * sub270_preds[:, 0] + w_219 * sub219['scaled_angle'].values
        blend['scaled_depth'] = w_launch * predictions[:, 1] + w_nba * sub270_preds[:, 1] + w_219 * sub219['scaled_depth'].values
        blend['scaled_left_right'] = w_launch * predictions[:, 2] + w_nba * sub270_preds[:, 2] + w_219 * sub219['scaled_left_right'].values

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

        std = blend['scaled_angle'].std()

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"  {w_launch:.0%} launch + {w_nba:.0%} NBA + {w_219:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
