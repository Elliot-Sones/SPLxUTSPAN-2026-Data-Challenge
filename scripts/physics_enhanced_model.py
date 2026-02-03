"""
Physics-Enhanced Model

Combine the best physics-informed features with our existing best features.
The physics features (vertical_dominance, est_release_angle) have -0.55 correlation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)

# NBA optimal values
NBA_OPTIMAL_ANGLE = 58.84  # degrees


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_all_features(timeseries, keypoint_idx):
    """Extract comprehensive physics + position features."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # === PHYSICS FEATURES (best from calibrated model) ===

    # Multi-frame velocity estimation
    frame_pairs = [(149, 152), (150, 153), (151, 154), (152, 155)]
    velocities = []
    angles = []

    for f_start, f_end in frame_pairs:
        wrist_start = get_joint("right_wrist", f_start)
        wrist_end = get_joint("right_wrist", f_end)

        if np.any(wrist_start) and np.any(wrist_end):
            dt = (f_end - f_start) / 60.0
            vel = (wrist_end - wrist_start) / dt
            velocities.append(vel)

            horizontal_speed = np.sqrt(vel[0]**2 + vel[1]**2)
            if horizontal_speed > 0.01:
                angle = np.degrees(np.arctan2(vel[2], horizontal_speed))
                angles.append(angle)

    if velocities:
        avg_vel = np.mean(velocities, axis=0)
        features['est_vx'] = avg_vel[0]
        features['est_vy'] = avg_vel[1]
        features['est_vz'] = avg_vel[2]
        features['est_speed'] = np.linalg.norm(avg_vel)
        features['vertical_dominance'] = avg_vel[2] / (np.linalg.norm(avg_vel) + 0.01)

        if abs(avg_vel[0]) > 0.01:
            features['vz_vx_ratio'] = avg_vel[2] / abs(avg_vel[0])

    if angles:
        features['est_release_angle'] = np.mean(angles)
        features['angle_vs_optimal'] = abs(np.mean(angles) - NBA_OPTIMAL_ANGLE)

    # === POSITION FEATURES (proven to work) ===

    wrist_153 = get_joint("right_wrist", 153)
    shoulder_153 = get_joint("right_shoulder", 153)
    elbow_153 = get_joint("right_elbow", 153)

    if np.any(wrist_153):
        features['wrist_x_153'] = wrist_153[0]
        features['wrist_y_153'] = wrist_153[1]
        features['wrist_z_153'] = wrist_153[2]

    if np.any(shoulder_153):
        features['shoulder_z_153'] = shoulder_153[2]

    if np.any(wrist_153) and np.any(shoulder_153):
        features['wrist_above_shoulder'] = wrist_153[2] - shoulder_153[2]

        arm_vec = wrist_153 - shoulder_153
        features['arm_extension'] = np.linalg.norm(arm_vec)
        features['arm_vertical_angle'] = np.degrees(np.arctan2(
            np.sqrt(arm_vec[0]**2 + arm_vec[1]**2), arm_vec[2]
        ))

    # === FOLLOW-THROUGH FEATURES ===

    wrist_156 = get_joint("right_wrist", 156)
    if np.any(wrist_153) and np.any(wrist_156):
        follow = wrist_156 - wrist_153
        features['followthrough_vz'] = follow[2]
        features['followthrough_mag'] = np.linalg.norm(follow)

    # === TRAJECTORY FEATURES ===

    frames = list(range(148, 160))
    wrist_positions = [get_joint("right_wrist", f) for f in frames]
    valid = [(f, p) for f, p in zip(frames, wrist_positions) if np.any(p)]

    if len(valid) >= 5:
        heights = np.array([p[2] for f, p in valid])
        features['wrist_peak_height'] = np.max(heights)
        features['wrist_height_range'] = np.max(heights) - np.min(heights)

    # === BODY ORIENTATION ===

    left_shoulder = get_joint("left_shoulder", 153)
    right_shoulder = get_joint("right_shoulder", 153)

    if np.any(left_shoulder) and np.any(right_shoulder):
        shoulder_vec = right_shoulder - left_shoulder
        features['shoulder_rotation'] = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))

    return features


def main():
    print("="*80)
    print("PHYSICS-ENHANCED MODEL")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Extract features
    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_all_features(timeseries, keypoint_idx)
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
        features = extract_all_features(timeseries, keypoint_idx)
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

    # Correlations
    print("\nFeature correlations with angle (top 10):")
    corrs = []
    for col in common_cols:
        corr = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
        if not np.isnan(corr):
            corrs.append((col, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs[:10]:
        print(f"  {col}: {corr:.4f}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test regularization
    print("\nCV Results:")
    for alpha in [10, 50, 100, 150, 200]:
        scores = []
        for i in range(3):
            y = y_train[:, i]
            gkf = GroupKFold(n_splits=5)
            fold_scores = []
            for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled[train_idx], y[train_idx])
                pred = model.predict(X_train_scaled[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                fold_scores.append(mse)
            scores.append(np.mean(fold_scores))
        print(f"  alpha={alpha}: angle={scores[0]:.4f}, depth={scores[1]:.4f}, lr={scores[2]:.4f}, mean={np.mean(scores):.4f}")

    # Train final model
    best_alpha = 100
    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        gkf = GroupKFold(n_splits=5)
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
            model = Ridge(alpha=best_alpha)
            model.fit(X_train_scaled[train_idx], y[train_idx])
            pred = model.predict(X_train_scaled[val_idx])
            mse = np.mean((pred - y[val_idx])**2)
            fold_scores.append(mse)

        cv_score = np.mean(fold_scores)
        cv_scores.append(cv_score)

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

    print(f"\nFinal CV: angle={cv_scores[0]:.4f}, depth={cv_scores[1]:.4f}, lr={cv_scores[2]:.4f}")
    print(f"Mean CV: {np.mean(cv_scores):.6f}")

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    angle_std = np.std(predictions[:, 0])
    print(f"\nangle_std: {angle_std:.6f}")
    print(f"depth_mean: {np.mean(predictions[:, 1]):.6f}")

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

    # Compare
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    corr219 = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    corr133 = np.corrcoef(predictions[:, 0], sub133['scaled_angle'].values)[0, 1]

    print(f"\nCorrelation with Sub 219: {corr219:.4f}")
    print(f"Correlation with Sub 133: {corr133:.4f}")

    # Blend with Sub 133 (best LB)
    print("\n" + "="*60)
    print("BLENDING WITH SUB 133 (BEST LB)")
    print("="*60)

    for w in [0.05, 0.10, 0.15, 0.20]:
        blend = submission.copy()
        blend['scaled_angle'] = w * predictions[:, 0] + (1-w) * sub133['scaled_angle'].values
        blend['scaled_depth'] = w * predictions[:, 1] + (1-w) * sub133['scaled_depth'].values
        blend['scaled_left_right'] = w * predictions[:, 2] + (1-w) * sub133['scaled_left_right'].values

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

        std = blend['scaled_angle'].std()

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"  {w:.0%} physics + {1-w:.0%} Sub133: angle_std={std:.6f} -> {blend_file}")

    # Blend with Sub 219
    print("\n" + "="*60)
    print("BLENDING WITH SUB 219")
    print("="*60)

    for w in [0.10, 0.15, 0.20]:
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
        print(f"  {w:.0%} physics + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
