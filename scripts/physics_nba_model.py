"""
Physics-Based Model Using NBA Insights

Use NBA data to create physics-informed features:
1. Estimate ball release parameters from body pose
2. Compare to NBA optimal release patterns
3. Use physics to predict shot outcome

Key physics:
- Ball leaves hand with velocity determined by arm speed
- Release angle = arm angle at release
- Higher release = better angle to basket
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
EXTERNAL_DIR = PROJECT_DIR / "external_data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def load_nba_reference():
    """Load NBA data and compute reference statistics."""
    player_df = pd.read_csv(EXTERNAL_DIR / "player_metrics.csv")
    player_df = player_df.dropna(subset=['rv', 'rz', 'rvz', 'rvx'])

    return {
        # Release velocity components
        'rv_mean': player_df['rv'].mean(),
        'rv_std': player_df['rv'].std(),
        'rvx_mean': player_df['rvx'].mean(),  # Horizontal
        'rvx_std': player_df['rvx'].std(),
        'rvy_mean': player_df['rvy'].mean(),  # Lateral
        'rvy_std': player_df['rvy'].std(),
        'rvz_mean': player_df['rvz'].mean(),  # Vertical
        'rvz_std': player_df['rvz'].std(),

        # Release position
        'rz_mean': player_df['rz'].mean(),  # Height
        'rz_std': player_df['rz'].std(),

        # Percentiles for normalization
        'rv_p25': player_df['rv'].quantile(0.25),
        'rv_p75': player_df['rv'].quantile(0.75),
    }


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def estimate_ball_release(timeseries, keypoint_idx):
    """
    Estimate ball release parameters from body pose.

    Physics model:
    - Ball velocity ≈ wrist velocity at release
    - Release angle = angle of arm at release
    - Release height = wrist height at release
    """
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Frames: pre-release, release, post-release
    f_pre = 150
    f_rel = 153
    f_post = 156

    # Get key positions
    wrist_pre = get_joint("right_wrist", f_pre)
    wrist_rel = get_joint("right_wrist", f_rel)
    wrist_post = get_joint("right_wrist", f_post)
    elbow_rel = get_joint("right_elbow", f_rel)
    shoulder_rel = get_joint("right_shoulder", f_rel)

    estimates = {}

    # 1. Release velocity (from wrist motion)
    if np.any(wrist_pre) and np.any(wrist_rel):
        dt = 3 / 60  # 3 frames at 60 fps
        velocity = (wrist_rel - wrist_pre) / dt

        estimates['est_rv'] = np.linalg.norm(velocity)
        estimates['est_rvx'] = velocity[0]
        estimates['est_rvy'] = velocity[1]
        estimates['est_rvz'] = velocity[2]

    # 2. Release height
    if np.any(wrist_rel):
        estimates['est_rz'] = wrist_rel[2]

    # 3. Release angle (arm angle relative to vertical)
    if np.any(shoulder_rel) and np.any(wrist_rel):
        arm_vector = wrist_rel - shoulder_rel
        vertical = np.array([0, 0, 1])
        cos_angle = np.dot(arm_vector, vertical) / (np.linalg.norm(arm_vector) + 1e-8)
        estimates['release_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # 4. Elbow angle at release
    if np.any(shoulder_rel) and np.any(elbow_rel) and np.any(wrist_rel):
        v1 = shoulder_rel - elbow_rel
        v2 = wrist_rel - elbow_rel
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        estimates['elbow_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    return estimates


def create_physics_features(timeseries, keypoint_idx, nba_ref):
    """Create features based on physics and NBA reference."""
    features = {}

    # Get ball release estimates
    estimates = estimate_ball_release(timeseries, keypoint_idx)
    features.update(estimates)

    # Compare to NBA reference
    if 'est_rv' in estimates and nba_ref:
        # Z-score relative to NBA
        features['rv_zscore'] = (estimates['est_rv'] - nba_ref['rv_mean']) / (nba_ref['rv_std'] + 1e-8)
        features['rvz_zscore'] = (estimates.get('est_rvz', 0) - nba_ref['rvz_mean']) / (nba_ref['rvz_std'] + 1e-8)
        features['rz_zscore'] = (estimates.get('est_rz', 0) - nba_ref['rz_mean']) / (nba_ref['rz_std'] + 1e-8)

        # Is the shot in NBA "normal" range?
        features['rv_in_iqr'] = 1 if nba_ref['rv_p25'] <= estimates['est_rv'] <= nba_ref['rv_p75'] else 0

    # Raw position features (for baseline)
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    for joint in ["right_wrist", "right_elbow", "right_shoulder"]:
        for f in [150, 153, 155]:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    return features


def main():
    print("="*80)
    print("PHYSICS-BASED MODEL WITH NBA REFERENCE")
    print("="*80)

    # Load NBA reference
    nba_ref = load_nba_reference()
    print("\nNBA Reference Statistics:")
    print(f"  Release velocity: {nba_ref['rv_mean']:.2f} +/- {nba_ref['rv_std']:.2f}")
    print(f"  Vertical velocity: {nba_ref['rvz_mean']:.2f} +/- {nba_ref['rvz_std']:.2f}")
    print(f"  Release height: {nba_ref['rz_mean']:.2f} +/- {nba_ref['rz_std']:.2f}")

    # Load data
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []
    train_ids = []

    print("\nLoading training data...")
    for metadata, timeseries in iterate_shots(train=True):
        features = create_physics_features(timeseries, keypoint_idx, nba_ref)
        train_features.append(features)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])
        train_ids.append(metadata['shot_id'])

    test_features = []
    test_ids = []

    print("Loading test data...")
    for metadata, timeseries in iterate_shots(train=False):
        features = create_physics_features(timeseries, keypoint_idx, nba_ref)
        test_features.append(features)
        test_ids.append(metadata['id'])

    print(f"\nTrain: {len(train_features)}, Test: {len(test_features)}")

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Features: {len(common_cols)}")
    print(f"Physics features: {len([c for c in common_cols if c.startswith('est_') or c.endswith('_zscore')])}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    print("\nTraining model...")
    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

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
        print(f"  {target} CV MSE: {cv_score:.6f}")

        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

    overall_cv = np.mean(cv_scores)
    print(f"\nOverall CV MSE: {overall_cv:.6f}")

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

    # Blend
    print("\nBlending with Sub 219...")
    for w in [0.1, 0.2, 0.3]:
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
