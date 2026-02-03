"""
NBA-Regularized Model

Instead of adding NBA features (which adds variance), use NBA data
to constrain/regularize predictions:
1. Shots with extreme estimated release parameters should be penalized
2. Predictions should be consistent with physical constraints

Key insight: NBA data tells us what's physically POSSIBLE, not what's OPTIMAL.
Use it to reject implausible predictions rather than guide toward "optimal".
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


def load_nba_bounds():
    """Load NBA data and compute physical bounds (what's possible)."""
    player_df = pd.read_csv(EXTERNAL_DIR / "player_metrics.csv")
    player_df = player_df.dropna(subset=['rv', 'rz', 'rvz'])

    # Use percentiles to define "plausible" ranges
    return {
        'rv_min': player_df['rv'].quantile(0.01),
        'rv_max': player_df['rv'].quantile(0.99),
        'rz_min': player_df['rz'].quantile(0.01),
        'rz_max': player_df['rz'].quantile(0.99),
        'rvz_min': player_df['rvz'].quantile(0.01),
        'rvz_max': player_df['rvz'].quantile(0.99),
    }


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def estimate_release_params(timeseries, keypoint_idx):
    """Estimate physical release parameters from body pose."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Frames around release
    f_pre = 150
    f_rel = 153

    wrist_pre = get_joint("right_wrist", f_pre)
    wrist_rel = get_joint("right_wrist", f_rel)

    params = {}

    if np.any(wrist_pre) and np.any(wrist_rel):
        dt = 3 / 60  # 3 frames at 60 fps
        velocity = (wrist_rel - wrist_pre) / dt
        params['est_rv'] = np.linalg.norm(velocity)
        params['est_rvz'] = velocity[2]

    if np.any(wrist_rel):
        params['est_rz'] = wrist_rel[2]

    return params


def compute_plausibility_score(params, bounds):
    """
    Compute how plausible the estimated release parameters are.
    Higher score = more plausible (within NBA observed range).
    """
    score = 0.0
    count = 0

    for param, (bound_min, bound_max) in [
        ('est_rv', (bounds['rv_min'], bounds['rv_max'])),
        ('est_rvz', (bounds['rvz_min'], bounds['rvz_max'])),
        ('est_rz', (bounds['rz_min'], bounds['rz_max'])),
    ]:
        if param in params:
            val = params[param]
            if bound_min <= val <= bound_max:
                # Within range - full score
                score += 1.0
            else:
                # Outside range - penalize based on distance
                if val < bound_min:
                    dist = (bound_min - val) / (bound_max - bound_min)
                else:
                    dist = (val - bound_max) / (bound_max - bound_min)
                score += max(0, 1 - dist)
            count += 1

    return score / max(count, 1)


def extract_features(timeseries, keypoint_idx, bounds):
    """Extract features with plausibility score."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Compute plausibility score
    params = estimate_release_params(timeseries, keypoint_idx)
    features['plausibility'] = compute_plausibility_score(params, bounds)

    # Standard position features (proven to work)
    for joint in ["right_wrist", "right_elbow", "right_shoulder"]:
        for f in [150, 153, 155]:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    # Add left side for balance
    for joint in ["left_wrist", "left_elbow", "left_shoulder"]:
        for f in [150, 153, 155]:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    # Core/stability features
    for joint in ["mid_hip", "neck"]:
        for f in [150, 153]:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    return features


def main():
    print("="*80)
    print("NBA-REGULARIZED MODEL")
    print("="*80)

    # Load NBA bounds
    bounds = load_nba_bounds()
    print("\nNBA Physical Bounds (1st-99th percentile):")
    print(f"  Release velocity: {bounds['rv_min']:.2f} - {bounds['rv_max']:.2f}")
    print(f"  Vertical velocity: {bounds['rvz_min']:.2f} - {bounds['rvz_max']:.2f}")
    print(f"  Release height: {bounds['rz_min']:.2f} - {bounds['rz_max']:.2f}")

    # Load data
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []
    train_ids = []

    print("\nLoading training data...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_features(timeseries, keypoint_idx, bounds)
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
        features = extract_features(timeseries, keypoint_idx, bounds)
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

    # Check plausibility distribution
    plaus_train = X_train['plausibility'].values
    plaus_test = X_test['plausibility'].values
    print(f"\nPlausibility scores:")
    print(f"  Train: {plaus_train.mean():.3f} +/- {plaus_train.std():.3f}")
    print(f"  Test:  {plaus_test.mean():.3f} +/- {plaus_test.std():.3f}")

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

    # Save base model
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

    # Blend with Sub 219 (conservative - use Sub 219 as anchor)
    print("\nBlending with Sub 219 (conservative)...")
    for w in [0.05, 0.10, 0.15]:
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
        print(f"  {w:.0%} new + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
