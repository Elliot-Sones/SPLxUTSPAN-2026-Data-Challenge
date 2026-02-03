"""
Predict Ball Velocity from Body Pose

Use NBA data to understand ball velocity patterns, then:
1. Estimate ball velocity from wrist motion in our data
2. Convert to percentiles (handles scale mismatch)
3. Use NBA patterns to create quality features

Key insight: We can't match absolute values (different scales),
but we CAN match DISTRIBUTIONS via percentile mapping.
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


def load_nba_velocity_distribution():
    """Load NBA velocity data and compute percentile mappings."""
    player_df = pd.read_csv(EXTERNAL_DIR / "player_metrics.csv")
    player_df = player_df.dropna(subset=['rv', 'rvx', 'rvz', 'plr'])

    # Quality metric: lower plr = straighter shot = better
    player_df['quality'] = 1 / (player_df['plr'] + 0.01)

    # Compute percentiles for velocity components
    percentiles = np.arange(0, 101, 5)

    nba_dist = {
        'rv_percentiles': np.percentile(player_df['rv'], percentiles),
        'rvx_percentiles': np.percentile(player_df['rvx'], percentiles),
        'rvz_percentiles': np.percentile(player_df['rvz'], percentiles),
        'percentile_values': percentiles,
    }

    # What's the quality at each velocity percentile?
    player_df['rv_pctl'] = pd.qcut(player_df['rv'], 10, labels=False, duplicates='drop')
    quality_by_pctl = player_df.groupby('rv_pctl')['quality'].mean()
    nba_dist['quality_by_velocity_pctl'] = quality_by_pctl.values

    print("NBA Velocity Distribution:")
    print(f"  Total velocity (rv): {player_df['rv'].mean():.2f} +/- {player_df['rv'].std():.2f}")
    print(f"  Horizontal (rvx): {player_df['rvx'].mean():.2f} +/- {player_df['rvx'].std():.2f}")
    print(f"  Vertical (rvz): {player_df['rvz'].mean():.2f} +/- {player_df['rvz'].std():.2f}")

    print("\nQuality by velocity percentile:")
    for i, q in enumerate(nba_dist['quality_by_velocity_pctl']):
        print(f"  Percentile {i*10}-{(i+1)*10}: quality={q:.3f}")

    return nba_dist


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def estimate_ball_velocity(timeseries, keypoint_idx):
    """
    Estimate ball velocity from body pose.

    Physics: Ball velocity at release ≈ wrist velocity + hand acceleration
    We approximate with wrist velocity over release frames.
    """
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    velocities = {}

    # Try multiple frame windows to get robust estimate
    for window_name, (f_start, f_end) in [
        ('narrow', (151, 154)),   # 3 frames around release
        ('wide', (148, 157)),     # Wider window
        ('pre', (147, 153)),      # Pre-release
        ('post', (153, 159)),     # Post-release
    ]:
        wrist_start = get_joint("right_wrist", f_start)
        wrist_end = get_joint("right_wrist", f_end)

        if np.any(wrist_start) and np.any(wrist_end):
            dt = (f_end - f_start) / 60  # frames to seconds at 60fps
            vel = (wrist_end - wrist_start) / dt

            velocities[f'{window_name}_speed'] = np.linalg.norm(vel)
            velocities[f'{window_name}_vx'] = vel[0]  # Horizontal
            velocities[f'{window_name}_vy'] = vel[1]  # Lateral
            velocities[f'{window_name}_vz'] = vel[2]  # Vertical

    # Also estimate from elbow (for comparison/validation)
    elbow_pre = get_joint("right_elbow", 150)
    elbow_rel = get_joint("right_elbow", 153)
    if np.any(elbow_pre) and np.any(elbow_rel):
        dt = 3 / 60
        vel = (elbow_rel - elbow_pre) / dt
        velocities['elbow_speed'] = np.linalg.norm(vel)

    return velocities


def map_to_nba_percentile(value, our_distribution, nba_percentiles, percentile_values):
    """
    Map our velocity value to NBA percentile space.

    1. Find what percentile our value is in OUR distribution
    2. Map that percentile to NBA velocity space
    """
    # Find our percentile
    our_pctl = np.searchsorted(our_distribution, value) / len(our_distribution) * 100
    our_pctl = np.clip(our_pctl, 0, 100)

    # Map to NBA value at that percentile
    idx = np.searchsorted(percentile_values, our_pctl)
    idx = np.clip(idx, 0, len(nba_percentiles) - 1)

    return our_pctl, nba_percentiles[idx]


def create_velocity_features(velocities, our_dist, nba_dist):
    """Create features based on velocity estimates and NBA patterns."""
    features = {}

    # Raw velocity estimates
    for key, val in velocities.items():
        features[f'est_{key}'] = val

    # Percentile-based features (handles scale mismatch)
    if 'narrow_speed' in velocities and len(our_dist['speed']) > 0:
        our_pctl, nba_equiv = map_to_nba_percentile(
            velocities['narrow_speed'],
            our_dist['speed'],
            nba_dist['rv_percentiles'],
            nba_dist['percentile_values']
        )
        features['velocity_percentile'] = our_pctl
        features['nba_equivalent_velocity'] = nba_equiv

        # Quality prediction from NBA patterns
        pctl_bin = int(our_pctl / 10)
        pctl_bin = np.clip(pctl_bin, 0, len(nba_dist['quality_by_velocity_pctl']) - 1)
        features['nba_quality_prediction'] = nba_dist['quality_by_velocity_pctl'][pctl_bin]

    # Vertical vs horizontal ratio (important for arc)
    if 'narrow_vz' in velocities and 'narrow_vx' in velocities:
        vz = velocities['narrow_vz']
        vx = velocities['narrow_vx']
        if abs(vx) > 0.01:
            features['velocity_angle'] = np.degrees(np.arctan2(vz, abs(vx)))
        features['vz_vx_ratio'] = vz / (abs(vx) + 0.01)

    # Consistency across windows (lower = more consistent release)
    if 'narrow_speed' in velocities and 'wide_speed' in velocities:
        features['velocity_consistency'] = abs(velocities['narrow_speed'] - velocities['wide_speed'])

    # Pre vs post release velocity (acceleration through release)
    if 'pre_speed' in velocities and 'post_speed' in velocities:
        features['release_acceleration'] = velocities['post_speed'] - velocities['pre_speed']

    return features


def extract_all_features(timeseries, keypoint_idx, our_dist, nba_dist):
    """Extract velocity features plus standard position features."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Velocity-based features
    velocities = estimate_ball_velocity(timeseries, keypoint_idx)
    velocity_features = create_velocity_features(velocities, our_dist, nba_dist)
    features.update(velocity_features)

    # Standard position features (proven to work)
    for joint in ["right_wrist", "right_elbow", "right_shoulder"]:
        for f in [150, 153, 155]:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    return features


def main():
    print("="*80)
    print("PREDICT BALL VELOCITY FROM BODY POSE")
    print("="*80)

    # Load NBA distribution
    nba_dist = load_nba_velocity_distribution()

    # First pass: compute OUR velocity distribution
    print("\nFirst pass: computing our velocity distribution...")
    keypoint_idx = get_keypoint_indices()

    all_velocities = []
    for metadata, timeseries in iterate_shots(train=True):
        vels = estimate_ball_velocity(timeseries, keypoint_idx)
        all_velocities.append(vels)

    vel_df = pd.DataFrame(all_velocities)
    our_dist = {
        'speed': np.sort(vel_df['narrow_speed'].dropna().values),
        'vx': np.sort(vel_df['narrow_vx'].dropna().values),
        'vz': np.sort(vel_df['narrow_vz'].dropna().values),
    }

    print(f"\nOur velocity distribution:")
    print(f"  Speed: {vel_df['narrow_speed'].mean():.4f} +/- {vel_df['narrow_speed'].std():.4f}")
    print(f"  Vx: {vel_df['narrow_vx'].mean():.4f} +/- {vel_df['narrow_vx'].std():.4f}")
    print(f"  Vz: {vel_df['narrow_vz'].mean():.4f} +/- {vel_df['narrow_vz'].std():.4f}")

    # Load data with features
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []

    print("\nSecond pass: extracting features...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_all_features(timeseries, keypoint_idx, our_dist, nba_dist)
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
        features = extract_all_features(timeseries, keypoint_idx, our_dist, nba_dist)
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

    # Analyze velocity feature correlations with targets
    print("\nVelocity feature correlations with targets:")
    vel_cols = [c for c in common_cols if 'velocity' in c.lower() or 'speed' in c.lower() or 'nba' in c.lower()]
    for col in vel_cols[:10]:
        corr_angle = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
        corr_depth = np.corrcoef(X_train[col].values, y_train[:, 1])[0, 1]
        print(f"  {col}: angle={corr_angle:.3f}, depth={corr_depth:.3f}")

    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    for w in [0.05, 0.10, 0.15, 0.20]:
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
        print(f"  {w:.0%} velocity + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
