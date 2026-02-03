"""
NBA Trajectory Analysis

The path_detail.csv contains frame-by-frame ball trajectories.
Extract insights about optimal shot characteristics:
1. Arc shape (peak height, curvature)
2. Entry angle to basket
3. Velocity profile during flight
4. Time of flight
5. Optimal release-to-peak ratios
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


def analyze_nba_trajectories():
    """Analyze NBA ball trajectories to understand optimal shot characteristics."""
    path_df = pd.read_csv(EXTERNAL_DIR / "path_detail.csv")
    player_df = pd.read_csv(EXTERNAL_DIR / "player_metrics.csv")

    print("="*60)
    print("NBA TRAJECTORY ANALYSIS")
    print("="*60)

    print(f"\npath_detail.csv: {len(path_df)} rows")
    print(f"Columns: {list(path_df.columns)}")

    # Group by player to analyze individual shots
    # Each player has one shot trajectory
    players = path_df.groupby('pid')

    trajectory_features = []

    for pid, group in players:
        group = group.sort_values('t')
        group = group.dropna(subset=['cz', 'cv'])  # Remove rows with NA values

        if len(group) < 5:
            continue

        # Extract trajectory characteristics
        features = {'pid': pid}

        # Time of flight
        features['time_of_flight'] = group['t'].max() - group['t'].min()

        # Peak height (maximum z)
        features['peak_height'] = group['cz'].max()
        features['release_height'] = group['cz'].iloc[0]
        features['peak_to_release_ratio'] = features['peak_height'] / (features['release_height'] + 0.01)

        # When does peak occur? (as fraction of flight time)
        peak_idx = group['cz'].idxmax()
        peak_time = group.loc[peak_idx, 't']
        total_time = features['time_of_flight']
        features['peak_time_fraction'] = peak_time / (total_time + 0.01) if total_time > 0 else 0.5

        # Release velocity
        features['release_velocity'] = group['cv'].iloc[0] if pd.notna(group['cv'].iloc[0]) else 0
        features['release_vx'] = group['cvx'].iloc[0] if pd.notna(group['cvx'].iloc[0]) else 0
        features['release_vy'] = group['cvy'].iloc[0] if pd.notna(group['cvy'].iloc[0]) else 0
        features['release_vz'] = group['cvz'].iloc[0] if pd.notna(group['cvz'].iloc[0]) else 0

        # Entry velocity (end of trajectory)
        features['entry_velocity'] = group['cv'].iloc[-1] if pd.notna(group['cv'].iloc[-1]) else 0
        features['entry_vz'] = group['cvz'].iloc[-1] if pd.notna(group['cvz'].iloc[-1]) else 0

        # Entry angle (angle ball enters basket)
        final_vx = group['cvx'].iloc[-1]
        final_vz = group['cvz'].iloc[-1]
        if pd.notna(final_vx) and pd.notna(final_vz) and abs(final_vx) > 0.001:
            features['entry_angle'] = np.degrees(np.arctan2(-final_vz, abs(final_vx)))

        # Arc curvature (average acceleration)
        features['avg_horizontal_accel'] = group['cax'].mean() if group['cax'].notna().any() else 0
        features['avg_vertical_accel'] = group['caz'].mean() if group['caz'].notna().any() else -32  # Gravity

        # Shot distance
        features['shot_distance'] = group['ddst'].iloc[0]

        # Player height
        features['player_height'] = group['hght'].iloc[0]

        trajectory_features.append(features)

    traj_df = pd.DataFrame(trajectory_features)

    # Merge with player metrics (which has quality info)
    player_df = player_df.dropna(subset=['plr'])
    player_df['quality'] = 1 / (player_df['plr'] + 0.01)

    merged = traj_df.merge(player_df[['pid', 'plr', 'quality']], on='pid', how='inner')

    print(f"\nAnalyzed {len(merged)} player trajectories")

    # Summary statistics
    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)

    for col in ['peak_height', 'release_height', 'peak_to_release_ratio',
                'peak_time_fraction', 'entry_angle', 'time_of_flight',
                'release_velocity', 'entry_velocity']:
        if col in merged.columns:
            print(f"{col}: {merged[col].mean():.3f} +/- {merged[col].std():.3f}")

    # Correlations with quality
    print("\n" + "="*60)
    print("CORRELATIONS WITH SHOT QUALITY")
    print("="*60)

    correlations = []
    for col in merged.columns:
        if col not in ['pid', 'plr', 'quality'] and merged[col].dtype in [np.float64, np.int64]:
            corr = merged['quality'].corr(merged[col])
            correlations.append({'feature': col, 'correlation': corr, 'abs_corr': abs(corr)})

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
    print(corr_df.to_string(index=False))

    return merged, corr_df


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_trajectory_inspired_features(timeseries, keypoint_idx, nba_optima):
    """
    Extract features inspired by NBA trajectory analysis.

    Key insights from NBA:
    1. Peak height matters
    2. Entry angle matters
    3. Time to peak matters
    """
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Track wrist through release to follow-through
    # This mimics the ball trajectory somewhat
    frames = list(range(145, 170))
    wrist_positions = [get_joint("right_wrist", f) for f in frames]
    valid = [(f, p) for f, p in zip(frames, wrist_positions) if np.any(p)]

    if len(valid) >= 5:
        frames_valid = [v[0] for v in valid]
        pos_array = np.array([v[1] for v in valid])
        heights = pos_array[:, 2]

        # Peak height of wrist trajectory
        features['wrist_peak_height'] = np.max(heights)
        features['wrist_release_height'] = heights[len(heights)//2]  # Around frame 153

        # Peak to release ratio (NBA shows this matters)
        if features['wrist_release_height'] > 0.01:
            features['wrist_peak_ratio'] = features['wrist_peak_height'] / features['wrist_release_height']

        # When does peak occur? (fraction of motion)
        peak_idx = np.argmax(heights)
        features['wrist_peak_fraction'] = peak_idx / len(heights)

        # Arc curvature (second derivative of height)
        if len(heights) >= 3:
            curvature = np.diff(heights, n=2)
            features['wrist_curvature'] = np.mean(curvature)

    # Velocity-based features (mimic release velocity)
    wrist_150 = get_joint("right_wrist", 150)
    wrist_153 = get_joint("right_wrist", 153)
    wrist_156 = get_joint("right_wrist", 156)

    if np.any(wrist_150) and np.any(wrist_153):
        vel = wrist_153 - wrist_150
        features['release_speed'] = np.linalg.norm(vel)
        features['release_vz'] = vel[2]
        features['release_vx'] = vel[0]

        if abs(vel[0]) > 0.001:
            features['release_angle'] = np.degrees(np.arctan2(vel[2], abs(vel[0])))

            # Compare to NBA optimal entry angle (~45-50 degrees)
            if 'optimal_entry_angle' in nba_optima:
                features['vs_optimal_angle'] = abs(features['release_angle'] - nba_optima['optimal_entry_angle'])

    # Follow-through velocity (mimic entry velocity)
    if np.any(wrist_153) and np.any(wrist_156):
        follow_vel = wrist_156 - wrist_153
        features['followthrough_speed'] = np.linalg.norm(follow_vel)
        features['followthrough_vz'] = follow_vel[2]

        if abs(follow_vel[0]) > 0.001:
            features['followthrough_angle'] = np.degrees(np.arctan2(follow_vel[2], abs(follow_vel[0])))

    # Position features
    for joint in ["right_wrist", "right_shoulder"]:
        pos = get_joint(joint, 153)
        features[f'{joint}_z_153'] = pos[2]

    return features


def main():
    # First, analyze NBA trajectories
    nba_traj, nba_corr = analyze_nba_trajectories()

    # Extract optimal values from NBA
    nba_optima = {
        'optimal_peak_height': nba_traj['peak_height'].mean(),
        'optimal_entry_angle': nba_traj['entry_angle'].mean() if 'entry_angle' in nba_traj.columns else 45,
        'optimal_peak_fraction': nba_traj['peak_time_fraction'].mean(),
    }

    print("\n" + "="*60)
    print("NBA OPTIMAL VALUES")
    print("="*60)
    for k, v in nba_optima.items():
        print(f"  {k}: {v:.3f}")

    # Now extract features from our data
    print("\n" + "="*60)
    print("EXTRACTING TRAJECTORY-INSPIRED FEATURES")
    print("="*60)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_trajectory_inspired_features(timeseries, keypoint_idx, nba_optima)
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
        features = extract_trajectory_inspired_features(timeseries, keypoint_idx, nba_optima)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"\nFeatures: {len(common_cols)}")

    # Correlations
    print("\nFeature correlations with targets:")
    for col in sorted(common_cols):
        corr_a = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
        corr_d = np.corrcoef(X_train[col].values, y_train[:, 1])[0, 1]
        print(f"  {col}: angle={corr_a:.3f}, depth={corr_d:.3f}")

    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    # Compare
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Blend
    print("\nBlending with Sub 219...")
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
        print(f"  {w:.0%} trajectory + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
