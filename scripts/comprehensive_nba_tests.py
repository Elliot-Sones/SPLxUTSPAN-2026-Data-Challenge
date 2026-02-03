"""
Comprehensive NBA Data Tests

Systematically test different ways to use NBA data:
1. Velocity features (different windows, components)
2. Release position features
3. Trajectory arc features
4. Per-player normalization using NBA baselines
5. NBA-informed optimal ranges
6. Acceleration patterns
7. Arm angle to velocity relationships
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GroupKFold
from scipy import stats
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_DIR / "external_data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"

np.random.seed(42)


def load_nba_data():
    """Load and analyze NBA data comprehensively."""
    player_df = pd.read_csv(EXTERNAL_DIR / "player_metrics.csv")
    player_df = player_df.dropna(subset=['rv', 'rvx', 'rvz', 'rz', 'plr'])

    # Quality metric
    player_df['quality'] = 1 / (player_df['plr'] + 0.01)

    # Compute derived features in NBA data
    player_df['launch_angle'] = np.degrees(np.arctan2(player_df['rvz'], np.abs(player_df['rvx'])))
    player_df['speed_ratio'] = player_df['rvz'] / (player_df['rv'] + 0.01)

    return player_df


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_velocity_features_v1(timeseries, keypoint_idx):
    """Basic velocity: wrist velocity at release."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}

    # Wrist velocity at release (frame 153)
    wrist_pre = get_joint("right_wrist", 150)
    wrist_rel = get_joint("right_wrist", 153)

    if np.any(wrist_pre) and np.any(wrist_rel):
        dt = 3 / 60
        vel = (wrist_rel - wrist_pre) / dt
        features['v1_speed'] = np.linalg.norm(vel)
        features['v1_vx'] = vel[0]
        features['v1_vy'] = vel[1]
        features['v1_vz'] = vel[2]
        if abs(vel[0]) > 0.001:
            features['v1_launch_angle'] = np.degrees(np.arctan2(vel[2], abs(vel[0])))

    return features


def extract_velocity_features_v2(timeseries, keypoint_idx):
    """Multi-window velocity: average across multiple frame windows."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}
    velocities = []

    # Multiple windows
    windows = [(150, 153), (151, 154), (152, 155), (149, 153), (150, 154)]

    for f_start, f_end in windows:
        wrist_start = get_joint("right_wrist", f_start)
        wrist_end = get_joint("right_wrist", f_end)

        if np.any(wrist_start) and np.any(wrist_end):
            dt = (f_end - f_start) / 60
            vel = (wrist_end - wrist_start) / dt
            velocities.append(vel)

    if velocities:
        avg_vel = np.mean(velocities, axis=0)
        std_vel = np.std(velocities, axis=0)

        features['v2_avg_speed'] = np.linalg.norm(avg_vel)
        features['v2_avg_vx'] = avg_vel[0]
        features['v2_avg_vz'] = avg_vel[2]
        features['v2_consistency'] = np.mean(std_vel)  # Lower = more consistent

        if abs(avg_vel[0]) > 0.001:
            features['v2_launch_angle'] = np.degrees(np.arctan2(avg_vel[2], abs(avg_vel[0])))

    return features


def extract_velocity_features_v3(timeseries, keypoint_idx):
    """Acceleration-based: capture how velocity changes through release."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}

    # Velocity before, at, and after release
    frames = [147, 150, 153, 156, 159]
    positions = [get_joint("right_wrist", f) for f in frames]

    if all(np.any(p) for p in positions):
        # Velocities between consecutive frames
        vels = []
        for i in range(len(positions) - 1):
            dt = (frames[i+1] - frames[i]) / 60
            vel = (positions[i+1] - positions[i]) / dt
            vels.append(vel)

        # Acceleration (change in velocity)
        accels = []
        for i in range(len(vels) - 1):
            dt = (frames[i+2] - frames[i+1]) / 60
            accel = (vels[i+1] - vels[i]) / dt
            accels.append(accel)

        if vels:
            # Velocity at release (middle)
            rel_vel = vels[len(vels)//2]
            features['v3_rel_speed'] = np.linalg.norm(rel_vel)
            features['v3_rel_vz'] = rel_vel[2]

            if abs(rel_vel[0]) > 0.001:
                features['v3_launch_angle'] = np.degrees(np.arctan2(rel_vel[2], abs(rel_vel[0])))

        if accels:
            # Peak acceleration
            accel_mags = [np.linalg.norm(a) for a in accels]
            features['v3_peak_accel'] = max(accel_mags)
            features['v3_accel_at_release'] = accel_mags[len(accel_mags)//2]

    return features


def extract_velocity_features_v4(timeseries, keypoint_idx):
    """Multi-joint velocity: combine wrist, elbow, shoulder."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}

    joints = ["right_wrist", "right_elbow", "right_shoulder"]
    joint_vels = {}

    for joint in joints:
        pos_pre = get_joint(joint, 150)
        pos_rel = get_joint(joint, 153)

        if np.any(pos_pre) and np.any(pos_rel):
            dt = 3 / 60
            vel = (pos_rel - pos_pre) / dt
            joint_vels[joint] = vel

            features[f'v4_{joint}_speed'] = np.linalg.norm(vel)
            features[f'v4_{joint}_vz'] = vel[2]

    # Velocity chain ratios (important for kinetic chain)
    if 'right_wrist' in joint_vels and 'right_elbow' in joint_vels:
        wrist_speed = np.linalg.norm(joint_vels['right_wrist'])
        elbow_speed = np.linalg.norm(joint_vels['right_elbow'])
        if elbow_speed > 0.001:
            features['v4_wrist_elbow_ratio'] = wrist_speed / elbow_speed

    if 'right_elbow' in joint_vels and 'right_shoulder' in joint_vels:
        elbow_speed = np.linalg.norm(joint_vels['right_elbow'])
        shoulder_speed = np.linalg.norm(joint_vels['right_shoulder'])
        if shoulder_speed > 0.001:
            features['v4_elbow_shoulder_ratio'] = elbow_speed / shoulder_speed

    # Combined launch angle from arm velocity
    if 'right_wrist' in joint_vels:
        wv = joint_vels['right_wrist']
        if abs(wv[0]) > 0.001:
            features['v4_launch_angle'] = np.degrees(np.arctan2(wv[2], abs(wv[0])))

    return features


def extract_position_features_v5(timeseries, keypoint_idx, nba_df):
    """Release position features compared to NBA patterns."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}

    # Release position
    wrist_rel = get_joint("right_wrist", 153)
    shoulder_rel = get_joint("right_shoulder", 153)
    hip_rel = get_joint("mid_hip", 153)

    if np.any(wrist_rel) and np.any(shoulder_rel):
        # Wrist height above shoulder
        features['v5_wrist_above_shoulder'] = wrist_rel[2] - shoulder_rel[2]

        # Arm extension (how far wrist is from shoulder)
        features['v5_arm_extension'] = np.linalg.norm(wrist_rel - shoulder_rel)

        # Release angle of arm
        arm_vec = wrist_rel - shoulder_rel
        if np.linalg.norm(arm_vec) > 0.001:
            features['v5_arm_angle'] = np.degrees(np.arctan2(arm_vec[2], np.linalg.norm(arm_vec[:2])))

    if np.any(wrist_rel) and np.any(hip_rel):
        # Wrist height above hip (proxy for release height)
        features['v5_release_height'] = wrist_rel[2] - hip_rel[2]

    return features


def extract_trajectory_features_v6(timeseries, keypoint_idx):
    """Trajectory arc features: how the wrist path looks like ball trajectory."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}

    # Track wrist through release window
    frames = list(range(145, 165))
    positions = [get_joint("right_wrist", f) for f in frames]
    valid_positions = [(f, p) for f, p in zip(frames, positions) if np.any(p)]

    if len(valid_positions) >= 5:
        frames_valid = [v[0] for v in valid_positions]
        pos_array = np.array([v[1] for v in valid_positions])

        # Path length
        path_lengths = np.linalg.norm(np.diff(pos_array, axis=0), axis=1)
        total_path = np.sum(path_lengths)

        # Direct distance
        direct_dist = np.linalg.norm(pos_array[-1] - pos_array[0])

        if direct_dist > 0.001:
            # Path length ratio (NBA uses this - closer to 1 = straighter)
            features['v6_path_length_ratio'] = total_path / direct_dist

        # Max height in trajectory
        heights = pos_array[:, 2]
        features['v6_max_height'] = np.max(heights)
        features['v6_height_at_release'] = heights[len(heights)//2]

        # Arc curvature (second derivative of height)
        if len(heights) >= 3:
            curvature = np.diff(heights, n=2)
            features['v6_avg_curvature'] = np.mean(curvature)
            features['v6_max_curvature'] = np.max(np.abs(curvature))

    return features


def extract_nba_comparison_features_v7(velocity_features, nba_df):
    """Compare our estimates to NBA distributions."""
    features = {}

    # Get launch angle percentile in NBA distribution
    if 'v1_launch_angle' in velocity_features:
        our_angle = velocity_features['v1_launch_angle']
        nba_angles = nba_df['launch_angle'].dropna()
        if len(nba_angles) > 0:
            percentile = stats.percentileofscore(nba_angles, our_angle)
            features['v7_angle_nba_percentile'] = percentile

            # Distance from NBA optimal (median)
            nba_median = nba_angles.median()
            features['v7_angle_vs_nba_median'] = our_angle - nba_median

            # Is it in optimal range? (25th-75th percentile)
            p25, p75 = nba_angles.quantile([0.25, 0.75])
            features['v7_angle_in_optimal'] = 1 if p25 <= our_angle <= p75 else 0

    # Speed ratio comparison
    if 'v1_vz' in velocity_features and 'v1_speed' in velocity_features:
        if velocity_features['v1_speed'] > 0.001:
            our_ratio = velocity_features['v1_vz'] / velocity_features['v1_speed']
            nba_ratios = nba_df['speed_ratio'].dropna()
            if len(nba_ratios) > 0:
                features['v7_ratio_nba_percentile'] = stats.percentileofscore(nba_ratios, our_ratio)

    return features


def extract_all_features(timeseries, keypoint_idx, nba_df):
    """Extract all feature sets."""
    all_features = {}

    # V1: Basic velocity
    v1 = extract_velocity_features_v1(timeseries, keypoint_idx)
    all_features.update(v1)

    # V2: Multi-window velocity
    v2 = extract_velocity_features_v2(timeseries, keypoint_idx)
    all_features.update(v2)

    # V3: Acceleration-based
    v3 = extract_velocity_features_v3(timeseries, keypoint_idx)
    all_features.update(v3)

    # V4: Multi-joint velocity
    v4 = extract_velocity_features_v4(timeseries, keypoint_idx)
    all_features.update(v4)

    # V5: Position features
    v5 = extract_position_features_v5(timeseries, keypoint_idx, nba_df)
    all_features.update(v5)

    # V6: Trajectory features
    v6 = extract_trajectory_features_v6(timeseries, keypoint_idx)
    all_features.update(v6)

    # V7: NBA comparison
    v7 = extract_nba_comparison_features_v7(all_features, nba_df)
    all_features.update(v7)

    return all_features


def test_feature_set(X_train, y_train, players, feature_prefix, target_name):
    """Test a specific feature set with CV."""
    cols = [c for c in X_train.columns if c.startswith(feature_prefix)]
    if not cols:
        return None, None

    X_subset = X_train[cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    gkf = GroupKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in gkf.split(X_scaled, y_train, players):
        model = Ridge(alpha=10.0)
        model.fit(X_scaled[train_idx], y_train[train_idx])
        pred = model.predict(X_scaled[val_idx])
        mse = np.mean((pred - y_train[val_idx])**2)
        scores.append(mse)

    return np.mean(scores), len(cols)


def main():
    print("="*80)
    print("COMPREHENSIVE NBA DATA TESTS")
    print("="*80)

    # Load NBA data
    nba_df = load_nba_data()
    print(f"\nLoaded NBA data: {len(nba_df)} players")
    print(f"NBA launch angle: {nba_df['launch_angle'].mean():.1f} +/- {nba_df['launch_angle'].std():.1f} degrees")

    # Load our data
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []

    print("\nExtracting features...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_all_features(timeseries, keypoint_idx, nba_df)
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
        features = extract_all_features(timeseries, keypoint_idx, nba_df)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"\nTotal features: {len(common_cols)}")

    # Test each feature set independently
    print("\n" + "="*60)
    print("FEATURE SET COMPARISON")
    print("="*60)

    feature_sets = ['v1_', 'v2_', 'v3_', 'v4_', 'v5_', 'v6_', 'v7_']
    set_names = [
        'V1: Basic velocity',
        'V2: Multi-window velocity',
        'V3: Acceleration-based',
        'V4: Multi-joint velocity',
        'V5: Position features',
        'V6: Trajectory features',
        'V7: NBA comparison'
    ]

    results = []
    for prefix, name in zip(feature_sets, set_names):
        for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
            cv_score, n_features = test_feature_set(
                X_train, y_train[:, target_idx], players, prefix, target_name
            )
            if cv_score is not None:
                results.append({
                    'feature_set': name,
                    'target': target_name,
                    'cv_mse': cv_score,
                    'n_features': n_features
                })

    results_df = pd.DataFrame(results)
    print("\nCV MSE by feature set and target:")
    pivot = results_df.pivot(index='feature_set', columns='target', values='cv_mse')
    print(pivot.to_string())

    # Feature correlations with targets
    print("\n" + "="*60)
    print("TOP FEATURE CORRELATIONS")
    print("="*60)

    correlations = []
    for col in X_train.columns:
        for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
            corr = np.corrcoef(X_train[col].values, y_train[:, target_idx])[0, 1]
            correlations.append({
                'feature': col,
                'target': target_name,
                'correlation': corr,
                'abs_corr': abs(corr)
            })

    corr_df = pd.DataFrame(correlations)

    print("\nTop 10 features by |correlation| with angle:")
    angle_corrs = corr_df[corr_df['target'] == 'angle'].sort_values('abs_corr', ascending=False)
    print(angle_corrs.head(10).to_string(index=False))

    print("\nTop 10 features by |correlation| with depth:")
    depth_corrs = corr_df[corr_df['target'] == 'depth'].sort_values('abs_corr', ascending=False)
    print(depth_corrs.head(10).to_string(index=False))

    # Find best feature combination
    print("\n" + "="*60)
    print("TESTING FEATURE COMBINATIONS")
    print("="*60)

    # Select top features by correlation
    top_angle_features = angle_corrs.head(15)['feature'].tolist()
    top_depth_features = depth_corrs.head(15)['feature'].tolist()
    top_features = list(set(top_angle_features + top_depth_features))

    print(f"\nSelected {len(top_features)} top features")

    # Test combinations
    combinations = [
        ('All features', common_cols),
        ('Top correlated', top_features),
        ('V1+V4 (velocity)', [c for c in common_cols if c.startswith('v1_') or c.startswith('v4_')]),
        ('V5+V6 (position+trajectory)', [c for c in common_cols if c.startswith('v5_') or c.startswith('v6_')]),
        ('Launch angle only', [c for c in common_cols if 'launch_angle' in c or 'arm_angle' in c]),
    ]

    combo_results = []
    for name, cols in combinations:
        cols = [c for c in cols if c in X_train.columns]
        if not cols:
            continue

        X_subset = X_train[cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)

        gkf = GroupKFold(n_splits=5)

        for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
            scores = []
            for train_idx, val_idx in gkf.split(X_scaled, y_train[:, target_idx], players):
                model = Ridge(alpha=10.0)
                model.fit(X_scaled[train_idx], y_train[:, target_idx][train_idx])
                pred = model.predict(X_scaled[val_idx])
                mse = np.mean((pred - y_train[:, target_idx][val_idx])**2)
                scores.append(mse)

            combo_results.append({
                'combination': name,
                'target': target_name,
                'cv_mse': np.mean(scores),
                'n_features': len(cols)
            })

    combo_df = pd.DataFrame(combo_results)
    print("\nCV MSE by combination:")
    pivot = combo_df.pivot(index='combination', columns='target', values='cv_mse')
    pivot['mean'] = pivot.mean(axis=1)
    print(pivot.sort_values('mean').to_string())

    # Train best model
    print("\n" + "="*60)
    print("TRAINING BEST MODEL")
    print("="*60)

    # Use top correlated features
    best_cols = top_features
    X_train_best = X_train[best_cols].values
    X_test_best = X_test[best_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_best)
    X_test_scaled = scaler.transform(X_test_best)

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

    # Calibrate and save
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

    # Blend variations
    print("\n" + "="*60)
    print("BLENDING WITH SUB 219")
    print("="*60)

    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
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
        print(f"  {w:.0%} NBA + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")

    # Save results
    results_df.to_csv(OUTPUT_DIR / "nba_feature_set_results.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "nba_feature_correlations.csv", index=False)
    combo_df.to_csv(OUTPUT_DIR / "nba_combination_results.csv", index=False)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
