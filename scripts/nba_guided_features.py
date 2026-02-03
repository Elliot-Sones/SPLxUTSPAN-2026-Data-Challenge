"""
NBA-Guided Feature Engineering

Instead of direct transfer, use NBA data to understand:
1. What release mechanics lead to consistent shots
2. What the optimal release velocity/angle patterns are
3. Create features in our data that capture these mechanics

Key insight from NBA data:
- Release velocity ~14 ft/s with low variance = consistent shooter
- Release height matters (8.2 ft avg = about 10 ft with jump)
- Path length ratio (plr) near 1.0 = straighter shot = more accurate
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


def analyze_nba_patterns():
    """Analyze NBA data to understand what makes good shooting mechanics."""
    player_df = pd.read_csv(EXTERNAL_DIR / "player_metrics.csv")

    print("="*60)
    print("NBA SHOOTING PATTERNS ANALYSIS")
    print("="*60)

    # Clean data
    player_df = player_df.dropna(subset=['rv', 'rz', 'plr'])

    # Path length ratio (plr) close to 1.0 means straighter/more efficient shot
    # We'll use this as proxy for shooting quality
    player_df['quality'] = 1 / (player_df['plr'] + 0.01)  # Lower plr = higher quality

    # Analyze correlations
    print("\nCorrelations with shooting quality (1/plr):")
    for col in ['rv', 'rvx', 'rvy', 'rvz', 'rz', 'hght']:
        if col in player_df.columns:
            corr = player_df['quality'].corr(player_df[col])
            print(f"  {col}: {corr:.3f}")

    # Find optimal ranges
    print("\nOptimal shooting mechanics (top 25% by quality):")
    top_quartile = player_df[player_df['quality'] > player_df['quality'].quantile(0.75)]

    print(f"  Release velocity (rv): {top_quartile['rv'].mean():.2f} +/- {top_quartile['rv'].std():.2f}")
    print(f"  Release height (rz): {top_quartile['rz'].mean():.2f} +/- {top_quartile['rz'].std():.2f}")
    print(f"  Vertical velocity (rvz): {top_quartile['rvz'].mean():.2f} +/- {top_quartile['rvz'].std():.2f}")

    # Key insight: consistency matters
    print("\nKey insight - Consistency of top shooters:")
    print(f"  Velocity std (top 25%): {top_quartile['rv'].std():.2f}")
    print(f"  Velocity std (bottom 25%): {player_df[player_df['quality'] < player_df['quality'].quantile(0.25)]['rv'].std():.2f}")

    return {
        'optimal_rv': top_quartile['rv'].mean(),
        'optimal_rv_std': top_quartile['rv'].std(),
        'optimal_rz': top_quartile['rz'].mean(),
        'optimal_rvz': top_quartile['rvz'].mean(),
    }


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_nba_guided_features(timeseries, keypoint_idx, nba_optima):
    """
    Extract features guided by NBA shooting patterns.

    Key insights from NBA:
    1. Release velocity consistency matters
    2. Release height matters
    3. Straighter path = better accuracy
    """
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Key frames around release
    pre_release = 150
    release = 153
    post_release = 156

    # 1. RELEASE VELOCITY FEATURES
    # NBA shows: consistent release velocity = better shooter
    wrist_pre = get_joint("right_wrist", pre_release)
    wrist_rel = get_joint("right_wrist", release)
    wrist_post = get_joint("right_wrist", post_release)

    if np.any(wrist_pre) and np.any(wrist_rel):
        vel_at_release = (wrist_rel - wrist_pre) / (3/60)  # 3 frames at 60fps
        features['release_speed'] = np.linalg.norm(vel_at_release)
        features['release_vz'] = vel_at_release[2]  # Vertical component
        features['release_vx'] = vel_at_release[0]
        features['release_vy'] = vel_at_release[1]

        # How close to NBA optimal?
        if 'optimal_rv' in nba_optima:
            features['vs_nba_optimal_speed'] = abs(features['release_speed'] - nba_optima['optimal_rv'])

    # 2. RELEASE HEIGHT FEATURES
    # NBA shows: higher release = better
    if np.any(wrist_rel):
        features['release_height'] = wrist_rel[2]

        shoulder_rel = get_joint("right_shoulder", release)
        if np.any(shoulder_rel):
            features['wrist_above_shoulder'] = wrist_rel[2] - shoulder_rel[2]

    # 3. ARM EXTENSION AT RELEASE
    # Full extension = more power and accuracy
    shoulder = get_joint("right_shoulder", release)
    elbow = get_joint("right_elbow", release)
    wrist = get_joint("right_wrist", release)

    if np.any(shoulder) and np.any(elbow) and np.any(wrist):
        upper_arm = np.linalg.norm(elbow - shoulder)
        forearm = np.linalg.norm(wrist - elbow)
        features['arm_extension'] = upper_arm + forearm

        # Elbow angle (straighter = more extended)
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        features['elbow_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # 4. FOLLOW-THROUGH CONSISTENCY
    # NBA shows: consistent follow-through = straighter path
    if np.any(wrist_rel) and np.any(wrist_post):
        follow_through = wrist_post - wrist_rel
        features['follow_through_z'] = follow_through[2]
        features['follow_through_magnitude'] = np.linalg.norm(follow_through)

        # Direction consistency
        if features.get('release_speed', 0) > 0:
            features['follow_through_alignment'] = np.dot(follow_through, vel_at_release) / (
                np.linalg.norm(follow_through) * np.linalg.norm(vel_at_release) + 1e-8)

    # 5. BODY STABILITY
    # Stable base = more accurate shot
    hip = get_joint("mid_hip", release)
    hip_pre = get_joint("mid_hip", pre_release)

    if np.any(hip) and np.any(hip_pre):
        hip_movement = np.linalg.norm(hip - hip_pre)
        features['body_stability'] = 1 / (hip_movement + 0.01)

    # 6. STANDARD POSITION FEATURES (for baseline)
    for joint in ["right_wrist", "right_elbow", "right_shoulder"]:
        for f in [150, 153, 155]:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    return features


def main():
    print("="*80)
    print("NBA-GUIDED FEATURE ENGINEERING")
    print("="*80)

    # Analyze NBA patterns first
    nba_optima = analyze_nba_patterns()

    # Load basketball data
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []
    train_ids = []

    print("\nLoading training data...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_nba_guided_features(timeseries, keypoint_idx, nba_optima)
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
        features = extract_nba_guided_features(timeseries, keypoint_idx, nba_optima)
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

    # Blend with Sub 219
    print("\nBlending with Sub 219...")
    best_blend = None
    best_std = float('inf')

    for w in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        blend = submission.copy()
        blend['scaled_angle'] = w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values
        blend['scaled_depth'] = w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values
        blend['scaled_left_right'] = w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

        std = blend['scaled_angle'].std()
        print(f"  {w:.0%} NBA-guided + {1-w:.0%} Sub219: angle_std={std:.6f}")

        if std < best_std:
            best_std = std
            best_blend = (w, blend.copy())

    # Save best blend
    if best_blend:
        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        best_blend[1].to_csv(blend_file, index=False)
        print(f"\nBest blend ({best_blend[0]:.0%}): {blend_file}")
        print(f"  angle_std: {best_std:.6f}")

    # Apply selective amplification to best blend
    print("\nApplying selective amplification to best blend...")
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")

    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    threshold = np.percentile(total_diff, 91)
    mask = total_diff > threshold

    amp = best_blend[1].copy()
    amp.loc[mask, 'scaled_angle'] += 1.1 * diff_angle[mask]
    amp.loc[mask, 'scaled_depth'] += 1.1 * diff_depth[mask]
    amp.loc[mask, 'scaled_left_right'] += 1.1 * diff_lr[mask]

    amp['scaled_angle'] = amp['scaled_angle'].clip(0, 1)
    amp['scaled_depth'] = amp['scaled_depth'].clip(0, 1)
    amp['scaled_left_right'] = amp['scaled_left_right'].clip(0, 1)

    amp['scaled_depth'] = amp['scaled_depth'] - amp['scaled_depth'].mean() + 0.5055
    amp['scaled_depth'] = amp['scaled_depth'].clip(0, 1)

    next_num += 1
    amp_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    amp.to_csv(amp_file, index=False)
    print(f"\nWith amplification: {amp_file}")
    print(f"  angle_std: {amp['scaled_angle'].std():.6f}")


if __name__ == "__main__":
    main()
