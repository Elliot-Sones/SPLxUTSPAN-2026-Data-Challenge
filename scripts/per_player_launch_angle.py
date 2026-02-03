"""
Per-Player Launch Angle Model

Each player may have different optimal launch angles.
Normalize launch angle features per player to capture
deviations from their personal baseline.
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


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def compute_launch_angle(timeseries, keypoint_idx):
    """Compute launch angle from wrist velocity."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Multi-frame average for stability
    angles = []
    for f_start, f_end in [(149, 152), (150, 153), (151, 154), (152, 155)]:
        w_start = get_joint("right_wrist", f_start)
        w_end = get_joint("right_wrist", f_end)
        if np.any(w_start) and np.any(w_end):
            v = w_end - w_start
            if abs(v[0]) > 0.001:
                angles.append(np.degrees(np.arctan2(v[2], abs(v[0]))))

    return np.mean(angles) if angles else 0


def extract_features(timeseries, keypoint_idx):
    """Extract features including raw launch angle."""
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    features = {}

    # Launch angle
    features['launch_angle'] = compute_launch_angle(timeseries, keypoint_idx)

    # Supporting features
    wrist_150 = get_joint("right_wrist", 150)
    wrist_153 = get_joint("right_wrist", 153)

    if np.any(wrist_150) and np.any(wrist_153):
        vel = wrist_153 - wrist_150
        features['vel_vx'] = vel[0]
        features['vel_vy'] = vel[1]
        features['vel_vz'] = vel[2]
        features['vel_speed'] = np.linalg.norm(vel)

    # Positions
    features['wrist_z_153'] = wrist_153[2] if np.any(wrist_153) else 0
    shoulder = get_joint("right_shoulder", 153)
    features['shoulder_z_153'] = shoulder[2] if np.any(shoulder) else 0

    if np.any(wrist_153) and np.any(shoulder):
        features['wrist_above_shoulder'] = wrist_153[2] - shoulder[2]

    return features


def main():
    print("="*80)
    print("PER-PLAYER LAUNCH ANGLE MODEL")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # First pass: compute per-player statistics
    print("\nComputing per-player statistics...")
    player_data = {}

    for metadata, timeseries in iterate_shots(train=True):
        player_id = metadata['participant_id']
        la = compute_launch_angle(timeseries, keypoint_idx)

        if player_id not in player_data:
            player_data[player_id] = []
        player_data[player_id].append(la)

    player_stats = {}
    print("\nPer-player launch angle statistics:")
    for player_id, angles in player_data.items():
        angles = np.array(angles)
        player_stats[player_id] = {
            'mean': np.mean(angles),
            'std': np.std(angles) + 0.01,  # Avoid division by zero
        }
        print(f"  Player {player_id}: mean={np.mean(angles):.2f}, std={np.std(angles):.2f}, n={len(angles)}")

    # Global stats for test players we haven't seen
    all_angles = []
    for angles in player_data.values():
        all_angles.extend(angles)
    global_stats = {
        'mean': np.mean(all_angles),
        'std': np.std(all_angles) + 0.01,
    }
    print(f"  Global: mean={global_stats['mean']:.2f}, std={global_stats['std']:.2f}")

    # Second pass: extract features with per-player normalization
    print("\nExtracting features...")
    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_features(timeseries, keypoint_idx)

        # Add per-player normalized launch angle
        player_id = metadata['participant_id']
        stats = player_stats.get(player_id, global_stats)
        features['la_normalized'] = (features['launch_angle'] - stats['mean']) / stats['std']
        features['la_zscore'] = features['la_normalized']  # Alias

        # Player identifier (for per-player models)
        features['player_id'] = player_id

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
        features = extract_features(timeseries, keypoint_idx)

        # Use global stats for test (we don't know player)
        features['la_normalized'] = (features['launch_angle'] - global_stats['mean']) / global_stats['std']
        features['la_zscore'] = features['la_normalized']
        features['player_id'] = -1  # Unknown

        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    # Remove player_id for modeling
    feature_cols = [c for c in X_train.columns if c != 'player_id']
    X_train_model = X_train[feature_cols]
    X_test_model = X_test[feature_cols]

    common_cols = list(set(X_train_model.columns) & set(X_test_model.columns))
    X_train_model = X_train_model[common_cols]
    X_test_model = X_test_model[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"\nFeatures: {len(common_cols)}")

    # Correlations
    print("\nFeature correlations with angle:")
    for col in sorted(common_cols):
        corr = np.corrcoef(X_train_model[col].values, y_train[:, 0])[0, 1]
        print(f"  {col}: {corr:.4f}")

    # Compare raw vs normalized launch angle
    print("\nRaw vs Normalized launch angle:")
    corr_raw = np.corrcoef(X_train_model['launch_angle'].values, y_train[:, 0])[0, 1]
    corr_norm = np.corrcoef(X_train_model['la_normalized'].values, y_train[:, 0])[0, 1]
    print(f"  Raw: {corr_raw:.4f}")
    print(f"  Normalized: {corr_norm:.4f}")

    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_model)
    X_test_scaled = scaler.transform(X_test_model)

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

    # Compare with others
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub270 = pd.read_csv(SUBMISSION_DIR / "submission_270.csv")
    sub277 = pd.read_csv(SUBMISSION_DIR / "submission_277.csv")

    print(f"Correlation with Sub 219: {np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]:.4f}")
    print(f"Correlation with Sub 270: {np.corrcoef(predictions[:, 0], sub270['scaled_angle'].values)[0, 1]:.4f}")
    print(f"Correlation with Sub 277: {np.corrcoef(predictions[:, 0], sub277['scaled_angle'].values)[0, 1]:.4f}")

    # Blends
    print("\n" + "="*60)
    print("BLENDING")
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
        print(f"  {w:.0%} per-player + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")

    # Four-way blend: per-player + comprehensive + launch + Sub219
    print("\nFour-way blend:")
    sub270_preds = sub270[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values
    sub277_preds = sub277[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values

    w_pp, w_comp, w_launch, w_219 = 0.10, 0.10, 0.10, 0.70
    blend = submission.copy()
    blend['scaled_angle'] = (w_pp * predictions[:, 0] + w_comp * sub270_preds[:, 0] +
                            w_launch * sub277_preds[:, 0] + w_219 * sub219['scaled_angle'].values)
    blend['scaled_depth'] = (w_pp * predictions[:, 1] + w_comp * sub270_preds[:, 1] +
                            w_launch * sub277_preds[:, 1] + w_219 * sub219['scaled_depth'].values)
    blend['scaled_left_right'] = (w_pp * predictions[:, 2] + w_comp * sub270_preds[:, 2] +
                                  w_launch * sub277_preds[:, 2] + w_219 * sub219['scaled_left_right'].values)

    blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
    blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

    std = blend['scaled_angle'].std()
    next_num += 1
    blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    blend.to_csv(blend_file, index=False)
    print(f"  10% each of 3 NBA models + 70% Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
