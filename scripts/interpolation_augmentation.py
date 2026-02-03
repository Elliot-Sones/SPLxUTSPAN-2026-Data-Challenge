"""
Interpolation-Based Data Augmentation

Generate synthetic training samples by interpolating between
existing shots. This creates novel training data that is
geometrically between known samples.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


SKELETON_EDGES = [
    ('left_shoulder', 'right_shoulder'),
    ('left_hip', 'right_hip'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
]


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def get_joint(timeseries, keypoint_idx, name, frame):
    if name not in keypoint_idx or frame >= len(timeseries):
        return None
    idx = keypoint_idx[name]
    pos = timeseries[frame, idx*3:(idx+1)*3]
    if np.all(pos == 0):
        return None
    return pos


def extract_features(timeseries, keypoint_idx):
    """Extract the strongest features identified previously."""
    features = {}
    frames = [80, 100, 120, 140, 153, 160, 170]

    for frame in frames:
        # Graph features (strongest correlations)
        for j1, j2 in SKELETON_EDGES:
            p1 = get_joint(timeseries, keypoint_idx, j1, frame)
            p2 = get_joint(timeseries, keypoint_idx, j2, frame)
            if p1 is not None and p2 is not None:
                features[f'edge_{j1}_{j2}_f{frame}'] = np.linalg.norm(p2 - p1)

        # Body twist and shoulder tilt
        ls = get_joint(timeseries, keypoint_idx, 'left_shoulder', frame)
        rs = get_joint(timeseries, keypoint_idx, 'right_shoulder', frame)
        lh = get_joint(timeseries, keypoint_idx, 'left_hip', frame)
        rh = get_joint(timeseries, keypoint_idx, 'right_hip', frame)

        if all(p is not None for p in [ls, rs, lh, rh]):
            shoulder_vec = rs - ls
            hip_vec = rh - lh
            cos_ang = np.dot(shoulder_vec, hip_vec) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8)
            features[f'body_twist_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            features[f'shoulder_tilt_z_f{frame}'] = rs[2] - ls[2]

        # Elbow position
        elbow = get_joint(timeseries, keypoint_idx, 'right_elbow', frame)
        if elbow is not None:
            features[f'right_elbow_x_f{frame}'] = elbow[0]
            features[f'right_elbow_z_f{frame}'] = elbow[2]

        # Arm configuration
        wrist = get_joint(timeseries, keypoint_idx, 'right_wrist', frame)
        elbow = get_joint(timeseries, keypoint_idx, 'right_elbow', frame)
        shoulder = get_joint(timeseries, keypoint_idx, 'right_shoulder', frame)

        if all(p is not None for p in [wrist, elbow, shoulder]):
            v1 = shoulder - elbow
            v2 = wrist - elbow
            cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            features[f'elbow_angle_f{frame}'] = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            features[f'arm_length_f{frame}'] = np.linalg.norm(wrist - shoulder)

        # Velocity
        if frame >= 10 and frame < len(timeseries) - 10:
            for joint in ['right_elbow', 'right_wrist']:
                p1 = get_joint(timeseries, keypoint_idx, joint, frame - 5)
                p2 = get_joint(timeseries, keypoint_idx, joint, frame + 5)
                if p1 is not None and p2 is not None:
                    vel = (p2 - p1) / 10
                    features[f'{joint}_vel_z_f{frame}'] = vel[2]

    return features


def interpolate_timeseries(ts1, ts2, alpha):
    """Linearly interpolate between two timeseries."""
    return ts1 * (1 - alpha) + ts2 * alpha


def interpolate_targets(t1, t2, alpha):
    """Linearly interpolate between two target dicts."""
    return {
        'angle': t1['angle'] * (1 - alpha) + t2['angle'] * alpha,
        'depth': t1['depth'] * (1 - alpha) + t2['depth'] * alpha,
        'left_right': t1['left_right'] * (1 - alpha) + t2['left_right'] * alpha
    }


def main():
    print("="*80)
    print("INTERPOLATION-BASED DATA AUGMENTATION")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Load all training data
    print("\nLoading training data...")
    train_data = []
    for metadata, timeseries in iterate_shots(train=True):
        train_data.append({
            'timeseries': timeseries,
            'metadata': metadata,
            'targets': {
                'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
                'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
                'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
            },
            'player': metadata['participant_id']
        })

    print(f"Original training samples: {len(train_data)}")

    # Generate interpolated samples
    print("\nGenerating interpolated samples...")
    interpolated_data = []

    # Only interpolate within the same player
    players = list(set(d['player'] for d in train_data))
    for player in players:
        player_data = [d for d in train_data if d['player'] == player]

        for i in range(len(player_data)):
            for j in range(i + 1, len(player_data)):
                # Generate multiple interpolation points
                for alpha in [0.25, 0.5, 0.75]:
                    ts_interp = interpolate_timeseries(
                        player_data[i]['timeseries'],
                        player_data[j]['timeseries'],
                        alpha
                    )
                    targets_interp = interpolate_targets(
                        player_data[i]['targets'],
                        player_data[j]['targets'],
                        alpha
                    )
                    interpolated_data.append({
                        'timeseries': ts_interp,
                        'targets': targets_interp,
                        'player': player
                    })

    print(f"Interpolated samples: {len(interpolated_data)}")

    # Extract features
    print("\nExtracting features...")
    train_features = []
    train_targets = []
    train_players = []

    # Original data
    for d in train_data:
        features = extract_features(d['timeseries'], keypoint_idx)
        train_features.append(features)
        train_targets.append(d['targets'])
        train_players.append(d['player'])

    # Interpolated data
    for d in interpolated_data:
        features = extract_features(d['timeseries'], keypoint_idx)
        train_features.append(features)
        train_targets.append(d['targets'])
        train_players.append(d['player'])

    print(f"Total training samples: {len(train_features)}")

    # Test data
    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    # Create DataFrames
    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Features: {len(common_cols)}")

    # Top correlations
    print("\nTop feature correlations with angle:")
    corrs = []
    for col in common_cols:
        if X_train[col].std() > 0.001:
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

    # Train
    print("\nTraining models...")
    predictions = np.zeros((len(X_test), 3))

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        best_alpha = 100
        best_score = float('inf')

        for alpha in [50, 100, 200, 500]:
            gkf = GroupKFold(n_splits=5)
            scores = []
            for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled[train_idx], y[train_idx])
                pred = model.predict(X_train_scaled[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                scores.append(mse)
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        print(f"  {target}: best_alpha={best_alpha}, CV MSE={best_score:.4f}")

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

    # Calibrate
    predictions[:, 1] = predictions[:, 1] - np.mean(predictions[:, 1]) + 0.5055
    predictions = np.clip(predictions, 0, 1)

    print(f"\nangle_std: {np.std(predictions[:, 0]):.6f}")

    # Compare with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

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

    # Blends with Sub 219
    for w in [0.2, 0.3, 0.4]:
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions[:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions[:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions[:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"Saved: {blend_file} ({w:.0%} interp + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")


if __name__ == "__main__":
    main()
