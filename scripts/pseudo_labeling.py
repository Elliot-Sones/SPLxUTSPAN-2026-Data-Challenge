"""
Pseudo-Labeling / Semi-Supervised Learning

Use Sub 219 predictions as pseudo-labels for test data,
retrain on combined data.
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


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoints = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoints:
            keypoints.append(name)
    return {name: i for i, name in enumerate(keypoints)}


def extract_features(timeseries, keypoint_idx):
    """Extract comprehensive features."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Release frame features
    for f in [150, 153, 156]:
        for joint in ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip']:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    # Velocity features
    for f_start, f_end in [(148, 153), (150, 155), (153, 158)]:
        wrist_s = get_joint("right_wrist", f_start)
        wrist_e = get_joint("right_wrist", f_end)
        if np.any(wrist_s) and np.any(wrist_e):
            vel = wrist_e - wrist_s
            features[f'wrist_vx_{f_start}_{f_end}'] = vel[0]
            features[f'wrist_vy_{f_start}_{f_end}'] = vel[1]
            features[f'wrist_vz_{f_start}_{f_end}'] = vel[2]
            features[f'wrist_speed_{f_start}_{f_end}'] = np.linalg.norm(vel)

    return features


def main():
    print("="*80)
    print("PSEUDO-LABELING / SEMI-SUPERVISED LEARNING")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Load training data
    print("\nLoading training data...")
    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_features(timeseries, keypoint_idx)
        train_features.append(features)
        train_targets.append({
            'angle': scalers['angle'].transform([[metadata['angle']]])[0, 0],
            'depth': scalers['depth'].transform([[metadata['depth']]])[0, 0],
            'left_right': scalers['left_right'].transform([[metadata['left_right']]])[0, 0]
        })
        train_players.append(metadata['participant_id'])

    # Load test data
    print("Loading test data...")
    test_features = []
    test_ids = []

    for metadata, timeseries in iterate_shots(train=False):
        features = extract_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    # Load pseudo-labels from Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    pseudo_labels = sub219[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values

    print(f"Train samples: {len(train_features)}")
    print(f"Test samples: {len(test_features)}")

    # Convert to DataFrames
    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    # Combine train + pseudo-labeled test
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = np.vstack([y_train, pseudo_labels])

    # Create pseudo-players for test data (use 0 to indicate pseudo)
    players_combined = np.concatenate([players, np.zeros(len(test_features))])

    print(f"\nCombined dataset: {len(X_combined)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_combined_scaled = scaler.transform(X_combined)

    # Method 1: Train on combined data
    print("\n" + "="*60)
    print("METHOD 1: TRAIN ON COMBINED DATA")
    print("="*60)

    predictions_m1 = np.zeros((len(X_test), 3))

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_combined[:, i]
        model = Ridge(alpha=50)
        model.fit(X_combined_scaled, y)
        predictions_m1[:, i] = model.predict(X_test_scaled)

    predictions_m1[:, 1] = predictions_m1[:, 1] - np.mean(predictions_m1[:, 1]) + 0.5055
    predictions_m1 = np.clip(predictions_m1, 0, 1)

    print(f"angle_std: {np.std(predictions_m1[:, 0]):.6f}")
    corr = np.corrcoef(predictions_m1[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Method 2: Weighted pseudo-labeling (give more weight to training data)
    print("\n" + "="*60)
    print("METHOD 2: WEIGHTED PSEUDO-LABELING")
    print("="*60)

    # Weight: training data gets weight 1, pseudo-labels get weight 0.3
    sample_weights = np.concatenate([
        np.ones(len(train_features)),
        0.3 * np.ones(len(test_features))
    ])

    predictions_m2 = np.zeros((len(X_test), 3))

    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_combined[:, i]
        model = Ridge(alpha=50)
        model.fit(X_combined_scaled, y, sample_weight=sample_weights)
        predictions_m2[:, i] = model.predict(X_test_scaled)

    predictions_m2[:, 1] = predictions_m2[:, 1] - np.mean(predictions_m2[:, 1]) + 0.5055
    predictions_m2 = np.clip(predictions_m2, 0, 1)

    print(f"angle_std: {np.std(predictions_m2[:, 0]):.6f}")
    corr = np.corrcoef(predictions_m2[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Method 3: Iterative pseudo-labeling
    print("\n" + "="*60)
    print("METHOD 3: ITERATIVE PSEUDO-LABELING")
    print("="*60)

    current_pseudo = pseudo_labels.copy()

    for iteration in range(3):
        y_iter = np.vstack([y_train, current_pseudo])

        predictions_iter = np.zeros((len(X_test), 3))
        for i in range(3):
            model = Ridge(alpha=50)
            model.fit(X_combined_scaled, y_iter[:, i])
            predictions_iter[:, i] = model.predict(X_test_scaled)

        predictions_iter[:, 1] = predictions_iter[:, 1] - np.mean(predictions_iter[:, 1]) + 0.5055
        predictions_iter = np.clip(predictions_iter, 0, 1)

        # Update pseudo-labels (blend with previous)
        current_pseudo = 0.5 * current_pseudo + 0.5 * predictions_iter

        print(f"Iteration {iteration+1}: angle_std={np.std(predictions_iter[:, 0]):.6f}")

    predictions_m3 = predictions_iter

    # Method 4: Ensemble of methods
    print("\n" + "="*60)
    print("METHOD 4: ENSEMBLE")
    print("="*60)

    predictions_ensemble = (predictions_m1 + predictions_m2 + predictions_m3) / 3
    predictions_ensemble[:, 1] = predictions_ensemble[:, 1] - np.mean(predictions_ensemble[:, 1]) + 0.5055

    print(f"angle_std: {np.std(predictions_ensemble[:, 0]):.6f}")
    corr = np.corrcoef(predictions_ensemble[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Save submissions
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1

    for name, preds in [('pseudo_combined', predictions_m1),
                        ('pseudo_weighted', predictions_m2),
                        ('pseudo_iterative', predictions_m3),
                        ('pseudo_ensemble', predictions_ensemble)]:
        submission = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': preds[:, 0],
            'scaled_depth': preds[:, 1],
            'scaled_left_right': preds[:, 2]
        })
        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        submission.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({name})")
        next_num += 1

    # Blend with Sub 219
    print("\n" + "="*60)
    print("BLENDING WITH SUB 219")
    print("="*60)

    for w in [0.1, 0.2, 0.3]:
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w * predictions_ensemble[:, 0] + (1-w) * sub219['scaled_angle'].values,
            'scaled_depth': w * predictions_ensemble[:, 1] + (1-w) * sub219['scaled_depth'].values,
            'scaled_left_right': w * predictions_ensemble[:, 2] + (1-w) * sub219['scaled_left_right'].values
        })
        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055

        output_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({w:.0%} pseudo + {1-w:.0%} Sub219, angle_std={blend['scaled_angle'].std():.6f})")
        next_num += 1


if __name__ == "__main__":
    main()
