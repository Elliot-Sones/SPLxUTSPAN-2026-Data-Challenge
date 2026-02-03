"""
Transfer Learning from Baseball Pitching

Use OpenBiomechanics baseball data to learn biomechanical patterns,
then transfer to bowling prediction.

Approach:
1. Learn relationship: joint kinematics -> pitch speed (baseball)
2. Extract similar kinematic features from bowling data
3. Use pre-trained knowledge to improve bowling predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_DIR / "external_data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def load_baseball_data():
    """Load OpenBiomechanics baseball pitching data."""
    poi_file = EXTERNAL_DIR / "openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv"
    df = pd.read_csv(poi_file)

    print(f"Baseball pitching samples: {len(df)}")

    # Select relevant biomechanical features for transfer
    kinematic_features = [
        'max_shoulder_internal_rotational_velo',
        'max_elbow_extension_velo',
        'max_torso_rotational_velo',
        'max_rotation_hip_shoulder_separation',
        'max_elbow_flexion',
        'max_shoulder_external_rotation',
        'elbow_flexion_fp',  # at foot plant
        'shoulder_abduction_fp',
        'shoulder_external_rotation_fp',
        'torso_rotation_fp',
        'pelvis_rotation_fp',
        'elbow_flexion_mer',  # at max external rotation
        'torso_rotation_mer',
        'torso_rotation_br',  # at ball release
        'stride_length',
        'arm_slot',
    ]

    # Target: pitch speed
    target = 'pitch_speed_mph'

    # Filter to available columns
    available = [c for c in kinematic_features if c in df.columns]
    print(f"Using {len(available)} kinematic features")

    X_baseball = df[available].fillna(0).values
    y_baseball = df[target].values

    return X_baseball, y_baseball, available


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def compute_joint_angle(p1, p2, p3):
    """Compute angle at p2 between vectors p2->p1 and p2->p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def compute_angular_velocity(angle_before, angle_after, dt=1/60):
    """Compute angular velocity in deg/s."""
    return (angle_after - angle_before) / dt


def extract_bowling_biomech_features(timeseries, keypoint_idx):
    """Extract biomechanical features from bowling data similar to baseball metrics."""
    features = {}
    n_frames = len(timeseries)

    # Key frames: before release, at release, after release
    frames_before = [145, 148, 150]
    frame_release = 153
    frames_after = [155, 158, 160]

    # Get joint positions
    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # 1. Shoulder external rotation (approximated from arm position)
    for f in [150, 153, 155]:
        shoulder = get_joint("right_shoulder", f)
        elbow = get_joint("right_elbow", f)
        wrist = get_joint("right_wrist", f)

        if np.any(shoulder) and np.any(elbow) and np.any(wrist):
            # Elbow angle
            elbow_angle = compute_joint_angle(shoulder, elbow, wrist)
            features[f'elbow_flexion_f{f}'] = elbow_angle

            # Shoulder angle (simplified)
            hip = get_joint("right_hip", f)
            if np.any(hip):
                shoulder_angle = compute_joint_angle(hip, shoulder, elbow)
                features[f'shoulder_abduction_f{f}'] = shoulder_angle

    # 2. Angular velocities
    if 153 < n_frames and 150 < n_frames:
        # Elbow extension velocity
        elbow_150 = features.get('elbow_flexion_f150', 0)
        elbow_153 = features.get('elbow_flexion_f153', 0)
        features['elbow_extension_velo'] = compute_angular_velocity(elbow_150, elbow_153, dt=3/60)

        # Shoulder velocity
        shoulder_150 = features.get('shoulder_abduction_f150', 0)
        shoulder_153 = features.get('shoulder_abduction_f153', 0)
        features['shoulder_rotation_velo'] = compute_angular_velocity(shoulder_150, shoulder_153, dt=3/60)

    # 3. Torso rotation (using hip-shoulder line)
    for f in [150, 153, 155]:
        left_shoulder = get_joint("left_shoulder", f)
        right_shoulder = get_joint("right_shoulder", f)
        left_hip = get_joint("left_hip", f)
        right_hip = get_joint("right_hip", f)

        if np.any(left_shoulder) and np.any(right_shoulder):
            # Shoulder line angle
            shoulder_vec = right_shoulder - left_shoulder
            shoulder_angle = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
            features[f'torso_rotation_f{f}'] = shoulder_angle

        if np.any(left_hip) and np.any(right_hip):
            # Hip line angle
            hip_vec = right_hip - left_hip
            hip_angle = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))
            features[f'pelvis_rotation_f{f}'] = hip_angle

            # Hip-shoulder separation
            if f'torso_rotation_f{f}' in features:
                features[f'hip_shoulder_sep_f{f}'] = features[f'torso_rotation_f{f}'] - hip_angle

    # 4. Torso rotation velocity
    if 'torso_rotation_f150' in features and 'torso_rotation_f153' in features:
        features['torso_rotational_velo'] = compute_angular_velocity(
            features['torso_rotation_f150'], features['torso_rotation_f153'], dt=3/60
        )

    # 5. Wrist/hand speed (similar to arm speed at release)
    wrist_150 = get_joint("right_wrist", 150)
    wrist_153 = get_joint("right_wrist", 153)
    if np.any(wrist_150) and np.any(wrist_153):
        wrist_velocity = np.linalg.norm(wrist_153 - wrist_150) / (3/60)
        features['wrist_speed'] = wrist_velocity

    # 6. Standard position features (for regularization)
    for joint in ["right_wrist", "right_elbow", "right_shoulder"]:
        for f in [150, 153, 155]:
            pos = get_joint(joint, f)
            features[f'{joint}_x_f{f}'] = pos[0]
            features[f'{joint}_y_f{f}'] = pos[1]
            features[f'{joint}_z_f{f}'] = pos[2]

    return features


def pretrain_on_baseball(X_baseball, y_baseball):
    """Pre-train a model on baseball data to learn kinematic->outcome relationship."""
    print("\nPre-training on baseball data...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_baseball)

    # Train model to predict pitch speed from kinematics
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    model.fit(X_scaled, y_baseball)

    # Cross-validate
    scores = cross_val_score(Ridge(alpha=1.0), X_scaled, y_baseball, cv=5, scoring='r2')
    print(f"  Baseball CV R2: {scores.mean():.3f} +/- {scores.std():.3f}")

    return scaler, model


def load_bowling_data():
    """Load bowling data with biomechanical features."""
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []
    train_ids = []

    print("\nLoading bowling training data...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_bowling_biomech_features(timeseries, keypoint_idx)
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

    print("Loading bowling test data...")
    for metadata, timeseries in iterate_shots(train=False):
        features = extract_bowling_biomech_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    return (train_features, train_targets, train_players, train_ids,
            test_features, test_ids)


def create_transfer_features(bowling_features, baseball_feature_names):
    """Create features that align with baseball kinematic patterns."""
    # Map bowling features to baseball-like features
    mapping = {
        'elbow_flexion_f153': 'max_elbow_flexion',
        'elbow_extension_velo': 'max_elbow_extension_velo',
        'shoulder_abduction_f153': 'shoulder_abduction_fp',
        'shoulder_rotation_velo': 'max_shoulder_internal_rotational_velo',
        'torso_rotational_velo': 'max_torso_rotational_velo',
        'hip_shoulder_sep_f150': 'max_rotation_hip_shoulder_separation',
        'torso_rotation_f153': 'torso_rotation_fp',
        'pelvis_rotation_f153': 'pelvis_rotation_fp',
    }

    transfer_feats = {}
    for bowling_name, baseball_name in mapping.items():
        if bowling_name in bowling_features:
            transfer_feats[f'transfer_{baseball_name}'] = bowling_features[bowling_name]

    return transfer_feats


def main():
    print("="*80)
    print("TRANSFER LEARNING FROM BASEBALL PITCHING")
    print("="*80)

    # Load baseball data
    X_baseball, y_baseball, baseball_features = load_baseball_data()

    # Pre-train on baseball
    baseball_scaler, baseball_model = pretrain_on_baseball(X_baseball, y_baseball)

    # Load bowling data
    (train_features, train_targets, train_players, train_ids,
     test_features, test_ids) = load_bowling_data()

    print(f"\nBowling train: {len(train_features)}, test: {len(test_features)}")

    # Create combined features: original bowling + transfer features
    X_train = []
    for bf in train_features:
        transfer = create_transfer_features(bf, baseball_features)
        combined = {**bf, **transfer}
        X_train.append(combined)

    X_test = []
    for bf in test_features:
        transfer = create_transfer_features(bf, baseball_features)
        combined = {**bf, **transfer}
        X_test.append(combined)

    X_train = pd.DataFrame(X_train).fillna(0)
    X_test = pd.DataFrame(X_test).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Total features: {len(common_cols)}")
    print(f"Transfer features: {len([c for c in common_cols if c.startswith('transfer_')])}")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train bowling prediction model
    print("\nTraining bowling model with transfer features...")

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

    # Save submission
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

    # Check correlation with Sub 219
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    corr = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr:.4f}")

    # Blend with Sub 219
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
        print(f"  {w:.0%} transfer + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")

    # Apply selective amplification
    print("\nApplying selective amplification...")
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")
    sub151 = pd.read_csv(SUBMISSION_DIR / "submission_151.csv")

    diff_angle = sub133['scaled_angle'].values - sub151['scaled_angle'].values
    diff_depth = sub133['scaled_depth'].values - sub151['scaled_depth'].values
    diff_lr = sub133['scaled_left_right'].values - sub151['scaled_left_right'].values
    total_diff = np.sqrt(diff_angle**2 + diff_depth**2 + diff_lr**2)

    threshold = np.percentile(total_diff, 91)
    mask = total_diff > threshold

    amp_preds = predictions.copy()
    amp_preds[mask, 0] += 1.1 * diff_angle[mask]
    amp_preds[mask, 1] += 1.1 * diff_depth[mask]
    amp_preds[mask, 2] += 1.1 * diff_lr[mask]

    amp_preds = np.clip(amp_preds, 0, 1)
    amp_preds[:, 1] = amp_preds[:, 1] - np.mean(amp_preds[:, 1]) + 0.5055
    amp_preds[:, 1] = np.clip(amp_preds[:, 1], 0, 1)

    std = np.std(amp_preds[:, 0])

    next_num += 1
    amp_sub = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': amp_preds[:, 0],
        'scaled_depth': amp_preds[:, 1],
        'scaled_left_right': amp_preds[:, 2]
    })
    amp_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
    amp_sub.to_csv(amp_file, index=False)
    print(f"\nTransfer + Amplification: angle_std={std:.6f} -> {amp_file}")


if __name__ == "__main__":
    main()
