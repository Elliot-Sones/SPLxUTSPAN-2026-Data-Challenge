"""
Baseball Time-Series Transfer Learning

Use the FULL time-series joint angles from OpenBiomechanics,
not just the POI summary metrics, to learn release velocity patterns.

Key insight: Extract joint angles at exact ball release time (BR_time)
and learn the relationship to pitch_speed_mph.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
EXTERNAL_DIR = PROJECT_DIR / "external_data"
SUBMISSION_DIR = PROJECT_DIR / "submission"

np.random.seed(42)


def load_baseball_timeseries():
    """Load full time-series joint angles and extract features at ball release."""
    print("Loading baseball time-series data...")

    # Load POI metrics for pitch speed and timing info
    poi_df = pd.read_csv(EXTERNAL_DIR / "openbiomechanics/poi_metrics.csv")
    print(f"  POI metrics: {len(poi_df)} pitches")

    # Load full time-series
    angles_df = pd.read_csv(EXTERNAL_DIR / "openbiomechanics/joint_angles.csv")
    print(f"  Time-series: {len(angles_df)} rows")

    # Get unique pitches
    unique_pitches = angles_df['session_pitch'].unique()
    print(f"  Unique pitches in time-series: {len(unique_pitches)}")

    # Extract features at ball release for each pitch
    features_list = []
    targets = []

    for pitch_id in unique_pitches:
        # Get pitch data
        pitch_ts = angles_df[angles_df['session_pitch'] == pitch_id].copy()
        pitch_poi = poi_df[poi_df['session_pitch'] == pitch_id]

        if len(pitch_poi) == 0:
            continue

        pitch_speed = pitch_poi['pitch_speed_mph'].values[0]
        if pd.isna(pitch_speed):
            continue

        # Get ball release time
        br_time = pitch_ts['BR_time'].iloc[0]
        if pd.isna(br_time):
            continue

        # Get MER time (max external rotation) for comparison
        mer_time = pitch_ts['MER_time'].iloc[0]

        # Find frames closest to BR and MER
        pitch_ts['time_to_br'] = abs(pitch_ts['time'] - br_time)
        pitch_ts['time_to_mer'] = abs(pitch_ts['time'] - mer_time) if not pd.isna(mer_time) else 999

        br_frame = pitch_ts.loc[pitch_ts['time_to_br'].idxmin()]
        mer_frame = pitch_ts.loc[pitch_ts['time_to_mer'].idxmin()] if not pd.isna(mer_time) else None

        # Extract joint angles at ball release
        angle_cols = [c for c in pitch_ts.columns if 'angle' in c and c not in ['time', 'session_pitch']]

        features = {}

        # Joint angles at ball release
        for col in angle_cols:
            features[f'br_{col}'] = br_frame[col]

        # Joint angles at max external rotation (if available)
        if mer_frame is not None:
            for col in angle_cols:
                features[f'mer_{col}'] = mer_frame[col]
                # Change from MER to BR
                features[f'change_mer_br_{col}'] = br_frame[col] - mer_frame[col]

        # Velocity features: change in angles leading to release
        # Get frame before BR (about 50ms before)
        pre_br_mask = (pitch_ts['time'] >= br_time - 0.05) & (pitch_ts['time'] < br_time)
        if pre_br_mask.sum() > 0:
            pre_br_frame = pitch_ts[pre_br_mask].iloc[-1]
            dt = br_time - pre_br_frame['time']
            if dt > 0.001:
                for col in angle_cols:
                    features[f'velo_{col}'] = (br_frame[col] - pre_br_frame[col]) / dt

        features_list.append(features)
        targets.append(pitch_speed)

    print(f"  Extracted features for {len(features_list)} pitches")

    X = pd.DataFrame(features_list).fillna(0)
    y = np.array(targets)

    print(f"  Feature shape: {X.shape}")
    print(f"  Pitch speed range: {y.min():.1f} - {y.max():.1f} mph")

    return X, y


def train_baseball_model(X, y):
    """Train model to predict pitch speed from joint angles."""
    print("\nTraining baseball model...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different models
    models = {
        'Ridge(1)': Ridge(alpha=1.0),
        'Ridge(10)': Ridge(alpha=10.0),
        'Ridge(100)': Ridge(alpha=100.0),
        'Lasso(0.1)': Lasso(alpha=0.1, max_iter=5000),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GBR': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    print("\nCross-validation results (R2):")
    best_model = None
    best_score = -999

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        mean_score = scores.mean()
        print(f"  {name}: {mean_score:.3f} +/- {scores.std():.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    print(f"\nBest model: {type(best_model).__name__} with R2={best_score:.3f}")

    # Train best model on full data
    best_model.fit(X_scaled, y)

    # Get feature importances if available
    if hasattr(best_model, 'coef_'):
        coefs = best_model.coef_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(coefs)
        }).sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    return scaler, best_model, X.columns.tolist()


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def compute_angle_3d(p1, p2, p3):
    """Compute angle at p2 between p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def extract_basketball_angles(timeseries, keypoint_idx):
    """Extract joint angles from basketball data similar to baseball format."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Key frames: 150 is pre-release, 153 is release, 156 is post-release
    release_frame = 153
    pre_release_frame = 150
    post_release_frame = 156

    # Elbow angle (shoulder-elbow-wrist)
    for f, prefix in [(release_frame, 'br'), (pre_release_frame, 'mer')]:
        shoulder = get_joint("right_shoulder", f)
        elbow = get_joint("right_elbow", f)
        wrist = get_joint("right_wrist", f)

        if np.any(shoulder) and np.any(elbow) and np.any(wrist):
            elbow_angle = compute_angle_3d(shoulder, elbow, wrist)
            features[f'{prefix}_elbow_angle_x'] = elbow_angle

            # Simplified rotation angles using positions
            arm_vec = wrist - shoulder
            features[f'{prefix}_shoulder_angle_x'] = np.degrees(np.arctan2(arm_vec[2], arm_vec[0]))
            features[f'{prefix}_shoulder_angle_y'] = np.degrees(np.arctan2(arm_vec[2], arm_vec[1]))
            features[f'{prefix}_shoulder_angle_z'] = np.degrees(np.arctan2(arm_vec[1], arm_vec[0]))

            # Wrist position relative to elbow
            wrist_vec = wrist - elbow
            features[f'{prefix}_wrist_angle_x'] = np.degrees(np.arctan2(wrist_vec[2], wrist_vec[0]))
            features[f'{prefix}_wrist_angle_y'] = np.degrees(np.arctan2(wrist_vec[2], wrist_vec[1]))

    # Compute change from pre-release to release
    for key in list(features.keys()):
        if key.startswith('br_'):
            mer_key = 'mer_' + key[3:]
            if mer_key in features:
                features[f'change_mer_br_{key[3:]}'] = features[key] - features[mer_key]

    # Velocity features (similar to angular velocity)
    dt = 3 / 60  # 3 frames at 60fps
    for f1, f2, prefix in [(pre_release_frame, release_frame, 'velo')]:
        shoulder1 = get_joint("right_shoulder", f1)
        elbow1 = get_joint("right_elbow", f1)
        wrist1 = get_joint("right_wrist", f1)
        shoulder2 = get_joint("right_shoulder", f2)
        elbow2 = get_joint("right_elbow", f2)
        wrist2 = get_joint("right_wrist", f2)

        if np.any(elbow1) and np.any(elbow2):
            angle1 = compute_angle_3d(shoulder1, elbow1, wrist1)
            angle2 = compute_angle_3d(shoulder2, elbow2, wrist2)
            features[f'{prefix}_elbow_angle_x'] = (angle2 - angle1) / dt

            # Arm direction change
            arm1 = wrist1 - shoulder1
            arm2 = wrist2 - shoulder2
            features[f'{prefix}_shoulder_angle_x'] = (np.degrees(np.arctan2(arm2[2], arm2[0])) -
                                                       np.degrees(np.arctan2(arm1[2], arm1[0]))) / dt

    # Torso/pelvis angles (for rotation)
    for f, prefix in [(release_frame, 'br'), (pre_release_frame, 'mer')]:
        left_shoulder = get_joint("left_shoulder", f)
        right_shoulder = get_joint("right_shoulder", f)
        left_hip = get_joint("left_hip", f)
        right_hip = get_joint("right_hip", f)

        if np.any(left_shoulder) and np.any(right_shoulder):
            shoulder_vec = right_shoulder - left_shoulder
            features[f'{prefix}_torso_angle_x'] = np.degrees(np.arctan2(shoulder_vec[2], shoulder_vec[0]))
            features[f'{prefix}_torso_angle_y'] = np.degrees(np.arctan2(shoulder_vec[2], shoulder_vec[1]))
            features[f'{prefix}_torso_angle_z'] = np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))

        if np.any(left_hip) and np.any(right_hip):
            hip_vec = right_hip - left_hip
            features[f'{prefix}_pelvis_angle_x'] = np.degrees(np.arctan2(hip_vec[2], hip_vec[0]))
            features[f'{prefix}_pelvis_angle_y'] = np.degrees(np.arctan2(hip_vec[2], hip_vec[1]))
            features[f'{prefix}_pelvis_angle_z'] = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))

    # Simple position features (as backup)
    wrist = get_joint("right_wrist", release_frame)
    if np.any(wrist):
        features['wrist_z_release'] = wrist[2]
        features['wrist_x_release'] = wrist[0]

    return features


def main():
    print("="*80)
    print("BASEBALL TIME-SERIES TRANSFER LEARNING")
    print("="*80)

    # Step 1: Load and train on baseball data
    X_baseball, y_baseball = load_baseball_timeseries()
    baseball_scaler, baseball_model, baseball_features = train_baseball_model(X_baseball, y_baseball)

    # Step 2: Load basketball data and extract similar features
    print("\n" + "="*60)
    print("EXTRACTING BASKETBALL FEATURES")
    print("="*60)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []

    for metadata, timeseries in iterate_shots(train=True):
        features = extract_basketball_angles(timeseries, keypoint_idx)
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
        features = extract_basketball_angles(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Basketball features: {len(common_cols)}")

    # Correlation with targets
    print("\nFeature correlations with angle (top 10):")
    corrs = []
    for col in common_cols:
        corr = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
        if not np.isnan(corr):
            corrs.append((col, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs[:10]:
        print(f"  {col}: {corr:.4f}")

    # Step 3: Train basketball model
    print("\n" + "="*60)
    print("TRAINING BASKETBALL MODEL")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for alpha in [1, 10, 50, 100]:
        scores = []
        for i in range(3):
            y = y_train[:, i]
            gkf = GroupKFold(n_splits=5)
            fold_scores = []
            for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled[train_idx], y[train_idx])
                pred = model.predict(X_train_scaled[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                fold_scores.append(mse)
            scores.append(np.mean(fold_scores))
        print(f"Ridge alpha={alpha}: angle={scores[0]:.4f}, depth={scores[1]:.4f}, lr={scores[2]:.4f}, mean={np.mean(scores):.4f}")

    # Train final model with best alpha
    best_alpha = 50
    for i, target in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, i]

        gkf = GroupKFold(n_splits=5)
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
            model = Ridge(alpha=best_alpha)
            model.fit(X_train_scaled[train_idx], y[train_idx])
            pred = model.predict(X_train_scaled[val_idx])
            mse = np.mean((pred - y[val_idx])**2)
            fold_scores.append(mse)

        cv_score = np.mean(fold_scores)
        cv_scores.append(cv_score)

        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, y)
        predictions[:, i] = model.predict(X_test_scaled)

    print(f"\nFinal CV: angle={cv_scores[0]:.4f}, depth={cv_scores[1]:.4f}, lr={cv_scores[2]:.4f}")
    print(f"Mean CV: {np.mean(cv_scores):.6f}")

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

    # Compare with existing submissions
    sub219 = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    sub133 = pd.read_csv(SUBMISSION_DIR / "submission_133.csv")

    corr219 = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    corr133 = np.corrcoef(predictions[:, 0], sub133['scaled_angle'].values)[0, 1]

    print(f"\nCorrelation with Sub 219: {corr219:.4f}")
    print(f"Correlation with Sub 133: {corr133:.4f}")

    # Blend with Sub 219
    print("\n" + "="*60)
    print("BLENDING WITH SUB 219")
    print("="*60)

    for w in [0.10, 0.15, 0.20, 0.25]:
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
