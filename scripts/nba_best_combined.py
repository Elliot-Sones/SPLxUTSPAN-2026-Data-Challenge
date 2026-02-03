"""
NBA Best Combined Model

Combine the best features discovered from all NBA analyses:
1. Launch angle (-0.60 correlation)
2. Follow-through angle (-0.53)
3. vs_optimal_angle (-0.59)
4. Wrist peak features
5. Velocity components
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

# NBA optimal values from trajectory analysis
NBA_OPTIMAL_ENTRY_ANGLE = -58.9  # degrees


def get_keypoint_indices():
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_best_features(timeseries, keypoint_idx):
    """Extract only the best features from all NBA analyses."""
    features = {}
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # === VELOCITY FEATURES (best from launch_angle_focused.py) ===

    # Multi-frame average launch angle (most stable)
    angles = []
    for f_start, f_end in [(149, 152), (150, 153), (151, 154), (152, 155)]:
        w_start = get_joint("right_wrist", f_start)
        w_end = get_joint("right_wrist", f_end)
        if np.any(w_start) and np.any(w_end):
            v = w_end - w_start
            if abs(v[0]) > 0.001:
                angles.append(np.degrees(np.arctan2(v[2], abs(v[0]))))

    if angles:
        features['launch_angle'] = np.mean(angles)
        features['launch_angle_std'] = np.std(angles)  # Consistency

        # Compare to NBA optimal
        features['vs_nba_optimal'] = abs(features['launch_angle'] - NBA_OPTIMAL_ENTRY_ANGLE)

    # Basic velocity at release
    wrist_150 = get_joint("right_wrist", 150)
    wrist_153 = get_joint("right_wrist", 153)

    if np.any(wrist_150) and np.any(wrist_153):
        vel = wrist_153 - wrist_150
        features['release_vx'] = vel[0]
        features['release_vy'] = vel[1]
        features['release_vz'] = vel[2]
        features['release_speed'] = np.linalg.norm(vel)

    # === FOLLOW-THROUGH FEATURES (best from trajectory analysis) ===

    wrist_156 = get_joint("right_wrist", 156)
    wrist_159 = get_joint("right_wrist", 159)

    if np.any(wrist_153) and np.any(wrist_156):
        follow = wrist_156 - wrist_153
        features['followthrough_vz'] = follow[2]
        features['followthrough_speed'] = np.linalg.norm(follow)

        if abs(follow[0]) > 0.001:
            features['followthrough_angle'] = np.degrees(np.arctan2(follow[2], abs(follow[0])))

    # Extended follow-through
    if np.any(wrist_153) and np.any(wrist_159):
        follow_ext = wrist_159 - wrist_153
        if abs(follow_ext[0]) > 0.001:
            features['followthrough_angle_ext'] = np.degrees(np.arctan2(follow_ext[2], abs(follow_ext[0])))

    # === WRIST TRAJECTORY FEATURES (best from trajectory analysis) ===

    frames = list(range(145, 170))
    wrist_positions = [get_joint("right_wrist", f) for f in frames]
    valid = [(f, p) for f, p in zip(frames, wrist_positions) if np.any(p)]

    if len(valid) >= 5:
        pos_array = np.array([v[1] for v in valid])
        heights = pos_array[:, 2]

        features['wrist_peak_height'] = np.max(heights)
        mid_idx = len(heights) // 2
        features['wrist_release_height'] = heights[mid_idx] if mid_idx < len(heights) else heights[0]

        # Peak timing
        peak_idx = np.argmax(heights)
        features['wrist_peak_fraction'] = peak_idx / len(heights)

        # Curvature
        if len(heights) >= 3:
            curvature = np.diff(heights, n=2)
            features['wrist_curvature'] = np.mean(curvature)

    # === POSITION FEATURES (proven to work) ===

    shoulder = get_joint("right_shoulder", 153)
    features['shoulder_z_153'] = shoulder[2] if np.any(shoulder) else 0

    if np.any(wrist_153):
        features['wrist_z_153'] = wrist_153[2]

    if np.any(wrist_153) and np.any(shoulder):
        features['wrist_above_shoulder'] = wrist_153[2] - shoulder[2]

    return features


def main():
    print("="*80)
    print("NBA BEST COMBINED MODEL")
    print("="*80)

    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    train_features = []
    train_targets = []
    train_players = []

    print("\nExtracting features...")
    for metadata, timeseries in iterate_shots(train=True):
        features = extract_best_features(timeseries, keypoint_idx)
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
        features = extract_best_features(timeseries, keypoint_idx)
        test_features.append(features)
        test_ids.append(metadata['id'])

    X_train = pd.DataFrame(train_features).fillna(0)
    X_test = pd.DataFrame(test_features).fillna(0)

    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    y_train = np.array([[t['angle'], t['depth'], t['left_right']] for t in train_targets])
    players = np.array(train_players)

    print(f"Features: {len(common_cols)}")
    print(f"Feature names: {sorted(common_cols)}")

    # Correlations
    print("\nFeature correlations with angle (sorted by |corr|):")
    corrs = []
    for col in common_cols:
        corr = np.corrcoef(X_train[col].values, y_train[:, 0])[0, 1]
        corrs.append((col, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in corrs:
        print(f"  {col}: {corr:.4f}")

    # Train with different regularizations
    print("\n" + "="*60)
    print("TESTING REGULARIZATION")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for alpha in [1, 10, 50, 100]:
        cv_scores = []
        for target_idx in range(3):
            y = y_train[:, target_idx]
            gkf = GroupKFold(n_splits=5)
            scores = []
            for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
                model = Ridge(alpha=alpha)
                model.fit(X_train_scaled[train_idx], y[train_idx])
                pred = model.predict(X_train_scaled[val_idx])
                mse = np.mean((pred - y[val_idx])**2)
                scores.append(mse)
            cv_scores.append(np.mean(scores))
        print(f"Ridge alpha={alpha}: angle={cv_scores[0]:.4f}, depth={cv_scores[1]:.4f}, lr={cv_scores[2]:.4f}, mean={np.mean(cv_scores):.4f}")

    # Train final model with best alpha
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL (alpha=50)")
    print("="*60)

    predictions = np.zeros((len(X_test), 3))
    cv_scores = []

    for target_idx, target_name in enumerate(['angle', 'depth', 'left_right']):
        y = y_train[:, target_idx]
        gkf = GroupKFold(n_splits=5)
        fold_scores = []

        for train_idx, val_idx in gkf.split(X_train_scaled, y, players):
            model = Ridge(alpha=50)
            model.fit(X_train_scaled[train_idx], y[train_idx])
            pred = model.predict(X_train_scaled[val_idx])
            mse = np.mean((pred - y[val_idx])**2)
            fold_scores.append(mse)

        cv_score = np.mean(fold_scores)
        cv_scores.append(cv_score)
        print(f"  {target_name} CV MSE: {cv_score:.6f}")

        model = Ridge(alpha=50)
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
    corr219 = np.corrcoef(predictions[:, 0], sub219['scaled_angle'].values)[0, 1]
    print(f"Correlation with Sub 219: {corr219:.4f}")

    # Also load previous NBA models
    sub270 = pd.read_csv(SUBMISSION_DIR / "submission_270.csv")
    sub277 = pd.read_csv(SUBMISSION_DIR / "submission_277.csv")
    sub291 = pd.read_csv(SUBMISSION_DIR / "submission_291.csv")

    corr270 = np.corrcoef(predictions[:, 0], sub270['scaled_angle'].values)[0, 1]
    corr277 = np.corrcoef(predictions[:, 0], sub277['scaled_angle'].values)[0, 1]
    corr291 = np.corrcoef(predictions[:, 0], sub291['scaled_angle'].values)[0, 1]

    print(f"Correlation with Sub 270 (comprehensive): {corr270:.4f}")
    print(f"Correlation with Sub 277 (launch): {corr277:.4f}")
    print(f"Correlation with Sub 291 (trajectory): {corr291:.4f}")

    # Blends
    print("\n" + "="*60)
    print("BLENDING WITH SUB 219")
    print("="*60)

    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
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
        print(f"  {w:.0%} best + {1-w:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")

    # Ultimate blend: combine all NBA models
    print("\n" + "="*60)
    print("ULTIMATE NBA ENSEMBLE")
    print("="*60)

    # Weights for each model (based on diversity and quality)
    nba_preds = {
        'best': predictions,
        'comprehensive': sub270[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values,
        'launch': sub277[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values,
        'trajectory': sub291[['scaled_angle', 'scaled_depth', 'scaled_left_right']].values,
    }

    # Equal weight NBA ensemble
    nba_ensemble = np.zeros_like(predictions)
    for name, preds in nba_preds.items():
        nba_ensemble += preds / len(nba_preds)

    # Blend NBA ensemble with Sub 219
    for w_nba in [0.15, 0.20, 0.25, 0.30]:
        blend = pd.DataFrame({
            'id': test_ids,
            'scaled_angle': w_nba * nba_ensemble[:, 0] + (1-w_nba) * sub219['scaled_angle'].values,
            'scaled_depth': w_nba * nba_ensemble[:, 1] + (1-w_nba) * sub219['scaled_depth'].values,
            'scaled_left_right': w_nba * nba_ensemble[:, 2] + (1-w_nba) * sub219['scaled_left_right'].values,
        })

        blend['scaled_depth'] = blend['scaled_depth'] - blend['scaled_depth'].mean() + 0.5055
        blend['scaled_depth'] = blend['scaled_depth'].clip(0, 1)

        std = blend['scaled_angle'].std()

        next_num += 1
        blend_file = SUBMISSION_DIR / f"submission_{next_num}.csv"
        blend.to_csv(blend_file, index=False)
        print(f"  {w_nba:.0%} NBA-ensemble + {1-w_nba:.0%} Sub219: angle_std={std:.6f} -> {blend_file}")


if __name__ == "__main__":
    main()
