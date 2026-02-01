"""
OPTIMAL PHYSICS MODEL

Find the EXACT optimal features for each player for each target,
build a model, test it, and estimate LB score.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DATA_DIR = Path(__file__).parent.parent / 'data'


class KeypointExtractor:
    def __init__(self, keypoint_cols: List[str]):
        self.keypoint_cols = keypoint_cols
        self.col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    def get(self, ts: np.ndarray, name: str, frame: int = None) -> np.ndarray:
        idx = self.col_to_idx.get(name)
        if idx is None:
            return np.nan if frame is not None else np.full(ts.shape[0], np.nan)
        if frame is not None:
            if frame < 0 or frame >= ts.shape[0]:
                return np.nan
            return ts[frame, idx]
        return ts[:, idx]


def extract_single_feature(ts: np.ndarray, kp: KeypointExtractor,
                           feat_spec: Dict[str, Any]) -> float:
    """Extract a single feature based on specification."""
    feat_type = feat_spec['type']

    if feat_type == 'position':
        return kp.get(ts, feat_spec['keypoint'], feat_spec['frame'])

    elif feat_type == 'velocity':
        val_start = kp.get(ts, feat_spec['keypoint'], feat_spec['start'])
        val_end = kp.get(ts, feat_spec['keypoint'], feat_spec['end'])
        if np.isnan(val_start) or np.isnan(val_end):
            return np.nan
        return (val_end - val_start) / (feat_spec['end'] - feat_spec['start'])

    elif feat_type == 'knee_angle':
        frame = feat_spec['frame']
        hip = kp.get(ts, 'right_hip_z', frame)
        knee = kp.get(ts, 'right_knee_z', frame)
        ankle = kp.get(ts, 'right_ankle_z', frame)
        if np.isnan(hip) or np.isnan(knee) or np.isnan(ankle):
            return np.nan
        return (hip - knee) + (ankle - knee)

    elif feat_type == 'elbow_angle':
        frame = feat_spec['frame']
        shoulder = kp.get(ts, 'right_shoulder_z', frame)
        elbow = kp.get(ts, 'right_elbow_z', frame)
        wrist = kp.get(ts, 'right_wrist_z', frame)
        if np.isnan(shoulder) or np.isnan(elbow) or np.isnan(wrist):
            return np.nan
        return (shoulder - elbow) + (wrist - elbow)

    elif feat_type == 'diff':
        val1 = kp.get(ts, feat_spec['keypoint1'], feat_spec['frame'])
        val2 = kp.get(ts, feat_spec['keypoint2'], feat_spec['frame'])
        if np.isnan(val1) or np.isnan(val2):
            return np.nan
        return val1 - val2

    return np.nan


def within_player_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     alpha: float = 1.0) -> Tuple[float, float]:
    """Within-player CV for a single player."""
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid]
    y_clean = y[valid]

    if len(y_clean) < n_splits * 2:
        return np.nan, np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_r2s = []

    for train_idx, test_idx in kf.split(X_clean):
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train, y_test = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        test_r2s.append(model.score(X_test_scaled, y_test))

    return np.mean(test_r2s), np.std(test_r2s)


def find_optimal_features(X_raw: np.ndarray, y: np.ndarray, kp: KeypointExtractor,
                          max_features: int = 5) -> Tuple[List[Dict], float]:
    """Find optimal features for a single player-target combination using greedy forward selection."""

    # Generate all candidate features
    candidates = []

    key_kps = [
        'right_wrist_z', 'left_wrist_z',
        'right_elbow_z', 'left_elbow_z',
        'right_shoulder_z', 'left_shoulder_z',
        'right_knee_z', 'left_knee_z',
        'right_ankle_z', 'left_ankle_z',
        'right_hip_z', 'left_hip_z',
    ]

    # Position features
    for kp_name in key_kps:
        for frame in range(60, 220, 5):
            candidates.append({
                'type': 'position',
                'keypoint': kp_name,
                'frame': frame,
                'name': f'{kp_name}_f{frame}'
            })

    # Velocity features
    for kp_name in key_kps:
        for start in range(60, 190, 10):
            for window in [5, 10, 15, 20, 30, 40]:
                end = start + window
                if end > 220:
                    continue
                candidates.append({
                    'type': 'velocity',
                    'keypoint': kp_name,
                    'start': start,
                    'end': end,
                    'name': f'{kp_name}_vel_{start}_{end}'
                })

    # Joint angles
    for frame in range(80, 200, 10):
        candidates.append({
            'type': 'knee_angle',
            'frame': frame,
            'name': f'knee_angle_f{frame}'
        })
        candidates.append({
            'type': 'elbow_angle',
            'frame': frame,
            'name': f'elbow_angle_f{frame}'
        })

    # Height differences
    for frame in range(100, 180, 10):
        candidates.append({
            'type': 'diff',
            'keypoint1': 'right_wrist_z',
            'keypoint2': 'right_hip_z',
            'frame': frame,
            'name': f'wrist_hip_diff_f{frame}'
        })

    # Extract all candidate features
    n_samples = len(y)
    feature_matrix = np.zeros((n_samples, len(candidates)))

    for i in range(n_samples):
        for j, feat_spec in enumerate(candidates):
            feature_matrix[i, j] = extract_single_feature(X_raw[i], kp, feat_spec)

    # Greedy forward selection
    selected_indices = []
    selected_features = []
    best_score = -999

    for n_feat in range(max_features):
        best_idx = None
        best_new_score = -999

        for j in range(len(candidates)):
            if j in selected_indices:
                continue

            # Check if feature is valid
            if np.isnan(feature_matrix[:, j]).mean() > 0.1:
                continue

            # Try adding this feature
            test_indices = selected_indices + [j]
            X_test = feature_matrix[:, test_indices]

            score, _ = within_player_cv(X_test, y, alpha=1.0)

            if not np.isnan(score) and score > best_new_score:
                best_new_score = score
                best_idx = j

        if best_idx is not None and best_new_score > best_score - 0.02:
            selected_indices.append(best_idx)
            selected_features.append(candidates[best_idx])
            best_score = best_new_score
        else:
            break

    return selected_features, best_score


def main():
    print("=" * 80)
    print("OPTIMAL PHYSICS MODEL - Finding Best Features Per Player Per Target")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = KeypointExtractor(keypoint_cols)

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2],
    }

    print(f"Loaded {len(y)} shots from {len(np.unique(participant_ids))} players")
    print()

    # ==========================================================================
    # Find optimal features for each player-target combination
    # ==========================================================================
    optimal_features = {}
    optimal_scores = {}

    for target_name, target in targets.items():
        print(f"\n{'='*80}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*80}")

        optimal_features[target_name] = {}
        optimal_scores[target_name] = {}

        for pid in sorted(np.unique(participant_ids)):
            print(f"\nPlayer {pid}:")
            print("-" * 60)

            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target[mask]

            # Find optimal features
            features, score = find_optimal_features(X_player, y_player, kp, max_features=5)

            optimal_features[target_name][pid] = features
            optimal_scores[target_name][pid] = score

            print(f"Best Test R²: {score:.4f}")
            print(f"Optimal features ({len(features)}):")
            for feat in features:
                print(f"  - {feat['name']}")

    # ==========================================================================
    # Summary table
    # ==========================================================================
    print()
    print("=" * 80)
    print("OPTIMAL FEATURES SUMMARY")
    print("=" * 80)
    print()

    summary_data = []
    for target_name in ['angle', 'depth', 'left_right']:
        for pid in sorted(np.unique(participant_ids)):
            features = optimal_features[target_name][pid]
            score = optimal_scores[target_name][pid]
            feature_names = [f['name'] for f in features]

            summary_data.append({
                'Target': target_name,
                'Player': pid,
                'Test R²': score,
                'N Features': len(features),
                'Features': ', '.join(feature_names[:3]) + ('...' if len(feature_names) > 3 else '')
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # ==========================================================================
    # Cross-validation to estimate CV score (mimics train/test split)
    # ==========================================================================
    print()
    print("=" * 80)
    print("FULL MODEL CV ESTIMATION")
    print("=" * 80)
    print()

    # For each target, train per-player models and get overall CV score
    cv_predictions = {t: np.zeros(len(y)) for t in targets}
    cv_actuals = {t: targets[t].copy() for t in targets}

    for target_name, target in targets.items():
        print(f"\n{target_name.upper()}:")

        all_preds = np.zeros(len(y))

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target[mask]
            features = optimal_features[target_name][pid]

            if len(features) == 0:
                # Fallback to mean
                all_preds[mask] = y_player.mean()
                continue

            # Extract features
            X_feat = np.zeros((mask.sum(), len(features)))
            for i, idx in enumerate(np.where(mask)[0]):
                for j, feat_spec in enumerate(features):
                    X_feat[i, j] = extract_single_feature(X[idx], kp, feat_spec)

            # 5-fold CV
            valid = ~(np.isnan(X_feat).any(axis=1) | np.isnan(y_player))
            X_clean = X_feat[valid]
            y_clean = y_player[valid]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            fold_preds = np.zeros(len(y_player))
            fold_preds[:] = np.nan

            for train_idx, test_idx in kf.split(X_clean):
                X_train, X_test = X_clean[train_idx], X_clean[test_idx]
                y_train, y_test = y_clean[train_idx], y_clean[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, y_train)

                # Map back to original indices
                valid_indices = np.where(valid)[0]
                for ti, orig_idx in enumerate(test_idx):
                    fold_preds[valid_indices[orig_idx]] = model.predict(X_test_scaled[ti:ti+1])[0]

            # Fill NaN with mean
            fold_preds[np.isnan(fold_preds)] = y_player.mean()
            all_preds[mask] = fold_preds

        cv_predictions[target_name] = all_preds

        # Calculate MSE for this target
        mse = np.mean((all_preds - target) ** 2)
        r2 = 1 - mse / np.var(target)
        print(f"  CV MSE: {mse:.6f}")
        print(f"  CV R²:  {r2:.4f}")

    # ==========================================================================
    # Estimate LB score
    # ==========================================================================
    print()
    print("=" * 80)
    print("ESTIMATED LEADERBOARD SCORE")
    print("=" * 80)
    print()

    # Load scalers to convert to scaled space (LB uses scaled targets)
    try:
        import joblib
        scalers = {}
        for t in ['angle', 'depth', 'left_right']:
            scalers[t] = joblib.load(DATA_DIR / f'scaler_{t}.pkl')

        # Scale predictions and actuals
        total_mse = 0
        for target_name in ['angle', 'depth', 'left_right']:
            preds = cv_predictions[target_name]
            actuals = cv_actuals[target_name]

            # Scale
            preds_scaled = scalers[target_name].transform(preds.reshape(-1, 1)).ravel()
            actuals_scaled = scalers[target_name].transform(actuals.reshape(-1, 1)).ravel()

            mse = np.mean((preds_scaled - actuals_scaled) ** 2)
            total_mse += mse
            print(f"{target_name}: Scaled MSE = {mse:.6f}")

        avg_mse = total_mse / 3
        print()
        print(f"Average Scaled MSE (CV estimate): {avg_mse:.6f}")
        print()
        print("Note: Actual LB score may differ due to:")
        print("- Different test set distribution")
        print("- CV optimism (features selected on same data)")
        print()
        print(f"PREDICTED LB RANGE: {avg_mse * 0.9:.6f} to {avg_mse * 1.3:.6f}")

    except Exception as e:
        print(f"Could not load scalers: {e}")
        print("Cannot estimate scaled MSE")

    # ==========================================================================
    # Save optimal features
    # ==========================================================================
    output_path = OUTPUT_DIR / 'optimal_physics_features.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'features': optimal_features,
            'scores': optimal_scores,
        }, f)

    print()
    print(f"Optimal features saved to {output_path}")

    # Also save as readable format
    readable_path = OUTPUT_DIR / 'optimal_physics_features.txt'
    with open(readable_path, 'w') as f:
        for target_name in ['angle', 'depth', 'left_right']:
            f.write(f"\n{'='*60}\n")
            f.write(f"TARGET: {target_name.upper()}\n")
            f.write(f"{'='*60}\n")

            for pid in sorted(np.unique(participant_ids)):
                features = optimal_features[target_name][pid]
                score = optimal_scores[target_name][pid]

                f.write(f"\nPlayer {pid} (Test R² = {score:.4f}):\n")
                for feat in features:
                    f.write(f"  {feat}\n")

    print(f"Readable format saved to {readable_path}")


if __name__ == "__main__":
    main()
