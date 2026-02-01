"""
PHYSICS VS BASELINE CV COMPARISON

Compare the best physics features against the baseline model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def extract_baseline_features(X_player, col_to_idx, target_name):
    """Extract baseline features (frame 153 for all, simple positions)."""
    n_samples = X_player.shape[0]

    if target_name == 'angle':
        frame = 153
    elif target_name == 'depth':
        frame = 102
    else:  # left_right
        frame = 153

    # Simple baseline: key body part positions at fixed frame
    parts = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip', 'right_knee']
    features = []

    for part in parts:
        cols = get_cols(col_to_idx, part)
        if cols:
            features.append(X_player[:, frame, cols])

    if features:
        return np.column_stack(features)
    return None


def extract_physics_features(X_player, col_to_idx, target_name, player_id):
    """Extract the best physics features per player per target."""
    n_samples = X_player.shape[0]

    # Best configs from exhaustive testing
    if target_name == 'angle':
        if player_id == 1:
            # Combined fingertip features at frame 165
            frame = 165
            window = 3
            features = []
            # Fingertip positions
            for part in ['right_second_finger_distal', 'right_third_finger_distal']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
                    # Velocity
                    if frame + window < X_player.shape[1]:
                        vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                        features.append(vel)
            # Wrist snap
            for axis in ['x', 'y', 'z']:
                w_col = f"right_wrist_{axis}"
                e_col = f"right_elbow_{axis}"
                if w_col in col_to_idx and e_col in col_to_idx:
                    w_vel = (X_player[:, frame + window, col_to_idx[w_col]] - X_player[:, frame, col_to_idx[w_col]]) / window
                    e_vel = (X_player[:, frame + window, col_to_idx[e_col]] - X_player[:, frame, col_to_idx[e_col]]) / window
                    features.append((w_vel - e_vel).reshape(-1, 1))
            # Guide hand
            cols = get_cols(col_to_idx, 'left_wrist')
            if cols:
                features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        elif player_id == 2:
            # Left knee pos+vel at frame 110
            frame = 110
            window = 5
            features = []
            cols = get_cols(col_to_idx, 'left_knee')
            if cols:
                features.append(X_player[:, frame, cols])
                vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                features.append(vel)
            # Add left wrist
            cols = get_cols(col_to_idx, 'left_wrist')
            if cols:
                features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        elif player_id == 3:
            # Left fourth finger distal velocity at frame 105
            frame = 105
            window = 5
            features = []
            cols = get_cols(col_to_idx, 'left_fourth_finger_distal')
            if cols and frame + window < X_player.shape[1]:
                vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                features.append(vel)
            # Add position
            if cols:
                features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        elif player_id == 4:
            # Right shoulder + right knee at frame 160
            frame = 160
            features = []
            for part in ['right_shoulder', 'right_knee']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
            # Add velocity
            window = 5
            for part in ['right_shoulder', 'right_knee']:
                cols = get_cols(col_to_idx, part)
                if cols and frame + window < X_player.shape[1]:
                    vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                    features.append(vel)
            return np.column_stack(features) if features else None

        else:  # player_id == 5
            # Combined features at frame 155
            frame = 155
            window = 3
            features = []
            # Positions
            for part in ['right_wrist', 'left_wrist', 'right_hip']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
            # Asymmetry
            for right, left in [('right_wrist', 'left_wrist'), ('right_shoulder', 'left_shoulder'), ('right_hip', 'left_hip')]:
                for axis in ['x', 'y', 'z']:
                    r_col = f"{right}_{axis}"
                    l_col = f"{left}_{axis}"
                    if r_col in col_to_idx and l_col in col_to_idx:
                        diff = X_player[:, frame, col_to_idx[r_col]] - X_player[:, frame, col_to_idx[l_col]]
                        features.append(diff.reshape(-1, 1))
            return np.column_stack(features) if features else None

    elif target_name == 'depth':
        if player_id == 1:
            # Multi-frame fingertip at frames 150-160
            features = []
            for frame in [150, 155, 160]:
                cols = get_cols(col_to_idx, 'right_second_finger_distal')
                if cols:
                    features.append(X_player[:, frame, cols])
            # Add arm chain
            for part in ['right_wrist', 'right_elbow', 'right_shoulder']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, 150, cols])
            return np.column_stack(features) if features else None

        elif player_id == 2:
            # Right elbow velocity at frame 95
            frame = 95
            window = 10
            features = []
            cols = get_cols(col_to_idx, 'right_elbow')
            if cols and frame + window < X_player.shape[1]:
                features.append(X_player[:, frame, cols])
                vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                features.append(vel)
            # Add wrist
            cols = get_cols(col_to_idx, 'right_wrist')
            if cols:
                features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        elif player_id == 3:
            # Left ankle velocity at frame 130
            frame = 130
            window = 2
            features = []
            cols = get_cols(col_to_idx, 'left_ankle')
            if cols and frame + window < X_player.shape[1]:
                features.append(X_player[:, frame, cols])
                vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                features.append(vel)
            # Add hip
            cols = get_cols(col_to_idx, 'right_hip')
            if cols:
                features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        elif player_id == 4:
            # Arm chain full at frame 135
            frame = 135
            window = 5
            features = []
            for part in ['right_wrist', 'right_elbow', 'right_shoulder']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
                    if frame + window < X_player.shape[1]:
                        vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                        features.append(vel)
            return np.column_stack(features) if features else None

        else:  # player_id == 5
            # Finger MCP position at frame 145
            frame = 145
            features = []
            cols = get_cols(col_to_idx, 'right_second_finger_mcp')
            if cols:
                features.append(X_player[:, frame, cols])
            # Add wrist
            cols = get_cols(col_to_idx, 'right_wrist')
            if cols:
                features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

    else:  # left_right
        if player_id == 1:
            # Guide hand at frame 175
            frame = 175
            features = []
            for part in ['left_wrist', 'left_second_finger_distal', 'right_knee']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        elif player_id == 2:
            # Both wrists + hip at frame 150
            frame = 150
            features = []
            for part in ['right_wrist', 'left_wrist', 'right_hip']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        elif player_id == 3:
            # Body alignment at frame 175
            frame = 175
            features = []
            for right, left in [('right_shoulder', 'left_shoulder'), ('right_hip', 'left_hip'), ('right_elbow', 'left_elbow')]:
                for axis in ['x', 'y', 'z']:
                    r_col = f"{right}_{axis}"
                    l_col = f"{left}_{axis}"
                    if r_col in col_to_idx and l_col in col_to_idx:
                        diff = X_player[:, frame, col_to_idx[r_col]] - X_player[:, frame, col_to_idx[l_col]]
                        features.append(diff.reshape(-1, 1))
            return np.column_stack(features) if features else None

        elif player_id == 4:
            # Arm chain at frame 150
            frame = 150
            features = []
            for part in ['right_second_finger_distal', 'right_wrist', 'right_elbow']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
            return np.column_stack(features) if features else None

        else:  # player_id == 5
            # Wrists + hip + velocity at frame 170
            frame = 170
            window = 18
            features = []
            for part in ['right_wrist', 'left_wrist', 'right_hip']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X_player[:, frame, cols])
            # Add velocity
            cols = get_cols(col_to_idx, 'right_wrist')
            if cols and frame + window < X_player.shape[1]:
                vel = (X_player[:, frame + window, cols] - X_player[:, frame, cols]) / window
                features.append(vel)
            return np.column_stack(features) if features else None

    return None


def cv_evaluate(X_features, y, model_type='ridge'):
    """Evaluate with 5-fold CV, return MSE and R²."""
    if X_features is None or len(y) < 15:
        return {'mse': np.nan, 'r2': np.nan}

    valid = ~(np.isnan(X_features).any(axis=1) | np.isnan(y))
    if valid.sum() < 15:
        return {'mse': np.nan, 'r2': np.nan}

    X_clean = X_features[valid]
    y_clean = y[valid]

    preds = np.zeros(len(y_clean))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X_clean):
        X_tr, X_te = X_clean[train_idx], X_clean[test_idx]
        y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42)
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=42)

        model.fit(X_tr_s, y_tr)
        preds[test_idx] = model.predict(X_te_s)

    mse = np.mean((preds - y_clean) ** 2)
    r2 = 1 - mse / np.var(y_clean)

    return {'mse': mse, 'r2': r2, 'preds': preds, 'y_true': y_clean}


def main():
    print("=" * 80)
    print("PHYSICS VS BASELINE CV COMPARISON")
    print("=" * 80)

    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2]
    }

    results = []

    for target_name, target_values in targets.items():
        print(f"\n{'='*80}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*80}")

        total_baseline_mse = 0
        total_physics_mse = 0
        total_samples = 0

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target_values[mask]
            n_samples = mask.sum()

            print(f"\nPlayer {pid} (n={n_samples}):")

            # Baseline features
            baseline_features = extract_baseline_features(X_player, col_to_idx, target_name)

            # Physics features
            physics_features = extract_physics_features(X_player, col_to_idx, target_name, pid)

            # Mean baseline (predicting player mean)
            mean_mse = np.var(y_player)

            # Evaluate baseline
            baseline_ridge = cv_evaluate(baseline_features, y_player, 'ridge')
            baseline_rf = cv_evaluate(baseline_features, y_player, 'rf')

            # Evaluate physics
            physics_ridge = cv_evaluate(physics_features, y_player, 'ridge')
            physics_rf = cv_evaluate(physics_features, y_player, 'rf')

            # Best models
            baseline_best_mse = min(baseline_ridge['mse'], baseline_rf['mse'])
            baseline_best_r2 = max(baseline_ridge['r2'], baseline_rf['r2'])
            baseline_best_model = 'ridge' if baseline_ridge['mse'] <= baseline_rf['mse'] else 'rf'

            physics_best_mse = min(physics_ridge['mse'], physics_rf['mse'])
            physics_best_r2 = max(physics_ridge['r2'], physics_rf['r2'])
            physics_best_model = 'ridge' if physics_ridge['mse'] <= physics_rf['mse'] else 'rf'

            # Improvement
            improvement = (baseline_best_mse - physics_best_mse) / baseline_best_mse * 100 if baseline_best_mse > 0 else 0

            print(f"  Mean baseline MSE: {mean_mse:.4f}")
            print(f"  Baseline ({baseline_best_model}): MSE={baseline_best_mse:.4f}, R²={baseline_best_r2:.3f}")
            print(f"  Physics ({physics_best_model}):  MSE={physics_best_mse:.4f}, R²={physics_best_r2:.3f}")
            print(f"  Improvement: {improvement:+.1f}%")

            total_baseline_mse += baseline_best_mse * n_samples
            total_physics_mse += physics_best_mse * n_samples
            total_samples += n_samples

            results.append({
                'target': target_name,
                'player': pid,
                'n_samples': n_samples,
                'mean_mse': mean_mse,
                'baseline_mse': baseline_best_mse,
                'baseline_r2': baseline_best_r2,
                'baseline_model': baseline_best_model,
                'physics_mse': physics_best_mse,
                'physics_r2': physics_best_r2,
                'physics_model': physics_best_model,
                'improvement_pct': improvement
            })

        # Weighted average MSE
        avg_baseline_mse = total_baseline_mse / total_samples
        avg_physics_mse = total_physics_mse / total_samples
        total_improvement = (avg_baseline_mse - avg_physics_mse) / avg_baseline_mse * 100

        print(f"\n{target_name.upper()} SUMMARY:")
        print(f"  Weighted Avg Baseline MSE: {avg_baseline_mse:.4f}")
        print(f"  Weighted Avg Physics MSE:  {avg_physics_mse:.4f}")
        print(f"  Overall Improvement: {total_improvement:+.1f}%")

    # Final summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for target_name in ['angle', 'depth', 'left_right']:
        target_results = results_df[results_df['target'] == target_name]

        avg_baseline_r2 = target_results['baseline_r2'].mean()
        avg_physics_r2 = target_results['physics_r2'].mean()
        avg_improvement = target_results['improvement_pct'].mean()

        # Weighted MSE
        total_samples = target_results['n_samples'].sum()
        weighted_baseline_mse = (target_results['baseline_mse'] * target_results['n_samples']).sum() / total_samples
        weighted_physics_mse = (target_results['physics_mse'] * target_results['n_samples']).sum() / total_samples

        print(f"\n{target_name.upper()}:")
        print(f"  Baseline: Avg R²={avg_baseline_r2:.3f}, Weighted MSE={weighted_baseline_mse:.4f}")
        print(f"  Physics:  Avg R²={avg_physics_r2:.3f}, Weighted MSE={weighted_physics_mse:.4f}")
        print(f"  Improvement: {avg_improvement:+.1f}% MSE reduction")

    # Overall comparison table
    print()
    print("=" * 80)
    print("DETAILED COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Target':<12} {'Player':<8} {'Baseline R²':<12} {'Physics R²':<12} {'Improvement':<12}")
    print("-" * 60)

    for _, row in results_df.iterrows():
        print(f"{row['target']:<12} {row['player']:<8} {row['baseline_r2']:.3f}        {row['physics_r2']:.3f}        {row['improvement_pct']:+.1f}%")

    # Save
    results_df.to_csv(OUTPUT_DIR / 'physics_vs_baseline_cv.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'physics_vs_baseline_cv.csv'}")


if __name__ == "__main__":
    main()
