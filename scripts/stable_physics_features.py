"""
Find Stable Physics Features

Goal: Features that have good CV score AND low prediction variance.
High angle_std correlates with worse LB scores, so we want lower variance.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

TARGETS = ['angle', 'depth', 'left_right']

TARGET_SCALERS = {
    'angle': joblib.load(DATA_DIR / 'scaler_angle.pkl'),
    'depth': joblib.load(DATA_DIR / 'scaler_depth.pkl'),
    'left_right': joblib.load(DATA_DIR / 'scaler_left_right.pkl'),
}


def get_cols(col_to_idx, body_part):
    return [col_to_idx[f"{body_part}_{a}"] for a in ['x', 'y', 'z'] if f"{body_part}_{a}" in col_to_idx]


def compute_scaled_mse(preds, truth, target_name):
    scaler = TARGET_SCALERS[target_name]
    scaled_preds = np.clip(scaler.transform(preds.reshape(-1, 1)).flatten(), 0, 1)
    scaled_truth = np.clip(scaler.transform(truth.reshape(-1, 1)).flatten(), 0, 1)
    return np.mean((scaled_preds - scaled_truth) ** 2)


def within_player_cv(X_features, y_target, pids, target_name, alpha=1.0):
    """
    Standard within-player 5-fold CV.
    Returns: MSE, prediction std (in scaled space)
    """
    unique_pids = sorted(np.unique(pids))
    all_preds = np.zeros(len(y_target))

    for pid in unique_pids:
        mask = pids == pid
        X_player = np.nan_to_num(X_features[mask], nan=0.0)
        y_player = y_target[mask]

        # Remove constant features
        std = np.std(X_player, axis=0)
        valid_cols = std > 1e-10
        if valid_cols.sum() == 0:
            all_preds[mask] = np.mean(y_player)
            continue

        X_player = X_player[:, valid_cols]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        player_preds = np.zeros(len(y_player))

        for train_idx, val_idx in kf.split(X_player):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_player[train_idx])
            X_val = scaler.transform(X_player[val_idx])

            model = Ridge(alpha=alpha)
            model.fit(X_tr, y_player[train_idx])
            player_preds[val_idx] = model.predict(X_val)

        all_preds[mask] = player_preds

    # Compute MSE and std in scaled space
    scaler = TARGET_SCALERS[target_name]
    scaled_preds = np.clip(scaler.transform(all_preds.reshape(-1, 1)).flatten(), 0, 1)
    scaled_truth = np.clip(scaler.transform(y_target.reshape(-1, 1)).flatten(), 0, 1)

    mse = np.mean((scaled_preds - scaled_truth) ** 2)
    pred_std = np.std(scaled_preds)

    return mse, pred_std, scaled_preds


def main():
    print("=" * 70)
    print("FINDING STABLE PHYSICS FEATURES")
    print("Goal: Good CV + Low variance (low angle_std)")
    print("=" * 70)

    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}
    pids = meta['participant_id'].values

    # Reference: Sub9 stats
    print("\nReference (Sub9):")
    print("  angle_std: 0.1390")
    print("  LB Score:  0.009109")

    # Reference: Previous physics stats
    print("\nPrevious Physics:")
    print("  angle_std: 0.1822 (too high)")
    print("  CV Score:  0.006372")

    results = []

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n{'='*70}")
        print(f"TARGET: {target_name.upper()}")
        print(f"{'='*70}")

        y_target = y[:, target_idx]

        # Test various feature configs
        configs_to_test = []

        # Single body parts at various frames
        for part in ['right_wrist', 'right_elbow', 'right_shoulder', 'left_wrist',
                     'right_hip', 'right_knee', 'neck', 'nose', 'mid_hip']:
            for frame in [100, 120, 140, 150, 160]:
                configs_to_test.append({
                    'name': f"{part}_f{frame}",
                    'parts': [part],
                    'frame': frame,
                    'velocity': False
                })

        # With velocity
        for part in ['right_wrist', 'right_elbow']:
            for frame in [120, 140, 150]:
                for window in [5, 10]:
                    configs_to_test.append({
                        'name': f"{part}_f{frame}_vel{window}",
                        'parts': [part],
                        'frame': frame,
                        'velocity': True,
                        'window': window
                    })

        # Pairs
        pairs = [
            (['right_wrist', 'right_elbow'], 'wrist+elbow'),
            (['right_wrist', 'left_wrist'], 'both_wrists'),
            (['right_shoulder', 'right_hip'], 'shoulder+hip'),
        ]
        for parts, name in pairs:
            for frame in [140, 150, 160]:
                configs_to_test.append({
                    'name': f"{name}_f{frame}",
                    'parts': parts,
                    'frame': frame,
                    'velocity': False
                })

        best_mse = float('inf')
        best_config = None
        best_std = None

        for config in configs_to_test:
            # Extract features
            features = []
            frame = config['frame']

            for part in config['parts']:
                cols = get_cols(col_to_idx, part)
                if cols:
                    features.append(X[:, frame, cols])

                    if config.get('velocity', False):
                        window = config.get('window', 5)
                        if frame + window < X.shape[1]:
                            vel = (X[:, frame + window, cols] - X[:, frame, cols]) / window
                            features.append(vel)

            if not features:
                continue

            feat_array = np.hstack(features)

            # Test with different regularization strengths
            for alpha in [1.0, 5.0, 10.0]:
                mse, pred_std, _ = within_player_cv(feat_array, y_target, pids, target_name, alpha)

                results.append({
                    'target': target_name,
                    'config': config['name'],
                    'alpha': alpha,
                    'cv_mse': mse,
                    'pred_std': pred_std,
                    'n_features': feat_array.shape[1]
                })

                # For angle, prioritize low std
                if target_name == 'angle':
                    # Score combines MSE and std penalty
                    score = mse + 0.1 * max(0, pred_std - 0.14)  # Penalty if std > 0.14
                else:
                    score = mse

                if score < best_mse:
                    best_mse = score
                    best_config = f"{config['name']}_a{alpha}"
                    best_std = pred_std

        print(f"\nBest for {target_name}:")
        print(f"  Config: {best_config}")
        print(f"  pred_std: {best_std:.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: BEST CONFIGS BY CV MSE")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    for target_name in TARGETS:
        target_df = results_df[results_df['target'] == target_name]
        print(f"\n{target_name.upper()} - Top 10 by CV MSE:")

        top10 = target_df.nsmallest(10, 'cv_mse')
        for _, row in top10.iterrows():
            print(f"  {row['config']} (a={row['alpha']}): MSE={row['cv_mse']:.6f}, std={row['pred_std']:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS: BEST CONFIGS BY LOW STD (for angle)")
    print("=" * 70)

    angle_df = results_df[results_df['target'] == 'angle']
    # Filter to reasonable MSE first
    reasonable = angle_df[angle_df['cv_mse'] < 0.01]
    if len(reasonable) > 0:
        low_std = reasonable.nsmallest(10, 'pred_std')
        print("\nANGLE - Low std configs (with reasonable MSE):")
        for _, row in low_std.iterrows():
            print(f"  {row['config']} (a={row['alpha']}): MSE={row['cv_mse']:.6f}, std={row['pred_std']:.4f}")

    # Save
    results_df.to_csv(OUTPUT_DIR / 'stable_physics_features.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'stable_physics_features.csv'}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    for target_name in TARGETS:
        target_df = results_df[results_df['target'] == target_name]

        if target_name == 'angle':
            # For angle, find config with std < 0.14 and lowest MSE
            low_std = target_df[target_df['pred_std'] < 0.145]
            if len(low_std) > 0:
                best = low_std.loc[low_std['cv_mse'].idxmin()]
            else:
                best = target_df.loc[target_df['cv_mse'].idxmin()]
        else:
            best = target_df.loc[target_df['cv_mse'].idxmin()]

        print(f"\n{target_name.upper()}:")
        print(f"  Config: {best['config']} (alpha={best['alpha']})")
        print(f"  CV MSE: {best['cv_mse']:.6f}")
        print(f"  pred_std: {best['pred_std']:.4f}")


if __name__ == "__main__":
    main()
