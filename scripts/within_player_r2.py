"""
WITHIN-PLAYER R² ANALYSIS

The key insight: ANGLE has 72% between-player variance.

For per-player models, we're only trying to explain the WITHIN-player variance.
Let's compute R² relative to within-player variance (which is what matters).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


def nested_cv_test(X_features: np.ndarray, y: np.ndarray) -> dict:
    """Test features with nested CV, returns predictions."""
    if len(y) < 15 or np.isnan(X_features).all():
        return {'r2': np.nan, 'mse': np.nan, 'preds': None}

    valid_mask = ~(np.isnan(X_features).any(axis=1) | np.isnan(y))
    if valid_mask.sum() < 15:
        return {'r2': np.nan, 'mse': np.nan, 'preds': None}

    X_clean = X_features[valid_mask]
    y_clean = y[valid_mask]

    preds = np.zeros(len(y_clean))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X_clean):
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train, y_test = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_train)
        preds[test_idx] = model.predict(X_te_s)

    mse = np.mean((preds - y_clean) ** 2)
    baseline_mse = np.var(y_clean)
    r2 = 1 - mse / baseline_mse

    return {'r2': r2, 'mse': mse, 'baseline_mse': baseline_mse, 'preds': preds}


def main():
    print("=" * 80)
    print("WITHIN-PLAYER R² ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2]
    }

    # Calculate variance decomposition
    print("VARIANCE DECOMPOSITION RECAP:")
    print("-" * 60)

    for name, target in targets.items():
        total_var = target.var()
        player_means = [target[participant_ids == pid].mean() for pid in np.unique(participant_ids)]
        between_var = np.var(player_means)
        within_vars = [target[participant_ids == pid].var() for pid in np.unique(participant_ids)]
        within_var = np.mean(within_vars)

        print(f"{name}: Total={total_var:.2f}, Between={between_var:.2f} ({between_var/total_var*100:.0f}%), Within={within_var:.2f} ({within_var/total_var*100:.0f}%)")

    # ==========================================================================
    # Test best physics features and compute WITHIN-player R²
    # ==========================================================================
    print()
    print("=" * 80)
    print("PHYSICS R² AS FRACTION OF WITHIN-PLAYER VARIANCE")
    print("=" * 80)
    print()
    print("This shows how much of the PREDICTABLE variance physics explains")
    print()

    # Best features from previous analysis
    BEST_FEATURES = {
        'angle': {
            1: {'frame': 135, 'type': 'arm_vector'},
            2: {'frame': 110, 'type': 'wrist_xyz'},
            3: {'frame': 120, 'type': 'both_wrists'},
            4: {'frame': 130, 'type': 'both_wrists'},
            5: {'frame': 150, 'type': 'both_wrists'},
        },
        'depth': {
            1: {'frame': 150, 'type': 'body_extension'},
            2: {'frame': 95, 'type': 'leg_drive'},
            3: {'frame': 120, 'type': 'hip_thrust'},
            4: {'frame': 125, 'type': 'leg_drive'},
            5: {'frame': 150, 'type': 'hip_thrust'},
        },
        'left_right': {
            1: {'frame': 165, 'type': 'hip_rotation'},
            2: {'frame': 155, 'type': 'hip_rotation'},
            3: {'frame': 170, 'type': 'shoulder_alignment'},
            4: {'frame': 155, 'type': 'elbow_alignment'},
            5: {'frame': 172, 'type': 'shoulder_alignment'},
        }
    }

    results = []

    for target_name, target in targets.items():
        print(f"\n{target_name.upper()}:")
        print("-" * 70)

        total_var = target.var()
        player_means = {pid: target[participant_ids == pid].mean() for pid in np.unique(participant_ids)}

        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]
            y_player = target[mask]

            within_var = y_player.var()
            n = mask.sum()

            # Extract features based on type
            config = BEST_FEATURES[target_name][pid]
            frame = config['frame']
            ftype = config['type']

            if ftype == 'arm_vector':
                shoulder = np.column_stack([
                    X_player[:, frame, col_to_idx['right_shoulder_x']],
                    X_player[:, frame, col_to_idx['right_shoulder_y']],
                    X_player[:, frame, col_to_idx['right_shoulder_z']]
                ])
                wrist = np.column_stack([
                    X_player[:, frame, col_to_idx['right_wrist_x']],
                    X_player[:, frame, col_to_idx['right_wrist_y']],
                    X_player[:, frame, col_to_idx['right_wrist_z']]
                ])
                arm = wrist - shoulder
                length = np.sqrt(np.sum(arm**2, axis=1))
                features = np.column_stack([
                    np.arccos(np.clip(arm[:, 2] / (length + 1e-8), -1, 1)),
                    np.arctan2(arm[:, 2], np.sqrt(arm[:, 0]**2 + arm[:, 1]**2)),
                    np.arctan2(arm[:, 1], arm[:, 0]),
                    length
                ])

            elif ftype == 'wrist_xyz':
                features = np.column_stack([
                    X_player[:, frame, col_to_idx['right_wrist_x']],
                    X_player[:, frame, col_to_idx['right_wrist_y']],
                    X_player[:, frame, col_to_idx['right_wrist_z']]
                ])

            elif ftype == 'both_wrists':
                features = np.column_stack([
                    X_player[:, frame, col_to_idx['right_wrist_x']],
                    X_player[:, frame, col_to_idx['right_wrist_y']],
                    X_player[:, frame, col_to_idx['right_wrist_z']],
                    X_player[:, frame, col_to_idx['left_wrist_x']],
                    X_player[:, frame, col_to_idx['left_wrist_y']],
                    X_player[:, frame, col_to_idx['left_wrist_z']]
                ])

            elif ftype == 'body_extension':
                features = (X_player[:, frame, col_to_idx['right_wrist_z']] -
                           X_player[:, frame, col_to_idx['right_hip_z']]).reshape(-1, 1)

            elif ftype == 'leg_drive':
                features = np.column_stack([
                    X_player[:, frame + 10, col_to_idx['right_knee_z']] - X_player[:, frame, col_to_idx['right_knee_z']],
                    X_player[:, frame + 10, col_to_idx['right_ankle_z']] - X_player[:, frame, col_to_idx['right_ankle_z']]
                ])

            elif ftype == 'hip_thrust':
                features = np.column_stack([
                    X_player[:, frame + 10, col_to_idx['right_hip_z']] - X_player[:, frame, col_to_idx['right_hip_z']],
                    X_player[:, frame, col_to_idx['right_hip_z']]
                ])

            elif ftype == 'hip_rotation':
                features = (X_player[:, frame, col_to_idx['right_hip_z']] -
                           X_player[:, frame, col_to_idx['left_hip_z']]).reshape(-1, 1)

            elif ftype == 'shoulder_alignment':
                features = np.column_stack([
                    X_player[:, frame, col_to_idx['right_shoulder_z']],
                    X_player[:, frame, col_to_idx['right_shoulder_z']] - X_player[:, frame, col_to_idx['left_shoulder_z']]
                ])

            elif ftype == 'elbow_alignment':
                features = (X_player[:, frame, col_to_idx['right_elbow_z']] -
                           X_player[:, frame, col_to_idx['right_shoulder_z']]).reshape(-1, 1)

            # Test
            result = nested_cv_test(features, y_player)

            if np.isnan(result['r2']):
                print(f"  Player {pid}: FAILED")
                continue

            # R² is already relative to within-player variance (since we train per-player)
            # But let's also show what fraction of TOTAL variance this explains

            # MSE achieved
            mse = result['mse']

            # Within-player baseline MSE
            within_baseline_mse = within_var

            # Total baseline MSE (if we just predicted overall mean)
            total_baseline_mse = ((y_player - y_player.mean())**2).mean()

            # R² relative to within-player variance
            r2_within = 1 - mse / within_baseline_mse

            # Variance explained as fraction of within-player
            var_explained_within = r2_within * within_var

            # As fraction of total
            pct_of_total = var_explained_within / total_var * 100

            print(f"  Player {pid}: R²={r2_within:.3f} | Explains {r2_within*100:.1f}% of within-player variance")
            print(f"            Within-var={within_var:.2f} | Total-var={total_var:.2f} | Contributes {pct_of_total:.1f}% of total")

            results.append({
                'target': target_name,
                'player': pid,
                'r2_within': r2_within,
                'within_var': within_var,
                'total_var': total_var,
                'pct_of_total': pct_of_total,
                'feature_type': ftype
            })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for target_name in ['angle', 'depth', 'left_right']:
        target_results = results_df[results_df['target'] == target_name]
        if len(target_results) == 0:
            continue

        avg_r2 = target_results['r2_within'].mean()
        total_var = target_results['total_var'].iloc[0]
        avg_within_var = target_results['within_var'].mean()
        within_pct = avg_within_var / total_var * 100

        print(f"\n{target_name.upper()}:")
        print(f"  Within-player variance is {within_pct:.0f}% of total")
        print(f"  Physics explains {avg_r2*100:.1f}% of within-player variance")
        print(f"  Net contribution to total R²: {avg_r2 * within_pct / 100 * 100:.1f}%")

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("For ANGLE:")
    print("  - 72% of variance is player identity (captured by per-player models)")
    print("  - 28% is shot-to-shot variation (what physics should explain)")
    print("  - Physics explains ~10-25% of this 28% = ~3-7% of total variance")
    print("  - This is EXPECTED given the small predictable component")
    print()
    print("For DEPTH and LEFT_RIGHT:")
    print("  - >90% of variance is shot-to-shot (predictable)")
    print("  - Physics explains ~35-68% of this")
    print("  - This is REAL SIGNAL worth using")
    print()
    print("CONCLUSION:")
    print("  - The baseline model already captures player differences")
    print("  - Physics features can ADD signal on top of player-specific baselines")
    print("  - DEPTH features should improve baseline significantly")
    print("  - LEFT_RIGHT features should help")
    print("  - ANGLE features provide marginal improvement")


if __name__ == "__main__":
    main()
