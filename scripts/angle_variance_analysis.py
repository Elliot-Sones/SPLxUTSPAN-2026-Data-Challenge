"""
ANGLE VARIANCE ANALYSIS

Player 4 shows physics signal for angle (R²=0.21), others don't.
Let's understand why:
1. What's the variance in angle per player?
2. Is Player 4's mechanics more consistent?
3. Are other players' angles more random?
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


def main():
    print("=" * 80)
    print("ANGLE VARIANCE ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()

    participant_ids = meta['participant_id'].values
    angle_target = y[:, 0]
    depth_target = y[:, 1]
    lr_target = y[:, 2]

    # ==========================================================================
    # Per-player target statistics
    # ==========================================================================
    print("PER-PLAYER TARGET STATISTICS:")
    print("-" * 80)
    print(f"{'Player':<10} {'N':<6} {'Angle Mean':<12} {'Angle Std':<12} {'Angle CV%':<12} {'Depth Std':<12}")
    print("-" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        n = mask.sum()

        angle_mean = angle_target[mask].mean()
        angle_std = angle_target[mask].std()
        angle_cv = (angle_std / angle_mean) * 100  # Coefficient of variation

        depth_std = depth_target[mask].std()

        print(f"{pid:<10} {n:<6} {angle_mean:<12.2f} {angle_std:<12.4f} {angle_cv:<12.2f} {depth_std:<12.4f}")

    # ==========================================================================
    # Feature variance per player
    # ==========================================================================
    print()
    print("PER-PLAYER FEATURE VARIANCE (right_wrist_z at frame 155):")
    print("-" * 60)

    col_idx = keypoint_cols.index('right_wrist_z') if 'right_wrist_z' in keypoint_cols else None

    if col_idx is not None:
        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            X_player = X[mask]

            wrist_z_155 = X_player[:, 155, col_idx]
            print(f"  Player {pid}: wrist_z mean={wrist_z_155.mean():.4f}, std={wrist_z_155.std():.4f}")

    # ==========================================================================
    # Correlation: angle variance vs physics signal
    # ==========================================================================
    print()
    print("HYPOTHESIS: Players with lower angle variance have cleaner physics signal")
    print("-" * 80)

    # Load the deep investigation results
    try:
        results_df = pd.read_csv(OUTPUT_DIR / 'angle_deep_investigation.csv')

        player_stats = []
        for pid in sorted(np.unique(participant_ids)):
            mask = participant_ids == pid
            angle_std = angle_target[mask].std()

            player_results = results_df[results_df['player'] == pid]
            max_r2 = player_results['r2'].max() if len(player_results) > 0 else np.nan

            player_stats.append({
                'player': pid,
                'angle_std': angle_std,
                'max_r2': max_r2
            })

        stats_df = pd.DataFrame(player_stats)
        print(stats_df.to_string(index=False))

        # Correlation
        corr = np.corrcoef(stats_df['angle_std'], stats_df['max_r2'])[0, 1]
        print(f"\nCorrelation between angle_std and max_r2: {corr:.4f}")

        if corr < -0.3:
            print("-> CONFIRMED: Lower variance = stronger physics signal")
        elif corr > 0.3:
            print("-> OPPOSITE: Higher variance = stronger signal (unexpected)")
        else:
            print("-> No clear relationship")

    except Exception as e:
        print(f"Could not load results: {e}")

    # ==========================================================================
    # Test: Is the issue signal-to-noise ratio?
    # ==========================================================================
    print()
    print("SIGNAL-TO-NOISE ANALYSIS:")
    print("-" * 80)

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        # Get wrist z position at frame 155
        if col_idx is not None:
            wrist_z = X_player[:, 155, col_idx]

            # Correlation between wrist position and angle
            valid = ~(np.isnan(wrist_z) | np.isnan(y_player))
            if valid.sum() > 10:
                corr = np.corrcoef(wrist_z[valid], y_player[valid])[0, 1]
                print(f"  Player {pid}: Correlation(wrist_z_155, angle) = {corr:.4f}")

    # ==========================================================================
    # Detailed look at Player 4 vs Player 5
    # ==========================================================================
    print()
    print("DETAILED COMPARISON: Player 4 (has signal) vs Player 5 (no signal)")
    print("-" * 80)

    for pid in [4, 5]:
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        print(f"\nPlayer {pid}:")
        print(f"  N samples: {mask.sum()}")
        print(f"  Angle: mean={y_player.mean():.2f}, std={y_player.std():.4f}, range=[{y_player.min():.2f}, {y_player.max():.2f}]")

        if col_idx is not None:
            # Check feature variance across frames
            feature_stds = []
            for frame in range(100, 200, 10):
                wrist_z = X_player[:, frame, col_idx]
                feature_stds.append(wrist_z.std())

            print(f"  Wrist_z std across frames 100-190: {np.mean(feature_stds):.4f}")

            # Check if wrist position varies with angle
            wrist_155 = X_player[:, 155, col_idx]
            valid = ~(np.isnan(wrist_155) | np.isnan(y_player))
            if valid.sum() > 5:
                corr = np.corrcoef(wrist_155[valid], y_player[valid])[0, 1]
                print(f"  Correlation(wrist_z_155, angle) = {corr:.4f}")

    # ==========================================================================
    # Conclusion
    # ==========================================================================
    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()
    print("The question: Why does Player 4 have physics signal for angle but others don't?")
    print()
    print("Possible explanations:")
    print("1. Player 4's shooting mechanics are more consistent (lower noise)")
    print("2. Other players have more random variation in their angle")
    print("3. The relationship between body position and angle differs by player")
    print("4. Measurement noise affects some players more than others")


if __name__ == "__main__":
    main()
