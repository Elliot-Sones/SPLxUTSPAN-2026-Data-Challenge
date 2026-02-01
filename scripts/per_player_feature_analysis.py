"""
Per-Player Feature Analysis

Step 1 of the implementation plan:
- Analyze which features work best for each player
- Compute permutation importance on within-player CV
- Identify top 20 features per player
- Check overlap to determine if player-specific feature sets make sense

Goal: Understand why per-player correlation with Sub 133 varies so much:
- Player 1: 16% (very different)
- Player 2: -5% (negative!)
- Player 3: 82%
- Player 4: 54%
- Player 5: 27%
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

TARGETS = ['angle', 'depth', 'left_right']


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_train_data():
    """Load training data with all features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    print("Loading training data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    print(f"Processing {len(train_df)} training shots...")

    all_features = []
    all_targets = []
    all_pids = []

    for idx, row in train_df.iterrows():
        timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
        for i, col in enumerate(keypoint_cols):
            timeseries[:, i] = parse_array_string(row[col])

        hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)

        all_targets.append([row['angle'], row['depth'], row['left_right']])
        all_pids.append(row['participant_id'])

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    print(f"Features shape: {X.shape}")
    return X, y, pids, feature_names


def compute_permutation_importance_per_player(X, y, pids, feature_names, target_idx, target_name):
    """
    Compute permutation importance for each player separately.

    Returns dict: player_id -> DataFrame of (feature, importance)
    """
    print(f"\n{'='*60}")
    print(f"Permutation Importance for {target_name.upper()}")
    print(f"{'='*60}")

    unique_pids = sorted(np.unique(pids))
    player_importances = {}

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask, target_idx]
        n_samples = len(X_player)

        print(f"\nPlayer {pid} ({n_samples} samples)...")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Train a Ridge model on all player data
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_scaled, y_player)

        # Compute permutation importance with 5-fold CV scoring
        result = permutation_importance(
            model, X_scaled, y_player,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        importance_df = importance_df.sort_values('importance_mean', ascending=False)

        player_importances[pid] = importance_df

        # Show top 10
        print(f"  Top 10 features for Player {pid} ({target_name}):")
        for i, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance_mean']:.6f} (+/- {row['importance_std']:.6f})")

    return player_importances


def analyze_feature_overlap(player_importances, top_n=20):
    """
    Analyze overlap in top features across players.
    """
    print(f"\n{'='*60}")
    print(f"Feature Overlap Analysis (Top {top_n})")
    print(f"{'='*60}")

    # Get top N features per player
    player_top_features = {}
    for pid, df in player_importances.items():
        player_top_features[pid] = set(df.head(top_n)['feature'].tolist())

    # Compute pairwise overlap
    pids = sorted(player_top_features.keys())

    print("\nPairwise Overlap (Jaccard similarity):")
    overlap_matrix = np.zeros((len(pids), len(pids)))

    for i, pid1 in enumerate(pids):
        for j, pid2 in enumerate(pids):
            set1 = player_top_features[pid1]
            set2 = player_top_features[pid2]
            jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
            overlap_matrix[i, j] = jaccard

    # Print overlap matrix
    print("\n       ", end="")
    for pid in pids:
        print(f"  P{pid}  ", end="")
    print()

    for i, pid1 in enumerate(pids):
        print(f"P{pid1}    ", end="")
        for j, pid2 in enumerate(pids):
            print(f" {overlap_matrix[i,j]:.2f}  ", end="")
        print()

    # Find features that appear in all players' top N
    common_features = player_top_features[1]
    for pid in pids[1:]:
        common_features = common_features & player_top_features[pid]

    print(f"\nFeatures in ALL players' top {top_n}: {len(common_features)}")
    if common_features:
        for f in sorted(common_features):
            print(f"  - {f}")

    # Find unique features per player
    print(f"\nUnique features per player (not in any other player's top {top_n}):")
    for pid in pids:
        others = set()
        for other_pid in pids:
            if other_pid != pid:
                others = others | player_top_features[other_pid]
        unique = player_top_features[pid] - others
        print(f"  Player {pid}: {len(unique)} unique features")
        if unique:
            for f in sorted(list(unique)[:5]):
                print(f"    - {f}")

    return overlap_matrix, common_features


def main():
    print("="*80)
    print("PER-PLAYER FEATURE ANALYSIS")
    print("Goal: Find which features work best for each player")
    print("="*80)

    X, y, pids, feature_names = load_train_data()

    results = {}
    all_overlap_matrices = {}
    all_common_features = {}

    for target_idx, target_name in enumerate(TARGETS):
        print(f"\n\n{'#'*80}")
        print(f"# TARGET: {target_name.upper()}")
        print(f"{'#'*80}")

        # Compute per-player permutation importance
        player_importances = compute_permutation_importance_per_player(
            X, y, pids, feature_names, target_idx, target_name
        )

        # Analyze overlap
        overlap_matrix, common_features = analyze_feature_overlap(player_importances, top_n=20)

        results[target_name] = player_importances
        all_overlap_matrices[target_name] = overlap_matrix
        all_common_features[target_name] = common_features

        # Save per-player feature importance
        for pid, df in player_importances.items():
            output_file = OUTPUT_DIR / f"player{pid}_{target_name}_feature_importance.csv"
            df.to_csv(output_file, index=False)
            print(f"\nSaved: {output_file}")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    print("\nOverlap matrices suggest:")
    for target_name, matrix in all_overlap_matrices.items():
        avg_overlap = (matrix.sum() - np.trace(matrix)) / (len(matrix) * (len(matrix) - 1))
        print(f"  {target_name}: Average pairwise Jaccard = {avg_overlap:.3f}")

        if avg_overlap < 0.3:
            print(f"    -> LOW overlap: Per-player specialized features may help")
        elif avg_overlap > 0.5:
            print(f"    -> HIGH overlap: Same features work for most players")
        else:
            print(f"    -> MODERATE overlap: Some specialization may help")

    print("\nConclusion:")
    print("  If overlap is low, different features work for different players,")
    print("  and per-player specialized models with player-specific features")
    print("  could improve predictions for players 1, 2, 5 (low correlation with Sub 133).")

    # Save combined results
    combined_results = []
    for target_name, player_imps in results.items():
        for pid, df in player_imps.items():
            top_df = df.head(20).copy()
            top_df['target'] = target_name
            top_df['player_id'] = pid
            top_df['rank'] = range(1, len(top_df) + 1)
            combined_results.append(top_df)

    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / 'per_player_top_features.csv', index=False)
    print(f"\nSaved combined results: {OUTPUT_DIR / 'per_player_top_features.csv'}")


if __name__ == "__main__":
    main()
