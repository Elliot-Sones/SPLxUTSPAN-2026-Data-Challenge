"""
Quick test: GroupKFold CV with advanced biomechanics features.
Tests if physics-informed features help within-player predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots

def main():
    print("=" * 80)
    print("BIOMECH FEATURES - WITHIN-PLAYER CV TEST")
    print("=" * 80)

    # Load the extracted features
    features_path = PROJECT_DIR / "output" / "advanced_biomech_features.csv"

    if not features_path.exists():
        print("Features not found. Run advanced_biomechanics.py first.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Get player IDs
    player_ids = []
    for metadata, _ in iterate_shots(train=True):
        player_ids.append(metadata['participant_id'])

    df['player_id'] = player_ids[:len(df)]

    # Define feature columns (exclude targets and IDs)
    exclude_cols = ['shot_id', 'player_id', 'angle', 'depth', 'left_right']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Using {len(feature_cols)} features")

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)

    groups = df['player_id'].values

    # Scale targets to 0-1
    targets = ['angle', 'depth', 'left_right']
    y_data = {}
    for t in targets:
        vals = df[t].values
        y_data[t] = (vals - vals.min()) / (vals.max() - vals.min())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=5)

    print("\n" + "=" * 80)
    print("GROUPKFOLD CV (WITHIN-PLAYER GENERALIZATION)")
    print("=" * 80)

    results = {}
    for target in targets:
        y = y_data[target]
        mses = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = Ridge(alpha=10.0)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            pred = np.clip(pred, 0, 1)

            mse = mean_squared_error(y_val, pred)
            mses.append(mse)

        results[target] = np.mean(mses)
        print(f"  {target}: CV MSE = {np.mean(mses):.6f} +/- {np.std(mses):.6f}")

    print(f"\n  Total: {sum(results.values())/3:.6f}")

    # Compare with within-player correlation
    print("\n" + "=" * 80)
    print("WITHIN-PLAYER CORRELATIONS (TOP 5)")
    print("=" * 80)

    for target in targets:
        print(f"\n{target.upper()}:")
        correlations = []

        for player in df['player_id'].unique():
            player_df = df[df['player_id'] == player]
            if len(player_df) < 10:
                continue

            for feat in feature_cols:
                r = np.corrcoef(player_df[feat].values, player_df[target].values)[0, 1]
                if not np.isnan(r):
                    correlations.append((feat, player, r))

        # Average correlation per feature
        feat_corrs = {}
        for feat, player, r in correlations:
            if feat not in feat_corrs:
                feat_corrs[feat] = []
            feat_corrs[feat].append(r)

        avg_corrs = [(f, np.mean(rs), np.std(rs)) for f, rs in feat_corrs.items()]
        avg_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, mean_r, std_r in avg_corrs[:5]:
            print(f"  {feat:35s}: r={mean_r:+.4f} (std={std_r:.4f})")


if __name__ == "__main__":
    main()
