"""
Split training data into per-player per-target structure.

Creates data_split/ folder with:
  player_1/
    angle.csv
    depth.csv
    left_right.csv
  player_2/
    ...
  (5 players x 3 targets = 15 CSV files)

Each CSV contains: id, feature columns, target column
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    cache_file = project_root / "the_rest" / "output" / "feature_cache" / "features_F4_smooth.pkl"
    output_dir = project_root / "data_split"

    # Load cached data
    print("Loading cached features...")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    feature_names = data['feature_names']
    meta = data['meta']

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {len(feature_names)}")

    # Extract participant IDs and shot IDs
    if isinstance(meta, pd.DataFrame):
        ids = meta['id'].values if 'id' in meta.columns else np.arange(len(X))
        participant_ids = meta['participant_id'].values
    else:
        # Assuming meta is array with [id, shot_id, participant_id]
        ids = meta[:, 0]
        participant_ids = meta[:, 2]

    print(f"  Unique participants: {np.unique(participant_ids)}")

    targets = ['angle', 'depth', 'left_right']

    # Create output directories
    output_dir.mkdir(exist_ok=True)
    for pid in range(1, 6):
        (output_dir / f"player_{pid}").mkdir(exist_ok=True)

    # Split and save data
    print("\nSplitting data...")

    summary = []
    for pid in range(1, 6):
        mask = participant_ids == pid
        n_samples = mask.sum()

        X_player = X[mask]
        y_player = y[mask]
        ids_player = ids[mask]

        print(f"\n  Player {pid}: {n_samples} samples")

        for target_idx, target in enumerate(targets):
            # Build DataFrame
            df = pd.DataFrame(X_player, columns=feature_names)
            df.insert(0, 'id', ids_player)
            df['target'] = y_player[:, target_idx]

            # Save
            filepath = output_dir / f"player_{pid}" / f"{target}.csv"
            df.to_csv(filepath, index=False)

            print(f"    {target}: saved {len(df)} rows to {filepath.name}")

            summary.append({
                'player': pid,
                'target': target,
                'samples': len(df),
                'file': str(filepath.relative_to(project_root))
            })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal files created: {len(summary)}")
    print(f"Output directory: {output_dir}")
    print(f"\nColumns per file: {len(feature_names) + 2} (id + {len(feature_names)} features + target)")

    print("\nSample counts:")
    for pid in range(1, 6):
        player_samples = [s for s in summary if s['player'] == pid]
        n = player_samples[0]['samples']
        print(f"  Player {pid}: {n} samples")

    # Verify a few files
    print("\nVerification:")
    for target in targets:
        filepath = output_dir / "player_1" / f"{target}.csv"
        df = pd.read_csv(filepath)
        print(f"  player_1/{target}.csv: {df.shape[0]} rows, {df.shape[1]} cols")
        print(f"    Columns: id, {len(df.columns) - 2} features, target")
        print(f"    Target range: [{df['target'].min():.4f}, {df['target'].max():.4f}]")


if __name__ == "__main__":
    main()
