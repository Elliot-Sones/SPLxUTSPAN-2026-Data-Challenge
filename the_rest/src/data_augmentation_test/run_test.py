"""
Run data augmentation test across all 7 models.

Test Protocol:
1. Load 344 original samples
2. Extract baseline features (mean + last value per column = 414 features)
3. Create augmented dataset (~700 samples with 1 augment per original)
4. For each of 7 models:
   - CV on original (344) -> MSE_original
   - CV on augmented (700) -> MSE_augmented
   - Report difference

Uses GroupKFold by participant_id to prevent leakage.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_all_as_arrays, load_scalers, scale_targets, NUM_FEATURES
from data_augmentation_test.models import get_all_models
from data_augmentation_test.augmentation import augment_dataset


# Suppress warnings
warnings.filterwarnings('ignore')


def extract_baseline_features(X: np.ndarray) -> np.ndarray:
    """
    Extract baseline features: mean + last value for each of 207 columns.

    This is the same feature extraction that achieved 0.010 MSE.

    Args:
        X: Timeseries data, shape (n_samples, 240, 207)

    Returns:
        Feature matrix, shape (n_samples, 414)
    """
    n_samples = X.shape[0]
    n_cols = X.shape[2]  # 207

    # 2 features per column: mean and last value
    features = np.zeros((n_samples, n_cols * 2), dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_cols):
            col_data = X[i, :, j]
            features[i, j * 2] = np.nanmean(col_data)       # mean
            features[i, j * 2 + 1] = col_data[-1]           # last value

    # Handle any NaN values
    features = np.nan_to_num(features, nan=0.0)

    return features


def compute_scaled_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute scaled MSE (average across 3 targets)."""
    # Targets are already scaled in [0, 1]
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def cross_validate_model(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5
) -> float:
    """
    Run GroupKFold CV and return mean MSE.

    Args:
        model_factory: Function that returns a fresh model instance
        X: Feature matrix
        y: Target matrix (scaled)
        groups: Participant IDs for GroupKFold
        n_splits: Number of CV folds

    Returns:
        Mean MSE across folds
    """
    # Determine actual n_splits based on unique groups
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))

    gkf = GroupKFold(n_splits=n_splits)
    fold_mses = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        model = model_factory()
        model.fit(X_train_scaled, y_train)

        # Predict and compute MSE
        y_pred = model.predict(X_val_scaled)
        mse = compute_scaled_mse(y_val, y_pred)
        fold_mses.append(mse)

    return np.mean(fold_mses)


def run_augmentation_test(verbose: bool = True) -> pd.DataFrame:
    """
    Run full augmentation test across all models.

    Returns:
        DataFrame with results for each model
    """
    print("=" * 70)
    print("DATA AUGMENTATION TEST")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Features: baseline (mean + last per column = 414 features)")
    print("  Augmentation: rotation +/- 0.1 deg, noise std = 0.0001 * range")
    print("  Validation: GroupKFold by participant_id")
    print()

    # Load data
    print("Loading data...")
    start = time.time()
    X_raw, y_raw, meta = load_all_as_arrays(train=True)
    print(f"  Loaded {X_raw.shape[0]} samples in {time.time() - start:.1f}s")
    print(f"  Raw shape: {X_raw.shape}")

    participant_ids = meta['participant_id'].values
    print(f"  Participants: {np.unique(participant_ids)}")

    # Scale targets to [0, 1]
    scalers = load_scalers()
    y_scaled = scale_targets(y_raw, scalers)
    print(f"  Target ranges after scaling: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")

    # Extract baseline features from original data
    print("\nExtracting baseline features...")
    start = time.time()
    X_original = extract_baseline_features(X_raw)
    print(f"  Original features shape: {X_original.shape} in {time.time() - start:.1f}s")

    # Create augmented dataset
    print("\nCreating augmented dataset...")
    start = time.time()
    X_aug_raw, y_aug_scaled, pids_aug = augment_dataset(
        X_raw, y_scaled, participant_ids,
        n_augmented_per_sample=1,
        rotation_range=0.1,
        noise_scale=0.0001,
        random_state=42
    )
    print(f"  Augmented raw shape: {X_aug_raw.shape}")

    # Extract features from augmented data
    X_augmented = extract_baseline_features(X_aug_raw)
    print(f"  Augmented features shape: {X_augmented.shape} in {time.time() - start:.1f}s")

    # Get models
    models = get_all_models()
    print(f"\nTesting {len(models)} models...")
    print("-" * 70)

    results = []

    for model_name, model_factory in models.items():
        print(f"\n{model_name}:")

        try:
            # CV on original data
            start = time.time()
            mse_original = cross_validate_model(
                model_factory, X_original, y_scaled, participant_ids
            )
            t_original = time.time() - start

            # CV on augmented data
            start = time.time()
            mse_augmented = cross_validate_model(
                model_factory, X_augmented, y_aug_scaled, pids_aug
            )
            t_augmented = time.time() - start

            # Compute difference
            diff = mse_augmented - mse_original
            pct_change = (diff / mse_original) * 100

            if diff < -0.0001:
                verdict = "HELPS"
            elif diff > 0.0001:
                verdict = "HURTS"
            else:
                verdict = "NEUTRAL"

            print(f"  Original MSE:  {mse_original:.6f} ({t_original:.1f}s)")
            print(f"  Augmented MSE: {mse_augmented:.6f} ({t_augmented:.1f}s)")
            print(f"  Difference:    {diff:+.6f} ({pct_change:+.1f}%) -> {verdict}")

            results.append({
                "Model": model_name,
                "MSE_Original": mse_original,
                "MSE_Augmented": mse_augmented,
                "Difference": diff,
                "Pct_Change": pct_change,
                "Verdict": verdict,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "Model": model_name,
                "MSE_Original": np.nan,
                "MSE_Augmented": np.nan,
                "Difference": np.nan,
                "Pct_Change": np.nan,
                "Verdict": "ERROR",
            })

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    # Overall verdict
    helps_count = (df['Verdict'] == 'HELPS').sum()
    hurts_count = (df['Verdict'] == 'HURTS').sum()
    neutral_count = (df['Verdict'] == 'NEUTRAL').sum()
    error_count = (df['Verdict'] == 'ERROR').sum()

    print("\n" + "-" * 70)
    print(f"HELPS: {helps_count} | HURTS: {hurts_count} | NEUTRAL: {neutral_count} | ERROR: {error_count}")

    if helps_count > hurts_count:
        print("\nOVERALL: Data augmentation appears to HELP more models than it hurts.")
    elif hurts_count > helps_count:
        print("\nOVERALL: Data augmentation appears to HURT more models than it helps.")
    else:
        print("\nOVERALL: Data augmentation has mixed/neutral effects.")

    return df


if __name__ == "__main__":
    results = run_augmentation_test()

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "augmentation_test_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
