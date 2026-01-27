"""
Quick Augmentation Test: More Data for Gradient Boosting

Test Configuration:
- Augmentation strength: Moderate (rotation +/- 1 deg, noise 0.001 * range)
- Data amount: 5x (1,720 samples from 344 original)
- Model: LightGBM only (best performer at 0.024 MSE)
- Features: Baseline (mean + last = 414 features)

Baseline to beat: LightGBM with baseline features (no augmentation): 0.024 MSE
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_all_as_arrays, load_scalers, scale_targets
from data_augmentation_test.augmentation import augment_dataset

# Suppress warnings
warnings.filterwarnings('ignore')

# Test configuration
ROTATION_RANGE = 1.0      # +/- 1 degree (moderate, was 0.1)
NOISE_SCALE = 0.001       # 0.001 * range (moderate, was 0.0001)
N_AUGMENTED_PER_SAMPLE = 4  # 5x total (1 original + 4 augmented)
RANDOM_STATE = 42


def extract_baseline_features(X: np.ndarray) -> np.ndarray:
    """
    Extract baseline features: mean + last value for each of 207 columns.

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


def get_lightgbm_model():
    """LightGBM with default parameters."""
    import lightgbm as lgb

    base = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def compute_scaled_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute scaled MSE (average across 3 targets)."""
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
    """
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


def run_quick_augmentation_test():
    """
    Run focused augmentation test: 5x data with moderate augmentation, LightGBM only.
    """
    print("=" * 70)
    print("QUICK AUGMENTATION TEST: More Data for Gradient Boosting")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Augmentation: rotation +/- {ROTATION_RANGE} deg, noise std = {NOISE_SCALE} * range")
    print(f"  Data multiplier: {N_AUGMENTED_PER_SAMPLE + 1}x ({N_AUGMENTED_PER_SAMPLE} augmented per original)")
    print("  Model: LightGBM only")
    print("  Features: baseline (mean + last per column = 414 features)")
    print("  Validation: GroupKFold by participant_id (5 folds)")
    print("  Random seed: 42")
    print()

    # Load data
    print("Loading data...")
    start = time.time()
    X_raw, y_raw, meta = load_all_as_arrays(train=True)
    n_original = X_raw.shape[0]
    print(f"  Loaded {n_original} samples in {time.time() - start:.1f}s")

    participant_ids = meta['participant_id'].values
    print(f"  Participants: {np.unique(participant_ids)}")

    # Scale targets to [0, 1]
    scalers = load_scalers()
    y_scaled = scale_targets(y_raw, scalers)
    print(f"  Target ranges after scaling: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")

    # Extract baseline features from original data
    print("\nExtracting baseline features (original data)...")
    start = time.time()
    X_original = extract_baseline_features(X_raw)
    print(f"  Shape: {X_original.shape} in {time.time() - start:.1f}s")

    # Create augmented dataset with MODERATE settings
    print(f"\nCreating augmented dataset (5x with moderate augmentation)...")
    start = time.time()
    X_aug_raw, y_aug_scaled, pids_aug = augment_dataset(
        X_raw, y_scaled, participant_ids,
        n_augmented_per_sample=N_AUGMENTED_PER_SAMPLE,
        rotation_range=ROTATION_RANGE,
        noise_scale=NOISE_SCALE,
        random_state=RANDOM_STATE
    )
    n_augmented = X_aug_raw.shape[0]
    print(f"  Augmented raw shape: {X_aug_raw.shape}")

    # Extract features from augmented data
    X_augmented = extract_baseline_features(X_aug_raw)
    print(f"  Augmented features shape: {X_augmented.shape} in {time.time() - start:.1f}s")

    # Run LightGBM on both datasets
    print("\n" + "-" * 70)
    print("LightGBM Results:")
    print("-" * 70)

    # CV on original data
    print(f"\n  Testing on original ({n_original} samples)...")
    start = time.time()
    mse_original = cross_validate_model(
        get_lightgbm_model, X_original, y_scaled, participant_ids
    )
    t_original = time.time() - start
    print(f"    MSE: {mse_original:.6f} ({t_original:.1f}s)")

    # CV on augmented data
    print(f"\n  Testing on augmented ({n_augmented} samples)...")
    start = time.time()
    mse_augmented = cross_validate_model(
        get_lightgbm_model, X_augmented, y_aug_scaled, pids_aug
    )
    t_augmented = time.time() - start
    print(f"    MSE: {mse_augmented:.6f} ({t_augmented:.1f}s)")

    # Compute difference
    diff = mse_augmented - mse_original
    pct_change = (diff / mse_original) * 100

    if diff < -0.0005:
        verdict = "HELPS"
    elif diff > 0.0005:
        verdict = "HURTS"
    else:
        verdict = "NEUTRAL"

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Original ({n_original} samples):   MSE = {mse_original:.6f}")
    print(f"  Augmented ({n_augmented} samples): MSE = {mse_augmented:.6f}")
    print(f"  Difference: {diff:+.6f} ({pct_change:+.1f}%)")
    print(f"\n  Verdict: {verdict}")

    if verdict == "HELPS":
        print("\n  -> More data HELPS LightGBM performance")
    elif verdict == "HURTS":
        print("\n  -> More data HURTS LightGBM performance (confirms prior result)")
    else:
        print("\n  -> More data has NEUTRAL effect on LightGBM")

    print()

    # Return results dict for logging
    return {
        "n_original": n_original,
        "n_augmented": n_augmented,
        "rotation_range": ROTATION_RANGE,
        "noise_scale": NOISE_SCALE,
        "mse_original": mse_original,
        "mse_augmented": mse_augmented,
        "diff": diff,
        "pct_change": pct_change,
        "verdict": verdict,
    }


if __name__ == "__main__":
    results = run_quick_augmentation_test()
