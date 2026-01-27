"""
Quick test script to verify the pipeline works on a small subset.

Per scaling principles: validate at small scale before scaling up.
"""

import numpy as np
import time
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_metadata, load_single_shot, iterate_shots,
    get_keypoint_columns, load_scalers, NUM_FRAMES, NUM_FEATURES
)
from feature_engineering import (
    init_keypoint_mapping, extract_all_features, extract_tier1_features
)


def test_data_loader():
    """Test data loading functions."""
    print("=" * 60)
    print("TEST 1: Data Loader")
    print("=" * 60)

    # Test metadata loading
    meta = load_metadata(train=True)
    print(f"[OK] Loaded metadata: {len(meta)} shots")
    print(f"     Participants: {meta['participant_id'].unique().tolist()}")
    print(f"     Columns: {meta.columns.tolist()}")

    # Test keypoint columns
    keypoint_cols = get_keypoint_columns()
    print(f"[OK] Keypoint columns: {len(keypoint_cols)}")
    print(f"     Expected: {NUM_FEATURES} (69 keypoints x 3 axes)")
    assert len(keypoint_cols) == NUM_FEATURES, f"Expected {NUM_FEATURES} columns"

    # Test single shot loading
    print("\nLoading single shot...")
    start = time.time()
    metadata, ts = load_single_shot(0, train=True)
    elapsed = time.time() - start
    print(f"[OK] Loaded shot in {elapsed:.2f}s")
    print(f"     ID: {metadata['id']}")
    print(f"     Participant: {metadata['participant_id']}")
    print(f"     Targets: angle={metadata['angle']}, depth={metadata['depth']}, lr={metadata['left_right']}")
    print(f"     Timeseries shape: {ts.shape}")
    assert ts.shape == (NUM_FRAMES, NUM_FEATURES), f"Expected ({NUM_FRAMES}, {NUM_FEATURES})"

    # Check for NaN
    nan_count = np.isnan(ts).sum()
    print(f"     NaN values: {nan_count}")

    # Test iteration
    print("\nTesting iteration (3 shots)...")
    count = 0
    for meta_dict, ts in iterate_shots(train=True, chunk_size=3):
        count += 1
        if count >= 3:
            break
    print(f"[OK] Iterated {count} shots")

    # Test scalers
    scalers = load_scalers()
    print(f"[OK] Loaded scalers: {list(scalers.keys())}")

    return True


def test_feature_engineering():
    """Test feature extraction."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Engineering")
    print("=" * 60)

    # Initialize
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)
    print("[OK] Initialized keypoint mapping")

    # Load a shot
    metadata, ts = load_single_shot(0, train=True)

    # Test Tier 1 features
    print("\nExtracting Tier 1 features...")
    start = time.time()
    t1_features = extract_tier1_features(ts)
    elapsed = time.time() - start
    print(f"[OK] Tier 1: {len(t1_features)} features in {elapsed:.2f}s")

    # Check for NaN
    nan_count = sum(1 for v in t1_features.values() if np.isnan(v))
    print(f"     NaN features: {nan_count}")

    # Test all features
    print("\nExtracting all features (Tier 1, 2, 3)...")
    start = time.time()
    all_features = extract_all_features(ts, metadata['participant_id'], tiers=[1, 2, 3])
    elapsed = time.time() - start
    print(f"[OK] All tiers: {len(all_features)} features in {elapsed:.2f}s")

    nan_count = sum(1 for v in all_features.values() if np.isnan(v))
    print(f"     NaN features: {nan_count}")

    # Print some sample feature names
    feature_names = sorted(all_features.keys())
    print(f"\nSample features (first 10):")
    for name in feature_names[:10]:
        print(f"     {name}: {all_features[name]:.4f}")

    return True


def test_mini_training():
    """Test training on a tiny subset."""
    print("\n" + "=" * 60)
    print("TEST 3: Mini Training (10 shots)")
    print("=" * 60)

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor

    # Initialize
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    # Extract features for 10 shots
    print("Extracting features for 10 shots...")
    all_features = []
    all_targets = []

    count = 0
    for metadata, ts in iterate_shots(train=True, chunk_size=10):
        features = extract_all_features(ts, metadata['participant_id'], tiers=[1])
        all_features.append(features)
        all_targets.append([metadata['angle'], metadata['depth'], metadata['left_right']])
        count += 1
        if count >= 10:
            break

    # Convert to arrays
    feature_names = sorted(all_features[0].keys())
    X = np.array([
        [f.get(name, 0.0) for name in feature_names]
        for f in all_features
    ], dtype=np.float32)

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    y = np.array(all_targets, dtype=np.float32)

    print(f"[OK] Feature matrix: {X.shape}")
    print(f"[OK] Target matrix: {y.shape}")

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a simple model (sklearn GradientBoosting - works without libomp)
    print("\nTraining sklearn GradientBoostingRegressor (10 trees)...")
    model = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train[:, 0])  # Just angle for quick test

    # Predict
    y_pred = model.predict(X_test)
    mse = np.mean((y_test[:, 0] - y_pred) ** 2)
    print(f"[OK] Training complete. Test MSE (angle): {mse:.4f}")

    return True


def main():
    """Run all tests."""
    print("SPLxUTSPAN 2026 - Quick Pipeline Test")
    print("=" * 60)

    tests = [
        ("Data Loader", test_data_loader),
        ("Feature Engineering", test_feature_engineering),
        ("Mini Training", test_mini_training),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")

    all_passed = all(s == "PASS" for _, s in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
