"""
Test winner techniques from Kaggle competitions:
1. Adversarial validation - check train/test distribution
2. Pseudo-labeling - expand training with test predictions
3. Multi-seed ensembling - train multiple seeds, average
4. Aggressive feature selection - fewer but robust features
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    return np.array(json.loads(s.replace("nan", "null")), dtype=np.float32)

def extract_simple_features(X_raw):
    """Extract simple statistical features."""
    features = np.concatenate([
        np.nanmean(X_raw, axis=1),
        np.nanstd(X_raw, axis=1),
        np.nanmax(X_raw, axis=1) - np.nanmin(X_raw, axis=1),
        np.nanpercentile(X_raw, 25, axis=1),
        np.nanpercentile(X_raw, 75, axis=1),
    ], axis=1)
    return np.nan_to_num(features, nan=0.0)

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols + TARGETS]

    # Load train
    n_train = len(train_df)
    X_train_raw = np.zeros((n_train, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in train_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_train_raw[idx, :, i] = parse_array_string(row[col])

    y_train = train_df[TARGETS].values.astype(np.float32)
    pids_train = train_df["participant_id"].values

    # Load test
    n_test = len(test_df)
    X_test_raw = np.zeros((n_test, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in test_df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X_test_raw[idx, :, i] = parse_array_string(row[col])

    test_ids = test_df["id"].values

    return X_train_raw, y_train, pids_train, X_test_raw, test_ids


# =============================================================================
# TEST 1: ADVERSARIAL VALIDATION
# =============================================================================
def test_adversarial_validation(X_train, X_test):
    """
    Check if model can distinguish train from test data.
    High AUC = train/test distributions differ (bad for generalization)
    Low AUC (~0.5) = distributions similar (good)
    """
    print("\n" + "="*60)
    print("TEST 1: ADVERSARIAL VALIDATION")
    print("="*60)

    # Create labels: 0 = train, 1 = test
    X_combined = np.vstack([X_train, X_test])
    y_adv = np.array([0] * len(X_train) + [1] * len(X_test))

    # Shuffle
    idx = np.random.permutation(len(X_combined))
    X_combined, y_adv = X_combined[idx], y_adv[idx]

    # Train classifier
    from sklearn.model_selection import cross_val_score
    clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, verbose=-1)

    scores = cross_val_score(clf, X_combined, y_adv, cv=5, scoring='roc_auc')
    mean_auc = scores.mean()

    print(f"  Adversarial AUC: {mean_auc:.4f} (+/- {scores.std():.4f})")

    if mean_auc > 0.7:
        print("  WARNING: Train/test distributions differ significantly!")
        print("  This explains CV-LB gap. Need distribution matching.")
    elif mean_auc > 0.6:
        print("  MODERATE: Some distribution difference detected.")
    else:
        print("  GOOD: Train/test distributions are similar.")

    return mean_auc


# =============================================================================
# TEST 2: PSEUDO-LABELING
# =============================================================================
def test_pseudo_labeling(X_train, y_train, pids_train, X_test, n_iterations=3):
    """
    Iteratively add high-confidence test predictions to training.
    """
    print("\n" + "="*60)
    print("TEST 2: PSEUDO-LABELING")
    print("="*60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline CV
    gkf = GroupKFold(n_splits=5)
    baseline_mses = []

    for t_idx, target in enumerate(TARGETS):
        fold_preds = np.zeros(len(X_train))
        for tr, va in gkf.split(X_train_scaled, y_train[:, t_idx], pids_train):
            m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, verbose=-1)
            m.fit(X_train_scaled[tr], y_train[tr, t_idx])
            fold_preds[va] = m.predict(X_train_scaled[va])
        mse = np.mean((y_train[:, t_idx] - fold_preds)**2)
        baseline_mses.append(mse)

    print(f"  Baseline CV MSE: {np.mean(baseline_mses):.4f}")

    # Pseudo-labeling iterations
    X_aug = X_train_scaled.copy()
    y_aug = y_train.copy()
    pids_aug = pids_train.copy()

    for iteration in range(n_iterations):
        print(f"\n  Iteration {iteration + 1}:")

        # Train on augmented data, predict test
        test_preds = np.zeros((len(X_test), 3))
        test_stds = np.zeros((len(X_test), 3))

        for t_idx in range(3):
            # Train multiple models for uncertainty
            preds_list = []
            for seed in range(5):
                m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, random_state=seed, verbose=-1)
                m.fit(X_aug, y_aug[:, t_idx])
                preds_list.append(m.predict(X_test_scaled))

            test_preds[:, t_idx] = np.mean(preds_list, axis=0)
            test_stds[:, t_idx] = np.std(preds_list, axis=0)

        # Select high-confidence predictions (low std)
        confidence = 1 / (test_stds.mean(axis=1) + 1e-6)
        threshold = np.percentile(confidence, 80)  # Top 20% confidence
        high_conf_mask = confidence >= threshold

        n_added = high_conf_mask.sum()
        print(f"    Adding {n_added} pseudo-labeled samples")

        if n_added == 0:
            break

        # Add to training
        X_aug = np.vstack([X_aug, X_test_scaled[high_conf_mask]])
        y_aug = np.vstack([y_aug, test_preds[high_conf_mask]])
        # Assign pseudo-labeled samples to a new "group" to prevent leakage
        pids_aug = np.concatenate([pids_aug, np.full(n_added, 99)])

    # Final CV with augmented data
    final_mses = []
    for t_idx, target in enumerate(TARGETS):
        fold_preds = np.zeros(len(X_train))
        for tr, va in gkf.split(X_train_scaled, y_train[:, t_idx], pids_train):
            # Train on augmented but validate on original
            m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, verbose=-1)
            m.fit(X_aug, y_aug[:, t_idx])
            fold_preds[va] = m.predict(X_train_scaled[va])
        mse = np.mean((y_train[:, t_idx] - fold_preds)**2)
        final_mses.append(mse)

    print(f"\n  Final CV MSE: {np.mean(final_mses):.4f}")
    improvement = (np.mean(baseline_mses) - np.mean(final_mses)) / np.mean(baseline_mses) * 100
    print(f"  Change: {improvement:+.2f}%")

    return np.mean(final_mses), test_preds


# =============================================================================
# TEST 3: MULTI-SEED ENSEMBLING
# =============================================================================
def test_multi_seed_ensemble(X_train, y_train, pids_train, n_seeds=10):
    """
    Train with multiple random seeds, average predictions.
    """
    print("\n" + "="*60)
    print("TEST 3: MULTI-SEED ENSEMBLING")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    gkf = GroupKFold(n_splits=5)

    # Single seed baseline
    single_mses = []
    for t_idx in range(3):
        fold_preds = np.zeros(len(X_train))
        for tr, va in gkf.split(X_scaled, y_train[:, t_idx], pids_train):
            m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, random_state=42, verbose=-1)
            m.fit(X_scaled[tr], y_train[tr, t_idx])
            fold_preds[va] = m.predict(X_scaled[va])
        single_mses.append(np.mean((y_train[:, t_idx] - fold_preds)**2))

    print(f"  Single seed (42) CV MSE: {np.mean(single_mses):.4f}")

    # Multi-seed ensemble
    multi_mses = []
    for t_idx in range(3):
        fold_preds_ensemble = np.zeros(len(X_train))
        for tr, va in gkf.split(X_scaled, y_train[:, t_idx], pids_train):
            seed_preds = []
            for seed in range(n_seeds):
                m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, random_state=seed, verbose=-1)
                m.fit(X_scaled[tr], y_train[tr, t_idx])
                seed_preds.append(m.predict(X_scaled[va]))
            fold_preds_ensemble[va] = np.mean(seed_preds, axis=0)
        multi_mses.append(np.mean((y_train[:, t_idx] - fold_preds_ensemble)**2))

    print(f"  Multi-seed ({n_seeds} seeds) CV MSE: {np.mean(multi_mses):.4f}")
    improvement = (np.mean(single_mses) - np.mean(multi_mses)) / np.mean(single_mses) * 100
    print(f"  Change: {improvement:+.2f}%")

    return np.mean(multi_mses)


# =============================================================================
# TEST 4: AGGRESSIVE FEATURE SELECTION
# =============================================================================
def test_feature_selection(X_train, y_train, pids_train, k_values=[20, 50, 100, 200]):
    """
    Test with different numbers of top features.
    """
    print("\n" + "="*60)
    print("TEST 4: AGGRESSIVE FEATURE SELECTION")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    gkf = GroupKFold(n_splits=5)

    # Full features baseline
    full_mses = []
    for t_idx in range(3):
        fold_preds = np.zeros(len(X_train))
        for tr, va in gkf.split(X_scaled, y_train[:, t_idx], pids_train):
            m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, verbose=-1)
            m.fit(X_scaled[tr], y_train[tr, t_idx])
            fold_preds[va] = m.predict(X_scaled[va])
        full_mses.append(np.mean((y_train[:, t_idx] - fold_preds)**2))

    print(f"  Full features ({X_scaled.shape[1]}): CV MSE = {np.mean(full_mses):.4f}")

    best_k = X_scaled.shape[1]
    best_mse = np.mean(full_mses)

    for k in k_values:
        if k >= X_scaled.shape[1]:
            continue

        k_mses = []
        for t_idx in range(3):
            fold_preds = np.zeros(len(X_train))
            for tr, va in gkf.split(X_scaled, y_train[:, t_idx], pids_train):
                # Select top k features based on mutual information
                selector = SelectKBest(mutual_info_regression, k=k)
                X_tr_sel = selector.fit_transform(X_scaled[tr], y_train[tr, t_idx])
                X_va_sel = selector.transform(X_scaled[va])

                m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, verbose=-1)
                m.fit(X_tr_sel, y_train[tr, t_idx])
                fold_preds[va] = m.predict(X_va_sel)
            k_mses.append(np.mean((y_train[:, t_idx] - fold_preds)**2))

        mse = np.mean(k_mses)
        change = (np.mean(full_mses) - mse) / np.mean(full_mses) * 100
        print(f"  Top {k} features: CV MSE = {mse:.4f} ({change:+.2f}%)")

        if mse < best_mse:
            best_mse = mse
            best_k = k

    print(f"\n  Best: {best_k} features with MSE = {best_mse:.4f}")
    return best_k, best_mse


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("KAGGLE WINNER TECHNIQUES TEST")
    print("="*60)

    X_train_raw, y_train, pids_train, X_test_raw, test_ids = load_data()

    # Extract features
    print("\nExtracting features...")
    X_train = extract_simple_features(X_train_raw)
    X_test = extract_simple_features(X_test_raw)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}

    # Test 1: Adversarial validation
    results['adversarial_auc'] = test_adversarial_validation(X_train, X_test)

    # Test 2: Pseudo-labeling
    results['pseudo_mse'], pseudo_preds = test_pseudo_labeling(
        X_train, y_train, pids_train, X_test
    )

    # Test 3: Multi-seed ensemble
    results['multi_seed_mse'] = test_multi_seed_ensemble(X_train, y_train, pids_train)

    # Test 4: Feature selection
    results['best_k'], results['feature_sel_mse'] = test_feature_selection(
        X_train, y_train, pids_train
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Adversarial AUC: {results['adversarial_auc']:.4f}")
    print(f"  Pseudo-labeling MSE: {results['pseudo_mse']:.4f}")
    print(f"  Multi-seed MSE: {results['multi_seed_mse']:.4f}")
    print(f"  Feature selection ({results['best_k']}): {results['feature_sel_mse']:.4f}")

    return results


if __name__ == "__main__":
    main()
