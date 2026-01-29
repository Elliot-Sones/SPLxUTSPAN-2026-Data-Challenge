"""
Signal vs Noise Detection using Adversarial Validation.

Strategy:
1. Train a classifier to distinguish train from test
2. Features with high importance for this classifier have distribution shift (noise)
3. Features with low importance are stable (signal)
4. Build complex models only on signal features
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_all_data():
    """Load all data with features."""
    from advanced_features import init_keypoint_mapping, extract_advanced_features
    from hybrid_features import extract_hybrid_features, init_keypoint_mapping as hybrid_init

    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)
    hybrid_init(keypoint_cols)

    def extract_features(df, is_train=True):
        all_features = []
        ids = []
        pids = []
        targets = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            hybrid_feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            advanced_feats = extract_advanced_features(timeseries, row['participant_id'])
            combined = {**hybrid_feats, **advanced_feats}
            all_features.append(combined)

            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        return all_features, ids, pids, targets if is_train else None

    print(f"Processing training shots...")
    train_feats, train_ids, train_pids, train_targets = extract_features(train_df, True)

    print(f"Processing test shots...")
    test_feats, test_ids, test_pids, _ = extract_features(test_df, False)

    feature_names = sorted(train_feats[0].keys())
    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "X_test": X_test,
        "test_ids": np.array(test_ids),
        "test_pids": np.array(test_pids),
        "feature_names": feature_names,
    }


def adversarial_validation(X_train, X_test, feature_names):
    """
    Train classifier to distinguish train from test.
    Returns feature importance scores - higher = more distribution shift.
    """
    print("\n" + "="*70)
    print("ADVERSARIAL VALIDATION - Detecting Distribution Shift")
    print("="*70)

    # Combine train and test
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    # Handle NaN
    X_all = np.nan_to_num(X_all, nan=0.0)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

    # Cross-validation score
    cv_scores = cross_val_score(clf, X_all, y_all, cv=5, scoring='roc_auc')
    print(f"  Adversarial CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    if cv_scores.mean() > 0.55:
        print("  WARNING: Significant distribution shift detected!")
    else:
        print("  Distribution looks similar between train and test")

    # Fit on all data to get feature importance
    clf.fit(X_all, y_all)
    importances = clf.feature_importances_

    # Sort features by importance (distribution shift)
    sorted_idx = np.argsort(importances)[::-1]

    print("\n  Top 20 features with MOST distribution shift (potential noise):")
    for i in range(min(20, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    print("\n  Top 20 features with LEAST distribution shift (stable signal):")
    for i in range(min(20, len(sorted_idx))):
        idx = sorted_idx[-(i+1)]
        print(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    return importances, sorted_idx


def compute_feature_stability(X_train, y_train, pids, feature_names):
    """
    Compute feature importance stability across CV folds.
    Stable features have consistent importance across folds.
    """
    print("\n" + "="*70)
    print("FEATURE STABILITY - Importance Consistency Across Folds")
    print("="*70)

    X_train = np.nan_to_num(X_train, nan=0.0)

    stability_scores = {}

    for target_idx, target in enumerate(TARGETS):
        y_target = y_train[:, target_idx]

        # Collect importance across folds
        fold_importances = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr = X_train[train_idx]
            y_tr = y_target[train_idx]

            model = lgb.LGBMRegressor(n_estimators=50, num_leaves=10, random_state=42, verbose=-1)
            model.fit(X_tr, y_tr)
            fold_importances.append(model.feature_importances_)

        fold_importances = np.array(fold_importances)

        # Compute stability: mean / std (higher = more stable)
        mean_imp = np.mean(fold_importances, axis=0)
        std_imp = np.std(fold_importances, axis=0) + 1e-6
        stability = mean_imp / std_imp

        stability_scores[target] = stability

        # Show top stable features
        sorted_idx = np.argsort(stability)[::-1]
        print(f"\n  {target.upper()} - Top 10 most stable important features:")
        for i in range(10):
            idx = sorted_idx[i]
            print(f"    {feature_names[idx]}: stability={stability[idx]:.2f}, imp={mean_imp[idx]:.4f}")

    return stability_scores


def select_signal_features(adv_importances, stability_scores, feature_names, n_features=100):
    """
    Select features that are both:
    1. Stable across CV folds (high stability score)
    2. NOT shifted between train/test (low adversarial importance)
    """
    print("\n" + "="*70)
    print(f"SELECTING TOP {n_features} SIGNAL FEATURES")
    print("="*70)

    # Combine scores: high stability, low adversarial importance
    # Normalize scores
    adv_norm = (adv_importances - adv_importances.min()) / (adv_importances.max() - adv_importances.min() + 1e-6)

    # Combined stability across all targets
    combined_stability = np.zeros(len(feature_names))
    for target in TARGETS:
        stab = stability_scores[target]
        stab_norm = (stab - stab.min()) / (stab.max() - stab.min() + 1e-6)
        combined_stability += stab_norm

    # Signal score: high stability - low adversarial
    signal_score = combined_stability - adv_norm * 2  # Penalize shifted features

    # Select top features
    sorted_idx = np.argsort(signal_score)[::-1]
    selected_features = [feature_names[i] for i in sorted_idx[:n_features]]
    selected_mask = np.array([f in selected_features for f in feature_names])

    print(f"  Selected {sum(selected_mask)} signal features")
    print(f"\n  Top 20 selected signal features:")
    for i in range(20):
        idx = sorted_idx[i]
        print(f"    {feature_names[idx]}: signal={signal_score[idx]:.3f}")

    return selected_mask, selected_features


def train_on_signal(data, feature_mask):
    """Train models only on signal features."""
    X_train = data["X_train"][:, feature_mask]
    y_train = data["y_train"]
    pids = data["train_pids"]

    unique_pids = sorted(np.unique(pids))

    all_models = {}
    all_scalers = {}
    oof_preds = np.zeros_like(y_train)

    print("\n" + "="*70)
    print(f"TRAINING ON {X_train.shape[1]} SIGNAL FEATURES")
    print("="*70)

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X_train[pid_mask]
        y_player = y_train[pid_mask]
        n_samples = len(X_player)

        print(f"\n--- Player {pid} ({n_samples} samples) ---")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[pid] = scaler

        player_indices = np.where(pid_mask)[0]

        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_preds = np.zeros(n_samples)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y_target[train_idx], y_target[val_idx]

                # LightGBM - can be more complex since features are cleaner
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=150,
                    num_leaves=15,
                    learning_rate=0.04,
                    reg_alpha=0.3,
                    reg_lambda=0.3,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                )
                lgb_model.fit(X_tr, y_tr)
                lgb_pred = lgb_model.predict(X_val)

                # CatBoost
                cat_model = CatBoostRegressor(
                    iterations=150,
                    depth=5,
                    learning_rate=0.04,
                    l2_leaf_reg=2.0,
                    random_state=42,
                    verbose=False
                )
                cat_model.fit(X_tr, y_tr)
                cat_pred = cat_model.predict(X_val)

                # Ridge
                ridge_model = Ridge(alpha=1.0, random_state=42)
                ridge_model.fit(X_tr, y_tr)
                ridge_pred = ridge_model.predict(X_val)

                fold_preds[val_idx] = 0.4 * lgb_pred + 0.4 * cat_pred + 0.2 * ridge_pred

            oof_preds[player_indices, target_idx] = fold_preds

            # Final models
            lgb_final = lgb.LGBMRegressor(
                n_estimators=150, num_leaves=15, learning_rate=0.04,
                reg_alpha=0.3, reg_lambda=0.3, random_state=42, verbose=-1
            )
            lgb_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'lgb')] = lgb_final

            cat_final = CatBoostRegressor(
                iterations=150, depth=5, learning_rate=0.04,
                l2_leaf_reg=2.0, random_state=42, verbose=False
            )
            cat_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'cat')] = cat_final

            ridge_final = Ridge(alpha=1.0, random_state=42)
            ridge_final.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_final

            mse = np.mean((fold_preds - y_target) ** 2)
            print(f"  {target} CV MSE: {mse:.4f}")

    # Evaluation
    print("\n" + "="*70)
    print("OVERALL CV RESULTS")
    print("="*70)

    angle_scaler = joblib.load(DATA_DIR / 'scaler_angle.pkl')
    depth_scaler = joblib.load(DATA_DIR / 'scaler_depth.pkl')
    lr_scaler = joblib.load(DATA_DIR / 'scaler_left_right.pkl')

    ranges = {
        "angle": angle_scaler.data_range_[0],
        "depth": depth_scaler.data_range_[0],
        "left_right": lr_scaler.data_range_[0],
    }

    total_scaled_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((oof_preds[:, target_idx] - y_train[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_scaled_mse += scaled_mse
        print(f"  {target}: raw MSE = {raw_mse:.4f}, scaled MSE = {scaled_mse:.6f}")

    avg_scaled_mse = total_scaled_mse / 3
    print(f"\n  AVERAGE SCALED MSE: {avg_scaled_mse:.6f}")

    return {
        "models": all_models,
        "scalers": all_scalers,
        "cv_score": avg_scaled_mse,
    }


def predict(data, trained, feature_mask):
    """Generate predictions."""
    X_test = data["X_test"][:, feature_mask]
    test_pids = data["test_pids"]

    models = trained["models"]
    scalers = trained["scalers"]

    predictions = np.zeros((len(X_test), 3))

    for i, (x, pid) in enumerate(zip(X_test, test_pids)):
        x_scaled = scalers[pid].transform(x.reshape(1, -1))
        x_scaled = np.nan_to_num(x_scaled, nan=0.0)

        for target_idx, target in enumerate(TARGETS):
            lgb_pred = models[(pid, target, 'lgb')].predict(x_scaled)[0]
            cat_pred = models[(pid, target, 'cat')].predict(x_scaled)[0]
            ridge_pred = models[(pid, target, 'ridge')].predict(x_scaled)[0]

            predictions[i, target_idx] = 0.4 * lgb_pred + 0.4 * cat_pred + 0.2 * ridge_pred

    return predictions


def create_submission(test_ids, predictions, submission_num, cv_score):
    """Create submission."""
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions[:, 0],
        'scaled_depth': scaled_predictions[:, 1],
        'scaled_left_right': scaled_predictions[:, 2],
    })

    filename = f"submission_{submission_num}.csv"
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\nSubmission saved to: {filepath}")
    print(f"CV Score: {cv_score:.6f}")

    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}")

    return filepath


def main():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    print("="*70)
    print(f"SIGNAL-BASED SUBMISSION {next_num}")
    print("="*70)

    # Load data
    data = load_all_data()

    # Step 1: Adversarial validation
    adv_importances, _ = adversarial_validation(
        data["X_train"], data["X_test"], data["feature_names"]
    )

    # Step 2: Feature stability
    stability_scores = compute_feature_stability(
        data["X_train"], data["y_train"], data["train_pids"], data["feature_names"]
    )

    # Step 3: Select signal features
    feature_mask, selected_features = select_signal_features(
        adv_importances, stability_scores, data["feature_names"], n_features=100
    )

    # Step 4: Train on signal
    trained = train_on_signal(data, feature_mask)

    # Step 5: Predict
    predictions = predict(data, trained, feature_mask)

    # Step 6: Submit
    create_submission(data["test_ids"], predictions, next_num, trained["cv_score"])

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
