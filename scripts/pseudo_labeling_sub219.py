"""
Pseudo-labeling / Semi-supervised Learning with Sub 219.

Strategy:
1. Use submission_219 predictions as pseudo-labels for test data (LB 0.007682)
2. Retrain on combined train + pseudo-labeled test data
3. Generate new predictions

Techniques implemented:
1. Basic pseudo-labeling (uniform weight)
2. Confidence-weighted pseudo-labeling (weight by prediction confidence)
3. Iterative pseudo-labeling (multiple rounds)
4. Consistency regularization (multiple models, consistency score)
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from copy import deepcopy

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SRC_DIR = PROJECT_DIR / "src"
SUBMISSION_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(SRC_DIR))

from hybrid_features import extract_hybrid_features, init_keypoint_mapping
from advanced_features import extract_advanced_features, init_keypoint_mapping as adv_init

TARGETS = ["angle", "depth", "left_right"]
SCALED_TARGETS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_data_with_pseudo_labels_sub219():
    """Load train data and test data with pseudo-labels from Sub 219."""
    print("Loading data with pseudo-labels from Sub 219...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Load Sub 219 for pseudo-labels (LB 0.007682)
    best_sub = pd.read_csv(SUBMISSION_DIR / "submission_219.csv")
    print(f"Using submission_219 as pseudo-labels (LB: 0.007682)")

    # Load target scalers to convert back to original scale
    target_scalers = {}
    for target in TARGETS:
        target_scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    meta_cols = ["id", "shot_id", "participant_id", "angle", "depth", "left_right"]
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    init_keypoint_mapping(keypoint_cols)

    def extract_features(df):
        all_features = []
        ids = []
        pids = []

        for idx, row in df.iterrows():
            timeseries = np.zeros((240, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            # Use hybrid features
            feats = extract_hybrid_features(timeseries, row['participant_id'], smooth=False)
            all_features.append(feats)
            ids.append(row['id'])
            pids.append(row['participant_id'])

        return all_features, ids, pids

    print("Processing training data...")
    train_feats, train_ids, train_pids = extract_features(train_df)

    print("Processing test data...")
    test_feats, test_ids, test_pids = extract_features(test_df)

    feature_names = sorted(train_feats[0].keys())

    X_train = np.array([[f.get(name, 0.0) for name in feature_names] for f in train_feats], dtype=np.float32)
    X_test = np.array([[f.get(name, 0.0) for name in feature_names] for f in test_feats], dtype=np.float32)

    # Replace NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Train targets (actual)
    y_train = np.zeros((len(train_df), 3), dtype=np.float32)
    y_train[:, 0] = train_df['angle'].values
    y_train[:, 1] = train_df['depth'].values
    y_train[:, 2] = train_df['left_right'].values

    # Test pseudo-labels (from Sub 219, convert back to original scale)
    y_test_pseudo = np.zeros((len(test_df), 3), dtype=np.float32)

    # Map test ids to submission
    sub_dict = best_sub.set_index('id').to_dict('index')
    for i, test_id in enumerate(test_ids):
        row = sub_dict[test_id]
        # Convert scaled predictions back to original scale
        y_test_pseudo[i, 0] = target_scalers['angle'].inverse_transform([[row['scaled_angle']]])[0, 0]
        y_test_pseudo[i, 1] = target_scalers['depth'].inverse_transform([[row['scaled_depth']]])[0, 0]
        y_test_pseudo[i, 2] = target_scalers['left_right'].inverse_transform([[row['scaled_left_right']]])[0, 0]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Pseudo-label stats (original scale):")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: mean={y_test_pseudo[:, i].mean():.4f}, std={y_test_pseudo[:, i].std():.4f}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_pids": np.array(train_pids),
        "train_ids": np.array(train_ids),
        "X_test": X_test,
        "y_test_pseudo": y_test_pseudo,
        "test_ids": np.array(test_ids),
        "test_pids": np.array(test_pids),
        "feature_names": feature_names,
        "target_scalers": target_scalers,
    }


def compute_confidence_weights(data, method='std_based'):
    """
    Compute confidence weights for pseudo-labeled samples.

    Methods:
    - std_based: Weight inversely proportional to predicted std within same player
    - distribution_based: Weight based on how close to train distribution
    """
    y_test_pseudo = data["y_test_pseudo"]
    test_pids = data["test_pids"]
    y_train = data["y_train"]
    train_pids = data["train_pids"]

    weights = np.ones(len(y_test_pseudo))

    if method == 'std_based':
        # For each player, compute std of predictions and weight inversely
        for pid in np.unique(test_pids):
            mask = test_pids == pid
            for t_idx in range(3):
                pseudo_vals = y_test_pseudo[mask, t_idx]
                std = np.std(pseudo_vals)
                if std > 0:
                    # Normalize std by train std for this player
                    train_mask = train_pids == pid
                    train_std = np.std(y_train[train_mask, t_idx])
                    if train_std > 0:
                        ratio = std / train_std
                        # Higher ratio = more spread = lower confidence
                        # Weight is 1/(1+ratio)
                        weights[mask] *= 1.0 / (1.0 + ratio)

    elif method == 'distribution_based':
        # Weight based on proximity to train distribution
        for pid in np.unique(test_pids):
            test_mask = test_pids == pid
            train_mask = train_pids == pid

            for t_idx in range(3):
                train_mean = np.mean(y_train[train_mask, t_idx])
                train_std = np.std(y_train[train_mask, t_idx]) + 1e-6

                pseudo_vals = y_test_pseudo[test_mask, t_idx]
                z_scores = np.abs((pseudo_vals - train_mean) / train_std)

                # Gaussian-like weighting: exp(-z^2/2)
                conf = np.exp(-z_scores**2 / 2)
                weights[test_mask] *= conf

    # Normalize weights to [0.1, 1.0]
    weights = 0.1 + 0.9 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)

    return weights


def train_basic_pseudo_labeling(data, pseudo_weight=0.5):
    """Train on combined train + weighted pseudo-labeled test data (per-player)."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    train_pids = data["train_pids"]
    X_test = data["X_test"]
    y_test_pseudo = data["y_test_pseudo"]
    test_pids = data["test_pids"]

    unique_pids = sorted(np.unique(train_pids))

    all_models = {}
    all_scalers = {}

    print("\n" + "="*70)
    print(f"BASIC PSEUDO-LABELING (weight={pseudo_weight})")
    print("="*70)

    for pid in unique_pids:
        # Get train data for this player
        train_mask = train_pids == pid
        X_player_train = X_train[train_mask]
        y_player_train = y_train[train_mask]

        # Get test data for this player (pseudo-labeled)
        test_mask = test_pids == pid
        X_player_test = X_test[test_mask]
        y_player_test = y_test_pseudo[test_mask]

        n_train = len(X_player_train)
        n_test = len(X_player_test)

        print(f"\n--- Player {pid} (train={n_train}, test={n_test}) ---")

        # Combine train + pseudo-labeled test
        X_combined = np.vstack([X_player_train, X_player_test])
        y_combined = np.vstack([y_player_train, y_player_test])

        # Sample weights: train=1.0, pseudo=pseudo_weight
        sample_weights = np.concatenate([
            np.ones(n_train),
            np.full(n_test, pseudo_weight)
        ])

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[pid] = scaler

        for target_idx, target in enumerate(TARGETS):
            y_target = y_combined[:, target_idx]

            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=12,
                learning_rate=0.03,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_samples=5,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            lgb_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'lgb')] = lgb_model

            # CatBoost
            cat_model = CatBoostRegressor(
                iterations=200,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                random_state=42,
                verbose=False
            )
            cat_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'cat')] = cat_model

            # Ridge
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_model

            # Evaluate on original train data only
            X_train_scaled = scaler.transform(X_player_train)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

            lgb_pred = lgb_model.predict(X_train_scaled)
            cat_pred = cat_model.predict(X_train_scaled)
            ridge_pred = ridge_model.predict(X_train_scaled)

            pred = (lgb_pred + cat_pred + ridge_pred) / 3
            mse = np.mean((pred - y_player_train[:, target_idx]) ** 2)
            print(f"  {target} train MSE: {mse:.6f}")

    return {
        "models": all_models,
        "scalers": all_scalers,
    }


def train_confidence_weighted_pseudo_labeling(data, base_weight=0.5, confidence_method='std_based'):
    """Train with confidence-weighted pseudo-labels."""
    X_train = data["X_train"]
    y_train = data["y_train"]
    train_pids = data["train_pids"]
    X_test = data["X_test"]
    y_test_pseudo = data["y_test_pseudo"]
    test_pids = data["test_pids"]

    # Compute confidence weights
    confidence_weights = compute_confidence_weights(data, method=confidence_method)

    unique_pids = sorted(np.unique(train_pids))

    all_models = {}
    all_scalers = {}

    print("\n" + "="*70)
    print(f"CONFIDENCE-WEIGHTED PSEUDO-LABELING (base_weight={base_weight}, method={confidence_method})")
    print("="*70)
    print(f"Confidence weights: min={confidence_weights.min():.3f}, max={confidence_weights.max():.3f}, mean={confidence_weights.mean():.3f}")

    for pid in unique_pids:
        train_mask = train_pids == pid
        X_player_train = X_train[train_mask]
        y_player_train = y_train[train_mask]

        test_mask = test_pids == pid
        X_player_test = X_test[test_mask]
        y_player_test = y_test_pseudo[test_mask]
        player_conf_weights = confidence_weights[test_mask]

        n_train = len(X_player_train)
        n_test = len(X_player_test)

        print(f"\n--- Player {pid} (train={n_train}, test={n_test}) ---")

        X_combined = np.vstack([X_player_train, X_player_test])
        y_combined = np.vstack([y_player_train, y_player_test])

        # Confidence-weighted sample weights
        sample_weights = np.concatenate([
            np.ones(n_train),
            base_weight * player_conf_weights
        ])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[pid] = scaler

        for target_idx, target in enumerate(TARGETS):
            y_target = y_combined[:, target_idx]

            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=12,
                learning_rate=0.03,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_samples=5,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            lgb_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'lgb')] = lgb_model

            cat_model = CatBoostRegressor(
                iterations=200,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                random_state=42,
                verbose=False
            )
            cat_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'cat')] = cat_model

            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_model

            X_train_scaled = scaler.transform(X_player_train)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

            lgb_pred = lgb_model.predict(X_train_scaled)
            cat_pred = cat_model.predict(X_train_scaled)
            ridge_pred = ridge_model.predict(X_train_scaled)

            pred = (lgb_pred + cat_pred + ridge_pred) / 3
            mse = np.mean((pred - y_player_train[:, target_idx]) ** 2)
            print(f"  {target} train MSE: {mse:.6f}")

    return {
        "models": all_models,
        "scalers": all_scalers,
    }


def train_iterative_pseudo_labeling(data, n_iterations=3, initial_weight=0.3, weight_growth=0.1):
    """
    Iterative pseudo-labeling: refine pseudo-labels over multiple rounds.
    Each iteration uses predictions from previous round as new pseudo-labels.
    Weight increases each iteration as pseudo-labels improve.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    train_pids = data["train_pids"]
    X_test = data["X_test"]
    test_pids = data["test_pids"]

    # Start with Sub 219 pseudo-labels
    current_pseudo = data["y_test_pseudo"].copy()

    print("\n" + "="*70)
    print(f"ITERATIVE PSEUDO-LABELING ({n_iterations} iterations)")
    print("="*70)

    final_models = None
    final_scalers = None

    for iteration in range(n_iterations):
        weight = initial_weight + iteration * weight_growth
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations} (weight={weight:.2f})")
        print('='*60)

        unique_pids = sorted(np.unique(train_pids))
        all_models = {}
        all_scalers = {}

        for pid in unique_pids:
            train_mask = train_pids == pid
            X_player_train = X_train[train_mask]
            y_player_train = y_train[train_mask]

            test_mask = test_pids == pid
            X_player_test = X_test[test_mask]
            y_player_test = current_pseudo[test_mask]

            n_train = len(X_player_train)
            n_test = len(X_player_test)

            X_combined = np.vstack([X_player_train, X_player_test])
            y_combined = np.vstack([y_player_train, y_player_test])

            sample_weights = np.concatenate([
                np.ones(n_train),
                np.full(n_test, weight)
            ])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combined)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            all_scalers[pid] = scaler

            for target_idx, target in enumerate(TARGETS):
                y_target = y_combined[:, target_idx]

                lgb_model = lgb.LGBMRegressor(
                    n_estimators=200,
                    num_leaves=12,
                    learning_rate=0.03,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    min_child_samples=5,
                    random_state=42 + iteration,
                    verbose=-1,
                    n_jobs=-1
                )
                lgb_model.fit(X_scaled, y_target, sample_weight=sample_weights)
                all_models[(pid, target, 'lgb')] = lgb_model

                cat_model = CatBoostRegressor(
                    iterations=200,
                    depth=4,
                    learning_rate=0.03,
                    l2_leaf_reg=3.0,
                    random_state=42 + iteration,
                    verbose=False
                )
                cat_model.fit(X_scaled, y_target, sample_weight=sample_weights)
                all_models[(pid, target, 'cat')] = cat_model

                ridge_model = Ridge(alpha=1.0, random_state=42 + iteration)
                ridge_model.fit(X_scaled, y_target)
                all_models[(pid, target, 'ridge')] = ridge_model

        # Update pseudo-labels with new predictions
        if iteration < n_iterations - 1:
            new_pseudo = np.zeros_like(current_pseudo)
            for i, (x, pid) in enumerate(zip(X_test, test_pids)):
                x_scaled = all_scalers[pid].transform(x.reshape(1, -1))
                x_scaled = np.nan_to_num(x_scaled, nan=0.0)

                for target_idx, target in enumerate(TARGETS):
                    lgb_pred = all_models[(pid, target, 'lgb')].predict(x_scaled)[0]
                    cat_pred = all_models[(pid, target, 'cat')].predict(x_scaled)[0]
                    ridge_pred = all_models[(pid, target, 'ridge')].predict(x_scaled)[0]
                    new_pseudo[i, target_idx] = (lgb_pred + cat_pred + ridge_pred) / 3

            # Blend old and new pseudo-labels
            blend_factor = 0.7  # How much to trust new predictions
            current_pseudo = blend_factor * new_pseudo + (1 - blend_factor) * current_pseudo

            print(f"  Updated pseudo-labels:")
            for i, t in enumerate(TARGETS):
                print(f"    {t}: mean={current_pseudo[:, i].mean():.4f}, std={current_pseudo[:, i].std():.4f}")

        final_models = all_models
        final_scalers = all_scalers

    return {
        "models": final_models,
        "scalers": final_scalers,
    }


def train_consistency_regularized(data, pseudo_weight=0.5, consistency_weight=0.1):
    """
    Consistency regularization: train multiple diverse models and use
    agreement between them as additional signal for pseudo-label quality.
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    train_pids = data["train_pids"]
    X_test = data["X_test"]
    y_test_pseudo = data["y_test_pseudo"]
    test_pids = data["test_pids"]

    print("\n" + "="*70)
    print(f"CONSISTENCY REGULARIZATION (pseudo_weight={pseudo_weight}, consistency_weight={consistency_weight})")
    print("="*70)

    unique_pids = sorted(np.unique(train_pids))

    # First pass: train initial diverse models on train data only
    print("\nPhase 1: Training initial diverse models on train data...")
    initial_preds = {}  # Store predictions from diverse models

    for pid in unique_pids:
        train_mask = train_pids == pid
        X_player_train = X_train[train_mask]
        y_player_train = y_train[train_mask]

        test_mask = test_pids == pid
        X_player_test = X_test[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_player_train)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_test_scaled = scaler.transform(X_player_test)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

        for target_idx, target in enumerate(TARGETS):
            y_target = y_player_train[:, target_idx]

            # Train 3 different models with different hyperparams
            preds_list = []

            # Model 1: LightGBM
            m1 = lgb.LGBMRegressor(n_estimators=100, num_leaves=8, learning_rate=0.05, random_state=1, verbose=-1)
            m1.fit(X_train_scaled, y_target)
            preds_list.append(m1.predict(X_test_scaled))

            # Model 2: LightGBM different config
            m2 = lgb.LGBMRegressor(n_estimators=150, num_leaves=15, learning_rate=0.02, random_state=2, verbose=-1)
            m2.fit(X_train_scaled, y_target)
            preds_list.append(m2.predict(X_test_scaled))

            # Model 3: CatBoost
            m3 = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.05, random_state=3, verbose=False)
            m3.fit(X_train_scaled, y_target)
            preds_list.append(m3.predict(X_test_scaled))

            # Model 4: XGBoost
            m4 = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=4, verbosity=0)
            m4.fit(X_train_scaled, y_target)
            preds_list.append(m4.predict(X_test_scaled))

            # Model 5: Ridge
            m5 = Ridge(alpha=1.0)
            m5.fit(X_train_scaled, y_target)
            preds_list.append(m5.predict(X_test_scaled))

            # Store predictions
            initial_preds[(pid, target)] = np.array(preds_list)

    # Compute consistency scores (inverse of std across models)
    consistency_scores = np.ones(len(X_test))
    for i, (pid) in enumerate(test_pids):
        test_mask = test_pids == pid
        test_indices = np.where(test_mask)[0]
        local_idx = np.where(test_indices == i)[0][0] if i in test_indices else 0

        stds = []
        for target in TARGETS:
            preds = initial_preds[(pid, target)][:, local_idx]
            stds.append(np.std(preds))

        avg_std = np.mean(stds)
        # Convert to consistency score: lower std = higher consistency
        consistency_scores[i] = 1.0 / (1.0 + avg_std)

    print(f"Consistency scores: min={consistency_scores.min():.3f}, max={consistency_scores.max():.3f}, mean={consistency_scores.mean():.3f}")

    # Second pass: train final models with consistency-weighted pseudo-labels
    print("\nPhase 2: Training final models with consistency-weighted pseudo-labels...")
    all_models = {}
    all_scalers = {}

    for pid in unique_pids:
        train_mask = train_pids == pid
        X_player_train = X_train[train_mask]
        y_player_train = y_train[train_mask]

        test_mask = test_pids == pid
        X_player_test = X_test[test_mask]
        y_player_test = y_test_pseudo[test_mask]
        player_consistency = consistency_scores[test_mask]

        n_train = len(X_player_train)
        n_test = len(X_player_test)

        print(f"\n--- Player {pid} (train={n_train}, test={n_test}) ---")

        X_combined = np.vstack([X_player_train, X_player_test])
        y_combined = np.vstack([y_player_train, y_player_test])

        # Combined weights: base pseudo_weight scaled by consistency
        sample_weights = np.concatenate([
            np.ones(n_train),
            pseudo_weight * (1.0 + consistency_weight * (player_consistency - player_consistency.mean()))
        ])
        sample_weights = np.clip(sample_weights, 0.1, 2.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        all_scalers[pid] = scaler

        for target_idx, target in enumerate(TARGETS):
            y_target = y_combined[:, target_idx]

            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=12,
                learning_rate=0.03,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_samples=5,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            lgb_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'lgb')] = lgb_model

            cat_model = CatBoostRegressor(
                iterations=200,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                random_state=42,
                verbose=False
            )
            cat_model.fit(X_scaled, y_target, sample_weight=sample_weights)
            all_models[(pid, target, 'cat')] = cat_model

            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X_scaled, y_target)
            all_models[(pid, target, 'ridge')] = ridge_model

            X_train_scaled = scaler.transform(X_player_train)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)

            lgb_pred = lgb_model.predict(X_train_scaled)
            cat_pred = cat_model.predict(X_train_scaled)
            ridge_pred = ridge_model.predict(X_train_scaled)

            pred = (lgb_pred + cat_pred + ridge_pred) / 3
            mse = np.mean((pred - y_player_train[:, target_idx]) ** 2)
            print(f"  {target} train MSE: {mse:.6f}")

    return {
        "models": all_models,
        "scalers": all_scalers,
    }


def predict(data, trained):
    """Generate final predictions."""
    X_test = data["X_test"]
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

            predictions[i, target_idx] = (lgb_pred + cat_pred + ridge_pred) / 3

    return predictions


def create_submission(test_ids, predictions, submission_num, method_name, data):
    """Create submission file."""
    target_scalers = data["target_scalers"]

    scaled_predictions = np.zeros_like(predictions)
    for i, target in enumerate(TARGETS):
        scaled_predictions[:, i] = target_scalers[target].transform(
            predictions[:, i].reshape(-1, 1)
        ).flatten()

    # Clip to [0, 1]
    scaled_predictions = np.clip(scaled_predictions, 0, 1)

    submission = pd.DataFrame({
        'id': test_ids,
        'scaled_angle': scaled_predictions[:, 0],
        'scaled_depth': scaled_predictions[:, 1],
        'scaled_left_right': scaled_predictions[:, 2],
    })

    filename = f"submission_{submission_num}.csv"
    filepath = SUBMISSION_DIR / filename
    submission.to_csv(filepath, index=False)

    print(f"\nSubmission {submission_num} saved: {filepath}")
    print(f"Method: {method_name}")
    print("Distribution stats:")
    for col in ['scaled_angle', 'scaled_depth', 'scaled_left_right']:
        print(f"  {col}: mean={submission[col].mean():.4f}, std={submission[col].std():.4f}")

    return filepath


def main():
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    nums = [int(f.stem.split('_')[1]) for f in existing if f.stem.split('_')[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1

    print("="*70)
    print(f"PSEUDO-LABELING WITH SUB 219 - Starting at submission {next_num}")
    print("="*70)

    data = load_data_with_pseudo_labels_sub219()

    results = []

    # Method 1: Basic pseudo-labeling with different weights
    for weight in [0.3, 0.5, 0.7]:
        print(f"\n{'#'*70}")
        print(f"# Method: Basic Pseudo-Labeling (weight={weight})")
        print('#'*70)

        trained = train_basic_pseudo_labeling(data, pseudo_weight=weight)
        predictions = predict(data, trained)
        method_name = f"Basic pseudo-labeling (weight={weight}) with Sub 219"
        filepath = create_submission(data["test_ids"], predictions, next_num, method_name, data)

        results.append({
            "submission": next_num,
            "method": method_name,
            "filepath": str(filepath)
        })
        next_num += 1

    # Method 2: Confidence-weighted pseudo-labeling
    for conf_method in ['std_based', 'distribution_based']:
        print(f"\n{'#'*70}")
        print(f"# Method: Confidence-Weighted Pseudo-Labeling ({conf_method})")
        print('#'*70)

        trained = train_confidence_weighted_pseudo_labeling(data, base_weight=0.5, confidence_method=conf_method)
        predictions = predict(data, trained)
        method_name = f"Confidence-weighted ({conf_method}) pseudo-labeling with Sub 219"
        filepath = create_submission(data["test_ids"], predictions, next_num, method_name, data)

        results.append({
            "submission": next_num,
            "method": method_name,
            "filepath": str(filepath)
        })
        next_num += 1

    # Method 3: Iterative pseudo-labeling
    print(f"\n{'#'*70}")
    print(f"# Method: Iterative Pseudo-Labeling (3 iterations)")
    print('#'*70)

    trained = train_iterative_pseudo_labeling(data, n_iterations=3, initial_weight=0.3, weight_growth=0.15)
    predictions = predict(data, trained)
    method_name = "Iterative pseudo-labeling (3 rounds) with Sub 219"
    filepath = create_submission(data["test_ids"], predictions, next_num, method_name, data)

    results.append({
        "submission": next_num,
        "method": method_name,
        "filepath": str(filepath)
    })
    next_num += 1

    # Method 4: Consistency regularization
    print(f"\n{'#'*70}")
    print(f"# Method: Consistency Regularization")
    print('#'*70)

    trained = train_consistency_regularized(data, pseudo_weight=0.5, consistency_weight=0.3)
    predictions = predict(data, trained)
    method_name = "Consistency-regularized pseudo-labeling with Sub 219"
    filepath = create_submission(data["test_ids"], predictions, next_num, method_name, data)

    results.append({
        "submission": next_num,
        "method": method_name,
        "filepath": str(filepath)
    })
    next_num += 1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF GENERATED SUBMISSIONS")
    print("="*70)

    for r in results:
        print(f"\nSubmission {r['submission']}:")
        print(f"  Method: {r['method']}")
        print(f"  File: {r['filepath']}")

    print("\n" + "="*70)
    print("All submissions generated successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
