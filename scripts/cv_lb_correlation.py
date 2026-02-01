"""
CV-LB Correlation Analysis

Goal: Find which CV strategy best predicts LB performance.

For each model config that produced a known LB submission, compute multiple CV metrics
and correlate with actual LB scores.

CV Metrics:
1. Within-player 5-fold CV - train/test within same player
2. LOPO CV - Leave one player out completely
3. Bootstrap CV - Multiple random splits
4. Pessimistic CV - Worst-fold instead of mean

Known LB submissions:
- Sub 8: 0.010220 - LightGBM baseline per-player
- Sub 9: 0.009109 - Ensemble with meta-stacking
- Sub 10: 0.008907 - Optuna-tuned ensemble
- Sub 11: 0.009848 - Ultra-optimized (overfit)
- Sub 20: 0.008619 - 80% sub9 + 20% sub10
- Sub 25: 0.008305 - 50% sub9 + 50% sub10 (BEST)
- Sub 34: 0.008377 - 30% sub9 + 70% sub10
- Sub 51: 0.008807 - Hybrid blend
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGETS = ["angle", "depth", "left_right"]

# Known LB scores
KNOWN_LB = {
    8: 0.010220,
    9: 0.009109,
    10: 0.008907,
    11: 0.009848,
    20: 0.008619,
    25: 0.008305,
    34: 0.008377,
    51: 0.008807,
}

# Load target scalers
TARGET_SCALERS = {
    "angle": joblib.load(DATA_DIR / "scaler_angle.pkl"),
    "depth": joblib.load(DATA_DIR / "scaler_depth.pkl"),
    "left_right": joblib.load(DATA_DIR / "scaler_left_right.pkl"),
}


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def get_target_ranges():
    """Get target ranges for scaling."""
    return {
        "angle": TARGET_SCALERS["angle"].data_range_[0],
        "depth": TARGET_SCALERS["depth"].data_range_[0],
        "left_right": TARGET_SCALERS["left_right"].data_range_[0],
    }


def load_train_data():
    """Load training data with advanced features (same as sub9/sub10)."""
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

        hybrid_feats = extract_hybrid_features(timeseries, row["participant_id"], smooth=False)
        advanced_feats = extract_advanced_features(timeseries, row["participant_id"])
        combined = {**hybrid_feats, **advanced_feats}
        all_features.append(combined)

        all_targets.append([row["angle"], row["depth"], row["left_right"]])
        all_pids.append(row["participant_id"])

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(train_df)}")

    feature_names = sorted(all_features[0].keys())
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    pids = np.array(all_pids)

    print(f"Features shape: {X.shape}")
    return X, y, pids, feature_names


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

def get_sub8_config():
    """Sub 8: LightGBM baseline per-player."""
    return {
        "name": "sub8_lgb_baseline",
        "model_class": lgb.LGBMRegressor,
        "params": {
            "n_estimators": 100,
            "num_leaves": 10,
            "learning_rate": 0.05,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        },
        "ensemble": False,
    }


def get_sub9_ensemble_configs():
    """Sub 9: Ensemble with meta-stacking."""
    return {
        "lgb": {
            "class": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 100,
                "num_leaves": 10,
                "learning_rate": 0.05,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
        },
        "lgb_deep": {
            "class": lgb.LGBMRegressor,
            "params": {
                "n_estimators": 200,
                "num_leaves": 31,
                "learning_rate": 0.03,
                "max_depth": 8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "min_child_samples": 10,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
        },
        "xgb": {
            "class": xgb.XGBRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.05,
                "reg_alpha": 0.5,
                "reg_lambda": 0.5,
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
            }
        },
        "ridge": {
            "class": Ridge,
            "params": {"alpha": 1.0, "random_state": 42}
        },
        "catboost": {
            "class": CatBoostRegressor,
            "params": {
                "iterations": 100,
                "depth": 4,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "random_state": 42,
                "verbose": False,
            }
        },
    }


# =============================================================================
# CV STRATEGIES
# =============================================================================

def compute_within_player_cv(X, y, pids, model_config, n_splits=5):
    """
    Within-player 5-fold CV: Train and test within same player.
    This is the standard approach used in most submissions.

    Returns: dict with mean, std, worst fold scores (scaled MSE)
    """
    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(pids))

    all_preds = np.zeros_like(y)
    fold_scores = []

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        player_indices = np.where(pid_mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_player[train_idx], y_player[val_idx]

            fold_preds = np.zeros((len(val_idx), 3))

            for target_idx, target in enumerate(TARGETS):
                model = model_config["model_class"](**model_config["params"])
                model.fit(X_train, y_train[:, target_idx])
                fold_preds[:, target_idx] = model.predict(X_val)

            # Store predictions
            global_val_idx = player_indices[val_idx]
            all_preds[global_val_idx] = fold_preds

            # Compute fold score
            fold_mse = 0
            for target_idx, target in enumerate(TARGETS):
                raw_mse = np.mean((fold_preds[:, target_idx] - y_val[:, target_idx]) ** 2)
                scaled_mse = raw_mse / (ranges[target] ** 2)
                fold_mse += scaled_mse
            fold_scores.append(fold_mse / 3)

    # Overall score
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((all_preds[:, target_idx] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_mse += scaled_mse

    return {
        "mean": total_mse / 3,
        "std": np.std(fold_scores),
        "worst": np.max(fold_scores),
        "best": np.min(fold_scores),
    }


def compute_lopo_cv(X, y, pids, model_config):
    """
    Leave-One-Player-Out CV: Train on 4 players, test on 1 held-out player.
    Tests generalization to completely unseen players.

    Returns: dict with mean, std, worst fold scores (scaled MSE)
    """
    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(pids))

    all_preds = np.zeros_like(y)
    fold_scores = []

    logo = LeaveOneGroupOut()

    for train_idx, val_idx in logo.split(X, y, pids):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

        fold_preds = np.zeros((len(val_idx), 3))

        for target_idx, target in enumerate(TARGETS):
            model = model_config["model_class"](**model_config["params"])
            model.fit(X_train_scaled, y_train[:, target_idx])
            fold_preds[:, target_idx] = model.predict(X_val_scaled)

        all_preds[val_idx] = fold_preds

        # Compute fold score
        fold_mse = 0
        for target_idx, target in enumerate(TARGETS):
            raw_mse = np.mean((fold_preds[:, target_idx] - y_val[:, target_idx]) ** 2)
            scaled_mse = raw_mse / (ranges[target] ** 2)
            fold_mse += scaled_mse
        fold_scores.append(fold_mse / 3)

    # Overall score
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((all_preds[:, target_idx] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_mse += scaled_mse

    return {
        "mean": total_mse / 3,
        "std": np.std(fold_scores),
        "worst": np.max(fold_scores),
        "best": np.min(fold_scores),
    }


def compute_bootstrap_cv(X, y, pids, model_config, n_bootstrap=10):
    """
    Bootstrap CV: Multiple random train/test splits.

    Returns: dict with mean, std, worst scores (scaled MSE)
    """
    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(pids))

    bootstrap_scores = []

    for seed in range(n_bootstrap):
        np.random.seed(seed)

        # Random 80/20 split within each player
        all_preds = np.zeros_like(y)
        valid_mask = np.zeros(len(y), dtype=bool)

        for pid in unique_pids:
            pid_mask = pids == pid
            X_player = X[pid_mask]
            y_player = y[pid_mask]
            player_indices = np.where(pid_mask)[0]

            n_samples = len(X_player)
            n_train = int(0.8 * n_samples)

            indices = np.random.permutation(n_samples)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_player[train_idx])
            X_val = scaler.transform(X_player[val_idx])
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)

            y_train = y_player[train_idx]
            y_val = y_player[val_idx]

            fold_preds = np.zeros((len(val_idx), 3))

            for target_idx, target in enumerate(TARGETS):
                model = model_config["model_class"](**model_config["params"])
                model.fit(X_train, y_train[:, target_idx])
                fold_preds[:, target_idx] = model.predict(X_val)

            global_val_idx = player_indices[val_idx]
            all_preds[global_val_idx] = fold_preds
            valid_mask[global_val_idx] = True

        # Compute score for this bootstrap
        total_mse = 0
        for target_idx, target in enumerate(TARGETS):
            raw_mse = np.mean((all_preds[valid_mask, target_idx] - y[valid_mask, target_idx]) ** 2)
            scaled_mse = raw_mse / (ranges[target] ** 2)
            total_mse += scaled_mse

        bootstrap_scores.append(total_mse / 3)

    return {
        "mean": np.mean(bootstrap_scores),
        "std": np.std(bootstrap_scores),
        "worst": np.max(bootstrap_scores),
        "best": np.min(bootstrap_scores),
    }


def compute_ensemble_within_player_cv(X, y, pids, ensemble_configs, n_splits=5):
    """
    Within-player CV for ensemble model (like sub9).
    Trains multiple base models and uses Ridge meta-stacking.
    """
    ranges = get_target_ranges()
    unique_pids = sorted(np.unique(pids))
    n_models = len(ensemble_configs)

    # OOF predictions for each base model
    oof_preds = {target: np.zeros((len(X), n_models)) for target in TARGETS}

    for pid in unique_pids:
        pid_mask = pids == pid
        X_player = X[pid_mask]
        y_player = y[pid_mask]
        player_indices = np.where(pid_mask)[0]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_player)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for target_idx, target in enumerate(TARGETS):
            y_target = y_player[:, target_idx]

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train = y_target[train_idx]

                for model_idx, (model_name, config) in enumerate(ensemble_configs.items()):
                    model = config["class"](**config["params"])
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    global_val_idx = player_indices[val_idx]
                    oof_preds[target][global_val_idx, model_idx] = pred

    # Meta-stacking with Ridge
    final_preds = {}
    for target_idx, target in enumerate(TARGETS):
        X_meta = oof_preds[target]
        y_target = y[:, target_idx]

        meta_preds = np.zeros(len(y_target))
        kf_meta = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf_meta.split(X_meta):
            X_train, X_val = X_meta[train_idx], X_meta[val_idx]
            y_train = y_target[train_idx]

            meta_model = Ridge(alpha=1.0, random_state=42)
            meta_model.fit(X_train, y_train)
            meta_preds[val_idx] = meta_model.predict(X_val)

        final_preds[target] = meta_preds

    # Compute score
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((final_preds[target] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_mse += scaled_mse

    return {"mean": total_mse / 3}


def compute_ensemble_lopo_cv(X, y, pids, ensemble_configs):
    """
    LOPO CV for ensemble model.
    """
    ranges = get_target_ranges()
    n_models = len(ensemble_configs)

    all_preds = {target: np.zeros(len(y)) for target in TARGETS}
    fold_scores = []

    logo = LeaveOneGroupOut()

    for train_idx, val_idx in logo.split(X, y, pids):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        pids_train = pids[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

        fold_mse = 0

        for target_idx, target in enumerate(TARGETS):
            y_target_train = y_train[:, target_idx]
            y_target_val = y_val[:, target_idx]

            # Train base models and get OOF predictions for meta-model training
            oof_train = np.zeros((len(X_train), n_models))

            # Inner CV for OOF predictions
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for fold_idx, (inner_train, inner_val) in enumerate(kf.split(X_train_scaled)):
                for model_idx, (model_name, config) in enumerate(ensemble_configs.items()):
                    model = config["class"](**config["params"])
                    model.fit(X_train_scaled[inner_train], y_target_train[inner_train])
                    oof_train[inner_val, model_idx] = model.predict(X_train_scaled[inner_val])

            # Train meta-model on OOF predictions
            meta_model = Ridge(alpha=1.0, random_state=42)
            meta_model.fit(oof_train, y_target_train)

            # Train final base models on all training data
            val_base_preds = np.zeros((len(X_val), n_models))
            for model_idx, (model_name, config) in enumerate(ensemble_configs.items()):
                model = config["class"](**config["params"])
                model.fit(X_train_scaled, y_target_train)
                val_base_preds[:, model_idx] = model.predict(X_val_scaled)

            # Make final predictions
            final_pred = meta_model.predict(val_base_preds)
            all_preds[target][val_idx] = final_pred

            raw_mse = np.mean((final_pred - y_target_val) ** 2)
            scaled_mse = raw_mse / (ranges[target] ** 2)
            fold_mse += scaled_mse

        fold_scores.append(fold_mse / 3)

    # Overall score
    total_mse = 0
    for target_idx, target in enumerate(TARGETS):
        raw_mse = np.mean((all_preds[target] - y[:, target_idx]) ** 2)
        scaled_mse = raw_mse / (ranges[target] ** 2)
        total_mse += scaled_mse

    return {
        "mean": total_mse / 3,
        "std": np.std(fold_scores),
        "worst": np.max(fold_scores),
    }


def main():
    print("=" * 80)
    print("CV-LB CORRELATION ANALYSIS")
    print("=" * 80)

    # Load data
    X, y, pids, feature_names = load_train_data()

    results = []

    # =============================================================================
    # Test Sub8 config (simple LightGBM baseline)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TESTING SUB8 CONFIG (LightGBM baseline)")
    print("=" * 80)

    sub8_config = get_sub8_config()

    print("\n  Within-player 5-fold CV...")
    wp_cv = compute_within_player_cv(X, y, pids, sub8_config)
    print(f"    Mean: {wp_cv['mean']:.6f}, Std: {wp_cv['std']:.6f}, Worst: {wp_cv['worst']:.6f}")

    print("\n  LOPO CV...")
    lopo_cv = compute_lopo_cv(X, y, pids, sub8_config)
    print(f"    Mean: {lopo_cv['mean']:.6f}, Std: {lopo_cv['std']:.6f}, Worst: {lopo_cv['worst']:.6f}")

    print("\n  Bootstrap CV (10 iterations)...")
    boot_cv = compute_bootstrap_cv(X, y, pids, sub8_config, n_bootstrap=10)
    print(f"    Mean: {boot_cv['mean']:.6f}, Std: {boot_cv['std']:.6f}, Worst: {boot_cv['worst']:.6f}")

    results.append({
        "submission": 8,
        "lb_score": KNOWN_LB[8],
        "model": "lgb_baseline",
        "within_player_mean": wp_cv["mean"],
        "within_player_std": wp_cv["std"],
        "within_player_worst": wp_cv["worst"],
        "lopo_mean": lopo_cv["mean"],
        "lopo_std": lopo_cv["std"],
        "lopo_worst": lopo_cv["worst"],
        "bootstrap_mean": boot_cv["mean"],
        "bootstrap_std": boot_cv["std"],
    })

    # =============================================================================
    # Test Sub9 config (Ensemble with meta-stacking)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TESTING SUB9 CONFIG (Ensemble with meta-stacking)")
    print("=" * 80)

    ensemble_configs = get_sub9_ensemble_configs()

    print("\n  Within-player 5-fold CV with ensemble...")
    wp_ens = compute_ensemble_within_player_cv(X, y, pids, ensemble_configs)
    print(f"    Mean: {wp_ens['mean']:.6f}")

    print("\n  LOPO CV with ensemble...")
    lopo_ens = compute_ensemble_lopo_cv(X, y, pids, ensemble_configs)
    print(f"    Mean: {lopo_ens['mean']:.6f}, Std: {lopo_ens['std']:.6f}, Worst: {lopo_ens['worst']:.6f}")

    results.append({
        "submission": 9,
        "lb_score": KNOWN_LB[9],
        "model": "ensemble_meta",
        "within_player_mean": wp_ens["mean"],
        "within_player_std": 0,  # Not computed for ensemble
        "within_player_worst": 0,
        "lopo_mean": lopo_ens["mean"],
        "lopo_std": lopo_ens["std"],
        "lopo_worst": lopo_ens["worst"],
        "bootstrap_mean": 0,  # Not computed
        "bootstrap_std": 0,
    })

    # =============================================================================
    # Test variations of LightGBM with different regularization (to simulate other submissions)
    # =============================================================================
    print("\n" + "=" * 80)
    print("TESTING VARIATIONS WITH DIFFERENT REGULARIZATION")
    print("=" * 80)

    # More aggressive (like sub10/11 style)
    aggressive_config = {
        "name": "lgb_aggressive",
        "model_class": lgb.LGBMRegressor,
        "params": {
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.03,
            "max_depth": 8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        },
    }

    print("\n  Testing aggressive config (like sub10/11)...")
    wp_agg = compute_within_player_cv(X, y, pids, aggressive_config)
    lopo_agg = compute_lopo_cv(X, y, pids, aggressive_config)
    print(f"    Within-player: {wp_agg['mean']:.6f}")
    print(f"    LOPO: {lopo_agg['mean']:.6f}")

    results.append({
        "submission": "10_approx",
        "lb_score": KNOWN_LB[10],
        "model": "lgb_aggressive",
        "within_player_mean": wp_agg["mean"],
        "within_player_std": wp_agg["std"],
        "within_player_worst": wp_agg["worst"],
        "lopo_mean": lopo_agg["mean"],
        "lopo_std": lopo_agg["std"],
        "lopo_worst": lopo_agg["worst"],
        "bootstrap_mean": 0,
        "bootstrap_std": 0,
    })

    # More conservative (higher regularization)
    conservative_config = {
        "name": "lgb_conservative",
        "model_class": lgb.LGBMRegressor,
        "params": {
            "n_estimators": 50,
            "num_leaves": 5,
            "learning_rate": 0.1,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        },
    }

    print("\n  Testing conservative config (high regularization)...")
    wp_cons = compute_within_player_cv(X, y, pids, conservative_config)
    lopo_cons = compute_lopo_cv(X, y, pids, conservative_config)
    print(f"    Within-player: {wp_cons['mean']:.6f}")
    print(f"    LOPO: {lopo_cons['mean']:.6f}")

    results.append({
        "submission": "conservative",
        "lb_score": None,
        "model": "lgb_conservative",
        "within_player_mean": wp_cons["mean"],
        "within_player_std": wp_cons["std"],
        "within_player_worst": wp_cons["worst"],
        "lopo_mean": lopo_cons["mean"],
        "lopo_std": lopo_cons["std"],
        "lopo_worst": lopo_cons["worst"],
        "bootstrap_mean": 0,
        "bootstrap_std": 0,
    })

    # =============================================================================
    # Correlation Analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    # Filter to only rows with known LB scores
    df = pd.DataFrame(results)
    known_df = df[df["lb_score"].notna()].copy()

    cv_metrics = [
        "within_player_mean",
        "within_player_worst",
        "lopo_mean",
        "lopo_worst",
    ]

    print("\nPearson correlations (CV metric vs LB score):")
    print("-" * 60)

    correlations = []
    for metric in cv_metrics:
        if known_df[metric].sum() > 0:  # Only if we have data
            r, p = pearsonr(known_df[metric], known_df["lb_score"])
            print(f"  {metric:25s}: r = {r:+.4f}, p = {p:.4f}")
            correlations.append({"metric": metric, "pearson_r": r, "p_value": p})

    print("\nSpearman correlations (rank-based):")
    print("-" * 60)
    for metric in cv_metrics:
        if known_df[metric].sum() > 0:
            rho, p = spearmanr(known_df[metric], known_df["lb_score"])
            print(f"  {metric:25s}: rho = {rho:+.4f}, p = {p:.4f}")

    # =============================================================================
    # Save Results
    # =============================================================================
    output_file = OUTPUT_DIR / "cv_lb_correlation.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    corr_file = OUTPUT_DIR / "cv_lb_correlations.csv"
    pd.DataFrame(correlations).to_csv(corr_file, index=False)
    print(f"Correlations saved to: {corr_file}")

    # =============================================================================
    # Key Findings
    # =============================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("\n1. CV-LB Gap Analysis:")
    for _, row in known_df.iterrows():
        sub = row["submission"]
        lb = row["lb_score"]
        wp = row["within_player_mean"]
        lopo = row["lopo_mean"]
        print(f"   Sub {sub}: LB={lb:.6f}, Within-player={wp:.6f}, LOPO={lopo:.6f}")
        print(f"            Gap (WP-LB): {wp - lb:+.6f}, Gap (LOPO-LB): {lopo - lb:+.6f}")

    print("\n2. Which CV metric best predicts LB?")
    best_metric = max(correlations, key=lambda x: abs(x["pearson_r"]))
    print(f"   Best: {best_metric['metric']} with r = {best_metric['pearson_r']:.4f}")

    print("\n3. Recommendations:")
    print("   - Use the CV metric with highest LB correlation for model selection")
    print("   - If LOPO correlates better, focus on cross-player generalization")
    print("   - If within-player correlates better, focus on within-player patterns")


if __name__ == "__main__":
    main()
