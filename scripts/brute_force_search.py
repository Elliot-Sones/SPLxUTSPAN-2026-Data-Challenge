"""
Brute Force Feature-Algorithm Search System.

Tests random feature subsets paired with multiple algorithms,
using Optuna for hyperparameter tuning. Designed for long-running
CPU-based search (10,000+ trials over multiple days).

Features:
- Random feature subset sampling (from 132 available features)
- 4 model types: LightGBM, XGBoost, CatBoost, Ridge
- Optuna hyperparameter tuning per (feature_subset, model) combination
- Incremental CSV logging with resumption support
- Progress logging for remote monitoring
- File locking for parallel instance support
"""

import argparse
import fcntl
import hashlib
import json
import logging
import numpy as np
import optuna
import pandas as pd
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

TOTAL_TRIALS = 10000
OPTUNA_TRIALS = 20
FEATURE_SIZES = [10, 20, 30, 50, 75, 100]
MODELS = ["lightgbm", "xgboost", "catboost", "ridge"]
N_FOLDS = 5
TARGETS = ["angle", "depth", "left_right"]

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_SPLIT_DIR = PROJECT_DIR / "data_split"
OUTPUT_DIR = PROJECT_DIR / "output"
RESULTS_FILE = OUTPUT_DIR / "brute_force_results.csv"
LOG_FILE = OUTPUT_DIR / "brute_force_search.log"

# Columns to exclude from features (id, participant flags, target)
EXCLUDE_COLS = [
    "id", "target",
    "participant_1", "participant_2", "participant_3",
    "participant_4", "participant_5", "participant_id"
]


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to file and console."""
    logger = logging.getLogger("brute_force_search")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    return logger


# =============================================================================
# Data Loading
# =============================================================================

def load_scalers() -> Dict:
    """Load target scalers for proper metric computation."""
    scalers = {}
    for target in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        if scaler_path.exists():
            scalers[target] = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    return scalers


def load_data_from_split() -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    """
    Load data from data_split directory.

    Returns:
        X_df: DataFrame with all features
        y_dict: Dict mapping target name to target array
        groups: participant IDs for GroupKFold
    """
    all_data = []

    # Load from all players and all targets
    for player_dir in sorted(DATA_SPLIT_DIR.iterdir()):
        if not player_dir.is_dir():
            continue

        # Use angle.csv as the base (they all have the same features)
        angle_file = player_dir / "angle.csv"
        if angle_file.exists():
            df = pd.read_csv(angle_file)
            df["_source_player"] = player_dir.name
            all_data.append(df)

    if not all_data:
        raise ValueError("No data found in data_split directory")

    # Combine all data
    full_df = pd.concat(all_data, ignore_index=True)

    # Extract participant_id (from one-hot columns if needed)
    if "participant_id" in full_df.columns:
        groups = full_df["participant_id"].values
    else:
        # Reconstruct from one-hot
        participant_cols = [c for c in full_df.columns if c.startswith("participant_")]
        for i, col in enumerate(sorted(participant_cols)):
            if full_df[col].iloc[0] == 1:
                groups = np.full(len(full_df), i + 1)
                break
        else:
            groups = np.ones(len(full_df))  # Fallback

    # Get feature columns
    feature_cols = [c for c in full_df.columns
                    if c not in EXCLUDE_COLS and c != "_source_player"]

    X_df = full_df[feature_cols].copy()

    # Load targets from each target file
    y_dict = {}
    for target in TARGETS:
        target_dfs = []
        for player_dir in sorted(DATA_SPLIT_DIR.iterdir()):
            if not player_dir.is_dir():
                continue
            target_file = player_dir / f"{target}.csv"
            if target_file.exists():
                df = pd.read_csv(target_file)
                target_dfs.append(df["target"].values)

        if target_dfs:
            y_dict[target] = np.concatenate(target_dfs)

    return X_df, y_dict, groups


# =============================================================================
# Feature Subset Generation
# =============================================================================

def generate_feature_subset(
    all_features: List[str],
    size: int,
    seed: int
) -> List[str]:
    """Generate a random subset of features."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(all_features), size=min(size, len(all_features)), replace=False)
    return [all_features[i] for i in sorted(indices)]


def get_trial_hash(features: List[str], model_type: str) -> str:
    """Generate a unique hash for a (feature_subset, model) combination."""
    feature_str = ",".join(sorted(features))
    combo_str = f"{model_type}:{feature_str}"
    return hashlib.md5(combo_str.encode()).hexdigest()[:16]


# =============================================================================
# Model Factories
# =============================================================================

def get_model_and_params(model_type: str, params: Optional[Dict] = None):
    """Get model class and parameters for a given model type."""
    if model_type == "lightgbm":
        import lightgbm as lgb
        base_params = {
            "n_estimators": 500,
            "num_leaves": 20,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 10,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        if params:
            base_params.update(params)
        return lgb.LGBMRegressor, base_params

    elif model_type == "xgboost":
        import xgboost as xgb
        base_params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "random_state": 42,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }
        if params:
            base_params.update(params)
        return xgb.XGBRegressor, base_params

    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        base_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        }
        if params:
            base_params.update(params)
        return CatBoostRegressor, base_params

    elif model_type == "ridge":
        base_params = {
            "alpha": 1.0,
        }
        if params:
            base_params.update(params)
        return Ridge, base_params

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_optuna_search_space(model_type: str) -> Dict[str, Any]:
    """Get Optuna search space for hyperparameter tuning."""
    if model_type == "lightgbm":
        return {
            "n_estimators": ("int", 100, 800),
            "num_leaves": ("int", 5, 50),
            "learning_rate": ("float_log", 0.005, 0.1),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "reg_alpha": ("float_log", 1e-4, 10),
            "reg_lambda": ("float_log", 1e-4, 10),
            "min_child_samples": ("int", 5, 50),
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": ("int", 100, 800),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 0.005, 0.1),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "reg_alpha": ("float_log", 1e-4, 10),
            "reg_lambda": ("float_log", 1e-4, 10),
            "min_child_weight": ("int", 1, 20),
        }
    elif model_type == "catboost":
        return {
            "iterations": ("int", 100, 800),
            "depth": ("int", 3, 10),
            "learning_rate": ("float_log", 0.005, 0.1),
            "l2_leaf_reg": ("float_log", 0.1, 30),
        }
    elif model_type == "ridge":
        return {
            "alpha": ("float_log", 1e-4, 100),
        }
    else:
        return {}


def sample_optuna_params(trial, model_type: str) -> Dict[str, Any]:
    """Sample hyperparameters using Optuna trial."""
    search_space = get_optuna_search_space(model_type)
    params = {}

    for param_name, spec in search_space.items():
        if spec[0] == "int":
            params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
        elif spec[0] == "float":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2])
        elif spec[0] == "float_log":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=True)

    return params


# =============================================================================
# Cross-Validation
# =============================================================================

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_class,
    model_params: Dict,
    n_folds: int = N_FOLDS,
) -> Tuple[float, np.ndarray]:
    """
    Run GroupKFold CV and return MSE and OOF predictions.

    Returns:
        (mean_mse, oof_predictions)
    """
    gkf = GroupKFold(n_splits=n_folds)
    fold_mses = []
    oof_predictions = np.zeros_like(y, dtype=float)

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        oof_predictions[val_idx] = y_pred
        fold_mses.append(mean_squared_error(y_val, y_pred))

    return np.mean(fold_mses), oof_predictions


# =============================================================================
# Trial Execution
# =============================================================================

def run_trial(
    trial_id: int,
    features: List[str],
    model_type: str,
    X_df: pd.DataFrame,
    y_dict: Dict[str, np.ndarray],
    groups: np.ndarray,
    scalers: Dict,
    n_optuna_trials: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run a single trial: Optuna tuning + CV evaluation.

    Returns dictionary with trial results.
    """
    start_time = time.time()

    # Extract feature subset
    X = X_df[features].values

    # Create Optuna objective for combined score
    def objective(optuna_trial):
        params = sample_optuna_params(optuna_trial, model_type)
        model_class, base_params = get_model_and_params(model_type, params)

        total_scaled_mse = 0
        for target in TARGETS:
            y = y_dict[target]
            _, oof = run_cv(X, y, groups, model_class, base_params, n_folds=N_FOLDS)

            # Scale predictions for proper metric
            y_scaled = scalers[target].transform(y.reshape(-1, 1)).ravel()
            oof_scaled = scalers[target].transform(oof.reshape(-1, 1)).ravel()
            total_scaled_mse += mean_squared_error(y_scaled, oof_scaled)

        return total_scaled_mse / 3

    # Run Optuna optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + trial_id)
    )
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

    best_params = study.best_params

    # Evaluate best params per target
    model_class, final_params = get_model_and_params(model_type, best_params)

    per_target_mse = {}
    for target in TARGETS:
        y = y_dict[target]
        _, oof = run_cv(X, y, groups, model_class, final_params, n_folds=N_FOLDS)

        y_scaled = scalers[target].transform(y.reshape(-1, 1)).ravel()
        oof_scaled = scalers[target].transform(oof.reshape(-1, 1)).ravel()
        per_target_mse[target] = mean_squared_error(y_scaled, oof_scaled)

    combined_mse = np.mean(list(per_target_mse.values()))
    duration = time.time() - start_time

    return {
        "trial_id": trial_id,
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "n_features": len(features),
        "feature_indices": ",".join(features),
        "best_params": json.dumps(best_params),
        "cv_score_angle": per_target_mse["angle"],
        "cv_score_depth": per_target_mse["depth"],
        "cv_score_left_right": per_target_mse["left_right"],
        "cv_score_combined": combined_mse,
        "optuna_trials": n_optuna_trials,
        "duration_seconds": round(duration, 1),
    }


# =============================================================================
# Results Management
# =============================================================================

def load_existing_results(results_file: Path) -> pd.DataFrame:
    """Load existing results from CSV."""
    if results_file.exists():
        return pd.read_csv(results_file)
    return pd.DataFrame()


def save_result(result: Dict, results_file: Path):
    """Append a single result to CSV with file locking."""
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Use file locking for safe concurrent writes
    with open(results_file, 'a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Check if file is empty (needs header)
            f.seek(0, 2)  # Go to end
            needs_header = f.tell() == 0

            df = pd.DataFrame([result])
            df.to_csv(f, header=needs_header, index=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_completed_hashes(results_df: pd.DataFrame) -> set:
    """Get set of completed trial hashes from existing results."""
    if results_df.empty:
        return set()

    hashes = set()
    for _, row in results_df.iterrows():
        features = row["feature_indices"].split(",")
        model_type = row["model_type"]
        hashes.add(get_trial_hash(features, model_type))

    return hashes


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Brute force feature-algorithm search"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting trial index"
    )
    parser.add_argument(
        "--end", type=int, default=TOTAL_TRIALS,
        help="Ending trial index"
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=OPTUNA_TRIALS,
        help="Number of Optuna trials per combination"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test mode (10 trials, 5 Optuna trials each)"
    )

    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_FILE)

    # Quick test mode
    if args.quick:
        args.end = min(args.start + 10, args.end)
        args.optuna_trials = 5
        logger.info("QUICK TEST MODE: 10 trials, 5 Optuna trials each")

    total_trials = args.end - args.start
    logger.info(f"Starting brute force search ({total_trials} trials, seed={args.seed})")
    logger.info(f"Trial range: {args.start} to {args.end}")
    logger.info(f"Optuna trials per combination: {args.optuna_trials}")
    logger.info(f"Feature sizes: {FEATURE_SIZES}")
    logger.info(f"Models: {MODELS}")

    # Load data
    logger.info("Loading data...")
    X_df, y_dict, groups = load_data_from_split()
    scalers = load_scalers()

    all_features = list(X_df.columns)
    n_features_total = len(all_features)
    n_samples = len(X_df)

    logger.info(f"Data loaded: {n_samples} samples, {n_features_total} features")

    # Load existing results for resumption
    existing_results = load_existing_results(RESULTS_FILE)
    completed_hashes = get_completed_hashes(existing_results)
    logger.info(f"Found {len(completed_hashes)} completed trials (will skip)")

    # Initialize RNG for reproducible trial generation
    rng = np.random.RandomState(args.seed)

    # Pre-generate all trial configurations
    trial_configs = []
    for trial_id in range(args.start, args.end):
        # Random feature size
        feature_size = rng.choice(FEATURE_SIZES)

        # Random model
        model_type = rng.choice(MODELS)

        # Random feature subset (use trial_id as part of seed for reproducibility)
        feature_seed = args.seed * 10000 + trial_id
        features = generate_feature_subset(all_features, feature_size, feature_seed)

        trial_configs.append({
            "trial_id": trial_id,
            "features": features,
            "model_type": model_type,
        })

    # Run trials
    completed = 0
    skipped = 0
    best_score = float("inf")
    best_trial_id = None

    for i, config in enumerate(trial_configs):
        trial_id = config["trial_id"]
        features = config["features"]
        model_type = config["model_type"]

        # Check if already completed
        trial_hash = get_trial_hash(features, model_type)
        if trial_hash in completed_hashes:
            skipped += 1
            continue

        # Run trial
        try:
            result = run_trial(
                trial_id=trial_id,
                features=features,
                model_type=model_type,
                X_df=X_df,
                y_dict=y_dict,
                groups=groups,
                scalers=scalers,
                n_optuna_trials=args.optuna_trials,
                seed=args.seed,
            )

            # Save result
            save_result(result, RESULTS_FILE)
            completed_hashes.add(trial_hash)
            completed += 1

            # Track best
            if result["cv_score_combined"] < best_score:
                best_score = result["cv_score_combined"]
                best_trial_id = trial_id

            # Log progress
            progress_pct = (completed + skipped) / total_trials * 100
            logger.info(
                f"Trial {trial_id}/{args.end-1}: {model_type}, "
                f"{len(features)} features -> {result['cv_score_combined']:.6f} "
                f"({result['duration_seconds']:.1f}s)"
            )

            # Periodic summary
            if completed % 10 == 0:
                logger.info(
                    f"Progress: {completed + skipped}/{total_trials} ({progress_pct:.1f}%) | "
                    f"Best so far: {best_score:.6f} (trial {best_trial_id})"
                )

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            continue

    # Final summary
    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info(f"Completed: {completed} trials")
    logger.info(f"Skipped (already done): {skipped} trials")
    logger.info(f"Best combined score: {best_score:.6f} (trial {best_trial_id})")
    logger.info(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
