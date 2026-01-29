"""
Per-Target Independent Hyperparameter Tuning Experiment.

This experiment properly tests whether separate models per target benefit from
independent hyperparameter tuning - something the previous S1/S2/S3/S4 comparison
failed to test (they all used identical hyperparameters).

Experiment Design:
1. Baseline: Per-player models with SHARED hyperparameters (current S3)
2. Test: Per-player + per-target models with INDEPENDENT hyperparameters

For each target (angle, depth, left_right), we tune hyperparameters separately
to find the optimal configuration for that specific prediction task.
"""

import argparse
import json
import numpy as np
import pandas as pd
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

import pickle
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from model_factory import get_model_config, sample_optuna_params
except ImportError:
    from src.model_factory import get_model_config, sample_optuna_params


OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_FILE = OUTPUT_DIR / "per_target_experiment_results.csv"
SEED = 42
N_FOLDS = 5
TARGETS = ["angle", "depth", "left_right"]


def load_scalers() -> Dict:
    """Load the target scalers."""
    scalers = {}
    for target in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        scalers[target] = joblib.load(scaler_path)
    return scalers


def load_cached_features(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached features from pickle file.

    Returns:
        X, y, participant_ids
    """
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']

    # Get participant IDs from meta
    if isinstance(data['meta'], pd.DataFrame):
        groups = data['meta']['participant_id'].values
    else:
        groups = data['meta'][:, 2]  # participant_id is 3rd column

    return X, y, groups


def apply_preprocessing(
    X_train: np.ndarray,
    X_val: np.ndarray,
    preprocessing_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply preprocessing to train and validation data."""
    if preprocessing_id in ["P1", "none"]:
        return X_train, X_val
    elif preprocessing_id in ["P3", "clipped"]:
        X_train_clipped = X_train.copy()
        X_val_clipped = X_val.copy()
        for i in range(X_train.shape[1]):
            p1, p99 = np.percentile(X_train[:, i], [1, 99])
            X_train_clipped[:, i] = np.clip(X_train[:, i], p1, p99)
            X_val_clipped[:, i] = np.clip(X_val[:, i], p1, p99)
        return X_train_clipped, X_val_clipped
    elif preprocessing_id in ["P4", "standardized"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled
    else:
        return X_train, X_val


def run_single_target_cv(
    X: np.ndarray,
    y_single: np.ndarray,
    groups: np.ndarray,
    model_class,
    model_params: Dict,
    preprocessing_id: str,
    per_participant: bool = False,
    n_folds: int = N_FOLDS,
) -> Dict[str, Any]:
    """
    Run CV for a single target.

    Args:
        X: Feature matrix
        y_single: Target values (1D array for single target)
        groups: Participant IDs for GroupKFold
        model_class: Model class to use
        model_params: Model hyperparameters
        preprocessing_id: Preprocessing to apply
        per_participant: If True, train separate model per participant
        n_folds: Number of CV folds

    Returns:
        Dictionary with CV results
    """
    gkf = GroupKFold(n_splits=n_folds)
    fold_mses = []
    oof_predictions = np.zeros_like(y_single)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_single, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_single[train_idx], y_single[val_idx]
        groups_train = groups[train_idx]
        groups_val = groups[val_idx]

        # Apply preprocessing
        X_train_proc, X_val_proc = apply_preprocessing(X_train, X_val, preprocessing_id)

        if per_participant:
            # Train separate model per participant in training set
            # For held-out participant (GroupKFold), use a model trained on all training data
            participant_models = {}
            for pid in np.unique(groups_train):
                train_mask = groups_train == pid
                model = model_class(**model_params)
                model.fit(X_train_proc[train_mask], y_train[train_mask])
                participant_models[pid] = model

            # Also train a fallback model on all training data for held-out participants
            fallback_model = model_class(**model_params)
            fallback_model.fit(X_train_proc, y_train)

            # Predict using participant-specific model if available, else fallback
            y_pred = np.zeros(len(val_idx))
            for pid in np.unique(groups_val):
                val_mask = groups_val == pid
                if pid in participant_models:
                    y_pred[val_mask] = participant_models[pid].predict(X_val_proc[val_mask])
                else:
                    # Held-out participant - use fallback model
                    y_pred[val_mask] = fallback_model.predict(X_val_proc[val_mask])
        else:
            # Single model for all participants
            model = model_class(**model_params)
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_val_proc)

        oof_predictions[val_idx] = y_pred
        fold_mse = mean_squared_error(y_val, y_pred)
        fold_mses.append(fold_mse)

    return {
        "mse_cv": np.mean(fold_mses),
        "mse_std": np.std(fold_mses),
        "oof_predictions": oof_predictions,
    }


def tune_hyperparameters_for_target(
    X: np.ndarray,
    y_single: np.ndarray,
    groups: np.ndarray,
    target_name: str,
    model_id: str,
    preprocessing_id: str,
    per_participant: bool,
    n_trials: int,
    scaler,
) -> Tuple[Dict, float]:
    """
    Tune hyperparameters for a single target using Optuna.

    Returns:
        (best_params, best_scaled_mse)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model_class, base_params = get_model_config(model_id)

    def objective(trial):
        trial_params = sample_optuna_params(trial, model_id)
        merged_params = {**base_params, **trial_params}

        results = run_single_target_cv(
            X, y_single, groups,
            model_class, merged_params,
            preprocessing_id, per_participant
        )

        # Scale predictions for proper metric
        oof = results["oof_predictions"]
        y_scaled = scaler.transform(y_single.reshape(-1, 1)).ravel()
        oof_scaled = scaler.transform(oof.reshape(-1, 1)).ravel()
        scaled_mse = mean_squared_error(y_scaled, oof_scaled)

        return scaled_mse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**base_params, **study.best_params}
    return best_params, study.best_value


def run_experiment(
    cache_dir: Path,
    n_trials: int = 30,
    feature_set: str = "F4",
    model_id: str = "M1",
    preprocessing: str = "P4",
    quick_test: bool = False,
):
    """
    Run the full per-target experiment.

    Compares:
    1. Baseline: Per-player, shared hyperparams across targets
    2. Test: Per-player, independent hyperparams per target
    """
    print("\n" + "="*70)
    print("PER-TARGET INDEPENDENT HYPERPARAMETER TUNING EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Feature set: {feature_set}")
    print(f"  Model: {model_id}")
    print(f"  Preprocessing: {preprocessing}")
    print(f"  Optuna trials per target: {n_trials}")

    if quick_test:
        n_trials = 5
        print(f"  [QUICK TEST MODE: {n_trials} trials]")

    # Load data from cache
    scalers = load_scalers()

    # Try to load from cache
    cache_file = cache_dir / f"features_{feature_set}_smooth.pkl"
    if not cache_file.exists():
        # Fallback to the_rest cache
        cache_file = Path(__file__).parent.parent / "the_rest" / "output" / "feature_cache" / f"features_{feature_set}_smooth.pkl"

    if not cache_file.exists():
        # Use default F1 features from output/features_train.pkl
        cache_file = OUTPUT_DIR / "features_train.pkl"
        print(f"  Using default features from {cache_file}")

    X, y, groups = load_cached_features(cache_file)
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")

    model_class, base_params = get_model_config(model_id)

    results_summary = []

    # =========================================================================
    # PHASE 1: Baseline - Shared hyperparameters
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 1: Baseline (shared hyperparams across targets)")
    print("-"*70)

    baseline_oof = np.zeros_like(y)
    baseline_per_target = {}

    for i, target in enumerate(TARGETS):
        print(f"\n  {target}...")
        y_target = y[:, i]

        results = run_single_target_cv(
            X, y_target, groups,
            model_class, base_params,
            preprocessing, per_participant=True
        )

        baseline_oof[:, i] = results["oof_predictions"]

        # Compute scaled MSE
        y_scaled = scalers[target].transform(y_target.reshape(-1, 1)).ravel()
        oof_scaled = scalers[target].transform(results["oof_predictions"].reshape(-1, 1)).ravel()
        scaled_mse = mean_squared_error(y_scaled, oof_scaled)
        baseline_per_target[target] = scaled_mse

        print(f"    Scaled MSE: {scaled_mse:.6f}")

    baseline_total = np.mean(list(baseline_per_target.values()))
    print(f"\n  BASELINE TOTAL: {baseline_total:.6f}")

    results_summary.append({
        "experiment": "baseline_shared_hyperparams",
        "angle_mse": baseline_per_target["angle"],
        "depth_mse": baseline_per_target["depth"],
        "left_right_mse": baseline_per_target["left_right"],
        "total_scaled_mse": baseline_total,
        "hyperparams": json.dumps({"shared": base_params}),
    })

    # =========================================================================
    # PHASE 2: Tune hyperparameters independently per target
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 2: Independent hyperparameter tuning per target")
    print("-"*70)

    tuned_params = {}
    tuned_per_target = {}
    tuned_oof = np.zeros_like(y)

    for i, target in enumerate(TARGETS):
        print(f"\n  Tuning {target} ({n_trials} trials)...")
        y_target = y[:, i]

        start_time = time.time()
        best_params, best_mse = tune_hyperparameters_for_target(
            X, y_target, groups,
            target, model_id, preprocessing,
            per_participant=True,
            n_trials=n_trials,
            scaler=scalers[target],
        )
        elapsed = time.time() - start_time

        tuned_params[target] = best_params
        tuned_per_target[target] = best_mse

        print(f"    Best scaled MSE: {best_mse:.6f} ({elapsed:.1f}s)")
        print(f"    Key params: n_est={best_params.get('n_estimators', best_params.get('iterations', 'N/A'))}, "
              f"lr={best_params.get('learning_rate', 'N/A'):.4f}")

        # Get OOF predictions with tuned params
        model_class_tuned, _ = get_model_config(model_id)
        results = run_single_target_cv(
            X, y_target, groups,
            model_class_tuned, best_params,
            preprocessing, per_participant=True
        )
        tuned_oof[:, i] = results["oof_predictions"]

    tuned_total = np.mean(list(tuned_per_target.values()))
    print(f"\n  TUNED TOTAL: {tuned_total:.6f}")

    results_summary.append({
        "experiment": "per_target_tuned_hyperparams",
        "angle_mse": tuned_per_target["angle"],
        "depth_mse": tuned_per_target["depth"],
        "left_right_mse": tuned_per_target["left_right"],
        "total_scaled_mse": tuned_total,
        "hyperparams": json.dumps(tuned_params),
    })

    # =========================================================================
    # PHASE 3: Compare to global tuning (tune on combined metric)
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 3: Global hyperparameter tuning (optimize combined metric)")
    print("-"*70)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def global_objective(trial):
        trial_params = sample_optuna_params(trial, model_id)
        merged_params = {**base_params, **trial_params}

        total_mse = 0
        for i, target in enumerate(TARGETS):
            y_target = y[:, i]
            results = run_single_target_cv(
                X, y_target, groups,
                model_class, merged_params,
                preprocessing, per_participant=True
            )

            y_scaled = scalers[target].transform(y_target.reshape(-1, 1)).ravel()
            oof_scaled = scalers[target].transform(results["oof_predictions"].reshape(-1, 1)).ravel()
            total_mse += mean_squared_error(y_scaled, oof_scaled)

        return total_mse / 3

    print(f"\n  Tuning globally ({n_trials} trials)...")
    start_time = time.time()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(global_objective, n_trials=n_trials, show_progress_bar=False)

    elapsed = time.time() - start_time
    global_params = {**base_params, **study.best_params}

    print(f"    Best combined MSE: {study.best_value:.6f} ({elapsed:.1f}s)")

    # Evaluate globally tuned params per target
    global_per_target = {}
    global_oof = np.zeros_like(y)

    for i, target in enumerate(TARGETS):
        y_target = y[:, i]
        results = run_single_target_cv(
            X, y_target, groups,
            model_class, global_params,
            preprocessing, per_participant=True
        )
        global_oof[:, i] = results["oof_predictions"]

        y_scaled = scalers[target].transform(y_target.reshape(-1, 1)).ravel()
        oof_scaled = scalers[target].transform(results["oof_predictions"].reshape(-1, 1)).ravel()
        global_per_target[target] = mean_squared_error(y_scaled, oof_scaled)

    global_total = np.mean(list(global_per_target.values()))

    print(f"\n  Per-target breakdown:")
    for target in TARGETS:
        print(f"    {target}: {global_per_target[target]:.6f}")
    print(f"  GLOBAL TUNED TOTAL: {global_total:.6f}")

    results_summary.append({
        "experiment": "global_tuned_hyperparams",
        "angle_mse": global_per_target["angle"],
        "depth_mse": global_per_target["depth"],
        "left_right_mse": global_per_target["left_right"],
        "total_scaled_mse": global_total,
        "hyperparams": json.dumps({"global": global_params}),
    })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print(f"\n{'Approach':<35} {'Angle':>10} {'Depth':>10} {'L/R':>10} {'Total':>10}")
    print("-"*75)

    print(f"{'Baseline (shared params)':<35} "
          f"{baseline_per_target['angle']:>10.6f} "
          f"{baseline_per_target['depth']:>10.6f} "
          f"{baseline_per_target['left_right']:>10.6f} "
          f"{baseline_total:>10.6f}")

    print(f"{'Global tuned params':<35} "
          f"{global_per_target['angle']:>10.6f} "
          f"{global_per_target['depth']:>10.6f} "
          f"{global_per_target['left_right']:>10.6f} "
          f"{global_total:>10.6f}")

    print(f"{'Per-target tuned params':<35} "
          f"{tuned_per_target['angle']:>10.6f} "
          f"{tuned_per_target['depth']:>10.6f} "
          f"{tuned_per_target['left_right']:>10.6f} "
          f"{tuned_total:>10.6f}")

    # Improvement calculations
    print("\n" + "-"*75)

    global_improvement = (baseline_total - global_total) / baseline_total * 100
    per_target_improvement = (baseline_total - tuned_total) / baseline_total * 100
    per_target_vs_global = (global_total - tuned_total) / global_total * 100

    print(f"Global tuning vs baseline:     {global_improvement:+.2f}%")
    print(f"Per-target tuning vs baseline: {per_target_improvement:+.2f}%")
    print(f"Per-target vs global tuning:   {per_target_vs_global:+.2f}%")

    # Save results
    df = pd.DataFrame(results_summary)
    df["timestamp"] = datetime.now().isoformat()
    df["feature_set"] = feature_set
    df["model"] = model_id
    df["preprocessing"] = preprocessing
    df["n_trials"] = n_trials

    if RESULTS_FILE.exists():
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)

    print(f"\nResults saved to: {RESULTS_FILE}")

    # Return best configuration
    if tuned_total <= min(baseline_total, global_total):
        print("\n*** CONCLUSION: Per-target tuning is BEST ***")
        return tuned_params, tuned_total
    elif global_total <= baseline_total:
        print("\n*** CONCLUSION: Global tuning is BEST ***")
        return global_params, global_total
    else:
        print("\n*** CONCLUSION: Baseline is BEST (tuning didn't help) ***")
        return base_params, baseline_total


def main():
    parser = argparse.ArgumentParser(description="Per-target hyperparameter tuning experiment")
    parser.add_argument("--feature-set", type=str, default="F4", help="Feature set to use (F4 cached)")
    parser.add_argument("--model", type=str, default="M1", help="Model to use")
    parser.add_argument("--preprocessing", type=str, default="P4", help="Preprocessing to use")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per target")
    parser.add_argument("--quick", action="store_true", help="Quick test with 5 trials")
    parser.add_argument("--cache-dir", type=str, default=str(OUTPUT_DIR / "feature_cache"))

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    run_experiment(
        cache_dir=cache_dir,
        n_trials=args.n_trials,
        feature_set=args.feature_set,
        model_id=args.model,
        preprocessing=args.preprocessing,
        quick_test=args.quick,
    )


if __name__ == "__main__":
    main()
