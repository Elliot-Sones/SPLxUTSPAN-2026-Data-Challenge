"""
Exhaustive Grid Search for Minimum MSE.

Systematically tests ALL combinations of:
- 8 feature sets (F1-F8)
- 8 models (M1-M8)
- 4 preprocessing options (P1-P4)
- 4 training strategies (S1-S4)

Plus Optuna hyperparameter tuning and ensemble construction.

Usage:
    uv run python src/exhaustive_grid_search.py --all-phases
    uv run python src/exhaustive_grid_search.py --phase 1
    uv run python src/exhaustive_grid_search.py --phase 2 --feature-set F3
"""

import argparse
import json
import numpy as np
import pandas as pd
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

try:
    from data_loader import (
        iterate_shots, load_metadata, get_keypoint_columns,
        load_scalers, DATA_DIR, TARGET_COLS, NUM_FRAMES
    )
    from feature_factory import (
        get_feature_extractor, FEATURE_SETS, ensure_keypoint_mapping,
        set_top100_features
    )
    from model_factory import (
        get_model_config, MODEL_CONFIGS, get_model_with_params,
        sample_optuna_params, get_available_models, is_model_available
    )
    from physics_features import init_keypoint_mapping
except ImportError:
    from src.data_loader import (
        iterate_shots, load_metadata, get_keypoint_columns,
        load_scalers, DATA_DIR, TARGET_COLS, NUM_FRAMES
    )
    from src.feature_factory import (
        get_feature_extractor, FEATURE_SETS, ensure_keypoint_mapping,
        set_top100_features
    )
    from src.model_factory import (
        get_model_config, MODEL_CONFIGS, get_model_with_params,
        sample_optuna_params, get_available_models, is_model_available
    )
    from src.physics_features import init_keypoint_mapping


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "grid_search_results.csv"
OOF_DIR = OUTPUT_DIR / "oof_predictions"
OOF_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "best_models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5


# Preprocessing options
PREPROCESSING_OPTIONS = {
    "P1": "none",
    "P2": "smoothed",
    "P3": "clipped",
    "P4": "standardized",
}

# Training strategies
TRAINING_STRATEGIES = {
    "S1": "joint",
    "S2": "separate",
    "S3": "per_participant",
    "S4": "per_participant_separate",
}


# =============================================================================
# Data Loading and Feature Extraction
# =============================================================================

def extract_features_for_dataset(
    feature_set_id: str,
    train: bool = True,
    smooth: bool = False,
    max_shots: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Extract features for the entire dataset using specified feature set.

    Args:
        feature_set_id: Feature set identifier (F1-F8)
        train: Use training data
        smooth: Apply smoothing during extraction
        max_shots: Limit number of shots
        cache_dir: Directory for caching

    Returns:
        X, y, feature_names, metadata
    """
    # Check cache
    cache_suffix = "_smooth" if smooth else ""
    cache_file = cache_dir / f"features_{feature_set_id}{cache_suffix}.pkl" if cache_dir else None

    if cache_file and cache_file.exists():
        print(f"  Loading cached features from {cache_file.name}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["X"], cached["y"], cached["feature_names"], cached["meta"]

    # Initialize
    ensure_keypoint_mapping()
    extractor = get_feature_extractor(feature_set_id)

    # Load metadata
    meta = load_metadata(train)
    n_shots = len(meta) if max_shots is None else min(max_shots, len(meta))

    print(f"  Extracting {feature_set_id} features for {n_shots} shots...")

    # Extract features
    all_features = []
    all_targets = []
    processed = 0

    start_time = time.time()
    for metadata, timeseries in iterate_shots(train, chunk_size=20):
        if processed >= n_shots:
            break

        features = extractor(
            timeseries,
            participant_id=metadata["participant_id"],
            smooth=smooth
        )
        all_features.append(features)

        if train:
            all_targets.append([
                metadata["angle"],
                metadata["depth"],
                metadata["left_right"]
            ])

        processed += 1
        if processed % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    Processed {processed}/{n_shots} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"  Feature extraction: {processed} shots in {elapsed:.1f}s")

    # Convert to arrays
    feature_names = sorted(all_features[0].keys())
    X = np.array([
        [f.get(name, np.nan) for name in feature_names]
        for f in all_features
    ], dtype=np.float32)

    y = np.array(all_targets, dtype=np.float32) if train else None

    # Handle NaN values (median imputation)
    for i in range(X.shape[1]):
        col = X[:, i]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            X[nan_mask, i] = median_val

    # Cache
    if cache_file:
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(cache_file, "wb") as f:
            pickle.dump({
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "meta": meta.iloc[:n_shots]
            }, f)

    return X, y, feature_names, meta.iloc[:n_shots]


# =============================================================================
# Preprocessing Functions
# =============================================================================

def apply_preprocessing(
    X_train: np.ndarray,
    X_val: np.ndarray,
    preprocessing_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply preprocessing to train and validation data.

    Note: smoothing (P2) should be applied during feature extraction.
    """
    if preprocessing_id in ["P1", "none"]:
        return X_train, X_val

    elif preprocessing_id in ["P3", "clipped"]:
        # Clip outliers to [1st, 99th] percentile
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


# =============================================================================
# Training Strategy Implementations
# =============================================================================

class JointModel:
    """S1: Single model for all targets using MultiOutputRegressor."""

    def __init__(self, model_class, model_params):
        from sklearn.multioutput import MultiOutputRegressor
        self.wrapper = MultiOutputRegressor(model_class(**model_params))

    def fit(self, X, y):
        self.wrapper.fit(X, y)
        return self

    def predict(self, X):
        return self.wrapper.predict(X)


class SeparateTargetModel:
    """S2: Separate model per target."""

    def __init__(self, model_class, model_params):
        self.model_class = model_class
        self.model_params = model_params
        self.models = {}

    def fit(self, X, y):
        for i, target in enumerate(["angle", "depth", "left_right"]):
            self.models[target] = self.model_class(**self.model_params)
            self.models[target].fit(X, y[:, i])
        return self

    def predict(self, X):
        preds = np.zeros((X.shape[0], 3))
        for i, target in enumerate(["angle", "depth", "left_right"]):
            preds[:, i] = self.models[target].predict(X)
        return preds


class PerParticipantModel:
    """S3: Separate model per participant (joint targets)."""

    def __init__(self, model_class, model_params):
        from sklearn.multioutput import MultiOutputRegressor
        self.model_class = model_class
        self.model_params = model_params
        self.models = {}  # pid -> model

    def fit(self, X, y, participant_ids):
        self.participant_ids = participant_ids
        for pid in np.unique(participant_ids):
            mask = participant_ids == pid
            from sklearn.multioutput import MultiOutputRegressor
            model = MultiOutputRegressor(self.model_class(**self.model_params))
            model.fit(X[mask], y[mask])
            self.models[pid] = model
        return self

    def predict(self, X, participant_ids):
        preds = np.zeros((X.shape[0], 3))
        for pid in np.unique(participant_ids):
            mask = participant_ids == pid
            if pid in self.models:
                preds[mask] = self.models[pid].predict(X[mask])
            else:
                # Fall back to any available model
                preds[mask] = list(self.models.values())[0].predict(X[mask])
        return preds


class PerParticipantSeparateModel:
    """S4: Separate model per participant AND per target (15 models)."""

    def __init__(self, model_class, model_params):
        self.model_class = model_class
        self.model_params = model_params
        self.models = {}  # (pid, target) -> model

    def fit(self, X, y, participant_ids):
        self.participant_ids = participant_ids
        for pid in np.unique(participant_ids):
            mask = participant_ids == pid
            for i, target in enumerate(["angle", "depth", "left_right"]):
                model = self.model_class(**self.model_params)
                model.fit(X[mask], y[mask, i])
                self.models[(pid, target)] = model
        return self

    def predict(self, X, participant_ids):
        preds = np.zeros((X.shape[0], 3))
        for pid in np.unique(participant_ids):
            mask = participant_ids == pid
            for i, target in enumerate(["angle", "depth", "left_right"]):
                key = (pid, target)
                if key in self.models:
                    preds[mask, i] = self.models[key].predict(X[mask])
                else:
                    # Fall back to any model for this target
                    fallback = [k for k in self.models.keys() if k[1] == target]
                    if fallback:
                        preds[mask, i] = self.models[fallback[0]].predict(X[mask])
        return preds


def create_model_wrapper(
    model_class,
    model_params: Dict,
    strategy_id: str
):
    """Create appropriate model wrapper based on training strategy."""
    if strategy_id in ["S1", "joint"]:
        return JointModel(model_class, model_params)
    elif strategy_id in ["S2", "separate"]:
        return SeparateTargetModel(model_class, model_params)
    elif strategy_id in ["S3", "per_participant"]:
        return PerParticipantModel(model_class, model_params)
    elif strategy_id in ["S4", "per_participant_separate"]:
        return PerParticipantSeparateModel(model_class, model_params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_id}")


# =============================================================================
# Cross-Validation
# =============================================================================

def compute_scaled_mse(y_true: np.ndarray, y_pred: np.ndarray, scalers: Dict) -> Tuple[float, Dict[str, float]]:
    """
    Compute scaled MSE matching competition metric.

    Returns:
        total_scaled_mse, per_target_mses
    """
    y_true_scaled = np.zeros_like(y_true)
    y_pred_scaled = np.zeros_like(y_pred)

    per_target = {}
    for i, target in enumerate(TARGET_COLS):
        y_true_scaled[:, i] = scalers[target].transform(y_true[:, i].reshape(-1, 1)).ravel()
        y_pred_scaled[:, i] = scalers[target].transform(y_pred[:, i].reshape(-1, 1)).ravel()
        per_target[target] = mean_squared_error(y_true_scaled[:, i], y_pred_scaled[:, i])

    total_mse = mean_squared_error(y_true_scaled, y_pred_scaled)
    return total_mse, per_target


def run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_class,
    model_params: Dict,
    preprocessing_id: str,
    strategy_id: str,
    scalers: Dict,
    n_folds: int = N_FOLDS,
) -> Dict[str, Any]:
    """
    Run a single cross-validation experiment.

    Returns:
        Dictionary with CV results
    """
    gkf = GroupKFold(n_splits=n_folds)

    fold_mses = []
    fold_per_target = {"angle": [], "depth": [], "left_right": []}
    oof_predictions = np.zeros_like(y)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]
        groups_val = groups[val_idx]

        # Apply preprocessing
        X_train_proc, X_val_proc = apply_preprocessing(X_train, X_val, preprocessing_id)

        # Create and fit model
        model = create_model_wrapper(model_class, model_params, strategy_id)

        if strategy_id in ["S3", "per_participant", "S4", "per_participant_separate"]:
            model.fit(X_train_proc, y_train, groups_train)
            y_pred = model.predict(X_val_proc, groups_val)
        else:
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_val_proc)

        oof_predictions[val_idx] = y_pred

        # Compute metrics
        fold_mse, per_target_mse = compute_scaled_mse(y_val, y_pred, scalers)
        fold_mses.append(fold_mse)
        for target in per_target_mse:
            fold_per_target[target].append(per_target_mse[target])

    # Aggregate results
    results = {
        "scaled_mse_cv": np.mean(fold_mses),
        "scaled_mse_std": np.std(fold_mses),
        "fold_mses": fold_mses,
        "angle_mse_cv": np.mean(fold_per_target["angle"]),
        "depth_mse_cv": np.mean(fold_per_target["depth"]),
        "left_right_mse_cv": np.mean(fold_per_target["left_right"]),
        "oof_predictions": oof_predictions,
    }

    return results


# =============================================================================
# Results Logging
# =============================================================================

def log_result(
    phase: int,
    experiment_id: str,
    feature_set: str,
    model: str,
    preprocessing: str,
    strategy: str,
    n_features: int,
    results: Dict,
    hyperparams: Dict = None,
):
    """Append experiment result to CSV."""
    row = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "experiment_id": experiment_id,
        "feature_set": feature_set,
        "model": model,
        "preprocessing": preprocessing,
        "strategy": strategy,
        "n_features": n_features,
        "angle_mse_cv": results["angle_mse_cv"],
        "depth_mse_cv": results["depth_mse_cv"],
        "left_right_mse_cv": results["left_right_mse_cv"],
        "scaled_mse_cv": results["scaled_mse_cv"],
        "scaled_mse_std": results["scaled_mse_std"],
        "hyperparams": json.dumps(hyperparams) if hyperparams else "",
    }

    df = pd.DataFrame([row])

    if RESULTS_FILE.exists():
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)


def print_results_summary(phase: int, top_n: int = 5):
    """Print summary of results for a phase."""
    if not RESULTS_FILE.exists():
        print("No results yet.")
        return

    df = pd.read_csv(RESULTS_FILE)
    phase_df = df[df["phase"] == phase].copy()

    if len(phase_df) == 0:
        print(f"No results for phase {phase}.")
        return

    print(f"\n{'='*70}")
    print(f"Phase {phase} Results - Top {top_n}")
    print(f"{'='*70}")

    phase_df = phase_df.sort_values("scaled_mse_cv")
    for i, row in phase_df.head(top_n).iterrows():
        print(f"\n{row['experiment_id']}: {row['scaled_mse_cv']:.6f}")
        print(f"  Feature: {row['feature_set']}, Model: {row['model']}")
        print(f"  Preprocess: {row['preprocessing']}, Strategy: {row['strategy']}")
        print(f"  Per-target: angle={row['angle_mse_cv']:.6f}, "
              f"depth={row['depth_mse_cv']:.6f}, "
              f"left_right={row['left_right_mse_cv']:.6f}")


def get_best_config(phase: int) -> Dict:
    """Get best configuration from a phase."""
    if not RESULTS_FILE.exists():
        return None

    df = pd.read_csv(RESULTS_FILE)
    phase_df = df[df["phase"] == phase]

    if len(phase_df) == 0:
        return None

    best = phase_df.loc[phase_df["scaled_mse_cv"].idxmin()]
    return best.to_dict()


# =============================================================================
# Phase Implementations
# =============================================================================

def run_phase_1(cache_dir: Path):
    """
    Phase 1: Quick Baseline Sweep

    Test most promising core combinations:
    - Features: F3 (hybrid), F4 (hybrid+pid)
    - Models: M1 (LightGBM), M3 (XGBoost), M5 (CatBoost)
    - Preprocessing: P1 (none)
    - Strategy: S2 (separate)

    6 experiments total.
    """
    print("\n" + "="*70)
    print("PHASE 1: Quick Baseline Sweep")
    print("="*70)

    feature_sets = ["F3", "F4"]
    models = ["M1", "M3", "M5"]
    preprocessing = "P1"
    strategy = "S2"

    scalers = load_scalers()
    available_models = get_available_models()

    # Filter to available models
    models = [m for m in models if m in available_models]
    if not models:
        print("ERROR: No models available. Install lightgbm, xgboost, or catboost.")
        return

    experiment_count = 0
    for feature_set in feature_sets:
        # Extract features once per feature set
        X, y, feature_names, meta = extract_features_for_dataset(
            feature_set, train=True, smooth=False, cache_dir=cache_dir
        )
        groups = meta["participant_id"].values

        for model_id in models:
            experiment_id = f"P1_{feature_set}_{model_id}"
            print(f"\nExperiment: {experiment_id}")

            model_class, model_params = get_model_config(model_id)

            start_time = time.time()
            results = run_cv_experiment(
                X, y, groups, model_class, model_params,
                preprocessing, strategy, scalers
            )
            elapsed = time.time() - start_time

            print(f"  Scaled MSE: {results['scaled_mse_cv']:.6f} +/- {results['scaled_mse_std']:.6f}")
            print(f"  Time: {elapsed:.1f}s")

            log_result(
                phase=1,
                experiment_id=experiment_id,
                feature_set=feature_set,
                model=model_id,
                preprocessing=preprocessing,
                strategy=strategy,
                n_features=X.shape[1],
                results=results,
                hyperparams=model_params
            )

            # Save OOF predictions
            np.save(OOF_DIR / f"{experiment_id}_oof.npy", results["oof_predictions"])

            experiment_count += 1

    print_results_summary(phase=1)
    return get_best_config(phase=1)


def run_phase_2(cache_dir: Path, best_model: str = None):
    """
    Phase 2: Feature Set Sweep

    Test all feature sets with best model from Phase 1:
    - Features: F1-F8 (all 8)
    - Models: Best from Phase 1 (default: M1)
    - Preprocessing: P1 (raw), P2 (smoothed)
    - Strategy: S2 (separate)

    16 experiments total.
    """
    print("\n" + "="*70)
    print("PHASE 2: Feature Set Sweep")
    print("="*70)

    # Get best model from Phase 1, or use default
    if best_model is None:
        best = get_best_config(phase=1)
        if best:
            best_model = best["model"]
        else:
            best_model = "M1"

    print(f"Using best model from Phase 1: {best_model}")

    feature_sets = list(FEATURE_SETS.keys())
    preprocessing_options = ["P1", "P2"]
    strategy = "S2"

    scalers = load_scalers()
    model_class, model_params = get_model_config(best_model)

    experiment_count = 0
    for feature_set in feature_sets:
        for preprocess in preprocessing_options:
            # For P2 (smoothed), extract with smoothing
            smooth = (preprocess == "P2")

            X, y, feature_names, meta = extract_features_for_dataset(
                feature_set, train=True, smooth=smooth, cache_dir=cache_dir
            )
            groups = meta["participant_id"].values

            experiment_id = f"P2_{feature_set}_{preprocess}"
            print(f"\nExperiment: {experiment_id} ({X.shape[1]} features)")

            start_time = time.time()
            results = run_cv_experiment(
                X, y, groups, model_class, model_params,
                preprocess if preprocess != "P2" else "P1",  # P2 already applied during extraction
                strategy, scalers
            )
            elapsed = time.time() - start_time

            print(f"  Scaled MSE: {results['scaled_mse_cv']:.6f} +/- {results['scaled_mse_std']:.6f}")
            print(f"  Time: {elapsed:.1f}s")

            log_result(
                phase=2,
                experiment_id=experiment_id,
                feature_set=feature_set,
                model=best_model,
                preprocessing=preprocess,
                strategy=strategy,
                n_features=X.shape[1],
                results=results,
            )

            np.save(OOF_DIR / f"{experiment_id}_oof.npy", results["oof_predictions"])
            experiment_count += 1

    print_results_summary(phase=2)
    return get_best_config(phase=2)


def run_phase_3(cache_dir: Path, best_feature: str = None, best_models: List[str] = None):
    """
    Phase 3: Preprocessing Sweep

    Test all preprocessing with best feature/model:
    - Features: Best from Phase 2
    - Models: Top 2 from Phase 1
    - Preprocessing: P1, P2, P3, P4 (all 4)
    - Strategy: S2 (separate)

    8 experiments total.
    """
    print("\n" + "="*70)
    print("PHASE 3: Preprocessing Sweep")
    print("="*70)

    # Get best feature set from Phase 2
    if best_feature is None:
        best = get_best_config(phase=2)
        if best:
            best_feature = best["feature_set"]
        else:
            best_feature = "F4"

    # Get top 2 models from Phase 1
    if best_models is None:
        if RESULTS_FILE.exists():
            df = pd.read_csv(RESULTS_FILE)
            p1 = df[df["phase"] == 1].sort_values("scaled_mse_cv")
            best_models = p1["model"].head(2).tolist()
        if not best_models:
            best_models = ["M1", "M3"]

    print(f"Using best feature set: {best_feature}")
    print(f"Using best models: {best_models}")

    preprocessing_options = list(PREPROCESSING_OPTIONS.keys())
    strategy = "S2"

    scalers = load_scalers()

    for preprocess in preprocessing_options:
        # For P2 (smoothed), extract with smoothing
        smooth = (preprocess == "P2")

        X, y, feature_names, meta = extract_features_for_dataset(
            best_feature, train=True, smooth=smooth, cache_dir=cache_dir
        )
        groups = meta["participant_id"].values

        for model_id in best_models:
            experiment_id = f"P3_{best_feature}_{model_id}_{preprocess}"
            print(f"\nExperiment: {experiment_id}")

            model_class, model_params = get_model_config(model_id)

            start_time = time.time()
            results = run_cv_experiment(
                X, y, groups, model_class, model_params,
                preprocess if preprocess != "P2" else "P1",
                strategy, scalers
            )
            elapsed = time.time() - start_time

            print(f"  Scaled MSE: {results['scaled_mse_cv']:.6f} +/- {results['scaled_mse_std']:.6f}")
            print(f"  Time: {elapsed:.1f}s")

            log_result(
                phase=3,
                experiment_id=experiment_id,
                feature_set=best_feature,
                model=model_id,
                preprocessing=preprocess,
                strategy=strategy,
                n_features=X.shape[1],
                results=results,
            )

            np.save(OOF_DIR / f"{experiment_id}_oof.npy", results["oof_predictions"])

    print_results_summary(phase=3)
    return get_best_config(phase=3)


def run_phase_4(cache_dir: Path, best_feature: str = None, best_models: List[str] = None, best_preprocess: str = None):
    """
    Phase 4: Training Strategy Sweep

    Test all training strategies:
    - Features: Best from Phase 2
    - Models: Top 2 from Phase 1
    - Preprocessing: Best from Phase 3
    - Strategy: S1, S2, S3, S4 (all 4)

    8 experiments total.
    """
    print("\n" + "="*70)
    print("PHASE 4: Training Strategy Sweep")
    print("="*70)

    # Get best configs from previous phases
    if best_feature is None:
        best = get_best_config(phase=2)
        best_feature = best["feature_set"] if best else "F4"

    if best_models is None:
        if RESULTS_FILE.exists():
            df = pd.read_csv(RESULTS_FILE)
            p1 = df[df["phase"] == 1].sort_values("scaled_mse_cv")
            best_models = p1["model"].head(2).tolist()
        if not best_models:
            best_models = ["M1"]

    if best_preprocess is None:
        best = get_best_config(phase=3)
        best_preprocess = best["preprocessing"] if best else "P1"

    print(f"Using: feature={best_feature}, models={best_models}, preprocess={best_preprocess}")

    strategies = list(TRAINING_STRATEGIES.keys())
    scalers = load_scalers()

    # Extract features once
    smooth = (best_preprocess == "P2")
    X, y, feature_names, meta = extract_features_for_dataset(
        best_feature, train=True, smooth=smooth, cache_dir=cache_dir
    )
    groups = meta["participant_id"].values

    for strategy in strategies:
        for model_id in best_models:
            experiment_id = f"P4_{best_feature}_{model_id}_{best_preprocess}_{strategy}"
            print(f"\nExperiment: {experiment_id}")

            model_class, model_params = get_model_config(model_id)

            start_time = time.time()
            results = run_cv_experiment(
                X, y, groups, model_class, model_params,
                best_preprocess if best_preprocess != "P2" else "P1",
                strategy, scalers
            )
            elapsed = time.time() - start_time

            print(f"  Scaled MSE: {results['scaled_mse_cv']:.6f} +/- {results['scaled_mse_std']:.6f}")
            print(f"  Time: {elapsed:.1f}s")

            log_result(
                phase=4,
                experiment_id=experiment_id,
                feature_set=best_feature,
                model=model_id,
                preprocessing=best_preprocess,
                strategy=strategy,
                n_features=X.shape[1],
                results=results,
            )

            np.save(OOF_DIR / f"{experiment_id}_oof.npy", results["oof_predictions"])

    print_results_summary(phase=4)
    return get_best_config(phase=4)


def run_phase_5(cache_dir: Path, n_trials: int = 100):
    """
    Phase 5: Hyperparameter Tuning with Optuna

    Tune top 3 configurations from previous phases.
    """
    print("\n" + "="*70)
    print("PHASE 5: Hyperparameter Tuning")
    print("="*70)

    try:
        import optuna
    except ImportError:
        print("ERROR: Optuna not installed. Run: pip install optuna")
        return

    # Get top 3 configs across all phases
    if not RESULTS_FILE.exists():
        print("No previous results. Run phases 1-4 first.")
        return

    df = pd.read_csv(RESULTS_FILE)
    top_configs = df.sort_values("scaled_mse_cv").head(3)

    scalers = load_scalers()

    for _, config in top_configs.iterrows():
        feature_set = config["feature_set"]
        model_id = config["model"]
        preprocess = config["preprocessing"]
        strategy = config["strategy"]

        print(f"\nTuning: {feature_set} + {model_id} + {preprocess} + {strategy}")

        # Extract features
        smooth = (preprocess == "P2")
        X, y, feature_names, meta = extract_features_for_dataset(
            feature_set, train=True, smooth=smooth, cache_dir=cache_dir
        )
        groups = meta["participant_id"].values

        model_class, base_params = get_model_config(model_id)

        def objective(trial):
            # Sample hyperparameters
            trial_params = sample_optuna_params(trial, model_id)
            merged_params = {**base_params, **trial_params}

            # Run CV
            results = run_cv_experiment(
                X, y, groups, model_class, merged_params,
                preprocess if preprocess != "P2" else "P1",
                strategy, scalers
            )
            return results["scaled_mse_cv"]

        # Create and run study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Log best result
        experiment_id = f"P5_{feature_set}_{model_id}_{preprocess}_{strategy}_tuned"
        print(f"\nBest trial: {study.best_value:.6f}")
        print(f"Best params: {study.best_params}")

        # Run CV with best params for full results
        best_params = {**base_params, **study.best_params}
        results = run_cv_experiment(
            X, y, groups, model_class, best_params,
            preprocess if preprocess != "P2" else "P1",
            strategy, scalers
        )

        log_result(
            phase=5,
            experiment_id=experiment_id,
            feature_set=feature_set,
            model=model_id,
            preprocessing=preprocess,
            strategy=strategy,
            n_features=X.shape[1],
            results=results,
            hyperparams=best_params
        )

        np.save(OOF_DIR / f"{experiment_id}_oof.npy", results["oof_predictions"])

    print_results_summary(phase=5)


def run_phase_6(cache_dir: Path):
    """
    Phase 6: Ensemble Construction

    Create ensembles from top single models:
    1. Simple average of top 3 models
    2. Weighted average (optimize weights)
    3. Stacking with Ridge meta-learner
    4. Stacking with LightGBM meta-learner
    """
    print("\n" + "="*70)
    print("PHASE 6: Ensemble Construction")
    print("="*70)

    if not RESULTS_FILE.exists():
        print("No previous results. Run phases 1-5 first.")
        return

    # Get top models
    df = pd.read_csv(RESULTS_FILE)
    top_configs = df.sort_values("scaled_mse_cv").head(5)

    scalers = load_scalers()
    meta = load_metadata(train=True)
    y = meta[TARGET_COLS].values

    # Load OOF predictions
    oof_preds = []
    oof_names = []

    for _, config in top_configs.iterrows():
        exp_id = config["experiment_id"]
        oof_file = OOF_DIR / f"{exp_id}_oof.npy"
        if oof_file.exists():
            oof = np.load(oof_file)
            oof_preds.append(oof)
            oof_names.append(exp_id)
            print(f"Loaded OOF: {exp_id}")

    if len(oof_preds) < 2:
        print("Need at least 2 OOF predictions for ensembling.")
        return

    oof_preds = np.array(oof_preds)  # (n_models, n_samples, 3)

    # --- Ensemble 1: Simple Average ---
    print("\n--- Ensemble 1: Simple Average ---")
    ensemble_avg = np.mean(oof_preds, axis=0)
    mse_avg, per_target_avg = compute_scaled_mse(y, ensemble_avg, scalers)
    print(f"Scaled MSE: {mse_avg:.6f}")

    log_result(
        phase=6,
        experiment_id="P6_ensemble_simple_avg",
        feature_set="ensemble",
        model="average",
        preprocessing="none",
        strategy="ensemble",
        n_features=len(oof_preds),
        results={
            "scaled_mse_cv": mse_avg,
            "scaled_mse_std": 0.0,
            "angle_mse_cv": per_target_avg["angle"],
            "depth_mse_cv": per_target_avg["depth"],
            "left_right_mse_cv": per_target_avg["left_right"],
        }
    )

    # --- Ensemble 2: Weighted Average ---
    print("\n--- Ensemble 2: Weighted Average ---")
    from scipy.optimize import minimize

    def weighted_mse(weights):
        w = np.array(weights)
        w = w / w.sum()
        pred = np.tensordot(w, oof_preds, axes=([0], [0]))
        mse, _ = compute_scaled_mse(y, pred, scalers)
        return mse

    n_models = len(oof_preds)
    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]

    result = minimize(weighted_mse, init_weights, method="SLSQP", bounds=bounds)
    best_weights = result.x / result.x.sum()

    ensemble_weighted = np.tensordot(best_weights, oof_preds, axes=([0], [0]))
    mse_weighted, per_target_weighted = compute_scaled_mse(y, ensemble_weighted, scalers)
    print(f"Scaled MSE: {mse_weighted:.6f}")
    print(f"Weights: {best_weights}")

    log_result(
        phase=6,
        experiment_id="P6_ensemble_weighted_avg",
        feature_set="ensemble",
        model="weighted_avg",
        preprocessing="none",
        strategy="ensemble",
        n_features=len(oof_preds),
        results={
            "scaled_mse_cv": mse_weighted,
            "scaled_mse_std": 0.0,
            "angle_mse_cv": per_target_weighted["angle"],
            "depth_mse_cv": per_target_weighted["depth"],
            "left_right_mse_cv": per_target_weighted["left_right"],
        },
        hyperparams={"weights": best_weights.tolist()}
    )

    # --- Ensemble 3: Stacking with Ridge ---
    print("\n--- Ensemble 3: Stacking with Ridge ---")
    from sklearn.linear_model import RidgeCV

    # Reshape OOF predictions for stacking
    # Stack all predictions as features: (n_samples, n_models * 3)
    stacked_features = oof_preds.transpose(1, 0, 2).reshape(y.shape[0], -1)

    # Train Ridge on stacked features
    ridge_stack = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    ridge_stack.fit(stacked_features, y)
    stacked_pred_ridge = ridge_stack.predict(stacked_features)

    mse_stack_ridge, per_target_stack_ridge = compute_scaled_mse(y, stacked_pred_ridge, scalers)
    print(f"Scaled MSE: {mse_stack_ridge:.6f}")

    log_result(
        phase=6,
        experiment_id="P6_ensemble_stack_ridge",
        feature_set="ensemble",
        model="stack_ridge",
        preprocessing="none",
        strategy="ensemble",
        n_features=stacked_features.shape[1],
        results={
            "scaled_mse_cv": mse_stack_ridge,
            "scaled_mse_std": 0.0,
            "angle_mse_cv": per_target_stack_ridge["angle"],
            "depth_mse_cv": per_target_stack_ridge["depth"],
            "left_right_mse_cv": per_target_stack_ridge["left_right"],
        }
    )

    # --- Ensemble 4: Stacking with LightGBM ---
    print("\n--- Ensemble 4: Stacking with LightGBM ---")
    try:
        import lightgbm as lgb
        from sklearn.multioutput import MultiOutputRegressor

        lgb_stack = MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=100, num_leaves=10, learning_rate=0.05,
            random_state=42, verbose=-1
        ))
        lgb_stack.fit(stacked_features, y)
        stacked_pred_lgb = lgb_stack.predict(stacked_features)

        mse_stack_lgb, per_target_stack_lgb = compute_scaled_mse(y, stacked_pred_lgb, scalers)
        print(f"Scaled MSE: {mse_stack_lgb:.6f}")

        log_result(
            phase=6,
            experiment_id="P6_ensemble_stack_lgb",
            feature_set="ensemble",
            model="stack_lgb",
            preprocessing="none",
            strategy="ensemble",
            n_features=stacked_features.shape[1],
            results={
                "scaled_mse_cv": mse_stack_lgb,
                "scaled_mse_std": 0.0,
                "angle_mse_cv": per_target_stack_lgb["angle"],
                "depth_mse_cv": per_target_stack_lgb["depth"],
                "left_right_mse_cv": per_target_stack_lgb["left_right"],
            }
        )
    except ImportError:
        print("LightGBM not available for stacking.")

    print_results_summary(phase=6)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exhaustive Grid Search for Minimum MSE")
    parser.add_argument("--all-phases", action="store_true", help="Run all phases")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], help="Run specific phase")
    parser.add_argument("--feature-set", type=str, help="Override feature set for Phase 2+")
    parser.add_argument("--model", type=str, help="Override model for Phase 2+")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials (Phase 5)")
    parser.add_argument("--cache-dir", type=str, default=str(OUTPUT_DIR / "feature_cache"),
                        help="Directory for caching extracted features")

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("EXHAUSTIVE GRID SEARCH FOR MINIMUM MSE")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cache directory: {cache_dir}")
    print(f"Results file: {RESULTS_FILE}")

    # Check available models
    available = get_available_models()
    print(f"\nAvailable models: {list(available.keys())}")

    if args.all_phases:
        # Run all phases sequentially
        print("\nRunning all phases...")

        best_p1 = run_phase_1(cache_dir)
        best_p2 = run_phase_2(cache_dir)
        best_p3 = run_phase_3(cache_dir)
        best_p4 = run_phase_4(cache_dir)
        run_phase_5(cache_dir, n_trials=args.n_trials)
        run_phase_6(cache_dir)

        # Final summary
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)

        if RESULTS_FILE.exists():
            df = pd.read_csv(RESULTS_FILE)
            best_overall = df.loc[df["scaled_mse_cv"].idxmin()]
            print(f"\nBest overall configuration:")
            print(f"  Experiment: {best_overall['experiment_id']}")
            print(f"  Scaled MSE: {best_overall['scaled_mse_cv']:.6f}")
            print(f"  Feature set: {best_overall['feature_set']}")
            print(f"  Model: {best_overall['model']}")
            print(f"  Preprocessing: {best_overall['preprocessing']}")
            print(f"  Strategy: {best_overall['strategy']}")

    elif args.phase:
        # Run specific phase
        if args.phase == 1:
            run_phase_1(cache_dir)
        elif args.phase == 2:
            run_phase_2(cache_dir, best_model=args.model)
        elif args.phase == 3:
            run_phase_3(cache_dir, best_feature=args.feature_set,
                       best_models=[args.model] if args.model else None)
        elif args.phase == 4:
            run_phase_4(cache_dir, best_feature=args.feature_set,
                       best_models=[args.model] if args.model else None)
        elif args.phase == 5:
            run_phase_5(cache_dir, n_trials=args.n_trials)
        elif args.phase == 6:
            run_phase_6(cache_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
