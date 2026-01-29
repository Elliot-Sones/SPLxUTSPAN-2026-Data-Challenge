#!/usr/bin/env python3
"""
Comprehensive Overnight Experiment System v2.

GOALS:
1. Maximum information extraction - understand WHAT works and WHY
2. Full reproducibility - save everything needed to recreate any result
3. Compare against known best (submission_3 = 0.0106 leaderboard)

Key insight: Best score (0.0106) came from per-player per-target models with tuned hyperparameters.
We must include this as a baseline and test whether new features improve it.

Experiments:
- Phase 1: Validation + known best baseline reproduction
- Phase 2: Feature engineering evaluation (which new features help?)
- Phase 3: Feature combinations with per-player per-target approach
- Phase 4: Deep learning experiments (attention, hybrid)
- Phase 5: Stacking/ensembles
- Phase 6: Per-target aggressive tuning with best features

All results include:
- Exact hyperparameters (JSON)
- Feature names (saved to file)
- OOF predictions (saved for post-hoc analysis)
- Feature importances (top 20)
- Reproducibility seed

Usage:
    ./scripts/run_overnight.sh           # Full run with sleep prevention
    uv run python scripts/overnight_experiments.py --validate  # Quick check
    uv run python scripts/overnight_experiments.py --phase 2   # Specific phase
"""

import argparse
import json
import numpy as np
import pandas as pd
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# Project setup
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import (
    load_all_as_arrays,
    load_scalers,
    NUM_FRAMES,
    NUM_FEATURES,
)
from src.feature_factory import get_feature_extractor, ensure_keypoint_mapping
from src.derivative_features import (
    extract_all_derivative_features,
    extract_velocity_features,
    extract_acceleration_features,
    extract_jerk_features,
    extract_critical_frame_features,
    extract_frame_difference_features,
)

# Configuration
OUTPUT_DIR = PROJECT_DIR / "output" / "overnight"
TARGETS = ["angle", "depth", "left_right"]
N_FOLDS = 5
SEED = 42


def setup_logging():
    """Setup logging to file."""
    import logging
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("overnight")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(OUTPUT_DIR / "overnight_search.log", mode='a')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    return logger


def setup_output_dirs():
    """Create all output directories."""
    for subdir in ["new_features", "oof_predictions", "feature_importances", "configs", "checkpoints"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load training data and scalers."""
    X_raw, y, meta = load_all_as_arrays(train=True)
    participant_ids = meta["participant_id"].values
    scalers = load_scalers()
    return X_raw, y, participant_ids, scalers


def extract_features_with_names(
    X_raw: np.ndarray,
    participant_ids: np.ndarray,
    feature_fn: Callable,
    cache_name: str,
) -> Tuple[np.ndarray, List[str]]:
    """Extract features and return both array and feature names."""
    cache_file = OUTPUT_DIR / "new_features" / f"{cache_name}.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['feature_names']

    feature_dicts = []
    for i in range(len(X_raw)):
        features = feature_fn(X_raw[i], participant_ids[i])
        feature_dicts.append(features)

    df = pd.DataFrame(feature_dicts)
    feature_names = list(df.columns)
    X = df.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    with open(cache_file, 'wb') as f:
        pickle.dump({'X': X, 'feature_names': feature_names}, f)

    return X, feature_names


# =============================================================================
# Model Training with Full Tracking
# =============================================================================

def get_lgbm_model(params: Dict = None):
    """Get LightGBM model with params."""
    import lightgbm as lgb
    default = {
        "n_estimators": 300, "num_leaves": 20, "learning_rate": 0.02,
        "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 1.0,
        "min_child_samples": 10, "verbose": -1, "n_jobs": -1, "random_state": SEED
    }
    if params:
        default.update(params)
    return lgb.LGBMRegressor(**default), default


def run_cv_with_tracking(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    scalers: Dict,
    model_fn: Callable,
    per_player: bool = False,
    per_target: bool = True,
    n_folds: int = N_FOLDS,
) -> Dict[str, Any]:
    """
    Run CV with full tracking of OOF predictions and feature importances.

    Args:
        X: Features
        y: Targets (n_samples, 3)
        groups: Participant IDs
        scalers: Target scalers
        model_fn: Function returning (model, params_dict)
        per_player: Train separate model per player
        per_target: Train separate model per target

    Returns:
        Dict with mse_per_target, total_mse, oof_predictions, feature_importances, params
    """
    gkf = GroupKFold(n_splits=n_folds)
    oof_predictions = np.zeros_like(y)
    feature_importances = {target: [] for target in TARGETS}
    all_params = {}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]
        groups_val = groups[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for target_idx, target in enumerate(TARGETS):
            y_target_train = y_train[:, target_idx]

            if per_player:
                # Per-player models
                player_preds = np.zeros(len(val_idx))
                player_importances = []

                for pid in np.unique(groups_train):
                    mask_train = groups_train == pid
                    model, params = model_fn()
                    model.fit(X_train_scaled[mask_train], y_target_train[mask_train])

                    if hasattr(model, 'feature_importances_'):
                        player_importances.append(model.feature_importances_)

                # Fallback model for held-out player
                fallback_model, params = model_fn()
                fallback_model.fit(X_train_scaled, y_target_train)
                all_params[target] = params

                if hasattr(fallback_model, 'feature_importances_'):
                    player_importances.append(fallback_model.feature_importances_)

                # Predict
                for pid in np.unique(groups_val):
                    mask_val = groups_val == pid
                    player_preds[mask_val] = fallback_model.predict(X_val_scaled[mask_val])

                oof_predictions[val_idx, target_idx] = player_preds

                if player_importances:
                    feature_importances[target].append(np.mean(player_importances, axis=0))
            else:
                # Single model
                model, params = model_fn()
                model.fit(X_train_scaled, y_target_train)
                all_params[target] = params

                oof_predictions[val_idx, target_idx] = model.predict(X_val_scaled)

                if hasattr(model, 'feature_importances_'):
                    feature_importances[target].append(model.feature_importances_)

    # Compute scaled MSE
    mse_per_target = {}
    for i, target in enumerate(TARGETS):
        y_true = y[:, i]
        y_pred = oof_predictions[:, i]
        y_true_scaled = scalers[target].transform(y_true.reshape(-1, 1)).ravel()
        y_pred_scaled = scalers[target].transform(y_pred.reshape(-1, 1)).ravel()
        mse_per_target[target] = float(mean_squared_error(y_true_scaled, y_pred_scaled))

    # Average feature importances
    avg_importances = {}
    for target in TARGETS:
        if feature_importances[target]:
            avg_importances[target] = np.mean(feature_importances[target], axis=0)

    return {
        "mse_per_target": mse_per_target,
        "total_mse": float(np.mean(list(mse_per_target.values()))),
        "oof_predictions": oof_predictions,
        "feature_importances": avg_importances,
        "params": all_params,
    }


def tune_per_target_params(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    scalers: Dict,
    target: str,
    target_idx: int,
    n_trials: int = 30,
    per_player: bool = True,
) -> Tuple[Dict, float]:
    """Tune hyperparameters for a specific target."""
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "verbose": -1, "n_jobs": -1, "random_state": SEED,
        }

        gkf = GroupKFold(n_splits=N_FOLDS)
        oof = np.zeros(len(y))

        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx, target_idx]
            groups_train = groups[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            if per_player:
                # Train on all, predict for held-out player
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train_scaled, y_train)
                oof[val_idx] = model.predict(X_val_scaled)
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train_scaled, y_train)
                oof[val_idx] = model.predict(X_val_scaled)

        y_true_scaled = scalers[target].transform(y[:, target_idx].reshape(-1, 1)).ravel()
        y_pred_scaled = scalers[target].transform(oof.reshape(-1, 1)).ravel()
        return mean_squared_error(y_true_scaled, y_pred_scaled)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"verbose": -1, "n_jobs": -1, "random_state": SEED})
    return best_params, study.best_value


# =============================================================================
# Result Tracking
# =============================================================================

class ExperimentTracker:
    """Track all experiments with full reproducibility info."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.results_file = output_dir / "overnight_results.csv"
        self.checkpoint_file = output_dir / "checkpoints" / "completed.json"
        self.completed = self._load_completed()

    def _load_completed(self) -> set:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return set(json.load(f).get("completed", []))
        return set()

    def _save_completed(self):
        self.checkpoint_file.parent.mkdir(exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump({"completed": list(self.completed)}, f)

    def is_completed(self, name: str) -> bool:
        return name in self.completed

    def save_result(
        self,
        name: str,
        phase: str,
        model_type: str,
        feature_set: str,
        feature_names: List[str],
        n_features: int,
        mse_per_target: Dict[str, float],
        total_mse: float,
        oof_predictions: np.ndarray,
        feature_importances: Dict[str, np.ndarray],
        params: Dict,
        per_player: bool,
        per_target: bool,
        training_time: float,
        notes: str = "",
    ):
        """Save complete experiment result."""
        timestamp = datetime.now().isoformat()

        # Save OOF predictions
        oof_file = self.output_dir / "oof_predictions" / f"{name}.npy"
        np.save(oof_file, oof_predictions)

        # Save feature importances with names
        if feature_importances:
            imp_file = self.output_dir / "feature_importances" / f"{name}.json"
            imp_data = {}
            for target, imp in feature_importances.items():
                if imp is not None and len(imp) == len(feature_names):
                    top_idx = np.argsort(imp)[::-1][:20]
                    imp_data[target] = {feature_names[i]: float(imp[i]) for i in top_idx}
            with open(imp_file, 'w') as f:
                json.dump(imp_data, f, indent=2)

        # Save config
        config = {
            "name": name,
            "timestamp": timestamp,
            "phase": phase,
            "model_type": model_type,
            "feature_set": feature_set,
            "feature_names": feature_names,
            "n_features": n_features,
            "params": params,
            "per_player": per_player,
            "per_target": per_target,
            "seed": SEED,
            "n_folds": N_FOLDS,
        }
        config_file = self.output_dir / "configs" / f"{name}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Append to results CSV
        row = {
            "experiment_name": name,
            "timestamp": timestamp,
            "phase": phase,
            "model_type": model_type,
            "feature_set": feature_set,
            "n_features": n_features,
            "angle_mse": mse_per_target.get("angle", np.nan),
            "depth_mse": mse_per_target.get("depth", np.nan),
            "left_right_mse": mse_per_target.get("left_right", np.nan),
            "total_scaled_mse": total_mse,
            "per_player": per_player,
            "per_target": per_target,
            "training_time_seconds": round(training_time, 1),
            "notes": notes,
            "config_file": str(config_file.name),
            "oof_file": str(oof_file.name),
        }

        df = pd.DataFrame([row])
        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

        self.completed.add(name)
        self._save_completed()


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiment(
    tracker: ExperimentTracker,
    logger,
    name: str,
    phase: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    scalers: Dict,
    feature_names: List[str],
    feature_set: str,
    model_fn: Callable,
    model_type: str,
    per_player: bool,
    per_target: bool,
    notes: str = "",
):
    """Run a single experiment with full tracking."""
    if tracker.is_completed(name):
        logger.info(f"SKIP (completed): {name}")
        return None

    logger.info(f"START: {name}")
    start_time = time.time()

    try:
        results = run_cv_with_tracking(
            X, y, groups, scalers, model_fn,
            per_player=per_player, per_target=per_target
        )

        training_time = time.time() - start_time

        tracker.save_result(
            name=name,
            phase=phase,
            model_type=model_type,
            feature_set=feature_set,
            feature_names=feature_names,
            n_features=X.shape[1],
            mse_per_target=results["mse_per_target"],
            total_mse=results["total_mse"],
            oof_predictions=results["oof_predictions"],
            feature_importances=results["feature_importances"],
            params=results["params"],
            per_player=per_player,
            per_target=per_target,
            training_time=training_time,
            notes=notes,
        )

        logger.info(
            f"DONE: {name} - Total: {results['total_mse']:.6f} "
            f"(A:{results['mse_per_target']['angle']:.4f} "
            f"D:{results['mse_per_target']['depth']:.4f} "
            f"LR:{results['mse_per_target']['left_right']:.4f}) "
            f"[{training_time:.1f}s]"
        )

        return results

    except Exception as e:
        logger.error(f"FAIL: {name} - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# PHASES
# =============================================================================

def phase1_baseline(tracker, logger, X_raw, y, groups, scalers):
    """Phase 1: Reproduce known best baseline for comparison."""
    logger.info("=" * 70)
    logger.info("PHASE 1: BASELINE REPRODUCTION")
    logger.info("=" * 70)

    # Extract F4 features (what submission_3 used)
    ensure_keypoint_mapping()
    extractor = get_feature_extractor("F4")
    X_f4, f4_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extractor(ts, pid, smooth=False),
        "base_f4"
    )

    # 1a. Simple global model (baseline)
    run_experiment(
        tracker, logger,
        name="baseline_global_f4",
        phase="baseline",
        X=X_f4, y=y, groups=groups, scalers=scalers,
        feature_names=f4_names,
        feature_set="F4_hybrid_pid",
        model_fn=get_lgbm_model,
        model_type="LightGBM",
        per_player=False,
        per_target=True,
        notes="Global model, F4 features - baseline comparison"
    )

    # 1b. Per-player model (submission_3 approach)
    run_experiment(
        tracker, logger,
        name="baseline_per_player_f4",
        phase="baseline",
        X=X_f4, y=y, groups=groups, scalers=scalers,
        feature_names=f4_names,
        feature_set="F4_hybrid_pid",
        model_fn=get_lgbm_model,
        model_type="LightGBM",
        per_player=True,
        per_target=True,
        notes="Per-player model, F4 features - this is submission_3 approach"
    )


def phase2_feature_evaluation(tracker, logger, X_raw, y, groups, scalers):
    """Phase 2: Evaluate each new feature type independently."""
    logger.info("=" * 70)
    logger.info("PHASE 2: FEATURE EVALUATION")
    logger.info("=" * 70)

    feature_configs = [
        ("velocity", lambda ts, pid: extract_velocity_features(ts)),
        ("acceleration", lambda ts, pid: extract_acceleration_features(ts)),
        ("jerk", lambda ts, pid: extract_jerk_features(ts)),
        ("critical_frames", lambda ts, pid: extract_critical_frame_features(ts)),
        ("frame_diffs", lambda ts, pid: extract_frame_difference_features(ts)),
        ("all_derivatives", lambda ts, pid: extract_all_derivative_features(ts, pid)),
    ]

    for feat_name, feat_fn in feature_configs:
        X_feat, feat_names = extract_features_with_names(X_raw, groups, feat_fn, feat_name)

        # Test with global model first (faster)
        run_experiment(
            tracker, logger,
            name=f"feat_eval_{feat_name}_global",
            phase="feature_eval",
            X=X_feat, y=y, groups=groups, scalers=scalers,
            feature_names=feat_names,
            feature_set=feat_name,
            model_fn=get_lgbm_model,
            model_type="LightGBM",
            per_player=False,
            per_target=True,
            notes=f"Evaluating {feat_name} features alone with global model"
        )

        # Then with per-player model
        run_experiment(
            tracker, logger,
            name=f"feat_eval_{feat_name}_per_player",
            phase="feature_eval",
            X=X_feat, y=y, groups=groups, scalers=scalers,
            feature_names=feat_names,
            feature_set=feat_name,
            model_fn=get_lgbm_model,
            model_type="LightGBM",
            per_player=True,
            per_target=True,
            notes=f"Evaluating {feat_name} features alone with per-player model"
        )


def phase3_feature_combinations(tracker, logger, X_raw, y, groups, scalers):
    """Phase 3: Test feature combinations with per-player approach."""
    logger.info("=" * 70)
    logger.info("PHASE 3: FEATURE COMBINATIONS")
    logger.info("=" * 70)

    # Load base features
    ensure_keypoint_mapping()
    extractor = get_feature_extractor("F4")
    X_f4, f4_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extractor(ts, pid, smooth=False),
        "base_f4"
    )

    # Load derivative features
    X_vel, vel_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extract_velocity_features(ts),
        "velocity"
    )
    X_acc, acc_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extract_acceleration_features(ts),
        "acceleration"
    )
    X_jerk, jerk_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extract_jerk_features(ts),
        "jerk"
    )
    X_crit, crit_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extract_critical_frame_features(ts),
        "critical_frames"
    )

    # Test combinations
    combos = [
        ("f4_plus_velocity", np.hstack([X_f4, X_vel]), f4_names + vel_names),
        ("f4_plus_critical", np.hstack([X_f4, X_crit]), f4_names + crit_names),
        ("f4_plus_all_deriv", np.hstack([X_f4, X_vel, X_acc, X_jerk]), f4_names + vel_names + acc_names + jerk_names),
        ("f4_plus_crit_vel", np.hstack([X_f4, X_crit, X_vel]), f4_names + crit_names + vel_names),
        ("all_features", np.hstack([X_f4, X_vel, X_acc, X_jerk, X_crit]), f4_names + vel_names + acc_names + jerk_names + crit_names),
    ]

    for combo_name, X_combo, combo_names in combos:
        run_experiment(
            tracker, logger,
            name=f"combo_{combo_name}_per_player",
            phase="feature_combo",
            X=X_combo, y=y, groups=groups, scalers=scalers,
            feature_names=combo_names,
            feature_set=combo_name,
            model_fn=get_lgbm_model,
            model_type="LightGBM",
            per_player=True,
            per_target=True,
            notes=f"Feature combination: {combo_name}"
        )


def phase4_dl_experiments(tracker, logger, X_raw, y, groups, scalers):
    """Phase 4: Deep learning experiments."""
    logger.info("=" * 70)
    logger.info("PHASE 4: DEEP LEARNING")
    logger.info("=" * 70)

    from src.creative_dl_models import (
        FrameAttentionModel, SimpleMLP, ConvAutoencoder, get_device
    )

    device = get_device()
    logger.info(f"DL Device: {device}")

    # 4a. Frame Attention Model
    if not tracker.is_completed("dl_frame_attention"):
        logger.info("START: dl_frame_attention")
        start_time = time.time()

        try:
            X_tensor = torch.tensor(X_raw, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            gkf = GroupKFold(n_splits=N_FOLDS)
            oof = np.zeros_like(y)

            for train_idx, val_idx in gkf.split(X_raw, y, groups):
                X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
                X_val = X_tensor[val_idx]

                dataset = torch.utils.data.TensorDataset(X_train, y_train)
                loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

                model = FrameAttentionModel(n_features=NUM_FEATURES, hidden_dim=64, n_targets=3)
                model = model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

                model.train()
                for epoch in range(50):
                    for bx, by in loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        pred, _ = model(bx)
                        loss = torch.nn.functional.mse_loss(pred, by)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    pred, attn = model(X_val.to(device), return_attention=True)
                    oof[val_idx] = pred.cpu().numpy()

            mse_per_target = {}
            for i, target in enumerate(TARGETS):
                y_true_scaled = scalers[target].transform(y[:, i].reshape(-1, 1)).ravel()
                y_pred_scaled = scalers[target].transform(oof[:, i].reshape(-1, 1)).ravel()
                mse_per_target[target] = float(mean_squared_error(y_true_scaled, y_pred_scaled))

            total_mse = float(np.mean(list(mse_per_target.values())))
            training_time = time.time() - start_time

            tracker.save_result(
                name="dl_frame_attention",
                phase="dl",
                model_type="FrameAttentionModel",
                feature_set="raw_240x207",
                feature_names=[f"frame_{f}_feat_{i}" for f in range(240) for i in range(207)],
                n_features=240*207,
                mse_per_target=mse_per_target,
                total_mse=total_mse,
                oof_predictions=oof,
                feature_importances={},
                params={"hidden_dim": 64, "epochs": 50, "lr": 1e-3},
                per_player=False,
                per_target=False,
                training_time=training_time,
                notes="Temporal attention model - learns frame importance"
            )

            logger.info(f"DONE: dl_frame_attention - {total_mse:.6f} [{training_time:.1f}s]")

        except Exception as e:
            logger.error(f"FAIL: dl_frame_attention - {e}")

    # 4b. Hybrid Autoencoder + GBDT
    if not tracker.is_completed("dl_hybrid_autoencoder_gbdt"):
        logger.info("START: dl_hybrid_autoencoder_gbdt")
        start_time = time.time()

        try:
            X_tensor = torch.tensor(X_raw, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(X_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

            ae = ConvAutoencoder(n_features=NUM_FEATURES, bottleneck_dim=32)
            ae = ae.to(device)
            optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-3)

            ae.train()
            for epoch in range(30):
                for batch in loader:
                    x = batch[0].to(device)
                    optimizer.zero_grad()
                    recon, _ = ae(x)
                    loss = torch.nn.functional.mse_loss(recon, x)
                    loss.backward()
                    optimizer.step()

            ae.eval()
            with torch.no_grad():
                bottleneck = ae.encode(X_tensor.to(device)).cpu().numpy()

            # Combine with F4
            ensure_keypoint_mapping()
            extractor = get_feature_extractor("F4")
            X_f4, f4_names = extract_features_with_names(
                X_raw, groups,
                lambda ts, pid: extractor(ts, pid, smooth=False),
                "base_f4"
            )

            X_hybrid = np.hstack([X_f4, bottleneck])
            hybrid_names = f4_names + [f"bottleneck_{i}" for i in range(32)]

            run_experiment(
                tracker, logger,
                name="dl_hybrid_autoencoder_gbdt",
                phase="dl",
                X=X_hybrid, y=y, groups=groups, scalers=scalers,
                feature_names=hybrid_names,
                feature_set="f4_plus_autoencoder_bottleneck",
                model_fn=get_lgbm_model,
                model_type="ConvAutoencoder+LightGBM",
                per_player=True,
                per_target=True,
                notes="Autoencoder bottleneck features combined with F4"
            )

        except Exception as e:
            logger.error(f"FAIL: dl_hybrid_autoencoder_gbdt - {e}")


def phase5_stacking(tracker, logger, X_raw, y, groups, scalers):
    """Phase 5: Stacking ensembles."""
    logger.info("=" * 70)
    logger.info("PHASE 5: STACKING ENSEMBLES")
    logger.info("=" * 70)

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.base import clone

    # Use best feature set
    ensure_keypoint_mapping()
    extractor = get_feature_extractor("F4")
    X_f4, f4_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extractor(ts, pid, smooth=False),
        "base_f4"
    )
    X_crit, crit_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extract_critical_frame_features(ts),
        "critical_frames"
    )

    X = np.hstack([X_f4, X_crit])
    feature_names = f4_names + crit_names

    # Simple average ensemble
    if not tracker.is_completed("ensemble_simple_average"):
        logger.info("START: ensemble_simple_average")
        start_time = time.time()

        models = {
            "lgbm": lgb.LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.02, verbose=-1, n_jobs=-1),
            "xgb": xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.02, verbosity=0, n_jobs=-1),
            "catboost": CatBoostRegressor(iterations=300, depth=6, learning_rate=0.03, verbose=False),
        }

        gkf = GroupKFold(n_splits=N_FOLDS)
        oof_all = {name: np.zeros_like(y) for name in models}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for train_idx, val_idx in gkf.split(X_scaled, y, groups):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train = y[train_idx]

            for model_name, model_template in models.items():
                for target_idx in range(3):
                    model = clone(model_template)
                    model.fit(X_train, y_train[:, target_idx])
                    oof_all[model_name][val_idx, target_idx] = model.predict(X_val)

        oof_avg = np.mean([oof_all[n] for n in models], axis=0)

        mse_per_target = {}
        for i, target in enumerate(TARGETS):
            y_true_scaled = scalers[target].transform(y[:, i].reshape(-1, 1)).ravel()
            y_pred_scaled = scalers[target].transform(oof_avg[:, i].reshape(-1, 1)).ravel()
            mse_per_target[target] = float(mean_squared_error(y_true_scaled, y_pred_scaled))

        training_time = time.time() - start_time

        tracker.save_result(
            name="ensemble_simple_average",
            phase="stacking",
            model_type="Avg(LightGBM,XGBoost,CatBoost)",
            feature_set="f4_plus_critical",
            feature_names=feature_names,
            n_features=X.shape[1],
            mse_per_target=mse_per_target,
            total_mse=float(np.mean(list(mse_per_target.values()))),
            oof_predictions=oof_avg,
            feature_importances={},
            params={"method": "simple_average"},
            per_player=False,
            per_target=True,
            training_time=training_time,
            notes="Simple average of 3 GBDT models"
        )

        logger.info(f"DONE: ensemble_simple_average - {np.mean(list(mse_per_target.values())):.6f} [{training_time:.1f}s]")


def phase6_per_target_tuning(tracker, logger, X_raw, y, groups, scalers):
    """Phase 6: Aggressive per-target hyperparameter tuning."""
    logger.info("=" * 70)
    logger.info("PHASE 6: PER-TARGET TUNING (50 trials each)")
    logger.info("=" * 70)

    # Use best feature combination
    ensure_keypoint_mapping()
    extractor = get_feature_extractor("F4")
    X_f4, f4_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extractor(ts, pid, smooth=False),
        "base_f4"
    )
    X_crit, crit_names = extract_features_with_names(
        X_raw, groups,
        lambda ts, pid: extract_critical_frame_features(ts),
        "critical_frames"
    )

    X = np.hstack([X_f4, X_crit])
    feature_names = f4_names + crit_names

    tuned_params = {}
    tuned_mses = {}

    for target_idx, target in enumerate(TARGETS):
        exp_name = f"tuned_50trials_{target}"
        if tracker.is_completed(exp_name):
            logger.info(f"SKIP (completed): {exp_name}")
            continue

        logger.info(f"START: {exp_name}")
        start_time = time.time()

        best_params, best_mse = tune_per_target_params(
            X, y, groups, scalers,
            target=target,
            target_idx=target_idx,
            n_trials=50,
            per_player=True,
        )

        tuned_params[target] = best_params
        tuned_mses[target] = best_mse

        training_time = time.time() - start_time
        logger.info(f"DONE: {exp_name} - {best_mse:.6f} [{training_time:.1f}s]")
        logger.info(f"  Best params: n_est={best_params.get('n_estimators')}, lr={best_params.get('learning_rate'):.4f}")

        # Save individual result
        config_file = OUTPUT_DIR / "configs" / f"{exp_name}.json"
        with open(config_file, 'w') as f:
            json.dump({
                "target": target,
                "best_mse": best_mse,
                "best_params": best_params,
                "n_trials": 50,
                "feature_set": "f4_plus_critical",
            }, f, indent=2)

        tracker.completed.add(exp_name)
        tracker._save_completed()

    # Final combined evaluation with tuned params
    if tuned_params and not tracker.is_completed("tuned_combined_final"):
        logger.info("Evaluating combined tuned model...")

        gkf = GroupKFold(n_splits=N_FOLDS)
        oof = np.zeros_like(y)

        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            import lightgbm as lgb
            for target_idx, target in enumerate(TARGETS):
                if target in tuned_params:
                    model = lgb.LGBMRegressor(**tuned_params[target])
                    model.fit(X_train_scaled, y_train[:, target_idx])
                    oof[val_idx, target_idx] = model.predict(X_val_scaled)

        mse_per_target = {}
        for i, target in enumerate(TARGETS):
            y_true_scaled = scalers[target].transform(y[:, i].reshape(-1, 1)).ravel()
            y_pred_scaled = scalers[target].transform(oof[:, i].reshape(-1, 1)).ravel()
            mse_per_target[target] = float(mean_squared_error(y_true_scaled, y_pred_scaled))

        tracker.save_result(
            name="tuned_combined_final",
            phase="tuning",
            model_type="LightGBM_per_target_tuned",
            feature_set="f4_plus_critical",
            feature_names=feature_names,
            n_features=X.shape[1],
            mse_per_target=mse_per_target,
            total_mse=float(np.mean(list(mse_per_target.values()))),
            oof_predictions=oof,
            feature_importances={},
            params=tuned_params,
            per_player=False,
            per_target=True,
            training_time=0,
            notes="Combined result with per-target tuned hyperparameters (50 trials each)"
        )

        logger.info(f"FINAL TUNED: {np.mean(list(mse_per_target.values())):.6f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight Experiment System v2")
    parser.add_argument("--validate", action="store_true", help="Quick validation only")
    parser.add_argument("--phase", type=int, help="Run specific phase (1-6)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (default behavior)")
    args = parser.parse_args()

    setup_output_dirs()
    logger = setup_logging()
    tracker = ExperimentTracker(OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("OVERNIGHT EXPERIMENT SYSTEM v2")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Seed: {SEED}, Folds: {N_FOLDS}")
    logger.info("=" * 70)

    logger.info("Loading data...")
    X_raw, y, groups, scalers = load_data()
    logger.info(f"Data: {X_raw.shape[0]} samples, {X_raw.shape[1]} frames x {X_raw.shape[2]} features")
    logger.info(f"Participants: {np.unique(groups)}")

    phases = {
        1: lambda: phase1_baseline(tracker, logger, X_raw, y, groups, scalers),
        2: lambda: phase2_feature_evaluation(tracker, logger, X_raw, y, groups, scalers),
        3: lambda: phase3_feature_combinations(tracker, logger, X_raw, y, groups, scalers),
        4: lambda: phase4_dl_experiments(tracker, logger, X_raw, y, groups, scalers),
        5: lambda: phase5_stacking(tracker, logger, X_raw, y, groups, scalers),
        6: lambda: phase6_per_target_tuning(tracker, logger, X_raw, y, groups, scalers),
    }

    if args.validate:
        # Just run phase 1 baseline
        phases[1]()
    elif args.phase:
        if args.phase in phases:
            phases[args.phase]()
        else:
            logger.error(f"Invalid phase: {args.phase}")
    else:
        # Run all phases
        for phase_num in sorted(phases.keys()):
            phases[phase_num]()

    # Final summary
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Finished: {datetime.now().isoformat()}")

    if tracker.results_file.exists():
        df = pd.read_csv(tracker.results_file)
        valid = df[df["total_scaled_mse"].notna()].copy()
        if not valid.empty:
            valid = valid.sort_values("total_scaled_mse")
            logger.info(f"\nTOP 5 RESULTS:")
            for _, row in valid.head(5).iterrows():
                logger.info(
                    f"  {row['experiment_name']}: {row['total_scaled_mse']:.6f} "
                    f"(A:{row['angle_mse']:.4f} D:{row['depth_mse']:.4f} LR:{row['left_right_mse']:.4f})"
                )

    logger.info(f"\nResults: {tracker.results_file}")
    logger.info(f"Configs: {OUTPUT_DIR / 'configs'}")
    logger.info(f"OOF predictions: {OUTPUT_DIR / 'oof_predictions'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
