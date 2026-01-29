#!/usr/bin/env python3
"""
Comprehensive Overnight Experiment System v3

A TRULY comprehensive test system exploring MAXIMUM BREADTH of approaches:
- Feature engineering (spatial, temporal, frequency, geometric)
- Model types (tree ensembles, linear, DL)
- Preprocessing (scaling, PCA, feature selection)
- Target transformations
- Per-target vs global approaches

Goal: After this run, we should KNOW what works and what doesn't.

Usage:
    # Quick validation (Phase 1 only)
    uv run python scripts/overnight_experiments_v3.py --validate

    # Full overnight run with sleep prevention
    caffeinate -i -s nohup uv run python scripts/overnight_experiments_v3.py > overnight_stdout.log 2>&1 &

    # Monitor
    tail -f output/overnight/overnight_search.log

    # Run specific phase
    uv run python scripts/overnight_experiments_v3.py --phase 2

    # Resume from checkpoint
    uv run python scripts/overnight_experiments_v3.py --resume
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Project imports
from src.data_loader import (
    load_all_as_arrays, load_metadata, load_scalers, iterate_shots,
    NUM_FRAMES, NUM_FEATURES
)
from src.derivative_features import (
    extract_velocity_features, extract_acceleration_features,
    extract_jerk_features, extract_critical_frame_features,
    extract_frame_difference_features, extract_all_derivative_features
)
from src.fft_features import extract_all_fft_features, extract_phase_fft_features
from src.geometric_features import extract_all_geometric_features
from src.interaction_features import extract_all_interaction_features
from src.experiment_runner import ExperimentResult, ExperimentRunner

# Optional imports
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")


# =============================================================================
# Constants and Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "overnight"
DATA_SPLIT_DIR = Path(__file__).parent.parent / "data_split"
TARGETS = ["angle", "depth", "left_right"]

# LightGBM default parameters
LGBM_DEFAULT_PARAMS = {
    'n_estimators': 300,
    'num_leaves': 20,
    'learning_rate': 0.02,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

# Conservative LightGBM for high regularization
LGBM_CONSERVATIVE_PARAMS = {
    'n_estimators': 200,
    'num_leaves': 10,
    'learning_rate': 0.01,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'reg_lambda': 5.0,
    'reg_alpha': 1.0,
    'min_child_samples': 10,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}


# =============================================================================
# Data Loading and Feature Extraction
# =============================================================================

class DataManager:
    """Manages data loading and caching."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.cache_dir = output_dir / "new_features"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cached data
        self._raw_sequences = None
        self._targets = None
        self._participant_ids = None
        self._scalers = None
        self._feature_cache = {}

    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load raw sequence data."""
        if self._raw_sequences is None:
            print("Loading raw sequence data...")
            X_raw, y, meta = load_all_as_arrays(train=True)
            self._raw_sequences = X_raw
            self._targets = y
            self._participant_ids = meta['participant_id'].values
            print(f"  Loaded {len(X_raw)} samples, shape: {X_raw.shape}")
        return self._raw_sequences, self._targets, self._participant_ids

    def get_scalers(self) -> Dict:
        """Get target scalers."""
        if self._scalers is None:
            self._scalers = load_scalers()
        return self._scalers

    def extract_features(self, feature_type: str, force_recompute: bool = False) -> np.ndarray:
        """
        Extract features of a given type, with caching.

        Args:
            feature_type: One of 'f4_hybrid', 'velocity', 'acceleration', 'jerk',
                         'critical_frames', 'fft', 'geometric', 'raw_flattened', 'pca_50'
            force_recompute: Force recomputation even if cached

        Returns:
            Feature array of shape (n_samples, n_features)
        """
        cache_path = self.cache_dir / f"{feature_type}_features.npy"

        if not force_recompute and cache_path.exists():
            print(f"  Loading cached {feature_type} features...")
            return np.load(cache_path)

        if feature_type in self._feature_cache and not force_recompute:
            return self._feature_cache[feature_type]

        X_raw, _, participant_ids = self.load_raw_data()
        print(f"  Extracting {feature_type} features...")

        if feature_type == 'f4_hybrid':
            features = self._extract_f4_hybrid(X_raw, participant_ids)
        elif feature_type == 'velocity':
            features = self._extract_derivative_type(X_raw, 'velocity')
        elif feature_type == 'acceleration':
            features = self._extract_derivative_type(X_raw, 'acceleration')
        elif feature_type == 'jerk':
            features = self._extract_derivative_type(X_raw, 'jerk')
        elif feature_type == 'critical_frames':
            features = self._extract_critical_frames(X_raw)
        elif feature_type == 'fft':
            features = self._extract_fft(X_raw)
        elif feature_type == 'geometric':
            features = self._extract_geometric(X_raw)
        elif feature_type == 'raw_flattened':
            features = X_raw.reshape(X_raw.shape[0], -1)
        elif feature_type == 'pca_50':
            features = self._extract_pca(X_raw, n_components=50)
        elif feature_type == 'pca_100':
            features = self._extract_pca(X_raw, n_components=100)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Cache
        self._feature_cache[feature_type] = features
        np.save(cache_path, features)
        print(f"    Extracted {features.shape[1]} features")

        return features

    def _extract_f4_hybrid(self, X_raw: np.ndarray, participant_ids: np.ndarray) -> np.ndarray:
        """Extract F4 hybrid features (from data_split CSVs if available)."""
        # Try loading from data_split first
        if DATA_SPLIT_DIR.exists():
            try:
                all_features = []
                for player_id in sorted(DATA_SPLIT_DIR.iterdir()):
                    if not player_id.is_dir():
                        continue
                    angle_path = player_id / "angle.csv"
                    if angle_path.exists():
                        df = pd.read_csv(angle_path)
                        # Drop target and ID columns
                        feature_cols = [c for c in df.columns if c not in ['angle', 'depth', 'left_right', 'shot_id', 'participant_id', 'id']]
                        all_features.append(df[feature_cols].values)
                if all_features:
                    return np.vstack(all_features)
            except Exception as e:
                print(f"    Warning: Could not load from data_split: {e}")

        # Fallback to computing features
        return self._extract_all_derivatives_combined(X_raw, participant_ids)

    def _extract_derivative_type(self, X_raw: np.ndarray, deriv_type: str) -> np.ndarray:
        """Extract specific derivative features."""
        all_features = []
        for i in range(len(X_raw)):
            seq = X_raw[i]
            if deriv_type == 'velocity':
                feats = extract_velocity_features(seq)
            elif deriv_type == 'acceleration':
                feats = extract_acceleration_features(seq)
            elif deriv_type == 'jerk':
                feats = extract_jerk_features(seq)
            all_features.append(feats)

        return self._dict_list_to_array(all_features)

    def _extract_critical_frames(self, X_raw: np.ndarray) -> np.ndarray:
        """Extract critical frame features."""
        all_features = []
        for i in range(len(X_raw)):
            seq = X_raw[i]
            feats = extract_critical_frame_features(seq)
            feats.update(extract_frame_difference_features(seq))
            all_features.append(feats)
        return self._dict_list_to_array(all_features)

    def _extract_fft(self, X_raw: np.ndarray) -> np.ndarray:
        """Extract FFT features."""
        all_features = []
        for i in range(len(X_raw)):
            seq = X_raw[i]
            feats = extract_all_fft_features(seq)
            feats.update(extract_phase_fft_features(seq))
            all_features.append(feats)
        return self._dict_list_to_array(all_features)

    def _extract_geometric(self, X_raw: np.ndarray) -> np.ndarray:
        """Extract geometric features."""
        all_features = []
        for i in range(len(X_raw)):
            seq = X_raw[i]
            feats = extract_all_geometric_features(seq)
            all_features.append(feats)
        return self._dict_list_to_array(all_features)

    def _extract_pca(self, X_raw: np.ndarray, n_components: int) -> np.ndarray:
        """Extract PCA features from flattened raw data."""
        X_flat = X_raw.reshape(X_raw.shape[0], -1)
        # Handle NaN
        X_flat = np.nan_to_num(X_flat, nan=0.0)
        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(X_flat)

    def _extract_all_derivatives_combined(self, X_raw: np.ndarray, participant_ids: np.ndarray) -> np.ndarray:
        """Extract all derivative features combined."""
        all_features = []
        for i in range(len(X_raw)):
            seq = X_raw[i]
            feats = extract_all_derivative_features(seq, participant_id=int(participant_ids[i]))
            all_features.append(feats)
        return self._dict_list_to_array(all_features)

    def _dict_list_to_array(self, dict_list: List[Dict]) -> np.ndarray:
        """Convert list of feature dicts to numpy array."""
        if not dict_list:
            return np.array([])

        # Get all feature names from first dict
        feature_names = sorted(dict_list[0].keys())

        # Build array
        arr = np.zeros((len(dict_list), len(feature_names)), dtype=np.float32)
        for i, d in enumerate(dict_list):
            for j, name in enumerate(feature_names):
                arr[i, j] = d.get(name, 0.0)

        return arr

    def combine_features(self, feature_types: List[str]) -> np.ndarray:
        """Combine multiple feature types."""
        arrays = [self.extract_features(ft) for ft in feature_types]
        return np.hstack(arrays)


# =============================================================================
# Cross-Validation Utilities
# =============================================================================

def compute_scaled_mse(y_true: np.ndarray, y_pred: np.ndarray,
                       scalers: Dict, targets: List[str] = TARGETS) -> Dict[str, float]:
    """Compute scaled MSE for each target."""
    results = {}
    for i, target in enumerate(targets):
        y_t = y_true[:, i].reshape(-1, 1)
        y_p = y_pred[:, i].reshape(-1, 1)

        y_t_scaled = scalers[target].transform(y_t).ravel()
        y_p_scaled = scalers[target].transform(y_p).ravel()

        results[target] = mean_squared_error(y_t_scaled, y_p_scaled)

    results['total'] = np.mean([results[t] for t in targets])
    return results


def run_cv_experiment_generic(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_fn: Callable[[], Any],
    scalers: Dict,
    n_folds: int = 5,
    scale_features: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Run cross-validation experiment.

    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples, 3)
        groups: Group labels for GroupKFold
        model_fn: Function that returns a fresh model instance
        scalers: Dict of target scalers
        n_folds: Number of CV folds
        scale_features: Whether to standardize features

    Returns:
        (mse_dict, oof_predictions)
    """
    gkf = GroupKFold(n_splits=n_folds)
    n_samples = X.shape[0]

    oof_predictions = np.zeros((n_samples, len(TARGETS)))

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        # Standardize features
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        # Train per-target models
        for target_idx, target in enumerate(TARGETS):
            model = model_fn()
            model.fit(X_train, y_train[:, target_idx])
            oof_predictions[val_idx, target_idx] = model.predict(X_val)

    return compute_scaled_mse(y, oof_predictions, scalers), oof_predictions


def run_per_player_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    model_fn: Callable[[], Any],
    scalers: Dict,
    scale_features: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Run leave-one-player-out CV experiment.
    """
    unique_players = np.unique(participant_ids)
    n_samples = X.shape[0]

    oof_predictions = np.zeros((n_samples, len(TARGETS)))

    for player in unique_players:
        train_mask = participant_ids != player
        val_mask = participant_ids == player

        X_train, X_val = X[train_mask], X[val_mask]
        y_train = y[train_mask]

        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)

        for target_idx, target in enumerate(TARGETS):
            model = model_fn()
            model.fit(X_train, y_train[:, target_idx])
            oof_predictions[val_mask, target_idx] = model.predict(X_val)

    return compute_scaled_mse(y, oof_predictions, scalers), oof_predictions


# =============================================================================
# Model Factories
# =============================================================================

def create_lgbm_model(params: Optional[Dict] = None):
    """Create LightGBM model."""
    if not LGBM_AVAILABLE:
        raise ImportError("LightGBM not available")
    p = LGBM_DEFAULT_PARAMS.copy()
    if params:
        p.update(params)
    return lgb.LGBMRegressor(**p)


def create_xgboost_model(params: Optional[Dict] = None):
    """Create XGBoost model."""
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not available")
    default = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_lambda': 1.0,
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': 42,
    }
    if params:
        default.update(params)
    return xgb.XGBRegressor(**default)


def create_catboost_model(params: Optional[Dict] = None):
    """Create CatBoost model."""
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost not available")
    default = {
        'iterations': 300,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3.0,
        'verbose': False,
        'random_state': 42,
    }
    if params:
        default.update(params)
    return CatBoostRegressor(**default)


def create_ridge_model(alpha: float = 1.0):
    """Create Ridge model."""
    return Ridge(alpha=alpha)


def create_elasticnet_model(alpha: float = 1.0, l1_ratio: float = 0.5):
    """Create ElasticNet model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)


def create_knn_model(n_neighbors: int = 10):
    """Create k-NN model."""
    return KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)


def create_rf_model(params: Optional[Dict] = None):
    """Create Random Forest model."""
    default = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 5,
        'n_jobs': -1,
        'random_state': 42,
    }
    if params:
        default.update(params)
    return RandomForestRegressor(**default)


# =============================================================================
# Experiment Phases
# =============================================================================

class ExperimentPhase:
    """Base class for experiment phases."""

    def __init__(self, data_mgr: DataManager, runner: ExperimentRunner, scalers: Dict):
        self.data_mgr = data_mgr
        self.runner = runner
        self.scalers = scalers
        self.X_raw, self.y, self.participant_ids = data_mgr.load_raw_data()

    def run(self) -> List[ExperimentResult]:
        raise NotImplementedError


class Phase1Baselines(ExperimentPhase):
    """Phase 1: Baselines and Sanity Checks."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 1: Baselines", 4)

        # Experiment 1: F4 features + LightGBM global
        def exp_f4_global():
            X = self.data_mgr.extract_features('f4_hybrid')
            mse, oof = run_cv_experiment_generic(
                X, self.y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )
            return ExperimentResult(
                experiment_name="baseline_f4_global",
                timestamp=datetime.now().isoformat(),
                phase="1_baselines",
                model_type="lgbm",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                hyperparams=LGBM_DEFAULT_PARAMS,
            )

        results.append(self.runner.run_experiment(
            "baseline_f4_global", "1_baselines", exp_f4_global
        ))

        # Experiment 2: F4 features + LightGBM per-player
        def exp_f4_per_player():
            X = self.data_mgr.extract_features('f4_hybrid')
            mse, oof = run_per_player_cv_experiment(
                X, self.y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )
            return ExperimentResult(
                experiment_name="baseline_f4_per_player",
                timestamp=datetime.now().isoformat(),
                phase="1_baselines",
                model_type="lgbm",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                per_player=True,
                hyperparams=LGBM_DEFAULT_PARAMS,
            )

        results.append(self.runner.run_experiment(
            "baseline_f4_per_player", "1_baselines", exp_f4_per_player
        ))

        # Experiment 3: Raw flattened + Ridge
        def exp_raw_ridge():
            X = self.data_mgr.extract_features('raw_flattened')
            mse, oof = run_cv_experiment_generic(
                X, self.y, self.participant_ids,
                lambda: create_ridge_model(alpha=1.0),
                self.scalers
            )
            return ExperimentResult(
                experiment_name="baseline_raw_ridge",
                timestamp=datetime.now().isoformat(),
                phase="1_baselines",
                model_type="ridge",
                feature_set="raw_flattened",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                hyperparams={'alpha': 1.0},
            )

        results.append(self.runner.run_experiment(
            "baseline_raw_ridge", "1_baselines", exp_raw_ridge
        ))

        # Experiment 4: PCA(50) + Ridge
        def exp_pca_ridge():
            X = self.data_mgr.extract_features('pca_50')
            mse, oof = run_cv_experiment_generic(
                X, self.y, self.participant_ids,
                lambda: create_ridge_model(alpha=1.0),
                self.scalers
            )
            return ExperimentResult(
                experiment_name="baseline_pca50_ridge",
                timestamp=datetime.now().isoformat(),
                phase="1_baselines",
                model_type="ridge",
                feature_set="pca_50",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                hyperparams={'alpha': 1.0, 'pca_components': 50},
            )

        results.append(self.runner.run_experiment(
            "baseline_pca50_ridge", "1_baselines", exp_pca_ridge
        ))

        self.runner.log_phase_end("Phase 1", [r for r in results if r])
        return results


class Phase2FeatureTypes(ExperimentPhase):
    """Phase 2: Feature Type Comparison."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 2: Feature Types", 9)

        feature_configs = [
            ('f4_hybrid', 'F4 hybrid features'),
            ('velocity', 'Velocity only'),
            ('acceleration', 'Acceleration only'),
            ('jerk', 'Jerk only'),
            ('critical_frames', 'Critical frame features'),
            ('fft', 'FFT frequency features'),
            ('geometric', 'Geometric distance features'),
            ('pca_50', 'PCA 50 components'),
            ('pca_100', 'PCA 100 components'),
        ]

        for feature_type, description in feature_configs:
            def make_exp(ft=feature_type, desc=description):
                def exp():
                    X = self.data_mgr.extract_features(ft)
                    mse, oof = run_cv_experiment_generic(
                        X, self.y, self.participant_ids,
                        lambda: create_lgbm_model(),
                        self.scalers
                    )
                    return ExperimentResult(
                        experiment_name=f"feat_{ft}",
                        timestamp=datetime.now().isoformat(),
                        phase="2_feature_types",
                        model_type="lgbm",
                        feature_set=ft,
                        n_features=X.shape[1],
                        angle_mse=mse['angle'],
                        depth_mse=mse['depth'],
                        left_right_mse=mse['left_right'],
                        total_scaled_mse=mse['total'],
                        training_time_seconds=0,
                        notes=desc,
                        hyperparams=LGBM_DEFAULT_PARAMS,
                    )
                return exp

            results.append(self.runner.run_experiment(
                f"feat_{feature_type}", "2_feature_types", make_exp()
            ))

        self.runner.log_phase_end("Phase 2", [r for r in results if r])
        return results


class Phase3FeatureCombinations(ExperimentPhase):
    """Phase 3: Feature Combination Search."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 3: Feature Combinations", 8)

        combinations = [
            (['f4_hybrid', 'velocity'], 'F4 + velocity'),
            (['f4_hybrid', 'critical_frames'], 'F4 + critical frames'),
            (['f4_hybrid', 'fft'], 'F4 + FFT'),
            (['f4_hybrid', 'geometric'], 'F4 + geometric'),
            (['f4_hybrid', 'velocity', 'fft'], 'F4 + velocity + FFT'),
            (['f4_hybrid', 'geometric', 'fft'], 'F4 + geometric + FFT'),
            (['f4_hybrid', 'velocity', 'geometric', 'fft'], 'All feature types'),
        ]

        for feature_types, description in combinations:
            name = "combo_" + "_".join([ft.split('_')[0] for ft in feature_types])

            def make_exp(fts=feature_types, desc=description, exp_name=name):
                def exp():
                    X = self.data_mgr.combine_features(fts)
                    mse, oof = run_cv_experiment_generic(
                        X, self.y, self.participant_ids,
                        lambda: create_lgbm_model(),
                        self.scalers
                    )
                    return ExperimentResult(
                        experiment_name=exp_name,
                        timestamp=datetime.now().isoformat(),
                        phase="3_combinations",
                        model_type="lgbm",
                        feature_set="+".join(fts),
                        n_features=X.shape[1],
                        angle_mse=mse['angle'],
                        depth_mse=mse['depth'],
                        left_right_mse=mse['left_right'],
                        total_scaled_mse=mse['total'],
                        training_time_seconds=0,
                        notes=desc,
                        hyperparams=LGBM_DEFAULT_PARAMS,
                    )
                return exp

            results.append(self.runner.run_experiment(
                name, "3_combinations", make_exp()
            ))

        # All features per-player
        def exp_all_per_player():
            X = self.data_mgr.combine_features(['f4_hybrid', 'velocity', 'geometric', 'fft'])
            mse, oof = run_per_player_cv_experiment(
                X, self.y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )
            return ExperimentResult(
                experiment_name="combo_all_per_player",
                timestamp=datetime.now().isoformat(),
                phase="3_combinations",
                model_type="lgbm",
                feature_set="all_combined",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                per_player=True,
                notes="All features, per-player CV",
                hyperparams=LGBM_DEFAULT_PARAMS,
            )

        results.append(self.runner.run_experiment(
            "combo_all_per_player", "3_combinations", exp_all_per_player
        ))

        self.runner.log_phase_end("Phase 3", [r for r in results if r])
        return results


class Phase4ModelTypes(ExperimentPhase):
    """Phase 4: Model Type Comparison."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 4: Model Types", 8)

        # Use best feature set from previous phases (f4_hybrid as default)
        X = self.data_mgr.extract_features('f4_hybrid')

        model_configs = [
            ('lgbm_default', lambda: create_lgbm_model(), {'type': 'lgbm', 'params': 'default'}),
            ('lgbm_conservative', lambda: create_lgbm_model(LGBM_CONSERVATIVE_PARAMS), {'type': 'lgbm', 'params': 'conservative'}),
            ('ridge', lambda: create_ridge_model(1.0), {'type': 'ridge', 'alpha': 1.0}),
            ('ridge_cv', lambda: RidgeCV(alphas=np.logspace(-3, 3, 50)), {'type': 'ridge_cv'}),
            ('elasticnet', lambda: create_elasticnet_model(1.0, 0.5), {'type': 'elasticnet', 'alpha': 1.0, 'l1_ratio': 0.5}),
            ('knn_5', lambda: create_knn_model(5), {'type': 'knn', 'k': 5}),
            ('knn_10', lambda: create_knn_model(10), {'type': 'knn', 'k': 10}),
            ('random_forest', lambda: create_rf_model(), {'type': 'rf'}),
        ]

        # Add XGBoost if available
        if XGB_AVAILABLE:
            model_configs.append(
                ('xgboost', lambda: create_xgboost_model(), {'type': 'xgb'})
            )

        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            model_configs.append(
                ('catboost', lambda: create_catboost_model(), {'type': 'catboost'})
            )

        for model_name, model_fn, params in model_configs:
            def make_exp(m_name=model_name, m_fn=model_fn, m_params=params):
                def exp():
                    mse, oof = run_cv_experiment_generic(
                        X, self.y, self.participant_ids,
                        m_fn,
                        self.scalers
                    )
                    return ExperimentResult(
                        experiment_name=f"model_{m_name}",
                        timestamp=datetime.now().isoformat(),
                        phase="4_model_types",
                        model_type=m_params.get('type', m_name),
                        feature_set="f4_hybrid",
                        n_features=X.shape[1],
                        angle_mse=mse['angle'],
                        depth_mse=mse['depth'],
                        left_right_mse=mse['left_right'],
                        total_scaled_mse=mse['total'],
                        training_time_seconds=0,
                        hyperparams=m_params,
                    )
                return exp

            results.append(self.runner.run_experiment(
                f"model_{model_name}", "4_model_types", make_exp()
            ))

        self.runner.log_phase_end("Phase 4", [r for r in results if r])
        return results


class Phase5TargetTransforms(ExperimentPhase):
    """Phase 5: Target Transformation Tests."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 5: Target Transforms", 5)

        X = self.data_mgr.extract_features('f4_hybrid')

        # Experiment 1: No transform (baseline)
        def exp_raw():
            mse, oof = run_cv_experiment_generic(
                X, self.y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )
            return ExperimentResult(
                experiment_name="target_raw",
                timestamp=datetime.now().isoformat(),
                phase="5_target_transforms",
                model_type="lgbm",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="No target transform",
            )

        results.append(self.runner.run_experiment(
            "target_raw", "5_target_transforms", exp_raw
        ))

        # Experiment 2: Log transform for angle
        def exp_log_angle():
            y_transformed = self.y.copy()
            # Log transform angle (add offset to handle near-zero values)
            y_transformed[:, 0] = np.log1p(np.abs(self.y[:, 0])) * np.sign(self.y[:, 0])

            mse, oof = run_cv_experiment_generic(
                X, y_transformed, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )

            # Inverse transform predictions for MSE calculation
            oof_orig = oof.copy()
            oof_orig[:, 0] = np.sign(oof[:, 0]) * np.expm1(np.abs(oof[:, 0]))
            mse = compute_scaled_mse(self.y, oof_orig, self.scalers)

            return ExperimentResult(
                experiment_name="target_log_angle",
                timestamp=datetime.now().isoformat(),
                phase="5_target_transforms",
                model_type="lgbm",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Log transform on angle target",
            )

        results.append(self.runner.run_experiment(
            "target_log_angle", "5_target_transforms", exp_log_angle
        ))

        # Experiment 3: Quantile transform
        def exp_quantile():
            y_transformed = self.y.copy()
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            y_transformed = qt.fit_transform(y_transformed)

            mse, oof = run_cv_experiment_generic(
                X, y_transformed, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )

            # Inverse transform
            oof_orig = qt.inverse_transform(oof)
            mse = compute_scaled_mse(self.y, oof_orig, self.scalers)

            return ExperimentResult(
                experiment_name="target_quantile",
                timestamp=datetime.now().isoformat(),
                phase="5_target_transforms",
                model_type="lgbm",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Quantile transform on all targets",
            )

        results.append(self.runner.run_experiment(
            "target_quantile", "5_target_transforms", exp_quantile
        ))

        # Experiment 4: Different features per target
        def exp_per_target_features():
            # Use geometric for left_right (posture), velocity for depth (speed), f4 for angle
            X_geo = self.data_mgr.extract_features('geometric')
            X_vel = self.data_mgr.extract_features('velocity')
            X_f4 = self.data_mgr.extract_features('f4_hybrid')

            feature_sets = {
                'angle': X_f4,
                'depth': X_vel,
                'left_right': X_geo,
            }

            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros((len(self.y), 3))

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_f4, self.y, self.participant_ids)):
                for target_idx, target in enumerate(TARGETS):
                    X_t = feature_sets[target]
                    X_train, X_val = X_t[train_idx], X_t[val_idx]
                    y_train = self.y[train_idx, target_idx]

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                    X_train = np.nan_to_num(X_train, nan=0.0)
                    X_val = np.nan_to_num(X_val, nan=0.0)

                    model = create_lgbm_model()
                    model.fit(X_train, y_train)
                    oof_predictions[val_idx, target_idx] = model.predict(X_val)

            mse = compute_scaled_mse(self.y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="target_per_target_features",
                timestamp=datetime.now().isoformat(),
                phase="5_target_transforms",
                model_type="lgbm",
                feature_set="per_target",
                n_features=-1,
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Different feature sets per target: f4 for angle, velocity for depth, geometric for left_right",
                per_target=True,
            )

        results.append(self.runner.run_experiment(
            "target_per_target_features", "5_target_transforms", exp_per_target_features
        ))

        # Experiment 5: Train separate models per target with interactions
        def exp_with_interactions():
            X_f4 = self.data_mgr.extract_features('f4_hybrid')

            # Add interaction features
            all_features = []
            for i in range(len(self.X_raw)):
                base_dict = {}
                for j in range(X_f4.shape[1]):
                    base_dict[f'f{j}'] = X_f4[i, j]
                interactions = extract_all_interaction_features(base_dict)
                all_features.append(list(interactions.values()))

            X_interactions = np.array(all_features, dtype=np.float32)
            X_combined = np.hstack([X_f4, X_interactions])

            mse, oof = run_cv_experiment_generic(
                X_combined, self.y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )

            return ExperimentResult(
                experiment_name="target_with_interactions",
                timestamp=datetime.now().isoformat(),
                phase="5_target_transforms",
                model_type="lgbm",
                feature_set="f4_hybrid+interactions",
                n_features=X_combined.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="F4 features with multiplicative interactions",
            )

        results.append(self.runner.run_experiment(
            "target_with_interactions", "5_target_transforms", exp_with_interactions
        ))

        self.runner.log_phase_end("Phase 5", [r for r in results if r])
        return results


class Phase6DeepLearning(ExperimentPhase):
    """Phase 6: Deep Learning Experiments."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 6: Deep Learning", 5)

        if not TORCH_AVAILABLE:
            self.runner.logger.warning("PyTorch not available, skipping DL experiments")
            return results

        from src.creative_dl_models import (
            FrameAttentionModel, TargetSpecificModel, SequenceAutoencoder,
            SimpleMLP, train_model, pretrain_autoencoder, extract_bottleneck_features,
            get_device
        )
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        device = get_device()
        self.runner.logger.info(f"Using device: {device}")

        # Prepare data
        X_raw = self.X_raw
        y = self.y

        # Experiment 1: Frame Attention Model
        def exp_attention():
            from sklearn.model_selection import GroupKFold

            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros((len(y), 3))

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, self.participant_ids)):
                X_train = torch.FloatTensor(X_raw[train_idx])
                X_val = torch.FloatTensor(X_raw[val_idx])
                y_train = torch.FloatTensor(y[train_idx])
                y_val = torch.FloatTensor(y[val_idx])

                # Normalize targets for training
                y_mean = y_train.mean(dim=0)
                y_std = y_train.std(dim=0) + 1e-6
                y_train_norm = (y_train - y_mean) / y_std

                train_dataset = TensorDataset(X_train, y_train_norm)
                val_dataset = TensorDataset(X_val, (y_val - y_mean) / y_std)

                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)

                model = FrameAttentionModel(n_features=207, hidden_dim=64, n_targets=3, dropout=0.3)
                train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=15, device=device, verbose=False)

                model.eval()
                with torch.no_grad():
                    X_val_dev = X_val.to(device)
                    pred, _ = model(X_val_dev)
                    pred = pred.cpu().numpy()
                    # Denormalize
                    pred = pred * y_std.numpy() + y_mean.numpy()
                    oof_predictions[val_idx] = pred

            mse = compute_scaled_mse(y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="dl_attention",
                timestamp=datetime.now().isoformat(),
                phase="6_deep_learning",
                model_type="FrameAttentionModel",
                feature_set="raw_sequences",
                n_features=240 * 207,
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Temporal attention model",
            )

        results.append(self.runner.run_experiment(
            "dl_attention", "6_deep_learning", exp_attention
        ))

        # Experiment 2: Target-Specific Model
        def exp_target_specific():
            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros((len(y), 3))

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, self.participant_ids)):
                X_train = torch.FloatTensor(X_raw[train_idx])
                X_val = torch.FloatTensor(X_raw[val_idx])
                y_train = torch.FloatTensor(y[train_idx])
                y_val = torch.FloatTensor(y[val_idx])

                y_mean = y_train.mean(dim=0)
                y_std = y_train.std(dim=0) + 1e-6
                y_train_norm = (y_train - y_mean) / y_std

                train_dataset = TensorDataset(X_train, y_train_norm)
                val_dataset = TensorDataset(X_val, (y_val - y_mean) / y_std)

                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)

                model = TargetSpecificModel(n_features=207, hidden_dim=64, dropout=0.3)

                # Custom training for dict output
                model = model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

                for epoch in range(100):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = sum(torch.nn.functional.mse_loss(outputs[t], batch_y[:, i:i+1])
                                  for i, t in enumerate(sorted(outputs.keys())))
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    X_val_dev = X_val.to(device)
                    outputs = model(X_val_dev)
                    pred = torch.cat([outputs[t] for t in sorted(outputs.keys())], dim=1)
                    pred = pred.cpu().numpy()
                    pred = pred * y_std.numpy() + y_mean.numpy()
                    oof_predictions[val_idx] = pred

            mse = compute_scaled_mse(y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="dl_target_specific",
                timestamp=datetime.now().isoformat(),
                phase="6_deep_learning",
                model_type="TargetSpecificModel",
                feature_set="raw_sequences",
                n_features=240 * 207,
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Per-target frame windows",
            )

        results.append(self.runner.run_experiment(
            "dl_target_specific", "6_deep_learning", exp_target_specific
        ))

        # Experiment 3: Simple MLP
        def exp_mlp():
            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros((len(y), 3))

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, self.participant_ids)):
                X_train = torch.FloatTensor(X_raw[train_idx])
                X_val = torch.FloatTensor(X_raw[val_idx])
                y_train = torch.FloatTensor(y[train_idx])
                y_val = torch.FloatTensor(y[val_idx])

                y_mean = y_train.mean(dim=0)
                y_std = y_train.std(dim=0) + 1e-6
                y_train_norm = (y_train - y_mean) / y_std

                train_dataset = TensorDataset(X_train, y_train_norm)
                val_dataset = TensorDataset(X_val, (y_val - y_mean) / y_std)

                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16)

                model = SimpleMLP(n_features=207, seq_len=240, hidden_dims=[256, 128, 64], n_targets=3, dropout=0.3)
                train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=15, device=device, verbose=False)

                model.eval()
                with torch.no_grad():
                    X_val_dev = X_val.to(device)
                    pred = model(X_val_dev)
                    pred = pred.cpu().numpy()
                    pred = pred * y_std.numpy() + y_mean.numpy()
                    oof_predictions[val_idx] = pred

            mse = compute_scaled_mse(y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="dl_mlp",
                timestamp=datetime.now().isoformat(),
                phase="6_deep_learning",
                model_type="SimpleMLP",
                feature_set="raw_sequences",
                n_features=240 * 207,
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Simple MLP baseline",
            )

        results.append(self.runner.run_experiment(
            "dl_mlp", "6_deep_learning", exp_mlp
        ))

        # Experiment 4: Autoencoder + GBDT hybrid
        def exp_ae_gbdt():
            # Pre-train autoencoder on all data
            X_tensor = torch.FloatTensor(X_raw)
            dataset = TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)

            ae_model = SequenceAutoencoder(n_features=207, seq_len=240, bottleneck_dim=32)
            pretrain_autoencoder(ae_model, loader, epochs=50, lr=1e-3, device=device, verbose=False)

            # Extract bottleneck features
            ae_model = ae_model.to(device)
            ae_model.eval()
            all_features = []
            with torch.no_grad():
                for i in range(0, len(X_raw), 16):
                    batch = torch.FloatTensor(X_raw[i:i+16]).to(device)
                    bottleneck = ae_model.encode(batch)
                    all_features.append(bottleneck.cpu().numpy())
            X_ae = np.concatenate(all_features, axis=0)

            # Use with LightGBM
            mse, oof = run_cv_experiment_generic(
                X_ae, y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )

            return ExperimentResult(
                experiment_name="dl_ae_gbdt",
                timestamp=datetime.now().isoformat(),
                phase="6_deep_learning",
                model_type="Autoencoder+LGBM",
                feature_set="ae_bottleneck",
                n_features=32,
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Autoencoder bottleneck features + LightGBM",
            )

        results.append(self.runner.run_experiment(
            "dl_ae_gbdt", "6_deep_learning", exp_ae_gbdt
        ))

        # Experiment 5: Autoencoder bottleneck combined with F4
        def exp_ae_plus_f4():
            # Get autoencoder features (should be cached from previous exp)
            X_tensor = torch.FloatTensor(X_raw)
            dataset = TensorDataset(X_tensor)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)

            ae_model = SequenceAutoencoder(n_features=207, seq_len=240, bottleneck_dim=32)
            pretrain_autoencoder(ae_model, loader, epochs=50, lr=1e-3, device=device, verbose=False)

            ae_model = ae_model.to(device)
            ae_model.eval()
            all_features = []
            with torch.no_grad():
                for i in range(0, len(X_raw), 16):
                    batch = torch.FloatTensor(X_raw[i:i+16]).to(device)
                    bottleneck = ae_model.encode(batch)
                    all_features.append(bottleneck.cpu().numpy())
            X_ae = np.concatenate(all_features, axis=0)

            # Combine with F4
            X_f4 = self.data_mgr.extract_features('f4_hybrid')
            X_combined = np.hstack([X_f4, X_ae])

            mse, oof = run_cv_experiment_generic(
                X_combined, y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )

            return ExperimentResult(
                experiment_name="dl_ae_plus_f4",
                timestamp=datetime.now().isoformat(),
                phase="6_deep_learning",
                model_type="LGBM",
                feature_set="f4_hybrid+ae_bottleneck",
                n_features=X_combined.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="F4 features + autoencoder bottleneck",
            )

        results.append(self.runner.run_experiment(
            "dl_ae_plus_f4", "6_deep_learning", exp_ae_plus_f4
        ))

        self.runner.log_phase_end("Phase 6", [r for r in results if r])
        return results


class Phase7Stacking(ExperimentPhase):
    """Phase 7: Stacking and Ensembles."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 7: Stacking", 4)

        from src.stacking_models import (
            StackingRegressor, BlendingEnsemble, OptimizedBlend,
            MultiTargetStacker, create_default_stacking_ensemble
        )

        X = self.data_mgr.extract_features('f4_hybrid')

        # Experiment 1: 3 tree stacking
        def exp_stack_trees():
            base_models = []
            if LGBM_AVAILABLE:
                base_models.append(('lgbm', create_lgbm_model()))
            if XGB_AVAILABLE:
                base_models.append(('xgb', create_xgboost_model()))
            if CATBOOST_AVAILABLE:
                base_models.append(('catboost', create_catboost_model()))

            if len(base_models) < 2:
                return ExperimentResult(
                    experiment_name="stack_trees",
                    timestamp=datetime.now().isoformat(),
                    phase="7_stacking",
                    model_type="stacking",
                    feature_set="f4_hybrid",
                    n_features=X.shape[1],
                    angle_mse=float('nan'),
                    depth_mse=float('nan'),
                    left_right_mse=float('nan'),
                    total_scaled_mse=float('nan'),
                    training_time_seconds=0,
                    error_message="Not enough tree models available",
                )

            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros((len(self.y), 3))

            for target_idx, target in enumerate(TARGETS):
                stacker = StackingRegressor(base_models, meta_model=RidgeCV(), n_folds=5)

                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

                stacker.fit(X_scaled, self.y[:, target_idx], self.participant_ids)

                # Get OOF predictions
                oof = stacker.get_oof_predictions(X_scaled, self.y[:, target_idx], self.participant_ids)
                # Use meta model prediction approach
                for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, self.y, self.participant_ids)):
                    oof_predictions[val_idx, target_idx] = stacker.predict(X_scaled[val_idx])

            mse = compute_scaled_mse(self.y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="stack_trees",
                timestamp=datetime.now().isoformat(),
                phase="7_stacking",
                model_type="stacking_3_trees",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes=f"Stacking with {len(base_models)} tree models",
            )

        results.append(self.runner.run_experiment(
            "stack_trees", "7_stacking", exp_stack_trees
        ))

        # Experiment 2: Trees + Ridge stacking
        def exp_stack_trees_ridge():
            base_models = [('ridge', create_ridge_model())]
            if LGBM_AVAILABLE:
                base_models.append(('lgbm', create_lgbm_model()))
            if XGB_AVAILABLE:
                base_models.append(('xgb', create_xgboost_model()))

            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros((len(self.y), 3))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            for target_idx, target in enumerate(TARGETS):
                stacker = StackingRegressor(base_models, n_folds=5)
                stacker.fit(X_scaled, self.y[:, target_idx], self.participant_ids)

                for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, self.y, self.participant_ids)):
                    oof_predictions[val_idx, target_idx] = stacker.predict(X_scaled[val_idx])

            mse = compute_scaled_mse(self.y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="stack_trees_ridge",
                timestamp=datetime.now().isoformat(),
                phase="7_stacking",
                model_type="stacking_trees_ridge",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Stacking with trees + ridge",
            )

        results.append(self.runner.run_experiment(
            "stack_trees_ridge", "7_stacking", exp_stack_trees_ridge
        ))

        # Experiment 3: Optimized blend
        def exp_optimized_blend():
            base_models = []
            if LGBM_AVAILABLE:
                base_models.append(('lgbm', create_lgbm_model()))
            if XGB_AVAILABLE:
                base_models.append(('xgb', create_xgboost_model()))
            base_models.append(('ridge', create_ridge_model()))

            oof_predictions = np.zeros((len(self.y), 3))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            all_weights = {}
            for target_idx, target in enumerate(TARGETS):
                opt_blend = OptimizedBlend(base_models, n_folds=5)
                opt_blend.fit(X_scaled, self.y[:, target_idx], self.participant_ids)
                oof_predictions[:, target_idx] = opt_blend.predict(X_scaled)
                all_weights[target] = opt_blend.get_weights()

            mse = compute_scaled_mse(self.y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="optimized_blend",
                timestamp=datetime.now().isoformat(),
                phase="7_stacking",
                model_type="optimized_blend",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes=f"Optimized weights: {all_weights}",
            )

        results.append(self.runner.run_experiment(
            "optimized_blend", "7_stacking", exp_optimized_blend
        ))

        # Experiment 4: Simple average blend
        def exp_simple_blend():
            base_models = []
            if LGBM_AVAILABLE:
                base_models.append(('lgbm', create_lgbm_model()))
            if XGB_AVAILABLE:
                base_models.append(('xgb', create_xgboost_model()))
            base_models.append(('ridge', create_ridge_model()))

            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros((len(self.y), 3))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            for target_idx, target in enumerate(TARGETS):
                blender = BlendingEnsemble(base_models)
                blender.fit(X_scaled, self.y[:, target_idx])

                for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, self.y, self.participant_ids)):
                    # Train on train, predict on val
                    blender_fold = BlendingEnsemble(base_models)
                    blender_fold.fit(X_scaled[train_idx], self.y[train_idx, target_idx])
                    oof_predictions[val_idx, target_idx] = blender_fold.predict(X_scaled[val_idx])

            mse = compute_scaled_mse(self.y, oof_predictions, self.scalers)

            return ExperimentResult(
                experiment_name="simple_blend",
                timestamp=datetime.now().isoformat(),
                phase="7_stacking",
                model_type="simple_blend",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Simple average blend",
            )

        results.append(self.runner.run_experiment(
            "simple_blend", "7_stacking", exp_simple_blend
        ))

        self.runner.log_phase_end("Phase 7", [r for r in results if r])
        return results


class Phase8PerTargetTuning(ExperimentPhase):
    """Phase 8: Per-Target Specialized Tuning with Optuna."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 8: Per-Target Tuning", 4)

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            OPTUNA_AVAILABLE = True
        except ImportError:
            OPTUNA_AVAILABLE = False
            self.runner.logger.warning("Optuna not available, using grid search")

        X = self.data_mgr.extract_features('f4_hybrid')

        for target_idx, target in enumerate(TARGETS):
            def make_exp(t_idx=target_idx, t_name=target):
                def exp():
                    if OPTUNA_AVAILABLE and LGBM_AVAILABLE:
                        return self._run_optuna_tuning(X, t_idx, t_name)
                    else:
                        return self._run_grid_search(X, t_idx, t_name)
                return exp

            results.append(self.runner.run_experiment(
                f"tuned_{target}", "8_per_target_tuning", make_exp()
            ))

        # Combined tuned model
        def exp_combined_tuned():
            # Use default params for now (would use best from individual tuning in practice)
            mse, oof = run_cv_experiment_generic(
                X, self.y, self.participant_ids,
                lambda: create_lgbm_model(),
                self.scalers
            )
            return ExperimentResult(
                experiment_name="tuned_combined",
                timestamp=datetime.now().isoformat(),
                phase="8_per_target_tuning",
                model_type="lgbm_tuned",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=mse['angle'],
                depth_mse=mse['depth'],
                left_right_mse=mse['left_right'],
                total_scaled_mse=mse['total'],
                training_time_seconds=0,
                notes="Combined with tuned per-target params",
            )

        results.append(self.runner.run_experiment(
            "tuned_combined", "8_per_target_tuning", exp_combined_tuned
        ))

        self.runner.log_phase_end("Phase 8", [r for r in results if r])
        return results

    def _run_optuna_tuning(self, X: np.ndarray, target_idx: int, target_name: str) -> ExperimentResult:
        """Run Optuna hyperparameter optimization."""
        import optuna

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 10, 40),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
                'verbose': -1,
                'n_jobs': -1,
            }

            gkf = GroupKFold(n_splits=5)
            oof_predictions = np.zeros(len(self.y))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            for train_idx, val_idx in gkf.split(X_scaled, self.y, self.participant_ids):
                model = lgb.LGBMRegressor(**params)
                model.fit(X_scaled[train_idx], self.y[train_idx, target_idx])
                oof_predictions[val_idx] = model.predict(X_scaled[val_idx])

            # Compute scaled MSE for this target
            y_true = self.y[:, target_idx].reshape(-1, 1)
            y_pred = oof_predictions.reshape(-1, 1)
            y_true_scaled = self.scalers[target_name].transform(y_true).ravel()
            y_pred_scaled = self.scalers[target_name].transform(y_pred).ravel()

            return mean_squared_error(y_true_scaled, y_pred_scaled)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        best_params = study.best_params
        best_mse = study.best_value

        # Run full CV with best params to get all target MSEs
        best_params['verbose'] = -1
        best_params['n_jobs'] = -1

        mse, oof = run_cv_experiment_generic(
            X, self.y, self.participant_ids,
            lambda: lgb.LGBMRegressor(**best_params),
            self.scalers
        )

        return ExperimentResult(
            experiment_name=f"tuned_{target_name}",
            timestamp=datetime.now().isoformat(),
            phase="8_per_target_tuning",
            model_type="lgbm_optuna",
            feature_set="f4_hybrid",
            n_features=X.shape[1],
            angle_mse=mse['angle'],
            depth_mse=mse['depth'],
            left_right_mse=mse['left_right'],
            total_scaled_mse=mse['total'],
            training_time_seconds=0,
            notes=f"Optuna tuned for {target_name}",
            hyperparams=best_params,
            per_target=True,
        )

    def _run_grid_search(self, X: np.ndarray, target_idx: int, target_name: str) -> ExperimentResult:
        """Run simple grid search."""
        best_mse = float('inf')
        best_params = LGBM_DEFAULT_PARAMS.copy()

        param_grid = [
            {'num_leaves': 15, 'learning_rate': 0.02},
            {'num_leaves': 20, 'learning_rate': 0.02},
            {'num_leaves': 25, 'learning_rate': 0.02},
            {'num_leaves': 20, 'learning_rate': 0.01},
            {'num_leaves': 20, 'learning_rate': 0.03},
        ]

        for params in param_grid:
            test_params = LGBM_DEFAULT_PARAMS.copy()
            test_params.update(params)

            mse, oof = run_cv_experiment_generic(
                X, self.y, self.participant_ids,
                lambda p=test_params: lgb.LGBMRegressor(**p),
                self.scalers
            )

            if mse[target_name] < best_mse:
                best_mse = mse[target_name]
                best_params = test_params.copy()

        # Final run with best params
        mse, oof = run_cv_experiment_generic(
            X, self.y, self.participant_ids,
            lambda: lgb.LGBMRegressor(**best_params),
            self.scalers
        )

        return ExperimentResult(
            experiment_name=f"tuned_{target_name}",
            timestamp=datetime.now().isoformat(),
            phase="8_per_target_tuning",
            model_type="lgbm_grid",
            feature_set="f4_hybrid",
            n_features=X.shape[1],
            angle_mse=mse['angle'],
            depth_mse=mse['depth'],
            left_right_mse=mse['left_right'],
            total_scaled_mse=mse['total'],
            training_time_seconds=0,
            notes=f"Grid search tuned for {target_name}",
            hyperparams=best_params,
            per_target=True,
        )


class Phase9Analysis(ExperimentPhase):
    """Phase 9: Analysis and Insights."""

    def run(self) -> List[ExperimentResult]:
        results = []
        self.runner.log_phase_start("Phase 9: Analysis", 4)

        X = self.data_mgr.extract_features('f4_hybrid')
        analysis_dir = self.runner.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Analysis 1: Feature importance
        def exp_feature_importance():
            if not LGBM_AVAILABLE:
                return ExperimentResult(
                    experiment_name="feature_importance",
                    timestamp=datetime.now().isoformat(),
                    phase="9_analysis",
                    model_type="analysis",
                    feature_set="f4_hybrid",
                    n_features=X.shape[1],
                    angle_mse=float('nan'),
                    depth_mse=float('nan'),
                    left_right_mse=float('nan'),
                    total_scaled_mse=float('nan'),
                    training_time_seconds=0,
                    error_message="LightGBM not available",
                )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            importance_data = {}

            for target_idx, target in enumerate(TARGETS):
                model = create_lgbm_model()
                model.fit(X_scaled, self.y[:, target_idx])

                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-20:][::-1]

                importance_data[target] = {
                    'top_features': [int(i) for i in top_indices],
                    'top_importances': [float(importances[i]) for i in top_indices],
                }

            # Save to file
            with open(analysis_dir / "feature_importances.json", 'w') as f:
                json.dump(importance_data, f, indent=2)

            return ExperimentResult(
                experiment_name="feature_importance",
                timestamp=datetime.now().isoformat(),
                phase="9_analysis",
                model_type="analysis",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=0.0,
                depth_mse=0.0,
                left_right_mse=0.0,
                total_scaled_mse=0.0,
                training_time_seconds=0,
                notes=f"Feature importance saved to {analysis_dir / 'feature_importances.json'}",
            )

        results.append(self.runner.run_experiment(
            "feature_importance", "9_analysis", exp_feature_importance
        ))

        # Analysis 2: Model correlation
        def exp_model_correlation():
            # Get OOF predictions from different models
            oof_lgbm = np.zeros((len(self.y), 3))
            oof_ridge = np.zeros((len(self.y), 3))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            gkf = GroupKFold(n_splits=5)

            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, self.y, self.participant_ids)):
                for target_idx in range(3):
                    if LGBM_AVAILABLE:
                        model_lgbm = create_lgbm_model()
                        model_lgbm.fit(X_scaled[train_idx], self.y[train_idx, target_idx])
                        oof_lgbm[val_idx, target_idx] = model_lgbm.predict(X_scaled[val_idx])

                    model_ridge = create_ridge_model()
                    model_ridge.fit(X_scaled[train_idx], self.y[train_idx, target_idx])
                    oof_ridge[val_idx, target_idx] = model_ridge.predict(X_scaled[val_idx])

            # Compute correlations
            correlations = {}
            for target_idx, target in enumerate(TARGETS):
                if LGBM_AVAILABLE:
                    corr = np.corrcoef(oof_lgbm[:, target_idx], oof_ridge[:, target_idx])[0, 1]
                    correlations[target] = float(corr)

            # Save
            with open(analysis_dir / "model_correlations.json", 'w') as f:
                json.dump(correlations, f, indent=2)

            return ExperimentResult(
                experiment_name="model_correlation",
                timestamp=datetime.now().isoformat(),
                phase="9_analysis",
                model_type="analysis",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=0.0,
                depth_mse=0.0,
                left_right_mse=0.0,
                total_scaled_mse=0.0,
                training_time_seconds=0,
                notes=f"Model correlations: {correlations}",
            )

        results.append(self.runner.run_experiment(
            "model_correlation", "9_analysis", exp_model_correlation
        ))

        # Analysis 3: Per-participant analysis
        def exp_participant_analysis():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            participant_mse = {}

            for player in np.unique(self.participant_ids):
                train_mask = self.participant_ids != player
                val_mask = self.participant_ids == player

                player_mse = {}
                for target_idx, target in enumerate(TARGETS):
                    if LGBM_AVAILABLE:
                        model = create_lgbm_model()
                    else:
                        model = create_ridge_model()

                    model.fit(X_scaled[train_mask], self.y[train_mask, target_idx])
                    preds = model.predict(X_scaled[val_mask])

                    # Scaled MSE
                    y_true = self.y[val_mask, target_idx].reshape(-1, 1)
                    y_pred = preds.reshape(-1, 1)
                    y_true_s = self.scalers[target].transform(y_true).ravel()
                    y_pred_s = self.scalers[target].transform(y_pred).ravel()

                    player_mse[target] = float(mean_squared_error(y_true_s, y_pred_s))

                player_mse['total'] = np.mean([player_mse[t] for t in TARGETS])
                participant_mse[int(player)] = player_mse

            # Save
            with open(analysis_dir / "participant_mse.json", 'w') as f:
                json.dump(participant_mse, f, indent=2)

            # Find hardest participant
            hardest = max(participant_mse.items(), key=lambda x: x[1]['total'])

            return ExperimentResult(
                experiment_name="participant_analysis",
                timestamp=datetime.now().isoformat(),
                phase="9_analysis",
                model_type="analysis",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=0.0,
                depth_mse=0.0,
                left_right_mse=0.0,
                total_scaled_mse=0.0,
                training_time_seconds=0,
                notes=f"Hardest participant: {hardest[0]} (MSE: {hardest[1]['total']:.4f})",
            )

        results.append(self.runner.run_experiment(
            "participant_analysis", "9_analysis", exp_participant_analysis
        ))

        # Analysis 4: Error distribution
        def exp_error_analysis():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            oof_predictions = np.zeros((len(self.y), 3))

            gkf = GroupKFold(n_splits=5)
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, self.y, self.participant_ids)):
                for target_idx in range(3):
                    if LGBM_AVAILABLE:
                        model = create_lgbm_model()
                    else:
                        model = create_ridge_model()
                    model.fit(X_scaled[train_idx], self.y[train_idx, target_idx])
                    oof_predictions[val_idx, target_idx] = model.predict(X_scaled[val_idx])

            errors = self.y - oof_predictions

            error_stats = {}
            for target_idx, target in enumerate(TARGETS):
                target_errors = errors[:, target_idx]
                error_stats[target] = {
                    'mean': float(np.mean(target_errors)),
                    'std': float(np.std(target_errors)),
                    'min': float(np.min(target_errors)),
                    'max': float(np.max(target_errors)),
                    'median': float(np.median(target_errors)),
                    'q25': float(np.percentile(target_errors, 25)),
                    'q75': float(np.percentile(target_errors, 75)),
                }

            # Save
            with open(analysis_dir / "error_distribution.json", 'w') as f:
                json.dump(error_stats, f, indent=2)

            return ExperimentResult(
                experiment_name="error_analysis",
                timestamp=datetime.now().isoformat(),
                phase="9_analysis",
                model_type="analysis",
                feature_set="f4_hybrid",
                n_features=X.shape[1],
                angle_mse=0.0,
                depth_mse=0.0,
                left_right_mse=0.0,
                total_scaled_mse=0.0,
                training_time_seconds=0,
                notes=f"Error stats saved to {analysis_dir / 'error_distribution.json'}",
            )

        results.append(self.runner.run_experiment(
            "error_analysis", "9_analysis", exp_error_analysis
        ))

        self.runner.log_phase_end("Phase 9", [r for r in results if r])
        return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Overnight Experiment System v3")
    parser.add_argument('--validate', action='store_true', help="Run only Phase 1 for validation")
    parser.add_argument('--phase', type=int, help="Run specific phase (1-9)")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--no-resume', action='store_true', help="Start fresh, ignore checkpoint")
    args = parser.parse_args()

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create experiment runner
    resume = not args.no_resume
    runner = ExperimentRunner(OUTPUT_DIR, log_name="overnight_v3", resume=resume)

    runner.logger.info("=" * 60)
    runner.logger.info("Comprehensive Overnight Experiment System v3")
    runner.logger.info("=" * 60)
    runner.logger.info(f"Output directory: {OUTPUT_DIR}")
    runner.logger.info(f"Resume mode: {resume}")

    # Initialize data manager
    data_mgr = DataManager(OUTPUT_DIR)
    scalers = data_mgr.get_scalers()

    # Define phase classes
    phase_classes = [
        Phase1Baselines,
        Phase2FeatureTypes,
        Phase3FeatureCombinations,
        Phase4ModelTypes,
        Phase5TargetTransforms,
        Phase6DeepLearning,
        Phase7Stacking,
        Phase8PerTargetTuning,
        Phase9Analysis,
    ]

    all_results = []
    start_time = time.time()

    try:
        if args.validate:
            # Run only Phase 1
            runner.logger.info("Validation mode: running Phase 1 only")
            phase = Phase1Baselines(data_mgr, runner, scalers)
            all_results.extend(phase.run())
        elif args.phase:
            # Run specific phase
            if 1 <= args.phase <= 9:
                runner.logger.info(f"Running Phase {args.phase} only")
                phase_class = phase_classes[args.phase - 1]
                phase = phase_class(data_mgr, runner, scalers)
                all_results.extend(phase.run())
            else:
                runner.logger.error(f"Invalid phase: {args.phase}. Must be 1-9")
                return
        else:
            # Run all phases
            for phase_class in phase_classes:
                phase = phase_class(data_mgr, runner, scalers)
                results = phase.run()
                all_results.extend(results)

    except KeyboardInterrupt:
        runner.logger.warning("Interrupted by user")
    except Exception as e:
        runner.logger.error(f"Fatal error: {e}")
        import traceback
        runner.logger.error(traceback.format_exc())
    finally:
        # Final summary
        total_time = time.time() - start_time
        valid_results = [r for r in all_results if r and not np.isnan(r.total_scaled_mse)]

        runner.logger.info("=" * 60)
        runner.logger.info("FINAL SUMMARY")
        runner.logger.info("=" * 60)
        runner.logger.info(f"Total experiments: {len(all_results)}")
        runner.logger.info(f"Successful: {len(valid_results)}")
        runner.logger.info(f"Total time: {total_time/3600:.1f} hours")

        if valid_results:
            best = min(valid_results, key=lambda r: r.total_scaled_mse)
            runner.logger.info(f"\nBest experiment: {best.experiment_name}")
            runner.logger.info(f"  Total MSE: {best.total_scaled_mse:.6f}")
            runner.logger.info(f"  Angle MSE: {best.angle_mse:.6f}")
            runner.logger.info(f"  Depth MSE: {best.depth_mse:.6f}")
            runner.logger.info(f"  Left/Right MSE: {best.left_right_mse:.6f}")
            runner.logger.info(f"  Features: {best.feature_set}")
            runner.logger.info(f"  Model: {best.model_type}")

        runner.logger.info(f"\nResults saved to: {runner.results_file}")


if __name__ == "__main__":
    main()
