"""
Experiment runner framework for the overnight experiment system.

Provides:
- Robust experiment execution with error handling
- Checkpoint/resume capability
- Detailed logging
- Result aggregation
"""

import json
import logging
import numpy as np
import pandas as pd
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_name: str
    timestamp: str
    phase: str
    model_type: str
    feature_set: str
    n_features: int
    angle_mse: float
    depth_mse: float
    left_right_mse: float
    total_scaled_mse: float
    training_time_seconds: float
    per_player: bool = False
    per_target: bool = False
    notes: str = ""
    error_message: str = ""
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["hyperparams"] = json.dumps(d["hyperparams"])
        return d


class ExperimentRunner:
    """
    Manages experiment execution with logging and checkpointing.
    """

    def __init__(
        self,
        output_dir: Path,
        log_name: str = "experiment",
        resume: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / f"{log_name}_results.csv"
        self.log_file = self.output_dir / f"{log_name}.log"
        self.checkpoint_file = self.output_dir / f"{log_name}_checkpoint.json"

        self.resume = resume
        self.completed_experiments = set()

        # Setup logging
        self.logger = self._setup_logging()

        # Load checkpoint if resuming
        if resume:
            self._load_checkpoint()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        logger = logging.getLogger(f"experiment_{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers = []

        # File handler
        fh = logging.FileHandler(self.log_file, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(ch)

        return logger

    def _load_checkpoint(self):
        """Load completed experiments from checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                    self.completed_experiments = set(data.get("completed", []))
                self.logger.info(f"Loaded checkpoint: {len(self.completed_experiments)} completed")
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load checkpoint: {e}")

        # Also check results file
        if self.results_file.exists():
            try:
                df = pd.read_csv(self.results_file)
                if "experiment_name" in df.columns:
                    self.completed_experiments.update(df["experiment_name"].tolist())
            except Exception as e:
                self.logger.warning(f"Could not load results file: {e}")

    def _save_checkpoint(self):
        """Save checkpoint to file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({"completed": list(self.completed_experiments)}, f)

    def _save_result(self, result: ExperimentResult):
        """Save single result to CSV."""
        df = pd.DataFrame([result.to_dict()])

        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

        self.completed_experiments.add(result.experiment_name)
        self._save_checkpoint()

    def is_completed(self, experiment_name: str) -> bool:
        """Check if experiment was already completed."""
        return experiment_name in self.completed_experiments

    def run_experiment(
        self,
        name: str,
        phase: str,
        experiment_fn: Callable[[], ExperimentResult],
        skip_if_completed: bool = True,
    ) -> Optional[ExperimentResult]:
        """
        Run a single experiment with error handling.

        Args:
            name: Unique experiment name
            phase: Phase name for grouping
            experiment_fn: Function that returns ExperimentResult
            skip_if_completed: Skip if already completed

        Returns:
            ExperimentResult or None if skipped/failed
        """
        if skip_if_completed and self.is_completed(name):
            self.logger.info(f"SKIP (completed): {name}")
            return None

        self.logger.info(f"START: {name}")
        start_time = time.time()

        try:
            result = experiment_fn()
            result.experiment_name = name
            result.phase = phase
            result.training_time_seconds = round(time.time() - start_time, 1)

            self._save_result(result)

            self.logger.info(
                f"DONE: {name} - MSE: {result.total_scaled_mse:.6f} "
                f"({result.training_time_seconds:.1f}s)"
            )

            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"FAIL: {name} - {error_msg}")
            self.logger.debug(traceback.format_exc())

            # Save failed result
            result = ExperimentResult(
                experiment_name=name,
                timestamp=datetime.now().isoformat(),
                phase=phase,
                model_type="",
                feature_set="",
                n_features=0,
                angle_mse=float("nan"),
                depth_mse=float("nan"),
                left_right_mse=float("nan"),
                total_scaled_mse=float("nan"),
                training_time_seconds=round(time.time() - start_time, 1),
                error_message=error_msg,
            )
            self._save_result(result)

            return result

    def log_phase_start(self, phase: str, n_experiments: int):
        """Log start of a phase."""
        self.logger.info("=" * 60)
        self.logger.info(f"PHASE: {phase} ({n_experiments} experiments)")
        self.logger.info("=" * 60)

    def log_phase_end(self, phase: str, results: List[ExperimentResult]):
        """Log end of a phase with summary."""
        valid_results = [r for r in results if not np.isnan(r.total_scaled_mse)]

        if valid_results:
            best = min(valid_results, key=lambda r: r.total_scaled_mse)
            self.logger.info(f"Phase {phase} complete:")
            self.logger.info(f"  Completed: {len(valid_results)}/{len(results)}")
            self.logger.info(f"  Best: {best.experiment_name} = {best.total_scaled_mse:.6f}")
        else:
            self.logger.info(f"Phase {phase} complete: no valid results")

    def get_results_df(self) -> pd.DataFrame:
        """Get all results as DataFrame."""
        if self.results_file.exists():
            return pd.read_csv(self.results_file)
        return pd.DataFrame()


def run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_fn: Callable[[], Any],
    scalers: Dict,
    n_folds: int = 5,
    targets: List[str] = ["angle", "depth", "left_right"],
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
        targets: Target names

    Returns:
        (per_target_mse_dict, oof_predictions)
    """
    gkf = GroupKFold(n_splits=n_folds)
    n_samples = X.shape[0]

    oof_predictions = np.zeros((n_samples, len(targets)))

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train per-target models
        for target_idx, target in enumerate(targets):
            model = model_fn()
            model.fit(X_train_scaled, y_train[:, target_idx])
            oof_predictions[val_idx, target_idx] = model.predict(X_val_scaled)

    # Compute scaled MSE per target
    per_target_mse = {}
    for target_idx, target in enumerate(targets):
        y_true = y[:, target_idx]
        y_pred = oof_predictions[:, target_idx]

        # Scale for metric
        y_true_scaled = scalers[target].transform(y_true.reshape(-1, 1)).ravel()
        y_pred_scaled = scalers[target].transform(y_pred.reshape(-1, 1)).ravel()

        per_target_mse[target] = mean_squared_error(y_true_scaled, y_pred_scaled)

    return per_target_mse, oof_predictions


def run_per_player_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    model_fn: Callable[[], Any],
    scalers: Dict,
    targets: List[str] = ["angle", "depth", "left_right"],
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Run per-player leave-one-out CV experiment.

    Each player's data is predicted using models trained on other players.

    Args:
        X: Features (n_samples, n_features)
        y: Targets (n_samples, 3)
        participant_ids: Participant IDs for each sample
        model_fn: Function that returns a fresh model instance
        scalers: Dict of target scalers
        targets: Target names

    Returns:
        (per_target_mse_dict, oof_predictions)
    """
    unique_players = np.unique(participant_ids)
    n_samples = X.shape[0]

    oof_predictions = np.zeros((n_samples, len(targets)))

    for player in unique_players:
        train_mask = participant_ids != player
        val_mask = participant_ids == player

        X_train, X_val = X[train_mask], X[val_mask]
        y_train = y[train_mask]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train per-target models
        for target_idx, target in enumerate(targets):
            model = model_fn()
            model.fit(X_train_scaled, y_train[:, target_idx])
            oof_predictions[val_mask, target_idx] = model.predict(X_val_scaled)

    # Compute scaled MSE per target
    per_target_mse = {}
    for target_idx, target in enumerate(targets):
        y_true = y[:, target_idx]
        y_pred = oof_predictions[:, target_idx]

        y_true_scaled = scalers[target].transform(y_true.reshape(-1, 1)).ravel()
        y_pred_scaled = scalers[target].transform(y_pred.reshape(-1, 1)).ravel()

        per_target_mse[target] = mean_squared_error(y_true_scaled, y_pred_scaled)

    return per_target_mse, oof_predictions


if __name__ == "__main__":
    print("Testing experiment runner...")

    # Create test runner
    test_dir = Path("/tmp/test_experiment_runner")
    runner = ExperimentRunner(test_dir, log_name="test", resume=False)

    # Run test experiment
    def test_experiment():
        time.sleep(0.1)  # Simulate work
        return ExperimentResult(
            experiment_name="test_exp",
            timestamp=datetime.now().isoformat(),
            phase="test",
            model_type="test_model",
            feature_set="test_features",
            n_features=10,
            angle_mse=0.1,
            depth_mse=0.2,
            left_right_mse=0.15,
            total_scaled_mse=0.15,
            training_time_seconds=0.1,
        )

    result = runner.run_experiment("test_exp", "test", test_experiment)
    print(f"Result: {result}")

    # Check resume
    runner2 = ExperimentRunner(test_dir, log_name="test", resume=True)
    print(f"Completed experiments after resume: {runner2.completed_experiments}")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

    print("Experiment runner test complete!")
