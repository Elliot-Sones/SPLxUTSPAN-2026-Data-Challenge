"""
Target Mapping: Convert physics simulation outputs to target predictions.

Maps ball landing position and entry angle to scaled target values (angle, depth, left_right).
"""

import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, Optional, List, Tuple


# Court constants
HOOP_Y = 0.15   # Rim center Y position (15cm forward of backboard)
HOOP_Z = 3.05   # Rim height (10 feet)


class TargetMapper:
    """
    Maps physics simulation outputs to target predictions.

    Uses per-player Ridge regression models for the mapping.
    """

    def __init__(self):
        self.models = {}  # {player_id: {'angle': Ridge, 'depth': Ridge, 'lr': Ridge}}
        self.fitted = False

    def fit(
        self,
        physics_outputs: List[Dict],
        targets: List[Dict],
        player_ids: List[int]
    ):
        """
        Fit mapping models using training data.

        Args:
            physics_outputs: List of dicts with 'landing_y', 'landing_z', 'entry_angle'
            targets: List of dicts with 'angle', 'depth', 'left_right' (scaled 0-1)
            player_ids: List of player IDs
        """
        # Organize by player
        player_data = {i: {'X': [], 'y_angle': [], 'y_depth': [], 'y_lr': []}
                       for i in range(1, 6)}

        for phys, tgt, pid in zip(physics_outputs, targets, player_ids):
            if phys['landing_y'] is None:
                continue

            # Features for mapping
            y_dev = phys['landing_y'] - HOOP_Y
            z_dev = phys['landing_z'] - HOOP_Z
            entry = phys['entry_angle']

            features = [
                y_dev,
                z_dev,
                entry,
                y_dev ** 2,
                z_dev ** 2,
                y_dev * z_dev,
                1.0  # Bias term
            ]

            player_data[pid]['X'].append(features)
            player_data[pid]['y_angle'].append(tgt['angle'])
            player_data[pid]['y_depth'].append(tgt['depth'])
            player_data[pid]['y_lr'].append(tgt['left_right'])

        # Fit models per player
        for pid in range(1, 6):
            if len(player_data[pid]['X']) < 5:
                continue

            X = np.array(player_data[pid]['X'])
            y_angle = np.array(player_data[pid]['y_angle'])
            y_depth = np.array(player_data[pid]['y_depth'])
            y_lr = np.array(player_data[pid]['y_lr'])

            self.models[pid] = {}

            # Angle model
            model_angle = Ridge(alpha=1.0)
            model_angle.fit(X, y_angle)
            self.models[pid]['angle'] = model_angle

            # Depth model
            model_depth = Ridge(alpha=1.0)
            model_depth.fit(X, y_depth)
            self.models[pid]['depth'] = model_depth

            # Left-right model
            model_lr = Ridge(alpha=1.0)
            model_lr.fit(X, y_lr)
            self.models[pid]['lr'] = model_lr

        self.fitted = True

    def predict(
        self,
        landing_y: Optional[float],
        landing_z: Optional[float],
        entry_angle: Optional[float],
        player_id: int
    ) -> Dict[str, float]:
        """
        Predict targets from physics outputs.

        Args:
            landing_y: Ball Y position at hoop plane (meters)
            landing_z: Ball Z position at hoop plane (meters)
            entry_angle: Ball entry angle (degrees)
            player_id: Player ID

        Returns:
            Dict with 'angle', 'depth', 'left_right' predictions (scaled 0-1)
        """
        if landing_y is None or landing_z is None or entry_angle is None:
            # Simulation failed - return neutral predictions
            return {'angle': 0.5, 'depth': 0.5, 'left_right': 0.5}

        # Build feature vector
        y_dev = landing_y - HOOP_Y
        z_dev = landing_z - HOOP_Z

        features = np.array([[
            y_dev,
            z_dev,
            entry_angle,
            y_dev ** 2,
            z_dev ** 2,
            y_dev * z_dev,
            1.0
        ]])

        # Use player-specific models if available
        if player_id in self.models and self.fitted:
            pred_angle = self.models[player_id]['angle'].predict(features)[0]
            pred_depth = self.models[player_id]['depth'].predict(features)[0]
            pred_lr = self.models[player_id]['lr'].predict(features)[0]
        else:
            # Fallback to simple linear mapping
            pred_angle = 0.5 + y_dev * 0.2 + z_dev * 0.1
            pred_depth = 0.5 + z_dev * 0.5
            pred_lr = 0.5 - y_dev * 0.3

        return {
            'angle': float(np.clip(pred_angle, 0, 1)),
            'depth': float(np.clip(pred_depth, 0, 1)),
            'left_right': float(np.clip(pred_lr, 0, 1))
        }


def calibrate_mean_correction(
    predictions: List[Dict],
    actuals: List[Dict]
) -> Dict[str, float]:
    """
    Compute mean corrections to align predictions with training distribution.

    Returns:
        Dict with correction offsets for each target
    """
    corrections = {}

    for target in ['angle', 'depth', 'left_right']:
        pred_mean = np.mean([p[target] for p in predictions])
        actual_mean = np.mean([a[target] for a in actuals])
        corrections[target] = actual_mean - pred_mean

    return corrections


def apply_corrections(
    predictions: Dict[str, float],
    corrections: Dict[str, float]
) -> Dict[str, float]:
    """
    Apply mean corrections to predictions.
    """
    return {
        'angle': np.clip(predictions['angle'] + corrections.get('angle', 0), 0, 1),
        'depth': np.clip(predictions['depth'] + corrections.get('depth', 0), 0, 1),
        'left_right': np.clip(predictions['left_right'] + corrections.get('left_right', 0), 0, 1)
    }


if __name__ == "__main__":
    print("Target mapping module loaded.")
