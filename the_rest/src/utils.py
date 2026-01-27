"""
Utility functions for SPLxUTSPAN 2026 Data Challenge.
"""

import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth (n_samples, 3)
        y_pred: Predictions (n_samples, 3)

    Returns:
        Dictionary with MSE metrics
    """
    target_names = ["angle", "depth", "left_right"]

    metrics = {
        "total_mse": mean_squared_error(y_true, y_pred),
    }

    for i, name in enumerate(target_names):
        metrics[f"{name}_mse"] = mean_squared_error(y_true[:, i], y_pred[:, i])

    return metrics


def print_metrics(metrics: dict):
    """Print metrics in a formatted way."""
    print("\nEvaluation Metrics:")
    print(f"  Total MSE: {metrics['total_mse']:.6f}")
    print(f"  Angle MSE: {metrics['angle_mse']:.6f}")
    print(f"  Depth MSE: {metrics['depth_mse']:.6f}")
    print(f"  Left/Right MSE: {metrics['left_right_mse']:.6f}")


def get_key_keypoint_indices() -> List[int]:
    """
    Get indices of key keypoints for analysis.

    Returns indices for: right_wrist, right_elbow, right_shoulder,
    right_hip, right_knee, neck, mid_hip
    """
    # These are approximate - actual indices depend on column ordering
    # Based on typical ordering: each keypoint has 3 columns (x, y, z)
    # right_wrist = index 10 (columns 30, 31, 32)
    # right_elbow = index 8 (columns 24, 25, 26)
    # right_shoulder = index 6 (columns 18, 19, 20)

    key_keypoints = [
        10,  # right_wrist
        8,   # right_elbow
        6,   # right_shoulder
        12,  # right_hip
        14,  # right_knee
        22,  # neck
        21,  # mid_hip
    ]

    # Convert to column indices (each keypoint has 3 columns)
    indices = []
    for kp in key_keypoints:
        indices.extend([kp * 3, kp * 3 + 1, kp * 3 + 2])

    return indices


def describe_data_stats(X: np.ndarray, y: np.ndarray = None):
    """Print basic statistics about the data."""
    print(f"Feature matrix shape: {X.shape}")
    print(f"Feature dtype: {X.dtype}")
    print(f"NaN count: {np.isnan(X).sum()}")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")

    if y is not None:
        print(f"\nTarget matrix shape: {y.shape}")
        target_names = ["angle", "depth", "left_right"]
        for i, name in enumerate(target_names):
            print(f"  {name}: mean={y[:, i].mean():.2f}, std={y[:, i].std():.2f}, "
                  f"range=[{y[:, i].min():.2f}, {y[:, i].max():.2f}]")
