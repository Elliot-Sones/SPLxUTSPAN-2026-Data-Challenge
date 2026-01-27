"""
Data augmentation functions for time series motion capture data.

Conservative augmentation settings:
- Rotation: +/- 0.1 degrees around vertical (z) axis
- Noise: std = 0.0001 * range of each feature
"""

import numpy as np
from typing import Tuple


def rotate_around_z(timeseries: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate 3D motion capture data around vertical (z) axis.

    Args:
        timeseries: Shape (240, 207) - 240 frames x 69 keypoints x 3 coords
                    Coords are interleaved: [x0, y0, z0, x1, y1, z1, ...]
        angle_degrees: Rotation angle in degrees

    Returns:
        Rotated timeseries with same shape
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Copy to avoid modifying original
    rotated = timeseries.copy()

    # Apply rotation to each keypoint (69 keypoints, 3 coords each)
    for kp_idx in range(69):
        x_col = kp_idx * 3
        y_col = kp_idx * 3 + 1
        # z stays the same

        x = timeseries[:, x_col]
        y = timeseries[:, y_col]

        # Rotation matrix for z-axis:
        # [cos -sin] [x]
        # [sin  cos] [y]
        rotated[:, x_col] = cos_a * x - sin_a * y
        rotated[:, y_col] = sin_a * x + cos_a * y

    return rotated


def add_gaussian_noise(
    timeseries: np.ndarray,
    noise_scale: float = 0.0001
) -> np.ndarray:
    """
    Add Gaussian noise scaled by feature range.

    Args:
        timeseries: Shape (240, 207)
        noise_scale: Noise std = noise_scale * range(feature)

    Returns:
        Noisy timeseries with same shape
    """
    noisy = timeseries.copy()

    for col in range(timeseries.shape[1]):
        col_data = timeseries[:, col]
        col_range = np.nanmax(col_data) - np.nanmin(col_data)

        if col_range > 0:
            noise_std = noise_scale * col_range
            noise = np.random.normal(0, noise_std, size=col_data.shape)
            noisy[:, col] = col_data + noise

    return noisy


def augment_single_sample(
    timeseries: np.ndarray,
    rotation_range: float = 0.1,
    noise_scale: float = 0.0001,
    seed: int = None
) -> np.ndarray:
    """
    Apply conservative augmentation to a single sample.

    Args:
        timeseries: Shape (240, 207)
        rotation_range: Max rotation in degrees (+/-)
        noise_scale: Noise std scale factor
        seed: Random seed for reproducibility

    Returns:
        Augmented timeseries
    """
    if seed is not None:
        np.random.seed(seed)

    # Random rotation within range
    angle = np.random.uniform(-rotation_range, rotation_range)
    augmented = rotate_around_z(timeseries, angle)

    # Add noise
    augmented = add_gaussian_noise(augmented, noise_scale)

    return augmented


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    n_augmented_per_sample: int = 1,
    rotation_range: float = 0.1,
    noise_scale: float = 0.0001,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment entire dataset.

    Args:
        X: Original timeseries data, shape (n_samples, 240, 207)
        y: Original targets, shape (n_samples, 3)
        participant_ids: Participant IDs for each sample
        n_augmented_per_sample: Number of augmented samples per original
        rotation_range: Max rotation in degrees
        noise_scale: Noise std scale factor
        random_state: Random seed

    Returns:
        X_aug: Augmented timeseries including originals, shape (n_total, 240, 207)
        y_aug: Augmented targets, shape (n_total, 3)
        participant_ids_aug: Participant IDs for augmented data
    """
    np.random.seed(random_state)

    n_samples = X.shape[0]
    n_total = n_samples * (1 + n_augmented_per_sample)

    X_aug = np.zeros((n_total, X.shape[1], X.shape[2]), dtype=X.dtype)
    y_aug = np.zeros((n_total, y.shape[1]), dtype=y.dtype)
    participant_ids_aug = np.zeros(n_total, dtype=participant_ids.dtype)

    # First: copy originals
    X_aug[:n_samples] = X
    y_aug[:n_samples] = y
    participant_ids_aug[:n_samples] = participant_ids

    # Then: create augmented versions
    aug_idx = n_samples
    for i in range(n_samples):
        for j in range(n_augmented_per_sample):
            seed = random_state + i * n_augmented_per_sample + j
            X_aug[aug_idx] = augment_single_sample(
                X[i],
                rotation_range=rotation_range,
                noise_scale=noise_scale,
                seed=seed
            )
            y_aug[aug_idx] = y[i]  # Same target
            participant_ids_aug[aug_idx] = participant_ids[i]
            aug_idx += 1

    return X_aug, y_aug, participant_ids_aug


if __name__ == "__main__":
    # Quick test
    print("Testing augmentation functions...")

    # Create dummy timeseries
    ts = np.random.randn(240, 207).astype(np.float32)
    y = np.random.randn(3).astype(np.float32)

    # Test rotation
    rotated = rotate_around_z(ts, 0.1)
    print(f"[OK] Rotation: input shape {ts.shape} -> output shape {rotated.shape}")
    print(f"     Max diff in z coords: {np.abs(ts[:, 2::3] - rotated[:, 2::3]).max():.6f}")

    # Test noise
    noisy = add_gaussian_noise(ts, 0.0001)
    print(f"[OK] Noise: max diff {np.abs(ts - noisy).max():.6f}")

    # Test single sample augmentation
    aug = augment_single_sample(ts, seed=42)
    print(f"[OK] Single sample augmentation: shape {aug.shape}")

    # Test dataset augmentation
    X = np.random.randn(10, 240, 207).astype(np.float32)
    y = np.random.randn(10, 3).astype(np.float32)
    pids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    X_aug, y_aug, pids_aug = augment_dataset(X, y, pids, n_augmented_per_sample=1)
    print(f"[OK] Dataset augmentation: {X.shape} -> {X_aug.shape}")
    print(f"     Targets: {y.shape} -> {y_aug.shape}")

    print("\nAugmentation test complete.")
