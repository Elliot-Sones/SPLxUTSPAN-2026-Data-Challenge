"""
FFT (Frequency Domain) Feature Extraction for Bowling Motion Analysis.

Extracts spectral features from motion capture time series to capture:
- Dominant frequencies (rhythm/timing patterns)
- Spectral centroid (center of mass of spectrum)
- Band powers (low freq = gross motion, high freq = tremor/noise)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Constants
FRAME_RATE = 60  # Hz
NUM_FRAMES = 240
NUM_KEYPOINTS = 69
COORDS_PER_KEYPOINT = 3
FEATURES_PER_FRAME = NUM_KEYPOINTS * COORDS_PER_KEYPOINT  # 207

# Frequency bands (in Hz)
LOW_FREQ_RANGE = (0, 5)      # Gross body motion
MID_FREQ_RANGE = (5, 15)     # Fine motor control
HIGH_FREQ_RANGE = (15, 30)   # Tremor/noise

# Key body parts for FFT analysis
KEY_KEYPOINTS = {
    'right_wrist': 16,
    'right_elbow': 14,
    'right_shoulder': 12,
    'left_wrist': 15,
    'left_elbow': 13,
    'left_shoulder': 11,
    'right_hip': 24,
    'left_hip': 23,
    'right_knee': 26,
    'left_knee': 25,
    'right_ankle': 28,
    'left_ankle': 27,
    'nose': 0,
    'mid_hip': 0,  # Will compute as average of left/right
}


def get_frequency_bins(n_samples: int, sample_rate: float) -> np.ndarray:
    """Get frequency values for FFT bins."""
    return np.fft.rfftfreq(n_samples, d=1/sample_rate)


def compute_band_power(fft_magnitudes: np.ndarray, freqs: np.ndarray,
                       freq_range: Tuple[float, float]) -> float:
    """Compute power in a specific frequency band."""
    mask = (freqs >= freq_range[0]) & (freqs < freq_range[1])
    if not mask.any():
        return 0.0
    return np.sum(fft_magnitudes[mask] ** 2)


def extract_fft_features_single(timeseries: np.ndarray) -> Dict[str, float]:
    """
    Extract FFT features from a single 1D timeseries.

    Args:
        timeseries: 1D array of shape (n_frames,)

    Returns:
        Dictionary of spectral features
    """
    # Handle NaN values
    if np.any(np.isnan(timeseries)):
        # Interpolate NaNs
        valid_mask = ~np.isnan(timeseries)
        if not valid_mask.any():
            return {
                'dominant_freq': 0.0,
                'spectral_centroid': 0.0,
                'spectral_bandwidth': 0.0,
                'low_freq_power': 0.0,
                'mid_freq_power': 0.0,
                'high_freq_power': 0.0,
                'spectral_entropy': 0.0,
                'spectral_flatness': 0.0,
            }
        indices = np.arange(len(timeseries))
        timeseries = np.interp(indices, indices[valid_mask], timeseries[valid_mask])

    # Remove DC component (mean)
    timeseries = timeseries - np.mean(timeseries)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(timeseries))
    timeseries_windowed = timeseries * window

    # Compute FFT
    fft_result = np.fft.rfft(timeseries_windowed)
    fft_magnitudes = np.abs(fft_result)
    freqs = get_frequency_bins(len(timeseries), FRAME_RATE)

    # Skip DC component (index 0)
    fft_magnitudes = fft_magnitudes[1:]
    freqs = freqs[1:]

    if len(fft_magnitudes) == 0 or np.sum(fft_magnitudes) == 0:
        return {
            'dominant_freq': 0.0,
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'low_freq_power': 0.0,
            'mid_freq_power': 0.0,
            'high_freq_power': 0.0,
            'spectral_entropy': 0.0,
            'spectral_flatness': 0.0,
        }

    # Dominant frequency
    dominant_idx = np.argmax(fft_magnitudes)
    dominant_freq = freqs[dominant_idx]

    # Spectral centroid (center of mass)
    total_magnitude = np.sum(fft_magnitudes)
    spectral_centroid = np.sum(freqs * fft_magnitudes) / total_magnitude

    # Spectral bandwidth (spread around centroid)
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * fft_magnitudes) / total_magnitude
    )

    # Band powers
    low_freq_power = compute_band_power(fft_magnitudes, freqs, LOW_FREQ_RANGE)
    mid_freq_power = compute_band_power(fft_magnitudes, freqs, MID_FREQ_RANGE)
    high_freq_power = compute_band_power(fft_magnitudes, freqs, HIGH_FREQ_RANGE)

    # Normalize powers
    total_power = low_freq_power + mid_freq_power + high_freq_power
    if total_power > 0:
        low_freq_power /= total_power
        mid_freq_power /= total_power
        high_freq_power /= total_power

    # Spectral entropy (measure of spectrum flatness/complexity)
    power_spectrum = fft_magnitudes ** 2
    power_spectrum_norm = power_spectrum / np.sum(power_spectrum)
    # Avoid log(0)
    power_spectrum_norm = np.clip(power_spectrum_norm, 1e-10, None)
    spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm))
    # Normalize by max entropy
    max_entropy = np.log2(len(power_spectrum_norm))
    if max_entropy > 0:
        spectral_entropy /= max_entropy

    # Spectral flatness (geometric mean / arithmetic mean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometric_mean = np.exp(np.mean(np.log(power_spectrum_norm + 1e-10)))
    arithmetic_mean = np.mean(power_spectrum_norm)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

    return {
        'dominant_freq': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'low_freq_power': low_freq_power,
        'mid_freq_power': mid_freq_power,
        'high_freq_power': high_freq_power,
        'spectral_entropy': spectral_entropy,
        'spectral_flatness': spectral_flatness,
    }


def extract_fft_features_keypoint(sequence: np.ndarray, keypoint_idx: int,
                                   keypoint_name: str) -> Dict[str, float]:
    """
    Extract FFT features for a specific keypoint across x, y, z axes.

    Args:
        sequence: Shape (n_frames, n_features) where n_features = 207
        keypoint_idx: Index of keypoint (0-68)
        keypoint_name: Name of keypoint for feature naming

    Returns:
        Dictionary of features with keypoint-prefixed names
    """
    features = {}

    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        col_idx = keypoint_idx * 3 + axis_idx
        timeseries = sequence[:, col_idx]

        axis_features = extract_fft_features_single(timeseries)

        for feat_name, feat_value in axis_features.items():
            features[f'fft_{keypoint_name}_{axis_name}_{feat_name}'] = feat_value

    # Also compute magnitude-based FFT
    x = sequence[:, keypoint_idx * 3]
    y = sequence[:, keypoint_idx * 3 + 1]
    z = sequence[:, keypoint_idx * 3 + 2]

    # Handle NaNs in magnitude calculation
    magnitude = np.sqrt(np.nan_to_num(x**2) + np.nan_to_num(y**2) + np.nan_to_num(z**2))
    mag_features = extract_fft_features_single(magnitude)

    for feat_name, feat_value in mag_features.items():
        features[f'fft_{keypoint_name}_mag_{feat_name}'] = feat_value

    return features


def extract_all_fft_features(sequence: np.ndarray) -> Dict[str, float]:
    """
    Extract FFT features for all key body parts.

    Args:
        sequence: Shape (n_frames, 207) - raw motion capture data

    Returns:
        Dictionary of all FFT features
    """
    features = {}

    # Extract for each key keypoint
    for keypoint_name, keypoint_idx in KEY_KEYPOINTS.items():
        if keypoint_name == 'mid_hip':
            # Compute mid hip as average of left and right hip
            left_hip_idx = KEY_KEYPOINTS['left_hip']
            right_hip_idx = KEY_KEYPOINTS['right_hip']

            # Create synthetic mid_hip sequence
            mid_hip_seq = np.zeros((sequence.shape[0], 3))
            for axis in range(3):
                left_col = left_hip_idx * 3 + axis
                right_col = right_hip_idx * 3 + axis
                mid_hip_seq[:, axis] = (sequence[:, left_col] + sequence[:, right_col]) / 2

            # Extract features manually for mid_hip
            for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                axis_features = extract_fft_features_single(mid_hip_seq[:, axis_idx])
                for feat_name, feat_value in axis_features.items():
                    features[f'fft_mid_hip_{axis_name}_{feat_name}'] = feat_value

            # Magnitude
            magnitude = np.sqrt(np.sum(mid_hip_seq**2, axis=1))
            mag_features = extract_fft_features_single(magnitude)
            for feat_name, feat_value in mag_features.items():
                features[f'fft_mid_hip_mag_{feat_name}'] = feat_value
        else:
            kp_features = extract_fft_features_keypoint(sequence, keypoint_idx, keypoint_name)
            features.update(kp_features)

    # Add cross-keypoint frequency relationships
    # Compare dominant frequencies between body parts
    right_wrist_dom = features.get('fft_right_wrist_mag_dominant_freq', 0)
    right_elbow_dom = features.get('fft_right_elbow_mag_dominant_freq', 0)
    right_shoulder_dom = features.get('fft_right_shoulder_mag_dominant_freq', 0)

    features['fft_arm_freq_consistency'] = 1.0 / (1.0 + np.std([right_wrist_dom, right_elbow_dom, right_shoulder_dom]))

    # Ratio of wrist to shoulder centroid (higher = more wrist activity relative to shoulder)
    wrist_centroid = features.get('fft_right_wrist_mag_spectral_centroid', 0)
    shoulder_centroid = features.get('fft_right_shoulder_mag_spectral_centroid', 0)
    if shoulder_centroid > 0:
        features['fft_wrist_shoulder_centroid_ratio'] = wrist_centroid / shoulder_centroid
    else:
        features['fft_wrist_shoulder_centroid_ratio'] = 0

    return features


def extract_phase_fft_features(sequence: np.ndarray,
                               phase_ranges: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
    """
    Extract FFT features for specific motion phases.

    Args:
        sequence: Shape (n_frames, 207)
        phase_ranges: List of (start_frame, end_frame) tuples
                     Default: [(0, 60), (60, 120), (120, 180), (180, 240)]

    Returns:
        Dictionary of phase-specific FFT features
    """
    if phase_ranges is None:
        phase_ranges = [
            (0, 60),      # Preparation
            (60, 120),    # Loading
            (120, 180),   # Propulsion
            (180, 240),   # Release/Follow-through
        ]

    features = {}

    # Focus on right wrist for phase analysis (most predictive)
    right_wrist_idx = KEY_KEYPOINTS['right_wrist']

    for phase_idx, (start, end) in enumerate(phase_ranges):
        phase_seq = sequence[start:end, :]

        if len(phase_seq) < 10:  # Need minimum samples for FFT
            continue

        # Extract magnitude timeseries for right wrist
        x = phase_seq[:, right_wrist_idx * 3]
        y = phase_seq[:, right_wrist_idx * 3 + 1]
        z = phase_seq[:, right_wrist_idx * 3 + 2]
        magnitude = np.sqrt(np.nan_to_num(x**2) + np.nan_to_num(y**2) + np.nan_to_num(z**2))

        phase_fft = extract_fft_features_single(magnitude)

        for feat_name, feat_value in phase_fft.items():
            features[f'fft_phase{phase_idx}_wrist_{feat_name}'] = feat_value

    return features


def get_fft_feature_names() -> List[str]:
    """Return list of all FFT feature names (for reference)."""
    feature_names = []

    # Per-keypoint features
    for keypoint_name in KEY_KEYPOINTS.keys():
        for axis in ['x', 'y', 'z', 'mag']:
            for metric in ['dominant_freq', 'spectral_centroid', 'spectral_bandwidth',
                          'low_freq_power', 'mid_freq_power', 'high_freq_power',
                          'spectral_entropy', 'spectral_flatness']:
                feature_names.append(f'fft_{keypoint_name}_{axis}_{metric}')

    # Cross-keypoint features
    feature_names.extend([
        'fft_arm_freq_consistency',
        'fft_wrist_shoulder_centroid_ratio',
    ])

    # Phase features
    for phase_idx in range(4):
        for metric in ['dominant_freq', 'spectral_centroid', 'spectral_bandwidth',
                      'low_freq_power', 'mid_freq_power', 'high_freq_power',
                      'spectral_entropy', 'spectral_flatness']:
            feature_names.append(f'fft_phase{phase_idx}_wrist_{metric}')

    return feature_names


if __name__ == '__main__':
    # Test with random data
    np.random.seed(42)
    test_sequence = np.random.randn(240, 207)

    features = extract_all_fft_features(test_sequence)
    phase_features = extract_phase_fft_features(test_sequence)
    features.update(phase_features)

    print(f"Total FFT features: {len(features)}")
    print("\nSample features:")
    for name, value in list(features.items())[:10]:
        print(f"  {name}: {value:.4f}")
