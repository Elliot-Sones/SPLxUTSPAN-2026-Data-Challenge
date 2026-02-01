"""
Fourier Phase Features (Plan 2.5)

Current FFT extracts magnitude only - phase contains timing information.
Add phase at dominant frequency and cross-keypoint phase differences.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import load_metadata, iterate_shots, load_scalers, NUM_FRAMES
from data_loader import get_keypoint_columns
import feature_engineering

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

# Key body parts for phase analysis
KEY_KEYPOINTS = [
    "right_wrist", "right_elbow", "right_shoulder",
    "right_hip", "right_knee", "right_ankle",
    "left_wrist", "neck", "mid_hip"
]


def get_keypoint_data(timeseries: np.ndarray, keypoint: str) -> np.ndarray:
    """Extract x, y, z data for a specific keypoint."""
    if KEYPOINT_INDEX is None:
        return None
    if keypoint not in KEYPOINT_INDEX:
        return None
    idx = KEYPOINT_INDEX[keypoint]
    return timeseries[:, idx*3:(idx+1)*3]


def extract_fft_features(signal: np.ndarray, prefix: str) -> dict:
    """Extract both magnitude and phase FFT features."""
    features = {}

    # Handle NaN
    signal = np.nan_to_num(signal, nan=0.0)

    # Remove mean (DC component)
    signal = signal - np.mean(signal)

    # FFT
    fft_result = np.fft.fft(signal)
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/60)  # 60 Hz sampling

    # Only positive frequencies
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_result[pos_mask]

    # Magnitude and phase
    magnitudes = np.abs(pos_fft)
    phases = np.angle(pos_fft)

    # Find dominant frequency (excluding very low frequencies)
    valid_mask = pos_freqs > 0.5  # Ignore frequencies below 0.5 Hz
    if np.any(valid_mask):
        valid_mags = magnitudes[valid_mask]
        valid_phases = phases[valid_mask]
        valid_freqs = pos_freqs[valid_mask]

        dom_idx = np.argmax(valid_mags)
        dom_freq = valid_freqs[dom_idx]
        dom_mag = valid_mags[dom_idx]
        dom_phase = valid_phases[dom_idx]

        features[f"{prefix}_dom_freq"] = dom_freq
        features[f"{prefix}_dom_mag"] = dom_mag
        features[f"{prefix}_dom_phase"] = dom_phase

        # Second dominant frequency
        if len(valid_mags) > 1:
            valid_mags_copy = valid_mags.copy()
            valid_mags_copy[dom_idx] = 0
            second_idx = np.argmax(valid_mags_copy)
            features[f"{prefix}_second_freq"] = valid_freqs[second_idx]
            features[f"{prefix}_second_mag"] = valid_mags_copy[second_idx]
            features[f"{prefix}_second_phase"] = valid_phases[second_idx]
    else:
        features[f"{prefix}_dom_freq"] = 0
        features[f"{prefix}_dom_mag"] = 0
        features[f"{prefix}_dom_phase"] = 0

    # Spectral centroid
    if np.sum(magnitudes) > 0:
        features[f"{prefix}_spectral_centroid"] = np.sum(pos_freqs * magnitudes) / np.sum(magnitudes)
    else:
        features[f"{prefix}_spectral_centroid"] = 0

    # Total energy
    features[f"{prefix}_fft_energy"] = np.sum(magnitudes ** 2)

    # Phase spread (variance of phase)
    if len(phases) > 0:
        # Circular variance for phase
        features[f"{prefix}_phase_spread"] = 1 - np.abs(np.mean(np.exp(1j * phases)))
    else:
        features[f"{prefix}_phase_spread"] = 0

    return features


def extract_cross_phase_features(timeseries: np.ndarray) -> dict:
    """Extract phase differences between keypoints."""
    features = {}

    # Key pairs for phase comparison (kinetic chain order)
    phase_pairs = [
        ("right_ankle", "right_knee"),
        ("right_knee", "right_hip"),
        ("right_hip", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("right_hip", "left_wrist"),  # Guide hand timing
    ]

    for kp1, kp2 in phase_pairs:
        data1 = get_keypoint_data(timeseries, kp1)
        data2 = get_keypoint_data(timeseries, kp2)

        if data1 is None or data2 is None:
            continue

        # Use z-coordinate (vertical movement)
        sig1 = data1[:, 2]
        sig2 = data2[:, 2]

        # Handle NaN
        sig1 = np.nan_to_num(sig1, nan=0.0)
        sig2 = np.nan_to_num(sig2, nan=0.0)

        # Remove mean
        sig1 = sig1 - np.mean(sig1)
        sig2 = sig2 - np.mean(sig2)

        # Cross-correlation to find lag
        cross_corr = np.correlate(sig1, sig2, mode="full")
        lags = np.arange(-len(sig1) + 1, len(sig1))
        max_idx = np.argmax(cross_corr)
        lag_frames = lags[max_idx]
        lag_seconds = lag_frames / 60.0  # Convert to seconds

        features[f"phase_lag_{kp1}_{kp2}"] = lag_seconds
        features[f"phase_corr_{kp1}_{kp2}"] = cross_corr[max_idx] / (np.std(sig1) * np.std(sig2) * len(sig1) + 1e-6)

        # Phase difference at dominant frequency
        fft1 = np.fft.fft(sig1)
        fft2 = np.fft.fft(sig2)
        freqs = np.fft.fftfreq(len(sig1), d=1/60)

        # Find dominant frequency in combined signal
        combined_mag = np.abs(fft1) + np.abs(fft2)
        pos_mask = freqs > 0.5
        if np.any(pos_mask):
            pos_mags = combined_mag[pos_mask]
            pos_freqs = freqs[pos_mask]
            dom_idx = np.argmax(pos_mags)
            dom_freq = pos_freqs[dom_idx]

            # Find index of dominant frequency in original FFT
            freq_idx = np.argmin(np.abs(freqs - dom_freq))

            phase1 = np.angle(fft1[freq_idx])
            phase2 = np.angle(fft2[freq_idx])
            phase_diff = np.angle(np.exp(1j * (phase1 - phase2)))  # Wrap to [-pi, pi]

            features[f"fft_phase_diff_{kp1}_{kp2}"] = phase_diff
            features[f"fft_dom_freq_{kp1}_{kp2}"] = dom_freq

    return features


def extract_all_fourier_features(timeseries: np.ndarray, participant_id: int) -> dict:
    """Extract all Fourier-based features."""
    features = {"participant_id": participant_id}

    # FFT features for key keypoints
    for kp in KEY_KEYPOINTS:
        data = get_keypoint_data(timeseries, kp)
        if data is None:
            continue

        for coord_idx, coord in enumerate(["x", "y", "z"]):
            signal = data[:, coord_idx]
            fft_feats = extract_fft_features(signal, f"{kp}_{coord}")
            features.update(fft_feats)

    # Cross-phase features
    cross_phase_feats = extract_cross_phase_features(timeseries)
    features.update(cross_phase_feats)

    return features


def build_feature_matrix(train: bool = True):
    """Build feature matrix with Fourier phase features."""
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    meta_df = load_metadata(train)
    n_shots = len(meta_df)

    all_features = []
    targets = []
    participant_ids = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i % 50 == 0:
            print(f"  Processing shot {i+1}/{n_shots}...")

        features = extract_all_fourier_features(timeseries, metadata["participant_id"])
        all_features.append(features)
        participant_ids.append(metadata["participant_id"])

        if train:
            targets.append([metadata["angle"], metadata["depth"], metadata["left_right"]])

    feature_df = pd.DataFrame(all_features)
    feature_df = feature_df.fillna(0)

    # Drop constant columns
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]

    X = feature_df.values
    y = np.array(targets) if train else None
    pids = np.array(participant_ids)

    return X, y, pids, feature_df.columns.tolist()


def check_feature_drift(train_features: pd.DataFrame, test_features: pd.DataFrame) -> pd.DataFrame:
    """Check drift of Fourier features."""
    results = []

    for col in train_features.columns:
        if col == "participant_id":
            continue

        train_vals = train_features[col].values
        test_vals = test_features[col].values

        train_mean = np.mean(train_vals)
        test_mean = np.mean(test_vals)
        train_std = np.std(train_vals)

        if train_std > 0:
            mean_shift_z = np.abs(test_mean - train_mean) / train_std
        else:
            mean_shift_z = 0

        results.append({
            "feature": col,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "mean_shift_z": mean_shift_z
        })

    return pd.DataFrame(results).sort_values("mean_shift_z", ascending=False)


def evaluate_lopo_cv(X, y, pids, alpha=100.0):
    """LOPO CV evaluation."""
    scalers_dict = load_scalers()

    unique_pids = np.unique(pids)
    all_preds = np.zeros_like(y)

    for pid in unique_pids:
        train_mask = pids != pid
        test_mask = pids == pid

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for target_idx in range(3):
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train[:, target_idx])
            all_preds[test_mask, target_idx] = model.predict(X_test_scaled)

    # Calculate scaled MSE
    mse_per_target = {}
    for i, target in enumerate(["angle", "depth", "left_right"]):
        scaler = scalers_dict[target]
        y_scaled = scaler.transform(y[:, i].reshape(-1, 1)).ravel()
        pred_scaled = scaler.transform(all_preds[:, i].reshape(-1, 1)).ravel()
        mse_per_target[target] = np.mean((y_scaled - pred_scaled) ** 2)

    total_mse = np.mean(list(mse_per_target.values()))
    return total_mse, mse_per_target


def main():
    print("=" * 60)
    print("FOURIER PHASE FEATURES (Plan 2.5)")
    print("=" * 60)

    print("\nBuilding Fourier feature matrix...")
    X, y, pids, feature_names = build_feature_matrix(train=True)
    print(f"X shape: {X.shape}")
    print(f"Number of Fourier features: {len(feature_names)}")

    # Check how many phase-related features we have
    phase_features = [f for f in feature_names if "phase" in f.lower()]
    print(f"Phase-related features: {len(phase_features)}")
    print(f"  Examples: {phase_features[:5]}")

    results = []

    # Test with different regularization
    for alpha in [10, 50, 100, 200, 500]:
        lopo_mse, per_target = evaluate_lopo_cv(X, y, pids, alpha=alpha)

        results.append({
            "alpha": alpha,
            "lopo_mse": lopo_mse,
            "angle": per_target["angle"],
            "depth": per_target["depth"],
            "left_right": per_target["left_right"],
        })

        print(f"alpha={alpha}: LOPO MSE={lopo_mse:.6f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "fourier_phase_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'fourier_phase_results.csv'}")

    # Check feature drift for new features
    print("\nChecking feature drift...")

    # Build test features
    keypoint_cols = get_keypoint_columns()
    init_keypoint_mapping(keypoint_cols)

    train_features_list = []
    test_features_list = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_all_fourier_features(timeseries, metadata["participant_id"])
        train_features_list.append(features)

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=False)):
        features = extract_all_fourier_features(timeseries, metadata["participant_id"])
        test_features_list.append(features)

    train_df = pd.DataFrame(train_features_list).fillna(0)
    test_df = pd.DataFrame(test_features_list).fillna(0)

    drift_df = check_feature_drift(train_df, test_df)
    print("\nTop 10 highest drift Fourier features:")
    print(drift_df.head(10).to_string(index=False))

    # Check which phase features are stable
    low_drift = drift_df[drift_df["mean_shift_z"] < 0.5]
    stable_phase = [f for f in low_drift["feature"].values if "phase" in f.lower()]
    print(f"\nStable phase features (drift < 0.5): {len(stable_phase)}")
    print(f"  Examples: {stable_phase[:5]}")

    # Best config
    best_idx = results_df["lopo_mse"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n=== BEST FOURIER PHASE CONFIG ===")
    print(f"Alpha: {best['alpha']}")
    print(f"LOPO MSE: {best['lopo_mse']:.6f}")


if __name__ == "__main__":
    main()
