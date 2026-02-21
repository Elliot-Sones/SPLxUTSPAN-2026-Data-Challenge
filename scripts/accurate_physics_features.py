"""
Accurate Physics Feature Extraction - All 9 Key Features

Features to extract (with correct calculations):
1. vz_at_peak: Vertical velocity at peak upward velocity frame
2. max_vz: Maximum vertical velocity in release window
3. speed_at_release: Total speed magnitude at release
4. release_height: Wrist height ABOVE ANKLE at release (not absolute Z)
5. peak_height: Maximum wrist height ABOVE ANKLE achieved
6. release_angle: arctan(vz/horizontal_speed) with soft clamping [25-65 deg]
7. backspin: Fingertip-wrist velocity differential
8. finger_max_vz: Maximum fingertip vertical velocity
9. kinetic_chain_timing: Time delay between hip/shoulder/elbow/wrist peak velocities
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

FPS = 60
DT = 1.0 / FPS
WINDOW = 7  # Savgol window size


def get_keypoint_map(keypoint_cols):
    """Build mapping from keypoint name to column indices."""
    keypoint_map = {}
    for i, col in enumerate(keypoint_cols):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in keypoint_map:
                keypoint_map[name] = {}
            keypoint_map[name][axis] = i
    return keypoint_map


def get_positions(timeseries, keypoint_map, name, start_frame=60, end_frame=200):
    """Get position time series for a keypoint."""
    if name not in keypoint_map:
        return None, None
    km = keypoint_map[name]
    if 'x' not in km or 'y' not in km or 'z' not in km:
        return None, None

    positions = []
    frames = []
    for frame in range(start_frame, min(end_frame, timeseries.shape[0])):
        pos = np.array([
            timeseries[frame, km['x']],
            timeseries[frame, km['y']],
            timeseries[frame, km['z']]
        ])
        if not np.any(np.isnan(pos)):
            positions.append(pos)
            frames.append(frame)

    if len(positions) < 20:
        return None, None

    return np.array(positions), np.array(frames)


def compute_velocity(positions, window=WINDOW):
    """Compute smoothed velocity using Savitzky-Golay filter."""
    vx = savgol_filter(positions[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(positions[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(positions[:, 2], window, 3, deriv=1) * FPS
    return vx, vy, vz


def soft_clamp(value, low, high):
    """Soft clamp using tanh-based smooth transition."""
    mid = (low + high) / 2
    half_range = (high - low) / 2
    # Normalize to [-2, 2] range then apply tanh
    normalized = (value - mid) / half_range
    clamped = np.tanh(normalized) * half_range + mid
    return clamped


def extract_accurate_physics_features(timeseries, keypoint_map):
    """
    Extract all 9 physics features with correct calculations.
    """
    features = {}

    # Get key joint positions
    wrist_pos, wrist_frames = get_positions(timeseries, keypoint_map, 'right_wrist')
    if wrist_pos is None:
        return None

    ankle_pos, ankle_frames = get_positions(timeseries, keypoint_map, 'right_ankle')
    finger_pos, finger_frames = get_positions(timeseries, keypoint_map, 'right_third_finger_distal')
    hip_pos, hip_frames = get_positions(timeseries, keypoint_map, 'mid_hip')
    shoulder_pos, shoulder_frames = get_positions(timeseries, keypoint_map, 'right_shoulder')
    elbow_pos, elbow_frames = get_positions(timeseries, keypoint_map, 'right_elbow')

    # Compute wrist velocities
    wrist_vx, wrist_vy, wrist_vz = compute_velocity(wrist_pos)
    wrist_speed = np.sqrt(wrist_vx**2 + wrist_vy**2 + wrist_vz**2)

    # Find key frames
    peak_vz_idx = np.argmax(wrist_vz)
    peak_vz_frame = wrist_frames[peak_vz_idx]

    # =========================================================================
    # Feature 1: vz_at_peak - Vertical velocity at peak upward velocity frame
    # =========================================================================
    features['vz_at_peak'] = wrist_vz[peak_vz_idx]

    # =========================================================================
    # Feature 2: max_vz - Maximum vertical velocity in release window
    # =========================================================================
    features['max_vz'] = np.max(wrist_vz)

    # =========================================================================
    # Feature 3: speed_at_release - Total speed magnitude at release
    # =========================================================================
    features['speed_at_release'] = wrist_speed[peak_vz_idx]

    # =========================================================================
    # Feature 4: release_height - Wrist height ABOVE ANKLE at release
    # =========================================================================
    if ankle_pos is not None:
        # Find ankle height at the same time as release
        # Match frames if possible, otherwise use mean ankle height
        ankle_z_at_release = None
        for i, af in enumerate(ankle_frames):
            if af == peak_vz_frame:
                ankle_z_at_release = ankle_pos[i, 2]
                break
        if ankle_z_at_release is None:
            # Use ankle height from closest frame
            ankle_z_at_release = np.mean(ankle_pos[:, 2])

        features['release_height'] = wrist_pos[peak_vz_idx, 2] - ankle_z_at_release
    else:
        # Fallback: use absolute height minus typical ankle height (~0.3 ft)
        features['release_height'] = wrist_pos[peak_vz_idx, 2] - 0.3

    # =========================================================================
    # Feature 5: peak_height - Maximum wrist height ABOVE ANKLE achieved
    # =========================================================================
    peak_height_idx = np.argmax(wrist_pos[:, 2])
    if ankle_pos is not None:
        ankle_z_mean = np.mean(ankle_pos[:, 2])
        features['peak_height'] = wrist_pos[peak_height_idx, 2] - ankle_z_mean
    else:
        features['peak_height'] = wrist_pos[peak_height_idx, 2] - 0.3

    # =========================================================================
    # Feature 6: release_angle - arctan(vz/horizontal_speed) with soft clamp [25-65 deg]
    # =========================================================================
    horizontal_speed = np.sqrt(wrist_vx[peak_vz_idx]**2 + wrist_vy[peak_vz_idx]**2)
    if horizontal_speed > 0.1:
        raw_angle = np.degrees(np.arctan2(wrist_vz[peak_vz_idx], horizontal_speed))
    else:
        # Nearly vertical release
        raw_angle = 85.0 if wrist_vz[peak_vz_idx] > 0 else -85.0

    # Soft clamp to [25, 65] degrees
    features['release_angle_raw'] = raw_angle
    features['release_angle'] = soft_clamp(raw_angle, 25, 65)

    # =========================================================================
    # Feature 7: backspin - Fingertip-wrist velocity differential
    # =========================================================================
    if finger_pos is not None:
        finger_vx, finger_vy, finger_vz = compute_velocity(finger_pos)

        # Find finger velocity at release time
        finger_vz_at_release = None
        for i, ff in enumerate(finger_frames):
            if ff == peak_vz_frame:
                finger_vz_at_release = finger_vz[i]
                break

        if finger_vz_at_release is None:
            # Use peak finger velocity as proxy
            finger_vz_at_release = np.max(finger_vz)

        # Backspin is proportional to (finger_vz - wrist_vz)
        # Higher finger velocity relative to wrist = more backspin
        features['backspin'] = finger_vz_at_release - wrist_vz[peak_vz_idx]

        # Also compute tangential velocity for better backspin estimate
        # Backspin (rad/s) = tangential_vel / ball_radius
        ball_radius = 0.39  # Basketball radius in feet
        tangential_vel = abs(finger_vz_at_release - wrist_vz[peak_vz_idx])
        features['backspin_rads'] = tangential_vel / ball_radius
    else:
        features['backspin'] = 0.0
        features['backspin_rads'] = 0.0

    # =========================================================================
    # Feature 8: finger_max_vz - Maximum fingertip vertical velocity
    # =========================================================================
    if finger_pos is not None:
        features['finger_max_vz'] = np.max(finger_vz)
    else:
        features['finger_max_vz'] = features['max_vz']  # Fallback to wrist

    # =========================================================================
    # Feature 9: kinetic_chain_timing - Time delay between joint peak velocities
    # =========================================================================
    kinetic_chain = {}

    # Compute peak velocity times for each joint
    if hip_pos is not None:
        hip_vx, hip_vy, hip_vz = compute_velocity(hip_pos)
        hip_speed = np.sqrt(hip_vx**2 + hip_vy**2 + hip_vz**2)
        hip_peak_frame = hip_frames[np.argmax(hip_speed)]
        kinetic_chain['hip'] = hip_peak_frame

    if shoulder_pos is not None:
        shoulder_vx, shoulder_vy, shoulder_vz = compute_velocity(shoulder_pos)
        shoulder_speed = np.sqrt(shoulder_vx**2 + shoulder_vy**2 + shoulder_vz**2)
        shoulder_peak_frame = shoulder_frames[np.argmax(shoulder_speed)]
        kinetic_chain['shoulder'] = shoulder_peak_frame

    if elbow_pos is not None:
        elbow_vx, elbow_vy, elbow_vz = compute_velocity(elbow_pos)
        elbow_speed = np.sqrt(elbow_vx**2 + elbow_vy**2 + elbow_vz**2)
        elbow_peak_frame = elbow_frames[np.argmax(elbow_speed)]
        kinetic_chain['elbow'] = elbow_peak_frame

    kinetic_chain['wrist'] = peak_vz_frame

    # Compute timing delays (in frames, convert to seconds)
    if 'hip' in kinetic_chain and 'shoulder' in kinetic_chain:
        features['kc_hip_to_shoulder'] = (kinetic_chain['shoulder'] - kinetic_chain['hip']) / FPS
    else:
        features['kc_hip_to_shoulder'] = 0.0

    if 'shoulder' in kinetic_chain and 'elbow' in kinetic_chain:
        features['kc_shoulder_to_elbow'] = (kinetic_chain['elbow'] - kinetic_chain['shoulder']) / FPS
    else:
        features['kc_shoulder_to_elbow'] = 0.0

    if 'elbow' in kinetic_chain:
        features['kc_elbow_to_wrist'] = (kinetic_chain['wrist'] - kinetic_chain['elbow']) / FPS
    else:
        features['kc_elbow_to_wrist'] = 0.0

    # Total kinetic chain timing (hip to wrist)
    if 'hip' in kinetic_chain:
        features['kc_total'] = (kinetic_chain['wrist'] - kinetic_chain['hip']) / FPS
    else:
        features['kc_total'] = 0.0

    # Check if kinetic chain is in correct order (proximal to distal)
    correct_order = True
    joints_order = ['hip', 'shoulder', 'elbow', 'wrist']
    prev_time = -np.inf
    for joint in joints_order:
        if joint in kinetic_chain:
            if kinetic_chain[joint] < prev_time:
                correct_order = False
                break
            prev_time = kinetic_chain[joint]
    features['kc_correct_order'] = 1.0 if correct_order else 0.0

    return features


def main():
    print("=" * 80)
    print("ACCURATE PHYSICS FEATURE EXTRACTION")
    print("=" * 80)
    print("\nExtracting all 9 key features with correct calculations")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Extract features for all training shots
    print("\nExtracting features from training data...")
    all_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_accurate_physics_features(timeseries, keypoint_map)
        if features is None:
            continue

        features['shot_id'] = metadata['id']
        features['player_id'] = metadata['participant_id']
        features['true_angle'] = metadata['angle']
        features['true_depth'] = metadata['depth']
        features['true_lr'] = metadata['left_right']

        all_data.append(features)

        if i % 50 == 0:
            print(f"  Processed {i+1} shots...", flush=True)

    df = pd.DataFrame(all_data)
    print(f"\nTotal shots: {len(df)}")

    # List of 9 key features
    key_features = [
        'vz_at_peak',
        'max_vz',
        'speed_at_release',
        'release_height',
        'peak_height',
        'release_angle',
        'backspin',
        'finger_max_vz',
        'kc_total'  # Main kinetic chain timing
    ]

    # Additional features
    extra_features = [
        'release_angle_raw',
        'backspin_rads',
        'kc_hip_to_shoulder',
        'kc_shoulder_to_elbow',
        'kc_elbow_to_wrist',
        'kc_correct_order'
    ]

    # Print feature statistics
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    for feat in key_features + extra_features:
        if feat in df.columns:
            vals = df[feat].dropna()
            print(f"{feat:25s}: mean={vals.mean():8.3f}, std={vals.std():8.3f}, "
                  f"min={vals.min():8.3f}, max={vals.max():8.3f}")

    # Correlation analysis with targets
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS WITH TARGETS")
    print("=" * 80)

    targets = ['true_angle', 'true_depth', 'true_lr']

    for target in targets:
        print(f"\n{target.upper()}:")
        print("-" * 50)

        correlations = []
        for feat in key_features + extra_features:
            if feat in df.columns:
                valid = df[[feat, target]].dropna()
                if len(valid) > 10:
                    r, p = pearsonr(valid[feat], valid[target])
                    correlations.append((feat, r, p))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        for feat, r, p in correlations:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:25s}: r={r:+.4f} (p={p:.4f}) {sig}")

    # Per-player correlation analysis
    print("\n" + "=" * 80)
    print("PER-PLAYER CORRELATION (vz_at_peak vs angle)")
    print("=" * 80)

    for player_id in sorted(df['player_id'].unique()):
        player_df = df[df['player_id'] == player_id]
        valid = player_df[['vz_at_peak', 'true_angle']].dropna()
        if len(valid) > 5:
            r, p = pearsonr(valid['vz_at_peak'], valid['true_angle'])
            print(f"  Player {player_id}: r={r:+.4f} (n={len(valid)})")

    # Save features
    output_path = PROJECT_DIR / "output" / "accurate_physics_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")

    # Summary of key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Best feature for each target
    for target in targets:
        correlations = []
        for feat in key_features:
            if feat in df.columns:
                valid = df[[feat, target]].dropna()
                if len(valid) > 10:
                    r, _ = pearsonr(valid[feat], valid[target])
                    correlations.append((feat, r))

        if correlations:
            best_feat, best_r = max(correlations, key=lambda x: abs(x[1]))
            print(f"  {target}: Best feature = {best_feat} (r={best_r:+.4f})")


if __name__ == "__main__":
    main()
