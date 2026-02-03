"""
Focus on what PREDICTS the targets, not theoretical velocity requirements.

The targets are:
- angle: entry angle (degrees) - how steep the ball enters
- depth: front-to-back position (inches) - short vs long
- left_right: lateral position (inches) - left vs right

All shots in training data reached the hoop. The question is:
what in the motion data predicts WHERE it landed?

We should extract MANY features and see which correlate with targets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

HOOP_POSITION = np.array([5.25, -25.0, 10.0])
FPS = 60


def get_keypoint_map(keypoint_cols):
    keypoint_map = {}
    for i, col in enumerate(keypoint_cols):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in keypoint_map:
                keypoint_map[name] = {}
            keypoint_map[name][axis] = i
    return keypoint_map


def get_position(timeseries, keypoint_map, name, frame):
    if name not in keypoint_map:
        return None
    km = keypoint_map[name]
    if 'x' not in km or 'y' not in km or 'z' not in km:
        return None
    pos = np.array([
        timeseries[frame, km['x']],
        timeseries[frame, km['y']],
        timeseries[frame, km['z']]
    ])
    if np.any(np.isnan(pos)):
        return None
    return pos


def extract_comprehensive_features(timeseries, keypoint_map):
    """
    Extract many features that might predict the targets.
    Don't assume what's important - measure everything and check correlations.
    """
    features = {}

    # Get positions for key joints across time
    joints = [
        'right_shoulder', 'right_elbow', 'right_wrist',
        'right_second_finger_distal', 'right_third_finger_distal',
    ]

    joint_data = {}
    for joint in joints:
        positions = []
        valid_frames = []
        for frame in range(60, 200):
            pos = get_position(timeseries, keypoint_map, joint, frame)
            if pos is not None:
                positions.append(pos)
                valid_frames.append(frame)
        if len(positions) > 20:
            joint_data[joint] = {
                'pos': np.array(positions),
                'frames': np.array(valid_frames)
            }

    if 'right_wrist' not in joint_data:
        return None

    wrist = joint_data['right_wrist']
    pos = wrist['pos']
    frames = wrist['frames']

    # Compute velocities
    window = 7
    vx = savgol_filter(pos[:, 0], window, 3, deriv=1) * FPS
    vy = savgol_filter(pos[:, 1], window, 3, deriv=1) * FPS
    vz = savgol_filter(pos[:, 2], window, 3, deriv=1) * FPS
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    # Compute accelerations
    ax = savgol_filter(pos[:, 0], window, 3, deriv=2) * FPS * FPS
    ay = savgol_filter(pos[:, 1], window, 3, deriv=2) * FPS * FPS
    az = savgol_filter(pos[:, 2], window, 3, deriv=2) * FPS * FPS

    # Find key frames
    peak_vz_idx = np.argmax(vz)
    peak_speed_idx = np.argmax(speed)
    peak_height_idx = np.argmax(pos[:, 2])

    # Find when vx is most negative (most forward toward hoop)
    peak_forward_idx = np.argmin(vx)

    # === POSITION FEATURES ===

    # Release height (at peak vz)
    features['release_height'] = pos[peak_vz_idx, 2]

    # Peak height reached
    features['peak_height'] = pos[peak_height_idx, 2]

    # Lateral position at release (Y deviation from hoop Y)
    features['release_y_offset'] = pos[peak_vz_idx, 1] - HOOP_POSITION[1]

    # X position at release (distance from hoop)
    features['release_x_dist'] = pos[peak_vz_idx, 0] - HOOP_POSITION[0]

    # === VELOCITY FEATURES (at different key frames) ===

    # At peak upward velocity
    features['vx_at_peak_vz'] = vx[peak_vz_idx]
    features['vy_at_peak_vz'] = vy[peak_vz_idx]
    features['vz_at_peak_vz'] = vz[peak_vz_idx]
    features['speed_at_peak_vz'] = speed[peak_vz_idx]

    # At peak forward velocity
    features['vx_at_peak_forward'] = vx[peak_forward_idx]
    features['vy_at_peak_forward'] = vy[peak_forward_idx]
    features['vz_at_peak_forward'] = vz[peak_forward_idx]
    features['speed_at_peak_forward'] = speed[peak_forward_idx]

    # At peak total speed
    features['vx_at_peak_speed'] = vx[peak_speed_idx]
    features['vy_at_peak_speed'] = vy[peak_speed_idx]
    features['vz_at_peak_speed'] = vz[peak_speed_idx]

    # === VELOCITY RATIOS (direction indicators) ===

    # Release angle approximations
    horiz_vel = np.sqrt(vx[peak_vz_idx]**2 + vy[peak_vz_idx]**2)
    if horiz_vel > 0.1:
        features['release_angle_ratio'] = vz[peak_vz_idx] / horiz_vel
    else:
        features['release_angle_ratio'] = 10.0  # Very steep

    # Lateral aiming ratio
    if abs(vx[peak_vz_idx]) > 0.1:
        features['lateral_aim_ratio'] = vy[peak_vz_idx] / abs(vx[peak_vz_idx])
    else:
        features['lateral_aim_ratio'] = 0.0

    # === PEAK VALUES (maximum effort/range) ===

    features['max_vx'] = np.min(vx)  # Most negative = most toward hoop
    features['max_vy'] = np.max(np.abs(vy))
    features['max_vz'] = np.max(vz)
    features['max_speed'] = np.max(speed)

    # === TIMING FEATURES ===

    features['peak_vz_frame'] = frames[peak_vz_idx]
    features['peak_forward_frame'] = frames[peak_forward_idx]
    features['frame_diff_vz_forward'] = frames[peak_vz_idx] - frames[peak_forward_idx]

    # === ACCELERATION FEATURES ===

    features['max_az'] = np.max(az)
    features['az_at_release'] = az[peak_vz_idx]

    # === COMBINED/DERIVED FEATURES ===

    # "Energy" approximation
    features['kinetic_proxy'] = 0.5 * features['max_speed']**2

    # Velocity vector angle at different points
    features['angle_at_peak_vz'] = np.degrees(np.arctan2(vz[peak_vz_idx],
                                              -vx[peak_vz_idx] + 0.01))
    features['angle_at_peak_forward'] = np.degrees(np.arctan2(vz[peak_forward_idx],
                                                   -vx[peak_forward_idx] + 0.01))

    # === FINGER-SPECIFIC FEATURES ===

    if 'right_third_finger_distal' in joint_data:
        finger = joint_data['right_third_finger_distal']
        fpos = finger['pos']
        fframes = finger['frames']

        fvz = savgol_filter(fpos[:, 2], window, 3, deriv=1) * FPS
        fvy = savgol_filter(fpos[:, 1], window, 3, deriv=1) * FPS

        features['finger_max_vz'] = np.max(fvz)
        features['finger_release_y'] = fpos[np.argmax(fvz), 1]

    # === BODY ALIGNMENT FEATURES ===

    if 'right_shoulder' in joint_data and 'right_elbow' in joint_data:
        shoulder = joint_data['right_shoulder']
        elbow = joint_data['right_elbow']

        # At release frame
        rel_idx = peak_vz_idx
        if rel_idx < len(shoulder['pos']) and rel_idx < len(elbow['pos']):
            s = shoulder['pos'][min(rel_idx, len(shoulder['pos'])-1)]
            e = elbow['pos'][min(rel_idx, len(elbow['pos'])-1)]
            w = pos[rel_idx]

            # Upper arm angle (shoulder to elbow)
            upper_arm = e - s
            features['upper_arm_angle_z'] = np.degrees(np.arctan2(upper_arm[2],
                                           np.sqrt(upper_arm[0]**2 + upper_arm[1]**2)))

            # Forearm angle
            forearm = w - e
            features['forearm_angle_z'] = np.degrees(np.arctan2(forearm[2],
                                         np.sqrt(forearm[0]**2 + forearm[1]**2)))

    return features


def main():
    print("=" * 80)
    print("COMPREHENSIVE FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    print("\nExtracting many features and checking which predict targets")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    all_data = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        features = extract_comprehensive_features(timeseries, keypoint_map)
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

    # Analyze correlations with each target
    targets = ['true_angle', 'true_depth', 'true_lr']
    feature_cols = [c for c in df.columns if c not in ['shot_id', 'player_id'] + targets]

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    for target in targets:
        print(f"\n{target.upper()}:")
        print("-" * 40)

        correlations = []
        for feat in feature_cols:
            try:
                valid = df[[feat, target]].dropna()
                if len(valid) > 10:
                    r, p = pearsonr(valid[feat], valid[target])
                    correlations.append((feat, r, p))
            except:
                pass

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        # Show top 10
        print(f"  Top correlated features:")
        for feat, r, p in correlations[:15]:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {feat:30s}: r={r:+.3f} (p={p:.4f}) {sig}")

    # Save correlations to CSV
    output_path = PROJECT_DIR / "output" / "physics_feature_correlations.csv"
    corr_data = []
    for target in targets:
        for feat in feature_cols:
            try:
                valid = df[[feat, target]].dropna()
                if len(valid) > 10:
                    r, p = pearsonr(valid[feat], valid[target])
                    corr_data.append({
                        'target': target,
                        'feature': feat,
                        'correlation': r,
                        'p_value': p,
                        'significant': p < 0.05
                    })
            except:
                pass

    corr_df = pd.DataFrame(corr_data)
    corr_df.to_csv(output_path, index=False)
    print(f"\nCorrelations saved to {output_path}")


if __name__ == "__main__":
    main()
