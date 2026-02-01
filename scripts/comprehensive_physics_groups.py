"""
COMPREHENSIVE PHYSICS GROUPS

Now with full knowledge of available keypoints including FINGER data.

For each target, test physics-based groups with variations:
- Position only
- Position + Velocity
- Position + Velocity + Acceleration
- Different velocity window sizes (3, 5, 10 frames)
- Joint angles
- Distances between joints

ANGLE PHYSICS:
- Fingertip positions/velocity (the ball leaves from fingertips!)
- Wrist angle relative to forearm
- Finger spread/curl
- Release trajectory (fingertip velocity direction)

DEPTH PHYSICS:
- Leg drive (knee, ankle extension)
- Hip thrust
- Full body extension
- Follow-through velocity

LEFT_RIGHT PHYSICS:
- Body alignment (shoulder, hip symmetry)
- Elbow position relative to center
- Finger alignment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


class FeatureExtractor:
    def __init__(self, X, col_to_idx):
        self.X = X
        self.col_to_idx = col_to_idx

    def get_pos(self, sample_idx, frame, keypoint):
        """Get position of a keypoint."""
        col = f"{keypoint}"
        if col not in self.col_to_idx:
            return np.nan
        return self.X[sample_idx, frame, self.col_to_idx[col]]

    def get_pos_xyz(self, sample_idx, frame, body_part):
        """Get XYZ position of a body part."""
        return np.array([
            self.get_pos(sample_idx, frame, f"{body_part}_x"),
            self.get_pos(sample_idx, frame, f"{body_part}_y"),
            self.get_pos(sample_idx, frame, f"{body_part}_z")
        ])

    def get_velocity(self, sample_idx, frame, keypoint, window=5):
        """Get velocity (position change over window)."""
        if frame + window >= self.X.shape[1]:
            return np.nan
        p1 = self.get_pos(sample_idx, frame, keypoint)
        p2 = self.get_pos(sample_idx, frame + window, keypoint)
        return (p2 - p1) / window

    def get_acceleration(self, sample_idx, frame, keypoint, window=5):
        """Get acceleration."""
        if frame + 2 * window >= self.X.shape[1]:
            return np.nan
        v1 = self.get_velocity(sample_idx, frame, keypoint, window)
        v2 = self.get_velocity(sample_idx, frame + window, keypoint, window)
        return (v2 - v1) / window


def extract_group_features(X_player, col_to_idx, group_name, frame, variation='pos'):
    """Extract features for a specific physics group."""
    n_samples = X_player.shape[0]
    fe = FeatureExtractor(X_player, col_to_idx)
    features = []

    # Define groups
    if group_name == 'fingertip_pos':
        # Fingertip positions (critical for angle)
        keypoints = [
            'right_first_finger_distal',  # thumb tip
            'right_second_finger_distal',  # index tip
            'right_third_finger_distal',   # middle tip
        ]
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col in col_to_idx:
                    features.append([X_player[i, frame, col_to_idx[col]] for i in range(n_samples)])

    elif group_name == 'fingertip_vel':
        # Fingertip velocities (release direction!)
        keypoints = [
            'right_first_finger_distal',
            'right_second_finger_distal',
            'right_third_finger_distal',
        ]
        window = 5
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col in col_to_idx and frame + window < X_player.shape[1]:
                    vel = [(X_player[i, frame + window, col_to_idx[col]] - X_player[i, frame, col_to_idx[col]]) / window
                           for i in range(n_samples)]
                    features.append(vel)

    elif group_name == 'fingertip_full':
        # Position + Velocity + Acceleration
        keypoints = [
            'right_second_finger_distal',  # index (main shooting finger)
            'right_third_finger_distal',   # middle
        ]
        window = 5
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col not in col_to_idx:
                    continue
                idx = col_to_idx[col]

                # Position
                features.append([X_player[i, frame, idx] for i in range(n_samples)])

                # Velocity
                if frame + window < X_player.shape[1]:
                    vel = [(X_player[i, frame + window, idx] - X_player[i, frame, idx]) / window
                           for i in range(n_samples)]
                    features.append(vel)

                # Acceleration
                if frame + 2 * window < X_player.shape[1]:
                    acc = []
                    for i in range(n_samples):
                        v1 = (X_player[i, frame + window, idx] - X_player[i, frame, idx]) / window
                        v2 = (X_player[i, frame + 2*window, idx] - X_player[i, frame + window, idx]) / window
                        acc.append((v2 - v1) / window)
                    features.append(acc)

    elif group_name == 'wrist_finger_angle':
        # Angle between wrist and fingertips (wrist snap)
        for i in range(n_samples):
            wrist = fe.get_pos_xyz(i, frame, 'right_wrist')
            index_tip = fe.get_pos_xyz(i, frame, 'right_second_finger_distal')
            middle_tip = fe.get_pos_xyz(i, frame, 'right_third_finger_distal')

            # Vector from wrist to fingertips
            v_index = index_tip - wrist
            v_middle = middle_tip - wrist

            # Angle with vertical (z-axis)
            if len(features) < 6:
                features = [[] for _ in range(6)]

            for j, v in enumerate([v_index, v_middle]):
                length = np.sqrt(np.sum(v**2))
                if length > 0:
                    angle_z = np.arccos(np.clip(v[2] / length, -1, 1))
                    angle_xy = np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2))
                    azimuth = np.arctan2(v[1], v[0])
                else:
                    angle_z = angle_xy = azimuth = np.nan
                features[j*3].append(angle_z)
                features[j*3+1].append(angle_xy)
                features[j*3+2].append(azimuth)

    elif group_name == 'finger_spread':
        # Distance between fingertips (hand shape)
        for i in range(n_samples):
            thumb = fe.get_pos_xyz(i, frame, 'right_first_finger_distal')
            index = fe.get_pos_xyz(i, frame, 'right_second_finger_distal')
            middle = fe.get_pos_xyz(i, frame, 'right_third_finger_distal')
            ring = fe.get_pos_xyz(i, frame, 'right_fourth_finger_distal')
            pinky = fe.get_pos_xyz(i, frame, 'right_fifth_finger_distal')

            if len(features) < 4:
                features = [[] for _ in range(4)]

            # Distances
            features[0].append(np.sqrt(np.sum((index - thumb)**2)))
            features[1].append(np.sqrt(np.sum((middle - index)**2)))
            features[2].append(np.sqrt(np.sum((ring - middle)**2)))
            features[3].append(np.sqrt(np.sum((pinky - ring)**2)))

    elif group_name == 'arm_chain_pos':
        # Full arm kinetic chain positions
        keypoints = ['right_shoulder', 'right_elbow', 'right_wrist']
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col in col_to_idx:
                    features.append([X_player[i, frame, col_to_idx[col]] for i in range(n_samples)])

    elif group_name == 'arm_chain_full':
        # Position + Velocity + Acceleration for arm
        keypoints = ['right_shoulder', 'right_elbow', 'right_wrist']
        window = 5
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col not in col_to_idx:
                    continue
                idx = col_to_idx[col]

                features.append([X_player[i, frame, idx] for i in range(n_samples)])

                if frame + window < X_player.shape[1]:
                    vel = [(X_player[i, frame + window, idx] - X_player[i, frame, idx]) / window
                           for i in range(n_samples)]
                    features.append(vel)

                if frame + 2 * window < X_player.shape[1]:
                    acc = []
                    for i in range(n_samples):
                        v1 = (X_player[i, frame + window, idx] - X_player[i, frame, idx]) / window
                        v2 = (X_player[i, frame + 2*window, idx] - X_player[i, frame + window, idx]) / window
                        acc.append((v2 - v1) / window)
                    features.append(acc)

    elif group_name == 'elbow_angle':
        # Elbow bend angle (arm extension)
        for i in range(n_samples):
            shoulder = fe.get_pos_xyz(i, frame, 'right_shoulder')
            elbow = fe.get_pos_xyz(i, frame, 'right_elbow')
            wrist = fe.get_pos_xyz(i, frame, 'right_wrist')

            upper = shoulder - elbow
            lower = wrist - elbow

            dot = np.sum(upper * lower)
            mag = np.sqrt(np.sum(upper**2)) * np.sqrt(np.sum(lower**2))

            if len(features) < 1:
                features = [[]]

            if mag > 0:
                features[0].append(np.arccos(np.clip(dot / mag, -1, 1)))
            else:
                features[0].append(np.nan)

    elif group_name == 'leg_drive_pos':
        # Leg positions
        keypoints = ['right_hip', 'right_knee', 'right_ankle']
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col in col_to_idx:
                    features.append([X_player[i, frame, col_to_idx[col]] for i in range(n_samples)])

    elif group_name == 'leg_drive_vel':
        # Leg velocities (power generation)
        keypoints = ['right_hip', 'right_knee', 'right_ankle']
        window = 5
        for kp in keypoints:
            for axis in ['z']:  # Mainly vertical for power
                col = f"{kp}_{axis}"
                if col in col_to_idx and frame + window < X_player.shape[1]:
                    vel = [(X_player[i, frame + window, col_to_idx[col]] - X_player[i, frame, col_to_idx[col]]) / window
                           for i in range(n_samples)]
                    features.append(vel)

    elif group_name == 'leg_drive_full':
        # Position + Velocity + Acceleration for legs
        keypoints = ['right_hip', 'right_knee', 'right_ankle']
        window = 5
        for kp in keypoints:
            col = f"{kp}_z"
            if col not in col_to_idx:
                continue
            idx = col_to_idx[col]

            features.append([X_player[i, frame, idx] for i in range(n_samples)])

            if frame + window < X_player.shape[1]:
                vel = [(X_player[i, frame + window, idx] - X_player[i, frame, idx]) / window
                       for i in range(n_samples)]
                features.append(vel)

            if frame + 2 * window < X_player.shape[1]:
                acc = []
                for i in range(n_samples):
                    v1 = (X_player[i, frame + window, idx] - X_player[i, frame, idx]) / window
                    v2 = (X_player[i, frame + 2*window, idx] - X_player[i, frame + window, idx]) / window
                    acc.append((v2 - v1) / window)
                features.append(acc)

    elif group_name == 'body_alignment':
        # Left-right symmetry
        pairs = [
            ('right_shoulder', 'left_shoulder'),
            ('right_hip', 'left_hip'),
            ('right_elbow', 'left_elbow'),
        ]
        for right, left in pairs:
            for axis in ['x', 'y', 'z']:
                r_col = f"{right}_{axis}"
                l_col = f"{left}_{axis}"
                if r_col in col_to_idx and l_col in col_to_idx:
                    diff = [(X_player[i, frame, col_to_idx[r_col]] - X_player[i, frame, col_to_idx[l_col]])
                           for i in range(n_samples)]
                    features.append(diff)

    elif group_name == 'guide_hand':
        # Left hand (guide hand) position - affects ball stability
        keypoints = ['left_wrist', 'left_second_finger_distal', 'left_third_finger_distal']
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col in col_to_idx:
                    features.append([X_player[i, frame, col_to_idx[col]] for i in range(n_samples)])

    elif group_name == 'release_trajectory':
        # Combined: wrist + fingertip velocity (ball trajectory)
        keypoints = ['right_wrist', 'right_second_finger_distal']
        window = 3  # Short window for release
        for kp in keypoints:
            for axis in ['x', 'y', 'z']:
                col = f"{kp}_{axis}"
                if col in col_to_idx and frame + window < X_player.shape[1]:
                    # Position
                    features.append([X_player[i, frame, col_to_idx[col]] for i in range(n_samples)])
                    # Velocity
                    vel = [(X_player[i, frame + window, col_to_idx[col]] - X_player[i, frame, col_to_idx[col]]) / window
                           for i in range(n_samples)]
                    features.append(vel)

    if len(features) == 0:
        return None

    return np.column_stack(features)


def nested_cv_test(X_features, y, model_type='ridge'):
    """Nested CV test."""
    if X_features is None or len(y) < 15:
        return {'r2': np.nan}

    valid = ~(np.isnan(X_features).any(axis=1) | np.isnan(y))
    if valid.sum() < 15:
        return {'r2': np.nan}

    X_clean = X_features[valid]
    y_clean = y[valid]

    preds = np.zeros(len(y_clean))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X_clean):
        X_tr, X_te = X_clean[train_idx], X_clean[test_idx]
        y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=3, random_state=42)

        model.fit(X_tr_s, y_tr)
        preds[test_idx] = model.predict(X_te_s)

    mse = np.mean((preds - y_clean) ** 2)
    r2 = 1 - mse / np.var(y_clean)

    return {'r2': r2}


def main():
    print("=" * 80)
    print("COMPREHENSIVE PHYSICS GROUPS TEST")
    print("=" * 80)
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

    participant_ids = meta['participant_id'].values
    targets = {
        'angle': y[:, 0],
        'depth': y[:, 1],
        'left_right': y[:, 2]
    }

    # Define physics groups for each target
    PHYSICS_GROUPS = {
        'angle': [
            # Fingertip features (ball leaves from here!)
            ('fingertip_pos', (140, 180), 'Fingertip positions'),
            ('fingertip_vel', (145, 175), 'Fingertip velocities'),
            ('fingertip_full', (150, 170), 'Fingertip pos+vel+acc'),
            ('wrist_finger_angle', (150, 175), 'Wrist-finger angle'),
            ('finger_spread', (140, 180), 'Finger spread distances'),
            ('release_trajectory', (155, 175), 'Release trajectory'),
            # Arm features
            ('arm_chain_pos', (140, 180), 'Arm chain positions'),
            ('arm_chain_full', (140, 175), 'Arm chain pos+vel+acc'),
            ('elbow_angle', (140, 180), 'Elbow bend angle'),
            # Guide hand
            ('guide_hand', (140, 180), 'Guide hand position'),
        ],
        'depth': [
            # Leg power
            ('leg_drive_pos', (80, 150), 'Leg positions'),
            ('leg_drive_vel', (90, 140), 'Leg velocities'),
            ('leg_drive_full', (90, 140), 'Leg pos+vel+acc'),
            # Body extension
            ('arm_chain_pos', (140, 180), 'Arm extension'),
            ('arm_chain_full', (130, 170), 'Arm pos+vel+acc'),
        ],
        'left_right': [
            # Body alignment
            ('body_alignment', (140, 180), 'Body symmetry'),
            ('arm_chain_pos', (140, 180), 'Arm positions'),
            ('guide_hand', (140, 180), 'Guide hand'),
            # Finger alignment
            ('fingertip_pos', (150, 175), 'Fingertip alignment'),
        ],
    }

    results = []

    for target_name, target in targets.items():
        print()
        print("=" * 80)
        print(f"TARGET: {target_name.upper()}")
        print("=" * 80)

        groups = PHYSICS_GROUPS[target_name]

        for group_name, frame_range, description in groups:
            print(f"\n  {group_name}: {description}")
            print(f"  Frame range: {frame_range}")

            for pid in sorted(np.unique(participant_ids)):
                mask = participant_ids == pid
                X_player = X[mask]
                y_player = target[mask]

                best_r2 = -999
                best_frame = frame_range[0]
                best_model = 'ridge'

                # Search frames
                for frame in range(frame_range[0], frame_range[1], 5):
                    features = extract_group_features(X_player, col_to_idx, group_name, frame)

                    if features is None:
                        continue

                    # Test Ridge and RF
                    for model_type in ['ridge', 'rf']:
                        result = nested_cv_test(features, y_player, model_type)

                        if not np.isnan(result['r2']) and result['r2'] > best_r2:
                            best_r2 = result['r2']
                            best_frame = frame
                            best_model = model_type

                indicator = ""
                if best_r2 > 0.3:
                    indicator = " ***STRONG***"
                elif best_r2 > 0.1:
                    indicator = " ***"
                elif best_r2 > 0:
                    indicator = " +"

                print(f"    Player {pid}: R²={best_r2:+.3f} (frame {best_frame}, {best_model}){indicator}")

                results.append({
                    'target': target_name,
                    'group': group_name,
                    'player': pid,
                    'best_r2': best_r2,
                    'best_frame': best_frame,
                    'best_model': best_model,
                    'description': description
                })

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY: BEST PHYSICS GROUP PER PLAYER PER TARGET")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    for target_name in ['angle', 'depth', 'left_right']:
        print(f"\n{target_name.upper()}:")
        target_results = results_df[results_df['target'] == target_name]

        for pid in sorted(np.unique(participant_ids)):
            player_results = target_results[target_results['player'] == pid]
            if len(player_results) > 0:
                best = player_results.loc[player_results['best_r2'].idxmax()]
                print(f"  Player {pid}: R²={best['best_r2']:.3f} | {best['group']} @ frame {best['best_frame']} ({best['best_model']})")

        avg_r2 = target_results.groupby('player')['best_r2'].max().mean()
        print(f"  AVERAGE MAX R²: {avg_r2:.3f}")

    # Save
    results_df.to_csv(OUTPUT_DIR / 'comprehensive_physics_groups.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'comprehensive_physics_groups.csv'}")


if __name__ == "__main__":
    main()
