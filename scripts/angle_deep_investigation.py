"""
ANGLE DEEP INVESTIGATION

The angle MUST be explained by physics. If we can't find it, we're looking wrong.

Possible issues with previous tests:
1. Only looked at Z coordinate - maybe X, Y matter for angle
2. Only looked at positions/velocities - maybe need joint ANGLES
3. Only looked at single features - maybe need combinations
4. Wrong frame range - maybe angle is determined earlier/later
5. Need acceleration, not just velocity

This test will exhaustively search for what explains angle.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_all_as_arrays, get_keypoint_columns

OUTPUT_DIR = Path(__file__).parent.parent / 'output'


class FullKeypointExtractor:
    """Extract X, Y, Z coordinates for any keypoint."""

    def __init__(self, keypoint_cols: List[str]):
        self.keypoint_cols = keypoint_cols
        self.col_to_idx = {col: i for i, col in enumerate(keypoint_cols)}

        # Build mapping for all coordinates
        self.keypoints = set()
        for col in keypoint_cols:
            # Column format: "joint_name_coord" e.g., "right_wrist_x"
            parts = col.rsplit('_', 1)
            if len(parts) == 2:
                self.keypoints.add(parts[0])

    def get(self, ts: np.ndarray, keypoint: str, coord: str, frame: int = None):
        """Get keypoint coordinate. coord is 'x', 'y', or 'z'."""
        col_name = f"{keypoint}_{coord}"
        idx = self.col_to_idx.get(col_name)
        if idx is None:
            return np.nan if frame is not None else np.full(ts.shape[0], np.nan)
        if frame is not None:
            if frame < 0 or frame >= ts.shape[0]:
                return np.nan
            return ts[frame, idx]
        return ts[:, idx]

    def get_3d(self, ts: np.ndarray, keypoint: str, frame: int) -> np.ndarray:
        """Get all 3 coordinates as array."""
        return np.array([
            self.get(ts, keypoint, 'x', frame),
            self.get(ts, keypoint, 'y', frame),
            self.get(ts, keypoint, 'z', frame),
        ])


def compute_joint_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle at p2 formed by p1-p2-p3 in 3D space."""
    v1 = p1 - p2
    v2 = p3 - p2

    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return np.nan

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.degrees(np.arccos(cos_angle))


def nested_cv_single_feature(X_feat: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    """Simple nested CV for a single feature array."""
    valid = ~(np.isnan(X_feat).any(axis=1) | np.isnan(y))
    X_clean = X_feat[valid]
    y_clean = y[valid]

    if len(y_clean) < 15:
        return np.nan

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X_clean):
        X_tr, X_te = X_clean[train_idx], X_clean[test_idx]
        y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = Ridge(alpha=alpha)
        model.fit(X_tr_s, y_tr)
        scores.append(model.score(X_te_s, y_te))

    return np.mean(scores)


def main():
    print("=" * 80)
    print("ANGLE DEEP INVESTIGATION")
    print("=" * 80)
    print()
    print("The angle MUST be explained by physics. Let's find what we're missing.")
    print()

    # Load data
    X, y, meta = load_all_as_arrays(train=True)
    keypoint_cols = get_keypoint_columns()
    kp = FullKeypointExtractor(keypoint_cols)

    participant_ids = meta['participant_id'].values
    angle_target = y[:, 0]

    print(f"Data: {len(y)} samples, {len(np.unique(participant_ids))} players")
    print(f"Available keypoints: {len(kp.keypoints)}")
    print()

    results = []

    # ==========================================================================
    # TEST 1: All coordinates (X, Y, Z) not just Z
    # ==========================================================================
    print("=" * 80)
    print("TEST 1: All Coordinates (X, Y, Z) at Release Frames")
    print("=" * 80)
    print()

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip']

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        print(f"Player {pid}:")

        best_r2 = -999
        best_config = None

        for joint in key_joints:
            for coord in ['x', 'y', 'z']:
                for frame in range(140, 180, 5):
                    features = np.array([kp.get(X_player[i], joint, coord, frame)
                                        for i in range(len(X_player))]).reshape(-1, 1)

                    r2 = nested_cv_single_feature(features, y_player)
                    if not np.isnan(r2) and r2 > best_r2:
                        best_r2 = r2
                        best_config = (joint, coord, frame)

                    if r2 > 0:
                        results.append({
                            'test': 'single_coord',
                            'player': pid,
                            'feature': f'{joint}_{coord}_f{frame}',
                            'r2': r2
                        })

        if best_config:
            indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
            print(f"  Best: {best_config[0]}_{best_config[1]} at frame {best_config[2]}: R²={best_r2:.4f} {indicator}")
        print()

    # ==========================================================================
    # TEST 2: True 3D Joint Angles (elbow angle, wrist angle)
    # ==========================================================================
    print("=" * 80)
    print("TEST 2: True 3D Joint Angles")
    print("=" * 80)
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        print(f"Player {pid}:")

        best_r2 = -999
        best_config = None

        for frame in range(130, 180, 5):
            # Elbow angle: shoulder-elbow-wrist
            elbow_angles = []
            for i in range(len(X_player)):
                shoulder = kp.get_3d(X_player[i], 'right_shoulder', frame)
                elbow = kp.get_3d(X_player[i], 'right_elbow', frame)
                wrist = kp.get_3d(X_player[i], 'right_wrist', frame)
                elbow_angles.append(compute_joint_angle_3d(shoulder, elbow, wrist))

            features = np.array(elbow_angles).reshape(-1, 1)
            r2 = nested_cv_single_feature(features, y_player)

            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_config = ('elbow_angle_3d', frame)

            if r2 > 0:
                results.append({
                    'test': '3d_joint_angle',
                    'player': pid,
                    'feature': f'elbow_angle_3d_f{frame}',
                    'r2': r2
                })

            # Shoulder angle: hip-shoulder-elbow
            shoulder_angles = []
            for i in range(len(X_player)):
                hip = kp.get_3d(X_player[i], 'right_hip', frame)
                shoulder = kp.get_3d(X_player[i], 'right_shoulder', frame)
                elbow = kp.get_3d(X_player[i], 'right_elbow', frame)
                shoulder_angles.append(compute_joint_angle_3d(hip, shoulder, elbow))

            features = np.array(shoulder_angles).reshape(-1, 1)
            r2 = nested_cv_single_feature(features, y_player)

            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_config = ('shoulder_angle_3d', frame)

            if r2 > 0:
                results.append({
                    'test': '3d_joint_angle',
                    'player': pid,
                    'feature': f'shoulder_angle_3d_f{frame}',
                    'r2': r2
                })

        if best_config:
            indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
            print(f"  Best: {best_config[0]} at frame {best_config[1]}: R²={best_r2:.4f} {indicator}")
        print()

    # ==========================================================================
    # TEST 3: Velocity and Acceleration
    # ==========================================================================
    print("=" * 80)
    print("TEST 3: Velocity and Acceleration of Wrist")
    print("=" * 80)
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        print(f"Player {pid}:")

        best_r2 = -999
        best_config = None

        for coord in ['x', 'y', 'z']:
            for frame in range(130, 175, 5):
                # Velocity (first derivative)
                velocities = []
                for i in range(len(X_player)):
                    v1 = kp.get(X_player[i], 'right_wrist', coord, frame)
                    v2 = kp.get(X_player[i], 'right_wrist', coord, frame + 5)
                    velocities.append(v2 - v1 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)

                features = np.array(velocities).reshape(-1, 1)
                r2 = nested_cv_single_feature(features, y_player)

                if not np.isnan(r2) and r2 > best_r2:
                    best_r2 = r2
                    best_config = (f'wrist_{coord}_velocity', frame)

                if r2 > 0:
                    results.append({
                        'test': 'velocity',
                        'player': pid,
                        'feature': f'wrist_{coord}_vel_f{frame}',
                        'r2': r2
                    })

                # Acceleration (second derivative)
                accelerations = []
                for i in range(len(X_player)):
                    v0 = kp.get(X_player[i], 'right_wrist', coord, frame - 5)
                    v1 = kp.get(X_player[i], 'right_wrist', coord, frame)
                    v2 = kp.get(X_player[i], 'right_wrist', coord, frame + 5)
                    if np.isnan(v0) or np.isnan(v1) or np.isnan(v2):
                        accelerations.append(np.nan)
                    else:
                        acc = (v2 - 2*v1 + v0)  # Second derivative
                        accelerations.append(acc)

                features = np.array(accelerations).reshape(-1, 1)
                r2 = nested_cv_single_feature(features, y_player)

                if not np.isnan(r2) and r2 > best_r2:
                    best_r2 = r2
                    best_config = (f'wrist_{coord}_acceleration', frame)

                if r2 > 0:
                    results.append({
                        'test': 'acceleration',
                        'player': pid,
                        'feature': f'wrist_{coord}_acc_f{frame}',
                        'r2': r2
                    })

        if best_config:
            indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
            print(f"  Best: {best_config[0]} at frame {best_config[1]}: R²={best_r2:.4f} {indicator}")
        print()

    # ==========================================================================
    # TEST 4: Release trajectory (multiple frames combined)
    # ==========================================================================
    print("=" * 80)
    print("TEST 4: Release Trajectory (Wrist path over multiple frames)")
    print("=" * 80)
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        print(f"Player {pid}:")

        best_r2 = -999
        best_config = None

        # Try different trajectory windows
        for start_frame in range(140, 165, 5):
            for window in [10, 15, 20]:
                end_frame = start_frame + window

                # Extract trajectory features: start pos, end pos, total displacement
                trajectory_features = []
                for i in range(len(X_player)):
                    feats = []
                    for coord in ['x', 'y', 'z']:
                        start_val = kp.get(X_player[i], 'right_wrist', coord, start_frame)
                        end_val = kp.get(X_player[i], 'right_wrist', coord, end_frame)
                        if np.isnan(start_val) or np.isnan(end_val):
                            feats.extend([np.nan, np.nan, np.nan])
                        else:
                            feats.extend([start_val, end_val, end_val - start_val])
                    trajectory_features.append(feats)

                features = np.array(trajectory_features)
                r2 = nested_cv_single_feature(features, y_player)

                if not np.isnan(r2) and r2 > best_r2:
                    best_r2 = r2
                    best_config = (f'trajectory_{start_frame}_{end_frame}', window)

                if r2 > 0:
                    results.append({
                        'test': 'trajectory',
                        'player': pid,
                        'feature': f'wrist_trajectory_{start_frame}_{end_frame}',
                        'r2': r2
                    })

        if best_config:
            indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
            print(f"  Best: {best_config[0]}: R²={best_r2:.4f} {indicator}")
        print()

    # ==========================================================================
    # TEST 5: Arm angle relative to vertical (release angle proxy)
    # ==========================================================================
    print("=" * 80)
    print("TEST 5: Arm Angle Relative to Vertical")
    print("=" * 80)
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        print(f"Player {pid}:")

        best_r2 = -999
        best_config = None

        for frame in range(145, 175, 5):
            # Angle of arm (elbow to wrist vector) relative to vertical
            arm_angles = []
            for i in range(len(X_player)):
                elbow = kp.get_3d(X_player[i], 'right_elbow', frame)
                wrist = kp.get_3d(X_player[i], 'right_wrist', frame)

                if np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)):
                    arm_angles.append(np.nan)
                    continue

                arm_vec = wrist - elbow
                vertical = np.array([0, 0, 1])  # Assuming Z is up

                cos_angle = np.dot(arm_vec, vertical) / (np.linalg.norm(arm_vec) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                arm_angles.append(angle)

            features = np.array(arm_angles).reshape(-1, 1)
            r2 = nested_cv_single_feature(features, y_player)

            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_config = ('arm_vertical_angle', frame)

            if r2 > 0:
                results.append({
                    'test': 'arm_vertical_angle',
                    'player': pid,
                    'feature': f'arm_vertical_angle_f{frame}',
                    'r2': r2
                })

        if best_config:
            indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
            print(f"  Best: {best_config[0]} at frame {best_config[1]}: R²={best_r2:.4f} {indicator}")
        print()

    # ==========================================================================
    # TEST 6: Full kinetic chain at release
    # ==========================================================================
    print("=" * 80)
    print("TEST 6: Full Kinetic Chain at Release")
    print("=" * 80)
    print()

    for pid in sorted(np.unique(participant_ids)):
        mask = participant_ids == pid
        y_player = angle_target[mask]
        X_player = X[mask]

        print(f"Player {pid}:")

        best_r2 = -999
        best_config = None

        for frame in range(145, 170, 5):
            # Full kinetic chain: ankle, knee, hip, shoulder, elbow, wrist (all Z coords)
            chain_features = []
            joints = ['right_ankle', 'right_knee', 'right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']

            for i in range(len(X_player)):
                feats = []
                for joint in joints:
                    feats.append(kp.get(X_player[i], joint, 'z', frame))
                chain_features.append(feats)

            features = np.array(chain_features)
            r2 = nested_cv_single_feature(features, y_player)

            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_config = ('kinetic_chain_z', frame)

            if r2 > 0:
                results.append({
                    'test': 'kinetic_chain',
                    'player': pid,
                    'feature': f'kinetic_chain_z_f{frame}',
                    'r2': r2
                })

        if best_config:
            indicator = "***" if best_r2 > 0.05 else "+" if best_r2 > 0 else ""
            print(f"  Best: {best_config[0]} at frame {best_config[1]}: R²={best_r2:.4f} {indicator}")
        print()

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("=" * 80)
    print("SUMMARY: ALL POSITIVE R² RESULTS FOR ANGLE")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Group by player and show best
        print("\nBest feature per player:")
        for pid in sorted(np.unique(participant_ids)):
            player_results = results_df[results_df['player'] == pid]
            if len(player_results) > 0:
                best = player_results.loc[player_results['r2'].idxmax()]
                indicator = "***" if best['r2'] > 0.05 else "+"
                print(f"  Player {pid}: {best['feature']} R²={best['r2']:.4f} {indicator}")
            else:
                print(f"  Player {pid}: No positive R² found")

        # Show all results > 0.05
        strong = results_df[results_df['r2'] > 0.05]
        print(f"\nStrong results (R² > 0.05): {len(strong)}")
        if len(strong) > 0:
            for _, row in strong.iterrows():
                print(f"  Player {row['player']}: {row['feature']} R²={row['r2']:.4f}")

        # Save
        results_df.to_csv(OUTPUT_DIR / 'angle_deep_investigation.csv', index=False)
        print(f"\nResults saved to {OUTPUT_DIR / 'angle_deep_investigation.csv'}")
    else:
        print("\nNo positive R² results found for any configuration.")


if __name__ == "__main__":
    main()
