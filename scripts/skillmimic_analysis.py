"""
SkillMimic Basketball Shooting Data Analysis

This script:
1. Loads and analyzes the SkillMimic shot style data
2. Extracts shooting form patterns from professional motion capture
3. Maps the 337 features to body parts and derives useful patterns
4. Creates features that can be applied to our SPL prediction task

Data format (from research):
- Features 0-2: root_pos (pelvis position in heading-relative frame)
- Features 3-5: root_rot (pelvis rotation, exponential map)
- Features 6-161: dof_pos (52 joints x 3 DOF = 156 joint positions)
- Features 162-317: dof_vel (52 joints x 3 DOF = 156 joint velocities)
- Features 318-320: ball position (X, Y, Z)
- Features 321-324: ball rotation (quaternion)
- Features 325-327: ball velocity (X, Y, Z)
- Features 328-335: key body positions
- Feature 336: contact flag (1=in hand, 0=released)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SKILLMIMIC_DIR = PROJECT_ROOT / "external_data" / "skillmimic"
OUTPUT_DIR = PROJECT_ROOT / "output"

# SkillMimic feature indices (from research docs)
SKILLMIMIC_FEATURES = {
    'root_pos': (0, 3),      # 3 features: pelvis position
    'root_rot': (3, 6),      # 3 features: pelvis rotation
    'dof_pos': (6, 162),     # 156 features: joint positions (52 joints x 3)
    'dof_vel': (162, 318),   # 156 features: joint velocities
    'ball_pos': (318, 321),  # 3 features: ball position
    'ball_rot': (321, 325),  # 4 features: ball rotation quaternion
    'ball_vel': (325, 328),  # 3 features: ball velocity
    'key_body': (328, 336),  # 8 features: key body positions
    'contact': 336           # 1 feature: ball contact flag
}

# Body part indices within dof_pos (52 joints)
# Each joint has 3 DOF (x, y, z)
JOINT_INDICES = {
    # Core
    'pelvis': 0, 'l_hip': 1, 'l_knee': 2, 'l_ankle': 3, 'l_toe': 4,
    'r_hip': 5, 'r_knee': 6, 'r_ankle': 7, 'r_toe': 8,
    'torso': 9, 'spine': 10, 'spine2': 11, 'chest': 12, 'neck': 13, 'head': 14,
    # Left arm (15-18)
    'l_thorax': 15, 'l_shoulder': 16, 'l_elbow': 17, 'l_wrist': 18,
    # Left fingers (19-33)
    'l_index1': 19, 'l_index2': 20, 'l_index3': 21,
    'l_middle1': 22, 'l_middle2': 23, 'l_middle3': 24,
    'l_pinky1': 25, 'l_pinky2': 26, 'l_pinky3': 27,
    'l_ring1': 28, 'l_ring2': 29, 'l_ring3': 30,
    'l_thumb1': 31, 'l_thumb2': 32, 'l_thumb3': 33,
    # Right arm (34-37)
    'r_thorax': 34, 'r_shoulder': 35, 'r_elbow': 36, 'r_wrist': 37,
    # Right fingers (38-52)
    'r_index1': 38, 'r_index2': 39, 'r_index3': 40,
    'r_middle1': 41, 'r_middle2': 42, 'r_middle3': 43,
    'r_pinky1': 44, 'r_pinky2': 45, 'r_pinky3': 46,
    'r_ring1': 47, 'r_ring2': 48, 'r_ring3': 49,
    'r_thumb1': 50, 'r_thumb2': 51, 'r_thumb3': 52,
}


def load_skillmimic_data():
    """Load all SkillMimic shot style files."""
    files = sorted(SKILLMIMIC_DIR.glob("shot_style*.pt"))

    print(f"Found {len(files)} SkillMimic files")

    data = {}
    for f in files:
        name = f.stem
        tensor = torch.load(f, map_location='cpu', weights_only=False)
        data[name] = tensor.numpy()
        print(f"  {name}: shape {tensor.shape}")

    return data


def analyze_data_structure(data):
    """Analyze the structure and statistics of SkillMimic data."""
    print("\n" + "="*80)
    print("DATA STRUCTURE ANALYSIS")
    print("="*80)

    for name, arr in data.items():
        print(f"\n{name}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Min: {arr.min():.4f}")
        print(f"  Max: {arr.max():.4f}")
        print(f"  Mean: {arr.mean():.4f}")
        print(f"  Std: {arr.std():.4f}")

        # Analyze specific feature groups
        for feature_name, indices in SKILLMIMIC_FEATURES.items():
            if isinstance(indices, tuple):
                start, end = indices
                subset = arr[:, start:end]
            else:
                subset = arr[:, indices:indices+1]

            print(f"  {feature_name}: mean={subset.mean():.4f}, std={subset.std():.4f}, range=[{subset.min():.4f}, {subset.max():.4f}]")

    return data


def find_release_frame(arr):
    """Find the ball release frame using contact flag."""
    contact = arr[:, SKILLMIMIC_FEATURES['contact']]

    # Find transition from 1 to 0 (or use threshold)
    # Contact flag might not be binary, so use threshold
    is_contact = contact > 0.5

    # Find last contact frame
    contact_frames = np.where(is_contact)[0]
    if len(contact_frames) > 0:
        release_frame = contact_frames[-1] + 1
        return min(release_frame, len(arr) - 1)

    return len(arr) // 2  # Default to middle if no clear release


def extract_shooting_patterns(data):
    """Extract key shooting patterns from the data."""
    print("\n" + "="*80)
    print("SHOOTING PATTERN EXTRACTION")
    print("="*80)

    patterns = {}

    for name, arr in data.items():
        print(f"\n{name}:")

        # Find release frame
        release_frame = find_release_frame(arr)
        print(f"  Release frame: {release_frame} / {len(arr)}")

        # Extract joint positions
        dof_pos_start, dof_pos_end = SKILLMIMIC_FEATURES['dof_pos']
        joint_pos = arr[:, dof_pos_start:dof_pos_end].reshape(len(arr), 52, 3)

        # Extract velocities
        dof_vel_start, dof_vel_end = SKILLMIMIC_FEATURES['dof_vel']
        joint_vel = arr[:, dof_vel_start:dof_vel_end].reshape(len(arr), 52, 3)

        # Extract ball data
        ball_pos = arr[:, SKILLMIMIC_FEATURES['ball_pos'][0]:SKILLMIMIC_FEATURES['ball_pos'][1]]
        ball_vel = arr[:, SKILLMIMIC_FEATURES['ball_vel'][0]:SKILLMIMIC_FEATURES['ball_vel'][1]]

        # Key measurements at release
        r_wrist_idx = JOINT_INDICES['r_wrist']
        r_elbow_idx = JOINT_INDICES['r_elbow']
        r_shoulder_idx = JOINT_INDICES['r_shoulder']

        # Wrist position and velocity at release
        wrist_pos_release = joint_pos[release_frame, r_wrist_idx]
        wrist_vel_release = joint_vel[release_frame, r_wrist_idx]

        # Elbow angle at release (angle between shoulder-elbow and elbow-wrist vectors)
        shoulder_pos = joint_pos[release_frame, r_shoulder_idx]
        elbow_pos = joint_pos[release_frame, r_elbow_idx]
        wrist_pos = joint_pos[release_frame, r_wrist_idx]

        upper_arm = elbow_pos - shoulder_pos
        forearm = wrist_pos - elbow_pos

        # Calculate angle
        dot = np.dot(upper_arm, forearm)
        mag = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
        if mag > 1e-8:
            elbow_angle = np.arccos(np.clip(dot / mag, -1, 1)) * 180 / np.pi
        else:
            elbow_angle = 0

        # Ball velocity at release
        ball_vel_release = ball_vel[release_frame]
        ball_speed = np.linalg.norm(ball_vel_release)

        # Calculate release angle (vertical component)
        if ball_speed > 1e-8:
            # Assume Y is vertical
            release_angle = np.arcsin(np.clip(ball_vel_release[1] / ball_speed, -1, 1)) * 180 / np.pi
        else:
            release_angle = 0

        # Motion dynamics: acceleration pattern
        wrist_vel_magnitude = np.linalg.norm(joint_vel[:, r_wrist_idx], axis=1)
        peak_velocity_frame = np.argmax(wrist_vel_magnitude)
        peak_velocity = wrist_vel_magnitude[peak_velocity_frame]

        # Store patterns
        patterns[name] = {
            'num_frames': len(arr),
            'release_frame': release_frame,
            'release_pct': release_frame / len(arr) * 100,
            'wrist_pos_release': wrist_pos_release.tolist(),
            'wrist_vel_release': wrist_vel_release.tolist(),
            'elbow_angle_release': elbow_angle,
            'ball_vel_release': ball_vel_release.tolist(),
            'ball_speed_release': ball_speed,
            'release_angle_vertical': release_angle,
            'peak_wrist_velocity': peak_velocity,
            'peak_velocity_frame': peak_velocity_frame,
            'peak_velocity_pct': peak_velocity_frame / len(arr) * 100,
        }

        print(f"  Wrist position at release: {wrist_pos_release}")
        print(f"  Elbow angle at release: {elbow_angle:.1f} degrees")
        print(f"  Ball speed at release: {ball_speed:.3f}")
        print(f"  Release angle (vertical): {release_angle:.1f} degrees")
        print(f"  Peak wrist velocity: {peak_velocity:.3f} at frame {peak_velocity_frame}")

    return patterns


def extract_temporal_features(data):
    """Extract temporal patterns that can inform feature engineering."""
    print("\n" + "="*80)
    print("TEMPORAL FEATURE ANALYSIS")
    print("="*80)

    temporal_features = {}

    for name, arr in data.items():
        print(f"\n{name}:")

        release_frame = find_release_frame(arr)

        # Joint positions
        dof_pos_start, dof_pos_end = SKILLMIMIC_FEATURES['dof_pos']
        joint_pos = arr[:, dof_pos_start:dof_pos_end].reshape(len(arr), 52, 3)

        # Joint velocities
        dof_vel_start, dof_vel_end = SKILLMIMIC_FEATURES['dof_vel']
        joint_vel = arr[:, dof_vel_start:dof_vel_end].reshape(len(arr), 52, 3)

        # Compute key temporal patterns
        r_wrist = JOINT_INDICES['r_wrist']
        r_elbow = JOINT_INDICES['r_elbow']

        # 1. Wrist height trajectory
        wrist_height = joint_pos[:, r_wrist, 1]  # Y coordinate

        # 2. Wrist velocity profile
        wrist_speed = np.linalg.norm(joint_vel[:, r_wrist], axis=1)

        # 3. Elbow extension rate
        elbow_speed = np.linalg.norm(joint_vel[:, r_elbow], axis=1)

        # Key metrics
        metrics = {
            'wrist_height_start': wrist_height[0],
            'wrist_height_release': wrist_height[release_frame],
            'wrist_height_change': wrist_height[release_frame] - wrist_height[0],
            'wrist_max_height': wrist_height.max(),
            'wrist_max_height_frame': wrist_height.argmax(),
            'wrist_speed_at_release': wrist_speed[release_frame],
            'wrist_peak_speed': wrist_speed.max(),
            'wrist_peak_speed_frame': wrist_speed.argmax(),
            'elbow_speed_at_release': elbow_speed[release_frame],
            'elbow_peak_speed': elbow_speed.max(),
            # Timing ratios
            'release_timing': release_frame / len(arr),
            'peak_speed_timing': wrist_speed.argmax() / len(arr),
            'max_height_timing': wrist_height.argmax() / len(arr),
        }

        temporal_features[name] = metrics

        print(f"  Wrist height change: {metrics['wrist_height_change']:.3f}")
        print(f"  Release timing: {metrics['release_timing']*100:.1f}%")
        print(f"  Peak speed timing: {metrics['peak_speed_timing']*100:.1f}%")

    return temporal_features


def compare_with_spl_format():
    """Compare SkillMimic format with SPL dataset format."""
    print("\n" + "="*80)
    print("SPL DATA FORMAT COMPARISON")
    print("="*80)

    # SPL format from data_loader.py
    spl_format = {
        'frames': 240,
        'keypoints': 69,  # 23 body + 21 left hand + 21 right hand + 4 face
        'coords': 3,  # x, y, z
        'features': 69 * 3,  # 207
        'frame_rate': 60,
    }

    # SkillMimic format
    skillmimic_format = {
        'frames': '~100-150 (variable)',
        'joints': 52,
        'features': 337,
        'includes_velocities': True,
        'includes_ball_state': True,
        'includes_contact': True,
        'coordinate_system': 'heading-relative',
    }

    print("\nSPL Dataset:")
    for k, v in spl_format.items():
        print(f"  {k}: {v}")

    print("\nSkillMimic Dataset:")
    for k, v in skillmimic_format.items():
        print(f"  {k}: {v}")

    print("\nKey Differences:")
    print("  1. SPL has more keypoints (69 vs 52), including detailed hand/face")
    print("  2. SkillMimic includes velocities directly; SPL needs them computed")
    print("  3. SkillMimic has ball state; SPL needs ball tracking")
    print("  4. SkillMimic uses heading-relative coords; SPL uses world coords")
    print("  5. SPL has fixed 240 frames; SkillMimic has variable length")

    return spl_format, skillmimic_format


def derive_transferable_features():
    """Derive features from SkillMimic that can be applied to SPL data."""
    print("\n" + "="*80)
    print("TRANSFERABLE FEATURE DERIVATION")
    print("="*80)

    features = {
        'position_based': [
            'wrist_height_at_frame_X',
            'elbow_extension_ratio',  # forearm / upper arm distance ratio
            'shoulder_elevation_angle',
            'trunk_lean_angle',
            'knee_bend_angle',
            'wrist_to_shoulder_vector',
            'hand_above_head_flag',
        ],
        'velocity_based': [
            'wrist_velocity_magnitude',
            'wrist_velocity_direction',
            'elbow_angular_velocity',
            'trunk_rotation_rate',
            'knee_extension_velocity',
        ],
        'timing_based': [
            'peak_velocity_frame_ratio',
            'release_frame_ratio',
            'preparation_phase_length',
            'execution_phase_length',
            'follow_through_length',
        ],
        'coordination_based': [
            'sequential_kinetic_chain_score',  # knee -> hip -> trunk -> shoulder -> elbow -> wrist
            'arm_leg_synchronization',
            'bilateral_symmetry_score',
        ],
        'physics_based': [
            'estimated_release_velocity',
            'estimated_release_angle',
            'kinetic_energy_at_release',
            'angular_momentum_transfer',
        ]
    }

    print("\nTransferable features by category:")
    for category, feats in features.items():
        print(f"\n  {category}:")
        for f in feats:
            print(f"    - {f}")

    return features


def create_skillmimic_feature_extractor():
    """Create a feature extraction function based on SkillMimic patterns."""

    print("\n" + "="*80)
    print("FEATURE EXTRACTOR TEMPLATE")
    print("="*80)

    # This is a template for SPL feature extraction based on SkillMimic insights
    code = '''
def extract_skillmimic_inspired_features(keypoints, frame_idx):
    """
    Extract features inspired by SkillMimic shooting patterns.

    Args:
        keypoints: np.ndarray of shape (240, 69, 3) - frames x keypoints x xyz
        frame_idx: int - frame to extract features from (e.g., release frame)

    Returns:
        dict: Feature name -> value
    """
    features = {}

    # Key keypoint indices for SPL (adjust based on actual mapping)
    # Approximate mapping:
    # r_wrist ~ index 10, l_wrist ~ index 9
    # r_elbow ~ index 8, l_elbow ~ index 7
    # r_shoulder ~ index 12, l_shoulder ~ index 11

    r_wrist = keypoints[frame_idx, 10]  # right wrist position
    r_elbow = keypoints[frame_idx, 8]
    r_shoulder = keypoints[frame_idx, 12]

    # 1. Wrist height (normalized by shoulder height)
    features['wrist_height_norm'] = r_wrist[1] / (r_shoulder[1] + 1e-8)

    # 2. Elbow angle
    upper_arm = r_elbow - r_shoulder
    forearm = r_wrist - r_elbow
    dot = np.dot(upper_arm, forearm)
    mag = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
    if mag > 1e-8:
        features['elbow_angle'] = np.arccos(np.clip(dot / mag, -1, 1))
    else:
        features['elbow_angle'] = 0

    # 3. Arm extension ratio
    arm_length = np.linalg.norm(upper_arm) + np.linalg.norm(forearm)
    shoulder_to_wrist = np.linalg.norm(r_wrist - r_shoulder)
    features['arm_extension_ratio'] = shoulder_to_wrist / (arm_length + 1e-8)

    # 4. Wrist velocity (if we have frame_idx > 0)
    if frame_idx > 0:
        r_wrist_prev = keypoints[frame_idx - 1, 10]
        wrist_vel = r_wrist - r_wrist_prev
        features['wrist_velocity_mag'] = np.linalg.norm(wrist_vel) * 60  # 60 FPS
        features['wrist_velocity_y'] = wrist_vel[1] * 60  # vertical component
    else:
        features['wrist_velocity_mag'] = 0
        features['wrist_velocity_y'] = 0

    # 5. Shoulder elevation angle
    # Angle of shoulder-wrist vector from horizontal
    shoulder_to_wrist_vec = r_wrist - r_shoulder
    horizontal = np.array([1, 0, 0])  # X direction
    if np.linalg.norm(shoulder_to_wrist_vec) > 1e-8:
        cos_angle = np.dot(shoulder_to_wrist_vec, horizontal) / np.linalg.norm(shoulder_to_wrist_vec)
        features['arm_elevation_angle'] = np.arccos(np.clip(cos_angle, -1, 1))
    else:
        features['arm_elevation_angle'] = 0

    return features
'''

    print(code)
    return code


def main():
    print("="*80)
    print("SKILLMIMIC BASKETBALL SHOOTING DATA ANALYSIS")
    print("="*80)

    # 1. Load data
    print("\n[Step 1] Loading SkillMimic data...")
    data = load_skillmimic_data()

    if not data:
        print("ERROR: No SkillMimic data found!")
        print(f"Expected files in: {SKILLMIMIC_DIR}")
        return

    # 2. Analyze data structure
    print("\n[Step 2] Analyzing data structure...")
    analyze_data_structure(data)

    # 3. Extract shooting patterns
    print("\n[Step 3] Extracting shooting patterns...")
    patterns = extract_shooting_patterns(data)

    # 4. Extract temporal features
    print("\n[Step 4] Extracting temporal features...")
    temporal_features = extract_temporal_features(data)

    # 5. Compare with SPL format
    print("\n[Step 5] Comparing with SPL format...")
    spl_format, skillmimic_format = compare_with_spl_format()

    # 6. Derive transferable features
    print("\n[Step 6] Deriving transferable features...")
    transferable = derive_transferable_features()

    # 7. Create feature extractor template
    print("\n[Step 7] Creating feature extractor template...")
    extractor_code = create_skillmimic_feature_extractor()

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save patterns as JSON-compatible format
    import json

    with open(OUTPUT_DIR / "skillmimic_patterns.json", 'w') as f:
        json.dump(convert_to_native(patterns), f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'skillmimic_patterns.json'}")

    with open(OUTPUT_DIR / "skillmimic_temporal.json", 'w') as f:
        json.dump(convert_to_native(temporal_features), f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'skillmimic_temporal.json'}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nAcross all shot styles:")
    all_release_angles = [p['release_angle_vertical'] for p in patterns.values()]
    all_elbow_angles = [p['elbow_angle_release'] for p in patterns.values()]
    all_ball_speeds = [p['ball_speed_release'] for p in patterns.values()]

    print(f"  Release angle (vertical): {np.mean(all_release_angles):.1f} +/- {np.std(all_release_angles):.1f} degrees")
    print(f"  Elbow angle at release: {np.mean(all_elbow_angles):.1f} +/- {np.std(all_elbow_angles):.1f} degrees")
    print(f"  Ball speed at release: {np.mean(all_ball_speeds):.3f} +/- {np.std(all_ball_speeds):.3f} units")

    print("\nRelease timing (% through motion):")
    for name, metrics in temporal_features.items():
        print(f"  {name}: {metrics['release_timing']*100:.1f}%")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
