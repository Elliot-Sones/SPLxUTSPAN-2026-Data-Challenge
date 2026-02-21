"""
Kinetic Chain Feature Engineering.

Extracts features related to the sequential transfer of energy through the body segments:
Legs -> Hips -> Torso -> Shoulder -> Elbow -> Wrist.

Key concepts:
- Sequencing: Time delays between peak velocities of adjacent segments.
- Amplification: Ratio of peak velocities between adjacent segments.
- Energy: Integrated power (velocity squared) during the propulsion phase.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Optional

# Constants
FRAME_RATE = 60
DT = 1.0 / FRAME_RATE
NUM_FRAMES = 240

def smooth_signal(data: np.ndarray, window: int = 7) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to reducing noise in derivatives."""
    if len(data) < window:
        return data
    try:
        if data.ndim == 1:
            return savgol_filter(data, window, 2)
        else:
            return savgol_filter(data, window, 2, axis=0)
    except Exception:
        return data

def compute_velocity(pos: np.ndarray) -> np.ndarray:
    """Compute first derivative (velocity)."""
    return np.gradient(pos, DT, axis=0)

def compute_acceleration(vel: np.ndarray) -> np.ndarray:
    """Compute second derivative (acceleration)."""
    return np.gradient(vel, DT, axis=0)

def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Compute angle at joint p2 formed by p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    denom = norm1 * norm2
    denom[denom == 0] = 1e-10
    return np.arccos(np.clip(dot / denom, -1, 1)) * 180 / np.pi

def get_keypoint_data_safe(timeseries: np.ndarray, mapping: Dict[str, int], keypoint: str) -> Optional[np.ndarray]:
    """Safely extract keypoint data using the mapping."""
    if keypoint not in mapping:
        return None
    idx = mapping[keypoint]
    return timeseries[:, idx*3:(idx+1)*3]

def extract_kinetic_chain_features(
    timeseries: np.ndarray, 
    mapping: Dict[str, int], 
    participant_id: Optional[int] = None
) -> Dict[str, float]:
    """
    Extract kinetic chain features from a single shot.
    
    Args:
        timeseries: (240, 207) array of keypoint data
        mapping: Dict mapping keypoint names to column indices
        participant_id: Optional ID for player-specific logic
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    # 1. Get relevant body parts
    # We focus on the right side (shooting side) for consistency
    joints_map = {
        "ankle": "right_ankle",
        "knee": "right_knee",
        "hip": "right_hip",
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist"
    }
    
    joints = {}
    for name, kp_name in joints_map.items():
        data = get_keypoint_data_safe(timeseries, mapping, kp_name)
        if data is None:
            return features # Cannot compute chain without all joints
        joints[name] = smooth_signal(data)

    # 2. Calculate Velocities
    velocities = {}
    vert_velocities = {}
    for k, v in joints.items():
        vel = compute_velocity(v)
        velocities[k] = np.linalg.norm(vel, axis=1)
        vert_velocities[k] = vel[:, 2] # Z component

    # 3. Find Release Frame (Peak Wrist Velocity in 2nd half)
    # This is our anchor point for the "Propulsion Phase"
    search_start = NUM_FRAMES // 2
    if search_start < NUM_FRAMES:
        release_frame = search_start + np.argmax(velocities["wrist"][search_start:])
    else:
        release_frame = NUM_FRAMES - 10
    
    features["kc_release_frame"] = release_frame

    # 4. Define Propulsion Phase
    # Energy generation happens before release. 
    # We look back 60 frames (1 second) or until start.
    prop_start = max(0, release_frame - 60)
    prop_end = min(NUM_FRAMES, release_frame + 5)

    # 5. Peak Analysis
    chain_order = ["knee", "hip", "shoulder", "elbow", "wrist"]
    peak_times = {}
    peak_mags = {}

    for joint in chain_order:
        # Define the window to search for peaks
        window_vel = velocities[joint][prop_start:prop_end]
        if len(window_vel) == 0:
            continue
            
        peak_frame = -1
        peak_val = 0.0

        # Special logic for Knee (Extension rate) and Hip (Vertical velocity)
        if joint == "knee":
            # Knee Extension Rate
            # Angle: Hip-Knee-Ankle
            angle = compute_joint_angle(joints["hip"], joints["knee"], joints["ankle"])
            ext_rate = compute_velocity(angle)
            # Extension is positive rate (angle increasing towards 180)
            window_ext = ext_rate[prop_start:prop_end]
            if len(window_ext) > 0:
                idx_max = np.argmax(window_ext)
                peak_val = window_ext[idx_max]
                peak_frame = prop_start + idx_max
        elif joint == "hip":
            # Hip Vertical Velocity (Jump/Upward drive)
            window_vert = vert_velocities["hip"][prop_start:prop_end]
            if len(window_vert) > 0:
                idx_max = np.argmax(window_vert)
                peak_val = window_vert[idx_max]
                peak_frame = prop_start + idx_max
        else:
            # Resultant Velocity Magnitude for upper body
            idx_max = np.argmax(window_vel)
            peak_val = window_vel[idx_max]
            peak_frame = prop_start + idx_max
        
        peak_times[joint] = peak_frame
        peak_mags[joint] = peak_val
        
        features[f"kc_{joint}_peak_frame"] = peak_frame
        features[f"kc_{joint}_peak_mag"] = peak_val
        
        # Time relative to release (negative = before release)
        features[f"kc_{joint}_time_to_release"] = (peak_frame - release_frame) * DT

    # 6. Sequencing (Proximal-to-Distal Timing)
    # Ideally: Knee -> Hip -> Shoulder -> Elbow -> Wrist
    # Positive delta means "curr" happened AFTER "prev" (correct sequence)
    pairs = [("knee", "hip"), ("hip", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist")]
    
    for prev, curr in pairs:
        if prev in peak_times and curr in peak_times:
            # Timing delta
            delta_frames = peak_times[curr] - peak_times[prev]
            features[f"kc_seq_{prev}_to_{curr}_frames"] = delta_frames
            features[f"kc_seq_{prev}_to_{curr}_time"] = delta_frames * DT
            
            # Amplification Ratio (Curr / Prev)
            # How much was velocity increased by this segment?
            if abs(peak_mags[prev]) > 1e-6:
                ratio = peak_mags[curr] / peak_mags[prev]
                features[f"kc_amp_{prev}_to_{curr}_ratio"] = ratio

    # 7. Integrated Energy (Work)
    # Sum of squared velocities during propulsion
    for joint in chain_order:
        if joint in velocities:
            # Use simple sum of squares as proxy for kinetic energy (ignoring mass)
            energy = np.sum(velocities[joint][prop_start:release_frame]**2)
            features[f"kc_{joint}_energy_propulsion"] = energy

    # 8. Release Mechanics
    # Specific breakdown of velocity vector at release
    v_rel = compute_velocity(joints["wrist"])[release_frame]
    features["kc_wrist_vx_release"] = v_rel[0]
    features["kc_wrist_vy_release"] = v_rel[1] # Forward
    features["kc_wrist_vz_release"] = v_rel[2] # Vertical
    features["kc_wrist_vmag_release"] = np.linalg.norm(v_rel)
    
    # Angle of release
    # Elevation: Angle from horizontal plane
    v_horiz = np.sqrt(v_rel[0]**2 + v_rel[1]**2)
    features["kc_release_angle_elev"] = np.arctan2(v_rel[2], v_horiz) * 180 / np.pi
    
    # Azimuth: Left/Right deviation
    features["kc_release_angle_azi"] = np.arctan2(v_rel[0], v_rel[1]) * 180 / np.pi

    # 9. Smoothness (Jerk)
    # High jerk = jerky movement, low jerk = smooth
    acc_wrist = compute_acceleration(compute_velocity(joints["wrist"]))
    jerk_wrist = np.gradient(acc_wrist, DT, axis=0)
    jerk_mag = np.linalg.norm(jerk_wrist, axis=1)
    
    features["kc_wrist_jerk_release"] = jerk_mag[release_frame]
    features["kc_wrist_jerk_mean"] = np.mean(jerk_mag[prop_start:release_frame])

    return features

def get_kinetic_chain_feature_names() -> List[str]:
    """Return list of all feature names generated by this module."""
    # This is a helper to see what we have, though actual names depend on data existence
    names = [
        "kc_release_frame",
        "kc_wrist_vx_release", "kc_wrist_vy_release", "kc_wrist_vz_release", "kc_wrist_vmag_release",
        "kc_release_angle_elev", "kc_release_angle_azi",
        "kc_wrist_jerk_release", "kc_wrist_jerk_mean"
    ]
    joints = ["knee", "hip", "shoulder", "elbow", "wrist"]
    for j in joints:
        names.extend([
            f"kc_{j}_peak_frame", f"kc_{j}_peak_mag", f"kc_{j}_time_to_release", 
            f"kc_{j}_energy_propulsion"
        ])
    
    pairs = [("knee", "hip"), ("hip", "shoulder"), ("shoulder", "elbow"), ("elbow", "wrist")]
    for p, c in pairs:
        names.extend([
            f"kc_seq_{p}_to_{c}_frames", f"kc_seq_{p}_to_{c}_time", 
            f"kc_amp_{p}_to_{c}_ratio"
        ])
    
    return names
