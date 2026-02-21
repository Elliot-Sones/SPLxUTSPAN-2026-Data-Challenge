"""
Novel Feature Types Ensemble

Goal: Build a model with fundamentally NEW feature types that capture different
information than standard keypoint statistics, then blend with Sub 784 (LB 0.007224).

Feature types implemented:
1. Per-player deviation features - how each shot deviates from player's mean trajectory
2. Body segment ratio features - scale-invariant pose geometry via joint distance ratios
3. Kinetic chain features - energy transfer timing from ground up (peak velocity delays)
4. Release window shape features - polynomial fits, arc length, smoothness (jerk)
5. Bilateral symmetry features - left vs right side comparison at release
6. Shot consistency features - distance to k-nearest player shots in feature space

Pipeline: Per-player StandardScaler, per-player per-target 5-fold CV with
LGB+XGB+CatBoost+Ridge (0.3/0.3/0.3/0.1), blend with Sub 784.
"""

import json
import sys
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])

# Key joints used across feature extractors
KEY_JOINTS = [
    'right_wrist', 'right_elbow', 'right_shoulder',
    'left_wrist', 'left_elbow', 'left_shoulder',
    'right_hip', 'left_hip', 'mid_hip',
    'right_knee', 'left_knee',
    'right_ankle', 'left_ankle',
    'neck', 'nose'
]

# Body segment connections for segment ratio features
SEGMENTS = [
    ('right_shoulder', 'right_elbow', 'r_upper_arm'),
    ('right_elbow', 'right_wrist', 'r_forearm'),
    ('left_shoulder', 'left_elbow', 'l_upper_arm'),
    ('left_elbow', 'left_wrist', 'l_forearm'),
    ('right_shoulder', 'right_hip', 'r_torso'),
    ('left_shoulder', 'left_hip', 'l_torso'),
    ('right_hip', 'right_knee', 'r_thigh'),
    ('right_knee', 'right_ankle', 'r_shin'),
    ('left_hip', 'left_knee', 'l_thigh'),
    ('left_knee', 'left_ankle', 'l_shin'),
    ('neck', 'mid_hip', 'spine'),
    ('right_shoulder', 'left_shoulder', 'shoulder_width'),
    ('right_hip', 'left_hip', 'hip_width'),
]

# Kinetic chain joint sequence (ground up)
KINETIC_CHAIN = [
    ('right_ankle', 'right_knee'),
    ('right_knee', 'right_hip'),
    ('right_hip', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
]

# Release phase frames
RELEASE_START = 120
RELEASE_END = 180
RELEASE_WINDOW_START = 140
RELEASE_WINDOW_END = 180
RELEASE_FRAME = 153


def get_next_submission_number():
    """Atomically get the next submission number using a lock file."""
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
            placeholder.touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def load_raw_data():
    """Load train and test, returning 3D timeseries + metadata."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    keypoint_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            keypoint_names.append(col[:-2])

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}, "
          f"Keypoint cols: {len(keypoint_cols)}, Keypoints: {len(keypoint_names)}")

    def process(df, is_train=True):
        n = len(df)
        X_3d = np.zeros((n, 240, len(keypoint_names), 3), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                kp_idx = col_i // 3
                coord_idx = col_i % 3
                X_3d[idx, :, kp_idx, coord_idx] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        result = {
            'X_3d': X_3d,
            'pids': np.array(pids),
            'ids': np.array(ids),
            'keypoint_names': keypoint_names,
        }
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


# ============================================================
# FEATURE TYPE 1: Per-Player Deviation Features
# ============================================================

def compute_player_mean_trajectories(X_3d, pids, kp_index):
    """Compute mean trajectory per player for key joints."""
    unique_pids = sorted(np.unique(pids))
    # Store mean 3D trajectory per player per key joint: (240, 3)
    player_means = {}
    for pid in unique_pids:
        mask = pids == pid
        player_means[pid] = {}
        for joint in KEY_JOINTS:
            if joint not in kp_index:
                continue
            jidx = kp_index[joint]
            # mean across all player's shots: (240, 3)
            mean_traj = np.nanmean(X_3d[mask, :, jidx, :], axis=0)
            player_means[pid][joint] = mean_traj
    return player_means


def extract_deviation_features(X_3d_single, pid, player_means, kp_index):
    """
    For a single shot, compute deviation from player's mean trajectory.
    Focus on release phase (frames 120-180).
    """
    feats = {}
    pm = player_means.get(pid, {})

    for joint in KEY_JOINTS:
        if joint not in kp_index or joint not in pm:
            continue
        jidx = kp_index[joint]
        shot_traj = X_3d_single[:, jidx, :]  # (240, 3)
        mean_traj = pm[joint]  # (240, 3)

        # 3D deviation at each frame
        dev = shot_traj - mean_traj  # (240, 3)
        dev_norm = np.linalg.norm(dev, axis=1)  # (240,)

        # Release phase stats (frames 120-180)
        rp = dev_norm[RELEASE_START:RELEASE_END]
        feats[f"dev_{joint}_release_mean"] = np.nanmean(rp)
        feats[f"dev_{joint}_release_std"] = np.nanstd(rp)
        feats[f"dev_{joint}_release_max"] = np.nanmax(rp)

        # Per-axis deviation stats in release phase
        for c, cname in enumerate(['x', 'y', 'z']):
            d = dev[RELEASE_START:RELEASE_END, c]
            feats[f"dev_{joint}_{cname}_release_mean"] = np.nanmean(d)
            feats[f"dev_{joint}_{cname}_release_std"] = np.nanstd(d)

        # Deviation at release frame
        feats[f"dev_{joint}_at_release"] = dev_norm[RELEASE_FRAME]

        # Deviation velocity: is deviation growing or shrinking?
        dev_vel = np.gradient(dev_norm)
        feats[f"dev_{joint}_vel_at_release"] = dev_vel[RELEASE_FRAME]
        feats[f"dev_{joint}_vel_release_mean"] = np.nanmean(dev_vel[RELEASE_START:RELEASE_END])

    return feats


# ============================================================
# FEATURE TYPE 2: Body Segment Ratio Features
# ============================================================

def compute_segment_length(X_3d_single, kp_index, joint_a, joint_b, frame):
    """Compute distance between two joints at a specific frame."""
    if joint_a not in kp_index or joint_b not in kp_index:
        return np.nan
    a = X_3d_single[frame, kp_index[joint_a], :]
    b = X_3d_single[frame, kp_index[joint_b], :]
    return np.linalg.norm(a - b)


def extract_segment_ratio_features(X_3d_single, kp_index):
    """
    Compute body segment lengths and ratios at key frames.
    Scale-invariant via ratios.
    """
    feats = {}
    key_frames = [60, 120, RELEASE_FRAME, 170]  # loading, pre-release, release, follow-through

    for frame in key_frames:
        fname = f"f{frame}"
        lengths = {}
        for ja, jb, seg_name in SEGMENTS:
            length = compute_segment_length(X_3d_single, kp_index, ja, jb, frame)
            lengths[seg_name] = length
            feats[f"seg_{seg_name}_{fname}"] = length

        # Compute meaningful ratios
        ratio_pairs = [
            ('r_forearm', 'r_upper_arm', 'r_forearm_upper_ratio'),
            ('l_forearm', 'l_upper_arm', 'l_forearm_upper_ratio'),
            ('r_shin', 'r_thigh', 'r_shin_thigh_ratio'),
            ('l_shin', 'l_thigh', 'l_shin_thigh_ratio'),
            ('r_forearm', 'r_torso', 'r_arm_torso_ratio'),
            ('shoulder_width', 'hip_width', 'shoulder_hip_ratio'),
            ('spine', 'r_thigh', 'spine_thigh_ratio'),
            ('r_upper_arm', 'spine', 'arm_spine_ratio'),
        ]

        for seg_a, seg_b, ratio_name in ratio_pairs:
            la = lengths.get(seg_a, np.nan)
            lb = lengths.get(seg_b, np.nan)
            if lb is not None and not np.isnan(lb) and abs(lb) > 1e-6:
                feats[f"ratio_{ratio_name}_{fname}"] = la / lb
            else:
                feats[f"ratio_{ratio_name}_{fname}"] = 0.0

    # Change in segment lengths between loading and release
    for ja, jb, seg_name in SEGMENTS:
        l_load = compute_segment_length(X_3d_single, kp_index, ja, jb, 120)
        l_rel = compute_segment_length(X_3d_single, kp_index, ja, jb, RELEASE_FRAME)
        feats[f"seg_change_{seg_name}"] = l_rel - l_load

    return feats


# ============================================================
# FEATURE TYPE 3: Kinetic Chain Timing Features
# ============================================================

def extract_kinetic_chain_features(X_3d_single, kp_index):
    """
    Extract peak velocity timing for each joint in the kinetic chain.
    Compute delays between consecutive peaks (energy transfer timing).
    """
    feats = {}
    fps = 60.0

    # Compute velocity magnitude for each joint in chain, focused on release phase
    analysis_start = 100
    analysis_end = 200
    peak_frames = {}

    for joint_a, joint_b in KINETIC_CHAIN:
        # We track the distal joint (joint_b) velocity
        if joint_b not in kp_index:
            continue
        jidx = kp_index[joint_b]
        pos = X_3d_single[analysis_start:analysis_end, jidx, :]  # (100, 3)
        vel = np.gradient(pos, 1.0 / fps, axis=0)  # (100, 3)
        speed = np.linalg.norm(vel, axis=1)  # (100,)

        # Find peak velocity frame (relative to analysis_start)
        peak_rel = np.argmax(speed)
        peak_abs = peak_rel + analysis_start

        peak_frames[joint_b] = peak_abs
        feats[f"kc_peak_frame_{joint_b}"] = peak_abs
        feats[f"kc_peak_speed_{joint_b}"] = speed[peak_rel]

        # Velocity stats
        feats[f"kc_mean_speed_{joint_b}"] = np.nanmean(speed)
        feats[f"kc_speed_std_{joint_b}"] = np.nanstd(speed)

        # Acceleration at peak
        acc = np.gradient(speed, 1.0 / fps)
        feats[f"kc_acc_at_peak_{joint_b}"] = acc[peak_rel]

    # Compute delays between consecutive peaks
    chain_joints = [jb for _, jb in KINETIC_CHAIN]
    for i in range(len(chain_joints) - 1):
        j_lower = chain_joints[i]
        j_upper = chain_joints[i + 1]
        if j_lower in peak_frames and j_upper in peak_frames:
            delay = peak_frames[j_upper] - peak_frames[j_lower]
            feats[f"kc_delay_{j_lower}_to_{j_upper}"] = delay / fps  # in seconds
        else:
            feats[f"kc_delay_{j_lower}_to_{j_upper}"] = 0.0

    # Total chain delay (ankle to wrist)
    first_joint = chain_joints[0]
    last_joint = chain_joints[-1]
    if first_joint in peak_frames and last_joint in peak_frames:
        feats["kc_total_chain_delay"] = (peak_frames[last_joint] - peak_frames[first_joint]) / fps
    else:
        feats["kc_total_chain_delay"] = 0.0

    # Is the chain sequential? (each peak after the previous)
    sequential = True
    for i in range(len(chain_joints) - 1):
        j1 = chain_joints[i]
        j2 = chain_joints[i + 1]
        if j1 in peak_frames and j2 in peak_frames:
            if peak_frames[j2] <= peak_frames[j1]:
                sequential = False
                break
    feats["kc_is_sequential"] = float(sequential)

    return feats


# ============================================================
# FEATURE TYPE 4: Release Window Shape Features
# ============================================================

def extract_release_shape_features(X_3d_single, kp_index):
    """
    Fit polynomials to wrist trajectory in release window.
    Compute arc length, smoothness (jerk), curvature.
    """
    feats = {}
    fps = 60.0

    for joint_name in ['right_wrist', 'right_elbow']:
        if joint_name not in kp_index:
            continue
        jidx = kp_index[joint_name]
        prefix = joint_name.replace('right_', 'r_')

        pos = X_3d_single[RELEASE_WINDOW_START:RELEASE_WINDOW_END, jidx, :]  # (40, 3)
        n_frames = pos.shape[0]
        t = np.arange(n_frames) / fps

        for c, cname in enumerate(['x', 'y', 'z']):
            series = pos[:, c]

            # Handle NaN
            valid = ~np.isnan(series)
            if valid.sum() < 5:
                feats[f"shape_{prefix}_{cname}_poly_c0"] = 0.0
                feats[f"shape_{prefix}_{cname}_poly_c1"] = 0.0
                feats[f"shape_{prefix}_{cname}_poly_c2"] = 0.0
                feats[f"shape_{prefix}_{cname}_residual"] = 0.0
                continue

            # Fit 2nd degree polynomial
            try:
                coeffs = np.polyfit(t[valid], series[valid], 2)
                fitted = np.polyval(coeffs, t[valid])
                residual = np.sqrt(np.mean((series[valid] - fitted) ** 2))
            except np.linalg.LinAlgError:
                coeffs = [0.0, 0.0, 0.0]
                residual = 0.0

            feats[f"shape_{prefix}_{cname}_poly_c0"] = coeffs[0]  # curvature (quadratic term)
            feats[f"shape_{prefix}_{cname}_poly_c1"] = coeffs[1]  # slope
            feats[f"shape_{prefix}_{cname}_poly_c2"] = coeffs[2]  # intercept
            feats[f"shape_{prefix}_{cname}_residual"] = residual

        # 3D arc length of wrist path
        diffs = np.diff(pos, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        arc_length = np.nansum(step_lengths)
        feats[f"shape_{prefix}_arc_length"] = arc_length

        # Straight-line distance (start to end)
        straight = np.linalg.norm(pos[-1] - pos[0])
        feats[f"shape_{prefix}_straight_dist"] = straight

        # Curvature ratio: arc_length / straight_dist
        if straight > 1e-6:
            feats[f"shape_{prefix}_curvature_ratio"] = arc_length / straight
        else:
            feats[f"shape_{prefix}_curvature_ratio"] = 1.0

        # Smoothness via jerk (3rd derivative of position)
        vel = np.gradient(pos, 1.0 / fps, axis=0)  # (40, 3)
        acc = np.gradient(vel, 1.0 / fps, axis=0)  # (40, 3)
        jerk = np.gradient(acc, 1.0 / fps, axis=0)  # (40, 3)
        jerk_mag = np.linalg.norm(jerk, axis=1)  # (40,)

        feats[f"shape_{prefix}_jerk_mean"] = np.nanmean(jerk_mag)
        feats[f"shape_{prefix}_jerk_max"] = np.nanmax(jerk_mag)
        feats[f"shape_{prefix}_jerk_std"] = np.nanstd(jerk_mag)

        # Smoothness metric: normalized jerk (lower = smoother)
        # Dimensionless smoothness: (jerk^2 * duration^5 / arc_length^2)
        duration = n_frames / fps
        jerk_sq_sum = np.nansum(jerk_mag ** 2) / fps
        if arc_length > 1e-6:
            feats[f"shape_{prefix}_norm_jerk"] = np.sqrt(
                jerk_sq_sum * (duration ** 5) / (arc_length ** 2)
            )
        else:
            feats[f"shape_{prefix}_norm_jerk"] = 0.0

        # Velocity profile shape: peak velocity timing within window
        speed = np.linalg.norm(vel, axis=1)
        peak_idx = np.argmax(speed)
        feats[f"shape_{prefix}_peak_vel_timing"] = peak_idx / n_frames  # 0-1 normalized
        feats[f"shape_{prefix}_peak_vel_value"] = speed[peak_idx]

    return feats


# ============================================================
# FEATURE TYPE 5: Bilateral Symmetry Features
# ============================================================

def extract_bilateral_symmetry_features(X_3d_single, kp_index):
    """
    Compare left vs right side at release and during release phase.
    """
    feats = {}

    symmetric_pairs = [
        ('right_shoulder', 'left_shoulder', 'shoulder'),
        ('right_hip', 'left_hip', 'hip'),
        ('right_elbow', 'left_elbow', 'elbow'),
        ('right_wrist', 'left_wrist', 'wrist'),
        ('right_knee', 'left_knee', 'knee'),
        ('right_ankle', 'left_ankle', 'ankle'),
    ]

    for rj, lj, name in symmetric_pairs:
        if rj not in kp_index or lj not in kp_index:
            continue
        ridx, lidx = kp_index[rj], kp_index[lj]

        # Height difference (z-axis) at release
        r_z = X_3d_single[RELEASE_FRAME, ridx, 2]
        l_z = X_3d_single[RELEASE_FRAME, lidx, 2]
        feats[f"sym_{name}_height_diff_release"] = r_z - l_z

        # Height difference mean over release phase
        r_z_rp = X_3d_single[RELEASE_START:RELEASE_END, ridx, 2]
        l_z_rp = X_3d_single[RELEASE_START:RELEASE_END, lidx, 2]
        feats[f"sym_{name}_height_diff_release_mean"] = np.nanmean(r_z_rp - l_z_rp)

        # Lateral (x-axis) symmetry
        r_x = X_3d_single[RELEASE_FRAME, ridx, 0]
        l_x = X_3d_single[RELEASE_FRAME, lidx, 0]
        feats[f"sym_{name}_lateral_diff_release"] = r_x - l_x

        # Distance between left and right at release
        dist = np.linalg.norm(
            X_3d_single[RELEASE_FRAME, ridx, :] - X_3d_single[RELEASE_FRAME, lidx, :]
        )
        feats[f"sym_{name}_dist_release"] = dist

    # Hip rotation angle (in XY plane)
    if 'right_hip' in kp_index and 'left_hip' in kp_index:
        rh = X_3d_single[RELEASE_FRAME, kp_index['right_hip'], :2]
        lh = X_3d_single[RELEASE_FRAME, kp_index['left_hip'], :2]
        hip_vec = rh - lh
        feats["sym_hip_rotation"] = np.arctan2(hip_vec[1], hip_vec[0])

        # Hip rotation change from loading to release
        rh_load = X_3d_single[60, kp_index['right_hip'], :2]
        lh_load = X_3d_single[60, kp_index['left_hip'], :2]
        hip_vec_load = rh_load - lh_load
        angle_load = np.arctan2(hip_vec_load[1], hip_vec_load[0])
        angle_rel = np.arctan2(hip_vec[1], hip_vec[0])
        feats["sym_hip_rotation_change"] = angle_rel - angle_load

    # Shoulder rotation angle
    if 'right_shoulder' in kp_index and 'left_shoulder' in kp_index:
        rs = X_3d_single[RELEASE_FRAME, kp_index['right_shoulder'], :2]
        ls = X_3d_single[RELEASE_FRAME, kp_index['left_shoulder'], :2]
        sh_vec = rs - ls
        feats["sym_shoulder_rotation"] = np.arctan2(sh_vec[1], sh_vec[0])

        # Shoulder-hip rotation difference
        if 'right_hip' in kp_index and 'left_hip' in kp_index:
            rh = X_3d_single[RELEASE_FRAME, kp_index['right_hip'], :2]
            lh = X_3d_single[RELEASE_FRAME, kp_index['left_hip'], :2]
            hip_angle = np.arctan2((rh - lh)[1], (rh - lh)[0])
            sh_angle = np.arctan2(sh_vec[1], sh_vec[0])
            feats["sym_torso_twist"] = sh_angle - hip_angle

    # Guide hand (left wrist) position relative to shooting hand (right wrist) at release
    if 'right_wrist' in kp_index and 'left_wrist' in kp_index:
        rw = X_3d_single[RELEASE_FRAME, kp_index['right_wrist'], :]
        lw = X_3d_single[RELEASE_FRAME, kp_index['left_wrist'], :]
        guide_vec = lw - rw
        feats["sym_guide_hand_dist"] = np.linalg.norm(guide_vec)
        feats["sym_guide_hand_x"] = guide_vec[0]
        feats["sym_guide_hand_y"] = guide_vec[1]
        feats["sym_guide_hand_z"] = guide_vec[2]

        # Guide hand velocity at release
        for joint, prefix in [('left_wrist', 'guide'), ('right_wrist', 'shoot')]:
            jidx = kp_index[joint]
            vel = np.gradient(X_3d_single[:, jidx, :], 1.0 / 60.0, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            feats[f"sym_{prefix}_speed_at_release"] = speed[RELEASE_FRAME]

        # Speed ratio: guide/shoot
        guide_speed = feats.get("sym_guide_speed_at_release", 0.0)
        shoot_speed = feats.get("sym_shoot_speed_at_release", 1e-6)
        if abs(shoot_speed) > 1e-6:
            feats["sym_guide_shoot_speed_ratio"] = guide_speed / shoot_speed
        else:
            feats["sym_guide_shoot_speed_ratio"] = 1.0

    return feats


# ============================================================
# FEATURE TYPE 6: Shot Consistency Features (per player)
# ============================================================

def compute_base_features_for_consistency(X_3d_single, kp_index):
    """
    Compute a compact feature vector per shot for nearest-neighbor consistency check.
    Uses release-phase positions and velocities of key joints.
    """
    vec = []
    for joint in KEY_JOINTS:
        if joint not in kp_index:
            continue
        jidx = kp_index[joint]
        # Release frame position
        pos = X_3d_single[RELEASE_FRAME, jidx, :]
        vec.extend(pos.tolist())
        # Mean release phase velocity
        vel = np.gradient(X_3d_single[RELEASE_START:RELEASE_END, jidx, :], 1.0 / 60.0, axis=0)
        vec.extend(np.nanmean(vel, axis=0).tolist())
    return np.array(vec, dtype=np.float32)


def extract_consistency_features(base_feat, player_base_feats, k=5):
    """
    Compute distance from this shot to its k-nearest neighbors among the same player's shots.
    """
    feats = {}
    n_player = len(player_base_feats)
    if n_player < 2:
        feats["consist_mean_dist_k"] = 0.0
        feats["consist_min_dist"] = 0.0
        feats["consist_max_dist_k"] = 0.0
        feats["consist_std_dist_k"] = 0.0
        feats["consist_centroid_dist"] = 0.0
        return feats

    actual_k = min(k, n_player - 1)
    nn = NearestNeighbors(n_neighbors=actual_k + 1, metric='euclidean')
    nn.fit(player_base_feats)
    dists, _ = nn.kneighbors(base_feat.reshape(1, -1))
    # Skip first neighbor (self) if present
    dists = dists[0, 1:actual_k + 1]

    feats["consist_mean_dist_k"] = np.mean(dists)
    feats["consist_min_dist"] = np.min(dists)
    feats["consist_max_dist_k"] = np.max(dists)
    feats["consist_std_dist_k"] = np.std(dists)

    # Distance to player centroid
    centroid = np.mean(player_base_feats, axis=0)
    feats["consist_centroid_dist"] = np.linalg.norm(base_feat - centroid)

    return feats


# ============================================================
# COMBINED EXTRACTION
# ============================================================

def extract_all_novel_features(X_3d, pids, keypoint_names, player_means=None,
                                player_base_feats_map=None, is_train=True):
    """
    Extract all 6 types of novel features for all samples.

    For training: computes player_means and player_base_feats internally.
    For test: uses pre-computed player_means and player_base_feats from training.
    """
    kp_index = {name: i for i, name in enumerate(keypoint_names)}
    n = len(X_3d)

    # Compute player means if training
    if player_means is None:
        player_means = compute_player_mean_trajectories(X_3d, pids, kp_index)

    # Compute base features for consistency
    base_feats_all = []
    for i in range(n):
        base_feats_all.append(compute_base_features_for_consistency(X_3d[i], kp_index))
    base_feats_all = np.array(base_feats_all)
    base_feats_all = np.nan_to_num(base_feats_all, nan=0.0)

    if player_base_feats_map is None:
        player_base_feats_map = {}
        unique_pids = sorted(np.unique(pids))
        for pid in unique_pids:
            mask = pids == pid
            player_base_feats_map[pid] = base_feats_all[mask]

    all_feats = []
    for i in range(n):
        feats = {}
        pid = pids[i]
        ts = X_3d[i]  # (240, n_keypoints, 3)

        # Type 1: Per-player deviation
        f1 = extract_deviation_features(ts, pid, player_means, kp_index)
        feats.update(f1)

        # Type 2: Body segment ratios
        f2 = extract_segment_ratio_features(ts, kp_index)
        feats.update(f2)

        # Type 3: Kinetic chain timing
        f3 = extract_kinetic_chain_features(ts, kp_index)
        feats.update(f3)

        # Type 4: Release window shape
        f4 = extract_release_shape_features(ts, kp_index)
        feats.update(f4)

        # Type 5: Bilateral symmetry
        f5 = extract_bilateral_symmetry_features(ts, kp_index)
        feats.update(f5)

        # Type 6: Shot consistency
        f6 = extract_consistency_features(base_feats_all[i], player_base_feats_map[pid], k=5)
        feats.update(f6)

        all_feats.append(feats)

        if (i + 1) % 100 == 0:
            print(f"    Extracted {i + 1}/{n}")

    # Convert to matrix
    feat_names = sorted(all_feats[0].keys())
    X_feat = np.array(
        [[f.get(name, 0.0) for name in feat_names] for f in all_feats],
        dtype=np.float32
    )
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Total novel features: {len(feat_names)}")

    # Print feature type counts
    type_prefixes = {
        'Deviation': 'dev_',
        'Segment/Ratio': ('seg_', 'ratio_'),
        'Kinetic Chain': 'kc_',
        'Release Shape': 'shape_',
        'Bilateral Symmetry': 'sym_',
        'Consistency': 'consist_',
    }
    for tname, prefix in type_prefixes.items():
        if isinstance(prefix, tuple):
            count = sum(1 for f in feat_names if any(f.startswith(p) for p in prefix))
        else:
            count = sum(1 for f in feat_names if f.startswith(prefix))
        print(f"      {tname}: {count}")

    return X_feat, feat_names, player_means, player_base_feats_map


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_per_player_per_target(X, y, pids, target_idx, target_name):
    """Per-player per-target ensemble with 5-fold CV."""
    unique_pids = sorted(np.unique(pids))
    oof_preds = np.zeros(len(y))
    models = {}
    scalers = {}

    print(f"\n--- {target_name.upper()} ---")

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_t = y[mask, target_idx]
        n = len(X_p)
        global_idx = np.where(mask)[0]

        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X_p), nan=0.0)
        scalers[pid] = scaler

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(n)

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr = y_t[tr_idx]

            preds = []

            # LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1
            )
            lgb_model.fit(X_tr, y_tr)
            preds.append(lgb_model.predict(X_val))

            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1
            )
            xgb_model.fit(X_tr, y_tr)
            preds.append(xgb_model.predict(X_val))

            # CatBoost
            cat_model = CatBoostRegressor(
                iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False
            )
            cat_model.fit(X_tr, y_tr)
            preds.append(cat_model.predict(X_val))

            # Ridge
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_tr, y_tr)
            preds.append(ridge_model.predict(X_val))

            # Ensemble: 0.3*LGB + 0.3*XGB + 0.3*Cat + 0.1*Ridge
            fold_preds[val_idx] = (
                0.3 * preds[0] + 0.3 * preds[1] + 0.3 * preds[2] + 0.1 * preds[3]
            )

        oof_preds[global_idx] = fold_preds
        cv_mse = np.mean((fold_preds - y_t) ** 2)
        print(f"  Player {pid}: n={n}, CV MSE = {cv_mse:.6f}")

        # Train final models on all player data
        pid_models = {}
        for name, cls, params in [
            ('lgb', lgb.LGBMRegressor, dict(n_estimators=100, num_leaves=10, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)),
            ('xgb', xgb.XGBRegressor, dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbosity=0, n_jobs=-1)),
            ('cat', CatBoostRegressor, dict(iterations=100, depth=4, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=False)),
            ('ridge', Ridge, dict(alpha=1.0)),
        ]:
            m = cls(**params)
            m.fit(X_scaled, y_t)
            pid_models[name] = m

        models[pid] = pid_models

    overall_mse = np.mean((oof_preds - y[:, target_idx]) ** 2)
    print(f"  Overall {target_name} CV MSE: {overall_mse:.6f}")
    return oof_preds, models, scalers, overall_mse


def predict_test(X_test, pids_test, models, scalers):
    """Generate test predictions."""
    preds = np.zeros(len(X_test))
    for i in range(len(X_test)):
        pid = pids_test[i]
        x = X_test[i:i+1]
        x_scaled = np.nan_to_num(scalers[pid].transform(x), nan=0.0)
        p = [models[pid][n].predict(x_scaled)[0] for n in ['lgb', 'xgb', 'cat', 'ridge']]
        preds[i] = 0.3 * p[0] + 0.3 * p[1] + 0.3 * p[2] + 0.1 * p[3]
    return preds


def scale_predictions(raw_preds, target):
    """Scale raw predictions to [0,1] using the target scaler."""
    scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scaler.transform(raw_preds.reshape(-1, 1)).flatten()


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("NOVEL FEATURES ENSEMBLE")
    print("=" * 70)
    print()

    # Load data
    train_data, test_data = load_raw_data()

    y = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    test_ids = test_data['ids']
    keypoint_names = train_data['keypoint_names']

    # Extract novel features for train
    print("\nExtracting novel features (train)...")
    X_train, feat_names, player_means, player_base_feats_map = extract_all_novel_features(
        train_data['X_3d'], pids_train, keypoint_names, is_train=True
    )

    # Extract novel features for test (use training player means for deviation features)
    print("\nExtracting novel features (test)...")
    X_test, _, _, _ = extract_all_novel_features(
        test_data['X_3d'], pids_test, keypoint_names,
        player_means=player_means, player_base_feats_map=player_base_feats_map,
        is_train=False
    )

    print(f"\nFeature shapes: train={X_train.shape}, test={X_test.shape}")

    # Train per-player per-target
    print("\n" + "=" * 70)
    print("TRAINING PER-PLAYER PER-TARGET MODELS")
    print("=" * 70)

    results = {}
    all_oof = {}
    all_models = {}
    all_scalers = {}

    for t_idx, target in enumerate(TARGETS):
        oof, models, scalers, cv_mse = train_per_player_per_target(
            X_train, y, pids_train, t_idx, target
        )
        all_oof[target] = oof
        all_models[target] = models
        all_scalers[target] = scalers
        results[target] = cv_mse

    # Overall CV
    total_cv = np.mean([results[t] for t in TARGETS])
    print(f"\n{'=' * 70}")
    print(f"OVERALL CV RESULTS")
    print(f"{'=' * 70}")
    for t in TARGETS:
        print(f"  {t}: CV MSE = {results[t]:.6f}")
    print(f"  Mean CV MSE: {total_cv:.6f}")

    # Per-player per-target breakdown
    print(f"\nPer-player per-target CV MSE:")
    unique_pids = sorted(np.unique(pids_train))
    header = f"  {'Player':>8}"
    for t in TARGETS:
        header += f"  {t:>12}"
    header += f"  {'mean':>12}"
    print(header)
    for pid in unique_pids:
        mask = pids_train == pid
        line = f"  {pid:>8}"
        pid_mses = []
        for t_idx, t in enumerate(TARGETS):
            mse = np.mean((all_oof[t][mask] - y[mask, t_idx]) ** 2)
            pid_mses.append(mse)
            line += f"  {mse:>12.6f}"
        line += f"  {np.mean(pid_mses):>12.6f}"
        print(line)

    # Generate test predictions
    print(f"\n{'=' * 70}")
    print("GENERATING TEST PREDICTIONS")
    print(f"{'=' * 70}")

    raw_preds = {}
    scaled_preds = {}
    for target in TARGETS:
        raw = predict_test(X_test, pids_test, all_models[target], all_scalers[target])
        raw_preds[target] = raw
        scaled = scale_predictions(raw, target)
        scaled_preds[target] = scaled
        print(f"  {target}: raw range [{raw.min():.4f}, {raw.max():.4f}] -> "
              f"scaled [{scaled.min():.4f}, {scaled.max():.4f}]")

    # Blend with Sub 784
    print(f"\n{'=' * 70}")
    print("BLENDING WITH SUB 784 (LB 0.007224)")
    print(f"{'=' * 70}")

    sub784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    our = pd.DataFrame({
        'id': test_ids,
        'new_angle': scaled_preds['angle'],
        'new_depth': scaled_preds['depth'],
        'new_left_right': scaled_preds['left_right'],
    })
    merged = sub784.merge(our, on='id')

    # Correlations
    print("\nCorrelations with Sub 784:")
    corr_map = {}
    for col, nc, target in [
        ('scaled_angle', 'new_angle', 'angle'),
        ('scaled_depth', 'new_depth', 'depth'),
        ('scaled_left_right', 'new_left_right', 'left_right'),
    ]:
        r = np.corrcoef(merged[col], merged[nc])[0, 1]
        corr_map[target] = r
        print(f"  {target}: r = {r:.4f}")

    # Find lowest-correlation target for target-specific blending
    lowest_corr_target = min(corr_map, key=corr_map.get)
    print(f"\n  Lowest correlation target: {lowest_corr_target} (r = {corr_map[lowest_corr_target]:.4f})")
    print(f"  -> Will use this for target-specific blending")

    # Blend at multiple weights
    print(f"\n{'=' * 70}")
    print("UNIFORM BLENDING (all 3 targets at same weight)")
    print(f"{'=' * 70}")

    blend_weights = [0.10, 0.20, 0.30, 0.50]
    submissions = []

    col_map = {'angle': 'scaled_angle', 'depth': 'scaled_depth', 'left_right': 'scaled_left_right'}
    new_map = {'angle': 'new_angle', 'depth': 'new_depth', 'left_right': 'new_left_right'}

    print(f"\n  {'w':>5} | {'div_angle':>10} {'div_depth':>10} {'div_lr':>10} | {'total_div':>10}")
    print(f"  " + "-" * 65)

    for w in blend_weights:
        blended = {}
        for target in TARGETS:
            blended[target] = (1 - w) * merged[col_map[target]] + w * merged[new_map[target]]

        divs = {}
        for target in TARGETS:
            divs[target] = np.mean((blended[target] - merged[col_map[target]].values) ** 2)
        div_total = sum(divs.values())

        print(f"  {w:>5.2f} | {divs['angle']:>10.6f} {divs['depth']:>10.6f} {divs['left_right']:>10.6f} | {div_total:>10.6f}")

        submissions.append({
            'w': w,
            'type': 'uniform',
            'blended': {t: blended[t].values for t in TARGETS},
            'diversity': div_total,
        })

    # Target-specific blending: only blend the lowest-correlation target
    print(f"\n{'=' * 70}")
    print(f"TARGET-SPECIFIC BLENDING (only {lowest_corr_target} at varying weights)")
    print(f"{'=' * 70}")

    target_weights = [0.10, 0.15, 0.20, 0.30, 0.40]

    print(f"\n  {'w':>5} | {'div_target':>12} | {'target':>10}")
    print(f"  " + "-" * 40)

    for w in target_weights:
        blended = {}
        for target in TARGETS:
            if target == lowest_corr_target:
                blended[target] = (1 - w) * merged[col_map[target]] + w * merged[new_map[target]]
            else:
                blended[target] = merged[col_map[target]].values  # keep Sub 784

        div_t = np.mean((blended[lowest_corr_target] - merged[col_map[lowest_corr_target]].values) ** 2)
        print(f"  {w:>5.2f} | {div_t:>12.6f} | {lowest_corr_target}")

        submissions.append({
            'w': w,
            'type': f'target_specific_{lowest_corr_target}',
            'blended': {t: (blended[t] if isinstance(blended[t], np.ndarray) else blended[t].values) for t in TARGETS},
            'diversity': div_t,
        })

    # Save all submissions
    print(f"\n{'=' * 70}")
    print("SAVING SUBMISSIONS")
    print(f"{'=' * 70}")

    for s in submissions:
        sub_num = get_next_submission_number()
        sub_df = pd.DataFrame({
            'id': merged['id'],
            'scaled_angle': s['blended']['angle'],
            'scaled_depth': s['blended']['depth'],
            'scaled_left_right': s['blended']['left_right'],
        })
        filepath = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_df.to_csv(filepath, index=False)

        if s['type'] == 'uniform':
            desc = f"Novel features blend: (1-{s['w']:.2f})*sub784 + {s['w']:.2f}*novel (all 3 targets)"
        else:
            desc = (f"Novel features target-specific: only {lowest_corr_target} blended "
                    f"at w={s['w']:.2f}, others from sub784")

        print(f"\n  Sub {sub_num}: {desc}")
        print(f"    -> {filepath}")
        print(f"    Model: Novel features ensemble (deviation, segment ratios, kinetic chain, "
              f"release shape, bilateral symmetry, consistency)")
        print(f"    Config: Per-player per-target, 5-fold CV, LGB/XGB/Cat/Ridge (0.3/0.3/0.3/0.1)")
        print(f"    Features: {X_train.shape[1]} novel features")
        print(f"    CV MSE: angle={results['angle']:.6f}, depth={results['depth']:.6f}, "
              f"lr={results['left_right']:.6f}, mean={total_cv:.6f}")
        print(f"    Correlations with Sub 784: angle={corr_map['angle']:.4f}, "
              f"depth={corr_map['depth']:.4f}, lr={corr_map['left_right']:.4f}")
        print(f"    Diversity (MSE vs Sub 784): {s['diversity']:.6f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
