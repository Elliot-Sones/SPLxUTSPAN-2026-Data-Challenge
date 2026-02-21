"""
Per-Player Feature Importance - Comprehensive Analysis

Core question: Do different players encode shot outcome information
in different body segments? Which specific keypoints predict each
player's angle/depth/LR?

Approach:
1. Extract 1000+ features: raw keypoints, body-normalized, hoop-relative,
   velocities, accelerations, joint angles - at frames 120-180
2. Per-player correlation analysis: for each player x target, rank ALL features
3. Identify player-specific vs universal features
4. Per-player feature selection model: each player uses their own top-k features
5. Test if per-player feature selection beats the global model

Feature types:
- Raw 3D positions at 8 frames (120,130,140,150,153,160,170,180)
- Body-normalized positions (relative to mid-hip)
- Hoop-relative positions (hoop-centered coordinate system)
- Velocities (frame differences, smoothed)
- Accelerations (second differences)
- Joint angles: elbow, shoulder, wrist-shoulder, knee, hip
- Kinematic chain ratios and distances
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
ANALYSIS_FRAMES = [120, 130, 140, 145, 150, 153, 155, 160, 165, 170, 175, 180]


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split('_')[1]) for fp in existing
                    if fp.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window=11, polyorder=3, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])

        X_3d = np.nan_to_num(X_3d, nan=0.0)
        result = {'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    print("  Parsing train...")
    train = process(train_df, True)
    print("  Parsing test...")
    test = process(test_df, False)
    return train, test


# ============================================================
# COMPREHENSIVE FEATURE EXTRACTION
# ============================================================

def angle_3pt(a, b, c):
    """Angle at point b formed by a-b-c (in degrees)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def get_hoop_transform(ts_3d, kp_index):
    """Hoop-relative coordinate transform."""
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0
    forward = HOOP_POS[:2] - player_pos[:2]
    fn = np.linalg.norm(forward)
    if fn > 1e-6:
        forward /= fn
    else:
        forward = np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]
    centered = ts_3d - player_pos.reshape(1, 1, 3)
    return np.einsum('ij,fkj->fki', R, centered), player_pos


def extract_all_features(ts_3d, kp_index):
    """
    Extract comprehensive feature set for one shot.
    Returns: feature vector and feature names.
    """
    feats = []
    feat_names = []

    # Smooth trajectories first
    n_kp = ts_3d.shape[1]
    ts_smooth = ts_3d.copy()
    for k in range(n_kp):
        for ax in range(3):
            ts_smooth[:, k, ax] = safe_savgol(ts_3d[:, k, ax])

    # Compute velocities and accelerations
    vel = np.gradient(ts_smooth, axis=0) / DT  # frames x kp x 3
    acc = np.gradient(vel, axis=0) / DT

    # Hoop-relative coordinates
    ts_hr, player_pos = get_hoop_transform(ts_smooth, kp_index)

    # Body-normalized: relative to mid-hip
    mid_hip_idx = kp_index.get('mid_hip', 0)
    shoulder_l_idx = kp_index.get('left_shoulder')
    shoulder_r_idx = kp_index.get('right_shoulder')
    hip_l_idx = kp_index.get('left_hip')
    hip_r_idx = kp_index.get('right_hip')
    elbow_r_idx = kp_index.get('right_elbow')
    elbow_l_idx = kp_index.get('left_elbow')
    wrist_r_idx = kp_index.get('right_wrist')
    wrist_l_idx = kp_index.get('left_wrist')
    knee_r_idx = kp_index.get('right_knee')
    knee_l_idx = kp_index.get('left_knee')
    ankle_r_idx = kp_index.get('right_ankle')
    ankle_l_idx = kp_index.get('left_ankle')
    nose_idx = kp_index.get('nose')
    neck_idx = kp_index.get('neck')

    # Mid-shoulder for body normalization
    if shoulder_l_idx is not None and shoulder_r_idx is not None:
        mid_shoulder = (ts_smooth[:, shoulder_l_idx, :] + ts_smooth[:, shoulder_r_idx, :]) / 2
    else:
        mid_shoulder = ts_smooth[:, mid_hip_idx, :]

    # Body scale: shoulder width
    if shoulder_l_idx is not None and shoulder_r_idx is not None:
        shoulder_width = np.linalg.norm(
            ts_smooth[120, shoulder_l_idx, :] - ts_smooth[120, shoulder_r_idx, :]
        )
        if shoulder_width < 0.01:
            shoulder_width = 1.0
    else:
        shoulder_width = 1.0

    # ---- FEATURE EXTRACTION AT EACH ANALYSIS FRAME ----

    key_joints = {
        'rw': wrist_r_idx, 'lw': wrist_l_idx,
        're': elbow_r_idx, 'le': elbow_l_idx,
        'rs': shoulder_r_idx, 'ls': shoulder_l_idx,
        'rh': hip_r_idx, 'lh': hip_l_idx,
        'rk': knee_r_idx, 'lk': knee_l_idx,
        'ra': ankle_r_idx, 'la': ankle_l_idx,
        'nose': nose_idx, 'neck': neck_idx,
        'mid_hip': mid_hip_idx,
    }

    for frame in ANALYSIS_FRAMES:
        f = int(np.clip(frame, 0, 239))

        # 1. Hoop-relative positions for all key joints
        for jname, jidx in key_joints.items():
            if jidx is not None:
                pos = ts_hr[f, jidx, :]
                feats.extend([float(pos[0]), float(pos[1]), float(pos[2])])
                feat_names.extend([
                    f"hr_{jname}_x_f{frame}",
                    f"hr_{jname}_y_f{frame}",
                    f"hr_{jname}_z_f{frame}"
                ])
            else:
                feats.extend([0.0, 0.0, 0.0])
                feat_names.extend([f"hr_{jname}_x_f{frame}", f"hr_{jname}_y_f{frame}", f"hr_{jname}_z_f{frame}"])

        # 2. Body-normalized positions for key joints
        body_ref = mid_shoulder[f]
        for jname, jidx in [('rw', wrist_r_idx), ('lw', wrist_l_idx),
                              ('re', elbow_r_idx), ('le', elbow_l_idx),
                              ('rs', shoulder_r_idx), ('ls', shoulder_l_idx)]:
            if jidx is not None:
                pos = (ts_smooth[f, jidx, :] - body_ref) / shoulder_width
                feats.extend([float(pos[0]), float(pos[1]), float(pos[2])])
            else:
                feats.extend([0.0, 0.0, 0.0])
            feat_names.extend([f"bn_{jname}_x_f{frame}", f"bn_{jname}_y_f{frame}", f"bn_{jname}_z_f{frame}"])

        # 3. Velocities at key joints (in hoop-relative frame)
        for jname, jidx in [('rw', wrist_r_idx), ('re', elbow_r_idx),
                              ('rs', shoulder_r_idx), ('rh', hip_r_idx)]:
            if jidx is not None:
                v = vel[f, jidx, :]
                feats.extend([float(v[0]), float(v[1]), float(v[2])])
                speed = float(np.linalg.norm(v))
                feats.append(speed)
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])
            feat_names.extend([f"vel_{jname}_x_f{frame}", f"vel_{jname}_y_f{frame}",
                                f"vel_{jname}_z_f{frame}", f"speed_{jname}_f{frame}"])

        # 4. Accelerations at wrist (key for release)
        if wrist_r_idx is not None:
            a = acc[f, wrist_r_idx, :]
            feats.extend([float(a[0]), float(a[1]), float(a[2])])
        else:
            feats.extend([0.0, 0.0, 0.0])
        feat_names.extend([f"acc_rw_x_f{frame}", f"acc_rw_y_f{frame}", f"acc_rw_z_f{frame}"])

        # 5. Joint angles at this frame
        # Elbow angle (shoulder-elbow-wrist)
        if shoulder_r_idx is not None and elbow_r_idx is not None and wrist_r_idx is not None:
            ang = angle_3pt(ts_smooth[f, shoulder_r_idx, :],
                            ts_smooth[f, elbow_r_idx, :],
                            ts_smooth[f, wrist_r_idx, :])
            feats.append(ang)
        else:
            feats.append(0.0)
        feat_names.append(f"elbow_angle_f{frame}")

        # Shoulder angle (elbow-shoulder-hip)
        if elbow_r_idx is not None and shoulder_r_idx is not None and hip_r_idx is not None:
            ang = angle_3pt(ts_smooth[f, elbow_r_idx, :],
                            ts_smooth[f, shoulder_r_idx, :],
                            ts_smooth[f, hip_r_idx, :])
            feats.append(ang)
        else:
            feats.append(0.0)
        feat_names.append(f"shoulder_angle_f{frame}")

        # Knee angle (hip-knee-ankle) right
        if hip_r_idx is not None and knee_r_idx is not None and ankle_r_idx is not None:
            ang = angle_3pt(ts_smooth[f, hip_r_idx, :],
                            ts_smooth[f, knee_r_idx, :],
                            ts_smooth[f, ankle_r_idx, :])
            feats.append(ang)
        else:
            feats.append(0.0)
        feat_names.append(f"knee_angle_r_f{frame}")

        # Hip-shoulder trunk lean (angle of trunk from vertical)
        if shoulder_r_idx is not None and hip_r_idx is not None:
            trunk = ts_smooth[f, shoulder_r_idx, :] - ts_smooth[f, hip_r_idx, :]
            tz = float(trunk[2])
            txy = float(np.linalg.norm(trunk[:2]))
            lean = float(np.degrees(np.arctan2(txy, max(tz, 1e-6))))
            feats.append(lean)
        else:
            feats.append(0.0)
        feat_names.append(f"trunk_lean_f{frame}")

        # Wrist height above shoulder
        if wrist_r_idx is not None and shoulder_r_idx is not None:
            dz = float(ts_smooth[f, wrist_r_idx, 2] - ts_smooth[f, shoulder_r_idx, 2])
            feats.append(dz)
        else:
            feats.append(0.0)
        feat_names.append(f"wrist_above_shoulder_f{frame}")

        # Arm extension: distance from shoulder to wrist vs max possible
        if shoulder_r_idx is not None and elbow_r_idx is not None and wrist_r_idx is not None:
            upper_arm = np.linalg.norm(ts_smooth[f, shoulder_r_idx, :] - ts_smooth[f, elbow_r_idx, :])
            forearm = np.linalg.norm(ts_smooth[f, elbow_r_idx, :] - ts_smooth[f, wrist_r_idx, :])
            reach = np.linalg.norm(ts_smooth[f, shoulder_r_idx, :] - ts_smooth[f, wrist_r_idx, :])
            max_reach = upper_arm + forearm
            ext = float(reach / max(max_reach, 1e-6))
            feats.append(ext)
            feats.append(float(reach))
        else:
            feats.extend([0.0, 0.0])
        feat_names.extend([f"arm_extension_f{frame}", f"arm_reach_f{frame}"])

        # Wrist-to-hoop vector
        if wrist_r_idx is not None:
            wrist_pos = ts_smooth[f, wrist_r_idx, :].copy()
            to_hoop = HOOP_POS - wrist_pos
            dist_to_hoop = float(np.linalg.norm(to_hoop))
            to_hoop_unit = to_hoop / max(dist_to_hoop, 1e-6)
            # Angle of wrist velocity with hoop direction
            if wrist_r_idx is not None:
                v = vel[f, wrist_r_idx, :]
                vn = np.linalg.norm(v)
                if vn > 1e-6:
                    aim_angle = float(np.degrees(np.arccos(np.clip(
                        np.dot(v / vn, to_hoop_unit), -1, 1))))
                else:
                    aim_angle = 0.0
            else:
                aim_angle = 0.0
            feats.extend([dist_to_hoop, aim_angle])
        else:
            feats.extend([0.0, 0.0])
        feat_names.extend([f"dist_wrist_hoop_f{frame}", f"aim_angle_f{frame}"])

        # Two-hand separation (guide hand vs shooting hand)
        if wrist_r_idx is not None and wrist_l_idx is not None:
            sep = ts_smooth[f, wrist_l_idx, :] - ts_smooth[f, wrist_r_idx, :]
            feats.extend([float(sep[0]), float(sep[1]), float(sep[2])])
            feats.append(float(np.linalg.norm(sep)))
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        feat_names.extend([f"handx_sep_f{frame}", f"handy_sep_f{frame}",
                            f"handz_sep_f{frame}", f"hand_dist_f{frame}"])

        # Hip rotation: vector from left-hip to right-hip, projected to XY
        if hip_l_idx is not None and hip_r_idx is not None:
            hip_vec = ts_smooth[f, hip_r_idx, :] - ts_smooth[f, hip_l_idx, :]
            hip_angle = float(np.degrees(np.arctan2(hip_vec[1], max(abs(hip_vec[0]), 1e-6))))
            feats.append(hip_angle)
            # Hip-to-shoulder rotation difference
            if shoulder_l_idx is not None and shoulder_r_idx is not None:
                sh_vec = ts_smooth[f, shoulder_r_idx, :] - ts_smooth[f, shoulder_l_idx, :]
                sh_angle = float(np.degrees(np.arctan2(sh_vec[1], max(abs(sh_vec[0]), 1e-6))))
                feats.append(sh_angle - hip_angle)  # shoulder rotation relative to hips
            else:
                feats.append(0.0)
        else:
            feats.extend([0.0, 0.0])
        feat_names.extend([f"hip_rotation_f{frame}", f"shoulder_hip_rot_diff_f{frame}"])

    # ---- TEMPORAL FEATURES (cross-frame) ----

    # Wrist trajectory statistics from frame 130 to 160
    if wrist_r_idx is not None:
        rw_traj = ts_hr[120:180, wrist_r_idx, :]  # hoop-relative
        # Mean position
        feats.extend(rw_traj.mean(axis=0).tolist())
        feat_names.extend(["rw_mean_x_120_180", "rw_mean_y_120_180", "rw_mean_z_120_180"])
        # Max height
        feats.append(float(rw_traj[:, 2].max()))
        feat_names.append("rw_max_z_120_180")
        # Frame of max height
        feats.append(float(np.argmax(rw_traj[:, 2])))
        feat_names.append("rw_argmax_z_120_180")
        # Variance of lateral position
        feats.append(float(rw_traj[:, 1].var()))
        feat_names.append("rw_var_y_120_180")

        # Change in position 120->153 vs 153->180
        delta_pre = ts_hr[153, wrist_r_idx, :] - ts_hr[120, wrist_r_idx, :]
        delta_post = ts_hr[180, wrist_r_idx, :] - ts_hr[153, wrist_r_idx, :]
        feats.extend(delta_pre.tolist())
        feats.extend(delta_post.tolist())
        feat_names.extend(["rw_dpre_x", "rw_dpre_y", "rw_dpre_z",
                            "rw_dpost_x", "rw_dpost_y", "rw_dpost_z"])

    # Elbow trajectory
    if elbow_r_idx is not None:
        re_traj = ts_hr[120:180, elbow_r_idx, :]
        feats.append(float(re_traj[:, 2].max()))
        feat_names.append("re_max_z_120_180")

    # How much does the shooting arm velocity direction change (arc consistency)
    if wrist_r_idx is not None:
        vel_frames = [140, 145, 150, 153, 155, 160]
        vels_fw = [vel[f, wrist_r_idx, 0] for f in vel_frames]  # forward component
        vels_lat = [vel[f, wrist_r_idx, 1] for f in vel_frames]  # lateral component
        vels_up = [vel[f, wrist_r_idx, 2] for f in vel_frames]   # vertical
        feats.extend([np.mean(vels_fw), np.std(vels_fw),
                      np.mean(vels_lat), np.std(vels_lat),
                      np.mean(vels_up), np.std(vels_up)])
        feat_names.extend(["rw_vel_fw_mean", "rw_vel_fw_std",
                            "rw_vel_lat_mean", "rw_vel_lat_std",
                            "rw_vel_up_mean", "rw_vel_up_std"])
        # Peak speed around release
        speeds = [np.linalg.norm(vel[f, wrist_r_idx, :]) for f in range(130, 170)]
        feats.append(float(max(speeds)))
        feat_names.append("rw_peak_speed")
        feats.append(float(np.argmax(speeds) + 130))
        feat_names.append("rw_peak_speed_frame")

    # Knee bend change (leg drive)
    if hip_r_idx is not None and knee_r_idx is not None and ankle_r_idx is not None:
        ang_120 = angle_3pt(ts_smooth[120, hip_r_idx, :],
                             ts_smooth[120, knee_r_idx, :],
                             ts_smooth[120, ankle_r_idx, :])
        ang_153 = angle_3pt(ts_smooth[153, hip_r_idx, :],
                             ts_smooth[153, knee_r_idx, :],
                             ts_smooth[153, ankle_r_idx, :])
        feats.append(ang_153 - ang_120)  # positive = extending (leg drive)
        feat_names.append("knee_extension_120_153")

    # Elbow angle change from 120 to 153 (arm unfolding)
    if shoulder_r_idx is not None and elbow_r_idx is not None and wrist_r_idx is not None:
        ang_120 = angle_3pt(ts_smooth[120, shoulder_r_idx, :],
                             ts_smooth[120, elbow_r_idx, :],
                             ts_smooth[120, wrist_r_idx, :])
        ang_153 = angle_3pt(ts_smooth[153, shoulder_r_idx, :],
                             ts_smooth[153, elbow_r_idx, :],
                             ts_smooth[153, wrist_r_idx, :])
        feats.append(ang_153 - ang_120)
        feat_names.append("elbow_extension_120_153")

    # Center of mass height trajectory
    hip_height_120 = float(ts_smooth[120, mid_hip_idx, 2]) if mid_hip_idx is not None else 0
    hip_height_153 = float(ts_smooth[153, mid_hip_idx, 2]) if mid_hip_idx is not None else 0
    feats.append(hip_height_153 - hip_height_120)
    feat_names.append("hip_rise_120_153")

    # Body facing direction vs hoop
    if hip_l_idx is not None and hip_r_idx is not None:
        hip_vec = ts_smooth[153, hip_r_idx, :2] - ts_smooth[153, hip_l_idx, :2]
        body_facing = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))
        to_hoop_dir = np.degrees(np.arctan2(
            HOOP_POS[1] - ts_smooth[153, mid_hip_idx, 1],
            HOOP_POS[0] - ts_smooth[153, mid_hip_idx, 0]
        ))
        feats.append(float(body_facing - to_hoop_dir))
        feat_names.append("body_hoop_alignment")
    else:
        feats.append(0.0)
        feat_names.append("body_hoop_alignment")

    return np.array(feats, dtype=np.float32), feat_names


# ============================================================
# BUILD FEATURE MATRIX FOR ALL SHOTS
# ============================================================

def build_feature_matrix(data):
    """Build comprehensive feature matrix for all shots."""
    n = data['X_3d'].shape[0]
    kp_index = data['kp_index']
    print(f"  Extracting features for {n} shots...")

    all_feats = []
    feat_names = None
    for i in range(n):
        ts = data['X_3d'][i]  # 240 x n_kp x 3
        f, names = extract_all_features(ts, kp_index)
        all_feats.append(f)
        if feat_names is None:
            feat_names = names
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n}")

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feat_names


# ============================================================
# PER-PLAYER CORRELATION ANALYSIS
# ============================================================

def per_player_correlation_analysis(X, y, pids, feat_names):
    """
    For each player x target: compute Pearson |r| for all features.
    Returns dict: player -> target -> sorted (feat_name, r) list.
    """
    unique_pids = sorted(set(pids))
    results = {}

    for pid in unique_pids:
        mask = pids == pid
        X_p = X[mask]
        y_p = y[mask]
        n_p = X_p.shape[0]
        results[pid] = {}

        for ti, tname in enumerate(TARGETS):
            y_t = y_p[:, ti]
            # Pearson r for each feature
            corrs = []
            for fi in range(X_p.shape[1]):
                f_vals = X_p[:, fi]
                # Only compute if not constant
                if np.std(f_vals) < 1e-8 or np.std(y_t) < 1e-8:
                    corrs.append(0.0)
                else:
                    r = float(np.corrcoef(f_vals, y_t)[0, 1])
                    corrs.append(0.0 if np.isnan(r) else r)
            corrs = np.array(corrs)
            # Rank by |r|
            order = np.argsort(-np.abs(corrs))
            results[pid][tname] = [(feat_names[i], float(corrs[i])) for i in order]

    return results


def print_top_features(corr_results, n_top=15):
    """Print top features per player per target."""
    unique_pids = sorted(corr_results.keys())

    for tname in TARGETS:
        print(f"\n{'='*70}")
        print(f"TARGET: {tname.upper()}")
        print(f"{'='*70}")

        # Collect top features per player
        player_tops = {}
        for pid in unique_pids:
            tops = corr_results[pid][tname][:n_top]
            player_tops[pid] = tops
            print(f"\nPlayer {pid} - Top {n_top} features:")
            for fname, r in tops[:n_top]:
                print(f"  r={r:+.3f}  {fname}")

        # Find player-specific features (high for one player, low for others)
        print(f"\n--- Player-Specific Features for {tname} ---")
        all_top_names = set()
        for pid in unique_pids:
            all_top_names.update([f for f, r in player_tops[pid][:20]])

        for fname in sorted(all_top_names):
            # Get rank of this feature for each player
            ranks = {}
            for pid in unique_pids:
                feat_list = [f for f, r in corr_results[pid][tname]]
                if fname in feat_list:
                    ranks[pid] = feat_list.index(fname)
                else:
                    ranks[pid] = 9999
            max_rank = max(ranks.values())
            min_rank = min(ranks.values())
            rank_spread = max_rank - min_rank
            if rank_spread > 100:  # feature ranks very differently across players
                r_vals = {pid: corr_results[pid][tname][ranks[pid]][1]
                          if ranks[pid] < len(corr_results[pid][tname]) else 0.0
                          for pid in unique_pids}
                print(f"  {fname}")
                for pid in unique_pids:
                    print(f"    P{pid}: rank={ranks[pid]:3d}, r={r_vals[pid]:+.3f}")


# ============================================================
# PER-PLAYER FEATURE SELECTION MODEL
# ============================================================

def per_player_feature_selection_loo(X, y, pids, feat_names, top_k=25,
                                      bw_quantile=0.3, alpha=10.0):
    """
    Honest LOO with per-player feature selection.
    For each held-out shot:
    - Use remaining shots of same player to select top-k features (by |r| with target)
    - Train kernel Ridge on selected features
    - Predict held-out shot

    Returns: oof predictions, per-target LOO MSE
    """
    n = len(y)
    oof_preds = np.zeros((n, 3), dtype=np.float32)
    unique_pids = sorted(set(pids))

    for ti, tname in enumerate(TARGETS):
        y_t = y[:, ti]

        for pid in unique_pids:
            pidx = np.where(pids == pid)[0]
            n_p = len(pidx)

            for i_local, i_global in enumerate(pidx):
                train_local = [j for j in range(n_p) if j != i_local]
                train_global = pidx[train_local]

                # Select features using training set only (no leakage)
                X_tr_raw = X[train_global]
                X_te_raw = X[i_global:i_global+1]
                y_tr = y_t[train_global]

                # Feature selection: top-k by |r| with target
                corrs = []
                for fi in range(X_tr_raw.shape[1]):
                    fv = X_tr_raw[:, fi]
                    if np.std(fv) < 1e-8:
                        corrs.append(0.0)
                    else:
                        r = float(np.corrcoef(fv, y_tr)[0, 1])
                        corrs.append(0.0 if np.isnan(r) else abs(r))
                top_feat_idx = np.argsort(corrs)[-top_k:]

                X_tr_sel = X_tr_raw[:, top_feat_idx]
                X_te_sel = X_te_raw[:, top_feat_idx]

                # Scale
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr_sel)
                X_te_s = sc.transform(X_te_sel)

                # Per-player kernel Ridge (Gaussian kernel)
                dists = cdist(X_tr_s, X_tr_s, metric='euclidean')
                bw = np.quantile(dists[dists > 0], bw_quantile)
                if bw < 1e-6:
                    bw = 1.0
                K_tr = np.exp(-0.5 * (dists / bw) ** 2)

                d_te = cdist(X_te_s, X_tr_s, metric='euclidean')
                K_te = np.exp(-0.5 * (d_te / bw) ** 2)

                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(K_tr, y_tr)
                pred = float(ridge.predict(K_te)[0])
                oof_preds[i_global, ti] = pred

    mse_per_target = {}
    for ti, tname in enumerate(TARGETS):
        mse_per_target[tname] = float(np.mean((oof_preds[:, ti] - y[:, ti]) ** 2))

    return oof_preds, mse_per_target


def per_player_feature_selection_test_preds(X_train, y_train, X_test, pids_train, pids_test,
                                             feat_names, top_k=25, bw_quantile=0.3, alpha=10.0):
    """
    Generate test predictions using per-player feature selection.
    For each test shot, use same player's training shots to select features.
    """
    n_test = len(X_test)
    test_preds = np.zeros((n_test, 3), dtype=np.float32)
    unique_pids = sorted(set(pids_train))

    for ti, tname in enumerate(TARGETS):
        y_t = y_train[:, ti]

        for pid in unique_pids:
            train_idx = np.where(pids_train == pid)[0]
            test_idx = np.where(pids_test == pid)[0]
            if len(test_idx) == 0:
                continue

            X_tr_raw = X_train[train_idx]
            X_te_raw = X_test[test_idx]
            y_tr = y_t[train_idx]

            # Select features from full player training set
            corrs = []
            for fi in range(X_tr_raw.shape[1]):
                fv = X_tr_raw[:, fi]
                if np.std(fv) < 1e-8:
                    corrs.append(0.0)
                else:
                    r = float(np.corrcoef(fv, y_tr)[0, 1])
                    corrs.append(0.0 if np.isnan(r) else abs(r))
            top_feat_idx = np.argsort(corrs)[-top_k:]

            X_tr_sel = X_tr_raw[:, top_feat_idx]
            X_te_sel = X_te_raw[:, top_feat_idx]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_sel)
            X_te_s = sc.transform(X_te_sel)

            dists = cdist(X_tr_s, X_tr_s, metric='euclidean')
            bw = np.quantile(dists[dists > 0], bw_quantile)
            if bw < 1e-6:
                bw = 1.0
            K_tr = np.exp(-0.5 * (dists / bw) ** 2)

            d_te = cdist(X_te_s, X_tr_s, metric='euclidean')
            K_te = np.exp(-0.5 * (d_te / bw) ** 2)

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(K_tr, y_tr)
            preds = ridge.predict(K_te).flatten()
            test_preds[test_idx, ti] = preds

    return test_preds


# ============================================================
# GLOBAL BASELINE (no feature selection) FOR COMPARISON
# ============================================================

def global_baseline_loo(X, y, pids, n_pls=15, bw_quantile=0.3, alpha=10.0):
    """Honest LOO with global PLS (refit per fold) - same as core pipeline."""
    n = len(y)
    oof_preds = np.zeros((n, 3), dtype=np.float32)
    unique_pids = sorted(set(pids))

    for ti, tname in enumerate(TARGETS):
        y_t = y[:, ti]
        n_pls_use = min(n_pls, X.shape[1] - 1)

        for pid in unique_pids:
            pidx = np.where(pids == pid)[0]
            n_p = len(pidx)

            for i_local, i_global in enumerate(pidx):
                train_local = [j for j in range(n_p) if j != i_local]
                train_global = pidx[train_local]

                X_tr_raw = X[train_global]
                X_te_raw = X[i_global:i_global+1]
                y_tr = y_t[train_global]

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr_raw)
                X_te_s = sc.transform(X_te_raw)

                pls = PLSRegression(n_components=n_pls_use, max_iter=500)
                pls.fit(X_tr_s, y_tr)
                X_tr_pls = pls.transform(X_tr_s)
                X_te_pls = pls.transform(X_te_s)

                sc2 = StandardScaler()
                X_tr_pls2 = sc2.fit_transform(X_tr_pls)
                X_te_pls2 = sc2.transform(X_te_pls)

                dists = cdist(X_tr_pls2, X_tr_pls2, metric='euclidean')
                bw = np.quantile(dists[dists > 0], bw_quantile)
                if bw < 1e-6:
                    bw = 1.0
                K_tr = np.exp(-0.5 * (dists / bw) ** 2)
                d_te = cdist(X_te_pls2, X_tr_pls2, metric='euclidean')
                K_te = np.exp(-0.5 * (d_te / bw) ** 2)

                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(K_tr, y_tr)
                oof_preds[i_global, ti] = float(ridge.predict(K_te)[0])

    mse_per_target = {t: float(np.mean((oof_preds[:, ti] - y[:, ti]) ** 2))
                      for ti, t in enumerate(TARGETS)}
    return oof_preds, mse_per_target


# ============================================================
# FIND WHICH FEATURES ARE PLAYER-SPECIFIC
# ============================================================

def analyze_feature_specificity(corr_results, feat_names, top_n=30):
    """
    Identify features that are high-value for one player but low for others.
    These are the "player-specific information channels."
    """
    unique_pids = sorted(corr_results.keys())
    print("\n" + "="*70)
    print("PLAYER-SPECIFIC INFORMATION CHANNELS")
    print("="*70)
    print("(Features where one player's r is much higher than others)")

    for tname in TARGETS:
        print(f"\n--- {tname.upper()} ---")

        # Build matrix: player x feature -> |r|
        n_feats = len(feat_names)
        r_matrix = np.zeros((len(unique_pids), n_feats))

        for pi, pid in enumerate(unique_pids):
            corrs_dict = dict(corr_results[pid][tname])
            for fi, fname in enumerate(feat_names):
                r_matrix[pi, fi] = abs(corrs_dict.get(fname, 0.0))

        # Find features where max player r >> mean of others
        specificity_scores = []
        for fi in range(n_feats):
            r_vals = r_matrix[:, fi]
            max_r = r_vals.max()
            max_pid_idx = r_vals.argmax()
            others = np.delete(r_vals, max_pid_idx)
            other_mean = others.mean() if len(others) > 0 else 0.0
            specificity = max_r - other_mean
            specificity_scores.append((specificity, max_r, max_pid_idx, fi))

        specificity_scores.sort(reverse=True)
        print(f"{'Feature':<45} {'BestPlayer':>10} {'BestR':>7} {'OtherMean':>10} {'Specificity':>12}")
        for spec, max_r, max_pid_i, fi in specificity_scores[:top_n]:
            pid = unique_pids[max_pid_i]
            others_r = np.delete(r_matrix[:, fi], max_pid_i)
            other_mean = others_r.mean()
            if max_r > 0.15:  # Only show if the top r is at least somewhat meaningful
                print(f"  {feat_names[fi]:<43} P{pid:>8} {max_r:>7.3f} {other_mean:>10.3f} {spec:>12.3f}")


# ============================================================
# COMPARE TOP-K SWEEP
# ============================================================

def topk_sweep(X, y, pids, feat_names, k_values=[10, 15, 20, 25, 30, 40, 50]):
    """Find best k for per-player feature selection."""
    print("\n--- Top-K Sweep ---")
    results = {}
    for k in k_values:
        t0 = time.time()
        _, mse = per_player_feature_selection_loo(X, y, pids, feat_names, top_k=k)
        mean_mse = np.mean(list(mse.values()))
        results[k] = {'mse': mse, 'mean': mean_mse, 'time': time.time() - t0}
        print(f"  k={k:3d}: angle={mse['angle']:.6f} depth={mse['depth']:.6f} "
              f"LR={mse['left_right']:.6f} mean={mean_mse:.6f} ({time.time()-t0:.1f}s)")
    return results


# ============================================================
# SUBMISSIONS
# ============================================================

def save_submission(test_ids, test_preds, sub_num, desc):
    """Save submission file."""
    sub_df = pd.DataFrame({
        'id': test_ids,
        'angle': np.clip(test_preds[:, 0], 0, 1),
        'depth': np.clip(test_preds[:, 1], 0, 1),
        'left_right': np.clip(test_preds[:, 2], 0, 1),
    })
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(path, index=False)
    print(f"  Sub {sub_num}: {desc}")
    print(f"    Saved to {path}")
    return path


def blend_with_best(test_preds_new, weight_new, sub_num_best=3507):
    """Blend new predictions with current best submission."""
    best_path = SUBMISSION_DIR / f"submission_{sub_num_best}.csv"
    best_df = pd.read_csv(best_path)
    cols = ['angle', 'depth', 'left_right']
    best_vals = best_df[cols].values
    blended = weight_new * test_preds_new + (1 - weight_new) * best_vals
    return blended


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    print("=== Per-Player Feature Importance Analysis ===")
    print(f"Honest LOO baseline: ~0.006830")

    # Load data
    train, test = load_data()
    X_3d_train = train['X_3d']
    y_train = train['y']
    pids_train = train['pids']
    X_3d_test = test['X_3d']
    pids_test = test['pids']

    # Build feature matrices
    print("\nBuilding training features...")
    X_train, feat_names = build_feature_matrix(train)
    print(f"  Feature matrix: {X_train.shape} ({len(feat_names)} features)")

    print("Building test features...")
    X_test, _ = build_feature_matrix(test)
    print(f"  Test feature matrix: {X_test.shape}")

    # ---- STEP 1: Per-player correlation analysis ----
    print("\n" + "="*70)
    print("STEP 1: Per-Player Correlation Analysis")
    print("="*70)
    corr_results = per_player_correlation_analysis(X_train, y_train, pids_train, feat_names)
    print_top_features(corr_results, n_top=15)

    # ---- STEP 2: Feature specificity analysis ----
    analyze_feature_specificity(corr_results, feat_names, top_n=25)

    # ---- STEP 3: Global baseline on these features ----
    print("\n" + "="*70)
    print("STEP 3: Global Baseline (PLS on all features)")
    print("="*70)
    t0 = time.time()
    oof_global, mse_global = global_baseline_loo(X_train, y_train, pids_train, n_pls=15)
    mean_global = np.mean(list(mse_global.values()))
    print(f"  angle={mse_global['angle']:.6f} depth={mse_global['depth']:.6f} "
          f"LR={mse_global['left_right']:.6f} mean={mean_global:.6f} ({time.time()-t0:.1f}s)")

    # ---- STEP 4: Per-player feature selection LOO sweep ----
    print("\n" + "="*70)
    print("STEP 4: Per-Player Feature Selection (Top-K Sweep)")
    print("="*70)
    k_results = topk_sweep(X_train, y_train, pids_train, feat_names,
                            k_values=[10, 15, 20, 25, 30, 40, 50])

    # Find best k
    best_k = min(k_results, key=lambda k: k_results[k]['mean'])
    best_mean = k_results[best_k]['mean']
    print(f"\n  Best k={best_k}: mean_LOO={best_mean:.6f}")
    print(f"  vs global baseline: {mean_global:.6f}")
    print(f"  vs honest LOO (0.006830)")
    improvement = (0.006830 - best_mean) / 0.006830 * 100
    print(f"  Improvement over baseline: {improvement:+.1f}%")

    # ---- STEP 5: Per-player analysis by player ----
    print("\n" + "="*70)
    print("STEP 5: Per-Player LOO Breakdown")
    print("="*70)
    unique_pids = sorted(set(pids_train))
    oof_best, _ = per_player_feature_selection_loo(
        X_train, y_train, pids_train, feat_names, top_k=best_k)
    oof_global_arr = oof_global

    for pid in unique_pids:
        mask = pids_train == pid
        for ti, tname in enumerate(TARGETS):
            mse_ps = float(np.mean((oof_best[mask, ti] - y_train[mask, ti]) ** 2))
            mse_gl = float(np.mean((oof_global_arr[mask, ti] - y_train[mask, ti]) ** 2))
            delta = (mse_ps - mse_gl) / mse_gl * 100
            print(f"  P{pid} {tname}: per-player-sel={mse_ps:.5f} global={mse_gl:.5f} "
                  f"delta={delta:+.1f}%")

    # ---- STEP 6: Generate test predictions and submissions ----
    print("\n" + "="*70)
    print("STEP 6: Generating Submissions")
    print("="*70)

    # Standalone per-player feature selection
    test_preds_ps = per_player_feature_selection_test_preds(
        X_train, y_train, X_test, pids_train, pids_test,
        feat_names, top_k=best_k)

    # Blends with current best (Sub 3507, LB 0.006205)
    BEST_SUB = 3507

    sub1 = get_next_submission_number()
    save_submission(test['ids'], test_preds_ps, sub1,
                    f"Standalone per-player feature selection k={best_k} (CV={best_mean:.6f})")

    blend_3pct = blend_with_best(test_preds_ps, 0.03, BEST_SUB)
    sub2 = get_next_submission_number()
    save_submission(test['ids'], blend_3pct, sub2,
                    f"3% per-player-sel + 97% Sub{BEST_SUB}")

    blend_5pct = blend_with_best(test_preds_ps, 0.05, BEST_SUB)
    sub3 = get_next_submission_number()
    save_submission(test['ids'], blend_5pct, sub3,
                    f"5% per-player-sel + 95% Sub{BEST_SUB}")

    blend_10pct = blend_with_best(test_preds_ps, 0.10, BEST_SUB)
    sub4 = get_next_submission_number()
    save_submission(test['ids'], blend_10pct, sub4,
                    f"10% per-player-sel + 90% Sub{BEST_SUB}")

    # Save full results
    total_time = time.time() - t_start
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {
        'global_baseline_mse': mse_global,
        'global_baseline_mean': mean_global,
        'best_k': best_k,
        'best_k_mse': k_results[best_k]['mse'],
        'best_k_mean': best_mean,
        'k_sweep': {str(k): {'mean': v['mean'], 'mse': v['mse']} for k, v in k_results.items()},
        'improvement_pct': improvement,
        'honest_loo_baseline': 0.006830,
        'total_time_s': total_time,
        'submissions': [
            {'num': sub1, 'desc': f'Standalone k={best_k}'},
            {'num': sub2, 'desc': f'3% blend + Sub{BEST_SUB}'},
            {'num': sub3, 'desc': f'5% blend + Sub{BEST_SUB}'},
            {'num': sub4, 'desc': f'10% blend + Sub{BEST_SUB}'},
        ]
    }
    out_path = OUTPUT_DIR / f"per_player_feature_importance_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\nTotal time: {total_time:.1f}s")
    print("\n=== Summary ===")
    print(f"  Global baseline (this feature set): {mean_global:.6f}")
    print(f"  Per-player feature selection:       {best_mean:.6f}")
    print(f"  Honest LOO reference:               0.006830")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"\n  Submissions created: {sub1}, {sub2}, {sub3}, {sub4}")
