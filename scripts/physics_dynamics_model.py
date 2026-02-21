"""
Physics Dynamics Model - Inverse Dynamics + Kalman-Smoothed Kinematics.

This derives quantities ONLY a physics simulation can produce:
1. Kalman/RTS smoothed joint velocities and accelerations (optimal filtering)
2. Joint torques from inverse dynamics (requires mass distribution)
3. Kinetic energy and momentum at release (requires mass)
4. Power flow through the kinematic chain (requires mass + dynamics)
5. Forward-simulated targets from Kalman-smoothed release velocity

These features cannot be learned by ML from raw positions because they require
knowledge of the mass distribution and a dynamics model.
"""
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
HOOP_POS = np.array([5.25, -25.0, 10.0])
FEET_TO_METERS = 0.3048
DT = 1.0 / 60.0
G = 9.81  # m/s^2

import lightgbm as lgb
import xgboost as xgb


# ============================================================
# ANTHROPOMETRIC MODEL
# ============================================================

# Dempster/Winter segment mass ratios (fraction of total body mass)
SEGMENT_MASS_RATIO = {
    'upper_arm': 0.028,
    'forearm': 0.016,
    'hand': 0.006,
}

# Segment COM position (fraction from proximal joint)
SEGMENT_COM_RATIO = {
    'upper_arm': 0.436,
    'forearm': 0.430,
    'hand': 0.506,
}

# Segment radius of gyration (fraction of segment length, about COM)
SEGMENT_GYRATION_RATIO = {
    'upper_arm': 0.322,
    'forearm': 0.303,
    'hand': 0.297,
}

BALL_MASS = 0.624  # kg (standard basketball)
BALL_RADIUS = 0.121  # m


def estimate_body_mass(bone_lengths_ft):
    """Estimate body mass from bone lengths using anthropometric ratios."""
    # Use total height estimate from bone chain
    # Rough: height ~ 2.1 * (upper_arm + forearm) for arm proportion
    arm_length = (bone_lengths_ft.get('upper_arm', 0.95) +
                  bone_lengths_ft.get('forearm', 0.80)) * FEET_TO_METERS
    # Arm span ~ height, arm = 0.44 * height (one side)
    est_height = arm_length / 0.44
    # BMI ~23 for athletes: mass = 23 * height^2
    est_mass = 23.0 * est_height ** 2
    return np.clip(est_mass, 60.0, 120.0)


# ============================================================
# RTS SMOOTHER (Rauch-Tung-Striebel)
# ============================================================

class RTSSmoother:
    """
    Rauch-Tung-Striebel smoother for optimal position/velocity/acceleration estimation.

    State: [x, vx, ax] (per axis, 3D done separately)
    Dynamics: constant acceleration + jerk noise
    Measurement: position + noise
    """

    def __init__(self, process_noise_std=2.0, measurement_noise_std=0.015):
        """
        Args:
            process_noise_std: jerk noise std in m/s^3 (how much acceleration can change per step)
            measurement_noise_std: position noise std in meters
        """
        self.dt = DT
        # State transition
        dt = self.dt
        self.F = np.array([
            [1, dt, 0.5*dt**2],
            [0, 1,  dt],
            [0, 0,  1]
        ])
        # Measurement
        self.H = np.array([[1, 0, 0]])

        # Process noise (jerk-driven)
        q = process_noise_std ** 2
        dt2 = dt**2; dt3 = dt**3; dt4 = dt**4; dt5 = dt**5
        self.Q = q * np.array([
            [dt5/20, dt4/8, dt3/6],
            [dt4/8,  dt3/3, dt2/2],
            [dt3/6,  dt2/2, dt]
        ])

        # Measurement noise
        self.R = np.array([[measurement_noise_std ** 2]])

    def smooth(self, measurements):
        """
        Run forward Kalman filter + backward RTS smoother.

        Args:
            measurements: (T,) array of position measurements in meters

        Returns:
            x_smooth: (T, 3) array of [position, velocity, acceleration] at each time
        """
        T = len(measurements)
        n = 3  # state dimension

        # Forward pass (Kalman filter)
        x_filt = np.zeros((T, n))
        P_filt = np.zeros((T, n, n))
        x_pred = np.zeros((T, n))
        P_pred = np.zeros((T, n, n))

        # Initialize
        x_filt[0] = [measurements[0], 0, 0]
        P_filt[0] = np.diag([self.R[0,0], 1.0, 10.0])  # uncertain velocity/accel

        for t in range(1, T):
            # Predict
            x_pred[t] = self.F @ x_filt[t-1]
            P_pred[t] = self.F @ P_filt[t-1] @ self.F.T + self.Q

            # Update
            z = measurements[t]
            if np.isnan(z) or np.isinf(z):
                # Skip this measurement
                x_filt[t] = x_pred[t]
                P_filt[t] = P_pred[t]
                continue

            y = z - self.H @ x_pred[t]
            S = self.H @ P_pred[t] @ self.H.T + self.R
            K = P_pred[t] @ self.H.T @ np.linalg.inv(S)
            x_filt[t] = x_pred[t] + K @ y
            P_filt[t] = (np.eye(n) - K @ self.H) @ P_pred[t]

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((T, n))
        x_smooth[-1] = x_filt[-1]

        for t in range(T-2, -1, -1):
            P_pred_t1 = self.F @ P_filt[t] @ self.F.T + self.Q
            P_pred_inv = np.linalg.inv(P_pred_t1 + 1e-10 * np.eye(n))
            C = P_filt[t] @ self.F.T @ P_pred_inv
            x_smooth[t] = x_filt[t] + C @ (x_smooth[t+1] - self.F @ x_filt[t])

        return x_smooth


# ============================================================
# DATA LOADING
# ============================================================

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


# ============================================================
# INVERSE DYNAMICS
# ============================================================

def compute_inverse_dynamics(shoulder_pos, elbow_pos, wrist_pos, body_mass,
                             smoother):
    """
    Compute joint torques using Newton-Euler inverse dynamics.

    Uses RTS-smoothed kinematics for optimal velocity/acceleration estimation.

    Returns dict of physics features.
    """
    # Convert to meters
    s_ft = shoulder_pos  # (240, 3) in feet
    e_ft = elbow_pos
    w_ft = wrist_pos

    # Bone lengths (median per shot for stability)
    ua_lengths = np.linalg.norm(e_ft - s_ft, axis=1)
    fa_lengths = np.linalg.norm(w_ft - e_ft, axis=1)
    ua_len = np.nanmedian(ua_lengths) * FEET_TO_METERS
    fa_len = np.nanmedian(fa_lengths) * FEET_TO_METERS

    bone_lengths = {
        'upper_arm': ua_len / FEET_TO_METERS,
        'forearm': fa_len / FEET_TO_METERS,
    }
    body_mass_est = estimate_body_mass(bone_lengths)

    # Segment masses
    m_ua = SEGMENT_MASS_RATIO['upper_arm'] * body_mass_est
    m_fa = SEGMENT_MASS_RATIO['forearm'] * body_mass_est
    m_hand = SEGMENT_MASS_RATIO['hand'] * body_mass_est

    # Segment moments of inertia (about COM, simplified as rod)
    I_ua = m_ua * (SEGMENT_GYRATION_RATIO['upper_arm'] * ua_len) ** 2
    I_fa = m_fa * (SEGMENT_GYRATION_RATIO['forearm'] * fa_len) ** 2

    # RTS smooth each joint (in meters)
    s_smooth = np.zeros((240, 3))
    e_smooth = np.zeros((240, 3))
    w_smooth = np.zeros((240, 3))

    s_vel = np.zeros((240, 3))
    e_vel = np.zeros((240, 3))
    w_vel = np.zeros((240, 3))

    s_acc = np.zeros((240, 3))
    e_acc = np.zeros((240, 3))
    w_acc = np.zeros((240, 3))

    for ax in range(3):
        # Shoulder
        result = smoother.smooth(s_ft[:, ax] * FEET_TO_METERS)
        s_smooth[:, ax] = result[:, 0]
        s_vel[:, ax] = result[:, 1]
        s_acc[:, ax] = result[:, 2]

        # Elbow
        result = smoother.smooth(e_ft[:, ax] * FEET_TO_METERS)
        e_smooth[:, ax] = result[:, 0]
        e_vel[:, ax] = result[:, 1]
        e_acc[:, ax] = result[:, 2]

        # Wrist
        result = smoother.smooth(w_ft[:, ax] * FEET_TO_METERS)
        w_smooth[:, ax] = result[:, 0]
        w_vel[:, ax] = result[:, 1]
        w_acc[:, ax] = result[:, 2]

    # Center of mass positions and accelerations
    ua_com = s_smooth + SEGMENT_COM_RATIO['upper_arm'] * (e_smooth - s_smooth)
    fa_com = e_smooth + SEGMENT_COM_RATIO['forearm'] * (w_smooth - e_smooth)
    ua_com_acc = s_acc + SEGMENT_COM_RATIO['upper_arm'] * (e_acc - s_acc)
    fa_com_acc = e_acc + SEGMENT_COM_RATIO['forearm'] * (w_acc - e_acc)

    # Gravity vector
    g_vec = np.array([0, 0, -G])

    # === INVERSE DYNAMICS (Newton-Euler, distal to proximal) ===

    # Wrist joint: forces and torques on hand segment
    # F_wrist = m_hand * (a_wrist - g) [simplified - hand COM ~ wrist]
    F_wrist = m_hand * (w_acc - g_vec)  # (240, 3)

    # Torque at wrist (simplified: no rotation of hand segment)
    tau_wrist = np.zeros((240, 3))

    # Elbow joint: forces and torques on forearm
    # F_elbow = m_fa * a_fa_com + F_wrist - m_fa * g
    F_elbow = m_fa * fa_com_acc + F_wrist - m_fa * g_vec  # (240, 3)

    # Torque at elbow
    r_wrist_from_elbow = w_smooth - e_smooth  # lever arm
    r_com_from_elbow = fa_com - e_smooth
    tau_elbow = (np.cross(r_com_from_elbow, m_fa * (fa_com_acc - g_vec)) +
                 np.cross(r_wrist_from_elbow, F_wrist))  # (240, 3)

    # Shoulder joint: forces and torques on upper arm
    F_shoulder = m_ua * ua_com_acc + F_elbow - m_ua * g_vec  # (240, 3)

    r_elbow_from_shoulder = e_smooth - s_smooth
    r_com_from_shoulder = ua_com - s_smooth
    tau_shoulder = (np.cross(r_com_from_shoulder, m_ua * (ua_com_acc - g_vec)) +
                    np.cross(r_elbow_from_shoulder, F_elbow) +
                    tau_elbow)  # (240, 3)

    # === ENERGY AND MOMENTUM ===

    # Kinetic energy of arm chain
    ke_ua = 0.5 * m_ua * np.sum(
        (s_vel + SEGMENT_COM_RATIO['upper_arm'] * (e_vel - s_vel))**2, axis=1)
    ke_fa = 0.5 * m_fa * np.sum(
        (e_vel + SEGMENT_COM_RATIO['forearm'] * (w_vel - e_vel))**2, axis=1)
    ke_hand = 0.5 * m_hand * np.sum(w_vel**2, axis=1)
    ke_total = ke_ua + ke_fa + ke_hand

    # Linear momentum of arm chain
    p_ua = m_ua * (s_vel + SEGMENT_COM_RATIO['upper_arm'] * (e_vel - s_vel))
    p_fa = m_fa * (e_vel + SEGMENT_COM_RATIO['forearm'] * (w_vel - e_vel))
    p_hand = m_hand * w_vel
    p_total = p_ua + p_fa + p_hand

    # Joint power (tau dot omega)
    # Angular velocity from Savgol on angle
    ua_vec = e_smooth - s_smooth
    fa_vec = w_smooth - e_smooth

    # Power at each joint (force * velocity)
    power_shoulder = np.sum(F_shoulder * s_vel, axis=1)
    power_elbow = np.sum(F_elbow * e_vel, axis=1)
    power_wrist = np.sum(F_wrist * w_vel, axis=1)

    # Ball velocity estimate from Kalman-smoothed wrist
    ball_vel = w_vel.copy()  # simplified: ball velocity ~ wrist velocity at release

    return {
        # Smoothed kinematics
        'w_vel': w_vel,  # (240, 3) wrist velocity in m/s
        'w_acc': w_acc,  # (240, 3) wrist acceleration
        'e_vel': e_vel,
        's_vel': s_vel,
        'ball_vel': ball_vel,
        # Joint forces (magnitudes)
        'F_wrist_mag': np.linalg.norm(F_wrist, axis=1),
        'F_elbow_mag': np.linalg.norm(F_elbow, axis=1),
        'F_shoulder_mag': np.linalg.norm(F_shoulder, axis=1),
        # Joint torques
        'tau_wrist': tau_wrist,
        'tau_elbow': tau_elbow,
        'tau_shoulder': tau_shoulder,
        'tau_wrist_mag': np.linalg.norm(tau_wrist, axis=1),
        'tau_elbow_mag': np.linalg.norm(tau_elbow, axis=1),
        'tau_shoulder_mag': np.linalg.norm(tau_shoulder, axis=1),
        # Energy
        'ke_total': ke_total,
        'ke_hand': ke_hand,
        # Momentum
        'p_total': p_total,
        'p_hand': p_hand,
        # Power
        'power_shoulder': power_shoulder,
        'power_elbow': power_elbow,
        'power_wrist': power_wrist,
        # Segment properties
        'body_mass': body_mass_est,
        'm_hand': m_hand,
        'ua_len': ua_len,
        'fa_len': fa_len,
    }


# ============================================================
# FORWARD SIMULATION
# ============================================================

def forward_simulate(release_pos_m, release_vel_ms):
    """
    Forward simulate ball trajectory from release to hoop plane (z=10ft).

    Returns (angle_deg, depth_inches, left_right_inches) or None if invalid.
    """
    hoop_m = HOOP_POS * FEET_TO_METERS
    hoop_z = hoop_m[2]

    x0, y0, z0 = release_pos_m
    vx, vy, vz = release_vel_ms

    # Solve for time when z(t) = hoop_z
    # z0 + vz*t - 0.5*g*t^2 = hoop_z
    a = -0.5 * G
    b = vz
    c = z0 - hoop_z

    disc = b**2 - 4*a*c
    if disc < 0:
        return None

    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)

    # Take the positive time that represents the ball going UP then coming DOWN
    t_candidates = [t for t in [t1, t2] if t > 0.1]
    if not t_candidates:
        return None
    t = max(t_candidates)  # take later time (descending)

    # Ball position at hoop plane
    x_hoop = x0 + vx * t
    y_hoop = y0 + vy * t

    # Velocity at hoop plane
    vz_hoop = vz - G * t
    vh_hoop = np.sqrt(vx**2 + vy**2)

    # Entry angle (relative to horizontal)
    if vh_hoop < 1e-6:
        angle_deg = 90.0
    else:
        angle_deg = np.degrees(np.arctan(-vz_hoop / vh_hoop))

    # Depth and left_right (displacement from hoop center in inches)
    depth_inches = (y_hoop - hoop_m[1]) / 0.0254
    left_right_inches = (x_hoop - hoop_m[0]) / 0.0254

    return angle_deg, depth_inches, left_right_inches


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_dynamics_features(dynamics, release_frame):
    """Extract features from inverse dynamics results at and around release."""
    feats = {}
    rf = int(np.clip(release_frame, 10, 229))

    # --- Kalman-smoothed velocity at release ---
    for comp, name in enumerate(['fwd', 'lat', 'vert']):
        feats[f'kalman_wrist_vel_{name}'] = dynamics['w_vel'][rf, comp]
        feats[f'kalman_wrist_acc_{name}'] = dynamics['w_acc'][rf, comp]
        feats[f'kalman_elbow_vel_{name}'] = dynamics['e_vel'][rf, comp]
        feats[f'kalman_shoulder_vel_{name}'] = dynamics['s_vel'][rf, comp]
        feats[f'kalman_ball_vel_{name}'] = dynamics['ball_vel'][rf, comp]

    feats['kalman_wrist_speed'] = np.linalg.norm(dynamics['w_vel'][rf])
    feats['kalman_ball_speed'] = np.linalg.norm(dynamics['ball_vel'][rf])

    # --- Joint torques at release ---
    for comp, name in enumerate(['x', 'y', 'z']):
        feats[f'tau_shoulder_{name}'] = dynamics['tau_shoulder'][rf, comp]
        feats[f'tau_elbow_{name}'] = dynamics['tau_elbow'][rf, comp]

    feats['tau_shoulder_mag_release'] = dynamics['tau_shoulder_mag'][rf]
    feats['tau_elbow_mag_release'] = dynamics['tau_elbow_mag'][rf]
    feats['tau_wrist_mag_release'] = dynamics['tau_wrist_mag'][rf]

    # --- Peak torques in release window ---
    win_start = max(0, rf - 30)
    win_end = min(240, rf + 10)

    feats['tau_shoulder_peak'] = np.max(dynamics['tau_shoulder_mag'][win_start:win_end])
    feats['tau_elbow_peak'] = np.max(dynamics['tau_elbow_mag'][win_start:win_end])

    # Timing of peak torque (frames before release)
    feats['tau_shoulder_peak_time'] = rf - (
        win_start + np.argmax(dynamics['tau_shoulder_mag'][win_start:win_end]))
    feats['tau_elbow_peak_time'] = rf - (
        win_start + np.argmax(dynamics['tau_elbow_mag'][win_start:win_end]))

    # Proximal-to-distal delay (shoulder peak before elbow peak)
    sh_peak_frame = win_start + np.argmax(dynamics['tau_shoulder_mag'][win_start:win_end])
    el_peak_frame = win_start + np.argmax(dynamics['tau_elbow_mag'][win_start:win_end])
    feats['torque_pd_delay'] = (el_peak_frame - sh_peak_frame) * DT  # seconds

    # --- Joint forces at release ---
    feats['F_shoulder_release'] = dynamics['F_shoulder_mag'][rf]
    feats['F_elbow_release'] = dynamics['F_elbow_mag'][rf]
    feats['F_wrist_release'] = dynamics['F_wrist_mag'][rf]

    # Force ratios (how load is distributed)
    total_f = dynamics['F_shoulder_mag'][rf] + dynamics['F_elbow_mag'][rf] + 1e-9
    feats['F_elbow_ratio'] = dynamics['F_elbow_mag'][rf] / total_f
    feats['F_wrist_ratio'] = dynamics['F_wrist_mag'][rf] / total_f

    # --- Energy at release ---
    feats['ke_total_release'] = dynamics['ke_total'][rf]
    feats['ke_hand_release'] = dynamics['ke_hand'][rf]
    feats['ke_hand_ratio'] = dynamics['ke_hand'][rf] / (dynamics['ke_total'][rf] + 1e-9)

    # Peak kinetic energy
    feats['ke_total_peak'] = np.max(dynamics['ke_total'][win_start:win_end])
    # Energy at release as fraction of peak
    feats['ke_release_vs_peak'] = dynamics['ke_total'][rf] / (
        np.max(dynamics['ke_total'][win_start:win_end]) + 1e-9)

    # --- Momentum at release ---
    for comp, name in enumerate(['fwd', 'lat', 'vert']):
        feats[f'momentum_total_{name}'] = dynamics['p_total'][rf, comp]
        feats[f'momentum_hand_{name}'] = dynamics['p_hand'][rf, comp]

    feats['momentum_total_mag'] = np.linalg.norm(dynamics['p_total'][rf])
    feats['momentum_hand_mag'] = np.linalg.norm(dynamics['p_hand'][rf])

    # Momentum change at release (delta over 5 frames)
    if rf + 5 < 240 and rf - 5 >= 0:
        dp = dynamics['p_total'][rf+5] - dynamics['p_total'][rf-5]
        feats['momentum_change_mag'] = np.linalg.norm(dp)
        for comp, name in enumerate(['fwd', 'lat', 'vert']):
            feats[f'momentum_change_{name}'] = dp[comp]
    else:
        feats['momentum_change_mag'] = 0.0
        for name in ['fwd', 'lat', 'vert']:
            feats[f'momentum_change_{name}'] = 0.0

    # --- Power at release ---
    feats['power_shoulder_release'] = dynamics['power_shoulder'][rf]
    feats['power_elbow_release'] = dynamics['power_elbow'][rf]
    feats['power_wrist_release'] = dynamics['power_wrist'][rf]

    # Peak power
    feats['power_shoulder_peak'] = np.max(np.abs(dynamics['power_shoulder'][win_start:win_end]))
    feats['power_elbow_peak'] = np.max(np.abs(dynamics['power_elbow'][win_start:win_end]))
    feats['power_wrist_peak'] = np.max(np.abs(dynamics['power_wrist'][win_start:win_end]))

    # --- Segment properties ---
    feats['body_mass_est'] = dynamics['body_mass']
    feats['ua_len'] = dynamics['ua_len']
    feats['fa_len'] = dynamics['fa_len']

    # --- Forward-simulated targets ---
    release_pos_m = dynamics['w_vel'][rf] * 0  # placeholder - use actual position
    # We'll set this from outside

    return feats


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_shot(ts_3d, kp_index, smoother):
    """Process a single shot through the dynamics pipeline."""
    rs_idx = kp_index.get('right_shoulder')
    re_idx = kp_index.get('right_elbow')
    rw_idx = kp_index.get('right_wrist')

    if any(idx is None for idx in [rs_idx, re_idx, rw_idx]):
        return None, None

    s_pos = ts_3d[:, rs_idx, :].copy()
    e_pos = ts_3d[:, re_idx, :].copy()
    w_pos = ts_3d[:, rw_idx, :].copy()

    # Interpolate NaN/inf
    for arr in [s_pos, e_pos, w_pos]:
        for ax in range(3):
            vals = arr[:, ax]
            bad = np.isnan(vals) | np.isinf(vals)
            if np.all(bad):
                vals[:] = 0
            elif np.any(bad):
                good = ~bad
                vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])

    # Compute inverse dynamics
    dynamics = compute_inverse_dynamics(s_pos, e_pos, w_pos, 80.0, smoother)

    # Detect release frame from Kalman-smoothed wrist speed
    w_speed = np.linalg.norm(dynamics['w_vel'], axis=1)
    # Peak speed in frames 80-160
    search = w_speed[80:160]
    if len(search) > 0:
        release_frame = 80 + np.argmax(search)
    else:
        release_frame = 120

    # Extract features
    feats = extract_dynamics_features(dynamics, release_frame)
    feats['release_frame'] = release_frame

    # Forward simulation
    release_pos_m = w_pos[release_frame] * FEET_TO_METERS
    release_vel_ms = dynamics['ball_vel'][release_frame]

    sim_result = forward_simulate(release_pos_m, release_vel_ms)
    if sim_result is not None:
        feats['sim_angle'] = sim_result[0]
        feats['sim_depth'] = sim_result[1]
        feats['sim_left_right'] = sim_result[2]
    else:
        feats['sim_angle'] = 50.0
        feats['sim_depth'] = 0.0
        feats['sim_left_right'] = 0.0

    return feats, dynamics


def main():
    print("="*70)
    print("PHYSICS DYNAMICS MODEL")
    print("="*70)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    # Initialize RTS smoother
    # Tune: process_noise controls smoothness (higher = less smooth)
    # measurement_noise is the mocap noise (~10-15mm = 0.01-0.015m)
    smoother = RTSSmoother(process_noise_std=3.0, measurement_noise_std=0.012)

    def process_df(df, label):
        n = len(df)
        n_kp = len(keypoint_cols)
        all_feats = []
        X_raw = np.zeros((n, n_kp * 240), dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            ts_3d = np.zeros((240, len(kp_names), 3), dtype=np.float32)
            for j, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                ts_3d[:, j//3, j%3] = arr
                X_raw[i, j*240:(j+1)*240] = arr

            feats, _ = process_shot(ts_3d, kp_index, smoother)
            if feats is None:
                feats = {'release_frame': 120}
            feats['participant_id'] = row['participant_id']
            all_feats.append(feats)

            if (i + 1) % 50 == 0:
                print(f"  [{label}] {i+1}/{n}")

        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(all_feats), X_raw

    print("\nProcessing training shots...")
    train_feats, X_raw_train = process_df(train_df, "train")
    print("Processing test shots...")
    test_feats, X_raw_test = process_df(test_df, "test")

    exclude = {'participant_id', 'release_frame'}
    feat_cols = [c for c in train_feats.columns if c not in exclude]
    print(f"\nDynamics features: {len(feat_cols)}")

    # Validate forward simulation on training data
    y = train_df[['angle', 'depth', 'left_right']].values
    sim_angle = train_feats['sim_angle'].values
    sim_depth = train_feats['sim_depth'].values
    sim_lr = train_feats['sim_left_right'].values

    print("\n--- Forward simulation validation (raw units) ---")
    for name, sim, true in [('angle', sim_angle, y[:, 0]),
                             ('depth', sim_depth, y[:, 1]),
                             ('left_right', sim_lr, y[:, 2])]:
        valid = ~(np.isnan(sim) | np.isinf(sim) | np.isnan(true) | np.isinf(true))
        if np.sum(valid) > 10:
            r = np.corrcoef(sim[valid], true[valid])[0, 1]
            rmse = np.sqrt(np.mean((sim[valid] - true[valid])**2))
            print(f"  {name:12s}: r={r:+.4f}, RMSE={rmse:.2f}")
        else:
            print(f"  {name:12s}: insufficient valid predictions")

    # Kalman velocity vs Savgol velocity comparison
    print("\n--- Kalman vs Savgol velocity at release (correlation with targets) ---")
    pids = train_df['participant_id'].values
    for comp, cname in enumerate(['fwd', 'lat', 'vert']):
        kalman_v = train_feats[f'kalman_wrist_vel_{cname}'].values
        for t_idx, tname in enumerate(['angle', 'depth', 'left_right']):
            r = np.corrcoef(kalman_v, y[:, t_idx])[0, 1]
            print(f"  kalman_wrist_vel_{cname} vs {tname}: r={r:+.4f}")

    # =====================================================
    # CV EVALUATION
    # =====================================================
    print("\n" + "="*70)
    print("CROSS-VALIDATION")
    print("="*70)

    X_dyn = train_feats[feat_cols].values.astype(np.float32)
    X_dyn = np.nan_to_num(X_dyn, nan=0.0, posinf=0.0, neginf=0.0)

    pids = train_df['participant_id'].values
    unique_pids = sorted(np.unique(pids))

    configs = [
        ("Dynamics only", X_dyn, None, False),
        ("Dynamics + PLS", X_dyn, X_raw_train, True),
    ]

    for config_name, X, X_raw, use_pls in configs:
        print(f"\n--- {config_name} ({X.shape[1]} features) ---")

        for t_idx, target in enumerate(['angle', 'depth', 'left_right']):
            y_raw = y[:, t_idx]
            scaler_path = DATA_DIR / f"scaler_{target}.pkl"
            scaler = joblib.load(scaler_path)
            y_s = scaler.transform(y_raw.reshape(-1, 1)).ravel()

            oof = np.full(len(y_s), np.nan)
            for pid in unique_pids:
                mask = pids == pid
                X_p = X[mask]
                y_p = y_s[mask]
                raw_p = X_raw[mask] if use_pls else None
                indices = np.where(mask)[0]

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                for tr_idx, val_idx in kf.split(X_p):
                    X_tr, X_val = X_p[tr_idx], X_p[val_idx]
                    y_tr = y_p[tr_idx]

                    if use_pls and raw_p is not None:
                        ss = StandardScaler()
                        raw_tr_s = ss.fit_transform(raw_p[tr_idx])
                        raw_val_s = ss.transform(raw_p[val_idx])
                        nc = min(20, len(raw_tr_s) - 1)
                        if nc >= 3:
                            pls = PLSRegression(n_components=nc)
                            pls.fit(raw_tr_s, y_tr)
                            X_tr = np.hstack([X_tr, pls.transform(raw_tr_s)])
                            X_val = np.hstack([X_val, pls.transform(raw_val_s)])

                    ridge = Ridge(alpha=10.0)
                    ridge.fit(X_tr, y_tr)
                    pred_r = ridge.predict(X_val)

                    lgb_m = lgb.LGBMRegressor(
                        n_estimators=100, num_leaves=8,
                        min_child_samples=max(5, len(X_tr)//10),
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        verbose=-1, n_jobs=1)
                    lgb_m.fit(X_tr, y_tr)
                    pred_l = lgb_m.predict(X_val)

                    xgb_m = xgb.XGBRegressor(
                        n_estimators=100, max_depth=3,
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                        verbosity=0, n_jobs=1)
                    xgb_m.fit(X_tr, y_tr)
                    pred_x = xgb_m.predict(X_val)

                    oof[indices[val_idx]] = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x

            nan_mask = np.isnan(oof)
            if np.any(nan_mask):
                oof[nan_mask] = np.nanmean(y_s)

            mse = mean_squared_error(y_s, oof)
            r = np.corrcoef(y_s, oof)[0, 1] if np.std(oof) > 1e-9 else 0.0
            print(f"  {target:12s}: MSE={mse:.6f}, r={r:+.4f}")

    # =====================================================
    # GENERATE SUBMISSIONS
    # =====================================================
    print("\n" + "="*70)
    print("GENERATING SUBMISSIONS")
    print("="*70)

    # Train full model and predict test
    X_dyn_test = test_feats[feat_cols].values.astype(np.float32)
    X_dyn_test = np.nan_to_num(X_dyn_test, nan=0.0, posinf=0.0, neginf=0.0)
    pids_test = test_df['participant_id'].values

    test_preds = {}
    for t_idx, target in enumerate(['angle', 'depth', 'left_right']):
        y_raw = y[:, t_idx]
        scaler = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
        y_s = scaler.transform(y_raw.reshape(-1, 1)).ravel()

        preds = np.zeros(len(X_dyn_test))
        for pid in unique_pids:
            tr_m = pids == pid
            te_m = pids_test == pid
            if not np.any(tr_m) or not np.any(te_m):
                continue

            X_tr = X_dyn[tr_m]
            X_te = X_dyn_test[te_m]
            y_tr = y_s[tr_m]

            # Add PLS
            ss = StandardScaler()
            raw_tr_s = ss.fit_transform(X_raw_train[tr_m])
            raw_te_s = ss.transform(X_raw_test[te_m])
            nc = min(20, len(raw_tr_s) - 1)
            if nc >= 3:
                pls = PLSRegression(n_components=nc)
                pls.fit(raw_tr_s, y_tr)
                X_tr = np.hstack([X_tr, pls.transform(raw_tr_s)])
                X_te = np.hstack([X_te, pls.transform(raw_te_s)])

            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr, y_tr)
            pred_r = ridge.predict(X_te)

            lgb_m = lgb.LGBMRegressor(
                n_estimators=100, num_leaves=8,
                min_child_samples=max(5, len(X_tr)//10),
                learning_rate=0.05, verbose=-1, n_jobs=1)
            lgb_m.fit(X_tr, y_tr)
            pred_l = lgb_m.predict(X_te)

            xgb_m = xgb.XGBRegressor(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, verbosity=0, n_jobs=1)
            xgb_m.fit(X_tr, y_tr)
            pred_x = xgb_m.predict(X_te)

            preds[te_m] = 0.3 * pred_r + 0.35 * pred_l + 0.35 * pred_x

        test_preds[target] = np.clip(preds, 0, 1)

    # Blend with Sub 784
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
    max_num = max([int(p.stem.split('_')[1]) for p in existing
                   if p.stem.split('_')[1].isdigit()]) if existing else 0

    print("\n  Blending with Sub 784:")
    for aw, dw, lw in [(0.0, 0.10, 0.10), (0.0, 0.15, 0.15),
                        (0.0, 0.20, 0.20), (0.05, 0.15, 0.20),
                        (0.10, 0.20, 0.30)]:
        sub = sub_784.copy()
        sub['scaled_angle'] = (1-aw) * sub_784['scaled_angle'] + aw * test_preds['angle']
        sub['scaled_depth'] = (1-dw) * sub_784['scaled_depth'] + dw * test_preds['depth']
        sub['scaled_left_right'] = (1-lw) * sub_784['scaled_left_right'] + lw * test_preds['left_right']

        a_std = sub['scaled_angle'].std()
        d_mean = sub['scaled_depth'].mean()

        max_num += 1
        path = SUBMISSION_DIR / f"submission_{max_num}.csv"
        sub.to_csv(path, index=False)
        print(f"    aw={aw:.2f},dw={dw:.2f},lw={lw:.2f}: "
              f"a_std={a_std:.4f} d_mean={d_mean:.4f} -> {path.name}")


if __name__ == "__main__":
    main()
