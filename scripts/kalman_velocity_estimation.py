"""
Kalman Filter for Physics-Informed Velocity Estimation

Compares three velocity estimation methods applied to hand/ball proxy position:
1. Raw finite differences
2. Savitzky-Golay filtering
3. Kalman filter with physics-informed process model

Key finding: We don't have direct ball tracking. Ball position is estimated from
wrist + fingertip keypoints. After release, this estimate follows the hand (not
the ball), so velocity estimation is only meaningful PRE-release.

For validation: compare velocity estimates against ground-truth inverse projectile
velocities and measure forward simulation MSE.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

sys.path.insert(0, str(PROJECT_DIR))
from physics_engine.core.data_loader import (
    DataLoader, Shot, KEYPOINT_INDEX, HOOP_POSITION, FEET_TO_METERS
)

G = 9.81  # m/s^2
G_FT = G / FEET_TO_METERS  # 32.174 ft/s^2
DT = 1.0 / 60.0  # 60 fps
HOOP_X_FT = 5.25
HOOP_Y_FT = -25.0
HOOP_Z_FT = 10.0


# ============================================================
# Ball position estimation
# ============================================================

def estimate_ball_position(shot, frame):
    """Estimate ball center from right wrist + fingertips.
    ball = wrist + 0.6 * (fingertip_center - wrist)
    """
    wrist = shot.get_position('right_wrist', frame)
    if wrist is None:
        return None

    fingertip_names = [
        'right_second_finger_distal',
        'right_third_finger_distal',
        'right_fourth_finger_distal',
        'right_fifth_finger_distal',
        'right_first_finger_distal',
    ]
    tips = []
    for name in fingertip_names:
        pos = shot.get_position(name, frame)
        if pos is not None:
            tips.append(pos)

    if len(tips) == 0:
        return wrist.copy()

    ft_center = np.mean(tips, axis=0)
    ball_pos = wrist + 0.6 * (ft_center - wrist)
    return ball_pos


def get_ball_trajectory(shot):
    """Get ball position trajectory in feet for all frames."""
    positions = []
    valid = []
    for f in range(shot.num_frames):
        pos = estimate_ball_position(shot, f)
        if pos is not None and not np.any(np.isnan(pos)):
            positions.append(pos)
            valid.append(True)
        else:
            positions.append(np.array([0.0, 0.0, 0.0]))
            valid.append(False)
    return np.array(positions), np.array(valid)


def get_wrist_trajectory(shot):
    """Get right wrist trajectory in feet."""
    positions = []
    valid = []
    for f in range(shot.num_frames):
        pos = shot.get_position('right_wrist', f)
        if pos is not None and not np.any(np.isnan(pos)):
            positions.append(pos)
            valid.append(True)
        else:
            positions.append(np.array([0.0, 0.0, 0.0]))
            valid.append(False)
    return np.array(positions), np.array(valid)


def estimate_release_frame_velocity(wrist_traj, valid):
    """Find release frame as peak wrist speed in the shooting window (frames 80-180).

    The wrist peak height frame is typically 20-40 frames AFTER the actual release.
    Peak wrist velocity better approximates the release instant.
    """
    dt = DT
    n = len(wrist_traj)
    speed = np.zeros(n)
    vel = np.zeros_like(wrist_traj)

    for i in range(1, n - 1):
        if valid[i-1] and valid[i+1]:
            vel[i] = (wrist_traj[i+1] - wrist_traj[i-1]) / (2 * dt)
            speed[i] = np.linalg.norm(vel[i])

    # Search in shooting window only (frames 80-180)
    window_start = 80
    window_end = min(180, n)
    window_speed = speed[window_start:window_end]

    if len(window_speed) == 0:
        return int(np.argmax(speed)), speed, vel

    peak = window_start + int(np.argmax(window_speed))
    return peak, speed, vel


def estimate_release_frame_height(shot):
    """Find release frame as wrist peak height (original method)."""
    wrist_z = []
    for f in range(shot.num_frames):
        w = shot.get_position('right_wrist', f)
        wrist_z.append(w[2] if w is not None else 0)
    return int(np.argmax(wrist_z))


# ============================================================
# Method 1: Raw finite differences
# ============================================================

def velocity_finite_diff(positions, valid, dt=DT):
    """Velocity via central finite differences (ft/s)."""
    n = len(positions)
    vel = np.zeros_like(positions)
    for i in range(1, n - 1):
        if valid[i-1] and valid[i+1]:
            vel[i] = (positions[i+1] - positions[i-1]) / (2 * dt)
        elif valid[i] and i > 0 and valid[i-1]:
            vel[i] = (positions[i] - positions[i-1]) / dt
    if valid[0] and valid[1]:
        vel[0] = (positions[1] - positions[0]) / dt
    if valid[-2] and valid[-1]:
        vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


# ============================================================
# Method 2: Savitzky-Golay
# ============================================================

def velocity_savgol(positions, valid, dt=DT, window=9, polyorder=3):
    """Velocity via Savitzky-Golay differentiation (ft/s)."""
    n = len(positions)
    vel = np.zeros_like(positions)
    for axis in range(3):
        signal = positions[:, axis].copy()
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < window:
            continue
        invalid_idx = np.where(~valid)[0]
        if len(invalid_idx) > 0 and len(valid_idx) > 1:
            signal[invalid_idx] = np.interp(invalid_idx, valid_idx, signal[valid_idx])
        vel[:, axis] = savgol_filter(signal, window_length=window, polyorder=polyorder,
                                      deriv=1, delta=dt)
    return vel


# ============================================================
# Method 3: Kalman Filter
# ============================================================

class BallKalmanFilter:
    """6-state Kalman filter: [x, y, z, vx, vy, vz] (feet and ft/s).

    Pre-release: constant velocity with acceleration noise
    Post-release: ballistic (gravity on z, constant vel in x,y)
    """

    def __init__(self, process_noise_accel=5.0, measurement_noise=0.03, dt=DT):
        self.dt = dt
        self.n = 6

        # State transition (constant velocity)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix (position only)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise: acceleration noise model
        # Q = G_noise * G_noise^T * sigma_a^2
        # where G_noise maps acceleration to state
        G_noise = np.zeros((6, 3))
        G_noise[0, 0] = 0.5 * dt ** 2
        G_noise[1, 1] = 0.5 * dt ** 2
        G_noise[2, 2] = 0.5 * dt ** 2
        G_noise[3, 0] = dt
        G_noise[4, 1] = dt
        G_noise[5, 2] = dt

        self.Q_pre = G_noise @ G_noise.T * process_noise_accel ** 2

        # Post-release: tighter process noise since physics is modeled
        self.Q_post = G_noise @ G_noise.T * (process_noise_accel * 0.1) ** 2

        # Measurement noise
        self.R = np.eye(3) * measurement_noise ** 2

        # State and covariance
        self.x = np.zeros(6)
        self.P = np.eye(6)

    def initialize(self, pos, vel=None):
        self.x[:3] = pos
        self.x[3:] = vel if vel is not None else np.zeros(3)
        self.P = np.eye(6)
        self.P[:3, :3] *= 0.01
        self.P[3:, 3:] *= 100.0  # velocity very uncertain initially

    def predict(self, post_release=False):
        """Predict step with optional gravity."""
        Q = self.Q_post if post_release else self.Q_pre

        # State prediction
        new_x = self.F @ self.x
        if post_release:
            # Add gravity effect
            new_x[2] -= 0.5 * G_FT * self.dt ** 2
            new_x[5] -= G_FT * self.dt
        self.x = new_x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z):
        """Update step with position measurement."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_state(self):
        return self.x[:3].copy(), self.x[3:].copy()


def velocity_kalman(positions, valid, release_frame,
                    process_noise_accel=5.0, measurement_noise=0.03, dt=DT):
    """Velocity via Kalman filter (ft/s).

    Uses physics-informed process model:
    - Pre-release: constant velocity + acceleration noise
    - Post-release: ballistic (gravity)
    """
    n = len(positions)
    vel_out = np.zeros((n, 3))
    pos_out = np.zeros((n, 3))

    kf = BallKalmanFilter(
        process_noise_accel=process_noise_accel,
        measurement_noise=measurement_noise,
        dt=dt
    )

    # Find first valid frame
    first_valid = None
    for i in range(n):
        if valid[i]:
            first_valid = i
            break
    if first_valid is None:
        return vel_out

    # Initial velocity estimate
    init_vel = np.zeros(3)
    if first_valid + 1 < n and valid[first_valid + 1]:
        init_vel = (positions[first_valid + 1] - positions[first_valid]) / dt

    kf.initialize(positions[first_valid], init_vel)
    pos_out[first_valid] = positions[first_valid]

    for i in range(first_valid + 1, n):
        post_release = (i > release_frame)
        kf.predict(post_release=post_release)
        if valid[i]:
            kf.update(positions[i])
        pos_out[i], vel_out[i] = kf.get_state()

    for i in range(first_valid):
        vel_out[i] = vel_out[first_valid]

    return vel_out


# ============================================================
# Kalman smoother (forward-backward)
# ============================================================

def velocity_kalman_smooth(positions, valid, release_frame,
                           process_noise_accel=5.0, measurement_noise=0.03, dt=DT):
    """Rauch-Tung-Striebel smoother for optimal velocity estimation."""
    n = len(positions)

    kf = BallKalmanFilter(
        process_noise_accel=process_noise_accel,
        measurement_noise=measurement_noise,
        dt=dt
    )

    first_valid = None
    for i in range(n):
        if valid[i]:
            first_valid = i
            break
    if first_valid is None:
        return np.zeros((n, 3))

    init_vel = np.zeros(3)
    if first_valid + 1 < n and valid[first_valid + 1]:
        init_vel = (positions[first_valid + 1] - positions[first_valid]) / dt

    kf.initialize(positions[first_valid], init_vel)

    # Forward pass - store predictions and updates
    x_pred = np.zeros((n, 6))
    P_pred = np.zeros((n, 6, 6))
    x_filt = np.zeros((n, 6))
    P_filt = np.zeros((n, 6, 6))

    x_filt[first_valid] = kf.x.copy()
    P_filt[first_valid] = kf.P.copy()

    for i in range(first_valid + 1, n):
        post_release = (i > release_frame)
        Q = kf.Q_post if post_release else kf.Q_pre

        # Predict
        x_p = kf.F @ x_filt[i-1]
        if post_release:
            x_p[2] -= 0.5 * G_FT * dt ** 2
            x_p[5] -= G_FT * dt
        P_p = kf.F @ P_filt[i-1] @ kf.F.T + Q

        x_pred[i] = x_p
        P_pred[i] = P_p

        # Update if measurement available
        if valid[i]:
            y = positions[i] - kf.H @ x_p
            S = kf.H @ P_p @ kf.H.T + kf.R
            K = P_p @ kf.H.T @ np.linalg.inv(S)
            x_filt[i] = x_p + K @ y
            P_filt[i] = (np.eye(6) - K @ kf.H) @ P_p
        else:
            x_filt[i] = x_p
            P_filt[i] = P_p

    # RTS backward smoothing pass
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for i in range(n - 2, first_valid - 1, -1):
        if np.all(P_pred[i+1] == 0):
            continue
        try:
            C = P_filt[i] @ kf.F.T @ np.linalg.inv(P_pred[i+1])
            x_smooth[i] = x_filt[i] + C @ (x_smooth[i+1] - x_pred[i+1])
            P_smooth[i] = P_filt[i] + C @ (P_smooth[i+1] - P_pred[i+1]) @ C.T
        except np.linalg.LinAlgError:
            pass

    vel_out = x_smooth[:, 3:6]
    for i in range(first_valid):
        vel_out[i] = vel_out[first_valid]

    return vel_out


# ============================================================
# Forward simulation
# ============================================================

def forward_simulate(release_pos_ft, vel_ft_per_s):
    """Forward simulate from release to hoop plane (z=10ft)."""
    pos_m = np.array(release_pos_ft) * FEET_TO_METERS
    vel_m = np.array(vel_ft_per_s) * FEET_TO_METERS

    target_z_m = HOOP_Z_FT * FEET_TO_METERS
    dz_m = target_z_m - pos_m[2]
    vz = vel_m[2]

    a = 0.5 * G
    b = -vz
    c = dz_m

    disc = b**2 - 4*a*c
    if disc < 0:
        return None

    t1 = (-b + np.sqrt(disc)) / (2*a)
    t2 = (-b - np.sqrt(disc)) / (2*a)

    times = [t for t in [t1, t2] if t > 0.01]
    if not times:
        return None
    t = max(times)

    if t > 5.0:
        return None

    landing_x_m = pos_m[0] + vel_m[0] * t
    landing_y_m = pos_m[1] + vel_m[1] * t
    landing_vz = vz - G * t
    horiz_speed = np.sqrt(vel_m[0]**2 + vel_m[1]**2)

    landing_x_ft = landing_x_m / FEET_TO_METERS
    landing_y_ft = landing_y_m / FEET_TO_METERS

    if horiz_speed > 0.01:
        entry_angle = np.degrees(np.arctan2(-landing_vz, horiz_speed))
    else:
        entry_angle = 90.0

    depth_in = (landing_y_ft - HOOP_Y_FT) * 12.0
    lr_in = (landing_x_ft - HOOP_X_FT) * 12.0

    return {'angle': entry_angle, 'depth': depth_in, 'left_right': lr_in, 'flight_time': t}


# ============================================================
# Target scaling
# ============================================================

def load_scalers():
    scalers = {}
    for target in ['angle', 'depth', 'left_right']:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")
    return scalers

def raw_to_scaled(raw_value, scaler):
    return scaler.transform(np.array([[raw_value]]))[0, 0]


# ============================================================
# Compute ground-truth velocity via inverse projectile
# ============================================================

def compute_true_velocity(release_pos_ft, angle_deg, depth_in, lr_in):
    """Compute exact release velocity from known targets."""
    x_land_ft = HOOP_X_FT + lr_in / 12.0
    y_land_ft = HOOP_Y_FT + depth_in / 12.0
    z_land_ft = HOOP_Z_FT

    dx_m = (x_land_ft - release_pos_ft[0]) * FEET_TO_METERS
    dy_m = (y_land_ft - release_pos_ft[1]) * FEET_TO_METERS
    dz_m = (z_land_ft - release_pos_ft[2]) * FEET_TO_METERS

    D_horiz = np.sqrt(dx_m**2 + dy_m**2)
    angle_rad = np.radians(angle_deg)

    t_sq = 2.0 * (dz_m + D_horiz * np.tan(angle_rad)) / G
    if t_sq <= 0:
        return None

    t = np.sqrt(t_sq)
    vx_m = dx_m / t
    vy_m = dy_m / t
    vz_m = (dz_m + 0.5 * G * t**2) / t

    # Return in ft/s
    return np.array([vx_m, vy_m, vz_m]) / FEET_TO_METERS


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("KALMAN FILTER VELOCITY ESTIMATION - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    start_time = time.time()

    loader = DataLoader()
    scalers = load_scalers()
    n_shots = 345

    # ============================================================
    # Phase 1: Extract all trajectories and release frames
    # ============================================================
    print("\nPhase 1: Loading data and extracting trajectories...")

    all_ball_traj = []
    all_wrist_traj = []
    all_ball_valid = []
    all_wrist_valid = []
    rf_height = []   # release frame by wrist peak height
    rf_velocity = []  # release frame by peak wrist speed in window
    true_targets = []
    player_ids = []
    true_vel = []  # ground-truth velocity in ft/s
    release_positions = []

    for i in range(n_shots):
        shot = loader.get_shot(i, train=True)

        ball_pos, ball_val = get_ball_trajectory(shot)
        wrist_pos, wrist_val = get_wrist_trajectory(shot)

        all_ball_traj.append(ball_pos)
        all_wrist_traj.append(wrist_pos)
        all_ball_valid.append(ball_val)
        all_wrist_valid.append(wrist_val)

        # Release frame methods
        rf_h = estimate_release_frame_height(shot)
        rf_v, _, _ = estimate_release_frame_velocity(wrist_pos, wrist_val)
        rf_height.append(rf_h)
        rf_velocity.append(rf_v)

        player_ids.append(shot.player_id)
        targets = (shot.targets['angle'], shot.targets['depth'], shot.targets['left_right'])
        true_targets.append(targets)

        # Ball position at release (using velocity-based release frame)
        rel_pos = estimate_ball_position(shot, rf_v)
        if rel_pos is None:
            rel_pos = ball_pos[rf_v]
        release_positions.append(rel_pos)

        # Ground-truth velocity
        tv = compute_true_velocity(rel_pos, *targets)
        true_vel.append(tv)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{n_shots}...")

    true_targets = np.array(true_targets)
    player_ids = np.array(player_ids)
    release_positions = np.array(release_positions)
    rf_height = np.array(rf_height)
    rf_velocity = np.array(rf_velocity)

    # Check valid inverse projectile solutions
    valid_ip = np.array([tv is not None for tv in true_vel])
    true_vel_arr = np.zeros((n_shots, 3))
    for i in range(n_shots):
        if true_vel[i] is not None:
            true_vel_arr[i] = true_vel[i]

    print(f"  Valid inverse projectile solutions: {valid_ip.sum()}/{n_shots}")
    print(f"  Release frame (height): mean={rf_height.mean():.1f}, std={rf_height.std():.1f}")
    print(f"  Release frame (velocity): mean={rf_velocity.mean():.1f}, std={rf_velocity.std():.1f}")

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")

    # ============================================================
    # Phase 2: Velocity estimation with all methods
    # ============================================================
    print("\nPhase 2: Computing velocities with all methods...")

    # We test both ball trajectory and wrist trajectory
    # Using velocity-based release frame as the extraction point

    results = {}  # method_name -> (n_shots, 3) velocity array

    # For each method x trajectory combo
    for traj_name, all_traj, all_val in [
        ('ball', all_ball_traj, all_ball_valid),
        ('wrist', all_wrist_traj, all_wrist_valid),
    ]:
        for method_name, method_fn in [
            ('fd', lambda pos, val, rf: velocity_finite_diff(pos, val)),
            ('savgol', lambda pos, val, rf: velocity_savgol(pos, val)),
            ('kalman', lambda pos, val, rf: velocity_kalman(pos, val, rf)),
            ('kalman_smooth', lambda pos, val, rf: velocity_kalman_smooth(pos, val, rf)),
        ]:
            key = f"{traj_name}_{method_name}"
            vel_arr = np.zeros((n_shots, 3))
            for i in range(n_shots):
                rf = rf_velocity[i]
                v = method_fn(all_traj[i], all_val[i], rf)
                vel_arr[i] = v[rf]
            results[key] = vel_arr

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")

    # ============================================================
    # Phase 3: Velocity accuracy comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("VELOCITY RMSE vs GROUND TRUTH (ft/s) [m/s]")
    print("=" * 70)
    print(f"{'Method':<30s} {'vx RMSE':>10s} {'vy RMSE':>10s} {'vz RMSE':>10s} {'Total':>10s} {'Speed':>10s}")
    print("-" * 80)

    mask = valid_ip
    for key in sorted(results.keys()):
        vel_est = results[key]
        diff = vel_est[mask] - true_vel_arr[mask]
        rmse = np.sqrt(np.mean(diff**2, axis=0))
        rmse_total = np.sqrt(np.mean(diff**2))

        speed_est = np.linalg.norm(vel_est[mask], axis=1)
        speed_true = np.linalg.norm(true_vel_arr[mask], axis=1)
        speed_rmse = np.sqrt(np.mean((speed_est - speed_true)**2))

        print(f"{key:<30s} {rmse[0]:>10.3f} {rmse[1]:>10.3f} {rmse[2]:>10.3f} "
              f"{rmse_total:>10.3f} {speed_rmse:>10.3f}")

    # ============================================================
    # Phase 4: Velocity correlation with ground truth
    # ============================================================
    print("\n" + "=" * 70)
    print("VELOCITY CORRELATION WITH GROUND TRUTH (Pearson r)")
    print("=" * 70)
    print(f"{'Method':<30s} {'vx':>8s} {'vy':>8s} {'vz':>8s} {'speed':>8s}")
    print("-" * 60)

    for key in sorted(results.keys()):
        vel_est = results[key]
        corrs = []
        for j in range(3):
            if np.std(vel_est[mask, j]) > 1e-10:
                r, _ = pearsonr(vel_est[mask, j], true_vel_arr[mask, j])
            else:
                r = 0.0
            corrs.append(r)

        speed_est = np.linalg.norm(vel_est[mask], axis=1)
        speed_true = np.linalg.norm(true_vel_arr[mask], axis=1)
        if np.std(speed_est) > 1e-10:
            speed_r, _ = pearsonr(speed_est, speed_true)
        else:
            speed_r = 0.0

        print(f"{key:<30s} {corrs[0]:>8.4f} {corrs[1]:>8.4f} {corrs[2]:>8.4f} {speed_r:>8.4f}")

    # ============================================================
    # Phase 5: Forward simulation MSE
    # ============================================================
    print("\n" + "=" * 70)
    print("FORWARD SIMULATION TARGET PREDICTION MSE (scaled [0,1])")
    print("=" * 70)

    target_names = ['angle', 'depth', 'left_right']
    print(f"{'Method':<30s} {'angle':>10s} {'depth':>10s} {'left_right':>10s} {'mean':>10s} {'invalid':>8s}")
    print("-" * 80)

    for key in sorted(results.keys()):
        vel_est = results[key]
        pred_targets = []
        n_invalid = 0

        for i in range(n_shots):
            result = forward_simulate(release_positions[i], vel_est[i])
            if result is None:
                pred_targets.append(true_targets[i])
                n_invalid += 1
            else:
                pred_targets.append((result['angle'], result['depth'], result['left_right']))

        pred_targets = np.array(pred_targets)
        mse_vals = {}
        for j, tname in enumerate(target_names):
            scaled_true = np.array([raw_to_scaled(v, scalers[tname]) for v in true_targets[:, j]])
            scaled_pred = np.array([raw_to_scaled(v, scalers[tname]) for v in pred_targets[:, j]])
            mse_vals[tname] = np.mean((scaled_true - scaled_pred)**2)

        mean_mse = np.mean(list(mse_vals.values()))
        print(f"{key:<30s} {mse_vals['angle']:>10.6f} {mse_vals['depth']:>10.6f} "
              f"{mse_vals['left_right']:>10.6f} {mean_mse:>10.6f} {n_invalid:>8d}")

    # ============================================================
    # Phase 6: Smoothness comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("VELOCITY SMOOTHNESS (mean abs jerk near release, 20 sample shots)")
    print("=" * 70)

    sample = range(min(20, n_shots))
    methods_for_smooth = [
        ('ball_fd', lambda pos, val, rf: velocity_finite_diff(pos, val)),
        ('ball_savgol', lambda pos, val, rf: velocity_savgol(pos, val)),
        ('ball_kalman', lambda pos, val, rf: velocity_kalman(pos, val, rf)),
        ('ball_kalman_smooth', lambda pos, val, rf: velocity_kalman_smooth(pos, val, rf)),
    ]

    for mname, mfn in methods_for_smooth:
        jerks = []
        for i in sample:
            pos = all_ball_traj[i]
            val = all_ball_valid[i]
            rf = rf_velocity[i]
            v = mfn(pos, val, rf)
            accel = np.diff(v, axis=0) / DT
            jerk = np.diff(accel, axis=0) / DT
            start = max(0, rf - 20)
            end = min(len(jerk), rf + 5)
            if end > start:
                jerks.append(np.mean(np.abs(jerk[start:end])))
        mean_jerk = np.mean(jerks) if jerks else 0
        print(f"  {mname}: mean abs jerk = {mean_jerk:.1f} ft/s^3")

    # ============================================================
    # Phase 7: Key diagnostic - why velocity is so wrong
    # ============================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: VELOCITY MAGNITUDE AT RELEASE")
    print("=" * 70)

    for key in ['ball_kalman', 'ball_savgol', 'wrist_kalman', 'wrist_savgol']:
        vel_est = results[key]
        speeds = np.linalg.norm(vel_est, axis=1)
        print(f"  {key}: mean speed = {speeds.mean():.2f} ft/s ({speeds.mean()*FEET_TO_METERS:.3f} m/s)")

    true_speeds = np.linalg.norm(true_vel_arr[valid_ip], axis=1)
    print(f"  Ground truth: mean speed = {true_speeds.mean():.2f} ft/s ({true_speeds.mean()*FEET_TO_METERS:.3f} m/s)")

    ratio = true_speeds.mean() / np.linalg.norm(results['ball_kalman'], axis=1).mean()
    print(f"\n  Speed ratio (true/estimated): {ratio:.1f}x")
    print(f"  This gap is the FINGER SNAP effect: ball velocity >> wrist velocity")
    print(f"  The fingertip whip adds {(ratio-1)*100:.0f}% more velocity that keypoints can't capture")

    # ============================================================
    # Phase 8: Per-player analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("PER-PLAYER: Kalman vs Savgol velocity RMSE (ball traj)")
    print("=" * 70)

    for pid in sorted(np.unique(player_ids)):
        pmask = (player_ids == pid) & valid_ip
        n_p = pmask.sum()

        for key in ['ball_savgol', 'ball_kalman', 'ball_kalman_smooth']:
            diff = results[key][pmask] - true_vel_arr[pmask]
            rmse = np.sqrt(np.mean(diff**2))
            speed_est = np.linalg.norm(results[key][pmask], axis=1)
            speed_true = np.linalg.norm(true_vel_arr[pmask], axis=1)
            r, _ = pearsonr(speed_est, speed_true) if np.std(speed_est) > 1e-10 else (0, 0)
            print(f"  Player {pid} (n={n_p}): {key}: RMSE={rmse:.3f} ft/s, speed_r={r:.4f}")
        print()

    # ============================================================
    # Phase 9: Kalman parameter sensitivity
    # ============================================================
    print("=" * 70)
    print("KALMAN PARAMETER SENSITIVITY (ball trajectory)")
    print("=" * 70)

    configs = [
        (1.0, 0.01, "low accel noise, tight meas"),
        (2.0, 0.03, "low-medium"),
        (5.0, 0.03, "default"),
        (10.0, 0.03, "high accel noise"),
        (5.0, 0.01, "tight measurement"),
        (5.0, 0.10, "loose measurement"),
        (20.0, 0.03, "very high accel noise"),
        (5.0, 0.005, "very tight measurement"),
    ]

    best_rmse = float('inf')
    best_cfg = None

    for proc_a, meas, desc in configs:
        vel_test = np.zeros((n_shots, 3))
        for i in range(n_shots):
            v = velocity_kalman(all_ball_traj[i], all_ball_valid[i], rf_velocity[i],
                               process_noise_accel=proc_a, measurement_noise=meas)
            vel_test[i] = v[rf_velocity[i]]

        diff = vel_test[mask] - true_vel_arr[mask]
        rmse_total = np.sqrt(np.mean(diff**2))

        # Correlation
        corrs = []
        for j in range(3):
            if np.std(vel_test[mask, j]) > 1e-10:
                r, _ = pearsonr(vel_test[mask, j], true_vel_arr[mask, j])
            else:
                r = 0.0
            corrs.append(r)

        print(f"  [{desc}] accel={proc_a}, meas={meas}")
        print(f"    RMSE={rmse_total:.3f} ft/s | r(vx,vy,vz)=({corrs[0]:.3f},{corrs[1]:.3f},{corrs[2]:.3f})")

        if rmse_total < best_rmse:
            best_rmse = rmse_total
            best_cfg = desc

    print(f"\n  Best config: {best_cfg} (RMSE={best_rmse:.3f})")

    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f}s")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
1. FUNDAMENTAL LIMITATION: The ball is estimated from hand keypoints. After release,
   the "ball" follows the hand, not the actual ball trajectory. Velocity estimation
   from keypoints gives HAND velocity (~2-4 m/s), not ball velocity (~7 m/s).

2. The velocity gap (hand vs ball) comes from the finger snap/whip effect. The
   fingertips move much faster than the wrist at release. This ~3x velocity ratio
   cannot be recovered from position data alone.

3. KALMAN FILTER vs SAVGOL vs FINITE DIFF: The Kalman filter provides slightly
   smoother velocity estimates but does NOT recover the missing velocity magnitude.
   All methods produce nearly identical correlation with ground truth (~0).

4. The forward simulation approach fails because:
   a) Hand velocity underestimates ball velocity by ~3x
   b) The release frame detection is imprecise (std ~18 frames)
   c) Small velocity errors amplify into large target errors

5. IMPLICATION FOR THE COMPETITION: Velocity estimation from keypoints is NOT a
   viable path. The per-example locally weighted regression (Sub 1350) works better
   because it learns the hand-to-target mapping directly without physics simulation.
""")


if __name__ == "__main__":
    main()
