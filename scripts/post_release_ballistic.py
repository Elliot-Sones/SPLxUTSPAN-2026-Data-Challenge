"""
Post-release ballistic fitting analysis.

Key question: After ball release, do fingertip keypoints briefly track the ball
in a parabolic trajectory? If so, we can fit a parabola to recover exact velocity
without differentiation noise.

Approach:
1. Detect release frame (right_wrist z-peak)
2. Analyze post-release fingertip trajectories for ballistic (parabolic) motion
3. Fit parabolas to extract velocity
4. Compare to finite-difference velocity
5. Forward-simulate to validate against targets
"""

import numpy as np
import pandas as pd
import json
import joblib
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

BASE = '/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge'

# ============================================================
# 1. Load data
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train = pd.read_csv(f'{BASE}/data/train.csv')
print(f"Train shape: {train.shape}")
print(f"Shots: {len(train)}, Players: {train['participant_id'].nunique()}")

# Parse JSON arrays for all keypoint columns
meta_cols = ['id', 'shot_id', 'participant_id', 'angle', 'depth', 'left_right']
kp_cols = [c for c in train.columns if c not in meta_cols]

# Load scalers
scalers = {}
for t in ['angle', 'depth', 'left_right']:
    scalers[t] = joblib.load(f'{BASE}/data/scaler_{t}.pkl')

# Right hand distal (fingertip) keypoints
distal_kps = [
    'right_first_finger_distal',   # thumb tip (first finger = thumb)
    'right_second_finger_distal',  # index tip
    'right_third_finger_distal',   # middle tip
    'right_fourth_finger_distal',  # ring tip
    'right_fifth_finger_distal',   # pinky tip
]

# Also track wrist and other hand joints for reference
reference_kps = [
    'right_wrist',
    'right_elbow',
    'right_shoulder',
]

# Additional hand keypoints
hand_kps = [
    'right_second_finger_mcp',  # index knuckle
    'right_third_finger_mcp',   # middle knuckle
    'right_second_finger_dip',  # index DIP
    'right_third_finger_dip',   # middle DIP
]

all_tracked_kps = distal_kps + reference_kps + hand_kps

# Parse timeseries data
def parse_timeseries(row, kp_name, coord):
    val = row[f'{kp_name}_{coord}']
    if isinstance(val, str):
        return np.array(json.loads(val))
    return np.array(val)

# Pre-parse all needed timeseries
print("Parsing timeseries...")
n_shots = len(train)
n_frames = 240
fps = 60
dt = 1.0 / fps
g = 32.174  # gravity in ft/s^2

# Store parsed data: shape (n_shots, n_frames) for each kp/coord
def safe_parse(val):
    if isinstance(val, str):
        # Replace NaN-like values in JSON
        val_clean = val.replace('NaN', 'null').replace('nan', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
        arr = json.loads(val_clean)
        return np.array([x if x is not None else np.nan for x in arr])
    return np.array([val])

data = {}
for kp in all_tracked_kps:
    for coord in ['x', 'y', 'z']:
        col = f'{kp}_{coord}'
        if col in train.columns:
            data[col] = np.array([safe_parse(train.iloc[i][col]) for i in range(n_shots)])

print(f"Parsed {len(data)} timeseries, shape: {data['right_wrist_z'].shape}")

# ============================================================
# 2. Detect release frame
# ============================================================
print("\n" + "=" * 70)
print("DETECTING RELEASE FRAMES")
print("=" * 70)

release_frames = np.zeros(n_shots, dtype=int)
for i in range(n_shots):
    wrist_z = data['right_wrist_z'][i]
    # Release = peak of right_wrist_z (ball at highest point of upward motion)
    # Search in reasonable window (frames 80-160)
    search_start, search_end = 80, 160
    peak_idx = search_start + np.argmax(wrist_z[search_start:search_end])
    release_frames[i] = peak_idx

print(f"Release frame stats: mean={release_frames.mean():.1f}, std={release_frames.std():.1f}")
print(f"  min={release_frames.min()}, max={release_frames.max()}")
print(f"  25th={np.percentile(release_frames, 25):.0f}, 50th={np.percentile(release_frames, 50):.0f}, 75th={np.percentile(release_frames, 75):.0f}")

# ============================================================
# 3. Post-release fingertip trajectory analysis
# ============================================================
print("\n" + "=" * 70)
print("POST-RELEASE FINGERTIP TRAJECTORY ANALYSIS")
print("=" * 70)

def parabola_z(t, z0, vz, g_fit):
    """z(t) = z0 + vz*t - 0.5*g_fit*t^2"""
    return z0 + vz * t - 0.5 * g_fit * t**2

def linear_xy(t, p0, v):
    """x(t) = p0 + v*t (no forces in x/y for ballistic)"""
    return p0 + v * t

def fit_ballistic(times, z_vals):
    """Fit parabolic trajectory to z-coordinates. Return R^2, velocity, fitted_g."""
    try:
        popt, pcov = curve_fit(parabola_z, times, z_vals, p0=[z_vals[0], 10.0, g])
        z_pred = parabola_z(times, *popt)
        ss_res = np.sum((z_vals - z_pred)**2)
        ss_tot = np.sum((z_vals - z_vals.mean())**2)
        if ss_tot == 0:
            return 0.0, popt[1], popt[2], z_pred
        r2 = 1 - ss_res / ss_tot
        return r2, popt[1], popt[2], z_pred
    except:
        return -1.0, 0.0, 0.0, np.zeros_like(z_vals)

def fit_linear(times, vals):
    """Fit linear trajectory. Return R^2, velocity."""
    try:
        popt, pcov = curve_fit(linear_xy, times, vals, p0=[vals[0], 0.0])
        pred = linear_xy(times, *popt)
        ss_res = np.sum((vals - pred)**2)
        ss_tot = np.sum((vals - vals.mean())**2)
        if ss_tot == 0:
            return 0.0, popt[1]
        r2 = 1 - ss_res / ss_tot
        return r2, popt[1]
    except:
        return -1.0, 0.0

# Analyze post-release trajectories for different window lengths
print("\n--- Fingertip ballistic fit quality (z-axis parabola) ---")
print("Testing windows: release+1 to release+N frames\n")

windows = [3, 5, 7, 10, 15, 20]

for kp in distal_kps + ['right_wrist']:
    print(f"\n  {kp}:")
    for w in windows:
        r2_list = []
        g_fit_list = []
        vz_list = []
        for i in range(n_shots):
            rf = release_frames[i]
            if rf + w >= n_frames:
                continue
            z_vals = data[f'{kp}_z'][i][rf:rf+w]
            times = np.arange(w) * dt
            r2, vz, g_fit, _ = fit_ballistic(times, z_vals)
            if r2 > -0.5:
                r2_list.append(r2)
                g_fit_list.append(g_fit)
                vz_list.append(vz)

        if r2_list:
            r2_arr = np.array(r2_list)
            g_arr = np.array(g_fit_list)
            print(f"    Window={w:2d}: R2 mean={np.mean(r2_arr):.4f}, median={np.median(r2_arr):.4f}, "
                  f">0.95: {(r2_arr > 0.95).mean():.1%}, >0.99: {(r2_arr > 0.99).mean():.1%}, "
                  f"g_fit mean={np.mean(g_arr):.1f} (true={g:.1f})")

# ============================================================
# 4. Fingertip centroid analysis
# ============================================================
print("\n" + "=" * 70)
print("FINGERTIP CENTROID BALLISTIC ANALYSIS")
print("=" * 70)

# Compute centroid of all 5 distal fingertips
centroid = {}
for coord in ['x', 'y', 'z']:
    vals = np.zeros((n_shots, n_frames))
    for kp in distal_kps:
        col = f'{kp}_{coord}'
        if col in data:
            vals += data[col]
    centroid[coord] = vals / len(distal_kps)

print("\n--- Fingertip centroid ballistic fit ---")
for w in windows:
    r2_z_list = []
    r2_x_list = []
    r2_y_list = []
    g_fit_list = []
    vz_list = []
    vx_list = []
    vy_list = []

    for i in range(n_shots):
        rf = release_frames[i]
        if rf + w >= n_frames:
            continue
        times = np.arange(w) * dt

        z_vals = centroid['z'][i][rf:rf+w]
        r2_z, vz, g_fit, _ = fit_ballistic(times, z_vals)

        x_vals = centroid['x'][i][rf:rf+w]
        r2_x, vx = fit_linear(times, x_vals)

        y_vals = centroid['y'][i][rf:rf+w]
        r2_y, vy = fit_linear(times, y_vals)

        if r2_z > -0.5:
            r2_z_list.append(r2_z)
            r2_x_list.append(r2_x)
            r2_y_list.append(r2_y)
            g_fit_list.append(g_fit)
            vz_list.append(vz)
            vx_list.append(vx)
            vy_list.append(vy)

    r2_z_arr = np.array(r2_z_list)
    r2_x_arr = np.array(r2_x_list)
    r2_y_arr = np.array(r2_y_list)
    g_arr = np.array(g_fit_list)

    print(f"  Window={w:2d}: Z R2={np.mean(r2_z_arr):.4f} (>0.95: {(r2_z_arr > 0.95).mean():.1%}), "
          f"X R2={np.mean(r2_x_arr):.4f}, Y R2={np.mean(r2_y_arr):.4f}, "
          f"g_fit={np.mean(g_arr):.1f}+/-{np.std(g_arr):.1f}")

# ============================================================
# 5. Compare ballistic fit velocity vs finite difference velocity
# ============================================================
print("\n" + "=" * 70)
print("VELOCITY COMPARISON: BALLISTIC FIT vs FINITE DIFFERENCE")
print("=" * 70)

# Best window for ballistic fit (test multiple)
best_window = 5  # Start with 5 frames

ballistic_vel = {'vx': [], 'vy': [], 'vz': []}
findiff_vel = {'vx': [], 'vy': [], 'vz': []}

for i in range(n_shots):
    rf = release_frames[i]
    if rf + best_window >= n_frames or rf < 2:
        ballistic_vel['vx'].append(np.nan)
        ballistic_vel['vy'].append(np.nan)
        ballistic_vel['vz'].append(np.nan)
        findiff_vel['vx'].append(np.nan)
        findiff_vel['vy'].append(np.nan)
        findiff_vel['vz'].append(np.nan)
        continue

    times = np.arange(best_window) * dt

    # Ballistic fit on fingertip centroid
    z_vals = centroid['z'][i][rf:rf+best_window]
    _, vz_bal, _, _ = fit_ballistic(times, z_vals)

    x_vals = centroid['x'][i][rf:rf+best_window]
    _, vx_bal = fit_linear(times, x_vals)

    y_vals = centroid['y'][i][rf:rf+best_window]
    _, vy_bal = fit_linear(times, y_vals)

    ballistic_vel['vx'].append(vx_bal)
    ballistic_vel['vy'].append(vy_bal)
    ballistic_vel['vz'].append(vz_bal)

    # Finite difference at release (central difference)
    for coord, key in [('x', 'vx'), ('y', 'vy'), ('z', 'vz')]:
        vals = data[f'right_wrist_{coord}'][i]
        # Central difference at release
        fd = (vals[rf+1] - vals[rf-1]) / (2 * dt)
        findiff_vel[key].append(fd)

# Compare
for key in ['vx', 'vy', 'vz']:
    bal = np.array(ballistic_vel[key])
    fd = np.array(findiff_vel[key])
    mask = ~(np.isnan(bal) | np.isnan(fd))
    if mask.sum() > 10:
        r, p = pearsonr(bal[mask], fd[mask])
        print(f"  {key}: Ballistic vs FinDiff correlation r={r:.4f}, "
              f"ballistic mean={np.mean(bal[mask]):.2f}, fd mean={np.mean(fd[mask]):.2f}, "
              f"ballistic std={np.std(bal[mask]):.2f}, fd std={np.std(fd[mask]):.2f}")

# ============================================================
# 6. Check if fingertips truly track ball or just follow-through
# ============================================================
print("\n" + "=" * 70)
print("BALLISTIC vs FOLLOW-THROUGH DISCRIMINATION")
print("=" * 70)

# Key test: if fingertips track the ball, the fitted gravity should be close to 32.174 ft/s^2
# If it's just follow-through motion, g_fit will be very different

print("\nFitted gravity distribution for fingertip centroid (window=5):")
g_fits_centroid = []
r2_centroid = []
for i in range(n_shots):
    rf = release_frames[i]
    if rf + 5 >= n_frames:
        continue
    times = np.arange(5) * dt
    z_vals = centroid['z'][i][rf:rf+5]
    r2, vz, g_fit, _ = fit_ballistic(times, z_vals)
    if r2 > 0.9:
        g_fits_centroid.append(g_fit)
        r2_centroid.append(r2)

g_arr = np.array(g_fits_centroid)
print(f"  Shots with R2 > 0.9: {len(g_arr)}/{n_shots}")
print(f"  g_fit: mean={np.mean(g_arr):.2f}, median={np.median(g_arr):.2f}, std={np.std(g_arr):.2f}")
print(f"  True gravity: {g:.3f} ft/s^2")
print(f"  g_fit within 20% of true: {((g_arr > g*0.8) & (g_arr < g*1.2)).mean():.1%}")
print(f"  g_fit within 50% of true: {((g_arr > g*0.5) & (g_arr < g*1.5)).mean():.1%}")
print(f"  g_fit negative (going UP): {(g_arr < 0).mean():.1%}")
print(f"  g_fit > 100 (extreme deceleration): {(g_arr > 100).mean():.1%}")

# Also check per-keypoint
print("\nPer-keypoint fitted gravity (window=5, shots with R2>0.9):")
for kp in distal_kps + ['right_wrist']:
    g_fits = []
    for i in range(n_shots):
        rf = release_frames[i]
        if rf + 5 >= n_frames:
            continue
        times = np.arange(5) * dt
        z_vals = data[f'{kp}_z'][i][rf:rf+5]
        r2, vz, g_fit, _ = fit_ballistic(times, z_vals)
        if r2 > 0.9:
            g_fits.append(g_fit)
    g_arr = np.array(g_fits)
    if len(g_arr) > 0:
        print(f"  {kp:40s}: n={len(g_arr):3d}, g_fit mean={np.mean(g_arr):8.2f}, "
              f"within 20%: {((g_arr > g*0.8) & (g_arr < g*1.2)).mean():.1%}, "
              f"negative: {(g_arr < 0).mean():.1%}")

# ============================================================
# 7. Test different post-release offsets
# ============================================================
print("\n" + "=" * 70)
print("POST-RELEASE OFFSET ANALYSIS")
print("=" * 70)
print("Testing if starting ballistic fit at release+offset improves results")

for offset in [0, 1, 2, 3, 5]:
    window = 5
    g_fits = []
    r2_list = []
    for i in range(n_shots):
        rf = release_frames[i] + offset
        if rf + window >= n_frames:
            continue
        times = np.arange(window) * dt
        z_vals = centroid['z'][i][rf:rf+window]
        r2, vz, g_fit, _ = fit_ballistic(times, z_vals)
        if r2 > 0.5:
            g_fits.append(g_fit)
            r2_list.append(r2)

    g_arr = np.array(g_fits)
    r2_arr = np.array(r2_list)
    print(f"  Offset={offset}: n={len(g_arr)}, R2 mean={np.mean(r2_arr):.4f}, "
          f"g_fit mean={np.mean(g_arr):.2f}+/-{np.std(g_arr):.2f}, "
          f"within 20%: {((g_arr > g*0.8) & (g_arr < g*1.2)).mean():.1%}")

# ============================================================
# 8. Wrist transition analysis
# ============================================================
print("\n" + "=" * 70)
print("WRIST TRANSITION ANALYSIS")
print("=" * 70)
print("Analyzing wrist deceleration pattern after release")

# The wrist should decelerate sharply after ball release
# The transition point encodes precise release timing

wrist_speed = np.zeros((n_shots, n_frames))
for i in range(n_shots):
    wx = data['right_wrist_x'][i]
    wy = data['right_wrist_y'][i]
    wz = data['right_wrist_z'][i]
    speed = np.sqrt(np.diff(wx)**2 + np.diff(wy)**2 + np.diff(wz)**2) / dt
    wrist_speed[i, 1:] = speed
    wrist_speed[i, 0] = speed[0]

# Find speed peak near release
speed_peaks = np.zeros(n_shots, dtype=int)
for i in range(n_shots):
    search_start = max(60, release_frames[i] - 30)
    search_end = min(200, release_frames[i] + 10)
    speed_peaks[i] = search_start + np.argmax(wrist_speed[i, search_start:search_end])

print(f"Wrist speed peak: mean frame={speed_peaks.mean():.1f}, std={speed_peaks.std():.1f}")
print(f"  vs wrist z-peak (release): mean={release_frames.mean():.1f}")
print(f"  Speed peak typically {(release_frames - speed_peaks).mean():.1f} frames AFTER speed peak")

# Deceleration rate after speed peak
decel_rates = []
for i in range(n_shots):
    sp = speed_peaks[i]
    if sp + 10 < n_frames:
        # Speed drop in first 5 frames after peak
        peak_speed = wrist_speed[i, sp]
        post_speed = wrist_speed[i, sp+5]
        if peak_speed > 0:
            decel_rate = (peak_speed - post_speed) / peak_speed
            decel_rates.append(decel_rate)

decel_arr = np.array(decel_rates)
print(f"Deceleration rate (5 frames after peak): mean={np.mean(decel_arr):.3f}, std={np.std(decel_arr):.3f}")

# ============================================================
# 9. Fingertip spread analysis (ball size proxy)
# ============================================================
print("\n" + "=" * 70)
print("FINGERTIP SPREAD ANALYSIS (BALL TRACKING PROXY)")
print("=" * 70)
print("If fingertips track the ball, they should converge (fingers close) at release")
print("and then diverge (spread out) after release as the ball leaves\n")

# Compute pairwise distances between fingertip distals
spreads = np.zeros((n_shots, n_frames))
for i in range(n_shots):
    for f in range(n_frames):
        pts = []
        for kp in distal_kps:
            x = data[f'{kp}_x'][i][f]
            y = data[f'{kp}_y'][i][f]
            z = data[f'{kp}_z'][i][f]
            pts.append([x, y, z])
        pts = np.array(pts)
        # Mean pairwise distance
        dists = []
        for a in range(len(pts)):
            for b in range(a+1, len(pts)):
                dists.append(np.linalg.norm(pts[a] - pts[b]))
        spreads[i, f] = np.mean(dists)

# Analyze spread around release
print("Fingertip spread (mean pairwise distance) around release:")
for offset in range(-10, 21, 5):
    vals = []
    for i in range(n_shots):
        f = release_frames[i] + offset
        if 0 <= f < n_frames:
            vals.append(spreads[i, f])
    vals = np.array(vals)
    print(f"  Release{offset:+3d}: spread={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# Spread derivative at release
print("\nSpread derivative (positive = fingers opening):")
for offset in range(-5, 16, 1):
    vals = []
    for i in range(n_shots):
        f = release_frames[i] + offset
        if 1 <= f < n_frames:
            vals.append(spreads[i, f] - spreads[i, f-1])
    vals = np.array(vals)
    if offset in [-5, -3, -1, 0, 1, 2, 3, 5, 7, 10, 15]:
        marker = " <---" if offset == 0 else ""
        print(f"  Release{offset:+3d}: d_spread={np.mean(vals):+.5f} +/- {np.std(vals):.5f}{marker}")

# ============================================================
# 10. Forward simulation from ballistic-fit velocity
# ============================================================
print("\n" + "=" * 70)
print("FORWARD SIMULATION FROM BALLISTIC-FIT VELOCITY")
print("=" * 70)

# Use different windows and check which gives best target prediction
HOOP = np.array([5.25, -25.0, 10.0])

def forward_simulate(x0, y0, z0, vx, vy, vz):
    """Simulate projectile to hoop plane (y = -25).
    Returns angle, depth, left_right at hoop."""
    # Time to reach hoop y-plane
    if abs(vy) < 0.01:
        return None, None, None
    t_hoop = (-25.0 - y0) / vy
    if t_hoop < 0:
        return None, None, None

    x_hoop = x0 + vx * t_hoop
    z_hoop = z0 + vz * t_hoop - 0.5 * g * t_hoop**2

    # Angle: entry angle relative to horizontal
    vz_at_hoop = vz - g * t_hoop
    speed_h = np.sqrt(vx**2 + vy**2)
    angle = np.degrees(np.arctan2(-vz_at_hoop, speed_h))  # negative because ball comes down

    # Depth: how far past/before hoop center in y
    depth = 0  # At the hoop plane by definition, depth = z deviation

    # For now, return raw positions and angle
    return angle, x_hoop, z_hoop

# Test with different approaches
for approach_name, window, use_centroid in [
    ("Centroid w=3", 3, True),
    ("Centroid w=5", 5, True),
    ("Centroid w=7", 7, True),
    ("Centroid w=10", 10, True),
    ("Wrist w=5", 5, False),
]:
    angles_pred = []
    targets_angle = []

    for i in range(n_shots):
        rf = release_frames[i]
        if rf + window >= n_frames:
            continue

        times = np.arange(window) * dt

        if use_centroid:
            src_x = centroid['x'][i]
            src_y = centroid['y'][i]
            src_z = centroid['z'][i]
        else:
            src_x = data['right_wrist_x'][i]
            src_y = data['right_wrist_y'][i]
            src_z = data['right_wrist_z'][i]

        # Fit ballistic
        z_vals = src_z[rf:rf+window]
        r2_z, vz, g_fit, _ = fit_ballistic(times, z_vals)

        x_vals = src_x[rf:rf+window]
        _, vx = fit_linear(times, x_vals)

        y_vals = src_y[rf:rf+window]
        _, vy = fit_linear(times, y_vals)

        x0, y0, z0 = src_x[rf], src_y[rf], src_z[rf]

        result = forward_simulate(x0, y0, z0, vx, vy, vz)
        if result[0] is not None:
            angles_pred.append(result[0])
            targets_angle.append(train.iloc[i]['angle'])

    if len(angles_pred) > 10:
        angles_pred = np.array(angles_pred)
        targets_angle = np.array(targets_angle)

        # Scale angle predictions using scaler
        angle_scaler = scalers['angle']
        targets_scaled = angle_scaler.transform(targets_angle.reshape(-1, 1)).ravel()

        # Normalize predicted angles to [0,1] range using same scaler
        angles_scaled = angle_scaler.transform(angles_pred.reshape(-1, 1)).ravel()

        mask = ~(np.isnan(angles_scaled) | np.isnan(targets_scaled))
        if mask.sum() > 10:
            mse = np.mean((angles_scaled[mask] - targets_scaled[mask])**2)
            r, _ = pearsonr(angles_pred[mask], targets_angle[mask])
            print(f"  {approach_name:20s}: n={mask.sum()}, angle r={r:.4f}, "
                  f"angle MSE(scaled)={mse:.6f}, pred range=[{angles_pred[mask].min():.1f}, {angles_pred[mask].max():.1f}]")

# ============================================================
# 11. Fingertip-wrist divergence as release signal
# ============================================================
print("\n" + "=" * 70)
print("FINGERTIP-WRIST DIVERGENCE AS RELEASE SIGNAL")
print("=" * 70)

# After release: fingertips should diverge from wrist if they tracked the ball
# Compute distance from fingertip centroid to wrist
fw_dist = np.zeros((n_shots, n_frames))
for i in range(n_shots):
    cx = centroid['x'][i]
    cy = centroid['y'][i]
    cz = centroid['z'][i]
    wx = data['right_wrist_x'][i]
    wy = data['right_wrist_y'][i]
    wz = data['right_wrist_z'][i]
    fw_dist[i] = np.sqrt((cx - wx)**2 + (cy - wy)**2 + (cz - wz)**2)

print("Fingertip centroid - wrist distance around release:")
for offset in range(-10, 21, 2):
    vals = []
    for i in range(n_shots):
        f = release_frames[i] + offset
        if 0 <= f < n_frames:
            vals.append(fw_dist[i, f])
    vals = np.array(vals)
    marker = " <---" if offset == 0 else ""
    print(f"  Release{offset:+3d}: dist={np.mean(vals):.4f} +/- {np.std(vals):.4f}{marker}")

# Rate of divergence
print("\nDivergence rate (distance derivative):")
for offset in range(-5, 16):
    vals = []
    for i in range(n_shots):
        f = release_frames[i] + offset
        if 1 <= f < n_frames:
            vals.append(fw_dist[i, f] - fw_dist[i, f-1])
    vals = np.array(vals)
    if offset in [-5, -3, -1, 0, 1, 2, 3, 5, 7, 10, 15]:
        marker = " <---" if offset == 0 else ""
        print(f"  Release{offset:+3d}: d_dist={np.mean(vals):+.5f} +/- {np.std(vals):.5f}{marker}")

# ============================================================
# 12. Ballistic velocity as features for prediction
# ============================================================
print("\n" + "=" * 70)
print("BALLISTIC VELOCITY FEATURES FOR PREDICTION")
print("=" * 70)

# Extract ballistic-fit features for different windows
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

targets = {
    'angle': scalers['angle'].transform(train['angle'].values.reshape(-1, 1)).ravel(),
    'depth': scalers['depth'].transform(train['depth'].values.reshape(-1, 1)).ravel(),
    'left_right': scalers['left_right'].transform(train['left_right'].values.reshape(-1, 1)).ravel(),
}

for window in [3, 5, 7]:
    # Build features: ballistic velocity (vx, vy, vz) + position at release + g_fit + R2
    features = np.zeros((n_shots, 10))
    valid = np.ones(n_shots, dtype=bool)

    for i in range(n_shots):
        rf = release_frames[i]
        if rf + window >= n_frames or rf < 2:
            valid[i] = False
            continue

        times = np.arange(window) * dt

        z_vals = centroid['z'][i][rf:rf+window]
        r2_z, vz, g_fit, _ = fit_ballistic(times, z_vals)

        x_vals = centroid['x'][i][rf:rf+window]
        r2_x, vx = fit_linear(times, x_vals)

        y_vals = centroid['y'][i][rf:rf+window]
        r2_y, vy = fit_linear(times, y_vals)

        features[i] = [
            vx, vy, vz,
            centroid['x'][i][rf], centroid['y'][i][rf], centroid['z'][i][rf],
            g_fit, r2_z,
            # Also finite-diff wrist velocity for comparison
            (data['right_wrist_y'][i][rf+1] - data['right_wrist_y'][i][rf-1]) / (2*dt),
            (data['right_wrist_z'][i][rf+1] - data['right_wrist_z'][i][rf-1]) / (2*dt),
        ]

    X = features[valid]

    print(f"\nWindow={window} (n_valid={valid.sum()}):")
    for tname in ['angle', 'depth', 'left_right']:
        y = targets[tname][valid]

        loo = LeaveOneOut()
        preds = np.zeros(len(y))
        for train_idx, test_idx in loo.split(X):
            model = Ridge(alpha=10)
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx])

        mse = np.mean((preds - y)**2)
        r, _ = pearsonr(preds, y)
        print(f"  {tname:12s}: LOO MSE={mse:.6f}, r={r:.4f}")

# ============================================================
# 13. Combined features: ballistic + static pose
# ============================================================
print("\n" + "=" * 70)
print("COMBINED: BALLISTIC + STATIC POSE FEATURES")
print("=" * 70)

# Static pose features at target-specific frames (from memory: angle=153, depth=150, LR=170)
target_frames = {'angle': 153, 'depth': 150, 'left_right': 170}

# Get all 207 keypoint features at target-specific frames
all_kp_names = []
for c in train.columns:
    if c.endswith('_x'):
        base = c[:-2]
        if f'{base}_y' in train.columns and f'{base}_z' in train.columns:
            all_kp_names.append(base)

print(f"Found {len(all_kp_names)} keypoints")

for tname in ['angle', 'depth', 'left_right']:
    frame = target_frames[tname]

    # Static features at target frame
    static_feats = np.zeros((n_shots, len(all_kp_names) * 3))
    for j, kp in enumerate(all_kp_names):
        for k, coord in enumerate(['x', 'y', 'z']):
            col = f'{kp}_{coord}'
            for i in range(n_shots):
                val = train.iloc[i][col]
                if isinstance(val, str):
                    arr = safe_parse(val)
                    static_feats[i, j*3+k] = arr[frame] if frame < len(arr) else np.nan
                else:
                    static_feats[i, j*3+k] = val

    # Ballistic features (window=5)
    window = 5
    bal_feats = np.zeros((n_shots, 8))
    for i in range(n_shots):
        rf = release_frames[i]
        if rf + window >= n_frames or rf < 2:
            continue
        times = np.arange(window) * dt
        z_vals = centroid['z'][i][rf:rf+window]
        r2_z, vz, g_fit, _ = fit_ballistic(times, z_vals)
        x_vals = centroid['x'][i][rf:rf+window]
        _, vx = fit_linear(times, x_vals)
        y_vals = centroid['y'][i][rf:rf+window]
        _, vy = fit_linear(times, y_vals)
        bal_feats[i] = [vx, vy, vz, g_fit, r2_z,
                        centroid['x'][i][rf], centroid['y'][i][rf], centroid['z'][i][rf]]

    # Static only
    y = targets[tname]

    # LOO CV with Ridge
    loo = LeaveOneOut()

    # Static only
    preds_static = np.zeros(n_shots)
    for train_idx, test_idx in loo.split(static_feats):
        model = Ridge(alpha=10)
        model.fit(static_feats[train_idx], y[train_idx])
        preds_static[test_idx] = model.predict(static_feats[test_idx])
    mse_static = np.mean((preds_static - y)**2)

    # Combined
    combined = np.hstack([static_feats, bal_feats])
    preds_combined = np.zeros(n_shots)
    for train_idx, test_idx in loo.split(combined):
        model = Ridge(alpha=10)
        model.fit(combined[train_idx], y[train_idx])
        preds_combined[test_idx] = model.predict(combined[test_idx])
    mse_combined = np.mean((preds_combined - y)**2)

    improvement = (mse_static - mse_combined) / mse_static * 100
    print(f"  {tname:12s}: Static MSE={mse_static:.6f}, Combined MSE={mse_combined:.6f}, "
          f"Improvement={improvement:+.1f}%")

# ============================================================
# 14. Correlation with Sub 784 predictions
# ============================================================
print("\n" + "=" * 70)
print("CORRELATION OF BALLISTIC FEATURES WITH SUB 784")
print("=" * 70)

sub784 = pd.read_csv(f'{BASE}/submission/submission_784.csv')
sub1350 = pd.read_csv(f'{BASE}/submission/submission_1350.csv')

print(f"Sub 784 shape: {sub784.shape}")
print(f"Sub 784 columns: {list(sub784.columns)}")

# We can only correlate ballistic predictions on TRAIN data with train targets
# For diversity assessment, correlate LOO predictions with each other
for tname in ['angle', 'depth', 'left_right']:
    r_static, _ = pearsonr(preds_static, y)
    r_combined, _ = pearsonr(preds_combined, y)
    r_static_combined, _ = pearsonr(preds_static, preds_combined)
    print(f"  {tname:12s}: static r={r_static:.4f}, combined r={r_combined:.4f}, "
          f"static-combined r={r_static_combined:.4f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
