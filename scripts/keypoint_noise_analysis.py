"""
Keypoint Noise Floor Analysis
Quantifies measurement noise in the 69-keypoint motion capture data.
Methods: bone length consistency, static frame jitter, velocity noise, FFT analysis.
"""
import numpy as np
import pandas as pd
import ast
import sys

BASE = "/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge"

# ---- Load data ----
print("Loading data...")
df = pd.read_csv(f"{BASE}/data/train.csv")
print(f"Loaded {len(df)} shots")

# Get all keypoint names (unique prefixes before _x, _y, _z)
cols = df.columns.tolist()
kp_names = []
for c in cols:
    if c.endswith('_x'):
        prefix = c[:-2]
        if f"{prefix}_y" in cols and f"{prefix}_z" in cols:
            kp_names.append(prefix)
print(f"Found {len(kp_names)} keypoints")

# Parse list strings into numpy arrays
# Shape per shot: (240, 69, 3)
print("Parsing trajectories...")
n_shots = len(df)
n_frames = 240
n_kp = len(kp_names)

def parse_list(s):
    """Parse string list, handling nan values."""
    s = s.replace('nan', 'np.nan').replace('None', 'np.nan')
    return np.array(eval(s, {"np": np, "__builtins__": {}}))

all_data = np.full((n_shots, n_frames, n_kp, 3), np.nan)
for i, row in df.iterrows():
    for j, kp in enumerate(kp_names):
        for k, axis in enumerate(['x', 'y', 'z']):
            col = f"{kp}_{axis}"
            vals = parse_list(str(row[col]))
            all_data[i, :len(vals), j, k] = vals[:n_frames]
    if (i + 1) % 50 == 0:
        print(f"  Parsed {i+1}/{n_shots} shots...")

print(f"Data shape: {all_data.shape}")

# Check units - what scale are the coordinates?
print("\n" + "="*80)
print("COORDINATE SCALE CHECK")
print("="*80)
# Sample a few keypoints to understand the scale
for kp_idx, kp_name in enumerate(kp_names[:5]):
    vals = all_data[:, 120, kp_idx, :]  # frame 120, all shots
    print(f"  {kp_name}: x=[{vals[:,0].min():.3f}, {vals[:,0].max():.3f}], "
          f"y=[{vals[:,1].min():.3f}, {vals[:,1].max():.3f}], "
          f"z=[{vals[:,2].min():.3f}, {vals[:,2].max():.3f}]")

# Also check right_shoulder and right_wrist to gauge body scale
rs_idx = kp_names.index('right_shoulder')
rw_idx = kp_names.index('right_wrist')
sample_arm = np.sqrt(np.sum((all_data[0, 120, rs_idx, :] - all_data[0, 120, rw_idx, :])**2))
print(f"\n  Sample shoulder-to-wrist distance: {sample_arm:.4f}")
print(f"  (If ~0.5-0.7, units are likely meters; if ~50-70, centimeters; if ~500-700, millimeters)")

# ====================================================================
# 1. BONE LENGTH CONSISTENCY
# ====================================================================
print("\n" + "="*80)
print("1. BONE LENGTH CONSISTENCY")
print("="*80)

bone_pairs = [
    ('right_shoulder', 'right_elbow', 'R upper arm'),
    ('right_elbow', 'right_wrist', 'R forearm'),
    ('right_hip', 'right_knee', 'R thigh'),
    ('right_knee', 'right_ankle', 'R shin'),
    ('left_shoulder', 'left_elbow', 'L upper arm'),
    ('left_elbow', 'left_wrist', 'L forearm'),
    ('left_hip', 'left_knee', 'L thigh'),
    ('left_knee', 'left_ankle', 'L shin'),
    ('right_shoulder', 'right_hip', 'R torso'),
    ('left_shoulder', 'left_hip', 'L torso'),
    ('right_wrist', 'right_second_finger_mcp', 'R hand'),
    ('left_wrist', 'left_second_finger_mcp', 'L hand'),
]

print(f"\n{'Bone':<20} {'Mean Len':>10} {'Mean Std':>10} {'CV (%)':>10} {'Noise Est':>12}")
print("-" * 65)

bone_results = {}
for kp1, kp2, name in bone_pairs:
    if kp1 not in kp_names or kp2 not in kp_names:
        continue
    idx1 = kp_names.index(kp1)
    idx2 = kp_names.index(kp2)

    # Bone length for each frame of each shot
    diff = all_data[:, :, idx1, :] - all_data[:, :, idx2, :]  # (n_shots, 240, 3)
    bone_len = np.sqrt(np.sum(diff**2, axis=2))  # (n_shots, 240)

    # Per-shot: mean and std of bone length across frames
    per_shot_mean = np.mean(bone_len, axis=1)  # (n_shots,)
    per_shot_std = np.std(bone_len, axis=1)     # (n_shots,)
    per_shot_cv = per_shot_std / per_shot_mean * 100

    avg_mean = np.mean(per_shot_mean)
    avg_std = np.mean(per_shot_std)
    avg_cv = np.mean(per_shot_cv)

    # Noise estimate: bone std is caused by noise on BOTH endpoints
    # If each endpoint has noise sigma, bone length noise ~ sqrt(2) * sigma (in 1D projection)
    # More precisely, for 3D noise sigma on each endpoint:
    # Var(bone_length) ~ 2 * sigma^2 (to first order)
    # So sigma ~ std / sqrt(2)
    noise_est = avg_std / np.sqrt(2)

    bone_results[name] = {
        'mean_len': avg_mean, 'mean_std': avg_std,
        'cv': avg_cv, 'noise_est': noise_est
    }
    print(f"{name:<20} {avg_mean:10.5f} {avg_std:10.5f} {avg_cv:10.2f} {noise_est:12.5f}")

# Per-player bone analysis
print("\n  Per-player bone length CV (%) for R upper arm:")
for pid in sorted(df['participant_id'].unique()):
    mask = df['participant_id'].values == pid
    idx1 = kp_names.index('right_shoulder')
    idx2 = kp_names.index('right_elbow')
    diff = all_data[mask, :, idx1, :] - all_data[mask, :, idx2, :]
    bone_len = np.sqrt(np.sum(diff**2, axis=2))
    per_shot_cv = np.std(bone_len, axis=1) / np.mean(bone_len, axis=1) * 100
    print(f"    Player {pid}: CV = {np.mean(per_shot_cv):.2f}% (mean bone len = {np.mean(bone_len):.5f})")

# ====================================================================
# 2. STATIC FRAME ANALYSIS (frames 0-30)
# ====================================================================
print("\n" + "="*80)
print("2. STATIC FRAME JITTER (frames 0-30)")
print("="*80)

static_frames = all_data[:, 0:31, :, :]  # (n_shots, 31, n_kp, 3)
# Jitter = std of position across static frames, per axis
jitter_per_axis = np.std(static_frames, axis=1)  # (n_shots, n_kp, 3)
jitter_3d = np.sqrt(np.sum(jitter_per_axis**2, axis=2))  # (n_shots, n_kp) - 3D jitter magnitude

# Average across shots
mean_jitter_3d = np.mean(jitter_3d, axis=0)  # (n_kp,)
mean_jitter_per_axis = np.mean(jitter_per_axis, axis=0)  # (n_kp, 3)

# Sort keypoints by jitter
sorted_idx = np.argsort(mean_jitter_3d)

print(f"\n  LOWEST jitter (most stable) keypoints:")
print(f"  {'Keypoint':<40} {'3D Jitter':>12} {'X Jitter':>10} {'Y Jitter':>10} {'Z Jitter':>10}")
for i in sorted_idx[:10]:
    print(f"  {kp_names[i]:<40} {mean_jitter_3d[i]:12.6f} {mean_jitter_per_axis[i,0]:10.6f} "
          f"{mean_jitter_per_axis[i,1]:10.6f} {mean_jitter_per_axis[i,2]:10.6f}")

print(f"\n  HIGHEST jitter (noisiest) keypoints:")
for i in sorted_idx[-10:][::-1]:
    print(f"  {kp_names[i]:<40} {mean_jitter_3d[i]:12.6f} {mean_jitter_per_axis[i,0]:10.6f} "
          f"{mean_jitter_per_axis[i,1]:10.6f} {mean_jitter_per_axis[i,2]:10.6f}")

# Category summary
body_kps = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
foot_kps = [k for k in kp_names if 'toe' in k or 'heel' in k]
hand_kps = [k for k in kp_names if 'finger' in k or 'thumb' in k or 'pinky' in k]

print("\n  Category averages (3D jitter):")
for cat_name, cat_kps in [('Body (17)', body_kps), ('Feet (6)', foot_kps), ('Hands (rest)', hand_kps)]:
    cat_idx = [kp_names.index(k) for k in cat_kps if k in kp_names]
    if cat_idx:
        cat_jitter = mean_jitter_3d[cat_idx]
        print(f"    {cat_name}: mean={np.mean(cat_jitter):.6f}, "
              f"min={np.min(cat_jitter):.6f}, max={np.max(cat_jitter):.6f}")

# ====================================================================
# 3. VELOCITY NOISE (static frames)
# ====================================================================
print("\n" + "="*80)
print("3. VELOCITY NOISE FLOOR (frames 0-30)")
print("="*80)

dt = 1.0 / 60.0  # 60 fps
# Finite difference velocity: v[t] = (x[t+1] - x[t]) / dt
static_pos = all_data[:, 0:31, :, :]  # (n_shots, 31, n_kp, 3)
static_vel = np.diff(static_pos, axis=1) / dt  # (n_shots, 30, n_kp, 3)

# Velocity magnitude
vel_mag = np.sqrt(np.sum(static_vel**2, axis=3))  # (n_shots, 30, n_kp)

# Mean velocity magnitude (should be ~0 for truly static, but noise creates non-zero)
mean_vel_mag = np.mean(vel_mag, axis=(0, 1))  # (n_kp,)

# Also compute velocity std per axis
vel_std_per_axis = np.std(static_vel, axis=1)  # (n_shots, n_kp, 3) - std across time per shot
mean_vel_std = np.mean(vel_std_per_axis, axis=0)  # (n_kp, 3)

print(f"\n  Key joints velocity noise (static frames):")
print(f"  {'Keypoint':<35} {'Mean |v|':>10} {'Std vx':>10} {'Std vy':>10} {'Std vz':>10} {'unit':>6}")
key_joints = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip',
              'right_knee', 'right_ankle', 'nose', 'left_wrist',
              'right_second_finger_mcp', 'right_first_finger_distal']
for kp in key_joints:
    if kp in kp_names:
        idx = kp_names.index(kp)
        print(f"  {kp:<35} {mean_vel_mag[idx]:10.4f} {mean_vel_std[idx,0]:10.4f} "
              f"{mean_vel_std[idx,1]:10.4f} {mean_vel_std[idx,2]:10.4f} {'u/s':>6}")

# Also check if frames 0-30 are truly "static" by looking at the actual motion
# Compare position range in frames 0-30 vs frames 100-150 (shooting motion)
print("\n  Position range comparison (static vs shooting):")
for kp in ['right_wrist', 'right_elbow', 'right_shoulder']:
    if kp in kp_names:
        idx = kp_names.index(kp)
        static_range = np.mean(np.ptp(all_data[:, 0:31, idx, :], axis=1), axis=0)
        shoot_range = np.mean(np.ptp(all_data[:, 100:151, idx, :], axis=1), axis=0)
        print(f"  {kp}: static range=[{static_range[0]:.4f}, {static_range[1]:.4f}, {static_range[2]:.4f}], "
              f"shooting range=[{shoot_range[0]:.4f}, {shoot_range[1]:.4f}, {shoot_range[2]:.4f}]")
        print(f"    Ratio (shooting/static): [{shoot_range[0]/max(static_range[0],1e-8):.1f}x, "
              f"{shoot_range[1]/max(static_range[1],1e-8):.1f}x, {shoot_range[2]/max(static_range[2],1e-8):.1f}x]")

# ====================================================================
# 4. ACCELERATION NOISE
# ====================================================================
print("\n" + "="*80)
print("4. ACCELERATION NOISE (static frames)")
print("="*80)

static_acc = np.diff(static_vel, axis=1) / dt  # (n_shots, 29, n_kp, 3)
acc_mag = np.sqrt(np.sum(static_acc**2, axis=3))  # (n_shots, 29, n_kp)
mean_acc = np.mean(acc_mag, axis=(0,1))  # (n_kp,)

print(f"\n  Key joints acceleration noise (static frames):")
print(f"  {'Keypoint':<35} {'Mean |a|':>12} {'unit':>8}")
for kp in key_joints:
    if kp in kp_names:
        idx = kp_names.index(kp)
        print(f"  {kp:<35} {mean_acc[idx]:12.2f} {'u/s^2':>8}")

# ====================================================================
# 5. FREQUENCY ANALYSIS (FFT)
# ====================================================================
print("\n" + "="*80)
print("5. FREQUENCY ANALYSIS (FFT)")
print("="*80)

# FFT of right_wrist trajectory for a representative shot
rw_idx = kp_names.index('right_wrist')
freqs = np.fft.rfftfreq(n_frames, d=dt)  # frequency bins

# Compute power spectrum for all shots, right_wrist, y-axis (main shooting direction)
print("\n  Power spectrum analysis for right_wrist (y-axis, main shooting direction):")
power_all = np.zeros((n_shots, len(freqs)))
for i in range(n_shots):
    signal = all_data[i, :, rw_idx, 1]  # y-axis
    signal_detrended = signal - np.mean(signal)
    fft_vals = np.fft.rfft(signal_detrended)
    power_all[i, :] = np.abs(fft_vals)**2

mean_power = np.mean(power_all, axis=0)
total_power = np.sum(mean_power)

# Power in frequency bands
bands = [(0, 2, 'DC-2Hz (slow motion)'), (2, 5, '2-5Hz (shooting motion)'),
         (5, 10, '5-10Hz (fast motion)'), (10, 20, '10-20Hz (possible noise)'),
         (20, 30, '20-30Hz (near Nyquist, likely noise)')]

print(f"  Total power: {total_power:.2f}")
for lo, hi, label in bands:
    mask = (freqs >= lo) & (freqs < hi)
    band_power = np.sum(mean_power[mask])
    pct = band_power / total_power * 100
    print(f"    {label:<40} {pct:6.2f}% of total power")

# SNR estimate: signal is < 10Hz, noise is > 10Hz
signal_power = np.sum(mean_power[freqs < 10])
noise_power = np.sum(mean_power[freqs >= 10])
snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
print(f"\n  Estimated SNR (signal < 10Hz, noise >= 10Hz): {snr:.1f} dB")

# Repeat for multiple joints
print("\n  SNR by joint (signal < 10Hz vs noise >= 10Hz):")
print(f"  {'Keypoint':<35} {'SNR (dB)':>10} {'Noise %':>10}")
for kp in ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip',
           'right_knee', 'right_ankle', 'right_second_finger_mcp', 'nose']:
    if kp not in kp_names:
        continue
    idx = kp_names.index(kp)
    snr_vals = []
    noise_pcts = []
    for axis in range(3):
        power_ax = np.zeros(len(freqs))
        for i in range(n_shots):
            sig = all_data[i, :, idx, axis] - np.mean(all_data[i, :, idx, axis])
            fft_v = np.fft.rfft(sig)
            power_ax += np.abs(fft_v)**2
        power_ax /= n_shots
        sp = np.sum(power_ax[freqs < 10])
        np_ = np.sum(power_ax[freqs >= 10])
        snr_vals.append(10 * np.log10(sp / max(np_, 1e-10)))
        noise_pcts.append(np_ / (sp + np_) * 100)
    print(f"  {kp:<35} {np.mean(snr_vals):10.1f} {np.mean(noise_pcts):10.1f}%")

# ====================================================================
# 6. INTER-SHOT BONE LENGTH CONSISTENCY (same player)
# ====================================================================
print("\n" + "="*80)
print("6. INTER-SHOT BONE LENGTH CONSISTENCY (per player)")
print("="*80)

# For rigid bones, the MEAN bone length should be the same across shots for the same player
# Variation = measurement error + soft tissue deformation
print(f"\n  {'Bone':<20} {'Player':>8} {'Mean':>10} {'Inter-shot Std':>15} {'Inter-shot CV%':>15}")
for kp1, kp2, name in [('right_shoulder', 'right_elbow', 'R upper arm'),
                         ('right_elbow', 'right_wrist', 'R forearm'),
                         ('right_hip', 'right_knee', 'R thigh')]:
    idx1 = kp_names.index(kp1)
    idx2 = kp_names.index(kp2)
    diff = all_data[:, :, idx1, :] - all_data[:, :, idx2, :]
    bone_len = np.sqrt(np.sum(diff**2, axis=2))
    mean_bone_per_shot = np.mean(bone_len, axis=1)  # (n_shots,)

    for pid in sorted(df['participant_id'].unique()):
        mask = df['participant_id'].values == pid
        player_means = mean_bone_per_shot[mask]
        inter_std = np.std(player_means)
        inter_mean = np.mean(player_means)
        inter_cv = inter_std / inter_mean * 100
        print(f"  {name:<20} {pid:>8} {inter_mean:10.5f} {inter_std:15.5f} {inter_cv:15.2f}%")

# ====================================================================
# 7. FRAME-TO-FRAME POSITION CHANGE IN STATIC PERIOD
# ====================================================================
print("\n" + "="*80)
print("7. FRAME-TO-FRAME POSITION JUMP STATISTICS (static frames 0-30)")
print("="*80)

# This is the most direct noise estimate
static_diff = np.diff(all_data[:, 0:31, :, :], axis=1)  # (n_shots, 30, n_kp, 3)
frame_jumps = np.sqrt(np.sum(static_diff**2, axis=3))  # (n_shots, 30, n_kp) - 3D jump per frame

mean_jump = np.mean(frame_jumps, axis=(0,1))  # (n_kp,)
median_jump = np.median(frame_jumps.reshape(-1, n_kp), axis=0)
p95_jump = np.percentile(frame_jumps.reshape(-1, n_kp), 95, axis=0)
max_jump = np.max(frame_jumps.reshape(-1, n_kp), axis=0)

print(f"\n  {'Keypoint':<40} {'Mean':>10} {'Median':>10} {'P95':>10} {'Max':>10}")
key_kps = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip',
           'right_knee', 'right_ankle', 'nose', 'right_second_finger_mcp',
           'right_first_finger_distal', 'left_wrist']
for kp in key_kps:
    if kp not in kp_names:
        continue
    idx = kp_names.index(kp)
    print(f"  {kp:<40} {mean_jump[idx]:10.5f} {median_jump[idx]:10.5f} "
          f"{p95_jump[idx]:10.5f} {max_jump[idx]:10.5f}")

# Overall stats
print(f"\n  Overall (all keypoints):")
print(f"    Mean frame-to-frame jump: {np.mean(mean_jump):.6f}")
print(f"    Body joints mean jump: {np.mean([mean_jump[kp_names.index(k)] for k in body_kps if k in kp_names]):.6f}")
print(f"    Hand joints mean jump: {np.mean([mean_jump[kp_names.index(k)] for k in hand_kps if k in kp_names]):.6f}")

# Noise estimate: if frame-to-frame jump in static period is all noise,
# then position noise sigma = jump_rms / sqrt(2) (since diff of two noisy measurements)
rms_jump = np.sqrt(np.mean(frame_jumps**2, axis=(0,1)))  # (n_kp,)
sigma_pos = rms_jump / np.sqrt(2)  # position noise per keypoint per axis (approximate)

print(f"\n  Estimated position noise (sigma = rms_jump / sqrt(2)):")
print(f"  {'Keypoint':<40} {'Sigma (pos)':>12} {'Sigma (vel)':>12}")
for kp in key_kps:
    if kp not in kp_names:
        continue
    idx = kp_names.index(kp)
    vel_sigma = sigma_pos[idx] * np.sqrt(2) / dt  # velocity noise from position noise
    print(f"  {kp:<40} {sigma_pos[idx]:12.6f} {vel_sigma:12.4f}")

# ====================================================================
# 8. SUMMARY
# ====================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Determine units from context
rs_idx = kp_names.index('right_shoulder')
re_idx = kp_names.index('right_elbow')
upper_arm = np.mean(np.sqrt(np.sum((all_data[:, 120, rs_idx, :] - all_data[:, 120, re_idx, :])**2, axis=1)))
print(f"\n  Average upper arm length: {upper_arm:.4f}")
print(f"  Typical human upper arm: ~30cm = 0.30m")
print(f"  Scale factor estimate: {0.30 / upper_arm:.2f} (multiply data units to get meters)")
if upper_arm > 0.1 and upper_arm < 1.0:
    unit_name = "meters"
    scale = 1.0
elif upper_arm > 10 and upper_arm < 100:
    unit_name = "centimeters"
    scale = 0.01
elif upper_arm > 100:
    unit_name = "millimeters"
    scale = 0.001
else:
    unit_name = "unknown (possibly feet)"
    scale = 0.3048  # feet to meters

body_sigma = np.mean([sigma_pos[kp_names.index(k)] for k in body_kps if k in kp_names])
hand_sigma = np.mean([sigma_pos[kp_names.index(k)] for k in hand_kps if k in kp_names])

print(f"\n  Data units appear to be: {unit_name}")
print(f"  Position noise (body joints):  {body_sigma:.6f} {unit_name} = {body_sigma*scale*100:.2f} cm = {body_sigma*scale*1000:.1f} mm")
print(f"  Position noise (hand joints):  {hand_sigma:.6f} {unit_name} = {hand_sigma*scale*100:.2f} cm = {hand_sigma*scale*1000:.1f} mm")
print(f"  Velocity noise (body, from position): {body_sigma * np.sqrt(2) / dt * scale:.4f} m/s")
print(f"  Velocity noise (hand, from position): {hand_sigma * np.sqrt(2) / dt * scale:.4f} m/s")

# Bone length CV summary
bone_cvs = [bone_results[name]['cv'] for name in bone_results]
print(f"\n  Bone length CV: mean={np.mean(bone_cvs):.2f}%, range=[{np.min(bone_cvs):.2f}%, {np.max(bone_cvs):.2f}%]")

# Critical question: is noise small enough for physics?
rw_sigma = sigma_pos[kp_names.index('right_wrist')]
rw_vel_noise = rw_sigma * np.sqrt(2) / dt * scale
print(f"\n  RIGHT WRIST (critical for release velocity):")
print(f"    Position noise: {rw_sigma:.6f} {unit_name} = {rw_sigma*scale*1000:.1f} mm")
print(f"    Velocity noise: {rw_vel_noise:.4f} m/s")
print(f"    Typical release velocity: ~7 m/s")
print(f"    Velocity SNR: {7.0 / rw_vel_noise:.1f}x")

# Check if frames 0-30 are really static or if there's real motion
# by looking at whether the motion is correlated across shots
print(f"\n  Are frames 0-30 truly static?")
rw_idx = kp_names.index('right_wrist')
static_y = all_data[:, 0:31, rw_idx, 1]  # (n_shots, 31) - y axis
# Motion: difference between frame 0 and frame 30
drift_y = static_y[:, -1] - static_y[:, 0]
print(f"    Right wrist Y drift (frame 0 to 30): mean={np.mean(drift_y):.5f}, std={np.std(drift_y):.5f}")
print(f"    (If std >> noise, there IS real motion in frames 0-30)")

# Also check truly static joints (like ankles)
ra_idx = kp_names.index('right_ankle')
static_ankle = all_data[:, 0:31, ra_idx, :]
ankle_drift = static_ankle[:, -1, :] - static_ankle[:, 0, :]
print(f"    Right ankle drift: mean_3d={np.mean(np.sqrt(np.sum(ankle_drift**2, axis=1))):.5f}")
print(f"    (Ankle should be very stable - this is closest to pure noise)")

# Better noise estimate: use ankle as ground truth noise reference
ankle_sigma = sigma_pos[kp_names.index('right_ankle')]
print(f"\n  BEST NOISE ESTIMATE (from ankle, most static joint):")
print(f"    Position: {ankle_sigma:.6f} {unit_name} = {ankle_sigma*scale*1000:.1f} mm")
print(f"    Velocity: {ankle_sigma * np.sqrt(2) / dt * scale:.4f} m/s")
print(f"    Acceleration: {ankle_sigma * 2 / dt**2 * scale:.2f} m/s^2")

print("\nDone!")
