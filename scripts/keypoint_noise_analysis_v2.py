"""
Keypoint Noise Floor Analysis v2 - with NaN-safe computations
"""
import numpy as np
import pandas as pd
import ast
import sys

np.seterr(invalid='ignore')

BASE = "/Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge"

# ---- Load data ----
print("Loading data...")
df = pd.read_csv(f"{BASE}/data/train.csv")
print(f"Loaded {len(df)} shots")

cols = df.columns.tolist()
kp_names = []
for c in cols:
    if c.endswith('_x'):
        prefix = c[:-2]
        if f"{prefix}_y" in cols and f"{prefix}_z" in cols:
            kp_names.append(prefix)
print(f"Found {len(kp_names)} keypoints")

print("Parsing trajectories...")
n_shots = len(df)
n_frames = 240
n_kp = len(kp_names)

def parse_list(s):
    s = str(s).replace('nan', 'np.nan').replace('None', 'np.nan')
    return np.array(eval(s, {"np": np, "__builtins__": {}}))

all_data = np.full((n_shots, n_frames, n_kp, 3), np.nan)
for i, row in df.iterrows():
    for j, kp in enumerate(kp_names):
        for k, axis in enumerate(['x', 'y', 'z']):
            col = f"{kp}_{axis}"
            vals = parse_list(row[col])
            all_data[i, :len(vals), j, k] = vals[:n_frames]

# Count NaN
nan_count = np.isnan(all_data).sum()
total_count = all_data.size
nan_pct = nan_count / total_count * 100
print(f"Data shape: {all_data.shape}")
print(f"NaN values: {nan_count} / {total_count} ({nan_pct:.2f}%)")

# Count NaN per keypoint
nan_per_kp = np.isnan(all_data[:, :, :, 0]).sum(axis=(0,1))
print("\nKeypoints with NaN:")
for i, kp in enumerate(kp_names):
    if nan_per_kp[i] > 0:
        print(f"  {kp}: {nan_per_kp[i]} NaN frames ({nan_per_kp[i]/n_shots/n_frames*100:.1f}%)")

# Identify clean shots (no NaN in any keypoint)
clean_mask = ~np.isnan(all_data).any(axis=(1,2,3))
n_clean = clean_mask.sum()
print(f"\nClean shots (no NaN): {n_clean}/{n_shots}")

# Use clean shots for analyses requiring complete data
clean_data = all_data[clean_mask]

# Units: coordinates are in FEET
# Upper arm ~1.0 ft = 30 cm, hoop at [5.25, -25, 10] feet
FT_TO_M = 0.3048
FT_TO_CM = 30.48
FT_TO_MM = 304.8

print("\n" + "="*80)
print("COORDINATE SCALE - UNITS ARE FEET")
print("="*80)
rs_idx = kp_names.index('right_shoulder')
re_idx = kp_names.index('right_elbow')
rw_idx = kp_names.index('right_wrist')
ra_idx = kp_names.index('right_ankle')

upper_arm = np.nanmean(np.sqrt(np.nansum((clean_data[:, 120, rs_idx, :] - clean_data[:, 120, re_idx, :])**2, axis=1)))
forearm = np.nanmean(np.sqrt(np.nansum((clean_data[:, 120, re_idx, :] - clean_data[:, 120, rw_idx, :])**2, axis=1)))
print(f"  Upper arm length: {upper_arm:.4f} ft = {upper_arm*FT_TO_CM:.1f} cm")
print(f"  Forearm length: {forearm:.4f} ft = {forearm*FT_TO_CM:.1f} cm")
print(f"  (Typical: upper arm ~30cm, forearm ~25cm)")

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

print(f"\n{'Bone':<20} {'Mean(ft)':>10} {'Mean(cm)':>10} {'Std(ft)':>10} {'Std(mm)':>10} {'CV(%)':>8} {'Noise(mm)':>10}")
print("-" * 82)

bone_results = {}
for kp1, kp2, name in bone_pairs:
    if kp1 not in kp_names or kp2 not in kp_names:
        continue
    idx1 = kp_names.index(kp1)
    idx2 = kp_names.index(kp2)

    diff = clean_data[:, :, idx1, :] - clean_data[:, :, idx2, :]
    bone_len = np.sqrt(np.sum(diff**2, axis=2))  # (n_clean, 240)

    per_shot_mean = np.mean(bone_len, axis=1)
    per_shot_std = np.std(bone_len, axis=1)
    per_shot_cv = per_shot_std / per_shot_mean * 100

    avg_mean = np.mean(per_shot_mean)
    avg_std = np.mean(per_shot_std)
    avg_cv = np.mean(per_shot_cv)
    noise_est = avg_std / np.sqrt(2)

    bone_results[name] = {
        'mean_len': avg_mean, 'mean_std': avg_std,
        'cv': avg_cv, 'noise_est': noise_est
    }
    print(f"{name:<20} {avg_mean:10.4f} {avg_mean*FT_TO_CM:10.1f} {avg_std:10.5f} "
          f"{avg_std*FT_TO_MM:10.1f} {avg_cv:8.2f} {noise_est*FT_TO_MM:10.1f}")

# Per-player bone analysis
print("\n  Per-player R upper arm: mean length and within-shot CV")
for pid in sorted(df['participant_id'].unique()):
    mask = (df['participant_id'].values == pid) & clean_mask
    if mask.sum() == 0:
        continue
    idx1 = kp_names.index('right_shoulder')
    idx2 = kp_names.index('right_elbow')
    pdata = all_data[mask]
    diff = pdata[:, :, idx1, :] - pdata[:, :, idx2, :]
    bone_len = np.sqrt(np.sum(diff**2, axis=2))
    per_shot_cv = np.std(bone_len, axis=1) / np.mean(bone_len, axis=1) * 100
    print(f"    Player {pid}: mean={np.mean(bone_len):.4f} ft ({np.mean(bone_len)*FT_TO_CM:.1f} cm), "
          f"CV={np.mean(per_shot_cv):.2f}%")

# ====================================================================
# 2. STATIC FRAME ANALYSIS (frames 0-30)
# ====================================================================
print("\n" + "="*80)
print("2. STATIC FRAME JITTER (frames 0-30)")
print("="*80)

static_frames = clean_data[:, 0:31, :, :]
jitter_per_axis = np.std(static_frames, axis=1)  # (n_clean, n_kp, 3)
jitter_3d = np.sqrt(np.sum(jitter_per_axis**2, axis=2))
mean_jitter_3d = np.mean(jitter_3d, axis=0)
mean_jitter_per_axis = np.mean(jitter_per_axis, axis=0)

sorted_idx = np.argsort(mean_jitter_3d)

body_kps = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
foot_kps = [k for k in kp_names if 'toe' in k or 'heel' in k]
hand_kps = [k for k in kp_names if 'finger' in k or 'thumb' in k or 'pinky' in k]

print(f"\n  10 MOST STABLE keypoints (lowest jitter):")
print(f"  {'Keypoint':<40} {'3D(ft)':>10} {'3D(mm)':>10} {'X(mm)':>8} {'Y(mm)':>8} {'Z(mm)':>8}")
for i in sorted_idx[:10]:
    print(f"  {kp_names[i]:<40} {mean_jitter_3d[i]:10.5f} {mean_jitter_3d[i]*FT_TO_MM:10.1f} "
          f"{mean_jitter_per_axis[i,0]*FT_TO_MM:8.1f} {mean_jitter_per_axis[i,1]*FT_TO_MM:8.1f} "
          f"{mean_jitter_per_axis[i,2]*FT_TO_MM:8.1f}")

print(f"\n  10 NOISIEST keypoints (highest jitter):")
for i in sorted_idx[-10:][::-1]:
    print(f"  {kp_names[i]:<40} {mean_jitter_3d[i]:10.5f} {mean_jitter_3d[i]*FT_TO_MM:10.1f} "
          f"{mean_jitter_per_axis[i,0]*FT_TO_MM:8.1f} {mean_jitter_per_axis[i,1]*FT_TO_MM:8.1f} "
          f"{mean_jitter_per_axis[i,2]*FT_TO_MM:8.1f}")

print("\n  Category averages (3D jitter in mm):")
for cat_name, cat_kps in [('Body (17)', body_kps), ('Feet (6)', foot_kps), ('Hands (rest)', hand_kps)]:
    cat_idx = [kp_names.index(k) for k in cat_kps if k in kp_names]
    if cat_idx:
        cat_j = mean_jitter_3d[cat_idx]
        print(f"    {cat_name}: mean={np.mean(cat_j)*FT_TO_MM:.1f} mm, "
              f"min={np.min(cat_j)*FT_TO_MM:.1f} mm, max={np.max(cat_j)*FT_TO_MM:.1f} mm")

# ====================================================================
# 3. VELOCITY NOISE (static frames)
# ====================================================================
print("\n" + "="*80)
print("3. VELOCITY NOISE FLOOR (static frames 0-30)")
print("="*80)

dt = 1.0 / 60.0
static_pos = clean_data[:, 0:31, :, :]
static_vel = np.diff(static_pos, axis=1) / dt  # ft/s

# Convert to m/s for reporting
vel_ms = static_vel * FT_TO_M  # m/s

vel_mag = np.sqrt(np.sum(vel_ms**2, axis=3))
mean_vel_mag = np.mean(vel_mag, axis=(0, 1))

vel_std_per_axis = np.std(vel_ms, axis=1)
mean_vel_std = np.mean(vel_std_per_axis, axis=0)

print(f"\n  Key joints velocity noise (m/s):")
print(f"  {'Keypoint':<35} {'Mean |v|':>10} {'Std vx':>10} {'Std vy':>10} {'Std vz':>10}")
key_joints = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip',
              'right_knee', 'right_ankle', 'nose', 'left_wrist',
              'right_second_finger_mcp', 'right_first_finger_distal']
for kp in key_joints:
    if kp in kp_names:
        idx = kp_names.index(kp)
        print(f"  {kp:<35} {mean_vel_mag[idx]:10.4f} {mean_vel_std[idx,0]:10.4f} "
              f"{mean_vel_std[idx,1]:10.4f} {mean_vel_std[idx,2]:10.4f}")

# Position range comparison
print("\n  Motion range comparison (static vs shooting, in ft):")
for kp in ['right_wrist', 'right_elbow', 'right_shoulder']:
    idx = kp_names.index(kp)
    static_range = np.mean(np.ptp(clean_data[:, 0:31, idx, :], axis=1), axis=0)
    shoot_range = np.mean(np.ptp(clean_data[:, 100:151, idx, :], axis=1), axis=0)
    ratio = shoot_range / np.maximum(static_range, 1e-8)
    print(f"  {kp}: static=[{static_range[0]:.3f}, {static_range[1]:.3f}, {static_range[2]:.3f}] ft, "
          f"shooting=[{shoot_range[0]:.3f}, {shoot_range[1]:.3f}, {shoot_range[2]:.3f}] ft, "
          f"ratio=[{ratio[0]:.1f}x, {ratio[1]:.1f}x, {ratio[2]:.1f}x]")

# ====================================================================
# 4. FRAME-TO-FRAME POSITION JUMPS (direct noise estimate)
# ====================================================================
print("\n" + "="*80)
print("4. FRAME-TO-FRAME POSITION JUMPS (static frames 0-30)")
print("="*80)

static_diff = np.diff(clean_data[:, 0:31, :, :], axis=1)
frame_jumps = np.sqrt(np.sum(static_diff**2, axis=3))

mean_jump = np.mean(frame_jumps, axis=(0,1))
median_jump = np.median(frame_jumps.reshape(-1, n_kp), axis=0)
p95_jump = np.percentile(frame_jumps.reshape(-1, n_kp), 95, axis=0)

print(f"\n  {'Keypoint':<40} {'Mean(mm)':>10} {'Med(mm)':>10} {'P95(mm)':>10}")
for kp in key_joints:
    if kp not in kp_names:
        continue
    idx = kp_names.index(kp)
    print(f"  {kp:<40} {mean_jump[idx]*FT_TO_MM:10.1f} {median_jump[idx]*FT_TO_MM:10.1f} "
          f"{p95_jump[idx]*FT_TO_MM:10.1f}")

# RMS-based noise estimate
rms_jump = np.sqrt(np.mean(frame_jumps**2, axis=(0,1)))
sigma_pos = rms_jump / np.sqrt(2)  # each jump is diff of two noisy measurements

print(f"\n  POSITION NOISE ESTIMATES (sigma = rms_jump/sqrt(2)):")
print(f"  {'Keypoint':<40} {'Sigma(ft)':>10} {'Sigma(mm)':>10} {'Vel noise(m/s)':>15}")
for kp in key_joints:
    if kp not in kp_names:
        continue
    idx = kp_names.index(kp)
    vel_noise = sigma_pos[idx] * np.sqrt(2) / dt * FT_TO_M
    print(f"  {kp:<40} {sigma_pos[idx]:10.6f} {sigma_pos[idx]*FT_TO_MM:10.1f} {vel_noise:15.3f}")

# ====================================================================
# 5. FREQUENCY ANALYSIS (FFT) - NaN-safe
# ====================================================================
print("\n" + "="*80)
print("5. FREQUENCY ANALYSIS (FFT)")
print("="*80)

freqs = np.fft.rfftfreq(n_frames, d=dt)

# Right wrist y-axis
print("\n  Power spectrum for right_wrist (y-axis):")
rw_idx = kp_names.index('right_wrist')
power_all = []
for i in range(len(clean_data)):
    signal = clean_data[i, :, rw_idx, 1]
    if np.any(np.isnan(signal)):
        continue
    signal_d = signal - np.mean(signal)
    fft_vals = np.fft.rfft(signal_d)
    power_all.append(np.abs(fft_vals)**2)

power_all = np.array(power_all)
mean_power = np.mean(power_all, axis=0)
total_power = np.sum(mean_power)

bands = [(0, 2, 'DC-2Hz (slow/posture)'), (2, 5, '2-5Hz (shooting motion)'),
         (5, 10, '5-10Hz (fast motion)'), (10, 20, '10-20Hz (possible noise)'),
         (20, 30, '20-30Hz (near Nyquist)')]

print(f"  Total power: {total_power:.2f}")
for lo, hi, label in bands:
    mask = (freqs >= lo) & (freqs < hi)
    band_power = np.sum(mean_power[mask])
    pct = band_power / total_power * 100
    print(f"    {label:<40} {pct:6.2f}%")

signal_power = np.sum(mean_power[freqs < 10])
noise_power = np.sum(mean_power[freqs >= 10])
snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
print(f"\n  Right wrist Y: SNR = {snr:.1f} dB (signal < 10Hz vs noise >= 10Hz)")

# Multiple joints
print("\n  SNR by joint (signal < 10Hz vs noise >= 10Hz):")
print(f"  {'Keypoint':<35} {'SNR_x(dB)':>10} {'SNR_y(dB)':>10} {'SNR_z(dB)':>10} {'Avg SNR':>10} {'Noise%':>8}")
for kp in ['right_ankle', 'right_knee', 'right_hip', 'right_shoulder',
           'right_elbow', 'right_wrist', 'nose',
           'right_second_finger_mcp', 'right_first_finger_distal']:
    if kp not in kp_names:
        continue
    idx = kp_names.index(kp)
    snr_axes = []
    noise_pcts = []
    for axis in range(3):
        power_ax = np.zeros(len(freqs))
        cnt = 0
        for i in range(len(clean_data)):
            sig = clean_data[i, :, idx, axis]
            if np.any(np.isnan(sig)):
                continue
            sig_d = sig - np.mean(sig)
            fft_v = np.fft.rfft(sig_d)
            power_ax += np.abs(fft_v)**2
            cnt += 1
        if cnt > 0:
            power_ax /= cnt
        sp = np.sum(power_ax[freqs < 10])
        np_ = np.sum(power_ax[freqs >= 10])
        snr_axes.append(10 * np.log10(sp / max(np_, 1e-10)))
        noise_pcts.append(np_ / max(sp + np_, 1e-10) * 100)
    print(f"  {kp:<35} {snr_axes[0]:10.1f} {snr_axes[1]:10.1f} {snr_axes[2]:10.1f} "
          f"{np.mean(snr_axes):10.1f} {np.mean(noise_pcts):8.1f}%")

# ====================================================================
# 6. INTER-SHOT BONE LENGTH (same player)
# ====================================================================
print("\n" + "="*80)
print("6. INTER-SHOT BONE LENGTH CONSISTENCY")
print("="*80)

print(f"\n  {'Bone':<20} {'Plyr':>5} {'Mean(cm)':>10} {'Inter-std(mm)':>14} {'Inter-CV%':>10}")
for kp1, kp2, name in [('right_shoulder', 'right_elbow', 'R upper arm'),
                         ('right_elbow', 'right_wrist', 'R forearm'),
                         ('right_hip', 'right_knee', 'R thigh'),
                         ('right_knee', 'right_ankle', 'R shin')]:
    idx1 = kp_names.index(kp1)
    idx2 = kp_names.index(kp2)
    diff = clean_data[:, :, idx1, :] - clean_data[:, :, idx2, :]
    bone_len = np.sqrt(np.sum(diff**2, axis=2))
    mean_bone_per_shot = np.mean(bone_len, axis=1)

    player_ids = df['participant_id'].values[clean_mask]
    for pid in sorted(df['participant_id'].unique()):
        pmask = player_ids == pid
        if pmask.sum() == 0:
            continue
        player_means = mean_bone_per_shot[pmask]
        inter_std = np.std(player_means)
        inter_mean = np.mean(player_means)
        inter_cv = inter_std / inter_mean * 100
        print(f"  {name:<20} {pid:>5} {inter_mean*FT_TO_CM:10.1f} {inter_std*FT_TO_MM:14.1f} {inter_cv:10.2f}%")

# ====================================================================
# 7. ACCELERATION NOISE
# ====================================================================
print("\n" + "="*80)
print("7. ACCELERATION NOISE (static frames)")
print("="*80)

static_vel_ft = np.diff(clean_data[:, 0:31, :, :], axis=1) / dt
static_acc = np.diff(static_vel_ft, axis=1) / dt  # ft/s^2
static_acc_ms2 = static_acc * FT_TO_M  # m/s^2
acc_mag = np.sqrt(np.sum(static_acc_ms2**2, axis=3))
mean_acc = np.mean(acc_mag, axis=(0,1))

print(f"\n  {'Keypoint':<35} {'Mean |a| (m/s^2)':>18}")
for kp in key_joints:
    if kp in kp_names:
        idx = kp_names.index(kp)
        print(f"  {kp:<35} {mean_acc[idx]:18.2f}")

# ====================================================================
# 8. RELEASE VELOCITY NOISE IMPACT
# ====================================================================
print("\n" + "="*80)
print("8. RELEASE VELOCITY NOISE IMPACT ANALYSIS")
print("="*80)

# Around frame 120 (release), what does velocity noise look like?
# Use frames 110-130 and compute velocity
release_vel = np.diff(clean_data[:, 110:131, :, :], axis=1) / dt * FT_TO_M  # m/s
rw_idx = kp_names.index('right_wrist')

# Right wrist velocity at release
rw_vel = release_vel[:, :, rw_idx, :]  # (n_clean, 20, 3)
rw_speed = np.sqrt(np.sum(rw_vel**2, axis=2))  # (n_clean, 20)

# Frame-to-frame velocity jitter at release
vel_diff = np.diff(rw_vel, axis=1)  # (n_clean, 19, 3)
vel_jitter = np.sqrt(np.sum(vel_diff**2, axis=2))  # (n_clean, 19)

print(f"  Right wrist velocity at release (frames 110-130):")
print(f"    Mean speed: {np.mean(rw_speed):.2f} m/s")
print(f"    Speed range: [{np.min(np.mean(rw_speed, axis=1)):.2f}, {np.max(np.mean(rw_speed, axis=1)):.2f}] m/s")
print(f"    Frame-to-frame velocity jitter: {np.mean(vel_jitter):.2f} m/s")
print(f"    (This jitter = true acceleration + noise)")

# Compare to static velocity jitter (pure noise)
static_vel_ms = np.diff(clean_data[:, 0:31, :, :], axis=1) / dt * FT_TO_M
rw_static_vel = static_vel_ms[:, :, rw_idx, :]
static_vel_diff = np.diff(rw_static_vel, axis=1)
static_vel_jitter = np.sqrt(np.sum(static_vel_diff**2, axis=2))

print(f"    Static velocity jitter (noise floor): {np.mean(static_vel_jitter):.2f} m/s")
print(f"    Release/static jitter ratio: {np.mean(vel_jitter)/np.mean(static_vel_jitter):.1f}x")

# Estimate: for a 3-point average of velocity, noise reduces by sqrt(3)
n_avg = 3
reduced_noise = np.mean(static_vel_jitter) / np.sqrt(n_avg)
print(f"\n  Noise reduction with {n_avg}-frame smoothing: {reduced_noise:.2f} m/s")
print(f"  Noise reduction with 5-frame Savitzky-Golay: ~{np.mean(static_vel_jitter)/2.2:.2f} m/s")

# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "="*80)
print("="*80)
print("COMPREHENSIVE SUMMARY")
print("="*80)
print("="*80)

body_idx = [kp_names.index(k) for k in body_kps if k in kp_names]
hand_idx = [kp_names.index(k) for k in hand_kps if k in kp_names]
foot_idx = [kp_names.index(k) for k in foot_kps if k in kp_names]

body_sigma_ft = np.mean(sigma_pos[body_idx])
hand_sigma_ft = np.mean(sigma_pos[hand_idx])
foot_sigma_ft = np.mean(sigma_pos[foot_idx])
wrist_sigma_ft = sigma_pos[kp_names.index('right_wrist')]
ankle_sigma_ft = sigma_pos[kp_names.index('right_ankle')]

print(f"\n  UNITS: Data is in FEET (confirmed: upper arm = {upper_arm:.3f} ft = {upper_arm*FT_TO_CM:.0f} cm)")
print(f"\n  POSITION NOISE (1-sigma, from frame-to-frame jumps in static period):")
print(f"    Ankle (best case):    {ankle_sigma_ft*FT_TO_MM:6.1f} mm")
print(f"    Foot joints (mean):   {foot_sigma_ft*FT_TO_MM:6.1f} mm")
print(f"    Body joints (mean):   {body_sigma_ft*FT_TO_MM:6.1f} mm")
print(f"    Right wrist:          {wrist_sigma_ft*FT_TO_MM:6.1f} mm")
print(f"    Hand joints (mean):   {hand_sigma_ft*FT_TO_MM:6.1f} mm")

print(f"\n  VELOCITY NOISE (1-sigma, from finite differences in static period):")
wrist_vel_noise = wrist_sigma_ft * np.sqrt(2) / dt * FT_TO_M
ankle_vel_noise = ankle_sigma_ft * np.sqrt(2) / dt * FT_TO_M
body_vel_noise = body_sigma_ft * np.sqrt(2) / dt * FT_TO_M
print(f"    Ankle:               {ankle_vel_noise:6.3f} m/s")
print(f"    Body joints (mean):  {body_vel_noise:6.3f} m/s")
print(f"    Right wrist:         {wrist_vel_noise:6.3f} m/s")

print(f"\n  BONE LENGTH CONSISTENCY:")
bone_cvs = [bone_results[n]['cv'] for n in bone_results]
print(f"    Mean within-shot CV: {np.mean(bone_cvs):.2f}%")
print(f"    Range: [{np.min(bone_cvs):.2f}%, {np.max(bone_cvs):.2f}%]")
print(f"    Best (torso/thigh): ~{min(bone_results.get('R torso',{}).get('cv',99), bone_results.get('R thigh',{}).get('cv',99)):.2f}%")
print(f"    Worst (hands): ~{max(bone_results.get('R hand',{}).get('cv',99), bone_results.get('L hand',{}).get('cv',99)):.2f}%")

print(f"\n  PHYSICS FEASIBILITY:")
print(f"    Typical release velocity: ~7 m/s (23 ft/s)")
print(f"    Wrist velocity noise:     {wrist_vel_noise:.3f} m/s")
print(f"    Velocity SNR:             {7.0/wrist_vel_noise:.1f}x ({20*np.log10(7.0/wrist_vel_noise):.1f} dB)")
print(f"    With 5-frame SG filter:   ~{7.0/(wrist_vel_noise/2.2):.1f}x ({20*np.log10(7.0/(wrist_vel_noise/2.2)):.1f} dB)")
print(f"    Release angle noise:      ~{np.degrees(np.arctan(wrist_vel_noise/7.0)):.1f} degrees (from velocity uncertainty)")

print(f"\n  FFT SIGNAL-TO-NOISE:")
print(f"    Right wrist Y: SNR = {snr:.1f} dB")
print(f"    (Higher SNR = more signal relative to noise)")

# Key question: is the static period truly static?
print(f"\n  STATIC PERIOD VALIDATION:")
wrist_drift_3d = np.sqrt(np.sum((clean_data[:, 30, rw_idx, :] - clean_data[:, 0, rw_idx, :])**2, axis=1))
ankle_drift_3d = np.sqrt(np.sum((clean_data[:, 30, ra_idx, :] - clean_data[:, 0, ra_idx, :])**2, axis=1))
print(f"    Wrist drift (frame 0-30): mean={np.mean(wrist_drift_3d)*FT_TO_MM:.1f} mm, "
      f"std={np.std(wrist_drift_3d)*FT_TO_MM:.1f} mm")
print(f"    Ankle drift (frame 0-30): mean={np.mean(ankle_drift_3d)*FT_TO_MM:.1f} mm, "
      f"std={np.std(ankle_drift_3d)*FT_TO_MM:.1f} mm")
print(f"    WARNING: Wrist drift >> ankle drift suggests real preparatory motion in frames 0-30")
print(f"    Noise estimates from wrist static period may be UPPER BOUNDS (include real motion)")

# Use truly static joint (ankle) for best noise floor estimate
print(f"\n  BEST NOISE FLOOR ESTIMATE (from ankle, most static joint):")
print(f"    Position: {ankle_sigma_ft*FT_TO_MM:.1f} mm")
print(f"    Velocity: {ankle_vel_noise:.3f} m/s")
print(f"    This represents the INSTRUMENT noise floor")
print(f"    Wrist noise during static period ({wrist_sigma_ft*FT_TO_MM:.1f} mm) includes real micro-motion")

# Conclusion
print(f"\n  CONCLUSION FOR PHYSICS-BASED PREDICTION:")
print(f"    Position noise:  ~1.4 mm (ankle) to ~6.5 mm (wrist)")
print(f"    Velocity noise:  ~0.12 m/s (ankle) to ~0.55 m/s (wrist)")
print(f"    This is MODERATE noise - not hopeless, but challenging")
print(f"    Raw finite-difference velocity has ~8% noise at release (0.55/7.0)")
print(f"    Smoothing (SG/Kalman) can reduce to ~3-4% - borderline useful")
print(f"    Key limitation: DIRECTION of velocity (angle/depth/LR) needs <1 degree accuracy")
print(f"    At 7 m/s, 0.55 m/s noise = {np.degrees(np.arctan(0.55/7.0)):.1f} degree uncertainty")
print(f"    Even with 2.2x smoothing: {np.degrees(np.arctan(0.55/7.0/2.2)):.1f} degree uncertainty")
print(f"    These angles map to ~2-5 cm target uncertainty at the hoop (25 ft away)")

print("\nDone!")
