# Mocap Noise Analysis - Raw Data Quality Assessment

**Date:** 2026-02-05
**Script:** physics_engine scratchpad/mocap_noise_analysis.py
**Data:** train.csv (345 shots, 5 players, 60fps, 240 frames)

## 1. Bone Length Consistency (Position Noise Floor)

Bone lengths should be constant across frames within a shot. Variation = noise.

### Arm Chain (reliable joints)

| Bone | Mean (ft) | Intra-shot Std (ft) | CV (%) | Std (mm) |
|------|-----------|---------------------|--------|----------|
| R shoulder-elbow | 0.949 | 0.052 | 5.5% | 15.8 |
| R elbow-wrist | 0.809 | 0.084 | 10.4% | 25.7 |
| L shoulder-elbow | 0.914 | 0.043 | 4.7% | 13.1 |
| L elbow-wrist | 0.785 | 0.055 | 7.0% | 16.7 |
| neck-mid_hip | 1.712 | 0.072 | 4.2% | 21.9 |
| L hip-R hip | 0.680 | 0.020 | 3.0% | 6.2 |
| R hip-knee | 1.411 | 0.041 | 2.9% | 12.4 |
| R knee-ankle | 1.356 | 0.042 | 3.1% | 12.7 |

### Hand/Finger Joints (very noisy)

| Bone | Mean (ft) | Intra-shot Std (ft) | CV (%) | Std (mm) |
|------|-----------|---------------------|--------|----------|
| R wrist-2nd MCP | 1.245 | 0.636 | 51.1% | 193.9 |
| R 2nd MCP-PIP | 0.257 | 0.251 | 97.7% | 76.5 |
| R 2nd PIP-DIP | 0.263 | 0.301 | 114.7% | 91.9 |
| R 2nd DIP-distal | 0.094 | 0.065 | 69.5% | 19.8 |
| R thumb MCP-IP | 0.086 | 0.044 | 51.5% | 13.4 |
| R 3rd MCP-PIP | 0.111 | 0.071 | 63.7% | 21.6 |
| R 5th MCP-PIP | 0.137 | 0.125 | 91.7% | 38.1 |

### Key Findings
- **Arm joints:** ~16-26 mm bone length noise, implies ~11-18 mm per-joint position noise
- **Finger joints:** EXTREMELY noisy - 20-194 mm bone length noise, CV often >50%
- **Wrist to 2nd finger MCP:** 0.636 ft std (194 mm) - this is not noise in the traditional sense, this indicates the finger tracking is fundamentally unreliable. The "bone" mean is 1.245 ft which is way too long (should be ~0.3 ft)
- **Lower body:** Much more stable (hip width CV = 3.0%, knee-ankle CV = 3.1%)

### Inter-shot Variation (Same Player)

Shoulder-to-elbow length inter-shot std per player:
- Player 1: 0.029 ft (8.7 mm)
- Player 2: 0.006 ft (2.0 mm)
- Player 3: 0.005 ft (1.6 mm)
- Player 4: 0.011 ft (3.3 mm)
- Player 5: 0.008 ft (2.3 mm)

Player 1 has notably more variable data than others.

## 2. Velocity Smoothness

### Maximum Values Across Shots

| Joint | Max Speed (m/s) mean | Max Accel (m/s^2) mean | Max Jerk (m/s^3) mean |
|-------|---------------------|------------------------|----------------------|
| right_wrist | 3.82 (max 19.28) | 44.7 (max 1013) | 1494 (max 47346) |
| right_elbow | 2.79 (max 7.39) | 30.4 (max 367) | 957 (max 18073) |
| right_shoulder | 1.38 (max 8.22) | 18.0 (max 368) | 586 (max 18178) |
| right_2nd_finger_distal | 4.58 (max 17.62) | 55.0 (max 667) | 1841 (max 35086) |

### Physical Plausibility
- Human limb max acceleration: ~20-50 m/s^2 for explosive movements
- 11.7% of shots show wrist max_accel > 50 m/s^2 (borderline)
- 2.4% show wrist max_accel > 100 m/s^2 (physically impossible - pure noise)
- Finger distal: 38% exceed 50 m/s^2 (finger data is very noisy)

### Velocity Noise Ratio (raw vs Savgol-filtered)
- Wrist: median noise ratio = 0.045 (4.5% of velocity is noise)
- Elbow: median noise ratio = 0.040 (4.0%)
- Shoulder: median noise ratio = 0.044 (4.4%)
- Finger distal: median noise ratio = 0.057 (5.7%)
- These are median values; p95 is 2-3x higher

## 3. Release Frame Region Analysis

### Release Frame Detection (Wrist Velocity Peak)
- Mean: 165.6, Std: 24.9
- Range: 120 to 199
- p5=120, p25=148, p50=173, p75=185, p95=199
- Very wide distribution - release timing varies significantly across shots

### Position at Release (Right Wrist)
- X: 18.199 +/- 0.772 ft (inter-shot variation, includes player differences)
- Y: -24.752 +/- 0.402 ft
- Z: 5.254 +/- 1.009 ft (large Z variation due to different release heights)

### Velocity at Release (Right Wrist)
- Raw: x=0.11+/-1.23, y=0.27+/-0.34, z=-1.73+/-2.76 m/s
- Smooth: x=0.10+/-1.20, y=0.26+/-0.34, z=-1.72+/-2.74 m/s
- Velocity noise (raw-smooth std): x=0.072, y=0.032, z=0.051 m/s

### Release Angle Estimate
- Raw: mean=-20.7 deg, std=60.2 deg
- Smooth: mean=-21.0 deg, std=60.5 deg
- Angle noise (raw-smooth): mean_abs=2.17 deg, std=5.31 deg
- NOTE: Negative angles and huge std suggest the wrist velocity peak is NOT the true ball release for many shots. The wrist is decelerating when ball is released.

## 4. Lateral (Left-Right) Signal vs Noise

### Wrist X Position at Release
- Overall: mean=18.199 ft, std=0.772 ft
- Per-player std (intra-player variation):
  - Player 1: 3.62 inches
  - Player 2: 10.55 inches (high - likely noisy data or varied technique)
  - Player 3: 2.58 inches
  - Player 4: 2.62 inches
  - Player 5: 4.09 inches

### Lateral Velocity
- Raw mean: 0.108 m/s, std: 1.232 m/s
- Smooth mean: 0.101 m/s, std: 1.203 m/s
- Noise (raw-smooth): 0.072 m/s

### Signal-to-Noise for Left-Right Target
- Target left_right: mean=-0.81 in, std=3.77 in, range=[-12.98, 10.06]
- Lateral velocity noise: 0.072 m/s
- At flight time ~0.7s: lateral displacement noise at hoop = 1.99 inches
- **SNR = 3.77 / 1.99 = 1.89** (marginal - explains difficulty)
- Correlation(wrist_X, left_right) = -0.014 (essentially zero)

The zero correlation between wrist X position and left_right target is important: it means the lateral ball displacement at the hoop is NOT determined by where the wrist is, but by the lateral release velocity and spin. This is a very subtle signal to extract from noisy mocap.

## 5. High-Frequency Content (FFT)

- Right wrist: 0.20% of signal power above 15 Hz
- Right finger distal: 0.18% of signal power above 15 Hz
- The data is NOT dominated by high-frequency noise in the spectral sense
- Most noise appears as low-frequency wandering (bone length changes)

## 6. Frame-to-Frame Jitter (Quiet Period, Frames 0-60)

| Joint | Mean Jitter (mm/frame) | Median | p95 |
|-------|------------------------|--------|-----|
| nose | 3.04 | 2.58 | 6.92 |
| right_elbow | 4.95 | 3.87 | 13.21 |
| right_wrist | 10.94 | 9.25 | 22.98 |
| right_2nd_finger_distal | 13.02 | 12.10 | 24.34 |

At 60fps, a stationary joint should have zero displacement. The jitter here IS the noise floor:
- **Wrist position noise: ~10 mm per frame** (during quiet standing)
- **Finger position noise: ~13 mm per frame**
- **Elbow position noise: ~5 mm per frame**
- **Head position noise: ~3 mm per frame**

## Implications for Physics Engine

1. **Finger data is essentially useless for precise physics.** With CV > 50% on bone lengths and mean bone lengths that are anatomically impossible (wrist to 2nd MCP = 1.25 ft, should be ~0.3 ft), the finger keypoints cannot be trusted for contact dynamics.

2. **Arm chain (shoulder-elbow-wrist) is usable** with ~16-26 mm noise. Filtering with physics constraints (bone length, joint angle limits) could reduce this to ~5-10 mm.

3. **Velocity estimation is feasible** with Savgol filtering. The noise ratio is ~4-6% of signal. However, acceleration is borderline (11% of shots have physically impossible wrist accelerations).

4. **Release frame identification is challenging.** The wrist velocity peak has std=24.9 frames - nearly half a second of uncertainty. Better release detection (using finger extension, ball-hand distance, or deceleration patterns) is needed.

5. **Left-right prediction is fundamentally SNR-limited.** The lateral velocity noise (0.072 m/s) propagates to ~2 inches of uncertainty at the hoop, while the target has std of 3.77 inches. SNR of 1.89 means at most r~0.65 is theoretically achievable from velocity alone. The actual r=0.14 suggests we are far from extracting even this limited signal.

6. **The data is markerless motion capture** (computer vision-based), not marker-based mocap. This explains:
   - Large noise on distal extremities (fingers)
   - Low noise on torso/head
   - Bone length variation (skeleton fitting errors)
   - Occasional extreme outlier frames
