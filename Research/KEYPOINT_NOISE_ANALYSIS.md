# Keypoint Noise Floor Analysis

**Date**: 2026-02-09
**Script**: `scripts/keypoint_noise_analysis_v2.py`
**Data**: 345 train shots, 69 keypoints, 240 frames at 60fps

## Units

Data is in **FEET**. Confirmed by:
- Upper arm (shoulder-to-elbow): 1.020 ft = 31.1 cm (typical human: ~30 cm)
- Forearm (elbow-to-wrist): 0.779 ft = 23.7 cm (typical human: ~25 cm)
- Hoop position: [5.25, -25, 10] feet (consistent with basketball court)

Conversion: 1 ft = 304.8 mm = 30.48 cm = 0.3048 m

## Data Quality

- 332/345 shots are clean (no NaN)
- 13 shots have NaN in ALL keypoints simultaneously (137 NaN frames per keypoint)
- NaN affects 0.17% of total data - minimal impact

## 1. Position Noise (from frame-to-frame jumps in static period, frames 0-30)

Position noise estimated as sigma = rms(frame_jump) / sqrt(2), since each jump is the difference of two noisy measurements.

| Category | Example Joint | Sigma (mm) | Sigma (ft) |
|----------|--------------|-----------|-----------|
| Feet (best) | right_ankle | 1.4 | 0.00465 |
| Feet (mean) | all foot joints | 1.8 | 0.00590 |
| Body (mean) | 17 body joints | 2.8 | 0.00901 |
| Right shoulder | | 2.4 | 0.00782 |
| Right elbow | | 4.2 | 0.01393 |
| Right wrist | | 6.6 | 0.02156 |
| Left wrist | | 5.4 | 0.01767 |
| Hand joints (mean) | fingers, thumb, pinky | 10.0 | 0.03260 |
| Worst (left 4th finger) | | ~60+ | ~0.20 |

**Key finding**: The instrument noise floor is approximately **1.4 mm** (from ankle, most static joint). The wrist shows 6.6 mm which is an UPPER BOUND because it includes real preparatory micro-motion during frames 0-30.

### Static Period Validation

- Ankle drift (frame 0 to 30): mean 14.8 mm - mostly noise
- Wrist drift (frame 0 to 30): mean 145.0 mm - significant real motion!
- The wrist is NOT truly static in frames 0-30. Players are making preparatory adjustments.
- Therefore wrist noise estimates from this period include real motion and are upper bounds.

## 2. Velocity Noise

| Joint | Velocity Noise (m/s) | Notes |
|-------|---------------------|-------|
| right_ankle | 0.120 | Best estimate of instrument noise |
| right_hip | 0.154 | |
| right_knee | 0.134 | |
| right_shoulder | 0.202 | |
| right_elbow | 0.360 | |
| right_wrist | 0.558 | Upper bound (includes micro-motion) |
| right_finger_mcp | 0.774 | Very noisy |

**Note**: Velocity noise from finite differences is sigma_vel = sigma_pos * sqrt(2) / dt. At 60fps (dt = 0.0167s), even 1.4mm position noise becomes 0.12 m/s velocity noise.

## 3. Bone Length Consistency

Rigid bones should have constant length. Frame-to-frame variation reveals measurement noise.

| Bone | Mean Length (cm) | Within-Shot Std (mm) | CV (%) | Noise Est (mm) |
|------|-----------------|---------------------|--------|----------------|
| L shin | 41.9 | 7.3 | 1.74 | 5.2 |
| L thigh | 42.9 | 11.9 | 2.75 | 8.4 |
| R thigh | 43.0 | 12.4 | 2.84 | 8.8 |
| R shin | 41.3 | 12.7 | 3.09 | 9.0 |
| L torso | 53.0 | 20.5 | 3.92 | 14.5 |
| L upper arm | 27.9 | 13.1 | 4.72 | 9.3 |
| R torso | 52.3 | 25.2 | 4.98 | 17.8 |
| R upper arm | 28.9 | 15.8 | 5.46 | 11.2 |
| L forearm | 23.9 | 16.7 | 6.97 | 11.8 |
| R forearm | 24.6 | 25.7 | 10.47 | 18.1 |
| R hand | 10.3 | 25.9 | 25.24 | 18.3 |
| L hand | 10.0 | 26.6 | 26.69 | 18.8 |

**Key finding**: Bone length noise estimates (8-18 mm for body joints) are HIGHER than the frame-to-frame jump estimates (1.4-6.6 mm). This means:
1. The bone length method captures cumulative drift, not just single-frame noise
2. Some "noise" is actually real soft tissue deformation and joint articulation
3. The right forearm (CV=10.47%) and hands (CV=25-27%) are extremely noisy - the motion capture system struggles with these smaller segments

### Inter-Shot Bone Length Consistency (per player)

Same player should have same bone lengths across shots:
- R shin inter-shot CV: 0.26-0.64% (excellent - rigid bone, well-tracked)
- R upper arm inter-shot CV: 0.59-2.81% (moderate - Player 1 worst at 2.81%)
- R forearm inter-shot CV: 0.95-2.82% (poor - Player 5 worst at 2.82%)

## 4. Frequency Analysis (FFT)

Power distribution for right wrist y-axis:

| Frequency Band | Power (%) | Interpretation |
|---------------|-----------|---------------|
| DC-2 Hz | 94.28% | Slow posture/standing |
| 2-5 Hz | 3.89% | Shooting motion |
| 5-10 Hz | 1.00% | Fast motion components |
| 10-20 Hz | 0.55% | Likely noise |
| 20-30 Hz (near Nyquist) | 0.27% | Definitely noise |

**SNR**: 20.8 dB for right wrist y-axis (signal < 10 Hz vs noise >= 10 Hz)

SNR by joint (average across x,y,z):
- Right ankle: 20.2 dB (1.2% noise)
- Right knee: 22.3 dB (1.0% noise)
- Right hip: 23.2 dB (0.9% noise)
- Right shoulder: 23.9 dB (0.8% noise)
- Right elbow: 24.8 dB (0.6% noise)
- Right wrist: 24.0 dB (0.6% noise)
- Finger joints: 23.8-23.9 dB (0.6% noise)

**Interpretation**: High-frequency noise is a small fraction of total power (<2%). Most "noise" in static-period analysis is low-frequency drift/micro-motion, not high-frequency measurement error. This means Savitzky-Golay and low-pass filters will remove only ~1% of power.

## 5. Acceleration Noise

| Joint | Mean |a| in static period (m/s^2) |
|-------|--------------------------------------|
| right_ankle | 1.35 |
| right_knee | 1.39 |
| right_hip | 1.59 |
| right_shoulder | 1.70 |
| nose | 2.20 |
| right_elbow | 3.53 |
| right_wrist | 6.32 |
| left_wrist | 6.36 |
| right_finger_mcp | 10.66 |

Acceleration noise amplifies position noise by factor of 1/dt^2 = 3600. This makes second-derivative quantities essentially unusable from raw data.

## 6. Release Velocity Analysis

At frames 110-130 (around release):
- Mean wrist speed: 1.55 m/s (NOTE: this is LOWER than expected ~7 m/s)
- Frame-to-frame velocity jitter at release: 0.21 m/s
- Static velocity jitter (noise floor): 0.11 m/s
- Release/static jitter ratio: 2.0x

The 1.55 m/s mean speed is surprisingly low. This suggests either:
1. The release frame varies significantly (not always at 120)
2. The wrist slows down before the actual ball release
3. Peak velocity occurs at a different frame

## Comprehensive Summary

### Position Noise Hierarchy
```
Instrument floor (ankle):     ~1.4 mm
Large body joints (knee/hip): ~1.6-1.8 mm
Shoulder:                     ~2.4 mm
Elbow:                        ~4.2 mm
Wrist:                        ~6.6 mm (upper bound, includes micro-motion)
Finger joints:                ~9-10 mm
Worst fingers:                ~60+ mm
```

### Velocity Noise Hierarchy
```
Instrument floor (ankle):     ~0.12 m/s
Large body joints:            ~0.13-0.15 m/s
Shoulder:                     ~0.20 m/s
Elbow:                        ~0.36 m/s
Wrist:                        ~0.56 m/s (upper bound)
Finger joints:                ~0.77-0.80 m/s
```

### Implications for Physics-Based Prediction

1. **Position accuracy**: At the instrument level (~1.4 mm), position is very accurately measured. Body joints have ~2-4 mm accuracy. This is quite good.

2. **Velocity is the bottleneck**: Finite-difference velocity from 60 fps data amplifies noise by factor of ~85x (sqrt(2)/dt). Even the best-case 1.4 mm position noise becomes 0.12 m/s velocity noise.

3. **Release velocity estimation**: At ~7 m/s release speed, raw velocity noise (~0.56 m/s) gives ~8% error. With smoothing this reduces to ~3-4%, but DIRECTIONAL accuracy (which determines angle/depth/LR) needs <1 degree, and current noise gives ~4.5 degree uncertainty (2.0 degrees with smoothing).

4. **Smoothing helps but is limited**: FFT shows only ~1% of power is above 10 Hz. Most "noise" is low-frequency, so high-frequency filters can only remove a small fraction. Kalman filtering could help more by incorporating physics constraints.

5. **Hand joints are nearly useless for physics**: With 10+ mm noise and 25% bone length CV, finger joints cannot provide reliable grip/release information.

6. **The real noise floor is ~1.4 mm (ankle)**. Wrist's apparent 6.6 mm noise is inflated by real preparatory motion. The true wrist instrument noise is likely closer to 2-3 mm (interpolating between ankle and the body joint trend).

### Verdict: MODERATE noise - physics is challenging but not hopeless

- For POSITION-BASED features (pose at a specific frame): noise is small relative to body scale (1-7 mm vs ~30 cm limbs). This is why the current per-example approach works well.
- For VELOCITY-BASED features (release velocity direction): noise is significant. Raw finite differences give ~4.5 degree directional uncertainty, reducible to ~2 degrees with smoothing. At 25 ft range, 2 degrees = ~10 inches lateral error at the hoop.
- For ACCELERATION-BASED features: essentially unusable from raw data. Need Kalman filter or physics model to denoise.
- **Key insight**: The current per-example pipeline uses POSITION features at specific frames, which is the right strategy given this noise profile. Velocity/physics approaches would need Kalman-level denoising to compete.
