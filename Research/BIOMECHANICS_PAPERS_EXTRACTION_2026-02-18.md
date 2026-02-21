# Biomechanics Papers Extraction - 2026-02-18

Three papers read in full. This document contains all variables, formulas, findings, and
implementation notes extracted for feature engineering.

---

## Paper 1: Cabarkapa et al. (2023) - Frontiers in Sports and Active Living
DOI: 10.3389/fspor.2023.1208915

### Setup
- 34 recreationally active males, 10 consecutive free-throw shots at 4.57 m
- Markerless motion capture: SwRI Enable system, 9 cameras at 120 Hz
- Classification: proficient >= 70% made; non-proficient < 70%

### Phase Definitions
- Preparatory phase (PP): initial concentric movement during the shooting motion
- Release phase (RP): the time point at which the ball leaves the shooter's hand
- ROM: amount of movement in each joint between PP and RP

### Complete Variable Table (Table 1) - Exact Definitions

| Variable | Definition |
|---|---|
| Release angle | Vertical angle to ground at which ball leaves hand at release phase |
| Knee angle | Internal angle between thigh and shank |
| Hip angle | Internal angle between torso and thigh |
| Ankle angle | Internal angle between shank and foot |
| Elbow angle | Internal angle between upper arm and forearm |
| COM height | Perpendicular distance of body COM from ground / participant height |
| COM velocity | Velocity of COM between preparatory and release phases |
| Stance width | Distance between right and left foot during preparatory phase |
| Stance alignment | Vertical feet alignment during PP. Positive = right foot more forward |
| Elbow height | Perpendicular distance of olecranon process from ground / participant height, at PP |
| Release height | Perpendicular distance of hand from ground at ball release / participant height |
| Trunk lean | Angle between torso and imaginary vertical axis at ball release. Positive = forward lean |
| Forearm angle | Angle between forearm and imaginary vertical axis. Positive = lateral elbow deviation |

For each joint, sub-variables:
- [Joint]-PP angle (deg): angle at preparatory phase
- [Joint]-RP angle (deg): angle at release phase
- [Joint] ROM (deg): range of motion from PP to RP
- [Joint] peak angular velocity (deg/s): maximal angular velocity between PP and RP
- [Joint] mean angular velocity (deg/s): rate of change in angular displacement between PP and RP

### Statistically Significant Differentiators: Proficient vs Non-Proficient (Table 2)

All p < 0.05, statistically significant:

| Variable | Non-proficient | Proficient | p | Effect size |
|---|---|---|---|---|
| Knee peak angular velocity (deg/s) | 269.4 (60.6) | 212.9 (47.6) | 0.005 | d=1.037 LARGE |
| Knee mean angular velocity (deg/s) | 170.7 (43.2) | 123.0 (65.6) | 0.012 | d=0.425 |
| COM peak velocity (m/s) | 1.07 (0.23) | 0.87 (0.17) | 0.007 | d=0.988 LARGE |
| COM mean velocity (m/s) | 0.69 (0.24) | 0.54 (0.15) | 0.036 | d=0.733 |
| Release height (normalized) | 1.12 (0.07) | 1.17 (0.05) | 0.010 | d=0.438 |
| Trunk lean (deg) | 1.87 (3.28) | -1.11 (3.49) | 0.016 | d=0.880 LARGE |

Non-significant but notable (moderate effect sizes):
- Knee angle-PP: 107.3 vs 113.3 deg (d=0.486 Medium)
- Elbow peak angular velocity: 975.2 vs 899.0 deg/s (d=0.627 Medium, p=0.075)

### Made vs Missed Within Proficient Group (Table 3)

Only two significant variables (both small effect):
- Release height: missed=1.19 (0.04) vs made=1.17 (0.05), p=0.035, d=0.161
- COM height at RP: missed=0.68 (0.02) vs made=0.67 (0.02), p=0.021, d=0.176

Key insight: MISSED shots had HIGHER release height. Overemphasizing height is counterproductive.

### Reference Values (from Table 2, proficient group)
- Release angle: 51.4 deg (SD=3.2)
- Knee angle at release: 165.8 deg (near full extension)
- Elbow angle at release: 159.6 deg (near full extension)
- Hip angle at release: 173.6 deg (near full extension)
- Trunk lean at release: -1.11 deg (slightly backward, near vertical)
- Stance width: 32.4 cm
- Elbow height at PP (normalized): 0.61 (relative to body height)
- Release height (normalized): 1.17 (relative to body height)

---

## Paper 2: Li et al. (2025) - Journal of Human Kinetics
DOI: 10.5114/jhk/203104

### Setup
- 10 college athletes + 10 recreational players
- 3 distances: 3.2 m, 5 m, 6.8 m; 3 successful shots per distance per player
- 13 cameras at 240 Hz (OptiTrack LEYARD)
- 57 markers on bony landmarks
- Low-pass filter: 4th-order zero-lag Butterworth, cut-off 10 Hz
- Joint angles: X-Y-Z Cardan rotation, distal relative to proximal

### Phase Definition
- Shooting phase: starts when COM drops to its lowest point, ends when ball is released
- All phase data temporally normalized to 0-100% before computing coupling angles

### Core Metric: Coupled Angular Variability (CAV) - Complete Formula Set

CAV quantifies coordination variability across trials for two adjacent joints (shoulder-elbow).

**Step 1: Coupling angle gamma_i at each instant i**

Case 1 - when (theta_P(i+1) - theta_Pi) > 0:
  gamma_i = Atan( (theta_D(i+1) - theta_Di) / (theta_P(i+1) - theta_Pi) ) * (180/pi)

Case 2 - when (theta_P(i+1) - theta_Pi) < 0:
  gamma_i = Atan( (theta_D(i+1) - theta_Di) / (theta_P(i+1) - theta_Pi) ) * (180/pi) + 180

Special cases:
  gamma_i = 90    if (theta_P diff = 0) AND (theta_D diff > 0)
  gamma_i = -90   if (theta_P diff = 0) AND (theta_D diff < 0)
  gamma_i = -180  if (theta_P diff < 0) AND (theta_D diff = 0)
  gamma_i = Undefined  if both diffs = 0

theta_P = proximal segmental angle (shoulder, sagittal plane)
theta_D = distal segmental angle (elbow, sagittal plane)

**Step 2: Correct to range 0-360 deg**
  gamma_i = gamma_i + 360  if gamma_i < 0
  gamma_i = gamma_i         if gamma_i >= 0

**Step 3: Circular mean components**
  x_bar_i = (1/n) * sum( cos(gamma_i) )
  y_bar_i = (1/n) * sum( sin(gamma_i) )

**Step 4: Mean coupling angle gamma_bar (corrected to 0-360)**
  if x_i > 0, y_i > 0:  gamma_bar = Atan(y_bar/x_bar) * (180/pi)
  if x_i < 0:            gamma_bar = Atan(y_bar/x_bar) * (180/pi) + 180
  if x_i > 0, y_i < 0:  gamma_bar = Atan(y_bar/x_bar) * (180/pi) + 360
  if x_i = 0, y_i > 0:  gamma_bar = 90
  if x_i = 0, y_i < 0:  gamma_bar = -90
  if x_i = 0, y_i = 0:  Undefined

**Step 5: Vector length**
  r_bar = sqrt(x_bar^2 + y_bar^2)

**Step 6: CAV (the final variability metric)**
  CAV = sqrt(2 * (1 - r_bar)) * (180/pi)

Higher CAV = greater coordination variability across trials.
r_bar close to 1.0 = low variability (highly consistent coupling angle across trials).
r_bar close to 0.0 = high variability.

### Eight Coordination Pattern Categories (polar plot quadrants/octants)

Based on coupling angle gamma_bar value:
1. Proximal-dominance (shoulder flexion) - shoulder moves more, both flexing
2. In-phase (shoulder flexion / elbow extension) - both move, opposite directions
3. Distal-dominance (elbow extension) - elbow extension dominates
4. Anti-phase (shoulder extension / elbow extension)
5. Proximal-dominance (shoulder extension)
6. In-phase (shoulder extension / elbow flexion)
7. Distal-dominance (elbow flexion)
8. Anti-phase (shoulder flexion / elbow flexion)

### Key Numerical Findings

CAV values (Median P50, with P25-P75):
- Recreational at 3.2 m: 16.078 (7.8, 33.8)
- Recreational at 5 m: 15.093 (8.8, 29.8)
- Recreational at 6.8 m: 16.947 (10.4, 40.6)
- College athletes at 3.2 m: 15.859 (6.3, 44.9)
- College athletes at 5 m: 18.487 (4.7, 38.7)
- College athletes at 6.8 m: 14.654 (3.4, 39.7)

Dominant patterns:
- Recreational at 3.2 m: distal-dominance/elbow extension (33/100)
- Recreational at 5-6.8 m: in-phase shoulder flexion/elbow extension (55, 46)
- College at 3.2 m: proximal-dominance/shoulder flexion (28)
- College at 5 m: in-phase shoulder flexion/elbow extension (32)
- College at 6.8 m: in-phase + anti-phase (30, 22)

### Key Findings

1. Higher CAV correlates with better accuracy in skilled athletes (p=0.035 at 5m)
2. Elite pattern: proximal (shoulder) initiates early, then distal (elbow/wrist) takes over
3. Recreational players "push" ball with simultaneous shoulder-elbow motion
4. Elbow extension + wrist flexion are planned feedforward based on shoulder displacement
5. College athletes adjust shoulder early in preparatory phase, enabling larger elbow ROM
6. Forearm positioning is a key kinematic differentiator (Cabarkapa 2021a cited)

---

## Paper 3: Chen et al. (2026) - PeerJ
DOI: 10.7717/peerj.20757

### Setup
- 15 experienced + 15 novice male collegiate basketball players
- Experienced (EG): >= 5 years organised play, completed >= 1 full CUBA season
  - Age 22.3 +/- 1.6 yr, height 185.8 +/- 4.3 cm, mass 78.7 +/- 5.7 kg
- Novice (NG): preparing for debut CUBA season
  - Age 19.2 +/- 0.6 yr, height 183.4 +/- 2.6 cm, mass 76.5 +/- 5.1 kg
- Distances: 4.8 m (mid-range) and 6.75 m (long-range)
- 13 OptiTrack cameras at 240 Hz, 57 markers
- 2 AMTI force plates at 2400 Hz
- Butterworth filter: 17 Hz cut-off for moment/power; 6 Hz for velocity

### Phase Windows
- Knee: from onset of countermovement to instant GRF fell below 20 N
- Shoulder, elbow, wrist: from flexion-to-extension switch in standing phase to ball release

### Variables - Exact Names, Units, Formulas

**RTD - Rate of Torque Development** (N m kg^-1 s^-1)
  RTD = Delta_Torque / Delta_Time
  = change in joint moment from onset to its peak / elapsed time between those instants
  Normalized to body mass.
  Derived from moment-time curves (inverse dynamics required).

**P_peak - Peak Power** (W kg^-1)
  Maximum instantaneous power in the analysis window.
  Derived from power-time curves.
  Normalized to body mass.

**AI - Angular Impulse** (N s kg^-1)
  AI = integral of joint torque over the analysis window (time integral).
  Normalized to body mass.
  Reliability: ICC(3,3) and CV (CV% = SD/mean * 100).

**VV - Vertical Release Velocity** (m s^-1)
  Vertical component of ball velocity at moment of release.

**HV - Horizontal Release Velocity** (m s^-1)
  Horizontal component of ball velocity at moment of release.

### Key Numerical Results

**Wrist (Table 1):**
- RTD: NG=0.58+/-0.14 (4.8m), EG=0.53+/-0.23. p(group)=0.047, eta_p^2=0.54 VERY LARGE
  NG has HIGHER RTD (novices more explosive but less sustained)
- AI: NG=0.0004+/-0.0002, EG=0.0015+/-0.0004. p<0.001, eta_p^2=0.54 VERY LARGE
  EG has ~3.75x higher wrist AI. THIS IS THE PRIMARY DIFFERENTIATOR.
- P_peak: NG=0.22+/-0.09, EG=0.35+/-0.15. p>0.05, eta_p^2=0.07 medium

**Elbow (Table 2):**
- RTD: NG=1.09+/-0.49, EG=1.41+/-0.31. p=0.002, eta_p^2=0.12
- P_peak: NG=0.92+/-0.40, EG=1.02+/-0.28. p=0.045, eta_p^2=0.09
- AI: NG=0.0051+/-0.0014, EG=0.0067+/-0.0018. p<0.001, eta_p^2=0.18 LARGE
  All three elbow metrics: EG significantly higher.

**Shoulder (Table 3):**
- RTD: NG=3.12+/-1.49, EG=2.02+/-1.16. p=0.014, eta_p^2=0.09
  NG has HIGHER shoulder RTD (novices over-fire shoulder)
- P_peak: EG higher at 6.75m only (distance effect p=0.036)
- AI: no significant group difference

**Knee (Table 4):**
- P_peak: NG=7.27+/-1.69, EG=9.08+/-2.63. p=0.002, eta_p^2=0.13 MEDIUM-LARGE
  EG significantly higher knee P_peak
- RTD and AI: no significant group difference

**Ball Release Velocities (Table 5):**
- VV: NG=3.89+/-0.37 (4.8m), 4.42+/-0.52 (6.75m); EG=4.59+/-0.47, 5.34+/-0.27
  EG significantly higher VV, p<0.001, eta_p^2=0.1 (group), 0.34 (distance)
- HV: NG=2.96+/-0.60 (4.8m), 3.50+/-0.68 (6.75m); EG=3.39+/-0.36, 3.78+/-0.32
  EG higher HV too, p=0.018, eta_p^2=0.34 (group), 0.28 (distance)
- VV/HV ratio: EG = 4.59/3.39 = 1.35 at 4.8m; NG = 3.89/2.96 = 1.31. Higher ratio = more vertical.

### Key Mechanistic Chain (Expert Pattern)

1. Greater knee P_peak -> higher jump -> higher release point -> better entry angle
2. Greater elbow AI -> more sustained elbow force -> higher VV
3. Greater wrist AI -> sustained wrist flexor action -> more ball spin + reduced HV
4. Result: steeper entry angle, wider rim tolerance, higher accuracy

Novice failure mode:
- Higher shoulder RTD (over-firing) but lower elbow/wrist AI
- Compensates with higher HV (flatter trajectory)
- Less vertical release -> smaller effective rim area

---

## Implementation Notes for Our Data

### What we have: 69 keypoints * 3 coords = 207 features, 240 frames, 60 fps, ~70 shots/player

### Directly implementable features (no force plates needed):

**Group A - Cabarkapa kinematic features:**

1. trunk_lean_at_release = angle(torso_vector, vertical) at frame TARGET_FRAMES['angle']=153
   - torso_vector = (shoulder_midpoint - hip_midpoint)
   - Proficient target: -1.11 deg (slightly backward)

2. com_velocity_at_release = magnitude(finite_diff(mean_of_all_keypoint_positions)) at release frame
   - Proficient: 0.54 m/s (lower is better)
   - mean over all 69 keypoints; divide frame diff by (1/60)

3. com_peak_velocity = max(magnitude(finite_diff(mean_keypoints))) over PP-to-RP window
   - Proficient: 0.87 m/s (lower is better)

4. release_height_normalized = wrist_y_at_release / player_height
   - Proficient: 1.17

5. knee_peak_angular_velocity = max(abs(diff(knee_angle_timeseries))) * 60
   - Proficient: 212.9 deg/s (lower is better)

6. knee_mean_angular_velocity = (knee_angle_RP - knee_angle_PP) / (n_frames_PP_to_RP / 60)
   - Proficient: 123.0 deg/s (lower is better)

7. elbow_angle_at_release = elbow angle at frame 153
   - Reference: 159.6 deg (near full extension)

8. All joint ROM (PP to RP) for knee, hip, ankle, elbow
   - Already partially implemented in existing joint angle features

**Group B - Li et al. CAV features (cross-shot variability per player):**

9. shoulder_elbow_CAV_per_player = apply CAV formula across all shots for that player
   - Input: shoulder angle timeseries + elbow angle timeseries, normalized to 0-100%
   - Output: one CAV value per player (player-level feature)
   - Higher CAV in skilled players at medium distance

10. dominant_coordination_pattern = mode of 8-category classification per shot
    - Proximal-dominance (shoulder) = elite pattern especially at long distance
    - In-phase shoulder flexion/elbow extension = common in both groups at medium distance

11. coupling_angle_at_release = gamma_bar in last 20% of shooting phase
    - Captures the coordination state at ball release

12. coupling_angle_shift = gamma_bar(first_50%) - gamma_bar(last_50%)
    - Elite: starts proximal-dominant, shifts to in-phase/distal at release

**Group C - Chen et al. kinematic proxies (approximations without force plates):**

13. wrist_angular_impulse_proxy = integral(abs(wrist_angular_velocity)) over release window
    - Proxy for AI. Experienced = much higher (~3.75x).
    - Window: last 30% of shot (flexion-to-extension switch to release)

14. elbow_angular_impulse_proxy = integral(abs(elbow_angular_velocity)) over release window
    - Experienced = ~1.31x higher

15. knee_peak_power_proxy = max(knee_angular_velocity * knee_angular_acceleration) over push-off
    - Proxy for P_peak. Need second derivative of knee angle.
    - Experienced = ~1.25x higher

16. VV_HV_ratio_at_release = wrist_vertical_velocity / wrist_horizontal_velocity at release frame
    - Proxy for VV/HV. Experienced = ~1.35 vs 1.31 for novice.
    - Use wrist/fingertip velocity since we don't have ball position

17. shoulder_RTD_proxy = max(second_diff(shoulder_angle)) * 60^2 in preparatory phase
    - Proxy for RTD. Paradoxically HIGHER in novices -> this could be a useful feature.

### Priority Ranking for Implementation

Priority 1 (high signal, easy to compute):
- trunk_lean_at_release (Cabarkapa d=0.880)
- com_peak_velocity (Cabarkapa d=0.988)
- knee_peak_angular_velocity (Cabarkapa d=1.037)
- wrist_angular_impulse_proxy (Chen eta_p^2=0.54)

Priority 2 (moderate signal):
- release_height_normalized (Cabarkapa d=0.438)
- elbow_angular_impulse_proxy (Chen eta_p^2=0.18)
- knee_peak_power_proxy (Chen eta_p^2=0.13)
- VV_HV_ratio proxy (Chen, group+distance effects)

Priority 3 (cross-shot features, more complex):
- CAV per player (Li et al., requires circular statistics)
- Coordination pattern classification (Li et al., requires 8-bin polar classification)
- Coupling angle at release (Li et al.)

### Caveats

- Chen et al. variables (RTD, P_peak, AI) require inverse dynamics + force plates for true values.
  Our proxies from kinematics only are approximations. They may still carry signal.
- Li et al. CAV is a cross-shot variability measure: it tells us about player consistency
  across shots, not within-shot. This is a player-level feature, not shot-level.
  We have ~70 shots per player which is enough to compute it.
- Cabarkapa used markerless mocap at 120 Hz on 34 subjects. Our data is similar setup.
  Their variable definitions map directly to our keypoint data.
- All angular velocity features from Cabarkapa were measured at 120 Hz; we have 60 Hz.
  This halves our max detectable peak velocity (Nyquist). Peak velocity estimates
  will be underestimates but relative differences between players should hold.
