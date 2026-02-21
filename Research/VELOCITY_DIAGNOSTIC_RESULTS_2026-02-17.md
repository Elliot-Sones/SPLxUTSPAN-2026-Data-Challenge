# Velocity Diagnostic Results (2026-02-17)

## Context
After building a sparse physics pipeline (24 features, 3 phases), validation showed ALL
correlations with targets near zero. This diagnostic investigated WHY.

## Root Causes Found

### 1. Force-Based Release Detection Finds WRONG Frame
- Force detection uses d2x/dt2 of ball proxy to find when non-gravitational force drops
- Result: peak force frame at ~163 (mean), where hand is moving BACKWARD and DOWNWARD
- v_forward = -0.62 m/s (backward!), v_up = -2.36 m/s (downward!)
- Detection finds follow-through deceleration, NOT the actual release
- **CONCLUSION: Force-based release detection is fundamentally broken for this data**

### 2. Ball Proxy Dampens Velocity
- ball = wrist + 0.6 * (fingertip - wrist) is a blend that averages fast/slow motions
- Ball proxy peak speed: 3.93 m/s vs fingertip: 4.38 m/s vs wrist: 3.43 m/s
- 10% velocity loss from the blending

### 3. 60fps Markerless Mocap Has Low Velocity Resolution
- Expected free throw release speed: 6.5-7.5 m/s
- Measured peak fingertip speed: 4.38 m/s (40% low)
- Raw (unsmoothed) barely better: 3.52 m/s (first 50 shots)
- Smoothing window has negligible effect: 3.48 (w=5) vs 3.37 (w=21)
- **Root cause: the mocap keypoints don't track fast motion accurately at 60fps**
- Positional error ~1-2 cm at 60fps -> velocity noise floor ~0.6-1.2 m/s

### 4. Position Data IS Correct
- Wrist height at f150: 7.73 +/- 0.08 feet (expected ~7-8 feet) - CORRECT
- Distance to hoop at f150: 13.19 +/- 0.11 feet (expected ~13-15 feet) - CORRECT
- No units error. Data is in feet as documented.

## Key Discovery: Fixed-Frame Velocities DO Have Signal

### Global Correlations (all 345 shots)
| Frame | Wrist speed -> angle | Wrist speed -> depth | Wrist v_lat -> LR |
|-------|---------------------|---------------------|-------------------|
| 140   | r = -0.51           | r = -0.32           | r = -0.08         |
| 150   | r = -0.53           | r = +0.15           | r = -0.10         |
| 153   | r = -0.53           | r = +0.05           | r = -0.04         |
| 170   | r = +0.05           | r = -0.19           | r = -0.05         |

Strong negative r for angle makes physical sense: faster wrist -> flatter trajectory -> lower angle at rim.

### Per-Player Correlations (THE KEY METRIC for our per-player model)

**DEPTH (strongest signal):**
| Joint | Frame | Component | Per-player r |
|-------|-------|-----------|-------------|
| left_wrist | 150 | v_up | +0.4629 |
| right_wrist | 153 | v_up | +0.4356 |
| right_elbow | 153 | v_up | +0.4303 |
| neck | 153 | v_up | +0.4270 |
| right_wrist | 140 | v_fwd | -0.4167 |

Physical: More upward velocity at release -> ball has more energy -> goes deeper into hoop.

**LEFT_RIGHT (moderate signal):**
| Joint | Frame | Component | Per-player r |
|-------|-------|-----------|-------------|
| neck | 170 | v_lat | +0.3668 |
| left_shoulder | 170 | v_lat | +0.3060 |
| right_shoulder | 170 | v_lat | +0.2933 |
| mid_hip | 170 | v_lat | +0.2319 |

Physical: Body lateral drift during follow-through -> ball lateral displacement.

**ANGLE (weak signal):**
| Joint | Frame | Component | Per-player r |
|-------|-------|-----------|-------------|
| right_knee | 153 | speed | +0.1792 |
| right_knee | 155 | speed | +0.1752 |
| neck | 145 | v_lat | -0.1650 |

Not much per-player velocity signal for angle.

## Validated Velocity Pipeline Results

### LOO Performance
- angle: 0.009182
- depth: 0.009841
- left_right: 0.009461
- **mean: 0.009495** (compare: baseline 0.006830, sparse physics 0.013831)

### Diversity vs Sub 3190
- angle: r = 0.8578
- **depth: r = 0.6937** (strong diversity)
- **LR: r = 0.7062** (strong diversity)

### Submissions Generated
- Sub 3204: standalone
- Sub 3205: 2% blend with Sub 3190
- Sub 3206: 5% blend with Sub 3190
- Sub 3207: 10% blend with Sub 3190
- Sub 3208: per-target weighted (angle 2%, depth 7%, LR 5%) + Sub 3190

## Scripts
- scripts/physics_velocity_diagnostic.py - Full diagnostic
- scripts/velocity_per_player_signal.py - Per-player correlation sweep
- scripts/validated_velocity_pipeline.py - Production pipeline

## Key Lesson
**NEVER use data-driven phase detection without validation.** The force-based release
detection produced zero-correlation features despite having a correct physical framework.
Fixed-frame features at known good frames (150-153 for release, 170 for follow-through)
have genuine, physically meaningful per-player correlations.
