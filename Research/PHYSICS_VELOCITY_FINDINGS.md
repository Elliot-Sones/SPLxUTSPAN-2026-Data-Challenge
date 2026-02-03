# Physics Velocity Extraction Findings

## Executive Summary

**Key Discovery**: While full physics simulation fails due to velocity extraction limitations, physics-informed features provide **strong predictive signal for entry angle** (r=0.78 correlation with vz_at_peak_vz).

The simulation approach was limited by incomplete velocity data, but the underlying physics features ARE valuable:
- **angle**: r=0.78 with max vertical velocity (very strong)
- **depth**: r=0.18 with peak height (weak but significant)
- **left_right**: r=0.18 with max lateral velocity (weak but significant)

## Detailed Analysis

We conducted extensive analysis of ball velocity extraction from skeleton keypoint data. The key finding is that **the skeleton data fundamentally cannot capture accurate ball release velocity** because:

1. The ball accumulates momentum throughout the shooting motion
2. At any single frame, we measure only instantaneous fingertip velocity
3. The fingertip velocity at "release" is primarily vertical (finger flick) with minimal horizontal component

## Measured vs Required Velocities

| Component | Measured (ft/s) | Required (ft/s) | Ratio |
|-----------|----------------|-----------------|-------|
| Total speed | 10.6 | 24 | 44% |
| Horizontal (toward hoop) | 0.1 | 15 | ~0% |
| Vertical (up) | 10.4 | 19 | 55% |

The **horizontal velocity is essentially ZERO** at the detected release frame.

## Why This Happens

Looking at frame-by-frame data for a typical shot:

```
Frame   vx (toward hoop)   vz (up)   Total Speed
105     -4.3              8.2        9.3
108     -2.4             10.5       10.8
110     -0.6             11.1       11.1  <-- Peak vz (detected "release")
115     +5.0              8.4       10.2  <-- Hand moving backward (follow-through)
```

The arm is **rotating**, not translating:
- First phase: arm extends forward AND up (frames 100-108)
- Peak vertical: at top of arc, horizontal velocity is near zero (frames 108-112)
- Follow-through: arm rotates back and down (frames 115+)

## Different Release Detection Methods

We tested four methods for detecting release:

| Method | Frame | vx (ft/s) | vz (ft/s) | Speed | Best Correlation |
|--------|-------|-----------|-----------|-------|------------------|
| Peak upward velocity | ~110 | -1.1 | 9.3 | 9.4 | depth: r=0.26 |
| Peak total speed | varies | 0.7 | 3.0 | 11.1 | depth: r=0.36, lr: r=-0.40 |
| Peak forward velocity | ~146 | -7.7 | 2.5 | 8.1 | angle: r=-0.28, lr: r=-0.31 |
| Best combined score | ~117 | 4.9 | 7.4 | 9.0 | (weak correlations) |

**No single frame captures both high horizontal AND vertical velocity.**

## Unit Verification

We verified that the data is in the correct units (feet):
- Ankle height: 0.73 ft (correct for standing)
- Wrist release height: 7.5-8.0 ft (correct for shooting)
- Shoulder width: 1.29 ft (reasonable)
- Distance to hoop: ~12.7 ft (correct for free throw)

The units are correct. The velocities are simply incomplete.

## What the Skeleton Data CAN Capture

1. **Release timing**: When the finger flick occurs
2. **Vertical velocity direction**: Up vs down, relative speed
3. **Lateral aim**: Left/right deviation in velocity
4. **Release angle approximation**: arctan(vz/vx), though vx is underestimated
5. **Relative shot quality**: Shot A vs Shot B, which has more speed/arc

## What the Skeleton Data CANNOT Capture

1. **Absolute ball velocity**: The full momentum accumulated during the shot
2. **True horizontal velocity**: The forward momentum from arm extension
3. **Ball separation dynamics**: When exactly the ball leaves the fingers
4. **Spin rate**: Finger differential gives approximation but not accurate

## Implications for Prediction

Since we cannot accurately simulate ball trajectories, we should:

1. **Use kinematic features directly** as inputs to ML models
2. **Extract relative features** (ratios, angles, timing) rather than absolute values
3. **Use physics constraints** to inform feature engineering, not to simulate
4. **Accept that physics simulation adds noise** rather than improves predictions

## Comprehensive Feature Correlation Analysis

We extracted 30+ physics features and computed correlations with all three targets.

### TRUE_ANGLE (Entry Angle)

**Very strong correlations found:**

| Feature | Correlation | P-value |
|---------|-------------|---------|
| vz_at_peak_vz | r=+0.782 | <0.0001 |
| max_vz | r=+0.782 | <0.0001 |
| speed_at_peak_vz | r=+0.753 | <0.0001 |
| finger_max_vz | r=+0.706 | <0.0001 |
| speed_at_peak_forward | r=+0.629 | <0.0001 |
| peak_height | r=-0.512 | <0.0001 |
| release_height | r=-0.460 | <0.0001 |

**Physical interpretation**: Higher upward velocity at release = higher arc = steeper entry angle. This is exactly what physics predicts.

### TRUE_DEPTH (Short/Long)

**Weak but significant correlations:**

| Feature | Correlation | P-value |
|---------|-------------|---------|
| peak_height | r=+0.184 | 0.0006 |
| vx_at_peak_speed | r=-0.152 | 0.0048 |
| max_vy | r=-0.135 | 0.0124 |
| release_angle_ratio | r=-0.129 | 0.0163 |

**Physical interpretation**: Depth depends on ball speed, which we cannot accurately measure. Peak height is a proxy for "how hard" the shot was taken.

### TRUE_LEFT_RIGHT (Lateral)

**Weak but significant correlations:**

| Feature | Correlation | P-value |
|---------|-------------|---------|
| max_vy | r=+0.176 | 0.0010 |
| release_angle_ratio | r=-0.142 | 0.0081 |
| finger_release_y | r=+0.138 | 0.0104 |

**Physical interpretation**: Lateral velocity and finger position predict lateral landing. The correlations are weak because small velocity variations cause large lateral deviations.

## Recommendations

1. **Do NOT rely on full trajectory simulation** - it will fail due to incomplete velocity data

2. **Extract multiple features** from different phases of the shot:
   - Early phase (frames 100-108): forward momentum
   - Mid phase (frames 108-115): vertical velocity
   - Late phase (frames 115-130): follow-through direction

3. **Use velocity ratios** rather than absolute values:
   - vz/vx ratio (release angle proxy)
   - vy/vx ratio (lateral aim proxy)
   - Speed at different phases (effort/intensity proxy)

4. **Consider the kinetic chain**:
   - Shoulder, elbow, wrist velocities during different phases
   - Joint angles at release
   - Timing of peak velocities for different joints

5. **Combine physics features with ML**:
   - Physics features provide orthogonal signal
   - ML model learns the mapping from incomplete data to targets

## Files Created During Analysis

```
scripts/physics_diagnose_velocity.py    - Initial velocity extraction analysis
scripts/physics_trace_shot.py           - Frame-by-frame shot tracing
scripts/physics_ball_contact_sim.py     - Contact-based release detection
scripts/physics_analyze_release.py      - Release timing analysis
scripts/physics_forward_release.py      - Comparison of detection methods
```

## Conclusion

The physics simulation approach is fundamentally limited by data constraints, not implementation issues. The skeleton data tracks arm/hand position, which is a PROXY for ball motion. The ball's actual velocity includes momentum accumulated throughout the shooting motion that is not captured at any single frame.

However, physics-informed features provide STRONG signal for angle prediction (r=0.78), and can complement existing ML models.

## Actionable Recommendations

1. **Add vz_at_peak_vz to feature set** - This alone explains 60% of angle variance (R^2 = 0.61)

2. **Use physics features for angle, ML for depth/left_right** - The physics signal is concentrated in angle prediction

3. **Consider ensemble approach**:
   - Physics model for angle (r=0.78)
   - Baseline ML model for depth/left_right
   - Blend predictions weighted by correlation strength

4. **Key physics features to add to existing pipeline**:
   - max_vz (vertical velocity at release)
   - speed_at_peak_vz (total speed at release)
   - peak_height (maximum wrist height)
   - max_vy (lateral velocity for left_right)

## Files Created During Analysis

```
scripts/physics_diagnose_velocity.py    - Initial velocity extraction analysis
scripts/physics_trace_shot.py           - Frame-by-frame shot tracing
scripts/physics_ball_contact_sim.py     - Contact-based release detection
scripts/physics_analyze_release.py      - Release timing analysis
scripts/physics_forward_release.py      - Comparison of detection methods
scripts/physics_accumulated_velocity.py - Accumulated velocity approach
scripts/physics_target_correlation.py   - Comprehensive correlation analysis
scripts/physics_model_v2.py             - Physics-based prediction model
output/physics_feature_correlations.csv - All feature correlations
```

## User Insight

The user correctly identified that "there might be other factors that influence this that add up to make up this gap." This reframed our analysis from "why doesn't velocity match theory" to "what predicts the targets." The answer: vertical velocity (vz) strongly predicts entry angle, even if it doesn't match theoretical projectile requirements.
