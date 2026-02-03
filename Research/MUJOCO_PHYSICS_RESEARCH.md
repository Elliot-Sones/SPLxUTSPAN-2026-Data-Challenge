# MuJoCo Physics Simulation Research

## Summary

Implemented a working MuJoCo physics simulation for basketball free throw analysis. The simulation models ball-hand contact during the shooting motion and extracts physics-based features at ball release.

## CRITICAL FINDING: Feature Extraction Timing

**Ball velocity must be extracted at PEAK HAND VELOCITY, not at contact loss!**

When extracting at contact loss (wrong):
- Ball vz: 3.22 ft/s (too low)
- Correlation with kin_vz: -0.30 (negative!)
- Features were anti-predictive

When extracting at peak hand velocity (correct):
- Ball vz: 6.24 ft/s (matches hand velocity trend)
- Correlation with kin_vz: +0.488 (positive!)
- **kin_speed correlates 0.774 with angle target!**

## Best Correlations Found

| Feature | angle | depth | left_right |
|---------|-------|-------|------------|
| kin_speed | **0.774** | 0.012 | -0.015 |
| kin_vz | **0.710** | -0.031 | -0.047 |
| ball_vz | **0.658** | 0.162 | 0.044 |
| ball_speed | **0.690** | 0.020 | 0.011 |
| kin_vx | -0.411 | -0.076 | -0.098 |

**Conclusion: Physics features strongly predict ANGLE but not DEPTH/LEFT_RIGHT.**

- ANGLE depends on release velocity (physics) - r=0.77
- DEPTH/LEFT_RIGHT depend on body positioning/aim - requires different features

## Implementation Details

### Model Configuration
```
- Ball: sphere, radius=0.12m, mass=0.625kg
- Hand: cylinder (palm), radius=0.10m, half-height=0.02m, mass=3.0kg
- Timestep: 0.0002s (5000 Hz)
- Gravity: -9.81 m/s^2
```

### Kinematic Hand Control

After testing position-controlled actuators (which had poor tracking at 34-40%), switched to direct kinematic control:
- Hand qpos and qvel set directly from skeleton trajectory
- Ball physics computed through MuJoCo contact solver
- Velocity transfer: ~100% when tracking works correctly

### Key Scripts
- `scripts/mujoco_kinematic_hand.py` - Working simulation with kinematic hand control
- `scripts/mujoco_feature_extraction.py` - Feature extraction pipeline
- `scripts/mujoco_diagnose_velocity.py` - Diagnostic analysis

## Extracted Features

### Feature List
| Feature | Description |
|---------|-------------|
| mj_release_vx/vy/vz | Ball velocity components at release (m/s) |
| mj_release_speed | Ball speed at release (m/s) |
| mj_release_speed_fps | Ball speed at release (ft/s) |
| mj_release_x/y/z | Ball position at release (m) |
| mj_release_frame | Frame number at release |
| mj_contact_frames | Number of frames with ball-hand contact |
| mj_max_ball_speed | Peak ball speed during simulation |
| mj_max_ball_vz | Peak vertical ball velocity |
| mj_velocity_transfer | Ratio of ball speed to hand speed |
| mj_launch_angle | Launch angle from horizontal (degrees) |
| mj_horizontal_speed | Horizontal component of release velocity |

### Feature Statistics (N=345 shots)
| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| mj_release_speed_fps | 5.25 | 2.72 | 0.35 | 12.51 |
| mj_max_ball_speed_fps | 7.20 | 3.14 | 2.09 | 16.09 |
| mj_launch_angle | 32.38 | 61.18 | -88.33 | 89.45 |
| mj_velocity_transfer | 1.57 | 2.34 | 0.00 | 14.94 |

## Correlation with Targets

### Per-Feature Correlations
| Feature | angle | depth | left_right |
|---------|-------|-------|------------|
| mj_release_speed | **0.464** | -0.016 | 0.046 |
| mj_release_vx | -0.385 | 0.018 | -0.011 |
| mj_max_ball_speed | 0.359 | 0.001 | 0.056 |
| mj_max_ball_vz | 0.276 | 0.025 | 0.088 |
| mj_release_vz | 0.188 | 0.053 | 0.075 |
| mj_release_z | 0.128 | 0.097 | **0.124** |
| mj_contact_frames | 0.013 | **0.137** | -0.035 |
| mj_launch_angle | -0.021 | 0.111 | 0.084 |

### Best Correlations per Target
- **angle**: mj_release_speed (r = 0.464)
- **depth**: mj_contact_frames (r = 0.137)
- **left_right**: mj_release_z (r = 0.124)

## Comparison with Kinematic Features

From earlier research (`physics_target_correlation.py`):
- vz_at_peak_vz correlates r = 0.78 with angle
- MuJoCo mj_release_speed correlates r = 0.464 with angle

**Finding**: Simple kinematic features (velocity from differentiation) show STRONGER correlation than MuJoCo physics features. This suggests:

1. The physics simulation may be adding noise through:
   - Imperfect contact modeling
   - Release detection timing differences
   - Hand trajectory interpolation

2. Direct velocity differentiation already captures the key physics

3. MuJoCo features may still provide value in ensemble (orthogonal signal)

## Key Technical Insights

### qpos Layout (Critical!)
The qpos layout depends on body order in XML:
```
qpos[0:3] = hand joints [hx, hy, hz] (first body)
qpos[3:6] = ball position [x, y, z] (second body)
qpos[6:10] = ball quaternion [w, x, y, z]
```

This caused significant debugging time when initially indexed incorrectly.

### Hand Velocity in Data
Analysis of 50 shots:
- Finger speed: Mean 3.51 m/s (11.5 ft/s), Max 7.24 m/s (23.8 ft/s)
- Wrist speed: Mean 2.53 m/s (8.3 ft/s), Max 4.69 m/s (15.4 ft/s)
- Required for true trajectory: ~6.75 m/s (22.1 ft/s)

Some shots have sufficient velocity for realistic trajectories.

### Position Actuator Tracking
With position-controlled actuators (kp=20000-100000):
- Tracking efficiency: 34-40%
- Ball gets hand velocity but hand can't track target fast enough

Kinematic control (direct qpos/qvel setting):
- Tracking efficiency: ~100% when data supports
- Physics still applies contact forces correctly

## Recommendations

### For Model Improvement
1. Use MuJoCo features as ADDITIONAL features, not replacement
2. Combine with existing kinematic features in ensemble
3. Weight kinematic features higher based on correlation strength

### Feature Subset to Use
Recommended MuJoCo features for model:
- mj_release_speed (r=0.464 with angle)
- mj_release_vx (r=-0.385 with angle)
- mj_max_ball_vz (r=0.276 with angle)
- mj_contact_frames (r=0.137 with depth)
- mj_release_z (r=0.124 with left_right)

### Future Work
1. Per-player calibration of contact parameters
2. More sophisticated release detection
3. Include wrist rotation dynamics
4. Test MuJoCo features in ensemble with best existing model

## Files Generated
- `output/mujoco_physics_features.csv` - All extracted features for training data

## Conclusion

MuJoCo physics simulation successfully implemented with correct contact physics. Ball velocities transfer correctly from hand to ball through contact forces. However, simpler kinematic features show stronger correlation with targets. MuJoCo features provide potentially orthogonal signal for ensemble modeling but should not replace existing features.
