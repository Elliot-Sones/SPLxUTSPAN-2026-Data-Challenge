# Physics Feature Extraction Results

## Summary

Successfully extracted physics-based features from mocap data using kinematic chain analysis. These features show meaningful correlations with target outcomes.

## Key Findings

### 1. Velocity Profile During Shooting Motion

The wrist/hand velocity during a basketball shot shows a characteristic pattern:
- **Shot start**: Velocity increases as arm moves upward
- **First peak**: Around frames 100-115, velocity reaches ~2.0-2.5 m/s
- **Deceleration**: Velocity drops as arm decelerates
- **Second peak**: Around frames 140-150 (just before release), velocity reaches ~1.5-2.0 m/s
- **Release**: At the release frame (wrist peak height), velocity is near zero (~0.3 m/s)

**Key insight**: The ball's release velocity is NOT the instantaneous velocity at release. It's the momentum accumulated just before the ball separates from the hand (the "pre-release velocity").

### 2. Feature Correlations with Target Angle

| Feature | Correlation |
|---------|-------------|
| pre_release_vx | +0.638 |
| release_vx | +0.608 |
| release_speed | +0.608 |
| pre_release_speed | +0.590 |
| peak_to_release | -0.563 |
| pre_release_vz | +0.529 |
| release_vy | -0.481 |
| peak_vx | +0.459 |
| wrist_velocity_speed | +0.448 |

### 3. Feature Correlations with Depth

Weak correlations (max ~0.14). Depth is harder to predict from kinematics alone.

| Feature | Correlation |
|---------|-------------|
| pre_release_angle_deg | -0.141 |
| peak_vy | +0.134 |
| finger_spread | +0.127 |

### 4. Feature Correlations with Left/Right

Very weak correlations (max ~0.13). Lateral aim is difficult to predict.

| Feature | Correlation |
|---------|-------------|
| finger_velocity_diff | +0.130 |
| fingertip_velocity_speed | +0.127 |

## Feature Statistics

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| peak_speed | 3.62 m/s | 1.48 | 0.00 | 5.58 |
| pre_release_speed | ~2.0 m/s | - | - | - |
| release_speed | 0.72 m/s | 0.68 | 0.00 | 2.69 |
| elbow_angle_deg | 159.84 | 31.92 | 0.00 | 174.84 |
| forearm_elevation_deg | 57.92 | 12.24 | 0.00 | 72.24 |
| upper_arm_elevation_deg | 46.93 | 10.16 | 0.00 | 59.67 |

## Physics Model Implementation

### Kinematic Chain Analysis

Ball velocity is computed using the Jacobian approach:
```
v_ball = Σ(ω_joint × r_joint_to_ball)
```

Where:
- ω_joint = angular velocity of each joint (shoulder, elbow, wrist)
- r_joint_to_ball = vector from joint to ball position

### Key Components

1. **Joint Position Tracking**: Track shoulder, elbow, wrist, fingertip positions over time
2. **Velocity Computation**: Central difference for smooth velocity estimates
3. **Angular Velocity**: Computed from parent-child joint velocity differences
4. **Ball Position**: Approximated as point between wrist and fingertip center

### Feature Extraction Pipeline

1. Find release frame (wrist peak height)
2. Find shot start (local minimum before release)
3. Compute kinematic velocities over shooting motion
4. Find global peak and pre-release peak velocities
5. Extract joint contributions at pre-release
6. Compute arm geometry features

## Implications for ML Model

### Strong Predictors for Angle
- Use pre_release_vx, pre_release_speed, pre_release_vz as primary features
- Include joint contribution percentages
- Include arm geometry (forearm/upper arm elevation)

### Weak Predictors for Depth/Left-Right
- Depth and left_right require additional features beyond basic kinematics
- May need:
  - Ball position relative to player center
  - Lateral velocity components
  - Body orientation/rotation features

## Files Created

- `core/physics_features.py`: Feature extraction implementation
- `scripts/analyze_physics_features.py`: Correlation analysis script
- `scripts/debug_velocity_computation.py`: Velocity debugging script

## Next Steps

1. Use physics features in ML ensemble
2. Investigate additional features for depth/left_right prediction
3. Consider temporal features (velocity profiles over time)
4. Test feature importance with tree-based models
