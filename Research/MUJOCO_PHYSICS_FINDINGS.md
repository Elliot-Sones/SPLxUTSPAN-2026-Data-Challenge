# MuJoCo Physics Simulation Findings

## Summary

We built a research-grade MuJoCo basketball physics simulation to predict shot outcomes from skeleton data. The simulation uses accurate NBA basketball parameters and aims to simulate ball trajectories from extracted release parameters.

**Result: The physics simulation did NOT improve predictions.**

CV MSE with simulation: 0.079
CV MSE without simulation (direct features): 0.027
Target: < 0.007
Current best (Sub 219): 0.008305

## What We Built

### MuJoCo Scene (`physics_engine/models/basketball_court.xml`)
- NBA regulation basketball: mass=625g, diameter=24cm (0.12m radius)
- 8-segment capsule rim for accurate collision
- RK4 integrator, 1ms timestep
- Hoop at Z=3.05m (10 feet), 15cm forward of backboard

### Release Parameter Extraction
1. **Scale Calibration**: Using forearm, shin, upper arm lengths
   - Computed scale: ~0.29 m/unit
   - Validated: produces 2.0m release height (correct)

2. **Release Frame Detection**: Player-specific windows
   - Player 1: frames 110-165
   - Player 2: frames 80-195
   - Player 3: frames 60-220
   - Player 4: frames 125-195
   - Player 5: frames 65-200

3. **Velocity Extraction**: Savitzky-Golay differentiation
   - window=7, polyorder=3
   - At 60fps gives smooth velocity estimate

4. **Coordinate Transformation**:
   - Data Y = lateral (shoulder width in Y axis)
   - Data Z = vertical (ankle to nose in Z axis)
   - Data X = forward/backward

## Key Problems Discovered

### 1. Data Velocities Are Too Small
- Raw wrist velocity: ~1-4 units/second
- After scaling (x0.29): ~0.3-1.2 m/s
- Required for free throw: 7-9 m/s
- **Gap: velocities are ~6x too slow**

This appears to be a data normalization artifact. Position scales correctly (forearm = 1 unit = 0.265m), but velocities do not scale the same way.

### 2. Extracted Direction Doesn't Predict Targets

| Metric | Correlation with target_angle |
|--------|------------------------------|
| landing_z | r = -0.38 (moderate) |
| entry_angle | r = 0.45 (moderate) |
| vel_z | r = 0.19 (weak) |

The physics outputs have weak correlations with targets because:
- Wrist motion ≠ ball release velocity
- Ball is released by fingertips, not wrist
- The targets describe OUTCOME, not just trajectory

### 3. Simulation Adds Noise

| Approach | CV MSE |
|----------|--------|
| Physics simulation | 0.079 |
| Direct physics features | 0.027 |
| Baseline (Sub 219) | 0.008 |

The MuJoCo simulation increases error because:
- Forcing velocity to reach hoop removes variation
- Preserving variation causes many shots to miss hoop
- The mapping from landing position to targets is weak

## Physics Features That Work

Direct features without simulation (CV MSE = 0.027):

```python
features = [
    pos[1],  # Lateral position (Y)
    pos[2],  # Release height (Z)
    vel[0],  # Forward velocity (vx)
    vel[1],  # Lateral velocity (vy)
    vel[2],  # Vertical velocity (vz)
    np.linalg.norm(vel),  # Total speed
    np.arctan2(vel[2], vel[0]),  # Release angle
    np.arctan2(vel[1], vel[0]),  # Lateral angle
    vel[2] / vel[0],  # vz/vx ratio
    vel[1] / vel[0],  # vy/vx ratio
    pos[2] * vel[2],  # Height x vertical velocity
    release_frame,
    backspin,
]
```

Training R-squared:
- angle: 0.39 (good signal)
- depth: 0.08 (weak)
- left_right: 0.07 (weak)

## Submissions Created

| Submission | Description | CV MSE | Notes |
|------------|-------------|--------|-------|
| 575 | Full physics simulation v1 | ~0.05 | 12% success rate |
| 579 | Fixed coordinates | ~0.02 | 72% success rate |
| 583 | Direct physics features | 0.027 | No simulation |
| 584-587 | Blends with Sub 219 | - | 5-20% physics weight |

## Why MuJoCo Physics Didn't Work

1. **Data Limitation**: Wrist position/velocity ≠ ball release parameters
   - The ball leaves the fingertips, not the wrist
   - Finger snap adds significant velocity at release

2. **Target Definition**: The targets describe shot QUALITY, not just trajectory
   - angle: entry angle at hoop (affected by spin, arc)
   - depth: long/short (affected by speed, not just direction)
   - left_right: lateral deviation

3. **Missing Information**:
   - Ball position during shot (we estimate from hand)
   - Actual release timing (we estimate from wrist peak velocity)
   - Spin magnitude and axis (we estimate from finger differential)

## Recommendations

1. **Don't use full trajectory simulation** - adds noise
2. **Use physics-informed features directly**:
   - Release angle (arctan(vz/vx))
   - Lateral velocity ratio (vy/vx)
   - Release height
   - Release timing (frame number)

3. **Blend with existing models**:
   - 5-10% physics features + 90-95% baseline
   - Physics features have 80% correlation with Sub 219
   - May provide small complementary signal

4. **Future work**:
   - Track ball position directly (if visible in data)
   - Use fingertip motion for release velocity
   - Per-player calibration of velocity scaling

## Files Created

```
physics_engine/
    __init__.py
    models/
        basketball_court.xml     # MuJoCo scene
    core/
        __init__.py
        simulator.py             # MuJoCo simulation
        scale_calibration.py     # Skeleton to meters
        release_extraction.py    # Release parameters
        target_mapping.py        # Physics to targets
scripts/
    physics_ball_simulation.py   # Full pipeline
    physics_diagnostics.py       # Diagnostic analysis
    physics_deep_analysis.py     # Deep correlation analysis
    physics_features_direct.py   # Direct features (no sim)
```

## Conclusion

The MuJoCo physics simulation approach did not beat the baseline. While the physics infrastructure is correctly built (accurate NBA ball, proper integrator, validated trajectories), the extracted release parameters from skeleton data don't accurately capture the ball's actual release conditions. The wrist motion is a proxy for ball release, not the actual release, and this proxy error dominates the physics signal.

The physics-informed features (velocity direction, release timing) do provide some signal (angle R²=0.39), but running the full trajectory simulation adds noise rather than improving predictions.
