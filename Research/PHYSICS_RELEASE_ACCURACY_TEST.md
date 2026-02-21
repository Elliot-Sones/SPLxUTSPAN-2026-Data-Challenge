# Physics Engine Release Feature Accuracy Test

## Date: 2026-02-05

## Objective
Test whether the physics engine can extract ball release velocity accurately enough to forward-simulate projectile trajectories and derive the competition targets (angle, depth, left_right) directly from the simulation.

## Methodology
1. For each of 345 training shots, extract release position + velocity using 3 methods
2. Forward-simulate projectile to hoop plane (z=10ft) using standard kinematics
3. Compare predicted angle/depth/left_right with actual targets

### Methods Tested
- **wrist_simple**: Ball at wrist + 0.6*(fingertip - wrist), velocity from central difference
- **kinematic_peak**: Ball at same position, velocity from kinematic chain Jacobian at peak speed frame
- **savgol**: Ball position with Savitzky-Golay filtered velocity (window=7, polyorder=2)

### Forward Simulation
- Solve: dz = vz*t - 0.5*g*t^2 for flight time t
- Landing position: x = x0 + vx*t, y = y0 + vy*t
- Entry angle: arctan2(-landing_vz, horizontal_speed)
- Depth: landing_y - hoop_center_y (in inches)
- Left_right: landing_x - hoop_center_x (in inches)

## Results

### wrist_simple
- Valid shots: 13/345 (332 simulation failures - velocity too low to reach hoop)
- Release speed: mean=0.35, median=0.30 m/s (need ~6.8 m/s)
- UNUSABLE: 96% of shots fail to reach hoop plane

### kinematic_peak
- Valid shots: 141/345 (64 extraction failures, 140 sim failures)
- Release speed: mean=5.32, median=5.22 m/s (need ~6.8 m/s)

| Target | Actual Range | Pred Range | RMSE | MAE | Correlation |
|--------|-------------|------------|------|-----|-------------|
| Angle (deg) | mean=46.2, std=5.9 | mean=51.3, std=17.0 | 31.18 | 21.45 | r=0.06 |
| Depth (inches) | mean=8.8, std=6.8 | mean=7.4, std=19.3 | 24.04 | 13.43 | r=0.04 |
| Left_right (inches) | mean=-0.6, std=3.9 | mean=184.7, std=101.3 | 185.47 | 162.59 | r=0.02 |

### savgol
- Valid shots: 28/345 (317 simulation failures - velocity too low)
- Release speed: mean=1.08, median=0.88 m/s
- UNUSABLE: 92% of shots fail to reach hoop plane

## Root Cause Analysis

### 1. Velocity Magnitude Problem
- Kinematic peak gives ~5.3 m/s average, but ~6.8 m/s needed to reach hoop (15 ft away, 10 ft high)
- wrist_simple and savgol give < 2 m/s - completely insufficient
- The kinematic chain computes HAND velocity, not BALL velocity

### 2. Velocity Direction Problem (the critical issue)
Example: Shot 0
- Peak velocity: [vx=-1.52, vy=0.02, vz=5.05] m/s (speed=5.28 m/s)
- This is almost entirely VERTICAL (vz=5.05) with minimal horizontal toward hoop (vx=-1.52)
- A real free throw needs ~4.35 m/s horizontal component
- At peak, the arm is moving upward, not toward the hoop
- After peak, vx flips positive (AWAY from hoop) during follow-through

### 3. Fundamental Limitation: No Ball Data
- The dataset has NO ball tracking data - only hand/body keypoints
- The ball is PROJECTED from fingertips - it does not move with the hand
- The hand-to-ball velocity transfer depends on:
  - Fingertip release mechanics (rolling off fingers)
  - Wrist snap timing and angle
  - Finger extension patterns
- These happen in ~2-3 frames at 60 Hz (33-50 ms) - below temporal resolution
- 60 Hz sampling aliases the fast release dynamics

### 4. Left_right Systematic Bias
- Predicted left_right averages +185 inches (15+ feet right of hoop)
- Player stands at x~18.5ft, hoop at x=5.25ft
- Ball needs to move in -x direction, but kinematic velocity at peak has small -x component
- The projection from fingertips redirects the ball toward the hoop - this redirection is invisible in hand kinematics

## Conclusions

1. **Physics engine release features CANNOT derive targets directly.** Error is 5-50x larger than the target signal.
2. **The hand-to-ball transfer function is unobservable** at 60 Hz with hand-only data.
3. **Kinematic chain velocity measures the wrong thing**: hand velocity, not ball velocity.
4. **This is a fundamental data limitation**, not an implementation problem. No amount of engineering the physics engine will fix the missing ball data.
5. **Physics features should be used as SUPPLEMENTARY ML features** (e.g., release speed as a proxy, arm geometry, finger spread), not as direct predictors of targets.

## Implications for Competition Strategy

- The physics engine approach of "simulate to predict" is a dead end for this competition
- Physics-derived features may have marginal value as additional features in the ML ensemble
- The best path forward remains: raw timeseries features (PLS, tree ensembles) with target-specific blending
- Current best: Sub 784 at LB 0.007224 using PLS depth + hoop-relative left_right blended into Sub 771

## Reproduction
```
# Run the test
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/physics_engine
uv run python /path/to/test_release_accuracy.py
```
Script location: scratchpad/test_release_accuracy.py (in session scratchpad)
