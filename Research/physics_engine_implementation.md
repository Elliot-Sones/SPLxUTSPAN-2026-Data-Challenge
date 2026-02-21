# Physics-Based State Estimator - Implementation Summary

Date: 2026-02-03

## What Was Built

A complete physics simulation pipeline for extracting ball release parameters from mocap skeleton data.

### Core Components

1. **Data Loader** (`physics_engine/core/data_loader.py`)
   - Parses JSON timeseries from CSV
   - 69 keypoints x 3 coordinates = 207 features per frame
   - 240 frames per shot at 60 fps
   - Coordinate system: X toward hoop, Y lateral, Z vertical
   - Hoop at [5.25, -25, 10] feet

2. **MuJoCo Hand-Ball Contact Model** (`physics_engine/models/hand_ball_contact.xml`)
   - Mocap body for kinematic hand control
   - Connect constraint for ball attachment
   - Tunable friction parameters
   - Contact forces computable via constraint monitoring

3. **Inverse Dynamics Engine** (`physics_engine/core/inverse_dynamics.py`)
   - Computes contact forces during shooting motion
   - Detects release frame via slip ratio (tangential/normal force)
   - Coulomb friction model for release detection

4. **Release State Extraction** (`physics_engine/core/release_state.py`)
   - Extracts ball position from fingertip positions
   - Estimates velocity from hand motion (scaled to realistic range)
   - Estimates backspin from finger-wrist differential

5. **Ball Trajectory Simulator** (`physics_engine/core/simulator.py`)
   - MuJoCo-based projectile simulation
   - Analytical trajectory option (no collisions)
   - Computes entry angle and landing position at hoop plane

6. **Physics Feature Extractor** (`physics_engine/core/feature_extractor.py`)
   - 37 physics-based features
   - Release position, velocity, acceleration
   - Arm geometry (elbow angle, forearm elevation)
   - Finger features (spread, velocity differential)
   - Motion timing and smoothness

### Key Findings

#### Coordinate System
- Data X: Player at ~20, hoop at ~5.25 (negative X = toward hoop)
- Data Y: Lateral (both player and hoop at Y ~ -25)
- Data Z: Vertical (ankle ~0.8, hoop at 10 feet)
- Distance to hoop: ~14-15 feet (free throw line)

#### Velocity Issue
The hand mocap velocity at release is much lower than expected:
- Measured hand velocity: ~3 m/s
- Required ball velocity: ~7-8 m/s

This is because mocap tracks hand position, not ball position. The ball gains additional velocity from arm extension and finger flick, which aren't captured by wrist/fingertip tracking.

#### Feature Correlations with Targets

**target_angle (entry angle):**
- release_ax: -0.683 (strongest)
- release_x: 0.565
- release_dist_to_hoop: 0.562
- upper_arm_elevation: 0.538

**target_depth:**
- release_z: 0.229
- max_vel_frame: -0.203
- Correlations are weak (~0.2)

**target_lr (left/right):**
- Correlations are very weak (<0.12)
- Left/right likely depends on lateral aim, not physics

### CV Results (Physics Features Only)

Using LightGBM with 5-fold GroupKFold by player:
- angle RMSE: 6.17 (worse than baseline of 4.87 std)
- depth RMSE: 8.09 (worse than baseline of 5.40 std)
- left_right RMSE: 4.28 (worse than baseline of 3.80 std)

Physics-only features underperform compared to predicting the mean. This suggests:
1. Physics features need to be combined with original mocap features
2. The entry angle has best signal in physics features
3. Depth and left_right require different approaches

### Files Created

```
physics_engine/
  core/
    data_loader.py          - Data parsing and coordinate handling
    feature_extractor.py    - 37 physics features
    inverse_dynamics.py     - Contact force computation
    release_state.py        - Release parameter extraction
    simulator.py            - Updated with analytical trajectory
    __init__.py             - Updated with new exports
  models/
    hand_ball_contact.xml   - MuJoCo contact model
  output/
    physics_features_train.csv  - Cached features for training
  scripts/
    run_full_pipeline.py    - End-to-end CV and submission
```

### Recommendations for Next Steps

1. **Combine physics features with original mocap features** - The physics features alone aren't predictive enough, but may add value when combined with the full feature set.

2. **Focus on angle prediction** - The physics features have strongest signal for entry angle (r=-0.68 for release_ax).

3. **Try different approaches for depth/left_right** - These targets don't correlate well with physics features. May need lateral aim features or different frame selection.

4. **Tune velocity scaling** - The 2.3x velocity scale factor was arbitrary. Could tune this with Optuna to maximize correlation with targets.

5. **Use physics features as residual model** - Train base model on mocap, then use physics features to predict residuals.
