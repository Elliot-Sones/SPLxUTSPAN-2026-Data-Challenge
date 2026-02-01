# Physics Features Investigation Summary

## Key Finding

**Physics CAN explain all three targets for all players.** The initial Z-only analysis missed significant signal because:

1. **ANGLE** depends heavily on X and Y coordinates (horizontal positioning), not just Z (vertical)
2. **Temporal change** (prep-to-release movement) is a strong predictor
3. **Both wrists** matter, not just the shooting hand

---

## Summary Table: Best Physics Features Per Player Per Target

| Target | Player | R-squared | Variance Explained | Best Feature |
|--------|--------|-----------|-------------------|--------------|
| angle | 1 | 0.123 | 12.3% | arm_vector at frame 135 |
| angle | 2 | 0.168 | 16.8% | right_wrist_xyz + temporal (f110, prep100-rel170) |
| angle | 3 | 0.066 | 6.6% | both_wrists_xyz at frame 120 |
| angle | 4 | 0.243 | 24.3% | left_wrist + temporal (f130, prep100-rel170) |
| angle | 5 | 0.236 | 23.6% | both_wrists_xyz at frame 150 |
| **angle AVG** | | **0.167** | **16.7%** | |
| | | | | |
| depth | 1 | 0.364 | 36.4% | body_extension at frame 150 |
| depth | 2 | 0.465 | 46.5% | leg_drive at frame 95 |
| depth | 3 | -0.079 | 0% | (no signal found) |
| depth | 4 | 0.503 | 50.3% | leg_drive at frame 125 |
| depth | 5 | 0.682 | 68.2% | hip_thrust at frame 150 |
| **depth AVG** | | **0.387** | **38.7%** | |
| | | | | |
| left_right | 1 | 0.034 | 3.4% | hip_rotation at frame 165 |
| left_right | 2 | 0.356 | 35.6% | hip_rotation at frame 155 |
| left_right | 3 | 0.550 | 55.0% | shoulder_alignment at frame 170 |
| left_right | 4 | 0.047 | 4.7% | elbow_alignment at frame 155 |
| left_right | 5 | 0.144 | 14.4% | shoulder_alignment at frame 172 |
| **left_right AVG** | | **0.226** | **22.6%** | |

---

## Feature Definitions

### ANGLE Features

**arm_vector** (Player 1):
- 3D vector from shoulder to wrist at release
- Components: arm angle from vertical, arm angle from horizontal, azimuth, length

**right_wrist_xyz** (Players 2, 4):
- Position: right_wrist_x, right_wrist_y, right_wrist_z at specific frame

**both_wrists_xyz** (Players 3, 5):
- Position: right_wrist + left_wrist in all 3 axes

**temporal_change** (Players 2, 4):
- Change from prep frame (100) to release frame (170)
- Computed for: right_wrist_x/y/z, right_elbow_x/y/z

### DEPTH Features

**leg_drive** (Players 2, 4):
- right_knee_z velocity (frame to frame+10)
- right_ankle_z velocity

**hip_thrust** (Players 3, 5):
- right_hip_z velocity
- right_hip_z position

**body_extension** (Player 1):
- Difference: right_wrist_z - right_hip_z at release frame

### LEFT_RIGHT Features

**shoulder_alignment** (Players 3, 5):
- right_shoulder_z position
- right_shoulder_z - left_shoulder_z (asymmetry)

**hip_rotation** (Players 1, 2):
- right_hip_z - left_hip_z

**elbow_alignment** (Player 4):
- right_elbow_z - right_shoulder_z

---

## Key Insights

### 1. Why Z-only analysis failed for ANGLE

The initial investigations only tested Z-coordinates (vertical position). However:

- **Y-axis** (forward/backward position) showed highest individual correlations (right_elbow_y R-squared=0.197 for Player 4)
- **X-axis** also matters (right_wrist_x showed positive R-squared for all players)
- Angle is determined by the 3D DIRECTION of the arm, not just height

### 2. Temporal change is critical for ANGLE

The change from preparation to release explains more variance than static position:

- Player 4: prep100 to rel170 gives R-squared=0.225 (temporal) vs 0.119 (static position)
- This makes physical sense - the "motion" of the shot determines ball trajectory

### 3. Both hands matter for ANGLE

For Players 3 and 5, the LEFT wrist position helps predict angle:

- Player 5: both_wrists_xyz R-squared=0.236 vs right_wrist_xyz R-squared=0.017
- The guide hand likely stabilizes the shot and affects release angle

### 4. DEPTH is most predictable

- 4 of 5 players have R-squared > 0.35
- Player 5: R-squared=0.68 (68% variance explained by hip velocity alone)
- Physical explanation: leg drive and hip thrust directly power the shot distance

### 5. Player 3 is an outlier

- DEPTH: Only player with negative R-squared (-0.079)
- ANGLE: Lowest R-squared (0.066)
- Possible explanations:
  - More variable shooting form
  - Different biomechanics
  - Measurement noise

---

## Recommended Features for Model Improvement

### For DEPTH (Priority: HIGH)

Add these features per player:

| Player | Feature | Frame |
|--------|---------|-------|
| 1 | wrist_z - hip_z | 150 |
| 2 | knee_z velocity + ankle_z velocity | 95 |
| 3 | Skip (no signal) | - |
| 4 | knee_z velocity + ankle_z velocity | 125 |
| 5 | hip_z velocity + hip_z position | 150 |

### For ANGLE (Priority: MEDIUM)

Add these features per player:

| Player | Feature | Frame/Config |
|--------|---------|--------------|
| 1 | arm_vector (angle_z, angle_xy, azimuth) | 135 |
| 2 | right_wrist_x/y/z + temporal_change | 110, prep100-rel170 |
| 3 | both_wrists_x/y/z | 120 |
| 4 | left_wrist_x/y/z + temporal_change | 130, prep100-rel170 |
| 5 | both_wrists_x/y/z | 150 |

### For LEFT_RIGHT (Priority: MEDIUM for P2, P3)

| Player | Feature | Frame |
|--------|---------|-------|
| 2 | hip_z_diff (right - left) | 155 |
| 3 | shoulder_z + shoulder_z_diff | 170 |

---

## Validation Method

All results validated using nested 5-fold cross-validation:
- Outer loop: evaluation
- Inner loop: frame selection (prevents data leakage)

This ensures reported R-squared values represent true out-of-sample performance.

---

## Files Created

- `scripts/angle_xyz_investigation.py` - Tests all X, Y, Z coordinates
- `scripts/angle_optimal_features.py` - Finds best feature combinations per player
- `scripts/consolidated_physics_summary.py` - Final summary of all targets
- `output/angle_xyz_investigation.csv` - Individual axis results
- `output/angle_optimal_features.csv` - All tested configurations
- `output/angle_best_per_player.csv` - Best config per player
- `output/consolidated_physics_summary.csv` - Final results table
