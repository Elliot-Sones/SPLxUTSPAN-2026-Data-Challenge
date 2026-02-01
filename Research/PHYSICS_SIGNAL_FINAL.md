# Final Physics Signal Summary

## Overview

After comprehensive investigation using all 207 keypoints including detailed finger tracking, here are the physics features that explain target variance.

---

## Results Summary

| Target | Avg Max R-squared | Range | Signal Strength |
|--------|------------------|-------|-----------------|
| **DEPTH** | **0.590** | 0.508 - 0.703 | STRONG |
| **LEFT_RIGHT** | **0.517** | 0.255 - 0.702 | STRONG |
| **ANGLE** | **0.202** | 0.004 - 0.378 | MODERATE |

---

## DEPTH (59% variance explained on average)

### Best Features Per Player

| Player | R-squared | Feature Group | Frame | Model |
|--------|----------|---------------|-------|-------|
| 1 | 0.579 | arm_chain_full (pos+vel+acc) | 150 | RF |
| 2 | 0.569 | leg_drive_vel | 90 | Ridge |
| 3 | 0.508 | leg_drive_full (pos+vel+acc) | 120 | RF |
| 4 | 0.589 | arm_chain_full | 135 | RF |
| 5 | 0.703 | arm_chain_pos | 145 | Ridge |

### Key Physics

**Leg Drive**: Knee and ankle Z-velocity during push phase (frames 80-140)
- `right_knee_z` velocity
- `right_ankle_z` velocity
- `right_hip_z` velocity

**Body Extension**: Full kinetic chain at release (frames 130-170)
- `right_shoulder`, `right_elbow`, `right_wrist` positions
- Velocity and acceleration of arm joints
- `wrist_z - hip_z` difference

### Feature Definition

```python
# Leg drive velocity
leg_drive_vel = [
    (right_knee_z[frame+5] - right_knee_z[frame]) / 5,
    (right_ankle_z[frame+5] - right_ankle_z[frame]) / 5,
    (right_hip_z[frame+5] - right_hip_z[frame]) / 5,
]

# Arm chain full (position + velocity + acceleration)
for joint in [right_shoulder, right_elbow, right_wrist]:
    for axis in [x, y, z]:
        position = joint_axis[frame]
        velocity = (joint_axis[frame+5] - joint_axis[frame]) / 5
        acceleration = (velocity_t2 - velocity_t1) / 5
```

---

## LEFT_RIGHT (52% variance explained on average)

### Best Features Per Player

| Player | R-squared | Feature Group | Frame | Model |
|--------|----------|---------------|-------|-------|
| 1 | 0.544 | guide_hand | 175 | RF |
| 2 | 0.702 | arm_chain_pos | 155 | Ridge |
| 3 | 0.672 | body_alignment | 175 | Ridge |
| 4 | 0.414 | guide_hand | 170 | Ridge |
| 5 | 0.255 | body_alignment | 175 | RF |

### Key Physics

**Body Alignment**: Left-right symmetry at release
- `right_shoulder_z - left_shoulder_z`
- `right_hip_z - left_hip_z`
- `right_elbow_z - left_elbow_z`

**Guide Hand**: Left hand position (stabilizes the ball)
- `left_wrist_x`, `left_wrist_y`, `left_wrist_z`
- `left_second_finger_distal` positions

**Arm Chain**: Shooting arm position
- All XYZ coordinates of shoulder, elbow, wrist

### Feature Definition

```python
# Body alignment (symmetry)
body_alignment = [
    right_shoulder_x - left_shoulder_x,
    right_shoulder_y - left_shoulder_y,
    right_shoulder_z - left_shoulder_z,
    right_hip_x - left_hip_x,
    right_hip_y - left_hip_y,
    right_hip_z - left_hip_z,
    right_elbow_x - left_elbow_x,
    right_elbow_y - left_elbow_y,
    right_elbow_z - left_elbow_z,
]

# Guide hand
guide_hand = [
    left_wrist_x, left_wrist_y, left_wrist_z,
    left_second_finger_distal_x, left_second_finger_distal_y, left_second_finger_distal_z,
]
```

---

## ANGLE (20% variance explained on average)

### Best Features Per Player

| Player | R-squared | Feature Group | Frame | Window | Model |
|--------|----------|---------------|-------|--------|-------|
| 1 | 0.284 | combined | 165 | 3 | Ridge |
| 2 | 0.051 | wrist_snap | 162 | 3 | Ridge |
| 3 | 0.004 | finger_curl | - | - | - |
| 4 | 0.292 | combined | 145 | 5 | RF |
| 5 | 0.378 | combined | 155 | 3 | Ridge |

### Key Physics

**Combined Features** (best for Players 1, 4, 5):
- Fingertip positions (index, middle)
- Fingertip velocities
- Wrist snap (wrist velocity relative to elbow)
- Guide hand position

**Wrist Snap** (best for Player 2):
- `right_wrist` velocity minus `right_elbow` velocity
- Captures the final flick motion

### Feature Definition

```python
# Combined angle features
window = 3  # or 5 for some players

# Fingertip positions
fingertip_pos = [
    right_second_finger_distal_x, right_second_finger_distal_y, right_second_finger_distal_z,
    right_third_finger_distal_x, right_third_finger_distal_y, right_third_finger_distal_z,
]

# Fingertip velocities
fingertip_vel = [
    (right_second_finger_distal_x[frame+window] - right_second_finger_distal_x[frame]) / window,
    # ... for all axes and fingers
]

# Wrist snap (relative velocity)
wrist_snap = [
    (right_wrist_x[frame+window] - right_wrist_x[frame]) / window -
    (right_elbow_x[frame+window] - right_elbow_x[frame]) / window,
    # ... for y and z
]

# Guide hand
guide_hand = [left_wrist_x, left_wrist_y, left_wrist_z]
```

### Why ANGLE is Harder

1. **72% of variance is between-player** (each player has characteristic angle ~41-52 degrees)
2. Only 28% is within-player shot-to-shot variation
3. We explain ~20% of this 28% = ~5.6% of total angle variance
4. Players 2 and 3 have very weak physics signal (different mechanics?)

---

## Player-Specific Notes

### Player 3 (Hardest to Predict)

- DEPTH: R-squared=0.508 (ok)
- LEFT_RIGHT: R-squared=0.672 (good)
- ANGLE: R-squared=0.004 (almost none)

Player 3 may have:
- More variable shooting form
- Different biomechanics
- The physics relationship differs from other players

### Player 5 (Highest Variance, Highest Signal)

- DEPTH: R-squared=0.703 (best)
- ANGLE: R-squared=0.378 (best)
- Has highest target variance but also most predictable by physics

---

## Recommended Features for Model

### High Priority (Add to Baseline)

**For DEPTH** (all players):
```python
# Frame 90-150 depending on player
leg_drive_features = [
    right_knee_z_velocity,
    right_ankle_z_velocity,
    right_hip_z_velocity,
    right_hip_z_position,
]

# Frame 130-170
arm_extension_features = [
    right_wrist_z - right_hip_z,  # body extension
    right_wrist_xyz_position,
    right_wrist_xyz_velocity,
    right_elbow_xyz_velocity,
]
```

**For LEFT_RIGHT** (all players):
```python
# Frame 155-175
body_alignment_features = [
    right_shoulder_z - left_shoulder_z,
    right_hip_z - left_hip_z,
    left_wrist_xyz,  # guide hand
]
```

### Medium Priority

**For ANGLE** (Players 1, 4, 5 - skip 2, 3):
```python
# Frame 145-165, window=3-5
angle_features = [
    right_second_finger_distal_xyz,
    right_third_finger_distal_xyz,
    fingertip_velocity_xyz,
    wrist_snap_xyz,  # wrist_vel - elbow_vel
    left_wrist_xyz,  # guide hand
]
```

---

## Validation Method

All results validated using:
- **Nested 5-fold cross-validation** (prevents data leakage)
- **Per-player models** (each player has unique biomechanics)
- **Ridge regression** and **Random Forest** tested for each config
- Frame selection done on inner CV loop only

---

## Files

- `scripts/comprehensive_physics_groups.py` - Main comprehensive test
- `scripts/angle_finger_deep_dive.py` - Detailed angle investigation
- `scripts/keypoint_exploration.py` - Available keypoints analysis
- `output/comprehensive_physics_groups.csv` - Full results
- `output/angle_finger_deep_dive.csv` - Angle detailed results
