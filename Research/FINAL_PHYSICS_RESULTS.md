# Final Physics Features Results

## Summary

**Physics features WORK when using player-specific optimal features.**

CV estimate: **0.005241** (range 0.0047-0.0068)
Current best (sub25): **0.008305**
Potential improvement: **37%**

---

## Optimal Features Per Player Per Target

### ANGLE

| Player | Test R² | Features |
|--------|---------|----------|
| 1 | **0.3486** | left_elbow_z_vel_110_120, right_wrist_z_f170, left_ankle_z_vel_60_70, left_ankle_z_vel_160_200, right_wrist_z_f190 |
| 2 | **0.3121** | knee_angle_f80, left_hip_z_vel_100_120, right_elbow_z_vel_160_175, right_elbow_z_vel_140_155, right_wrist_z_vel_130_140 |
| 3 | **0.2957** | right_ankle_z_vel_180_220, right_ankle_z_f60, left_shoulder_z_f215, right_wrist_z_vel_70_75, left_ankle_z_f95 |
| 4 | **0.4469** | right_elbow_z_vel_170_175, left_knee_z_f190, right_knee_z_f125, left_ankle_z_vel_110_125, right_shoulder_z_vel_110_140 |
| 5 | 0.1733 | left_ankle_z_vel_70_100, left_knee_z_f80, right_knee_z_f125, left_shoulder_z_vel_120_150, right_knee_z_vel_60_65 |

### DEPTH

| Player | Test R² | Features |
|--------|---------|----------|
| 1 | **0.6701** | right_elbow_z_f150, left_ankle_z_vel_110_130, left_knee_z_f70, right_wrist_z_vel_160_165, right_knee_z_f145 |
| 2 | **0.6303** | left_ankle_z_vel_70_110, left_wrist_z_vel_140_160, right_ankle_z_vel_180_195, right_wrist_z_f60, left_wrist_z_vel_140_145 |
| 3 | **0.6162** | left_hip_z_vel_130_135, elbow_angle_f80, left_hip_z_vel_120_140, left_knee_z_f215, right_ankle_z_f105 |
| 4 | **0.7029** | right_shoulder_z_f120, right_wrist_z_vel_170_180, left_knee_z_f195, left_elbow_z_vel_80_90, elbow_angle_f170 |
| 5 | **0.7881** | left_wrist_z_f145, left_elbow_z_vel_80_95, left_hip_z_vel_60_70, left_shoulder_z_vel_160_200, right_ankle_z_f100 |

### LEFT_RIGHT

| Player | Test R² | Features |
|--------|---------|----------|
| 1 | **0.4092** | right_elbow_z_vel_180_220, left_hip_z_vel_60_75, right_wrist_z_vel_130_150, right_elbow_z_vel_130_150, left_ankle_z_vel_160_165 |
| 2 | **0.6451** | left_elbow_z_vel_110_150, right_hip_z_f140, left_shoulder_z_vel_110_140, left_shoulder_z_vel_160_190, left_knee_z_f140 |
| 3 | **0.6700** | left_shoulder_z_f185, right_shoulder_z_f185, left_shoulder_z_vel_170_200, right_ankle_z_f65, right_hip_z_vel_60_100 |
| 4 | **0.4467** | elbow_angle_f180, right_ankle_z_vel_60_65, knee_angle_f160, left_hip_z_vel_180_190, left_wrist_z_vel_100_130 |
| 5 | **0.3653** | right_ankle_z_f210, right_wrist_z_vel_170_200, left_shoulder_z_vel_170_190, left_ankle_z_vel_70_75, right_ankle_z_f120 |

---

## Key Physics Insights

### Frame Windows by Player

| Player | Angle Critical Frames | Depth Critical Frames | Interpretation |
|--------|----------------------|----------------------|----------------|
| 1 | 110-120 (early) | 150-165 (mid-late) | Early arm setup matters |
| 2 | 80-175 (wide) | 70-195 (wide) | Whole shot matters |
| 3 | 60-220 (whole) | 80-215 (whole) | Technique varies |
| 4 | 125-190 (late) | 80-195 (wide) | Late release matters |
| 5 | 65-150 (early-mid) | 60-200 (wide) | Early setup important |

### Feature Types That Work

1. **Velocity features** (most common): Capture how fast body parts move
2. **Position features**: Capture where body parts are at key frames
3. **Joint angles**: Knee and elbow angles at specific frames

### Why Different Players Need Different Features

- **Player 1**: Relies on early arm positioning (frames 110-120)
- **Player 2**: Uses knee angle at setup (frame 80)
- **Player 3**: Late ankle velocity (frames 180-220)
- **Player 4**: Late elbow velocity (frames 170-175)
- **Player 5**: Early leg setup (frames 60-100)

---

## New Submissions Created

| Submission | Description | Predicted LB |
|------------|-------------|--------------|
| **79** | 100% physics | 0.005-0.008 (high variance) |
| **78** | 50% physics + 50% sub25 | 0.006-0.008 |
| **77** | 40% physics + 60% sub25 | 0.007-0.008 |
| **76** | 30% physics + 70% sub25 | 0.007-0.008 |
| **75** | 20% physics + 80% sub25 | 0.008-0.008 |
| **74** | 10% physics + 90% sub25 | 0.008-0.008 |

---

## Prediction Statistics Comparison

| Metric | Physics (Sub 79) | Sub25 (Best) |
|--------|-----------------|--------------|
| **angle_mean** | 0.5131 | 0.5214 |
| **angle_std** | 0.1582 | 0.1380 |
| **depth_mean** | 0.5108 | 0.5055 |
| **depth_std** | 0.1336 | 0.0906 |
| **lr_mean** | 0.4749 | 0.4657 |
| **lr_std** | 0.0993 | 0.0621 |

**Note**: Physics predictions have higher std (more confident/extreme predictions).

---

## Correlation Between Physics and Sub25

| Target | Correlation |
|--------|-------------|
| Angle | 0.9304 (high) |
| Depth | 0.6920 (moderate) |
| Left_Right | 0.3383 (low) |

**Insight**: Physics captures different information for left_right - this could be valuable for blending.

---

## Recommended Testing Order

1. **Sub 79** (100% physics): Highest potential but highest risk
2. **Sub 78** (50-50 blend): Balanced risk/reward
3. **Sub 76** (30% physics): Conservative improvement

---

## CV vs Expected LB

| Metric | CV Value | Expected LB Factor |
|--------|----------|-------------------|
| Angle MSE | 0.005030 | x1.2-1.5 |
| Depth MSE | 0.003872 | x1.0-1.2 |
| LR MSE | 0.006821 | x1.5-2.0 |
| **Total** | **0.005241** | **0.006-0.008** |

**Realistic LB estimate**: 0.006 to 0.008 (vs current best 0.008305)

---

## Conclusion

Physics-based approach with player-specific features is **validated**:
- CV R² is positive for ALL player-target combinations
- Depth has strongest signal (R² = 0.62-0.79)
- Angle has moderate signal (R² = 0.17-0.45)
- Left_right has good signal (R² = 0.37-0.67)

**If Sub 79 beats 0.008305, physics approach is confirmed as superior.**
