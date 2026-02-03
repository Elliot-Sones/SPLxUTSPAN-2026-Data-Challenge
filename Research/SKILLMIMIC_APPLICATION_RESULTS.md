# SkillMimic Basketball Data Analysis and Application Results

## Overview

This document summarizes the analysis of SkillMimic basketball shooting data and its application to the SPL prediction task.

## SkillMimic Data Structure

### Files Analyzed
- shot_style1.pt: 104 frames x 337 features
- shot_style2.pt: 97 frames x 337 features
- shot_style3.pt: 100 frames x 337 features

### Feature Composition (337 total)
| Feature Range | Content | Description |
|--------------|---------|-------------|
| 0-2 | root_pos | Pelvis position (heading-relative) |
| 3-5 | root_rot | Pelvis rotation (exponential map) |
| 6-161 | dof_pos | 52 joints x 3 DOF = 156 joint positions |
| 162-317 | dof_vel | 52 joints x 3 DOF = 156 joint velocities |
| 318-320 | ball_pos | Ball position (X, Y, Z) |
| 321-324 | ball_rot | Ball rotation (quaternion) |
| 325-327 | ball_vel | Ball velocity (X, Y, Z) |
| 328-335 | key_body | Key body positions (8 features) |
| 336 | contact | Ball contact flag (1=in hand, 0=released) |

## Key Shooting Patterns Extracted

### Release Timing
All shot styles show consistent release timing:
- shot_style1: Release at frame 59/104 (56.7% of motion)
- shot_style2: Release at frame 51/97 (52.6% of motion)
- shot_style3: Release at frame 62/100 (62.0% of motion)

**For SPL (240 frames)**: Release occurs approximately at frames 125-155 (52-65% of motion)

### Release Biomechanics
| Metric | shot_style1 | shot_style2 | shot_style3 | Mean |
|--------|-------------|-------------|-------------|------|
| Release angle (vertical) | 69.9 deg | 69.7 deg | 72.1 deg | 70.6 deg |
| Elbow angle at release | 26.3 deg | 15.7 deg | 69.1 deg | 37.1 deg |
| Ball speed at release | 2.734 | 2.736 | 2.578 | 2.683 |
| Peak wrist velocity | 2.317 | 2.276 | 2.213 | 2.269 |

### Key Insights
1. **Consistent release angle**: ~70 degrees vertical across all styles
2. **Variable elbow angle**: 15-69 degrees - indicates different shooting styles
3. **Peak velocity timing**: Coincides with release frame
4. **Kinetic chain**: Hip -> shoulder -> elbow -> wrist sequential activation

## SPL Data Comparison

| Aspect | SPL Dataset | SkillMimic |
|--------|-------------|------------|
| Frames | 240 | ~100 |
| Keypoints | 69 | 52 joints |
| Features per frame | 207 | 337 |
| Velocities | Must compute | Included |
| Ball state | Not included | Included |
| Coordinate system | World | Heading-relative |

## Feature Engineering Results

### SkillMimic-Inspired Features Extracted (44 features)
Categories:
1. **Position-based**: wrist height, elbow angle, shoulder angle, arm extension
2. **Velocity-based**: wrist velocity magnitude, vertical/forward components
3. **Temporal**: peak velocity frame ratio, max height frame ratio
4. **Kinetic chain**: timing delays, coordination scores

### Feature Correlations with SPL Targets

**For angle prediction (strongest correlations)**:
| Feature | |r| |
|---------|-----|
| wrist_vel_y_ratio_v145 | 0.6415 |
| wrist_asymmetry_x_f125 | 0.6064 |
| wrist_asymmetry_x_f140 | 0.6023 |
| elbow_angle_f140 | 0.5272 |
| arm_extension_f140 | 0.5013 |

**For depth prediction**:
| Feature | |r| |
|---------|-----|
| wrist_vel_z_ratio_v155 | 0.3762 |
| wrist_above_nose_f140 | 0.1584 |
| wrist_vel_y_ratio_v135 | 0.1559 |

**For left_right prediction**:
| Feature | |r| |
|---------|-----|
| wrist_rel_shoulder_f140 | 0.2490 |
| arm_vertical_angle_f155 | 0.2013 |
| wrist_rel_shoulder_f125 | 0.1935 |

### LOPO CV Results (Leave-One-Player-Out)

| Configuration | angle MSE | depth MSE | left_right MSE |
|--------------|-----------|-----------|----------------|
| Raw features | 40.11 | 40.66 | 28.57 |
| With player means | 31.04 | 46.09 | 26.28 |

**Key Finding**: Despite high within-player correlations (|r| up to 0.64), features do NOT generalize across players in LOPO CV.

## Conclusions

### What Works
1. **SkillMimic provides valuable biomechanical insights**:
   - Release timing (52-65% of motion)
   - Kinetic chain patterns
   - Key body positions at release

2. **High correlation features identified**:
   - Wrist velocity components
   - Arm asymmetry
   - Elbow angle at release frames

### What Doesn't Work
1. **Cross-player generalization**: Features that correlate strongly within-player do not predict well for unseen players
2. **Raw biomechanical features**: Without player-specific calibration, these features lead to overfitting

### Recommendations for Future Work
1. **Player-relative features**: Instead of absolute values, use features relative to each player's own baseline
2. **Temporal pattern matching**: Use SkillMimic patterns as templates for phase detection
3. **Transfer learning**: Pre-train on SkillMimic velocity patterns, fine-tune on SPL

## Files Created
- scripts/skillmimic_analysis.py - Data structure analysis
- scripts/skillmimic_features.py - Feature extraction
- scripts/skillmimic_combined_model.py - Enhanced model with normalized features
- output/skillmimic_patterns.json - Extracted shooting patterns
- output/skillmimic_temporal.json - Temporal features
- output/skillmimic_features.csv - Extracted features for all SPL shots
- output/skillmimic_enhanced_features.csv - Normalized features
- output/skillmimic_enhanced_results.csv - CV results
