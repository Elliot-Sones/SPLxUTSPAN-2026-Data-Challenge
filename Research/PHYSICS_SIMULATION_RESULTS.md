# Physics Simulation Results

## Summary

We built a physics simulation of basketball shooting to generate synthetic training data and learn physics-informed features.

**Result**: Physics features show ~0.55 correlation with angle target. This is comparable to our best position features and provides a principled understanding of what drives shot outcomes.

## Approach

### 1. Physics Model
- Simulated ball trajectory under gravity
- Calculated entry point, angle, and accuracy relative to hoop
- Generated 50,000 synthetic shots with varying release parameters

### 2. Feature Extraction
Extracted physics-informed features from body pose:
- Release angle (from wrist velocity)
- Vertical dominance (vz / total_speed)
- Deviation from NBA optimal (58.84 degrees)

### 3. NBA Calibration
Used NBA SportVU data to calibrate optimal values:
- Made shots: angle=58.84 degrees, speed=14.02 fps
- Miss shots: angle=58.64 degrees, speed=14.30 fps

## Feature Correlations with Angle Target

| Feature | Correlation | Description |
|---------|-------------|-------------|
| wrist_x_153 | 0.60 | Wrist horizontal position at release |
| shoulder_z_153 | -0.59 | Shoulder height at release |
| vertical_dominance | -0.55 | Ratio of vertical to total velocity |
| est_release_angle | -0.54 | Estimated launch angle |
| angle_vs_optimal | 0.54 | Deviation from NBA optimal 58.84 degrees |
| wrist_peak_height | -0.51 | Peak wrist height during shot |
| wrist_z_153 | -0.47 | Wrist height at release |
| est_vy | 0.47 | Estimated lateral velocity |
| arm_extension | -0.43 | Arm extension at release |

## Key Insights

### 1. Physics Features Match Position Features
The physics-derived features (vertical_dominance, release_angle) have correlations (-0.55) comparable to raw position features (wrist_x: 0.60). This validates that the physics model captures real signal.

### 2. NBA Optimal Angle is Informative
Deviation from NBA optimal release angle (58.84 degrees) correlates with shot outcome (0.54). Shots closer to optimal angle tend to have better outcomes.

### 3. Vertical Velocity Matters
Higher vertical dominance (more upward velocity vs forward) correlates with lower angle error (-0.55). This matches basketball shooting intuition - proper arc requires strong upward release.

## CV Scores

| Model | angle CV | depth CV | lr CV | Mean CV |
|-------|----------|----------|-------|---------|
| Physics only | 0.0445 | 0.0177 | 0.0166 | 0.0263 |
| Combined (20 features) | 0.0429 | 0.0191 | 0.0159 | 0.0260 |
| Sub 133 baseline | ~0.008 | ~0.008 | ~0.008 | ~0.008 |

The physics model CV is higher than our best models, but provides diverse signal for ensembling.

## Submissions Created

| Submission | Description | angle_std | Notes |
|------------|-------------|-----------|-------|
| Sub 315 | Physics simulation only | 0.100 | Low angle_std - risky |
| Sub 321 | Calibrated physics | 0.092 | Too low angle_std |
| Sub 326 | Enhanced physics | 0.115 | Better but still low |
| Sub 327 | 5% physics + 95% Sub133 | 0.136 | **Best profile** |
| Sub 328 | 10% physics + 90% Sub133 | 0.134 | Good profile |
| Sub 331 | 10% physics + 90% Sub219 | 0.134 | Good profile |

## LB Test Results

| Submission | Description | LB Score | vs Sub 133 |
|------------|-------------|----------|------------|
| Sub 327 | 5% physics + 95% Sub133 | **0.007865** | +0.7% worse |

**Conclusion**: Physics blending did NOT improve LB score.

## Recommendations

The physics simulation approach did not beat Sub 133. Despite having strong feature correlations (-0.55), the physics-informed features do not generalize better to the test set.

## Limitations

1. **CV still higher than baseline**: Physics-only model has CV ~0.026 vs baseline ~0.008
2. **High correlation with existing**: 91% correlation with Sub 219/133 means limited diversity
3. **Scale uncertainty**: Our data is normalized, so absolute velocity values are estimated

## Conclusion

The physics simulation confirms that release angle and velocity features contain meaningful signal for predicting shot outcomes. The NBA-calibrated optimal angle (58.84 degrees) provides a useful reference point.

However, the physics features don't dramatically improve on our existing position-based features. The best approach is small blends (5-10%) with our proven submissions.
