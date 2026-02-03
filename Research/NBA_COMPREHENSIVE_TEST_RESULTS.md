# NBA Data Comprehensive Test Results

## Key Discovery: Launch Angle is the Strongest Signal

The launch angle (arctan(vz/vx) of wrist velocity at release) has **-0.61 correlation** with the angle target. This is the strongest single feature we've found.

## Feature Correlations with Angle Target

| Feature | Correlation | Description |
|---------|-------------|-------------|
| launch_angle | -0.604 | Wrist velocity direction at release |
| la_avg | -0.604 | Multi-frame average launch angle |
| shoulder_z_153 | -0.592 | Shoulder height at release |
| la_elbow | -0.530 | Elbow velocity direction |
| la_follow | -0.526 | Follow-through direction |
| v6_max_height | -0.520 | Maximum wrist height in trajectory |
| vel_vy | +0.467 | Lateral wrist velocity |
| wrist_z_153 | -0.470 | Wrist height at release |
| vel_vz | -0.393 | Vertical wrist velocity |

## Feature Correlations with Depth Target

| Feature | Correlation | Description |
|---------|-------------|-------------|
| vel_vx | -0.374 | Horizontal velocity |
| v2_avg_vx | -0.358 | Average horizontal velocity |
| la_basic | +0.336 | Launch angle (opposite direction!) |

## Critical Finding: Per-Player Normalization Fails

| Metric | Raw | Normalized |
|--------|-----|------------|
| Correlation with angle | -0.604 | -0.020 |

**Conclusion**: The absolute physics of the shot matters, not deviation from player baseline.

## Per-Player Launch Angle Statistics

| Player | Mean Launch Angle | Std |
|--------|-------------------|-----|
| Player 1 | 34.1 degrees | 10.8 |
| Player 2 | -18.9 degrees | 9.3 |
| Player 3 | -78.9 degrees | 14.5 |
| Player 4 | -56.8 degrees | 23.1 |
| Player 5 | 42.0 degrees | 43.7 |
| **Global** | -14.3 degrees | 54.1 |

Players have dramatically different shooting mechanics but absolute launch angle still predicts shot angle error.

## Feature Set Comparison (CV MSE for Angle)

| Feature Set | CV MSE | Notes |
|-------------|--------|-------|
| V7: NBA comparison | 0.028 | Best single set |
| V1: Basic velocity | 0.033 | Simple, effective |
| V2: Multi-window velocity | 0.038 | More stable |
| V4: Multi-joint velocity | 0.041 | Kinetic chain |
| V3: Acceleration-based | 0.049 | Harder to extract |
| V5: Position features | 0.091 | Position alone weak |
| V6: Trajectory features | 0.102 | Path features weak |

## Model Configurations Tested

| Configuration | Overall CV MSE |
|---------------|----------------|
| Ridge alpha=100 | 0.026 |
| Ridge alpha=10 | 0.043 |
| Ridge alpha=1 | 0.049 |
| Huber | 0.049 |

Higher regularization helps - more features = more regularization needed.

## Submissions Created

### Base Models (NBA-derived)

| Sub | Model | CV MSE | angle_std | Corr w/ Sub219 |
|-----|-------|--------|-----------|----------------|
| 270 | Comprehensive NBA features | 0.029 | 0.128 | 0.854 |
| 277 | Launch angle focused | 0.043 | 0.118 | 0.826 |
| 286 | Per-player launch angle | 0.038 | 0.107 | 0.805 |

### Blends with Sub 219

| Sub | Blend | angle_std |
|-----|-------|-----------|
| 271 | 5% NBA + 95% Sub219 | 0.136 |
| 272 | 10% NBA + 90% Sub219 | 0.135 |
| 273 | 15% NBA + 85% Sub219 | 0.133 |
| 274 | 20% NBA + 80% Sub219 | 0.132 |
| 275 | 25% NBA + 75% Sub219 | 0.131 |
| 276 | 30% NBA + 70% Sub219 | 0.130 |
| 278 | 10% launch + 90% Sub219 | 0.133 |
| 279 | 15% launch + 85% Sub219 | 0.132 |
| 280 | 20% launch + 80% Sub219 | 0.130 |
| 281 | 25% launch + 75% Sub219 | 0.129 |
| 287 | 10% per-player + 90% Sub219 | 0.132 |
| 288 | 15% per-player + 85% Sub219 | 0.130 |
| 289 | 20% per-player + 80% Sub219 | 0.128 |

### Multi-Model Blends

| Sub | Blend | angle_std |
|-----|-------|-----------|
| 282 | 10% launch + 10% NBA + 80% Sub219 | 0.131 |
| 283 | 15% launch + 10% NBA + 75% Sub219 | 0.129 |
| 284 | 10% launch + 15% NBA + 75% Sub219 | 0.130 |
| 285 | 15% launch + 15% NBA + 70% Sub219 | 0.128 |
| 290 | 10% each of 3 NBA + 70% Sub219 | 0.126 |

## Recommended Submissions to Test

Based on diversity from Sub 219 and good profiles:

1. **Sub 290** (four-way blend): angle_std=0.126, combines all NBA signals
2. **Sub 274** (20% comprehensive + 80% Sub219): angle_std=0.132, good balance
3. **Sub 280** (20% launch + 80% Sub219): angle_std=0.130, launch angle focused
4. **Sub 289** (20% per-player + 80% Sub219): angle_std=0.128, most diverse

## Why Launch Angle Works

The launch angle = arctan(vz/vx) captures:
1. **Ball trajectory arc**: Higher angle = higher arc shot
2. **Shot distance proxy**: Different angles for different distances
3. **Release timing**: Angle changes through the shooting motion

NBA data showed that optimal launch angles exist (~59 degrees for NBA), but our data uses normalized coordinates so absolute values differ. The key insight is that the RATIO of vertical to horizontal velocity at release strongly predicts shot outcome.

## Conclusion

The NBA data helped us discover that **launch angle (wrist velocity direction at release)** is a powerful predictor (-0.61 correlation). This physical insight - that the angle at which the ball leaves the hand matters - is valuable even though we cannot directly use NBA velocity values.

The best submissions combine:
- Launch angle features (capture release mechanics)
- Standard position features (capture body pose)
- Sub 219 as anchor (proven LB performer)
