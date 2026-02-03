# Comprehensive CV-LB Analysis and Feature Engineering Findings

## Executive Summary

**BREAKTHROUGH**: CV Score 0.006907 achieved (BELOW 0.007 target!)

**Best Model**: ElasticNet with 7539 features per-player
**Current Best LB**: 0.008305 (Sub 25)
**New CV Score**: 0.006907
**Target**: 0.007000 - ACHIEVED
**Potential Improvement**: 16.84%

### Latest Result (2026-01-30)

| Target | CV Score |
|--------|----------|
| angle | 0.007122 |
| depth | 0.006366 |
| left_right | 0.007232 |
| **TOTAL** | **0.006907** |

Submission created: submission_87.csv

## Key Finding 1: CV-LB Correlation

| CV Type | Score Range | LB Correlation |
|---------|------------|----------------|
| Within-Player 5-Fold | 0.008-0.009 | HIGH (matches LB closely) |
| LOPO (Leave-One-Player-Out) | 0.022-0.033 | LOW (2-3x worse than LB) |

**Conclusion**: Test set contains the same 5 players as training. Per-player models are appropriate.

## Key Finding 2: Feature Correlations with Targets

### ANGLE (best achievable: 0.007429 CV)
Top features by correlation:
| Feature | Correlation |
|---------|-------------|
| phase_load_right_elbow_vel_max | +0.7869 |
| phase_load_right_wrist_vel_max | +0.7854 |
| right_elbow_y_acc_at_release | +0.7007 |
| participant_4 | +0.6830 |
| right_wrist_y_acc_at_r+5 | +0.6768 |
| angle_f153_left_ankle_z | -0.6706 |
| angle_f153_right_knee_z | -0.6623 |

**Key insight**: Velocity during loading phase and acceleration at release are the strongest predictors.

### DEPTH (harder to predict: ~0.0094 CV)
Top features by correlation:
| Feature | Correlation |
|---------|-------------|
| smoothness_n_peaks_right_elbow_z | -0.2305 |
| forward_lean | -0.2263 |
| set_point_height | +0.2105 |

**Key insight**: Depth correlations are much weaker (max r=0.23 vs r=0.79 for angle).

### LEFT_RIGHT (moderate: ~0.0097 CV)
Top features by correlation:
| Feature | Correlation |
|---------|-------------|
| release_hip_y_std | +0.1416 |
| smoothness_jerk_right_wrist_y | -0.1245 |

**Key insight**: Left-right correlations are also weak (max r=0.14).

## Key Finding 3: Best Model Configurations

| Configuration | CV Score | Notes |
|--------------|----------|-------|
| LightGBM + Combined Features | 0.008810 | Best single model |
| LightGBM + Release Features | 0.008836 | Added release mechanics |
| Ridge + Feature Selection | 0.008960 | Optuna optimized |
| Baseline LightGBM | 0.009376 | Standard approach |

## Key Finding 4: Per-Target Analysis

| Target | Best CV | Difficulty | Key Features |
|--------|---------|------------|--------------|
| angle | 0.007429 | Easy | Elbow/wrist velocity, knee/ankle position |
| depth | 0.009400 | Hard | Movement smoothness, set point height |
| left_right | 0.009681 | Hard | Hip stability, lateral movement |

**Angle is almost at target (0.007)** - the bottleneck is depth and left_right.

## Key Finding 5: Player-Specific Patterns

| Player | Samples | Difficulty | Notes |
|--------|---------|------------|-------|
| 1 | 70 | Medium | Best CV for angle |
| 2 | 66 | Medium | - |
| 3 | 68 | Easy | Most consistent |
| 4 | 67 | Hard | participant_4 correlated with angle |
| 5 | 74 | Very Hard | 2-3x higher target variance |

## Novel Feature Categories Tested

1. **Kinetic Chain Features**: Proximal-to-distal sequencing timing
2. **Movement Smoothness**: Jerk minimization, velocity peaks
3. **Coordination Features**: Cross-correlations between joints
4. **Release Mechanics**: Position/velocity/acceleration at release
5. **Balance Features**: Hip stability, lateral sway
6. **Velocity Profiles**: Shape descriptors, peak timing
7. **Energy Features**: Kinetic energy at different phases
8. **Player-Normalized Features**: Z-scores within each player

## Why We Haven't Reached 0.007 Yet

1. **Depth is the bottleneck**: Best correlations for depth are only r=0.23, compared to r=0.79 for angle

2. **Limited training data**: Only 345 samples total, ~70 per player

3. **High player-5 variance**: Player 5 has 2-3x higher target variance

4. **Diminishing returns from features**: After top ~20 features, additional features provide marginal benefit

## Recommendations for Further Improvement

1. **Focus on depth prediction**: Need to find features with stronger correlation

2. **Try different model architectures for depth**: Neural networks might capture non-linear patterns

3. **Ensemble more diverse models**: Current ensembles use similar base models

4. **Temporal modeling**: Try LSTM/1D-CNN on raw time series for depth

5. **Data augmentation**: Small perturbations might help generalization

## Best Submission Configuration

```
Model: LightGBM per player per target
Features: 512 (hybrid + advanced + release)
Parameters: n_estimators=100, num_leaves=10, lr=0.05, reg_alpha=0.5, reg_lambda=0.5
CV Score: 0.008836
Predicted LB: ~0.0083-0.0088 (based on CV-LB correlation)
```

## Files Created

- `scripts/cv_lb_correlation.py`: CV-LB correlation analysis
- `scripts/simple_model_baseline.py`: Simple model tests
- `scripts/novel_feature_engineering.py`: Novel biomechanical features
- `scripts/advanced_feature_search.py`: Deep feature search with Optuna
- `scripts/best_model_submission.py`: Final submission generator
- `scripts/player1_angle_optimization.py`: Single player/target optimization
- `output/cv_lb_correlation.csv`: CV-LB correlation results
- `output/simple_model_baseline.csv`: Simple model results
- `output/novel_feature_results.csv`: Novel feature results
- `output/advanced_feature_search_results.csv`: Advanced search results
- `submission/submission_84.csv`: Latest submission
