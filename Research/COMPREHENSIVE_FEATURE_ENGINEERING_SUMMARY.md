# Comprehensive Feature Engineering Summary

## Goal
Find a strategy 100% confident will score < 0.007 on LB (10.4% improvement from current best 0.007809).

## What Was Tested

### 1. Feature Engineering Approaches (scripts/comprehensive_feature_test.py)
| Approach | CV MSE | angle_std | depth_mean | Profile Dist |
|----------|--------|-----------|------------|--------------|
| V1: Standard frame-based | 0.0084 | 0.1392 | 0.5158 | 0.0118 |
| V2: Player-normalized | 0.0093 | 0.1434 | 0.5174 | 0.0175 |
| V3: Release-aligned | 0.0113 | 0.1392 | 0.5164 | 0.0124 |
| V4: Velocity/Acceleration | 0.0088 | 0.1392 | 0.5160 | 0.0121 |
| V5: Relative features only | 0.0086 | 0.1398 | 0.5157 | 0.0123 |
| V6: Combined (pos+vel+rel) | 0.0078 | 0.1405 | 0.5166 | 0.0139 |
| V7: Minimal stable | 0.0100 | 0.1380 | 0.5160 | 0.0109 |

**Finding**: Best CV (V6) still has profile mismatch. All approaches have depth_mean ~0.516 vs target 0.5055.

### 2. Finger/Hand Features (scripts/finger_features.py)
- 291 features focused on hand/finger positions at release
- **Best angle CV**: 0.0064 (best seen!)
- But profile mismatch: angle_std=0.1652 vs target 0.1377

### 3. Calibration Approaches (scripts/distribution_calibration.py)
| Approach | Profile Dist | Corr with Sub 133 |
|----------|--------------|-------------------|
| Base (uncalibrated) | 0.1376 | 0.3001 |
| Shift calibrated | 0.0001 | 0.5037 |
| Quantile calibrated | 0.0044 | 0.8331 |

**Finding**: Calibration can match profile perfectly but doesn't improve underlying predictions.

### 4. Teacher-Student Learning (scripts/teacher_student.py)
- Train model to mimic Sub 133's predictions
- Per-target optimal weights found
- Result: 99.83% correlation with Sub 133

### 5. Uncertainty Analysis (scripts/uncertainty_analysis.py)
- Sample 17 (Player 1) is massive outlier in feature space
- Score 414 vs threshold 5.77
- Our predictions for this sample are completely wrong

### 6. Outlier-Robust Model (scripts/outlier_robust_model.py)
- Huber regression failed (profile completely off)
- Outlier-weighted blend: profile_dist=0.0044, corr=0.9940

## Key Findings

### 1. Profile Matching is Necessary
- Sub 133 profile: angle_std=0.1377, depth_mean=0.5055
- All successful submissions must match this profile
- Training data mean depth=0.5157, but Sub 133 predicts 0.5055 (lower)

### 2. CV Does Not Predict LB
- Within-player CV correlates better than LOPO
- But even within-player CV doesn't guarantee LB improvement
- Example: ElasticNet CV 0.007 -> LB 0.067 (10x overfit)

### 3. All Good Submissions Converge to Sub 133
- Submissions with good profiles are 99%+ correlated with Sub 133
- When predictions differ significantly, profiles become bad
- This limits potential improvement

### 4. Per-Player Correlation Varies Wildly
- Player 1: 16% angle correlation with Sub 133
- Player 2: -5% angle correlation (negative!)
- Player 3: 82% correlation
- This suggests different features work for different players

## Submissions Created (145-160)

| Sub | Description | Profile Dist | Corr w/133 |
|-----|-------------|--------------|------------|
| 151 | Optimized blend (45/45/10/0) | 0.0003 | 0.9911 |
| 152 | Per-target optimal | 0.0006 | 0.9983 |
| 153 | Uniform 0.15 blend | 0.0006 | 0.9968 |
| 155 | Finger blend | 0.0007 | 0.9997 |
| 149 | Strategic blend | 0.0012 | 0.9788 |
| 146 | Quantile calibrated | 0.0038 | 0.8331 |
| 160 | Outlier-weighted | 0.0044 | 0.9940 |

## Recommendations for Testing

### High Confidence Submissions (most likely to match Sub 133)
1. **Sub 151** - Optimized blend with different weights
2. **Sub 152** - Per-target optimal weights
3. **Sub 153** - Uniform blend with student model

### Experimental Submissions (might be better or worse)
1. **Sub 146** - Quantile calibrated (0.83 corr, different approach)
2. **Sub 154** - Finger features only (0.89 corr, best angle CV)

## Honest Assessment

### Why 10% Improvement is Hard
1. **Limited signal in data**: Motion capture features have finite predictive power
2. **Sub 133 already captures most signal**: Blend of 4 models, optimized on LB
3. **Profile constraint**: Must match Sub 133's distribution to perform well
4. **No cross-validation of LB**: Can't validate improvements offline

### What Would Be Needed
To be **100% confident** in beating 0.007 LB:
1. A new data source with additional information
2. Discovery of a systematic error in Sub 133's predictions
3. A modeling breakthrough that extracts more signal from existing data

### Current Confidence Level
- Sub 133 (0.007809) is a strong baseline
- New submissions are variations, not fundamental improvements
- Without LB testing, cannot be 100% confident any will beat 0.007

## Next Steps
1. Submit top candidates (151, 152, 153) to get LB feedback
2. If feedback shows improvement, iterate on that direction
3. If no improvement, may need to accept 0.0078 as near-optimal
