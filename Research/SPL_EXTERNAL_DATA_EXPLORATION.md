# SPL External Data Exploration Results

**Date**: 2026-02-09
**Agent**: external-data-agent

## Summary

Explored the SPL Open Data basketball free throw dataset (125 trials, 1 participant) for potential transfer to our competition task. The SPL data has compatible targets but limited keypoints and a different recording setup. After extensive experimentation, SPL external data does NOT provide meaningful improvement over our existing pipeline.

## SPL Dataset Details

- **Source**: external_data/SPL-Open-Data/basketball/freethrow/data/P0001/
- **Trials**: 125 free throw shots from 1 participant (88 made, 37 missed)
- **Format**: 240 frames at 30fps (8 seconds per shot), JSON per trial
- **Keypoints**: 27 body keypoints (R/L: eye, ear, shoulder, elbow, wrist, hip, knee, ankle, 1st/5th finger, 1st/5th toe, heel)
- **Ball tracking**: 3D ball positions (x, y, z) during shot, often NaN outside capture volume
- **Targets**: entry_angle (degrees), landing_x (inches, lateral), landing_y (inches, depth)

## Compatibility Analysis

| Feature | SPL Data | Our Data |
|---------|----------|----------|
| Frame count | 240 | 240 |
| Frame rate | 30fps | 60fps |
| Duration | 8 seconds | 4 seconds |
| Keypoints | 27 | 69 |
| Shared keypoints | 27 | 27/69 |
| Coordinate system | Court-centered | Different origin |
| Units | Feet | Feet |
| Participants | 1 | 5 |
| Shots | 125 | 345 train + 113 test |

### Target Distribution Comparison

| Target | SPL (mean +/- std) | Our Data (mean +/- std) |
|--------|-------|----------|
| Angle (degrees) | 43.86 +/- 1.79 | 45.48 +/- 4.87 |
| Depth (inches) | 9.75 +/- 3.98 | 9.66 +/- 5.40 |
| Left/Right (inches) | 0.15 +/- 3.53 | -0.78 +/- 3.80 |

SPL participant has much lower angle variance (std=1.79 vs 4.87) - very consistent shooter.

### Missing Keypoints (42 not in SPL)

All detailed hand joints (finger DIP/PIP/MCP/distal for all 5 fingers on both hands), mid_hip, and neck. These hand features are critical for our best pipeline.

## Experiments Conducted

### Experiment 1: Direct Feature Transfer (Combined Training)

Combined SPL + our training data with body-relative features on shared keypoints.

| Target | Our Only MSE | Combined (w=0.3) | Change |
|--------|-------------|-------------------|--------|
| Angle | 9.290 | 9.403 | +1.2% WORSE |
| Depth | 24.913 | 24.816 | -0.4% |
| LR | 12.345 | 12.210 | -1.1% |

**Verdict**: Negligible difference. SPL data does not help with direct combined training.

### Experiment 2: Pretrain + Finetune

Pretrain Ridge on SPL data, fine-tune residuals on our data.

| Target | Baseline MSE | Pretrain+FT | Change |
|--------|-------------|-------------|--------|
| Angle | 9.290 | 9.321 | +0.3% |
| Depth | 24.913 | 27.056 | +8.6% WORSE |
| LR | 12.345 | 12.256 | -0.7% |

**Verdict**: Pretraining on SPL actively hurts depth prediction.

### Experiment 3: PCA Pretraining (Combined Data)

PCA on combined SPL + our release-window features (5 frames x 27 kps x 3 coords = 405 features).

| PCA Components | Combined angle | Our-only angle | Combined depth | Our-only depth |
|---------------|----------------|----------------|----------------|----------------|
| 10 | 7.100 | 7.145 | 27.614 | 26.567 |
| 20 | 6.460 | 6.276 | 24.693 | 25.283 |
| 30 | 6.482 | 6.615 | 24.856 | 24.478 |

**Verdict**: Mixed results. Combined PCA hurts angle at 20 components, helps depth marginally. Overall negligible.

### Experiment 4: SPL Template Similarity Features

Built a motion template from SPL's consistent shooting pattern. Computed 28 similarity features measuring how each of our shots deviates from the SPL template.

**Similarity features alone:**
| Target | CV MSE | Pearson r |
|--------|--------|-----------|
| Angle | 6.433 | 0.853 |
| Depth | 26.190 | 0.332 |
| LR | 12.964 | 0.321 |

Angle similarity features are strong (r=0.85), but depth/LR are weak.

### Experiment 5: Combined HC + PLS + SPL Similarity in Per-Example Pipeline

This is the definitive test - can SPL similarity features improve our best pipeline?

| Target | HC+PLS (baseline) | HC+PLS+SIM | Change |
|--------|-------------------|------------|--------|
| Angle | 0.004123 | 0.004462 | +8.2% WORSE |
| Depth | 0.004895 | 0.005282 | +7.9% WORSE |
| LR | 0.006038 | 0.005920 | -2.0% better |
| Mean | 0.005018 | 0.005221 | +4.0% WORSE |

**Verdict**: SPL similarity features HURT overall performance. The 2% LR improvement does not compensate for angle/depth degradation.

### Correlation with Sub 1350 (Best Submission)

All SPL-enhanced submissions have r > 0.98 with Sub 1350 - minimal diversity.

## Submissions Generated

| Sub # | Config | dw | lw | CV Mean MSE | Notes |
|-------|--------|----|----|-------------|-------|
| 1518 | HC+PLS | 0.30 | 0.50 | 0.005018 | Equivalent to Sub 1350 |
| 1519 | HC+PLS | 0.20 | 0.40 | 0.005018 | Conservative blend |
| 1520 | HC+PLS+SIM | 0.30 | 0.50 | 0.005221 | SPL features, worse |
| 1521 | HC+PLS+SIM | 0.20 | 0.40 | 0.005221 | SPL features, conservative |
| 1522 | SIM angle-only | 0.10 | 0.30/0.50 | 0.005131 | SPL for angle only |
| 1523 | SIM angle-only | 0.05 | 0.30/0.50 | 0.005131 | SPL angle minimal |
| 1524 | SIM angle-only | 0.00 | 0.30/0.50 | 0.005131 | No angle weight (same as 1518) |

## Key Findings

1. **SPL and our data have incompatible feature spaces**: SPL has 27 keypoints, we have 69. The 42 missing hand keypoints are critical for our predictions.

2. **Coordinate systems differ**: SPL uses court-centered coordinates, our data uses a different origin. Body-relative features must be used for transfer, which lose the hoop-relative signal that's key to our pipeline.

3. **SPL participant is too consistent**: With angle std=1.79 (vs our 4.87), the SPL data represents a narrow slice of the variation we need to predict.

4. **Feature-target relationships differ between datasets**: Top predictive features in SPL data and our data have only 20-40% overlap, indicating different shooting mechanics patterns.

5. **SPL similarity features are redundant with existing features**: They capture body posture information already encoded in our hoop-relative features, but with less precision (missing hand joints, missing hoop reference frame).

6. **30fps vs 60fps temporal mismatch**: SPL at 30fps captures slower dynamics, requiring resampling that adds interpolation noise.

## Conclusion

The SPL Open Data free throw dataset is NOT useful for improving our predictions due to:
- Keypoint incompatibility (27 vs 69, missing hand joints)
- Single participant with narrow variation
- Different recording setup (frame rate, coordinate system)
- Feature-target relationships that don't transfer

The 125 SPL shots add noise rather than signal to our 345-shot training set. Our existing per-example locally weighted Ridge pipeline with HC+PLS features remains the best approach.

## Recommendations

- Do NOT include SPL data in the main pipeline
- Do NOT submit SPL-enhanced submissions to LB (r>0.98 with Sub 1350, unlikely to improve)
- If more external basketball motion capture data becomes available, prioritize data with:
  - Matching keypoint sets (69+ keypoints with detailed hand joints)
  - Similar frame rates (60fps)
  - Multiple diverse participants
  - Compatible target definitions
