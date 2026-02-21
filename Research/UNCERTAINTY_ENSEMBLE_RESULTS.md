# Uncertainty-Weighted Ensemble Results

## Approach

Use prediction diversity (variance across models) as uncertainty proxy.
Weight models by:
1. Inverse variance (low disagreement = high confidence)
2. Quality scores (known LB performance)
3. Hybrid combination of uncertainty and quality

## Base Models

- Sub 784: LB 0.007224 (baseline)
- Sub 1350: LB 0.006776 (best, per-example V1)
- Sub 1421: LB 0.006789 (per-example V2)

## Prediction Variance Analysis

Mean variance per target:
- Angle: 0.00000000
- Depth: 0.00003654
- Left_right: 0.00023866

Per-model mean absolute deviation:
- Sub 784 (LB 0.007224): angle=0.000000, depth=0.006317, lr=0.015088
- Sub 1350 (LB 0.006776): angle=0.000000, depth=0.002740, lr=0.007141
- Sub 1421 (LB 0.006789): angle=0.000000, depth=0.003658, lr=0.007966

Quality scores (normalized inverse LB):
- Sub 784: 0.0000
- Sub 1350: 1.0000
- Sub 1421: 0.9710

## Generated Submissions

### Sub 1451: uniform
- Weights: uniform (1/3 each)
- Correlation with Sub 1350: angle=1.000, depth=0.999, lr=0.995

### Sub 1452: inverse_var
- Weights: mean=[0.19660905 0.42423601 0.37915495]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=0.999

### Sub 1453: quality_only
- Weights: mean=[0.         0.50736127 0.49263873]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1454: hybrid_50_50
- Weights: mean=[0.19606097 0.42444095 0.37949808]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=0.999

### Sub 1455: hybrid_70_30
- Weights: mean=[0.19533813 0.42471152 0.37995036]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=0.999

### Sub 1456: hybrid_30_70
- Weights: mean=[0.19637353 0.42432405 0.37930242]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=0.999

### Sub 1457: softmax_t0.1
- Weights: mean=[2.59698945e-05 5.72024994e-01 4.27949036e-01]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1458: softmax_t0.5
- Weights: mean=[0.06509785 0.4810117  0.45389045]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1459: softmax_t1.0
- Weights: mean=[0.15726192 0.42748222 0.41525585]
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=0.999

### Sub 1460: per_target
- Weights: per-target adaptive
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=0.999

## Strategy Explanations

1. **uniform**: Simple average (baseline)
2. **inverse_var**: Weight by inverse of average absolute deviation
3. **quality_only**: Weight by known LB scores only
4. **hybrid_X_Y**: X% quality + Y% uncertainty weighting
5. **softmax_tN**: Softmax over quality scores with temperature N
6. **per_target**: Different weights per target based on target-specific variance

## Expected Performance

- Uniform baseline: ~0.006780 (midpoint of 1350 and 1421)
- Quality-weighted: closer to 1350 (best model gets higher weight)
- Hybrid: balance between best model and diversity
- Per-target: may help if models have different strengths per target

## Key Insights

1. Prediction variance is highest for left_right (most uncertain target)
2. Sub 1350 has lowest deviation (most confident/stable)
3. Hybrid weighting balances trusting best model vs. ensemble diversity
4. Per-target weighting allows target-specific model selection
5. **CRITICAL**: Subs 784, 1350, 1421 have IDENTICAL angle predictions (variance=0)
   - This severely limits ensemble potential
   - Angle accounts for 1/3 of evaluation metric
   - Need angle-diverse models (Sub 1109 provides this)

## Limitations

These submissions (1451-1460) likely have LIMITED BENEFIT because:
- Zero angle diversity (all use Sub 784 angle)
- Only blending depth and left_right predictions
- Correlations with Sub 1350 are r > 0.999 (minimal diversity)

See ANGLE_DIVERSE_ENSEMBLE_RESULTS.md for follow-up with angle diversity.

## Next Steps

1. Test submissions 1451-1460 on leaderboard (expect marginal improvement at best)
2. Focus on angle-diverse ensembles (Subs 1461-1470)
3. Explore per-shot adaptive weighting using feature space similarity
4. Consider stacking with meta-learner

