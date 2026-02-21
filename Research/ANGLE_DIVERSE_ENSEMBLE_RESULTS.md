# Angle-Diverse Ensemble Results

## Problem

Most submissions (784, 1350, 1421, 1366, etc.) use IDENTICAL angle predictions.
This limits ensemble diversity and potential improvement.

## Solution

Sub 1109 (physics-based angle) provides angle diversity:
- Angle correlation with Sub 784: 0.999495
- Mean absolute difference: 0.003217
- Sub 1109 LB: 0.007223 (nearly tied with 784 at 0.007224)

## Strategy

1. Blend angle predictions from Sub 784 (ML) and Sub 1109 (physics)
2. Use best depth/LR from Sub 1350 (LB 0.006776) or Sub 1421 (LB 0.006789)
3. Test different angle blend ratios (10%, 20%, 30%, 50%, 100% physics)

## Generated Submissions

### Sub 1461: angle_10pct_physics_depth_1350
- Angle weights: {784: 0.9, 1109: 0.1}
- Depth/LR source: Sub 1350
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1462: angle_20pct_physics_depth_1350
- Angle weights: {784: 0.8, 1109: 0.2}
- Depth/LR source: Sub 1350
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1463: angle_30pct_physics_depth_1350
- Angle weights: {784: 0.7, 1109: 0.3}
- Depth/LR source: Sub 1350
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1464: angle_50pct_physics_depth_1350
- Angle weights: {784: 0.5, 1109: 0.5}
- Depth/LR source: Sub 1350
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1465: angle_1109_depth_1350
- Angle weights: {1109: 1.0}
- Depth/LR source: Sub 1350
- Correlation with Sub 1350: angle=0.999, depth=1.000, lr=1.000

### Sub 1466: angle_10pct_physics_depth_1421
- Angle weights: {784: 0.9, 1109: 0.1}
- Depth/LR source: Sub 1421
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1467: angle_20pct_physics_depth_1421
- Angle weights: {784: 0.8, 1109: 0.2}
- Depth/LR source: Sub 1421
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1468: angle_50pct_physics_depth_1421
- Angle weights: {784: 0.5, 1109: 0.5}
- Depth/LR source: Sub 1421
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1469: angle_3way_uniform_depth_1350
- Angle weights: {784: 0.33, 1109: 0.33, 1350: 0.34}
- Depth/LR source: Sub 1350
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

### Sub 1470: angle_3way_quality_depth_1350
- Angle weights: {784: 0.1, 1109: 0.1, 1350: 0.8}
- Depth/LR source: Sub 1350
- Correlation with Sub 1350: angle=1.000, depth=1.000, lr=1.000

## Expected Performance

- If physics angle adds value: may beat Sub 1350 (0.006776)
- If physics angle hurts: will be worse than Sub 1350
- Optimal blend likely in 10-30% physics range (Sub 1109 used 10%)

## Key Insights

1. Sub 1109 angle differs from Sub 784 by mean 0.0048 (small but non-zero)
2. Physics angle correlation 0.996 with ML angle (high but not 1.0)
3. Breaking angle duplication may unlock ensemble potential
4. Per-shot adaptive angle selection could be next step

## Next Steps

1. Test submissions 1461-1470 on leaderboard
2. If improvement, implement per-shot angle selection (ML vs physics)
3. Find or create more angle-diverse models
4. Try per-shot adaptive weighting using shot similarity

