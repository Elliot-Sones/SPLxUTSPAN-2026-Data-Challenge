# Temporal Dynamics Submissions - Testing Priority

**Generated**: 2026-02-08
**Total Submissions**: 9 (Sub 1478-1486)
**Best CV**: 0.006888 (Sub 1480, Blend standalone)

## Quick Reference Table

| Sub  | Type          | Config                     | Mean CV  | Angle CV | Depth CV | LR CV    | Angle r | Depth r | LR r   | Notes |
|------|---------------|----------------------------|----------|----------|----------|----------|---------|---------|--------|-------|
| 1480 | Blend         | Standalone                 | 0.006888 | 0.007106 | 0.005916 | 0.007642 | 0.9610  | 0.8967  | 0.8826 | **BEST CV**, good diversity |
| 1481 | Blend+Sub784  | dw=0.30, lw=0.50          | -        | -        | -        | -        | -       | -       | -      | Optimal Sub784 weights |
| 1486 | DTW+Sub784    | dw=0.40, lw=0.60 (agg)    | -        | -        | -        | -        | -       | -       | -      | Aggressive depth focus |
| 1478 | Ensemble      | Standalone                 | 0.007625 | 0.006840 | 0.007763 | 0.008271 | 0.9721  | 0.9162  | 0.8522 | Conservative baseline |
| 1479 | DTW           | Standalone                 | 0.007999 | 0.009417 | 0.006035 | 0.008544 | 0.9187  | 0.8684  | 0.8715 | Depth specialist |
| 1482 | Blend+Sub784  | dw=0.20, lw=0.40 (cons)   | -        | -        | -        | -        | -       | -       | -      | Conservative blend |
| 1483 | Blend+Sub784  | dw=0.40, lw=0.60 (agg)    | -        | -        | -        | -        | -       | -       | -      | Aggressive blend |
| 1484 | DTW+Sub784    | dw=0.30, lw=0.50          | -        | -        | -        | -        | -       | -       | -      | DTW optimal weights |
| 1485 | DTW+Sub784    | dw=0.20, lw=0.40 (cons)   | -        | -        | -        | -        | -       | -       | -      | DTW conservative |

## Testing Priority (Top 3)

### 1. Sub 1481 (HIGH PRIORITY)
- **Config**: Blend + Sub 784, dw=0.30, lw=0.50
- **Why**: Matches Sub 1350's optimal weights (0.30 depth, 0.50 LR)
- **Expected LB**: 0.0068-0.0072
- **Diversity**: r=0.8967 (depth), r=0.8826 (LR) - good complementary signal
- **Risk**: Low (proven weight configuration)

### 2. Sub 1480 (MEDIUM PRIORITY)
- **Config**: Standalone blend (best CV)
- **Why**: Best overall CV (0.006888), tests temporal approach in isolation
- **Expected LB**: 0.0074-0.0080
- **Diversity**: Moderate (r=0.8967 depth, r=0.9610 angle)
- **Risk**: Medium (may violate profile constraints: angle_std=0.1478, depth_mean=0.5070)

### 3. Sub 1486 (MEDIUM PRIORITY)
- **Config**: DTW + Sub 784, dw=0.40, lw=0.60 (aggressive)
- **Why**: Depth specialist approach (DTW CV 0.006035), high blend weight
- **Expected LB**: 0.0068-0.0072
- **Diversity**: Good for depth (r=0.8684)
- **Risk**: Medium (aggressive weights may overcorrect)

## Backup Submissions (Test if top 3 underperform)

### 4. Sub 1478 (Conservative baseline)
- Ensemble on trajectory features
- Safe bet if blend approaches fail
- Expected LB: 0.0076-0.0082

### 5. Sub 1482 (Conservative blend)
- dw=0.20, lw=0.40 - lower risk
- Expected LB: 0.0070-0.0074

## Expected Outcomes

### Best Case Scenario
- Sub 1481 or 1486 achieves LB 0.0068-0.0070
- Improvement over Sub 784 (0.007224) but not Sub 1350 (0.006776)
- Moderate diversity enables ensembling

### Realistic Scenario
- LB range: 0.0070-0.0075
- Competitive with Sub 784 but not breakthrough
- Good diversity for final ensemble (r < 0.90 for depth/LR)

### Worst Case Scenario
- LB > 0.0080 (worse than Sub 784)
- Temporal dynamics don't generalize to test set
- Full trajectories are too noisy for 345 training samples

## Key Insights for Interpretation

### If Sub 1481 performs well (LB < 0.0070):
- Trajectory dynamics ADD value for depth/LR
- Consider HYBRID approach: combine with Sub 1350's 3-frame features
- Generate more aggressive blends (dw=0.50+)

### If Sub 1480 performs well (LB < 0.0075):
- Temporal approach is viable standalone
- Profile constraints are indeed SOFT
- Generate more standalone temporal variants

### If all submissions underperform (LB > 0.0075):
- Full trajectories are TOO NOISY for this dataset
- Focus on Sub 1350's targeted frame approach
- Use temporal features only as AUXILIARY signal

## Next Steps After LB Testing

### If successful (LB < 0.0070):
1. Generate hybrid models: Sub 1350 features + temporal features
2. Target-specific approach: DTW for depth only, Sub 1350 for angle/LR
3. Ensemble Sub 1350 + temporal models at test time

### If moderate (LB 0.0070-0.0075):
1. Use temporal models for final ensemble diversity
2. Tune blend weights more aggressively
3. Try shorter trajectory windows (100-180 frames only)

### If unsuccessful (LB > 0.0075):
1. Abandon full trajectory approach
2. Focus on refining Sub 1350's per-example method
3. Investigate other diversity sources (different feature sets, cross-player transfer)

## File Locations

- **Script**: `scripts/temporal_dynamics_pipeline.py`
- **Research**: `Research/TEMPORAL_DYNAMICS_RESULTS.md`
- **Submissions**: `submission/submission_1478.csv` through `submission/submission_1486.csv`
- **Current best**: `submission/submission_1350.csv` (LB 0.006776)
- **Blend baseline**: `submission/submission_784.csv` (LB 0.007224)
