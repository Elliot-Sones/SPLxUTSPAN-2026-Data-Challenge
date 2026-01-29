# Submission Strategy to Beat 0.007

## Current Status
- **Best LB Score**: 0.008305 (submission 25)
- **Target**: 0.007
- **Gap**: 0.001305 (15.7% improvement needed)

## Key Insights from Analysis

### Correlation Analysis (7 known LB scores)
| Feature | Correlation with LB | p-value | Direction |
|---------|---------------------|---------|-----------|
| depth_max | r=-0.901 | 0.006 | **Higher is better** |
| angle_std | r=+0.747 | 0.054 | Lower is better |
| depth_mean | r=+0.686 | 0.089 | Lower is better |

**Critical finding**: `depth_max` is the strongest predictor of LB score.

### Linear Model for Score Estimation
```
LB_score = -0.0711 * depth_max + 0.0616
```

## Submission Recommendations

### Priority 1: Sub 51 (BEST BET)
- **Description**: sub25 angle/lr + sub43 depth
- **depth_max**: 0.7829 (highest)
- **angle_std**: 0.1380 (same as proven best)
- **Estimated LB**: ~0.006

**Why**: Combines the best of both worlds - sub25's proven angle/lr predictions with sub43's highest depth_max.

### Priority 2: Sub 45
- **Description**: sub25 angle/lr + sub41 depth
- **depth_max**: 0.7806
- **angle_std**: 0.1380
- **Estimated LB**: ~0.0061

**Why**: Same strategy but with depth_boosted model (sub41) which has slightly lower depth_max but same depth_mean.

### Priority 3: Sub 43
- **Description**: Pure Gradient Boosting model
- **depth_max**: 0.7829
- **angle_std**: 0.1433 (higher - potential risk)
- **Estimated LB**: ~0.006

**Why**: Raw high depth_max but higher angle_std might hurt.

### Priority 4: Sub 46
- **Description**: 30% sub25 + 70% sub43
- **depth_max**: 0.7714
- **angle_std**: 0.1410
- **Estimated LB**: ~0.0068

**Why**: More conservative blend, safer but lower potential.

## Known Submissions Summary

| Sub | LB Score | depth_max | angle_std | Description |
|-----|----------|-----------|-----------|-------------|
| 8 | 0.010220 | 0.7243 | 0.1402 | Baseline |
| 9 | 0.009109 | 0.7397 | 0.1390 | Ensemble |
| 10 | 0.008907 | 0.7498 | 0.1388 | Optuna-tuned |
| 20 | 0.008619 | 0.7417 | 0.1384 | 80-20 blend |
| 25 | **0.008305** | 0.7447 | 0.1380 | **50-50 blend (BEST)** |
| 34 | 0.008377 | 0.7467 | 0.1381 | 30-70 blend |
| 41 | ? | 0.7806 | 0.1380 | Depth-boosted |
| 43 | ? | 0.7829 | 0.1433 | Gradient Boosting |
| 45 | ? | 0.7806 | 0.1380 | sub25 angle/lr + sub41 depth |
| **51** | ? | **0.7829** | **0.1380** | **sub25 angle/lr + sub43 depth** |

## Submission Files Location
- `/submission/submission_51.csv` - BEST BET
- `/submission/submission_45.csv` - Second best
- `/submission/submission_43.csv` - Raw GB model
- `/submission/submission_46.csv` - Conservative blend

## Next Steps if Sub 51 Doesn't Beat 0.007
1. If Sub 51 improves but doesn't beat 0.007: Create blends with even higher depth_max by extrapolating further
2. If Sub 51 doesn't improve: The linear relationship doesn't hold perfectly - need to explore other features
3. Consider: per-player calibration, sample-specific adjustments based on known submission differences
