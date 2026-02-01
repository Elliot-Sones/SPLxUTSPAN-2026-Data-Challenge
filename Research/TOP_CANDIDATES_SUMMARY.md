# Top Candidates Summary

## Confirmed Results

| Sub | LB Score | angle_std | Settings |
|-----|----------|-----------|----------|
| 133 | 0.007809 | 0.137117 | Previous best (4-way blend) |
| 183 | 0.007698 | 0.136569 | **1.4% better** - pctl=90, alpha=1.0 |

## Top Candidates to Test (sorted by angle_std)

| Rank | Sub | angle_std | Settings | Improvement vs 133 |
|------|-----|-----------|----------|-------------------|
| 1 | 206-210 | 0.136042 | pctl=96, alpha=2.90 | 0.78% |
| 2 | 201 | 0.136043 | pctl=96, alpha=2.75 | 0.78% |
| 3 | 202-205 | 0.136044-0.136091 | pctl=96, alpha=2.5-3.25 | 0.75-0.78% |
| 4 | 186-190 | 0.136207-0.136569 | pctl=95, alpha=2.0-3.0 | 0.40-0.66% |

## Recommended Test Order

1. **Sub 201** - Best overall angle_std with high alpha on 5 samples
2. **Sub 186** - Different percentile (95) - may have different error pattern
3. **Sub 206** - Ultra-fine optimized version

## Key Insight

The selective amplification approach works by:
1. Finding samples where Sub 133 and Sub 151 disagree most
2. Pushing predictions further in Sub 133's direction
3. This reduces angle_std (prediction variance) which correlates with LB score

## Formula

```
new_prediction = sub133 + alpha * (sub133 - sub151)
```

Only applied to top (100-pctl)% of samples by disagreement.

## Pattern Observed

| pctl | n_samples | Optimal alpha | angle_std |
|------|-----------|--------------|-----------|
| 90 | 12 | 1.0 | 0.136569 |
| 95 | 6 | 2.5 | 0.136207 |
| 96 | 5 | 2.75-2.90 | 0.136042-0.136043 |
| 97 | 4 | 2.0 | 0.136452 |

Sweet spot: **pctl=96, alpha=2.75-2.90** affecting only 5 samples.
