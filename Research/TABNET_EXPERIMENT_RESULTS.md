# TabNet Experiment Results

## Summary

TabNet was the only model from the recommended 4-phase approach that had NOT been tested. Initial CV testing showed +76% improvement over Ridge baseline.

## CV Results (5-Fold GroupKFold)

| Target | Ridge MSE | TabNet MSE | Improvement |
|--------|-----------|------------|-------------|
| angle | 0.090 | 0.011 | +88% |
| depth | 0.136 | 0.024 | +83% |
| left_right | 0.050 | 0.030 | +39% |
| **Total** | **0.092** | **0.022** | **+76%** |

## TabNet Configuration

```python
TabNetRegressor(
    n_d=16, n_a=16,      # Width of decision/attention
    n_steps=4,            # Number of decision steps
    gamma=1.5,            # Coefficient for feature reusage
    lambda_sparse=1e-4,   # Sparsity regularization
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax'
)
```

## Training Results

### Per-Target Best Epochs (5-fold CV)
- **angle**: 103, 62, 99, 50, 50 -> avg 72
- **depth**: 29, 80, 34, 135, 47 -> avg 65
- **left_right**: 35, 93, 34, 54, 30 -> avg 49

### Final Model Predictions
- angle: mean=0.6683, std=0.1384
- depth: mean=0.5645, std=0.0985
- left_right: mean=0.4227, std=0.0549

## Submissions Created

### TabNet-Only
- **Sub 663**: TabNet only
  - angle_std: 0.1390 (target ~0.137)
  - depth_mean: 0.5055 (target 0.5055)

### Blended with Sub 219 (Best LB: 0.007682)
| Sub | Blend | angle_std |
|-----|-------|-----------|
| 664 | 10% TabNet + 90% Sub219 | 0.1367 |
| 665 | 20% TabNet + 80% Sub219 | 0.1364 |
| 666 | 30% TabNet + 70% Sub219 | 0.1362 |
| 667 | 40% TabNet + 60% Sub219 | 0.1362 |
| 668 | 50% TabNet + 50% Sub219 | 0.1363 |

### Blended with Sub 133 (Previous Best: 0.007809)
| Sub | Blend | angle_std |
|-----|-------|-----------|
| 669 | 10% TabNet + 90% Sub133 | 0.1371 |
| 670 | 20% TabNet + 80% Sub133 | 0.1366 |
| 671 | 30% TabNet + 70% Sub133 | 0.1363 |

## Recommendations

1. **Test Sub 663** (TabNet-only) first to establish baseline
2. **Test Sub 664** (10% TabNet blend) as conservative approach
3. **Test Sub 667** (40% TabNet blend) if 10% shows improvement

## Key Insight

TabNet's attention mechanism may be learning feature interactions that tree-based models miss. The +76% CV improvement is substantial, but CV-to-LB gap is the key uncertainty.

## Files Created

- `scripts/tabnet_experiment.py` - Initial CV testing
- `scripts/tabnet_submission.py` - Full submission generation
- Submissions 663-671 in `/submission/`
