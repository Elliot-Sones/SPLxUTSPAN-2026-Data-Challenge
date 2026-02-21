# Semi-Supervised / Transductive Pipeline Results

## Date: 2026-02-09

## Overview

Tested whether using the 113 unlabeled test shots (features known, labels unknown)
can improve predictions over the current best approach (per-example locally weighted Ridge).

## Setup
- Script: `scripts/semisupervised_pipeline.py`
- Baseline: per-example locally weighted Ridge (same as Sub 1350)
  - StandardScaler + PLS fit on training data only (345 shots)
  - Gaussian kernel weighting, bandwidth=0.5 quantile, alpha=10
- All methods use the same HC features (198) + PLS (15) = 213 features per target

## Approaches Tested

### 1. Transductive Feature Normalization
Fit StandardScaler on all 458 shots (train + test) instead of just 345 train.

**Result: WORSE (+3-8% LOO MSE)**
- Angle: 0.002727 vs 0.002511 (+8.61%)
- Depth: 0.004654 vs 0.004510 (+3.19%)
- Left_right: 0.004474 vs 0.004209 (+6.31%)

**Why**: Including test features in the scaler distorts within-player standardization.
Since we have only 5 players with varying numbers of train/test shots, adding test
features shifts the mean/std in ways that harm the locally weighted model.

### 1b. Transductive PLS Only
Fit PLS scaler on all 458 shots but use original scaler for Ridge.

**Result: TIE (0.00% change)**
- PLS is fitted using labels (supervised), so the scaler has negligible impact on
  the PLS components themselves. The PLS model is identical.

### 2. Self-Training with Pseudo-Labels
Iterative: train on 345 labeled, predict test, add 25% most confident test predictions
as pseudo-labels, retrain on 345 + pseudo, repeat 3x.

**Result: MIXED**
- Angle: 0.002519 vs 0.002511 (+0.36%) -- tie
- Depth: 0.004175 vs 0.004510 (-7.43%) -- IMPROVEMENT
- Left_right: 0.004129 vs 0.004209 (-1.90%) -- slight improvement
- Mean: 0.003608 vs 0.003743 (-3.61%)

Self-training converges by iteration 2 (30 pseudo-labeled shots per player).
The depth improvement is notable and comes from pseudo-labels adding nearby-ish
support data that helps the locally weighted Ridge find better neighborhoods.

**CAUTION**: LOO for per-example models has 33-81% CV-LB gap historically.
The depth -7.43% LOO improvement may not transfer to LB.

### 3. Label Spreading (Graph-based)
sklearn LabelSpreading-style iterative propagation.

**Result: TERRIBLE (100-10000x worse)**
- Alpha=0.1: LOO MSE 0.20-0.25 (vs baseline ~0.003)
- Alpha=0.5: LOO MSE 0.09-0.10

**Why**: Label spreading works by propagating values through graph edges.
With 5 players and continuous targets, the affinity graph is too noisy to
propagate meaningful regression values. This approach is designed for classification.

### 4. Graph-Regularized Ridge
Ridge regression with Laplacian regularization over all 458 shots.

**Result: TERRIBLE (100-1000x worse)**
- All beta values: LOO MSE 0.24-0.88

**Why**: The Laplacian regularization encourages ALL predictions to be similar
(smooth over the graph). This overwrites the regression signal with a smoothing
prior that averages everything toward the mean. Fundamentally wrong for this problem.

## Submission Summary

| Sub # | Description | CV (mean) | Notes |
|-------|-------------|-----------|-------|
| 1569 | Standalone best-per-target | 0.003605 | angle=baseline, depth/LR=self_training |
| 1570 | Blend with Sub 784 (aw=0, dw=0.30, lw=0.50) | - | Optimal blend weights |
| 1571 | Blend with Sub 784 (aw=0, dw=0.20, lw=0.30) | - | Conservative |
| 1572 | Blend with Sub 784 (aw=0, dw=0.40, lw=0.60) | - | Aggressive |
| 1573 | Blend with Sub 784 (aw=0.10, dw=0.30, lw=0.50) | - | With angle |
| 1574 | 10% blend with Sub 1350 | - | Very similar to 1350 |
| 1575 | 20% blend with Sub 1350 | - | |
| 1576 | 30% blend with Sub 1350 | - | |
| 1577 | Transductive standalone | 0.003952 | WORSE than baseline |
| 1578 | Transductive blend (aw=0, dw=0.30, lw=0.50) | - | |
| 1579 | Trans. PLS only standalone | 0.003743 | Same as baseline |
| 1580 | Trans. PLS only blend | - | |
| 1581 | Self-training standalone | 0.003608 | Best CV |
| 1582 | Self-training blend (aw=0, dw=0.30, lw=0.50) | - | |

## Correlation with References

Best-per-target predictions vs references:
- Angle: r(Sub784)=0.9334, r(Sub1350)=0.9334
- Depth: r(Sub784)=0.9445, r(Sub1350)=0.9724
- Left_right: r(Sub784)=0.8620, r(Sub1350)=0.9779

Sub 1570 (semisup blend) vs Sub 1350:
- Very high correlation (r > 0.99 for all targets)
- Mean absolute difference: 0.002-0.004 per target

## Key Findings

1. **Transductive feature normalization HURTS**: Including test shots in StandardScaler
   distorts per-player statistics. The original train-only scaler is better.

2. **Self-training shows LOO improvement for depth (-7.43%)**: Adding 30 pseudo-labeled
   test shots per player helps the locally weighted Ridge. But LOO is systematically
   optimistic for per-example models.

3. **Label spreading and graph-regularized Ridge are catastrophic failures**: These methods
   are fundamentally unsuited for small-sample regression. They average everything toward
   the mean.

4. **Low diversity vs Sub 1350**: All semi-supervised predictions have r > 0.97 with
   Sub 1350. Even if self-training helps on LB, the improvement via blending would be
   marginal.

## Recommended for LB Testing

Priority order:
1. **Sub 1570**: Blend with Sub 784 at optimal weights (most likely to help if self-training works)
2. **Sub 1574**: 10% blend with Sub 1350 (minimal risk, minimal change)
3. **Sub 1582**: Self-training blend (aw=0, dw=0.30, lw=0.50)

## Conclusion

Semi-supervised approaches provide limited value for this problem. The self-training
shows some LOO improvement for depth but predictions are highly correlated with the
existing best (Sub 1350). The 33-81% CV-LB gap makes it uncertain whether the LOO
improvement will transfer. The most promising submission is Sub 1570 (self-training
best-per-target blended with Sub 784 at dw=0.30, lw=0.50).
