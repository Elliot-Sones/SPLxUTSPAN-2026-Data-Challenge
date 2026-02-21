# Target Denoising Results

## Date: 2026-02-09

## Hypothesis
Training targets contain measurement noise from camera-based tracking. Denoising labels
before training per-example locally weighted Ridge could reduce overfitting to noise.

## Methods Tested

### Method A: KNN Smoothing
Replace each target with blend of original (1-alpha) and distance-weighted KNN average (alpha).
- Parameters: k in {5, 10, 15}, blend_alpha in {0.1, 0.2, 0.3, 0.5}
- 12 configurations tested

### Method B: Ridge LOO Smoothing
For each training sample, compute its leave-one-out locally weighted Ridge prediction,
then blend with original: y_new = (1-alpha) * y_orig + alpha * y_ridge_loo
- Parameters: blend_alpha in {0.1, 0.2, 0.3, 0.5}, ridge_alpha in {10, 50}
- 8 configurations tested

### Method C: Per-Player Mean Shrinkage
Pull each target toward the player's mean: y_new = (1-s) * y_orig + s * player_mean
- Parameters: shrinkage in {0.05, 0.10, 0.15, 0.20, 0.30}
- 5 configurations tested

### Method D: Adaptive KNN (outlier-only)
Only denoise samples whose target is > sigma_threshold std devs from their KNN neighborhood mean.
- Parameters: k in {8, 12}, sigma_threshold in {1.0, 1.5, 2.0}
- 6 configurations tested

## Pipeline
- Features: 213 (198 HC at target-specific frames + 15 PLS) per target - same as Sub 1350
- Model: Per-example locally weighted Ridge, bandwidth=0.5 quantile, alpha=10
- Evaluation: LOO on ORIGINAL (undenoised) targets
- Baseline: same pipeline with no denoising (LOO MSE = 0.003743)

## Results

### Top 10 Configurations (by LOO MSE on original targets)

| Rank | Config | Mean LOO MSE | Delta vs Baseline | Angle | Depth | LR |
|------|--------|-------------|-------------------|-------|-------|-----|
| 1 | ridge_a0.5_ra10 | 0.002599 | -30.6% | 0.002186 | 0.002838 | 0.002773 |
| 2 | ridge_a0.3_ra10 | 0.002946 | -21.3% | 0.002261 | 0.003369 | 0.003207 |
| 3 | ridge_a0.2_ra10 | 0.003174 | -15.2% | 0.002326 | 0.003703 | 0.003494 |
| 4 | ridge_a0.5_ra50 | 0.003191 | -14.8% | 0.002667 | 0.003416 | 0.003491 |
| 5 | ridge_a0.3_ra50 | 0.003308 | -11.6% | 0.002533 | 0.003741 | 0.003650 |
| 6 | ridge_a0.2_ra50 | 0.003418 | -8.7% | 0.002501 | 0.003960 | 0.003794 |
| 7 | ridge_a0.1_ra10 | 0.003440 | -8.1% | 0.002409 | 0.004084 | 0.003828 |
| 8 | ridge_a0.1_ra50 | 0.003563 | -4.8% | 0.002494 | 0.004216 | 0.003980 |
| 9 | knn_k5_a0.1 | 0.003745 | +0.0% | 0.002629 | 0.004392 | 0.004214 |
| 10 | knn_k10_a0.1 | 0.003754 | +0.3% | 0.002621 | 0.004426 | 0.004216 |

### Method Summary

| Method | Best Config | Best Mean LOO | Delta |
|--------|-------------|---------------|-------|
| Ridge LOO | alpha=0.5, ridge_alpha=10 | 0.002599 | -30.6% |
| KNN | k=5, alpha=0.1 | 0.003745 | +0.0% |
| Mean Shrinkage | s=0.05 | 0.003762 | +0.5% |
| Adaptive KNN | k=12, sigma=2.0 | 0.003853 | +2.9% |

### Key Observations

1. **Ridge LOO smoothing is the only method that helps in LOO evaluation**
   - Replaces each training target with a blend of original and LOO Ridge prediction
   - Higher blend alpha = more smoothing = better LOO (up to alpha=0.5)
   - Lower Ridge alpha (10 vs 50) is better (less regularized Ridge = more smoothing)

2. **KNN smoothing, mean shrinkage, and adaptive KNN all hurt or are neutral**
   - These methods add noise rather than removing it
   - The geometric smoothing (pull toward neighbors) is not aligned with the error structure

3. **WARNING: The LOO improvements are likely optimistic**
   - Ridge LOO denoising replaces each target with alpha * (LOO Ridge prediction)
   - This is essentially replacing the training label with a smoothed version of what
     the model would already predict - circular reasoning
   - The model then trains on targets that are closer to what it would predict anyway,
     making LOO residuals artificially small
   - Previous experience shows LOO for per-example models has ~80% CV-LB gap
   - A -30.6% LOO improvement might translate to ~0% or worse on LB

4. **High correlation with Sub 1350** (r > 0.94-0.98)
   - Low diversity means blending won't help
   - The denoised predictions are very similar to the original

## Diversity Analysis (vs Sub 1350)

| Config | Angle r | Depth r | LR r |
|--------|---------|---------|------|
| ridge_a0.5_ra10 | 0.9506 | 0.9802 | 0.9828 |
| ridge_a0.3_ra10 | 0.9445 | 0.9807 | 0.9824 |
| ridge_a0.2_ra10 | 0.9410 | 0.9793 | 0.9811 |

## Submissions Generated

| Sub # | Config | Description |
|-------|--------|-------------|
| 1538 | ridge_a0.5_ra10 | Standalone (best overall LOO) |
| 1539 | ridge_a0.5_ra10 | Blended with Sub 784 (aw=0, dw=0.30, lw=0.50) |
| 1540 | ridge_a0.3_ra10 | Standalone |
| 1541 | ridge_a0.3_ra10 | Blended with Sub 784 (aw=0, dw=0.30, lw=0.50) |
| 1542 | ridge_a0.2_ra10 | Standalone |
| 1543 | ridge_a0.2_ra10 | Blended with Sub 784 (aw=0, dw=0.30, lw=0.50) |
| 1544 | Best-per-target | Blended with Sub 784 (same as 1539 since same config won all targets) |

## LB Priority Recommendations

1. **Sub 1539** (ridge_a0.5_ra10 blended) - most aggressive denoising, strongest LOO improvement
2. **Sub 1543** (ridge_a0.2_ra10 blended) - conservative denoising, may generalize better

## Interpretation

The Ridge LOO denoising is mathematically interesting but likely overfits the LOO evaluation:
- By replacing training targets with alpha * LOO_prediction, we are smoothing toward the
  model's own predictions, which reduces LOO residuals by construction
- The real question is whether this smoothing removes true measurement noise or just
  reduces the effective training signal
- With alpha=0.5, half the target information comes from the model's own predictions,
  which could lead to a "regression to the mean" effect

Script: scripts/target_denoising.py
