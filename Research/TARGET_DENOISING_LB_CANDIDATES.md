# Target Denoising - LB Test Candidates

## Date: 2026-02-14

## Context
Target denoising was tested on 2026-02-09 (see TARGET_DENOISING_RESULTS.md) and showed
a -30.6% LOO improvement using Ridge LOO smoothing. However it was NEVER tested on the
Kaggle LB. The original submissions (1538-1544) blended with Sub 784.

This run generates blends with the current LB best (Sub 2169, LB 0.006552) and Sub 2063
(LB 0.006603) for LB testing.

## Reproduced LOO Results

### Baseline (no denoising, bandwidth_quantile=0.5, Ridge alpha=10)
- angle:      LOO MSE = 0.002511
- depth:      LOO MSE = 0.004510
- left_right: LOO MSE = 0.004209
- MEAN:       LOO MSE = 0.003743

### ridge_a0.5_ra10 (best config, blend_alpha=0.5)
- angle:      LOO MSE = 0.002186 (-12.9%)
- depth:      LOO MSE = 0.002838 (-37.1%)
- left_right: LOO MSE = 0.002773 (-34.1%)
- MEAN:       LOO MSE = 0.002599 (-30.6%)
- Standalone: Sub 1538 (confirmed identical to Sub 2349 re-run, max_diff=0.0)

### ridge_a0.3_ra10 (blend_alpha=0.3)
- angle:      LOO MSE = 0.002261 (-9.9%)
- depth:      LOO MSE = 0.003369 (-25.3%)
- left_right: LOO MSE = 0.003207 (-23.8%)
- MEAN:       LOO MSE = 0.002946 (-21.3%)
- Standalone: Sub 1540 (confirmed identical to Sub 2350)

### ridge_a0.2_ra10 (blend_alpha=0.2)
- angle:      LOO MSE = 0.002326 (-7.3%)
- depth:      LOO MSE = 0.003703 (-17.9%)
- left_right: LOO MSE = 0.003494 (-17.0%)
- MEAN:       LOO MSE = 0.003174 (-15.2%)
- Standalone: Sub 1542 (confirmed identical to Sub 2351)

## Correlation with Sub 2169

| Config | Angle r | Depth r | LR r |
|--------|---------|---------|------|
| ridge_a0.5_ra10 | 0.994641 | 0.990423 | 0.981842 |
| ridge_a0.3_ra10 | 0.993174 | 0.991678 | 0.981882 |
| ridge_a0.2_ra10 | 0.991992 | 0.990640 | 0.980783 |

Correlations are very high (>0.98), indicating low diversity. This means blending is
unlikely to produce large improvements, but small gains are possible.

## Generated Submissions

### Blends with Sub 2169 (current LB best = 0.006552)

| Sub # | Description | Priority |
|-------|-------------|----------|
| 2283 | 10% ridge_a0.5_ra10 + 90% Sub 2169 | HIGH |
| 2284 | 20% ridge_a0.5_ra10 + 80% Sub 2169 | MEDIUM |
| 2285 | 30% ridge_a0.5_ra10 + 70% Sub 2169 | LOW |
| 2286 | 10% ridge_a0.3_ra10 + 90% Sub 2169 | MEDIUM |
| 2287 | 20% ridge_a0.3_ra10 + 80% Sub 2169 | LOW |
| 2288 | 30% ridge_a0.3_ra10 + 70% Sub 2169 | LOW |
| 2289 | 10% ridge_a0.2_ra10 + 90% Sub 2169 | MEDIUM |
| 2290 | 20% ridge_a0.2_ra10 + 80% Sub 2169 | LOW |
| 2291 | 30% ridge_a0.2_ra10 + 70% Sub 2169 | LOW |

### Blends with Sub 2063 (LB 0.006603)

| Sub # | Description | Priority |
|-------|-------------|----------|
| 2292 | 10% ridge_a0.5_ra10 + 90% Sub 2063 | MEDIUM |
| 2293 | 20% ridge_a0.5_ra10 + 80% Sub 2063 | LOW |

### Re-run Standalones (confirmed identical)

| Sub # | Description |
|-------|-------------|
| 2349 | STANDALONE ridge_a0.5_ra10 (identical to Sub 1538) |
| 2350 | STANDALONE ridge_a0.3_ra10 (identical to Sub 1540) |
| 2351 | STANDALONE ridge_a0.2_ra10 (identical to Sub 1542) |

## LB Testing Priority

1. **Sub 2283** (10% denoising + 90% Sub 2169) - safest blend, most likely to beat 2169
2. **Sub 2286** (10% ridge_a0.3 + 90% Sub 2169) - more conservative denoising
3. **Sub 2284** (20% denoising + 80% Sub 2169) - medium risk
4. **Sub 1538** (standalone best denoising) - baseline for standalone performance

## Risk Assessment

The LOO improvements are likely OPTIMISTIC because:
- Ridge LOO denoising replaces each target with alpha * (LOO Ridge prediction)
- This creates circular reasoning: the model trains on smoothed versions of its own predictions
- LOO residuals shrink by construction, not because of genuine noise removal
- Previous experience shows CV-LB gap of ~80% for this pipeline
- Cauchy kernel had similar issue: -4.42% LOO but only -0.21% LB at 10% blend

Conservative blend weights (10%) are recommended. The correlation with Sub 2169
is >0.98, so any genuine signal should show at even small blend weights.

## Pipeline Details
- Features: 213 (198 hoop-relative coords at target-specific frames + 15 PLS) per target
- Model: Per-example locally weighted Ridge, bandwidth_quantile=0.5, Ridge alpha=10
- Denoising: Ridge LOO smoothing with Gaussian kernel (sigma = median pairwise distance)
- Denoising formula: y_denoised[i] = (1-alpha) * y_orig[i] + alpha * ridge_loo_pred[i]
- Scripts: scripts/target_denoising.py (original), scripts/target_denoising_focused.py (this run)
