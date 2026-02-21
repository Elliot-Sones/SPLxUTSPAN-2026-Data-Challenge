# Kernel Experiments for Per-Example Locally Weighted Ridge Regression

Date: 2026-02-12
Script: scripts/kernel_experiments.py
Runtime: 44.2s

## Objective

Test alternative kernel functions for the locally weighted Ridge regression that forms
the core of our best submissions. The Gaussian kernel drops off quadratically in the
exponent, potentially underweighting moderately-distant training points. Heavier-tailed
kernels could use more effective training data per prediction, reducing variance.

## Experiment Setup

- Base model: per-player locally weighted Ridge (same as per_example_pipeline.py)
- Features: 198 hoop-relative + 15 PLS per target at target-specific frames
- Target-specific frames: angle=153, depth=150, LR=170
- Baseline: Gaussian kernel, bandwidth_quantile=0.3, alpha=10

## Kernels Tested

| Kernel | Formula | Tail behavior | Support |
|--------|---------|---------------|---------|
| Gaussian | exp(-d^2 / (2*sigma^2)) | Light (quadratic exponent) | Infinite |
| Laplacian | exp(-d / sigma) | Moderate (linear exponent) | Infinite |
| Cauchy | 1 / (1 + (d/sigma)^2) | Very heavy (polynomial decay) | Infinite |
| Tricube | (1 - (d/sigma)^3)^3 | N/A | Compact (d < sigma) |
| Epanechnikov | max(0, 1 - (d/sigma)^2) | N/A | Compact (d < sigma) |

## Phase 1: Gaussian Baseline (bw=0.3, alpha=10)

| Target | LOO MSE | Eff Neighbors |
|--------|---------|---------------|
| angle | 0.002645 | 61.4 |
| depth | 0.004601 | 61.3 |
| left_right | 0.004331 | 61.4 |
| **mean** | **0.003859** | **61.3** |

## Phase 2: Laplacian Kernel Sweep

| BW | Mean MSE | Delta% | Eff N | Angle | Depth | LR |
|----|----------|--------|-------|-------|-------|-----|
| 0.1 | 0.004135 | +7.16% | 61.5 | 0.003131 | 0.004750 | 0.004524 |
| 0.2 | 0.004027 | +4.34% | 62.6 | 0.003012 | 0.004670 | 0.004397 |
| 0.3 | 0.003961 | +2.64% | 63.4 | 0.002931 | 0.004620 | 0.004332 |
| 0.4 | 0.003911 | +1.36% | 64.1 | 0.002866 | 0.004583 | 0.004286 |
| 0.5 | 0.003867 | +0.22% | 64.6 | 0.002811 | 0.004549 | 0.004243 |
| 0.6 | 0.003830 | -0.76% | 65.1 | 0.002760 | 0.004519 | 0.004210 |
| 0.7 | 0.003801 | -1.49% | 65.5 | 0.002709 | 0.004494 | 0.004202 |

Best Laplacian: bw=0.7, all targets prefer wider bandwidth.

## Phase 3: All Kernels at Best Bandwidth

| Kernel | Best BW | Mean MSE | Delta% | Eff N | Angle | Depth | LR |
|--------|---------|----------|--------|-------|-------|-------|-----|
| gaussian | 0.3 | 0.003859 | baseline | 61.3 | 0.002645 | 0.004601 | 0.004331 |
| laplacian | 0.7 | 0.003801 | -1.49% | 65.5 | 0.002709 | 0.004494 | 0.004202 |
| **cauchy** | **0.7** | **0.003716** | **-3.69%** | **65.7** | **0.002524** | **0.004451** | **0.004174** |
| epanechnikov | 0.7 | 0.005813 | +50.62% | 39.5 | 0.004477 | 0.006874 | 0.006087 |
| tricube | 0.7 | 0.006657 | +72.51% | 28.0 | 0.004987 | 0.007622 | 0.007362 |

Key finding: **Cauchy kernel is the best**, beating Gaussian by -3.69% in LOO MSE.
Compact-support kernels (Tricube, Epanechnikov) are much worse -- they use too few
effective neighbors (28-40 vs 61-66).

## Phase 4: Alpha Sweep for Cauchy (bw=0.7)

| Alpha | Mean MSE | Delta% | Angle | Depth | LR |
|-------|----------|--------|-------|-------|-----|
| 1.0 | 0.004424 | +14.63% | 0.002433 | 0.005183 | 0.005656 |
| 3.0 | 0.003811 | -1.25% | 0.002189 | 0.004611 | 0.004632 |
| **5.0** | **0.003689** | **-4.42%** | **0.002251** | **0.004483** | **0.004332** |
| 10.0 | 0.003716 | -3.69% | 0.002524 | 0.004451 | 0.004174 |
| 30.0 | 0.004229 | +9.59% | 0.003388 | 0.004705 | 0.004594 |
| 50.0 | 0.004648 | +20.45% | 0.003917 | 0.004947 | 0.005080 |
| 100.0 | 0.005357 | +38.81% | 0.004655 | 0.005413 | 0.006002 |

Best alpha for Cauchy: 5.0 (slightly lower regularization than baseline).

Laplacian alpha sweep (bw=0.7):

| Alpha | Mean MSE | Delta% |
|-------|----------|--------|
| 1.0 | 0.004194 | +8.67% |
| 3.0 | 0.003714 | -3.75% |
| **5.0** | **0.003668** | **-4.96%** |
| 10.0 | 0.003801 | -1.49% |
| 30.0 | 0.004472 | +15.89% |

Best Laplacian: bw=0.7, alpha=5.0, MSE=0.003668 (-4.96%).

## Phase 5: Per-Target Optimal Configurations

| Target | Kernel | BW | Alpha | LOO MSE | Delta vs Gaussian |
|--------|--------|----|-------|---------|-------------------|
| angle | cauchy | 0.7 | 3.0 | 0.002189 | -17.23% |
| depth | cauchy | 0.7 | 10.0 | 0.004451 | -3.26% |
| left_right | cauchy | 0.6 | 10.0 | 0.004165 | -3.84% |
| **mean** | | | | **0.003602** | **-6.67%** |

Angle benefits most from the Cauchy kernel, consistent with angle being the target
that overfits the most (2.57x LOO-to-LB gap).

## Phase 6: Effective Neighbors

| Kernel | Best BW | Angle | Depth | LR | Mean |
|--------|---------|-------|-------|-----|------|
| gaussian | 0.3 | 61.4 | 61.3 | 61.4 | 61.3 |
| laplacian | 0.7 | 65.5 | 65.5 | 65.5 | 65.5 |
| cauchy | 0.7 | 65.8 | 65.7 | 65.8 | 65.7 |
| epanechnikov | 0.7 | 39.8 | 39.7 | 39.0 | 39.5 |
| tricube | 0.7 | 28.2 | 28.4 | 27.4 | 28.0 |

Cauchy and Laplacian use ~4 more effective neighbors than Gaussian (65.7 vs 61.3).
For per-player models with ~66-74 training shots, this means using virtually ALL
training data vs leaving out ~10 points with Gaussian.

## Phase 7: Diversity with Sub 2063

Per-target optimal (Cauchy):

| Target | r(Sub 2063) | r(Sub 784) |
|--------|-------------|------------|
| angle | 0.9588 | 0.9100 |
| depth | 0.9586 | 0.9323 |
| left_right | 0.9198 | 0.7831 |

Per-kernel at best bw:

| Kernel | BW | r_angle | r_depth | r_LR | mean r |
|--------|-----|---------|---------|------|--------|
| laplacian | 0.7 | 0.9795 | 0.9695 | 0.9462 | 0.9651 |
| cauchy | 0.7 | 0.9759 | 0.9586 | 0.9100 | 0.9482 |
| epanechnikov | 0.7 | 0.9822 | 0.9066 | 0.9256 | 0.9382 |
| tricube | 0.7 | 0.9761 | 0.8814 | 0.8937 | 0.9171 |

Cauchy has moderate diversity (r=0.95) -- enough for small blending gains but not
revolutionary diversity. Left_right has most diversity (r=0.92).

## Submissions Generated

### Per-target optimal Cauchy + Sub 784
| Sub | Blend | Config |
|-----|-------|--------|
| 2191 | aw=0.50 dw=0.30 lw=0.50 | angle: cauchy bw=0.7 a=3.0, depth: cauchy bw=0.7 a=10, LR: cauchy bw=0.6 a=10 |
| 2192 | aw=0.30 dw=0.20 lw=0.30 | same |
| 2193 | aw=0.70 dw=0.40 lw=0.60 | same |

### Per-target optimal Cauchy + Sub 2063
| Sub | Blend |
|-----|-------|
| 2194 | 10% kernel + 90% Sub2063 |
| 2195 | 20% kernel + 80% Sub2063 |
| 2196 | 30% kernel + 70% Sub2063 |

### Cauchy bw=0.7 alpha=5.0 (single best config) + Sub 784
| Sub | Blend |
|-----|-------|
| 2197 | aw=0.50 dw=0.30 lw=0.50 |
| 2198 | aw=0.30 dw=0.20 lw=0.30 |

### Cauchy single config + Sub 2063
| Sub | Blend |
|-----|-------|
| 2199 | 10% cauchy + 90% Sub2063 |
| 2200 | 20% cauchy + 80% Sub2063 |
| 2201 | 30% cauchy + 70% Sub2063 |

## Key Findings

1. **Cauchy kernel is best** -- -3.69% mean LOO MSE at alpha=10, -4.42% at alpha=5.0.
   Per-target optimal gives -6.67%.

2. **Heavy tails > compact support** -- Cauchy (very heavy tails, -3.69%) >
   Laplacian (moderate tails, -1.49%) > Gaussian (light tails, baseline) >>
   Epanechnikov (+50.62%) > Tricube (+72.51%).

3. **More effective neighbors = better** -- Cauchy/Laplacian use 65+ effective
   neighbors vs 61 for Gaussian. With only ~70 training points per player,
   using all of them matters.

4. **Lower alpha preferred** -- Both Cauchy and Laplacian prefer alpha=5.0 over 10.0.
   The heavier tails already regularize by spreading weight, so less explicit
   Ridge regularization is needed.

5. **Angle benefits most** -- Angle LOO MSE drops 17.23% with per-target optimal
   Cauchy (0.002189 vs 0.002645). This is the target that overfits most (2.57x),
   so using more training data per prediction should help the most on LB.

6. **Moderate diversity** -- r=0.95 with Sub 2063 overall. Not enough for large
   blending gains, but enough that a small blend (10-20%) could help.

## Recommended LB Tests (Priority Order)

1. **Sub 2191**: Per-target Cauchy + Sub784, standard weights. This is the "if kernel
   helps on LB" test.
2. **Sub 2194**: 10% per-target Cauchy + 90% Sub2063. Low-risk blend with current best.
3. **Sub 2195**: 20% blend with Sub2063 for stronger signal if 2194 looks good.

## Interpretation Warning

LOO MSE is systematically optimistic (LOO ~0.0037 vs LB ~0.0066). The Cauchy kernel's
LOO improvement could be real (heavier tails = less variance = less overfitting) or
could be another form of overfitting (the kernel itself is tuned on LOO). The 4-6 extra
effective neighbors should reduce variance, which is the right direction for closing the
LOO-LB gap. But this needs LB validation.
