# Regularization Sweep Results

Date: 2026-02-14
Script: `scripts/regularization_sweep.py`

## Summary

Tested 20 configurations to see if stronger regularization could reduce the CV-LB gap (angle 2.57x, depth 1.51x, LR 1.62x). All forms of stronger regularization make things WORSE.

## Key Finding: The pipeline is NOT under-regularized

| Approach | LOO Mean | vs Baseline | Verdict |
|----------|----------|-------------|---------|
| bw=0.45 alpha=10 (current) | 0.003352 | -13.14% | BEST overall |
| bw=0.30 alpha=10 | 0.003355 | -13.06% | Nearly identical |
| bw=0.55 alpha=10 | 0.003386 | -12.27% | Slightly worse |
| bw=0.65 alpha=10 | 0.003429 | -11.15% | Worse |
| bw=0.80 alpha=10 | 0.003518 | -8.83% | Much worse |
| alpha=50 | 0.004452 | +15.38% | Catastrophic |
| alpha=100 | 0.005237 | +35.71% | Catastrophic |
| Shrinkage (calibrated) | 0.005051 | +30.88% | Catastrophic |
| Shrinkage (light) | 0.003848 | -0.29% | Flat/worse |
| k-NN k=10 | 0.008476 | +119.6% | Catastrophic |

## Per-Target Bandwidth Tradeoff

Wider bandwidth helps angle but hurts left_right:

| Bandwidth | Angle LOO | Depth LOO | LR LOO | Mean LOO |
|-----------|-----------|-----------|--------|----------|
| 0.30 | 0.002534 | 0.004093 | 0.003438 | 0.003355 |
| 0.45 | 0.002434 | 0.004057 | 0.003564 | 0.003352 |
| 0.55 | 0.002381 | 0.004052 | 0.003724 | 0.003386 |
| 0.65 | 0.002332 | 0.004060 | 0.003894 | 0.003429 |
| 0.80 | 0.002270 | 0.004106 | 0.004178 | 0.003518 |

Angle improves monotonically with wider bandwidth (-14.2% at bw=0.80).
LR degrades monotonically with wider bandwidth (-3.5% at bw=0.80 vs -20.6% at bw=0.30).
Depth is stable across all bandwidths (~0.004050-0.004106).

## Diversity Analysis

All configurations produce nearly identical predictions to Sub 2402:
- bw=0.45: r=0.980
- bw=0.55: r=0.976
- bw=0.80: r=0.952

No blending opportunity exists.

## Conclusions

1. bw=0.45, alpha=10 is already near-optimal
2. Higher Ridge alpha destroys signal (even alpha=50 is +15% worse)
3. Post-hoc shrinkage toward player mean hurts - model predictions are already well-calibrated
4. k-NN is catastrophically bad - confirms the model needs to fit slopes, not just local means
5. The CV-LB gap is NOT addressable via regularization - it's a fundamental sample size limitation
6. To improve beyond 0.006511, we need a structurally different approach, not parameter tuning
