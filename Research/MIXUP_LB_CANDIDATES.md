# Mixup Augmentation - LB Candidate Submissions

Date: 2026-02-14

## Configuration

- Mixup: alpha=1.0 (Beta distribution), 2x augmentation, full weight (no discount), seed=44
- Pipeline: bandwidth_quantile=0.3, Ridge alpha=10.0
- Features: 198 hoop-relative + 15 PLS per target (same as core pipeline)
- Target frames: angle=153, depth=150, left_right=170
- 690 synthetic samples generated (2x of 345 original)

## Baseline (no augmentation, bw_q=0.3)

| Target     | LOO MSE   |
|------------|-----------|
| angle      | 0.002645  |
| depth      | 0.004601  |
| left_right | 0.004331  |
| **mean**   | **0.003859** |

## Mixup Results (alpha=1.0, 2x, full weight)

| Target     | LOO MSE   | Delta    |
|------------|-----------|----------|
| angle      | 0.001980  | -25.16%  |
| depth      | 0.004566  | -0.76%   |
| left_right | 0.004240  | -2.10%   |
| **mean**   | **0.003595** | **-9.34%** |

## Key Observations

1. **Angle dominates the improvement** (-25.16%). Depth (-0.76%) and LR (-2.10%) barely change.
2. This matches the Feb 9 result pattern exactly - mixup is primarily an angle regularizer.
3. Mean correlation with Sub 2169: r=0.9773 (very high, small perturbation).
4. The LOO improvement is -9.34%, similar to the -9.00% from the Feb 9 run with bw_q=0.5.

## Warning

Historically, LOO improvements for per-example models have NOT translated to LB gains:
- Cauchy kernel: -4.42% LOO, LB standalone WORSE (0.006681 vs 0.006603)
- Cross-fitting: -2-10% theoretical, LOO flat
- Biomech features: +5.46% LOO, LB 0.007794 (much worse)

The -9.34% LOO gain is heavily driven by angle (-25.16%), but angle has 2.57x overfit ratio.
Angle is the HIDDEN BOTTLENECK: LOO 0.002645 vs test ~0.006454.

Mixup's huge angle improvement may NOT translate because:
- The angle overfit gap is 2.57x, so LOO improvements are misleading
- Mixup interpolation may create unrealistically smooth training space
- The high r=0.977 correlation means small absolute changes

## Submissions Generated

| Sub  | Description                                    |
|------|------------------------------------------------|
| 2364 | Mixup standalone (alpha=1.0, 2x, bw_q=0.3)   |
| 2365 | 10% mixup + 90% Sub 2169                      |
| 2366 | 20% mixup + 80% Sub 2169                      |
| 2367 | 30% mixup + 70% Sub 2169                      |
| 2368 | 10% baseline (no mixup) + 90% Sub 2169        |
| 2369 | 20% baseline (no mixup) + 80% Sub 2169        |
| 2370 | 30% baseline (no mixup) + 70% Sub 2169        |

## Recommended LB Testing Priority

1. **Sub 2365** (10% mixup + 90% Sub 2169) - conservative blend, least risk
2. **Sub 2366** (20% mixup + 80% Sub 2169) - moderate blend
3. **Sub 2364** (mixup standalone) - to understand mixup's standalone LB performance

The baseline blends (2368-2370) serve as controls: if mixup blends beat these, the signal is real.
