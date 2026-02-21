# Mixup Augmentation Results

Date: 2026-02-09

## Approach

Generate synthetic training samples via mixup interpolation between same-player shots:
- Pick two shots from same player
- Blend ratio lambda ~ Beta(alpha, alpha)
- Synthetic 3D keypoints = lambda * shot_A + (1-lambda) * shot_B (frame-by-frame)
- Synthetic targets = lambda * target_A + (1-lambda) * target_B

Then retrain per-example locally weighted Ridge on augmented data (original + synthetic).

Key safeguard: During LOO CV, leave out the original shot AND any synthetic shot derived from it to prevent data leakage.

## Script

`scripts/mixup_augmentation.py`

## Baseline (no augmentation)

Bandwidth=0.5, alpha=10.0 Ridge, same as Sub 1350 approach.

| Target     | LOO MSE   |
|------------|-----------|
| angle      | 0.002511  |
| depth      | 0.004510  |
| left_right | 0.004209  |
| **mean**   | **0.003743** |

## Config Search Results

7 configs tested. Ranked by mean LOO MSE:

| Rank | Config                          | Mean MSE  | Delta   | Angle MSE | Depth MSE | LR MSE   |
|------|----------------------------------|-----------|---------|-----------|-----------|-----------|
| 1    | alpha=1.0, 2x, full weight       | 0.003507  | -9.00%  | 0.001878  | 0.004522  | 0.004121  |
| 2    | alpha=0.5, 2x, half weight       | 0.003519  | -6.78%  | 0.002217  | 0.004298  | 0.004043  |
| 3    | alpha=0.5, 2x, full weight       | 0.003536  | -7.23%  | 0.002068  | 0.004407  | 0.004134  |
| 4    | alpha=0.2, 2x, full weight       | 0.003536  | -7.66%  | 0.002009  | 0.004621  | 0.003979  |
| 5    | alpha=0.2, 3x, half weight       | 0.003564  | -6.73%  | 0.002042  | 0.004427  | 0.004223  |
| 6    | alpha=0.5, 1x, full weight       | 0.003567  | -5.94%  | 0.002175  | 0.004437  | 0.004090  |
| 7    | alpha=0.5, 3x, full weight       | 0.003593  | -6.59%  | 0.001952  | 0.004563  | 0.004264  |

## Key Observations

1. **Angle benefits most from mixup** (-11% to -25% improvement). Angle is the best-predicted target (r=0.85) and mixup provides more interpolated samples in the well-predicted space.

2. **Depth is mostly flat** (-5% to +2%). Depth is the bottleneck and mixup doesn't help much. The alpha=0.5, 2x, half weight config is best for depth specifically (-4.71%).

3. **Left_right gets small improvement** (-1.8% to -5.5%). The alpha=0.2, 2x config is best for LR specifically (-5.46%).

4. **No single config dominates all targets**. Best overall (alpha=1.0, 2x) hurts depth (+0.27%) but crushes angle (-25.19%).

5. **Half-weight discount helps depth**. Giving synthetic samples less weight in the kernel prevents them from dominating depth predictions where they add noise.

6. **Correlation with baseline is very high** (r=0.97-0.99). Mixup adjusts predictions slightly rather than producing qualitatively different answers.

7. **Correlation with Sub 784**: angle r=0.92, depth r=0.92, LR r=0.78-0.81. Similar diversity profile as original per-example approach.

## Submissions Generated

- Sub 1583: Baseline (no augmentation) standalone
- Sub 1584: Baseline blended with Sub 784 (dw=0.30, lw=0.50)
- Sub 1585: Best config (alpha=1.0, 2x) standalone
- Sub 1586: Best config blended with Sub 784 (dw=0.30, lw=0.50)
- Sub 1587: Best config conservative blend (dw=0.20, lw=0.35)
- Sub 1588: Config 2 (alpha=0.5, 2x, half weight) standalone
- Sub 1589: Config 2 blended with Sub 784 (dw=0.30, lw=0.50)
- Sub 1590: Config 2 conservative blend (dw=0.20, lw=0.35)
- Sub 1591: Config 3 (alpha=0.5, 2x, full weight) standalone
- Sub 1592: Config 3 blended with Sub 784 (dw=0.30, lw=0.50)
- Sub 1593: Config 3 conservative blend (dw=0.20, lw=0.35)
- Sub 1594: 10% best mixup + 90% Sub 1350
- Sub 1595: 20% best mixup + 80% Sub 1350
- Sub 1596: 30% best mixup + 70% Sub 1350

## Expectations

The LOO CV improvements (-5.9% to -9.0%) are somewhat optimistic given the 33-81% CV-LB gap seen historically. However:
- Mixup is a regularization technique, not adding new signal, so the gap may be smaller
- The very high correlation with baseline (r>0.97) suggests small perturbations, not radical changes
- Sub 1589 (half-weight config, balanced across targets) may be best for LB

Most promising submissions for LB testing:
1. **Sub 1589** - alpha=0.5, 2x, half weight, blended dw=0.30 lw=0.50 (best depth improvement)
2. **Sub 1586** - alpha=1.0, 2x, full weight, blended dw=0.30 lw=0.50 (best overall LOO)
3. **Sub 1594** - 10% best mixup + 90% Sub 1350 (conservative blend with LB champion)

## Warning

LOO CV for per-example models has been systematically optimistic (33-81% gap). The -9% LOO improvement does NOT guarantee LB improvement. All previous attempts to improve on Sub 1350 via LOO-improving methods have failed on LB:
- Per-example V2: LOO better, LB 0.006789 vs 0.006776
- MTL-Ridge: LOO 0.005093, LB 0.006803
- Biomech features: LOO +5.46%, LB 0.007794
- Temporal dynamics: LOO validated, LB 0.007528
