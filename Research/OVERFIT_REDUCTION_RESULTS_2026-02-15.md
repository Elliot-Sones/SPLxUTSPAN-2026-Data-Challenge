# Overfit Reduction Results - 2026-02-15

## Experiment
Tested 3 proven overfit reduction approaches on the combined pipeline (same features/settings as Sub 2503):

1. **Nadaraya-Watson (NW)**: Weighted mean of neighbor targets, no regression coefficients
2. **Bagging**: 50 bootstrap Ridge models averaged per prediction
3. **PLS Leak Fix**: Refit PLS excluding held-out point in each LOO iteration

Script: scripts/overfit_reduction.py

## Results

| Config | angle LOO | depth LOO | LR LOO | mean LOO | vs baseline |
|--------|-----------|-----------|--------|----------|-------------|
| baseline | 0.001605 | 0.002290 | 0.002755 | 0.002217 | -- |
| NW angle only | 0.007483 | 0.002290 | 0.002755 | 0.004176 | +88.4% |
| NW all targets | 0.007483 | 0.014394 | 0.013364 | 0.011747 | +430.0% |
| bagged Ridge angle | 0.002371 | 0.002290 | 0.002755 | 0.002472 | +11.5% |
| bagged Ridge all | 0.002371 | 0.002830 | 0.003367 | 0.002856 | +28.9% |
| bagged NW angle | 0.007488 | 0.002290 | 0.002755 | 0.004178 | +88.5% |
| **PLS fix only** | **0.007721** | **0.006043** | **0.006727** | **0.006830** | **+208.2%** |
| NW angle + PLS fix | 0.007497 | 0.006043 | 0.006727 | 0.006756 | +204.8% |
| bagged NW + PLS fix | 0.007501 | 0.006043 | 0.006727 | 0.006757 | +204.9% |

## Critical Finding: PLS Data Leakage Explains "Overfit"

The PLS fix reveals that the LOO was massively biased by data leakage:

| Target | Leaky LOO | Honest LOO | Inflation factor | LB (Sub 2503) |
|--------|-----------|------------|------------------|----------------|
| angle | 0.001605 | 0.007721 | 4.81x | ~0.006454 |
| depth | 0.002290 | 0.006043 | 2.64x | ~0.005806 |
| LR | 0.002755 | 0.006727 | 2.44x | ~0.007152 |
| **mean** | **0.002217** | **0.006830** | **3.08x** | **0.006471** |

**The honest LOO (0.006830) is only 5.6% above the LB (0.006471).**

This means:
1. The "2.57x angle overfit ratio" was mostly PLS leakage, NOT model overfitting
2. The model is generalizing well - honest LOO closely tracks LB
3. Post-hoc shrinkage toward player means was addressing a phantom problem
4. The real bottleneck is model expressiveness, not overfitting

## Why PLS Leaks in LOO

In standard LOO, when predicting example i:
- PLS is fit on ALL player data (including i's features and target)
- Then we remove i from the weighted regression
- But PLS components already "saw" i's target through the projection

This makes LOO optimistic because the feature space was shaped by the held-out target.
The effect is largest for angle (4.81x) because angle has low within-player signal -
the leaked information from PLS dominates the genuine signal.

## Implications

1. **Shrinkage submissions (2532-2548) are unlikely to help on LB** - they correct for overfit that doesn't exist on test
2. **NW is much worse than Ridge** - regression coefficients add real value, not overfit
3. **Bagging hurts** - bootstrap introduces noise in small-sample per-player setting
4. **Sub 2503 is near-optimal for this feature set** - gap to LB is only 5.6%
5. **To improve further**: need genuinely better features/model, not regularization tricks

## Submissions Generated
- Subs 2557-2572 (8 standalone + 8 blends with Sub 2503)
- All are expected to perform WORSE than Sub 2503 on LB
- NONE recommended for the user's final submission

## Recommendation
Do NOT use the final submission on any of these. Sub 2503 (LB 0.006471) or Sub 2169 (LB 0.006552) remain the best options. The honest LOO analysis shows the model is NOT overfitting - it's near its capability ceiling for this feature set.
