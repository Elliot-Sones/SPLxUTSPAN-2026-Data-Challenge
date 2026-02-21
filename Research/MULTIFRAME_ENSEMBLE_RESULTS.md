# Multi-Frame Ensemble for Variance Reduction

**Date:** 2026-02-12
**Script:** scripts/multiframe_ensemble.py
**Runtime:** 312.5s

## Concept

Instead of extracting features at ONE optimal frame per target, train separate per-example locally weighted Ridge models at MULTIPLE frames and average predictions. Each frame sees slightly different body configurations, so averaging across frames reduces prediction variance without adding features (which would overfit on 345 samples).

## Phase 1: LOO MSE vs Frame Curve (frames 130-180, step 5)

### Angle
| Frame | LOO MSE | Rank |
|-------|---------|------|
| 145   | 0.002528 | 1 |
| 150   | 0.002576 | 2 |
| 140   | 0.002659 | 3 |
| 155   | 0.002662 | 4 |
| 135   | 0.002691 | 5 |
| 165   | 0.002737 | 6 |
| 160   | 0.002772 | 7 |
| 130   | 0.002812 | 8 |
| 170   | 0.002823 | 9 |
| 175   | 0.003021 | 10 |
| 180   | 0.003022 | 11 |

**Note:** Best frame is 145, not 153 (current default). The curve is U-shaped centered near 145.

### Depth
| Frame | LOO MSE | Rank |
|-------|---------|------|
| 145   | 0.004519 | 1 |
| 155   | 0.004564 | 2 |
| 150   | 0.004601 | 3 (current default) |
| 140   | 0.004685 | 4 |
| 135   | 0.004780 | 5 |
| 160   | 0.005000 | 6 |
| 130   | 0.005337 | 7 |
| 165   | 0.005609 | 8 |
| 180   | 0.005656 | 9 |
| 175   | 0.005838 | 10 |
| 170   | 0.006105 | 11 |

**Note:** Best frame is 145, not 150 (current default). Curve centered around 145-150.

### Left/Right
| Frame | LOO MSE | Rank |
|-------|---------|------|
| 165   | 0.004166 | 1 |
| 160   | 0.004268 | 2 |
| 175   | 0.004315 | 3 |
| 170   | 0.004331 | 4 (current default) |
| 180   | 0.004417 | 5 |
| 155   | 0.004553 | 6 |
| 130   | 0.004554 | 7 |
| 150   | 0.004602 | 8 |
| 145   | 0.004612 | 9 |
| 135   | 0.004684 | 10 |
| 140   | 0.004751 | 11 |

**Note:** Best frame is 165, not 170 (current default). Also note anomalous good score at frame 130.

### Key Finding: Current TARGET_FRAMES are NOT optimal
- Angle: 153 -> 145 (difference: 0.002528 vs ~0.002576, improvement likely)
- Depth: 150 -> 145 (difference: 0.004519 vs 0.004601)
- LR: 170 -> 165 (difference: 0.004166 vs 0.004331)

All current defaults are close but not quite optimal. However, LOO is optimistic, so slight differences could be noise.

## Phase 2: Pairwise Prediction Correlations

### Angle (top 7 frames)
Correlations range from 0.985 to 0.996. All very high - angle features are very stable across frames.

### Depth (top 7 frames)
Correlations range from 0.924 to 0.983. More diversity than angle - ensembling should help more for depth.

### Left/Right (top 7 frames)
Correlations range from 0.911 to 0.985. Most diversity - especially frame 130 (r=0.917-0.939 with other frames).

## Phase 3: Greedy Frame Selection

### Angle
Frames selected: [145, 165, 150, 135, 155, 140, 170]
- 2-frame: -1.42%
- 3-frame: -2.22% (cumulative)
- 4-frame: -2.65% (diminishing returns)
- After 4 frames, MSE increases (overfitting to LOO)

### Depth
Frames selected: [145, 155, 135, 180, 140, 150, 160]
- 2-frame: -3.93%
- 3-frame: -4.89%
- 5-frame: -5.76%
- After 6 frames, MSE increases

### Left/Right
Frames selected: [165, 130, 180, 160, 175, 135, 170]
- 2-frame: -3.27%
- 3-frame: -4.84%
- 4-frame: -5.08%
- After 4 frames, MSE increases

## Phase 4: Ensemble Sizes Comparison

### Equal-weight vs Inverse-MSE-weight
| Target | 1-frame | 2-frame eq | 2-frame w | 3-frame eq | 3-frame w | 5-frame eq | 5-frame w |
|--------|---------|-----------|-----------|-----------|-----------|-----------|-----------|
| angle  | 0.002528 | 0.002492 | 0.002488 | 0.002472 | 0.002471 | 0.002470 | 0.002470 |
| depth  | 0.004519 | 0.004341 | 0.004341 | 0.004298 | 0.004296 | 0.004261 | 0.004258 |
| LR     | 0.004166 | 0.004029 | 0.004021 | 0.003964 | 0.003962 | 0.003960 | 0.003960 |
| MEAN   | 0.003738 | 0.003621 | 0.003617 | 0.003578 | 0.003576 | 0.003564 | 0.003563 |

**Key finding:** Inv-MSE weighting provides negligible benefit (0.03% at best) because all frames are close in accuracy. Equal weighting is just as good and simpler.

**Best improvement:** 5-frame ensemble achieves 4.68% LOO MSE reduction over single-frame.

## Phase 5: Joint Angles Interaction

3-frame ensemble with 10 extra joint angle features:
- Angle: +1.70% WORSE (0.002514 vs 0.002472)
- Depth: -0.11% (basically flat, 0.004293 vs 0.004298)
- LR: +0.37% WORSE (0.003978 vs 0.003964)

**Conclusion:** Joint angles HURT when combined with multi-frame ensemble. The added features increase overfitting more than the extra signal helps.

## Phase 6: Per-Player Breakdown (3-frame ensemble vs single-frame)

### Angle
| Player | Single | Ensemble | Delta |
|--------|--------|----------|-------|
| 1      | 0.000418 | 0.000417 | -0.12% |
| 2      | 0.003777 | 0.003834 | +1.52% |
| 3      | 0.001822 | 0.001697 | -6.88% |
| 4      | 0.001516 | 0.001368 | -9.80% |
| 5      | 0.004975 | 0.004913 | -1.24% |

### Depth
| Player | Single | Ensemble | Delta |
|--------|--------|----------|-------|
| 1      | 0.003701 | 0.003007 | -18.75% |
| 2      | 0.003958 | 0.003768 | -4.80% |
| 3      | 0.000844 | 0.000767 | -9.10% |
| 4      | 0.005100 | 0.004837 | -5.15% |
| 5      | 0.008645 | 0.008747 | +1.18% |

### Left/Right
| Player | Single | Ensemble | Delta |
|--------|--------|----------|-------|
| 1      | 0.008139 | 0.007867 | -3.34% |
| 2      | 0.003925 | 0.003371 | -14.12% |
| 3      | 0.001842 | 0.001699 | -7.80% |
| 4      | 0.003027 | 0.002924 | -3.42% |
| 5      | 0.003787 | 0.003824 | +0.97% |

**Key finding:** Improvement is broad-based across 4 of 5 players. Player 5 slightly worse (variance too high, averaging helps less). Biggest depth improvement is Player 1 (-18.75%).

## Phase 7: Submissions

### Diversity with existing submissions
| Target | r with Sub 784 | r with Sub 2063 |
|--------|----------------|-----------------|
| angle  | 0.9384 | 0.9808 |
| depth  | 0.9142 | 0.9498 |
| left_right | 0.8665 | 0.9726 |

Diversity is moderate: r=0.95-0.98 with Sub 2063 (high but not 0.99+). Depth and LR have more diversity than angle.

### Submissions generated
| Sub | Description | Notes |
|-----|-------------|-------|
| 2163 | Standalone multiframe (5f_inv_mse for angle+depth, 5f_equal for LR) | Pure multiframe, no blending |
| 2164 | Multiframe + Sub 784 (aw=0, dw=0.30, lw=0.50) | Standard proven blend weights |
| 2165 | Multiframe + Sub 784 (aw=0.50, dw=0.30, lw=0.50) | With angle fix weight |
| 2166 | 30% multiframe + 70% Sub 2063 | Small multiframe weight |
| 2167 | 50% multiframe + 50% Sub 2063 | Equal blend |
| 2168 | 3-frame equal + Sub 784 (aw=0, dw=0.30, lw=0.50) | Using 3-frame instead of 5-frame |
| 2169 | 30% 3-frame equal + 70% Sub 2063 | Small weight of 3-frame |
| 2170 | 50% 3-frame equal + 50% Sub 2063 | Equal weight |
| 2171 | 30% 3-frame+JA + 70% Sub 2063 | Joint angles variant |

### Recommended submissions to test on LB (priority order)
1. **Sub 2166** (30% multiframe + 70% Sub 2063) - conservative blend with current best
2. **Sub 2169** (30% 3-frame + 70% Sub 2063) - similar but fewer frames
3. **Sub 2165** (multiframe + Sub 784 with angle fix) - includes angle correction

## Conclusions

1. **Multi-frame ensembling provides genuine LOO improvement**: 4.68% mean MSE reduction across targets (single 0.003738 -> 5-frame 0.003563)
2. **Depth benefits most** (5.76% reduction) because frame-to-frame prediction diversity is highest for depth
3. **Angle benefits least** (2.29% reduction) because predictions are highly correlated (r>0.98) across frames
4. **LR also benefits well** (4.95% reduction) especially from adding diverse frame 130
5. **Equal weighting is sufficient** - inverse-MSE weighting adds <0.03%
6. **Joint angles HURT** when combined with multi-frame (+1.70% angle, +0.37% LR)
7. **Current TARGET_FRAMES are slightly suboptimal** - best single frames are 145/145/165 vs current 153/150/170
8. **Optimal ensemble size is 3-5 frames** - beyond that, diminishing returns and potential overfitting
9. **Improvement is broad-based** across players, not driven by one player
10. **Diversity with Sub 2063 is moderate** (r=0.95-0.98) - may provide small blending benefit

### Caveats
- LOO is systematically optimistic for per-example models (known issue)
- The 4.68% LOO improvement may translate to a smaller LB improvement
- If the LOO improvement is real, blending with Sub 2063 at 30% weight should yield a small but measurable improvement
