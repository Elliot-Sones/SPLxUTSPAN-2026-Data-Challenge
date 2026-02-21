# Angle Overfit Diagnosis - 2026-02-15

## Problem
Angle target overfits 2.57x (LOO 0.002511 vs test 0.006454) while depth (1.51x) and LR (1.62x) are moderate. If angle overfit dropped to 1.56x, theoretical LB = 0.005856.

## Root Cause: Low Signal-to-Noise Ratio

| Target | Between-player var | Within-player var | Within-player std | Best feature |r| |
|--------|-------------------|-------------------|-------------------|-------------|
| angle | **72.0%** | **28.0%** | **0.0858** | **0.23** |
| depth | 6.5% | 93.5% | 0.1242 | 0.45 |
| left_right | 1.0% | 99.0% | 0.1180 | 0.41 |

**Angle is fundamentally different from depth/LR:**
- 72% of angle variance is just "which player" - the model mostly needs player identity
- Within-player std is tiny (P1: 0.043, P2: 0.068, P3: 0.054, P4: 0.090, P5: 0.136)
- Best individual feature within-player correlation is only r=0.23 (depth: 0.45, LR: 0.41)
- With 208 features, 15 PLS components, ~65 samples per player: easy to memorize noise

## Key Finding: Player Mean Beats Full Model

For angle, the player mean alone (LOO MSE=0.007360) is BETTER than the full Ridge model (0.010824). The model's within-player predictions are net noise.

## Shrinkage Results

`pred_final = shrink * model_pred + (1 - shrink) * player_mean`

| shrink | angle LOO MSE | % change |
|--------|--------------|----------|
| 0.0 (player mean) | 0.007360 | -32.0% |
| 0.1 | 0.006452 | -40.4% |
| 0.2 | 0.005822 | -46.2% |
| 0.3 | 0.005471 | -49.5% |
| **0.4** | **0.005398** | **-50.1%** |
| 0.5 | 0.005605 | -48.2% |
| 0.6 | 0.006091 | -43.7% |
| 0.7 | 0.006856 | -36.7% |
| 0.8 | 0.007900 | -27.0% |
| 0.9 | 0.009222 | -14.8% |
| 1.0 (no shrink) | 0.010824 | baseline |

Optimal shrink=0.4 gives 50% LOO improvement. This is on a standalone baseline model.
The combined pipeline (Sub 2503) already uses many tricks, but the overfit mechanism is the same.

## Depth and LR Also Benefit

| Target | Baseline LOO | shrink=0.7 | % change |
|--------|-------------|------------|----------|
| depth | 0.009101 | 0.006604 | -27.4% |
| left_right | 0.008685 | 0.006670 | -23.2% |

## Error Distribution

Top 10% of angle examples account for 60.5% of total MSE.
Worst examples are high-leverage outliers (leverage 33-47 vs mean 12).

## Submissions Generated

### Angle-only shrinkage on Sub 2503:
- Sub 2532: shrink=0.3
- Sub 2533: shrink=0.4
- Sub 2534: shrink=0.5
- Sub 2535: shrink=0.6
- Sub 2536: shrink=0.7
- Sub 2537: shrink=0.8
- Sub 2538: shrink=0.9

### All-target shrinkage on Sub 2503:
- Sub 2539: angle=0.5, depth=0.85, LR=0.80
- Sub 2540: angle=0.4, depth=0.80, LR=0.75
- Sub 2541: angle=0.3, depth=0.75, LR=0.70
- Sub 2542: angle=0.5, depth=0.90, LR=0.90
- Sub 2543: angle=0.6, depth=0.90, LR=0.90
- Sub 2544: angle=0.7, depth=0.90, LR=0.90
- Sub 2545: angle=0.8, depth=0.95, LR=0.95

### Per-player angle shrinkage:
- Sub 2546: base=0.3 (P1=0.30, P5=1.00)
- Sub 2547: base=0.5 (P1=0.50, P5=1.00)
- Sub 2548: base=0.7 (P1=0.70, P5=1.00)

## Priority for LB Testing
1. **Sub 2537** (angle shrink=0.8) - safest, smallest change
2. **Sub 2536** (angle shrink=0.7) - moderate change
3. **Sub 2534** (angle shrink=0.5) - best LOO for standalone model
4. **Sub 2542** (angle=0.5, depth=0.90, LR=0.90) - if angle-only helps
5. **Sub 2547** (per-player base=0.5) - most principled

## Why This Should Work on LB
Unlike previous LOO improvements that didn't transfer:
- Shrinkage is a REGULARIZATION technique, not a new feature
- It corrects the model's systematic overconfidence in within-player deviations
- Player means are stable (computed from all ~65-74 training shots per player)
- The same overfit mechanism applies to test predictions

## Scripts
- scripts/angle_overfit_diagnosis.py - diagnostic analysis
- scripts/angle_shrinkage.py - shrinkage experiments with LOO
- scripts/posthoc_angle_shrinkage.py - generate post-hoc shrinkage submissions
