# Adaptive Bandwidth + Stacked Residual Correction Results

Date: 2026-02-14
Script: `scripts/adaptive_bw_residual.py`

## Summary

Tested two approaches to improve the core per-example locally weighted Ridge pipeline:

| Approach | LOO Mean MSE | vs Baseline (0.003859) | Verdict |
|----------|-------------|----------------------|---------|
| A: Adaptive per-player bandwidth | 0.003664 | -5.06% | WINNER |
| B: Stacked residual correction | 0.003859 | +0.00% | DEAD |

## Approach A: Adaptive Per-Player Bandwidth

### Method
- Current pipeline uses fixed `bandwidth_quantile=0.3` for all players and all targets
- Swept bw from {0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45} per player via LOO
- Each player gets its own optimal bandwidth independently

### Results by target

| Target | Fixed (bw=0.3) | Adaptive | Change |
|--------|----------------|----------|--------|
| angle | 0.002645 | 0.002539 | -4.00% |
| depth | 0.004601 | 0.004522 | -1.72% |
| left_right | 0.004331 | 0.003930 | -9.25% |
| **MEAN** | **0.003859** | **0.003664** | **-5.06%** |

### Per-player optimal bandwidths

#### Angle (frame 153)
| Player | n | Best bw | Fixed MSE | Best MSE | Change |
|--------|---|---------|-----------|----------|--------|
| 1 | 70 | 0.45 | 0.000482 | 0.000454 | -5.67% |
| 2 | 66 | 0.45 | 0.003833 | 0.003769 | -1.67% |
| 3 | 68 | 0.45 | 0.001917 | 0.001852 | -3.34% |
| 4 | 67 | 0.45 | 0.001474 | 0.001452 | -1.50% |
| 5 | 74 | 0.45 | 0.005362 | 0.005030 | -6.17% |

All players prefer wider bandwidth (0.45) for angle prediction.

#### Depth (frame 150)
| Player | n | Best bw | Fixed MSE | Best MSE | Change |
|--------|---|---------|-----------|----------|--------|
| 1 | 70 | 0.45 | 0.003409 | 0.003364 | -1.31% |
| 2 | 66 | 0.45 | 0.004239 | 0.004029 | -4.96% |
| 3 | 68 | 0.30 | 0.000840 | 0.000840 | +0.00% |
| 4 | 67 | 0.45 | 0.005108 | 0.005102 | -0.11% |
| 5 | 74 | 0.45 | 0.009050 | 0.008917 | -1.48% |

Most players prefer wider bandwidth. Player 3 is already optimal at bw=0.30.

#### Left_right (frame 170)
| Player | n | Best bw | Fixed MSE | Best MSE | Change |
|--------|---|---------|-----------|----------|--------|
| 1 | 70 | 0.15 | 0.007655 | 0.007113 | -7.08% |
| 2 | 66 | 0.45 | 0.004139 | 0.003315 | -19.91% |
| 3 | 68 | 0.45 | 0.002103 | 0.001786 | -15.06% |
| 4 | 67 | 0.45 | 0.002876 | 0.002704 | -5.97% |
| 5 | 74 | 0.15 | 0.004722 | 0.004549 | -3.68% |

Mixed results: Players 2/3/4 prefer wider, Players 1/5 prefer tighter.
This is the target with the biggest overall gain (-9.25%).

### Key Insight
- The LOO monotonic trend toward bw=0.45 suggests the search should extend beyond 0.45
- However, wider bandwidth = more regularization = potentially more transferable to test set
- RISK: per-player bandwidth selection on 66-74 samples has high variance - potential overfitting
- The Cauchy kernel experiment (see MEMORY.md) showed that LOO improvements don't always transfer to LB

## Approach B: Stacked Residual Correction

### Method
1. Run core pipeline LOO to get base predictions and residuals
2. Build residual features: 10 joint angles + release frame + per-player mean residual
3. Train Ridge regression on residuals using nested LOO (leave-one-out within each player)
4. Add damped correction: pred_final = pred_base + damp * pred_residual
5. Sweep Ridge alpha {0.5, 1, 5, 10, 50, 100, 500, 1000} and damping {0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}

### Results
- No Ridge alpha produced MSE below baseline (all positive delta)
- Even at alpha=1000, residual model adds noise rather than signal
- Best damping factor for all targets: 0.0 (don't use residual correction at all)
- Residual alpha=500 was closest to break-even: +5.35% (angle), +6.46% (depth), +4.24% (LR)

### Why It Failed
- Joint angles and release frame timing do not contain enough information to predict residuals
- The base model already captures most of the signal available from body mechanics
- With only 66-74 samples per player, the second-stage model overfits to training residuals
- The residuals are essentially noise relative to the available features

## Submissions Generated

| Sub # | Description | LOO Mean MSE |
|-------|-------------|-------------|
| 2352 | Adaptive bandwidth standalone | 0.003664 |
| 2353 | Residual corrected standalone | 0.003859 (no change) |
| 2354 | 10% adaptive + 90% Sub2169 | -- |
| 2355 | 10% residual + 90% Sub2169 | -- |
| 2356 | 20% adaptive + 80% Sub2169 | -- |
| 2357 | 20% residual + 80% Sub2169 | -- |
| 2358 | 30% adaptive + 70% Sub2169 | -- |
| 2359 | 30% residual + 70% Sub2169 | -- |
| 2360 | Best-of-both per target | 0.003664 (same as adaptive) |
| 2361 | 10% best-of-both + 90% Sub2169 | -- |
| 2362 | 20% best-of-both + 80% Sub2169 | -- |
| 2363 | 30% best-of-both + 70% Sub2169 | -- |

## Recommended LB Testing Priority

1. **Sub 2354** (10% adaptive + 90% Sub2169) - conservative blend, least likely to overfit
2. **Sub 2356** (20% adaptive + 80% Sub2169) - moderate blend
3. **Sub 2352** (adaptive standalone) - highest LOO but risk of bandwidth selection overfitting

Residual correction subs are NOT recommended for LB testing.

## Overfitting Warning

The adaptive bandwidth approach picks per-player bandwidth from 7 options on ~70 samples each.
This is a relatively safe hyperparameter search (only 5 players x 7 options = 35 decisions on 345 samples).
However, the fact that most players converge to bw=0.45 (the boundary) suggests:
- The search space should have extended further
- OR the apparent improvement is partly due to fitting to LOO validation noise

A safer variant would be to simply use bw=0.45 for all players (no per-player selection).

## Reproducibility

```
Script: scripts/adaptive_bw_residual.py
Data: data/train.csv (345 shots), data/test.csv (113 shots)
Config Approach A: bw_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45], Ridge alpha=10
Config Approach B: Ridge alpha sweep [0.5, 1, 5, 10, 50, 100, 500, 1000], damping [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
Features: 198 hoop-relative + 15 PLS = 213 total (same as core pipeline)
Runtime: 134.8s
```
