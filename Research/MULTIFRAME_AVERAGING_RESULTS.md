# Multi-Frame Prediction Averaging Results

## Approach

Instead of extracting features at a single optimal frame per target (angle=153, depth=150, LR=170), extract features at multiple frames around the optimal, run per-example locally weighted Ridge for each frame independently, and average the predictions.

### Key Design Decisions
- PLS components fitted ONCE per target (frame-independent) - avoids overfitting PLS per frame
- HC features extracted per frame (198 features change with frame, 15 PLS stay fixed)
- Locally weighted Ridge: bandwidth_quantile=0.5, alpha=10.0 (same as Sub 1350)
- Per-player standardization and distance computation (same as Sub 1350)

## Configurations Tested

Grid of 9 configs: {3, 5, 7} frames x {1, 2, 3} spacing

| Config | Uniform MSE | MSE-weighted | Best Mix | vs Baseline |
|--------|-------------|--------------|----------|-------------|
| n3_s1 | 0.003703 | 0.003702 | 0.003703 | -1.1% |
| n3_s2 | 0.003672 | 0.003671 | 0.003672 | -1.9% |
| n3_s3 | 0.003634 | 0.003633 | 0.003634 | -2.9% |
| n5_s1 | 0.003677 | 0.003676 | 0.003677 | -1.8% |
| n5_s2 | 0.003611 | 0.003609 | 0.003611 | -3.6% |
| n5_s3 | 0.003560 | 0.003558 | 0.003560 | -5.0% |
| n7_s1 | 0.003649 | 0.003648 | 0.003649 | -2.5% |
| n7_s2 | 0.003564 | 0.003562 | 0.003564 | -4.8% |
| n7_s3 | 0.003546 | **0.003544** | 0.003546 | **-5.3%** |

Baseline single-frame: 0.003743 (reproduces Sub 1350 LOO)

## Best Config: n7_s3 MSE-weighted

7 frames, spacing 3, inverse-MSE weighted averaging

### Per-Target Results
| Target | Baseline | Multi-frame | Delta |
|--------|----------|-------------|-------|
| angle | 0.002511 | 0.002365 | -5.8% |
| depth | 0.004510 | 0.004242 | -5.9% |
| left_right | 0.004209 | 0.004026 | -4.3% |
| **mean** | **0.003743** | **0.003544** | **-5.3%** |

### Frames Used
- Angle: [144, 147, 150, 153, 156, 159, 162]
- Depth: [141, 144, 147, 150, 153, 156, 159]
- Left_right: [161, 164, 167, 170, 173, 176, 179]

## Key Findings

1. **Wider spacing beats narrow spacing**: For any window size, spacing=3 beats spacing=2 beats spacing=1. This suggests that adjacent frames are highly correlated and don't add much diversity. Spreading frames further apart captures more independent information.

2. **More frames help with diminishing returns**: 7 > 5 > 3 frames at any spacing. The improvement from 5 to 7 frames is smaller than 3 to 5, suggesting an asymptote.

3. **MSE-weighted averaging barely beats uniform**: The difference is tiny (0.003544 vs 0.003546). The per-frame MSE weights are nearly uniform, since individual frames have similar LOO performance.

4. **Center vs average mixing always picks cw=0.0**: The full average always beats any mix that upweights the center frame. This confirms that averaging reduces variance without adding bias.

5. **Consistent improvement across all targets**: All three targets benefit from multi-frame averaging (angle -5.8%, depth -5.9%, LR -4.3%).

## Diversity Analysis

Correlation of multi-frame test predictions with Sub 784:
- angle: r=0.935 (similar to single-frame r~0.93)
- depth: r=0.931 (similar)
- left_right: r=0.847 (similar)

Multi-frame averaging does NOT increase diversity - it smooths predictions while maintaining the same correlation structure. The improvement is purely from variance reduction.

## Submissions Generated

| Sub | Config | Type | LOO MSE |
|-----|--------|------|---------|
| 1545 | n7_s3 wmse | standalone | 0.003544 |
| 1546 | n7_s3 wmse | blended dw=0.30 lw=0.50 | - |
| 1547 | n7_s3 best_mix | standalone | 0.003546 |
| 1548 | n7_s3 best_mix | blended dw=0.30 lw=0.50 | - |
| 1549 | n7_s3 uniform | standalone | 0.003546 |
| 1550 | n7_s3 uniform | blended dw=0.30 lw=0.50 | - |
| 1551 | n5_s3 wmse | standalone | 0.003558 |
| 1552 | n5_s3 wmse | blended dw=0.30 lw=0.50 | - |
| 1553 | n5_s3 best_mix | standalone | 0.003560 |
| 1554 | n5_s3 best_mix | blended dw=0.30 lw=0.50 | - |
| 1555 | baseline single-frame | blended dw=0.30 lw=0.50 | 0.003743 |
| 1556 | per-target best | blended dw=0.30 lw=0.50 | - |

**Recommended for LB testing**: Sub 1546 (n7_s3 wmse blended, best overall LOO)

## Caveats

- LOO CV for per-example models is systematically optimistic (as seen with Sub 1350: LOO=0.003743 vs LB=0.006776)
- The 5.3% LOO improvement may not fully translate to LB improvement
- Previous experience shows that larger CV-LB gaps correlate with more model complexity
- Multi-frame averaging adds model complexity (7x more predictions to average) which could widen the CV-LB gap

## Reproduction

Script: `scripts/multiframe_averaging.py`
Run: `uv run python scripts/multiframe_averaging.py`
Runtime: ~260 seconds
Dependencies: numpy, pandas, scipy, sklearn, joblib, lightgbm (unused but imported)
