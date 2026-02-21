# 1D Temporal CNN with Heavy Augmentation - Results

## Date: 2026-02-09

## Summary
Trained a small 1D CNN (53K params) on full 240-frame temporal pose sequences with heavy augmentation. Used MPS GPU on Apple Silicon. Goal was to produce DIVERSE predictions (r < 0.80 with Sub 1350) for ensemble improvement, not standalone accuracy.

## Model Architecture
- Input: (batch, 207 channels [69 keypoints x 3 coords], 240 frames)
- Conv1d(207->32, k=7) -> BN -> ReLU -> AvgPool(4)
- Conv1d(32->32, k=5) -> BN -> ReLU -> AvgPool(4)
- Conv1d(32->16, k=3) -> BN -> ReLU -> AdaptiveAvgPool(1)
- Player embedding: 5 players -> 4-dim embedding
- FC(20) -> 16 -> 1
- Total params: 53,641

## Augmentation
- Temporal shift: roll by 1-5 frames (50% probability)
- Gaussian noise: std 0.01-0.05 on keypoints (70% probability)
- Keypoint dropout: zero out 1-9 random joints (50% probability)
- Temporal scaling: stretch/compress by 0.95-1.05x (30% probability)
- Spatial jitter: small camera-like offset (30% probability)
- Mixup: within-batch, alpha=0.2 (50% probability per batch)

## Training Config
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-2
- Scheduler: CosineAnnealingLR, T_max=200, eta_min=1e-5
- Early stopping: patience=30
- Batch size: 32
- Per-player 5-fold CV
- Test predictions: 5-seed ensemble (70% of max epochs)

## Data Pipeline
- Hoop-relative coordinate transformation (same as per_example_pipeline.py)
- Per-player normalization (zero mean, unit std)
- NaN -> 0 filling

## CV Results

| Target     | CV MSE   | Mean Fold Loss |
|-----------|----------|----------------|
| Angle     | 0.008685 | 0.008479       |
| Depth     | 0.012108 | 0.011831       |
| Left_right| 0.010268 | 0.010287       |
| **Mean**  | **0.010354** | **0.010199** |

For reference:
- Sub 1350 (per-example regression): LOO CV mean ~0.003743
- Sub 784 (tree ensemble): CV mean ~0.0078

The CNN's CV is worse than both baselines in absolute terms.

## Diversity Analysis (KEY RESULT)

### vs Sub 784 (tree ensemble)
| Target     | Pearson r |
|-----------|-----------|
| Angle     | 0.9135    |
| Depth     | 0.6749    |
| Left_right| 0.6917    |

### vs Sub 1350 (per-example regression, best LB)
| Target     | Pearson r |
|-----------|-----------|
| Angle     | 0.9135    |
| Depth     | 0.6935    |
| Left_right| 0.7125    |

**KEY FINDING: Depth and left_right predictions are HIGHLY DIVERSE (r < 0.72).**

This is the most diverse depth model we have ever produced (r=0.69 vs Sub 1350). Previous diverse models:
- Biomech depth: r=0.87 with HR predictions
- FPCA/KNN: r=0.65 but terrible accuracy (13.89+ MSE)
- Physics LR: r=0.46 but terrible accuracy (MSE=0.019)

The CNN achieves moderate accuracy (depth MSE=0.012, LR MSE=0.010) with very high diversity. This makes it a good candidate for ensemble blending if the blend weights are kept small.

## Per-Player Analysis
Notable: Player 5 has high variance across folds (depth fold losses range from 0.012 to 0.049), consistent with known Player 5 difficulties (only 74 samples, overfits easily).

## Submissions Generated
- Sub 1557: CNN standalone
- Sub 1558-1561: 10-50% CNN + Sub 784
- Sub 1562-1564: 10-30% CNN + Sub 1350
- Sub 1565-1566: Depth-only 15-25% CNN + Sub 1350 (r=0.693)
- Sub 1567-1568: Left_right-only 15-25% CNN + Sub 1350 (r=0.712)

## Recommended Submissions to Test
1. **Sub 1565**: Depth-only 15% CNN + Sub 1350 - most promising since depth is the bottleneck and CNN depth is most diverse (r=0.69)
2. **Sub 1562**: 10% CNN across all targets + Sub 1350 - conservative overall blend
3. **Sub 1567**: LR-only 15% CNN + Sub 1350 - exploits LR diversity (r=0.71)

## Total Runtime
1381.9 seconds (23 minutes) on MPS GPU

## Script
`scripts/temporal_cnn.py`

## Conclusion
The 1D CNN with heavy augmentation successfully produces diverse predictions (r < 0.72 on depth and LR), which is the primary goal. While standalone accuracy is worse than our best models, the diversity makes it a valuable ensemble member. The key question is whether small-weight blending (10-25%) with Sub 1350 can improve LB from 0.006776.

Theory: diverse models help when they correct for different error patterns. The CNN's temporal processing (full 240 frames with convolutions) learns different features than the per-example ridge (single-frame hoop-relative features). The depth and LR diversity is particularly promising since these are the current bottleneck targets.
