# Physics-Constrained Per-Example Regression Results

## Date: 2026-02-09

## Approach

Instead of predicting targets directly (like Sub 1350), predict the ball's release velocity (vx, vy, vz) using per-example locally weighted Ridge regression, then forward simulate through ballistic equations to get targets.

Key differences from previous physics attempt (inverse_projectile_pipeline.py):
- 213 features (HC + PLS) instead of 33 hand geometry features
- Per-example locally weighted Ridge instead of per-player Ridge+LGB+PLS
- Ensemble physics predictions with direct predictions

## Velocity Prediction (Per-Example Weighted Ridge)

Huge improvement over previous attempt:

| Component | Previous R | New R | Previous RMSE | New RMSE |
|-----------|-----------|-------|--------------|----------|
| vx (lateral) | 0.8275 | 0.8949 | 0.1639 | 0.1558 |
| vy (toward hoop) | 0.5642 | 0.8926 | 0.1363 | 0.0722 |
| vz (vertical) | 0.9420 | 0.9478 | 0.1508 | 0.1780 |
| speed | 0.9393 | 0.9564 | - | - |

vy (lateral velocity, drives depth) improved from R=0.56 to R=0.89 - a massive improvement.

Best configs:
- vx: bw=0.3, alpha=20
- vy: bw=0.6, alpha=5
- vz: bw=0.6, alpha=5

## Target Prediction (Scaled LOO MSE)

| Target | Physics Only | Direct Only | Ensemble | Best Physics Weight |
|--------|-------------|------------|----------|-------------------|
| angle | 0.012194 | 0.002511 | 0.002511 | 0.00 |
| depth | 0.004593 | 0.004510 | 0.004450 | 0.40 |
| left_right | 0.018819 | 0.004209 | 0.004209 | 0.00 |
| MEAN | 0.011869 | 0.003743 | 0.003723 | - |

Forward simulation still amplifies velocity errors for angle and LR.
Only depth benefits from physics (0.40 weight, -1.3% MSE improvement).
Overall ensemble: -0.5% improvement over direct.

## Diversity vs Sub 1350

| Target | Physics r | Direct r | Ensemble r |
|--------|----------|---------|-----------|
| angle | 0.8991 | 0.9334 | 0.9334 |
| depth | 0.9382 | 0.9738 | 0.9656 |
| left_right | 0.4593 | 0.9766 | 0.9766 |

Physics LR predictions are very diverse (r=0.46) but too poor (MSE=0.019) to use.

## Submissions Generated

| Sub | Description | Notes |
|-----|-------------|-------|
| 1525 | Physics standalone | Physics-only predictions |
| 1526 | Physics+Direct ensemble | Best per-target physics weights |
| 1527 | Physics + Sub784 standard | aw=0, dw=0.30, lw=0.50 |
| 1528 | Physics + Sub784 with angle | aw=0.10, dw=0.30, lw=0.50 |
| 1529 | Physics + Sub784 conservative | aw=0, dw=0.20, lw=0.30 |
| 1530 | Physics + Sub784 aggressive | aw=0, dw=0.40, lw=0.60 |
| 1531 | Ensemble + Sub784 standard | aw=0, dw=0.30, lw=0.50 |
| 1532 | Ensemble + Sub784 conservative | aw=0, dw=0.20, lw=0.30 |
| 1533 | 10% physics + 90% Sub 1350 | Conservative blend |
| 1534 | 20% physics + 80% Sub 1350 | Moderate blend |
| 1535 | 30% physics + 70% Sub 1350 | Aggressive blend |
| 1536 | 10% ensemble + 90% Sub 1350 | Conservative ensemble blend |
| 1537 | 20% ensemble + 80% Sub 1350 | Moderate ensemble blend |

## Conclusions

1. **Velocity prediction is much better** with 213 features vs 33, especially vy (R=0.56 -> R=0.89)
2. **Forward simulation still destroys signal** for angle and LR - the physics equations amplify errors
3. **Only depth benefits** from physics (optimal weight 0.40, tiny improvement)
4. **The physics inductive bias doesn't help** because the error amplification outweighs the regularization benefit
5. **Direct prediction beats physics** on every target individually

The core problem: even with R=0.89 velocity predictions, a 0.07 m/s RMSE in vy translates to ~1 inch error in depth, which is already the scale of the target variation. For angle, the 0.16 m/s vx RMSE translates to several degrees of entry angle error. Physics simulation is an extremely error-sensitive transformation.

## Recommended LB Test

Sub 1533 (10% physics + 90% Sub 1350) - tests if any physics signal helps on LB.
Expectation: unlikely to beat Sub 1350 (0.006776) given physics standalone MSE is 3x worse.

## Reproduction

```bash
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge
uv run python scripts/physics_per_example.py
```

Runtime: 42 seconds.
