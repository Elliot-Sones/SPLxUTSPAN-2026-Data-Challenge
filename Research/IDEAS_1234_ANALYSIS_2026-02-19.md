# Ideas #1-4 Analysis - 2026-02-19

## Idea #1: Player 1 Angle Calibration Fix - DEBUNKED

**Claim**: P1 angle has negative calibration slope (-1.07), is largest error source (MSE ~0.12), model predicts opposite of reality.

**Reality**: Every aspect of the claim is FALSE.

| Metric | Claimed | Actual |
|--------|---------|--------|
| P1 angle MSE | ~0.12 (largest) | 0.006006 (LOWEST of all players) |
| Calibration slope | -1.07 | +0.1805 (positive) |
| Correlation | Negative | +0.3364 (positive) |
| Error contribution | Largest source | 10.6% (smallest of all 5 players) |

### Actual Error Sources (Per-Player LOO)

**Angle**: P4 (MSE 0.019919, 33.6%) >> P5 (0.015872, 29.6%) >> P3 >> P2 >> P1
**Depth**: P5 (MSE 0.016142, 30.2%) >> P4 (0.011770, 20.0%) >> P1 >> P2 >> P3
**LR**: P2 (MSE 0.009986, 23.7%) > P5 (0.008742, 23.3%) > P3 > P4 > P1

All calibration slopes are POSITIVE (0.16-0.82). No player has negative calibration.
"Weak" slopes indicate under-dispersion (regression to mean), not reversal.

## Idea #2: High-Frequency "Flick" Features - TESTED, WEAK LB SIGNAL

Features: velocity, acceleration, jerk in frames 140-160 for hand/wrist/elbow joints.
8 features per joint x 8 joints = 64 features total.

### LOO Results (NOTE: includes PLS leakage)
- Angle: 0.011514 -> 0.008154 (-29.2%)
- Depth: 0.011446 -> 0.009027 (-21.1%)
- LR: 0.008065 -> 0.005148 (-36.2%)
- P5 angle improved most (-43.6%), P4 angle also strong (-26.7%)

### Diversity vs Sub 3411
- Angle: r = 0.8394
- **Depth: r = 0.5161** (best diversity of any model!)
- LR: r = 0.6155

### LB Results
| Sub | Blend | LB |
|-----|-------|-----|
| 3438 | 1% flick + 99% Sub3411 | 0.006234 (ties best) |
| 3433 | 3% flick + 97% Sub3411 | 0.006243 |
| 3434 | 5% flick + 95% Sub3411 | 0.006263 |
| 3435 | 8% flick + 92% Sub3411 | 0.006312 |
| 3447 | 3% flick depth-only + Sub3411 | 0.006237 |
| 3445 | 1%flick+2%traj+1%pulse+96%Sub3336 | 0.006234 (ties best) |
| 3443 | 2%flick+2%traj+1%pulse+95%Sub3336 | 0.006237 |

**Conclusion**: Flick features have genuine diversity (depth r=0.52) but standalone quality
is insufficient to beat current best at any weight. At 1% they tie.

## Idea #4: Fourier Rhythm Signatures - TESTED, WEAK MODEL

Features: FFT of velocity profiles for 14 joints x 3 signals x 14 features = 588 features.
Window: frames 100-180 (shooting motion).

### LOO Results
- Angle: 0.036523
- Depth: 0.035233
- LR: 0.029147
- Mean: 0.033634 (3.3x WORSE than baseline - too many features)

### Diversity vs Sub 3411
- Angle: r = 0.871
- Depth: r = 0.678
- LR: r = 0.655

**Conclusion**: Model is too weak standalone (588 features overwhelm PLS/Ridge with 66-74 samples).
Decent diversity but probably won't help even at 1% given the poor signal quality.
Untested on LB (hit daily submission limit).

## Idea #3: Risk-Gated Fallback - NOT TESTED

Deprioritized: requires meta-classifier which adds complexity but unlikely to beat
current ensemble given the small sample size and overfitting risk.

## Key Takeaway

The biggest remaining error sources are P4 angle (33.6% of angle error) and P5 depth (30.2%).
These players have the weakest calibration slopes (0.25 and 0.79 respectively).
To improve further, we need better predictions for P4/P5 specifically, not more diversity sources.

## Submissions for Tomorrow (Feb 20)
1. Sub 3452: 1% fourier + 99% Sub3411 (untested)
2. Sub 3457: 1%fourier+1%flick+2%traj+1%pulse+95%Sub3336 (5-way, untested)
3. New approaches targeting P4 angle and P5 depth specifically
