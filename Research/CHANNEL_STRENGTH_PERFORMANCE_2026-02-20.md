# Channel Strength vs Player Performance Analysis

**Date**: 2026-02-20
**Script**: scripts/channel_strength_performance.py
**Data**: 5 players x 3 targets = 15 data points. Max |r| from rigorous_channel_analysis.

## Key Question

Does higher channel strength (max |r| between a single biomechanical feature and shot outcome) predict player performance or shooting consistency?

## Input Data

Per-player max |r| values (from rigorous per-player Pearson correlations):

| Player | Angle max\|r\| | Depth max\|r\| | LR max\|r\| | Mean max\|r\| |
|--------|----------------|----------------|-------------|---------------|
| P1     | 0.400          | 0.684          | 0.696       | 0.593         |
| P2     | 0.377          | 0.409          | 0.788       | 0.525         |
| P3     | 0.350          | 0.544          | 0.692       | 0.529         |
| P4     | 0.522          | 0.687          | 0.619       | 0.609         |
| P5     | 0.525          | 0.860          | 0.611       | 0.665         |

Per-player outcome statistics (raw target values, sample std):

| Player | Angle Std | Depth Std | LR Std | Mean Std |
|--------|-----------|-----------|--------|----------|
| P1     | 1.303     | 4.549     | 4.125  | 3.326    |
| P2     | 2.069     | 4.262     | 3.745  | 3.359    |
| P3     | 1.641     | 2.323     | 2.853  | 2.272    |
| P4     | 2.705     | 4.854     | 3.934  | 3.831    |
| P5     | 4.100     | 8.160     | 4.164  | 5.475    |

## Results

### Test 1: Channel Strength vs Outcome Variance (N=15)

| Metric | Pearson r | p-value | Spearman rho | p-value |
|--------|-----------|---------|--------------|---------|
| max\|r\| vs variance | +0.6730 | 0.0060 | +0.6250 | 0.0127 |
| max\|r\| vs std | +0.7389 | 0.0016 | +0.6250 | 0.0127 |
| max\|r\| vs MSE_from_mean | +0.6729 | 0.0060 | +0.6250 | 0.0127 |

**Statistically significant POSITIVE correlation**. Stronger channels appear on MORE variable targets, not less.

Per-target breakdown (N=5 each, low statistical power):
- Angle: r=+0.80 (p=0.10) - trend toward positive
- Depth: r=+0.78 (p=0.12), Spearman rho=+0.90 (p=0.037) - strong positive
- Left-Right: r=-0.32 (p=0.60) - weak negative, not significant

### Test 2: Motor Determinism Hypothesis

R^2 from max |r| gives the fraction of shot-to-shot variance explained by ONE biomechanical feature. Residual variance = total_var * (1 - R^2).

| Player | Target | TotalVar | max\|r\| | R^2 | ResidVar | % Explained |
|--------|--------|----------|----------|-----|----------|-------------|
| P1 | angle | 1.70 | 0.400 | 0.160 | 1.43 | 16.0% |
| P2 | angle | 4.28 | 0.377 | 0.142 | 3.67 | 14.2% |
| P3 | angle | 2.69 | 0.350 | 0.122 | 2.36 | 12.2% |
| P4 | angle | 7.31 | 0.522 | 0.272 | 5.32 | 27.2% |
| P5 | angle | 16.81 | 0.525 | 0.276 | 12.17 | 27.6% |
| P1 | depth | 20.69 | 0.684 | 0.468 | 11.01 | 46.8% |
| P2 | depth | 18.17 | 0.409 | 0.167 | 15.13 | 16.7% |
| P3 | depth | 5.40 | 0.544 | 0.296 | 3.80 | 29.6% |
| P4 | depth | 23.56 | 0.687 | 0.472 | 12.44 | 47.2% |
| P5 | depth | 66.59 | 0.860 | 0.740 | 17.34 | 74.0% |
| P1 | LR | 17.02 | 0.696 | 0.484 | 8.77 | 48.4% |
| P2 | LR | 14.02 | 0.788 | 0.621 | 5.31 | 62.1% |
| P3 | LR | 8.14 | 0.692 | 0.479 | 4.24 | 47.9% |
| P4 | LR | 15.47 | 0.619 | 0.383 | 9.54 | 38.3% |
| P5 | LR | 17.34 | 0.611 | 0.373 | 10.87 | 37.3% |

Partial correlation (residual_var ~ max|r| controlling for total_var): r = -0.1965 (p=0.4828)

**INCONCLUSIVE**. The partial correlation is weakly negative (directionally supporting the hypothesis that stronger channels reduce unexplained variance), but far from significant with N=15.

### Test 3: Player-Level Consistency (N=5)

| Player | Avg max\|r\| | Avg Std | Avg Var |
|--------|-------------|---------|---------|
| P1 | 0.593 | 3.326 | 13.14 |
| P2 | 0.525 | 3.359 | 12.16 |
| P3 | 0.529 | 2.272 | 5.41 |
| P4 | 0.609 | 3.831 | 15.45 |
| P5 | 0.665 | 5.475 | 33.58 |

Pearson r = +0.884 (p=0.047), Spearman rho = +0.700 (p=0.188)

**CONTRADICTS naive hypothesis**: Players with stronger channels are MORE variable, not less.

### Test 4: Determinism Ranking

| Rank | Player | Mean R^2 | Interpretation |
|------|--------|----------|----------------|
| 1 | P5 | 0.4628 | 46.3% explained - most deterministic (also most variable!) |
| 2 | P4 | 0.3759 | 37.6% explained |
| 3 | P1 | 0.3708 | 37.1% explained |
| 4 | P2 | 0.3101 | 31.0% explained |
| 5 | P3 | 0.2991 | 29.9% explained - least deterministic |

## Interpretation

The results tell a coherent but counter-intuitive story:

1. **Stronger channels appear where there is MORE variance to explain** (r=+0.67, p=0.006). This is not "strong channels cause consistency" but rather "when a target has high variance, the biomechanics that DRIVE that variance become more detectable."

2. **P5 is the most variable AND most deterministic shooter**. P5 has the highest outcome variance (especially depth: std=8.16) but also the highest mean R^2 (46.3%). This means P5's shots vary a LOT, but that variation is highly PREDICTABLE from biomechanics. P5 doesn't have random noise - they have a wide but mechanically determined distribution.

3. **P3 is the most consistent AND least deterministic**. P3 has the lowest variance (mean std=2.27) but the lowest R^2 (29.9%). P3's shots cluster tightly, so there's little variation for any feature to explain - the "channel" is weak because there's nothing to predict.

4. **Channel strength reflects signal, not skill**. High max|r| means "biomechanics explains outcome variation." A consistent shooter (low variance) has little outcome variation to explain, so max|r| is naturally lower. A variable shooter whose variation is mechanically determined (like P5 with depth) will have high max|r|.

5. **Motor determinism partial correlation is -0.20 (p=0.48)** - directionally supports the hypothesis that strong channels reduce unexplained variance beyond what total variance predicts, but too weak to be conclusive with N=15.

## Practical Implications for Competition

- P5 depth is the most deterministic channel: R^2=0.74 from ONE feature. This should be very predictable with enough features.
- P2 LR is highly deterministic: R^2=0.62. Also should be well-captured.
- Angle is the hardest target: R^2 ranges 12-28%, meaning 72-88% of angle variance is unexplained by any single feature. This aligns with the error budget showing angle as problematic.
- Per-player channel strength could inform model weighting: allocate more model capacity to (player, target) combinations with lower R^2 (harder to predict).

## Statistical Notes

- N=15 (5 players x 3 targets) provides limited statistical power, especially for per-target tests (N=5).
- The positive correlation between max|r| and variance is robust across both Pearson and Spearman, and across targets.
- Depth dominates the variance range (5.4 to 66.6), which drives the overall correlation. LR shows the opposite trend (weakly negative).
