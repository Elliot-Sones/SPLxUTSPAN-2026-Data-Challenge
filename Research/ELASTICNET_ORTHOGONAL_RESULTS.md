# ElasticNet Orthogonal Model Results

Date: 2026-02-15
Script: scripts/elasticnet_orthogonal.py
Anchor: Sub 2475 (LB 0.006485)

## Concept

Test Gemini's ElasticNet approach (7,912 statistical features + KC features) as an
orthogonal model for blending with our locally-weighted Ridge pipeline. Different model
class + different features = different errors = potential blend improvement.

## Feature Extraction

- 7,912 total features per sample
- 69 keypoints x 3 coords x (5 angle frames + velocities + window stats + 4 phase stats)
- ~40 kinetic chain features (sequencing, amplification, energy, release mechanics, jerk)
- 5 player one-hot features
- Fixed KC indexing bug from Gemini's original implementation

## Hyperparameter Sweep (5-fold CV)

Best configs (mean MSE):
1. alpha=0.10, l1=0.1: 0.007090 (90% L2 / 10% L1 - nearly Ridge)
2. alpha=0.01, l1=0.9: 0.007262
3. alpha=0.05, l1=0.5: 0.007585

L1 regularization (sparsity) hurts at higher alpha - model wants to use many features
with small weights rather than few features with large weights.

## LOO Evaluation (best config: alpha=0.1, l1=0.1)

| Target | LOO MSE | vs Our Pipeline (0.002217) |
|--------|---------|---------------------------|
| angle | 0.006987 | 4.36x worse |
| depth | 0.005321 | 1.83x worse |
| LR | 0.006110 | 2.22x worse |
| Mean | 0.006139 | 2.77x worse |

Per-player breakdown (worst offenders):
- P5 angle: 0.016050 (extreme)
- P5 depth: 0.009274
- P4 angle: 0.008233

## Diversity Analysis

### Before clipping (raw predictions)
| Target | Correlation with Sub 2475 | Prediction Range |
|--------|--------------------------|-----------------|
| angle | r=0.36 | [-4.53, 0.79] |
| depth | r=0.87 | [0.21, 1.03] |
| LR | r=0.38 | [-0.44, 1.35] |

### After clipping to training range
| Target | Correlation with Sub 2475 |
|--------|--------------------------|
| angle | r=0.90 |
| depth | r=0.90 |
| LR | r=0.67 |

KEY FINDING: The "diversity" was mostly from extreme outlier predictions that had to be
clipped. After clipping, angle and depth become highly correlated (r=0.90) with our best.
Only LR retains moderate diversity (r=0.65-0.67).

## Per-Target Optimal Alpha (LOO)

| Target | Best Model | Alpha | LOO MSE |
|--------|-----------|-------|---------|
| angle | ElasticNet | 0.10 | 0.006987 |
| depth | ElasticNet | 0.10 | 0.005321 |
| LR | ElasticNet | 0.05 | 0.006071 |

Ridge (pure L2) did not beat ElasticNet for any target.

## Submissions Generated

| Sub | Description |
|-----|-------------|
| 2484 | 5% clipped ElasticNet + 95% Sub 2475 |
| 2485 | 10% clipped ElasticNet + 90% Sub 2475 |
| 2486 | 20% clipped ElasticNet + 80% Sub 2475 |
| 2487 | 5% per-target optimal + 95% Sub 2475 |
| 2488 | 10% per-target optimal + 90% Sub 2475 |
| 2489 | 20% per-target optimal + 80% Sub 2475 |
| 2490 | 5% angle+LR only (per-target opt) + Sub 2475 |
| 2491 | 10% angle+LR only (per-target opt) + Sub 2475 |

## Assessment

WEAK SIGNAL expected. Reasons:
1. ElasticNet standalone LOO (0.006139) is 2.77x worse than our pipeline (0.002217)
2. After clipping, diversity is only meaningful for LR (r=0.65)
3. 7,912 features with ~70 samples per player is extremely underdetermined
4. The model extrapolates wildly (predictions going to -4.5 for angle)
5. At safe blend weights (5%), the actual perturbation is tiny

Best candidates for LB testing:
1. Sub 2490 (5% angle+LR only) - targets diverse dimensions, avoids adding noise to depth
2. Sub 2484 (5% clipped all targets) - safest blend
3. Sub 2487 (5% per-target optimal) - marginally different alpha for LR

This approach does NOT provide the strong orthogonal signal we hoped for. The ElasticNet
with 7,912 features simply overfits too much with only 70 samples per player. The useful
signal is already captured better by our locally-weighted Ridge.
